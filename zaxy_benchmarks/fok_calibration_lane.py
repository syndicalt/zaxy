"""Internal FoK calibration lane: feeling-of-knowing versus real retrieval.

The lane seeds real embedded in-temp-dir fabrics with deterministic
word-composition corpora at several sizes, builds the feeling-of-knowing
index exactly the way the MCP server does (from the projection store's
``active_entity_names``, no cue counts, no salience state), and scores
:func:`zaxy.metacognition.feeling_of_knowing` predictions against ground
truth produced by the REAL retrieval path: a query is labeled positive iff
``MemoryFabric.query`` returns at least one context containing one of the
query's topic terms. No label is assumed; every label comes from running
retrieval.

Query families per corpus:

- ``present``: queries about seeded entities (terse and natural phrasings);
- ``absent``: queries about topics built from a disjoint vocabulary;
- ``partial``: one present topic word mixed with absent topic words.

Metrics per corpus size: Brier score of the FoK raw score against the labels,
the base-rate predictor's Brier score (always predicting the query set's
positive rate) as the comparison floor, verdict-bucket hit/miss rates, and
false-positive/false-negative rates. The roadmap exit criterion is that the
FoK Brier score beats the base-rate predictor; the lane reports whichever way
that comes out.

Deterministic by construction: fixed word tables, hash embeddings, embedded
projection backend, no LLM calls, no wall-clock dependence in any reported
number. Every result is labeled ``"validation": "internal"``: synthetic
corpora, mechanism-level evidence only — not a claim about organic-usage
calibration.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zaxy.metacognition import (
    FOK_LIKELY,
    FOK_POSSIBLE,
    FOK_UNLIKELY,
    build_feeling_of_knowing_index,
    feeling_of_knowing,
)

if TYPE_CHECKING:
    from zaxy.core import MemoryFabric

FOK_CALIBRATION_LANE_VERSION = "fok-calibration-v1"
VALIDATION_LABEL = "internal"

_LANE_SESSION_ID = "fok-lane"
_QUERY_LIMIT = 10

#: Default corpus sizes (entity count). Small enough for CI; larger sweeps
#: (500/5000) are exposed through the ``sizes`` parameter and the CLI.
DEFAULT_FOK_CORPUS_SIZES: tuple[int, ...] = (50, 200)

#: Queries sampled per family. Present and absent families use two phrasings
#: each, so the default query set per corpus is 10*2 + 10*2 + 8 = 48 queries.
_PRESENT_ENTITY_SAMPLES = 10
_ABSENT_TOPIC_SAMPLES = 10
_PARTIAL_TOPIC_SAMPLES = 8

#: Present-corpus word tables: 24 * 22 * 12 = 6336 unique compositions.
#: Disjoint from the absent tables and from the query/decision template words.
_PRESENT_ADJECTIVES: tuple[str, ...] = (
    "amber", "brindle", "cerulean", "dapple", "ember", "fawn", "garnet", "hazel",
    "indigo", "jasper", "kestrel", "lilac", "maroon", "nutmeg", "ochre", "pewter",
    "quartz", "russet", "sable", "teal", "umber", "viridian", "wisteria", "zircon",
)
_PRESENT_NOUNS: tuple[str, ...] = (
    "anchor", "beacon", "compass", "dynamo", "estuary", "fjord", "glacier", "harbor",
    "isthmus", "jetty", "keel", "lagoon", "meridian", "nebula", "orchard", "pylon",
    "quarry", "reef", "summit", "trellis", "uplink", "vault",
)
_PRESENT_DOMAINS: tuple[str, ...] = (
    "pipeline", "gateway", "scheduler", "registry", "cache", "indexer",
    "router", "broker", "archive", "ledger", "monitor", "queue",
)

#: Absent-topic word tables, disjoint from every present table and template.
_ABSENT_ADJECTIVES: tuple[str, ...] = (
    "basalt", "cobalt", "damask", "emerald", "fuchsia", "ginger",
    "henna", "ivory", "juniper", "khaki",
)
_ABSENT_NOUNS: tuple[str, ...] = (
    "abyss", "bluff", "crater", "delta", "escarpment", "foothill",
    "gorge", "headland", "inlet", "knoll",
)
_ABSENT_DOMAINS: tuple[str, ...] = (
    "turbine", "smelter", "loom", "kiln", "foundry", "windmill",
    "sawmill", "derrick", "crucible", "bellows",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class _LaneQuery:
    """One ground-truth-labelable query with its designated topic terms."""

    family: str
    phrasing: str
    text: str
    #: Topic terms the query is *about*; ground truth checks real retrieval
    #: results for these, not for template filler words.
    topic_terms: tuple[str, ...]


def _entity_phrase(index: int) -> str:
    """Return the deterministic three-word phrase for corpus entity ``index``."""
    adjective = _PRESENT_ADJECTIVES[index % len(_PRESENT_ADJECTIVES)]
    noun = _PRESENT_NOUNS[(index // len(_PRESENT_ADJECTIVES)) % len(_PRESENT_NOUNS)]
    domain = _PRESENT_DOMAINS[
        (index // (len(_PRESENT_ADJECTIVES) * len(_PRESENT_NOUNS))) % len(_PRESENT_DOMAINS)
    ]
    return f"{adjective} {noun} {domain}"


def _absent_phrase(index: int) -> str:
    """Return a deterministic absent-topic phrase varying all three words.

    The multipliers are coprime with the table lengths so consecutive indices
    change adjective, noun, and domain together: absent queries must not share
    topic words, or a single bloom false positive would correlate across the
    whole family instead of costing the ~1% per token the filter is sized for.
    """
    adjective = _ABSENT_ADJECTIVES[index % len(_ABSENT_ADJECTIVES)]
    noun = _ABSENT_NOUNS[(index * 3 + 1) % len(_ABSENT_NOUNS)]
    domain = _ABSENT_DOMAINS[(index * 7 + 2) % len(_ABSENT_DOMAINS)]
    return f"{adjective} {noun} {domain}"


def _corpus_events(size: int) -> list[tuple[str, str, dict[str, Any]]]:
    """Build ``size`` decision memories whose entity names carry topic words.

    ``decision.made`` extraction names the projected entity by the decision
    text, so the production feeling-of-knowing index (built from active
    entity names) sees exactly these topic words plus the shared template
    words.
    """
    if size > len(_PRESENT_ADJECTIVES) * len(_PRESENT_NOUNS) * len(_PRESENT_DOMAINS):
        raise ValueError(
            "corpus size exceeds the unique word-composition space "
            f"({len(_PRESENT_ADJECTIVES) * len(_PRESENT_NOUNS) * len(_PRESENT_DOMAINS)})"
        )
    return [
        (
            "decision.made",
            "agent",
            {"decision": f"Adopt the {_entity_phrase(index)} rollout"},
        )
        for index in range(size)
    ]


def _sample_indices(population: int, samples: int) -> list[int]:
    """Spread ``samples`` deterministic indices across ``population``."""
    count = min(samples, population)
    if count <= 0:
        return []
    step = max(1, population // count)
    return [(offset * step) % population for offset in range(count)]


def _build_query_set(size: int) -> list[_LaneQuery]:
    """Build the deterministic labeled-by-retrieval query set for one corpus."""
    queries: list[_LaneQuery] = []
    for index in _sample_indices(size, _PRESENT_ENTITY_SAMPLES):
        phrase = _entity_phrase(index)
        terms = tuple(phrase.split())
        queries.append(_LaneQuery("present", "terse", phrase, terms))
        queries.append(
            _LaneQuery("present", "natural", f"what do we know about the {phrase}", terms)
        )
    for offset in range(_ABSENT_TOPIC_SAMPLES):
        phrase = _absent_phrase(offset)
        terms = tuple(phrase.split())
        queries.append(_LaneQuery("absent", "terse", phrase, terms))
        queries.append(
            _LaneQuery("absent", "natural", f"what do we know about the {phrase}", terms)
        )
    present_indices = _sample_indices(size, _PARTIAL_TOPIC_SAMPLES)
    for offset, index in enumerate(present_indices):
        present_word = _entity_phrase(index).split()[0]
        absent_words = _absent_phrase(offset + _ABSENT_TOPIC_SAMPLES).split()[1:]
        text = " ".join([present_word, *absent_words])
        queries.append(
            _LaneQuery("partial", "terse", text, (present_word, *absent_words))
        )
    return queries


# ----------------------------------------------------------------------
# Ground truth from the real retrieval path
# ----------------------------------------------------------------------


async def _seed_corpus_fabric(workdir: Path, size: int) -> MemoryFabric:
    """Build and seed a real embedded fabric with the ``size``-entity corpus."""
    from zaxy.core import MemoryFabric
    from zaxy.embedding import HashEmbeddingProvider

    eventloom_path = workdir / ".eventloom"
    fabric = MemoryFabric(
        eventloom_path=str(eventloom_path),
        projection_backend="embedded",
        embedded_graph_path=eventloom_path / "projections" / "embedded.kuzu",
        tracer_disabled=True,
    )
    fabric.embedding_provider = HashEmbeddingProvider(
        dimension=fabric.settings.embedding_dimension
    )
    await fabric.connect()
    for event_type, actor, payload in _corpus_events(size):
        await fabric.append(
            event_type,
            actor=actor,
            payload=dict(payload),
            thread=_LANE_SESSION_ID,
            session_id=_LANE_SESSION_ID,
        )
    return fabric


def _contains_topic_term(content: str, topic_terms: Sequence[str]) -> bool:
    tokens = set(_TOKEN_RE.findall(content.casefold()))
    return any(term in tokens for term in topic_terms)


async def _ground_truth_label(fabric: MemoryFabric, query: _LaneQuery) -> int:
    """Label one query by running the real retrieval path.

    Positive iff explicit ``memory_query`` returns at least one context whose
    content contains one of the query's topic terms — non-empty *relevant*
    results, since vector lanes always return nearest neighbors for any text.
    """
    contexts = await fabric.query(
        query.text, session_id=_LANE_SESSION_ID, limit=_QUERY_LIMIT
    )
    return int(
        any(_contains_topic_term(context.content, query.topic_terms) for context in contexts)
    )


# ----------------------------------------------------------------------
# Calibration metrics
# ----------------------------------------------------------------------


def _brier(predictions: Sequence[float], labels: Sequence[int]) -> float:
    return sum(
        (prediction - label) ** 2
        for prediction, label in zip(predictions, labels, strict=True)
    ) / len(labels)


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _verdict_bucket_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-verdict counts plus the bucket calibration rates."""
    stats: dict[str, Any] = {}
    for verdict in (FOK_LIKELY, FOK_POSSIBLE, FOK_UNLIKELY):
        bucket = [row for row in rows if row["verdict"] == verdict]
        positives = sum(1 for row in bucket if row["label"] == 1)
        stats[verdict] = {
            "count": len(bucket),
            "positive_count": positives,
            "positive_rate": _safe_fraction(positives, len(bucket)),
        }
    likely = stats[FOK_LIKELY]
    unlikely = stats[FOK_UNLIKELY]
    stats["likely_hit_rate"] = likely["positive_rate"]
    stats["unlikely_miss_rate"] = (
        round(1.0 - unlikely["positive_rate"], 4)
        if unlikely["positive_rate"] is not None
        else None
    )
    return stats


def _family_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for family in ("present", "absent", "partial"):
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        breakdown[family] = {
            "count": len(family_rows),
            "positive_rate": _safe_fraction(
                sum(row["label"] for row in family_rows), len(family_rows)
            ),
            "brier": round(
                _brier(
                    [row["score"] for row in family_rows],
                    [row["label"] for row in family_rows],
                ),
                4,
            ),
            "mean_score": round(
                sum(row["score"] for row in family_rows) / len(family_rows), 4
            ),
        }
    return breakdown


async def _run_corpus(workdir: Path, size: int) -> dict[str, Any]:
    """Run one corpus size end to end: seed, index, label, score, summarize."""
    fabric = await _seed_corpus_fabric(workdir, size)
    try:
        # Production index path: active entity names only, exactly like the
        # MCP server's memory_feeling_of_knowing handler (no cue counts, no
        # salience state, same feature-detection seam for the accessor).
        names_provider = getattr(fabric.graph, "active_entity_names", None)
        entity_names: list[str] = (
            list(await names_provider(session_id=_LANE_SESSION_ID))
            if names_provider is not None
            else []
        )
        index = build_feeling_of_knowing_index(entity_names)

        rows: list[dict[str, Any]] = []
        for query in _build_query_set(size):
            verdict = feeling_of_knowing(index, query.text)
            label = await _ground_truth_label(fabric, query)
            rows.append(
                {
                    "family": query.family,
                    "phrasing": query.phrasing,
                    "query": query.text,
                    "score": round(verdict.score, 6),
                    "verdict": verdict.verdict,
                    "label": label,
                }
            )
    finally:
        await fabric.close()

    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    positive_rate = sum(labels) / len(labels)
    brier_fok = _brier(scores, labels)
    brier_base_rate = _brier([positive_rate] * len(labels), labels)

    negatives = [row for row in rows if row["label"] == 0]
    positives = [row for row in rows if row["label"] == 1]
    false_positives = sum(1 for row in negatives if row["verdict"] == FOK_LIKELY)
    false_negatives = sum(1 for row in positives if row["verdict"] == FOK_UNLIKELY)

    return {
        "corpus_size": size,
        "entity_name_count": len(entity_names),
        "query_count": len(rows),
        "positive_rate": round(positive_rate, 4),
        "brier_fok": round(brier_fok, 4),
        "brier_base_rate": round(brier_base_rate, 4),
        "brier_skill_vs_base_rate": round(brier_base_rate - brier_fok, 4),
        "beats_base_rate": brier_fok < brier_base_rate,
        "false_positive_rate": _safe_fraction(false_positives, len(negatives)),
        "false_negative_rate": _safe_fraction(false_negatives, len(positives)),
        "verdict_buckets": _verdict_bucket_stats(rows),
        "families": _family_breakdown(rows),
        "queries": rows,
    }


# ----------------------------------------------------------------------
# Lane runner
# ----------------------------------------------------------------------


def run_fok_calibration_lane(
    workdir: Path,
    *,
    sizes: Sequence[int] = DEFAULT_FOK_CORPUS_SIZES,
) -> dict[str, Any]:
    """Run the FoK calibration lane and return one labeled report.

    ``sizes`` are corpus entity counts; each size seeds its own fabric in a
    ``workdir`` subdirectory.
    """
    validated: list[int] = []
    for size in sizes:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("sizes must be positive integers")
        validated.append(size)
    if not validated:
        raise ValueError("sizes must include at least one corpus size")
    return asyncio.run(_run_fok_calibration_lane_async(workdir, sizes=validated))


async def _run_fok_calibration_lane_async(
    workdir: Path, *, sizes: Sequence[int]
) -> dict[str, Any]:
    corpora = [await _run_corpus(workdir / f"size-{size}", size) for size in sizes]
    return {
        "lane": "fok_calibration",
        "version": FOK_CALIBRATION_LANE_VERSION,
        "validation": VALIDATION_LABEL,
        "measurement": (
            "Feeling-of-knowing calibration against ground truth from the real explicit "
            "memory_query path over deterministic word-composition corpora; index built "
            "from the projection store's active entity names exactly like the MCP "
            "handler (no cue counts, no salience state). Synthetic corpora and template "
            "query phrasings — mechanism-level evidence only, not organic-usage "
            "calibration."
        ),
        "fixture": {
            "session_id": _LANE_SESSION_ID,
            "query_limit": _QUERY_LIMIT,
            "corpus_sizes": list(sizes),
            "present_query_samples": _PRESENT_ENTITY_SAMPLES,
            "absent_query_samples": _ABSENT_TOPIC_SAMPLES,
            "partial_query_samples": _PARTIAL_TOPIC_SAMPLES,
        },
        "corpora": corpora,
        # Roadmap 2.2-beta.1 exit criterion: FoK Brier beats the base-rate
        # predictor — required at every measured corpus size.
        "contract": {
            "beats_base_rate_per_size": {
                str(corpus["corpus_size"]): corpus["beats_base_rate"] for corpus in corpora
            },
            "status": (
                "pass" if all(corpus["beats_base_rate"] for corpus in corpora) else "fail"
            ),
        },
    }
