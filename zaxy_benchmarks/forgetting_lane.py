"""Internal forgetting lane: salience attenuation safety under the cognitive profile.

The lane seeds real embedded in-temp-dir fabrics through the production write
path, synthesizes reinforcement histories by appending real
``memory.reinforcement`` events built with the ``zaxy.salience`` builders at
fixed timestamps, and measures four flip-safety properties for the Zaxy 2.2
salience/attenuation mechanism (plain versus cognitive retrieval profile):

- **Cold-start parity**: with zero reinforcement events, cognitive-profile
  checkout must return the same facts and evidence, in the same order, as the
  plain profile. Measured at the checkout-ranking layer (one shared assembly)
  and end to end across two identically seeded fabrics.
- **No recall loss**: every below-floor attenuated memory must stay reachable
  via explicit ``memory_query`` and ``memory_replay``, and must be labeled
  ``attenuated`` in checkout diagnostics.
- **Ranking lift**: for deterministically constructed pairs of equally
  relevant memories, the confirmed-reinforced member should outrank its
  never-reinforced peer under the cognitive profile but not under plain.
- **Exemption correctness**: pinned and authority-bearing memories that fall
  below the salience floor must still surface in cognitive checkout.

Determinism rules: hash embeddings, embedded projection backend, fixed seed
content, fixed reinforcement timestamps, and a fixed ``now`` for every
salience replay (the ledger and ``build_memory_checkout`` both take explicit
``now``; the lane never lets the wall clock into a measured number). Lane
results carry no event hashes or wall-clock timestamps, so two runs produce
identical reports.

Every result is labeled ``"validation": "internal"`` per the
external-validation policy: synthetic corpora, mechanism-level evidence only —
no claim about organic-usage memory quality.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zaxy.retrieval_profile import RetrievalProfile, resolve_retrieval_profile
from zaxy.salience import (
    build_confirmed_reinforcement_event,
    build_invalidated_reinforcement_event,
)

if TYPE_CHECKING:
    from zaxy.core import MemoryCheckout, MemoryFabric

FORGETTING_LANE_VERSION = "forgetting-v1"
VALIDATION_LABEL = "internal"

_LANE_SESSION_ID = "forgetting-lane"
_LANE_CHECKOUT_LIMIT = 10

#: Fixed instants anchoring every salience replay. ``_LANE_NOW`` is the lane's
#: "current time"; ``_AGED_AT`` is 59 days earlier, so a single invalidation
#: (multiplier 0.2) decays to ``0.2 * 0.5 ** (59 / 30)`` ~= 0.0512 — below the
#: default 0.15 floor — while a recent single invalidation (0.2) stays above.
_LANE_NOW = datetime(2026, 3, 1, tzinfo=UTC)
_AGED_AT = datetime(2026, 1, 1, tzinfo=UTC)

#: Production defaults measured by the lane (see ``zaxy.config.Settings``).
_SALIENCE_FLOOR = 0.15
_SALIENCE_HALF_LIFE_DAYS = 30.0

#: Topics for the zero-reinforcement parity corpus (12 topics x 4 events = 48
#: memories across decision/task/document/issue entity types).
_PARITY_TOPICS: tuple[str, ...] = (
    "payments",
    "billing",
    "ingest",
    "telemetry",
    "search",
    "checkout",
    "catalog",
    "identity",
    "ledger",
    "routing",
    "archive",
    "notification",
)

_PARITY_TOOLS: tuple[str, ...] = (
    "blue-green deploys",
    "canary rollouts",
    "feature flags",
    "shadow traffic",
)

_PARITY_QUERIES: tuple[str, ...] = tuple(
    f"how do we deploy the {topic} service and what rollback steps apply"
    for topic in ("payments", "telemetry", "catalog", "ledger", "archive", "notification")
)

#: Topics for the ranking-lift pairs (reinforced vs never-reinforced peers).
_PAIR_TOPICS: tuple[str, ...] = ("quasar", "krypton", "meridian", "obsidian", "saffron", "tundra")

#: Topics for below-floor attenuation: invalidated-then-aged and
#: double-invalidated memories, each with an above-floor "current" sibling.
_AGED_TOPICS: tuple[str, ...] = ("garnet", "cobalt")
_DOUBLE_INVALIDATED_TOPICS: tuple[str, ...] = ("umber", "verdant")

#: Topics for pinned / authority-accepted below-floor exemption memories.
_PINNED_TOPICS: tuple[str, ...] = ("basalt", "cinder")
_AUTHORITY_TOPICS: tuple[str, ...] = ("dolomite", "ember")

#: Topic for the above-floor single-invalidation memory (score 0.2 >= floor).
_RECENT_INVALIDATED_TOPIC = "fennel"

_HEX64_RE = re.compile(r"\b[0-9a-f]{64}\b")
_HEX12_FRAGMENT_RE = re.compile(r"#[0-9a-f]{12}\b")
_ISO_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T[0-9:.]+(?:\+\d{2}:\d{2}|Z)?")


# ----------------------------------------------------------------------
# Deterministic corpora and fabric seeding
# ----------------------------------------------------------------------


def _parity_corpus() -> list[tuple[str, str, dict[str, Any]]]:
    """Build the zero-reinforcement parity corpus (48 mixed-type memories)."""
    events: list[tuple[str, str, dict[str, Any]]] = []
    for index, topic in enumerate(_PARITY_TOPICS):
        tool = _PARITY_TOOLS[index % len(_PARITY_TOOLS)]
        events.append(
            (
                "decision.made",
                "agent",
                {"decision": f"Use {tool} for the {topic} service deployment"},
            )
        )
        events.append(
            (
                "task.completed",
                "agent",
                {
                    "taskId": f"task-{topic}",
                    "summary": f"Verify the {topic} service rollout and rollback drill",
                },
            )
        )
        events.append(
            (
                "document.indexed",
                "agent",
                {
                    "path": f"docs/runbooks/{topic}.md",
                    "content": (
                        f"Runbook for the {topic} service: deploy preflight, verify "
                        "canary thresholds, and rollback triggers."
                    ),
                },
            )
        )
        events.append(
            (
                "issue.diagnosed",
                "agent",
                {"summary": f"Alert noise in the {topic} service traced to a stale config"},
            )
        )
    return events


@contextmanager
def _retrieval_profile_env(profile_name: str) -> Iterator[None]:
    """Pin ``RETRIEVAL_PROFILE`` for fabric construction and use, then restore.

    Mirrors the monkeypatch pattern in ``tests/test_cognitive_profile.py``:
    the fabric resolves its retrieval profile from settings at construction,
    so the lane pins the profile explicitly instead of inheriting ambient
    environment state.
    """
    from zaxy.config import get_settings

    previous = os.environ.get("RETRIEVAL_PROFILE")
    os.environ["RETRIEVAL_PROFILE"] = profile_name
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RETRIEVAL_PROFILE", None)
        else:
            os.environ["RETRIEVAL_PROFILE"] = previous
        get_settings.cache_clear()


async def _build_lane_fabric(workdir: Path) -> MemoryFabric:
    """Build a real embedded MemoryFabric in ``workdir`` with hash embeddings."""
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
    return fabric


async def _seed_events(
    fabric: MemoryFabric, events: Sequence[tuple[str, str, dict[str, Any]]]
) -> list[Any]:
    """Append seed events through the production write path, in order."""
    appended: list[Any] = []
    for event_type, actor, payload in events:
        appended.append(
            await fabric.append(
                event_type,
                actor=actor,
                payload=dict(payload),
                thread=_LANE_SESSION_ID,
                session_id=_LANE_SESSION_ID,
            )
        )
    return appended


def _append_reinforcement(
    fabric: MemoryFabric, spec: dict[str, Any], *, at: datetime
) -> None:
    """Append one builder-validated reinforcement event at a fixed timestamp.

    Reinforcement events are ranking state only — they project no retrievable
    content — so the lane appends them directly to the session's hash-chained
    Eventloom log, which is the only append surface accepting an explicit
    timestamp. The payload comes verbatim from the ``zaxy.salience`` builders,
    and the salience ledger replays these sealed events exactly as it replays
    fabric-appended ones.
    """
    log = fabric.session_manager.get(_LANE_SESSION_ID).eventlog
    log.append(
        spec["event_type"],
        actor=spec["actor"],
        payload=spec["payload"],
        thread=_LANE_SESSION_ID,
        timestamp=at,
    )


def _confirm(fabric: MemoryFabric, event: Any, *, feedback_id: str, at: datetime) -> None:
    spec = build_confirmed_reinforcement_event(
        actor="forgetting-lane",
        session_id=_LANE_SESSION_ID,
        feedback_id=feedback_id,
        targets=[{"seq": event.seq, "hash": event.hash}],
    )
    _append_reinforcement(fabric, spec, at=at)


def _invalidate(fabric: MemoryFabric, event: Any, *, invalidation_id: str, at: datetime) -> None:
    spec = build_invalidated_reinforcement_event(
        actor="forgetting-lane",
        session_id=_LANE_SESSION_ID,
        invalidation_id=invalidation_id,
        targets=[{"seq": event.seq, "hash": event.hash}],
    )
    _append_reinforcement(fabric, spec, at=at)


# ----------------------------------------------------------------------
# Checkout helpers
# ----------------------------------------------------------------------


def _cognitive_profile() -> RetrievalProfile:
    """Resolve the production ``cognitive`` retrieval profile."""

    class _ProfileSettings:
        retrieval_profile = "cognitive"

    return resolve_retrieval_profile(_ProfileSettings())


async def _checkout(
    fabric: MemoryFabric,
    query: str,
    *,
    profile: RetrievalProfile | None,
) -> MemoryCheckout:
    """Build a checkout at the fixed lane ``now``; cognitive when ``profile``.

    Uses ``assemble_context`` + ``build_memory_checkout`` (the production
    checkout composition) rather than ``checkout_memory`` so the salience
    replay is anchored at the deterministic lane instant instead of the wall
    clock, and so no surfacing reinforcement is recorded as a side effect.
    """
    from zaxy.core import build_memory_checkout

    assembly = await fabric.assemble_context(
        query, session_id=_LANE_SESSION_ID, limit=_LANE_CHECKOUT_LIMIT
    )
    if profile is None:
        return build_memory_checkout(query=query, assembly=assembly, now=_LANE_NOW)
    return build_memory_checkout(
        query=query,
        assembly=assembly,
        now=_LANE_NOW,
        retrieval_profile=profile,
        salience_floor=_SALIENCE_FLOOR,
        salience_half_life_days=_SALIENCE_HALF_LIFE_DAYS,
    )


def _fact_contents(checkout: MemoryCheckout) -> list[str]:
    return [str(fact.get("content", "")) for fact in checkout.current_facts]


def _has_fact(checkout: MemoryCheckout, marker: str) -> bool:
    return any(marker in content for content in _fact_contents(checkout))


def _first_fact_position(checkout: MemoryCheckout, marker: str) -> int | None:
    for index, content in enumerate(_fact_contents(checkout)):
        if marker in content:
            return index
    return None


def _normalize_volatile_identifiers(text: str) -> str:
    """Strip run-varying identifiers (event hashes, wall-clock timestamps).

    Sealed event hashes embed append wall-clock timestamps, so byte-level
    comparison across two independently seeded fabrics would fail spuriously.
    Sequence numbers, ordering, and all human-readable content survive
    normalization untouched.
    """
    text = _HEX64_RE.sub("<hash>", text)
    text = _HEX12_FRAGMENT_RE.sub("#<hash>", text)
    return _ISO_TIMESTAMP_RE.sub("<timestamp>", text)


def _normalized_results(checkout: MemoryCheckout) -> str:
    """Serialize facts + evidence with volatile identifiers normalized."""
    payload = {"current_facts": checkout.current_facts, "evidence": checkout.evidence}
    return _normalize_volatile_identifiers(json.dumps(payload, sort_keys=True))


# ----------------------------------------------------------------------
# Check 1: cold-start parity (zero reinforcement events)
# ----------------------------------------------------------------------


async def _run_cold_start_parity(workdir: Path) -> dict[str, Any]:
    """Measure plain-vs-cognitive checkout parity on a reinforcement-free corpus.

    Two measurements, both over the same 48-memory corpus and query set, with
    no cues and no multi-hop graph structure:

    - ``checkout_layer``: one plain fabric, one assembly per query, plain and
      cognitive ``build_memory_checkout`` over that same assembly — facts and
      evidence compared byte-for-byte (this isolates the salience/cue ranking
      blend, the layer the 2.2 forgetting change owns).
    - ``full_path``: an independent fabric constructed under
      ``RETRIEVAL_PROFILE=cognitive`` (graph-walk router stage armed), seeded
      identically — facts and evidence compared after normalizing event
      hashes and timestamps, which differ across fabrics by construction.
    """
    corpus = _parity_corpus()
    cognitive = _cognitive_profile()

    with _retrieval_profile_env("local_fast"):
        plain_fabric = await _build_lane_fabric(workdir / "plain")
        try:
            await _seed_events(plain_fabric, corpus)
            checkout_layer_identical: list[bool] = []
            plain_normalized: list[str] = []
            for query in _PARITY_QUERIES:
                plain_checkout = await _checkout(plain_fabric, query, profile=None)
                cognitive_checkout = await _checkout(plain_fabric, query, profile=cognitive)
                checkout_layer_identical.append(
                    plain_checkout.current_facts == cognitive_checkout.current_facts
                    and plain_checkout.evidence == cognitive_checkout.evidence
                )
                plain_normalized.append(_normalized_results(plain_checkout))
        finally:
            await plain_fabric.close()

    with _retrieval_profile_env("cognitive"):
        cognitive_fabric = await _build_lane_fabric(workdir / "cognitive")
        try:
            await _seed_events(cognitive_fabric, corpus)
            full_path_identical: list[bool] = []
            for query, plain_results in zip(_PARITY_QUERIES, plain_normalized, strict=True):
                end_to_end = await _checkout(
                    cognitive_fabric, query, profile=cognitive_fabric.retrieval_profile
                )
                full_path_identical.append(_normalized_results(end_to_end) == plain_results)
        finally:
            await cognitive_fabric.close()

    checkout_layer_fraction = sum(checkout_layer_identical) / len(checkout_layer_identical)
    full_path_fraction = sum(full_path_identical) / len(full_path_identical)
    return {
        "measurement": (
            "Zero-reinforcement corpus, no cues, no multi-hop edges: cognitive checkout "
            "facts/evidence compared against plain at the checkout-ranking layer "
            "(shared assembly, byte-identical) and end to end across two identically "
            "seeded fabrics (event hashes and timestamps normalized)."
        ),
        "query_count": len(_PARITY_QUERIES),
        "corpus_memory_count": len(corpus),
        "checkout_layer": {
            "identical_fraction": round(checkout_layer_fraction, 4),
            "per_query_identical": checkout_layer_identical,
        },
        "full_path": {
            "identical_fraction": round(full_path_fraction, 4),
            "per_query_identical": full_path_identical,
        },
        "status": (
            "pass" if checkout_layer_fraction == 1.0 and full_path_fraction == 1.0 else "fail"
        ),
    }


# ----------------------------------------------------------------------
# Checks 2-4: reinforcement scenarios on one cognitive fabric
# ----------------------------------------------------------------------


def _scenario_corpus() -> list[tuple[str, str, dict[str, Any]]]:
    """Build the reinforcement-scenario corpus (deterministic order)."""
    events: list[tuple[str, str, dict[str, Any]]] = []
    for topic in _PAIR_TOPICS:
        events.append(
            ("decision.made", "agent", {"decision": f"{topic} storage decision alpha route"})
        )
        events.append(
            ("decision.made", "agent", {"decision": f"{topic} storage decision bravo route"})
        )
    for topic in (*_AGED_TOPICS, *_DOUBLE_INVALIDATED_TOPICS):
        events.append(
            (
                "decision.made",
                "agent",
                {"decision": f"Legacy {topic} cache invalidation strategy for storage"},
            )
        )
        events.append(
            (
                "decision.made",
                "agent",
                {"decision": f"Current {topic} cache invalidation strategy for storage"},
            )
        )
    for topic in _PINNED_TOPICS:
        events.append(
            (
                "decision.made",
                "agent",
                {
                    "decision": f"Pinned {topic} rollback runbook for storage incidents",
                    "pinned": True,
                },
            )
        )
    for topic in _AUTHORITY_TOPICS:
        events.append(
            (
                "decision.made",
                "agent",
                {
                    "decision": f"Accepted {topic} incident escalation policy",
                    "review_status": "accepted",
                },
            )
        )
    events.append(
        (
            "decision.made",
            "agent",
            {
                "decision": (
                    f"Deprecated {_RECENT_INVALIDATED_TOPIC} sync mechanism choice for storage"
                )
            },
        )
    )
    events.append(
        (
            "decision.made",
            "agent",
            {"decision": "Heavily confirmed winterhold deployment freeze decision"},
        )
    )
    return events


def _apply_reinforcement_histories(
    fabric: MemoryFabric, appended: Sequence[Any]
) -> dict[str, list[Any]]:
    """Synthesize the controlled reinforcement histories over seeded events.

    Returns the seeded events grouped by scenario role. Reinforcement
    timestamps are fixed (`_LANE_NOW` / `_AGED_AT`), so every replayed
    salience score is an exact, reproducible number at the lane ``now``:

    - pair "bravo" members: confirmed once at now -> 1.5
    - heavily confirmed: confirmed three times at now -> 3.375
    - recent single invalidation: 0.2 (above the 0.15 floor)
    - invalidated then aged 59 days: 0.2 * 0.5 ** (59 / 30) ~= 0.0512
    - double-invalidated (incl. pinned/authority): 0.2 * 0.2 = 0.04
    """
    by_text = {str(event.payload.get("decision", "")): event for event in appended}

    def event_for(marker: str) -> Any:
        for text, event in by_text.items():
            if marker in text:
                return event
        raise LookupError(f"seed event matching {marker!r} not found")

    groups: dict[str, list[Any]] = {
        "pair_baseline": [],
        "pair_reinforced": [],
        "aged_below_floor": [],
        "double_below_floor": [],
        "pinned_below_floor": [],
        "authority_below_floor": [],
        "recent_invalidated": [],
        "heavily_confirmed": [],
    }
    for topic in _PAIR_TOPICS:
        groups["pair_baseline"].append(event_for(f"{topic} storage decision alpha"))
        reinforced = event_for(f"{topic} storage decision bravo")
        groups["pair_reinforced"].append(reinforced)
        _confirm(fabric, reinforced, feedback_id=f"feedback:{topic}", at=_LANE_NOW)

    for topic in _AGED_TOPICS:
        aged = event_for(f"Legacy {topic} cache invalidation")
        groups["aged_below_floor"].append(aged)
        _invalidate(fabric, aged, invalidation_id=f"invalidate:{topic}:aged", at=_AGED_AT)
    for topic in _DOUBLE_INVALIDATED_TOPICS:
        doubled = event_for(f"Legacy {topic} cache invalidation")
        groups["double_below_floor"].append(doubled)
        for index in range(2):
            _invalidate(
                fabric, doubled, invalidation_id=f"invalidate:{topic}:{index}", at=_LANE_NOW
            )

    for group, topics, marker in (
        ("pinned_below_floor", _PINNED_TOPICS, "Pinned {topic} rollback runbook"),
        ("authority_below_floor", _AUTHORITY_TOPICS, "Accepted {topic} incident escalation"),
    ):
        for topic in topics:
            exempt = event_for(marker.format(topic=topic))
            groups[group].append(exempt)
            for index in range(2):
                _invalidate(
                    fabric,
                    exempt,
                    invalidation_id=f"invalidate:{topic}:{index}",
                    at=_LANE_NOW,
                )

    recent = event_for(f"Deprecated {_RECENT_INVALIDATED_TOPIC} sync mechanism")
    groups["recent_invalidated"].append(recent)
    _invalidate(
        fabric, recent, invalidation_id=f"invalidate:{_RECENT_INVALIDATED_TOPIC}:0", at=_LANE_NOW
    )

    heavy = event_for("Heavily confirmed winterhold deployment freeze")
    groups["heavily_confirmed"].append(heavy)
    for index in range(3):
        _confirm(fabric, heavy, feedback_id=f"feedback:winterhold:{index}", at=_LANE_NOW)
    return groups


def _attenuation_entries(checkout: MemoryCheckout) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attenuation = checkout.diagnostics.get("attenuation")
    if not isinstance(attenuation, dict):
        return [], []
    excluded = [entry for entry in attenuation.get("excluded", []) if isinstance(entry, dict)]
    exempt = [entry for entry in attenuation.get("exempt", []) if isinstance(entry, dict)]
    return excluded, exempt


async def _run_no_recall_loss(
    fabric: MemoryFabric, groups: dict[str, list[Any]]
) -> dict[str, Any]:
    """Check 2: below-floor memories stay reachable and honestly labeled."""
    profile = fabric.retrieval_profile
    below_floor = [
        ("aged", topic, event)
        for topic, event in zip(_AGED_TOPICS, groups["aged_below_floor"], strict=True)
    ] + [
        ("double_invalidated", topic, event)
        for topic, event in zip(
            _DOUBLE_INVALIDATED_TOPICS, groups["double_below_floor"], strict=True
        )
    ]

    per_memory: list[dict[str, Any]] = []
    replay = await fabric.replay(session_id=_LANE_SESSION_ID)
    replay_seqs = {event.seq for event in replay.events}
    salience_scores: dict[str, float] = {}
    for history, topic, event in below_floor:
        marker = f"Legacy {topic} cache invalidation"
        query = f"{topic} cache invalidation strategy"
        checkout = await _checkout(fabric, query, profile=profile)
        excluded, _ = _attenuation_entries(checkout)
        excluded_entry = next((entry for entry in excluded if entry.get("seq") == event.seq), None)
        attenuated_out_of_ranking = not _has_fact(checkout, marker)
        labeled = excluded_entry is not None and excluded_entry.get("label") == "attenuated"
        if excluded_entry is not None:
            salience_scores[f"{history}:{topic}"] = float(excluded_entry["salience_score"])
        contexts = await fabric.query(query, session_id=_LANE_SESSION_ID, limit=10)
        explicit_query_retrieved = any(marker in context.content for context in contexts)
        per_memory.append(
            {
                "history": history,
                "topic": topic,
                "seq": event.seq,
                "attenuated_out_of_default_ranking": attenuated_out_of_ranking,
                "labeled_attenuated_in_diagnostics": labeled,
                "explicit_query_retrieved": explicit_query_retrieved,
                "replay_reachable": event.seq in replay_seqs,
            }
        )

    total = len(per_memory)
    retrieved_fraction = sum(1 for row in per_memory if row["explicit_query_retrieved"]) / total
    labeled_fraction = (
        sum(1 for row in per_memory if row["labeled_attenuated_in_diagnostics"]) / total
    )
    attenuated_fraction = (
        sum(1 for row in per_memory if row["attenuated_out_of_default_ranking"]) / total
    )
    replay_fraction = sum(1 for row in per_memory if row["replay_reachable"]) / total
    return {
        "measurement": (
            "Every below-floor memory must be out of default cognitive checkout ranking, "
            "labeled 'attenuated' in checkout diagnostics, retrievable via explicit "
            "memory_query, and reachable via memory_replay."
        ),
        "below_floor_memory_count": total,
        "explicit_query_retrieved_fraction": round(retrieved_fraction, 4),
        "labeled_attenuated_fraction": round(labeled_fraction, 4),
        "attenuated_out_of_default_ranking_fraction": round(attenuated_fraction, 4),
        "replay_reachable_fraction": round(replay_fraction, 4),
        "below_floor_salience_scores": salience_scores,
        "per_memory": per_memory,
        "status": (
            "pass"
            if retrieved_fraction == 1.0 and labeled_fraction == 1.0 and replay_fraction == 1.0
            else "fail"
        ),
    }


async def _run_ranking_lift(fabric: MemoryFabric) -> dict[str, Any]:
    """Check 3: confirmed reinforcement lifts one of two equally relevant peers.

    Each pair appends "alpha route" then "bravo route" with identical token
    structure and confirms only bravo. The pair members tie on relevance, so
    plain ranking keeps its tie behavior and the cognitive salience multiplier
    is the only differentiator. Position is the first occurrence of each
    member among checkout facts.
    """
    profile = fabric.retrieval_profile
    per_pair: list[dict[str, Any]] = []
    for topic in _PAIR_TOPICS:
        query = f"{topic} storage decision"
        baseline_marker = f"{topic} storage decision alpha route"
        reinforced_marker = f"{topic} storage decision bravo route"
        plain = await _checkout(fabric, query, profile=None)
        cognitive = await _checkout(fabric, query, profile=profile)
        row: dict[str, Any] = {"topic": topic}
        for label, checkout in (("plain", plain), ("cognitive", cognitive)):
            baseline_at = _first_fact_position(checkout, baseline_marker)
            reinforced_at = _first_fact_position(checkout, reinforced_marker)
            row[f"{label}_reinforced_first"] = (
                baseline_at is not None
                and reinforced_at is not None
                and reinforced_at < baseline_at
            )
            row[f"{label}_both_present"] = baseline_at is not None and reinforced_at is not None
        per_pair.append(row)

    total = len(per_pair)
    cognitive_fraction = sum(1 for row in per_pair if row["cognitive_reinforced_first"]) / total
    plain_fraction = sum(1 for row in per_pair if row["plain_reinforced_first"]) / total
    return {
        "measurement": (
            "Deterministic equally-relevant pairs (identical token structure, append-order "
            "tie): fraction of pairs where the confirmed-reinforced member ranks first, "
            "cognitive vs plain checkout over the same assemblies."
        ),
        "pair_count": total,
        "cognitive_reinforced_first_fraction": round(cognitive_fraction, 4),
        "plain_reinforced_first_fraction": round(plain_fraction, 4),
        "per_pair": per_pair,
        # Exit bar: reinforcement must change cognitive ranking on every pair
        # while plain ranking stays at its tie behavior (no lift).
        "status": (
            "pass" if cognitive_fraction == 1.0 and cognitive_fraction > plain_fraction else "fail"
        ),
    }


async def _run_exemption_correctness(
    fabric: MemoryFabric, groups: dict[str, list[Any]]
) -> dict[str, Any]:
    """Check 4: pinned/authority below-floor memories still surface."""
    profile = fabric.retrieval_profile
    cases = [
        ("pinned", topic, event, f"Pinned {topic} rollback runbook", f"{topic} rollback runbook")
        for topic, event in zip(_PINNED_TOPICS, groups["pinned_below_floor"], strict=True)
    ] + [
        (
            "authority",
            topic,
            event,
            f"Accepted {topic} incident escalation",
            f"{topic} incident escalation policy",
        )
        for topic, event in zip(_AUTHORITY_TOPICS, groups["authority_below_floor"], strict=True)
    ]

    per_memory: list[dict[str, Any]] = []
    for expected_reason, topic, event, marker, query in cases:
        checkout = await _checkout(fabric, query, profile=profile)
        _, exempt = _attenuation_entries(checkout)
        entry = next((item for item in exempt if item.get("seq") == event.seq), None)
        per_memory.append(
            {
                "topic": topic,
                "seq": event.seq,
                "expected_reason": expected_reason,
                "surfaced_in_cognitive_checkout": _has_fact(checkout, marker),
                "listed_exempt": entry is not None,
                "exempt_reason": entry.get("exempt_reason") if entry is not None else None,
                "salience_score": (
                    float(entry["salience_score"]) if entry is not None else None
                ),
            }
        )

    total = len(per_memory)
    surfaced_fraction = (
        sum(1 for row in per_memory if row["surfaced_in_cognitive_checkout"]) / total
    )
    reason_fraction = (
        sum(1 for row in per_memory if row["exempt_reason"] == row["expected_reason"]) / total
    )
    return {
        "measurement": (
            "Pinned and authority-accepted memories pushed below the salience floor must "
            "still surface under cognitive checkout and be listed exempt with the right "
            "reason."
        ),
        "exempt_memory_count": total,
        "surfaced_fraction": round(surfaced_fraction, 4),
        "exempt_reason_correct_fraction": round(reason_fraction, 4),
        "per_memory": per_memory,
        "status": "pass" if surfaced_fraction == 1.0 and reason_fraction == 1.0 else "fail",
    }


async def _run_above_floor_invalidation(
    fabric: MemoryFabric, groups: dict[str, list[Any]]
) -> dict[str, Any]:
    """Informational: a single recent invalidation (0.2) stays above the floor."""
    profile = fabric.retrieval_profile
    event = groups["recent_invalidated"][0]
    marker = f"Deprecated {_RECENT_INVALIDATED_TOPIC} sync mechanism"
    checkout = await _checkout(
        fabric, f"{_RECENT_INVALIDATED_TOPIC} sync mechanism choice", profile=profile
    )
    excluded, _ = _attenuation_entries(checkout)
    return {
        "measurement": (
            "A single recent invalidation (salience 0.2 >= floor 0.15) down-weights but "
            "must not attenuate; informational, not an exit criterion."
        ),
        "seq": event.seq,
        "still_in_default_ranking": _has_fact(checkout, marker),
        "listed_excluded": any(entry.get("seq") == event.seq for entry in excluded),
    }


# ----------------------------------------------------------------------
# Lane runner
# ----------------------------------------------------------------------


def run_forgetting_lane(workdir: Path) -> dict[str, Any]:
    """Run the forgetting lane in ``workdir`` and return one labeled report."""
    return asyncio.run(_run_forgetting_lane_async(workdir))


async def _run_forgetting_lane_async(workdir: Path) -> dict[str, Any]:
    cold_start = await _run_cold_start_parity(workdir / "parity")

    with _retrieval_profile_env("cognitive"):
        fabric = await _build_lane_fabric(workdir / "scenarios")
        try:
            appended = await _seed_events(fabric, _scenario_corpus())
            groups = _apply_reinforcement_histories(fabric, appended)
            no_recall_loss = await _run_no_recall_loss(fabric, groups)
            ranking_lift = await _run_ranking_lift(fabric)
            exemptions = await _run_exemption_correctness(fabric, groups)
            above_floor = await _run_above_floor_invalidation(fabric, groups)
        finally:
            await fabric.close()

    checks = {
        "cold_start_parity": cold_start,
        "no_recall_loss": no_recall_loss,
        "ranking_lift": ranking_lift,
        "exemption_correctness": exemptions,
    }
    return {
        "lane": "forgetting",
        "version": FORGETTING_LANE_VERSION,
        "validation": VALIDATION_LABEL,
        "measurement": (
            "Salience attenuation flip-safety over real seeded embedded fabrics with "
            "synthesized reinforcement histories at fixed timestamps; deterministic, "
            "no LLM scoring. Synthetic corpora, mechanism-level evidence only — not a "
            "claim about organic-usage memory quality. Cue and graph-walk effects are "
            "excluded by querying without cues over corpora without multi-hop structure."
        ),
        "fixture": {
            "session_id": _LANE_SESSION_ID,
            "checkout_limit": _LANE_CHECKOUT_LIMIT,
            "salience_floor": _SALIENCE_FLOOR,
            "salience_half_life_days": _SALIENCE_HALF_LIFE_DAYS,
            "lane_now": _LANE_NOW.isoformat().replace("+00:00", "Z"),
            "aged_reinforcement_at": _AGED_AT.isoformat().replace("+00:00", "Z"),
            "parity_memory_count": len(_parity_corpus()),
            "scenario_memory_count": len(_scenario_corpus()),
        },
        "checks": checks,
        "above_floor_invalidation": above_floor,
        "contract": {
            name: check["status"] for name, check in checks.items()
        }
        | {
            "status": (
                "pass"
                if all(check["status"] == "pass" for check in checks.values())
                else "fail"
            )
        },
    }
