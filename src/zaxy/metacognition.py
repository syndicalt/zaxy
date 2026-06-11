"""Non-authoritative metacognition event contracts and feeling-of-knowing.

These helpers build Eventloom append specs for uncertainty, conflict,
confidence, and re-verification state, and provide the deterministic
feeling-of-knowing pre-check core (:func:`build_feeling_of_knowing_index`,
:func:`feeling_of_knowing`). Generated metacognition is observable diagnostic
state only; it never promotes claims to authority, and a feeling-of-knowing
verdict is a cheap prediction about checkout, never a memory answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

_AUTHORITY_STATUS = "non_authoritative"
_OPEN_STATUS = "open"
_UNRESOLVED_STATUS = "unresolved"

_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENTLOOM_CITATION_RE = re.compile(
    r"^eventloom://[^/\s]+/events/[1-9][0-9]*#(?:[0-9a-f]{12}|[0-9a-f]{64})$"
)
_METACOGNITION_ID_RE = re.compile(r"^metacognition:[a-z_]+:[0-9a-f]{24}$")

METACOGNITION_PRIORITIES = frozenset({"low", "normal", "high", "urgent"})


def build_known_unknown_event(
    *,
    actor: str,
    session_id: str,
    question: str,
    reason: str,
    source_events: Sequence[Mapping[str, Any]],
    claim_key: str | None = None,
    gap_type: str | None = None,
    reverify_query: str | None = None,
    unknown_id: str | None = None,
) -> dict[str, Any]:
    """Build an open, cited known-unknown event append spec."""
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    question = _validate_text(question, field_name="question")
    reason = _validate_text(reason, field_name="reason")
    cited_source_events = _snapshot_source_events(source_events)
    claim_key = _validate_optional_text(claim_key, field_name="claim_key")
    gap_type = _validate_optional_text(gap_type, field_name="gap_type")
    reverify_query = _validate_optional_text(reverify_query, field_name="reverify_query")
    unknown_id = _validate_or_build_id(
        explicit_id=unknown_id,
        id_type="unknown",
        identity={
            "claim_key": claim_key,
            "gap_type": gap_type,
            "question": question,
            "source_events": cited_source_events,
        },
    )

    payload: dict[str, Any] = {
        "unknown_id": unknown_id,
        "question": question,
        "reason": reason,
        "source_events": cited_source_events,
        "status": _OPEN_STATUS,
        "authority_status": _AUTHORITY_STATUS,
    }
    _add_optional(payload, "claim_key", claim_key)
    _add_optional(payload, "gap_type", gap_type)
    _add_optional(payload, "reverify_query", reverify_query)

    return {
        "event_type": "metacognition.unknown.recorded",
        "actor": actor,
        "thread": session_id,
        "payload": payload,
    }


def build_confidence_assessment_event(
    *,
    actor: str,
    session_id: str,
    claim: str,
    confidence: float,
    support_count: int,
    conflict_count: int,
    evidence: Sequence[Mapping[str, Any]],
    method: str,
    requires_reverify: bool = False,
    claim_key: str | None = None,
    assessment_id: str | None = None,
) -> dict[str, Any]:
    """Build an append-only confidence assessment point."""
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    claim = _validate_text(claim, field_name="claim")
    confidence = _validate_confidence(confidence)
    support_count = _validate_non_negative_int(support_count, field_name="support_count")
    conflict_count = _validate_non_negative_int(conflict_count, field_name="conflict_count")
    evidence = _snapshot_evidence(evidence)
    method = _validate_text(method, field_name="method")
    if not isinstance(requires_reverify, bool):
        raise ValueError("requires_reverify must be a boolean")
    claim_key = _validate_optional_text(claim_key, field_name="claim_key")
    assessment_id = _validate_or_build_id(
        explicit_id=assessment_id,
        id_type="confidence",
        identity={
            "claim": claim,
            "claim_key": claim_key,
            "confidence": confidence,
            "conflict_count": conflict_count,
            "evidence": evidence,
            "method": method,
            "requires_reverify": requires_reverify,
            "support_count": support_count,
        },
    )

    payload: dict[str, Any] = {
        "assessment_id": assessment_id,
        "claim": claim,
        "confidence": confidence,
        "support_count": support_count,
        "conflict_count": conflict_count,
        "evidence": evidence,
        "method": method,
        "requires_reverify": requires_reverify,
        "authority_status": _AUTHORITY_STATUS,
    }
    _add_optional(payload, "claim_key", claim_key)

    return {
        "event_type": "metacognition.confidence.assessed",
        "actor": actor,
        "thread": session_id,
        "payload": payload,
    }


def build_conflict_cluster_event(
    *,
    actor: str,
    session_id: str,
    claim_key: str,
    claim: str,
    supporting_source_events: Sequence[Mapping[str, Any]],
    conflicting_source_events: Sequence[Mapping[str, Any]],
    confidence: float,
    reason: str,
    cluster_id: str | None = None,
) -> dict[str, Any]:
    """Build an unresolved conflict cluster event append spec."""
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    claim_key = _validate_text(claim_key, field_name="claim_key")
    claim = _validate_text(claim, field_name="claim")
    supporting_source_events = _snapshot_source_events(
        supporting_source_events,
        field_name="supporting_source_events",
    )
    conflicting_source_events = _snapshot_source_events(
        conflicting_source_events,
        field_name="conflicting_source_events",
    )
    confidence = _validate_confidence(confidence)
    reason = _validate_text(reason, field_name="reason")
    cluster_id = _validate_or_build_id(
        explicit_id=cluster_id,
        id_type="conflict_cluster",
        identity={
            "claim": claim,
            "claim_key": claim_key,
            "conflicting_source_events": conflicting_source_events,
            "supporting_source_events": supporting_source_events,
        },
    )

    return {
        "event_type": "metacognition.conflict.clustered",
        "actor": actor,
        "thread": session_id,
        "payload": {
            "cluster_id": cluster_id,
            "claim_key": claim_key,
            "claim": claim,
            "supporting_source_events": supporting_source_events,
            "conflicting_source_events": conflicting_source_events,
            "confidence": confidence,
            "reason": reason,
            "resolution_status": _UNRESOLVED_STATUS,
            "authority_status": _AUTHORITY_STATUS,
        },
    }


def build_reverify_request_event(
    *,
    actor: str,
    session_id: str,
    query: str,
    reason: str,
    source_events: Sequence[Mapping[str, Any]],
    priority: str = "normal",
    claim_key: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build an open re-verification request event append spec."""
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    query = _validate_text(query, field_name="query")
    reason = _validate_text(reason, field_name="reason")
    source_events = _snapshot_source_events(source_events)
    priority = _validate_priority(priority)
    claim_key = _validate_optional_text(claim_key, field_name="claim_key")
    request_id = _validate_or_build_id(
        explicit_id=request_id,
        id_type="reverify",
        identity={
            "claim_key": claim_key,
            "query": query,
            "source_events": source_events,
        },
    )

    payload: dict[str, Any] = {
        "reverify_id": request_id,
        "query": query,
        "reason": reason,
        "source_events": source_events,
        "priority": priority,
        "status": _OPEN_STATUS,
        "authority_status": _AUTHORITY_STATUS,
    }
    _add_optional(payload, "claim_key", claim_key)

    return {
        "event_type": "metacognition.reverify.requested",
        "actor": actor,
        "thread": session_id,
        "payload": payload,
    }


def summarize_metacognition_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize replayed metacognition events without changing authority."""
    summary: dict[str, Any] = {
        "unknown_count": 0,
        "open_unknown_count": 0,
        "confidence_assessment_count": 0,
        "conflict_cluster_count": 0,
        "unresolved_conflict_cluster_count": 0,
        "reverify_request_count": 0,
        "reverify_needed_count": 0,
        "open_unknowns": [],
        "reverify_requests": [],
        "conflict_clusters": [],
    }

    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("event_type") or event.get("type")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue

        if event_type == "metacognition.unknown.recorded":
            summary["unknown_count"] += 1
            if payload.get("status") == _OPEN_STATUS:
                summary["open_unknown_count"] += 1
                summary["open_unknowns"].append(dict(payload))
        elif event_type == "metacognition.confidence.assessed":
            summary["confidence_assessment_count"] += 1
        elif event_type == "metacognition.conflict.clustered":
            summary["conflict_cluster_count"] += 1
            if payload.get("resolution_status") == _UNRESOLVED_STATUS:
                summary["unresolved_conflict_cluster_count"] += 1
                summary["conflict_clusters"].append(dict(payload))
        elif event_type == "metacognition.reverify.requested":
            summary["reverify_request_count"] += 1
            if payload.get("status") == _OPEN_STATUS:
                summary["reverify_needed_count"] += 1
                summary["reverify_requests"].append(dict(payload))

    return summary


# --- Feeling-of-knowing pre-check -------------------------------------------
#
# The feeling-of-knowing core answers "would checkout likely return something
# for this query?" from in-memory projection state only: an entity-name token
# bloom filter, cue/term hit counts, and a salience summary. It is
# deterministic, O(query terms), and performs no embedding call, no graph
# query, and no I/O. Verdicts are calibration targets, never authority.

FOK_LIKELY: Final = "likely"
FOK_POSSIBLE: Final = "possible"
FOK_UNLIKELY: Final = "unlikely"

FoKVerdictLabel = Literal["likely", "possible", "unlikely"]

# Bloom sizing math. For ``n`` distinct name tokens at design false-positive
# rate ``p``: bits ``m = ceil(-n * ln(p) / ln(2)^2) ~= 9.59 * n`` (rounded up
# to a byte boundary) and hash count ``k = round(log2(1 / p)) = 7``, the
# optimal k for that m/n ratio. At the design point (n = 10,000 tokens,
# p = 0.01) this yields m ~= 95,851 bits (~11.7 KiB) and an expected
# false-positive rate of ``(1 - e^(-k * n / m))^k ~= 0.0100``. The filter is
# sized from the actual distinct-token count at build time so smaller
# projections stay smaller, with a 512-bit floor so tiny corpora are not
# saturated by the k probes and stay at or below the design rate.
FOK_BLOOM_FALSE_POSITIVE_RATE = 0.01
_FOK_BLOOM_MIN_BITS = 512

# Raw-score blend weights (they sum to 1.0, keeping the score in [0, 1]).
# Bloom membership of query terms in projected entity names is the strongest
# cheap predictor that checkout will surface something, so it dominates; cue
# hits corroborate (the term was an encoding-time cue, not just a name token);
# salience mass rewards queries that touch currently-reinforced memories.
FOK_BLOOM_WEIGHT = 0.6
FOK_CUE_WEIGHT = 0.25
FOK_SALIENCE_WEIGHT = 0.15

# Verdict thresholds. "likely" requires either every query term to be a known
# name token (bloom ratio 1.0 alone scores 0.6) or a near-complete bloom match
# corroborated by cue/salience signal. "possible" starts where roughly one
# third of the query terms are known names (1/3 * 0.6 = 0.2) — partial
# overlap that checkout may or may not convert. Below that, the projection
# has essentially no lexical evidence for the query: "unlikely".
FOK_LIKELY_THRESHOLD = 0.55
FOK_POSSIBLE_THRESHOLD = 0.2

# Verdict comparisons tolerate binary floating-point representation error at
# the threshold boundaries: a 3-term query with exactly one bloom hit scores
# 0.6 * (1/3) = 0.19999999999999998, which is mathematically exactly the
# possible threshold 0.2 but compares below it without the tolerance,
# misclassifying a designed-boundary "possible" as "unlikely". 1e-9 is far
# above accumulated double rounding error for this three-term blend and far
# below the smallest meaningful score step (one term in a 10,000-term query
# changes the score by at least 1.5e-5).
_FOK_THRESHOLD_EPSILON = 1e-9

# Histogram bucket upper bounds for replayed salience scores (see
# ``zaxy.salience``: scores are clamped to [0.01, 10.0] around base 1.0).
# Buckets: strongly decayed [0, 0.5), mildly decayed [0.5, 1.0), baseline to
# lightly reinforced [1.0, 2.0), reinforced [2.0, 5.0), and the clamp tail
# [5.0, inf).
FOK_SALIENCE_BUCKET_UPPER_BOUNDS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)

_FOK_BLOOM_PERSON = b"zaxy-fok-v1"
_FOK_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
# Mirrors the keyword-search stop-word list used by projection backends so
# the pre-check and lexical retrieval discount the same function words.
_FOK_STOP_WORDS = frozenset(
    {
        "am",
        "and",
        "are",
        "at",
        "did",
        "do",
        "does",
        "first",
        "for",
        "had",
        "have",
        "how",
        "in",
        "it",
        "me",
        "of",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)
_LN_2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class FeelingOfKnowingIndex:
    """Deterministic in-memory feeling-of-knowing index for one session.

    Built once from projection state by :func:`build_feeling_of_knowing_index`
    and queried by :func:`feeling_of_knowing`. All fields are plain data so
    the index is comparable, picklable, and cheap to hold per session.
    """

    entity_count: int
    token_count: int
    bloom_bits: bytes
    bloom_bit_count: int
    bloom_hash_count: int
    cue_counts: Mapping[str, int]
    token_salience_mass: Mapping[str, float]
    total_salience_mass: float
    salience_histogram: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FoKSignals:
    """Signal breakdown behind one feeling-of-knowing verdict."""

    query_term_count: int
    bloom_hits: int
    bloom_hit_ratio: float
    cue_hits: int
    cue_hit_total: int
    cue_hit_ratio: float
    matched_salience_mass: float
    total_salience_mass: float
    salience_mass_ratio: float


@dataclass(frozen=True, slots=True)
class FoKVerdict:
    """A non-authoritative feeling-of-knowing verdict with its evidence."""

    verdict: FoKVerdictLabel
    score: float
    signals: FoKSignals

    def to_dict(self) -> dict[str, Any]:
        """Return the verdict as plain diagnostics-ready data."""
        return {
            "verdict": self.verdict,
            "score": self.score,
            "signals": asdict(self.signals),
            "authority_status": _AUTHORITY_STATUS,
        }


def build_feeling_of_knowing_index(
    entity_names: Iterable[str],
    *,
    cue_counts: Mapping[str, int] | None = None,
    salience_by_name: Mapping[str, float] | None = None,
) -> FeelingOfKnowingIndex:
    """Build the feeling-of-knowing index from plain projection state.

    The builder is deliberately decoupled from any backend: callers pass the
    active entity names the projection already holds in memory, optional
    cue-term hit counts (encoding-specificity cues observed at append time),
    and optional replayed salience scores keyed by entity name
    (``SalienceState.score`` values). Building is deterministic: identical
    inputs produce an identical index.
    """
    names: list[str] = []
    for position, name in enumerate(entity_names):
        if not isinstance(name, str):
            raise ValueError(f"entity_names[{position}] must be a string")
        names.append(name)
    validated_cues = _validate_cue_counts(cue_counts if cue_counts is not None else {})
    validated_salience = _validate_salience_scores(
        salience_by_name if salience_by_name is not None else {}
    )

    tokens = sorted({token for name in names for token in _fok_terms(name)})
    bit_count, hash_count = _fok_bloom_parameters(len(tokens))
    bits = bytearray(bit_count // 8)
    for token in tokens:
        for bit in _fok_bloom_positions(token, bit_count=bit_count, hash_count=hash_count):
            bits[bit >> 3] |= 1 << (bit & 7)

    token_salience_mass: dict[str, float] = {}
    histogram = [0] * (len(FOK_SALIENCE_BUCKET_UPPER_BOUNDS) + 1)
    total_salience_mass = 0.0
    for name in sorted(validated_salience):
        score = validated_salience[name]
        histogram[_fok_salience_bucket(score)] += 1
        total_salience_mass += score
        for token in _fok_terms(name):
            token_salience_mass[token] = token_salience_mass.get(token, 0.0) + score

    return FeelingOfKnowingIndex(
        entity_count=len(names),
        token_count=len(tokens),
        bloom_bits=bytes(bits),
        bloom_bit_count=bit_count,
        bloom_hash_count=hash_count,
        cue_counts=validated_cues,
        token_salience_mass=token_salience_mass,
        total_salience_mass=total_salience_mass,
        salience_histogram=tuple(histogram),
    )


def feeling_of_knowing(index: FeelingOfKnowingIndex, query: str) -> FoKVerdict:
    """Predict whether checkout would likely return something for ``query``.

    Deterministic and O(query terms): each unique query term costs one bloom
    membership probe plus two dictionary lookups. A query whose terms are all
    stop words (or an empty index) yields zero signal and an "unlikely"
    verdict rather than an error — that is an honest prediction, not a caller
    bug. The salience-mass ratio is capped at 1.0 because token-level masses
    double-count multi-token entity names; it is a presence-weighted signal,
    not a probability measure.
    """
    if not isinstance(index, FeelingOfKnowingIndex):
        raise ValueError("index must be a FeelingOfKnowingIndex")
    query = _validate_text(query, field_name="query")

    terms = _fok_terms(query)
    term_count = len(terms)
    bloom_hits = sum(1 for term in terms if _fok_bloom_contains(index, term))
    cue_hits = sum(1 for term in terms if index.cue_counts.get(term, 0) > 0)
    cue_hit_total = sum(index.cue_counts.get(term, 0) for term in terms)
    matched_salience_mass = sum(index.token_salience_mass.get(term, 0.0) for term in terms)

    bloom_hit_ratio = bloom_hits / term_count if term_count else 0.0
    cue_hit_ratio = cue_hits / term_count if term_count else 0.0
    salience_mass_ratio = (
        min(1.0, matched_salience_mass / index.total_salience_mass)
        if index.total_salience_mass > 0.0
        else 0.0
    )

    score = (
        FOK_BLOOM_WEIGHT * bloom_hit_ratio
        + FOK_CUE_WEIGHT * cue_hit_ratio
        + FOK_SALIENCE_WEIGHT * salience_mass_ratio
    )
    verdict: FoKVerdictLabel
    if score >= FOK_LIKELY_THRESHOLD - _FOK_THRESHOLD_EPSILON:
        verdict = FOK_LIKELY
    elif score >= FOK_POSSIBLE_THRESHOLD - _FOK_THRESHOLD_EPSILON:
        verdict = FOK_POSSIBLE
    else:
        verdict = FOK_UNLIKELY

    return FoKVerdict(
        verdict=verdict,
        score=score,
        signals=FoKSignals(
            query_term_count=term_count,
            bloom_hits=bloom_hits,
            bloom_hit_ratio=bloom_hit_ratio,
            cue_hits=cue_hits,
            cue_hit_total=cue_hit_total,
            cue_hit_ratio=cue_hit_ratio,
            matched_salience_mass=matched_salience_mass,
            total_salience_mass=index.total_salience_mass,
            salience_mass_ratio=salience_mass_ratio,
        ),
    )


def _fok_terms(text: str) -> list[str]:
    """Tokenize to unique, ordered, casefolded terms; drop stop/1-char words."""
    terms = [term for term in _FOK_TOKEN_RE.findall(text.casefold()) if len(term) > 1]
    return list(dict.fromkeys(term for term in terms if term not in _FOK_STOP_WORDS))


def _fok_bloom_parameters(token_count: int) -> tuple[int, int]:
    """Size the bloom filter for ``token_count`` distinct tokens.

    Returns ``(bit_count, hash_count)`` with ``bit_count`` rounded up to a
    byte boundary and floored at ``_FOK_BLOOM_MIN_BITS``. See the sizing math
    next to :data:`FOK_BLOOM_FALSE_POSITIVE_RATE`. An empty token set keeps a
    single all-zero byte so membership probes are well-defined and always
    miss.
    """
    if token_count == 0:
        return 8, 1
    bits = math.ceil(
        -token_count * math.log(FOK_BLOOM_FALSE_POSITIVE_RATE) / (_LN_2 * _LN_2)
    )
    bit_count = max(((bits + 7) // 8) * 8, _FOK_BLOOM_MIN_BITS)
    hash_count = max(1, round(-math.log(FOK_BLOOM_FALSE_POSITIVE_RATE) / _LN_2))
    return bit_count, hash_count


def _fok_bloom_positions(term: str, *, bit_count: int, hash_count: int) -> list[int]:
    """Derive ``hash_count`` deterministic bit positions for one term.

    Uses salted blake2b double hashing (Kirsch-Mitzenmacher): one 128-bit
    digest split into two 64-bit halves ``h1``/``h2`` drives the probe
    sequence ``(h1 + i * step) % m`` with ``step = 1 + (h2 % (m - 1))`` so
    the step is never zero modulo ``m`` and the k probes stay distinct in
    expectation, matching k independent hash functions.
    """
    digest = hashlib.blake2b(
        term.encode("utf-8"), digest_size=16, person=_FOK_BLOOM_PERSON
    ).digest()
    h1 = int.from_bytes(digest[:8], "big")
    h2 = int.from_bytes(digest[8:], "big")
    step = 1 + (h2 % (bit_count - 1)) if bit_count > 1 else 0
    return [(h1 + probe * step) % bit_count for probe in range(hash_count)]


def _fok_bloom_contains(index: FeelingOfKnowingIndex, term: str) -> bool:
    if index.token_count == 0:
        return False
    return all(
        index.bloom_bits[bit >> 3] >> (bit & 7) & 1
        for bit in _fok_bloom_positions(
            term,
            bit_count=index.bloom_bit_count,
            hash_count=index.bloom_hash_count,
        )
    )


def _fok_salience_bucket(score: float) -> int:
    for position, upper_bound in enumerate(FOK_SALIENCE_BUCKET_UPPER_BOUNDS):
        if score < upper_bound:
            return position
    return len(FOK_SALIENCE_BUCKET_UPPER_BOUNDS)


def _validate_cue_counts(cue_counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(cue_counts, Mapping):
        raise ValueError("cue_counts must be a mapping of cue terms to hit counts")
    validated: dict[str, int] = {}
    for key, value in cue_counts.items():
        cue = _validate_text(key, field_name="cue_counts key").casefold()
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"cue_counts[{key!r}] must be a non-negative integer")
        validated[cue] = validated.get(cue, 0) + value
    return validated


def _validate_salience_scores(salience_by_name: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(salience_by_name, Mapping):
        raise ValueError("salience_by_name must be a mapping of entity names to scores")
    validated: dict[str, float] = {}
    for key, value in salience_by_name.items():
        name = _validate_text(key, field_name="salience_by_name key")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"salience_by_name[{key!r}] must be a finite non-negative number")
        validated[name] = float(value)
    return validated


def _validate_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _validate_text(value, field_name=field_name)


def _validate_confidence(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("confidence must be a number between 0.0 and 1.0")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return confidence


def _validate_non_negative_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_priority(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("priority must be one of: " + ", ".join(sorted(METACOGNITION_PRIORITIES)))
    priority = value.strip().casefold()
    if priority not in METACOGNITION_PRIORITIES:
        raise ValueError("priority must be one of: " + ", ".join(sorted(METACOGNITION_PRIORITIES)))
    return priority


def _snapshot_source_events(
    source_events: Sequence[Mapping[str, Any]],
    *,
    field_name: str = "source_events",
) -> list[dict[str, Any]]:
    if not isinstance(source_events, Sequence) or isinstance(source_events, str | bytes):
        raise ValueError(f"{field_name} must be a non-empty sequence of citations")
    if not source_events:
        raise ValueError(f"{field_name} must be non-empty")

    return [
        _snapshot_source_event(source_event, index=index, field_name=field_name)
        for index, source_event in enumerate(source_events)
    ]


def _snapshot_source_event(
    source_event: Mapping[str, Any],
    *,
    index: int,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(source_event, Mapping):
        raise ValueError(f"{field_name}[{index}] must be a citation mapping")

    seq = source_event.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
        raise ValueError(f"{field_name}[{index}].seq must be a positive integer")

    event_hash = source_event.get("hash")
    if not isinstance(event_hash, str) or _EVENT_HASH_RE.fullmatch(event_hash) is None:
        raise ValueError(f"{field_name}[{index}].hash must be exactly 64 lowercase hex characters")

    return {"seq": seq, "hash": event_hash}


def _snapshot_evidence(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(evidence, Sequence) or isinstance(evidence, str | bytes):
        raise ValueError("evidence must be a sequence of Eventloom citations")

    return [_snapshot_evidence_item(item, index=index) for index, item in enumerate(evidence)]


def _snapshot_evidence_item(item: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"evidence[{index}] must be a citation mapping")

    citation = item.get("citation")
    if not isinstance(citation, str) or _EVENTLOOM_CITATION_RE.fullmatch(citation) is None:
        raise ValueError(
            f"evidence[{index}].citation must be an Eventloom citation with a 12 or 64 character hash"
        )

    snapshot = dict(item)
    for key, value in snapshot.items():
        if isinstance(value, str):
            snapshot[key] = _validate_text(value, field_name=f"evidence[{index}].{key}")
    return snapshot


def _validate_or_build_id(
    *,
    explicit_id: str | None,
    id_type: str,
    identity: Mapping[str, Any],
) -> str:
    if explicit_id is not None:
        explicit_id = _validate_text(explicit_id, field_name=f"{id_type}_id")
        if _METACOGNITION_ID_RE.fullmatch(explicit_id) is None:
            raise ValueError(
                f"{id_type}_id must match metacognition:{{id_type}}:"
                "{24 lowercase hex characters}"
            )
        return explicit_id
    digest = hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"metacognition:{id_type}:{digest}"


def _add_optional(payload: dict[str, Any], key: str, value: object | None) -> None:
    if value is not None:
        payload[key] = value
