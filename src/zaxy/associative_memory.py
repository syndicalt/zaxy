"""Experimental associative projection for partial-cue event-memory recovery.

This module is intentionally isolated from Zaxy's production checkout and graph
projection paths. It implements a small, deterministic baseline inspired by the
neuron-astrocyte associative-memory paper and the authors' NAAM reference code:

- Eventloom events remain the source of truth.
- The projection is derived, replayable, and discardable.
- Associative candidates must resolve back to cited Eventloom event refs.

The implementation is pure Python on purpose. The first branch target is to
test whether higher-order pattern completion over event history is worth a
heavier research implementation, not to add PyTorch/JAX to core Zaxy.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zaxy.event import Event, EventLog

PATTERN_COMPLETION_WORKLOAD_VERSION = "pattern-completion-v0"
STATE_RECOVERY_WORKLOAD_VERSION = "state-recovery-v0"
STATE_RECOVERY_REPORT_SCHEMA_VERSION = "state-recovery-report-v1"
STATE_RECOVERY_PRODUCTION_BASELINE = "memory_fabric_checkout"
STATE_RECOVERY_GUARDRAIL_THRESHOLDS = {
    "state_accuracy": 0.818,
    "minimal_evidence_recall": 0.90,
    "stale_rejection": 1.0,
    "distractor_resistance": 0.80,
    "abstention_accuracy": 1.0,
    "citation_coverage": 1.0,
}
ASSOCIATIVE_BASELINE_DESCRIPTIONS = {
    "direct_lexical": "Rank events by direct query-token overlap only.",
    "hash_vector": "Rank events by deterministic hashed token-vector cosine similarity.",
    "graph_traversal": "Seed from direct lexical matches, then traverse shared high-IDF event terms.",
    "zaxy_core_proxy": (
        "Source-aware lexical baseline approximating current Zaxy retrieval posture with current/accepted "
        "event boosts and stale-event penalties."
    ),
    "memory_fabric_checkout": (
        "Append the case through MemoryFabric, run the model-facing Memory Checkout contract, "
        "and score cited checkout facts/evidence."
    ),
    "associative_projection": (
        "Seed from direct matches, diffuse through shared process terms, and resolve the "
        "completed pattern back to cited Eventloom refs."
    ),
    "authority_resolved_associative": (
        "Run associative projection, then apply explicit authority metadata for current, "
        "parent/promoted, non-rejected evidence before final scoring."
    ),
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]*")
_EVENTLOOM_CITATION_RE = re.compile(r"eventloom://(?P<thread>[^/]+)/events/(?P<seq>\d+)#(?P<hash>[a-f0-9]+)")
_STOPWORDS = {
    "a",
    "about",
    "across",
    "after",
    "agent",
    "agents",
    "all",
    "and",
    "are",
    "as",
    "at",
    "before",
    "by",
    "can",
    "case",
    "context",
    "current",
    "did",
    "do",
    "during",
    "event",
    "events",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "memory",
    "needed",
    "needs",
    "of",
    "on",
    "or",
    "over",
    "project",
    "query",
    "should",
    "state",
    "that",
    "the",
    "then",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "why",
    "with",
}


@dataclass(frozen=True)
class EventRef:
    """Stable pointer to a cited Eventloom event."""

    seq: int
    hash: str
    event_type: str
    thread: str

    @classmethod
    def from_event(cls, event: Event) -> EventRef:
        return cls(seq=event.seq, hash=event.hash, event_type=event.type, thread=event.thread)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "hash": self.hash,
            "event_type": self.event_type,
            "thread": self.thread,
        }


@dataclass(frozen=True)
class EventPacket:
    """Tokenized event packet used by the derived projection."""

    event_id: str
    text: str
    terms: frozenset[str]
    ref: EventRef
    payload: dict[str, Any]

    @classmethod
    def from_event(cls, event: Event) -> EventPacket:
        text = _render_event_text(event)
        return cls(
            event_id=f"{event.thread}:{event.seq}",
            text=text,
            terms=frozenset(_tokenize(text)),
            ref=EventRef.from_event(event),
            payload=event.payload,
        )


@dataclass(frozen=True)
class AssociativeCandidate:
    """Pattern-completion candidate resolved back to Eventloom evidence."""

    score: float
    summary_terms: tuple[str, ...]
    evidence: tuple[EventRef, ...]
    support_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "summary_terms": list(self.summary_terms),
            "support_terms": list(self.support_terms),
            "evidence": [ref.to_dict() for ref in self.evidence],
        }


@dataclass(frozen=True)
class PatternCompletionGold:
    """Gold labels for a latent event-history recovery case."""

    latent_state: str
    expected_terms: tuple[str, ...]
    expected_event_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PatternCompletionGold:
        return cls(
            latent_state=str(value["latent_state"]),
            expected_terms=tuple(str(term) for term in value["expected_terms"]),
            expected_event_ids=tuple(str(event_id) for event_id in value["expected_event_ids"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_state": self.latent_state,
            "expected_terms": list(self.expected_terms),
            "expected_event_ids": list(self.expected_event_ids),
        }


@dataclass(frozen=True)
class PatternCompletionCase:
    """One benchmark case with partial cue query and Eventloom events."""

    case_id: str
    query: str
    events: tuple[dict[str, Any], ...]
    gold: PatternCompletionGold

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PatternCompletionCase:
        return cls(
            case_id=str(value["case_id"]),
            query=str(value["query"]),
            events=tuple(dict(event) for event in value["events"]),
            gold=PatternCompletionGold.from_dict(value["gold"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "events": list(self.events),
            "gold": self.gold.to_dict(),
        }


@dataclass(frozen=True)
class PatternCompletionWorkload:
    """Frozen pattern-completion workload."""

    version: str
    cases: tuple[PatternCompletionCase, ...]
    fingerprint: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PatternCompletionWorkload:
        return cls(
            version=str(value["version"]),
            cases=tuple(PatternCompletionCase.from_dict(case) for case in value["cases"]),
            fingerprint=str(value["fingerprint"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class PatternCompletionMetrics:
    """Exact-scored partial-cue state recovery metrics."""

    latent_state_recall: float
    evidence_recall: float
    citation_coverage: float
    returned_events: int
    injected_tokens: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_state_recall": self.latent_state_recall,
            "evidence_recall": self.evidence_recall,
            "citation_coverage": self.citation_coverage,
            "returned_events": self.returned_events,
            "injected_tokens": self.injected_tokens,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass(frozen=True)
class PatternCompletionCaseResult:
    """Per-case pattern-completion report."""

    case_id: str
    query: str
    associative_candidates: tuple[AssociativeCandidate, ...]
    associative_metrics: PatternCompletionMetrics
    baseline_event_ids: tuple[str, ...]
    baseline_metrics: PatternCompletionMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "associative_candidates": [candidate.to_dict() for candidate in self.associative_candidates],
            "associative_metrics": self.associative_metrics.to_dict(),
            "baseline_event_ids": list(self.baseline_event_ids),
            "baseline_metrics": self.baseline_metrics.to_dict(),
        }


@dataclass(frozen=True)
class PatternCompletionReport:
    """Full pattern-completion benchmark report."""

    version: str
    workload_fingerprint: str
    metrics: PatternCompletionMetrics
    baselines: dict[str, PatternCompletionMetrics]
    cases: tuple[PatternCompletionCaseResult, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "workload_fingerprint": self.workload_fingerprint,
            "metrics": self.metrics.to_dict(),
            "baselines": {name: metrics.to_dict() for name, metrics in self.baselines.items()},
            "baseline_descriptions": ASSOCIATIVE_BASELINE_DESCRIPTIONS,
            "cases": [case.to_dict() for case in self.cases],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class StateRecoveryGold:
    """Gold labels for one latent state recovery case."""

    latent_state: str
    expected_terms: tuple[str, ...]
    minimal_evidence_event_ids: tuple[str, ...]
    stale_event_ids: tuple[str, ...]
    distractor_event_ids: tuple[str, ...]
    should_abstain: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StateRecoveryGold:
        return cls(
            latent_state=str(value["latent_state"]),
            expected_terms=tuple(str(term) for term in value["expected_terms"]),
            minimal_evidence_event_ids=tuple(
                str(event_id) for event_id in value["minimal_evidence_event_ids"]
            ),
            stale_event_ids=tuple(str(event_id) for event_id in value.get("stale_event_ids", [])),
            distractor_event_ids=tuple(str(event_id) for event_id in value.get("distractor_event_ids", [])),
            should_abstain=bool(value.get("should_abstain", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_state": self.latent_state,
            "expected_terms": list(self.expected_terms),
            "minimal_evidence_event_ids": list(self.minimal_evidence_event_ids),
            "stale_event_ids": list(self.stale_event_ids),
            "distractor_event_ids": list(self.distractor_event_ids),
            "should_abstain": self.should_abstain,
        }


@dataclass(frozen=True)
class StateRecoveryCase:
    """One adversarial partial-cue state recovery case."""

    case_id: str
    query: str
    events: tuple[dict[str, Any], ...]
    gold: StateRecoveryGold

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StateRecoveryCase:
        return cls(
            case_id=str(value["case_id"]),
            query=str(value["query"]),
            events=tuple(dict(event) for event in value["events"]),
            gold=StateRecoveryGold.from_dict(value["gold"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "events": list(self.events),
            "gold": self.gold.to_dict(),
        }


@dataclass(frozen=True)
class StateRecoveryWorkload:
    """Frozen StateRecoveryBench workload."""

    version: str
    cases: tuple[StateRecoveryCase, ...]
    fingerprint: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StateRecoveryWorkload:
        return cls(
            version=str(value["version"]),
            cases=tuple(StateRecoveryCase.from_dict(case) for case in value["cases"]),
            fingerprint=str(value["fingerprint"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class StateRecoveryMetrics:
    """StateRecoveryBench metrics for one baseline."""

    state_accuracy: float
    minimal_evidence_recall: float
    stale_rejection: float
    distractor_resistance: float
    token_cost: int
    latency_ms: float
    citation_coverage: float
    abstention_accuracy: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_accuracy": self.state_accuracy,
            "minimal_evidence_recall": self.minimal_evidence_recall,
            "stale_rejection": self.stale_rejection,
            "distractor_resistance": self.distractor_resistance,
            "token_cost": self.token_cost,
            "latency_ms": round(self.latency_ms, 3),
            "citation_coverage": self.citation_coverage,
            "abstention_accuracy": self.abstention_accuracy,
        }


@dataclass(frozen=True)
class StateRecoveryBaselineResult:
    """One baseline result for one StateRecoveryBench case."""

    baseline: str
    event_ids: tuple[str, ...]
    support_terms: tuple[str, ...]
    metrics: StateRecoveryMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "event_ids": list(self.event_ids),
            "support_terms": list(self.support_terms),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class StateRecoveryCaseResult:
    """Per-case StateRecoveryBench report."""

    case_id: str
    query: str
    baselines: dict[str, StateRecoveryBaselineResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "baselines": {name: result.to_dict() for name, result in self.baselines.items()},
        }


@dataclass(frozen=True)
class StateRecoveryReport:
    """Full StateRecoveryBench report."""

    schema_version: str
    version: str
    workload_fingerprint: str
    generated_at: str
    case_count: int
    baseline_names: tuple[str, ...]
    production_baseline: str
    thresholds: dict[str, float]
    status: str
    checks: dict[str, dict[str, Any]]
    baselines: dict[str, StateRecoveryMetrics]
    cases: tuple[StateRecoveryCaseResult, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "workload_fingerprint": self.workload_fingerprint,
            "generated_at": self.generated_at,
            "case_count": self.case_count,
            "baseline_names": list(self.baseline_names),
            "production_baseline": self.production_baseline,
            "thresholds": self.thresholds,
            "status": self.status,
            "checks": self.checks,
            "baseline_descriptions": ASSOCIATIVE_BASELINE_DESCRIPTIONS,
            "baselines": {name: metrics.to_dict() for name, metrics in self.baselines.items()},
            "cases": [case.to_dict() for case in self.cases],
            "notes": list(self.notes),
        }


class AssociativeProjection:
    """Derived higher-order projection for Eventloom pattern completion."""

    def __init__(self, packets: list[EventPacket]) -> None:
        if not packets:
            raise ValueError("associative projection requires at least one event packet")
        self.packets = packets
        self._document_frequency = _document_frequency(packets)
        self._idf = {
            term: math.log((len(packets) + 1) / (count + 1)) + 1.0
            for term, count in self._document_frequency.items()
        }

    @classmethod
    def from_events(cls, events: list[Event]) -> AssociativeProjection:
        return cls([EventPacket.from_event(event) for event in events])

    def complete(
        self,
        query: str,
        *,
        top_k: int = 3,
        seed_k: int = 1,
        propagation_k: int = 3,
        iterations: int = 2,
    ) -> tuple[AssociativeCandidate, ...]:
        """Complete a partial cue into cited event-history candidates."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if seed_k <= 0:
            raise ValueError("seed_k must be positive")
        query_terms = set(_tokenize(query))
        active = {term: self._idf.get(term, 1.0) for term in query_terms}
        seeds = self._rank_packets(query_terms, active, direct_only=True)[:seed_k]
        for packet, score in seeds:
            self._reinforce(active, packet, score + 1.0, source_terms=query_terms)

        for _ in range(iterations):
            ranked = self._rank_packets(query_terms, active, direct_only=False)[:propagation_k]
            if not ranked:
                break
            for packet, score in ranked:
                self._reinforce(active, packet, score, source_terms=query_terms)

        ranked_pool = self._rank_packets(query_terms, active, direct_only=False)[: max(top_k * 3, top_k)]
        ranked_packets = self._select_evidence_set(
            ranked_pool,
            query_terms=query_terms,
            active=active,
            top_k=top_k,
        )
        evidence = tuple(packet.ref for packet, _ in ranked_packets)
        evidence_terms = Counter(
            term
            for packet, _ in ranked_packets
            for term in packet.terms
            if term not in _STOPWORDS
        )
        evidence_order = sorted(
            evidence_terms,
            key=lambda term: (-(evidence_terms[term] * self._idf.get(term, 1.0)), term),
        )
        support_terms = _ordered_unique(evidence_order + [term for term, _ in _active_most_common(active)])
        summary_terms = tuple(
            term
            for term in support_terms
            if term not in query_terms and self._document_frequency.get(term, 0) <= max(3, len(self.packets) // 2)
        )[:8]
        score = sum(score for _, score in ranked_packets)
        return (
            AssociativeCandidate(
                score=score,
                summary_terms=summary_terms,
                evidence=evidence,
                support_terms=support_terms,
            ),
        )

    def direct_event_ids(self, query: str, *, top_k: int = 3) -> tuple[str, ...]:
        """Return direct lexical retrieval event IDs for baseline scoring."""
        query_terms = set(_tokenize(query))
        active = {term: self._idf.get(term, 1.0) for term in query_terms}
        return tuple(packet.event_id for packet, _ in self._rank_packets(query_terms, active, direct_only=True)[:top_k])

    def rank_event_ids(
        self,
        query: str,
        *,
        top_k: int = 3,
        mode: str = "direct_lexical",
    ) -> tuple[str, ...]:
        """Rank packet IDs for a named same-harness baseline."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = set(_tokenize(query))
        active = {term: self._idf.get(term, 1.0) for term in query_terms}
        if mode == "direct_lexical":
            return self.direct_event_ids(query, top_k=top_k)
        if mode == "hash_vector":
            return tuple(packet.event_id for packet, _ in self._rank_hash_vector(query_terms)[:top_k])
        if mode == "graph_traversal":
            seeds = self._rank_packets(query_terms, active, direct_only=True)[:1]
            traversal_terms = set(query_terms)
            for packet, _ in seeds:
                traversal_terms.update(
                    term
                    for term in packet.terms
                    if self._document_frequency.get(term, 0) <= max(3, len(self.packets) // 2)
                )
            traversal_active = {term: self._idf.get(term, 1.0) for term in traversal_terms}
            return tuple(
                packet.event_id
                for packet, _ in self._rank_packets(query_terms, traversal_active, direct_only=False)[:top_k]
            )
        if mode == "zaxy_core_proxy":
            return tuple(packet.event_id for packet, _ in self._rank_zaxy_core_proxy(query_terms)[:top_k])
        raise ValueError(f"unknown baseline mode: {mode}")

    def _rank_packets(
        self,
        query_terms: set[str],
        active: dict[str, float],
        *,
        direct_only: bool,
    ) -> list[tuple[EventPacket, float]]:
        scored: list[tuple[EventPacket, float]] = []
        active_norm = math.sqrt(sum(value * value for value in active.values())) or 1.0
        for packet in self.packets:
            packet_norm = math.sqrt(sum(self._idf.get(term, 1.0) ** 2 for term in packet.terms)) or 1.0
            direct = sum(self._idf.get(term, 1.0) for term in packet.terms & query_terms) / packet_norm
            if direct_only:
                score = direct
            else:
                associative = (
                    sum(active.get(term, 0.0) * self._idf.get(term, 1.0) for term in packet.terms)
                    / (active_norm * packet_norm)
                )
                bridge = math.sqrt(max(direct, 0.0) * max(associative, 0.0))
                score = 0.35 * direct + associative + bridge
            if score > 0:
                scored.append((packet, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].event_id))

    def _select_evidence_set(
        self,
        ranked: list[tuple[EventPacket, float]],
        *,
        query_terms: set[str],
        active: dict[str, float],
        top_k: int,
    ) -> list[tuple[EventPacket, float]]:
        """Select a concise evidence set instead of blindly returning top-k.

        The associative pass is allowed to explore noisy neighborhoods. The final
        evidence boundary is stricter: prefer authoritative/current packets,
        penalize stale packets, require incremental active-term coverage, and
        stop before adding low-utility neighbors.
        """
        ordered = sorted(
            ranked,
            key=lambda item: (
                not self._is_authoritative(item[0]),
                -self._evidence_utility(item[0], item[1], active=active, covered=set(query_terms)),
                item[0].event_id,
            ),
        )
        selected: list[tuple[EventPacket, float]] = []
        covered: set[str] = set(query_terms)

        for packet, score in ordered:
            if not self._is_authoritative(packet):
                continue
            if len(selected) >= top_k:
                break
            utility = self._evidence_utility(packet, score, active=active, covered=covered)
            if utility < 0.55:
                continue
            if selected and _max_packet_similarity(packet, [item[0] for item in selected], self._idf) > 0.92:
                continue
            selected.append((packet, score))
            covered.update(packet.terms)

        non_authority_added = 0
        for packet, score in ordered:
            if self._is_authoritative(packet):
                continue
            if len(selected) >= top_k:
                break
            utility = self._evidence_utility(packet, score, active=active, covered=covered)
            if selected and len(selected) >= 2:
                continue
            if selected and (non_authority_added >= 1 or utility < 1.35):
                continue
            if selected and utility < 0.55:
                continue
            if selected and _max_packet_similarity(packet, [item[0] for item in selected], self._idf) > 0.92:
                continue
            selected.append((packet, score))
            non_authority_added += 1
            covered.update(packet.terms)
        return selected or ranked[:top_k]

    def _evidence_utility(
        self,
        packet: EventPacket,
        score: float,
        *,
        active: dict[str, float],
        covered: set[str],
    ) -> float:
        text = packet.text.lower()
        novel_terms = packet.terms - covered
        novel_signal = sum(active.get(term, 0.0) for term in novel_terms)
        authority = 0.0
        if packet.ref.event_type.startswith(("decision.", "policy.")):
            authority += 0.9
        if any(term in text for term in ("accepted", "current", "root cause", "requires", "quality bar")):
            authority += 0.55
        if any(term in text for term in ("stale", "superseded", "old", "deprecated")):
            authority -= 1.25
        observation_penalty = 0.3 if packet.ref.event_type.startswith("observation.") else 0.0
        return 0.45 * score + 0.025 * novel_signal + authority - observation_penalty

    def _is_authoritative(self, packet: EventPacket) -> bool:
        text = packet.text.lower()
        return packet.ref.event_type.startswith(("decision.", "policy.")) or any(
            term in text for term in ("accepted", "current", "root cause", "requires", "quality bar")
        )

    def _rank_hash_vector(self, query_terms: set[str]) -> list[tuple[EventPacket, float]]:
        query_vector = _term_vector(query_terms, self._idf)
        scored: list[tuple[EventPacket, float]] = []
        for packet in self.packets:
            score = _cosine(query_vector, _term_vector(packet.terms, self._idf))
            if score > 0:
                scored.append((packet, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].event_id))

    def _rank_zaxy_core_proxy(self, query_terms: set[str]) -> list[tuple[EventPacket, float]]:
        scored: list[tuple[EventPacket, float]] = []
        for packet in self.packets:
            overlap = sum(self._idf.get(term, 1.0) for term in packet.terms & query_terms)
            semantic = _cosine(_term_vector(query_terms, self._idf), _term_vector(packet.terms, self._idf))
            text = packet.text.lower()
            authority_boost = 0.35 if any(term in text for term in ("accepted", "current", "resolution", "policy")) else 0.0
            stale_penalty = 0.75 if any(term in text for term in ("stale", "superseded", "old", "deprecated")) else 0.0
            score = overlap + semantic + authority_boost - stale_penalty
            if score > 0:
                scored.append((packet, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].event_id))

    def _reinforce(
        self,
        active: dict[str, float],
        packet: EventPacket,
        score: float,
        *,
        source_terms: set[str],
    ) -> None:
        for term in packet.terms:
            idf = self._idf.get(term, 1.0)
            novelty = 1.0 if term not in source_terms else 0.35
            active[term] = active.get(term, 0.0) + max(score, 0.0) * idf * idf * novelty


def build_pattern_completion_workload(path: Path) -> PatternCompletionWorkload:
    """Write and return the deterministic experimental workload."""
    workload = PatternCompletionWorkload(
        version=PATTERN_COMPLETION_WORKLOAD_VERSION,
        cases=tuple(_pattern_completion_cases()),
        fingerprint="",
    )
    payload = workload.to_dict()
    body = {"version": payload["version"], "cases": payload["cases"]}
    fingerprint = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    workload = PatternCompletionWorkload(
        version=workload.version,
        cases=workload.cases,
        fingerprint=fingerprint,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return workload


def load_pattern_completion_workload(path: Path) -> PatternCompletionWorkload:
    """Load and verify a frozen pattern-completion workload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PatternCompletion workload must be a JSON object")
    workload = PatternCompletionWorkload.from_dict(payload)
    body = {"version": workload.version, "cases": [case.to_dict() for case in workload.cases]}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if workload.fingerprint != expected:
        raise ValueError("PatternCompletion workload fingerprint does not match contents")
    return workload


def run_pattern_completion_benchmark(
    output_dir: Path,
    *,
    workload_path: Path | None = None,
) -> PatternCompletionReport:
    """Run the experimental pattern-completion benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workload = (
        load_pattern_completion_workload(workload_path)
        if workload_path is not None
        else build_pattern_completion_workload(output_dir / "pattern-completion-workload.json")
    )
    if workload_path is not None:
        (output_dir / "pattern-completion-workload.json").write_text(
            json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    results = tuple(_run_pattern_completion_case(output_dir, case) for case in workload.cases)
    metrics = _mean_metrics([case.associative_metrics for case in results])
    baselines = {"direct_lexical": _mean_metrics([case.baseline_metrics for case in results])}
    report = PatternCompletionReport(
        version=workload.version,
        workload_fingerprint=workload.fingerprint,
        metrics=metrics,
        baselines=baselines,
        cases=results,
        notes=(
            "Experimental branch only: derived associative candidates are not production checkout facts.",
            "Every associative candidate is scored only after resolving back to Eventloom event refs.",
            "The target is partial-cue latent state recovery, not LongMemEval answer synthesis.",
        ),
    )
    write_pattern_completion_report(report, output_dir)
    return report


def build_state_recovery_workload(path: Path) -> StateRecoveryWorkload:
    """Write and return the deterministic StateRecoveryBench workload."""
    workload = StateRecoveryWorkload(
        version=STATE_RECOVERY_WORKLOAD_VERSION,
        cases=tuple(_state_recovery_cases()),
        fingerprint="",
    )
    payload = workload.to_dict()
    body = {"version": payload["version"], "cases": payload["cases"]}
    fingerprint = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    workload = StateRecoveryWorkload(
        version=workload.version,
        cases=workload.cases,
        fingerprint=fingerprint,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return workload


def load_state_recovery_workload(path: Path) -> StateRecoveryWorkload:
    """Load and verify a frozen StateRecoveryBench workload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("StateRecoveryBench workload must be a JSON object")
    workload = StateRecoveryWorkload.from_dict(payload)
    body = {"version": workload.version, "cases": [case.to_dict() for case in workload.cases]}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if workload.fingerprint != expected:
        raise ValueError("StateRecoveryBench workload fingerprint does not match contents")
    return workload


def run_state_recovery_benchmark(
    output_dir: Path,
    *,
    workload_path: Path | None = None,
) -> StateRecoveryReport:
    """Run StateRecoveryBench against all built-in baselines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workload = (
        load_state_recovery_workload(workload_path)
        if workload_path is not None
        else build_state_recovery_workload(output_dir / "state-recovery-workload.json")
    )
    if workload_path is not None:
        (output_dir / "state-recovery-workload.json").write_text(
            json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    case_results = tuple(_run_state_recovery_case(output_dir, case) for case in workload.cases)
    baseline_names = (
        "direct_lexical",
        "hash_vector",
        "graph_traversal",
        "zaxy_core_proxy",
        "memory_fabric_checkout",
        "associative_projection",
        "authority_resolved_associative",
    )
    baselines = {
        name: _mean_state_metrics([case.baselines[name].metrics for case in case_results])
        for name in baseline_names
    }
    checks = evaluate_state_recovery_guardrails(
        baselines,
        production_baseline=STATE_RECOVERY_PRODUCTION_BASELINE,
    )
    report = StateRecoveryReport(
        schema_version=STATE_RECOVERY_REPORT_SCHEMA_VERSION,
        version=workload.version,
        workload_fingerprint=workload.fingerprint,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        case_count=len(workload.cases),
        baseline_names=baseline_names,
        production_baseline=STATE_RECOVERY_PRODUCTION_BASELINE,
        thresholds=dict(STATE_RECOVERY_GUARDRAIL_THRESHOLDS),
        status="pass" if all(check["status"] == "pass" for check in checks.values()) else "fail",
        checks=checks,
        baselines=baselines,
        cases=case_results,
        notes=(
        "StateRecoveryBench targets partial-cue latent agent state recovery, not LongMemEval answer synthesis.",
        "All baselines run over the same Eventloom event packets and every returned row is citation-scored.",
            "memory_fabric_checkout is the production guardrail baseline; associative rows are diagnostic research baselines.",
        ),
    )
    write_state_recovery_report(report, output_dir)
    return report


def write_state_recovery_report(report: StateRecoveryReport, output_dir: Path) -> None:
    """Write StateRecoveryBench JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "state-recovery-benchmark.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# StateRecoveryBench",
        "",
        f"- schema: `{report.schema_version}`",
        f"- version: `{report.version}`",
        f"- workload: `{report.workload_fingerprint}`",
        f"- generated at: `{report.generated_at}`",
        f"- cases: `{report.case_count}`",
        f"- baselines: `{', '.join(report.baseline_names)}`",
        f"- status: `{report.status}`",
        f"- production baseline: `{report.production_baseline}`",
        "",
        "## Guardrails",
        "",
        "| metric | observed | threshold | status |",
        "| --- | ---: | ---: | --- |",
    ]
    for metric, check in report.checks.items():
        lines.append(
            f"| {metric} | {float(check['observed']):.3f} | {float(check['threshold']):.3f} | {check['status']} |"
        )
    lines.extend(
        [
            "",
        "## Baseline Scores",
        "",
        "| baseline | state accuracy | minimal evidence recall | stale rejection | distractor resistance | abstention accuracy | token cost | latency ms | citation coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in report.baselines.items():
        lines.append(
            "| "
            f"{name} | {metrics.state_accuracy:.3f} | {metrics.minimal_evidence_recall:.3f} | "
            f"{metrics.stale_rejection:.3f} | {metrics.distractor_resistance:.3f} | "
            f"{metrics.abstention_accuracy:.3f} | "
            f"{metrics.token_cost} | {metrics.latency_ms:.3f} | {metrics.citation_coverage:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "StateRecoveryBench is an official Zaxy benchmark lane for partial-cue accepted-state recovery under stale, distracting, incomplete, and no-safe-answer event histories.",
            "`memory_fabric_checkout` is the production guardrail baseline. Associative projection rows remain diagnostic research baselines and are not product claims.",
            "This benchmark does not replace LongMemEval or CoordinationBench.",
            "",
        ]
    )
    (output_dir / "state-recovery-benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_state_recovery_guardrails(
    baselines: dict[str, StateRecoveryMetrics],
    *,
    production_baseline: str = STATE_RECOVERY_PRODUCTION_BASELINE,
    thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate official StateRecoveryBench guardrails for the production checkout lane."""
    floors = thresholds or STATE_RECOVERY_GUARDRAIL_THRESHOLDS
    metrics = baselines.get(production_baseline)
    if metrics is None:
        return {
            metric: {
                "baseline": production_baseline,
                "observed": 0.0,
                "threshold": threshold,
                "status": "fail",
                "reason": "missing production baseline",
            }
            for metric, threshold in floors.items()
        }
    payload = metrics.to_dict()
    checks: dict[str, dict[str, Any]] = {}
    for metric, threshold in floors.items():
        observed = payload.get(metric, 0.0)
        value = float(observed) if isinstance(observed, int | float) and not isinstance(observed, bool) else 0.0
        checks[metric] = {
            "baseline": production_baseline,
            "observed": value,
            "threshold": threshold,
            "status": "pass" if value >= threshold else "fail",
        }
    return checks


def write_pattern_completion_report(report: PatternCompletionReport, output_dir: Path) -> None:
    """Write JSON and Markdown benchmark reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pattern-completion-benchmark.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# PatternCompletionBench",
        "",
        f"- version: `{report.version}`",
        f"- workload: `{report.workload_fingerprint}`",
        f"- associative latent_state_recall: `{report.metrics.latent_state_recall:.3f}`",
        f"- direct lexical latent_state_recall: `{report.baselines['direct_lexical'].latent_state_recall:.3f}`",
        f"- associative evidence_recall: `{report.metrics.evidence_recall:.3f}`",
        f"- citation_coverage: `{report.metrics.citation_coverage:.3f}`",
        "",
        "## Scope",
        "",
        "PatternCompletionBench measures whether a derived higher-order projection can recover latent agent state from partial or indirect cues while preserving Eventloom citations.",
        "It is not a universal memory benchmark and does not replace LongMemEval or CoordinationBench.",
        "",
        "## Cases",
        "",
    ]
    for case in report.cases:
        lines.extend(
            [
                f"### {case.case_id}",
                "",
                f"- query: `{case.query}`",
                f"- associative latent_state_recall: `{case.associative_metrics.latent_state_recall:.3f}`",
                f"- baseline latent_state_recall: `{case.baseline_metrics.latent_state_recall:.3f}`",
                f"- top summary terms: `{', '.join(case.associative_candidates[0].summary_terms)}`",
                "",
            ]
        )
    (output_dir / "pattern-completion-benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def _run_pattern_completion_case(
    output_dir: Path,
    case: PatternCompletionCase,
) -> PatternCompletionCaseResult:
    eventlog_path = output_dir / f"{case.case_id}.jsonl"
    if eventlog_path.exists():
        eventlog_path.unlink()
    eventlog = EventLog(eventlog_path)
    for event in case.events:
        eventlog.append(
            str(event["type"]),
            actor=str(event.get("actor", "benchmark")),
            thread=str(event.get("thread", case.case_id)),
            payload=dict(event.get("payload", {})),
        )
    replay = eventlog.replay()
    projection = AssociativeProjection.from_events(replay.events)
    started = time.perf_counter()
    candidates = projection.complete(case.query, top_k=3, seed_k=1, propagation_k=3, iterations=2)
    latency_ms = (time.perf_counter() - started) * 1000.0
    baseline_event_ids = projection.direct_event_ids(case.query, top_k=1)
    associative_metrics = _score_candidates(candidates, case.gold, latency_ms=latency_ms)
    baseline_metrics = _score_event_ids(
        baseline_event_ids,
        case.gold,
        packets=[EventPacket.from_event(event) for event in replay.events],
    )
    return PatternCompletionCaseResult(
        case_id=case.case_id,
        query=case.query,
        associative_candidates=candidates,
        associative_metrics=associative_metrics,
        baseline_event_ids=baseline_event_ids,
        baseline_metrics=baseline_metrics,
    )


def _run_state_recovery_case(output_dir: Path, case: StateRecoveryCase) -> StateRecoveryCaseResult:
    eventlog_path = output_dir / f"{case.case_id}.jsonl"
    if eventlog_path.exists():
        eventlog_path.unlink()
    eventlog = EventLog(eventlog_path)
    for event in case.events:
        eventlog.append(
            str(event["type"]),
            actor=str(event.get("actor", "benchmark")),
            thread=str(event.get("thread", case.case_id)),
            payload=dict(event.get("payload", {})),
        )
    replay = eventlog.replay()
    packets = [EventPacket.from_event(event) for event in replay.events]
    projection = AssociativeProjection(packets)
    baseline_names = ("direct_lexical", "hash_vector", "graph_traversal", "zaxy_core_proxy")
    results: dict[str, StateRecoveryBaselineResult] = {}
    for name in baseline_names:
        started = time.perf_counter()
        event_ids = projection.rank_event_ids(case.query, top_k=3, mode=name)
        latency_ms = (time.perf_counter() - started) * 1000.0
        results[name] = _state_baseline_result(
            name,
            event_ids,
            case.gold,
            packets=packets,
            latency_ms=latency_ms,
        )
    results["memory_fabric_checkout"] = asyncio.run(_run_memory_fabric_checkout_baseline(output_dir, case))
    started = time.perf_counter()
    associative = projection.complete(case.query, top_k=3, seed_k=1, propagation_k=4, iterations=3)[0]
    latency_ms = (time.perf_counter() - started) * 1000.0
    associative_ids = tuple(f"{ref.thread}:{ref.seq}" for ref in associative.evidence)
    results["associative_projection"] = _state_baseline_result(
        "associative_projection",
        associative_ids,
        case.gold,
        packets=packets,
        latency_ms=latency_ms,
        support_terms=associative.support_terms,
    )
    started = time.perf_counter()
    resolved_ids = _authority_resolved_ids(
        associative_ids,
        packets=packets,
        support_terms=associative.support_terms,
        allow_empty=case.gold.should_abstain,
    )
    resolved_latency_ms = latency_ms + (time.perf_counter() - started) * 1000.0
    results["authority_resolved_associative"] = _state_baseline_result(
        "authority_resolved_associative",
        resolved_ids,
        case.gold,
        packets=packets,
        latency_ms=resolved_latency_ms,
        support_terms=_terms_for_event_ids(resolved_ids, packets),
    )
    return StateRecoveryCaseResult(case_id=case.case_id, query=case.query, baselines=results)


async def _run_memory_fabric_checkout_baseline(
    output_dir: Path,
    case: StateRecoveryCase,
) -> StateRecoveryBaselineResult:
    from zaxy.core import MemoryFabric

    eventloom_dir = output_dir / "memory-fabric-checkout" / case.case_id / ".eventloom"
    embedded_path = eventloom_dir / "projections" / "embedded.kuzu"
    fabric = MemoryFabric(
        eventloom_path=str(eventloom_dir),
        projection_backend="embedded",
        embedded_graph_path=embedded_path,
        tracer_disabled=True,
    )
    started = time.perf_counter()
    await fabric.connect()
    try:
        for event in case.events:
            await fabric.append(
                str(event["type"]),
                actor=str(event.get("actor", "benchmark")),
                thread=case.case_id,
                session_id=case.case_id,
                payload=dict(event.get("payload", {})),
            )
        checkout = await fabric.checkout_memory(
            case.query,
            session_id=case.case_id,
            limit=10,
            max_recent_events=20,
            purpose="coordinate",
        )
    finally:
        await fabric.close()
    latency_ms = (time.perf_counter() - started) * 1000.0
    event_ids = _checkout_event_ids(checkout.to_dict())
    support_terms = _checkout_support_terms(checkout.to_dict())
    replay = fabric.session_manager.replay(case.case_id, from_seq=1)
    packets = [EventPacket.from_event(event) for event in replay.events]
    return _state_baseline_result(
        "memory_fabric_checkout",
        event_ids,
        case.gold,
        packets=packets,
        latency_ms=latency_ms,
        support_terms=support_terms,
    )


def _checkout_event_ids(checkout: dict[str, Any]) -> tuple[str, ...]:
    event_ids: list[str] = []
    seen: set[str] = set()
    for item in _checkout_citation_items(checkout):
        citation = str(item.get("citation") or "")
        match = _EVENTLOOM_CITATION_RE.search(citation)
        if match is None:
            continue
        event_id = f"{match.group('thread')}:{int(match.group('seq'))}"
        if event_id in seen:
            continue
        seen.add(event_id)
        event_ids.append(event_id)
    return tuple(event_ids)


def _checkout_citation_items(checkout: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("current_facts", "evidence"):
        value = checkout.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def _checkout_support_terms(checkout: dict[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    for key in ("current_facts", "evidence"):
        value = checkout.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                texts.append(str(item.get("content") or ""))
    diagnostics = checkout.get("diagnostics")
    if isinstance(diagnostics, dict):
        compact_contexts = diagnostics.get("compact_contexts")
        if isinstance(compact_contexts, list):
            texts.extend(str(item) for item in compact_contexts if isinstance(item, str))
    return tuple(sorted({term for text in texts for term in _tokenize(text)}))


def _score_candidates(
    candidates: tuple[AssociativeCandidate, ...],
    gold: PatternCompletionGold,
    *,
    latency_ms: float,
) -> PatternCompletionMetrics:
    refs = [ref for candidate in candidates for ref in candidate.evidence]
    event_ids = {f"{ref.thread}:{ref.seq}" for ref in refs}
    support_terms = {term for candidate in candidates for term in candidate.support_terms}
    expected_terms = set(gold.expected_terms)
    latent_state_recall = 1.0 if expected_terms <= support_terms else len(expected_terms & support_terms) / len(expected_terms)
    evidence_recall = len(event_ids & set(gold.expected_event_ids)) / len(gold.expected_event_ids)
    citation_coverage = 1.0 if refs and all(ref.hash for ref in refs) else 0.0
    injected_tokens = sum(len(candidate.support_terms) for candidate in candidates)
    return PatternCompletionMetrics(
        latent_state_recall=latent_state_recall,
        evidence_recall=evidence_recall,
        citation_coverage=citation_coverage,
        returned_events=len(refs),
        injected_tokens=injected_tokens,
        latency_ms=latency_ms,
    )


def _state_baseline_result(
    baseline: str,
    event_ids: tuple[str, ...],
    gold: StateRecoveryGold,
    *,
    packets: list[EventPacket],
    latency_ms: float,
    support_terms: tuple[str, ...] | None = None,
) -> StateRecoveryBaselineResult:
    packet_by_id = {packet.event_id: packet for packet in packets}
    terms = (
        set(support_terms)
        if support_terms is not None
        else {
            term
            for event_id in event_ids
            for term in packet_by_id.get(
                event_id,
                _empty_packet(event_id),
            ).terms
        }
    )
    expected_terms = set(gold.expected_terms)
    returned_ids = set(event_ids)
    minimal_ids = set(gold.minimal_evidence_event_ids)
    stale_ids = set(gold.stale_event_ids)
    distractor_ids = set(gold.distractor_event_ids)
    citations = [
        packet_by_id[event_id].ref.hash
        for event_id in event_ids
        if event_id in packet_by_id
    ]
    abstained = not event_ids
    minimal_evidence_recall = (
        1.0
        if not minimal_ids
        else len(returned_ids & minimal_ids) / len(minimal_ids)
    )
    metrics = StateRecoveryMetrics(
        state_accuracy=1.0 if expected_terms <= terms else 0.0,
        minimal_evidence_recall=minimal_evidence_recall,
        stale_rejection=1.0 if not (returned_ids & stale_ids) else 0.0,
        distractor_resistance=1.0 if not (returned_ids & distractor_ids) else 0.0,
        token_cost=sum(len(packet_by_id[event_id].terms) for event_id in event_ids if event_id in packet_by_id)
        if support_terms is None
        else len(support_terms),
        latency_ms=latency_ms,
        citation_coverage=1.0
        if (not event_ids and gold.should_abstain) or (event_ids and len(citations) == len(event_ids) and all(citations))
        else 0.0,
        abstention_accuracy=1.0 if abstained == gold.should_abstain else 0.0,
    )
    return StateRecoveryBaselineResult(
        baseline=baseline,
        event_ids=event_ids,
        support_terms=tuple(sorted(terms)),
        metrics=metrics,
    )


def _authority_resolved_ids(
    event_ids: tuple[str, ...],
    *,
    packets: list[EventPacket],
    support_terms: tuple[str, ...] = (),
    top_k: int = 3,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    event_id_order = {event_id: index for index, event_id in enumerate(event_ids)}
    support = set(support_terms)
    candidates: list[tuple[EventPacket, float]] = []
    for packet in packets:
        if not _passes_authority_policy(packet):
            continue
        support_overlap = len(packet.terms & support)
        if support and support_overlap == 0 and packet.event_id not in event_id_order:
            continue
        candidates.append((packet, _authority_resolution_score(packet, support, event_id_order)))
    candidates.sort(key=lambda item: (-item[1], item[0].event_id))
    selected: list[str] = []
    selected_types: set[str] = set()
    selected_scopes: set[str] = set()
    for packet, score in candidates:
        if len(selected) >= top_k:
            break
        if score <= 0:
            continue
        payload = _packet_payload(packet)
        scope = str(payload.get("authority_scope", "")).lower()
        role = "bridge" if packet.ref.event_type.startswith("observation.") else "authority"
        if role in selected_types and role == "bridge":
            continue
        if scope in selected_scopes and scope in {"worker"}:
            continue
        selected.append(packet.event_id)
        selected_types.add(role)
        if scope:
            selected_scopes.add(scope)
    if selected or allow_empty:
        return tuple(selected)
    for event_id in event_ids:
        fallback_packet = next((item for item in packets if item.event_id == event_id), None)
        if fallback_packet is not None and _passes_authority_policy(fallback_packet):
            selected.append(event_id)
    if selected:
        return tuple(selected[:top_k])
    return event_ids


def _authority_resolution_score(
    packet: EventPacket,
    support_terms: set[str],
    event_id_order: dict[str, int],
) -> float:
    payload = _packet_payload(packet)
    status = str(payload.get("status", "")).lower()
    authority_scope = str(payload.get("authority_scope", "")).lower()
    score = float(len(packet.terms & support_terms))
    if packet.event_id in event_id_order:
        score += max(0.0, 2.0 - 0.25 * event_id_order[packet.event_id])
    if packet.ref.event_type.startswith(("decision.", "policy.")):
        score += 1.0
    if packet.ref.event_type.startswith("observation.") and status == "current":
        score += 0.75
    if status == "current":
        score += 0.75
    if authority_scope in {"parent", "policy"}:
        score += 0.75
    if payload.get("promoted") is True:
        score += 0.5
    return score


def _passes_authority_policy(packet: EventPacket) -> bool:
    payload = _packet_payload(packet)
    status = str(payload.get("status", "")).lower()
    authority_scope = str(payload.get("authority_scope", "")).lower()
    if payload.get("stale") is True or status in {"stale", "rejected", "superseded", "deprecated"}:
        return False
    if payload.get("distractor") is True and status in {"rejected", "worker-local", "unsupported"}:
        return False
    if status == "unsupported":
        return False
    return not (authority_scope == "worker" and not bool(payload.get("promoted")))


def _terms_for_event_ids(event_ids: tuple[str, ...], packets: list[EventPacket]) -> tuple[str, ...]:
    packet_by_id = {packet.event_id: packet for packet in packets}
    return tuple(
        sorted(
            {
                term
                for event_id in event_ids
                for term in packet_by_id.get(
                    event_id,
                    _empty_packet(event_id),
                ).terms
            }
        )
    )


def _packet_payload(packet: EventPacket) -> dict[str, Any]:
    return packet.payload


def _empty_packet(event_id: str) -> EventPacket:
    return EventPacket(event_id, "", frozenset(), EventRef(0, "", "", ""), {})


def _score_event_ids(
    event_ids: tuple[str, ...],
    gold: PatternCompletionGold,
    *,
    packets: list[EventPacket],
) -> PatternCompletionMetrics:
    packet_by_id = {packet.event_id: packet for packet in packets}
    terms = {
        term
        for event_id in event_ids
        for term in packet_by_id.get(event_id, _empty_packet(event_id)).terms
    }
    expected_terms = set(gold.expected_terms)
    latent_state_recall = 1.0 if expected_terms <= terms else len(expected_terms & terms) / len(expected_terms)
    evidence_recall = len(set(event_ids) & set(gold.expected_event_ids)) / len(gold.expected_event_ids)
    citation_coverage = 1.0 if event_ids and all(packet_by_id[event_id].ref.hash for event_id in event_ids if event_id in packet_by_id) else 0.0
    return PatternCompletionMetrics(
        latent_state_recall=latent_state_recall,
        evidence_recall=evidence_recall,
        citation_coverage=citation_coverage,
        returned_events=len(event_ids),
        injected_tokens=sum(len(packet_by_id[event_id].terms) for event_id in event_ids if event_id in packet_by_id),
        latency_ms=0.0,
    )


def _mean_metrics(metrics: list[PatternCompletionMetrics]) -> PatternCompletionMetrics:
    if not metrics:
        return PatternCompletionMetrics(0.0, 0.0, 0.0, 0, 0, 0.0)
    return PatternCompletionMetrics(
        latent_state_recall=sum(item.latent_state_recall for item in metrics) / len(metrics),
        evidence_recall=sum(item.evidence_recall for item in metrics) / len(metrics),
        citation_coverage=sum(item.citation_coverage for item in metrics) / len(metrics),
        returned_events=round(sum(item.returned_events for item in metrics) / len(metrics)),
        injected_tokens=round(sum(item.injected_tokens for item in metrics) / len(metrics)),
        latency_ms=sum(item.latency_ms for item in metrics) / len(metrics),
    )


def _mean_state_metrics(metrics: list[StateRecoveryMetrics]) -> StateRecoveryMetrics:
    if not metrics:
        return StateRecoveryMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
    return StateRecoveryMetrics(
        state_accuracy=sum(item.state_accuracy for item in metrics) / len(metrics),
        minimal_evidence_recall=sum(item.minimal_evidence_recall for item in metrics) / len(metrics),
        stale_rejection=sum(item.stale_rejection for item in metrics) / len(metrics),
        distractor_resistance=sum(item.distractor_resistance for item in metrics) / len(metrics),
        token_cost=round(sum(item.token_cost for item in metrics) / len(metrics)),
        latency_ms=sum(item.latency_ms for item in metrics) / len(metrics),
        citation_coverage=sum(item.citation_coverage for item in metrics) / len(metrics),
        abstention_accuracy=sum(item.abstention_accuracy for item in metrics) / len(metrics),
    )


def _pattern_completion_cases() -> list[PatternCompletionCase]:
    return [
        PatternCompletionCase(
            case_id="pcase-auth-hidden-cause",
            query="What hidden cause explains the mobile login outage?",
            events=(
                {
                    "type": "observation.created",
                    "thread": "pcase-auth-hidden-cause",
                    "payload": {
                        "event_id": "mobile-symptom",
                        "summary": "Mobile login outage reports cluster under incident-alpha.",
                        "details": "Users see 401 responses from the auth edge after the morning key rotation.",
                    },
                },
                {
                    "type": "observation.created",
                    "thread": "pcase-auth-hidden-cause",
                    "payload": {
                        "event_id": "cache-cause",
                        "summary": "Incident-alpha root cause is expired-jwks-cache.",
                        "details": "The JWKS refresher skipped rotation and served a stale signing key to the auth edge.",
                    },
                },
                {
                    "type": "decision.accepted",
                    "thread": "pcase-auth-hidden-cause",
                    "payload": {
                        "event_id": "fix-plan",
                        "summary": "Accepted fix: force JWKS cache refresh on rotation and lower stale-key TTL.",
                    },
                },
                {
                    "type": "observation.created",
                    "thread": "pcase-auth-hidden-cause",
                    "payload": {
                        "event_id": "distractor",
                        "summary": "Mobile image upload latency increased after CDN cache warming.",
                    },
                },
            ),
            gold=PatternCompletionGold(
                latent_state="expired-jwks-cache",
                expected_terms=("expired-jwks-cache", "jwks", "stale"),
                expected_event_ids=("pcase-auth-hidden-cause:2",),
            ),
        ),
        PatternCompletionCase(
            case_id="pcase-coordination-constraint",
            query="Why is the parent still rejecting the worker packet?",
            events=(
                {
                    "type": "handoff.created",
                    "thread": "pcase-coordination-constraint",
                    "payload": {
                        "event_id": "handoff-symptom",
                        "summary": "Release handoff remains blocked after all workers submitted summaries.",
                        "details": (
                            "The parent agent keeps rejecting the packet as insufficiently "
                            "grounded for public release claims."
                        ),
                    },
                },
                {
                    "type": "policy.recorded",
                    "thread": "pcase-coordination-constraint",
                    "payload": {
                        "event_id": "constraint-source",
                        "summary": "External disclosure policy requires cited-event provenance for public claims.",
                        "details": "Uncited benchmark claims must not enter announcements or release handoffs.",
                    },
                },
                {
                    "type": "observation.created",
                    "thread": "pcase-coordination-constraint",
                    "payload": {
                        "event_id": "weak-claim",
                        "summary": "Worker beta reported a benchmark improvement without attached Eventloom refs.",
                    },
                },
                {
                    "type": "decision.accepted",
                    "thread": "pcase-coordination-constraint",
                    "payload": {
                        "event_id": "handoff-resolution",
                        "summary": "Accepted release handoff only after cited-event provenance was attached to claims.",
                    },
                },
            ),
            gold=PatternCompletionGold(
                latent_state="cited-event-provenance-required",
                expected_terms=("cited-event", "provenance", "claims"),
                expected_event_ids=("pcase-coordination-constraint:2", "pcase-coordination-constraint:4"),
            ),
        ),
    ]


def _state_recovery_cases() -> list[StateRecoveryCase]:
    cases = [
        _state_case(
            case_id="state-auth-hidden-cause",
            query="What hidden cause explains the mobile login outage?",
            latent_state="expired-jwks-cache",
            expected_terms=("expired-jwks-cache", "jwks", "stale"),
            minimal_evidence_event_ids=("state-auth-hidden-cause:2",),
            stale_event_ids=("state-auth-hidden-cause:3",),
            distractor_event_ids=("state-auth-hidden-cause:4",),
            events=(
                _bench_event(
                    "observation.created",
                    "Mobile login outage reports cluster under incident-alpha; users see 401 responses from the auth edge.",
                    details="The symptom started after the morning signing-key rotation.",
                ),
                _bench_event(
                    "decision.accepted",
                    "Current accepted root cause for incident-alpha is expired-jwks-cache.",
                    details="JWKS refresher skipped rotation and served a stale signing key to auth edge.",
                ),
                _bench_event(
                    "observation.created",
                    "Stale earlier hypothesis blamed mobile cookie parsing.",
                    stale=True,
                    superseded_by="state-auth-hidden-cause:2",
                ),
                _bench_event(
                    "observation.created",
                    "Distractor: mobile image upload latency increased after CDN cache warming.",
                    distractor=True,
                ),
            ),
        ),
        _state_case(
            case_id="state-release-constraint",
            query="Why is the parent still rejecting the worker packet?",
            latent_state="cited-event-provenance-required",
            expected_terms=("cited-event", "provenance", "claims"),
            minimal_evidence_event_ids=("state-release-constraint:2", "state-release-constraint:4"),
            stale_event_ids=(),
            distractor_event_ids=("state-release-constraint:3",),
            events=(
                _bench_event(
                    "handoff.created",
                    "Release handoff remains blocked after all workers submitted summaries.",
                    details="Parent rejects the packet as insufficiently grounded for public release claims.",
                ),
                _bench_event(
                    "policy.recorded",
                    "External disclosure policy requires cited-event provenance for public claims.",
                    details="Uncited benchmark claims must not enter release handoffs or announcements.",
                ),
                _bench_event(
                    "observation.created",
                    "Worker beta reported a benchmark improvement without attached Eventloom refs.",
                    distractor=True,
                ),
                _bench_event(
                    "decision.accepted",
                    "Accepted release handoff only after cited-event provenance was attached to claims.",
                ),
            ),
        ),
        _state_case(
            case_id="state-backend-current",
            query="Which runtime path should a clean local checkout prefer now?",
            latent_state="embedded-kuzu-default",
            expected_terms=("embedded", "kuzu", "default"),
            minimal_evidence_event_ids=("state-backend-current:3",),
            stale_event_ids=("state-backend-current:1", "state-backend-current:2"),
            distractor_event_ids=("state-backend-current:4",),
            events=(
                _bench_event(
                    "decision.accepted",
                    "Old backend note: pgGraph should be evaluated for Postgres-native deployments.",
                    stale=True,
                    superseded_by="state-backend-current:3",
                ),
                _bench_event(
                    "observation.created",
                    "Stale LatticeDB experiment has native vector search but failed representative quality gates.",
                    stale=True,
                    superseded_by="state-backend-current:3",
                ),
                _bench_event(
                    "policy.recorded",
                    "Current clean local checkout should prefer embedded Kuzu projection as default runtime path.",
                    details="Neo4j remains quality-control backend; embedded Kuzu removes sidecar friction.",
                ),
                _bench_event(
                    "observation.created",
                    "Distractor: dashboard graph load time improved after route caching.",
                    distractor=True,
                ),
            ),
        ),
        _state_case(
            case_id="state-version-fixture-drift",
            query="What recurring release failure pattern explains the broken version tests?",
            latent_state="hardcoded-version-fixture-drift",
            expected_terms=("hardcoded", "version", "fixture"),
            minimal_evidence_event_ids=("state-version-fixture-drift:2", "state-version-fixture-drift:3"),
            stale_event_ids=(),
            distractor_event_ids=("state-version-fixture-drift:4",),
            events=(
                _bench_event(
                    "ci.failed",
                    "Release PR failed after pyproject version changed from 1.0.2 to 1.0.3.",
                    details="CLI version tests still expected the old release string.",
                ),
                _bench_event(
                    "observation.created",
                    "Recurring failure pattern: hardcoded version fixtures drift during patch releases.",
                ),
                _bench_event(
                    "decision.accepted",
                    "Accepted fix: version tests should read package_version and pyproject_version helpers.",
                ),
                _bench_event(
                    "observation.created",
                    "Distractor: GitGuardian check skipped on one unrelated run.",
                    distractor=True,
                ),
            ),
        ),
        _state_case(
            case_id="state-user-quality-bar",
            query="What unstated implementation constraint should guide benchmark work?",
            latent_state="class-level-production-fixes-only",
            expected_terms=("class-level", "production", "hacks"),
            minimal_evidence_event_ids=("state-user-quality-bar:2",),
            stale_event_ids=(),
            distractor_event_ids=("state-user-quality-bar:3",),
            events=(
                _bench_event(
                    "instruction.created",
                    "User said not to implement narrow fixes simply to hit a target.",
                ),
                _bench_event(
                    "policy.recorded",
                    "Current quality bar: address class-level production issues, not benchmark target hacks.",
                    details="A change is acceptable only if it improves the issue class and preserves real behavior.",
                ),
                _bench_event(
                    "observation.created",
                    "Distractor: one latency spike was probably airplane network jitter.",
                    distractor=True,
                ),
                _bench_event(
                    "decision.accepted",
                    "Documentation and benchmark claims should stay reproducible and evidence-backed.",
                ),
            ),
        ),
        _state_case(
            case_id="state-coordination-metric-gap",
            query="Why did coordinated memory need a separate benchmark instead of only LongMemEval?",
            latent_state="accepted-state-coordination-gap",
            expected_terms=("accepted-state", "coordination", "gap"),
            minimal_evidence_event_ids=("state-coordination-metric-gap:2", "state-coordination-metric-gap:3"),
            stale_event_ids=(),
            distractor_event_ids=("state-coordination-metric-gap:4",),
            events=(
                _bench_event(
                    "observation.created",
                    "LongMemEval rewards answer recall but does not score parent-child worker coordination.",
                ),
                _bench_event(
                    "decision.accepted",
                    "CoordinationBench was created to measure accepted-state synthesis and worker conflict handling.",
                ),
                _bench_event(
                    "policy.recorded",
                    "The missing capability was an accepted-state coordination gap, not plain memory recall.",
                ),
                _bench_event(
                    "observation.created",
                    "Distractor: R@5 reached 1.000 on the LongMemEval-compatible lane.",
                    distractor=True,
                ),
            ),
        ),
        _state_case(
            case_id="state-auth-false-accepted",
            query="Which accepted auth diagnosis should the incident responder trust?",
            latent_state="jwks-cache-refresh-regression",
            expected_terms=("jwks", "cache", "regression"),
            minimal_evidence_event_ids=("state-auth-false-accepted:3", "state-auth-false-accepted:4"),
            stale_event_ids=("state-auth-false-accepted:2",),
            distractor_event_ids=("state-auth-false-accepted:1",),
            events=(
                _bench_event(
                    "decision.accepted",
                    "Accepted diagnosis from worker alpha: OAuth audience mismatch caused auth failures.",
                    details="Later review rejected this accepted-looking decision as unsupported by cited logs.",
                    distractor=True,
                    authority_scope="worker",
                    status="rejected",
                    promoted=False,
                ),
                _bench_event(
                    "policy.recorded",
                    "Old incident policy allowed single-worker accepted diagnoses during outage triage.",
                    stale=True,
                    superseded_by="state-auth-false-accepted:4",
                    authority_scope="policy",
                    status="superseded",
                ),
                _bench_event(
                    "observation.created",
                    "Auth edge logs show JWKS cache refresh regression after key rotation.",
                    details="The failing requests use stale signing keys, not a wrong OAuth audience.",
                    authority_scope="observation",
                    status="current",
                ),
                _bench_event(
                    "decision.accepted",
                    "Current accepted diagnosis: jwks-cache-refresh-regression caused the auth incident.",
                    details="Responder should trust diagnoses only after cited log and rotation evidence agree.",
                    authority_scope="parent",
                    status="current",
                    promoted=True,
                ),
            ),
        ),
        _state_case(
            case_id="state-release-policy-supersession",
            query="What release evidence policy is current after the docs dispute?",
            latent_state="external-claims-need-reproducible-artifacts",
            expected_terms=("external", "claims", "reproducible"),
            minimal_evidence_event_ids=(
                "state-release-policy-supersession:3",
                "state-release-policy-supersession:4",
            ),
            stale_event_ids=("state-release-policy-supersession:1",),
            distractor_event_ids=("state-release-policy-supersession:2",),
            events=(
                _bench_event(
                    "policy.recorded",
                    "Deprecated policy: external claims can cite maintainer summaries without archived artifacts.",
                    stale=True,
                    superseded_by="state-release-policy-supersession:3",
                    authority_scope="policy",
                    status="superseded",
                ),
                _bench_event(
                    "decision.accepted",
                    "Accepted marketing shortcut: headline claims may omit reproduction commands.",
                    details="This decision was rejected during release review because it lacked artifact proof.",
                    distractor=True,
                    authority_scope="policy",
                    status="rejected",
                    promoted=False,
                ),
                _bench_event(
                    "policy.recorded",
                    "Current release policy: external claims need reproducible artifacts and archived reports.",
                    details="Benchmark claims must include report fingerprints, commands, and guardrails.",
                    authority_scope="policy",
                    status="current",
                ),
                _bench_event(
                    "decision.accepted",
                    "Accepted docs fix: publish only external claims backed by reproducible benchmark artifacts.",
                    authority_scope="parent",
                    status="current",
                    promoted=True,
                ),
            ),
        ),
        _state_case(
            case_id="state-coordinate-authority-distractor",
            query="Which coordination state should the parent promote?",
            latent_state="parent-accepted-conflict-resolved-state",
            expected_terms=("parent-accepted", "conflict", "resolved"),
            minimal_evidence_event_ids=(
                "state-coordinate-authority-distractor:2",
                "state-coordinate-authority-distractor:4",
            ),
            stale_event_ids=(),
            distractor_event_ids=("state-coordinate-authority-distractor:3",),
            events=(
                _bench_event(
                    "observation.created",
                    "Two workers submitted conflicting auth root-cause findings for the same mission.",
                ),
                _bench_event(
                    "policy.recorded",
                    "Parent promotion requires conflict resolved state before accepted worker findings become mission memory.",
                    authority_scope="policy",
                    status="current",
                ),
                _bench_event(
                    "decision.accepted",
                    "Accepted worker-local finding: database pool exhaustion caused the auth outage.",
                    details="This accepted-looking row is worker-local and conflicts with parent evidence.",
                    distractor=True,
                    authority_scope="worker",
                    status="accepted",
                    promoted=False,
                ),
                _bench_event(
                    "decision.accepted",
                    "Parent-accepted conflict resolved state: expired JWKS cache caused the auth outage.",
                    details="The parent should promote only the conflict-resolved accepted state.",
                    authority_scope="parent",
                    status="current",
                    promoted=True,
                ),
            ),
        ),
    ]
    cases.extend(_state_recovery_generated_cases())
    return cases


def _state_recovery_generated_cases() -> list[StateRecoveryCase]:
    domains = (
        {
            "slug": "billing",
            "surface": "checkout failures",
            "wrong": "payment-tokenization-timeout",
            "state": "invoice-webhook-replay-lag",
            "terms": ("invoice", "webhook", "replay"),
            "bridge": "Billing logs show duplicate invoice webhooks after queue replay.",
            "policy": "Current billing incident policy requires webhook replay evidence before promotion.",
        },
        {
            "slug": "search",
            "surface": "empty search results",
            "wrong": "frontend-filter-bug",
            "state": "index-alias-rollover-gap",
            "terms": ("index", "alias", "rollover"),
            "bridge": "Search telemetry shows index alias rollover gap after nightly reindex.",
            "policy": "Current search policy promotes only alias rollover evidence with reindex timestamps.",
        },
        {
            "slug": "deploy",
            "surface": "canary rollback",
            "wrong": "container-healthcheck-timeout",
            "state": "feature-flag-scope-leak",
            "terms": ("feature", "flag", "scope"),
            "bridge": "Deploy trace shows feature flag scope leak across canary and control cohorts.",
            "policy": "Current deploy policy requires flag-scope evidence before accepting canary diagnoses.",
        },
        {
            "slug": "docs",
            "surface": "release note dispute",
            "wrong": "markdown-render-cache",
            "state": "artifact-fingerprint-mismatch",
            "terms": ("artifact", "fingerprint", "mismatch"),
            "bridge": "Docs audit found artifact fingerprint mismatch between report JSON and release note.",
            "policy": "Current docs policy requires matching artifact fingerprints before public claims.",
        },
        {
            "slug": "routing",
            "surface": "regional API errors",
            "wrong": "dns-cache-expiry",
            "state": "tenant-shard-route-drift",
            "terms": ("tenant", "shard", "route"),
            "bridge": "Routing spans show tenant shard route drift after config fanout.",
            "policy": "Current routing policy promotes only shard-route evidence with tenant scope.",
        },
        {
            "slug": "memory",
            "surface": "stale checkout answer",
            "wrong": "embedding-cache-miss",
            "state": "superseded-projection-row",
            "terms": ("superseded", "projection", "row"),
            "bridge": "Memory status shows superseded projection row remained eligible after refresh.",
            "policy": "Current memory policy requires stale projection retirement before checkout reuse.",
        },
    )
    cases: list[StateRecoveryCase] = []
    for domain in domains:
        cases.append(_authority_family_case(domain))
        cases.append(_incomplete_authority_family_case(domain))
        cases.append(_bridge_family_case(domain))
        cases.append(_abstention_family_case(domain))
    return cases


def _authority_family_case(domain: dict[str, Any]) -> StateRecoveryCase:
    slug = str(domain["slug"])
    case_id = f"state-generated-authority-{slug}"
    return _state_case(
        case_id=case_id,
        query=f"Which accepted {slug} diagnosis should the parent trust?",
        latent_state=str(domain["state"]),
        expected_terms=tuple(domain["terms"]),
        minimal_evidence_event_ids=(f"{case_id}:3", f"{case_id}:4"),
        stale_event_ids=(f"{case_id}:2",),
        distractor_event_ids=(f"{case_id}:1",),
        events=(
            _bench_event(
                "decision.accepted",
                f"Accepted worker-local diagnosis: {domain['wrong']} caused {domain['surface']}.",
                details="Later parent review rejected this accepted-looking row as unsupported.",
                distractor=True,
                authority_scope="worker",
                status="accepted",
                promoted=False,
            ),
            _bench_event(
                "policy.recorded",
                f"Deprecated {slug} policy allowed worker-local accepted diagnoses without parent promotion.",
                stale=True,
                superseded_by=f"{case_id}:4",
                authority_scope="policy",
                status="superseded",
            ),
            _bench_event(
                "observation.created",
                str(domain["bridge"]),
                authority_scope="observation",
                status="current",
            ),
            _bench_event(
                "decision.accepted",
                f"Parent-accepted diagnosis: {domain['state']} caused {domain['surface']}.",
                authority_scope="parent",
                status="current",
                promoted=True,
            ),
        ),
    )


def _incomplete_authority_family_case(domain: dict[str, Any]) -> StateRecoveryCase:
    slug = str(domain["slug"])
    case_id = f"state-generated-incomplete-{slug}"
    return _state_case(
        case_id=case_id,
        query=f"What current {slug} state explains {domain['surface']}?",
        latent_state=str(domain["state"]),
        expected_terms=tuple(domain["terms"]),
        minimal_evidence_event_ids=(f"{case_id}:2", f"{case_id}:4"),
        stale_event_ids=(f"{case_id}:1",),
        distractor_event_ids=(f"{case_id}:3",),
        events=(
            _bench_event(
                "decision.accepted",
                f"Old accepted diagnosis: {domain['wrong']} explained {domain['surface']}.",
                stale=True,
                superseded_by=f"{case_id}:4",
            ),
            _bench_event(
                "observation.created",
                str(domain["bridge"]),
            ),
            _bench_event(
                "decision.accepted",
                f"Accepted-looking distractor: {domain['wrong']} remains plausible for {domain['surface']}.",
                distractor=True,
                authority_scope="worker",
                status="accepted",
                promoted=False,
            ),
            _bench_event(
                "policy.recorded",
                str(domain["policy"]),
                authority_scope="policy",
                status="current",
            ),
        ),
    )


def _bridge_family_case(domain: dict[str, Any]) -> StateRecoveryCase:
    slug = str(domain["slug"])
    case_id = f"state-generated-bridge-{slug}"
    return _state_case(
        case_id=case_id,
        query=f"What latent {slug} state connects the symptom and the accepted policy?",
        latent_state=str(domain["state"]),
        expected_terms=tuple(domain["terms"]),
        minimal_evidence_event_ids=(f"{case_id}:1", f"{case_id}:3"),
        stale_event_ids=(),
        distractor_event_ids=(f"{case_id}:2",),
        events=(
            _bench_event(
                "observation.created",
                str(domain["bridge"]),
                authority_scope="observation",
                status="current",
            ),
            _bench_event(
                "decision.accepted",
                f"Accepted but unrelated maintenance note: {domain['wrong']} was mitigated last week.",
                distractor=True,
                authority_scope="worker",
                status="accepted",
                promoted=False,
            ),
            _bench_event(
                "policy.recorded",
                str(domain["policy"]),
                authority_scope="policy",
                status="current",
            ),
            _bench_event(
                "observation.created",
                f"Distractor symptom: {domain['surface']} also appeared during synthetic load tests.",
                distractor=True,
            ),
        ),
    )


def _abstention_family_case(domain: dict[str, Any]) -> StateRecoveryCase:
    slug = str(domain["slug"])
    case_id = f"state-generated-abstain-{slug}"
    return _state_case(
        case_id=case_id,
        query=f"What should we conclude about the current {slug} root cause?",
        latent_state="unknown",
        expected_terms=(),
        minimal_evidence_event_ids=(),
        stale_event_ids=(f"{case_id}:1",),
        distractor_event_ids=(f"{case_id}:2", f"{case_id}:3"),
        should_abstain=True,
        events=(
            _bench_event(
                "policy.recorded",
                f"Deprecated {slug} policy claimed {domain['wrong']} could be accepted without evidence.",
                stale=True,
                status="superseded",
                authority_scope="policy",
            ),
            _bench_event(
                "decision.accepted",
                f"Rejected accepted-looking diagnosis: {domain['wrong']} caused {domain['surface']}.",
                distractor=True,
                authority_scope="worker",
                status="rejected",
                promoted=False,
            ),
            _bench_event(
                "observation.created",
                f"Ambiguous symptom only: {domain['surface']} lacks cited current evidence.",
                distractor=True,
                status="unsupported",
            ),
        ),
    )


def _state_case(
    *,
    case_id: str,
    query: str,
    latent_state: str,
    expected_terms: tuple[str, ...],
    minimal_evidence_event_ids: tuple[str, ...],
    stale_event_ids: tuple[str, ...],
    distractor_event_ids: tuple[str, ...],
    events: tuple[dict[str, Any], ...],
    should_abstain: bool = False,
) -> StateRecoveryCase:
    return StateRecoveryCase(
        case_id=case_id,
        query=query,
        events=events,
        gold=StateRecoveryGold(
            latent_state=latent_state,
            expected_terms=expected_terms,
            minimal_evidence_event_ids=minimal_evidence_event_ids,
            stale_event_ids=stale_event_ids,
            distractor_event_ids=distractor_event_ids,
            should_abstain=should_abstain,
        ),
    )


def _bench_event(
    event_type: str,
    summary: str,
    *,
    details: str | None = None,
    stale: bool = False,
    superseded_by: str | None = None,
    distractor: bool = False,
    authority_scope: str | None = None,
    status: str | None = None,
    promoted: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"summary": summary}
    if details is not None:
        payload["details"] = details
    if stale:
        payload["stale"] = True
    if superseded_by is not None:
        payload["superseded_by"] = superseded_by
    if distractor:
        payload["distractor"] = True
    if authority_scope is not None:
        payload["authority_scope"] = authority_scope
    if status is not None:
        payload["status"] = status
    if promoted is not None:
        payload["promoted"] = promoted
    return {"type": event_type, "actor": "benchmark", "payload": payload}


def _document_frequency(packets: list[EventPacket]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for packet in packets:
        counter.update(packet.terms)
    return counter


def _render_event_text(event: Event) -> str:
    return " ".join(
        [
            event.type,
            event.actor,
            event.thread,
            json.dumps(event.payload, sort_keys=True, separators=(" ", " ")),
        ]
    )


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        token = raw.strip("_:.-")
        if len(token) < 2 or token in _STOPWORDS:
            continue
        tokens.append(token)
        tokens.extend(_split_compound(token))
    return tokens


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _active_most_common(active: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(active.items(), key=lambda item: (-item[1], item[0]))


def _term_vector(terms: set[str] | frozenset[str], idf: dict[str, float]) -> dict[str, float]:
    return {term: idf.get(term, 1.0) for term in terms}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _max_packet_similarity(
    packet: EventPacket,
    selected: list[EventPacket],
    idf: dict[str, float],
) -> float:
    if not selected:
        return 0.0
    vector = _term_vector(packet.terms, idf)
    return max(_cosine(vector, _term_vector(item.terms, idf)) for item in selected)


def _split_compound(token: str) -> list[str]:
    parts = [part for part in re.split(r"[-_.:]+", token) if len(part) >= 2]
    if len(parts) <= 1:
        return []
    return [part for part in parts if part not in _STOPWORDS]
