"""Alpha.2 consolidation guardrail scoring.

The guardrail is project-internal: it checks that generated consolidation
candidates preserve type intent, source-event citations, non-authoritative
boundaries, and explicit review gates.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zaxy.consolidation import (
    CONSOLIDATION_CANDIDATE_TYPES,
    CONSOLIDATION_INITIAL_REVIEW_STATUS,
    CONSOLIDATION_REVIEW_STATUSES,
)

_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_REVIEW_STATUSES = CONSOLIDATION_REVIEW_STATUSES | {CONSOLIDATION_INITIAL_REVIEW_STATUS}


@dataclass(frozen=True)
class ConsolidationGuardrailCase:
    """Expected source-backed shape for one generated candidate."""

    case_id: str
    candidate_type: str
    required_source_events: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if self.candidate_type not in CONSOLIDATION_CANDIDATE_TYPES:
            raise ValueError("candidate_type must be a supported consolidation type")
        if not self.required_source_events:
            raise ValueError("required_source_events must be non-empty")
        _source_event_refs(self.required_source_events)


def evaluate_consolidation_guardrail(
    case: ConsolidationGuardrailCase,
    candidate: Mapping[str, Any],
) -> dict[str, float | str]:
    """Score one candidate against alpha.2 review-gated invariants."""
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be a mapping")

    type_match = 1.0 if candidate.get("candidate_type") == case.candidate_type else 0.0
    source_event_fidelity = _source_event_fidelity(
        required=case.required_source_events,
        candidate_events=candidate.get("source_events"),
    )
    authority_boundary = 1.0 if candidate.get("authority_status") == "non_authoritative" else 0.0
    review_gate = 1.0 if candidate.get("review_status") in _VALID_REVIEW_STATUSES else 0.0
    score = (type_match + source_event_fidelity + authority_boundary + review_gate) / 4.0

    return {
        "case_id": case.case_id,
        "type_match": round(type_match, 4),
        "source_event_fidelity": round(source_event_fidelity, 4),
        "authority_boundary": round(authority_boundary, 4),
        "review_gate": round(review_gate, 4),
        "score": round(score, 4),
    }


def summarize_consolidation_guardrail(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Summarize guardrail rows with deterministic mean metrics."""
    if not rows:
        return {
            "case_count": 0,
            "mean_type_match": 0.0,
            "mean_source_event_fidelity": 0.0,
            "mean_authority_boundary": 0.0,
            "mean_review_gate": 0.0,
            "mean_score": 0.0,
        }
    return {
        "case_count": len(rows),
        "mean_type_match": _mean(rows, "type_match"),
        "mean_source_event_fidelity": _mean(rows, "source_event_fidelity"),
        "mean_authority_boundary": _mean(rows, "authority_boundary"),
        "mean_review_gate": _mean(rows, "review_gate"),
        "mean_score": _mean(rows, "score"),
    }


def _source_event_fidelity(
    *,
    required: Sequence[Mapping[str, Any]],
    candidate_events: object,
) -> float:
    if not isinstance(candidate_events, Sequence) or isinstance(candidate_events, str | bytes):
        return 0.0
    try:
        required_refs = set(_source_event_refs(required))
        candidate_refs = set(_source_event_refs(candidate_events))
    except ValueError:
        return 0.0
    if not required_refs:
        return 0.0
    return len(required_refs & candidate_refs) / len(required_refs)


def _source_event_refs(events: Sequence[Mapping[str, Any]]) -> list[tuple[int, str]]:
    refs: list[tuple[int, str]] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"source_events[{index}] must be a mapping")
        seq = event.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
            raise ValueError(f"source_events[{index}].seq must be a positive integer")
        event_hash = event.get("hash")
        if not isinstance(event_hash, str) or _EVENT_HASH_RE.fullmatch(event_hash) is None:
            raise ValueError(f"source_events[{index}].hash must be 64 lowercase hex")
        refs.append((seq, event_hash))
    return refs


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(key, 0.0)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{key} must be numeric in every guardrail row")
        total += float(value)
    return round(total / len(rows), 4)
