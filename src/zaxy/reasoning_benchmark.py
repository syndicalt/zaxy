"""Internal guardrail scoring for reasoning-loop primitives.

The guardrail checks product-contract properties only. It does not score task
answers, tune retrieval, or encode LongMemEval/LongMemBench-specific behavior.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_OBSERVABLE_EVENT_TYPES = frozenset({"reasoning.primitive.called", "belief.update.proposed"})
_METACOGNITION_EVENT_TYPES = frozenset(
    {
        "metacognition.unknown.recorded",
        "metacognition.confidence.assessed",
        "metacognition.conflict.clustered",
        "metacognition.reverify.requested",
    }
)
_PROCEDURAL_SKILL_EVENT_TYPES = frozenset(
    {
        "skill.proposed",
        "skill.validated",
        "skill.revised",
        "skill.deprecated",
        "skill.contradicted",
        "skill.applied",
        "skill.outcome_recorded",
    }
)
_OPEN_REVERIFY_STATUSES = frozenset({"open", "requested", "pending"})


def score_reasoning_guardrail(cases: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Score observable reasoning-loop primitive contract compliance.

    Each case is a primitive or proposal observation. The scorer intentionally
    uses transparent ratios so failures remain inspectable in beta gates.
    """
    rows = list(cases)
    case_count = len(rows)
    if case_count == 0:
        return {
            "case_count": 0,
            "observable_call": 0.0,
            "phase_match": 0.0,
            "citation_presence": 0.0,
            "authority_boundary": 0.0,
            "score": 0.0,
        }
    observable_call = _ratio(_is_observable_call(row) for row in rows)
    phase_match = _ratio(_phase_matches(row) for row in rows)
    citation_presence = _ratio(_has_citation(row) for row in rows)
    authority_boundary = _ratio(_preserves_authority_boundary(row) for row in rows)
    score = round(
        (observable_call + phase_match + citation_presence + authority_boundary) / 4,
        3,
    )
    return {
        "case_count": case_count,
        "observable_call": observable_call,
        "phase_match": phase_match,
        "citation_presence": citation_presence,
        "authority_boundary": authority_boundary,
        "score": score,
    }


def score_metacognition_guardrail(cases: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Score beta.2 metacognition and procedural-planning contract compliance.

    Rows are scored only through product contract fields: event type,
    reverify status, Eventloom citation, reasoning phase, and authority status.
    Free-text answers and benchmark metadata are intentionally irrelevant.
    """
    rows = list(cases)
    case_count = len(rows)
    if case_count == 0:
        return {
            "case_count": 0,
            "observable_metacognition": 0.0,
            "open_reverify_status": 0.0,
            "procedural_citation_presence": 0.0,
            "planning_phase_match": 0.0,
            "authority_boundary": 0.0,
            "score": 0.0,
        }

    metacognition_rows = [row for row in rows if not _is_procedural_row(row)]
    reverify_rows = [row for row in rows if _is_reverify_row(row)]
    procedural_rows = [row for row in rows if _is_procedural_row(row)]

    observable_metacognition = _ratio(_is_observable_metacognition(row) for row in metacognition_rows)
    open_reverify_status = _ratio(_has_open_reverify_status(row) for row in reverify_rows)
    procedural_citation_presence = _ratio(_has_citation(row) for row in procedural_rows)
    planning_phase_match = _ratio(_is_planning_phase(row) for row in procedural_rows)
    authority_boundary = _ratio(_preserves_beta2_authority_boundary(row) for row in rows)
    score = round(
        (
            observable_metacognition
            + open_reverify_status
            + procedural_citation_presence
            + planning_phase_match
            + authority_boundary
        )
        / 5,
        3,
    )
    return {
        "case_count": case_count,
        "observable_metacognition": observable_metacognition,
        "open_reverify_status": open_reverify_status,
        "procedural_citation_presence": procedural_citation_presence,
        "planning_phase_match": planning_phase_match,
        "authority_boundary": authority_boundary,
        "score": score,
    }


def _ratio(values: Iterable[bool]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return round(sum(1 for value in rows if value) / len(rows), 3)


def _is_observable_call(row: Mapping[str, Any]) -> bool:
    return _text(row.get("event_type")) in _OBSERVABLE_EVENT_TYPES


def _is_observable_metacognition(row: Mapping[str, Any]) -> bool:
    return _text(row.get("event_type")) in _METACOGNITION_EVENT_TYPES


def _is_reverify_row(row: Mapping[str, Any]) -> bool:
    return _text(row.get("event_type")) == "metacognition.reverify.requested"


def _has_open_reverify_status(row: Mapping[str, Any]) -> bool:
    if not _is_reverify_row(row):
        return False
    if row.get("open") is True:
        return True
    status = _text(row.get("reverify_status")) or _text(row.get("status"))
    return status in _OPEN_REVERIFY_STATUSES


def _is_procedural_row(row: Mapping[str, Any]) -> bool:
    event_type = _text(row.get("event_type"))
    if event_type in _PROCEDURAL_SKILL_EVENT_TYPES:
        return True
    if _text(row.get("primitive")) == "retrieve_similar_procedures":
        return True
    return bool(
        _text(row.get("procedural_bucket"))
        or _text(row.get("procedure_id"))
        or _text(row.get("skill_id"))
    )


def _is_planning_phase(row: Mapping[str, Any]) -> bool:
    return _text(row.get("phase")) == "planning"


def _phase_matches(row: Mapping[str, Any]) -> bool:
    expected = _text(row.get("expected_phase"))
    observed = _text(row.get("phase"))
    return bool(expected and observed and expected == observed)


def _has_citation(row: Mapping[str, Any]) -> bool:
    citation = _text(row.get("citation"))
    if citation.startswith("eventloom://"):
        return True
    citations = row.get("citations")
    if not isinstance(citations, list):
        return False
    return any(_text(item).startswith("eventloom://") for item in citations)


def _preserves_authority_boundary(row: Mapping[str, Any]) -> bool:
    if _text(row.get("authority_status")) != "non_authoritative":
        return False
    if _text(row.get("event_type")) == "belief.update.proposed":
        return _text(row.get("review_status")) in {"pending", ""}
    return True


def _preserves_beta2_authority_boundary(row: Mapping[str, Any]) -> bool:
    return _text(row.get("authority_status")) == "non_authoritative"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
