"""Internal guardrail scoring for beta.1 reasoning-loop primitives.

The guardrail checks product-contract properties only. It does not score task
answers, tune retrieval, or encode LongMemEval/LongMemBench-specific behavior.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

_OBSERVABLE_EVENT_TYPES = frozenset({"reasoning.primitive.called", "belief.update.proposed"})


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


def _ratio(values: Iterable[bool]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return round(sum(1 for value in rows if value) / len(rows), 3)


def _is_observable_call(row: Mapping[str, Any]) -> bool:
    return _text(row.get("event_type")) in _OBSERVABLE_EVENT_TYPES


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


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
