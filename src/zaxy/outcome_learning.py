"""Outcome-driven learning loop contracts (Zaxy 3 / I1).

Agents report outcomes on recalled memory; the loop reinforces salience and, on
failure/partial, proposes a **preventive rule** routed through the governed
evolution gate (op ``rule_generate``). Every event is non-authoritative, cited,
and replayable; nothing here mutates the log or auto-promotes outside the gate.

This is Zaxy's governed answer to outcome/feedback learning loops: agents get
better over time (failures become guardrails) while every learned rule is a
gated, auditable, reversible event with a citation back to the failure that
produced it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

_AUTHORITY_STATUS = "non_authoritative"

OUTCOMES: tuple[str, ...] = ("success", "failure", "partial")

OUTCOME_EVENT_TYPE = "memory.outcome.recorded"
RULE_PROPOSED_EVENT_TYPE = "memory.rule.proposed"
RULE_GENERATED_EVENT_TYPE = "memory.rule.generated"

#: Default confidence the loop assigns to a preventive rule by the outcome that
#: produced it. Failure is a strong signal (auto-applies under the default
#: ``auto_with_rollback`` tier and the default 0.85 threshold); partial is weaker
#: (held for review by default). Both ends are per-deployment tunable via
#: ``outcome_rule_confidence_failure`` / ``outcome_rule_confidence_partial``,
#: which is the knob that — together with ``evolution_confidence_threshold`` —
#: decides which outcomes may auto-apply at all.
DEFAULT_RULE_CONFIDENCE: Mapping[str, float] = {"failure": 0.9, "partial": 0.7}

#: Confidence used for an outcome with no entry in the confidence table.
FALLBACK_RULE_CONFIDENCE = 0.7

#: The observed outcome value the loop compares against the agent's prior to
#: derive prediction error (surprise): success and failure anchor the ends,
#: partial sits at the midpoint.
OUTCOME_ACTUAL: Mapping[str, float] = {"success": 1.0, "failure": 0.0, "partial": 0.5}

_HASH_RE = "0123456789abcdef"


def validate_outcome(outcome: object) -> str:
    """Return a valid outcome label or raise ValueError."""
    if outcome not in OUTCOMES:
        valid = ", ".join(OUTCOMES)
        raise ValueError(f"outcome must be one of: {valid}")
    return str(outcome)


def preventive_rule_confidence(
    outcome: str,
    explicit: float | None = None,
    *,
    defaults: Mapping[str, float] | None = None,
) -> float:
    """Return the confidence for a preventive rule from its outcome (or explicit).

    ``defaults`` overrides the built-in per-outcome table (see
    :func:`resolve_rule_confidence`); an explicit confidence always wins over both.
    """
    if explicit is not None:
        if (
            isinstance(explicit, bool)
            or not isinstance(explicit, int | float)
            or not 0.0 <= float(explicit) <= 1.0
        ):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(explicit)
    table = DEFAULT_RULE_CONFIDENCE if defaults is None else defaults
    return table.get(validate_outcome(outcome), FALLBACK_RULE_CONFIDENCE)


def resolve_rule_confidence(settings: Any) -> dict[str, float]:
    """Build the per-outcome preventive-rule confidence table from Settings.

    Reads ``outcome_rule_confidence_failure`` / ``outcome_rule_confidence_partial``
    defensively so a minimal or mocked settings object still yields the built-in
    defaults. Raises ``ValueError`` on a present-but-out-of-range value rather than
    silently falling back, so misconfiguration fails fast.
    """
    table = dict(DEFAULT_RULE_CONFIDENCE)
    for outcome, attribute in (
        ("failure", "outcome_rule_confidence_failure"),
        ("partial", "outcome_rule_confidence_partial"),
    ):
        value = getattr(settings, attribute, None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{attribute} must be a number between 0.0 and 1.0")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{attribute} must be between 0.0 and 1.0")
        table[outcome] = float(value)
    return table


def prediction_error(outcome: str, prior: float) -> float:
    """Return the prediction error (surprise) of an outcome against a prior.

    The prediction error is ``abs(actual - prior)`` where ``actual`` is the
    observed :data:`OUTCOME_ACTUAL` value for ``outcome`` and ``prior`` is the
    agent's reported confidence in ``[0.0, 1.0]`` that the recalled memory
    would lead to success. It is ``0`` when the result was fully anticipated
    and ``1`` when maximally surprising -- the signal the salience update is
    scaled by.
    """
    actual = OUTCOME_ACTUAL[validate_outcome(outcome)]
    return abs(actual - _validate_unit_interval(prior, field_name="prior"))


def build_outcome_event(
    *,
    actor: str,
    session_id: str,
    outcome: str,
    summary: str,
    target: Mapping[str, Any] | None = None,
    task_id: str | None = None,
    prior: float | None = None,
    prediction_error: float | None = None,
) -> dict[str, Any]:
    """Build a non-authoritative ``memory.outcome.recorded`` event spec."""
    _validate_non_empty_string(actor, field_name="actor")
    _validate_non_empty_string(session_id, field_name="session_id")
    outcome = validate_outcome(outcome)
    _validate_non_empty_string(summary, field_name="summary")
    payload: dict[str, Any] = {
        "outcome": outcome,
        "summary": summary,
        "authority_status": _AUTHORITY_STATUS,
    }
    if target is not None:
        payload["target"] = _snapshot_event_ref(target)
    if task_id is not None:
        _validate_non_empty_string(task_id, field_name="task_id")
        payload["task_id"] = task_id
    if prior is not None:
        payload["prior"] = _validate_unit_interval(prior, field_name="prior")
    if prediction_error is not None:
        payload["prediction_error"] = _validate_unit_interval(
            prediction_error, field_name="prediction_error"
        )
    return {
        "event_type": OUTCOME_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": session_id,
    }


def build_rule_event(
    *,
    actor: str,
    session_id: str,
    auto_applied: bool,
    rule: str,
    trigger: str,
    confidence: float,
    outcome: str,
    source_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a preventive-rule event spec (generated when auto-applied, else proposed)."""
    _validate_non_empty_string(actor, field_name="actor")
    _validate_non_empty_string(session_id, field_name="session_id")
    _validate_non_empty_string(rule, field_name="rule")
    _validate_non_empty_string(trigger, field_name="trigger")
    outcome = validate_outcome(outcome)
    cited = [_snapshot_event_ref(ref) for ref in source_events]
    if not cited:
        raise ValueError("source_events must cite at least one event")
    payload: dict[str, Any] = {
        "rule_id": _rule_id(rule, trigger, cited),
        "rule": rule,
        "trigger": trigger,
        "confidence": confidence,
        "outcome": outcome,
        "source_events": cited,
        "review_status": "active" if auto_applied else "pending",
        "authority_status": _AUTHORITY_STATUS,
    }
    return {
        "event_type": RULE_GENERATED_EVENT_TYPE if auto_applied else RULE_PROPOSED_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": session_id,
    }


def _rule_id(rule: str, trigger: str, source_events: Sequence[Mapping[str, Any]]) -> str:
    identity = {"rule": rule, "trigger": trigger, "source_events": list(source_events)}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"rule:{digest}"


def _snapshot_event_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(ref, Mapping):
        raise ValueError("event ref must be a mapping with seq and hash")
    seq = ref.get("seq")
    event_hash = ref.get("hash")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ValueError("event ref seq must be a positive integer")
    if not isinstance(event_hash, str) or len(event_hash) != 64 or any(c not in _HASH_RE for c in event_hash):
        raise ValueError("event ref hash must be a 64-character hex digest")
    return {"seq": seq, "hash": event_hash}


def _validate_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_unit_interval(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field_name} must be a number between 0.0 and 1.0")
    return float(value)
