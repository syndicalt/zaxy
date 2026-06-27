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
#: ``auto_with_rollback`` tier); partial is weaker (held for review by default).
DEFAULT_RULE_CONFIDENCE: Mapping[str, float] = {"failure": 0.9, "partial": 0.7}

_HASH_RE = "0123456789abcdef"


def validate_outcome(outcome: object) -> str:
    """Return a valid outcome label or raise ValueError."""
    if outcome not in OUTCOMES:
        valid = ", ".join(OUTCOMES)
        raise ValueError(f"outcome must be one of: {valid}")
    return str(outcome)


def preventive_rule_confidence(outcome: str, explicit: float | None = None) -> float:
    """Return the confidence for a preventive rule from its outcome (or explicit)."""
    if explicit is not None:
        if (
            isinstance(explicit, bool)
            or not isinstance(explicit, int | float)
            or not 0.0 <= float(explicit) <= 1.0
        ):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(explicit)
    return DEFAULT_RULE_CONFIDENCE.get(validate_outcome(outcome), 0.7)


def build_outcome_event(
    *,
    actor: str,
    session_id: str,
    outcome: str,
    summary: str,
    target: Mapping[str, Any] | None = None,
    task_id: str | None = None,
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
