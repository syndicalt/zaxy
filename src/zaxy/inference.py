"""Conservative inferred-edge event producers.

The functions in this module generate Eventloom events that can later be
projected through the normal extraction path. Producers must be evidence-first:
they should return no event unless the source event contains explicit,
auditable citations.
"""

from __future__ import annotations

from typing import Any

from zaxy.event import Event

_HASH_LENGTH = 64
_TASK_DECISION_CONFIDENCE = 0.86
_TASK_DECISION_METHOD = "task_completed_decision_citation_v1"
_RETRACTION_METHOD = "contradicting_evidence_retraction_v1"


def build_inferred_edge_events(event: Event) -> list[dict[str, Any]]:
    """Return inferred-edge Eventloom event specs generated from cited evidence."""
    if event.type == "task.completed":
        inferred = _task_completed_decision_inference(event)
        return [inferred] if inferred is not None else []
    if event.type == "inference.edge.contradicted":
        retracted = _inference_edge_retraction(event)
        return [retracted] if retracted is not None else []
    return []


def _task_completed_decision_inference(event: Event) -> dict[str, Any] | None:
    """Infer a task-to-decision edge from an explicit cited decision reference."""
    task_id = _text(event.payload.get("taskId") or event.payload.get("task_id") or event.payload.get("task"))
    decision = _text(event.payload.get("decision") or event.payload.get("decision_name"))
    decision_event_seq = _positive_int(event.payload.get("decision_event_seq"))
    decision_event_hash = _event_hash(event.payload.get("decision_event_hash"))
    if not (task_id and decision and decision_event_seq and decision_event_hash):
        return None

    payload: dict[str, Any] = {
        "source": {
            "name": task_id,
            "entity_type": "task",
        },
        "target": {
            "name": decision,
            "entity_type": "decision",
        },
        "relation_type": "likely_implemented_decision",
        "confidence": _TASK_DECISION_CONFIDENCE,
        "inference_method": _TASK_DECISION_METHOD,
        "evidence": {
            "source_event_seq": event.seq,
            "source_event_hash": event.hash,
            "decision_event_seq": decision_event_seq,
            "decision_event_hash": decision_event_hash,
            "reason": "task.completed explicitly cited a decision Eventloom event",
        },
    }
    if summary := _text(event.payload.get("summary")):
        payload["source"]["summary"] = summary
    return {
        "event_type": "inference.edge.generated",
        "actor": "zaxy-inference",
        "payload": payload,
        "thread": event.thread,
    }


def _inference_edge_retraction(event: Event) -> dict[str, Any] | None:
    """Build a retraction event when a prior inference is contradicted."""
    source = _entity_ref(event.payload.get("source"))
    target = _entity_ref(event.payload.get("target"))
    relation_type = _text(event.payload.get("relation_type"))
    valid_from = _text(event.payload.get("valid_from"))
    original_event_seq = _positive_int(event.payload.get("original_event_seq"))
    original_event_hash = _event_hash(event.payload.get("original_event_hash"))
    reason = _text(event.payload.get("reason"))
    if not (
        source
        and target
        and relation_type
        and valid_from
        and original_event_seq
        and original_event_hash
        and reason
    ):
        return None
    return {
        "event_type": "inference.edge.retracted",
        "actor": "zaxy-inference",
        "payload": {
            "source": source,
            "target": target,
            "relation_type": relation_type,
            "valid_from": valid_from,
            "valid_to": event.timestamp,
            "confidence": 0.0,
            "inference_method": _RETRACTION_METHOD,
            "evidence": {
                "source_event_seq": event.seq,
                "source_event_hash": event.hash,
                "original_event_seq": original_event_seq,
                "original_event_hash": original_event_hash,
                "reason": reason,
            },
        },
        "thread": event.thread,
    }


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _event_hash(value: object) -> str | None:
    text = _text(value)
    if text is None or len(text) != _HASH_LENGTH:
        return None
    if any(char not in "0123456789abcdef" for char in text):
        return None
    return text


def _entity_ref(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    name = _text(value.get("name"))
    entity_type = _text(value.get("entity_type"))
    if not name or not entity_type:
        return None
    ref = {"name": name, "entity_type": entity_type}
    if summary := _text(value.get("summary")):
        ref["summary"] = summary
    return ref
