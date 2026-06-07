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
_OUTCOME_EXPLANATION_METHOD = "explicit_outcome_explanation_v1"
_OUTCOME_EXPLANATION_REASON = "outcome.explained explicitly cited Eventloom evidence"


def build_inferred_edge_events(event: Event) -> list[dict[str, Any]]:
    """Return inferred-edge Eventloom event specs generated from cited evidence."""
    if event.type == "task.completed":
        inferred = _task_completed_decision_inference(event)
        return [inferred] if inferred is not None else []
    if event.type == "inference.edge.contradicted":
        retracted = _inference_edge_retraction(event)
        return [retracted] if retracted is not None else []
    if event.type == "outcome.explained":
        causal = _outcome_explained_causal_edge(event)
        return [causal] if causal is not None else []
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


def _outcome_explained_causal_edge(event: Event) -> dict[str, Any] | None:
    """Build a causal edge from an explicit, cited outcome explanation."""
    from zaxy.causal import CAUSAL_RELATION_TYPES, build_causal_edge_event

    source = _causal_entity_ref(event.payload.get("cause"))
    target = _causal_entity_ref(event.payload.get("effect"))
    relation_type = _text(event.payload.get("relation_type"))
    confidence = _confidence(event.payload.get("confidence"))
    evidence_value = event.payload.get("evidence")
    if not isinstance(evidence_value, dict):
        return None
    source_event_seq = _causal_positive_int(evidence_value.get("source_event_seq"))
    source_event_hash = _event_hash(evidence_value.get("source_event_hash"))
    if not (
        source
        and target
        and relation_type in CAUSAL_RELATION_TYPES
        and confidence is not None
        and source_event_seq
        and source_event_hash
    ):
        return None

    evidence: dict[str, Any] = {
        "source_event_seq": source_event_seq,
        "source_event_hash": source_event_hash,
        "reason": _text(evidence_value.get("reason")) or _OUTCOME_EXPLANATION_REASON,
    }
    try:
        return build_causal_edge_event(
            actor="zaxy-causal",
            session_id=event.thread,
            source=source,
            target=target,
            relation_type=relation_type,
            confidence=confidence,
            method=_OUTCOME_EXPLANATION_METHOD,
            evidence=evidence,
        )
    except ValueError:
        return None


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _confidence(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _causal_positive_int(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if value > 0 else None


def _causal_entity_ref(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    name = _strict_text(value.get("name"))
    entity_type = _strict_text(value.get("entity_type"))
    if not name or not entity_type:
        return None
    return {"name": name, "entity_type": entity_type}


def _strict_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
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
