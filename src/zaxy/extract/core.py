"""Hybrid extraction engine: rule-based + LLM fallback.

Typed Eventloom events are mapped deterministically to graph nodes and edges by
registered extractors. Unknown or unstructured events can use an explicit LLM
extractor implementation without adding a graph-memory abstraction dependency to
the core runtime.

This design cuts LLM extraction costs by 60–80%% for agents that emit
structured event types.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from zaxy.consolidation import (
    CONSOLIDATION_CANDIDATE_TYPES,
    validate_consolidation_candidate_id,
)
from zaxy.event import Event
from zaxy.neutral import (
    audit_ingestion_purpose_labels,
)


@dataclass(frozen=True)
class ExtractedEntity:
    """A node extracted from an event."""

    name: str
    entity_type: str
    observed_at: str  # ISO-8601 timestamp from the event
    summary: str | None = None
    embedding: list[float] | None = None
    properties: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExtractedEdge:
    """A relationship extracted from an event."""

    source: str  # source entity name
    target: str  # target entity name
    relation_type: str
    valid_from: str
    valid_to: str | None = None
    inferred: bool = False
    confidence: float = 1.0
    inference_method: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate edge audit metadata at construction time."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ExtractedEdge confidence must be between 0.0 and 1.0")
        if self.inferred and not self.inference_method:
            raise ValueError("Inferred ExtractedEdge values require inference_method")
        if not self.inferred and self.inference_method:
            raise ValueError("Deterministic ExtractedEdge values cannot set inference_method")


@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting a single event."""

    entities: list[ExtractedEntity]
    edges: list[ExtractedEdge]
    source_event_seq: int
    source_event_hash: str | None = None
    source_event_prev_hash: str | None = None
    source_event_type: str | None = None
    source_thread: str | None = None


_Registry = dict[str, Callable[[Event], ExtractionResult]]


_RULES: _Registry = {}


_CONSOLIDATION_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


_CONSOLIDATION_AUTHORITY_STATUS = "non_authoritative"


def register_extractor(event_type: str, fn: Callable[[Event], ExtractionResult]) -> None:
    """Register a rule-based extractor for an event type (non-decorator form).

    The imperative twin of :func:`register`. External plugins call this through
    :class:`zaxy.plugins.PluginAPI` to install extractors without decorator
    sugar. Registering the same event type again replaces the prior extractor,
    matching the decorator's last-writer-wins behavior.
    """
    _RULES[event_type] = fn


def register(event_type: str) -> Callable[[Callable[[Event], ExtractionResult]], Callable[[Event], ExtractionResult]]:
    """Decorator to register a rule-based extractor for an event type.

    Example::

        @register("user.preference_changed")
        def extract_pref(event: Event) -> ExtractionResult:
            ...
    """

    def decorator(fn: Callable[[Event], ExtractionResult]) -> Callable[[Event], ExtractionResult]:
        register_extractor(event_type, fn)
        return fn

    return decorator


def extract(event: Event) -> ExtractionResult:
    """Extract entities and edges from an event.

    Uses a registered rule if one exists, otherwise falls back to a
    generic identity extractor (preserves the event as a single entity).
    """
    if event.type in _RULES:
        return _with_source(event, _RULES[event.type](event))

    # Generic fallback: treat the event itself as an untyped entity node.
    # This avoids LLM costs for unknown events while still preserving them
    # in the graph for temporal replay.
    entity_name = f"event:{event.type}:{event.seq}"
    entity = ExtractedEntity(
        name=entity_name,
        entity_type="event",
        observed_at=event.timestamp,
        summary=_join_summary(
            f"{event.actor} emitted {event.type}",
            _fallback_payload_summary(event.payload),
        ),
        properties=_retention_properties(event.payload),
    )
    return _with_source(
        event,
        ExtractionResult(
            entities=[entity],
            edges=[],
            source_event_seq=event.seq,
        ),
    )


def _with_source(event: Event, result: ExtractionResult) -> ExtractionResult:
    """Attach stable Eventloom provenance to extraction results."""
    return replace(
        result,
        source_event_seq=event.seq,
        source_event_hash=event.hash,
        source_event_prev_hash=event.prev_hash,
        source_event_type=event.type,
        source_thread=event.thread,
    )


def _extract_skill_version_event(event: Event, *, status: str, relation_type: str) -> ExtractionResult:
    """Extract a skill lifecycle event that creates or updates a version node."""
    skill_id = _skill_id(event)
    version = _skill_version(event.payload)
    skill, version_entity = _skill_entities(event, skill_id=skill_id, version=version, status=status)
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    return ExtractionResult(
        entities=[skill, version_entity, actor],
        edges=[
            _skill_version_edge(skill, version_entity, event),
            ExtractedEdge(
                source=event.actor,
                target=version_entity.name,
                relation_type=relation_type,
                valid_from=event.timestamp,
            ),
        ],
        source_event_seq=event.seq,
    )


def _skill_id(event: Event) -> str:
    if skill_id := _optional_text(event.payload.get("skill_id")):
        return skill_id
    raise ValueError(f"{event.type} event {event.seq} missing required skill_id")


def _skill_version(payload: dict[str, Any]) -> str:
    return _optional_text(payload.get("version")) or "1"


def _skill_entities(
    event: Event,
    *,
    skill_id: str,
    version: str,
    status: str,
) -> tuple[ExtractedEntity, ExtractedEntity]:
    skill = ExtractedEntity(
        name=f"skill:{skill_id}",
        entity_type="skill",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("name")) or skill_id,
        properties={"skill_id": skill_id},
    )
    version_entity = ExtractedEntity(
        name=f"skill:{skill_id}:v{version}",
        entity_type="skill_version",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties=_compact_properties(
            {
                "skill_id": skill_id,
                "version": version,
                "procedure": _string_list(event.payload.get("procedure")),
                "applicability": _string_list(event.payload.get("applicability")),
                "citations": _string_list(event.payload.get("citations")),
                "failure_modes": _string_list(event.payload.get("failure_modes")),
                "rollback": _optional_text(event.payload.get("rollback")),
                "contradiction_reason": _optional_text(event.payload.get("contradiction_reason")),
                "evidence": _string_list(event.payload.get("evidence")),
                "status": status,
            }
        ),
    )
    return skill, version_entity


def _skill_version_edge(
    skill: ExtractedEntity,
    version: ExtractedEntity,
    event: Event,
) -> ExtractedEdge:
    return ExtractedEdge(
        source=skill.name,
        target=version.name,
        relation_type="has_version",
        valid_from=event.timestamp,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _optional_text(item))]


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _compact_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in properties.items()
        if value is not None and value != []
    }


def _event_ref(event: Event) -> str:
    return f"eventloom://{event.thread}/events/{event.seq}#{event.hash}"


def _neutral_audit_projection(
    event: Event,
    neutral_substrate_id: str,
) -> tuple[ExtractedEntity | None, ExtractedEdge | None]:
    audit = audit_ingestion_purpose_labels(event.payload, source_event_ref=_event_ref(event))
    if audit.safe:
        return None, None
    entity = ExtractedEntity(
        name=f"neutral-audit:{event.thread}:{event.seq}",
        entity_type="neutral_projection_audit",
        observed_at=event.timestamp,
        summary="Ingestion payload contains irreversible purpose-specific labels.",
        properties=audit.to_dict(),
    )
    edge = ExtractedEdge(
        source=entity.name,
        target=neutral_substrate_id,
        relation_type="flags_ingestion_purpose_label",
        valid_from=event.timestamp,
    )
    return entity, edge


def _required_causal_graph_relation_type(payload: dict[str, Any], *, event_seq: int) -> str:
    graph_relation_type = payload.get("graph_relation_type")
    if not isinstance(graph_relation_type, str) or not graph_relation_type.strip():
        raise ValueError(f"causal.edge.generated event {event_seq} missing required graph_relation_type")
    return graph_relation_type


def _required_consolidation_candidate_id(value: object) -> str:
    return validate_consolidation_candidate_id(value)


def _required_consolidation_candidate_type(value: object) -> str:
    if value not in CONSOLIDATION_CANDIDATE_TYPES:
        valid = ", ".join(sorted(CONSOLIDATION_CANDIDATE_TYPES))
        raise ValueError(f"candidate_type must be one of: {valid}")
    return str(value)


def _required_consolidation_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_consolidation_confidence(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("confidence must be a number between 0.0 and 1.0")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return confidence


def _required_consolidation_authority_status(value: object) -> str:
    if value != _CONSOLIDATION_AUTHORITY_STATUS:
        raise ValueError("authority_status must remain non_authoritative")
    return _CONSOLIDATION_AUTHORITY_STATUS


def _required_reasoning_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_reasoning_phase(value: object) -> str:
    phase = _required_reasoning_text(value, field="phase").casefold()
    if phase not in {"planning", "execution", "review", "reflection"}:
        raise ValueError("phase must be one of: execution, planning, reflection, review")
    return phase


def _snapshot_consolidation_source_events(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("source_events must be a non-empty list")
    if not value:
        raise ValueError("source_events must be non-empty")

    source_events = []
    for index, source_event in enumerate(value):
        if not isinstance(source_event, Mapping):
            raise ValueError(f"source_events[{index}] must be a mapping")
        seq = source_event.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
            raise ValueError(f"source_events[{index}].seq must be a positive integer")
        event_hash = source_event.get("hash")
        if not isinstance(event_hash, str) or _CONSOLIDATION_EVENT_HASH_RE.fullmatch(event_hash) is None:
            raise ValueError(
                f"source_events[{index}].hash must be exactly 64 lowercase hex characters"
            )
        source_events.append({"seq": seq, "hash": event_hash})
    return copy.deepcopy(source_events)


def _optional_text(value: object) -> str | None:
    """Return non-empty text for extracted summaries."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coordination_mission_id(event: Event) -> str:
    """Return the mission identifier for a coordination event."""
    return _optional_text(event.payload.get("mission_id") or event.payload.get("parent_session_id") or event.thread) or "default"


def _coordination_worker_id(event: Event) -> str:
    """Return the worker identifier for a coordination event."""
    return _optional_text(event.payload.get("worker_id") or event.payload.get("worker_session_id")) or "worker"


def _coordination_finding_id(event: Event) -> str:
    """Return the finding identifier for a coordination event."""
    return _optional_text(event.payload.get("finding_id")) or f"{_coordination_worker_id(event)}:finding:{event.seq}"


def _coordination_proof_row_id(proof_id: str, row: dict[str, Any]) -> str | None:
    """Return a stable graph id for one proof-packet diagnostic row."""
    row_identity = _optional_text(row.get("fact_id")) or _optional_text(row.get("source_group"))
    if row_identity is None:
        return None
    status = _optional_text(row.get("status")) or "non_authoritative"
    return f"{proof_id}:row:{status}:{row_identity}"


def _synthesis_candidate_id(scope: str, candidate: dict[str, Any]) -> str:
    """Return a stable graph id for a synthesis answer candidate."""
    answer_key = _optional_text(candidate.get("answer_key"))
    if answer_key:
        return f"{scope}:candidate:{answer_key}"
    rank = _optional_text(candidate.get("rank"))
    candidate_type = _optional_text(candidate.get("type"))
    answer = _optional_text(candidate.get("answer"))
    raw = json.dumps(
        {"rank": rank, "type": candidate_type, "answer": answer},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{scope}:candidate:{digest}"


def _synthesis_ledger_row_id(artifact_id: str, row: dict[str, Any]) -> str | None:
    """Return a stable graph id for an artifact ledger row."""
    row_identity = _optional_text(row.get("fact_id")) or _optional_text(row.get("source_group"))
    if row_identity:
        return f"{artifact_id}:ledger:{row_identity}"
    citation = _optional_text(row.get("citation"))
    if citation is None:
        return None
    digest = hashlib.sha256(citation.encode("utf-8")).hexdigest()[:16]
    return f"{artifact_id}:ledger:{digest}"


def _required_text(
    value: object,
    *,
    field: str,
    event_seq: int,
    event_type: str = "inference.edge.generated",
) -> str:
    """Return required text or raise a precise extraction error."""
    if text := _optional_text(value):
        return text
    raise ValueError(f"{event_type} event {event_seq} missing required {field}")


def _required_confidence(
    value: object,
    *,
    event_seq: int,
    event_type: str = "inference.edge.generated",
) -> float:
    """Return a required 0..1 confidence value for an inferred edge event."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{event_type} event {event_seq} missing required confidence")
    try:
        confidence = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{event_type} event {event_seq} has invalid confidence") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{event_type} event {event_seq} confidence must be between 0.0 and 1.0")
    return confidence


def _required_numeric_confidence(
    value: object,
    *,
    event_seq: int,
    event_type: str,
) -> float:
    """Return required numeric confidence without accepting string coercion."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{event_type} event {event_seq} missing required confidence")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{event_type} event {event_seq} confidence must be between 0.0 and 1.0")
    return confidence


def _required_strict_text(
    value: object,
    *,
    field: str,
    event_seq: int,
    event_type: str,
) -> str:
    """Return required text without accepting non-string coercion."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{event_type} event {event_seq} missing required {field}")
    return value.strip()


def _entity_reference(
    value: object,
    *,
    role: str,
    event_seq: int,
    observed_at: str,
) -> ExtractedEntity:
    """Return a source or target entity reference for an inferred-edge event."""
    if not isinstance(value, dict):
        raise ValueError(f"inference.edge.generated event {event_seq} missing {role} entity")
    return _entity_reference_from_mapping(
        value,
        role=role,
        event_seq=event_seq,
        observed_at=observed_at,
        event_type="inference.edge.generated",
    )


def _entity_reference_from_mapping(
    value: object,
    *,
    role: str,
    event_seq: int,
    observed_at: str,
    event_type: str,
) -> ExtractedEntity:
    """Return a graph entity reference from a validated payload mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{event_type} event {event_seq} missing {role} entity")
    name = _required_text(
        value.get("name"),
        field=f"{role}.name",
        event_seq=event_seq,
        event_type=event_type,
    )
    entity_type = _required_text(
        value.get("entity_type"),
        field=f"{role}.entity_type",
        event_seq=event_seq,
        event_type=event_type,
    )
    return ExtractedEntity(
        name=name,
        entity_type=entity_type,
        observed_at=observed_at,
        summary=_optional_text(value.get("summary")),
    )


def _explicit_task_id(payload: dict[str, Any]) -> str | None:
    """Return an explicitly supplied task identifier from common event taxonomies."""
    return _optional_text(payload.get("task_id") or payload.get("taskId"))


def _task_identity_properties(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return stable task identity fields that should stay prompt-visible."""
    task_id = _explicit_task_id(payload)
    if task_id is None:
        return None
    key = "task_id" if "task_id" in payload else "taskId"
    return {key: task_id}


def _with_explicit_task_observation(
    event: Event,
    entities: list[ExtractedEntity],
    edges: list[ExtractedEdge],
    *,
    target: str,
    relation_type: str,
) -> tuple[list[ExtractedEntity], list[ExtractedEdge]]:
    """Add a deterministic task-observation edge only when a task id is explicit."""
    task_id = _explicit_task_id(event.payload)
    if task_id is None:
        return entities, edges
    return (
        [
            *entities,
            ExtractedEntity(name=task_id, entity_type="task", observed_at=event.timestamp),
        ],
        [
            *edges,
            ExtractedEdge(
                source=task_id,
                target=target,
                relation_type=relation_type,
                valid_from=event.timestamp,
            ),
        ],
    )


def _preference_summary(key: object, value: object) -> str | None:
    """Return a compact preference summary."""
    key_text = _optional_text(key) or "preference"
    value_text = _optional_text(value)
    if value_text is None:
        return key_text
    return f"{key_text}={value_text}"


def _join_summary(*values: object) -> str | None:
    """Return a readable summary from scalar or list payload fields."""
    parts: list[str] = []
    for value in values:
        if isinstance(value, list):
            parts.extend(text for item in value if (text := _optional_text(item)))
            continue
        if text := _optional_text(value):
            parts.append(text)
    return " ".join(parts) or None


def _fallback_payload_summary(payload: dict[str, Any]) -> str | None:
    """Return safe top-level payload text for unknown event searchability."""
    parts: list[str] = []
    blocked_keys = {"secret", "token", "password", "api_key", "access_token"}
    for key in sorted(payload):
        if key.casefold() in blocked_keys:
            continue
        value = payload[key]
        if isinstance(value, list):
            parts.extend(text for item in value if (text := _optional_text(item)))
            continue
        if isinstance(value, dict):
            continue
        if text := _optional_text(value):
            parts.append(f"{key}={text}")
    return " ".join(parts) or None


def _retention_properties(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return safe retrieval-retention metadata from an event payload."""
    properties: dict[str, Any] = {}
    for key in (
        "authority",
        "authority_scope",
        "coordination_status",
        "finding_status",
        "promoted",
        "stale",
        "status",
        "superseded_by",
    ):
        value = payload.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            properties[key] = value
    if expires_at := _optional_text(payload.get("expires_at")):
        properties["expires_at"] = expires_at
    if last_reinforced_at := _optional_text(payload.get("last_reinforced_at")):
        properties["last_reinforced_at"] = last_reinforced_at
    importance = _bounded_float(payload.get("importance"))
    if importance is not None:
        properties["importance"] = importance
    if reinforcement_count := _optional_positive_int(payload.get("reinforcement_count")):
        properties["reinforcement_count"] = reinforcement_count
    return properties or None


def _feedback_purpose_properties(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return compact purpose and authority metadata from feedback payloads."""
    properties: dict[str, Any] = {}
    purpose = payload.get("purpose")
    if isinstance(purpose, dict):
        for source_key, target_key in (
            ("profile", "purpose_profile"),
            ("role", "purpose_role"),
            ("task", "purpose_task"),
            ("risk", "purpose_risk"),
            ("expected_action", "purpose_expected_action"),
            ("evidence_policy", "purpose_evidence_policy"),
            ("retention_policy", "purpose_retention_policy"),
        ):
            if value := _optional_text(purpose.get(source_key)):
                properties[target_key] = value
    elif value := _optional_text(purpose):
        properties["purpose_profile"] = value
    for key in (
        "authority",
        "authority_scope",
        "coordination_status",
        "finding_id",
        "mission_id",
        "outcome",
        "worker_id",
    ):
        if value := _optional_text(payload.get(key)):
            properties[key] = value
    stale = payload.get("stale")
    if isinstance(stale, bool):
        properties["stale"] = stale
    return properties or None


def _retrieval_salience_properties(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return ranking salience metadata for compact memory artifacts."""
    salience = _positive_float(
        payload.get("retrieval_salience")
        or payload.get("memory_salience")
        or payload.get("salience")
    )
    if salience is None and any(
        bool(payload.get(key))
        for key in (
            "salient_memory_turn",
            "longmemeval_salient_memory_turn",
        )
    ):
        salience = 4.0
    if salience is None:
        return None
    return {"retrieval_salience": salience}


def _longmemeval_document_properties(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return LongMemEval source metadata for benchmark document projections."""
    session_id = _optional_text(payload.get("longmemeval_session_id"))
    if session_id is None:
        return None
    properties: dict[str, Any] = {"longmemeval_session_id": session_id}
    if session_date := _optional_text(payload.get("longmemeval_session_date")):
        properties["longmemeval_session_date"] = session_date
    if chunk_index := _optional_positive_int(payload.get("longmemeval_chunk_index")):
        properties["longmemeval_chunk_index"] = chunk_index
    if chunk_count := _optional_positive_int(payload.get("longmemeval_chunk_count")):
        properties["longmemeval_chunk_count"] = chunk_count
    if turn_index := _optional_positive_int(payload.get("turn_index")):
        properties["turn_index"] = turn_index
    if role := _optional_text(payload.get("role")):
        properties["role"] = role
    if payload.get("longmemeval_salient_memory_turn") is not None:
        properties["longmemeval_salient_memory_turn"] = bool(payload.get("longmemeval_salient_memory_turn"))
    return properties


def _document_session_context(
    event: Event,
    *,
    document_name: str,
) -> tuple[list[ExtractedEntity], list[ExtractedEdge]]:
    """Link benchmark document chunks to their source conversation session."""
    session_id = _optional_text(event.payload.get("longmemeval_session_id"))
    if session_id is None:
        return [], []
    session_name = f"longmemeval:session:{session_id}"
    session = ExtractedEntity(
        name=session_name,
        entity_type="longmemeval_session",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("longmemeval_session_date")),
        properties={"longmemeval_session_id": session_id},
    )
    edge = ExtractedEdge(
        source=session_name,
        target=document_name,
        relation_type="has_document_chunk",
        valid_from=event.timestamp,
    )
    entities = [session]
    edges = [edge]
    salient = _longmemeval_salient_memory(event, session_name=session_name, document_name=document_name)
    if salient is not None:
        memory, memory_edges = salient
        entities.append(memory)
        edges.extend(memory_edges)
    return entities, edges


def _longmemeval_salient_memory(
    event: Event,
    *,
    session_name: str,
    document_name: str,
) -> tuple[ExtractedEntity, list[ExtractedEdge]] | None:
    """Promote LongMemEval salient turns into first-class memory nodes."""
    if event.payload.get("longmemeval_salient_memory_turn") is not True:
        return None
    session_id = _optional_text(event.payload.get("longmemeval_session_id"))
    turn_index = _optional_positive_int(event.payload.get("turn_index"))
    content = _optional_text(event.payload.get("content"))
    if session_id is None or turn_index is None or content is None:
        return None
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    end_line = _positive_int(event.payload.get("end_line"), default=start_line)
    path = _optional_text(event.payload.get("path")) or "document"
    memory_name = f"longmemeval:memory:{session_id}:{turn_index}"
    properties: dict[str, Any] = {
        "longmemeval_session_id": session_id,
        "turn_index": turn_index,
        "source_path": path,
        "source_start_line": start_line,
        "source_end_line": end_line,
    }
    if session_date := _optional_text(event.payload.get("longmemeval_session_date")):
        properties["longmemeval_session_date"] = session_date
    if role := _optional_text(event.payload.get("role")):
        properties["role"] = role
    memory = ExtractedEntity(
        name=memory_name,
        entity_type="longmemeval_memory",
        observed_at=event.timestamp,
        summary=content,
        properties=properties,
    )
    return memory, [
        ExtractedEdge(
            source=session_name,
            target=memory_name,
            relation_type="has_salient_memory",
            valid_from=event.timestamp,
        ),
        ExtractedEdge(
            source=memory_name,
            target=document_name,
            relation_type="derived_from_document",
            valid_from=event.timestamp,
        ),
    ]


def _refresh_transform_properties(payload: dict[str, Any]) -> dict[str, Any]:
    """Return context-refresh transform lineage metadata."""
    properties: dict[str, Any] = {}
    if transform_version := _optional_text(payload.get("transform_version")):
        properties["transform_version"] = transform_version
    if transform_id := _optional_text(payload.get("transform_id")):
        properties["transform_id"] = transform_id
    return properties


def _source_sha256_property(payload: dict[str, Any]) -> dict[str, str]:
    """Return source hash metadata when refresh provided it."""
    source_sha256 = _optional_text(payload.get("source_sha256"))
    return {"source_sha256": source_sha256} if source_sha256 else {}


def _merge_properties(*values: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for value in values:
        if value:
            merged.update(value)
    return merged or None


def _bounded_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _positive_float(value: object, *, maximum: float = 10.0) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return min(parsed, maximum)


def _optional_positive_int(value: object) -> int | None:
    parsed = _positive_int(value, default=0)
    return parsed or None


def _non_negative_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if not isinstance(value, int | str | bytes | bytearray):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if not isinstance(value, int | str | bytes | bytearray):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
