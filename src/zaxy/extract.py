"""Hybrid extraction engine: rule-based + LLM fallback.

Typed Eventloom events are mapped deterministically to graph nodes and edges by
registered extractors. Unknown or unstructured events can use an explicit LLM
extractor implementation without adding a graph-memory abstraction dependency to
the core runtime.

This design cuts LLM extraction costs by 60–80%% for agents that emit
structured event types.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from zaxy.event import Event
from zaxy.neutral import (
    audit_ingestion_purpose_labels,
    neutral_document_record,
    neutral_transcript_record,
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


# Registry of rule-based extractors: event_type -> extractor function
_Registry = dict[str, Callable[[Event], ExtractionResult]]
_RULES: _Registry = {}


def register(event_type: str) -> Callable[[Callable[[Event], ExtractionResult]], Callable[[Event], ExtractionResult]]:
    """Decorator to register a rule-based extractor for an event type.

    Example::

        @register("user.preference_changed")
        def extract_pref(event: Event) -> ExtractionResult:
            ...
    """

    def decorator(fn: Callable[[Event], ExtractionResult]) -> Callable[[Event], ExtractionResult]:
        _RULES[event_type] = fn
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


# ------------------------------------------------------------------
# Built-in rule-based extractors
# ------------------------------------------------------------------

@register("goal.created")
def _extract_goal_created(event: Event) -> ExtractionResult:
    """Extract a goal entity from a goal.created event."""
    title = event.payload.get("title", "untitled")
    goal = ExtractedEntity(
        name=title,
        entity_type="goal",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("description")),
        properties=_retention_properties(event.payload),
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=title,
        relation_type="created_goal",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[goal, actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("task.proposed")
def _extract_task_proposed(event: Event) -> ExtractionResult:
    """Extract task and actor from task.proposed."""
    tid = event.payload.get("taskId", f"task_{event.seq}")
    goal_title = _optional_text(event.payload.get("goalTitle"))
    task = ExtractedEntity(
        name=tid,
        entity_type="task",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary") or event.payload.get("title")),
        properties=_merge_properties(
            _task_identity_properties(event.payload),
            _retention_properties(event.payload),
        ),
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=tid,
        relation_type="proposed_task",
        valid_from=event.timestamp,
    )
    entities = [task, actor]
    edges = [edge]
    if goal_title:
        entities.append(
            ExtractedEntity(
                name=goal_title,
                entity_type="goal",
                observed_at=event.timestamp,
            )
        )
        edges.append(
            ExtractedEdge(
                source=goal_title,
                target=tid,
                relation_type="has_task",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(
        entities=entities,
        edges=edges,
        source_event_seq=event.seq,
    )


@register("task.claimed")
def _extract_task_claimed(event: Event) -> ExtractionResult:
    """Link actor to claimed task."""
    tid = event.payload.get("taskId", "unknown")
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=tid,
        relation_type="claimed_task",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("task.completed")
def _extract_task_completed(event: Event) -> ExtractionResult:
    """Mark task as completed."""
    tid = _optional_text(event.payload.get("taskId") or event.payload.get("task")) or "unknown"
    summary = _optional_text(event.payload.get("summary"))
    task = ExtractedEntity(
        name=tid,
        entity_type="task",
        observed_at=event.timestamp,
        summary=summary,
        properties=_merge_properties(
            _task_identity_properties(event.payload),
            _retention_properties(event.payload),
        ),
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=tid,
        relation_type="completed_task",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[task, actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("decision.made")
def _extract_decision_made(event: Event) -> ExtractionResult:
    """Extract an agent decision with rationale into searchable graph context."""
    decision = _optional_text(event.payload.get("decision")) or f"decision:{event.seq}"
    entity = ExtractedEntity(
        name=decision,
        entity_type="decision",
        observed_at=event.timestamp,
        summary=_join_summary(
            event.payload.get("summary"),
            event.payload.get("rationale"),
            event.payload.get("alternatives_considered"),
        ),
        properties=_retention_properties(event.payload),
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=decision,
        relation_type="made_decision",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[entity, actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("context.policy")
def _extract_context_policy(event: Event) -> ExtractionResult:
    """Extract durable project/session guidance into searchable graph context."""
    source = _optional_text(event.payload.get("source")) or "context"
    project = _optional_text(event.payload.get("project"))
    name = f"{project}:{source}" if project else source
    entity = ExtractedEntity(
        name=name,
        entity_type="context_policy",
        observed_at=event.timestamp,
        summary=_join_summary(
            event.payload.get("status"),
            event.payload.get("instructions"),
        ),
        properties=_merge_properties({
            "source": source,
            "project": project,
        }, _retention_properties(event.payload)),
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=name,
        relation_type="set_context_policy",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[entity, actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("memory.reinforced")
def _extract_memory_reinforced(event: Event) -> ExtractionResult:
    """Extract reinforcement metadata for an existing memory entity."""
    entity_name = _optional_text(event.payload.get("entity_name")) or f"memory:{event.seq}"
    entity_type = _optional_text(event.payload.get("entity_type")) or "memory"
    properties = _merge_properties(
        _retention_properties(event.payload),
        _feedback_purpose_properties(event.payload),
        {
            "last_reinforced_at": event.timestamp,
            "reinforcement_count": _positive_int(event.payload.get("reinforcement_count"), default=1),
        },
    )
    entity = ExtractedEntity(
        name=entity_name,
        entity_type=entity_type,
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties=properties,
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=entity_name,
        relation_type="reinforced_memory",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[entity, actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("memory.evidence.reinforced")
@register("memory.evidence.excluded")
def _extract_memory_evidence_feedback(event: Event) -> ExtractionResult:
    """Extract row-level synthesis evidence feedback for cited facts."""
    fact_id = _optional_text(event.payload.get("fact_id"))
    source_group = _optional_text(event.payload.get("source_group"))
    citation = _optional_text(event.payload.get("citation"))
    entity_name = fact_id or source_group or citation or f"synthesis_evidence:{event.seq}"
    outcome = _optional_text(event.payload.get("outcome"))
    timestamp_key = "last_reinforced_at" if event.type == "memory.evidence.reinforced" else "last_excluded_at"
    feedback_properties = {
        key: value
        for key, value in {
            "outcome": outcome,
            "source_group": source_group,
            "citation": citation,
            timestamp_key: event.timestamp,
        }.items()
        if value is not None
    }
    properties = _merge_properties(
        _retention_properties(event.payload),
        feedback_properties,
    )
    entity = ExtractedEntity(
        name=entity_name,
        entity_type="synthesis_evidence",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("reason")) or _optional_text(event.payload.get("query")),
        properties=properties,
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    relation_type = (
        "reinforced_synthesis_evidence"
        if event.type == "memory.evidence.reinforced"
        else "excluded_synthesis_evidence"
    )
    edges = [
        ExtractedEdge(
            source=event.actor,
            target=entity_name,
            relation_type=relation_type,
            valid_from=event.timestamp,
        )
    ]
    entities = [entity, actor]
    if source_group:
        entities.append(
            ExtractedEntity(
                name=source_group,
                entity_type="source_group",
                observed_at=event.timestamp,
            )
        )
        edges.append(
            ExtractedEdge(
                source=entity_name,
                target=source_group,
                relation_type="cites_source_group",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(
        entities=entities,
        edges=edges,
        source_event_seq=event.seq,
    )


@register("memory.synthesis.artifact.created")
def _extract_memory_synthesis_artifact_created(event: Event) -> ExtractionResult:
    """Extract a deterministic Memory Checkout synthesis artifact."""
    artifact_id = _optional_text(event.payload.get("artifact_id")) or f"synthesis_artifact:{event.seq}"
    session_id = _optional_text(event.payload.get("session_id") or event.thread)
    candidates = _dict_list(event.payload.get("answer_candidates"))
    ledger_rows = _dict_list(event.payload.get("ledger_rows"))
    support_packet = event.payload.get("support_packet")
    source_groups = _string_list(support_packet.get("source_groups")) if isinstance(support_packet, dict) else []
    artifact = ExtractedEntity(
        name=artifact_id,
        entity_type="synthesis_artifact",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("query")),
        properties=_compact_properties(
            {
                "schema_version": _optional_text(event.payload.get("schema_version")),
                "session_id": session_id,
                "answer_candidate_count": len(candidates),
                "ledger_row_count": len(ledger_rows),
                "support_source_group_count": len(source_groups),
            }
        ),
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    entities = [artifact, actor]
    edges = [
        ExtractedEdge(
            source=event.actor,
            target=artifact_id,
            relation_type="created_synthesis_artifact",
            valid_from=event.timestamp,
        )
    ]
    for candidate in candidates:
        candidate_id = _synthesis_candidate_id(artifact_id, candidate)
        entities.append(
            ExtractedEntity(
                name=candidate_id,
                entity_type="synthesis_answer_candidate",
                observed_at=event.timestamp,
                summary=_optional_text(candidate.get("answer")),
                properties=_compact_properties(
                    {
                        "artifact_id": artifact_id,
                        "rank": candidate.get("rank"),
                        "type": _optional_text(candidate.get("type")),
                        "answer_key": _optional_text(candidate.get("answer_key")),
                        "confidence": _bounded_float(candidate.get("confidence")),
                    }
                ),
            )
        )
        edges.append(
            ExtractedEdge(
                source=artifact_id,
                target=candidate_id,
                relation_type="artifact_has_answer_candidate",
                valid_from=event.timestamp,
            )
        )
        for source_group in _string_list(candidate.get("support_source_ids")):
            entities.append(ExtractedEntity(name=source_group, entity_type="source_group", observed_at=event.timestamp))
            edges.append(
                ExtractedEdge(
                    source=candidate_id,
                    target=source_group,
                    relation_type="candidate_supported_by_source_group",
                    valid_from=event.timestamp,
                )
            )
    for row in ledger_rows:
        row_id = _synthesis_ledger_row_id(artifact_id, row)
        if row_id is None:
            continue
        entities.append(
            ExtractedEntity(
                name=row_id,
                entity_type="synthesis_ledger_row",
                observed_at=event.timestamp,
                summary=_join_summary(row.get("kind"), row.get("value"), row.get("label")),
                properties=_compact_properties(
                    {
                        "artifact_id": artifact_id,
                        "fact_id": _optional_text(row.get("fact_id")),
                        "source_group": _optional_text(row.get("source_group")),
                        "citation": _optional_text(row.get("citation")),
                        "include_reason": _optional_text(row.get("include_reason")),
                        "exclude_reason": _optional_text(row.get("exclude_reason")),
                        "confidence": _bounded_float(row.get("confidence")),
                    }
                ),
            )
        )
        edges.append(
            ExtractedEdge(
                source=artifact_id,
                target=row_id,
                relation_type="artifact_has_ledger_row",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


@register("memory.synthesis.used")
@register("memory.synthesis.rejected")
@register("memory.synthesis.corrected")
def _extract_memory_synthesis_outcome(event: Event) -> ExtractionResult:
    """Extract answer-candidate outcome feedback."""
    candidate = event.payload.get("answer_candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    candidate_id = _synthesis_candidate_id("candidate_feedback", candidate)
    outcome = _optional_text(event.payload.get("outcome")) or event.type.removeprefix("memory.synthesis.")
    entity = ExtractedEntity(
        name=candidate_id,
        entity_type="synthesis_answer_candidate",
        observed_at=event.timestamp,
        summary=_optional_text(candidate.get("answer")) or _optional_text(event.payload.get("reason")),
        properties=_compact_properties(
            {
                "outcome": outcome,
                "last_outcome_at": event.timestamp,
                "rank": candidate.get("rank"),
                "type": _optional_text(candidate.get("type")),
                "answer_key": _optional_text(candidate.get("answer_key")),
                "confidence": _bounded_float(candidate.get("confidence")),
            }
        ),
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edges = [
        ExtractedEdge(
            source=event.actor,
            target=candidate_id,
            relation_type=f"recorded_synthesis_{outcome}",
            valid_from=event.timestamp,
        )
    ]
    entities = [entity, actor]
    for source_group in _string_list(event.payload.get("support_source_ids")):
        entities.append(ExtractedEntity(name=source_group, entity_type="source_group", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=candidate_id,
                target=source_group,
                relation_type="candidate_supported_by_source_group",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


@register("skill.proposed")
def _extract_skill_proposed(event: Event) -> ExtractionResult:
    """Extract a proposed reusable procedural skill and its version."""
    return _extract_skill_version_event(event, status="proposed", relation_type="proposed_skill")


@register("skill.validated")
def _extract_skill_validated(event: Event) -> ExtractionResult:
    """Extract a validated reusable procedural skill version."""
    return _extract_skill_version_event(event, status="validated", relation_type="validated_skill")


@register("skill.revised")
def _extract_skill_revised(event: Event) -> ExtractionResult:
    """Extract a revised skill version while preserving earlier versions."""
    return _extract_skill_version_event(event, status="revised", relation_type="revised_skill")


@register("skill.deprecated")
def _extract_skill_deprecated(event: Event) -> ExtractionResult:
    """Extract skill deprecation metadata without deleting old versions."""
    return _extract_skill_version_event(event, status="deprecated", relation_type="deprecated_skill")


@register("skill.contradicted")
def _extract_skill_contradicted(event: Event) -> ExtractionResult:
    """Extract a contradicted skill version for audit and rollback."""
    return _extract_skill_version_event(event, status="contradicted", relation_type="contradicted_skill")


@register("skill.applied")
def _extract_skill_applied(event: Event) -> ExtractionResult:
    """Extract a task application of a skill version."""
    skill_id = _skill_id(event)
    version = _skill_version(event.payload)
    skill, version_entity = _skill_entities(event, skill_id=skill_id, version=version, status="applied")
    application_name = f"skill:{skill_id}:v{version}:application:{event.seq}"
    application = ExtractedEntity(
        name=application_name,
        entity_type="skill_application",
        observed_at=event.timestamp,
        summary=_join_summary(event.payload.get("task"), event.payload.get("summary")),
        properties=_compact_properties(
            {
                "skill_id": skill_id,
                "version": version,
                "task": _optional_text(event.payload.get("task")),
                "task_id": _optional_text(event.payload.get("task_id") or event.payload.get("taskId")),
                "context": _optional_text(event.payload.get("context")),
            }
        ),
    )
    return ExtractionResult(
        entities=[skill, version_entity, application],
        edges=[
            _skill_version_edge(skill, version_entity, event),
            ExtractedEdge(
                source=version_entity.name,
                target=application.name,
                relation_type="applied_to_task",
                valid_from=event.timestamp,
            ),
        ],
        source_event_seq=event.seq,
    )


@register("skill.outcome_recorded")
def _extract_skill_outcome_recorded(event: Event) -> ExtractionResult:
    """Extract outcome metrics for an applied skill version."""
    skill_id = _skill_id(event)
    version = _skill_version(event.payload)
    skill, version_entity = _skill_entities(event, skill_id=skill_id, version=version, status="outcome_recorded")
    outcome_name = f"skill:{skill_id}:v{version}:outcome:{event.seq}"
    outcome = ExtractedEntity(
        name=outcome_name,
        entity_type="skill_outcome",
        observed_at=event.timestamp,
        summary=_join_summary(event.payload.get("task"), event.payload.get("feedback")),
        properties=_compact_properties(
            {
                "skill_id": skill_id,
                "version": version,
                "task": _optional_text(event.payload.get("task")),
                "success_score": _bounded_float(event.payload.get("success_score")),
                "feedback": _optional_text(event.payload.get("feedback")),
                "evidence": _string_list(event.payload.get("evidence")),
            }
        ),
    )
    return ExtractionResult(
        entities=[skill, version_entity, outcome],
        edges=[
            _skill_version_edge(skill, version_entity, event),
            ExtractedEdge(
                source=version_entity.name,
                target=outcome.name,
                relation_type="recorded_outcome",
                valid_from=event.timestamp,
            ),
        ],
        source_event_seq=event.seq,
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


@register("hook.checkpoint")
def _extract_hook_checkpoint(event: Event) -> ExtractionResult:
    """Extract a searchable observer checkpoint."""
    session_id = _optional_text(event.thread) or _optional_text(event.payload.get("session_id")) or "default"
    source = _optional_text(event.payload.get("source")) or "hook"
    reason = _optional_text(event.payload.get("reason")) or "checkpoint"
    turn_count = _positive_int(event.payload.get("turn_count"), default=0)
    workspace = _optional_text(event.payload.get("workspace"))
    checkpoint = ExtractedEntity(
        name=f"{session_id}:checkpoint:{event.seq}",
        entity_type="hook_checkpoint",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties={
            "session_id": session_id,
            "source": source,
            "reason": reason,
            "turn_count": turn_count,
            "workspace": workspace,
        },
    )
    session = ExtractedEntity(
        name=session_id,
        entity_type="session",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=session.name,
        target=checkpoint.name,
        relation_type="recorded_checkpoint",
        valid_from=event.timestamp,
    )
    entities, edges = _with_explicit_task_observation(
        event,
        [session, checkpoint],
        [edge],
        target=checkpoint.name,
        relation_type="has_checkpoint",
    )
    return ExtractionResult(
        entities=entities,
        edges=edges,
        source_event_seq=event.seq,
    )


@register("issue.diagnosed")
def _extract_issue_diagnosed(event: Event) -> ExtractionResult:
    """Extract a diagnosed issue with root cause and supporting evidence."""
    issue = _optional_text(event.payload.get("issue")) or f"issue:{event.seq}"
    entity = ExtractedEntity(
        name=issue,
        entity_type="issue",
        observed_at=event.timestamp,
        summary=_join_summary(
            event.payload.get("root_cause"),
            event.payload.get("evidence"),
            event.payload.get("fix"),
        ),
        properties={"status": "diagnosed"},
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=issue,
        relation_type="diagnosed_issue",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[entity, actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("verification.recorded")
def _extract_verification_recorded(event: Event) -> ExtractionResult:
    """Extract verification evidence such as test, lint, build, or smoke checks."""
    command = _optional_text(event.payload.get("command")) or f"verification:{event.seq}"
    outcome = _optional_text(event.payload.get("outcome")) or "unknown"
    entity = ExtractedEntity(
        name=command,
        entity_type="verification",
        observed_at=event.timestamp,
        summary=_join_summary(
            outcome,
            event.payload.get("summary"),
            event.payload.get("evidence"),
        ),
        properties={"outcome": outcome},
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=command,
        relation_type="recorded_verification",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[entity, actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("handoff.created")
def _extract_handoff_created(event: Event) -> ExtractionResult:
    """Extract a handoff summary with next steps and residual risks."""
    name = _optional_text(event.payload.get("title")) or f"handoff:{event.seq}"
    entity = ExtractedEntity(
        name=name,
        entity_type="handoff",
        observed_at=event.timestamp,
        summary=_join_summary(
            event.payload.get("summary"),
            event.payload.get("next_steps"),
            event.payload.get("risks"),
        ),
        properties={"status": "created"},
    )
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=name,
        relation_type="created_handoff",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[entity, actor],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("coordination.mission.created")
def _extract_coordination_mission_created(event: Event) -> ExtractionResult:
    """Extract a high-level coordinator mission."""
    mission_id = _coordination_mission_id(event)
    status = _optional_text(event.payload.get("status")) or "active"
    mission = ExtractedEntity(
        name=mission_id,
        entity_type="mission",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("objective")),
        properties=_compact_properties({"status": status}),
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=mission_id,
        relation_type="started_mission",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, actor], edges=[edge], source_event_seq=event.seq)


@register("coordination.worker.created")
def _extract_coordination_worker_created(event: Event) -> ExtractionResult:
    """Extract a registered worker under a mission."""
    mission_id = _coordination_mission_id(event)
    worker_id = _coordination_worker_id(event)
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    worker = ExtractedEntity(
        name=worker_id,
        entity_type="worker",
        observed_at=event.timestamp,
        properties=_compact_properties(
            {
                "mission_id": mission_id,
                "status": _optional_text(event.payload.get("status")) or "active",
            }
        ),
    )
    edge = ExtractedEdge(
        source=mission_id,
        target=worker_id,
        relation_type="mission_has_worker",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, worker], edges=[edge], source_event_seq=event.seq)


@register("coordination.assignment.created")
def _extract_coordination_assignment_created(event: Event) -> ExtractionResult:
    """Extract a worker assignment."""
    mission_id = _coordination_mission_id(event)
    worker_id = _coordination_worker_id(event)
    assignment_id = (
        _optional_text(event.payload.get("assignment_id"))
        or f"{mission_id}:{worker_id}:assignment:{event.seq}"
    )
    assignment = ExtractedEntity(
        name=assignment_id,
        entity_type="assignment",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("assignment")),
        properties=_compact_properties(
            {
                "status": _optional_text(event.payload.get("status")) or "assigned",
                "mission_id": mission_id,
                "worker_id": worker_id,
            }
        ),
    )
    worker = ExtractedEntity(name=worker_id, entity_type="worker", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=worker_id,
        target=assignment_id,
        relation_type="worker_has_assignment",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[worker, assignment], edges=[edge], source_event_seq=event.seq)


@register("coordination.finding.reported")
def _extract_coordination_finding_reported(event: Event) -> ExtractionResult:
    """Extract a worker-local finding with evidence metadata."""
    mission_id = _coordination_mission_id(event)
    worker_id = _coordination_worker_id(event)
    finding_id = _coordination_finding_id(event)
    evidence = event.payload.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    finding = ExtractedEntity(
        name=finding_id,
        entity_type="finding",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties=_compact_properties(
            {
                "status": _optional_text(event.payload.get("status")) or "pending",
                "mission_id": mission_id,
                "worker_id": worker_id,
                "evidence_count": evidence_count,
                "confidence": _bounded_float(event.payload.get("confidence")),
                "claim_key": _optional_text(event.payload.get("claim_key")),
                "claim_value": _optional_text(event.payload.get("claim_value")),
            }
        ),
    )
    worker = ExtractedEntity(name=worker_id, entity_type="worker", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=worker_id,
        target=finding_id,
        relation_type="worker_reported_finding",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[worker, finding], edges=[edge], source_event_seq=event.seq)


@register("coordination.finding.reviewed")
def _extract_coordination_finding_reviewed(event: Event) -> ExtractionResult:
    """Extract a coordinator review of a worker finding."""
    mission_id = _coordination_mission_id(event)
    worker_id = _coordination_worker_id(event)
    finding_id = _coordination_finding_id(event)
    status = _optional_text(event.payload.get("status")) or "reviewed"
    review_id = f"{mission_id}:{finding_id}:review:{event.seq}"
    review = ExtractedEntity(
        name=review_id,
        entity_type="finding_review",
        observed_at=event.timestamp,
        summary=_join_summary(status, event.payload.get("rationale")),
        properties=_compact_properties({"status": status, "mission_id": mission_id, "worker_id": worker_id}),
    )
    finding = ExtractedEntity(name=finding_id, entity_type="finding", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=finding_id,
        relation_type="coordinator_reviewed_finding",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[finding, review], edges=[edge], source_event_seq=event.seq)


@register("coordination.finding.promoted")
def _extract_coordination_finding_promoted(event: Event) -> ExtractionResult:
    """Extract accepted parent state promoted from a worker finding."""
    mission_id = _coordination_mission_id(event)
    finding_id = _coordination_finding_id(event)
    promotion_id = f"{mission_id}:{finding_id}:promotion:{event.seq}"
    promotion = ExtractedEntity(
        name=promotion_id,
        entity_type="promotion",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties=_compact_properties(
            {
                "status": _optional_text(event.payload.get("status")) or "accepted",
                "mission_id": mission_id,
                "worker_id": _optional_text(event.payload.get("worker_id")),
                "finding_id": finding_id,
                "source_event_seq": event.payload.get("source_event_seq"),
                "source_event_hash": _optional_text(event.payload.get("source_event_hash")),
                "claim_key": _optional_text(event.payload.get("claim_key")),
                "claim_value": _optional_text(event.payload.get("claim_value")),
            }
        ),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=finding_id,
        target=promotion_id,
        relation_type="finding_promoted_to_parent",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, promotion], edges=[edge], source_event_seq=event.seq)


@register("coordination.conflict.detected")
def _extract_coordination_conflict_detected(event: Event) -> ExtractionResult:
    """Extract a deterministic conflict between worker findings."""
    mission_id = _coordination_mission_id(event)
    conflict_id = _optional_text(event.payload.get("conflict_id")) or f"{mission_id}:conflict:{event.seq}"
    finding_ids = _string_list(event.payload.get("finding_ids"))
    conflict = ExtractedEntity(
        name=conflict_id,
        entity_type="conflict",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties=_compact_properties(
            {
                "mission_id": mission_id,
                "claim_key": _optional_text(event.payload.get("claim_key")),
                "conflict_type": _optional_text(event.payload.get("conflict_type")) or "exact_claim",
                "reason": _optional_text(event.payload.get("reason")),
                "source_reference": _optional_text(event.payload.get("source_reference")),
                "finding_count": len(finding_ids),
            }
        ),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    entities = [mission, conflict]
    edges = [
        ExtractedEdge(
            source=mission_id,
            target=conflict_id,
            relation_type="mission_has_conflict",
            valid_from=event.timestamp,
        )
    ]
    for finding_id in finding_ids:
        entities.append(ExtractedEntity(name=finding_id, entity_type="finding", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=finding_id,
                target=conflict_id,
                relation_type="finding_conflicts_with",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


@register("coordination.decision.recorded")
def _extract_coordination_decision_recorded(event: Event) -> ExtractionResult:
    """Extract a coordinator decision."""
    mission_id = _coordination_mission_id(event)
    decision_id = _optional_text(event.payload.get("decision_id")) or f"{mission_id}:decision:{event.seq}"
    decision = ExtractedEntity(
        name=decision_id,
        entity_type="decision",
        observed_at=event.timestamp,
        summary=_join_summary(event.payload.get("decision"), event.payload.get("rationale")),
        properties=_compact_properties({"mission_id": mission_id}),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=mission_id,
        target=decision_id,
        relation_type="mission_has_decision",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, decision], edges=[edge], source_event_seq=event.seq)


@register("coordination.handoff.created")
def _extract_coordination_handoff_created(event: Event) -> ExtractionResult:
    """Extract a final mission handoff."""
    mission_id = _coordination_mission_id(event)
    handoff_id = _optional_text(event.payload.get("handoff_id")) or f"{mission_id}:handoff:{event.seq}"
    handoff = ExtractedEntity(
        name=handoff_id,
        entity_type="handoff",
        observed_at=event.timestamp,
        summary=_join_summary(event.payload.get("summary"), event.payload.get("next_steps"), event.payload.get("risks")),
        properties=_compact_properties({"mission_id": mission_id, "status": "created"}),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=mission_id,
        target=handoff_id,
        relation_type="mission_has_handoff",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, handoff], edges=[edge], source_event_seq=event.seq)


@register("coordination.proof_packet.created")
def _extract_coordination_proof_packet_created(event: Event) -> ExtractionResult:
    """Extract a mission-scoped synthesis proof packet for coordinator memory."""
    mission_id = _coordination_mission_id(event)
    artifact_id = _optional_text(event.payload.get("artifact_id"))
    proof_id = artifact_id or f"{mission_id}:proof_packet:{event.seq}"
    accepted_ids = _string_list(event.payload.get("accepted_finding_ids"))
    diagnostic_pending_ids = _string_list(event.payload.get("diagnostic_pending_ids"))
    conflict_ids = _string_list(event.payload.get("conflict_ids"))
    non_authoritative_rows = _dict_list(event.payload.get("non_authoritative_rows"))
    excluded_row_reasons = _dict_list(event.payload.get("excluded_row_reasons"))
    handoff_ref = event.payload.get("handoff_event_ref")
    handoff_id = _optional_text(handoff_ref.get("handoff_id")) if isinstance(handoff_ref, dict) else None
    proof = ExtractedEntity(
        name=proof_id,
        entity_type="coordination_proof_packet",
        observed_at=event.timestamp,
        summary=_join_summary(event.payload.get("query"), event.payload.get("decision_scope")),
        properties=_compact_properties(
            {
                "mission_id": mission_id,
                "schema_version": _optional_text(event.payload.get("schema_version")),
                "artifact_id": artifact_id,
                "decision_scope": _optional_text(event.payload.get("decision_scope")),
                "authority_scope": _optional_text(event.payload.get("authority_scope")),
                "accepted_finding_count": len(accepted_ids),
                "diagnostic_pending_count": len(diagnostic_pending_ids),
                "conflict_count": len(conflict_ids),
                "non_authoritative_row_count": len(non_authoritative_rows),
                "excluded_row_reason_count": len(excluded_row_reasons),
                "handoff_id": handoff_id,
            }
        ),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    entities = [mission, proof]
    edges = [
        ExtractedEdge(
            source=mission_id,
            target=proof_id,
            relation_type="mission_has_proof_packet",
            valid_from=event.timestamp,
        )
    ]
    if artifact_id:
        entities.append(ExtractedEntity(name=artifact_id, entity_type="synthesis_artifact", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=artifact_id,
                relation_type="proof_links_synthesis_artifact",
                valid_from=event.timestamp,
            )
        )
    for finding_id in accepted_ids:
        entities.append(ExtractedEntity(name=finding_id, entity_type="finding", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=finding_id,
                relation_type="proof_uses_accepted_finding",
                valid_from=event.timestamp,
            )
        )
    for row in non_authoritative_rows:
        row_id = _coordination_proof_row_id(proof_id, row)
        if row_id is None:
            continue
        entities.append(
            ExtractedEntity(
                name=row_id,
                entity_type="coordination_non_authoritative_row",
                observed_at=event.timestamp,
                summary=_join_summary(row.get("status"), row.get("include_reason"), row.get("exclude_reason")),
                properties=_compact_properties(
                    {
                        "mission_id": mission_id,
                        "proof_packet_id": proof_id,
                        "source_group": _optional_text(row.get("source_group")),
                        "fact_id": _optional_text(row.get("fact_id")),
                        "status": _optional_text(row.get("status")),
                        "include_reason": _optional_text(row.get("include_reason")),
                        "exclude_reason": _optional_text(row.get("exclude_reason")),
                    }
                ),
            )
        )
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=row_id,
                relation_type="proof_excludes_non_authoritative_row",
                valid_from=event.timestamp,
            )
        )
    for conflict_id in conflict_ids:
        entities.append(ExtractedEntity(name=conflict_id, entity_type="conflict", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=conflict_id,
                relation_type="proof_diagnoses_conflict",
                valid_from=event.timestamp,
            )
        )
    if handoff_id:
        entities.append(ExtractedEntity(name=handoff_id, entity_type="handoff", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=handoff_id,
                relation_type="proof_binds_handoff",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


@register("user.preference_changed")
def _extract_preference_changed(event: Event) -> ExtractionResult:
    """Extract user preference as a persistent entity property."""
    user_id = event.payload.get("userId", event.actor)
    key = event.payload.get("key", "preference")
    user = ExtractedEntity(
        name=user_id,
        entity_type="user",
        observed_at=event.timestamp,
    )
    pref = ExtractedEntity(
        name=f"{user_id}:{key}",
        entity_type="preference",
        observed_at=event.timestamp,
        summary=_preference_summary(key, event.payload.get("value")),
    )
    edge = ExtractedEdge(
        source=user_id,
        target=f"{user_id}:{key}",
        relation_type=f"has_{key}",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[user, pref],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("document.indexed")
def _extract_document_indexed(event: Event) -> ExtractionResult:
    """Extract a cited document chunk from filesystem ingestion."""
    path = _optional_text(event.payload.get("path")) or "document"
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    end_line = _positive_int(event.payload.get("end_line"), default=start_line)
    content = _optional_text(event.payload.get("content")) or ""
    sha256 = _optional_text(event.payload.get("sha256"))
    document_name = f"{path}:{start_line}-{end_line}"
    entity = ExtractedEntity(
        name=document_name,
        entity_type="document",
        observed_at=event.timestamp,
        summary=content,
        properties=_merge_properties(
            {
                "source_path": path,
                "source_start_line": start_line,
                "source_end_line": end_line,
                "source_sha256": sha256,
                **_refresh_transform_properties(event.payload),
            },
            _longmemeval_document_properties(event.payload),
            _retrieval_salience_properties(event.payload),
        )
        or {},
    )
    neutral = neutral_document_record(
        actor=event.actor,
        timestamp=event.timestamp,
        path=path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        source_event_ref=_event_ref(event),
        permission_scope=_optional_text(event.payload.get("permission_scope")),
        uncertainty=_optional_text(event.payload.get("uncertainty")),
        candidate_claim=_optional_text(event.payload.get("candidate_claim")),
    )
    neutral_entity = ExtractedEntity(
        name=neutral.substrate_id,
        entity_type="neutral_substrate",
        observed_at=event.timestamp,
        summary=neutral.quote,
        properties=neutral.to_properties(),
    )
    neutral_edge = ExtractedEdge(
        source=neutral.substrate_id,
        target=document_name,
        relation_type="neutral_substrate_cites_source",
        valid_from=event.timestamp,
    )
    audit_entity, audit_edge = _neutral_audit_projection(event, neutral.substrate_id)
    entities, edges = _document_session_context(
        event,
        document_name=document_name,
    )
    return ExtractionResult(
        entities=[entity, neutral_entity, *([audit_entity] if audit_entity is not None else []), *entities],
        edges=[neutral_edge, *([audit_edge] if audit_edge is not None else []), *edges],
        source_event_seq=event.seq,
    )


@register("code.file.indexed")
def _extract_code_file_indexed(event: Event) -> ExtractionResult:
    """Extract a code file inventory node from codebase indexing."""
    path = _optional_text(event.payload.get("path")) or "code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    sha256 = _optional_text(event.payload.get("sha256"))
    byte_count = _positive_int(event.payload.get("bytes"), default=0)
    line_count = _positive_int(event.payload.get("lines"), default=0)
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    code_file = ExtractedEntity(
        name=path,
        entity_type="code_file",
        observed_at=event.timestamp,
        summary=f"{language} source file with {line_count} lines",
        properties={
            "source_path": path,
            "language": language,
            "source_sha256": sha256,
            "bytes": byte_count,
            "lines": line_count,
            **_refresh_transform_properties(event.payload),
        },
    )
    edge = ExtractedEdge(
        source=actor.name,
        target=code_file.name,
        relation_type="indexed_code_file",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[actor, code_file],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("source.discovered")
@register("source.changed")
@register("source.unchanged")
@register("source.deleted")
def _extract_source_refresh_event(event: Event) -> ExtractionResult:
    """Extract source freshness metadata from context refresh events."""
    path = _optional_text(event.payload.get("path")) or "source"
    source_kind = _optional_text(event.payload.get("source_kind")) or "unknown"
    sha256 = _optional_text(event.payload.get("sha256"))
    previous_sha256 = _optional_text(event.payload.get("previous_sha256"))
    byte_count = _positive_int(event.payload.get("bytes"), default=0)
    status = event.type.removeprefix("source.")
    refresh_properties = _refresh_transform_properties(event.payload)
    if refresh_reason := _optional_text(event.payload.get("refresh_reason")):
        refresh_properties["refresh_reason"] = refresh_reason
    entity = ExtractedEntity(
        name=path,
        entity_type="source",
        observed_at=event.timestamp,
        summary=f"{source_kind} source {path} {status}",
        properties=_merge_properties(
            {
                "source_path": path,
                "source_kind": source_kind,
                "source_sha256": sha256,
                "previous_sha256": previous_sha256,
                "bytes": byte_count,
                "refresh_status": status,
                **refresh_properties,
            },
            {},
        )
        or {},
    )
    return ExtractionResult(entities=[entity], edges=[], source_event_seq=event.seq)


@register("projection.updated")
@register("projection.retired")
def _extract_projection_refresh_event(event: Event) -> ExtractionResult:
    """Extract projection lifecycle metadata from context refresh events."""
    path = _optional_text(event.payload.get("path")) or "source"
    source_kind = _optional_text(event.payload.get("source_kind")) or "unknown"
    projection = _optional_text(event.payload.get("projection")) or "memory"
    status = event.type.removeprefix("projection.")
    projection_properties = _refresh_transform_properties(event.payload)
    if source_sha256 := _optional_text(event.payload.get("source_sha256")):
        projection_properties["source_sha256"] = source_sha256
    source = ExtractedEntity(
        name=path,
        entity_type="source",
        observed_at=event.timestamp,
        properties={
            "source_path": path,
            "source_kind": source_kind,
        },
    )
    projection_entity = ExtractedEntity(
        name=f"projection:{source_kind}:{path}",
        entity_type="projection",
        observed_at=event.timestamp,
        summary=f"{projection} projection {status} for {path}",
        properties={
            "source_path": path,
            "source_kind": source_kind,
            "projection": projection,
            "projection_status": status,
            "source_event": _optional_text(event.payload.get("source_event")),
            "reason": _optional_text(event.payload.get("reason")),
            **projection_properties,
        },
    )
    edge = ExtractedEdge(
        source=source.name,
        target=projection_entity.name,
        relation_type=f"projection_{status}",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[source, projection_entity],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("memory.feedback")
def _extract_memory_feedback(event: Event) -> ExtractionResult:
    """Extract negative or neutral context feedback without mutating target memory."""
    entity_name = _optional_text(event.payload.get("entity_name")) or f"memory:{event.seq}"
    entity_type = _optional_text(event.payload.get("entity_type")) or "memory"
    feedback = _optional_text(event.payload.get("feedback")) or "unknown"
    citation = _optional_text(event.payload.get("citation"))
    feedback_entity = ExtractedEntity(
        name=f"{entity_type}:{entity_name}:feedback:{event.seq}",
        entity_type="memory_feedback",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("reason")) or f"{feedback} feedback for {entity_name}",
        properties=_merge_properties(
            _feedback_purpose_properties(event.payload),
            {
                "feedback": feedback,
                "citation": citation,
                "last_feedback_at": event.timestamp,
            },
        )
        or {},
    )
    target = ExtractedEntity(
        name=entity_name,
        entity_type=entity_type,
        observed_at=event.timestamp,
        properties=_merge_properties(
            _feedback_purpose_properties(event.payload),
            {
                "feedback": feedback,
                "citation": citation,
                "last_feedback_at": event.timestamp,
            },
        )
        or {},
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edges = [
        ExtractedEdge(
            source=event.actor,
            target=feedback_entity.name,
            relation_type="recorded_memory_feedback",
            valid_from=event.timestamp,
        ),
        ExtractedEdge(
            source=feedback_entity.name,
            target=entity_name,
            relation_type="feedback_about_memory",
            valid_from=event.timestamp,
        ),
    ]
    return ExtractionResult(
        entities=[feedback_entity, target, actor],
        edges=edges,
        source_event_seq=event.seq,
    )


@register("memory.bootstrap.shown")
@register("memory.checkout.completed")
@register("memory.feedback.recorded")
def _extract_memory_activity_event(event: Event) -> ExtractionResult:
    """Extract model-facing memory activity markers."""
    activity = _optional_text(event.payload.get("activity")) or event.type.removeprefix("memory.").replace(".", "_")
    source = _optional_text(event.payload.get("source")) or event.actor
    query = _optional_text(event.payload.get("query"))
    entity = ExtractedEntity(
        name=f"{event.thread}:memory:{activity}:{event.seq}",
        entity_type="memory_activity",
        observed_at=event.timestamp,
        summary=f"{activity} memory activity from {source}",
        properties=_merge_properties(
            {
                "activity": activity,
                "source": source,
                "session_id": event.thread,
            },
            {"query": query} if query else None,
        )
        or {},
    )
    return ExtractionResult(entities=[entity], edges=[], source_event_seq=event.seq)


@register("memory.reminder.suggested")
def _extract_memory_reminder_suggested(event: Event) -> ExtractionResult:
    """Extract suggested memory reminders for agent recall hardening."""
    trigger = _optional_text(event.payload.get("trigger")) or "unknown"
    recommended_tool = _optional_text(event.payload.get("recommended_tool")) or "memory_checkout"
    query = _optional_text(event.payload.get("query"))
    entity = ExtractedEntity(
        name=f"{event.thread}:memory-reminder:{event.seq}",
        entity_type="memory_reminder",
        observed_at=event.timestamp,
        summary=f"Memory reminder suggested after {trigger}: call {recommended_tool}",
        properties=_merge_properties(
            {
                "trigger": trigger,
                "recommended_tool": recommended_tool,
                "session_id": event.thread,
            },
            {"query": query} if query else None,
        )
        or {},
    )
    return ExtractionResult(entities=[entity], edges=[], source_event_seq=event.seq)


@register("code.symbol.indexed")
def _extract_code_symbol_indexed(event: Event) -> ExtractionResult:
    """Extract a code symbol and connect it to the defining file."""
    path = _optional_text(event.payload.get("path")) or "code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    name = _optional_text(event.payload.get("name")) or "symbol"
    qualified_name = _optional_text(event.payload.get("qualified_name")) or name
    kind = _optional_text(event.payload.get("kind")) or "symbol"
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    end_line = _positive_int(event.payload.get("end_line"), default=start_line)
    code_file = ExtractedEntity(
        name=path,
        entity_type="code_file",
        observed_at=event.timestamp,
        properties={
            "source_path": path,
            "language": language,
        },
    )
    symbol = ExtractedEntity(
        name=f"{path}::{qualified_name}",
        entity_type="code_symbol",
        observed_at=event.timestamp,
        summary=f"{language} {kind} {qualified_name} defined in {path}:{start_line}-{end_line}",
        properties={
            "source_path": path,
            "language": language,
            "symbol_name": name,
            "qualified_name": qualified_name,
            "symbol_kind": kind,
            "source_start_line": start_line,
            "source_end_line": end_line,
            **_source_sha256_property(event.payload),
            **_refresh_transform_properties(event.payload),
        },
    )
    edge = ExtractedEdge(
        source=code_file.name,
        target=symbol.name,
        relation_type="defines_symbol",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[code_file, symbol],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("code.import.indexed")
def _extract_code_import_indexed(event: Event) -> ExtractionResult:
    """Extract a code import and connect it to the importing file."""
    path = _optional_text(event.payload.get("path")) or "code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    module = _optional_text(event.payload.get("module")) or "unknown"
    name = _optional_text(event.payload.get("name")) or module
    kind = _optional_text(event.payload.get("kind")) or "import"
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    code_file = ExtractedEntity(
        name=path,
        entity_type="code_file",
        observed_at=event.timestamp,
        properties={
            "source_path": path,
            "language": language,
        },
    )
    imported = ExtractedEntity(
        name=f"import:{module}:{name}",
        entity_type="code_import",
        observed_at=event.timestamp,
        summary=f"{language} {kind} {name} from {module} in {path}:{start_line}",
        properties={
            "source_path": path,
            "language": language,
            "module": module,
            "import_name": name,
            "import_kind": kind,
            "source_start_line": start_line,
            **_source_sha256_property(event.payload),
            **_refresh_transform_properties(event.payload),
        },
    )
    edge = ExtractedEdge(
        source=code_file.name,
        target=imported.name,
        relation_type="imports",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[code_file, imported],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("code.dependency.indexed")
def _extract_code_dependency_indexed(event: Event) -> ExtractionResult:
    """Extract a resolved local code dependency between files."""
    source_path = _optional_text(event.payload.get("source_path")) or "source-code-file"
    target_path = _optional_text(event.payload.get("target_path")) or "target-code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    module = _optional_text(event.payload.get("module")) or "unknown"
    import_name = _optional_text(event.payload.get("import_name")) or module
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    resolution = _optional_text(event.payload.get("resolution")) or "unknown"
    source_file = ExtractedEntity(
        name=source_path,
        entity_type="code_file",
        observed_at=event.timestamp,
        properties={
            "source_path": source_path,
            "language": language,
        },
    )
    target_file = ExtractedEntity(
        name=target_path,
        entity_type="code_file",
        observed_at=event.timestamp,
        properties={
            "source_path": target_path,
            "language": language,
        },
    )
    edge = ExtractedEdge(
        source=source_path,
        target=target_path,
        relation_type="depends_on_file",
        valid_from=event.timestamp,
    )
    dependency = ExtractedEntity(
        name=f"{source_path}->{target_path}:{start_line}",
        entity_type="code_dependency",
        observed_at=event.timestamp,
        summary=f"{source_path} imports {import_name} from {module} via {target_path}:{start_line}",
        properties={
            "source_path": source_path,
            "target_path": target_path,
            "language": language,
            "module": module,
            "import_name": import_name,
            "source_start_line": start_line,
            "resolution": resolution,
            **_source_sha256_property(event.payload),
            **_refresh_transform_properties(event.payload),
        },
    )
    return ExtractionResult(
        entities=[source_file, target_file, dependency],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("code.call.indexed")
def _extract_code_call_indexed(event: Event) -> ExtractionResult:
    """Extract a code call-site and resolved call edge when available."""
    path = _optional_text(event.payload.get("path")) or "code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    caller = _optional_text(event.payload.get("caller")) or "caller"
    callee = _optional_text(event.payload.get("callee")) or "callee"
    callee_qualified_name = _optional_text(event.payload.get("callee_qualified_name")) or callee
    target_path = _optional_text(event.payload.get("target_path"))
    target_qualified_name = _optional_text(event.payload.get("target_qualified_name"))
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    resolution = _optional_text(event.payload.get("resolution")) or "unresolved"
    caller_symbol = ExtractedEntity(
        name=f"{path}::{caller}",
        entity_type="code_symbol",
        observed_at=event.timestamp,
        properties={
            "source_path": path,
            "language": language,
            "qualified_name": caller,
        },
    )
    call = ExtractedEntity(
        name=f"{path}::{caller}->{callee_qualified_name}:{start_line}",
        entity_type="code_call",
        observed_at=event.timestamp,
        summary=f"{caller} calls {callee_qualified_name} in {path}:{start_line}",
        properties={
            "source_path": path,
            "language": language,
            "caller": caller,
            "callee": callee,
            "callee_qualified_name": callee_qualified_name,
            "target_path": target_path,
            "target_qualified_name": target_qualified_name,
            "source_start_line": start_line,
            "resolution": resolution,
            **_source_sha256_property(event.payload),
            **_refresh_transform_properties(event.payload),
        },
    )
    entities = [caller_symbol, call]
    edges: list[ExtractedEdge] = []
    if target_path and target_qualified_name:
        target_symbol = ExtractedEntity(
            name=f"{target_path}::{target_qualified_name}",
            entity_type="code_symbol",
            observed_at=event.timestamp,
            properties={
                "source_path": target_path,
                "language": language,
                "qualified_name": target_qualified_name,
            },
        )
        entities.append(target_symbol)
        edges.append(
            ExtractedEdge(
                source=caller_symbol.name,
                target=target_symbol.name,
                relation_type="calls_symbol",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(
        entities=entities,
        edges=edges,
        source_event_seq=event.seq,
    )


@register("code.coverage.indexed")
def _extract_code_coverage_indexed(event: Event) -> ExtractionResult:
    """Extract a test-to-production symbol coverage link."""
    test_path = _optional_text(event.payload.get("test_path")) or "test-code-file"
    test_name = _optional_text(event.payload.get("test_name")) or "test"
    test_qualified_name = _optional_text(event.payload.get("test_qualified_name")) or test_name
    target_path = _optional_text(event.payload.get("target_path")) or "target-code-file"
    target_name = _optional_text(event.payload.get("target_name")) or "target"
    target_qualified_name = _optional_text(event.payload.get("target_qualified_name")) or target_name
    language = _optional_text(event.payload.get("language")) or "unknown"
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    resolution = _optional_text(event.payload.get("resolution")) or "unknown"
    test_symbol = ExtractedEntity(
        name=f"{test_path}::{test_qualified_name}",
        entity_type="code_symbol",
        observed_at=event.timestamp,
        properties={
            "source_path": test_path,
            "language": language,
            "symbol_name": test_name,
            "qualified_name": test_qualified_name,
            "symbol_kind": "test",
        },
    )
    target_symbol = ExtractedEntity(
        name=f"{target_path}::{target_qualified_name}",
        entity_type="code_symbol",
        observed_at=event.timestamp,
        properties={
            "source_path": target_path,
            "language": language,
            "symbol_name": target_name,
            "qualified_name": target_qualified_name,
        },
    )
    coverage = ExtractedEntity(
        name=f"{test_symbol.name}=>{target_symbol.name}:{start_line}",
        entity_type="code_coverage",
        observed_at=event.timestamp,
        summary=f"{test_qualified_name} tests {target_qualified_name} at {test_path}:{start_line}",
        properties={
            "test_path": test_path,
            "test_name": test_name,
            "test_qualified_name": test_qualified_name,
            "target_path": target_path,
            "target_name": target_name,
            "target_qualified_name": target_qualified_name,
            "language": language,
            "source_start_line": start_line,
            "resolution": resolution,
        },
    )
    edge = ExtractedEdge(
        source=test_symbol.name,
        target=target_symbol.name,
        relation_type="tests_symbol",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[test_symbol, target_symbol, coverage],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("session.genesis")
def _extract_session_genesis(event: Event) -> ExtractionResult:
    """Extract workspace genesis metadata."""
    root = _optional_text(event.payload.get("root")) or "workspace"
    workspace_type = _optional_text(event.payload.get("workspace_type")) or "generic_workspace"
    instructions_profile = _optional_text(event.payload.get("instructions_profile")) or "generic"
    session_id = _optional_text(event.payload.get("session_id")) or event.thread or "default"
    confidence = event.payload.get("confidence", 0.0)
    signals = event.payload.get("signals")
    if not isinstance(signals, list):
        signals = []
    workspace = ExtractedEntity(
        name=root,
        entity_type="workspace",
        observed_at=event.timestamp,
        summary=f"{workspace_type} workspace profile {instructions_profile}",
        properties={
            "root": root,
            "workspace_type": workspace_type,
            "confidence": confidence,
            "signals": signals,
            "instructions_profile": instructions_profile,
            "session_id": session_id,
        },
    )
    session = ExtractedEntity(
        name=session_id,
        entity_type="session",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=session.name,
        target=workspace.name,
        relation_type="initialized_workspace",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[session, workspace],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("session.profile.corrected")
def _extract_session_profile_corrected(event: Event) -> ExtractionResult:
    """Extract a durable workspace profile correction."""
    session_id = _optional_text(event.payload.get("session_id")) or event.thread or "default"
    from_profile = _optional_text(event.payload.get("from")) or "unknown"
    to_profile = _optional_text(event.payload.get("to")) or "unknown"
    reason = _optional_text(event.payload.get("reason"))
    root = _optional_text(event.payload.get("root"))
    correction = ExtractedEntity(
        name=f"{session_id}:{from_profile}->{to_profile}",
        entity_type="workspace_profile_correction",
        observed_at=event.timestamp,
        summary=reason,
        properties={
            "session_id": session_id,
            "root": root,
            "from": from_profile,
            "to": to_profile,
            "reason": reason,
        },
    )
    session = ExtractedEntity(
        name=session_id,
        entity_type="session",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=session.name,
        target=correction.name,
        relation_type="corrected_workspace_profile",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[session, correction],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("workspace.instructions.discovered")
@register("workspace.instructions.updated")
def _extract_workspace_instructions_discovered(event: Event) -> ExtractionResult:
    """Extract a workspace instruction snapshot."""
    session_id = _optional_text(event.payload.get("session_id")) or event.thread or "default"
    root = _optional_text(event.payload.get("root")) or "workspace"
    signature = _optional_text(event.payload.get("signature")) or str(event.seq)
    summary = _optional_text(event.payload.get("summary"))
    files = event.payload.get("files")
    if not isinstance(files, list):
        files = []
    file_paths = [
        path
        for file in files
        if isinstance(file, dict) and (path := _optional_text(file.get("path")))
    ]
    file_kinds = [
        kind
        for file in files
        if isinstance(file, dict) and (kind := _optional_text(file.get("kind")))
    ]
    instruction = ExtractedEntity(
        name=f"{root}:instructions:{signature}",
        entity_type="workspace_instructions",
        observed_at=event.timestamp,
        summary=summary,
        properties={
            "session_id": session_id,
            "root": root,
            "signature": signature,
            "file_count": len(files),
            "file_paths": file_paths,
            "file_kinds": file_kinds,
        },
    )
    session = ExtractedEntity(
        name=session_id,
        entity_type="session",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=session.name,
        target=instruction.name,
        relation_type="uses_workspace_instructions",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[session, instruction],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("tool.call.completed")
def _extract_tool_call_completed(event: Event) -> ExtractionResult:
    """Extract a completed tool call lifecycle event."""
    session_id = _optional_text(event.payload.get("session_id")) or event.thread or "default"
    tool_name = _optional_text(event.payload.get("tool_name")) or "tool"
    status = _optional_text(event.payload.get("status")) or "unknown"
    call_id = _optional_text(event.payload.get("call_id"))
    name = f"{session_id}:{tool_name}:{call_id or event.seq}"
    tool_call = ExtractedEntity(
        name=name,
        entity_type="tool_call",
        observed_at=event.timestamp,
        summary=_join_summary(status, event.payload.get("result_summary")),
        properties={
            "tool_name": tool_name,
            "status": status,
            "session_id": session_id,
            "call_id": call_id,
        },
    )
    session = ExtractedEntity(name=session_id, entity_type="session", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=session.name,
        target=tool_call.name,
        relation_type="completed_tool_call",
        valid_from=event.timestamp,
    )
    entities, edges = _with_explicit_task_observation(
        event,
        [session, tool_call],
        [edge],
        target=tool_call.name,
        relation_type="observed_tool_call",
    )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


@register("command.completed")
def _extract_command_completed(event: Event) -> ExtractionResult:
    """Extract a completed shell/process command lifecycle event."""
    session_id = _optional_text(event.payload.get("session_id")) or event.thread or "default"
    command_text = _optional_text(event.payload.get("command")) or "command"
    outcome = _optional_text(event.payload.get("outcome")) or "unknown"
    command = ExtractedEntity(
        name=f"{session_id}:{command_text}:{event.seq}",
        entity_type="command_run",
        observed_at=event.timestamp,
        summary=_join_summary(outcome, command_text),
        properties={
            "command": command_text,
            "exit_code": event.payload.get("exit_code"),
            "outcome": outcome,
            "session_id": session_id,
        },
    )
    session = ExtractedEntity(name=session_id, entity_type="session", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=session.name,
        target=command.name,
        relation_type="completed_command",
        valid_from=event.timestamp,
    )
    entities, edges = _with_explicit_task_observation(
        event,
        [session, command],
        [edge],
        target=command.name,
        relation_type="observed_command",
    )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


@register("file.edit.applied")
def _extract_file_edit_applied(event: Event) -> ExtractionResult:
    """Extract a file-edit lifecycle event without source content."""
    session_id = _optional_text(event.payload.get("session_id")) or event.thread or "default"
    path = _optional_text(event.payload.get("path")) or "file"
    operation = _optional_text(event.payload.get("operation")) or "modified"
    edit = ExtractedEntity(
        name=f"{session_id}:{path}:{event.seq}",
        entity_type="file_edit",
        observed_at=event.timestamp,
        summary=_join_summary(operation, event.payload.get("summary")),
        properties={
            "path": path,
            "operation": operation,
            "session_id": session_id,
            "line_count": event.payload.get("line_count"),
        },
    )
    session = ExtractedEntity(name=session_id, entity_type="session", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=session.name,
        target=edit.name,
        relation_type="applied_file_edit",
        valid_from=event.timestamp,
    )
    entities, edges = _with_explicit_task_observation(
        event,
        [session, edit],
        [edge],
        target=edit.name,
        relation_type="observed_file_edit",
    )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


@register("compaction.completed")
def _extract_compaction_completed(event: Event) -> ExtractionResult:
    """Extract a completed compaction lifecycle event."""
    session_id = _optional_text(event.payload.get("session_id")) or event.thread or "default"
    mode = _optional_text(event.payload.get("mode")) or "unknown"
    status = _optional_text(event.payload.get("status")) or "unknown"
    log_path = _optional_text(event.payload.get("log_path")) or "eventloom"
    compaction = ExtractedEntity(
        name=f"{session_id}:compaction:{event.seq}",
        entity_type="compaction_run",
        observed_at=event.timestamp,
        summary=_join_summary(status, mode, log_path),
        properties={
            "session_id": session_id,
            "mode": mode,
            "status": status,
            "log_path": log_path,
            "event_count": event.payload.get("event_count"),
            "output_path": event.payload.get("output_path"),
            "projection_path": event.payload.get("projection_path"),
            "snapshot_path": event.payload.get("snapshot_path"),
            "strategy": event.payload.get("strategy"),
        },
    )
    session = ExtractedEntity(name=session_id, entity_type="session", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=session.name,
        target=compaction.name,
        relation_type="completed_compaction",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[session, compaction],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("subagent.completed")
def _extract_subagent_completed(event: Event) -> ExtractionResult:
    """Extract a completed subagent lifecycle event."""
    parent_session_id = _optional_text(event.payload.get("parent_session_id")) or "default"
    subagent_session_id = _optional_text(event.payload.get("subagent_session_id")) or event.thread or "subagent"
    status = _optional_text(event.payload.get("status")) or "unknown"
    summary = _optional_text(event.payload.get("summary"))
    subagent = ExtractedEntity(
        name=f"{parent_session_id}:{subagent_session_id}:{event.seq}",
        entity_type="subagent_run",
        observed_at=event.timestamp,
        summary=_join_summary(status, summary),
        properties={
            "parent_session_id": parent_session_id,
            "subagent_session_id": subagent_session_id,
            "status": status,
        },
    )
    parent = ExtractedEntity(
        name=parent_session_id,
        entity_type="session",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=parent.name,
        target=subagent.name,
        relation_type="completed_subagent",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[parent, subagent],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("session.ended")
def _extract_session_ended(event: Event) -> ExtractionResult:
    """Extract a session-end lifecycle event."""
    session_id = _optional_text(event.payload.get("session_id")) or event.thread or "default"
    reason = _optional_text(event.payload.get("reason")) or "ended"
    status = _optional_text(event.payload.get("status")) or "unknown"
    ended = ExtractedEntity(
        name=f"{session_id}:session-ended:{event.seq}",
        entity_type="session_end",
        observed_at=event.timestamp,
        summary=_join_summary(status, reason),
        properties={
            "session_id": session_id,
            "reason": reason,
            "status": status,
        },
    )
    session = ExtractedEntity(name=session_id, entity_type="session", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=session.name,
        target=ended.name,
        relation_type="ended_session",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[session, ended], edges=[edge], source_event_seq=event.seq)


@register("transcript.turn")
def _extract_transcript_turn(event: Event) -> ExtractionResult:
    """Extract a sanitized session transcript turn."""
    source = _optional_text(event.payload.get("source")) or "transcript"
    turn_index = _positive_int(event.payload.get("turn_index"), default=event.seq)
    role = _optional_text(event.payload.get("role")) or event.actor
    content = _optional_text(event.payload.get("content")) or ""
    redacted_paths = event.payload.get("redacted_paths")
    if not isinstance(redacted_paths, list):
        redacted_paths = []
    entity = ExtractedEntity(
        name=f"{source}:turn-{turn_index}",
        entity_type="transcript_turn",
        observed_at=event.timestamp,
        summary=f"{role}: {content}",
        properties={
            "transcript_source": source,
            "transcript_role": role,
            "transcript_turn_index": turn_index,
            "redacted_paths": redacted_paths,
        },
    )
    neutral = neutral_transcript_record(
        actor=event.actor,
        timestamp=event.timestamp,
        source=source,
        turn_index=turn_index,
        role=role,
        content=content,
        source_event_ref=_event_ref(event),
        permission_scope=_optional_text(event.payload.get("permission_scope")),
        uncertainty=_optional_text(event.payload.get("uncertainty")),
        candidate_claim=_optional_text(event.payload.get("candidate_claim")),
    )
    neutral_entity = ExtractedEntity(
        name=neutral.substrate_id,
        entity_type="neutral_substrate",
        observed_at=event.timestamp,
        summary=neutral.quote,
        properties=neutral.to_properties(),
    )
    neutral_edge = ExtractedEdge(
        source=neutral.substrate_id,
        target=entity.name,
        relation_type="neutral_substrate_cites_source",
        valid_from=event.timestamp,
    )
    audit_entity, audit_edge = _neutral_audit_projection(event, neutral.substrate_id)
    return ExtractionResult(
        entities=[entity, neutral_entity, *([audit_entity] if audit_entity is not None else [])],
        edges=[neutral_edge, *([audit_edge] if audit_edge is not None else [])],
        source_event_seq=event.seq,
    )


@register("llm.packet.projected")
def _extract_llm_packet_projected(event: Event) -> ExtractionResult:
    """Extract a cold-path LLM packet projection."""
    session_id = _optional_text(event.payload.get("session_id")) or event.thread
    source_event_seq = _positive_int(event.payload.get("source_event_seq"), default=event.seq)
    source_event_hash = _optional_text(event.payload.get("source_event_hash"))
    provider_path = _optional_text(event.payload.get("provider_path")) or "unknown-provider-path"
    status_code = _positive_int(event.payload.get("status_code"), default=0)
    model = _optional_text(event.payload.get("model"))
    usage_counts = event.payload.get("usage_counts")
    if not isinstance(usage_counts, dict):
        usage_counts = {}
    packet = ExtractedEntity(
        name=f"{session_id}:llm-packet:{source_event_seq}",
        entity_type="llm_packet_projection",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties={
            "session_id": session_id,
            "source_event_seq": source_event_seq,
            "source_event_hash": source_event_hash,
            "provider_path": provider_path,
            "status_code": status_code,
            "model": model,
            "prompt_tokens": usage_counts.get("prompt"),
            "completion_tokens": usage_counts.get("completion"),
            "total_tokens": usage_counts.get("total"),
        },
    )
    session = ExtractedEntity(
        name=session_id,
        entity_type="session",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=session.name,
        target=packet.name,
        relation_type="projected_llm_packet",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[session, packet],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("inference.edge.generated")
def _extract_inference_edge_generated(event: Event) -> ExtractionResult:
    """Project an explicit, auditable inferred relationship event."""
    source = _entity_reference(
        event.payload.get("source"),
        role="source",
        event_seq=event.seq,
        observed_at=event.timestamp,
    )
    target = _entity_reference(
        event.payload.get("target"),
        role="target",
        event_seq=event.seq,
        observed_at=event.timestamp,
    )
    relation_type = _required_text(
        event.payload.get("relation_type"),
        field="relation_type",
        event_seq=event.seq,
    )
    inference_method = _required_text(
        event.payload.get("inference_method"),
        field="inference_method",
        event_seq=event.seq,
    )
    confidence = _required_confidence(event.payload.get("confidence"), event_seq=event.seq)
    evidence = event.payload.get("evidence")
    edge = ExtractedEdge(
        source=source.name,
        target=target.name,
        relation_type=relation_type,
        valid_from=event.timestamp,
        inferred=True,
        confidence=confidence,
        inference_method=inference_method,
        evidence=evidence if isinstance(evidence, dict) else {},
    )
    return ExtractionResult(
        entities=[source, target],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("inference.edge.retracted")
def _extract_inference_edge_retracted(event: Event) -> ExtractionResult:
    """Project a contradicted inferred edge as a closed validity interval."""
    source = _entity_reference(
        event.payload.get("source"),
        role="source",
        event_seq=event.seq,
        observed_at=event.timestamp,
    )
    target = _entity_reference(
        event.payload.get("target"),
        role="target",
        event_seq=event.seq,
        observed_at=event.timestamp,
    )
    relation_type = _required_text(
        event.payload.get("relation_type"),
        field="relation_type",
        event_seq=event.seq,
    )
    inference_method = _required_text(
        event.payload.get("inference_method"),
        field="inference_method",
        event_seq=event.seq,
    )
    valid_from = _required_text(
        event.payload.get("valid_from"),
        field="valid_from",
        event_seq=event.seq,
    )
    valid_to = _required_text(
        event.payload.get("valid_to"),
        field="valid_to",
        event_seq=event.seq,
    )
    confidence = _required_confidence(event.payload.get("confidence"), event_seq=event.seq)
    evidence = event.payload.get("evidence")
    edge = ExtractedEdge(
        source=source.name,
        target=target.name,
        relation_type=relation_type,
        valid_from=valid_from,
        valid_to=valid_to,
        inferred=True,
        confidence=confidence,
        inference_method=inference_method,
        evidence=evidence if isinstance(evidence, dict) else {},
    )
    return ExtractionResult(
        entities=[source, target],
        edges=[edge],
        source_event_seq=event.seq,
    )


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


def _required_text(value: object, *, field: str, event_seq: int) -> str:
    """Return required text or raise a precise extraction error."""
    if text := _optional_text(value):
        return text
    raise ValueError(f"inference.edge.generated event {event_seq} missing required {field}")


def _required_confidence(value: object, *, event_seq: int) -> float:
    """Return a required 0..1 confidence value for an inferred edge event."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"inference.edge.generated event {event_seq} missing required confidence")
    try:
        confidence = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"inference.edge.generated event {event_seq} has invalid confidence"
        ) from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"inference.edge.generated event {event_seq} confidence must be between 0.0 and 1.0"
        )
    return confidence


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
    name = _required_text(value.get("name"), field=f"{role}.name", event_seq=event_seq)
    entity_type = _required_text(
        value.get("entity_type"),
        field=f"{role}.entity_type",
        event_seq=event_seq,
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
