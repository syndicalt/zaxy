"""Rule extractors: memory lifecycle (reinforced, evidence, synthesis, feedback, correction, rollback, reminder, bootstrap)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zaxy.event import Event
from zaxy.extract.core import (
    _CONSOLIDATION_AUTHORITY_STATUS,
    ExtractedEdge,
    ExtractedEntity,
    ExtractionResult,
    _bounded_float,
    _compact_properties,
    _dict_list,
    _feedback_purpose_properties,
    _join_summary,
    _merge_properties,
    _optional_text,
    _positive_int,
    _retention_properties,
    _string_list,
    _synthesis_candidate_id,
    _synthesis_ledger_row_id,
    register,
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


@register("memory.reinforcement")
def _extract_memory_reinforcement(event: Event) -> ExtractionResult:
    """Skip projection for non-authoritative salience reinforcement markers.

    Reinforcement events are observability state replayed by the salience
    ledger; projecting them as entities would let reinforcement bookkeeping
    leak into ranked retrieval.
    """
    return ExtractionResult(entities=[], edges=[], source_event_seq=event.seq)


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


@register("memory.corrected")
def _extract_memory_corrected(event: Event) -> ExtractionResult:
    """Project a cited, non-authoritative human correction (re-ingest of an edit).

    The corrected content is surfaced as a searchable entity citing the original
    event; the original event is never mutated or deleted, so the correction is
    purely additive.
    """
    correction_id = _optional_text(event.payload.get("correction_id")) or f"correction:{event.seq}"
    content = _optional_text(event.payload.get("content"))
    reason = _optional_text(event.payload.get("reason"))
    target = event.payload.get("target")
    target = target if isinstance(target, Mapping) else {}
    target_seq = _positive_int(target.get("seq"), default=0)
    target_hash = _optional_text(target.get("hash"))
    properties: dict[str, Any] = {
        "correction_id": correction_id,
        "reason": reason,
        "authority_status": event.payload.get("authority_status", "non_authoritative"),
    }
    if target_seq:
        properties["target_seq"] = target_seq
    if target_hash is not None:
        properties["target_hash"] = target_hash
        properties["target_ref"] = f"{target_seq}:{target_hash}"
    entity = ExtractedEntity(
        name=correction_id,
        entity_type="memory_correction",
        observed_at=event.timestamp,
        summary=content,
        properties=_compact_properties(properties),
    )
    return ExtractionResult(entities=[entity], edges=[], source_event_seq=event.seq)


@register("memory.rolled_back")
def _extract_memory_rolled_back(event: Event) -> ExtractionResult:
    """Project a reversible rollback: undo a cited evolution without deletion.

    For a rolled-back consolidation review the cited candidate is reverted to its
    prior review status (additive, mirroring fleet rollback). Every rollback is
    also retained as a cited, non-authoritative marker for audit.
    """
    rollback_id = _optional_text(event.payload.get("rollback_id")) or f"rollback:{event.seq}"
    reason = _optional_text(event.payload.get("reason"))
    target = event.payload.get("target")
    target = target if isinstance(target, Mapping) else {}
    target_seq = _positive_int(target.get("seq"), default=0)
    target_hash = _optional_text(target.get("hash"))
    reverts = event.payload.get("reverts")
    reverts = reverts if isinstance(reverts, Mapping) else {}
    reverts_type = _optional_text(reverts.get("event_type"))
    candidate_id = _optional_text(reverts.get("candidate_id"))
    to_status = _optional_text(reverts.get("to_status"))

    entities: list[ExtractedEntity] = []
    if (
        reverts_type == "consolidation.candidate.reviewed"
        and candidate_id is not None
        and to_status is not None
    ):
        entities.append(
            ExtractedEntity(
                name=candidate_id,
                entity_type="consolidation_candidate",
                observed_at=event.timestamp,
                properties={
                    "review_status": to_status,
                    "authority_status": _CONSOLIDATION_AUTHORITY_STATUS,
                },
            )
        )

    properties: dict[str, Any] = {
        "rollback_id": rollback_id,
        "reason": reason,
        "reverts_event_type": reverts_type,
        "authority_status": event.payload.get("authority_status", "non_authoritative"),
    }
    if target_seq:
        properties["target_seq"] = target_seq
    if target_hash is not None:
        properties["target_hash"] = target_hash
        properties["target_ref"] = f"{target_seq}:{target_hash}"
    entities.append(
        ExtractedEntity(
            name=rollback_id,
            entity_type="memory_rollback",
            observed_at=event.timestamp,
            summary=reason,
            properties=_compact_properties(properties),
        )
    )
    return ExtractionResult(entities=entities, edges=[], source_event_seq=event.seq)
