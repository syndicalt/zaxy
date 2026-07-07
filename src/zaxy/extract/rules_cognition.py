"""Rule extractors: session/workspace/turn telemetry, inference, causal, consolidation, metacognition, reasoning, belief."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from zaxy.causal import CausalEdge, causal_relation_to_graph_relation
from zaxy.consolidation import (
    CONSOLIDATION_INITIAL_REVIEW_STATUS,
    CONSOLIDATION_REVIEW_STATUSES,
)
from zaxy.event import Event
from zaxy.extract.core import (
    _CONSOLIDATION_AUTHORITY_STATUS,
    ExtractedEdge,
    ExtractedEntity,
    ExtractionResult,
    _compact_properties,
    _dict_list,
    _entity_reference,
    _entity_reference_from_mapping,
    _event_ref,
    _join_summary,
    _neutral_audit_projection,
    _non_negative_int,
    _optional_text,
    _positive_int,
    _required_causal_graph_relation_type,
    _required_confidence,
    _required_consolidation_authority_status,
    _required_consolidation_candidate_id,
    _required_consolidation_candidate_type,
    _required_consolidation_confidence,
    _required_consolidation_text,
    _required_numeric_confidence,
    _required_reasoning_phase,
    _required_reasoning_text,
    _required_strict_text,
    _required_text,
    _snapshot_consolidation_source_events,
    _string_list,
    _with_explicit_task_observation,
    register,
)
from zaxy.neutral import (
    neutral_transcript_record,
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


@register("causal.edge.generated")
def _extract_causal_edge_generated(event: Event) -> ExtractionResult:
    """Project an explicit, cited causal edge as non-authoritative graph evidence."""
    graph_relation_type = _required_causal_graph_relation_type(event.payload, event_seq=event.seq)
    source_payload = event.payload.get("source")
    target_payload = event.payload.get("target")
    evidence_payload = event.payload.get("evidence")
    if not isinstance(source_payload, Mapping):
        raise ValueError(f"causal.edge.generated event {event.seq} missing source entity")
    if not isinstance(target_payload, Mapping):
        raise ValueError(f"causal.edge.generated event {event.seq} missing target entity")
    if not isinstance(evidence_payload, Mapping):
        raise ValueError(f"causal.edge.generated event {event.seq} missing evidence")
    edge_contract = CausalEdge(
        source=source_payload,
        target=target_payload,
        relation_type=_required_text(
            event.payload.get("relation_type"),
            field="relation_type",
            event_seq=event.seq,
            event_type="causal.edge.generated",
        ),
        graph_relation_type=graph_relation_type,
        confidence=_required_numeric_confidence(
            event.payload.get("confidence"),
            event_seq=event.seq,
            event_type="causal.edge.generated",
        ),
        method=_required_strict_text(
            event.payload.get("causal_method"),
            field="causal method",
            event_seq=event.seq,
            event_type="causal.edge.generated",
        ),
        review_status=event.payload.get("review_status", "proposed"),
        authority_status=event.payload.get("authority_status", "non_authoritative"),
        evidence=evidence_payload,
    )
    source = _entity_reference_from_mapping(
        edge_contract.source,
        role="source",
        event_seq=event.seq,
        observed_at=event.timestamp,
        event_type="causal.edge.generated",
    )
    target = _entity_reference_from_mapping(
        edge_contract.target,
        role="target",
        event_seq=event.seq,
        observed_at=event.timestamp,
        event_type="causal.edge.generated",
    )
    edge = ExtractedEdge(
        source=source.name,
        target=target.name,
        relation_type=causal_relation_to_graph_relation(edge_contract.relation_type),
        valid_from=event.timestamp,
        inferred=True,
        confidence=edge_contract.confidence,
        inference_method=edge_contract.method,
        evidence={
            **copy.deepcopy(dict(edge_contract.evidence)),
            "causal_relation_type": edge_contract.relation_type,
            "review_status": edge_contract.review_status,
            "authority_status": edge_contract.authority_status,
        },
    )
    return ExtractionResult(
        entities=[source, target],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("consolidation.candidate.created")
def _extract_consolidation_candidate_created(event: Event) -> ExtractionResult:
    """Project a cited, review-pending consolidation candidate."""
    candidate_id = _required_consolidation_candidate_id(event.payload.get("candidate_id"))
    candidate_type = _required_consolidation_candidate_type(event.payload.get("candidate_type"))
    if not candidate_id.startswith(f"consolidation:{candidate_type}:"):
        raise ValueError("candidate_id candidate_type must match candidate_type")
    title = _required_consolidation_text(event.payload.get("title"), field="title")
    summary = _required_consolidation_text(event.payload.get("summary"), field="summary")
    source_events = _snapshot_consolidation_source_events(event.payload.get("source_events"))
    source_event_refs = [f"{source_event['seq']}:{source_event['hash']}" for source_event in source_events]
    source_event_seqs = [source_event["seq"] for source_event in source_events]
    source_event_hashes = [source_event["hash"] for source_event in source_events]
    confidence = _required_consolidation_confidence(event.payload.get("confidence"))
    method = _required_consolidation_text(event.payload.get("method"), field="method")
    review_status = event.payload.get("review_status")
    if review_status != CONSOLIDATION_INITIAL_REVIEW_STATUS:
        raise ValueError(
            "review_status must be "
            f"{CONSOLIDATION_INITIAL_REVIEW_STATUS!r} for consolidation candidates"
        )
    authority_status = _required_consolidation_authority_status(event.payload.get("authority_status"))
    purpose = event.payload.get("purpose")
    if purpose is not None:
        purpose = _required_consolidation_text(purpose, field="purpose")

    properties: dict[str, Any] = {
        "candidate_type": candidate_type,
        "title": title,
        "confidence": confidence,
        "method": method,
        "review_status": review_status,
        "authority_status": authority_status,
        "source_event_count": len(source_events),
        "source_event_refs": source_event_refs,
        "source_event_seqs": source_event_seqs,
        "source_event_hashes": source_event_hashes,
        "source_events": source_events,
    }
    if purpose is not None:
        properties["purpose"] = purpose

    candidate = ExtractedEntity(
        name=candidate_id,
        entity_type="consolidation_candidate",
        observed_at=event.timestamp,
        summary=summary,
        properties=properties,
    )
    return ExtractionResult(entities=[candidate], edges=[], source_event_seq=event.seq)


@register("consolidation.candidate.reviewed")
def _extract_consolidation_candidate_reviewed(event: Event) -> ExtractionResult:
    """Project a human review outcome without promoting candidate authority."""
    candidate_id = _required_consolidation_candidate_id(event.payload.get("candidate_id"))
    status = event.payload.get("status")
    if status not in CONSOLIDATION_REVIEW_STATUSES:
        valid = ", ".join(sorted(CONSOLIDATION_REVIEW_STATUSES))
        raise ValueError(f"status must be one of: {valid}")
    authority_status = _required_consolidation_authority_status(event.payload.get("authority_status"))
    rationale = _required_consolidation_text(event.payload.get("rationale"), field="rationale")

    review_id = f"consolidation_review:{candidate_id}:{event.seq}"
    review = ExtractedEntity(
        name=review_id,
        entity_type="consolidation_review",
        observed_at=event.timestamp,
        summary=rationale,
        properties={
            "candidate_id": candidate_id,
            "status": status,
            "authority_status": authority_status,
            "rationale": rationale,
        },
    )
    candidate = ExtractedEntity(
        name=candidate_id,
        entity_type="consolidation_candidate",
        observed_at=event.timestamp,
        properties={
            "review_status": status,
            "authority_status": authority_status,
        },
    )
    edge = ExtractedEdge(
        source=review_id,
        target=candidate_id,
        relation_type="reviewed_consolidation_candidate",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[review, candidate], edges=[edge], source_event_seq=event.seq)


@register("metacognition.unknown.recorded")
def _extract_metacognition_unknown_recorded(event: Event) -> ExtractionResult:
    """Project an open known-unknown diagnostic without granting authority."""
    unknown_id = _required_reasoning_text(event.payload.get("unknown_id"), field="unknown_id")
    question = _required_reasoning_text(event.payload.get("question"), field="question")
    reason = _required_reasoning_text(event.payload.get("reason"), field="reason")
    claim_key = _required_reasoning_text(event.payload.get("claim_key"), field="claim_key")
    gap_type = _required_reasoning_text(event.payload.get("gap_type"), field="gap_type")
    status = _required_reasoning_text(event.payload.get("status"), field="status")
    if status != "open":
        raise ValueError("status must be 'open' for known unknowns")
    authority_status = _required_consolidation_authority_status(event.payload.get("authority_status"))
    source_events = _snapshot_consolidation_source_events(event.payload.get("source_events"))
    source_event_refs = [f"{source_event['seq']}:{source_event['hash']}" for source_event in source_events]
    source_event_seqs = [source_event["seq"] for source_event in source_events]
    source_event_hashes = [source_event["hash"] for source_event in source_events]
    unknown = ExtractedEntity(
        name=unknown_id,
        entity_type="known_unknown",
        observed_at=event.timestamp,
        summary=question,
        properties=_compact_properties(
            {
                "event_type": event.type,
                "question": question,
                "reason": reason,
                "claim_key": claim_key,
                "gap_type": gap_type,
                "status": status,
                "reverify_query": _optional_text(event.payload.get("reverify_query")),
                "source_event_count": len(source_events),
                "source_event_refs": source_event_refs,
                "source_event_seqs": source_event_seqs,
                "source_event_hashes": source_event_hashes,
                "source_events": source_events,
                "authority_status": authority_status,
            }
        ),
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=unknown_id,
        relation_type="recorded_known_unknown",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[unknown, actor], edges=[edge], source_event_seq=event.seq)


@register("metacognition.confidence.assessed")
def _extract_metacognition_confidence_assessed(event: Event) -> ExtractionResult:
    """Project an append-only confidence trajectory point as diagnostic state."""
    assessment_id = _required_reasoning_text(event.payload.get("assessment_id"), field="assessment_id")
    claim = _required_reasoning_text(event.payload.get("claim"), field="claim")
    claim_key = _required_reasoning_text(event.payload.get("claim_key"), field="claim_key")
    confidence = _required_consolidation_confidence(event.payload.get("confidence"))
    support_count = _non_negative_int(event.payload.get("support_count"), default=0)
    conflict_count = _non_negative_int(event.payload.get("conflict_count"), default=0)
    method = _required_reasoning_text(event.payload.get("method"), field="method")
    authority_status = _required_consolidation_authority_status(event.payload.get("authority_status"))
    evidence = _dict_list(event.payload.get("evidence"))
    assessment = ExtractedEntity(
        name=assessment_id,
        entity_type="confidence_assessment",
        observed_at=event.timestamp,
        summary=claim,
        properties=_compact_properties(
            {
                "event_type": event.type,
                "claim": claim,
                "claim_key": claim_key,
                "confidence": confidence,
                "support_count": support_count,
                "conflict_count": conflict_count,
                "requires_reverify": bool(event.payload.get("requires_reverify")),
                "method": method,
                "evidence_count": len(evidence),
                "evidence": evidence,
                "authority_status": authority_status,
            }
        ),
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=assessment_id,
        relation_type="assessed_confidence",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[assessment, actor], edges=[edge], source_event_seq=event.seq)


@register("metacognition.conflict.clustered")
def _extract_metacognition_conflict_clustered(event: Event) -> ExtractionResult:
    """Project unresolved support/conflict clusters as diagnostic state."""
    cluster_id = _required_reasoning_text(event.payload.get("cluster_id"), field="cluster_id")
    claim_key = _required_reasoning_text(event.payload.get("claim_key"), field="claim_key")
    claim = _required_reasoning_text(event.payload.get("claim"), field="claim")
    confidence = _required_consolidation_confidence(event.payload.get("confidence"))
    reason = _required_reasoning_text(event.payload.get("reason"), field="reason")
    resolution_status = _required_reasoning_text(
        event.payload.get("resolution_status"),
        field="resolution_status",
    )
    if resolution_status != "unresolved":
        raise ValueError("resolution_status must be 'unresolved' for conflict clusters")
    authority_status = _required_consolidation_authority_status(event.payload.get("authority_status"))
    supporting_source_events = _snapshot_consolidation_source_events(
        event.payload.get("supporting_source_events")
    )
    conflicting_source_events = _snapshot_consolidation_source_events(
        event.payload.get("conflicting_source_events")
    )
    cluster = ExtractedEntity(
        name=cluster_id,
        entity_type="conflict_cluster",
        observed_at=event.timestamp,
        summary=claim,
        properties={
            "event_type": event.type,
            "claim_key": claim_key,
            "claim": claim,
            "confidence": confidence,
            "reason": reason,
            "resolution_status": resolution_status,
            "supporting_source_event_count": len(supporting_source_events),
            "supporting_source_event_refs": [
                f"{source_event['seq']}:{source_event['hash']}"
                for source_event in supporting_source_events
            ],
            "supporting_source_event_seqs": [source_event["seq"] for source_event in supporting_source_events],
            "supporting_source_event_hashes": [source_event["hash"] for source_event in supporting_source_events],
            "supporting_source_events": supporting_source_events,
            "conflicting_source_event_count": len(conflicting_source_events),
            "conflicting_source_event_refs": [
                f"{source_event['seq']}:{source_event['hash']}"
                for source_event in conflicting_source_events
            ],
            "conflicting_source_event_seqs": [source_event["seq"] for source_event in conflicting_source_events],
            "conflicting_source_event_hashes": [source_event["hash"] for source_event in conflicting_source_events],
            "conflicting_source_events": conflicting_source_events,
            "authority_status": authority_status,
        },
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=cluster_id,
        relation_type="clustered_conflicting_evidence",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[cluster, actor], edges=[edge], source_event_seq=event.seq)


@register("metacognition.reverify.requested")
def _extract_metacognition_reverify_requested(event: Event) -> ExtractionResult:
    """Project an open re-verification request as non-authoritative diagnostic state."""
    reverify_id = _required_reasoning_text(event.payload.get("reverify_id"), field="reverify_id")
    query = _required_reasoning_text(event.payload.get("query"), field="query")
    reason = _required_reasoning_text(event.payload.get("reason"), field="reason")
    claim_key = _required_reasoning_text(event.payload.get("claim_key"), field="claim_key")
    priority = _required_reasoning_text(event.payload.get("priority"), field="priority")
    status = _required_reasoning_text(event.payload.get("status"), field="status")
    if status != "open":
        raise ValueError("status must be 'open' for reverify requests")
    authority_status = _required_consolidation_authority_status(event.payload.get("authority_status"))
    source_events = _snapshot_consolidation_source_events(event.payload.get("source_events"))
    source_event_refs = [f"{source_event['seq']}:{source_event['hash']}" for source_event in source_events]
    request = ExtractedEntity(
        name=reverify_id,
        entity_type="reverify_request",
        observed_at=event.timestamp,
        summary=query,
        properties={
            "event_type": event.type,
            "query": query,
            "reason": reason,
            "claim_key": claim_key,
            "priority": priority,
            "status": status,
            "source_event_count": len(source_events),
            "source_event_refs": source_event_refs,
            "source_event_seqs": [source_event["seq"] for source_event in source_events],
            "source_event_hashes": [source_event["hash"] for source_event in source_events],
            "source_events": source_events,
            "authority_status": authority_status,
        },
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=reverify_id,
        relation_type="requested_reverification",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[request, actor], edges=[edge], source_event_seq=event.seq)


@register("reasoning.primitive.called")
def _extract_reasoning_primitive_called(event: Event) -> ExtractionResult:
    """Project an observable reasoning-loop primitive call as trace evidence."""
    primitive = _required_reasoning_text(event.payload.get("primitive"), field="primitive")
    phase = _required_reasoning_phase(event.payload.get("phase"))
    status = _optional_text(event.payload.get("status")) or "succeeded"
    result_count = _non_negative_int(event.payload.get("result_count"), default=0)
    evidence_count = _non_negative_int(event.payload.get("evidence_count"), default=0)
    citations = _string_list(event.payload.get("citations"))
    observation_id = f"reasoning:{primitive}:{event.seq}"
    observation = ExtractedEntity(
        name=observation_id,
        entity_type="reasoning_primitive_observation",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("query")),
        properties=_compact_properties(
            {
                "event_type": event.type,
                "primitive": primitive,
                "phase": phase,
                "status": status,
                "result_count": result_count,
                "evidence_count": evidence_count,
                "citations": citations,
                "authority_status": _CONSOLIDATION_AUTHORITY_STATUS,
            }
        ),
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=observation_id,
        relation_type="called_reasoning_primitive",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[observation, actor], edges=[edge], source_event_seq=event.seq)


@register("belief.update.proposed")
def _extract_belief_update_proposed(event: Event) -> ExtractionResult:
    """Project a review-pending belief proposal without granting authority."""
    claim = _required_reasoning_text(event.payload.get("claim"), field="claim")
    rationale = _required_reasoning_text(event.payload.get("rationale"), field="rationale")
    phase = _required_reasoning_phase(event.payload.get("phase"))
    confidence = _required_consolidation_confidence(event.payload.get("confidence"))
    source_events = _snapshot_consolidation_source_events(event.payload.get("source_events"))
    source_event_refs = [f"{source_event['seq']}:{source_event['hash']}" for source_event in source_events]
    source_event_seqs = [source_event["seq"] for source_event in source_events]
    source_event_hashes = [source_event["hash"] for source_event in source_events]
    authority_status = _required_consolidation_authority_status(event.payload.get("authority_status"))
    review_status = event.payload.get("review_status")
    if review_status != "pending":
        raise ValueError("review_status must be 'pending' for belief update proposals")
    proposal_id = f"belief:proposal:{event.seq}"
    proposal = ExtractedEntity(
        name=proposal_id,
        entity_type="belief_update_proposal",
        observed_at=event.timestamp,
        summary=claim,
        properties={
            "event_type": event.type,
            "claim": claim,
            "rationale": rationale,
            "phase": phase,
            "confidence": confidence,
            "source_event_count": len(source_events),
            "source_event_refs": source_event_refs,
            "source_event_seqs": source_event_seqs,
            "source_event_hashes": source_event_hashes,
            "authority_status": authority_status,
            "review_status": review_status,
        },
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=proposal_id,
        relation_type="proposed_belief_update",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[proposal, actor], edges=[edge], source_event_seq=event.seq)
