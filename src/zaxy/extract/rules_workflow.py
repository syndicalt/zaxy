"""Rule extractors: goals, tasks, decisions, policy, preferences, skills, hooks, issues, verification, handoff."""

from __future__ import annotations

from zaxy.event import Event
from zaxy.extract.core import (
    ExtractedEdge,
    ExtractedEntity,
    ExtractionResult,
    _bounded_float,
    _compact_properties,
    _extract_skill_version_event,
    _join_summary,
    _merge_properties,
    _optional_text,
    _positive_int,
    _preference_summary,
    _retention_properties,
    _skill_entities,
    _skill_id,
    _skill_version,
    _skill_version_edge,
    _string_list,
    _task_identity_properties,
    _with_explicit_task_observation,
    register,
)


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
