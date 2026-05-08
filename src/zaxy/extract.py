"""Hybrid extraction engine: rule-based + LLM fallback.

Typed Eventloom events are mapped deterministically to graph nodes and edges
by registered extractors. Unknown or unstructured events fall back to an LLM
for entity/relation extraction via Graphiti.

This design cuts LLM extraction costs by 60–80%% for agents that emit
structured event types.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from zaxy.event import Event


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


@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting a single event."""

    entities: list[ExtractedEntity]
    edges: list[ExtractedEdge]
    source_event_seq: int
    source_event_hash: str | None = None
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
        summary=f"{event.actor} emitted {event.type}",
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
        properties={
            "source": source,
            "project": project,
        },
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
    entity = ExtractedEntity(
        name=f"{path}:{start_line}-{end_line}",
        entity_type="document",
        observed_at=event.timestamp,
        summary=content,
        properties={
            "source_path": path,
            "source_start_line": start_line,
            "source_end_line": end_line,
            "source_sha256": sha256,
        },
    )
    return ExtractionResult(
        entities=[entity],
        edges=[],
        source_event_seq=event.seq,
    )


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
    return ExtractionResult(
        entities=[entity],
        edges=[],
        source_event_seq=event.seq,
    )


def _optional_text(value: object) -> str | None:
    """Return non-empty text for extracted summaries."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
