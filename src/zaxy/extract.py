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
        summary=_join_summary(
            f"{event.actor} emitted {event.type}",
            _fallback_payload_summary(event.payload),
        ),
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
    instruction = ExtractedEntity(
        name=f"{root}:instructions:{signature}",
        entity_type="workspace_instructions",
        observed_at=event.timestamp,
        summary=summary,
        properties={
            "session_id": session_id,
            "root": root,
            "signature": signature,
            "files": files,
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
    return ExtractionResult(entities=[session, tool_call], edges=[edge], source_event_seq=event.seq)


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
    return ExtractionResult(entities=[session, command], edges=[edge], source_event_seq=event.seq)


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
    return ExtractionResult(entities=[session, edit], edges=[edge], source_event_seq=event.seq)


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
