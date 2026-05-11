"""MCP server exposing memory tools.

Provides stdio and SSE transport for agent frameworks to interact
with Zaxy via the Model Context Protocol.

Tools exposed:
- memory_append: Append a typed event to the log.
- memory_query: Query the temporal knowledge graph.
- memory_feedback: Record retrieval feedback for a graph entity.
- memory_replay: Replay events from a session.
- memory_invalidate: Mark a fact as superseded.
"""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import inspect
import json
import re
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import jwt
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from zaxy.capabilities import build_memory_capabilities
from zaxy.config import get_settings
from zaxy.context import Context, ContextAssemblyPolicy, context_counts
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.lifecycle import (
    build_session_ended_event,
    build_subagent_completed_event,
    build_tool_call_completed_event,
)
from zaxy.log import get_logger, setup_logging
from zaxy.metrics import get_metrics
from zaxy.query import QueryRouter, build_retention_policy
from zaxy.refs import MemoryRef, MemoryRefStore
from zaxy.remote_security import AuditEventExporter, RemoteAuditEvent, SessionRateLimiter
from zaxy.runtime import LocalNeo4jRuntime
from zaxy.security import (
    MAX_REPLAY_EVENTS,
    validate_from_seq,
    validate_limit,
    validate_payload,
    validate_query,
    validate_session_id,
)
from zaxy.session import SessionManager
from zaxy.trace import MemoryTracer
from zaxy.verbatim import VerbatimIndex
from zaxy.working_set import build_working_set, format_working_set
from zaxy.workspace import (
    WorkspaceProfile,
    build_session_genesis_event,
    build_workspace_instruction_event,
    existing_session_genesis_profile,
    existing_workspace_instructions_signature,
    mark_workspace_instruction_event_updated,
    workspace_profile_from_payload,
)

app = Server("zaxy-memory")
remote_session_scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "remote_session_scope",
    default=None,
)


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

TOOLS = [
    Tool(
        name="memory_append",
        description="Append a typed event to the agent's persistent memory log.",
        inputSchema={
            "type": "object",
            "required": ["event_type", "actor", "payload"],
            "properties": {
                "event_type": {"type": "string", "description": "Event type, e.g. 'goal.created'"},
                "actor": {"type": "string", "description": "Actor that emitted the event"},
                "payload": {"type": "object", "description": "Structured payload"},
                "thread": {"type": "string", "description": "Logical thread / session ID (legacy, use session_id)"},
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_query",
        description="Query the temporal knowledge graph for relevant context.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "temporal_filter": {"type": "string", "description": "ISO-8601 point-in-time filter"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_verbatim",
        description="Retrieve exact Eventloom source chunks with citations.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "session_id": {"type": "string", "description": "Session ID for source recall"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_feedback",
        description="Record feedback for retrieved context and reinforce useful memories.",
        inputSchema={
            "type": "object",
            "required": ["entity_name", "entity_type", "feedback"],
            "properties": {
                "entity_name": {"type": "string", "description": "Retrieved graph entity name"},
                "entity_type": {"type": "string", "description": "Retrieved graph entity type"},
                "feedback": {
                    "type": "string",
                    "enum": ["used", "helpful", "irrelevant"],
                    "description": "Retrieval outcome to record",
                },
                "actor": {"type": "string", "description": "Actor recording feedback", "default": "zaxy"},
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
                "query": {"type": "string", "description": "Query that returned the context"},
                "source": {"type": "string", "description": "Retrieval source", "default": "mcp"},
                "score": {"type": "number", "description": "Original retrieval score"},
                "citation": {"type": "string", "description": "Eventloom citation for the retrieved context"},
                "reason": {"type": "string", "description": "Short rationale for the feedback"},
                "importance": {
                    "type": "number",
                    "description": "Optional 0..1 reinforcement importance for positive feedback",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_replay",
        description="Replay events from a session starting at a sequence number.",
        inputSchema={
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "Session / thread ID"},
                "from_seq": {"type": "integer", "description": "Start sequence", "default": 1},
                "admin_token": {"type": "string", "description": "Admin token if configured"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_invalidate",
        description="Mark a fact as invalid at a given time (bi-temporal update).",
        inputSchema={
            "type": "object",
            "required": ["entity_name", "entity_type", "invalid_at"],
            "properties": {
                "entity_name": {"type": "string"},
                "entity_type": {"type": "string"},
                "invalid_at": {"type": "string", "description": "ISO-8601 timestamp"},
                "admin_token": {"type": "string", "description": "Admin token if configured"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_capabilities",
        description="Describe Zaxy's active memory capabilities and ambient usage loop for this session.",
        inputSchema={
            "type": "object",
            "required": [],
            "properties": {
                "session_id": {"type": "string"},
                "current_task": {"type": "string", "description": "Current task or question to seed checkout guidance"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_checkout",
        description="Checkout current, cited, prompt-ready memory state for a session.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "session_id": {"type": "string"},
                "ref": {"type": "string", "description": "Memory ref to checkout, e.g. HEAD or refs/heads/main"},
                "replay_from_seq": {"type": "integer", "default": 1},
                "limit": {"type": "integer", "default": 10},
                "max_recent_events": {"type": "integer", "default": 20},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="context_assemble",
        description="Assemble replay plus ranked retrieval into a prompt-ready context bundle.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "session_id": {"type": "string"},
                "replay_from_seq": {"type": "integer", "default": 1},
                "limit": {"type": "integer", "default": 10},
                "max_recent_events": {"type": "integer", "default": 20},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="context_after_turn",
        description="Persist a completed turn and return compact context for the next turn.",
        inputSchema={
            "type": "object",
            "required": ["role", "content"],
            "properties": {
                "role": {"type": "string"},
                "content": {"type": "string"},
                "query": {"type": "string"},
                "source": {"type": "string", "default": "mcp"},
                "session_id": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "max_recent_events": {"type": "integer", "default": 20},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="subagent_cleanup",
        description="Finalize a subagent session and return its handoff bundle.",
        inputSchema={
            "type": "object",
            "required": ["parent_session_id", "subagent_session_id", "summary"],
            "properties": {
                "parent_session_id": {"type": "string"},
                "subagent_session_id": {"type": "string"},
                "summary": {"type": "string"},
                "query": {"type": "string", "default": "subagent handoff"},
                "limit": {"type": "integer", "default": 10},
            },
            "additionalProperties": False,
        },
    ),
]


# ------------------------------------------------------------------
# Server lifecycle
# ------------------------------------------------------------------

class ZaxyMCPServer:
    """MCP server wiring for Zaxy memory operations.

    Args:
        eventloom_path: Directory containing .jsonl logs.
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
    """

    def __init__(
        self,
        eventloom_path: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self._admin_token = settings.mcp_admin_token
        self._default_session_id = validate_session_id(settings.eventloom_thread)
        self._lifecycle_capture_enabled = settings.mcp_lifecycle_capture_enabled
        self._workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._initialized_workspaces: dict[tuple[str, str], WorkspaceProfile] = {}
        self._initialized_instruction_signatures: dict[tuple[str, str], str] = {}
        self._eventloom_path = eventloom_path or settings.eventloom_path
        self.session_manager = SessionManager(base_path=self._eventloom_path)
        self.refs = MemoryRefStore(self._eventloom_path)
        self._neo4j_uri = neo4j_uri or settings.neo4j_uri
        self._neo4j_user = neo4j_user or settings.neo4j_user
        self._neo4j_password = neo4j_password or settings.neo4j_password
        self.local_neo4j = LocalNeo4jRuntime(
            uri=self._neo4j_uri,
            user=self._neo4j_user,
            password=self._neo4j_password,
            enabled=settings.neo4j_auto_start and settings.zaxy_env.lower() != "production",
            image=settings.neo4j_auto_start_image,
            container_name=settings.neo4j_auto_start_container,
        )
        self.graph = GraphStore(
            self._neo4j_uri,
            self._neo4j_user,
            self._neo4j_password,
            ca_cert=settings.neo4j_ca_cert,
            trust_all=settings.neo4j_trust_all,
        )
        self.tracer = MemoryTracer(
            base_url=settings.pathlight_url,
            project_id=settings.pathlight_project_id,
            disabled=not settings.pathlight_enabled,
        )
        self._retention_policy = build_retention_policy(settings)
        self.context_assembly_policy = ContextAssemblyPolicy(
            verbatim_enabled=settings.context_verbatim_enabled,
            verbatim_slots=settings.context_verbatim_slots,
        )

    async def setup(self) -> None:
        """Connect to Neo4j and initialize schema."""
        self.local_neo4j.ensure_available()
        await self.graph.connect()
        await self.graph.init_schema()
        await self.tracer.connect()
        await self.ensure_session_initialized(
            self._workspace_root,
            session_id=self._default_session_id,
        )

    async def teardown(self) -> None:
        """Close connections."""
        await self.capture_session_ended(reason="teardown", status="succeeded")
        await self.graph.close()
        await self.tracer.close()

    async def _append_lifecycle_event(
        self,
        event_input: dict[str, Any],
        *,
        session_id: str,
    ) -> None:
        sid = validate_session_id(session_id)
        eventlog = self.session_manager.get(sid).eventlog
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=validate_payload(event_input["payload"]),
            thread=sid,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=sid)
        await self.tracer.trace_append(event_input["event_type"], event_input["actor"], event.seq)

    async def capture_tool_call_completed(
        self,
        *,
        tool_name: str,
        status: str,
        session_id: str,
        arguments: dict[str, Any],
        result_summary: str | None = None,
    ) -> None:
        """Append and project a redacted tool-call lifecycle event."""
        sid = validate_session_id(session_id)
        event_input = build_tool_call_completed_event(
            tool_name=tool_name,
            status=status,
            session_id=sid,
            arguments=arguments,
            result_summary=result_summary,
        )
        await self._append_lifecycle_event(event_input, session_id=sid)

    async def capture_session_ended(
        self,
        *,
        reason: str,
        status: str,
    ) -> None:
        """Append and project a session-end lifecycle event."""
        if not self._lifecycle_capture_enabled:
            return
        event_input = build_session_ended_event(
            session_id=self._default_session_id,
            reason=reason,
            status=status,
        )
        with suppress(Exception):
            await self._append_lifecycle_event(event_input, session_id=self._default_session_id)

    async def ensure_session_initialized(
        self,
        path: str | Path,
        *,
        session_id: str,
    ) -> WorkspaceProfile:
        """Idempotently append and project a workspace genesis event."""
        sid = validate_session_id(session_id)
        root = str(Path(path).resolve())
        key = (root, sid)
        cached = self._initialized_workspaces.get(key)
        if cached is not None:
            await self._ensure_workspace_instructions(root, session_id=sid)
            return cached

        eventlog = self.session_manager.get(sid).eventlog
        profile = existing_session_genesis_profile(
            eventlog.read_all(),
            root=root,
            session_id=sid,
        )
        if profile is not None:
            self._initialized_workspaces[key] = profile
            await self._ensure_workspace_instructions(root, session_id=sid)
            return profile

        event_input = build_session_genesis_event(root, session_id=sid)
        payload = validate_payload(event_input["payload"])
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=payload,
            thread=sid,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=sid)
        await self.tracer.trace_append(event_input["event_type"], event_input["actor"], event.seq)
        profile = workspace_profile_from_payload(payload)
        self._initialized_workspaces[key] = profile
        await self._ensure_workspace_instructions(root, session_id=sid)
        return profile

    async def _ensure_workspace_instructions(
        self,
        path: str | Path,
        *,
        session_id: str,
    ) -> None:
        """Idempotently append and project discovered workspace instruction summaries."""
        root = str(Path(path).resolve())
        key = (root, session_id)
        event_input = build_workspace_instruction_event(root, session_id=session_id)
        if event_input is None:
            return
        signature = str(event_input["payload"]["signature"])
        if self._initialized_instruction_signatures.get(key) == signature:
            return

        eventlog = self.session_manager.get(session_id).eventlog
        existing_signature = existing_workspace_instructions_signature(
            eventlog.read_all(),
            root=root,
            session_id=session_id,
        )
        if existing_signature == signature:
            self._initialized_instruction_signatures[key] = signature
            return
        if existing_signature is not None:
            event_input = mark_workspace_instruction_event_updated(
                event_input,
                previous_signature=existing_signature,
            )

        payload = validate_payload(event_input["payload"])
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=payload,
            thread=session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_input["event_type"], event_input["actor"], event.seq)
        self._initialized_instruction_signatures[key] = signature

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def handle_memory_append(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_append tool call."""
        event_type = arguments["event_type"]
        actor = arguments["actor"]
        payload = validate_payload(arguments.get("payload", {}))
        session_id = self._session_id_from_arguments(
            arguments,
            default=self._default_session_id,
        )

        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(event_type, actor=actor, payload=payload, thread=session_id)

        # Project to graph
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_type, actor, event.seq)

        return [TextContent(type="text", text=json.dumps({"seq": event.seq, "hash": event.hash}))]

    async def handle_memory_query(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_query tool call."""
        query = validate_query(arguments["query"])
        temporal = arguments.get("temporal_filter")
        limit = validate_limit(arguments.get("limit"), default=10)
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)

        router = QueryRouter(
            self.graph,
            session_id=session_id,
            retention_policy=self._retention_policy,
        )
        results = await router.query(query, temporal_point=temporal, limit=limit)
        await self.tracer.trace_query(query, len(results), 0.0, temporal)

        output = [
            {
                "content": r.content,
                "source": r.source,
                "score": r.score,
                "valid_from": r.valid_from,
                "valid_to": r.valid_to,
                "citation": r.citation,
                "score_explanation": r.score_explanation,
            }
            for r in results
        ]
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_verbatim(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_verbatim source-recall tool call."""
        query = validate_query(arguments["query"])
        limit = validate_limit(arguments.get("limit"), default=10)
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)

        eventlog = self.session_manager.get(session_id).eventlog
        hits = VerbatimIndex.from_event_logs([eventlog]).query(query, limit=limit)
        output = [
            {
                "content": hit.content,
                "source": "verbatim",
                "score": hit.score,
                "citation": hit.citation,
                "source_kind": hit.source_kind,
                "metadata": hit.metadata,
            }
            for hit in hits
        ]
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_feedback(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_feedback tool call."""
        feedback = _normalize_feedback(arguments["feedback"])
        entity_name = _required_text(arguments["entity_name"], "entity_name")
        entity_type = _required_text(arguments["entity_type"], "entity_type")
        actor = _optional_text(arguments.get("actor")) or "zaxy"
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)

        payload: dict[str, Any] = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "query": _optional_text(arguments.get("query")),
            "source": _optional_text(arguments.get("source")) or "mcp",
            "score": arguments.get("score"),
            "citation": _optional_text(arguments.get("citation")),
            "reason": _optional_text(arguments.get("reason")),
        }
        event_type = "memory.feedback"
        if feedback in {"used", "helpful"}:
            event_type = "memory.reinforced"
            importance = arguments.get("importance")
            if importance is not None:
                payload["importance"] = max(0.0, min(1.0, float(importance)))
        else:
            payload["feedback"] = feedback
        payload = {key: value for key, value in payload.items() if value is not None}

        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(
            event_type,
            actor=actor,
            payload=validate_payload(payload),
            thread=session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_type, actor, event.seq)

        return [
            TextContent(
                type="text",
                text=json.dumps({"seq": event.seq, "hash": event.hash, "event_type": event_type}),
            )
        ]

    async def handle_memory_replay(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_replay tool call."""
        self._require_admin(arguments)
        session_id = self._session_id_from_arguments(arguments)
        from_seq = validate_from_seq(arguments.get("from_seq"))

        replay = self.session_manager.replay(session_id, from_seq=from_seq)
        events = replay.events[:MAX_REPLAY_EVENTS]

        output = {
            "integrity": replay.integrity.model_dump(),
            "events": [e.model_dump() for e in events],
            "truncated": len(replay.events) > len(events),
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_invalidate(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_invalidate tool call."""
        self._require_admin(arguments)
        name = arguments["entity_name"]
        entity_type = arguments["entity_type"]
        invalid_at = arguments["invalid_at"]

        await self.graph.invalidate_entity(
            name,
            entity_type,
            invalid_at,
            session_id=self._session_id_from_arguments(arguments, default=self._default_session_id),
        )
        return [TextContent(type="text", text=json.dumps({"status": "invalidated"}))]

    async def handle_memory_capabilities(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_capabilities tool call."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        manifest = build_memory_capabilities(
            eventloom_path=self._eventloom_path,
            session_id=session_id,
            workspace_root=self._workspace_root,
            current_task=_optional_text(arguments.get("current_task")),
        )
        return [TextContent(type="text", text=json.dumps(manifest, indent=2))]

    async def handle_context_assemble(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle context_assemble tool call."""
        query = validate_query(arguments["query"])
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        replay_from_seq = validate_from_seq(arguments.get("replay_from_seq"))
        limit = validate_limit(arguments.get("limit"), default=10)
        max_recent_events = validate_limit(arguments.get("max_recent_events"), default=20)

        output = await self._assemble_context_payload(
            query=query,
            session_id=session_id,
            replay_from_seq=replay_from_seq,
            limit=limit,
            max_recent_events=max_recent_events,
        )
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_checkout(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_checkout tool call."""
        query = validate_query(arguments["query"])
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        replay_from_seq = validate_from_seq(arguments.get("replay_from_seq"))
        limit = validate_limit(arguments.get("limit"), default=10)
        max_recent_events = validate_limit(arguments.get("max_recent_events"), default=20)
        ref = _optional_text(arguments.get("ref"))
        resolved_ref = self._resolve_checkout_ref(ref, session_id=session_id)
        checkout_session_id = resolved_ref.session_id if resolved_ref is not None else session_id

        assembly = await self._assemble_context_payload(
            query=query,
            session_id=checkout_session_id,
            replay_from_seq=replay_from_seq,
            limit=limit,
            max_recent_events=max_recent_events,
            as_of_seq=resolved_ref.target_seq if resolved_ref is not None else None,
        )
        output = _memory_checkout_payload(query=query, assembly=assembly, ref=resolved_ref)
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_context_after_turn(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle context_after_turn tool call."""
        role = arguments["role"]
        content = arguments["content"]
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        source = arguments.get("source", "mcp")
        query = validate_query(arguments.get("query") or content)
        limit = validate_limit(arguments.get("limit"), default=10)
        max_recent_events = validate_limit(arguments.get("max_recent_events"), default=20)

        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(
            "transcript.turn",
            actor=role,
            payload={"role": role, "content": content, "source": source},
            thread=session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append("transcript.turn", role, event.seq)

        output = await self._assemble_context_payload(
            query=query,
            session_id=session_id,
            replay_from_seq=1,
            limit=limit,
            max_recent_events=max_recent_events,
        )
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_subagent_cleanup(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle subagent_cleanup tool call."""
        parent_session_id = validate_session_id(arguments["parent_session_id"])
        subagent_session_id = validate_session_id(arguments["subagent_session_id"])
        summary_text = str(arguments["summary"])
        query = validate_query(arguments.get("query") or "subagent handoff")
        limit = validate_limit(arguments.get("limit"), default=10)

        if remote_session_scope.get() is not None:
            self._session_id_from_arguments({"session_id": subagent_session_id})

        eventlog = self.session_manager.get(subagent_session_id).eventlog
        event = eventlog.append(
            "subagent.cleaned",
            actor="zaxy",
            payload={
                "parent_session_id": parent_session_id,
                "subagent_session_id": subagent_session_id,
                "summary": summary_text,
            },
            thread=subagent_session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=subagent_session_id)
        await self.tracer.trace_append("subagent.cleaned", "zaxy", event.seq)
        lifecycle = build_subagent_completed_event(
            parent_session_id=parent_session_id,
            subagent_session_id=subagent_session_id,
            status="succeeded",
            summary=summary_text,
        )
        await self._append_lifecycle_event(lifecycle, session_id=subagent_session_id)

        assembly = await self._assemble_context_payload(
            query=query,
            session_id=subagent_session_id,
            replay_from_seq=1,
            limit=limit,
            max_recent_events=20,
        )
        replay = self.session_manager.replay(subagent_session_id, from_seq=1)
        output = {
            **assembly,
            "summary": self.session_manager.handoff_summary(subagent_session_id),
            "integrity_ok": bool(getattr(getattr(replay, "integrity", None), "ok", False)),
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def _assemble_context_payload(
        self,
        *,
        query: str,
        session_id: str,
        replay_from_seq: int,
        limit: int,
        max_recent_events: int,
        as_of_seq: int | None = None,
    ) -> dict[str, Any]:
        replay = self.session_manager.replay(session_id, from_seq=replay_from_seq)
        events = list(replay.events)
        if as_of_seq is not None:
            events = [event for event in events if event.seq <= as_of_seq]
        compacted = len(events) > max_recent_events
        recent_events = events[-max_recent_events:] if compacted else events
        router = QueryRouter(
            self.graph,
            session_id=session_id,
            retention_policy=self._retention_policy,
        )
        results = await router.query(query, limit=limit)
        graph_contexts = [_context_from_query_result(result) for result in results]
        verbatim_hits = (
            VerbatimIndex.from_event_logs(
                [self.session_manager.get(session_id).eventlog]
            ).query(query, limit=limit)
            if self.context_assembly_policy.should_query_verbatim(limit=limit)
            else []
        )
        verbatim_contexts = [
            Context(
                content=hit.content,
                source="verbatim",
                score=hit.score,
                metadata={
                    "citation": hit.citation,
                    "source_kind": hit.source_kind,
                    **hit.metadata,
                },
            )
            for hit in verbatim_hits
        ]
        contexts = self.context_assembly_policy.assemble(
            graph_contexts,
            verbatim_contexts,
            limit=limit,
        )
        if as_of_seq is not None:
            contexts = _contexts_as_of_seq(contexts, as_of_seq)
        working_set = build_working_set(recent_events, contexts)
        await self.tracer.trace_query(query, len(results), 0.0, None)
        return {
            "session_id": session_id,
            "prompt": _format_prompt(recent_events, contexts, working_set=working_set),
            "contexts": [_context_payload(context) for context in contexts],
            "replay_event_count": len(recent_events),
            "compacted": compacted,
            "assembly_policy": self.context_assembly_policy.describe(),
            "context_counts": context_counts(contexts, replay_count=len(recent_events)),
            "working_set": working_set.to_dict(),
        }

    def _require_admin(self, arguments: dict[str, Any]) -> None:
        """Require an admin token for destructive or bulk-read tools when configured."""
        if self._admin_token and arguments.get("admin_token") != self._admin_token:
            raise PermissionError("admin_token is required for this tool")

    def _session_id_from_arguments(
        self,
        arguments: dict[str, Any],
        default: str | None = None,
    ) -> str:
        """Resolve and enforce a tool session ID.

        Remote SSE requests run with a context-scoped session. They may omit a
        session argument, but they cannot cross into another session.
        """
        explicit = arguments.get("session_id") or arguments.get("thread")
        requested = explicit or default
        scoped = remote_session_scope.get()
        if scoped is not None:
            if explicit is None:
                return scoped
            safe_requested = validate_session_id(explicit)
            if safe_requested != scoped:
                raise PermissionError("session scope does not permit this session_id")
            return safe_requested
        if requested is None:
            raise ValueError("session_id is required")
        return validate_session_id(requested)

    def _enforce_optional_session(self, arguments: dict[str, Any]) -> None:
        """Reject cross-session optional tool arguments under remote scope."""
        if (
            remote_session_scope.get() is not None
            and ("session_id" in arguments or "thread" in arguments)
        ):
            self._session_id_from_arguments(arguments)

    def _resolve_checkout_ref(self, ref: str | None, *, session_id: str) -> MemoryRef | None:
        if ref is None:
            return None
        if ref == "HEAD":
            replay = self.session_manager.replay(session_id, from_seq=1)
            events = list(replay.events)
            if not events:
                return None
            latest = events[-1]
            return MemoryRef(
                name="HEAD",
                session_id=session_id,
                target_seq=latest.seq,
                target_hash=latest.hash,
                ref_type="head",
                updated_at=latest.timestamp,
            )
        resolved = self.refs.resolve(ref)
        if resolved is None:
            raise ValueError(f"Unknown memory ref: {ref}")
        return resolved


class JWTDecoder(Protocol):
    def __call__(self, token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
        """Decode and validate a JWT."""


class JWKSClient(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any:
        """Return the signing key for a JWT."""


class MCPTransportAuth:
    """Authenticate and scope remote MCP/SSE HTTP requests."""

    def __init__(
        self,
        token: str | None,
        session_header: str = "x-zaxy-session-id",
        oidc_issuer: str | None = None,
        oidc_audience: str | None = None,
        oidc_jwks_url: str | None = None,
        oidc_required_scope: str = "zaxy:mcp",
        oidc_session_claim: str = "zaxy_session",
        jwt_client: JWKSClient | None = None,
        jwt_decoder: JWTDecoder | None = None,
    ) -> None:
        self._token = token
        self._session_header = session_header.casefold()
        self._oidc_issuer = oidc_issuer
        self._oidc_audience = oidc_audience
        self._oidc_jwks_url = oidc_jwks_url
        self._oidc_required_scope = oidc_required_scope
        self._oidc_session_claim = oidc_session_claim
        self._jwt_client = jwt_client
        self._jwt_decoder = jwt_decoder or jwt.decode

    def authorize(self, headers: Mapping[str, str]) -> str:
        """Validate request headers and return the remote session scope."""
        normalized = {key.casefold(): value for key, value in headers.items()}
        if self._oidc_enabled:
            return self._authorize_oidc(normalized)
        if self._token is not None:
            header = normalized.get("authorization")
            if not header or not header.startswith("Bearer "):
                raise PermissionError("Authorization bearer token is required")
            supplied = header.removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied, self._token):
                raise PermissionError("Authorization bearer token is invalid")
        return validate_session_id(normalized.get(self._session_header, "default"))

    @property
    def _oidc_enabled(self) -> bool:
        return bool(self._oidc_issuer and self._oidc_audience and self._oidc_jwks_url)

    def _authorize_oidc(self, headers: Mapping[str, str]) -> str:
        header = headers.get("authorization")
        if not header or not header.startswith("Bearer "):
            raise PermissionError("Authorization bearer token is required")
        token = header.removeprefix("Bearer ").strip()
        if not token:
            raise PermissionError("Authorization bearer token is required")
        try:
            jwks_client = self._jwt_client or jwt.PyJWKClient(str(self._oidc_jwks_url))
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            claims = self._jwt_decoder(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=self._oidc_audience,
                issuer=self._oidc_issuer,
                options={"require": ["exp", "iat", "iss", "aud"]},
            )
        except Exception as exc:
            raise PermissionError("Authorization bearer token is invalid") from exc

        scopes = _claim_values(claims.get("scope")) | _claim_values(claims.get("scp"))
        if self._oidc_required_scope and self._oidc_required_scope not in scopes:
            raise PermissionError("Authorization bearer token missing required scope")

        session_claim = claims.get(self._oidc_session_claim)
        if not isinstance(session_claim, str) or not session_claim:
            raise PermissionError("Authorization bearer token missing session claim")
        return validate_session_id(session_claim)


class RemoteRateLimitError(PermissionError):
    """Raised when a remote session exceeds its request rate limit."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("remote MCP rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


class RemoteRequestGuard:
    """Authorize, rate-limit, and audit remote MCP/SSE HTTP requests."""

    def __init__(
        self,
        *,
        auth: MCPTransportAuth,
        rate_limit_enabled: bool,
        rate_limit_requests: int,
        rate_limit_window_seconds: int,
        audit_enabled: bool,
        audit_path: Path | str,
    ) -> None:
        self._auth = auth
        self._limiter = SessionRateLimiter(
            enabled=rate_limit_enabled,
            max_requests=rate_limit_requests,
            window_seconds=rate_limit_window_seconds,
        )
        self._audit = AuditEventExporter(path=Path(audit_path), enabled=audit_enabled)

    def authorize(
        self,
        headers: Mapping[str, str],
        *,
        route: str,
        method: str,
        client_host: str | None,
    ) -> str:
        """Return authorized session ID or raise an auth/rate-limit error."""
        try:
            session_id = self._auth.authorize(headers)
        except (PermissionError, ValueError) as exc:
            self._write_audit(
                session_id=None,
                route=route,
                method=method,
                outcome="denied_auth",
                reason=str(exc),
                client_host=client_host,
            )
            raise

        decision = self._limiter.check(session_id)
        if not decision.allowed:
            get_metrics().record_rate_limit_denial(session_id)
            self._write_audit(
                session_id=session_id,
                route=route,
                method=method,
                outcome="denied_rate_limit",
                reason="rate limit exceeded",
                client_host=client_host,
            )
            raise RemoteRateLimitError(decision.retry_after_seconds)

        self._write_audit(
            session_id=session_id,
            route=route,
            method=method,
            outcome="allowed",
            reason=None,
            client_host=client_host,
        )
        return session_id

    def _write_audit(
        self,
        *,
        session_id: str | None,
        route: str,
        method: str,
        outcome: str,
        reason: str | None,
        client_host: str | None,
    ) -> None:
        self._audit.write(
            RemoteAuditEvent(
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                session_id=session_id,
                route=route,
                method=method,
                outcome=outcome,  # type: ignore[arg-type]
                reason=reason,
                client_host=client_host,
            )
        )


def _memory_checkout_payload(
    *,
    query: str,
    assembly: dict[str, Any],
    ref: MemoryRef | None = None,
) -> dict[str, Any]:
    contexts = sorted(
        [context for context in assembly.get("contexts", []) if isinstance(context, dict)],
        key=lambda context: _checkout_rank(context, query),
        reverse=True,
    )
    current_facts = [
        _checkout_fact_payload(context)
        for context in contexts
        if context.get("valid_to") is None
    ]
    evidence = [
        _checkout_evidence_payload(context)
        for context in contexts
        if context.get("citation")
    ]
    provenance = [
        _checkout_provenance_payload(context)
        for context in contexts
        if context.get("citation")
    ]
    warnings: list[str] = []
    if assembly.get("compacted") is True:
        warnings.append("Recent replay was compacted to fit the checkout budget.")
    if current_facts and not evidence:
        warnings.append("Checkout contains current facts without Eventloom citations.")
    retention = {
        "policy": "current_only",
        "superseded_contexts_excluded": sum(
            1
            for context in contexts
            if context.get("valid_to") is not None
        ),
    }
    diagnostics = _checkout_diagnostics_payload(
        contexts=contexts,
        current_facts=current_facts,
        evidence=evidence,
        retention=retention,
        warnings=warnings,
    )
    guidance = _checkout_guidance_payload(
        query=query,
        current_facts=current_facts,
        retention=retention,
        evidence=evidence,
    )
    quality = _checkout_quality_payload(
        diagnostics=diagnostics,
        guidance=guidance,
        warnings=warnings,
    )
    return {
        **assembly,
        "query": query,
        "ref": ref.to_dict() if ref is not None else None,
        "prompt": _format_memory_checkout_prompt(
            query=query,
            assembly_prompt=str(assembly.get("prompt", "")),
            current_facts=current_facts,
            evidence=evidence,
            quality=quality,
            guidance=guidance,
            diagnostics=diagnostics,
        ),
        "current_facts": current_facts,
        "evidence": evidence,
        "provenance": provenance,
        "retention": retention,
        "warnings": warnings,
        "guidance": guidance,
        "quality": quality,
        "diagnostics": diagnostics,
    }


def _format_memory_checkout_prompt(
    *,
    query: str,
    assembly_prompt: str,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    quality: dict[str, Any],
    guidance: dict[str, Any],
    diagnostics: dict[str, Any],
) -> str:
    lines = ["# Memory Checkout", f"Query: {query}", "", "## Current Facts"]
    if current_facts:
        for fact in current_facts:
            citation = f" ({fact['citation']})" if fact.get("citation") else ""
            lines.append(f"- {fact['content']}{citation}")
    else:
        lines.append("- No current facts were retrieved.")
    lines.extend(["", "## Evidence"])
    if evidence:
        for item in evidence:
            lines.append(f"- {item['citation']}: {item['content']}")
    else:
        lines.append("- No cited evidence was retrieved.")
    lines.extend(["", "## Checkout Quality"])
    lines.append(f"- Answerability: {quality.get('answerability')}")
    lines.append(f"- Confidence: {quality.get('confidence')}")
    for reason in quality.get("reasons", []):
        lines.append(f"- Reason: {reason}")
    required_action = quality.get("required_action")
    if isinstance(required_action, dict):
        lines.append(
            "- Required action: "
            f"{required_action.get('tool')}({required_action.get('query')!r})"
        )
    else:
        lines.append("- Required action: none")
    lines.extend(["", "## Checkout Guidance"])
    for item in guidance.get("trust", []):
        lines.append(f"- Trust: {item}")
    for item in guidance.get("ignore", []):
        lines.append(f"- Ignore: {item}")
    recommended_next_call = guidance.get("recommended_next_call")
    if isinstance(recommended_next_call, dict):
        lines.append(
            "- Suggested next call: "
            f"{recommended_next_call.get('tool')}({recommended_next_call.get('query')!r})"
        )
    feedback = guidance.get("feedback")
    if isinstance(feedback, dict) and feedback.get("payloads"):
        lines.append(f"- Feedback: call {feedback.get('tool')} with a listed payload after use.")
    source_lanes = diagnostics.get("source_lanes")
    lines.extend(["", "## Checkout Diagnostics"])
    lines.append(f"- Source lanes: {_format_source_lanes(source_lanes)}")
    lines.append(f"- Citations: {diagnostics.get('citation_count', 0)}")
    lines.append(f"- Current facts: {diagnostics.get('current_fact_count', 0)}")
    lines.append(
        f"- Superseded contexts excluded: {diagnostics.get('superseded_contexts_excluded', 0)}"
    )
    if diagnostics.get("feedback_recommended"):
        lines.append(
            "- Feedback: call "
            f"{diagnostics.get('feedback_tool', 'memory_feedback')} after using cited context."
        )
    lines.extend(["", assembly_prompt])
    return "\n".join(lines).strip()


def _checkout_fact_payload(context: dict[str, Any]) -> dict[str, Any]:
    metadata = context.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    fact: dict[str, Any] = {
        "content": context.get("content"),
        "source": context.get("source"),
        "score": context.get("score"),
        "citation": context.get("citation"),
        "valid_from": context.get("valid_from"),
        "valid_to": context.get("valid_to"),
        "source_lane": _checkout_source_lane_payload(context),
    }
    for key in ("entity_name", "entity_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            fact[key] = value
    return fact


def _checkout_guidance_payload(
    *,
    query: str,
    current_facts: list[dict[str, Any]],
    retention: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    feedback_payloads = [
        payload
        for fact in current_facts
        if (payload := _checkout_feedback_payload(fact, query)) is not None
    ][:3]
    trust = [
        "Use current_facts as the primary working memory for this turn.",
        "Use cited evidence and provenance when making claims about remembered context.",
    ]
    ignore = [
        "Do not treat superseded contexts as current facts.",
        "Do not rely on uncited facts without checking memory again or asking the user.",
    ]
    if not evidence:
        trust.append("Treat this checkout as low-confidence because it has no cited evidence.")
    if retention.get("superseded_contexts_excluded", 0):
        ignore.append("Superseded contexts were excluded from current_facts but remain auditable.")
    return {
        "trust": trust,
        "ignore": ignore,
        "recommended_next_call": {
            "tool": "memory_checkout",
            "query": f"current decisions, blockers, and next actions for: {query}",
            "reason": (
                "Refresh memory before major follow-up work, after compaction/resume, "
                "or when task scope changes."
            ),
        },
        "feedback": {
            "tool": "memory_feedback",
            "when": "After cited context materially informs a response.",
            "payloads": feedback_payloads,
        },
    }


def _checkout_quality_payload(
    *,
    diagnostics: dict[str, Any],
    guidance: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    current_fact_count = _int_metric(diagnostics.get("current_fact_count"))
    citation_count = _int_metric(diagnostics.get("citation_count"))
    superseded_excluded = _int_metric(diagnostics.get("superseded_contexts_excluded"))
    warning_count = _int_metric(diagnostics.get("warning_count"))
    reasons: list[str] = []
    if current_fact_count and citation_count:
        reasons.append("Retrieved current facts with Eventloom citations.")
    elif current_fact_count:
        reasons.append("Retrieved current facts, but they lack Eventloom citations.")
    else:
        reasons.append("No current facts were retrieved.")
    if superseded_excluded:
        reasons.append("Superseded contexts were excluded from current facts.")
    if warning_count:
        reasons.append("Checkout contains warnings that reduce confidence.")
    confidence = 0.25
    confidence += min(current_fact_count, 2) * 0.2
    confidence += min(citation_count, 2) * 0.15
    if superseded_excluded:
        confidence += 0.07
    confidence -= min(0.25, warning_count * 0.12)
    confidence = round(max(0.0, min(0.95, confidence)), 2)
    recommended_next_call = guidance.get("recommended_next_call")
    required_action = recommended_next_call if isinstance(recommended_next_call, dict) else None
    if current_fact_count and citation_count and confidence >= 0.75:
        answerability = "answer_from_memory"
        required_action = None
    elif current_fact_count or citation_count:
        answerability = "refresh_recommended"
    else:
        answerability = "ask_user"
    return {
        "answerability": answerability,
        "confidence": confidence,
        "reasons": reasons,
        "required_action": required_action,
    }


def _int_metric(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _checkout_feedback_payload(fact: dict[str, Any], query: str) -> dict[str, Any] | None:
    citation = fact.get("citation")
    if not isinstance(citation, str) or not citation:
        return None
    entity_name = fact.get("entity_name")
    entity_type = fact.get("entity_type")
    payload: dict[str, Any] = {
        "entity_name": entity_name if isinstance(entity_name, str) and entity_name else fact.get("content"),
        "entity_type": entity_type if isinstance(entity_type, str) and entity_type else "memory",
        "feedback": "used",
        "actor": "assistant",
        "query": query,
        "source": fact.get("source"),
        "score": fact.get("score"),
        "citation": citation,
        "importance": 0.6,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _checkout_evidence_payload(context: dict[str, Any]) -> dict[str, Any]:
    citation = context.get("citation") if isinstance(context.get("citation"), str) else None
    seq, event_hash = _citation_event_identity(citation)
    return {
        "citation": citation,
        "content": context.get("content"),
        "source": context.get("source"),
        "source_lane": _checkout_source_lane_payload(context),
        "score": context.get("score"),
        "event_seq": seq,
        "event_hash": event_hash,
    }


def _checkout_provenance_payload(context: dict[str, Any]) -> dict[str, Any]:
    citation = context.get("citation") if isinstance(context.get("citation"), str) else None
    seq, event_hash = _citation_event_identity(citation)
    return {
        "citation": citation,
        "event_seq": seq,
        "event_hash": event_hash,
        "source": context.get("source"),
        "source_lane": _checkout_source_lane_payload(context),
        "valid_from": context.get("valid_from"),
        "valid_to": context.get("valid_to"),
    }


def _checkout_diagnostics_payload(
    *,
    contexts: list[dict[str, Any]],
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    retention: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    source_lanes: dict[str, int] = {}
    for context in contexts:
        lane = _checkout_source_lane_payload(context)
        source_lanes[lane] = source_lanes.get(lane, 0) + 1
    return {
        "source_lanes": source_lanes,
        "citation_count": len(evidence),
        "current_fact_count": len(current_facts),
        "superseded_contexts_excluded": retention.get("superseded_contexts_excluded", 0),
        "warning_count": len(warnings),
        "feedback_recommended": bool(evidence),
        "feedback_tool": "memory_feedback",
        "feedback_reason": "Reinforce cited context if it materially informed the next response.",
    }


def _checkout_source_lane_payload(context: dict[str, Any]) -> str:
    metadata = context.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    lane = metadata.get("assembly_lane")
    if isinstance(lane, str) and lane:
        return lane
    source = context.get("source")
    if source in {"verbatim", "packet_memory", "projection", "eventloom"}:
        return str(source)
    return "graph"


def _format_source_lanes(source_lanes: Any) -> str:
    if not isinstance(source_lanes, dict) or not source_lanes:
        return "none"
    return ", ".join(f"{lane}={count}" for lane, count in source_lanes.items())


def _checkout_rank(context: dict[str, Any], query: str) -> tuple[float, int, str, float]:
    query_tokens = _checkout_tokens(query)
    content_tokens = _checkout_tokens(str(context.get("content") or ""))
    overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
    metadata = context.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    entity_type = metadata.get("entity_type")
    type_priority = 1 if entity_type in {"task", "decision", "goal", "memory"} else 0
    score = context.get("score")
    numeric_score = float(score) if isinstance(score, int | float) else 0.0
    return (overlap, type_priority, str(context.get("valid_from") or ""), numeric_score)


def _contexts_as_of_seq(contexts: list[Context], as_of_seq: int) -> list[Context]:
    filtered = []
    for context in contexts:
        citation = _result_citation(context)
        seq, _event_hash = _citation_event_identity(citation)
        if seq is None or seq <= as_of_seq:
            filtered.append(context)
    return filtered


def _checkout_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 2}


def _citation_event_identity(citation: str | None) -> tuple[int | None, str | None]:
    if not citation:
        return None, None
    event_seq: int | None = None
    event_hash: str | None = None
    if "/events/" in citation:
        tail = citation.split("/events/", 1)[1]
        seq_text = tail.split("#", 1)[0].split("/", 1)[0]
        if seq_text.isdigit():
            event_seq = int(seq_text)
    if "#" in citation:
        fragment = citation.rsplit("#", 1)[1]
        event_hash = fragment or None
    return event_seq, event_hash


def _format_prompt(events: list[Any], results: list[Any], *, working_set: Any | None = None) -> str:
    lines = []
    if working_set is not None:
        lines.extend([format_working_set(working_set), ""])
    lines.append("# Recent Events")
    for event in events:
        lines.append(f"[{event.seq}] {event.type} by {event.actor}")
        content = _event_content(event)
        if content:
            lines.append(content)
    lines.append("")
    lines.append("# Retrieved Context")
    for result in results:
        citation_value = _result_citation(result)
        citation = f" ({citation_value})" if citation_value else ""
        lines.append(f"- {result.content}{citation}")
    return "\n".join(lines).strip()


def _event_content(event: Any) -> str:
    payload = getattr(event, "payload", {})
    if not isinstance(payload, dict):
        return ""
    parts = [
        str(payload[key])
        for key in ("title", "summary", "content", "text", "decision", "task")
        if payload.get(key)
    ]
    return " ".join(parts)


def _context_payload(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", None) or {}
    return {
        "content": result.content,
        "source": result.source,
        "score": result.score,
        "valid_from": result.valid_from,
        "valid_to": result.valid_to,
        "citation": _result_citation(result),
        "score_explanation": metadata.get("score_explanation")
        or getattr(result, "score_explanation", None),
        "metadata": metadata,
    }


def _context_from_query_result(result: Any) -> Context:
    metadata: dict[str, Any] = {}
    citation = getattr(result, "citation", None)
    if citation:
        metadata["citation"] = citation
    score_explanation = getattr(result, "score_explanation", None)
    if score_explanation:
        metadata["score_explanation"] = score_explanation
    entity_name = getattr(result, "entity_name", None)
    if isinstance(entity_name, str) and entity_name:
        metadata["entity_name"] = entity_name
    entity_type = getattr(result, "entity_type", None)
    if isinstance(entity_type, str) and entity_type:
        metadata["entity_type"] = entity_type
    return Context(
        content=result.content,
        source=result.source,
        score=result.score,
        valid_from=result.valid_from,
        valid_to=result.valid_to,
        metadata=metadata or None,
    )


def _result_citation(result: Any) -> str | None:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        metadata_citation = metadata.get("citation")
        if isinstance(metadata_citation, str):
            return metadata_citation
    citation = getattr(result, "citation", None)
    return citation if isinstance(citation, str) else None


def _claim_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {part for part in value.split() if part}
    if isinstance(value, list):
        return {str(part) for part in value if part}
    return set()


def _normalize_feedback(feedback: object) -> str:
    normalized = str(feedback).casefold().strip()
    if normalized not in {"used", "helpful", "irrelevant"}:
        raise ValueError("feedback must be one of: used, helpful, irrelevant")
    return normalized


def _required_text(value: object, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

server: ZaxyMCPServer | None = None


async def _dispatch_tool_call(
    active_server: ZaxyMCPServer,
    name: str,
    arguments: dict[str, Any],
) -> list[TextContent]:
    if name == "memory_append":
        return await active_server.handle_memory_append(arguments)
    if name == "memory_query":
        return await active_server.handle_memory_query(arguments)
    if name == "memory_verbatim":
        return await active_server.handle_memory_verbatim(arguments)
    if name == "memory_feedback":
        return await active_server.handle_memory_feedback(arguments)
    if name == "memory_replay":
        return await active_server.handle_memory_replay(arguments)
    if name == "memory_invalidate":
        return await active_server.handle_memory_invalidate(arguments)
    if name == "memory_capabilities":
        return await active_server.handle_memory_capabilities(arguments)
    if name == "memory_checkout":
        return await active_server.handle_memory_checkout(arguments)
    if name == "context_assemble":
        return await active_server.handle_context_assemble(arguments)
    if name == "context_after_turn":
        return await active_server.handle_context_after_turn(arguments)
    if name == "subagent_cleanup":
        return await active_server.handle_subagent_cleanup(arguments)
    raise ValueError(f"Unknown tool: {name}")


def _capture_session_id(active_server: ZaxyMCPServer, arguments: dict[str, Any]) -> str:
    fallback = _default_capture_session_id(active_server)
    try:
        resolved = active_server._session_id_from_arguments(
            arguments,
            default=fallback,
        )
        if inspect.isawaitable(resolved):
            if inspect.iscoroutine(resolved):
                resolved.close()
            return fallback
        if isinstance(resolved, str):
            return resolved
        return fallback
    except Exception:
        return fallback


def _default_capture_session_id(active_server: ZaxyMCPServer) -> str:
    default = vars(active_server).get("_default_session_id", "default")
    return default if isinstance(default, str) else "default"


async def _capture_tool_call_best_effort(
    active_server: ZaxyMCPServer,
    *,
    name: str,
    arguments: dict[str, Any],
    session_id: str,
    status: str,
    result_summary: str | None,
) -> None:
    if vars(active_server).get("_lifecycle_capture_enabled", True) is False:
        return
    with suppress(Exception):
        await active_server.capture_tool_call_completed(
            tool_name=name,
            status=status,
            session_id=session_id,
            arguments=arguments,
            result_summary=result_summary,
        )


def _tool_result_summary(result: list[TextContent]) -> str:
    if not result:
        return "0 content items"
    first = result[0]
    text = getattr(first, "text", "")
    if text:
        return str(text)[:240]
    return f"{len(result)} content item{'s' if len(result) != 1 else ''}"


async def main() -> None:
    """Run the MCP stdio server with graceful shutdown."""
    setup_logging()
    logger = get_logger("mcp_server")

    # Allow external configuration (e.g. from CLI) via module-level override
    active_server = server or ZaxyMCPServer()

    await active_server.setup()
    logger.info("zaxy_mcp_server_ready server=%s", get_settings().server_name)

    # Graceful shutdown on SIGTERM/SIGINT
    import signal

    shutdown_event: asyncio.Event = asyncio.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("zaxy_mcp_server_signal_received signal=%s", sig_name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)

    @app.list_tools()  # type: ignore[untyped-decorator, no-untyped-call]
    async def list_tools() -> list[Tool]:
        return TOOLS

    @app.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        session_id = _capture_session_id(active_server, arguments)
        try:
            result = await _dispatch_tool_call(active_server, name, arguments)
        except Exception:
            await _capture_tool_call_best_effort(
                active_server,
                name=name,
                arguments=arguments,
                session_id=session_id,
                status="failed",
                result_summary=None,
            )
            raise
        await _capture_tool_call_best_effort(
            active_server,
            name=name,
            arguments=arguments,
            session_id=session_id,
            status="succeeded",
            result_summary=_tool_result_summary(result),
        )
        return result

    try:
        async with stdio_server() as (read_stream, write_stream):
            run_task = asyncio.create_task(
                app.run(read_stream, write_stream, app.create_initialization_options())
            )
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            done, pending = await asyncio.wait(
                [run_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
    finally:
        logger.info("zaxy_mcp_server_shutting_down")
        await active_server.teardown()
        logger.info("zaxy_mcp_server_stopped")


async def main_sse(port: int = 8080, host: str = "127.0.0.1") -> None:
    """Run the MCP server with SSE transport over HTTP.

    This allows the server to run as a background daemon, accessible
    via HTTP/SSE endpoints for MCP clients that support remote servers.
    """
    setup_logging()
    logger = get_logger("mcp_server")
    settings = get_settings()

    active_server = server or ZaxyMCPServer()

    await active_server.setup()
    logger.info("zaxy_mcp_server_ready transport=sse host=%s port=%s", host, port)

    import signal

    shutdown_event: asyncio.Event = asyncio.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("zaxy_mcp_server_signal_received signal=%s", sig_name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)

    # SSE transport setup
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    sse_transport = SseServerTransport("/messages/")
    transport_auth = MCPTransportAuth(
        token=settings.mcp_remote_auth_token,
        session_header=settings.mcp_remote_session_header,
        oidc_issuer=settings.mcp_oidc_issuer,
        oidc_audience=settings.mcp_oidc_audience,
        oidc_jwks_url=settings.mcp_oidc_jwks_url,
        oidc_required_scope=settings.mcp_oidc_required_scope,
        oidc_session_claim=settings.mcp_oidc_session_claim,
    )
    request_guard = RemoteRequestGuard(
        auth=transport_auth,
        rate_limit_enabled=settings.mcp_rate_limit_enabled,
        rate_limit_requests=settings.mcp_rate_limit_requests,
        rate_limit_window_seconds=settings.mcp_rate_limit_window_seconds,
        audit_enabled=settings.mcp_audit_enabled,
        audit_path=settings.mcp_audit_path,
    )

    async def _sse_handler(request: Any) -> Any:
        try:
            session_id = request_guard.authorize(
                request.headers,
                route="/sse",
                method=request.method,
                client_host=getattr(request.client, "host", None),
            )
        except RemoteRateLimitError as exc:
            return PlainTextResponse(
                str(exc),
                status_code=429,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except (PermissionError, ValueError) as exc:
            return PlainTextResponse(str(exc), status_code=401)
        token = remote_session_scope.set(session_id)
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            try:
                await app.run(read_stream, write_stream, app.create_initialization_options())
            finally:
                remote_session_scope.reset(token)

    async def _messages_handler(request: Any) -> Any:
        try:
            session_id = request_guard.authorize(
                request.headers,
                route="/messages/",
                method=request.method,
                client_host=getattr(request.client, "host", None),
            )
        except RemoteRateLimitError as exc:
            return PlainTextResponse(
                str(exc),
                status_code=429,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except (PermissionError, ValueError) as exc:
            return PlainTextResponse(str(exc), status_code=401)
        token = remote_session_scope.set(session_id)
        try:
            return await sse_transport.handle_post_message(
                request.scope, request.receive, request._send
            )
        finally:
            remote_session_scope.reset(token)

    starlette_app = Starlette(
        debug=False,
        routes=[
            Route("/sse", _sse_handler),
            Route("/messages/", _messages_handler, methods=["POST"]),
        ],
    )

    import uvicorn

    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="warning")
    uvicorn_server = uvicorn.Server(config)

    serve_task = asyncio.create_task(uvicorn_server.serve())
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    done, pending = await asyncio.wait(
        [serve_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()

    logger.info("zaxy_mcp_server_shutting_down")
    await active_server.teardown()
    logger.info("zaxy_mcp_server_stopped")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
