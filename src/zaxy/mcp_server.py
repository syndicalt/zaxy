"""MCP server exposing memory tools.

Provides stdio and SSE transport for agent frameworks to interact
with Zaxy via the Model Context Protocol.

Tools exposed:
- memory_append: Append a typed event to the log.
- memory_query: Query the temporal knowledge graph.
- memory_replay: Replay events from a session.
- memory_invalidate: Mark a fact as superseded.
"""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
from collections.abc import Mapping
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from zaxy.config import get_settings
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.log import get_logger, setup_logging
from zaxy.query import QueryRouter
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
    ) -> None:
        settings = get_settings()
        self._admin_token = settings.mcp_admin_token
        self.session_manager = SessionManager(base_path=eventloom_path or settings.eventloom_path)
        self.graph = GraphStore(
            neo4j_uri or settings.neo4j_uri,
            neo4j_user or settings.neo4j_user,
            neo4j_password or settings.neo4j_password,
            ca_cert=settings.neo4j_ca_cert,
            trust_all=settings.neo4j_trust_all,
        )
        self.tracer = MemoryTracer(
            base_url=settings.pathlight_url,
            project_id=settings.pathlight_project_id,
            disabled=not settings.pathlight_enabled,
        )

    async def setup(self) -> None:
        """Connect to Neo4j and initialize schema."""
        await self.graph.connect()
        await self.graph.init_schema()
        await self.tracer.connect()

    async def teardown(self) -> None:
        """Close connections."""
        await self.graph.close()
        await self.tracer.close()

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
            default="default",
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
        session_id = self._session_id_from_arguments(arguments, default="default")

        router = QueryRouter(self.graph, session_id=session_id)
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
            session_id=self._session_id_from_arguments(arguments, default="default"),
        )
        return [TextContent(type="text", text=json.dumps({"status": "invalidated"}))]

    async def handle_context_assemble(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle context_assemble tool call."""
        query = validate_query(arguments["query"])
        session_id = self._session_id_from_arguments(arguments, default="default")
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

    async def handle_context_after_turn(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle context_after_turn tool call."""
        role = arguments["role"]
        content = arguments["content"]
        session_id = self._session_id_from_arguments(arguments, default="default")
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
    ) -> dict[str, Any]:
        replay = self.session_manager.replay(session_id, from_seq=replay_from_seq)
        events = list(replay.events)
        compacted = len(events) > max_recent_events
        recent_events = events[-max_recent_events:] if compacted else events
        router = QueryRouter(self.graph, session_id=session_id)
        results = await router.query(query, limit=limit)
        await self.tracer.trace_query(query, len(results), 0.0, None)
        return {
            "session_id": session_id,
            "prompt": _format_prompt(recent_events, results),
            "contexts": [_context_payload(result) for result in results],
            "replay_event_count": len(recent_events),
            "compacted": compacted,
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


class MCPTransportAuth:
    """Authenticate and scope remote MCP/SSE HTTP requests."""

    def __init__(
        self,
        token: str | None,
        session_header: str = "x-zaxy-session-id",
    ) -> None:
        self._token = token
        self._session_header = session_header.casefold()

    def authorize(self, headers: Mapping[str, str]) -> str:
        """Validate request headers and return the remote session scope."""
        normalized = {key.casefold(): value for key, value in headers.items()}
        if self._token is not None:
            header = normalized.get("authorization")
            if not header or not header.startswith("Bearer "):
                raise PermissionError("Authorization bearer token is required")
            supplied = header.removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied, self._token):
                raise PermissionError("Authorization bearer token is invalid")
        return validate_session_id(normalized.get(self._session_header, "default"))


def _format_prompt(events: list[Any], results: list[Any]) -> str:
    lines = ["# Recent Events"]
    for event in events:
        lines.append(f"[{event.seq}] {event.type} by {event.actor}")
        content = _event_content(event)
        if content:
            lines.append(content)
    lines.append("")
    lines.append("# Retrieved Context")
    for result in results:
        citation = f" ({result.citation})" if getattr(result, "citation", None) else ""
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
    return {
        "content": result.content,
        "source": result.source,
        "score": result.score,
        "valid_from": result.valid_from,
        "valid_to": result.valid_to,
        "citation": result.citation,
        "score_explanation": result.score_explanation,
    }


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

server: ZaxyMCPServer | None = None


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
        if name == "memory_append":
            return await active_server.handle_memory_append(arguments)
        if name == "memory_query":
            return await active_server.handle_memory_query(arguments)
        if name == "memory_replay":
            return await active_server.handle_memory_replay(arguments)
        if name == "memory_invalidate":
            return await active_server.handle_memory_invalidate(arguments)
        if name == "context_assemble":
            return await active_server.handle_context_assemble(arguments)
        if name == "context_after_turn":
            return await active_server.handle_context_after_turn(arguments)
        if name == "subagent_cleanup":
            return await active_server.handle_subagent_cleanup(arguments)
        raise ValueError(f"Unknown tool: {name}")

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
    )

    def _authorize_request(request: Any) -> str:
        return transport_auth.authorize(request.headers)

    async def _sse_handler(request: Any) -> Any:
        try:
            session_id = _authorize_request(request)
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
            session_id = _authorize_request(request)
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
