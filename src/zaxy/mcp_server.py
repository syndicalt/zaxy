"""MCP server exposing memory tools.

Provides stdio (and future SSE) transport for agent frameworks to interact
with Zaxy via the Model Context Protocol.

Tools exposed:
- memory_append: Append a typed event to the log.
- memory_query: Query the temporal knowledge graph.
- memory_replay: Replay events from a session.
- memory_invalidate: Mark a fact as superseded.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from zaxy.config import get_settings
from zaxy.event import EventLog
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.log import get_logger, setup_logging
from zaxy.query import QueryRouter
from zaxy.trace import MemoryTracer

app = Server("zaxy-memory")


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
                "thread": {"type": "string", "description": "Logical thread / session ID"},
            },
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
            },
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
            },
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
        self.eventloom_path = eventloom_path or settings.eventloom_path
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
        payload = arguments.get("payload", {})
        thread = arguments.get("thread", "default")

        log_path = f"{self.eventloom_path}/{thread}.jsonl"
        log = EventLog(log_path)
        event = log.append(event_type, actor=actor, payload=payload, thread=thread)

        # Project to graph
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction)
        await self.tracer.trace_append(event_type, actor, event.seq)

        return [TextContent(type="text", text=json.dumps({"seq": event.seq, "hash": event.hash}))]

    async def handle_memory_query(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_query tool call."""
        query = arguments["query"]
        temporal = arguments.get("temporal_filter")
        limit = arguments.get("limit", 10)

        router = QueryRouter(self.graph)
        results = await router.query(query, temporal_point=temporal, limit=limit)
        await self.tracer.trace_query(query, len(results), 0.0, temporal)

        output = [
            {
                "content": r.content,
                "source": r.source,
                "score": r.score,
                "valid_from": r.valid_from,
                "valid_to": r.valid_to,
            }
            for r in results
        ]
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_replay(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_replay tool call."""
        session_id = arguments["session_id"]
        from_seq = arguments.get("from_seq", 1)

        log_path = f"{self.eventloom_path}/{session_id}.jsonl"
        log = EventLog(log_path)
        replay = log.replay(from_seq=from_seq)

        output = {
            "integrity": replay.integrity.model_dump(),
            "events": [e.model_dump() for e in replay.events],
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_invalidate(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_invalidate tool call."""
        name = arguments["entity_name"]
        entity_type = arguments["entity_type"]
        invalid_at = arguments["invalid_at"]

        await self.graph.invalidate_entity(name, entity_type, invalid_at)
        return [TextContent(type="text", text=json.dumps({"status": "invalidated"}))]


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

async def main() -> None:
    """Run the MCP stdio server with graceful shutdown."""
    setup_logging()
    logger = get_logger("mcp_server")

    # Allow external configuration (e.g. from CLI) via module-level override
    server: ZaxyMCPServer
    if hasattr(sys.modules[__name__], "server"):
        server = sys.modules[__name__].server  # type: ignore[attr-defined]
    else:
        server = ZaxyMCPServer()

    await server.setup()
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
            return await server.handle_memory_append(arguments)
        if name == "memory_query":
            return await server.handle_memory_query(arguments)
        if name == "memory_replay":
            return await server.handle_memory_replay(arguments)
        if name == "memory_invalidate":
            return await server.handle_memory_invalidate(arguments)
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
        await server.teardown()
        logger.info("zaxy_mcp_server_stopped")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
