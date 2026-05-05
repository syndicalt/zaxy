"""Core memory fabric API.

The MemoryFabric is the primary interface for agents to persist and query
context. It coordinates between Eventloom (immutable log), the temporal
knowledge graph (Neo4j), hybrid extraction, and Pathlight tracing.

Example::

    fabric = MemoryFabric(
        eventloom_path=".eventloom/agent.jsonl",
        neo4j_uri="bolt://localhost:7687",
    )
    await fabric.connect()
    await fabric.append("goal.created", actor="user", payload={"title": "Ship it"})
    context = await fabric.query("What are our goals?")
    await fabric.close()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zaxy.config import get_settings
from zaxy.event import EventLog, ReplayResult
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.query import QueryRouter
from zaxy.trace import MemoryTracer


@dataclass(frozen=True)
class Context:
    """A piece of retrieved context for injection into an agent prompt."""

    content: str
    source: str  # e.g. "graphiti", "eventloom", "cache"
    score: float
    valid_from: str | None = None
    valid_to: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryFabric:
    """Framework-agnostic persistent memory for AI agents.

    Orchestrates the full pipeline: event logging → hybrid extraction →
    temporal graph storage → hybrid retrieval, with full observability.
    """

    def __init__(
        self,
        eventloom_path: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        pathlight_url: str | None = None,
        pathlight_project_id: str | None = None,
        tracer_disabled: bool = False,
    ) -> None:
        """Initialize fabric with configuration.

        All arguments default to environment variables (via Settings).
        Explicit values override env vars for framework integrations.
        """
        settings = get_settings()

        self.eventloom = EventLog(eventloom_path or settings.eventloom_log())
        self.graph = GraphStore(
            neo4j_uri or settings.neo4j_uri,
            neo4j_user or settings.neo4j_user,
            neo4j_password or settings.neo4j_password,
        )
        self.query_router = QueryRouter(
            self.graph, default_limit=settings.query_default_limit
        )
        self.tracer = MemoryTracer(
            base_url=pathlight_url or settings.pathlight_url,
            project_id=pathlight_project_id or settings.pathlight_project_id,
            disabled=tracer_disabled,
        )
        self._connected = False

    async def connect(self) -> None:
        """Connect to Neo4j and Pathlight. Idempotent."""
        if self._connected:
            return
        await self.graph.connect()
        await self.graph.init_schema()
        await self.tracer.connect()
        self._connected = True

    async def close(self) -> None:
        """Close all connections. Idempotent."""
        await self.graph.close()
        await self.tracer.close()
        self._connected = False

    async def append(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None = None,
        thread: str = "default",
    ) -> None:
        """Append a typed event to the immutable log and project to the graph.

        This is the primary write path. It:
        1. Appends to Eventloom JSONL with hash-chain integrity.
        2. Extracts entities/edges via hybrid extraction (rule-based + fallback).
        3. Upserts into the bi-temporal Neo4j graph.
        4. Emits a Pathlight trace span.
        """
        if not self._connected:
            await self.connect()

        event = self.eventloom.append(
            event_type,
            actor=actor,
            payload=payload or {},
            thread=thread,
        )

        extraction = extract(event)
        await self.graph.upsert_extraction(extraction)
        await self.tracer.trace_append(event_type, actor, event.seq)

    async def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[Context]:
        """Query the temporal knowledge graph for relevant context.

        This is the primary read path. It runs hybrid retrieval
        (exact + keyword + traversal) and returns ranked context chunks.
        """
        if not self._connected:
            await self.connect()

        import time

        start = time.perf_counter()
        chunks = await self.query_router.query(
            query,
            temporal_point=temporal_point,
            limit=limit,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        await self.tracer.trace_query(query, len(chunks), duration_ms, temporal_point)

        return [
            Context(
                content=c.content,
                source=c.source,
                score=c.score,
                valid_from=c.valid_from,
                valid_to=c.valid_to,
            )
            for c in chunks
        ]

    async def replay(self, from_seq: int = 1) -> ReplayResult:
        """Replay events from the log starting at a sequence number.

        Returns the full replay result including integrity verification.
        """
        return self.eventloom.replay(from_seq=from_seq)

    async def invalidate(self, entity_name: str, entity_type: str, invalid_at: str) -> None:
        """Mark a fact as invalid at a given time (bi-temporal update).

        This performs a "soft delete" by setting valid_to on the live
        entity record, preserving history.
        """
        if not self._connected:
            await self.connect()
        await self.graph.invalidate_entity(entity_name, entity_type, invalid_at)

    async def handoff_summary(self) -> dict[str, Any]:
        """Generate a concise handoff summary from the event log.

        Suitable for resuming an agent session across restarts.
        """
        return self.eventloom.handoff_summary()
