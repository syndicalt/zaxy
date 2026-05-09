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

from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from zaxy.codebase import collect_codebase_events
from zaxy.compaction import (
    CompactionProjection,
    load_compaction_projection,
    search_compaction_projections,
)
from zaxy.config import get_settings
from zaxy.documents import collect_document_events
from zaxy.embedding import build_embedding_provider, embed_extraction
from zaxy.event import EventLog, ReplayResult  # noqa: F401 - compatibility for existing tests
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.metrics import get_metrics
from zaxy.query import QueryRouter, build_reranker
from zaxy.security import validate_payload, validate_query, validate_session_id
from zaxy.session import SessionManager
from zaxy.trace import MemoryTracer
from zaxy.transcripts import collect_transcript_events
from zaxy.workspace import WorkspaceProfile, build_session_genesis_event


@dataclass(frozen=True)
class Context:
    """A piece of retrieved context for injection into an agent prompt."""

    content: str
    source: str  # e.g. "graphiti", "eventloom", "cache"
    score: float
    valid_from: str | None = None
    valid_to: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ContextAssembly:
    """Prompt-ready assembled context from replay plus retrieval."""

    session_id: str
    prompt: str
    contexts: list[Context]
    replay_event_count: int
    compacted: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HandoffBundle:
    """Portable handoff package for resuming a session or subagent."""

    session_id: str
    summary: dict[str, Any]
    prompt: str
    contexts: list[Context]
    replay_event_count: int
    integrity_ok: bool


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
        projection_paths: list[str | Path] | tuple[str | Path, ...] = (),
    ) -> None:
        """Initialize fabric with configuration.

        All arguments default to environment variables (via Settings).
        Explicit values override env vars for framework integrations.
        """
        settings = get_settings()

        self.session_manager = SessionManager(base_path=eventloom_path or settings.eventloom_path)
        self.eventloom = self.session_manager.get("default").eventlog
        self.graph = GraphStore(
            neo4j_uri or settings.neo4j_uri,
            neo4j_user or settings.neo4j_user,
            neo4j_password or settings.neo4j_password,
            ca_cert=settings.neo4j_ca_cert,
            trust_all=settings.neo4j_trust_all,
        )
        self.query_router = QueryRouter(
            self.graph,
            default_limit=settings.query_default_limit,
            session_id=settings.eventloom_thread,
            scoring_profile=settings.query_scoring_profile,
            reranker=build_reranker(settings),
        )
        self.embedding_provider = build_embedding_provider(settings)
        self.tracer = MemoryTracer(
            base_url=pathlight_url or settings.pathlight_url,
            project_id=pathlight_project_id or settings.pathlight_project_id,
            disabled=tracer_disabled or not settings.pathlight_enabled,
        )
        projection_search_base = Path(eventloom_path or settings.eventloom_path)
        self.projections: tuple[CompactionProjection, ...] = tuple(
            load_compaction_projection(path)
            for path in _compaction_projection_paths(
                projection_search_base,
                projection_paths,
            )
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
        session_id: str | None = None,
    ) -> None:
        """Append a typed event to the immutable log and project to the graph.

        This is the primary write path. It:
        1. Appends to Eventloom JSONL with hash-chain integrity.
        2. Extracts entities/edges via hybrid extraction (rule-based + fallback).
        3. Upserts into the bi-temporal Neo4j graph.
        4. Emits a Pathlight trace span.

        Args:
            session_id: Optional session ID. Defaults to ``thread`` for
                backward compatibility.
        """
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                get_metrics().record_degraded_operation("append", "graph_connect_unavailable")
                self._connected = False

        sid = validate_session_id(session_id or thread)
        safe_payload = validate_payload(payload or {})
        eventlog = self.session_manager.get(sid).eventlog

        event = eventlog.append(
            event_type,
            actor=actor,
            payload=safe_payload,
            thread=sid,
        )

        extraction = extract(event)
        if self.embedding_provider is not None:
            try:
                extraction = embed_extraction(extraction, self.embedding_provider)
            except Exception:
                get_metrics().record_degraded_operation("append", "embedding_provider_unavailable")
        try:
            await self.graph.upsert_extraction(extraction, session_id=sid)
        except Exception:
            get_metrics().record_degraded_operation("append", "graph_projection_unavailable")
        with suppress(Exception):
            await self.tracer.trace_append(event_type, actor, event.seq)

        # Metrics
        metrics = get_metrics()
        metrics.record_event_append(event_type)
        for ent in extraction.entities:
            metrics.record_upsert(ent.entity_type)

    async def ingest_documents(
        self,
        path: str | Path,
        *,
        session_id: str = "default",
        max_lines: int = 80,
    ) -> int:
        """Ingest local Markdown/text documents as cited memory events."""
        sid = validate_session_id(session_id)
        events = collect_document_events(path, max_lines=max_lines)
        for event in events:
            await self.append(
                event["event_type"],
                actor=event["actor"],
                payload=event["payload"],
                session_id=sid,
            )
        return len(events)

    async def ingest_codebase(
        self,
        path: str | Path,
        *,
        session_id: str = "default",
        max_bytes: int = 512 * 1024,
    ) -> int:
        """Ingest local codebase file, symbol, and import mapping events."""
        sid = validate_session_id(session_id)
        events = collect_codebase_events(path, max_bytes=max_bytes)
        for event in events:
            await self.append(
                event["event_type"],
                actor=event["actor"],
                payload=event["payload"],
                session_id=sid,
            )
        return len(events)

    async def initialize_session(
        self,
        path: str | Path,
        *,
        session_id: str = "default",
    ) -> WorkspaceProfile:
        """Append a workspace genesis event for a session."""
        sid = validate_session_id(session_id)
        event = build_session_genesis_event(path, session_id=sid)
        await self.append(
            event["event_type"],
            actor=event["actor"],
            payload=event["payload"],
            session_id=sid,
        )
        payload = event["payload"]
        return WorkspaceProfile(
            workspace_type=str(payload["workspace_type"]),
            confidence=float(payload["confidence"]),
            signals=list(payload["signals"]),
            instructions_profile=str(payload["instructions_profile"]),
        )

    async def ingest_transcript(
        self,
        turns: list[dict[str, Any]],
        *,
        source: str = "transcript",
        session_id: str = "default",
    ) -> int:
        """Ingest sanitized transcript turns as Eventloom-backed memory."""
        sid = validate_session_id(session_id)
        events = collect_transcript_events(turns, source=source)
        for event in events:
            await self.append(
                event["event_type"],
                actor=event["actor"],
                payload=event["payload"],
                session_id=sid,
            )
        return len(events)

    async def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
        session_id: str = "default",
    ) -> list[Context]:
        """Query the temporal knowledge graph for relevant context.

        This is the primary read path. It runs hybrid retrieval
        (exact + vector + keyword + traversal) and returns ranked context chunks.
        """
        import time

        validate_query(query)
        sid = validate_session_id(session_id)
        start = time.perf_counter()
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                get_metrics().record_degraded_operation("query", "graph_unavailable")
                chunks = self._merge_projection_contexts(
                    self._query_eventlog_fallback(query, sid, limit, reason="graph unavailable"),
                    query,
                    limit,
                )
                duration_ms = (time.perf_counter() - start) * 1000
                await self._trace_query_best_effort(query, len(chunks), duration_ms, temporal_point)
                get_metrics().record_query(duration_ms / 1000.0, source="eventloom")
                return chunks

        query_embedding = embedding
        if query_embedding is None and self.embedding_provider is not None:
            try:
                query_embedding = self.embedding_provider.embed(query)
            except Exception:
                get_metrics().record_degraded_operation("query", "embedding_provider_unavailable")
                query_embedding = None

        try:
            router_chunks = await self.query_router.query(
                query,
                temporal_point=temporal_point,
                limit=limit,
                embedding=query_embedding,
                session_id=sid,
            )
        except Exception:
            get_metrics().record_degraded_operation("query", "graph_retrieval_unavailable")
            contexts = self._merge_projection_contexts(
                self._query_eventlog_fallback(query, sid, limit, reason="graph retrieval unavailable"),
                query,
                limit,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            await self._trace_query_best_effort(query, len(contexts), duration_ms, temporal_point)
            get_metrics().record_query(duration_ms / 1000.0, source="eventloom")
            return contexts
        duration_ms = (time.perf_counter() - start) * 1000

        await self._trace_query_best_effort(query, len(router_chunks), duration_ms, temporal_point)

        # Metrics
        get_metrics().record_query(duration_ms / 1000.0)

        contexts = []
        for c in router_chunks:
            metadata: dict[str, Any] = {}
            if c.citation:
                metadata["citation"] = c.citation
            if c.score_explanation:
                metadata["score_explanation"] = c.score_explanation
            contexts.append(
                Context(
                    content=c.content,
                    source=c.source,
                    score=c.score,
                    valid_from=c.valid_from,
                    valid_to=c.valid_to,
                    metadata=metadata or None,
                )
            )
        return self._merge_projection_contexts(contexts, query, limit)

    def _query_eventlog_fallback(
        self,
        query: str,
        session_id: str,
        limit: int,
        *,
        reason: str,
    ) -> list[Context]:
        """Return best-effort context from Eventloom when graph retrieval is down."""
        replay = self.session_manager.replay(session_id, from_seq=1)
        query_tokens = _tokens(query)
        contexts: list[Context] = []
        for event in replay.events:
            content = _event_content(event)
            if not content:
                continue
            event_tokens = _tokens(content)
            if query_tokens:
                score = len(query_tokens & event_tokens) / len(query_tokens)
                if score == 0:
                    continue
            else:
                score = 0.1
            contexts.append(
                Context(
                    content=content,
                    source="eventloom",
                    score=round(score, 4),
                    valid_from=getattr(event, "timestamp", None),
                    valid_to=None,
                    metadata={
                        "degraded": True,
                        "reason": reason,
                        "event_seq": getattr(event, "seq", None),
                        "event_type": getattr(event, "type", None),
                    },
                )
            )
        return sorted(contexts, key=lambda item: item.score, reverse=True)[:limit]

    def _merge_projection_contexts(
        self,
        contexts: list[Context],
        query: str,
        limit: int,
    ) -> list[Context]:
        """Merge projection routing hits while preserving source citations."""
        if not self.projections:
            return contexts[:limit]
        projection_contexts = [
            Context(
                content=result.record.text,
                source="projection",
                score=result.score,
                metadata={
                    "projection_id": result.projection_id,
                    "projection_strategy": result.strategy,
                    "event_ref": result.record.event_ref,
                    "citation": result.citations[0] if result.citations else result.record.event_ref,
                    "citations": list(result.citations),
                },
            )
            for result in search_compaction_projections(
                self.projections,
                query,
                limit=limit,
            )
        ]
        merged = [*contexts, *projection_contexts]
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:limit]

    async def _trace_query_best_effort(
        self,
        query: str,
        result_count: int,
        duration_ms: float,
        temporal_point: str | None,
    ) -> None:
        with suppress(Exception):
            await self.tracer.trace_query(query, result_count, duration_ms, temporal_point)

    async def replay(self, from_seq: int = 1, session_id: str = "default") -> ReplayResult:
        """Replay events from the log starting at a sequence number.

        Returns the full replay result including integrity verification.
        """
        return cast(ReplayResult, self.session_manager.replay(session_id, from_seq=from_seq))

    async def assemble_context(
        self,
        query: str,
        *,
        session_id: str = "default",
        replay_from_seq: int = 1,
        limit: int = 10,
        max_recent_events: int | None = None,
    ) -> ContextAssembly:
        """Assemble recent replay plus retrieval into prompt-ready context."""
        sid = validate_session_id(session_id)
        replay = await self.replay(from_seq=replay_from_seq, session_id=sid)
        contexts = await self.query(query, limit=limit, session_id=sid)
        replay_events = list(replay.events)
        compacted = False
        if max_recent_events is not None and len(replay_events) > max_recent_events:
            replay_events = replay_events[-max_recent_events:]
            compacted = True
        lines = ["# Recent Events"]
        for event in replay_events:
            lines.append(f"[{event.seq}] {event.type} by {event.actor}")
            content = _event_content(event)
            if content:
                lines.append(str(content))
        lines.append("")
        lines.append("# Retrieved Context")
        for context in contexts:
            citation = ""
            if context.metadata and context.metadata.get("citation"):
                citation = f" ({context.metadata['citation']})"
            lines.append(f"- {context.content}{citation}")
        warnings = _context_warnings(contexts, compacted=compacted)
        if warnings:
            lines.append("")
            lines.append("# Context Warnings")
            for warning in warnings:
                lines.append(f"- {warning}")
        return ContextAssembly(
            session_id=sid,
            prompt="\n".join(lines).strip(),
            contexts=contexts,
            replay_event_count=len(replay_events),
            compacted=compacted,
            warnings=warnings,
        )

    async def after_turn(
        self,
        *,
        role: str,
        content: str,
        session_id: str = "default",
        query: str | None = None,
        source: str = "after-turn",
        max_recent_events: int = 20,
        limit: int = 10,
    ) -> ContextAssembly:
        """Persist a completed turn and assemble compact context for the next turn."""
        sid = validate_session_id(session_id)
        await self.append(
            "transcript.turn",
            actor=role,
            payload={"role": role, "content": content, "source": source},
            session_id=sid,
        )
        return await self.assemble_context(
            query or content,
            session_id=sid,
            replay_from_seq=1,
            limit=limit,
            max_recent_events=max_recent_events,
        )

    async def handoff_bundle(
        self,
        *,
        session_id: str = "default",
        query: str = "session handoff",
        replay_from_seq: int = 1,
        limit: int = 10,
        max_recent_events: int = 20,
    ) -> HandoffBundle:
        """Build a portable handoff bundle with summary, replay, and retrieval."""
        sid = validate_session_id(session_id)
        summary = self.session_manager.handoff_summary(sid)
        replay = await self.replay(from_seq=replay_from_seq, session_id=sid)
        assembly = await self.assemble_context(
            query,
            session_id=sid,
            replay_from_seq=replay_from_seq,
            limit=limit,
            max_recent_events=max_recent_events,
        )
        integrity = getattr(replay, "integrity", None)
        return HandoffBundle(
            session_id=sid,
            summary=summary,
            prompt=assembly.prompt,
            contexts=assembly.contexts,
            replay_event_count=assembly.replay_event_count,
            integrity_ok=bool(getattr(integrity, "ok", False)),
        )

    async def cleanup_subagent(
        self,
        *,
        parent_session_id: str,
        subagent_session_id: str,
        summary: str,
        query: str = "subagent handoff",
        limit: int = 10,
    ) -> HandoffBundle:
        """Finalize a subagent session and return a handoff bundle for the parent."""
        parent_sid = validate_session_id(parent_session_id)
        subagent_sid = validate_session_id(subagent_session_id)
        await self.append(
            "subagent.cleaned",
            actor="zaxy",
            payload={
                "parent_session_id": parent_sid,
                "subagent_session_id": subagent_sid,
                "summary": summary,
            },
            session_id=subagent_sid,
        )
        return await self.handoff_bundle(
            session_id=subagent_sid,
            query=query,
            replay_from_seq=1,
            limit=limit,
        )

    async def invalidate(
        self,
        entity_name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Mark a fact as invalid at a given time (bi-temporal update).

        This performs a "soft delete" by setting valid_to on the live
        entity record, preserving history.
        """
        if not self._connected:
            await self.connect()
        await self.graph.invalidate_entity(
            entity_name,
            entity_type,
            invalid_at,
            session_id=validate_session_id(session_id),
        )

    async def handoff_summary(self, session_id: str = "default") -> dict[str, Any]:
        """Generate a concise handoff summary from the event log.

        Suitable for resuming an agent session across restarts.
        """
        return self.session_manager.handoff_summary(session_id)


def _event_content(event: Any) -> str:
    payload = getattr(event, "payload", {})
    if not isinstance(payload, dict):
        return ""
    parts = [
        str(payload[key])
        for key in ("title", "summary", "content", "text", "decision", "task")
        if payload.get(key)
    ]
    if not parts:
        parts = [f"{getattr(event, 'type', 'event')} by {getattr(event, 'actor', 'unknown')}"]
    return " ".join(parts)


def _context_warnings(contexts: list[Context], *, compacted: bool) -> list[str]:
    warnings: list[str] = []
    for context in contexts:
        if _is_compacted_context(context) and not _has_source_support(context):
            warnings.append(
                f"{context.source} context '{_warning_label(context.content)}' "
                "lacks source-level citation"
            )
    if compacted and not any(_has_source_support(context) for context in contexts):
        warnings.append(
            "recent replay was truncated and no retrieved source context was available"
        )
    return warnings


def _is_compacted_context(context: Context) -> bool:
    source = context.source.casefold()
    if source in {"projection", "compaction", "compacted"}:
        return True
    metadata = context.metadata or {}
    return bool(
        metadata.get("compacted")
        or metadata.get("projection_id")
        or metadata.get("compaction_projection")
    )


def _has_source_support(context: Context) -> bool:
    metadata = context.metadata or {}
    if metadata.get("citation"):
        return True
    citations = metadata.get("citations")
    return bool(citations)


def _warning_label(content: str) -> str:
    compact = " ".join(content.split())
    if len(compact) <= 80:
        return compact
    return f"{compact[:77]}..."


def _compaction_projection_paths(
    eventloom_path: Path,
    explicit_paths: list[str | Path] | tuple[str | Path, ...],
) -> tuple[Path, ...]:
    discovered = (
        sorted(eventloom_path.rglob("*.compaction.json"))
        if eventloom_path.exists() and eventloom_path.is_dir()
        else []
    )
    ordered = [*discovered, *(Path(path) for path in explicit_paths)]
    unique: dict[Path, Path] = {}
    for path in ordered:
        unique.setdefault(path.resolve(), path)
    return tuple(unique.values())


def _tokens(value: str) -> set[str]:
    import re

    return {token for token in re.findall(r"[A-Za-z0-9]+", value.lower()) if len(token) > 1}
