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

import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from zaxy.checkout import (
    build_checkout_diagnostics,
    build_checkout_guidance,
    build_checkout_quality,
    build_compact_answer_contexts,
    format_memory_checkout_prompt,
)
from zaxy.codebase import collect_codebase_events
from zaxy.compaction import (
    CompactionProjection,
    load_compaction_projection,
    search_compaction_projections,
)
from zaxy.config import get_settings
from zaxy.context import Context, ContextAssemblyPolicy, context_counts
from zaxy.context_refresh import (
    ContextRefreshPlan,
    load_refresh_state,
    plan_context_refresh,
    save_refresh_state,
)
from zaxy.documents import collect_document_events
from zaxy.embedding import build_embedding_provider, embed_extraction
from zaxy.event import EventLog, ReplayResult  # noqa: F401 - compatibility for existing tests
from zaxy.evidence import select_checkout_evidence
from zaxy.extract import extract
from zaxy.inference import build_inferred_edge_events
from zaxy.lifecycle import build_subagent_completed_event
from zaxy.metrics import get_metrics
from zaxy.pagination import encode_query_cursor, validate_query_cursor
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store
from zaxy.query import QueryRouter, build_reranker, build_retention_policy
from zaxy.recall import RecallCandidateSet, build_recall_candidate_set, empty_recall_candidate_set
from zaxy.refs import MemoryRef, MemoryRefStore
from zaxy.retrieval_plan import build_evidence_plan, source_lane_candidate_limit
from zaxy.retrieval_profile import (
    RetrievalProfile,
    apply_retrieval_profile,
    resolve_retrieval_profile,
)
from zaxy.security import (
    MAX_QUERY_LIMIT,
    validate_limit,
    validate_payload,
    validate_query,
    validate_session_id,
)
from zaxy.session import SessionManager
from zaxy.trace import MemoryTracer
from zaxy.transcripts import collect_transcript_events
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


@dataclass(frozen=True)
class ContextAssembly:
    """Prompt-ready assembled context from replay plus retrieval."""

    session_id: str
    prompt: str
    contexts: list[Context]
    replay_event_count: int
    compacted: bool = False
    warnings: list[str] = field(default_factory=list)
    assembly_policy: dict[str, bool | int] = field(default_factory=dict)
    context_counts: dict[str, int] = field(default_factory=dict)
    working_set: dict[str, object] = field(default_factory=dict)
    recall: RecallCandidateSet = field(default_factory=empty_recall_candidate_set)


@dataclass(frozen=True)
class MemoryCheckout:
    """Cited, prompt-ready current memory state for an agent turn."""

    session_id: str
    query: str
    prompt: str
    working_set: dict[str, object]
    ref: dict[str, object] | None
    current_facts: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    retention: dict[str, Any]
    warnings: list[str]
    guidance: dict[str, Any]
    quality: dict[str, Any]
    diagnostics: dict[str, Any]
    context_counts: dict[str, int]
    replay_event_count: int
    compacted: bool = False
    assembly_policy: dict[str, bool | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable payload for tools and CLIs."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "prompt": self.prompt,
            "working_set": self.working_set,
            "ref": self.ref,
            "current_facts": self.current_facts,
            "evidence": self.evidence,
            "provenance": self.provenance,
            "retention": self.retention,
            "warnings": self.warnings,
            "guidance": self.guidance,
            "quality": self.quality,
            "diagnostics": self.diagnostics,
            "context_counts": self.context_counts,
            "replay_event_count": self.replay_event_count,
            "compacted": self.compacted,
            "assembly_policy": self.assembly_policy,
        }


@dataclass(frozen=True)
class QueryPage:
    """A stable page of ranked memory query results."""

    contexts: list[Context]
    next_cursor: str | None
    cursor: str | None
    has_more: bool
    offset: int

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable pagination payload."""
        return {
            "contexts": [
                {
                    "content": context.content,
                    "source": context.source,
                    "score": context.score,
                    "valid_from": context.valid_from,
                    "valid_to": context.valid_to,
                    "metadata": context.metadata,
                }
                for context in self.contexts
            ],
            "next_cursor": self.next_cursor,
            "cursor": self.cursor,
            "has_more": self.has_more,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class HandoffBundle:
    """Portable handoff package for resuming a session or subagent."""

    session_id: str
    summary: dict[str, Any]
    prompt: str
    contexts: list[Context]
    replay_event_count: int
    integrity_ok: bool


@dataclass(frozen=True)
class ContextRefreshReport:
    """Result of an incremental source refresh."""

    session_id: str
    kind: str
    event_count: int
    summary: dict[str, int | str]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable payload."""
        return {
            "session_id": self.session_id,
            "kind": self.kind,
            "event_count": self.event_count,
            "summary": self.summary,
        }


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
        neo4j_ca_cert: str | None = None,
        neo4j_trust_all: bool | None = None,
        pathlight_url: str | None = None,
        pathlight_project_id: str | None = None,
        tracer_disabled: bool = False,
        projection_paths: list[str | Path] | tuple[str | Path, ...] = (),
        projection_backend: str | None = None,
        pggraph_dsn: str | None = None,
    ) -> None:
        """Initialize fabric with configuration.

        All arguments default to environment variables (via Settings).
        Explicit values override env vars for framework integrations.
        """
        settings = get_settings()
        retrieval_profile = resolve_retrieval_profile(settings)
        resolved_settings = apply_retrieval_profile(settings, retrieval_profile)
        self.settings = resolved_settings
        self.retrieval_profile: RetrievalProfile = retrieval_profile

        self.eventloom_path = Path(eventloom_path or resolved_settings.eventloom_path)
        self.session_manager = SessionManager(base_path=str(self.eventloom_path))
        self.eventloom = self.session_manager.get("default").eventlog
        self.graph = build_projection_store(
            ProjectionBackendConfig(
                backend=projection_backend or resolved_settings.projection_backend,
                neo4j_uri=neo4j_uri or resolved_settings.neo4j_uri,
                neo4j_user=neo4j_user or resolved_settings.neo4j_user,
                neo4j_password=neo4j_password or resolved_settings.neo4j_password,
                neo4j_ca_cert=neo4j_ca_cert if neo4j_ca_cert is not None else resolved_settings.neo4j_ca_cert,
                neo4j_trust_all=neo4j_trust_all if neo4j_trust_all is not None else resolved_settings.neo4j_trust_all,
                pggraph_dsn=pggraph_dsn or resolved_settings.pggraph_dsn,
            )
        )
        self.query_router = QueryRouter(
            self.graph,
            default_limit=resolved_settings.query_default_limit,
            session_id=resolved_settings.eventloom_thread,
            scoring_profile=resolved_settings.query_scoring_profile,
            reranker=build_reranker(resolved_settings),
            retention_policy=build_retention_policy(resolved_settings),
        )
        self.embedding_provider = build_embedding_provider(resolved_settings)
        self.tracer = MemoryTracer(
            base_url=pathlight_url or resolved_settings.pathlight_url,
            project_id=pathlight_project_id or resolved_settings.pathlight_project_id,
            disabled=tracer_disabled or not resolved_settings.pathlight_enabled,
        )
        projection_search_base = Path(eventloom_path or resolved_settings.eventloom_path)
        self.refs = MemoryRefStore(projection_search_base)
        self.projections: tuple[CompactionProjection, ...] = tuple(
            load_compaction_projection(path)
            for path in _compaction_projection_paths(
                projection_search_base,
                projection_paths,
            )
        )
        self.context_assembly_policy = ContextAssemblyPolicy(
            verbatim_enabled=resolved_settings.context_verbatim_enabled,
            verbatim_slots=resolved_settings.context_verbatim_slots,
            packet_memory_enabled=resolved_settings.context_packet_memory_enabled,
            packet_memory_slots=resolved_settings.context_packet_memory_slots,
        )
        self._initialized_workspaces: dict[tuple[str, str], WorkspaceProfile] = {}
        self._initialized_instruction_signatures: dict[tuple[str, str], str] = {}
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

        await self._project_event(event, session_id=sid)
        await self._append_generated_inferences(eventlog, source_event=event, session_id=sid)

    async def _project_event(self, event: Any, *, session_id: str) -> None:
        """Extract, project, trace, and record metrics for one sealed event."""
        extraction = extract(event)
        if self.embedding_provider is not None:
            try:
                extraction = embed_extraction(extraction, self.embedding_provider)
            except Exception:
                get_metrics().record_degraded_operation("append", "embedding_provider_unavailable")
        try:
            await self.graph.upsert_extraction(extraction, session_id=session_id)
        except Exception:
            get_metrics().record_degraded_operation("append", "graph_projection_unavailable")
        with suppress(Exception):
            await self.tracer.trace_append(event.type, event.actor, event.seq)

        # Metrics
        metrics = get_metrics()
        metrics.record_event_append(event.type)
        for ent in extraction.entities:
            metrics.record_upsert(ent.entity_type)

    async def _append_generated_inferences(
        self,
        eventlog: EventLog,
        *,
        source_event: Any,
        session_id: str,
    ) -> None:
        """Append and project inferred-edge events generated from cited evidence."""
        if source_event.type == "inference.edge.generated":
            return
        for generated in build_inferred_edge_events(source_event):
            event = eventlog.append(
                generated["event_type"],
                actor=generated["actor"],
                payload=validate_payload(generated["payload"]),
                thread=session_id,
            )
            await self._project_event(event, session_id=session_id)

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

    async def refresh_context(
        self,
        path: str | Path,
        *,
        kind: str,
        session_id: str = "default",
        max_lines: int = 80,
        max_bytes: int = 512 * 1024,
    ) -> ContextRefreshReport:
        """Refresh document or codebase context incrementally from source fingerprints."""
        sid = validate_session_id(session_id)
        previous = load_refresh_state(self.eventloom_path, session_id=sid, kind=kind)
        plan: ContextRefreshPlan = plan_context_refresh(
            path,
            kind=kind,
            previous=previous,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )
        for event in plan.events:
            await self.append(
                event["event_type"],
                actor=event["actor"],
                payload=event["payload"],
                session_id=sid,
            )
        save_refresh_state(self.eventloom_path, session_id=sid, state=plan.next_state)
        return ContextRefreshReport(
            session_id=sid,
            kind=plan.kind,
            event_count=len(plan.events),
            summary=plan.summary,
        )

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
        profile = workspace_profile_from_payload(payload)
        self._initialized_workspaces[(str(Path(path).resolve()), sid)] = profile
        return profile

    async def ensure_session_initialized(
        self,
        path: str | Path,
        *,
        session_id: str = "default",
    ) -> WorkspaceProfile:
        """Idempotently append a workspace genesis event for a session."""
        sid = validate_session_id(session_id)
        root = str(Path(path).resolve())
        key = (root, sid)
        eventlog = self.session_manager.get(sid).eventlog
        cached = self._initialized_workspaces.get(key)
        if cached is not None:
            await self._ensure_workspace_instructions(root, session_id=sid)
            return cached
        profile = existing_session_genesis_profile(
            eventlog.read_all(),
            root=root,
            session_id=sid,
        )
        if profile is not None:
            self._initialized_workspaces[key] = profile
            await self._ensure_workspace_instructions(root, session_id=sid)
            return profile
        profile = await self.initialize_session(root, session_id=sid)
        await self._ensure_workspace_instructions(root, session_id=sid)
        return profile

    async def _ensure_workspace_instructions(
        self,
        path: str | Path,
        *,
        session_id: str,
    ) -> None:
        """Idempotently append discovered workspace instruction summaries."""
        root = str(Path(path).resolve())
        key = (root, session_id)
        event = build_workspace_instruction_event(root, session_id=session_id)
        if event is None:
            return
        signature = str(event["payload"]["signature"])
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
            event = mark_workspace_instruction_event_updated(
                event,
                previous_signature=existing_signature,
            )
        await self.append(
            event["event_type"],
            actor=event["actor"],
            payload=event["payload"],
            session_id=session_id,
        )
        self._initialized_instruction_signatures[key] = signature

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
            if c.entity_name:
                metadata["entity_name"] = c.entity_name
            if c.entity_type:
                metadata["entity_type"] = c.entity_type
            if c.metadata:
                metadata.update(c.metadata)
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

    async def query_page(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
        session_id: str = "default",
        cursor: str | None = None,
    ) -> QueryPage:
        """Query memory with an opaque continuation cursor.

        The cursor is bound to the query text, temporal filter, and session so an
        agent can ask for more results without accidentally paging a different
        memory scope.
        """
        validated_query = validate_query(query)
        sid = validate_session_id(session_id)
        page_limit = validate_limit(limit)
        offset = 0
        if cursor:
            decoded = validate_query_cursor(
                cursor,
                query=validated_query,
                session_id=sid,
                temporal_point=temporal_point,
            )
            offset = decoded.offset
        fetch_limit = min(offset + page_limit + 1, MAX_QUERY_LIMIT)
        contexts = await self.query(
            validated_query,
            temporal_point=temporal_point,
            limit=fetch_limit,
            embedding=embedding,
            session_id=sid,
        )
        page_contexts = contexts[offset : offset + page_limit]
        has_more = len(contexts) > offset + page_limit
        next_cursor = None
        if has_more:
            next_cursor = encode_query_cursor(
                query=validated_query,
                session_id=sid,
                temporal_point=temporal_point,
                offset=offset + page_limit,
            )
        return QueryPage(
            contexts=page_contexts,
            next_cursor=next_cursor,
            cursor=cursor,
            has_more=has_more,
            offset=offset,
        )

    async def query_verbatim(
        self,
        query: str,
        *,
        session_id: str = "default",
        limit: int = 10,
    ) -> list[Context]:
        """Retrieve exact Eventloom source chunks without requiring graph services."""
        validate_query(query)
        sid = validate_session_id(session_id)
        index = VerbatimIndex.from_event_logs([self.session_manager.get(sid).eventlog])
        contexts: list[Context] = []
        for hit in index.query(query, limit=limit):
            contexts.append(
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
            )
        return contexts

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

    def _recent_packet_memory_contexts(self, events: list[Any]) -> list[Context]:
        """Return newest projected packet memories as proactive context candidates."""
        reinforcements = _packet_memory_reinforcements(events)
        contexts: list[Context] = []
        for event in reversed(events):
            if getattr(event, "type", "") != "llm.packet.projected":
                continue
            payload = getattr(event, "payload", {})
            if not isinstance(payload, dict):
                continue
            summary = payload.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                continue
            metadata: dict[str, Any] = {
                "citation": _event_citation(event),
                "source_kind": "packet_projection",
                "event_seq": getattr(event, "seq", None),
                "event_type": getattr(event, "type", None),
                "event_thread": getattr(event, "thread", None),
                "event_timestamp": getattr(event, "timestamp", None),
                "source_event_seq": payload.get("source_event_seq"),
                "source_event_hash": payload.get("source_event_hash"),
                "provider_path": payload.get("provider_path"),
                "model": payload.get("model"),
            }
            source_hash = payload.get("source_event_hash")
            reinforcement = (
                reinforcements.get(source_hash)
                if isinstance(source_hash, str)
                else None
            )
            score = 0.6
            if reinforcement is not None:
                metadata["reinforcement_count"] = reinforcement["count"]
                metadata["importance"] = reinforcement["importance"]
                score += min(0.3, 0.1 * reinforcement["count"])
                score += min(0.1, 0.1 * reinforcement["importance"])
            contexts.append(
                Context(
                    content=" ".join(summary.split()),
                    source="packet_memory",
                    score=round(score, 4),
                    valid_from=getattr(event, "timestamp", None),
                    valid_to=None,
                    metadata={key: value for key, value in metadata.items() if value is not None},
                )
            )
        return sorted(contexts, key=lambda context: context.score, reverse=True)

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
        recall_limit: int | None = None,
        max_recent_events: int | None = None,
        as_of_seq: int | None = None,
    ) -> ContextAssembly:
        """Assemble recent replay plus retrieval into prompt-ready context."""
        sid = validate_session_id(session_id)
        prompt_limit = validate_limit(limit)
        candidate_limit = prompt_limit if recall_limit is None else validate_limit(max(prompt_limit, recall_limit))
        replay = await self.replay(from_seq=replay_from_seq, session_id=sid)
        graph_contexts = await self.query(query, limit=candidate_limit, session_id=sid)
        verbatim_candidate_limit = self.context_assembly_policy.verbatim_candidate_limit(
            query=query,
            limit=candidate_limit,
        )
        verbatim_contexts = (
            await self.query_verbatim(query, limit=verbatim_candidate_limit, session_id=sid)
            if verbatim_candidate_limit > 0
            else []
        )
        packet_memory_contexts = self._recent_packet_memory_contexts(list(replay.events))
        recall_contexts = [*graph_contexts, *verbatim_contexts, *packet_memory_contexts]
        recall = build_recall_candidate_set(recall_contexts, budget=candidate_limit)
        contexts = self.context_assembly_policy.assemble(
            graph_contexts,
            verbatim_contexts,
            packet_memory_contexts,
            limit=prompt_limit,
            query=query,
        )
        if as_of_seq is not None:
            contexts = _contexts_as_of_seq(contexts, as_of_seq)
            recall = build_recall_candidate_set(
                _contexts_as_of_seq(recall.contexts(), as_of_seq),
                budget=candidate_limit,
            )
        replay_events = list(replay.events)
        if as_of_seq is not None:
            replay_events = [event for event in replay_events if event.seq <= as_of_seq]
        compacted = False
        if max_recent_events is not None and len(replay_events) > max_recent_events:
            replay_events = replay_events[-max_recent_events:]
            compacted = True
        working_set = build_working_set(replay_events, contexts)
        lines = [format_working_set(working_set), "", "# Recent Events"]
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
        working_set_payload = working_set.to_dict()
        working_set_payload["retrieval_profile"] = self.retrieval_profile.to_diagnostics()
        return ContextAssembly(
            session_id=sid,
            prompt="\n".join(lines).strip(),
            contexts=contexts,
            replay_event_count=len(replay_events),
            compacted=compacted,
            warnings=warnings,
            assembly_policy=self.context_assembly_policy.describe(),
            context_counts=context_counts(contexts, replay_count=len(replay_events)),
            working_set=working_set_payload,
            recall=recall,
        )

    async def checkout_memory(
        self,
        query: str,
        *,
        session_id: str = "default",
        replay_from_seq: int = 1,
        limit: int = 10,
        max_recent_events: int | None = 20,
        ref: str | None = None,
    ) -> MemoryCheckout:
        """Checkout the current cited memory state an agent should condition on."""
        resolved_ref = self._resolve_checkout_ref(ref, session_id=session_id)
        checkout_session_id = resolved_ref.session_id if resolved_ref is not None else session_id
        as_of_seq = resolved_ref.target_seq if resolved_ref is not None else None
        assembly = await self.assemble_context(
            query,
            session_id=checkout_session_id,
            replay_from_seq=replay_from_seq,
            limit=limit,
            recall_limit=_checkout_recall_limit(query, limit),
            max_recent_events=max_recent_events,
            as_of_seq=as_of_seq,
        )
        return build_memory_checkout(
            query=query,
            assembly=assembly,
            ref=resolved_ref,
        )

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

    async def record_context_feedback(
        self,
        contexts: list[Context],
        *,
        feedback: str,
        session_id: str = "default",
        actor: str = "zaxy",
        importance: float | None = None,
    ) -> int:
        """Append feedback events for retrieved context without mutating history."""
        sid = validate_session_id(session_id)
        normalized = _normalize_context_feedback(feedback)
        count = 0
        for context in contexts:
            identity = _context_identity(context)
            payload: dict[str, Any] = {
                "entity_name": identity["entity_name"],
                "entity_type": identity["entity_type"],
                "feedback": normalized,
                "source": context.source,
                "score": context.score,
            }
            if context.metadata and (citation := context.metadata.get("citation")):
                payload["citation"] = citation
            if context.metadata:
                payload.update(_context_feedback_metadata(context.metadata))
            if normalized in {"used", "helpful"}:
                payload.pop("feedback")
                if importance is not None:
                    payload["importance"] = max(0.0, min(1.0, float(importance)))
                await self.append(
                    "memory.reinforced",
                    actor=actor,
                    payload=payload,
                    session_id=sid,
                )
            else:
                await self.append(
                    "memory.feedback",
                    actor=actor,
                    payload=payload,
                    session_id=sid,
                )
            count += 1
        return count

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
        event = build_subagent_completed_event(
            parent_session_id=parent_sid,
            subagent_session_id=subagent_sid,
            status="succeeded",
            summary=summary,
        )
        await self.append(
            event["event_type"],
            actor=event["actor"],
            payload=event["payload"],
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


def _event_citation(event: Any) -> str | None:
    thread = getattr(event, "thread", None)
    seq = getattr(event, "seq", None)
    event_hash = getattr(event, "hash", None)
    if not isinstance(thread, str) or not isinstance(seq, int) or not isinstance(event_hash, str):
        return None
    return f"eventloom://{thread}/events/{seq}#{event_hash[:12]}"


def _packet_memory_reinforcements(events: list[Any]) -> dict[str, dict[str, float | int]]:
    reinforcements: dict[str, dict[str, float | int]] = {}
    for event in events:
        if getattr(event, "type", "") != "memory.reinforced":
            continue
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict) or payload.get("entity_type") != "packet_memory":
            continue
        source_event_hash = payload.get("source_event_hash")
        if not isinstance(source_event_hash, str) or not source_event_hash:
            continue
        existing = reinforcements.setdefault(
            source_event_hash,
            {"count": 0, "importance": 0.0},
        )
        existing["count"] = int(existing["count"]) + 1
        importance = payload.get("importance")
        if isinstance(importance, int | float) and not isinstance(importance, bool):
            existing["importance"] = max(float(existing["importance"]), float(importance))
    return reinforcements


def _normalize_context_feedback(feedback: str) -> str:
    normalized = feedback.casefold().strip()
    if normalized not in {"used", "helpful", "irrelevant"}:
        raise ValueError("feedback must be one of: used, helpful, irrelevant")
    return normalized


def build_memory_checkout(
    *,
    query: str,
    assembly: ContextAssembly,
    ref: MemoryRef | None = None,
) -> MemoryCheckout:
    """Build the Memory Checkout contract from assembled context."""
    checkout_contexts = assembly.recall.contexts() or assembly.contexts
    ranked_contexts = sorted(
        checkout_contexts,
        key=lambda context: _checkout_rank(context, query),
        reverse=True,
    )
    candidate_current_facts = [
        _checkout_fact(context) for context in ranked_contexts if context.valid_to is None
    ]
    candidate_evidence = [
        _checkout_evidence(context) for context in ranked_contexts if _context_citation(context)
    ]
    selection = select_checkout_evidence(
        query=query,
        evidence_plan=build_evidence_plan(query, limit=10),
        current_facts=candidate_current_facts,
        evidence=candidate_evidence,
    )
    current_facts = selection.current_facts
    evidence = selection.evidence
    provenance = [_checkout_provenance(context) for context in ranked_contexts if _context_citation(context)]
    warnings = list(assembly.warnings)
    if assembly.compacted and not any("compacted" in warning for warning in warnings):
        warnings.append("Recent replay was compacted to fit the checkout budget.")
    if current_facts and not evidence:
        warnings.append("Checkout contains current facts without Eventloom citations.")
    retention = {
        "policy": "current_only",
        "superseded_contexts_excluded": sum(1 for context in assembly.contexts if context.valid_to is not None),
    }
    diagnostics = build_checkout_diagnostics(
        query=query,
        source_lanes=_checkout_source_lanes(ranked_contexts),
        current_facts=current_facts,
        evidence=evidence,
        retention=retention,
        warnings=warnings,
    )
    skills = _checkout_skills(ranked_contexts, query)
    if skills:
        diagnostics = {**diagnostics, "skills": {"count": len(skills), "items": skills}}
    skill_analytics = _checkout_skill_analytics(ranked_contexts)
    if skill_analytics["version_count"] or skill_analytics["outcome_count"]:
        diagnostics = {**diagnostics, "skill_analytics": skill_analytics}
    retrieval_profile = assembly.working_set.get("retrieval_profile")
    if isinstance(retrieval_profile, dict):
        diagnostics = {**diagnostics, "retrieval_profile": retrieval_profile}
    recall_diagnostics = assembly.recall.to_diagnostics()
    if recall_diagnostics["candidate_count"] and recall_diagnostics["candidate_count"] != len(assembly.contexts):
        diagnostics = {**diagnostics, "recall": recall_diagnostics}
    guidance = build_checkout_guidance(
        query=query,
        current_facts=current_facts,
        retention=retention,
        evidence=evidence,
    )
    quality = build_checkout_quality(
        diagnostics=diagnostics,
        guidance=guidance,
    )
    compact_contexts = build_compact_answer_contexts(
        query=query,
        current_facts=current_facts,
        evidence=evidence,
        diagnostics=diagnostics,
        quality=quality,
    )
    if compact_contexts and "synthesis" in diagnostics:
        diagnostics = {**diagnostics, "compact_contexts": compact_contexts}
    prompt = format_memory_checkout_prompt(
        query=query,
        assembly_prompt=assembly.prompt,
        current_facts=current_facts,
        evidence=evidence,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )
    return MemoryCheckout(
        session_id=assembly.session_id,
        query=query,
        prompt=prompt,
        working_set=assembly.working_set,
        ref=ref.to_dict() if ref is not None else None,
        current_facts=current_facts,
        evidence=evidence,
        provenance=provenance,
        retention=retention,
        warnings=warnings,
        guidance=guidance,
        quality=quality,
        diagnostics=diagnostics,
        context_counts=assembly.context_counts,
        replay_event_count=assembly.replay_event_count,
        compacted=assembly.compacted,
        assembly_policy=assembly.assembly_policy,
    )


def _checkout_fact(context: Context) -> dict[str, Any]:
    metadata = context.metadata or {}
    fact: dict[str, Any] = {
        "content": context.content,
        "source": context.source,
        "score": context.score,
        "citation": _context_citation(context),
        "valid_from": context.valid_from,
        "valid_to": context.valid_to,
        "source_lane": _checkout_source_lane(context),
    }
    for key in ("entity_name", "entity_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            fact[key] = value
    score_explanation = metadata.get("score_explanation")
    if isinstance(score_explanation, dict):
        fact["score_explanation"] = score_explanation
    return fact

def _checkout_evidence(context: Context) -> dict[str, Any]:
    citation = _context_citation(context)
    seq, event_hash = _citation_event_identity(citation)
    evidence: dict[str, Any] = {
        "citation": citation,
        "content": context.content,
        "source": context.source,
        "source_lane": _checkout_source_lane(context),
        "score": context.score,
        "event_seq": seq,
        "event_hash": event_hash,
    }
    metadata = context.metadata or {}
    score_explanation = metadata.get("score_explanation")
    if isinstance(score_explanation, dict):
        evidence["score_explanation"] = score_explanation
    return evidence


def _checkout_provenance(context: Context) -> dict[str, Any]:
    citation = _context_citation(context)
    seq, event_hash = _citation_event_identity(citation)
    return {
        "citation": citation,
        "event_seq": seq,
        "event_hash": event_hash,
        "source": context.source,
        "source_lane": _checkout_source_lane(context),
        "valid_from": context.valid_from,
        "valid_to": context.valid_to,
    }


def _checkout_source_lanes(contexts: list[Context]) -> dict[str, int]:
    source_lanes: dict[str, int] = {}
    for context in contexts:
        lane = _checkout_source_lane(context)
        source_lanes[lane] = source_lanes.get(lane, 0) + 1
    return source_lanes


def _checkout_skills(contexts: list[Context], query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    skill_contexts = [
        context
        for context in contexts
        if (context.metadata or {}).get("entity_type") == "skill_version"
    ]
    if not skill_contexts:
        return []
    query_tokens = _checkout_tokens(query)
    skills: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for context in skill_contexts:
        metadata = context.metadata or {}
        skill_id = metadata.get("skill_id")
        entity_name = metadata.get("entity_name")
        if not isinstance(skill_id, str) or not skill_id:
            if isinstance(entity_name, str) and entity_name.startswith("skill:"):
                skill_id = entity_name.removeprefix("skill:").split(":v", 1)[0]
            else:
                continue
        version = str(metadata.get("version") or _skill_version_from_entity(entity_name) or "1")
        key = (skill_id, version)
        if key in seen:
            continue
        applicability = _metadata_text_list(metadata.get("applicability"))
        procedure = _metadata_text_list(metadata.get("procedure"))
        haystack = " ".join([context.content, *applicability, str(metadata.get("summary") or "")])
        if query_tokens and not (_checkout_tokens(haystack) & query_tokens):
            continue
        seen.add(key)
        skills.append(
            {
                "skill_id": skill_id,
                "version": version,
                "status": str(metadata.get("status") or "unknown"),
                "summary": str(metadata.get("summary") or context.content),
                "procedure": procedure,
                "applicability": applicability,
                "citation": _context_citation(context),
                "score": context.score,
            }
        )
        if len(skills) >= limit:
            break
    return skills


def _checkout_skill_analytics(contexts: list[Context]) -> dict[str, Any]:
    """Summarize skill outcome history without mutating Skill Memory."""
    versions: dict[tuple[str, str], dict[str, Any]] = {}
    outcomes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for context in contexts:
        metadata = context.metadata or {}
        entity_type = metadata.get("entity_type")
        if entity_type not in {"skill_version", "skill_outcome"}:
            continue
        key = _skill_context_key(metadata)
        if key is None:
            continue
        if entity_type == "skill_version":
            versions[key] = {
                "skill_id": key[0],
                "version": key[1],
                "status": str(metadata.get("status") or "unknown"),
                "citation": _context_citation(context),
                "failure_modes": _metadata_text_list(metadata.get("failure_modes")),
                "rollback": str(metadata.get("rollback") or "").strip(),
            }
        else:
            outcomes.setdefault(key, []).append(
                {
                    "success_score": _optional_float(metadata.get("success_score")),
                    "feedback": str(metadata.get("feedback") or "").casefold().strip(),
                    "citation": _context_citation(context),
                }
            )

    promotion_candidates: list[dict[str, Any]] = []
    rollback_candidates: list[dict[str, Any]] = []
    contradicted_keys = {
        key for key, version in versions.items() if version["status"] == "contradicted"
    }
    for key in sorted(set(versions) | set(outcomes)):
        version = versions.get(
            key,
            {
                "skill_id": key[0],
                "version": key[1],
                "status": "unknown",
                "citation": "",
                "failure_modes": [],
                "rollback": "",
            },
        )
        outcome_items = outcomes.get(key, [])
        scores = [
            score
            for item in outcome_items
            if (score := item.get("success_score")) is not None
        ]
        average_score = round(sum(scores) / len(scores), 4) if scores else None
        success_count = sum(
            1
            for item in outcome_items
            if _skill_outcome_is_success(item.get("feedback"), item.get("success_score"))
        )
        failure_count = sum(
            1
            for item in outcome_items
            if _skill_outcome_is_failure(item.get("feedback"), item.get("success_score"))
        )
        latest_citation = _latest_skill_citation(version, outcome_items)
        status = version["status"]
        if (
            status in {"validated", "revised", "outcome_recorded"}
            and key not in contradicted_keys
            and success_count > 0
            and failure_count == 0
            and (average_score is None or average_score >= 0.8)
        ):
            promotion_candidates.append(
                {
                    "skill_id": key[0],
                    "version": key[1],
                    "status": status,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "average_success_score": average_score,
                    "latest_citation": latest_citation,
                }
            )
        rollback_reason = _skill_rollback_reason(status, success_count, failure_count, average_score)
        if rollback_reason is not None:
            rollback_candidates.append(
                {
                    "skill_id": key[0],
                    "version": key[1],
                    "status": status,
                    "reason": rollback_reason,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "average_success_score": average_score,
                    "failure_modes": version["failure_modes"],
                    "rollback": version["rollback"],
                    "latest_citation": latest_citation,
                }
            )

    return {
        "version_count": len(versions),
        "outcome_count": sum(len(items) for items in outcomes.values()),
        "contradiction_count": len(contradicted_keys),
        "promotion_candidates": promotion_candidates[:5],
        "rollback_candidates": rollback_candidates[:5],
    }


def _skill_context_key(metadata: dict[str, Any]) -> tuple[str, str] | None:
    skill_id = metadata.get("skill_id")
    entity_name = metadata.get("entity_name")
    if not isinstance(skill_id, str) or not skill_id:
        if isinstance(entity_name, str) and entity_name.startswith("skill:"):
            skill_id = entity_name.removeprefix("skill:").split(":v", 1)[0]
        else:
            return None
    version = str(metadata.get("version") or _skill_version_from_entity(entity_name) or "1")
    return skill_id, version


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _skill_outcome_is_success(feedback: object, score: object) -> bool:
    normalized = str(feedback or "").casefold().strip()
    numeric_score = score if isinstance(score, float) else None
    return normalized in {"used", "helpful", "passed", "success"} or (
        numeric_score is not None and numeric_score >= 0.8
    )


def _skill_outcome_is_failure(feedback: object, score: object) -> bool:
    normalized = str(feedback or "").casefold().strip()
    numeric_score = score if isinstance(score, float) else None
    return normalized in {"failed", "failure", "irrelevant", "regressed"} or (
        numeric_score is not None and numeric_score < 0.5
    )


def _latest_skill_citation(version: dict[str, Any], outcomes: list[dict[str, Any]]) -> str:
    for outcome in reversed(outcomes):
        citation = outcome.get("citation")
        if isinstance(citation, str) and citation:
            return citation
    citation = version.get("citation")
    return citation if isinstance(citation, str) else ""


def _skill_rollback_reason(
    status: str,
    success_count: int,
    failure_count: int,
    average_score: float | None,
) -> str | None:
    if status in {"contradicted", "deprecated"}:
        return status
    if failure_count > success_count:
        return "failed_outcomes"
    if average_score is not None and average_score < 0.5:
        return "low_success_score"
    return None


def _metadata_text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            texts.append(text)
    return texts


def _skill_version_from_entity(value: object) -> str | None:
    if not isinstance(value, str) or ":v" not in value:
        return None
    version = value.rsplit(":v", 1)[1].strip()
    return version or None


def _checkout_recall_limit(query: str, limit: int) -> int:
    """Return the internal recall budget for checkout without inflating prompt context."""
    prompt_limit = validate_limit(limit)
    plan = build_evidence_plan(query, limit=max(prompt_limit, 10))
    if not plan.promote_cited_sources:
        return prompt_limit
    source_budget = source_lane_candidate_limit(query, limit=max(prompt_limit, 10))
    return min(
        MAX_QUERY_LIMIT,
        max(
            prompt_limit,
            source_budget,
            plan.required_source_groups * 8,
        ),
    )


def _checkout_source_lane(context: Context) -> str:
    metadata = context.metadata or {}
    lane = metadata.get("assembly_lane")
    if isinstance(lane, str) and lane:
        return lane
    if context.source in {"verbatim", "packet_memory", "projection", "eventloom"}:
        return context.source
    return "graph"


def _context_citation(context: Context) -> str | None:
    metadata = context.metadata or {}
    citation = metadata.get("citation")
    return citation if isinstance(citation, str) and citation else None


def _contexts_as_of_seq(contexts: list[Context], as_of_seq: int) -> list[Context]:
    filtered = []
    for context in contexts:
        citation = _context_citation(context)
        seq, _event_hash = _citation_event_identity(citation)
        if seq is None or seq <= as_of_seq:
            filtered.append(context)
    return filtered


def _checkout_rank(context: Context, query: str) -> tuple[float, int, int, int, str, float]:
    query_tokens = _checkout_tokens(query)
    content_tokens = _checkout_tokens(context.content)
    overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
    metadata = context.metadata or {}
    entity_type = metadata.get("entity_type")
    type_priority = 1 if entity_type in {"task", "decision", "goal", "memory"} else 0
    citation_priority = 1 if _context_citation(context) else 0
    source_lane = _checkout_source_lane(context)
    source_priority = 1 if source_lane in {"verbatim", "eventloom", "projection"} else 0
    return (
        overlap,
        citation_priority,
        source_priority,
        type_priority,
        context.valid_from or "",
        context.score,
    )


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


def _context_identity(context: Context) -> dict[str, str]:
    metadata = context.metadata or {}
    entity_name = metadata.get("entity_name")
    entity_type = metadata.get("entity_type")
    if isinstance(entity_name, str) and entity_name.strip():
        name = entity_name.strip()
    else:
        name = _context_content_identity(context.content)
    if isinstance(entity_type, str) and entity_type.strip():
        kind = entity_type.strip()
    elif context.source == "packet_memory":
        kind = "packet_memory"
    else:
        kind = "memory"
    return {"entity_name": name, "entity_type": kind}


def _context_feedback_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "source_kind",
        "source_event_seq",
        "source_event_hash",
        "provider_path",
        "model",
    }
    return {
        key: value
        for key, value in metadata.items()
        if key in allowed and isinstance(value, str | int | float | bool)
    }


def _context_content_identity(content: str) -> str:
    text = content.strip()
    if not text:
        return "context"
    if " (" in text:
        return text.split(" (", 1)[0].strip() or "context"
    return text.split(" — ", 1)[0].strip() or "context"


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
