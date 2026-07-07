"""Query, retrieval, pagination, and verbatim operations for MemoryFabric (phase 4).

Extracted per ``docs/superpowers/specs/2026-07-06-fabric-decomposition-design.md``
following the phase-1/2/3 pattern: :class:`QueryEngine` owns the read-path
cluster behind a structural :class:`QueryHost` protocol, and ``MemoryFabric``
delegates.

Phase-4 seam notes (all late-bound, evidence-driven):

- ``QueryRouter`` and ``build_reranker`` are patch-targeted fabric globals but
  are only used at *construction* time, which stays in ``MemoryFabric.__init__``
  — the engine reads the resulting ``host.query_router`` attribute per call.
- ``get_metrics`` routes through two host seams: ``_record_degraded_operation``
  and the phase-4 ``_record_query_metric``.
- ``source_synthesis_bundle_result`` is patch-targeted, so the synthesis bundle
  builds through ``host._source_synthesis_bundle_result``.
- Cache STORAGE stays on the fabric (``_query_page_cache``,
  ``_event_ref_index_cache``, ``_session_cue_index_cache``): ``close()`` resets
  them as part of fabric lifecycle and ``__new__``-constructed partial fabrics
  expect attribute-level state. Ownership here means the engine is the only
  code that reads or writes them.
- Intra-cluster calls to public methods (``query_page -> query``,
  ``query -> retrieve``, ``_query_source_lane -> query_verbatim``) route via
  the host — tests instance-patch ``fabric.query`` (and peers) after
  construction.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

from zaxy.compaction import search_compaction_projections
from zaxy.context import Context
from zaxy.core.checkout_build import (
    _citation_event_identity,
    _context_citation,
    _event_citation,
    _event_content,
    _packet_memory_reinforcements,
    _prefer_verbatim_for_duplicate_source_groups,
    _source_context_text,
    _synthesis_packet_metadata,
    _tokens,
)
from zaxy.core.models import QueryPage
from zaxy.event import ReplayResult
from zaxy.forgetting import FORGOTTEN_MARKER_KEY
from zaxy.pagination import encode_query_cursor, validate_query_cursor
from zaxy.query import ScoringProfile
from zaxy.retrieval_cache import _eventlog_file_signature
from zaxy.retrieval_intent import classify_retrieval_intent
from zaxy.retrieval_plan import (
    absence_check_bundle,
    bridge_source_lane_queries,
    should_try_absence_bundle_first,
    source_context_group,
    source_lane_queries,
)
from zaxy.salience import (
    CUE_MATCH_WEIGHT,
    REINFORCEMENT_EVENT_TYPE,
    cue_overlap,
    cue_pairs,
    event_ref_index,
)
from zaxy.security import MAX_QUERY_LIMIT, validate_limit, validate_query, validate_session_id
from zaxy.verbatim import VerbatimIndex

__all__ = [
    "QUERY_PAGE_CACHE_MAX_ENTRIES",
    "QUERY_PAGE_CACHE_TTL_SECONDS",
    "QueryEngine",
    "QueryHost",
]

QUERY_PAGE_CACHE_TTL_SECONDS = 30.0
QUERY_PAGE_CACHE_MAX_ENTRIES = 32


class QueryHost(Protocol):
    """The exact fabric surface the query cluster depends on."""

    # Components (constructed in fabric __init__, where patched globals resolve).
    session_manager: Any
    tracer: Any
    projections: Any
    embedding_provider: Any
    query_router: Any
    context_assembly_policy: Any
    retrieval_profile: Any
    _retrieval_cache: Any
    _connected: bool
    # Cache storage (engine-owned logically; fabric-owned for lifecycle).
    _query_page_cache: dict[
        tuple[str, str, str | None, tuple[float, ...] | None],
        tuple[float, int, tuple[int, int] | None, list[Context]],
    ]
    _event_ref_index_cache: dict[str, tuple[tuple[int, int], dict[int, tuple[str, str]]]]
    _session_cue_index_cache: dict[str, tuple[tuple[int, int], dict[int, frozenset[str]]]]

    async def connect(self) -> None: ...

    async def query(
        self,
        query: str,
        temporal_point: str | None = ...,
        limit: int = ...,
        embedding: list[float] | None = ...,
        session_id: str = ...,
        include_source_lane: bool = ...,
        scoring_profile: str | ScoringProfile | None = ...,
        cues: dict[str, str] | None = ...,
    ) -> list[Context]: ...

    async def retrieve(
        self,
        query: str,
        temporal_point: str | None = ...,
        limit: int = ...,
        embedding: list[float] | None = ...,
        session_id: str = ...,
        trace: bool = ...,
        scoring_profile: str | ScoringProfile | None = ...,
    ) -> list[Context]: ...

    async def query_verbatim(
        self, query: str, *, session_id: str = ..., limit: int = ...
    ) -> list[Context]: ...

    async def _warm_projection_session(self, session_id: str) -> None: ...

    def _decrypt_event_view(self, event: Any) -> Any: ...

    def _record_degraded_operation(self, operation: str, reason: str) -> None: ...

    def _record_query_metric(self, duration_seconds: float, *, source: str | None = ...) -> None: ...

    def _source_synthesis_bundle_result(self, **kwargs: Any) -> Any: ...


class QueryEngine:
    """Hybrid retrieval, source-lane assembly, pagination, and verbatim reads.

    Method bodies are moved verbatim from ``MemoryFabric``; only the ``self``
    surface was renamed to the injected host.
    """

    def __init__(self, *, host: QueryHost) -> None:
        self._host = host

    async def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
        session_id: str = "default",
        include_source_lane: bool = True,
        scoring_profile: str | ScoringProfile | None = None,
        cues: dict[str, str] | None = None,
    ) -> list[Context]:
        """Return answer-ready context assembled from retrieval and source evidence.

        ``cues`` (optional, additive) carries the caller's encoding-specificity
        context (``mission``/``workspace``/``tool``/``phase``). It only affects
        ranking under the cognitive retrieval profile; explicit queries never
        route through the salience attenuation floor, so attenuated memories
        stay fully reachable here.
        """
        import time

        validate_query(query)
        sid = validate_session_id(session_id)
        start = time.perf_counter()
        contexts = await self._host.retrieve(
            query,
            temporal_point=temporal_point,
            limit=limit,
            embedding=embedding,
            session_id=sid,
            scoring_profile=scoring_profile,
        )
        source_contexts = (
            await self._query_source_lane(query, contexts, sid, limit)
            if include_source_lane
            else []
        )
        if source_contexts:
            contexts = self._host.context_assembly_policy.assemble(
                contexts,
                source_contexts,
                [],
                limit=limit,
                query=query,
            )
        else:
            contexts = contexts[:limit]
        contexts = self._merge_projection_contexts(contexts, query, limit)
        contexts = self._blend_query_cues(contexts, cues=cues, session_id=sid)
        duration_ms = (time.perf_counter() - start) * 1000

        await self._trace_query_best_effort(query, len(contexts), duration_ms, temporal_point)

        query_source = "eventloom" if contexts and all(context.source == "eventloom" for context in contexts) else None
        if query_source is None:
            self._host._record_query_metric(duration_ms / 1000.0)
        else:
            self._host._record_query_metric(duration_ms / 1000.0, source=query_source)

        return contexts

    async def retrieve(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
        session_id: str = "default",
        trace: bool = False,
        scoring_profile: str | ScoringProfile | None = None,
    ) -> list[Context]:
        """Retrieve backend evidence without source-lane answer assembly."""
        import time

        validate_query(query)
        sid = validate_session_id(session_id)
        start = time.perf_counter()
        if not self._host._connected:
            try:
                await self._host.connect()
            except Exception:
                self._host._record_degraded_operation("query", "graph_unavailable")
                chunks = self._merge_projection_contexts(
                    self._query_eventlog_fallback(query, sid, limit, reason="graph unavailable"),
                    query,
                    limit,
                )
                duration_ms = (time.perf_counter() - start) * 1000
                if trace:
                    await self._trace_query_best_effort(query, len(chunks), duration_ms, temporal_point)
                if trace:
                    self._host._record_query_metric(duration_ms / 1000.0, source="eventloom")
                return chunks
        await self._host._warm_projection_session(sid)

        query_embedding = embedding
        if query_embedding is None and self._host.embedding_provider is not None:
            try:
                query_embedding = self._host.embedding_provider.embed(query)
            except Exception:
                self._host._record_degraded_operation("query", "embedding_provider_unavailable")
                query_embedding = None

        try:
            router_chunks = await self._host.query_router.query(
                query,
                temporal_point=temporal_point,
                limit=limit,
                embedding=query_embedding,
                session_id=sid,
                scoring_profile=scoring_profile,
            )
        except Exception:
            self._host._record_degraded_operation("query", "graph_retrieval_unavailable")
            contexts = self._merge_projection_contexts(
                self._query_eventlog_fallback(query, sid, limit, reason="graph retrieval unavailable"),
                query,
                limit,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            if trace:
                await self._trace_query_best_effort(query, len(contexts), duration_ms, temporal_point)
            if trace:
                self._host._record_query_metric(duration_ms / 1000.0, source="eventloom")
            return contexts
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
        contexts = contexts[:limit]
        duration_ms = (time.perf_counter() - start) * 1000

        if trace:
            await self._trace_query_best_effort(query, len(contexts), duration_ms, temporal_point)

        if trace:
            self._host._record_query_metric(duration_ms / 1000.0)

        return contexts

    async def _query_source_lane(
        self,
        query: str,
        graph_contexts: list[Context],
        session_id: str,
        limit: int,
    ) -> list[Context]:
        """Return bounded verbatim source evidence for raw query results."""
        candidate_limit = self._host.context_assembly_policy.verbatim_candidate_limit(
            query=query,
            limit=limit,
        )
        if candidate_limit <= 0:
            return []
        graph_texts = [context.content for context in graph_contexts]
        source_contexts: list[Context] = []
        seen: set[tuple[str, str]] = set()
        queries = self._ordered_source_lane_queries(query, graph_texts, limit)
        for source_query in queries:
            try:
                hits = await self._host.query_verbatim(
                    source_query, limit=candidate_limit, session_id=session_id
                )
            except Exception:
                self._host._record_degraded_operation("query", "source_lane_unavailable")
                continue
            self._extend_unique_source_contexts(source_contexts, hits, seen)
            for bridge_query in bridge_source_lane_queries(query, [context.content for context in hits]):
                try:
                    bridge_hits = await self._host.query_verbatim(
                        bridge_query,
                        limit=candidate_limit,
                        session_id=session_id,
                    )
                except Exception:
                    self._host._record_degraded_operation("query", "source_lane_unavailable")
                    continue
                self._extend_unique_source_contexts(source_contexts, bridge_hits, seen)
        for source_group in self._graph_source_groups_for_backfill(graph_contexts, limit):
            try:
                hits = await self._host.query_verbatim(
                    source_group, limit=max(1, min(candidate_limit, 3)), session_id=session_id
                )
            except Exception:
                self._host._record_degraded_operation("query", "source_lane_unavailable")
                continue
            self._extend_unique_source_contexts(source_contexts, hits, seen)
        source_contexts = self._order_source_contexts_for_assembly(query, source_contexts)
        return self._with_source_synthesis_bundle(query, graph_contexts, source_contexts, limit)

    @staticmethod
    def _ordered_source_lane_queries(query: str, graph_contexts: list[str], limit: int) -> list[str]:
        """Return source-lane queries in runtime recall order."""
        queries = list(source_lane_queries(query, graph_contexts))
        if len(queries) <= 1:
            return queries
        intent = classify_retrieval_intent(query, limit=limit)
        if {"aggregation", "aggregation_question"} & set(intent.reasons):
            return [*queries[1:], queries[0]]
        return queries

    def _with_source_synthesis_bundle(
        self,
        query: str,
        graph_contexts: list[Context],
        source_contexts: list[Context],
        limit: int,
    ) -> list[Context]:
        """Prepend compact multi-source evidence when the query needs synthesis."""
        synthesis_contexts = _prefer_verbatim_for_duplicate_source_groups(
            source_contexts,
            graph_contexts,
        )
        if not synthesis_contexts:
            return source_contexts
        preferred_source_groups = [
            source_context_group(_source_context_text(context))
            for context in graph_contexts
        ]
        source_kind = "source_absence"
        assembly_hint = "source_absence"
        synthesis_packet: dict[str, Any] | None = None
        if should_try_absence_bundle_first(query, limit=limit):
            bundle = absence_check_bundle(
                query=query,
                source_results=synthesis_contexts,
                limit=limit,
            )
            if bundle is None:
                source_kind = "source_synthesis"
                assembly_hint = "source_synthesis"
                result = self._host._source_synthesis_bundle_result(
                    query=query,
                    source_results=synthesis_contexts,
                    limit=limit,
                    preferred_source_groups=preferred_source_groups,
                )
                if result is not None:
                    bundle = result.content
                    synthesis_packet = result.packet
        else:
            source_kind = "source_synthesis"
            assembly_hint = "source_synthesis"
            result = self._host._source_synthesis_bundle_result(
                query=query,
                source_results=synthesis_contexts,
                limit=limit,
                preferred_source_groups=preferred_source_groups,
            )
            bundle = result.content if result is not None else None
            synthesis_packet = result.packet if result is not None else None
            if bundle is None:
                source_kind = "source_absence"
                assembly_hint = "source_absence"
                bundle = absence_check_bundle(
                    query=query,
                    source_results=synthesis_contexts,
                    limit=limit,
                )
        if bundle is None:
            return source_contexts
        synthesis = Context(
            content=bundle,
            source="verbatim",
            score=max((context.score for context in source_contexts), default=0.0) + 1.0,
            metadata={
                "source_kind": source_kind,
                "assembly_hint": assembly_hint,
                **_synthesis_packet_metadata(bundle, synthesis_packet),
            },
        )
        return [synthesis, *source_contexts]

    @staticmethod
    def _graph_source_groups_for_backfill(
        graph_contexts: list[Context],
        limit: int,
    ) -> list[str]:
        """Return graph-ranked provenance groups that should be verbatim-backfilled."""
        groups: list[str] = []
        seen: set[str] = set()
        for context in graph_contexts[: max(0, limit)]:
            group = source_context_group(_source_context_text(context))
            if group == "unknown" or group in seen:
                continue
            seen.add(group)
            groups.append(group)
        return groups

    @staticmethod
    def _order_source_contexts_for_assembly(
        query: str,
        source_contexts: list[Context],
    ) -> list[Context]:
        """Preserve source-rank order before synthesis performs typed ranking."""
        del query
        return source_contexts

    @staticmethod
    def _extend_unique_source_contexts(
        target: list[Context],
        contexts: list[Context],
        seen: set[tuple[str, str]],
    ) -> None:
        """Append source contexts once while preserving retrieval order."""
        for context in contexts:
            metadata = context.metadata or {}
            citation = str(metadata.get("citation") or "")
            key = (citation, context.content)
            if key in seen:
                continue
            seen.add(key)
            target.append(context)

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
        if cursor:
            # Continuation pages fetch the full ranked window once so later
            # pages slice the cached list instead of re-running retrieval.
            fetch_limit = MAX_QUERY_LIMIT
        cache_key = (
            validated_query,
            sid,
            temporal_point,
            tuple(embedding) if embedding is not None else None,
        )
        log_signature = self._query_page_log_signature(sid)
        contexts = self._cached_query_page_contexts(cache_key, fetch_limit, log_signature)
        if contexts is None:
            contexts = await self._host.query(
                validated_query,
                temporal_point=temporal_point,
                limit=fetch_limit,
                embedding=embedding,
                session_id=sid,
            )
            self._store_query_page_contexts(cache_key, fetch_limit, log_signature, contexts)
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

    def _query_page_log_signature(self, session_id: str) -> tuple[int, int] | None:
        """Return the session eventlog's (mtime_ns, size) freshness signature.

        Cached pages are bound to this signature so writers that bypass
        ``MemoryFabric.append`` (direct EventLog appends, other processes)
        still invalidate them. ``None`` means the log cannot be stat-ed;
        the cache then degrades to TTL-only freshness.
        """
        try:
            stat = os.stat(Path(self._host.session_manager.get(session_id).eventlog.path))
        except (OSError, TypeError, ValueError):
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _cached_query_page_contexts(
        self,
        key: tuple[str, str, str | None, tuple[float, ...] | None],
        fetch_limit: int,
        log_signature: tuple[int, int] | None,
    ) -> list[Context] | None:
        """Return cached ranked contexts when they already cover this page depth."""
        import time

        entry = self._host._query_page_cache.get(key)
        if entry is None:
            return None
        expires_at, cached_fetch_limit, cached_signature, contexts = entry
        if time.monotonic() >= expires_at or cached_signature != log_signature:
            del self._host._query_page_cache[key]
            return None
        if cached_fetch_limit >= fetch_limit or cached_fetch_limit >= MAX_QUERY_LIMIT:
            return contexts
        if len(contexts) < cached_fetch_limit:
            # The ranked result set was exhausted below the cached fetch
            # window, so a deeper fetch cannot add results.
            return contexts
        return None

    def _store_query_page_contexts(
        self,
        key: tuple[str, str, str | None, tuple[float, ...] | None],
        fetch_limit: int,
        log_signature: tuple[int, int] | None,
        contexts: list[Context],
    ) -> None:
        import time

        while len(self._host._query_page_cache) >= QUERY_PAGE_CACHE_MAX_ENTRIES:
            self._host._query_page_cache.pop(next(iter(self._host._query_page_cache)))
        self._host._query_page_cache[key] = (
            time.monotonic() + QUERY_PAGE_CACHE_TTL_SECONDS,
            fetch_limit,
            log_signature,
            contexts,
        )

    def _invalidate_query_page_cache(self, session_id: str) -> None:
        """Drop cached pages for a session after its memory changes."""
        self._host._query_page_cache = {
            key: value
            for key, value in self._host._query_page_cache.items()
            if key[1] != session_id
        }

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
        # Index build/extend + BM25 query are CPU-bound; offload so async callers
        # (the MCP checkout path) don't block the event loop.
        hits = await asyncio.to_thread(lambda: self._verbatim_index(sid).query(query, limit=limit))
        return [
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
            for hit in hits
        ]

    def _verbatim_index(self, session_id: str) -> VerbatimIndex:
        """Return a per-session verbatim index (cached, incrementally extended).

        Delegates to :class:`SessionRetrievalCache`; see its ``verbatim_index``.
        """
        return cast(VerbatimIndex, self._host._retrieval_cache.verbatim_index(session_id))

    def _session_event_ref_index(self, session_id: str) -> dict[int, tuple[str, str]]:
        """Return a cached seq -> (hash, type) index for the current log state.

        Follows the verbatim-index pattern: rebuilt whenever the Eventloom
        file signature changes, so reinforcement emitters that run outside a
        checkout (feedback) can canonicalize 12-char citation fragments into
        full-hash target refs without re-reading the log per call.
        """
        eventlog = self._host.session_manager.get(session_id).eventlog
        signature = _eventlog_file_signature(eventlog)
        cached = self._host._event_ref_index_cache.get(session_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        index = event_ref_index(eventlog.read_all())
        self._host._event_ref_index_cache[session_id] = (signature, index)
        return index

    def _session_cue_index(self, session_id: str) -> dict[int, frozenset[str]]:
        """Return a cached seq -> normalized-cue-pairs index for the session log.

        Follows the verbatim-index signature pattern; only events whose
        payload carries a well-formed ``cues`` record appear.
        """
        eventlog = self._host.session_manager.get(session_id).eventlog
        signature = _eventlog_file_signature(eventlog)
        cached = self._host._session_cue_index_cache.get(session_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        index: dict[int, frozenset[str]] = {}
        for event in eventlog.read_all():
            payload = getattr(event, "payload", None)
            seq = getattr(event, "seq", None)
            if not isinstance(payload, dict) or not isinstance(seq, int):
                continue
            pairs = cue_pairs(payload.get("cues"))
            if pairs:
                index[seq] = pairs
        self._host._session_cue_index_cache[session_id] = (signature, index)
        return index

    def _blend_query_cues(
        self,
        contexts: list[Context],
        *,
        cues: dict[str, str] | None,
        session_id: str,
    ) -> list[Context]:
        """Blend a bounded cue-overlap bonus into explicit query results.

        Active only under the cognitive retrieval profile and only when the
        caller provided cues; otherwise the input list is returned untouched
        (byte parity with the pre-cue contract). The bonus is
        ``CUE_MATCH_WEIGHT * jaccard`` added to the context score, and the
        list is re-sorted only when at least one bonus applied.
        """
        if not cues or not self._host.retrieval_profile.cue_blending or not contexts:
            return contexts
        query_cues = cue_pairs(cues)
        if not query_cues:
            return contexts
        cue_index = self._session_cue_index(session_id)
        if not cue_index:
            return contexts
        blended: list[Context] = []
        applied = False
        for context in contexts:
            seq, _event_hash = _citation_event_identity(_context_citation(context))
            stored = cue_index.get(seq) if seq is not None else None
            overlap = cue_overlap(query_cues, stored) if stored else 0.0
            if overlap <= 0.0:
                blended.append(context)
                continue
            applied = True
            metadata = dict(context.metadata or {})
            metadata["cue_overlap"] = round(overlap, 4)
            blended.append(
                replace(
                    context,
                    score=round(context.score + CUE_MATCH_WEIGHT * overlap, 4),
                    metadata=metadata,
                )
            )
        if not applied:
            return contexts
        return sorted(blended, key=lambda item: item.score, reverse=True)

    def _query_eventlog_fallback(
        self,
        query: str,
        session_id: str,
        limit: int,
        *,
        reason: str,
    ) -> list[Context]:
        """Return best-effort context from Eventloom when graph retrieval is down."""
        replay = self._host.session_manager.replay(session_id, from_seq=1)
        query_tokens = _tokens(query)
        contexts: list[Context] = []
        for raw_event in replay.events:
            event = self._host._decrypt_event_view(raw_event)
            if getattr(event, "type", None) == REINFORCEMENT_EVENT_TYPE:
                # Salience bookkeeping is never retrievable context.
                continue
            event_payload = getattr(event, "payload", None)
            if isinstance(event_payload, dict) and event_payload.get(FORGOTTEN_MARKER_KEY):
                # A forgotten memory is excluded from retrieval.
                continue
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
        if not self._host.projections:
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
                self._host.projections,
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
            await self._host.tracer.trace_query(query, result_count, duration_ms, temporal_point)

    async def replay(self, from_seq: int = 1, session_id: str = "default") -> ReplayResult:
        """Replay events from the log starting at a sequence number.

        Returns the full replay result including integrity verification. The
        verified replay is cached per session and extended incrementally as the
        append-only log grows: only newly appended events are parsed and
        integrity-checked (their seals plus the chain link to the cached,
        already-verified prefix) instead of re-reading and re-hashing the entire
        log on every call. A full re-verify happens on a cold cache, if the log
        shrank / was rewritten, or if incremental verification detects a break.

        The verified replay is CPU/IO-bound and runs in a worker thread so async
        callers (including the MCP checkout path) never block the event loop.
        """
        result = await asyncio.to_thread(
            self._host._retrieval_cache.verified_replay, session_id, from_seq
        )
        return cast(ReplayResult, result)
