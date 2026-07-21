"""MemoryFabric: the framework-agnostic Python memory API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from zaxy.causal import (
    CausalQueryResult,
)
from zaxy.compaction import (
    CompactionProjection,
    load_compaction_projection,
)
from zaxy.config import get_settings
from zaxy.context import Context, ContextAssemblyPolicy
from zaxy.core.checkout_build import (
    _compaction_projection_paths,
    _invalidation_source_id,
    build_memory_checkout,
    entity_reinforcement_targets,
)
from zaxy.core.fabric_checkout import CheckoutOps
from zaxy.core.fabric_coordination import CoordinationOps
from zaxy.core.fabric_query import (
    QUERY_PAGE_CACHE_MAX_ENTRIES as QUERY_PAGE_CACHE_MAX_ENTRIES,
)
from zaxy.core.fabric_query import (
    QUERY_PAGE_CACHE_TTL_SECONDS as QUERY_PAGE_CACHE_TTL_SECONDS,
)
from zaxy.core.fabric_query import (
    QueryEngine,
)
from zaxy.core.fabric_reasoning import ReasoningOps
from zaxy.core.fabric_write import (
    PRODUCER_REF_PAYLOAD_KEY as PRODUCER_REF_PAYLOAD_KEY,
)
from zaxy.core.fabric_write import (
    ForgetTombstoneUnauditedError as ForgetTombstoneUnauditedError,
)
from zaxy.core.fabric_write import (
    WriteEngine,
)
from zaxy.core.models import (
    ContextAssembly,
    ContextRefreshReport,
    HandoffBundle,
    MemoryCheckout,
    QueryPage,
)
from zaxy.core.reinforcement_queue import DeferredReinforcementQueue
from zaxy.embedding import build_embedding_provider
from zaxy.event import (  # noqa: F401 - ReplayResult re-export for existing tests
    EventLog,
    IntegrityReport,
    ReplayResult,
    verify_event_chain,
)
from zaxy.evolution_policy import (
    EvolutionGateDecision,
)
from zaxy.metrics import get_metrics
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store
from zaxy.purpose import PurposeProfile
from zaxy.query import QueryRouter, ScoringProfile, build_reranker, build_retention_policy
from zaxy.refs import MemoryRef, MemoryRefStore
from zaxy.retrieval_cache import SessionRetrievalCache
from zaxy.retrieval_plan import (
    source_synthesis_bundle_result,
)
from zaxy.retrieval_profile import (
    RetrievalProfile,
    apply_retrieval_profile,
    resolve_retrieval_profile,
)
from zaxy.salience import (
    build_invalidated_reinforcement_event,
)
from zaxy.security import (
    validate_session_id,
)
from zaxy.session import SessionManager
from zaxy.trace import MemoryTracer
from zaxy.verbatim import VerbatimIndex
from zaxy.workspace import (
    WorkspaceProfile,
    build_session_genesis_event,
    build_workspace_instruction_event,
    existing_session_genesis_profile,
    existing_workspace_instructions_signature,
    mark_workspace_instruction_event_updated,
    workspace_profile_from_payload,
)

# Query-page cache tuning lives with the engine; re-exported for compatibility.

# Reserved payload key carrying an external producer's source reference, used to
# dedup re-ingest of the same producer event (see ``MemoryFabric.append_batch``).
# Follows the ``__zaxy_*`` reserved-key convention.
# Producer-ref helpers, PRODUCER_REF_PAYLOAD_KEY, and
# ForgetTombstoneUnauditedError moved to zaxy.core.fabric_write (phase 5);
# re-exported for compatibility (tests import the error from this module).


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
        embedded_graph_path: str | Path | None = None,
        latticedb_path: str | Path | None = None,
        *,
        graph: Any | None = None,
        tracer: Any | None = None,
        session_manager: SessionManager | None = None,
        retrieval_cache: SessionRetrievalCache | None = None,
        refs: MemoryRefStore | None = None,
        owns_connections: bool = True,
    ) -> None:
        """Initialize fabric with configuration.

        All arguments default to environment variables (via Settings).
        Explicit values override env vars for framework integrations.

        Component injection: pass pre-built ``graph``/``tracer``/
        ``session_manager``/``retrieval_cache``/``refs`` to reuse a host's
        components instead of constructing new ones — this lets the MCP server
        hold one persistent fabric over its existing projection store rather than
        standing up a second one. With ``owns_connections=False`` the fabric
        treats those shared components as already connected and never connects or
        closes them (the host owns their lifecycle).
        """
        settings = get_settings()
        retrieval_profile = resolve_retrieval_profile(settings)
        resolved_settings = apply_retrieval_profile(settings, retrieval_profile)
        self.settings = resolved_settings
        self.retrieval_profile: RetrievalProfile = retrieval_profile
        self._owns_connections = owns_connections

        # External plugins extend extractors / projection backends in-process
        # (entry points + ZAXY_PLUGINS). Loading is isolated and idempotent; a
        # failure here is logged inside load_plugins and is never fatal.
        try:
            from zaxy.plugins import load_plugins

            load_plugins(resolved_settings)
        except Exception:
            get_metrics().record_degraded_operation("init", "plugin_load_unavailable")

        self.eventloom_path = Path(eventloom_path or resolved_settings.eventloom_path)
        self.session_manager = (
            session_manager
            if session_manager is not None
            else SessionManager(base_path=str(self.eventloom_path))
        )
        self.eventloom = self.session_manager.get("default").eventlog
        if graph is not None:
            self.graph = graph
        else:
            resolved_embedded_graph_path = (
                Path(embedded_graph_path)
                if embedded_graph_path is not None
                else self.eventloom_path / "projections" / "embedded.kuzu"
                if eventloom_path is not None
                else Path(resolved_settings.embedded_graph_path)
            )
            self.graph = build_projection_store(
                ProjectionBackendConfig(
                    backend=projection_backend or resolved_settings.projection_backend,
                    neo4j_uri=neo4j_uri or resolved_settings.neo4j_uri,
                    neo4j_user=neo4j_user or resolved_settings.neo4j_user,
                    neo4j_password=neo4j_password or resolved_settings.neo4j_password,
                    neo4j_ca_cert=neo4j_ca_cert if neo4j_ca_cert is not None else resolved_settings.neo4j_ca_cert,
                    neo4j_trust_all=neo4j_trust_all if neo4j_trust_all is not None else resolved_settings.neo4j_trust_all,
                    pggraph_dsn=pggraph_dsn or resolved_settings.pggraph_dsn,
                    embedded_graph_path=resolved_embedded_graph_path,
                    latticedb_path=Path(latticedb_path or resolved_settings.latticedb_path),
                    embedding_dimension=resolved_settings.embedding_dimension,
                )
            )
        self.query_router = QueryRouter(
            self.graph,
            default_limit=resolved_settings.query_default_limit,
            session_id=resolved_settings.eventloom_thread,
            scoring_profile=resolved_settings.query_scoring_profile,
            reranker=build_reranker(resolved_settings),
            retention_policy=build_retention_policy(resolved_settings),
            graph_walk_enabled=retrieval_profile.graph_walk,
        )
        self._salience_half_life_days = float(resolved_settings.salience_half_life_days)
        self._salience_floor = float(resolved_settings.salience_floor)
        self._encoding_gate_enabled = bool(resolved_settings.encoding_gate_enabled)
        self.embedding_provider = build_embedding_provider(resolved_settings)
        self.tracer = (
            tracer
            if tracer is not None
            else MemoryTracer(
                base_url=pathlight_url or resolved_settings.pathlight_url,
                project_id=pathlight_project_id or resolved_settings.pathlight_project_id,
                disabled=tracer_disabled or not resolved_settings.pathlight_enabled,
            )
        )
        projection_search_base = Path(eventloom_path or resolved_settings.eventloom_path)
        self.refs = refs if refs is not None else MemoryRefStore(projection_search_base)
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
        self._retrieval_cache = (
            retrieval_cache
            if retrieval_cache is not None
            else SessionRetrievalCache(self.session_manager)
        )
        self._event_ref_index_cache: dict[str, tuple[tuple[int, int], dict[int, tuple[str, str]]]] = {}
        self._session_cue_index_cache: dict[str, tuple[tuple[int, int], dict[int, frozenset[str]]]] = {}
        self._query_page_cache: dict[
            tuple[str, str, str | None, tuple[float, ...] | None],
            tuple[float, int, tuple[int, int] | None, list[Context]],
        ] = {}
        self._initialized_workspaces: dict[tuple[str, str], WorkspaceProfile] = {}
        self._initialized_instruction_signatures: dict[tuple[str, str], str] = {}
        self._warmed_projection_sessions: set[str] = set()
        self._reinforcement_queue = DeferredReinforcementQueue(sink=self)
        # Injected (shared) components are connected/closed by their owner, so an
        # injected fabric starts "connected" and its connect/close are no-ops for
        # those components.
        self._connected = not owns_connections


    @property
    def _reasoning(self) -> ReasoningOps:
        """Reasoning/metacognition collaborator (decomposition phase 1), lazily built.

        Lazy (not eager in ``__init__``) because several test suites construct
        partial fabrics via ``MemoryFabric.__new__`` to exercise pure methods;
        construction here needs nothing but ``self``. The host protocol
        late-binds every fabric lookup, so instance patches and runtime
        component swaps (graph-degraded fallback) keep intercepting exactly as
        before the extraction.
        """
        ops = self.__dict__.get("_reasoning_ops")
        if ops is None:
            ops = ReasoningOps(host=self)
            self.__dict__["_reasoning_ops"] = ops
        return cast(ReasoningOps, ops)

    @property
    def _coordination(self) -> CoordinationOps:
        """Coordination/fleet/handoff collaborator (phase 2), lazily built.

        Same contract as ``_reasoning``: lazy for ``__new__``-constructed
        partial fabrics; every host lookup late-binds so instance patches
        (e.g. ``fabric._fleet_manager``) keep intercepting.
        """
        ops = self.__dict__.get("_coordination_ops")
        if ops is None:
            ops = CoordinationOps(host=self)
            self.__dict__["_coordination_ops"] = ops
        return cast(CoordinationOps, ops)

    @property
    def _checkout(self) -> CheckoutOps:
        """Checkout/assembly/feedback/synthesis collaborator (phase 3), lazily built."""
        ops = self.__dict__.get("_checkout_ops")
        if ops is None:
            ops = CheckoutOps(host=self)
            self.__dict__["_checkout_ops"] = ops
        return cast(CheckoutOps, ops)

    def _build_memory_checkout(self, **kwargs: Any) -> MemoryCheckout:
        """Checkout-builder seam for collaborators.

        Resolves ``build_memory_checkout`` in THIS module so existing
        ``patch("zaxy.core.fabric.build_memory_checkout")`` targets keep
        intercepting the moved checkout path.
        """
        return build_memory_checkout(**kwargs)

    @property
    def _query_engine(self) -> QueryEngine:
        """Query/retrieval collaborator (phase 4), lazily built."""
        ops = self.__dict__.get("_query_engine_ops")
        if ops is None:
            ops = QueryEngine(host=self)
            self.__dict__["_query_engine_ops"] = ops
        return cast(QueryEngine, ops)

    def _record_query_metric(self, duration_seconds: float, *, source: str | None = None) -> None:
        """Query-metrics seam for collaborators (resolves get_metrics here)."""
        if source is None:
            get_metrics().record_query(duration_seconds)
        else:
            get_metrics().record_query(duration_seconds, source=source)

    def _source_synthesis_bundle_result(self, **kwargs: Any) -> Any:
        """Synthesis-bundle seam for collaborators.

        Resolves ``source_synthesis_bundle_result`` in THIS module so existing
        ``patch("zaxy.core.fabric.source_synthesis_bundle_result")`` targets
        keep intercepting the moved source lane.
        """
        return source_synthesis_bundle_result(**kwargs)

    @property
    def _write(self) -> WriteEngine:
        """Write-path collaborator (phase 5, the hub), lazily built."""
        ops = self.__dict__.get("_write_engine_ops")
        if ops is None:
            ops = WriteEngine(host=self)
            self.__dict__["_write_engine_ops"] = ops
        return cast(WriteEngine, ops)

    def _metrics(self) -> Any:
        """Metrics seam for collaborators.

        Resolves ``get_metrics`` in THIS module so existing
        ``patch("zaxy.core.fabric.get_metrics")`` targets keep intercepting
        every moved metrics call (event/upsert/query counters and degrades).
        """
        return get_metrics()

    def _record_degraded_operation(self, operation: str, reason: str) -> None:
        """Metrics seam for collaborators.

        Resolves ``get_metrics`` in THIS module so existing
        ``patch("zaxy.core.fabric.get_metrics")`` targets keep intercepting
        call sites that moved into collaborator modules.
        """
        get_metrics().record_degraded_operation(operation, reason)

    async def connect(self) -> None:
        """Connect to projection backend and tracer. Idempotent.

        No-op when components are injected (``owns_connections=False``): the host
        owns their connection lifecycle.
        """
        if self._connected:
            return
        await self.graph.connect()
        await self.graph.init_schema()
        await self._warm_projection_session(self.settings.eventloom_thread)
        self._warm_source_index(self.settings.eventloom_thread)
        await self.tracer.connect()
        self._connected = True

    async def close(self) -> None:
        """Close all connections. Idempotent.

        When components are injected (``owns_connections=False``) the host owns
        their lifecycle, so this is a no-op: the shared graph/tracer/retrieval
        cache stay open and this fabric stays connected and warm across calls.
        Tearing down here would drop ``_connected`` to ``False`` and force the
        next call to reopen an embedded store the process already holds a lock
        on — the contention this guard exists to prevent. Symmetric with
        :meth:`connect`, which is likewise a no-op once connected.

        Deferred reinforcement is flushed BEFORE that guard: an injected fabric
        skips connection teardown but must still not drop queued writes, and the
        MCP server closes an injected fabric after most tool calls.
        """
        await self.flush_pending_reinforcements()
        if not self._owns_connections:
            return
        await self.graph.close()
        await self.tracer.close()
        self._retrieval_cache.invalidate()
        self._event_ref_index_cache = {}
        self._session_cue_index_cache = {}
        self._query_page_cache = {}
        self._warmed_projection_sessions = set()
        self._connected = False

    # -- reasoning / metacognition primitives (delegated; decomposition phase 1) --
    # Bodies live in zaxy.core.fabric_reasoning.ReasoningOps. Delegations keep the
    # public surface, signatures, and instance-patchability exactly as before.

    async def query_causal_successors(
        self,
        entity_name: str,
        *,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[CausalQueryResult]:
        """Return directed causal effects of an entity."""
        return await self._reasoning.query_causal_successors(
            entity_name,
            relation_type=relation_type,
            depth=depth,
            temporal_point=temporal_point,
            session_id=session_id,
        )

    async def query_causal_predecessors(
        self,
        entity_name: str,
        *,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[CausalQueryResult]:
        """Return directed causal causes of an entity."""
        return await self._reasoning.query_causal_predecessors(
            entity_name,
            relation_type=relation_type,
            depth=depth,
            temporal_point=temporal_point,
            session_id=session_id,
        )

    async def explain_outcome(
        self,
        outcome: str,
        *,
        phase: str = "planning",
        session_id: str = "default",
        depth: int = 2,
    ) -> dict[str, Any]:
        """Explain an outcome with causal predecessors and cited checkout fallback."""
        return await self._reasoning.explain_outcome(
            outcome, phase=phase, session_id=session_id, depth=depth
        )

    async def propose_belief_update(
        self,
        claim: str,
        *,
        rationale: str,
        confidence: float,
        source_events: list[dict[str, Any]],
        phase: str = "reflection",
        session_id: str = "default",
        actor: str = "zaxy-reasoning",
    ) -> dict[str, Any]:
        """Append a cited, review-pending belief proposal and observe the primitive call."""
        return await self._reasoning.propose_belief_update(
            claim,
            rationale=rationale,
            confidence=confidence,
            source_events=source_events,
            phase=phase,
            session_id=session_id,
            actor=actor,
        )

    async def get_claim_confidence(
        self,
        claim: str,
        *,
        phase: str = "review",
        session_id: str = "default",
        limit: int = 5,
        record_assessment: bool = True,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Score cited support and conflict evidence for a claim."""
        return await self._reasoning.get_claim_confidence(
            claim,
            phase=phase,
            session_id=session_id,
            limit=limit,
            record_assessment=record_assessment,
            min_confidence=min_confidence,
        )

    async def retrieve_similar_procedures(
        self,
        query: str,
        *,
        phase: str = "planning",
        session_id: str = "default",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Retrieve cited Skill Memory or consolidation procedure candidates."""
        return await self._reasoning.retrieve_similar_procedures(
            query, phase=phase, session_id=session_id, limit=limit
        )

    async def record_known_unknown(
        self,
        question: str,
        *,
        reason: str,
        source_events: list[dict[str, Any]],
        claim_key: str,
        gap_type: str = "missing_evidence",
        reverify_query: str | None = None,
        phase: str = "review",
        session_id: str = "default",
        actor: str = "zaxy-reasoning",
    ) -> dict[str, Any]:
        """Append an open, non-authoritative known-unknown diagnostic event."""
        return await self._reasoning.record_known_unknown(
            question,
            reason=reason,
            source_events=source_events,
            claim_key=claim_key,
            gap_type=gap_type,
            reverify_query=reverify_query,
            phase=phase,
            session_id=session_id,
            actor=actor,
        )

    async def list_known_unknowns(
        self,
        *,
        session_id: str = "default",
        status: str = "open",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return replay-derived known unknowns for a session."""
        return await self._reasoning.list_known_unknowns(
            session_id=session_id, status=status, limit=limit
        )

    async def list_conflict_clusters(
        self,
        *,
        session_id: str = "default",
        unresolved_only: bool = True,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return replay-derived metacognitive conflict clusters."""
        return await self._reasoning.list_conflict_clusters(
            session_id=session_id, unresolved_only=unresolved_only, limit=limit
        )

    async def list_confidence_trajectory(
        self,
        claim: str,
        *,
        session_id: str = "default",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return append-only confidence trajectory points for a claim."""
        return await self._reasoning.list_confidence_trajectory(
            claim, session_id=session_id, limit=limit
        )

    async def list_reverification_needs(
        self,
        query: str | None = None,
        *,
        session_id: str = "default",
        limit: int = 10,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Return replay-derived claims and unknowns that need re-verification."""
        return await self._reasoning.list_reverification_needs(
            query, session_id=session_id, limit=limit, min_confidence=min_confidence
        )

    async def plan_from_procedures(
        self,
        goal: str,
        *,
        phase: str = "planning",
        session_id: str = "default",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Return a non-authoritative planning packet from applicable procedures."""
        return await self._reasoning.plan_from_procedures(
            goal, phase=phase, session_id=session_id, limit=limit
        )


    def _enqueue_reinforcement(self, spec: dict[str, Any], *, session_id: str) -> None:
        """Queue a reinforcement spec for append after the current read returns."""
        self._reinforcement_queue.enqueue(spec, session_id=session_id)

    async def flush_pending_reinforcements(self) -> None:
        """Append every deferred salience reinforcement queued so far.

        Idempotent, and safe to call when nothing is pending. Callers that need
        the next checkout to rank against this turn's reinforcement must await
        this first; salience is otherwise at most one flush window stale.
        """
        await self._reinforcement_queue.flush()

    async def _append_event_spec(self, event: dict[str, Any], *, session_id: str) -> Any:
        return await self.append(
            str(event["event_type"]),
            actor=str(event["actor"]),
            payload=cast(dict[str, Any], event["payload"]),
            session_id=session_id,
        )


    async def _warm_projection_session(self, session_id: str) -> None:
        """Warm optional backend read indexes once per session."""
        if session_id in self._warmed_projection_sessions:
            return
        warm_session = getattr(self.graph, "warm_session", None)
        if warm_session is None:
            self._warmed_projection_sessions.add(session_id)
            return
        try:
            await warm_session(session_id=session_id)
        except Exception:
            get_metrics().record_degraded_operation("query", "projection_warmup_unavailable")
        self._warmed_projection_sessions.add(session_id)

    def _warm_source_index(self, session_id: str) -> None:
        """Warm Eventloom source recall index for the active session."""
        try:
            self._verbatim_index(session_id)
        except Exception:
            get_metrics().record_degraded_operation("query", "source_index_warmup_unavailable")

    # -- write path / evolution / editability / forgetting / ingest (phase 5) --
    # Bodies live in zaxy.core.fabric_write.WriteEngine.

    async def append(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None = None,
        thread: str = "default",
        session_id: str | None = None,
        *,
        forgettable: bool = False,
    ) -> Any:
        """Append a typed event to the immutable log and project to the graph.

        See :meth:`zaxy.core.fabric_write.WriteEngine.append`.
        """
        return await self._write.append(
            event_type,
            actor,
            payload=payload,
            thread=thread,
            session_id=session_id,
            forgettable=forgettable,
        )

    async def append_batch(
        self,
        items: list[dict[str, Any]],
        *,
        session_id: str | None = None,
    ) -> list[Any]:
        """Ingest a batch of external-producer events under one atomic seal.

        See :meth:`zaxy.core.fabric_write.WriteEngine.append_batch`.
        """
        return await self._write.append_batch(items, session_id=session_id)

    async def evaluate_evolution_gate(
        self,
        op: str,
        confidence: float,
        *,
        candidate_ref: dict[str, Any] | None = None,
        actor: str = "zaxy-evolution",
        session_id: str | None = None,
    ) -> EvolutionGateDecision:
        """Evaluate the governed memory-evolution policy for one op and record it."""
        return await self._write.evaluate_evolution_gate(
            op, confidence, candidate_ref=candidate_ref, actor=actor, session_id=session_id
        )

    async def record_outcome(
        self,
        *,
        outcome: str,
        summary: str,
        target_seq: int | None = None,
        target_hash: str | None = None,
        lesson: str | None = None,
        trigger: str | None = None,
        confidence: float | None = None,
        task_id: str | None = None,
        prior: float | None = None,
        actor: str = "zaxy-agent",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Record an outcome on recalled memory and run the governed learning loop."""
        return await self._write.record_outcome(
            outcome=outcome,
            summary=summary,
            target_seq=target_seq,
            target_hash=target_hash,
            lesson=lesson,
            trigger=trigger,
            confidence=confidence,
            task_id=task_id,
            prior=prior,
            actor=actor,
            session_id=session_id,
        )

    async def edit_memory(
        self,
        *,
        target_seq: int,
        target_hash: str,
        new_content: str,
        reason: str,
        actor: str = "zaxy-editor",
        confidence: float = 1.0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Re-ingest a human edit as a cited, non-authoritative ``memory.corrected`` event."""
        return await self._write.edit_memory(
            target_seq=target_seq,
            target_hash=target_hash,
            new_content=new_content,
            reason=reason,
            actor=actor,
            confidence=confidence,
            session_id=session_id,
        )

    async def rollback_memory(
        self,
        *,
        target_seq: int,
        target_hash: str,
        reason: str,
        actor: str = "zaxy-editor",
        confidence: float = 1.0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Reverse a prior evolution with a cited ``memory.rolled_back`` event."""
        return await self._write.rollback_memory(
            target_seq=target_seq,
            target_hash=target_hash,
            reason=reason,
            actor=actor,
            confidence=confidence,
            session_id=session_id,
        )

    async def verified_forget(
        self,
        *,
        target_seq: int,
        target_hash: str,
        reason: str,
        actor: str = "zaxy-forgetter",
        confidence: float = 1.0,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Cryptographically erase a forgettable memory (verified forgetting / I5b)."""
        return await self._write.verified_forget(
            target_seq=target_seq,
            target_hash=target_hash,
            reason=reason,
            actor=actor,
            confidence=confidence,
            session_id=session_id,
        )

    def _decrypt_event_view(self, event: Any) -> Any:
        return self._write._decrypt_event_view(event)

    async def _project_event(self, event: Any, *, session_id: str) -> None:
        """Extract, project, trace, and record metrics for one sealed event."""
        await self._write._project_event(event, session_id=session_id)

    async def _append_generated_inferences(
        self,
        eventlog: EventLog,
        *,
        source_event: Any,
        session_id: str,
    ) -> None:
        await self._write._append_generated_inferences(
            eventlog, source_event=source_event, session_id=session_id
        )

    async def ingest_documents(
        self,
        path: str | Path,
        *,
        session_id: str = "default",
        max_lines: int = 80,
    ) -> int:
        """Ingest local Markdown/text documents as cited memory events."""
        return await self._write.ingest_documents(path, session_id=session_id, max_lines=max_lines)

    async def ingest_codebase(
        self,
        path: str | Path,
        *,
        session_id: str = "default",
        max_bytes: int = 512 * 1024,
    ) -> int:
        """Ingest local codebase file, symbol, and import mapping events."""
        return await self._write.ingest_codebase(path, session_id=session_id, max_bytes=max_bytes)

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
        return await self._write.refresh_context(
            path,
            kind=kind,
            session_id=session_id,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    async def ingest_transcript(
        self,
        turns: list[dict[str, Any]],
        *,
        source: str = "transcript",
        session_id: str = "default",
    ) -> int:
        """Ingest sanitized transcript turns as Eventloom-backed memory."""
        return await self._write.ingest_transcript(turns, source=source, session_id=session_id)

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

    # -- query / retrieval / pagination / verbatim (delegated; phase 4) -------
    # Bodies live in zaxy.core.fabric_query.QueryEngine.

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

        See :meth:`zaxy.core.fabric_query.QueryEngine.query`.
        """
        return await self._query_engine.query(
            query,
            temporal_point=temporal_point,
            limit=limit,
            embedding=embedding,
            session_id=session_id,
            include_source_lane=include_source_lane,
            scoring_profile=scoring_profile,
            cues=cues,
        )

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
        return await self._query_engine.retrieve(
            query,
            temporal_point=temporal_point,
            limit=limit,
            embedding=embedding,
            session_id=session_id,
            trace=trace,
            scoring_profile=scoring_profile,
        )

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

        See :meth:`zaxy.core.fabric_query.QueryEngine.query_page`.
        """
        return await self._query_engine.query_page(
            query,
            temporal_point=temporal_point,
            limit=limit,
            embedding=embedding,
            session_id=session_id,
            cursor=cursor,
        )

    async def query_verbatim(
        self,
        query: str,
        *,
        session_id: str = "default",
        limit: int = 10,
    ) -> list[Context]:
        """Retrieve exact Eventloom source chunks without requiring graph services."""
        return await self._query_engine.query_verbatim(query, session_id=session_id, limit=limit)

    async def replay(self, from_seq: int = 1, session_id: str = "default") -> ReplayResult:
        """Replay events from the log starting at a sequence number.

        See :meth:`zaxy.core.fabric_query.QueryEngine.replay`.
        """
        return await self._query_engine.replay(from_seq=from_seq, session_id=session_id)

    def _verbatim_index(self, session_id: str) -> VerbatimIndex:
        return self._query_engine._verbatim_index(session_id)

    def _session_event_ref_index(self, session_id: str) -> dict[int, tuple[str, str]]:
        return self._query_engine._session_event_ref_index(session_id)

    def _cached_query_page_contexts(
        self,
        key: tuple[str, str, str | None, tuple[float, ...] | None],
        fetch_limit: int,
        log_signature: tuple[int, int] | None,
    ) -> list[Context] | None:
        return self._query_engine._cached_query_page_contexts(key, fetch_limit, log_signature)

    def _store_query_page_contexts(
        self,
        key: tuple[str, str, str | None, tuple[float, ...] | None],
        fetch_limit: int,
        log_signature: tuple[int, int] | None,
        contexts: list[Context],
    ) -> None:
        self._query_engine._store_query_page_contexts(key, fetch_limit, log_signature, contexts)

    def _query_page_log_signature(self, session_id: str) -> tuple[int, int] | None:
        return self._query_engine._query_page_log_signature(session_id)

    def _invalidate_query_page_cache(self, session_id: str) -> None:
        """Drop cached pages for a session after its memory changes."""
        self._query_engine._invalidate_query_page_cache(session_id)

    def _query_eventlog_fallback(
        self,
        query: str,
        session_id: str,
        limit: int,
        *,
        reason: str,
    ) -> list[Context]:
        return self._query_engine._query_eventlog_fallback(
            query, session_id, limit, reason=reason
        )

    def _recent_packet_memory_contexts(self, events: list[Any]) -> list[Context]:
        return self._query_engine._recent_packet_memory_contexts(events)

    @staticmethod
    def _order_source_contexts_for_assembly(
        query: str,
        source_contexts: list[Context],
    ) -> list[Context]:
        """Preserve source-rank order (kept static: tests call it on the fabric)."""
        return QueryEngine._order_source_contexts_for_assembly(query, source_contexts)

    # -- checkout / assembly / consolidation / feedback / synthesis (phase 3) --
    # Bodies live in zaxy.core.fabric_checkout.CheckoutOps.

    async def propose_consolidation_candidates(
        self,
        *,
        session_id: str = "default",
        actor: str = "zaxy-consolidation",
        purpose: str | None = None,
        window_size: int = 8,
    ) -> dict[str, Any]:
        """Append cited, review-pending consolidation candidates for a session log."""
        return await self._checkout.propose_consolidation_candidates(
            session_id=session_id, actor=actor, purpose=purpose, window_size=window_size
        )

    async def consolidation_status(self, *, session_id: str = "default") -> dict[str, Any]:
        """Summarize consolidation candidate and review state from Eventloom replay."""
        return await self._checkout.consolidation_status(session_id=session_id)

    def _fleet_manager(self) -> Any:
        """Return a FleetManager bound to this fabric's session manager.

        Kept as a fabric method (delegating to the coordination collaborator)
        because tests instance-patch it before driving the fleet lane; the
        moved lane resolves the manager back through this host method.
        """
        return self._coordination.fleet_manager()

    def _fleet_lane_contexts(
        self, fleet_ids: list[str] | None, *, agent_id: str
    ) -> list[Context]:
        """Resolve the enrollment-gated fleet lane (see CoordinationOps)."""
        return self._coordination.fleet_lane_contexts(fleet_ids, agent_id=agent_id)

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
        purpose: PurposeProfile | dict[str, Any] | str | None = None,
        cues: dict[str, str] | None = None,
        fleet_ids: list[str] | None = None,
        agent_id: str | None = None,
        long_horizon: bool | None = None,
    ) -> ContextAssembly:
        """Assemble recent replay plus retrieval into prompt-ready context.

        See :meth:`zaxy.core.fabric_checkout.CheckoutOps.assemble_context`.
        """
        return await self._checkout.assemble_context(
            query,
            session_id=session_id,
            replay_from_seq=replay_from_seq,
            limit=limit,
            recall_limit=recall_limit,
            max_recent_events=max_recent_events,
            as_of_seq=as_of_seq,
            purpose=purpose,
            cues=cues,
            fleet_ids=fleet_ids,
            agent_id=agent_id,
            long_horizon=long_horizon,
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
        purpose: PurposeProfile | dict[str, Any] | str | None = None,
        record_reinforcement: bool = True,
        cues: dict[str, str] | None = None,
        fleet_ids: list[str] | None = None,
        agent_id: str | None = None,
        long_horizon: bool | None = None,
    ) -> MemoryCheckout:
        """Checkout the current cited memory state an agent should condition on.

        See :meth:`zaxy.core.fabric_checkout.CheckoutOps.checkout_memory`.
        """
        return await self._checkout.checkout_memory(
            query,
            session_id=session_id,
            replay_from_seq=replay_from_seq,
            limit=limit,
            max_recent_events=max_recent_events,
            ref=ref,
            purpose=purpose,
            record_reinforcement=record_reinforcement,
            cues=cues,
            fleet_ids=fleet_ids,
            agent_id=agent_id,
            long_horizon=long_horizon,
        )

    def _resolve_checkout_ref(self, ref: str | None, *, session_id: str) -> MemoryRef | None:
        return self._checkout._resolve_checkout_ref(ref, session_id=session_id)

    async def record_context_feedback(
        self,
        contexts: list[Context],
        *,
        feedback: str,
        session_id: str = "default",
        actor: str = "zaxy",
        importance: float | None = None,
        purpose: PurposeProfile | dict[str, Any] | str | None = None,
        outcome: str | None = None,
    ) -> int:
        """Append feedback events for retrieved context without mutating history."""
        return await self._checkout.record_context_feedback(
            contexts,
            feedback=feedback,
            session_id=session_id,
            actor=actor,
            importance=importance,
            purpose=purpose,
            outcome=outcome,
        )

    async def record_synthesis_candidate(
        self,
        checkout: MemoryCheckout,
        *,
        candidate: dict[str, Any],
        outcome: str,
        actor: str = "zaxy",
        reason: str | None = None,
    ) -> Any:
        """Append an auditable synthesis answer-candidate artifact event."""
        return await self._checkout.record_synthesis_candidate(
            checkout, candidate=candidate, outcome=outcome, actor=actor, reason=reason
        )

    async def record_synthesis_evidence(
        self,
        checkout: MemoryCheckout,
        *,
        row: dict[str, Any],
        outcome: str,
        candidate: dict[str, Any] | None = None,
        actor: str = "zaxy",
        reason: str | None = None,
    ) -> Any:
        """Append auditable feedback for one synthesis evidence ledger row."""
        return await self._checkout.record_synthesis_evidence(
            checkout, row=row, outcome=outcome, candidate=candidate, actor=actor, reason=reason
        )

    async def record_synthesis_artifact(
        self,
        checkout: MemoryCheckout,
        *,
        actor: str = "zaxy",
    ) -> Any:
        """Append a deterministic synthesis artifact created from checkout state."""
        return await self._checkout.record_synthesis_artifact(checkout, actor=actor)

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
        return await self._checkout.after_turn(
            role=role,
            content=content,
            session_id=session_id,
            query=query,
            source=source,
            max_recent_events=max_recent_events,
            limit=limit,
        )

    # -- coordination / fleet / handoff (delegated; decomposition phase 2) ----
    # Bodies live in zaxy.core.fabric_coordination.CoordinationOps.

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
        return await self._coordination.handoff_bundle(
            session_id=session_id,
            query=query,
            replay_from_seq=replay_from_seq,
            limit=limit,
            max_recent_events=max_recent_events,
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
        return await self._coordination.cleanup_subagent(
            parent_session_id=parent_session_id,
            subagent_session_id=subagent_session_id,
            summary=summary,
            query=query,
            limit=limit,
        )

    async def coordinate_start_mission(
        self, mission_id: str, *, objective: str, actor: str = "coordinator"
    ) -> Any:
        """Start a parent coordination mission and project it."""
        return await self._coordination.coordinate_start_mission(
            mission_id, objective=objective, actor=actor
        )

    async def coordinate_create_worker(
        self, mission_id: str, worker_id: str, *, actor: str = "coordinator"
    ) -> Any:
        """Register a worker session under a parent mission and project it."""
        return await self._coordination.coordinate_create_worker(mission_id, worker_id, actor=actor)

    async def coordinate_assign(
        self, mission_id: str, worker_id: str, assignment: str, *, actor: str = "coordinator"
    ) -> Any:
        """Assign scoped work to a coordination worker and project it."""
        return await self._coordination.coordinate_assign(
            mission_id, worker_id, assignment, actor=actor
        )

    async def coordinate_report_finding(
        self,
        mission_id: str,
        worker_id: str,
        *,
        summary: str,
        actor: str,
        evidence: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        claim_key: str | None = None,
        claim_value: str | None = None,
        finding_id: str | None = None,
    ) -> Any:
        """Record a worker-local finding and project it in the worker session."""
        return await self._coordination.coordinate_report_finding(
            mission_id,
            worker_id,
            summary=summary,
            actor=actor,
            evidence=evidence,
            confidence=confidence,
            claim_key=claim_key,
            claim_value=claim_value,
            finding_id=finding_id,
        )

    async def coordinate_review_finding(
        self,
        mission_id: str,
        finding_id: str,
        *,
        status: str,
        actor: str = "coordinator",
        rationale: str | None = None,
    ) -> Any:
        """Record a coordinator review decision and project it."""
        return await self._coordination.coordinate_review_finding(
            mission_id, finding_id, status=status, actor=actor, rationale=rationale
        )

    async def coordinate_promote_finding(
        self, mission_id: str, finding_id: str, *, actor: str = "coordinator", force: bool = False
    ) -> Any:
        """Promote a finding into the parent mission history and project it."""
        return await self._coordination.coordinate_promote_finding(
            mission_id, finding_id, actor=actor, force=force
        )

    async def coordinate_brief(self, mission_id: str) -> Any:
        """Return a replay-backed coordination brief."""
        return await self._coordination.coordinate_brief(mission_id)

    async def coordinate_checkout(self, mission_id: str, *, include_diagnostics: bool = False) -> Any:
        """Return accepted coordination state for prompt injection."""
        return await self._coordination.coordinate_checkout(
            mission_id, include_diagnostics=include_diagnostics
        )

    async def coordinate_record_synthesis_artifact(
        self,
        mission_id: str,
        checkout: MemoryCheckout,
        *,
        decision_scope: str = "brief",
        handoff_id: str | None = None,
        actor: str = "coordinator",
    ) -> dict[str, Any]:
        """Persist a synthesis artifact plus a mission-scoped Coordinate proof packet."""
        return await self._coordination.coordinate_record_synthesis_artifact(
            mission_id,
            checkout,
            decision_scope=decision_scope,
            handoff_id=handoff_id,
            actor=actor,
        )

    async def coordinate_performance_ledger(self, mission_id: str) -> Any:
        """Return replay-backed worker outcome metrics for a coordination mission."""
        return await self._coordination.coordinate_performance_ledger(mission_id)

    async def coordinate_create_handoff(
        self,
        mission_id: str,
        *,
        summary: str,
        actor: str = "coordinator",
        next_steps: list[str] | None = None,
        risks: list[str] | None = None,
    ) -> Any:
        """Create a final parent mission handoff and project it."""
        return await self._coordination.coordinate_create_handoff(
            mission_id, summary=summary, actor=actor, next_steps=next_steps, risks=risks
        )

    async def coordinate_approval_packet(self, mission_id: str) -> Any:
        """Return a portable remote approval packet for pending coordination findings."""
        return await self._coordination.coordinate_approval_packet(mission_id)

    async def coordinate_apply_approval_decisions(
        self,
        mission_id: str,
        decisions: list[dict[str, Any]],
        *,
        actor: str = "coordinator",
    ) -> Any:
        """Apply remote approval decisions and project all resulting events."""
        return await self._coordination.coordinate_apply_approval_decisions(
            mission_id, decisions, actor=actor
        )

    async def coordinate_record_detected_conflicts(
        self, mission_id: str, *, actor: str = "zaxy"
    ) -> Any:
        """Materialize deterministic coordination conflicts and project them."""
        return await self._coordination.coordinate_record_detected_conflicts(mission_id, actor=actor)

    def _coordination_manager(self) -> Any:
        """Return a coordination manager bound to this fabric's session manager."""
        return self._coordination.coordination_manager()

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
        sid = validate_session_id(session_id)
        reinforcement: dict[str, Any] | None = None
        try:
            # Resolve source-event provenance before the validity window closes.
            entities = await self.graph.search_exact(entity_name, entity_type, session_id=sid)
            targets = entity_reinforcement_targets(entities)
            if targets:
                reinforcement = build_invalidated_reinforcement_event(
                    actor="zaxy-memory",
                    session_id=sid,
                    invalidation_id=_invalidation_source_id(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        invalid_at=invalid_at,
                    ),
                    targets=targets,
                )
        except Exception:
            get_metrics().record_degraded_operation("append", "salience_reinforcement_unavailable")
        await self.graph.invalidate_entity(
            entity_name,
            entity_type,
            invalid_at,
            session_id=sid,
        )
        if reinforcement is not None:
            try:
                await self._append_event_spec(reinforcement, session_id=sid)
            except Exception:
                get_metrics().record_degraded_operation("append", "salience_reinforcement_unavailable")

    async def handoff_summary(self, session_id: str = "default") -> dict[str, Any]:
        """Generate a concise handoff summary from the event log.

        Suitable for resuming an agent session across restarts.
        """
        return await self._coordination.handoff_summary(session_id)
