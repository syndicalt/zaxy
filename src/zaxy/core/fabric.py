"""MemoryFabric: the framework-agnostic Python memory API."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from zaxy.causal import (
    CausalQueryResult,
)
from zaxy.codebase import collect_codebase_events
from zaxy.compaction import (
    CompactionProjection,
    load_compaction_projection,
)
from zaxy.config import get_settings
from zaxy.context import Context, ContextAssemblyPolicy
from zaxy.context_refresh import (
    ContextRefreshPlan,
    load_refresh_state,
    plan_context_refresh,
    save_refresh_state,
)
from zaxy.core.checkout_build import (
    _citation_event_identity,
    _compaction_projection_paths,
    _conflicting_property_value,
    _encoding_classification_content,
    _encoding_gate_eligible,
    _encoding_tokens,
    _event_citation,
    _invalidation_source_id,
    _payload_entity_names,
    _payloads_by_seq,
    _token_jaccard,
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
from zaxy.core.models import (
    ContextAssembly,
    ContextRefreshReport,
    HandoffBundle,
    MemoryCheckout,
    QueryPage,
)
from zaxy.documents import collect_document_events
from zaxy.editable import (
    ROLLBACKABLE_EVENT_TYPES,
    build_memory_correction_event,
    build_memory_rollback_event,
)
from zaxy.embedding import build_embedding_provider, embed_extraction
from zaxy.event import (  # noqa: F401 - ReplayResult re-export for existing tests
    EventLog,
    IntegrityReport,
    ReplayResult,
    verify_event_chain,
)
from zaxy.evolution_policy import (
    EvolutionGateDecision,
    build_evolution_gate_event,
    evaluate_evolution_gate,
    resolve_evolution_policy,
)
from zaxy.extract import extract
from zaxy.forgetting import (
    CIPHER_PAYLOAD_KEY,
    build_memory_forgotten_event,
    cipher_cell,
    decrypt_payload,
)
from zaxy.inference import build_inferred_edge_events
from zaxy.log import get_logger
from zaxy.metrics import get_metrics
from zaxy.outcome_learning import (
    build_outcome_event,
    build_rule_event,
    prediction_error,
    preventive_rule_confidence,
    validate_outcome,
)
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
    SALIENCE_BASE,
    EncodingDecision,
    EventRef,
    SalienceLedger,
    build_confirmed_reinforcement_event,
    build_invalidated_reinforcement_event,
    build_reinforcement_event,
    classify_append,
    prediction_error_weight,
    reinforcement_targets_from_citations,
    target_ref,
)
from zaxy.security import (
    validate_event_text,
    validate_payload,
    validate_session_id,
)
from zaxy.session import SessionManager
from zaxy.trace import MemoryTracer
from zaxy.transcripts import collect_transcript_events
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
PRODUCER_REF_PAYLOAD_KEY = "__zaxy_producer_ref"


def _existing_producer_refs(eventlog: EventLog) -> set[str]:
    """Collect producer source refs already recorded in a session's log."""
    refs: set[str] = set()
    for event in eventlog.read_all():
        ref = event.payload.get(PRODUCER_REF_PAYLOAD_KEY)
        if isinstance(ref, str):
            refs.add(ref)
    return refs


def _inferred_edge_candidate_ref(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a candidate reference for a withheld inferred-edge gate decision."""
    ref: dict[str, Any] = {}
    for key in ("target", "source"):
        node = payload.get(key)
        if isinstance(node, dict) and isinstance(node.get("name"), str):
            ref["name"] = node["name"]
            break
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        seq = evidence.get("source_event_seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            ref["seq"] = seq
        event_hash = evidence.get("source_event_hash")
        if isinstance(event_hash, str):
            ref["hash"] = event_hash
    if not ref:
        ref["name"] = "inferred_edge"
    return ref


class ForgetTombstoneUnauditedError(RuntimeError):
    """Verified forgetting destroyed a DEK but failed to append its tombstone.

    Raised by :meth:`MemoryFabric.verified_forget` when the out-of-log key
    erasure has already succeeded (the plaintext is permanently unrecoverable)
    but the cited ``memory.forgotten`` tombstone could not be appended. The log
    is now missing the audit record for an erasure that really happened, so this
    must not be swallowed: callers should treat it as an integrity alert and
    re-append the tombstone. The forget spec is deterministic, so replaying it
    is safe. ``cell_id``, ``target``, and ``forget_id`` carry everything needed
    to reconstruct that tombstone.
    """

    def __init__(self, *, cell_id: str, target: dict[str, Any], forget_id: str) -> None:
        self.cell_id = cell_id
        self.target = target
        self.forget_id = forget_id
        super().__init__(
            f"erased DEK cell_id={cell_id} (seq={target.get('seq')}) but the "
            f"memory.forgotten tombstone (forget_id={forget_id}) failed to append; "
            "memory is erased-but-unaudited and the tombstone must be re-appended"
        )


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
        """
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

        if forgettable and not self.settings.forgetting_enabled:
            raise ValueError(
                "forgettable append requires verified forgetting; set FORGETTING_ENABLED=true"
            )

        encoding = None
        # Forgettable payloads are sealed as ciphertext; the encoding gate (which
        # reads/classifies plaintext content and can project it) is skipped so no
        # plaintext analysis of an erasable memory is denormalized into the graph.
        if (
            not forgettable
            and self._encoding_classification_active()
            and _encoding_gate_eligible(event_type, safe_payload)
        ):
            encoding = await self._classify_append_encoding(safe_payload, session_id=sid)
            if encoding is not None and self._encoding_gate_enabled:
                # Tag only: the event is always appended and hash-chained;
                # the tag rides inside the sealed payload so it is replayable.
                safe_payload = {**safe_payload, "encoding": encoding.tag_payload()}

        # Offload the blocking write to a worker thread: eventlog.append does a
        # synchronous open + exclusive flock + fsync, which would otherwise stall
        # the whole event loop (and every concurrently in-flight MCP request) for
        # the duration of the disk write and any lock wait. The exclusive flock
        # inside append still serializes concurrent writers correctly. Mirrors
        # the to_thread offload already used by query_verbatim/replay.
        event = await asyncio.to_thread(
            eventlog.append,
            event_type,
            actor=actor,
            payload=safe_payload,
            thread=sid,
            forgettable=forgettable,
        )

        interference = None
        if encoding is not None and encoding.classification == "novel":
            # Detected against the pre-append projection state, before this
            # event's own extraction is upserted.
            interference = await self._detect_interference(event, session_id=sid)

        await self._project_event(event, session_id=sid)
        await self._append_generated_inferences(eventlog, source_event=event, session_id=sid)
        self._invalidate_query_page_cache(sid)
        if (
            encoding is not None
            and encoding.classification == "redundant"
            and self._encoding_gate_enabled
        ):
            await self._record_redundant_reinforcement(event, encoding, session_id=sid)
        if interference is not None:
            await self._propose_interference_update(interference, session_id=sid)
        return event

    async def append_batch(
        self,
        items: list[dict[str, Any]],
        *,
        session_id: str | None = None,
    ) -> list[Any]:
        """Ingest a batch of external-producer events under one atomic seal.

        Each item records its producer via ``actor`` and may carry the
        producer's causal links (``parent_event_id``, ``caused_by``, external
        ``id``) plus a ``producer_ref`` used for idempotent dedup. Zaxy always
        computes its own ``seq``/``prev_hash``/``hash`` from the locked tail;
        the causal links round-trip on replay and are hash-sealed when the
        event is written as ``eventloom.v1``. Every appended event is projected
        to the graph so it is immediately retrievable.

        Unlike :meth:`append`, batch ingest skips the agent-turn encoding gate
        and generated-inference appends. Returns only the events appended
        (deduped items are excluded).
        """
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                get_metrics().record_degraded_operation("append", "graph_connect_unavailable")
                self._connected = False

        sid = validate_session_id(session_id or "default")
        eventlog = self.session_manager.get(sid).eventlog

        if not items:
            return []

        # Validate every item up front; on any invalid item reject the whole
        # batch with no writes (atomic).
        validated: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"ingest item {index} must be an object")
            event_type = validate_event_text(item.get("event_type"), "event_type")
            actor = validate_event_text(item.get("actor"), "actor")
            payload = dict(validate_payload(item.get("payload") or {}))
            producer_ref = item.get("producer_ref")
            if producer_ref is not None and not isinstance(producer_ref, str):
                raise ValueError(f"ingest item {index} producer_ref must be a string")
            parent_event_id = item.get("parent_event_id")
            if parent_event_id is not None and not isinstance(parent_event_id, str):
                raise ValueError(f"ingest item {index} parent_event_id must be a string")
            event_id = item.get("id")
            if event_id is not None and not isinstance(event_id, str):
                raise ValueError(f"ingest item {index} id must be a string")
            caused_by = item.get("caused_by")
            if caused_by is not None and (
                not isinstance(caused_by, list) or not all(isinstance(c, str) for c in caused_by)
            ):
                raise ValueError(f"ingest item {index} caused_by must be a list of strings")
            validated.append(
                {
                    "event_type": event_type,
                    "actor": actor,
                    "payload": payload,
                    "producer_ref": producer_ref,
                    "parent_event_id": parent_event_id,
                    "id": event_id,
                    "caused_by": caused_by,
                }
            )

        # Dedup by producer_ref against this session's log and within the batch.
        existing_refs = _existing_producer_refs(eventlog)
        seen: set[str] = set()
        append_items: list[dict[str, Any]] = []
        for item in validated:
            ref = item["producer_ref"]
            if isinstance(ref, str):
                if ref in existing_refs or ref in seen:
                    continue
                seen.add(ref)
                item["payload"][PRODUCER_REF_PAYLOAD_KEY] = ref
            append_items.append(
                {
                    "event_type": item["event_type"],
                    "actor": item["actor"],
                    "payload": item["payload"],
                    "thread": sid,
                    "id": item["id"],
                    "parent_event_id": item["parent_event_id"],
                    "caused_by": item["caused_by"],
                }
            )

        if not append_items:
            return []

        events = eventlog.append_many(append_items)
        for event in events:
            await self._project_event(event, session_id=sid)
        self._invalidate_query_page_cache(sid)
        return events

    async def evaluate_evolution_gate(
        self,
        op: str,
        confidence: float,
        *,
        candidate_ref: dict[str, Any] | None = None,
        actor: str = "zaxy-evolution",
        session_id: str | None = None,
    ) -> EvolutionGateDecision:
        """Evaluate the governed memory-evolution policy for one op and record it.

        Resolves the configured autonomy policy (default ``auto_with_rollback``),
        decides whether ``op`` may auto-apply at ``confidence``, and appends a
        non-authoritative, replayable ``evolution.gate.evaluated`` event so the
        decision itself is auditable. Returns the :class:`EvolutionGateDecision`.
        This is the single gate that I1/I2/I7 evolution producers route through;
        the default auto-applies above threshold (reversible within the rollback
        window) while stricter tiers stay available. See ``ZAXY-3.md`` (I4).
        """
        sid = validate_session_id(session_id or "default")
        policy = resolve_evolution_policy(self.settings)
        decision = evaluate_evolution_gate(op, confidence, policy=policy)
        spec = build_evolution_gate_event(
            actor=actor,
            session_id=sid,
            decision=decision,
            candidate_ref=candidate_ref,
        )
        await self.append(
            spec["event_type"],
            spec["actor"],
            payload=spec["payload"],
            session_id=sid,
        )
        return decision

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
        """Record an outcome on recalled memory and run the governed learning loop.

        Appends a cited ``memory.outcome.recorded`` event; reinforces the cited
        target memory (success → confirmed, failure → invalidated salience); and,
        on failure/partial with a ``lesson``, proposes a **preventive rule** routed
        through the evolution gate (op ``rule_generate``) — auto-applied
        (``memory.rule.generated``) above threshold under the default
        auto_with_rollback tier, otherwise held as ``memory.rule.proposed``. All
        events are non-authoritative, cited, and replayable. See ``ZAXY-3.md`` (I1).

        When ``prior`` (the agent's confidence the recalled memory would
        succeed, in ``[0, 1]``) is supplied, the surprise
        ``pe = |actual - prior|`` is recorded on the outcome event and scales
        the reinforcement ``weight`` (continuous with the fixed multiplier
        table at ``pe == 0.5``); omitting it leaves behavior unchanged.
        """
        sid = validate_session_id(session_id or "default")
        outcome = validate_outcome(outcome)
        target = target_ref(target_seq, target_hash)
        pe = prediction_error(outcome, prior) if prior is not None else None

        outcome_spec = build_outcome_event(
            actor=actor,
            session_id=sid,
            outcome=outcome,
            summary=summary,
            target=target,
            task_id=task_id,
            prior=prior,
            prediction_error=pe,
        )
        outcome_event = await self.append(
            outcome_spec["event_type"], outcome_spec["actor"], payload=outcome_spec["payload"], session_id=sid
        )
        outcome_ref = {"seq": outcome_event.seq, "hash": outcome_event.hash}
        result: dict[str, Any] = {"outcome": outcome, "outcome_event": outcome_ref}

        if target is not None and outcome in ("success", "failure"):
            citation = f"eventloom://{sid}/events/{outcome_event.seq}#{outcome_event.hash}"
            kind = "confirmed" if outcome == "success" else "invalidated"
            weight = prediction_error_weight(kind, pe) if pe is not None else None
            if outcome == "success":
                reinforce_spec = build_confirmed_reinforcement_event(
                    actor=actor,
                    session_id=sid,
                    feedback_id=citation,
                    targets=[target],
                    weight=weight,
                )
            else:
                reinforce_spec = build_invalidated_reinforcement_event(
                    actor=actor,
                    session_id=sid,
                    invalidation_id=citation,
                    targets=[target],
                    weight=weight,
                )
            await self.append(
                reinforce_spec["event_type"], reinforce_spec["actor"], payload=reinforce_spec["payload"], session_id=sid
            )
            result["reinforced"] = "confirmed" if outcome == "success" else "invalidated"

        if outcome in ("failure", "partial") and lesson:
            rule_confidence = preventive_rule_confidence(outcome, confidence)
            decision = await self.evaluate_evolution_gate(
                "rule_generate", rule_confidence, candidate_ref=outcome_ref, actor=actor, session_id=sid
            )
            rule_spec = build_rule_event(
                actor=actor,
                session_id=sid,
                auto_applied=decision.auto_apply,
                rule=lesson,
                trigger=trigger or summary,
                confidence=rule_confidence,
                outcome=outcome,
                source_events=[outcome_ref],
            )
            rule_event = await self.append(
                rule_spec["event_type"], rule_spec["actor"], payload=rule_spec["payload"], session_id=sid
            )
            result["rule"] = {
                "event_type": rule_spec["event_type"],
                "seq": rule_event.seq,
                "auto_applied": decision.auto_apply,
                "review_status": rule_spec["payload"]["review_status"],
            }
        return result

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
        """Re-ingest a human edit as a cited, non-authoritative ``memory.corrected`` event.

        Validates that the target ({``target_seq``, ``target_hash``}) is a sealed
        event in the session log, routes the change through the I4 ``update``
        evolution gate (recording an auditable ``evolution.gate.evaluated`` event),
        then appends a ``memory.corrected`` event that cites the original and
        carries the corrected content + reason. The original event is never
        mutated; the correction is purely additive (the hash chain stays intact)
        and surfaces alongside the retained original on retrieval. See
        ``ZAXY-3.md`` (I5a). Returns the correction event ref, the cited target,
        the deterministic ``correction_id``, and the gate decision.
        """
        sid = validate_session_id(session_id or "default")
        target_event = self._require_target_event(target_seq, target_hash, session_id=sid)
        target = {"seq": target_event.seq, "hash": target_event.hash}

        decision = await self.evaluate_evolution_gate(
            "update", confidence, candidate_ref=target, actor=actor, session_id=sid
        )
        spec = build_memory_correction_event(
            actor=actor,
            session_id=sid,
            target=target,
            new_content=new_content,
            reason=reason,
        )
        event = await self.append(
            spec["event_type"], spec["actor"], payload=spec["payload"], session_id=sid
        )
        return {
            "correction_id": spec["payload"]["correction_id"],
            "correction_event": {"seq": event.seq, "hash": event.hash},
            "target": target,
            "gate": decision.to_payload(candidate_ref=target),
        }

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
        """Reverse a prior evolution with a cited, non-authoritative ``memory.rolled_back`` event.

        Validates that the target is a sealed, *reversible* evolution event (a
        consolidation acceptance, a generated/proposed preventive rule, a gate
        decision, a fleet review, or an earlier correction), routes the reversal
        through the I4 ``update`` gate, and appends a ``memory.rolled_back`` event
        citing the target. On replay/projection the cited evolution is undone --
        e.g. a rolled-back consolidation acceptance reverts the candidate to its
        prior (pre-acceptance) review status -- additively and reversibly, without
        ever mutating the sealed event. See ``ZAXY-3.md`` (I5a). Returns the
        rollback event ref, the cited target, the ``reverts`` descriptor, the
        deterministic ``rollback_id``, and the gate decision.
        """
        sid = validate_session_id(session_id or "default")
        target_event = self._require_target_event(target_seq, target_hash, session_id=sid)
        if target_event.type not in ROLLBACKABLE_EVENT_TYPES:
            valid = ", ".join(sorted(ROLLBACKABLE_EVENT_TYPES))
            raise ValueError(
                f"event {target_event.type!r} is not a reversible evolution; "
                f"rollback supports: {valid}"
            )
        if target_event.type == "consolidation.candidate.reviewed":
            candidate_id = target_event.payload.get("candidate_id")
            if (
                isinstance(candidate_id, str)
                and candidate_id
                and self._has_later_consolidation_review(
                    candidate_id, after_seq=target_event.seq, session_id=sid
                )
            ):
                raise ValueError(
                    "cannot roll back a superseded consolidation review at seq "
                    f"{target_event.seq}; a later review exists for candidate "
                    f"{candidate_id!r} -- only the current (latest) review is reversible"
                )
        target = {"seq": target_event.seq, "hash": target_event.hash}
        reverts = self._reverts_descriptor(target_event, session_id=sid)

        decision = await self.evaluate_evolution_gate(
            "update", confidence, candidate_ref=target, actor=actor, session_id=sid
        )
        spec = build_memory_rollback_event(
            actor=actor,
            session_id=sid,
            target=target,
            reason=reason,
            reverts=reverts,
        )
        event = await self.append(
            spec["event_type"], spec["actor"], payload=spec["payload"], session_id=sid
        )
        return {
            "rollback_id": spec["payload"]["rollback_id"],
            "rollback_event": {"seq": event.seq, "hash": event.hash},
            "target": target,
            "reverts": reverts,
            "gate": decision.to_payload(candidate_ref=target),
        }

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
        """Cryptographically erase a forgettable memory (verified forgetting / I5b).

        Validates that the target is a sealed forgettable memory carrying a
        ``__zaxy_cipher`` cell, routes the erasure through the I4 ``forget`` gate
        (auditable ``evolution.gate.evaluated``), destroys the wrapped DEK in the
        out-of-log erasure vault, and appends a cited, non-authoritative
        ``memory.forgotten`` tombstone. The on-disk ciphertext and its hash are
        untouched -- ``verify()`` stays green -- while the plaintext becomes
        permanently unrecoverable and every reader now sees ``[FORGOTTEN]``. See
        ``ZAXY-3.md`` (I5b). Returns the forget event ref, the cited target, the
        ``cell_id``, whether a live key was destroyed, and the gate decision.
        """
        if not self.settings.forgetting_enabled:
            raise ValueError(
                "verified forgetting is disabled; set FORGETTING_ENABLED=true to enable crypto-erasure"
            )
        sid = validate_session_id(session_id or "default")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        target_event = self._require_target_event(target_seq, target_hash, session_id=sid)
        cell = cipher_cell(target_event.payload)
        if cell is None or not isinstance(cell.get("cell_id"), str):
            raise ValueError(
                f"event at seq {target_seq} is not a forgettable (encrypted) memory; "
                "verified_forget requires a __zaxy_cipher cell"
            )
        cell_id = cell["cell_id"]
        target = {"seq": target_event.seq, "hash": target_event.hash}
        # Gate first (records intent), then destroy the key, then append the
        # tombstone: the security-critical erase precedes the audit record so a
        # tombstone can never claim an erasure that did not happen.
        decision = await self.evaluate_evolution_gate(
            "forget", confidence, candidate_ref=target, actor=actor, session_id=sid
        )
        erased_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        erased = self.session_manager.vault.erase(cell_id, erased_at=erased_at)
        spec = build_memory_forgotten_event(
            actor=actor, session_id=sid, target=target, cell_id=cell_id, reason=reason
        )
        try:
            event = await self.append(
                spec["event_type"], spec["actor"], payload=spec["payload"], session_id=sid
            )
        except Exception as exc:
            # The DEK is already destroyed, so the memory is permanently
            # unrecoverable — but the audit tombstone did not land. Do not let
            # this surface as a routine error: it is an erased-but-unaudited
            # integrity gap that an operator must see and repair by re-appending
            # the (deterministic, replay-safe) tombstone spec.
            get_metrics().record_degraded_operation("forget", "tombstone_append_failed")
            get_logger(__name__).error(
                "verified_forget erased DEK cell_id=%s (seq=%s) but the "
                "memory.forgotten tombstone append failed: %s",
                cell_id,
                target_seq,
                exc,
            )
            raise ForgetTombstoneUnauditedError(
                cell_id=cell_id, target=target, forget_id=spec["payload"]["forget_id"]
            ) from exc
        self._invalidate_query_page_cache(sid)
        return {
            "forget_id": spec["payload"]["forget_id"],
            "forget_event": {"seq": event.seq, "hash": event.hash},
            "target": target,
            "cell_id": cell_id,
            "erased": erased,
            "erased_at": erased_at,
            "gate": decision.to_payload(candidate_ref=target),
        }

    def _decrypt_event_view(self, event: Any) -> Any:
        """Return an event whose forgettable cipher cell is decrypted for reading.

        Plaintext events pass through untouched (no copy). A forgettable event is
        copied with its payload decrypted to plaintext (DEK still live) or the
        ``[FORGOTTEN]`` sentinel (DEK erased). The sealed ``hash`` is preserved so
        citations stay stable. NEVER used by ``verify``/``read_all``.
        """
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict) or CIPHER_PAYLOAD_KEY not in payload:
            return event
        decrypted = decrypt_payload(payload, vault=self.session_manager.vault)
        return event.model_copy(update={"payload": decrypted})

    def _require_target_event(
        self, target_seq: object, target_hash: object, *, session_id: str
    ) -> Any:
        """Return the sealed event identified by ``(seq, hash)`` or raise ValueError."""
        if not isinstance(target_seq, int) or isinstance(target_seq, bool) or target_seq < 1:
            raise ValueError("target_seq must be a positive integer")
        if not isinstance(target_hash, str) or len(target_hash) != 64:
            raise ValueError("target_hash must be a 64-character hex digest")
        eventlog = self.session_manager.get(session_id).eventlog
        for event in eventlog.read_all():
            if event.seq == target_seq:
                if event.hash != target_hash:
                    raise ValueError(
                        f"target_hash does not match the sealed event at seq {target_seq}"
                    )
                return event
        raise ValueError(f"no event at seq {target_seq} in session {session_id!r}")

    def _reverts_descriptor(self, target_event: Any, *, session_id: str) -> dict[str, Any]:
        """Describe what a rollback of ``target_event`` restores, for replay/projection.

        For a consolidation review the descriptor carries the candidate id and the
        prior effective review status (the status before this review, or
        ``pending``) so the projection can revert the candidate. Other reversible
        events only need their type.
        """
        descriptor: dict[str, Any] = {"event_type": target_event.type}
        if target_event.type == "consolidation.candidate.reviewed":
            candidate_id = target_event.payload.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                descriptor["candidate_id"] = candidate_id
                descriptor["to_status"] = self._prior_consolidation_status(
                    candidate_id, before_seq=target_event.seq, session_id=session_id
                )
        return descriptor

    def _prior_consolidation_status(
        self, candidate_id: str, *, before_seq: int, session_id: str
    ) -> str:
        """Return the effective consolidation review status before ``before_seq``."""
        eventlog = self.session_manager.get(session_id).eventlog
        status = "pending"
        for event in eventlog.read_all():
            if event.seq >= before_seq:
                break
            if (
                event.type == "consolidation.candidate.reviewed"
                and event.payload.get("candidate_id") == candidate_id
            ):
                candidate_status = event.payload.get("status")
                if isinstance(candidate_status, str) and candidate_status:
                    status = candidate_status
        return status

    def _has_later_consolidation_review(
        self, candidate_id: str, *, after_seq: int, session_id: str
    ) -> bool:
        """True if a later (higher-seq) review exists for ``candidate_id``.

        A rollback may only target a candidate's current (latest) review: rolling
        back a historically-superseded review would project a stale review status
        onto the graph entity (the projection reverts to the pre-target status,
        ignoring the later surviving review) while the authoritative
        ``consolidation_status`` replay stays correct -- a divergence we reject
        outright instead of projecting.
        """
        eventlog = self.session_manager.get(session_id).eventlog
        for event in eventlog.read_all():
            if (
                event.seq > after_seq
                and event.type == "consolidation.candidate.reviewed"
                and event.payload.get("candidate_id") == candidate_id
            ):
                return True
        return False

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
        """Append and project inferred-edge events generated from cited evidence.

        Autonomous edge *generation* (``inference.edge.generated``) routes through
        the governed evolution gate (op ``update``), which defaults to auto-applying
        so this migration is non-breaking (I4 option A). An operator can set
        ``evolution_op_autonomy=update=propose_only`` (or require_review) to withhold
        autonomous edges; a withheld edge is recorded as an auditable
        ``evolution.gate.evaluated`` event instead of being applied. Deterministic
        safety corrections (retractions, ``causal.edge.generated``) are not gated.
        """
        if source_event.type == "inference.edge.generated":
            return
        policy = resolve_evolution_policy(self.settings)
        for generated in build_inferred_edge_events(source_event):
            if generated["event_type"] == "inference.edge.generated":
                payload = generated["payload"]
                raw_confidence = payload.get("confidence")
                confidence = (
                    float(raw_confidence)
                    if isinstance(raw_confidence, int | float) and not isinstance(raw_confidence, bool)
                    else 0.0
                )
                decision = evaluate_evolution_gate("update", confidence, policy=policy)
                if not decision.auto_apply:
                    gate_spec = build_evolution_gate_event(
                        actor="zaxy-inference",
                        session_id=session_id,
                        decision=decision,
                        candidate_ref=_inferred_edge_candidate_ref(payload),
                    )
                    gate_event = eventlog.append(
                        gate_spec["event_type"],
                        actor=gate_spec["actor"],
                        payload=validate_payload(gate_spec["payload"]),
                        thread=session_id,
                    )
                    await self._project_event(gate_event, session_id=session_id)
                    continue
            event = eventlog.append(
                generated["event_type"],
                actor=generated["actor"],
                payload=validate_payload(generated["payload"]),
                thread=session_id,
            )
            await self._project_event(event, session_id=session_id)

    def _encoding_classification_active(self) -> bool:
        """Return whether append-time encoding classification should run.

        The write-time gate tags payloads only when ``ENCODING_GATE_ENABLED``;
        interference detection additionally runs under the cognitive
        retrieval profile (classification is its novelty signal). With both
        off, appends are byte-identical to the pre-gate contract.
        """
        return self._encoding_gate_enabled or self.retrieval_profile.salience_ranking

    async def _classify_append_encoding(
        self,
        payload: dict[str, Any],
        *,
        session_id: str,
    ) -> EncodingDecision | None:
        """Classify one append against pre-append memory state, best-effort.

        Signals (no embedding calls): token Jaccard between the payload's
        canonical text and the closest existing verbatim-index chunk, plus
        the fraction of payload-declared entity names already projected.
        Returns ``None`` when signals cannot be computed; a failure here
        never fails the append itself.
        """
        try:
            content = _encoding_classification_content(payload)
            if not content:
                return None
            content_tokens = _encoding_tokens(content)
            if not content_tokens:
                return None
            best_overlap = 0.0
            duplicate_of: str | None = None
            index = self._verbatim_index(session_id)
            payloads_by_seq: dict[int, dict[str, Any]] | None = None
            for hit in index.query(content[:2000], limit=5):
                # Compare against the source payload's canonical content when
                # resolvable so earlier gate/cue metadata never dilutes the
                # duplicate signal; fall back to the raw chunk text.
                hit_tokens: set[str] | None = None
                hit_seq, _hit_hash = _citation_event_identity(hit.citation)
                if hit_seq is not None:
                    if payloads_by_seq is None:
                        payloads_by_seq = _payloads_by_seq(
                            self.session_manager.get(session_id).eventlog.read_all()
                        )
                    hit_payload = payloads_by_seq.get(hit_seq)
                    if isinstance(hit_payload, dict):
                        hit_tokens = _encoding_tokens(
                            _encoding_classification_content(hit_payload)
                        )
                if hit_tokens is None:
                    hit_tokens = _encoding_tokens(hit.content)
                overlap = _token_jaccard(content_tokens, hit_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    duplicate_of = hit.citation
            entity_overlap = await self._encoding_entity_overlap(payload, session_id=session_id)
            classification = classify_append(
                content_overlap=best_overlap,
                entity_overlap=entity_overlap,
            )
            return EncodingDecision(
                classification=classification,
                content_overlap=best_overlap,
                entity_overlap=entity_overlap,
                duplicate_of=duplicate_of if classification == "redundant" else None,
            )
        except Exception:
            get_metrics().record_degraded_operation("append", "encoding_classification_unavailable")
            return None

    async def _encoding_entity_overlap(
        self,
        payload: dict[str, Any],
        *,
        session_id: str,
    ) -> float:
        """Return the fraction of payload-declared entity names already projected."""
        names = _payload_entity_names(payload)
        if not names:
            return 0.0
        matched = 0
        for name in names:
            try:
                hits = await self.graph.search_exact(name, session_id=session_id)
            except Exception:
                continue
            if isinstance(hits, list) and hits:
                matched += 1
        return matched / len(names)

    async def _detect_interference(self, event: Any, *, session_id: str) -> dict[str, Any] | None:
        """Detect a contradiction between a novel append and projected memory.

        Contradiction is defined honestly from available write-time signals:
        the new event's extraction names an already-active entity (same name
        and entity type) whose projected state carries a different value for
        the same scalar property key (summaries and bookkeeping/provenance
        keys are excluded — free text changing is not a value conflict).
        Runs against the pre-append projection and only flags memories whose
        replayed salience is at or above the attenuation floor. Best-effort:
        a failure never fails the append.
        """
        try:
            extraction = extract(event)
            for entity in extraction.entities:
                properties = entity.properties
                if not properties or entity.entity_type == "event":
                    continue
                try:
                    existing = await self.graph.search_exact(
                        entity.name,
                        entity.entity_type,
                        session_id=session_id,
                    )
                except Exception:
                    continue
                if not isinstance(existing, list):
                    continue
                for old in existing:
                    if getattr(old, "valid_to", None) is not None:
                        continue
                    old_properties = getattr(old, "properties", None)
                    if not isinstance(old_properties, dict):
                        continue
                    conflict = _conflicting_property_value(old_properties, properties)
                    if conflict is None:
                        continue
                    contradicted = target_ref(
                        old_properties.get("source_event_seq"),
                        old_properties.get("source_event_hash"),
                    )
                    if contradicted is None or contradicted["seq"] == event.seq:
                        continue
                    if not self._memory_above_floor(contradicted, session_id=session_id):
                        continue
                    key, old_value, new_value = conflict
                    claim = (
                        f"{entity.name} {key} is now {new_value} (previously {old_value})"
                    )[:400]
                    return {
                        "claim": claim,
                        "rationale": (
                            "Write-time interference: a novel append contradicts an "
                            f"above-floor memory on {entity.entity_type} "
                            f"'{entity.name}' property '{key}'."
                        ),
                        "source_events": [
                            contradicted,
                            {"seq": event.seq, "hash": event.hash},
                        ],
                    }
        except Exception:
            get_metrics().record_degraded_operation("append", "interference_detection_unavailable")
        return None

    def _memory_above_floor(self, target: dict[str, Any], *, session_id: str) -> bool:
        """Return whether a memory's replayed salience clears the attenuation floor.

        Memories with no reinforcement history carry the implicit base
        salience (1.0) and are always above the default floor.
        """
        events = self.session_manager.get(session_id).eventlog.read_all()
        ledger = SalienceLedger(half_life_days=self._salience_half_life_days)
        states = ledger.replay(events, now=datetime.now(UTC))
        state = states.get(EventRef(seq=int(target["seq"]), hash=str(target["hash"])))
        score = state.score if state is not None else SALIENCE_BASE
        return score >= self._salience_floor

    async def _propose_interference_update(
        self,
        finding: dict[str, Any],
        *,
        session_id: str,
    ) -> None:
        """Emit one review-gated belief-update proposal for a detected conflict.

        Routes through the existing :meth:`propose_belief_update` path so the
        proposal is review-pending, non-authoritative, and cites both the
        contradicted and the contradicting event. Best-effort: a proposal
        failure never fails the append that triggered it.
        """
        try:
            await self.propose_belief_update(
                finding["claim"],
                rationale=finding["rationale"],
                confidence=0.5,
                source_events=finding["source_events"],
                phase="reflection",
                session_id=session_id,
                actor="zaxy-memory",
            )
        except Exception:
            get_metrics().record_degraded_operation("append", "interference_proposal_unavailable")

    async def _record_redundant_reinforcement(
        self,
        event: Any,
        encoding: EncodingDecision,
        *,
        session_id: str,
    ) -> None:
        """Project a redundant append as weak reinforcement of the duplicate.

        The honest minimal mechanism: the duplicate event is still appended,
        hash-chained, and projected (its extraction upserts into the same
        projected entities, so it never creates a new ranked entry), and the
        gate additionally appends one 'surfaced'-strength reinforcement
        toward the duplicated memory so repetition raises that memory's
        salience instead of minting new ranked content. Best-effort: a
        failure never fails the append.
        """
        try:
            if encoding.duplicate_of is None:
                return
            index = self._session_event_ref_index(session_id)
            targets = reinforcement_targets_from_citations(
                [encoding.duplicate_of],
                event_index=index,
            )
            if not targets:
                return
            citation = _event_citation(event) or f"{session_id}:append"
            spec = build_reinforcement_event(
                actor="zaxy-memory",
                session_id=session_id,
                kind="surfaced",
                targets=targets,
                source={"encoding_gate": citation},
            )
            await self._append_event_spec(spec, session_id=session_id)
        except Exception:
            get_metrics().record_degraded_operation("append", "salience_reinforcement_unavailable")

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
        self, mission_id: str, finding_id: str, *, actor: str = "coordinator"
    ) -> Any:
        """Promote a finding into the parent mission history and project it."""
        return await self._coordination.coordinate_promote_finding(
            mission_id, finding_id, actor=actor
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
