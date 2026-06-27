"""MemoryFabric: the framework-agnostic Python memory API."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from zaxy.causal import (
    CausalQueryResult,
    causal_query_result_from_projection,
    causal_relation_to_graph_relation,
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
from zaxy.core.checkout_build import (
    _apply_purpose_outcome_learning,
    _bounded_threshold,
    _causal_result_reasoning_evidence,
    _checkout_reasoning_evidence,
    _checkout_recall_limit,
    _checkout_source_id,
    _citation_event_identity,
    _claim_key,
    _compaction_projection_paths,
    _conflicting_property_value,
    _consolidation_candidate_ids,
    _context_citation,
    _context_feedback_metadata,
    _context_identity,
    _context_warnings,
    _contexts_as_of_seq,
    _encoding_classification_content,
    _encoding_gate_eligible,
    _encoding_tokens,
    _event_citation,
    _event_content,
    _feedback_outcome,
    _feedback_purpose_payload,
    _increment_count,
    _invalidation_source_id,
    _metacognition_payloads_reasoning_evidence,
    _normalize_context_feedback,
    _packet_memory_reinforcements,
    _payload_entity_names,
    _payloads_by_seq,
    _prefer_verbatim_for_duplicate_source_groups,
    _procedure_reasoning_evidence,
    _purpose_outcome_aggregates,
    _reverification_needs_from_events,
    _score_claim_evidence,
    _source_context_text,
    _source_events_from_reasoning_evidence,
    _source_events_reasoning_evidence,
    _strict_reasoning_evidence,
    _synthesis_packet_metadata,
    _token_jaccard,
    _tokens,
    build_memory_checkout,
    entity_reinforcement_targets,
)
from zaxy.core.models import (
    ContextAssembly,
    ContextRefreshReport,
    HandoffBundle,
    MemoryCheckout,
    QueryPage,
)
from zaxy.documents import collect_document_events
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
from zaxy.inference import build_inferred_edge_events
from zaxy.lifecycle import build_subagent_completed_event
from zaxy.metacognition import (
    build_confidence_assessment_event,
    build_conflict_cluster_event,
    build_known_unknown_event,
    build_reverify_request_event,
    summarize_metacognition_events,
)
from zaxy.metrics import get_metrics
from zaxy.pagination import encode_query_cursor, validate_query_cursor
from zaxy.procedural_planning import classify_procedure_contexts
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store
from zaxy.purpose import PurposeProfile, purpose_profile, purpose_retrieval_policy
from zaxy.query import QueryRouter, ScoringProfile, build_reranker, build_retention_policy
from zaxy.reasoning_primitives import (
    ReasoningPrimitiveCall,
    build_belief_update_proposal_event,
    phase_purpose_profile,
    validate_reasoning_phase,
)
from zaxy.recall import build_recall_candidate_set
from zaxy.refs import MemoryRef, MemoryRefStore
from zaxy.retrieval_cache import SessionRetrievalCache, _eventlog_file_signature
from zaxy.retrieval_intent import classify_retrieval_intent
from zaxy.retrieval_plan import (
    absence_check_bundle,
    bridge_source_lane_queries,
    should_try_absence_bundle_first,
    source_context_group,
    source_lane_queries,
    source_synthesis_bundle_result,
)
from zaxy.retrieval_profile import (
    RetrievalProfile,
    apply_retrieval_profile,
    resolve_retrieval_profile,
)
from zaxy.salience import (
    CUE_MATCH_WEIGHT,
    REINFORCEMENT_EVENT_TYPE,
    SALIENCE_BASE,
    EncodingDecision,
    EventRef,
    SalienceLedger,
    build_confirmed_reinforcement_event,
    build_invalidated_reinforcement_event,
    build_reinforcement_event,
    build_surfaced_reinforcement_event,
    classify_append,
    cue_overlap,
    cue_pairs,
    event_ref_index,
    reinforcement_targets_from_citations,
    target_ref,
)
from zaxy.security import (
    MAX_QUERY_LIMIT,
    validate_event_text,
    validate_limit,
    validate_payload,
    validate_query,
    validate_session_id,
    validate_traversal_depth,
)
from zaxy.session import SessionManager
from zaxy.synthesis_artifact import (
    build_synthesis_artifact,
    build_synthesis_candidate_event_payload,
    build_synthesis_evidence_event_payload,
    normalize_synthesis_outcome,
    synthesis_outcome_event_type,
)
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

QUERY_PAGE_CACHE_TTL_SECONDS = 30.0
QUERY_PAGE_CACHE_MAX_ENTRIES = 32

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

        When components are injected (``owns_connections=False``) the shared
        graph/tracer/retrieval-cache are left untouched — the host owns them.
        """
        if self._owns_connections:
            await self.graph.close()
            await self.tracer.close()
            self._retrieval_cache.invalidate()
        self._event_ref_index_cache = {}
        self._session_cue_index_cache = {}
        self._query_page_cache = {}
        self._warmed_projection_sessions = set()
        self._connected = False

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
        return await self._query_causal_neighbors(
            entity_name,
            direction="successors",
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
        return await self._query_causal_neighbors(
            entity_name,
            direction="predecessors",
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
        safe_outcome = validate_query(outcome)
        safe_phase = validate_reasoning_phase(phase)
        sid = validate_session_id(session_id)
        safe_depth = validate_traversal_depth(depth)
        profile = phase_purpose_profile(safe_phase)
        evidence: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        try:
            causal_results = await self.query_causal_predecessors(
                safe_outcome,
                depth=safe_depth,
                session_id=sid,
            )
            for result in causal_results:
                item = result.to_dict()
                results.append(item)
                evidence.append(_causal_result_reasoning_evidence(item))
            fallback_used = False
            if not results:
                checkout = await self.checkout_memory(
                    safe_outcome,
                    session_id=sid,
                    limit=max(1, min(MAX_QUERY_LIMIT, safe_depth * 2)),
                    purpose=profile,
                )
                for item in checkout.evidence:
                    evidence_item = _checkout_reasoning_evidence(item)
                    if evidence_item is not None:
                        evidence.append(evidence_item)
                        results.append(
                            {
                                "source": "checkout",
                                "content": evidence_item.get("content", ""),
                                "citation": evidence_item["citation"],
                            }
                        )
                fallback_used = True
            await self._append_reasoning_primitive_call(
                primitive="explain_outcome",
                phase=safe_phase,
                session_id=sid,
                query=safe_outcome,
                result_count=len(results),
                evidence=evidence,
                status="succeeded",
            )
            return {
                "primitive": "explain_outcome",
                "phase": safe_phase,
                "session_id": sid,
                "outcome": safe_outcome,
                "depth": safe_depth,
                "fallback_used": fallback_used,
                "result_count": len(results),
                "results": results,
                "evidence": evidence,
            }
        except Exception:
            await self._append_reasoning_primitive_call(
                primitive="explain_outcome",
                phase=safe_phase,
                session_id=sid,
                query=safe_outcome,
                result_count=0,
                evidence=[],
                status="failed",
            )
            raise

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
        sid = validate_session_id(session_id)
        safe_phase = validate_reasoning_phase(phase)
        event = build_belief_update_proposal_event(
            actor=actor,
            session_id=sid,
            claim=validate_query(claim),
            rationale=validate_query(rationale),
            confidence=confidence,
            source_events=source_events,
            phase=safe_phase,
        )
        evidence = [
            {
                "citation": f"eventloom://{sid}/events/{source['seq']}#{source['hash'][:12]}",
                "source_event_seq": source["seq"],
                "source_event_hash": source["hash"],
            }
            for source in event["payload"]["source_events"]
        ]
        try:
            await self.append(
                event["event_type"],
                actor=event["actor"],
                payload=event["payload"],
                session_id=sid,
            )
            await self._append_reasoning_primitive_call(
                primitive="propose_belief_update",
                phase=safe_phase,
                session_id=sid,
                query=str(event["payload"]["claim"]),
                result_count=1,
                evidence=evidence,
                status="succeeded",
                actor="zaxy-reasoning",
            )
            return event
        except Exception:
            await self._append_reasoning_primitive_call(
                primitive="propose_belief_update",
                phase=safe_phase,
                session_id=sid,
                query=str(event["payload"]["claim"]),
                result_count=0,
                evidence=evidence,
                status="failed",
                actor="zaxy-reasoning",
            )
            raise

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
        safe_claim = validate_query(claim)
        safe_phase = validate_reasoning_phase(phase)
        sid = validate_session_id(session_id)
        safe_limit = validate_limit(limit)
        safe_min_confidence = _bounded_threshold(min_confidence)
        profile = phase_purpose_profile(safe_phase)
        evidence: list[dict[str, Any]] = []
        try:
            checkout = await self.checkout_memory(
                safe_claim,
                session_id=sid,
                limit=safe_limit,
                purpose=profile,
            )
            scored = _score_claim_evidence(safe_claim, checkout.evidence, limit=safe_limit)
            evidence = scored["evidence"]
            if record_assessment:
                await self._append_metacognition_for_claim_confidence(
                    claim=safe_claim,
                    session_id=sid,
                    phase=safe_phase,
                    scored=scored,
                    min_confidence=safe_min_confidence,
                )
            await self._append_reasoning_primitive_call(
                primitive="get_claim_confidence",
                phase=safe_phase,
                session_id=sid,
                query=safe_claim,
                result_count=len(evidence),
                evidence=evidence,
                status="succeeded",
            )
            return {
                "primitive": "get_claim_confidence",
                "phase": safe_phase,
                "session_id": sid,
                "claim": safe_claim,
                "min_confidence": safe_min_confidence,
                **scored,
            }
        except Exception:
            await self._append_reasoning_primitive_call(
                primitive="get_claim_confidence",
                phase=safe_phase,
                session_id=sid,
                query=safe_claim,
                result_count=0,
                evidence=[],
                status="failed",
            )
            raise

    async def retrieve_similar_procedures(
        self,
        query: str,
        *,
        phase: str = "planning",
        session_id: str = "default",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Retrieve cited Skill Memory or consolidation procedure candidates."""
        safe_query = validate_query(query)
        safe_phase = validate_reasoning_phase(phase)
        sid = validate_session_id(session_id)
        safe_limit = validate_limit(limit)
        profile = phase_purpose_profile(safe_phase)
        evidence: list[dict[str, Any]] = []
        try:
            contexts = await self.query(
                safe_query,
                session_id=sid,
                limit=min(MAX_QUERY_LIMIT, max(safe_limit * 2, safe_limit)),
                include_source_lane=True,
                scoring_profile=purpose_retrieval_policy(
                    profile,
                    safe_query,
                    prompt_limit=safe_limit,
                    base_recall_limit=safe_limit,
                ).scoring_profile,
            )
            classified = classify_procedure_contexts(contexts, limit=safe_limit)
            procedures = cast(list[dict[str, Any]], classified["applicable"])
            evidence = [
                item
                for procedure in procedures
                if (item := _procedure_reasoning_evidence(procedure)) is not None
            ]
            await self._append_reasoning_primitive_call(
                primitive="retrieve_similar_procedures",
                phase=safe_phase,
                session_id=sid,
                query=safe_query,
                result_count=len(procedures),
                evidence=evidence,
                status="succeeded",
            )
            return {
                "primitive": "retrieve_similar_procedures",
                "phase": safe_phase,
                "session_id": sid,
                "query": safe_query,
                "procedure_count": len(procedures),
                "procedures": procedures,
                "applicable": procedures,
                "diagnostic": classified["diagnostic"],
                "excluded": classified["excluded"],
                "procedural_memory": classified["procedural_memory"],
                "evidence": evidence,
            }
        except Exception:
            await self._append_reasoning_primitive_call(
                primitive="retrieve_similar_procedures",
                phase=safe_phase,
                session_id=sid,
                query=safe_query,
                result_count=0,
                evidence=[],
                status="failed",
            )
            raise

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
        safe_question = validate_query(question)
        safe_phase = validate_reasoning_phase(phase)
        sid = validate_session_id(session_id)
        event = build_known_unknown_event(
            actor=actor,
            session_id=sid,
            question=safe_question,
            reason=validate_query(reason),
            source_events=source_events,
            claim_key=validate_query(claim_key),
            gap_type=validate_query(gap_type),
            reverify_query=reverify_query,
        )
        await self._append_event_spec(event, session_id=sid)
        evidence = _source_events_reasoning_evidence(sid, event["payload"]["source_events"])
        await self._append_reasoning_primitive_call(
            primitive="record_known_unknown",
            phase=safe_phase,
            session_id=sid,
            query=safe_question,
            result_count=1,
            evidence=evidence,
            status="succeeded",
        )
        return event

    async def list_known_unknowns(
        self,
        *,
        session_id: str = "default",
        status: str = "open",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return replay-derived known unknowns for a session."""
        sid = validate_session_id(session_id)
        safe_limit = validate_limit(limit)
        normalized_status = status.strip().casefold() if isinstance(status, str) else "open"
        events = self._metacognition_event_specs(sid)
        unknowns = [
            dict(event["payload"])
            for event in events
            if event["event_type"] == "metacognition.unknown.recorded"
            and (normalized_status == "all" or str(event["payload"].get("status") or "") == normalized_status)
        ][:safe_limit]
        result = {
            "primitive": "known_unknowns",
            "session_id": sid,
            "status": normalized_status,
            "unknown_count": len(unknowns),
            "unknowns": unknowns,
            "summary": summarize_metacognition_events(events),
        }
        await self._append_reasoning_primitive_call(
            primitive="list_known_unknowns",
            phase="review",
            session_id=sid,
            query=f"known_unknowns:{normalized_status}",
            result_count=len(unknowns),
            evidence=_metacognition_payloads_reasoning_evidence(sid, unknowns),
            status="succeeded",
        )
        return result

    async def list_conflict_clusters(
        self,
        *,
        session_id: str = "default",
        unresolved_only: bool = True,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return replay-derived metacognitive conflict clusters."""
        sid = validate_session_id(session_id)
        safe_limit = validate_limit(limit)
        events = self._metacognition_event_specs(sid)
        clusters = [
            dict(event["payload"])
            for event in events
            if event["event_type"] == "metacognition.conflict.clustered"
            and (
                not unresolved_only
                or event["payload"].get("resolution_status") == "unresolved"
            )
        ][:safe_limit]
        result = {
            "primitive": "conflict_clusters",
            "session_id": sid,
            "unresolved_only": bool(unresolved_only),
            "cluster_count": len(clusters),
            "clusters": clusters,
            "summary": summarize_metacognition_events(events),
        }
        await self._append_reasoning_primitive_call(
            primitive="list_conflict_clusters",
            phase="review",
            session_id=sid,
            query="unresolved_conflict_clusters" if unresolved_only else "all_conflict_clusters",
            result_count=len(clusters),
            evidence=_metacognition_payloads_reasoning_evidence(sid, clusters),
            status="succeeded",
        )
        return result

    async def list_confidence_trajectory(
        self,
        claim: str,
        *,
        session_id: str = "default",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return append-only confidence trajectory points for a claim."""
        safe_claim = validate_query(claim)
        sid = validate_session_id(session_id)
        safe_limit = validate_limit(limit)
        target = safe_claim.casefold()
        events = self._metacognition_event_specs(sid)
        trajectory = [
            dict(event["payload"])
            for event in events
            if event["event_type"] == "metacognition.confidence.assessed"
            and (
                str(event["payload"].get("claim") or "").casefold() == target
                or str(event["payload"].get("claim_key") or "").casefold() == target
            )
        ][-safe_limit:]
        result = {
            "primitive": "confidence_trajectory",
            "session_id": sid,
            "claim": safe_claim,
            "trajectory_count": len(trajectory),
            "trajectory": trajectory,
        }
        await self._append_reasoning_primitive_call(
            primitive="list_confidence_trajectory",
            phase="review",
            session_id=sid,
            query=safe_claim,
            result_count=len(trajectory),
            evidence=_metacognition_payloads_reasoning_evidence(sid, trajectory),
            status="succeeded",
        )
        return result

    async def list_reverification_needs(
        self,
        query: str | None = None,
        *,
        session_id: str = "default",
        limit: int = 10,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Return replay-derived claims and unknowns that need re-verification."""
        sid = validate_session_id(session_id)
        safe_limit = validate_limit(limit)
        safe_min_confidence = _bounded_threshold(min_confidence)
        query_text = validate_query(query) if query else None
        events = self._metacognition_event_specs(sid)
        needs = _reverification_needs_from_events(
            events,
            query=query_text,
            limit=safe_limit,
            min_confidence=safe_min_confidence,
        )
        result = {
            "primitive": "reverification_needs",
            "session_id": sid,
            "query": query_text,
            "min_confidence": safe_min_confidence,
            "need_count": len(needs),
            "needs": needs,
            "summary": summarize_metacognition_events(events),
        }
        await self._append_reasoning_primitive_call(
            primitive="list_reverification_needs",
            phase="review",
            session_id=sid,
            query=query_text or "reverification_needs",
            result_count=len(needs),
            evidence=_metacognition_payloads_reasoning_evidence(sid, needs),
            status="succeeded",
        )
        return result

    async def plan_from_procedures(
        self,
        goal: str,
        *,
        phase: str = "planning",
        session_id: str = "default",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Return a non-authoritative planning packet from applicable procedures."""
        result = await self.retrieve_similar_procedures(
            goal,
            phase=phase,
            session_id=session_id,
            limit=limit,
        )
        steps: list[str] = []
        for procedure in result.get("applicable", []):
            for step in procedure.get("procedure", []):
                if isinstance(step, str) and step not in steps:
                    steps.append(step)
        packet = {
            "primitive": "plan_from_procedures",
            "phase": result["phase"],
            "session_id": result["session_id"],
            "goal": result["query"],
            "steps": steps[: validate_limit(limit)],
            "source_procedures": result.get("applicable", []),
            "procedural_memory": result.get("procedural_memory", {}),
            "authority_status": "non_authoritative",
        }
        await self._append_reasoning_primitive_call(
            primitive="plan_from_procedures",
            phase=str(result["phase"]),
            session_id=str(result["session_id"]),
            query=str(result["query"]),
            result_count=len(steps),
            evidence=list(result.get("evidence") or []),
            status="succeeded",
        )
        return packet

    async def _append_metacognition_for_claim_confidence(
        self,
        *,
        claim: str,
        session_id: str,
        phase: str,
        scored: dict[str, Any],
        min_confidence: float,
    ) -> None:
        evidence = list(scored.get("evidence") or [])
        confidence = float(scored.get("confidence") or 0.0)
        support_count = int(scored.get("support_count") or 0)
        conflict_count = int(scored.get("conflict_count") or 0)
        assessment = build_confidence_assessment_event(
            actor="zaxy-reasoning",
            session_id=session_id,
            claim=claim,
            confidence=confidence,
            support_count=support_count,
            conflict_count=conflict_count,
            evidence=evidence,
            method="deterministic_token_overlap_v1",
            requires_reverify=confidence < min_confidence or conflict_count > 0,
            claim_key=_claim_key(claim),
        )
        assessment_event = await self._append_event_spec(assessment, session_id=session_id)
        source_events = _source_events_from_reasoning_evidence(evidence)
        if not source_events and confidence < min_confidence:
            source_events = [{"seq": assessment_event.seq, "hash": assessment_event.hash}]
        if support_count > 0 and conflict_count > 0:
            supports = _source_events_from_reasoning_evidence(
                [item for item in evidence if item.get("stance") == "support"]
            )
            conflicts = _source_events_from_reasoning_evidence(
                [item for item in evidence if item.get("stance") == "conflict"]
            )
            if supports and conflicts:
                cluster = build_conflict_cluster_event(
                    actor="zaxy-reasoning",
                    session_id=session_id,
                    claim_key=_claim_key(claim),
                    claim=claim,
                    supporting_source_events=supports,
                    conflicting_source_events=conflicts,
                    confidence=confidence,
                    reason="Support and conflict evidence both present.",
                )
                await self._append_event_spec(cluster, session_id=session_id)
        if confidence < min_confidence or conflict_count > 0:
            reverify = build_reverify_request_event(
                actor="zaxy-reasoning",
                session_id=session_id,
                query=claim,
                reason="Low confidence or conflicting cited evidence requires re-verification.",
                source_events=source_events,
                priority="high" if conflict_count > 0 else "normal",
                claim_key=_claim_key(claim),
            )
            await self._append_event_spec(reverify, session_id=session_id)

    async def _append_event_spec(self, event: dict[str, Any], *, session_id: str) -> Any:
        return await self.append(
            str(event["event_type"]),
            actor=str(event["actor"]),
            payload=cast(dict[str, Any], event["payload"]),
            session_id=session_id,
        )

    def _metacognition_event_specs(self, session_id: str) -> list[dict[str, Any]]:
        replayed = self.session_manager.get(session_id).eventlog.read_all()
        events: list[dict[str, Any]] = []
        for event in replayed:
            if not str(event.type).startswith("metacognition."):
                continue
            events.append(
                {
                    "event_type": event.type,
                    "actor": event.actor,
                    "thread": event.thread,
                    "payload": dict(event.payload),
                    "seq": event.seq,
                    "hash": event.hash,
                    "timestamp": event.timestamp,
                }
            )
        return events

    async def _append_reasoning_primitive_call(
        self,
        *,
        primitive: str,
        phase: str,
        session_id: str,
        query: str,
        result_count: int,
        evidence: list[dict[str, Any]],
        status: str,
        actor: str = "zaxy-reasoning",
    ) -> None:
        call = ReasoningPrimitiveCall(
            primitive=primitive,
            phase=phase,
            session_id=session_id,
            query=query,
            result_count=result_count,
            evidence=_strict_reasoning_evidence(evidence),
            status=status,
        )
        event = call.to_event(actor=actor)
        await self.append(
            event["event_type"],
            actor=event["actor"],
            payload=event["payload"],
            session_id=session_id,
        )

    async def _query_causal_neighbors(
        self,
        entity_name: str,
        *,
        direction: Literal["successors", "predecessors"],
        relation_type: str | None,
        depth: int,
        temporal_point: str | None,
        session_id: str,
    ) -> list[CausalQueryResult]:
        safe_entity_name = validate_query(entity_name)
        safe_depth = validate_traversal_depth(depth)
        safe_session_id = validate_session_id(session_id)
        graph_relation_type = (
            causal_relation_to_graph_relation(relation_type) if relation_type is not None else None
        )
        neighbors = await self.graph.search_causal_neighbors(
            safe_entity_name,
            direction=direction,
            relation_type=graph_relation_type,
            depth=safe_depth,
            temporal_point=temporal_point,
            session_id=safe_session_id,
        )
        results: list[CausalQueryResult] = []
        for entity in neighbors:
            result = causal_query_result_from_projection(entity, direction=direction)
            if result is not None:
                results.append(result)
        return results

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

        encoding = None
        if self._encoding_classification_active() and _encoding_gate_eligible(event_type, safe_payload):
            encoding = await self._classify_append_encoding(safe_payload, session_id=sid)
            if encoding is not None and self._encoding_gate_enabled:
                # Tag only: the event is always appended and hash-chained;
                # the tag rides inside the sealed payload so it is replayable.
                safe_payload = {**safe_payload, "encoding": encoding.tag_payload()}

        event = eventlog.append(
            event_type,
            actor=actor,
            payload=safe_payload,
            thread=sid,
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
        contexts = await self.retrieve(
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
            contexts = self.context_assembly_policy.assemble(
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
            get_metrics().record_query(duration_ms / 1000.0)
        else:
            get_metrics().record_query(duration_ms / 1000.0, source=query_source)

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
                if trace:
                    await self._trace_query_best_effort(query, len(chunks), duration_ms, temporal_point)
                if trace:
                    get_metrics().record_query(duration_ms / 1000.0, source="eventloom")
                return chunks
        await self._warm_projection_session(sid)

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
                scoring_profile=scoring_profile,
            )
        except Exception:
            get_metrics().record_degraded_operation("query", "graph_retrieval_unavailable")
            contexts = self._merge_projection_contexts(
                self._query_eventlog_fallback(query, sid, limit, reason="graph retrieval unavailable"),
                query,
                limit,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            if trace:
                await self._trace_query_best_effort(query, len(contexts), duration_ms, temporal_point)
            if trace:
                get_metrics().record_query(duration_ms / 1000.0, source="eventloom")
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
            get_metrics().record_query(duration_ms / 1000.0)

        return contexts

    async def _query_source_lane(
        self,
        query: str,
        graph_contexts: list[Context],
        session_id: str,
        limit: int,
    ) -> list[Context]:
        """Return bounded verbatim source evidence for raw query results."""
        candidate_limit = self.context_assembly_policy.verbatim_candidate_limit(
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
                hits = await self.query_verbatim(source_query, limit=candidate_limit, session_id=session_id)
            except Exception:
                get_metrics().record_degraded_operation("query", "source_lane_unavailable")
                continue
            self._extend_unique_source_contexts(source_contexts, hits, seen)
            for bridge_query in bridge_source_lane_queries(query, [context.content for context in hits]):
                try:
                    bridge_hits = await self.query_verbatim(
                        bridge_query,
                        limit=candidate_limit,
                        session_id=session_id,
                    )
                except Exception:
                    get_metrics().record_degraded_operation("query", "source_lane_unavailable")
                    continue
                self._extend_unique_source_contexts(source_contexts, bridge_hits, seen)
        for source_group in self._graph_source_groups_for_backfill(graph_contexts, limit):
            try:
                hits = await self.query_verbatim(source_group, limit=max(1, min(candidate_limit, 3)), session_id=session_id)
            except Exception:
                get_metrics().record_degraded_operation("query", "source_lane_unavailable")
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

    @staticmethod
    def _with_source_synthesis_bundle(
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
                result = source_synthesis_bundle_result(
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
            result = source_synthesis_bundle_result(
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
            contexts = await self.query(
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
            stat = os.stat(Path(self.session_manager.get(session_id).eventlog.path))
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

        entry = self._query_page_cache.get(key)
        if entry is None:
            return None
        expires_at, cached_fetch_limit, cached_signature, contexts = entry
        if time.monotonic() >= expires_at or cached_signature != log_signature:
            del self._query_page_cache[key]
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

        while len(self._query_page_cache) >= QUERY_PAGE_CACHE_MAX_ENTRIES:
            self._query_page_cache.pop(next(iter(self._query_page_cache)))
        self._query_page_cache[key] = (
            time.monotonic() + QUERY_PAGE_CACHE_TTL_SECONDS,
            fetch_limit,
            log_signature,
            contexts,
        )

    def _invalidate_query_page_cache(self, session_id: str) -> None:
        """Drop cached pages for a session after its memory changes."""
        self._query_page_cache = {
            key: value for key, value in self._query_page_cache.items() if key[1] != session_id
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
        return self._retrieval_cache.verbatim_index(session_id)

    def _session_event_ref_index(self, session_id: str) -> dict[int, tuple[str, str]]:
        """Return a cached seq -> (hash, type) index for the current log state.

        Follows the verbatim-index pattern: rebuilt whenever the Eventloom
        file signature changes, so reinforcement emitters that run outside a
        checkout (feedback) can canonicalize 12-char citation fragments into
        full-hash target refs without re-reading the log per call.
        """
        eventlog = self.session_manager.get(session_id).eventlog
        signature = _eventlog_file_signature(eventlog)
        cached = self._event_ref_index_cache.get(session_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        index = event_ref_index(eventlog.read_all())
        self._event_ref_index_cache[session_id] = (signature, index)
        return index

    def _session_cue_index(self, session_id: str) -> dict[int, frozenset[str]]:
        """Return a cached seq -> normalized-cue-pairs index for the session log.

        Follows the verbatim-index signature pattern; only events whose
        payload carries a well-formed ``cues`` record appear.
        """
        eventlog = self.session_manager.get(session_id).eventlog
        signature = _eventlog_file_signature(eventlog)
        cached = self._session_cue_index_cache.get(session_id)
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
        self._session_cue_index_cache[session_id] = (signature, index)
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
        if not cues or not self.retrieval_profile.cue_blending or not contexts:
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
        replay = self.session_manager.replay(session_id, from_seq=1)
        query_tokens = _tokens(query)
        contexts: list[Context] = []
        for event in replay.events:
            if getattr(event, "type", None) == REINFORCEMENT_EVENT_TYPE:
                # Salience bookkeeping is never retrievable context.
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
        return await asyncio.to_thread(self._retrieval_cache.verified_replay, session_id, from_seq)

    async def propose_consolidation_candidates(
        self,
        *,
        session_id: str = "default",
        actor: str = "zaxy-consolidation",
        purpose: str | None = None,
        window_size: int = 8,
    ) -> dict[str, Any]:
        """Append cited, review-pending consolidation candidates for a session log."""
        from zaxy.consolidation_pipeline import (
            generate_consolidation_proposals,
            select_consolidation_segments,
        )

        sid = validate_session_id(session_id)
        eventlog = self.session_manager.get(sid).eventlog
        segments = select_consolidation_segments(
            eventlog.read_all(),
            session_id=sid,
            window_size=window_size,
        )
        proposals = generate_consolidation_proposals(segments, purpose=purpose)

        appended: list[dict[str, Any]] = []
        skipped_existing: list[str] = []
        existing_candidate_ids = _consolidation_candidate_ids(eventlog.read_all())
        for proposal in proposals:
            event_spec = proposal.to_candidate_event(actor=actor)
            payload = event_spec["payload"]
            candidate_id = payload["candidate_id"]
            if candidate_id in existing_candidate_ids:
                skipped_existing.append(candidate_id)
                continue
            event = eventlog.append(
                event_spec["event_type"],
                actor=event_spec["actor"],
                payload=validate_payload(payload),
                thread=sid,
            )
            await self._project_event(event, session_id=sid)
            appended.append(
                {
                    "event_type": event.type,
                    "seq": event.seq,
                    "hash": event.hash,
                    "candidate_id": candidate_id,
                    "candidate_type": payload["candidate_type"],
                }
            )
            existing_candidate_ids.add(candidate_id)

        return {
            "session_id": sid,
            "segment_count": len(segments),
            "candidate_count": len(appended),
            "skipped_existing_count": len(skipped_existing),
            "skipped_existing_candidate_ids": skipped_existing,
            "events": appended,
        }

    async def consolidation_status(self, *, session_id: str = "default") -> dict[str, Any]:
        """Summarize consolidation candidate and review state from Eventloom replay."""
        sid = validate_session_id(session_id)
        replay = await self.replay(session_id=sid)

        candidates: dict[str, dict[str, Any]] = {}
        review_count = 0
        duplicate_candidate_count = 0
        for event in replay.events:
            if event.type == "consolidation.candidate.created":
                candidate_id = event.payload.get("candidate_id")
                if isinstance(candidate_id, str) and candidate_id:
                    if candidate_id in candidates:
                        duplicate_candidate_count += 1
                        continue
                    candidates[candidate_id] = {
                        "candidate_id": candidate_id,
                        "candidate_type": event.payload.get("candidate_type"),
                        "review_status": event.payload.get("review_status", "pending"),
                        "authority_status": "non_authoritative",
                        "created_seq": event.seq,
                        "created_hash": event.hash,
                    }
                    if event.payload.get("stale") is True:
                        candidates[candidate_id]["stale"] = True
                    superseded_by = event.payload.get("superseded_by")
                    if isinstance(superseded_by, str) and superseded_by:
                        candidates[candidate_id]["superseded_by"] = superseded_by
                    valid_to = event.payload.get("valid_to")
                    if isinstance(valid_to, str) and valid_to:
                        candidates[candidate_id]["valid_to"] = valid_to
            elif event.type == "consolidation.candidate.reviewed":
                candidate_id = event.payload.get("candidate_id")
                if isinstance(candidate_id, str) and candidate_id in candidates:
                    review_count += 1
                    candidates[candidate_id]["review_status"] = event.payload.get(
                        "status",
                        candidates[candidate_id]["review_status"],
                    )
                    candidates[candidate_id]["authority_status"] = "non_authoritative"
                    candidates[candidate_id]["reviewed_seq"] = event.seq
                    candidates[candidate_id]["reviewed_hash"] = event.hash

        review_status_counts: dict[str, int] = {}
        authority_status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        stale_count = 0
        superseded_count = 0
        valid_to_count = 0
        for candidate in candidates.values():
            _increment_count(review_status_counts, str(candidate.get("review_status", "unknown")))
            _increment_count(
                authority_status_counts,
                str(candidate.get("authority_status", "unknown")),
            )
            _increment_count(type_counts, str(candidate.get("candidate_type", "unknown")))
            if candidate.get("stale") is True or candidate.get("review_status") == "stale":
                stale_count += 1
            if isinstance(candidate.get("superseded_by"), str) and candidate.get("superseded_by"):
                superseded_count += 1
            if isinstance(candidate.get("valid_to"), str) and candidate.get("valid_to"):
                valid_to_count += 1

        return {
            "session_id": sid,
            "candidate_count": len(candidates),
            "review_count": review_count,
            "duplicate_candidate_count": duplicate_candidate_count,
            "pending_count": review_status_counts.get("pending", 0),
            "accepted_count": review_status_counts.get("accepted", 0),
            "rejected_count": review_status_counts.get("rejected", 0),
            "deferred_count": review_status_counts.get("deferred", 0),
            "conflicted_count": review_status_counts.get("conflicted", 0),
            "stale_count": stale_count,
            "superseded_count": superseded_count,
            "valid_to_count": valid_to_count,
            "review_status_counts": dict(sorted(review_status_counts.items())),
            "authority_status_counts": dict(sorted(authority_status_counts.items())),
            "candidate_type_counts": dict(sorted(type_counts.items())),
            "candidates": sorted(candidates.values(), key=lambda item: item["created_seq"]),
        }

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
    ) -> ContextAssembly:
        """Assemble recent replay plus retrieval into prompt-ready context.

        ``cues`` is additive and only affects retrieval under the cognitive
        retrieval profile (see :meth:`query`).
        """
        sid = validate_session_id(session_id)
        prompt_limit = validate_limit(limit)
        base_candidate_limit = prompt_limit if recall_limit is None else validate_limit(max(prompt_limit, recall_limit))
        profile = purpose_profile(purpose)
        retrieval_policy = purpose_retrieval_policy(
            profile,
            query,
            prompt_limit=prompt_limit,
            base_recall_limit=base_candidate_limit,
        )
        candidate_limit = validate_limit(
            min(MAX_QUERY_LIMIT, max(base_candidate_limit, retrieval_policy.min_recall_limit))
        )
        retrieval_query = retrieval_policy.retrieval_query
        replay = await self.replay(from_seq=replay_from_seq, session_id=sid)
        graph_contexts = await self.query(
            retrieval_query,
            limit=candidate_limit,
            session_id=sid,
            include_source_lane=False,
            scoring_profile=retrieval_policy.scoring_profile,
            cues=cues,
        )
        verbatim_candidate_limit = self.context_assembly_policy.verbatim_candidate_limit(
            query=retrieval_query,
            limit=candidate_limit,
        )
        verbatim_contexts = (
            await self.query_verbatim(retrieval_query, limit=verbatim_candidate_limit, session_id=sid)
            if verbatim_candidate_limit > 0
            else []
        )
        replay_events = list(replay.events)
        if as_of_seq is not None:
            replay_events = [event for event in replay_events if event.seq <= as_of_seq]
        session_events = list(replay_events)
        purpose_outcomes = _purpose_outcome_aggregates(replay_events, profile)
        graph_contexts = _apply_purpose_outcome_learning(graph_contexts, purpose_outcomes)
        verbatim_contexts = _apply_purpose_outcome_learning(verbatim_contexts, purpose_outcomes)
        packet_memory_contexts = self._recent_packet_memory_contexts(replay_events)
        packet_memory_contexts = _apply_purpose_outcome_learning(packet_memory_contexts, purpose_outcomes)
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
        working_set_payload["purpose_retrieval_policy"] = retrieval_policy.to_diagnostics(
            base_recall_limit=base_candidate_limit,
            resolved_recall_limit=candidate_limit,
        )
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
            replay_events=session_events,
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
    ) -> MemoryCheckout:
        """Checkout the current cited memory state an agent should condition on.

        ``record_reinforcement=False`` skips the best-effort 'surfaced'
        salience reinforcement append for read-only inspection surfaces
        (e.g. the dashboard) that must not write to the log.

        ``cues`` (optional, additive) carries the caller's
        encoding-specificity context; it only affects ranking under the
        cognitive retrieval profile.
        """
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
            purpose=purpose,
            cues=cues,
        )
        checkout = build_memory_checkout(
            query=query,
            assembly=assembly,
            ref=resolved_ref,
            purpose=purpose,
            now=datetime.now(UTC),
            retrieval_profile=self.retrieval_profile,
            cues=cues,
            salience_floor=self._salience_floor,
            salience_half_life_days=self._salience_half_life_days,
        )
        if record_reinforcement:
            await self._record_surfaced_reinforcement(
                checkout,
                assembly,
                session_id=checkout_session_id,
                ref=resolved_ref,
            )
        return checkout

    async def _record_surfaced_reinforcement(
        self,
        checkout: MemoryCheckout,
        assembly: ContextAssembly,
        *,
        session_id: str,
        ref: MemoryRef | None,
    ) -> None:
        """Append one batched 'surfaced' salience reinforcement for a checkout.

        Best-effort observability state: targets are the sealed event refs of
        the facts/evidence actually carried by the packet, resolved against
        the replay the checkout was computed from (no extra log scan). A
        failure here never fails the checkout itself.
        """
        try:
            events = assembly.replay_events
            if not events:
                return
            index = event_ref_index(events)
            citations = [
                item.get("citation")
                for item in [*checkout.current_facts, *checkout.evidence]
            ]
            targets = reinforcement_targets_from_citations(citations, event_index=index)
            if not targets:
                return
            checkout_id = _checkout_source_id(ref, events, session_id=session_id)
            spec = build_surfaced_reinforcement_event(
                actor="zaxy-memory",
                session_id=session_id,
                checkout_id=checkout_id,
                targets=targets,
            )
            await self._append_event_spec(spec, session_id=session_id)
        except Exception:
            get_metrics().record_degraded_operation("append", "salience_reinforcement_unavailable")

    def _resolve_checkout_ref(self, ref: str | None, *, session_id: str) -> MemoryRef | None:
        if ref is None:
            return None
        if ref == "HEAD":
            latest = self.session_manager.get(session_id).eventlog.last_event()
            if latest is None:
                return None
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
        purpose: PurposeProfile | dict[str, Any] | str | None = None,
        outcome: str | None = None,
    ) -> int:
        """Append feedback events for retrieved context without mutating history."""
        sid = validate_session_id(session_id)
        normalized = _normalize_context_feedback(feedback)
        purpose_payload = _feedback_purpose_payload(purpose)
        outcome_value = _feedback_outcome(outcome)
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
            if purpose_payload:
                payload["purpose"] = purpose_payload
            if outcome_value:
                payload["outcome"] = outcome_value
            if normalized in {"used", "helpful"}:
                payload.pop("feedback")
                if importance is not None:
                    payload["importance"] = max(0.0, min(1.0, float(importance)))
                feedback_event = await self.append(
                    "memory.reinforced",
                    actor=actor,
                    payload=payload,
                    session_id=sid,
                )
                await self._record_confirmed_reinforcement(
                    context,
                    feedback_event=feedback_event,
                    session_id=sid,
                    actor=actor,
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

    async def _record_confirmed_reinforcement(
        self,
        context: Context,
        *,
        feedback_event: Any,
        session_id: str,
        actor: str,
    ) -> None:
        """Append a 'confirmed' salience reinforcement for positive feedback.

        Best-effort observability state: emitted only when the reinforced
        context carries a citation that resolves to a sealed event in this
        session's log. A failure here never fails the feedback itself.
        """
        try:
            citation = (context.metadata or {}).get("citation")
            if not isinstance(citation, str) or not citation:
                return
            index = self._session_event_ref_index(session_id)
            targets = reinforcement_targets_from_citations([citation], event_index=index)
            if not targets:
                return
            feedback_id = _event_citation(feedback_event) or f"{session_id}:feedback"
            spec = build_confirmed_reinforcement_event(
                actor=actor,
                session_id=session_id,
                feedback_id=feedback_id,
                targets=targets,
            )
            await self._append_event_spec(spec, session_id=session_id)
        except Exception:
            get_metrics().record_degraded_operation("append", "salience_reinforcement_unavailable")

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
        normalized = normalize_synthesis_outcome(outcome)
        sid = validate_session_id(checkout.session_id)
        payload = build_synthesis_candidate_event_payload(
            checkout=checkout,
            candidate=candidate,
            outcome=normalized,
            reason=reason,
        )
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                get_metrics().record_degraded_operation("append", "graph_connect_unavailable")
                self._connected = False
        eventlog = self.session_manager.get(sid).eventlog
        event = eventlog.append(
            synthesis_outcome_event_type(normalized),
            actor=actor,
            payload=validate_payload(payload),
            thread=sid,
        )
        await self._project_event(event, session_id=sid)
        await self._append_generated_inferences(eventlog, source_event=event, session_id=sid)
        self._invalidate_query_page_cache(sid)
        return event

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
        normalized = normalize_synthesis_outcome(outcome)
        sid = validate_session_id(checkout.session_id)
        payload = build_synthesis_evidence_event_payload(
            checkout=checkout,
            row=row,
            outcome=normalized,
            candidate=candidate,
            reason=reason,
        )
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                get_metrics().record_degraded_operation("append", "graph_connect_unavailable")
                self._connected = False
        eventlog = self.session_manager.get(sid).eventlog
        event_type = "memory.evidence.reinforced" if normalized == "used" else synthesis_outcome_event_type(normalized)
        event = eventlog.append(
            event_type,
            actor=actor,
            payload=validate_payload(payload),
            thread=sid,
        )
        await self._project_event(event, session_id=sid)
        await self._append_generated_inferences(eventlog, source_event=event, session_id=sid)
        self._invalidate_query_page_cache(sid)
        return event

    async def record_synthesis_artifact(
        self,
        checkout: MemoryCheckout,
        *,
        actor: str = "zaxy",
    ) -> Any:
        """Append a deterministic synthesis artifact created from checkout state."""
        sid = validate_session_id(checkout.session_id)
        payload = build_synthesis_artifact(checkout)
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                get_metrics().record_degraded_operation("append", "graph_connect_unavailable")
                self._connected = False
        eventlog = self.session_manager.get(sid).eventlog
        event = eventlog.append(
            "memory.synthesis.artifact.created",
            actor=actor,
            payload=validate_payload(payload),
            thread=sid,
        )
        await self._project_event(event, session_id=sid)
        await self._append_generated_inferences(eventlog, source_event=event, session_id=sid)
        self._invalidate_query_page_cache(sid)
        return event

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

    async def coordinate_start_mission(
        self,
        mission_id: str,
        *,
        objective: str,
        actor: str = "coordinator",
    ) -> Any:
        """Start a parent coordination mission and project it."""
        result = self._coordination_manager().start_mission(mission_id, objective=objective, actor=actor)
        await self._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_create_worker(
        self,
        mission_id: str,
        worker_id: str,
        *,
        actor: str = "coordinator",
    ) -> Any:
        """Register a worker session under a parent mission and project it."""
        result = self._coordination_manager().create_worker(mission_id, worker_id, actor=actor)
        await self._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_assign(
        self,
        mission_id: str,
        worker_id: str,
        assignment: str,
        *,
        actor: str = "coordinator",
    ) -> Any:
        """Assign scoped work to a coordination worker and project it."""
        result = self._coordination_manager().assign(mission_id, worker_id, assignment, actor=actor)
        await self._project_event(result.event, session_id=result.mission_id)
        return result

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
        result = self._coordination_manager().report_finding(
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
        await self._project_event(result.event, session_id=result.worker_id or worker_id)
        return result

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
        result = self._coordination_manager().review_finding(
            mission_id,
            finding_id,
            status=status,
            actor=actor,
            rationale=rationale,
        )
        await self._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_promote_finding(
        self,
        mission_id: str,
        finding_id: str,
        *,
        actor: str = "coordinator",
    ) -> Any:
        """Promote a finding into the parent mission history and project it."""
        result = self._coordination_manager().promote_finding(mission_id, finding_id, actor=actor)
        await self._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_brief(self, mission_id: str) -> Any:
        """Return a replay-backed coordination brief."""
        return self._coordination_manager().brief(mission_id)

    async def coordinate_checkout(self, mission_id: str, *, include_diagnostics: bool = False) -> Any:
        """Return accepted coordination state for prompt injection."""
        return self._coordination_manager().checkout(mission_id, include_diagnostics=include_diagnostics)

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
        mission_sid = validate_session_id(mission_id)
        if validate_session_id(checkout.session_id) != mission_sid:
            raise ValueError("Coordinate synthesis checkout session_id must match mission_id")
        artifact_payload = build_synthesis_artifact(checkout)
        proof_packet = self._coordination_manager().proof_packet(
            mission_sid,
            artifact_payload,
            decision_scope=decision_scope,
            handoff_id=handoff_id,
        )
        proof_payload = validate_payload(proof_packet.to_dict())
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                get_metrics().record_degraded_operation("append", "graph_connect_unavailable")
                self._connected = False
        eventlog = self.session_manager.get(mission_sid).eventlog
        artifact_event = eventlog.append(
            "memory.synthesis.artifact.created",
            actor=actor,
            payload=validate_payload(artifact_payload),
            thread=mission_sid,
        )
        await self._project_event(artifact_event, session_id=mission_sid)
        await self._append_generated_inferences(eventlog, source_event=artifact_event, session_id=mission_sid)
        proof_event = eventlog.append(
            "coordination.proof_packet.created",
            actor=actor,
            payload=proof_payload,
            thread=mission_sid,
        )
        await self._project_event(proof_event, session_id=mission_sid)
        await self._append_generated_inferences(eventlog, source_event=proof_event, session_id=mission_sid)
        return {
            "artifact_id": artifact_payload["artifact_id"],
            "artifact_event": {
                "seq": artifact_event.seq,
                "hash": artifact_event.hash,
                "event_type": artifact_event.type,
            },
            "proof_event": {
                "seq": proof_event.seq,
                "hash": proof_event.hash,
                "event_type": proof_event.type,
            },
            "proof_packet": proof_payload,
        }

    async def coordinate_performance_ledger(self, mission_id: str) -> Any:
        """Return replay-backed worker outcome metrics for a coordination mission."""
        return self._coordination_manager().performance_ledger(mission_id)

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
        result = self._coordination_manager().create_handoff(
            mission_id,
            summary=summary,
            actor=actor,
            next_steps=next_steps,
            risks=risks,
        )
        await self._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_approval_packet(self, mission_id: str) -> Any:
        """Return a portable remote approval packet for pending coordination findings."""
        return self._coordination_manager().approval_packet(mission_id)

    async def coordinate_apply_approval_decisions(
        self,
        mission_id: str,
        decisions: list[dict[str, Any]],
        *,
        actor: str = "coordinator",
    ) -> Any:
        """Apply remote approval decisions and project all resulting events."""
        result = self._coordination_manager().apply_approval_decisions(
            mission_id,
            decisions,
            actor=actor,
        )
        for event in result.events:
            await self._project_event(event, session_id=result.mission_id)
        return result

    async def coordinate_record_detected_conflicts(
        self,
        mission_id: str,
        *,
        actor: str = "zaxy",
    ) -> Any:
        """Materialize deterministic coordination conflicts and project them."""
        results = self._coordination_manager().record_detected_conflicts(
            mission_id,
            actor=actor,
        )
        for result in results:
            await self._project_event(result.event, session_id=result.mission_id)
        return results

    def _coordination_manager(self) -> Any:
        """Return a coordination manager bound to this fabric's session manager."""
        from zaxy.coordination import CoordinationManager
        from zaxy.coordination_semantic import build_semantic_conflict_detector

        manager = CoordinationManager(
            eventloom_path=self.eventloom_path,
            semantic_conflict_detector=build_semantic_conflict_detector(self.settings),
        )
        manager.session_manager = self.session_manager
        return manager

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
        return self.session_manager.handoff_summary(session_id)
