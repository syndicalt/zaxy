"""Core memory fabric API.

The MemoryFabric is the primary interface for agents to persist and query
context. It coordinates between Eventloom (immutable log), the temporal
selected projection graph, hybrid extraction, and optional tracing.

Example::

    fabric = MemoryFabric(
        eventloom_path=".eventloom/agent.jsonl",
    )
    await fabric.connect()
    await fabric.append("goal.created", actor="user", payload={"title": "Ship it"})
    context = await fabric.query("What are our goals?")
    await fabric.close()
"""

from __future__ import annotations

import json
import os
import re
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from zaxy.causal import (
    CausalQueryResult,
    causal_query_result_from_projection,
    causal_relation_to_graph_relation,
)
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
from zaxy.event import (  # noqa: F401 - ReplayResult re-export for existing tests
    EventLog,
    IntegrityReport,
    ReplayResult,
    verify_event_chain,
)
from zaxy.evidence import select_checkout_evidence
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
from zaxy.recall import RecallCandidateSet, build_recall_candidate_set, empty_recall_candidate_set
from zaxy.refs import MemoryRef, MemoryRefStore
from zaxy.retrieval_intent import classify_retrieval_intent
from zaxy.retrieval_plan import (
    absence_check_bundle,
    bridge_source_lane_queries,
    build_evidence_plan,
    should_try_absence_bundle_first,
    source_context_group,
    source_lane_candidate_limit,
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
    SALIENCE_HALF_LIFE_DAYS,
    SALIENCE_MAX,
    SALIENCE_MIN,
    EncodingDecision,
    EventRef,
    SalienceLedger,
    SalienceState,
    build_confirmed_reinforcement_event,
    build_invalidated_reinforcement_event,
    build_reinforcement_event,
    build_surfaced_reinforcement_event,
    classify_append,
    cue_overlap,
    cue_pairs,
    event_ref_index,
    reinforcement_targets_from_citations,
    resolve_citation_target,
    target_ref,
)
from zaxy.security import (
    MAX_QUERY_LIMIT,
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
from zaxy.synthesis_packet import synthesis_packet_from_items
from zaxy.trace import MemoryTracer
from zaxy.transcripts import collect_transcript_events
from zaxy.verbatim import VerbatimIndex, _chunks_from_events
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
    #: Full as-of-filtered replay the assembly was computed against. Carried so
    #: checkout can resolve citations to sealed event refs and replay salience
    #: without re-reading the log; never serialized into payloads.
    replay_events: list[Any] = field(default_factory=list)


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
    purpose: dict[str, Any] = field(default_factory=dict)

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
            "token_efficiency": checkout_token_efficiency(
                prompt=self.prompt,
                current_fact_count=len(self.current_facts),
                evidence_count=len(self.evidence),
            ),
            "compacted": self.compacted,
            "assembly_policy": self.assembly_policy,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class QueryPage:
    """A stable page of ranked memory query results."""

    contexts: list[Context]
    next_cursor: str | None
    cursor: str | None
    has_more: bool
    offset: int


def checkout_token_efficiency(
    *,
    prompt: str,
    current_fact_count: int,
    evidence_count: int,
) -> dict[str, int | float]:
    """Estimate Memory Checkout token efficiency for activation diagnostics."""
    prompt_tokens = _approx_tokens(prompt)
    return {
        "prompt_tokens": prompt_tokens,
        "current_fact_count": current_fact_count,
        "evidence_count": evidence_count,
        "facts_per_1k_prompt_tokens": round((current_fact_count / prompt_tokens) * 1000, 3)
        if prompt_tokens
        else 0.0,
    }


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)

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
        embedded_graph_path: str | Path | None = None,
        latticedb_path: str | Path | None = None,
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
        self._verbatim_index_cache: dict[str, tuple[tuple[int, int], VerbatimIndex, int]] = {}
        self._replay_cache: dict[str, tuple[tuple[int, int], ReplayResult, int]] = {}
        self._event_ref_index_cache: dict[str, tuple[tuple[int, int], dict[int, tuple[str, str]]]] = {}
        self._session_cue_index_cache: dict[str, tuple[tuple[int, int], dict[int, frozenset[str]]]] = {}
        self._query_page_cache: dict[
            tuple[str, str, str | None, tuple[float, ...] | None],
            tuple[float, int, tuple[int, int] | None, list[Context]],
        ] = {}
        self._initialized_workspaces: dict[tuple[str, str], WorkspaceProfile] = {}
        self._initialized_instruction_signatures: dict[tuple[str, str], str] = {}
        self._warmed_projection_sessions: set[str] = set()
        self._connected = False

    async def connect(self) -> None:
        """Connect to projection backend and tracer. Idempotent."""
        if self._connected:
            return
        await self.graph.connect()
        await self.graph.init_schema()
        await self._warm_projection_session(self.settings.eventloom_thread)
        self._warm_source_index(self.settings.eventloom_thread)
        await self.tracer.connect()
        self._connected = True

    async def close(self) -> None:
        """Close all connections. Idempotent."""
        await self.graph.close()
        await self.tracer.close()
        self._verbatim_index_cache = {}
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
        index = self._verbatim_index(sid)
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

    def _verbatim_index(self, session_id: str) -> VerbatimIndex:
        """Return a verbatim index for the current Eventloom file state.

        The index is cached per session and extended incrementally: when the
        append-only log has only grown, just the newly appended events are read
        and tokenized (:meth:`VerbatimIndex.append_chunks`) instead of
        rebuilding the BM25 index over the whole corpus on every change. The
        stored cursor is the exact byte offset that was indexed, so concurrent
        appends during a build never cause missed or duplicated events. A full
        rebuild only happens on a cold cache or if the log shrank / was
        rewritten (e.g. compaction).
        """
        eventlog = self.session_manager.get(session_id).eventlog
        signature = _eventlog_file_signature(eventlog)
        cached = self._verbatim_index_cache.get(session_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        if cached is not None and signature[1] > cached[2]:
            new_events, end_offset = eventlog.read_from_offset(cached[2])
            index = cached[1].append_chunks(_chunks_from_events(new_events))
            self._verbatim_index_cache[session_id] = (signature, index, end_offset)
            return index
        events, end_offset = eventlog.read_from_offset(0)
        index = VerbatimIndex.from_events(events)
        self._verbatim_index_cache[session_id] = (signature, index, end_offset)
        return index

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
        """
        result = self._cached_full_replay(session_id)
        if from_seq <= 1:
            return result
        filtered = [event for event in result.events if event.seq >= from_seq]
        return ReplayResult.model_construct(events=filtered, integrity=result.integrity)

    def _cached_full_replay(self, session_id: str) -> ReplayResult:
        """Return the full verified replay for a session, cached + incremental.

        The cold/full path delegates to ``session_manager.replay`` (the
        authoritative read + full integrity verify). When the cached log has
        only grown, the appended tail is read and verified against the cached
        prefix instead. The tail verification doubles as a consistency guard:
        any offset skew (a concurrent append during the cold read, a rewrite)
        surfaces as a seq/hash mismatch and falls back to a full replay, so the
        fast path can never silently miss, duplicate, or trust a bad event.
        """
        eventlog = self.session_manager.get(session_id).eventlog
        signature = _eventlog_file_signature(eventlog)
        cached = self._replay_cache.get(session_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        if (
            cached is not None
            and isinstance(cached[2], int)
            and isinstance(signature[1], int)
            and signature[1] > cached[2]
            and cached[1].integrity is not None
            and cached[1].integrity.ok
        ):
            new_events, end_offset = eventlog.read_from_offset(cached[2])
            extended = self._extend_replay(cached[1], new_events)
            if extended is not None:
                self._replay_cache[session_id] = (signature, extended, end_offset)
                return extended
        result = cast(ReplayResult, self.session_manager.replay(session_id, from_seq=1))
        offset = _eventlog_file_signature(eventlog)[1]
        self._replay_cache[session_id] = (signature, result, offset)
        return result

    @staticmethod
    def _extend_replay(cached: ReplayResult, new_events: list[Any]) -> ReplayResult | None:
        """Extend a verified replay with appended events, or None to rebuild.

        Verifies only the new tail against the cached prefix's last event.
        Returns ``None`` (signalling a full re-verify) when the tail fails
        verification, so a tampered or reordered append never silently passes.
        """
        if not new_events:
            return cached
        last = cached.events[-1] if cached.events else None
        tail = verify_event_chain(
            new_events,
            first_seq=(last.seq + 1) if last is not None else 1,
            prev_hash=last.hash if last is not None else None,
        )
        if not tail.ok:
            return None
        combined = [*cached.events, *new_events]
        return ReplayResult(
            events=combined,
            integrity=IntegrityReport(ok=True, total_events=len(combined)),
        )

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


#: Event types the encoding gate never classifies: cognitive-memory
#: bookkeeping that must not recurse through (or be reshaped by) the gate.
_ENCODING_GATE_SKIPPED_EVENT_TYPES = frozenset(
    {
        REINFORCEMENT_EVENT_TYPE,
        "belief.update.proposed",
        "reasoning.primitive.called",
        "inference.edge.generated",
    }
)

#: Payload keys whose string values declare a candidate entity identity at
#: append time (used for the gate's entity-name overlap signal).
_ENCODING_ENTITY_NAME_KEYS = ("entity_name", "name", "taskId", "task_id", "task")

#: Projected-property keys excluded from interference value comparison:
#: provenance, retention bookkeeping, and free-text summaries are not values.
_INTERFERENCE_EXCLUDED_PROPERTY_KEYS = frozenset(
    {
        "summary",
        "embedding",
        "embedding_version",
        "created_at",
        "updated_at",
        "observed_at",
        "expires_at",
        "last_reinforced_at",
        "importance",
        "reinforcement_count",
        "retrieval_salience",
        "source_event_seq",
        "source_event_hash",
        "source_event_prev_hash",
        "source_thread",
        "node_key",
        "session_id",
    }
)

_ENCODING_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _encoding_gate_eligible(event_type: str, payload: dict[str, Any]) -> bool:
    """Return whether an append should be classified by the encoding gate."""
    if event_type in _ENCODING_GATE_SKIPPED_EVENT_TYPES:
        return False
    return "encoding" not in payload


#: Payload keys excluded from gate content comparison: write-time metadata
#: (gate tags, cue records) is context about the memory, not its content.
_ENCODING_CONTENT_EXCLUDED_KEYS = frozenset({"encoding", "cues"})


def _encoding_classification_content(payload: dict[str, Any]) -> str:
    """Return the canonical payload text the gate compares against memory.

    Mirrors the verbatim index's event-chunk text (sorted-key JSON), with
    gate/cue metadata stripped so tagging an event never dilutes later
    duplicate detection against it.
    """
    comparable = {
        key: value
        for key, value in payload.items()
        if key not in _ENCODING_CONTENT_EXCLUDED_KEYS
    }
    if not comparable:
        return ""
    try:
        return json.dumps(comparable, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return ""


def _encoding_tokens(text: str) -> set[str]:
    return set(_ENCODING_TOKEN_RE.findall(text.casefold()))


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / (len(left) + len(right) - intersection)


def _payload_entity_names(payload: dict[str, Any]) -> list[str]:
    """Return bounded candidate entity names declared by a payload."""
    names: list[str] = []
    for key in _ENCODING_ENTITY_NAME_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and len(value) <= 200:
            name = value.strip()
            if name not in names:
                names.append(name)
    return names


def _conflicting_property_value(
    old_properties: dict[str, Any],
    new_properties: dict[str, Any],
) -> tuple[str, Any, Any] | None:
    """Return the first shared scalar property whose values conflict."""
    for key in sorted(set(old_properties) & set(new_properties)):
        if key in _INTERFERENCE_EXCLUDED_PROPERTY_KEYS or key.startswith("_"):
            continue
        old_value = old_properties[key]
        new_value = new_properties[key]
        if not _is_comparable_scalar(old_value) or not _is_comparable_scalar(new_value):
            continue
        old_text = str(old_value).strip().casefold()
        new_text = str(new_value).strip().casefold()
        if old_text and new_text and old_text != new_text:
            return (key, old_value, new_value)
    return None


def _is_comparable_scalar(value: Any) -> bool:
    return isinstance(value, str | int | float | bool)


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


def _checkout_source_id(ref: MemoryRef | None, events: list[Any], *, session_id: str) -> str:
    """Return the stable as-of identity a checkout packet was computed against.

    A ref checkout is identified by the resolved ref target; a HEAD checkout
    by the citation of the last replayed event — both identities checkout
    already produces, derived from data in hand (no log read).
    """
    if ref is not None:
        return f"eventloom://{ref.session_id}/events/{ref.target_seq}#{ref.target_hash[:12]}"
    citation = _event_citation(events[-1]) if events else None
    return citation if citation is not None else f"{session_id}:HEAD"


def _invalidation_source_id(*, entity_name: str, entity_type: str, invalid_at: str) -> str:
    """Return the natural key of one invalidation operation."""
    return f"invalidate:{entity_type}:{entity_name}@{invalid_at}"


def entity_reinforcement_targets(entities: Any) -> list[dict[str, Any]]:
    """Return builder-ready reinforcement targets from projected entities.

    Reads the ``source_event_seq`` / ``source_event_hash`` provenance stored
    on projected entity properties; entities without sealed full-hash
    provenance are skipped, and duplicates collapse to one target.
    """
    targets: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for entity in entities or []:
        properties = getattr(entity, "properties", None)
        if not isinstance(properties, dict):
            continue
        target = target_ref(properties.get("source_event_seq"), properties.get("source_event_hash"))
        if target is None:
            continue
        key = (int(target["seq"]), str(target["hash"]))
        if key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return targets


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


_POSITIVE_PURPOSE_OUTCOMES = {
    "avoided_failed_path",
    "blocked_unsafe_action",
    "changed_answer",
    "helpful",
    "prevented_redundant_investigation",
    "resolved_conflict",
    "supported_handoff",
    "used",
}
_NEGATIVE_PURPOSE_OUTCOMES = {
    "caused_regression",
    "corrected",
    "excluded",
    "failed",
    "irrelevant",
    "rejected",
}


def _purpose_outcome_aggregates(
    events: list[Any],
    profile: PurposeProfile,
) -> dict[str, dict[str, Any]]:
    """Return replay-derived purpose outcome counts keyed by stable memory identity."""
    if profile.profile == "general":
        return {}
    aggregates: dict[str, dict[str, Any]] = {}
    for event in events:
        event_type = getattr(event, "type", "")
        if event_type not in {
            "memory.reinforced",
            "memory.feedback",
            "memory.evidence.reinforced",
            "memory.evidence.excluded",
        }:
            continue
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict) or not _purpose_payload_matches(payload, profile):
            continue
        outcome = _purpose_feedback_outcome(event_type, payload)
        if outcome is None:
            continue
        polarity = _purpose_outcome_polarity(outcome)
        if polarity is None:
            continue
        keys = _purpose_outcome_payload_keys(payload)
        if not keys:
            continue
        for key in keys:
            aggregate = aggregates.setdefault(
                key,
                {
                    "positive_count": 0,
                    "negative_count": 0,
                    "outcomes": [],
                    "latest_event_seq": None,
                },
            )
            count_key = "positive_count" if polarity == "positive" else "negative_count"
            aggregate[count_key] = int(aggregate[count_key]) + 1
            if outcome not in aggregate["outcomes"]:
                aggregate["outcomes"].append(outcome)
            seq = getattr(event, "seq", None)
            if isinstance(seq, int):
                aggregate["latest_event_seq"] = seq
    return aggregates


def _apply_purpose_outcome_learning(
    contexts: list[Context],
    aggregates: dict[str, dict[str, Any]],
) -> list[Context]:
    """Return contexts scored with bounded replay-derived outcome learning."""
    if not aggregates:
        return contexts
    learned: list[Context] = []
    for context in contexts:
        aggregate = _purpose_outcome_for_context(context, aggregates)
        if aggregate is None:
            learned.append(context)
            continue
        positive_count = int(aggregate.get("positive_count", 0))
        negative_count = int(aggregate.get("negative_count", 0))
        boost = min(0.20, positive_count * 0.06)
        penalty = min(0.18, negative_count * 0.06)
        score_multiplier = max(0.1, 1.0 + boost - penalty)
        metadata = dict(context.metadata or {})
        outcome_payload = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "score_boost": round(boost, 4),
            "score_penalty": round(penalty, 4),
            "outcomes": list(aggregate.get("outcomes", [])),
            "suppression_candidate": negative_count >= 2 and negative_count >= positive_count,
        }
        if aggregate.get("latest_event_seq") is not None:
            outcome_payload["latest_event_seq"] = aggregate["latest_event_seq"]
        metadata.update(
            {
                "purpose_outcome_positive_count": positive_count,
                "purpose_outcome_negative_count": negative_count,
                "purpose_outcome_score_boost": round(boost, 4),
                "purpose_outcome_score_penalty": round(penalty, 4),
                "purpose_outcome_suppression_candidate": outcome_payload["suppression_candidate"],
            }
        )
        score_explanation = dict(metadata.get("score_explanation") or {})
        score_explanation["purpose_outcome"] = outcome_payload
        metadata["score_explanation"] = score_explanation
        learned.append(
            replace(
                context,
                score=context.score * score_multiplier,
                metadata=metadata,
            )
        )
    return sorted(learned, key=lambda item: item.score, reverse=True)


def _purpose_payload_matches(payload: dict[str, Any], profile: PurposeProfile) -> bool:
    purpose = payload.get("purpose")
    value = purpose.get("profile") if isinstance(purpose, dict) else purpose
    if not isinstance(value, str) or not value.strip():
        return False
    return value.strip().casefold().replace(" ", "-") == profile.profile


def _purpose_feedback_outcome(event_type: str, payload: dict[str, Any]) -> str | None:
    outcome = payload.get("outcome")
    if isinstance(outcome, str) and outcome.strip():
        return outcome.strip().casefold().replace(" ", "_")
    feedback = payload.get("feedback")
    if isinstance(feedback, str) and feedback.strip():
        return feedback.strip().casefold().replace(" ", "_")
    if event_type in {"memory.reinforced", "memory.evidence.reinforced"}:
        return "used"
    if event_type == "memory.evidence.excluded":
        return "excluded"
    return None


def _purpose_outcome_polarity(outcome: str) -> str | None:
    if outcome in _POSITIVE_PURPOSE_OUTCOMES:
        return "positive"
    if outcome in _NEGATIVE_PURPOSE_OUTCOMES:
        return "negative"
    return None


def _purpose_outcome_payload_keys(payload: dict[str, Any]) -> list[str]:
    citation = payload.get("citation")
    if isinstance(citation, str) and citation.strip():
        return [f"citation:{citation.strip()}"]
    source_event_hash = payload.get("source_event_hash")
    if isinstance(source_event_hash, str) and source_event_hash.strip():
        return [f"hash:{source_event_hash.strip()}"]
    source_group = payload.get("source_group")
    if isinstance(source_group, str) and source_group.strip():
        return [f"source_group:{source_group.strip()}"]
    entity_name = payload.get("entity_name")
    entity_type = payload.get("entity_type")
    if isinstance(entity_name, str) and entity_name.strip():
        kind = entity_type.strip() if isinstance(entity_type, str) and entity_type.strip() else "memory"
        return [f"entity:{kind}:{entity_name.strip()}"]
    return []


def _purpose_outcome_for_context(
    context: Context,
    aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    keys = _purpose_outcome_context_keys(context)
    merged: dict[str, Any] | None = None
    for key in keys:
        aggregate = aggregates.get(key)
        if aggregate is None:
            continue
        if merged is None:
            merged = {
                "positive_count": 0,
                "negative_count": 0,
                "outcomes": [],
                "latest_event_seq": None,
            }
        merged["positive_count"] = int(merged["positive_count"]) + int(aggregate.get("positive_count", 0))
        merged["negative_count"] = int(merged["negative_count"]) + int(aggregate.get("negative_count", 0))
        for outcome in aggregate.get("outcomes", []):
            if outcome not in merged["outcomes"]:
                merged["outcomes"].append(outcome)
        latest = aggregate.get("latest_event_seq")
        if isinstance(latest, int):
            current = merged.get("latest_event_seq")
            merged["latest_event_seq"] = latest if not isinstance(current, int) else max(current, latest)
    return merged


def _purpose_outcome_context_keys(context: Context) -> list[str]:
    metadata = context.metadata or {}
    citation = _context_citation(context)
    if citation:
        return [f"citation:{citation.strip()}"]
    source_event_hash = metadata.get("source_event_hash")
    if isinstance(source_event_hash, str) and source_event_hash.strip():
        return [f"hash:{source_event_hash.strip()}"]
    identity = _context_identity(context)
    return [f"entity:{identity['entity_type']}:{identity['entity_name']}"]


def _purpose_outcome_suppression_candidates(current_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for fact in current_facts:
        explanation = fact.get("score_explanation")
        if not isinstance(explanation, dict):
            continue
        outcome = explanation.get("purpose_outcome")
        if not isinstance(outcome, dict) or not outcome.get("suppression_candidate"):
            continue
        candidates.append(
            {
                "entity_name": fact.get("entity_name") or _context_content_identity(str(fact.get("content", ""))),
                "entity_type": fact.get("entity_type") or "memory",
                "citation": fact.get("citation"),
                "negative_count": int(outcome.get("negative_count", 0)),
                "positive_count": int(outcome.get("positive_count", 0)),
                "outcomes": [str(value) for value in outcome.get("outcomes", [])],
            }
        )
    return candidates


def _normalize_context_feedback(feedback: str) -> str:
    normalized = feedback.casefold().strip()
    if normalized not in {"used", "helpful", "irrelevant"}:
        raise ValueError("feedback must be one of: used, helpful, irrelevant")
    return normalized


def _feedback_purpose_payload(
    purpose: PurposeProfile | dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    if purpose is None:
        return None
    return purpose_profile(purpose).to_dict()


def _feedback_outcome(outcome: str | None) -> str | None:
    if outcome is None:
        return None
    value = str(outcome).strip()
    return value or None


def build_memory_checkout(
    *,
    query: str,
    assembly: ContextAssembly,
    ref: MemoryRef | None = None,
    purpose: PurposeProfile | dict[str, Any] | str | None = None,
    now: datetime | None = None,
    retrieval_profile: RetrievalProfile | None = None,
    cues: dict[str, str] | None = None,
    salience_floor: float = 0.15,
    salience_half_life_days: float = SALIENCE_HALF_LIFE_DAYS,
) -> MemoryCheckout:
    """Build the Memory Checkout contract from assembled context.

    ``now`` anchors the salience replay; callers on the serving paths pass
    an explicit ``datetime.now(UTC)`` and omitted values fall back to the
    same clock.

    Under every pre-cognitive profile (``retrieval_profile`` omitted or
    with its cognitive flags off) salience never changes ranking, ordering,
    or selection — it is exposed in diagnostics only, byte-identical to the
    pre-cognitive contract. Only the opt-in cognitive profile blends
    salience and cue overlap into ranking and applies the attenuation floor
    (see :func:`_rank_cognitive_contexts` for the blend).
    """
    checkout_now = now if now is not None else datetime.now(UTC)
    profile = purpose_profile(purpose)
    purpose_payload = profile.to_dict()
    checkout_contexts = _checkout_contexts_with_synthesis(query, assembly)
    cognitive = retrieval_profile is not None and (
        retrieval_profile.salience_ranking or retrieval_profile.cue_blending
    )
    attenuation: dict[str, Any] | None = None
    if cognitive and retrieval_profile is not None:
        ranked_contexts, attenuation = _rank_cognitive_contexts(
            checkout_contexts,
            query=query,
            replay_events=assembly.replay_events,
            retrieval_profile=retrieval_profile,
            cues=cues,
            salience_floor=salience_floor,
            salience_half_life_days=salience_half_life_days,
            now=checkout_now,
        )
    else:
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
    candidate_current_facts, candidate_evidence, purpose_policy = _apply_purpose_checkout_policy(
        profile,
        current_facts=candidate_current_facts,
        evidence=candidate_evidence,
    )
    selection = select_checkout_evidence(
        query=query,
        purpose=profile,
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
    if purpose_policy["suppressed_count"]:
        retention["purpose_policy"] = purpose_policy
    suppression_candidates = _purpose_outcome_suppression_candidates(current_facts)
    if suppression_candidates:
        existing_policy = retention.get("purpose_policy")
        policy_payload = dict(existing_policy) if isinstance(existing_policy, dict) else {}
        policy_payload["suppression_candidates"] = suppression_candidates
        retention["purpose_policy"] = policy_payload
        warnings.append("Purpose outcome history marks retrieved memory as a suppression candidate.")
    diagnostics = build_checkout_diagnostics(
        query=query,
        purpose=purpose_payload,
        source_lanes=_checkout_source_lanes(ranked_contexts),
        current_facts=current_facts,
        evidence=evidence,
        retention=retention,
        warnings=warnings,
    )
    if selection.accepted_state is not None:
        diagnostics = {**diagnostics, "accepted_state": _accepted_state_diagnostics(selection.accepted_state)}
    skills = _checkout_skills(ranked_contexts, query)
    if skills:
        diagnostics = {**diagnostics, "skills": {"count": len(skills), "items": skills}}
    skill_analytics = _checkout_skill_analytics(ranked_contexts)
    if skill_analytics["version_count"] or skill_analytics["outcome_count"]:
        diagnostics = {**diagnostics, "skill_analytics": skill_analytics}
    retrieval_profile_diagnostics = assembly.working_set.get("retrieval_profile")
    if isinstance(retrieval_profile_diagnostics, dict):
        diagnostics = {**diagnostics, "retrieval_profile": retrieval_profile_diagnostics}
    purpose_retrieval = assembly.working_set.get("purpose_retrieval_policy")
    if isinstance(purpose_retrieval, dict):
        diagnostics = {**diagnostics, "purpose_retrieval_policy": purpose_retrieval}
    recall_diagnostics = assembly.recall.to_diagnostics()
    if recall_diagnostics["candidate_count"] and recall_diagnostics["candidate_count"] != len(assembly.contexts):
        diagnostics = {**diagnostics, "recall": recall_diagnostics}
    salience = _checkout_salience_diagnostics(
        replay_events=assembly.replay_events,
        current_facts=current_facts,
        evidence=evidence,
        now=checkout_now,
        half_life_days=salience_half_life_days,
    )
    if salience is not None:
        diagnostics = {**diagnostics, "salience": salience}
    if attenuation is not None:
        diagnostics = {**diagnostics, "attenuation": attenuation}
    guidance = build_checkout_guidance(
        query=query,
        purpose=purpose_payload,
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
        purpose=purpose_payload,
    )


def _rank_cognitive_contexts(
    contexts: list[Context],
    *,
    query: str,
    replay_events: list[Any],
    retrieval_profile: RetrievalProfile,
    cues: dict[str, str] | None,
    salience_floor: float,
    salience_half_life_days: float,
    now: datetime,
) -> tuple[list[Context], dict[str, Any] | None]:
    """Rank checkout contexts under the cognitive retrieval profile.

    Blend (documented contract):

    - **Salience multiplier** (``salience_ranking``): relevance is multiplied
      by normalized salience ``m = clamp(score / SALIENCE_BASE,
      [SALIENCE_MIN, SALIENCE_MAX])`` — never-reinforced memories keep the
      implicit base salience 1.0 and rank exactly as before. The multiplier
      applies to the primary token-overlap key and to the fused-score
      tiebreak; the citation/source/type priority keys are unchanged.
    - **Cue bonus** (``cue_blending``): ``CUE_MATCH_WEIGHT *
      jaccard(query_cues, stored_cues)`` is added to the primary key — a
      bounded bonus of at most ``CUE_MATCH_WEIGHT`` (0.25) for a perfect cue
      match. No cues on either side means zero bonus.
    - **Attenuation floor**: memories whose replayed salience is strictly
      below ``salience_floor`` are excluded from default checkout ranking
      and listed (labeled ``attenuated``) in the returned diagnostics.
      Authority-bearing state (accepted review status / accepted-authority
      scope) and payloads pinned with ``"pinned": true`` are exempt: they
      stay in ranking and are listed as exempt. Excluded memories remain
      fully reachable via explicit ``memory_query``/``memory_replay``, which
      never route through this function.
    """
    ref_index = event_ref_index(replay_events)
    payloads_by_seq = _payloads_by_seq(replay_events)
    states: dict[EventRef, SalienceState] = {}
    if retrieval_profile.salience_ranking:
        states = SalienceLedger(half_life_days=salience_half_life_days).replay(
            replay_events,
            now=now,
        )
    query_cues = cue_pairs(cues) if retrieval_profile.cue_blending else frozenset()
    ranked: list[tuple[tuple[float, int, int, int, float, str, float], Context]] = []
    excluded: list[dict[str, Any]] = []
    exempt: list[dict[str, Any]] = []
    for context in contexts:
        citation = _context_citation(context)
        ref = resolve_citation_target(citation, event_index=ref_index)
        payload = payloads_by_seq.get(ref.seq) if ref is not None else None
        multiplier = 1.0
        if retrieval_profile.salience_ranking:
            state = states.get(ref) if ref is not None else None
            salience_score = state.score if state is not None else SALIENCE_BASE
            multiplier = min(max(salience_score / SALIENCE_BASE, SALIENCE_MIN), SALIENCE_MAX)
            if state is not None and ref is not None and state.score < salience_floor:
                entry = {
                    "citation": citation,
                    "seq": ref.seq,
                    "hash": ref.hash,
                    "salience_score": round(state.score, 4),
                    "label": "attenuated",
                }
                exempt_reason = _attenuation_exempt_reason(context, payload)
                if exempt_reason is None:
                    excluded.append(entry)
                    continue
                exempt.append({**entry, "exempt_reason": exempt_reason})
        cue_bonus = 0.0
        if query_cues and payload is not None:
            cue_bonus = CUE_MATCH_WEIGHT * cue_overlap(query_cues, cue_pairs(payload.get("cues")))
        base = _checkout_rank(context, query)
        ranked.append(
            (
                (
                    base[0] * multiplier + cue_bonus,
                    base[1],
                    base[2],
                    base[3],
                    base[4],
                    base[5],
                    base[6] * multiplier,
                ),
                context,
            )
        )
    ordered = [context for _, context in sorted(ranked, key=lambda item: item[0], reverse=True)]
    attenuation: dict[str, Any] | None = None
    if retrieval_profile.salience_ranking:
        attenuation = {
            "authority_status": "non_authoritative",
            "floor": salience_floor,
            "label": "attenuated",
            "excluded_count": len(excluded),
            "excluded": excluded[:20],
            "exempt_count": len(exempt),
            "exempt": exempt[:20],
        }
    return ordered, attenuation


def _payloads_by_seq(replay_events: list[Any]) -> dict[int, dict[str, Any]]:
    """Map replayed event seq -> payload dict for cue/pin/authority lookups."""
    payloads: dict[int, dict[str, Any]] = {}
    for event in replay_events:
        seq = getattr(event, "seq", None)
        payload = getattr(event, "payload", None)
        if isinstance(seq, int) and isinstance(payload, dict):
            payloads[seq] = payload
    return payloads


_ATTENUATION_EXEMPT_AUTHORITIES = frozenset(
    {"accepted", "authoritative", "parent-accepted", "promoted"}
)


def _attenuation_exempt_reason(
    context: Context,
    payload: dict[str, Any] | None,
) -> str | None:
    """Return why a below-floor memory survives attenuation, or None.

    Exempt: payloads pinned with the additive ``"pinned": true`` metadata
    flag, and authority-bearing state — an accepted review status or an
    accepted/authoritative authority scope on either the projected context
    metadata or the source event payload.
    """
    if payload is not None and payload.get("pinned") is True:
        return "pinned"
    metadata = context.metadata or {}
    for source in (metadata, payload or {}):
        if _checkout_policy_text(source.get("review_status")) == "accepted":
            return "authority"
        authority = _checkout_policy_text(
            source.get("authority")
            or source.get("authority_scope")
            or source.get("authority_status")
        )
        if authority in _ATTENUATION_EXEMPT_AUTHORITIES:
            return "authority"
    return None


def _checkout_salience_diagnostics(
    *,
    replay_events: list[Any],
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    now: datetime,
    half_life_days: float = SALIENCE_HALF_LIFE_DAYS,
) -> dict[str, Any] | None:
    """Replay salience over the checkout's own replay for diagnostics only.

    Pure function of the replayed log slice and ``now``: reinforcement events
    fold into per-memory scores, then surfaced packet citations are resolved
    to their sealed refs and annotated with the score composition. Returns
    None when nothing surfaced carries replayed salience, keeping the
    diagnostics payload byte-identical to the pre-salience contract until
    reinforcement events exist.
    """
    if not replay_events:
        return None
    states = SalienceLedger(half_life_days=half_life_days).replay(replay_events, now=now)
    if not states:
        return None
    index = event_ref_index(replay_events)
    items: list[dict[str, Any]] = []
    seen: set[EventRef] = set()
    for item in [*current_facts, *evidence]:
        ref = resolve_citation_target(item.get("citation"), event_index=index)
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        state = states.get(ref)
        if state is None:
            continue
        items.append(
            {
                "citation": item.get("citation"),
                "seq": ref.seq,
                "hash": ref.hash,
                "composition": state.composition(),
            }
        )
    if not items:
        return None
    return {
        "authority_status": "non_authoritative",
        "half_life_days": half_life_days,
        "scored_count": len(items),
        "items": items,
    }


def _checkout_contexts_with_synthesis(query: str, assembly: ContextAssembly) -> list[Context]:
    """Return recall contexts plus a compact checkout-only synthesis proof when available."""
    checkout_contexts = list(assembly.recall.contexts() or assembly.contexts)
    if any(
        (context.metadata or {}).get("source_kind") == "source_synthesis"
        or "zaxy_synthesis_bundle=true" in context.content
        for context in checkout_contexts
    ):
        return checkout_contexts
    source_contexts = [
        context
        for context in checkout_contexts
        if _checkout_source_lane(context) in {"verbatim", "eventloom", "projection"}
    ]
    graph_contexts = [
        context
        for context in checkout_contexts
        if _checkout_source_lane(context) == "graph"
    ]
    synthesis_contexts = _prefer_verbatim_for_duplicate_source_groups(source_contexts, graph_contexts)
    if not synthesis_contexts:
        return checkout_contexts
    result = source_synthesis_bundle_result(
        query=query,
        source_results=synthesis_contexts,
        limit=10,
        preferred_source_groups=[
            source_context_group(_source_context_text(context))
            for context in graph_contexts
        ],
    )
    if result is None:
        return checkout_contexts
    bundle = result.content
    score = max((context.score for context in checkout_contexts), default=0.0) + 1.0
    return [
        Context(
            content=bundle,
            source="verbatim",
            score=score,
            metadata={
                "source_kind": "source_synthesis",
                "assembly_hint": "source_synthesis",
                "checkout_only": True,
                **_synthesis_packet_metadata(bundle, result.packet),
            },
        ),
        *checkout_contexts,
    ]


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
    for key in _CHECKOUT_METADATA_FIELDS:
        value = metadata.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            fact[key] = value
    score_explanation = metadata.get("score_explanation")
    if isinstance(score_explanation, dict):
        fact["score_explanation"] = score_explanation
    synthesis_packet = metadata.get("synthesis_packet")
    if isinstance(synthesis_packet, dict):
        fact["synthesis_packet"] = synthesis_packet
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
    for key in _CHECKOUT_METADATA_FIELDS:
        value = metadata.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            evidence[key] = value
    synthesis_packet = metadata.get("synthesis_packet")
    if isinstance(synthesis_packet, dict):
        evidence["synthesis_packet"] = synthesis_packet
    return evidence


_CHECKOUT_METADATA_FIELDS = (
    "entity_name",
    "entity_type",
    "event_type",
    "primitive",
    "phase",
    "review_status",
    "authority_status",
    "mission_id",
    "worker_id",
    "finding_id",
    "claim_key",
    "claim_value",
    "coordination_status",
    "finding_status",
    "promoted",
    "status",
    "authority",
    "authority_scope",
    "stale",
    "superseded_by",
    "purpose_outcome_positive_count",
    "purpose_outcome_negative_count",
    "purpose_outcome_score_boost",
    "purpose_outcome_score_penalty",
    "purpose_outcome_suppression_candidate",
)


def _apply_purpose_checkout_policy(
    profile: PurposeProfile,
    *,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply purpose suppress rules before facts become model-facing memory."""
    if profile.profile == "general" or not profile.suppress:
        return current_facts, evidence, _empty_purpose_policy(profile)
    kept_facts: list[dict[str, Any]] = []
    kept_evidence: list[dict[str, Any]] = []
    suppressed_ids: set[str] = set()
    reasons: dict[str, int] = {}
    examples: list[dict[str, str]] = []
    for item in current_facts:
        reason = _purpose_suppression_reason(profile, item)
        if reason is None:
            kept_facts.append(item)
            continue
        identity = _checkout_policy_item_id(item)
        suppressed_ids.add(identity)
        reasons[reason] = reasons.get(reason, 0) + 1
        if len(examples) < 5:
            examples.append({"id": identity, "reason": reason})
    for item in evidence:
        reason = _purpose_suppression_reason(profile, item)
        identity = _checkout_policy_item_id(item)
        if reason is None and identity not in suppressed_ids:
            kept_evidence.append(item)
            continue
        if identity in suppressed_ids:
            continue
        suppressed_ids.add(identity)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
    return kept_facts, kept_evidence, {
        "profile": profile.profile,
        "suppressed_count": len(suppressed_ids),
        "suppressed_reasons": reasons,
        "suppressed_examples": examples,
        "retain": list(profile.retain),
        "suppress": list(profile.suppress),
    }


def _empty_purpose_policy(profile: PurposeProfile) -> dict[str, Any]:
    return {
        "profile": profile.profile,
        "suppressed_count": 0,
        "suppressed_reasons": {},
        "suppressed_examples": [],
        "retain": list(profile.retain),
        "suppress": list(profile.suppress),
    }


def _accepted_state_diagnostics(selection: dict[str, Any]) -> dict[str, Any]:
    """Return bounded accepted-state selection diagnostics for checkout clients."""
    selected_citations = selection.get("selected_citations")
    return {
        "mode": str(selection.get("mode") or "coordinate_accepted_state"),
        "selected_count": int(selection.get("selected_count") or 0),
        "diagnostic_count": int(selection.get("diagnostic_count") or 0),
        "selected_citations": [
            citation
            for citation in selected_citations
            if isinstance(citation, str)
        ][:10]
        if isinstance(selected_citations, list)
        else [],
    }


def _purpose_suppression_reason(profile: PurposeProfile, item: dict[str, Any]) -> str | None:
    suppress = set(profile.suppress)
    status = _checkout_policy_status(item)
    authority = _checkout_policy_text(item.get("authority") or item.get("authority_scope"))
    if "worker_local_pending" in suppress and (
        status == "pending"
        or authority in {"worker", "worker-local", "worker_local", "pending"}
        or (authority.startswith("worker") and item.get("promoted") is False)
    ):
        return "worker_local_pending"
    if "pending_unreviewed_claim" in suppress and status == "pending":
        return "pending_unreviewed_claim"
    if "rejected_finding" in suppress and status in {"rejected", "unsupported"}:
        return "rejected_finding"
    if "stale_unpromoted_finding" in suppress and (
        bool(item.get("stale")) or status in {"stale", "superseded", "deprecated"}
    ) and authority not in {"accepted", "parent-accepted", "parent_accepted", "promoted"}:
        return "stale_unpromoted_finding"
    if "low_trust_inference" in suppress and _low_trust_inferred_item(item):
        return "low_trust_inference"
    if "superseded_context" in suppress and item.get("valid_to"):
        return "superseded_context"
    if "uncited_claim" in suppress and not item.get("citation"):
        return "uncited_claim"
    return None


def _checkout_policy_status(item: dict[str, Any]) -> str:
    return _checkout_policy_text(
        item.get("coordination_status")
        or item.get("finding_status")
        or item.get("status")
    )


def _checkout_policy_text(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "-")


def _low_trust_inferred_item(item: dict[str, Any]) -> bool:
    explanation = item.get("score_explanation")
    if not isinstance(explanation, dict):
        return False
    trust = explanation.get("inferred_edge_trust")
    return isinstance(trust, int | float) and not isinstance(trust, bool) and float(trust) < 0.7


def _checkout_policy_item_id(item: dict[str, Any]) -> str:
    for key in ("finding_id", "citation", "content"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _synthesis_packet_metadata(content: str, packet_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if packet_payload is not None:
        return {"synthesis_packet": packet_payload}
    packet = synthesis_packet_from_items([{"content": content}])
    if not packet.answer_candidates and not packet.ledger_rows:
        return {}
    return {
        "synthesis_packet": {
            "schema_version": "synthesis_packet_v1",
            "answer_candidates": packet.answer_candidates,
            "ledger_rows": packet.ledger_rows,
            "content": content,
        }
    }


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


def _source_context_text(context: Context) -> str:
    """Return source text with compact metadata used by retrieval planning helpers."""
    metadata = context.metadata or {}
    prefixes: list[str] = []
    citation = metadata.get("citation")
    if citation:
        prefixes.append(f"citation={citation}")
    source_path = metadata.get("source_path")
    if source_path:
        prefixes.append(f"source_path={source_path}")
    event_thread = metadata.get("event_thread")
    if event_thread:
        prefixes.append(f"thread={event_thread}")
    source_kind = metadata.get("source_kind")
    if source_kind:
        prefixes.append(f"source_kind={source_kind}")
    if not prefixes:
        return context.content
    return " ".join([*prefixes, context.content])


def _unique_synthesis_context_texts(contexts: list[Context]) -> list[str]:
    """Return synthesis candidate text once while preserving rank order."""
    texts: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        text = _source_context_text(context)
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return texts


def _prefer_verbatim_for_duplicate_source_groups(
    source_contexts: list[Context],
    graph_contexts: list[Context],
) -> list[str]:
    """Return synthesis candidates while avoiding graph summaries over full source text."""
    source_groups = {
        source_context_group(_source_context_text(context))
        for context in source_contexts
    }
    contexts = [
        *source_contexts,
        *[
            context
            for context in graph_contexts
            if source_context_group(_source_context_text(context)) not in source_groups
        ],
    ]
    return _unique_synthesis_context_texts(contexts)


def _append_context_once(
    target: list[Context],
    context: Context,
    seen: set[tuple[str, str]],
) -> None:
    metadata = context.metadata or {}
    citation = str(metadata.get("citation") or "")
    key = (citation, context.content)
    if key in seen:
        return
    seen.add(key)
    target.append(context)


def _context_citation(context: Context) -> str | None:
    metadata = context.metadata or {}
    citation = metadata.get("citation")
    return citation if isinstance(citation, str) and citation else None


_REASONING_EVENT_CITATION_RE = re.compile(
    r"^eventloom://[^/\s]+/events/[1-9][0-9]*#(?:[0-9a-f]{12}|[0-9a-f]{64})$"
)
_CLAIM_NEGATION_TERMS = {
    "not",
    "never",
    "no",
    "none",
    "false",
    "refute",
    "refuted",
    "conflict",
    "conflicted",
    "contradict",
    "contradicted",
}


def _strict_reasoning_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict: list[dict[str, Any]] = []
    for item in evidence:
        citation = item.get("citation")
        if isinstance(citation, str) and _REASONING_EVENT_CITATION_RE.fullmatch(citation):
            strict.append(dict(item))
    return strict


def _causal_result_reasoning_evidence(item: dict[str, Any]) -> dict[str, Any]:
    raw_source = item.get("source")
    raw_target = item.get("target")
    raw_evidence = item.get("evidence")
    source = raw_source if isinstance(raw_source, dict) else {}
    target = raw_target if isinstance(raw_target, dict) else {}
    evidence = raw_evidence if isinstance(raw_evidence, dict) else {}
    content = evidence.get("summary") or (
        f"{source.get('name', 'unknown source')} {item.get('relation_type', 'related')} "
        f"{target.get('name', 'unknown target')}"
    )
    return {
        "citation": item.get("citation", ""),
        "content": str(content),
        "source": "causal_predecessor",
        "confidence": item.get("confidence"),
        "review_status": item.get("review_status"),
        "authority_status": item.get("authority_status"),
    }


def _checkout_reasoning_evidence(item: dict[str, Any]) -> dict[str, Any] | None:
    citation = item.get("citation")
    if not isinstance(citation, str):
        return None
    content = item.get("content") or item.get("summary") or item.get("text")
    evidence: dict[str, Any] = {
        "citation": citation,
        "content": str(content or ""),
        "source": str(item.get("source") or "checkout"),
    }
    for key in (
        "entity_name",
        "entity_type",
        "event_type",
        "event_seq",
        "event_hash",
        "source_event_seq",
        "source_event_hash",
        "authority_status",
        "authority",
        "review_status",
        "status",
        "stale",
        "superseded_by",
        "primitive",
        "phase",
    ):
        value = item.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            evidence[key] = value
    return evidence


def _score_claim_evidence(
    claim: str,
    evidence_items: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    claim_tokens = _tokens(claim)
    evidence: list[dict[str, Any]] = []
    support_count = 0
    conflict_count = 0
    for item in evidence_items:
        if not _eligible_claim_confidence_evidence(item):
            continue
        evidence_item = _checkout_reasoning_evidence(item)
        if evidence_item is None:
            continue
        content = str(evidence_item.get("content") or "")
        content_tokens = _tokens(content)
        overlap = claim_tokens & content_tokens
        if not overlap:
            continue
        label = "support"
        if _is_conflicting_claim_evidence(content_tokens, content):
            label = "conflict"
            conflict_count += 1
        else:
            support_count += 1
        evidence_item["stance"] = label
        evidence_item["matched_terms"] = sorted(overlap)
        evidence.append(evidence_item)
        if len(evidence) >= limit:
            break
    denominator = support_count + conflict_count
    confidence = support_count / denominator if denominator else 0.0
    return {
        "confidence": round(confidence, 4),
        "support_count": support_count,
        "conflict_count": conflict_count,
        "evidence": evidence,
    }


def _bounded_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("min_confidence must be a number between 0.0 and 1.0")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    return parsed


def _claim_key(claim: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", claim.casefold()).strip("-")[:80] or "claim"


def _source_events_from_reasoning_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_events: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in evidence:
        seq = item.get("source_event_seq") or item.get("event_seq")
        event_hash = item.get("source_event_hash") or item.get("event_hash")
        if not isinstance(seq, int) or not isinstance(event_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", event_hash):
            citation = item.get("citation")
            parsed_seq, parsed_hash = _citation_event_identity(citation if isinstance(citation, str) else None)
            seq = parsed_seq
            event_hash = parsed_hash
        if not isinstance(seq, int) or not isinstance(event_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", event_hash):
            continue
        key = (seq, event_hash)
        if key in seen:
            continue
        seen.add(key)
        source_events.append({"seq": seq, "hash": event_hash})
    return source_events


def _source_events_reasoning_evidence(session_id: str, source_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for source_event in source_events:
        seq = source_event.get("seq")
        event_hash = source_event.get("hash")
        if isinstance(seq, int) and isinstance(event_hash, str):
            evidence.append(
                {
                    "citation": f"eventloom://{session_id}/events/{seq}#{event_hash[:12]}",
                    "source_event_seq": seq,
                    "source_event_hash": event_hash,
                }
            )
    return evidence


def _metacognition_payloads_reasoning_evidence(
    session_id: str,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for payload in payloads:
        source_groups = [
            payload.get("source_events"),
            payload.get("supporting_source_events"),
            payload.get("conflicting_source_events"),
        ]
        for group in source_groups:
            if not isinstance(group, list):
                continue
            for item in _source_events_reasoning_evidence(session_id, group):
                seq = item.get("source_event_seq")
                event_hash = item.get("source_event_hash")
                if not isinstance(seq, int) or not isinstance(event_hash, str):
                    continue
                key = (seq, event_hash)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(item)
        payload_evidence = payload.get("evidence")
        evidence_items = payload_evidence if isinstance(payload_evidence, list) else []
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            citation = item.get("citation")
            if isinstance(citation, str) and citation.strip():
                evidence.append({"citation": citation.strip()})
    return evidence


def _reverification_needs_from_events(
    events: list[dict[str, Any]],
    *,
    query: str | None,
    limit: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    terms = _tokens(query or "")
    needs: list[dict[str, Any]] = []
    for event in events:
        payload = event["payload"]
        haystack = " ".join(str(payload.get(key) or "") for key in ("claim", "claim_key", "query", "question", "reason"))
        if terms and not (_tokens(haystack) & terms):
            continue
        event_type = event["event_type"]
        if event_type == "metacognition.unknown.recorded" and payload.get("status") == "open":
            needs.append({"reason": "known_unknown_open", "event_type": event_type, **dict(payload)})
        elif event_type == "metacognition.reverify.requested" and payload.get("status") == "open":
            needs.append({"reason": "reverify_request_open", "event_type": event_type, **dict(payload)})
        elif event_type == "metacognition.conflict.clustered" and payload.get("resolution_status") == "unresolved":
            needs.append({"reason": "conflict_unresolved", "event_type": event_type, **dict(payload)})
        elif event_type == "metacognition.confidence.assessed":
            confidence = payload.get("confidence")
            conflict_count = payload.get("conflict_count")
            if (
                isinstance(confidence, int | float)
                and not isinstance(confidence, bool)
                and float(confidence) < min_confidence
            ) or (isinstance(conflict_count, int) and conflict_count > 0) or payload.get("requires_reverify") is True:
                needs.append({"reason": "confidence_requires_reverify", "event_type": event_type, **dict(payload)})
        if len(needs) >= limit:
            break
    return needs


def _eligible_claim_confidence_evidence(item: dict[str, Any]) -> bool:
    event_type = str(item.get("event_type") or "").strip()
    entity_type = str(item.get("entity_type") or "").strip()
    if event_type in {
        "belief.update.proposed",
        "reasoning.primitive.called",
        "metacognition.unknown.recorded",
        "metacognition.confidence.assessed",
        "metacognition.conflict.clustered",
        "metacognition.reverify.requested",
    }:
        return False
    if entity_type in {
        "belief_update_proposal",
        "reasoning_primitive_observation",
        "known_unknown",
        "confidence_assessment",
        "conflict_cluster",
        "reverify_request",
    }:
        return False
    review_status = str(item.get("review_status") or item.get("status") or "").casefold().strip()
    if review_status in {"pending", "rejected", "deferred", "unsupported", "stale", "conflicted"}:
        return False
    if item.get("stale") is True:
        return False
    superseded_by = item.get("superseded_by")
    return not (isinstance(superseded_by, str) and superseded_by.strip())


def _is_conflicting_claim_evidence(content_tokens: set[str], content: str) -> bool:
    lowered = content.casefold()
    if "did not" in lowered or "does not" in lowered or "not caused" in lowered:
        return True
    return bool(content_tokens & _CLAIM_NEGATION_TERMS)


def _procedure_contexts(contexts: list[Context], *, limit: int) -> list[dict[str, Any]]:
    procedures: list[dict[str, Any]] = []
    for context in contexts:
        metadata = context.metadata or {}
        if not _is_procedure_context(context):
            continue
        if _excluded_procedure_candidate(context):
            continue
        citation = _context_citation(context)
        procedures.append(
            {
                "content": context.content,
                "source": context.source,
                "score": context.score,
                "citation": citation,
                "metadata": dict(metadata),
            }
        )
        if len(procedures) >= limit:
            break
    return procedures


def _is_procedure_context(context: Context) -> bool:
    metadata = context.metadata or {}
    source = context.source.casefold()
    candidate_type = str(metadata.get("candidate_type") or metadata.get("kind") or "").casefold()
    event_type = str(metadata.get("event_type") or "").casefold()
    content = context.content.casefold()
    is_procedure = (
        candidate_type == "procedure"
        or "procedure" in event_type
        or content.startswith("procedure:")
        or "procedure" in content.split()[:5]
    )
    is_skill_or_consolidation = (
        "skill" in source
        or "consolidation" in source
        or event_type.startswith("skill.")
        or candidate_type == "procedure"
    )
    return is_procedure and is_skill_or_consolidation


def _excluded_procedure_candidate(context: Context) -> bool:
    metadata = context.metadata or {}
    review_status = str(metadata.get("review_status") or "").casefold()
    if review_status in {"rejected", "stale", "conflicted"}:
        return True
    if metadata.get("stale") is True:
        return True
    if context.valid_to is not None:
        return True
    return bool(metadata.get("superseded_by"))


def _procedure_reasoning_evidence(procedure: dict[str, Any]) -> dict[str, Any] | None:
    citation = procedure.get("citation")
    if not isinstance(citation, str):
        return None
    return {
        "citation": citation,
        "content": str(procedure.get("content") or ""),
        "source": str(procedure.get("source") or "procedure"),
    }


def _contexts_as_of_seq(contexts: list[Context], as_of_seq: int) -> list[Context]:
    filtered = []
    for context in contexts:
        citation = _context_citation(context)
        seq, _event_hash = _citation_event_identity(citation)
        if seq is None or seq <= as_of_seq:
            filtered.append(context)
    return filtered


def _checkout_rank(context: Context, query: str) -> tuple[float, int, int, int, float, str, float]:
    query_tokens = _checkout_tokens(query)
    content_tokens = _checkout_tokens(context.content)
    overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
    metadata = context.metadata or {}
    entity_type = metadata.get("entity_type")
    type_priority = 1 if entity_type in {"task", "decision", "goal", "memory"} else 0
    citation_priority = 1 if _context_citation(context) else 0
    source_lane = _checkout_source_lane(context)
    source_priority = 1 if source_lane in {"verbatim", "eventloom", "projection"} else 0
    purpose_outcome_rank = _purpose_outcome_rank(metadata)
    return (
        overlap,
        citation_priority,
        source_priority,
        type_priority,
        purpose_outcome_rank,
        context.valid_from or "",
        context.score,
    )


def _checkout_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 2}


def _purpose_outcome_rank(metadata: dict[str, Any]) -> float:
    positive = _numeric_metadata(metadata.get("purpose_outcome_positive_count"))
    negative = _numeric_metadata(metadata.get("purpose_outcome_negative_count"))
    return max(-0.18, min(0.20, positive * 0.06 - negative * 0.06))


def _numeric_metadata(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


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
        "authority",
        "authority_scope",
        "coordination_status",
        "finding_id",
        "mission_id",
        "source_kind",
        "source_event_seq",
        "source_event_hash",
        "stale",
        "status",
        "worker_id",
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


def _eventlog_file_signature(eventlog: EventLog) -> tuple[int, int]:
    """Return a cheap invalidation signature for a local Eventloom log."""
    try:
        stat = eventlog.path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


def _consolidation_candidate_ids(events: list[Any]) -> set[str]:
    candidate_ids: set[str] = set()
    for event in events:
        if getattr(event, "type", None) != "consolidation.candidate.created":
            continue
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            continue
        candidate_id = payload.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            candidate_ids.add(candidate_id)
    return candidate_ids


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _tokens(value: str) -> set[str]:
    import re

    return {token for token in re.findall(r"[A-Za-z0-9]+", value.lower()) if len(token) > 1}
