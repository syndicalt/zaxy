"""Reasoning and metacognition primitives for MemoryFabric (decomposition phase 1).

Extracted per ``docs/superpowers/specs/2026-07-06-fabric-decomposition-design.md``:
:class:`ReasoningOps` owns the reasoning/metacognition cluster behind a single
narrow, structural :class:`ReasoningHost` protocol, and ``MemoryFabric``
delegates its public reasoning methods here.

Every host access is **late-bound**: the protocol is looked up on the fabric
instance at call time, never captured at construction. This is a behavioral
requirement, not a style choice — existing tests monkeypatch fabric instance
attributes (``checkout_memory``, ``query``, even ``query_causal_predecessors``)
after construction, and the projection store (``host.graph``) can be swapped at
runtime by the graph-degraded fallback. For the same reason, calls between
*public* reasoning primitives route back through the host (fabric delegations)
rather than short-circuiting internally, exactly mirroring the pre-extraction
dynamic dispatch. This module deliberately references none of the fabric
module's patch-targeted globals.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast

from zaxy.causal import (
    CausalQueryResult,
    causal_query_result_from_projection,
    causal_relation_to_graph_relation,
)
from zaxy.core.checkout_build import (
    _bounded_threshold,
    _causal_result_reasoning_evidence,
    _checkout_reasoning_evidence,
    _claim_key,
    _metacognition_payloads_reasoning_evidence,
    _procedure_reasoning_evidence,
    _reverification_needs_from_events,
    _score_claim_evidence,
    _source_events_from_reasoning_evidence,
    _source_events_reasoning_evidence,
    _strict_reasoning_evidence,
)
from zaxy.metacognition import (
    build_confidence_assessment_event,
    build_conflict_cluster_event,
    build_known_unknown_event,
    build_reverify_request_event,
    summarize_metacognition_events,
)
from zaxy.procedural_planning import classify_procedure_contexts
from zaxy.purpose import purpose_retrieval_policy
from zaxy.reasoning_primitives import (
    ReasoningPrimitiveCall,
    build_belief_update_proposal_event,
    phase_purpose_profile,
    validate_reasoning_phase,
)
from zaxy.security import (
    MAX_QUERY_LIMIT,
    validate_limit,
    validate_query,
    validate_session_id,
    validate_traversal_depth,
)

__all__ = ["ReasoningHost", "ReasoningOps"]


class ReasoningHost(Protocol):
    """The exact fabric surface the reasoning cluster depends on.

    Structural (no fabric import): ``MemoryFabric`` satisfies it at the
    ``ReasoningOps(host=self)`` construction site, where mypy verifies the
    contract. Methods that fabric implements by delegating back to
    :class:`ReasoningOps` (``query_causal_predecessors``,
    ``retrieve_similar_procedures``) are declared here on purpose — routing
    those calls through the host keeps instance-level test patches and
    runtime component swaps intercepting, as they did pre-extraction.
    """

    # Runtime-swappable components (looked up per call, never cached here).
    graph: Any
    session_manager: Any

    async def append(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None = ...,
        thread: str = ...,
        session_id: str | None = ...,
        *,
        forgettable: bool = ...,
    ) -> Any: ...

    async def _append_event_spec(self, event: dict[str, Any], *, session_id: str) -> Any: ...

    async def checkout_memory(
        self,
        query: str,
        *,
        session_id: str = ...,
        limit: int = ...,
        purpose: Any = ...,
    ) -> Any: ...

    async def query(
        self,
        query: str,
        *,
        session_id: str = ...,
        limit: int = ...,
        include_source_lane: bool = ...,
        scoring_profile: Any = ...,
    ) -> list[Any]: ...

    async def query_causal_predecessors(
        self,
        entity_name: str,
        *,
        relation_type: str | None = ...,
        depth: int = ...,
        temporal_point: str | None = ...,
        session_id: str = ...,
    ) -> list[CausalQueryResult]: ...

    async def retrieve_similar_procedures(
        self,
        query: str,
        *,
        phase: str = ...,
        session_id: str = ...,
        limit: int = ...,
    ) -> dict[str, Any]: ...


class ReasoningOps:
    """Reasoning primitives: causal queries, claim confidence, metacognition.

    Method bodies are moved verbatim from ``MemoryFabric`` (zero behavior
    change); only the ``self`` surface was renamed to the injected host.
    """

    def __init__(self, *, host: ReasoningHost) -> None:
        self._host = host

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
            causal_results = await self._host.query_causal_predecessors(
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
                checkout = await self._host.checkout_memory(
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
            await self._host.append(
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
            checkout = await self._host.checkout_memory(
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
            contexts = await self._host.query(
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
        await self._host._append_event_spec(event, session_id=sid)
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
        result = await self._host.retrieve_similar_procedures(
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

    # -- internal helpers (no external callers; verified before the move) ----

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
        assessment_event = await self._host._append_event_spec(assessment, session_id=session_id)
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
                await self._host._append_event_spec(cluster, session_id=session_id)
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
            await self._host._append_event_spec(reverify, session_id=session_id)

    def _metacognition_event_specs(self, session_id: str) -> list[dict[str, Any]]:
        replayed = self._host.session_manager.get(session_id).eventlog.read_all()
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
        await self._host.append(
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
        neighbors = await self._host.graph.search_causal_neighbors(
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
