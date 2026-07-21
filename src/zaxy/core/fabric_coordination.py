"""Coordination, fleet-lane, and handoff operations for MemoryFabric (phase 2).

Extracted per ``docs/superpowers/specs/2026-07-06-fabric-decomposition-design.md``
following the phase-1 (``fabric_reasoning``) pattern: :class:`CoordinationOps`
owns the coordination/fleet/handoff cluster behind a structural
:class:`CoordinationHost` protocol, and ``MemoryFabric`` delegates.

Late-binding rules carried over from phase 1, with two phase-2 additions:

- ``tests/test_fleet_surface.py`` instance-patches ``fabric._fleet_manager``
  and then drives the fleet lane, so the lane resolves the manager through the
  host (``self._host._fleet_manager()``), never internally.
- ``get_metrics`` is one of the fabric module's patch-targeted globals, so
  moved call sites report degraded operations through the host's
  ``_record_degraded_operation`` seam, which resolves ``get_metrics`` inside
  ``zaxy.core.fabric`` where existing patches intercept.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from zaxy.context import Context
from zaxy.core.models import HandoffBundle, MemoryCheckout
from zaxy.lifecycle import build_subagent_completed_event
from zaxy.security import validate_payload, validate_session_id
from zaxy.synthesis_artifact import build_synthesis_artifact

__all__ = ["CoordinationHost", "CoordinationOps"]


class CoordinationHost(Protocol):
    """The exact fabric surface the coordination cluster depends on."""

    # Runtime attributes (read; ``_connected`` is also written on degrade).
    session_manager: Any
    eventloom_path: Any
    settings: Any
    _connected: bool

    async def connect(self) -> None: ...

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

    async def replay(self, from_seq: int = ..., session_id: str = ...) -> Any: ...

    async def assemble_context(
        self,
        query: str,
        *,
        session_id: str = ...,
        replay_from_seq: int = ...,
        limit: int = ...,
        max_recent_events: int | None = ...,
    ) -> Any: ...

    async def handoff_bundle(
        self,
        *,
        session_id: str = ...,
        query: str = ...,
        replay_from_seq: int = ...,
        limit: int = ...,
    ) -> HandoffBundle: ...

    def _fleet_manager(self) -> Any: ...

    def _coordination_manager(self) -> Any: ...

    async def _project_event(self, event: Any, *, session_id: str) -> None: ...

    async def _append_generated_inferences(
        self,
        eventlog: Any,
        *,
        source_event: Any,
        session_id: str,
    ) -> None: ...

    def _record_degraded_operation(self, operation: str, reason: str) -> None: ...


class CoordinationOps:
    """Coordination missions, the fleet checkout lane, and session handoffs.

    Method bodies are moved verbatim from ``MemoryFabric``; only the ``self``
    surface was renamed to the injected host.
    """

    def __init__(self, *, host: CoordinationHost) -> None:
        self._host = host

    # -- fleet plane ------------------------------------------------------

    def fleet_manager(self) -> Any:
        """Return a FleetManager bound to this fabric's session manager.

        Twin of the MCP server's ``_coordination_manager``: the fleet plane
        replays the same Eventloom store this fabric reads, so checkout's fleet
        lane sees exactly the governed state the ``fleet_*`` tools wrote.
        """
        from zaxy.fleet import FleetManager

        manager = FleetManager(
            eventloom_path=self._host.eventloom_path, settings=self._host.settings
        )
        manager.session_manager = self._host.session_manager
        return manager

    def fleet_lane_contexts(
        self, fleet_ids: list[str] | None, *, agent_id: str
    ) -> list[Context]:
        """Resolve the enrollment-gated, cited, non-authoritative fleet lane.

        Returns nothing unless ``fleet_enabled`` is on and ``fleet_ids`` are
        requested (default-off: byte-identical checkout otherwise). For each
        requested fleet the agent must be enrolled at a tier above ``untrusted``
        (a non-enrolled or ``untrusted`` agent never receives a fleet's promoted
        memory); only ``active`` promotions whose visibility scope reaches the
        fleet are surfaced, each carrying its Eventloom citation, fleet
        provenance, and a ``fleet`` source-lane marker.
        """
        if not getattr(self._host.settings, "fleet_enabled", False) or not fleet_ids:
            return []
        from zaxy.fleet import fleet_thread

        manager = self._host._fleet_manager()
        contexts: list[Context] = []
        seen: set[str] = set()
        for fleet_id in fleet_ids:
            try:
                brief = manager.fleet_brief(fleet_id)
            except Exception:
                self._host._record_degraded_operation("query", "fleet_lane_unavailable")
                continue
            tier = next(
                (agent.trust_tier for agent in brief.agents if agent.agent_id == agent_id),
                None,
            )
            if tier is None or tier == "untrusted":
                # Not enrolled, or sandboxed/untrusted: never receives fleet memory.
                continue
            thread = fleet_thread(fleet_id)
            for memory in brief.active_promotions:
                if memory.visibility_scope not in ("fleet", "global"):
                    continue
                if memory.promotion_id in seen:
                    continue
                seen.add(memory.promotion_id)
                contexts.append(self._fleet_memory_context(memory, thread=thread))
        return contexts

    @staticmethod
    def _fleet_memory_context(memory: Any, *, thread: str) -> Context:
        """Project one active fleet memory into a cited, non-authoritative Context."""
        citation = f"eventloom://{thread}/events/{memory.event_seq}#{memory.event_hash[:12]}"
        confidence = memory.confidence if isinstance(memory.confidence, int | float) else 0.5
        return Context(
            content=memory.summary or memory.promotion_id,
            source="fleet",
            score=float(confidence),
            valid_from=memory.timestamp or None,
            metadata={
                "assembly_lane": "fleet",
                "source_lane": "fleet",
                "citation": citation,
                "non_authoritative": True,
                "authority_status": "non_authoritative",
                "fleet_id": memory.fleet_id,
                "promotion_id": memory.promotion_id,
                "kind": memory.kind,
                "review_status": memory.review_status,
                "visibility_scope": memory.visibility_scope,
                "keystone": memory.keystone,
                "origin_actor": memory.origin_actor,
                "origin_session": memory.origin_session,
                "entity_type": "fleet_promotion",
                "entity_name": memory.promotion_id,
            },
        )

    # -- handoff -----------------------------------------------------------

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
        summary = self._host.session_manager.handoff_summary(sid)
        replay = await self._host.replay(from_seq=replay_from_seq, session_id=sid)
        assembly = await self._host.assemble_context(
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
        await self._host.append(
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
        await self._host.append(
            event["event_type"],
            actor=event["actor"],
            payload=event["payload"],
            session_id=subagent_sid,
        )
        return await self._host.handoff_bundle(
            session_id=subagent_sid,
            query=query,
            replay_from_seq=1,
            limit=limit,
        )

    async def handoff_summary(self, session_id: str = "default") -> dict[str, Any]:
        """Generate a concise handoff summary from the event log.

        Suitable for resuming an agent session across restarts.
        """
        return cast(dict[str, Any], self._host.session_manager.handoff_summary(session_id))

    # -- coordination missions ----------------------------------------------

    def coordination_manager(self) -> Any:
        """Return a coordination manager bound to this fabric's session manager."""
        from zaxy.coordination import CoordinationManager
        from zaxy.coordination_semantic import build_semantic_conflict_detector

        manager = CoordinationManager(
            eventloom_path=self._host.eventloom_path,
            semantic_conflict_detector=build_semantic_conflict_detector(self._host.settings),
        )
        manager.session_manager = self._host.session_manager
        return manager

    async def coordinate_start_mission(
        self,
        mission_id: str,
        *,
        objective: str,
        actor: str = "coordinator",
    ) -> Any:
        """Start a parent coordination mission and project it."""
        result = self._host._coordination_manager().start_mission(mission_id, objective=objective, actor=actor)
        await self._host._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_create_worker(
        self,
        mission_id: str,
        worker_id: str,
        *,
        actor: str = "coordinator",
    ) -> Any:
        """Register a worker session under a parent mission and project it."""
        result = self._host._coordination_manager().create_worker(mission_id, worker_id, actor=actor)
        await self._host._project_event(result.event, session_id=result.mission_id)
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
        result = self._host._coordination_manager().assign(mission_id, worker_id, assignment, actor=actor)
        await self._host._project_event(result.event, session_id=result.mission_id)
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
        result = self._host._coordination_manager().report_finding(
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
        await self._host._project_event(result.event, session_id=result.worker_id or worker_id)
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
        result = self._host._coordination_manager().review_finding(
            mission_id,
            finding_id,
            status=status,
            actor=actor,
            rationale=rationale,
        )
        await self._host._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_promote_finding(
        self,
        mission_id: str,
        finding_id: str,
        *,
        actor: str = "coordinator",
        force: bool = False,
    ) -> Any:
        """Promote a finding into the parent mission history and project it."""
        result = self._host._coordination_manager().promote_finding(
            mission_id, finding_id, actor=actor, force=force
        )
        await self._host._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_brief(self, mission_id: str) -> Any:
        """Return a replay-backed coordination brief."""
        return self._host._coordination_manager().brief(mission_id)

    async def coordinate_checkout(self, mission_id: str, *, include_diagnostics: bool = False) -> Any:
        """Return accepted coordination state for prompt injection."""
        return self._host._coordination_manager().checkout(mission_id, include_diagnostics=include_diagnostics)

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
        proof_packet = self._host._coordination_manager().proof_packet(
            mission_sid,
            artifact_payload,
            decision_scope=decision_scope,
            handoff_id=handoff_id,
        )
        proof_payload = validate_payload(proof_packet.to_dict())
        if not self._host._connected:
            try:
                await self._host.connect()
            except Exception:
                self._host._record_degraded_operation("append", "graph_connect_unavailable")
                self._host._connected = False
        eventlog = self._host.session_manager.get(mission_sid).eventlog
        artifact_event = eventlog.append(
            "memory.synthesis.artifact.created",
            actor=actor,
            payload=validate_payload(artifact_payload),
            thread=mission_sid,
        )
        await self._host._project_event(artifact_event, session_id=mission_sid)
        await self._host._append_generated_inferences(
            eventlog, source_event=artifact_event, session_id=mission_sid
        )
        proof_event = eventlog.append(
            "coordination.proof_packet.created",
            actor=actor,
            payload=proof_payload,
            thread=mission_sid,
        )
        await self._host._project_event(proof_event, session_id=mission_sid)
        await self._host._append_generated_inferences(
            eventlog, source_event=proof_event, session_id=mission_sid
        )
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
        return self._host._coordination_manager().performance_ledger(mission_id)

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
        result = self._host._coordination_manager().create_handoff(
            mission_id,
            summary=summary,
            actor=actor,
            next_steps=next_steps,
            risks=risks,
        )
        await self._host._project_event(result.event, session_id=result.mission_id)
        return result

    async def coordinate_approval_packet(self, mission_id: str) -> Any:
        """Return a portable remote approval packet for pending coordination findings."""
        return self._host._coordination_manager().approval_packet(mission_id)

    async def coordinate_apply_approval_decisions(
        self,
        mission_id: str,
        decisions: list[dict[str, Any]],
        *,
        actor: str = "coordinator",
    ) -> Any:
        """Apply remote approval decisions and project all resulting events."""
        result = self._host._coordination_manager().apply_approval_decisions(
            mission_id,
            decisions,
            actor=actor,
        )
        for event in result.events:
            await self._host._project_event(event, session_id=result.mission_id)
        return result

    async def coordinate_record_detected_conflicts(
        self,
        mission_id: str,
        *,
        actor: str = "zaxy",
    ) -> Any:
        """Materialize deterministic coordination conflicts and project them."""
        results = self._host._coordination_manager().record_detected_conflicts(
            mission_id,
            actor=actor,
        )
        for result in results:
            await self._host._project_event(result.event, session_id=result.mission_id)
        return results
