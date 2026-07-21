"""Checkout, assembly, feedback, and synthesis operations for MemoryFabric (phase 3).

Extracted per ``docs/superpowers/specs/2026-07-06-fabric-decomposition-design.md``
following the phase-1/2 pattern: :class:`CheckoutOps` owns the checkout/
assembly/consolidation/feedback/synthesis cluster behind a structural
:class:`CheckoutHost` protocol, and ``MemoryFabric`` delegates.

Phase-3 seam notes (all late-bound, evidence-driven):

- ``build_memory_checkout`` is a patch-targeted fabric global
  (``patch("zaxy.core.fabric.build_memory_checkout")`` in tests), so the moved
  ``checkout_memory`` builds packets through the host's
  ``_build_memory_checkout`` seam, which resolves the name inside
  ``zaxy.core.fabric``.
- ``get_metrics`` likewise routes through ``host._record_degraded_operation``.
- Intra-cluster calls to *public* methods (``assemble_context`` from
  ``checkout_memory``/``after_turn``) route back through the host so existing
  instance patches (e.g. the reasoning tests' ``fabric.checkout_memory``)
  remain the single dispatch point.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, cast

from zaxy.context import Context, context_counts
from zaxy.core.checkout_build import (
    _apply_purpose_outcome_learning,
    _checkout_recall_limit,
    _checkout_source_id,
    _consolidation_candidate_ids,
    _context_feedback_metadata,
    _context_identity,
    _context_warnings,
    _contexts_as_of_seq,
    _drop_governance_withheld_contexts,
    _event_citation,
    _event_content,
    _feedback_outcome,
    _feedback_purpose_payload,
    _governance_withheld_event_seqs,
    _increment_count,
    _normalize_context_feedback,
    _purpose_outcome_aggregates,
)
from zaxy.core.models import ContextAssembly, MemoryCheckout
from zaxy.editable import MEMORY_ROLLBACK_EVENT_TYPE
from zaxy.long_horizon import build_long_horizon_plan
from zaxy.purpose import PurposeProfile, purpose_profile, purpose_retrieval_policy
from zaxy.recall import build_recall_candidate_set
from zaxy.refs import MemoryRef
from zaxy.salience import (
    build_confirmed_reinforcement_event,
    build_surfaced_reinforcement_event,
    event_ref_index,
    reinforcement_targets_from_citations,
)
from zaxy.security import MAX_QUERY_LIMIT, validate_limit, validate_payload, validate_session_id
from zaxy.synthesis_artifact import (
    build_synthesis_artifact,
    build_synthesis_candidate_event_payload,
    build_synthesis_evidence_event_payload,
    normalize_synthesis_outcome,
    synthesis_outcome_event_type,
)
from zaxy.working_set import build_working_set, format_working_set

__all__ = ["CheckoutHost", "CheckoutOps"]


class CheckoutHost(Protocol):
    """The exact fabric surface the checkout cluster depends on."""

    # Runtime attributes (read; ``_connected`` is also written on degrade).
    session_manager: Any
    settings: Any
    context_assembly_policy: Any
    retrieval_profile: Any
    refs: Any
    _salience_floor: Any
    _salience_half_life_days: Any
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

    async def query(
        self,
        query: str,
        *,
        limit: int = ...,
        session_id: str = ...,
        include_source_lane: bool = ...,
        scoring_profile: Any = ...,
        cues: dict[str, str] | None = ...,
    ) -> list[Any]: ...

    async def query_verbatim(
        self, query: str, *, limit: int = ..., session_id: str = ...
    ) -> list[Any]: ...

    async def assemble_context(
        self,
        query: str,
        *,
        session_id: str = ...,
        replay_from_seq: int = ...,
        limit: int = ...,
        recall_limit: int | None = ...,
        max_recent_events: int | None = ...,
        as_of_seq: int | None = ...,
        purpose: Any = ...,
        cues: dict[str, str] | None = ...,
        fleet_ids: list[str] | None = ...,
        agent_id: str | None = ...,
        long_horizon: bool | None = ...,
    ) -> ContextAssembly: ...

    def _fleet_lane_contexts(
        self, fleet_ids: list[str] | None, *, agent_id: str
    ) -> list[Context]: ...

    async def _append_event_spec(self, event: dict[str, Any], *, session_id: str) -> Any: ...

    async def _project_event(self, event: Any, *, session_id: str) -> None: ...

    async def _append_generated_inferences(
        self, eventlog: Any, *, source_event: Any, session_id: str
    ) -> None: ...

    def _decrypt_event_view(self, event: Any) -> Any: ...

    def _recent_packet_memory_contexts(self, replay_events: list[Any]) -> list[Any]: ...

    def _session_event_ref_index(self, session_id: str) -> Any: ...

    def _invalidate_query_page_cache(self, session_id: str) -> None: ...

    def _build_memory_checkout(self, **kwargs: Any) -> MemoryCheckout: ...

    def _record_degraded_operation(self, operation: str, reason: str) -> None: ...


class CheckoutOps:
    """Checkout, context assembly, consolidation, feedback, and synthesis.

    Method bodies are moved verbatim from ``MemoryFabric``; only the ``self``
    surface was renamed to the injected host.
    """

    def __init__(self, *, host: CheckoutHost) -> None:
        self._host = host

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
        eventlog = self._host.session_manager.get(sid).eventlog
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
            await self._host._project_event(event, session_id=sid)
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
        replay = await self._host.replay(session_id=sid)

        candidates: dict[str, dict[str, Any]] = {}
        reviews_by_candidate: dict[str, list[dict[str, Any]]] = {}
        rolled_back_targets: set[tuple[int, str]] = set()
        review_count = 0
        duplicate_candidate_count = 0
        rollback_count = 0
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
                    reviews_by_candidate.setdefault(candidate_id, []).append(
                        {
                            "seq": event.seq,
                            "hash": event.hash,
                            "status": event.payload.get("status"),
                        }
                    )
            elif event.type == MEMORY_ROLLBACK_EVENT_TYPE:
                target = event.payload.get("target")
                if isinstance(target, dict):
                    target_seq = target.get("seq")
                    target_hash = target.get("hash")
                    if isinstance(target_seq, int) and isinstance(target_hash, str):
                        rolled_back_targets.add((target_seq, target_hash))
                        rollback_count += 1

        # Honor reversals: a memory.rolled_back citing a review event undoes that
        # acceptance on replay, reverting the candidate to its prior effective
        # review status (the latest surviving review, else the created default).
        for candidate_id, candidate in candidates.items():
            rolled_back_reviews = 0
            for review in reviews_by_candidate.get(candidate_id, []):
                if (review["seq"], review["hash"]) in rolled_back_targets:
                    rolled_back_reviews += 1
                    continue
                status = review["status"]
                if status is not None:
                    candidate["review_status"] = status
                candidate["authority_status"] = "non_authoritative"
                candidate["reviewed_seq"] = review["seq"]
                candidate["reviewed_hash"] = review["hash"]
            if rolled_back_reviews:
                candidate["rolled_back_review_count"] = rolled_back_reviews

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
            "rollback_count": rollback_count,
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
        fleet_ids: list[str] | None = None,
        agent_id: str | None = None,
        long_horizon: bool | None = None,
    ) -> ContextAssembly:
        """Assemble recent replay plus retrieval into prompt-ready context.

        ``cues`` is additive and only affects retrieval under the cognitive
        retrieval profile (see ``MemoryFabric.query``).

        ``long_horizon`` engages the two-tier (episodic recent + consolidated
        remote) assembly. ``None`` falls back to ``long_horizon_enabled``;
        ``False`` forces single-tier (byte-identical) assembly. When engaged and
        the session exceeds ``long_horizon_recent_window``, older history is
        surfaced as cited, non-authoritative consolidation candidates.
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
        replay = await self._host.replay(from_seq=replay_from_seq, session_id=sid)
        graph_contexts = await self._host.query(
            retrieval_query,
            limit=candidate_limit,
            session_id=sid,
            include_source_lane=False,
            scoring_profile=retrieval_policy.scoring_profile,
            cues=cues,
        )
        verbatim_candidate_limit = self._host.context_assembly_policy.verbatim_candidate_limit(
            query=retrieval_query,
            limit=candidate_limit,
        )
        verbatim_contexts = (
            await self._host.query_verbatim(
                retrieval_query, limit=verbatim_candidate_limit, session_id=sid
            )
            if verbatim_candidate_limit > 0
            else []
        )
        replay_events = [self._host._decrypt_event_view(event) for event in replay.events]
        if as_of_seq is not None:
            replay_events = [event for event in replay_events if event.seq <= as_of_seq]
        session_events = list(replay_events)
        purpose_outcomes = _purpose_outcome_aggregates(replay_events, profile)
        graph_contexts = _apply_purpose_outcome_learning(graph_contexts, purpose_outcomes)
        verbatim_contexts = _apply_purpose_outcome_learning(verbatim_contexts, purpose_outcomes)
        packet_memory_contexts = self._host._recent_packet_memory_contexts(replay_events)
        packet_memory_contexts = _apply_purpose_outcome_learning(packet_memory_contexts, purpose_outcomes)
        # A rule the evolution gate held for review must not be assembled into the
        # prompt on any lane. Applied here, at the point every lane is materialised,
        # so the exclusion covers the retrieved-context prompt, the working set, and
        # the recall candidate set together rather than only the checkout payload.
        # The withheld events themselves stay in the replay/audit trail.
        withheld_seqs = _governance_withheld_event_seqs(session_events)
        if withheld_seqs:
            graph_contexts = _drop_governance_withheld_contexts(graph_contexts, withheld_seqs)
            verbatim_contexts = _drop_governance_withheld_contexts(verbatim_contexts, withheld_seqs)
            packet_memory_contexts = _drop_governance_withheld_contexts(
                packet_memory_contexts, withheld_seqs
            )
        recall_contexts = [*graph_contexts, *verbatim_contexts, *packet_memory_contexts]
        recall = build_recall_candidate_set(recall_contexts, budget=candidate_limit)
        contexts = self._host.context_assembly_policy.assemble(
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
        fleet_contexts = self._host._fleet_lane_contexts(fleet_ids, agent_id=agent_id or sid)
        if fleet_contexts:
            contexts = [*contexts, *fleet_contexts]
        long_horizon_engaged = (
            getattr(self._host.settings, "long_horizon_enabled", False)
            if long_horizon is None
            else long_horizon
        )
        long_horizon_contexts: list[Context] = []
        long_horizon_summary: dict[str, Any] | None = None
        if long_horizon_engaged:
            plan = build_long_horizon_plan(
                session_events,
                session_id=sid,
                recent_window=max(
                    getattr(self._host.settings, "long_horizon_recent_window", 50),
                    max_recent_events or 0,
                ),
                budget=prompt_limit,
            )
            long_horizon_summary = plan.to_diagnostics()
            long_horizon_contexts = plan.consolidated_contexts
            if long_horizon_contexts:
                contexts = [*contexts, *long_horizon_contexts]
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
        working_set_payload["retrieval_profile"] = self._host.retrieval_profile.to_diagnostics()
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
            assembly_policy=self._host.context_assembly_policy.describe(),
            context_counts=context_counts(contexts, replay_count=len(replay_events)),
            working_set=working_set_payload,
            recall=recall,
            replay_events=session_events,
            fleet_contexts=fleet_contexts,
            long_horizon_contexts=long_horizon_contexts,
            long_horizon=long_horizon_summary,
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

        ``record_reinforcement=False`` skips the best-effort 'surfaced'
        salience reinforcement append for read-only inspection surfaces
        (e.g. the dashboard) that must not write to the log.

        ``cues`` (optional, additive) carries the caller's
        encoding-specificity context; it only affects ranking under the
        cognitive retrieval profile.

        ``long_horizon`` engages the two-tier (episodic + consolidated) assembly
        (``None`` -> ``long_horizon_enabled``; ``False`` forces single-tier).
        """
        resolved_ref = self._resolve_checkout_ref(ref, session_id=session_id)
        checkout_session_id = resolved_ref.session_id if resolved_ref is not None else session_id
        as_of_seq = resolved_ref.target_seq if resolved_ref is not None else None
        assembly = await self._host.assemble_context(
            query,
            session_id=checkout_session_id,
            replay_from_seq=replay_from_seq,
            limit=limit,
            recall_limit=_checkout_recall_limit(query, limit),
            max_recent_events=max_recent_events,
            as_of_seq=as_of_seq,
            purpose=purpose,
            cues=cues,
            fleet_ids=fleet_ids,
            agent_id=agent_id,
            long_horizon=long_horizon,
        )
        checkout = self._host._build_memory_checkout(
            query=query,
            assembly=assembly,
            ref=resolved_ref,
            purpose=purpose,
            now=datetime.now(UTC),
            retrieval_profile=self._host.retrieval_profile,
            cues=cues,
            salience_floor=self._host._salience_floor,
            salience_half_life_days=self._host._salience_half_life_days,
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
            await self._host._append_event_spec(spec, session_id=session_id)
        except Exception:
            self._host._record_degraded_operation("append", "salience_reinforcement_unavailable")

    def _resolve_checkout_ref(self, ref: str | None, *, session_id: str) -> MemoryRef | None:
        if ref is None:
            return None
        if ref == "HEAD":
            latest = self._host.session_manager.get(session_id).eventlog.last_event()
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
        resolved = self._host.refs.resolve(ref)
        if resolved is None:
            raise ValueError(f"Unknown memory ref: {ref}")
        return cast(MemoryRef, resolved)

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
                feedback_event = await self._host.append(
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
                await self._host.append(
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
            index = self._host._session_event_ref_index(session_id)
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
            await self._host._append_event_spec(spec, session_id=session_id)
        except Exception:
            self._host._record_degraded_operation("append", "salience_reinforcement_unavailable")

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
        if not self._host._connected:
            try:
                await self._host.connect()
            except Exception:
                self._host._record_degraded_operation("append", "graph_connect_unavailable")
                self._host._connected = False
        eventlog = self._host.session_manager.get(sid).eventlog
        event = eventlog.append(
            synthesis_outcome_event_type(normalized),
            actor=actor,
            payload=validate_payload(payload),
            thread=sid,
        )
        await self._host._project_event(event, session_id=sid)
        await self._host._append_generated_inferences(eventlog, source_event=event, session_id=sid)
        self._host._invalidate_query_page_cache(sid)
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
        if not self._host._connected:
            try:
                await self._host.connect()
            except Exception:
                self._host._record_degraded_operation("append", "graph_connect_unavailable")
                self._host._connected = False
        eventlog = self._host.session_manager.get(sid).eventlog
        event_type = "memory.evidence.reinforced" if normalized == "used" else synthesis_outcome_event_type(normalized)
        event = eventlog.append(
            event_type,
            actor=actor,
            payload=validate_payload(payload),
            thread=sid,
        )
        await self._host._project_event(event, session_id=sid)
        await self._host._append_generated_inferences(eventlog, source_event=event, session_id=sid)
        self._host._invalidate_query_page_cache(sid)
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
        if not self._host._connected:
            try:
                await self._host.connect()
            except Exception:
                self._host._record_degraded_operation("append", "graph_connect_unavailable")
                self._host._connected = False
        eventlog = self._host.session_manager.get(sid).eventlog
        event = eventlog.append(
            "memory.synthesis.artifact.created",
            actor=actor,
            payload=validate_payload(payload),
            thread=sid,
        )
        await self._host._project_event(event, session_id=sid)
        await self._host._append_generated_inferences(eventlog, source_event=event, session_id=sid)
        self._host._invalidate_query_page_cache(sid)
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
        await self._host.append(
            "transcript.turn",
            actor=role,
            payload={"role": role, "content": content, "source": source},
            session_id=sid,
        )
        return await self._host.assemble_context(
            query or content,
            session_id=sid,
            replay_from_seq=1,
            limit=limit,
            max_recent_events=max_recent_events,
        )
