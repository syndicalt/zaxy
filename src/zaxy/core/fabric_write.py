"""Write-path operations for MemoryFabric (decomposition phase 5 — the hub).

Extracted per ``docs/superpowers/specs/2026-07-06-fabric-decomposition-design.md``
following the phase-1..4 pattern: :class:`WriteEngine` owns the append/
evolution/edit/forget/ingest cluster behind a structural :class:`WriteHost`
protocol, and ``MemoryFabric`` delegates. Bodies are moved verbatim via an
automated slice+rename; every fabric state lookup late-binds through the host.

Phase-5 seam notes:

- ``fabric.append``, ``fabric._project_event`` and
  ``fabric._append_generated_inferences`` are instance-patched by tests, so
  every moved call to them routes via the host (fabric keeps delegations as
  the single dispatch points).
- ``get_metrics`` (patch-targeted fabric global) is reached only through the
  host's ``_metrics()`` seam, resolving inside ``zaxy.core.fabric``.
- ``ForgetTombstoneUnauditedError`` and the producer-ref helpers moved here;
  the fabric re-exports them (tests import the error from
  ``zaxy.core.fabric``).
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from zaxy.codebase import collect_codebase_events
from zaxy.context_refresh import (
    ContextRefreshPlan,
    load_refresh_state,
    plan_context_refresh,
    save_refresh_state,
)
from zaxy.core.checkout_build import (
    _citation_event_identity,
    _conflicting_property_value,
    _encoding_classification_content,
    _encoding_gate_eligible,
    _encoding_tokens,
    _event_citation,
    _payload_entity_names,
    _payloads_by_seq,
    _token_jaccard,
)
from zaxy.core.models import ContextRefreshReport
from zaxy.documents import collect_document_events
from zaxy.editable import (
    ROLLBACKABLE_EVENT_TYPES,
    build_memory_correction_event,
    build_memory_rollback_event,
)
from zaxy.embedding import embed_extraction
from zaxy.event import EventLog
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
from zaxy.outcome_learning import (
    build_outcome_event,
    build_rule_event,
    prediction_error,
    preventive_rule_confidence,
    validate_outcome,
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
from zaxy.security import validate_event_text, validate_payload, validate_session_id
from zaxy.transcripts import collect_transcript_events

__all__ = [
    "PRODUCER_REF_PAYLOAD_KEY",
    "ForgetTombstoneUnauditedError",
    "WriteEngine",
    "WriteHost",
]


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


class WriteHost(Protocol):
    """The exact fabric surface the write cluster depends on."""

    session_manager: Any
    settings: Any
    graph: Any
    tracer: Any
    embedding_provider: Any
    eventloom_path: Any
    retrieval_profile: Any
    _connected: bool
    _encoding_gate_enabled: bool
    _salience_floor: Any
    _salience_half_life_days: Any

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

    async def evaluate_evolution_gate(
        self,
        op: str,
        confidence: float,
        *,
        candidate_ref: dict[str, Any] | None = ...,
        actor: str = ...,
        session_id: str | None = ...,
    ) -> EvolutionGateDecision: ...

    async def propose_belief_update(
        self,
        claim: str,
        *,
        rationale: str,
        confidence: float,
        source_events: list[dict[str, Any]],
        phase: str = ...,
        session_id: str = ...,
        actor: str = ...,
    ) -> dict[str, Any]: ...

    async def _project_event(self, event: Any, *, session_id: str) -> None: ...

    async def _append_generated_inferences(
        self, eventlog: Any, *, source_event: Any, session_id: str
    ) -> None: ...

    async def _append_event_spec(self, event: dict[str, Any], *, session_id: str) -> Any: ...

    def _invalidate_query_page_cache(self, session_id: str) -> None: ...

    def _verbatim_index(self, session_id: str) -> Any: ...

    def _session_event_ref_index(self, session_id: str) -> Any: ...

    def _metrics(self) -> Any: ...


class WriteEngine:
    """Append/write path, governed evolution, editability, forgetting, ingest.

    Method bodies are moved verbatim (automated slice) from ``MemoryFabric``;
    only the ``self`` surface was renamed to the injected host.
    """

    def __init__(self, *, host: WriteHost) -> None:
        self._host = host

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
        if not self._host._connected:
            try:
                await self._host.connect()
            except Exception:
                self._host._metrics().record_degraded_operation("append", "graph_connect_unavailable")
                self._host._connected = False

        sid = validate_session_id(session_id or thread)
        safe_payload = validate_payload(payload or {})
        eventlog = self._host.session_manager.get(sid).eventlog

        if forgettable and not self._host.settings.forgetting_enabled:
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
            if encoding is not None and self._host._encoding_gate_enabled:
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

        await self._host._project_event(event, session_id=sid)
        await self._host._append_generated_inferences(eventlog, source_event=event, session_id=sid)
        self._host._invalidate_query_page_cache(sid)
        if (
            encoding is not None
            and encoding.classification == "redundant"
            and self._host._encoding_gate_enabled
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
        if not self._host._connected:
            try:
                await self._host.connect()
            except Exception:
                self._host._metrics().record_degraded_operation("append", "graph_connect_unavailable")
                self._host._connected = False

        sid = validate_session_id(session_id or "default")
        eventlog = self._host.session_manager.get(sid).eventlog

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
            await self._host._project_event(event, session_id=sid)
        self._host._invalidate_query_page_cache(sid)
        return cast(list[Any], events)

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
        policy = resolve_evolution_policy(self._host.settings)
        decision = evaluate_evolution_gate(op, confidence, policy=policy)
        spec = build_evolution_gate_event(
            actor=actor,
            session_id=sid,
            decision=decision,
            candidate_ref=candidate_ref,
        )
        await self._host.append(
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
        # Resolve the target against the sealed log rather than format-checking it.
        # `target_ref` only validates shape, so an unresolvable (seq, hash) used to
        # append a reinforcement citing an event that does not exist -- an uncitable
        # citation in an append-only log -- and supplying just one of the pair
        # silently dropped the reinforcement entirely. Match the sibling evolution
        # ops (edit/rollback/forget), which all require a resolvable target.
        target: dict[str, Any] | None = None
        if target_seq is not None or target_hash is not None:
            target_event = self._require_target_event(target_seq, target_hash, session_id=sid)
            target = {"seq": target_event.seq, "hash": target_event.hash}
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
        outcome_event = await self._host.append(
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
            await self._host.append(
                reinforce_spec["event_type"], reinforce_spec["actor"], payload=reinforce_spec["payload"], session_id=sid
            )
            result["reinforced"] = "confirmed" if outcome == "success" else "invalidated"

        if outcome in ("failure", "partial") and lesson:
            rule_confidence = preventive_rule_confidence(outcome, confidence)
            decision = await self._host.evaluate_evolution_gate(
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
            rule_event = await self._host.append(
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

        decision = await self._host.evaluate_evolution_gate(
            "update", confidence, candidate_ref=target, actor=actor, session_id=sid
        )
        spec = build_memory_correction_event(
            actor=actor,
            session_id=sid,
            target=target,
            new_content=new_content,
            reason=reason,
        )
        event = await self._host.append(
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

        decision = await self._host.evaluate_evolution_gate(
            "update", confidence, candidate_ref=target, actor=actor, session_id=sid
        )
        spec = build_memory_rollback_event(
            actor=actor,
            session_id=sid,
            target=target,
            reason=reason,
            reverts=reverts,
        )
        event = await self._host.append(
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
        if not self._host.settings.forgetting_enabled:
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
        decision = await self._host.evaluate_evolution_gate(
            "forget", confidence, candidate_ref=target, actor=actor, session_id=sid
        )
        erased_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        erased = self._host.session_manager.vault.erase(cell_id, erased_at=erased_at)
        spec = build_memory_forgotten_event(
            actor=actor, session_id=sid, target=target, cell_id=cell_id, reason=reason
        )
        try:
            event = await self._host.append(
                spec["event_type"], spec["actor"], payload=spec["payload"], session_id=sid
            )
        except Exception as exc:
            # The DEK is already destroyed, so the memory is permanently
            # unrecoverable — but the audit tombstone did not land. Do not let
            # this surface as a routine error: it is an erased-but-unaudited
            # integrity gap that an operator must see and repair by re-appending
            # the (deterministic, replay-safe) tombstone spec.
            self._host._metrics().record_degraded_operation("forget", "tombstone_append_failed")
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
        self._host._invalidate_query_page_cache(sid)
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
        decrypted = decrypt_payload(payload, vault=self._host.session_manager.vault)
        return event.model_copy(update={"payload": decrypted})

    def _require_target_event(
        self, target_seq: object, target_hash: object, *, session_id: str
    ) -> Any:
        """Return the sealed event identified by ``(seq, hash)`` or raise ValueError."""
        if not isinstance(target_seq, int) or isinstance(target_seq, bool) or target_seq < 1:
            raise ValueError("target_seq must be a positive integer")
        if not isinstance(target_hash, str) or len(target_hash) != 64:
            raise ValueError("target_hash must be a 64-character hex digest")
        eventlog = self._host.session_manager.get(session_id).eventlog
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
        eventlog = self._host.session_manager.get(session_id).eventlog
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
        eventlog = self._host.session_manager.get(session_id).eventlog
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
        if self._host.embedding_provider is not None:
            try:
                extraction = embed_extraction(extraction, self._host.embedding_provider)
            except Exception:
                self._host._metrics().record_degraded_operation("append", "embedding_provider_unavailable")
        try:
            await self._host.graph.upsert_extraction(extraction, session_id=session_id)
        except Exception:
            self._host._metrics().record_degraded_operation("append", "graph_projection_unavailable")
        with suppress(Exception):
            await self._host.tracer.trace_append(event.type, event.actor, event.seq)

        # Metrics
        metrics = self._host._metrics()
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
        policy = resolve_evolution_policy(self._host.settings)
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
                    await self._host._project_event(gate_event, session_id=session_id)
                    continue
            event = eventlog.append(
                generated["event_type"],
                actor=generated["actor"],
                payload=validate_payload(generated["payload"]),
                thread=session_id,
            )
            await self._host._project_event(event, session_id=session_id)

    def _encoding_classification_active(self) -> bool:
        """Return whether append-time encoding classification should run.

        The write-time gate tags payloads only when ``ENCODING_GATE_ENABLED``;
        interference detection additionally runs under the cognitive
        retrieval profile (classification is its novelty signal). With both
        off, appends are byte-identical to the pre-gate contract.
        """
        return self._host._encoding_gate_enabled or self._host.retrieval_profile.salience_ranking

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
            index = self._host._verbatim_index(session_id)
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
                            self._host.session_manager.get(session_id).eventlog.read_all()
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
            self._host._metrics().record_degraded_operation("append", "encoding_classification_unavailable")
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
                hits = await self._host.graph.search_exact(name, session_id=session_id)
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
                    existing = await self._host.graph.search_exact(
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
            self._host._metrics().record_degraded_operation("append", "interference_detection_unavailable")
        return None

    def _memory_above_floor(self, target: dict[str, Any], *, session_id: str) -> bool:
        """Return whether a memory's replayed salience clears the attenuation floor.

        Memories with no reinforcement history carry the implicit base
        salience (1.0) and are always above the default floor.
        """
        events = self._host.session_manager.get(session_id).eventlog.read_all()
        ledger = SalienceLedger(half_life_days=self._host._salience_half_life_days)
        states = ledger.replay(events, now=datetime.now(UTC))
        state = states.get(EventRef(seq=int(target["seq"]), hash=str(target["hash"])))
        score = state.score if state is not None else SALIENCE_BASE
        return bool(score >= self._host._salience_floor)

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
            await self._host.propose_belief_update(
                finding["claim"],
                rationale=finding["rationale"],
                confidence=0.5,
                source_events=finding["source_events"],
                phase="reflection",
                session_id=session_id,
                actor="zaxy-memory",
            )
        except Exception:
            self._host._metrics().record_degraded_operation("append", "interference_proposal_unavailable")

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
            index = self._host._session_event_ref_index(session_id)
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
            await self._host._append_event_spec(spec, session_id=session_id)
        except Exception:
            self._host._metrics().record_degraded_operation("append", "salience_reinforcement_unavailable")

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
            await self._host.append(
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
            await self._host.append(
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
        previous = load_refresh_state(self._host.eventloom_path, session_id=sid, kind=kind)
        plan: ContextRefreshPlan = plan_context_refresh(
            path,
            kind=kind,
            previous=previous,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )
        for event in plan.events:
            await self._host.append(
                event["event_type"],
                actor=event["actor"],
                payload=event["payload"],
                session_id=sid,
            )
        save_refresh_state(self._host.eventloom_path, session_id=sid, state=plan.next_state)
        return ContextRefreshReport(
            session_id=sid,
            kind=plan.kind,
            event_count=len(plan.events),
            summary=plan.summary,
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
            await self._host.append(
                event["event_type"],
                actor=event["actor"],
                payload=event["payload"],
                session_id=sid,
            )
        return len(events)

