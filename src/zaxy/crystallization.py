"""Governed sleep-time crystallization runner (Zaxy 3 / I2).

A config-gated, **operator/cron-triggered one-shot** reflection pass over a
session's Eventloom log. In an idle window it *schedules the existing
primitives* — it does **not** reimplement them — to emit **additive,
review-gated, non-authoritative, cited** consolidation / skill / metacognition
candidates, turning today's on-demand pipelines into a default loop.

Design (mirrors :mod:`zaxy.export_sinks`: operator-side, no daemon):

- **Not a daemon.** One pass per invocation; recurring scheduling is left to an
  external scheduler (cron / the OS). There is no MCP tool — the MCP surface
  stays pull-only.
- **Reuses primitives.** Consolidation via
  :meth:`MemoryFabric.propose_consolidation_candidates`, skills via
  :func:`zaxy.procedure_mining.mine_and_propose`, the additive compaction
  projection via :func:`zaxy.compaction.build_compaction_projection` (gated by
  :func:`zaxy.compaction.audit_event_log`), the metacognition monitor via
  :func:`zaxy.metacognition.summarize_metacognition_events` +
  :func:`zaxy.metacognition.build_reverify_request_event`, and a read-only
  salience diagnostic via :class:`zaxy.salience.SalienceLedger`.
- **Governed.** Every auto-apply decision routes through the shipped I4 gate
  (:meth:`MemoryFabric.evaluate_evolution_gate`). "Auto-apply" only ever means
  emitting a ``consolidation.candidate.reviewed status=accepted`` review — still
  ``non_authoritative`` and reversible — never a promotion to authority.
- **Idempotent.** The underlying builders use deterministic candidate ids and
  the propose / mine paths skip ids that already exist, so re-running a pass over
  the same log adds no duplicate candidates; the metacognition monitor dedups on
  the deterministic reverify id as well.
- **Auditable.** Each pass emits one ``crystallization.run.completed`` summary
  event (non-authoritative, citing the candidate ids it generated) so the pass
  itself is replayable. Stages are isolated from one another: a failing stage is
  recorded on the summary event rather than aborting the pass before it is
  appended, so an unattended run never leaves a partially-applied log with no
  audit record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from zaxy.compaction import CompactionProjection, audit_event_log, build_compaction_projection
from zaxy.consolidation import build_consolidation_review_event
from zaxy.learned_context import (
    LEARNED_CONTEXT_EVENT_TYPE,
    build_projection_built_payload,
    covered_head,
    learned_context_path,
    write_learned_context,
)
from zaxy.metacognition import (
    build_reverify_request_event,
    summarize_metacognition_events,
)
from zaxy.procedure_mining import mine_and_propose
from zaxy.salience import SalienceLedger
from zaxy.security import validate_session_id

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.core.fabric import MemoryFabric

#: Event type for the per-pass audit/summary record.
CRYSTALLIZATION_EVENT_TYPE = "crystallization.run.completed"

#: Default actor stamped on crystallization-emitted events.
DEFAULT_CRYSTALLIZER_ACTOR = "zaxy-crystallizer"

#: The governed evolution op every crystallization candidate is gated under.
_CRYSTALLIZATION_GATE_OP = "consolidate"

#: Non-authoritative status reused by every event this runner emits.
_AUTHORITY_STATUS = "non_authoritative"

#: Number of top-salient memories surfaced in the read-only diagnostic.
DEFAULT_TOP_SALIENT = 5

_CANDIDATE_EVENT_TYPE = "consolidation.candidate.created"
_REVERIFY_EVENT_TYPE = "metacognition.reverify.requested"

#: Stage labels recorded on per-stage error entries.
STAGE_CONSOLIDATION = "consolidation"
STAGE_PROCEDURE_MINING = "procedure_mining"
STAGE_GATING = "gating"
STAGE_METACOGNITION = "metacognition"
STAGE_COMPACTION = "compaction"
STAGE_SALIENCE = "salience"


def _record_stage_error(
    errors: list[dict[str, Any]],
    stage: str,
    exc: BaseException,
    *,
    candidate_id: str | None = None,
) -> None:
    """Append a JSON-serializable record of one failed crystallization stage."""
    entry: dict[str, Any] = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc) or repr(exc),
    }
    if candidate_id is not None:
        entry["candidate_id"] = candidate_id
    errors.append(entry)


@dataclass(frozen=True)
class CrystallizationReport:
    """Outcome of one governed crystallization pass.

    All counts are for candidates *freshly proposed by this pass* (so a
    re-run over an unchanged log reports zeros — the idempotency contract).
    ``latency_ms`` is real wall-clock time and is intentionally excluded from
    :meth:`to_dict`-based equality comparisons by callers. ``stage_errors``
    holds one entry per stage that raised; it is empty on a fully successful
    pass.
    """

    session_id: str
    consolidation_candidates: int
    procedure_candidates: int
    auto_accepted: int
    left_pending: int
    metacognition: dict[str, Any]
    reverify_requested: int
    compaction: dict[str, Any] | None
    top_salient: list[dict[str, Any]]
    gate_decisions: list[dict[str, Any]]
    stage_errors: list[dict[str, Any]]
    latency_ms: float

    @property
    def ok(self) -> bool:
        """Whether every stage of the pass completed without raising."""
        return not self.stage_errors

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation of the report."""
        return {
            "ok": self.ok,
            "stage_errors": self.stage_errors,
            "session_id": self.session_id,
            "consolidation_candidates": self.consolidation_candidates,
            "procedure_candidates": self.procedure_candidates,
            "auto_accepted": self.auto_accepted,
            "left_pending": self.left_pending,
            "metacognition": self.metacognition,
            "reverify_requested": self.reverify_requested,
            "compaction": self.compaction,
            "top_salient": self.top_salient,
            "gate_decisions": self.gate_decisions,
            "latency_ms": self.latency_ms,
        }


async def run_crystallization_pass(
    fabric: MemoryFabric,
    *,
    session_id: str = "default",
    consolidation: bool = True,
    procedure_mining: bool = True,
    compaction: bool = False,
    metacognition: bool = True,
    auto_apply: bool = True,
    now: datetime | None = None,
    actor: str = DEFAULT_CRYSTALLIZER_ACTOR,
) -> CrystallizationReport:
    """Run one governed, additive, cited crystallization pass over a session log.

    The pass is a *scheduler* over existing primitives:

    1. **Consolidation** — propose cited, review-pending consolidation
       candidates (no-op when ``consolidation`` is false).
    2. **Procedure mining** — mine recurring successful tool sequences into
       review-pending procedure candidates (no-op when ``procedure_mining`` is
       false). Mining is inherently cross-session, so the whole session log is
       considered rather than a single thread.
    3. **Auto-apply gating** — every freshly proposed candidate is routed
       through the I4 evolution gate. When ``auto_apply`` is true *and* the gate
       returns ``auto_apply``, a ``status=accepted`` review is emitted (still
       non-authoritative and reversible); otherwise the candidate is left
       pending. Every gate decision is recorded.
    4. **Metacognition monitor** — for each open gap (unresolved conflict
       cluster / open known-unknown) without an open re-verification request,
       emit a cited re-verification request. Deduped on the deterministic
       reverify id so re-runs add nothing.
    5. **Compaction** (opt-in) — audit the log and, only if the audit is
       ``safe``, build the *additive* compaction projection and record a summary.
       When ``learned_context_enabled`` is set the projection is additionally
       persisted as an I2 learned-context artifact and a non-authoritative
       ``crystallization.projection.built`` event records the build; with the
       setting off (the default) nothing is written and the stage is report-only.
    6. **Salience diagnostic** — a read-only top-salient listing; no
       reinforcement is emitted here.

    Every stage is isolated: a stage that raises is recorded in ``stage_errors``
    (and in the summary event's ``stage_errors`` / ``ok`` fields) and the pass
    continues, so an unattended run always leaves an audit record describing
    exactly which stages succeeded and which failed. Errors are never swallowed
    silently — callers surface them, and :attr:`CrystallizationReport.ok` is
    false whenever any stage failed.

    Finally one ``crystallization.run.completed`` summary event is appended so
    the pass is itself auditable and replayable.
    """
    started = time.perf_counter()
    sid = validate_session_id(session_id)
    moment = now if now is not None else datetime.now(UTC)
    eventlog = fabric.session_manager.get(sid).eventlog

    fresh_candidates: list[dict[str, Any]] = []
    consolidation_count = 0
    procedure_count = 0
    # Stages are isolated so an unattended (cron) pass still reaches the summary
    # append below: a partially-applied pass that emitted no audit record at all
    # would be unreplayable, which is the worse failure for a background job.
    stage_errors: list[dict[str, Any]] = []

    if consolidation:
        try:
            result = await fabric.propose_consolidation_candidates(session_id=sid, actor=actor)
            consolidation_count = int(result["candidate_count"])
            for entry in result["events"]:
                fresh_candidates.append(
                    {
                        "candidate_id": str(entry["candidate_id"]),
                        "seq": int(entry["seq"]),
                        "hash": str(entry["hash"]),
                        "source": "consolidation",
                    }
                )
        except Exception as exc:
            _record_stage_error(stage_errors, STAGE_CONSOLIDATION, exc)

    if procedure_mining:
        try:
            # Procedure mining requires support across >=2 distinct sessions; it is
            # mined over the whole session log (every thread present) rather than a
            # single-thread filter, which would make it a guaranteed no-op.
            summary = mine_and_propose(eventlog, session_ids=None, actor=actor)
            procedure_count = int(summary.appended_count)
            for appended in summary.appended:
                fresh_candidates.append(
                    {
                        "candidate_id": appended.candidate_id,
                        "seq": appended.seq,
                        "hash": appended.hash,
                        "source": "procedure",
                    }
                )
        except Exception as exc:
            _record_stage_error(stage_errors, STAGE_PROCEDURE_MINING, exc)

    confidence_by_id = _confidence_by_candidate_id(eventlog) if fresh_candidates else {}

    gate_decisions: list[dict[str, Any]] = []
    auto_accepted = 0
    left_pending = 0
    for candidate in fresh_candidates:
        candidate_id = candidate["candidate_id"]
        confidence = confidence_by_id.get(candidate_id, 0.0)
        # Isolated per candidate: one un-gateable candidate must not cost the
        # rest of the batch its gate decision. A failed candidate is counted
        # neither accepted nor pending — only the recorded error describes it.
        try:
            decision = await fabric.evaluate_evolution_gate(
                _CRYSTALLIZATION_GATE_OP,
                confidence,
                candidate_ref={"candidate_id": candidate_id},
                actor=actor,
                session_id=sid,
            )
            gate_decisions.append(
                {
                    "candidate_id": candidate_id,
                    "op": decision.op,
                    "confidence": decision.confidence,
                    "auto_apply": decision.auto_apply,
                    "tier": decision.tier,
                }
            )
            if auto_apply and decision.auto_apply:
                review = build_consolidation_review_event(
                    actor=actor,
                    session_id=sid,
                    candidate_id=candidate_id,
                    status="accepted",
                    rationale=f"auto-accepted by crystallization under {decision.tier}",
                )
                await fabric.append(
                    review["event_type"],
                    review["actor"],
                    payload=review["payload"],
                    session_id=sid,
                )
                auto_accepted += 1
            else:
                left_pending += 1
        except Exception as exc:
            _record_stage_error(stage_errors, STAGE_GATING, exc, candidate_id=candidate_id)

    metacognition_summary: dict[str, Any] = {}
    reverify_requested = 0
    if metacognition:
        try:
            metacognition_summary, reverify_requested = await _run_metacognition_monitor(
                fabric, eventlog, session_id=sid, actor=actor
            )
        except Exception as exc:
            _record_stage_error(stage_errors, STAGE_METACOGNITION, exc)

    compaction_summary: dict[str, Any] | None = None
    if compaction:
        try:
            compaction_summary, projection = _run_compaction_diagnostic(eventlog)
            # I2: persisting is gated separately from building. With the gate off
            # the pass behaves exactly as it did pre-I2 (build, report counts,
            # write nothing), so the feature is inert by default on this path too.
            if projection is not None and getattr(
                fabric.settings, "learned_context_enabled", False
            ):
                compaction_summary = {
                    **compaction_summary,
                    "learned_context": await _persist_learned_context(
                        fabric, eventlog, projection, session_id=sid, actor=actor
                    ),
                }
        except Exception as exc:
            _record_stage_error(stage_errors, STAGE_COMPACTION, exc)

    top_salient: list[dict[str, Any]] = []
    try:
        top_salient = _top_salient(fabric, eventlog, now=moment, limit=DEFAULT_TOP_SALIENT)
    except Exception as exc:
        _record_stage_error(stage_errors, STAGE_SALIENCE, exc)

    citations = [
        {"candidate_id": candidate["candidate_id"], "seq": candidate["seq"], "hash": candidate["hash"]}
        for candidate in fresh_candidates
    ]
    await fabric.append(
        CRYSTALLIZATION_EVENT_TYPE,
        actor,
        payload={
            "authority_status": _AUTHORITY_STATUS,
            "session_id": sid,
            "consolidation_candidates": consolidation_count,
            "procedure_candidates": procedure_count,
            "auto_accepted": auto_accepted,
            "left_pending": left_pending,
            "reverify_requested": reverify_requested,
            "compaction_safe": None if compaction_summary is None else bool(compaction_summary.get("safe")),
            "candidates": citations,
            "ok": not stage_errors,
            "stage_errors": stage_errors,
        },
        session_id=sid,
    )

    latency_ms = (time.perf_counter() - started) * 1000.0
    return CrystallizationReport(
        session_id=sid,
        consolidation_candidates=consolidation_count,
        procedure_candidates=procedure_count,
        auto_accepted=auto_accepted,
        left_pending=left_pending,
        metacognition=metacognition_summary,
        reverify_requested=reverify_requested,
        compaction=compaction_summary,
        top_salient=top_salient,
        gate_decisions=gate_decisions,
        stage_errors=stage_errors,
        latency_ms=latency_ms,
    )


def _confidence_by_candidate_id(eventlog: Any) -> dict[str, float]:
    """Map every consolidation candidate id in the log to its confidence."""
    confidence: dict[str, float] = {}
    for event in eventlog.read_all():
        if event.type != _CANDIDATE_EVENT_TYPE:
            continue
        payload = event.payload
        candidate_id = payload.get("candidate_id")
        if not isinstance(candidate_id, str):
            continue
        raw = payload.get("confidence")
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            confidence[candidate_id] = float(raw)
    return confidence


async def _run_metacognition_monitor(
    fabric: MemoryFabric,
    eventlog: Any,
    *,
    session_id: str,
    actor: str,
) -> tuple[dict[str, Any], int]:
    """Fire re-verification requests for open metacognitive gaps; idempotent.

    Reads the replayed metacognition surface and, for each unresolved conflict
    cluster and open known-unknown that does not already carry an open
    re-verification request, emits one cited ``metacognition.reverify.requested``
    event. Dedup is on the deterministic reverify id (and the gap's claim key),
    so re-running over the same log emits nothing new.
    """
    events = eventlog.read_all()
    metacognition_events = [
        {"event_type": event.type, "payload": dict(event.payload), "seq": event.seq, "hash": event.hash}
        for event in events
        if str(event.type).startswith("metacognition.")
    ]
    summary = summarize_metacognition_events(metacognition_events)

    existing_reverify_ids: set[str] = set()
    open_reverify_claim_keys: set[str] = set()
    for event in metacognition_events:
        if event["event_type"] != _REVERIFY_EVENT_TYPE:
            continue
        payload = event["payload"]
        reverify_id = payload.get("reverify_id")
        if isinstance(reverify_id, str):
            existing_reverify_ids.add(reverify_id)
        if payload.get("status") == "open":
            claim_key = payload.get("claim_key")
            if isinstance(claim_key, str) and claim_key:
                open_reverify_claim_keys.add(claim_key)

    specs = _plan_reverify_requests(
        summary,
        actor=actor,
        session_id=session_id,
        existing_reverify_ids=existing_reverify_ids,
        open_reverify_claim_keys=open_reverify_claim_keys,
    )
    for spec in specs:
        await fabric.append(
            spec["event_type"],
            spec["actor"],
            payload=spec["payload"],
            session_id=session_id,
        )
    return summary, len(specs)


def _plan_reverify_requests(
    summary: dict[str, Any],
    *,
    actor: str,
    session_id: str,
    existing_reverify_ids: set[str],
    open_reverify_claim_keys: set[str],
) -> list[dict[str, Any]]:
    """Build the deduplicated set of reverify-request specs for open gaps."""
    specs: list[dict[str, Any]] = []

    for cluster in summary.get("conflict_clusters", []):
        claim_key = _optional_str(cluster.get("claim_key"))
        if claim_key is not None and claim_key in open_reverify_claim_keys:
            continue
        sources = _merge_citations(
            cluster.get("supporting_source_events"),
            cluster.get("conflicting_source_events"),
        )
        if not sources:
            continue
        claim = _optional_str(cluster.get("claim")) or claim_key or "conflicting claim"
        spec = build_reverify_request_event(
            actor=actor,
            session_id=session_id,
            query=f"Re-verify conflicting claim: {claim}",
            reason=(
                "Crystallization metacognition monitor flagged an unresolved conflict cluster "
                f"({cluster.get('cluster_id', 'unknown')})"
            ),
            source_events=sources,
            priority="high",
            claim_key=claim_key,
        )
        if _register_spec(spec, specs, existing_reverify_ids, open_reverify_claim_keys, claim_key):
            continue

    for unknown in summary.get("open_unknowns", []):
        claim_key = _optional_str(unknown.get("claim_key"))
        if claim_key is not None and claim_key in open_reverify_claim_keys:
            continue
        sources = _normalize_citations(unknown.get("source_events"))
        if not sources:
            continue
        query = (
            _optional_str(unknown.get("reverify_query"))
            or _optional_str(unknown.get("question"))
            or "Re-verify open known-unknown"
        )
        spec = build_reverify_request_event(
            actor=actor,
            session_id=session_id,
            query=query,
            reason=(
                "Crystallization metacognition monitor flagged an open known-unknown "
                f"({unknown.get('unknown_id', 'unknown')})"
            ),
            source_events=sources,
            priority="normal",
            claim_key=claim_key,
        )
        if _register_spec(spec, specs, existing_reverify_ids, open_reverify_claim_keys, claim_key):
            continue

    return specs


def _register_spec(
    spec: dict[str, Any],
    specs: list[dict[str, Any]],
    existing_reverify_ids: set[str],
    open_reverify_claim_keys: set[str],
    claim_key: str | None,
) -> bool:
    """Record ``spec`` unless its reverify id already exists. Returns True if skipped."""
    reverify_id = str(spec["payload"]["reverify_id"])
    if reverify_id in existing_reverify_ids:
        return True
    specs.append(spec)
    existing_reverify_ids.add(reverify_id)
    if claim_key is not None:
        open_reverify_claim_keys.add(claim_key)
    return False


def _run_compaction_diagnostic(
    eventlog: Any,
) -> tuple[dict[str, Any], CompactionProjection | None]:
    """Audit the log and, only if safe, build the additive compaction projection.

    Returns the summary alongside the projection itself so the caller can persist
    it as an I2 learned-context artifact; before I2 the projection was built and
    discarded here.
    """
    audit = audit_event_log(eventlog)
    if not audit.safe:
        return {
            "safe": False,
            "event_count": audit.event_count,
            "identity_recall": audit.identity_recall,
            "citation_coverage": audit.citation_coverage,
            "unsafe_reasons": list(audit.unsafe_reasons),
        }, None
    projection = build_compaction_projection(eventlog)
    return {
        "safe": True,
        "strategy": projection.strategy,
        "source_event_count": projection.source_event_count,
        "record_count": len(projection.records),
        "projection_id": projection.projection_id,
        "identity_recall": audit.identity_recall,
        "citation_coverage": audit.citation_coverage,
    }, projection


async def _persist_learned_context(
    fabric: MemoryFabric,
    eventlog: Any,
    projection: CompactionProjection,
    *,
    session_id: str,
    actor: str,
) -> dict[str, Any]:
    """Persist the projection as an I2 artifact and record the build in the log.

    The artifact is written *before* the event is appended. That ordering is the
    safe one: a crash between the two leaves a file no event vouches for, and an
    unvouched file is defined as untrusted and ignored on load. The reverse order
    would leave an event pointing at a file that does not exist, which is merely
    a wasted lookup — but writing first also means the covered head is the log tip
    as it stood when the projection was built, not after this event advanced it.
    """
    events = eventlog.read_all()
    head = covered_head(events)
    if head is None:
        return {"persisted": False, "reason": "empty_log"}
    covered_seq, covered_hash = head
    path = learned_context_path(fabric.eventloom_path, session_id)
    write_learned_context(
        projection,
        path,
        session_id=session_id,
        covered_seq=covered_seq,
        covered_hash=covered_hash,
    )
    payload = build_projection_built_payload(
        projection,
        session_id=session_id,
        covered_seq=covered_seq,
        covered_hash=covered_hash,
        artifact_path=str(path),
    )
    await fabric.append(
        LEARNED_CONTEXT_EVENT_TYPE,
        actor,
        payload=payload,
        session_id=session_id,
    )
    return {
        "persisted": True,
        "artifact_path": str(path),
        "covered_seq": covered_seq,
        "covered_hash": covered_hash,
        "record_count": payload["record_count"],
    }


def _top_salient(
    fabric: MemoryFabric,
    eventlog: Any,
    *,
    now: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    """Return the top-``limit`` salient memory citations (read-only diagnostic)."""
    ledger = SalienceLedger(
        half_life_days=fabric._salience_half_life_days,
        multipliers=fabric._salience_multipliers,
    )
    states = ledger.replay(eventlog.read_all(), now=now)
    ranked = sorted(states.items(), key=lambda item: (-item[1].score, item[0].seq, item[0].hash))
    return [
        {"seq": ref.seq, "hash": ref.hash, "score": round(state.score, 6)}
        for ref, state in ranked[:limit]
    ]


def _merge_citations(*groups: Any) -> list[dict[str, Any]]:
    """Merge several citation lists, deduping on (seq, hash) and preserving order."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for group in groups:
        for citation in _normalize_citations(group):
            key = (citation["seq"], citation["hash"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(citation)
    return merged


def _normalize_citations(value: Any) -> list[dict[str, Any]]:
    """Coerce a payload citation list into validated ``{seq, hash}`` mappings."""
    if not isinstance(value, list):
        return []
    citations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        seq = item.get("seq")
        event_hash = item.get("hash")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
            continue
        if not isinstance(event_hash, str):
            continue
        citations.append({"seq": seq, "hash": event_hash})
    return citations


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None
