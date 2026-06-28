"""Two-tier (episodic + consolidated) long-horizon checkout assembly.

For very long ("never-ending thread") sessions, Memory Checkout can split a
session's history into two explicit tiers:

* the EPISODIC (recent) tier — the most recent ``recent_window`` events, kept at
  full detail exactly as the single-tier checkout does today; and
* the CONSOLIDATED (remote) tier — older history represented *not* by raw old
  events but by the session's already-cited I2 consolidation artifacts: the
  ACCEPTED/ACTIVE consolidation candidates whose source events have scrolled out
  of the recent window. Each surfaces as a bounded, cited, non-authoritative
  context item carrying its source-event citation(s).

Nothing is summarized afresh here (no destructive summarization): the remote
tier only replays the consolidation candidates the consolidation pipeline
(``consolidation.py`` / ``consolidation_pipeline.py``) already produced and that
a review event accepted. This module owns the partition logic;
:meth:`zaxy.core.MemoryFabric.assemble_context` builds the plan when the
long-horizon flag is engaged and the session exceeds the window, and
:func:`zaxy.core.checkout_build.build_memory_checkout` renders the
``long_horizon`` diagnostics section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zaxy.context import Context

#: Source-lane / assembly-lane marker for consolidated remote-tier contexts.
CONSOLIDATED_LANE = "consolidated"

#: Consolidation review statuses whose candidates may surface in the remote tier.
#: ``accepted`` is the consolidation-candidate review disposition; ``active`` is
#: accepted too for forward-compatibility with promotion-style lifecycles.
SURFACEABLE_REVIEW_STATUSES = frozenset({"accepted", "active"})

_CANDIDATE_CREATED = "consolidation.candidate.created"
_CANDIDATE_REVIEWED = "consolidation.candidate.reviewed"


@dataclass(frozen=True)
class LongHorizonPlan:
    """Resolved episodic/consolidated partition for one checkout."""

    enabled: bool
    recent_window: int
    episodic_count: int
    horizon_split_seq: int | None
    consolidated_contexts: list[Context] = field(default_factory=list)

    @property
    def consolidated_count(self) -> int:
        return len(self.consolidated_contexts)

    def to_diagnostics(self) -> dict[str, Any]:
        """Return the split summary threaded into ``ContextAssembly.long_horizon``.

        The consolidated count and the per-item list are recomputed by the
        checkout builder from the ranked contexts so the diagnostics reflect what
        actually reached the packet; only the immutable split facts live here.
        """
        return {
            "enabled": self.enabled,
            "recent_window": self.recent_window,
            "episodic_count": self.episodic_count,
            "horizon_split_seq": self.horizon_split_seq,
        }


def build_long_horizon_plan(
    session_events: list[Any],
    *,
    session_id: str,
    recent_window: int,
    budget: int,
) -> LongHorizonPlan:
    """Partition a session into episodic + consolidated tiers.

    ``session_events`` is the full (as-of-filtered) replay in seq order. The most
    recent ``recent_window`` events form the episodic tier; older history is
    represented by ACCEPTED/ACTIVE consolidation candidates whose source events
    fall at or before the horizon split. ``budget`` (> 0) bounds the number of
    consolidated items surfaced.

    When the session does not exceed the window there is no older region: the
    plan is engaged but the remote tier is empty (graceful — just episodic).
    """
    total = len(session_events)
    if recent_window <= 0 or total <= recent_window:
        return LongHorizonPlan(
            enabled=True,
            recent_window=recent_window,
            episodic_count=total,
            horizon_split_seq=None,
            consolidated_contexts=[],
        )
    older_events = session_events[: total - recent_window]
    split_seq = _event_seq(older_events[-1])
    consolidated: list[Context] = []
    for candidate in _replay_consolidation_candidates(session_events, session_id=session_id):
        if candidate["review_status"] not in SURFACEABLE_REVIEW_STATUSES:
            continue
        max_source_seq = candidate["max_source_seq"]
        if max_source_seq is None or max_source_seq > split_seq:
            # No older source events (none cited, or all still inside the recent
            # window): the episodic tier already carries them, so surfacing the
            # candidate here would duplicate rather than consolidate.
            continue
        consolidated.append(_consolidated_context(candidate))
        if budget > 0 and len(consolidated) >= budget:
            break
    return LongHorizonPlan(
        enabled=True,
        recent_window=recent_window,
        episodic_count=recent_window,
        horizon_split_seq=split_seq,
        consolidated_contexts=consolidated,
    )


def _replay_consolidation_candidates(
    session_events: list[Any], *, session_id: str
) -> list[dict[str, Any]]:
    """Replay candidate.created/reviewed events into current candidate state.

    Mirrors :meth:`MemoryFabric.consolidation_status`'s replay: a candidate's
    current ``review_status`` is the latest review disposition (or ``pending``).
    """
    candidates: dict[str, dict[str, Any]] = {}
    for event in session_events:
        event_type = getattr(event, "type", None)
        if event_type == _CANDIDATE_CREATED:
            payload = getattr(event, "payload", None) or {}
            candidate_id = payload.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidates:
                continue
            thread = getattr(event, "thread", None) or session_id
            source_events = _source_event_rows(payload.get("source_events"))
            source_citations = [
                citation
                for src in source_events
                if (citation := _event_citation(thread, src["seq"], src["hash"])) is not None
            ]
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "candidate_type": payload.get("candidate_type"),
                "summary": payload.get("summary") or payload.get("title") or candidate_id,
                "confidence": payload.get("confidence"),
                "review_status": str(payload.get("review_status") or "pending"),
                "citation": _event_citation(
                    thread, getattr(event, "seq", None), getattr(event, "hash", None)
                ),
                "source_event_citations": source_citations,
                "max_source_seq": max((src["seq"] for src in source_events), default=None),
            }
        elif event_type == _CANDIDATE_REVIEWED:
            payload = getattr(event, "payload", None) or {}
            candidate_id = payload.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id in candidates:
                status = payload.get("status")
                if isinstance(status, str) and status:
                    candidates[candidate_id]["review_status"] = status
    return list(candidates.values())


def _consolidated_context(candidate: dict[str, Any]) -> Context:
    """Project one accepted/active candidate into a cited, non-authoritative Context."""
    confidence = candidate["confidence"]
    score = (
        float(confidence)
        if isinstance(confidence, int | float) and not isinstance(confidence, bool)
        else 0.5
    )
    source_citations = list(candidate["source_event_citations"])
    return Context(
        content=str(candidate["summary"]),
        source=CONSOLIDATED_LANE,
        score=score,
        metadata={
            "assembly_lane": CONSOLIDATED_LANE,
            "source_lane": CONSOLIDATED_LANE,
            "tier": CONSOLIDATED_LANE,
            "citation": candidate["citation"],
            "source_event_citations": source_citations,
            "source_event_count": len(source_citations),
            "non_authoritative": True,
            "authority_status": "non_authoritative",
            "review_status": candidate["review_status"],
            "candidate_id": candidate["candidate_id"],
            "candidate_type": candidate["candidate_type"],
            "confidence": confidence,
            "entity_type": "consolidation_candidate",
            "entity_name": candidate["candidate_id"],
        },
    )


def _source_event_rows(value: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            seq = item.get("seq")
            event_hash = item.get("hash")
            if (
                isinstance(seq, int)
                and not isinstance(seq, bool)
                and seq > 0
                and isinstance(event_hash, str)
                and event_hash
            ):
                rows.append({"seq": seq, "hash": event_hash})
    return rows


def _event_citation(thread: object, seq: object, event_hash: object) -> str | None:
    if (
        not isinstance(thread, str)
        or not thread
        or not isinstance(seq, int)
        or isinstance(seq, bool)
        or seq <= 0
        or not isinstance(event_hash, str)
        or not event_hash
    ):
        return None
    return f"eventloom://{thread}/events/{seq}#{event_hash[:12]}"


def _event_seq(event: Any) -> int:
    seq = getattr(event, "seq", None)
    return seq if isinstance(seq, int) and not isinstance(seq, bool) else 0
