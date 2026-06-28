"""Tests for I3 two-tier (episodic + consolidated) long-horizon checkout assembly.

Covers the partition helper (``zaxy.long_horizon``) and the end-to-end fabric
wiring: a long thread gets a bounded, cited consolidated remote tier for older
history while the default-off path stays byte-identical to the single-tier
checkout. Uses the real embedded MemoryFabric, the real consolidation builders,
and accept-via-review-event, mirroring the consolidation pipeline tests.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from zaxy.config import Settings
from zaxy.consolidation import build_consolidation_review_event
from zaxy.context import Context
from zaxy.core import MemoryFabric
from zaxy.core.checkout_build import (
    _checkout_long_horizon,
    _checkout_recall_limit,
    build_memory_checkout,
)
from zaxy.long_horizon import CONSOLIDATED_LANE, build_long_horizon_plan

_CITATION_RE = re.compile(r"^eventloom://[^/\s]+/events/[1-9][0-9]*#[0-9a-f]{12}$")
_RECENT_WINDOW = 10
_QUERY = "what did we learn from the earlier refactor work"


# --------------------------------------------------------------------------
# Partition-helper unit tests (synthetic events) — branch coverage.
# --------------------------------------------------------------------------


def _ev(seq: int, event_type: str, payload: dict | None = None, thread: str = "thread-x") -> SimpleNamespace:
    return SimpleNamespace(seq=seq, type=event_type, payload=payload or {}, hash="a" * 64, thread=thread)


def _candidate_created(seq: int, candidate_id: str, *, review_status: str, source_seqs: list[int], summary: str = "summary") -> SimpleNamespace:
    return _ev(
        seq,
        "consolidation.candidate.created",
        {
            "candidate_id": candidate_id,
            "candidate_type": candidate_id.split(":")[1],
            "summary": summary,
            "confidence": 0.7,
            "review_status": review_status,
            "source_events": [{"seq": s, "hash": "b" * 64} for s in source_seqs],
        },
    )


def test_build_long_horizon_plan_short_session_is_episodic_only() -> None:
    events = [_ev(i, "tool.call.completed") for i in range(1, 4)]
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=10, budget=10)
    assert plan.enabled is True
    assert plan.horizon_split_seq is None
    assert plan.episodic_count == 3
    assert plan.consolidated_count == 0
    assert plan.consolidated_contexts == []
    assert plan.to_diagnostics() == {
        "enabled": True,
        "recent_window": 10,
        "episodic_count": 3,
        "horizon_split_seq": None,
    }


def test_build_long_horizon_plan_surfaces_accepted_old_candidates_only() -> None:
    events: list[SimpleNamespace] = [_ev(i, "tool.call.completed", {"summary": f"e{i}"}) for i in range(1, 6)]
    # Accepted via a later review event, sourced from old events (seqs 1-2).
    events.append(_candidate_created(6, "consolidation:episode:" + "0" * 24, review_status="pending", source_seqs=[1, 2]))
    events.append(
        _ev(7, "consolidation.candidate.reviewed", {"candidate_id": "consolidation:episode:" + "0" * 24, "status": "accepted"})
    )
    # Accepted but sourced from a RECENT event (seq 8 > split) -> excluded.
    events.append(_candidate_created(8, "consolidation:claim:" + "1" * 24, review_status="accepted", source_seqs=[8]))
    # Old sources but never accepted (pending) -> excluded.
    events.append(_candidate_created(9, "consolidation:procedure:" + "2" * 24, review_status="pending", source_seqs=[1]))

    # total=9, window=3 -> older = seqs 1..6, split_seq = 6.
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=3, budget=10)
    assert plan.horizon_split_seq == 6
    assert plan.episodic_count == 3
    assert plan.consolidated_count == 1
    ctx = plan.consolidated_contexts[0]
    assert ctx.source == CONSOLIDATED_LANE
    assert ctx.metadata["candidate_id"] == "consolidation:episode:" + "0" * 24
    assert ctx.metadata["assembly_lane"] == CONSOLIDATED_LANE
    assert ctx.metadata["review_status"] == "accepted"  # latest review disposition wins
    assert ctx.metadata["non_authoritative"] is True
    assert ctx.metadata["authority_status"] == "non_authoritative"
    assert ctx.metadata["source_event_citations"] == [
        "eventloom://thread-x/events/1#" + "b" * 12,
        "eventloom://thread-x/events/2#" + "b" * 12,
    ]
    # The candidate's own created-event citation is real and distinct from sources.
    assert ctx.metadata["citation"] == "eventloom://thread-x/events/6#" + "a" * 12


def test_build_long_horizon_plan_excludes_candidate_without_source_events() -> None:
    events = [_ev(i, "tool.call.completed") for i in range(1, 6)]
    events.append(_candidate_created(6, "consolidation:episode:" + "f" * 24, review_status="accepted", source_seqs=[]))
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=2, budget=10)
    # max_source_seq is None (no cited sources) -> not surfaced.
    assert plan.consolidated_count == 0


def test_build_long_horizon_plan_bounds_consolidated_by_budget() -> None:
    events = [_ev(i, "tool.call.completed") for i in range(1, 6)]
    for n in range(3):
        events.append(
            _candidate_created(10 + n, f"consolidation:episode:{str(n) * 24}", review_status="accepted", source_seqs=[1])
        )
    # total=8, window=2 -> split allows all 3; budget caps at 2.
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=2, budget=2)
    assert plan.consolidated_count == 2


def test_build_long_horizon_plan_tolerates_malformed_candidate_events() -> None:
    candidate_id = "consolidation:episode:" + "0" * 24
    # Null hash -> the candidate's own citation resolves to None; a non-dict source
    # row is skipped while a valid one is kept; a duplicate created event is ignored.
    null_hash_created = SimpleNamespace(
        seq=6,
        type="consolidation.candidate.created",
        payload={
            "candidate_id": candidate_id,
            "candidate_type": "episode",
            "summary": "episode with one valid source",
            "confidence": 0.7,
            "review_status": "accepted",
            "source_events": ["not-a-dict", {"seq": 1, "hash": "b" * 64}],
        },
        hash=None,
        thread="thread-x",
    )
    duplicate_created = _candidate_created(7, candidate_id, review_status="accepted", source_seqs=[2])
    events = [_ev(i, "tool.call.completed") for i in range(1, 6)] + [null_hash_created, duplicate_created]
    # total=7, window=2 -> older = seqs 1..5, split_seq = 5.
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=2, budget=10)
    assert plan.consolidated_count == 1
    ctx = plan.consolidated_contexts[0]
    assert ctx.metadata["citation"] is None
    assert ctx.metadata["source_event_citations"] == ["eventloom://thread-x/events/1#" + "b" * 12]


def test_checkout_long_horizon_summarizer_dedups_and_skips_unidentified() -> None:
    candidate_id = "consolidation:episode:" + "0" * 24
    citation = "eventloom://t/events/1#" + "b" * 12

    def _consolidated(cid: str | None) -> Context:
        return Context(
            content="consolidated summary",
            source=CONSOLIDATED_LANE,
            score=0.5,
            metadata={
                "assembly_lane": CONSOLIDATED_LANE,
                "candidate_id": cid,
                "candidate_type": "episode",
                "citation": citation,
                "source_event_citations": [citation, "", 7],
                "review_status": "accepted",
                "confidence": 0.7,
            },
        )

    non_lane = Context(content="x", source="graph", score=0.9, metadata={"assembly_lane": "graph"})
    missing_id = Context(content="y", source=CONSOLIDATED_LANE, score=0.1, metadata={"assembly_lane": CONSOLIDATED_LANE})
    items = _checkout_long_horizon([non_lane, _consolidated(candidate_id), _consolidated(candidate_id), missing_id])

    assert len(items) == 1  # non-lane skipped, duplicate deduped, missing id skipped
    item = items[0]
    assert item["candidate_id"] == candidate_id
    assert item["non_authoritative"] is True
    assert item["authority_status"] == "non_authoritative"
    assert item["source_event_citations"] == [citation]  # blank/non-str citations dropped
    assert item["source_event_count"] == 1


# --------------------------------------------------------------------------
# End-to-end fabric tests (real embedded MemoryFabric).
# --------------------------------------------------------------------------


async def _build_fabric(tmp_path: Path, *, enabled: bool, recent_window: int = _RECENT_WINDOW) -> MemoryFabric:
    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    fabric.settings = Settings(long_horizon_enabled=enabled, long_horizon_recent_window=recent_window)
    await fabric.connect()
    return fabric


async def _seed_old_actionable(fabric: MemoryFabric, sid: str, count: int) -> list:
    events = []
    for i in range(count):
        if i % 2 == 0:
            event = await fabric.append(
                "tool.call.completed",
                actor="agent",
                payload={
                    "tool_name": "pytest",
                    "status": "failed" if i % 4 == 0 else "succeeded",
                    "summary": f"early step {i} executed",
                },
                session_id=sid,
            )
        else:
            event = await fabric.append(
                "file.edit.applied",
                actor="agent",
                payload={"path": f"src/zaxy/module_{i}.py", "summary": f"edited module {i}"},
                session_id=sid,
            )
        events.append(event)
    return events


async def _seed_recent(fabric: MemoryFabric, sid: str, count: int) -> list:
    events = []
    for i in range(count):
        event = await fabric.append(
            "tool.call.completed",
            actor="agent",
            payload={"tool_name": "ruff", "status": "succeeded", "summary": f"recent activity {i}"},
            session_id=sid,
        )
        events.append(event)
    return events


async def _accept_all(fabric: MemoryFabric, sid: str, result: dict) -> None:
    for entry in result["events"]:
        review = build_consolidation_review_event(
            actor="reviewer",
            session_id=sid,
            candidate_id=entry["candidate_id"],
            status="accepted",
            rationale="Cited and useful consolidated knowledge for older history.",
        )
        await fabric.append(review["event_type"], actor=review["actor"], payload=review["payload"], session_id=sid)


async def _seed_long_thread_with_consolidation(fabric: MemoryFabric, sid: str) -> dict:
    """Seed an old actionable region, accept its consolidation candidates, then recent events."""
    await _seed_old_actionable(fabric, sid, 8)
    result = await fabric.propose_consolidation_candidates(session_id=sid, window_size=8)
    assert result["candidate_count"] >= 1
    await _accept_all(fabric, sid, result)
    await _seed_recent(fabric, sid, 20)
    return result


@pytest.mark.asyncio
async def test_long_horizon_engaged_surfaces_cited_consolidated_remote_tier(tmp_path) -> None:
    fabric = await _build_fabric(tmp_path, enabled=True)
    sid = "long-thread"
    try:
        result = await _seed_long_thread_with_consolidation(fabric, sid)
        checkout = await fabric.checkout_memory(
            _QUERY,
            session_id=sid,
            long_horizon=True,
            max_recent_events=_RECENT_WINDOW,
            record_reinforcement=False,
        )
    finally:
        await fabric.close()

    long_horizon = checkout.diagnostics["long_horizon"]
    assert long_horizon["enabled"] is True
    assert long_horizon["recent_window"] == _RECENT_WINDOW
    assert long_horizon["episodic_count"] == _RECENT_WINDOW
    split_seq = long_horizon["horizon_split_seq"]
    assert isinstance(split_seq, int) and split_seq >= 8
    assert long_horizon["consolidated_count"] >= 1
    assert long_horizon["non_authoritative"] is True

    items = long_horizon["items"]
    assert items
    surfaced_ids = {item["candidate_id"] for item in items}
    assert surfaced_ids
    assert surfaced_ids <= {entry["candidate_id"] for entry in result["events"]}
    for item in items:
        # Non-authoritative, cited consolidation artifact (not a raw old event).
        assert item["non_authoritative"] is True
        assert item["authority_status"] == "non_authoritative"
        assert item["candidate_id"].startswith("consolidation:")
        assert item["candidate_type"] in {"episode", "claim", "procedure"}
        assert _CITATION_RE.match(item["citation"])
        assert item["source_event_citations"]
        for citation in item["source_event_citations"]:
            assert _CITATION_RE.match(citation)
            source_seq = int(citation.split("/events/")[1].split("#")[0])
            # Every cited source is older history, beyond the episodic window.
            assert source_seq <= split_seq


@pytest.mark.asyncio
async def test_long_horizon_default_off_is_byte_identical(tmp_path) -> None:
    fabric = await _build_fabric(tmp_path, enabled=False)
    sid = "long-thread"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    recall_limit = _checkout_recall_limit(_QUERY, 10)

    async def _checkout(long_horizon):
        assembly = await fabric.assemble_context(
            _QUERY,
            session_id=sid,
            limit=10,
            recall_limit=recall_limit,
            max_recent_events=20,
            long_horizon=long_horizon,
        )
        checkout = build_memory_checkout(
            query=_QUERY,
            assembly=assembly,
            now=now,
            retrieval_profile=fabric.retrieval_profile,
            salience_floor=fabric._salience_floor,
            salience_half_life_days=fabric._salience_half_life_days,
        )
        return assembly, checkout

    try:
        await _seed_long_thread_with_consolidation(fabric, sid)
        assembly_default, checkout_default = await _checkout(None)
        assembly_off, checkout_off = await _checkout(False)
        assembly_on, checkout_on = await _checkout(True)
    finally:
        await fabric.close()

    # The disabled assembly is provably inert: both new fields hold their defaults,
    # so neither the consolidated merge nor the diagnostics branch can fire.
    assert assembly_default.long_horizon is None
    assert assembly_default.long_horizon_contexts == []
    assert assembly_off.long_horizon is None
    assert assembly_off.long_horizon_contexts == []

    # Default and explicit-False yield the identical single-tier checkout, with no
    # long_horizon diagnostics key — byte-identical to the pre-I3 contract.
    assert "long_horizon" not in checkout_default.diagnostics
    assert "long_horizon" not in checkout_off.diagnostics
    assert checkout_off.to_dict() == checkout_default.to_dict()
    # No consolidated-lane content leaks into the single-tier checkout.
    assert all("consolidation:" not in str(entry) for entry in checkout_off.provenance)
    assert all(fact.get("source_lane") != CONSOLIDATED_LANE for fact in checkout_off.current_facts)

    # Engaging the feature on the SAME seed adds the consolidated tier and changes
    # the packet — proving the two-tier path is doing real work, not a no-op.
    assert assembly_on.long_horizon is not None
    assert assembly_on.long_horizon_contexts
    assert "long_horizon" in checkout_on.diagnostics
    assert checkout_on.diagnostics["long_horizon"]["consolidated_count"] >= 1
    assert checkout_on.to_dict() != checkout_off.to_dict()


@pytest.mark.asyncio
async def test_long_horizon_short_session_is_graceful(tmp_path) -> None:
    fabric = await _build_fabric(tmp_path, enabled=True)
    sid = "short-thread"
    try:
        await _seed_old_actionable(fabric, sid, 4)  # below the recent window
        checkout = await fabric.checkout_memory(
            _QUERY, session_id=sid, long_horizon=True, record_reinforcement=False
        )
    finally:
        await fabric.close()

    long_horizon = checkout.diagnostics["long_horizon"]
    assert long_horizon["enabled"] is True
    assert long_horizon["horizon_split_seq"] is None
    assert long_horizon["episodic_count"] == 4
    assert long_horizon["consolidated_count"] == 0
    assert long_horizon["items"] == []
    assert all("consolidation:" not in str(entry) for entry in checkout.provenance)


@pytest.mark.asyncio
async def test_long_horizon_window_guarded_against_raw_recent_overlap(tmp_path) -> None:
    """A configured recent_window smaller than max_recent_events widens to the latter,
    so an event rendered in the raw Recent Events section is never ALSO consolidated."""
    fabric = await _build_fabric(tmp_path, enabled=True)
    sid = "long-thread"
    try:
        await _seed_long_thread_with_consolidation(fabric, sid)
        # configured long_horizon_recent_window (_RECENT_WINDOW=10) < max_recent_events (20)
        checkout = await fabric.checkout_memory(
            _QUERY,
            session_id=sid,
            long_horizon=True,
            max_recent_events=20,
            record_reinforcement=False,
        )
    finally:
        await fabric.close()
    long_horizon = checkout.diagnostics["long_horizon"]
    # the effective split widened to max_recent_events (20), not the configured 10
    assert long_horizon["recent_window"] == 20
    assert long_horizon["episodic_count"] <= 20
