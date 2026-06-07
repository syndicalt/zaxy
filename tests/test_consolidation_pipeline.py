from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from zaxy.consolidation_pipeline import (
    ConsolidationSegment,
    ProposedConsolidation,
    build_segment_id,
    event_type_counts,
    generate_consolidation_proposals,
    select_consolidation_segments,
)


class EventLike:
    def __init__(
        self,
        seq: int,
        event_hash: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        thread: str = "agent-1",
    ) -> None:
        self.seq = seq
        self.hash = event_hash
        self.type = event_type
        self.event_type = event_type
        self.payload = payload or {}
        self.thread = thread
        self.timestamp = datetime(2026, 6, 7, 12, min(seq, 59), tzinfo=UTC)


def test_segment_requires_cited_source_events() -> None:
    segment = ConsolidationSegment(
        session_id="agent-1",
        segment_id="segment:agent-1:000001-000003",
        event_type_counts={"tool.call.completed": 2, "file.edit.applied": 1},
        source_events=[
            {
                "seq": 1,
                "hash": "a" * 64,
                "event_type": "tool.call.completed",
                "summary": "pytest failed",
            },
            {
                "seq": 2,
                "hash": "b" * 64,
                "event_type": "file.edit.applied",
                "summary": "patched checkout",
            },
            {
                "seq": 3,
                "hash": "c" * 64,
                "event_type": "tool.call.completed",
                "summary": "pytest passed",
            },
        ],
    )

    assert segment.source_event_refs == [
        "1:" + "a" * 64,
        "2:" + "b" * 64,
        "3:" + "c" * 64,
    ]
    assert segment.source_event_count == 3


def test_segment_rejects_missing_source_citations() -> None:
    with pytest.raises(ValueError, match="hash"):
        ConsolidationSegment(
            session_id="agent-1",
            segment_id="segment:agent-1:000001-000001",
            event_type_counts={"tool.call.completed": 1},
            source_events=[
                {"seq": 1, "hash": "short", "event_type": "tool.call.completed"}
            ],
        )


def test_segment_rejects_non_session_scoped_segment_id() -> None:
    with pytest.raises(ValueError, match="session-scoped"):
        ConsolidationSegment(
            session_id="agent-1",
            segment_id="segment:other:000001-000001",
            event_type_counts={"tool.call.completed": 1},
            source_events=[
                {
                    "seq": 1,
                    "hash": "a" * 64,
                    "event_type": "tool.call.completed",
                    "summary": "pytest failed",
                }
            ],
        )


def test_segment_rejects_counts_that_do_not_match_source_events() -> None:
    with pytest.raises(ValueError, match="event_type_counts"):
        ConsolidationSegment(
            session_id="agent-1",
            segment_id="segment:agent-1:000001-000002",
            event_type_counts={"tool.call.completed": 2},
            source_events=[
                {
                    "seq": 1,
                    "hash": "a" * 64,
                    "event_type": "tool.call.completed",
                    "summary": "pytest failed",
                },
                {
                    "seq": 2,
                    "hash": "b" * 64,
                    "event_type": "file.edit.applied",
                    "summary": "patched checkout",
                },
            ],
        )


def test_proposed_consolidation_preserves_non_authoritative_boundary() -> None:
    segment = ConsolidationSegment(
        session_id="agent-1",
        segment_id="segment:agent-1:000001-000002",
        event_type_counts={"tool.call.completed": 2},
        source_events=[
            {
                "seq": 1,
                "hash": "a" * 64,
                "event_type": "tool.call.completed",
                "summary": "run failed",
            },
            {
                "seq": 2,
                "hash": "b" * 64,
                "event_type": "tool.call.completed",
                "summary": "run passed",
            },
        ],
    )
    proposal = ProposedConsolidation(
        segment=segment,
        candidate_type="episode",
        title="Test run recovery",
        summary="The agent observed a failed run and then a passing run.",
        confidence=0.72,
        method="deterministic_segment_summary_v1",
        purpose="coding",
    )

    event = proposal.to_candidate_event(actor="zaxy-consolidation")

    assert event["event_type"] == "consolidation.candidate.created"
    assert event["thread"] == "agent-1"
    assert event["payload"]["candidate_type"] == "episode"
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert event["payload"]["review_status"] == "pending"
    assert event["payload"]["source_events"] == [
        {"seq": 1, "hash": "a" * 64},
        {"seq": 2, "hash": "b" * 64},
    ]


def test_build_segment_id_is_stable_and_session_scoped() -> None:
    assert build_segment_id("agent-1", [3, 4, 9]) == "segment:agent-1:000003-000009"


def test_event_type_counts_validates_and_sorts_counts() -> None:
    assert event_type_counts(
        [
            {"seq": 2, "hash": "b" * 64, "event_type": "file.edit.applied"},
            {"seq": 1, "hash": "a" * 64, "event_type": "tool.call.completed"},
            {"seq": 3, "hash": "c" * 64, "event_type": "tool.call.completed"},
        ]
    ) == {"file.edit.applied": 1, "tool.call.completed": 2}


def test_select_consolidation_segments_groups_adjacent_relevant_events() -> None:
    events = [
        EventLike(
            1,
            "a" * 64,
            "tool.call.completed",
            {"tool_name": "pytest", "status": "failed"},
        ),
        EventLike(2, "b" * 64, "file.edit.applied", {"path": "src/zaxy/checkout.py"}),
        EventLike(
            3,
            "c" * 64,
            "tool.call.completed",
            {"tool_name": "pytest", "status": "succeeded"},
        ),
        EventLike(4, "d" * 64, "memory.checkout.completed", {"query": "unrelated"}),
    ]

    segments = select_consolidation_segments(events, session_id="agent-1", window_size=3)

    assert len(segments) == 1
    assert segments[0].segment_id == "segment:agent-1:000001-000003"
    assert segments[0].event_type_counts == {
        "file.edit.applied": 1,
        "tool.call.completed": 2,
    }
    assert [event["seq"] for event in segments[0].source_events] == [1, 2, 3]
    assert segments[0].source_events[0]["summary"] == "tool.call.completed | failed | pytest"


def test_select_consolidation_segments_ignores_non_actionable_noise() -> None:
    events = [
        EventLike(1, "a" * 64, "memory.checkout.completed", {"query": "status"}),
        EventLike(
            2,
            "b" * 64,
            "tool.call.completed",
            {"tool_name": "pytest", "status": "failed"},
        ),
    ]

    segments = select_consolidation_segments(events, session_id="agent-1", window_size=2)

    assert len(segments) == 1
    assert [event["seq"] for event in segments[0].source_events] == [2]


def test_select_consolidation_segments_filters_other_sessions() -> None:
    events = [
        EventLike(1, "a" * 64, "tool.call.completed", thread="agent-1"),
        EventLike(2, "b" * 64, "tool.call.completed", thread="agent-2"),
    ]

    segments = select_consolidation_segments(events, session_id="agent-1", window_size=8)

    assert len(segments) == 1
    assert [event["seq"] for event in segments[0].source_events] == [1]


def test_generate_consolidation_proposals_creates_episode_claim_and_procedure() -> None:
    segment = ConsolidationSegment(
        session_id="agent-1",
        segment_id="segment:agent-1:000001-000003",
        event_type_counts={"tool.call.completed": 2, "file.edit.applied": 1},
        source_events=[
            {
                "seq": 1,
                "hash": "a" * 64,
                "event_type": "tool.call.completed",
                "summary": "pytest failed",
            },
            {
                "seq": 2,
                "hash": "b" * 64,
                "event_type": "file.edit.applied",
                "summary": "patched checkout",
            },
            {
                "seq": 3,
                "hash": "c" * 64,
                "event_type": "tool.call.completed",
                "summary": "pytest succeeded",
            },
        ],
    )

    proposals = generate_consolidation_proposals([segment], purpose="coding")

    assert [proposal.candidate_type for proposal in proposals] == [
        "episode",
        "claim",
        "procedure",
    ]
    assert all(proposal.segment == segment for proposal in proposals)
    assert all(proposal.purpose == "coding" for proposal in proposals)
    assert all(proposal.method.startswith("deterministic_") for proposal in proposals)
    assert all(
        proposal.to_candidate_event(actor="zaxy-consolidation")["payload"]["review_status"]
        == "pending"
        for proposal in proposals
    )


def test_generate_consolidation_proposals_keeps_low_signal_to_episode_only() -> None:
    segment = ConsolidationSegment(
        session_id="agent-1",
        segment_id="segment:agent-1:000001-000001",
        event_type_counts={"tool.call.completed": 1},
        source_events=[
            {
                "seq": 1,
                "hash": "a" * 64,
                "event_type": "tool.call.completed",
                "summary": "listed tools",
            }
        ],
    )

    proposals = generate_consolidation_proposals([segment])

    assert [proposal.candidate_type for proposal in proposals] == ["episode"]


@pytest.mark.asyncio
async def test_memory_fabric_proposes_consolidation_candidates_from_session_events(
    tmp_path,
) -> None:
    from zaxy.core import MemoryFabric

    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    await fabric.connect()
    try:
        await fabric.append(
            "tool.call.completed",
            actor="agent",
            payload={"tool_name": "pytest", "status": "failed"},
            session_id="agent-1",
        )
        await fabric.append(
            "file.edit.applied",
            actor="agent",
            payload={"path": "src/zaxy/checkout.py"},
            session_id="agent-1",
        )
        await fabric.append(
            "tool.call.completed",
            actor="agent",
            payload={"tool_name": "pytest", "status": "succeeded"},
            session_id="agent-1",
        )

        result = await fabric.propose_consolidation_candidates(
            session_id="agent-1",
            actor="zaxy-consolidation",
            purpose="coding",
            window_size=3,
        )
        status = await fabric.consolidation_status(session_id="agent-1")
    finally:
        await fabric.close()

    assert result["session_id"] == "agent-1"
    assert result["segment_count"] == 1
    assert result["candidate_count"] == 3
    assert all(item["event_type"] == "consolidation.candidate.created" for item in result["events"])
    assert all(len(str(item["hash"])) == 64 for item in result["events"])
    assert status["candidate_count"] == 3
    assert status["pending_count"] == 3
    assert status["authority_status_counts"] == {"non_authoritative": 3}


@pytest.mark.asyncio
async def test_memory_fabric_consolidation_proposal_rerun_preserves_review_state(
    tmp_path,
) -> None:
    from zaxy.consolidation import build_consolidation_review_event
    from zaxy.core import MemoryFabric

    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    await fabric.connect()
    try:
        await fabric.append(
            "tool.call.completed",
            actor="agent",
            payload={"tool_name": "pytest", "status": "failed"},
            session_id="agent-1",
        )
        await fabric.append(
            "file.edit.applied",
            actor="agent",
            payload={"path": "src/zaxy/checkout.py"},
            session_id="agent-1",
        )
        first = await fabric.propose_consolidation_candidates(session_id="agent-1", window_size=2)
        candidate_id = first["events"][0]["candidate_id"]
        review = build_consolidation_review_event(
            actor="reviewer",
            session_id="agent-1",
            candidate_id=candidate_id,
            status="accepted",
            rationale="Cited and useful as a review disposition.",
        )
        await fabric.append(
            review["event_type"],
            actor=review["actor"],
            payload=review["payload"],
            session_id="agent-1",
        )

        second = await fabric.propose_consolidation_candidates(session_id="agent-1", window_size=2)
        status = await fabric.consolidation_status(session_id="agent-1")
    finally:
        await fabric.close()

    assert second["candidate_count"] == 0
    assert second["skipped_existing_count"] == first["candidate_count"]
    reviewed = next(item for item in status["candidates"] if item["candidate_id"] == candidate_id)
    assert reviewed["review_status"] == "accepted"
    assert reviewed["authority_status"] == "non_authoritative"
    assert status["accepted_count"] == 1


@pytest.mark.asyncio
async def test_memory_fabric_consolidation_status_never_reports_authoritative_candidates(
    tmp_path,
) -> None:
    from zaxy.core import MemoryFabric

    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    await fabric.connect()
    try:
        eventlog = fabric.session_manager.get("agent-1").eventlog
        eventlog.append(
            "consolidation.candidate.created",
            actor="unsafe-writer",
            payload={
                "candidate_id": "consolidation:claim:" + ("a" * 24),
                "candidate_type": "claim",
                "review_status": "pending",
                "authority_status": "authoritative",
            },
            thread="agent-1",
        )

        status = await fabric.consolidation_status(session_id="agent-1")
    finally:
        await fabric.close()

    assert status["candidate_count"] == 1
    assert status["candidates"][0]["authority_status"] == "non_authoritative"
    assert status["authority_status_counts"] == {"non_authoritative": 1}


@pytest.mark.asyncio
async def test_memory_fabric_consolidation_status_reports_stale_superseded_and_conflicted(
    tmp_path,
) -> None:
    from zaxy.core import MemoryFabric

    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    await fabric.connect()
    try:
        eventlog = fabric.session_manager.get("agent-1").eventlog
        eventlog.append(
            "consolidation.candidate.created",
            actor="zaxy-consolidation",
            payload={
                "candidate_id": "consolidation:claim:" + ("a" * 24),
                "candidate_type": "claim",
                "review_status": "conflicted",
                "authority_status": "non_authoritative",
                "stale": True,
                "superseded_by": "consolidation:claim:" + ("b" * 24),
                "valid_to": "2026-06-07T00:00:00Z",
            },
            thread="agent-1",
        )

        status = await fabric.consolidation_status(session_id="agent-1")
    finally:
        await fabric.close()

    assert status["conflicted_count"] == 1
    assert status["stale_count"] == 1
    assert status["superseded_count"] == 1
    assert status["valid_to_count"] == 1
