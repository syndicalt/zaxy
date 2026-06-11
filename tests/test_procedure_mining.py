from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zaxy.event import Event, EventLog
from zaxy.lifecycle import build_command_completed_event, build_tool_call_completed_event
from zaxy.procedure_mining import (
    MAX_PROCEDURE_LENGTH,
    MINING_MIN_SUPPORT,
    PROCEDURE_MINING_METHOD,
    TRACE_EVENT_TYPES,
    build_procedure_proposal,
    confidence_from_support,
    extract_session_traces,
    mine_and_propose,
    mine_procedures,
)


def _append_tool_call(
    log: EventLog,
    *,
    session_id: str,
    tool_name: str,
    status: str = "succeeded",
) -> Event:
    spec = build_tool_call_completed_event(
        tool_name=tool_name,
        status=status,
        session_id=session_id,
        arguments={"query": "redacted"},
        result_summary=f"{tool_name} {status}",
    )
    return log.append(
        spec["event_type"],
        actor=spec["actor"],
        payload=spec["payload"],
        thread=session_id,
    )


def _append_command(
    log: EventLog,
    *,
    session_id: str,
    command: str,
    exit_code: int,
) -> Event:
    spec = build_command_completed_event(
        command=command,
        exit_code=exit_code,
        session_id=session_id,
        duration_ms=12,
        stdout="ok" if exit_code == 0 else "",
        stderr="" if exit_code == 0 else "boom",
    )
    return log.append(
        spec["event_type"],
        actor=spec["actor"],
        payload=spec["payload"],
        thread=session_id,
    )


def _seed_sequence(log: EventLog, session_id: str, tool_names: list[str]) -> list[Event]:
    return [
        _append_tool_call(log, session_id=session_id, tool_name=tool_name)
        for tool_name in tool_names
    ]


def _log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "agent.jsonl")


def _candidate_events(log: EventLog) -> list[Event]:
    return [event for event in log.read_all() if event.type == "consolidation.candidate.created"]


def test_recurring_sequence_across_two_sessions_is_mined(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _seed_sequence(log, "agent-1", ["memory_query", "memory_checkout", "memory_feedback"])
    _seed_sequence(log, "agent-2", ["memory_query", "memory_checkout", "memory_feedback"])

    mined = mine_procedures(extract_session_traces(log.read_all()))

    assert len(mined) == 1
    procedure = mined[0]
    assert procedure.steps == ("tool:memory_query", "tool:memory_checkout", "tool:memory_feedback")
    assert procedure.support_sessions == ("agent-1", "agent-2")
    assert procedure.support == 2
    events_by_seq = {event.seq: event for event in log.read_all()}
    for occurrence in procedure.occurrences:
        for step in occurrence.steps:
            assert events_by_seq[step.seq].hash == step.hash
            assert events_by_seq[step.seq].thread == occurrence.session_id


def test_sequence_in_a_single_session_is_below_support_threshold(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _seed_sequence(log, "agent-1", ["memory_query", "memory_checkout", "memory_feedback"])

    assert mine_procedures(extract_session_traces(log.read_all())) == []


def test_failed_steps_break_sequences(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for session_id in ("agent-1", "agent-2"):
        _append_tool_call(log, session_id=session_id, tool_name="memory_query")
        _append_tool_call(log, session_id=session_id, tool_name="memory_checkout", status="failed")
        _append_tool_call(log, session_id=session_id, tool_name="memory_feedback")

    assert mine_procedures(extract_session_traces(log.read_all())) == []


def test_failed_command_exit_codes_break_sequences(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for session_id in ("agent-1", "agent-2"):
        _append_command(log, session_id=session_id, command="pytest -q", exit_code=0)
        _append_command(log, session_id=session_id, command="ruff check src", exit_code=1)
        _append_command(log, session_id=session_id, command="git commit", exit_code=0)

    mined = mine_procedures(extract_session_traces(log.read_all()))

    assert mined == []


def test_commands_and_tool_calls_mine_together_with_normalized_names(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for session_id in ("agent-1", "agent-2"):
        _append_command(log, session_id=session_id, command="pytest -q --no-cov", exit_code=0)
        _append_tool_call(log, session_id=session_id, tool_name="memory_feedback")

    mined = mine_procedures(extract_session_traces(log.read_all()))

    assert [procedure.steps for procedure in mined] == [("command:pytest", "tool:memory_feedback")]


def test_nested_ngrams_with_identical_support_are_subsumed(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _seed_sequence(log, "agent-1", ["a_tool", "b_tool", "c_tool"])
    _seed_sequence(log, "agent-2", ["a_tool", "b_tool", "c_tool"])

    mined = mine_procedures(extract_session_traces(log.read_all()))

    assert [procedure.steps for procedure in mined] == [("tool:a_tool", "tool:b_tool", "tool:c_tool")]


def test_shorter_ngram_with_wider_support_survives_subsumption(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _seed_sequence(log, "agent-1", ["a_tool", "b_tool", "c_tool"])
    _seed_sequence(log, "agent-2", ["a_tool", "b_tool", "c_tool"])
    _seed_sequence(log, "agent-3", ["a_tool", "b_tool"])

    mined = mine_procedures(extract_session_traces(log.read_all()))

    assert [(procedure.steps, procedure.support) for procedure in mined] == [
        (("tool:a_tool", "tool:b_tool"), 3),
        (("tool:a_tool", "tool:b_tool", "tool:c_tool"), 2),
    ]


def test_mined_output_ordering_is_deterministic(tmp_path: Path) -> None:
    log = _log(tmp_path)
    # (x_tool, y_tool) and (a_tool, b_tool) both support 2; lexicographic tie-break.
    _seed_sequence(log, "agent-1", ["x_tool", "y_tool"])
    _append_tool_call(log, session_id="agent-1", tool_name="x_tool", status="failed")
    _seed_sequence(log, "agent-1", ["a_tool", "b_tool"])
    _seed_sequence(log, "agent-2", ["x_tool", "y_tool"])
    _append_tool_call(log, session_id="agent-2", tool_name="x_tool", status="failed")
    _seed_sequence(log, "agent-2", ["a_tool", "b_tool"])
    # (a_tool, b_tool) also occurs in agent-3: support 3 ranks first.
    _seed_sequence(log, "agent-3", ["a_tool", "b_tool"])

    mined = mine_procedures(extract_session_traces(log.read_all()))

    assert [(procedure.steps, procedure.support) for procedure in mined] == [
        (("tool:a_tool", "tool:b_tool"), 3),
        (("tool:x_tool", "tool:y_tool"), 2),
    ]


def test_max_length_is_respected(tmp_path: Path) -> None:
    log = _log(tmp_path)
    names = ["a_tool", "b_tool", "c_tool", "d_tool"]
    _seed_sequence(log, "agent-1", names)
    _seed_sequence(log, "agent-2", names)

    mined = mine_procedures(extract_session_traces(log.read_all()), max_length=3)

    assert mined
    assert all(len(procedure.steps) <= 3 for procedure in mined)
    assert {procedure.steps for procedure in mined} == {
        ("tool:a_tool", "tool:b_tool", "tool:c_tool"),
        ("tool:b_tool", "tool:c_tool", "tool:d_tool"),
    }


def test_default_max_procedure_length_bounds_mined_ngrams(tmp_path: Path) -> None:
    log = _log(tmp_path)
    names = [f"tool_{index:02d}" for index in range(MAX_PROCEDURE_LENGTH + 2)]
    _seed_sequence(log, "agent-1", names)
    _seed_sequence(log, "agent-2", names)

    mined = mine_procedures(extract_session_traces(log.read_all()))

    assert mined
    assert max(len(procedure.steps) for procedure in mined) == MAX_PROCEDURE_LENGTH


def test_session_ids_filter_and_parameter_validation(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _seed_sequence(log, "agent-1", ["a_tool", "b_tool"])
    _seed_sequence(log, "agent-2", ["a_tool", "b_tool"])
    _seed_sequence(log, "agent-3", ["a_tool", "b_tool"])

    traces = extract_session_traces(log.read_all(), session_ids=["agent-1", "agent-2"])
    assert sorted(traces) == ["agent-1", "agent-2"]

    with pytest.raises(ValueError, match="session_id"):
        extract_session_traces(log.read_all(), session_ids=["../escape"])
    with pytest.raises(ValueError, match="min_support"):
        mine_procedures(traces, min_support=1)
    with pytest.raises(ValueError, match="max_length"):
        mine_procedures(traces, max_length=1)


def test_extract_session_traces_rejects_malformed_trace_envelopes() -> None:
    class BrokenEvent:
        type = "tool.call.completed"
        thread = "agent-1"
        seq = 1
        hash = "not-a-hash"
        payload: dict[str, Any] = {"tool_name": "memory_query", "status": "succeeded"}

    with pytest.raises(ValueError, match="hash"):
        extract_session_traces([BrokenEvent()])


def test_unnameable_trace_payloads_break_sequences_instead_of_merging(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for session_id in ("agent-1", "agent-2"):
        _append_tool_call(log, session_id=session_id, tool_name="memory_query")
        # Status claims success but the payload has no tool name: honest break.
        log.append(
            "tool.call.completed",
            actor="zaxy",
            payload={"status": "succeeded", "session_id": session_id},
            thread=session_id,
        )
        _append_tool_call(log, session_id=session_id, tool_name="memory_feedback")

    assert mine_procedures(extract_session_traces(log.read_all())) == []


def test_confidence_mapping_from_support() -> None:
    assert confidence_from_support(2) == 0.5
    assert confidence_from_support(3) == 0.55
    assert confidence_from_support(9) == 0.85
    assert confidence_from_support(20) == 0.85
    with pytest.raises(ValueError, match="support"):
        confidence_from_support(1)


def test_mine_and_propose_appends_review_pending_procedure_candidates(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _seed_sequence(log, "agent-1", ["memory_query", "memory_checkout", "memory_feedback"])
    _seed_sequence(log, "agent-2", ["memory_query", "memory_checkout", "memory_feedback"])

    summary = mine_and_propose(log, actor="zaxy-procedure-miner")

    assert summary.session_ids == ("agent-1", "agent-2")
    assert summary.mined_count == 1
    assert summary.appended_count == 1
    assert summary.skipped_duplicate_count == 0

    candidates = _candidate_events(log)
    assert len(candidates) == 1
    candidate = candidates[0]
    payload = candidate.payload
    assert candidate.actor == "zaxy-procedure-miner"
    assert candidate.thread == "agent-1"  # earliest contributing occurrence
    assert payload["candidate_type"] == "procedure"
    assert payload["review_status"] == "pending"
    assert payload["authority_status"] == "non_authoritative"
    assert payload["method"] == PROCEDURE_MINING_METHOD
    assert payload["confidence"] == 0.5
    assert "tool:memory_query -> tool:memory_checkout -> tool:memory_feedback" in payload["title"]
    assert "across 2 sessions" in payload["summary"]
    assert payload["candidate_id"] == summary.appended[0].candidate_id

    # Every citation resolves to a real trace event in the log.
    events_by_seq = {event.seq: event for event in log.read_all()}
    assert len(payload["source_events"]) == 6
    cited_threads = set()
    for citation in payload["source_events"]:
        cited = events_by_seq[citation["seq"]]
        assert cited.hash == citation["hash"]
        assert cited.type in TRACE_EVENT_TYPES
        cited_threads.add(cited.thread)
    assert cited_threads == {"agent-1", "agent-2"}


def test_mine_and_propose_is_idempotent_over_the_same_log(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _seed_sequence(log, "agent-1", ["memory_query", "memory_checkout"])
    _seed_sequence(log, "agent-2", ["memory_query", "memory_checkout"])

    first = mine_and_propose(log)
    second = mine_and_propose(log)

    assert first.appended_count == 1
    assert second.mined_count == first.mined_count
    assert second.appended_count == 0
    assert second.skipped_duplicate_count == first.appended_count
    assert second.skipped_candidate_ids == tuple(
        proposal.candidate_id for proposal in first.appended
    )
    assert len(_candidate_events(log)) == first.appended_count


def test_citation_cap_keeps_earliest_occurrence_per_session(tmp_path: Path) -> None:
    log = _log(tmp_path)
    first_per_session: dict[str, list[Event]] = {}
    for session_id in ("agent-1", "agent-2"):
        for repeat in range(3):
            events = _seed_sequence(log, session_id, ["a_tool", "b_tool"])
            if repeat == 0:
                first_per_session[session_id] = events
            _append_tool_call(log, session_id=session_id, tool_name="break_tool", status="failed")

    summary = mine_and_propose(log, max_cited_occurrences=2)

    assert summary.mined_count == 1
    assert len(summary.mined[0].occurrences) == 6
    payload = _candidate_events(log)[0].payload
    # 2 cited occurrences x 2 steps each: the earliest occurrence per session.
    assert len(payload["source_events"]) == 4
    cited_seqs = {citation["seq"] for citation in payload["source_events"]}
    expected_seqs = {
        event.seq for events in first_per_session.values() for event in events
    }
    assert cited_seqs == expected_seqs


def test_citation_cap_fills_remaining_slots_in_seq_order(tmp_path: Path) -> None:
    log = _log(tmp_path)
    for session_id in ("agent-1", "agent-2"):
        for _ in range(3):
            _seed_sequence(log, session_id, ["a_tool", "b_tool"])
            _append_tool_call(log, session_id=session_id, tool_name="break_tool", status="failed")

    proposal = build_procedure_proposal(
        mine_procedures(extract_session_traces(log.read_all()))[0],
        max_cited_occurrences=3,
    )
    event_spec = proposal.to_candidate_event(actor="zaxy-procedure-miner")

    citations = event_spec["payload"]["source_events"]
    assert len(citations) == 6
    seqs = [citation["seq"] for citation in citations]
    assert seqs == sorted(seqs)


@pytest.mark.asyncio
async def test_mined_proposal_flows_through_existing_review_acceptance(tmp_path: Path) -> None:
    from zaxy.consolidation import build_consolidation_review_event
    from zaxy.core import MemoryFabric

    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    await fabric.connect()
    try:
        eventlog = fabric.session_manager.get("agent-1").eventlog
        _seed_sequence(eventlog, "agent-1", ["memory_query", "memory_checkout"])
        _seed_sequence(eventlog, "agent-2", ["memory_query", "memory_checkout"])

        summary = mine_and_propose(eventlog)
        assert summary.appended_count == 1
        candidate_id = summary.appended[0].candidate_id

        review = build_consolidation_review_event(
            actor="zaxy-reviewer",
            session_id="agent-1",
            candidate_id=candidate_id,
            status="accepted",
            rationale="Recurring cited sequence verified by reviewer.",
        )
        await fabric.append(
            review["event_type"],
            actor=review["actor"],
            payload=review["payload"],
            session_id="agent-1",
        )

        status = await fabric.consolidation_status(session_id="agent-1")
    finally:
        await fabric.close()

    assert status["candidate_count"] == 1
    assert status["accepted_count"] == 1
    accepted = status["candidates"][0]
    assert accepted["candidate_id"] == candidate_id
    assert accepted["candidate_type"] == "procedure"
    assert accepted["review_status"] == "accepted"
    assert accepted["authority_status"] == "non_authoritative"


def test_mine_and_propose_with_no_recurring_sequences_appends_nothing(tmp_path: Path) -> None:
    log = _log(tmp_path)
    _seed_sequence(log, "agent-1", ["memory_query", "memory_checkout"])

    summary = mine_and_propose(log)

    assert summary.mined_count == 0
    assert summary.appended_count == 0
    assert summary.skipped_duplicate_count == 0
    assert _candidate_events(log) == []
    assert MINING_MIN_SUPPORT == 2
