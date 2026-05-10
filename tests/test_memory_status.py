"""Tests for read-only Eventloom memory status inspection."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.event import EventLog
from zaxy.memory_status import (
    format_memory_diff,
    format_memory_log,
    format_memory_status,
    inspect_memory_diff,
    inspect_memory_log,
    inspect_memory_status,
)


def test_inspect_memory_status_reports_integrity_failure(tmp_path: Path) -> None:
    """Status should expose hash-chain failures without raising."""
    path = tmp_path / ".eventloom" / "agent.jsonl"
    event = EventLog(path).append(
        "goal.created",
        actor="user",
        payload={"title": "Ship it"},
        thread="agent",
    )
    record = json.loads(path.read_text(encoding="utf-8").strip())
    record["payload"]["title"] = "Tampered"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    status = inspect_memory_status(path.parent)

    assert status.session_count == 1
    assert status.total_events == 1
    assert status.sessions[0].session_id == "agent"
    assert status.sessions[0].latest_hash == event.hash
    assert status.sessions[0].integrity_ok is False
    assert status.sessions[0].integrity_reason == "Event 1 hash mismatch"
    assert "integrity=FAILED" in format_memory_status(status)


def test_inspect_memory_status_accepts_single_log_file(tmp_path: Path) -> None:
    """Status can inspect one JSONL log path directly."""
    path = tmp_path / "worker.jsonl"
    EventLog(path).append(
        "task.completed",
        actor="assistant",
        payload={"summary": "Indexed docs"},
        thread="worker",
    )

    status = inspect_memory_status(path)

    assert status.eventloom_path == str(path.resolve())
    assert status.session_count == 1
    assert status.sessions[0].session_id == "worker"
    assert status.sessions[0].latest_type == "task.completed"


def test_inspect_memory_log_orders_recent_events_across_sessions(tmp_path: Path) -> None:
    """Memory log should show newest events first across Eventloom sessions."""
    EventLog(tmp_path / ".eventloom" / "agent-a.jsonl").append(
        "goal.created",
        actor="user",
        payload={"title": "Ship it"},
        thread="agent-a",
    )
    EventLog(tmp_path / ".eventloom" / "agent-b.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use source-aware assembly."},
        thread="agent-b",
    )

    log = inspect_memory_log(tmp_path / ".eventloom", limit=2)

    assert [entry.session_id for entry in log.entries] == ["agent-b", "agent-a"]
    assert log.entries[0].summary == "Use source-aware assembly."
    assert log.entries[1].summary == "Ship it"
    assert log.entries[0].integrity_ok is True


def test_inspect_memory_log_filters_session_and_limit(tmp_path: Path) -> None:
    """Memory log should support one-session inspection with a bounded result count."""
    agent_log = EventLog(tmp_path / ".eventloom" / "agent.jsonl")
    first = agent_log.append(
        "transcript.turn",
        actor="user",
        payload={"content": "Older turn"},
        thread="agent",
    )
    second = agent_log.append(
        "task.completed",
        actor="assistant",
        payload={"summary": "Finished source-aware status."},
        thread="agent",
    )
    EventLog(tmp_path / ".eventloom" / "other.jsonl").append(
        "goal.created",
        actor="user",
        payload={"title": "Ignore me"},
        thread="other",
    )

    log = inspect_memory_log(tmp_path / ".eventloom", session_id="agent", limit=1)

    assert first.seq == 1
    assert len(log.entries) == 1
    assert log.entries[0].seq == second.seq
    assert log.entries[0].summary == "Finished source-aware status."


def test_format_memory_log_prints_git_style_lines(tmp_path: Path) -> None:
    """Human memory log output should be compact and source-oriented."""
    event = EventLog(tmp_path / "agent.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use Eventloom as source of truth."},
        thread="agent",
    )

    output = format_memory_log(inspect_memory_log(tmp_path / "agent.jsonl", limit=10))

    assert f"agent [{event.seq}] {event.hash[:12]}" in output
    assert "decision.recorded by assistant" in output
    assert "Use Eventloom as source of truth." in output


def test_inspect_memory_diff_returns_events_in_sequence_range(tmp_path: Path) -> None:
    """Memory diff should expose added Eventloom events in a seq range."""
    log = EventLog(tmp_path / ".eventloom" / "agent.jsonl")
    log.append("goal.created", actor="user", payload={"title": "Older"}, thread="agent")
    decision = log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use event-level diff."},
        thread="agent",
    )
    task = log.append(
        "task.completed",
        actor="codex",
        payload={"summary": "Added memory diff."},
        thread="agent",
    )

    diff = inspect_memory_diff(
        tmp_path / ".eventloom",
        session_id="agent",
        from_seq=2,
        to_seq=3,
    )

    assert diff.session_id == "agent"
    assert diff.from_seq == 2
    assert diff.to_seq == 3
    assert diff.integrity_ok is True
    assert [entry.seq for entry in diff.added] == [decision.seq, task.seq]
    assert diff.added[0].summary == "Use event-level diff."


def test_format_memory_diff_prints_added_events(tmp_path: Path) -> None:
    """Human diff output should mark events as added without semantic overclaiming."""
    event = EventLog(tmp_path / "agent.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Diff immutable events."},
        thread="agent",
    )

    output = format_memory_diff(
        inspect_memory_diff(tmp_path / "agent.jsonl", session_id=None, from_seq=1, to_seq=1)
    )

    assert f"agent +[{event.seq}] {event.hash[:12]} decision.recorded by assistant" in output
    assert "Diff immutable events." in output


def test_inspect_memory_diff_rejects_invalid_ranges(tmp_path: Path) -> None:
    """Diff ranges should be explicit and ordered."""
    try:
        inspect_memory_diff(tmp_path / ".eventloom", session_id="agent", from_seq=3, to_seq=2)
    except ValueError as exc:
        assert "from_seq must be <= to_seq" in str(exc)
    else:
        raise AssertionError("expected invalid range to fail")
