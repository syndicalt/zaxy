"""Tests for read-only Eventloom memory status inspection."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.event import EventLog
from zaxy.memory_status import format_memory_status, inspect_memory_status


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
