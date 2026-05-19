"""Tests for agent recall hardening and Zaxy reminder policy."""

from __future__ import annotations

from zaxy.event import EventLog
from zaxy.memory_persistence import (
    build_memory_reminder,
    inspect_memory_persistence,
    record_memory_activity,
    suggest_memory_reminder,
)


def test_memory_persistence_detects_stale_long_session(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Long sessions should request a checkout reminder when memory was not used recently."""
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "agent.jsonl")
    record_memory_activity(eventloom, session_id="agent", activity="bootstrap", source="cli")
    for index in range(10):
        log.append("transcript.turn", actor="codex", payload={"content": f"turn {index}"}, thread="agent")

    reminder = suggest_memory_reminder(
        eventloom,
        session_id="agent",
        trigger="checkpoint",
        reason="interval",
        turn_count=10,
        current_task="continue",
    )

    assert reminder is not None
    assert reminder["event_type"] == "memory.reminder.suggested"
    assert reminder["payload"]["recommended_tool"] == "memory_checkout"
    assert "stale_memory_activity" in reminder["payload"]["reasons"]
    assert "where_are_we_query" in reminder["payload"]["reasons"]
    assert "Call memory_checkout" in build_memory_reminder(reminder["payload"])


def test_memory_persistence_suppresses_recent_checkout_reminder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Recent checkout should keep hooks from generating noisy reminders."""
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "agent.jsonl")
    for index in range(3):
        log.append("transcript.turn", actor="codex", payload={"content": f"turn {index}"}, thread="agent")
    record_memory_activity(eventloom, session_id="agent", activity="checkout", source="mcp")

    reminder = suggest_memory_reminder(
        eventloom,
        session_id="agent",
        trigger="checkpoint",
        reason="interval",
        turn_count=4,
        current_task="implement feature",
    )

    assert reminder is None
    status = inspect_memory_persistence(eventloom, session_id="agent")
    assert status["last_checkout_seq"] == 4
    assert status["stale"] is False


def test_memory_persistence_forces_resume_and_compaction_boundaries(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Resume and compaction boundaries should always reintroduce checkout guidance."""
    eventloom = tmp_path / ".eventloom"
    record_memory_activity(eventloom, session_id="agent", activity="checkout", source="cli")

    reminder = suggest_memory_reminder(
        eventloom,
        session_id="agent",
        trigger="precompact",
        reason="compaction",
        current_task="roadmap",
    )

    assert reminder is not None
    assert "context_boundary" in reminder["payload"]["reasons"]
    assert reminder["payload"]["query"] == "roadmap"
