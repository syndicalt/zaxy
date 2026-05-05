"""Tests for zaxy.session — multi-agent session sharding.

Tests cover SessionManager lifecycle, per-session isolation, and
handoff/replay operations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from zaxy.event import EventLog
from zaxy.session import Session, SessionManager


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@pytest.fixture
def tmp_base() -> str:
    """Return a temporary directory for session logs."""
    with tempfile.TemporaryDirectory() as td:
        yield td


# ------------------------------------------------------------------
# SessionManager
# ------------------------------------------------------------------

class TestSessionManagerInit:
    def test_creates_base_directory(self, tmp_base: str) -> None:
        """SessionManager should create the base path if missing."""
        nested = Path(tmp_base) / "deep" / "nested"
        assert not nested.exists()
        SessionManager(base_path=str(nested))
        assert nested.exists()

    def test_uses_default_eventloom_path(self, monkeypatch: pytest.MonkeyPatch, tmp_base: str) -> None:
        """When no base_path is given, use settings.eventloom_path."""
        import zaxy.config
        settings = zaxy.config.get_settings()
        monkeypatch.setattr(settings, "eventloom_path", tmp_base)
        mgr = SessionManager()
        assert mgr._base == Path(tmp_base)


class TestSessionManagerGet:
    def test_creates_session(self, tmp_base: str) -> None:
        """get() should create a new session with its own EventLog."""
        mgr = SessionManager(base_path=tmp_base)
        session = mgr.get("agent-1")
        assert isinstance(session, Session)
        assert session.session_id == "agent-1"
        assert isinstance(session.eventlog, EventLog)

    def test_returns_existing_session(self, tmp_base: str) -> None:
        """get() should return the same instance for repeated calls."""
        mgr = SessionManager(base_path=tmp_base)
        s1 = mgr.get("agent-1")
        s2 = mgr.get("agent-1")
        assert s1 is s2

    def test_isolates_sessions(self, tmp_base: str) -> None:
        """Different session IDs should get different EventLogs."""
        mgr = SessionManager(base_path=tmp_base)
        s1 = mgr.get("agent-1")
        s2 = mgr.get("agent-2")
        assert s1.eventlog is not s2.eventlog
        assert s1.eventlog.path != s2.eventlog.path

    def test_session_log_file_path(self, tmp_base: str) -> None:
        """Session log file should be named <session_id>.jsonl."""
        mgr = SessionManager(base_path=tmp_base)
        session = mgr.get("my-session")
        expected = Path(tmp_base) / "my-session.jsonl"
        assert Path(session.eventlog.path) == expected

    @pytest.mark.parametrize("session_id", ["../escape", "nested/path", "", "x" * 129])
    def test_rejects_unsafe_session_ids(self, tmp_base: str, session_id: str) -> None:
        """Session IDs must not escape the configured Eventloom base path."""
        mgr = SessionManager(base_path=tmp_base)
        with pytest.raises(ValueError, match="Invalid session_id"):
            mgr.get(session_id)


class TestSessionManagerListSessions:
    def test_empty_initially(self, tmp_base: str) -> None:
        """list_sessions should return empty list when no sessions exist."""
        mgr = SessionManager(base_path=tmp_base)
        assert mgr.list_sessions() == []

    def test_lists_created_sessions(self, tmp_base: str) -> None:
        """list_sessions should reflect all get() calls."""
        mgr = SessionManager(base_path=tmp_base)
        mgr.get("a")
        mgr.get("b")
        assert set(mgr.list_sessions()) == {"a", "b"}


class TestSessionManagerHandoffSummary:
    def test_returns_summary(self, tmp_base: str) -> None:
        """handoff_summary should delegate to the session's EventLog."""
        mgr = SessionManager(base_path=tmp_base)
        session = mgr.get("agent-1")
        # Seed with an event so summary has data
        session.eventlog.append("goal.created", "user", {"title": "t1"})
        summary = mgr.handoff_summary("agent-1")
        assert isinstance(summary, dict)
        assert summary["goals"] == ["t1"]
        assert summary["event_count"] == 1


class TestSessionManagerReplay:
    def test_replays_session_events(self, tmp_base: str) -> None:
        """replay should return events from the session's EventLog."""
        mgr = SessionManager(base_path=tmp_base)
        session = mgr.get("agent-1")
        session.eventlog.append("goal.created", "user", {"title": "t1"})
        result = mgr.replay("agent-1")
        assert len(result.events) == 1
        assert result.events[0].type == "goal.created"

    def test_replay_from_seq(self, tmp_base: str) -> None:
        """replay should support from_seq parameter."""
        mgr = SessionManager(base_path=tmp_base)
        session = mgr.get("agent-1")
        session.eventlog.append("goal.created", "user", {"title": "t1"})
        session.eventlog.append("task.proposed", "user", {"task_id": "t2"})
        result = mgr.replay("agent-1", from_seq=2)
        assert len(result.events) == 1
        assert result.events[0].type == "task.proposed"
