"""Multi-agent session sharding for Eventloom.

Each agent/session gets its own JSONL log file while sharing the Neo4j graph.
This solves the single-writer bottleneck for multi-agent deployments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zaxy.config import get_settings
from zaxy.event import EventLog
from zaxy.security import eventlog_path, validate_session_id


@dataclass(frozen=True)
class Session:
    """A single agent session with its own Eventloom log."""

    session_id: str
    eventlog: EventLog


class SessionManager:
    """Manages per-session Eventloom logs for multi-agent deployments.

    Each session gets an isolated JSONL file while all sessions share
    the same Neo4j graph for querying.

    Example::

        manager = SessionManager()
        session = manager.get("agent-1")
        session.eventlog.append("goal.created", "user", {...})
    """

    def __init__(self, base_path: str | None = None) -> None:
        settings = get_settings()
        self._base = Path(base_path or settings.eventloom_path)
        self._base.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        """Get or create a session by ID."""
        safe_id = validate_session_id(session_id)
        if safe_id not in self._sessions:
            log_path = eventlog_path(self._base, safe_id)
            self._sessions[session_id] = Session(
                session_id=safe_id,
                eventlog=EventLog(str(log_path)),
            )
        return self._sessions[safe_id]

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    def handoff_summary(self, session_id: str) -> dict[str, Any]:
        """Generate a handoff summary for a specific session."""
        session = self.get(session_id)
        return session.eventlog.handoff_summary()

    def replay(self, session_id: str, from_seq: int = 1) -> Any:
        """Replay events from a specific session."""
        session = self.get(session_id)
        return session.eventlog.replay(from_seq=from_seq)
