"""Read-only Eventloom memory status for git-style memory inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from zaxy.event import Event, EventLog
from zaxy.security import eventlog_path, validate_session_id


@dataclass(frozen=True)
class SessionStatus:
    """Status for one Eventloom session log."""

    session_id: str
    path: str
    event_count: int
    latest_seq: int | None
    latest_hash: str | None
    latest_type: str | None
    latest_actor: str | None
    latest_timestamp: str | None
    integrity_ok: bool
    integrity_reason: str | None


@dataclass(frozen=True)
class MemoryStatus:
    """Status for an Eventloom memory directory."""

    eventloom_path: str
    session_count: int
    total_events: int
    sessions: list[SessionStatus]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class MemoryLogEntry:
    """One Eventloom event rendered for git-style memory log inspection."""

    session_id: str
    seq: int
    hash: str
    timestamp: str
    type: str
    actor: str
    summary: str
    integrity_ok: bool


@dataclass(frozen=True)
class MemoryLog:
    """Recent Eventloom events across one or more sessions."""

    eventloom_path: str
    session_id: str | None
    limit: int
    entries: list[MemoryLogEntry]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""
        return asdict(self)


def inspect_memory_status(eventloom_path: str | Path) -> MemoryStatus:
    """Inspect Eventloom JSONL session logs without requiring graph services."""
    base = Path(eventloom_path).resolve()
    sessions = [_inspect_log(path) for path in _eventlog_paths(base)]
    return MemoryStatus(
        eventloom_path=str(base),
        session_count=len(sessions),
        total_events=sum(session.event_count for session in sessions),
        sessions=sessions,
    )


def inspect_memory_log(
    eventloom_path: str | Path,
    *,
    session_id: str | None = None,
    limit: int = 20,
) -> MemoryLog:
    """Return recent Eventloom events without requiring graph services."""
    base = Path(eventloom_path).resolve()
    paths = [_session_log_path(base, session_id)] if session_id else _eventlog_paths(base)
    entries: list[MemoryLogEntry] = []
    for path in paths:
        log = EventLog(path)
        integrity = log.verify()
        for event in log.read_all():
            entries.append(_log_entry(path.stem, event, integrity_ok=integrity.ok))
    entries.sort(key=lambda entry: (entry.timestamp, entry.session_id, entry.seq), reverse=True)
    safe_limit = max(0, limit)
    return MemoryLog(
        eventloom_path=str(base),
        session_id=session_id,
        limit=safe_limit,
        entries=entries[:safe_limit],
    )


def format_memory_status(status: MemoryStatus) -> str:
    """Format memory status for humans."""
    lines = [
        f"Eventloom: {status.eventloom_path}",
        f"Sessions: {status.session_count}",
        f"Total events: {status.total_events}",
    ]
    if not status.sessions:
        return "\n".join(lines)
    lines.append("")
    lines.append("Session logs:")
    for session in status.sessions:
        integrity = "OK" if session.integrity_ok else "FAILED"
        latest = session.latest_seq if session.latest_seq is not None else "-"
        short_hash = session.latest_hash[:12] if session.latest_hash else "-"
        latest_type = session.latest_type or "-"
        lines.append(
            f"  {session.session_id}: events={session.event_count} "
            f"latest={latest} hash={short_hash} type={latest_type} integrity={integrity}"
        )
        if session.integrity_reason:
            lines.append(f"    reason={session.integrity_reason}")
    return "\n".join(lines)


def format_memory_log(memory_log: MemoryLog) -> str:
    """Format recent Eventloom events for humans."""
    if not memory_log.entries:
        return "No memory events found."
    lines: list[str] = []
    for entry in memory_log.entries:
        lines.append(
            f"{entry.session_id} [{entry.seq}] {entry.hash[:12]} "
            f"{entry.timestamp} {entry.type} by {entry.actor}"
        )
        if entry.summary:
            lines.append(f"  {entry.summary}")
        if not entry.integrity_ok:
            lines.append("  integrity=FAILED")
    return "\n".join(lines)


def _eventlog_paths(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    if not base.exists():
        return []
    return sorted(path for path in base.glob("*.jsonl") if path.is_file())


def _session_log_path(base: Path, session_id: str | None) -> Path:
    if base.is_file():
        return base
    if session_id is None:
        raise ValueError("session_id is required")
    return eventlog_path(base, validate_session_id(session_id))


def _inspect_log(path: Path) -> SessionStatus:
    log = EventLog(path)
    events = log.read_all()
    integrity = log.verify()
    latest = events[-1] if events else None
    return SessionStatus(
        session_id=path.stem,
        path=str(path.resolve()),
        event_count=len(events),
        latest_seq=latest.seq if latest else None,
        latest_hash=latest.hash if latest else None,
        latest_type=latest.type if latest else None,
        latest_actor=latest.actor if latest else None,
        latest_timestamp=latest.timestamp if latest else None,
        integrity_ok=integrity.ok,
        integrity_reason=integrity.broken_reason,
    )


def _log_entry(session_id: str, event: Event, *, integrity_ok: bool) -> MemoryLogEntry:
    return MemoryLogEntry(
        session_id=session_id,
        seq=event.seq,
        hash=event.hash,
        timestamp=event.timestamp,
        type=event.type,
        actor=event.actor,
        summary=_event_summary(event),
        integrity_ok=integrity_ok,
    )


def _event_summary(event: Event) -> str:
    for key in ("summary", "decision", "title", "content", "text", "task"):
        value = event.payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    if event.payload:
        return ", ".join(sorted(str(key) for key in event.payload))
    return ""
