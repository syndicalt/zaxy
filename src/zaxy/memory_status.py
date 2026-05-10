"""Read-only Eventloom memory status for git-style memory inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from zaxy.event import EventLog


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


def _eventlog_paths(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    if not base.exists():
        return []
    return sorted(path for path in base.glob("*.jsonl") if path.is_file())


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
