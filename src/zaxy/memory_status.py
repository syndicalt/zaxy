"""Read-only Eventloom memory status for git-style memory inspection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import ValidationError

from zaxy.event import Event, EventLog
from zaxy.security import eventlog_path, validate_session_id

_REQUIRED_ZAXY_EVENT_FIELDS = frozenset({"seq", "timestamp", "type", "actor", "payload", "hash"})
_EVENTLOOM_V1_SHAPE_FIELDS = frozenset({"id", "type", "actorId", "threadId", "payload", "integrity"})


@dataclass(frozen=True)
class SkippedLog:
    """A JSONL file skipped during a broad Eventloom directory scan."""

    path: str
    reason: str


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
    skipped_logs: list[SkippedLog] = field(default_factory=list)

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
    skipped_logs: list[SkippedLog] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class MemoryDiff:
    """Event-level diff for one Eventloom session/log range."""

    eventloom_path: str
    session_id: str | None
    from_seq: int
    to_seq: int
    integrity_ok: bool
    integrity_reason: str | None
    added: list[MemoryLogEntry]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""
        return asdict(self)


def inspect_memory_status(eventloom_path: str | Path) -> MemoryStatus:
    """Inspect Eventloom JSONL session logs without requiring graph services."""
    base = Path(eventloom_path).resolve()
    paths, skipped_logs = _eventlog_paths(base)
    sessions: list[SessionStatus] = []
    for path in paths:
        try:
            sessions.append(_inspect_log(path))
        except ValidationError as exc:
            skipped_logs.append(_skipped_log(path, _validation_error_reason(exc)))
    return MemoryStatus(
        eventloom_path=str(base),
        session_count=len(sessions),
        total_events=sum(session.event_count for session in sessions),
        sessions=sessions,
        skipped_logs=skipped_logs,
    )


def inspect_memory_log(
    eventloom_path: str | Path,
    *,
    session_id: str | None = None,
    limit: int = 20,
) -> MemoryLog:
    """Return recent Eventloom events without requiring graph services."""
    base = Path(eventloom_path).resolve()
    if session_id:
        paths = [_session_log_path(base, session_id)]
        skipped_logs: list[SkippedLog] = []
    else:
        paths, skipped_logs = _eventlog_paths(base)
    entries: list[MemoryLogEntry] = []
    for path in paths:
        log = EventLog(path)
        try:
            integrity = log.verify()
            events = log.read_all()
        except ValidationError as exc:
            skipped_logs.append(_skipped_log(path, _validation_error_reason(exc)))
            continue
        for event in events:
            entries.append(_log_entry(path.stem, event, integrity_ok=integrity.ok))
    entries.sort(key=lambda entry: (entry.timestamp, entry.session_id, entry.seq), reverse=True)
    safe_limit = max(0, limit)
    return MemoryLog(
        eventloom_path=str(base),
        session_id=session_id,
        limit=safe_limit,
        entries=entries[:safe_limit],
        skipped_logs=skipped_logs,
    )


def inspect_memory_diff(
    eventloom_path: str | Path,
    *,
    session_id: str | None,
    from_seq: int,
    to_seq: int,
) -> MemoryDiff:
    """Return Eventloom events added in an inclusive sequence range."""
    if from_seq < 1:
        raise ValueError("from_seq must be >= 1")
    if to_seq < 1:
        raise ValueError("to_seq must be >= 1")
    if from_seq > to_seq:
        raise ValueError("from_seq must be <= to_seq")
    base = Path(eventloom_path).resolve()
    path = _session_log_path(base, session_id)
    log = EventLog(path)
    integrity = log.verify()
    added = [
        _log_entry(path.stem, event, integrity_ok=integrity.ok)
        for event in log.read_all()
        if from_seq <= event.seq <= to_seq
    ]
    return MemoryDiff(
        eventloom_path=str(base),
        session_id=session_id or path.stem,
        from_seq=from_seq,
        to_seq=to_seq,
        integrity_ok=integrity.ok,
        integrity_reason=integrity.broken_reason,
        added=added,
    )


def format_memory_status(status: MemoryStatus) -> str:
    """Format memory status for humans."""
    lines = [
        f"Eventloom: {status.eventloom_path}",
        f"Sessions: {status.session_count}",
        f"Total events: {status.total_events}",
    ]
    if not status.sessions:
        _append_skipped_logs(lines, status.skipped_logs)
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
    _append_skipped_logs(lines, status.skipped_logs)
    return "\n".join(lines)


def format_memory_diff(diff: MemoryDiff) -> str:
    """Format an event-level memory diff for humans."""
    if not diff.added:
        return "No memory events in range."
    lines: list[str] = []
    for entry in diff.added:
        lines.append(
            f"{entry.session_id} +[{entry.seq}] {entry.hash[:12]} "
            f"{entry.type} by {entry.actor}"
        )
        if entry.summary:
            lines.append(f"  {entry.summary}")
    if not diff.integrity_ok:
        lines.append("integrity=FAILED")
        if diff.integrity_reason:
            lines.append(f"reason={diff.integrity_reason}")
    return "\n".join(lines)


def format_memory_log(memory_log: MemoryLog) -> str:
    """Format recent Eventloom events for humans."""
    if not memory_log.entries:
        empty_lines = ["No memory events found."]
        _append_skipped_logs(empty_lines, memory_log.skipped_logs)
        return "\n".join(empty_lines)
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
    _append_skipped_logs(lines, memory_log.skipped_logs)
    return "\n".join(lines)


def _eventlog_paths(base: Path) -> tuple[list[Path], list[SkippedLog]]:
    if base.is_file():
        reason = _eventlog_skip_reason(base)
        return ([], [_skipped_log(base, reason)]) if reason else ([base], [])
    if not base.exists():
        return [], []
    paths: list[Path] = []
    skipped_logs: list[SkippedLog] = []
    for path in sorted(path for path in base.glob("*.jsonl") if path.is_file()):
        reason = _eventlog_skip_reason(path)
        if reason:
            skipped_logs.append(_skipped_log(path, reason))
        else:
            paths.append(path)
    return paths, skipped_logs


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


def _eventlog_skip_reason(path: Path) -> str | None:
    """Return why a broad scan should skip this JSONL file, if applicable."""
    try:
        with path.open(encoding="utf-8") as fh:
            first_line = next((line.strip() for line in fh if line.strip()), "")
    except OSError as exc:
        return f"could not read log: {exc}"
    if not first_line:
        return None
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return "first non-empty line is not valid JSON"
    if not isinstance(record, dict):
        return "first non-empty line is not a JSON object"
    if _EVENTLOOM_V1_SHAPE_FIELDS.issubset(record):
        return None
    missing = sorted(_REQUIRED_ZAXY_EVENT_FIELDS - set(record))
    if missing:
        return f"missing required Eventloom event fields: {', '.join(missing)}"
    return None


def _validation_error_reason(exc: ValidationError) -> str:
    missing = sorted(
        str(error["loc"][0])
        for error in exc.errors()
        if error.get("type") == "missing" and error.get("loc")
    )
    if missing:
        return f"invalid Eventloom event log; missing fields after preflight: {', '.join(missing)}"
    return "invalid Eventloom event log; event validation failed"


def _skipped_log(path: Path, reason: str) -> SkippedLog:
    return SkippedLog(path=str(path.resolve()), reason=reason)


def _append_skipped_logs(lines: list[str], skipped_logs: list[SkippedLog]) -> None:
    if not skipped_logs:
        return
    lines.append("")
    lines.append("Skipped logs:")
    for skipped in skipped_logs:
        lines.append(f"  {skipped.path}: {skipped.reason}")
