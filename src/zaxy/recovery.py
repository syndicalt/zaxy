"""Compaction recovery packet assembly over Eventloom-backed state.

After a harness compacts its context and resumes, the recovery packet
restates the durable session state a summary plausibly lost: open tasks and
assignments, accepted coordination findings, recorded known unknowns, and
recent verbatim activity since the last consolidation or precompact anchor.

Safety rails mirror identity-preserving compaction (`zaxy.compaction`):
every packet line cites the sealed Eventloom event it was derived from via
the canonical ``eventloom://`` reference, citations are re-resolved against
the log before the packet is returned, and assembly is read-only — the packet
is a pure, deterministic function of the log state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zaxy.compaction import event_ref
from zaxy.event import Event, EventLog

MAX_OPEN_TASK_LINES = 10
MAX_ACCEPTED_FINDING_LINES = 10
MAX_KNOWN_UNKNOWN_LINES = 10
MAX_RECENT_ACTIVITY_LINES = 8
MAX_LINE_TEXT_CHARS = 240

ANCHOR_EVENT_TYPES = frozenset(
    {
        "hook.precompact",
        "consolidation.candidate.created",
        "consolidation.candidate.reviewed",
    }
)
ACCEPTED_FINDING_EVENT_TYPES = frozenset(
    {
        "coordination.finding.promoted",
        "coordination.finding.accepted",
    }
)
ACCEPTED_REVIEW_STATUSES = frozenset({"accepted", "promoted", "approved"})
RECENT_ACTIVITY_EVENT_TYPES = frozenset(
    {
        "transcript.turn",
        "command.completed",
        "file.edit.applied",
        "tool.call.completed",
        "document.indexed",
    }
)

_SECTION_CAPS = {
    "open_tasks": MAX_OPEN_TASK_LINES,
    "accepted_findings": MAX_ACCEPTED_FINDING_LINES,
    "known_unknowns": MAX_KNOWN_UNKNOWN_LINES,
    "recent_activity": MAX_RECENT_ACTIVITY_LINES,
}


class RecoveryPacketError(ValueError):
    """Raised when a recovery packet line is not Eventloom-backed."""


@dataclass(frozen=True)
class RecoveryPacketLine:
    """One cited recovery statement derived from a sealed event."""

    section: str
    text: str
    citation: str
    event_seq: int
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this line."""
        return {
            "section": self.section,
            "text": self.text,
            "citation": self.citation,
            "event_seq": self.event_seq,
            "event_hash": self.event_hash,
        }


@dataclass(frozen=True)
class RecoveryPacket:
    """Deterministic, cited post-compaction recovery context."""

    session_id: str
    event_count: int
    integrity_ok: bool
    anchor_event_type: str | None
    anchor_event_seq: int | None
    anchor_citation: str | None
    open_tasks: tuple[RecoveryPacketLine, ...]
    accepted_findings: tuple[RecoveryPacketLine, ...]
    known_unknowns: tuple[RecoveryPacketLine, ...]
    recent_activity: tuple[RecoveryPacketLine, ...]
    truncated_sections: dict[str, int]

    @property
    def lines(self) -> tuple[RecoveryPacketLine, ...]:
        """Return every packet line across sections in render order."""
        return (
            *self.open_tasks,
            *self.accepted_findings,
            *self.known_unknowns,
            *self.recent_activity,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of this packet."""
        return {
            "session_id": self.session_id,
            "event_count": self.event_count,
            "integrity_ok": self.integrity_ok,
            "anchor_event_type": self.anchor_event_type,
            "anchor_event_seq": self.anchor_event_seq,
            "anchor_citation": self.anchor_citation,
            "open_tasks": [line.to_dict() for line in self.open_tasks],
            "accepted_findings": [line.to_dict() for line in self.accepted_findings],
            "known_unknowns": [line.to_dict() for line in self.known_unknowns],
            "recent_activity": [line.to_dict() for line in self.recent_activity],
            "truncated_sections": dict(self.truncated_sections),
        }


def assemble_recovery_packet(eventlog: EventLog, *, session_id: str) -> RecoveryPacket:
    """Assemble a cited recovery packet from one session's Eventloom log.

    Assembly is read-only and deterministic: the same log state always
    produces an identical packet. Every line is validated against the log so
    the packet can only contain Eventloom-backed content.
    """
    events = eventlog.read_all()
    integrity = eventlog.verify()
    anchor = _latest_anchor_event(events)
    sections = {
        "open_tasks": _open_task_lines(events),
        "accepted_findings": _accepted_finding_lines(events),
        "known_unknowns": _known_unknown_lines(events),
        "recent_activity": _recent_activity_lines(events, anchor),
    }
    truncated_sections: dict[str, int] = {}
    bounded: dict[str, tuple[RecoveryPacketLine, ...]] = {}
    for section, lines in sections.items():
        cap = _SECTION_CAPS[section]
        if len(lines) > cap:
            truncated_sections[section] = len(lines)
            lines = lines[-cap:] if section == "recent_activity" else lines[:cap]
        bounded[section] = tuple(lines)
    packet = RecoveryPacket(
        session_id=session_id,
        event_count=len(events),
        integrity_ok=integrity.ok,
        anchor_event_type=anchor.type if anchor is not None else None,
        anchor_event_seq=anchor.seq if anchor is not None else None,
        anchor_citation=event_ref(anchor) if anchor is not None else None,
        open_tasks=bounded["open_tasks"],
        accepted_findings=bounded["accepted_findings"],
        known_unknowns=bounded["known_unknowns"],
        recent_activity=bounded["recent_activity"],
        truncated_sections=truncated_sections,
    )
    _require_eventloom_backed(packet, events)
    return packet


def render_recovery_packet(packet: RecoveryPacket) -> str:
    """Render a recovery packet for stdout re-injection by a harness hook."""
    lines = [
        f"=== ZAXY RECOVERY PACKET session={packet.session_id} ===",
        "Eventloom-backed recovery context; every line cites its sealed source event.",
    ]
    if not packet.integrity_ok:
        lines.append(
            "WARNING: event log integrity verification failed; "
            "treat recovered lines as suspect until the log is restored."
        )
    if packet.anchor_citation is not None:
        lines.append(f"anchor: {packet.anchor_event_type} [{packet.anchor_citation}]")
    else:
        lines.append("anchor: none recorded (showing recent activity from the full log)")
    sections = (
        ("Open tasks", "open_tasks", packet.open_tasks),
        ("Accepted findings", "accepted_findings", packet.accepted_findings),
        ("Known unknowns", "known_unknowns", packet.known_unknowns),
        ("Recent activity since anchor", "recent_activity", packet.recent_activity),
    )
    for title, section, section_lines in sections:
        total = packet.truncated_sections.get(section, len(section_lines))
        if not section_lines:
            lines.append(f"{title}: none recorded")
            continue
        suffix = f" (showing {len(section_lines)} of {total})" if total > len(section_lines) else ""
        lines.append(f"{title} ({len(section_lines)}){suffix}:")
        lines.extend(f"- {line.text} [{line.citation}]" for line in section_lines)
    lines.append("=== END ZAXY RECOVERY PACKET ===")
    return "\n".join(lines)


def _require_eventloom_backed(packet: RecoveryPacket, events: list[Event]) -> None:
    """Reject any packet line whose citation does not resolve to a logged event."""
    by_seq = {event.seq: event for event in events}
    for line in packet.lines:
        source = by_seq.get(line.event_seq)
        if source is None or source.hash != line.event_hash or event_ref(source) != line.citation:
            raise RecoveryPacketError(
                f"recovery packet line in {line.section!r} cites {line.citation}, "
                "which does not resolve to a sealed event in this log"
            )


def _latest_anchor_event(events: list[Event]) -> Event | None:
    anchor: Event | None = None
    for event in events:
        if event.type in ANCHOR_EVENT_TYPES:
            anchor = event
    return anchor


def _open_task_lines(events: list[Event]) -> list[RecoveryPacketLine]:
    completed_task_ids = {
        str(event.payload.get("taskId"))
        for event in events
        if event.type.startswith("task.completed") and event.payload.get("taskId")
    }
    open_tasks: dict[str, Event] = {}
    for event in events:
        if not event.type.startswith("task.") or event.type.startswith("task.completed"):
            continue
        task_id = _string(event.payload.get("taskId"))
        if task_id is None or task_id in completed_task_ids:
            continue
        open_tasks[task_id] = event
    promoted_workers = {
        (str(event.payload.get("worker_id")), event.seq)
        for event in events
        if event.type in ACCEPTED_FINDING_EVENT_TYPES and event.payload.get("worker_id")
    }
    open_assignments: dict[str, Event] = {}
    for event in events:
        if event.type != "coordination.assignment.created":
            continue
        worker_id = _string(event.payload.get("worker_id"))
        if worker_id is None:
            continue
        resolved = any(worker == worker_id and seq > event.seq for worker, seq in promoted_workers)
        if not resolved:
            open_assignments[worker_id] = event
    lines = [
        _line("open_tasks", _task_text(task_id, event), event)
        for task_id, event in open_tasks.items()
    ]
    lines.extend(
        _line(
            "open_tasks",
            f"assignment for {worker_id}: {_string(event.payload.get('assignment')) or 'unspecified'}",
            event,
        )
        for worker_id, event in open_assignments.items()
    )
    lines.sort(key=lambda line: line.event_seq)
    return lines


def _accepted_finding_lines(events: list[Event]) -> list[RecoveryPacketLine]:
    accepted: dict[str, Event] = {}
    for event in events:
        status = _string(event.payload.get("status"))
        is_promotion = event.type in ACCEPTED_FINDING_EVENT_TYPES
        is_accepted_review = (
            event.type == "coordination.finding.reviewed"
            and status is not None
            and status.casefold() in ACCEPTED_REVIEW_STATUSES
        )
        if not is_promotion and not is_accepted_review:
            continue
        finding_id = _string(event.payload.get("finding_id")) or f"event:{event.seq}"
        accepted[finding_id] = event
    lines = [
        _line("accepted_findings", _finding_text(finding_id, event), event)
        for finding_id, event in accepted.items()
    ]
    lines.sort(key=lambda line: line.event_seq)
    return lines


def _known_unknown_lines(events: list[Event]) -> list[RecoveryPacketLine]:
    open_unknowns: dict[str, Event] = {}
    for event in events:
        if event.type != "metacognition.unknown.recorded":
            continue
        if _string(event.payload.get("status")) != "open":
            continue
        unknown_id = _string(event.payload.get("unknown_id")) or f"event:{event.seq}"
        open_unknowns[unknown_id] = event
    lines = [
        _line(
            "known_unknowns",
            _string(event.payload.get("question")) or unknown_id,
            event,
        )
        for unknown_id, event in open_unknowns.items()
    ]
    lines.sort(key=lambda line: line.event_seq)
    return lines


def _recent_activity_lines(events: list[Event], anchor: Event | None) -> list[RecoveryPacketLine]:
    anchor_seq = anchor.seq if anchor is not None else 0
    return [
        _line("recent_activity", _activity_text(event), event)
        for event in events
        if event.seq > anchor_seq and event.type in RECENT_ACTIVITY_EVENT_TYPES
    ]


def _task_text(task_id: str, event: Event) -> str:
    for key in ("title", "summary", "description"):
        value = _string(event.payload.get(key))
        if value:
            return f"{value} (task {task_id})"
    return f"task {task_id}"


def _finding_text(finding_id: str, event: Event) -> str:
    summary = _string(event.payload.get("summary"))
    if summary:
        return f"{summary} (finding {finding_id})"
    claim_key = _string(event.payload.get("claim_key"))
    claim_value = _string(event.payload.get("claim_value"))
    if claim_key and claim_value:
        return f"{claim_key}={claim_value} (finding {finding_id})"
    return f"finding {finding_id}"


def _activity_text(event: Event) -> str:
    payload = event.payload
    if event.type == "transcript.turn":
        role = _string(payload.get("role")) or event.actor
        return f"{role}: {_string(payload.get('content')) or ''}".strip()
    if event.type == "command.completed":
        exit_code = payload.get("exit_code")
        suffix = f" (exit {exit_code})" if isinstance(exit_code, int) else ""
        return f"$ {_string(payload.get('command')) or 'command'}{suffix}"
    if event.type == "file.edit.applied":
        operation = _string(payload.get("operation")) or "edited"
        return f"{operation} {_string(payload.get('path')) or 'file'}"
    if event.type == "tool.call.completed":
        status = _string(payload.get("status")) or "ok"
        return f"tool {_string(payload.get('tool_name')) or 'unknown'} {status}"
    if event.type == "document.indexed":
        return f"indexed {_string(payload.get('path')) or 'document'}"
    return event.type


def _line(section: str, text: str, event: Event) -> RecoveryPacketLine:
    return RecoveryPacketLine(
        section=section,
        text=_clip(text),
        citation=event_ref(event),
        event_seq=event.seq,
        event_hash=event.hash,
    )


def _clip(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= MAX_LINE_TEXT_CHARS:
        return cleaned
    return cleaned[: MAX_LINE_TEXT_CHARS - 1] + "…"


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
