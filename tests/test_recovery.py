"""Tests for compaction recovery packet assembly."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from zaxy.event import EventLog
from zaxy.recovery import (
    MAX_OPEN_TASK_LINES,
    MAX_RECENT_ACTIVITY_LINES,
    RecoveryPacket,
    RecoveryPacketError,
    RecoveryPacketLine,
    _require_eventloom_backed,
    assemble_recovery_packet,
    render_recovery_packet,
)

_CITATION_RE = re.compile(r"^eventloom://agent-1/events/[1-9][0-9]*#[0-9a-f]{12}$")


def _seed_fixture_log(tmp_path: Path) -> EventLog:
    """Seed one session log with an open task, accepted finding, known unknown, and precompact."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append(
        "task.created",
        actor="user",
        payload={"taskId": "task-1", "title": "Ship the recovery loop"},
        thread="agent-1",
    )
    log.append(
        "task.created",
        actor="user",
        payload={"taskId": "task-2", "title": "Write the migration note"},
        thread="agent-1",
    )
    log.append("task.completed", actor="agent", payload={"taskId": "task-2"}, thread="agent-1")
    log.append(
        "transcript.turn",
        actor="assistant",
        payload={"role": "assistant", "content": "Pre-compaction context turn", "turn_index": 1},
        thread="agent-1",
    )
    log.append(
        "coordination.finding.promoted",
        actor="coordinator",
        payload={
            "mission_id": "mission-1",
            "worker_id": "worker-1",
            "finding_id": "worker-1:finding:1",
            "summary": "Token refresh owns the auth regression",
            "status": "accepted",
        },
        thread="agent-1",
    )
    log.append(
        "metacognition.unknown.recorded",
        actor="agent",
        payload={
            "unknown_id": "metacognition:unknown:abcdefabcdefabcdefabcdef",
            "question": "Which provider enforces the rate limit?",
            "reason": "no cited evidence in session",
            "status": "open",
            "authority_status": "non_authoritative",
        },
        thread="agent-1",
    )
    log.append(
        "hook.precompact",
        actor="zaxy-hook",
        payload={"trigger": "precompact", "source": "claude-code"},
        thread="agent-1",
    )
    log.append(
        "transcript.turn",
        actor="assistant",
        payload={"role": "assistant", "content": "Resumed after compaction", "turn_index": 2},
        thread="agent-1",
    )
    log.append(
        "command.completed",
        actor="zaxy-observer",
        payload={"command": "pytest -q", "exit_code": 0},
        thread="agent-1",
    )
    return log


def test_assemble_recovery_packet_recovers_state_with_citations(tmp_path: Path) -> None:
    """The packet should recover seeded session state, each line cited to a sealed event."""
    log = _seed_fixture_log(tmp_path)
    events = log.read_all()

    packet = assemble_recovery_packet(log, session_id="agent-1")

    assert packet.session_id == "agent-1"
    assert packet.integrity_ok is True
    assert packet.event_count == len(events)
    assert [line.text for line in packet.open_tasks] == ["Ship the recovery loop (task task-1)"]
    assert packet.open_tasks[0].event_seq == 1
    assert [line.text for line in packet.accepted_findings] == [
        "Token refresh owns the auth regression (finding worker-1:finding:1)"
    ]
    assert [line.text for line in packet.known_unknowns] == [
        "Which provider enforces the rate limit?"
    ]
    assert packet.anchor_event_type == "hook.precompact"
    assert packet.anchor_event_seq == 7
    assert packet.anchor_citation is not None and _CITATION_RE.fullmatch(packet.anchor_citation)
    assert [line.text for line in packet.recent_activity] == [
        "assistant: Resumed after compaction",
        "$ pytest -q (exit 0)",
    ]
    by_seq = {event.seq: event for event in events}
    for line in packet.lines:
        assert _CITATION_RE.fullmatch(line.citation)
        assert by_seq[line.event_seq].hash == line.event_hash
        assert line.citation.endswith(by_seq[line.event_seq].hash[:12])


def test_assemble_recovery_packet_is_deterministic(tmp_path: Path) -> None:
    """Two assemblies over the same log state must produce identical packets."""
    log = _seed_fixture_log(tmp_path)

    first = assemble_recovery_packet(log, session_id="agent-1")
    second = assemble_recovery_packet(log, session_id="agent-1")

    assert first == second
    assert render_recovery_packet(first) == render_recovery_packet(second)


def test_assemble_recovery_packet_enforces_section_bounds(tmp_path: Path) -> None:
    """Sections must stay capped, keeping the newest verbatim activity."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    for index in range(MAX_OPEN_TASK_LINES + 5):
        log.append(
            "task.created",
            actor="user",
            payload={"taskId": f"task-{index}", "title": f"Task {index}"},
            thread="agent-1",
        )
    log.append(
        "hook.precompact",
        actor="zaxy-hook",
        payload={"trigger": "precompact", "source": "claude-code"},
        thread="agent-1",
    )
    for index in range(MAX_RECENT_ACTIVITY_LINES + 4):
        log.append(
            "command.completed",
            actor="zaxy-observer",
            payload={"command": f"echo {index}", "exit_code": 0},
            thread="agent-1",
        )

    packet = assemble_recovery_packet(log, session_id="agent-1")

    assert len(packet.open_tasks) == MAX_OPEN_TASK_LINES
    assert packet.truncated_sections["open_tasks"] == MAX_OPEN_TASK_LINES + 5
    assert len(packet.recent_activity) == MAX_RECENT_ACTIVITY_LINES
    assert packet.truncated_sections["recent_activity"] == MAX_RECENT_ACTIVITY_LINES + 4
    assert packet.recent_activity[-1].text == f"$ echo {MAX_RECENT_ACTIVITY_LINES + 3} (exit 0)"
    assert packet.recent_activity[0].text == "$ echo 4 (exit 0)"
    rendered = render_recovery_packet(packet)
    assert f"(showing {MAX_OPEN_TASK_LINES} of {MAX_OPEN_TASK_LINES + 5})" in rendered


def test_recovery_packet_excludes_completed_tasks_and_resolved_assignments(tmp_path: Path) -> None:
    """Completed tasks and promotion-resolved assignments must not resurface."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append(
        "coordination.assignment.created",
        actor="coordinator",
        payload={"mission_id": "m1", "worker_id": "worker-done", "assignment": "Audit auth", "status": "assigned"},
        thread="agent-1",
    )
    log.append(
        "coordination.assignment.created",
        actor="coordinator",
        payload={"mission_id": "m1", "worker_id": "worker-open", "assignment": "Audit billing", "status": "assigned"},
        thread="agent-1",
    )
    log.append(
        "coordination.finding.promoted",
        actor="coordinator",
        payload={
            "mission_id": "m1",
            "worker_id": "worker-done",
            "finding_id": "worker-done:finding:1",
            "summary": "Auth audit complete",
            "status": "accepted",
        },
        thread="agent-1",
    )

    packet = assemble_recovery_packet(log, session_id="agent-1")

    assert [line.text for line in packet.open_tasks] == [
        "assignment for worker-open: Audit billing"
    ]


def test_render_recovery_packet_brackets_output_for_reinjection(tmp_path: Path) -> None:
    """Rendered packets must carry stable markers and per-line citations."""
    log = _seed_fixture_log(tmp_path)

    rendered = render_recovery_packet(assemble_recovery_packet(log, session_id="agent-1"))

    assert rendered.startswith("=== ZAXY RECOVERY PACKET session=agent-1 ===")
    assert rendered.endswith("=== END ZAXY RECOVERY PACKET ===")
    assert "anchor: hook.precompact [eventloom://agent-1/events/7#" in rendered
    assert "- Ship the recovery loop (task task-1) [eventloom://agent-1/events/1#" in rendered


def test_render_recovery_packet_reports_empty_sections(tmp_path: Path) -> None:
    """An empty log should render explicit none-recorded sections, not silence."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")

    rendered = render_recovery_packet(assemble_recovery_packet(log, session_id="agent-1"))

    assert "anchor: none recorded" in rendered
    assert "Open tasks: none recorded" in rendered
    assert "Known unknowns: none recorded" in rendered


def test_recovery_packet_rejects_lines_not_backed_by_the_log(tmp_path: Path) -> None:
    """The Eventloom-backed rail must reject fabricated or cross-log lines."""
    log = _seed_fixture_log(tmp_path)
    packet = assemble_recovery_packet(log, session_id="agent-1")
    forged = RecoveryPacket(
        session_id=packet.session_id,
        event_count=packet.event_count,
        integrity_ok=packet.integrity_ok,
        anchor_event_type=packet.anchor_event_type,
        anchor_event_seq=packet.anchor_event_seq,
        anchor_citation=packet.anchor_citation,
        open_tasks=(
            RecoveryPacketLine(
                section="open_tasks",
                text="Fabricated task that never happened",
                citation="eventloom://agent-1/events/999#aaaaaaaaaaaa",
                event_seq=999,
                event_hash="a" * 64,
            ),
        ),
        accepted_findings=packet.accepted_findings,
        known_unknowns=packet.known_unknowns,
        recent_activity=packet.recent_activity,
        truncated_sections=packet.truncated_sections,
    )

    with pytest.raises(RecoveryPacketError, match="does not resolve to a sealed event"):
        _require_eventloom_backed(forged, log.read_all())


def test_recovery_packet_flags_broken_log_integrity(tmp_path: Path) -> None:
    """A tampered log should still assemble but be flagged as integrity-broken."""
    log = _seed_fixture_log(tmp_path)
    path = log.path
    tampered = path.read_text(encoding="utf-8").replace(
        "Ship the recovery loop",
        "Ship the rewritten loop",
    )
    path.write_text(tampered, encoding="utf-8")

    packet = assemble_recovery_packet(EventLog(path), session_id="agent-1")

    assert packet.integrity_ok is False
    assert "integrity verification failed" in render_recovery_packet(packet)
