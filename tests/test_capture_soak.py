"""Tests for deterministic capture soak reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.capture_soak import build_capture_soak_report, format_capture_soak_report
from zaxy.event import EventLog


def test_capture_soak_report_passes_when_all_capture_lanes_are_active(tmp_path: Path) -> None:
    """Capture soak should pass when deterministic observation lanes are present."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    heartbeat = log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    for event_type in ("command.completed", "file.edit.applied", "tool.call.completed", "transcript.turn"):
        log.append(event_type, actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")

    report = build_capture_soak_report(
        eventloom_path=tmp_path / ".eventloom",
        workspace_root=tmp_path,
        session_id="agent-1",
    )

    assert report["status"] == "ok"
    assert report["beta_criteria"]["status"] == "pass"
    assert report["latest_hook_event"]["seq"] == heartbeat.seq
    assert report["latest_hook_event"]["hash"] == heartbeat.hash
    assert report["observation_coverage"]["transcript.turn"]["latest"]["hash"]
    assert report["missing_observation_types"] == []
    assert report["stale_observation_types"] == []
    assert report["actions"] == []


def test_capture_soak_detects_runtime_when_session_scopes_eventloom_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session-scoped coverage should still detect the directory-level capture watcher."""
    workspace = tmp_path
    eventloom = workspace / ".eventloom"
    (workspace / ".codex").mkdir()
    (workspace / ".codex" / "zaxy-capture.json").write_text(
        json.dumps(
            {
                "client": "codex",
                "capture": "local-session-jsonl",
                "eventloom_path": ".eventloom",
                "session_id": "agent-1",
            }
        ),
        encoding="utf-8",
    )
    log = EventLog(eventloom / "agent-1.jsonl")
    log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    for event_type in ("command.completed", "file.edit.applied", "tool.call.completed", "transcript.turn"):
        log.append(event_type, actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    monkeypatch.setattr(
        "zaxy.hooks._iter_process_cmdlines",
        lambda: [
            (
                4242,
                [
                    "zaxy",
                    "codex-capture",
                    "--watch",
                    "--workspace",
                    str(workspace),
                    "--eventloom-path",
                    str(eventloom),
                ],
            )
        ],
    )
    monkeypatch.setattr("zaxy.hooks._process_cwd", lambda _pid: workspace)

    report = build_capture_soak_report(
        eventloom_path=eventloom,
        workspace_root=workspace,
        session_id="agent-1",
    )

    assert report["eventloom_path"] == str(eventloom / "agent-1.jsonl")
    assert report["codex_capture"]["runtime"]["running"] is True
    assert report["beta_criteria"]["status"] == "pass"


def test_capture_soak_report_fails_for_missing_or_stale_capture_lanes(tmp_path: Path) -> None:
    """Capture soak should produce concrete remediation when coverage is incomplete or stale."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    log.append("command.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")

    report = build_capture_soak_report(
        eventloom_path=tmp_path / ".eventloom",
        workspace_root=tmp_path,
        session_id="agent-1",
        max_stale_minutes=0,
    )

    assert report["status"] == "warning"
    assert report["beta_criteria"]["status"] == "fail"
    assert report["missing_observation_types"] == [
        "file.edit.applied",
        "tool.call.completed",
        "transcript.turn",
    ]
    assert report["stale_observation_types"] == ["command.completed"]
    assert "Wire hooks or adapter sinks for: file.edit.applied, tool.call.completed, transcript.turn." in report["actions"]
    assert "Refresh stale capture lanes: command.completed." in report["actions"]


def test_capture_soak_command_emits_json(tmp_path: Path) -> None:
    """The CLI should expose machine-readable soak evidence."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    for event_type in ("command.completed", "file.edit.applied", "tool.call.completed", "transcript.turn"):
        log.append(event_type, actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")

    result = CliRunner().invoke(
        app,
        [
            "capture",
            "soak",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--session-id",
            "agent-1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["beta_criteria"]["status"] == "pass"
    assert payload["session_id"] == "agent-1"


def test_capture_soak_command_exits_nonzero_when_beta_criteria_fail(tmp_path: Path) -> None:
    """The CLI should be usable as a release gate."""
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "hook.heartbeat",
        actor="zaxy-hook",
        payload={"source": "codex"},
        thread="agent-1",
    )

    result = CliRunner().invoke(
        app,
        [
            "capture-soak",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--session-id",
            "agent-1",
        ],
    )

    assert result.exit_code == 1
    assert "Zaxy capture soak: warning" in result.output
    assert "beta criteria: fail" in result.output
    assert "transcript.turn: missing" in result.output


def test_format_capture_soak_report_includes_latest_hashes(tmp_path: Path) -> None:
    """Human output should include cited event positions for auditability."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    event = log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    log.append("command.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")

    report = build_capture_soak_report(
        eventloom_path=tmp_path / ".eventloom",
        workspace_root=tmp_path,
        session_id="agent-1",
        max_stale_minutes=0,
    )

    text = format_capture_soak_report(report)

    assert f"hook.heartbeat seq={event.seq} hash={event.hash}" in text
    assert "command.completed: count=1" in text
    assert "stale" in text
