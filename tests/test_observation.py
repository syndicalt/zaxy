"""Tests for normalized automatic observation payloads."""

from __future__ import annotations

from zaxy.observation import build_command_observation, build_file_edit_observation


def test_build_command_observation_redacts_and_bounds_output() -> None:
    """Command observations should be structured, bounded, and source-tagged."""
    event = build_command_observation(
        command="pytest --token secret",
        exit_code=1,
        session_id="agent-1",
        source="codex",
        workspace="/repo",
        duration_ms=1200,
        stdout="x" * 300,
        stderr="failed",
    )

    assert event["event_type"] == "command.completed"
    assert event["actor"] == "zaxy-observer"
    assert event["payload"]["command"] == "pytest [REDACTED]"
    assert event["payload"]["exit_code"] == 1
    assert event["payload"]["outcome"] == "failed"
    assert event["payload"]["source"] == "codex"
    assert event["payload"]["workspace"] == "/repo"
    assert event["payload"]["duration_ms"] == 1200
    assert event["payload"]["stdout_excerpt"] == "x" * 240
    assert event["payload"]["stderr_excerpt"] == "failed"


def test_build_file_edit_observation_records_metadata_without_content() -> None:
    """File edit observations should not persist source text."""
    event = build_file_edit_observation(
        path="src/zaxy/core.py",
        operation="modified",
        session_id="agent-1",
        source="codex",
        workspace="/repo",
        summary="Updated context assembly.",
        line_count=12,
        content="do not store this",
    )

    assert event == {
        "event_type": "file.edit.applied",
        "actor": "zaxy-observer",
        "payload": {
            "path": "src/zaxy/core.py",
            "operation": "modified",
            "session_id": "agent-1",
            "source": "codex",
            "workspace": "/repo",
            "summary": "Updated context assembly.",
            "line_count": 12,
        },
    }
