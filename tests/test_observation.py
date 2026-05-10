"""Tests for normalized automatic observation payloads."""

from __future__ import annotations

from zaxy.observation import (
    build_command_observation,
    build_file_edit_observation,
    build_tool_call_observation,
    build_transcript_turn_observation,
)


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


def test_build_tool_call_observation_records_redacted_argument_metadata() -> None:
    """Tool-call observations should store argument keys but not argument values."""
    event = build_tool_call_observation(
        tool_name="memory_append",
        status="ok",
        session_id="agent-1",
        source="codex",
        workspace="/repo",
        call_id="call-1",
        arguments={"event_type": "task.completed", "token": "secret"},
        result_summary="seq=12",
    )

    assert event["event_type"] == "tool.call.completed"
    assert event["actor"] == "zaxy-observer"
    assert event["payload"]["tool_name"] == "memory_append"
    assert event["payload"]["status"] == "ok"
    assert event["payload"]["call_id"] == "call-1"
    assert event["payload"]["argument_keys"] == ["event_type", "token"]
    assert event["payload"]["arguments_redacted"] is True
    assert event["payload"]["result_summary"] == "seq=12"
    assert event["payload"]["source"] == "codex"
    assert event["payload"]["workspace"] == "/repo"
    assert "arguments" not in event["payload"]


def test_build_transcript_turn_observation_sanitizes_content() -> None:
    """Transcript observations should preserve role metadata and redact secrets."""
    event = build_transcript_turn_observation(
        role="assistant",
        content="The key is sk-test-secret.",
        session_id="agent-1",
        source="codex",
        turn_index=2,
    )

    assert event["event_type"] == "transcript.turn"
    assert event["actor"] == "assistant"
    assert event["payload"]["role"] == "assistant"
    assert event["payload"]["source"] == "codex"
    assert event["payload"]["session_id"] == "agent-1"
    assert event["payload"]["turn_index"] == 2
    assert event["payload"]["content"] == "[REDACTED]"
    assert event["payload"]["redacted_paths"] == ["content"]
