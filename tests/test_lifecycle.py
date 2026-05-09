"""Tests for typed lifecycle event builders."""

from __future__ import annotations

from zaxy.lifecycle import (
    build_command_completed_event,
    build_compaction_completed_event,
    build_file_edit_applied_event,
    build_session_ended_event,
    build_subagent_completed_event,
    build_tool_call_completed_event,
)


def test_build_tool_call_completed_event_redacts_arguments_by_default() -> None:
    event = build_tool_call_completed_event(
        tool_name="shell",
        status="succeeded",
        session_id="agent-1",
        call_id="call-123",
        arguments={"command": "pytest", "token": "secret"},
        result_summary="443 passed",
    )

    assert event == {
        "event_type": "tool.call.completed",
        "actor": "zaxy",
        "payload": {
            "tool_name": "shell",
            "status": "succeeded",
            "session_id": "agent-1",
            "call_id": "call-123",
            "arguments_redacted": True,
            "argument_keys": ["command", "token"],
            "result_summary": "443 passed",
        },
    }


def test_build_command_completed_event_records_bounded_output() -> None:
    event = build_command_completed_event(
        command="pytest",
        exit_code=0,
        session_id="agent-1",
        duration_ms=1200,
        stdout="x" * 300,
        stderr="",
    )

    payload = event["payload"]
    assert event["event_type"] == "command.completed"
    assert payload["command"] == "pytest"
    assert payload["exit_code"] == 0
    assert payload["outcome"] == "passed"
    assert payload["duration_ms"] == 1200
    assert payload["stdout_excerpt"] == "x" * 240
    assert payload["stderr_excerpt"] == ""


def test_build_file_edit_applied_event_records_paths_not_content() -> None:
    event = build_file_edit_applied_event(
        path="src/zaxy/core.py",
        operation="modified",
        session_id="agent-1",
        summary="Added lifecycle hook.",
        line_count=12,
    )

    assert event == {
        "event_type": "file.edit.applied",
        "actor": "zaxy",
        "payload": {
            "path": "src/zaxy/core.py",
            "operation": "modified",
            "session_id": "agent-1",
            "summary": "Added lifecycle hook.",
            "line_count": 12,
        },
    }


def test_build_compaction_completed_event_records_artifact_metadata() -> None:
    event = build_compaction_completed_event(
        session_id="agent-1",
        mode="rewrite",
        status="succeeded",
        log_path=".eventloom/agent.jsonl",
        event_count=12,
        output_path=".eventloom/agent.jsonl",
        snapshot_path=".eventloom/agent.snapshot-12.json",
    )

    assert event == {
        "event_type": "compaction.completed",
        "actor": "zaxy",
        "payload": {
            "session_id": "agent-1",
            "mode": "rewrite",
            "status": "succeeded",
            "log_path": ".eventloom/agent.jsonl",
            "event_count": 12,
            "output_path": ".eventloom/agent.jsonl",
            "snapshot_path": ".eventloom/agent.snapshot-12.json",
        },
    }


def test_build_subagent_completed_event_records_summary_without_transcript() -> None:
    event = build_subagent_completed_event(
        parent_session_id="main",
        subagent_session_id="worker-1",
        status="succeeded",
        summary="Worker finished retrieval.",
    )

    assert event == {
        "event_type": "subagent.completed",
        "actor": "zaxy",
        "payload": {
            "parent_session_id": "main",
            "subagent_session_id": "worker-1",
            "status": "succeeded",
            "summary": "Worker finished retrieval.",
        },
    }


def test_build_session_ended_event_records_reason() -> None:
    event = build_session_ended_event(
        session_id="agent-1",
        reason="teardown",
        status="succeeded",
    )

    assert event == {
        "event_type": "session.ended",
        "actor": "zaxy",
        "payload": {
            "session_id": "agent-1",
            "reason": "teardown",
            "status": "succeeded",
        },
    }
