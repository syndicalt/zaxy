"""Typed lifecycle event builders for agent integrations."""

from __future__ import annotations

from typing import Any

OUTPUT_EXCERPT_CHARS = 240


def build_tool_call_completed_event(
    *,
    tool_name: str,
    status: str,
    session_id: str,
    call_id: str | None = None,
    arguments: dict[str, Any] | None = None,
    result_summary: str | None = None,
) -> dict[str, Any]:
    """Build a safe tool-call lifecycle event without raw argument values."""
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "status": status,
        "session_id": session_id,
        "call_id": call_id,
        "arguments_redacted": True,
        "argument_keys": sorted((arguments or {}).keys()),
    }
    if result_summary is not None:
        payload["result_summary"] = result_summary
    return {
        "event_type": "tool.call.completed",
        "actor": "zaxy",
        "payload": payload,
    }


def build_command_completed_event(
    *,
    command: str,
    exit_code: int,
    session_id: str,
    duration_ms: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    """Build a command-result lifecycle event with bounded output excerpts."""
    payload: dict[str, Any] = {
        "command": command,
        "exit_code": exit_code,
        "outcome": "passed" if exit_code == 0 else "failed",
        "session_id": session_id,
        "stdout_excerpt": stdout[:OUTPUT_EXCERPT_CHARS],
        "stderr_excerpt": stderr[:OUTPUT_EXCERPT_CHARS],
    }
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    return {
        "event_type": "command.completed",
        "actor": "zaxy",
        "payload": payload,
    }


def build_file_edit_applied_event(
    *,
    path: str,
    operation: str,
    session_id: str,
    summary: str | None = None,
    line_count: int | None = None,
) -> dict[str, Any]:
    """Build a file-edit lifecycle event without source content."""
    payload: dict[str, Any] = {
        "path": path,
        "operation": operation,
        "session_id": session_id,
    }
    if summary is not None:
        payload["summary"] = summary
    if line_count is not None:
        payload["line_count"] = line_count
    return {
        "event_type": "file.edit.applied",
        "actor": "zaxy",
        "payload": payload,
    }


def build_compaction_completed_event(
    *,
    session_id: str,
    mode: str,
    status: str,
    log_path: str,
    event_count: int,
    output_path: str | None = None,
    projection_path: str | None = None,
    snapshot_path: str | None = None,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Build a compaction lifecycle event with artifact paths, not content."""
    payload: dict[str, Any] = {
        "session_id": session_id,
        "mode": mode,
        "status": status,
        "log_path": log_path,
        "event_count": event_count,
    }
    if output_path is not None:
        payload["output_path"] = output_path
    if projection_path is not None:
        payload["projection_path"] = projection_path
    if snapshot_path is not None:
        payload["snapshot_path"] = snapshot_path
    if strategy is not None:
        payload["strategy"] = strategy
    return {
        "event_type": "compaction.completed",
        "actor": "zaxy",
        "payload": payload,
    }


def build_subagent_completed_event(
    *,
    parent_session_id: str,
    subagent_session_id: str,
    status: str,
    summary: str,
) -> dict[str, Any]:
    """Build a subagent completion lifecycle event with bounded summary text."""
    return {
        "event_type": "subagent.completed",
        "actor": "zaxy",
        "payload": {
            "parent_session_id": parent_session_id,
            "subagent_session_id": subagent_session_id,
            "status": status,
            "summary": summary[:OUTPUT_EXCERPT_CHARS],
        },
    }


def build_session_ended_event(
    *,
    session_id: str,
    reason: str,
    status: str,
) -> dict[str, Any]:
    """Build a session-end lifecycle event."""
    return {
        "event_type": "session.ended",
        "actor": "zaxy",
        "payload": {
            "session_id": session_id,
            "reason": reason,
            "status": status,
        },
    }
