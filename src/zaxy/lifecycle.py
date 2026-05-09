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
