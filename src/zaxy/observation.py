"""Normalized automatic observation events for hook adapters."""

from __future__ import annotations

import re
from typing import Any

from zaxy.lifecycle import (
    OUTPUT_EXCERPT_CHARS,
    build_command_completed_event,
    build_file_edit_applied_event,
    build_tool_call_completed_event,
)
from zaxy.security import secure_payload

SECRET_ARG_PATTERN = re.compile(
    r"(?i)--(?:api[-_]?key|authorization|bearer|cookie|credential|password|private[-_]?key|secret|token)"
    r"(?:=|\s+)\S+"
)


def build_command_observation(
    *,
    command: str,
    exit_code: int,
    session_id: str,
    source: str,
    workspace: str | None = None,
    duration_ms: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    """Build a normalized command observation for automatic capture."""
    event = build_command_completed_event(
        command=_redact_command(command),
        exit_code=exit_code,
        session_id=session_id,
        duration_ms=duration_ms,
        stdout=stdout[:OUTPUT_EXCERPT_CHARS],
        stderr=stderr[:OUTPUT_EXCERPT_CHARS],
    )
    event["actor"] = "zaxy-observer"
    event["payload"]["source"] = source
    if workspace:
        event["payload"]["workspace"] = workspace
    return event


def build_file_edit_observation(
    *,
    path: str,
    operation: str,
    session_id: str,
    source: str,
    workspace: str | None = None,
    summary: str | None = None,
    line_count: int | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    """Build a normalized file-edit observation without persisting source content."""
    _ = content
    event = build_file_edit_applied_event(
        path=path,
        operation=operation,
        session_id=session_id,
        summary=summary,
        line_count=line_count,
    )
    event["actor"] = "zaxy-observer"
    event["payload"]["source"] = source
    if workspace:
        event["payload"]["workspace"] = workspace
    return event


def build_tool_call_observation(
    *,
    tool_name: str,
    status: str,
    session_id: str,
    source: str,
    workspace: str | None = None,
    call_id: str | None = None,
    arguments: dict[str, Any] | None = None,
    result_summary: str | None = None,
) -> dict[str, Any]:
    """Build a normalized tool-call observation without raw argument values."""
    event = build_tool_call_completed_event(
        tool_name=tool_name,
        status=status,
        session_id=session_id,
        call_id=call_id,
        arguments=arguments,
        result_summary=result_summary,
    )
    event["actor"] = "zaxy-observer"
    event["payload"]["source"] = source
    if workspace:
        event["payload"]["workspace"] = workspace
    return event


def build_transcript_turn_observation(
    *,
    role: str,
    content: str,
    session_id: str,
    source: str,
    turn_index: int | None = None,
) -> dict[str, Any]:
    """Build a normalized transcript turn observation with sanitized content."""
    normalized_role = role.strip() or "unknown"
    secured = secure_payload({"content": content})
    payload: dict[str, Any] = {
        "source": source,
        "role": normalized_role,
        "content": secured.payload["content"],
        "redacted_paths": secured.redacted_paths,
        "session_id": session_id,
    }
    if turn_index is not None:
        payload["turn_index"] = turn_index
    return {
        "event_type": "transcript.turn",
        "actor": normalized_role,
        "payload": payload,
    }


def _redact_command(command: str) -> str:
    return " ".join(SECRET_ARG_PATTERN.sub("[REDACTED]", command).split())
