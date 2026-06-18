"""Deterministic local capture for Claude Code session JSONL logs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zaxy.event import Event, EventLog
from zaxy.observation import (
    build_command_observation,
    build_file_edit_observation,
    build_tool_call_observation,
    build_transcript_turn_observation,
)
from zaxy.salience import build_cue_record
from zaxy.session import SessionManager

DEFAULT_CLAUDE_SOURCE = "claude-local"
CLAUDE_PROJECTS_DIR = "projects"


@dataclass(frozen=True)
class ClaudeCaptureResult:
    """Summary of a local Claude Code capture pass."""

    imported: int
    scanned_files: int
    skipped: int
    events: tuple[Event, ...] = ()


def capture_claude_sessions(
    *,
    workspace: str | Path,
    claude_home: str | Path | None = None,
    eventloom_path: str | Path = ".eventloom",
    session_id: str = "default",
    source: str = DEFAULT_CLAUDE_SOURCE,
    max_records_per_file: int | None = None,
) -> ClaudeCaptureResult:
    """Import locally written Claude Code session records into Eventloom observations."""
    root = Path(workspace).resolve()
    home = Path(claude_home) if claude_home is not None else _default_claude_home()
    eventlog = SessionManager(base_path=str(eventloom_path)).get(session_id).eventlog
    existing_refs = _existing_source_refs(eventlog)
    event_inputs: list[dict[str, Any]] = []
    imported = 0
    skipped = 0
    scanned_files = 0
    for path in _claude_session_paths(home):
        cwd = _session_cwd(path)
        if cwd is None or not _matches_workspace(cwd, root):
            continue
        rows = _read_jsonl(path)
        if not rows:
            continue
        scanned_files += 1
        pending_calls: dict[str, dict[str, Any]] = {}
        records = rows[-max_records_per_file:] if max_records_per_file is not None else rows
        for line_number, record in records:
            produced = _record_to_event_inputs(
                record,
                path=path,
                line_number=line_number,
                source=source,
                session_id=session_id,
                workspace=root,
                pending_calls=pending_calls,
            )
            new_events = 0
            for event_input in produced:
                ref = event_input["payload"]["claude_source_ref"]
                if ref in existing_refs:
                    continue
                event_inputs.append(event_input)
                existing_refs.add(ref)
                imported += 1
                new_events += 1
            if new_events == 0:
                skipped += 1
    events = eventlog.append_many(
        [
            {
                "event_type": event_input["event_type"],
                "actor": event_input["actor"],
                "payload": event_input["payload"],
                "thread": session_id,
            }
            for event_input in event_inputs
        ]
    )
    return ClaudeCaptureResult(
        imported=imported,
        scanned_files=scanned_files,
        skipped=skipped,
        events=tuple(events),
    )


def _default_claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))


def _claude_session_paths(home: Path) -> list[Path]:
    projects = home / CLAUDE_PROJECTS_DIR
    if not projects.is_dir():
        return []
    return sorted(projects.glob("*/*.jsonl"))


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append((line_number, value))
    except OSError:
        return []
    return records


def _session_cwd(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    cwd = value.get("cwd")
                    if isinstance(cwd, str):
                        return cwd
    except OSError:
        return None
    return None


def _matches_workspace(cwd: str, workspace: Path) -> bool:
    try:
        return Path(cwd).resolve() == workspace
    except OSError:
        return False


def _existing_source_refs(eventlog: EventLog) -> set[str]:
    refs: set[str] = set()
    for event in eventlog.read_all():
        ref = event.payload.get("claude_source_ref")
        if isinstance(ref, str):
            refs.add(ref)
    return refs


def _record_to_event_inputs(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    source: str,
    session_id: str,
    workspace: Path,
    pending_calls: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if record.get("type") not in ("user", "assistant"):
        return []
    if record.get("isMeta") is True:
        return []
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    role = message.get("role")
    if role not in ("user", "assistant"):
        return []
    blocks = _message_blocks(message.get("content"))
    events: list[dict[str, Any]] = []
    text = _text_from_blocks(blocks)
    if text:
        turn = build_transcript_turn_observation(
            role=str(role),
            content=text,
            session_id=session_id,
            source=source,
        )
        events.append(_with_ref(_with_capture_cues(turn, workspace=workspace), f"{path}:{line_number}:turn"))
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            ev = _tool_use_to_event(block, source=source, session_id=session_id, workspace=workspace)
            if ev is not None:
                events.append(_with_ref(ev, f"{path}:{line_number}:tool:{block.get('id')}"))
            tid = block.get("id")
            if isinstance(tid, str):
                pending_calls[tid] = {"name": block.get("name"), "input": block.get("input")}
        elif block_type == "tool_result":
            tuid = block.get("tool_use_id")
            call = pending_calls.get(tuid) if isinstance(tuid, str) else None
            if call is None:
                continue
            ev = _tool_result_to_event(block, call=call, source=source, session_id=session_id, workspace=workspace)
            if ev is not None:
                events.append(_with_ref(ev, f"{path}:{line_number}:result:{tuid}"))
    return events


def _message_blocks(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def _text_from_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


def _tool_use_to_event(
    block: dict[str, Any],
    *,
    source: str,
    session_id: str,
    workspace: Path,
) -> dict[str, Any] | None:
    name = block.get("name")
    if not isinstance(name, str):
        return None
    args = _safe_input(block.get("input"))
    if name in ("Edit", "Write", "NotebookEdit"):
        file_path = args.get("file_path") or args.get("notebook_path")
        if isinstance(file_path, str) and file_path:
            event = build_file_edit_observation(
                path=file_path,
                operation="modified",
                session_id=session_id,
                source=source,
                workspace=str(workspace),
                summary=f"Claude {name} edited {file_path}",
            )
            return _with_capture_cues(event, workspace=workspace, tool=name)
    event = build_tool_call_observation(
        tool_name=name,
        status="called",
        session_id=session_id,
        source=source,
        workspace=str(workspace),
        call_id=_optional_str(block.get("id")),
        arguments=args,
    )
    return _with_capture_cues(event, workspace=workspace, tool=name)


def _tool_result_to_event(
    block: dict[str, Any],
    *,
    call: dict[str, Any],
    source: str,
    session_id: str,
    workspace: Path,
) -> dict[str, Any] | None:
    name = call.get("name")
    if name != "Bash":
        return None
    command = _safe_input(call.get("input")).get("command")
    if not isinstance(command, str) or not command:
        return None
    output_text = _content_text(block.get("content"))
    exit_code = 1 if bool(block.get("is_error")) else 0
    event = build_command_observation(
        command=command,
        exit_code=exit_code,
        session_id=session_id,
        source=source,
        workspace=str(workspace),
        stdout=output_text,
    )
    return _with_capture_cues(event, workspace=workspace, tool=str(name))


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


def _safe_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _with_ref(event: dict[str, Any], ref: str) -> dict[str, Any]:
    event["payload"]["claude_source_ref"] = ref
    return event


def _with_capture_cues(
    event: dict[str, Any],
    *,
    workspace: Path,
    tool: str | None = None,
) -> dict[str, Any]:
    """Attach the encoding-specificity cue record capture actually knows.

    Workspace identity is always in hand here; the originating tool only for
    tool-call shaped records. Mission and session phase are not observable
    from Claude Code session logs, so they are honestly omitted rather than
    guessed.
    """
    cues = build_cue_record(workspace=str(workspace), tool=tool)
    if cues:
        event["payload"]["cues"] = cues
    return event


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None
