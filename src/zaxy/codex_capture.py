"""Deterministic local capture for Codex session JSONL logs."""

from __future__ import annotations

import json
import os
import re
import subprocess
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
from zaxy.session import SessionManager

CODEX_CAPTURE_CONFIG = ".codex/zaxy-capture.json"
DEFAULT_CODEX_SOURCE = "codex-local"
_EXIT_CODE_PATTERN = re.compile(r"Process exited with code (-?\d+)")
_PATCH_FILE_PATTERN = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class CodexCaptureResult:
    """Summary of a local Codex capture pass."""

    imported: int
    scanned_files: int
    skipped: int
    events: tuple[Event, ...] = ()


def write_codex_capture_config(
    *,
    workspace: str | Path,
    eventloom_path: str | Path,
    session_id: str,
    codex_home: str | Path | None = None,
    source: str = DEFAULT_CODEX_SOURCE,
    force: bool = False,
) -> Path:
    """Write repo-local Codex capture settings owned by Zaxy."""
    root = Path(workspace)
    target = root / CODEX_CAPTURE_CONFIG
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to overwrite")
    payload = {
        "client": "codex",
        "capture": "local-session-jsonl",
        "codex_home": str(Path(codex_home) if codex_home is not None else _default_codex_home()),
        "eventloom_path": str(Path(eventloom_path)),
        "session_id": session_id,
        "source": source,
        "workspace": str(root),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def capture_codex_sessions(
    *,
    workspace: str | Path,
    codex_home: str | Path | None = None,
    eventloom_path: str | Path = ".eventloom",
    session_id: str = "default",
    source: str = DEFAULT_CODEX_SOURCE,
    max_records_per_file: int | None = None,
) -> CodexCaptureResult:
    """Import locally written Codex session records into Eventloom observations."""
    root = Path(workspace).resolve()
    home = Path(codex_home) if codex_home is not None else _default_codex_home()
    eventlog = SessionManager(base_path=str(eventloom_path)).get(session_id).eventlog
    existing_refs = _existing_source_refs(eventlog)
    event_inputs: list[dict[str, Any]] = []
    imported = 0
    skipped = 0
    scanned_files = 0
    for path in _codex_session_paths(home):
        rows = _read_jsonl(path)
        if not rows:
            continue
        meta = _session_meta(rows)
        if not _matches_workspace(meta, root):
            continue
        scanned_files += 1
        pending_calls: dict[str, dict[str, Any]] = {}
        records = rows[-max_records_per_file:] if max_records_per_file is not None else rows
        for line_number, record in records:
            ref = f"{path}:{line_number}"
            if ref in existing_refs:
                skipped += 1
                continue
            event_input = _record_to_event_input(
                record,
                ref=ref,
                source=source,
                session_id=session_id,
                workspace=root,
                pending_calls=pending_calls,
            )
            if event_input is None:
                skipped += 1
                continue
            event_inputs.append(event_input)
            existing_refs.add(ref)
            imported += 1
    for event_input in _git_file_edit_events(root, session_id=session_id, source=source, existing_refs=existing_refs):
        event_inputs.append(event_input)
        existing_refs.add(event_input["payload"]["codex_source_ref"])
        imported += 1
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
    return CodexCaptureResult(
        imported=imported,
        scanned_files=scanned_files,
        skipped=skipped,
        events=tuple(events),
    )


def _default_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))


def _codex_session_paths(codex_home: Path) -> list[Path]:
    sessions = codex_home / "sessions"
    if not sessions.is_dir():
        return []
    return sorted(sessions.glob("**/*.jsonl"))


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


def _session_meta(records: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    for _, record in records:
        payload = record.get("payload")
        if record.get("type") == "session_meta" and isinstance(payload, dict):
            return payload
    return {}


def _matches_workspace(meta: dict[str, Any], workspace: Path) -> bool:
    cwd = meta.get("cwd")
    if not isinstance(cwd, str):
        return False
    try:
        return Path(cwd).resolve() == workspace
    except OSError:
        return False


def _existing_source_refs(eventlog: EventLog) -> set[str]:
    refs: set[str] = set()
    for event in eventlog.read_all():
        ref = event.payload.get("codex_source_ref")
        if isinstance(ref, str):
            refs.add(ref)
    return refs


def _record_to_event_input(
    record: dict[str, Any],
    *,
    ref: str,
    source: str,
    session_id: str,
    workspace: Path,
    pending_calls: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    record_type = record.get("type")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if record_type == "session_meta":
        event = {
            "event_type": "hook.session_started",
            "actor": "zaxy-hook",
            "payload": {
                "trigger": "session-start",
                "source": source,
                "workspace": str(workspace),
                "codex_session_id": payload.get("id"),
                "codex_cli_version": payload.get("cli_version"),
            },
        }
        return _with_ref(event, ref)
    if record_type == "event_msg" and payload.get("type") == "user_message":
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            return None
        event = build_transcript_turn_observation(
            role="user",
            content=message,
            session_id=session_id,
            source=source,
        )
        return _with_ref(event, ref)
    if record_type != "response_item":
        return None
    item_type = payload.get("type")
    if item_type == "message":
        role = payload.get("role")
        content = _message_content_text(payload.get("content"))
        if role not in {"user", "assistant"} or not content:
            return None
        event = build_transcript_turn_observation(
            role=str(role),
            content=content,
            session_id=session_id,
            source=source,
        )
        return _with_ref(event, ref)
    if item_type == "function_call":
        call_id = payload.get("call_id")
        if isinstance(call_id, str):
            pending_calls[call_id] = payload
        return _function_call_to_event(payload, ref=ref, source=source, session_id=session_id, workspace=workspace)
    if item_type == "function_call_output":
        call_id = payload.get("call_id")
        call = pending_calls.get(call_id) if isinstance(call_id, str) else None
        if call is None:
            return None
        return _function_output_to_event(
            call,
            payload,
            ref=ref,
            source=source,
            session_id=session_id,
            workspace=workspace,
        )
    return None


def _function_call_to_event(
    payload: dict[str, Any],
    *,
    ref: str,
    source: str,
    session_id: str,
    workspace: Path,
) -> dict[str, Any] | None:
    name = payload.get("name")
    if not isinstance(name, str):
        return None
    if name == "apply_patch":
        files = _patch_files(payload.get("arguments"))
        if not files:
            return None
        event = build_file_edit_observation(
            path=files[0],
            operation="modified",
            session_id=session_id,
            source=source,
            workspace=str(workspace),
            summary=f"Codex apply_patch touched {len(files)} file(s)",
            line_count=None,
        )
        event["payload"]["files"] = files
        return _with_ref(event, ref)
    event = build_tool_call_observation(
        tool_name=name,
        status="called",
        session_id=session_id,
        source=source,
        workspace=str(workspace),
        call_id=_optional_str(payload.get("call_id")),
        arguments=_safe_arguments(payload.get("arguments")),
    )
    return _with_ref(event, ref)


def _function_output_to_event(
    call: dict[str, Any],
    output: dict[str, Any],
    *,
    ref: str,
    source: str,
    session_id: str,
    workspace: Path,
) -> dict[str, Any] | None:
    name = call.get("name")
    if name != "exec_command":
        return None
    command = _safe_arguments(call.get("arguments")).get("cmd")
    output_text = output.get("output")
    if not isinstance(command, str) or not isinstance(output_text, str):
        return None
    exit_code_match = _EXIT_CODE_PATTERN.search(output_text)
    if exit_code_match is None:
        return None
    event = build_command_observation(
        command=command,
        exit_code=int(exit_code_match.group(1)),
        session_id=session_id,
        source=source,
        workspace=str(workspace),
        stdout=output_text,
    )
    return _with_ref(event, ref)


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


def _safe_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _patch_files(arguments: Any) -> list[str]:
    patch = arguments if isinstance(arguments, str) else ""
    return [match.strip() for match in _PATCH_FILE_PATTERN.findall(patch) if match.strip()]


def _with_ref(event: dict[str, Any], ref: str) -> dict[str, Any]:
    event["payload"]["codex_source_ref"] = ref
    return event


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _git_file_edit_events(
    workspace: Path,
    *,
    session_id: str,
    source: str,
    existing_refs: set[str],
) -> list[dict[str, Any]]:
    if not (workspace / ".git").exists():
        return []
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    events: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parsed = _parse_git_status_line(line)
        if parsed is None:
            continue
        status, relative_path = parsed
        path = workspace / relative_path
        ref = _git_file_ref(status, relative_path, path)
        if ref in existing_refs:
            continue
        event = build_file_edit_observation(
            path=relative_path,
            operation=_git_status_operation(status),
            session_id=session_id,
            source=source,
            workspace=str(workspace),
            summary=f"Git workspace status {status}",
        )
        event["payload"]["git_status"] = status
        event["payload"]["codex_source_ref"] = ref
        events.append(event)
    return events


def _parse_git_status_line(line: str) -> tuple[str, str] | None:
    if len(line) < 4:
        return None
    status = line[:2]
    raw_path = line[3:]
    if " -> " in raw_path:
        raw_path = raw_path.rsplit(" -> ", maxsplit=1)[1]
    relative_path = raw_path.strip()
    if not relative_path:
        return None
    return status, relative_path


def _git_file_ref(status: str, relative_path: str, path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return f"git-status:{status}:{relative_path}:missing"
    return f"git-status:{status}:{relative_path}:{stat.st_mtime_ns}:{stat.st_size}"


def _git_status_operation(status: str) -> str:
    if status == "??":
        return "untracked"
    if "D" in status:
        return "deleted"
    if "R" in status:
        return "renamed"
    if "A" in status:
        return "added"
    return "modified"
