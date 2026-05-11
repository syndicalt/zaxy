"""Tests for deterministic local Codex session capture."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from zaxy.codex_capture import (
    _git_status_operation,
    _parse_git_status_line,
    capture_codex_sessions,
    write_codex_capture_config,
)
from zaxy.event import EventLog


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_capture_codex_sessions_imports_transcript_tool_command_and_file_edit(tmp_path: Path) -> None:
    """Codex local capture should convert session JSONL into first-class Eventloom observations."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex-home"
    session_path = codex_home / "sessions" / "2026" / "05" / "11" / "rollout.jsonl"
    _write_jsonl(
        session_path,
        [
            {
                "timestamp": "2026-05-11T01:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "codex-session-1",
                    "cwd": str(workspace),
                    "cli_version": "0.128.0",
                    "source": "cli",
                },
            },
            {
                "timestamp": "2026-05-11T01:00:01.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "Run tests"},
            },
            {
                "timestamp": "2026-05-11T01:00:02.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call-1",
                    "arguments": json.dumps({"cmd": "pytest tests/test_example.py"}),
                },
            },
            {
                "timestamp": "2026-05-11T01:00:03.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "Process exited with code 0\nOutput:\nok\n",
                },
            },
            {
                "timestamp": "2026-05-11T01:00:04.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "apply_patch",
                    "call_id": "call-2",
                    "arguments": "*** Begin Patch\n*** Update File: src/zaxy/core.py\n@@\n-pass\n+ok\n*** End Patch\n",
                },
            },
            {
                "timestamp": "2026-05-11T01:00:05.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Tests passed."}],
                },
            },
        ],
    )

    result = capture_codex_sessions(
        workspace=workspace,
        codex_home=codex_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 6
    assert result.scanned_files == 1
    events = EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()
    assert [event.type for event in events] == [
        "hook.session_started",
        "transcript.turn",
        "tool.call.completed",
        "command.completed",
        "file.edit.applied",
        "transcript.turn",
    ]
    assert events[1].payload["role"] == "user"
    assert events[1].payload["content"] == "Run tests"
    assert events[2].payload["tool_name"] == "exec_command"
    assert events[3].payload["command"] == "pytest tests/test_example.py"
    assert events[3].payload["exit_code"] == 0
    assert events[4].payload["path"] == "src/zaxy/core.py"
    assert events[5].payload["role"] == "assistant"
    assert all(event.payload["source"] == "codex-local" for event in events)
    assert all(event.payload["codex_source_ref"].startswith(str(session_path)) for event in events)


def test_capture_codex_sessions_is_idempotent_by_source_ref(tmp_path: Path) -> None:
    """Repeated local capture should not duplicate already imported Codex records."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_jsonl(
        codex_home / "sessions" / "2026" / "05" / "11" / "rollout.jsonl",
        [
            {
                "timestamp": "2026-05-11T01:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "codex-session-1", "cwd": str(workspace)},
            },
            {
                "timestamp": "2026-05-11T01:00:01.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "Remember this"},
            },
        ],
    )

    first = capture_codex_sessions(
        workspace=workspace,
        codex_home=codex_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )
    second = capture_codex_sessions(
        workspace=workspace,
        codex_home=codex_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert first.imported == 2
    assert second.imported == 0
    assert len(EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()) == 2


def test_capture_codex_sessions_can_limit_records_per_file_for_watch_mode(tmp_path: Path) -> None:
    """Watch-mode capture should be able to avoid backfilling a whole large transcript."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_jsonl(
        codex_home / "sessions" / "2026" / "05" / "11" / "rollout.jsonl",
        [
            {
                "timestamp": "2026-05-11T01:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "codex-session-1", "cwd": str(workspace)},
            },
            {
                "timestamp": "2026-05-11T01:00:01.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "old turn"},
            },
            {
                "timestamp": "2026-05-11T01:00:02.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "recent turn"},
            },
        ],
    )

    result = capture_codex_sessions(
        workspace=workspace,
        codex_home=codex_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
        max_records_per_file=1,
    )

    assert result.imported == 1
    events = EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()
    assert [event.payload["content"] for event in events] == ["recent turn"]


def test_write_codex_capture_config_records_local_observer_settings(tmp_path: Path) -> None:
    """local-codex onboarding should have a safe repo-local capture config to detect."""
    path = write_codex_capture_config(
        workspace=tmp_path,
        eventloom_path=tmp_path / ".eventloom",
        session_id="repo-default",
        codex_home=tmp_path / "codex-home",
        force=False,
    )

    assert path == tmp_path / ".codex" / "zaxy-capture.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "client": "codex",
        "capture": "local-session-jsonl",
        "codex_home": str(tmp_path / "codex-home"),
        "eventloom_path": str(tmp_path / ".eventloom"),
        "session_id": "repo-default",
        "source": "codex-local",
        "workspace": str(tmp_path),
    }


def test_capture_codex_sessions_records_git_file_edits(tmp_path: Path) -> None:
    """Local Codex capture should observe changed workspace files even when tool logs omit edits."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    changed = workspace / "src" / "zaxy" / "core.py"
    changed.parent.mkdir(parents=True)
    changed.write_text("changed\n", encoding="utf-8")

    result = capture_codex_sessions(
        workspace=workspace,
        codex_home=tmp_path / "codex-home",
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 1
    events = EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()
    assert events[0].type == "file.edit.applied"
    assert events[0].payload["path"] == "src/zaxy/core.py"
    assert events[0].payload["operation"] == "untracked"
    assert events[0].payload["source"] == "codex-local"
    assert events[0].payload["codex_source_ref"].startswith("git-status:??:src/zaxy/core.py:")


def test_capture_codex_sessions_skips_unusable_codex_records(tmp_path: Path) -> None:
    """Malformed, unrelated, and incomplete Codex rows should not become memory."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex-home"
    sessions = codex_home / "sessions" / "2026" / "05" / "11"
    sessions.mkdir(parents=True)
    (sessions / "empty.jsonl").write_text("", encoding="utf-8")
    (sessions / "broken.jsonl").write_text("{not json}\n[]\n", encoding="utf-8")
    _write_jsonl(
        sessions / "other-workspace.jsonl",
        [
            {
                "timestamp": "2026-05-11T01:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "other", "cwd": str(tmp_path / "other")},
            },
            {
                "timestamp": "2026-05-11T01:00:01.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "not ours"},
            },
        ],
    )
    _write_jsonl(
        sessions / "unusable.jsonl",
        [
            {
                "timestamp": "2026-05-11T01:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "codex-session-1", "cwd": str(workspace)},
            },
            {"timestamp": "2026-05-11T01:00:01.000Z", "type": "event_msg", "payload": {}},
            {
                "timestamp": "2026-05-11T01:00:02.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "   "},
            },
            {
                "timestamp": "2026-05-11T01:00:03.000Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "developer", "content": "skip"},
            },
            {
                "timestamp": "2026-05-11T01:00:04.000Z",
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "missing", "output": "none"},
            },
            {
                "timestamp": "2026-05-11T01:00:05.000Z",
                "type": "response_item",
                "payload": {"type": "unknown"},
            },
            {"timestamp": "2026-05-11T01:00:06.000Z", "type": "turn_context", "payload": {}},
        ],
    )

    result = capture_codex_sessions(
        workspace=workspace,
        codex_home=codex_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 1
    assert result.scanned_files == 1
    assert result.skipped == 6
    assert EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()[0].type == "hook.session_started"


def test_capture_codex_sessions_handles_codex_edge_shapes(tmp_path: Path) -> None:
    """Codex capture should tolerate partial tool records without failing the pass."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_jsonl(
        codex_home / "sessions" / "2026" / "05" / "11" / "rollout.jsonl",
        [
            {
                "timestamp": "2026-05-11T01:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "codex-session-1", "cwd": str(workspace)},
            },
            {"timestamp": "2026-05-11T01:00:01.000Z", "type": "response_item", "payload": "bad"},
            {
                "timestamp": "2026-05-11T01:00:02.000Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": 123},
            },
            {
                "timestamp": "2026-05-11T01:00:03.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": ["bad", {"type": "output_text", "text": "kept"}],
                },
            },
            {
                "timestamp": "2026-05-11T01:00:04.000Z",
                "type": "response_item",
                "payload": {"type": "function_call", "name": 123, "call_id": "bad-name"},
            },
            {
                "timestamp": "2026-05-11T01:00:05.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "apply_patch",
                    "call_id": "empty-patch",
                    "arguments": "*** Begin Patch\n*** End Patch\n",
                },
            },
            {
                "timestamp": "2026-05-11T01:00:06.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "memory_query",
                    "call_id": "memory-call",
                    "arguments": "{not json}",
                },
            },
            {
                "timestamp": "2026-05-11T01:00:07.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "memory-call",
                    "output": "ok",
                },
            },
            {
                "timestamp": "2026-05-11T01:00:08.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "exec-no-exit",
                    "arguments": {"cmd": "pytest"},
                },
            },
            {
                "timestamp": "2026-05-11T01:00:09.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "exec-no-exit",
                    "output": "still running",
                },
            },
            {
                "timestamp": "2026-05-11T01:00:10.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "exec-no-cmd",
                    "arguments": 42,
                },
            },
            {
                "timestamp": "2026-05-11T01:00:11.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "exec-no-cmd",
                    "output": "Process exited with code 0",
                },
            },
        ],
    )

    result = capture_codex_sessions(
        workspace=workspace,
        codex_home=codex_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 5
    assert result.skipped == 7
    events = EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()
    assert [event.type for event in events] == [
        "hook.session_started",
        "transcript.turn",
        "tool.call.completed",
        "tool.call.completed",
        "tool.call.completed",
    ]
    assert events[1].payload["content"] == "kept"


def test_capture_codex_sessions_skips_sessions_without_matching_cwd(tmp_path: Path) -> None:
    """Session logs without a usable matching cwd should not be imported."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_jsonl(
        codex_home / "sessions" / "2026" / "05" / "11" / "no-cwd.jsonl",
        [
            {"timestamp": "2026-05-11T01:00:00.000Z", "type": "session_meta", "payload": {"cwd": None}},
            {
                "timestamp": "2026-05-11T01:00:01.000Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "skip"},
            },
        ],
    )

    result = capture_codex_sessions(
        workspace=workspace,
        codex_home=codex_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 0
    assert result.scanned_files == 0
    assert EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all() == []


def test_capture_codex_sessions_records_apply_patch_function_call(tmp_path: Path) -> None:
    """If a Codex log exposes apply_patch, capture it as file-edit metadata."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex-home"
    _write_jsonl(
        codex_home / "sessions" / "2026" / "05" / "11" / "rollout.jsonl",
        [
            {
                "timestamp": "2026-05-11T01:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "codex-session-1", "cwd": str(workspace)},
            },
            {
                "timestamp": "2026-05-11T01:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "apply_patch",
                    "call_id": "patch-1",
                    "arguments": "*** Begin Patch\n*** Delete File: old.py\n*** End Patch\n",
                },
            },
        ],
    )

    result = capture_codex_sessions(
        workspace=workspace,
        codex_home=codex_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 2
    events = EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()
    assert events[1].type == "file.edit.applied"
    assert events[1].payload["path"] == "old.py"
    assert events[1].payload["files"] == ["old.py"]


def test_git_status_parser_helpers_cover_repo_edge_cases() -> None:
    """Git status parsing should handle short, rename, and operation variants."""
    assert _parse_git_status_line("") is None
    assert _parse_git_status_line(" M") is None
    assert _parse_git_status_line("R  old.py -> new.py") == ("R ", "new.py")
    assert _git_status_operation("??") == "untracked"
    assert _git_status_operation(" D") == "deleted"
    assert _git_status_operation("R ") == "renamed"
    assert _git_status_operation("A ") == "added"
    assert _git_status_operation(" M") == "modified"
