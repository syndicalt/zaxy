"""Tests for deterministic local Claude Code session capture."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy import claude_capture as cc
from zaxy.claude_capture import capture_claude_sessions
from zaxy.event import EventLog


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _session_path(claude_home: Path, slug: str, uuid: str) -> Path:
    return claude_home / "projects" / slug / f"{uuid}.jsonl"


def test_capture_claude_sessions_imports_transcript_tool_command_and_file_edit(tmp_path: Path) -> None:
    """Claude local capture should convert session JSONL into first-class Eventloom observations."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cwd = str(workspace.resolve())
    claude_home = tmp_path / ".claude"
    session_path = _session_path(claude_home, "-repo", "session-1")
    _write_jsonl(
        session_path,
        [
            {"type": "user", "cwd": cwd, "message": {"role": "user", "content": "Run tests"}},
            {
                "type": "assistant",
                "cwd": cwd,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "secret reasoning"},
                        {"type": "text", "text": "Running the suite."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "pytest tests/test_example.py"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "cwd": cwd,
                "message": {
                    "role": "user",
                    "content": [{"tool_use_id": "toolu_1", "type": "tool_result", "content": "ok"}],
                },
            },
            {
                "type": "assistant",
                "cwd": cwd,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "more private reasoning"},
                        {"type": "text", "text": "Patching core."},
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "Edit",
                            "input": {"file_path": "src/zaxy/core.py", "old_string": "a", "new_string": "b"},
                        },
                    ],
                },
            },
        ],
    )

    result = capture_claude_sessions(
        workspace=workspace,
        claude_home=claude_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 6
    assert result.scanned_files == 1
    events = EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()
    assert [event.type for event in events] == [
        "transcript.turn",
        "transcript.turn",
        "tool.call.completed",
        "command.completed",
        "transcript.turn",
        "file.edit.applied",
    ]
    assert events[0].payload["role"] == "user"
    assert events[0].payload["content"] == "Run tests"
    assert events[1].payload["role"] == "assistant"
    assert events[1].payload["content"] == "Running the suite."
    assert "secret reasoning" not in events[1].payload["content"]
    assert events[2].payload["tool_name"] == "Bash"
    assert events[3].payload["command"] == "pytest tests/test_example.py"
    assert events[3].payload["exit_code"] == 0
    assert events[4].payload["content"] == "Patching core."
    assert events[5].payload["path"] == "src/zaxy/core.py"
    assert all(event.payload["source"] == "claude-local" for event in events)
    assert all(event.payload["claude_source_ref"].startswith(str(session_path)) for event in events)
    assert events[0].payload["claude_source_ref"].endswith(":turn")
    assert events[2].payload["claude_source_ref"] == f"{session_path}:2:tool:toolu_1"
    assert events[3].payload["claude_source_ref"] == f"{session_path}:3:result:toolu_1"
    assert events[5].payload["claude_source_ref"] == f"{session_path}:4:tool:toolu_2"
    # Capture enriches the encoding-specificity cue record with what it actually
    # knows: workspace identity always, the tool for tool-shaped records;
    # mission/phase are not observable and stay honestly absent.
    assert events[0].payload["cues"] == {"workspace": str(workspace)}
    assert events[2].payload["cues"] == {"workspace": str(workspace), "tool": "Bash"}
    assert events[3].payload["cues"] == {"workspace": str(workspace), "tool": "Bash"}
    assert events[5].payload["cues"] == {"workspace": str(workspace), "tool": "Edit"}
    assert all("mission" not in event.payload.get("cues", {}) for event in events)


def test_capture_claude_sessions_records_failed_bash_command(tmp_path: Path) -> None:
    """A tool_result flagged is_error should record a non-zero command exit signal."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cwd = str(workspace.resolve())
    claude_home = tmp_path / ".claude"
    _write_jsonl(
        _session_path(claude_home, "-repo", "session-1"),
        [
            {
                "type": "assistant",
                "cwd": cwd,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_9",
                            "name": "Bash",
                            "input": {"command": "make build"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "cwd": cwd,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "tool_use_id": "toolu_9",
                            "type": "tool_result",
                            "is_error": True,
                            "content": [{"type": "text", "text": "build failed"}],
                        }
                    ],
                },
            },
        ],
    )

    result = capture_claude_sessions(
        workspace=workspace,
        claude_home=claude_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 2
    events = EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()
    assert [event.type for event in events] == ["tool.call.completed", "command.completed"]
    assert events[1].payload["command"] == "make build"
    assert events[1].payload["exit_code"] == 1


def test_capture_claude_sessions_is_idempotent_by_source_ref(tmp_path: Path) -> None:
    """Repeated local capture should not duplicate already imported Claude records."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cwd = str(workspace.resolve())
    claude_home = tmp_path / ".claude"
    _write_jsonl(
        _session_path(claude_home, "-repo", "session-1"),
        [
            {"type": "user", "cwd": cwd, "message": {"role": "user", "content": "Remember this"}},
            {"type": "assistant", "cwd": cwd, "message": {"role": "assistant", "content": "Noted."}},
        ],
    )

    first = capture_claude_sessions(
        workspace=workspace,
        claude_home=claude_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )
    second = capture_claude_sessions(
        workspace=workspace,
        claude_home=claude_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert first.imported == 2
    assert second.imported == 0
    assert len(EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()) == 2


def test_capture_claude_sessions_can_limit_records_per_file_for_watch_mode(tmp_path: Path) -> None:
    """Watch-mode capture should be able to avoid backfilling a whole large transcript."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cwd = str(workspace.resolve())
    claude_home = tmp_path / ".claude"
    _write_jsonl(
        _session_path(claude_home, "-repo", "session-1"),
        [
            {"type": "user", "cwd": cwd, "message": {"role": "user", "content": "old turn"}},
            {"type": "user", "cwd": cwd, "message": {"role": "user", "content": "recent turn"}},
        ],
    )

    result = capture_claude_sessions(
        workspace=workspace,
        claude_home=claude_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
        max_records_per_file=1,
    )

    assert result.imported == 1
    events = EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all()
    assert [event.payload["content"] for event in events] == ["recent turn"]


def test_capture_claude_sessions_skips_sessions_without_matching_cwd(tmp_path: Path) -> None:
    """Session logs whose cwd does not match the workspace should not be imported."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    claude_home = tmp_path / ".claude"
    _write_jsonl(
        _session_path(claude_home, "-elsewhere", "session-1"),
        [
            {"type": "user", "cwd": str(other.resolve()), "message": {"role": "user", "content": "not mine"}},
        ],
    )

    result = capture_claude_sessions(
        workspace=workspace,
        claude_home=claude_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 0
    assert result.scanned_files == 0
    assert EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all() == []


def test_capture_claude_sessions_skips_unusable_records(tmp_path: Path) -> None:
    """Meta, non-conversation, and thinking-only records should not become memory."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cwd = str(workspace.resolve())
    claude_home = tmp_path / ".claude"
    _write_jsonl(
        _session_path(claude_home, "-repo", "session-1"),
        [
            {"type": "system", "cwd": cwd, "subtype": "init"},
            {"type": "user", "cwd": cwd, "isMeta": True, "message": {"role": "user", "content": "injected"}},
            {
                "type": "assistant",
                "cwd": cwd,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "private only"}],
                },
            },
            {"type": "user", "cwd": cwd, "message": {"role": "user", "content": "   "}},
        ],
    )

    result = capture_claude_sessions(
        workspace=workspace,
        claude_home=claude_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 0
    assert result.scanned_files == 1
    assert result.skipped == 4
    assert EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all() == []


def test_capture_claude_sessions_excludes_subagent_transcripts(tmp_path: Path) -> None:
    """Only top-level session files are scanned; deeper subagent logs are excluded."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cwd = str(workspace.resolve())
    claude_home = tmp_path / ".claude"
    subagent_path = claude_home / "projects" / "-repo" / "session-1" / "subagents" / "agent-1.jsonl"
    _write_jsonl(
        subagent_path,
        [
            {"type": "user", "cwd": cwd, "message": {"role": "user", "content": "subagent only"}},
        ],
    )

    result = capture_claude_sessions(
        workspace=workspace,
        claude_home=claude_home,
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result.imported == 0
    assert result.scanned_files == 0
    assert EventLog(workspace / ".eventloom" / "repo-default.jsonl").read_all() == []


def test_capture_claude_sessions_tolerates_missing_home(tmp_path: Path) -> None:
    """A non-existent Claude home should yield an empty capture, not an error."""
    workspace = tmp_path / "repo"
    workspace.mkdir()

    result = capture_claude_sessions(
        workspace=workspace,
        claude_home=tmp_path / "nope",
        eventloom_path=workspace / ".eventloom",
        session_id="repo-default",
    )

    assert result == result.__class__(imported=0, scanned_files=0, skipped=0, events=())


def test_read_jsonl_tolerates_missing_file_and_bad_lines(tmp_path: Path) -> None:
    """_read_jsonl returns [] on OSError and skips undecodable or non-object lines."""
    assert cc._read_jsonl(tmp_path / "nope.jsonl") == []
    path = tmp_path / "mixed.jsonl"
    path.write_text('not json\n[1, 2, 3]\n{"type": "user"}\n', encoding="utf-8")
    rows = cc._read_jsonl(path)
    assert rows == [(3, {"type": "user"})]


def test_session_cwd_scans_to_first_string_cwd(tmp_path: Path) -> None:
    """_session_cwd skips bad/cwd-less lines and returns the first string cwd, else None."""
    assert cc._session_cwd(tmp_path / "nope.jsonl") is None
    no_cwd = tmp_path / "no_cwd.jsonl"
    no_cwd.write_text('{"type": "mode"}\n{"type": "user"}\n', encoding="utf-8")
    assert cc._session_cwd(no_cwd) is None
    found = tmp_path / "found.jsonl"
    found.write_text('bad\n{"cwd": 5}\n{"cwd": "/repo"}\n', encoding="utf-8")
    assert cc._session_cwd(found) == "/repo"


def test_matches_workspace_compares_resolved_paths(tmp_path: Path) -> None:
    """_matches_workspace is true only when the resolved cwd equals the workspace."""
    workspace = (tmp_path / "repo").resolve()
    assert cc._matches_workspace(str(workspace), workspace) is True
    assert cc._matches_workspace(str(tmp_path / "other"), workspace) is False


def test_message_blocks_normalizes_string_list_and_other() -> None:
    """_message_blocks wraps strings, passes lists through, and rejects other types."""
    assert cc._message_blocks("hi") == [{"type": "text", "text": "hi"}]
    blocks = [{"type": "text", "text": "a"}]
    assert cc._message_blocks(blocks) is blocks
    assert cc._message_blocks(7) == []


def test_text_from_blocks_joins_only_text_blocks() -> None:
    """_text_from_blocks ignores non-dict and non-text blocks."""
    blocks = [
        "loose",
        {"type": "thinking", "thinking": "secret"},
        {"type": "text", "text": "one"},
        {"type": "text", "text": "two"},
        {"type": "text"},
    ]
    assert cc._text_from_blocks(blocks) == "one\ntwo"


def test_content_text_handles_string_list_and_other() -> None:
    """_content_text strips strings, joins text items, and returns '' otherwise."""
    assert cc._content_text("  out  ") == "out"
    assert cc._content_text([{"type": "text", "text": "a"}, {"type": "image"}, "x"]) == "a"
    assert cc._content_text(42) == ""


def test_safe_input_parses_dicts_and_json_objects_only() -> None:
    """_safe_input returns dicts directly, parses JSON-object strings, and rejects the rest."""
    assert cc._safe_input({"a": 1}) == {"a": 1}
    assert cc._safe_input('{"command": "ls"}') == {"command": "ls"}
    assert cc._safe_input("not json") == {}
    assert cc._safe_input("[1, 2]") == {}
    assert cc._safe_input(9) == {}


def test_optional_str_returns_string_or_none() -> None:
    """_optional_str passes strings through and maps non-strings to None."""
    assert cc._optional_str("id") == "id"
    assert cc._optional_str(123) is None


def test_tool_use_to_event_handles_bad_name_and_pathless_edit() -> None:
    """A non-string tool name is dropped; an Edit with no path falls through to a tool call."""
    workspace = Path("/repo")
    assert (
        cc._tool_use_to_event({"name": 5}, source="claude-local", session_id="s", workspace=workspace)
        is None
    )
    event = cc._tool_use_to_event(
        {"name": "Edit", "id": "t1", "input": {}},
        source="claude-local",
        session_id="s",
        workspace=workspace,
    )
    assert event is not None
    assert event["event_type"] == "tool.call.completed"
    assert event["payload"]["tool_name"] == "Edit"


def test_tool_result_to_event_only_records_bash_with_command() -> None:
    """Tool results pair into command observations only for Bash calls with a command."""
    workspace = Path("/repo")
    assert (
        cc._tool_result_to_event(
            {"content": "x"},
            call={"name": "Read", "input": {"file_path": "a"}},
            source="claude-local",
            session_id="s",
            workspace=workspace,
        )
        is None
    )
    assert (
        cc._tool_result_to_event(
            {"content": "x"},
            call={"name": "Bash", "input": {}},
            source="claude-local",
            session_id="s",
            workspace=workspace,
        )
        is None
    )
    event = cc._tool_result_to_event(
        {"content": "boom", "is_error": True},
        call={"name": "Bash", "input": {"command": "make"}},
        source="claude-local",
        session_id="s",
        workspace=workspace,
    )
    assert event is not None
    assert event["event_type"] == "command.completed"
    assert event["payload"]["command"] == "make"
    assert event["payload"]["exit_code"] == 1
