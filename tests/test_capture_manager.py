"""Tests for managed deterministic capture runtime helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zaxy.capture_manager import inspect_codex_capture, start_codex_capture, stop_codex_capture
from zaxy.event import EventLog


def _write_codex_config(workspace: Path) -> None:
    config = workspace / ".codex" / "zaxy-capture.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "codex_home": str(workspace / ".codex-home"),
                "eventloom_path": str(workspace / ".eventloom"),
                "session_id": "agent-1",
                "source": "codex-local",
                "workspace": str(workspace),
            }
        ),
        encoding="utf-8",
    )


def test_inspect_codex_capture_reports_latest_observation(tmp_path: Path) -> None:
    """Managed status should surface the latest imported observation when present."""
    _write_codex_config(tmp_path)
    event = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "codex-local", "content": "Captured local context."},
        thread="agent-1",
    )

    report = inspect_codex_capture(workspace=tmp_path)

    assert report["configured"] is True
    assert report["latest_observation"] == {
        "seq": event.seq,
        "timestamp": event.timestamp,
        "type": "transcript.turn",
        "thread": "agent-1",
        "source": "codex-local",
    }


@patch.object(subprocess, "Popen")
@patch("zaxy.hooks._iter_process_cmdlines")
def test_start_codex_capture_reuses_existing_watcher(
    mock_processes: MagicMock,
    mock_popen: MagicMock,
    tmp_path: Path,
) -> None:
    """Starting capture should not spawn a duplicate watcher for the same workspace."""
    _write_codex_config(tmp_path)
    mock_processes.return_value = [
        (
            321,
            [
                "python",
                "-m",
                "zaxy",
                "codex-capture",
                "--workspace",
                str(tmp_path),
                "--eventloom-path",
                str(tmp_path / ".eventloom"),
                "--watch",
            ],
        )
    ]

    result = start_codex_capture(workspace=tmp_path)

    assert result["started"] is False
    assert result["pid"] == 321
    assert "already running" in result["message"]
    mock_popen.assert_not_called()


def test_start_codex_capture_requires_repo_local_config(tmp_path: Path) -> None:
    """Starting managed capture should fail clearly before any process launch without config."""
    with pytest.raises(FileNotFoundError, match="zaxy-capture.json"):
        start_codex_capture(workspace=tmp_path)


@patch("os.kill")
@patch("zaxy.hooks._iter_process_cmdlines")
def test_stop_codex_capture_removes_stale_state_without_killing(
    mock_processes: MagicMock,
    mock_kill: MagicMock,
    tmp_path: Path,
) -> None:
    """A stale managed state file should be removed without killing an unrelated PID."""
    state_dir = tmp_path / ".eventloom" / "runtime"
    state_dir.mkdir(parents=True)
    state = state_dir / "codex-capture.json"
    state.write_text(
        json.dumps(
            {
                "client": "codex",
                "pid": 321,
                "workspace": str(tmp_path),
                "eventloom_path": str(tmp_path / ".eventloom"),
            }
        ),
        encoding="utf-8",
    )
    mock_processes.return_value = []

    result = stop_codex_capture(workspace=tmp_path)

    assert result["stopped"] is False
    assert "Removed stale" in result["message"]
    assert not state.exists()
    mock_kill.assert_not_called()
