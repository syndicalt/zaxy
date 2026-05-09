"""Tests for the static Eventloom/session viewer."""

from __future__ import annotations

from pathlib import Path

from zaxy.event import EventLog
from zaxy.viewer import build_viewer_model, render_viewer_html, write_viewer_html


def test_build_viewer_model_groups_file_events_by_session(tmp_path: Path) -> None:
    """The viewer model should summarize sessions and highlight important event classes."""
    log_path = tmp_path / "zaxy-default.jsonl"
    log = EventLog(log_path)
    log.append(
        "session.genesis",
        actor="zaxy",
        payload={
            "session_id": "zaxy-default",
            "workspace_type": "codebase",
            "root": str(tmp_path),
        },
        thread="zaxy-default",
    )
    log.append(
        "tool.call.completed",
        actor="zaxy",
        payload={
            "session_id": "zaxy-default",
            "tool_name": "memory_query",
            "status": "succeeded",
        },
        thread="zaxy-default",
    )

    model = build_viewer_model(log_path)

    assert model["source_path"] == str(log_path)
    assert model["total_events"] == 2
    assert model["sessions"][0]["session_id"] == "zaxy-default"
    assert model["sessions"][0]["event_count"] == 2
    assert model["sessions"][0]["bootstrap_count"] == 1
    assert model["sessions"][0]["lifecycle_count"] == 1
    assert [event["category"] for event in model["events"]] == ["bootstrap", "lifecycle"]


def test_build_viewer_model_reads_eventloom_directory(tmp_path: Path) -> None:
    """Directory input should include every JSONL session log."""
    EventLog(tmp_path / "alpha.jsonl").append(
        "session.genesis",
        actor="zaxy",
        payload={"session_id": "alpha"},
        thread="alpha",
    )
    EventLog(tmp_path / "beta.jsonl").append(
        "session.ended",
        actor="zaxy",
        payload={"session_id": "beta", "reason": "teardown"},
        thread="beta",
    )

    model = build_viewer_model(tmp_path)

    assert model["total_events"] == 2
    assert [session["session_id"] for session in model["sessions"]] == ["alpha", "beta"]


def test_render_viewer_html_escapes_payload_and_embeds_model(tmp_path: Path) -> None:
    """Generated HTML should be standalone and safe for payload text display."""
    log_path = tmp_path / "default.jsonl"
    EventLog(log_path).append(
        "workspace.instructions.discovered",
        actor="zaxy",
        payload={
            "session_id": "default",
            "summary": "<script>alert('x')</script>",
        },
        thread="default",
    )
    model = build_viewer_model(log_path)

    html = render_viewer_html(model)

    assert "Eventloom Session Viewer" in html
    assert "window.__ZAXY_VIEWER_DATA__" in html
    assert "<script>alert('x')</script>" not in html
    assert "\\u003cscript\\u003ealert" in html


def test_write_viewer_html_creates_parent_directories(tmp_path: Path) -> None:
    """write_viewer_html should create the output artifact and return its path."""
    log_path = tmp_path / "default.jsonl"
    EventLog(log_path).append("goal.created", actor="user", payload={"title": "Ship"})
    output = tmp_path / "reports" / "viewer.html"

    written = write_viewer_html(log_path, output)

    assert written == output
    assert output.exists()
    assert "Eventloom Session Viewer" in output.read_text(encoding="utf-8")
