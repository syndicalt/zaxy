"""Tests for per-turn memory-state injection (the UserPromptSubmit recall lever)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.event import EventLog
from zaxy.hooks import hook_event_type, render_hook_config
from zaxy.memory_persistence import build_injection_context, record_memory_activity

runner = CliRunner()


def _make_stale(tmp_path):  # type: ignore[no-untyped-def]
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "agent.jsonl")
    for index in range(12):
        log.append("transcript.turn", actor="codex", payload={"content": f"turn {index}"}, thread="agent")
    return eventloom


def _make_fresh(tmp_path):  # type: ignore[no-untyped-def]
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "agent.jsonl")
    for index in range(3):
        log.append("transcript.turn", actor="codex", payload={"content": f"turn {index}"}, thread="agent")
    record_memory_activity(eventloom, session_id="agent", activity="checkout", source="mcp")
    return eventloom


def test_injection_context_when_stale(tmp_path) -> None:  # type: ignore[no-untyped-def]
    eventloom = _make_stale(tmp_path)
    ctx = build_injection_context(eventloom, session_id="agent")
    assert ctx is not None
    assert "agent" in ctx
    assert "memory_checkout" in ctx
    assert "stale" in ctx


def test_injection_silent_when_fresh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    eventloom = _make_fresh(tmp_path)
    assert build_injection_context(eventloom, session_id="agent") is None


def test_hook_event_type_user_prompt_submit() -> None:
    assert hook_event_type("user-prompt-submit") == "hook.user_prompt_submitted"
    assert hook_event_type("user_prompt_submit") == "hook.user_prompt_submitted"


def test_render_hook_config_emits_user_prompt_submit() -> None:
    config = json.loads(render_hook_config("claude-code", eventloom_path=".eventloom", domain="zaxy"))
    hooks = config["hooks"]
    # New lever present, existing levers preserved.
    assert "UserPromptSubmit" in hooks
    assert "Stop" in hooks and "PreCompact" in hooks
    cmd = hooks["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert "hook-event user-prompt-submit" in cmd
    assert "zaxy-default" in cmd  # canonical session derived from domain


def test_cli_emits_additional_context_when_stale(tmp_path) -> None:  # type: ignore[no-untyped-def]
    eventloom = _make_stale(tmp_path)
    result = runner.invoke(
        app,
        ["hook-event", "user-prompt-submit", "--eventloom-path", str(eventloom),
         "--session-id", "agent", "--source", "claude-code"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip())
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "memory_checkout" in payload["hookSpecificOutput"]["additionalContext"]


def test_cli_silent_when_fresh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    eventloom = _make_fresh(tmp_path)
    result = runner.invoke(
        app,
        ["hook-event", "user-prompt-submit", "--eventloom-path", str(eventloom),
         "--session-id", "agent", "--source", "claude-code"],
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == ""
