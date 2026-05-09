"""Tests for first-run onboarding orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from zaxy.event import EventLog
from zaxy.onboarding import run_onboarding


@pytest.mark.asyncio
async def test_run_onboarding_writes_requested_configs_and_registers_session(tmp_path: Path) -> None:
    """Onboarding should compose existing primitives into one idempotent first-run flow."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    eventloom_path = workspace / ".eventloom"
    mcp_output = workspace / "mcp.json"
    hook_output = workspace / ".claude" / "settings.local.json"
    local_profile_output = workspace / ".env.local"
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "codebase"
    fabric.ensure_session_initialized.return_value.confidence = 0.7
    fabric.ensure_session_initialized.return_value.signals = ["pyproject.toml"]
    fabric.ensure_session_initialized.return_value.instructions_profile = "codebase"
    fabric.close = AsyncMock()

    result = await run_onboarding(
        workspace,
        eventloom_path=eventloom_path,
        domain="demo",
        session_id="demo-default",
        mcp_client="claude-desktop",
        mcp_output=mcp_output,
        hook_client="claude-code",
        hook_output=hook_output,
        local_profile_output=local_profile_output,
        fabric_factory=lambda eventloom_path: fabric,
    )

    assert result.status == "ok"
    assert result.session_id == "demo-default"
    assert result.profile["workspace_type"] == "codebase"
    assert [step.name for step in result.steps] == [
        "eventloom",
        "local_profile",
        "mcp_config",
        "hook_config",
        "session_genesis",
        "heartbeat",
        "doctor",
        "hook_status",
    ]
    assert json.loads(mcp_output.read_text(encoding="utf-8"))["mcpServers"]["zaxy"]["env"]["EVENTLOOM_THREAD"] == "demo-default"
    assert "zaxy hook-event stop" in hook_output.read_text(encoding="utf-8")
    assert "RERANKER_PROVIDER=lexical" in local_profile_output.read_text(encoding="utf-8")
    fabric.ensure_session_initialized.assert_awaited_once_with(workspace, session_id="demo-default")
    fabric.close.assert_awaited_once()
    events = EventLog(eventloom_path / "demo-default.jsonl").read_all()
    assert events[-1].type == "hook.heartbeat"
    assert events[-1].payload["source"] == "zaxy-init"


@pytest.mark.asyncio
async def test_run_onboarding_is_non_destructive_for_existing_outputs(tmp_path: Path) -> None:
    """Onboarding should refuse to overwrite generated files unless force is explicit."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    output = workspace / "mcp.json"
    output.write_text("existing\n", encoding="utf-8")
    fabric = MagicMock()
    fabric.close = AsyncMock()

    with pytest.raises(FileExistsError):
        await run_onboarding(
            workspace,
            eventloom_path=workspace / ".eventloom",
            domain="demo",
            mcp_client="claude-desktop",
            mcp_output=output,
            fabric_factory=lambda eventloom_path: fabric,
        )

    assert output.read_text(encoding="utf-8") == "existing\n"
    fabric.close.assert_not_called()


@pytest.mark.asyncio
async def test_run_onboarding_rejects_output_without_matching_client(tmp_path: Path) -> None:
    """Output paths should not be accepted when their renderer is not selected."""
    workspace = tmp_path / "repo"
    workspace.mkdir()

    with pytest.raises(ValueError, match="mcp_client is required"):
        await run_onboarding(
            workspace,
            eventloom_path=workspace / ".eventloom",
            domain="demo",
            mcp_output=workspace / "mcp.json",
        )

    with pytest.raises(ValueError, match="hook_client is required"):
        await run_onboarding(
            workspace,
            eventloom_path=workspace / ".eventloom",
            domain="demo",
            hook_output=workspace / ".claude" / "settings.local.json",
        )
