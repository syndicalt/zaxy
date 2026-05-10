"""Tests for first-run onboarding orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from zaxy.event import EventLog
from zaxy.onboarding import (
    OnboardingResult,
    OnboardingStep,
    apply_onboarding_preset,
    format_onboarding_result,
    run_onboarding,
)


class FakeRuntime:
    """Runtime double for onboarding infra actions."""

    def __init__(self, *, status: str = "ok", message: str = "Neo4j reachable") -> None:
        self.status = status
        self.message = message
        self.checked = False
        self.started = False

    def check(self) -> object:
        self.checked = True
        return {"status": self.status, "message": self.message}

    def ensure_available(self) -> None:
        self.started = True


def test_apply_onboarding_preset_expands_local_claude_defaults(tmp_path: Path) -> None:
    """local-claude should expand to the documented non-destructive golden path."""
    options = apply_onboarding_preset(
        "local-claude",
        workspace=tmp_path,
        mcp_client=None,
        mcp_output=None,
        hook_client=None,
        hook_output=None,
        local_profile_output=None,
        infra="none",
    )

    assert options["mcp_client"] == "claude-desktop"
    assert options["mcp_output"] == tmp_path / "zaxy-mcp.json"
    assert options["hook_client"] == "claude-code"
    assert options["hook_output"] == tmp_path / ".claude" / "settings.local.json"
    assert options["local_profile_output"] == tmp_path / ".env.local"
    assert options["infra"] == "check"


def test_apply_onboarding_preset_preserves_explicit_overrides(tmp_path: Path) -> None:
    """Preset expansion should never override explicit user choices."""
    options = apply_onboarding_preset(
        "local-claude",
        workspace=tmp_path,
        mcp_client="cursor",
        mcp_output=tmp_path / "cursor.json",
        hook_client=None,
        hook_output=None,
        local_profile_output=None,
        infra="start",
    )

    assert options["mcp_client"] == "cursor"
    assert options["mcp_output"] == tmp_path / "cursor.json"
    assert options["hook_client"] == "claude-code"
    assert options["infra"] == "start"


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
        zaxy_executable="/opt/zaxy/bin/zaxy",
        fabric_factory=lambda eventloom_path: fabric,
    )

    assert result.status == "ok"
    assert result.session_id == "demo-default"
    assert result.profile["workspace_type"] == "codebase"
    assert result.doctor["status"] == "warning"
    assert any(check["name"] == "observation_coverage" for check in result.doctor["checks"])
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
    mcp_config = json.loads(mcp_output.read_text(encoding="utf-8"))["mcpServers"]["zaxy"]
    assert mcp_config["command"] == "/opt/zaxy/bin/zaxy"
    assert mcp_config["env"]["EVENTLOOM_THREAD"] == "demo-default"
    assert "zaxy hook-event stop" in hook_output.read_text(encoding="utf-8")
    assert "RERANKER_PROVIDER=lexical" in local_profile_output.read_text(encoding="utf-8")
    fabric.ensure_session_initialized.assert_awaited_once_with(workspace, session_id="demo-default")
    fabric.close.assert_awaited_once()
    events = EventLog(eventloom_path / "demo-default.jsonl").read_all()
    assert events[-1].type == "hook.heartbeat"
    assert events[-1].payload["source"] == "zaxy-init"
    assert result.next_steps[0] == f"Add {mcp_output} to your claude-desktop MCP client config."
    assert result.next_steps[1] == "Restart the MCP client so it loads the Zaxy server config."
    assert f"Run zaxy hook-status --eventloom-path {eventloom_path}" in result.next_steps
    assert (
        f"Optional LLM packet capture: run zaxy packet-analyzer --eventloom-path {eventloom_path} "
        f"--session-id demo-default --upstream-base-url <provider-v1-url>."
    ) in result.next_steps
    assert (
        f"Optional packet projection: run zaxy packet-project --eventloom-path {eventloom_path} "
        "--session-id demo-default --watch."
    ) in result.next_steps


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


@pytest.mark.asyncio
async def test_run_onboarding_can_check_infra_without_starting_it(tmp_path: Path) -> None:
    """Infra check should report local runtime posture without mutating Docker state."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "generic_workspace"
    fabric.ensure_session_initialized.return_value.confidence = 0.2
    fabric.ensure_session_initialized.return_value.signals = []
    fabric.ensure_session_initialized.return_value.instructions_profile = "generic"
    fabric.close = AsyncMock()
    runtime = FakeRuntime(status="warning", message="Neo4j is not reachable; Docker is available")

    result = await run_onboarding(
        workspace,
        eventloom_path=workspace / ".eventloom",
        domain="demo",
        infra="check",
        fabric_factory=lambda eventloom_path: fabric,
        runtime_factory=lambda: runtime,
    )

    assert runtime.checked is True
    assert runtime.started is False
    infra_step = next(step for step in result.steps if step.name == "infra")
    assert infra_step.status == "warning"
    assert "Docker is available" in infra_step.message
    assert any(f"Run zaxy init {workspace} --infra start" in step for step in result.next_steps)


@pytest.mark.asyncio
async def test_run_onboarding_can_start_explicit_local_infra(tmp_path: Path) -> None:
    """Infra start should call the existing runtime bootstrap path explicitly."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "generic_workspace"
    fabric.ensure_session_initialized.return_value.confidence = 0.2
    fabric.ensure_session_initialized.return_value.signals = []
    fabric.ensure_session_initialized.return_value.instructions_profile = "generic"
    fabric.close = AsyncMock()
    runtime = FakeRuntime()

    result = await run_onboarding(
        workspace,
        eventloom_path=workspace / ".eventloom",
        domain="demo",
        infra="start",
        fabric_factory=lambda eventloom_path: fabric,
        runtime_factory=lambda: runtime,
    )

    assert runtime.started is True
    infra_step = next(step for step in result.steps if step.name == "infra")
    assert infra_step.status == "ok"
    assert "Neo4j local runtime is available" in infra_step.message


def test_format_onboarding_result_includes_next_section(tmp_path: Path) -> None:
    """Human output should make the next manual steps obvious."""
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        steps=[
            OnboardingStep("mcp_config", "ok", "MCP config written", "mcp.json"),
            OnboardingStep("hook_status", "ok", "Latest hook event is hook.heartbeat"),
        ],
        next_steps=[
            "Add mcp.json to your MCP client config.",
            "Restart the MCP client.",
        ],
    )

    output = format_onboarding_result(result)

    assert "Next:" in output
    assert "- Add mcp.json to your MCP client config." in output
    assert "- Restart the MCP client." in output
