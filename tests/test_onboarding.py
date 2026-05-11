"""Tests for first-run onboarding orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    assert options["capture_mode"] == "deterministic"


def test_apply_onboarding_preset_expands_local_codex_defaults(tmp_path: Path) -> None:
    """local-codex should expand to deterministic Codex MCP and local capture config."""
    options = apply_onboarding_preset(
        "local-codex",
        workspace=tmp_path,
        mcp_client=None,
        mcp_output=None,
        hook_client=None,
        hook_output=None,
        local_profile_output=None,
        infra="none",
        capture_mode="deterministic",
    )

    assert options["mcp_client"] == "codex"
    assert options["mcp_output"] is None
    assert options["hook_client"] == "codex"
    assert options["hook_output"] == tmp_path / ".codex" / "zaxy-capture.json"
    assert options["local_profile_output"] == tmp_path / ".env.local"
    assert options["infra"] == "check"
    assert options["capture_mode"] == "deterministic"


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
        capture_mode="hybrid",
    )

    assert options["mcp_client"] == "cursor"
    assert options["mcp_output"] == tmp_path / "cursor.json"
    assert options["hook_client"] == "claude-code"
    assert options["infra"] == "start"
    assert options["capture_mode"] == "hybrid"


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
        f"Inspect model-facing memory bootstrap: zaxy memory bootstrap "
        f"--eventloom-path {eventloom_path} --session-id demo-default"
    ) in result.next_steps
    assert "Default capture mode: deterministic MCP lifecycle and observer hooks; no provider proxy required." in result.next_steps
    assert "Optional packet capture is disabled by default because it can consume provider quota." in result.next_steps


@pytest.mark.asyncio
async def test_run_onboarding_can_add_packet_capture_activation_steps(tmp_path: Path) -> None:
    """Packet capture onboarding should print concrete analyzer and projector commands."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
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
        capture_mode="packet",
        packet_upstream_base_url="https://api.openai.com/v1",
        packet_port=8787,
        fabric_factory=lambda eventloom_path: fabric,
    )

    assert (
        f"Start packet analyzer: zaxy packet-analyzer --eventloom-path {eventloom_path} "
        "--session-id demo-default --upstream-base-url https://api.openai.com/v1 "
        '--upstream-api-key "$OPENAI_API_KEY" --host 127.0.0.1 --port 8787'
    ) in result.next_steps
    assert (
        f"Start packet projector: zaxy packet-project --eventloom-path {eventloom_path} "
        "--session-id demo-default --watch --graph"
    ) in result.next_steps
    assert "Point OpenAI-compatible clients at http://127.0.0.1:8787/v1." in result.next_steps


@pytest.mark.asyncio
async def test_run_onboarding_renders_codex_install_command_and_local_capture_config(tmp_path: Path) -> None:
    """Codex onboarding should use CLI-assisted MCP plus safe local session capture config."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
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
        mcp_client="codex",
        zaxy_executable="/opt/zaxy/bin/zaxy",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        fabric_factory=lambda eventloom_path: fabric,
    )

    assert result.status == "ok"
    assert any(
        step.name == "mcp_config"
        and step.status == "preview"
        and step.message == "codex MCP install command rendered"
        for step in result.steps
    )
    assert any(step.startswith("Run this Codex MCP install command: codex mcp add zaxy") for step in result.next_steps)
    assert "-- /opt/zaxy/bin/zaxy serve" in "\n".join(result.next_steps)
    assert not (workspace / "zaxy-mcp.json").exists()
    assert (workspace / ".codex" / "zaxy-capture.json").is_file()
    assert not (workspace / ".codex" / "hooks.json").exists()
    assert any("Start managed deterministic Codex capture: zaxy capture start" in step for step in result.next_steps)


@pytest.mark.asyncio
@patch("zaxy.onboarding.start_codex_capture")
@patch("zaxy.onboarding.inspect_codex_capture")
async def test_run_onboarding_can_start_managed_codex_capture(
    mock_inspect_capture: MagicMock,
    mock_start_capture: MagicMock,
    tmp_path: Path,
) -> None:
    """Onboarding should optionally start and summarize deterministic capture."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "codebase"
    fabric.ensure_session_initialized.return_value.confidence = 0.7
    fabric.ensure_session_initialized.return_value.signals = []
    fabric.ensure_session_initialized.return_value.instructions_profile = "codebase"
    fabric.close = AsyncMock()
    mock_start_capture.return_value = {
        "started": True,
        "pid": 321,
        "message": "Started Codex capture watcher pid=321",
        "state_file": str(eventloom_path / "runtime" / "codex-capture.json"),
    }
    mock_inspect_capture.return_value = {
        "client": "codex",
        "configured": True,
        "running": True,
        "pids": [321],
        "latest_observation": {
            "type": "transcript.turn",
            "seq": 7,
            "thread": "demo-default",
            "source": "codex-local",
        },
    }

    result = await run_onboarding(
        workspace,
        eventloom_path=eventloom_path,
        domain="demo",
        session_id="demo-default",
        mcp_client="codex",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        capture_action="start",
        fabric_factory=lambda eventloom_path: fabric,
    )

    assert result.status == "ok"
    assert any(
        step.name == "capture_runtime"
        and step.status == "ok"
        and step.message == "Started Codex capture watcher pid=321"
        for step in result.steps
    )
    assert result.capture == {
        "configured": True,
        "running": True,
        "pids": [321],
        "latest_observation": {
            "type": "transcript.turn",
            "seq": 7,
            "thread": "demo-default",
            "source": "codex-local",
        },
        "doctor_status": "warning",
        "doctor_message": "automatic capture is incomplete: 0 of 4 high-value lanes are active",
    }
    mock_start_capture.assert_called_once_with(workspace=workspace.resolve())
    mock_inspect_capture.assert_called_once_with(workspace=workspace.resolve())


def test_format_onboarding_result_includes_capture_summary() -> None:
    """Human init output should show capture runtime and doctor health at the end of onboarding."""
    result = OnboardingResult(
        status="ok",
        workspace="/tmp/repo",
        domain="demo",
        session_id="demo-default",
        profile={
            "workspace_type": "codebase",
            "confidence": 0.9,
            "signals": [],
            "instructions_profile": "codebase",
        },
        capture={
            "configured": True,
            "running": True,
            "pids": [321],
            "latest_observation": {
                "type": "tool.call.completed",
                "seq": 9,
                "thread": "demo-default",
                "source": "codex-local",
            },
            "doctor_status": "ok",
            "doctor_message": "automatic capture is healthy: 4 of 4 high-value lanes are active",
        },
    )

    output = format_onboarding_result(result)

    assert "capture: configured, running" in output
    assert "capture pids: 321" in output
    assert "latest capture: tool.call.completed seq=9 session=demo-default source=codex-local" in output
    assert "capture health: ok - automatic capture is healthy: 4 of 4 high-value lanes are active" in output


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
