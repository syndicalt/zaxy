"""Tests for first-run onboarding orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.config import Settings
from zaxy.event import EventLog
from zaxy.onboarding import (
    OnboardingResult,
    OnboardingStep,
    _build_runtime,
    apply_onboarding_preset,
    format_onboarding_result,
    onboarding_result_payload,
    run_onboarding,
    write_agent_activation_instructions,
)


class FakeRuntime:
    """Runtime double for onboarding infra actions."""

    def __init__(self, *, status: str = "ok", message: str = "Neo4j reachable") -> None:
        self.status = status
        self.message = message
        self.checked = False
        self.started = False
        self.display_name = "Neo4j"

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


def test_apply_onboarding_preset_expands_local_embedded_codex_defaults(tmp_path: Path) -> None:
    """local-embedded-codex should be the one-command no-sidecar local path."""
    options = apply_onboarding_preset(
        "local-embedded-codex",
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
    assert options["hook_client"] == "codex"
    assert options["hook_output"] == tmp_path / ".codex" / "zaxy-capture.json"
    assert options["local_profile_output"] == tmp_path / ".env.local"
    assert options["infra"] == "check"
    assert options["projection_backend"] == "embedded"
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
        "agent_instructions",
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
    agents = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert "<!-- zaxy-memory-activation:start -->" in agents
    assert f"zaxy activate codex --eventloom-path {eventloom_path}" in agents
    assert "--session-id demo-default" in agents
    assert f"--workspace-root {workspace}" in agents
    assert f"zaxy memory checkout \"<task>\" --eventloom-path {eventloom_path}" in agents
    fabric.ensure_session_initialized.assert_awaited_once_with(workspace, session_id="demo-default")
    fabric.close.assert_awaited_once()
    events = EventLog(eventloom_path / "demo-default.jsonl").read_all()
    assert events[-1].type == "hook.heartbeat"
    assert events[-1].payload["source"] == "zaxy-init"
    assert result.next_steps[0] == f"Add {mcp_output} to your claude-desktop MCP client config."
    assert result.next_steps[1] == "Restart the MCP client so it loads the Zaxy server config."
    assert f"Data lives in {eventloom_path}; each session is an append-only JSONL log." in result.next_steps
    assert f"Run zaxy hook-status --eventloom-path {eventloom_path}" in result.next_steps
    assert (
        f"Smoke test recent memory: zaxy memory log --eventloom-path {eventloom_path} "
        "--session-id demo-default --limit 5"
    ) in result.next_steps
    assert (
        f"Inspect model-facing memory bootstrap: zaxy memory bootstrap "
        f"--eventloom-path {eventloom_path} --session-id demo-default"
    ) in result.next_steps
    assert "Default capture mode: deterministic MCP lifecycle and observer hooks; no provider proxy required." in result.next_steps
    assert "Optional packet capture is disabled by default because it can consume provider quota." in result.next_steps


@pytest.mark.asyncio
async def test_run_onboarding_repairs_stale_embedded_mcp_runtime(tmp_path: Path) -> None:
    """Init should clean stale embedded MCP owner metadata before workers start."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    runtime = eventloom_path / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "zaxy-embedded-owner.json").write_text(
        '{"pid": 999999999, "socket_path": "/tmp/missing-zaxy.sock"}',
        encoding="utf-8",
    )
    (runtime / "zaxy-embedded-owner.sock").write_text("", encoding="utf-8")
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "codebase"
    fabric.ensure_session_initialized.return_value.confidence = 0.7
    fabric.ensure_session_initialized.return_value.signals = []
    fabric.ensure_session_initialized.return_value.instructions_profile = "codebase"
    fabric.close = AsyncMock()

    result = await run_onboarding(
        workspace,
        eventloom_path=eventloom_path,
        projection_backend="embedded",
        fabric_factory=lambda eventloom_path: fabric,
    )

    runtime_step = next(step for step in result.steps if step.name == "embedded_mcp_runtime")
    assert runtime_step.status == "ok"
    assert "stale embedded MCP runtime metadata was removed" in runtime_step.message
    assert not (runtime / "zaxy-embedded-owner.json").exists()
    assert not (runtime / "zaxy-embedded-owner.sock").exists()


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
    agents = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert "Zaxy Memory Activation" in agents
    assert "zaxy hook-event resume" in agents
    activate_step = next(step for step in result.next_steps if step.startswith("Start or restart Codex"))
    assert f"--eventloom-path {eventloom_path}" in activate_step
    assert "--session-id demo-default" in activate_step
    assert f"--workspace-root {workspace}" in activate_step
    assert "--launch" in activate_step
    assert "Restart Codex so it loads the Zaxy MCP server." not in result.next_steps
    assert (
        f"After Codex resume or update, emit the resume boundary: zaxy hook-event resume "
        f"--eventloom-path {eventloom_path} --session-id demo-default --source codex --summary \"<task>\""
    ) in result.next_steps
    assert (
        f"If Zaxy MCP tools are absent, use the CLI checkout fallback before substantial work: "
        f"zaxy memory checkout \"<task>\" --eventloom-path {eventloom_path} --session-id demo-default"
    ) in result.next_steps
    assert not any("Start managed deterministic Codex capture: zaxy capture start" in step for step in result.next_steps)
    assert "Managed deterministic Codex capture starts through the activation launcher." in result.next_steps


@pytest.mark.asyncio
async def test_run_onboarding_can_install_codex_user_mcp_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex onboarding should support an explicit no-copy-paste install path."""
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
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
        codex_mcp_install="user",
        zaxy_executable="/opt/zaxy/bin/zaxy",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        fabric_factory=lambda eventloom_path: fabric,
    )

    config = codex_home / "config.toml"
    text = config.read_text(encoding="utf-8")
    assert result.status == "ok"
    assert '[mcp_servers.zaxy]' in text
    assert 'command = "/opt/zaxy/bin/zaxy"' in text
    assert 'args = ["serve"]' in text
    assert not any(step.startswith("Run this Codex MCP install command:") for step in result.next_steps)
    assert f"Codex MCP config installed at {config}" in result.next_steps
    assert any(
        step.name == "mcp_config"
        and step.status == "ok"
        and step.message == "codex MCP config installed"
        and step.path == str(config)
        for step in result.steps
    )


@pytest.mark.asyncio
async def test_run_onboarding_requires_trust_for_codex_project_mcp_config(tmp_path: Path) -> None:
    """Project-scoped Codex MCP install should keep the existing trust gate."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    fabric = MagicMock()

    with pytest.raises(PermissionError, match="trusted project"):
        await run_onboarding(
            workspace,
            eventloom_path=workspace / ".eventloom",
            domain="demo",
            session_id="demo-default",
            mcp_client="codex",
            codex_mcp_install="project",
            fabric_factory=lambda eventloom_path: fabric,
        )


@pytest.mark.asyncio
async def test_run_onboarding_uses_session_id_for_activation_remediation(tmp_path: Path) -> None:
    """Fresh init should not tell users to run checkout against the global default session."""
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

    result = await run_onboarding(
        workspace,
        eventloom_path=eventloom_path,
        domain="demo",
        session_id="demo-default",
        mcp_client="codex",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        fabric_factory=lambda eventloom_path: fabric,
    )

    memory_activation = result.hook_status["memory_activation"]
    command = memory_activation["remediations"][0]["command"]
    assert "--session-id demo-default" in command
    assert "--session-id default" not in command


@pytest.mark.asyncio
async def test_run_onboarding_keeps_stale_global_codex_config_as_readiness_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale global Codex config should not make first-run workspace init look failed."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "\n".join(
            [
                "[mcp_servers.zaxy]",
                'command = "zaxy"',
                'args = ["serve", "--eventloom-path", "/old/repo/.eventloom"]',
                "",
                "[mcp_servers.zaxy.env]",
                'EVENTLOOM_PATH = "/old/repo/.eventloom"',
                'EVENTLOOM_THREAD = "old-default"',
                'ZAXY_DOMAIN = "old"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
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

    result = await run_onboarding(
        workspace,
        eventloom_path=eventloom_path,
        domain="demo",
        session_id="demo-default",
        mcp_client="codex",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        fabric_factory=lambda eventloom_path: fabric,
    )

    codex_scope = next(check for check in result.doctor["checks"] if check["name"] == "codex_mcp_scope")
    assert codex_scope["status"] == "warning"
    assert result.doctor["status"] == "warning"
    assert result.status == "ok"
    payload = onboarding_result_payload(result)
    assert not any("codex_mcp_scope" in reason for reason in payload["readiness"]["reasons"])
    assert payload["readiness"]["blocking_diagnostics"] == []
    non_blocking_names = {check["name"] for check in payload["readiness"]["non_blocking_diagnostics"]}
    assert "codex_mcp_scope" in non_blocking_names
    assert codex_scope in payload["readiness"]["non_blocking_diagnostics"]


def test_write_agent_activation_instructions_replaces_only_managed_block(tmp_path: Path) -> None:
    """Agent instructions should stay model-visible without rewriting unrelated AGENTS content."""
    agents = tmp_path / "AGENTS.md"
    agents.write_text(
        "\n".join(
            [
                "# Existing Rules",
                "",
                "Keep this line.",
                "",
                "<!-- zaxy-memory-activation:start -->",
                "old generated block",
                "<!-- zaxy-memory-activation:end -->",
                "",
                "Keep this tail.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    written = write_agent_activation_instructions(
        tmp_path,
        eventloom_path=tmp_path / ".eventloom",
        session_id="demo-default",
    )

    text = written.read_text(encoding="utf-8")
    assert text.count("<!-- zaxy-memory-activation:start -->") == 1
    assert "old generated block" not in text
    assert "Keep this line." in text
    assert "Keep this tail." in text
    assert f"zaxy activate codex --eventloom-path {tmp_path / '.eventloom'}" in text
    assert "--session-id demo-default" in text
    assert f"--workspace-root {tmp_path}" in text
    assert f"zaxy memory checkout \"<task>\" --eventloom-path {tmp_path / '.eventloom'}" in text


@pytest.mark.asyncio
async def test_run_onboarding_embedded_codex_next_steps_avoid_neo4j_graph_hint(tmp_path: Path) -> None:
    """Embedded Codex onboarding should not steer users back to Neo4j live projection."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "codebase"
    fabric.ensure_session_initialized.return_value.confidence = 0.7
    fabric.ensure_session_initialized.return_value.signals = ["pyproject.toml"]
    fabric.ensure_session_initialized.return_value.instructions_profile = "codebase"
    fabric.close = AsyncMock()

    result = await run_onboarding(
        workspace,
        eventloom_path=workspace / ".eventloom",
        domain="demo",
        session_id="demo-default",
        mcp_client="codex",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        projection_backend="embedded",
        infra="check",
        fabric_factory=lambda eventloom_path: fabric,
    )

    next_steps = "\n".join(result.next_steps)
    assert "Start managed deterministic Codex capture: zaxy capture start" not in next_steps
    assert "Managed deterministic Codex capture starts through the activation launcher." in next_steps
    assert "Add --graph" not in next_steps
    assert "Neo4j" not in next_steps
    assert "NEO4J_" not in next_steps
    assert "embedded projection" in next_steps


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
        "doctor_message": "Codex capture is configured, but the managed watcher is not running",
    }
    mock_start_capture.assert_called_once_with(workspace=workspace.resolve())
    mock_inspect_capture.assert_called_once_with(workspace=workspace.resolve())
    next_steps = "\n".join(result.next_steps)
    assert "Start managed deterministic Codex capture: zaxy capture start" not in next_steps


@pytest.mark.asyncio
@patch("zaxy.onboarding.inspect_codex_capture")
@patch("zaxy.onboarding.inspect_hook_status")
@patch("zaxy.onboarding.run_doctor")
async def test_run_onboarding_summarizes_actionable_doctor_warnings(
    mock_run_doctor: MagicMock,
    mock_inspect_hook_status: MagicMock,
    mock_inspect_capture: MagicMock,
    tmp_path: Path,
) -> None:
    """Compact setup issues should explain the failing doctor check, not just say doctor ran."""
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
    mock_run_doctor.return_value = {
        "status": "warning",
        "checks": [
            {"name": "eventloom", "status": "ok", "message": "eventloom is writable"},
            {
                "name": "agent_instructions",
                "status": "warning",
                "message": "AGENTS.md is missing Zaxy Memory Activation instructions",
                "action": "Run zaxy init without --no-agent-instructions.",
            },
            {
                "name": "capture_health",
                "status": "warning",
                "message": "Codex capture is configured, but the managed watcher is not running",
            },
        ],
    }
    mock_inspect_hook_status.return_value = {
        "status": "ok",
        "message": "Latest hook event is hook.heartbeat",
        "clients": {},
        "latest_event": {"type": "hook.heartbeat"},
    }
    mock_inspect_capture.return_value = {
        "configured": True,
        "running": False,
        "pids": [],
        "latest_observation": None,
    }

    result = await run_onboarding(
        workspace,
        eventloom_path=eventloom_path,
        domain="demo",
        fabric_factory=lambda eventloom_path: fabric,
    )

    doctor_step = next(step for step in result.steps if step.name == "doctor")
    assert doctor_step.status == "warning"
    assert doctor_step.message == (
        "agent_instructions warning: AGENTS.md is missing Zaxy Memory Activation instructions "
        "(action: Run zaxy init without --no-agent-instructions.)"
    )
    assert "capture_health" not in doctor_step.message


@pytest.mark.asyncio
@patch("zaxy.onboarding.inspect_codex_capture")
@patch("zaxy.onboarding.inspect_hook_status")
@patch("zaxy.onboarding.run_doctor")
async def test_run_onboarding_treats_disabled_agent_instructions_as_intentional(
    mock_run_doctor: MagicMock,
    mock_inspect_hook_status: MagicMock,
    mock_inspect_capture: MagicMock,
    tmp_path: Path,
) -> None:
    """--no-agent-instructions should not make intentional onboarding look degraded."""
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
    mock_run_doctor.return_value = {
        "status": "warning",
        "checks": [
            {"name": "eventloom", "status": "ok", "message": "eventloom is writable"},
            {
                "name": "agent_instructions",
                "status": "warning",
                "message": "No AGENTS.md activation instructions found",
                "action": f"Run zaxy init {workspace} to install the marker-managed Zaxy Memory Activation block.",
            },
            {
                "name": "capture_health",
                "status": "warning",
                "message": "Codex capture is configured, but the managed watcher is not running",
            },
        ],
    }
    mock_inspect_hook_status.return_value = {
        "status": "ok",
        "message": "Latest hook event is hook.heartbeat",
        "clients": {},
        "latest_event": {"type": "hook.heartbeat"},
    }
    mock_inspect_capture.return_value = {
        "configured": True,
        "running": False,
        "pids": [],
        "latest_observation": None,
    }

    result = await run_onboarding(
        workspace,
        eventloom_path=eventloom_path,
        domain="demo",
        agent_instructions=False,
        fabric_factory=lambda eventloom_path: fabric,
    )

    doctor_step = next(step for step in result.steps if step.name == "doctor")
    assert doctor_step.status == "ok"
    assert doctor_step.message == "Doctor checks completed"


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


def test_format_onboarding_result_softens_expected_capture_not_running_warning() -> None:
    """Default init should not make configured-but-not-started capture look like a failure."""
    result = OnboardingResult(
        status="ok",
        workspace="/tmp/repo",
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.9},
        capture={
            "configured": True,
            "running": False,
            "pids": [],
            "latest_observation": None,
            "doctor_status": "warning",
            "doctor_message": "Codex capture is configured, but the managed watcher is not running",
        },
    )

    output = format_onboarding_result(result)

    assert "capture: configured, not running" in output
    assert "capture next: start Codex through the activation launcher when you want live local capture" in output
    assert "capture health: warning" not in output


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
async def test_run_onboarding_can_rerun_with_same_generated_local_codex_outputs(tmp_path: Path) -> None:
    """Repeated local Codex init should be safe when generated outputs are unchanged."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    eventloom = workspace / ".eventloom"

    first = await run_onboarding(
        workspace,
        eventloom_path=eventloom,
        domain="demo",
        mcp_client="codex",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        local_profile_output=workspace / ".env.local",
        projection_backend="embedded",
        codex_mcp_install="user",
        codex_home=tmp_path / "codex-home",
        agent_instructions=False,
    )
    second = await run_onboarding(
        workspace,
        eventloom_path=eventloom,
        domain="demo",
        mcp_client="codex",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        local_profile_output=workspace / ".env.local",
        projection_backend="embedded",
        codex_mcp_install="user",
        codex_home=tmp_path / "codex-home",
        agent_instructions=False,
    )

    assert first.status in {"ok", "warning"}
    assert second.status in {"ok", "warning"}
    assert (workspace / ".env.local").is_file()
    assert (workspace / ".codex" / "zaxy-capture.json").is_file()


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


@pytest.mark.asyncio
async def test_run_onboarding_can_check_pggraph_infra(tmp_path: Path) -> None:
    """pgGraph onboarding should render backend-specific bootstrap guidance."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "generic_workspace"
    fabric.ensure_session_initialized.return_value.confidence = 0.2
    fabric.ensure_session_initialized.return_value.signals = []
    fabric.ensure_session_initialized.return_value.instructions_profile = "generic"
    fabric.close = AsyncMock()
    runtime = FakeRuntime(status="warning", message="pgGraph is not reachable; Docker is available")

    result = await run_onboarding(
        workspace,
        eventloom_path=workspace / ".eventloom",
        domain="demo",
        infra="check",
        projection_backend="pggraph",
        pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        fabric_factory=lambda eventloom_path: fabric,
        runtime_factory=lambda: runtime,
    )

    infra_step = next(step for step in result.steps if step.name == "infra")
    assert infra_step.status == "warning"
    assert "pgGraph is not reachable" in infra_step.message
    assert any("--projection-backend pggraph --infra start" in step for step in result.next_steps)


def test_build_runtime_uses_pggraph_bootstrapper_for_pggraph_backend() -> None:
    """Runtime factory should follow the selected projection backend, not Neo4j unconditionally."""
    settings = Settings(
        _env_file=None,
        projection_backend="pggraph",
        pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo="/opt/pggraph",
    )

    runtime = _build_runtime(settings)

    assert runtime.display_name == "pgGraph"


def test_build_runtime_uses_embedded_projection_runtime_for_embedded_backend(tmp_path: Path) -> None:
    """Embedded onboarding should not ask users to start sidecar graph services."""
    graph_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    settings = Settings(
        _env_file=None,
        projection_backend="embedded",
        embedded_graph_path=str(graph_path),
    )

    runtime = _build_runtime(settings)
    check = runtime.check()

    assert runtime.display_name == "embedded graph"
    assert check.status == "ok"
    assert "will be created lazily" in check.message
    runtime.ensure_available()
    assert graph_path.parent.is_dir()


@pytest.mark.asyncio
async def test_run_onboarding_checks_embedded_infra_without_sidecar_next_step(tmp_path: Path) -> None:
    """Embedded infra check should report local projection posture and avoid sidecar start guidance."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    local_profile = workspace / ".env.local"
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "generic_workspace"
    fabric.ensure_session_initialized.return_value.confidence = 0.2
    fabric.ensure_session_initialized.return_value.signals = []
    fabric.ensure_session_initialized.return_value.instructions_profile = "generic"
    fabric.close = AsyncMock()

    result = await run_onboarding(
        workspace,
        eventloom_path=workspace / ".eventloom",
        domain="demo",
        infra="check",
        projection_backend="embedded",
        local_profile_output=local_profile,
        fabric_factory=lambda eventloom_path: fabric,
    )

    infra_step = next(step for step in result.steps if step.name == "infra")
    assert infra_step.status == "ok"
    assert "Embedded graph projection" in infra_step.message
    assert not any("--infra start" in step for step in result.next_steps)
    profile_text = local_profile.read_text(encoding="utf-8")
    assert "PROJECTION_BACKEND=embedded" in profile_text
    assert "NEO4J_AUTO_START=false" in profile_text
    doctor_checks = {check["name"]: check for check in result.doctor["checks"]}
    assert "neo4j" not in doctor_checks
    assert doctor_checks["embedded_graph"]["status"] == "ok"
    assert str(workspace / ".eventloom" / "projections") in doctor_checks["embedded_graph"]["message"]


def test_format_onboarding_result_compacts_successful_setup_by_default(tmp_path: Path) -> None:
    """Default human output should lead with actions instead of every successful setup row."""
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

    assert output.startswith("OK  Zaxy init complete: ok")
    assert "Readiness: needs action (2 required actions)" in output
    assert "Setup: 2 ok" in output
    assert "[OK] mcp_config - MCP config written (mcp.json)" not in output
    assert "Required next actions:" in output
    assert "1. Add mcp.json to your MCP client config." in output
    assert "2. Restart the MCP client." in output
    assert "Next:" not in output


def test_format_onboarding_result_marks_compact_output_ready_without_required_actions(tmp_path: Path) -> None:
    """Compact output should not look incomplete when setup passed and no required actions remain."""
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
            "Data lives in /tmp/repo/.eventloom; each session is an append-only JSONL log.",
            "Run zaxy hook-status --eventloom-path /tmp/repo/.eventloom",
        ],
    )

    output = format_onboarding_result(result)

    assert "Readiness: ready" in output
    assert "Required next actions:" not in output
    assert "More: run zaxy init --verbose to show checks, fallbacks, later commands, and notes." in output


def test_format_onboarding_result_verbose_includes_full_setup_rows(tmp_path: Path) -> None:
    """Verbose human output should retain full setup diagnostics for support."""
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
        next_steps=["Restart the MCP client."],
    )

    output = format_onboarding_result(result, verbose=True)

    assert "Setup:" in output
    assert "[OK] mcp_config - MCP config written (mcp.json)" in output
    assert "[OK] hook_status - Latest hook event is hook.heartbeat" in output


def test_format_onboarding_result_compact_keeps_setup_issues_visible(tmp_path: Path) -> None:
    """Compact output should hide noise, not hide onboarding warnings or errors."""
    result = OnboardingResult(
        status="warning",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        steps=[
            OnboardingStep("eventloom", "ok", "Eventloom directory is ready"),
            OnboardingStep("doctor", "warning", "Doctor checks completed"),
        ],
    )

    output = format_onboarding_result(result)

    assert "Setup: 1 ok, 1 warning" in output
    assert "Setup issues:" in output
    assert "[WARN] doctor - Doctor checks completed" in output
    assert "[OK] eventloom" not in output


def test_format_onboarding_result_splits_codex_actions_checks_and_notes(tmp_path: Path) -> None:
    """Bare Codex onboarding should keep compact output focused on required actions."""
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        steps=[
            OnboardingStep("mcp_config", "preview", "codex MCP install command rendered"),
            OnboardingStep("hook_status", "ok", "Latest hook event is hook.heartbeat"),
        ],
        next_steps=[
            "Run this Codex MCP install command: codex mcp add zaxy -- zaxy serve",
            "Start or restart Codex through the activation launcher: zaxy activate codex --session-id demo-default --current-task \"<task>\" --launch",
            "After Codex resume or update, emit the resume boundary: zaxy hook-event resume --eventloom-path .eventloom --session-id demo-default --source codex --summary \"<task>\"",
            "If Zaxy MCP tools are absent, use the CLI checkout fallback before substantial work: zaxy memory checkout \"<task>\" --eventloom-path .eventloom --session-id demo-default",
            "Managed deterministic Codex capture starts through the activation launcher.",
            "Data lives in /tmp/repo/.eventloom; each session is an append-only JSONL log.",
            "Run zaxy hook-status --eventloom-path /tmp/repo/.eventloom",
            "Smoke test recent memory: zaxy memory log --eventloom-path /tmp/repo/.eventloom --session-id demo-default --limit 5",
            "Default capture mode: deterministic MCP lifecycle and observer hooks; no provider proxy required.",
        ],
    )

    output = format_onboarding_result(result)

    assert "Setup: 1 ok, 1 preview" in output
    assert "Setup issues:" not in output
    assert "[preview] mcp_config - codex MCP install command rendered" not in output
    assert "Required next actions:" in output
    assert "1. Install Codex MCP:" in output
    assert "   codex mcp add zaxy -- zaxy serve" in output
    assert "2. Start or restart Codex through the activation launcher:" in output
    assert "   zaxy activate codex --session-id demo-default --current-task \"<task>\" --launch" in output
    assert "   Tip: replace <task> with the work you are starting." in output
    assert "\x1b[" not in output
    assert "3. Start managed deterministic Codex capture:" not in output
    assert "zaxy capture start --workspace /tmp/repo" not in output
    assert "Restart Codex so it loads the Zaxy MCP server." not in output
    assert "4. After Codex resume" not in output
    assert "Useful checks:" not in output
    assert "Fallbacks:" not in output
    assert "Later:" not in output
    assert "Notes:" not in output
    assert "More: run zaxy init --verbose to show checks, fallbacks, later commands, and notes." in output


def test_format_onboarding_result_explains_long_codex_install_command(tmp_path: Path) -> None:
    """Long resolved executable paths should be explained without changing the copyable command."""
    command = "codex mcp add zaxy --env LOG_LEVEL=ERROR -- /opt/zaxy/bin/zaxy serve"
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        next_steps=[
            f"Run this Codex MCP install command: {command}",
        ],
    )

    output = format_onboarding_result(result)

    assert f"   {command}" in output
    assert "   Tip: this uses the resolved zaxy executable for MCP client reliability." in output
    assert "   Tip: use zaxy init --codex-mcp-install user when Codex has no conflicting zaxy entry." in output
    assert "   Tip: add --force only when you intentionally want to replace an existing zaxy MCP entry." in output


@pytest.mark.asyncio
async def test_run_onboarding_codex_activation_command_is_workspace_stable(tmp_path: Path) -> None:
    """Printed activation commands should work even when copied outside the initialized repo."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    fabric = MagicMock()
    fabric.ensure_session_initialized = AsyncMock()
    fabric.ensure_session_initialized.return_value.workspace_type = "generic_workspace"
    fabric.ensure_session_initialized.return_value.confidence = 0.2
    fabric.ensure_session_initialized.return_value.signals = []
    fabric.ensure_session_initialized.return_value.instructions_profile = "generic"
    fabric.close = AsyncMock()

    result = await run_onboarding(
        workspace,
        eventloom_path=eventloom_path,
        domain="demo",
        mcp_client="codex",
        codex_mcp_install="user",
        codex_home=tmp_path / "codex-home",
        hook_client="codex",
        hook_output=workspace / ".codex" / "zaxy-capture.json",
        local_profile_output=workspace / ".env.local",
        agent_instructions=False,
        fabric_factory=lambda eventloom_path: fabric,
        runtime_factory=FakeRuntime,
    )

    activate_step = next(step for step in result.next_steps if step.startswith("Start or restart Codex"))

    assert f"--eventloom-path {eventloom_path}" in activate_step
    assert f"--workspace-root {workspace}" in activate_step


@pytest.mark.asyncio
async def test_run_onboarding_codex_conflict_requires_review_instead_of_replace_command(
    tmp_path: Path,
) -> None:
    """Conflicting Codex zaxy entries should not be routed to silent replacement commands."""
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
    conflict_path = tmp_path / "codex-home" / "config.toml"
    conflict_path.parent.mkdir()
    conflict_path.write_text(
        '[mcp_servers.zaxy]\ncommand = "/custom/zaxy"\nargs = ["serve"]\n',
        encoding="utf-8",
    )

    result = await run_onboarding(
        workspace,
        mcp_client="codex",
        codex_mcp_install="command",
        codex_mcp_conflict_path=conflict_path,
        fabric_factory=lambda eventloom_path: fabric,
        runtime_factory=lambda: runtime,
    )

    assert result.status == "warning"
    assert any(step.name == "mcp_config" and step.status == "warning" for step in result.steps)
    assert not any(step.startswith("Run this Codex MCP install command:") for step in result.next_steps)
    assert any(str(conflict_path) in step for step in result.next_steps)
    payload = onboarding_result_payload(result)
    assert payload["readiness"]["actions"] == [
        f"Review existing Codex MCP config before replacing zaxy at {conflict_path}.",
    ]
    assert payload["readiness"]["action_items"] == [
        {
            "label": "Review existing Codex MCP config before replacing zaxy:",
            "command": None,
            "source": f"Review existing Codex MCP config before replacing zaxy at {conflict_path}.",
            "hints": ["Tip: rerun with --codex-mcp-install user --force after reviewing it."],
        }
    ]
    output = format_onboarding_result(result)
    assert "Install Codex MCP:" not in output
    assert "Review existing Codex MCP config before replacing zaxy" in output
    assert str(conflict_path) in output
    assert f"1. Review existing Codex MCP config before replacing zaxy at {conflict_path}." in output
    assert "   Tip: rerun with --codex-mcp-install user --force after reviewing it." in output


def test_format_onboarding_result_uses_exact_verbose_command_when_provided(tmp_path: Path) -> None:
    """Compact output should be able to print the exact verbose rerun command."""
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        next_steps=[
            "Start or restart Codex through the activation launcher: zaxy activate codex --session-id demo-default --current-task \"<task>\" --launch",
            "Run zaxy hook-status --eventloom-path /tmp/repo/.eventloom",
        ],
    )

    output = format_onboarding_result(
        result,
        verbose_command="zaxy init '/tmp/repo with spaces' --domain demo --verbose",
    )

    assert "More: run zaxy init '/tmp/repo with spaces' --domain demo --verbose to show checks, fallbacks, later commands, and notes." in output


def test_format_onboarding_result_verbose_splits_codex_checks_and_notes(tmp_path: Path) -> None:
    """Verbose Codex onboarding should retain optional support diagnostics."""
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        steps=[
            OnboardingStep("mcp_config", "preview", "codex MCP install command rendered"),
            OnboardingStep("hook_status", "ok", "Latest hook event is hook.heartbeat"),
        ],
        next_steps=[
            "Run this Codex MCP install command: codex mcp add zaxy -- zaxy serve",
            "Start or restart Codex through the activation launcher: zaxy activate codex --session-id demo-default --current-task \"<task>\" --launch",
            "After Codex resume or update, emit the resume boundary: zaxy hook-event resume --eventloom-path .eventloom --session-id demo-default --source codex --summary \"<task>\"",
            "If Zaxy MCP tools are absent, use the CLI checkout fallback before substantial work: zaxy memory checkout \"<task>\" --eventloom-path .eventloom --session-id demo-default",
            "Managed deterministic Codex capture starts through the activation launcher.",
            "Data lives in /tmp/repo/.eventloom; each session is an append-only JSONL log.",
            "Run zaxy hook-status --eventloom-path /tmp/repo/.eventloom",
            "Smoke test recent memory: zaxy memory log --eventloom-path /tmp/repo/.eventloom --session-id demo-default --limit 5",
            "Default capture mode: deterministic MCP lifecycle and observer hooks; no provider proxy required.",
        ],
    )

    output = format_onboarding_result(result, verbose=True)

    assert "Required next actions:" in output
    assert "1. Install Codex MCP:" in output
    assert "   codex mcp add zaxy -- zaxy serve" in output
    assert "2. Start or restart Codex through the activation launcher:" in output
    assert "   zaxy activate codex --session-id demo-default --current-task \"<task>\" --launch" in output
    assert "3. Start managed deterministic Codex capture:" not in output
    assert "zaxy capture start --workspace /tmp/repo" not in output
    assert "Useful checks:" in output
    assert "- zaxy hook-status --eventloom-path /tmp/repo/.eventloom" in output
    assert "- zaxy memory log --eventloom-path /tmp/repo/.eventloom --session-id demo-default --limit 5" in output
    assert "Fallbacks:" in output
    assert "- zaxy memory checkout \"<task>\" --eventloom-path .eventloom --session-id demo-default" in output
    assert "Later:" in output
    assert "- After Codex resume or update, emit the resume boundary:" in output
    assert "  zaxy hook-event resume --eventloom-path .eventloom --session-id demo-default --source codex --summary \"<task>\"" in output
    assert "Notes:" in output
    assert "- Data lives in /tmp/repo/.eventloom; each session is an append-only JSONL log." in output
    assert "- Managed deterministic Codex capture starts through the activation launcher." in output
    assert "- Default capture mode: deterministic MCP lifecycle and observer hooks; no provider proxy required." in output


def test_format_onboarding_result_keeps_installed_codex_config_out_of_required_actions(tmp_path: Path) -> None:
    """Successful Codex MCP installation should not appear as another manual task."""
    config_path = tmp_path / "codex-home" / "config.toml"
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        steps=[OnboardingStep("mcp_config", "ok", "codex MCP config installed", str(config_path))],
        next_steps=[
            f"Codex MCP config installed at {config_path}",
            "Start or restart Codex through the activation launcher: zaxy activate codex --session-id demo-default --current-task \"<task>\" --launch",
        ],
    )

    output = format_onboarding_result(result)

    assert "Required next actions:" in output
    assert "1. Start or restart Codex through the activation launcher:" in output
    assert f"1. Codex MCP config installed at {config_path}" not in output
    assert "Notes:" not in output
    assert f"- Codex MCP config installed at {config_path}" not in output
    assert "More: run zaxy init --verbose" in output


def test_format_onboarding_result_keeps_running_capture_out_of_required_actions(tmp_path: Path) -> None:
    """Already-started capture should be shown as state, not as a manual task."""
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        steps=[OnboardingStep("capture_runtime", "ok", "Started Codex capture watcher pid=321")],
        next_steps=[
            "Start or restart Codex through the activation launcher: zaxy activate codex --session-id demo-default --current-task \"<task>\" --launch",
            "Managed deterministic Codex capture is already running.",
        ],
    )

    output = format_onboarding_result(result)

    assert "Required next actions:" in output
    assert "1. Start or restart Codex through the activation launcher:" in output
    assert "2. Managed deterministic Codex capture is already running." not in output
    assert "Notes:" not in output
    assert "- Managed deterministic Codex capture is already running." not in output
    assert "More: run zaxy init --verbose" in output


def test_onboarding_result_payload_exposes_structured_readiness_actions(tmp_path: Path) -> None:
    """Machine output should not require parsing human next-step strings."""
    install = "codex mcp add zaxy -- zaxy serve"
    activate = 'zaxy activate codex --session-id demo-default --current-task "<task>" --launch'
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        steps=[
            OnboardingStep("mcp_config", "preview", "codex MCP install command rendered"),
            OnboardingStep("hook_status", "ok", "Latest hook event is hook.heartbeat"),
        ],
        next_steps=[
            f"Run this Codex MCP install command: {install}",
            f"Start or restart Codex through the activation launcher: {activate}",
            "Run zaxy hook-status --eventloom-path /tmp/repo/.eventloom",
            "Managed deterministic Codex capture starts through the activation launcher.",
        ],
    )

    payload = onboarding_result_payload(result)

    assert payload["setup"]["summary"] == "Setup: 1 ok, 1 preview"
    assert payload["readiness"]["summary"] == "Readiness: needs action (2 required actions)"
    assert payload["readiness"]["required_action_count"] == 2
    assert payload["readiness"]["reason_count"] == 0
    assert payload["readiness"]["actions"] == [
        f"Run this Codex MCP install command: {install}",
        f"Start or restart Codex through the activation launcher: {activate}",
    ]
    assert payload["readiness"]["action_items"] == [
        {
            "label": "Install Codex MCP:",
            "command": install,
            "source": f"Run this Codex MCP install command: {install}",
            "hints": [],
        },
        {
            "label": "Start or restart Codex through the activation launcher:",
            "command": activate,
            "source": f"Start or restart Codex through the activation launcher: {activate}",
            "hints": ["Tip: replace <task> with the work you are starting."],
        },
    ]


def test_onboarding_result_payload_explains_path_stable_activation_command(tmp_path: Path) -> None:
    """Activation action hints should explain long explicit paths without parsing human output."""
    activate = (
        "zaxy activate codex --eventloom-path /tmp/repo/.eventloom "
        "--session-id demo-default --current-task '<task>' --workspace-root /tmp/repo --launch"
    )
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        next_steps=[
            f"Start or restart Codex through the activation launcher: {activate}",
        ],
    )

    payload = onboarding_result_payload(result)

    assert payload["readiness"]["action_items"] == [
        {
            "label": "Start or restart Codex through the activation launcher:",
            "command": activate,
            "source": f"Start or restart Codex through the activation launcher: {activate}",
            "hints": [
                "Tip: replace <task> with the work you are starting.",
                "Tip: explicit --eventloom-path and --workspace-root values keep activation tied to this repo from any shell.",
            ],
        },
    ]


def test_onboarding_result_payload_keeps_idle_activation_capture_advisory(tmp_path: Path) -> None:
    """Configured-but-idle capture should not block readiness when no manual action remains."""
    result = OnboardingResult(
        status="ok",
        workspace=str(tmp_path),
        domain="demo",
        session_id="demo-default",
        profile={"workspace_type": "codebase", "confidence": 0.7},
        steps=[OnboardingStep("mcp_config", "ok", "codex MCP config installed")],
        next_steps=[
            "Managed deterministic Codex capture starts through the activation launcher.",
            "Data lives in /tmp/repo/.eventloom; each session is an append-only JSONL log.",
        ],
        capture={
            "configured": True,
            "running": False,
            "doctor_status": "warning",
            "doctor_message": "Codex capture is configured, but the managed watcher is not running",
        },
    )

    payload = onboarding_result_payload(result)

    assert payload["readiness"]["status"] == "ready"
    assert payload["readiness"]["actions"] == []
    assert payload["readiness"]["reasons"] == []
    assert payload["readiness"]["capture"]["running"] is False
