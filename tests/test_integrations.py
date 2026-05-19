"""Tests for first-run IDE and framework integration helpers."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

from zaxy import (
    recommend_framework_integration_target as public_recommend_framework_integration_target,
)
from zaxy.core import Context, HandoffBundle
from zaxy.integrations import (
    list_framework_integration_specs,
    recommend_framework_integration_target,
    render_agent_integration_template,
    render_codex_mcp_add_command,
    render_framework_install_command,
    render_handoff_adapter,
    render_mcp_client_config,
    write_codex_mcp_config,
    write_hermes_mcp_config,
    write_project_mcp_client_config,
)


def test_renders_claude_desktop_mcp_config_without_secrets() -> None:
    """First-run MCP config should be copyable and avoid secret material."""
    config = render_mcp_client_config(
        "claude-desktop",
        eventloom_path='.eventloom',
        transport="stdio",
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
    )

    server = config["mcpServers"]["zaxy"]
    assert server["command"] == "/opt/zaxy/bin/zaxy"
    assert server["args"] == ["serve", "--eventloom-path", ".eventloom"]
    assert server["startup_timeout_sec"] == 90
    assert server["env"] == {
        "EVENTLOOM_PATH": ".eventloom",
        "EVENTLOOM_THREAD": "zaxy-default",
        "LOG_LEVEL": "ERROR",
        "MCP_ADMIN_TOKEN_FILE": "",
        "MCP_REMOTE_AUTH_TOKEN_FILE": "",
        "NEO4J_CA_CERT": "",
        "NEO4J_AUTO_START": "true",
        "NEO4J_PASSWORD_FILE": "",
        "NEO4J_URI": "bolt://localhost:7687",
        "OPENAI_API_KEY_FILE": "",
        "PATHLIGHT_ACCESS_TOKEN_FILE": "",
        "ZAXY_DOMAIN": "zaxy",
        "ZAXY_ENV": "development",
    }
    assert "testpassword" not in str(config).casefold()


def test_renders_vscode_mcp_config_with_servers_key() -> None:
    """VS Code uses a workspace mcp.json shape rather than mcpServers."""
    config = render_mcp_client_config("vscode", eventloom_path='.eventloom', zaxy_executable="/opt/zaxy/bin/zaxy")

    assert "servers" in config
    assert config["servers"]["zaxy"]["command"] == "/opt/zaxy/bin/zaxy"
    assert config["servers"]["zaxy"]["args"][0] == "serve"


def test_renders_sse_config_with_domain_session_header() -> None:
    """Remote MCP config should avoid raw default session scope."""
    config = render_mcp_client_config(
        "cursor",
        eventloom_path='.eventloom',
        transport="sse",
        domain="gallerie",
    )

    server = config["mcpServers"]["zaxy"]
    assert server["headers"]["x-zaxy-session-id"] == "gallerie-default"


def test_renders_hermes_mcp_config_as_workspace_neutral_yaml_shape() -> None:
    """Hermes uses global YAML, so Zaxy should not bake repo-local session state into it."""
    config = render_mcp_client_config(
        "hermes",
        eventloom_path=".eventloom",
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
    )

    server = config["mcp_servers"]["zaxy"]
    assert server["command"] == "/opt/zaxy/bin/zaxy"
    assert server["args"] == ["serve"]
    assert server["timeout"] == 120
    assert server["connect_timeout"] == 60
    assert server["enabled"] is True
    assert server["tools"]["resources"] is False
    assert server["tools"]["prompts"] is False
    assert "memory_checkout" in server["tools"]["include"]
    assert server["env"]["NEO4J_URI"] == "bolt://localhost:7687"
    assert "EVENTLOOM_PATH" not in server["env"]
    assert "EVENTLOOM_THREAD" not in server["env"]
    assert "ZAXY_DOMAIN" not in server["env"]


def test_writes_hermes_mcp_config_without_removing_existing_settings(tmp_path: Path) -> None:
    """Hermes install should merge into config.yaml and preserve unrelated YAML data."""
    target = tmp_path / "config.yaml"
    target.write_text(
        "model:\n  default: anthropic/claude-opus-4.6\n"
        "mcp_servers:\n"
        "  filesystem:\n"
        "    command: npx\n"
        "    args: ['-y', '@modelcontextprotocol/server-filesystem', '/tmp']\n",
        encoding="utf-8",
    )

    written = write_hermes_mcp_config(
        config_path=target,
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
    )

    config = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert written == target
    assert config["model"]["default"] == "anthropic/claude-opus-4.6"
    assert config["mcp_servers"]["filesystem"]["command"] == "npx"
    assert config["mcp_servers"]["zaxy"]["command"] == "/opt/zaxy/bin/zaxy"
    assert config["mcp_servers"]["zaxy"]["args"] == ["serve"]
    assert "EVENTLOOM_PATH" not in config["mcp_servers"]["zaxy"]["env"]


def test_hermes_mcp_config_refuses_existing_zaxy_without_force(tmp_path: Path) -> None:
    """Hermes install should not replace an existing zaxy server unless forced."""
    target = tmp_path / "config.yaml"
    target.write_text("mcp_servers:\n  zaxy:\n    command: old-zaxy\n", encoding="utf-8")

    try:
        write_hermes_mcp_config(config_path=target, zaxy_executable="/opt/zaxy/bin/zaxy")
    except FileExistsError as exc:
        assert "already contains a zaxy MCP server" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("existing Hermes zaxy server should require force")
    assert "old-zaxy" in target.read_text(encoding="utf-8")


def test_writes_cursor_project_mcp_config(tmp_path: Path) -> None:
    """Cursor project install should create .cursor/mcp.json with mcpServers."""
    written = write_project_mcp_client_config(
        "cursor",
        workspace=tmp_path,
        eventloom_path=".eventloom",
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
    )

    assert written == tmp_path / ".cursor" / "mcp.json"
    config = written.read_text(encoding="utf-8")
    assert '"mcpServers"' in config
    assert '"command": "/opt/zaxy/bin/zaxy"' in config


def test_merges_vscode_project_mcp_config_without_removing_existing_servers(tmp_path: Path) -> None:
    """VS Code project install should preserve unrelated servers."""
    target = tmp_path / ".vscode" / "mcp.json"
    target.parent.mkdir()
    target.write_text(
        '{"servers": {"playwright": {"command": "npx", "args": ["playwright"]}}}\n',
        encoding="utf-8",
    )

    write_project_mcp_client_config(
        "vscode",
        workspace=tmp_path,
        eventloom_path=".eventloom",
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
    )

    config = json.loads(target.read_text(encoding="utf-8"))
    assert config["servers"]["playwright"]["command"] == "npx"
    assert config["servers"]["zaxy"]["command"] == "/opt/zaxy/bin/zaxy"


def test_project_mcp_install_refuses_existing_zaxy_without_force(tmp_path: Path) -> None:
    """Install should not replace an existing zaxy server unless forced."""
    target = tmp_path / ".mcp.json"
    target.write_text('{"mcpServers": {"zaxy": {"command": "old-zaxy"}}}\n', encoding="utf-8")

    try:
        write_project_mcp_client_config(
            "claude-code",
            workspace=tmp_path,
            eventloom_path=".eventloom",
            domain="zaxy",
            zaxy_executable="/opt/zaxy/bin/zaxy",
        )
    except FileExistsError as exc:
        assert "already contains a zaxy MCP server" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("existing zaxy server should require force")
    assert "old-zaxy" in target.read_text(encoding="utf-8")


def test_project_mcp_install_rejects_malformed_json(tmp_path: Path) -> None:
    """Install should fail clearly instead of repairing invalid client config."""
    target = tmp_path / ".cursor" / "mcp.json"
    target.parent.mkdir()
    target.write_text("{not-json", encoding="utf-8")

    try:
        write_project_mcp_client_config("cursor", workspace=tmp_path)
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("malformed JSON should raise ValueError")
    assert target.read_text(encoding="utf-8") == "{not-json"


def test_project_mcp_install_force_replaces_existing_zaxy(tmp_path: Path) -> None:
    """Force should replace only the zaxy entry while preserving other servers."""
    target = tmp_path / ".mcp.json"
    target.write_text(
        '{"mcpServers": {"zaxy": {"command": "old-zaxy"}, "other": {"command": "other"}}}\n',
        encoding="utf-8",
    )

    write_project_mcp_client_config(
        "claude-code",
        workspace=tmp_path,
        eventloom_path=".eventloom",
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
        force=True,
    )

    config = json.loads(target.read_text(encoding="utf-8"))
    assert config["mcpServers"]["zaxy"]["command"] == "/opt/zaxy/bin/zaxy"
    assert config["mcpServers"]["other"]["command"] == "other"


def test_renders_codex_mcp_add_command_with_env_and_command_separator() -> None:
    """Codex install should not bake one repo's Eventloom scope into global config."""
    command = render_codex_mcp_add_command(
        eventloom_path=".eventloom",
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
    )

    assert command[:4] == ["codex", "mcp", "add", "zaxy"]
    assert "--" in command
    assert command[command.index("--") + 1 :] == [
        "/opt/zaxy/bin/zaxy",
        "serve",
    ]
    assert "--env" in command
    assert "NEO4J_URI=bolt://localhost:7687" in command
    assert "NEO4J_CA_CERT=" in command
    assert "NEO4J_PASSWORD_FILE=" in command
    assert "ZAXY_ENV=development" in command
    assert not any("EVENTLOOM_" in part or "ZAXY_DOMAIN" in part for part in command)


def test_writes_trusted_project_codex_config_without_removing_existing_servers(
    tmp_path: Path,
) -> None:
    """Codex project TOML merge should require trust and preserve unrelated servers."""
    target = tmp_path / ".codex" / "config.toml"
    target.parent.mkdir()
    target.write_text(
        '[mcp_servers.context7]\ncommand = "npx"\nargs = ["-y", "@upstash/context7-mcp"]\n',
        encoding="utf-8",
    )

    written = write_codex_mcp_config(
        scope="project",
        workspace=tmp_path,
        trusted_project=True,
        eventloom_path=".eventloom",
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
    )

    config = tomllib.loads(written.read_text(encoding="utf-8"))
    assert written == target
    assert config["mcp_servers"]["context7"]["command"] == "npx"
    assert config["mcp_servers"]["zaxy"]["command"] == "/opt/zaxy/bin/zaxy"
    assert config["mcp_servers"]["zaxy"]["args"] == ["serve"]
    assert config["mcp_servers"]["zaxy"]["env"]["NEO4J_URI"] == "bolt://localhost:7687"
    assert config["mcp_servers"]["zaxy"]["env"]["NEO4J_CA_CERT"] == ""
    assert config["mcp_servers"]["zaxy"]["env"]["NEO4J_PASSWORD_FILE"] == ""
    assert "EVENTLOOM_PATH" not in config["mcp_servers"]["zaxy"]["env"]
    assert "EVENTLOOM_THREAD" not in config["mcp_servers"]["zaxy"]["env"]
    assert "ZAXY_DOMAIN" not in config["mcp_servers"]["zaxy"]["env"]


def test_project_codex_config_requires_trusted_project_acknowledgement(tmp_path: Path) -> None:
    """Project-scoped Codex config should not be written without explicit trust acknowledgement."""
    try:
        write_codex_mcp_config(
            scope="project",
            workspace=tmp_path,
            trusted_project=False,
        )
    except PermissionError as exc:
        assert "trusted project" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("project Codex config should require trust acknowledgement")
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_codex_config_rejects_malformed_toml(tmp_path: Path) -> None:
    """Codex TOML merge should fail clearly and leave malformed config untouched."""
    target = tmp_path / ".codex" / "config.toml"
    target.parent.mkdir()
    target.write_text("[mcp_servers.zaxy\ncommand = 'broken'\n", encoding="utf-8")

    try:
        write_codex_mcp_config(
            scope="project",
            workspace=tmp_path,
            trusted_project=True,
        )
    except ValueError as exc:
        assert "invalid TOML" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("malformed TOML should raise ValueError")
    assert target.read_text(encoding="utf-8") == "[mcp_servers.zaxy\ncommand = 'broken'\n"


def test_codex_config_refuses_existing_zaxy_without_force(tmp_path: Path) -> None:
    """Codex TOML merge should not replace an existing zaxy server unless forced."""
    target = tmp_path / ".codex" / "config.toml"
    target.parent.mkdir()
    target.write_text('[mcp_servers.zaxy]\ncommand = "old-zaxy"\n', encoding="utf-8")

    try:
        write_codex_mcp_config(
            scope="project",
            workspace=tmp_path,
            trusted_project=True,
            zaxy_executable="/opt/zaxy/bin/zaxy",
        )
    except FileExistsError as exc:
        assert "already contains a zaxy MCP server" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("existing zaxy server should require force")
    assert "old-zaxy" in target.read_text(encoding="utf-8")


def test_codex_user_config_writes_to_codex_home(tmp_path: Path) -> None:
    """User-scoped Codex config should target CODEX_HOME/config.toml."""
    codex_home = tmp_path / "codex-home"

    written = write_codex_mcp_config(
        scope="user",
        workspace=tmp_path / "repo",
        codex_home=codex_home,
        eventloom_path=".eventloom",
        domain="zaxy",
        zaxy_executable="/opt/zaxy/bin/zaxy",
    )

    config = tomllib.loads(written.read_text(encoding="utf-8"))
    assert written == codex_home / "config.toml"
    assert config["mcp_servers"]["zaxy"]["command"] == "/opt/zaxy/bin/zaxy"


def test_handoff_adapter_preserves_prompt_context_and_integrity() -> None:
    """Framework adapters should wrap handoff bundles without losing evidence."""
    bundle = HandoffBundle(
        session_id="agent-1",
        summary={"goals": ["Ship adapters"]},
        prompt="Use this context.",
        contexts=[
            Context(
                content="Ship adapters",
                source="graph",
                score=1.0,
                metadata={"citation": "eventloom://default/events/1#abc"},
            )
        ],
        replay_event_count=3,
        integrity_ok=True,
    )

    payload = render_handoff_adapter(bundle, "langgraph")

    assert payload["messages"] == [{"role": "system", "content": "Use this context."}]
    assert payload["zaxy"]["session_id"] == "agent-1"
    assert payload["zaxy"]["integrity_ok"] is True
    assert payload["zaxy"]["contexts"][0]["metadata"]["citation"].startswith("eventloom://")


def test_renders_langgraph_agent_integration_template() -> None:
    """LangGraph template should wire lifecycle APIs without importing LangGraph."""
    template = render_agent_integration_template(
        "langgraph",
        session_id='zaxy-default',
        eventloom_path='.eventloom',
    )

    assert "from zaxy import MemoryFabric" in template
    assert "async def zaxy_langgraph_memory_node" in template
    assert "session_id='zaxy-default'" in template
    assert "eventloom_path='.eventloom'" in template
    assert "await fabric.after_turn" in template
    assert "await fabric.handoff_bundle" in template
    assert "import langgraph" not in template.casefold()


def test_renders_crewai_agent_integration_template() -> None:
    """CrewAI template should expose a dependency-light task helper."""
    template = render_agent_integration_template("crewai")

    assert "from zaxy.adapters.crewai import CrewAIMemoryAdapter" in template
    assert "async def zaxy_crewai_memory_step" in template
    assert "session_id='default'" in template
    assert "await adapter.before_task" in template
    assert "await adapter.after_task" in template


def test_renders_autogen_template_with_pre_reply_memory_checkout() -> None:
    """AutoGen starter should call Memory Checkout before the agent replies."""
    template = render_agent_integration_template("autogen", session_id="agent-1")

    assert "async def zaxy_autogen_context" in template
    assert "await fabric.checkout_memory" in template
    assert "session_id='agent-1'" in template
    assert '"zaxy_context": checkout.prompt' in template


def test_renders_framework_extra_install_commands() -> None:
    """Framework install guidance should map each framework to its optional extra."""
    assert render_framework_install_command("langgraph") == [
        "python",
        "-m",
        "pip",
        "install",
        "zaxy-memory[langgraph]",
    ]
    assert render_framework_install_command("autogen")[-1] == "zaxy-memory[autogen]"
    assert render_framework_install_command("frameworks")[-1] == "zaxy-memory[frameworks]"


def test_lists_framework_integration_specs() -> None:
    """Framework support metadata should have one typed registry."""
    specs = {spec.framework: spec for spec in list_framework_integration_specs()}

    assert list(specs) == ["langgraph", "crewai", "autogen"]
    assert specs["langgraph"].display_name == "LangGraph"
    assert specs["langgraph"].extra == "langgraph"
    assert specs["langgraph"].package == "langgraph"
    assert specs["langgraph"].template_function == "create_langgraph_memory_node"
    assert specs["langgraph"].maturity == "native-preview"
    assert specs["langgraph"].native_adapter == "zaxy.adapters.langgraph"
    assert specs["crewai"].template_function == "create_crewai_memory_step"
    assert specs["crewai"].maturity == "native-preview"
    assert specs["crewai"].native_adapter == "zaxy.adapters.crewai"
    assert specs["autogen"].package == "autogen-agentchat"


def test_recommends_framework_integration_hardening_target() -> None:
    """Native-preview usage should choose UX hardening over speculative adapters."""
    decision = recommend_framework_integration_target()

    assert decision.track == "model-facing-ux"
    assert decision.target == "common-native-preview-contract"
    assert decision.recommended is True
    assert decision.evidence_frameworks == ("langgraph", "crewai")
    assert decision.hold_frameworks == ("autogen",)
    assert "LangGraph" in decision.rationale
    assert "CrewAI" in decision.rationale
    assert "AutoGen" in decision.rationale
    assert decision.next_actions == (
        "stabilize shared payload keys across native-preview adapters",
        "keep AutoGen template-only until runtime hooks are validated",
        "use adapter feedback events to decide the next native package",
    )
    assert public_recommend_framework_integration_target().target == decision.target


def test_agent_integration_template_rejects_unknown_framework() -> None:
    """Unsupported framework names should fail explicitly."""
    try:
        render_agent_integration_template("unknown")
    except ValueError as exc:
        assert "framework must be one of" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("unknown framework should raise ValueError")
