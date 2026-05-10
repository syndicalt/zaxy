"""Tests for first-run IDE and framework integration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.core import Context, HandoffBundle
from zaxy.integrations import (
    render_agent_integration_template,
    render_handoff_adapter,
    render_mcp_client_config,
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

    assert "async def zaxy_crewai_memory_step" in template
    assert "MemoryFabric" in template
    assert "session_id='default'" in template
    assert "await fabric.after_turn" in template


def test_agent_integration_template_rejects_unknown_framework() -> None:
    """Unsupported framework names should fail explicitly."""
    try:
        render_agent_integration_template("unknown")
    except ValueError as exc:
        assert "framework must be one of" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("unknown framework should raise ValueError")
