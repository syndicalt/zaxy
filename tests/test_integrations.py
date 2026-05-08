"""Tests for first-run IDE and framework integration helpers."""

from __future__ import annotations

from zaxy.core import Context, HandoffBundle
from zaxy.integrations import render_handoff_adapter, render_mcp_client_config


def test_renders_claude_desktop_mcp_config_without_secrets() -> None:
    """First-run MCP config should be copyable and avoid secret material."""
    config = render_mcp_client_config(
        "claude-desktop",
        eventloom_path=".eventloom",
        transport="stdio",
    )

    server = config["mcpServers"]["zaxy"]
    assert server["command"] == "zaxy"
    assert server["args"] == ["serve", "--eventloom-path", ".eventloom"]
    assert server["env"] == {"EVENTLOOM_PATH": ".eventloom"}
    assert "token" not in str(config).casefold()
    assert "password" not in str(config).casefold()


def test_renders_vscode_mcp_config_with_servers_key() -> None:
    """VS Code uses a workspace mcp.json shape rather than mcpServers."""
    config = render_mcp_client_config("vscode", eventloom_path=".eventloom")

    assert "servers" in config
    assert config["servers"]["zaxy"]["command"] == "zaxy"
    assert config["servers"]["zaxy"]["args"][0] == "serve"


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
