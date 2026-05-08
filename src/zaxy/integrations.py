"""First-run integration helpers for MCP clients and agent frameworks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from zaxy.core import HandoffBundle

MCPClient = Literal["claude-desktop", "cursor", "vscode"]
HandoffAdapter = Literal["generic", "langgraph", "crewai", "autogen"]


def render_mcp_client_config(
    client: MCPClient | str,
    *,
    eventloom_path: str = ".eventloom",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> dict[str, Any]:
    """Render a copyable MCP client config fragment for a first-run setup."""
    normalized = _normalize_client(client)
    server = _server_config(
        eventloom_path=eventloom_path,
        transport=transport,
        host=host,
        port=port,
    )
    if normalized == "vscode":
        return {"servers": {"zaxy": server}}
    return {"mcpServers": {"zaxy": server}}


def render_handoff_adapter(
    bundle: HandoffBundle,
    adapter: HandoffAdapter | str = "generic",
) -> dict[str, Any]:
    """Render a portable handoff payload for common agent frameworks."""
    normalized = _normalize_adapter(adapter)
    payload = _bundle_payload(bundle)
    if normalized == "generic":
        return payload
    if normalized == "langgraph":
        return {
            "messages": [{"role": "system", "content": bundle.prompt}],
            "zaxy": payload,
        }
    if normalized == "crewai":
        return {
            "memory": bundle.prompt,
            "metadata": {"zaxy": payload},
        }
    return {
        "system_message": bundle.prompt,
        "context_variables": {"zaxy": payload},
    }


def _server_config(
    *,
    eventloom_path: str,
    transport: str,
    host: str,
    port: int,
) -> dict[str, Any]:
    normalized_transport = transport.casefold()
    if normalized_transport == "stdio":
        return {
            "command": "zaxy",
            "args": ["serve", "--eventloom-path", eventloom_path],
            "startup_timeout_sec": 90,
            "env": {
                "EVENTLOOM_PATH": eventloom_path,
                "LOG_LEVEL": "ERROR",
                "MCP_ADMIN_TOKEN_FILE": "",
                "MCP_REMOTE_AUTH_TOKEN_FILE": "",
                "NEO4J_CA_CERT": "",
                "NEO4J_AUTO_START": "true",
                "NEO4J_PASSWORD_FILE": "",
                "NEO4J_URI": "bolt://localhost:7687",
                "OPENAI_API_KEY_FILE": "",
                "PATHLIGHT_ACCESS_TOKEN_FILE": "",
                "ZAXY_ENV": "development",
            },
        }
    if normalized_transport == "sse":
        return {
            "url": f"http://{host}:{port}/sse",
            "headers": {"x-zaxy-session-id": "default"},
        }
    raise ValueError("transport must be 'stdio' or 'sse'")


def _bundle_payload(bundle: HandoffBundle) -> dict[str, Any]:
    return {
        "session_id": bundle.session_id,
        "summary": bundle.summary,
        "prompt": bundle.prompt,
        "contexts": [asdict(context) for context in bundle.contexts],
        "replay_event_count": bundle.replay_event_count,
        "integrity_ok": bundle.integrity_ok,
    }


def _normalize_client(client: str) -> MCPClient:
    normalized = client.casefold().replace("_", "-")
    if normalized in {"claude", "claude-desktop"}:
        return "claude-desktop"
    if normalized == "cursor":
        return "cursor"
    if normalized in {"vscode", "vs-code", "visual-studio-code"}:
        return "vscode"
    raise ValueError("client must be one of: claude-desktop, cursor, vscode")


def _normalize_adapter(adapter: str) -> HandoffAdapter:
    normalized = adapter.casefold().replace("_", "-")
    if normalized in {"generic", "langgraph", "crewai", "autogen"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("adapter must be one of: generic, langgraph, crewai, autogen")
