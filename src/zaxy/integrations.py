"""First-run integration helpers for MCP clients and agent frameworks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from zaxy.core import HandoffBundle
from zaxy.domain import derive_domain, domain_default_session, slug_domain

MCPClient = Literal["claude-desktop", "cursor", "vscode"]
HandoffAdapter = Literal["generic", "langgraph", "crewai", "autogen"]
AgentFramework = Literal["langgraph", "crewai", "autogen"]


def render_mcp_client_config(
    client: MCPClient | str,
    *,
    eventloom_path: str = ".eventloom",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
    domain: str | None = None,
) -> dict[str, Any]:
    """Render a copyable MCP client config fragment for a first-run setup."""
    normalized = _normalize_client(client)
    resolved_domain = slug_domain(domain) if domain else derive_domain()
    server = _server_config(
        eventloom_path=eventloom_path,
        transport=transport,
        host=host,
        port=port,
        domain=resolved_domain,
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


def render_agent_integration_template(
    framework: AgentFramework | str,
    *,
    session_id: str = "default",
    eventloom_path: str = ".eventloom",
) -> str:
    """Render a dependency-light Python starter for an agent framework."""
    normalized = _normalize_framework(framework)
    if normalized == "langgraph":
        return _langgraph_template(session_id=session_id, eventloom_path=eventloom_path)
    if normalized == "crewai":
        return _crewai_template(session_id=session_id, eventloom_path=eventloom_path)
    return _autogen_template(session_id=session_id, eventloom_path=eventloom_path)


def _langgraph_template(*, session_id: str, eventloom_path: str) -> str:
    return f'''"""LangGraph starter for Zaxy memory.

Paste this into your LangGraph app and call `zaxy_langgraph_memory_node` from
your graph where you want durable memory capture and prompt context assembly.
"""

from zaxy import MemoryFabric


async def zaxy_langgraph_memory_node(state: dict) -> dict:
    fabric = MemoryFabric(eventloom_path={eventloom_path!r})
    await fabric.connect()
    try:
        content = str(state.get("latest_message", ""))
        context = await fabric.after_turn(
            role="assistant",
            content=content,
            session_id={session_id!r},
            query=content or "session context",
        )
        handoff = await fabric.handoff_bundle(
            session_id={session_id!r},
            query=content or "session handoff",
        )
        return {{**state, "zaxy_context": context.prompt, "zaxy_handoff": handoff.prompt}}
    finally:
        await fabric.close()
'''


def _crewai_template(*, session_id: str, eventloom_path: str) -> str:
    return f'''"""CrewAI starter for Zaxy memory.

Call `zaxy_crewai_memory_step` inside a task callback or before a task hands
context to the next crew member.
"""

from zaxy import MemoryFabric


async def zaxy_crewai_memory_step(message: str) -> str:
    fabric = MemoryFabric(eventloom_path={eventloom_path!r})
    await fabric.connect()
    try:
        context = await fabric.after_turn(
            role="assistant",
            content=message,
            session_id={session_id!r},
            query=message or "crew context",
        )
        handoff = await fabric.handoff_bundle(
            session_id={session_id!r},
            query=message or "crew handoff",
        )
        return "\n\n".join([context.prompt, handoff.prompt])
    finally:
        await fabric.close()
'''


def _autogen_template(*, session_id: str, eventloom_path: str) -> str:
    return f'''"""AutoGen starter for Zaxy memory.

Call `zaxy_autogen_context` from an agent hook before replying, then place the
returned prompt in your agent's system/context variables.
"""

from zaxy import MemoryFabric


async def zaxy_autogen_context(message: str) -> dict[str, str]:
    fabric = MemoryFabric(eventloom_path={eventloom_path!r})
    await fabric.connect()
    try:
        context = await fabric.after_turn(
            role="assistant",
            content=message,
            session_id={session_id!r},
            query=message or "autogen context",
        )
        handoff = await fabric.handoff_bundle(
            session_id={session_id!r},
            query=message or "autogen handoff",
        )
        return {{"zaxy_context": context.prompt, "zaxy_handoff": handoff.prompt}}
    finally:
        await fabric.close()
'''


def _server_config(
    *,
    eventloom_path: str,
    transport: str,
    host: str,
    port: int,
    domain: str,
) -> dict[str, Any]:
    normalized_transport = transport.casefold()
    default_session = domain_default_session(domain)
    if normalized_transport == "stdio":
        return {
            "command": "zaxy",
            "args": ["serve", "--eventloom-path", eventloom_path],
            "startup_timeout_sec": 90,
            "env": {
                "EVENTLOOM_PATH": eventloom_path,
                "EVENTLOOM_THREAD": default_session,
                "LOG_LEVEL": "ERROR",
                "MCP_ADMIN_TOKEN_FILE": "",
                "MCP_REMOTE_AUTH_TOKEN_FILE": "",
                "NEO4J_CA_CERT": "",
                "NEO4J_AUTO_START": "true",
                "NEO4J_PASSWORD_FILE": "",
                "NEO4J_URI": "bolt://localhost:7687",
                "OPENAI_API_KEY_FILE": "",
                "PATHLIGHT_ACCESS_TOKEN_FILE": "",
                "ZAXY_DOMAIN": domain,
                "ZAXY_ENV": "development",
            },
        }
    if normalized_transport == "sse":
        return {
            "url": f"http://{host}:{port}/sse",
            "headers": {"x-zaxy-session-id": default_session},
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


def _normalize_framework(framework: str) -> AgentFramework:
    normalized = framework.casefold().replace("_", "-")
    if normalized in {"langgraph", "crewai", "autogen"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("framework must be one of: langgraph, crewai, autogen")
