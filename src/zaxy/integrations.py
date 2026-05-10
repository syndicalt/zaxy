"""First-run integration helpers for MCP clients and agent frameworks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import tomlkit

from zaxy.core import HandoffBundle
from zaxy.domain import derive_domain, domain_default_session, slug_domain
from zaxy.install import resolve_zaxy_executable

MCPClient = Literal["claude-desktop", "claude-code", "codex", "cursor", "vscode"]
CodexConfigScope = Literal["project", "user"]
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
    zaxy_executable: str | None = None,
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
        zaxy_executable=resolve_zaxy_executable(zaxy_executable),
    )
    if normalized == "vscode":
        return {"servers": {"zaxy": server}}
    if normalized == "codex":
        raise ValueError("Codex uses `codex mcp add`; use render_codex_mcp_add_command")
    return {"mcpServers": {"zaxy": server}}


def render_codex_mcp_add_command(
    *,
    eventloom_path: str = ".eventloom",
    domain: str | None = None,
    zaxy_executable: str | None = None,
) -> list[str]:
    """Render the official Codex CLI command for adding Zaxy as an MCP server."""
    resolved_domain = slug_domain(domain) if domain else derive_domain()
    server = _server_config(
        eventloom_path=eventloom_path,
        transport="stdio",
        host="127.0.0.1",
        port=8080,
        domain=resolved_domain,
        zaxy_executable=resolve_zaxy_executable(zaxy_executable),
    )
    env = server["env"]
    command = ["codex", "mcp", "add", "zaxy"]
    for key in sorted(env):
        command.extend(["--env", f"{key}={env[key]}"])
    command.append("--")
    command.append(str(server["command"]))
    command.extend(str(arg) for arg in server["args"])
    return command


def write_project_mcp_client_config(
    client: MCPClient | str,
    *,
    workspace: str | Path,
    eventloom_path: str = ".eventloom",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
    domain: str | None = None,
    zaxy_executable: str | None = None,
    force: bool = False,
) -> Path:
    """Merge Zaxy into a verified project-local MCP client config."""
    normalized = _normalize_client(client)
    if normalized == "codex":
        raise ValueError("Codex install is CLI-assisted; use render_codex_mcp_add_command")
    target = project_mcp_client_config_path(normalized, workspace=workspace)
    rendered = render_mcp_client_config(
        normalized,
        eventloom_path=eventloom_path,
        transport=transport,
        host=host,
        port=port,
        domain=domain,
        zaxy_executable=zaxy_executable,
    )
    root_key = _mcp_root_key(normalized)
    merged = _merge_mcp_config(target, root_key=root_key, server=rendered[root_key]["zaxy"], force=force)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_codex_mcp_config(
    *,
    scope: CodexConfigScope | str,
    workspace: str | Path,
    eventloom_path: str = ".eventloom",
    domain: str | None = None,
    zaxy_executable: str | None = None,
    force: bool = False,
    trusted_project: bool = False,
    codex_home: str | Path | None = None,
) -> Path:
    """Merge Zaxy into an explicit Codex TOML config scope."""
    normalized_scope = _normalize_codex_scope(scope)
    if normalized_scope == "project" and not trusted_project:
        raise PermissionError(
            "project-scoped Codex config requires an explicit trusted project acknowledgement"
        )
    target = codex_mcp_config_path(
        scope=normalized_scope,
        workspace=workspace,
        codex_home=codex_home,
    )
    server = _server_config(
        eventloom_path=eventloom_path,
        transport="stdio",
        host="127.0.0.1",
        port=8080,
        domain=slug_domain(domain) if domain else derive_domain(),
        zaxy_executable=resolve_zaxy_executable(zaxy_executable),
    )
    document = _read_toml_document(target)
    mcp_servers = document.setdefault("mcp_servers", tomlkit.table())
    if not isinstance(mcp_servers, dict):
        raise ValueError(f"{target} field 'mcp_servers' must contain a TOML table")
    if "zaxy" in mcp_servers and not force:
        raise FileExistsError(f"{target} already contains a zaxy MCP server; pass --force to replace it")
    zaxy = tomlkit.table()
    zaxy.add("command", server["command"])
    zaxy.add("args", server["args"])
    env = tomlkit.table()
    for key, value in server["env"].items():
        env.add(key, value)
    zaxy.add("env", env)
    zaxy.add("startup_timeout_sec", server["startup_timeout_sec"])
    mcp_servers["zaxy"] = zaxy
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tomlkit.dumps(document), encoding="utf-8")
    return target


def codex_mcp_config_path(
    *,
    scope: CodexConfigScope | str,
    workspace: str | Path,
    codex_home: str | Path | None = None,
) -> Path:
    """Return the explicit Codex TOML config path for a scope."""
    normalized_scope = _normalize_codex_scope(scope)
    if normalized_scope == "project":
        return Path(workspace) / ".codex" / "config.toml"
    home = Path(codex_home) if codex_home is not None else Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    return home / "config.toml"


def project_mcp_client_config_path(client: MCPClient | str, *, workspace: str | Path) -> Path:
    """Return the verified project-local MCP config path for a supported client."""
    normalized = _normalize_client(client)
    root = Path(workspace)
    if normalized in {"claude-code", "claude-desktop"}:
        return root / ".mcp.json"
    if normalized == "codex":
        raise ValueError("Codex does not have a safe JSON project config target")
    if normalized == "cursor":
        return root / ".cursor" / "mcp.json"
    return root / ".vscode" / "mcp.json"


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
    zaxy_executable: str,
) -> dict[str, Any]:
    normalized_transport = transport.casefold()
    default_session = domain_default_session(domain)
    if normalized_transport == "stdio":
        return {
            "command": zaxy_executable,
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
    if normalized in {"claude-code", "claude-cli"}:
        return "claude-code"
    if normalized == "codex":
        return "codex"
    if normalized == "cursor":
        return "cursor"
    if normalized in {"vscode", "vs-code", "visual-studio-code"}:
        return "vscode"
    raise ValueError("client must be one of: claude-desktop, claude-code, codex, cursor, vscode")


def _normalize_codex_scope(scope: str) -> CodexConfigScope:
    normalized = scope.casefold().strip()
    if normalized in {"project", "user"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("Codex config scope must be one of: project, user")


def _read_toml_document(path: Path) -> Any:
    if not path.exists():
        return tomlkit.document()
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except tomlkit.exceptions.TOMLKitError as exc:
        raise ValueError(f"{path} contains invalid TOML; repair it before installing Zaxy") from exc


def _mcp_root_key(client: MCPClient) -> str:
    if client == "vscode":
        return "servers"
    return "mcpServers"


def _merge_mcp_config(
    path: Path,
    *,
    root_key: str,
    server: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    config: dict[str, Any]
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} contains invalid JSON; repair it before installing Zaxy") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{path} must contain a JSON object")
        config = parsed
    else:
        config = {}
    servers = config.setdefault(root_key, {})
    if not isinstance(servers, dict):
        raise ValueError(f"{path} field {root_key!r} must contain a JSON object")
    if "zaxy" in servers and not force:
        raise FileExistsError(f"{path} already contains a zaxy MCP server; pass --force to replace it")
    servers["zaxy"] = server
    return config


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
