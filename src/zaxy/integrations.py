"""First-run integration helpers for MCP clients and agent frameworks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import tomlkit
import yaml

from zaxy.domain import derive_domain, domain_default_session, slug_domain
from zaxy.install import resolve_zaxy_executable

if TYPE_CHECKING:
    from zaxy.core import HandoffBundle

MCPClient = Literal["claude-desktop", "claude-code", "codex", "cursor", "hermes", "vscode"]
CodexConfigScope = Literal["project", "user"]
HandoffAdapter = Literal["generic", "langgraph", "crewai", "autogen"]
AgentFramework = Literal["langgraph", "crewai", "autogen"]
FrameworkExtra = Literal["langgraph", "crewai", "autogen", "frameworks"]
CoordinationAdapter = Literal["codex", "langgraph", "crewai", "mcp"]

HERMES_MODEL_FACING_TOOLS: tuple[str, ...] = (
    "memory_capabilities",
    "memory_bootstrap",
    "memory_checkout",
    "memory_feedback",
    "memory_query",
    "memory_verbatim",
    "memory_append",
)


@dataclass(frozen=True)
class FrameworkIntegrationSpec:
    """Discovery metadata for a direct framework integration path."""

    framework: AgentFramework
    display_name: str
    package: str
    extra: FrameworkExtra
    template_function: str
    maturity: Literal["template", "native-preview", "native"]
    native_adapter: str


@dataclass(frozen=True)
class FrameworkIntegrationDecision:
    """Maintained adapter roadmap decision derived from preview integration state."""

    target: str
    track: Literal["native-adapter", "model-facing-ux"]
    recommended: bool
    evidence_frameworks: tuple[AgentFramework, ...]
    hold_frameworks: tuple[AgentFramework, ...]
    rationale: str
    next_actions: tuple[str, ...]


_FRAMEWORK_SPECS: tuple[FrameworkIntegrationSpec, ...] = (
    FrameworkIntegrationSpec(
        framework="langgraph",
        display_name="LangGraph",
        package="langgraph",
        extra="langgraph",
        template_function="create_langgraph_memory_node",
        maturity="native-preview",
        native_adapter="zaxy.adapters.langgraph",
    ),
    FrameworkIntegrationSpec(
        framework="crewai",
        display_name="CrewAI",
        package="crewai",
        extra="crewai",
        template_function="create_crewai_memory_step",
        maturity="native-preview",
        native_adapter="zaxy.adapters.crewai",
    ),
    FrameworkIntegrationSpec(
        framework="autogen",
        display_name="AutoGen",
        package="autogen-agentchat",
        extra="autogen",
        template_function="zaxy_autogen_context",
        maturity="template",
        native_adapter="not-yet-packaged",
    ),
)


def list_framework_integration_specs() -> tuple[FrameworkIntegrationSpec, ...]:
    """Return direct framework integration metadata in display order."""
    return _FRAMEWORK_SPECS


def recommend_framework_integration_target() -> FrameworkIntegrationDecision:
    """Return the next framework integration target from maintained preview usage.

    LangGraph and CrewAI already exercise the same lifecycle surface without
    requiring framework imports. AutoGen is still template-only because its
    stable runtime hook shape is not yet proven by local usage.
    """
    return FrameworkIntegrationDecision(
        target="common-native-preview-contract",
        track="model-facing-ux",
        recommended=True,
        evidence_frameworks=("langgraph", "crewai"),
        hold_frameworks=("autogen",),
        rationale=(
            "LangGraph and CrewAI native-preview adapters already prove the "
            "shared Memory Checkout, observation, and feedback flow. AutoGen "
            "should stay template-only until its runtime hooks are validated, "
            "so the next maintained target is hardening the common "
            "native-preview payload contract rather than adding another "
            "speculative adapter."
        ),
        next_actions=(
            "stabilize shared payload keys across native-preview adapters",
            "keep AutoGen template-only until runtime hooks are validated",
            "use adapter feedback events to decide the next native package",
        ),
    )


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
    resolved_executable = resolve_zaxy_executable(zaxy_executable)
    if normalized == "codex":
        raise ValueError("Codex uses `codex mcp add`; use render_codex_mcp_add_command")
    if normalized == "hermes":
        if transport.casefold() != "stdio":
            raise ValueError("Hermes first-class install currently supports stdio MCP only")
        return {
            "mcp_servers": {
                "zaxy": _hermes_server_config(zaxy_executable=resolved_executable)
            }
        }

    resolved_domain = slug_domain(domain) if domain else derive_domain()
    server = _server_config(
        eventloom_path=eventloom_path,
        transport=transport,
        host=host,
        port=port,
        domain=resolved_domain,
        zaxy_executable=resolved_executable,
    )
    if normalized == "vscode":
        return {"servers": {"zaxy": server}}
    return {"mcpServers": {"zaxy": server}}


def render_codex_mcp_add_command(
    *,
    eventloom_path: str = ".eventloom",
    domain: str | None = None,
    zaxy_executable: str | None = None,
) -> list[str]:
    """Render the official Codex CLI command for adding Zaxy as an MCP server."""
    server = _codex_server_config(zaxy_executable=resolve_zaxy_executable(zaxy_executable))
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
    server = _codex_server_config(zaxy_executable=resolve_zaxy_executable(zaxy_executable))
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


def write_hermes_mcp_config(
    *,
    config_path: str | Path | None = None,
    hermes_home: str | Path | None = None,
    zaxy_executable: str | None = None,
    force: bool = False,
    domain: str | None = None,
) -> Path:
    """Merge Zaxy into a Hermes Agent config.yaml MCP server block.

    Hermes MCP config is global YAML under ``mcp_servers``. Zaxy therefore
    writes a workspace-neutral server: `zaxy serve` resolves the active
    workspace and Eventloom defaults at runtime instead of pinning one repo into
    the global Hermes config.
    """
    _ = domain
    target = hermes_mcp_config_path(config_path=config_path, hermes_home=hermes_home)
    document = _read_yaml_document(target)
    raw_servers = document.setdefault("mcp_servers", {})
    if not isinstance(raw_servers, dict):
        raise ValueError(f"{target} field 'mcp_servers' must contain a YAML mapping")
    servers = cast(dict[str, Any], raw_servers)
    if "zaxy" in servers and not force:
        raise FileExistsError(
            f"{target} already contains a zaxy MCP server; pass --force to replace it"
        )
    servers["zaxy"] = _hermes_server_config(
        zaxy_executable=resolve_zaxy_executable(zaxy_executable)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return target


def hermes_mcp_config_path(
    *,
    config_path: str | Path | None = None,
    hermes_home: str | Path | None = None,
) -> Path:
    """Return the Hermes Agent YAML config path."""
    if config_path is not None:
        return Path(config_path)
    home = (
        Path(hermes_home)
        if hermes_home is not None
        else Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    )
    return home / "config.yaml"


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
    if normalized == "hermes":
        raise ValueError("Hermes uses global config.yaml; use write_hermes_mcp_config")
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


def render_coordination_adapter_template(
    adapter: CoordinationAdapter | str,
    *,
    mission_id: str,
    worker_id: str,
    eventloom_path: str = ".eventloom",
) -> str:
    """Render a dependency-light starter for Zaxy Coordinate workflows."""
    normalized = _normalize_coordination_adapter(str(adapter))
    if normalized == "codex":
        return _codex_coordination_template(
            mission_id=mission_id,
            worker_id=worker_id,
            eventloom_path=eventloom_path,
        )
    if normalized == "langgraph":
        return _langgraph_coordination_template(
            mission_id=mission_id,
            worker_id=worker_id,
            eventloom_path=eventloom_path,
        )
    if normalized == "crewai":
        return _crewai_coordination_template(
            mission_id=mission_id,
            worker_id=worker_id,
            eventloom_path=eventloom_path,
        )
    return _mcp_coordination_template(
        mission_id=mission_id,
        worker_id=worker_id,
        eventloom_path=eventloom_path,
    )


def render_framework_install_command(
    framework: FrameworkExtra | str,
    *,
    package_name: str = "zaxy-memory",
) -> list[str]:
    """Render a pip install command for an optional framework integration extra."""
    normalized = _normalize_framework_extra(framework)
    return ["python", "-m", "pip", "install", f"{package_name}[{normalized}]"]


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

from zaxy.adapters.crewai import CrewAIMemoryAdapter


async def zaxy_crewai_memory_step(message: str) -> str:
    adapter = CrewAIMemoryAdapter(session_id={session_id!r}, eventloom_path={eventloom_path!r})
    payload = await adapter.before_task(message)
    return str(payload["memory"])


async def zaxy_crewai_record_result(result: str) -> str:
    adapter = CrewAIMemoryAdapter(session_id={session_id!r}, eventloom_path={eventloom_path!r})
    payload = await adapter.after_task(result)
    return str(payload["memory"])
'''


def _autogen_template(*, session_id: str, eventloom_path: str) -> str:
    return f'''"""AutoGen starter for Zaxy memory.

Call `zaxy_autogen_context` from an agent hook before replying. It runs Memory
Checkout so the agent conditions on cited current memory instead of relying on
stale session state.
"""

from zaxy import MemoryFabric


async def zaxy_autogen_context(message: str) -> dict[str, str]:
    fabric = MemoryFabric(eventloom_path={eventloom_path!r})
    await fabric.connect()
    try:
        checkout = await fabric.checkout_memory(
            message or "autogen context",
            session_id={session_id!r},
        )
        return {{"zaxy_context": checkout.prompt}}
    finally:
        await fabric.close()


async def zaxy_autogen_record_reply(reply: str) -> dict[str, str]:
    fabric = MemoryFabric(eventloom_path={eventloom_path!r})
    await fabric.connect()
    try:
        context = await fabric.after_turn(
            role="assistant",
            content=reply,
            session_id={session_id!r},
            query=reply or "autogen reply",
        )
        return {{"zaxy_context": context.prompt}}
    finally:
        await fabric.close()
'''


def _codex_coordination_template(*, mission_id: str, worker_id: str, eventloom_path: str) -> str:
    return f'''"""Codex-style local worker starter for Zaxy Coordinate.

Use this from a local agent wrapper, task script, or post-run hook. The helper
records worker-local findings only; a coordinator still decides what becomes
accepted parent mission state.
"""

from zaxy.adapters.coordination import CoordinationAdapter


mission_id={mission_id!r}
worker_id={worker_id!r}
adapter = CoordinationAdapter(eventloom_path={eventloom_path!r}, actor=worker_id)


def report_finding(summary: str, evidence: list[dict] | None = None, confidence: float | None = None) -> dict:
    return adapter.report_finding(
        mission_id,
        worker_id,
        summary=summary,
        evidence=evidence or [],
        confidence=confidence,
    )


def finish_worker(summary: str, next_steps: list[str] | None = None, risks: list[str] | None = None) -> dict:
    return adapter.handoff(
        mission_id,
        summary=summary,
        next_steps=next_steps or [],
        risks=risks or [],
    )
'''


def _langgraph_coordination_template(*, mission_id: str, worker_id: str, eventloom_path: str) -> str:
    return f'''"""LangGraph node starter for Zaxy Coordinate.

The node is dependency-light and has no framework import. It reports a
worker-local finding from explicit state fields and returns Zaxy coordination
metadata beside the original state.
"""

from zaxy.adapters.coordination import CoordinationAdapter


async def zaxy_coordinate_langgraph_node(state: dict) -> dict:
    adapter = CoordinationAdapter(eventloom_path={eventloom_path!r}, actor={worker_id!r})
    finding = adapter.report_finding(
        mission_id={mission_id!r},
        worker_id={worker_id!r},
        summary=str(state.get("coordination_summary") or state.get("latest_message") or ""),
        evidence=state.get("coordination_evidence") or [],
        confidence=state.get("coordination_confidence"),
        claim_key=state.get("coordination_claim_key"),
        claim_value=state.get("coordination_claim_value"),
    )
    return {{**state, "zaxy_coordination": finding}}
'''


def _crewai_coordination_template(*, mission_id: str, worker_id: str, eventloom_path: str) -> str:
    return f'''"""CrewAI task-step starter for Zaxy Coordinate.

Call this from a task callback or application-owned wrapper. It avoids CrewAI
imports so your application keeps control of its Crew runtime objects.
"""

from zaxy.adapters.coordination import CoordinationAdapter


async def zaxy_coordinate_crewai_step(summary: str, evidence: list[dict] | None = None) -> dict:
    adapter = CoordinationAdapter(eventloom_path={eventloom_path!r}, actor={worker_id!r})
    return adapter.report_finding(
        mission_id={mission_id!r},
        worker_id={worker_id!r},
        summary=summary,
        evidence=evidence or [],
    )
'''


def _mcp_coordination_template(*, mission_id: str, worker_id: str, eventloom_path: str) -> str:
    payload = [
        {
            "tool": "coordination_start",
            "arguments": {
                "eventloom_path": eventloom_path,
                "mission_id": mission_id,
                "objective": "replace with mission objective",
            },
        },
        {
            "tool": "coordination_worker_create",
            "arguments": {
                "eventloom_path": eventloom_path,
                "mission_id": mission_id,
                "worker_id": worker_id,
            },
        },
        {
            "tool": "coordination_report_finding",
            "arguments": {
                "eventloom_path": eventloom_path,
                "mission_id": mission_id,
                "worker_id": worker_id,
                "summary": "replace with worker-local finding",
                "evidence": [],
            },
        },
        {
            "tool": "coordination_checkout",
            "arguments": {
                "eventloom_path": eventloom_path,
                "mission_id": mission_id,
                "include_diagnostics": True,
            },
        },
    ]
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


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
        embedded_graph_path = f"{eventloom_path.rstrip('/')}/projections/embedded.kuzu"
        return {
            "command": zaxy_executable,
            "args": ["serve", "--eventloom-path", eventloom_path],
            "startup_timeout_sec": 90,
            "env": {
                "EMBEDDED_GRAPH_PATH": embedded_graph_path,
                "EVENTLOOM_PATH": eventloom_path,
                "EVENTLOOM_THREAD": default_session,
                "LOG_LEVEL": "ERROR",
                "MCP_ADMIN_TOKEN_FILE": "",
                "MCP_REMOTE_AUTH_TOKEN_FILE": "",
                "NEO4J_CA_CERT": "",
                "NEO4J_AUTO_START": "false",
                "NEO4J_PASSWORD_FILE": "",
                "NEO4J_URI": "bolt://localhost:7687",
                "OPENAI_API_KEY_FILE": "",
                "PATHLIGHT_ACCESS_TOKEN_FILE": "",
                "PGGRAPH_AUTO_START": "false",
                "PROJECTION_BACKEND": "embedded",
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


def _codex_server_config(*, zaxy_executable: str) -> dict[str, Any]:
    """Return a workspace-neutral Codex MCP server config.

    Codex MCP config can be global; repo-specific Eventloom/session state must
    be resolved by `zaxy serve` from the process workspace at runtime.
    """
    return {
        "command": zaxy_executable,
        "args": ["serve"],
        "startup_timeout_sec": 90,
        "env": {
            "LOG_LEVEL": "ERROR",
            "MCP_ADMIN_TOKEN_FILE": "",
            "MCP_REMOTE_AUTH_TOKEN_FILE": "",
            "OPENAI_API_KEY_FILE": "",
            "PATHLIGHT_ACCESS_TOKEN_FILE": "",
            "ZAXY_ENV": "development",
        },
    }


def _hermes_server_config(*, zaxy_executable: str) -> dict[str, Any]:
    """Return a workspace-neutral Hermes MCP server config."""
    server = _codex_server_config(zaxy_executable=zaxy_executable)
    return {
        "command": server["command"],
        "args": server["args"],
        "env": server["env"],
        "enabled": True,
        "timeout": 120,
        "connect_timeout": 60,
        "tools": {
            "include": list(HERMES_MODEL_FACING_TOOLS),
            "resources": False,
            "prompts": False,
        },
    }


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
    if normalized == "hermes":
        return "hermes"
    if normalized in {"vscode", "vs-code", "visual-studio-code"}:
        return "vscode"
    raise ValueError(
        "client must be one of: claude-desktop, claude-code, codex, cursor, hermes, vscode"
    )


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


def _read_yaml_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} contains invalid YAML; repair it before installing Zaxy") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return cast(dict[str, Any], parsed)


def _mcp_root_key(client: MCPClient) -> str:
    if client == "vscode":
        return "servers"
    if client == "hermes":
        return "mcp_servers"
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


def _normalize_framework_extra(framework: str) -> FrameworkExtra:
    normalized = framework.casefold().replace("_", "-")
    if normalized in {"all", "framework", "frameworks"}:
        return "frameworks"
    for spec in _FRAMEWORK_SPECS:
        if normalized == spec.framework:
            return spec.extra
    raise ValueError("framework extra must be one of: langgraph, crewai, autogen, frameworks")


def _normalize_coordination_adapter(adapter: str) -> CoordinationAdapter:
    normalized = adapter.casefold().replace("_", "-")
    if normalized in {"codex", "langgraph", "crewai", "mcp"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("coordination adapter must be one of: codex, langgraph, crewai, mcp")
