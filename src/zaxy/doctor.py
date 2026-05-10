"""Local environment preflight checks for Zaxy onboarding."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from zaxy.config import Settings, get_settings
from zaxy.event import EventLog
from zaxy.hooks import HOOK_CLIENTS, inspect_hook_status, render_hook_config
from zaxy.install import resolve_zaxy_executable
from zaxy.local_profile import check_local_profile
from zaxy.security import eventlog_path
from zaxy.viewer import write_viewer_html


def run_doctor(
    *,
    settings: Settings | None = None,
    workspace_root: str | Path | None = None,
    zaxy_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Run local setup checks without starting external services."""
    active = settings or get_settings()
    root = Path(workspace_root or Path.cwd())
    hook_status = inspect_hook_status(eventloom_path=active.eventloom_path, workspace_root=root)
    checks = [
        _check_eventloom(active),
        _check_local_profile(),
        _check_viewer(root),
        _check_cli_install(zaxy_executable),
        _check_mcp_defaults(active),
        _check_hooks(active),
        _check_hook_installation(hook_status),
        _check_hook_activity(active, hook_status),
        _check_observation_coverage(hook_status),
        _check_packet_memory(active),
        _check_neo4j(active),
        _check_production(active),
    ]
    return {
        "status": _overall_status(checks),
        "checks": checks,
    }


def format_doctor_report(report: dict[str, Any]) -> str:
    """Format a doctor report for humans."""
    lines = [f"Zaxy doctor: {report['status']}"]
    for check in report["checks"]:
        lines.append(f"- {check['name']}: {check['status']} - {check['message']}")
        details = check.get("details")
        if isinstance(details, dict) and details:
            rendered = " ".join(f"{key}={value}" for key, value in details.items())
            lines.append(f"  details: {rendered}")
        action = check.get("action")
        if action:
            lines.append(f"  action: {action}")
    return "\n".join(lines)


def _check_eventloom(settings: Settings) -> dict[str, str]:
    path = Path(settings.eventloom_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor-write-test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return {
            "name": "eventloom",
            "status": "error",
            "message": f"{path} is not writable: {exc}",
            "action": "Set EVENTLOOM_PATH to a writable directory.",
        }
    return {
        "name": "eventloom",
        "status": "ok",
        "message": f"{path} is writable",
    }


def _check_local_profile() -> dict[str, str]:
    try:
        report = check_local_profile()
    except Exception as exc:
        return {
            "name": "local_profile",
            "status": "error",
            "message": f"local embedding/reranker providers failed: {exc}",
            "action": "Run zaxy local-profile --check for details.",
        }
    return {
        "name": "local_profile",
        "status": "ok",
        "message": (
            f"{report['embedding_provider']} embeddings and "
            f"{report['reranker_provider']} reranker are available"
        ),
    }


def _check_viewer(workspace_root: Path) -> dict[str, str]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "doctor.jsonl"
            EventLog(log_path).append(
                "session.genesis",
                actor="zaxy",
                payload={"session_id": "doctor", "root": str(workspace_root)},
                thread="doctor",
            )
            write_viewer_html(log_path, tmp_path / "viewer.html")
    except Exception as exc:
        return {
            "name": "viewer",
            "status": "error",
            "message": f"static viewer generation failed: {exc}",
            "action": "Run zaxy viewer .eventloom --output eventloom-viewer.html.",
        }
    return {
        "name": "viewer",
        "status": "ok",
        "message": "static Eventloom viewer generation works",
    }


def _check_cli_install(zaxy_executable: str | Path | None) -> dict[str, str]:
    executable = resolve_zaxy_executable(zaxy_executable)
    return {
        "name": "cli_install",
        "status": "ok",
        "message": f"Zaxy CLI executable resolved to {executable}",
    }


def _check_mcp_defaults(settings: Settings) -> dict[str, str]:
    if settings.eventloom_thread == "default":
        return {
            "name": "mcp_defaults",
            "status": "warning",
            "message": "EVENTLOOM_THREAD is still default and may bleed across projects",
            "action": "Run zaxy ide-config <client> --domain <project> and use the generated env.",
        }
    if not settings.mcp_lifecycle_capture_enabled:
        return {
            "name": "mcp_defaults",
            "status": "warning",
            "message": "MCP lifecycle capture is disabled",
            "action": "Set MCP_LIFECYCLE_CAPTURE_ENABLED=true for automatic tool-call metadata.",
        }
    return {
        "name": "mcp_defaults",
        "status": "ok",
        "message": f"default session is {settings.eventloom_thread}",
    }


def _check_hooks(settings: Settings) -> dict[str, str]:
    try:
        render_hook_config(
            "claude-code",
            eventloom_path=settings.eventloom_path,
            domain=settings.zaxy_domain,
        )
    except Exception as exc:
        return {
            "name": "hooks",
            "status": "error",
            "message": f"observer hook adapter rendering failed: {exc}",
            "action": (
                "Run zaxy hooks claude-code --eventloom-path .eventloom "
                "--domain <project> --output .claude/settings.local.json."
            ),
        }
    return {
        "name": "hooks",
        "status": "ok",
        "message": "observer hook adapters are available",
        "action": (
            "Run zaxy hooks claude-code --eventloom-path .eventloom "
            "--domain <project> --output .claude/settings.local.json."
        ),
    }


def _check_hook_installation(hook_status: dict[str, Any]) -> dict[str, str]:
    for client in HOOK_CLIENTS:
        info = hook_status["clients"][client]
        if info["paths"]:
            rel = info["paths"][0]
            label = client
            if client == "claude-code":
                label = "Claude Code"
            return {
                "name": "hook_installation",
                "status": "ok",
                "message": f"{label} observer hook config found at {rel}",
            }
    return {
        "name": "hook_installation",
        "status": "warning",
        "message": "No installed observer hook config found in this workspace",
        "action": (
            "Run zaxy hooks claude-code --eventloom-path .eventloom "
            "--domain <project> --output .claude/settings.local.json."
        ),
    }


def _check_hook_activity(settings: Settings, hook_status: dict[str, Any]) -> dict[str, str]:
    latest = hook_status.get("latest_event")
    if latest:
        return {
            "name": "hook_activity",
            "status": "ok",
            "message": (
                f"latest observed hook event is {latest['type']} in "
                f"{latest['thread']} at {latest['timestamp']}"
            ),
        }
    installed = any(client["installed"] for client in hook_status["clients"].values())
    if installed:
        return {
            "name": "hook_activity",
            "status": "warning",
            "message": "No hook lifecycle events observed after installed hook config",
            "action": (
                "Run zaxy hook-event heartbeat --eventloom-path .eventloom "
                f"--session-id {settings.eventloom_thread} --source manual."
            ),
        }
    return {
        "name": "hook_activity",
        "status": "warning",
        "message": "No hook lifecycle events observed",
        "action": (
            "Install hooks with zaxy hooks claude-code --eventloom-path .eventloom "
            "--domain <project> --output .claude/settings.local.json, then run "
            "zaxy hook-event heartbeat."
        ),
    }


def _check_observation_coverage(hook_status: dict[str, Any]) -> dict[str, str]:
    missing = hook_status.get("missing_observation_types", [])
    if not missing:
        return {
            "name": "observation_coverage",
            "status": "ok",
            "message": "high-value automatic observation types have been captured",
        }
    missing_types = ", ".join(str(event_type) for event_type in missing)
    return {
        "name": "observation_coverage",
        "status": "warning",
        "message": f"missing high-value automatic observation types: {missing_types}",
        "action": "Confirm hooks emit command, file-edit, tool-call, and transcript observations for this client.",
    }


def _check_packet_memory(settings: Settings) -> dict[str, str]:
    log = EventLog(eventlog_path(Path(settings.eventloom_path), settings.eventloom_thread))
    status = _packet_memory_status(log.read_all())
    details = status["details"]
    if details["captured"] == 0:
        return {
            "name": "packet_memory",
            "status": "warning",
            "message": "no LLM packet captures observed for this session",
            "details": details,
            "action": (
                "Optional: run zaxy packet-analyzer --eventloom-path "
                f"{settings.eventloom_path} --session-id {settings.eventloom_thread} "
                "--upstream-base-url <provider-v1-url>, then run zaxy packet-project --watch."
            ),
        }
    if details["unprojected"]:
        count = details["unprojected"]
        noun = "event has" if count == 1 else "events have"
        return {
            "name": "packet_memory",
            "status": "warning",
            "message": f"{count} captured packet {noun} not been projected",
            "details": details,
            "action": (
                "Run zaxy packet-project --watch --eventloom-path "
                f"{settings.eventloom_path} --session-id {settings.eventloom_thread}."
            ),
        }
    count = details["captured"]
    noun = "capture has" if count == 1 else "captures have"
    return {
        "name": "packet_memory",
        "status": "ok",
        "message": f"{count} packet {noun} projected memory",
        "details": details,
    }


def _packet_memory_status(events: list[Any]) -> dict[str, Any]:
    completed = [event for event in events if event.type == "llm.packet.completed"]
    projected_hashes = {
        event.payload.get("source_event_hash")
        for event in events
        if event.type == "llm.packet.projected"
    }
    reinforced_hashes = {
        event.payload.get("source_event_hash")
        for event in events
        if event.type == "memory.reinforced"
        and event.payload.get("entity_type") == "packet_memory"
        and event.payload.get("source_event_hash")
    }
    unprojected = [event for event in completed if event.hash not in projected_hashes]
    eligible_hashes = {event.hash for event in completed if event.hash in projected_hashes}
    return {
        "details": {
            "captured": len(completed),
            "projected": len(projected_hashes),
            "unprojected": len(unprojected),
            "reinforced": len(reinforced_hashes),
            "eligible": len(eligible_hashes),
        }
    }


def _check_neo4j(settings: Settings) -> dict[str, str]:
    parsed = urlparse(settings.neo4j_uri)
    if parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}:
        return {
            "name": "neo4j",
            "status": "error",
            "message": f"unsupported Neo4j URI scheme: {settings.neo4j_uri}",
            "action": "Use a bolt://, bolt+s://, neo4j://, or neo4j+s:// URI.",
        }
    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"} and settings.neo4j_auto_start:
        return {
            "name": "neo4j",
            "status": "ok",
            "message": "local Neo4j can be auto-started when needed",
        }
    return {
        "name": "neo4j",
        "status": "warning",
        "message": f"Neo4j is configured at {settings.neo4j_uri}; reachability not checked",
        "action": "Run zaxy status to test a live graph connection.",
    }


def _check_production(settings: Settings) -> dict[str, str]:
    if settings.zaxy_env.lower() != "production":
        return {
            "name": "production",
            "status": "ok",
            "message": f"running in {settings.zaxy_env} mode",
        }
    warnings: list[str] = []
    if settings.neo4j_password == "testpassword":
        warnings.append("NEO4J_PASSWORD uses the development default")
    if settings.neo4j_uri.startswith("bolt://") and not settings.neo4j_ca_cert:
        warnings.append("Neo4j TLS evidence is missing")
    if not settings.mcp_admin_token:
        warnings.append("MCP_ADMIN_TOKEN is not configured")
    if warnings:
        return {
            "name": "production",
            "status": "error",
            "message": "; ".join(warnings),
            "action": "Use scripts/validate-deployment.sh --root . before production exposure.",
        }
    return {
        "name": "production",
        "status": "ok",
        "message": "production security posture has required settings",
    }


def _overall_status(checks: list[dict[str, str]]) -> str:
    statuses = {check["status"] for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"
