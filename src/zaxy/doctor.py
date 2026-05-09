"""Local environment preflight checks for Zaxy onboarding."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from zaxy.config import Settings, get_settings
from zaxy.event import EventLog
from zaxy.local_profile import check_local_profile
from zaxy.viewer import write_viewer_html


def run_doctor(
    *,
    settings: Settings | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run local setup checks without starting external services."""
    active = settings or get_settings()
    root = Path(workspace_root or Path.cwd())
    checks = [
        _check_eventloom(active),
        _check_local_profile(),
        _check_viewer(root),
        _check_mcp_defaults(active),
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
