"""Local environment preflight checks for Zaxy onboarding."""

from __future__ import annotations

import os
import socket
import tempfile
import tomllib
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from zaxy.config import Settings, get_settings
from zaxy.embedded_graph_store import (
    LEGACY_EMBEDDING_VERSION,
    VECTOR_INDEX_CACHE_MAX_BYTES,
    VECTOR_INDEX_CACHE_MAX_ENTRIES,
)
from zaxy.embedding import (
    HashEmbeddingProvider,
    active_embedding_version_tag,
    build_embedding_provider,
)
from zaxy.event import Event, EventLog
from zaxy.hooks import HOOK_CLIENTS, inspect_hook_status, render_hook_config
from zaxy.install import resolve_zaxy_executable
from zaxy.local_profile import check_local_profile
from zaxy.mcp_runtime import EmbeddedMcpRuntimeCoordinator
from zaxy.packet_guidance import build_packet_capture_guidance
from zaxy.runtime import LocalEmbeddedGraphRuntime, LocalPgGraphRuntime
from zaxy.security import eventlog_path
from zaxy.viewer import write_viewer_html

AGENT_ACTIVATION_BEGIN = "<!-- zaxy-memory-activation:start -->"
AGENT_ACTIVATION_END = "<!-- zaxy-memory-activation:end -->"

# Full hash-chain verification is cheap below this log size; larger logs get a
# bounded tail verification so doctor stays fast on long-lived sessions.
EVENT_CHAIN_FULL_VERIFY_MAX_BYTES = 8 * 1024 * 1024
EVENT_CHAIN_TAIL_EVENTS = 512
# Minimum float64 vectors the in-process vector index cache budget should hold
# at the configured embedding dimension before doctor flags the headroom.
VECTOR_CACHE_MIN_VECTOR_HEADROOM = 1024


def run_doctor(
    *,
    settings: Settings | None = None,
    workspace_root: str | Path | None = None,
    zaxy_executable: str | Path | None = None,
    repair: bool = False,
) -> dict[str, Any]:
    """Run local setup checks without starting external services."""
    active = settings or get_settings()
    root = Path(workspace_root or Path.cwd())
    hook_status = inspect_hook_status(
        eventloom_path=active.eventloom_path,
        workspace_root=root,
        session_id=active.eventloom_thread,
    )
    checks = [
        _check_version_consistency(root),
        _check_eventloom(active),
        _check_event_chain(active),
        _check_local_profile(),
        _check_embedding_provider(active),
        _check_vector_cache_budget(active),
        _check_embedding_versions(active),
        _check_viewer(root),
        _check_cli_install(zaxy_executable),
        _check_mcp_defaults(active),
        _check_codex_mcp_scope(root),
        _check_agent_instructions(root),
        _check_hooks(active),
        _check_hook_installation(hook_status),
        _check_hook_activity(active, hook_status),
        _check_observation_coverage(hook_status),
        _check_capture_health(hook_status),
        _check_memory_activation(hook_status),
        _check_packet_memory(active),
        _check_embedded_mcp_runtime(active, repair=repair),
        _check_projection_backend(active),
        _check_projection_freshness(active),
        _check_projection_backup_artifacts(active),
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


def _check_event_chain(settings: Settings) -> dict[str, Any]:
    """Verify hash-chain integrity over the active session event log."""
    log_path = eventlog_path(Path(settings.eventloom_path), settings.eventloom_thread)
    if not log_path.exists():
        return {
            "name": "event_chain",
            "status": "ok",
            "message": f"no active event log at {log_path} yet",
        }
    log = EventLog(log_path)
    try:
        size = log_path.stat().st_size
        if size <= EVENT_CHAIN_FULL_VERIFY_MAX_BYTES:
            mode = "full"
            report = log.verify()
            ok = report.ok
            verified = report.total_events
            reason = report.broken_reason
        else:
            mode = f"tail({EVENT_CHAIN_TAIL_EVENTS})"
            ok, verified, reason = _verify_event_tail(log.tail_events(EVENT_CHAIN_TAIL_EVENTS))
    except Exception as exc:
        return {
            "name": "event_chain",
            "status": "error",
            "message": f"event log {log_path} is unreadable: {exc}",
            "action": "Restore the Eventloom log from backup; never edit the append-only log in place.",
        }
    if not ok:
        return {
            "name": "event_chain",
            "status": "error",
            "message": f"hash chain broken in {log_path}: {reason}",
            "details": {"mode": mode, "events_verified": verified},
            "action": "Restore the Eventloom log from backup; never edit the append-only log in place.",
        }
    return {
        "name": "event_chain",
        "status": "ok",
        "message": f"hash chain verified over {verified} events ({mode})",
        "details": {"mode": mode, "events_verified": verified},
    }


def _verify_event_tail(events: list[Event]) -> tuple[bool, int, str | None]:
    """Verify per-event seals and prev_hash linkage within a log tail window."""
    previous: Event | None = None
    for event in events:
        if not event.verify():
            return False, len(events), f"Event {event.seq} hash mismatch"
        if previous is not None:
            if event.seq != previous.seq + 1:
                return False, len(events), f"Event sequence expected {previous.seq + 1} but found {event.seq}"
            if event.prev_hash != previous.hash:
                return False, len(events), f"Event {event.seq} prev_hash does not link to previous"
        previous = event
    return True, len(events), None


def _check_embedding_provider(settings: Settings) -> dict[str, Any]:
    """Confirm the configured embedding provider builds and agrees on dimension."""
    if not settings.embedding_enabled:
        return {
            "name": "embedding",
            "status": "ok",
            "message": "embeddings are disabled (EMBEDDING_ENABLED=false)",
        }
    try:
        provider = build_embedding_provider(settings)
    except Exception as exc:
        return {
            "name": "embedding",
            "status": "error",
            "message": f"embedding provider {settings.embedding_provider} is unavailable: {exc}",
            "action": "Configure the missing provider credentials or set EMBEDDING_PROVIDER=hash for the offline default.",
        }
    if provider is None:
        return {
            "name": "embedding",
            "status": "ok",
            "message": "embeddings are disabled (EMBEDDING_ENABLED=false)",
        }
    configured = settings.embedding_dimension
    probed: int | None = None
    if isinstance(provider, HashEmbeddingProvider):
        probed = len(provider.embed("zaxy doctor embedding probe"))
    actual = probed if probed is not None else provider.dimension
    if actual != configured:
        return {
            "name": "embedding",
            "status": "error",
            "message": (
                f"embedding provider {settings.embedding_provider} returns dimension "
                f"{actual} but EMBEDDING_DIMENSION={configured}"
            ),
            "details": {"configured_dimension": configured, "provider_dimension": actual},
            "action": "Set EMBEDDING_DIMENSION to the provider's actual vector size before projecting new vectors.",
        }
    suffix = "" if probed is not None else " (dimension not probed for hosted providers)"
    return {
        "name": "embedding",
        "status": "ok",
        "message": f"{settings.embedding_provider} embeddings available at dimension {configured}{suffix}",
        "details": {"configured_dimension": configured, "provider_dimension": actual},
    }


def _check_vector_cache_budget(settings: Settings) -> dict[str, Any]:
    """Report in-process vector index cache headroom at the configured dimension."""
    if not settings.embedding_enabled:
        return {
            "name": "vector_cache",
            "status": "ok",
            "message": "not applicable while embeddings are disabled",
        }
    bytes_per_vector = settings.embedding_dimension * 8
    budget_capacity = VECTOR_INDEX_CACHE_MAX_BYTES // bytes_per_vector
    details = {
        "cache_max_bytes": VECTOR_INDEX_CACHE_MAX_BYTES,
        "cache_max_entries": VECTOR_INDEX_CACHE_MAX_ENTRIES,
        "bytes_per_vector": bytes_per_vector,
        "budget_vector_capacity": budget_capacity,
    }
    if budget_capacity < VECTOR_CACHE_MIN_VECTOR_HEADROOM:
        return {
            "name": "vector_cache",
            "status": "warning",
            "message": (
                f"vector index cache budget fits only {budget_capacity} float64 vectors "
                f"at EMBEDDING_DIMENSION={settings.embedding_dimension}"
            ),
            "details": details,
            "action": "Lower EMBEDDING_DIMENSION so per-session vector indexes fit the in-process cache budget.",
        }
    return {
        "name": "vector_cache",
        "status": "ok",
        "message": (
            f"vector index cache budget holds about {budget_capacity} vectors "
            f"across up to {VECTOR_INDEX_CACHE_MAX_ENTRIES} cached sessions"
        ),
        "details": details,
    }


def _check_embedding_versions(settings: Settings) -> dict[str, Any]:
    """Report mixed embedding-version corpora in the embedded projection."""
    if not settings.embedding_enabled:
        return {
            "name": "embedding_versions",
            "status": "ok",
            "message": "not applicable while embeddings are disabled",
        }
    backend = settings.projection_backend.casefold().strip()
    if backend != "embedded":
        return {
            "name": "embedding_versions",
            "status": "ok",
            "message": f"embedding versions are not file-checked for projection backend {backend}",
        }
    projection_path = Path(settings.embedded_graph_path)
    if not projection_path.exists():
        return {
            "name": "embedding_versions",
            "status": "ok",
            "message": "no embedded projection yet; nothing to audit",
        }
    from zaxy.retrieval_profile import apply_retrieval_profile, resolve_retrieval_profile

    resolved = apply_retrieval_profile(settings, resolve_retrieval_profile(settings))
    active_tag = active_embedding_version_tag(resolved) or LEGACY_EMBEDDING_VERSION
    try:
        version_counts = _embedded_vector_version_counts(projection_path)
    except Exception as exc:
        return {
            "name": "embedding_versions",
            "status": "ok",
            "message": f"embedded projection not inspectable right now ({exc}); audit skipped",
        }
    total = sum(count for counts in version_counts.values() for count in counts.values())
    if total == 0:
        return {
            "name": "embedding_versions",
            "status": "ok",
            "message": "no projected vectors yet",
        }
    stale_sessions = sorted(
        session
        for session, counts in version_counts.items()
        if any(version != active_tag for version in counts)
    )
    details = {
        "active_version": active_tag,
        "sampled_vectors": total,
        "versions": {
            session: dict(sorted(counts.items()))
            for session, counts in sorted(version_counts.items())
        },
    }
    if stale_sessions:
        remediation = "; ".join(
            f"zaxy memory re-embed --session-id {session} "
            f"--eventloom-path {settings.eventloom_path}"
            for session in stale_sessions
        )
        return {
            "name": "embedding_versions",
            "status": "warning",
            "message": (
                f"projected vectors span embedding versions other than the active "
                f"{active_tag} in sessions: {', '.join(stale_sessions)}; stale-version "
                "vectors are isolated from search until re-embedded"
            ),
            "details": details,
            "action": f"Run {remediation}",
        }
    return {
        "name": "embedding_versions",
        "status": "ok",
        "message": f"all {total} sampled vectors carry the active embedding version {active_tag}",
        "details": details,
    }


# Bounded sample so doctor stays fast on very large embedded projections.
EMBEDDING_VERSION_SAMPLE_LIMIT = 5000


def _embedded_vector_version_counts(
    projection_path: Path,
    *,
    sample_limit: int = EMBEDDING_VERSION_SAMPLE_LIMIT,
) -> dict[str, dict[str, int]]:
    """Sample active embedded Entity rows and count vectors per version tag."""
    import json

    import ladybug

    from zaxy.embedded_graph_store import _is_missing_projection_table_error

    database = ladybug.Database(str(projection_path), read_only=True)
    connection = ladybug.Connection(database)
    try:
        result = connection.execute(
            "MATCH (e:Entity) "
            "WHERE e.valid_to IS NULL AND contains(e.properties_json, '\"embedding\"') "
            f"RETURN e.session_id, e.properties_json LIMIT {int(sample_limit)}"
        )
        rows = cast("list[list[Any]]", cast(Any, result).get_all())
    except RuntimeError as exc:
        if _is_missing_projection_table_error(exc):
            return {}
        raise
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        try:
            properties = json.loads(str(row[1] or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(properties, dict):
            continue
        embedding = properties.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        version = properties.get("embedding_version")
        tag = version if isinstance(version, str) and version else LEGACY_EMBEDDING_VERSION
        session_counts = counts.setdefault(str(row[0]), {})
        session_counts[tag] = session_counts.get(tag, 0) + 1
    return counts


def _check_projection_freshness(settings: Settings) -> dict[str, Any]:
    """Compare embedded projection state against the active event log signature."""
    backend = settings.projection_backend.casefold().strip()
    if backend != "embedded":
        return {
            "name": "projection_freshness",
            "status": "ok",
            "message": f"freshness is not file-checked for projection backend {backend}",
            "action": "Run zaxy status for live projection posture on server backends.",
        }
    log_path = eventlog_path(Path(settings.eventloom_path), settings.eventloom_thread)
    try:
        log_stat = os.stat(log_path)
    except OSError:
        return {
            "name": "projection_freshness",
            "status": "ok",
            "message": "no active event log yet; nothing to project",
        }
    if log_stat.st_size == 0:
        return {
            "name": "projection_freshness",
            "status": "ok",
            "message": "active event log is empty; nothing to project",
        }
    log_signature = (log_stat.st_mtime_ns, log_stat.st_size)
    refresh_action = (
        "Run zaxy memory checkout '<query>' --eventloom-path "
        f"{settings.eventloom_path} --session-id {settings.eventloom_thread} "
        "to refresh the embedded projection."
    )
    projection_path = Path(settings.embedded_graph_path)
    projection_mtime_ns = _newest_mtime_ns(projection_path)
    if projection_mtime_ns is None:
        return {
            "name": "projection_freshness",
            "status": "warning",
            "message": f"embedded projection at {projection_path} has no state for the active event log",
            "details": {"log_mtime_ns": log_signature[0], "log_size": log_signature[1]},
            "action": refresh_action,
        }
    if projection_mtime_ns < log_signature[0]:
        return {
            "name": "projection_freshness",
            "status": "warning",
            "message": "embedded projection state is older than the active event log",
            "details": {
                "log_mtime_ns": log_signature[0],
                "log_size": log_signature[1],
                "projection_mtime_ns": projection_mtime_ns,
            },
            "action": refresh_action,
        }
    return {
        "name": "projection_freshness",
        "status": "ok",
        "message": "embedded projection state is at least as new as the active event log",
        "details": {
            "log_mtime_ns": log_signature[0],
            "log_size": log_signature[1],
            "projection_mtime_ns": projection_mtime_ns,
        },
    }


def _newest_mtime_ns(path: Path) -> int | None:
    """Return the newest st_mtime_ns under a projection file or directory."""
    try:
        stat = path.stat()
    except OSError:
        return None
    newest = stat.st_mtime_ns
    if path.is_dir():
        for child in path.rglob("*"):
            try:
                newest = max(newest, child.stat().st_mtime_ns)
            except OSError:
                continue
    return newest


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


def _check_version_consistency(workspace_root: Path) -> dict[str, Any]:
    """Warn if the imported zaxy package drifts from the repo on disk.

    Delegates to :func:`zaxy.release.check_version_consistency`, which compares
    the repo's declared version against the installed dist version and import
    path. This catches a stale site-packages copy shadowing the source tree —
    the failure that once made tests import the wrong code.

    The doctor's resolved ``workspace_root`` is passed through so the repo walk
    starts from the tree the operator is actually inspecting (e.g. ``zaxy doctor
    <path>``) rather than the process cwd — without it the check silently
    compared against the wrong directory and could report ``ok`` for a drift it
    never looked at.
    """
    from zaxy.release import check_version_consistency

    return check_version_consistency(project_root=workspace_root)


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


def _check_codex_mcp_scope(workspace_root: Path) -> dict[str, Any]:
    """Warn when global Codex MCP config contains repo-specific Zaxy state."""
    config_path = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser() / "config.toml"
    if not config_path.exists():
        return {
            "name": "codex_mcp_scope",
            "status": "ok",
            "message": "no Codex user MCP config found",
        }
    try:
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return {
            "name": "codex_mcp_scope",
            "status": "warning",
            "message": f"Codex user config is invalid TOML: {exc}",
            "action": f"Fix or remove malformed Codex config at {config_path}.",
        }
    zaxy = document.get("mcp_servers", {}).get("zaxy")
    if not isinstance(zaxy, dict):
        return {
            "name": "codex_mcp_scope",
            "status": "ok",
            "message": "Codex user config has no global zaxy MCP server",
        }
    args = [str(arg) for arg in zaxy.get("args", [])] if isinstance(zaxy.get("args", []), list) else []
    env = zaxy.get("env", {})
    env_keys = set(env) if isinstance(env, dict) else set()
    repo_scoped = "--eventloom-path" in args or bool({"EVENTLOOM_PATH", "EVENTLOOM_THREAD", "ZAXY_DOMAIN"} & env_keys)
    if repo_scoped:
        workspace = workspace_root.resolve()
        return {
            "name": "codex_mcp_scope",
            "status": "warning",
            "message": "Codex user-level zaxy MCP config contains repo-specific Eventloom/session state",
            "details": {
                "config": str(config_path),
                "workspace": str(workspace),
            },
            "action": (
                f"Review {config_path}; if you intentionally want Zaxy to replace the global "
                f"server, run zaxy init {workspace} --codex-mcp-install user --force."
            ),
        }
    return {
        "name": "codex_mcp_scope",
        "status": "ok",
        "message": "Codex user-level zaxy MCP config is workspace-neutral",
    }


def _check_agent_instructions(workspace_root: Path) -> dict[str, str]:
    agents = workspace_root / "AGENTS.md"
    if not agents.exists():
        return {
            "name": "agent_instructions",
            "status": "warning",
            "message": "No AGENTS.md activation instructions found",
            "action": f"Run zaxy init {workspace_root} to install the marker-managed Zaxy Memory Activation block.",
        }
    text = agents.read_text(encoding="utf-8", errors="replace")
    if AGENT_ACTIVATION_BEGIN in text and AGENT_ACTIVATION_END in text:
        return {
            "name": "agent_instructions",
            "status": "ok",
            "message": "AGENTS.md contains marker-managed Zaxy Memory Activation instructions",
        }
    return {
        "name": "agent_instructions",
        "status": "warning",
        "message": "AGENTS.md is present but missing the Zaxy Memory Activation block",
        "action": (
            f"Run zaxy init {workspace_root} to add the marker-managed activation block, "
            "or pass --no-agent-instructions if another instruction system owns this."
        ),
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
    readiness = hook_status.get("capture_readiness", {})
    missing = hook_status.get("missing_observation_types", [])
    if not missing:
        return {
            "name": "observation_coverage",
            "status": "ok",
            "message": "high-value automatic observation types have been captured",
            "details": readiness,
        }
    missing_types = ", ".join(str(event_type) for event_type in missing)
    return {
        "name": "observation_coverage",
        "status": "warning",
        "message": f"missing high-value automatic observation types: {missing_types}",
        "details": readiness,
        "action": "Confirm hooks emit command, file-edit, tool-call, and transcript observations for this client.",
    }


def _check_capture_health(hook_status: dict[str, Any]) -> dict[str, Any]:
    readiness = hook_status.get("capture_readiness", {})
    status = str(readiness.get("status", "warning"))
    clients = hook_status.get("clients", {})
    codex = clients.get("codex") if isinstance(clients, dict) else None
    codex_runtime = codex.get("runtime") if isinstance(codex, dict) else None
    codex_configured_stopped = (
        isinstance(codex, dict)
        and bool(codex.get("installed", False))
        and isinstance(codex_runtime, dict)
        and not bool(codex_runtime.get("running", False))
    )
    message = str(readiness.get("message", "0 of 4 high-value automatic capture lanes are active")).replace(
        "high-value automatic capture lanes",
        "high-value lanes",
    )
    if status == "ok":
        return {
            "name": "capture_health",
            "status": "ok",
            "message": f"automatic capture is healthy: {message}",
            "details": readiness,
        }
    actions = [str(action) for action in readiness.get("actions", [])]
    check = {
        "name": "capture_health",
        "status": "warning",
        "message": (
            "Codex capture is configured, but the managed watcher is not running"
            if codex_configured_stopped
            else f"automatic capture is incomplete: {message}"
        ),
        "details": readiness,
    }
    if actions:
        check["action"] = " ".join(actions)
    else:
        check["action"] = "Run zaxy hook-status --eventloom-path .eventloom to inspect automatic capture coverage."
    return check


def _check_memory_activation(hook_status: dict[str, Any]) -> dict[str, Any]:
    activation = hook_status.get("memory_activation", {})
    status = str(activation.get("status", "warning"))
    message = str(activation.get("message", "Memory activation status is unavailable"))
    check: dict[str, Any] = {
        "name": "memory_activation",
        "status": status,
        "message": message,
        "details": activation,
    }
    if status != "ok":
        remediations = activation.get("remediations", [])
        if isinstance(remediations, list) and remediations:
            first = remediations[0]
            if isinstance(first, dict) and first.get("command"):
                check["action"] = str(first["command"])
                return check
        actions = activation.get("actions", [])
        if isinstance(actions, list) and actions:
            check["action"] = " ".join(str(action) for action in actions)
        else:
            check["action"] = "Run zaxy hook-status --eventloom-path .eventloom to inspect memory activation."
    return check


def _check_packet_memory(settings: Settings) -> dict[str, str]:
    report = packet_memory_report(
        eventloom_path=Path(settings.eventloom_path),
        session_id=settings.eventloom_thread,
    )
    details = report["details"]
    if details["captured"] == 0:
        return {
            "name": "packet_memory",
            "status": "warning",
            "message": report["message"],
            "details": details,
            "action": (
                "Optional: run zaxy packet-analyzer --eventloom-path "
                f"{settings.eventloom_path} --session-id {settings.eventloom_thread} "
                "--upstream-base-url <provider-v1-url>, then run zaxy packet-project --watch."
            ),
        }
    if details["unprojected"]:
        return {
            "name": "packet_memory",
            "status": "warning",
            "message": report["message"],
            "details": details,
            "action": (
                "Run zaxy packet-project --watch --eventloom-path "
                f"{settings.eventloom_path} --session-id {settings.eventloom_thread}."
            ),
        }
    return {
        "name": "packet_memory",
        "status": "ok",
        "message": report["message"],
        "details": details,
    }


def packet_memory_report(
    *,
    eventloom_path: str | Path,
    session_id: str,
    analyzer_host: str = "127.0.0.1",
    analyzer_port: int = 8787,
) -> dict[str, Any]:
    """Return packet-memory pipeline status for one Eventloom session."""
    log = EventLog(eventlog_path(Path(eventloom_path), session_id))
    status = _packet_memory_status(log.read_all())
    details = status["details"]
    guidance = build_packet_capture_guidance(
        eventloom_path=eventloom_path,
        session_id=session_id,
        host=analyzer_host,
        port=analyzer_port,
    )
    if details["captured"] == 0:
        message = "no LLM packet captures observed for this session"
        state = "warning"
    elif details["unprojected"]:
        count = details["unprojected"]
        noun = "event has" if count == 1 else "events have"
        message = f"{count} captured packet {noun} not been projected"
        state = "warning"
    else:
        count = details["captured"]
        noun = "capture has" if count == 1 else "captures have"
        message = f"{count} packet {noun} projected memory"
        state = "ok"
    return {
        "status": state,
        "session_id": session_id,
        "message": message,
        "details": details,
        "capture": {
            "analyzer_host": analyzer_host,
            "analyzer_port": analyzer_port,
            "analyzer_listening": _tcp_port_open(analyzer_host, analyzer_port),
            "client_base_url": guidance.client_base_url,
        },
        "next_steps": guidance.next_steps() if details["captured"] == 0 else [],
    }


def format_packet_memory_report(report: dict[str, Any]) -> str:
    """Format a packet-memory report for operators."""
    details = report["details"]
    rendered = " ".join(f"{key}={value}" for key, value in details.items())
    lines = [
        f"Zaxy packet memory: {report['status']}",
        f"session: {report['session_id']}",
        f"message: {report['message']}",
        f"details: {rendered}",
    ]
    if report.get("capture"):
        capture = report["capture"]
        state = "listening" if capture["analyzer_listening"] else "inactive"
        lines.append(f"analyzer: {state} ({capture['client_base_url']})")
    if report.get("next_steps"):
        lines.append("Next:")
        lines.extend(f"- {step}" for step in report["next_steps"])
    return "\n".join(lines)


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


def _tcp_port_open(host: str, port: int) -> bool:
    """Return whether a local TCP listener accepts connections."""
    try:
        with socket.create_connection((host, port), timeout=0.05):
            return True
    except OSError:
        return False


def _check_projection_backend(settings: Settings) -> dict[str, str]:
    backend = settings.projection_backend.casefold().strip()
    if backend == "embedded":
        check = LocalEmbeddedGraphRuntime(settings.embedded_graph_path).check()
        return {
            "name": "embedded_graph",
            "status": check.status,
            "message": check.message,
        }
    if backend == "pggraph":
        check = LocalPgGraphRuntime(
            settings.pggraph_dsn,
            enabled=settings.pggraph_auto_start and settings.zaxy_env.lower() != "production",
            image=settings.pggraph_auto_start_image,
            container_name=settings.pggraph_auto_start_container,
            pggraph_repo=settings.pggraph_repo,
        ).check()
        return {
            "name": "pggraph",
            "status": check.status,
            "message": check.message,
            "action": "Run zaxy status --projection-backend pggraph for live runtime posture.",
        }
    return _check_neo4j(settings)


def _check_embedded_mcp_runtime(settings: Settings, *, repair: bool = False) -> dict[str, Any]:
    backend = settings.projection_backend.casefold().strip()
    if backend != "embedded":
        return {
            "name": "embedded_mcp_runtime",
            "status": "ok",
            "message": f"not applicable for projection backend {backend}",
        }
    report = EmbeddedMcpRuntimeCoordinator.from_embedded_graph_path(
        settings.embedded_graph_path
    ).repair_stale_runtime(reap=repair, expected_graph_path=settings.embedded_graph_path)
    check = {
        "name": "embedded_mcp_runtime",
        "status": report["status"],
        "message": report["message"],
        "details": {
            "repaired": report["repaired"],
            "owner_active": report["owner_active"],
            "owner_path": report["owner_path"],
            "socket_path": report["socket_path"],
        },
    }
    if "action" in report:
        check["action"] = report["action"]
    return check


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


def _check_projection_backup_artifacts(settings: Settings) -> dict[str, Any]:
    """Report leftover pre-LadybugDB projection backups from the 2.3 engine swap.

    When the store first opens a projection written by the pre-fork Kuzu
    engine, the unreadable file is moved aside to ``<path>.pre-ladybug.bak``
    (never deleted) and the projection is rebuilt from the Eventloom log.
    The backup is purely a safety net; once the rebuilt projection is
    verified, it is dead weight worth flagging.
    """
    from zaxy.embedded_graph_store import pre_ladybug_backup_paths

    backups = pre_ladybug_backup_paths(Path(settings.embedded_graph_path))
    if not backups:
        return {
            "name": "projection_backup_artifacts",
            "status": "ok",
            "message": "no pre-LadybugDB projection backups present",
        }
    rendered = ", ".join(str(path) for path in backups)
    return {
        "name": "projection_backup_artifacts",
        "status": "warning",
        "message": (
            f"pre-LadybugDB projection backup present: {rendered}; the active projection "
            "was rebuilt for the LadybugDB storage format and the backup is no longer read"
        ),
        "action": (
            "Verify the rebuilt projection (zaxy status, or zaxy reproject to replay full "
            "history), then delete the .pre-ladybug.bak file(s) to reclaim the space."
        ),
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
