"""Observer hook helpers for client lifecycle capture."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Literal

from zaxy.domain import derive_domain, domain_default_session, slug_domain
from zaxy.event import Event, EventLog

HookClient = Literal["claude-code", "codex", "generic"]
HOOK_CLIENTS = ("claude-code", "codex", "generic")


def render_hook_config(
    client: HookClient | str,
    *,
    eventloom_path: str = ".eventloom",
    domain: str | None = None,
    source: str | None = None,
) -> str:
    """Render copyable hook adapter config for a client."""
    normalized = _normalize_client(client)
    resolved_domain = slug_domain(domain) if domain else derive_domain()
    session_id = domain_default_session(resolved_domain)
    hook_source = source or normalized
    if normalized == "claude-code":
        return json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "matcher": "*",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": _hook_command(
                                        "stop",
                                        eventloom_path=eventloom_path,
                                        session_id=session_id,
                                        source=hook_source,
                                    ),
                                }
                            ],
                        }
                    ],
                    "PreCompact": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": _hook_command(
                                        "precompact",
                                        eventloom_path=eventloom_path,
                                        session_id=session_id,
                                        source=hook_source,
                                    ),
                                }
                            ],
                        }
                    ],
                }
            },
            indent=2,
            sort_keys=True,
        )
    return "\n".join(
        [
            "# Zaxy observer hook commands",
            _hook_command("session-start", eventloom_path=eventloom_path, session_id=session_id, source=hook_source),
            _hook_command("stop", eventloom_path=eventloom_path, session_id=session_id, source=hook_source),
            _hook_command("precompact", eventloom_path=eventloom_path, session_id=session_id, source=hook_source),
            "",
        ]
    )


def write_hook_config(
    path: str | Path,
    content: str,
    *,
    force: bool = False,
) -> Path:
    """Write hook config to disk without overwriting unless forced."""
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def hook_event_type(trigger: str) -> str:
    """Return the normalized Eventloom type for a hook trigger."""
    normalized = trigger.casefold().strip().replace("_", "-")
    event_types = {
        "session-start": "hook.session_started",
        "start": "hook.session_started",
        "stop": "hook.stop",
        "precompact": "hook.precompact",
        "checkpoint": "hook.checkpoint",
        "heartbeat": "hook.heartbeat",
    }
    try:
        return event_types[normalized]
    except KeyError as exc:
        raise ValueError("hook trigger must be one of: session-start, stop, precompact, checkpoint, heartbeat") from exc


def build_hook_payload(
    *,
    trigger: str,
    source: str,
    workspace: str | None = None,
    transcript_path: str | None = None,
    summary: str | None = None,
    reason: str | None = None,
    turn_count: int | None = None,
) -> dict[str, Any]:
    """Build a compact, non-blocking lifecycle payload for hook adapters."""
    payload: dict[str, Any] = {
        "trigger": trigger.casefold().strip().replace("_", "-"),
        "source": source,
    }
    if workspace:
        payload["workspace"] = workspace
    if transcript_path:
        payload["transcript_path"] = transcript_path
    if summary:
        payload["summary"] = summary
    if reason:
        payload["reason"] = reason
    if turn_count is not None:
        payload["turn_count"] = turn_count
    return payload


def inspect_hook_status(
    *,
    eventloom_path: str | Path = ".eventloom",
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect installed hook configs and recent Eventloom lifecycle activity."""
    root = Path(workspace_root or Path.cwd())
    eventloom = Path(eventloom_path)
    installations = _detect_hook_installations(root)
    latest = _latest_hook_event(eventloom)
    installed_any = any(client["installed"] for client in installations.values())
    status = "ok" if latest is not None else "warning"
    if not installed_any and latest is None:
        message = "No installed observer hook config or hook lifecycle events found"
    elif latest is None:
        message = "Observer hook config is installed, but no hook lifecycle events have been observed"
    else:
        message = f"Latest hook event is {latest['type']} in {latest['thread']} at {latest['timestamp']}"
    return {
        "status": status,
        "message": message,
        "eventloom_path": str(eventloom),
        "clients": installations,
        "latest_event": latest,
    }


def format_hook_status(report: dict[str, Any]) -> str:
    """Format hook status for humans."""
    lines = [f"Zaxy hooks: {report['status']}", f"- activity: {report['message']}"]
    for client in HOOK_CLIENTS:
        info = report["clients"][client]
        installed = "installed" if info["installed"] else "not installed"
        suffix = f" ({', '.join(info['paths'])})" if info["paths"] else ""
        lines.append(f"- {client}: {installed}{suffix}")
    latest = report.get("latest_event")
    if latest:
        lines.append(
            f"- last event: {latest['type']} seq={latest['seq']} "
            f"session={latest['thread']} source={latest['source']}"
        )
    return "\n".join(lines)


def _detect_hook_installations(workspace_root: Path) -> dict[str, dict[str, Any]]:
    candidates = {
        "claude-code": [
            workspace_root / ".claude" / "settings.local.json",
            workspace_root / ".claude" / "settings.json",
        ],
        "codex": [workspace_root / ".codex" / "hooks.json"],
        "generic": [],
    }
    installations: dict[str, dict[str, Any]] = {}
    for client, paths in candidates.items():
        installed = [
            str(path.relative_to(workspace_root))
            for path in paths
            if _looks_like_zaxy_hook_config(path)
        ]
        installations[client] = {
            "installed": bool(installed),
            "paths": installed,
        }
    return installations


def _latest_hook_event(eventloom_path: Path) -> dict[str, Any] | None:
    latest: Event | None = None
    if eventloom_path.is_file():
        paths = [eventloom_path]
    elif eventloom_path.is_dir():
        paths = sorted(eventloom_path.glob("*.jsonl"))
    else:
        paths = []
    for path in paths:
        try:
            events = EventLog(path).read_all()
        except Exception:
            continue
        for event in events:
            if not event.type.startswith("hook."):
                continue
            if latest is None or event.timestamp > latest.timestamp or (
                event.timestamp == latest.timestamp and event.seq > latest.seq
            ):
                latest = event
    if latest is None:
        return None
    return {
        "seq": latest.seq,
        "timestamp": latest.timestamp,
        "type": latest.type,
        "thread": latest.thread,
        "source": latest.payload.get("source", "unknown"),
        "trigger": latest.payload.get("trigger", latest.type.removeprefix("hook.")),
    }


def _looks_like_zaxy_hook_config(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "zaxy hook-event" in text


def _hook_command(
    trigger: str,
    *,
    eventloom_path: str,
    session_id: str,
    source: str,
) -> str:
    return " ".join(
        [
            "zaxy",
            "hook-event",
            shlex.quote(trigger),
            "--eventloom-path",
            shlex.quote(eventloom_path),
            "--session-id",
            shlex.quote(session_id),
            "--source",
            shlex.quote(source),
        ]
    )


def _normalize_client(client: str) -> str:
    normalized = client.casefold().strip().replace("_", "-")
    if normalized in {"claude", "claude-code"}:
        return "claude-code"
    if normalized in {"codex", "generic"}:
        return normalized
    raise ValueError("hook client must be one of: claude-code, codex, generic")
