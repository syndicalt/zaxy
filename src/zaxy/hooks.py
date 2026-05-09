"""Observer hook helpers for client lifecycle capture."""

from __future__ import annotations

import json
import shlex
from typing import Any, Literal

from zaxy.domain import derive_domain, domain_default_session, slug_domain

HookClient = Literal["claude-code", "codex", "generic"]


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


def hook_event_type(trigger: str) -> str:
    """Return the normalized Eventloom type for a hook trigger."""
    normalized = trigger.casefold().strip().replace("_", "-")
    event_types = {
        "session-start": "hook.session_started",
        "start": "hook.session_started",
        "stop": "hook.stop",
        "precompact": "hook.precompact",
        "checkpoint": "hook.checkpoint",
    }
    try:
        return event_types[normalized]
    except KeyError as exc:
        raise ValueError("hook trigger must be one of: session-start, stop, precompact, checkpoint") from exc


def build_hook_payload(
    *,
    trigger: str,
    source: str,
    workspace: str | None = None,
    transcript_path: str | None = None,
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
    return payload


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
