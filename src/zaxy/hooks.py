"""Observer hook helpers for client lifecycle capture."""

from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from zaxy.domain import derive_domain, domain_default_session, slug_domain
from zaxy.event import Event, EventLog

HookClient = Literal["claude-code", "codex", "generic"]
HOOK_CLIENTS = ("claude-code", "codex", "generic")
OBSERVATION_COVERAGE_TYPES = (
    "hook",
    "command.completed",
    "file.edit.applied",
    "tool.call.completed",
    "transcript.turn",
)
HIGH_VALUE_OBSERVATION_TYPES = (
    "command.completed",
    "file.edit.applied",
    "tool.call.completed",
    "transcript.turn",
)


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
            _hook_command("resume", eventloom_path=eventloom_path, session_id=session_id, source=hook_source),
            _hook_command("stop", eventloom_path=eventloom_path, session_id=session_id, source=hook_source),
            _hook_command("precompact", eventloom_path=eventloom_path, session_id=session_id, source=hook_source),
            "# Optional first-class observation sinks for richer automatic capture",
            f"# {_hook_command('command', eventloom_path=eventloom_path, session_id=session_id, source=hook_source)} "
            "--command '<cmd>' --exit-code 0",
            f"# {_hook_command('file-edit', eventloom_path=eventloom_path, session_id=session_id, source=hook_source)} "
            "--path '<path>' --operation modified",
            f"# {_hook_command('tool-call', eventloom_path=eventloom_path, session_id=session_id, source=hook_source)} "
            "--tool-name '<tool>' --tool-status ok",
            f"# {_hook_command('transcript-turn', eventloom_path=eventloom_path, session_id=session_id, source=hook_source)} "
            "--role assistant --content '<turn>'",
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


def write_claude_code_hook_config(
    path: str | Path,
    content: str,
    *,
    force: bool = False,
) -> Path:
    """Merge Claude Code hook settings without disturbing unrelated settings."""
    target = Path(path)
    if not target.exists():
        return write_hook_config(target, content, force=force)
    generated = _parse_json_object(content, source="generated Claude Code hook config")
    existing = _parse_json_object(target.read_text(encoding="utf-8"), source=str(target))
    if _contains_zaxy_hook_command(existing) and not force:
        raise FileExistsError(f"{target} already contains Zaxy hook handlers; pass --force to replace them")
    if force:
        _remove_zaxy_hook_handlers(existing)
    _merge_claude_hook_settings(existing, generated, path=target)
    target.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def hook_event_type(trigger: str) -> str:
    """Return the normalized Eventloom type for a hook trigger."""
    normalized = trigger.casefold().strip().replace("_", "-")
    event_types = {
        "session-start": "hook.session_started",
        "start": "hook.session_started",
        "resume": "hook.resumed",
        "stop": "hook.stop",
        "precompact": "hook.precompact",
        "checkpoint": "hook.checkpoint",
        "heartbeat": "hook.heartbeat",
    }
    try:
        return event_types[normalized]
    except KeyError as exc:
        raise ValueError(
            "hook trigger must be one of: session-start, resume, stop, precompact, checkpoint, heartbeat"
        ) from exc


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
    max_checkout_stale_minutes: int = 120,
    session_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Inspect installed hook configs and recent Eventloom lifecycle activity."""
    root = Path(workspace_root or Path.cwd())
    eventloom = Path(eventloom_path)
    installations = _detect_hook_installations(root)
    installations["codex"]["runtime"] = _detect_codex_capture_runtime(root, eventloom)
    latest = _latest_hook_event(eventloom)
    coverage = _observation_coverage(eventloom)
    missing = [event_type for event_type in HIGH_VALUE_OBSERVATION_TYPES if coverage[event_type]["count"] == 0]
    readiness = _capture_readiness(coverage, installations=installations, workspace_root=root, eventloom_path=eventloom)
    activation = _memory_activation(
        eventloom,
        stale_after_minutes=max_checkout_stale_minutes,
        preferred_session_id=session_id,
        now=now,
    )
    installed_any = any(client["installed"] for client in installations.values())
    status = "ok" if latest is not None else "warning"
    codex_runtime = installations["codex"].get("runtime", {})
    if installations["codex"]["installed"] and not codex_runtime.get("running", False):
        status = "warning"
    if activation["status"] != "ok":
        status = "warning"
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
        "observation_coverage": coverage,
        "missing_observation_types": missing,
        "capture_readiness": readiness,
        "memory_activation": activation,
    }


def inspect_memory_activation(
    *,
    eventloom_path: str | Path = ".eventloom",
    max_checkout_stale_minutes: int = 120,
    session_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Inspect whether memory checkout is actively being used."""
    return _memory_activation(
        Path(eventloom_path),
        stale_after_minutes=max_checkout_stale_minutes,
        preferred_session_id=session_id,
        now=now,
    )


def format_hook_status(report: dict[str, Any]) -> str:
    """Format hook status for humans."""
    lines = [
        f"Zaxy hooks: {report['status']}",
        "",
        "Summary",
        f"  {report['message']}",
        f"  Eventloom: {report['eventloom_path']}",
        "",
        "Client setup",
    ]
    for client in HOOK_CLIENTS:
        info = report["clients"][client]
        installed = "installed" if info["installed"] else "missing"
        suffix = f" ({', '.join(info['paths'])})" if info["paths"] else ""
        lines.append(f"  {_client_setup_label(client)}: {installed}{suffix}")
        runtime = info.get("runtime")
        if runtime:
            if runtime.get("running"):
                pids = ", ".join(str(pid) for pid in runtime.get("pids", []))
                lines.append(f"  Codex capture watcher: running pid={pids}")
            else:
                lines.append("  Codex capture watcher: not running")
    latest = report.get("latest_event")
    if latest:
        lines.extend(
            [
                "",
                "Last observed event",
                f"  type: {latest['type']}",
                f"  seq: {latest['seq']}",
                f"  session: {latest['thread']}",
                f"  source: {latest['source']}",
            ]
        )
    readiness = report.get("capture_readiness")
    if readiness:
        active_count = len(readiness.get("active_observation_types", []))
        total = len(HIGH_VALUE_OBSERVATION_TYPES)
        lines.extend(
            [
                "",
                "Capture readiness",
                f"  status: {readiness['status']}",
                f"  active lanes: {active_count} of {total}",
            ]
        )
        actions = readiness.get("actions", [])
        if actions:
            lines.extend(["", "Next steps"])
            for index, action in enumerate(actions, start=1):
                lines.extend(_format_next_step(index, str(action)))
    activation = report.get("memory_activation")
    if activation:
        lines.extend(
            [
                "",
                "Memory activation",
                f"  status: {activation['status']}",
                f"  {activation['message']}",
                f"  stale after: {activation['stale_after_minutes']} minutes",
            ]
        )
        latest_checkout = activation.get("latest_checkout")
        if latest_checkout:
            lines.append(
                f"  latest checkout: seq={latest_checkout['seq']} "
                f"session={latest_checkout['thread']} at {latest_checkout['timestamp']}"
            )
            token_efficiency = latest_checkout.get("token_efficiency")
            if isinstance(token_efficiency, dict):
                prompt_tokens = token_efficiency.get("prompt_tokens")
                facts_per_1k = token_efficiency.get("facts_per_1k_prompt_tokens")
                if isinstance(prompt_tokens, int | float) and isinstance(facts_per_1k, int | float):
                    lines.append(
                        f"  checkout tokens: {int(prompt_tokens)} prompt, "
                        f"{float(facts_per_1k):.1f} facts/1k prompt tokens"
                    )
        latest_capture = activation.get("latest_capture")
        if latest_capture:
            lines.append(
                f"  latest capture: {latest_capture['type']} seq={latest_capture['seq']} "
                f"session={latest_capture['thread']} source={latest_capture['source']}"
            )
        latest_reminder = activation.get("latest_reminder")
        if latest_reminder:
            lines.append(
                f"  latest reminder: seq={latest_reminder['seq']} "
                f"session={latest_reminder['thread']} at {latest_reminder['timestamp']}"
            )
        efficiency = activation.get("activation_efficiency")
        if efficiency:
            rate = efficiency.get("fresh_checkout_rate")
            fresh = int(efficiency.get("fresh_checkout_session_count", 0))
            total = int(efficiency.get("high_context_session_count", 0))
            rate_label = "-" if rate is None else f"{float(rate) * 100:.1f}%"
            lines.append(f"  activation efficiency: {rate_label} ({fresh}/{total} high-context sessions)")
        actions = activation.get("actions", [])
        if actions:
            lines.extend(["", "Memory next steps"])
            for index, action in enumerate(actions, start=1):
                lines.append(f"  {index}. {action}")
        remediations = activation.get("remediations", [])
        if remediations:
            if not actions:
                lines.extend(["", "Memory next steps"])
            start_index = len(actions) + 1
            for index, remediation in enumerate(remediations, start=start_index):
                if not isinstance(remediation, dict):
                    continue
                message = remediation.get("message")
                command = remediation.get("command")
                if message:
                    lines.append(f"  {index}. {message}")
                if command:
                    lines.append(f"     {command}")
    coverage = report.get("observation_coverage", {})
    if coverage:
        lines.extend(["", "Observation coverage"])
        for event_type in OBSERVATION_COVERAGE_TYPES:
            entry = coverage.get(event_type, {})
            count = entry.get("count", 0)
            latest_observation = entry.get("latest")
            label = "hook.*" if event_type == "hook" else event_type
            if latest_observation:
                lines.append(
                    f"  [x] {label} count={count} latest={latest_observation['type']} "
                    f"seq={latest_observation['seq']} session={latest_observation['thread']} "
                    f"source={latest_observation['source']}"
                )
            else:
                lines.append(f"  [ ] {label}")
    return "\n".join(lines)


def _client_setup_label(client: str) -> str:
    labels = {
        "claude-code": "Claude Code hook config",
        "codex": "Codex capture config",
        "generic": "Generic hook config",
    }
    return labels.get(client, f"{client} hook config")


def _format_next_step(index: int, action: str) -> list[str]:
    wire_prefix = "Wire hooks or adapter sinks for: "
    start_prefix = "Start managed deterministic Codex capture: "
    if action.startswith(wire_prefix):
        lanes = [
            lane.strip().rstrip(".")
            for lane in action.removeprefix(wire_prefix).split(",")
            if lane.strip()
        ]
        lines = [f"  {index}. Wire hooks or adapter sinks for missing lanes:"]
        lines.extend(f"     - {lane}" for lane in lanes)
        return lines
    if action.startswith(start_prefix):
        command = action.removeprefix(start_prefix).rstrip(".")
        return [f"  {index}. Start managed deterministic Codex capture:", f"     {command}"]
    return [f"  {index}. {action}"]


def _capture_readiness(
    coverage: dict[str, dict[str, Any]],
    *,
    installations: dict[str, dict[str, Any]] | None = None,
    workspace_root: Path | None = None,
    eventloom_path: Path | None = None,
) -> dict[str, Any]:
    active = [
        event_type
        for event_type in HIGH_VALUE_OBSERVATION_TYPES
        if coverage[event_type]["count"] > 0
    ]
    missing = [
        event_type
        for event_type in HIGH_VALUE_OBSERVATION_TYPES
        if coverage[event_type]["count"] == 0
    ]
    total = len(HIGH_VALUE_OBSERVATION_TYPES)
    active_count = len(active)
    status = "ok" if not missing else "warning"
    actions = []
    if missing:
        actions.append("Wire hooks or adapter sinks for: " + ", ".join(missing) + ".")
    codex = installations.get("codex") if installations else None
    codex_runtime = codex.get("runtime") if codex else None
    if codex and codex.get("installed") and codex_runtime and not codex_runtime.get("running"):
        status = "warning"
        if workspace_root is not None and eventloom_path is not None:
            actions.append(_codex_capture_start_action(workspace_root, eventloom_path))
    return {
        "status": status,
        "message": f"{active_count} of {total} high-value automatic capture lanes are active",
        "active_observation_types": active,
        "missing_observation_types": missing,
        "actions": actions,
    }


def _codex_capture_start_action(workspace_root: Path, _eventloom_path: Path) -> str:
    return (
        f"Start managed deterministic Codex capture: zaxy capture start --workspace {workspace_root}."
    )


def _detect_hook_installations(workspace_root: Path) -> dict[str, dict[str, Any]]:
    candidates = {
        "claude-code": [
            workspace_root / ".claude" / "settings.local.json",
            workspace_root / ".claude" / "settings.json",
        ],
        "codex": [
            workspace_root / ".codex" / "zaxy-capture.json",
            workspace_root / ".codex" / "hooks.json",
        ],
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


def detect_codex_capture_runtime(workspace_root: Path, eventloom_path: Path) -> dict[str, Any]:
    """Detect whether a Codex capture watcher is running for this workspace."""
    return _detect_codex_capture_runtime(workspace_root, eventloom_path)


def _detect_codex_capture_runtime(workspace_root: Path, eventloom_path: Path) -> dict[str, Any]:
    expected_workspace = workspace_root.resolve()
    expected_eventloom = _resolve_against(eventloom_path, expected_workspace)
    if expected_eventloom.suffix == ".jsonl":
        expected_eventloom = expected_eventloom.parent
    pids = [
        pid
        for pid, cmdline in _iter_process_cmdlines()
        if _is_matching_codex_capture_process(
            cmdline,
            workspace_root=expected_workspace,
            eventloom_path=expected_eventloom,
            process_cwd=_process_cwd(pid),
        )
    ]
    if pids:
        return {
            "running": True,
            "pids": sorted(pids),
            "message": "Codex capture watcher is running",
        }
    return {
        "running": False,
        "pids": [],
        "message": "Codex capture watcher is not running",
    }


def _iter_process_cmdlines() -> list[tuple[int, list[str]]]:
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    processes: list[tuple[int, list[str]]] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        cmdline = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if cmdline:
            processes.append((int(entry.name), cmdline))
    return processes


def _is_matching_codex_capture_process(
    cmdline: list[str],
    *,
    workspace_root: Path,
    eventloom_path: Path,
    process_cwd: Path | None = None,
) -> bool:
    if "codex-capture" not in cmdline or "--watch" not in cmdline:
        return False
    base = process_cwd or Path.cwd()
    process_workspace = _option_path(cmdline, "--workspace", default=base, base=base)
    process_eventloom = _option_path(
        cmdline,
        "--eventloom-path",
        default=process_workspace / ".eventloom",
        base=process_workspace,
    )
    return process_workspace == workspace_root and process_eventloom == eventloom_path


def _process_cwd(pid: int) -> Path | None:
    try:
        return (Path("/proc") / str(pid) / "cwd").resolve()
    except OSError:
        return None


def _option_path(cmdline: list[str], option: str, *, default: Path, base: Path | None = None) -> Path:
    try:
        value = cmdline[cmdline.index(option) + 1]
    except (ValueError, IndexError):
        return default.resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base or Path.cwd()) / path).resolve()


def _resolve_against(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _latest_hook_event(eventloom_path: Path) -> dict[str, Any] | None:
    latest: Event | None = None
    for path in _eventlog_paths(eventloom_path):
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
        "hash": latest.hash,
        "timestamp": latest.timestamp,
        "type": latest.type,
        "thread": latest.thread,
        "source": latest.payload.get("source", "unknown"),
        "trigger": latest.payload.get("trigger", latest.type.removeprefix("hook.")),
    }


def _observation_coverage(eventloom_path: Path) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {
        event_type: {"count": 0, "latest": None} for event_type in OBSERVATION_COVERAGE_TYPES
    }
    for path in _eventlog_paths(eventloom_path):
        try:
            events = EventLog(path).read_all()
        except Exception:
            continue
        for event in events:
            event_type = _observation_coverage_type(event.type)
            if event_type is None:
                continue
            entry = coverage[event_type]
            entry["count"] += 1
            latest = entry["latest"]
            if latest is None or _event_is_newer(event, latest):
                entry["latest"] = _summarize_observation_event(event)
    return coverage


def _memory_activation(
    eventloom_path: Path,
    *,
    stale_after_minutes: int,
    preferred_session_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    latest_checkout: dict[str, Any] | None = None
    latest_capture: dict[str, Any] | None = None
    latest_reminder: dict[str, Any] | None = None
    activation_efficiency = _activation_efficiency(
        eventloom_path,
        stale_after_minutes=stale_after_minutes,
    )
    for path in _eventlog_paths(eventloom_path):
        try:
            events = EventLog(path).read_all()
        except Exception:
            continue
        for event in events:
            if event.type == "memory.checkout.completed":
                summary = _summarize_observation_event(event)
                if latest_checkout is None or _event_is_newer(event, latest_checkout):
                    latest_checkout = summary
            if _observation_coverage_type(event.type) in HIGH_VALUE_OBSERVATION_TYPES:
                summary = _summarize_observation_event(event)
                if latest_capture is None or _event_is_newer(event, latest_capture):
                    latest_capture = summary
            if event.type == "memory.reminder.suggested":
                summary = _summarize_observation_event(event)
                if latest_reminder is None or _event_is_newer(event, latest_reminder):
                    latest_reminder = summary

    actions: list[str] = []
    if latest_checkout is None:
        actions.append("Run memory checkout before relying on Zaxy context.")
        remediations = [
            _checkout_remediation(
                code="missing_checkout",
                eventloom_path=eventloom_path,
                session_id=preferred_session_id or _activation_session_id(latest_capture, latest_reminder),
            )
        ]
        return {
            "status": "warning",
            "message": "No memory checkout events found",
            "stale_after_minutes": stale_after_minutes,
            "latest_checkout": None,
            "latest_capture": latest_capture,
            "latest_reminder": latest_reminder,
            "activation_efficiency": activation_efficiency,
            "actions": actions,
            "remediations": remediations,
        }

    reference_time = now or datetime.now(UTC)
    checkout_time = _parse_event_timestamp(str(latest_checkout["timestamp"]))
    age_seconds = (reference_time - checkout_time).total_seconds()
    if age_seconds > stale_after_minutes * 60:
        actions.append("Run memory checkout before relying on Zaxy context.")
        remediations = [
            _checkout_remediation(
                code="stale_checkout",
                eventloom_path=eventloom_path,
                session_id=str(latest_checkout["thread"]),
            )
        ]
        return {
            "status": "warning",
            "message": "Latest memory checkout is stale",
            "stale_after_minutes": stale_after_minutes,
            "latest_checkout": latest_checkout,
            "latest_capture": latest_capture,
            "latest_reminder": latest_reminder,
            "activation_efficiency": activation_efficiency,
            "actions": actions,
            "remediations": remediations,
        }
    stale_sessions = int(activation_efficiency.get("stale_checkout_session_count", 0))
    missing_sessions = int(activation_efficiency.get("missing_checkout_session_count", 0))
    if stale_sessions or missing_sessions:
        actions.append("Run memory checkout before continuing sessions without fresh Zaxy context.")
        remediations = [
            _checkout_remediation(
                code="sessions_without_fresh_checkout",
                eventloom_path=eventloom_path,
                session_id=str(latest_checkout["thread"]),
            )
        ]
        return {
            "status": "warning",
            "message": "Some high-context sessions lack fresh memory checkout",
            "stale_after_minutes": stale_after_minutes,
            "latest_checkout": latest_checkout,
            "latest_capture": latest_capture,
            "latest_reminder": latest_reminder,
            "activation_efficiency": activation_efficiency,
            "actions": actions,
            "remediations": remediations,
        }
    return {
        "status": "ok",
        "message": "Latest memory checkout is fresh",
        "stale_after_minutes": stale_after_minutes,
        "latest_checkout": latest_checkout,
        "latest_capture": latest_capture,
        "latest_reminder": latest_reminder,
        "activation_efficiency": activation_efficiency,
        "actions": [],
        "remediations": [],
    }


def _activation_session_id(*events: dict[str, Any] | None) -> str:
    for event in events:
        if event and event.get("thread"):
            return str(event["thread"])
    return "default"


def _checkout_remediation(
    *,
    code: str,
    eventloom_path: Path,
    session_id: str,
) -> dict[str, str]:
    query = "current project memory and next useful action"
    return {
        "code": code,
        "message": "Run Memory Checkout before the next model or task call.",
        "command": (
            f"zaxy memory checkout {shlex.quote(query)} "
            f"--eventloom-path {shlex.quote(str(eventloom_path))} "
            f"--session-id {shlex.quote(session_id)}"
        ),
    }


def _activation_efficiency(
    eventloom_path: Path,
    *,
    stale_after_minutes: int,
) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    for path in _eventlog_paths(eventloom_path):
        try:
            events = EventLog(path).read_all()
        except Exception:
            continue
        first_substantive = _first_substantive_event(events)
        if first_substantive is None:
            continue
        checkout = _latest_checkout_before(events, first_substantive)
        status = "missing_checkout"
        if checkout is not None:
            checkout_time = _parse_event_timestamp(checkout.timestamp)
            first_time = _parse_event_timestamp(first_substantive.timestamp)
            age_seconds = (first_time - checkout_time).total_seconds()
            status = "fresh_checkout" if age_seconds <= stale_after_minutes * 60 else "stale_checkout"
        sessions.append(
            {
                "session_id": first_substantive.thread,
                "status": status,
                "first_substantive_event": _summarize_observation_event(first_substantive),
                "checkout": _summarize_observation_event(checkout) if checkout is not None else None,
            }
        )
    sessions.sort(key=lambda item: str(item["session_id"]))
    high_context_count = len(sessions)
    fresh_count = sum(1 for session in sessions if session["status"] == "fresh_checkout")
    stale_count = sum(1 for session in sessions if session["status"] == "stale_checkout")
    missing_count = sum(1 for session in sessions if session["status"] == "missing_checkout")
    return {
        "high_context_session_count": high_context_count,
        "fresh_checkout_session_count": fresh_count,
        "stale_checkout_session_count": stale_count,
        "missing_checkout_session_count": missing_count,
        "fresh_checkout_rate": fresh_count / high_context_count if high_context_count else None,
        "sessions": sessions,
    }


def _first_substantive_event(events: list[Event]) -> Event | None:
    first: Event | None = None
    for event in events:
        if _observation_coverage_type(event.type) not in HIGH_VALUE_OBSERVATION_TYPES:
            continue
        if first is None or event.timestamp < first.timestamp or (
            event.timestamp == first.timestamp and event.seq < first.seq
        ):
            first = event
    return first


def _latest_checkout_before(events: list[Event], event: Event) -> Event | None:
    checkout: Event | None = None
    event_time = _parse_event_timestamp(event.timestamp)
    for candidate in events:
        if candidate.type != "memory.checkout.completed":
            continue
        candidate_time = _parse_event_timestamp(candidate.timestamp)
        if candidate_time > event_time:
            continue
        if checkout is None or candidate.timestamp > checkout.timestamp or (
            candidate.timestamp == checkout.timestamp and candidate.seq > checkout.seq
        ):
            checkout = candidate
    return checkout


def _eventlog_paths(eventloom_path: Path) -> list[Path]:
    if eventloom_path.is_file():
        return [eventloom_path]
    if eventloom_path.is_dir():
        return sorted(eventloom_path.glob("*.jsonl"))
    return []


def _observation_coverage_type(event_type: str) -> str | None:
    if event_type.startswith("hook."):
        return "hook"
    if event_type in OBSERVATION_COVERAGE_TYPES:
        return event_type
    return None


def _event_is_newer(event: Event, latest: dict[str, Any]) -> bool:
    latest_timestamp = str(latest["timestamp"])
    latest_seq = int(latest["seq"])
    return event.timestamp > latest_timestamp or (
        event.timestamp == latest_timestamp and event.seq > latest_seq
    )


def _summarize_observation_event(event: Event) -> dict[str, Any]:
    summary = {
        "seq": event.seq,
        "hash": event.hash,
        "timestamp": event.timestamp,
        "type": event.type,
        "thread": event.thread,
        "source": event.payload.get("source", "unknown"),
    }
    token_efficiency = event.payload.get("token_efficiency")
    if isinstance(token_efficiency, dict):
        summary["token_efficiency"] = token_efficiency
    return summary


def _parse_event_timestamp(timestamp: str) -> datetime:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _looks_like_zaxy_hook_config(path: Path, *, allow_text: bool = False) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if allow_text and "zaxy hook-event" in content:
        return True
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    if _looks_like_codex_capture_config(payload):
        return True
    return _contains_zaxy_hook_command(payload)


def _looks_like_codex_capture_config(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("client") == "codex"
        and value.get("capture") == "local-session-jsonl"
        and isinstance(value.get("eventloom_path"), str)
        and isinstance(value.get("session_id"), str)
    )


def _parse_json_object(text: str, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} contains invalid JSON; repair it before installing Zaxy hooks") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def _merge_claude_hook_settings(existing: dict[str, Any], generated: dict[str, Any], *, path: Path) -> None:
    existing_hooks = existing.setdefault("hooks", {})
    generated_hooks = generated.get("hooks", {})
    if not isinstance(existing_hooks, dict):
        raise ValueError(f"{path} field 'hooks' must contain a JSON object")
    if not isinstance(generated_hooks, dict):
        raise ValueError("generated Claude Code hook config field 'hooks' must contain a JSON object")
    for event_name, groups in generated_hooks.items():
        if not isinstance(groups, list):
            raise ValueError(f"generated Claude Code hook event {event_name!r} must contain a list")
        existing_event = existing_hooks.setdefault(event_name, [])
        if not isinstance(existing_event, list):
            raise ValueError(f"{path} hook event {event_name!r} must contain a list")
        existing_event.extend(groups)


def _contains_zaxy_hook_command(value: Any) -> bool:
    if isinstance(value, dict):
        command = value.get("command")
        if isinstance(command, str) and "zaxy hook-event" in command:
            return True
        return any(_contains_zaxy_hook_command(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_zaxy_hook_command(child) for child in value)
    return False


def _remove_zaxy_hook_handlers(settings: dict[str, Any]) -> None:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return
    for event_name, groups in list(hooks.items()):
        if not isinstance(groups, list):
            continue
        kept_groups = []
        for group in groups:
            if not isinstance(group, dict):
                kept_groups.append(group)
                continue
            handlers = group.get("hooks")
            if not isinstance(handlers, list):
                kept_groups.append(group)
                continue
            kept_handlers = [handler for handler in handlers if not _contains_zaxy_hook_command(handler)]
            if kept_handlers:
                updated = dict(group)
                updated["hooks"] = kept_handlers
                kept_groups.append(updated)
        if kept_groups:
            hooks[event_name] = kept_groups
        else:
            hooks.pop(event_name, None)


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
