"""Deterministic capture soak reporting for beta readiness."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zaxy.hooks import HIGH_VALUE_OBSERVATION_TYPES, inspect_hook_status


def build_capture_soak_report(
    *,
    eventloom_path: str | Path = ".eventloom",
    workspace_root: str | Path | None = None,
    session_id: str | None = None,
    max_stale_minutes: int = 30,
) -> dict[str, Any]:
    """Build a read-only capture soak report from Eventloom observation state."""
    scoped_eventloom = _scoped_eventloom_path(Path(eventloom_path), session_id=session_id)
    root = Path(workspace_root or Path.cwd()).resolve()
    hook_report = inspect_hook_status(
        eventloom_path=scoped_eventloom,
        workspace_root=root,
    )
    coverage = hook_report["observation_coverage"]
    missing = list(hook_report["capture_readiness"]["missing_observation_types"])
    stale = _stale_observation_types(coverage, max_stale_minutes=max_stale_minutes)
    actions = _dedupe_actions(
        [
            *hook_report["capture_readiness"].get("actions", []),
            *([f"Refresh stale capture lanes: {', '.join(stale)}."] if stale else []),
        ]
    )
    passes = hook_report["capture_readiness"]["status"] == "ok" and not stale
    status = "ok" if passes else "warning"
    active_count = len(hook_report["capture_readiness"]["active_observation_types"])
    total = len(HIGH_VALUE_OBSERVATION_TYPES)
    return {
        "status": status,
        "message": f"{active_count} of {total} deterministic capture lanes are active",
        "checked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "eventloom_path": str(scoped_eventloom),
        "workspace_root": str(root),
        "session_id": session_id,
        "max_stale_minutes": max_stale_minutes,
        "latest_hook_event": hook_report.get("latest_event"),
        "codex_capture": hook_report["clients"].get("codex", {}),
        "observation_coverage": coverage,
        "active_observation_types": list(hook_report["capture_readiness"]["active_observation_types"]),
        "missing_observation_types": missing,
        "stale_observation_types": stale,
        "actions": actions,
        "beta_criteria": {
            "status": "pass" if passes else "fail",
            "requires": [
                "all high-value deterministic capture lanes observed",
                f"latest events no older than {max_stale_minutes} minutes",
            ],
        },
    }


def format_capture_soak_report(report: dict[str, Any]) -> str:
    """Format a capture soak report for human operators."""
    lines = [
        f"Zaxy capture soak: {report['status']}",
        f"- beta criteria: {report['beta_criteria']['status']}",
        f"- session: {report.get('session_id') or 'all'}",
        f"- eventloom: {report['eventloom_path']}",
        f"- checked at: {report['checked_at']}",
        f"- freshness window: {report['max_stale_minutes']} minutes",
    ]
    latest = report.get("latest_hook_event")
    if latest:
        lines.append(
            f"- latest hook: {latest['type']} seq={latest['seq']} hash={latest['hash']} "
            f"session={latest['thread']} source={latest['source']}"
        )
    else:
        lines.append("- latest hook: missing")
    codex = report.get("codex_capture", {})
    runtime = codex.get("runtime", {}) if isinstance(codex, dict) else {}
    if runtime:
        state = "running" if runtime.get("running") else "not running"
        pids = runtime.get("pids", [])
        suffix = f" pid={', '.join(str(pid) for pid in pids)}" if pids else ""
        lines.append(f"- codex capture: {state}{suffix}")
    lines.append("- observation coverage:")
    for event_type in HIGH_VALUE_OBSERVATION_TYPES:
        entry = report["observation_coverage"].get(event_type, {})
        latest_observation = entry.get("latest")
        if latest_observation:
            freshness = " stale" if event_type in report["stale_observation_types"] else ""
            lines.append(
                f"  {event_type}: count={entry['count']} latest seq={latest_observation['seq']} "
                f"hash={latest_observation['hash']} session={latest_observation['thread']}{freshness}"
            )
        else:
            lines.append(f"  {event_type}: missing")
    actions = report.get("actions", [])
    if actions:
        lines.append("- actions:")
        lines.extend(f"  {action}" for action in actions)
    return "\n".join(lines)


def _scoped_eventloom_path(eventloom_path: Path, *, session_id: str | None) -> Path:
    if session_id and eventloom_path.is_dir():
        return eventloom_path / f"{session_id}.jsonl"
    return eventloom_path


def _stale_observation_types(
    coverage: dict[str, dict[str, Any]],
    *,
    max_stale_minutes: int,
) -> list[str]:
    if max_stale_minutes < 0:
        return []
    now = datetime.now(UTC)
    stale: list[str] = []
    for event_type in HIGH_VALUE_OBSERVATION_TYPES:
        latest = coverage.get(event_type, {}).get("latest")
        if latest is None:
            continue
        timestamp = _parse_timestamp(str(latest["timestamp"]))
        if (now - timestamp).total_seconds() >= max_stale_minutes * 60:
            stale.append(event_type)
    return stale


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _dedupe_actions(actions: list[str]) -> list[str]:
    deduped: list[str] = []
    for action in actions:
        if action and action not in deduped:
            deduped.append(action)
    return deduped
