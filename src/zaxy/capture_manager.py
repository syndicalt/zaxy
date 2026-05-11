"""Managed deterministic capture runtime helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zaxy import hooks
from zaxy.codex_capture import CODEX_CAPTURE_CONFIG
from zaxy.event import EventLog

CODEX_CAPTURE_STATE = "codex-capture.json"


def inspect_codex_capture(*, workspace: str | Path) -> dict[str, Any]:
    """Return configured and live runtime status for deterministic Codex capture."""
    root = Path(workspace).expanduser().resolve()
    config = _read_codex_capture_config(root)
    eventloom = _configured_eventloom(root, config)
    runtime = hooks.detect_codex_capture_runtime(root, eventloom)
    state_file = _state_file(eventloom)
    state = _read_json_file(state_file) if state_file.exists() else None
    latest = _latest_observation(eventloom, source=str(config.get("source", "codex-local")) if config else None)
    return {
        "client": "codex",
        "configured": config is not None,
        "config_file": str(root / CODEX_CAPTURE_CONFIG),
        "workspace": str(root),
        "eventloom_path": str(eventloom),
        "state_file": str(state_file),
        "state": state,
        "running": bool(runtime["running"]),
        "pids": runtime["pids"],
        "message": runtime["message"] if config else "Codex capture is not configured",
        "latest_observation": latest,
    }


def start_codex_capture(
    *,
    workspace: str | Path,
    graph: bool = False,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    max_records_per_file: int = 500,
) -> dict[str, Any]:
    """Start a managed Codex capture watcher from repo-local config."""
    root = Path(workspace).expanduser().resolve()
    config = _require_codex_capture_config(root)
    eventloom = _configured_eventloom(root, config)
    runtime = hooks.detect_codex_capture_runtime(root, eventloom)
    if runtime["running"]:
        return {
            "started": False,
            "pid": runtime["pids"][0],
            "message": f"Codex capture watcher already running pid={runtime['pids'][0]}",
            "state_file": str(_state_file(eventloom)),
        }
    command = _codex_capture_command(
        config,
        graph=graph,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        max_records_per_file=max_records_per_file,
    )
    eventloom.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    state_file = _state_file(eventloom)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "client": "codex",
        "pid": process.pid,
        "command": command,
        "workspace": str(root),
        "eventloom_path": str(eventloom),
        "graph": graph,
        "started_at": datetime.now(UTC).isoformat(),
    }
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "started": True,
        "pid": process.pid,
        "message": f"Started Codex capture watcher pid={process.pid}",
        "state_file": str(state_file),
    }


def stop_codex_capture(*, workspace: str | Path) -> dict[str, Any]:
    """Stop the managed Codex capture watcher if the recorded PID still matches."""
    root = Path(workspace).expanduser().resolve()
    config = _read_codex_capture_config(root)
    eventloom = _configured_eventloom(root, config)
    state_file = _state_file(eventloom)
    if not state_file.exists():
        return {"stopped": False, "pid": None, "message": "No managed Codex capture watcher state found"}
    state = _read_json_file(state_file)
    pid = int(state.get("pid", 0))
    runtime = hooks.detect_codex_capture_runtime(root, eventloom)
    if pid not in runtime["pids"]:
        state_file.unlink(missing_ok=True)
        return {"stopped": False, "pid": pid, "message": f"Removed stale Codex capture watcher state pid={pid}"}
    os.kill(pid, signal.SIGTERM)
    state_file.unlink(missing_ok=True)
    return {"stopped": True, "pid": pid, "message": f"Stopped Codex capture watcher pid={pid}"}


def _read_codex_capture_config(workspace: Path) -> dict[str, Any] | None:
    path = workspace / CODEX_CAPTURE_CONFIG
    if not path.exists():
        return None
    payload = _read_json_file(path)
    if (
        payload.get("client") != "codex"
        or payload.get("capture") != "local-session-jsonl"
        or not isinstance(payload.get("workspace"), str)
        or not isinstance(payload.get("eventloom_path"), str)
        or not isinstance(payload.get("session_id"), str)
    ):
        raise ValueError(f"{path} is not a valid Codex capture config")
    return payload


def _require_codex_capture_config(workspace: Path) -> dict[str, Any]:
    config = _read_codex_capture_config(workspace)
    if config is None:
        raise FileNotFoundError(f"{workspace / CODEX_CAPTURE_CONFIG} not found; run zaxy init --preset local-codex")
    return config


def _configured_eventloom(workspace: Path, config: dict[str, Any] | None) -> Path:
    if config is None:
        return workspace / ".eventloom"
    return _resolve_path(workspace, str(config["eventloom_path"]))


def _codex_capture_command(
    config: dict[str, Any],
    *,
    graph: bool,
    neo4j_uri: str | None,
    neo4j_user: str | None,
    neo4j_password: str | None,
    max_records_per_file: int,
) -> list[str]:
    workspace = Path(str(config["workspace"])).expanduser().resolve()
    eventloom = _resolve_path(workspace, str(config["eventloom_path"]))
    command = [
        sys.executable,
        "-m",
        "zaxy",
        "codex-capture",
        "--workspace",
        str(workspace),
        "--codex-home",
        str(Path(str(config["codex_home"])).expanduser()),
        "--eventloom-path",
        str(eventloom),
        "--session-id",
        str(config["session_id"]),
        "--source",
        str(config.get("source", "codex-local")),
        "--watch",
        "--max-records-per-file",
        str(max_records_per_file),
    ]
    if graph:
        command.append("--graph")
        if neo4j_uri:
            command.extend(["--neo4j-uri", neo4j_uri])
        if neo4j_user:
            command.extend(["--neo4j-user", neo4j_user])
        if neo4j_password:
            command.extend(["--neo4j-password", neo4j_password])
    return command


def _state_file(eventloom_path: Path) -> Path:
    return eventloom_path / "runtime" / CODEX_CAPTURE_STATE


def _read_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _latest_observation(eventloom_path: Path, *, source: str | None) -> dict[str, Any] | None:
    if not eventloom_path.exists():
        return None
    latest = None
    paths = [eventloom_path] if eventloom_path.is_file() else sorted(eventloom_path.glob("*.jsonl"))
    for path in paths:
        try:
            events = EventLog(path).read_all()
        except Exception:
            continue
        for event in events:
            if source is not None and event.payload.get("source") != source:
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
    }
