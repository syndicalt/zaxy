"""Workspace profile discovery and session genesis events."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkspaceProfile:
    """A lightweight classification of the active workspace root."""

    workspace_type: str
    confidence: float
    signals: list[str]
    instructions_profile: str


CODEBASE_EVENTS = [
    "decision.made",
    "task.completed",
    "code.file.indexed",
    "code.symbol.indexed",
    "code.import.indexed",
    "code.dependency.indexed",
    "code.call.indexed",
    "code.coverage.indexed",
]

GENERIC_EVENTS = [
    "observation.recorded",
    "decision.made",
    "task.completed",
    "artifact.indexed",
]

MEMORY_ACTIVATION_INSTRUCTIONS = {
    "required_tool": "memory_checkout",
    "before": ["roadmap", "implementation", "release", "review", "resume", "high_context_question"],
    "reason": "Keep model work grounded in fresh, cited Zaxy memory.",
    "session_start_command": (
        "zaxy activate codex --eventloom-path <eventloom_path> "
        "--session-id <session_id> --current-task '<task>' --workspace-root <workspace_root>"
    ),
    "launch_command": (
        "zaxy activate codex --eventloom-path <eventloom_path> "
        "--session-id <session_id> --current-task '<task>' --workspace-root <workspace_root> --launch"
    ),
    "resume_command": (
        "zaxy hook-event resume --eventloom-path <eventloom_path> "
        "--session-id <session_id> --source codex --summary '<task>'"
    ),
    "checkout_fallback_command": (
        "zaxy memory checkout '<task>' --eventloom-path <eventloom_path> --session-id <session_id>"
    ),
    "mcp_tools_status": "runtime_unverified",
    "fail_closed": True,
    "blocker": "If activation packet or fresh checkout is absent, pause substantial work and run the CLI fallback.",
}

MEMORY_FRONT_DOOR = {
    "tool": "memory_checkout",
    "guidance": (
        "memory_checkout is the front door to Zaxy memory: call it first with the current "
        "task before substantial work. Use memory_capabilities to discover the rest of the "
        "tool surface."
    ),
}

INSTRUCTION_FILES = {
    "AGENTS.md": "agents",
    "CLAUDE.md": "claude",
    "SOUL.md": "soul",
    ".github/copilot-instructions.md": "copilot",
}


def discover_workspace_profile(root: str | Path) -> WorkspaceProfile:
    """Classify a workspace from lightweight filesystem signals."""
    root_path = Path(root).resolve()
    signals: list[str] = []
    if (root_path / "pyproject.toml").exists():
        signals.append("pyproject.toml")
    if (root_path / "package.json").exists():
        signals.append("package.json")
    if (root_path / "go.mod").exists():
        signals.append("go.mod")
    if (root_path / "Cargo.toml").exists():
        signals.append("Cargo.toml")
    if (root_path / "src").is_dir():
        signals.append("src/")
    if (root_path / "tests").is_dir():
        signals.append("tests/")
    if (root_path / ".git").exists():
        signals.append(".git/")

    codebase_score = _codebase_score(signals)
    if codebase_score >= 0.5:
        return WorkspaceProfile(
            workspace_type="codebase",
            confidence=codebase_score,
            signals=signals,
            instructions_profile="codebase",
        )
    return WorkspaceProfile(
        workspace_type="generic_workspace",
        confidence=0.2,
        signals=[],
        instructions_profile="generic",
    )


def build_session_genesis_event(root: str | Path, *, session_id: str) -> dict[str, Any]:
    """Build a session.genesis event input for Eventloom append."""
    root_path = Path(root).resolve()
    profile = discover_workspace_profile(root_path)
    return {
        "event_type": "session.genesis",
        "actor": "zaxy",
        "payload": {
            "root": str(root_path),
            "workspace_type": profile.workspace_type,
            "confidence": profile.confidence,
            "signals": profile.signals,
            "instructions_profile": profile.instructions_profile,
            "session_id": session_id,
            "write_instructions": _write_instructions(profile.instructions_profile),
        },
    }


def build_workspace_instruction_event(root: str | Path, *, session_id: str) -> dict[str, Any] | None:
    """Build a compact event describing discovered workspace instruction files."""
    root_path = Path(root).resolve()
    files = [
        _instruction_file_payload(root_path, relative_path, kind)
        for relative_path, kind in INSTRUCTION_FILES.items()
    ]
    discovered = [item for item in files if item is not None]
    if not discovered:
        return None
    return {
        "event_type": "workspace.instructions.discovered",
        "actor": "zaxy",
        "payload": {
            "root": str(root_path),
            "session_id": session_id,
            "summary": " ".join(str(item["summary"]) for item in discovered if item.get("summary")),
            "signature": _instruction_signature(discovered),
            "files": discovered,
            "memory_front_door": dict(MEMORY_FRONT_DOOR),
        },
    }


def mark_workspace_instruction_event_updated(
    event: dict[str, Any],
    *,
    previous_signature: str,
) -> dict[str, Any]:
    """Return an instruction event marked as an update from a previous signature."""
    updated = {
        "event_type": "workspace.instructions.updated",
        "actor": event["actor"],
        "payload": {
            **event["payload"],
            "previous_signature": previous_signature,
        },
    }
    return updated


def workspace_profile_from_payload(payload: dict[str, Any]) -> WorkspaceProfile:
    """Reconstruct a workspace profile from a session genesis payload."""
    signals = payload.get("signals", [])
    if not isinstance(signals, list):
        signals = []
    return WorkspaceProfile(
        workspace_type=str(payload.get("workspace_type", "generic_workspace")),
        confidence=float(payload.get("confidence", 0.0)),
        signals=[str(signal) for signal in signals],
        instructions_profile=str(payload.get("instructions_profile", "generic")),
    )


def existing_session_genesis_profile(
    events: Iterable[object],
    *,
    root: str | Path,
    session_id: str,
) -> WorkspaceProfile | None:
    """Return an existing genesis profile for the root/session pair, if present."""
    resolved_root = str(Path(root).resolve())
    for event in events:
        if getattr(event, "type", None) != "session.genesis":
            continue
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            continue
        if payload.get("root") == resolved_root and payload.get("session_id") == session_id:
            return workspace_profile_from_payload(payload)
    return None


def existing_workspace_instructions_signature(
    events: Iterable[object],
    *,
    root: str | Path,
    session_id: str,
) -> str | None:
    """Return the latest instruction discovery signature for the root/session pair."""
    resolved_root = str(Path(root).resolve())
    signature: str | None = None
    for event in events:
        if getattr(event, "type", None) not in {
            "workspace.instructions.discovered",
            "workspace.instructions.updated",
        }:
            continue
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            continue
        if payload.get("root") == resolved_root and payload.get("session_id") == session_id:
            raw_signature = payload.get("signature")
            signature = str(raw_signature) if raw_signature is not None else None
    return signature


def _codebase_score(signals: list[str]) -> float:
    score = 0.0
    for signal in signals:
        if signal in {"pyproject.toml", "package.json", "go.mod", "Cargo.toml"}:
            score += 0.45
        elif signal in {"src/", "tests/"}:
            score += 0.25
        elif signal == ".git/":
            score += 0.1
    return min(round(score, 2), 0.95)


def _write_instructions(profile: str) -> dict[str, Any]:
    if profile == "codebase":
        return {
            "preferred_events": CODEBASE_EVENTS,
            "avoid_writing": ["raw_secrets", "full_source_bodies", "transient_chatter"],
            "indexing_strategy": "metadata_only_codebase_map",
            "memory_activation": MEMORY_ACTIVATION_INSTRUCTIONS,
        }
    return {
        "preferred_events": GENERIC_EVENTS,
        "avoid_writing": ["raw_secrets", "transient_chatter"],
        "indexing_strategy": "metadata_only_artifact_map",
        "memory_activation": MEMORY_ACTIVATION_INSTRUCTIONS,
    }


def _instruction_file_payload(root: Path, relative_path: str, kind: str) -> dict[str, Any] | None:
    path = root / relative_path
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    return {
        "path": relative_path,
        "kind": kind,
        "size_bytes": len(content.encode("utf-8")),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "summary": _summarize_instruction_lines(lines),
        "citation": f"{path}:1-{max(len(lines), 1)}",
    }


def _summarize_instruction_lines(lines: list[str]) -> str:
    heading = "Workspace instructions"
    body = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and heading == "Workspace instructions":
            heading = stripped.lstrip("#").strip() or heading
            continue
        body = stripped.lstrip("-*0123456789. ").strip()
        break
    if body and body[-1] not in ".!?":
        body = f"{body}."
    return f"{heading}: {body}" if body else heading


def _instruction_signature(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
