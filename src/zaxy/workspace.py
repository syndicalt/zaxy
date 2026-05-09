"""Workspace profile discovery and session genesis events."""

from __future__ import annotations

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
        }
    return {
        "preferred_events": GENERIC_EVENTS,
        "avoid_writing": ["raw_secrets", "transient_chatter"],
        "indexing_strategy": "metadata_only_artifact_map",
    }
