"""Incremental source refresh planning for agent context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from zaxy.codebase import EXCLUDED_DIRS, LANGUAGE_BY_SUFFIX, collect_codebase_events
from zaxy.documents import SUPPORTED_SUFFIXES, collect_document_events
from zaxy.security import validate_session_id

ContextRefreshKind = Literal["documents", "codebase"]
REFRESH_ACTOR = "zaxy-context-refresh"
STATE_VERSION = 1
TRANSFORM_VERSION_BY_KIND = {
    "documents": "documents-v1",
    "codebase": "codebase-v1",
}


@dataclass(frozen=True)
class SourceSnapshot:
    """Fingerprint for one source file considered during refresh."""

    path: str
    source_kind: str
    sha256: str
    bytes: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON state for this source."""
        return {
            "path": self.path,
            "source_kind": self.source_kind,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "mtime_ns": self.mtime_ns,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SourceSnapshot:
        """Load a source snapshot from persisted JSON."""
        return cls(
            path=str(payload["path"]),
            source_kind=str(payload["source_kind"]),
            sha256=str(payload["sha256"]),
            bytes=int(payload["bytes"]),
            mtime_ns=int(payload["mtime_ns"]),
        )


@dataclass(frozen=True)
class ContextRefreshState:
    """Persisted source fingerprints for one refresh kind."""

    kind: str
    sources: dict[str, SourceSnapshot]
    transform_version: str | None = None
    version: int = STATE_VERSION

    @classmethod
    def empty(cls, kind: str = "documents") -> ContextRefreshState:
        """Return an empty refresh state."""
        return cls(kind=kind, sources={}, transform_version=_transform_version(kind))

    @classmethod
    def from_snapshots(cls, *, kind: str, snapshots: list[SourceSnapshot]) -> ContextRefreshState:
        """Build state from current source snapshots."""
        return cls(
            kind=kind,
            sources={snapshot.path: snapshot for snapshot in snapshots},
            transform_version=_transform_version(kind),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON state."""
        return {
            "version": self.version,
            "kind": self.kind,
            "transform_version": self.transform_version,
            "sources": {
                path: snapshot.to_dict()
                for path, snapshot in sorted(self.sources.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContextRefreshState:
        """Load refresh state from JSON."""
        return cls(
            version=int(payload.get("version", STATE_VERSION)),
            kind=str(payload.get("kind", "documents")),
            transform_version=str(payload["transform_version"]) if payload.get("transform_version") else None,
            sources={
                path: SourceSnapshot.from_dict(snapshot)
                for path, snapshot in dict(payload.get("sources", {})).items()
            },
        )


@dataclass(frozen=True)
class ContextRefreshPlan:
    """Planned Eventloom writes for an incremental refresh."""

    kind: str
    events: list[dict[str, Any]]
    next_state: ContextRefreshState
    summary: dict[str, int | str]


def collect_source_snapshots(
    root: str | Path,
    *,
    kind: str,
    max_bytes: int = 512 * 1024,
) -> list[SourceSnapshot]:
    """Collect source fingerprints for a supported refresh kind."""
    normalized = _normalize_kind(kind)
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise ValueError(f"context refresh root does not exist: {root_path}")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    suffixes = SUPPORTED_SUFFIXES if normalized == "documents" else set(LANGUAGE_BY_SUFFIX)
    paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
    snapshots: list[SourceSnapshot] = []
    for path in paths:
        if not path.is_file():
            continue
        if root_path.is_dir() and _is_excluded(path.relative_to(root_path), kind=normalized):
            continue
        if path.suffix.casefold() not in suffixes:
            continue
        stat = path.stat()
        if stat.st_size > max_bytes:
            continue
        content = path.read_bytes()
        snapshots.append(
            SourceSnapshot(
                path=_relative_path(path, root_path),
                source_kind=normalized,
                sha256=hashlib.sha256(content).hexdigest(),
                bytes=len(content),
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return sorted(snapshots, key=lambda snapshot: snapshot.path)


def plan_context_refresh(
    root: str | Path,
    *,
    kind: str,
    previous: ContextRefreshState,
    max_lines: int = 80,
    max_bytes: int = 512 * 1024,
) -> ContextRefreshPlan:
    """Plan incremental source and projection events for changed context."""
    normalized = _normalize_kind(kind)
    snapshots = collect_source_snapshots(root, kind=normalized, max_bytes=max_bytes)
    transform_version = _transform_version(normalized)
    transform_changed = bool(
        previous.sources
        and previous.kind == normalized
        and previous.transform_version != transform_version
    )
    current = {snapshot.path: snapshot for snapshot in snapshots}
    prior = previous.sources if previous.kind == normalized else {}
    discovered = sorted(path for path in current if path not in prior)
    changed = sorted(
        path
        for path, snapshot in current.items()
        if path in prior and (snapshot.sha256 != prior[path].sha256 or transform_changed)
    )
    unchanged = sorted(
        path
        for path, snapshot in current.items()
        if path in prior and snapshot.sha256 == prior[path].sha256 and not transform_changed
    )
    deleted = sorted(path for path in prior if path not in current)
    active_paths = set(discovered) | set(changed)
    events: list[dict[str, Any]] = []
    for path in discovered:
        events.append(_source_event("source.discovered", current[path]))
    for path in changed:
        reason = "source_changed" if current[path].sha256 != prior[path].sha256 else "transform_changed"
        events.append(_projection_retired_event(prior[path], reason=reason, transform_version=transform_version))
        events.append(
            _source_event(
                "source.changed",
                current[path],
                previous=prior[path],
                transform_version=transform_version,
                refresh_reason=reason,
            )
        )
    for path in unchanged:
        events.append(_source_event("source.unchanged", current[path]))
    for path in deleted:
        events.append(_projection_retired_event(prior[path], reason="source_deleted", transform_version=transform_version))
        events.append(_source_event("source.deleted", prior[path], transform_version=transform_version))

    index_events = _index_events_for_paths(
        root,
        kind=normalized,
        paths=active_paths,
        snapshots=current,
        transform_version=transform_version,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )
    events.extend(index_events)
    for event in index_events:
        events.append(_projection_updated_event(event, source_kind=normalized, transform_version=transform_version))

    next_state = ContextRefreshState.from_snapshots(kind=normalized, snapshots=snapshots)
    summary: dict[str, int | str] = {
        "kind": normalized,
        "discovered": len(discovered),
        "changed": len(changed),
        "unchanged": len(unchanged),
        "deleted": len(deleted),
        "indexed": len(index_events),
        "retired": len(changed) + len(deleted),
        "transform_changed": int(transform_changed),
    }
    return ContextRefreshPlan(kind=normalized, events=events, next_state=next_state, summary=summary)


def load_refresh_state(eventloom_path: str | Path, *, session_id: str, kind: str) -> ContextRefreshState:
    """Load persisted refresh state for a session and source kind."""
    path = _state_path(eventloom_path, session_id=session_id, kind=_normalize_kind(kind))
    if not path.exists():
        return ContextRefreshState.empty(kind=_normalize_kind(kind))
    return ContextRefreshState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_refresh_state(eventloom_path: str | Path, *, session_id: str, state: ContextRefreshState) -> None:
    """Persist refresh state for a session and source kind."""
    path = _state_path(eventloom_path, session_id=session_id, kind=state.kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_path(eventloom_path: str | Path, *, session_id: str, kind: str) -> Path:
    safe_session = validate_session_id(session_id)
    return Path(eventloom_path).resolve() / "context-refresh" / f"{safe_session}.{kind}.json"


def _normalize_kind(kind: str) -> ContextRefreshKind:
    normalized = kind.casefold().strip().replace("_", "-")
    if normalized in {"document", "documents", "docs"}:
        return "documents"
    if normalized in {"code", "codebase"}:
        return "codebase"
    raise ValueError("context refresh kind must be one of: documents, codebase")


def _transform_version(kind: str) -> str:
    return TRANSFORM_VERSION_BY_KIND[_normalize_kind(kind)]


def _source_event(
    event_type: str,
    snapshot: SourceSnapshot,
    *,
    previous: SourceSnapshot | None = None,
    transform_version: str | None = None,
    refresh_reason: str | None = None,
) -> dict[str, Any]:
    payload = snapshot.to_dict()
    if previous is not None:
        payload["previous_sha256"] = previous.sha256
    payload["transform_version"] = transform_version or _transform_version(snapshot.source_kind)
    if refresh_reason is not None:
        payload["refresh_reason"] = refresh_reason
    return {"event_type": event_type, "actor": REFRESH_ACTOR, "payload": payload}


def _projection_updated_event(event: dict[str, Any], *, source_kind: str, transform_version: str) -> dict[str, Any]:
    payload = dict(event["payload"])
    path = _event_path(payload)
    return {
        "event_type": "projection.updated",
        "actor": REFRESH_ACTOR,
        "payload": {
            "path": path,
            "source_kind": source_kind,
            "source_event": event["event_type"],
            "projection": "memory",
            "source_sha256": payload.get("source_sha256") or payload.get("sha256"),
            "transform_version": transform_version,
        },
    }


def _projection_retired_event(snapshot: SourceSnapshot, *, reason: str, transform_version: str) -> dict[str, Any]:
    return {
        "event_type": "projection.retired",
        "actor": REFRESH_ACTOR,
        "payload": {
            "path": snapshot.path,
            "source_kind": snapshot.source_kind,
            "source_sha256": snapshot.sha256,
            "projection": "memory",
            "reason": reason,
            "transform_version": transform_version,
        },
    }


def _index_events_for_paths(
    root: str | Path,
    *,
    kind: str,
    paths: set[str],
    snapshots: dict[str, SourceSnapshot],
    transform_version: str,
    max_lines: int,
    max_bytes: int,
) -> list[dict[str, Any]]:
    if not paths:
        return []
    if kind == "documents":
        events = collect_document_events(root, max_lines=max_lines, max_bytes=max_bytes)
    else:
        events = collect_codebase_events(root, max_bytes=max_bytes)
    return [
        _annotate_index_event(event, snapshots=snapshots, transform_version=transform_version)
        for event in events
        if _event_touches_paths(event, paths)
    ]


def _annotate_index_event(
    event: dict[str, Any],
    *,
    snapshots: dict[str, SourceSnapshot],
    transform_version: str,
) -> dict[str, Any]:
    payload = dict(event.get("payload", {}))
    path = _event_path(payload)
    snapshot = snapshots.get(path)
    if snapshot is not None:
        payload.setdefault("source_sha256", snapshot.sha256)
    payload["transform_version"] = transform_version
    return {**event, "payload": payload}


def _event_touches_paths(event: dict[str, Any], paths: set[str]) -> bool:
    payload = event.get("payload", {})
    for key in ("path", "source_path", "target_path", "test_path", "covered_path"):
        value = payload.get(key)
        if isinstance(value, str) and value in paths:
            return True
    return False


def _is_excluded(relative_path: Path, *, kind: str) -> bool:
    if any(part.startswith(".") for part in relative_path.parts):
        return True
    return kind == "codebase" and any(part in EXCLUDED_DIRS for part in relative_path.parts[:-1])


def _event_path(payload: dict[str, Any]) -> str:
    for key in ("path", "source_path", "target_path", "test_path", "covered_path"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return "unknown"


def _relative_path(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    return path.relative_to(root).as_posix()
