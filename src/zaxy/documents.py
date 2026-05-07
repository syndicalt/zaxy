"""Filesystem document ingestion helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
DEFAULT_MAX_LINES = 80
DEFAULT_MAX_BYTES = 512 * 1024


def collect_document_events(
    root: str | Path,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[dict[str, Any]]:
    """Collect supported local documents as document.indexed event inputs."""
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise ValueError(f"document root does not exist: {root_path}")
    if max_lines < 1:
        raise ValueError("max_lines must be positive")

    paths = [root_path] if root_path.is_file() else _iter_supported_files(root_path, max_bytes)
    events: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            continue
        if path.stat().st_size > max_bytes:
            continue
        content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        rel_path = _relative_path(path, root_path)
        lines = content.splitlines()
        for start in range(0, len(lines), max_lines):
            chunk_lines = lines[start : start + max_lines]
            if not any(line.strip() for line in chunk_lines):
                continue
            events.append(
                {
                    "event_type": "document.indexed",
                    "actor": "zaxy-doc-ingest",
                    "payload": {
                        "path": rel_path,
                        "start_line": start + 1,
                        "end_line": start + len(chunk_lines),
                        "content": "\n".join(chunk_lines),
                        "sha256": digest,
                    },
                }
            )
    return events


def _iter_supported_files(root: Path, max_bytes: int) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            continue
        if path.stat().st_size > max_bytes:
            continue
        files.append(path)
    return sorted(files)


def _relative_path(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    return path.relative_to(root).as_posix()
