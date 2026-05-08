"""Codebase file inventory ingestion helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

DEFAULT_MAX_BYTES = 512 * 1024
EXCLUDED_DIRS = {
    ".eventloom",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
LANGUAGE_BY_SUFFIX = {
    ".css": "css",
    ".go": "go",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "javascript",
    ".md": "markdown",
    ".py": "python",
    ".rs": "rust",
    ".sh": "shell",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def collect_codebase_events(
    root: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[dict[str, Any]]:
    """Collect supported source files as code.file.indexed event inputs."""
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise ValueError(f"codebase root does not exist: {root_path}")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    paths = [root_path] if root_path.is_file() else _iter_supported_files(root_path, max_bytes)
    events: list[dict[str, Any]] = []
    for path in paths:
        language = LANGUAGE_BY_SUFFIX.get(path.suffix.casefold())
        if language is None:
            continue
        size = path.stat().st_size
        if size > max_bytes:
            continue
        content = path.read_bytes()
        rel_path = _relative_path(path, root_path)
        events.append(
            {
                "event_type": "code.file.indexed",
                "actor": "zaxy-codebase-indexer",
                "payload": {
                    "path": rel_path,
                    "language": language,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                    "lines": _line_count(content),
                },
            }
        )
    return events


def _iter_supported_files(root: Path, max_bytes: int) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") or part in EXCLUDED_DIRS for part in relative_parts[:-1]):
            continue
        if not path.is_file():
            continue
        if path.suffix.casefold() not in LANGUAGE_BY_SUFFIX:
            continue
        if path.stat().st_size > max_bytes:
            continue
        files.append(path)
    return sorted(files)


def _relative_path(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    return path.relative_to(root).as_posix()


def _line_count(content: bytes) -> int:
    if not content:
        return 0
    return len(content.decode("utf-8", errors="replace").splitlines())
