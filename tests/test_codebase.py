"""Tests for codebase file inventory ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from zaxy.codebase import collect_codebase_events


def test_collects_supported_code_files_with_metadata(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    content = "def main():\n    return 42\n"
    source.write_text(content, encoding="utf-8")

    events = collect_codebase_events(tmp_path)

    assert events == [
        {
            "event_type": "code.file.indexed",
            "actor": "zaxy-codebase-indexer",
            "payload": {
                "path": "src/app.py",
                "language": "python",
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "bytes": len(content.encode("utf-8")),
                "lines": 2,
            },
        }
    ]


def test_collect_codebase_events_skips_hidden_cache_dependency_and_large_files(tmp_path: Path) -> None:
    keep = tmp_path / "pkg" / "mod.ts"
    keep.parent.mkdir()
    keep.write_text("x=1\n", encoding="utf-8")
    hidden = tmp_path / ".git" / "config"
    hidden.parent.mkdir()
    hidden.write_text("secret-ish\n", encoding="utf-8")
    cache = tmp_path / "__pycache__" / "mod.py"
    cache.parent.mkdir()
    cache.write_text("cached\n", encoding="utf-8")
    deps = tmp_path / "node_modules" / "lib.js"
    deps.parent.mkdir()
    deps.write_text("dependency\n", encoding="utf-8")
    large = tmp_path / "big.py"
    large.write_text("x" * 40, encoding="utf-8")

    events = collect_codebase_events(tmp_path, max_bytes=10)

    assert [event["payload"]["path"] for event in events] == ["pkg/mod.ts"]


def test_collect_codebase_events_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="codebase root does not exist"):
        collect_codebase_events(tmp_path / "missing")
