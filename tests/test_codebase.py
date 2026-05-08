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

    assert events[0] == {
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


def test_collect_codebase_events_indexes_python_symbols_and_imports_by_default(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "service.py"
    source.parent.mkdir()
    source.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "class Service:\n"
        "    def run(self):\n"
        "        return Path(json.dumps({}))\n",
        encoding="utf-8",
    )

    events = collect_codebase_events(tmp_path)

    symbols = [event["payload"] for event in events if event["event_type"] == "code.symbol.indexed"]
    imports = [event["payload"] for event in events if event["event_type"] == "code.import.indexed"]
    assert symbols == [
        {
            "path": "pkg/service.py",
            "language": "python",
            "name": "Service",
            "qualified_name": "Service",
            "kind": "class",
            "start_line": 4,
            "end_line": 6,
        },
        {
            "path": "pkg/service.py",
            "language": "python",
            "name": "run",
            "qualified_name": "Service.run",
            "kind": "function",
            "start_line": 5,
            "end_line": 6,
        },
    ]
    assert imports == [
        {
            "path": "pkg/service.py",
            "language": "python",
            "module": "json",
            "name": "json",
            "kind": "import",
            "start_line": 1,
        },
        {
            "path": "pkg/service.py",
            "language": "python",
            "module": "pathlib",
            "name": "Path",
            "kind": "from_import",
            "start_line": 2,
        },
    ]


def test_collect_codebase_events_indexes_symbols_for_multiple_file_types(tmp_path: Path) -> None:
    ts = tmp_path / "web" / "app.ts"
    ts.parent.mkdir()
    ts.write_text(
        "import { createApp } from 'vue';\n"
        "export class Dashboard {}\n"
        "export function mountDashboard() {}\n",
        encoding="utf-8",
    )
    go = tmp_path / "cmd" / "server.go"
    go.parent.mkdir()
    go.write_text(
        "package main\n\n"
        "import \"net/http\"\n\n"
        "type Server struct {}\n"
        "func Start() {}\n",
        encoding="utf-8",
    )

    events = collect_codebase_events(tmp_path)

    event_payloads = [
        (event["event_type"], event["payload"])
        for event in events
        if event["event_type"] in {"code.symbol.indexed", "code.import.indexed"}
    ]
    assert (
        "code.import.indexed",
        {
            "path": "cmd/server.go",
            "language": "go",
            "module": "net/http",
            "name": "net/http",
            "kind": "import",
            "start_line": 3,
        },
    ) in event_payloads
    assert (
        "code.symbol.indexed",
        {
            "path": "cmd/server.go",
            "language": "go",
            "name": "Server",
            "qualified_name": "Server",
            "kind": "type",
            "start_line": 5,
            "end_line": 5,
        },
    ) in event_payloads
    assert (
        "code.symbol.indexed",
        {
            "path": "web/app.ts",
            "language": "typescript",
            "name": "mountDashboard",
            "qualified_name": "mountDashboard",
            "kind": "function",
            "start_line": 3,
            "end_line": 3,
        },
    ) in event_payloads
    assert (
        "code.import.indexed",
        {
            "path": "web/app.ts",
            "language": "typescript",
            "module": "vue",
            "name": "createApp",
            "kind": "import",
            "start_line": 1,
        },
    ) in event_payloads


def test_collect_codebase_events_resolves_python_imports_to_local_files(tmp_path: Path) -> None:
    core = tmp_path / "src" / "zaxy" / "core.py"
    core.parent.mkdir(parents=True)
    core.write_text("class MemoryFabric:\n    pass\n", encoding="utf-8")
    server = tmp_path / "src" / "zaxy" / "mcp_server.py"
    server.write_text(
        "from pathlib import Path\n"
        "from zaxy.core import MemoryFabric\n"
        "from .core import MemoryFabric as LocalFabric\n",
        encoding="utf-8",
    )

    events = collect_codebase_events(tmp_path)

    dependencies = [event["payload"] for event in events if event["event_type"] == "code.dependency.indexed"]
    assert dependencies == [
        {
            "source_path": "src/zaxy/mcp_server.py",
            "target_path": "src/zaxy/core.py",
            "language": "python",
            "module": "zaxy.core",
            "import_name": "MemoryFabric",
            "start_line": 2,
            "resolution": "module_file",
        },
        {
            "source_path": "src/zaxy/mcp_server.py",
            "target_path": "src/zaxy/core.py",
            "language": "python",
            "module": ".core",
            "import_name": "LocalFabric",
            "start_line": 3,
            "resolution": "relative_file",
        },
    ]


def test_collect_codebase_events_resolves_javascript_relative_imports(tmp_path: Path) -> None:
    app = tmp_path / "web" / "app.ts"
    app.parent.mkdir()
    app.write_text(
        "import { helper } from './util';\n"
        "import { createApp } from 'vue';\n",
        encoding="utf-8",
    )
    util = tmp_path / "web" / "util.ts"
    util.write_text("export function helper() {}\n", encoding="utf-8")

    events = collect_codebase_events(tmp_path)

    dependencies = [event["payload"] for event in events if event["event_type"] == "code.dependency.indexed"]
    assert dependencies == [
        {
            "source_path": "web/app.ts",
            "target_path": "web/util.ts",
            "language": "typescript",
            "module": "./util",
            "import_name": "helper",
            "start_line": 1,
            "resolution": "relative_file",
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
