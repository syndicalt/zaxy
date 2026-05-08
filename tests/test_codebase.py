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


def test_collect_codebase_events_indexes_python_call_sites(tmp_path: Path) -> None:
    source = tmp_path / "src" / "zaxy" / "workflow.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from zaxy.core import MemoryFabric\n"
        "\n"
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def run():\n"
        "    helper()\n"
        "    MemoryFabric()\n",
        encoding="utf-8",
    )
    core = tmp_path / "src" / "zaxy" / "core.py"
    core.write_text("class MemoryFabric:\n    pass\n", encoding="utf-8")

    events = collect_codebase_events(tmp_path)

    calls = [event["payload"] for event in events if event["event_type"] == "code.call.indexed"]
    assert calls == [
        {
            "path": "src/zaxy/workflow.py",
            "language": "python",
            "caller": "run",
            "callee": "helper",
            "callee_qualified_name": "helper",
            "target_path": "src/zaxy/workflow.py",
            "target_qualified_name": "helper",
            "start_line": 7,
            "resolution": "same_file_symbol",
        },
        {
            "path": "src/zaxy/workflow.py",
            "language": "python",
            "caller": "run",
            "callee": "MemoryFabric",
            "callee_qualified_name": "MemoryFabric",
            "target_path": "src/zaxy/core.py",
            "target_qualified_name": "MemoryFabric",
            "start_line": 8,
            "resolution": "imported_symbol",
        },
    ]


def test_collect_codebase_events_links_python_tests_to_imported_symbols(tmp_path: Path) -> None:
    core = tmp_path / "src" / "zaxy" / "core.py"
    core.parent.mkdir(parents=True)
    core.write_text("class MemoryFabric:\n    pass\n", encoding="utf-8")
    test_core = tmp_path / "tests" / "test_core.py"
    test_core.parent.mkdir()
    test_core.write_text(
        "from zaxy.core import MemoryFabric\n"
        "\n"
        "def test_memory_fabric_starts():\n"
        "    fabric = MemoryFabric()\n"
        "    assert fabric\n",
        encoding="utf-8",
    )

    events = collect_codebase_events(tmp_path)

    coverage = [event["payload"] for event in events if event["event_type"] == "code.coverage.indexed"]
    assert coverage == [
        {
            "test_path": "tests/test_core.py",
            "test_name": "test_memory_fabric_starts",
            "test_qualified_name": "test_memory_fabric_starts",
            "target_path": "src/zaxy/core.py",
            "target_name": "MemoryFabric",
            "target_qualified_name": "MemoryFabric",
            "language": "python",
            "start_line": 4,
            "resolution": "imported_symbol",
        }
    ]


def test_collect_codebase_events_indexes_typescript_call_sites(tmp_path: Path) -> None:
    app = tmp_path / "web" / "app.ts"
    app.parent.mkdir()
    app.write_text(
        "import { helper } from './util';\n"
        "\n"
        "export function start() {\n"
        "  helper();\n"
        "}\n",
        encoding="utf-8",
    )
    util = tmp_path / "web" / "util.ts"
    util.write_text("export function helper() {}\n", encoding="utf-8")

    events = collect_codebase_events(tmp_path)

    calls = [event["payload"] for event in events if event["event_type"] == "code.call.indexed"]
    assert calls == [
        {
            "path": "web/app.ts",
            "language": "typescript",
            "caller": "start",
            "callee": "helper",
            "callee_qualified_name": "helper",
            "target_path": "web/util.ts",
            "target_qualified_name": "helper",
            "start_line": 4,
            "resolution": "imported_symbol",
        }
    ]


def test_collect_codebase_events_indexes_go_rust_and_java_same_file_call_sites(tmp_path: Path) -> None:
    go_file = tmp_path / "cmd" / "server.go"
    go_file.parent.mkdir()
    go_file.write_text(
        "package main\n\n"
        "func helper() {}\n"
        "func Start() {\n"
        "    helper()\n"
        "}\n",
        encoding="utf-8",
    )
    rust_file = tmp_path / "src" / "lib.rs"
    rust_file.parent.mkdir()
    rust_file.write_text(
        "fn helper() {}\n"
        "pub fn start() {\n"
        "    helper();\n"
        "}\n",
        encoding="utf-8",
    )
    java_file = tmp_path / "src" / "App.java"
    java_file.write_text(
        "class App {\n"
        "  void helper() {}\n"
        "  void start() {\n"
        "    helper();\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    events = collect_codebase_events(tmp_path)

    calls = [event["payload"] for event in events if event["event_type"] == "code.call.indexed"]
    assert {
        (
            call["path"],
            call["language"],
            call["caller"],
            call["callee"],
            call["target_path"],
            call["target_qualified_name"],
            call["start_line"],
            call["resolution"],
        )
        for call in calls
    } == {
        ("cmd/server.go", "go", "Start", "helper", "cmd/server.go", "helper", 5, "same_file_symbol"),
        ("src/lib.rs", "rust", "start", "helper", "src/lib.rs", "helper", 3, "same_file_symbol"),
        ("src/App.java", "java", "start", "helper", "src/App.java", "helper", 4, "same_file_symbol"),
    }


def test_collect_codebase_events_resolves_go_cross_file_package_calls(tmp_path: Path) -> None:
    main_file = tmp_path / "cmd" / "server.go"
    main_file.parent.mkdir()
    main_file.write_text(
        "package main\n\n"
        "import \"example.com/project/pkg/worker\"\n\n"
        "func Start() {\n"
        "    worker.Run()\n"
        "}\n",
        encoding="utf-8",
    )
    worker_file = tmp_path / "pkg" / "worker" / "worker.go"
    worker_file.parent.mkdir(parents=True)
    worker_file.write_text(
        "package worker\n\n"
        "func Run() {}\n",
        encoding="utf-8",
    )

    events = collect_codebase_events(tmp_path)

    calls = [event["payload"] for event in events if event["event_type"] == "code.call.indexed"]
    assert {
        (
            call["path"],
            call["language"],
            call["caller"],
            call["callee"],
            call["callee_qualified_name"],
            call["target_path"],
            call["target_qualified_name"],
            call["start_line"],
            call["resolution"],
        )
        for call in calls
    } == {
        (
            "cmd/server.go",
            "go",
            "Start",
            "Run",
            "worker.Run",
            "pkg/worker/worker.go",
            "Run",
            6,
            "imported_symbol",
        )
    }


def test_collect_codebase_events_resolves_rust_cross_file_use_calls(tmp_path: Path) -> None:
    lib = tmp_path / "src" / "lib.rs"
    lib.parent.mkdir()
    lib.write_text(
        "mod worker;\n"
        "use crate::worker::run_worker;\n\n"
        "pub fn start() {\n"
        "    run_worker();\n"
        "}\n",
        encoding="utf-8",
    )
    worker = tmp_path / "src" / "worker.rs"
    worker.write_text("pub fn run_worker() {}\n", encoding="utf-8")

    events = collect_codebase_events(tmp_path)

    calls = [event["payload"] for event in events if event["event_type"] == "code.call.indexed"]
    assert {
        (
            call["path"],
            call["language"],
            call["caller"],
            call["callee"],
            call["callee_qualified_name"],
            call["target_path"],
            call["target_qualified_name"],
            call["start_line"],
            call["resolution"],
        )
        for call in calls
    } == {
        (
            "src/lib.rs",
            "rust",
            "start",
            "run_worker",
            "run_worker",
            "src/worker.rs",
            "run_worker",
            5,
            "imported_symbol",
        )
    }


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
