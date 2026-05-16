# Runtime Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only localhost dashboard for runtime memory/debugging with scoped status, events, and graph-inspection API endpoints.

**Architecture:** Add `src/zaxy/dashboard.py` with a dependency-light HTTP server built on Python's standard library. The first slice serves bundled HTML/CSS/JS plus read-only JSON endpoints backed by existing Eventloom memory status/log helpers and bounded graph adapter methods. Add `zaxy dashboard` CLI wiring without coupling the dashboard to MCP.

**Tech Stack:** Python 3.11 standard library HTTP server, Typer CLI, Eventloom helpers, Neo4j graph adapter interfaces, plain HTML/CSS/JS, Cytoscape.js loaded by the browser UI.

---

## File Structure

- Create `src/zaxy/dashboard.py`: scope resolution, API model serialization, read-only HTTP handler, static assets, and server runner.
- Modify `src/zaxy/__main__.py`: add the `zaxy dashboard` Typer command.
- Create `tests/test_dashboard.py`: unit tests for scope, API routing, read-only enforcement, events/status serialization, static UI references, and graph adapter limits.
- Optionally modify `src/zaxy/graph.py` in a later task only if the dashboard needs concrete read-only graph query methods beyond the test adapter interface.

## Task 1: Dashboard Scope And API Models

**Files:**
- Create: `src/zaxy/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Add tests that prove dashboard scope defaults to the active workspace and does not discover unrelated projects:

```python
from pathlib import Path

from zaxy.dashboard import DashboardConfig, resolve_dashboard_scope


def test_dashboard_scope_defaults_to_current_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()

    scope = resolve_dashboard_scope(DashboardConfig(workspace=workspace))

    assert scope.workspace == workspace.resolve()
    assert scope.eventloom_path == workspace.resolve() / ".eventloom"
    assert scope.host == "127.0.0.1"
    assert scope.port == 8765
    assert scope.read_only is True


def test_dashboard_scope_accepts_explicit_eventloom_and_session(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    custom_eventloom = tmp_path / "memory"
    workspace.mkdir()

    scope = resolve_dashboard_scope(
        DashboardConfig(
            workspace=workspace,
            eventloom_path=custom_eventloom,
            session_id="agent-1",
            domain="demo",
            host="localhost",
            port=9000,
        )
    )

    assert scope.workspace == workspace.resolve()
    assert scope.eventloom_path == custom_eventloom.resolve()
    assert scope.session_id == "agent-1"
    assert scope.domain == "demo"
    assert scope.host == "localhost"
    assert scope.port == 9000
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_dashboard.py::test_dashboard_scope_defaults_to_current_workspace tests/test_dashboard.py::test_dashboard_scope_accepts_explicit_eventloom_and_session -q`

Expected: import failure for `zaxy.dashboard`.

- [ ] **Step 3: Implement minimal scope models**

Create `DashboardConfig`, `DashboardScope`, and `resolve_dashboard_scope()`. The implementation resolves paths, defaults to `.eventloom`, defaults host to `127.0.0.1`, port to `8765`, and marks the scope read-only.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest tests/test_dashboard.py::test_dashboard_scope_defaults_to_current_workspace tests/test_dashboard.py::test_dashboard_scope_accepts_explicit_eventloom_and_session -q`

Expected: both tests pass.

## Task 2: Read-Only Eventloom API

**Files:**
- Modify: `src/zaxy/dashboard.py`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Add tests for `DashboardApp.handle_api()`:

```python
from zaxy.dashboard import DashboardApp, DashboardConfig, resolve_dashboard_scope
from zaxy.event import EventLog


def test_dashboard_status_and_events_use_resolved_eventloom(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    eventloom = workspace / ".eventloom"
    log = EventLog(eventloom / "default.jsonl")
    event = log.append("decision.recorded", actor="tester", payload={"decision": "Build dashboard."})
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, headers, body = app.handle_api("GET", "/api/status", "")
    assert status_code == 200
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert body["scope"]["workspace"] == str(workspace.resolve())
    assert body["memory"]["total_events"] == 1
    assert body["memory"]["sessions"][0]["latest_hash"] == event.hash

    status_code, _headers, body = app.handle_api("GET", "/api/events", "session_id=default&limit=5")
    assert status_code == 200
    assert body["events"][0]["type"] == "decision.recorded"
    assert body["events"][0]["summary"] == "Build dashboard."


def test_dashboard_rejects_non_get_api_methods(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("POST", "/api/events", "")

    assert status_code == 405
    assert body["error"] == "read_only"
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_dashboard.py::test_dashboard_status_and_events_use_resolved_eventloom tests/test_dashboard.py::test_dashboard_rejects_non_get_api_methods -q`

Expected: `DashboardApp` or `handle_api` missing.

- [ ] **Step 3: Implement status/events endpoints**

Add `DashboardApp.handle_api()` with:

- `GET /api/status`: scope metadata and `inspect_memory_status()`
- `GET /api/sessions`: session list from status
- `GET /api/events`: recent entries from `inspect_memory_log()`, supporting `session_id` and bounded `limit`
- non-GET rejection with `405`
- unknown route `404`

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest tests/test_dashboard.py::test_dashboard_status_and_events_use_resolved_eventloom tests/test_dashboard.py::test_dashboard_rejects_non_get_api_methods -q`

Expected: tests pass.

## Task 3: Graph API Contract

**Files:**
- Modify: `src/zaxy/dashboard.py`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Add a fake read-only graph provider and tests for scoped, bounded graph endpoints:

```python
class FakeGraphProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        self.calls.append(("summary", {"session_id": session_id}))
        return {"nodes": 2, "edges": 1}

    def neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "neighborhood",
                {
                    "session_id": session_id,
                    "node_id": node_id,
                    "view": view,
                    "hops": hops,
                    "limit": limit,
                },
            )
        )
        return {"nodes": [{"id": node_id}], "edges": [], "omitted_nodes": 0, "omitted_edges": 0}


def test_dashboard_graph_summary_uses_session_scope(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    graph = FakeGraphProvider()
    app = DashboardApp(
        resolve_dashboard_scope(DashboardConfig(workspace=workspace, session_id="agent-1")),
        graph_provider=graph,
    )

    status_code, _headers, body = app.handle_api("GET", "/api/graph/summary", "")

    assert status_code == 200
    assert body["graph"] == {"nodes": 2, "edges": 1}
    assert graph.calls == [("summary", {"session_id": "agent-1"})]


def test_dashboard_graph_neighborhood_enforces_bounds(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    graph = FakeGraphProvider()
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)), graph_provider=graph)

    status_code, _headers, body = app.handle_api(
        "GET",
        "/api/graph/neighborhood",
        "node_id=n1&view=temporal&hops=99&limit=5000",
    )

    assert status_code == 200
    assert body["graph"]["nodes"] == [{"id": "n1"}]
    assert graph.calls[0][1]["hops"] == 2
    assert graph.calls[0][1]["limit"] == 250
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_dashboard.py::test_dashboard_graph_summary_uses_session_scope tests/test_dashboard.py::test_dashboard_graph_neighborhood_enforces_bounds -q`

Expected: graph endpoints return `404`.

- [ ] **Step 3: Implement graph provider protocol and routes**

Add:

- `DashboardGraphProvider` protocol with `summary()` and `neighborhood()`
- `UnavailableGraphProvider` returning degraded graph state
- `GET /api/graph/summary`
- `GET /api/graph/neighborhood`
- bounds: `hops` clamped to `1..2`, `limit` clamped to `1..250`

- [ ] **Step 4: Run tests to verify GREEN**

Run: `pytest tests/test_dashboard.py::test_dashboard_graph_summary_uses_session_scope tests/test_dashboard.py::test_dashboard_graph_neighborhood_enforces_bounds -q`

Expected: tests pass.

## Task 4: Static Dashboard UI

**Files:**
- Modify: `src/zaxy/dashboard.py`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Add tests for the HTML shell:

```python
def test_dashboard_index_html_references_core_tabs_and_api() -> None:
    html = render_dashboard_html()

    assert "Runtime Dashboard" in html
    assert "Overview" in html
    assert "Sessions" in html
    assert "Graph" in html
    assert "Checkout" in html
    assert "Events" in html
    assert "/api/status" in html
    assert "/api/graph/summary" in html
    assert "cytoscape" in html.lower()
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_dashboard.py::test_dashboard_index_html_references_core_tabs_and_api -q`

Expected: `render_dashboard_html` missing.

- [ ] **Step 3: Implement static HTML renderer**

Add `render_dashboard_html()` returning a self-contained operational UI shell. Include a graph canvas placeholder, read-only badge, scope header, tab buttons, and JavaScript that fetches `/api/status`, `/api/events`, and `/api/graph/summary`.

- [ ] **Step 4: Run test to verify GREEN**

Run: `pytest tests/test_dashboard.py::test_dashboard_index_html_references_core_tabs_and_api -q`

Expected: test passes.

## Task 5: HTTP Server And CLI Command

**Files:**
- Modify: `src/zaxy/dashboard.py`
- Modify: `src/zaxy/__main__.py`
- Modify: `tests/test_dashboard.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Add tests that `create_dashboard_handler()` serves HTML and JSON without starting a real long-running server, and that the CLI exposes the command:

```python
from typer.testing import CliRunner

from zaxy.__main__ import app as cli_app
from zaxy.dashboard import create_dashboard_handler


def test_dashboard_handler_serves_index_and_api(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    handler_cls = create_dashboard_handler(
        DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))
    )

    assert handler_cls.dashboard_app.scope.workspace == workspace.resolve()


def test_dashboard_cli_help_exposes_localhost_default() -> None:
    result = CliRunner().invoke(cli_app, ["dashboard", "--help"])

    assert result.exit_code == 0
    assert "127.0.0.1" in result.output
    assert "8765" in result.output
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_dashboard.py::test_dashboard_handler_serves_index_and_api tests/test_cli.py::test_dashboard_cli_help_exposes_localhost_default -q`

Expected: handler or CLI command missing.

- [ ] **Step 3: Implement handler and CLI wiring**

Add:

- `create_dashboard_handler(app: DashboardApp) -> type[BaseHTTPRequestHandler]`
- `run_dashboard(scope: DashboardScope) -> None`
- `@app.command("dashboard")` in `src/zaxy/__main__.py`

The command accepts `--workspace`, `--eventloom-path`, `--session-id`, `--domain`, `--host`, and `--port`, then prints `Zaxy dashboard listening on http://host:port` before serving.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `pytest tests/test_dashboard.py tests/test_cli.py::test_dashboard_cli_help_exposes_localhost_default -q`

Expected: tests pass.

## Task 6: Quality Gate

**Files:**
- Review all changed files

- [ ] **Step 1: Run focused dashboard tests**

Run: `pytest tests/test_dashboard.py tests/test_cli.py::test_dashboard_cli_help_exposes_localhost_default -q`

Expected: all selected tests pass.

- [ ] **Step 2: Run lint**

Run: `ruff check src/zaxy/dashboard.py tests/test_dashboard.py src/zaxy/__main__.py tests/test_cli.py`

Expected: no lint failures.

- [ ] **Step 3: Run formatting check**

Run: `ruff format --check src/zaxy/dashboard.py tests/test_dashboard.py src/zaxy/__main__.py tests/test_cli.py`

Expected: no formatting changes needed.

- [ ] **Step 4: Run existing CLI and viewer regression tests**

Run: `pytest tests/test_viewer.py tests/test_memory_status.py tests/test_cli.py::test_viewer_command_writes_static_html -q`

Expected: tests pass.

- [ ] **Step 5: Commit implementation**

```bash
git add src/zaxy/dashboard.py src/zaxy/__main__.py tests/test_dashboard.py tests/test_cli.py docs/superpowers/plans/2026-05-16-runtime-dashboard.md
git commit -m "feat: add read-only runtime dashboard"
```
