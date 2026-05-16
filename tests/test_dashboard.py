from __future__ import annotations

from pathlib import Path

from zaxy.dashboard import (
    DashboardApp,
    DashboardConfig,
    create_dashboard_handler,
    render_dashboard_html,
    resolve_dashboard_scope,
)
from zaxy.event import EventLog


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


def test_dashboard_status_and_events_use_resolved_eventloom(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    eventloom = workspace / ".eventloom"
    log = EventLog(eventloom / "default.jsonl")
    event = log.append(
        "decision.recorded",
        actor="tester",
        payload={"decision": "Build dashboard."},
    )
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

    def search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "search",
                {"session_id": session_id, "query": query, "view": view, "limit": limit},
            )
        )
        return {"nodes": [{"id": "n1", "label": query}], "edges": []}

    def path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "path_to_event",
                {"session_id": session_id, "node_id": node_id, "limit": limit},
            )
        )
        return {"nodes": [{"id": node_id}], "edges": []}


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
    app = DashboardApp(
        resolve_dashboard_scope(DashboardConfig(workspace=workspace)), graph_provider=graph
    )

    status_code, _headers, body = app.handle_api(
        "GET",
        "/api/graph/neighborhood",
        "node_id=n1&view=temporal&hops=99&limit=5000",
    )

    assert status_code == 200
    assert body["graph"]["nodes"] == [{"id": "n1"}]
    assert graph.calls[0][1]["hops"] == 2
    assert graph.calls[0][1]["limit"] == 250


def test_dashboard_graph_search_and_path_are_bounded(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    graph = FakeGraphProvider()
    app = DashboardApp(
        resolve_dashboard_scope(DashboardConfig(workspace=workspace, session_id="agent-1")),
        graph_provider=graph,
    )

    status_code, _headers, body = app.handle_api(
        "GET",
        "/api/graph/search",
        "q=decision&view=memory&limit=5000",
    )
    assert status_code == 200
    assert body["graph"]["nodes"][0]["label"] == "decision"
    assert graph.calls[-1] == (
        "search",
        {"session_id": "agent-1", "query": "decision", "view": "memory", "limit": 250},
    )

    status_code, _headers, body = app.handle_api(
        "GET",
        "/api/graph/path-to-event",
        "node_id=n1&limit=5000",
    )
    assert status_code == 200
    assert body["graph"]["nodes"] == [{"id": "n1"}]
    assert graph.calls[-1] == (
        "path_to_event",
        {"session_id": "agent-1", "node_id": "n1", "limit": 250},
    )


def test_dashboard_checkout_endpoint_is_read_only_placeholder(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/checkout", "query=hello")

    assert status_code == 200
    assert body["checkout"]["available"] is False
    assert body["checkout"]["query"] == "hello"


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


def test_dashboard_handler_keeps_app_reference(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    dashboard_app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))
    handler_cls = create_dashboard_handler(dashboard_app)

    assert handler_cls.dashboard_app.scope.workspace == workspace.resolve()
