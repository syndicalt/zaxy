from __future__ import annotations

from datetime import date
from pathlib import Path

from zaxy.dashboard import (
    DashboardApp,
    DashboardConfig,
    EventloomDashboardGraphProvider,
    FallbackDashboardGraphProvider,
    Neo4jDashboardGraphProvider,
    UnavailableGraphProvider,
    build_dashboard_graph_provider,
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

    status_code, _headers, body = app.handle_api("GET", "/api/sessions", "")
    assert status_code == 200
    assert body["sessions"][0]["session_id"] == "default"

    status_code, _headers, body = app.handle_api("GET", "/api/events", "limit=not-a-number")
    assert status_code == 200
    assert len(body["events"]) == 1


def test_default_dashboard_graph_uses_eventloom_when_neo4j_is_absent(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    log = EventLog(workspace / ".eventloom" / "default.jsonl")
    first = log.append("session.genesis", actor="zaxy", payload={"session_id": "default"})
    second = log.append(
        "decision.recorded", actor="tester", payload={"decision": "Use fallback graph."}
    )
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/graph/summary", "")

    assert status_code == 200
    assert body["graph"]["available"] is True
    assert body["graph"]["source"] == "eventloom"
    assert body["graph"]["nodes"] == 2
    assert body["graph"]["edges"] == 1
    assert body["graph"]["elements"]["nodes"][0]["id"] == f"event:default:{first.seq}"
    assert body["graph"]["elements"]["nodes"][1]["id"] == f"event:default:{second.seq}"
    assert body["graph"]["elements"]["edges"][0]["label"] == "NEXT_EVENT"


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


def test_unavailable_dashboard_graph_provider_reports_degraded_payloads() -> None:
    provider = UnavailableGraphProvider()

    assert provider.summary(session_id="agent-1") == {
        "available": False,
        "session_id": "agent-1",
        "nodes": 0,
        "edges": 0,
        "warning": "graph provider unavailable",
    }
    assert provider.neighborhood(
        session_id="agent-1",
        node_id="n1",
        view="memory",
        hops=2,
        limit=10,
    ) == {
        "available": False,
        "session_id": "agent-1",
        "node_id": "n1",
        "view": "memory",
        "hops": 2,
        "limit": 10,
        "nodes": [],
        "edges": [],
        "omitted_nodes": 0,
        "omitted_edges": 0,
        "warning": "graph provider unavailable",
    }
    assert provider.search(session_id="agent-1", query="decision", view="memory", limit=10)[
        "available"
    ] is False
    assert provider.path_to_event(session_id="agent-1", node_id="n1", limit=10)[
        "available"
    ] is False


def test_fallback_dashboard_graph_provider_uses_eventloom_when_primary_is_unavailable(
    tmp_path: Path,
) -> None:
    log = EventLog(tmp_path / ".eventloom" / "default.jsonl")
    event = log.append("decision.recorded", actor="tester", payload={"decision": "Use fallback."})
    fallback = EventloomDashboardGraphProvider(tmp_path / ".eventloom")
    provider = FallbackDashboardGraphProvider(UnavailableGraphProvider(), fallback)

    summary = provider.summary(session_id="default")
    assert summary["source"] == "eventloom"
    assert summary["nodes"] == 1

    neighborhood = provider.neighborhood(
        session_id="default",
        node_id=f"event:default:{event.seq}",
        view="provenance",
        hops=1,
        limit=5,
    )
    assert neighborhood["nodes"][0]["id"] == f"event:default:{event.seq}"

    search = provider.search(session_id="default", query="fallback", view="provenance", limit=5)
    assert search["nodes"][0]["id"] == f"event:default:{event.seq}"

    path = provider.path_to_event(session_id="default", node_id=f"event:default:{event.seq}", limit=5)
    assert path["nodes"][0]["id"] == f"event:default:{event.seq}"


def test_fallback_dashboard_graph_provider_keeps_available_primary_results() -> None:
    primary = FakeGraphProvider()
    provider = FallbackDashboardGraphProvider(primary, UnavailableGraphProvider())

    assert provider.summary(session_id="agent-1") == {"nodes": 2, "edges": 1}
    assert provider.neighborhood(
        session_id="agent-1",
        node_id="n1",
        view="memory",
        hops=1,
        limit=5,
    )["nodes"] == [{"id": "n1"}]
    assert provider.search(session_id="agent-1", query="decision", view="memory", limit=5)[
        "nodes"
    ] == [{"id": "n1", "label": "decision"}]
    assert provider.path_to_event(session_id="agent-1", node_id="n1", limit=5)["nodes"] == [
        {"id": "n1"}
    ]


def test_build_dashboard_graph_provider_uses_eventloom_without_neo4j_credentials(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    scope = resolve_dashboard_scope(DashboardConfig(workspace=workspace))

    provider = build_dashboard_graph_provider(scope)

    assert isinstance(provider, EventloomDashboardGraphProvider)


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


def test_dashboard_graph_routes_validate_required_parameters(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/graph/neighborhood", "")
    assert status_code == 400
    assert body["error"] == "node_id_required"

    status_code, _headers, body = app.handle_api("GET", "/api/graph/search", "")
    assert status_code == 400
    assert body["error"] == "query_required"

    status_code, _headers, body = app.handle_api("GET", "/api/graph/path-to-event", "")
    assert status_code == 400
    assert body["error"] == "node_id_required"

    status_code, _headers, body = app.handle_api("GET", "/api/unknown", "")
    assert status_code == 404
    assert body["error"] == "not_found"


def test_dashboard_checkout_endpoint_is_read_only_placeholder(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/checkout", "query=hello")

    assert status_code == 200
    assert body["checkout"]["available"] is False
    assert body["checkout"]["query"] == "hello"


def test_neo4j_dashboard_provider_uses_direct_reads_without_transaction_retry() -> None:
    assert "execute_read" not in Neo4jDashboardGraphProvider.summary.__code__.co_names
    assert "execute_read" not in Neo4jDashboardGraphProvider.neighborhood.__code__.co_names
    assert "execute_read" not in Neo4jDashboardGraphProvider.search.__code__.co_names
    assert "execute_read" not in Neo4jDashboardGraphProvider.path_to_event.__code__.co_names


def test_eventloom_graph_provider_handles_bad_or_cross_session_node_ids(tmp_path: Path) -> None:
    log = EventLog(tmp_path / ".eventloom" / "default.jsonl")
    log.append("decision.recorded", actor="tester", payload={"decision": "Use fallback graph."})
    provider = EventloomDashboardGraphProvider(tmp_path / ".eventloom")

    fallback_summary = provider.neighborhood(
        session_id="default",
        node_id="not-an-event-node",
        view="provenance",
        hops=1,
        limit=5,
    )
    assert fallback_summary["source"] == "eventloom"
    assert fallback_summary["nodes"] == 1

    cross_session = provider.neighborhood(
        session_id="agent-1",
        node_id="event:default:1",
        view="provenance",
        hops=1,
        limit=5,
    )
    assert cross_session == {"available": True, "source": "eventloom", "nodes": [], "edges": []}


def test_neo4j_dashboard_summary_returns_renderable_elements() -> None:
    class FakeNeo4jDateTime:
        def iso_format(self) -> str:
            return "2026-05-16T00:00:00Z"

    class FakeNode:
        element_id = "n1"
        labels = ["Entity"]

        def items(self) -> list[tuple[str, object]]:
            return [
                ("name", "Decision"),
                ("created_at", FakeNeo4jDateTime()),
                ("business_date", date(2026, 5, 16)),
                ("tags", [date(2026, 5, 17)]),
                ("metadata", {"reviewed_at": date(2026, 5, 18)}),
            ]

    class FakeCountRecord(dict[str, object]):
        pass

    class FakePathRecord(dict[str, object]):
        pass

    class FakeSession:
        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, query: str, **_params: object) -> object:
            if "count(n)" in query:
                return _FakeResult([FakeCountRecord(count=1)])
            if "count(r)" in query:
                return _FakeResult([FakeCountRecord(count=0)])
            return _FakeResult([FakePathRecord(n=FakeNode())])

    class FakeDriver:
        def session(self) -> FakeSession:
            return FakeSession()

    provider = Neo4jDashboardGraphProvider.__new__(Neo4jDashboardGraphProvider)
    provider._driver = FakeDriver()

    result = provider.summary(session_id="agent-1")

    assert result["source"] == "neo4j"
    assert result["nodes"] == 1
    assert result["edges"] == 0
    assert result["elements"] == {
        "nodes": [
            {
                "id": "n1",
                "label": "Decision",
                "labels": ["Entity"],
                "properties": {
                    "business_date": "2026-05-16",
                    "created_at": "2026-05-16T00:00:00Z",
                    "metadata": {"reviewed_at": "2026-05-18"},
                    "name": "Decision",
                    "tags": ["2026-05-17"],
                },
            }
        ],
        "edges": [],
    }


def test_neo4j_dashboard_provider_returns_degraded_payload_on_driver_error() -> None:
    class FailingDriver:
        def session(self) -> object:
            raise RuntimeError("neo4j offline")

    provider = Neo4jDashboardGraphProvider.__new__(Neo4jDashboardGraphProvider)
    provider._driver = FailingDriver()

    result = provider.summary(session_id="agent-1")

    assert result["available"] is False
    assert result["warning"] == "neo4j offline"


def test_neo4j_dashboard_summary_accepts_record_get_paths() -> None:
    class FakeNode:
        def __init__(self, element_id: str, name: str) -> None:
            self.element_id = element_id
            self.labels = ["Entity"]
            self.name = name

        def items(self) -> list[tuple[str, object]]:
            return [("name", self.name)]

    class FakeRelationship:
        element_id = "r1"
        type = "RELATES"

        def __init__(self, start_node: FakeNode, end_node: FakeNode) -> None:
            self.start_node = start_node
            self.end_node = end_node

        def items(self) -> list[tuple[str, object]]:
            return []

    class FakePath:
        def __init__(self) -> None:
            start = FakeNode("n1", "Start")
            end = FakeNode("n2", "End")
            self.nodes = [start, end]
            self.relationships = [FakeRelationship(start, end)]

    class FakeRecord:
        def __contains__(self, _key: object) -> bool:
            return False

        def get(self, key: str) -> object:
            if key == "p":
                return FakePath()
            raise KeyError(key)

    class FakeCountRecord(dict[str, object]):
        pass

    class FakeSession:
        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, query: str, **_params: object) -> object:
            if "count(n)" in query:
                return _FakeResult([FakeCountRecord(count=2)])
            if "count(r)" in query:
                return _FakeResult([FakeCountRecord(count=1)])
            return _FakeResult([FakeRecord()])

    class FakeDriver:
        def session(self) -> FakeSession:
            return FakeSession()

    provider = Neo4jDashboardGraphProvider.__new__(Neo4jDashboardGraphProvider)
    provider._driver = FakeDriver()

    result = provider.summary(session_id=None)

    assert result["elements"]["edges"] == [
        {
            "id": "r1",
            "label": "RELATES",
            "properties": {},
            "source": "n1",
            "target": "n2",
            "type": "RELATES",
        }
    ]


def test_neo4j_dashboard_neighborhood_path_and_close_use_driver() -> None:
    class FakeNode:
        def __init__(self, element_id: str, name: str) -> None:
            self.element_id = element_id
            self.labels = ["Entity"]
            self.name = name

        def items(self) -> list[tuple[str, object]]:
            return [("name", self.name)]

    class FakeRelationship:
        element_id = "r1"
        type = "RELATES"

        def __init__(self, start_node: FakeNode, end_node: FakeNode) -> None:
            self.start_node = start_node
            self.end_node = end_node

        def items(self) -> list[tuple[str, object]]:
            return [("confidence", 0.9)]

    class FakePath:
        def __init__(self) -> None:
            start = FakeNode("n1", "Start")
            end = FakeNode("n2", "End")
            self.nodes = [start, end]
            self.relationships = [FakeRelationship(start, end)]

    class FakeSession:
        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, _query: str, **_params: object) -> object:
            return _FakeResult([{"p": FakePath()}])

    class FakeDriver:
        def __init__(self) -> None:
            self.closed = False

        def session(self) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            self.closed = True

    driver = FakeDriver()
    provider = Neo4jDashboardGraphProvider.__new__(Neo4jDashboardGraphProvider)
    provider._driver = driver

    neighborhood = provider.neighborhood(
        session_id="agent-1",
        node_id="n1",
        view="memory",
        hops=2,
        limit=5,
    )
    path = provider.path_to_event(session_id="agent-1", node_id="n1", limit=5)
    provider.close()

    assert neighborhood["nodes"][0]["label"] == "Start"
    assert neighborhood["edges"][0]["properties"] == {"confidence": 0.9}
    assert path["edges"][0]["type"] == "RELATES"
    assert driver.closed is True


def test_neo4j_dashboard_search_avoids_driver_query_parameter_collision() -> None:
    class FakeNode:
        element_id = "n1"
        labels = ["Entity"]

        def items(self) -> list[tuple[str, object]]:
            return [("name", "Decision")]

    class FakeSession:
        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, query: str, **params: object) -> object:
            assert "query" not in params
            assert params["search_text"] == "decision"
            return _FakeResult([{"n": FakeNode()}])

    class FakeDriver:
        def session(self) -> FakeSession:
            return FakeSession()

    provider = Neo4jDashboardGraphProvider.__new__(Neo4jDashboardGraphProvider)
    provider._driver = FakeDriver()

    result = provider.search(session_id=None, query="decision", view="memory", limit=5)

    assert result["available"] is True
    assert result["source"] == "neo4j"
    assert result["nodes"][0]["label"] == "Decision"


class _FakeResult:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def single(self) -> dict[str, object] | None:
        return self.records[0] if self.records else None

    def __iter__(self) -> object:
        return iter(self.records)


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
    assert "/api/graph/search" in html
    assert "/api/graph/neighborhood" in html
    assert "graph-search" in html
    assert "graph-detail" in html
    assert "expandSelectedNode" in html
    assert 'cy.on("tap", "node"' in html
    assert "refreshGraph().catch" in html
    assert "cy.add" in html
    assert "cytoscape" in html.lower()


def test_eventloom_graph_provider_supports_search(tmp_path: Path) -> None:
    log = EventLog(tmp_path / ".eventloom" / "default.jsonl")
    log.append("decision.recorded", actor="tester", payload={"decision": "Use fallback graph."})
    provider = EventloomDashboardGraphProvider(tmp_path / ".eventloom")

    result = provider.search(session_id=None, query="fallback", view="provenance", limit=5)

    assert result["available"] is True
    assert result["source"] == "eventloom"
    assert result["nodes"] == [
        {
            "id": "event:default:1",
            "label": "decision.recorded #1",
            "kind": "event",
            "properties": {
                "actor": "tester",
                "hash": log.read_all()[0].hash,
                "seq": 1,
                "session_id": "default",
                "summary": "Use fallback graph.",
                "timestamp": log.read_all()[0].timestamp,
                "type": "decision.recorded",
            },
        }
    ]


def test_eventloom_graph_provider_reads_single_log_files_and_empty_payloads(tmp_path: Path) -> None:
    log_path = tmp_path / "agent.jsonl"
    log = EventLog(log_path)
    log.append("heartbeat", actor="tester", payload={})
    provider = EventloomDashboardGraphProvider(log_path)

    summary = provider.summary(session_id=None)
    search = provider.search(session_id=None, query="heartbeat", view="provenance", limit=5)

    assert summary["nodes"] == 1
    assert search["nodes"][0]["properties"]["summary"] == ""


def test_dashboard_handler_keeps_app_reference(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    dashboard_app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))
    handler_cls = create_dashboard_handler(dashboard_app)

    assert handler_cls.dashboard_app.scope.workspace == workspace.resolve()
