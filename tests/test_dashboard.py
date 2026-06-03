from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zaxy.coordination import CoordinationManager
from zaxy.dashboard import (
    DashboardApp,
    DashboardConfig,
    EmbeddedDashboardGraphProvider,
    EventloomDashboardGraphProvider,
    FallbackDashboardGraphProvider,
    Neo4jDashboardGraphProvider,
    ProjectionDashboardGraphProvider,
    UnavailableGraphProvider,
    build_dashboard_graph_provider,
    create_dashboard_handler,
    render_dashboard_html,
    resolve_dashboard_scope,
)
from zaxy.embedded_graph_store import EmbeddedGraphStore
from zaxy.event import EventLog
from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult


def test_dashboard_scope_defaults_to_current_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()

    scope = resolve_dashboard_scope(DashboardConfig(workspace=workspace))

    assert scope.workspace == workspace.resolve()
    assert scope.eventloom_path == workspace.resolve() / ".eventloom"
    assert scope.host == "127.0.0.1"
    assert scope.port == 8765
    assert scope.projection_backend == "embedded"
    assert scope.embedded_graph_path == workspace.resolve() / ".eventloom" / "projections" / "embedded.kuzu"
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


def test_dashboard_scope_accepts_pggraph_backend(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()

    scope = resolve_dashboard_scope(
        DashboardConfig(
            workspace=workspace,
            projection_backend="pggraph",
            pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        )
    )

    assert scope.projection_backend == "pggraph"
    assert scope.pggraph_dsn == "postgresql://postgres:postgres@localhost:5432/zaxy"


def test_dashboard_scope_accepts_latticedb_candidate_backend(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()

    scope = resolve_dashboard_scope(DashboardConfig(workspace=workspace, projection_backend="latticedb"))

    assert scope.projection_backend == "latticedb"


def test_dashboard_scope_rejects_unknown_projection_backend(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()

    with pytest.raises(ValueError, match="projection backend"):
        resolve_dashboard_scope(DashboardConfig(workspace=workspace, projection_backend="pggrph"))


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


def test_dashboard_surfaces_memory_persistence_status(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    log = EventLog(workspace / ".eventloom" / "default.jsonl")
    log.append("memory.bootstrap.shown", actor="zaxy-memory", payload={"source": "cli"})
    log.append("memory.checkout.completed", actor="zaxy-memory", payload={"source": "cli"})
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/status", "")
    assert status_code == 200
    assert body["memory_persistence"]["last_bootstrap_seq"] == 1
    assert body["memory_persistence"]["last_checkout_seq"] == 2
    assert body["memory_persistence"]["stale"] is False

    status_code, _headers, body = app.handle_api("GET", "/api/memory-persistence", "")
    assert status_code == 200
    assert body["memory_persistence"]["last_checkout_seq"] == 2


def test_dashboard_surfaces_purpose_control_plane_without_graph_backend(tmp_path: Path) -> None:
    """Purpose dashboard APIs should summarize replay state without graph services."""
    workspace = tmp_path / "project"
    eventloom = workspace / ".eventloom"
    workspace.mkdir()
    log = EventLog(eventloom / "default.jsonl")
    log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        thread="default",
        payload={
            "purpose": {"profile": "support", "evidence_policy": "customer_thread_and_current_status_required"},
            "retention": {"purpose_policy": {"suppressed_count": 1, "suppressed_reasons": {"stale_status": 1}}},
            "diagnostics": {
                "evidence_policy": {
                    "status": "missing",
                    "missing": ["current_status"],
                    "suggested_queries": ["refresh support status"],
                }
            },
            "quality": {"required_action": {"type": "memory_checkout", "query": "refresh support status"}},
        },
    )
    log.append(
        "memory.feedback",
        actor="assistant",
        thread="default",
        payload={"purpose": {"profile": "support"}, "citation": "event:default:1", "feedback": "rejected"},
    )
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/purpose/status", "")
    lanes_code, _lanes_headers, lanes_body = app.handle_api("GET", "/api/purpose/lanes", "")
    feedback_code, _feedback_headers, feedback_body = app.handle_api(
        "GET",
        "/api/purpose/feedback",
        "profile=support&outcome=negative",
    )

    assert status_code == 200
    assert body["purpose"]["active_profile"] == "support"
    assert body["purpose"]["suppression"]["count"] == 1
    assert body["purpose"]["refresh_suggestions"][0]["query"] == "refresh support status"
    assert lanes_code == 200
    assert lanes_body["purpose_lanes"]["lanes"][0]["profile"] == "support"
    assert feedback_code == 200
    assert feedback_body["purpose_feedback"]["targets"][0]["negative_count"] == 1


def test_dashboard_surfaces_memory_activation_status(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    log = EventLog(workspace / ".eventloom" / "default.jsonl")
    checkout = log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={"source": "dashboard-test"},
    )
    capture = log.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "codex", "role": "assistant"},
    )
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/status", "")

    assert status_code == 200
    assert body["memory_activation"]["status"] == "ok"
    assert body["memory_activation"]["message"] == "Latest memory checkout is fresh"
    assert body["memory_activation"]["latest_checkout"]["seq"] == checkout.seq
    assert body["memory_activation"]["latest_capture"]["seq"] == capture.seq
    assert body["memory_activation"]["actions"] == []


def test_dashboard_warns_when_memory_activation_is_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    log = EventLog(workspace / ".eventloom" / "default.jsonl")
    capture = log.append(
        "command.completed",
        actor="zaxy-observer",
        payload={"source": "codex", "command": "pytest"},
    )
    reminder = log.append(
        "memory.reminder.suggested",
        actor="zaxy-memory",
        payload={"recommended_tool": "memory_checkout", "source": "codex"},
    )
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/status", "")

    assert status_code == 200
    assert body["memory_activation"]["status"] == "warning"
    assert body["memory_activation"]["message"] == "No memory checkout events found"
    assert body["memory_activation"]["latest_checkout"] is None
    assert body["memory_activation"]["latest_capture"]["seq"] == capture.seq
    assert body["memory_activation"]["latest_reminder"]["seq"] == reminder.seq
    assert body["memory_activation"]["actions"] == [
        "Run memory checkout before relying on Zaxy context.",
    ]


def test_dashboard_surfaces_activation_efficiency_metric(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    ready = EventLog(workspace / ".eventloom" / "ready.jsonl")
    ready.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={
            "source": "dashboard-test",
            "token_efficiency": {
                "prompt_tokens": 160,
                "current_fact_count": 2,
                "evidence_count": 3,
                "facts_per_1k_prompt_tokens": 12.5,
            },
        },
        thread="ready",
    )
    ready.append("transcript.turn", actor="assistant", payload={"source": "codex", "role": "assistant"}, thread="ready")
    missing = EventLog(workspace / ".eventloom" / "missing.jsonl")
    missing.append("command.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="missing")
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/status", "")

    assert status_code == 200
    efficiency = body["memory_activation"]["activation_efficiency"]
    assert efficiency["high_context_session_count"] == 2
    assert efficiency["fresh_checkout_session_count"] == 1
    assert efficiency["fresh_checkout_rate"] == 0.5
    assert {session["status"] for session in efficiency["sessions"]} == {
        "fresh_checkout",
        "missing_checkout",
    }
    ready_session = next(session for session in efficiency["sessions"] if session["session_id"] == "ready")
    assert ready_session["checkout"]["token_efficiency"] == {
        "prompt_tokens": 160,
        "current_fact_count": 2,
        "evidence_count": 3,
        "facts_per_1k_prompt_tokens": 12.5,
    }


def test_dashboard_shell_shows_memory_persistence_metrics() -> None:
    html = render_dashboard_html()

    assert "Last checkout" in html
    assert "Last feedback" in html
    assert "Last bootstrap" in html
    assert "memory-persistence-warning" in html
    assert "Activation" in html
    assert "Activation rate" in html
    assert "metric-activation-rate" in html
    assert "Checkout tokens" in html
    assert "metric-checkout-tokens" in html
    assert "Facts / 1k tokens" in html
    assert "metric-checkout-facts-per-token" in html
    assert "Latest capture" in html
    assert "Latest reminder" in html
    assert "metric-latest-reminder" in html
    assert "memory-activation-warning" in html
    assert "checkout-query" in html
    assert "Run checkout" in html
    assert "checkout-json" in html
    assert 'data-tab="purpose"' in html
    assert "/api/purpose/status" in html
    assert "purpose-lanes-body" in html


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

    assert isinstance(provider, FallbackDashboardGraphProvider)
    assert isinstance(provider.primary, EmbeddedDashboardGraphProvider)
    assert isinstance(provider.fallback, EventloomDashboardGraphProvider)


def test_build_dashboard_graph_provider_can_use_pggraph_projection_store(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    scope = resolve_dashboard_scope(
        DashboardConfig(
            workspace=workspace,
            projection_backend="pggraph",
            pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        )
    )

    provider = build_dashboard_graph_provider(scope)

    assert isinstance(provider, FallbackDashboardGraphProvider)
    assert isinstance(provider.primary, ProjectionDashboardGraphProvider)


@pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")
def test_build_dashboard_graph_provider_can_use_embedded_projection_store(
    tmp_path: Path,
) -> None:
    """Dashboard graph routes should read the actual embedded projection, not Eventloom fallback."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    graph_path = tmp_path / "embedded.kuzu"

    async def project() -> None:
        store = EmbeddedGraphStore(graph_path)
        await store.connect()
        await store.init_schema()
        await store.upsert_extraction(
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Embedded Goal",
                        entity_type="goal",
                        observed_at="2026-05-20T01:00:00Z",
                        summary="ship embedded graph",
                    ),
                    ExtractedEntity(
                        name="Embedded Task",
                        entity_type="task",
                        observed_at="2026-05-20T01:00:00Z",
                        summary="wire dashboard",
                    ),
                ],
                edges=[
                    ExtractedEdge(
                        source="Embedded Task",
                        target="Embedded Goal",
                        relation_type="supports",
                        valid_from="2026-05-20T01:00:00Z",
                    )
                ],
                source_event_seq=7,
                source_event_hash="hash-7",
                source_event_type="task.proposed",
                source_thread="agent-1",
            ),
            session_id="agent-1",
        )
        await store.close()

    asyncio.run(project())
    scope = resolve_dashboard_scope(
        DashboardConfig(
            workspace=workspace,
            session_id="agent-1",
            projection_backend="embedded",
            embedded_graph_path=graph_path,
        )
    )

    provider = build_dashboard_graph_provider(scope)

    assert isinstance(provider, FallbackDashboardGraphProvider)
    summary = provider.summary(session_id="agent-1")
    assert summary["source"] == "embedded"
    assert summary["nodes"] == 2
    assert summary["edges"] == 1
    assert {node["label"] for node in summary["elements"]["nodes"]} == {
        "Embedded Goal",
        "Embedded Task",
    }
    assert summary["elements"]["edges"][0]["type"] == "supports"

    app = DashboardApp(scope, graph_provider=provider)
    status_code, _headers, body = app.handle_api("GET", "/api/graph/search", "q=dashboard")
    assert status_code == 200
    assert body["graph"]["source"] == "embedded"
    assert body["graph"]["nodes"][0]["label"] == "Embedded Task"

    status_code, _headers, body = app.handle_api(
        "GET",
        "/api/graph/neighborhood",
        "node_id=Embedded%20Task&hops=1",
    )
    assert status_code == 200
    assert body["graph"]["source"] == "embedded"
    assert body["graph"]["edges"][0]["type"] == "supports"

    task_node = next(node for node in summary["elements"]["nodes"] if node["label"] == "Embedded Task")
    status_code, _headers, body = app.handle_api(
        "GET",
        "/api/graph/path-to-event",
        f"node_id={task_node['id']}",
    )
    assert status_code == 200
    assert body["graph"]["source"] == "embedded"
    assert body["graph"]["nodes"][1]["id"] == "event:agent-1:7"


@pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")
def test_embedded_dashboard_graph_provider_reports_empty_uninitialized_projection(
    tmp_path: Path,
) -> None:
    """Bare embedded dashboard graph routes should not fall back to Eventloom projection."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    log = EventLog(workspace / ".eventloom" / "agent-1.jsonl")
    log.append("hook.heartbeat", actor="zaxy-init", payload={"source": "zaxy-init"}, thread="agent-1")
    scope = resolve_dashboard_scope(
        DashboardConfig(
            workspace=workspace,
            session_id="agent-1",
            projection_backend="embedded",
            embedded_graph_path=workspace / ".eventloom" / "projections" / "embedded.kuzu",
        )
    )

    provider = build_dashboard_graph_provider(scope)

    summary = provider.summary(session_id="agent-1")
    assert summary["available"] is True
    assert summary["source"] == "embedded"
    assert summary["nodes"] == 0
    assert summary["edges"] == 0
    assert summary["elements"] == {"nodes": [], "edges": []}

    app = DashboardApp(scope, graph_provider=provider)
    status_code, _headers, body = app.handle_api("GET", "/api/graph/search", "q=heartbeat")
    assert status_code == 200
    assert body["graph"]["source"] == "embedded"
    assert body["graph"]["nodes"] == []


def test_projection_dashboard_graph_provider_renders_pggraph_rows() -> None:
    class FakeStore:
        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def _fetch_all(
            self,
            sql: str,
            _params: dict[str, object],
        ) -> list[dict[str, object]]:
            if "node_count" in sql:
                return [{"node_count": 1, "edge_count": 1}]
            if "FROM zaxy_pggraph_entities" in sql:
                return [
                    {
                        "node_key": "node-1",
                        "name": "Checkout",
                        "entity_type": "memory",
                        "summary": "Memory Checkout",
                        "properties": {"confidence": 0.9},
                        "session_id": "agent-1",
                        "source_event_seq": 7,
                        "source_event_hash": "abc",
                        "valid_from": "2026-05-19T00:00:00Z",
                        "valid_to": None,
                    }
                ]
            return [
                {
                    "edge_key": "edge-1",
                    "source_node_key": "node-1",
                    "target_node_key": "node-2",
                    "relation_type": "RELATES",
                    "properties": {"weight": 1.0},
                    "session_id": "agent-1",
                    "source_event_seq": 7,
                    "source_event_hash": "abc",
                }
            ]

    provider = ProjectionDashboardGraphProvider.__new__(ProjectionDashboardGraphProvider)
    provider.backend = "pggraph"
    provider._store = FakeStore()

    summary = provider.summary(session_id="agent-1")

    assert summary["available"] is True
    assert summary["source"] == "pggraph"
    assert summary["nodes"] == 1
    assert summary["edges"] == 1
    assert summary["elements"]["nodes"][0]["id"] == "node-1"
    assert summary["elements"]["nodes"][0]["label"] == "Checkout"
    assert summary["elements"]["edges"][0]["type"] == "RELATES"


def test_projection_dashboard_graph_provider_reads_pggraph_views() -> None:
    node_row = {
        "node_key": "node-1",
        "name": "Checkout",
        "entity_type": "memory",
        "summary": "Memory Checkout",
        "properties": {"confidence": 0.9},
        "session_id": "agent-1",
        "source_event_seq": 7,
        "source_event_hash": "abc",
        "valid_from": "2026-05-19T00:00:00Z",
        "valid_to": None,
    }
    edge_row = {
        "edge_key": "edge-1",
        "source_node_key": "node-1",
        "target_node_key": "node-2",
        "relation_type": "RELATES",
        "properties": {"weight": 1.0},
        "session_id": "agent-1",
        "source_event_seq": 7,
        "source_event_hash": "abc",
    }

    class FakeStore:
        def __init__(self, responses: list[list[dict[str, object]]]) -> None:
            self.responses = responses
            self.closed = False

        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def _fetch_all(
            self,
            _sql: str,
            _params: dict[str, object],
        ) -> list[dict[str, object]]:
            return self.responses.pop(0)

    provider = ProjectionDashboardGraphProvider.__new__(ProjectionDashboardGraphProvider)
    provider.backend = "pggraph"
    provider._store = FakeStore([[node_row], [edge_row]])

    neighborhood = provider.neighborhood(
        session_id="agent-1",
        node_id="node-1",
        view="entity",
        hops=1,
        limit=10,
    )

    assert neighborhood["available"] is True
    assert neighborhood["source"] == "pggraph"
    assert neighborhood["nodes"][0]["id"] == "node-1"
    assert neighborhood["edges"][0]["type"] == "RELATES"
    assert provider._store.closed is True

    provider._store = FakeStore([[node_row]])
    search = provider.search(session_id="agent-1", query="checkout", view="entity", limit=10)

    assert search["nodes"][0]["label"] == "Checkout"
    assert search["edges"] == []

    provider._store = FakeStore([[node_row]])
    path = provider.path_to_event(session_id="agent-1", node_id="node-1", limit=10)

    assert path["nodes"][1]["id"] == "event:agent-1:7"
    assert path["edges"][0]["type"] == "SOURCE_EVENT"


def test_projection_dashboard_graph_provider_handles_missing_source_event() -> None:
    class FakeStore:
        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def _fetch_all(
            self,
            _sql: str,
            _params: dict[str, object],
        ) -> list[dict[str, object]]:
            return [
                {
                    "node_key": "node-1",
                    "name": "Checkout",
                    "entity_type": "memory",
                    "summary": "Memory Checkout",
                    "properties": {},
                    "session_id": "agent-1",
                    "source_event_seq": None,
                    "source_event_hash": None,
                    "valid_from": "2026-05-19T00:00:00Z",
                    "valid_to": None,
                }
            ]

    provider = ProjectionDashboardGraphProvider.__new__(ProjectionDashboardGraphProvider)
    provider.backend = "pggraph"
    provider._store = FakeStore()

    path = provider.path_to_event(session_id="agent-1", node_id="node-1", limit=10)

    assert path["nodes"][0]["id"] == "node-1"
    assert path["edges"] == []


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


def test_dashboard_checkout_endpoint_returns_read_only_checkout_diagnostics(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    log = EventLog(workspace / ".eventloom" / "default.jsonl")
    log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use embedded graph for local memory."},
    )
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/checkout", "query=embedded+memory&limit=3")

    assert status_code == 200
    assert body["checkout"]["available"] is True
    assert body["checkout"]["session_id"] == "default"
    assert body["checkout"]["query"] == "embedded memory"
    assert body["checkout"]["payload"]["query"] == "embedded memory"
    assert 0.0 <= body["checkout"]["payload"]["quality"]["confidence"] <= 1.0
    assert "diagnostics" in body["checkout"]["payload"]
    assert [event.type for event in EventLog(workspace / ".eventloom" / "default.jsonl").read_all()] == [
        "decision.recorded"
    ]


@patch("zaxy.dashboard.MemoryFabric")
def test_dashboard_checkout_passes_embedded_projection_path(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Dashboard checkout should use the same embedded projection path as graph routes."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    embedded_path = tmp_path / "embedded.kuzu"

    class FakeCheckout:
        def to_dict(self) -> dict[str, object]:
            return {"query": "embedded", "quality": {"confidence": 1.0}}

    class FakeFabric:
        async def checkout_memory(self, *_args: object, **_kwargs: object) -> FakeCheckout:
            return FakeCheckout()

        async def close(self) -> None:
            return None

    mock_fabric_cls.return_value = FakeFabric()
    app = DashboardApp(
        resolve_dashboard_scope(
            DashboardConfig(
                workspace=workspace,
                projection_backend="embedded",
                embedded_graph_path=embedded_path,
            )
        )
    )

    status_code, _headers, body = app.handle_api("GET", "/api/checkout", "query=embedded")

    assert status_code == 200
    assert body["checkout"]["available"] is True
    kwargs = mock_fabric_cls.call_args.kwargs
    assert kwargs["projection_backend"] == "embedded"
    assert kwargs["embedded_graph_path"] == embedded_path


def test_dashboard_coordination_mission_view_is_read_only_and_replay_backed(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    manager = CoordinationManager(eventloom_path=eventloom_path)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API auth failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    manager.review_finding("auth-main", finding.finding_id, status="accepted", actor="lead")
    manager.promote_finding("auth-main", finding.finding_id, actor="lead")
    before_events = {
        path.name: len(EventLog(path).read_all()) for path in sorted(eventloom_path.glob("*.jsonl"))
    }
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api(
        "GET",
        "/api/coordination/mission",
        "mission_id=auth-main&include_diagnostics=true",
    )

    assert status_code == 200
    assert body["coordination"]["available"] is True
    assert body["coordination"]["read_only"] is True
    assert body["coordination"]["mission_id"] == "auth-main"
    assert body["coordination"]["brief"]["objective"] == "Ship auth refactor"
    assert body["coordination"]["brief"]["workers"][0]["worker_id"] == "auth-api"
    assert body["coordination"]["checkout"]["accepted_findings"][0]["finding_id"] == finding.finding_id
    assert body["coordination"]["ledger"]["workers"][0]["accepted_findings"] == 1
    assert body["coordination"]["approval_packet"]["mission_id"] == "auth-main"
    after_events = {
        path.name: len(EventLog(path).read_all()) for path in sorted(eventloom_path.glob("*.jsonl"))
    }
    assert after_events == before_events


def test_dashboard_coordination_mission_view_requires_mission_id(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api("GET", "/api/coordination/mission", "")

    assert status_code == 400
    assert body["error"] == "mission_id_required"


def test_dashboard_coordinate_brief_ledger_and_approval_routes(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    manager = CoordinationManager(eventloom_path=workspace / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    stale = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Old flag-missing theory from a superseded branch.",
        actor="auth-api-agent",
        evidence=[
            {
                "kind": "transcript",
                "reference": "eventloom://old/events/3#abc",
                "stale": True,
                "superseded_by": "decision:jwks-cache",
            }
        ],
    )
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    brief_status, _headers, brief_body = app.handle_api(
        "GET",
        "/api/coordinate/brief",
        "mission_id=auth-main",
    )
    ledger_status, _headers, ledger_body = app.handle_api(
        "GET",
        "/api/coordinate/ledger",
        "mission_id=auth-main",
    )
    packet_status, _headers, packet_body = app.handle_api(
        "GET",
        "/api/coordinate/approval-packet",
        "mission_id=auth-main",
    )

    assert brief_status == 200
    assert brief_body["brief"]["stale_findings"][0]["finding_id"] == stale.finding_id
    assert ledger_status == 200
    assert ledger_body["ledger"]["workers"][0]["stale_claim_count"] == 1
    assert packet_status == 200
    assert packet_body["approval_packet"]["findings"][0]["stale"] is True


def test_dashboard_coordinate_review_export_route_is_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    manager = CoordinationManager(eventloom_path=eventloom_path)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
    )
    before_events = {
        path.name: len(EventLog(path).read_all()) for path in sorted(eventloom_path.glob("*.jsonl"))
    }
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api(
        "GET",
        "/api/coordinate/review-export",
        "mission_id=auth-main",
    )

    assert status_code == 200
    assert body["review_export"]["read_only"] is True
    assert body["review_export"]["packet"]["findings"][0]["finding_id"] == finding.finding_id
    assert "# Zaxy Coordinate Review: auth-main" in body["review_export"]["markdown"]
    after_events = {
        path.name: len(EventLog(path).read_all()) for path in sorted(eventloom_path.glob("*.jsonl"))
    }
    assert after_events == before_events


def test_dashboard_coordinate_review_route_requires_explicit_enablement(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    manager = CoordinationManager(eventloom_path=workspace / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API auth failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
    )
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status_code, _headers, body = app.handle_api(
        "POST",
        "/api/coordinate/review",
        f"mission_id=auth-main&finding_id={finding.finding_id}&status=accepted&promote=true",
    )

    assert status_code == 403
    assert body["error"] == "coordinate_review_disabled"
    assert manager.brief("auth-main").accepted_findings == []


def test_dashboard_coordinate_review_route_reviews_and_promotes_when_enabled(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    manager = CoordinationManager(eventloom_path=eventloom_path)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API auth failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    app = DashboardApp(
        resolve_dashboard_scope(
            DashboardConfig(workspace=workspace, coordinate_review_enabled=True)
        )
    )

    status_code, _headers, body = app.handle_api(
        "POST",
        "/api/coordinate/review",
        (
            f"mission_id=auth-main&finding_id={finding.finding_id}"
            "&status=accepted&promote=true&rationale=verified"
        ),
    )

    assert status_code == 200
    assert body["review"]["reviewed_count"] == 1
    assert body["review"]["promoted_count"] == 1
    refreshed = CoordinationManager(eventloom_path=eventloom_path).brief("auth-main")
    assert [item.finding_id for item in refreshed.accepted_findings] == [finding.finding_id]
    event_types = [event.type for event in EventLog(eventloom_path / "auth-main.jsonl").read_all()]
    assert "coordination.finding.reviewed" in event_types
    assert "coordination.finding.promoted" in event_types


def test_dashboard_coordinate_apply_approval_route_applies_json_decisions(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    manager = CoordinationManager(eventloom_path=eventloom_path)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    accepted = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API auth failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
    )
    rejected = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Unbacked claim that OAuth scope drift caused failures.",
        actor="auth-api-agent",
        evidence=[],
    )
    app = DashboardApp(
        resolve_dashboard_scope(
            DashboardConfig(workspace=workspace, coordinate_review_enabled=True)
        )
    )

    status_code, _headers, body = app.handle_api(
        "POST",
        "/api/coordinate/apply-approval",
        "",
        body=json.dumps(
            {
                "mission_id": "auth-main",
                "actor": "dashboard-reviewer",
                "decisions": [
                    {"finding_id": accepted.finding_id, "status": "accepted", "promote": True},
                    {"finding_id": rejected.finding_id, "status": "rejected", "promote": False},
                ],
            }
        ),
    )

    assert status_code == 200
    assert body["approval_result"]["reviewed_count"] == 2
    assert body["approval_result"]["promoted_count"] == 1
    refreshed = CoordinationManager(eventloom_path=eventloom_path).brief("auth-main")
    assert [item.finding_id for item in refreshed.accepted_findings] == [accepted.finding_id]
    assert [item.finding_id for item in refreshed.rejected_findings] == [rejected.finding_id]


def test_dashboard_coordinate_apply_approval_route_validates_json_decisions(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = DashboardApp(
        resolve_dashboard_scope(
            DashboardConfig(workspace=workspace, coordinate_review_enabled=True)
        )
    )

    status_code, _headers, body = app.handle_api(
        "POST",
        "/api/coordinate/apply-approval",
        "",
        body=json.dumps({"mission_id": "auth-main", "decisions": {"finding_id": "x"}}),
    )

    assert status_code == 400
    assert body["error"] == "invalid_decisions"


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
    assert "Coordinate" in html
    assert "Checkout" in html
    assert "Events" in html
    assert "/api/status" in html
    assert "/api/coordination/mission" in html
    assert "/api/coordinate/review-export" in html
    assert "/api/coordinate/review-finding" in html
    assert "/api/coordinate/apply-approval" in html
    assert "/api/graph/summary" in html
    assert "/api/graph/search" in html
    assert "/api/graph/neighborhood" in html
    assert "graph-search" in html
    assert "graph-detail" in html
    assert "expandSelectedNode" in html
    assert 'cy.on("tap", "node"' in html
    assert "refreshGraph().catch" in html
    assert "coordination-mission-id" in html
    assert "loadCoordinationMission" in html
    assert "metric-coordinate-workers" in html
    assert "metric-coordinate-stale" in html
    assert "coordinate-workers-body" in html
    assert "coordinate-findings-body" in html
    assert "renderCoordinationMission" in html
    assert "coordinate-review-export" in html
    assert "loadCoordinationReviewExport" in html
    assert "coordinate-review-status" in html
    assert "reviewFinding" in html
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
