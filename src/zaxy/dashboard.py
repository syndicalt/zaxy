"""Read-only local dashboard for runtime memory debugging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from neo4j import GraphDatabase

from zaxy.memory_status import inspect_memory_log, inspect_memory_status


@dataclass(frozen=True)
class DashboardConfig:
    """User-provided dashboard configuration before path resolution."""

    workspace: Path | None = None
    eventloom_path: Path | None = None
    session_id: str | None = None
    domain: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None


@dataclass(frozen=True)
class DashboardScope:
    """Resolved read-only dashboard scope."""

    workspace: Path
    eventloom_path: Path
    session_id: str | None
    domain: str | None
    host: str
    port: int
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    read_only: bool = True


class DashboardGraphProvider(Protocol):
    """Read-only graph data provider for dashboard routes."""

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        """Return a bounded graph summary for one session."""

    def neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        """Return a bounded graph neighborhood for one node."""

    def search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        """Search graph nodes in a bounded read-only result set."""

    def path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        """Return a bounded path from one graph node to Eventloom provenance."""


class UnavailableGraphProvider:
    """Graph provider used when Neo4j graph access is unavailable."""

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        """Return degraded graph summary metadata."""
        return {
            "available": False,
            "session_id": session_id,
            "nodes": 0,
            "edges": 0,
            "warning": "graph provider unavailable",
        }

    def neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        """Return an empty degraded neighborhood."""
        return {
            "available": False,
            "session_id": session_id,
            "node_id": node_id,
            "view": view,
            "hops": hops,
            "limit": limit,
            "nodes": [],
            "edges": [],
            "omitted_nodes": 0,
            "omitted_edges": 0,
            "warning": "graph provider unavailable",
        }

    def search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        """Return an empty degraded search result."""
        return {
            "available": False,
            "session_id": session_id,
            "query": query,
            "view": view,
            "limit": limit,
            "nodes": [],
            "edges": [],
            "warning": "graph provider unavailable",
        }

    def path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        """Return an empty degraded provenance path."""
        return {
            "available": False,
            "session_id": session_id,
            "node_id": node_id,
            "limit": limit,
            "nodes": [],
            "edges": [],
            "warning": "graph provider unavailable",
        }


class Neo4jDashboardGraphProvider:
    """Read-only Neo4j graph provider for the local dashboard."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        """Return node and edge counts for the selected graph scope."""
        try:
            with self._driver.session() as session:
                node_count = session.execute_read(self._count_nodes, session_id)
                edge_count = session.execute_read(self._count_edges, session_id)
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)
        return {"available": True, "nodes": node_count, "edges": edge_count}

    def neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        """Return a bounded neighborhood around one graph node."""
        try:
            with self._driver.session() as session:
                elements = session.execute_read(
                    self._neighborhood,
                    session_id,
                    node_id,
                    view,
                    hops,
                    limit,
                )
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)
        return elements

    def search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        """Search graph nodes by common text-bearing properties."""
        try:
            with self._driver.session() as session:
                nodes = session.execute_read(self._search, session_id, query, view, limit)
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)
        return {"available": True, "nodes": nodes, "edges": []}

    def path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        """Return a bounded provenance path from a selected node to Eventloom events."""
        try:
            with self._driver.session() as session:
                elements = session.execute_read(self._path_to_event, session_id, node_id, limit)
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)
        return elements

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    @staticmethod
    def _count_nodes(tx: Any, session_id: str | None) -> int:
        record = tx.run(
            """
            MATCH (n)
            WHERE $session_id IS NULL OR n.session_id = $session_id OR n.thread = $session_id
            RETURN count(n) AS count
            """,
            session_id=session_id,
        ).single()
        return int(record["count"]) if record else 0

    @staticmethod
    def _count_edges(tx: Any, session_id: str | None) -> int:
        record = tx.run(
            """
            MATCH (a)-[r]->(b)
            WHERE $session_id IS NULL
               OR a.session_id = $session_id
               OR b.session_id = $session_id
               OR r.session_id = $session_id
            RETURN count(r) AS count
            """,
            session_id=session_id,
        ).single()
        return int(record["count"]) if record else 0

    @staticmethod
    def _search(
        tx: Any, session_id: str | None, query: str, view: str, limit: int
    ) -> list[dict[str, Any]]:
        records = tx.run(
            """
            MATCH (n)
            WHERE ($session_id IS NULL OR n.session_id = $session_id OR n.thread = $session_id)
              AND any(value IN [
                n.name, n.summary, n.type, n.entity_type, n.path, n.source_path, n.actor
              ] WHERE value IS NOT NULL AND toLower(toString(value)) CONTAINS toLower($query))
            RETURN n
            LIMIT $limit
            """,
            session_id=session_id,
            query=query,
            view=view,
            limit=limit,
        )
        return [_node_payload(record["n"]) for record in records]

    @staticmethod
    def _neighborhood(
        tx: Any,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        records = tx.run(
            f"""
            MATCH p=(center)-[*1..{hops}]-(neighbor)
            WHERE (elementId(center) = $node_id OR center.name = $node_id OR center.id = $node_id)
              AND ($session_id IS NULL OR center.session_id = $session_id OR neighbor.session_id = $session_id)
            RETURN p
            LIMIT $limit
            """,
            session_id=session_id,
            node_id=node_id,
            view=view,
            limit=limit,
        )
        return _paths_payload([record["p"] for record in records], limit=limit)

    @staticmethod
    def _path_to_event(
        tx: Any, session_id: str | None, node_id: str, limit: int
    ) -> dict[str, object]:
        records = tx.run(
            """
            MATCH p=shortestPath((n)-[*1..4]-(e:Event))
            WHERE (elementId(n) = $node_id OR n.name = $node_id OR n.id = $node_id)
              AND ($session_id IS NULL OR n.session_id = $session_id OR e.session_id = $session_id)
            RETURN p
            LIMIT $limit
            """,
            session_id=session_id,
            node_id=node_id,
            limit=limit,
        )
        return _paths_payload([record["p"] for record in records], limit=limit)


def resolve_dashboard_scope(config: DashboardConfig) -> DashboardScope:
    """Resolve the active workspace and Eventloom directory for the dashboard."""
    workspace = (config.workspace or Path.cwd()).resolve()
    eventloom_path = (config.eventloom_path or workspace / ".eventloom").resolve()
    return DashboardScope(
        workspace=workspace,
        eventloom_path=eventloom_path,
        session_id=config.session_id,
        domain=config.domain,
        host=config.host,
        port=config.port,
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password,
    )


class DashboardApp:
    """Read-only dashboard API facade."""

    def __init__(
        self,
        scope: DashboardScope,
        *,
        graph_provider: DashboardGraphProvider | None = None,
    ) -> None:
        self.scope = scope
        self.graph_provider = graph_provider or UnavailableGraphProvider()

    def handle_api(
        self, method: str, path: str, query: str
    ) -> tuple[int, dict[str, str], dict[str, Any]]:
        """Return a JSON-compatible API response for a dashboard route."""
        headers = {"content-type": "application/json; charset=utf-8"}
        if method.upper() != "GET":
            return 405, headers, {"error": "read_only"}

        params = parse_qs(query, keep_blank_values=False)
        if path == "/api/status":
            return 200, headers, self._status_body()
        if path == "/api/sessions":
            status = inspect_memory_status(self.scope.eventloom_path)
            return 200, headers, {"sessions": [asdict(session) for session in status.sessions]}
        if path == "/api/events":
            limit = _int_param(params, "limit", default=20, minimum=0, maximum=250)
            session_id = _str_param(params, "session_id")
            memory_log = inspect_memory_log(
                self.scope.eventloom_path,
                session_id=session_id,
                limit=limit,
            )
            return 200, headers, {"events": [asdict(entry) for entry in memory_log.entries]}
        if path == "/api/checkout":
            checkout_query = _str_param(params, "query")
            session_id = _str_param(params, "session_id") or self.scope.session_id
            return (
                200,
                headers,
                {
                    "checkout": {
                        "available": False,
                        "session_id": session_id,
                        "query": checkout_query,
                        "warning": "checkout diagnostics are not wired in this dashboard slice",
                    }
                },
            )
        if path == "/api/graph/summary":
            session_id = _str_param(params, "session_id") or self.scope.session_id
            return 200, headers, {"graph": self.graph_provider.summary(session_id=session_id)}
        if path == "/api/graph/neighborhood":
            session_id = _str_param(params, "session_id") or self.scope.session_id
            node_id = _str_param(params, "node_id")
            if node_id is None:
                return 400, headers, {"error": "node_id_required"}
            view = _str_param(params, "view") or "memory"
            hops = _int_param(params, "hops", default=1, minimum=1, maximum=2)
            limit = _int_param(params, "limit", default=100, minimum=1, maximum=250)
            return (
                200,
                headers,
                {
                    "graph": self.graph_provider.neighborhood(
                        session_id=session_id,
                        node_id=node_id,
                        view=view,
                        hops=hops,
                        limit=limit,
                    )
                },
            )
        if path == "/api/graph/search":
            session_id = _str_param(params, "session_id") or self.scope.session_id
            query_text = _str_param(params, "q")
            if query_text is None:
                return 400, headers, {"error": "query_required"}
            view = _str_param(params, "view") or "memory"
            limit = _int_param(params, "limit", default=50, minimum=1, maximum=250)
            return (
                200,
                headers,
                {
                    "graph": self.graph_provider.search(
                        session_id=session_id,
                        query=query_text,
                        view=view,
                        limit=limit,
                    )
                },
            )
        if path == "/api/graph/path-to-event":
            session_id = _str_param(params, "session_id") or self.scope.session_id
            node_id = _str_param(params, "node_id")
            if node_id is None:
                return 400, headers, {"error": "node_id_required"}
            limit = _int_param(params, "limit", default=50, minimum=1, maximum=250)
            return (
                200,
                headers,
                {
                    "graph": self.graph_provider.path_to_event(
                        session_id=session_id,
                        node_id=node_id,
                        limit=limit,
                    )
                },
            )
        return 404, headers, {"error": "not_found"}

    def _status_body(self) -> dict[str, Any]:
        status = inspect_memory_status(self.scope.eventloom_path)
        return {
            "scope": {
                "workspace": str(self.scope.workspace),
                "eventloom_path": str(self.scope.eventloom_path),
                "session_id": self.scope.session_id,
                "domain": self.scope.domain,
                "read_only": self.scope.read_only,
            },
            "memory": status.to_dict(),
        }


def _str_param(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def _int_param(
    params: dict[str, list[str]],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    values = params.get(name)
    if not values:
        return default
    try:
        parsed = int(values[0])
    except ValueError:
        return default
    return min(max(parsed, minimum), maximum)


def build_dashboard_graph_provider(scope: DashboardScope) -> DashboardGraphProvider:
    """Build the configured graph provider, degrading when credentials are incomplete."""
    if not (scope.neo4j_uri and scope.neo4j_user and scope.neo4j_password):
        return UnavailableGraphProvider()
    return Neo4jDashboardGraphProvider(scope.neo4j_uri, scope.neo4j_user, scope.neo4j_password)


def _graph_error(exc: Exception) -> dict[str, object]:
    return {
        "available": False,
        "nodes": [],
        "edges": [],
        "warning": str(exc),
    }


def _paths_payload(paths: list[Any], *, limit: int) -> dict[str, object]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for path in paths:
        for node in path.nodes:
            payload = _node_payload(node)
            nodes[payload["id"]] = payload
        for relationship in path.relationships:
            payload = _edge_payload(relationship)
            edges[payload["id"]] = payload
    return {
        "available": True,
        "nodes": list(nodes.values())[:limit],
        "edges": list(edges.values())[:limit],
        "omitted_nodes": max(0, len(nodes) - limit),
        "omitted_edges": max(0, len(edges) - limit),
    }


def _node_payload(node: Any) -> dict[str, Any]:
    properties = _json_safe_properties(dict(node.items()))
    label = (
        properties.get("name")
        or properties.get("type")
        or properties.get("entity_type")
        or properties.get("path")
        or node.element_id
    )
    return {
        "id": node.element_id,
        "label": str(label),
        "labels": list(node.labels),
        "properties": properties,
    }


def _edge_payload(relationship: Any) -> dict[str, Any]:
    return {
        "id": relationship.element_id,
        "source": relationship.start_node.element_id,
        "target": relationship.end_node.element_id,
        "label": relationship.type,
        "type": relationship.type,
        "properties": _json_safe_properties(dict(relationship.items())),
    }


def _json_safe_properties(properties: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in properties.items():
        if isinstance(value, datetime | date):
            safe[key] = value.isoformat()
        elif isinstance(value, list):
            safe[key] = [
                item.isoformat() if isinstance(item, datetime | date) else item for item in value
            ]
        else:
            safe[key] = value
    return safe


def render_dashboard_html() -> str:
    """Render the read-only runtime dashboard shell."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zaxy Runtime Dashboard</title>
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <style>
    :root {
      --bg: #f5f7f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #64717f;
      --line: #d7dde5;
      --accent: #1663c7;
      --warn: #9a5b00;
      --ok: #167a55;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    .scope { color: var(--muted); margin-top: 3px; overflow-wrap: anywhere; }
    .badge {
      border: 1px solid var(--ok);
      color: var(--ok);
      border-radius: 4px;
      padding: 4px 8px;
      font-weight: 700;
      font-size: 12px;
    }
    nav {
      display: flex;
      gap: 6px;
      padding: 10px 18px 0;
      background: var(--panel);
    }
    nav button {
      border: 1px solid var(--line);
      background: #fbfcfe;
      border-radius: 6px 6px 0 0;
      padding: 8px 11px;
      font: inherit;
      cursor: pointer;
    }
    nav button.active { border-bottom-color: var(--panel); background: var(--panel); color: var(--accent); }
    main { padding: 18px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 14px; }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric { padding: 12px; }
    .metric strong { display: block; font-size: 22px; }
    .panel { padding: 12px; margin-bottom: 14px; }
    .tab { display: none; }
    .tab.active { display: block; }
    #graph-canvas { height: 560px; border: 1px solid var(--line); border-radius: 8px; background: #ffffff; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre { overflow: auto; white-space: pre-wrap; background: #f1f4f7; padding: 10px; border-radius: 6px; }
    .warning { color: var(--warn); }
    @media (max-width: 820px) {
      header { grid-template-columns: 1fr; }
      nav { overflow-x: auto; }
      .grid { grid-template-columns: 1fr 1fr; }
      #graph-canvas { height: 420px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Runtime Dashboard</h1>
      <div class="scope" id="scope">Loading scope...</div>
    </div>
    <div class="badge">READ ONLY</div>
  </header>
  <nav>
    <button class="active" data-tab="overview">Overview</button>
    <button data-tab="sessions">Sessions</button>
    <button data-tab="graph">Graph</button>
    <button data-tab="checkout">Checkout</button>
    <button data-tab="events">Events</button>
  </nav>
  <main>
    <section class="tab active" id="overview">
      <div class="grid">
        <div class="metric"><span>Sessions</span><strong id="metric-sessions">0</strong></div>
        <div class="metric"><span>Events</span><strong id="metric-events">0</strong></div>
        <div class="metric"><span>Graph nodes</span><strong id="metric-nodes">0</strong></div>
        <div class="metric"><span>Graph edges</span><strong id="metric-edges">0</strong></div>
      </div>
      <div class="panel"><pre id="status-json">{}</pre></div>
    </section>
    <section class="tab" id="sessions"><div class="panel"><table><thead><tr><th>Session</th><th>Events</th><th>Latest</th><th>Integrity</th></tr></thead><tbody id="sessions-body"></tbody></table></div></section>
    <section class="tab" id="graph"><div class="panel warning" id="graph-warning"></div><div id="graph-canvas"></div></section>
    <section class="tab" id="checkout"><div class="panel"><pre>Checkout diagnostics endpoint is reserved for the next dashboard slice.</pre></div></section>
    <section class="tab" id="events"><div class="panel"><table><thead><tr><th>Session</th><th>Seq</th><th>Type</th><th>Actor</th><th>Summary</th></tr></thead><tbody id="events-body"></tbody></table></div></section>
  </main>
  <script>
    const statusUrl = "/api/status";
    const eventsUrl = "/api/events?limit=25";
    const graphUrl = "/api/graph/summary";

    document.querySelectorAll("nav button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("nav button").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
      });
    });

    let cy = null;
    if (window.cytoscape) {
      cy = cytoscape({
        container: document.getElementById("graph-canvas"),
        elements: [],
        style: [
          { selector: "node", style: { "label": "data(label)", "background-color": "#1663c7", "color": "#17202a", "font-size": 10 } },
          { selector: "edge", style: { "line-color": "#8c98a8", "target-arrow-color": "#8c98a8", "target-arrow-shape": "triangle", "curve-style": "bezier" } }
        ],
        layout: { name: "grid" }
      });
    }

    async function refresh() {
      const status = await fetch(statusUrl).then((response) => response.json());
      const events = await fetch(eventsUrl).then((response) => response.json());
      const graph = await fetch(graphUrl).then((response) => response.json());
      document.getElementById("scope").textContent = `${status.scope.workspace} | ${status.scope.eventloom_path} | session=${status.scope.session_id || "all"}`;
      document.getElementById("metric-sessions").textContent = status.memory.session_count;
      document.getElementById("metric-events").textContent = status.memory.total_events;
      document.getElementById("metric-nodes").textContent = graph.graph.nodes || 0;
      document.getElementById("metric-edges").textContent = graph.graph.edges || 0;
      document.getElementById("status-json").textContent = JSON.stringify(status, null, 2);
      document.getElementById("graph-warning").textContent = graph.graph.warning || "";
      document.getElementById("sessions-body").innerHTML = status.memory.sessions.map((session) => `
        <tr><td><code>${session.session_id}</code></td><td>${session.event_count}</td><td>${session.latest_type || ""}</td><td>${session.integrity_ok ? "OK" : "FAILED"}</td></tr>
      `).join("");
      document.getElementById("events-body").innerHTML = events.events.map((event) => `
        <tr><td><code>${event.session_id}</code></td><td>${event.seq}</td><td>${event.type}</td><td>${event.actor}</td><td>${event.summary || ""}</td></tr>
      `).join("");
    }
    refresh().catch((error) => {
      document.getElementById("status-json").textContent = String(error);
    });
  </script>
</body>
</html>
"""


def create_dashboard_handler(dashboard_app: DashboardApp) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler bound to a dashboard app instance."""

    class DashboardRequestHandler(BaseHTTPRequestHandler):
        dashboard_app: DashboardApp

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._write_text(200, "text/html; charset=utf-8", render_dashboard_html())
                return
            if parsed.path.startswith("/api/"):
                status, headers, body = self.dashboard_app.handle_api(
                    "GET",
                    parsed.path,
                    parsed.query,
                )
                self._write_json(status, headers, body)
                return
            self._write_json(
                404, {"content-type": "application/json; charset=utf-8"}, {"error": "not_found"}
            )

        def do_POST(self) -> None:  # noqa: N802
            self._reject_mutation()

        def do_PUT(self) -> None:  # noqa: N802
            self._reject_mutation()

        def do_PATCH(self) -> None:  # noqa: N802
            self._reject_mutation()

        def do_DELETE(self) -> None:  # noqa: N802
            self._reject_mutation()

        def log_message(self, format: str, *args: object) -> None:
            """Suppress default stderr request logging."""

        def _reject_mutation(self) -> None:
            status, headers, body = self.dashboard_app.handle_api(
                self.command,
                urlparse(self.path).path,
                urlparse(self.path).query,
            )
            self._write_json(status, headers, body)

        def _write_json(self, status: int, headers: dict[str, str], body: dict[str, Any]) -> None:
            payload = json.dumps(body, sort_keys=True).encode("utf-8")
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _write_text(self, status: int, content_type: str, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    DashboardRequestHandler.dashboard_app = dashboard_app
    return DashboardRequestHandler


def run_dashboard(scope: DashboardScope) -> None:
    """Run the read-only dashboard server until interrupted."""
    dashboard_app = DashboardApp(scope, graph_provider=build_dashboard_graph_provider(scope))
    handler = create_dashboard_handler(dashboard_app)
    server = ThreadingHTTPServer((scope.host, scope.port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
