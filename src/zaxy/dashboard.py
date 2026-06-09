"""Local dashboard for runtime memory debugging and explicit coordination review."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Protocol, cast
from urllib.parse import parse_qs, urlparse

from zaxy.core import MemoryFabric
from zaxy.event import Event, EventLog
from zaxy.hooks import inspect_memory_activation
from zaxy.memory_persistence import inspect_memory_persistence
from zaxy.memory_status import inspect_memory_log, inspect_memory_status


def _load_sync_neo4j_graph_database() -> Any:
    """Load Neo4j's sync driver only for the explicit Neo4j dashboard backend."""
    try:
        neo4j = import_module("neo4j")
    except ImportError as exc:
        raise RuntimeError('Neo4j dashboard backend requires `pip install "zaxy-memory[neo4j]"`') from exc
    return neo4j.GraphDatabase


@dataclass(frozen=True)
class DashboardConfig:
    """User-provided dashboard configuration before path resolution."""

    workspace: Path | None = None
    eventloom_path: Path | None = None
    session_id: str | None = None
    domain: str | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    projection_backend: str = "embedded"
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    pggraph_dsn: str | None = None
    embedded_graph_path: Path | None = None
    coordinate_review_enabled: bool = False


@dataclass(frozen=True)
class DashboardScope:
    """Resolved dashboard scope."""

    workspace: Path
    eventloom_path: Path
    session_id: str | None
    domain: str | None
    host: str
    port: int
    projection_backend: str = "embedded"
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    pggraph_dsn: str | None = None
    embedded_graph_path: Path | None = None
    read_only: bool = True
    coordinate_review_enabled: bool = False


def _dashboard_origin_allowed(scope: DashboardScope, headers: Mapping[str, str] | None) -> bool:
    """Return whether a dashboard mutation request came from this dashboard origin."""
    normalized = {key.lower(): value for key, value in (headers or {}).items()}
    expected_host = f"{scope.host}:{scope.port}"
    host = normalized.get("host", "")
    if host and host != expected_host:
        return False
    origin = normalized.get("origin")
    if origin is None:
        return True
    return origin in {f"http://{expected_host}", f"https://{expected_host}"}


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


class EventloomDashboardGraphProvider:
    """Read-only graph provider backed by local Eventloom provenance."""

    def __init__(self, eventloom_path: str | Path, *, max_events: int = 100) -> None:
        self.eventloom_path = Path(eventloom_path)
        self.max_events = max_events

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        """Return a bounded Eventloom provenance graph."""
        events = self._events(session_id=session_id, limit=self.max_events)
        nodes = [_event_node(session_id, event) for session_id, event in events]
        edges = [
            _event_edge(previous_session, previous, current)
            for (previous_session, previous), (current_session, current) in zip(
                events,
                events[1:],
                strict=False,
            )
            if previous_session == current_session
        ]
        return {
            "available": True,
            "source": "eventloom",
            "nodes": len(nodes),
            "edges": len(edges),
            "elements": {"nodes": nodes, "edges": edges},
            "warning": None,
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
        """Return nearby Eventloom events for the selected event node."""
        _view = view
        _hops = hops
        try:
            _prefix, selected_session, raw_seq = node_id.split(":", 2)
            selected_seq = int(raw_seq)
        except ValueError:
            return self.summary(session_id=session_id)
        if session_id is not None and selected_session != session_id:
            return {"available": True, "source": "eventloom", "nodes": [], "edges": []}
        events = [
            item
            for item in self._events(session_id=selected_session, limit=max(limit, self.max_events))
            if abs(item[1].seq - selected_seq) <= 2
        ][:limit]
        nodes = [_event_node(item_session, event) for item_session, event in events]
        edges = [
            _event_edge(previous_session, previous, current)
            for (previous_session, previous), (current_session, current) in zip(
                events,
                events[1:],
                strict=False,
            )
            if previous_session == current_session
        ]
        return {
            "available": True,
            "source": "eventloom",
            "nodes": nodes,
            "edges": edges,
            "omitted_nodes": 0,
            "omitted_edges": 0,
        }

    def search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        """Search Eventloom event summaries and metadata."""
        _view = view
        needle = query.casefold()
        nodes = [
            _event_node(item_session, event)
            for item_session, event in self._events(session_id=session_id, limit=self.max_events)
            if needle in _event_search_text(item_session, event)
        ][:limit]
        return {"available": True, "source": "eventloom", "nodes": nodes, "edges": []}

    def path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        """Eventloom nodes are already provenance nodes, so return their neighborhood."""
        return self.neighborhood(
            session_id=session_id,
            node_id=node_id,
            view="provenance",
            hops=1,
            limit=limit,
        )

    def _events(self, *, session_id: str | None, limit: int) -> list[tuple[str, Event]]:
        paths = (
            [_session_log_path(self.eventloom_path, session_id)]
            if session_id
            else _eventlog_paths(self.eventloom_path)
        )
        events: list[tuple[str, Event]] = []
        for path in paths:
            for event in EventLog(path).read_all():
                events.append((path.stem, event))
        events.sort(key=lambda item: (item[0], item[1].seq))
        return events[-limit:]


class FallbackDashboardGraphProvider:
    """Use a primary graph provider, falling back to Eventloom when unavailable."""

    def __init__(self, primary: DashboardGraphProvider, fallback: DashboardGraphProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        result = self.primary.summary(session_id=session_id)
        return (
            result
            if result.get("available") is not False
            else self.fallback.summary(session_id=session_id)
        )

    def neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        result = self.primary.neighborhood(
            session_id=session_id,
            node_id=node_id,
            view=view,
            hops=hops,
            limit=limit,
        )
        return (
            result
            if result.get("available") is not False
            else self.fallback.neighborhood(
                session_id=session_id,
                node_id=node_id,
                view=view,
                hops=hops,
                limit=limit,
            )
        )

    def search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        result = self.primary.search(session_id=session_id, query=query, view=view, limit=limit)
        return (
            result
            if result.get("available") is not False
            else self.fallback.search(
                session_id=session_id,
                query=query,
                view=view,
                limit=limit,
            )
        )

    def path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        result = self.primary.path_to_event(session_id=session_id, node_id=node_id, limit=limit)
        return (
            result
            if result.get("available") is not False
            else self.fallback.path_to_event(
                session_id=session_id,
                node_id=node_id,
                limit=limit,
            )
        )


class Neo4jDashboardGraphProvider:
    """Read-only Neo4j graph provider for the local dashboard."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        graph_database = _load_sync_neo4j_graph_database()
        self._driver = graph_database.driver(
            uri,
            auth=(user, password),
            connection_timeout=1.0,
            max_transaction_retry_time=0.0,
        )

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        """Return node and edge counts for the selected graph scope."""
        try:
            with self._driver.session() as session:
                node_count = self._count_nodes(session, session_id)
                edge_count = self._count_edges(session, session_id)
                elements = self._overview_elements(session, session_id, limit=100)
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)
        return {
            "available": True,
            "source": "neo4j",
            "nodes": node_count,
            "edges": edge_count,
            "elements": elements,
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
        """Return a bounded neighborhood around one graph node."""
        try:
            with self._driver.session() as session:
                elements = self._neighborhood(
                    session,
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
                nodes = self._search(session, session_id, query, view, limit)
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)
        return {"available": True, "source": "neo4j", "nodes": nodes, "edges": []}

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
                elements = self._path_to_event(session, session_id, node_id, limit)
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
    def _overview_elements(
        tx: Any, session_id: str | None, limit: int
    ) -> dict[str, list[dict[str, Any]]]:
        path_records = tx.run(
            """
            MATCH p=(a)-[r]->(b)
            WHERE $session_id IS NULL
               OR a.session_id = $session_id
               OR b.session_id = $session_id
               OR r.session_id = $session_id
            RETURN p
            LIMIT $limit
            """,
            session_id=session_id,
            limit=limit,
        )
        paths = [path for record in path_records if (path := record.get("p")) is not None]
        if paths:
            payload = _paths_payload(paths, limit=limit)
            nodes = payload["nodes"]
            edges = payload["edges"]
            if not isinstance(nodes, list) or not isinstance(edges, list):
                return {"nodes": [], "edges": []}
            return {
                "nodes": nodes,
                "edges": edges,
            }
        node_records = tx.run(
            """
            MATCH (n)
            WHERE $session_id IS NULL OR n.session_id = $session_id OR n.thread = $session_id
            RETURN n
            LIMIT $limit
            """,
            session_id=session_id,
            limit=limit,
        )
        return {"nodes": [_node_payload(record["n"]) for record in node_records], "edges": []}

    @staticmethod
    def _search(
        tx: Any, session_id: str | None, query: str, view: str, limit: int
    ) -> list[dict[str, Any]]:
        records = tx.run(
            """
            MATCH (n)
            WHERE ($session_id IS NULL OR n.session_id = $session_id OR n.thread = $session_id)
              AND any(value IN [
                properties(n)["name"],
                properties(n)["summary"],
                properties(n)["type"],
                properties(n)["entity_type"],
                properties(n)["path"],
                properties(n)["source_path"],
                properties(n)["actor"]
              ] WHERE value IS NOT NULL AND toLower(toString(value)) CONTAINS toLower($search_text))
            RETURN n
            LIMIT $limit
            """,
            session_id=session_id,
            search_text=query,
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


class ProjectionDashboardGraphProvider:
    """Read-only dashboard graph provider backed by the projection contract tables."""

    def __init__(self, backend: str, *, pggraph_dsn: str | None = None) -> None:
        normalized = backend.casefold().strip()
        if normalized != "pggraph":
            raise ValueError("Projection dashboard provider currently supports pggraph only")
        if not pggraph_dsn:
            raise ValueError("pgGraph dashboard backend requires pggraph_dsn")
        from zaxy.pggraph_store import PgGraphStore

        self.backend = normalized
        self._store = PgGraphStore(pggraph_dsn)

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        """Return node and edge counts plus bounded overview elements."""
        try:
            return cast(
                dict[str, object],
                _run_async(self._summary(session_id=session_id)),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)

    def neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        """Return a bounded pgGraph neighborhood around one projected node."""
        try:
            return cast(
                dict[str, object],
                _run_async(
                    self._neighborhood(
                        session_id=session_id,
                        node_id=node_id,
                        view=view,
                        hops=hops,
                        limit=limit,
                    )
                ),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)

    def search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        """Search projected pgGraph nodes."""
        try:
            return cast(
                dict[str, object],
                _run_async(
                    self._search(session_id=session_id, query=query, view=view, limit=limit)
                ),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)

    def path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        """Return the projected node and its source Eventloom event when available."""
        try:
            return cast(
                dict[str, object],
                _run_async(
                    self._path_to_event(session_id=session_id, node_id=node_id, limit=limit)
                ),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)

    async def _summary(self, *, session_id: str | None) -> dict[str, object]:
        await self._store.connect()
        try:
            counts = await self._store._fetch_all(
                """
                SELECT
                    (SELECT count(*) FROM zaxy_pggraph_entities
                     WHERE (%(session_id)s::text IS NULL OR session_id = %(session_id)s::text)
                       AND valid_to IS NULL) AS node_count,
                    (SELECT count(*) FROM zaxy_pggraph_edges
                     WHERE (%(session_id)s::text IS NULL OR session_id = %(session_id)s::text)
                       AND valid_to IS NULL) AS edge_count
                """,
                {"session_id": session_id},
            )
            node_rows = await self._store._fetch_all(
                """
                SELECT node_key, name, entity_type, summary, properties, session_id,
                       source_event_seq, source_event_hash, valid_from, valid_to
                FROM zaxy_pggraph_entities
                WHERE (%(session_id)s::text IS NULL OR session_id = %(session_id)s::text)
                  AND valid_to IS NULL
                ORDER BY updated_at DESC
                LIMIT 100
                """,
                {"session_id": session_id},
            )
            edge_rows = await self._store._fetch_all(
                """
                SELECT edge_key, source_node_key, target_node_key, relation_type, properties,
                       session_id, source_event_seq, source_event_hash
                FROM zaxy_pggraph_edges
                WHERE (%(session_id)s::text IS NULL OR session_id = %(session_id)s::text)
                  AND valid_to IS NULL
                ORDER BY updated_at DESC
                LIMIT 100
                """,
                {"session_id": session_id},
            )
        finally:
            await self._store.close()
        count_row = counts[0] if counts else {}
        return {
            "available": True,
            "source": self.backend,
            "nodes": int(count_row.get("node_count") or 0),
            "edges": int(count_row.get("edge_count") or 0),
            "elements": _pggraph_elements(node_rows, edge_rows),
        }

    async def _neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        await self._store.connect()
        try:
            node_rows = await self._store._fetch_all(
                """
                WITH center AS (
                    SELECT node_key
                    FROM zaxy_pggraph_entities
                    WHERE (node_key = %(node_id)s OR name = %(node_id)s)
                      AND (%(session_id)s::text IS NULL OR session_id = %(session_id)s::text)
                      AND valid_to IS NULL
                    LIMIT 1
                ), edges AS (
                    SELECT edge.*
                    FROM zaxy_pggraph_edges edge, center
                    WHERE edge.valid_to IS NULL
                      AND (%(session_id)s::text IS NULL OR edge.session_id = %(session_id)s::text)
                      AND (
                        edge.source_node_key = center.node_key
                        OR edge.target_node_key = center.node_key
                      )
                    LIMIT %(limit)s
                )
                SELECT DISTINCT entity.node_key, entity.name, entity.entity_type, entity.summary,
                       entity.properties, entity.session_id, entity.source_event_seq,
                       entity.source_event_hash, entity.valid_from, entity.valid_to
                FROM zaxy_pggraph_entities entity
                JOIN edges ON entity.node_key IN (edges.source_node_key, edges.target_node_key)
                WHERE entity.valid_to IS NULL
                LIMIT %(limit)s
                """,
                {"session_id": session_id, "node_id": node_id, "limit": limit},
            )
            edge_rows = await self._store._fetch_all(
                """
                WITH center AS (
                    SELECT node_key
                    FROM zaxy_pggraph_entities
                    WHERE (node_key = %(node_id)s OR name = %(node_id)s)
                      AND (%(session_id)s::text IS NULL OR session_id = %(session_id)s::text)
                      AND valid_to IS NULL
                    LIMIT 1
                )
                SELECT edge_key, source_node_key, target_node_key, relation_type, properties,
                       session_id, source_event_seq, source_event_hash
                FROM zaxy_pggraph_edges edge, center
                WHERE edge.valid_to IS NULL
                  AND (%(session_id)s::text IS NULL OR edge.session_id = %(session_id)s::text)
                  AND (
                    edge.source_node_key = center.node_key
                    OR edge.target_node_key = center.node_key
                  )
                LIMIT %(limit)s
                """,
                {"session_id": session_id, "node_id": node_id, "limit": limit},
            )
        finally:
            await self._store.close()
        elements = _pggraph_elements(node_rows, edge_rows)
        return {
            "available": True,
            "source": self.backend,
            "view": view,
            "hops": hops,
            "nodes": elements["nodes"],
            "edges": elements["edges"],
            "omitted_nodes": 0,
            "omitted_edges": 0,
        }

    async def _search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        await self._store.connect()
        try:
            node_rows = await self._store._fetch_all(
                """
                SELECT node_key, name, entity_type, summary, properties, session_id,
                       source_event_seq, source_event_hash, valid_from, valid_to
                FROM zaxy_pggraph_entities
                WHERE (%(session_id)s::text IS NULL OR session_id = %(session_id)s::text)
                  AND valid_to IS NULL
                  AND (name ILIKE %(query)s OR summary ILIKE %(query)s)
                ORDER BY updated_at DESC
                LIMIT %(limit)s
                """,
                {"session_id": session_id, "query": f"%{query}%", "limit": limit},
            )
        finally:
            await self._store.close()
        return {
            "available": True,
            "source": self.backend,
            "view": view,
            "nodes": [_pggraph_node_payload(row) for row in node_rows],
            "edges": [],
        }

    async def _path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        await self._store.connect()
        try:
            node_rows = await self._store._fetch_all(
                """
                SELECT node_key, name, entity_type, summary, properties, session_id,
                       source_event_seq, source_event_hash, valid_from, valid_to
                FROM zaxy_pggraph_entities
                WHERE (node_key = %(node_id)s OR name = %(node_id)s)
                  AND (%(session_id)s::text IS NULL OR session_id = %(session_id)s::text)
                  AND valid_to IS NULL
                LIMIT 1
                """,
                {"session_id": session_id, "node_id": node_id},
            )
        finally:
            await self._store.close()
        if not node_rows:
            return {"available": True, "source": self.backend, "nodes": [], "edges": []}
        node = _pggraph_node_payload(node_rows[0])
        event_seq = node_rows[0].get("source_event_seq")
        if event_seq is None:
            return {"available": True, "source": self.backend, "nodes": [node], "edges": []}
        event_id = f"event:{node_rows[0].get('session_id')}:{event_seq}"
        event = {
            "id": event_id,
            "label": f"Event #{event_seq}",
            "kind": "event",
            "properties": {
                "seq": event_seq,
                "hash": node_rows[0].get("source_event_hash"),
                "session_id": node_rows[0].get("session_id"),
            },
        }
        return {
            "available": True,
            "source": self.backend,
            "nodes": [node, event][:limit],
            "edges": [
                {
                    "id": f"source-event:{node['id']}:{event_id}",
                    "source": node["id"],
                    "target": event_id,
                    "label": "SOURCE_EVENT",
                    "type": "SOURCE_EVENT",
                    "properties": {},
                }
            ][:limit],
        }


class EmbeddedDashboardGraphProvider:
    """Read-only dashboard graph provider backed by the embedded projection."""

    def __init__(self, path: Path) -> None:
        from zaxy.embedded_graph_store import EmbeddedGraphStore

        self.backend = "embedded"
        self._store = EmbeddedGraphStore(Path(path))

    def summary(self, *, session_id: str | None) -> dict[str, object]:
        """Return active embedded graph nodes and edges for dashboard rendering."""
        try:
            return cast(dict[str, object], _run_async(self._summary(session_id=session_id)))
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)

    def neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        """Return a bounded neighborhood around one embedded graph node."""
        try:
            return cast(
                dict[str, object],
                _run_async(
                    self._neighborhood(
                        session_id=session_id,
                        node_id=node_id,
                        view=view,
                        hops=hops,
                        limit=limit,
                    )
                ),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)

    def search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        """Search active embedded graph nodes by visible text."""
        try:
            return cast(
                dict[str, object],
                _run_async(self._search(session_id=session_id, query=query, view=view, limit=limit)),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)

    def path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        """Return a projected node and its source Eventloom event when present."""
        try:
            return cast(
                dict[str, object],
                _run_async(self._path_to_event(session_id=session_id, node_id=node_id, limit=limit)),
            )
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            return _graph_error(exc)

    async def _summary(self, *, session_id: str | None) -> dict[str, object]:
        await self._store.connect()
        try:
            conn = self._store._require_connection()
            try:
                node_rows = conn.execute(
                    """
                    MATCH (e:Entity)
                    WHERE ($session_id IS NULL OR e.session_id = $session_id)
                      AND e.valid_to IS NULL
                    RETURN e.node_key, e.name, e.entity_type, e.summary, e.properties_json,
                           e.session_id, e.source_event_seq, e.source_event_hash,
                           e.valid_from, e.valid_to
                    ORDER BY e.source_event_seq DESC
                    LIMIT 100
                    """,
                    {"session_id": session_id},
                ).get_all()
                edge_rows = conn.execute(
                    """
                    MATCH (source:Entity)-[r:RELATES]->(target:Entity)
                    WHERE ($session_id IS NULL OR r.session_id = $session_id)
                      AND r.valid_to IS NULL
                      AND source.valid_to IS NULL
                      AND target.valid_to IS NULL
                    RETURN source.node_key, target.node_key, r.relation_type, r.session_id,
                           r.source_event_seq, r.source_event_hash, r.evidence_json
                    LIMIT 100
                    """,
                    {"session_id": session_id},
                ).get_all()
            except RuntimeError as exc:
                if not _is_missing_embedded_projection_table_error(exc):
                    raise
                return _empty_embedded_dashboard_summary()
        finally:
            await self._store.close()
        return {
            "available": True,
            "source": self.backend,
            "nodes": len(node_rows),
            "edges": len(edge_rows),
            "elements": _embedded_elements(node_rows, edge_rows),
        }

    async def _neighborhood(
        self,
        *,
        session_id: str | None,
        node_id: str,
        view: str,
        hops: int,
        limit: int,
    ) -> dict[str, object]:
        await self._store.connect()
        try:
            conn = self._store._require_connection()
            try:
                edge_rows = conn.execute(
                    """
                    MATCH (source:Entity)-[r:RELATES]->(target:Entity)
                    WHERE ($session_id IS NULL OR r.session_id = $session_id)
                      AND r.valid_to IS NULL
                      AND source.valid_to IS NULL
                      AND target.valid_to IS NULL
                      AND (
                        source.node_key = $node_id OR source.name = $node_id
                        OR target.node_key = $node_id OR target.name = $node_id
                      )
                    RETURN source.node_key, target.node_key, r.relation_type, r.session_id,
                           r.source_event_seq, r.source_event_hash, r.evidence_json
                    LIMIT $limit
                    """,
                    {"session_id": session_id, "node_id": node_id, "limit": limit},
                ).get_all()
                node_keys = sorted({str(row[0]) for row in edge_rows} | {str(row[1]) for row in edge_rows})
                node_rows = _embedded_fetch_nodes(
                    conn,
                    session_id=session_id,
                    node_keys=node_keys,
                    limit=limit,
                )
            except RuntimeError as exc:
                if not _is_missing_embedded_projection_table_error(exc):
                    raise
                return _empty_embedded_dashboard_elements(view=view, hops=hops)
        finally:
            await self._store.close()
        elements = _embedded_elements(node_rows, edge_rows)
        return {
            "available": True,
            "source": self.backend,
            "view": view,
            "hops": hops,
            "nodes": elements["nodes"],
            "edges": elements["edges"],
            "omitted_nodes": 0,
            "omitted_edges": 0,
        }

    async def _search(
        self,
        *,
        session_id: str | None,
        query: str,
        view: str,
        limit: int,
    ) -> dict[str, object]:
        await self._store.connect()
        try:
            try:
                rows = self._store._require_connection().execute(
                    """
                    MATCH (e:Entity)
                    WHERE ($session_id IS NULL OR e.session_id = $session_id)
                      AND e.valid_to IS NULL
                    RETURN e.node_key, e.name, e.entity_type, e.summary, e.properties_json,
                           e.session_id, e.source_event_seq, e.source_event_hash,
                           e.valid_from, e.valid_to
                    LIMIT 250
                    """,
                    {"session_id": session_id},
                ).get_all()
            except RuntimeError as exc:
                if not _is_missing_embedded_projection_table_error(exc):
                    raise
                return _empty_embedded_dashboard_elements(view=view)
        finally:
            await self._store.close()
        needle = query.casefold()
        nodes = [
            _embedded_node_payload(row)
            for row in rows
            if needle in f"{row[1]} {row[2]} {row[3] or ''}".casefold()
        ][:limit]
        return {"available": True, "source": self.backend, "view": view, "nodes": nodes, "edges": []}

    async def _path_to_event(
        self,
        *,
        session_id: str | None,
        node_id: str,
        limit: int,
    ) -> dict[str, object]:
        await self._store.connect()
        try:
            try:
                rows = self._store._require_connection().execute(
                    """
                    MATCH (e:Entity)
                    WHERE (e.node_key = $node_id OR e.name = $node_id)
                      AND ($session_id IS NULL OR e.session_id = $session_id)
                      AND e.valid_to IS NULL
                    RETURN e.node_key, e.name, e.entity_type, e.summary, e.properties_json,
                           e.session_id, e.source_event_seq, e.source_event_hash,
                           e.valid_from, e.valid_to
                    LIMIT 1
                    """,
                    {"session_id": session_id, "node_id": node_id},
                ).get_all()
            except RuntimeError as exc:
                if not _is_missing_embedded_projection_table_error(exc):
                    raise
                return _empty_embedded_dashboard_elements()
        finally:
            await self._store.close()
        if not rows:
            return {"available": True, "source": self.backend, "nodes": [], "edges": []}
        node = _embedded_node_payload(rows[0])
        event_seq = rows[0][6]
        if event_seq is None:
            return {"available": True, "source": self.backend, "nodes": [node], "edges": []}
        event_id = f"event:{rows[0][5]}:{event_seq}"
        event = {
            "id": event_id,
            "label": f"Event #{event_seq}",
            "kind": "event",
            "properties": {
                "seq": event_seq,
                "hash": rows[0][7],
                "session_id": rows[0][5],
            },
        }
        return {
            "available": True,
            "source": self.backend,
            "nodes": [node, event][:limit],
            "edges": [
                {
                    "id": f"source-event:{node['id']}:{event_id}",
                    "source": node["id"],
                    "target": event_id,
                    "label": "SOURCE_EVENT",
                    "type": "SOURCE_EVENT",
                    "properties": {},
                }
            ][:limit],
        }


def resolve_dashboard_scope(config: DashboardConfig) -> DashboardScope:
    """Resolve the active workspace and Eventloom directory for the dashboard."""
    workspace = (config.workspace or Path.cwd()).resolve()
    eventloom_path = (config.eventloom_path or workspace / ".eventloom").resolve()
    projection_backend = config.projection_backend.casefold().strip()
    if projection_backend not in {"neo4j", "pggraph", "embedded", "latticedb"}:
        raise ValueError("projection backend must be one of: embedded, neo4j, pggraph, latticedb")
    return DashboardScope(
        workspace=workspace,
        eventloom_path=eventloom_path,
        session_id=config.session_id,
        domain=config.domain,
        host=config.host,
        port=config.port,
        projection_backend=projection_backend,
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password,
        pggraph_dsn=config.pggraph_dsn,
        embedded_graph_path=(config.embedded_graph_path or eventloom_path / "projections" / "embedded.kuzu"),
        coordinate_review_enabled=config.coordinate_review_enabled,
    )


class DashboardApp:
    """Dashboard API facade."""

    def __init__(
        self,
        scope: DashboardScope,
        *,
        graph_provider: DashboardGraphProvider | None = None,
    ) -> None:
        self.scope = scope
        self.graph_provider = graph_provider or EventloomDashboardGraphProvider(
            scope.eventloom_path
        )

    def handle_api(
        self,
        method: str,
        path: str,
        query: str,
        *,
        body: str | bytes | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, str], dict[str, Any]]:
        """Return a JSON-compatible API response for a dashboard route."""
        headers = {"content-type": "application/json; charset=utf-8"}
        if method.upper() == "POST" and path.startswith("/api/coordinate/"):
            if not _dashboard_origin_allowed(self.scope, request_headers):
                return 403, headers, {"error": "forbidden_origin"}
        if method.upper() == "POST" and path in {"/api/coordinate/review", "/api/coordinate/review-finding"}:
            params = parse_qs(query, keep_blank_values=False)
            return self._coordinate_review_body(params, headers, body=body)
        if method.upper() == "POST" and path == "/api/coordinate/apply-approval":
            params = parse_qs(query, keep_blank_values=False)
            return self._coordinate_apply_approval_body(params, headers, body=body)
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
        if path == "/api/memory-persistence":
            session_id = _str_param(params, "session_id") or self.scope.session_id or "default"
            return (
                200,
                headers,
                {
                    "memory_persistence": inspect_memory_persistence(
                        self.scope.eventloom_path,
                        session_id=session_id,
                    )
                },
            )
        if path == "/api/purpose/status":
            session_id = _str_param(params, "session_id") or self.scope.session_id
            from zaxy.purpose_control import build_purpose_status

            return 200, headers, {"purpose": build_purpose_status(self.scope.eventloom_path, session_id=session_id)}
        if path == "/api/purpose/lanes":
            session_id = _str_param(params, "session_id") or self.scope.session_id
            from zaxy.purpose_control import build_purpose_lanes

            return 200, headers, {"purpose_lanes": build_purpose_lanes(self.scope.eventloom_path, session_id=session_id)}
        if path == "/api/purpose/feedback":
            session_id = _str_param(params, "session_id") or self.scope.session_id
            profile = _str_param(params, "profile")
            outcome = _str_param(params, "outcome") or "all"
            limit = _int_param(params, "limit", default=20, minimum=1, maximum=250)
            from zaxy.purpose_control import build_purpose_feedback

            return 200, headers, {
                "purpose_feedback": build_purpose_feedback(
                    self.scope.eventloom_path,
                    session_id=session_id,
                    profile=profile,
                    outcome=outcome,
                    limit=limit,
                )
            }
        if path == "/api/checkout":
            checkout_query = _str_param(params, "query")
            if checkout_query is None:
                return 400, headers, {"error": "query_required"}
            session_id = _str_param(params, "session_id") or self.scope.session_id
            limit = _int_param(params, "limit", default=10, minimum=1, maximum=50)
            replay_from_seq = _int_param(params, "replay_from_seq", default=1, minimum=1, maximum=1_000_000)
            max_recent_events = _int_param(params, "max_recent_events", default=20, minimum=1, maximum=250)
            ref = _str_param(params, "ref")
            return (
                200,
                headers,
                {
                    "checkout": self._checkout_body(
                        query=checkout_query,
                        session_id=session_id or "default",
                        limit=limit,
                        replay_from_seq=replay_from_seq,
                        max_recent_events=max_recent_events,
                        ref=ref,
                    )
                },
            )
        if path == "/api/coordination/mission":
            mission_id = _str_param(params, "mission_id")
            if mission_id is None:
                return 400, headers, {"error": "mission_id_required"}
            include_diagnostics = _bool_param(params, "include_diagnostics", default=False)
            return 200, headers, {
                "coordination": self._coordination_mission_body(
                    mission_id=mission_id,
                    include_diagnostics=include_diagnostics,
                )
            }
        if path == "/api/coordinate/brief":
            mission_id = _str_param(params, "mission_id")
            if mission_id is None:
                return 400, headers, {"error": "mission_id_required"}
            return 200, headers, {"brief": self._coordination_manager().brief(mission_id).to_dict()}
        if path == "/api/coordinate/ledger":
            mission_id = _str_param(params, "mission_id")
            if mission_id is None:
                return 400, headers, {"error": "mission_id_required"}
            return 200, headers, {
                "ledger": self._coordination_manager().performance_ledger(mission_id).to_dict()
            }
        if path == "/api/coordinate/approval-packet":
            mission_id = _str_param(params, "mission_id")
            if mission_id is None:
                return 400, headers, {"error": "mission_id_required"}
            return 200, headers, {
                "approval_packet": self._coordination_manager().approval_packet(mission_id).to_dict()
            }
        if path == "/api/coordinate/review-export":
            mission_id = _str_param(params, "mission_id")
            if mission_id is None:
                return 400, headers, {"error": "mission_id_required"}
            return 200, headers, {
                "review_export": self._coordination_manager().review_export(mission_id).to_dict()
            }
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

    def _checkout_body(
        self,
        *,
        query: str,
        session_id: str,
        limit: int,
        replay_from_seq: int,
        max_recent_events: int,
        ref: str | None,
    ) -> dict[str, Any]:
        """Return a read-only Memory Checkout payload for dashboard inspection."""

        async def _checkout() -> dict[str, Any]:
            fabric = MemoryFabric(
                eventloom_path=str(self.scope.eventloom_path),
                projection_backend=self.scope.projection_backend,
                neo4j_uri=self.scope.neo4j_uri,
                neo4j_user=self.scope.neo4j_user,
                neo4j_password=self.scope.neo4j_password,
                pggraph_dsn=self.scope.pggraph_dsn,
                embedded_graph_path=self.scope.embedded_graph_path,
                tracer_disabled=True,
            )
            try:
                checkout = await fabric.checkout_memory(
                    query,
                    session_id=session_id,
                    replay_from_seq=replay_from_seq,
                    limit=limit,
                    max_recent_events=max_recent_events,
                    ref=ref,
                )
                return checkout.to_dict()
            finally:
                await fabric.close()

        payload = asyncio.run(_checkout())
        return {
            "available": True,
            "read_only": True,
            "session_id": session_id,
            "query": query,
            "payload": payload,
        }

    def _coordination_mission_body(
        self,
        *,
        mission_id: str,
        include_diagnostics: bool,
    ) -> dict[str, Any]:
        """Return a read-only coordination mission dashboard payload."""
        manager = self._coordination_manager()
        brief = manager.brief(mission_id)
        checkout = manager.checkout(mission_id, include_diagnostics=include_diagnostics)
        ledger = manager.performance_ledger(mission_id)
        approval_packet = manager.approval_packet(mission_id)
        return {
            "available": True,
            "read_only": True,
            "review_enabled": self.scope.coordinate_review_enabled,
            "mission_id": mission_id,
            "include_diagnostics": include_diagnostics,
            "brief": brief.to_dict(),
            "checkout": checkout.to_dict(),
            "ledger": ledger.to_dict(),
            "approval_packet": approval_packet.to_dict(),
        }

    def _coordination_manager(self) -> Any:
        """Return a coordination manager bound to the dashboard Eventloom scope."""
        from zaxy.coordination import CoordinationManager

        return CoordinationManager(eventloom_path=self.scope.eventloom_path)

    def _coordinate_review_body(
        self,
        params: dict[str, list[str]],
        headers: dict[str, str],
        *,
        body: str | bytes | None,
    ) -> tuple[int, dict[str, str], dict[str, Any]]:
        """Apply an explicitly enabled dashboard review action."""
        if not self.scope.coordinate_review_enabled:
            return 403, headers, {"error": "coordinate_review_disabled"}
        body_payload = _json_body(body)
        if isinstance(body_payload, tuple):
            return body_payload[0], headers, body_payload[1]
        mission_id = _str_param(params, "mission_id") or _optional_body_str(body_payload, "mission_id")
        finding_id = _str_param(params, "finding_id") or _optional_body_str(body_payload, "finding_id")
        status = _str_param(params, "status") or _optional_body_str(body_payload, "status")
        if mission_id is None:
            return 400, headers, {"error": "mission_id_required"}
        if finding_id is None:
            return 400, headers, {"error": "finding_id_required"}
        if status is None:
            return 400, headers, {"error": "status_required"}
        rationale = _str_param(params, "rationale") or _optional_body_str(body_payload, "rationale")
        actor = _str_param(params, "actor") or _optional_body_str(body_payload, "actor") or "dashboard"
        promote = _body_bool(body_payload, "promote", default=_bool_param(params, "promote", default=False))
        manager = self._coordination_manager()
        try:
            result = manager.apply_approval_decisions(
                mission_id,
                [
                    {
                        "finding_id": finding_id,
                        "status": status,
                        "rationale": rationale,
                        "promote": promote,
                    }
                ],
                actor=actor,
            )
        except ValueError as exc:
            return 400, headers, {"error": "invalid_coordinate_review", "message": str(exc)}
        return 200, headers, {
            "review": result.to_dict(),
            "coordination": self._coordination_mission_body(
                mission_id=mission_id,
                include_diagnostics=True,
            ),
        }

    def _coordinate_apply_approval_body(
        self,
        params: dict[str, list[str]],
        headers: dict[str, str],
        *,
        body: str | bytes | None,
    ) -> tuple[int, dict[str, str], dict[str, Any]]:
        """Apply an explicitly enabled dashboard approval packet."""
        if not self.scope.coordinate_review_enabled:
            return 403, headers, {"error": "coordinate_review_disabled"}
        payload = _json_body(body)
        if isinstance(payload, tuple):
            return payload[0], headers, payload[1]
        mission_id = _str_param(params, "mission_id") or _optional_body_str(payload, "mission_id")
        if mission_id is None:
            return 400, headers, {"error": "mission_id_required"}
        actor = _str_param(params, "actor") or _optional_body_str(payload, "actor") or "dashboard"
        decisions = payload.get("decisions")
        if not isinstance(decisions, list) or not all(isinstance(item, dict) for item in decisions):
            return 400, headers, {"error": "invalid_decisions"}
        try:
            result = self._coordination_manager().apply_approval_decisions(
                mission_id,
                decisions,
                actor=actor,
            )
        except ValueError as exc:
            return 400, headers, {"error": "invalid_coordinate_review", "message": str(exc)}
        return 200, headers, {
            "approval_result": result.to_dict(),
            "coordination": self._coordination_mission_body(
                mission_id=mission_id,
                include_diagnostics=True,
            ),
        }

    def _status_body(self) -> dict[str, Any]:
        status = inspect_memory_status(self.scope.eventloom_path)
        session_id = self.scope.session_id or "default"
        return {
            "scope": {
                "workspace": str(self.scope.workspace),
                "eventloom_path": str(self.scope.eventloom_path),
                "session_id": self.scope.session_id,
                "domain": self.scope.domain,
                "read_only": self.scope.read_only,
                "coordinate_review_enabled": self.scope.coordinate_review_enabled,
                "projection_backend": self.scope.projection_backend,
            },
            "memory": status.to_dict(),
            "memory_persistence": inspect_memory_persistence(
                self.scope.eventloom_path,
                session_id=session_id,
            ),
            "memory_activation": inspect_memory_activation(
                eventloom_path=self.scope.eventloom_path,
            ),
            "purpose": self._purpose_status_body(session_id=session_id),
        }

    def _purpose_status_body(self, *, session_id: str | None) -> dict[str, Any]:
        """Return replay-only purpose diagnostics for the dashboard overview."""
        from zaxy.purpose_control import build_purpose_status

        return build_purpose_status(self.scope.eventloom_path, session_id=session_id)


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


def _bool_param(
    params: dict[str, list[str]],
    name: str,
    *,
    default: bool,
) -> bool:
    value = _str_param(params, name)
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}


def _json_body(body: str | bytes | None) -> dict[str, Any] | tuple[int, dict[str, Any]]:
    if body is None or body == b"" or body == "":
        return {}
    try:
        text = body.decode("utf-8") if isinstance(body, bytes) else body
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 400, {"error": "invalid_json"}
    if not isinstance(payload, dict):
        return 400, {"error": "invalid_json_object"}
    return payload


def _optional_body_str(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _body_bool(payload: dict[str, Any], name: str, *, default: bool) -> bool:
    value = payload.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).casefold() in {"1", "true", "yes", "on"}


def build_dashboard_graph_provider(scope: DashboardScope) -> DashboardGraphProvider:
    """Build the configured graph provider with Eventloom as the local fallback."""
    fallback = EventloomDashboardGraphProvider(scope.eventloom_path)
    if scope.projection_backend == "pggraph":
        if not scope.pggraph_dsn:
            return fallback
        return FallbackDashboardGraphProvider(
            ProjectionDashboardGraphProvider("pggraph", pggraph_dsn=scope.pggraph_dsn),
            fallback,
        )
    if scope.projection_backend == "embedded":
        if scope.embedded_graph_path is None:
            return fallback
        return FallbackDashboardGraphProvider(
            EmbeddedDashboardGraphProvider(scope.embedded_graph_path),
            fallback,
        )
    if not (scope.neo4j_uri and scope.neo4j_user and scope.neo4j_password):
        return fallback
    return FallbackDashboardGraphProvider(
        Neo4jDashboardGraphProvider(scope.neo4j_uri, scope.neo4j_user, scope.neo4j_password),
        fallback,
    )


def _graph_error(exc: Exception) -> dict[str, object]:
    return {
        "available": False,
        "nodes": [],
        "edges": [],
        "warning": str(exc),
    }


def _empty_embedded_dashboard_summary() -> dict[str, object]:
    return {
        "available": True,
        "source": "embedded",
        "nodes": 0,
        "edges": 0,
        "elements": {"nodes": [], "edges": []},
    }


def _empty_embedded_dashboard_elements(*, view: str | None = None, hops: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "available": True,
        "source": "embedded",
        "nodes": [],
        "edges": [],
    }
    if view is not None:
        result["view"] = view
    if hops is not None:
        result["hops"] = hops
    return result


def _is_missing_embedded_projection_table_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "Table Entity does not exist" in message or "Table RELATES does not exist" in message


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def _eventlog_paths(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    if not base.exists():
        return []
    return sorted(path for path in base.glob("*.jsonl") if path.is_file())


def _session_log_path(base: Path, session_id: str | None) -> Path:
    if base.is_file():
        return base
    if session_id is None:
        raise ValueError("session_id is required")
    return base / f"{session_id}.jsonl"


def _event_node(session_id: str, event: Event) -> dict[str, Any]:
    return {
        "id": f"event:{session_id}:{event.seq}",
        "label": f"{event.type} #{event.seq}",
        "kind": "event",
        "properties": {
            "actor": event.actor,
            "hash": event.hash,
            "seq": event.seq,
            "session_id": session_id,
            "summary": _event_summary(event),
            "timestamp": event.timestamp,
            "type": event.type,
        },
    }


def _event_edge(session_id: str, previous: Event, current: Event) -> dict[str, Any]:
    return {
        "id": f"event-edge:{session_id}:{previous.seq}:{current.seq}",
        "source": f"event:{session_id}:{previous.seq}",
        "target": f"event:{session_id}:{current.seq}",
        "label": "NEXT_EVENT",
        "type": "NEXT_EVENT",
        "properties": {"session_id": session_id},
    }


def _event_search_text(session_id: str, event: Event) -> str:
    return " ".join(
        [
            session_id,
            event.type,
            event.actor,
            _event_summary(event),
            json.dumps(event.payload, sort_keys=True, default=str),
        ]
    ).casefold()


def _event_summary(event: Event) -> str:
    for key in ("summary", "decision", "title", "content", "text", "task"):
        value = event.payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    if event.payload:
        return ", ".join(sorted(str(key) for key in event.payload))
    return ""


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


def _pggraph_elements(
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "nodes": [_pggraph_node_payload(row) for row in node_rows],
        "edges": [_pggraph_edge_payload(row) for row in edge_rows],
    }


def _embedded_elements(
    node_rows: list[list[Any]],
    edge_rows: list[list[Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "nodes": [_embedded_node_payload(row) for row in node_rows],
        "edges": [_embedded_edge_payload(row) for row in edge_rows],
    }


def _embedded_fetch_nodes(
    conn: Any,
    *,
    session_id: str | None,
    node_keys: list[str],
    limit: int,
) -> list[list[Any]]:
    if not node_keys:
        return []
    return cast(list[list[Any]], conn.execute(
        """
        MATCH (e:Entity)
        WHERE ($session_id IS NULL OR e.session_id = $session_id)
          AND e.node_key IN $node_keys
          AND e.valid_to IS NULL
        RETURN e.node_key, e.name, e.entity_type, e.summary, e.properties_json,
               e.session_id, e.source_event_seq, e.source_event_hash,
               e.valid_from, e.valid_to
        LIMIT $limit
        """,
        {"session_id": session_id, "node_keys": node_keys, "limit": limit},
    ).get_all())


def _embedded_node_payload(row: list[Any]) -> dict[str, Any]:
    properties = _json_dict(row[4])
    properties.update(
        {
            "entity_type": row[2],
            "name": row[1],
            "session_id": row[5],
            "source_event_seq": row[6],
            "source_event_hash": row[7],
            "summary": row[3],
            "valid_from": row[8],
            "valid_to": row[9],
        }
    )
    return {
        "id": str(row[0]),
        "label": str(row[1]),
        "labels": [str(row[2] or "Entity")],
        "properties": _json_safe_properties(properties),
    }


def _embedded_edge_payload(row: list[Any]) -> dict[str, Any]:
    properties = _json_dict(row[6])
    properties.update(
        {
            "session_id": row[3],
            "source_event_seq": row[4],
            "source_event_hash": row[5],
        }
    )
    source = str(row[0])
    target = str(row[1])
    relation = str(row[2] or "")
    return {
        "id": f"{source}->{relation}->{target}",
        "source": source,
        "target": target,
        "label": relation,
        "type": relation,
        "properties": _json_safe_properties(properties),
    }


def _json_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _pggraph_node_payload(row: dict[str, Any]) -> dict[str, Any]:
    properties = _json_safe_properties(dict(row.get("properties") or {}))
    properties.update(
        {
            "entity_type": row.get("entity_type"),
            "name": row.get("name"),
            "session_id": row.get("session_id"),
            "source_event_seq": row.get("source_event_seq"),
            "source_event_hash": row.get("source_event_hash"),
            "summary": row.get("summary"),
            "valid_from": row.get("valid_from"),
            "valid_to": row.get("valid_to"),
        }
    )
    safe_properties = _json_safe_properties(properties)
    return {
        "id": str(row.get("node_key") or row.get("name") or ""),
        "label": str(row.get("name") or row.get("node_key") or ""),
        "labels": [str(row.get("entity_type") or "Entity")],
        "properties": safe_properties,
    }


def _pggraph_edge_payload(row: dict[str, Any]) -> dict[str, Any]:
    properties = _json_safe_properties(dict(row.get("properties") or {}))
    properties.update(
        {
            "session_id": row.get("session_id"),
            "source_event_seq": row.get("source_event_seq"),
            "source_event_hash": row.get("source_event_hash"),
        }
    )
    return {
        "id": str(row.get("edge_key") or ""),
        "source": str(row.get("source_node_key") or ""),
        "target": str(row.get("target_node_key") or ""),
        "label": str(row.get("relation_type") or ""),
        "type": str(row.get("relation_type") or ""),
        "properties": _json_safe_properties(properties),
    }


def _json_safe_properties(properties: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in properties.items():
        safe[key] = _json_safe_value(value)
    return safe


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return value


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
    .graph-layout { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 12px; }
    .graph-toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
    .graph-toolbar input {
      flex: 1;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    .graph-toolbar button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }
    .checkout-toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
    .checkout-toolbar input {
      flex: 1;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    .checkout-toolbar button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }
    #graph-canvas { height: 560px; border: 1px solid var(--line); border-radius: 8px; background: #ffffff; }
    #graph-detail { max-height: 560px; overflow: auto; }
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
      .graph-layout { grid-template-columns: 1fr; }
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
    <button data-tab="coordinate">Coordinate</button>
    <button data-tab="purpose">Purpose</button>
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
        <div class="metric"><span>Last bootstrap</span><strong id="metric-last-bootstrap">-</strong></div>
        <div class="metric"><span>Last checkout</span><strong id="metric-last-checkout">-</strong></div>
        <div class="metric"><span>Activation</span><strong id="metric-activation">-</strong></div>
        <div class="metric"><span>Activation rate</span><strong id="metric-activation-rate">-</strong></div>
        <div class="metric"><span>Checkout tokens</span><strong id="metric-checkout-tokens">-</strong></div>
        <div class="metric"><span>Facts / 1k tokens</span><strong id="metric-checkout-facts-per-token">-</strong></div>
        <div class="metric"><span>Latest capture</span><strong id="metric-latest-capture">-</strong></div>
        <div class="metric"><span>Latest reminder</span><strong id="metric-latest-reminder">-</strong></div>
        <div class="metric"><span>Last feedback</span><strong id="metric-last-feedback">-</strong></div>
        <div class="metric"><span>Purpose profile</span><strong id="metric-purpose-profile">-</strong></div>
        <div class="metric"><span>Suppressed rows</span><strong id="metric-purpose-suppressed">-</strong></div>
      </div>
      <div class="panel warning" id="memory-persistence-warning"></div>
      <div class="panel warning" id="memory-activation-warning"></div>
      <div class="panel"><pre id="status-json">{}</pre></div>
    </section>
    <section class="tab" id="sessions"><div class="panel"><table><thead><tr><th>Session</th><th>Events</th><th>Latest</th><th>Integrity</th></tr></thead><tbody id="sessions-body"></tbody></table></div></section>
    <section class="tab" id="graph">
      <div class="panel warning" id="graph-warning"></div>
      <div class="graph-toolbar">
        <input id="graph-search" type="search" placeholder="Search graph nodes">
        <button id="graph-search-button" type="button">Search</button>
        <button id="graph-reset-button" type="button">Overview</button>
        <button id="graph-expand-button" type="button">Expand</button>
      </div>
      <div class="graph-layout">
        <div id="graph-canvas"></div>
        <div class="panel" id="graph-detail"><pre>Select a node or edge.</pre></div>
      </div>
    </section>
    <section class="tab" id="checkout">
      <div class="panel">
        <div class="checkout-toolbar">
          <input id="checkout-query" type="search" placeholder="Checkout query">
          <button id="checkout-run-button" type="button">Run checkout</button>
        </div>
        <pre id="checkout-json">{}</pre>
      </div>
    </section>
    <section class="tab" id="purpose">
      <div class="panel">
        <div class="grid">
          <div class="metric"><span>Active profile</span><strong id="purpose-active-profile">-</strong></div>
          <div class="metric"><span>Evidence policy</span><strong id="purpose-evidence-status">-</strong></div>
          <div class="metric"><span>Suppressed</span><strong id="purpose-suppressed-count">-</strong></div>
          <div class="metric"><span>Feedback</span><strong id="purpose-feedback-count">-</strong></div>
          <div class="metric"><span>Coordinate accepted</span><strong id="purpose-coordinate-accepted">-</strong></div>
          <div class="metric"><span>Coordinate pending</span><strong id="purpose-coordinate-pending">-</strong></div>
          <div class="metric"><span>Coordinate stale</span><strong id="purpose-coordinate-stale">-</strong></div>
          <div class="metric"><span>Proof packets</span><strong id="purpose-coordinate-proof-packets">-</strong></div>
        </div>
        <table>
          <thead><tr><th>Profile</th><th>Checkouts</th><th>Evidence failures</th><th>Suppressed</th><th>Feedback</th></tr></thead>
          <tbody id="purpose-lanes-body"></tbody>
        </table>
        <table>
          <thead><tr><th>Target</th><th>Profile</th><th>Outcome</th><th>Latest</th></tr></thead>
          <tbody id="purpose-feedback-body"></tbody>
        </table>
        <pre id="purpose-json">{}</pre>
      </div>
    </section>
    <section class="tab" id="coordinate">
      <div class="panel">
        <div class="checkout-toolbar">
          <input id="coordination-mission-id" type="search" placeholder="Mission ID">
          <button id="coordination-load-button" type="button">Load mission</button>
          <button id="coordination-review-export-button" type="button">Review export</button>
        </div>
        <div id="coordinate-review-status"></div>
        <div class="grid">
          <div class="metric"><span>Workers</span><strong id="metric-coordinate-workers">-</strong></div>
          <div class="metric"><span>Accepted</span><strong id="metric-coordinate-accepted">-</strong></div>
          <div class="metric"><span>Pending</span><strong id="metric-coordinate-pending">-</strong></div>
          <div class="metric"><span>Conflicts</span><strong id="metric-coordinate-conflicts">-</strong></div>
          <div class="metric"><span>Stale</span><strong id="metric-coordinate-stale">-</strong></div>
        </div>
        <table>
          <thead><tr><th>Worker</th><th>Assignment</th><th>Status</th></tr></thead>
          <tbody id="coordinate-workers-body"></tbody>
        </table>
        <table>
          <thead><tr><th>Status</th><th>Finding</th><th>Worker</th><th>Summary</th><th>Evidence</th></tr></thead>
          <tbody id="coordinate-findings-body"></tbody>
        </table>
        <pre id="coordinate-review-export"></pre>
        <pre id="coordination-json">{}</pre>
      </div>
    </section>
    <section class="tab" id="events"><div class="panel"><table><thead><tr><th>Session</th><th>Seq</th><th>Type</th><th>Actor</th><th>Summary</th></tr></thead><tbody id="events-body"></tbody></table></div></section>
  </main>
  <script>
    const statusUrl = "/api/status";
    const eventsUrl = "/api/events?limit=25";
    const graphUrl = "/api/graph/summary";
    const graphSearchUrl = "/api/graph/search";
    const graphNeighborhoodUrl = "/api/graph/neighborhood";
    const checkoutUrl = "/api/checkout";
    const purposeStatusUrl = "/api/purpose/status";
    const purposeFeedbackUrl = "/api/purpose/feedback";
    const coordinationMissionUrl = "/api/coordination/mission";
    const coordinationReviewExportUrl = "/api/coordinate/review-export";
    const coordinationReviewUrl = "/api/coordinate/review-finding";
    const coordinationApplyApprovalUrl = "/api/coordinate/apply-approval";
    let selectedGraphNodeId = null;

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
          { selector: "node", style: { "label": "data(label)", "background-color": "#1663c7", "color": "#17202a", "font-size": 10, "width": 18, "height": 18 } },
          { selector: "node:selected", style: { "border-width": 3, "border-color": "#1d7f5f" } },
          { selector: "edge", style: { "label": "data(label)", "line-color": "#8c98a8", "target-arrow-color": "#8c98a8", "target-arrow-shape": "triangle", "curve-style": "bezier", "font-size": 8 } }
        ],
        layout: { name: "grid" }
      });
      cy.on("tap", "node", (event) => {
        selectedGraphNodeId = event.target.id();
        renderGraphDetail(event.target.data());
      });
      cy.on("tap", "edge", (event) => {
        renderGraphDetail(event.target.data());
      });
    }

    async function refresh() {
      const status = await fetch(statusUrl).then((response) => response.json());
      const events = await fetch(eventsUrl).then((response) => response.json());
      document.getElementById("scope").textContent = `${status.scope.workspace} | ${status.scope.eventloom_path} | session=${status.scope.session_id || "all"}`;
      document.getElementById("metric-sessions").textContent = status.memory.session_count;
      document.getElementById("metric-events").textContent = status.memory.total_events;
      document.getElementById("metric-last-bootstrap").textContent = status.memory_persistence.last_bootstrap_seq || "-";
      document.getElementById("metric-last-checkout").textContent = status.memory_persistence.last_checkout_seq || "-";
      document.getElementById("metric-activation").textContent = status.memory_activation.status || "-";
      document.getElementById("metric-activation-rate").textContent = formatActivationRate(status.memory_activation.activation_efficiency);
      const checkoutEfficiency = status.memory_activation.latest_checkout ? status.memory_activation.latest_checkout.token_efficiency : null;
      document.getElementById("metric-checkout-tokens").textContent = checkoutEfficiency && checkoutEfficiency.prompt_tokens !== undefined ? checkoutEfficiency.prompt_tokens : "-";
      document.getElementById("metric-checkout-facts-per-token").textContent = checkoutEfficiency && checkoutEfficiency.facts_per_1k_prompt_tokens !== undefined ? checkoutEfficiency.facts_per_1k_prompt_tokens : "-";
      document.getElementById("metric-latest-capture").textContent = status.memory_activation.latest_capture ? status.memory_activation.latest_capture.seq : "-";
      document.getElementById("metric-latest-reminder").textContent = status.memory_activation.latest_reminder ? status.memory_activation.latest_reminder.seq : "-";
      document.getElementById("metric-last-feedback").textContent = status.memory_persistence.last_feedback_seq || "-";
      renderPurposeStatus(status.purpose || {});
      document.getElementById("memory-persistence-warning").textContent = status.memory_persistence.warning || "";
      document.getElementById("memory-activation-warning").textContent = status.memory_activation.status === "ok" ? "" : `${status.memory_activation.message}: ${(status.memory_activation.actions || []).join(" ")}`;
      document.getElementById("status-json").textContent = JSON.stringify(status, null, 2);
      document.getElementById("sessions-body").innerHTML = status.memory.sessions.map((session) => `
        <tr><td><code>${escapeHtml(session.session_id)}</code></td><td>${session.event_count}</td><td>${escapeHtml(session.latest_type || "")}</td><td>${session.integrity_ok ? "OK" : "FAILED"}</td></tr>
      `).join("");
      document.getElementById("events-body").innerHTML = events.events.map((event) => `
        <tr><td><code>${escapeHtml(event.session_id)}</code></td><td>${event.seq}</td><td>${escapeHtml(event.type)}</td><td>${escapeHtml(event.actor)}</td><td>${escapeHtml(event.summary || "")}</td></tr>
      `).join("");
    }

    function formatActivationRate(efficiency) {
      if (!efficiency || efficiency.fresh_checkout_rate === null || efficiency.fresh_checkout_rate === undefined) {
        return "-";
      }
      return `${Math.round(efficiency.fresh_checkout_rate * 1000) / 10}%`;
    }
    async function refreshGraph() {
      const graph = await fetch(graphUrl).then((response) => response.json());
      document.getElementById("metric-nodes").textContent = graph.graph.nodes || 0;
      document.getElementById("metric-edges").textContent = graph.graph.edges || 0;
      document.getElementById("graph-warning").textContent = graph.graph.warning || "";
      renderGraphPayload(graph.graph);
    }
    async function searchGraph() {
      const query = document.getElementById("graph-search").value.trim();
      if (!query) {
        await refreshGraph();
        return;
      }
      const graph = await fetch(`${graphSearchUrl}?q=${encodeURIComponent(query)}&limit=80`).then((response) => response.json());
      renderGraphPayload(graph.graph);
    }
    async function expandSelectedNode() {
      if (!selectedGraphNodeId) {
        document.getElementById("graph-detail").innerHTML = "<pre>Select a node before expanding.</pre>";
        return;
      }
      const graph = await fetch(`${graphNeighborhoodUrl}?node_id=${encodeURIComponent(selectedGraphNodeId)}&hops=1&limit=120`).then((response) => response.json());
      renderGraphPayload(graph.graph);
    }
    async function runCheckout() {
      const query = document.getElementById("checkout-query").value.trim();
      if (!query) {
        document.getElementById("checkout-json").textContent = "Enter a checkout query.";
        return;
      }
      const checkout = await fetch(`${checkoutUrl}?query=${encodeURIComponent(query)}&limit=10`).then((response) => response.json());
      document.getElementById("checkout-json").textContent = JSON.stringify(checkout, null, 2);
    }
    function renderPurposeStatus(purpose) {
      const suppression = purpose.suppression || {};
      const consequence = purpose.consequence_history || {};
      const coordinate = purpose.coordinate || {};
      const missions = coordinate.missions || [];
      const coordinateTotals = missions.reduce((totals, mission) => ({
        accepted: totals.accepted + (mission.accepted_count || 0),
        pending: totals.pending + (mission.pending_count || 0),
        stale: totals.stale + (mission.stale_count || 0),
        proofPackets: totals.proofPackets + (mission.proof_packet_count || 0)
      }), { accepted: 0, pending: 0, stale: 0, proofPackets: 0 });
      document.getElementById("metric-purpose-profile").textContent = purpose.active_profile || "-";
      document.getElementById("metric-purpose-suppressed").textContent = suppression.count || 0;
      document.getElementById("purpose-active-profile").textContent = purpose.active_profile || "-";
      document.getElementById("purpose-evidence-status").textContent = purpose.evidence_policy_status ? purpose.evidence_policy_status.status : "-";
      document.getElementById("purpose-suppressed-count").textContent = suppression.count || 0;
      document.getElementById("purpose-feedback-count").textContent = `+${consequence.positive_count || 0}/-${consequence.negative_count || 0}`;
      document.getElementById("purpose-coordinate-accepted").textContent = coordinateTotals.accepted;
      document.getElementById("purpose-coordinate-pending").textContent = coordinateTotals.pending;
      document.getElementById("purpose-coordinate-stale").textContent = coordinateTotals.stale;
      document.getElementById("purpose-coordinate-proof-packets").textContent = coordinateTotals.proofPackets;
      document.getElementById("purpose-lanes-body").innerHTML = (purpose.lanes || []).map((lane) => `
        <tr><td><code>${escapeHtml(lane.profile || "")}</code></td><td>${lane.checkout_count || 0}</td><td>${lane.evidence_policy_fail_count || 0}</td><td>${lane.suppressed_count || 0}</td><td>+${lane.positive_feedback_count || 0}/-${lane.negative_feedback_count || 0}</td></tr>
      `).join("");
      document.getElementById("purpose-feedback-body").innerHTML = (consequence.targets || []).map((target) => `
        <tr><td><code>${escapeHtml(target.target || "")}</code></td><td>${escapeHtml(target.profile || "")}</td><td>+${target.positive_count || 0}/-${target.negative_count || 0}${target.suppression_candidate ? " suppress" : ""}</td><td>${target.latest_event ? target.latest_event.seq : ""}</td></tr>
      `).join("");
      document.getElementById("purpose-json").textContent = JSON.stringify(purpose, null, 2);
    }
    async function loadCoordinationMission() {
      const missionId = document.getElementById("coordination-mission-id").value.trim();
      if (!missionId) {
        document.getElementById("coordination-json").textContent = "Enter a mission ID.";
        return;
      }
      const payload = await fetch(`${coordinationMissionUrl}?mission_id=${encodeURIComponent(missionId)}&include_diagnostics=true`).then((response) => response.json());
      renderCoordinationMission(payload);
    }
    async function loadCoordinationReviewExport() {
      const missionId = document.getElementById("coordination-mission-id").value.trim();
      if (!missionId) {
        document.getElementById("coordinate-review-export").textContent = "Enter a mission ID.";
        return;
      }
      const payload = await fetch(`${coordinationReviewExportUrl}?mission_id=${encodeURIComponent(missionId)}`).then((response) => response.json());
      document.getElementById("coordinate-review-export").textContent = payload.review_export ? payload.review_export.markdown : JSON.stringify(payload, null, 2);
    }
    function renderCoordinationMission(payload) {
      const mission = payload.coordination || {};
      const brief = mission.brief || {};
      const reviewEnabled = Boolean(mission.review_enabled);
      const findings = [
        ...tagFindings("accepted", brief.accepted_findings),
        ...tagFindings("pending", brief.pending_findings),
        ...tagFindings("rejected", brief.rejected_findings),
        ...tagFindings("deferred", brief.deferred_findings),
        ...tagFindings("conflicted", brief.conflicted_findings),
        ...tagFindings("stale", brief.stale_findings)
      ];
      document.getElementById("metric-coordinate-workers").textContent = (brief.workers || []).length;
      document.getElementById("metric-coordinate-accepted").textContent = (brief.accepted_findings || []).length;
      document.getElementById("metric-coordinate-pending").textContent = (brief.pending_findings || []).length;
      document.getElementById("metric-coordinate-conflicts").textContent = (brief.conflicts || []).length;
      document.getElementById("metric-coordinate-stale").textContent = (brief.stale_findings || []).length;
      document.getElementById("coordinate-workers-body").innerHTML = (brief.workers || []).map((worker) => `
        <tr><td><code>${escapeHtml(worker.worker_id || "")}</code></td><td>${escapeHtml(worker.assignment || "")}</td><td>${escapeHtml(worker.status || "")}</td></tr>
      `).join("");
      document.getElementById("coordinate-findings-body").innerHTML = findings.map((finding) => `
        <tr><td>${escapeHtml(finding._status || "")}${renderReviewControls(finding, reviewEnabled)}</td><td><code>${escapeHtml(finding.finding_id || "")}</code></td><td><code>${escapeHtml(finding.worker_id || "")}</code></td><td>${escapeHtml(finding.summary || "")}</td><td>${escapeHtml(formatEvidence(finding.evidence || []))}</td></tr>
      `).join("");
      document.getElementById("coordination-json").textContent = JSON.stringify(payload, null, 2);
    }
    function renderReviewControls(finding, reviewEnabled) {
      if (!reviewEnabled || finding._status === "accepted") {
        return "";
      }
      const findingId = escapeHtml(finding.finding_id || "");
      return `
        <div class="review-controls">
          <button type="button" data-finding-id="${findingId}" data-review-status="accepted" data-promote="true">Accept</button>
          <button type="button" data-finding-id="${findingId}" data-review-status="rejected" data-promote="false">Reject</button>
          <button type="button" data-finding-id="${findingId}" data-review-status="deferred" data-promote="false">Defer</button>
        </div>
      `;
    }
    async function reviewFinding(findingId, status, promote) {
      const missionId = document.getElementById("coordination-mission-id").value.trim();
      if (!missionId || !findingId || !status) {
        return;
      }
      const params = new URLSearchParams({
        mission_id: missionId,
        finding_id: findingId,
        status,
        promote: promote ? "true" : "false"
      });
      const payload = await fetch(`${coordinationReviewUrl}?${params.toString()}`, { method: "POST" }).then((response) => response.json());
      document.getElementById("coordinate-review-status").textContent = JSON.stringify(payload.review || payload.approval_result || payload, null, 2);
      renderCoordinationMission(payload);
    }
    function tagFindings(status, findings) {
      return (findings || []).map((finding) => ({ ...finding, _status: status }));
    }
    function formatEvidence(evidence) {
      return evidence.map((item) => item.reference || item.summary || item.kind || "").filter(Boolean).join("; ");
    }
    function renderGraphPayload(graph) {
      const elements = graph.elements || { nodes: graph.nodes || [], edges: graph.edges || [] };
      if (!cy || !elements.nodes) {
        return;
      }
      cy.elements().remove();
      cy.add([
        ...elements.nodes.map((node) => ({ data: graphNodeData(node) })),
        ...(elements.edges || []).map((edge) => ({ data: graphEdgeData(edge) }))
      ]);
      cy.layout({ name: "breadthfirst", directed: true, padding: 24 }).run();
      document.getElementById("graph-warning").textContent = graph.warning || "";
    }
    function graphNodeData(node) {
      return {
        id: node.id,
        label: node.label || node.id,
        labels: node.labels || [],
        kind: node.kind || "node",
        properties: node.properties || {}
      };
    }
    function graphEdgeData(edge) {
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.label || edge.type || "",
        type: edge.type || edge.label || "",
        properties: edge.properties || {}
      };
    }
    function renderGraphDetail(data) {
      document.getElementById("graph-detail").innerHTML = `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
    }
    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, (character) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[character]));
    }
    document.getElementById("graph-search-button").addEventListener("click", () => {
      searchGraph().catch((error) => {
        document.getElementById("graph-warning").textContent = String(error);
      });
    });
    document.getElementById("graph-search").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("graph-search-button").click();
      }
    });
    document.getElementById("graph-reset-button").addEventListener("click", () => {
      refreshGraph().catch((error) => {
        document.getElementById("graph-warning").textContent = String(error);
      });
    });
    document.getElementById("graph-expand-button").addEventListener("click", () => {
      expandSelectedNode().catch((error) => {
        document.getElementById("graph-warning").textContent = String(error);
      });
    });
    document.getElementById("checkout-run-button").addEventListener("click", () => {
      runCheckout().catch((error) => {
        document.getElementById("checkout-json").textContent = String(error);
      });
    });
    document.getElementById("checkout-query").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("checkout-run-button").click();
      }
    });
    document.getElementById("coordination-load-button").addEventListener("click", () => {
      loadCoordinationMission().catch((error) => {
        document.getElementById("coordination-json").textContent = String(error);
      });
    });
    document.getElementById("coordination-review-export-button").addEventListener("click", () => {
      loadCoordinationReviewExport().catch((error) => {
        document.getElementById("coordinate-review-export").textContent = String(error);
      });
    });
    document.getElementById("coordinate-findings-body").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-review-status]");
      if (!button) {
        return;
      }
      reviewFinding(button.dataset.findingId, button.dataset.reviewStatus, button.dataset.promote === "true").catch((error) => {
        document.getElementById("coordination-json").textContent = String(error);
      });
    });
    document.getElementById("coordination-mission-id").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("coordination-load-button").click();
      }
    });
    refresh().catch((error) => {
      document.getElementById("status-json").textContent = String(error);
    });
    refreshGraph().catch((error) => {
      document.getElementById("graph-warning").textContent = String(error);
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
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/coordinate/"):
                content_length = int(self.headers.get("content-length") or "0")
                if content_length > 65_536:
                    self._write_json(
                        413,
                        {"content-type": "application/json; charset=utf-8"},
                        {"error": "request_too_large"},
                    )
                    return
                body = self.rfile.read(content_length) if content_length else b""
                status, headers, response = self.dashboard_app.handle_api(
                    "POST",
                    parsed.path,
                    parsed.query,
                    body=body,
                    request_headers=dict(self.headers.items()),
                )
                self._write_json(status, headers, response)
                return
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
