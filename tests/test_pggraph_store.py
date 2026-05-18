"""Tests for the experimental pgGraph projection store."""

from __future__ import annotations

from typing import Any

import pytest

from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult
from zaxy.graph import GraphEntity
from zaxy.pggraph_store import PgGraphStore


class FakeCursor:
    """Small async cursor capturing SQL statements."""

    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.rows: list[dict[str, Any]] = []

    async def __aenter__(self) -> FakeCursor:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def execute(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] | None = None,
    ) -> None:
        self.connection.statements.append((sql, params))

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class FakeConnection:
    """Small async connection capturing statements and transaction calls."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, object | None]] = []
        self.cursor_obj = FakeCursor(self)
        self.commits = 0
        self.closed = False

    def cursor(self, *, row_factory: object | None = None) -> FakeCursor:
        return self.cursor_obj

    async def execute(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] | None = None,
    ) -> None:
        self.statements.append((sql, params))

    async def commit(self) -> None:
        self.commits += 1

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_pggraph_store_init_schema_creates_projection_tables_and_registers_pggraph() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.init_schema()

    sql = "\n".join(statement for statement, _params in connection.statements)
    assert "CREATE TABLE IF NOT EXISTS zaxy_pggraph_entities" in sql
    assert "CREATE TABLE IF NOT EXISTS zaxy_pggraph_edges" in sql
    assert "graph.add_table" in sql
    assert "graph.add_edge" in sql
    assert "graph.build" in sql
    assert connection.commits == 1


@pytest.mark.asyncio
async def test_pggraph_store_close_closes_existing_connection() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.close()

    assert connection.closed is True


@pytest.mark.asyncio
async def test_pggraph_store_upsert_extraction_writes_entities_edges_and_events() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)
    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Zaxy",
                entity_type="project",
                observed_at="2026-05-18T00:00:00Z",
                summary="Memory product",
                properties={"path": "README.md"},
            ),
            ExtractedEntity(
                name="pgGraph",
                entity_type="backend",
                observed_at="2026-05-18T00:00:00Z",
                summary="Postgres graph extension",
            ),
        ],
        edges=[
            ExtractedEdge(
                source="Zaxy",
                target="pgGraph",
                relation_type="evaluates",
                valid_from="2026-05-18T00:00:00Z",
            )
        ],
        source_event_seq=7,
        source_event_hash="a" * 64,
        source_event_type="decision.created",
    )

    await store.upsert_extraction(result, session_id="agent-1")

    sql = "\n".join(statement for statement, _params in connection.statements)
    assert "INSERT INTO zaxy_pggraph_events" in sql
    assert "INSERT INTO zaxy_pggraph_entities" in sql
    assert "INSERT INTO zaxy_pggraph_edges" in sql
    assert connection.commits == 1


@pytest.mark.asyncio
async def test_pggraph_store_search_exact_maps_rows_to_graph_entities() -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [
        {
            "name": "Zaxy",
            "entity_type": "project",
            "valid_from": "2026-05-18T00:00:00Z",
            "valid_to": None,
            "summary": "Memory product",
            "properties": {"path": "README.md"},
            "session_id": "agent-1",
        }
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    results = await store.search_exact("Zaxy", session_id="agent-1")

    assert results == [
        GraphEntity(
            name="Zaxy",
            entity_type="project",
            valid_from="2026-05-18T00:00:00Z",
            valid_to=None,
            properties={"summary": "Memory product", "path": "README.md"},
            session_id="agent-1",
        )
    ]


@pytest.mark.asyncio
async def test_pggraph_store_search_keyword_returns_scored_results() -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [
        {
            "name": "Zaxy",
            "entity_type": "project",
            "valid_from": "2026-05-18T00:00:00Z",
            "valid_to": None,
            "summary": "Memory product",
            "properties": {},
            "session_id": "agent-1",
            "score": 0.8,
        }
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    results = await store.search_keyword("memory", session_id="agent-1")

    assert results[0].entity.name == "Zaxy"
    assert results[0].source == "keyword"
    assert results[0].score == 0.8
    assert "ILIKE" in connection.statements[-1][0]


@pytest.mark.asyncio
async def test_pggraph_store_invalidate_entity_closes_active_rows() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.invalidate_entity(
        name="Zaxy",
        entity_type="project",
        invalid_at="2026-05-19T00:00:00Z",
        session_id="agent-1",
    )

    sql = connection.statements[-1][0]
    assert "UPDATE zaxy_pggraph_entities" in sql
    assert "valid_to = %(invalid_at)s" in sql
    assert connection.commits == 1
