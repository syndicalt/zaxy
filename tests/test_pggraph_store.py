"""Tests for the experimental pgGraph projection store."""

from __future__ import annotations

import os
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
        if self.connection.row_sets:
            return self.connection.row_sets.pop(0)
        return self.rows


class FakeConnection:
    """Small async connection capturing statements and transaction calls."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, object | None]] = []
        self.cursor_obj = FakeCursor(self)
        self.row_sets: list[list[dict[str, Any]]] = []
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
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE IF NOT EXISTS zaxy_pggraph_entities" in sql
    assert "embedding_vector vector" in sql
    assert "ALTER TABLE zaxy_pggraph_entities ADD COLUMN IF NOT EXISTS embedding_vector vector" in sql
    assert "search_vector tsvector" in sql
    assert "ALTER TABLE zaxy_pggraph_entities ADD COLUMN IF NOT EXISTS search_vector tsvector" in sql
    assert "zaxy_pggraph_entities_search_vector_idx" in sql
    assert "CREATE TABLE IF NOT EXISTS zaxy_pggraph_edges" in sql
    assert "graph.add_table" in sql
    assert "graph.add_edge" in sql
    assert "to_column := 'target_node_key'" in sql
    assert "graph.build" in sql
    assert connection.commits == 1


@pytest.mark.asyncio
async def test_pggraph_store_close_closes_existing_connection() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.close()

    assert connection.closed is True


def test_pggraph_store_dsn_returns_configured_connection_string() -> None:
    store = PgGraphStore("postgresql://test")

    assert store.dsn == "postgresql://test"


@pytest.mark.asyncio
async def test_pggraph_store_requires_connection_before_operations() -> None:
    store = PgGraphStore("postgresql://test")

    with pytest.raises(AssertionError, match="Call connect"):
        await store.init_schema()


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
                embedding=[0.1, 0.2, 0.3],
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
    assert "%(embedding_vector)s::vector" in sql
    assert "search_vector = to_tsvector" in sql
    assert "INSERT INTO zaxy_pggraph_edges" in sql
    entity_params = [
        params
        for statement, params in connection.statements
        if "INSERT INTO zaxy_pggraph_entities" in statement and isinstance(params, dict)
    ][0]
    assert entity_params["embedding_vector"] == "[0.1,0.2,0.3]"
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
    sql = connection.statements[-1][0]
    assert "%(entity_type)s::text IS NULL" in sql


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
    sql = connection.statements[-1][0]
    assert "to_tsquery('simple', %(tsquery)s)" in sql
    assert "entity.search_vector @@ search.query" in sql
    assert "ts_rank_cd" in sql
    assert "ILIKE" in sql
    assert isinstance(connection.statements[-1][1], dict)
    assert connection.statements[-1][1]["tsquery"] == "memory"


@pytest.mark.asyncio
async def test_pggraph_store_search_keyword_builds_natural_language_tsquery() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.search_keyword(
        "Could you remind me of that vegan eatery with multiple locations in the city?",
        session_id="agent-1",
    )

    _sql, params = connection.statements[-1]
    assert isinstance(params, dict)
    assert params["tsquery"] == "vegan | eatery | multiple | locations | city"


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


@pytest.mark.asyncio
async def test_pggraph_store_search_traversal_uses_pggraph_traverse() -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [
        {
            "node": {
                "name": "pgGraph",
                "entity_type": "backend",
                "valid_from": "2026-05-18T00:00:00Z",
                "valid_to": None,
                "summary": "Postgres graph extension",
                "properties": {},
                "session_id": "agent-1",
            }
        }
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    results = await store.search_traversal(
        "Zaxy",
        relation_type="evaluates",
        session_id="agent-1",
    )

    assert results[0].name == "pgGraph"
    assert "graph.traverse" in connection.statements[-1][0]


@pytest.mark.asyncio
async def test_pggraph_store_search_traversal_accepts_flat_rows_and_json_properties() -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [
        {
            "name": "pgGraph",
            "entity_type": "backend",
            "valid_from": "2026-05-18T00:00:00Z",
            "valid_to": None,
            "summary": "Postgres graph extension",
            "properties": '{"path": "docs/pggraph.md"}',
            "session_id": "agent-1",
        }
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    results = await store.search_traversal("Zaxy", session_id="agent-1")

    assert results[0].name == "pgGraph"
    assert results[0].properties["summary"] == "Postgres graph extension"
    assert results[0].properties["path"] == "docs/pggraph.md"


@pytest.mark.asyncio
async def test_pggraph_store_has_traversal_edges_checks_active_edges() -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [{"has_edges": True}]
    store = PgGraphStore("postgresql://test", connection=connection)

    assert await store.has_traversal_edges(session_id="agent-1") is True

    sql, params = connection.statements[-1]
    assert "zaxy_pggraph_edges" in sql
    assert "edge.valid_to IS NULL" in sql
    assert params == {"session_id": "agent-1"}


@pytest.mark.asyncio
async def test_pggraph_store_reset_benchmark_projection_clears_projection_tables() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.reset_benchmark_projection()

    sql = "\n".join(statement for statement, _params in connection.statements)
    assert "TRUNCATE TABLE" in sql
    assert "zaxy_pggraph_edges" in sql
    assert "zaxy_pggraph_entities" in sql
    assert "zaxy_pggraph_events" in sql
    assert "graph.build" in sql
    assert connection.commits == 1


@pytest.mark.asyncio
async def test_pggraph_store_inspect_event_projection_status_reports_lag_and_missing_links() -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [
        {
            "event_count": 3,
            "latest_seq": 3,
            "latest_hash": "b" * 64,
            "next_event_edges": 1,
            "previous_event_edges": 1,
            "missing_chain_links": 1,
        }
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=4,
        eventloom_latest_hash="c" * 64,
    )

    assert status.session_id == "agent-1"
    assert status.event_count == 3
    assert status.latest_seq == 3
    assert status.latest_hash == "b" * 64
    assert status.eventloom_latest_seq == 4
    assert status.eventloom_latest_hash == "c" * 64
    assert status.projection_lag == 1
    assert status.latest_hash_matches is False
    assert status.next_event_edges == 1
    assert status.previous_event_edges == 1
    assert status.missing_chain_links == 1
    assert status.integrity_ok is False
    sql, params = connection.statements[-1]
    assert "zaxy_pggraph_events" in sql
    assert "prev.hash = event.prev_hash" in sql
    assert params == {"session_id": "agent-1"}


@pytest.mark.asyncio
async def test_pggraph_store_inspect_inferred_edge_status_reports_method_and_evidence_coverage() -> None:
    connection = FakeConnection()
    connection.row_sets = [
        [
            {
                "method": "task_completed_decision_citation_v1",
                "edge_count": 2,
                "relation_types": ["likely_implemented_decision"],
                "average_confidence": 0.85,
                "minimum_confidence": 0.8,
                "evidence_count": 1,
                "missing_evidence_count": 1,
                "missing_source_event_count": 0,
            },
            {
                "method": "unknown",
                "edge_count": 1,
                "relation_types": ["likely_informed"],
                "average_confidence": None,
                "minimum_confidence": None,
                "evidence_count": 0,
                "missing_evidence_count": 1,
                "missing_source_event_count": 1,
            },
        ],
        [
            {
                "source": "task-7",
                "target": "decision:Use graph audit",
                "relation_type": "likely_implemented_decision",
                "confidence": 0.86,
                "method": "task_completed_decision_citation_v1",
                "source_event_seq": 12,
                "source_event_hash": "a" * 64,
                "evidence_keys": ["evidence_reason", "evidence_source_event_seq"],
            }
        ],
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    status = await store.inspect_inferred_edge_status("agent-1", limit=5)

    assert status.session_id == "agent-1"
    assert status.total_edges == 3
    assert status.method_count == 2
    assert status.evidence_count == 1
    assert status.missing_evidence_count == 2
    assert status.missing_source_event_count == 1
    assert status.evidence_coverage == pytest.approx(1 / 3)
    assert status.methods[0].method == "task_completed_decision_citation_v1"
    assert status.methods[0].edge_count == 2
    assert status.methods[0].average_confidence == pytest.approx(0.85)
    assert status.methods[1].method == "unknown"
    assert status.samples[0].source == "task-7"
    assert status.samples[0].evidence_keys == (
        "evidence_reason",
        "evidence_source_event_seq",
    )
    sql = "\n".join(statement for statement, _params in connection.statements)
    assert "zaxy_pggraph_edges" in sql
    assert "jsonb_object_keys(properties)" in sql
    assert "inferred = true" in sql


@pytest.mark.asyncio
async def test_pggraph_store_inspect_inferred_edge_status_handles_empty_projection() -> None:
    connection = FakeConnection()
    connection.row_sets = [[], []]
    store = PgGraphStore("postgresql://test", connection=connection)

    status = await store.inspect_inferred_edge_status("agent-1")

    assert status.total_edges == 0
    assert status.method_count == 0
    assert status.evidence_coverage == 1.0
    assert status.methods == ()
    assert status.samples == ()


@pytest.mark.asyncio
async def test_pggraph_store_inspect_inferred_edge_status_normalizes_unexpected_array_values() -> None:
    connection = FakeConnection()
    connection.row_sets = [
        [
            {
                "method": None,
                "edge_count": "1",
                "relation_types": "not-an-array",
                "average_confidence": "0.7",
                "minimum_confidence": "0.6",
                "evidence_count": "0",
                "missing_evidence_count": "1",
                "missing_source_event_count": "1",
            }
        ],
        [
            {
                "source": "task-7",
                "target": "decision:Use graph audit",
                "relation_type": "likely_informed",
                "confidence": "0.7",
                "method": None,
                "source_event_seq": "12",
                "source_event_hash": None,
                "evidence_keys": "not-an-array",
            }
        ],
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    status = await store.inspect_inferred_edge_status("agent-1")

    assert status.methods[0].method == "unknown"
    assert status.methods[0].edge_count == 1
    assert status.methods[0].relation_types == ()
    assert status.methods[0].average_confidence == pytest.approx(0.7)
    assert status.samples[0].method == "unknown"
    assert status.samples[0].source_event_seq == 12
    assert status.samples[0].source_event_hash is None
    assert status.samples[0].evidence_keys == ()


@pytest.mark.asyncio
async def test_pggraph_store_search_vector_uses_pgvector_cosine_distance() -> None:
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
            "score": 0.91,
        }
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    results = await store.search_vector([0.1, 0.2, 0.3], limit=3, session_id="agent-1")

    assert results[0].entity.name == "Zaxy"
    assert results[0].source == "vector"
    assert results[0].score == 0.91
    sql, params = connection.statements[-1]
    assert "embedding_vector <=> %(embedding)s::vector" in sql
    assert "%(temporal_point)s::timestamptz IS NULL" in sql
    assert "embedding_vector IS NOT NULL" in sql
    assert isinstance(params, dict)
    assert params["embedding"] == "[0.1,0.2,0.3]"
    assert params["limit"] == 3


@pytest.mark.integration
async def test_pggraph_store_real_postgres_vector_roundtrip() -> None:
    """Real pgGraph/Postgres coverage should include pgvector ranking."""
    dsn = os.environ.get("PGGRAPH_INTEGRATION_DSN")
    if not dsn:
        pytest.skip("PGGRAPH_INTEGRATION_DSN is required for pgGraph integration tests")
    pytest.importorskip("psycopg")
    store = PgGraphStore(dsn)
    await store.connect()
    try:
        connection = store._require_connection()
        await connection.execute("DROP TABLE IF EXISTS zaxy_pggraph_edges")
        await connection.execute("DROP TABLE IF EXISTS zaxy_pggraph_entities")
        await connection.execute("DROP TABLE IF EXISTS zaxy_pggraph_events")
        await connection.commit()
        await store.init_schema()
        await store.upsert_extraction(
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Vector Match",
                        entity_type="memory",
                        observed_at="2026-05-18T00:00:00Z",
                        summary="Nearest vector entity",
                        embedding=[1.0, 0.0, 0.0],
                    ),
                    ExtractedEntity(
                        name="Vector Distractor",
                        entity_type="memory",
                        observed_at="2026-05-18T00:00:00Z",
                        summary="Distant vector entity",
                        embedding=[0.0, 1.0, 0.0],
                    ),
                ],
                edges=[],
                source_event_seq=1,
                source_event_hash="b" * 64,
                source_event_type="integration.vector",
            ),
            session_id="pggraph-integration",
        )

        results = await store.search_vector(
            [1.0, 0.0, 0.0],
            limit=2,
            session_id="pggraph-integration",
        )

        assert [result.entity.name for result in results] == [
            "Vector Match",
            "Vector Distractor",
        ]
        assert results[0].score > results[1].score
    finally:
        await store.close()
