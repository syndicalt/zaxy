"""Experimental pgGraph-backed projection store."""

from __future__ import annotations

from typing import Any

from zaxy.extract import ExtractionResult
from zaxy.graph import (
    GraphEntity,
    GraphEventProjectionStatus,
    GraphInferredEdgeStatus,
    SearchResult,
)

PGGRAPH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS zaxy_pggraph_events (
    session_id text NOT NULL,
    seq bigint NOT NULL,
    hash text,
    prev_hash text,
    event_type text,
    source_thread text,
    projected_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS zaxy_pggraph_entities (
    node_key text PRIMARY KEY,
    session_id text NOT NULL,
    name text NOT NULL,
    entity_type text NOT NULL,
    valid_from timestamptz NOT NULL,
    valid_to timestamptz,
    summary text,
    embedding jsonb,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_event_seq bigint,
    source_event_hash text,
    source_event_type text,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS zaxy_pggraph_edges (
    edge_key text PRIMARY KEY,
    session_id text NOT NULL,
    source_node_key text NOT NULL,
    target_node_key text NOT NULL,
    source_name text NOT NULL,
    target_name text NOT NULL,
    relation_type text NOT NULL,
    valid_from timestamptz NOT NULL,
    valid_to timestamptz,
    inferred boolean NOT NULL DEFAULT false,
    confidence double precision NOT NULL DEFAULT 1.0,
    inference_method text,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_event_seq bigint,
    source_event_hash text,
    source_event_type text,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS zaxy_pggraph_entities_lookup_idx
    ON zaxy_pggraph_entities (session_id, name, entity_type, valid_to);
CREATE INDEX IF NOT EXISTS zaxy_pggraph_entities_keyword_idx
    ON zaxy_pggraph_entities USING gin (to_tsvector('simple', coalesce(name, '') || ' ' || coalesce(summary, '')));
CREATE INDEX IF NOT EXISTS zaxy_pggraph_edges_source_idx
    ON zaxy_pggraph_edges (session_id, source_node_key, relation_type, valid_to);

SELECT graph.add_table(
    table_name := 'zaxy_pggraph_entities'::regclass,
    id_column := 'node_key',
    columns := ARRAY['name', 'summary', 'entity_type'],
    tenant_column := 'session_id'
);

SELECT graph.add_edge(
    from_table := 'zaxy_pggraph_edges'::regclass,
    from_column := 'source_node_key',
    to_table := 'zaxy_pggraph_entities'::regclass,
    to_column := 'node_key',
    label := 'relates',
    bidirectional := false,
    weight_column := NULL,
    label_column := 'relation_type'
);

SELECT * FROM graph.build();
"""


class PgGraphStore:
    """Async PostgreSQL/pgGraph projection backend.

    The implementation is intentionally experimental. Neo4j remains the default
    production and benchmark backend until this adapter passes the same gates.
    """

    def __init__(self, dsn: str, *, connection: Any | None = None) -> None:
        self._dsn = dsn
        self._connection = connection

    @property
    def dsn(self) -> str:
        """Return the configured PostgreSQL DSN."""
        return self._dsn

    async def connect(self) -> None:
        """Open PostgreSQL resources."""
        if self._connection is not None:
            return
        try:
            from psycopg import AsyncConnection
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("pgGraph backend requires installing zaxy-memory[pggraph]") from exc
        self._connection = await AsyncConnection.connect(self._dsn, row_factory=dict_row)

    async def close(self) -> None:
        """Close PostgreSQL resources."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def init_schema(self) -> None:
        """Initialize projection schema."""
        connection = self._require_connection()
        await connection.execute(PGGRAPH_SCHEMA_SQL)
        await connection.commit()

    async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None:
        """Project an extracted event."""
        raise NotImplementedError("pgGraph projection writes are not implemented yet")

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search by exact entity identity."""
        raise NotImplementedError("pgGraph exact search is not implemented yet")

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by lexical relevance."""
        raise NotImplementedError("pgGraph keyword search is not implemented yet")

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search graph neighbors from a starting entity."""
        raise NotImplementedError("pgGraph traversal search is not implemented yet")

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        raise RuntimeError("pgGraph vector search requires pgvector support and benchmark gates")

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Close the validity window for a projected entity."""
        raise NotImplementedError("pgGraph invalidation is not implemented yet")

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        """Inspect Eventloom-to-projection integrity."""
        raise NotImplementedError("pgGraph projection status is not implemented yet")

    async def inspect_inferred_edge_status(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> GraphInferredEdgeStatus:
        """Inspect inferred-edge audit status."""
        raise NotImplementedError("pgGraph inferred-edge status is not implemented yet")

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise AssertionError("Call connect() first")
        return self._connection
