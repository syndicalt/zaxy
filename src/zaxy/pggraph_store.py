"""Experimental pgGraph-backed projection store."""

from __future__ import annotations

import json
from typing import Any

from zaxy.extract import ExtractionResult
from zaxy.graph import (
    GraphEntity,
    GraphEventProjectionStatus,
    GraphInferredEdgeStatus,
    SearchResult,
)
from zaxy.security import validate_limit, validate_session_id

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
        safe_session_id = validate_session_id(session_id)
        connection = self._require_connection()
        await connection.execute(
            """
            INSERT INTO zaxy_pggraph_events (session_id, seq, hash, prev_hash, event_type, source_thread)
            VALUES (%(session_id)s, %(seq)s, %(hash)s, %(prev_hash)s, %(event_type)s, %(source_thread)s)
            ON CONFLICT (session_id, seq) DO UPDATE SET
                hash = EXCLUDED.hash,
                prev_hash = EXCLUDED.prev_hash,
                event_type = EXCLUDED.event_type,
                source_thread = EXCLUDED.source_thread,
                projected_at = now()
            """,
            {
                "session_id": safe_session_id,
                "seq": result.source_event_seq,
                "hash": result.source_event_hash,
                "prev_hash": result.source_event_prev_hash,
                "event_type": result.source_event_type,
                "source_thread": result.source_thread,
            },
        )
        for entity in result.entities:
            await connection.execute(
                """
                INSERT INTO zaxy_pggraph_entities (
                    node_key, session_id, name, entity_type, valid_from, valid_to,
                    summary, embedding, properties, source_event_seq, source_event_hash, source_event_type
                )
                VALUES (
                    %(node_key)s, %(session_id)s, %(name)s, %(entity_type)s, %(valid_from)s, NULL,
                    %(summary)s, %(embedding)s::jsonb, %(properties)s::jsonb,
                    %(source_event_seq)s, %(source_event_hash)s, %(source_event_type)s
                )
                ON CONFLICT (node_key) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    embedding = EXCLUDED.embedding,
                    properties = EXCLUDED.properties,
                    source_event_seq = EXCLUDED.source_event_seq,
                    source_event_hash = EXCLUDED.source_event_hash,
                    source_event_type = EXCLUDED.source_event_type,
                    updated_at = now()
                """,
                {
                    "node_key": _node_key(
                        safe_session_id,
                        entity.entity_type,
                        entity.name,
                        entity.observed_at,
                    ),
                    "session_id": safe_session_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "valid_from": entity.observed_at,
                    "summary": entity.summary,
                    "embedding": json.dumps(entity.embedding),
                    "properties": json.dumps(entity.properties or {}),
                    "source_event_seq": result.source_event_seq,
                    "source_event_hash": result.source_event_hash,
                    "source_event_type": result.source_event_type,
                },
            )
        for edge in result.edges:
            await connection.execute(
                """
                INSERT INTO zaxy_pggraph_edges (
                    edge_key, session_id, source_node_key, target_node_key, source_name, target_name,
                    relation_type, valid_from, valid_to, inferred, confidence, inference_method,
                    properties, source_event_seq, source_event_hash, source_event_type
                )
                SELECT
                    %(edge_key)s, %(session_id)s, source.node_key, target.node_key,
                    %(source_name)s, %(target_name)s, %(relation_type)s, %(valid_from)s, %(valid_to)s,
                    %(inferred)s, %(confidence)s, %(inference_method)s, %(properties)s::jsonb,
                    %(source_event_seq)s, %(source_event_hash)s, %(source_event_type)s
                FROM zaxy_pggraph_entities source, zaxy_pggraph_entities target
                WHERE source.session_id = %(session_id)s
                  AND target.session_id = %(session_id)s
                  AND source.name = %(source_name)s
                  AND target.name = %(target_name)s
                  AND source.valid_to IS NULL
                  AND target.valid_to IS NULL
                ORDER BY source.valid_from DESC, target.valid_from DESC
                LIMIT 1
                ON CONFLICT (edge_key) DO UPDATE SET
                    valid_to = EXCLUDED.valid_to,
                    inferred = EXCLUDED.inferred,
                    confidence = EXCLUDED.confidence,
                    inference_method = EXCLUDED.inference_method,
                    properties = EXCLUDED.properties,
                    source_event_seq = EXCLUDED.source_event_seq,
                    source_event_hash = EXCLUDED.source_event_hash,
                    source_event_type = EXCLUDED.source_event_type,
                    updated_at = now()
                """,
                {
                    "edge_key": _edge_key(
                        safe_session_id,
                        edge.source,
                        edge.target,
                        edge.relation_type,
                        edge.valid_from,
                    ),
                    "session_id": safe_session_id,
                    "source_name": edge.source,
                    "target_name": edge.target,
                    "relation_type": edge.relation_type,
                    "valid_from": edge.valid_from,
                    "valid_to": edge.valid_to,
                    "inferred": edge.inferred,
                    "confidence": edge.confidence,
                    "inference_method": edge.inference_method,
                    "properties": json.dumps(edge.evidence),
                    "source_event_seq": result.source_event_seq,
                    "source_event_hash": result.source_event_hash,
                    "source_event_type": result.source_event_type,
                },
            )
        await connection.commit()

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search by exact entity identity."""
        rows = await self._fetch_all(
            """
            SELECT name, entity_type, valid_from, valid_to, summary, properties, session_id
            FROM zaxy_pggraph_entities
            WHERE session_id = %(session_id)s
              AND name = %(name)s
              AND (%(entity_type)s IS NULL OR entity_type = %(entity_type)s)
              AND (
                (%(temporal_point)s IS NULL AND valid_to IS NULL)
                OR (
                    %(temporal_point)s IS NOT NULL
                    AND valid_from <= %(temporal_point)s
                    AND (valid_to IS NULL OR valid_to > %(temporal_point)s)
                )
              )
            ORDER BY valid_from DESC
            """,
            {
                "session_id": validate_session_id(session_id),
                "name": name,
                "entity_type": entity_type,
                "temporal_point": temporal_point,
            },
        )
        return [_row_to_entity(row) for row in rows]

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by lexical relevance."""
        rows = await self._fetch_all(
            """
            SELECT
                name,
                entity_type,
                valid_from,
                valid_to,
                summary,
                properties,
                session_id,
                CASE
                    WHEN lower(name) = lower(%(query)s) THEN 1.0
                    WHEN name ILIKE %(prefix_query)s THEN 0.9
                    ELSE 0.8
                END AS score
            FROM zaxy_pggraph_entities
            WHERE session_id = %(session_id)s
              AND (name ILIKE %(contains_query)s OR coalesce(summary, '') ILIKE %(contains_query)s)
              AND (
                (%(temporal_point)s IS NULL AND valid_to IS NULL)
                OR (
                    %(temporal_point)s IS NOT NULL
                    AND valid_from <= %(temporal_point)s
                    AND (valid_to IS NULL OR valid_to > %(temporal_point)s)
                )
              )
            ORDER BY score DESC, valid_from DESC
            LIMIT %(limit)s
            """,
            {
                "session_id": validate_session_id(session_id),
                "query": query,
                "prefix_query": f"{query}%",
                "contains_query": f"%{query}%",
                "temporal_point": temporal_point,
                "limit": validate_limit(limit),
            },
        )
        return [
            SearchResult(
                entity=_row_to_entity(row),
                score=float(row.get("score") or 0.0),
                raw_score=float(row.get("score") or 0.0),
                source="keyword",
            )
            for row in rows
        ]

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
        connection = self._require_connection()
        await connection.execute(
            """
            UPDATE zaxy_pggraph_entities
            SET valid_to = %(invalid_at)s,
                updated_at = now()
            WHERE session_id = %(session_id)s
              AND name = %(name)s
              AND entity_type = %(entity_type)s
              AND valid_to IS NULL
            """,
            {
                "session_id": validate_session_id(session_id),
                "name": name,
                "entity_type": entity_type,
                "invalid_at": invalid_at,
            },
        )
        await connection.commit()

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

    async def _fetch_all(
        self,
        sql: str,
        params: dict[str, object],
    ) -> list[dict[str, Any]]:
        connection = self._require_connection()
        async with connection.cursor() as cursor:
            await cursor.execute(sql, params)
            return list(await cursor.fetchall())


def _node_key(session_id: str, entity_type: str, name: str, valid_from: str) -> str:
    return f"{session_id}\x1f{entity_type}\x1f{name}\x1f{valid_from}"


def _edge_key(session_id: str, source: str, target: str, relation_type: str, valid_from: str) -> str:
    return f"{session_id}\x1f{source}\x1f{target}\x1f{relation_type}\x1f{valid_from}"


def _row_to_entity(row: dict[str, Any]) -> GraphEntity:
    properties = _properties_from_row(row)
    summary = row.get("summary")
    if summary is not None:
        properties = {"summary": summary, **properties}
    return GraphEntity(
        name=str(row.get("name") or ""),
        entity_type=str(row.get("entity_type") or ""),
        valid_from=_stringify_temporal(row.get("valid_from")),
        valid_to=_stringify_temporal(row.get("valid_to")) if row.get("valid_to") is not None else None,
        properties=properties,
        session_id=str(row.get("session_id") or "default"),
    )


def _properties_from_row(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("properties") or {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    return {}


def _stringify_temporal(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value or "")
