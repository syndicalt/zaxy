"""Experimental pgGraph-backed projection store."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from typing import Any, Literal

from zaxy.extract import ExtractionResult
from zaxy.graph import (
    GraphEntity,
    GraphEventProjectionStatus,
    GraphInferredEdgeMethodStatus,
    GraphInferredEdgeSample,
    GraphInferredEdgeStatus,
    SearchResult,
)
from zaxy.security import (
    validate_limit,
    validate_session_id,
    validate_traversal_depth,
    vector_has_signal,
)

PGGRAPH_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

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
    embedding_vector vector,
    search_vector tsvector,
    properties jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_event_seq bigint,
    source_event_hash text,
    source_event_type text,
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE zaxy_pggraph_entities ADD COLUMN IF NOT EXISTS embedding_vector vector;
ALTER TABLE zaxy_pggraph_entities ADD COLUMN IF NOT EXISTS search_vector tsvector;
UPDATE zaxy_pggraph_entities
SET search_vector = to_tsvector('simple', coalesce(name, '') || ' ' || coalesce(summary, ''))
WHERE search_vector IS NULL;

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
CREATE INDEX IF NOT EXISTS zaxy_pggraph_entities_search_vector_idx
    ON zaxy_pggraph_entities USING gin (search_vector);
CREATE INDEX IF NOT EXISTS zaxy_pggraph_edges_source_idx
    ON zaxy_pggraph_edges (session_id, source_node_key, relation_type, valid_to);
CREATE INDEX IF NOT EXISTS zaxy_pggraph_entities_vector_filter_idx
    ON zaxy_pggraph_entities (session_id, valid_to)
    WHERE embedding_vector IS NOT NULL;

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
    to_column := 'target_node_key',
    label := 'relates',
    bidirectional := false,
    weight_column := NULL,
    label_column := 'relation_type'
);

SELECT * FROM graph.build();
"""

_KEYWORD_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "can",
        "complement",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "planning",
        "remind",
        "suggest",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "would",
        "you",
        "another",
        "current",
        "currently",
        "last",
        "one",
        "throughout",
        "time",
        "wondering",
    }
)


class PgGraphStore:
    """Async PostgreSQL/pgGraph projection backend.

    The implementation is intentionally experimental. Embedded Kuzu remains the
    default backend, while pgGraph is available only for explicit sidecar
    experiments until this adapter passes the same gates.
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
        if result.source_event_type == "projection.retired":
            source_path = _retired_source_path(result)
            observed_at = _extraction_observed_at(result)
            if source_path and observed_at:
                await self._retire_source_projections(
                    connection,
                    {
                        "session_id": safe_session_id,
                        "source_path": source_path,
                        "invalid_at": observed_at,
                    },
                )
        for entity in result.entities:
            await connection.execute(
                """
                INSERT INTO zaxy_pggraph_entities (
                    node_key, session_id, name, entity_type, valid_from, valid_to,
                    summary, embedding, embedding_vector, search_vector, properties,
                    source_event_seq, source_event_hash, source_event_type
                )
                VALUES (
                    %(node_key)s, %(session_id)s, %(name)s, %(entity_type)s, %(valid_from)s, NULL,
                    %(summary)s,
                    %(embedding)s::jsonb,
                    %(embedding_vector)s::vector,
                    to_tsvector('simple', coalesce(%(name)s, '') || ' ' || coalesce(%(summary)s, '')),
                    %(properties)s::jsonb,
                    %(source_event_seq)s, %(source_event_hash)s, %(source_event_type)s
                )
                ON CONFLICT (node_key) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    embedding = EXCLUDED.embedding,
                    embedding_vector = EXCLUDED.embedding_vector,
                    search_vector = to_tsvector(
                        'simple',
                        coalesce(EXCLUDED.name, '') || ' ' || coalesce(EXCLUDED.summary, '')
                    ),
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
                    "embedding_vector": _pgvector_literal(entity.embedding),
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
            SELECT name, entity_type, valid_from, valid_to, summary, properties, session_id,
                   source_event_seq, source_event_hash
            FROM zaxy_pggraph_entities
            WHERE session_id = %(session_id)s
              AND name = %(name)s
              AND (%(entity_type)s::text IS NULL OR entity_type = %(entity_type)s::text)
              AND (
                (%(temporal_point)s::timestamptz IS NULL AND valid_to IS NULL)
                OR (
                    %(temporal_point)s::timestamptz IS NOT NULL
                    AND valid_from <= %(temporal_point)s::timestamptz
                    AND (valid_to IS NULL OR valid_to > %(temporal_point)s::timestamptz)
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
        if limit <= 0:
            return []
        rows = await self._fetch_all(
            """
            WITH search AS (
                SELECT to_tsquery('simple', %(tsquery)s) AS query
            )
            SELECT
                entity.name,
                entity.entity_type,
                entity.valid_from,
                entity.valid_to,
                entity.summary,
                entity.properties,
                entity.session_id,
                entity.source_event_seq,
                entity.source_event_hash,
                GREATEST(
                    CASE
                        WHEN lower(entity.name) = lower(%(query)s) THEN 1.0
                        WHEN entity.name ILIKE %(prefix_query)s THEN 0.9
                        ELSE 0.0
                    END,
                    ts_rank_cd(entity.search_vector, search.query)
                ) AS score
            FROM zaxy_pggraph_entities entity
            CROSS JOIN search
            WHERE entity.session_id = %(session_id)s
              AND (
                entity.search_vector @@ search.query
                OR entity.name ILIKE %(contains_query)s
              )
              AND (
                (%(temporal_point)s::timestamptz IS NULL AND entity.valid_to IS NULL)
                OR (
                    %(temporal_point)s::timestamptz IS NOT NULL
                    AND entity.valid_from <= %(temporal_point)s::timestamptz
                    AND (entity.valid_to IS NULL OR entity.valid_to > %(temporal_point)s::timestamptz)
                )
              )
            ORDER BY score DESC, entity.valid_from DESC
            LIMIT %(limit)s
            """,
            {
                "session_id": validate_session_id(session_id),
                "query": query,
                "tsquery": _keyword_tsquery(query),
                "prefix_query": f"{query}%",
                "contains_query": f"%{query}%",
                "temporal_point": temporal_point,
                "limit": validate_limit(limit),
            },
        )
        results = []
        for row in rows:
            score = float(row.get("score") or 0.0)
            results.append(SearchResult(entity=_row_to_entity(row), score=score, raw_score=score, source="keyword"))
        return results

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search graph neighbors from a starting entity."""
        rows = await self._fetch_all(
            """
            WITH seed AS (
                SELECT node_key
                FROM zaxy_pggraph_entities
                WHERE session_id = %(session_id)s
                  AND name = %(start_name)s
                  AND (
                    (%(temporal_point)s::timestamptz IS NULL AND valid_to IS NULL)
                    OR (
                        %(temporal_point)s::timestamptz IS NOT NULL
                        AND valid_from <= %(temporal_point)s::timestamptz
                        AND (valid_to IS NULL OR valid_to > %(temporal_point)s::timestamptz)
                    )
                  )
                ORDER BY valid_from DESC
                LIMIT 1
            )
            SELECT traversal.node
            FROM seed
            CROSS JOIN LATERAL graph.traverse(
                seed_table := 'zaxy_pggraph_entities'::regclass,
                seed_id := seed.node_key,
                max_depth := %(depth)s,
                edge_types := CASE
                    WHEN %(relation_type)s::text IS NULL THEN NULL
                    ELSE ARRAY[%(relation_type)s::text]
                END,
                direction := 'out',
                node_tables := ARRAY['zaxy_pggraph_entities'::regclass],
                filter := NULL,
                tenant := %(session_id)s,
                strategy := 'bfs',
                uniqueness := 'node_global',
                include_start := false,
                hydrate := true,
                max_rows := 100,
                row_offset := 0
            ) AS traversal
            """,
            {
                "session_id": validate_session_id(session_id),
                "start_name": start_name,
                "relation_type": relation_type,
                "depth": validate_traversal_depth(depth),
                "temporal_point": temporal_point,
            },
        )
        return [_row_to_entity(_node_row(row)) for row in rows]

    async def search_causal_neighbors(
        self,
        entity_name: str,
        *,
        direction: Literal["successors", "predecessors"],
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search directed causal neighbors from a starting entity."""
        if direction not in {"successors", "predecessors"}:
            raise ValueError("direction must be 'successors' or 'predecessors'")
        rows = await self._fetch_all(
            """
            SELECT edge.relation_type,
                   edge.confidence,
                   edge.inference_method,
                   edge.source_event_seq AS edge_source_event_seq,
                   edge.source_event_hash AS edge_source_event_hash,
                   edge.properties AS edge_properties,
                   source.node_key AS source_node_key,
                   source.name AS source_name,
                   source.entity_type AS source_entity_type,
                   source.valid_from AS source_valid_from,
                   source.valid_to AS source_valid_to,
                   source.summary AS source_summary,
                   source.properties AS source_properties,
                   source.session_id AS source_session_id,
                   source.source_event_seq AS source_source_event_seq,
                   source.source_event_hash AS source_source_event_hash,
                   target.node_key AS target_node_key,
                   target.name AS target_name,
                   target.entity_type AS target_entity_type,
                   target.valid_from AS target_valid_from,
                   target.valid_to AS target_valid_to,
                   target.summary AS target_summary,
                   target.properties AS target_properties,
                   target.session_id AS target_session_id,
                   target.source_event_seq AS target_source_event_seq,
                   target.source_event_hash AS target_source_event_hash
            FROM zaxy_pggraph_edges edge
            JOIN zaxy_pggraph_entities source ON source.node_key = edge.source_node_key
            JOIN zaxy_pggraph_entities target ON target.node_key = edge.target_node_key
            WHERE edge.session_id = %(session_id)s
              AND source.session_id = %(session_id)s
              AND target.session_id = %(session_id)s
              AND edge.relation_type LIKE 'causal_%%'
              AND (%(relation_type)s::text IS NULL OR edge.relation_type = %(relation_type)s)
              AND (
                (%(temporal_point)s::timestamptz IS NULL AND edge.valid_to IS NULL AND source.valid_to IS NULL AND target.valid_to IS NULL)
                OR (
                    %(temporal_point)s::timestamptz IS NOT NULL
                    AND edge.valid_from <= %(temporal_point)s::timestamptz
                    AND (edge.valid_to IS NULL OR edge.valid_to > %(temporal_point)s::timestamptz)
                    AND source.valid_from <= %(temporal_point)s::timestamptz
                    AND (source.valid_to IS NULL OR source.valid_to > %(temporal_point)s::timestamptz)
                    AND target.valid_from <= %(temporal_point)s::timestamptz
                    AND (target.valid_to IS NULL OR target.valid_to > %(temporal_point)s::timestamptz)
                )
              )
            """,
            {
                "session_id": validate_session_id(session_id),
                "relation_type": relation_type,
                "temporal_point": temporal_point,
            },
        )
        return _causal_neighbors_from_rows(
            rows,
            entity_name=entity_name,
            direction=direction,
            depth=validate_traversal_depth(depth),
            session_id=session_id,
        )

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        if limit <= 0:
            return []
        if not vector_has_signal(embedding):
            return []
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
                source_event_seq,
                source_event_hash,
                1.0 - (embedding_vector <=> %(embedding)s::vector) AS score
            FROM zaxy_pggraph_entities
            WHERE session_id = %(session_id)s
              AND embedding_vector IS NOT NULL
              AND (
                (%(temporal_point)s::timestamptz IS NULL AND valid_to IS NULL)
                OR (
                    %(temporal_point)s::timestamptz IS NOT NULL
                    AND valid_from <= %(temporal_point)s::timestamptz
                    AND (valid_to IS NULL OR valid_to > %(temporal_point)s::timestamptz)
                )
              )
            ORDER BY embedding_vector <=> %(embedding)s::vector, valid_from DESC
            LIMIT %(limit)s
            """,
            {
                "session_id": validate_session_id(session_id),
                "embedding": _pgvector_literal(embedding),
                "temporal_point": temporal_point,
                "limit": validate_limit(limit),
            },
        )
        results = []
        for row in rows:
            score = float(row.get("score") or 0.0)
            results.append(SearchResult(entity=_row_to_entity(row), score=score, raw_score=score, source="vector"))
        return results

    async def has_traversal_edges(self, session_id: str = "default") -> bool:
        """Return whether a session has active pgGraph edges for traversal."""
        rows = await self._fetch_all(
            """
            SELECT true AS has_edges
            FROM zaxy_pggraph_edges edge
            JOIN zaxy_pggraph_entities source ON source.node_key = edge.source_node_key
            JOIN zaxy_pggraph_entities target ON target.node_key = edge.target_node_key
            WHERE edge.session_id = %(session_id)s
              AND edge.valid_to IS NULL
              AND source.valid_to IS NULL
              AND target.valid_to IS NULL
            LIMIT 1
            """,
            {"session_id": validate_session_id(session_id)},
        )
        return bool(rows)

    async def reset_benchmark_projection(self) -> None:
        """Clear pgGraph projection tables for a reproducible benchmark rerun."""
        connection = self._require_connection()
        await connection.execute(
            """
            TRUNCATE TABLE
                zaxy_pggraph_edges,
                zaxy_pggraph_entities,
                zaxy_pggraph_events
            """
        )
        await connection.execute("SELECT * FROM graph.build()")
        await connection.commit()

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

    async def retire_source_projections(
        self,
        *,
        source_path: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Expire active entities and edges derived from one source path."""
        safe_session_id = validate_session_id(session_id)
        params = {
            "session_id": safe_session_id,
            "source_path": source_path,
            "invalid_at": invalid_at,
        }
        connection = self._require_connection()
        await self._retire_source_projections(connection, params)
        await connection.commit()

    async def _retire_source_projections(
        self,
        connection: Any,
        params: dict[str, str],
    ) -> None:
        await connection.execute(
            """
            UPDATE zaxy_pggraph_entities
            SET valid_to = %(invalid_at)s,
                updated_at = now()
            WHERE session_id = %(session_id)s
              AND valid_to IS NULL
              AND (
                properties ->> 'source_path' = %(source_path)s
                OR properties ->> 'target_path' = %(source_path)s
                OR properties ->> 'test_path' = %(source_path)s
                OR properties ->> 'covered_path' = %(source_path)s
              )
            """,
            params,
        )
        await connection.execute(
            """
            UPDATE zaxy_pggraph_edges edge
            SET valid_to = %(invalid_at)s,
                updated_at = now()
            WHERE edge.session_id = %(session_id)s
              AND edge.valid_to IS NULL
              AND (
                EXISTS (
                    SELECT 1
                    FROM zaxy_pggraph_entities entity
                    WHERE entity.node_key IN (edge.source_node_key, edge.target_node_key)
                      AND entity.session_id = %(session_id)s
                      AND (
                        entity.properties ->> 'source_path' = %(source_path)s
                        OR entity.properties ->> 'target_path' = %(source_path)s
                        OR entity.properties ->> 'test_path' = %(source_path)s
                        OR entity.properties ->> 'covered_path' = %(source_path)s
                      )
                )
                OR edge.properties ->> 'source_path' = %(source_path)s
                OR edge.properties ->> 'target_path' = %(source_path)s
                OR edge.properties ->> 'test_path' = %(source_path)s
                OR edge.properties ->> 'covered_path' = %(source_path)s
              )
            """,
            params,
        )

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        """Inspect Eventloom-to-projection integrity."""
        validated_session_id = validate_session_id(session_id)
        rows = await self._fetch_all(
            """
            WITH events AS (
                SELECT session_id, seq, hash, prev_hash
                FROM zaxy_pggraph_events
                WHERE session_id = %(session_id)s
            ),
            aggregate AS (
                SELECT
                    count(*)::int AS event_count,
                    max(seq)::bigint AS latest_seq
                FROM events
            ),
            latest AS (
                SELECT event.hash AS latest_hash
                FROM events event
                JOIN aggregate ON event.seq = aggregate.latest_seq
                LIMIT 1
            ),
            chain AS (
                SELECT
                    count(*) FILTER (
                        WHERE event.seq > 1 AND prev.hash IS NOT NULL
                    )::int AS linked_event_edges,
                    count(*) FILTER (
                        WHERE event.seq > 1 AND prev.hash IS NULL
                    )::int AS missing_chain_links
                FROM events event
                LEFT JOIN events prev ON prev.hash = event.prev_hash
            )
            SELECT
                aggregate.event_count,
                aggregate.latest_seq,
                latest.latest_hash,
                chain.linked_event_edges AS next_event_edges,
                chain.linked_event_edges AS previous_event_edges,
                chain.missing_chain_links
            FROM aggregate
            LEFT JOIN latest ON true
            CROSS JOIN chain
            """,
            {"session_id": validated_session_id},
        )
        row = rows[0] if rows else {}
        event_count = _int_value(row.get("event_count"))
        latest_seq = _optional_int_value(row.get("latest_seq"))
        latest_hash = _optional_str_value(row.get("latest_hash"))
        next_event_edges = _int_value(row.get("next_event_edges"))
        previous_event_edges = _int_value(row.get("previous_event_edges"))
        missing_chain_links = _int_value(row.get("missing_chain_links"))
        projection_lag = (
            max(0, eventloom_latest_seq - (latest_seq or 0))
            if eventloom_latest_seq is not None
            else None
        )
        latest_hash_matches = latest_hash == eventloom_latest_hash if eventloom_latest_hash else True
        expected_edges = max(0, event_count - 1)
        integrity_ok = (
            missing_chain_links == 0
            and latest_hash_matches
            and (projection_lag is None or projection_lag == 0)
            and next_event_edges == expected_edges
            and previous_event_edges == expected_edges
        )
        return GraphEventProjectionStatus(
            session_id=validated_session_id,
            event_count=event_count,
            latest_seq=latest_seq,
            latest_hash=latest_hash,
            eventloom_latest_seq=eventloom_latest_seq,
            eventloom_latest_hash=eventloom_latest_hash,
            projection_lag=projection_lag,
            latest_hash_matches=latest_hash_matches,
            next_event_edges=next_event_edges,
            previous_event_edges=previous_event_edges,
            missing_chain_links=missing_chain_links,
            integrity_ok=integrity_ok,
        )

    async def inspect_inferred_edge_status(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> GraphInferredEdgeStatus:
        """Inspect inferred-edge audit status."""
        validated_session_id = validate_session_id(session_id)
        validated_limit = validate_limit(limit)
        method_rows = await self._fetch_all(
            """
            WITH inferred AS (
                SELECT
                    relation_type,
                    confidence,
                    coalesce(inference_method, 'unknown') AS method,
                    source_event_seq,
                    source_event_hash,
                    coalesce(
                        (
                            SELECT array_agg(key ORDER BY key)
                            FROM jsonb_object_keys(properties) AS key
                            WHERE key LIKE 'evidence_%'
                        ),
                        ARRAY[]::text[]
                    ) AS evidence_keys
                FROM zaxy_pggraph_edges
                WHERE session_id = %(session_id)s
                  AND inferred = true
            )
            SELECT
                method,
                count(*)::int AS edge_count,
                array_agg(DISTINCT relation_type ORDER BY relation_type) AS relation_types,
                avg(confidence)::float AS average_confidence,
                min(confidence)::float AS minimum_confidence,
                count(*) FILTER (WHERE cardinality(evidence_keys) > 0)::int AS evidence_count,
                count(*) FILTER (WHERE cardinality(evidence_keys) = 0)::int AS missing_evidence_count,
                count(*) FILTER (
                    WHERE source_event_seq IS NULL OR source_event_hash IS NULL
                )::int AS missing_source_event_count
            FROM inferred
            GROUP BY method
            ORDER BY edge_count DESC, method ASC
            """,
            {"session_id": validated_session_id},
        )
        sample_rows = await self._fetch_all(
            """
            WITH inferred AS (
                SELECT
                    source_name AS source,
                    target_name AS target,
                    relation_type,
                    confidence,
                    coalesce(inference_method, 'unknown') AS method,
                    source_event_seq,
                    source_event_hash,
                    coalesce(
                        (
                            SELECT array_agg(key ORDER BY key)
                            FROM jsonb_object_keys(properties) AS key
                            WHERE key LIKE 'evidence_%'
                        ),
                        ARRAY[]::text[]
                    ) AS evidence_keys
                FROM zaxy_pggraph_edges
                WHERE session_id = %(session_id)s
                  AND inferred = true
            )
            SELECT
                source,
                target,
                relation_type,
                confidence,
                method,
                source_event_seq,
                source_event_hash,
                evidence_keys
            FROM inferred
            ORDER BY source_event_seq DESC NULLS LAST, source ASC, target ASC
            LIMIT %(limit)s
            """,
            {"session_id": validated_session_id, "limit": validated_limit},
        )
        methods = tuple(_row_to_inferred_edge_method(row) for row in method_rows)
        samples = tuple(_row_to_inferred_edge_sample(row) for row in sample_rows)
        total_edges = sum(method.edge_count for method in methods)
        evidence_count = sum(method.evidence_count for method in methods)
        missing_evidence_count = sum(method.missing_evidence_count for method in methods)
        missing_source_event_count = sum(method.missing_source_event_count for method in methods)
        evidence_coverage = evidence_count / total_edges if total_edges else 1.0
        return GraphInferredEdgeStatus(
            session_id=validated_session_id,
            total_edges=total_edges,
            method_count=len(methods),
            evidence_count=evidence_count,
            missing_evidence_count=missing_evidence_count,
            missing_source_event_count=missing_source_event_count,
            evidence_coverage=evidence_coverage,
            methods=methods,
            samples=samples,
        )

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
            try:
                await cursor.execute(sql, params)
                return list(await cursor.fetchall())
            except Exception:
                await connection.rollback()
                raise


def _node_key(session_id: str, entity_type: str, name: str, valid_from: str) -> str:
    return f"{session_id}\x1f{entity_type}\x1f{name}\x1f{valid_from}"


def _edge_key(session_id: str, source: str, target: str, relation_type: str, valid_from: str) -> str:
    return f"{session_id}\x1f{source}\x1f{target}\x1f{relation_type}\x1f{valid_from}"


def _row_to_entity(row: dict[str, Any]) -> GraphEntity:
    properties = _properties_from_row(row)
    summary = row.get("summary")
    if summary is not None:
        properties = {"summary": summary, **properties}
    if row.get("source_event_seq") is not None:
        properties["source_event_seq"] = int(row["source_event_seq"])
    if row.get("source_event_hash") is not None:
        properties["source_event_hash"] = str(row["source_event_hash"])
    return GraphEntity(
        name=str(row.get("name") or ""),
        entity_type=str(row.get("entity_type") or ""),
        valid_from=_stringify_temporal(row.get("valid_from")),
        valid_to=_stringify_temporal(row.get("valid_to")) if row.get("valid_to") is not None else None,
        properties=properties,
        session_id=str(row.get("session_id") or "default"),
    )


def _node_row(row: dict[str, Any]) -> dict[str, Any]:
    node = row.get("node")
    if isinstance(node, dict):
        return node
    return row


def _causal_neighbors_from_rows(
    rows: list[dict[str, Any]],
    *,
    entity_name: str,
    direction: Literal["successors", "predecessors"],
    depth: int,
    session_id: str,
) -> list[GraphEntity]:
    adjacency: dict[str, list[tuple[str, GraphEntity, dict[str, Any]]]] = {}
    keys_by_name: dict[str, set[str]] = {}
    for row in rows:
        source = _causal_row_entity(row, prefix="source")
        target = _causal_row_entity(row, prefix="target")
        source_key = str(row.get("source_node_key") or "")
        target_key = str(row.get("target_node_key") or "")
        edge_metadata = _causal_edge_metadata(row, source=source, target=target, session_id=session_id)
        keys_by_name.setdefault(source.name, set()).add(source_key)
        keys_by_name.setdefault(target.name, set()).add(target_key)
        if direction == "successors":
            adjacency.setdefault(source_key, []).append((target_key, target, edge_metadata))
        else:
            adjacency.setdefault(target_key, []).append((source_key, source, edge_metadata))

    frontier = set(keys_by_name.get(entity_name, set()))
    seen = set(frontier)
    found: dict[str, GraphEntity] = {}
    path_relations_by_key: dict[str, list[str]] = {key: [] for key in frontier}
    path_citations_by_key: dict[str, list[str]] = {key: [] for key in frontier}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for current_key in frontier:
            for neighbor_key, neighbor, edge_metadata in adjacency.get(current_key, []):
                path_relations = [*path_relations_by_key.get(current_key, []), str(edge_metadata["relation_type"])]
                path_citations = [*path_citations_by_key.get(current_key, []), str(edge_metadata["citation"])]
                if neighbor_key not in found:
                    found[neighbor_key] = _entity_with_causal_metadata(
                        neighbor,
                        edge_metadata=edge_metadata,
                        path_relation_types=path_relations,
                        path_citations=path_citations,
                    )
                if neighbor_key not in seen:
                    seen.add(neighbor_key)
                    path_relations_by_key[neighbor_key] = path_relations
                    path_citations_by_key[neighbor_key] = path_citations
                    next_frontier.add(neighbor_key)
        frontier = next_frontier
        if not frontier:
            break
    return list(found.values())


def _causal_row_entity(row: dict[str, Any], *, prefix: str) -> GraphEntity:
    return _row_to_entity(
        {
            "name": row.get(f"{prefix}_name"),
            "entity_type": row.get(f"{prefix}_entity_type"),
            "valid_from": row.get(f"{prefix}_valid_from"),
            "valid_to": row.get(f"{prefix}_valid_to"),
            "summary": row.get(f"{prefix}_summary"),
            "properties": row.get(f"{prefix}_properties") or {},
            "session_id": row.get(f"{prefix}_session_id"),
            "source_event_seq": row.get(f"{prefix}_source_event_seq"),
            "source_event_hash": row.get(f"{prefix}_source_event_hash"),
        }
    )


def _causal_edge_metadata(
    row: dict[str, Any],
    *,
    source: GraphEntity,
    target: GraphEntity,
    session_id: str,
) -> dict[str, Any]:
    evidence = _properties_from_row({"properties": row.get("edge_properties") or {}})
    graph_relation_type = str(row.get("relation_type") or "")
    source_event_seq = row.get("edge_source_event_seq")
    source_event_hash = _optional_str_value(row.get("edge_source_event_hash"))
    return {
        "causal_source_name": source.name,
        "causal_source_type": source.entity_type,
        "causal_target_name": target.name,
        "causal_target_type": target.entity_type,
        "relation_type": graph_relation_type,
        "graph_relation_type": graph_relation_type,
        "causal_relation_type": evidence.get("causal_relation_type") or graph_relation_type.removeprefix("causal_"),
        "confidence": _optional_float_value(row.get("confidence")) or 1.0,
        "inference_method": _optional_str_value(row.get("inference_method")) or "unknown",
        "citation": _edge_citation(session_id, source_event_seq, source_event_hash),
        "review_status": evidence.get("review_status") or "proposed",
        "authority_status": evidence.get("authority_status") or "non_authoritative",
        "source_event_seq": _optional_int_value(source_event_seq),
        "source_event_hash": source_event_hash,
        "evidence": evidence,
        "session_id": session_id,
    }


def _entity_with_causal_metadata(
    entity: GraphEntity,
    *,
    edge_metadata: dict[str, Any],
    path_relation_types: list[str],
    path_citations: list[str],
) -> GraphEntity:
    return GraphEntity(
        name=entity.name,
        entity_type=entity.entity_type,
        valid_from=entity.valid_from,
        valid_to=entity.valid_to,
        properties={
            **entity.properties,
            **edge_metadata,
            "_path_relation_types": path_relation_types,
            "_path_citations": path_citations,
            "_path_length": len(path_relation_types),
        },
        session_id=entity.session_id,
    )


def _edge_citation(session_id: str, source_event_seq: Any, source_event_hash: str | None) -> str:
    if source_event_seq is not None and source_event_hash:
        return f"eventloom://{session_id}/events/{source_event_seq}#{source_event_hash[:12]}"
    return "eventloom://unknown/events/unknown#unknown"


def _properties_from_row(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("properties") or {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    return {}


def _row_to_inferred_edge_method(row: dict[str, Any]) -> GraphInferredEdgeMethodStatus:
    relation_types = row.get("relation_types") or []
    if not isinstance(relation_types, list | tuple):
        relation_types = []
    return GraphInferredEdgeMethodStatus(
        method=str(row.get("method") or "unknown"),
        edge_count=_int_value(row.get("edge_count")),
        relation_types=tuple(str(relation_type) for relation_type in relation_types),
        average_confidence=_optional_float_value(row.get("average_confidence")),
        minimum_confidence=_optional_float_value(row.get("minimum_confidence")),
        evidence_count=_int_value(row.get("evidence_count")),
        missing_evidence_count=_int_value(row.get("missing_evidence_count")),
        missing_source_event_count=_int_value(row.get("missing_source_event_count")),
    )


def _row_to_inferred_edge_sample(row: dict[str, Any]) -> GraphInferredEdgeSample:
    evidence_keys = row.get("evidence_keys") or []
    if not isinstance(evidence_keys, list | tuple):
        evidence_keys = []
    return GraphInferredEdgeSample(
        source=str(row.get("source") or ""),
        target=str(row.get("target") or ""),
        relation_type=str(row.get("relation_type") or ""),
        confidence=_optional_float_value(row.get("confidence")),
        method=str(row.get("method") or "unknown"),
        source_event_seq=_optional_int_value(row.get("source_event_seq")),
        source_event_hash=_optional_str_value(row.get("source_event_hash")),
        evidence_keys=tuple(sorted(str(key) for key in evidence_keys)),
    )


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if value is None:
        return 0
    if isinstance(value, float | Decimal | str):
        return int(value)
    raise TypeError(f"expected int-compatible value, got {type(value).__name__}")


def _optional_int_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | Decimal | str):
        return int(value)
    raise TypeError(f"expected int-compatible value, got {type(value).__name__}")


def _optional_str_value(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | Decimal):
        return float(value)
    return float(str(value))


def _extraction_observed_at(result: ExtractionResult) -> str | None:
    values = [entity.observed_at for entity in result.entities]
    values.extend(edge.valid_from for edge in result.edges)
    return min(values) if values else None


def _retired_source_path(result: ExtractionResult) -> str | None:
    for entity in result.entities:
        source_path = entity.properties.get("source_path") if entity.properties else None
        if isinstance(source_path, str) and source_path:
            return source_path
    return None


def _pgvector_literal(embedding: list[float] | None) -> str | None:
    if embedding is None:
        return None
    if not embedding:
        raise ValueError("embedding must not be empty")
    values: list[str] = []
    for value in embedding:
        if isinstance(value, bool):
            raise ValueError("embedding values must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding values must be finite numbers")
        values.append(f"{number:g}")
    return "[" + ",".join(values) + "]"


def _keyword_tsquery(query: str) -> str:
    """Return a safe OR tsquery for natural-language keyword search."""
    tokens = re.findall(r"[A-Za-z0-9]+", query.casefold())
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 2 or token in _KEYWORD_STOP_WORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    selected = unique_tokens[:10]
    return " | ".join(selected) if selected else "__zaxy_no_terms__"


def _stringify_temporal(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value or "")
