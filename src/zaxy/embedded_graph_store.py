"""Embedded graph projection backend.

This module owns the zero-friction local graph runtime prototype. The first
implementation target is Kuzu, but the adapter is intentionally exposed through
the existing ProjectionStore contract so Neo4j remains the control backend.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zaxy.extract import ExtractionResult
    from zaxy.graph import (
        GraphEntity,
        GraphEventProjectionStatus,
        GraphInferredEdgeStatus,
        SearchResult,
    )


@dataclass(frozen=True)
class _KeywordIndex:
    entities: list[GraphEntity]
    terms: list[list[str]]
    document_frequency: Counter[str]
    average_length: float


@dataclass(frozen=True)
class _VectorIndex:
    entities: list[GraphEntity]
    sparse_vectors: list[dict[int, float]]
    norms: list[float]


@dataclass(frozen=True)
class _TraversalIndex:
    adjacency: dict[str, list[tuple[str, GraphEntity, str]]]
    keys_by_name: dict[str, set[str]]


class EmbeddedGraphStore:
    """Kuzu-backed embedded projection store shell.

    The store is selectable behind `PROJECTION_BACKEND=embedded` before it is the
    default. Methods fail clearly until the Kuzu implementation lands.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._database: Any | None = None
        self._connection: Any | None = None
        self._active_entity_cache: dict[tuple[str, str, str], tuple[str, str | None, str]] = {}
        self._keyword_index_cache: dict[str, _KeywordIndex] = {}
        self._vector_index_cache: dict[tuple[str, str | None], _VectorIndex] = {}
        self._traversal_index_cache: dict[str, _TraversalIndex] = {}
        self._bulk_projection_open = False
        self._dirty_bulk_sessions: set[str] = set()

    async def connect(self) -> None:
        """Open embedded graph resources."""
        if importlib.util.find_spec("kuzu") is None:
            raise RuntimeError('embedded graph backend requires `pip install "zaxy-memory[embedded]"`')
        import kuzu

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._database = kuzu.Database(str(self.path))
        self._connection = kuzu.Connection(self._database)
        self._active_entity_cache = {}
        self._keyword_index_cache = {}
        self._vector_index_cache = {}
        self._traversal_index_cache = {}
        self._dirty_bulk_sessions = set()

    async def close(self) -> None:
        """Close embedded graph resources."""
        self._connection = None
        self._database = None
        self._keyword_index_cache = {}
        self._vector_index_cache = {}
        self._traversal_index_cache = {}
        self._bulk_projection_open = False
        self._dirty_bulk_sessions = set()

    async def init_schema(self) -> None:
        """Initialize embedded graph schema."""
        conn = self._require_connection()
        conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                node_key STRING,
                session_id STRING,
                name STRING,
                entity_type STRING,
                summary STRING,
                properties_json STRING,
                valid_from STRING,
                valid_to STRING,
                source_event_seq INT64,
                source_event_hash STRING,
                PRIMARY KEY(node_key)
            )
            """
        )
        conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Event(
                event_key STRING,
                session_id STRING,
                seq INT64,
                hash STRING,
                prev_hash STRING,
                event_type STRING,
                source_thread STRING,
                PRIMARY KEY(event_key)
            )
            """
        )
        conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS RELATES(
                FROM Entity TO Entity,
                session_id STRING,
                relation_type STRING,
                valid_from STRING,
                valid_to STRING,
                inferred BOOL,
                confidence DOUBLE,
                inference_method STRING,
                source_event_seq INT64,
                source_event_hash STRING,
                evidence_json STRING
            )
            """
        )
        conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS NEXT_EVENT(
                FROM Event TO Event,
                session_id STRING
            )
            """
        )
        conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS PREVIOUS_EVENT(
                FROM Event TO Event,
                session_id STRING
            )
            """
        )

    async def reset_benchmark_projection(self) -> None:
        """Remove and recreate the embedded projection artifact."""
        await self.close()
        if self.path.exists():
            if self.path.is_dir():
                shutil.rmtree(self.path)
            else:
                self.path.unlink()
        self._active_entity_cache = {}
        self._keyword_index_cache = {}
        self._vector_index_cache = {}
        self._traversal_index_cache = {}
        self._dirty_bulk_sessions = set()
        await self.connect()
        await self.init_schema()

    async def begin_bulk_projection(self) -> None:
        """Begin an explicit Kuzu write transaction for bulk Eventloom replay."""
        if self._bulk_projection_open:
            return
        self._require_connection().execute("BEGIN TRANSACTION")
        self._bulk_projection_open = True

    async def commit_bulk_projection(self) -> None:
        """Commit an explicit Kuzu write transaction for bulk Eventloom replay."""
        if not self._bulk_projection_open:
            return
        self._require_connection().execute("COMMIT")
        self._bulk_projection_open = False
        for session_id in sorted(self._dirty_bulk_sessions):
            self._keyword_index(session_id)
            self._vector_index(session_id, None)
            self._traversal_index(session_id)
        self._dirty_bulk_sessions = set()

    async def rollback_bulk_projection(self) -> None:
        """Rollback an explicit Kuzu write transaction for bulk Eventloom replay."""
        if not self._bulk_projection_open:
            return
        self._require_connection().execute("ROLLBACK")
        self._bulk_projection_open = False
        self._active_entity_cache = {}
        self._keyword_index_cache = {}
        self._vector_index_cache = {}
        self._traversal_index_cache = {}
        self._dirty_bulk_sessions = set()

    async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None:
        """Project an extracted Eventloom event."""
        conn = self._require_connection()
        self._keyword_index_cache.pop(session_id, None)
        self._clear_vector_index_cache(session_id)
        self._traversal_index_cache.pop(session_id, None)
        if self._bulk_projection_open:
            self._dirty_bulk_sessions.add(session_id)
        conn.execute(
            """
            MERGE (ev:Event {event_key: $event_key})
            SET ev.session_id = $session_id,
                ev.seq = $seq,
                ev.hash = $hash,
                ev.prev_hash = $prev_hash,
                ev.event_type = $event_type,
                ev.source_thread = $source_thread
            """,
            {
                "event_key": _event_key(session_id, result.source_event_seq),
                "session_id": session_id,
                "seq": result.source_event_seq,
                "hash": result.source_event_hash,
                "prev_hash": result.source_event_prev_hash,
                "event_type": result.source_event_type,
                "source_thread": result.source_thread,
            },
        )
        if result.source_event_prev_hash:
            conn.execute(
                """
                MATCH (prev:Event), (current:Event {event_key: $event_key})
                WHERE prev.session_id = $session_id AND prev.hash = $prev_hash
                MERGE (prev)-[next:NEXT_EVENT]->(current)
                SET next.session_id = $session_id
                MERGE (current)-[previous:PREVIOUS_EVENT]->(prev)
                SET previous.session_id = $session_id
                """,
                {
                    "event_key": _event_key(session_id, result.source_event_seq),
                    "session_id": session_id,
                    "prev_hash": result.source_event_prev_hash,
                },
            )
        entity_types = {entity.name: entity.entity_type for entity in result.entities}
        for entity in result.entities:
            properties_json = _entity_properties_json(entity)
            active_entity = self._active_entity_state(
                session_id=session_id,
                entity_type=entity.entity_type,
                name=entity.name,
            )
            if active_entity is not None and active_entity[1] == entity.summary and active_entity[2] == properties_json:
                continue
            if active_entity is not None:
                conn.execute(
                    """
                    MATCH (e:Entity {node_key: $node_key})
                    SET e.valid_to = $valid_to
                    """,
                    {
                        "node_key": active_entity[0],
                        "valid_to": entity.observed_at,
                    },
                )
            conn.execute(
                """
                MERGE (e:Entity {node_key: $node_key})
                SET e.session_id = $session_id,
                    e.name = $name,
                    e.entity_type = $entity_type,
                    e.summary = $summary,
                    e.properties_json = $properties_json,
                    e.valid_from = $valid_from,
                    e.valid_to = NULL,
                    e.source_event_seq = $source_event_seq,
                    e.source_event_hash = $source_event_hash
                """,
                {
                    "node_key": _node_key(session_id, entity.entity_type, entity.name, result.source_event_seq),
                    "session_id": session_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "summary": entity.summary,
                    "properties_json": properties_json,
                    "valid_from": entity.observed_at,
                    "source_event_seq": result.source_event_seq,
                    "source_event_hash": result.source_event_hash,
                },
            )
            self._active_entity_cache[(session_id, entity.entity_type, entity.name)] = (
                _node_key(session_id, entity.entity_type, entity.name, result.source_event_seq),
                entity.summary,
                properties_json,
            )
            if active_entity is not None:
                self._copy_active_relationships_to_new_version(
                    session_id=session_id,
                    entity_type=entity.entity_type,
                    name=entity.name,
                    previous_valid_to=entity.observed_at,
                    new_node_key=_node_key(session_id, entity.entity_type, entity.name, result.source_event_seq),
                )
        for edge in result.edges:
            source_type = entity_types.get(edge.source, "entity")
            target_type = entity_types.get(edge.target, "entity")
            source_key = self._active_node_key(session_id, source_type, edge.source)
            target_key = self._active_node_key(session_id, target_type, edge.target)
            if source_key is None or target_key is None:
                continue
            conn.execute(
                """
                MATCH (source:Entity {node_key: $source_key}), (target:Entity {node_key: $target_key})
                MERGE (source)-[r:RELATES {relation_type: $relation_type}]->(target)
                SET r.session_id = $session_id,
                    r.valid_from = $valid_from,
                    r.valid_to = $valid_to,
                    r.inferred = $inferred,
                    r.confidence = $confidence,
                    r.inference_method = $inference_method,
                    r.source_event_seq = $source_event_seq,
                    r.source_event_hash = $source_event_hash,
                    r.evidence_json = $evidence_json
                """,
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "relation_type": edge.relation_type,
                    "session_id": session_id,
                    "valid_from": edge.valid_from,
                    "valid_to": edge.valid_to,
                    "inferred": edge.inferred,
                    "confidence": edge.confidence,
                    "inference_method": edge.inference_method,
                    "source_event_seq": edge.evidence.get("source_event_seq", result.source_event_seq),
                    "source_event_hash": edge.evidence.get("source_event_hash", result.source_event_hash),
                    "evidence_json": json.dumps(edge.evidence, sort_keys=True),
                },
            )

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search by exact identity."""
        rows = self._require_connection().execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id
              AND e.name = $name
              AND ($entity_type IS NULL OR e.entity_type = $entity_type)
              AND (
                ($temporal_point IS NULL AND e.valid_to IS NULL)
                OR (
                    $temporal_point IS NOT NULL
                    AND e.valid_from <= $temporal_point
                    AND (e.valid_to IS NULL OR $temporal_point < e.valid_to)
                )
              )
            RETURN e.name, e.entity_type, e.valid_from, e.valid_to, e.summary, e.properties_json, e.session_id,
                   e.source_event_seq, e.source_event_hash
            ORDER BY e.valid_from DESC
            """,
            {"session_id": session_id, "name": name, "entity_type": entity_type, "temporal_point": temporal_point},
        ).get_all()
        return [_row_to_entity(row) for row in rows]

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by lexical relevance."""
        from zaxy.graph import SearchResult

        query_terms = _keyword_query_terms(query)
        if not query_terms:
            return []
        index = self._keyword_index(session_id)
        document_count = len(index.terms)
        matches = []
        for entity, terms in zip(index.entities, index.terms, strict=True):
            score = _bm25_score(
                query_terms,
                terms,
                document_frequency=index.document_frequency,
                document_count=document_count,
                average_length=index.average_length,
            )
            if score:
                matches.append(SearchResult(entity=entity, score=float(score), source="keyword", raw_score=float(score)))
        return sorted(matches, key=lambda item: item.score, reverse=True)[:limit]

    def _keyword_index(self, session_id: str) -> _KeywordIndex:
        cached = self._keyword_index_cache.get(session_id)
        if cached is not None:
            return cached
        rows = self._require_connection().execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id AND e.valid_to IS NULL
            RETURN e.name, e.entity_type, e.valid_from, e.valid_to, e.summary, e.properties_json, e.session_id,
                   e.source_event_seq, e.source_event_hash
            """,
            {"session_id": session_id},
        ).get_all()
        entities = [_row_to_entity(row) for row in rows]
        entity_terms = [_terms(_entity_keyword_text(entity)) for entity in entities]
        document_frequency = Counter(term for terms in entity_terms for term in set(terms))
        average_length = (
            sum(len(terms) for terms in entity_terms) / len(entity_terms)
            if entity_terms
            else 0.0
        )
        index = _KeywordIndex(
            entities=entities,
            terms=entity_terms,
            document_frequency=document_frequency,
            average_length=average_length,
        )
        self._keyword_index_cache[session_id] = index
        return index

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search graph neighbors from a starting entity."""
        safe_depth = max(1, min(depth, 5))
        index = self._traversal_index(session_id)
        start_keys = set(index.keys_by_name.get(start_name, set()))
        if not start_keys:
            return []

        frontier = set(start_keys)
        path_relations_by_key: dict[str, list[str]] = {start_key: [] for start_key in start_keys}
        seen_sources = set(start_keys)
        found: dict[str, GraphEntity] = {}
        for _ in range(safe_depth):
            next_frontier: set[str] = set()
            for source_key in frontier:
                source_path_relations = path_relations_by_key.get(source_key, [])
                for target_key, target_entity, relation in index.adjacency.get(source_key, []):
                    if relation_type is not None and relation != relation_type:
                        continue
                    if target_key in start_keys:
                        continue
                    target_path_relations = [*source_path_relations, relation]
                    found.setdefault(
                        target_key,
                        _entity_with_path_metadata(target_entity, relation_types=target_path_relations),
                    )
                    if target_key not in seen_sources:
                        seen_sources.add(target_key)
                        path_relations_by_key[target_key] = target_path_relations
                        next_frontier.add(target_key)
            frontier = next_frontier
            if not frontier:
                break
        return list(found.values())

    def _traversal_index(self, session_id: str) -> _TraversalIndex:
        cached = self._traversal_index_cache.get(session_id)
        if cached is not None:
            return cached
        rows = self._require_connection().execute(
            """
            MATCH (source:Entity)-[r:RELATES]->(target:Entity)
            WHERE source.session_id = $session_id
              AND target.session_id = $session_id
              AND r.session_id = $session_id
              AND source.valid_to IS NULL
              AND target.valid_to IS NULL
              AND r.valid_to IS NULL
            RETURN source.node_key,
                   source.name,
                   source.entity_type,
                   source.valid_from,
                   source.valid_to,
                   source.summary,
                   source.properties_json,
                   source.session_id,
                   source.source_event_seq,
                   source.source_event_hash,
                   r.relation_type,
                   target.node_key,
                   target.name,
                   target.entity_type,
                   target.valid_from,
                   target.valid_to,
                   target.summary,
                   target.properties_json,
                   target.session_id,
                   target.source_event_seq,
                   target.source_event_hash
            """,
            {"session_id": session_id},
        ).get_all()
        adjacency: dict[str, list[tuple[str, GraphEntity, str]]] = {}
        keys_by_name: dict[str, set[str]] = {}
        for row in rows:
            relation = str(row[10])
            source_key = str(row[0])
            target_key = str(row[11])
            keys_by_name.setdefault(str(row[1]), set()).add(source_key)
            keys_by_name.setdefault(str(row[12]), set()).add(target_key)
            source_entity = _row_to_entity(list(row[1:10]))
            target_entity = _row_to_entity(list(row[12:]))
            adjacency.setdefault(source_key, []).append((target_key, target_entity, relation))
            adjacency.setdefault(target_key, []).append((source_key, source_entity, relation))
        index = _TraversalIndex(adjacency=adjacency, keys_by_name=keys_by_name)
        self._traversal_index_cache[session_id] = index
        return index

    async def has_traversal_edges(self, session_id: str = "default") -> bool:
        """Return whether the embedded projection has active graph edges."""
        rows = self._require_connection().execute(
            """
            MATCH (:Entity)-[r:RELATES]->(:Entity)
            WHERE r.session_id = $session_id AND r.valid_to IS NULL
            RETURN count(r)
            """,
            {"session_id": session_id},
        ).get_all()
        return bool(rows and rows[0][0])

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        from zaxy.graph import SearchResult

        index = self._vector_index(session_id, temporal_point)
        query_norm = _vector_norm(embedding)
        if query_norm == 0.0:
            return []
        query_sparse = _sparse_vector(embedding)
        matches = []
        for entity, sparse_vector, norm in zip(index.entities, index.sparse_vectors, index.norms, strict=True):
            dimension = entity.properties.get("embedding_dimension")
            if isinstance(dimension, int) and dimension != len(embedding):
                continue
            score = _sparse_cosine_similarity(query_sparse, sparse_vector, query_norm, norm)
            matches.append(SearchResult(entity=entity, score=score, source="vector", raw_score=score))
        return sorted(matches, key=lambda item: item.score, reverse=True)[:limit]

    def _vector_index(self, session_id: str, temporal_point: str | None) -> _VectorIndex:
        key = (session_id, temporal_point)
        cached = self._vector_index_cache.get(key)
        if cached is not None:
            return cached
        rows = self._require_connection().execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id
              AND (
                ($temporal_point IS NULL AND e.valid_to IS NULL)
                OR (
                    $temporal_point IS NOT NULL
                    AND e.valid_from <= $temporal_point
                    AND (e.valid_to IS NULL OR $temporal_point < e.valid_to)
                )
              )
            RETURN e.name, e.entity_type, e.valid_from, e.valid_to, e.summary, e.properties_json, e.session_id,
                   e.source_event_seq, e.source_event_hash
            """,
            {"session_id": session_id, "temporal_point": temporal_point},
        ).get_all()
        entities: list[GraphEntity] = []
        sparse_vectors: list[dict[int, float]] = []
        norms: list[float] = []
        for row in rows:
            entity = _row_to_entity(row)
            vector = _embedding_vector(entity.properties.get("embedding"))
            if vector is None:
                continue
            norm = _vector_norm(vector)
            if norm == 0.0:
                continue
            entity.properties["embedding_dimension"] = len(vector)
            entities.append(entity)
            sparse_vectors.append(_sparse_vector(vector))
            norms.append(norm)
        index = _VectorIndex(entities=entities, sparse_vectors=sparse_vectors, norms=norms)
        self._vector_index_cache[key] = index
        return index

    def _clear_vector_index_cache(self, session_id: str) -> None:
        self._vector_index_cache = {
            key: value
            for key, value in self._vector_index_cache.items()
            if key[0] != session_id
        }

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Close an entity validity window."""
        self._require_connection().execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id
              AND e.name = $name
              AND e.entity_type = $entity_type
              AND e.valid_to IS NULL
            SET e.valid_to = $invalid_at
            """,
            {
                "session_id": session_id,
                "name": name,
                "entity_type": entity_type,
                "invalid_at": invalid_at,
            },
        )
        self._active_entity_cache.pop((session_id, entity_type, name), None)
        self._keyword_index_cache.pop(session_id, None)
        self._clear_vector_index_cache(session_id)
        self._traversal_index_cache.pop(session_id, None)

    async def retire_source_projections(
        self,
        *,
        source_path: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Close projections derived from one source."""
        conn = self._require_connection()
        rows = conn.execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id AND e.valid_to IS NULL
            RETURN e.node_key, e.properties_json
            """,
            {"session_id": session_id},
        ).get_all()
        retired_node_keys = [
            str(row[0])
            for row in rows
            if _properties_reference_source(_json_dict(row[1]), source_path)
        ]
        for node_key in retired_node_keys:
            conn.execute(
                """
                MATCH (e:Entity {node_key: $node_key})
                SET e.valid_to = $invalid_at
                """,
                {"node_key": node_key, "invalid_at": invalid_at},
            )
            conn.execute(
                """
                MATCH (source:Entity)-[r:RELATES]->(target:Entity)
                WHERE r.session_id = $session_id
                  AND r.valid_to IS NULL
                  AND (source.node_key = $node_key OR target.node_key = $node_key)
                SET r.valid_to = $invalid_at
                """,
                {"session_id": session_id, "node_key": node_key, "invalid_at": invalid_at},
            )
        self._active_entity_cache = {
            key: value
            for key, value in self._active_entity_cache.items()
            if not (key[0] == session_id and value[0] in retired_node_keys)
        }
        if retired_node_keys:
            self._keyword_index_cache.pop(session_id, None)
            self._clear_vector_index_cache(session_id)
            self._traversal_index_cache.pop(session_id, None)

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        """Inspect Eventloom projection integrity."""
        from zaxy.graph import GraphEventProjectionStatus

        conn = self._require_connection()
        try:
            event_rows = conn.execute(
                """
                MATCH (e:Event)
                WHERE e.session_id = $session_id
                RETURN e.seq, e.hash, e.prev_hash
                ORDER BY e.seq
                """,
                {"session_id": session_id},
            ).get_all()
        except RuntimeError as exc:
            if not _is_missing_projection_table_error(exc):
                raise
            projection_lag = eventloom_latest_seq if eventloom_latest_seq is not None else None
            latest_hash_matches = eventloom_latest_hash is None
            return GraphEventProjectionStatus(
                session_id=session_id,
                event_count=0,
                latest_seq=None,
                latest_hash=None,
                eventloom_latest_seq=eventloom_latest_seq,
                eventloom_latest_hash=eventloom_latest_hash,
                projection_lag=projection_lag,
                latest_hash_matches=latest_hash_matches,
                next_event_edges=0,
                previous_event_edges=0,
                missing_chain_links=0,
                integrity_ok=projection_lag in (None, 0) and latest_hash_matches,
            )
        event_count = len(event_rows)
        latest_seq = event_rows[-1][0] if event_rows else None
        latest_hash = event_rows[-1][1] if event_rows else None
        known_hashes = {row[1] for row in event_rows if row[1] is not None}
        missing_chain_links = sum(1 for row in event_rows if row[2] is not None and row[2] not in known_hashes)
        next_event_edges = _first_count(
            conn.execute(
                """
                MATCH (:Event)-[r:NEXT_EVENT]->(:Event)
                WHERE r.session_id = $session_id
                RETURN count(r)
                """,
                {"session_id": session_id},
            ).get_all()
        )
        previous_event_edges = _first_count(
            conn.execute(
                """
                MATCH (:Event)-[r:PREVIOUS_EVENT]->(:Event)
                WHERE r.session_id = $session_id
                RETURN count(r)
                """,
                {"session_id": session_id},
            ).get_all()
        )
        projection_lag = None if eventloom_latest_seq is None or latest_seq is None else eventloom_latest_seq - latest_seq
        latest_hash_matches = eventloom_latest_hash is None or latest_hash == eventloom_latest_hash
        integrity_ok = (
            missing_chain_links == 0
            and (projection_lag in (None, 0))
            and latest_hash_matches
        )
        return GraphEventProjectionStatus(
            session_id=session_id,
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
        """Inspect inferred edge audit status."""
        from zaxy.graph import (
            GraphInferredEdgeMethodStatus,
            GraphInferredEdgeSample,
            GraphInferredEdgeStatus,
        )

        try:
            rows = self._require_connection().execute(
                """
                MATCH (source:Entity)-[r:RELATES]->(target:Entity)
                WHERE r.session_id = $session_id AND r.inferred = true
                RETURN source.name,
                       target.name,
                       r.relation_type,
                       r.confidence,
                       r.inference_method,
                       r.source_event_seq,
                       r.source_event_hash,
                       r.evidence_json
                """,
                {"session_id": session_id},
            ).get_all()
        except RuntimeError as exc:
            if not _is_missing_projection_table_error(exc):
                raise
            return GraphInferredEdgeStatus(
                session_id=session_id,
                total_edges=0,
                method_count=0,
                evidence_count=0,
                missing_evidence_count=0,
                missing_source_event_count=0,
                evidence_coverage=1.0,
                methods=(),
                samples=(),
            )
        method_rows: dict[str, list[list[Any]]] = {}
        evidence_count = 0
        missing_source_event_count = 0
        samples = []
        for row in rows[:limit]:
            evidence = _json_dict(row[7])
            if evidence:
                evidence_count += 1
            if row[5] is None:
                missing_source_event_count += 1
            method = str(row[4] or "unknown")
            method_rows.setdefault(method, []).append(row)
            samples.append(
                GraphInferredEdgeSample(
                    source=str(row[0]),
                    target=str(row[1]),
                    relation_type=str(row[2]),
                    confidence=float(row[3]) if row[3] is not None else None,
                    method=method,
                    source_event_seq=int(row[5]) if row[5] is not None else None,
                    source_event_hash=str(row[6]) if row[6] is not None else None,
                    evidence_keys=tuple(sorted(evidence)),
                )
            )
        missing_evidence_count = max(0, len(rows) - evidence_count)
        methods = []
        for method, grouped_rows in sorted(method_rows.items()):
            confidences = [float(row[3]) for row in grouped_rows if row[3] is not None]
            relation_types = tuple(sorted({str(row[2]) for row in grouped_rows if row[2] is not None}))
            method_evidence_count = sum(1 for row in grouped_rows if _json_dict(row[7]))
            methods.append(
                GraphInferredEdgeMethodStatus(
                    method=method,
                    edge_count=len(grouped_rows),
                    relation_types=relation_types,
                    average_confidence=sum(confidences) / len(confidences) if confidences else None,
                    minimum_confidence=min(confidences) if confidences else None,
                    evidence_count=method_evidence_count,
                    missing_evidence_count=max(0, len(grouped_rows) - method_evidence_count),
                    missing_source_event_count=sum(1 for row in grouped_rows if row[5] is None),
                )
            )
        return GraphInferredEdgeStatus(
            session_id=session_id,
            total_edges=len(rows),
            method_count=len(methods),
            evidence_count=evidence_count,
            missing_evidence_count=missing_evidence_count,
            missing_source_event_count=missing_source_event_count,
            evidence_coverage=round(evidence_count / len(rows), 4) if rows else 1.0,
            methods=tuple(methods),
            samples=tuple(samples),
        )

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise RuntimeError("embedded graph store is not connected")
        return self._connection

    def _active_node_key(self, session_id: str, entity_type: str, name: str) -> str | None:
        cached = self._active_entity_cache.get((session_id, entity_type, name))
        if cached is not None:
            return cached[0]
        rows = self._require_connection().execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id
              AND e.name = $name
              AND e.entity_type = $entity_type
              AND e.valid_to IS NULL
            RETURN e.node_key, e.summary, e.properties_json
            ORDER BY e.valid_from DESC
            LIMIT 1
            """,
            {"session_id": session_id, "entity_type": entity_type, "name": name},
        ).get_all()
        if not rows:
            return None
        node_key = str(rows[0][0])
        self._active_entity_cache[(session_id, entity_type, name)] = (
            node_key,
            str(rows[0][1]) if rows[0][1] is not None else None,
            str(rows[0][2] or "{}"),
        )
        return node_key

    def _active_entity_state(
        self,
        *,
        session_id: str,
        entity_type: str,
        name: str,
    ) -> tuple[str, str | None, str] | None:
        cached = self._active_entity_cache.get((session_id, entity_type, name))
        if cached is not None:
            return cached
        rows = self._require_connection().execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id
              AND e.name = $name
              AND e.entity_type = $entity_type
              AND e.valid_to IS NULL
            RETURN e.node_key, e.summary, e.properties_json
            ORDER BY e.valid_from DESC
            LIMIT 1
            """,
            {
                "session_id": session_id,
                "entity_type": entity_type,
                "name": name,
            },
        ).get_all()
        if not rows:
            return None
        active_entity = (
            str(rows[0][0]),
            str(rows[0][1]) if rows[0][1] is not None else None,
            str(rows[0][2] or "{}"),
        )
        self._active_entity_cache[(session_id, entity_type, name)] = active_entity
        return active_entity

    def _copy_active_relationships_to_new_version(
        self,
        *,
        session_id: str,
        entity_type: str,
        name: str,
        previous_valid_to: str,
        new_node_key: str,
    ) -> None:
        """Carry active relationships from the immediately previous entity version."""
        conn = self._require_connection()
        incoming_rows = conn.execute(
            """
            MATCH (source:Entity)-[r:RELATES]->(prev:Entity)
            WHERE prev.session_id = $session_id
              AND prev.name = $name
              AND prev.entity_type = $entity_type
              AND prev.valid_to = $previous_valid_to
              AND source.valid_to IS NULL
              AND r.session_id = $session_id
              AND r.valid_to IS NULL
            RETURN source.node_key,
                   r.relation_type,
                   r.valid_from,
                   r.inferred,
                   r.confidence,
                   r.inference_method,
                   r.source_event_seq,
                   r.source_event_hash,
                   r.evidence_json
            """,
            {
                "session_id": session_id,
                "name": name,
                "entity_type": entity_type,
                "previous_valid_to": previous_valid_to,
            },
        ).get_all()
        for row in incoming_rows:
            self._merge_relationship(
                source_key=str(row[0]),
                target_key=new_node_key,
                session_id=session_id,
                relation_type=str(row[1]),
                valid_from=str(row[2]),
                inferred=bool(row[3]),
                confidence=float(row[4]) if row[4] is not None else None,
                inference_method=str(row[5] or ""),
                source_event_seq=int(row[6]) if row[6] is not None else None,
                source_event_hash=str(row[7] or ""),
                evidence_json=str(row[8] or "{}"),
            )

        outgoing_rows = conn.execute(
            """
            MATCH (prev:Entity)-[r:RELATES]->(target:Entity)
            WHERE prev.session_id = $session_id
              AND prev.name = $name
              AND prev.entity_type = $entity_type
              AND prev.valid_to = $previous_valid_to
              AND target.valid_to IS NULL
              AND r.session_id = $session_id
              AND r.valid_to IS NULL
            RETURN target.node_key,
                   r.relation_type,
                   r.valid_from,
                   r.inferred,
                   r.confidence,
                   r.inference_method,
                   r.source_event_seq,
                   r.source_event_hash,
                   r.evidence_json
            """,
            {
                "session_id": session_id,
                "name": name,
                "entity_type": entity_type,
                "previous_valid_to": previous_valid_to,
            },
        ).get_all()
        for row in outgoing_rows:
            self._merge_relationship(
                source_key=new_node_key,
                target_key=str(row[0]),
                session_id=session_id,
                relation_type=str(row[1]),
                valid_from=str(row[2]),
                inferred=bool(row[3]),
                confidence=float(row[4]) if row[4] is not None else None,
                inference_method=str(row[5] or ""),
                source_event_seq=int(row[6]) if row[6] is not None else None,
                source_event_hash=str(row[7] or ""),
                evidence_json=str(row[8] or "{}"),
            )

    def _merge_relationship(
        self,
        *,
        source_key: str,
        target_key: str,
        session_id: str,
        relation_type: str,
        valid_from: str,
        inferred: bool,
        confidence: float | None,
        inference_method: str,
        source_event_seq: int | None,
        source_event_hash: str,
        evidence_json: str,
    ) -> None:
        self._require_connection().execute(
            """
            MATCH (source:Entity {node_key: $source_key}), (target:Entity {node_key: $target_key})
            MERGE (source)-[r:RELATES {relation_type: $relation_type}]->(target)
            SET r.session_id = $session_id,
                r.valid_from = $valid_from,
                r.valid_to = NULL,
                r.inferred = $inferred,
                r.confidence = $confidence,
                r.inference_method = $inference_method,
                r.source_event_seq = $source_event_seq,
                r.source_event_hash = $source_event_hash,
                r.evidence_json = $evidence_json
            """,
            {
                "source_key": source_key,
                "target_key": target_key,
                "relation_type": relation_type,
                "session_id": session_id,
                "valid_from": valid_from,
                "inferred": inferred,
                "confidence": confidence,
                "inference_method": inference_method,
                "source_event_seq": source_event_seq,
                "source_event_hash": source_event_hash,
                "evidence_json": evidence_json,
            },
        )


def _node_key(session_id: str, entity_type: str, name: str, source_event_seq: int) -> str:
    return f"{session_id}\x1f{entity_type}\x1f{name}\x1f{source_event_seq}"


def _event_key(session_id: str, seq: int) -> str:
    return f"{session_id}\x1f{seq}"


def _entity_properties_json(entity: Any) -> str:
    properties = dict(entity.properties or {})
    if entity.embedding is not None:
        properties["embedding"] = entity.embedding
    return json.dumps(properties, sort_keys=True)


def _first_count(rows: list[list[Any]]) -> int:
    return int(rows[0][0]) if rows else 0


def _is_missing_projection_table_error(exc: RuntimeError) -> bool:
    message = str(exc)
    missing_tables = ("Table Entity does not exist", "Table Event does not exist", "Table NEXT_EVENT does not exist")
    return any(table in message for table in missing_tables)


def _row_to_entity(row: list[Any]) -> GraphEntity:
    from zaxy.graph import GraphEntity

    properties = json.loads(row[5] or "{}")
    if row[4] is not None:
        properties.setdefault("summary", row[4])
    if len(row) > 7 and row[7] is not None:
        properties["source_event_seq"] = int(row[7])
    if len(row) > 8 and row[8] is not None:
        properties["source_event_hash"] = str(row[8])
    return GraphEntity(
        name=row[0],
        entity_type=row[1],
        valid_from=row[2],
        valid_to=row[3],
        properties=properties,
        session_id=row[6],
    )


def _entity_with_path_metadata(entity: GraphEntity, *, relation_types: list[str]) -> GraphEntity:
    from zaxy.graph import GraphEntity

    return GraphEntity(
        name=entity.name,
        entity_type=entity.entity_type,
        valid_from=entity.valid_from,
        valid_to=entity.valid_to,
        properties={
            **entity.properties,
            "_path_relation_types": relation_types,
            "_path_length": len(relation_types),
        },
        session_id=entity.session_id,
    )


def _entity_keyword_text(entity: GraphEntity) -> str:
    values = [entity.name, entity.entity_type]
    for key, value in entity.properties.items():
        if key == "embedding" or key.startswith("_"):
            continue
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, int | float | bool):
            values.append(str(value))
    return " ".join(values)


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9]+", text.casefold()) if len(term) > 1]


def _keyword_query_terms(text: str) -> list[str]:
    stopwords = {
        "am",
        "are",
        "did",
        "does",
        "first",
        "had",
        "have",
        "how",
        "the",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
    return [term for term in _terms(text) if term not in stopwords]


def _bm25_score(
    query_terms: list[str],
    document_terms: list[str],
    *,
    document_frequency: Counter[str],
    document_count: int,
    average_length: float,
) -> float:
    if not document_terms or document_count <= 0 or average_length <= 0:
        return 0.0
    counts = Counter(document_terms)
    score = 0.0
    for term in query_terms:
        score += _bm25_term_score(
            term_frequency=counts[term],
            document_frequency=document_frequency[term],
            document_count=document_count,
            document_length=len(document_terms),
            average_length=average_length,
        )
    return score


def _bm25_term_score(
    *,
    term_frequency: int,
    document_frequency: int,
    document_count: int,
    document_length: int,
    average_length: float,
) -> float:
    if term_frequency <= 0 or document_frequency <= 0:
        return 0.0
    k1 = 1.5
    b = 0.75
    idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
    denominator = term_frequency + k1 * (1 - b + b * document_length / average_length)
    return idf * (term_frequency * (k1 + 1)) / denominator


def _embedding_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    vector = []
    for item in value:
        if not isinstance(item, int | float):
            return None
        vector.append(float(item))
    return vector


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _sparse_vector(vector: list[float]) -> dict[int, float]:
    return {index: value for index, value in enumerate(vector) if value != 0.0}


def _sparse_cosine_similarity(
    left: dict[int, float],
    right: dict[int, float],
    left_norm: float,
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    dot = sum(value * larger.get(index, 0.0) for index, value in smaller.items())
    return dot / (left_norm * right_norm)


def _json_dict(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(str(raw))
    return parsed if isinstance(parsed, dict) else {}


def _properties_reference_source(properties: dict[str, Any], source_path: str) -> bool:
    return any(properties.get(key) == source_path for key in ("source_path", "target_path", "test_path", "covered_path"))
