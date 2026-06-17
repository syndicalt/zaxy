"""Embedded graph projection backend.

This module owns the zero-friction local graph runtime. The LadybugDB adapter
(LadybugDB is the maintained fork of the archived Kuzu engine) is exposed
through the ProjectionStore contract so local default deployments and
same-harness backend comparisons share the same retrieval surface.
"""

from __future__ import annotations

import heapq
import importlib.util
import json
import shutil
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt

from zaxy.graph_walk import AdjacencySnapshot
from zaxy.log import get_logger

if TYPE_CHECKING:
    from zaxy.extract import ExtractionResult
    from zaxy.graph import (
        GraphEntity,
        GraphEventProjectionStatus,
        GraphInferredEdgeStatus,
        SearchResult,
    )

from zaxy.embedded_graph_internals import (
    _ANN_DELTA_REBUILD_FRACTION,
    _ANN_INSERT_BATCH_SIZE,
    _ANN_SHADOW_TABLE_PREFIX,
    _QUERY_PARAMETER_RE,
    LEGACY_EMBEDDING_VERSION,
    VECTOR_INDEX_CACHE_MAX_ENTRIES,
    VECTOR_SEARCH_OVERSAMPLE,
    _adjacency_signature,
    _ann_content_digest,
    _ann_scope_digest,
    _AnnGenerationState,
    _AnnVectorGroup,
    _AnyVectorGroup,
    _armor_json_shaped_string_parameters,
    _bm25_score_from_precomputed,
    _causal_edge_metadata_from_row,
    _dense_vector_results,
    _embedding_vector,
    _embedding_version,
    _entity_properties_json,
    _entity_with_causal_metadata,
    _entity_with_path_metadata,
    _event_key,
    _exact_rerank_results,
    _first_count,
    _is_corrupt_store_error,
    _is_incompatible_storage_error,
    _is_missing_projection_table_error,
    _json_dict,
    _keyword_candidate_terms,
    _keyword_index_from_entities,
    _keyword_query_terms,
    _KeywordIndex,
    _move_incompatible_store_aside,
    _node_key,
    _properties_reference_source,
    _quantize_unit_matrix,
    _quantized_candidate_entity_indexes,
    _QuantizedVectorGroup,
    _row_to_entity,
    _terms,
    _TraversalIndex,
    _VectorGroup,
    _VectorIndex,
    pre_ladybug_backup_paths,
)

logger = get_logger("embedded_graph_store")


_CacheValue = TypeVar("_CacheValue")


VECTOR_INDEX_CACHE_MAX_BYTES = 256 * 1024 * 1024


class EmbeddedGraphStore:
    """LadybugDB-backed embedded projection store."""

    def __init__(
        self,
        path: Path,
        *,
        vector_ann_threshold: int | None = None,
        vector_ann_max_dimension: int | None = None,
        vector_ann_efs: int | None = None,
        vector_ann_byte_budget_engagement: bool | None = None,
        vector_quantization: str | None = None,
        active_embedding_version: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._database: Any | None = None
        self._connection: Any | None = None
        self._vector_ann_threshold_override = vector_ann_threshold
        self._vector_ann_max_dimension_override = vector_ann_max_dimension
        self._vector_ann_efs_override = vector_ann_efs
        self._vector_ann_byte_budget_engagement_override = vector_ann_byte_budget_engagement
        self._vector_quantization_override = vector_quantization
        self._active_embedding_version_override = active_embedding_version
        self._ann_supported: bool | None = None
        self._ann_indexed_tables: set[str] = set()
        self._ann_generation_states: dict[tuple[str, str, int], _AnnGenerationState] = {}
        self._active_entity_cache: dict[tuple[str, str, str], tuple[str, str | None, str]] = {}
        self._current_entity_index_cache: dict[str, list[GraphEntity]] = {}
        self._current_entity_lookup_cache: dict[str, dict[tuple[str, str | None], list[GraphEntity]]] = {}
        self._temporal_entity_index_cache: dict[tuple[str, str], list[GraphEntity]] = {}
        self._temporal_entity_lookup_cache: dict[tuple[str, str], dict[tuple[str, str | None], list[GraphEntity]]] = {}
        self._keyword_index_cache: dict[str, _KeywordIndex] = {}
        self._temporal_keyword_index_cache: dict[tuple[str, str], _KeywordIndex] = {}
        self._vector_index_cache: dict[tuple[str, str | None], _VectorIndex] = {}
        self._traversal_index_cache: dict[str, _TraversalIndex] = {}
        self._temporal_traversal_index_cache: dict[tuple[str, str], _TraversalIndex] = {}
        self._adjacency_snapshot_cache: dict[str, AdjacencySnapshot] = {}
        self._bulk_projection_open = False
        self._dirty_bulk_sessions: set[str] = set()
        self._bulk_active_state_loaded_sessions: set[str] = set()

    async def connect(self) -> None:
        """Open embedded graph resources.

        A projection file written by the pre-fork Kuzu engine is refused by
        LadybugDB (the storage formats are incompatible). Projections are
        derived state, so the incompatible artifact is moved aside to
        ``<path>.pre-ladybug.bak`` — never deleted — and a fresh store opens
        in its place; the projection content is rebuilt from the Eventloom
        log through the existing replay path (``zaxy reproject`` for full
        history; new checkins project incrementally as always).
        """
        if importlib.util.find_spec("ladybug") is None:
            raise RuntimeError('embedded graph backend requires `pip install "zaxy-memory"`')
        import ladybug

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._database = ladybug.Database(str(self.path))
        except RuntimeError as exc:
            incompatible = _is_incompatible_storage_error(exc)
            corrupt = _is_corrupt_store_error(exc)
            if not (incompatible or corrupt):
                raise
            backup_path = _move_incompatible_store_aside(self.path)
            reason = (
                "was written by the pre-fork Kuzu engine and is not readable by LadybugDB"
                if incompatible
                else "is structurally corrupt (e.g. a dirty WAL left by an uncleanly-killed "
                "process); the projection is derived state"
            )
            logger.warning(
                "embedded projection at %s %s; it was moved aside to %s (no data deleted) and "
                "a fresh projection store was created. Rebuild full history with "
                "`zaxy reproject <eventloom log>`; the backup is safe to delete once the "
                "rebuilt projection is verified (zaxy doctor reports it until then).",
                self.path,
                reason,
                backup_path,
            )
            self._database = ladybug.Database(str(self.path))
        self._connection = ladybug.Connection(self._database)
        # The `vector` extension is an official LadybugDB extension that is NOT
        # bundled in the pip wheel (unlike Kuzu 0.11.3, which auto-registered a
        # statically-linked vector index). It must be INSTALLed once — a small
        # network download cached under the LadybugDB home (~/.lbdb) — before
        # LOAD works. Fast path: LOAD an already-cached extension. On failure,
        # attempt a one-time INSTALL then LOAD. If both fail (e.g. an offline
        # first run), approximate (HNSW) search degrades to exact search; the
        # default exact float path is pure numpy and stays fully offline, so
        # only opt-in ANN engagement depends on the extension being present.
        try:
            self._execute("LOAD vector")
        except RuntimeError:
            try:
                self._execute("INSTALL vector")
                self._execute("LOAD vector")
            except RuntimeError:
                logger.warning(
                    "LadybugDB `vector` extension unavailable (INSTALL/LOAD failed); "
                    "approximate HNSW vector search is disabled and queries use exact "
                    "float search. The extension downloads on first use and is cached "
                    "under ~/.lbdb; for offline ANN, pre-install it on a networked host "
                    "and ship the cache."
                )
                self._ann_supported = False
            else:
                self._ann_supported = None
        else:
            self._ann_supported = None
        self._clear_all_caches()

    async def close(self) -> None:
        """Close embedded graph resources."""
        self._connection = None
        self._database = None
        self._clear_all_caches()
        self._bulk_projection_open = False

    async def init_schema(self) -> None:
        """Initialize embedded graph schema."""
        self._execute(
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
        self._execute(
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
        self._execute(
            """
            CREATE NODE TABLE IF NOT EXISTS BenchmarkProjection(
                key STRING,
                event_count INT64,
                latest_seq INT64,
                latest_hash STRING,
                PRIMARY KEY(key)
            )
            """
        )
        self._execute(
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
        self._execute(
            """
            CREATE REL TABLE IF NOT EXISTS NEXT_EVENT(
                FROM Event TO Event,
                session_id STRING
            )
            """
        )
        self._execute(
            """
            CREATE REL TABLE IF NOT EXISTS PREVIOUS_EVENT(
                FROM Event TO Event,
                session_id STRING
            )
            """
        )

    async def warm_session(self, session_id: str = "default") -> None:
        """Preload read indexes for a session to avoid first-checkout stalls."""
        self._current_entities(session_id)
        self._current_entity_lookup(session_id)
        self._keyword_index(session_id)
        self._vector_index(session_id, None)
        self._traversal_index(session_id)

    async def reset_benchmark_projection(self) -> None:
        """Remove and recreate the embedded projection artifact."""
        await self.close()
        if self.path.exists():
            if self.path.is_dir():
                shutil.rmtree(self.path)
            else:
                self.path.unlink()
        self._clear_all_caches()
        await self.connect()
        await self.init_schema()

    async def benchmark_projection_present(self, key: str) -> bool:
        """Return whether this embedded projection is marked for a benchmark workload."""
        if not key:
            return False
        try:
            rows = self._execute(
                """
                MATCH (p:BenchmarkProjection {key: $key})
                RETURN p.key
                LIMIT 1
                """,
                {"key": key},
            ).get_all()
        except RuntimeError as exc:
            if not _is_missing_projection_table_error(exc):
                raise
            return False
        return bool(rows)

    async def mark_benchmark_projection(self, key: str, events: Sequence[object]) -> None:
        """Persist a semantic benchmark projection marker for reuse."""
        if not key:
            return
        latest = events[-1] if events else None
        self._execute(
            """
            MERGE (p:BenchmarkProjection {key: $key})
            SET p.event_count = $event_count,
                p.latest_seq = $latest_seq,
                p.latest_hash = $latest_hash
            """,
            {
                "key": key,
                "event_count": len(events),
                "latest_seq": getattr(latest, "seq", None),
                "latest_hash": getattr(latest, "hash", None),
            },
        )

    async def begin_bulk_projection(self) -> None:
        """Begin an explicit embedded-engine write transaction for bulk Eventloom replay."""
        if self._bulk_projection_open:
            return
        self._execute("BEGIN TRANSACTION")
        self._bulk_projection_open = True

    async def commit_bulk_projection(self) -> None:
        """Commit an explicit embedded-engine write transaction for bulk Eventloom replay."""
        if not self._bulk_projection_open:
            return
        self._execute("COMMIT")
        self._bulk_projection_open = False
        for session_id in sorted(self._dirty_bulk_sessions):
            self._current_entity_lookup(session_id)
            self._keyword_index(session_id)
            self._vector_index(session_id, None)
            self._traversal_index(session_id)
        self._dirty_bulk_sessions = set()
        self._bulk_active_state_loaded_sessions = set()

    async def rollback_bulk_projection(self) -> None:
        """Rollback an explicit embedded-engine write transaction for bulk Eventloom replay."""
        if not self._bulk_projection_open:
            return
        self._execute("ROLLBACK")
        self._bulk_projection_open = False
        self._clear_all_caches()

    async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None:
        """Project an extracted Eventloom event."""
        projected_indexed_content = False
        self._execute(
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
            self._execute(
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
            projected_indexed_content = True
            if active_entity is not None:
                self._execute(
                    """
                    MATCH (e:Entity {node_key: $node_key})
                    SET e.valid_to = $valid_to
                    """,
                    {
                        "node_key": active_entity[0],
                        "valid_to": entity.observed_at,
                    },
                )
            self._execute(
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
            projected_indexed_content = True
            self._execute(
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
        if projected_indexed_content:
            self._clear_read_caches(session_id)
            if self._bulk_projection_open:
                self._dirty_bulk_sessions.add(session_id)

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search by exact identity."""
        if temporal_point is None:
            lookup = self._current_entity_lookup(session_id)
            return list(lookup.get((name, entity_type), ()))
        lookup = self._temporal_entity_lookup(session_id, temporal_point)
        return list(lookup.get((name, entity_type), ()))

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by lexical relevance."""
        from zaxy.graph import SearchResult

        if limit <= 0:
            return []
        query_terms = _keyword_query_terms(query)
        if not query_terms:
            return []
        index = self._keyword_index(session_id, temporal_point)
        matches = []
        matched_terms_by_candidate: dict[int, list[str]] = {}
        for term in _keyword_candidate_terms(index, query_terms):
            for entity_index in index.term_entity_ids.get(term, ()):
                matched_terms_by_candidate.setdefault(entity_index, []).append(term)
        if not matched_terms_by_candidate:
            return []
        for entity_index, matched_terms in matched_terms_by_candidate.items():
            entity = index.entities[entity_index]
            term_counts = index.term_counts[entity_index]
            score = _bm25_score_from_precomputed(
                matched_terms,
                term_counts,
                document_length_norm=index.document_length_norms[entity_index],
                term_idf=index.term_idf,
            )
            if score:
                score_value = float(score)
                matches.append(SearchResult(entity=entity, score=score_value, source="keyword", raw_score=score_value))
        return heapq.nlargest(limit, matches, key=lambda item: item.score)

    def _keyword_index(self, session_id: str, temporal_point: str | None = None) -> _KeywordIndex:
        if temporal_point is not None:
            key = (session_id, temporal_point)
            cached = self._temporal_keyword_index_cache.get(key)
            if cached is not None:
                return cached
            index = _keyword_index_from_entities(self._temporal_entities(session_id, temporal_point))
            self._temporal_keyword_index_cache[key] = index
            return index
        cached = self._keyword_index_cache.get(session_id)
        if cached is not None:
            return cached
        index = _keyword_index_from_entities(self._current_entities(session_id))
        self._keyword_index_cache[session_id] = index
        return index

    async def active_entity_names(self, session_id: str = "default") -> list[str]:
        """Return the names of the session's active (non-superseded) entities.

        Served from the per-session current-entity cache, so a warm call is a
        pure in-memory read; the cache is invalidated alongside the other
        derived read indexes whenever the session's projection changes. This
        is the minimal public surface the feeling-of-knowing pre-check needs
        to build its per-session index without a graph query per call.
        """
        return [entity.name for entity in self._current_entities(session_id)]

    def _current_entities(self, session_id: str) -> list[GraphEntity]:
        cached = self._current_entity_index_cache.get(session_id)
        if cached is not None:
            return cached
        rows = self._execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id AND e.valid_to IS NULL
            RETURN e.name, e.entity_type, e.valid_from, e.valid_to, e.summary, e.properties_json, e.session_id,
                   e.source_event_seq, e.source_event_hash
            """,
            {"session_id": session_id},
        ).get_all()
        entities = [_row_to_entity(row) for row in rows]
        self._current_entity_index_cache[session_id] = entities
        return entities

    def _current_entity_lookup(self, session_id: str) -> dict[tuple[str, str | None], list[GraphEntity]]:
        cached = self._current_entity_lookup_cache.get(session_id)
        if cached is not None:
            return cached
        lookup: dict[tuple[str, str | None], list[GraphEntity]] = {}
        for entity in self._current_entities(session_id):
            lookup.setdefault((entity.name, None), []).append(entity)
            lookup.setdefault((entity.name, entity.entity_type), []).append(entity)
        for matches in lookup.values():
            matches.sort(key=lambda entity: entity.valid_from or "", reverse=True)
        self._current_entity_lookup_cache[session_id] = lookup
        return lookup

    def _temporal_entities(self, session_id: str, temporal_point: str) -> list[GraphEntity]:
        key = (session_id, temporal_point)
        cached = self._temporal_entity_index_cache.get(key)
        if cached is not None:
            return cached
        rows = self._execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id
              AND e.valid_from <= $temporal_point
              AND (e.valid_to IS NULL OR $temporal_point < e.valid_to)
            RETURN e.name, e.entity_type, e.valid_from, e.valid_to, e.summary, e.properties_json, e.session_id,
                   e.source_event_seq, e.source_event_hash
            """,
            {"session_id": session_id, "temporal_point": temporal_point},
        ).get_all()
        entities = [_row_to_entity(row) for row in rows]
        self._temporal_entity_index_cache[key] = entities
        return entities

    def _temporal_entity_lookup(
        self,
        session_id: str,
        temporal_point: str,
    ) -> dict[tuple[str, str | None], list[GraphEntity]]:
        key = (session_id, temporal_point)
        cached = self._temporal_entity_lookup_cache.get(key)
        if cached is not None:
            return cached
        lookup: dict[tuple[str, str | None], list[GraphEntity]] = {}
        for entity in self._temporal_entities(session_id, temporal_point):
            lookup.setdefault((entity.name, None), []).append(entity)
            lookup.setdefault((entity.name, entity.entity_type), []).append(entity)
        for matches in lookup.values():
            matches.sort(key=lambda entity: entity.valid_from or "", reverse=True)
        self._temporal_entity_lookup_cache[key] = lookup
        return lookup

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
        index = self._traversal_index(session_id, temporal_point)
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
                    if target_key not in found:
                        found[target_key] = _entity_with_path_metadata(
                            target_entity,
                            relation_types=target_path_relations,
                        )
                    if target_key not in seen_sources:
                        seen_sources.add(target_key)
                        path_relations_by_key[target_key] = target_path_relations
                        next_frontier.add(target_key)
            frontier = next_frontier
            if not frontier:
                break
        return list(found.values())

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
        safe_depth = max(1, min(depth, 5))
        rows = self._causal_edge_rows(session_id=session_id, temporal_point=temporal_point)
        adjacency: dict[str, list[tuple[str, GraphEntity, dict[str, Any]]]] = {}
        keys_by_name: dict[str, set[str]] = {}
        for row in rows:
            source_key = str(row[0])
            target_key = str(row[11])
            source_entity = _row_to_entity(list(row[1:10]))
            target_entity = _row_to_entity(list(row[12:21]))
            edge_metadata = _causal_edge_metadata_from_row(row, source_entity=source_entity, target_entity=target_entity)
            if edge_metadata is None:
                continue
            if relation_type is not None and edge_metadata["relation_type"] != relation_type:
                continue
            keys_by_name.setdefault(source_entity.name, set()).add(source_key)
            keys_by_name.setdefault(target_entity.name, set()).add(target_key)
            if direction == "successors":
                adjacency.setdefault(source_key, []).append((target_key, target_entity, edge_metadata))
            else:
                adjacency.setdefault(target_key, []).append((source_key, source_entity, edge_metadata))

        start_keys = set(keys_by_name.get(entity_name, set()))
        frontier = set(start_keys)
        seen = set(start_keys)
        found: dict[str, GraphEntity] = {}
        path_relations_by_key: dict[str, list[str]] = {start_key: [] for start_key in start_keys}
        path_citations_by_key: dict[str, list[str]] = {start_key: [] for start_key in start_keys}
        for _ in range(safe_depth):
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

    def _causal_edge_rows(self, *, session_id: str, temporal_point: str | None) -> list[Any]:
        if temporal_point is None:
            rows = self._execute(
                """
                MATCH (source:Entity)-[r:RELATES]->(target:Entity)
                WHERE source.session_id = $session_id
                  AND target.session_id = $session_id
                  AND r.session_id = $session_id
                  AND source.valid_to IS NULL
                  AND target.valid_to IS NULL
                  AND r.valid_to IS NULL
                  AND r.relation_type STARTS WITH 'causal_'
                RETURN source.node_key,
                       source.name, source.entity_type, source.valid_from, source.valid_to,
                       source.summary, source.properties_json, source.session_id,
                       source.source_event_seq, source.source_event_hash,
                       r.relation_type,
                       target.node_key,
                       target.name, target.entity_type, target.valid_from, target.valid_to,
                       target.summary, target.properties_json, target.session_id,
                       target.source_event_seq, target.source_event_hash,
                       r.confidence, r.inference_method, r.source_event_seq,
                       r.source_event_hash, r.evidence_json
                """,
                {"session_id": session_id},
            ).get_all()
            return cast(list[Any], rows)
        rows = self._execute(
            """
            MATCH (source:Entity)-[r:RELATES]->(target:Entity)
            WHERE source.session_id = $session_id
              AND target.session_id = $session_id
              AND r.session_id = $session_id
              AND source.valid_from <= $temporal_point
              AND (source.valid_to IS NULL OR $temporal_point < source.valid_to)
              AND target.valid_from <= $temporal_point
              AND (target.valid_to IS NULL OR $temporal_point < target.valid_to)
              AND r.valid_from <= $temporal_point
              AND (r.valid_to IS NULL OR $temporal_point < r.valid_to)
              AND r.relation_type STARTS WITH 'causal_'
            RETURN source.node_key,
                   source.name, source.entity_type, source.valid_from, source.valid_to,
                   source.summary, source.properties_json, source.session_id,
                   source.source_event_seq, source.source_event_hash,
                   r.relation_type,
                   target.node_key,
                   target.name, target.entity_type, target.valid_from, target.valid_to,
                   target.summary, target.properties_json, target.session_id,
                   target.source_event_seq, target.source_event_hash,
                   r.confidence, r.inference_method, r.source_event_seq,
                   r.source_event_hash, r.evidence_json
            """,
            {"session_id": session_id, "temporal_point": temporal_point},
        ).get_all()
        return cast(list[Any], rows)

    def _traversal_index(self, session_id: str, temporal_point: str | None = None) -> _TraversalIndex:
        if temporal_point is not None:
            key = (session_id, temporal_point)
            cached = self._temporal_traversal_index_cache.get(key)
            if cached is not None:
                return cached
            index = self._build_traversal_index(session_id, temporal_point)
            self._temporal_traversal_index_cache[key] = index
            return index
        cached = self._traversal_index_cache.get(session_id)
        if cached is not None:
            return cached
        index = self._build_traversal_index(session_id, None)
        self._traversal_index_cache[session_id] = index
        return index

    def _build_traversal_index(self, session_id: str, temporal_point: str | None) -> _TraversalIndex:
        if temporal_point is None:
            rows = self._execute(
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
        else:
            rows = self._execute(
                """
                MATCH (source:Entity)-[r:RELATES]->(target:Entity)
                WHERE source.session_id = $session_id
                  AND target.session_id = $session_id
                  AND r.session_id = $session_id
                  AND source.valid_from <= $temporal_point
                  AND (source.valid_to IS NULL OR $temporal_point < source.valid_to)
                  AND target.valid_from <= $temporal_point
                  AND (target.valid_to IS NULL OR $temporal_point < target.valid_to)
                  AND r.valid_from <= $temporal_point
                  AND (r.valid_to IS NULL OR $temporal_point < r.valid_to)
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
                {"session_id": session_id, "temporal_point": temporal_point},
            ).get_all()
        adjacency: dict[str, list[tuple[str, GraphEntity, str]]] = {}
        keys_by_name: dict[str, set[str]] = {}
        entities_by_key: dict[str, GraphEntity] = {}
        for row in rows:
            relation = str(row[10])
            source_key = str(row[0])
            target_key = str(row[11])
            keys_by_name.setdefault(str(row[1]), set()).add(source_key)
            keys_by_name.setdefault(str(row[12]), set()).add(target_key)
            source_entity = entities_by_key.get(source_key)
            if source_entity is None:
                source_entity = _row_to_entity(list(row[1:10]))
                entities_by_key[source_key] = source_entity
            target_entity = entities_by_key.get(target_key)
            if target_entity is None:
                target_entity = _row_to_entity(list(row[12:]))
                entities_by_key[target_key] = target_entity
            adjacency.setdefault(source_key, []).append((target_key, target_entity, relation))
            adjacency.setdefault(target_key, []).append((source_key, source_entity, relation))
        return _TraversalIndex(adjacency=adjacency, keys_by_name=keys_by_name)

    async def has_traversal_edges(self, session_id: str = "default") -> bool:
        """Return whether the embedded projection has active graph edges."""
        cached = self._traversal_index_cache.get(session_id)
        if cached is not None:
            return bool(cached.adjacency)
        rows = self._execute(
            """
            MATCH (:Entity)-[r:RELATES]->(:Entity)
            WHERE r.session_id = $session_id AND r.valid_to IS NULL
            RETURN count(r)
            """,
            {"session_id": session_id},
        ).get_all()
        return bool(rows and rows[0][0])

    async def fetch_adjacency(self, session_id: str = "default") -> AdjacencySnapshot:
        """Fetch the session's active-graph adjacency snapshot for graph walks.

        This is the embedded implementation of the
        :class:`zaxy.graph_walk.AdjacencyProvider` contract. Node identity is
        ``Entity.node_key``; every active entity version appears as a node
        (including isolated ones, so query-matched seeds resolve before they
        gain edges). Each active ``RELATES`` row uses the same active-edge
        query family as ``_build_traversal_index`` and contributes both its
        stored direction and the reverse direction, exactly matching the
        traversal index's undirected adjacency semantics (it registers every
        edge under both endpoints). Parallel ``RELATES`` rows between the same
        pair keep their multiplicity and proportionally weight the walk.

        The snapshot ``signature`` is a deterministic content hash of the
        node and edge sets, so it changes exactly when the projected graph
        changes. Snapshots are cached per session and invalidated alongside
        the other derived read indexes (``_clear_read_caches``).
        """
        cached = self._adjacency_snapshot_cache.get(session_id)
        if cached is not None:
            return cached
        snapshot = self._build_adjacency_snapshot(session_id)
        self._adjacency_snapshot_cache[session_id] = snapshot
        return snapshot

    def _build_adjacency_snapshot(self, session_id: str) -> AdjacencySnapshot:
        node_rows = self._execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id AND e.valid_to IS NULL
            RETURN e.node_key
            """,
            {"session_id": session_id},
        ).get_all()
        edge_rows = self._execute(
            """
            MATCH (source:Entity)-[r:RELATES]->(target:Entity)
            WHERE source.session_id = $session_id
              AND target.session_id = $session_id
              AND r.session_id = $session_id
              AND source.valid_to IS NULL
              AND target.valid_to IS NULL
              AND r.valid_to IS NULL
            RETURN source.node_key, target.node_key
            """,
            {"session_id": session_id},
        ).get_all()
        node_ids = sorted({str(row[0]) for row in node_rows})
        edges: list[tuple[str, str]] = []
        for row in edge_rows:
            source_key = str(row[0])
            target_key = str(row[1])
            edges.append((source_key, target_key))
            edges.append((target_key, source_key))
        edges.sort()
        return AdjacencySnapshot.from_edges(
            node_ids,
            edges,
            signature=_adjacency_signature(session_id, node_ids, edges),
        )

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
        embedding_version: str | None = None,
    ) -> list[SearchResult]:
        """Search by vector similarity within one embedding version group.

        All groups share one strategy pipeline: candidate selection (exact
        dense matrix, engine-native HNSW, or int8 dot products) followed by an exact
        float64 rerank for the approximate selectors. Dense selection already
        scores exactly, so it returns directly with ``exact=True``; the
        approximate paths stay ``exact=False`` because their candidate set is
        approximate even though the final ordering is exact.
        """
        if limit <= 0:
            return []
        query = np.asarray(embedding, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []
        version = embedding_version or self._resolved_active_embedding_version()
        index = self._vector_index(session_id, temporal_point)
        group = index.groups.get((len(embedding), version))
        if group is None:
            return []
        unit_query = query / query_norm
        if isinstance(group, _VectorGroup):
            return _dense_vector_results(group, unit_query, limit=limit, entities=index.entities)
        if isinstance(group, _AnnVectorGroup):
            candidates = self._ann_candidate_entity_indexes(group, unit_query, limit=limit)
        else:
            candidates = _quantized_candidate_entity_indexes(group, unit_query, limit=limit)
        return _exact_rerank_results(candidates, unit_query, limit=limit, entities=index.entities)

    def _ann_candidate_entity_indexes(
        self,
        group: _AnnVectorGroup,
        unit_query: npt.NDArray[np.float64],
        *,
        limit: int,
    ) -> list[int]:
        """Select oversampled HNSW candidates from the group's shadow table.

        The per-(session, version) shadow table is queried directly by name:
        the group owns the whole table, so no session/version predicate and no
        projected graph is needed (the per-query prefilter mask scan — not
        graph create/drop — dominated filtered-query latency at 10^5). HNSW
        retrieves ``limit * VECTOR_SEARCH_OVERSAMPLE`` candidates and the
        exact float64 rerank restores the true ordering: the shadow table
        stores float32, and near-tie flips at that precision boundary were the
        measured recall deficit of the unreranked path.
        """
        k = min(group.vector_count, limit * VECTOR_SEARCH_OVERSAMPLE)
        if k <= 0:
            return []
        # efs is the HNSW query-time candidate-list size; it can never be
        # smaller than the number of results asked of the index.
        efs = max(self._resolved_vector_ann_efs(), k)
        rows = self._execute(
            f"CALL QUERY_VECTOR_INDEX('{group.table_name}', '{group.index_name}', "
            f"$query_vector, $k, efs := {efs}) RETURN node.entity_row",
            {"query_vector": unit_query.tolist(), "k": k},
        ).get_all()
        return sorted({int(row[0]) for row in rows})

    def _resolved_active_embedding_version(self) -> str:
        if self._active_embedding_version_override is not None:
            return self._active_embedding_version_override
        from zaxy.embedding import resolved_active_embedding_version_tag

        return resolved_active_embedding_version_tag() or LEGACY_EMBEDDING_VERSION

    def _resolved_vector_ann_threshold(self) -> int:
        if self._vector_ann_threshold_override is not None:
            return self._vector_ann_threshold_override
        from zaxy.config import get_settings

        return get_settings().vector_ann_threshold

    def _resolved_vector_ann_max_dimension(self) -> int:
        if self._vector_ann_max_dimension_override is not None:
            return self._vector_ann_max_dimension_override
        from zaxy.config import get_settings

        return get_settings().vector_ann_max_dimension

    def _resolved_vector_ann_efs(self) -> int:
        if self._vector_ann_efs_override is not None:
            return self._vector_ann_efs_override
        from zaxy.config import get_settings

        return get_settings().vector_ann_efs

    def _resolved_vector_ann_byte_budget_engagement(self) -> bool:
        if self._vector_ann_byte_budget_engagement_override is not None:
            return self._vector_ann_byte_budget_engagement_override
        from zaxy.config import get_settings

        return get_settings().vector_ann_byte_budget_engagement

    def _ann_engagement_reason(self, *, count: int, dimension: int) -> str | None:
        """Return why the ANN path engages for a scope, or None to stay resident.

        Engagement rule (2.2 gate G4): the scope's vector dimension must be
        at or below ``vector_ann_max_dimension`` — a hard precondition — and
        then either clause may engage:

        - ``"count"``: the per-scope vector count is at or above
          ``vector_ann_threshold`` — the lane-proven count clause (two
          consecutive ALL-criteria vector-scale lane passes at exactly
          10^5/dim 64; see
          docs/research/artifacts/ann-2026-06/ann3-d64-100k-r1.json and -r2.json).
        - ``"byte_budget"``: the exact float64 matrix for the scope
          (count x dimension x 8 bytes) would exceed
          ``VECTOR_INDEX_CACHE_MAX_BYTES``, regardless of any explicit
          threshold. It is skipped when int8 quantization is opted in (int8
          keeps roughly 1/8 of the float64 bytes resident, so the explicit
          opt-in keeps its precedence below the count threshold) or when
          ``vector_ann_byte_budget_engagement`` is disabled.

        The dimension precondition exists because the G4 evidence does not
        transfer upward: at dim 1536/50k (gaussian) HNSW recall@10 is 0.6
        even at efs 400, while the exact matrix answers in 22ms p50 despite
        sitting 2.4x over the cache byte budget — the LRU eviction always
        keeps the newest matrix resident, so a single over-budget scope is a
        cache of one, not a thrash (the budget bounds multi-scope totals).
        See docs/research/artifacts/ann-2026-06/ann3-d1536-50k-gauss-crossover.json.
        This is the single ANN selection choke point — query-time strategy
        choice and shadow rebuild triggers both pass through it, so no shadow
        generation is ever built for a scope that would not be queried
        through it.

        The check is intentionally cheap: count and dimension are known
        before any matrix is constructed.
        """
        if dimension > self._resolved_vector_ann_max_dimension():
            return None
        if count >= self._resolved_vector_ann_threshold():
            return "count"
        if (
            self._resolved_vector_quantization() != "int8"
            and self._resolved_vector_ann_byte_budget_engagement()
            and count * dimension * 8 > VECTOR_INDEX_CACHE_MAX_BYTES
        ):
            return "byte_budget"
        return None

    def _resolved_vector_quantization(self) -> str:
        if self._vector_quantization_override is not None:
            return self._vector_quantization_override
        from zaxy.config import get_settings

        return get_settings().vector_quantization

    def _vector_index(self, session_id: str, temporal_point: str | None) -> _VectorIndex:
        key = (session_id, temporal_point)
        cached = self._vector_index_cache.get(key)
        if cached is not None:
            # Move-to-end so LRU eviction drops the least recently used index.
            self._vector_index_cache[key] = self._vector_index_cache.pop(key)
            return cached
        candidate_entities = (
            self._current_entities(session_id)
            if temporal_point is None
            else self._temporal_entities(session_id, temporal_point)
        )
        entities: list[GraphEntity] = []
        unit_vectors: dict[tuple[int, str], list[npt.NDArray[np.float64]]] = {}
        group_entity_indexes: dict[tuple[int, str], list[int]] = {}
        for entity in candidate_entities:
            vector = _embedding_vector(entity.properties.get("embedding"))
            if vector is None:
                continue
            values = np.asarray(vector, dtype=np.float64)
            norm = float(np.linalg.norm(values))
            if norm == 0.0:
                continue
            entity.properties["embedding_dimension"] = len(vector)
            entity_index = len(entities)
            entities.append(entity)
            group_key = (len(vector), _embedding_version(entity.properties))
            unit_vectors.setdefault(group_key, []).append(values / norm)
            group_entity_indexes.setdefault(group_key, []).append(entity_index)
        groups: dict[tuple[int, str], _AnyVectorGroup] = {
            group_key: self._build_vector_group(
                session_id=session_id,
                temporal_point=temporal_point,
                dimension=group_key[0],
                version=group_key[1],
                vectors=vectors,
                entity_indexes=group_entity_indexes[group_key],
            )
            for group_key, vectors in unit_vectors.items()
        }
        index = _VectorIndex(entities=entities, groups=groups)
        self._vector_index_cache[key] = index
        self._evict_vector_indexes_over_budget()
        return index

    def _build_vector_group(
        self,
        *,
        session_id: str,
        temporal_point: str | None,
        dimension: int,
        version: str,
        vectors: list[npt.NDArray[np.float64]],
        entity_indexes: list[int],
    ) -> _AnyVectorGroup:
        if (
            temporal_point is None
            and not self._bulk_projection_open
            and self._ann_engagement_reason(count=len(vectors), dimension=dimension) is not None
        ):
            ann_group = self._try_build_ann_group(
                session_id=session_id,
                dimension=dimension,
                version=version,
                vectors=vectors,
                entity_indexes=entity_indexes,
            )
            if ann_group is not None:
                return ann_group
        matrix = np.vstack(vectors)
        if self._resolved_vector_quantization() == "int8":
            quantized, scales = _quantize_unit_matrix(matrix)
            return _QuantizedVectorGroup(matrix=quantized, scales=scales, entity_indexes=entity_indexes)
        return _VectorGroup(matrix=matrix, entity_indexes=entity_indexes)

    def _try_build_ann_group(
        self,
        *,
        session_id: str,
        dimension: int,
        version: str,
        vectors: list[npt.NDArray[np.float64]],
        entity_indexes: list[int],
    ) -> _AnnVectorGroup | None:
        """Sync this group into an HNSW shadow table, if the runtime supports it.

        Each (session, version, dimension) scope owns dedicated shadow tables
        so queries pass the table name straight to ``QUERY_VECTOR_INDEX`` —
        no predicate and no projected graph. The rebuild trigger is the same
        lazy signature change that rebuilds the dense matrix; the delta policy
        decides the sync mechanics:

        - **Unchanged corpus** (content digest matches the resident
          generation): reuse the resident table with no writes at all.
        - **Small extension** (digest-verified prefix, delta at most
          ``_ANN_DELTA_REBUILD_FRACTION`` of the resident count): insert only
          the new rows into the live mutable index — inserts are reflected in
          queries on LadybugDB 0.17.1.
        - **Anything else**: full rebuild into a fresh generation table via
          ``COPY FROM`` an in-memory Arrow table (the COPY family measured
          ~100x faster than batched UNWIND; falls back to UNWIND without
          pyarrow), index built after the load, then an atomic state swap.

        A fresh generation's HNSW index only ever sees inserts. Generations
        remain the swap mechanism for atomicity (queries only ever hit a
        fully built table) and because one residual engine hole persists on
        LadybugDB 0.17.1: a single-statement delete-ALL under a live index
        permanently breaks subsequent index searches (re-verified), so a
        live generation is never bulk-emptied. Superseded generations are
        dropped outright — ``DROP_VECTOR_INDEX`` first, then ``DROP TABLE``,
        because the binder still rejects dropping a table that an index
        references — for full space reclaim; the kuzu#6040 drop-index
        metadata corruption is fixed in the fork and re-verified through
        this store's own rebuild cycle.
        """
        if not self._vector_index_supported():
            return None
        entity_rows = np.asarray(entity_indexes, dtype=np.int64)
        matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)

        def group_for(table_name: str, index_name: str) -> _AnnVectorGroup:
            return _AnnVectorGroup(
                table_name=table_name,
                index_name=index_name,
                dimension=dimension,
                version=version,
                session_id=session_id,
                vector_count=len(entity_indexes),
            )

        scope_key = (session_id, version, dimension)
        state = self._ann_generation_states.get(scope_key)
        if state is not None:
            delta = len(entity_indexes) - state.vector_count
            if 0 <= delta <= max(1, int(state.vector_count * _ANN_DELTA_REBUILD_FRACTION)) and (
                _ann_content_digest(entity_rows[: state.vector_count], matrix[: state.vector_count])
                == state.content_digest
            ):
                if delta:
                    self._ann_insert_rows(state.table_name, entity_rows, matrix, start=state.vector_count)
                    state = _AnnGenerationState(
                        table_name=state.table_name,
                        index_name=state.index_name,
                        generation=state.generation,
                        vector_count=len(entity_indexes),
                        content_digest=_ann_content_digest(entity_rows, matrix),
                    )
                    self._ann_generation_states[scope_key] = state
                return group_for(state.table_name, state.index_name)
        scope = _ann_scope_digest(session_id, version)
        scope_prefix = f"{_ANN_SHADOW_TABLE_PREFIX}{dimension}_{scope}_g"
        previous_generations = self._ann_shadow_generations(scope_prefix)
        generation = previous_generations[-1] + 1 if previous_generations else 0
        table_name = f"{scope_prefix}{generation}"
        index_name = f"zaxy_vector_ann_{dimension}_{scope}_g{generation}"
        self._execute(
            f"""
            CREATE NODE TABLE {table_name}(
                entity_row INT64,
                vec FLOAT[{dimension}],
                PRIMARY KEY(entity_row)
            )
            """
        )
        self._ann_bulk_load(table_name, entity_rows, matrix)
        # The index is created after the bulk load: building over resident
        # rows is the documented fast path, and bulk mutation under a live
        # index remains the engine lineage's historically buggiest path
        # (SET-on-indexed is still rejected; delete-all still breaks it).
        self._ensure_ann_index(table_name, index_name)
        self._ann_generation_states[scope_key] = _AnnGenerationState(
            table_name=table_name,
            index_name=index_name,
            generation=generation,
            vector_count=len(entity_indexes),
            content_digest=_ann_content_digest(entity_rows, matrix),
        )
        for old_generation in previous_generations:
            self._drop_ann_generation(
                f"{scope_prefix}{old_generation}",
                f"zaxy_vector_ann_{dimension}_{scope}_g{old_generation}",
            )
        return group_for(table_name, index_name)

    def _drop_ann_generation(self, table_name: str, index_name: str) -> None:
        """Drop one superseded shadow generation (index first, then table).

        Full space reclaim: LadybugDB fixed the kuzu#6040 DROP_VECTOR_INDEX
        metadata corruption, so superseded generations no longer linger as
        emptied husks. ``DROP TABLE`` is still binder-rejected while an index
        references the table, so the index is dropped first; a generation
        whose index build was interrupted has no index to drop and falls
        through the missing-index guard to the table drop.
        """
        try:
            self._execute(f"CALL DROP_VECTOR_INDEX('{table_name}', '{index_name}')")
        except RuntimeError as exc:
            if "have an index" not in str(exc):
                raise
        self._execute(f"DROP TABLE {table_name}")
        self._ann_indexed_tables.discard(table_name)

    def _ann_bulk_load(
        self,
        table_name: str,
        entity_rows: npt.NDArray[np.int64],
        matrix: npt.NDArray[np.float32],
    ) -> None:
        """Bulk load one shadow generation, preferring in-memory Arrow ``COPY``.

        ``COPY FROM`` is the documented bulk path and measured ~100x faster
        than batched UNWIND at 10k vectors. Kuzu 0.11.3 segfaulted on a COPY
        from an in-memory Arrow table with a fixed-size-list column, forcing
        a parquet-tempfile round-trip; LadybugDB 0.17.1 fixed that (verified
        through this exact path and schema), so the data now flows straight
        from the Arrow table with no disk round-trip. pyarrow is a transitive
        (not guaranteed) dependency, so its absence degrades to the
        batched-UNWIND load rather than failing.
        """
        if importlib.util.find_spec("pyarrow") is None:
            self._ann_insert_rows(table_name, entity_rows, matrix)
            return
        import pyarrow as pa

        vec_column = pa.FixedSizeListArray.from_arrays(
            pa.array(matrix.reshape(-1), type=pa.float32()),
            matrix.shape[1],
        )
        arrow_table = pa.table({"entity_row": pa.array(entity_rows, type=pa.int64()), "vec": vec_column})
        self._execute(f"COPY {table_name} FROM $arrow_rows", {"arrow_rows": arrow_table})

    def _ann_insert_rows(
        self,
        table_name: str,
        entity_rows: npt.NDArray[np.int64],
        matrix: npt.NDArray[np.float32],
        *,
        start: int = 0,
    ) -> None:
        """Insert shadow rows with batched UNWIND statements from ``start`` on.

        Serves the incremental delta path (inserts into the live mutable
        index) and the full-load fallback when pyarrow is unavailable.
        """
        for batch_start in range(start, len(entity_rows), _ANN_INSERT_BATCH_SIZE):
            batch = [
                {
                    "entity_row": int(entity_rows[position]),
                    "vec": matrix[position].tolist(),
                }
                for position in range(batch_start, min(batch_start + _ANN_INSERT_BATCH_SIZE, len(entity_rows)))
            ]
            self._execute(
                f"UNWIND $rows AS row CREATE (:{table_name} {{entity_row: row.entity_row, vec: row.vec}})",
                {"rows": batch},
            )

    def _ann_shadow_generations(self, scope_prefix: str) -> list[int]:
        """List existing shadow generation numbers for one scope, ascending."""
        rows = self._execute("CALL SHOW_TABLES() RETURN name").get_all()
        generations = []
        for row in rows:
            suffix = str(row[0]).removeprefix(scope_prefix)
            if suffix != str(row[0]) and suffix.isdigit():
                generations.append(int(suffix))
        return sorted(generations)

    def _ensure_ann_index(self, table_name: str, index_name: str) -> None:
        """Create the persistent HNSW index for a shadow table exactly once."""
        if table_name in self._ann_indexed_tables:
            return
        try:
            self._execute(
                f"CALL CREATE_VECTOR_INDEX('{table_name}', '{index_name}', 'vec', metric := 'cosine')"
            )
        except RuntimeError as exc:
            if "already exists" not in str(exc):
                raise
        self._ann_indexed_tables.add(table_name)

    def _vector_index_supported(self) -> bool:
        """Probe once per connection whether the runtime ships the vector index.

        ``connect()`` installs (once, network-cached) and ``LOAD``s the
        ``vector`` extension the LadybugDB runtime requires — unlike Kuzu
        0.11.3, it is neither bundled nor auto-registered — and resets this
        probe, so the cached answer always describes the live connection.
        When the extension cannot be installed or loaded (e.g. an offline
        first run) the probe stays ``False`` and retrieval uses exact search.
        The probe uses
        the actual vector-index operation instead of catalog introspection:
        LadybugDB wheels can differ in how table functions appear in
        ``SHOW_FUNCTIONS()``, but Zaxy only needs to know whether a shadow
        vector table can build and drop an index.
        """
        if self._ann_supported is None:
            probe_table = "zaxy_vector_capability_probe"
            probe_index = "zaxy_vector_capability_probe_idx"
            try:
                self._execute(
                    f"CREATE NODE TABLE IF NOT EXISTS {probe_table}("
                    "id INT64, vec FLOAT[2], PRIMARY KEY(id))"
                )
                self._execute(
                    f"CALL CREATE_VECTOR_INDEX('{probe_table}', '{probe_index}', 'vec', metric := 'cosine')"
                )
            except RuntimeError:
                self._ann_supported = False
                with suppress(RuntimeError):
                    self._execute(f"DROP TABLE {probe_table}")
            else:
                self._ann_supported = True
                with suppress(RuntimeError):
                    self._execute(f"CALL DROP_VECTOR_INDEX('{probe_table}', '{probe_index}')")
                with suppress(RuntimeError):
                    self._execute(f"DROP TABLE {probe_table}")
        return self._ann_supported

    async def re_embed_session(
        self,
        *,
        session_id: str,
        provider: Any,
        version_tag: str | None = None,
    ) -> dict[str, int]:
        """Re-embed projected entity vectors onto the provider's version tag.

        This mutates only projection state (projected Entity rows); Eventloom events
        are never touched. Both active and historical entity versions are
        migrated so temporal vector search stays version-consistent.
        """
        from zaxy.embedding import embedding_text, provider_version_tag

        tag = version_tag or provider_version_tag(provider)
        if not tag:
            raise ValueError("embedding provider does not expose a version tag")
        try:
            rows = self._execute(
                """
                MATCH (e:Entity)
                WHERE e.session_id = $session_id
                  AND contains(e.properties_json, '"embedding"')
                RETURN e.node_key, e.name, e.entity_type, e.summary, e.properties_json
                """,
                {"session_id": session_id},
            ).get_all()
        except RuntimeError as exc:
            if not _is_missing_projection_table_error(exc):
                raise
            rows = []
        scanned = 0
        re_embedded = 0
        already_current = 0
        for row in rows:
            properties = _json_dict(row[4])
            vector = _embedding_vector(properties.get("embedding"))
            if vector is None:
                continue
            scanned += 1
            if _embedding_version(properties) == tag:
                already_current += 1
                continue
            text = embedding_text(
                str(row[1]),
                str(row[2]),
                str(row[3]) if row[3] is not None else None,
            )
            new_vector = provider.embed(text)
            properties["embedding"] = [float(value) for value in new_vector]
            properties["embedding_version"] = tag
            self._execute(
                """
                MATCH (e:Entity {node_key: $node_key})
                SET e.properties_json = $properties_json
                """,
                {
                    "node_key": str(row[0]),
                    "properties_json": json.dumps(properties, sort_keys=True),
                },
            )
            re_embedded += 1
        if re_embedded:
            self._active_entity_cache = {
                cache_key: value
                for cache_key, value in self._active_entity_cache.items()
                if cache_key[0] != session_id
            }
            self._clear_read_caches(session_id)
        return {
            "scanned": scanned,
            "re_embedded": re_embedded,
            "already_current": already_current,
        }

    async def embedding_version_counts(self, session_id: str) -> dict[str, int]:
        """Count active projected vectors per embedding version tag."""
        counts: dict[str, int] = {}
        for entity in self._current_entities(session_id):
            if _embedding_vector(entity.properties.get("embedding")) is None:
                continue
            version = _embedding_version(entity.properties)
            counts[version] = counts.get(version, 0) + 1
        return counts

    def _evict_vector_indexes_over_budget(self) -> None:
        """Drop least-recently-used vector indexes beyond the entry/byte budget.

        The newest index always survives, even alone over budget, so a single
        large session degrades to cache-of-one rather than rebuild-per-query.
        """
        while len(self._vector_index_cache) > 1 and (
            len(self._vector_index_cache) > VECTOR_INDEX_CACHE_MAX_ENTRIES
            or sum(index.matrix_bytes for index in self._vector_index_cache.values())
            > VECTOR_INDEX_CACHE_MAX_BYTES
        ):
            self._vector_index_cache.pop(next(iter(self._vector_index_cache)))

    def _clear_vector_index_cache(self, session_id: str) -> None:
        self._vector_index_cache = self._clear_session_keyed_cache(self._vector_index_cache, session_id)

    def _clear_temporal_read_caches(self, session_id: str) -> None:
        self._temporal_entity_index_cache = self._clear_session_keyed_cache(
            self._temporal_entity_index_cache,
            session_id,
        )
        self._temporal_entity_lookup_cache = self._clear_session_keyed_cache(
            self._temporal_entity_lookup_cache,
            session_id,
        )
        self._temporal_keyword_index_cache = self._clear_session_keyed_cache(
            self._temporal_keyword_index_cache,
            session_id,
        )
        self._temporal_traversal_index_cache = self._clear_session_keyed_cache(
            self._temporal_traversal_index_cache,
            session_id,
        )

    def _clear_session_keyed_cache(
        self,
        cache: dict[tuple[str, Any], _CacheValue],
        session_id: str,
    ) -> dict[tuple[str, Any], _CacheValue]:
        return {key: value for key, value in cache.items() if key[0] != session_id}

    def _clear_read_caches(self, session_id: str) -> None:
        self._current_entity_index_cache.pop(session_id, None)
        self._current_entity_lookup_cache.pop(session_id, None)
        self._clear_temporal_read_caches(session_id)
        self._keyword_index_cache.pop(session_id, None)
        self._clear_vector_index_cache(session_id)
        self._traversal_index_cache.pop(session_id, None)
        self._adjacency_snapshot_cache.pop(session_id, None)

    def _clear_all_caches(self) -> None:
        self._active_entity_cache = {}
        self._current_entity_index_cache = {}
        self._current_entity_lookup_cache = {}
        self._temporal_entity_index_cache = {}
        self._temporal_entity_lookup_cache = {}
        self._keyword_index_cache = {}
        self._temporal_keyword_index_cache = {}
        self._vector_index_cache = {}
        self._traversal_index_cache = {}
        self._temporal_traversal_index_cache = {}
        self._adjacency_snapshot_cache = {}
        self._dirty_bulk_sessions = set()
        self._bulk_active_state_loaded_sessions = set()
        self._ann_indexed_tables = set()
        self._ann_generation_states = {}

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Close an entity validity window."""
        self._execute(
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
        self._clear_read_caches(session_id)

    async def retire_source_projections(
        self,
        *,
        source_path: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Close projections derived from one source."""
        rows = self._execute(
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
            self._execute(
                """
                MATCH (e:Entity {node_key: $node_key})
                SET e.valid_to = $invalid_at
                """,
                {"node_key": node_key, "invalid_at": invalid_at},
            )
            self._execute(
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
            self._clear_read_caches(session_id)

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        """Inspect Eventloom projection integrity."""
        from zaxy.graph import GraphEventProjectionStatus

        try:
            event_rows = self._execute(
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
            self._execute(
                """
                MATCH (:Event)-[r:NEXT_EVENT]->(:Event)
                WHERE r.session_id = $session_id
                RETURN count(r)
                """,
                {"session_id": session_id},
            ).get_all()
        )
        previous_event_edges = _first_count(
            self._execute(
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
            and next_event_edges == max(0, event_count - 1)
            and previous_event_edges == max(0, event_count - 1)
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
            rows = self._execute(
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
        for row in rows:
            evidence = _json_dict(row[7])
            if evidence:
                evidence_count += 1
            if row[5] is None:
                missing_source_event_count += 1
            method = str(row[4] or "unknown")
            method_rows.setdefault(method, []).append(row)

        samples = []
        for row in rows[:limit]:
            evidence = _json_dict(row[7])
            method = str(row[4] or "unknown")
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

    def _execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute one store query with guaranteed parameter binding.

        This is the single query-execution choke point: LadybugDB 0.17.1
        silently evaluates an unbound ``$parameter`` to NULL — the query
        "succeeds" with wrong answers, strictly more dangerous than the
        Kuzu 0.11.3 segfault this guard was built for — so every
        ``$placeholder`` must arrive with a binding before the statement
        reaches the runtime. JSON-shaped string bindings are additionally
        rewritten to literals here because the runtime corrupts them (see
        :func:`_armor_json_shaped_string_parameters`).
        """
        placeholders = set(_QUERY_PARAMETER_RE.findall(query))
        missing = placeholders.difference(parameters or ())
        if missing:
            raise RuntimeError(f"query references unbound parameters: {sorted(missing)}")
        connection = self._require_connection()
        if parameters is None:
            return connection.execute(query)
        query, parameters = _armor_json_shaped_string_parameters(query, parameters)
        return connection.execute(query, parameters)

    def _active_node_key(self, session_id: str, entity_type: str, name: str) -> str | None:
        active_entity = self._active_entity_state(session_id=session_id, entity_type=entity_type, name=name)
        return active_entity[0] if active_entity is not None else None

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
        if self._bulk_projection_open:
            self._load_bulk_active_state(session_id)
            return self._active_entity_cache.get((session_id, entity_type, name))
        rows = self._execute(
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

    def _load_bulk_active_state(self, session_id: str) -> None:
        """Load active entity state once for a bulk projection transaction."""
        if session_id in self._bulk_active_state_loaded_sessions:
            return
        rows = self._execute(
            """
            MATCH (e:Entity)
            WHERE e.session_id = $session_id AND e.valid_to IS NULL
            RETURN e.node_key, e.name, e.entity_type, e.summary, e.properties_json
            """,
            {"session_id": session_id},
        ).get_all()
        for row in rows:
            self._active_entity_cache[(session_id, str(row[2]), str(row[1]))] = (
                str(row[0]),
                str(row[3]) if row[3] is not None else None,
                str(row[4] or "{}"),
            )
        self._bulk_active_state_loaded_sessions.add(session_id)

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
        incoming_rows = self._execute(
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

        outgoing_rows = self._execute(
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
        self._execute(
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


__all__ = [
    "EmbeddedGraphStore",
    "LEGACY_EMBEDDING_VERSION",
    "VECTOR_INDEX_CACHE_MAX_BYTES",
    "VECTOR_INDEX_CACHE_MAX_ENTRIES",
    "_AnnVectorGroup",
    "_QuantizedVectorGroup",
    "_causal_edge_metadata_from_row",
    "_is_missing_projection_table_error",
    "_keyword_candidate_terms",
    "_keyword_index_from_entities",
    "_keyword_query_terms",
    "_properties_reference_source",
    "_terms",
    "pre_ladybug_backup_paths",
]
