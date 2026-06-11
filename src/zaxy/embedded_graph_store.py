"""Embedded graph projection backend.

This module owns the zero-friction local graph runtime. The Kuzu adapter is
exposed through the ProjectionStore contract so local default deployments and
same-harness backend comparisons share the same retrieval surface.
"""

from __future__ import annotations

import hashlib
import heapq
import importlib.util
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt

from zaxy.graph_walk import AdjacencySnapshot

_CacheValue = TypeVar("_CacheValue")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_KEYWORD_STOP_WORDS = frozenset(
    {
        "am",
        "and",
        "are",
        "at",
        "did",
        "do",
        "does",
        "first",
        "for",
        "had",
        "have",
        "how",
        "in",
        "it",
        "me",
        "of",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)

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
    term_counts: list[Counter[str]]
    term_entity_ids: dict[str, tuple[int, ...]]
    term_idf: dict[str, float]
    document_length_norms: list[float]


VECTOR_INDEX_CACHE_MAX_ENTRIES = 8
VECTOR_INDEX_CACHE_MAX_BYTES = 256 * 1024 * 1024

# Vectors persisted before embedding versioning carry no tag; they are grouped
# under this sentinel and never compared against tagged vectors.
LEGACY_EMBEDDING_VERSION = "legacy"

# Approximate candidate selection (int8 dot products or HNSW) gathers
# top-k * oversample candidates before the shared exact float64 rerank.
VECTOR_SEARCH_OVERSAMPLE = 4

_ANN_SHADOW_TABLE_PREFIX = "ZaxyVectorAnnShadow"
_ANN_INSERT_BATCH_SIZE = 1024

# Delta policy for shadow rebuilds: when the new corpus extends the resident
# generation unchanged (same leading rows, same vectors), a delta up to this
# fraction of the resident count rides live-index UNWIND inserts; anything
# larger (or any non-extension change) triggers a full COPY generation swap.
# Measured at 10k/dim64: live-index inserts cost ~6.9ms/row while a COPY
# load plus post-load index build costs ~0.63ms/row, so raw build time
# crosses over near a 9% delta. The boundary sits at 10% because incremental
# inserts also keep the resident HNSW graph (no rebuild recall variance) and
# add no superseded generation table; that is worth the <=10% time premium
# in the 9-10% band.
_ANN_DELTA_REBUILD_FRACTION = 0.1

# Kuzu 0.11.3 segfaults (rather than raising) when a query references an
# unbound $parameter, so every store query is checked against this pattern at
# the single execution choke point before it reaches the runtime.
_QUERY_PARAMETER_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class _VectorGroup:
    """Unit-normalized embedding matrix for one (dimension, version) group."""

    matrix: npt.NDArray[np.float64]
    entity_indexes: list[int]

    @property
    def matrix_bytes(self) -> int:
        return int(self.matrix.nbytes)


@dataclass(frozen=True)
class _QuantizedVectorGroup:
    """Int8-quantized unit vectors with per-vector scale factors.

    Original float vectors are not duplicated here: the exact rerank reads them
    from the ``embedding`` property of the entity objects the index already
    references, so quantization saves the full float64 matrix.
    """

    matrix: npt.NDArray[np.int8]
    scales: npt.NDArray[np.float64]
    entity_indexes: list[int]

    @property
    def matrix_bytes(self) -> int:
        return int(self.matrix.nbytes + self.scales.nbytes)


@dataclass(frozen=True)
class _AnnVectorGroup:
    """Kuzu-native HNSW-backed group; vectors are resident in the database."""

    table_name: str
    index_name: str
    dimension: int
    version: str
    session_id: str
    vector_count: int

    @property
    def matrix_bytes(self) -> int:
        return 0


@dataclass(frozen=True)
class _AnnGenerationState:
    """Resident shadow generation for one (session, version, dimension) scope.

    ``content_digest`` hashes the (entity_row, float32 vector) sequence the
    generation table holds, so a rebuild can prove the new corpus is an
    unchanged extension before riding the incremental insert path. The swap
    to a fresh generation replaces this record atomically (single assignment)
    before superseded generations are emptied.
    """

    table_name: str
    index_name: str
    generation: int
    vector_count: int
    content_digest: str


_AnyVectorGroup = _VectorGroup | _QuantizedVectorGroup | _AnnVectorGroup


@dataclass(frozen=True)
class _VectorIndex:
    entities: list[GraphEntity]
    groups: dict[tuple[int, str], _AnyVectorGroup]

    @property
    def matrix_bytes(self) -> int:
        return sum(group.matrix_bytes for group in self.groups.values())


@dataclass(frozen=True)
class _TraversalIndex:
    adjacency: dict[str, list[tuple[str, GraphEntity, str]]]
    keys_by_name: dict[str, set[str]]


class EmbeddedGraphStore:
    """Kuzu-backed embedded projection store."""

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
        """Open embedded graph resources."""
        if importlib.util.find_spec("kuzu") is None:
            raise RuntimeError('embedded graph backend requires `pip install "zaxy-memory"`')
        import kuzu

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._database = kuzu.Database(str(self.path))
        self._connection = kuzu.Connection(self._database)
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
        """Begin an explicit Kuzu write transaction for bulk Eventloom replay."""
        if self._bulk_projection_open:
            return
        self._execute("BEGIN TRANSACTION")
        self._bulk_projection_open = True

    async def commit_bulk_projection(self) -> None:
        """Commit an explicit Kuzu write transaction for bulk Eventloom replay."""
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
        """Rollback an explicit Kuzu write transaction for bulk Eventloom replay."""
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
        dense matrix, Kuzu HNSW, or int8 dot products) followed by an exact
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
        """Sync this group into a Kuzu HNSW shadow table, if the runtime supports it.

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
          queries on the pinned Kuzu 0.11.3.
        - **Anything else**: full rebuild into a fresh generation table via
          ``COPY FROM`` a parquet tempfile (100x faster than batched UNWIND;
          falls back to UNWIND without pyarrow), index built after the load,
          then an atomic state swap.

        A fresh generation's HNSW index only ever sees inserts: mutating a
        live index in place (delete + reinsert) silently breaks subsequent
        direct-table searches on the pinned Kuzu 0.11.3, and the upstream
        drop paths are closed too (``DROP_VECTOR_INDEX`` leaves an
        un-checkpointed index removal per kuzu#6040, ``DROP TABLE`` is
        rejected by the binder while an index references the table — both
        re-verified against kuzu 0.11.3). Superseded generations are
        therefore emptied — never queried again, near-zero load cost at
        database open — which is the maximum reclaim the frozen runtime
        allows.
        """
        if not self._kuzu_vector_index_supported():
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
        # index is the historically buggiest path of the frozen runtime.
        self._ensure_ann_index(table_name, index_name)
        self._ann_generation_states[scope_key] = _AnnGenerationState(
            table_name=table_name,
            index_name=index_name,
            generation=generation,
            vector_count=len(entity_indexes),
            content_digest=_ann_content_digest(entity_rows, matrix),
        )
        for old_generation in previous_generations:
            old_table = f"{scope_prefix}{old_generation}"
            self._execute(f"MATCH (n:{old_table}) DETACH DELETE n")
            self._ann_indexed_tables.discard(old_table)
        return group_for(table_name, index_name)

    def _ann_bulk_load(
        self,
        table_name: str,
        entity_rows: npt.NDArray[np.int64],
        matrix: npt.NDArray[np.float32],
    ) -> None:
        """Bulk load one shadow generation, preferring ``COPY FROM`` parquet.

        ``COPY FROM`` is the documented bulk path and measured ~100x faster
        than batched UNWIND at 10k vectors. The data must round-trip through
        a parquet tempfile: ``COPY FROM`` an in-memory Arrow table with a
        fixed-size-list column segfaults the pinned Kuzu 0.11.3. pyarrow is a
        transitive (not guaranteed) dependency, so its absence degrades to
        the batched-UNWIND load rather than failing.
        """
        if importlib.util.find_spec("pyarrow") is None:
            self._ann_insert_rows(table_name, entity_rows, matrix)
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        temp_dir = tempfile.mkdtemp(prefix="zaxy-ann-shadow-")
        try:
            parquet_path = Path(temp_dir) / "shadow.parquet"
            vec_column = pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.reshape(-1), type=pa.float32()),
                matrix.shape[1],
            )
            # no-untyped-call fires only where pyarrow is installed; unused-ignore
            # covers environments (CI) where it is absent and treated as Any.
            pq.write_table(  # type: ignore[no-untyped-call, unused-ignore]
                pa.table({"entity_row": pa.array(entity_rows, type=pa.int64()), "vec": vec_column}),
                str(parquet_path),
            )
            escaped_path = str(parquet_path).replace("\\", "\\\\").replace("'", "\\'")
            self._execute(f"COPY {table_name} FROM '{escaped_path}'")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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

    def _kuzu_vector_index_supported(self) -> bool:
        """Probe once whether the pinned Kuzu runtime ships the vector index."""
        if self._ann_supported is None:
            try:
                rows = self._execute(
                    "CALL SHOW_FUNCTIONS() WHERE name = 'CREATE_VECTOR_INDEX' RETURN name"
                ).get_all()
            except RuntimeError:
                self._ann_supported = False
            else:
                self._ann_supported = bool(rows)
        return self._ann_supported

    async def re_embed_session(
        self,
        *,
        session_id: str,
        provider: Any,
        version_tag: str | None = None,
    ) -> dict[str, int]:
        """Re-embed projected entity vectors onto the provider's version tag.

        This mutates only projection state (Kuzu Entity rows); Eventloom events
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

        This is the single query-execution choke point: Kuzu 0.11.3 segfaults
        — it does not raise — when a query references an unbound
        ``$parameter``, so every ``$placeholder`` must arrive with a binding
        before the statement reaches the runtime.
        """
        placeholders = set(_QUERY_PARAMETER_RE.findall(query))
        missing = placeholders.difference(parameters or ())
        if missing:
            raise RuntimeError(f"query references unbound parameters: {sorted(missing)}")
        connection = self._require_connection()
        if parameters is None:
            return connection.execute(query)
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


def _adjacency_signature(
    session_id: str,
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> str:
    """Hash the snapshot content so the signature changes exactly with the graph.

    Nodes and directed edges are hashed in their (already deterministic)
    sorted order with type/field separators so distinct graphs cannot collide
    by concatenation. Content hashing is the embedded analogue of the
    log-signature pattern: cached walk results keyed on this signature stay
    valid until the projected graph itself changes.
    """
    hasher = hashlib.sha256()
    hasher.update(session_id.encode("utf-8"))
    for node_id in node_ids:
        hasher.update(b"\x00n")
        hasher.update(node_id.encode("utf-8"))
    for source_key, target_key in edges:
        hasher.update(b"\x00e")
        hasher.update(source_key.encode("utf-8"))
        hasher.update(b"\x00>")
        hasher.update(target_key.encode("utf-8"))
    return f"adjacency:sha256:{hasher.hexdigest()}"


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


def _causal_edge_metadata_from_row(
    row: Any,
    *,
    source_entity: GraphEntity,
    target_entity: GraphEntity,
) -> dict[str, Any] | None:
    evidence = _json_dict(row[25])
    source_event_seq = row[23]
    source_event_hash = str(row[24] or "")
    relation_type = str(row[10])
    confidence = _optional_float(row[21])
    cited_seq = _optional_int(row[23])
    if confidence is None or cited_seq is None or not source_event_hash or not evidence:
        return None
    citation = _edge_citation(source_entity.session_id, source_event_seq, source_event_hash)
    return {
        "causal_source_name": source_entity.name,
        "causal_source_type": source_entity.entity_type,
        "causal_target_name": target_entity.name,
        "causal_target_type": target_entity.entity_type,
        "relation_type": relation_type,
        "graph_relation_type": relation_type,
        "causal_relation_type": evidence.get("causal_relation_type") or relation_type.removeprefix("causal_"),
        "confidence": confidence,
        "inference_method": str(row[22] or "unknown"),
        "citation": citation,
        "review_status": evidence.get("review_status") or "proposed",
        "authority_status": evidence.get("authority_status") or "non_authoritative",
        "source_event_seq": cited_seq,
        "source_event_hash": source_event_hash or None,
        "evidence": evidence,
        "session_id": source_entity.session_id,
    }


def _entity_with_causal_metadata(
    entity: GraphEntity,
    *,
    edge_metadata: dict[str, Any],
    path_relation_types: list[str],
    path_citations: list[str],
) -> GraphEntity:
    from zaxy.graph import GraphEntity

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


def _edge_citation(session_id: str, source_event_seq: Any, source_event_hash: str) -> str:
    if source_event_seq is not None and source_event_hash:
        return f"eventloom://{session_id}/events/{source_event_seq}#{source_event_hash[:12]}"
    return "eventloom://unknown/events/unknown#unknown"


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    return [term for term in _TOKEN_RE.findall(text.casefold()) if len(term) > 1]


def _keyword_query_terms(text: str) -> list[str]:
    return [term for term in _terms(text) if term not in _KEYWORD_STOP_WORDS]


def _keyword_index_from_entities(entities: list[GraphEntity]) -> _KeywordIndex:
    term_counts: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []
    total_document_length = 0
    for entity in entities:
        terms = _terms(_entity_keyword_text(entity))
        counts = Counter(terms)
        term_counts.append(counts)
        length = len(terms)
        document_lengths.append(length)
        total_document_length += length
        document_frequency.update(counts.keys())
    average_length = total_document_length / len(term_counts) if term_counts else 0.0
    return _KeywordIndex(
        entities=entities,
        term_counts=term_counts,
        term_entity_ids=_term_entity_ids(term_counts),
        term_idf=_term_idf(document_frequency, len(term_counts)),
        document_length_norms=_document_length_norms(document_lengths, average_length),
    )


def _keyword_candidate_terms(
    index: _KeywordIndex,
    query_terms: list[str],
    *,
    max_candidates: int = 1000,
    min_terms: int = 4,
) -> list[str]:
    unique_terms = list(dict.fromkeys(query_terms))
    if not unique_terms:
        return []
    if _candidate_union_size_at_most(index, unique_terms, max_candidates):
        return unique_terms

    selected: list[str] = []
    selected_candidates: set[int] = {*()}
    sorted_terms = sorted(
        unique_terms,
        key=lambda term: (
            len(index.term_entity_ids.get(term, ())),
            -index.term_idf.get(term, 0.0),
        ),
    )
    for term in sorted_terms:
        postings = index.term_entity_ids.get(term, ())
        if not postings:
            continue
        remaining_capacity = max_candidates - len(selected_candidates)
        new_candidate_count = _new_candidate_count_until_overflow(
            postings,
            selected_candidates,
            max_new_candidates=remaining_capacity,
        )
        if len(selected) < min_terms or len(selected_candidates) + new_candidate_count <= max_candidates:
            selected.append(term)
            selected_candidates.update(postings)
    selected_lookup = {*selected}
    return [term for term in unique_terms if term in selected_lookup]


def _candidate_union_size_at_most(index: _KeywordIndex, terms: list[str], max_candidates: int) -> bool:
    candidates: set[int] = {*()}
    for term in terms:
        for entity_index in index.term_entity_ids.get(term, ()):
            candidates.add(entity_index)
            if len(candidates) > max_candidates:
                return False
    return True


def _new_candidate_count_until_overflow(
    postings: Sequence[int],
    selected_candidates: set[int],
    *,
    max_new_candidates: int,
) -> int:
    new_candidate_count = 0
    for entity_index in postings:
        if entity_index in selected_candidates:
            continue
        new_candidate_count += 1
        if new_candidate_count > max_new_candidates:
            return new_candidate_count
    return new_candidate_count


def _term_entity_ids(term_counts: list[Counter[str]]) -> dict[str, tuple[int, ...]]:
    postings: dict[str, list[int]] = {}
    for index, counts in enumerate(term_counts):
        for term in counts:
            postings.setdefault(term, []).append(index)
    return {term: tuple(indices) for term, indices in postings.items()}


def _term_idf(document_frequency: Counter[str], document_count: int) -> dict[str, float]:
    if document_count <= 0:
        return {}
    return {
        term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }


def _document_length_norms(
    document_lengths: list[int],
    average_length: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if average_length <= 0:
        return [0.0 for _ in document_lengths]
    return [k1 * (1 - b + b * document_length / average_length) for document_length in document_lengths]


def _bm25_score_from_precomputed(
    query_terms: Sequence[str],
    term_counts: Counter[str],
    *,
    document_length_norm: float,
    term_idf: dict[str, float],
    k1: float = 1.5,
) -> float:
    if document_length_norm <= 0:
        return 0.0
    score = 0.0
    for term in query_terms:
        term_frequency = term_counts[term]
        if term_frequency <= 0:
            continue
        denominator = term_frequency + document_length_norm
        score += term_idf.get(term, 0.0) * (term_frequency * (k1 + 1)) / denominator
    return score


def _ann_scope_digest(session_id: str, version: str) -> str:
    """Identifier-safe digest naming one (session, version) shadow scope."""
    return hashlib.sha256(f"{session_id}\x1f{version}".encode()).hexdigest()[:16]


def _ann_content_digest(
    entity_rows: npt.NDArray[np.int64],
    matrix: npt.NDArray[np.float32],
) -> str:
    """Hash a shadow generation's (entity_row, float32 vector) content.

    Hashed in the exact float32 representation the shadow table stores, so
    the digest of a leading slice of a new corpus proves the resident rows
    (values *and* entity-row mapping) are unchanged before the delta policy
    chooses the incremental insert path over a generation swap.
    """
    hasher = hashlib.sha256()
    hasher.update(np.int64(entity_rows.shape[0]).tobytes())
    hasher.update(np.ascontiguousarray(entity_rows).tobytes())
    hasher.update(np.ascontiguousarray(matrix).tobytes())
    return hasher.hexdigest()


def _quantized_candidate_entity_indexes(
    group: _QuantizedVectorGroup,
    unit_query: npt.NDArray[np.float64],
    *,
    limit: int,
) -> list[int]:
    """Select oversampled candidate entities with int8 dot products."""
    query_max = float(np.abs(unit_query).max())
    if query_max == 0.0:
        return []
    quantized_query = np.clip(np.rint(unit_query * (127.0 / query_max)), -127, 127).astype(np.int8)
    integer_scores = group.matrix.astype(np.int32) @ quantized_query.astype(np.int32)
    approximate_scores = integer_scores.astype(np.float64) * group.scales
    candidate_count = min(approximate_scores.size, limit * VECTOR_SEARCH_OVERSAMPLE)
    if candidate_count <= 0:
        return []
    if candidate_count < approximate_scores.size:
        candidate_rows = np.argpartition(-approximate_scores, candidate_count - 1)[:candidate_count]
    else:
        candidate_rows = np.arange(approximate_scores.size)
    return sorted(group.entity_indexes[int(row)] for row in candidate_rows)


def _exact_rerank_results(
    candidate_entity_indexes: Sequence[int],
    unit_query: npt.NDArray[np.float64],
    *,
    limit: int,
    entities: list[GraphEntity],
) -> list[SearchResult]:
    """Rerank approximate candidates with exact float64 scores.

    Shared by the ANN and int8 selectors: scores read the float64 vectors
    already resident on the indexed entities, so candidate selection never
    decides the final ordering. Candidates arrive in ascending entity order,
    which keeps the stable descending sort's tie behavior aligned with the
    dense path. Results stay ``exact=False`` because the candidate set is
    approximate.

    Candidate embeddings were validated when the vector index admitted them
    into a group, so the hot path converts them in one bulk pass (the
    per-candidate conversion loop measurably dominated high-dimension query
    latency); if a property has been mutated since admission, the rerank
    falls back to per-candidate validation and skips the bad rows.
    """
    from zaxy.graph import SearchResult

    if not candidate_entity_indexes:
        return []
    kept = list(candidate_entity_indexes)
    try:
        matrix = np.asarray(
            [entities[entity_index].properties["embedding"] for entity_index in kept],
            dtype=np.float64,
        )
        if matrix.ndim != 2:
            raise ValueError("candidate embeddings are not uniform vectors")
    except (KeyError, TypeError, ValueError):
        kept = []
        vectors: list[list[float]] = []
        for entity_index in candidate_entity_indexes:
            vector = _embedding_vector(entities[entity_index].properties.get("embedding"))
            if vector is None:
                continue
            kept.append(entity_index)
            vectors.append(vector)
        if not kept:
            return []
        matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    scores = np.zeros(len(kept), dtype=np.float64)
    nonzero_rows = norms > 0.0
    scores[nonzero_rows] = (matrix[nonzero_rows] / norms[nonzero_rows, np.newaxis]) @ unit_query
    positive_rows = np.flatnonzero(scores > 0.0)
    if positive_rows.size == 0:
        return []
    ordered_rows = positive_rows[np.argsort(-scores[positive_rows], kind="stable")]
    return [
        SearchResult(
            entity=entities[kept[int(row)]],
            score=float(scores[row]),
            source="vector",
            raw_score=float(scores[row]),
            exact=False,
        )
        for row in ordered_rows[:limit]
    ]


def _dense_vector_results(
    group: _VectorGroup,
    unit_query: npt.NDArray[np.float64],
    *,
    limit: int,
    entities: list[GraphEntity],
) -> list[SearchResult]:
    """Exact dense-matrix scoring; the pre-versioning behavior, unchanged."""
    from zaxy.graph import SearchResult

    scores = group.matrix @ unit_query
    positive_rows = np.flatnonzero(scores > 0.0)
    if positive_rows.size == 0:
        return []
    # Stable sort keeps first-projected entities ahead on score ties,
    # matching the previous heapq.nlargest behavior.
    ordered_rows = positive_rows[np.argsort(-scores[positive_rows], kind="stable")]
    results: list[SearchResult] = []
    for row in ordered_rows[:limit]:
        score = float(scores[row])
        results.append(
            SearchResult(
                entity=entities[group.entity_indexes[int(row)]],
                score=score,
                source="vector",
                raw_score=score,
                exact=True,
            )
        )
    return results


def _quantize_unit_matrix(
    matrix: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
    """Quantize unit-vector rows to int8 with per-vector scale factors."""
    row_max = np.abs(matrix).max(axis=1)
    # Unit vectors always have a nonzero component; guard anyway so a zero row
    # quantizes to zeros instead of dividing by zero.
    safe_row_max = np.where(row_max == 0.0, 1.0, row_max)
    quantized = np.clip(
        np.rint(matrix * (127.0 / safe_row_max)[:, np.newaxis]),
        -127,
        127,
    ).astype(np.int8)
    scales = row_max / 127.0
    return quantized, scales


def _embedding_version(properties: dict[str, Any]) -> str:
    """Return the stored embedding version tag; absent means legacy."""
    version = properties.get("embedding_version")
    if isinstance(version, str) and version:
        return version
    return LEGACY_EMBEDDING_VERSION


def _embedding_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    vector = []
    for item in value:
        if not isinstance(item, int | float):
            return None
        vector.append(float(item))
    return vector


def _json_dict(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _properties_reference_source(properties: dict[str, Any], source_path: str) -> bool:
    for key in ("source_path", "target_path", "test_path", "covered_path"):
        if properties.get(key) == source_path:
            return True
    return False
