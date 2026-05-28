"""LatticeDB projection backend candidate.

This adapter is intentionally explicit and optional. LatticeDB is promising for
Zaxy because it combines graph traversal, vector search, BM25 full-text, and
durable streams in one embedded file. Until the dependency and query semantics
pass Zaxy's same-harness gates, this backend remains a candidate rather than a
default.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zaxy.security import vector_has_signal

if TYPE_CHECKING:
    from zaxy.extract import ExtractionResult
    from zaxy.graph import (
        GraphEntity,
        GraphEventProjectionStatus,
        GraphInferredEdgeStatus,
        SearchResult,
    )


class LatticeDBStore:
    """Import-safe shell for the LatticeDB projection backend."""

    def __init__(self, path: Path, *, vector_dimensions: int = 1536) -> None:
        self.path = Path(path)
        self.vector_dimensions = vector_dimensions
        self._database: Any | None = None

    async def connect(self) -> None:
        """Open the local LatticeDB file."""
        if importlib.util.find_spec("latticedb") is None:
            raise RuntimeError('LatticeDB backend requires `pip install "zaxy-memory[latticedb]"`')
        latticedb_module: Any = importlib.import_module("latticedb")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._database = latticedb_module.Database(
            str(self.path),
            create=True,
            enable_vectors=True,
            vector_dimensions=self.vector_dimensions,
        )
        self._database.open()

    async def close(self) -> None:
        """Close the local LatticeDB file."""
        close = getattr(self._database, "close", None)
        if callable(close):
            close()
        self._database = None

    async def init_schema(self) -> None:
        """Initialize schema.

        LatticeDB uses dynamic labels/properties for graph data, so the first
        adapter slice has no DDL. Contract coverage will harden this once the
        package is available in CI.
        """
        self._require_database()

    async def reset_benchmark_projection(self) -> None:
        """Remove and recreate the local LatticeDB projection artifact."""
        await self.close()
        if self.path.exists():
            if self.path.is_dir():
                shutil.rmtree(self.path)
            else:
                self.path.unlink()
        await self.connect()
        await self.init_schema()

    async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None:
        """Project an extracted Eventloom event."""
        database = self._require_database()
        with database.write() as txn:
            txn.create_node(
                labels=["Event"],
                properties={
                    "event_key": _event_key(session_id, result.source_event_seq),
                    "session_id": session_id,
                    "seq": result.source_event_seq,
                    "hash": result.source_event_hash or "",
                    "prev_hash": result.source_event_prev_hash or "",
                    "event_type": result.source_event_type or "",
                    "source_thread": result.source_thread or "",
                },
            )
            entity_types = {entity.name: entity.entity_type for entity in result.entities}
            created_node_ids: dict[tuple[str, str], int] = {}
            for entity in result.entities:
                for node_id in self._active_node_ids(session_id, entity.entity_type, entity.name):
                    txn.set_property(node_id, "valid_to", entity.observed_at)
                properties = dict(entity.properties or {})
                if entity.summary is not None:
                    properties.setdefault("summary", entity.summary)
                node = txn.create_node(
                    labels=["Entity", _label(entity.entity_type)],
                    properties={
                        "node_key": _node_key(session_id, entity.entity_type, entity.name, result.source_event_seq),
                        "session_id": session_id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "summary": entity.summary or "",
                        "properties_json": json.dumps(properties, sort_keys=True),
                        "valid_from": entity.observed_at,
                        "valid_to": "",
                        "source_event_seq": result.source_event_seq,
                        "source_event_hash": result.source_event_hash or "",
                    },
                )
                if entity.embedding is not None:
                    txn.set_vector(node.id, "embedding", _vector(entity.embedding))
                txn.fts_index(node.id, _fts_text(entity.name, entity.entity_type, properties))
                created_node_ids[(entity.entity_type, entity.name)] = int(node.id)
            for edge in result.edges:
                source_type = entity_types.get(edge.source, "entity")
                target_type = entity_types.get(edge.target, "entity")
                source_id = created_node_ids.get((source_type, edge.source)) or self._active_node_id(
                    session_id,
                    source_type,
                    edge.source,
                )
                target_id = created_node_ids.get((target_type, edge.target)) or self._active_node_id(
                    session_id,
                    target_type,
                    edge.target,
                )
                if source_id is None or target_id is None:
                    continue
                evidence = dict(edge.evidence)
                txn.create_edge(
                    source_id,
                    target_id,
                    "RELATES",
                    properties={
                        "session_id": session_id,
                        "relation_type": edge.relation_type,
                        "valid_from": edge.valid_from,
                        "valid_to": edge.valid_to or "",
                        "inferred": edge.inferred,
                        "confidence": edge.confidence if edge.confidence is not None else -1.0,
                        "inference_method": edge.inference_method or "",
                        "source_event_seq": evidence.get("source_event_seq", result.source_event_seq),
                        "source_event_hash": evidence.get("source_event_hash", result.source_event_hash or ""),
                        "evidence_json": json.dumps(evidence, sort_keys=True),
                    },
                )
            txn.commit()

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search by exact identity."""
        matches = [
            self._entity_from_node_id(node_id)
            for node_id in self._entity_node_ids()
            if self._node_property(node_id, "session_id") == session_id
            and self._node_property(node_id, "name") == name
            and (entity_type is None or self._node_property(node_id, "entity_type") == entity_type)
            and self._is_visible_at(node_id, temporal_point)
        ]
        return sorted(matches, key=lambda entity: entity.valid_from or "", reverse=True)

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
        native_matches = self._search_keyword_native(query, limit=limit, temporal_point=temporal_point, session_id=session_id)
        if native_matches:
            return native_matches

        terms = query.casefold().split()
        matches = []
        for node_id in self._entity_node_ids():
            if self._node_property(node_id, "session_id") != session_id or not self._is_visible_at(node_id, temporal_point):
                continue
            entity = self._entity_from_node_id(node_id)
            haystack = f"{entity.name} {entity.entity_type} {entity.properties.get('summary', '')}".casefold()
            score = sum(1 for term in terms if term in haystack)
            if score:
                score_value = float(score)
                matches.append(SearchResult(entity=entity, score=score_value, source="keyword", raw_score=score_value))
        return sorted(matches, key=lambda item: item.score, reverse=True)[:limit]

    def _search_keyword_native(
        self,
        query: str,
        *,
        limit: int,
        temporal_point: str | None,
        session_id: str,
    ) -> list[SearchResult]:
        from zaxy.graph import SearchResult

        database = self._require_database()
        with database.read() as txn:
            matches = list(txn.fts_search(query, limit=limit * 4))
        results: list[SearchResult] = []
        for match in matches:
            node_id = int(match.node_id)
            if self._node_property(node_id, "session_id") != session_id or not self._is_visible_at(node_id, temporal_point):
                continue
            score = float(match.score)
            results.append(
                SearchResult(
                    entity=self._entity_from_node_id(node_id),
                    score=score,
                    source="keyword",
                    raw_score=score,
                )
            )
            if len(results) >= limit:
                break
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
        safe_depth = max(1, min(depth, 5))
        start_ids = [
            node_id
            for node_id in self._entity_node_ids()
            if self._node_property(node_id, "session_id") == session_id
            and self._node_property(node_id, "name") == start_name
            and self._is_visible_at(node_id, temporal_point)
        ]
        frontier = set(start_ids)
        visited = set(start_ids)
        found: dict[int, GraphEntity] = {}
        for _ in range(safe_depth):
            next_frontier: set[int] = set()
            for node_id in frontier:
                for edge in self._outgoing_edges(node_id):
                    target_id = int(edge.target_id)
                    if not self._active_edge(edge.id, session_id, relation_type) or not self._is_visible_at(target_id, temporal_point):
                        continue
                    found.setdefault(target_id, self._entity_from_node_id(target_id))
                    if target_id not in visited:
                        visited.add(target_id)
                        next_frontier.add(target_id)
            frontier = next_frontier
            if not frontier:
                break
        return list(found.values())

    async def has_traversal_edges(self, session_id: str = "default") -> bool:
        """Return whether the session has active graph edges."""
        return any(
            self._active_edge(edge.id, session_id, relation_type=None)
            for node_id in self._entity_node_ids()
            for edge in self._outgoing_edges(node_id)
        )

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        from zaxy.graph import SearchResult

        if limit <= 0:
            return []
        if not vector_has_signal(embedding):
            return []
        database = self._require_database()
        with database.read() as txn:
            matches = list(txn.vector_search(_vector(embedding), k=limit))
        results: list[SearchResult] = []
        for match in matches:
            node_id = int(match.node_id)
            if self._node_property(node_id, "session_id") != session_id or not self._is_visible_at(node_id, temporal_point):
                continue
            distance = float(match.distance)
            score = 1.0 / (1.0 + max(0.0, distance))
            results.append(
                SearchResult(
                    entity=self._entity_from_node_id(node_id),
                    score=score,
                    source="vector",
                    raw_score=distance,
                )
            )
            if len(results) >= limit:
                break
        return results

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Close an entity validity window."""
        database = self._require_database()
        with database.write() as txn:
            for node_id in self._active_node_ids(session_id, entity_type, name):
                txn.set_property(node_id, "valid_to", invalid_at)
            txn.commit()

    async def retire_source_projections(
        self,
        *,
        source_path: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Close active projections derived from one source path."""
        database = self._require_database()
        with database.write() as txn:
            retired = [
                node_id
                for node_id in self._entity_node_ids()
                if self._node_property(node_id, "session_id") == session_id
                and self._node_property(node_id, "valid_to") == ""
                and _properties_reference_source(_json_dict(self._node_property(node_id, "properties_json")), source_path)
            ]
            for node_id in retired:
                txn.set_property(node_id, "valid_to", invalid_at)
                for edge in [*self._outgoing_edges(node_id), *self._incoming_edges(node_id)]:
                    if self._edge_property(edge.id, "session_id") == session_id and self._edge_property(edge.id, "valid_to") == "":
                        txn.set_edge_property(edge.id, "valid_to", invalid_at)
            txn.commit()

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        """Inspect Eventloom-to-graph projection integrity."""
        from zaxy.graph import GraphEventProjectionStatus

        events: list[dict[str, int | str]] = [
            {
                "seq": int(self._node_property(node_id, "seq") or 0),
                "hash": str(self._node_property(node_id, "hash") or ""),
                "prev_hash": str(self._node_property(node_id, "prev_hash") or ""),
            }
            for node_id in self._event_node_ids()
            if self._node_property(node_id, "session_id") == session_id
        ]
        events.sort(key=lambda item: int(item["seq"]))
        latest_seq = int(events[-1]["seq"]) if events else None
        latest_hash = str(events[-1]["hash"]) if events else None
        known_hashes = {str(event["hash"]) for event in events if event["hash"]}
        missing_chain_links = sum(1 for event in events if event["prev_hash"] and event["prev_hash"] not in known_hashes)
        projection_lag = None if eventloom_latest_seq is None or latest_seq is None else eventloom_latest_seq - latest_seq
        latest_hash_matches = eventloom_latest_hash is None or latest_hash == eventloom_latest_hash
        integrity_ok = missing_chain_links == 0 and (projection_lag in (None, 0)) and latest_hash_matches
        return GraphEventProjectionStatus(
            session_id=session_id,
            event_count=len(events),
            latest_seq=latest_seq,
            latest_hash=latest_hash,
            eventloom_latest_seq=eventloom_latest_seq,
            eventloom_latest_hash=eventloom_latest_hash,
            projection_lag=projection_lag,
            latest_hash_matches=latest_hash_matches,
            next_event_edges=0,
            previous_event_edges=0,
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
        from zaxy.graph import (
            GraphInferredEdgeMethodStatus,
            GraphInferredEdgeSample,
            GraphInferredEdgeStatus,
        )

        edges = [
            edge
            for node_id in self._entity_node_ids()
            for edge in self._outgoing_edges(node_id)
            if self._edge_property(edge.id, "session_id") == session_id
            and self._edge_property(edge.id, "valid_to") == ""
            and bool(self._edge_property(edge.id, "inferred"))
        ]
        method_edges: dict[str, list[Any]] = defaultdict(list)
        evidence_count = 0
        missing_source_event_count = 0
        for edge in edges:
            method = str(self._edge_property(edge.id, "inference_method") or "unknown")
            method_edges[method].append(edge)
            evidence = _json_dict(self._edge_property(edge.id, "evidence_json"))
            if evidence:
                evidence_count += 1
            if not self._edge_property(edge.id, "source_event_seq") or not self._edge_property(edge.id, "source_event_hash"):
                missing_source_event_count += 1

        methods = []
        for method, grouped_edges in sorted(method_edges.items()):
            confidences = [
                float(confidence)
                for edge in grouped_edges
                if (confidence := self._edge_property(edge.id, "confidence")) is not None and float(confidence) >= 0.0
            ]
            grouped_evidence_count = sum(1 for edge in grouped_edges if _json_dict(self._edge_property(edge.id, "evidence_json")))
            grouped_missing_source_count = sum(
                1
                for edge in grouped_edges
                if not self._edge_property(edge.id, "source_event_seq") or not self._edge_property(edge.id, "source_event_hash")
            )
            relation_types = tuple(sorted({str(self._edge_property(edge.id, "relation_type") or "") for edge in grouped_edges}))
            methods.append(
                GraphInferredEdgeMethodStatus(
                    method=method,
                    edge_count=len(grouped_edges),
                    relation_types=relation_types,
                    average_confidence=(sum(confidences) / len(confidences)) if confidences else None,
                    minimum_confidence=min(confidences) if confidences else None,
                    evidence_count=grouped_evidence_count,
                    missing_evidence_count=len(grouped_edges) - grouped_evidence_count,
                    missing_source_event_count=grouped_missing_source_count,
                )
            )

        samples = []
        for edge in edges[:limit]:
            evidence = _json_dict(self._edge_property(edge.id, "evidence_json"))
            confidence = self._edge_property(edge.id, "confidence")
            samples.append(
                GraphInferredEdgeSample(
                    source=str(self._node_property(int(edge.source_id), "name") or ""),
                    target=str(self._node_property(int(edge.target_id), "name") or ""),
                    relation_type=str(self._edge_property(edge.id, "relation_type") or ""),
                    confidence=float(confidence) if confidence is not None and float(confidence) >= 0.0 else None,
                    method=str(self._edge_property(edge.id, "inference_method") or "unknown"),
                    source_event_seq=_optional_int(self._edge_property(edge.id, "source_event_seq")),
                    source_event_hash=str(source_hash) if (source_hash := self._edge_property(edge.id, "source_event_hash")) else None,
                    evidence_keys=tuple(sorted(evidence.keys())),
                )
            )

        total_edges = len(edges)
        missing_evidence_count = total_edges - evidence_count
        return GraphInferredEdgeStatus(
            session_id=session_id,
            total_edges=total_edges,
            method_count=len(methods),
            evidence_count=evidence_count,
            missing_evidence_count=missing_evidence_count,
            missing_source_event_count=missing_source_event_count,
            evidence_coverage=(evidence_count / total_edges) if total_edges else 1.0,
            methods=tuple(methods),
            samples=tuple(samples),
        )

    def _require_database(self) -> Any:
        if self._database is None:
            raise RuntimeError("LatticeDB store is not connected")
        return self._database

    def _entity_node_ids(self) -> list[int]:
        database = self._require_database()
        with database.read() as txn:
            return [int(node_id) for node_id in txn.get_nodes_by_label("Entity")]

    def _event_node_ids(self) -> list[int]:
        database = self._require_database()
        with database.read() as txn:
            return [int(node_id) for node_id in txn.get_nodes_by_label("Event")]

    def _node_property(self, node_id: int, key: str) -> Any:
        database = self._require_database()
        with database.read() as txn:
            return txn.get_property(node_id, key)

    def _edge_property(self, edge_id: int, key: str) -> Any:
        database = self._require_database()
        with database.read() as txn:
            return txn.get_edge_property(edge_id, key)

    def _outgoing_edges(self, node_id: int) -> list[Any]:
        database = self._require_database()
        with database.read() as txn:
            return list(txn.get_outgoing_edges_by_type(node_id, "RELATES"))

    def _incoming_edges(self, node_id: int) -> list[Any]:
        database = self._require_database()
        with database.read() as txn:
            return list(txn.get_incoming_edges_by_type(node_id, "RELATES"))

    def _active_node_ids(self, session_id: str, entity_type: str, name: str) -> list[int]:
        return [
            node_id
            for node_id in self._entity_node_ids()
            if self._node_property(node_id, "session_id") == session_id
            and self._node_property(node_id, "entity_type") == entity_type
            and self._node_property(node_id, "name") == name
            and self._node_property(node_id, "valid_to") == ""
        ]

    def _active_node_id(self, session_id: str, entity_type: str, name: str) -> int | None:
        ids = self._active_node_ids(session_id, entity_type, name)
        if not ids:
            return None
        return max(ids, key=lambda node_id: str(self._node_property(node_id, "valid_from") or ""))

    def _is_visible_at(self, node_id: int, temporal_point: str | None) -> bool:
        valid_to = self._node_property(node_id, "valid_to") or ""
        if temporal_point is None:
            return valid_to == ""
        valid_from = str(self._node_property(node_id, "valid_from") or "")
        return valid_from <= temporal_point and (not valid_to or temporal_point < str(valid_to))

    def _active_edge(self, edge_id: int, session_id: str, relation_type: str | None) -> bool:
        return (
            self._edge_property(edge_id, "session_id") == session_id
            and self._edge_property(edge_id, "valid_to") == ""
            and (relation_type is None or self._edge_property(edge_id, "relation_type") == relation_type)
        )

    def _entity_from_node_id(self, node_id: int) -> GraphEntity:
        from zaxy.graph import GraphEntity

        properties = _json_dict(self._node_property(node_id, "properties_json"))
        summary = self._node_property(node_id, "summary")
        if summary:
            properties.setdefault("summary", summary)
        source_event_seq = self._node_property(node_id, "source_event_seq")
        source_event_hash = self._node_property(node_id, "source_event_hash")
        if source_event_seq is not None:
            properties["source_event_seq"] = int(source_event_seq)
        if source_event_hash:
            properties["source_event_hash"] = str(source_event_hash)
        valid_to = self._node_property(node_id, "valid_to") or None
        return GraphEntity(
            name=str(self._node_property(node_id, "name")),
            entity_type=str(self._node_property(node_id, "entity_type")),
            valid_from=str(self._node_property(node_id, "valid_from") or ""),
            valid_to=str(valid_to) if valid_to else None,
            properties=properties,
            session_id=str(self._node_property(node_id, "session_id")),
        )


def _node_key(session_id: str, entity_type: str, name: str, source_event_seq: int) -> str:
    return f"{session_id}\x1f{entity_type}\x1f{name}\x1f{source_event_seq}"


def _event_key(session_id: str, seq: int) -> str:
    return f"{session_id}\x1f{seq}"


def _label(value: str) -> str:
    label = "".join(character if character.isalnum() else "_" for character in value.title())
    return label or "EntityType"


def _json_dict(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(str(raw))
    return parsed if isinstance(parsed, dict) else {}


def _fts_text(name: str, entity_type: str, properties: dict[str, Any]) -> str:
    values = [name, entity_type]
    for key, value in sorted(properties.items()):
        if key == "embedding":
            continue
        values.append(str(value))
    return " ".join(values)


def _vector(values: list[float]) -> Any:
    numpy = importlib.import_module("numpy")
    return numpy.asarray(values, dtype=numpy.float32)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None and value != "" else None


def _properties_reference_source(properties: dict[str, Any], source_path: str) -> bool:
    return any(properties.get(key) == source_path for key in ("source_path", "target_path", "test_path", "covered_path"))
