"""Graph store: Neo4j wrapper with bi-temporal support.

This module provides a thin, testable abstraction over Neo4j for upserting
entities and edges with validity windows, plus hybrid retrieval (vector +
keyword + traversal + temporal filters).

We use the official neo4j driver directly rather than Graphiti's higher-level
API so that Zaxy controls the exact bi-temporal schema and extraction pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, TrustAll, TrustCustomCAs

from zaxy.extract import ExtractionResult
from zaxy.schema import apply_schema_migrations
from zaxy.security import validate_limit, validate_session_id, validate_traversal_depth

_Neo4jPropertyValue = str | int | float | bool | list[str] | list[int] | list[float] | list[bool]
_RELATION_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class GraphEntity:
    """Entity as stored in Neo4j."""

    name: str
    entity_type: str
    valid_from: str
    valid_to: str | None
    properties: dict[str, Any]
    session_id: str = "default"


@dataclass(frozen=True)
class GraphEdge:
    """Edge as stored in Neo4j."""

    source: str
    target: str
    relation_type: str
    valid_from: str
    valid_to: str | None
    properties: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """A single result from hybrid search."""

    entity: GraphEntity
    score: float
    source: str  # 'vector', 'keyword', 'traversal', 'exact'
    raw_score: float | None = None
    source_weight: float | None = None
    ranking_score: float | None = None
    matched_query: str | None = None
    query_weight: float | None = None
    temporal_score: float | None = None
    temporal_weight: float | None = None
    scoring_profile: str | None = None
    reranker: str | None = None
    rerank_score: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEventProjectionStatus:
    """Integrity status for one session's Eventloom graph projection."""

    session_id: str
    event_count: int
    latest_seq: int | None
    latest_hash: str | None
    eventloom_latest_seq: int | None
    eventloom_latest_hash: str | None
    projection_lag: int | None
    latest_hash_matches: bool
    next_event_edges: int
    previous_event_edges: int
    missing_chain_links: int
    integrity_ok: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "session_id": self.session_id,
            "event_count": self.event_count,
            "latest_seq": self.latest_seq,
            "latest_hash": self.latest_hash,
            "eventloom_latest_seq": self.eventloom_latest_seq,
            "eventloom_latest_hash": self.eventloom_latest_hash,
            "projection_lag": self.projection_lag,
            "latest_hash_matches": self.latest_hash_matches,
            "next_event_edges": self.next_event_edges,
            "previous_event_edges": self.previous_event_edges,
            "missing_chain_links": self.missing_chain_links,
            "integrity_ok": self.integrity_ok,
        }


class GraphStore:
    """Async Neo4j wrapper for bi-temporal knowledge graph operations.

    Args:
        uri: Bolt URI, e.g. 'bolt://localhost:7687'.
        user: Neo4j username.
        password: Neo4j password.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        ca_cert: str | None = None,
        trust_all: bool = False,
    ) -> None:
        self._uri = uri
        self._auth = (user, password)
        self._ca_cert = ca_cert
        self._trust_all = trust_all
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize the async driver with optional TLS."""
        kwargs: dict[str, Any] = {"auth": self._auth}
        if self._trust_all:
            kwargs["encrypted"] = True
            kwargs["trusted_certificates"] = TrustAll()
        elif self._ca_cert:
            kwargs["encrypted"] = True
            kwargs["trusted_certificates"] = TrustCustomCAs(self._ca_cert)
        self._driver = AsyncGraphDatabase.driver(self._uri, **kwargs)

    async def close(self) -> None:
        """Close the driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def init_schema(self) -> None:
        """Create constraints and indexes idempotently.

        Must be called once before ingestion.
        """
        assert self._driver is not None, "Call connect() first"
        await apply_schema_migrations(self._driver)

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        """Inspect Event nodes and hash-chain edges for one projected session."""
        assert self._driver is not None
        safe_session_id = validate_session_id(session_id)
        records, _, _ = await self._driver.execute_query(
            """
            MATCH (e:Event {session_id: $session_id})
            WITH count(e) AS event_count, max(e.seq) AS latest_seq
            OPTIONAL MATCH (latest:Event {session_id: $session_id, seq: latest_seq})
            WITH event_count, latest_seq, latest.hash AS latest_hash
            CALL () {
                MATCH (:Event {session_id: $session_id})-[n:NEXT_EVENT]->(:Event {session_id: $session_id})
                RETURN count(n) AS next_event_edges
            }
            CALL () {
                MATCH (:Event {session_id: $session_id})-[p:PREVIOUS_EVENT]->(:Event {session_id: $session_id})
                RETURN count(p) AS previous_event_edges
            }
            CALL () {
                MATCH (e:Event {session_id: $session_id})
                WHERE e.seq > 1
                OPTIONAL MATCH (prev:Event {session_id: $session_id, hash: e.prev_hash})
                OPTIONAL MATCH (prev)-[n:NEXT_EVENT]->(e)
                OPTIONAL MATCH (e)-[p:PREVIOUS_EVENT]->(prev)
                WITH e, prev, n, p
                WHERE prev IS NULL OR n IS NULL OR p IS NULL
                RETURN count(e) AS missing_chain_links
            }
            RETURN event_count,
                   latest_seq,
                   latest_hash,
                   next_event_edges,
                   previous_event_edges,
                   missing_chain_links
            """,
            session_id=safe_session_id,
        )
        record = records[0] if records else {}
        event_count = _int_record_value(record.get("event_count"))
        latest_seq = _optional_int_record_value(record.get("latest_seq"))
        latest_hash = _optional_str_record_value(record.get("latest_hash"))
        next_event_edges = _int_record_value(record.get("next_event_edges"))
        previous_event_edges = _int_record_value(record.get("previous_event_edges"))
        missing_chain_links = _int_record_value(record.get("missing_chain_links"))
        projection_lag = (
            max(0, eventloom_latest_seq - (latest_seq or 0))
            if eventloom_latest_seq is not None
            else None
        )
        latest_hash_matches = (
            latest_hash == eventloom_latest_hash
            if eventloom_latest_hash is not None
            else True
        )
        integrity_ok = (
            missing_chain_links == 0
            and latest_hash_matches
            and (projection_lag in (None, 0))
            and next_event_edges == max(0, event_count - 1)
            and previous_event_edges == max(0, event_count - 1)
        )
        return GraphEventProjectionStatus(
            session_id=safe_session_id,
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

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def upsert_extraction(
        self,
        result: ExtractionResult,
        session_id: str = "default",
    ) -> None:
        """Project an ExtractionResult into the graph.

        Entities are versioned by (name, type, valid_from). Reasserting an
        entity closes the version that was current at that time and preserves
        the historical node. Edges are merged by (source, target, type,
        valid_from) so that re-ingestion is idempotent.
        """
        assert self._driver is not None
        safe_session_id = validate_session_id(session_id)
        observed_at = _extraction_observed_at(result)

        await self._driver.execute_query(
            """
            MERGE (s:Session {id: $session_id})
            ON CREATE SET s.created_at = datetime()
            SET s.updated_at = datetime()
            MERGE (ev:Event {session_id: $session_id, seq: $source_event_seq})
            ON CREATE SET ev.created_at = datetime()
            SET ev.updated_at = datetime(),
                ev.hash = $source_event_hash,
                ev.prev_hash = $source_event_prev_hash,
                ev.type = $source_event_type,
                ev.thread = $source_thread,
                ev.observed_at = CASE
                    WHEN $observed_at IS NULL THEN ev.observed_at
                    ELSE datetime($observed_at)
                END
            MERGE (s)-[r:HAS_EVENT]->(ev)
            ON CREATE SET r.created_at = datetime()
            SET r.updated_at = datetime()
            WITH ev
            CALL (ev) {
                MATCH (prev:Event {session_id: $session_id, hash: $source_event_prev_hash})
                MERGE (prev)-[next:NEXT_EVENT]->(ev)
                ON CREATE SET next.created_at = datetime()
                SET next.updated_at = datetime(),
                    next.session_id = $session_id,
                    next.from_hash = $source_event_prev_hash,
                    next.to_hash = $source_event_hash
                MERGE (ev)-[previous:PREVIOUS_EVENT]->(prev)
                ON CREATE SET previous.created_at = datetime()
                SET previous.updated_at = datetime(),
                    previous.session_id = $session_id,
                    previous.from_hash = $source_event_hash,
                    previous.to_hash = $source_event_prev_hash
                RETURN count(prev) AS previous_event_links
            }
            WITH ev, previous_event_links
            CALL (ev) {
                MATCH (next_event:Event {session_id: $session_id, prev_hash: $source_event_hash})
                MERGE (ev)-[next_rel:NEXT_EVENT]->(next_event)
                ON CREATE SET next_rel.created_at = datetime()
                SET next_rel.updated_at = datetime(),
                    next_rel.session_id = $session_id,
                    next_rel.from_hash = $source_event_hash,
                    next_rel.to_hash = next_event.hash
                MERGE (next_event)-[previous_rel:PREVIOUS_EVENT]->(ev)
                ON CREATE SET previous_rel.created_at = datetime()
                SET previous_rel.updated_at = datetime(),
                    previous_rel.session_id = $session_id,
                    previous_rel.from_hash = next_event.hash,
                    previous_rel.to_hash = $source_event_hash
                RETURN count(next_event) AS next_event_links
            }
            RETURN previous_event_links, next_event_links
            """,
            session_id=safe_session_id,
            source_event_seq=result.source_event_seq,
            source_event_hash=result.source_event_hash,
            source_event_prev_hash=result.source_event_prev_hash,
            source_event_type=result.source_event_type,
            source_thread=result.source_thread,
            observed_at=observed_at,
        )

        for ent in result.entities:
            await self._driver.execute_query(
                """
                MERGE (e:Entity {
                    session_id: $session_id,
                    name: $name,
                    entity_type: $entity_type,
                    valid_from: datetime($observed_at)
                })
                ON CREATE SET e.created_at = datetime($observed_at)
                SET e.updated_at = datetime($observed_at),
                    e.summary = coalesce($summary, e.summary),
                    e.embedding = coalesce($embedding, e.embedding),
                    e.source_event_seq = $source_event_seq,
                    e.source_event_hash = $source_event_hash,
                    e.source_event_type = $source_event_type,
                    e.source_thread = $source_thread
                SET e += $properties
                WITH e
                OPTIONAL MATCH (prev:Entity {name: $name, entity_type: $entity_type})
                WHERE prev.session_id = $session_id
                  AND prev.valid_from < e.valid_from
                  AND (prev.valid_to IS NULL OR prev.valid_to > e.valid_from)
                SET prev.valid_to = e.valid_from,
                    prev.updated_at = datetime($observed_at)
                WITH e
                OPTIONAL MATCH (next:Entity {name: $name, entity_type: $entity_type})
                WHERE next.session_id = $session_id
                  AND next.valid_from > e.valid_from
                WITH e, min(next.valid_from) AS next_valid_from
                SET e.valid_to = next_valid_from
                WITH e
                CALL (e) {
                    OPTIONAL MATCH (prev:Entity {session_id: $session_id, name: $name, entity_type: $entity_type})
                    WHERE prev.valid_to = e.valid_from
                    OPTIONAL MATCH (source:Entity)-[old_in:RELATES]->(prev)
                    WHERE old_in.session_id = $session_id
                      AND old_in.valid_to IS NULL
                    WITH e, source, old_in
                    WHERE old_in IS NOT NULL
                    MERGE (source)-[new_in:RELATES {
                        relation_type: old_in.relation_type,
                        valid_from: old_in.valid_from
                    }]->(e)
                    ON CREATE SET new_in.created_at = datetime($observed_at)
                    SET new_in.session_id = $session_id,
                        new_in.valid_to = null,
                        new_in.source_event_seq = old_in.source_event_seq,
                        new_in.source_event_hash = old_in.source_event_hash,
                        new_in.source_event_type = old_in.source_event_type,
                        new_in.source_thread = old_in.source_thread
                    RETURN count(new_in) AS copied_incoming_relationships
                }
                WITH e
                CALL (e) {
                    OPTIONAL MATCH (prev:Entity {session_id: $session_id, name: $name, entity_type: $entity_type})
                    WHERE prev.valid_to = e.valid_from
                    OPTIONAL MATCH (prev)-[old_out:RELATES]->(target:Entity)
                    WHERE old_out.session_id = $session_id
                      AND old_out.valid_to IS NULL
                    WITH e, target, old_out
                    WHERE old_out IS NOT NULL
                    MERGE (e)-[new_out:RELATES {
                        relation_type: old_out.relation_type,
                        valid_from: old_out.valid_from
                    }]->(target)
                    ON CREATE SET new_out.created_at = datetime($observed_at)
                    SET new_out.session_id = $session_id,
                        new_out.valid_to = null,
                        new_out.source_event_seq = old_out.source_event_seq,
                        new_out.source_event_hash = old_out.source_event_hash,
                        new_out.source_event_type = old_out.source_event_type,
                        new_out.source_thread = old_out.source_thread
                    RETURN count(new_out) AS copied_outgoing_relationships
                }
                WITH e
                MATCH (ev:Event {session_id: $session_id, seq: $source_event_seq})
                MERGE (ev)-[pe:PROJECTED_ENTITY {
                    name: $name,
                    entity_type: $entity_type,
                    valid_from: datetime($observed_at)
                }]->(e)
                ON CREATE SET pe.created_at = datetime($observed_at)
                SET pe.source_event_hash = $source_event_hash,
                    pe.source_event_type = $source_event_type
                """,
                session_id=safe_session_id,
                name=ent.name,
                entity_type=ent.entity_type,
                observed_at=ent.observed_at,
                summary=ent.summary,
                embedding=ent.embedding,
                source_event_seq=result.source_event_seq,
                source_event_hash=result.source_event_hash,
                source_event_type=result.source_event_type,
                source_thread=result.source_thread,
                properties=_neo4j_properties(ent.properties),
            )

        for edge in result.edges:
            typed_relationship_label = _typed_relationship_label(edge.relation_type)
            await self._driver.execute_query(
                f"""
                MATCH (s:Entity {{name: $source}})
                WHERE s.session_id = $session_id
                  AND s.valid_from <= datetime($valid_from)
                  AND (s.valid_to IS NULL OR s.valid_to > datetime($valid_from))
                MATCH (t:Entity {{name: $target}})
                WHERE t.session_id = $session_id
                  AND t.valid_from <= datetime($valid_from)
                  AND (t.valid_to IS NULL OR t.valid_to > datetime($valid_from))
                MERGE (s)-[r:RELATES {{relation_type: $relation_type, valid_from: datetime($valid_from)}}]->(t)
                ON CREATE SET r.created_at = datetime($valid_from)
                SET r.session_id = $session_id,
                    r.valid_to = null,
                    r.source_event_seq = $source_event_seq,
                    r.source_event_hash = $source_event_hash,
                    r.source_event_type = $source_event_type,
                    r.source_thread = $source_thread
                MERGE (s)-[typed:{typed_relationship_label} {{valid_from: datetime($valid_from)}}]->(t)
                ON CREATE SET typed.created_at = datetime($valid_from)
                SET typed.session_id = $session_id,
                    typed.relation_type = $relation_type,
                    typed.valid_to = null,
                    typed.source_event_seq = $source_event_seq,
                    typed.source_event_hash = $source_event_hash,
                    typed.source_event_type = $source_event_type,
                    typed.source_thread = $source_thread
                WITH s, t, r, typed
                MATCH (ev:Event {{session_id: $session_id, seq: $source_event_seq}})
                MERGE (ev)-[pr:PROJECTED_RELATION {{
                    source: $source,
                    target: $target,
                    relation_type: $relation_type,
                    valid_from: datetime($valid_from)
                }}]->(t)
                ON CREATE SET pr.created_at = datetime($valid_from)
                SET pr.source_event_hash = $source_event_hash,
                    pr.source_event_type = $source_event_type
                """,
                session_id=safe_session_id,
                source=edge.source,
                target=edge.target,
                relation_type=edge.relation_type,
                valid_from=edge.valid_from,
                source_event_seq=result.source_event_seq,
                source_event_hash=result.source_event_hash,
                source_event_type=result.source_event_type,
                source_thread=result.source_thread,
            )

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Mark an entity as invalid after a given time (bi-temporal update)."""
        assert self._driver is not None
        safe_session_id = validate_session_id(session_id)
        await self._driver.execute_query(
            """
            MATCH (e:Entity {name: $name, entity_type: $entity_type})
            WHERE e.session_id = $session_id
              AND e.valid_to IS NULL
            SET e.valid_to = datetime($invalid_at)
            """,
            session_id=safe_session_id,
            name=name,
            entity_type=entity_type,
            invalid_at=invalid_at,
        )

    async def invalidate_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        valid_from: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Mark an edge as invalid after a given time."""
        assert self._driver is not None
        safe_session_id = validate_session_id(session_id)
        await self._driver.execute_query(
            """
            MATCH (s:Entity {name: $source})-[r:RELATES {relation_type: $relation_type, valid_from: datetime($valid_from)}]->(t:Entity {name: $target})
            WHERE s.session_id = $session_id
              AND t.session_id = $session_id
              AND r.session_id = $session_id
              AND r.valid_to IS NULL
            SET r.valid_to = datetime($invalid_at)
            """,
            session_id=safe_session_id,
            source=source,
            target=target,
            relation_type=relation_type,
            valid_from=valid_from,
            invalid_at=invalid_at,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Exact-match lookup by name, optionally filtered by type and time."""
        assert self._driver is not None

        cypher = "MATCH (e:Entity {name: $name})"
        params: dict[str, Any] = {
            "name": name,
            "session_id": validate_session_id(session_id),
        }
        where_clauses: list[str] = ["e.session_id = $session_id"]

        if entity_type:
            where_clauses.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type

        if temporal_point:
            where_clauses.append(
                "e.valid_from <= datetime($t) AND (e.valid_to IS NULL OR e.valid_to > datetime($t))"
            )
            params["t"] = temporal_point
        else:
            where_clauses.append("e.valid_to IS NULL")

        if where_clauses:
            cypher += " WHERE " + " AND ".join(where_clauses)

        cypher += " RETURN e"

        records, _, _ = await self._driver.execute_query(cypher, **params)
        return [_record_to_entity(r["e"]) for r in records]

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Graph traversal from a starting entity, optionally filtered."""
        assert self._driver is not None
        depth = validate_traversal_depth(depth)

        params: dict[str, Any] = {
            "start_name": start_name,
            "session_id": validate_session_id(session_id),
            "relation_type": relation_type,
        }

        temporal_checks = " AND rel.valid_to IS NULL AND ($relation_type IS NULL OR rel.relation_type = $relation_type)"
        entity_checks = (
            "start.session_id = $session_id"
            " AND neighbor.session_id = $session_id"
            " AND start.valid_to IS NULL"
            " AND neighbor.valid_to IS NULL"
        )
        if temporal_point:
            temporal_checks = (
                " AND rel.valid_from <= datetime($t)"
                " AND (rel.valid_to IS NULL OR rel.valid_to > datetime($t))"
                " AND ($relation_type IS NULL OR rel.relation_type = $relation_type)"
            )
            entity_checks = (
                "start.session_id = $session_id"
                " AND neighbor.session_id = $session_id"
                " AND start.valid_from <= datetime($t)"
                " AND (start.valid_to IS NULL OR start.valid_to > datetime($t))"
                " AND neighbor.valid_from <= datetime($t)"
                " AND (neighbor.valid_to IS NULL OR neighbor.valid_to > datetime($t))"
            )
            params["t"] = temporal_point

        cypher = f"""
        MATCH path = (start:Entity {{name: $start_name}})-[r:RELATES*1..{depth}]-(neighbor:Entity)
        WHERE {entity_checks}
          AND neighbor <> start
          AND ALL(rel IN relationships(path) WHERE true{temporal_checks})
        RETURN neighbor,
               [rel IN relationships(path) | rel.relation_type] AS path_relation_types,
               length(path) AS path_length
        ORDER BY path_length ASC
        """

        records, _, _ = await self._driver.execute_query(cypher, **params)
        entities: list[GraphEntity] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            entity = _record_to_entity(record["neighbor"])
            key = (entity.name, entity.entity_type)
            if key in seen:
                continue
            seen.add(key)
            properties = {
                **entity.properties,
                "_path_relation_types": list(record.get("path_relation_types") or []),
                "_path_length": int(record.get("path_length") or 0),
            }
            entities.append(
                GraphEntity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    valid_from=entity.valid_from,
                    valid_to=entity.valid_to,
                    properties=properties,
                    session_id=entity.session_id,
                )
            )
        return entities

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """BM25 full-text search over entity names and summaries."""
        assert self._driver is not None
        limit = validate_limit(limit)

        cypher = "CALL db.index.fulltext.queryNodes('entity_fulltext', $query) YIELD node, score"
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "session_id": validate_session_id(session_id),
        }

        if temporal_point:
            cypher += (
                " WHERE node.session_id = $session_id"
                " AND (node.valid_from <= datetime($t) AND (node.valid_to IS NULL OR node.valid_to > datetime($t)))"
            )
            params["t"] = temporal_point
        else:
            cypher += " WHERE node.session_id = $session_id AND node.valid_to IS NULL"

        cypher += " RETURN node, score LIMIT $limit"

        records, _, _ = await self._driver.execute_query(cypher, **params)
        return [
            SearchResult(entity=_record_to_entity(r["node"]), score=r["score"], source="keyword")
            for r in records
        ]

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Vector similarity search over entity embeddings."""
        assert self._driver is not None
        limit = validate_limit(limit)

        cypher = """
        CALL db.index.vector.queryNodes('entity_vector', $limit, $embedding)
        YIELD node, score
        """
        params: dict[str, Any] = {
            "embedding": embedding,
            "limit": limit,
            "session_id": validate_session_id(session_id),
        }

        if temporal_point:
            cypher += (
                " WHERE node.session_id = $session_id"
                " AND (node.valid_from <= datetime($t) AND (node.valid_to IS NULL OR node.valid_to > datetime($t)))"
            )
            params["t"] = temporal_point
        else:
            cypher += " WHERE node.session_id = $session_id AND node.valid_to IS NULL"

        cypher += " RETURN node, score"

        records, _, _ = await self._driver.execute_query(cypher, **params)
        return [
            SearchResult(entity=_record_to_entity(r["node"]), score=r["score"], source="vector")
            for r in records
        ]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_ENTITY_STORAGE_PROPERTIES = frozenset(
    {
        "created_at",
        "embedding",
        "entity_type",
        "name",
        "session_id",
        "source_event_hash",
        "source_event_seq",
        "source_event_type",
        "source_thread",
        "summary",
        "updated_at",
        "valid_from",
        "valid_to",
    }
)


def _record_to_entity(node: Any) -> GraphEntity:
    """Convert a Neo4j node record to a GraphEntity."""
    props = dict(node)
    return GraphEntity(
        name=props.get("name", ""),
        entity_type=props.get("entity_type", ""),
        valid_from=str(props.get("valid_from", "")),
        valid_to=str(props.get("valid_to")) if props.get("valid_to") else None,
        session_id=props.get("session_id", "default"),
        properties={
            k: v
            for k, v in props.items()
            if k not in {"session_id", "name", "entity_type", "valid_from", "valid_to"}
        },
    )


def _int_record_value(value: Any) -> int:
    if isinstance(value, int):
        return value
    return 0


def _optional_int_record_value(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _optional_str_record_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _neo4j_properties(properties: dict[str, Any] | None) -> dict[str, _Neo4jPropertyValue]:
    """Filter extracted properties to values accepted by Neo4j node properties."""
    if not properties:
        return {}
    safe_properties: dict[str, _Neo4jPropertyValue] = {}
    for key, value in properties.items():
        if not _is_neo4j_property_value(value):
            continue
        safe_key = f"payload_{key}" if key in _ENTITY_STORAGE_PROPERTIES else key
        safe_properties[safe_key] = value
    return safe_properties


def _extraction_observed_at(result: ExtractionResult) -> str | None:
    """Return the earliest available event timestamp carried by an extraction."""
    values = [entity.observed_at for entity in result.entities]
    values.extend(edge.valid_from for edge in result.edges)
    return min(values) if values else None


def _typed_relationship_label(relation_type: str) -> str:
    """Convert a safe extractor relation type into a Neo4j relationship label."""
    if not _RELATION_TYPE_RE.fullmatch(relation_type):
        raise ValueError(f"Invalid relation_type for typed relationship label: {relation_type!r}")
    return relation_type.upper()


def _is_neo4j_property_value(value: Any) -> bool:
    """Return whether a value is a Neo4j scalar or scalar list."""
    if value is None:
        return False
    if isinstance(value, str | int | float | bool):
        return True
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str | int | float | bool) for item in value)
