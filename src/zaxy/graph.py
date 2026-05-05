"""Graph store: Neo4j wrapper with bi-temporal support.

This module provides a thin, testable abstraction over Neo4j for upserting
entities and edges with validity windows, plus hybrid retrieval (vector +
keyword + traversal + temporal filters).

We use the official neo4j driver directly rather than Graphiti's higher-level
API so that Zaxy controls the exact bi-temporal schema and extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from zaxy.extract import ExtractionResult


@dataclass(frozen=True)
class GraphEntity:
    """Entity as stored in Neo4j."""

    name: str
    entity_type: str
    valid_from: str
    valid_to: str | None
    properties: dict[str, Any]


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


class GraphStore:
    """Async Neo4j wrapper for bi-temporal knowledge graph operations.

    Args:
        uri: Bolt URI, e.g. 'bolt://localhost:7687'.
        user: Neo4j username.
        password: Neo4j password.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._auth = (user, password)
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize the async driver."""
        self._driver = AsyncGraphDatabase.driver(self._uri, auth=self._auth)

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

        # Unique constraint on (Entity {name, entity_type})
        await self._driver.execute_query(
            "CREATE CONSTRAINT entity_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.entity_type) IS UNIQUE"
        )

        # Vector index for semantic search on entity summaries
        await self._driver.execute_query(
            "CREATE VECTOR INDEX entity_vector IF NOT EXISTS "
            "FOR (e:Entity) ON (e.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
        )

        # Full-text index for BM25 keyword search
        await self._driver.execute_query(
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.summary]"
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def upsert_extraction(self, result: ExtractionResult) -> None:
        """Project an ExtractionResult into the graph.

        Entities are merged by (name, type). Edges are merged by
        (source, target, type, valid_from) so that re-ingestion is idempotent.
        """
        assert self._driver is not None

        for ent in result.entities:
            await self._driver.execute_query(
                """
                MERGE (e:Entity {name: $name, entity_type: $entity_type})
                ON CREATE SET e.created_at = datetime($observed_at)
                ON MATCH SET e.updated_at = datetime($observed_at)
                SET e.valid_from = datetime($observed_at),
                    e.valid_to = null
                """,
                name=ent.name,
                entity_type=ent.entity_type,
                observed_at=ent.observed_at,
            )

        for edge in result.edges:
            await self._driver.execute_query(
                """
                MATCH (s:Entity {name: $source})
                MATCH (t:Entity {name: $target})
                MERGE (s)-[r:RELATES {relation_type: $relation_type, valid_from: datetime($valid_from)}]->(t)
                ON CREATE SET r.created_at = datetime($valid_from)
                SET r.valid_to = null
                """,
                source=edge.source,
                target=edge.target,
                relation_type=edge.relation_type,
                valid_from=edge.valid_from,
            )

    async def invalidate_entity(self, name: str, entity_type: str, invalid_at: str) -> None:
        """Mark an entity as invalid after a given time (bi-temporal update)."""
        assert self._driver is not None
        await self._driver.execute_query(
            """
            MATCH (e:Entity {name: $name, entity_type: $entity_type})
            WHERE e.valid_to IS NULL
            SET e.valid_to = datetime($invalid_at)
            """,
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
    ) -> None:
        """Mark an edge as invalid after a given time."""
        assert self._driver is not None
        await self._driver.execute_query(
            """
            MATCH (s:Entity {name: $source})-[r:RELATES {relation_type: $relation_type, valid_from: datetime($valid_from)}]->(t:Entity {name: $target})
            WHERE r.valid_to IS NULL
            SET r.valid_to = datetime($invalid_at)
            """,
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
    ) -> list[GraphEntity]:
        """Exact-match lookup by name, optionally filtered by type and time."""
        assert self._driver is not None

        cypher = "MATCH (e:Entity {name: $name})"
        params: dict[str, Any] = {"name": name}
        where_clauses: list[str] = []

        if entity_type:
            where_clauses.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type

        if temporal_point:
            where_clauses.append(
                "e.valid_from <= datetime($t) AND (e.valid_to IS NULL OR e.valid_to > datetime($t))"
            )
            params["t"] = temporal_point

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
    ) -> list[GraphEntity]:
        """Graph traversal from a starting entity, optionally filtered."""
        assert self._driver is not None

        rel_filter = "{relation_type: $relation_type}" if relation_type else ""
        params: dict[str, Any] = {"start_name": start_name, "depth": depth}

        if relation_type:
            params["relation_type"] = relation_type

        temporal_clause = ""
        if temporal_point:
            temporal_clause = (
                "AND (r.valid_from <= datetime($t) AND (r.valid_to IS NULL OR r.valid_to > datetime($t)))"
            )
            params["t"] = temporal_point

        cypher = f"""
        MATCH path = (start:Entity {{name: $start_name}})-[r:RELATES{rel_filter}*1..$depth]->(neighbor:Entity)
        WHERE ALL(rel IN relationships(path) WHERE rel.valid_to IS NULL {temporal_clause.replace('AND ', '')})
        RETURN DISTINCT neighbor
        """

        records, _, _ = await self._driver.execute_query(cypher, **params)
        return [_record_to_entity(r["neighbor"]) for r in records]

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
    ) -> list[SearchResult]:
        """BM25 full-text search over entity names and summaries."""
        assert self._driver is not None

        cypher = "CALL db.index.fulltext.queryNodes('entity_fulltext', $query) YIELD node, score"
        params: dict[str, Any] = {"query": query, "limit": limit}

        if temporal_point:
            cypher += (
                " WHERE (node.valid_from <= datetime($t) AND (node.valid_to IS NULL OR node.valid_to > datetime($t)))"
            )
            params["t"] = temporal_point

        cypher += " RETURN node, score LIMIT $limit"

        records, _, _ = await self._driver.execute_query(cypher, **params)
        return [
            SearchResult(entity=_record_to_entity(r["node"]), score=r["score"], source="keyword")
            for r in records
        ]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _record_to_entity(node: Any) -> GraphEntity:
    """Convert a Neo4j node record to a GraphEntity."""
    props = dict(node)
    return GraphEntity(
        name=props.get("name", ""),
        entity_type=props.get("entity_type", ""),
        valid_from=str(props.get("valid_from", "")),
        valid_to=str(props.get("valid_to")) if props.get("valid_to") else None,
        properties={k: v for k, v in props.items() if k not in {"name", "entity_type", "valid_from", "valid_to"}},
    )
