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
        raise NotImplementedError("pgGraph schema initialization is not implemented yet")

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
