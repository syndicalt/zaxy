"""Projection-store contracts for Eventloom-backed memory indexes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from zaxy.extract import ExtractionResult

if TYPE_CHECKING:
    from zaxy.graph import (
        GraphEntity,
        GraphEventProjectionStatus,
        GraphInferredEdgeStatus,
        SearchResult,
    )


class ProjectionStore(Protocol):
    """Backend contract for projecting Eventloom facts into a queryable index."""

    async def connect(self) -> None:
        """Open backend resources."""
        ...

    async def close(self) -> None:
        """Close backend resources."""
        ...

    async def init_schema(self) -> None:
        """Initialize backend schema and indexes."""
        ...

    async def upsert_extraction(
        self,
        result: ExtractionResult,
        session_id: str = "default",
    ) -> None:
        """Project an extracted event into a queryable memory index."""
        ...

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search by exact entity identity."""
        ...

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by lexical relevance."""
        ...

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        """Search graph neighbors from a starting entity."""
        ...

    async def has_traversal_edges(self, session_id: str = "default") -> bool:
        """Return whether the session has active graph relationship edges."""
        ...

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        ...

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        """Close the validity window for a projected entity."""
        ...

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        """Inspect Eventloom-to-graph projection integrity for one session."""
        ...

    async def inspect_inferred_edge_status(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> GraphInferredEdgeStatus:
        """Inspect inferred-edge audit status for one session."""
        ...
