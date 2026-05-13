"""Projection-store contracts for Eventloom-backed memory indexes."""

from __future__ import annotations

from typing import Protocol

from zaxy.extract import ExtractionResult


class ProjectionStore(Protocol):
    """Backend contract for projecting extracted Eventloom facts."""

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
    ) -> object:
        """Search by exact entity identity."""
        ...

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> object:
        """Search by lexical relevance."""
        ...

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> object:
        """Search graph neighbors from a starting entity."""
        ...

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> object:
        """Search by vector similarity."""
        ...
