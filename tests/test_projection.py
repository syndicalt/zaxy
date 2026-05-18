"""Tests for projection-store backend contracts."""

from __future__ import annotations

from typing import assert_type

from zaxy.extract import ExtractionResult
from zaxy.graph import (
    GraphEntity,
    GraphEventProjectionStatus,
    GraphInferredEdgeStatus,
    SearchResult,
)
from zaxy.projection import ProjectionStore


class FakeProjectionStore:
    """Small structural implementation used to pin the projection contract."""

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def init_schema(self) -> None:
        pass

    async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None:
        pass

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        pass

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        return []

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        return []

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        return []

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        return []

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        raise NotImplementedError

    async def inspect_inferred_edge_status(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> GraphInferredEdgeStatus:
        raise NotImplementedError

def test_fake_projection_store_satisfies_contract() -> None:
    store: ProjectionStore = FakeProjectionStore()
    assert store is not None


async def _type_probe(store: ProjectionStore) -> None:
    exact = await store.search_exact("Alice")
    keyword = await store.search_keyword("ship memory")
    traversal = await store.search_traversal("Alice")
    vector = await store.search_vector([0.1, 0.2])

    assert_type(exact, list[GraphEntity])
    assert_type(keyword, list[SearchResult])
    assert_type(traversal, list[GraphEntity])
    assert_type(vector, list[SearchResult])
