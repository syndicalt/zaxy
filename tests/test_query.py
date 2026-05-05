"""Tests for zaxy.query — hybrid query router."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from zaxy.graph import GraphEntity, GraphStore, SearchResult
from zaxy.query import QueryRouter


@pytest.fixture
def mock_store() -> AsyncMock:
    """Return a mock GraphStore."""
    store = AsyncMock(spec=GraphStore)
    store.search_exact = AsyncMock(return_value=[])
    store.search_keyword = AsyncMock(return_value=[])
    store.search_traversal = AsyncMock(return_value=[])
    return store


@pytest.fixture
def router(mock_store: AsyncMock) -> QueryRouter:
    """Return a QueryRouter wired to the mock store."""
    return QueryRouter(store=mock_store, default_limit=5)


# ------------------------------------------------------------------
# Query routing tests
# ------------------------------------------------------------------

class TestQueryRouting:
    """Tests for the hybrid query pipeline."""

    async def test_empty_store_returns_empty(self, router: QueryRouter) -> None:
        """When nothing matches, return an empty list."""
        results = await router.query("something obscure")
        assert results == []

    async def test_exact_match_boosted(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Exact matches should appear with the highest weight."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="Alice",
                entity_type="user",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={},
            )
        ]
        results = await router.query("Alice")
        assert len(results) == 1
        assert results[0].source == "exact"
        assert results[0].score == 1.0

    async def test_keyword_results_included(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Keyword hits should be included with keyword weight."""
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Goal1", entity_type="goal",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=1.5,
                source="keyword",
            )
        ]
        results = await router.query("ship mvp")
        assert len(results) == 1
        assert results[0].source == "keyword"
        assert results[0].score == pytest.approx(1.5 * 0.8)

    async def test_traversal_expansion(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Traversal from keyword hits should bring in neighbors."""
        keyword_ent = GraphEntity(
            name="Alice", entity_type="user",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )
        neighbor = GraphEntity(
            name="Bob", entity_type="user",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )
        mock_store.search_keyword.return_value = [
            SearchResult(entity=keyword_ent, score=1.0, source="keyword")
        ]
        mock_store.search_traversal.return_value = [neighbor]

        results = await router.query("Alice")
        sources = {r.source for r in results}
        assert "keyword" in sources
        assert "traversal" in sources

    async def test_deduplication_keeps_highest_score(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """If the same entity appears from multiple sources, keep the highest score."""
        ent = GraphEntity(
            name="Alice", entity_type="user",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )
        mock_store.search_exact.return_value = [ent]
        mock_store.search_keyword.return_value = [
            SearchResult(entity=ent, score=0.5, source="keyword")
        ]

        results = await router.query("Alice")
        assert len(results) == 1
        # exact weight is 1.0, keyword weight is 0.5*0.8=0.4 -> exact wins
        assert results[0].score == 1.0

    async def test_limit_truncates(self, mock_store: AsyncMock) -> None:
        """Result list should be truncated to the limit."""
        router = QueryRouter(store=mock_store, default_limit=2)
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name=f"E{i}", entity_type="x",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=float(i),
                source="keyword",
            )
            for i in range(10)
        ]
        results = await router.query("many")
        assert len(results) == 2

    async def test_temporal_filter_passed_through(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Temporal point should be forwarded to all search methods."""
        await router.query("x", temporal_point="2024-03-01T00:00:00Z")
        mock_store.search_exact.assert_awaited_once()
        assert mock_store.search_exact.await_args.kwargs["temporal_point"] == "2024-03-01T00:00:00Z"
        mock_store.search_keyword.assert_awaited_once()
        assert mock_store.search_keyword.await_args.kwargs["temporal_point"] == "2024-03-01T00:00:00Z"

    async def test_results_sorted_by_score(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Results should be ordered by descending score."""
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Low", entity_type="x",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=0.1,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="High", entity_type="x",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=2.0,
                source="keyword",
            ),
        ]
        results = await router.query("x")
        assert results[0].content.startswith("High")
        assert results[1].content.startswith("Low")


# ------------------------------------------------------------------
# Context chunk tests
# ------------------------------------------------------------------

class TestContextChunk:
    """Tests for the output formatting."""

    async def test_chunk_content_format(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Chunks should contain entity name, type, and a few properties."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="Alice",
                entity_type="user",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"theme": "dark", "lang": "en", "region": "us", "extra": "ignored"},
            )
        ]
        results = await router.query("Alice")
        assert len(results) == 1
        chunk = results[0]
        assert "Alice" in chunk.content
        assert "user" in chunk.content
        assert "theme=dark" in chunk.content
        assert "lang=en" in chunk.content
        assert "region=us" in chunk.content
        # Should only include first few properties
        assert "extra=ignored" not in chunk.content

    async def test_chunk_validity_window(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Chunks should preserve temporal validity."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="OldFact",
                entity_type="fact",
                valid_from="2024-01-01T00:00:00Z",
                valid_to="2024-06-01T00:00:00Z",
                properties={},
            )
        ]
        results = await router.query("OldFact")
        assert results[0].valid_from == "2024-01-01T00:00:00Z"
        assert results[0].valid_to == "2024-06-01T00:00:00Z"
