"""Performance benchmarks for core operations.

Targets:
- Event append: <50ms
- Rule-based extraction: <10ms
- Neo4j upsert: <100ms (integration)
- Hybrid query: <200ms (integration)
- Total context retrieval: <300ms (integration)
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from zaxy.event import EventLog
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.query import QueryRouter

# ------------------------------------------------------------------
# Event log benchmarks
# ------------------------------------------------------------------

class TestBenchmarkEventLog:
    """Benchmarks for Eventloom JSONL operations."""

    def test_append_latency(self, benchmark: pytest.BenchmarkFixture) -> None:
        """Event append should complete in <50ms."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            path = fh.name
        log = EventLog(path)

        def _append() -> None:
            log.append("test.event", "actor", {"x": 1})

        benchmark(_append)
        assert benchmark.stats["mean"] < 0.050
        Path(path).unlink(missing_ok=True)

    def test_read_all_latency(self, benchmark: pytest.BenchmarkFixture) -> None:
        """Reading 100 events should be fast (<10ms mean)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            path = fh.name
        log = EventLog(path)
        for i in range(100):
            log.append("test.event", "actor", {"idx": i})

        def _read() -> None:
            log.read_all()

        benchmark(_read)
        assert benchmark.stats["mean"] < 0.010
        Path(path).unlink(missing_ok=True)


# ------------------------------------------------------------------
# Extraction benchmarks
# ------------------------------------------------------------------

class TestBenchmarkExtraction:
    """Benchmarks for the hybrid extraction engine."""

    def test_goal_created_extraction(self, benchmark: pytest.BenchmarkFixture) -> None:
        """Rule-based extraction should complete in <10ms."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            path = fh.name
        log = EventLog(path)
        event = log.append("goal.created", "user", {"title": "Ship MVP", "description": "Launch the product"})

        def _extract() -> None:
            extract(event)

        benchmark(_extract)
        assert benchmark.stats["mean"] < 0.010
        Path(path).unlink(missing_ok=True)

    def test_task_proposed_extraction(self, benchmark: pytest.BenchmarkFixture) -> None:
        """Task extraction should also be <10ms."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            path = fh.name
        log = EventLog(path)
        event = log.append("task.proposed", "agent", {"task_id": "t1", "summary": "Write tests"})

        def _extract() -> None:
            extract(event)

        benchmark(_extract)
        assert benchmark.stats["mean"] < 0.010
        Path(path).unlink(missing_ok=True)


# ------------------------------------------------------------------
# Graph store benchmarks (mocked driver)
# ------------------------------------------------------------------

class TestBenchmarkGraphStore:
    """Benchmarks for graph operations with a mocked driver."""

    @pytest.fixture
    def mock_store(self) -> GraphStore:
        """Return a GraphStore with a mocked driver."""
        gs = GraphStore("bolt://localhost:7687", "neo4j", "test")
        gs._driver = AsyncMock()
        return gs

    def test_search_exact_latency(self, mock_store: GraphStore, benchmark: pytest.BenchmarkFixture) -> None:
        """Exact search Cypher generation + result parsing should be <5ms."""
        mock_store._driver.execute_query.return_value = ([], None, None)

        def _search() -> None:
            asyncio.run(mock_store.search_exact("Alice"))

        benchmark(_search)
        assert benchmark.stats["mean"] < 0.005

    def test_upsert_extraction_latency(self, mock_store: GraphStore, benchmark: pytest.BenchmarkFixture) -> None:
        """Upserting 3 entities + 2 edges should be <10ms mocked."""
        from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult

        result = ExtractionResult(
            entities=[
                ExtractedEntity(name=f"E{i}", entity_type="x", observed_at="2024-01-01T00:00:00Z")
                for i in range(3)
            ],
            edges=[
                ExtractedEdge(source="E0", target="E1", relation_type="rel", valid_from="2024-01-01T00:00:00Z"),
                ExtractedEdge(source="E1", target="E2", relation_type="rel", valid_from="2024-01-01T00:00:00Z"),
            ],
            source_event_seq=1,
        )
        mock_store._driver.execute_query.return_value = (None, None, None)

        def _upsert() -> None:
            asyncio.run(mock_store.upsert_extraction(result))

        benchmark(_upsert)
        assert benchmark.stats["mean"] < 0.010


# ------------------------------------------------------------------
# Query router benchmarks (mocked store)
# ------------------------------------------------------------------

class TestBenchmarkQueryRouter:
    """Benchmarks for the hybrid retrieval pipeline."""

    @pytest.fixture
    def mock_router(self) -> QueryRouter:
        """Return a QueryRouter with a mocked store."""
        store = AsyncMock()
        store.search_exact = AsyncMock(return_value=[])
        store.search_keyword = AsyncMock(return_value=[])
        store.search_traversal = AsyncMock(return_value=[])
        return QueryRouter(store=store, default_limit=10)

    def test_empty_query_latency(self, mock_router: QueryRouter, benchmark: pytest.BenchmarkFixture) -> None:
        """Empty query should return fast (<5ms)."""
        def _query() -> None:
            asyncio.run(mock_router.query("nonexistent"))

        benchmark(_query)
        assert benchmark.stats["mean"] < 0.005
