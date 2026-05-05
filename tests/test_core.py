"""Tests for zaxy.core — MemoryFabric orchestrator.

Tests cover the full orchestration pipeline: event → extract → graph → query,
with all external dependencies mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.core import Context, MemoryFabric
from zaxy.event import ReplayResult
from zaxy.query import ContextChunk

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def fabric() -> MemoryFabric:
    """Return a MemoryFabric with mocked dependencies."""
    with (
        patch("zaxy.core.EventLog") as mock_log_cls,
        patch("zaxy.core.GraphStore") as mock_graph_cls,
        patch("zaxy.core.QueryRouter") as mock_router_cls,
        patch("zaxy.core.MemoryTracer") as mock_tracer_cls,
    ):
        log = MagicMock()
        log.append.return_value = MagicMock(seq=1, hash="a" * 64, type="x", actor="y", timestamp="2024-01-01T00:00:00Z")
        mock_log_cls.return_value = log

        graph = AsyncMock()
        mock_graph_cls.return_value = graph

        router = AsyncMock()
        mock_router_cls.return_value = router

        tracer = AsyncMock()
        mock_tracer_cls.return_value = tracer

        f = MemoryFabric()
        f.eventloom = log
        f.graph = graph
        f.query_router = router
        f.tracer = tracer
        yield f


# ------------------------------------------------------------------
# Lifecycle tests
# ------------------------------------------------------------------

class TestLifecycle:
    """Tests for connect/close behavior."""

    async def test_connect_initializes_graph_and_tracer(self, fabric: MemoryFabric) -> None:
        """connect() should init schema and connect tracer."""
        await fabric.connect()
        fabric.graph.connect.assert_awaited_once()
        fabric.graph.init_schema.assert_awaited_once()
        fabric.tracer.connect.assert_awaited_once()
        assert fabric._connected is True

    async def test_connect_is_idempotent(self, fabric: MemoryFabric) -> None:
        """Multiple connect() calls should not re-initialize."""
        await fabric.connect()
        await fabric.connect()
        fabric.graph.connect.assert_awaited_once()

    async def test_close_closes_all(self, fabric: MemoryFabric) -> None:
        """close() should close graph and tracer."""
        await fabric.connect()
        await fabric.close()
        fabric.graph.close.assert_awaited_once()
        fabric.tracer.close.assert_awaited_once()
        assert fabric._connected is False


# ------------------------------------------------------------------
# Append tests
# ------------------------------------------------------------------

class TestAppend:
    """Tests for the write path."""

    async def test_appends_event(self, fabric: MemoryFabric) -> None:
        """append() should write to Eventloom."""
        await fabric.append("goal.created", actor="user", payload={"title": "T"})
        fabric.eventloom.append.assert_called_once_with(
            "goal.created", actor="user", payload={"title": "T"}, thread="default"
        )

    async def test_extracts_and_upserts(self, fabric: MemoryFabric) -> None:
        """append() should extract entities and upsert to graph."""
        await fabric.append("goal.created", actor="user", payload={"title": "T"})
        fabric.graph.upsert_extraction.assert_awaited_once()

    async def test_traces_append(self, fabric: MemoryFabric) -> None:
        """append() should emit a Pathlight trace."""
        await fabric.append("goal.created", actor="user", payload={"title": "T"})
        fabric.tracer.trace_append.assert_awaited_once()

    async def test_auto_connects(self, fabric: MemoryFabric) -> None:
        """append() should auto-connect if not already connected."""
        await fabric.append("x", actor="y")
        fabric.graph.connect.assert_awaited_once()


# ------------------------------------------------------------------
# Query tests
# ------------------------------------------------------------------

class TestQuery:
    """Tests for the read path."""

    async def test_queries_router(self, fabric: MemoryFabric) -> None:
        """query() should delegate to QueryRouter."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Alice (user)",
                source="exact",
                score=1.0,
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
            )
        ]
        results = await fabric.query("Alice")
        fabric.query_router.query.assert_awaited_once_with("Alice", temporal_point=None, limit=10)
        assert len(results) == 1
        assert isinstance(results[0], Context)

    async def test_passes_temporal_filter(self, fabric: MemoryFabric) -> None:
        """query() should forward temporal_point to router."""
        fabric.query_router.query.return_value = []
        await fabric.query("x", temporal_point="2024-03-01T00:00:00Z", limit=5)
        fabric.query_router.query.assert_awaited_once_with(
            "x", temporal_point="2024-03-01T00:00:00Z", limit=5
        )

    async def test_traces_query(self, fabric: MemoryFabric) -> None:
        """query() should emit a Pathlight trace with result count."""
        fabric.query_router.query.return_value = []
        await fabric.query("x")
        fabric.tracer.trace_query.assert_awaited_once()
        args = fabric.tracer.trace_query.await_args
        assert args.args[0] == "x"
        assert args.args[1] == 0

    async def test_auto_connects(self, fabric: MemoryFabric) -> None:
        """query() should auto-connect if not already connected."""
        fabric.query_router.query.return_value = []
        await fabric.query("x")
        fabric.graph.connect.assert_awaited_once()


# ------------------------------------------------------------------
# Replay tests
# ------------------------------------------------------------------

class TestReplay:
    """Tests for event replay."""

    async def test_replay_delegates_to_eventlog(self, fabric: MemoryFabric) -> None:
        """replay() should delegate to EventLog.replay()."""
        mock_result = MagicMock(spec=ReplayResult)
        fabric.eventloom.replay.return_value = mock_result
        result = await fabric.replay(from_seq=5)
        fabric.eventloom.replay.assert_called_once_with(from_seq=5)
        assert result is mock_result


# ------------------------------------------------------------------
# Invalidation tests
# ------------------------------------------------------------------

class TestInvalidate:
    """Tests for bi-temporal invalidation."""

    async def test_invalidate_entity(self, fabric: MemoryFabric) -> None:
        """invalidate() should call graph.invalidate_entity."""
        await fabric.invalidate("OldFact", "fact", "2024-06-01T00:00:00Z")
        fabric.graph.invalidate_entity.assert_awaited_once_with(
            "OldFact", "fact", "2024-06-01T00:00:00Z"
        )

    async def test_auto_connects(self, fabric: MemoryFabric) -> None:
        """invalidate() should auto-connect if not already connected."""
        await fabric.invalidate("x", "y", "2024-01-01T00:00:00Z")
        fabric.graph.connect.assert_awaited_once()


# ------------------------------------------------------------------
# Handoff tests
# ------------------------------------------------------------------

class TestHandoff:
    """Tests for handoff summary generation."""

    async def test_handoff_delegates_to_eventlog(self, fabric: MemoryFabric) -> None:
        """handoff_summary() should delegate to EventLog.handoff_summary()."""
        fabric.eventloom.handoff_summary.return_value = {"event_count": 42}
        summary = await fabric.handoff_summary()
        fabric.eventloom.handoff_summary.assert_called_once()
        assert summary["event_count"] == 42
