"""Tests for zaxy.graph — Neo4j graph store.

Unit tests mock the Neo4j driver. Integration tests hit a real Neo4j
instance via Docker (marked with `integration`)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult
from zaxy.graph import GraphStore, SearchResult, _record_to_entity

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_node(**props: Any) -> Any:
    """Create a mock Neo4j node record that supports dict-like access."""
    class FakeNode:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def keys(self) -> Any:
            return self._data.keys()

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def get(self, key: str, default: Any = None) -> Any:
            return self._data.get(key, default)

    return FakeNode(props)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_driver() -> AsyncMock:
    """Return a mock Neo4j async driver."""
    return AsyncMock()


@pytest.fixture
def store(mock_driver: AsyncMock) -> GraphStore:
    """Return a GraphStore with a patched driver."""
    gs = GraphStore("bolt://localhost:7687", "neo4j", "test")
    gs._driver = mock_driver
    return gs


# ------------------------------------------------------------------
# Connection tests
# ------------------------------------------------------------------

class TestConnection:
    """Tests for driver lifecycle."""

    @patch("zaxy.graph.AsyncGraphDatabase.driver")
    async def test_connect_creates_driver(self, mock_factory: MagicMock) -> None:
        """connect() should instantiate the driver."""
        gs = GraphStore("bolt://x", "u", "p")
        await gs.connect()
        mock_factory.assert_called_once_with("bolt://x", auth=("u", "p"))

    async def test_close_closes_driver(self, mock_driver: AsyncMock) -> None:
        """close() should close the driver and clear the reference."""
        gs = GraphStore("bolt://x", "u", "p")
        gs._driver = mock_driver
        await gs.close()
        mock_driver.close.assert_awaited_once()
        assert gs._driver is None


# ------------------------------------------------------------------
# Schema tests
# ------------------------------------------------------------------

class TestSchema:
    """Tests for idempotent schema creation."""

    async def test_init_schema_creates_constraints(self, store: GraphStore) -> None:
        """init_schema should issue constraint and index Cypher."""
        await store.init_schema()
        calls = store._driver.execute_query.await_args_list
        cypher_statements = [call.args[0] for call in calls]
        assert any("CREATE CONSTRAINT" in s for s in cypher_statements)
        assert any("CREATE VECTOR INDEX" in s for s in cypher_statements)
        assert any("CREATE FULLTEXT INDEX" in s for s in cypher_statements)

    async def test_init_schema_requires_connect(self) -> None:
        """Calling init_schema before connect should raise AssertionError."""
        gs = GraphStore("bolt://x", "u", "p")
        with pytest.raises(AssertionError):
            await gs.init_schema()


# ------------------------------------------------------------------
# Ingestion tests
# ------------------------------------------------------------------

class TestIngestion:
    """Tests for upserting extraction results into Neo4j."""

    async def test_upsert_entity(self, store: GraphStore) -> None:
        """Upserting an entity should MERGE it with temporal properties."""
        result = ExtractionResult(
            entities=[ExtractedEntity(name="Alice", entity_type="user", observed_at="2024-01-01T00:00:00Z")],
            edges=[],
            source_event_seq=1,
        )
        await store.upsert_extraction(result)

        call = store._driver.execute_query.await_args_list[0]
        cypher, kwargs = call.args[0], call.kwargs
        assert "MERGE (e:Entity" in cypher
        assert kwargs["name"] == "Alice"
        assert kwargs["entity_type"] == "user"

    async def test_upsert_edge(self, store: GraphStore) -> None:
        """Upserting an edge should MATCH source/target then MERGE the rel."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", entity_type="user", observed_at="2024-01-01T00:00:00Z"),
                ExtractedEntity(name="Goal1", entity_type="goal", observed_at="2024-01-01T00:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Alice",
                    target="Goal1",
                    relation_type="created_goal",
                    valid_from="2024-01-01T00:00:00Z",
                )
            ],
            source_event_seq=1,
        )
        await store.upsert_extraction(result)

        # Entities are upserted first (2 calls), then edges (1 call)
        assert store._driver.execute_query.await_count == 3
        call = store._driver.execute_query.await_args_list[2]
        cypher, kwargs = call.args[0], call.kwargs
        assert "MATCH (s:Entity" in cypher
        assert "MATCH (t:Entity" in cypher
        assert "MERGE (s)-[r:RELATES" in cypher
        assert kwargs["source"] == "Alice"
        assert kwargs["target"] == "Goal1"

    async def test_upsert_multiple_entities(self, store: GraphStore) -> None:
        """Multiple entities should produce multiple MERGE calls."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="A", entity_type="x", observed_at="2024-01-01T00:00:00Z"),
                ExtractedEntity(name="B", entity_type="y", observed_at="2024-01-01T00:00:00Z"),
                ExtractedEntity(name="C", entity_type="z", observed_at="2024-01-01T00:00:00Z"),
            ],
            edges=[],
            source_event_seq=1,
        )
        await store.upsert_extraction(result)
        assert store._driver.execute_query.await_count == 3

    async def test_invalidate_entity(self, store: GraphStore) -> None:
        """invalidate_entity should set valid_to on the live node."""
        await store.invalidate_entity("Alice", "user", "2024-06-01T00:00:00Z")
        call = store._driver.execute_query.await_args
        cypher, kwargs = call.args[0], call.kwargs
        assert "SET e.valid_to = datetime($invalid_at)" in cypher
        assert kwargs["invalid_at"] == "2024-06-01T00:00:00Z"

    async def test_invalidate_edge(self, store: GraphStore) -> None:
        """invalidate_edge should set valid_to on the live relationship."""
        await store.invalidate_edge("A", "B", "rel", "2024-01-01T00:00:00Z", "2024-06-01T00:00:00Z")
        call = store._driver.execute_query.await_args
        cypher, kwargs = call.args[0], call.kwargs
        assert "SET r.valid_to = datetime($invalid_at)" in cypher
        assert kwargs["invalid_at"] == "2024-06-01T00:00:00Z"


# ------------------------------------------------------------------
# Retrieval tests
# ------------------------------------------------------------------

class TestRetrieval:
    """Tests for search methods."""

    async def test_search_exact_by_name(self, store: GraphStore) -> None:
        """Exact search should query by name."""
        node = _make_node(name="Alice", entity_type="user", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = ([{"e": node}], None, None)

        results = await store.search_exact("Alice")
        assert len(results) == 1
        assert results[0].name == "Alice"

    async def test_search_exact_with_type_filter(self, store: GraphStore) -> None:
        """Exact search should optionally filter by entity_type."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_exact("Alice", entity_type="user")
        call = store._driver.execute_query.await_args
        assert "e.entity_type = $entity_type" in call.args[0]

    async def test_search_exact_with_temporal_filter(self, store: GraphStore) -> None:
        """Exact search should optionally filter by temporal point."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_exact("Alice", temporal_point="2024-03-01T00:00:00Z")
        call = store._driver.execute_query.await_args
        assert "datetime($t)" in call.args[0]

    async def test_search_traversal(self, store: GraphStore) -> None:
        """Traversal should follow relationships to a given depth."""
        node = _make_node(name="Bob", entity_type="user", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = ([{"neighbor": node}], None, None)

        results = await store.search_traversal("Alice", depth=2)
        assert len(results) == 1
        assert results[0].name == "Bob"

    async def test_search_keyword(self, store: GraphStore) -> None:
        """Keyword search should use the full-text index."""
        node = _make_node(name="Goal1", entity_type="goal", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = ([{"node": node, "score": 1.23}], None, None)

        results = await store.search_keyword("ship mvp")
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 1.23
        assert results[0].source == "keyword"

    async def test_search_keyword_with_temporal_filter(self, store: GraphStore) -> None:
        """Keyword search should optionally filter by temporal point."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_keyword("ship", temporal_point="2024-03-01T00:00:00Z")
        call = store._driver.execute_query.await_args
        assert "datetime($t)" in call.args[0]

    async def test_search_traversal_with_relation_type(self, store: GraphStore) -> None:
        """Traversal should optionally filter by relation_type."""
        node = _make_node(name="Bob", entity_type="user", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = ([{"neighbor": node}], None, None)

        results = await store.search_traversal("Alice", relation_type="created_goal")
        call = store._driver.execute_query.await_args
        assert "relation_type: $relation_type" in call.args[0]
        assert call.kwargs["relation_type"] == "created_goal"
        assert len(results) == 1
        assert results[0].name == "Bob"

    async def test_search_traversal_with_temporal_filter(self, store: GraphStore) -> None:
        """Traversal should optionally filter by temporal point."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_traversal("Alice", temporal_point="2024-03-01T00:00:00Z")
        call = store._driver.execute_query.await_args
        assert "datetime($t)" in call.args[0]


# ------------------------------------------------------------------
# Helper tests
# ------------------------------------------------------------------

class TestHelpers:
    """Tests for internal helper functions."""

    def test_record_to_entity(self) -> None:
        """_record_to_entity should map Neo4j properties correctly."""
        node = _make_node(
            name="Alice",
            entity_type="user",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            extra="value",
        )

        entity = _record_to_entity(node)
        assert entity.name == "Alice"
        assert entity.entity_type == "user"
        assert entity.valid_to is None
        assert entity.properties == {"extra": "value"}

    def test_record_to_entity_with_valid_to(self) -> None:
        """valid_to should be converted to string when present."""
        node = _make_node(
            name="X",
            entity_type="y",
            valid_from="2024-01-01T00:00:00Z",
            valid_to="2024-06-01T00:00:00Z",
        )

        entity = _record_to_entity(node)
        assert entity.valid_to == "2024-06-01T00:00:00Z"


# ------------------------------------------------------------------
# Integration tests (require Docker)
# ------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Integration tests against a real Neo4j instance."""

    @pytest.fixture
    async def real_store(self) -> GraphStore:
        """Connect to the test Neo4j container."""
        gs = GraphStore("bolt://localhost:7688", "neo4j", "testpassword")
        await gs.connect()
        await gs.init_schema()
        yield gs
        # Clean up
        await gs._driver.execute_query("MATCH (n) DETACH DELETE n")
        await gs.close()

    async def test_roundtrip_upsert_and_search(self, real_store: GraphStore) -> None:
        """Upsert an extraction result and retrieve it via exact search."""
        result = ExtractionResult(
            entities=[ExtractedEntity(name="TestUser", entity_type="user", observed_at="2024-01-01T00:00:00Z")],
            edges=[],
            source_event_seq=1,
        )
        await real_store.upsert_extraction(result)

        found = await real_store.search_exact("TestUser")
        assert len(found) == 1
        assert found[0].name == "TestUser"
        assert found[0].entity_type == "user"

    async def test_temporal_invalidation(self, real_store: GraphStore) -> None:
        """Invalidating an entity should hide it from temporal queries."""
        result = ExtractionResult(
            entities=[ExtractedEntity(name="OldFact", entity_type="fact", observed_at="2024-01-01T00:00:00Z")],
            edges=[],
            source_event_seq=1,
        )
        await real_store.upsert_extraction(result)
        await real_store.invalidate_entity("OldFact", "fact", "2024-06-01T00:00:00Z")

        # Before invalidation: should find
        found_before = await real_store.search_exact("OldFact", temporal_point="2024-03-01T00:00:00Z")
        assert len(found_before) == 1

        # After invalidation: should not find
        found_after = await real_store.search_exact("OldFact", temporal_point="2024-07-01T00:00:00Z")
        assert len(found_after) == 0

    async def test_full_pipeline_event_to_query(self, real_store: GraphStore) -> None:
        """End-to-end: event -> extract -> upsert -> query -> context chunk."""
        from zaxy.query import QueryRouter

        # 1. Simulate an agent session with multiple events
        events = [
            {
                "event_type": "goal.created",
                "actor": "user",
                "payload": {"title": "Ship MVP", "description": "Get product to market"},
            },
            {
                "event_type": "task.proposed",
                "actor": "agent",
                "payload": {"task_id": "t1", "summary": "Design landing page"},
            },
            {
                "event_type": "task.claimed",
                "actor": "user",
                "payload": {"task_id": "t1"},
            },
        ]

        # 2. Extract and upsert each event
        from datetime import datetime, timezone
        from zaxy.event import EventLog
        from zaxy.extract import extract
        log = EventLog("/tmp/zaxy_pipeline_test.jsonl")
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for ev in events:
            event = log.append(
                ev["event_type"], ev["actor"], ev["payload"],
                timestamp=ts,
            )
            result = extract(event)
            await real_store.upsert_extraction(result)

        # 3. Query the graph via the router
        router = QueryRouter(store=real_store, default_limit=10)
        chunks = await router.query("Ship MVP")

        # 4. Verify context chunks contain the goal
        assert len(chunks) > 0
        assert any("Ship MVP" in c.content for c in chunks)

        # 5. Verify temporal filtering works end-to-end
        future_chunks = await router.query("Ship MVP", temporal_point="2025-01-01T00:00:00Z")
        # Should still find it (not invalidated, and 2025 > 2024)
        assert len(future_chunks) > 0
