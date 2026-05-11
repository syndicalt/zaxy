"""Tests for zaxy.graph — Neo4j graph store.

Unit tests mock the Neo4j driver. Integration tests hit a real Neo4j
instance via Docker (marked with `integration`)."""

from __future__ import annotations

from datetime import UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult
from zaxy.graph import (
    GraphStore,
    SearchResult,
    _record_to_entity,
    _typed_relationship_label,
)

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

    @patch("zaxy.graph.AsyncGraphDatabase.driver")
    async def test_connect_uses_trust_all_object(self, mock_factory: MagicMock) -> None:
        """trust_all should pass Neo4j's trust object, not a bare bool."""
        from neo4j import TrustAll

        gs = GraphStore("bolt://x", "u", "p", trust_all=True)
        await gs.connect()
        kwargs = mock_factory.call_args.kwargs
        assert kwargs["encrypted"] is True
        assert isinstance(kwargs["trusted_certificates"], TrustAll)

    @patch("zaxy.graph.AsyncGraphDatabase.driver")
    async def test_connect_uses_custom_ca_object(self, mock_factory: MagicMock) -> None:
        """ca_cert should pass Neo4j's custom CA trust object."""
        from neo4j import TrustCustomCAs

        gs = GraphStore("bolt://x", "u", "p", ca_cert="/tmp/ca.pem")
        await gs.connect()
        kwargs = mock_factory.call_args.kwargs
        assert kwargs["encrypted"] is True
        assert isinstance(kwargs["trusted_certificates"], TrustCustomCAs)

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
        assert any("DROP CONSTRAINT entity_id IF EXISTS" in s for s in cypher_statements)
        assert any(
            "REQUIRE (e.session_id, e.name, e.entity_type, e.valid_from) IS UNIQUE" in s
            for s in cypher_statements
        )
        assert any(
            "FOR (s:Session) REQUIRE s.id IS UNIQUE" in s for s in cypher_statements
        )
        assert any(
            "FOR (ev:Event) REQUIRE (ev.session_id, ev.seq) IS UNIQUE" in s
            for s in cypher_statements
        )
        assert any("CREATE INDEX event_prev_hash" in s for s in cypher_statements)
        assert any("CREATE INDEX entity_lookup" in s for s in cypher_statements)
        assert any(
            "FOR (src:Source) REQUIRE (src.session_id, src.path) IS UNIQUE" in s
            for s in cypher_statements
        )
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
            entities=[
                ExtractedEntity(
                    name="Alice",
                    entity_type="user",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="Works on memory",
                    embedding=[0.1, 0.2],
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="b" * 64,
            source_event_prev_hash="a" * 64,
            source_event_type="goal.created",
            source_thread="agent-1",
        )
        await store.upsert_extraction(result, session_id="agent-1")

        call = store._driver.execute_query.await_args_list[0]
        cypher, kwargs = call.args[0], call.kwargs
        assert "MERGE (s:Session {id: $session_id})" in cypher
        assert "MERGE (ev:Event {session_id: $session_id, seq: $source_event_seq})" in cypher
        assert "ev.prev_hash = $source_event_prev_hash" in cypher
        assert "MERGE (s)-[r:HAS_EVENT]->(ev)" in cypher
        assert "CALL (ev) {" in cypher
        assert "MATCH (prev:Event {session_id: $session_id, hash: $source_event_prev_hash})" in cypher
        assert "MERGE (prev)-[next:NEXT_EVENT]->(ev)" in cypher
        assert "MERGE (ev)-[previous:PREVIOUS_EVENT]->(prev)" in cypher
        assert "MATCH (next_event:Event {session_id: $session_id, prev_hash: $source_event_hash})" in cypher
        assert "MERGE (ev)-[next_rel:NEXT_EVENT]->(next_event)" in cypher
        assert "MERGE (next_event)-[previous_rel:PREVIOUS_EVENT]->(ev)" in cypher
        assert "RETURN previous_event_links, next_event_links" in cypher
        assert kwargs["session_id"] == "agent-1"
        assert kwargs["source_event_seq"] == 1
        assert kwargs["source_event_hash"] == "b" * 64
        assert kwargs["source_event_prev_hash"] == "a" * 64
        assert kwargs["source_event_type"] == "goal.created"
        assert kwargs["source_thread"] == "agent-1"

        call = store._driver.execute_query.await_args_list[1]
        cypher, kwargs = call.args[0], call.kwargs
        assert "MERGE (e:Entity" in cypher
        assert "session_id: $session_id" in cypher
        assert "valid_from: datetime($observed_at)" in cypher
        assert "SET prev.valid_to = e.valid_from" in cypher
        assert "SET e.valid_to = next_valid_from" in cypher
        assert "copied_incoming_relationships" in cypher
        assert "copied_outgoing_relationships" in cypher
        assert "e.summary = coalesce($summary, e.summary)" in cypher
        assert "e.embedding = coalesce($embedding, e.embedding)" in cypher
        assert "e.source_event_seq = $source_event_seq" in cypher
        assert "e.source_event_hash = $source_event_hash" in cypher
        assert "MATCH (ev:Event {session_id: $session_id, seq: $source_event_seq})" in cypher
        assert "MERGE (ev)-[pe:PROJECTED_ENTITY" in cypher
        assert kwargs["name"] == "Alice"
        assert kwargs["entity_type"] == "user"
        assert kwargs["session_id"] == "agent-1"
        assert kwargs["source_event_seq"] == 1
        assert kwargs["source_event_hash"] == "b" * 64
        assert kwargs["source_event_type"] == "goal.created"
        assert kwargs["source_thread"] == "agent-1"
        assert kwargs["summary"] == "Works on memory"
        assert kwargs["embedding"] == [0.1, 0.2]
        assert kwargs["properties"] == {}

    async def test_upsert_entity_applies_extracted_properties(self, store: GraphStore) -> None:
        """Extractor-supplied safe properties should be projected to Neo4j."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="docs/guide.md:4-8",
                    entity_type="document",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="Document chunk",
                    properties={
                        "source_path": "docs/guide.md",
                        "source_start_line": 4,
                        "ignored_none": None,
                    },
                )
            ],
            edges=[],
            source_event_seq=1,
        )

        await store.upsert_extraction(result, session_id="agent-1")

        call = store._driver.execute_query.await_args_list[1]
        cypher, kwargs = call.args[0], call.kwargs
        assert "SET e += $properties" in cypher
        assert kwargs["properties"] == {
            "source_path": "docs/guide.md",
            "source_start_line": 4,
        }

    async def test_upsert_entity_projects_source_citation_edges(self, store: GraphStore) -> None:
        """Source-backed entities should create traversable citation nodes and edges."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="docs/guide.md:4-8",
                    entity_type="document",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="Document chunk",
                    properties={
                        "source_path": "docs/guide.md",
                        "source_start_line": 4,
                        "source_end_line": 8,
                        "source_sha256": "abc123",
                    },
                )
            ],
            edges=[],
            source_event_seq=11,
            source_event_hash="d" * 64,
            source_event_type="document.indexed",
        )

        await store.upsert_extraction(result, session_id="agent-1")

        call = store._driver.execute_query.await_args_list[1]
        cypher, kwargs = call.args[0], call.kwargs
        assert "MERGE (src:Source {session_id: $session_id, path: $source_path})" in cypher
        assert "MERGE (e)-[cs:CITES_SOURCE]->(src)" in cypher
        assert "MERGE (ev)-[ecs:CITES_SOURCE]->(src)" in cypher
        assert "cs.source_start_line = $source_start_line" in cypher
        assert "ecs.source_event_hash = $source_event_hash" in cypher
        assert kwargs["source_path"] == "docs/guide.md"
        assert kwargs["source_start_line"] == 4
        assert kwargs["source_end_line"] == 8
        assert kwargs["source_sha256"] == "abc123"

    async def test_upsert_entity_namespaces_storage_reserved_properties(
        self,
        store: GraphStore,
    ) -> None:
        """Extractor properties must not overwrite graph storage identity."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="agent-1:checkpoint:7",
                    entity_type="hook_checkpoint",
                    observed_at="2024-01-01T00:00:00Z",
                    properties={
                        "session_id": "agent-1",
                        "source_event_seq": 7,
                        "summary": "Payload summary should remain payload metadata",
                        "reason": "checkpoint",
                    },
                )
            ],
            edges=[],
            source_event_seq=7,
        )

        await store.upsert_extraction(result, session_id="graph-scope")

        call = store._driver.execute_query.await_args_list[1]
        assert call.kwargs["session_id"] == "graph-scope"
        assert call.kwargs["properties"] == {
            "payload_session_id": "agent-1",
            "payload_source_event_seq": 7,
            "payload_summary": "Payload summary should remain payload metadata",
            "reason": "checkpoint",
        }

    async def test_upsert_entity_drops_neo4j_unsafe_nested_properties(self, store: GraphStore) -> None:
        """Nested extracted properties should not reach Neo4j node projection."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="/repo:instructions:abc123",
                    entity_type="workspace_instructions",
                    observed_at="2024-01-01T00:00:00Z",
                    properties={
                        "root": "/repo",
                        "files": [{"path": "AGENTS.md", "kind": "agents"}],
                        "metadata": {"signature": "abc123"},
                        "file_paths": ["AGENTS.md"],
                        "ignored_none": None,
                    },
                )
            ],
            edges=[],
            source_event_seq=1,
        )

        await store.upsert_extraction(result, session_id="agent-1")

        call = store._driver.execute_query.await_args_list[1]
        assert call.kwargs["properties"] == {
            "root": "/repo",
            "file_paths": ["AGENTS.md"],
        }

    async def test_upsert_entity_versions_by_valid_from(self, store: GraphStore) -> None:
        """Reassertions should create new versions instead of overwriting history."""
        result = ExtractionResult(
            entities=[ExtractedEntity(name="Alice", entity_type="user", observed_at="2024-01-01T00:00:00Z")],
            edges=[],
            source_event_seq=1,
        )
        await store.upsert_extraction(result, session_id="agent-1")

        cypher = store._driver.execute_query.await_args_list[1].args[0]
        assert "MERGE (e:Entity" in cypher
        assert "prev.session_id = $session_id" in cypher
        assert "valid_from: datetime($observed_at)" in cypher
        assert "ON MATCH SET e.updated_at = datetime($observed_at)" not in cypher
        assert "SET prev.valid_to = e.valid_from" in cypher

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
            source_event_hash="c" * 64,
            source_event_type="goal.created",
        )
        await store.upsert_extraction(result, session_id="agent-1")

        # Backbone is upserted first, then entities (2 calls), then edges (1 call).
        assert store._driver.execute_query.await_count == 4
        call = store._driver.execute_query.await_args_list[3]
        cypher, kwargs = call.args[0], call.kwargs
        assert "MATCH (s:Entity" in cypher
        assert "MATCH (t:Entity" in cypher
        assert "s.session_id = $session_id" in cypher
        assert "t.session_id = $session_id" in cypher
        assert "s.valid_from <= datetime($valid_from)" in cypher
        assert "t.valid_from <= datetime($valid_from)" in cypher
        assert "MERGE (s)-[r:RELATES" in cypher
        assert "r.session_id = $session_id" in cypher
        assert "r.source_event_seq = $source_event_seq" in cypher
        assert "r.source_event_hash = $source_event_hash" in cypher
        assert "MERGE (s)-[typed:CREATED_GOAL" in cypher
        assert "typed.relation_type = $relation_type" in cypher
        assert "typed.source_event_hash = $source_event_hash" in cypher
        assert "MATCH (ev:Event {session_id: $session_id, seq: $source_event_seq})" in cypher
        assert "MERGE (ev)-[pr:PROJECTED_RELATION" in cypher
        assert kwargs["source"] == "Alice"
        assert kwargs["target"] == "Goal1"
        assert kwargs["session_id"] == "agent-1"
        assert kwargs["source_event_hash"] == "c" * 64

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
        await store.upsert_extraction(result, session_id="agent-1")
        assert store._driver.execute_query.await_count == 4

    def test_typed_relationship_label_converts_valid_relation_types(self) -> None:
        """Relation types should become readable Neo4j relationship labels."""
        assert _typed_relationship_label("calls_symbol") == "CALLS_SYMBOL"
        assert _typed_relationship_label("projected_llm_packet") == "PROJECTED_LLM_PACKET"

    def test_typed_relationship_label_rejects_unsafe_relation_types(self) -> None:
        """Dynamic relationship labels must reject Cypher injection characters."""
        with pytest.raises(ValueError, match="Invalid relation_type"):
            _typed_relationship_label("calls_symbol`) DELETE r //")

    async def test_invalidate_entity(self, store: GraphStore) -> None:
        """invalidate_entity should set valid_to on the live node."""
        await store.invalidate_entity(
            "Alice", "user", "2024-06-01T00:00:00Z", session_id="agent-1"
        )
        call = store._driver.execute_query.await_args
        cypher, kwargs = call.args[0], call.kwargs
        assert "e.session_id = $session_id" in cypher
        assert "SET e.valid_to = datetime($invalid_at)" in cypher
        assert kwargs["session_id"] == "agent-1"
        assert kwargs["invalid_at"] == "2024-06-01T00:00:00Z"

    async def test_invalidate_edge(self, store: GraphStore) -> None:
        """invalidate_edge should set valid_to on the live relationship."""
        await store.invalidate_edge(
            "A",
            "B",
            "rel",
            "2024-01-01T00:00:00Z",
            "2024-06-01T00:00:00Z",
            session_id="agent-1",
        )
        call = store._driver.execute_query.await_args
        cypher, kwargs = call.args[0], call.kwargs
        assert "s.session_id = $session_id" in cypher
        assert "r.session_id = $session_id" in cypher
        assert "SET r.valid_to = datetime($invalid_at)" in cypher
        assert kwargs["session_id"] == "agent-1"
        assert kwargs["invalid_at"] == "2024-06-01T00:00:00Z"


class TestProjectionStatus:
    """Tests for graph/Eventloom projection integrity inspection."""

    async def test_inspect_event_projection_status_reports_lag_and_missing_links(
        self,
        store: GraphStore,
    ) -> None:
        """Projection status should compare graph chain state with Eventloom latest."""
        store._driver.execute_query.return_value = (
            [
                {
                    "event_count": 2,
                    "latest_seq": 2,
                    "latest_hash": "b" * 64,
                    "next_event_edges": 0,
                    "previous_event_edges": 1,
                    "missing_chain_links": 1,
                }
            ],
            None,
            None,
        )

        status = await store.inspect_event_projection_status(
            "agent-1",
            eventloom_latest_seq=3,
            eventloom_latest_hash="c" * 64,
        )

        assert status.session_id == "agent-1"
        assert status.event_count == 2
        assert status.latest_seq == 2
        assert status.latest_hash == "b" * 64
        assert status.projection_lag == 1
        assert status.latest_hash_matches is False
        assert status.next_event_edges == 0
        assert status.previous_event_edges == 1
        assert status.missing_chain_links == 1
        assert status.integrity_ok is False


# ------------------------------------------------------------------
# Retrieval tests
# ------------------------------------------------------------------

class TestRetrieval:
    """Tests for search methods."""

    async def test_search_exact_by_name(self, store: GraphStore) -> None:
        """Exact search should query by name."""
        node = _make_node(name="Alice", entity_type="user", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = ([{"e": node}], None, None)

        results = await store.search_exact("Alice", session_id="agent-1")
        assert len(results) == 1
        assert results[0].name == "Alice"
        call = store._driver.execute_query.await_args
        assert "e.session_id = $session_id" in call.args[0]
        assert call.kwargs["session_id"] == "agent-1"
        assert "e.valid_to IS NULL" in call.args[0]

    async def test_search_exact_with_type_filter(self, store: GraphStore) -> None:
        """Exact search should optionally filter by entity_type."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_exact("Alice", entity_type="user", session_id="agent-1")
        call = store._driver.execute_query.await_args
        assert "e.entity_type = $entity_type" in call.args[0]

    async def test_search_exact_with_temporal_filter(self, store: GraphStore) -> None:
        """Exact search should optionally filter by temporal point."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_exact(
            "Alice",
            temporal_point="2024-03-01T00:00:00Z",
            session_id="agent-1",
        )
        call = store._driver.execute_query.await_args
        assert "datetime($t)" in call.args[0]

    async def test_search_traversal(self, store: GraphStore) -> None:
        """Traversal should follow relationships to a given depth."""
        node = _make_node(name="Bob", entity_type="user", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = ([{"neighbor": node}], None, None)

        results = await store.search_traversal("Alice", depth=2, session_id="agent-1")
        assert len(results) == 1
        assert results[0].name == "Bob"
        call = store._driver.execute_query.await_args
        assert "start.session_id = $session_id" in call.args[0]
        assert "neighbor.session_id = $session_id" in call.args[0]
        assert call.kwargs["session_id"] == "agent-1"

    async def test_search_traversal_crosses_incoming_and_outgoing_edges(
        self,
        store: GraphStore,
    ) -> None:
        """Traversal should support semantic paths like goal -> task <- actor."""
        store._driver.execute_query.return_value = ([], None, None)

        await store.search_traversal("graph-goal-0001", depth=2, session_id="agent-1")

        call = store._driver.execute_query.await_args
        assert ")-[r:RELATES*1..2]-(neighbor:Entity)" in call.args[0]
        assert "neighbor <> start" in call.args[0]

    async def test_search_keyword(self, store: GraphStore) -> None:
        """Keyword search should use the full-text index."""
        node = _make_node(name="Goal1", entity_type="goal", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = ([{"node": node, "score": 1.23}], None, None)

        results = await store.search_keyword("ship mvp", session_id="agent-1")
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 1.23
        assert results[0].source == "keyword"
        call = store._driver.execute_query.await_args
        assert "node.session_id = $session_id" in call.args[0]
        assert call.kwargs["session_id"] == "agent-1"

    async def test_search_keyword_with_temporal_filter(self, store: GraphStore) -> None:
        """Keyword search should optionally filter by temporal point."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_keyword(
            "ship",
            temporal_point="2024-03-01T00:00:00Z",
            session_id="agent-1",
        )
        call = store._driver.execute_query.await_args
        assert "datetime($t)" in call.args[0]

    async def test_search_traversal_with_relation_type(self, store: GraphStore) -> None:
        """Traversal should optionally filter by relation_type."""
        node = _make_node(name="Bob", entity_type="user", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = ([{"neighbor": node}], None, None)

        results = await store.search_traversal(
            "Alice",
            relation_type="created_goal",
            session_id="agent-1",
        )
        call = store._driver.execute_query.await_args
        assert "rel.relation_type = $relation_type" in call.args[0]
        assert call.kwargs["relation_type"] == "created_goal"
        assert len(results) == 1
        assert results[0].name == "Bob"

    async def test_search_traversal_returns_path_metadata(
        self,
        store: GraphStore,
    ) -> None:
        """Traversal results should expose relation evidence for ranking."""
        node = _make_node(name="Bob", entity_type="actor", valid_from="2024-01-01T00:00:00Z")
        store._driver.execute_query.return_value = (
            [
                {
                    "neighbor": node,
                    "path_relation_types": ["has_task", "completed_task"],
                    "path_length": 2,
                }
            ],
            None,
            None,
        )

        results = await store.search_traversal("Goal", session_id="agent-1")

        assert results[0].properties["_path_relation_types"] == ["has_task", "completed_task"]
        assert results[0].properties["_path_length"] == 2

    async def test_search_traversal_with_temporal_filter(self, store: GraphStore) -> None:
        """Traversal should optionally filter by temporal point."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_traversal(
            "Alice",
            temporal_point="2024-03-01T00:00:00Z",
            session_id="agent-1",
        )
        call = store._driver.execute_query.await_args
        assert "datetime($t)" in call.args[0]
        assert "start.valid_from <= datetime($t)" in call.args[0]
        assert "neighbor.valid_from <= datetime($t)" in call.args[0]

    async def test_search_keyword_defaults_to_current_versions(self, store: GraphStore) -> None:
        """Keyword search without a temporal point should hide historical versions."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_keyword("ship", session_id="agent-1")
        call = store._driver.execute_query.await_args
        assert "node.session_id = $session_id AND node.valid_to IS NULL" in call.args[0]

    async def test_search_vector_defaults_to_current_versions(self, store: GraphStore) -> None:
        """Vector search without a temporal point should hide historical versions."""
        store._driver.execute_query.return_value = ([], None, None)
        await store.search_vector([0.1, 0.2], session_id="agent-1")
        call = store._driver.execute_query.await_args
        assert "node.session_id = $session_id" in call.args[0]
        assert call.kwargs["session_id"] == "agent-1"
        assert "node.session_id = $session_id AND node.valid_to IS NULL" in call.args[0]

    async def test_search_traversal_rejects_invalid_depth(self, store: GraphStore) -> None:
        """Traversal depth must be bounded before being interpolated into Cypher."""
        with pytest.raises(ValueError, match="depth"):
            await store.search_traversal("Alice", depth=1000)


# ------------------------------------------------------------------
# Helper tests
# ------------------------------------------------------------------

class TestHelpers:
    """Tests for internal helper functions."""

    def test_record_to_entity(self) -> None:
        """_record_to_entity should map Neo4j properties correctly."""
        node = _make_node(
            session_id="agent-1",
            name="Alice",
            entity_type="user",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            source_event_seq=7,
            source_event_hash="d" * 64,
            source_event_type="goal.created",
            extra="value",
        )

        entity = _record_to_entity(node)
        assert entity.session_id == "agent-1"
        assert entity.name == "Alice"
        assert entity.entity_type == "user"
        assert entity.valid_to is None
        assert entity.properties == {
            "source_event_seq": 7,
            "source_event_hash": "d" * 64,
            "source_event_type": "goal.created",
            "extra": "value",
        }

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
        await gs._driver.execute_query("MATCH (n) DETACH DELETE n")
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

    async def test_source_citation_projection(self, real_store: GraphStore) -> None:
        """Source-backed entities should be inspectable through CITES_SOURCE paths."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="docs/guide.md:4-8",
                    entity_type="document",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="Document chunk",
                    properties={
                        "source_path": "docs/guide.md",
                        "source_start_line": 4,
                        "source_end_line": 8,
                        "source_sha256": "abc123",
                    },
                )
            ],
            edges=[],
            source_event_seq=11,
            source_event_hash="d" * 64,
            source_event_type="document.indexed",
        )

        await real_store.upsert_extraction(result, session_id="agent-1")

        records, _, _ = await real_store._driver.execute_query(
            """
            MATCH (:Event {session_id: $session_id, seq: $seq})-[:CITES_SOURCE]->(src:Source)
            MATCH (:Entity {session_id: $session_id, name: $name})-[citation:CITES_SOURCE]->(src)
            RETURN src.path AS path,
                   src.sha256 AS sha256,
                   citation.source_start_line AS source_start_line,
                   citation.source_end_line AS source_end_line
            """,
            session_id="agent-1",
            seq=11,
            name="docs/guide.md:4-8",
        )

        assert records == [
            {
                "path": "docs/guide.md",
                "sha256": "abc123",
                "source_start_line": 4,
                "source_end_line": 8,
            }
        ]

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

    async def test_reassertion_creates_temporal_versions(self, real_store: GraphStore) -> None:
        """Reasserting an entity should preserve old and new versions."""
        first = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Fact",
                    entity_type="fact",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="old",
                )
            ],
            edges=[],
            source_event_seq=1,
        )
        second = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Fact",
                    entity_type="fact",
                    observed_at="2024-06-01T00:00:00Z",
                    summary="new",
                )
            ],
            edges=[],
            source_event_seq=2,
        )

        await real_store.upsert_extraction(first)
        await real_store.upsert_extraction(second)

        current = await real_store.search_exact("Fact")
        before = await real_store.search_exact(
            "Fact",
            temporal_point="2024-03-01T00:00:00Z",
        )
        after = await real_store.search_exact(
            "Fact",
            temporal_point="2024-07-01T00:00:00Z",
        )

        assert len(current) == 1
        assert current[0].properties["summary"] == "new"
        assert len(before) == 1
        assert before[0].properties["summary"] == "old"
        assert before[0].valid_to is not None
        assert len(after) == 1
        assert after[0].properties["summary"] == "new"

    async def test_reasserted_entity_preserves_current_logical_relationships(
        self,
        real_store: GraphStore,
    ) -> None:
        """Current traversal should cross relationships inherited by a new version."""
        proposed = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="graph-goal-0001",
                    entity_type="goal",
                    observed_at="2024-03-01T00:00:00Z",
                ),
                ExtractedEntity(
                    name="graph-task-0001",
                    entity_type="task",
                    observed_at="2024-03-01T00:00:00Z",
                    summary="Implementation task for graph-goal-0001.",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="graph-goal-0001",
                    target="graph-task-0001",
                    relation_type="has_task",
                    valid_from="2024-03-01T00:00:00Z",
                )
            ],
            source_event_seq=1,
        )
        completed = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="graph-task-0001",
                    entity_type="task",
                    observed_at="2024-03-02T00:00:00Z",
                    summary="Completion recorded.",
                ),
                ExtractedEntity(
                    name="graph-finisher-0001",
                    entity_type="actor",
                    observed_at="2024-03-02T00:00:00Z",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="graph-finisher-0001",
                    target="graph-task-0001",
                    relation_type="completed_task",
                    valid_from="2024-03-02T00:00:00Z",
                )
            ],
            source_event_seq=2,
        )

        await real_store.upsert_extraction(proposed)
        await real_store.upsert_extraction(completed)

        results = await real_store.search_traversal("graph-goal-0001", depth=2)
        names = {entity.name for entity in results}

        assert "graph-task-0001" in names
        assert "graph-finisher-0001" in names

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
        from datetime import datetime

        from zaxy.event import EventLog
        from zaxy.extract import extract
        log = EventLog("/tmp/zaxy_pipeline_test.jsonl")
        ts = datetime(2024, 1, 1, tzinfo=UTC)
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


@pytest.mark.integration
class TestTLSIntegration:
    """Integration tests against a TLS-enabled Neo4j instance."""

    async def test_connects_with_custom_ca_over_bolt_s(self) -> None:
        """GraphStore should connect to TLS Neo4j using the generated CA."""
        gs = GraphStore(
            "bolt://localhost:7689",
            "neo4j",
            "testpassword",
            ca_cert=".certs/ca.crt",
        )
        await gs.connect()
        assert gs._driver is not None
        records, _, _ = await gs._driver.execute_query("RETURN 1 AS n")
        assert records[0]["n"] == 1
        await gs.close()
