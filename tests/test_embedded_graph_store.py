"""Tests for the embedded graph projection backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from zaxy.embedded_graph_store import EmbeddedGraphStore
from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult

pytestmark = pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")


@pytest.mark.asyncio
async def test_embedded_store_projects_entities_and_traversal_edges(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Goal A",
                entity_type="goal",
                observed_at="2026-05-20T01:00:00Z",
                summary="ship memory",
            ),
            ExtractedEntity(
                name="Task B",
                entity_type="task",
                observed_at="2026-05-20T01:00:00Z",
                summary="write tests",
            ),
        ],
        edges=[
            ExtractedEdge(
                source="Task B",
                target="Goal A",
                relation_type="supports",
                valid_from="2026-05-20T01:00:00Z",
            )
        ],
        source_event_seq=1,
        source_event_hash="abc",
        source_event_type="task.proposed",
        source_thread="agent-1",
    )

    await store.upsert_extraction(result, session_id="agent-1")

    exact = await store.search_exact("Goal A", session_id="agent-1")
    keyword = await store.search_keyword("ship", session_id="agent-1")
    neighbors = await store.search_traversal("Task B", depth=1, session_id="agent-1")

    assert [entity.name for entity in exact] == ["Goal A"]
    assert exact[0].properties["source_event_seq"] == 1
    assert exact[0].properties["source_event_hash"] == "abc"
    assert [result.entity.name for result in keyword] == ["Goal A"]
    assert keyword[0].entity.properties["source_event_seq"] == 1
    assert [entity.name for entity in neighbors] == ["Goal A"]
    assert neighbors[0].properties["source_event_hash"] == "abc"
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_event_status_handles_missing_projection_schema(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=3,
        eventloom_latest_hash="hash-3",
    )

    assert status.session_id == "agent-1"
    assert status.event_count == 0
    assert status.latest_seq is None
    assert status.eventloom_latest_seq == 3
    assert status.eventloom_latest_hash == "hash-3"
    assert status.projection_lag == 3
    assert status.latest_hash_matches is False
    assert status.integrity_ok is False
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_inferred_status_handles_missing_projection_schema(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()

    status = await store.inspect_inferred_edge_status("agent-1")

    assert status.session_id == "agent-1"
    assert status.total_edges == 0
    assert status.method_count == 0
    assert status.evidence_count == 0
    assert status.missing_evidence_count == 0
    assert status.missing_source_event_count == 0
    assert status.evidence_coverage == 1.0
    assert status.methods == ()
    assert status.samples == ()
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_keyword_search_prioritizes_rare_phrase_matches(tmp_path: Path) -> None:
    """Keyword search should not let common question words bury answer-bearing phrases."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="common-words",
                    entity_type="document",
                    observed_at="2026-05-20T01:00:00Z",
                    summary=(
                        "Which group did I join first was the question I kept asking "
                        "while comparing several professional communities."
                    ),
                ),
                ExtractedEntity(
                    name="answer-bearing",
                    entity_type="document",
                    observed_at="2026-05-20T01:01:00Z",
                    summary="I joined the Page Turners book club before the other networking group.",
                ),
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    results = await store.search_keyword(
        "Which group did I join first, Page Turners or Marketing Professionals?",
        limit=2,
        session_id="agent-1",
    )

    assert [result.entity.name for result in results] == ["answer-bearing", "common-words"]
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_traversal_matches_neo4j_undirected_paths(tmp_path: Path) -> None:
    """Traversal should cross incoming and outgoing edges like the Neo4j control backend."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Graph Goal", entity_type="goal", observed_at="2026-05-20T01:00:00Z"),
                ExtractedEntity(name="Graph Task", entity_type="task", observed_at="2026-05-20T01:00:00Z"),
                ExtractedEntity(name="Finisher", entity_type="actor", observed_at="2026-05-20T01:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Graph Goal",
                    target="Graph Task",
                    relation_type="has_task",
                    valid_from="2026-05-20T01:00:00Z",
                ),
                ExtractedEdge(
                    source="Finisher",
                    target="Graph Task",
                    relation_type="completed_task",
                    valid_from="2026-05-20T01:00:00Z",
                ),
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    neighbors = await store.search_traversal("Graph Goal", depth=2, session_id="agent-1")

    assert {entity.name for entity in neighbors} == {"Graph Task", "Finisher"}
    finisher = next(entity for entity in neighbors if entity.name == "Finisher")
    assert finisher.properties["_path_relation_types"] == ["has_task", "completed_task"]
    assert finisher.properties["_path_length"] == 2
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_caches_traversal_index_and_invalidates_on_projection(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Root", entity_type="goal", observed_at="2026-05-20T01:00:00Z"),
                ExtractedEntity(name="First", entity_type="task", observed_at="2026-05-20T01:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Root",
                    target="First",
                    relation_type="has_task",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    first = await store.search_traversal("Root", depth=1, session_id="agent-1")
    assert [entity.name for entity in first] == ["First"]
    assert "agent-1" in store._traversal_index_cache

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Root", entity_type="goal", observed_at="2026-05-20T01:01:00Z"),
                ExtractedEntity(name="Second", entity_type="task", observed_at="2026-05-20T01:01:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Root",
                    target="Second",
                    relation_type="has_task",
                    valid_from="2026-05-20T01:01:00Z",
                )
            ],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )
    assert "agent-1" not in store._traversal_index_cache

    second = await store.search_traversal("Root", depth=1, session_id="agent-1")
    assert {entity.name for entity in second} == {"First", "Second"}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_preserves_active_relationships_across_entity_reassertion(tmp_path: Path) -> None:
    """Reasserting an entity should not sever active graph paths from prior events."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Graph Goal", entity_type="goal", observed_at="2026-05-20T01:00:00Z"),
                ExtractedEntity(name="Graph Task", entity_type="task", observed_at="2026-05-20T01:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Graph Goal",
                    target="Graph Task",
                    relation_type="has_task",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Graph Task", entity_type="task", observed_at="2026-05-20T02:00:00Z"),
                ExtractedEntity(name="Finisher", entity_type="actor", observed_at="2026-05-20T02:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Finisher",
                    target="Graph Task",
                    relation_type="completed_task",
                    valid_from="2026-05-20T02:00:00Z",
                )
            ],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )

    neighbors = await store.search_traversal("Graph Goal", depth=2, session_id="agent-1")

    assert {entity.name for entity in neighbors} == {"Graph Task", "Finisher"}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reuses_identical_active_entity_without_reversioning(tmp_path: Path) -> None:
    """Stable helper nodes should not be re-versioned for every projected document chunk."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    session = ExtractedEntity(
        name="longmemeval:session:s1",
        entity_type="longmemeval_session",
        observed_at="2024-01-01T00:00:00Z",
        summary="2024/01/01",
        properties={"longmemeval_session_id": "s1"},
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                session,
                ExtractedEntity(
                    name="chunk-1",
                    entity_type="document",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="first chunk",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source=session.name,
                    target="chunk-1",
                    relation_type="has_document_chunk",
                    valid_from="2024-01-01T00:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                session,
                ExtractedEntity(
                    name="chunk-2",
                    entity_type="document",
                    observed_at="2024-01-01T00:01:00Z",
                    summary="second chunk",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source=session.name,
                    target="chunk-2",
                    relation_type="has_document_chunk",
                    valid_from="2024-01-01T00:01:00Z",
                )
            ],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )

    rows = store._require_connection().execute(
        """
        MATCH (e:Entity)
        WHERE e.session_id = 'agent-1'
          AND e.name = 'longmemeval:session:s1'
          AND e.entity_type = 'longmemeval_session'
        RETURN count(e)
        """
    ).get_all()
    neighbors = await store.search_traversal("longmemeval:session:s1", depth=1, session_id="agent-1")

    assert rows[0][0] == 1
    assert {neighbor.name for neighbor in neighbors} == {"chunk-1", "chunk-2"}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_bulk_projection_transaction_commits_events(tmp_path: Path) -> None:
    """Bulk projection should group Kuzu writes without changing committed graph state."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.begin_bulk_projection()
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Bulk Goal",
                    entity_type="goal",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="bulk projection",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    await store.commit_bulk_projection()

    exact = await store.search_exact("Bulk Goal", session_id="agent-1")

    assert [entity.name for entity in exact] == ["Bulk Goal"]
    assert "agent-1" in store._keyword_index_cache
    assert ("agent-1", None) in store._vector_index_cache
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_versions_reasserted_entities_and_invalidates(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Policy",
                    entity_type="decision",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="old rule",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Policy",
                    entity_type="decision",
                    observed_at="2026-05-20T02:00:00Z",
                    summary="new rule",
                )
            ],
            edges=[],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )

    current = await store.search_exact("Policy", entity_type="decision", session_id="agent-1")
    assert len(current) == 1
    assert current[0].valid_from == "2026-05-20T02:00:00Z"
    assert current[0].properties["summary"] == "new rule"

    previous = await store.search_exact(
        "Policy",
        entity_type="decision",
        temporal_point="2026-05-20T01:30:00Z",
        session_id="agent-1",
    )
    assert len(previous) == 1
    assert previous[0].valid_from == "2026-05-20T01:00:00Z"
    assert previous[0].valid_to == "2026-05-20T02:00:00Z"
    assert previous[0].properties["summary"] == "old rule"

    await store.invalidate_entity(
        "Policy",
        "decision",
        invalid_at="2026-05-20T03:00:00Z",
        session_id="agent-1",
    )

    assert await store.search_exact("Policy", entity_type="decision", session_id="agent-1") == []
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reports_event_projection_status(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Goal A",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
            source_event_prev_hash=None,
        ),
        session_id="agent-1",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Goal B",
                    entity_type="goal",
                    observed_at="2026-05-20T02:00:00Z",
                )
            ],
            edges=[],
            source_event_seq=2,
            source_event_hash="hash-2",
            source_event_prev_hash="hash-1",
        ),
        session_id="agent-1",
    )

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=2,
        eventloom_latest_hash="hash-2",
    )

    assert status.event_count == 2
    assert status.latest_seq == 2
    assert status.latest_hash == "hash-2"
    assert status.projection_lag == 0
    assert status.latest_hash_matches is True
    assert status.integrity_ok is True
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reports_missing_schema_as_empty_projection(tmp_path: Path) -> None:
    """Read-only status should not crash before the embedded projection is initialized."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=2,
        eventloom_latest_hash="hash-2",
    )

    assert status.event_count == 0
    assert status.latest_seq is None
    assert status.latest_hash is None
    assert status.projection_lag == 2
    assert status.latest_hash_matches is False
    assert status.integrity_ok is False
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_searches_entity_vectors(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Vector Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="closest vector",
                    embedding=vector_a,
                ),
                ExtractedEntity(
                    name="Distant Task",
                    entity_type="task",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="far vector",
                    embedding=vector_b,
                ),
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    results = await store.search_vector(vector_a, limit=2, session_id="agent-1")

    assert [result.entity.name for result in results] == ["Vector Goal", "Distant Task"]
    assert results[0].source == "vector"
    assert results[0].score > results[1].score
    assert results[0].entity.properties["source_event_hash"] == "hash-1"

    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_caches_vector_index_and_invalidates_on_projection(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Original Vector",
                    entity_type="memory",
                    observed_at="2026-05-20T01:00:00Z",
                    embedding=[1.0, 0.0],
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    first = await store.search_vector([1.0, 0.0], limit=1, session_id="agent-1")
    assert [result.entity.name for result in first] == ["Original Vector"]
    assert ("agent-1", None) in store._vector_index_cache
    assert store._vector_index_cache[("agent-1", None)].sparse_vectors == [{0: 1.0}]

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="New Vector",
                    entity_type="memory",
                    observed_at="2026-05-20T01:01:00Z",
                    embedding=[1.0, 0.0],
                )
            ],
            edges=[],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )
    assert ("agent-1", None) not in store._vector_index_cache

    second = await store.search_vector([1.0, 0.0], limit=2, session_id="agent-1")
    assert {result.entity.name for result in second} == {"Original Vector", "New Vector"}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reports_inferred_edge_status(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Decision A",
                    entity_type="decision",
                    observed_at="2026-05-20T01:00:00Z",
                ),
                ExtractedEntity(
                    name="Source B",
                    entity_type="source",
                    observed_at="2026-05-20T01:00:00Z",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Decision A",
                    target="Source B",
                    relation_type="supported_by",
                    valid_from="2026-05-20T01:00:00Z",
                    inferred=True,
                    confidence=0.8,
                    inference_method="cited_decision",
                    evidence={"source_event_seq": 7, "source_event_hash": "hash-7", "quote": "because"},
                )
            ],
            source_event_seq=7,
            source_event_hash="hash-7",
        ),
        session_id="agent-1",
    )

    status = await store.inspect_inferred_edge_status("agent-1")

    assert status.total_edges == 1
    assert status.method_count == 1
    assert status.evidence_count == 1
    assert status.evidence_coverage == 1.0
    assert status.methods[0].method == "cited_decision"
    assert status.methods[0].relation_types == ("supported_by",)
    assert status.samples[0].source == "Decision A"
    assert status.samples[0].target == "Source B"
    assert status.samples[0].source_event_seq == 7
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reports_missing_schema_as_empty_inferred_status(tmp_path: Path) -> None:
    """Read-only inferred-edge audit should not crash before projection exists."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()

    status = await store.inspect_inferred_edge_status("agent-1")

    assert status.session_id == "agent-1"
    assert status.total_edges == 0
    assert status.method_count == 0
    assert status.evidence_count == 0
    assert status.missing_evidence_count == 0
    assert status.missing_source_event_count == 0
    assert status.evidence_coverage == 1.0
    assert status.methods == ()
    assert status.samples == ()
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reset_rebuilds_projection_artifact(tmp_path: Path) -> None:
    graph_path = tmp_path / "embedded.kuzu"
    store = EmbeddedGraphStore(graph_path)
    await store.connect()
    await store.init_schema()
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Temporary",
                    entity_type="fact",
                    observed_at="2026-05-20T01:00:00Z",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    assert await store.search_exact("Temporary", session_id="agent-1")

    await store.reset_benchmark_projection()

    assert graph_path.exists()
    assert await store.search_exact("Temporary", session_id="agent-1") == []
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_retires_source_projections_and_relationships(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="docs/guide.md",
                    entity_type="source",
                    observed_at="2026-05-20T01:00:00Z",
                    properties={"source_path": "docs/guide.md"},
                ),
                ExtractedEntity(
                    name="Guide Claim",
                    entity_type="fact",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="claim from retired source",
                    properties={"source_path": "docs/guide.md"},
                ),
                ExtractedEntity(
                    name="Stable Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="goal remains active",
                    properties={"source_path": "docs/other.md"},
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Guide Claim",
                    target="Stable Goal",
                    relation_type="supports",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    assert await store.search_exact("Guide Claim", entity_type="fact", session_id="agent-1")
    assert [entity.name for entity in await store.search_traversal("Guide Claim", depth=1, session_id="agent-1")] == [
        "Stable Goal"
    ]

    await store.retire_source_projections(
        source_path="docs/guide.md",
        invalid_at="2026-05-20T02:00:00Z",
        session_id="agent-1",
    )

    assert await store.search_exact("Guide Claim", entity_type="fact", session_id="agent-1") == []
    assert await store.search_exact("Stable Goal", entity_type="goal", session_id="agent-1")
    assert await store.search_traversal("Guide Claim", depth=1, session_id="agent-1") == []

    historical = await store.search_exact(
        "Guide Claim",
        entity_type="fact",
        temporal_point="2026-05-20T01:30:00Z",
        session_id="agent-1",
    )
    assert len(historical) == 1
    assert historical[0].valid_to == "2026-05-20T02:00:00Z"
    await store.close()
