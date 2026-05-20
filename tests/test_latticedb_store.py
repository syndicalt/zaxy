"""Tests for the LatticeDB projection backend candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult
from zaxy.latticedb_store import LatticeDBStore


@pytest.mark.asyncio
async def test_latticedb_store_fails_clearly_when_optional_dependency_is_missing(tmp_path: Path) -> None:
    """The candidate backend should be import-safe even without the package installed."""
    if importlib.util.find_spec("latticedb") is not None:
        pytest.skip("latticedb is installed")
    store = LatticeDBStore(tmp_path / "memory.latticedb")

    with pytest.raises(RuntimeError, match='zaxy-memory\\[latticedb\\]'):
        await store.connect()


@pytest.mark.skipif(importlib.util.find_spec("latticedb") is None, reason="latticedb is not installed")
@pytest.mark.asyncio
async def test_latticedb_store_projects_entities_and_traversal_edges(tmp_path: Path) -> None:
    store = LatticeDBStore(tmp_path / "memory.latticedb")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
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
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    exact = await store.search_exact("Goal A", session_id="agent-1")
    keyword = await store.search_keyword("ship", session_id="agent-1")
    neighbors = await store.search_traversal("Task B", depth=1, session_id="agent-1")

    assert [entity.name for entity in exact] == ["Goal A"]
    assert exact[0].properties["source_event_seq"] == 1
    assert exact[0].properties["source_event_hash"] == "hash-1"
    assert [result.entity.name for result in keyword] == ["Goal A"]
    assert [entity.name for entity in neighbors] == ["Goal A"]
    await store.close()


@pytest.mark.skipif(importlib.util.find_spec("latticedb") is None, reason="latticedb is not installed")
@pytest.mark.asyncio
async def test_latticedb_store_reset_recreates_projection_artifact(tmp_path: Path) -> None:
    graph_path = tmp_path / "memory.latticedb"
    graph_path.write_text("stale projection", encoding="utf-8")
    store = LatticeDBStore(graph_path)

    await store.reset_benchmark_projection()

    await store.init_schema()
    assert store._database is not None
    await store.close()


@pytest.mark.skipif(importlib.util.find_spec("latticedb") is None, reason="latticedb is not installed")
@pytest.mark.asyncio
async def test_latticedb_store_retires_source_projections_and_incident_edges(tmp_path: Path) -> None:
    store = LatticeDBStore(tmp_path / "memory.latticedb")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Retired Source",
                    entity_type="document",
                    observed_at="2026-05-20T01:00:00Z",
                    properties={"source_path": "docs/old.md"},
                ),
                ExtractedEntity(
                    name="Dependent Task",
                    entity_type="task",
                    observed_at="2026-05-20T01:00:00Z",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Retired Source",
                    target="Dependent Task",
                    relation_type="supports",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    assert await store.has_traversal_edges(session_id="agent-1") is True

    await store.retire_source_projections(
        source_path="docs/old.md",
        invalid_at="2026-05-20T02:00:00Z",
        session_id="agent-1",
    )

    assert await store.search_exact("Retired Source", session_id="agent-1") == []
    assert await store.search_traversal("Retired Source", depth=1, session_id="agent-1") == []
    assert await store.has_traversal_edges(session_id="agent-1") is False
    await store.close()


@pytest.mark.skipif(importlib.util.find_spec("latticedb") is None, reason="latticedb is not installed")
@pytest.mark.asyncio
async def test_latticedb_store_versions_and_invalidates_entities(tmp_path: Path) -> None:
    store = LatticeDBStore(tmp_path / "memory.latticedb")
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
    assert previous[0].valid_to == "2026-05-20T02:00:00Z"

    await store.invalidate_entity(
        "Policy",
        "decision",
        invalid_at="2026-05-20T03:00:00Z",
        session_id="agent-1",
    )

    assert await store.search_exact("Policy", entity_type="decision", session_id="agent-1") == []
    await store.close()


@pytest.mark.skipif(importlib.util.find_spec("latticedb") is None, reason="latticedb is not installed")
@pytest.mark.asyncio
async def test_latticedb_store_searches_entity_vectors(tmp_path: Path) -> None:
    store = LatticeDBStore(tmp_path / "memory.latticedb", vector_dimensions=128)
    await store.connect()
    await store.init_schema()

    vector_a = [1.0, *([0.0] * 127)]
    vector_b = [0.0, 1.0, *([0.0] * 126)]
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


@pytest.mark.skipif(importlib.util.find_spec("latticedb") is None, reason="latticedb is not installed")
@pytest.mark.asyncio
async def test_latticedb_store_delegates_keyword_search_to_native_fts(tmp_path: Path) -> None:
    store = LatticeDBStore(tmp_path / "memory.latticedb")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Native FTS Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="alpha memory graph",
                ),
                ExtractedEntity(
                    name="Distractor",
                    entity_type="task",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="beta unrelated",
                ),
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    results = await store.search_keyword("memory graph", limit=2, session_id="agent-1")

    assert [result.entity.name for result in results] == ["Native FTS Goal"]
    assert results[0].source == "keyword"
    assert results[0].raw_score is not None
    assert results[0].raw_score != 2.0
    await store.close()


@pytest.mark.skipif(importlib.util.find_spec("latticedb") is None, reason="latticedb is not installed")
@pytest.mark.asyncio
async def test_latticedb_store_reports_inferred_edge_audit_status(tmp_path: Path) -> None:
    store = LatticeDBStore(tmp_path / "memory.latticedb")
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
    assert status.samples[0].source_event_hash == "hash-7"
    await store.close()
