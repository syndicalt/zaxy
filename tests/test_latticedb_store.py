"""Tests for the LatticeDB projection backend candidate."""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest

from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult
from zaxy.graph import GraphEntity
from zaxy.latticedb_store import (
    LatticeDBStore,
    _edge_citation,
    _fts_text,
    _json_dict,
    _optional_float,
    _optional_int,
    _properties_reference_source,
)


class _Edge:
    def __init__(self, edge_id: int, source_id: int, target_id: int) -> None:
        self.id = edge_id
        self.source_id = source_id
        self.target_id = target_id


@pytest.mark.asyncio
async def test_latticedb_store_fails_clearly_when_optional_dependency_is_missing(tmp_path: Path) -> None:
    """The candidate backend should be import-safe even without the package installed."""
    if importlib.util.find_spec("latticedb") is not None:
        pytest.skip("latticedb is installed")
    store = LatticeDBStore(tmp_path / "memory.latticedb")

    with pytest.raises(RuntimeError, match='zaxy-memory\\[latticedb\\]'):
        await store.connect()


@pytest.mark.asyncio
async def test_latticedb_store_keyword_fallback_converts_score_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback keyword scoring should reuse one float conversion for score and raw_score."""
    store = LatticeDBStore(Path("unused.latticedb"))
    entity = GraphEntity(
        name="Fallback Goal",
        entity_type="goal",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={"summary": "memory graph"},
        session_id="agent-1",
    )
    original_float = builtins.float
    float_calls = 0

    def tracking_float(value: object) -> float:
        nonlocal float_calls
        float_calls += 1
        return original_float(value)

    monkeypatch.setattr(store, "_search_keyword_native", lambda *args, **kwargs: [])
    monkeypatch.setattr(store, "_entity_node_ids", lambda: [1])
    monkeypatch.setattr(
        store,
        "_node_property",
        lambda node_id, property_name: {
            "session_id": "agent-1",
            "name": entity.name,
            "entity_type": entity.entity_type,
            "summary": entity.properties["summary"],
        }[property_name],
    )
    monkeypatch.setattr(store, "_is_visible_at", lambda node_id, temporal_point: True)
    monkeypatch.setattr(store, "_entity_from_node_id", lambda node_id: entity)
    monkeypatch.setattr(builtins, "float", tracking_float)

    results = await store.search_keyword("memory", session_id="agent-1")

    assert [result.entity.name for result in results] == ["Fallback Goal"]
    assert results[0].score == 1.0
    assert results[0].raw_score == 1.0
    assert float_calls == 1


@pytest.mark.asyncio
async def test_latticedb_store_keyword_zero_limit_skips_projection_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-result keyword queries should not call native FTS or scan fallback nodes."""
    store = LatticeDBStore(Path("unused.latticedb"))

    def fail_keyword_native(*args: object, **kwargs: object) -> list[object]:
        raise AssertionError("zero-limit keyword query should not call native FTS")

    def fail_entity_node_ids() -> list[object]:
        raise AssertionError("zero-limit keyword query should not scan fallback nodes")

    monkeypatch.setattr(store, "_search_keyword_native", fail_keyword_native)
    monkeypatch.setattr(store, "_entity_node_ids", fail_entity_node_ids)

    assert await store.search_keyword("memory graph", limit=0, session_id="agent-1") == []


@pytest.mark.asyncio
async def test_latticedb_store_vector_zero_limit_skips_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-result vector queries should not require a database read transaction."""
    store = LatticeDBStore(Path("unused.latticedb"))

    def fail_database() -> object:
        raise AssertionError("zero-limit vector query should not require LatticeDB")

    monkeypatch.setattr(store, "_require_database", fail_database)

    assert await store.search_vector([1.0, 0.0], limit=0, session_id="agent-1") == []


@pytest.mark.asyncio
async def test_latticedb_store_vector_zero_norm_skips_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-norm vector queries should not require a database read transaction."""
    store = LatticeDBStore(Path("unused.latticedb"))

    def fail_database() -> object:
        raise AssertionError("zero-norm vector query should not require LatticeDB")

    monkeypatch.setattr(store, "_require_database", fail_database)

    assert await store.search_vector([0.0, 0.0], session_id="agent-1") == []


@pytest.mark.asyncio
async def test_latticedb_store_causal_neighbor_search_preserves_edge_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LatticeDBStore(Path("unused.latticedb"))
    nodes = {
        1: GraphEntity(
            name="root cause",
            entity_type="issue",
            valid_from="2026-06-07T00:00:00Z",
            valid_to=None,
            properties={},
            session_id="agent-1",
        ),
        2: GraphEntity(
            name="first effect",
            entity_type="outcome",
            valid_from="2026-06-07T00:00:00Z",
            valid_to=None,
            properties={},
            session_id="agent-1",
        ),
        3: GraphEntity(
            name="second effect",
            entity_type="outcome",
            valid_from="2026-06-07T00:00:00Z",
            valid_to=None,
            properties={},
            session_id="agent-1",
        ),
    }
    node_properties = {
        1: {"session_id": "agent-1", "name": "root cause", "valid_to": "", "valid_from": "2026-06-07T00:00:00Z"},
        2: {"session_id": "agent-1", "name": "first effect", "valid_to": "", "valid_from": "2026-06-07T00:00:00Z"},
        3: {"session_id": "agent-1", "name": "second effect", "valid_to": "", "valid_from": "2026-06-07T00:00:00Z"},
    }
    edge_properties = {
        10: {
            "session_id": "agent-1",
            "relation_type": "causal_caused",
            "valid_from": "2026-06-07T00:00:00Z",
            "valid_to": "",
            "evidence_json": (
                '{"authority_status":"non_authoritative",'
                '"causal_relation_type":"caused","review_status":"proposed"}'
            ),
            "source_event_seq": 42,
            "source_event_hash": "a" * 64,
            "confidence": 0.8,
            "inference_method": "explicit_outcome_explanation_v1",
        },
        11: {
            "session_id": "agent-1",
            "relation_type": "causal_enabled",
            "valid_from": "2026-06-07T00:00:00Z",
            "valid_to": "",
            "evidence_json": "{}",
            "source_event_seq": 43,
            "source_event_hash": "b" * 64,
            "confidence": -1.0,
            "inference_method": "",
        },
    }
    outgoing = {1: [_Edge(10, 1, 2)], 2: [_Edge(11, 2, 3)], 3: []}

    monkeypatch.setattr(store, "_entity_node_ids", lambda: [1, 2, 3])
    monkeypatch.setattr(store, "_node_property", lambda node_id, key: node_properties[node_id].get(key))
    monkeypatch.setattr(store, "_edge_property", lambda edge_id, key: edge_properties[edge_id].get(key))
    monkeypatch.setattr(store, "_outgoing_edges", lambda node_id: outgoing[node_id])
    monkeypatch.setattr(store, "_incoming_edges", lambda node_id: [])
    monkeypatch.setattr(store, "_entity_from_node_id", lambda node_id: nodes[node_id])

    results = await store.search_causal_neighbors(
        "root cause",
        direction="successors",
        depth=3,
        session_id="agent-1",
    )

    assert [entity.name for entity in results] == ["first effect", "second effect"]
    assert results[0].properties["citation"] == "eventloom://agent-1/events/42#aaaaaaaaaaaa"
    assert results[0].properties["_path_relation_types"] == ["causal_caused"]
    assert results[0].properties["_path_length"] == 1
    assert results[1].properties["confidence"] == 1.0
    assert results[1].properties["_path_relation_types"] == ["causal_caused", "causal_enabled"]


@pytest.mark.asyncio
async def test_latticedb_store_causal_neighbor_search_rejects_invalid_direction() -> None:
    store = LatticeDBStore(Path("unused.latticedb"))

    with pytest.raises(ValueError, match="direction"):
        await store.search_causal_neighbors("root cause", direction="sideways")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_latticedb_store_event_projection_status_reports_lag_and_chain_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LatticeDBStore(Path("unused.latticedb"))
    node_properties = {
        100: {"session_id": "agent-1", "seq": 1, "hash": "a" * 64, "prev_hash": ""},
        101: {"session_id": "agent-1", "seq": 2, "hash": "b" * 64, "prev_hash": "missing"},
        102: {"session_id": "other", "seq": 3, "hash": "c" * 64, "prev_hash": ""},
    }

    monkeypatch.setattr(store, "_event_node_ids", lambda: [101, 100, 102])
    monkeypatch.setattr(store, "_node_property", lambda node_id, key: node_properties[node_id].get(key))

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=3,
        eventloom_latest_hash="expected-latest",
    )

    assert status.session_id == "agent-1"
    assert status.event_count == 2
    assert status.latest_seq == 2
    assert status.projection_lag == 1
    assert status.latest_hash_matches is False
    assert status.missing_chain_links == 1
    assert status.integrity_ok is False


def test_latticedb_store_entity_from_node_id_merges_projection_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LatticeDBStore(Path("unused.latticedb"))
    properties = {
        "properties_json": '{"source_path":"docs/release.md"}',
        "summary": "Release note",
        "source_event_seq": "42",
        "source_event_hash": "a" * 64,
        "valid_to": "",
        "name": "Release Note",
        "entity_type": "document",
        "valid_from": "2026-06-07T00:00:00Z",
        "session_id": "agent-1",
    }
    monkeypatch.setattr(store, "_node_property", lambda node_id, key: properties.get(key))

    entity = store._entity_from_node_id(7)

    assert entity.name == "Release Note"
    assert entity.valid_to is None
    assert entity.properties["summary"] == "Release note"
    assert entity.properties["source_event_seq"] == 42
    assert entity.properties["source_event_hash"] == "a" * 64


def test_latticedb_store_active_edge_at_applies_session_relation_and_temporal_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LatticeDBStore(Path("unused.latticedb"))
    edge_properties = {
        1: {
            "session_id": "agent-1",
            "relation_type": "causal_caused",
            "valid_from": "2026-06-07T00:00:00Z",
            "valid_to": "2026-06-08T00:00:00Z",
        }
    }
    monkeypatch.setattr(store, "_edge_property", lambda edge_id, key: edge_properties[edge_id].get(key))

    assert store._active_edge_at(1, "other", None, None) is False
    assert store._active_edge_at(1, "agent-1", "causal_fixed", None) is False
    assert store._active_edge_at(1, "agent-1", "causal_caused", None) is False
    assert store._active_edge_at(1, "agent-1", "causal_caused", "2026-06-07T12:00:00Z") is True
    assert store._active_edge_at(1, "agent-1", "causal_caused", "2026-06-08T00:00:00Z") is False


def test_latticedb_store_active_node_helpers_choose_latest_open_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LatticeDBStore(Path("unused.latticedb"))
    node_properties = {
        1: {
            "session_id": "agent-1",
            "entity_type": "decision",
            "name": "Policy",
            "valid_to": "",
            "valid_from": "2026-06-07T00:00:00Z",
        },
        2: {
            "session_id": "agent-1",
            "entity_type": "decision",
            "name": "Policy",
            "valid_to": "",
            "valid_from": "2026-06-08T00:00:00Z",
        },
        3: {
            "session_id": "agent-1",
            "entity_type": "decision",
            "name": "Policy",
            "valid_to": "2026-06-07T12:00:00Z",
            "valid_from": "2026-06-06T00:00:00Z",
        },
    }
    monkeypatch.setattr(store, "_entity_node_ids", lambda: [1, 2, 3])
    monkeypatch.setattr(store, "_node_property", lambda node_id, key: node_properties[node_id].get(key))

    assert store._active_node_ids("agent-1", "decision", "Policy") == [1, 2]
    assert store._active_node_id("agent-1", "decision", "Policy") == 2
    assert store._active_node_id("agent-1", "decision", "Missing") is None


def test_latticedb_store_pure_helpers_preserve_backend_contracts() -> None:
    assert _json_dict("") == {}
    assert _json_dict("not-json") == {}
    assert _json_dict("[1, 2]") == {}
    assert _json_dict('{"source_path":"docs/a.md"}') == {"source_path": "docs/a.md"}

    assert _edge_citation("agent-1", 42, "a" * 64) == "eventloom://agent-1/events/42#aaaaaaaaaaaa"
    assert _edge_citation("agent-1", None, "") == "eventloom://unknown/events/unknown#unknown"

    assert _optional_int("42") == 42
    assert _optional_int(True) is None
    assert _optional_int("bad") is None
    assert _optional_float("0.5") == 0.5
    assert _optional_float(True) is None
    assert _optional_float("bad") is None

    assert _fts_text("Entity", "document", {"embedding": [1.0], "summary": "Release"}) == (
        "Entity document Release"
    )
    assert _properties_reference_source({"target_path": "docs/a.md"}, "docs/a.md") is True
    assert _properties_reference_source({"source_path": "docs/b.md"}, "docs/a.md") is False


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
