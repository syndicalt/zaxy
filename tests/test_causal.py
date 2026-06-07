from __future__ import annotations

import importlib.util

import pytest

from zaxy.causal import (
    CAUSAL_RELATION_TYPES,
    CausalEdge,
    CausalQueryResult,
    build_causal_edge_event,
    causal_relation_to_graph_relation,
)
from zaxy.core import MemoryFabric
from zaxy.embedded_graph_store import (
    EmbeddedGraphStore,
    _causal_edge_metadata_from_row,
)
from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult
from zaxy.graph import GraphEntity
from zaxy.latticedb_store import _json_dict as _latticedb_json_dict
from zaxy.pggraph_store import (
    _causal_edge_metadata as _pggraph_causal_edge_metadata,
)
from zaxy.pggraph_store import (
    _properties_from_row as _pggraph_properties_from_row,
)


class _DirectionalCausalStore:
    def __init__(self, entities: list[GraphEntity] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.entities = entities

    async def search_causal_neighbors(
        self,
        entity_name: str,
        *,
        direction: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        self.calls.append(
            {
                "entity_name": entity_name,
                "direction": direction,
                "relation_type": relation_type,
                "depth": depth,
                "temporal_point": temporal_point,
                "session_id": session_id,
            }
        )
        if self.entities is not None:
            return self.entities
        if direction == "successors":
            return [
                _causal_entity(
                    name="effect",
                    entity_type="outcome",
                    source_name=entity_name,
                    source_type="event",
                    target_name="effect",
                    target_type="outcome",
                    relation_type="caused",
                    citation="eventloom://agent-1/events/42#aaaaaaaaaaaa",
                )
            ]
        if direction == "predecessors":
            return [
                _causal_entity(
                    name="cause",
                    entity_type="event",
                    source_name="cause",
                    source_type="event",
                    target_name=entity_name,
                    target_type="outcome",
                    relation_type="caused",
                    citation="eventloom://agent-1/events/43#bbbbbbbbbbbb",
                )
            ]
        raise AssertionError(f"unexpected direction: {direction}")


def _fabric_with_graph(graph: object) -> MemoryFabric:
    fabric = MemoryFabric.__new__(MemoryFabric)
    fabric.graph = graph
    return fabric


def _causal_entity(
    *,
    name: str,
    entity_type: str,
    source_name: str,
    source_type: str,
    target_name: str,
    target_type: str,
    relation_type: str,
    citation: str,
) -> GraphEntity:
    return GraphEntity(
        name=name,
        entity_type=entity_type,
        valid_from="2026-06-07T00:00:00Z",
        valid_to=None,
        session_id="agent-1",
        properties={
            "causal_source_name": source_name,
            "causal_source_type": source_type,
            "causal_target_name": target_name,
            "causal_target_type": target_type,
            "causal_relation_type": relation_type,
            "relation_type": causal_relation_to_graph_relation(relation_type),
            "confidence": 0.91,
            "inference_method": "explicit_outcome_citation_v1",
            "citation": citation,
            "review_status": "proposed",
            "authority_status": "non_authoritative",
            "evidence": {"source_event_seq": 42, "source_event_hash": "a" * 64},
            "_path_length": 1,
            "_path_relation_types": [causal_relation_to_graph_relation(relation_type)],
        },
    )


def _backend_causal_entity(**overrides: object) -> GraphEntity:
    properties: dict[str, object] = {
        "causal_source_name": "cause",
        "causal_source_type": "event",
        "causal_target_name": "effect",
        "causal_target_type": "outcome",
        "causal_relation_type": "caused",
        "relation_type": "causal_caused",
        "confidence": 0.91,
        "inference_method": "explicit_outcome_citation_v1",
        "review_status": "proposed",
        "authority_status": "non_authoritative",
        "source_event_seq": 42,
        "source_event_hash": "a" * 64,
        "session_id": "agent-1",
    }
    properties.update(overrides)
    return GraphEntity(
        name="effect",
        entity_type="outcome",
        valid_from="2026-06-07T00:00:00Z",
        valid_to=None,
        properties=properties,
        session_id="agent-1",
    )


def test_causal_relation_taxonomy_is_stable() -> None:
    expected_relation_types = {
        "caused",
        "enabled",
        "blocked",
        "prevented",
        "regressed",
        "fixed",
        "explained",
    }
    assert expected_relation_types == CAUSAL_RELATION_TYPES
    assert causal_relation_to_graph_relation("caused") == "causal_caused"
    assert causal_relation_to_graph_relation("fixed") == "causal_fixed"


@pytest.mark.asyncio
async def test_memory_fabric_causal_queries_drop_malformed_backend_rows() -> None:
    malformed = GraphEntity(
        name="effect",
        entity_type="outcome",
        valid_from="2026-06-07T00:00:00Z",
        valid_to=None,
        properties={
            "relation_type": "causal_caused",
            "causal_relation_type": "caused",
            "confidence": 0.8,
            "inference_method": "explicit_outcome_citation_v1",
            "review_status": "proposed",
            "authority_status": "non_authoritative",
        },
        session_id="agent-1",
    )
    fabric = _fabric_with_graph(_DirectionalCausalStore([malformed]))

    assert await fabric.query_causal_successors("cause", session_id="agent-1") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_key",
    [
        "causal_source_name",
        "causal_source_type",
        "causal_target_name",
        "causal_target_type",
    ],
)
async def test_memory_fabric_causal_queries_drop_rows_missing_endpoint_metadata(missing_key: str) -> None:
    malformed = _backend_causal_entity(**{missing_key: ""})
    fabric = _fabric_with_graph(_DirectionalCausalStore([malformed]))

    assert await fabric.query_causal_successors("cause", session_id="agent-1") == []
    assert await fabric.query_causal_predecessors("effect", session_id="agent-1") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_event_seq", "source_event_hash"),
    [
        (0, "a" * 64),
        (-1, "a" * 64),
        ("42", "a" * 64),
        (42, "A" * 64),
        (42, "not-a-hash"),
    ],
)
async def test_memory_fabric_causal_queries_drop_rows_with_malformed_source_refs(
    source_event_seq: object,
    source_event_hash: object,
) -> None:
    malformed = _backend_causal_entity(source_event_seq=source_event_seq, source_event_hash=source_event_hash)
    fabric = _fabric_with_graph(_DirectionalCausalStore([malformed]))

    assert await fabric.query_causal_successors("cause", session_id="agent-1") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "citation",
    [
        "eventloom://unknown/events/42#aaaaaaaaaaaa",
        "https://example.invalid/events/42",
        "eventloom://agent-1/events/not-int#aaaaaaaaaaaa",
        "eventloom://agent-1/events/42#AAAAAAAAAAAA",
        "eventloom://agent-1/events/42#not-a-hash",
    ],
)
async def test_memory_fabric_causal_queries_drop_rows_with_invalid_explicit_citation(citation: str) -> None:
    malformed = _backend_causal_entity(citation=citation)
    malformed.properties.pop("source_event_seq", None)
    malformed.properties.pop("source_event_hash", None)
    fabric = _fabric_with_graph(_DirectionalCausalStore([malformed]))

    assert await fabric.query_causal_successors("cause", session_id="agent-1") == []


@pytest.mark.asyncio
async def test_memory_fabric_causal_queries_do_not_fallback_from_malformed_explicit_citation() -> None:
    malformed = _backend_causal_entity(citation="https://example.invalid/events/42")
    fabric = _fabric_with_graph(_DirectionalCausalStore([malformed]))

    assert await fabric.query_causal_successors("cause", session_id="agent-1") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_event_seq", "source_event_hash"),
    [
        ("42", "a" * 64),
        (42, "not-a-hash"),
        (42, None),
        (None, "a" * 64),
    ],
)
async def test_memory_fabric_causal_queries_drop_valid_citation_with_invalid_source_refs(
    source_event_seq: object,
    source_event_hash: object,
) -> None:
    malformed = _backend_causal_entity(
        citation="eventloom://agent-1/events/42#aaaaaaaaaaaa",
        source_event_seq=source_event_seq,
        source_event_hash=source_event_hash,
    )
    fabric = _fabric_with_graph(_DirectionalCausalStore([malformed]))

    assert await fabric.query_causal_successors("cause", session_id="agent-1") == []


@pytest.mark.asyncio
async def test_memory_fabric_causal_queries_accept_valid_explicit_citation_without_source_refs() -> None:
    entity = _backend_causal_entity(citation="eventloom://agent-1/events/42#aaaaaaaaaaaa")
    entity.properties.pop("source_event_seq", None)
    entity.properties.pop("source_event_hash", None)
    fabric = _fabric_with_graph(_DirectionalCausalStore([entity]))

    results = await fabric.query_causal_successors("cause", session_id="agent-1")

    assert [result.target["name"] for result in results] == ["effect"]
    assert results[0].citation == "eventloom://agent-1/events/42#aaaaaaaaaaaa"


@pytest.mark.asyncio
async def test_memory_fabric_query_causal_successors_filters_relation_and_returns_effect() -> None:
    store = _DirectionalCausalStore()
    fabric = _fabric_with_graph(store)

    results = await fabric.query_causal_successors(
        "cause",
        relation_type="caused",
        depth=3,
        temporal_point="2026-06-07T00:00:00Z",
        session_id="agent-1",
    )

    assert store.calls == [
        {
            "entity_name": "cause",
            "direction": "successors",
            "relation_type": "causal_caused",
            "depth": 3,
            "temporal_point": "2026-06-07T00:00:00Z",
            "session_id": "agent-1",
        }
    ]
    assert [result.target["name"] for result in results] == ["effect"]
    assert results[0].to_dict()["method"] == "explicit_outcome_citation_v1"
    assert results[0].to_dict()["citation"] == "eventloom://agent-1/events/42#aaaaaaaaaaaa"
    assert results[0].to_dict()["review_status"] == "proposed"
    assert results[0].to_dict()["authority_status"] == "non_authoritative"


@pytest.mark.asyncio
async def test_memory_fabric_query_causal_predecessors_uses_incoming_direction_and_returns_cause() -> None:
    store = _DirectionalCausalStore()
    fabric = _fabric_with_graph(store)

    results = await fabric.query_causal_predecessors("effect", relation_type="caused", session_id="agent-1")

    assert store.calls == [
        {
            "entity_name": "effect",
            "direction": "predecessors",
            "relation_type": "causal_caused",
            "depth": 2,
            "temporal_point": None,
            "session_id": "agent-1",
        }
    ]
    assert [result.source["name"] for result in results] == ["cause"]
    assert [result.target["name"] for result in results] == ["effect"]


@pytest.mark.asyncio
async def test_memory_fabric_causal_queries_reject_invalid_relation_type() -> None:
    fabric = _fabric_with_graph(_DirectionalCausalStore())

    with pytest.raises(ValueError, match="causal relation_type"):
        await fabric.query_causal_successors("cause", relation_type="not-causal")
    with pytest.raises(ValueError, match="causal relation_type"):
        await fabric.query_causal_predecessors("effect", relation_type="not-causal")


def test_build_causal_edge_event_requires_cited_source_event() -> None:
    edge = CausalEdge(
        source={"name": "command:pytest", "entity_type": "command"},
        target={"name": "test failure", "entity_type": "outcome"},
        relation_type="caused",
        confidence=0.91,
        method="explicit_outcome_citation_v1",
        evidence={
            "source_event_seq": 42,
            "source_event_hash": "a" * 64,
            "reason": "The command output contained the failure.",
        },
    )
    assert edge.to_payload()["authority_status"] == "non_authoritative"

    event = build_causal_edge_event(
        actor="zaxy-causal",
        session_id="agent-1",
        source={"name": "command:pytest", "entity_type": "command"},
        target={"name": "test failure", "entity_type": "outcome"},
        relation_type="caused",
        confidence=0.91,
        method="explicit_outcome_citation_v1",
        evidence={
            "source_event_seq": 42,
            "source_event_hash": "a" * 64,
            "reason": "The command output contained the failure.",
        },
    )
    assert event == {
        "event_type": "causal.edge.generated",
        "actor": "zaxy-causal",
        "payload": {
            "source": {"name": "command:pytest", "entity_type": "command"},
            "target": {"name": "test failure", "entity_type": "outcome"},
            "relation_type": "caused",
            "graph_relation_type": "causal_caused",
            "confidence": 0.91,
            "causal_method": "explicit_outcome_citation_v1",
            "review_status": "proposed",
            "authority_status": "non_authoritative",
            "evidence": {
                "source_event_seq": 42,
                "source_event_hash": "a" * 64,
                "reason": "The command output contained the failure.",
            },
        },
        "thread": "agent-1",
    }


@pytest.mark.parametrize("relation_type", ["", "CAUSES", "likely_informed", "causal_caused"])
def test_build_causal_edge_event_rejects_non_taxonomy_relation(relation_type: str) -> None:
    with pytest.raises(ValueError, match="causal relation_type"):
        build_causal_edge_event(
            actor="zaxy-causal",
            session_id="agent-1",
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type=relation_type,
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": 1, "source_event_hash": "b" * 64},
        )


def test_build_causal_edge_event_rejects_uncited_evidence() -> None:
    with pytest.raises(ValueError, match="source_event_hash"):
        build_causal_edge_event(
            actor="zaxy-causal",
            session_id="agent-1",
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": 1},
        )


@pytest.mark.parametrize("source_event_seq", [0, -1, True])
def test_build_causal_edge_event_rejects_invalid_source_event_seq(
    source_event_seq: int | bool,
) -> None:
    with pytest.raises(ValueError, match="source_event_seq"):
        build_causal_edge_event(
            actor="zaxy-causal",
            session_id="agent-1",
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": source_event_seq, "source_event_hash": "b" * 64},
        )


@pytest.mark.parametrize(
    "source_event_hash",
    [
        "",
        "b" * 63,
        "b" * 65,
        "B" * 64,
        "g" * 64,
        True,
    ],
)
def test_build_causal_edge_event_rejects_invalid_source_event_hash(
    source_event_hash: str | bool,
) -> None:
    with pytest.raises(ValueError, match="source_event_hash"):
        build_causal_edge_event(
            actor="zaxy-causal",
            session_id="agent-1",
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": 1, "source_event_hash": source_event_hash},
        )


def test_build_causal_edge_event_rejects_whitespace_method() -> None:
    with pytest.raises(ValueError, match="causal method"):
        build_causal_edge_event(
            actor="zaxy-causal",
            session_id="agent-1",
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            confidence=0.8,
            method="   ",
            evidence={"source_event_seq": 1, "source_event_hash": "b" * 64},
        )


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        ({"name": "   ", "entity_type": "event"}, {"name": "b", "entity_type": "outcome"}, "source.name"),
        ({"name": "a", "entity_type": "   "}, {"name": "b", "entity_type": "outcome"}, "source.entity_type"),
        ({"name": "a", "entity_type": "event"}, {"name": "   ", "entity_type": "outcome"}, "target.name"),
        ({"name": "a", "entity_type": "event"}, {"name": "b", "entity_type": "   "}, "target.entity_type"),
    ],
)
def test_build_causal_edge_event_rejects_whitespace_entity_refs(
    source: dict[str, str],
    target: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_causal_edge_event(
            actor="zaxy-causal",
            session_id="agent-1",
            source=source,
            target=target,
            relation_type="caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": 1, "source_event_hash": "b" * 64},
        )


@pytest.mark.parametrize("review_status", ["", "   ", "approved", "authoritative"])
def test_causal_edge_rejects_invalid_review_status(review_status: str) -> None:
    with pytest.raises(ValueError, match="review_status"):
        CausalEdge(
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": 1, "source_event_hash": "b" * 64},
            review_status=review_status,
        )


@pytest.mark.parametrize(
    "review_status",
    ["proposed", "accepted", "rejected", "deferred", "conflicted"],
)
def test_causal_edge_accepts_review_lifecycle_statuses(review_status: str) -> None:
    edge = CausalEdge(
        source={"name": "a", "entity_type": "event"},
        target={"name": "b", "entity_type": "outcome"},
        relation_type="caused",
        confidence=0.8,
        method="explicit_outcome_citation_v1",
        evidence={"source_event_seq": 1, "source_event_hash": "b" * 64},
        review_status=review_status,
    )

    assert edge.to_payload()["review_status"] == review_status


@pytest.mark.parametrize("authority_status", ["", "   ", "authoritative", "accepted"])
def test_causal_edge_rejects_invalid_authority_status(authority_status: str) -> None:
    with pytest.raises(ValueError, match="authority_status"):
        CausalEdge(
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": 1, "source_event_hash": "b" * 64},
            authority_status=authority_status,
        )


def test_causal_edge_snapshots_mutable_mapping_inputs() -> None:
    source = {"name": "a", "entity_type": "event"}
    target = {"name": "b", "entity_type": "outcome"}
    evidence: dict[str, object] = {
        "source_event_seq": 1,
        "source_event_hash": "b" * 64,
        "reason": "original",
    }
    edge = CausalEdge(
        source=source,
        target=target,
        relation_type="caused",
        confidence=0.8,
        method="explicit_outcome_citation_v1",
        evidence=evidence,
    )

    source["name"] = "mutated-source"
    target["entity_type"] = "mutated-target"
    evidence["source_event_hash"] = "g" * 64
    evidence["reason"] = "mutated"

    assert edge.to_payload()["source"] == {"name": "a", "entity_type": "event"}
    assert edge.to_payload()["target"] == {"name": "b", "entity_type": "outcome"}
    assert edge.to_payload()["evidence"] == {
        "source_event_seq": 1,
        "source_event_hash": "b" * 64,
        "reason": "original",
    }


def test_causal_query_result_to_dict_preserves_authority_boundary() -> None:
    result = CausalQueryResult(
        source={"name": "command:pytest", "entity_type": "command"},
        target={"name": "test failure", "entity_type": "outcome"},
        relation_type="caused",
        graph_relation_type="causal_caused",
        confidence=0.91,
        method="explicit_outcome_citation_v1",
        citation="eventloom://agent-1/events/42#aaaaaaaaaaaa",
        review_status="proposed",
        authority_status="non_authoritative",
        evidence={"source_event_seq": 42},
        path_length=1,
    )
    assert result.to_dict()["authority_status"] == "non_authoritative"
    assert result.to_dict()["review_status"] == "proposed"
    assert result.to_dict()["citation"] == "eventloom://agent-1/events/42#aaaaaaaaaaaa"
    assert result.to_dict()["method"] == "explicit_outcome_citation_v1"
    assert "causal_method" not in result.to_dict()


def test_embedded_causal_metadata_returns_none_for_malformed_backend_fields() -> None:
    source = GraphEntity("cause", "event", "2026-06-07T00:00:00Z", None, {}, session_id="agent-1")
    target = GraphEntity("effect", "outcome", "2026-06-07T00:00:00Z", None, {}, session_id="agent-1")
    row = [
        "source-key",
        "cause",
        "event",
        "2026-06-07T00:00:00Z",
        None,
        None,
        "{}",
        "agent-1",
        1,
        "a" * 64,
        "causal_caused",
        "target-key",
        "effect",
        "outcome",
        "2026-06-07T00:00:00Z",
        None,
        None,
        "{}",
        "agent-1",
        2,
        "b" * 64,
        "not-a-float",
        "explicit_outcome_citation_v1",
        "not-an-int",
        "not-a-hash",
        "{broken-json",
    ]

    assert _causal_edge_metadata_from_row(row, source_entity=source, target_entity=target) is None


def test_backend_json_helpers_omit_malformed_json_without_crashing() -> None:
    assert _latticedb_json_dict("{broken-json") == {}
    assert _pggraph_properties_from_row({"properties": "{broken-json"}) == {}


def test_pggraph_causal_metadata_defaults_bad_numeric_fields_without_crashing() -> None:
    source = GraphEntity("cause", "event", "2026-06-07T00:00:00Z", None, {}, session_id="agent-1")
    target = GraphEntity("effect", "outcome", "2026-06-07T00:00:00Z", None, {}, session_id="agent-1")

    metadata = _pggraph_causal_edge_metadata(
        {
            "relation_type": "causal_caused",
            "confidence": "not-a-float",
            "inference_method": "explicit_outcome_citation_v1",
            "edge_source_event_seq": "not-an-int",
            "edge_source_event_hash": "a" * 64,
            "edge_properties": "{broken-json",
        },
        source=source,
        target=target,
        session_id="agent-1",
    )

    assert metadata["confidence"] == 1.0
    assert metadata["source_event_seq"] is None
    assert metadata["evidence"] == {}


@pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")
@pytest.mark.asyncio
async def test_embedded_store_search_causal_neighbors_preserves_direction(tmp_path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    try:
        await store.init_schema()
        await store.upsert_extraction(
            ExtractionResult(
                entities=[
                    ExtractedEntity(name="cause", entity_type="event", observed_at="2026-06-07T00:00:00Z"),
                    ExtractedEntity(name="effect", entity_type="outcome", observed_at="2026-06-07T00:00:00Z"),
                ],
                edges=[
                    ExtractedEdge(
                        source="cause",
                        target="effect",
                        relation_type="causal_caused",
                        valid_from="2026-06-07T00:00:00Z",
                        inferred=True,
                        confidence=0.91,
                        inference_method="explicit_outcome_citation_v1",
                        evidence={
                            "source_event_seq": 1,
                            "source_event_hash": "a" * 64,
                            "causal_relation_type": "caused",
                            "review_status": "proposed",
                            "authority_status": "non_authoritative",
                        },
                    )
                ],
                source_event_seq=1,
                source_event_hash="a" * 64,
                source_event_type="causal.edge.generated",
                source_thread="agent-1",
            ),
            session_id="agent-1",
        )

        successors = await store.search_causal_neighbors("cause", direction="successors", session_id="agent-1")
        predecessors = await store.search_causal_neighbors("effect", direction="predecessors", session_id="agent-1")
        wrong_direction = await store.search_causal_neighbors("effect", direction="successors", session_id="agent-1")

        assert [entity.name for entity in successors] == ["effect"]
        assert [entity.name for entity in predecessors] == ["cause"]
        assert wrong_direction == []
    finally:
        await store.close()


@pytest.mark.parametrize("citation", ["", "   "])
def test_causal_query_result_rejects_invalid_citation(citation: str) -> None:
    with pytest.raises(ValueError, match="citation"):
        CausalQueryResult(
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            graph_relation_type="causal_caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            citation=citation,
            review_status="proposed",
            authority_status="non_authoritative",
        )


@pytest.mark.parametrize("path_length", [0, -1])
def test_causal_query_result_rejects_invalid_path_length(path_length: int) -> None:
    with pytest.raises(ValueError, match="path_length"):
        CausalQueryResult(
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            graph_relation_type="causal_caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            citation="eventloom://agent-1/events/1#bbbbbbbbbbbb",
            review_status="proposed",
            authority_status="non_authoritative",
            path_length=path_length,
        )


def test_causal_query_result_rejects_invalid_evidence_mapping() -> None:
    with pytest.raises(ValueError, match="evidence"):
        CausalQueryResult(
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            graph_relation_type="causal_caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            citation="eventloom://agent-1/events/1#bbbbbbbbbbbb",
            review_status="proposed",
            authority_status="non_authoritative",
            evidence=[("source_event_seq", 1)],  # type: ignore[arg-type]
        )


def test_causal_query_result_rejects_invalid_statuses() -> None:
    with pytest.raises(ValueError, match="review_status"):
        CausalQueryResult(
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            graph_relation_type="causal_caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            citation="eventloom://agent-1/events/1#bbbbbbbbbbbb",
            review_status="approved",
            authority_status="non_authoritative",
        )
    with pytest.raises(ValueError, match="authority_status"):
        CausalQueryResult(
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            graph_relation_type="causal_caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            citation="eventloom://agent-1/events/1#bbbbbbbbbbbb",
            review_status="proposed",
            authority_status="authoritative",
        )


def test_causal_query_result_snapshots_mutable_mapping_inputs() -> None:
    source = {"name": "a", "entity_type": "event"}
    target = {"name": "b", "entity_type": "outcome"}
    evidence: dict[str, object] = {"source_event_seq": 1, "reason": "original"}
    result = CausalQueryResult(
        source=source,
        target=target,
        relation_type="caused",
        graph_relation_type="causal_caused",
        confidence=0.8,
        method="explicit_outcome_citation_v1",
        citation="eventloom://agent-1/events/1#bbbbbbbbbbbb",
        review_status="proposed",
        authority_status="non_authoritative",
        evidence=evidence,
    )

    source["name"] = "mutated-source"
    target["entity_type"] = "mutated-target"
    evidence["reason"] = "mutated"

    assert result.to_dict()["source"] == {"name": "a", "entity_type": "event"}
    assert result.to_dict()["target"] == {"name": "b", "entity_type": "outcome"}
    assert result.to_dict()["evidence"] == {"source_event_seq": 1, "reason": "original"}
