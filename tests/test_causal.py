from __future__ import annotations

import pytest

from zaxy.causal import (
    CAUSAL_RELATION_TYPES,
    CausalEdge,
    CausalQueryResult,
    build_causal_edge_event,
    causal_relation_to_graph_relation,
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
