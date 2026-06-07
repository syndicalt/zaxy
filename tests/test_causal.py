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
