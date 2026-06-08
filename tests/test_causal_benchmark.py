"""Tests for the alpha.1 causal/consolidation benchmark helpers."""

from __future__ import annotations

import pytest

from zaxy.causal_benchmark import (
    CausalBenchmarkCase,
    ConsolidationBenchmarkCase,
    evaluate_causal_results,
    evaluate_consolidation_candidate,
    summarize_causal_benchmark,
)
from zaxy.consolidation import build_consolidation_candidate_event


class GraphEntityLike:
    def __init__(self, name: str, properties: dict[str, object]) -> None:
        self.name = name
        self.properties = properties


def test_causal_case_rejects_unknown_query_type() -> None:
    with pytest.raises(ValueError, match="query_type"):
        CausalBenchmarkCase(
            case_id="bad-direction",
            query="What caused the deployment rollback?",
            query_type="neighbor",
            source={"name": "config drift", "entity_type": "Task"},
            target={"name": "deployment rollback", "entity_type": "Task"},
            relation_type="caused",
            citation="eventloom://session-alpha/events/42#abcdefabcdef",
        )


@pytest.mark.parametrize("relation_type", ["CAUSES", "causal_caused", "depends_on"])
def test_causal_case_rejects_non_production_relation_labels(relation_type: str) -> None:
    with pytest.raises(ValueError, match="relation_type"):
        CausalBenchmarkCase(
            case_id="bad-relation",
            query="What caused the deployment rollback?",
            query_type="predecessor",
            source={"name": "config drift", "entity_type": "Task"},
            target={"name": "deployment rollback", "entity_type": "Task"},
            relation_type=relation_type,
            citation="eventloom://session-alpha/events/42#abcdefabcdef",
        )


def test_successor_scoring_uses_target_endpoint_and_prefers_non_authoritative_cited_match() -> None:
    case = CausalBenchmarkCase(
        case_id="successor-target",
        query="What did the config drift cause?",
        query_type="successor",
        source={"name": "config drift", "entity_type": "Task"},
        target={"name": "deployment rollback", "entity_type": "Task"},
        relation_type="caused",
        citation="eventloom://session-alpha/events/42#abcdefabcdef",
    )
    results = [
        {
            "source": {"name": "config drift", "entity_type": "Task"},
            "target": {"name": "deployment rollback", "entity_type": "Task"},
            "relation_type": "caused",
            "authority_status": "promoted",
            "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
        },
        {
            "source": {"name": "config drift", "entity_type": "Task"},
            "target": {"name": "deployment rollback", "entity_type": "Task"},
            "relation_type": "caused",
            "authority_status": "non_authoritative",
            "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
        },
    ]

    row = evaluate_causal_results(case, results)

    assert row["case_id"] == "successor-target"
    assert row["hit"] is True
    assert row["relation_match"] is True
    assert row["citation"] is True
    assert row["authority_boundary"] is True
    assert row["score"] == 1.0
    assert row["matched_result"]["authority_status"] == "non_authoritative"


@pytest.mark.parametrize(
    "result",
    [
        {
            "target": {"name": "deployment rollback", "entity_type": "Incident"},
            "relation_type": "caused",
            "authority_status": "non_authoritative",
            "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
        },
        {
            "target_name": "deployment rollback",
            "target_entity_type": "Incident",
            "relation_type": "caused",
            "authority_status": "non_authoritative",
            "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
        },
        GraphEntityLike(
            name="causal-edge",
            properties={
                "causal_target_name": "deployment rollback",
                "causal_target_type": "Incident",
                "relation_type": "caused",
                "authority_status": "non_authoritative",
                "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
            },
        ),
    ],
)
def test_causal_scoring_rejects_endpoint_with_matching_name_but_wrong_type(
    result: object,
) -> None:
    case = CausalBenchmarkCase(
        case_id="successor-target-type",
        query="What did the config drift cause?",
        query_type="successor",
        source={"name": "config drift", "entity_type": "Task"},
        target={"name": "deployment rollback", "entity_type": "Task"},
        relation_type="caused",
        citation="eventloom://session-alpha/events/42#abcdefabcdef",
    )

    row = evaluate_causal_results(case, [result])

    assert row["hit"] is False
    assert row["relation_match"] is False
    assert row["citation"] is False
    assert row["authority_boundary"] is False
    assert row["score"] == 0.0
    assert row["matched_result"] is None


def test_causal_scoring_accepts_production_graph_projection_endpoint_keys() -> None:
    case = CausalBenchmarkCase(
        case_id="successor-production-graph",
        query="What did the config drift cause?",
        query_type="successor",
        source={"name": "config drift", "entity_type": "Task"},
        target={"name": "deployment rollback", "entity_type": "Task"},
        relation_type="caused",
        citation="eventloom://session-alpha/events/42#abcdefabcdef",
    )
    result = GraphEntityLike(
        name="causal-edge",
        properties={
            "causal_source_name": "config drift",
            "causal_source_type": "Task",
            "causal_target_name": "deployment rollback",
            "causal_target_type": "Task",
            "relation_type": "caused",
            "authority_status": "non_authoritative",
            "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
        },
    )

    row = evaluate_causal_results(case, [result])

    assert row["hit"] is True
    assert row["relation_match"] is True
    assert row["citation"] is True
    assert row["authority_boundary"] is True
    assert row["score"] == 1.0


def test_predecessor_scoring_uses_source_endpoint_and_penalizes_distractor_defects() -> None:
    case = CausalBenchmarkCase(
        case_id="predecessor-source",
        query="What caused the deployment rollback?",
        query_type="predecessor",
        source={"name": "config drift", "entity_type": "Task"},
        target={"name": "deployment rollback", "entity_type": "Task"},
        relation_type="caused",
        citation="eventloom://session-alpha/events/42#abcdefabcdef",
    )
    results = [
        {
            "source": {"name": "config drift", "entity_type": "Task"},
            "target": {"name": "deployment rollback", "entity_type": "Task"},
            "relation_type": "enabled",
            "authority_status": "promoted",
            "citation": "note://not-eventloom",
        },
        {
            "source": {"name": "unrelated alert", "entity_type": "Task"},
            "target": {"name": "deployment rollback", "entity_type": "Task"},
            "relation_type": "caused",
            "authority_status": "non_authoritative",
            "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
        },
    ]

    row = evaluate_causal_results(case, results)

    assert row["hit"] is True
    assert row["relation_match"] is False
    assert row["citation"] is False
    assert row["authority_boundary"] is False
    assert row["score"] == 0.25


def test_causal_scoring_prefers_current_match_over_stale_matching_endpoint() -> None:
    case = CausalBenchmarkCase(
        case_id="stale-distractor",
        query="What did the config drift cause?",
        query_type="successor",
        source={"name": "config drift", "entity_type": "Task"},
        target={"name": "deployment rollback", "entity_type": "Task"},
        relation_type="caused",
        citation="eventloom://session-alpha/events/42#abcdefabcdef",
    )
    results = [
        {
            "target": {"name": "deployment rollback", "entity_type": "Task"},
            "relation_type": "caused",
            "authority_status": "non_authoritative",
            "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
            "superseded_by": "eventloom://session-alpha/events/43#bbbbbbbbbbbb",
        },
        {
            "target": {"name": "deployment rollback", "entity_type": "Task"},
            "relation_type": "caused",
            "authority_status": "non_authoritative",
            "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
        },
    ]

    row = evaluate_causal_results(case, results)

    assert row["score"] == 1.0
    assert "superseded_by" not in row["matched_result"]


def test_causal_summary_reports_mean_rates_and_empty_defaults() -> None:
    rows = [
        {"score": 1.0, "hit": True, "citation": True, "authority_boundary": True},
        {"score": 0.25, "hit": True, "citation": False, "authority_boundary": False},
    ]

    summary = summarize_causal_benchmark(rows)

    assert summary == {
        "case_count": 2,
        "mean": 0.625,
        "hit_rate": 1.0,
        "citation_coverage": 0.5,
        "authority_boundary": 0.5,
    }
    assert summarize_causal_benchmark([]) == {
        "case_count": 0,
        "mean": 0.0,
        "hit_rate": 0.0,
        "citation_coverage": 0.0,
        "authority_boundary": 0.0,
    }


def test_causal_case_requires_eventloom_style_citation() -> None:
    with pytest.raises(ValueError, match="citation"):
        CausalBenchmarkCase(
            case_id="bad-citation",
            query="What caused the deployment rollback?",
            query_type="predecessor",
            source={"name": "config drift", "entity_type": "Task"},
            target={"name": "deployment rollback", "entity_type": "Task"},
            relation_type="caused",
            citation="plain text",
        )


@pytest.mark.parametrize(
    "citation",
    [
        "eventloom://unknown/events/42#abcdefabcdef",
        "eventloom://session-alpha/events/0#abcdefabcdef",
        "eventloom://session-alpha/events/042#abcdefabcdef",
        "eventloom://session-alpha/events/42#abcdeabcdea",
        "eventloom://session-alpha/events/42#abcdefabcdef0",
    ],
)
def test_causal_case_rejects_invalid_eventloom_citation_contract(citation: str) -> None:
    with pytest.raises(ValueError, match="citation"):
        CausalBenchmarkCase(
            case_id="bad-citation-contract",
            query="What caused the deployment rollback?",
            query_type="predecessor",
            source={"name": "config drift", "entity_type": "Task"},
            target={"name": "deployment rollback", "entity_type": "Task"},
            relation_type="caused",
            citation=citation,
        )


@pytest.mark.parametrize(
    "citation",
    [
        "eventloom://unknown/events/42#abcdefabcdef",
        "eventloom://session-alpha/events/0#abcdefabcdef",
        "eventloom://session-alpha/events/042#abcdefabcdef",
        "eventloom://session-alpha/events/42#abcdeabcdea",
        "eventloom://session-alpha/events/42#abcdefabcdef0",
    ],
)
def test_consolidation_case_rejects_invalid_eventloom_citation_contract(
    citation: str,
) -> None:
    with pytest.raises(ValueError, match="citation"):
        ConsolidationBenchmarkCase(
            case_id="bad-consolidation-citation-contract",
            candidate_id="consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb",
            candidate_type="claim",
            source_events=({"seq": 42, "hash": "d" * 64},),
            citation=citation,
        )


@pytest.mark.parametrize(
    ("candidate_id", "candidate_type"),
    [
        ("projection:deploy-root-cause", "claim"),
        ("consolidation:memory:bbbbbbbbbbbbbbbbbbbbbbbb", "memory"),
        ("consolidation:claim:BBBBBBBBBBBBBBBBBBBBBBBB", "claim"),
        ("consolidation:episode:bbbbbbbbbbbbbbbbbbbbbbbb", "claim"),
    ],
)
def test_consolidation_case_rejects_invalid_or_mismatched_candidate_contract(
    candidate_id: str, candidate_type: str
) -> None:
    with pytest.raises(ValueError, match="candidate"):
        ConsolidationBenchmarkCase(
            case_id="bad-candidate",
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            source_events=({"seq": 42, "hash": "d" * 64},),
            citation=f"eventloom://session-alpha/events/42#{'d' * 12}",
        )


def test_consolidation_candidate_scores_source_fidelity_and_non_authoritative_boundary() -> None:
    case = ConsolidationBenchmarkCase(
        case_id="projection-fidelity",
        candidate_id="consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb",
        candidate_type="claim",
        source_events=(
            {"seq": 41, "hash": "a" * 64},
            {"seq": 42, "hash": "d" * 64},
        ),
        citation=f"eventloom://session-alpha/events/42#{'d' * 12}",
        authority_status="non_authoritative",
    )
    candidate = {
        "candidate_id": "consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb",
        "candidate_type": "claim",
        "source_events": [
            {"seq": 41, "hash": "a" * 64},
            {"seq": 42, "hash": "d" * 64},
        ],
        "source_event_refs": [f"41:{'a' * 64}", f"42:{'d' * 64}"],
        "authority_status": "non_authoritative",
    }

    row = evaluate_consolidation_candidate(case, candidate)

    assert row == {
        "case_id": "projection-fidelity",
        "candidate_match": True,
        "source_event_fidelity": True,
        "citation_coverage": True,
        "authority_boundary": True,
        "score": 1.0,
    }


def test_consolidation_candidate_scores_graph_entity_name_as_candidate_id() -> None:
    candidate_id = "consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb"
    case = ConsolidationBenchmarkCase(
        case_id="graph-entity-projection",
        candidate_id=candidate_id,
        candidate_type="claim",
        source_events=(
            {"seq": 41, "hash": "a" * 64},
            {"seq": 42, "hash": "d" * 64},
        ),
        citation="eventloom://session-alpha/events/42#dddddddddddd",
        authority_status="non_authoritative",
    )
    candidate = GraphEntityLike(
        name=candidate_id,
        properties={
            "candidate_type": "claim",
            "source_events": [
                {"seq": 41, "hash": "a" * 64},
                {"seq": 42, "hash": "d" * 64},
            ],
            "source_event_refs": [f"41:{'a' * 64}", f"42:{'d' * 64}"],
            "authority_status": "non_authoritative",
        },
    )

    row = evaluate_consolidation_candidate(case, candidate)

    assert row == {
        "case_id": "graph-entity-projection",
        "candidate_match": True,
        "source_event_fidelity": True,
        "citation_coverage": True,
        "authority_boundary": True,
        "score": 1.0,
    }


def test_consolidation_candidate_penalizes_promoted_or_missing_source_refs() -> None:
    case = ConsolidationBenchmarkCase(
        case_id="projection-boundary",
        candidate_id="consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb",
        candidate_type="claim",
        source_events=(
            {"seq": 41, "hash": "a" * 64},
            {"seq": 42, "hash": "d" * 64},
        ),
        citation=f"eventloom://session-alpha/events/42#{'d' * 12}",
        authority_status="non_authoritative",
    )
    candidate = {
        "candidate_id": "consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb",
        "candidate_type": "claim",
        "source_events": [{"seq": 41, "hash": "a" * 64}],
        "source_event_refs": [f"41:{'a' * 64}"],
        "authority_status": "promoted",
    }

    row = evaluate_consolidation_candidate(case, candidate)

    assert row["candidate_match"] is True
    assert row["source_event_fidelity"] is False
    assert row["citation_coverage"] is False
    assert row["authority_boundary"] is False
    assert row["score"] == 0.25


def test_consolidation_candidate_accepts_case_citation_in_citation_list() -> None:
    case = ConsolidationBenchmarkCase(
        case_id="citation-list-coverage",
        candidate_id="consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb",
        candidate_type="claim",
        source_events=({"seq": 42, "hash": "d" * 64},),
        citation=f"eventloom://session-alpha/events/42#{'d' * 12}",
        authority_status="non_authoritative",
    )
    candidate = {
        "candidate_id": "consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb",
        "candidate_type": "claim",
        "source_events": [{"seq": 42, "hash": "d" * 64}],
        "citations": [case.citation],
        "authority_status": "non_authoritative",
    }

    row = evaluate_consolidation_candidate(case, candidate)

    assert row["candidate_match"] is True
    assert row["source_event_fidelity"] is True
    assert row["citation_coverage"] is True
    assert row["authority_boundary"] is True
    assert row["score"] == 1.0


def test_consolidation_case_rejects_non_production_source_event_shape() -> None:
    with pytest.raises(ValueError, match=r"source_events\[0\]\.seq"):
        ConsolidationBenchmarkCase(
            case_id="bad-source-event",
            candidate_id="consolidation:claim:bbbbbbbbbbbbbbbbbbbbbbbb",
            candidate_type="claim",
            source_events=(
                {
                    "ref": f"eventloom://session-alpha/events/42#{'d' * 12}",
                    "hash": "d" * 64,
                },
            ),
            citation=f"eventloom://session-alpha/events/42#{'d' * 12}",
        )


def test_production_consolidation_candidate_payload_scores_contract_fields() -> None:
    source_hashes = ("a" * 64, "b" * 64)
    production_event = build_consolidation_candidate_event(
        actor="zaxy-consolidation",
        session_id="session-alpha",
        candidate_type="claim",
        title="Deployment rollback root cause",
        summary="Config drift caused the deployment rollback.",
        source_events=[
            {"seq": 41, "hash": source_hashes[0]},
            {"seq": 42, "hash": source_hashes[1]},
        ],
        confidence=0.8,
        method="event_segment_cluster_v1",
    )
    payload = production_event["payload"]
    case = ConsolidationBenchmarkCase(
        case_id="production-payload",
        candidate_id=payload["candidate_id"],
        candidate_type=payload["candidate_type"],
        source_events=payload["source_events"],
        citation=f"eventloom://session-alpha/events/42#{source_hashes[1][:12]}",
    )

    row = evaluate_consolidation_candidate(case, payload)

    assert row["candidate_match"] is True
    assert row["source_event_fidelity"] is True
    assert row["citation_coverage"] is True
    assert row["authority_boundary"] is True
    assert row["score"] == 1.0


def test_causal_scoring_returns_empty_row_when_no_results() -> None:
    case = CausalBenchmarkCase(
        case_id="empty",
        query="What caused the deployment rollback?",
        query_type="predecessor",
        source={"name": "config drift", "entity_type": "Task"},
        target={"name": "deployment rollback", "entity_type": "Task"},
        relation_type="caused",
        citation="eventloom://session-alpha/events/42#abcdefabcdef",
    )

    row = evaluate_causal_results(case, [])

    assert row == {
        "case_id": "empty",
        "query_type": "predecessor",
        "hit": False,
        "relation_match": False,
        "citation": False,
        "authority_boundary": False,
        "score": 0.0,
        "matched_result": None,
    }


def test_causal_scoring_reads_nested_properties_from_mapping_endpoint() -> None:
    case = CausalBenchmarkCase(
        case_id="nested-properties",
        query="What did the config drift cause?",
        query_type="successor",
        source={"name": "config drift", "entity_type": "Task"},
        target={"name": "deployment rollback", "entity_type": "Task"},
        relation_type="caused",
        citation="eventloom://session-alpha/events/42#abcdefabcdef",
    )
    result = {
        "target": {
            "properties": {
                "name": "deployment rollback",
                "entity_type": "Task",
            }
        },
        "properties": {
            "relation_type": "caused",
            "authority_status": "non_authoritative",
            "citations": ["eventloom://session-alpha/events/42#abcdefabcdef"],
        },
    }

    row = evaluate_causal_results(case, [result])

    assert row["hit"] is True
    assert row["citation"] is True
    assert row["authority_boundary"] is True


@pytest.mark.parametrize(
    "stale_field",
    [
        {"stale": True},
        {"valid_to": "2026-06-07T00:00:00Z"},
        {"superseded_by": "eventloom://session-alpha/events/43#bbbbbbbbbbbb"},
    ],
)
def test_causal_scoring_penalizes_stale_matching_rows(stale_field: dict[str, object]) -> None:
    case = CausalBenchmarkCase(
        case_id="stale-row",
        query="What did the config drift cause?",
        query_type="successor",
        source={"name": "config drift", "entity_type": "Task"},
        target={"name": "deployment rollback", "entity_type": "Task"},
        relation_type="caused",
        citation="eventloom://session-alpha/events/42#abcdefabcdef",
    )
    result = {
        "target": {"name": "deployment rollback", "entity_type": "Task"},
        "relation_type": "caused",
        "authority_status": "non_authoritative",
        "citation": "eventloom://session-alpha/events/42#abcdefabcdef",
        **stale_field,
    }

    row = evaluate_causal_results(case, [result])

    assert row["hit"] is True
    assert row["authority_boundary"] is False
    assert row["score"] == 0.75


def test_consolidation_candidate_scores_entity_name_fallback_and_source_ref_contract() -> None:
    candidate_id = "consolidation:claim:cccccccccccccccccccccccc"
    case = ConsolidationBenchmarkCase(
        case_id="entity-name-fallback",
        candidate_id=candidate_id,
        candidate_type="claim",
        source_events=({"seq": 42, "hash": "d" * 64},),
        citation="eventloom://session-alpha/events/42#dddddddddddd",
    )
    candidate = {
        "entity_name": candidate_id,
        "properties": {
            "candidate_type": "claim",
            "source_events": [{"seq": 42, "hash": "d" * 64}],
            "source_event_refs": [f"42:{'d' * 64}", 123, "bad-ref"],
            "authority_status": "non_authoritative",
        },
    }

    row = evaluate_consolidation_candidate(case, candidate)

    assert row["candidate_match"] is True
    assert row["source_event_fidelity"] is True
    assert row["citation_coverage"] is True
    assert row["authority_boundary"] is True


def test_consolidation_candidate_handles_non_sequence_sources_as_missing_evidence() -> None:
    case = ConsolidationBenchmarkCase(
        case_id="missing-source-shape",
        candidate_id="consolidation:claim:dddddddddddddddddddddddd",
        candidate_type="claim",
        source_events=({"seq": 42, "hash": "d" * 64},),
        citation="eventloom://session-alpha/events/42#dddddddddddd",
    )
    candidate = {
        "candidate_id": case.candidate_id,
        "candidate_type": "claim",
        "source_events": "not-a-source-list",
        "source_event_refs": "42:" + "d" * 64,
        "authority_status": "non_authoritative",
    }

    row = evaluate_consolidation_candidate(case, candidate)

    assert row["candidate_match"] is True
    assert row["source_event_fidelity"] is False
    assert row["citation_coverage"] is False
    assert row["authority_boundary"] is True
    assert row["score"] == 0.5


def test_consolidation_case_rejects_authoritative_case_boundary() -> None:
    with pytest.raises(ValueError, match="authority_status"):
        ConsolidationBenchmarkCase(
            case_id="authoritative-case",
            candidate_id="consolidation:claim:eeeeeeeeeeeeeeeeeeeeeeee",
            candidate_type="claim",
            source_events=({"seq": 42, "hash": "d" * 64},),
            citation="eventloom://session-alpha/events/42#dddddddddddd",
            authority_status="authoritative",
        )
