from __future__ import annotations

import pytest

from zaxy.consolidation_benchmark import (
    ConsolidationGuardrailCase,
    evaluate_consolidation_guardrail,
    summarize_consolidation_guardrail,
)


def test_guardrail_scores_cited_non_authoritative_candidate() -> None:
    case = ConsolidationGuardrailCase(
        case_id="claim-1",
        candidate_type="claim",
        required_source_events=[
            {"seq": 1, "hash": "a" * 64},
            {"seq": 2, "hash": "b" * 64},
        ],
    )
    row = evaluate_consolidation_guardrail(
        case,
        {
            "candidate_type": "claim",
            "source_events": [
                {"seq": 1, "hash": "a" * 64},
                {"seq": 2, "hash": "b" * 64},
            ],
            "review_status": "pending",
            "authority_status": "non_authoritative",
            "confidence": 0.7,
        },
    )

    assert row == {
        "case_id": "claim-1",
        "type_match": 1.0,
        "source_event_fidelity": 1.0,
        "authority_boundary": 1.0,
        "review_gate": 1.0,
        "score": 1.0,
    }


def test_guardrail_penalizes_authority_promotion() -> None:
    case = ConsolidationGuardrailCase(
        case_id="claim-2",
        candidate_type="claim",
        required_source_events=[{"seq": 1, "hash": "a" * 64}],
    )
    row = evaluate_consolidation_guardrail(
        case,
        {
            "candidate_type": "claim",
            "source_events": [{"seq": 1, "hash": "a" * 64}],
            "review_status": "accepted",
            "authority_status": "authoritative",
            "confidence": 0.7,
        },
    )

    assert row["authority_boundary"] == 0.0
    assert row["review_gate"] == 1.0
    assert row["score"] == 0.75


def test_guardrail_penalizes_missing_source_event_fidelity() -> None:
    case = ConsolidationGuardrailCase(
        case_id="procedure-1",
        candidate_type="procedure",
        required_source_events=[
            {"seq": 1, "hash": "a" * 64},
            {"seq": 2, "hash": "b" * 64},
        ],
    )

    row = evaluate_consolidation_guardrail(
        case,
        {
            "candidate_type": "procedure",
            "source_events": [{"seq": 1, "hash": "a" * 64}],
            "review_status": "pending",
            "authority_status": "non_authoritative",
        },
    )

    assert row["source_event_fidelity"] == 0.5
    assert row["score"] == 0.875


def test_guardrail_rejects_invalid_case_source_hash() -> None:
    with pytest.raises(ValueError, match="hash"):
        ConsolidationGuardrailCase(
            case_id="bad",
            candidate_type="claim",
            required_source_events=[{"seq": 1, "hash": "short"}],
        )


def test_summarize_consolidation_guardrail_averages_rows() -> None:
    rows = [
        {
            "case_id": "one",
            "type_match": 1.0,
            "source_event_fidelity": 1.0,
            "authority_boundary": 1.0,
            "review_gate": 1.0,
            "score": 1.0,
        },
        {
            "case_id": "two",
            "type_match": 1.0,
            "source_event_fidelity": 0.5,
            "authority_boundary": 1.0,
            "review_gate": 0.0,
            "score": 0.625,
        },
    ]

    summary = summarize_consolidation_guardrail(rows)

    assert summary == {
        "case_count": 2,
        "mean_type_match": 1.0,
        "mean_source_event_fidelity": 0.75,
        "mean_authority_boundary": 1.0,
        "mean_review_gate": 0.5,
        "mean_score": 0.8125,
    }


def test_summarize_consolidation_guardrail_empty_defaults() -> None:
    assert summarize_consolidation_guardrail([]) == {
        "case_count": 0,
        "mean_type_match": 0.0,
        "mean_source_event_fidelity": 0.0,
        "mean_authority_boundary": 0.0,
        "mean_review_gate": 0.0,
        "mean_score": 0.0,
    }


def test_guardrail_rejects_invalid_case_identity_and_candidate_type() -> None:
    with pytest.raises(ValueError, match="case_id"):
        ConsolidationGuardrailCase(
            case_id="",
            candidate_type="claim",
            required_source_events=[{"seq": 1, "hash": "a" * 64}],
        )

    with pytest.raises(ValueError, match="candidate_type"):
        ConsolidationGuardrailCase(
            case_id="bad-type",
            candidate_type="unsupported",
            required_source_events=[{"seq": 1, "hash": "a" * 64}],
        )


def test_guardrail_rejects_non_mapping_candidate() -> None:
    case = ConsolidationGuardrailCase(
        case_id="claim-3",
        candidate_type="claim",
        required_source_events=[{"seq": 1, "hash": "a" * 64}],
    )

    with pytest.raises(ValueError, match="candidate"):
        evaluate_consolidation_guardrail(case, ["not", "a", "mapping"])  # type: ignore[arg-type]


def test_guardrail_scores_malformed_candidate_source_events_as_zero_fidelity() -> None:
    case = ConsolidationGuardrailCase(
        case_id="claim-4",
        candidate_type="claim",
        required_source_events=[{"seq": 1, "hash": "a" * 64}],
    )

    row = evaluate_consolidation_guardrail(
        case,
        {
            "candidate_type": "claim",
            "source_events": [{"seq": True, "hash": "a" * 64}],
            "review_status": "unknown",
            "authority_status": "non_authoritative",
        },
    )

    assert row["source_event_fidelity"] == 0.0
    assert row["review_gate"] == 0.0
    assert row["score"] == 0.5


def test_summarize_consolidation_guardrail_rejects_non_numeric_metrics() -> None:
    with pytest.raises(ValueError, match="score"):
        summarize_consolidation_guardrail(
            [
                {
                    "type_match": 1.0,
                    "source_event_fidelity": 1.0,
                    "authority_boundary": 1.0,
                    "review_gate": 1.0,
                    "score": True,
                }
            ]
        )
