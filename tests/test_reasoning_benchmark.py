"""Internal beta.1 reasoning-loop guardrail tests."""

from __future__ import annotations

from zaxy.reasoning_benchmark import score_reasoning_guardrail


def test_reasoning_guardrail_scores_observable_cited_phase_matched_non_authority_cases() -> None:
    result = score_reasoning_guardrail(
        [
            {
                "primitive": "explain_outcome",
                "expected_phase": "planning",
                "phase": "planning",
                "event_type": "reasoning.primitive.called",
                "citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa",
                "authority_status": "non_authoritative",
            },
            {
                "primitive": "propose_belief_update",
                "expected_phase": "reflection",
                "phase": "reflection",
                "event_type": "belief.update.proposed",
                "citation": "eventloom://agent-1/events/43#bbbbbbbbbbbb",
                "review_status": "pending",
                "authority_status": "non_authoritative",
            },
        ]
    )

    assert result == {
        "case_count": 2,
        "observable_call": 1.0,
        "phase_match": 1.0,
        "citation_presence": 1.0,
        "authority_boundary": 1.0,
        "score": 1.0,
    }


def test_reasoning_guardrail_penalizes_hidden_uncited_wrong_phase_or_authoritative_cases() -> None:
    result = score_reasoning_guardrail(
        [
            {
                "primitive": "get_claim_confidence",
                "expected_phase": "review",
                "phase": "execution",
                "event_type": "memory.fact.updated",
                "citation": "",
                "authority_status": "authoritative",
            },
            {
                "primitive": "retrieve_similar_procedures",
                "expected_phase": "planning",
                "phase": "planning",
                "event_type": "reasoning.primitive.called",
                "citations": ["eventloom://agent-1/events/44#cccccccccccc"],
                "authority_status": "non_authoritative",
            },
        ]
    )

    assert result["case_count"] == 2
    assert result["observable_call"] == 0.5
    assert result["phase_match"] == 0.5
    assert result["citation_presence"] == 0.5
    assert result["authority_boundary"] == 0.5
    assert result["score"] == 0.5


def test_reasoning_guardrail_empty_input_scores_zero_without_rewarding_absence() -> None:
    assert score_reasoning_guardrail([]) == {
        "case_count": 0,
        "observable_call": 0.0,
        "phase_match": 0.0,
        "citation_presence": 0.0,
        "authority_boundary": 0.0,
        "score": 0.0,
    }
