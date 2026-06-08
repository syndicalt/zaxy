"""Internal reasoning-loop guardrail tests."""

from __future__ import annotations

from zaxy.reasoning_benchmark import score_metacognition_guardrail, score_reasoning_guardrail


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


def test_beta2_guardrail_scores_metacognition_and_procedural_contracts() -> None:
    result = score_metacognition_guardrail(
        [
            {
                "event_type": "metacognition.unknown.recorded",
                "authority_status": "non_authoritative",
            },
            {
                "event_type": "metacognition.confidence.assessed",
                "authority_status": "non_authoritative",
            },
            {
                "event_type": "metacognition.conflict.clustered",
                "authority_status": "non_authoritative",
            },
            {
                "event_type": "metacognition.reverify.requested",
                "status": "open",
                "authority_status": "non_authoritative",
            },
            {
                "event_type": "skill.validated",
                "procedural_bucket": "applicable",
                "phase": "planning",
                "citation": "eventloom://agent-1/events/55#dddddddddddd",
                "authority_status": "non_authoritative",
                "question_id": "longmemeval-should-not-matter",
                "answer": "answer text must not be a scoring target",
                "benchmark_name": "LongMemEval",
            },
        ]
    )

    assert result == {
        "case_count": 5,
        "observable_metacognition": 1.0,
        "open_reverify_status": 1.0,
        "procedural_citation_presence": 1.0,
        "planning_phase_match": 1.0,
        "authority_boundary": 1.0,
        "score": 1.0,
    }


def test_beta2_guardrail_penalizes_contract_failures_without_answer_scoring() -> None:
    base_rows = [
        {
            "event_type": "memory.fact.updated",
            "authority_status": "authoritative",
            "answer": "wrong answer",
            "benchmark_name": "LongMemBench",
            "question_id": "q-1",
        },
        {
            "event_type": "metacognition.reverify.requested",
            "status": "closed",
            "authority_status": "non_authoritative",
            "answer": "right answer",
            "benchmark_name": "LongMemEval",
            "question_id": "q-2",
        },
        {
            "event_type": "skill.revised",
            "procedural_bucket": "applicable",
            "phase": "execution",
            "citation": "",
            "authority_status": "non_authoritative",
            "answer": "another answer",
            "benchmark_name": "custom-suite",
            "question_id": "q-3",
        },
    ]

    result = score_metacognition_guardrail(base_rows)
    changed_non_contract_fields = [
        {**row, "answer": "changed", "benchmark_name": "changed", "question_id": "changed"}
        for row in base_rows
    ]

    assert result["case_count"] == 3
    assert result["observable_metacognition"] == 0.5
    assert result["open_reverify_status"] == 0.0
    assert result["procedural_citation_presence"] == 0.0
    assert result["planning_phase_match"] == 0.0
    assert result["authority_boundary"] == 0.667
    assert result["score"] == 0.233
    assert score_metacognition_guardrail(changed_non_contract_fields) == result


def test_beta2_guardrail_empty_input_scores_zero_without_rewarding_absence() -> None:
    assert score_metacognition_guardrail([]) == {
        "case_count": 0,
        "observable_metacognition": 0.0,
        "open_reverify_status": 0.0,
        "procedural_citation_presence": 0.0,
        "planning_phase_match": 0.0,
        "authority_boundary": 0.0,
        "score": 0.0,
    }
