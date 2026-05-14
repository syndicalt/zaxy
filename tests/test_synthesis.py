"""Tests for structured synthesis planning and evidence ledgers."""

from __future__ import annotations

from zaxy.synthesis import (
    build_currency_ledger,
    build_synthesis_plan,
    render_currency_result,
)


def test_build_synthesis_plan_classifies_currency_sum() -> None:
    """Money-total queries should produce a deterministic sum plan."""
    plan = build_synthesis_plan(
        "How much total money have I spent on bike-related expenses?"
    )

    assert plan.answer_type == "sum"
    assert plan.operation == "sum_values"
    assert plan.required_kinds == ("currency",)
    assert "money" in plan.subject_terms
    assert "total" in plan.reasons


def test_build_synthesis_plan_classifies_currency_difference() -> None:
    """Comparison money queries should produce a deterministic difference plan."""
    plan = build_synthesis_plan(
        "How much more did I spend on accommodations in Hawaii compared to Tokyo?"
    )

    assert plan.answer_type == "difference"
    assert plan.operation == "difference_between"
    assert plan.required_kinds == ("currency",)
    assert "comparison" in plan.reasons


def test_build_synthesis_plan_does_not_treat_duration_spent_as_currency() -> None:
    """Duration uses of 'spent' should not open the currency synthesis lane."""
    plan = build_synthesis_plan(
        "How many hours did I spend on chess and piano practice?"
    )

    assert plan.answer_type == "count"
    assert "currency" not in plan.required_kinds


def test_currency_ledger_deduplicates_repeated_items_with_exclusions() -> None:
    """Currency evidence should preserve duplicate decisions in the ledger."""
    ledger = build_currency_ledger(
        "How much total money have I spent on bike-related expenses?",
        [
            "session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "session_id=answer-3 I got a new set of bike lights installed, which were $40.",
            "session_id=answer-4 I recently got a new set of bike lights installed, which were $40.",
        ],
    )

    included = ledger.included(kind="currency")
    excluded = ledger.excluded(kind="currency")

    assert [row.source_group for row in included] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
    assert [row.value for row in included] == ["120.0", "25.0", "40.0"]
    assert len(excluded) == 1
    assert excluded[0].source_group == "answer-4"
    assert excluded[0].exclude_reason == "duplicate_identity"
    assert excluded[0].normalized_identity == included[2].normalized_identity


def test_render_currency_result_projects_sum_and_exclusion_diagnostics() -> None:
    """Currency synthesis should render answer fields from the evidence ledger."""
    ledger = build_currency_ledger(
        "How much total money have I spent on bike-related expenses?",
        [
            "session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "session_id=answer-3 I got a new set of bike lights installed, which were $40.",
            "session_id=answer-4 I recently got a new set of bike lights installed, which were $40.",
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "candidate_rank=1 candidate_type=currency" in result.lines
    assert "currency_values=$120,$40,$25" in result.lines
    assert "currency_total_answer=$185" in result.lines
    assert "currency_source_ids=answer-1,answer-2,answer-3" in result.lines
    assert "currency_excluded_source_ids=answer-4" in result.lines
    assert result.support_source_groups == ("answer-1", "answer-2", "answer-3")
    assert result.excluded_source_groups == ("answer-4",)


def test_render_currency_result_projects_difference_answer() -> None:
    """Currency difference queries should expose a direct answer candidate."""
    ledger = build_currency_ledger(
        "How much more did I spend on accommodations in Hawaii compared to Tokyo?",
        [
            "session_id=answer-1 I booked a Maui resort for $300 per night.",
            "session_id=answer-2 I stayed in a Tokyo hostel for $30 per night.",
        ],
    )

    result = render_currency_result(ledger, rank=2)

    assert "candidate_rank=2 candidate_type=currency" in result.lines
    assert "currency_difference=$270" in result.lines
    assert "currency_difference_answer=$270" in result.lines
