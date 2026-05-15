"""Tests for structured synthesis planning and evidence ledgers."""

from __future__ import annotations

from zaxy.synthesis import (
    build_count_ledger,
    build_currency_ledger,
    build_date_ledger,
    build_duration_ledger,
    build_synthesis_plan,
    render_count_result,
    render_currency_result,
    render_date_interval_result,
    render_duration_result,
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

    assert plan.answer_type == "sum"
    assert plan.operation == "sum_values"
    assert plan.required_kinds == ("duration",)


def test_count_ledger_counts_distinct_relevant_sources() -> None:
    """Count evidence should preserve exclusions for duplicate cited sources."""
    ledger = build_count_ledger(
        "How many movie festivals did I attend, and which were they?",
        [
            "content=longmemeval_session_id=answer-1 I attended the Spring Film Festival. festival festival.",
            "content=longmemeval_session_id=answer-2 I attended the Lakeside Film Festival.",
            "content=longmemeval_session_id=answer-3 I attended the Indie Film Festival.",
            "content=longmemeval_session_id=answer-4 I attended the Documentary Film Festival.",
            "content=longmemeval_session_id=answer-4 I attended the Documentary Film Festival again.",
            "content=longmemeval_session_id=distractor-1 I watched a movie at home.",
        ],
    )

    included = ledger.included(kind="event")
    excluded = ledger.excluded(kind="event")

    assert [row.source_group for row in included] == [
        "answer-1",
        "answer-2",
        "answer-3",
        "answer-4",
    ]
    assert [row.value for row in included] == ["1", "1", "1", "1"]
    assert [row.label for row in included] == [
        "attended the Spring Film Festival",
        "attended the Lakeside Film Festival",
        "attended the Indie Film Festival",
        "attended the Documentary Film Festival",
    ]
    assert {row.source_group for row in excluded} == {"answer-4", "distractor-1"}
    assert {row.exclude_reason for row in excluded} == {
        "duplicate_source_group",
        "query_focus_mismatch",
    }


def test_count_ledger_requires_query_action_for_event_counts() -> None:
    """Count synthesis should not count broad topical overlap without the queried action."""
    ledger = build_count_ledger(
        "How many movie festivals that I attended?",
        [
            "session_id=answer-1 I attended the Sundance Film Festival.",
            "session_id=answer-2 I went to AFI Fest in LA.",
            "session_id=distractor-1 I watched all 22 Marvel movies in two weeks.",
            (
                "session_id=distractor-2 Film festival distribution strategy "
                "is interesting, but I did not attend one."
            ),
        ],
    )

    included = ledger.included(kind="event")
    excluded = ledger.excluded(kind="event")

    assert [row.source_group for row in included] == ["answer-1", "answer-2"]
    assert [row.label for row in included] == [
        "attended the Sundance Film Festival",
        "went to AFI Fest in LA",
    ]
    assert {row.source_group for row in excluded} == {"distractor-1", "distractor-2"}


def test_count_ledger_uses_relevant_user_span_inside_noisy_session() -> None:
    """Count evidence should choose a compact user memory span from noisy sessions."""
    ledger = build_count_ledger(
        "How many properties did I view before making an offer?",
        [
            (
                "session_id=answer-1 assistant: You can compare mortgage rates. "
                "user: I viewed a bungalow, but the kitchen needed renovation. "
                "assistant: Neighborhood research can help."
            ),
            (
                "session_id=answer-2 assistant: I can suggest listings. "
                "user: I toured a condo near the highway and passed because of the noise."
            ),
            (
                "session_id=distractor-1 assistant: Property taxes vary by city. "
                "user: I made an offer on the Brookside townhouse."
            ),
        ],
    )

    included = ledger.included(kind="event")

    assert [row.source_group for row in included] == ["answer-1", "answer-2"]
    assert [row.label for row in included] == [
        "viewed a bungalow, but the kitchen needed renovation",
        "toured a condo near the highway and passed because of the noise",
    ]


def test_count_ledger_rejects_music_festival_for_movie_festival_query() -> None:
    """Count facets should distinguish film festivals from generic festival mentions."""
    ledger = build_count_ledger(
        "How many movie festivals did I attend?",
        [
            "session_id=answer-1 I attended Sundance.",
            "session_id=answer-2 I went to AFI Fest in LA.",
            "session_id=distractor-1 I attended the iHeartRadio Music Festival.",
            "session_id=distractor-2 I went to a neighborhood food festival.",
        ],
    )

    included = ledger.included(kind="event")
    excluded = ledger.excluded(kind="event")

    assert [row.source_group for row in included] == ["answer-1", "answer-2"]
    assert {row.source_group for row in excluded} == {"distractor-1", "distractor-2"}


def test_count_ledger_excludes_target_offer_when_query_asks_viewed_before_offer() -> None:
    """Property count synthesis should not count the final offer as a prior viewing."""
    ledger = build_count_ledger(
        "How many properties did I view before making an offer?",
        [
            "session_id=answer-1 I viewed a bungalow, but the kitchen needed renovation.",
            "session_id=answer-2 I toured a condo near the highway and passed.",
            "session_id=distractor-1 I made an offer on the Brookside townhouse.",
        ],
    )

    assert [row.source_group for row in ledger.included(kind="event")] == [
        "answer-1",
        "answer-2",
    ]
    assert [row.source_group for row in ledger.excluded(kind="event")] == ["distractor-1"]


def test_count_ledger_rejects_request_to_see_doctor_without_visit() -> None:
    """Doctor visit counts should require an actual visit or appointment event."""
    ledger = build_count_ledger(
        "How many doctors did I visit?",
        [
            "session_id=answer-1 I visited my primary care physician.",
            "session_id=answer-2 I saw an ENT about my sinus infection.",
            "session_id=answer-3 I had an appointment with a dermatologist.",
            "session_id=distractor-1 I requested to see a specific doctor.",
        ],
    )

    assert [row.source_group for row in ledger.included(kind="event")] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
    assert [row.source_group for row in ledger.excluded(kind="event")] == ["distractor-1"]


def test_count_ledger_counts_multiple_model_kits_in_one_span() -> None:
    """A single cited memory can contain multiple countable items."""
    ledger = build_count_ledger(
        "How many model kits have I worked on, and which were they?",
        [
            (
                "session_id=answer-1 I got a 1/72 scale B-29 bomber model kit "
                "and a 1/24 scale '69 Camaro."
            ),
            "session_id=answer-2 I finished a Tamiya 1/48 scale Spitfire Mk.V.",
            "session_id=distractor-1 I bought paint for my model kits.",
        ],
    )

    included = ledger.included(kind="event")

    assert [row.source_group for row in included] == ["answer-1", "answer-1", "answer-2"]
    assert [row.label for row in included] == [
        "1/72 scale B-29 bomber model kit",
        "1/24 scale '69 Camaro",
        "Tamiya 1/48 scale Spitfire Mk.V",
    ]
    assert [row.value for row in included] == ["1", "1", "1"]


def test_count_ledger_handles_first_person_contractions() -> None:
    """Count evidence should parse common first-person contractions."""
    ledger = build_count_ledger(
        "How many model kits have I worked on, and which were they?",
        [
            "session_id=answer-1 I've recently finished a simple Revell F-15 Eagle kit.",
            "session_id=answer-2 I'm starting a Tamiya 1/48 scale Spitfire Mk.V.",
            "session_id=distractor-1 I'd recommend checking local hobby stores.",
        ],
    )

    included = ledger.included(kind="event")

    assert [row.source_group for row in included] == ["answer-1", "answer-2"]
    assert [row.label for row in included] == [
        "Revell F-15 Eagle kit",
        "Tamiya 1/48 scale Spitfire Mk.V",
    ]


def test_render_count_result_projects_list_details_from_ledger() -> None:
    """Count/list synthesis should render the existing model-facing answer surface."""
    ledger = build_count_ledger(
        "How many weddings did I attend and who were the couples?",
        [
            "session_id=answer-1 I attended Rachel and Mike's wedding.",
            "session_id=answer-2 I attended Emily and Sarah's wedding.",
            "session_id=answer-3 I attended Jen and Tom's wedding.",
            "session_id=distractor-1 I planned a birthday dinner.",
        ],
    )

    result = render_count_result(
        ledger,
        "How many weddings did I attend and who were the couples?",
        rank=2,
    )

    assert "candidate_rank=2 candidate_type=count" in result.lines
    assert "count_answer=3" in result.lines
    assert "count_unit=events" in result.lines
    assert "count_source_ids=answer-1,answer-2,answer-3" in result.lines
    assert "count_answer_text=I attended three weddings." in result.lines
    assert "list_item_count=3" in result.lines
    assert (
        "list_items=attended Rachel and Mike's wedding | "
        "attended Emily and Sarah's wedding | attended Jen and Tom's wedding"
    ) in result.lines
    assert "list_source_ids=answer-1,answer-2,answer-3" in result.lines
    assert result.support_source_groups == ("answer-1", "answer-2", "answer-3")
    assert result.excluded_source_groups == ("distractor-1",)


def test_date_ledger_extracts_dates_with_session_year_and_exclusions() -> None:
    """Date evidence should keep duplicate and query-mismatch rows auditable."""
    ledger = build_date_ledger(
        "How many days had passed between Sunday mass and the Ash Wednesday service?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/02/20 (Mon) "
                "I attended Sunday mass at St. Mary's Church on January 2nd. "
                '{"content": "longmemeval_session_id=answer-1"}'
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/02/20 (Mon) "
                "I came from the Ash Wednesday service at the cathedral on February 1st."
            ),
            (
                "content=longmemeval_session_id=distractor-1 "
                "longmemeval_session_date=2023/03/26 (Sun) "
                "I attended the Holi celebration at my local temple on February 26th."
            ),
        ],
    )

    included = ledger.included(kind="date")
    excluded = ledger.excluded(kind="date")

    assert [(row.source_group, row.value) for row in included] == [
        ("answer-1", "2023-01-02"),
        ("answer-2", "2023-02-01"),
    ]
    assert [(row.source_group, row.value, row.exclude_reason) for row in excluded] == [
        ("distractor-1", "2023-02-26", "query_focus_mismatch")
    ]


def test_render_date_interval_result_ranks_query_specific_interval() -> None:
    """Date interval synthesis should render the existing cited answer fields."""
    ledger = build_date_ledger(
        "How many days had passed between Sunday mass and the Ash Wednesday service?",
        [
            (
                "content=longmemeval_session_id=distractor-1 "
                "longmemeval_session_date=2023/03/26 (Sun) "
                "I just got back from Sunday mass at St. Mary's Church on March 19th."
            ),
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/02/20 (Mon) "
                "I came from the Ash Wednesday service at the cathedral on February 1st."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/02/20 (Mon) "
                "I attended Sunday mass at St. Mary's Church on January 2nd."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=1)

    assert "candidate_rank=1 candidate_type=date_interval" in result.lines
    assert "candidate_support=answer-1,answer-2" in result.lines
    assert "date_interval_days=30" in result.lines
    assert (
        "date_interval_answer=30 days. 31 days (including the last day) is also acceptable."
        in result.lines
    )
    assert "date_interval_source_ids=answer-1,answer-2" in result.lines
    assert result.support_source_groups == ("answer-1", "answer-2")


def test_date_ledger_parses_black_friday_relative_dates() -> None:
    """Named holiday-relative dates should stay available in the date ledger."""
    ledger = build_date_ledger(
        "How many days before I bought the iPhone 13 Pro did I attend the Holiday Market?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/12/10 (Sun) "
                "I attended the annual Holiday Market a week before Black Friday."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/12/10 (Sun) "
                "I bought the iPhone 13 Pro from Best Buy on Black Friday."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=2)

    assert "candidate_rank=2 candidate_type=date_interval" in result.lines
    assert "date_interval_days=7" in result.lines
    assert "date_interval_source_ids=answer-1,answer-2" in result.lines


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


def test_duration_ledger_normalizes_mixed_units() -> None:
    """Duration evidence should normalize minute and hour values into one ledger."""
    ledger = build_duration_ledger(
        "How many hours did I spend on chess and piano practice?",
        [
            "content=longmemeval_session_id=answer-1 I played chess for 90 minutes.",
            "content=longmemeval_session_id=answer-2 I practiced piano for 2 hours.",
            "content=longmemeval_session_id=distractor-1 I bought a book for $12.",
        ],
    )

    included = ledger.included(kind="duration")

    assert [row.source_group for row in included] == ["answer-1", "answer-2"]
    assert [row.value for row in included] == ["90.0", "120.0"]
    assert [row.unit for row in included] == ["minutes", "minutes"]
    assert [row.label for row in included] == ["90 minutes", "2 hours"]


def test_duration_ledger_deduplicates_json_echoes() -> None:
    """Duration evidence should not count Eventloom JSON payload echoes twice."""
    ledger = build_duration_ledger(
        "How many days did I spend on camping trips?",
        [
            (
                "# Event 1 citation=eventloom://benchmark/events/1#abc "
                "content=longmemeval_session_id=answer-1 I completed a 5-day camping trip. "
                '{"content": "longmemeval_session_id=answer-1 I completed a 5-day camping trip."}'
            ),
            (
                "# Event 2 citation=eventloom://benchmark/events/2#abc "
                "content=longmemeval_session_id=answer-2 I completed a 3-day camping trip. "
                '{"content": "longmemeval_session_id=answer-2 I completed a 3-day camping trip."}'
            ),
        ],
    )

    result = render_duration_result(ledger, rank=1)

    assert "day_values=5,3" in result.lines
    assert "day_total=8 days" in result.lines
    assert "day_total=16 days" not in result.lines


def test_render_duration_result_projects_answer_and_compatibility_lines() -> None:
    """Duration synthesis should preserve model-facing total and compatibility fields."""
    ledger = build_duration_ledger(
        "How many hours did I spend on chess and piano practice?",
        [
            "content=longmemeval_session_id=answer-1 I played chess for 90 minutes.",
            "content=longmemeval_session_id=answer-2 I practiced piano for 2 hours.",
        ],
    )

    result = render_duration_result(ledger, rank=3)

    assert "candidate_rank=3 candidate_type=duration" in result.lines
    assert "duration_values=90 minutes,2 hours" in result.lines
    assert "duration_total_minutes=210 minutes" in result.lines
    assert "duration_total_hours=3.5 hours" in result.lines
    assert "duration_total_answer=3.5 hours" in result.lines
    assert "duration_source_ids=answer-1,answer-2" in result.lines
    assert "minute_values=90" in result.lines
    assert "minute_total_hours=1.5 hours" in result.lines
    assert "hour_values=2" in result.lines
    assert "hour_total=2 hours" in result.lines
