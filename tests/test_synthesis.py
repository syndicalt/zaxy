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
        "Spring Film Festival",
        "Lakeside Film Festival",
        "Indie Film Festival",
        "Documentary Film Festival",
    ]
    assert {row.source_group for row in excluded} == {"answer-4", "distractor-1"}
    assert {row.exclude_reason for row in excluded} == {
        "duplicate_identity",
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
        "Sundance Film Festival",
        "AFI Fest",
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


def test_count_ledger_counts_multiple_model_kits_across_one_session() -> None:
    """Itemized count extraction should not collapse one source to one memory span."""
    ledger = build_count_ledger(
        "How many model kits have I worked on or bought?",
        [
            (
                "session_id=answer-1 I recently finished a simple Revell F-15 Eagle kit. "
                "I'm thinking of working on a 1/72 scale B-29 bomber next."
            ),
        ],
    )

    included = ledger.included(kind="event")

    assert [row.source_group for row in included] == ["answer-1", "answer-1"]
    assert [row.label for row in included] == [
        "Revell F-15 Eagle kit",
        "1/72 scale B-29 bomber",
    ]


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


def test_count_ledger_counts_planned_model_kit_and_multiple_new_kits() -> None:
    """Model-kit synthesis should count active project language and multi-item purchases."""
    ledger = build_count_ledger(
        "How many model kits have I worked on or bought?",
        [
            "session_id=answer-1 I'm thinking of working on a 1/72 scale B-29 bomber next.",
            "session_id=answer-2 I just got this kit and a 1/24 scale '69 Camaro at a model show.",
            "session_id=answer-3 I started working on a diorama featuring a 1/16 scale German Tiger I tank.",
        ],
    )

    result = render_count_result(
        ledger,
        "How many model kits have I worked on or bought?",
        rank=1,
    )

    assert "count_answer=3" in result.lines
    assert "count_answer_text=I have worked on or bought three model kits." in result.lines
    assert any("1/72 scale B-29 bomber" in line for line in result.lines)
    assert any("1/24 scale '69 Camaro" in line for line in result.lines)


def test_count_ledger_counts_new_model_kit_possession() -> None:
    """New owned model-kit wording should count as acquired project evidence."""
    ledger = build_count_ledger(
        "How many model kits have I worked on or bought?",
        [
            "session_id=answer-1 I'm looking for tips on photo-etching for my new 1/72 scale B-29 bomber model kit.",
        ],
    )

    included = ledger.included(kind="event")

    assert [row.label for row in included] == ["1/72 scale B-29 bomber model kit"]


def test_count_ledger_deduplicates_model_kit_suffix_variants() -> None:
    """Model-kit identities should survive repeated shorthand mentions."""
    ledger = build_count_ledger(
        "How many model kits have I worked on or bought?",
        [
            "session_id=answer-1 I got a new 1/72 scale B-29 bomber model kit.",
            "session_id=answer-2 I'm working on my 1/72 scale B-29 bomber.",
            "session_id=answer-3 I got a 1/24 scale '69 Camaro at a model show.",
            "session_id=answer-4 I'm working on my 1/24 scale '69 Camaro model as well.",
        ],
    )

    result = render_count_result(
        ledger,
        "How many model kits have I worked on or bought?",
        rank=1,
    )

    assert "count_answer=2" in result.lines


def test_count_ledger_extracts_distinct_doctor_roles_from_consultations() -> None:
    """Doctor visit synthesis should count provider roles, not only source sessions."""
    ledger = build_count_ledger(
        "How many different doctors did I visit?",
        [
            "session_id=answer-1 I was prescribed antibiotics by my primary care physician, Dr. Smith.",
            "session_id=answer-2 I got diagnosed with chronic sinusitis by an ENT specialist, Dr. Patel.",
            "session_id=answer-3 I got back from a follow-up appointment with my dermatologist, Dr. Lee.",
        ],
    )

    result = render_count_result(ledger, "How many different doctors did I visit?", rank=1)

    assert "count_answer=3" in result.lines
    assert any("primary care physician" in line for line in result.lines)
    assert any("ENT specialist" in line for line in result.lines)
    assert any("dermatologist" in line for line in result.lines)


def test_count_ledger_excludes_generic_doctor_when_specific_roles_are_present() -> None:
    """Doctor synthesis should not double-count generic follow-up wording."""
    ledger = build_count_ledger(
        "How many different doctors did I visit?",
        [
            "session_id=answer-1 I recently had a UTI and was prescribed antibiotics by my primary care physician.",
            "session_id=answer-2 I've recently been diagnosed by an ENT specialist.",
            "session_id=answer-3 I just got back from a follow-up appointment with my dermatologist. I asked my doctor about biopsy results.",
            "session_id=answer-4 I asked my doctor what questions to bring to the next appointment.",
        ],
    )

    result = render_count_result(ledger, "How many different doctors did I visit?", rank=1)

    assert "count_answer=3" in result.lines
    assert any(
        "list_items=primary care physician | ENT specialist | dermatologist" in line
        for line in result.lines
    )
    assert (
        "count_answer_text=I visited three different doctors: "
        "primary care physician, ENT specialist, and dermatologist."
    ) in result.lines


def test_count_ledger_extracts_current_musical_instruments_with_durations() -> None:
    """Instrument ownership synthesis should list owned instruments and ignore planned ones."""
    ledger = build_count_ledger(
        "How many musical instruments do I currently own?",
        [
            "session_id=answer-1 I've had my black Fender Stratocaster electric guitar for about 5 years now.",
            "session_id=answer-2 I've had my acoustic guitar, a Yamaha FG800, for about 8 years.",
            "session_id=answer-3 I'm thinking of selling my old drum set, a 5-piece Pearl Export.",
            "session_id=answer-4 I'm looking to find a piano technician to service my Korg B1, which I've had for about 3 years.",
            "session_id=distractor I'm thinking about getting a new ukulele.",
        ],
    )

    result = render_count_result(
        ledger,
        "How many musical instruments do I currently own?",
        rank=1,
    )

    assert "count_answer=4" in result.lines
    assert "count_answer_text=I currently own four musical instruments." in result.lines
    assert any(
        "list_items=Fender Stratocaster electric guitar | Yamaha FG800 acoustic guitar | "
        "5-piece Pearl Export drum set | Korg B1 piano" in line
        for line in result.lines
    )
    assert any("instrument_ownership_answer=" in line for line in result.lines)
    assert any("Fender Stratocaster electric guitar for 5 years" in line for line in result.lines)
    assert any("5-piece Pearl Export drum set for an unspecified amount of time" in line for line in result.lines)


def test_count_ledger_counts_participatory_film_festival_memories() -> None:
    """Festival memories can be attendance evidence through volunteered/participated wording."""
    ledger = build_count_ledger(
        "How many movie festivals that I attended?",
        [
            "session_id=answer-1 I volunteered at the Portland Film Festival.",
            "session_id=answer-2 I participated in the 48-hour film challenge at the Austin Film Festival.",
            "session_id=answer-3 I got back from AFI Fest in LA, where I attended a screening.",
            "session_id=distractor I watched a movie at home.",
        ],
    )

    result = render_count_result(ledger, "How many movie festivals that I attended?", rank=1)

    assert "count_answer=3" in result.lines
    assert result.support_source_groups == ("answer-1", "answer-2", "answer-3")


def test_count_ledger_counts_multiple_named_festivals_inside_one_session() -> None:
    """A cited session can contain multiple distinct attended festival memories."""
    ledger = build_count_ledger(
        "How many movie festivals that I attended?",
        [
            (
                "session_id=answer-1 I participated in the 48-hour film challenge at the Austin Film Festival. "
                "I got to discuss The Weight of Water with the director after a Q&A at the Seattle International Film Festival."
            ),
        ],
    )

    included = ledger.included(kind="event")

    assert [row.source_group for row in included] == ["answer-1", "answer-1"]
    assert [row.label for row in included] == [
        "Austin Film Festival",
        "Seattle International Film Festival",
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
    assert (
        "count_answer_text=I attended three weddings. The couples were "
        "Rachel and Mike, Emily and Sarah, and Jen and Tom."
    ) in result.lines
    assert "list_item_count=3" in result.lines
    assert (
        "list_items=attended Rachel and Mike's wedding | "
        "attended Emily and Sarah's wedding | attended Jen and Tom's wedding"
    ) in result.lines
    assert "list_source_ids=answer-1,answer-2,answer-3" in result.lines
    assert result.support_source_groups == ("answer-1", "answer-2", "answer-3")
    assert result.excluded_source_groups == ("distractor-1",)


def test_count_ledger_counts_weddings_without_explicit_attended_word() -> None:
    """Wedding attendance can be phrased as being at or returning from a wedding."""
    ledger = build_count_ledger(
        "How many weddings have I attended in this year?",
        [
            "session_id=answer-1 I've been to my cousin's wedding at a vineyard in August.",
            (
                "session_id=answer-2 I just got back from my college roommate's wedding; "
                "Emily tied the knot with her partner Sarah."
            ),
            (
                "session_id=answer-3 I just got back from a friend's wedding last weekend, "
                "and the bride, Jen, looked stunning with her husband, Tom."
            ),
        ],
    )

    result = render_count_result(
        ledger,
        "How many weddings have I attended in this year?",
        rank=1,
    )

    assert "count_answer=3" in result.lines
    assert result.support_source_groups == ("answer-1", "answer-2", "answer-3")


def test_count_ledger_counts_property_search_outcomes_before_final_offer() -> None:
    """Property-search synthesis should include rejected/out-of-budget prior candidates."""
    ledger = build_count_ledger(
        "How many properties did I view before making an offer on the townhouse in Brookside?",
        [
            "session_id=answer-1 I saw a 3-bedroom bungalow, but the kitchen needed renovation.",
            "session_id=answer-2 I had seen some properties that did not fit my budget, like Cedar Creek.",
            "session_id=answer-3 I viewed a 1-bedroom condo, but highway noise was a deal-breaker.",
            "session_id=answer-4 I fell in love with a 2-bedroom condo, but my offer got rejected due to a higher bid.",
            "session_id=target-view I forgot to mention that I saw the 3-bedroom townhouse in the Brookside neighborhood.",
            "session_id=final I put in an offer on a 3-bedroom townhouse in Brookside and agreed on a price.",
        ],
    )

    result = render_count_result(
        ledger,
        "How many properties did I view before making an offer on the townhouse in Brookside?",
        rank=1,
    )

    assert "count_answer=4" in result.lines
    assert any("Cedar Creek" in line for line in result.lines)
    assert "count_answer_text=I viewed four properties." in result.lines
    assert result.support_source_groups == ("answer-1", "answer-2", "answer-3", "answer-4")
    assert result.excluded_source_groups == ("target-view", "final")


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


def test_date_ledger_preserves_lower_relevance_cross_source_anchor() -> None:
    """Date intervals need cross-source anchors, not only the highest-overlap source."""
    ledger = build_date_ledger(
        "How many days did it take for me to find a house I loved after starting to work with Rachel?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2022/03/02 (Wed) "
                "user: Since I started working with her on 2/15, I want new listings."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2022/03/02 (Wed) "
                "user: I got mortgage pre-approval on February 10th. "
                "I saw a house I really love after starting to work with Rachel on 3/1."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=1)

    assert "date_interval_days=14" in result.lines
    assert (
        "date_interval_answer=14 days. 15 days (including the last day) is also acceptable."
        in result.lines
    )
    assert "date_interval_source_ids=answer-1,answer-2" in result.lines


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


def test_date_interval_result_projects_week_answers_for_week_queries() -> None:
    """Week interval queries should synthesize from explicit dates."""
    ledger = build_date_ledger(
        "How many weeks had I been accepted when I started orientation?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/04/19 (Wed) "
                "user: I got accepted on March 20th."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/04/19 (Wed) "
                "user: I started orientation sessions on 3/27."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=1)

    assert "date_interval_days=7" in result.lines
    assert "date_interval_weeks=1 weeks" in result.lines
    assert "date_interval_week_answer=One week" in result.lines


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


def test_currency_ledger_deduplicates_item_repeated_in_later_session() -> None:
    """Repeated expense mentions should not double count the same concrete item."""
    ledger = build_currency_ledger(
        "How much total money have I spent on bike-related expenses since the start of the year?",
        [
            "session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "session_id=answer-2 The mechanic replaced the bike chain, which cost me $25.",
            "session_id=answer-3 I recently got a new set of bike lights installed, which were $40.",
            (
                "session_id=answer-4 Speaking of my bike, I recently got a new set of "
                "bike lights installed, which were $40."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_total_answer=$185" in result.lines
    assert "currency_source_ids=answer-1,answer-2,answer-3" in result.lines
    assert "answer-4" in ",".join(result.excluded_source_groups)


def test_currency_ledger_deduplicates_unlabeled_repeat_inside_source_group() -> None:
    """An unlabeled amount from the same source/value should not double count labeled evidence."""
    ledger = build_currency_ledger(
        "How much total money have I spent on bike-related expenses since the start of the year?",
        [
            (
                "session_id=answer-1 I recently got a new set of bike lights installed, "
                "which were $40. The receipt total was $40."
            ),
        ],
    )

    included = ledger.included(kind="currency")
    excluded = ledger.excluded(kind="currency")

    assert [row.value for row in included] == ["40.0"]
    assert [row.exclude_reason for row in excluded] == ["duplicate_source_value"]


def test_currency_ledger_excludes_planned_purchase_for_spent_queries() -> None:
    """Spent-total synthesis should ignore future or hypothetical prices."""
    ledger = build_currency_ledger(
        "How much total money have I spent on bike-related expenses since the start of the year?",
        [
            "session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            (
                "session_id=distractor I found a good deal on a bike rack for $500, "
                "which I think I'm going to order next week."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_total_answer=$120" in result.lines
    assert "distractor" in ",".join(result.excluded_source_groups)


def test_currency_result_projects_max_label_answer_for_store_queries() -> None:
    """Most-spent queries should expose the highest-value merchant as an answer."""
    ledger = build_currency_ledger(
        "Which grocery store did I spend the most money at in the past month?",
        [
            "session_id=answer-1 I went grocery shopping and spent around $120 at Walmart.",
            "session_id=answer-2 My sister and I went to Trader Joe's and spent around $80.",
            (
                "session_id=answer-3 I placed an online order with Thrive Market "
                "last month and spent around $150."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_max=$150" in result.lines
    assert "currency_max_label=Thrive Market" in result.lines
    assert "currency_max_answer=Thrive Market" in result.lines


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


def test_currency_ledger_uses_local_value_relevance_for_comparisons() -> None:
    """Currency comparison should not select prices from broad topical distractors."""
    ledger = build_currency_ledger(
        "How much more did I spend on accommodations per night in Hawaii compared to Tokyo?",
        [
            (
                "session_id=distractor Hawaii was mentioned in this travel chat, "
                "but the local price list was for Southeast Asia hostels at $200, $80, and $50."
            ),
            "session_id=answer-hawaii I booked a Maui resort in Hawaii for $300 per night.",
            "session_id=answer-tokyo I stayed in a Tokyo hostel for $30 per night.",
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_values=$300,$30" in result.lines
    assert "currency_difference_answer=$270" in result.lines
    assert "currency_source_ids=answer-hawaii,answer-tokyo" in result.lines
    assert "distractor" in ",".join(result.excluded_source_groups)


def test_currency_ledger_ignores_advice_prices_for_personal_spend_queries() -> None:
    """Personal money synthesis should use user memory amounts, not assistant price lists."""
    ledger = build_currency_ledger(
        "How much more did I spend on accommodations per night in Hawaii compared to Tokyo?",
        [
            (
                "session_id=tokyo assistant: Dormitories range from $18 to $45 per night. "
                "user: I stayed in a hostel in Tokyo that cost around $30 per night."
            ),
            (
                "session_id=hawaii assistant: Maui tours can cost $50 or more. "
                "user: I've already booked a luxurious resort in Maui that costs over $300 per night."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_values=$300,$30" in result.lines
    assert "currency_difference_answer=$270" in result.lines


def test_currency_ledger_includes_team_raised_amounts_for_charity_totals() -> None:
    """Fundraising totals should include first-person team amounts and exclude advice amounts."""
    ledger = build_currency_ledger(
        "How much money did I raise in total through all the charity events I participated in?",
        [
            (
                "session_id=walk user: I participated in a charity walk and managed "
                "to raise $250 through sponsors."
            ),
            (
                "session_id=bike assistant: Sponsorship tiers can be $100 or $500. "
                "user: I recently participated in a Bike-a-Thon for Cancer Research "
                "and my team managed to raise $5,000."
            ),
            (
                "session_id=yoga user: I just helped organize a charity yoga event "
                "that raised $600 for a local animal shelter."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_total_answer=$5,850" in result.lines
    assert "currency_values=$5,000,$600,$250" in result.lines


def test_currency_ledger_deduplicates_labeled_source_value_echoes() -> None:
    """One source/value should not be counted twice when full and salient source rows both match."""
    ledger = build_currency_ledger(
        "How much money did I raise in total through all the charity events I participated in?",
        [
            (
                "session_id=yoga user: I just helped organize a charity yoga event "
                "that raised $600 for a local animal shelter."
            ),
            (
                "# Event 1 citation=eventloom://benchmark/events/1#abc "
                "content=longmemeval_session_id=yoga user: I just helped organize "
                "a charity yoga event that raised $600 for a local animal shelter."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_values=$600" in result.lines
    assert "currency_total_answer=$600" in result.lines
    assert "currency_total_answer=$1,200" not in result.lines


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


def test_duration_ledger_parses_word_number_plural_hours() -> None:
    """Duration synthesis should parse prose values like four hours and six hours."""
    ledger = build_duration_ledger(
        "How many hours in total did I spend driving to my three road trip destinations combined?",
        [
            "session_id=answer-1 It took me four hours to drive to Outer Banks.",
            "session_id=answer-2 I drove for six hours to Washington D.C.",
            "session_id=answer-3 I drove for five hours to the mountains in Tennessee.",
            "session_id=distractor assistant: One option takes 16 hours.",
            "session_id=distractor End at Zion, approx. 1.5 hours, then return via I-70.",
        ],
    )

    result = render_duration_result(ledger, rank=1)

    assert "duration_values=4 hours,6 hours,5 hours" in result.lines
    assert "duration_total_answer=15 hours" in result.lines


def test_duration_result_answers_in_days_for_day_queries() -> None:
    """Day-count queries should project day totals from mixed week/day evidence."""
    ledger = build_duration_ledger(
        "How many days did I take social media breaks in total?",
        [
            "session_id=answer-1 I took a week-long break from social media in January.",
            "session_id=answer-2 I took a 10-day break from social media in February.",
            "session_id=distractor I limited social media to 30 minutes for 24 hours.",
        ],
    )

    result = render_duration_result(ledger, rank=1)

    assert "duration_total_days=17 days" in result.lines
    assert "duration_total_answer=17 days" in result.lines
    assert "duration_source_ids=answer-1,answer-2" in result.lines
    assert "distractor" in ",".join(result.excluded_source_groups)


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


def test_duration_ledger_ignores_advice_hours_for_personal_game_totals() -> None:
    """Personal duration synthesis should not aggregate assistant gameplay estimates."""
    ledger = build_duration_ledger(
        "How many hours have I spent playing games in total?",
        [
            (
                "session_id=odyssey assistant: This game can take 100 hours for completionists. "
                "user: I spent around 70 hours playing Assassin's Creed Odyssey."
            ),
            (
                "session_id=last-of-us assistant: Similar games are often 40 hours long. "
                "user: I completed The Last of Us Part II and it took me 30 hours."
            ),
            (
                "session_id=zelda user: I spent 40 hours playing Breath of the Wild."
            ),
        ],
    )

    result = render_duration_result(ledger, rank=1)

    assert "duration_total_answer=140 hours" in result.lines
    assert "duration_values=70 hours,30 hours,40 hours" in result.lines
