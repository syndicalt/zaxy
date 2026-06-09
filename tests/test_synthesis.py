"""Tests for structured synthesis planning and evidence ledgers."""

from __future__ import annotations

import builtins

from zaxy import synthesis
from zaxy.synthesis import (
    AverageValuesOperation,
    DifferenceBetweenOperation,
    EvidenceLedger,
    EvidenceLedgerRow,
    ListItemsOperation,
    SumValuesOperation,
    SynthesisPlan,
    TemporalIntervalOperation,
    build_age_average_ledger,
    build_count_ledger,
    build_currency_ledger,
    build_date_ledger,
    build_duration_ledger,
    build_numeric_state_ledger,
    build_synthesis_plan,
    build_temporal_sequence_ledger,
    render_count_result,
    render_currency_result,
    render_date_interval_result,
    render_duration_result,
    render_numeric_state_result,
    render_temporal_sequence_result,
    source_group,
    synthesis_operation_for_plan,
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


def test_source_group_reads_source_id_metadata() -> None:
    """Source-backed synthesis contexts should retain compact source identifiers."""
    assert source_group("source_id=answer-123 longmemeval_session_date=2024/01/01 user: hello") == "answer-123"


def test_sum_operation_preserves_currency_renderer_contract() -> None:
    """Operation objects should execute existing ledger renderers without changing answer lines."""
    ledger = build_currency_ledger(
        "How much total money have I spent on bike-related expenses?",
        [
            "longmemeval_session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "longmemeval_session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "longmemeval_session_id=answer-3 I got bike lights for $40.",
        ],
    )

    result = SumValuesOperation(kind="currency").execute(ledger, rank=1)

    assert result.lines == render_currency_result(ledger, rank=1).lines
    assert "currency_total_answer=$185" in result.lines
    assert result.answer_candidate == {
        "rank": 1,
        "type": "currency",
        "confidence": 0.81,
        "answer_key": "currency_total_answer",
        "answer": "$185",
        "support_source_ids": ["answer-1", "answer-2", "answer-3"],
        "excluded_source_ids": [],
    }


def test_difference_operation_preserves_currency_difference_contract() -> None:
    """Difference operations should be a named pure projection over ledger rows."""
    ledger = build_currency_ledger(
        "How much more did I spend on accommodations in Hawaii compared to Tokyo?",
        [
            "longmemeval_session_id=answer-1 I spent $300 on the hotel in Hawaii.",
            "longmemeval_session_id=answer-2 I spent $30 on the capsule hotel in Tokyo.",
        ],
    )

    result = DifferenceBetweenOperation(kind="currency").execute(ledger, rank=2)

    assert result.lines == render_currency_result(ledger, rank=2).lines
    assert "currency_difference_answer=$270" in result.lines
    assert result.answer_candidate == {
        "rank": 2,
        "type": "currency",
        "confidence": 0.71,
        "answer_key": "currency_difference_answer",
        "answer": "$270",
        "support_source_ids": ["answer-1", "answer-2"],
        "excluded_source_ids": [],
    }


def test_list_operation_preserves_count_list_contract() -> None:
    """List operations should wrap count/list synthesis without duplicating renderer logic."""
    query = "How many movie festivals did I attend, and which were they?"
    ledger = build_count_ledger(
        query,
        [
            "longmemeval_session_id=answer-1 I attended the Spring Film Festival.",
            "longmemeval_session_id=answer-2 I attended the Lakeside Film Festival.",
        ],
    )

    result = ListItemsOperation().execute(ledger, query=query, rank=1)

    assert result.lines == render_count_result(ledger, query, rank=1).lines
    assert "count_answer=2" in result.lines
    assert result.answer_candidate == {
        "rank": 1,
        "type": "count",
        "confidence": 0.75,
        "answer_key": "count_answer_text",
        "answer": "I attended two movie festivals.",
        "support_source_ids": ["answer-1", "answer-2"],
        "excluded_source_ids": [],
    }


def test_count_candidate_prefers_longmemeval_answer_text() -> None:
    """Count synthesis should expose the answer-ready sentence as the primary candidate."""
    query = "How many weddings have I attended in this year?"
    ledger = build_count_ledger(
        query,
        [
            "longmemeval_session_id=answer-1 I attended Rachel and Mike's wedding.",
            "longmemeval_session_id=answer-2 I attended Emily and Sarah's wedding.",
            "longmemeval_session_id=answer-3 I attended Jen and Tom's wedding.",
        ],
    )

    result = render_count_result(ledger, query, rank=1)

    assert "count_answer=3" in result.lines
    assert "count_answer_text=I attended three weddings. The couples were Rachel and Mike, Emily and Sarah, and Jen and Tom." in result.lines
    assert result.answer_candidate == {
        "rank": 1,
        "type": "count",
        "confidence": 0.77,
        "answer_key": "count_answer_text",
        "answer": "I attended three weddings. The couples were Rachel and Mike, Emily and Sarah, and Jen and Tom.",
        "support_source_ids": ["answer-1", "answer-2", "answer-3"],
        "excluded_source_ids": [],
    }


def test_count_candidate_lists_kitchen_repair_and_replacement_items() -> None:
    """Kitchen repair/replacement queries should expose labeled item evidence."""
    query = "How many kitchen items did I replace or fix?"
    ledger = build_count_ledger(
        query,
        [
            "longmemeval_session_id=answer-1 I just replaced my old kitchen faucet with a new Moen one last Sunday.",
            "longmemeval_session_id=answer-2 my kitchen has been feeling so much more functional lately, especially with my new kitchen mat in front of the sink.",
            "longmemeval_session_id=answer-3 I just got rid of the old toaster and replaced it with a toaster oven that can do so much more.",
            "longmemeval_session_id=answer-4 I donated my old coffee maker to Goodwill and I'm really enjoying the upgrade.",
            "longmemeval_session_id=answer-5 I finally fixed the kitchen shelves last weekend.",
        ],
    )

    result = render_count_result(ledger, query, rank=1)

    assert "count_answer=5" in result.lines
    assert (
        "count_answer_text=I replaced or fixed five items: "
        "the kitchen faucet, the kitchen mat, the toaster, the coffee maker, and the kitchen shelves."
    ) in result.lines
    assert result.answer_candidate == {
        "rank": 1,
        "type": "count",
        "confidence": 0.93,
        "answer_key": "count_answer_text",
        "answer": (
            "I replaced or fixed five items: "
            "the kitchen faucet, the kitchen mat, the toaster, the coffee maker, and the kitchen shelves."
        ),
        "support_source_ids": ["answer-1", "answer-2", "answer-3", "answer-4", "answer-5"],
        "excluded_source_ids": [],
    }


def test_count_candidate_extracts_multiple_kitchen_items_from_one_context() -> None:
    """One cited source can contain multiple distinct durable item changes."""
    query = "How many kitchen items did I replace or fix?"
    ledger = build_count_ledger(
        query,
        [
            (
                "longmemeval_session_id=answer-1 "
                "1. user: I donated my old coffee maker to Goodwill and I'm really enjoying the upgrade. "
                "2. user: I've been decluttering my kitchen countertops and got rid of the old toaster, "
                "replacing it with a toaster oven."
            ),
            "longmemeval_session_id=answer-2 I finally fixed the kitchen shelves last weekend.",
        ],
    )

    result = render_count_result(ledger, query, rank=1)

    assert "count_answer=3" in result.lines
    assert (
        "count_answer_text=I replaced or fixed three items: "
        "the coffee maker, the toaster, and the kitchen shelves."
    ) in result.lines


def test_temporal_interval_operation_preserves_date_interval_contract() -> None:
    """Temporal interval operations should wrap date interval synthesis."""
    query = "How many days had passed between Sunday mass and the Ash Wednesday service?"
    ledger = build_date_ledger(
        query,
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/02/20 (Mon) "
                "I attended Sunday mass at St. Mary's Church on January 2nd."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/02/20 (Mon) "
                "I came from the Ash Wednesday service at the cathedral on February 1st."
            ),
        ],
    )

    result = TemporalIntervalOperation().execute(ledger, rank=1)

    assert result.lines == render_date_interval_result(ledger, rank=1).lines
    assert any(line.startswith("date_interval_answer=30 days") for line in result.lines)


def test_average_operation_projects_mean_from_included_ledger_values() -> None:
    """Average operations should expose a reusable pure projection for numeric ledgers."""
    plan = synthesis.SynthesisPlan(
        answer_type="average",
        operation="average_values",
        subject_terms=("age",),
        required_kinds=("number",),
        required_source_groups=2,
        reasons=("average",),
    )
    rows = (
        synthesis.EvidenceLedgerRow(
            fact_id="age:1",
            source_group="answer-1",
            citation="eventloom://agent/events/1#aaaaaaaaaaaa",
            kind="number",
            value="32",
            unit="years",
            label="self",
            raw_span="32",
            context="I am 32.",
            normalized_identity="age:self",
            relevance=3,
            include_reason="age_average_input",
            confidence=0.8,
        ),
        synthesis.EvidenceLedgerRow(
            fact_id="age:2",
            source_group="answer-2",
            citation="eventloom://agent/events/2#bbbbbbbbbbbb",
            kind="number",
            value="58",
            unit="years",
            label="dad",
            raw_span="58",
            context="Dad is 58.",
            normalized_identity="age:dad",
            relevance=3,
            include_reason="age_average_input",
            confidence=0.8,
        ),
    )
    ledger = synthesis.EvidenceLedger(plan=plan, rows=rows)

    result = AverageValuesOperation(kind="number", output_prefix="age").execute(ledger, rank=1)

    assert result.support_source_groups == ("answer-1", "answer-2")
    assert "age_values=32,58" in result.lines
    assert "age_average=45" in result.lines


def test_build_age_average_ledger_extracts_family_age_rows_from_sources() -> None:
    """Family age average evidence should be a typed number ledger."""
    ledger = build_age_average_ledger(
        "What is the average age of me, my parents, and my grandparents?",
        [
            "longmemeval_session_id=answer_1 I just turned 32 on February 12th.",
            "longmemeval_session_id=answer_2 my parents are getting older too - my mom is 55 and my dad is 58.",
            "longmemeval_session_id=answer_3 My grandma is 75 and my grandpa is 78.",
        ],
    )

    rows = ledger.included(kind="number")

    assert [(row.source_group, row.value, row.unit, row.include_reason) for row in rows] == [
        ("answer_1", "32", "years", "age_average_input"),
        ("answer_2", "55", "years", "age_average_input"),
        ("answer_2", "58", "years", "age_average_input"),
        ("answer_3", "75", "years", "age_average_input"),
        ("answer_3", "78", "years", "age_average_input"),
    ]


def test_age_average_operation_uses_typed_ledger_rows() -> None:
    """Age average should flow through the generic average operation."""
    ledger = build_age_average_ledger(
        "What is the average age of me, my parents, and my grandparents?",
        [
            "longmemeval_session_id=answer_1 I just turned 32 on February 12th.",
            "longmemeval_session_id=answer_2 my parents are getting older too - my mom is 55 and my dad is 58.",
            "longmemeval_session_id=answer_3 My grandma is 75 and my grandpa is 78.",
        ],
    )

    result = AverageValuesOperation(kind="number", output_prefix="age").execute(ledger, rank=1)

    assert "age_values=32,55,58,75,78" in result.lines
    assert "age_average=59.6" in result.lines
    assert result.answer_candidate == {
        "rank": 1,
        "type": "number",
        "confidence": 0.89,
        "answer_key": "age_average",
        "answer": "59.6",
        "support_source_ids": ["answer_1", "answer_2", "answer_3"],
        "excluded_source_ids": [],
    }


def test_synthesis_operation_registry_maps_plans_to_operation_objects() -> None:
    """Plan operation strings should resolve to reusable operation objects."""
    currency_sum = build_synthesis_plan("How much total money have I spent on bike-related expenses?")
    currency_difference = build_synthesis_plan("How much more did I spend on Hawaii compared to Tokyo?")
    duration_sum = build_synthesis_plan("How many hours did I spend on chess and piano practice?")
    date_interval = build_date_ledger(
        "How many days passed between Sunday mass and Ash Wednesday service?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/02/20 (Mon) "
                "I attended Sunday mass at St. Mary's Church on January 2nd."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/02/20 (Mon) "
                "I came from the Ash Wednesday service at the cathedral on February 1st."
            ),
        ],
    ).plan
    count = build_synthesis_plan("How many movie festivals did I attend, and which were they?")

    assert isinstance(synthesis_operation_for_plan(currency_sum), SumValuesOperation)
    assert isinstance(synthesis_operation_for_plan(currency_difference), DifferenceBetweenOperation)
    assert isinstance(synthesis_operation_for_plan(duration_sum), SumValuesOperation)
    assert isinstance(synthesis_operation_for_plan(date_interval), TemporalIntervalOperation)
    assert isinstance(synthesis_operation_for_plan(count), ListItemsOperation)


def test_build_synthesis_plan_tokenizes_query_once(monkeypatch) -> None:
    """Planning should reuse one query tokenization for tokens and subject terms."""
    calls = 0
    original_source_tokens = synthesis.source_tokens

    def tracking_source_tokens(text: str) -> list[str]:
        nonlocal calls
        calls += 1
        return original_source_tokens(text)

    monkeypatch.setattr(synthesis, "source_tokens", tracking_source_tokens)

    plan = synthesis.build_synthesis_plan(
        "How much total money have I spent on bike-related expenses?"
    )

    assert plan.required_kinds == ("currency",)
    assert calls == 1


def test_currency_label_scans_bounded_prefix_for_pre_amount_labels(monkeypatch) -> None:
    """Currency labels should come from nearby text without regex scanning the full source."""
    scanned_prefix_lengths: list[int] = []
    original_before_amount = synthesis._currency_label_before_amount

    def tracking_before_amount(prefix: str) -> str:
        scanned_prefix_lengths.append(len(prefix))
        return original_before_amount(prefix)

    monkeypatch.setattr(synthesis, "_currency_label_before_amount", tracking_before_amount)
    prefix = ("This older unrelated budget note mentions $5 and planning details. " * 80)
    prefix += "I replaced the bike chain cost me "
    text = prefix + "$45 after tax."

    label = synthesis.currency_label(text, len(prefix), len(prefix) + 3)

    assert label == "the bike chain"
    assert scanned_prefix_lengths
    assert max(scanned_prefix_lengths) <= 240


def test_source_tokens_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Hot-path tokenization should not compile regex strings on every call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("source_tokens should use compiled regex helpers")

    monkeypatch.setattr(synthesis.re, "findall", fail)
    monkeypatch.setattr(synthesis.re, "search", fail)
    monkeypatch.setattr(synthesis.re, "split", fail)

    assert synthesis.source_tokens("source_path=longmemeval/foo-bar.md") == [
        "source_path",
        "source",
        "path",
        "longmemeval/foo-bar.md",
        "longmemeval",
        "foo",
        "bar.md",
    ]


def test_source_tokens_uses_constant_time_separator_check(monkeypatch) -> None:
    """Token splitting should not run a regex search for every token."""
    monkeypatch.setattr(
        builtins,
        "any",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("separator checks should not allocate generator scans")
        ),
    )

    assert not hasattr(synthesis, "_SOURCE_TOKEN_HAS_SEPARATOR_RE")

    assert synthesis.source_tokens("source_path=longmemeval/foo-bar.md") == [
        "source_path",
        "source",
        "path",
        "longmemeval/foo-bar.md",
        "longmemeval",
        "foo",
        "bar.md",
    ]


def test_source_tokens_caches_repeated_text_without_mutation_leak(monkeypatch) -> None:
    """Repeated synthesis tokenization should reuse parsing while returning safe lists."""
    synthesis._source_token_tuple.cache_clear()
    calls = 0
    original_token_re = synthesis._SOURCE_TOKEN_RE

    class TrackingTokenRegex:
        def findall(self, text: str) -> list[str]:
            nonlocal calls
            calls += 1
            return original_token_re.findall(text)

    monkeypatch.setattr(synthesis, "_SOURCE_TOKEN_RE", TrackingTokenRegex())
    text = "source_path=longmemeval/foo-bar.md I bought bike gear."

    first = synthesis.source_tokens(text)
    first.append("mutated")
    second = synthesis.source_tokens(text)

    assert "mutated" not in second
    assert second[:7] == [
        "source_path",
        "source",
        "path",
        "longmemeval/foo-bar.md",
        "longmemeval",
        "foo",
        "bar.md",
    ]
    assert calls == 1


def test_currency_label_before_amount_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Repeated currency-label recovery should avoid dynamic regex compilation."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("currency label recovery should use compiled regex helpers")

    monkeypatch.setattr(synthesis.re, "search", fail)

    assert synthesis._currency_label_before_amount("I recently bought bike lights for ") == "bike lights"


def test_currency_label_before_amount_skips_patterns_without_trigger(monkeypatch) -> None:
    """Irrelevant prefixes should avoid regex pattern scans in the hot path."""

    class FailingPattern:
        def search(self, value: str):  # noqa: ANN001
            del value
            raise AssertionError("irrelevant prefixes should not scan currency-label patterns")

    monkeypatch.setattr(
        synthesis,
        "_CURRENCY_LABEL_BEFORE_AMOUNT_PATTERNS",
        (FailingPattern(),),
    )

    assert synthesis._currency_label_before_amount("I mentioned several older purchases") == ""


def test_build_synthesis_plan_classifies_currency_difference() -> None:
    """Comparison money queries should produce a deterministic difference plan."""
    plan = build_synthesis_plan(
        "How much more did I spend on accommodations in Hawaii compared to Tokyo?"
    )

    assert plan.answer_type == "difference"
    assert plan.operation == "difference_between"
    assert plan.required_kinds == ("currency",)
    assert "comparison" in plan.reasons


def test_build_synthesis_plan_classifies_savings_as_currency_difference() -> None:
    """Savings queries should produce a deterministic currency difference plan."""
    plan = build_synthesis_plan("How much did I save on the designer handbag?")

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


def test_build_synthesis_plan_prefers_count_subject_over_time_modifier() -> None:
    """Incidental time modifiers should not turn count-subject queries into duration sums."""
    plan = build_synthesis_plan("How many pieces of writing had I completed three weeks ago?")

    assert plan.answer_type == "count"
    assert plan.operation == "count_distinct"
    assert plan.required_kinds == ("event",)
    assert plan.reasons == ("count",)


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


def test_count_ledger_counts_writing_pieces_with_time_modifier() -> None:
    """Writing-piece count queries should use completed items, not the modifier duration."""
    query = "How many pieces of writing had I completed three weeks ago?"
    ledger = build_count_ledger(
        query,
        [
            "session_id=essay user: I completed the personal essay three weeks ago.",
            "session_id=story user: I finished a short story three weeks ago.",
            "session_id=duration user: I spent three weeks thinking about a future poem.",
        ],
    )

    result = render_count_result(ledger, query=query, rank=1)

    assert [(row.source_group, row.label) for row in ledger.included(kind="event")] == [
        ("essay", "personal essay"),
        ("story", "short story"),
    ]
    assert "count_answer=2" in result.lines
    assert "duration" in ",".join(result.excluded_source_groups)


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


def test_count_ledger_sums_rollercoaster_ride_occurrences() -> None:
    """Rollercoaster count synthesis should count ride occurrences, not sessions."""
    query = "How many times did I ride rollercoasters from July to October?"
    ledger = build_count_ledger(
        query,
        [
            "session_id=july user: In July I rode Mako, Kraken, and Manta rollercoasters.",
            "session_id=august user: In August I rode Revenge of the Mummy three times.",
            "session_id=september user: In September I rode Space Mountain: Ghost Galaxy three times.",
            "session_id=october user: In October I rode the Xcelerator rollercoaster.",
        ],
    )

    result = render_count_result(ledger, query, rank=1)

    assert "count_answer=10" in result.lines
    assert "count_answer_text=I rode rollercoasters 10 times." in result.lines


def test_count_ledger_sums_aquarium_fish_inventory() -> None:
    """Aquarium inventory synthesis should sum species counts and singular fish."""
    query = "How many fish are there in total in both of my aquariums?"
    ledger = build_count_ledger(
        query,
        [
            (
                "session_id=community user: My new 20-gallon tank currently has "
                "10 neon tetras, 5 golden honey gouramis, and a small pleco catfish."
            ),
            (
                "session_id=betta user: I also upgraded my old 10-gallon tank, "
                "which has my betta fish, Bubbles."
            ),
        ],
    )

    result = render_count_result(ledger, query, rank=1)

    assert "count_answer=17" in result.lines
    assert "count_answer_text=There are 17 fish in my aquariums." in result.lines


def test_numeric_state_ledger_uses_latest_stated_total() -> None:
    """Current count-state queries should prefer cited latest totals over event counts."""
    ledger = build_numeric_state_ledger(
        "How many different species of birds have I seen in my local park?",
        [
            (
                "longmemeval_session_id=birds_1 "
                "user: I've managed to spot 27 different species so far in my local park."
            ),
            (
                "longmemeval_session_id=birds_2 "
                "user: I just saw a Northern Flicker, which brings my total species count to 32."
            ),
        ],
    )

    result = render_numeric_state_result(ledger, rank=1)

    assert [(row.source_group, row.value, row.include_reason) for row in ledger.included(kind="numeric_state")] == [
        ("birds_1", "27", "stated_total"),
        ("birds_2", "32", "stated_total"),
    ]
    assert "numeric_state_answer=32" in result.lines
    assert "numeric_state_operation=latest_total(32)" in result.lines
    assert result.answer_candidate is not None
    assert result.answer_candidate["type"] == "numeric_state"


def test_numeric_state_ledger_applies_increment_after_prior_total() -> None:
    """State updates should carry forward a prior total when a later source adds items."""
    ledger = build_numeric_state_ledger(
        "How many pre-1920 American coins do I have in my collection?",
        [
            (
                "longmemeval_session_id=coins_1 "
                "user: I have a total of 37 pre-1920 American coins in that collection."
            ),
            (
                "longmemeval_session_id=coins_2 "
                "user: I just added a new coin to my collection of pre-1920 American coins - a 1915-S Barber quarter."
            ),
        ],
    )

    result = render_numeric_state_result(ledger, rank=1)

    assert [(row.source_group, row.value, row.include_reason) for row in ledger.included(kind="numeric_state")] == [
        ("coins_1", "37", "stated_total"),
        ("coins_2", "1", "incremental_update"),
    ]
    assert "numeric_state_answer=38" in result.lines
    assert "numeric_state_operation=37+1" in result.lines


def test_numeric_state_ledger_extracts_now_at_current_totals() -> None:
    """Current-state ledgers should understand common milestone phrasing."""
    ledger = build_numeric_state_ledger(
        "How many Instagram followers do I currently have?",
        [
            (
                "longmemeval_session_id=followers_1 "
                "user: I just reached 500 followers last week."
            ),
            (
                "longmemeval_session_id=followers_2 "
                "user: I just checked and I'm now at 600 followers, which is a nice milestone."
            ),
        ],
    )

    result = render_numeric_state_result(ledger, rank=1)

    assert [(row.source_group, row.value, row.include_reason) for row in ledger.included(kind="numeric_state")] == [
        ("followers_1", "500", "stated_total"),
        ("followers_2", "600", "stated_total"),
    ]
    assert "numeric_state_answer=600" in result.lines
    assert "numeric_state_operation=latest_total(600)" in result.lines


def test_numeric_state_result_answers_count_increase_between_totals() -> None:
    """Increase queries should subtract earlier count-state totals from later totals."""
    query = "What was the approximate increase in Instagram followers I experienced in two weeks?"
    ledger = build_numeric_state_ledger(
        query,
        [
            (
                "longmemeval_session_id=followers_1 "
                "user: I started the year with 250 followers on Instagram, by the way."
            ),
            (
                "longmemeval_session_id=followers_2 "
                "user: After two weeks of posting regularly, I had around 350 followers on Instagram."
            ),
        ],
    )

    result = render_numeric_state_result(ledger, query=query, rank=1)

    assert [(row.source_group, row.value, row.include_reason) for row in ledger.included(kind="numeric_state")] == [
        ("followers_1", "250", "stated_total"),
        ("followers_2", "350", "stated_total"),
    ]
    assert "numeric_state_operation=350-250" in result.lines
    assert "numeric_state_difference_answer=100" in result.lines
    assert result.answer_candidate is not None
    assert result.answer_candidate["answer_key"] == "numeric_state_difference_answer"
    assert result.answer_candidate["answer"] == "100"


def test_numeric_state_ledger_extracts_current_team_size_words() -> None:
    """Team-lead state questions should parse word-number team-size totals."""
    ledger = build_numeric_state_ledger(
        "How many engineers do I currently lead?",
        [
            (
                "longmemeval_session_id=team_1 "
                "user: In my new role as Senior Software Engineer, I lead a team of 4 engineers."
            ),
            (
                "longmemeval_session_id=team_2 "
                "user: I now lead a team of five engineers, and they are working well together."
            ),
        ],
    )

    result = render_numeric_state_result(ledger, rank=1)

    assert [(row.source_group, row.value, row.include_reason) for row in ledger.included(kind="numeric_state")] == [
        ("team_1", "4", "stated_total"),
        ("team_2", "5", "stated_total"),
    ]
    assert "numeric_state_answer=5" in result.lines


def test_numeric_state_result_answers_initial_and_current_state() -> None:
    """Dual-state questions should expose both earliest and latest cited totals."""
    query = "How many engineers did I lead when I started, and how many do I lead now?"
    ledger = build_numeric_state_ledger(
        query,
        [
            "longmemeval_session_id=team_1 user: I lead a team of 4 engineers in my new role.",
            "longmemeval_session_id=team_2 user: I now lead a team of five engineers.",
        ],
    )

    result = render_numeric_state_result(ledger, query=query, rank=1)

    assert "numeric_state_initial_answer=4 engineers" in result.lines
    assert "numeric_state_current_answer=5 engineers" in result.lines
    assert "numeric_state_transition_answer=Initially, I led 4 engineers. Now, I lead 5 engineers." in result.lines
    assert result.answer_candidate is not None
    assert result.answer_candidate["answer_key"] == "numeric_state_transition_answer"
    assert result.answer_candidate["answer"] == "Initially, I led 4 engineers. Now, I lead 5 engineers."
    assert result.answer_candidate["support_source_ids"] == ["team_1", "team_2"]


def test_numeric_state_ledger_requires_role_qualifier_slot() -> None:
    """State totals should not answer when a required role qualifier is unsupported."""
    query = "How many engineers do I lead when I started my new role as Software Engineer Manager?"
    ledger = build_numeric_state_ledger(
        query,
        [
            (
                "longmemeval_session_id=team_1 "
                "user: I lead a team of 4 engineers in my new role as Senior Software Engineer."
            ),
            (
                "longmemeval_session_id=team_2 "
                "user: I now lead a team of five engineers in my Senior Software Engineer role."
            ),
        ],
    )

    assert ledger.included(kind="numeric_state") == ()
    assert {row.exclude_reason for row in ledger.excluded(kind="numeric_state")} == {
        "missing_required_state_qualifier"
    }
    result = render_numeric_state_result(ledger, query=query, rank=1)
    assert result.lines == ()
    assert result.answer_candidate is None


def test_numeric_state_ledger_defers_for_plain_event_counts() -> None:
    """Plain event-count questions should stay on the count ledger."""
    ledger = build_numeric_state_ledger(
        "How many weddings have I attended this year?",
        [
            "longmemeval_session_id=wedding_1 user: I attended Rachel and Mike's wedding.",
            "longmemeval_session_id=wedding_2 user: I attended Emily and Sarah's wedding.",
        ],
    )

    assert ledger.included(kind="numeric_state") == ()
    assert render_numeric_state_result(ledger, rank=1).lines == ()


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
    assert (
        "property_outcome_answer=I viewed four properties before making an offer on "
        "the townhouse in Brookside. The reasons I didn't make an offer on them were: "
        "the kitchen of the bungalow needed renovation, "
        "the property in Cedar Creek was out of my budget, "
        "highway noise was a deal-breaker for the 1-bedroom condo, and "
        "my offer on the 2-bedroom condo was rejected due to a higher bid."
    ) in result.lines
    assert result.support_source_groups == ("answer-1", "answer-2", "answer-3", "answer-4")
    assert result.excluded_source_groups == ("target-view", "final")


def test_count_ledger_renders_clean_property_outcome_reasons() -> None:
    """Property outcome answers should separate property labels from rejection reasons."""
    query = "How many properties did I view before making an offer on the townhouse in the Brookside neighborhood?"
    ledger = build_count_ledger(
        query,
        [
            (
                "session_id=bungalow I recently saw a beautiful 3-bedroom bungalow in the "
                "Oakwood neighborhood on January 22nd that I really liked, but the kitchen "
                "needed some serious renovation work."
            ),
            (
                "session_id=budget I've been searching for a home for a while now, and I've "
                "seen some properties that just didn't fit my budget, like that one in "
                "Cedar Creek on February 1st - it was way out of my league."
            ),
            (
                "session_id=noise I viewed a 1-bedroom condo on February 10th, but the noise "
                "from the highway was a deal-breaker."
            ),
            (
                "session_id=rejected I actually fell in love with a 2-bedroom condo on "
                "February 15th, it had amazing modern appliances and a community pool, but "
                "unfortunately, my offer got rejected on the 17th due to a higher bid."
            ),
            (
                "session_id=target I put in an offer on a 3-bedroom townhouse in the "
                "Brookside neighborhood on February 25th."
            ),
        ],
    )

    result = render_count_result(ledger, query, rank=1)

    assert "count_answer=4" in result.lines
    assert (
        "property_outcome_answer=I viewed four properties before making an offer on "
        "the townhouse in the Brookside neighborhood. The reasons I didn't make an offer on them were: "
        "the kitchen of the bungalow needed serious renovation, "
        "the property in Cedar Creek was out of my budget, "
        "the noise from the highway was a deal-breaker for the 1-bedroom condo, and "
        "my offer on the 2-bedroom condo was rejected due to a higher bid."
    ) in result.lines


def test_count_ledger_counts_past_competitive_sports() -> None:
    """Past competitive-sport counts should bind only sports played competitively."""
    ledger = build_count_ledger(
        "How many sports have I played competitively in the past?",
        [
            (
                "session_id=swim user: I used to swim competitively in college, and "
                "I'm looking to get back into lap swimming."
            ),
            (
                "session_id=tennis user: I've been playing soccer and tennis lately. "
                "I used to play tennis competitively in high school."
            ),
            (
                "session_id=soccer user: I've been playing soccer lately to stay active."
            ),
        ],
    )

    result = render_count_result(
        ledger,
        "How many sports have I played competitively in the past?",
        rank=1,
    )

    assert "count_answer=2" in result.lines
    assert "count_answer_text=I played two sports competitively in the past: swimming and tennis." in result.lines
    assert result.support_source_groups == ("swim", "tennis")


def test_count_ledger_counts_distinct_attended_dinner_parties() -> None:
    """Dinner-party counts should extract distinct attended party locations from cited memories."""
    ledger = build_count_ledger(
        "How many dinner parties have I attended in the past month?",
        [
            (
                "session_id=sarah user: I attended a lovely Italian feast at Sarah's "
                "place last week, and it inspired me to try new dishes."
            ),
            (
                "session_id=alex-mike user: I've had experience with dinner parties that "
                "are more low-key, like the ones we had at Alex's place yesterday, where "
                "we had a potluck, and also at Mike's place, where we had a BBQ and "
                "watched a football game together."
            ),
        ],
    )

    result = render_count_result(
        ledger,
        "How many dinner parties have I attended in the past month?",
        rank=1,
    )

    assert "count_answer=3" in result.lines
    assert "count_answer_text=I attended three dinner parties: Sarah's place, Alex's place, and Mike's place." in result.lines
    assert result.support_source_groups == ("sarah", "alex-mike")


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


def test_date_ledger_scores_explicit_dates_from_local_event_spans() -> None:
    """Explicit date rows should carry local event context, not the whole source text."""
    ledger = build_date_ledger(
        "How many days passed between my hiking trip and my pottery class?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/03/20 (Mon) "
                "user: I went to a concert on February 1st. "
                "I went on a hiking trip on March 1st."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/03/20 (Mon) "
                "user: I attended my pottery class on March 15th."
            ),
        ],
    )

    rows = ledger.included(kind="date")
    excluded = ledger.excluded(kind="date")

    assert [(row.value, row.relevance) for row in rows] == [
        ("2023-03-01", 2),
        ("2023-03-15", 2),
    ]
    assert [(row.value, row.relevance, row.exclude_reason) for row in excluded] == [
        ("2023-02-01", 0, "query_focus_mismatch")
    ]
    assert "concert" in excluded[0].context
    assert "hiking trip" not in excluded[0].context


def test_date_interval_prefers_role_covered_anchors_over_incidental_dates() -> None:
    """Between-A-and-B queries should choose anchors covering both named roles."""
    ledger = build_date_ledger(
        "How many days passed between my hiking trip and my pottery class?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/03/20 (Mon) "
                "user: I went to a concert on February 1st. "
                "I went on a hiking trip on March 1st."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/03/20 (Mon) "
                "user: I attended my pottery class on March 15th."
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
    assert "date_interval_days=44" not in result.lines[:6]


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


def test_date_ledger_uses_cited_session_dates_as_event_anchors() -> None:
    """Session metadata can anchor answer events when the cited source text matches the query."""
    ledger = build_date_ledger(
        "How many days passed between my visit to MoMA and the Ancient Civilizations exhibit?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/01/08 (Sun) "
                "user: I just got back from a guided tour at the Museum of Modern Art focused on modern art movements."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/01/15 (Sun) "
                "user: I attended the Ancient Civilizations exhibit at the Metropolitan Museum of Art today."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=1)

    assert [(row.source_group, row.value, row.include_reason) for row in ledger.included(kind="date")] == [
        ("answer-1", "2023-01-08", "session_date_anchor"),
        ("answer-2", "2023-01-15", "session_date_anchor"),
    ]
    assert "date_interval_days=7" in result.lines
    assert "date_interval_source_ids=answer-1,answer-2" in result.lines


def test_date_ledger_allows_rich_how_many_days_passed_between_queries() -> None:
    """Long event names should not make interval queries look like count queries."""
    ledger = build_date_ledger(
        (
            "How many days passed between my visit to the Museum of Modern Art (MoMA) "
            "and the 'Ancient Civilizations' exhibit at the Metropolitan Museum of Art?"
        ),
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/01/08 (Sun) "
                "user: I just got back from a guided tour at the Museum of Modern Art "
                "focused on 20th-century modern art movements."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/01/15 (Sun) "
                "user: I attended the Ancient Civilizations exhibit at the Metropolitan Museum of Art today."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="date")] == [
        ("answer-1", "2023-01-08"),
        ("answer-2", "2023-01-15"),
    ]
    assert "date_interval_days=7" in result.lines


def test_date_ledger_offsets_relative_session_date_anchors() -> None:
    """Relative source phrasing should adjust typed session-date anchors."""
    ledger = build_date_ledger(
        "How many days ago did I attend a baking class when I made my friend's birthday cake?",
        [
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2022/03/21 (Mon) "
                "user: I took an amazing baking class at a local culinary school yesterday."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2022/04/10 (Sun) "
                "user: I just baked a chocolate cake for my friend's birthday party today."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=1)

    assert [(row.source_group, row.value, row.include_reason) for row in ledger.included(kind="date")] == [
        ("answer-1", "2022-03-20", "relative_session_date_anchor"),
        ("answer-2", "2022-04-10", "session_date_anchor"),
    ]
    assert "date_interval_days=21" in result.lines


def test_date_ledger_uses_query_temporal_anchor_for_days_ago_questions() -> None:
    """Days-ago synthesis should compare cited event dates to the query-time anchor."""
    ledger = build_date_ledger(
        "How many days ago did I meet Emma?",
        [
            (
                "query_temporal_anchor=true "
                "longmemeval_session_id=query-temporal-anchor "
                "longmemeval_session_date=2023/04/20 (Thu) "
                "role=query The question was asked today."
            ),
            (
                "longmemeval_session_id=answer-emma "
                "longmemeval_session_date=2023/04/11 (Tue) "
                "role=user I caught up with Emma over lunch today."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=1)

    assert [(row.source_group, row.value, row.include_reason) for row in ledger.included(kind="date")] == [
        ("query-temporal-anchor", "2023-04-20", "query_temporal_anchor"),
        ("answer-emma", "2023-04-11", "session_date_anchor"),
    ]
    assert "date_interval_answer=9 days. 10 days (including the last day) is also acceptable." in result.lines
    assert result.answer_candidate is not None
    assert result.answer_candidate["answer"].startswith("9 days")


def test_date_interval_prefers_cited_event_pair_over_query_anchor_for_before_queries() -> None:
    """Before/after event-pair queries should not bind the query date as an operand."""
    ledger = build_date_ledger(
        "How many days before my best friend's birthday party did I order her gift?",
        [
            (
                "query_temporal_anchor=true "
                "longmemeval_session_id=query-temporal-anchor "
                "longmemeval_session_date=2022/05/15 (Sun) "
                "role=query The question was asked today."
            ),
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2022/05/15 (Sun) "
                "assistant: Recently, I can suggest personalized birthday party gift options "
                "for your best friend."
            ),
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2022/05/15 (Sun) "
                "user: I ordered the personalized photo album on the 15th of April "
                "for my best friend's birthday."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2022/05/15 (Sun) "
                "user: I celebrated my best friend's birthday party on the 22nd of April."
            ),
        ],
    )

    result = render_date_interval_result(ledger, rank=1)

    assert "date_interval_days=7" in result.lines[:8]
    assert "date_interval_source_ids=answer-1,answer-2" in result.lines
    assert all(
        not (row.source_group == "answer-1" and row.value == "2022-05-15" and not row.exclude_reason)
        for row in ledger.rows
    )
    assert "date_interval_days=23" not in result.lines[:8]


def test_temporal_sequence_uses_just_as_session_date_anchor() -> None:
    """Recent first-person event phrasing should anchor sequence order to session dates."""
    ledger = build_temporal_sequence_ledger(
        (
            "Which three events happened in the order from first to last: the day I "
            "helped my friend prepare the nursery, the day I helped my cousin pick "
            "out stuff for her baby shower, and the day I ordered a phone case?"
        ),
        [
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/02/10 (Fri) "
                "user: I just helped my cousin pick out some stuff for her baby shower."
            ),
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/02/05 (Sun) "
                "user: I just helped my friend prepare a nursery."
            ),
            (
                "content=longmemeval_session_id=answer-3 "
                "longmemeval_session_date=2023/02/25 (Sat) "
                "user: I ordered a customized phone case for my friend's birthday today."
            ),
        ],
    )

    result = render_temporal_sequence_result(ledger, rank=1)

    assert [(row.source_group, row.include_reason) for row in ledger.included(kind="temporal_event")] == [
        ("answer-1", "session_date_anchor"),
        ("answer-2", "session_date_anchor"),
        ("answer-3", "session_date_anchor"),
    ]
    assert (
        "temporal_sequence_answer=First, I helped my friend prepare the nursery. "
        "Then, I helped my cousin pick out stuff for her baby shower. "
        "Lastly, I ordered a phone case."
    ) in result.lines


def test_temporal_sequence_extracts_common_action_verbs_from_quoted_events() -> None:
    """Temporal order questions should cover routine action verbs, not only travel verbs."""
    ledger = build_temporal_sequence_ledger(
        (
            "What is the order of the three events: 'I signed up for the rewards "
            "program at ShopRite', 'I used a Buy One Get One Free coupon on Luvs "
            "diapers at Walmart', and 'I redeemed $12 cashback for a $10 Amazon "
            "gift card from Ibotta'?"
        ),
        [
            (
                "content=longmemeval_session_id=answer-3 "
                "longmemeval_session_date=2023/03/20 (Mon) "
                "user: I signed up for their rewards program today while shopping at ShopRite."
            ),
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/03/05 (Sun) "
                "user: I used a Buy One Get One Free coupon on Luvs diapers at Walmart today."
            ),
            (
                "content=longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/03/12 (Sun) "
                "user: I redeemed $12 cashback for a $10 Amazon gift card from Ibotta today."
            ),
        ],
    )

    result = render_temporal_sequence_result(ledger, rank=1)

    assert [row.source_group for row in ledger.included(kind="temporal_event")] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
    assert (
        "temporal_sequence_answer=First, I used a Buy One Get One Free coupon on Luvs diapers at Walmart. "
        "Then, I redeemed $12 cashback for a $10 Amazon gift card from Ibotta. "
        "Lastly, I signed up for the rewards program at ShopRite."
    ) in result.lines


def test_date_ledger_does_not_promote_unrelated_session_dates() -> None:
    """Session metadata should not become date evidence without query-specific event overlap."""
    ledger = build_date_ledger(
        "How many days passed between my visit to MoMA and the Ancient Civilizations exhibit?",
        [
            (
                "content=longmemeval_session_id=distractor-1 "
                "longmemeval_session_date=2023/01/01 (Sun) "
                "user: I asked for general museum gift shop recommendations today."
            ),
            (
                "content=longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/01/08 (Sun) "
                "user: I just got back from a guided tour at the Museum of Modern Art focused on modern art movements."
            ),
        ],
    )

    assert [(row.source_group, row.value) for row in ledger.included(kind="date")] == [
        ("answer-1", "2023-01-08")
    ]


def test_temporal_sequence_orders_relative_events_from_earliest_to_latest() -> None:
    """Sequence synthesis should order cited event rows, not just select the first event."""
    ledger = build_temporal_sequence_ledger(
        "What is the order of the three trips I took in the past three months, from earliest to latest?",
        [
            (
                "content=longmemeval_session_id=answer-trip-3 "
                "user: I just got back from a solo camping trip to Yosemite National Park today."
            ),
            (
                "content=longmemeval_session_id=answer-trip-1 "
                "user: I went on a day hike to Muir Woods about two months ago."
            ),
            (
                "content=longmemeval_session_id=answer-trip-2 "
                "user: I got back from a road trip with friends to Big Sur and Monterey last month."
            ),
        ],
    )

    result = render_temporal_sequence_result(ledger, rank=1)

    assert [(row.source_group, row.label, row.include_reason) for row in ledger.included(kind="temporal_event")] == [
        ("answer-trip-1", "day hike to Muir Woods", "relative_time_anchor"),
        ("answer-trip-2", "road trip with friends to Big Sur and Monterey", "relative_time_anchor"),
        ("answer-trip-3", "solo camping trip to Yosemite National Park", "relative_time_anchor"),
    ]
    assert (
        "temporal_sequence_answer=First, day hike to Muir Woods. "
        "Then, road trip with friends to Big Sur and Monterey. "
        "Lastly, solo camping trip to Yosemite National Park."
    ) in result.lines
    assert result.support_source_groups == ("answer-trip-1", "answer-trip-2", "answer-trip-3")
    assert result.answer_candidate is not None
    assert result.answer_candidate["type"] == "temporal_sequence"


def test_temporal_sequence_uses_provenance_order_for_equal_vague_dates() -> None:
    """Equal/vague temporal anchors should use source provenance order as a deterministic tie-breaker."""
    ledger = build_temporal_sequence_ledger(
        (
            "Which three events happened in the order from first to last: helping prepare the nursery, "
            "helping pick out baby shower stuff, and ordering a customized phone case?"
        ),
        [
            (
                "content=longmemeval_session_id=answer-family_2 "
                "longmemeval_session_date=2023/04/03 (Mon) "
                "user: I just helped my cousin pick out some stuff for her baby shower today."
            ),
            (
                "content=longmemeval_session_id=answer-family_1 "
                "longmemeval_session_date=2023/04/03 (Mon) "
                "user: I just helped my friend prepare the nursery today."
            ),
            (
                "content=longmemeval_session_id=answer-family_3 "
                "longmemeval_session_date=2023/04/03 (Mon) "
                "user: I ordered a customized phone case for my friend's birthday today."
            ),
        ],
    )

    result = render_temporal_sequence_result(ledger, rank=1)

    assert [row.source_group for row in ledger.included(kind="temporal_event")] == [
        "answer-family_1",
        "answer-family_2",
        "answer-family_3",
    ]
    assert (
        "temporal_sequence_answer=First, I helped my friend prepare the nursery. "
        "Then, I helped my cousin pick out some stuff for her baby shower. "
        "Lastly, I ordered a customized phone case for my friend's birthday."
    ) in result.lines


def test_temporal_sequence_defers_without_enough_event_slots() -> None:
    """Sequence synthesis should not invent an ordered list from one cited event."""
    ledger = build_temporal_sequence_ledger(
        "What is the order of the three sports events I watched in January?",
        [
            (
                "content=longmemeval_session_id=answer-sports_1 "
                "longmemeval_session_date=2023/01/04 (Wed) "
                "user: I watched an NBA game at Staples Center today."
            ),
        ],
    )

    result = render_temporal_sequence_result(ledger, rank=1)

    assert len(ledger.included(kind="temporal_event")) == 1
    assert result.lines == ()


def test_temporal_sequence_answer_preserves_all_included_list_items() -> None:
    """Temporal list answers should not silently drop cited ledger items above five."""
    labels = (
        "Science Museum",
        "Museum of Contemporary Art",
        "Metropolitan Museum of Art",
        "Museum of History",
        "Modern Art Museum",
        "Natural History Museum",
    )
    plan = SynthesisPlan(
        answer_type="list",
        operation="temporal_sequence",
        subject_terms=("museum",),
        required_kinds=("temporal_event",),
        required_source_groups=1,
        reasons=("temporal_sequence", "list_answer"),
    )
    rows = tuple(
        EvidenceLedgerRow(
            fact_id=f"temporal_sequence:{index}",
            source_group=f"answer-{index + 1}",
            citation=f"eventloom://agent/events/{index + 1}#hash{index + 1}",
            kind="temporal_event",
            value=str(index + 1),
            unit="order",
            label=label,
            raw_span=f"I visited the {label}.",
            context=f"I visited the {label}.",
            normalized_identity=f"temporal_event={label.casefold()}",
            relevance=3,
            include_reason="explicit_temporal_order",
            confidence=0.9,
        )
        for index, label in enumerate(labels)
    )

    result = render_temporal_sequence_result(EvidenceLedger(plan=plan, rows=rows), rank=1)

    assert result.answer_candidate is not None
    assert "Natural History Museum" in result.answer_candidate["answer"]
    assert result.support_source_groups == (
        "answer-1",
        "answer-2",
        "answer-3",
        "answer-4",
        "answer-5",
        "answer-6",
    )


def test_temporal_sequence_uses_venue_entity_labels_for_museum_order_queries() -> None:
    """Ordered venue-list synthesis should expose the venue entity, not adjacent event details."""
    ledger = build_temporal_sequence_ledger(
        "What is the order of the six museums I visited from earliest to latest?",
        [
            (
                "content=longmemeval_session_id=answer-museum-2 "
                "longmemeval_session_date=2023/01/22 (Sun) "
                "user: Speaking of feminist art, I just came back from a lecture series "
                "at the Museum of Contemporary Art, where Dr. Maria Rodriguez spoke "
                "about the role of feminist art in the 1970s."
            ),
            (
                "content=longmemeval_session_id=answer-museum-1 "
                "longmemeval_session_date=2023/01/15 (Sun) "
                "user: I visited the Science Museum's Space Exploration exhibition today."
            ),
        ],
    )

    result = render_temporal_sequence_result(ledger, rank=1)

    assert [row.label for row in ledger.included(kind="temporal_event")] == [
        "Science Museum",
        "Museum of Contemporary Art",
    ]
    assert (
        "temporal_sequence_answer=First, Science Museum. Then, Museum of Contemporary Art."
    ) in result.lines


def test_temporal_sequence_ignores_query_echo_and_normalizes_museum_possessives() -> None:
    """Checkout/query metadata should not become evidence for ordered venue lists."""
    ledger = build_temporal_sequence_ledger(
        "What is the order of the museums I visited from earliest to latest?",
        [
            (
                "query=What is the order of the museums I visited from earliest to latest? "
                "content=longmemeval_session_id=answer-met "
                "longmemeval_session_date=2023/02/10 (Fri) "
                "user: I saw it in person today at the Metropolitan Museum of Art's "
                "\"Ancient Egyptian Artifacts\" exhibition."
            ),
            (
                "content=longmemeval_session_id=answer-history "
                "longmemeval_session_date=2023/02/15 (Wed) "
                "user: I participated in a behind-the-scenes tour of the Museum of History's "
                "conservation lab today."
            ),
        ],
    )

    result = render_temporal_sequence_result(ledger, rank=1)

    assert [row.label for row in ledger.included(kind="temporal_event")] == [
        "Metropolitan Museum of Art",
        "Museum of History",
    ]
    assert "from earliest to latest" not in result.answer_candidate["answer"]  # type: ignore[index]


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


def test_currency_ledger_excludes_price_filter_ranges_for_spend_totals() -> None:
    """Spent-total synthesis should ignore assistant search filters and price-range examples."""
    ledger = build_currency_ledger(
        "How much total money did I spend on attending workshops?",
        [
            "session_id=paid-1 user: I paid $500 to attend a digital marketing workshop.",
            "session_id=paid-2 user: I paid $200 to attend a writing workshop.",
            (
                "session_id=assistant-filter **Google Search:** Use specific search terms like "
                "\"2-day writing workshops near me under $500\" or filter by cost ranges "
                "such as $100-$500 and $500-$1000."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("paid-1", "500.0"),
        ("paid-2", "200.0"),
    ]
    assert {row.value for row in ledger.excluded(kind="currency")} >= {"100.0", "500.0", "1000.0"}
    assert "currency_total_answer=$700" in result.lines
    assert "currency_total_answer=$1,700" not in result.lines


def test_currency_ledger_requires_itemized_targets_for_conjunctive_cost_queries() -> None:
    """Itemized cost queries should not include unrelated amounts from the same owner/session."""
    ledger = build_currency_ledger(
        "What is the total cost of Lola's vet visit and flea medication?",
        [
            (
                "session_id=vet user: I just took Lola to the vet last week and "
                "got a discounted consultation fee of $50 as a first-time customer."
            ),
            (
                "session_id=supplies user: I also got her flea and tick prevention "
                "medication, it was $25 for a 3-month supply. I got Lola a bag of "
                "cat food from Petco, it was $35. Her carrier was $80 when I bought it last year."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("vet", "50.0"),
        ("supplies", "25.0"),
    ]
    assert {row.value for row in ledger.excluded(kind="currency")} >= {"35.0", "80.0"}
    assert "currency_total_answer=$75" in result.lines
    assert "currency_total_answer=$190" not in result.lines


def test_currency_ledger_filters_itemized_targets_within_one_source_group() -> None:
    """Itemized spend queries should use requested item slots even when evidence shares one source."""
    ledger = build_currency_ledger(
        "How much did I spend on car wash and parking ticket?",
        [
            (
                "session_id=car-budget user: I'm trying to keep track of my car expenses. "
                "My annual insurance premium is $3,500, and my mechanic quoted $200 for a tune-up. "
                "This week I paid $50 for a car wash and $15 for a parking ticket."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("car-budget", "50.0"),
        ("car-budget", "15.0"),
    ]
    assert {row.value for row in ledger.excluded(kind="currency")} >= {"3500.0", "200.0"}
    assert "currency_total_answer=$65" in result.lines
    assert "currency_total_answer=$3,765" not in result.lines


def test_currency_ledger_requires_head_terms_for_multiword_item_slots() -> None:
    """A broad category token like car should not satisfy a specific car-wash slot."""
    ledger = build_currency_ledger(
        "How much did I spend on car wash and parking ticket?",
        [
            "session_id=budget user: I am budgeting my car expenses. My annual car insurance premium is $3,500.",
            "session_id=wash user: I had a car wash on February 3rd that cost $15.",
            "session_id=ticket user: I also got a parking ticket near my work for $50.",
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("wash", "15.0"),
        ("ticket", "50.0"),
    ]
    assert "budget" in ",".join(result.excluded_source_groups)
    assert "currency_total_answer=$65" in result.lines
    assert "currency_total_answer=$3,565" not in result.lines


def test_currency_ledger_requires_substantive_item_slot_matches() -> None:
    """Generic adjectives like new should not make unrelated items satisfy itemized slots."""
    ledger = build_currency_ledger(
        "What is the total cost of the new food bowl, measuring cup, dental chews, and flea and tick collar I got for Max?",
        [
            (
                "session_id=chews user: the dental chews - I started using a new one "
                "to help with his teeth, and the chews are $10 a pack."
            ),
            (
                "session_id=bed user: I also got a new dog bed for Max recently, "
                "it was around $40, but that was a one-time expense."
            ),
            (
                "session_id=bowl-cup user: I just got him a new stainless steel food bowl "
                "from Amazon for $15, and a measuring cup from the pet store down the street for $5."
            ),
            (
                "session_id=collar user: I forgot to mention that I also got a flea and tick "
                "collar for Max recently, which was $20."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("chews", "10.0"),
        ("bowl-cup", "15.0"),
        ("bowl-cup", "5.0"),
        ("collar", "20.0"),
    ]
    assert "bed" in ",".join(result.excluded_source_groups)
    assert "currency_total_answer=$50" in result.lines
    assert "currency_total_answer=$90" not in result.lines


def test_currency_ledger_excludes_generic_category_amounts_for_recipient_gift_slots() -> None:
    """Recipient-scoped gift totals should not include broad category spend amounts."""
    ledger = build_currency_ledger(
        "What is the total amount I spent on gifts for my coworker and brother?",
        [
            (
                "session_id=gifts user: I spent $500 on holiday gifts. "
                "I bought a coffee mug for my coworker for $60. "
                "I bought headphones for my brother for $140."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("gifts", "60.0"),
        ("gifts", "140.0"),
    ]
    assert {row.value for row in ledger.excluded(kind="currency")} >= {"500.0"}
    assert "currency_total_answer=$200" in result.lines
    assert "currency_total_answer=$700" not in result.lines


def test_currency_ledger_uses_local_anaphora_for_recipient_gift_slots() -> None:
    """Recipient-scoped gift totals should include local pronoun evidence and exclude broad category totals."""
    ledger = build_currency_ledger(
        "What is the total amount I spent on gifts for my coworker and brother?",
        [
            (
                "session_id=coworker user: I was at a baby shower recently and made sure "
                "to get a gift receipt, just in case my coworker wanted to exchange anything. "
                "I purchased her a set of adorable baby clothes and toys from Buy Buy Baby, totaling $100."
            ),
            (
                "session_id=brother user: I know I spent a total of $500 on gifts recently, "
                "but I'm having trouble breaking it down. By the way, I did get my brother "
                "a really nice graduation gift in May - a $100 gift card to his favorite electronics store."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("coworker", "100.0"),
        ("brother", "100.0"),
    ]
    assert {row.value for row in ledger.excluded(kind="currency")} >= {"500.0"}
    assert "currency_total_answer=$200" in result.lines
    assert "currency_total_answer=$600" not in result.lines


def test_currency_ledger_promotes_unit_price_for_each_item_queries() -> None:
    """Each-item price queries should answer with the unit amount, not aggregate spend."""
    ledger = build_currency_ledger(
        "How much did I spend on each coffee mug for my coworkers?",
        [
            (
                "session_id=mugs user: I spent $60 on coffee mugs for my coworkers, "
                "buying 5 mugs at $12 each. I also bought snacks for $20."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("mugs", "12.0"),
    ]
    assert {row.value for row in ledger.excluded(kind="currency")} >= {"60.0", "20.0"}
    assert "currency_total_answer=$12" in result.lines
    assert "currency_total_answer=$72" not in result.lines


def test_currency_ledger_sums_floor_values_for_minimum_sale_queries() -> None:
    """Minimum sale-value queries should treat worth and at-least value as realized floor operands."""
    ledger = build_currency_ledger(
        "What is the minimum amount I could get if I sold the vintage diamond necklace and the antique vanity?",
        [
            (
                "session_id=necklace user: I'm thinking of selling my vintage diamond necklace, "
                "which is worth $5,000."
            ),
            (
                "session_id=vanity user: I bought the antique vanity for $150 and put in some work "
                "to restore it, so I'm confident it's worth at least that amount now."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("necklace", "5000.0"),
        ("vanity", "150.0"),
    ]
    assert "currency_total_answer=$5,150" in result.lines


def test_currency_ledger_promotes_difference_for_savings_queries() -> None:
    """Savings queries should answer with reference price minus paid price."""
    ledger = build_currency_ledger(
        "How much did I save on the designer handbag?",
        [
            "session_id=paid user: I bought a designer handbag at TK Maxx for $200.",
            "session_id=reference user: The same designer handbag normally retails for $500.",
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_values=$500,$200" in result.lines
    assert "currency_difference_answer=$300" in result.lines
    assert result.answer_candidate is not None
    assert result.answer_candidate["answer_key"] == "currency_difference_answer"
    assert result.answer_candidate["answer"] == "$300"


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


def test_currency_ledger_filters_rewards_for_earned_market_totals() -> None:
    """Earned-money queries should total sale proceeds, not spend rewards or thresholds."""
    ledger = build_currency_ledger(
        "How much money did I earn in total from markets?",
        [
            "session_id=market-1 user: I sold pottery at the winter market and earned $225.",
            "session_id=market-2 user: I made $120 selling prints at a neighborhood market.",
            "session_id=market-3 user: I sold jam at the spring market for $150.",
            (
                "session_id=distractor-1 user: The market loyalty card gives a $50 reward "
                "after customers spend $120."
            ),
            (
                "session_id=distractor-2 user: I priced the candles at $7.50 each "
                "for the next craft market."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert "currency_values=$225,$150,$120" in result.lines
    assert "currency_total_answer=$495" in result.lines
    assert "distractor-1" in ",".join(result.excluded_source_groups)
    assert "distractor-2" in ",".join(result.excluded_source_groups)


def test_currency_ledger_multiplies_realized_unit_price_sales() -> None:
    """Earned-money totals should use sale quantity times unit price when the sale happened."""
    ledger = build_currency_ledger(
        "What is the total amount of money I earned from selling my products at the markets?",
        [
            "session_id=market-1 user: I sold 15 jars of homemade jam at the market for $225 total.",
            "session_id=market-2 user: I sold 12 bunches of herbs at the farmers' market and made $120.",
            (
                "session_id=market-3 user: I just sold 20 potted herb plants at the "
                "Summer Solstice Market for $7.50 each, and it was a great day."
            ),
        ],
    )

    result = render_currency_result(ledger, rank=1)

    assert [(row.source_group, row.value) for row in ledger.included(kind="currency")] == [
        ("market-1", "225.0"),
        ("market-2", "120.0"),
        ("market-3", "150.0"),
    ]
    assert "currency_values=$225,$150,$120" in result.lines
    assert "currency_total_answer=$495" in result.lines


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


def test_duration_ledger_parses_fractional_word_durations() -> None:
    """Prose fractions like two and a half weeks should be typed duration values."""
    ledger = build_duration_ledger(
        "How long did I take to finish 'The Seven Husbands of Evelyn Hugo' and 'The Nightingale' combined?",
        [
            (
                "session_id=evelyn user: I just finished \"The Seven Husbands of Evelyn Hugo\", "
                "which took me two and a half weeks to finish."
            ),
            (
                "session_id=nightingale user: I recently finished \"The Nightingale\" by Kristin Hannah, "
                "which took me three weeks to finish."
            ),
            (
                "session_id=distractor user: I've been getting into audiobooks lately and have "
                "managed to finish three in the last six weeks, which is great for me."
            ),
        ],
    )

    result = render_duration_result(ledger, rank=1)

    assert [(row.source_group, row.label, row.value) for row in ledger.included(kind="duration")] == [
        ("evelyn", "2.5 weeks", "25200.0"),
        ("nightingale", "3 weeks", "30240.0"),
    ]
    assert "duration_values=2.5 weeks,3 weeks" in result.lines
    assert "duration_total_answer=5.5 weeks" in result.lines
    assert "distractor" in ",".join(result.excluded_source_groups)


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


def test_duration_ledger_filters_itinerary_lengths_for_actual_travel_totals() -> None:
    """Travel day totals should use actual trip durations, not advice or planning spans."""
    ledger = build_duration_ledger(
        "How many days did I spend traveling in Hawaii and New York City in total?",
        [
            "session_id=hawaii user: I got back from a 10-day island-hopping trip to Hawaii.",
            "session_id=nyc user: I took a solo trip to New York City for five days.",
            "session_id=advice user: I asked for a 4-day itinerary for New York City.",
            "session_id=planning user: I was planning a 3-day Hawaii weekend option.",
        ],
    )

    result = render_duration_result(ledger, rank=1)

    assert "duration_values=10 days,5 days" in result.lines
    assert "duration_total_answer=15 days" in result.lines
    assert "advice" in ",".join(result.excluded_source_groups)
    assert "planning" in ",".join(result.excluded_source_groups)


def test_duration_ledger_uses_session_trip_context_for_local_planning_wording() -> None:
    """Trip durations can be expressed later as planning inflexibility inside the same cited session."""
    ledger = build_duration_ledger(
        "How many days did I spend in total traveling in Hawaii and in New York City?",
        [
            (
                "session_id=nyc user: I recently got back from a solo trip to New York City "
                "for five days and I was able to save a lot by staying at a hostel."
            ),
            (
                "session_id=hawaii user: By the way, I just got back from an amazing "
                "island-hopping trip to Hawaii with my family. Later I said that with my family, "
                "we had to plan everything out for the 10-day so far in advance."
            ),
        ],
    )

    result = render_duration_result(ledger, rank=1)

    assert [(row.source_group, row.label) for row in ledger.included(kind="duration")] == [
        ("nyc", "5 days"),
        ("hawaii", "10 days"),
    ]
    assert "duration_total_answer=15 days" in result.lines


def test_duration_ledger_binds_trip_duration_across_same_source_group() -> None:
    """Completed-trip destination evidence can bind a duration mentioned in another turn from the same session."""
    ledger = build_duration_ledger(
        "How many days did I spend in total traveling in Hawaii and in New York City?",
        [
            (
                "session_id=nyc user: I recently got back from a solo trip to New York City "
                "for five days and I was able to save a lot by staying at a hostel."
            ),
            (
                "session_id=hawaii user: By the way, I just got back from an amazing "
                "island-hopping trip to Hawaii with my family."
            ),
            (
                "session_id=hawaii user: With my family, we had to plan everything out "
                "for the 10-day so far in advance, and it was hard to make changes on the fly."
            ),
        ],
    )

    result = render_duration_result(ledger, rank=1)

    assert [(row.source_group, row.label) for row in ledger.included(kind="duration")] == [
        ("nyc", "5 days"),
        ("hawaii", "10 days"),
    ]
    assert "duration_total_answer=15 days" in result.lines


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
