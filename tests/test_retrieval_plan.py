"""Tests for retrieval planning and source-lane synthesis helpers."""

from __future__ import annotations

import builtins
import json

from zaxy import evidence_candidates, retrieval_plan, synthesis
from zaxy.evidence_candidates import EvidenceProjection
from zaxy.synthesis import EvidenceLedger, SynthesisPlan, build_currency_ledger


def test_source_synthesis_reuses_candidate_evidence_scores(monkeypatch) -> None:
    """Source-lane ranking should score each candidate once before synthesis."""
    calls: list[int] = []

    def fake_projection(query: str, contexts: list[str]) -> EvidenceProjection:
        del query
        calls.append(len(contexts))
        groups = tuple(
            retrieval_plan.source_context_group(context)
            for context in contexts[:2]
        )
        return EvidenceProjection(
            lines=("candidate_rank=1 candidate_type=currency", "currency_total_answer=$1"),
            source_groups=groups,
        )

    monkeypatch.setattr(
        retrieval_plan,
        "aggregate_candidate_projection",
        fake_projection,
    )
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", lambda query, context: 1)
    contexts = [
        (
            f"source_path=longmemeval/events/{index} "
            f"longmemeval_session_id=answer-{index} "
            f"I bought bike gear for ${index + 1}."
        )
        for index in range(20)
    ]

    bundle = retrieval_plan.source_synthesis_bundle(
        query="How much total money have I spent on bike-related expenses?",
        source_results=contexts,
        limit=5,
    )

    assert bundle is not None
    assert len(calls) <= len(contexts) + 1


def test_source_synthesis_bundle_result_preserves_string_api_and_typed_packet(monkeypatch) -> None:
    """Typed bundle results should preserve exact legacy content while carrying packet data."""
    typed_candidate = {
        "rank": 1,
        "type": "currency",
        "confidence": 0.91,
        "answer_key": "currency_total_answer",
        "answer": "$145",
        "support_source_ids": ["answer-1", "answer-2"],
        "excluded_source_ids": [],
    }
    typed_row = {
        "fact_id": "typed:currency:1",
        "source_group": "answer-1",
        "citation": "eventloom://agent/events/1#aaaaaaaaaaaa",
        "kind": "currency",
        "value": "120",
        "include_reason": "currency_amount",
    }

    def fake_projection(query: str, contexts: list[str]) -> EvidenceProjection:
        del query, contexts
        return EvidenceProjection(
            lines=(
                "candidate_rank=1 candidate_type=currency candidate_confidence=0.10",
                "currency_total_answer=$1",
            ),
            source_groups=("answer-1", "answer-2"),
            ledger_rows=(typed_row,),
            answer_candidates=(typed_candidate,),
        )

    monkeypatch.setattr(retrieval_plan, "aggregate_candidate_projection", fake_projection)
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", lambda query, context: 1)
    source_results = [
        "longmemeval_session_id=answer-1 I bought a bike helmet for $120.",
        "longmemeval_session_id=answer-2 I bought a bike chain for $25.",
    ]

    result = retrieval_plan.source_synthesis_bundle_result(
        query="How much total money have I spent on bike-related expenses?",
        source_results=source_results,
        limit=5,
    )
    legacy = retrieval_plan.source_synthesis_bundle(
        query="How much total money have I spent on bike-related expenses?",
        source_results=source_results,
        limit=5,
    )

    assert result is not None
    assert result.content == legacy
    assert result.packet["schema_version"] == "synthesis_packet_v1"
    assert result.packet["content"] == result.content
    assert result.packet["answer_candidates"] == [typed_candidate]
    assert result.packet["ledger_rows"][0] == typed_row


def test_source_synthesis_bundle_result_includes_operation_result_metadata() -> None:
    """Generated typed bundle packets should include operation/result metadata."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How much total money have I spent on bike-related expenses?",
        source_results=[
            "longmemeval_session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "longmemeval_session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "longmemeval_session_id=answer-3 I got a new set of bike lights installed, which were $40.",
        ],
        limit=5,
    )

    assert result is not None
    assert result.packet["operations"][0]["name"] == "sum_values"
    assert result.packet["operations"][0]["kind"] == "currency"
    assert result.packet["result"] == {
        "answer_key": "currency_total_answer",
        "answer": "$185",
        "confidence": 0.81,
        "support_source_ids": ["answer-1", "answer-2", "answer-3"],
        "excluded_source_ids": [],
    }


def test_aggregate_candidate_projection_exposes_typed_answer_candidates() -> None:
    """Aggregate synthesis should expose operation-produced candidates without text reparsing."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How much total money have I spent on bike-related expenses?",
        [
            "longmemeval_session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "longmemeval_session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "longmemeval_session_id=answer-3 I got a new set of bike lights installed, which were $40.",
        ],
    )

    assert projection.answer_candidates == (
        {
            "rank": 1,
            "type": "currency",
            "confidence": 0.81,
            "answer_key": "currency_total_answer",
            "answer": "$185",
            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
            "excluded_source_ids": [],
        },
    )


def test_source_synthesis_bundle_emits_auditable_ledger_rows() -> None:
    """Generated synthesis bundles should carry ledger include/exclude decisions."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How much total money have I spent on bike-related expenses?",
        source_results=[
            "longmemeval_session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "longmemeval_session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "longmemeval_session_id=answer-3 I got a new set of bike lights installed, which were $40.",
            "longmemeval_session_id=answer-4 I recently got a new set of bike lights installed, which were $40.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]

    assert [row["source_group"] for row in rows if not row.get("exclude_reason")] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
    assert [(row["source_group"], row["exclude_reason"]) for row in rows if row.get("exclude_reason")] == [
        ("answer-4", "duplicate_identity")
    ]


def test_age_average_bundle_uses_typed_aggregate_projection(monkeypatch) -> None:
    """Age-average output should come from typed synthesis operations, not ad hoc line rendering."""
    monkeypatch.setattr(retrieval_plan, "_age_average_synthesis_lines", lambda query, contexts: [])
    bundle = retrieval_plan.source_synthesis_bundle(
        query="What is the average age of me, my parents, and my grandparents?",
        source_results=[
            "longmemeval_session_id=answer_1 I just turned 32 on February 12th.",
            "longmemeval_session_id=answer_2 my parents are getting older too - my mom is 55 and my dad is 58.",
            "longmemeval_session_id=answer_3 My grandma is 75 and my grandpa is 78.",
        ],
        limit=5,
    )

    assert bundle is not None
    assert "candidate_type=number" in bundle
    assert "age_values=32,55,58,75,78" in bundle
    assert "age_average=59.6" in bundle
    assert bundle.count("age_average=59.6") == 1
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    age_rows = [row for row in rows if row.get("include_reason") == "age_average_input"]
    assert [row["value"] for row in age_rows] == ["32", "55", "58", "75", "78"]
    assert {"fact_id", "citation", "kind", "value", "include_reason"} <= set(rows[0])


def test_elapsed_duration_at_event_bundle_emits_ledger_rows() -> None:
    """Elapsed-duration arithmetic should preserve both input rows in the ledger."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How long had I been taking guitar lessons when I bought the new guitar amp?",
        source_results=[
            (
                "longmemeval_session_id=answer_436d4309_1 "
                "I've been taking weekly guitar lessons with Alex for six weeks now."
            ),
            (
                "longmemeval_session_id=answer_436d4309_2 "
                "I just got a new amp two weeks ago and want to get the most out of it."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]

    assert "elapsed_at_event_answer=Four weeks" in bundle
    assert [(row["source_group"], row["kind"], row["value"], row["unit"]) for row in rows] == [
        ("answer_436d4309_1", "duration", "6", "weeks"),
        ("answer_436d4309_2", "duration", "2", "weeks_ago"),
    ]
    assert {row["include_reason"] for row in rows} == {
        "current_activity_duration",
        "event_age_duration",
    }


def test_social_media_break_bundle_emits_break_specific_ledger_rows() -> None:
    """Social-media break totals should be backed by break-specific ledger rows."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many days did I take social media breaks in total?",
        source_results=[
            (
                "longmemeval_session_id=answer_a4204937_1 "
                "Choose a daily time limit like 15 minutes or 1 hour for social media."
            ),
            (
                "longmemeval_session_id=answer_a4204937_1 "
                "I've been making an effort to cut down on social media lately - "
                "I even took a week-long break from it in mid-January."
            ),
            (
                "longmemeval_session_id=answer_a4204937_2 "
                "I've been making an effort to cut down on social media lately - "
                "I actually just got back from a 10-day break in mid-February."
            ),
            (
                "longmemeval_session_id=distractor "
                "Set a daily time limit of 15 minutes for Instagram Monday to Friday."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    break_rows = [row for row in rows if row.get("include_reason") == "social_media_break_duration"]

    assert "social_media_break_total=17 days" in bundle
    assert [(row["source_group"], row["value"], row["unit"]) for row in break_rows] == [
        ("answer_a4204937_1", "7", "days"),
        ("answer_a4204937_2", "10", "days"),
    ]
    assert any(
        row["source_group"] == "distractor" and row.get("exclude_reason") == "not_personal_memory"
        for row in rows
    )


def test_road_trip_drive_bundle_emits_drive_specific_ledger_rows() -> None:
    """Road-trip drive totals should be backed by destination-drive ledger rows."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many hours in total did I spend driving to my three road trip destinations combined?",
        source_results=[
            (
                "longmemeval_session_id=answer_526354c8_1 "
                "my recent trip to Outer Banks in North Carolina - "
                "it only took me four hours to drive there from my place."
            ),
            (
                "longmemeval_session_id=answer_526354c8_2 "
                "when I drove for six hours to Washington D.C. recently"
            ),
            (
                "longmemeval_session_id=answer_526354c8_3 "
                "my recent trip to the mountains in Tennessee - "
                "I drove for five hours to get there and it was totally worth it."
            ),
            (
                "longmemeval_session_id=distractor "
                "From the Outer Banks, it is about a 2-hour drive to Topsail Island, "
                "and then another 4-5 hours to Tybee Island from there."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    drive_rows = [row for row in rows if row.get("include_reason") == "road_trip_destination_drive_duration"]

    assert "road_trip_drive_total=15 hours" in bundle
    assert "road_trip_drive_total_round_trip=30 hours" in bundle
    assert [(row["source_group"], row["value"], row["unit"]) for row in drive_rows] == [
        ("answer_526354c8_1", "4", "hours"),
        ("answer_526354c8_3", "5", "hours"),
        ("answer_526354c8_2", "6", "hours"),
    ]
    assert any(
        row["source_group"] == "distractor" and row.get("exclude_reason") == "not_personal_memory"
        for row in rows
    )


def test_currency_synthesis_does_not_emit_unledgered_duration_fallback() -> None:
    """Currency-only synthesis should not leak unrelated duration totals."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How much more did I spend on accommodations per night in Hawaii compared to Tokyo?",
        source_results=[
            "longmemeval_session_id=answer-1 I spent $300 per night for the Hawaii hotel.",
            "longmemeval_session_id=answer-2 I spent $30 per night for the Tokyo capsule hotel.",
            "longmemeval_session_id=distractor I jogged for 30 minutes before checkout.",
        ],
        limit=5,
    )

    assert bundle is not None
    assert "currency_difference_answer=$270" in bundle
    assert "minute_total_hours=0.5 hours" not in bundle


def test_age_at_event_bundle_emits_operation_ledger_rows() -> None:
    """Age-at-event subtraction should preserve current age and elapsed-year inputs."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How old was I when I moved to the United States?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer_1 "
                "I'm 32-year-old male and have been updating my immigration paperwork."
            ),
            (
                "citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_id=answer_2 "
                "I've been living in the United States for the past five years on a work visa."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    age_rows = [row for row in rows if row.get("fact_id", "").startswith("age_at_event:")]

    assert "age_at_event_answer=27" in bundle
    assert [(row["source_group"], row["value"], row["unit"], row["include_reason"]) for row in age_rows] == [
        ("answer_1", "32", "years", "current_age"),
        ("answer_2", "5", "elapsed_years", "elapsed_since_event"),
    ]


def test_career_prior_duration_bundle_emits_operation_ledger_rows() -> None:
    """Career-prior subtraction should preserve total and current-role inputs."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How long have I been working before I started my current job at NovaTech?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer_1 "
                "I've been working professionally for 9 years and I'm currently using a notebook."
            ),
            (
                "citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_id=answer_2 "
                "I've been working at NovaTech for about 4 years and 3 months now."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    career_rows = [row for row in rows if row.get("fact_id", "").startswith("career_prior_duration:")]

    assert "career_prior_duration_answer=4 years and 9 months" in bundle
    assert [(row["source_group"], row["value"], row["unit"], row["include_reason"]) for row in career_rows] == [
        ("answer_1", "108", "months", "total_career_duration"),
        ("answer_2", "51", "months", "current_role_duration"),
    ]


def test_age_average_bundle_emits_age_ledger_rows() -> None:
    """Age-average fields should preserve each age input with source groups."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="What is the average age of me, my parents, and my grandparents?",
        source_results=[
            "longmemeval_session_id=answer_2504635e_1 I just turned 32 on February 12th.",
            "longmemeval_session_id=answer_2504635e_2 my parents are getting older too - my mom is 55 and my dad is 58.",
            "longmemeval_session_id=answer_2504635e_3 My grandma is 75 and my grandpa is 78.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    age_rows = [row for row in rows if row.get("include_reason") == "age_average_input"]

    assert "age_average=59.6" in bundle
    assert [(row["source_group"], row["value"], row["unit"]) for row in age_rows] == [
        ("answer_2504635e_1", "32", "years"),
        ("answer_2504635e_2", "55", "years"),
        ("answer_2504635e_2", "58", "years"),
        ("answer_2504635e_3", "75", "years"),
        ("answer_2504635e_3", "78", "years"),
    ]


def test_relative_week_interval_bundle_emits_anchor_ledger_rows() -> None:
    """Week-interval fields should preserve both relative-time anchors."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How long had I been a member of Book Lovers Unite when I attended the meetup?",
        source_results=[
            "longmemeval_session_id=joined I joined Book Lovers Unite three weeks ago.",
            "longmemeval_session_id=meetup I attended a meetup organized by Book Lovers Unite last week.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    interval_rows = [row for row in rows if row.get("include_reason") == "relative_week_anchor"]

    assert "week_interval_answer=Two weeks" in bundle
    assert [(row["source_group"], row["value"], row["unit"]) for row in interval_rows] == [
        ("joined", "3", "weeks_ago"),
        ("meetup", "1", "weeks_ago"),
    ]


def test_mixed_relative_interval_bundle_emits_month_and_week_ledger_rows() -> None:
    """Mixed month/week interval fields should preserve normalized anchors."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How long had I been using the new area rug when I rearranged my living room furniture?",
        source_results=[
            "longmemeval_session_id=answer_1 I recently got a new area rug for my living room a month ago.",
            "longmemeval_session_id=answer_2 I rearranged the furniture three weeks ago.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    interval_rows = [row for row in rows if row.get("include_reason") in {"relative_month_anchor", "relative_week_anchor"}]

    assert "relative_week_interval_answer=One week" in bundle
    assert [(row["source_group"], row["value"], row["unit"], row["include_reason"]) for row in interval_rows] == [
        ("answer_1", "1", "months_ago", "relative_month_anchor"),
        ("answer_2", "3", "weeks_ago", "relative_week_anchor"),
    ]


def test_anniversary_interval_bundle_emits_month_day_ledger_rows() -> None:
    """Anniversary subtraction should preserve both dated inputs."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many months before my anniversary did Rachel get engaged?",
        source_results=[
            "longmemeval_session_id=answer_aaf71ce2_2 My close friend Rachel got engaged last month on May 15th.",
            "longmemeval_session_id=answer_aaf71ce2_3 Our anniversary is coming up on July 22nd.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    anniversary_rows = [row for row in rows if row.get("fact_id", "").startswith("anniversary_engagement:")]

    assert "anniversary_engagement_interval_answer=2 months" in bundle
    assert [(row["source_group"], row["value"], row["unit"], row["include_reason"]) for row in anniversary_rows] == [
        ("answer_aaf71ce2_2", "5/15", "month_day", "engagement_date"),
        ("answer_aaf71ce2_3", "7/22", "month_day", "anniversary_date"),
    ]


def test_categorical_temporal_bundle_emits_choice_ledger_rows() -> None:
    """Parent, recency, and temporal-order answers should expose compared candidates."""
    cases = [
        (
            "Who became a parent first, Rachel or Alex?",
            [
                "longmemeval_session_id=answer_1 Rachel's twins Jackson and Julia were born on February 12th.",
                "longmemeval_session_id=answer_2 My cousin Alex just adopted a baby girl from China in January.",
            ],
            "parent_order_answer=Alex",
            "parent_order_candidate",
            [("answer_2", "Alex"), ("answer_1", "Rachel")],
        ),
        (
            "Which streaming service did I start using most recently?",
            [
                "longmemeval_session_id=answer_1 I started using Hulu a few months ago.",
                "longmemeval_session_id=answer_2 I started using Disney+ last week.",
            ],
            "recency_answer=Disney+",
            "recency_candidate",
            [("answer_2", "Disney+"), ("answer_1", "Hulu")],
        ),
        (
            "Who did I meet first, Mark and Sarah or Tom?",
            [
                "longmemeval_session_id=answer_1 I met Mark and Sarah on a beach trip about a month ago.",
                "longmemeval_session_id=answer_2 A few months ago, I volunteered and met a guy named Tom.",
            ],
            "temporal_order_answer=Tom",
            "temporal_order_candidate",
            [("answer_2", "Tom"), ("answer_1", "Mark and Sarah")],
        ),
    ]

    for query, source_results, answer_line, include_reason, expected in cases:
        bundle = retrieval_plan.source_synthesis_bundle(
            query=query,
            source_results=source_results,
            limit=5,
        )

        assert bundle is not None
        rows = [
            json.loads(line.removeprefix("ledger_row="))
            for line in bundle.splitlines()
            if line.startswith("ledger_row=")
        ]
        choice_rows = [row for row in rows if row.get("include_reason") == include_reason]

        assert answer_line in bundle
        assert [(row["source_group"], row["candidate"]) for row in choice_rows] == expected


def test_source_ordering_reuses_context_tokens_across_ranking_passes(monkeypatch) -> None:
    """Source synthesis ordering should not tokenize every context for every sort pass."""
    tokenized_contexts: list[str] = []
    original_source_tokens = retrieval_plan.source_tokens

    def tracking_source_tokens(text: str) -> list[str]:
        if text.startswith("source_path="):
            tokenized_contexts.append(text)
        return original_source_tokens(text)

    monkeypatch.setattr(retrieval_plan, "source_tokens", tracking_source_tokens)
    contexts = [
        f"source_path=doc-{index}.md longmemeval_session_id=answer-{index} I bought bike gear for ${index}."
        for index in range(8)
    ]
    token_cache = retrieval_plan._SourceTokenCache(tokens={})

    ordered = retrieval_plan.query_specific_source_order(
        "How much total money have I spent on bike-related expenses?",
        contexts,
        token_cache=token_cache,
    )
    retrieval_plan.evidence_source_order(
        "How much total money have I spent on bike-related expenses?",
        ordered,
        score_cache=retrieval_plan._SourceEvidenceScoreCache(
            query="irrelevant",
            scores=dict.fromkeys(contexts, 0),
        ),
        token_cache=token_cache,
    )

    assert len(tokenized_contexts) == len(contexts)


def test_retrieval_source_tokens_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Retrieval hot-path tokenization should not compile regex strings on every call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("source_tokens should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "findall", fail)
    monkeypatch.setattr(retrieval_plan.re, "search", fail)
    monkeypatch.setattr(retrieval_plan.re, "split", fail)

    assert retrieval_plan.source_tokens("source_path=longmemeval/foo-bar.md") == [
        "source_path",
        "source",
        "path",
        "longmemeval/foo-bar.md",
        "longmemeval",
        "foo",
        "bar.md",
    ]


def test_retrieval_source_tokens_uses_constant_time_separator_check(monkeypatch) -> None:
    """Retrieval token splitting should not run a regex search for every token."""
    monkeypatch.setattr(
        builtins,
        "any",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("separator checks should not allocate generator scans")
        ),
    )

    assert not hasattr(retrieval_plan, "_SOURCE_TOKEN_HAS_SEPARATOR_RE")

    assert retrieval_plan.source_tokens("source_path=longmemeval/foo-bar.md") == [
        "source_path",
        "source",
        "path",
        "longmemeval/foo-bar.md",
        "longmemeval",
        "foo",
        "bar.md",
    ]


def test_retrieval_source_tokens_caches_repeated_text_without_mutation_leak(monkeypatch) -> None:
    """Repeated source tokenization should reuse parsing while returning safe lists."""
    retrieval_plan._source_token_tuple.cache_clear()
    calls = 0
    original_token_re = retrieval_plan._SOURCE_TOKEN_RE

    class TrackingTokenRegex:
        def findall(self, text: str) -> list[str]:
            nonlocal calls
            calls += 1
            return original_token_re.findall(text)

    monkeypatch.setattr(retrieval_plan, "_SOURCE_TOKEN_RE", TrackingTokenRegex())
    text = "source_path=longmemeval/foo-bar.md I bought bike gear."

    first = retrieval_plan.source_tokens(text)
    first.append("mutated")
    second = retrieval_plan.source_tokens(text)

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


def test_source_context_provenance_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Source-lane provenance parsing should not compile regex strings per candidate."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("source provenance parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    context = (
        "citation=file://longmemeval/session-1/chunk-0001.md "
        "source_path=longmemeval/session-1/chunk-0001.md "
        "longmemeval_session_id=answer-bike"
    )

    assert retrieval_plan.source_context_group(context) == "answer-bike"
    assert retrieval_plan.source_context_namespace(context) == "longmemeval/session-1"


def test_source_context_citation_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Citation extraction should not compile regex strings per source candidate."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("source citation parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert (
        retrieval_plan.source_context_citation(
            "role=user citation=eventloom://default/events/10#abc source_path=docs/guide.md"
        )
        == "eventloom://default/events/10#abc"
    )


def test_graph_answer_concepts_use_compiled_regex_helpers(monkeypatch) -> None:
    """Graph-to-source query expansion should not compile regex strings per result."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("graph concept extraction should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "findall", fail)
    monkeypatch.setattr(retrieval_plan.re, "fullmatch", fail)

    assert retrieval_plan.graph_answer_concepts(
        [
            "entity=Bike Goal summary=Rachel Project Alpha source=Event",
            "entity=deadbeefcafebabe summary=hash should not become a concept",
        ],
        limit=3,
    ) == ["Bike Goal", "Rachel Project Alpha"]


def test_valid_entity_alias_uses_compiled_regex_helper(monkeypatch) -> None:
    """Possessive alias validation should not compile regex strings per candidate."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("alias validation should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan.valid_entity_alias("Rachel", "parent") is True
    assert retrieval_plan.valid_entity_alias("parent", "parent") is False


def test_possessive_entity_targets_use_compiled_regex_helper(monkeypatch) -> None:
    """Possessive bridge target extraction should not compile regex strings per query."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("possessive target extraction should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan.possessive_entity_targets("What is my new bike timeline?") == ("bike",)
    assert retrieval_plan.possessive_entity_targets("What is my old project status?") == ()


def test_possessive_alias_extraction_uses_cached_compiled_regex_helpers(monkeypatch) -> None:
    """Possessive alias extraction should reuse compiled target-specific regexes."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("possessive alias extraction should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan.aliases_for_possessive_target(
        "I bought my new bike Trek last month.",
        "bike",
    ) == ("Trek",)
    assert retrieval_plan.aliases_for_possessive_target(
        "I ordered lights for Trek because it is my bike.",
        "bike",
    ) == ("Trek",)


def test_source_evidence_score_uses_bounded_scoring_context(monkeypatch) -> None:
    """Ranking should score a compact evidence view instead of full source chunks."""
    score_lengths: list[int] = []

    def fake_score(query: str, context: str) -> int:
        del query
        score_lengths.append(len(context))
        return 0

    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", fake_score)
    context = (
        "citation=eventloom://default/events/10#abc "
        "source_path=longmemeval/session/chunk-0001.md "
        "longmemeval_session_id=answer-1 "
        "I bought bike lights for $40. "
        + "noise " * 1_000
    )

    retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        context,
    )

    assert score_lengths
    assert max(score_lengths) <= 1_600


def test_source_evidence_score_uses_ledger_score_without_rendering_projection(monkeypatch) -> None:
    """Ranking should avoid per-source projection rendering in the hot path."""
    score_calls = 0

    def fake_projection(query: str, contexts: list[str]) -> EvidenceProjection:
        del query, contexts
        raise AssertionError("source ranking should not render aggregate projections")

    def fake_score(query: str, context: str) -> int:
        del query, context
        nonlocal score_calls
        score_calls += 1
        return 7

    monkeypatch.setattr(
        retrieval_plan,
        "aggregate_candidate_projection",
        fake_projection,
    )
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", fake_score)

    score = retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert score_calls == 1
    assert score >= 7


def test_currency_personal_evidence_hint_uses_compiled_regex_helper(monkeypatch) -> None:
    """Currency source filtering should not compile amount regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("currency personal evidence should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._currency_personal_evidence_hint("I bought bike lights for $40.") is True
    assert retrieval_plan._currency_personal_evidence_hint("Estimated travel budget is $400.") is False


def test_alternative_terms_uses_compiled_single_letter_regex(monkeypatch) -> None:
    """Alternative-term parsing should not compile identifier regexes per candidate."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("alternative terms should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._alternative_terms("Task A or task B") == ("a", "b")


def test_query_person_alternatives_uses_compiled_regex_helper(monkeypatch) -> None:
    """Person alternative parsing should not compile name regexes per query."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("person alternatives should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._query_person_alternatives(
        "Who became a parent first, Rachel or Alex?"
    ) == ("rachel", "alex")


def test_flight_count_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Flight count parsing should not compile fixed regexes per source context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("flight count parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)
    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._flight_count_in_context("I took two flights each way with United.") == 4
    assert retrieval_plan._flight_count_in_context("I booked three flights during April.") == 3


def test_road_trip_drive_hours_use_compiled_regex_helpers(monkeypatch) -> None:
    """Road-trip duration parsing should not compile fixed regexes per context batch."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("road-trip duration parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)
    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan._road_trip_drive_hour_values(
        [
            "longmemeval_session_id=drive-1 I went on a road trip. It took me 4 hours to drive there.",
            "longmemeval_session_id=drive-2 I drove for three hours to Tennessee mountains.",
            "longmemeval_session_id=route-noise I drove from home 2 hours then another 3 hours.",
        ]
    ) == [4, 3]


def test_current_activity_weeks_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Current-activity duration parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("current activity duration parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)
    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan._current_activity_weeks(
        "How long have I been taking guitar lessons?",
        ["I have been taking guitar lessons for six weeks now."],
    ) == 6


def test_event_weeks_ago_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Event-age duration parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("event weeks-ago parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._event_weeks_ago(
        "How long ago did I start pottery classes?",
        ["I started pottery classes four weeks ago."],
    ) == 4


def test_role_duration_months_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Current-role duration parsing should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("role duration parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert (
        retrieval_plan._role_duration_months(
            "I've been working at NovaTech for about 4 years and 3 months now."
        )
        == 51
    )


def test_career_total_months_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Career-total parsing should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("career total parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan._career_total_months(
        ["I've been working professionally for 9 years and currently use a notebook."]
    ) == 108


def test_current_role_months_uses_compiled_employer_regex(monkeypatch) -> None:
    """Current-role lookup should not compile employer-token regexes per query."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("current role parsing should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "findall", fail)

    assert retrieval_plan._current_role_months(
        "How long have I been working before I started my current job at NovaTech?",
        [
            "I've been working at Acme for about 2 years now.",
            "I've been working at NovaTech for about 4 years and 3 months now.",
        ],
    ) == 51


def test_current_role_months_casefolds_each_context_once() -> None:
    """Current-role lookup should not normalize the same context per employer token."""

    class TrackingContext(str):
        casefold_calls = 0

        def casefold(self) -> str:
            self.casefold_calls += 1
            return super().casefold()

    context = TrackingContext("I've been working at Acme for about 2 years now.")

    assert retrieval_plan._current_role_months(
        "How long at Google, Meta, or NovaTech?",
        [context],
    ) is None
    assert context.casefold_calls == 1


def test_personal_current_age_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Personal age extraction should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("personal age parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._personal_current_age_values(
        [
            "I am 32 years old.",
            "I just turned 33.",
            "my age is 34",
        ]
    ) == [32, 33, 34]


def test_elapsed_year_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Elapsed-year extraction should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("elapsed year parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._elapsed_year_values(
        [
            "I started piano lessons five years ago.",
            "I have spent the past 3 years training.",
        ]
    ) == [5, 3]


def test_age_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Average-age extraction should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("age parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._age_values(
        [
            "I just turned 32.",
            "mom is 55 and dad is 58.",
        ]
    ) == [32, 55, 58]


def test_unit_values_caches_parameterized_regex(monkeypatch) -> None:
    """Parameterized unit extraction should compile once per unit pattern."""
    compile_calls = 0
    original_compile = retrieval_plan.re.compile

    def tracking_compile(*args, **kwargs):  # noqa: ANN001
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(retrieval_plan.re, "compile", tracking_compile)

    assert retrieval_plan._unit_values(["I spent 3 hours there."], unit_pattern=r"hours?") == [3.0]
    assert retrieval_plan._unit_values(["I spent 4 hours there."], unit_pattern=r"hours?") == [4.0]
    assert compile_calls == 1


def test_week_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Week duration parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("week duration parsing should use compiled regex helpers")

    retrieval_plan._unit_value_pattern.cache_clear()
    retrieval_plan._unit_value_pattern(r"weeks?")
    monkeypatch.setattr(retrieval_plan.re, "compile", fail)
    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan._week_values(["I trained for two weeks.", "I went last weekend."]) == [2.0, 1.0]


def test_month_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Month duration parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("month duration parsing should use compiled regex helpers")

    retrieval_plan._unit_value_pattern.cache_clear()
    retrieval_plan._unit_value_pattern(r"months?")
    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._month_values(["I trained for two months."]) == [2.0]


def test_clock_time_values_use_compiled_regex_helper(monkeypatch) -> None:
    """Clock time parsing should not compile a fixed regex per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("clock time parsing should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._clock_time_values(["I woke at 6:30 AM and slept at 10 PM."]) == [
        390,
        1320,
    ]


def test_relative_minute_offsets_use_compiled_regex_helper(monkeypatch) -> None:
    """Relative minute parsing should not compile a fixed regex per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("relative minute parsing should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._relative_minute_offsets(
        ["I woke 15 minutes earlier and went out 20 minutes later."]
    ) == [-15, 20]


def test_relative_days_ago_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Relative-day parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("relative-day parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._relative_days_ago("I met Tom about two weeks ago.") == 14


def test_source_evidence_score_reuses_query_synthesis_plan(monkeypatch) -> None:
    """Ranking should not rebuild the same query plan through each helper."""
    plan_calls = 0
    plan = SynthesisPlan(
        answer_type="sum",
        operation="sum_values",
        subject_terms=("bike",),
        required_kinds=("currency",),
        required_source_groups=2,
        reasons=("money",),
    )

    def fake_plan(query: str) -> SynthesisPlan:
        del query
        nonlocal plan_calls
        plan_calls += 1
        return plan

    monkeypatch.setattr(retrieval_plan, "build_synthesis_plan", fake_plan)
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", lambda query, context: 7)

    score = retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert score >= 7
    assert plan_calls == 1


def test_source_evidence_score_reuses_query_tokens_across_helpers(monkeypatch) -> None:
    """Ranking should not retokenize the same query in every helper branch."""
    query = "How much total money have I spent on bike-related expenses?"
    query_token_calls = 0
    original_source_tokens = retrieval_plan.source_tokens

    def tracking_source_tokens(text: str) -> list[str]:
        nonlocal query_token_calls
        if text == query:
            query_token_calls += 1
        return original_source_tokens(text)

    monkeypatch.setattr(retrieval_plan, "source_tokens", tracking_source_tokens)

    score = retrieval_plan.source_evidence_score(
        query,
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert score > 0
    assert query_token_calls <= 8


def test_source_evidence_score_cache_reuses_query_state_across_contexts(monkeypatch) -> None:
    """One source-ranking pass should not rebuild query state per context."""
    query = "How much total money have I spent on bike-related expenses?"
    query_token_calls = 0
    plan_calls = 0
    original_source_tokens = retrieval_plan.source_tokens
    plan = SynthesisPlan(
        answer_type="sum",
        operation="sum_values",
        subject_terms=("bike",),
        required_kinds=("currency",),
        required_source_groups=2,
        reasons=("money",),
    )

    def tracking_source_tokens(text: str) -> list[str]:
        nonlocal query_token_calls
        if text == query:
            query_token_calls += 1
        return original_source_tokens(text)

    def tracking_plan(text: str) -> SynthesisPlan:
        del text
        nonlocal plan_calls
        plan_calls += 1
        return plan

    monkeypatch.setattr(retrieval_plan, "source_tokens", tracking_source_tokens)
    monkeypatch.setattr(retrieval_plan, "build_synthesis_plan", tracking_plan)
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", lambda query, context: 0)

    retrieval_plan.evidence_source_order(
        query,
        [
            f"source_path=doc-{index}.md I bought bike gear for ${index}."
            for index in range(10)
        ],
        score_cache=retrieval_plan._SourceEvidenceScoreCache(query=query, scores={}),
    )

    assert query_token_calls <= 2
    assert plan_calls == 1


def test_aggregate_evidence_score_builds_only_required_currency_ledger(monkeypatch) -> None:
    """Currency query ranking should not build unrelated count/duration/date ledgers."""
    calls: list[str] = []
    plan = SynthesisPlan(
        answer_type="sum",
        operation="sum_values",
        subject_terms=("bike",),
        required_kinds=("currency",),
        required_source_groups=2,
        reasons=("money",),
    )

    def empty_ledger(name: str):
        def build(query: str, contexts: list[str], **kwargs: object) -> EvidenceLedger:
            del query, contexts, kwargs
            calls.append(name)
            return EvidenceLedger(plan=plan, rows=())

        return build

    monkeypatch.setattr(evidence_candidates, "build_count_ledger", empty_ledger("count"))
    monkeypatch.setattr(evidence_candidates, "build_currency_ledger", empty_ledger("currency"))
    monkeypatch.setattr(evidence_candidates, "build_duration_ledger", empty_ledger("duration"))
    monkeypatch.setattr(evidence_candidates, "build_date_ledger", empty_ledger("date"))

    evidence_candidates.aggregate_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert calls == ["currency"]


def test_aggregate_evidence_score_reuses_plan_in_required_ledger(monkeypatch) -> None:
    """Single-source ranking should not rebuild the same synthesis plan inside ledgers."""
    plan_calls = 0
    plan = SynthesisPlan(
        answer_type="sum",
        operation="sum_values",
        subject_terms=("bike",),
        required_kinds=("currency",),
        required_source_groups=2,
        reasons=("money",),
    )

    def fake_plan(query: str) -> SynthesisPlan:
        del query
        nonlocal plan_calls
        plan_calls += 1
        return plan

    monkeypatch.setattr(evidence_candidates, "build_synthesis_plan", fake_plan)
    monkeypatch.setattr(synthesis, "build_synthesis_plan", fake_plan)

    score = evidence_candidates.aggregate_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert score > 0
    assert plan_calls == 1


def test_source_evidence_score_skips_irrelevant_currency_documents(monkeypatch) -> None:
    """Ranking should avoid expensive currency parsing for unrelated money chunks."""
    score_calls = 0

    def fake_score(query: str, context: str) -> int:
        del query, context
        nonlocal score_calls
        score_calls += 1
        return 99

    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", fake_score)

    score = retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        (
            "longmemeval_session_id=answer-noise "
            "assistant: You might consider a budget of $1,000 for flights, "
            "$300 for lodging, and $200 for meals."
        ),
    )

    assert score_calls == 0
    assert score == 0


def test_source_evidence_score_skips_currency_ledger_when_context_has_no_amount(monkeypatch) -> None:
    """Currency ranking should not build a ledger for focused contexts with no money evidence."""
    score_calls = 0

    def fake_score(query: str, context: str) -> int:
        del query, context
        nonlocal score_calls
        score_calls += 1
        return 7

    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", fake_score)

    score = retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        (
            "longmemeval_session_id=answer-bike role=user "
            "I cleaned my bike and checked the chain before the weekend ride."
        ),
    )

    assert score_calls == 0
    assert score >= 0


def test_luxury_currency_ledger_keeps_sentence_label_before_amount() -> None:
    """Currency filtering should keep labelled luxury purchases with amount-only clauses."""
    ledger = build_currency_ledger(
        "What is the total amount I spent on luxury items in the past few months?",
        [
            (
                "longmemeval_session_id=answer-gown role=user "
                "I recently bought a luxury evening gown for a wedding. "
                "It was a big purchase, $800, but I felt like I needed it."
            )
        ],
    )

    included = ledger.included(kind="currency")

    assert len(included) == 1
    assert included[0].value == "800.0"
    assert included[0].label == "luxury evening gown"


def test_luxury_currency_filter_uses_recovered_purchase_label() -> None:
    """Focused currency filtering should consider labels recovered from prior sentences."""
    ledger = build_currency_ledger(
        "What is the total amount I spent on luxury items in the past few months?",
        [
            (
                "longmemeval_session_id=answer-bag role=user "
                "I got a luxury designer handbag from Gucci for $1,200."
            ),
            (
                "longmemeval_session_id=answer-gown role=user "
                "I recently bought a luxury evening gown for a wedding. "
                "It was a big purchase, $800, but I felt like I needed it."
            ),
            (
                "longmemeval_session_id=answer-shirts role=user "
                "I recently bought a pack of graphic tees from H&M for $20."
            ),
        ],
    )

    included_values = {row.value for row in ledger.included(kind="currency")}
    excluded_values = {
        row.value: row.exclude_reason
        for row in ledger.excluded(kind="currency")
    }

    assert included_values == {"1200.0", "800.0"}
    assert excluded_values["20.0"] == "query_focus_mismatch"
