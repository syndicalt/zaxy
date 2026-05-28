"""Tests for retrieval planning and source-lane synthesis helpers."""

from __future__ import annotations

import builtins

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
