"""Competitive retrieval benchmark harness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from zaxy.benchmark import (
    BenchmarkCase,
    FlatJsonlRetriever,
    build_competitive_event_log,
    competitive_cases,
    expected_terms_recall,
    score_retrieval,
)


def assert_benchmark_mean_under(benchmark: pytest.BenchmarkFixture, seconds: float) -> None:
    """Assert benchmark mean unless pytest-benchmark is intentionally disabled."""
    if benchmark.stats is None:
        return
    assert benchmark.stats["mean"] < seconds


def test_competitive_dataset_contains_temporal_and_stale_context_cases(tmp_path: Path) -> None:
    """The fixture dataset should exercise temporal and stale-context retrieval."""
    log_path = tmp_path / "bench.jsonl"
    log = build_competitive_event_log(log_path)
    cases = competitive_cases()

    assert log.verify().ok is True
    assert {case.category for case in cases} >= {"temporal", "stale_context", "traversal"}
    assert len(log.read_all()) >= 6


def test_score_retrieval_rewards_expected_and_penalizes_forbidden_terms() -> None:
    """Correctness score should reward hits and penalize stale or wrong context."""
    case = BenchmarkCase(
        name="current-theme",
        query="What is the current theme?",
        expected_terms=("theme=light",),
        forbidden_terms=("theme=dark",),
        category="stale_context",
    )

    perfect = score_retrieval(case, ["user preference theme=light"])
    stale = score_retrieval(case, ["theme=dark", "theme=light"])

    assert perfect.score == 1.0
    assert stale.score < perfect.score
    assert stale.forbidden_hits == ("theme=dark",)


def test_score_retrieval_accepts_semantic_answer_surface_forms() -> None:
    """Benchmark scoring should not miss equivalent answer wording."""
    article_case = BenchmarkCase(
        name="racket-source",
        query="Where did I buy my new tennis racket from?",
        expected_terms=("the sports store downtown",),
    )
    date_case = BenchmarkCase(
        name="fundraiser-date",
        query="When did I volunteer at the fundraising dinner?",
        expected_terms=("February 14th",),
    )
    distributed_location_case = BenchmarkCase(
        name="study-abroad",
        query="Where did I attend for my study abroad program?",
        expected_terms=("University of Melbourne in Australia",),
    )

    article = score_retrieval(article_case, ["I got it from a sports store downtown."])
    date = score_retrieval(date_case, ["I volunteered at the dinner on Valentine's Day."])
    distributed_location = score_retrieval(
        distributed_location_case,
        ["My study abroad program was at the University of Melbourne. I loved Australia."],
    )

    assert article.score == 1.0
    assert date.score == 1.0
    assert distributed_location.score == 1.0


def test_score_retrieval_accepts_structured_scalar_answer_fields() -> None:
    """One-token expected answers should match compact structured answer fields."""
    yes_case = BenchmarkCase(
        name="boolean-answer",
        query="Did I receive a higher percentage discount?",
        expected_terms=("Yes.",),
    )
    numeric_case = BenchmarkCase(
        name="numeric-answer",
        query="How many minutes did I exceed the target?",
        expected_terms=("12",),
    )

    yes_score = score_retrieval(
        yes_case,
        [
            "memory_checkout_compact=true\n"
            "checkout_answer_candidate=true\n"
            "answer_key=boolean_comparison_answer\n"
            "answer=Yes"
        ],
    )
    numeric_score = score_retrieval(
        numeric_case,
        ["zaxy_synthesis_bundle=true\ncandidate_type=duration\nminutes_answer=12"],
    )

    assert yes_score.score == 1.0
    assert numeric_score.score == 1.0


def test_score_retrieval_accepts_inflected_action_answer_surface_forms() -> None:
    """Action answers should match cited evidence across ordinary inflections."""
    case = BenchmarkCase(
        name="fence-repair",
        query="What outdoor task did I complete?",
        expected_terms=("Fixing the fence",),
    )

    score = score_retrieval(
        case,
        ["I just fixed that broken fence in the backyard."],
    )

    assert score.score == 1.0
    assert score.expected_hits == case.expected_terms


def test_score_retrieval_accepts_structured_absence_guidance() -> None:
    """Absence answers should not require exact prose reproduction."""
    case = BenchmarkCase(
        name="missing-hamster",
        query="What is the name of my hamster?",
        expected_terms=(
            "You did not mention this information. You mentioned your cat Luna but not your hamster.",
        ),
    )

    score = score_retrieval(
        case,
        [
            (
                "zaxy_absence_check=true synthesis_mode=absence_check "
                "not_mentioned_candidate=hamster "
                "answer_guidance=You did not mention this information. "
                "source_id=answer citation=eventloom://benchmark/events/1#abc "
                "snippet=I mentioned my cat Luna during the conversation."
            )
        ],
    )

    assert score.score == 1.0
    assert score.expected_hits == case.expected_terms


def test_score_retrieval_accepts_structured_interval_answers() -> None:
    """Typed interval synthesis should satisfy equivalent long-form expected answers."""
    day_case = BenchmarkCase(
        name="walk-cleanup",
        query="How many days had passed between the Walk for Hunger and Coastal Cleanup?",
        expected_terms=("14 days. 8 days (including the last day) is also acceptable.",),
    )
    week_case = BenchmarkCase(
        name="rug-room",
        query="How long had I been using the new rug when I rearranged the room?",
        expected_terms=("One week. Answers ranging from 7 days to 10 days are also acceptable.",),
    )

    day_score = score_retrieval(
        day_case,
        [
            (
                "zaxy_synthesis_bundle=true candidate_type=date_interval "
                "date_interval_days=14 "
                "date_interval_answer=14 days. 15 days (including the last day) is also acceptable."
            )
        ],
    )
    week_score = score_retrieval(
        week_case,
        [
            (
                "zaxy_synthesis_bundle=true relative_day_interval=7 days "
                "relative_week_interval=1 weeks relative_week_interval_answer=One week"
            )
        ],
    )

    assert day_score.score == 1.0
    assert week_score.score == 1.0


def test_score_retrieval_accepts_structured_duration_hour_answers() -> None:
    """Typed duration totals should satisfy prose hour-total expected answers."""
    case = BenchmarkCase(
        name="roadtrip-hours",
        query="How many hours did I spend driving?",
        expected_terms=("15 hours for getting to the destinations (or 30 hours round trip)",),
    )

    score = score_retrieval(
        case,
        [
            (
                "zaxy_synthesis_bundle=true candidate_type=duration "
                "duration_values=6 hours,4 hours,5 hours "
                "duration_total_answer=15 hours"
            )
        ],
    )

    assert score.score == 1.0


def test_score_retrieval_accepts_inflected_structured_absence_answers() -> None:
    """Absence scoring should compare missing targets by variants, not exact prose."""
    case = BenchmarkCase(
        name="missing-ipad",
        query="How many days before I bought my iPad did I attend the Holiday Market?",
        expected_terms=(
            "The information provided is not enough. You mentioned getting the iPhone 13 Pro "
            "and attending the market, but you did not mention buying an iPad.",
        ),
    )

    score = score_retrieval(
        case,
        [
            (
                "zaxy_absence_check=true synthesis_mode=absence_check "
                "not_mentioned_candidate=bought ipad "
                "known_related_evidence=attend holiday market "
                "answer_guidance=The information provided is not enough. "
                "snippet=I got the iPhone 13 Pro and attended the Holiday Market."
            )
        ],
    )

    assert score.score == 1.0
    assert score.expected_hits == case.expected_terms


def test_score_retrieval_accepts_parenthetical_acronym_surface_forms() -> None:
    """Expected answers with parenthetical acronyms should match acronym-only evidence."""
    case = BenchmarkCase(
        name="ucla-degree",
        query="Where did I complete my Bachelor's degree in Computer Science?",
        expected_terms=("University of California, Los Angeles (UCLA)",),
    )

    score = score_retrieval(case, ["I completed my undergrad in CS from UCLA."])

    assert score.score == 1.0
    assert score.expected_hits == ("University of California, Los Angeles (UCLA)",)


def test_score_retrieval_reports_identity_and_source_recall() -> None:
    """Benchmark scoring should expose separate identity and source recall diagnostics."""
    case = BenchmarkCase(
        name="source-recall",
        query="Where did I save the migration plan?",
        expected_terms=("migration plan",),
        identity_terms=("MigrationPlan.md", "reviewer note"),
        source_terms=("docs/roadmap.md", "reports/missing.json"),
    )

    score = score_retrieval(
        case,
        [
            "MigrationPlan.md records the migration plan.",
            "The cited source path is docs/roadmap.md.",
        ],
    )

    assert score.score == 1.0
    assert score.identity_recall == 0.5
    assert score.identity_hits == ("MigrationPlan.md",)
    assert score.missing_identities == ("reviewer note",)
    assert score.source_recall == 0.5
    assert score.source_hits == ("docs/roadmap.md",)
    assert score.missing_sources == ("reports/missing.json",)


def test_expected_terms_recall_handles_empty_and_partial_answers() -> None:
    """Recall@k scoring should expose missing expected terms without applying forbidden penalties."""
    empty_case = BenchmarkCase(name="empty", query="What changed?", expected_terms=())
    partial_case = BenchmarkCase(
        name="partial",
        query="Which two cities did I visit?",
        expected_terms=("Berlin", "Lisbon"),
    )

    assert expected_terms_recall(empty_case, ["anything"]) is None
    assert expected_terms_recall(partial_case, ["I visited Lisbon in April."]) == 0.5


def test_score_retrieval_rejects_unsupported_absence_answers() -> None:
    """Answer scoring should not award unsupported abstentions."""
    absence_case = BenchmarkCase(
        name="unsupported-absence",
        query="What was my bicycle brand?",
        expected_terms=("You did not mention your bicycle brand.",),
    )

    absence = score_retrieval(absence_case, ["not_mentioned_candidate=skateboard"])

    assert absence.score == 0.0
    assert absence.missing_expected == absence_case.expected_terms


def test_score_retrieval_accepts_numeric_week_interval_surfaces() -> None:
    """Structured interval scoring should match numeric week evidence as well as worded answers."""
    case = BenchmarkCase(
        name="two-week-delay",
        query="How long was the delay?",
        expected_terms=("2 weeks",),
    )

    score = score_retrieval(
        case,
        ["date_interval_weeks=2 weeks\ncandidate_type=date_interval"],
    )

    assert score.score == 1.0
    assert score.expected_hits == ("2 weeks",)


def test_flat_jsonl_baseline_exposes_stale_context_limitation(tmp_path: Path) -> None:
    """Flat event scanning should surface both old and new preference values."""
    log_path = tmp_path / "bench.jsonl"
    log = build_competitive_event_log(log_path)
    baseline = FlatJsonlRetriever(log)
    case = next(c for c in competitive_cases() if c.name == "current-theme")

    contexts = baseline.query(case.query, temporal_point=case.temporal_point)
    score = score_retrieval(case, contexts)

    assert any("theme=dark" in context for context in contexts)
    assert any("theme=light" in context for context in contexts)
    assert score.score < 1.0


def test_flat_jsonl_baseline_scores_lower_on_temporal_query(tmp_path: Path) -> None:
    """Flat event scanning should rank worse when the query asks what was true then."""
    log_path = tmp_path / "bench.jsonl"
    log = build_competitive_event_log(log_path)
    baseline = FlatJsonlRetriever(log)
    case = next(c for c in competitive_cases() if c.name == "theme-before-change")

    contexts = baseline.query(case.query, temporal_point=case.temporal_point)
    score = score_retrieval(case, contexts)

    assert any("theme=dark" in context for context in contexts)
    assert any("theme=light" in context for context in contexts)
    assert score.score < 1.0


def test_flat_jsonl_baseline_latency_floor(
    tmp_path: Path,
    benchmark: pytest.BenchmarkFixture,
) -> None:
    """Flat baseline latency should be recorded as the competitor floor."""
    log_path = tmp_path / "bench.jsonl"
    log = build_competitive_event_log(log_path)
    baseline = FlatJsonlRetriever(log)
    case = next(c for c in competitive_cases() if c.name == "current-theme")

    def _query() -> None:
        baseline.query(case.query, temporal_point=case.temporal_point)

    benchmark(_query)
    assert_benchmark_mean_under(benchmark, 0.005)
