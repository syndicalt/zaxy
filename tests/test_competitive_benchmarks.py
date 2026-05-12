"""Competitive retrieval benchmark harness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from zaxy.benchmark import (
    BenchmarkCase,
    FlatJsonlRetriever,
    build_competitive_event_log,
    competitive_cases,
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
