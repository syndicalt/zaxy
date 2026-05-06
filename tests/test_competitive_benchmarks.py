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
    assert benchmark.stats["mean"] < 0.005
