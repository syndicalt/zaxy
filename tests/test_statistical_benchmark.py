"""Tests for statistically powered retrieval benchmark evaluation."""

from __future__ import annotations

from pathlib import Path

from zaxy.embedding import HashEmbeddingProvider
from zaxy.live_benchmark import (
    MarkdownRetriever,
    VectorRetriever,
    benchmark_retrievers,
    build_statistical_event_log,
    compare_target_to_baselines,
    corpus_from_event_log,
    report_to_markdown,
)


def test_statistical_workload_generates_hundreds_of_paired_queries(tmp_path: Path) -> None:
    """The statistical workload should be large enough for paired inference."""
    eventlog, cases = build_statistical_event_log(tmp_path / "stats.jsonl", subjects=100)

    assert len(eventlog.read_all()) == 500
    assert len(cases) == 300
    assert {case.category for case in cases} == {"current", "temporal", "traversal"}


def test_paired_comparison_reports_confidence_interval_and_p_value(tmp_path: Path) -> None:
    """Reports should include paired effect estimates against baselines."""
    eventlog, cases = build_statistical_event_log(tmp_path / "stats.jsonl", subjects=4)
    corpus = corpus_from_event_log(eventlog)
    provider = HashEmbeddingProvider(dimension=64)
    report = benchmark_retrievers(
        {
            "md": MarkdownRetriever(corpus),
            "vector": VectorRetriever(corpus, provider),
            "zaxy": MarkdownRetriever(corpus),
        },
        cases,
        runs=1,
        limit=8,
    )

    comparisons = compare_target_to_baselines(
        report,
        target_backend="zaxy",
        bootstrap_samples=200,
        seed=7,
    )
    markdown = report_to_markdown(report)

    assert {comparison.baseline_backend for comparison in comparisons} == {"md", "vector"}
    assert all(0.0 <= comparison.p_value <= 1.0 for comparison in comparisons)
    assert all(
        comparison.ci_low <= comparison.mean_difference <= comparison.ci_high
        for comparison in comparisons
    )
    assert "Paired comparisons" in markdown
    assert "Category summaries" in markdown
