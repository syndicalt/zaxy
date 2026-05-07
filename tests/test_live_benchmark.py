"""Tests for live retrieval benchmark runners."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.benchmark import build_competitive_event_log, competitive_cases
from zaxy.embedding import HashEmbeddingProvider
from zaxy.live_benchmark import (
    FROZEN_WORKLOAD_SUBJECTS,
    FROZEN_WORKLOAD_VERSION,
    MarkdownRetriever,
    MarkdownVectorRetriever,
    VectorRetriever,
    benchmark_retrievers,
    build_frozen_statistical_workload,
    corpus_from_event_log,
    report_to_markdown,
    workload_fingerprint,
    write_benchmark_report,
)


def test_cli_exposes_live_benchmark_command() -> None:
    """The public CLI should expose a reproducible live benchmark command."""
    cli = Path("src/zaxy/__main__.py").read_text(encoding="utf-8")
    script = Path("scripts/live-benchmark.sh").read_text(encoding="utf-8")

    assert "def benchmark(" in cli
    assert 'embedding_provider: str = typer.Option("openai"' in cli
    assert "build_live_zaxy_retriever" in cli
    assert "build_statistical_event_log" in cli
    assert "build_frozen_statistical_workload" in cli
    assert "--workload" in script
    assert "--subjects" in script
    assert "zaxy benchmark" in script


def test_live_benchmark_compares_all_retriever_backends(tmp_path: Path) -> None:
    """The benchmark core should compare md, vector, md+vector, and zaxy rows."""
    log = build_competitive_event_log(tmp_path / "bench.jsonl")
    corpus = corpus_from_event_log(log)
    provider = HashEmbeddingProvider(dimension=64)

    retrievers = {
        "md": MarkdownRetriever(corpus),
        "vector": VectorRetriever(corpus, provider),
        "md+vector": MarkdownVectorRetriever(corpus, provider),
        "zaxy": MarkdownRetriever(corpus),
    }

    report = benchmark_retrievers(retrievers, competitive_cases(), runs=2, limit=5)

    assert {summary.backend for summary in report.summaries} == {
        "md",
        "vector",
        "md+vector",
        "zaxy",
    }
    assert all(summary.runs == 2 for summary in report.summaries)
    assert all(summary.case_count == len(competitive_cases()) for summary in report.summaries)
    assert all(summary.latency_ms_p95 >= summary.latency_ms_p50 for summary in report.summaries)


def test_live_benchmark_outputs_machine_and_markdown_reports(tmp_path: Path) -> None:
    """Reports should be usable for automation and public documentation."""
    log = build_competitive_event_log(tmp_path / "bench.jsonl")
    corpus = corpus_from_event_log(log)
    provider = HashEmbeddingProvider(dimension=64)
    report = benchmark_retrievers(
        {
            "md": MarkdownRetriever(corpus),
            "vector": VectorRetriever(corpus, provider),
        },
        competitive_cases(),
        runs=1,
        limit=3,
    )

    output = write_benchmark_report(report, tmp_path / "report")
    markdown = report_to_markdown(report)
    payload = json.loads(output.json_path.read_text(encoding="utf-8"))

    assert output.json_path.name == "live-benchmark.json"
    assert output.markdown_path.name == "live-benchmark.md"
    assert payload["summaries"][0]["backend"] in {"md", "vector"}
    assert payload["workload"]["version"] == "ad-hoc"
    assert "| Backend | Mean score | p50 ms | p95 ms |" in markdown
    assert "Approx tokens" in markdown


def test_frozen_statistical_workload_has_stable_identity(tmp_path: Path) -> None:
    """Frozen workload metadata should make benchmark reports reproducible."""
    first_log, first_cases, first_workload = build_frozen_statistical_workload(tmp_path / "first.jsonl")
    second_log, second_cases, second_workload = build_frozen_statistical_workload(tmp_path / "second.jsonl")

    assert first_workload.version == FROZEN_WORKLOAD_VERSION
    assert first_workload.subjects == FROZEN_WORKLOAD_SUBJECTS
    assert first_workload.event_count == 500
    assert first_workload.case_count == 300
    assert first_workload.sha256 == second_workload.sha256
    assert first_workload.sha256 == workload_fingerprint(first_log, first_cases, FROZEN_WORKLOAD_VERSION)
    assert first_workload == second_workload
