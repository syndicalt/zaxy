"""Tests for live retrieval benchmark runners."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from zaxy.benchmark import build_competitive_event_log, competitive_cases
from zaxy.embedding import HashEmbeddingProvider
from zaxy.live_benchmark import (
    CONSOLIDATION_WORKLOAD_VERSION,
    FROZEN_WORKLOAD_SUBJECTS,
    FROZEN_WORKLOAD_VERSION,
    LONGMEMEVAL_WORKLOAD_VERSION,
    SUITE_WORKLOAD_VERSION,
    BenchmarkChunk,
    BM25Retriever,
    CachedEmbeddingProvider,
    CentroidConsolidationRetriever,
    MarkdownRetriever,
    MarkdownVectorRetriever,
    RankFusionRetriever,
    VectorRetriever,
    ZaxyRetriever,
    benchmark_retrievers,
    build_benchmark_suite_workload,
    build_consolidation_collapse_workload,
    build_frozen_statistical_workload,
    build_longmemeval_workload,
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
    assert "build_benchmark_suite_workload" in cli
    assert "build_consolidation_collapse_workload" in cli
    assert "build_longmemeval_workload" in cli
    assert "lexical_retriever=BM25Retriever(corpus)" in cli
    assert "--workload" in script
    assert "--dataset" in script
    assert "--subjects" in script
    assert "--documents" in script
    assert "--sessions" in script
    assert "zaxy benchmark" in script


def test_longmemeval_workload_loads_public_memory_dataset(tmp_path: Path) -> None:
    """LongMemEval loader should convert answer sessions into identity-recall cases."""
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "What degree did I graduate with?",
                    "answer": "Business Administration",
                    "answer_session_ids": ["answer-1"],
                    "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 03:24"],
                    "haystack_session_ids": ["distractor-1", "answer-1"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "What is a graph database?"}],
                        [
                            {
                                "role": "user",
                                "content": "I graduated with a Business Administration degree.",
                            }
                        ],
                    ],
                },
                {
                    "question_id": "q2",
                    "question_type": "multi-session-assistant",
                    "question": "Which project did we rename?",
                    "answer": "Atlas",
                    "answer_session_ids": ["answer-2"],
                    "haystack_dates": ["2023/05/22 (Mon) 08:00"],
                    "haystack_session_ids": ["answer-2"],
                    "haystack_sessions": [
                        [{"role": "assistant", "content": "We renamed Project Atlas."}]
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    eventlog, cases, workload = build_longmemeval_workload(
        tmp_path / "longmemeval.jsonl",
        dataset,
        questions=1,
    )
    corpus = corpus_from_event_log(eventlog)

    assert workload.version == LONGMEMEVAL_WORKLOAD_VERSION
    assert workload.subjects == 1
    assert workload.sessions == 2
    assert workload.event_count == 2
    assert workload.case_count == 1
    assert workload.lanes == ("longmemeval",)
    assert cases[0].name == "longmemeval-q1"
    assert cases[0].expected_terms == ("Business Administration",)
    assert cases[0].identity_terms == ("answer-1",)
    assert any("longmemeval_session_id=answer-1" in chunk.text for chunk in corpus)
    assert workload.sha256 == workload_fingerprint(
        eventlog,
        cases,
        LONGMEMEVAL_WORKLOAD_VERSION,
    )


def test_live_benchmark_compares_all_retriever_backends(tmp_path: Path) -> None:
    """The benchmark core should compare md, BM25, vector, md+vector, and zaxy rows."""
    log = build_competitive_event_log(tmp_path / "bench.jsonl")
    corpus = corpus_from_event_log(log)
    provider = HashEmbeddingProvider(dimension=64)

    retrievers = {
        "md": MarkdownRetriever(corpus),
        "bm25": BM25Retriever(corpus),
        "vector": VectorRetriever(corpus, provider),
        "md+vector": MarkdownVectorRetriever(corpus, provider),
        "zaxy": MarkdownRetriever(corpus),
    }

    report = benchmark_retrievers(retrievers, competitive_cases(), runs=2, limit=5)

    assert {summary.backend for summary in report.summaries} == {
        "md",
        "bm25",
        "vector",
        "md+vector",
        "zaxy",
    }
    assert all(summary.runs == 2 for summary in report.summaries)
    assert all(summary.case_count == len(competitive_cases()) for summary in report.summaries)
    assert all(summary.latency_ms_p95 >= summary.latency_ms_p50 for summary in report.summaries)


def test_bm25_retriever_ranks_specific_identifier_above_generic_overlap() -> None:
    """BM25 should prefer rare identity terms over broad keyword overlap."""
    corpus = (
        BenchmarkChunk(
            "generic",
            "release cache context mentions rollback planning and deployment notes",
        ),
        BenchmarkChunk(
            "target",
            "release cache context records doc-code-0001 as the rollback owner source",
        ),
    )

    results = BM25Retriever(corpus).query("cache rollback doc-code-0001", limit=1)

    assert results == [corpus[1].text]


def test_rank_fusion_retriever_preserves_complementary_hits() -> None:
    """Fusion should keep lexical hits when another retriever misses them."""
    corpus = (
        BenchmarkChunk("answer", "longmemeval_session_id=answer-1 degree Business Administration"),
        BenchmarkChunk("distractor", "degree graph-shaped context about unrelated project planning"),
    )

    fused = RankFusionRetriever(
        {
            "graph": MarkdownRetriever((corpus[1],)),
            "bm25": BM25Retriever(corpus),
        }
    )

    results = fused.query("What degree did I graduate with?", limit=2)

    assert any("answer-1" in result for result in results)
    assert len(results) == 2


async def test_zaxy_retriever_can_fuse_graph_and_lexical_results() -> None:
    """Live Zaxy benchmark retrieval should support lexical fusion."""
    corpus = (
        BenchmarkChunk("answer", "longmemeval_session_id=answer-1 degree Business Administration"),
    )

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, limit, embedding
            return [SimpleNamespace(content="graph context about the user degree")]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async("What degree did I graduate with?", limit=2)

    assert any("graph context" in result for result in results)
    assert any("answer-1" in result for result in results)


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
    assert "| Backend | Mean score | Identity recall | p50 ms | p95 ms |" in markdown
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


def test_benchmark_suite_workload_covers_memory_document_and_transcript_lanes(
    tmp_path: Path,
) -> None:
    """The suite workload should be broad enough to represent production context."""
    eventlog, cases, workload = build_benchmark_suite_workload(
        tmp_path / "suite.jsonl",
        subjects=4,
        documents=5,
        sessions=3,
    )

    assert workload.version == SUITE_WORKLOAD_VERSION
    assert workload.subjects == 4
    assert workload.documents == 5
    assert workload.sessions == 3
    assert workload.lanes == (
        "current",
        "temporal",
        "traversal",
        "document",
        "transcript",
        "mixed",
    )
    assert workload.event_count == len(eventlog.read_all())
    assert workload.case_count == len(cases)
    assert workload.event_count == 20 + 5 + 6
    assert workload.case_count == 12 + 5 + 3 + 3
    assert {case.category for case in cases} == set(workload.lanes)
    assert any(case.name.startswith("document-source-") for case in cases)
    assert any(case.name.startswith("session-decision-") for case in cases)
    assert any(case.name.startswith("mixed-release-") for case in cases)


def test_benchmark_suite_workload_has_stable_identity(tmp_path: Path) -> None:
    """Suite fingerprints should be stable across output paths."""
    first_log, first_cases, first_workload = build_benchmark_suite_workload(
        tmp_path / "first.jsonl",
        subjects=3,
        documents=4,
        sessions=2,
    )
    second_log, second_cases, second_workload = build_benchmark_suite_workload(
        tmp_path / "second.jsonl",
        subjects=3,
        documents=4,
        sessions=2,
    )

    assert first_workload == second_workload
    assert first_workload.sha256 == workload_fingerprint(
        first_log,
        first_cases,
        SUITE_WORKLOAD_VERSION,
    )
    assert first_workload.sha256 == workload_fingerprint(
        second_log,
        second_cases,
        SUITE_WORKLOAD_VERSION,
    )


def test_consolidation_workload_targets_identity_collapse(tmp_path: Path) -> None:
    """The consolidation workload should isolate identity-preservation failures."""
    eventlog, cases, workload = build_consolidation_collapse_workload(
        tmp_path / "collapse.jsonl",
        identities=8,
    )

    assert workload.version == CONSOLIDATION_WORKLOAD_VERSION
    assert workload.event_count == 8
    assert workload.case_count == 8
    assert workload.documents == 8
    assert workload.lanes == ("consolidation",)
    assert {case.category for case in cases} == {"consolidation"}
    assert all(case.identity_terms for case in cases)
    assert any("identity-code-0007" in case.identity_terms for case in cases)
    assert workload.sha256 == workload_fingerprint(
        eventlog,
        cases,
        CONSOLIDATION_WORKLOAD_VERSION,
    )


def test_identity_recall_exposes_centroid_consolidation_collapse(
    tmp_path: Path,
) -> None:
    """Centroid-style consolidation can score as relevant while losing exact identities."""
    eventlog, cases, workload = build_consolidation_collapse_workload(
        tmp_path / "collapse.jsonl",
        identities=6,
    )
    corpus = corpus_from_event_log(eventlog)
    provider = HashEmbeddingProvider(dimension=64)
    report = benchmark_retrievers(
        {
            "md": MarkdownRetriever(corpus),
            "centroid": CentroidConsolidationRetriever(corpus, provider),
        },
        cases,
        runs=1,
        limit=8,
        workload=workload,
    )

    summaries = {summary.backend: summary for summary in report.summaries}
    assert summaries["md"].mean_identity_recall == 1.0
    assert summaries["centroid"].mean_identity_recall is not None
    assert summaries["centroid"].mean_identity_recall < summaries["md"].mean_identity_recall
    assert any(run.missing_identities for run in report.runs if run.backend == "centroid")
    assert "Identity recall" in report_to_markdown(report)


def test_suite_report_renders_workload_dimensions(tmp_path: Path) -> None:
    """Reports should disclose suite dimensions for reproducibility."""
    eventlog, cases, workload = build_benchmark_suite_workload(
        tmp_path / "suite.jsonl",
        subjects=2,
        documents=3,
        sessions=2,
    )
    corpus = corpus_from_event_log(eventlog)
    report = benchmark_retrievers(
        {"md": MarkdownRetriever(corpus)},
        cases,
        runs=1,
        limit=4,
        workload=workload,
    )

    markdown = report_to_markdown(report)

    assert f"Workload: `{SUITE_WORKLOAD_VERSION}`" in markdown
    assert "- Documents: `3`" in markdown
    assert "- Sessions: `2`" in markdown
    assert "- Lanes: `current, temporal, traversal, document, transcript, mixed`" in markdown


def test_cached_embedding_provider_reuses_repeated_text() -> None:
    """Benchmark runs should not repeat hosted embedding calls for identical text."""
    class CountingProvider:
        dimension = 2

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> list[float]:
            self.calls += 1
            return [float(len(text)), 1.0]

    inner = CountingProvider()
    provider = CachedEmbeddingProvider(inner)

    assert provider.embed("same") == [4.0, 1.0]
    assert provider.embed("same") == [4.0, 1.0]
    assert provider.embed("other") == [5.0, 1.0]
    assert provider.cache_size == 2
    assert inner.calls == 2


def test_live_benchmark_script_help_mentions_frozen_workload() -> None:
    script = Path("scripts/live-benchmark.sh").read_text(encoding="utf-8")

    assert "--workload fixture|statistical|frozen|suite|consolidation" in script
