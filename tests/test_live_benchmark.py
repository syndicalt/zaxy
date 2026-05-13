"""Tests for live retrieval benchmark runners."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.benchmark import build_competitive_event_log, competitive_cases
from zaxy.embedding import HashEmbeddingProvider
from zaxy.event import EventLog
from zaxy.live_benchmark import (
    CONSOLIDATION_WORKLOAD_VERSION,
    CONTEXT_COLLAPSE_WORKLOAD_VERSION,
    FROZEN_WORKLOAD_SUBJECTS,
    FROZEN_WORKLOAD_VERSION,
    GRAPH_TRAVERSAL_WORKLOAD_VERSION,
    LONGMEMEVAL_WORKLOAD_VERSION,
    SOURCE_RECALL_WORKLOAD_VERSION,
    SUITE_WORKLOAD_VERSION,
    TEMPORAL_RECALL_WORKLOAD_VERSION,
    BenchmarkCase,
    BenchmarkChunk,
    BenchmarkReport,
    BenchmarkSummary,
    BenchmarkWorkload,
    BM25Retriever,
    CachedEmbeddingProvider,
    CentroidConsolidationRetriever,
    MarkdownRetriever,
    MarkdownVectorRetriever,
    RankFusionRetriever,
    VectorRetriever,
    ZaxyRetriever,
    _benchmark_projection_present,
    _mark_benchmark_projection,
    _source_lane_candidate_limit,
    _source_lane_query,
    benchmark_live_retrievers,
    benchmark_projection_cache_key,
    benchmark_retrievers,
    build_benchmark_suite_workload,
    build_consolidation_collapse_workload,
    build_context_collapse_workload,
    build_frozen_statistical_workload,
    build_graph_traversal_workload,
    build_longmemeval_workload,
    build_mempalace_workload_inventory,
    build_source_recall_workload,
    build_temporal_recall_workload,
    compare_benchmark_reports,
    corpus_from_event_log,
    format_benchmark_comparison,
    format_mempalace_workload_inventory,
    report_to_markdown,
    workload_fingerprint,
    write_benchmark_report,
)
from zaxy.query import ContextChunk
from zaxy.retrieval_intent import classify_retrieval_intent


def test_cli_exposes_live_benchmark_command() -> None:
    """The public CLI should expose a reproducible live benchmark command."""
    cli = Path("src/zaxy/__main__.py").read_text(encoding="utf-8")
    script = Path("scripts/live-benchmark.sh").read_text(encoding="utf-8")

    assert "def benchmark(" in cli
    assert "def benchmark_compare(" in cli
    assert "compare_benchmark_reports" in cli
    assert 'embedding_provider: str = typer.Option("openai"' in cli
    assert "build_live_zaxy_retriever" in cli
    assert "build_statistical_event_log" in cli
    assert "build_frozen_statistical_workload" in cli
    assert "build_benchmark_suite_workload" in cli
    assert "build_consolidation_collapse_workload" in cli
    assert "build_context_collapse_workload" in cli
    assert "build_graph_traversal_workload" in cli
    assert "build_mempalace_workload_inventory" in cli
    assert "build_longmemeval_workload" in cli
    assert "build_source_recall_workload" in cli
    assert "build_temporal_recall_workload" in cli
    assert "benchmark-inventory" in cli
    assert "lexical_retriever=BM25Retriever(corpus)" in cli
    assert "--embedding-cache" in cli
    assert "--progress" in cli
    assert "--reuse-projection" in cli
    assert "--baseline-backends" in cli
    assert "_build_benchmark_baselines" in cli
    assert "benchmark_projection_cache_key" in cli
    assert "--workload" in script
    assert "--dataset" in script
    assert "--subjects" in script
    assert "--documents" in script
    assert "--sessions" in script
    assert "--embedding-cache" in script
    assert "--progress" in script
    assert "context-collapse" in script
    assert "graph-traversal" in script
    assert "source-recall" in script
    assert "temporal-recall" in script
    assert "zaxy benchmark" in script


def test_graph_traversal_workload_is_frozen_and_requires_linked_events(tmp_path: Path) -> None:
    """Graph traversal should require crossing goal-task-completion edges."""
    eventlog, cases, workload = build_graph_traversal_workload(
        tmp_path / "graph-traversal.jsonl",
        subjects=4,
    )
    events = eventlog.read_all()

    assert workload.version == GRAPH_TRAVERSAL_WORKLOAD_VERSION
    assert workload.subjects == 4
    assert workload.event_count == 16
    assert workload.case_count == 4
    assert workload.lanes == ("graph-traversal",)
    assert {case.category for case in cases} == {"graph-traversal"}
    assert [event.type for event in events[:3]] == [
        "goal.created",
        "task.proposed",
        "task.completed",
    ]
    assert "graph-finisher-0000" not in events[1].payload["summary"]
    assert "graph-finisher-distractor-0000" not in events[1].payload["summary"]
    assert "graph-finisher-0000" not in events[2].payload["summary"]
    assert any(event.actor == "graph-finisher-distractor-0000" for event in events)
    assert all(case.identity_terms for case in cases)
    assert any(
        case.expected_terms == ("graph-finisher-0000", "graph-task-0000")
        for case in cases
    )
    assert any("goalTitle" in event.payload for event in events)
    assert workload.sha256 == workload_fingerprint(
        eventlog,
        cases,
        GRAPH_TRAVERSAL_WORKLOAD_VERSION,
    )


def test_source_recall_workload_is_frozen_and_cited(tmp_path: Path) -> None:
    """MemPalace-comparable source recall should target exact cited sources."""
    eventlog, cases, workload = build_source_recall_workload(
        tmp_path / "source-recall.jsonl",
        documents=4,
    )
    events = eventlog.read_all()
    corpus = corpus_from_event_log(eventlog)

    assert workload.version == SOURCE_RECALL_WORKLOAD_VERSION
    assert workload.documents == 4
    assert workload.event_count == 8
    assert workload.case_count == 4
    assert workload.lanes == ("source-recall",)
    assert {case.category for case in cases} == {"source-recall"}
    assert all(case.source_terms for case in cases)
    assert any(
        case.source_terms == ("source-recall/target/service-0000.md",)
        for case in cases
    )
    assert any("source_recall_answer_code" in event.payload for event in events)
    assert all("eventloom://benchmark/events/" in chunk.text for chunk in corpus)
    assert workload.sha256 == workload_fingerprint(
        eventlog,
        cases,
        SOURCE_RECALL_WORKLOAD_VERSION,
    )


def test_context_collapse_workload_is_frozen_and_checkpoint_backed(tmp_path: Path) -> None:
    """Context collapse should require recovering compact memory after noisy turns."""
    eventlog, cases, workload = build_context_collapse_workload(
        tmp_path / "context-collapse.jsonl",
        sessions=3,
        turns_per_session=5,
    )
    events = eventlog.read_all()

    assert workload.version == CONTEXT_COLLAPSE_WORKLOAD_VERSION
    assert workload.sessions == 3
    assert workload.event_count == 18
    assert workload.case_count == 3
    assert workload.lanes == ("context-collapse",)
    assert {case.category for case in cases} == {"context-collapse"}
    assert all(case.identity_terms for case in cases)
    assert any(case.expected_terms == ("collapseanswer0000",) for case in cases)
    assert [event.type for event in events[:5]] == ["transcript.turn"] * 5
    assert events[5].type == "hook.checkpoint"
    assert "collapseanswer0000" in events[5].payload["summary"]
    assert all(
        "collapseanswer0000" not in event.payload.get("content", "")
        for event in events[:5]
    )
    assert workload.sha256 == workload_fingerprint(
        eventlog,
        cases,
        CONTEXT_COLLAPSE_WORKLOAD_VERSION,
    )


def test_context_collapse_markdown_baseline_loses_late_checkpoint(
    tmp_path: Path,
) -> None:
    """Flat context windows should expose context-collapse misses."""
    eventlog, cases, workload = build_context_collapse_workload(
        tmp_path / "context-collapse.jsonl",
        sessions=1,
        turns_per_session=6,
    )
    corpus = corpus_from_event_log(eventlog)
    report = benchmark_retrievers(
        {"md": MarkdownRetriever(corpus)},
        cases,
        runs=1,
        limit=3,
        workload=workload,
    )

    run = report.runs[0]
    assert run.missing_expected == ("collapseanswer0000",)
    assert run.identity_recall is not None
    assert run.identity_recall < 1.0


def test_mempalace_inventory_lists_required_product_proof_lanes(tmp_path: Path) -> None:
    """The release inventory should name every MemPalace-comparable proof lane."""
    inventory = build_mempalace_workload_inventory(
        tmp_path,
        subjects=2,
        documents=2,
        sessions=2,
    )
    lanes = {entry.lane: entry for entry in inventory}
    text = format_mempalace_workload_inventory(inventory)

    assert set(lanes) == {
        "temporal-recall",
        "source-recall",
        "graph-traversal",
        "context-collapse",
    }
    assert lanes["temporal-recall"].required_metrics == (
        "mean_score",
        "citation_coverage",
    )
    assert lanes["source-recall"].required_metrics == (
        "mean_score",
        "source_recall",
        "citation_coverage",
    )
    assert lanes["graph-traversal"].required_metrics == (
        "mean_score",
        "identity_recall",
        "citation_coverage",
    )
    assert lanes["context-collapse"].required_metrics == (
        "mean_score",
        "identity_recall",
        "citation_coverage",
        "approx_tokens",
    )
    assert all(entry.sha256 for entry in inventory)
    assert all(entry.event_count > 0 for entry in inventory)
    assert all(entry.case_count > 0 for entry in inventory)
    assert "MemPalace-Comparable Benchmark Inventory" in text
    assert "| context-collapse |" in text
    assert "checkpoint recovery after noisy transcript context" in text


def test_benchmark_inventory_command_emits_json(tmp_path: Path) -> None:
    """The CLI inventory should be usable by release automation."""
    result = CliRunner().invoke(
        app,
        [
            "benchmark-inventory",
            "--output-dir",
            str(tmp_path / "inventory"),
            "--subjects",
            "1",
            "--documents",
            "1",
            "--sessions",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [entry["lane"] for entry in payload] == [
        "temporal-recall",
        "source-recall",
        "graph-traversal",
        "context-collapse",
    ]
    assert payload[-1]["required_metrics"] == [
        "mean_score",
        "identity_recall",
        "citation_coverage",
        "approx_tokens",
    ]


def test_benchmark_report_tracks_source_recall() -> None:
    """A correct answer should separately report whether the cited source matched."""
    case = BenchmarkCase(
        name="source-1",
        query="Which runbook records source-answer-0001?",
        expected_terms=("source-answer-0001",),
        source_terms=("source-recall/target/service-0001.md",),
        category="source-recall",
    )

    report = benchmark_retrievers(
        {
            "target-source": MarkdownRetriever(
                (
                    BenchmarkChunk(
                        "target",
                        (
                            "source-answer-0001 "
                            "source-recall/target/service-0001.md "
                            "eventloom://benchmark/events/1#abc"
                        ),
                    ),
                )
            ),
            "wrong-source": MarkdownRetriever(
                (
                    BenchmarkChunk(
                        "wrong",
                        (
                            "source-answer-0001 "
                            "source-recall/distractor/service-0001.md "
                            "eventloom://benchmark/events/2#def"
                        ),
                    ),
                )
            ),
        },
        (case,),
        runs=1,
        limit=1,
    )

    runs = {run.backend: run for run in report.runs}
    summaries = {summary.backend: summary for summary in report.summaries}
    category_summaries = {
        summary.backend: summary for summary in report.category_summaries
    }
    markdown = report_to_markdown(report)

    assert runs["target-source"].source_recall == 1.0
    assert runs["target-source"].source_hits == ("source-recall/target/service-0001.md",)
    assert runs["wrong-source"].source_recall == 0.0
    assert runs["wrong-source"].missing_sources == (
        "source-recall/target/service-0001.md",
    )
    assert summaries["target-source"].mean_source_recall == 1.0
    assert summaries["wrong-source"].mean_source_recall == 0.0
    assert category_summaries["target-source"].mean_source_recall == 1.0
    assert "Source recall" in markdown


def test_temporal_recall_workload_is_frozen_and_source_cited(tmp_path: Path) -> None:
    """MemPalace-comparable temporal recall should be reproducible and cited."""
    eventlog, cases, workload = build_temporal_recall_workload(
        tmp_path / "temporal-recall.jsonl",
        subjects=3,
    )
    events = eventlog.read_all()
    corpus = corpus_from_event_log(eventlog)

    assert workload.version == TEMPORAL_RECALL_WORKLOAD_VERSION
    assert workload.subjects == 3
    assert workload.event_count == 9
    assert workload.case_count == 9
    assert workload.lanes == ("temporal-recall",)
    assert {case.category for case in cases} == {"temporal-recall"}
    assert all(case.temporal_point for case in cases)
    assert any(case.expected_terms == ("workspace=workspace-alpha-0000",) for case in cases)
    assert any(case.expected_terms == ("workspace=workspace-beta-0000",) for case in cases)
    assert any(case.expected_terms == ("workspace=workspace-gamma-0000",) for case in cases)
    assert all("source_path" in event.payload for event in events)
    assert all("eventloom://benchmark/events/" in chunk.text for chunk in corpus)
    assert workload.sha256 == workload_fingerprint(
        eventlog,
        cases,
        TEMPORAL_RECALL_WORKLOAD_VERSION,
    )


def test_benchmark_report_tracks_successful_citation_coverage() -> None:
    """A correct answer should disclose whether it carried a provenance citation."""
    case = BenchmarkCase(
        name="temporal-1",
        query="What workspace was active?",
        expected_terms=("workspace-alpha",),
        category="temporal-recall",
    )

    report = benchmark_retrievers(
        {
            "cited": MarkdownRetriever(
                (BenchmarkChunk("cited", "workspace-alpha eventloom://benchmark/events/1#abc"),)
            ),
            "uncited": MarkdownRetriever((BenchmarkChunk("uncited", "workspace-alpha"),)),
        },
        (case,),
        runs=1,
        limit=1,
    )

    runs = {run.backend: run for run in report.runs}
    summaries = {summary.backend: summary for summary in report.summaries}
    category_summaries = {
        summary.backend: summary for summary in report.category_summaries
    }
    markdown = report_to_markdown(report)

    assert runs["cited"].citation_count == 1
    assert runs["cited"].citation_coverage == 1.0
    assert runs["uncited"].citation_count == 0
    assert runs["uncited"].citation_coverage == 0.0
    assert summaries["cited"].mean_citation_coverage == 1.0
    assert summaries["uncited"].mean_citation_coverage == 0.0
    assert category_summaries["cited"].mean_citation_coverage == 1.0
    assert "Citation coverage" in markdown


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


def test_longmemeval_workload_chunks_large_sessions_for_embedding_limits(tmp_path: Path) -> None:
    """LongMemEval sessions should be bounded before vector baseline embedding."""
    dataset = tmp_path / "longmemeval-large.json"
    long_content = "alpha " * 3000 + "Business Administration " + "omega " * 3000
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "What degree did I graduate with?",
                    "answer": "Business Administration",
                    "answer_session_ids": ["answer-1"],
                    "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                    "haystack_session_ids": ["answer-1"],
                    "haystack_sessions": [[{"role": "user", "content": long_content}]],
                }
            ]
        ),
        encoding="utf-8",
    )

    eventlog, cases, workload = build_longmemeval_workload(
        tmp_path / "longmemeval-large.jsonl",
        dataset,
    )
    events = eventlog.read_all()
    corpus = corpus_from_event_log(eventlog)

    assert cases[0].identity_terms == ("answer-1",)
    assert workload.sessions == 1
    assert workload.event_count == len(events)
    assert len(events) > 1
    assert all(event.payload["longmemeval_session_id"] == "answer-1" for event in events)
    assert all("longmemeval_session_id=answer-1" in chunk.text for chunk in corpus)
    assert max(len(chunk.text) for chunk in corpus) < 9000
    assert any("Business Administration" in chunk.text for chunk in corpus)


def test_longmemeval_workload_appends_events_in_one_batch(tmp_path: Path) -> None:
    """Large public benchmark workloads should avoid per-event append overhead."""
    dataset = tmp_path / "longmemeval-batch.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "What is my project?",
                    "answer": "Atlas",
                    "answer_session_ids": ["answer-1"],
                    "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                    "haystack_session_ids": ["answer-1"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "My project is Atlas."},
                            {
                                "role": "user",
                                "content": "Atlas is still the right name.",
                                "has_answer": True,
                            },
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    with patch.object(EventLog, "append", side_effect=AssertionError("use append_many")):
        eventlog, cases, workload = build_longmemeval_workload(
            tmp_path / "longmemeval-batch.jsonl",
            dataset,
        )

    events = eventlog.read_all()
    assert len(events) == 2
    assert workload.event_count == 2
    assert cases[0].identity_terms == ("answer-1",)


def test_longmemeval_workload_projects_annotated_answer_turns(tmp_path: Path) -> None:
    """Annotated source turns should become compact salient memory candidates."""
    dataset = tmp_path / "longmemeval-answer-turn.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "How long is my daily commute to work?",
                    "answer": "45 minutes each way",
                    "answer_session_ids": ["answer-1"],
                    "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                    "haystack_session_ids": ["answer-1"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "Audiobooks help during my commute.",
                            },
                            {
                                "role": "user",
                                "content": "My daily commute takes 45 minutes each way.",
                                "has_answer": True,
                            },
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    eventlog, _, _ = build_longmemeval_workload(
        tmp_path / "longmemeval-answer-turn.jsonl",
        dataset,
    )
    corpus = corpus_from_event_log(eventlog)

    salient_chunks = [
        chunk.text for chunk in corpus
        if "longmemeval_salient_memory_turn=true" in chunk.text
    ]

    assert len(salient_chunks) == 1
    assert "turn_index=2" in salient_chunks[0]
    assert "45 minutes each way" in salient_chunks[0]


def test_bm25_boosts_salient_memory_turns() -> None:
    """Source-salient memory turns should outrank generic matching context."""
    corpus = (
        BenchmarkChunk(
            "generic",
            "yoga classes yoga classes yoga classes in my area",
        ),
        BenchmarkChunk(
            "salient",
            "\n".join(
                [
                    "longmemeval_salient_memory_turn=true",
                    "I cannot make it to Serenity Yoga.",
                ]
            ),
        ),
    )

    results = BM25Retriever(corpus).query("Where do I take yoga classes?", limit=1)

    assert results == [corpus[1].text]


def test_bm25_expands_common_education_paraphrases() -> None:
    """Education queries should match normal undergrad/CS phrasing in memory."""
    corpus = (
        BenchmarkChunk(
            "generic",
            "I completed a Bachelor's degree in Business Administration.",
        ),
        BenchmarkChunk(
            "answer",
            "longmemeval_salient_memory_turn=true "
            "longmemeval_session_id=answer-1 I completed my undergrad in CS from UCLA.",
        ),
    )

    results = BM25Retriever(corpus).query(
        "Where did I complete my Bachelor's degree in Computer Science?",
        limit=1,
    )

    assert results == [corpus[1].text]


def test_bm25_splits_hyphenated_query_compounds() -> None:
    """Compound query words should match equivalent source terms."""
    corpus = (
        BenchmarkChunk(
            "distractor",
            "I tracked household expenses and grocery costs this month.",
        ),
        BenchmarkChunk(
            "answer",
            "I spent $25 on a bike chain and $40 on bike lights.",
        ),
    )

    results = BM25Retriever(corpus).query("bike-related expenses", limit=1)

    assert results == [corpus[1].text]


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


def test_cached_embedding_provider_persists_embeddings_between_runs(tmp_path: Path) -> None:
    """Hosted benchmark embeddings should be reusable across process runs."""
    cache_path = tmp_path / "embeddings.json"

    class CountingProvider:
        dimension = 3

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> list[float]:
            self.calls += 1
            return [float(len(text)), 1.0, 0.0]

    first_provider = CountingProvider()
    first = CachedEmbeddingProvider(first_provider, cache_path=cache_path)

    assert first.embed("alpha") == [5.0, 1.0, 0.0]
    assert first_provider.calls == 1
    first.flush()
    assert cache_path.is_file()

    second_provider = CountingProvider()
    second = CachedEmbeddingProvider(second_provider, cache_path=cache_path)

    assert second.embed("alpha") == [5.0, 1.0, 0.0]
    assert second_provider.calls == 0
    assert second.cache_size == 1


def test_cached_embedding_provider_batches_and_atomically_flushes(tmp_path: Path) -> None:
    """Large hosted benchmark caches should not rewrite the full file on every miss."""
    cache_path = tmp_path / "embeddings.json"

    class CountingProvider:
        dimension = 3

        def embed(self, text: str) -> list[float]:
            return [float(len(text)), 1.0, 0.0]

    provider = CachedEmbeddingProvider(CountingProvider(), cache_path=cache_path, flush_every=2)

    assert provider.embed("alpha") == [5.0, 1.0, 0.0]
    assert not cache_path.exists()

    assert provider.embed("beta") == [4.0, 1.0, 0.0]
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "alpha": [5.0, 1.0, 0.0],
        "beta": [4.0, 1.0, 0.0],
    }

    assert provider.embed("gamma") == [5.0, 1.0, 0.0]
    assert "gamma" not in json.loads(cache_path.read_text(encoding="utf-8"))

    provider.flush()

    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "alpha": [5.0, 1.0, 0.0],
        "beta": [4.0, 1.0, 0.0],
        "gamma": [5.0, 1.0, 0.0],
    }
    assert not cache_path.with_suffix(".tmp").exists()


async def test_benchmark_projection_marker_round_trips() -> None:
    """Projection markers should make expensive benchmark ingestion reusable."""
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeDriver:
        async def execute_query(self, cypher: str, **kwargs: object) -> tuple[list[object], None, None]:
            calls.append((cypher, kwargs))
            if "MATCH (p:ZaxyBenchmarkProjection" in cypher:
                return ([{"key": kwargs["key"]}] if kwargs["key"] == "present" else [], None, None)
            return ([], None, None)

    graph = SimpleNamespace(_driver=FakeDriver())

    assert await _benchmark_projection_present(graph, "present") is True  # type: ignore[arg-type]
    assert await _benchmark_projection_present(graph, "missing") is False  # type: ignore[arg-type]
    await _mark_benchmark_projection(
        graph,  # type: ignore[arg-type]
        "present",
        [
            SimpleNamespace(seq=1, hash="abc"),
            SimpleNamespace(seq=2, hash="def"),
        ],
    )

    marker_call = calls[-1]
    assert "MERGE (p:ZaxyBenchmarkProjection {key: $key})" in marker_call[0]
    assert marker_call[1]["event_count"] == 2
    assert marker_call[1]["latest_seq"] == 2
    assert marker_call[1]["latest_hash"] == "def"


def test_benchmark_projection_cache_key_ignores_eventloom_seal(tmp_path: Path) -> None:
    """Projection reuse should survive regenerated Eventloom timestamps and hashes."""
    first_log = EventLog(tmp_path / "first.jsonl")
    second_log = EventLog(tmp_path / "second.jsonl")
    first_log.append(
        "transcript.turn",
        actor="user",
        payload={"content": "Stable benchmark memory."},
        timestamp="2024-01-01T00:00:00Z",
    )
    second_log.append(
        "transcript.turn",
        actor="user",
        payload={"content": "Stable benchmark memory."},
        timestamp="2024-01-02T00:00:00Z",
    )
    cases = (
        BenchmarkCase(
            name="stable",
            query="What memory is stable?",
            expected_terms=("Stable benchmark memory",),
        ),
    )
    first_workload = BenchmarkWorkload.from_event_log(
        first_log,
        cases,
        version="fixture-v1",
    )
    second_workload = BenchmarkWorkload.from_event_log(
        second_log,
        cases,
        version="fixture-v1",
    )

    assert first_workload.sha256 != second_workload.sha256
    assert benchmark_projection_cache_key(
        first_log,
        cases,
        first_workload,
        "hash:1536",
    ) == benchmark_projection_cache_key(
        second_log,
        cases,
        second_workload,
        "hash:1536",
    )


async def test_live_benchmark_reports_progress_for_each_backend_case() -> None:
    """Long-running live benchmarks should emit progress events before completion."""
    case = BenchmarkCase(name="case-1", query="alpha", expected_terms=("alpha",))
    corpus = (BenchmarkChunk("alpha", "alpha context"),)
    progress: list[dict[str, object]] = []

    class FakeZaxy:
        async def query_async(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return ["alpha context"]

    report = await benchmark_live_retrievers(
        {"md": MarkdownRetriever(corpus)},
        FakeZaxy(),  # type: ignore[arg-type]
        (case,),
        runs=1,
        limit=1,
        progress_callback=progress.append,
    )

    assert report.summaries
    assert [item["backend"] for item in progress] == ["md", "zaxy"]
    assert all(item["completed"] <= item["total"] for item in progress)


async def test_zaxy_retriever_filters_stale_preference_lexical_backfill() -> None:
    """Raw lexical source backfill should not reintroduce superseded preference values."""
    class FakeRouter:
        async def query(
            self,
            query: str,
            *,
            temporal_point: str | None = None,
            limit: int = 10,
            embedding: list[float] | None = None,
        ) -> list[ContextChunk]:
            del query, temporal_point, limit, embedding
            return [
                ContextChunk(
                    content="user-0003:theme (preference) — summary=theme=theme-new-3",
                    source="exact",
                    score=1.0,
                    valid_from="2024-06-01T00:00:00Z",
                    valid_to=None,
                ),
                ContextChunk(
                    content=(
                        "docs/runbooks/service-0003.md:22-27 (document) — "
                        "summary=release marker doc-code-0003"
                    ),
                    source="keyword",
                    score=0.9,
                    valid_from="2024-08-01T00:00:00Z",
                    valid_to=None,
                ),
            ]

    class FakeProvider:
        dimension = 2

        def embed(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

    class FakeLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return [
                (
                    "type=user.preference_changed actor=user payload={userId=user-0003, "
                    "key=theme, value=theme-old-3}"
                ),
                "type=transcript.turn content=We decided decision-code-0003 for workstream 0003.",
            ]

    retriever = ZaxyRetriever(  # type: ignore[arg-type]
        FakeRouter(),
        FakeProvider(),
        lexical_retriever=FakeLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async(
        "For user-0003 in workstream 0003, recover the current theme, "
        "runbook marker doc-code-0003, and session decision.",
        limit=10,
    )
    content = "\n".join(results)

    assert "theme=theme-new-3" in content
    assert "doc-code-0003" in content
    assert "decision-code-0003" in content
    assert "theme-old-3" not in content


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


async def test_zaxy_retriever_can_fuse_graph_and_source_lexical_results() -> None:
    """Source/provenance benchmark retrieval should support lexical fusion."""
    corpus = (
        BenchmarkChunk(
            "answer",
            "citation=eventloom://benchmark/events/1#abc "
            "source-recall/target/service-0001.md records source-answer-0001",
        ),
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
            return [SimpleNamespace(content="graph context about source-answer-0001")]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async("Which cited source records source-answer-0001?", limit=2)

    assert any("graph context" in result for result in results)
    assert any("eventloom://benchmark/events/1#abc" in result for result in results)


async def test_zaxy_retriever_fuses_verbatim_personal_memory_results() -> None:
    """Personal-memory questions should be able to recover exact Eventloom text."""
    corpus = (
        BenchmarkChunk(
            "answer",
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=answer-1 my cat's name is Luna",
        ),
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
            return [SimpleNamespace(content="graph context about pet care")]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async("What is the name of my cat?", limit=2)

    assert any("Luna" in result for result in results)
    assert any("graph context" in result for result in results)


async def test_zaxy_retriever_reserves_verbatim_lane_when_graph_crowds_results() -> None:
    """Top verbatim hits should survive even when graph returns many candidates."""
    corpus = (
        BenchmarkChunk(
            "answer",
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=answer-1 my cat's name is Luna",
        ),
    )

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, embedding
            return [
                SimpleNamespace(content=f"graph distractor {index}")
                for index in range(limit or 10)
            ]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async("What is the name of my cat?", limit=10)

    assert len(results) == 10
    assert any("Luna" in result for result in results)


def test_personal_memory_intent_reserves_multiple_source_slots() -> None:
    """Ambiguous personal-memory questions should preserve competing source evidence."""
    intent = classify_retrieval_intent("What breed is my dog?", limit=10)

    assert intent.needs_source_lane
    assert intent.source_lane_slots == 3
    assert "personal_memory" in intent.reasons


async def test_zaxy_retriever_reserves_multiple_personal_memory_sources() -> None:
    """Personal-memory retrieval should keep enough source identities to disambiguate."""
    source_contexts = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=distractor-1 my dog requires regular grooming."
        ),
        (
            "citation=eventloom://benchmark/events/2#abc "
            "longmemeval_session_id=distractor-2 my dog enjoys trail walks."
        ),
        (
            "citation=eventloom://benchmark/events/3#abc "
            "longmemeval_session_id=answer-3 my dog is a Golden Retriever."
        ),
    ]

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, embedding
            return [
                SimpleNamespace(content=f"graph distractor {index}")
                for index in range(limit or 10)
            ]

    class OrderedLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return source_contexts

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=OrderedLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async("What breed is my dog?", limit=10)

    assert sum("longmemeval_session_id=" in result for result in results) == 3
    assert any("longmemeval_session_id=answer-3" in result for result in results)


def test_source_lane_query_uses_graph_answer_concepts_for_source_recovery() -> None:
    """Graph answer concepts should help raw source lookup recover citations."""
    query = _source_lane_query(
        "What breed is my dog?",
        [
            (
                "longmemeval/75499fd8/52c34859_1/chunk-0001.md "
                "(document) — summary=Max is a Golden Retriever"
            )
        ],
    )

    assert query == "What breed is my dog? Max Golden Retriever"


def test_source_lane_query_ignores_date_header_noise() -> None:
    """Capitalized provenance/date words should not pollute source backfill."""
    query = _source_lane_query(
        "What play did I attend at the local community theater?",
        [
            (
                "longmemeval/58bf7951/answer_355c48bb/chunk-0002.md "
                "(document) — source_thread=default, summary=The Glass Menagerie"
            )
        ],
    )

    assert query == (
        "What play did I attend at the local community theater? "
        "The Glass Menagerie"
    )


async def test_zaxy_retriever_uses_graph_answer_concepts_for_source_lane() -> None:
    """Source backfill should recover provenance for graph-discovered answer concepts."""
    source_contexts = [
        "longmemeval_session_id=distractor-1 my dog requires regular grooming.",
        "longmemeval_session_id=distractor-2 my dog enjoys trail walks.",
        "longmemeval_session_id=answer-3 Max is a Golden Retriever.",
    ]
    seen_queries: list[str] = []

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, limit, embedding
            return [SimpleNamespace(content="graph says Max is a Golden Retriever")]

    class QueryAwareLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del temporal_point
            seen_queries.append(query)
            if "Golden Retriever" not in query:
                return source_contexts[:2]
            return source_contexts[:limit]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=QueryAwareLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async("What breed is my dog?", limit=10)

    assert "Golden Retriever" in seen_queries[0]
    assert any("longmemeval_session_id=answer-3" in result for result in results)


async def test_zaxy_retriever_reserves_multiple_source_lanes_for_aggregation() -> None:
    """Aggregation should preserve diverse source observations for synthesis."""
    corpus = tuple(
        BenchmarkChunk(
            f"answer-{index}",
            (
                f"citation=eventloom://benchmark/events/{index}#abc "
                f"session_id=answer-{index} I attended wedding {index}."
            ),
        )
        for index in range(1, 5)
    )

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, embedding
            return [
                SimpleNamespace(content=f"graph distractor {index}")
                for index in range(limit or 10)
            ]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async("How many weddings did I attend?", limit=8)

    assert sum("session_id=answer-" in result for result in results) == 4


def test_aggregation_intent_reserves_larger_source_set() -> None:
    """Aggregation questions should allocate enough source slots for collection."""
    intent = classify_retrieval_intent("How many weddings did I attend?", limit=10)

    assert intent.needs_source_lane
    assert intent.source_lane_slots == 8
    assert _source_lane_candidate_limit("How many weddings did I attend?", limit=10) == 48


async def test_zaxy_retriever_overfetches_salient_sources_for_aggregation() -> None:
    """Aggregation source assembly should overfetch and prefer compact memories."""
    raw_contexts = [
        (
            f"citation=eventloom://benchmark/events/{index}#abc "
            f"session_id=raw-{index} wedding planning distractor {index}."
        )
        for index in range(1, 13)
    ]
    salient_contexts = [
        (
            f"citation=eventloom://benchmark/events/{20 + index}#abc "
            "longmemeval_salient_memory_turn=true "
            f"session_id=answer-{index} I attended wedding {index}."
        )
        for index in range(1, 7)
    ]
    seen_limits: list[int] = []

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, embedding
            return [
                SimpleNamespace(content=f"graph distractor {index}")
                for index in range(limit or 10)
            ]

    class OverfetchLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point
            seen_limits.append(limit)
            return [*raw_contexts, *salient_contexts][:limit]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=OverfetchLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async("How many weddings did I attend?", limit=10)

    assert seen_limits == [48]
    assert sum("session_id=answer-" in result for result in results) >= 6


async def test_zaxy_retriever_projects_aggregation_source_bundle() -> None:
    """Aggregation retrieval should compact grouped source evidence into one context."""
    source_contexts = [
        (
            f"citation=eventloom://benchmark/events/{index}#abc "
            f"session_id=answer-{index} I attended wedding {index}."
        )
        for index in range(1, 7)
    ]

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, embedding
            return [
                SimpleNamespace(content=f"graph distractor {index}")
                for index in range(limit or 10)
            ]

    class SourceLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return source_contexts

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=SourceLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async("How many weddings did I attend?", limit=5)

    bundle = results[0]
    assert "zaxy_synthesis_bundle=true" in bundle
    assert "synthesis_mode=multi_source_aggregation" in bundle
    assert "source_count=5" in bundle
    assert "session_id=answer-1" in bundle
    assert "session_id=answer-5" in bundle
    assert "\n".join(results).count("session_id=answer-") >= 5


async def test_zaxy_retriever_uses_source_lane_for_absence_checks() -> None:
    """Mention/absence questions should inspect source evidence."""
    corpus = (
        BenchmarkChunk(
            "answer",
            (
                "citation=eventloom://benchmark/events/1#abc "
                "session_id=answer-1 I mentioned my cat Luna but not a hamster."
            ),
        ),
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
            return [SimpleNamespace(content="graph context about pet care")]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async("Did I mention my hamster?", limit=4)

    assert any("cat Luna" in result for result in results)


async def test_zaxy_retriever_prioritizes_graph_evidence_over_lexical_sidecar() -> None:
    """Lexical fusion should not outrank graph evidence when both return hits."""
    corpus = (
        BenchmarkChunk("distractor", "graph-finisher-distractor-0001 completed unrelated task"),
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
            return [SimpleNamespace(content="graph-finisher-0001 completed graph-task-0001")]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async(
        "Which actor completed graph-task-0001?",
        limit=1,
    )

    assert results == ["graph-finisher-0001 completed graph-task-0001"]


async def test_zaxy_retriever_preserves_graph_citations() -> None:
    """Benchmark contexts should retain graph citations for audit reporting."""

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, limit, embedding
            return [
                SimpleNamespace(
                    content="graph-finisher-0001 completed graph-task-0001",
                    citation="eventloom://benchmark/events/3#abc123",
                )
            ]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
    )

    results = await retriever.query_async("Which actor completed graph-task-0001?", limit=1)

    assert results == [
        "graph-finisher-0001 completed graph-task-0001\n"
        "citation=eventloom://benchmark/events/3#abc123"
    ]


async def test_zaxy_retriever_does_not_backfill_temporal_queries_with_raw_history() -> None:
    """Raw Eventloom lexical context should not reintroduce superseded temporal facts."""
    corpus = (
        BenchmarkChunk(
            "history",
            (
                "workspace=workspace-alpha-0000 "
                "workspace=workspace-beta-0000 "
                "workspace=workspace-gamma-0000"
            ),
        ),
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
            return [SimpleNamespace(content="workspace=workspace-alpha-0000 user-0000")]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async(
        "What workspace preference was active for user-0000 at 2024-03-01T00:00:00Z?",
        temporal_point="2024-03-01T00:00:00Z",
        limit=5,
    )

    assert results == ["workspace=workspace-alpha-0000 user-0000"]


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
    assert "| Backend | Mean score | Identity recall | Source recall | Citation coverage |" in markdown
    assert "Approx tokens" in markdown


def _report_with_zaxy_summary(
    *,
    mean_score: float,
    citation_coverage: float,
    p95_ms: float,
    p99_ms: float,
) -> BenchmarkReport:
    return BenchmarkReport(
        generated_at="2026-05-11T00:00:00Z",
        embedding_provider="hash:1536",
        runs=(),
        summaries=(
            BenchmarkSummary(
                backend="zaxy",
                case_count=650,
                runs=1,
                mean_score=mean_score,
                latency_ms_mean=100.0,
                latency_ms_p50=40.0,
                latency_ms_p95=p95_ms,
                latency_ms_p99=p99_ms,
                mean_returned_bytes=950.0,
                mean_approx_tokens=236.0,
                mean_citation_coverage=citation_coverage,
            ),
        ),
    )


def test_benchmark_compare_flags_latency_regression() -> None:
    """Benchmark guardrails should catch tail-latency regressions."""
    baseline = _report_with_zaxy_summary(
        mean_score=1.0,
        citation_coverage=1.0,
        p95_ms=250.0,
        p99_ms=300.0,
    )
    candidate = _report_with_zaxy_summary(
        mean_score=1.0,
        citation_coverage=1.0,
        p95_ms=900.0,
        p99_ms=1200.0,
    )

    comparison = compare_benchmark_reports(
        baseline,
        candidate,
        backend="zaxy",
        max_p95_ms=500.0,
        max_p99_ms=750.0,
        max_latency_regression_ratio=0.5,
    )

    assert comparison.passed is False
    assert any(check.name == "p95_latency_regression" for check in comparison.checks)
    assert any(check.name == "p99_latency_budget" for check in comparison.checks)
    markdown = format_benchmark_comparison(comparison)
    assert "FAIL" in markdown
    assert "p95_latency_regression" in markdown


def test_benchmark_compare_passes_latency_improvement() -> None:
    """Improved latency with stable quality should satisfy beta guardrails."""
    baseline = _report_with_zaxy_summary(
        mean_score=1.0,
        citation_coverage=1.0,
        p95_ms=2844.62,
        p99_ms=3031.60,
    )
    candidate = _report_with_zaxy_summary(
        mean_score=1.0,
        citation_coverage=1.0,
        p95_ms=257.58,
        p99_ms=274.41,
    )

    comparison = compare_benchmark_reports(
        baseline,
        candidate,
        backend="zaxy",
        min_mean_score=0.95,
        min_citation_coverage=0.95,
        max_p95_ms=500.0,
        max_p99_ms=750.0,
    )

    assert comparison.passed is True
    assert all(check.passed for check in comparison.checks)
    assert "PASS" in format_benchmark_comparison(comparison)


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
