"""Tests for live retrieval benchmark runners."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from zaxy.__main__ import _parse_benchmark_baselines, app
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
    EvidenceFrontierRetriever,
    MarkdownRetriever,
    MarkdownVectorRetriever,
    RankFusionRetriever,
    VectorRetriever,
    ZaxyCheckoutRetriever,
    ZaxyRetriever,
    _benchmark_contexts_from_checkout,
    _benchmark_projection_present,
    _build_source_lane_retriever,
    _mark_benchmark_projection,
    _reset_benchmark_graph,
    _scope_augmented_source_query,
    _scoped_fetch_limit,
    _scoped_source_fallback_limit,
    _scoped_source_fetch_limit,
    benchmark_case_scope_terms,
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
    summarize_miss_taxonomy,
    workload_fingerprint,
    write_benchmark_report,
)
from zaxy.query import ContextChunk
from zaxy.retrieval_intent import classify_retrieval_intent
from zaxy.retrieval_plan import (
    absence_check_bundle,
    bridge_source_lane_queries,
    source_evidence_score,
    source_lane_candidate_limit,
    source_lane_queries,
    source_lane_query,
    source_synthesis_bundle,
    source_synthesis_candidate_limit,
)


def test_cli_exposes_live_benchmark_command() -> None:
    """The public CLI should expose a reproducible live benchmark command."""
    cli = Path("src/zaxy/__main__.py").read_text(encoding="utf-8")
    script = Path("scripts/live-benchmark.sh").read_text(encoding="utf-8")

    assert "def benchmark(" in cli
    assert "def benchmark_compare(" in cli
    assert "compare_benchmark_reports" in cli
    assert "embedding_provider: str = typer.Option(" in cli
    assert "local-http" in cli
    assert "sentence-transformers" in cli
    assert "build_live_zaxy_retriever" in cli
    assert "build_statistical_event_log" in cli
    assert "build_frozen_statistical_workload" in cli
    assert "build_benchmark_suite_workload" in cli
    assert "build_consolidation_collapse_workload" in cli
    assert "build_context_collapse_workload" in cli
    assert "build_graph_traversal_workload" in cli
    assert "build_mempalace_workload_inventory" in cli
    assert "build_longmemeval_workload" in cli
    assert "--zaxy-backend" in cli
    assert "checkout_retriever=checkout_retriever" in cli
    assert "build_source_recall_workload" in cli
    assert "build_temporal_recall_workload" in cli
    assert "benchmark-inventory" in cli
    assert "lexical_retriever=_build_source_lane_retriever(corpus, provider)" in cli
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


def test_benchmark_checkout_contexts_prioritize_synthesis_evidence() -> None:
    """Answer-bearing synthesis evidence should stay inside Answer@5."""
    checkout = SimpleNamespace(
        quality={"answerability": "answer_from_memory", "confidence": 0.9},
        diagnostics={},
        current_facts=[
            {
                "content": f"ordinary fact {index}",
                "citation": f"eventloom://benchmark/events/{index}#hash",
                "source_lane": "verbatim",
                "score": 1.0,
            }
            for index in range(1, 6)
        ],
        evidence=[
            {
                "content": (
                    "zaxy_synthesis_bundle=true\n"
                    "issue_candidate=GPS system not functioning correctly"
                ),
                "citation": "eventloom://benchmark/events/99#hash",
                "source_lane": "verbatim",
                "score": 1.2,
            }
        ],
    )

    contexts = _benchmark_contexts_from_checkout(checkout)

    assert "issue_candidate=GPS system not functioning correctly" in contexts[0]
    assert "checkout_item=evidence" in contexts[0]


def test_parse_benchmark_baselines_allows_zaxy_only_runs() -> None:
    """Operators should be able to skip baselines once release floors are established."""
    assert _parse_benchmark_baselines("none", allow_centroid=False) == ()


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


def test_benchmark_report_tracks_recall_at_k() -> None:
    """LongMemEval-style reports should expose Recall@K source recovery."""
    case = BenchmarkCase(
        name="identity-1",
        query="What source contains my trip memory?",
        expected_terms=("trip answer",),
        identity_terms=("answer-session-1",),
        category="longmemeval:multi-session",
    )

    class OrderedRetriever:
        def __init__(self, contexts: tuple[str, ...]) -> None:
            self._contexts = contexts

        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point
            return list(self._contexts[:limit])

    report = benchmark_retrievers(
        {
            "late-hit": OrderedRetriever(
                tuple(
                    f"distractor {index}"
                    for index in range(4)
                )
                + (
                    "answer-session-1 contains the trip answer",
                )
            ),
            "miss": OrderedRetriever(("unrelated trip answer",)),
        },
        (case,),
        runs=1,
        limit=5,
    )

    runs = {run.backend: run for run in report.runs}
    summaries = {summary.backend: summary for summary in report.summaries}
    markdown = report_to_markdown(report)

    assert runs["late-hit"].recall_at_1 == 0.0
    assert runs["late-hit"].recall_at_5 == 1.0
    assert runs["late-hit"].recall_at_10 == 1.0
    assert runs["miss"].recall_at_5 == 0.0
    assert summaries["late-hit"].mean_recall_at_5 == 1.0
    assert summaries["miss"].mean_recall_at_5 == 0.0
    assert "Recall@5" in markdown


def test_benchmark_report_separates_answer_and_evidence_recall() -> None:
    """Answer presence and source identity presence should be visible separately."""
    case = BenchmarkCase(
        name="answered-without-source-id",
        query="Which music service did I mention?",
        expected_terms=("Spotify",),
        identity_terms=("answer_session_1",),
    )

    class AnswerOnlyRetriever:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return ["I mentioned Spotify as my music streaming service."]

    report = benchmark_retrievers(
        {"answer-only": AnswerOnlyRetriever()},
        (case,),
        runs=1,
        limit=5,
    )

    run = report.runs[0]
    assert run.score == 1.0
    assert run.identity_recall == 0.0
    assert run.recall_at_5 == 0.0
    assert run.answer_recall_at_5 == 1.0
    assert run.miss_category == "projection_miss"
    assert report.miss_taxonomy == {
        "answer-only": {
            "projection_miss": 1,
        }
    }


def test_benchmark_miss_taxonomy_classifies_ranking_and_synthesis_misses() -> None:
    """Run-level miss categories should identify the likely failure layer."""
    answer_case = BenchmarkCase(
        name="answer-case",
        query="Which service?",
        expected_terms=("Spotify",),
        identity_terms=("answer-session",),
    )
    source_case = BenchmarkCase(
        name="source-case",
        query="Which source?",
        expected_terms=("Spotify",),
        identity_terms=("answer-session",),
    )

    class MixedRetriever:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del temporal_point, limit
            if "service" in query.casefold():
                return [
                    "distractor service note 1",
                    "distractor service note 2",
                    "distractor service note 3",
                    "distractor service note 4",
                    "distractor service note 5",
                    "answer-session says Spotify",
                ]
            return ["answer-session contains no answer text"]

    report = benchmark_retrievers(
        {"mixed": MixedRetriever()},
        (answer_case, source_case),
        runs=1,
        limit=10,
    )

    runs = {run.case_name: run for run in report.runs}
    assert runs["answer-case"].miss_category == "ranking_miss", runs["answer-case"]
    assert runs["source-case"].miss_category == "synthesis_miss", runs["source-case"]
    assert summarize_miss_taxonomy(report.runs) == {
        "mixed": {
            "ranking_miss": 1,
            "synthesis_miss": 1,
        }
    }
    markdown = report_to_markdown(report)
    assert "## Miss taxonomy" in markdown
    assert "ranking_miss" in markdown
    assert "synthesis_miss" in markdown


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


def test_longmemeval_workload_fingerprint_ignores_transient_eventloom_seal(
    tmp_path: Path,
) -> None:
    """LongMemEval workload identity should be stable across regenerated logs."""
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
                    "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                    "haystack_session_ids": ["answer-1"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "I graduated with a Business Administration degree.",
                            }
                        ],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    first_log, first_cases, first_workload = build_longmemeval_workload(
        tmp_path / "first.jsonl",
        dataset,
        questions=1,
    )
    second_log, second_cases, second_workload = build_longmemeval_workload(
        tmp_path / "second.jsonl",
        dataset,
        questions=1,
    )

    assert first_workload.sha256 == second_workload.sha256
    assert workload_fingerprint(first_log, first_cases, LONGMEMEVAL_WORKLOAD_VERSION) == (
        workload_fingerprint(second_log, second_cases, LONGMEMEVAL_WORKLOAD_VERSION)
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


async def test_reset_benchmark_graph_deletes_in_relationship_and_node_batches() -> None:
    """Benchmark reset should not delete a large graph in one transaction."""
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeDriver:
        def __init__(self) -> None:
            self.relationship_counts = [1000, 500, 0]
            self.node_counts = [1000, 1, 0]

        async def execute_query(self, cypher: str, **kwargs: object) -> tuple[list[dict[str, int]], None, None]:
            calls.append((cypher, kwargs))
            if "MATCH ()-[r]->()" in cypher:
                return ([{"deleted": self.relationship_counts.pop(0)}], None, None)
            if "MATCH (n)" in cypher:
                return ([{"deleted": self.node_counts.pop(0)}], None, None)
            raise AssertionError(f"unexpected reset query: {cypher}")

    totals = await _reset_benchmark_graph(
        SimpleNamespace(_driver=FakeDriver()),  # type: ignore[arg-type]
        batch_size=1000,
    )

    assert totals == {"relationships": 1500, "nodes": 1001}
    assert len(calls) == 6
    assert all(call[1]["batch_size"] == 1000 for call in calls)
    assert "MATCH (n) DETACH DELETE n" not in "\n".join(call[0] for call in calls)


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

    assert first_workload.sha256 == second_workload.sha256
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


async def test_live_benchmark_can_include_memory_checkout_backend() -> None:
    """Benchmarks should be able to exercise the production Memory Checkout path."""
    case = BenchmarkCase(
        name="checkout-aggregation",
        query="How many weddings did I attend?",
        expected_terms=("answerability=answer_from_memory", "evidence_plan_mode=multi_source_aggregation"),
        identity_terms=("answer-1", "answer-2"),
    )
    progress: list[dict[str, object]] = []

    class FakeZaxy:
        async def query_async(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return ["plain zaxy context"]

    class FakeCheckout:
        async def query_async(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return [
                "memory_checkout=true\n"
                "answerability=answer_from_memory\n"
                "evidence_plan_mode=multi_source_aggregation\n"
                "source_id=answer-1\n"
                "source_id=answer-2"
            ]

    report = await benchmark_live_retrievers(
        {},
        FakeZaxy(),  # type: ignore[arg-type]
        (case,),
        runs=1,
        limit=1,
        checkout_retriever=FakeCheckout(),  # type: ignore[arg-type]
        progress_callback=progress.append,
    )

    by_backend = {run.backend: run for run in report.runs}
    assert by_backend["zaxy-checkout"].score == 1.0
    assert by_backend["zaxy-checkout"].identity_recall == 1.0
    assert [item["backend"] for item in progress] == ["zaxy", "zaxy-checkout"]


async def test_live_benchmark_can_run_checkout_backend_without_graph_backend() -> None:
    """Checkout-only mode should make the benchmark target explicit."""
    case = BenchmarkCase(
        name="checkout-only",
        query="What should I use?",
        expected_terms=("memory_checkout=true",),
    )

    class FakeZaxy:
        async def query_async(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return ["graph"]

    class FakeCheckout:
        async def query_async(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point, limit
            return ["memory_checkout=true"]

    report = await benchmark_live_retrievers(
        {},
        FakeZaxy(),  # type: ignore[arg-type]
        (case,),
        runs=1,
        checkout_retriever=FakeCheckout(),  # type: ignore[arg-type]
        include_zaxy=False,
    )

    assert [run.backend for run in report.runs] == ["zaxy-checkout"]


async def test_zaxy_checkout_retriever_returns_checkout_contract() -> None:
    """The checkout benchmark backend should expose model-facing diagnostics."""
    corpus = (
        BenchmarkChunk(
            "answer-1",
            (
                "citation=eventloom://benchmark/events/1#abc "
                "session_id=answer-1 I attended Rachel and Mike's wedding."
            ),
        ),
        BenchmarkChunk(
            "answer-2",
            (
                "citation=eventloom://benchmark/events/2#abc "
                "session_id=answer-2 I attended Emily and Sarah's wedding."
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
            return [SimpleNamespace(content="uncited graph summary", source="keyword", score=0.99)]

    retriever = ZaxyCheckoutRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
    )

    results = await retriever.query_async("How many weddings did I attend?", limit=5)
    output = "\n".join(results)

    assert results[0].startswith("memory_checkout_compact=true")
    assert "memory_checkout=true" in output
    assert "answerability=answer_from_memory" in output
    assert "evidence_plan_mode=multi_source_aggregation" in output
    assert "evidence_plan_satisfied=True" in output
    assert "source_id=answer-1" in output
    assert "source_id=answer-2" in output
    assert sum(len(result) for result in results) < 3000


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


async def test_zaxy_retriever_ranks_personal_source_lane_before_graph_context() -> None:
    """Personal memory checkout should surface cited source text in top slots."""
    corpus = (
        BenchmarkChunk(
            "answer",
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer-1 my cat's name is Luna"
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

    results = await retriever.query_async("What is the name of my cat?", limit=5)

    assert "longmemeval_session_id=answer-1" in results[0]


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
    query = source_lane_query(
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
    query = source_lane_query(
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


def test_source_lane_queries_expand_aggregation_event_actions() -> None:
    """Aggregation source lookup should search likely memory phrasings, not just query wording."""
    queries = source_lane_queries(
        "How many model kits have I worked on or bought?",
        [],
    )

    assert queries == (
        "How many model kits have I worked on or bought?",
        "model kits finished started picked up got bought scale",
    )


def test_source_lane_queries_expand_instrument_ownership_actions() -> None:
    """Instrument aggregation should search owned instrument brands and families."""
    queries = source_lane_queries(
        "How many musical instruments do I currently own?",
        [],
    )

    assert queries == (
        "How many musical instruments do I currently own?",
        "musical instruments guitar piano drum set acoustic electric korg yamaha fender pearl owned had playing",
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

    assert seen_queries[0] == "What breed is my dog?"
    assert any("Golden Retriever" in query for query in seen_queries[1:])
    assert any("longmemeval_session_id=answer-3" in result for result in results)


async def test_zaxy_retriever_merges_expanded_source_queries_before_truncation() -> None:
    """Expanded aggregation recall should not be starved by the original query results."""
    original_contexts = [
        f"longmemeval_session_id=distractor-{index} model kit discussion {index}"
        for index in range(1, 9)
    ]
    expanded_contexts = [
        "longmemeval_session_id=answer-1 I finished a simple Revell F-15 Eagle kit.",
        "longmemeval_session_id=answer-2 I started a Tamiya 1/48 scale Spitfire Mk.V.",
    ]

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, limit, embedding
            return [SimpleNamespace(content="graph distractor")]

    class QueryAwareLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del temporal_point
            if "finished started picked up" in query:
                return expanded_contexts[:limit]
            return original_contexts[:limit]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=QueryAwareLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async(
        "How many model kits have I worked on or bought?",
        limit=5,
    )

    assert any("answer-1" in result for result in results)
    assert any("answer-2" in result for result in results)


async def test_zaxy_retriever_uses_graph_evidence_for_temporal_synthesis() -> None:
    """Synthesis should use graph-retrieved cited turns as well as lexical backfill."""
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
                    content=(
                        "longmemeval_session_id=answer-1 "
                        "I got my new binoculars exactly three weeks ago."
                    )
                )
            ]

    class QueryAwareLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del query, temporal_point
            return [
                (
                    "longmemeval_session_id=answer-2 I saw the American goldfinches "
                    "returning to the area a week ago."
                ),
                (
                    "longmemeval_session_id=answer-1 I waited months for the "
                    "binoculars to arrive."
                ),
            ][:limit]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=QueryAwareLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async(
        "How long did I use my new binoculars before I saw the American goldfinches returning to the area?",
        limit=5,
    )

    assert "week_interval=2 weeks" in results[0]
    assert "week_interval_answer=Two weeks" in results[0]


def test_absence_check_ignores_direct_fact_question_words() -> None:
    """Absence guidance should not fire when direct fact terms are present in evidence."""
    bundle = absence_check_bundle(
        query="What brand are my favorite running shoes?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer Nike has been my favourite brand "
                "so far for running shoes."
            )
        ],
        limit=5,
    )

    assert bundle is None


def test_source_synthesis_bundle_requires_typed_evidence_for_aggregation() -> None:
    """Unsupported aggregation should fall through to absence guidance instead of hallucinating."""
    contexts = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=answer I spent two weeks traveling solo around Japan."
        )
    ]

    bundle = source_synthesis_bundle(
        query="How long was I in Korea for?",
        source_results=contexts,
        limit=5,
    )

    assert bundle is None
    absence = absence_check_bundle(
        query="How long was I in Korea for?",
        source_results=contexts,
        limit=5,
    )
    assert absence is not None
    assert "not_mentioned_candidate=korea" in absence
    assert "Japan" in absence


def test_source_synthesis_bundle_defers_to_absence_when_comparison_target_is_missing() -> None:
    """Alternative questions should not synthesize an answer when one side lacks user evidence."""
    contexts = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=answer-1 "
            "user: I just fixed the broken fence on the east side of my property three weeks ago."
        ),
        (
            "citation=eventloom://benchmark/events/2#abc "
            "longmemeval_session_id=answer-2 "
            "assistant: Peter can advise you about dairy cows, but no purchase was mentioned."
        ),
    ]

    bundle = source_synthesis_bundle(
        query="Which task did I complete first, fixing the fence or purchasing three cows from Peter?",
        source_results=contexts,
        limit=5,
    )
    absence = absence_check_bundle(
        query="Which task did I complete first, fixing the fence or purchasing three cows from Peter?",
        source_results=contexts,
        limit=5,
    )

    assert bundle is None
    assert absence is not None
    assert "not_mentioned_candidate=purchasing three cows peter" in absence
    assert "fixing the fence" in absence


def test_source_synthesis_bundle_defers_to_absence_when_temporal_anchor_is_missing() -> None:
    """Temporal calculations should not proceed when an explicit event anchor is absent."""
    contexts = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=answer-1 "
            "user: I attended the annual Holiday Market at the local mall a week before Black Friday."
        ),
        (
            "citation=eventloom://benchmark/events/2#abc "
            "longmemeval_session_id=answer-2 "
            "user: I got the iPhone 13 Pro for my sister's birthday."
        ),
    ]

    bundle = source_synthesis_bundle(
        query="How many days before I bought my iPad did I attend the Holiday Market?",
        source_results=contexts,
        limit=5,
    )
    absence = absence_check_bundle(
        query="How many days before I bought my iPad did I attend the Holiday Market?",
        source_results=contexts,
        limit=5,
    )

    assert bundle is None
    assert absence is not None
    assert "not_mentioned_candidate=bought ipad" in absence
    assert "holiday market" in absence.casefold()


def test_source_evidence_score_detects_single_property_evidence() -> None:
    """Source ordering should prefer single cited evidence rows before aggregation is complete."""
    query = "How many properties did I view before making an offer on the townhouse in Brookside?"
    evidence = (
        "longmemeval_session_id=answer-1 "
        "user: I viewed a 1-bedroom condo on February 10th, "
        "but the noise from the highway was a deal-breaker."
    )
    topical_non_evidence = (
        "longmemeval_session_id=answer-2 "
        "assistant: Research helps you understand condo fees, taxes, insurance, "
        "and other expenses before buying a property."
    )

    assert source_evidence_score(query, evidence) > source_evidence_score(
        query,
        topical_non_evidence,
    )


def test_absence_check_bundle_preserves_model_identifier_targets() -> None:
    """Missing alternative targets should keep numeric and single-letter model identifiers."""
    absence = absence_check_bundle(
        query="Which project did I start first, the Ferrari model or the Porsche 991 Turbo S model?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer-1 "
                "user: I started the Ferrari F40 model kit last month."
            ),
            (
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=answer-2 "
                "assistant: The Porsche 911 Turbo is an interesting sports car, but no project was mentioned."
            ),
        ],
        limit=5,
    )

    assert absence is not None
    assert "not_mentioned_candidate=porsche 991 turbo s model" in absence


def test_source_synthesis_bundle_projects_last_week_relative_interval() -> None:
    """Relative-time synthesis should derive elapsed intervals instead of summing anchors."""
    bundle = source_synthesis_bundle(
        query="How long had I been a member of Book Lovers Unite when I attended the meetup?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=joined "
                "user: I joined Book Lovers Unite three weeks ago."
            ),
            (
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=meetup "
                "user: I attended a meetup organized by Book Lovers Unite last week."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "week_values=" in bundle
    assert "week_interval_answer=Two weeks" in bundle


def test_source_synthesis_bundle_preserves_cross_source_relative_week_evidence() -> None:
    """Relative-time filtering should keep both event anchors across sources."""
    bundle = source_synthesis_bundle(
        query="How long did I use my new binoculars before I saw the American goldfinches returning to the area?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=goldfinches "
                "user: I've been listening to bird calls online for about a month. "
                "I noticed the American goldfinches returning to the area a week ago."
            ),
            (
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=goldfinches "
                "user: The American goldfinches returning to the area a week ago "
                "made bird identification practice more exciting."
            ),
            (
                "citation=eventloom://benchmark/events/3#abc "
                "longmemeval_session_id=binoculars "
                "user: The binoculars arrived exactly three weeks ago."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "week_interval_answer=Two weeks" in bundle


def test_source_synthesis_bundle_keeps_temporal_synthesis_for_generic_missing_words() -> None:
    """Generic query words should not trigger absence routing when temporal evidence is present."""
    bundle = source_synthesis_bundle(
        query="How many days before Rachel's party did I find the house I loved?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer-1 "
                "user: On 2026/03/01 I found the perfect house for the party."
            ),
            (
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=answer-2 "
                "user: Rachel's party happened on 2026/03/15."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "zaxy_absence_check=true" not in bundle
    assert "date_interval_answer=14 days. 15 days (including the last day) is also acceptable." in bundle


def test_absence_check_bundle_uses_missing_concrete_action_target() -> None:
    """Direct absence questions should cite nearby memories without inventing absent actions."""
    absence = absence_check_bundle(
        query="How many days before I bought my iPad did I visit Seattle?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer-1 "
                "user: I visited Seattle on 2026/02/01 and had dinner downtown."
            ),
            (
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=answer-2 "
                "assistant: I can help compare tablet models, but no purchase was mentioned."
            ),
        ],
        limit=5,
    )

    assert absence is not None
    assert "not_mentioned_candidate=bought ipad" in absence
    assert "visit seattle" in absence


def test_absence_check_bundle_does_not_route_generic_count_gaps() -> None:
    """Count/list synthesis should not become absence just because generic query words are absent."""
    absence = absence_check_bundle(
        query="How many musical instruments do I currently own?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer-1 "
                "user: I've been playing my Fender Stratocaster electric guitar for years."
            ),
            (
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=answer-2 "
                "user: I'm thinking of selling my old Pearl Export drum set."
            ),
        ],
        limit=5,
    )

    assert absence is None


def test_source_synthesis_bundle_keeps_model_kit_count_evidence_out_of_absence() -> None:
    """Countable aggregation evidence should not be replaced by absence guidance."""
    bundle = source_synthesis_bundle(
        query="How many model kits have I worked on or bought?",
        source_results=[
            "session_id=answer-1 I recently finished a simple Revell F-15 Eagle kit.",
            "session_id=answer-2 I finished a Tamiya 1/48 scale Spitfire Mk.V.",
            "session_id=answer-3 I started working on a 1/16 scale German Tiger I tank.",
            "session_id=answer-4 I just got this kit and a 1/24 scale '69 Camaro at a model show.",
        ],
        limit=5,
    )

    assert bundle is not None
    assert "zaxy_synthesis_bundle=true" in bundle
    assert "zaxy_absence_check=true" not in bundle
    assert "count_answer=4" in bundle


def test_absence_check_bundle_does_not_route_missing_generic_temporal_words() -> None:
    """Temporal source recall needs more evidence, not a false absence answer."""
    absence = absence_check_bundle(
        query="Which streaming service did I start using most recently?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer-1 "
                "user: I started using Hulu a few months ago."
            ),
            (
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=answer-2 "
                "user: I started using Apple TV+ last month."
            ),
        ],
        limit=5,
    )

    assert absence is None


def test_source_synthesis_bundle_treats_got_gift_as_bought_gift_evidence() -> None:
    """Concrete absence routing should not reject equivalent first-person gift wording."""
    contexts = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=graduation "
            "longmemeval_session_date=2023/03/29 (Wed) "
            "user: I recently got a wireless headphone for my brother as a graduation gift on the 3/8."
        ),
        (
            "citation=eventloom://benchmark/events/2#abc "
            "longmemeval_session_id=birthday "
            "longmemeval_session_date=2023/03/29 (Wed) "
            "user: I bought a birthday gift for my best friend on 3/15."
        ),
    ]

    absence = absence_check_bundle(
        query=(
            "How many days had passed between the day I bought a gift for my brother's "
            "graduation ceremony and the day I bought a birthday gift for my best friend?"
        ),
        source_results=contexts,
        limit=5,
    )
    bundle = source_synthesis_bundle(
        query=(
            "How many days had passed between the day I bought a gift for my brother's "
            "graduation ceremony and the day I bought a birthday gift for my best friend?"
        ),
        source_results=contexts,
        limit=5,
    )

    assert absence is None
    assert bundle is not None
    assert "date_interval_days=7" in bundle


def test_source_synthesis_bundle_projects_direct_attribute_answers() -> None:
    """Direct facts should expose a compact answer when source wording is paraphrastic."""
    bundle = source_synthesis_bundle(
        query="What is my ethnicity?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer "
                "I've been thinking about my mixed ethnicity - Irish and Italian - lately."
            )
        ],
        limit=5,
    )

    assert bundle is not None
    assert "direct_fact_type=attribute" in bundle
    assert "direct_fact_attribute=ethnicity" in bundle
    assert "direct_answer=A mix of Irish and Italian" in bundle


def test_source_synthesis_bundle_prefers_typed_evidence_within_source_groups() -> None:
    """Aggregation synthesis should choose evidence-bearing snippets per source group."""
    bundle = source_synthesis_bundle(
        query="How much total money have I spent on bike-related expenses since the start of the year?",
        source_results=[
            (
                "longmemeval/gpt4_d84a3211/answer_1/chunk-0001.md "
                "longmemeval_session_id=answer_1 bike trails and route planning without expenses."
            ),
            *[
                (
                    f"longmemeval/gpt4_d84a3211/distractor_{index}/chunk-0001.md "
                    f"longmemeval_session_id=distractor_{index} bike route planning context."
                )
                for index in range(1, 15)
            ],
            (
                "longmemeval/gpt4_d84a3211/answer_1/salient-turn.md "
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer_1 I bought my Bell Zephyr bike helmet for $120."
            ),
            (
                "longmemeval/gpt4_d84a3211/answer_2/salient-turn.md "
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=answer_2 I replaced my bike chain, which cost $25."
            ),
            (
                "longmemeval/gpt4_d84a3211/answer_3/salient-turn.md "
                "citation=eventloom://benchmark/events/3#abc "
                "longmemeval_session_id=answer_3 I got bike lights installed for $40."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "currency_total_answer=$185" in bundle


def test_source_synthesis_bundle_derives_age_at_event_from_elapsed_years() -> None:
    """Age-at-event queries should expose deterministic cited arithmetic."""
    bundle = source_synthesis_bundle(
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
    assert "age_current=32" in bundle
    assert "age_elapsed_years=5" in bundle
    assert "age_at_event_operation=32-5" in bundle
    assert "age_at_event_answer=27" in bundle


def test_source_synthesis_bundle_derives_prior_work_duration_from_current_role_tenure() -> None:
    """Career-duration queries should subtract current role tenure from total experience."""
    bundle = source_synthesis_bundle(
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
    assert "career_total_months=108" in bundle
    assert "career_current_role_months=51" in bundle
    assert "career_prior_duration_answer=4 years and 9 months" in bundle


async def test_zaxy_retriever_preserves_original_source_query_when_graph_concepts_are_noisy() -> None:
    """Graph-expanded source lookup should supplement, not replace, original recall."""
    source_contexts = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=distractor play directing next month"
        ),
        (
            "citation=eventloom://benchmark/events/2#abc "
            "longmemeval_session_id=answer_355c48bb "
            "I attended The Glass Menagerie at the local community theater."
        ),
        (
            "citation=eventloom://benchmark/events/3#abc "
            "longmemeval_session_id=noisy Provenance Story Netflix Adding"
        ),
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
            return [
                SimpleNamespace(content="graph says Provenance Story Netflix Adding")
            ]

    class QueryAwareLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del temporal_point
            seen_queries.append(query)
            if "Provenance Story Netflix Adding" in query:
                return [source_contexts[2], source_contexts[0]][:limit]
            return source_contexts[:limit]

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=QueryAwareLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async(
        "What play did I attend at the local community theater?",
        limit=5,
    )

    assert seen_queries[0] == "What play did I attend at the local community theater?"
    assert any("The Glass Menagerie" in result for result in results)


def test_source_lane_results_preserve_literal_personal_recall_before_expansion() -> None:
    """Literal personal-memory evidence should not be evicted by expanded-query distractors."""
    query = "What play did I attend at the local community theater?"
    graph_results = ["graph says Provenance Story Netflix Adding"]
    primary_answer = (
        "citation=eventloom://benchmark/events/2#abc "
        "longmemeval_session_id=answer_355c48bb "
        "I attended The Glass Menagerie at the local community theater."
    )

    class QueryAwareLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del temporal_point
            if "Provenance Story Netflix Adding" in query:
                return [
                    f"citation=eventloom://benchmark/events/{index}#abc expanded distractor {index}"
                    for index in range(10, 10 + limit)
                ]
            return [
                "citation=eventloom://benchmark/events/1#abc topical theater distractor",
                primary_answer,
            ][:limit]

    retriever = ZaxyRetriever(
        SimpleNamespace(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=QueryAwareLexical(),  # type: ignore[arg-type]
    )

    results = retriever._source_lane_results(
        source_lane_queries(query, graph_results),
        temporal_point=None,
        limit=5,
    )

    assert primary_answer in results


def test_bridge_source_lane_queries_expands_possessive_pet_mentions() -> None:
    """Source evidence should bridge possessive references to concrete entity names."""
    queries = bridge_source_lane_queries(
        "What breed is my dog?",
        [
            "longmemeval_session_id=seed I am thinking of getting a new leash for my dog Max.",
            "longmemeval_session_id=other My cat Luna needs food.",
        ],
    )

    assert queries == ("Max breed",)


def test_bridge_source_lane_queries_ignores_lowercase_false_aliases() -> None:
    """Bridge alias extraction should not treat ordinary lowercase words as names."""
    queries = bridge_source_lane_queries(
        "What breed is my dog?",
        [
            (
                "longmemeval_session_id=seed Can you show me some options "
                "for hands-free leashes for my dog?"
            ),
            "longmemeval_session_id=answer I use this with my dog Max.",
        ],
    )

    assert queries == ("Max breed",)


def test_bridge_source_lane_queries_ignores_category_phrases() -> None:
    """Bridge alias extraction should not treat title-cased product phrases as names."""
    queries = bridge_source_lane_queries(
        "What breed is my dog?",
        [
            "longmemeval_session_id=seed I use Down Dog for my home practice.",
            "longmemeval_session_id=other Interactive dog toys are useful.",
            "longmemeval_session_id=answer I bought a collar for my dog Max.",
        ],
    )

    assert queries == ("Max breed",)


def test_bridge_source_lane_queries_ignores_abstract_possessive_targets() -> None:
    """Entity bridges should not rewrite broad personal-state questions."""
    queries = bridge_source_lane_queries(
        "What is my job?",
        [
            (
                "longmemeval_session_id=seed I talked about my job "
                "as a marketing specialist."
            ),
        ],
    )

    assert queries == ()


def test_bridge_source_lane_queries_ignores_name_questions() -> None:
    """Name questions are answered by the alias-bearing source itself."""
    queries = bridge_source_lane_queries(
        "What is the name of my cat?",
        [
            "longmemeval_session_id=seed My cat Luna needs food.",
        ],
    )

    assert queries == ()


async def test_zaxy_retriever_uses_source_entity_bridge_for_cross_turn_properties() -> None:
    """Source lookup should bridge my dog -> Max before looking for breed evidence."""
    source_contexts = [
        "longmemeval_session_id=seed I am thinking of getting a new leash for my dog Max.",
        "longmemeval_session_id=distractor my dog likes puzzle toys.",
        "longmemeval_session_id=answer_723bf11f Golden Retriever like Max.",
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
            return [SimpleNamespace(content="graph context about generic dogs")]

    class BridgeAwareLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del temporal_point
            seen_queries.append(query)
            if query == "What breed is my dog?":
                return source_contexts[:2]
            if query == "Max breed":
                return [source_contexts[2]]
            return []

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BridgeAwareLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async("What breed is my dog?", limit=5)

    assert "Max breed" in seen_queries
    assert any("Golden Retriever" in result for result in results)


async def test_zaxy_retriever_promotes_bridge_hits_when_source_lane_is_full() -> None:
    """Bridge evidence should not be starved by generic primary source hits."""
    primary_contexts = [
        "longmemeval_session_id=seed I am thinking of getting a new leash for my dog Max.",
        "longmemeval_session_id=distractor-1 my dog likes puzzle toys.",
        "longmemeval_session_id=distractor-2 my dog needs a bath.",
        "longmemeval_session_id=distractor-3 my dog sleeps on the couch.",
        "longmemeval_session_id=distractor-4 my dog knows a trick.",
    ]
    answer_context = "longmemeval_session_id=answer_723bf11f Golden Retriever like Max."

    class FakeRouter:
        async def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int | None = None,
            embedding: list[float] | None = None,
        ) -> list[SimpleNamespace]:
            del query, temporal_point, limit, embedding
            return [SimpleNamespace(content="graph context about generic dogs")]

    class BridgeAwareLexical:
        def query(
            self,
            query: str,
            temporal_point: str | None = None,
            limit: int = 10,
        ) -> list[str]:
            del temporal_point
            if query == "What breed is my dog?":
                return primary_contexts[:limit]
            if query == "Max breed":
                return [answer_context]
            return []

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BridgeAwareLexical(),  # type: ignore[arg-type]
    )

    results = await retriever.query_async("What breed is my dog?", limit=5)

    assert any("Golden Retriever" in result for result in results)


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


def test_longmemeval_case_scope_terms_are_path_scoped() -> None:
    """LongMemEval benchmark cases should resolve to their per-question document namespace."""
    case = BenchmarkCase(
        name="longmemeval-gpt4_d84a3211",
        query="How much total money have I spent on bike-related expenses?",
        expected_terms=("$185",),
    )

    assert benchmark_case_scope_terms(case) == ("longmemeval/gpt4_d84a3211/",)


def test_scoped_fetch_limit_reaches_cap_for_sparse_domain_hits() -> None:
    """Scoped post-filtering should overfetch enough for sparse LongMemEval domains."""
    assert _scoped_fetch_limit(10) == 100
    assert _scoped_source_fetch_limit(12) == 96
    assert _scoped_source_fallback_limit(96) == 24
    assert _scope_augmented_source_query(
        "Which book did I finish a week ago?",
        ("longmemeval/2ebe6c92/",),
    ) == "Which book did I finish a week ago? longmemeval/2ebe6c92/"


async def test_zaxy_retriever_scopes_source_synthesis_before_aggregation() -> None:
    """Source synthesis should not mix unrelated benchmark/user domains."""
    corpus = (
        BenchmarkChunk(
            "target-1",
            (
                "longmemeval/gpt4_d84a3211/answer_1/salient-turn.md "
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer_1 I bought a Bell Zephyr bike helmet for $120."
            ),
        ),
        BenchmarkChunk(
            "target-2",
            (
                "longmemeval/gpt4_d84a3211/answer_2/salient-turn.md "
                "citation=eventloom://benchmark/events/2#abc "
                "longmemeval_session_id=answer_2 I replaced my bike chain, which was $25."
            ),
        ),
        BenchmarkChunk(
            "target-3",
            (
                "longmemeval/gpt4_d84a3211/answer_3/salient-turn.md "
                "citation=eventloom://benchmark/events/3#abc "
                "longmemeval_session_id=answer_3 I got bike lights installed for $40."
            ),
        ),
        BenchmarkChunk(
            "distractor",
            (
                "longmemeval/129d1232/answer_distractor/salient-turn.md "
                "citation=eventloom://benchmark/events/4#abc "
                "longmemeval_session_id=answer_distractor "
                "I participated in a Bike-a-Thon for Cancer Research and raised $5,000."
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
            return []

    retriever = ZaxyRetriever(
        FakeRouter(),  # type: ignore[arg-type]
        HashEmbeddingProvider(dimension=8),
        lexical_retriever=BM25Retriever(corpus),
        scope_resolver=lambda query: ("longmemeval/gpt4_d84a3211/",),
    )

    results = await retriever.query_async(
        "How much total money have I spent on bike-related expenses since the start of the year?",
        limit=5,
    )

    assert "$185" in results[0]
    assert "$5,000" not in results[0]


def test_aggregation_intent_reserves_larger_source_set() -> None:
    """Aggregation questions should allocate enough source slots for collection."""
    intent = classify_retrieval_intent("How many weddings did I attend?", limit=10)

    assert intent.needs_source_lane
    assert intent.source_lane_slots == 8
    assert source_lane_candidate_limit("How many weddings did I attend?", limit=10) == 48
    assert source_synthesis_candidate_limit(intent, limit=10) == 32


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
    assert "candidate_rank=1 candidate_type=count" in bundle
    assert "candidate_confidence=" in bundle
    assert "candidate_support=answer-1,answer-2,answer-3,answer-4,answer-5,answer-6" in bundle
    assert "source_count=6" in bundle
    assert "count_answer=6" in bundle
    assert "session_id=answer-1" in bundle
    assert "session_id=answer-6" in bundle
    assert "\n".join(results).count("session_id=answer-") >= 6


async def test_zaxy_retriever_counts_distinct_event_sources_not_duplicate_mentions() -> None:
    """Count synthesis should count distinct cited memories rather than repeated terms."""
    source_contexts = [
        "content=longmemeval_session_id=answer-1 I attended the Spring Film Festival. festival festival.",
        "content=longmemeval_session_id=answer-2 I attended the Lakeside Film Festival.",
        "content=longmemeval_session_id=answer-3 I attended the Indie Film Festival.",
        "content=longmemeval_session_id=answer-4 I attended the Documentary Film Festival.",
        "content=longmemeval_session_id=answer-4 I attended the Documentary Film Festival again.",
        "content=longmemeval_session_id=distractor-1 I watched a movie at home.",
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

    results = await retriever.query_async(
        "How many movie festivals did I attend, and which were they?",
        limit=5,
    )

    bundle = results[0]
    assert "count_answer=4" in bundle
    assert "count_unit=events" in bundle
    assert "count_answer_text=I attended four movie festivals." in bundle
    assert "list_item_count=4" in bundle
    assert (
        "list_items=attended the Spring Film Festival | attended the Lakeside Film Festival | "
        "attended the Indie Film Festival | attended the Documentary Film Festival"
    ) in bundle
    assert "source_id=distractor-1" not in bundle


async def test_zaxy_retriever_projects_plain_count_answer_text() -> None:
    """Plain count synthesis should render a compact answer surface without list bloat."""
    source_contexts = [
        "session_id=answer-1 I attended the Portland Film Festival.",
        "session_id=answer-2 I attended the Seattle Film Festival.",
        "session_id=answer-3 I attended the Austin Film Festival.",
        "session_id=answer-4 I attended the Denver Film Festival.",
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

    results = await retriever.query_async(
        "How many movie festivals that I attended?",
        limit=5,
    )

    bundle = results[0]
    assert "count_answer=4" in bundle
    assert "count_answer_text=I attended four movie festivals." in bundle
    assert "list_item_count=" not in bundle


async def test_zaxy_retriever_projects_count_list_answer_details() -> None:
    """Count synthesis should preserve list details needed to answer follow-up clauses."""
    source_contexts = [
        "session_id=answer-1 I attended Rachel and Mike's wedding.",
        "session_id=answer-2 I attended Emily and Sarah's wedding.",
        "session_id=answer-3 I attended Jen and Tom's wedding.",
        "session_id=distractor-1 I planned a birthday dinner.",
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

    results = await retriever.query_async(
        "How many weddings did I attend and who were the couples?",
        limit=5,
    )

    bundle = results[0]
    assert "count_answer=3" in bundle
    assert "count_answer_text=I attended three weddings." in bundle
    assert "list_item_count=3" in bundle
    assert (
        "list_items=attended Rachel and Mike's wedding | "
        "attended Emily and Sarah's wedding | attended Jen and Tom's wedding"
    ) in bundle
    assert "list_source_ids=answer-1,answer-2,answer-3" in bundle


async def test_zaxy_retriever_projects_numeric_operators_in_aggregation_bundle() -> None:
    """Aggregation bundles should expose deterministic numeric operations."""
    source_contexts = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "session_id=answer-1 I booked a Maui resort for $300 per night."
        ),
        (
            "citation=eventloom://benchmark/events/2#abc "
            "session_id=answer-2 I stayed in a Tokyo hostel for $30 per night."
        ),
        (
            "citation=eventloom://benchmark/events/3#abc "
            "session_id=answer-3 I went for a 30-minute jog."
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

    results = await retriever.query_async(
        "How much more did I spend on accommodations per night in Hawaii compared to Tokyo?",
        limit=5,
    )

    bundle = results[0]
    assert "currency_values=$300,$30" in bundle
    assert "currency_difference=$270" in bundle
    assert "currency_difference_answer=$270" in bundle
    assert "minute_total_hours=0.5 hours" in bundle


async def test_zaxy_retriever_deduplicates_repeated_currency_items() -> None:
    """Currency synthesis should not double-count repeated mentions of the same expense."""
    source_contexts = [
        "session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
        "session_id=answer-2 I replaced the bike chain and it cost me $25.",
        "session_id=answer-3 I got a new set of bike lights installed, which were $40.",
        "session_id=answer-4 I recently got a new set of bike lights installed, which were $40.",
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

    results = await retriever.query_async(
        "How much total money have I spent on bike-related expenses since the start of the year?",
        limit=5,
    )

    bundle = results[0]
    assert "currency_values=$120,$40,$25" in bundle
    assert "currency_total=$185" in bundle
    assert "currency_total_answer=$185" in bundle
    assert "currency_total=$225" not in bundle
    assert "currency_source_ids=answer-1,answer-2,answer-3" in bundle


async def test_zaxy_retriever_deduplicates_numeric_values_from_eventloom_context() -> None:
    """Numeric projection should not count payload and JSON echoes twice."""
    source_contexts = [
        (
            "# Event 1\n"
            "citation=eventloom://benchmark/events/1#abc "
            "content=longmemeval_session_id=answer-1 I completed a 5-day camping trip. "
            '{"content": "longmemeval_session_id=answer-1 I completed a 5-day camping trip."}'
        ),
        (
            "# Event 2\n"
            "citation=eventloom://benchmark/events/2#abc "
            "content=longmemeval_session_id=answer-2 I completed a 3-day camping trip. "
            '{"content": "longmemeval_session_id=answer-2 I completed a 3-day camping trip."}'
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

    results = await retriever.query_async(
        "How many days did I spend on camping trips?",
        limit=5,
    )

    bundle = results[0]
    assert "day_values=5,3" in bundle
    assert "day_total=8 days" in bundle
    assert "day_total=16 days" not in bundle


async def test_zaxy_retriever_normalizes_duration_candidates_across_units() -> None:
    """Duration evidence should normalize mixed minute/hour sources with support ids."""
    source_contexts = [
        "content=longmemeval_session_id=answer-1 I played chess for 90 minutes.",
        "content=longmemeval_session_id=answer-2 I practiced piano for 2 hours.",
        "content=longmemeval_session_id=distractor-1 I bought a book for $12.",
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

    results = await retriever.query_async(
        "How many hours did I spend on chess and piano practice?",
        limit=5,
    )

    bundle = results[0]
    assert "duration_values=2 hours,90 minutes" in bundle
    assert "duration_total_minutes=210 minutes" in bundle
    assert "duration_total_hours=3.5 hours" in bundle
    assert "duration_total_answer=3.5 hours" in bundle
    assert "duration_source_ids=answer-2,answer-1" in bundle
    assert "source_id=distractor-1" not in bundle


async def test_zaxy_retriever_formats_large_currency_totals_for_synthesis() -> None:
    """Currency aggregation should preserve thousands separators in answer candidates."""
    source_contexts = [
        "content=longmemeval_session_id=answer-1 I bought a designer bag for $1,500.",
        "content=longmemeval_session_id=answer-2 I bought luxury shoes for $1,000.",
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

    results = await retriever.query_async(
        "What is the total amount I spent on luxury items in the past few months?",
        limit=5,
    )

    bundle = results[0]
    assert "candidate_rank=1 candidate_type=currency" in bundle
    assert "candidate_support=answer-2,answer-1" in bundle
    assert "currency_total=$2,500" in bundle
    assert "currency_total_answer=$2,500" in bundle


async def test_zaxy_retriever_filters_currency_aggregate_to_query_focus() -> None:
    """Aggregate synthesis should exclude unrelated money when query focus is clear."""
    source_contexts = [
        "content=longmemeval_session_id=answer-1 I bought a designer bag for $1,500.",
        "content=longmemeval_session_id=answer-2 I bought luxury shoes for $1,000.",
        "content=longmemeval_session_id=distractor-1 I bought lunch for $20.",
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

    results = await retriever.query_async(
        "What is the total amount I spent on luxury items?",
        limit=5,
    )

    bundle = results[0]
    assert "currency_total=$2,500" in bundle
    assert "$20" not in bundle.split("currency_values=", 1)[1].splitlines()[0]


async def test_zaxy_retriever_names_max_currency_source_for_most_money_query() -> None:
    """Most-money queries should expose the entity/source tied to the max value."""
    source_contexts = [
        "content=longmemeval_session_id=answer-1 I spent $85 at Thrive Market.",
        "content=longmemeval_session_id=answer-2 I spent $42 at Whole Foods.",
        "content=longmemeval_session_id=answer-3 I spent $18 at Trader Joe's.",
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

    results = await retriever.query_async(
        "Which grocery store did I spend the most money at?",
        limit=5,
    )

    bundle = results[0]
    assert "currency_max=$85" in bundle
    assert "currency_max_label=Thrive Market" in bundle


async def test_zaxy_retriever_builds_date_interval_source_synthesis() -> None:
    """Temporal aggregation should expose cited date-difference candidates."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "longmemeval_session_date=2023/02/20 (Mon) "
            "I attended Sunday mass at St. Mary's Church on January 2nd. "
            '{"content": "longmemeval_session_id=answer-1"}'
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "longmemeval_session_date=2023/02/20 (Mon) "
            "I came from the Ash Wednesday service at the cathedral on February 1st. "
            '{"content": "longmemeval_session_id=answer-2"}'
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

    results = await retriever.query_async(
        "How many days had passed between Sunday mass and the Ash Wednesday service?",
        limit=5,
    )

    bundle = results[0]
    assert "candidate_rank=1 candidate_type=date_interval" in bundle
    assert "candidate_support=answer-1,answer-2" in bundle
    assert "date_interval_days=30" in bundle
    assert "date_interval_answer=30 days. 31 days (including the last day) is also acceptable." in bundle
    assert "date_interval_source_ids=answer-1,answer-2" in bundle


async def test_zaxy_retriever_ranks_query_specific_date_intervals_before_distractors() -> None:
    """Temporal synthesis should rank derived intervals by query evidence coverage."""
    source_contexts = [
        (
            "content=longmemeval_session_id=distractor-1 "
            "longmemeval_session_date=2023/03/26 (Sun) "
            "I just got back from Sunday mass at St. Mary's Church on March 19th."
        ),
        (
            "content=longmemeval_session_id=distractor-2 "
            "longmemeval_session_date=2023/03/26 (Sun) "
            "I attended the Holi celebration at my local temple on February 26th."
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
        (
            "content=longmemeval_session_id=distractor-3 "
            "longmemeval_session_date=2023/05/22 (Mon) "
            "I finished a book club assignment two weeks ago."
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
                SimpleNamespace(
                    content=(
                        "longmemeval/08f4fc43/answer-1/salient-turn-0001.md "
                        "longmemeval_session_id=answer-1 "
                        "I came from the Ash Wednesday service at the cathedral on February 1st."
                    )
                ),
                SimpleNamespace(
                    content=(
                        "longmemeval/08f4fc43/answer-2/salient-turn-0001.md "
                        "longmemeval_session_id=answer-2 "
                        "I attended Sunday mass at St. Mary's Church on January 2nd."
                    )
                ),
                *[
                    SimpleNamespace(content=f"graph distractor {index}")
                    for index in range(max(0, (limit or 10) - 2))
                ],
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

    results = await retriever.query_async(
        "How many days had passed between the Sunday mass at St. Mary's Church and the Ash Wednesday service at the cathedral?",
        limit=5,
    )

    bundle = results[0]
    first_interval = bundle.index("date_interval_answer=")
    assert "date_interval_answer=30 days. 31 days (including the last day) is also acceptable." in bundle
    assert first_interval == bundle.index("date_interval_answer=30 days")
    assert "date_interval_source_ids=answer-1,answer-2" in bundle


async def test_zaxy_retriever_prioritizes_query_specific_source_synthesis() -> None:
    """Synthesis should prefer sources matching query concepts over generic numeric hits."""
    source_contexts = [
        (
            "content=longmemeval_session_id=distractor "
            "I trimmed goat hooves two weeks ago and it went well."
        ),
        (
            "content=longmemeval_session_id=answer-1 "
            "longmemeval_session_date=2022/03/02 (Wed) "
            "Since I started working with Rachel on 2/15, she can advise on listings."
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "longmemeval_session_date=2022/03/02 (Wed) "
            "The house I saw on March 1st really checks all the boxes and I loved it."
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

    results = await retriever.query_async(
        "How many days did it take for me to find a house I loved after starting to work with Rachel?",
        limit=2,
    )

    top = "\n".join(results[:5])
    assert "answer-1" in top
    assert "answer-2" in top
    assert "date_interval_answer=14 days. 15 days (including the last day) is also acceptable." in top


async def test_zaxy_retriever_builds_relative_month_source_synthesis() -> None:
    """Relative month evidence should expose a compact derived answer."""
    source_contexts = [
        (
            "content=longmemeval_session_id=distractor "
            "I attended a data analysis webinar two months ago."
        ),
        (
            "content=longmemeval_session_id=answer-1 "
            "I booked my Airbnb three months in advance for the wedding. "
            '{"content": "longmemeval_session_id=answer-1"}'
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "I visited San Francisco exactly two months ago for the wedding. "
            '{"content": "longmemeval_session_id=answer-2"}'
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

    results = await retriever.query_async(
        "How many months ago did I book the Airbnb in San Francisco?",
        limit=5,
    )

    bundle = results[0]
    assert "month_values=3,2" in bundle
    assert "month_total_words=Five months ago" in bundle
    assert "Seven months ago" not in bundle


async def test_zaxy_retriever_builds_relative_month_interval_synthesis() -> None:
    """Relative month evidence should expose duration differences when requested."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "I've been getting into bird watching for about three months now."
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "I recently attended a bird watching workshop at the local Audubon society a month ago."
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

    results = await retriever.query_async(
        "How long had I been bird watching when I attended the bird watching workshop?",
        limit=5,
    )

    bundle = results[0]
    assert "month_values=1,3" in bundle
    assert "month_interval_answer=Two months" in bundle


async def test_zaxy_retriever_builds_relative_week_interval_synthesis() -> None:
    """Relative week evidence should expose duration differences when requested."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "I've been taking weekly guitar lessons with Alex for six weeks now."
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "I just got a new amp two weeks ago."
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

    results = await retriever.query_async(
        "How long had I been taking guitar lessons when I bought the new guitar amp?",
        limit=5,
    )

    bundle = results[0]
    assert "week_values=6,2" in bundle
    assert "week_interval_answer=Four weeks" in bundle


async def test_zaxy_retriever_builds_mixed_relative_time_interval_synthesis() -> None:
    """Relative month/week evidence should expose a day and week interval candidate."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "I recently got a new area rug for my living room a month ago."
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "I rearranged the furniture three weeks ago."
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

    results = await retriever.query_async(
        "How long had I been using the new area rug when I rearranged my living room furniture?",
        limit=5,
    )

    bundle = results[0]
    assert "relative_day_interval=7 days" in bundle
    assert "relative_week_interval_answer=One week" in bundle


async def test_zaxy_retriever_parses_day_of_month_date_intervals() -> None:
    """Day-of-month phrasing should support deterministic date interval synthesis."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "longmemeval_session_date=2022/05/15 (Sun) "
            "I ordered the personalized photo album on the 15th of April."
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "longmemeval_session_date=2022/05/15 (Sun) "
            "I celebrated my best friend's birthday party on the 22nd of April."
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

    results = await retriever.query_async(
        "How many days before my best friend's birthday party did I order her gift?",
        limit=5,
    )

    bundle = results[0]
    assert "date_interval_answer=7 days. 8 days (including the last day) is also acceptable." in bundle


async def test_zaxy_retriever_parses_black_friday_relative_dates() -> None:
    """Named holiday-relative dates should support deterministic interval synthesis."""
    source_contexts = [
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

    results = await retriever.query_async(
        "How many days before I bought the iPhone 13 Pro did I attend the Holiday Market?",
        limit=5,
    )

    bundle = results[0]
    assert "date_interval_answer=7 days. 8 days (including the last day) is also acceptable." in bundle


async def test_zaxy_retriever_builds_absence_bundle_for_missing_query_target() -> None:
    """Source-sensitive queries should warn when the requested target is absent."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "I booked an Airbnb in San Francisco for the wedding."
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "I visited San Francisco two months ago."
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

    results = await retriever.query_async(
        "When did I book the Airbnb in Sacramento?",
        limit=5,
    )

    bundle = results[0]
    assert "zaxy_absence_check=true" in bundle
    assert "The information provided is not enough." in bundle
    assert "did not mention sacramento" in bundle.casefold()
    assert "San Francisco" in bundle


async def test_zaxy_retriever_builds_average_age_synthesis() -> None:
    """Average-age queries should expose a deterministic arithmetic answer."""
    source_contexts = [
        "content=longmemeval_session_id=answer-1 I just turned 32 on February 12th.",
        "content=longmemeval_session_id=answer-2 My mom is 55 and my dad is 58.",
        "content=longmemeval_session_id=answer-3 My grandma is 75 and my grandpa is 78.",
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

    results = await retriever.query_async(
        "What is the average age of me, my parents, and my grandparents?",
        limit=5,
    )

    bundle = results[0]
    assert "age_values=32,55,58,75,78" in bundle
    assert "age_average=59.6" in bundle


async def test_zaxy_retriever_builds_relative_time_offset_synthesis() -> None:
    """Time evidence should expose derived answers from cited relative offsets."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "My usual weekday wake-up time is 7:00 AM."
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "On Tuesdays and Thursdays I wake up 15 minutes earlier for training."
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

    results = await retriever.query_async(
        "What time do I wake up on Tuesdays and Thursdays?",
        limit=5,
    )

    bundle = results[0]
    assert "time_values=7:00 AM" in bundle
    assert "time_offset_minutes=-15" in bundle
    assert "time_offset_answer=6:45 AM" in bundle


async def test_zaxy_retriever_builds_temporal_order_synthesis() -> None:
    """Ordering questions should expose the earliest cited event candidate."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "I received my new phone case about a month ago."
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "I lost my phone charger last week while traveling."
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

    results = await retriever.query_async(
        "Which event happened first, losing phone charger or receiving new phone case?",
        limit=5,
    )

    bundle = results[0]
    assert "temporal_order_answer=received my new phone case" in bundle
    assert "temporal_order_rank=1 relative_days_ago=30" in bundle
    assert "temporal_order_rank=2 relative_days_ago=7" in bundle


async def test_zaxy_retriever_builds_issue_source_synthesis() -> None:
    """Issue questions should expose normalized issue candidates from cited text."""
    source_contexts = [
        (
            "content=longmemeval_session_id=answer-1 "
            "I had an issue with my car's GPS system and took it back to the dealership. "
            '{"content": "longmemeval_session_id=answer-1"}'
        ),
        (
            "content=longmemeval_session_id=answer-2 "
            "I got my car serviced for the first time on March 15th. "
            '{"content": "longmemeval_session_id=answer-2"}'
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

    results = await retriever.query_async(
        "What was the first issue I had with my new car after its first service?",
        limit=5,
    )

    bundle = results[0]
    assert "issue_candidate=GPS system not functioning correctly" in bundle


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


async def test_zaxy_retriever_projects_absence_check_bundle() -> None:
    """Absence checks should expose cited no-direct-mention guidance."""
    corpus = (
        BenchmarkChunk(
            "answer",
            (
                "citation=eventloom://benchmark/events/1#abc "
                "session_id=answer-1 I mentioned my cat Luna during the conversation."
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

    bundle = results[0]
    assert "zaxy_absence_check=true" in bundle
    assert "not_mentioned_candidate=hamster" in bundle
    assert "You did not mention this information." in bundle
    assert "cat Luna" in bundle


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
    answer_recall_at_5: float | None = None,
    recall_at_5: float | None = None,
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
                mean_recall_at_5=recall_at_5,
                mean_answer_recall_at_5=answer_recall_at_5,
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


def test_benchmark_compare_enforces_answer_and_retrieval_recall_floors() -> None:
    """Benchmark guardrails should treat Answer@5 and R@5 as beta quality floors."""
    candidate = _report_with_zaxy_summary(
        mean_score=0.99,
        citation_coverage=1.0,
        p95_ms=250.0,
        p99_ms=300.0,
        answer_recall_at_5=0.94,
        recall_at_5=0.98,
    )

    comparison = compare_benchmark_reports(
        None,
        candidate,
        backend="zaxy",
        min_mean_score=0.95,
        min_citation_coverage=0.95,
        min_answer_recall_at_5=0.95,
        min_recall_at_5=0.99,
        max_p95_ms=500.0,
        max_p99_ms=750.0,
    )

    assert comparison.passed is False
    assert any(
        check.name == "answer_recall_at_5_floor" and check.passed is False
        for check in comparison.checks
    )
    assert any(
        check.name == "recall_at_5_floor" and check.passed is False
        for check in comparison.checks
    )
    markdown = format_benchmark_comparison(comparison)
    assert "answer_recall_at_5_floor" in markdown
    assert "recall_at_5_floor" in markdown


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


def test_source_lane_retriever_uses_semantic_candidates_when_bm25_misses() -> None:
    """Verbatim source recall should not be limited to lexical overlap."""
    corpus = (
        BenchmarkChunk("target", "I met with my primary care physician yesterday."),
        BenchmarkChunk("distractor", "I bought a lamp for the office yesterday."),
    )

    class FakeProvider:
        dimension = 2

        def embed(self, text: str) -> list[float]:
            lowered = text.casefold()
            if "doctor" in lowered or "physician" in lowered:
                return [1.0, 0.0]
            return [0.0, 1.0]

    bm25_results = BM25Retriever(corpus).query("doctor?", limit=2)
    source_results = _build_source_lane_retriever(corpus, FakeProvider()).query(
        "doctor?",
        limit=2,
    )

    assert not any("physician" in result for result in bm25_results)
    assert "primary care physician" in source_results[0]


def test_source_lane_retriever_keeps_hash_provider_lexical_by_default() -> None:
    """Hash embeddings should not activate experimental frontier evidence by default."""
    corpus = (
        BenchmarkChunk("target", "I met with my primary care physician yesterday."),
        BenchmarkChunk("distractor", "I bought a lamp for the office yesterday."),
    )

    source_results = _build_source_lane_retriever(
        corpus,
        HashEmbeddingProvider(dimension=16),
    ).query("doctor?", limit=2)

    assert source_results == []


def test_source_lane_retriever_keeps_cached_hash_provider_lexical() -> None:
    """Benchmark caching should preserve hash-provider capability detection."""
    corpus = (
        BenchmarkChunk("target", "I met with my primary care physician yesterday."),
        BenchmarkChunk("distractor", "I bought a lamp for the office yesterday."),
    )

    retriever = _build_source_lane_retriever(
        corpus,
        CachedEmbeddingProvider(HashEmbeddingProvider(dimension=8)),
    )

    assert type(retriever) is BM25Retriever


def test_evidence_frontier_prefers_first_person_evidence_over_topical_overlap() -> None:
    """Frontier retrieval should rank answerable memory evidence above topical text."""
    corpus = (
        BenchmarkChunk(
            "distractor",
            "This document compares film festival distribution strategies and movie marketing.",
        ),
        BenchmarkChunk(
            "answer",
            "longmemeval_salient_memory_turn=true session_id=answer I went to AFI Fest in LA.",
        ),
    )

    results = EvidenceFrontierRetriever(corpus).query(
        "How many movie festivals did I attend?",
        limit=1,
    )

    assert results == [corpus[1].text]


def test_evidence_frontier_recovers_model_kit_action_paraphrases() -> None:
    """Frontier retrieval should search how memories are written, not only query wording."""
    corpus = (
        BenchmarkChunk("distractor", "model kit manufacturers and custom kit maker advice"),
        BenchmarkChunk(
            "answer",
            "longmemeval_salient_memory_turn=true session_id=answer "
            "I've recently finished a simple Revell F-15 Eagle kit.",
        ),
    )

    results = EvidenceFrontierRetriever(corpus).query(
        "How many model kits have I worked on or bought?",
        limit=1,
    )

    assert results == [corpus[1].text]


def test_live_benchmark_script_help_mentions_frozen_workload() -> None:
    script = Path("scripts/live-benchmark.sh").read_text(encoding="utf-8")

    assert "--workload fixture|statistical|frozen|suite|consolidation" in script
