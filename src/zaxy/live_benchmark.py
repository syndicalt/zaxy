"""Live retrieval benchmark runner for Zaxy and baseline memories."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from zaxy.benchmark import BenchmarkCase, RetrievalScore, _event_context, score_retrieval
from zaxy.embedding import EmbeddingProvider, embed_extraction
from zaxy.event import EventLog
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.query import QueryRouter

FROZEN_WORKLOAD_VERSION = "statistical-v1"
FROZEN_WORKLOAD_SUBJECTS = 100


@dataclass(frozen=True)
class BenchmarkChunk:
    """A retrievable benchmark document chunk."""

    chunk_id: str
    text: str


@dataclass(frozen=True)
class BenchmarkRun:
    """One measured retrieval attempt."""

    backend: str
    case_name: str
    category: str
    run: int
    score: float
    latency_ms: float
    result_count: int
    returned_bytes: int
    approx_tokens: int
    expected_hits: tuple[str, ...]
    missing_expected: tuple[str, ...]
    forbidden_hits: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark statistics for one backend."""

    backend: str
    case_count: int
    runs: int
    mean_score: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    mean_returned_bytes: float
    mean_approx_tokens: float


@dataclass(frozen=True)
class BackendComparison:
    """Paired statistical comparison between target and baseline backend."""

    target_backend: str
    baseline_backend: str
    paired_units: int
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class CategorySummary:
    """Aggregate score for one backend/category pair."""

    backend: str
    category: str
    query_count: int
    mean_score: float
    miss_count: int


@dataclass(frozen=True)
class BenchmarkWorkload:
    """Identity metadata for a benchmark workload."""

    version: str
    subjects: int | None
    event_count: int
    case_count: int
    sha256: str

    @classmethod
    def from_event_log(
        cls,
        eventlog: EventLog,
        cases: tuple[BenchmarkCase, ...],
        *,
        version: str,
        subjects: int | None = None,
    ) -> BenchmarkWorkload:
        """Create workload metadata from an Eventloom log and cases."""
        return cls(
            version=version,
            subjects=subjects,
            event_count=len(eventlog.read_all()),
            case_count=len(cases),
            sha256=workload_fingerprint(eventlog, cases, version),
        )


@dataclass(frozen=True)
class ExternalBenchmarkResult:
    """Operator-supplied benchmark row for an external context system."""

    system: str
    version: str
    mean_score: float
    latency_ms_p95: float | None = None
    source: str = "operator-supplied"
    notes: str | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark output."""

    generated_at: str
    embedding_provider: str
    runs: tuple[BenchmarkRun, ...]
    summaries: tuple[BenchmarkSummary, ...]
    category_summaries: tuple[CategorySummary, ...] = ()
    comparisons: tuple[BackendComparison, ...] = ()
    workload: BenchmarkWorkload | None = None
    external_results: tuple[ExternalBenchmarkResult, ...] = ()


@dataclass(frozen=True)
class WrittenBenchmarkReport:
    """Paths written for a benchmark report."""

    json_path: Path
    markdown_path: Path


class Retriever(Protocol):
    """Protocol implemented by all benchmark retrievers."""

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return context strings for a query."""


class CachedEmbeddingProvider:
    """In-memory embedding cache for benchmark runs.

    This keeps hosted benchmark runs from re-embedding identical corpus chunks
    and repeated paired queries across baselines.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self.dimension = provider.dimension
        self._cache: dict[str, list[float]] = {}

    @property
    def cache_size(self) -> int:
        """Number of cached embedding texts."""
        return len(self._cache)

    def embed(self, text: str) -> list[float]:
        """Return a cached embedding for text, computing it once."""
        cached = self._cache.get(text)
        if cached is None:
            cached = self._provider.embed(text)
            self._cache[text] = cached
        return list(cached)


class MarkdownRetriever:
    """Markdown/file-memory baseline using direct token scanning."""

    def __init__(self, corpus: tuple[BenchmarkChunk, ...]) -> None:
        self._corpus = corpus

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return chunks containing at least one query token."""
        del temporal_point
        tokens = _tokens(query)
        matches: list[str] = []
        for chunk in self._corpus:
            searchable = chunk.text.casefold()
            if any(token in searchable for token in tokens):
                matches.append(chunk.text)
            if len(matches) >= limit:
                break
        return matches


class VectorRetriever:
    """Vector-only baseline over the same benchmark corpus."""

    def __init__(self, corpus: tuple[BenchmarkChunk, ...], provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._indexed = tuple((chunk, provider.embed(chunk.text)) for chunk in corpus)

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return top vector-similar chunks."""
        del temporal_point
        query_embedding = self._provider.embed(query)
        scored = [
            (_cosine(query_embedding, embedding), chunk.text)
            for chunk, embedding in self._indexed
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]


class MarkdownVectorRetriever:
    """Hybrid markdown baseline: token candidate generation plus vector ranking."""

    def __init__(self, corpus: tuple[BenchmarkChunk, ...], provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._corpus = corpus
        self._embeddings = {chunk.chunk_id: provider.embed(chunk.text) for chunk in corpus}

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return token-filtered chunks ranked by vector similarity."""
        del temporal_point
        tokens = _tokens(query)
        candidates = [
            chunk
            for chunk in self._corpus
            if any(token in chunk.text.casefold() for token in tokens)
        ]
        if not candidates:
            candidates = list(self._corpus)

        query_embedding = self._provider.embed(query)
        scored = [
            (_cosine(query_embedding, self._embeddings[chunk.chunk_id]), chunk.text)
            for chunk in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]


class ZaxyRetriever:
    """Synchronous wrapper around Zaxy's live graph retrieval path."""

    def __init__(self, router: QueryRouter, provider: EmbeddingProvider) -> None:
        self._router = router
        self._provider = provider

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return Zaxy graph contexts for a query."""
        embedding = self._provider.embed(query)

        async def _query() -> list[str]:
            chunks = await self._router.query(
                query,
                temporal_point=temporal_point,
                limit=limit,
                embedding=embedding,
            )
            return [chunk.content for chunk in chunks]

        return asyncio.run(_query())

    async def query_async(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return Zaxy graph contexts inside an existing event loop."""
        embedding = self._provider.embed(query)
        chunks = await self._router.query(
            query,
            temporal_point=temporal_point,
            limit=limit,
            embedding=embedding,
        )
        return [chunk.content for chunk in chunks]


def corpus_from_event_log(eventlog: EventLog) -> tuple[BenchmarkChunk, ...]:
    """Build a markdown-like corpus from Eventloom events."""
    return tuple(
        BenchmarkChunk(
            chunk_id=f"event-{event.seq}",
            text=f"# Event {event.seq}\n\n{_event_context(event.model_dump())}",
        )
        for event in eventlog.read_all()
    )


def build_statistical_event_log(
    path: str | Path,
    subjects: int = 100,
) -> tuple[EventLog, tuple[BenchmarkCase, ...]]:
    """Build a larger deterministic temporal workload for statistical evaluation."""
    if subjects <= 0:
        raise ValueError("subjects must be positive")
    log = EventLog(path)
    cases: list[BenchmarkCase] = []
    for idx in range(subjects):
        user_id = f"user-{idx:04d}"
        goal = f"Goal {idx:04d}"
        task_id = f"task-{idx:04d}"
        old_theme = f"theme-old-{idx % 7}"
        new_theme = f"theme-new-{idx % 7}"

        log.append(
            "goal.created",
            actor="user",
            payload={"title": goal, "description": f"Launch workstream {idx:04d}"},
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )
        log.append(
            "task.proposed",
            actor="agent",
            payload={
                "taskId": task_id,
                "goalTitle": goal,
                "summary": f"{task_id} implements {goal} release path",
            },
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
        )
        log.append(
            "user.preference_changed",
            actor="user",
            payload={"userId": user_id, "key": "theme", "value": old_theme},
            timestamp=datetime(2024, 2, 1, tzinfo=UTC),
        )
        log.append(
            "user.preference_changed",
            actor="user",
            payload={"userId": user_id, "key": "theme", "value": new_theme},
            timestamp=datetime(2024, 6, 1, tzinfo=UTC),
        )
        log.append(
            "task.completed",
            actor="agent",
            payload={"taskId": task_id},
            timestamp=datetime(2024, 7, 1, tzinfo=UTC),
        )

        cases.append(
            BenchmarkCase(
                name=f"current-theme-{idx:04d}",
                query=f"What is the current theme preference for {user_id}?",
                expected_terms=(f"theme={new_theme}",),
                forbidden_terms=(f"theme={old_theme}",),
                category="current",
            )
        )
        cases.append(
            BenchmarkCase(
                name=f"historical-theme-{idx:04d}",
                query=f"What was the theme preference for {user_id} in March 2024?",
                temporal_point="2024-03-01T00:00:00Z",
                expected_terms=(f"theme={old_theme}",),
                forbidden_terms=(f"theme={new_theme}",),
                category="temporal",
            )
        )
        cases.append(
            BenchmarkCase(
                name=f"task-for-goal-{idx:04d}",
                query=f"Which task is connected to {goal}?",
                expected_terms=(task_id, goal),
                category="traversal",
            )
        )
    return log, tuple(cases)


def build_frozen_statistical_workload(
    path: str | Path,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build the frozen statistical workload used for publishable comparisons."""
    eventlog, cases = build_statistical_event_log(path, subjects=FROZEN_WORKLOAD_SUBJECTS)
    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        cases,
        version=FROZEN_WORKLOAD_VERSION,
        subjects=FROZEN_WORKLOAD_SUBJECTS,
    )
    return eventlog, cases, workload


def workload_fingerprint(
    eventlog: EventLog,
    cases: tuple[BenchmarkCase, ...],
    version: str,
) -> str:
    """Return a deterministic SHA-256 fingerprint for a benchmark workload."""
    payload = {
        "version": version,
        "events": [
            event.model_dump(mode="json")
            for event in eventlog.read_all()
        ],
        "cases": [asdict(case) for case in cases],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_retrievers(
    retrievers: dict[str, Retriever],
    cases: tuple[BenchmarkCase, ...],
    runs: int = 5,
    limit: int = 10,
    embedding_provider: str = "unknown",
    workload: BenchmarkWorkload | None = None,
    external_results: tuple[ExternalBenchmarkResult, ...] = (),
) -> BenchmarkReport:
    """Benchmark retrievers against shared cases and return statistics."""
    if runs <= 0:
        raise ValueError("runs must be positive")
    measurements: list[BenchmarkRun] = []
    for backend, retriever in retrievers.items():
        for case in cases:
            for run in range(1, runs + 1):
                start = time.perf_counter()
                contexts = retriever.query(
                    case.query,
                    temporal_point=case.temporal_point,
                    limit=limit,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                score = score_retrieval(case, contexts)
                measurements.append(
                    _measurement(
                        backend=backend,
                        case=case,
                        run=run,
                        score=score,
                        contexts=contexts,
                        latency_ms=latency_ms,
                    )
                )

    report = BenchmarkReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        embedding_provider=embedding_provider,
        runs=tuple(measurements),
        summaries=tuple(_summaries(measurements, cases, runs)),
        category_summaries=tuple(_category_summaries(measurements)),
        workload=workload,
        external_results=external_results,
    )
    return _with_comparisons(report)


async def benchmark_live_retrievers(
    retrievers: dict[str, Retriever],
    zaxy_retriever: ZaxyRetriever,
    cases: tuple[BenchmarkCase, ...],
    runs: int = 5,
    limit: int = 10,
    embedding_provider: str = "unknown",
    workload: BenchmarkWorkload | None = None,
    external_results: tuple[ExternalBenchmarkResult, ...] = (),
) -> BenchmarkReport:
    """Benchmark sync baselines and live Zaxy retrieval in one event loop."""
    if runs <= 0:
        raise ValueError("runs must be positive")
    measurements: list[BenchmarkRun] = []
    for backend, retriever in retrievers.items():
        for case in cases:
            for run in range(1, runs + 1):
                start = time.perf_counter()
                contexts = retriever.query(
                    case.query,
                    temporal_point=case.temporal_point,
                    limit=limit,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                score = score_retrieval(case, contexts)
                measurements.append(
                    _measurement(
                        backend=backend,
                        case=case,
                        run=run,
                        score=score,
                        contexts=contexts,
                        latency_ms=latency_ms,
                    )
                )

    for case in cases:
        for run in range(1, runs + 1):
            start = time.perf_counter()
            contexts = await zaxy_retriever.query_async(
                case.query,
                temporal_point=case.temporal_point,
                limit=limit,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            score = score_retrieval(case, contexts)
            measurements.append(
                _measurement(
                    backend="zaxy",
                    case=case,
                    run=run,
                    score=score,
                    contexts=contexts,
                    latency_ms=latency_ms,
                )
            )

    report = BenchmarkReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        embedding_provider=embedding_provider,
        runs=tuple(measurements),
        summaries=tuple(_summaries(measurements, cases, runs)),
        category_summaries=tuple(_category_summaries(measurements)),
        workload=workload,
        external_results=external_results,
    )
    return _with_comparisons(report)


async def build_live_zaxy_retriever(
    eventlog: EventLog,
    provider: EmbeddingProvider,
    neo4j_uri: str = "bolt://localhost:7688",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "testpassword",
    reset_graph: bool = False,
) -> tuple[ZaxyRetriever, GraphStore]:
    """Ingest the benchmark event log into Neo4j and return a live retriever.

    The caller owns the returned ``GraphStore`` and must close it.
    """
    graph = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
    await graph.connect()
    await graph.init_schema()
    if reset_graph:
        assert graph._driver is not None
        await graph._driver.execute_query("MATCH (n:Entity) DETACH DELETE n")
    for event in eventlog.read_all():
        extraction = embed_extraction(extract(event), provider)
        await graph.upsert_extraction(extraction)
    return ZaxyRetriever(QueryRouter(graph), provider), graph


def write_benchmark_report(
    report: BenchmarkReport,
    output_dir: Path,
) -> WrittenBenchmarkReport:
    """Write benchmark JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "live-benchmark.json"
    markdown_path = output_dir / "live-benchmark.md"
    json_path.write_text(
        json.dumps(_report_payload(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
    return WrittenBenchmarkReport(json_path=json_path, markdown_path=markdown_path)


def report_to_markdown(report: BenchmarkReport) -> str:
    """Render benchmark summaries as Markdown."""
    workload = report.workload or _ad_hoc_workload(report)
    lines = [
        "# Live Retrieval Benchmark",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Embedding provider: `{report.embedding_provider}`",
        f"- Workload: `{workload.version}`",
        f"- Workload SHA-256: `{workload.sha256}`",
        f"- Events: `{workload.event_count}`",
        f"- Queries: `{workload.case_count}`",
    ]
    if workload.subjects is not None:
        lines.append(f"- Subjects: `{workload.subjects}`")
    lines.extend(
        [
        "",
        "| Backend | Mean score | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |",
        "|---------|------------|--------|--------|--------|----------------|---------------|",
        ]
    )
    for backend_summary in report.summaries:
        lines.append(
            "| "
            f"{backend_summary.backend} | "
            f"{backend_summary.mean_score:.3f} | "
            f"{backend_summary.latency_ms_p50:.2f} | "
            f"{backend_summary.latency_ms_p95:.2f} | "
            f"{backend_summary.latency_ms_p99:.2f} | "
            f"{backend_summary.mean_returned_bytes:.0f} | "
            f"{backend_summary.mean_approx_tokens:.0f} |"
        )
    if report.category_summaries:
        lines.extend(
            [
                "",
                "## Category summaries",
                "",
                "| Backend | Category | Queries | Mean score | Misses |",
                "|---------|----------|---------|------------|--------|",
            ]
        )
        for category_summary in report.category_summaries:
            lines.append(
                "| "
                f"{category_summary.backend} | "
                f"{category_summary.category} | "
                f"{category_summary.query_count} | "
                f"{category_summary.mean_score:.3f} | "
                f"{category_summary.miss_count} |"
            )
    if report.comparisons:
        lines.extend(
            [
                "",
                "## Paired comparisons",
                "",
                "| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |",
                "|--------|----------|------------------|--------|---------|-------------|",
            ]
        )
        for comparison in report.comparisons:
            lines.append(
                "| "
                f"{comparison.target_backend} | "
                f"{comparison.baseline_backend} | "
                f"{comparison.mean_difference:.4f} | "
                f"[{comparison.ci_low:.4f}, {comparison.ci_high:.4f}] | "
                f"{comparison.p_value:.4f} | "
                f"{'yes' if comparison.significant else 'no'} |"
            )
    if report.external_results:
        lines.extend(
            [
                "",
                "## External comparison disclosures",
                "",
                "| System | Version | Mean score | p95 ms | Source | Notes |",
                "|--------|---------|------------|--------|--------|-------|",
            ]
        )
        for result in report.external_results:
            p95 = "" if result.latency_ms_p95 is None else f"{result.latency_ms_p95:.2f}"
            lines.append(
                "| "
                f"{result.system} | "
                f"{result.version} | "
                f"{result.mean_score:.3f} | "
                f"{p95} | "
                f"{result.source} | "
                f"{result.notes or ''} |"
            )
    return "\n".join(lines) + "\n"


def compare_target_to_baselines(
    report: BenchmarkReport,
    target_backend: str = "zaxy",
    bootstrap_samples: int = 2000,
    seed: int = 1,
) -> tuple[BackendComparison, ...]:
    """Compare target backend against each baseline using paired score deltas."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    backends = sorted({run.backend for run in report.runs})
    baselines = [backend for backend in backends if backend != target_backend]
    comparisons: list[BackendComparison] = []
    for baseline in baselines:
        deltas = _paired_score_deltas(report.runs, target_backend, baseline)
        if not deltas:
            continue
        mean_difference = statistics.fmean(deltas)
        ci_low, ci_high = _bootstrap_ci(deltas, bootstrap_samples, seed)
        p_value = _paired_randomization_p_value(deltas)
        comparisons.append(
            BackendComparison(
                target_backend=target_backend,
                baseline_backend=baseline,
                paired_units=len(deltas),
                mean_difference=round(mean_difference, 4),
                ci_low=round(ci_low, 4),
                ci_high=round(ci_high, 4),
                p_value=round(p_value, 6),
                significant=ci_low > 0.0 and p_value < 0.05,
            )
        )
    return tuple(comparisons)


def _measurement(
    backend: str,
    case: BenchmarkCase,
    run: int,
    score: RetrievalScore,
    contexts: list[str],
    latency_ms: float,
) -> BenchmarkRun:
    returned_text = "\n".join(contexts)
    returned_bytes = len(returned_text.encode("utf-8"))
    return BenchmarkRun(
        backend=backend,
        case_name=case.name,
        category=case.category,
        run=run,
        score=score.score,
        latency_ms=round(latency_ms, 4),
        result_count=len(contexts),
        returned_bytes=returned_bytes,
        approx_tokens=max(1, math.ceil(len(returned_text) / 4)) if returned_text else 0,
        expected_hits=score.expected_hits,
        missing_expected=score.missing_expected,
        forbidden_hits=score.forbidden_hits,
    )


def _summaries(
    measurements: list[BenchmarkRun],
    cases: tuple[BenchmarkCase, ...],
    runs: int,
) -> list[BenchmarkSummary]:
    summaries: list[BenchmarkSummary] = []
    backends = sorted({measurement.backend for measurement in measurements})
    for backend in backends:
        rows = [measurement for measurement in measurements if measurement.backend == backend]
        latencies = [row.latency_ms for row in rows]
        summaries.append(
            BenchmarkSummary(
                backend=backend,
                case_count=len(cases),
                runs=runs,
                mean_score=round(statistics.fmean(row.score for row in rows), 4),
                latency_ms_mean=round(statistics.fmean(latencies), 4),
                latency_ms_p50=round(_percentile(latencies, 50), 4),
                latency_ms_p95=round(_percentile(latencies, 95), 4),
                latency_ms_p99=round(_percentile(latencies, 99), 4),
                mean_returned_bytes=round(
                    statistics.fmean(row.returned_bytes for row in rows), 4
                ),
                mean_approx_tokens=round(
                    statistics.fmean(row.approx_tokens for row in rows), 4
                ),
            )
        )
    return summaries


def _category_summaries(measurements: list[BenchmarkRun]) -> list[CategorySummary]:
    summaries: list[CategorySummary] = []
    keys = sorted({(row.backend, row.category) for row in measurements})
    for backend, category in keys:
        rows = [
            row for row in measurements
            if row.backend == backend and row.category == category
        ]
        summaries.append(
            CategorySummary(
                backend=backend,
                category=category,
                query_count=len(rows),
                mean_score=round(statistics.fmean(row.score for row in rows), 4),
                miss_count=sum(
                    1 for row in rows
                    if row.missing_expected or row.forbidden_hits
                ),
            )
        )
    return summaries


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _tokens(query: str) -> list[str]:
    return [token for token in query.casefold().replace("?", " ").split() if len(token) > 2]


def _report_payload(report: BenchmarkReport) -> dict[str, object]:
    return {
        "generated_at": report.generated_at,
        "embedding_provider": report.embedding_provider,
        "workload": asdict(report.workload or _ad_hoc_workload(report)),
        "summaries": [asdict(summary) for summary in report.summaries],
        "category_summaries": [asdict(summary) for summary in report.category_summaries],
        "comparisons": [asdict(comparison) for comparison in report.comparisons],
        "external_results": [asdict(result) for result in report.external_results],
        "runs": [asdict(run) for run in report.runs],
    }


def _with_comparisons(report: BenchmarkReport) -> BenchmarkReport:
    if "zaxy" not in {run.backend for run in report.runs}:
        return report
    return BenchmarkReport(
        generated_at=report.generated_at,
        embedding_provider=report.embedding_provider,
        runs=report.runs,
        summaries=report.summaries,
        category_summaries=report.category_summaries,
        comparisons=compare_target_to_baselines(report),
        workload=report.workload,
        external_results=report.external_results,
    )


def _ad_hoc_workload(report: BenchmarkReport) -> BenchmarkWorkload:
    case_names = sorted({run.case_name for run in report.runs})
    payload = {
        "embedding_provider": report.embedding_provider,
        "cases": case_names,
        "runs": len(report.runs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return BenchmarkWorkload(
        version="ad-hoc",
        subjects=None,
        event_count=0,
        case_count=len(case_names),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _paired_score_deltas(
    runs: tuple[BenchmarkRun, ...],
    target_backend: str,
    baseline_backend: str,
) -> list[float]:
    target_scores: dict[tuple[str, int], float] = {}
    baseline_scores: dict[tuple[str, int], float] = {}
    for run in runs:
        key = (run.case_name, run.run)
        if run.backend == target_backend:
            target_scores[key] = run.score
        elif run.backend == baseline_backend:
            baseline_scores[key] = run.score
    return [
        target_scores[key] - baseline_scores[key]
        for key in sorted(target_scores.keys() & baseline_scores.keys())
    ]


def _bootstrap_ci(
    deltas: list[float],
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [rng.choice(deltas) for _ in deltas]
        means.append(statistics.fmean(sample))
    alpha = (1.0 - confidence) / 2
    return _quantile(means, alpha), _quantile(means, 1.0 - alpha)


def _paired_randomization_p_value(deltas: list[float]) -> float:
    observed = abs(statistics.fmean(deltas))
    n = len(deltas)
    if n == 0:
        return 1.0
    exact_limit = 20
    if n <= exact_limit:
        extreme = 0
        total = 2**n
        for mask in range(total):
            signed = [
                delta if (mask >> idx) & 1 else -delta
                for idx, delta in enumerate(deltas)
            ]
            if abs(statistics.fmean(signed)) >= observed:
                extreme += 1
        return float(extreme / total)

    rng = random.Random(1)
    samples = 10000
    extreme = 0
    for _ in range(samples):
        signed = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        if abs(statistics.fmean(signed)) >= observed:
            extreme += 1
    return float((extreme + 1) / (samples + 1))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
