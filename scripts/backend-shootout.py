#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Run a same-query projection-backend shootout.

The harness is intentionally local-first: BM25 always runs directly over
Eventloom, while graph backends use the normal MemoryFabric projection contract.
Backends that require unavailable infrastructure fail as backend rows instead of
aborting the whole comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on some platforms.
    resource = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if TYPE_CHECKING:
    from zaxy.context import Context
    from zaxy.core import MemoryFabric
    from zaxy.event import Event


SUPPORTED_BACKENDS = ("embedded", "latticedb", "neo4j", "pggraph", "bm25")
DEFAULT_BACKENDS = ("embedded", "bm25")
REPORT_SCHEMA_VERSION = 1
HARNESS_NAME = "zaxy-backend-shootout"


@dataclass(frozen=True)
class QuerySpec:
    query: str
    expected_terms: tuple[str, ...] = ()
    identity_terms: tuple[str, ...] = ()
    source_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendMetrics:
    backend: str
    contract: str
    status: str
    query_count: int
    cold_bootstrap_ms: float | None
    first_useful_init_ms: float | None
    first_checkout_ms: float | None
    append_to_projection_p95_ms: float | None
    projection_events_per_second: float | None
    checkout_p95_ms: float | None
    checkout_p99_ms: float | None
    exact_p50_ms: float | None
    exact_p95_ms: float | None
    exact_p99_ms: float | None
    keyword_p50_ms: float | None
    keyword_p95_ms: float | None
    keyword_p99_ms: float | None
    vector_p50_ms: float | None
    vector_p95_ms: float | None
    vector_p99_ms: float | None
    traversal_p50_ms: float | None
    traversal_p95_ms: float | None
    traversal_p99_ms: float | None
    dashboard_graph_load_ms: float | None
    dashboard_graph_source: str | None
    dashboard_graph_nodes: int | None
    dashboard_graph_edges: int | None
    mean_returned_tokens: float | None
    quality_per_1k_returned_tokens: float | None
    answer_at_5_per_1k_returned_tokens: float | None
    mean_injected_tokens: float | None
    quality_per_1k_injected_tokens: float | None
    answer_at_5_per_1k_injected_tokens: float | None
    citation_coverage: float | None
    mean_quality: float | None
    answer_at_5: float | None
    recall_at_5: float | None
    memory_footprint_bytes: int | None
    resident_memory_delta_bytes: int | None
    on_disk_footprint_bytes: int | None
    rebuild_recovery_ms: float | None
    error: str | None = None


@dataclass(frozen=True)
class BackendRun:
    metrics: BackendMetrics
    query_results: list[dict[str, Any]]


@dataclass(frozen=True)
class _BM25Document:
    event: Event
    text: str


@dataclass
class _ContractAccumulator:
    contract: str
    latencies: list[float] = field(default_factory=list)
    returned_tokens: list[int] = field(default_factory=list)
    injected_tokens: list[int] = field(default_factory=list)
    citation_hits: int = 0
    quality: list[float] = field(default_factory=list)
    recall_quality: list[float] = field(default_factory=list)
    query_results: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, spec: QuerySpec, contexts: list[Context], latency_ms: float) -> None:
        returned = "\n".join(context.content for context in contexts)
        returned_token_count = _approx_tokens(returned)
        injected_token_count = _approx_tokens(_injected_context_text(contexts))
        citation_hit = any(_context_has_citation(context) for context in contexts)
        query_quality = _expected_term_quality(returned, spec.expected_terms)
        query_recall_quality = _retrieval_target_quality(returned, spec)
        self.latencies.append(latency_ms)
        self.returned_tokens.append(returned_token_count)
        self.injected_tokens.append(injected_token_count)
        if citation_hit:
            self.citation_hits += 1
        self.quality.append(query_quality)
        self.recall_quality.append(query_recall_quality)
        self.query_results.append(
            _query_diagnostic(
                spec,
                returned,
                quality=query_quality,
                recall_quality=query_recall_quality,
                latency_ms=latency_ms,
                returned_tokens=returned_token_count,
                injected_tokens=injected_token_count,
                citation_hit=citation_hit,
                top_contexts=_context_diagnostics(contexts),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Zaxy projection backends on identical queries.")
    parser.add_argument("--eventloom-path", required=True, type=Path, help="Eventloom directory or JSONL file to read")
    parser.add_argument("--session-id", default="default", help="Session ID to benchmark")
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=(
            "Comma-separated backends: embedded, neo4j, pggraph, bm25, or explicit latticedb candidate. "
            "Defaults run embedded plus BM25 without optional sidecar infrastructure."
        ),
    )
    parser.add_argument("--queries-file", required=True, type=Path, help="JSON or line-delimited query file")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report path")
    parser.add_argument("--limit", default=5, type=_positive_int, help="Contexts to retrieve per query")
    parser.add_argument(
        "--include-query-results",
        action="store_true",
        help="Include per-query quality diagnostics for backend parity investigations",
    )
    args = parser.parse_args()

    backends = _parse_backends(args.backends)
    queries = _load_queries(args.queries_file)
    events = _load_events(args.eventloom_path, args.session_id)
    backend_runs = asyncio.run(_run_all(backends, queries, events, args))
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "harness": HARNESS_NAME,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "eventloom_path": str(args.eventloom_path),
        "queries_file": str(args.queries_file),
        "session_id": args.session_id,
        "query_count": len(queries),
        "event_count": len(events),
        "limit": args.limit,
        "source_fingerprints": {
            "eventloom_sha256": _path_fingerprint(args.eventloom_path),
            "queries_sha256": _path_fingerprint(args.queries_file),
        },
        "workload_fingerprints": {
            "events_sha256": _events_fingerprint(events),
            "queries_sha256": _queries_fingerprint(queries),
        },
        "summaries": [asdict(run.metrics) for run in backend_runs],
    }
    if args.include_query_results:
        report["query_results"] = {
            _run_result_key(run): run.query_results
            for run in backend_runs
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_strict_json_dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_format_markdown(report), encoding="utf-8")
    print(f"Wrote JSON report: {args.output}")
    print(f"Wrote Markdown report: {markdown_path}")


def _run_result_key(run: BackendRun) -> str:
    return f"{run.metrics.backend}:{run.metrics.contract}"


async def _run_all(
    backends: list[str],
    queries: list[QuerySpec],
    events: list[Event],
    args: argparse.Namespace,
) -> list[BackendRun]:
    results = []
    for backend in backends:
        if backend == "bm25":
            results.append(_run_bm25(queries, events, limit=args.limit))
        else:
            results.extend(await _run_graph_backend(backend, queries, events, args))
    return results


def _run_bm25(queries: list[QuerySpec], events: list[Event], *, limit: int) -> BackendRun:
    memory_tracker = _ResidentMemoryTracker(start_bytes=_current_rss_bytes())
    start = time.perf_counter()
    corpus = [_BM25Document(event=event, text=_event_text(event)) for event in events]
    init_ms = (time.perf_counter() - start) * 1000
    memory_tracker.observe()
    latencies = []
    returned_tokens = []
    injected_tokens = []
    citation_hits = 0
    quality = []
    recall_quality = []
    query_results = []
    for spec in queries:
        query_start = time.perf_counter()
        hits = _bm25_search(spec.query, corpus, limit=limit)
        latencies.append((time.perf_counter() - query_start) * 1000)
        returned = "\n".join(document.text for _, document in hits)
        returned_token_count = _approx_tokens(returned)
        injected_token_count = _approx_tokens(returned)
        returned_tokens.append(returned_token_count)
        injected_tokens.append(injected_token_count)
        query_quality = _expected_term_quality(returned, spec.expected_terms)
        query_recall_quality = _retrieval_target_quality(returned, spec)
        quality.append(query_quality)
        recall_quality.append(query_recall_quality)
        diagnostic_contexts = _bm25_diagnostic_contexts(hits)
        citation_hit = any(
            isinstance(context.get("citation"), str) and bool(context["citation"].strip())
            for context in diagnostic_contexts
        )
        if citation_hit:
            citation_hits += 1
        query_results.append(
            _query_diagnostic(
                spec,
                returned,
                quality=query_quality,
                recall_quality=query_recall_quality,
                latency_ms=latencies[-1],
                returned_tokens=returned_token_count,
                injected_tokens=injected_token_count,
                citation_hit=citation_hit,
                top_contexts=diagnostic_contexts,
            )
        )
        memory_tracker.observe()
    mean_tokens = _mean(returned_tokens)
    mean_injected_tokens = _mean(injected_tokens)
    answer_at_5 = _answer_at_5(queries, quality)
    memory_footprint = sum(len(document.text.encode("utf-8")) for document in corpus)
    return BackendRun(
        metrics=BackendMetrics(
            backend="bm25",
            contract="retrieve",
            status="ok",
            query_count=len(queries),
            cold_bootstrap_ms=round(init_ms, 3),
            first_useful_init_ms=round(init_ms, 3),
            first_checkout_ms=_first_or_none(latencies),
            append_to_projection_p95_ms=None,
            projection_events_per_second=None,
            checkout_p95_ms=_percentile(latencies, 95),
            checkout_p99_ms=_percentile(latencies, 99),
            exact_p50_ms=None,
            exact_p95_ms=None,
            exact_p99_ms=None,
            keyword_p50_ms=None,
            keyword_p95_ms=None,
            keyword_p99_ms=None,
            vector_p50_ms=None,
            vector_p95_ms=None,
            vector_p99_ms=None,
            traversal_p50_ms=None,
            traversal_p95_ms=None,
            traversal_p99_ms=None,
            dashboard_graph_load_ms=None,
            dashboard_graph_source=None,
            dashboard_graph_nodes=None,
            dashboard_graph_edges=None,
            mean_returned_tokens=mean_tokens,
            quality_per_1k_returned_tokens=_per_1k_tokens(_mean(quality), mean_tokens),
            answer_at_5_per_1k_returned_tokens=_per_1k_tokens(answer_at_5, mean_tokens),
            mean_injected_tokens=mean_injected_tokens,
            quality_per_1k_injected_tokens=_per_1k_tokens(_mean(quality), mean_injected_tokens),
            answer_at_5_per_1k_injected_tokens=_per_1k_tokens(answer_at_5, mean_injected_tokens),
            citation_coverage=round(citation_hits / len(queries), 4) if queries else None,
            mean_quality=_mean(quality),
            answer_at_5=answer_at_5,
            recall_at_5=_recall_at_5(queries, recall_quality),
            memory_footprint_bytes=memory_footprint,
            resident_memory_delta_bytes=memory_tracker.delta_bytes(),
            on_disk_footprint_bytes=0,
            rebuild_recovery_ms=0.0,
        ),
        query_results=query_results,
    )


async def _run_graph_backend(
    backend: str,
    queries: list[QuerySpec],
    events: list[Event],
    args: argparse.Namespace,
) -> list[BackendRun]:
    from zaxy.core import MemoryFabric

    embedded_path = args.output.parent / f"{backend}.kuzu"
    latticedb_path = args.output.parent / "latticedb.db"
    if backend == "embedded" and embedded_path.exists():
        if embedded_path.is_dir():
            shutil.rmtree(embedded_path)
        else:
            embedded_path.unlink()
    if backend == "latticedb" and latticedb_path.exists():
        if latticedb_path.is_dir():
            shutil.rmtree(latticedb_path)
        else:
            latticedb_path.unlink()
    fabric_eventloom_path = _prepare_fabric_eventloom(
        events,
        output_parent=args.output.parent,
        backend=backend,
        session_id=args.session_id,
    )
    fabric = MemoryFabric(
        eventloom_path=str(fabric_eventloom_path),
        projection_backend=backend,
        embedded_graph_path=embedded_path,
        latticedb_path=latticedb_path,
        tracer_disabled=True,
    )
    exact_latencies: list[float] = []
    keyword_latencies: list[float] = []
    vector_latencies: list[float] = []
    traversal_latencies: list[float] = []
    retrieve_stats = _ContractAccumulator("retrieve")
    answer_stats = _ContractAccumulator("answer_ready")
    memory_tracker = _ResidentMemoryTracker(start_bytes=_current_rss_bytes())
    try:
        start = time.perf_counter()
        await fabric.connect()
        cold_bootstrap_ms = (time.perf_counter() - start) * 1000
        memory_tracker.observe()
        projection_start = time.perf_counter()
        projection_latencies = await _project_events(fabric, events, args.session_id)
        projection_ms = (time.perf_counter() - projection_start) * 1000
        init_ms = (time.perf_counter() - start) * 1000
        memory_tracker.observe()
        for spec in queries:
            query_embedding = _embed_query(fabric, spec.query)
            exact_start = time.perf_counter()
            await fabric.graph.search_exact(spec.query, session_id=args.session_id)
            exact_latencies.append((time.perf_counter() - exact_start) * 1000)
            keyword_start = time.perf_counter()
            await fabric.graph.search_keyword(spec.query, limit=args.limit, session_id=args.session_id)
            keyword_latencies.append((time.perf_counter() - keyword_start) * 1000)
            if query_embedding is not None:
                vector_start = time.perf_counter()
                try:
                    await fabric.graph.search_vector(query_embedding, limit=args.limit, session_id=args.session_id)
                    vector_latencies.append((time.perf_counter() - vector_start) * 1000)
                except Exception:
                    pass
            retrieve_start = time.perf_counter()
            retrieval_contexts = await fabric.retrieve(
                spec.query,
                limit=args.limit,
                embedding=query_embedding,
                session_id=args.session_id,
            )
            retrieve_stats.observe(spec, retrieval_contexts, (time.perf_counter() - retrieve_start) * 1000)
            query_start = time.perf_counter()
            contexts = await fabric.query(
                spec.query,
                limit=args.limit,
                embedding=query_embedding,
                session_id=args.session_id,
            )
            answer_stats.observe(spec, contexts, (time.perf_counter() - query_start) * 1000)
            traversal_start = time.perf_counter()
            await fabric.graph.search_traversal(spec.query, depth=2, session_id=args.session_id)
            traversal_latencies.append((time.perf_counter() - traversal_start) * 1000)
            memory_tracker.observe()
        rebuild_recovery_ms = await _measure_rebuild_recovery(fabric, events, args.session_id)
        memory_tracker.observe()
        await fabric.close()
        memory_tracker.observe()
        dashboard_graph = await asyncio.to_thread(
            _load_dashboard_graph_summary,
            backend=backend,
            eventloom_path=args.eventloom_path,
            session_id=args.session_id,
            embedded_path=embedded_path,
        )
        memory_tracker.observe()
        on_disk_footprint = _backend_memory_footprint(backend, embedded_path, latticedb_path)
        common = {
            "backend": backend,
            "status": "ok",
            "query_count": len(queries),
            "cold_bootstrap_ms": round(cold_bootstrap_ms, 3),
            "first_useful_init_ms": round(init_ms, 3),
            "append_to_projection_p95_ms": _percentile(projection_latencies, 95),
            "projection_events_per_second": _events_per_second(len(events), projection_ms),
            "exact_p50_ms": _percentile(exact_latencies, 50),
            "exact_p95_ms": _percentile(exact_latencies, 95),
            "exact_p99_ms": _percentile(exact_latencies, 99),
            "keyword_p50_ms": _percentile(keyword_latencies, 50),
            "keyword_p95_ms": _percentile(keyword_latencies, 95),
            "keyword_p99_ms": _percentile(keyword_latencies, 99),
            "vector_p50_ms": _percentile(vector_latencies, 50),
            "vector_p95_ms": _percentile(vector_latencies, 95),
            "vector_p99_ms": _percentile(vector_latencies, 99),
            "traversal_p50_ms": _percentile(traversal_latencies, 50),
            "traversal_p95_ms": _percentile(traversal_latencies, 95),
            "traversal_p99_ms": _percentile(traversal_latencies, 99),
            "dashboard_graph_load_ms": dashboard_graph["load_ms"],
            "dashboard_graph_source": dashboard_graph["source"],
            "dashboard_graph_nodes": dashboard_graph["nodes"],
            "dashboard_graph_edges": dashboard_graph["edges"],
            "memory_footprint_bytes": on_disk_footprint,
            "resident_memory_delta_bytes": memory_tracker.delta_bytes(),
            "on_disk_footprint_bytes": on_disk_footprint,
            "rebuild_recovery_ms": rebuild_recovery_ms,
        }
        return [
            _build_contract_run(
                stats,
                queries=queries,
                common=common,
            )
            for stats in (retrieve_stats, answer_stats)
        ]
    except Exception as exc:
        return [
            _build_backend_error_run(
                backend=backend,
                contract=contract,
                query_count=len(queries),
                memory_footprint=_backend_memory_footprint(backend, embedded_path, latticedb_path),
                resident_memory_delta=memory_tracker.delta_bytes(),
                error=str(exc),
            )
            for contract in ("retrieve", "answer_ready")
        ]
    finally:
        await fabric.close()
        shutil.rmtree(fabric_eventloom_path, ignore_errors=True)


def _build_backend_error_run(
    *,
    backend: str,
    contract: str,
    query_count: int,
    memory_footprint: int | None,
    resident_memory_delta: int | None,
    error: str,
) -> BackendRun:
    return BackendRun(
        metrics=BackendMetrics(
            backend=backend,
            contract=contract,
            status="error",
            query_count=query_count,
            cold_bootstrap_ms=None,
            first_useful_init_ms=None,
            first_checkout_ms=None,
            append_to_projection_p95_ms=None,
            projection_events_per_second=None,
            checkout_p95_ms=None,
            checkout_p99_ms=None,
            exact_p50_ms=None,
            exact_p95_ms=None,
            exact_p99_ms=None,
            keyword_p50_ms=None,
            keyword_p95_ms=None,
            keyword_p99_ms=None,
            vector_p50_ms=None,
            vector_p95_ms=None,
            vector_p99_ms=None,
            traversal_p50_ms=None,
            traversal_p95_ms=None,
            traversal_p99_ms=None,
            dashboard_graph_load_ms=None,
            dashboard_graph_source=None,
            dashboard_graph_nodes=None,
            dashboard_graph_edges=None,
            mean_returned_tokens=None,
            quality_per_1k_returned_tokens=None,
            answer_at_5_per_1k_returned_tokens=None,
            mean_injected_tokens=None,
            quality_per_1k_injected_tokens=None,
            answer_at_5_per_1k_injected_tokens=None,
            citation_coverage=None,
            mean_quality=None,
            answer_at_5=None,
            recall_at_5=None,
            memory_footprint_bytes=memory_footprint,
            resident_memory_delta_bytes=resident_memory_delta,
            on_disk_footprint_bytes=memory_footprint,
            rebuild_recovery_ms=None,
            error=error,
        ),
        query_results=[],
    )


def _build_contract_run(
    stats: _ContractAccumulator,
    *,
    queries: list[QuerySpec],
    common: dict[str, Any],
) -> BackendRun:
    mean_tokens = _mean(stats.returned_tokens)
    mean_injected_tokens = _mean(stats.injected_tokens)
    answer_at_5 = _answer_at_5(queries, stats.quality)
    return BackendRun(
        metrics=BackendMetrics(
            backend=common["backend"],
            contract=stats.contract,
            status=common["status"],
            query_count=common["query_count"],
            cold_bootstrap_ms=common["cold_bootstrap_ms"],
            first_useful_init_ms=common["first_useful_init_ms"],
            first_checkout_ms=_first_or_none(stats.latencies),
            append_to_projection_p95_ms=common["append_to_projection_p95_ms"],
            projection_events_per_second=common["projection_events_per_second"],
            checkout_p95_ms=_percentile(stats.latencies, 95),
            checkout_p99_ms=_percentile(stats.latencies, 99),
            exact_p50_ms=common["exact_p50_ms"],
            exact_p95_ms=common["exact_p95_ms"],
            exact_p99_ms=common["exact_p99_ms"],
            keyword_p50_ms=common["keyword_p50_ms"],
            keyword_p95_ms=common["keyword_p95_ms"],
            keyword_p99_ms=common["keyword_p99_ms"],
            vector_p50_ms=common["vector_p50_ms"],
            vector_p95_ms=common["vector_p95_ms"],
            vector_p99_ms=common["vector_p99_ms"],
            traversal_p50_ms=common["traversal_p50_ms"],
            traversal_p95_ms=common["traversal_p95_ms"],
            traversal_p99_ms=common["traversal_p99_ms"],
            dashboard_graph_load_ms=common["dashboard_graph_load_ms"],
            dashboard_graph_source=common["dashboard_graph_source"],
            dashboard_graph_nodes=common["dashboard_graph_nodes"],
            dashboard_graph_edges=common["dashboard_graph_edges"],
            mean_returned_tokens=mean_tokens,
            quality_per_1k_returned_tokens=_per_1k_tokens(_mean(stats.quality), mean_tokens),
            answer_at_5_per_1k_returned_tokens=_per_1k_tokens(answer_at_5, mean_tokens),
            mean_injected_tokens=mean_injected_tokens,
            quality_per_1k_injected_tokens=_per_1k_tokens(_mean(stats.quality), mean_injected_tokens),
            answer_at_5_per_1k_injected_tokens=_per_1k_tokens(answer_at_5, mean_injected_tokens),
            citation_coverage=round(stats.citation_hits / len(queries), 4) if queries else None,
            mean_quality=_mean(stats.quality),
            answer_at_5=answer_at_5,
            recall_at_5=_recall_at_5(queries, stats.recall_quality),
            memory_footprint_bytes=common["memory_footprint_bytes"],
            resident_memory_delta_bytes=common["resident_memory_delta_bytes"],
            on_disk_footprint_bytes=common["on_disk_footprint_bytes"],
            rebuild_recovery_ms=common["rebuild_recovery_ms"],
        ),
        query_results=stats.query_results,
    )


def _parse_backends(raw: str) -> list[str]:
    backends = [item.strip().casefold() for item in raw.split(",") if item.strip()]
    if not backends:
        raise SystemExit("At least one backend must be selected")
    duplicates = sorted({backend for backend in backends if backends.count(backend) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate backend selection(s): {', '.join(duplicates)}")
    unknown = sorted(set(backends) - set(SUPPORTED_BACKENDS))
    if unknown:
        raise SystemExit(f"Unsupported backend(s): {', '.join(unknown)}")
    return backends


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return value


def _load_dashboard_graph_summary(
    *,
    backend: str,
    eventloom_path: Path,
    session_id: str,
    embedded_path: Path,
) -> dict[str, Any]:
    from zaxy.dashboard import DashboardConfig, build_dashboard_graph_provider, resolve_dashboard_scope

    scope = resolve_dashboard_scope(
        DashboardConfig(
            eventloom_path=eventloom_path,
            session_id=session_id,
            projection_backend=backend,
            embedded_graph_path=embedded_path,
        )
    )
    provider = build_dashboard_graph_provider(scope)
    start = time.perf_counter()
    summary = provider.summary(session_id=session_id)
    load_ms = (time.perf_counter() - start) * 1000
    return {
        "load_ms": round(load_ms, 3),
        "source": summary.get("source"),
        "nodes": _int_or_none(summary.get("nodes")),
        "edges": _int_or_none(summary.get("edges")),
    }


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced >= 0 else None


def _load_queries(path: Path) -> list[QuerySpec]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = _strict_json_loads(text)
    except json.JSONDecodeError as exc:
        if _looks_like_json(text):
            raise SystemExit("queries file contains malformed JSON") from exc
        return [QuerySpec(line.strip()) for line in text.splitlines() if line.strip()]
    except ValueError as exc:
        raise SystemExit(f"queries file {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit("queries file must contain a JSON list or line-delimited queries")
    specs = []
    for item in payload:
        if isinstance(item, str):
            specs.append(_query_spec(item))
        elif isinstance(item, dict) and isinstance(item.get("query"), str):
            expected_terms = _term_list(item, "expected_terms")
            identity_terms = _term_list(item, "identity_terms")
            source_terms = _term_list(item, "source_terms")
            specs.append(
                _query_spec(
                    item["query"],
                    _expected_terms(expected_terms),
                    identity_terms=_expected_terms(identity_terms),
                    source_terms=_expected_terms(source_terms),
                )
            )
        else:
            raise SystemExit("query entries must be strings or objects with a query field")
    if not specs:
        raise SystemExit("queries file must contain at least one query")
    return specs


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("[") or stripped.startswith("{")


def _term_list(item: dict[str, Any], key: str) -> list[str]:
    terms = item.get(key, [])
    if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
        raise SystemExit(f"{key} must be a list of strings")
    return terms


def _query_spec(
    query: str,
    expected_terms: tuple[str, ...] = (),
    *,
    identity_terms: tuple[str, ...] = (),
    source_terms: tuple[str, ...] = (),
) -> QuerySpec:
    query = query.strip()
    if not query:
        raise SystemExit("query entries must not be empty")
    return QuerySpec(query, expected_terms, identity_terms, source_terms)


def _expected_terms(terms: list[str]) -> tuple[str, ...]:
    normalized = tuple(term.strip() for term in terms)
    if len(normalized) != len([term for term in normalized if term]):
        raise SystemExit("expected_terms must contain non-empty strings")
    return normalized


def _strict_json_loads(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"contains non-standard numeric constant {value}")

    return json.loads(text, parse_constant=reject_constant)


def _load_events(path: Path, session_id: str) -> list[Event]:
    from zaxy.event import EventLog

    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    events = []
    for eventlog_path in paths:
        for event in EventLog(eventlog_path).read_all():
            if event.thread == session_id:
                events.append(event)
    return events


def _path_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        _update_digest_for_file(digest, path, label=path.name)
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        _update_digest_for_file(digest, child, label=child.relative_to(path).as_posix())
    return digest.hexdigest()


def _update_digest_for_file(digest: Any, path: Path, *, label: str) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")


def _events_fingerprint(events: list[Event]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(_event_text(event).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _queries_fingerprint(queries: list[QuerySpec]) -> str:
    payload = [_query_fingerprint_payload(spec) for spec in queries]
    return hashlib.sha256(
        _strict_json_dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _query_fingerprint_payload(spec: QuerySpec) -> dict[str, object]:
    payload: dict[str, object] = {
        "query": spec.query,
        "expected_terms": list(spec.expected_terms),
    }
    if spec.identity_terms:
        payload["identity_terms"] = list(spec.identity_terms)
    if spec.source_terms:
        payload["source_terms"] = list(spec.source_terms)
    return payload


def _fabric_eventloom_path(path: Path) -> Path:
    return path.parent if path.is_file() else path


def _prepare_fabric_eventloom(
    events: list[Event],
    *,
    output_parent: Path,
    backend: str,
    session_id: str,
) -> Path:
    """Materialize workload events so MemoryFabric source lanes see the same corpus."""
    eventloom_path = output_parent / f"{backend}.eventloom"
    if eventloom_path.exists():
        shutil.rmtree(eventloom_path)
    eventloom_path.mkdir(parents=True, exist_ok=True)
    log_path = eventloom_path / f"{session_id}.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(event.model_dump_json() + "\n")
    return eventloom_path


def _event_text(event: Event) -> str:
    return _strict_json_dumps(
        {
            "seq": event.seq,
            "type": event.type,
            "actor": event.actor,
            "thread": event.thread,
            "payload": event.payload,
        },
        sort_keys=True,
    )


def _strict_json_dumps(payload: object, **kwargs: Any) -> str:
    return json.dumps(payload, allow_nan=False, **kwargs)


def _bm25_search(query: str, corpus: list[_BM25Document], *, limit: int) -> list[tuple[float, _BM25Document]]:
    query_terms = _terms(query)
    if not query_terms:
        return []
    document_terms = [_terms(document.text) for document in corpus]
    doc_freq = Counter(term for terms in document_terms for term in set(terms))
    average_length = sum(len(terms) for terms in document_terms) / len(document_terms) if document_terms else 0.0
    scores = []
    for document, terms in zip(corpus, document_terms, strict=True):
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            score += _bm25_term_score(
                term_frequency=counts[term],
                doc_frequency=doc_freq[term],
                document_count=len(corpus),
                document_length=len(terms),
                average_length=average_length,
            )
        if score > 0:
            scores.append((score, document))
    return sorted(scores, key=lambda item: item[0], reverse=True)[:limit]


def _bm25_term_score(
    *,
    term_frequency: int,
    doc_frequency: int,
    document_count: int,
    document_length: int,
    average_length: float,
) -> float:
    if term_frequency <= 0 or doc_frequency <= 0 or average_length <= 0:
        return 0.0
    k1 = 1.5
    b = 0.75
    idf = math.log(1 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
    denominator = term_frequency + k1 * (1 - b + b * document_length / average_length)
    return idf * (term_frequency * (k1 + 1)) / denominator


def _terms(text: str) -> list[str]:
    return [term for term in "".join(char.lower() if char.isalnum() else " " for char in text).split() if term]


def _expected_term_quality(text: str, expected_terms: tuple[str, ...]) -> float:
    from zaxy_benchmarks.benchmark import BenchmarkCase, expected_terms_recall

    if not expected_terms:
        return 1.0
    case = BenchmarkCase(name="backend-shootout-query", query="", expected_terms=expected_terms)
    return expected_terms_recall(case, [text]) or 0.0


def _retrieval_target_quality(text: str, spec: QuerySpec) -> float:
    identity_or_source_terms = spec.source_terms or spec.identity_terms
    if identity_or_source_terms:
        lowered = text.casefold()
        return 1.0 if any(term.casefold() in lowered for term in identity_or_source_terms) else 0.0
    return _expected_term_quality(text, spec.expected_terms)


def _retrieval_target_terms(spec: QuerySpec) -> tuple[str, ...]:
    return spec.source_terms or spec.identity_terms or spec.expected_terms


def _query_diagnostic(
    spec: QuerySpec,
    returned: str,
    *,
    quality: float,
    recall_quality: float,
    latency_ms: float,
    returned_tokens: int,
    injected_tokens: int,
    citation_hit: bool,
    top_contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    lowered = returned.casefold()
    missing = [term for term in spec.expected_terms if term.casefold() not in lowered]
    retrieval_terms = _retrieval_target_terms(spec)
    missing_retrieval = [term for term in retrieval_terms if term.casefold() not in lowered]
    return {
        "query": spec.query,
        "expected_terms": list(spec.expected_terms),
        "identity_terms": list(spec.identity_terms),
        "source_terms": list(spec.source_terms),
        "retrieval_terms": list(retrieval_terms),
        "quality": quality,
        "answer_hit": quality >= 1.0,
        "recall_quality": recall_quality,
        "recall_hit": recall_quality >= 1.0,
        "missing_expected_terms": missing,
        "missing_retrieval_terms": missing_retrieval,
        "latency_ms": round(latency_ms, 3),
        "returned_tokens": returned_tokens,
        "injected_tokens": injected_tokens,
        "citation_hit": citation_hit,
        "top_contexts": top_contexts,
    }


def _bm25_diagnostic_contexts(hits: list[tuple[float, _BM25Document]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "source": "bm25",
            "score": round(score, 6),
            "citation": _event_citation(document.event),
            "snippet": _event_text_snippet(document.text),
        }
        for rank, (score, document) in enumerate(hits, start=1)
    ]


def _event_citation(event: Event) -> str:
    return f"eventloom://{event.thread}/events/{event.seq}#{event.hash}"


def _context_diagnostics(contexts: list[Context]) -> list[dict[str, Any]]:
    diagnostics = []
    for rank, context in enumerate(contexts, start=1):
        metadata = context.metadata or {}
        diagnostics.append(
            {
                "rank": rank,
                "source": context.source,
                "score": round(context.score, 6),
                "citation": metadata.get("citation"),
                "snippet": _snippet(context.content),
            }
        )
    return diagnostics


def _event_text_snippet(text: str) -> str:
    try:
        event = _strict_json_loads(text)
    except (json.JSONDecodeError, ValueError):
        return _snippet(text)
    if not isinstance(event, dict):
        return _snippet(text)
    payload = event.get("payload")
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return _snippet(content)
        return _snippet(_strict_json_dumps(payload, sort_keys=True))
    return _snippet(text)


def _snippet(text: str, *, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _answer_at_5(queries: list[QuerySpec], qualities: list[float]) -> float | None:
    labeled = _labeled_qualities(queries, qualities)
    if not labeled:
        return None
    return round(sum(1.0 for quality in labeled if quality >= 1.0) / len(labeled), 4)


def _recall_at_5(queries: list[QuerySpec], qualities: list[float]) -> float | None:
    labeled = [
        quality
        for spec, quality in zip(queries, qualities, strict=True)
        if _retrieval_target_terms(spec)
    ]
    return _mean(labeled) if labeled else None


def _labeled_qualities(queries: list[QuerySpec], qualities: list[float]) -> list[float]:
    return [quality for spec, quality in zip(queries, qualities, strict=True) if spec.expected_terms]


def _context_has_citation(context: Context) -> bool:
    metadata = context.metadata or {}
    return bool(metadata.get("citation") or "eventloom://" in context.content)


def _injected_context_text(contexts: list[Context]) -> str:
    blocks = []
    for context in contexts:
        metadata = context.metadata or {}
        citation = metadata.get("citation")
        source_path = metadata.get("source_path")
        relation_path = metadata.get("relation_path")
        block = context.content
        if citation:
            block += f"\ncitation: {citation}"
        if source_path:
            block += f"\nsource_path: {source_path}"
        if relation_path:
            block += f"\nrelation_path: {relation_path}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _embed_query(fabric: MemoryFabric, query: str) -> list[float] | None:
    provider = getattr(fabric, "embedding_provider", None)
    if provider is None:
        return None
    try:
        return cast(list[float], provider.embed(query))
    except Exception:
        return None


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _mean(values: list[float] | list[int]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _first_or_none(values: list[float]) -> float | None:
    return round(values[0], 3) if values else None


def _per_1k_tokens(metric: float | None, mean_tokens: float | None) -> float | None:
    if metric is None or mean_tokens is None or mean_tokens <= 0:
        return None
    return round(metric / (mean_tokens / 1000), 4)


def _events_per_second(event_count: int, elapsed_ms: float) -> float | None:
    if event_count == 0 or elapsed_ms <= 0:
        return None
    return round(event_count / (elapsed_ms / 1000), 3)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(ordered[index], 3)


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Backend Shootout",
        "",
        f"- Report schema version: `{report['report_schema_version']}`",
        f"- Harness: `{report['harness']}`",
        f"- Generated at UTC: `{report['generated_at_utc']}`",
        f"- Eventloom path: `{report['eventloom_path']}`",
        f"- Queries file: `{report['queries_file']}`",
        f"- Session ID: `{report['session_id']}`",
        f"- Queries: `{report['query_count']}`",
        f"- Events: `{report['event_count']}`",
        f"- Limit: `{report['limit']}`",
        f"- Source Eventloom SHA-256: `{report['source_fingerprints']['eventloom_sha256']}`",
        f"- Source queries SHA-256: `{report['source_fingerprints']['queries_sha256']}`",
        f"- Workload events SHA-256: `{report['workload_fingerprints']['events_sha256']}`",
        f"- Workload queries SHA-256: `{report['workload_fingerprints']['queries_sha256']}`",
        "",
        "| Backend | Contract | Status | Cold bootstrap ms | First useful init ms | First checkout ms | Append projection p95 ms | Projection eps | Checkout p95 ms | Checkout p99 ms | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected | Citation coverage | Mean quality | Answer@5 | Recall@5 | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |",
        "|---------|----------|--------|-------------------|----------------------|-------------------|--------------------------|----------------|-----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|-------------------|--------------|----------|----------|--------------|-----------------------------|-------------------------|---------------------|",
    ]
    for summary in report["summaries"]:
        lines.append(
            "| {backend} | {contract} | {status} | {cold_bootstrap_ms} | {first_useful_init_ms} | "
            "{first_checkout_ms} | {append_to_projection_p95_ms} | {projection_events_per_second} | "
            "{checkout_p95_ms} | {checkout_p99_ms} | {exact_p50_ms} | {exact_p95_ms} | "
            "{exact_p99_ms} | {keyword_p50_ms} | {keyword_p95_ms} | {keyword_p99_ms} | "
            "{vector_p50_ms} | {vector_p95_ms} | {vector_p99_ms} | {traversal_p50_ms} | "
            "{traversal_p95_ms} | {traversal_p99_ms} | {dashboard_graph_load_ms} | {dashboard_graph_source} | "
            "{dashboard_graph_nodes} | {dashboard_graph_edges} | {mean_returned_tokens} | {quality_per_1k_returned_tokens} | "
            "{answer_at_5_per_1k_returned_tokens} | {mean_injected_tokens} | {quality_per_1k_injected_tokens} | "
            "{answer_at_5_per_1k_injected_tokens} | {citation_coverage} | {mean_quality} | "
            "{answer_at_5} | {recall_at_5} | {memory_footprint_bytes} | {resident_memory_delta_bytes} | "
            "{on_disk_footprint_bytes} | {rebuild_recovery_ms} |".format(
                **{key: _display(value) for key, value in summary.items()}
            )
        )
    lines.append("")
    return "\n".join(lines)


def _display(value: Any) -> str:
    return "-" if value is None else str(value)


async def _measure_rebuild_recovery(fabric: MemoryFabric, events: list[Event], session_id: str) -> float | None:
    reset = getattr(fabric.graph, "reset_benchmark_projection", None)
    if not callable(reset):
        return None
    start = time.perf_counter()
    await reset()
    await _project_events(fabric, events, session_id)
    return round((time.perf_counter() - start) * 1000, 3)


async def _project_events(fabric: Any, events: list[Any], session_id: str) -> list[float]:
    begin = getattr(fabric.graph, "begin_bulk_projection", None)
    commit = getattr(fabric.graph, "commit_bulk_projection", None)
    rollback = getattr(fabric.graph, "rollback_bulk_projection", None)
    use_bulk = callable(begin) and callable(commit)
    if use_bulk:
        await cast(Callable[[], Awaitable[None]], begin)()
    latencies: list[float] = []
    try:
        for event in events:
            start = time.perf_counter()
            await fabric._project_event(event, session_id=session_id)
            latencies.append((time.perf_counter() - start) * 1000)
        if use_bulk:
            await cast(Callable[[], Awaitable[None]], commit)()
    except Exception:
        if use_bulk and callable(rollback):
            with contextlib.suppress(Exception):
                await cast(Callable[[], Awaitable[None]], rollback)()
        raise
    return latencies


def _backend_memory_footprint(backend: str, embedded_path: Path, latticedb_path: Path) -> int | None:
    if backend == "embedded":
        return _path_size(embedded_path)
    if backend == "latticedb":
        return _path_size(latticedb_path)
    return None


def _path_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _current_rss_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            rss = _rss_bytes_from_linux_statm(
                Path("/proc/self/statm").read_text(encoding="utf-8"),
                page_size=os.sysconf("SC_PAGE_SIZE"),
            )
        except (OSError, ValueError):
            rss = None
        if rss is not None:
            return rss
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return max_rss
    return max_rss * 1024


def _rss_bytes_from_linux_statm(raw: str, *, page_size: int) -> int | None:
    fields = raw.split()
    if len(fields) < 2:
        return None
    try:
        resident_pages = int(fields[1])
    except ValueError:
        return None
    if resident_pages < 0 or page_size <= 0:
        return None
    return resident_pages * page_size


class _ResidentMemoryTracker:
    def __init__(self, *, start_bytes: int | None) -> None:
        self.start_bytes = start_bytes
        self.max_delta_bytes: int | None = 0 if start_bytes is not None else None

    def observe(self, current_bytes: int | None = None) -> None:
        if self.start_bytes is None:
            return
        if current_bytes is None:
            current_bytes = _current_rss_bytes()
        if current_bytes is None:
            return
        delta = max(0, current_bytes - self.start_bytes)
        self.max_delta_bytes = max(self.max_delta_bytes or 0, delta)

    def delta_bytes(self) -> int | None:
        return self.max_delta_bytes


if __name__ == "__main__":
    main()
