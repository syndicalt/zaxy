"""Split from cli.py (mechanical decomposition)."""


from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from zaxy.cli.runtime import (
    _benchmark_module,
    app,
)

# Single source of truth for ``zaxy benchmark --workload`` values: it renders the
# option help, rejects unknown values before any workload is built, and is what
# the beta-readiness roadmap gate resolves BETA.md's ``--workload`` claims
# against. Renaming a lane here without renaming it in BETA.md turns that gate
# red, which is the point — the names in the docs and the names the CLI accepts
# are one fact, not two.
BENCHMARK_WORKLOADS: tuple[str, ...] = (
    "fixture",
    "statistical",
    "frozen",
    "suite",
    "consolidation",
    "context-collapse",
    "graph-traversal",
    "source-recall",
    "temporal-recall",
    "longmemeval",
)


def _parse_benchmark_baselines(value: str, *, allow_centroid: bool) -> tuple[str, ...]:
    """Parse the benchmark baseline selection string."""
    allowed = {"md", "bm25", "vector", "md+vector"}
    if allow_centroid:
        allowed.add("centroid")
    if value.strip().casefold() in {"none", "zaxy-only", "zaxy_only"}:
        return ()
    selected = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not selected:
        raise typer.BadParameter("--baseline-backends must include at least one backend")
    invalid = sorted(set(selected) - allowed)
    if invalid:
        allowed_text = ", ".join(sorted(allowed))
        raise typer.BadParameter(
            f"Unsupported baseline backend(s): {', '.join(invalid)}. Allowed: {allowed_text}"
        )
    return selected


def _build_benchmark_baselines(
    corpus: tuple[Any, ...],
    provider: Any,
    selected: tuple[str, ...],
) -> dict[str, Any]:
    """Build only the requested non-Zaxy benchmark baselines."""
    live_benchmark_module = _benchmark_module("live_benchmark")
    bm25_retriever_cls = live_benchmark_module.BM25Retriever
    centroid_consolidation_retriever_cls = live_benchmark_module.CentroidConsolidationRetriever
    markdown_retriever_cls = live_benchmark_module.MarkdownRetriever
    markdown_vector_retriever_cls = live_benchmark_module.MarkdownVectorRetriever
    vector_retriever_cls = live_benchmark_module.VectorRetriever

    retrievers: dict[str, Any] = {}
    for backend in selected:
        if backend == "md":
            retrievers[backend] = markdown_retriever_cls(corpus)
        elif backend == "bm25":
            retrievers[backend] = bm25_retriever_cls(corpus)
        elif backend == "vector":
            retrievers[backend] = vector_retriever_cls(corpus, provider)
        elif backend == "md+vector":
            retrievers[backend] = markdown_vector_retriever_cls(corpus, provider)
        elif backend == "centroid":
            retrievers[backend] = centroid_consolidation_retriever_cls(corpus, provider)
    return retrievers


def _load_external_results(path: Path | None) -> tuple[Any, ...]:
    """Load operator-supplied external benchmark rows from JSON."""
    live_benchmark_module = _benchmark_module("live_benchmark")
    external_benchmark_result_cls = live_benchmark_module.ExternalBenchmarkResult

    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise typer.BadParameter("external results JSON must be a list")
    results: list[Any] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise typer.BadParameter(f"external result {idx} must be an object")
        try:
            results.append(external_benchmark_result_cls(**item))
        except TypeError as exc:
            raise typer.BadParameter(f"invalid external result {idx}: {exc}") from exc
    return tuple(results)


def _parse_agent_experience_lanes(value: str, valid_lanes: tuple[str, ...]) -> tuple[str, ...]:
    """Parse the agent-experience lane selection string."""
    normalized = value.strip().casefold()
    if normalized == "all":
        return valid_lanes
    selected = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not selected:
        raise typer.BadParameter("--lanes must include at least one lane")
    invalid = sorted(set(selected) - set(valid_lanes))
    if invalid:
        valid_text = ", ".join((*valid_lanes, "all"))
        raise typer.BadParameter(
            f"Unsupported agent-experience lane(s): {', '.join(invalid)}. Allowed: {valid_text}"
        )
    return selected


def _parse_agent_experience_budgets(value: str) -> tuple[int | None, ...]:
    """Parse the budget sweep string into token budgets with optional unlimited."""
    budgets: list[int | None] = []
    for part in value.split(","):
        token = part.strip().casefold()
        if not token:
            continue
        if token in {"unlimited", "none"}:
            budgets.append(None)
            continue
        try:
            budget = int(token)
        except ValueError as exc:
            raise typer.BadParameter(
                f"invalid budget {part.strip()!r}; use integers or 'unlimited'"
            ) from exc
        if budget < 0:
            raise typer.BadParameter("budgets must be >= 0")
        budgets.append(budget)
    if not budgets:
        raise typer.BadParameter("--budgets must include at least one budget")
    return tuple(budgets)


@app.command("agent-experience-lanes")
def agent_experience_lanes(
    lanes: str = typer.Option(
        "all",
        "--lanes",
        help="Comma-separated internal lanes to run: tool-adoption, budget, cache, or all",
    ),
    budgets: str = typer.Option(
        "256,512,1024,2048,4096,8192,unlimited",
        "--budgets",
        help="Budget-lane token sweep: comma-separated integers plus optional 'unlimited'",
    ),
    repeats: int = typer.Option(
        5,
        "--repeats",
        min=2,
        help="Cache-lane repeated checkouts used to verify byte-identical stable prefixes",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Optional directory for the agent-experience-lanes.json report",
    ),
) -> None:
    """Run the internal Phase 1 agent-experience lanes: tool-adoption, budget, cache."""
    lanes_module = _benchmark_module("agent_experience_lanes")
    run_agent_experience_lanes = lanes_module.run_agent_experience_lanes
    valid_lanes = tuple(lanes_module.AGENT_EXPERIENCE_LANE_NAMES)

    selected_lanes = _parse_agent_experience_lanes(lanes, valid_lanes)
    budget_sweep = _parse_agent_experience_budgets(budgets)
    with tempfile.TemporaryDirectory(prefix="zaxy-agent-experience-") as tmp:
        try:
            payload = run_agent_experience_lanes(
                Path(tmp),
                lanes=selected_lanes,
                budgets=budget_sweep,
                repeats=repeats,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "agent-experience-lanes.json"
        report_path.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"Wrote JSON report: {report_path}", err=True)


COGNITIVE_LANE_NAMES: tuple[str, ...] = ("forgetting", "fok-calibration")


def _parse_cognitive_lanes(value: str) -> tuple[str, ...]:
    """Parse the cognitive lane selection string."""
    normalized = value.strip().casefold()
    if normalized == "all":
        return COGNITIVE_LANE_NAMES
    selected = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not selected:
        raise typer.BadParameter("--lanes must include at least one lane")
    invalid = sorted(set(selected) - set(COGNITIVE_LANE_NAMES))
    if invalid:
        valid_text = ", ".join((*COGNITIVE_LANE_NAMES, "all"))
        raise typer.BadParameter(
            f"Unsupported cognitive lane(s): {', '.join(invalid)}. Allowed: {valid_text}"
        )
    return selected


def _parse_fok_corpus_sizes(value: str) -> tuple[int, ...]:
    """Parse the FoK-calibration corpus-size sweep string."""
    sizes: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            size = int(token)
        except ValueError as exc:
            raise typer.BadParameter(
                f"invalid corpus size {token!r}; use positive integers"
            ) from exc
        if size <= 0:
            raise typer.BadParameter("corpus sizes must be positive integers")
        sizes.append(size)
    if not sizes:
        raise typer.BadParameter("--fok-sizes must include at least one corpus size")
    return tuple(sizes)


@app.command("cognitive-lanes")
def cognitive_lanes(
    lanes: str = typer.Option(
        "all",
        "--lanes",
        help="Comma-separated internal lanes to run: forgetting, fok-calibration, or all",
    ),
    fok_sizes: str = typer.Option(
        "50,200",
        "--fok-sizes",
        help="FoK-calibration corpus entity counts, comma-separated positive integers",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Optional directory for the cognitive-lanes.json report",
    ),
) -> None:
    """Run the internal cognitive memory lanes: forgetting and FoK calibration."""
    selected_lanes = _parse_cognitive_lanes(lanes)
    sizes = _parse_fok_corpus_sizes(fok_sizes)
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="zaxy-cognitive-lanes-") as tmp:
        try:
            if "forgetting" in selected_lanes:
                forgetting_module = _benchmark_module("forgetting_lane")
                results["forgetting"] = forgetting_module.run_forgetting_lane(
                    Path(tmp) / "forgetting"
                )
            if "fok-calibration" in selected_lanes:
                fok_module = _benchmark_module("fok_calibration_lane")
                results["fok_calibration"] = fok_module.run_fok_calibration_lane(
                    Path(tmp) / "fok-calibration",
                    sizes=sizes,
                )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    payload = {
        "version": "cognitive-lanes-v1",
        "validation": "internal",
        "lanes": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "cognitive-lanes.json"
        report_path.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"Wrote JSON report: {report_path}", err=True)


@app.command()
def benchmark(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks"),
        help="Directory for JSON and Markdown benchmark reports",
    ),
    embedding_provider: str = typer.Option(
        "openai",
        help="Embedding provider: openai, hash, local-http, or sentence-transformers",
    ),
    runs: int = typer.Option(5, min=1, help="Measured runs per backend/case"),
    limit: int = typer.Option(10, min=1, max=50, help="Returned contexts per query"),
    neo4j_uri: str = typer.Option("bolt://localhost:7688", help="Benchmark Neo4j Bolt URI"),
    neo4j_user: str = typer.Option("neo4j", help="Benchmark Neo4j username"),
    neo4j_password: str = typer.Option("testpassword", help="Benchmark Neo4j password"),
    reset_graph: bool = typer.Option(
        False,
        help="Delete benchmark projection contents before ingestion",
    ),
    workload: str = typer.Option(
        "fixture",
        help="Workload: " + ", ".join(BENCHMARK_WORKLOADS),
    ),
    dataset: Path | None = typer.Option(  # noqa: B008
        None,
        "--dataset",
        help="Public benchmark dataset path, required for workload=longmemeval",
    ),
    questions: int | None = typer.Option(  # noqa: B008
        None,
        min=1,
        help="Limit public benchmark questions for smoke runs",
    ),
    subjects: int = typer.Option(
        100,
        min=1,
        help="Subject count for statistical/suite workloads; each subject creates 3 memory queries",
    ),
    documents: int = typer.Option(
        250,
        min=1,
        help="Document count for suite workloads; identity count for consolidation",
    ),
    sessions: int = typer.Option(
        50,
        min=1,
        help="Transcript session count for the suite workload",
    ),
    external_results: Path | None = typer.Option(  # noqa: B008
        None,
        help="Optional JSON file with operator-supplied external comparison rows",
    ),
    embedding_cache: Path | None = typer.Option(  # noqa: B008
        None,
        "--embedding-cache",
        help="Optional JSON cache for benchmark embeddings across runs",
    ),
    progress: bool = typer.Option(False, "--progress", help="Print benchmark progress to stderr"),
    reuse_projection: bool = typer.Option(
        False,
        "--reuse-projection",
        help="Reuse an existing benchmark graph projection for the same workload and embedding provider",
    ),
    projection_backend: str = typer.Option(
        "embedded",
        "--projection-backend",
        help=(
            "Projection backend for graph-backed Zaxy benchmarks; embedded default, "
            "or neo4j, pggraph, or latticedb"
        ),
    ),
    pggraph_dsn: str | None = typer.Option(  # noqa: B008
        None,
        "--pggraph-dsn",
        help="Experimental pgGraph/PostgreSQL DSN for --projection-backend pggraph",
    ),
    baseline_backends: str = typer.Option(
        "md,bm25,vector,md+vector",
        "--baseline-backends",
        help="Comma-separated non-Zaxy baselines to run: md,bm25,vector,md+vector,centroid,none",
    ),
    zaxy_backend: str = typer.Option(
        "graph",
        "--zaxy-backend",
        help="Zaxy backend to benchmark: graph, checkout, or both",
    ),
) -> None:
    """Run live retrieval benchmarks against baseline memories and Zaxy."""
    import asyncio

    if workload not in BENCHMARK_WORKLOADS:
        raise typer.BadParameter(
            f"workload must be one of: {', '.join(BENCHMARK_WORKLOADS)}",
            param_hint="--workload",
        )

    benchmark_module = _benchmark_module("benchmark")
    live_benchmark_module = _benchmark_module("live_benchmark")
    build_competitive_event_log = benchmark_module.build_competitive_event_log
    competitive_cases = benchmark_module.competitive_cases
    benchmark_workload_cls = live_benchmark_module.BenchmarkWorkload
    cached_embedding_provider_cls = live_benchmark_module.CachedEmbeddingProvider
    _build_source_lane_retriever = live_benchmark_module._build_source_lane_retriever
    benchmark_live_retrievers = live_benchmark_module.benchmark_live_retrievers
    benchmark_projection_cache_key = live_benchmark_module.benchmark_projection_cache_key
    benchmark_query_scope_resolver = live_benchmark_module.benchmark_query_scope_resolver
    build_benchmark_suite_workload = live_benchmark_module.build_benchmark_suite_workload
    build_consolidation_collapse_workload = live_benchmark_module.build_consolidation_collapse_workload
    build_context_collapse_workload = live_benchmark_module.build_context_collapse_workload
    build_frozen_statistical_workload = live_benchmark_module.build_frozen_statistical_workload
    build_graph_traversal_workload = live_benchmark_module.build_graph_traversal_workload
    build_live_zaxy_retriever = live_benchmark_module.build_live_zaxy_retriever
    build_longmemeval_workload = live_benchmark_module.build_longmemeval_workload
    build_source_recall_workload = live_benchmark_module.build_source_recall_workload
    build_statistical_event_log = live_benchmark_module.build_statistical_event_log
    build_temporal_recall_workload = live_benchmark_module.build_temporal_recall_workload
    corpus_from_event_log = live_benchmark_module.corpus_from_event_log
    report_to_markdown = live_benchmark_module.report_to_markdown
    write_benchmark_report = live_benchmark_module.write_benchmark_report

    from zaxy.config import get_settings
    from zaxy.embedding import (
        HashEmbeddingProvider,
        LocalHTTPEmbeddingProvider,
        OpenAIEmbeddingProvider,
        SentenceTransformersEmbeddingProvider,
    )
    from zaxy.projection_backends import ProjectionBackendConfig

    settings = get_settings()
    provider_name = embedding_provider.casefold()
    provider: Any
    if provider_name == "openai":
        if not settings.openai_api_key:
            raise typer.BadParameter("OPENAI_API_KEY is required for OpenAI benchmarks")
        provider = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            base_url=settings.openai_base_url,
        )
        provider_label = f"openai:{settings.openai_embedding_model}"
    elif provider_name == "hash":
        provider = HashEmbeddingProvider(dimension=settings.embedding_dimension)
        provider_label = f"hash:{settings.embedding_dimension}"
    elif provider_name in {"local-http", "local_http", "http"}:
        if not settings.embedding_http_url:
            raise typer.BadParameter(
                "EMBEDDING_HTTP_URL is required for local-http benchmarks"
            )
        provider = LocalHTTPEmbeddingProvider(
            url=settings.embedding_http_url,
            model=settings.embedding_http_model,
            api_key=settings.embedding_http_api_key,
            dimension=settings.embedding_dimension,
        )
        label_model = settings.embedding_http_model or settings.embedding_http_url
        provider_label = f"local-http:{label_model}:{settings.embedding_dimension}"
    elif provider_name in {
        "sentence-transformers",
        "sentence_transformers",
        "sentence-transformer",
        "sentence_transformer",
        "local-model",
        "local_model",
    }:
        try:
            provider = SentenceTransformersEmbeddingProvider(
                model_name=settings.embedding_sentence_transformer_model,
                dimension=settings.embedding_dimension,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        provider_label = (
            "sentence-transformers:"
            f"{settings.embedding_sentence_transformer_model}:{settings.embedding_dimension}"
        )
    else:
        raise typer.BadParameter(
            "embedding provider must be 'openai', 'hash', 'local-http', "
            "or 'sentence-transformers'"
        )
    provider = cached_embedding_provider_cls(provider, cache_path=embedding_cache)

    async def _run() -> None:
        with tempfile.TemporaryDirectory(prefix="zaxy-live-benchmark-") as tmp:
            benchmark_workload: Any
            if workload == "fixture":
                eventlog = build_competitive_event_log(Path(tmp) / "bench.jsonl")
                cases = competitive_cases()
                benchmark_workload = benchmark_workload_cls.from_event_log(
                    eventlog,
                    cases,
                    version="fixture-v1",
                )
            elif workload == "statistical":
                eventlog, cases = build_statistical_event_log(
                    Path(tmp) / "bench.jsonl",
                    subjects=subjects,
                )
                benchmark_workload = benchmark_workload_cls.from_event_log(
                    eventlog,
                    cases,
                    version=f"statistical-subjects-{subjects}",
                    subjects=subjects,
                )
            elif workload == "frozen":
                eventlog, cases, benchmark_workload = build_frozen_statistical_workload(
                    Path(tmp) / "bench.jsonl"
                )
            elif workload == "suite":
                eventlog, cases, benchmark_workload = build_benchmark_suite_workload(
                    Path(tmp) / "bench.jsonl",
                    subjects=subjects,
                    documents=documents,
                    sessions=sessions,
                )
            elif workload == "consolidation":
                eventlog, cases, benchmark_workload = build_consolidation_collapse_workload(
                    Path(tmp) / "bench.jsonl",
                    identities=documents,
                )
            elif workload == "context-collapse":
                eventlog, cases, benchmark_workload = build_context_collapse_workload(
                    Path(tmp) / "bench.jsonl",
                    sessions=sessions,
                )
            elif workload == "graph-traversal":
                eventlog, cases, benchmark_workload = build_graph_traversal_workload(
                    Path(tmp) / "bench.jsonl",
                    subjects=subjects,
                )
            elif workload == "source-recall":
                eventlog, cases, benchmark_workload = build_source_recall_workload(
                    Path(tmp) / "bench.jsonl",
                    documents=documents,
                )
            elif workload == "temporal-recall":
                eventlog, cases, benchmark_workload = build_temporal_recall_workload(
                    Path(tmp) / "bench.jsonl",
                    subjects=subjects,
                )
            elif workload == "longmemeval":
                if dataset is None:
                    raise typer.BadParameter("--dataset is required for workload=longmemeval")
                eventlog, cases, benchmark_workload = build_longmemeval_workload(
                    Path(tmp) / "bench.jsonl",
                    dataset,
                    questions=questions,
                )
            else:
                # Unreachable while every name in BENCHMARK_WORKLOADS has a
                # branch above; it fires if a name is added to the tuple without
                # a builder, which is a coding error rather than user input.
                raise typer.BadParameter(
                    f"workload {workload!r} is registered but has no builder; "
                    "add a branch for it in zaxy.cli.benchmarks.benchmark"
                )
            corpus = corpus_from_event_log(eventlog)
            selected_baselines = _parse_benchmark_baselines(
                baseline_backends,
                allow_centroid=workload == "consolidation",
            )
            projection_cache_key = benchmark_projection_cache_key(
                eventlog,
                cases,
                benchmark_workload,
                provider_label,
            )
            projection_backend_config = ProjectionBackendConfig(
                backend=projection_backend,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_ca_cert=None,
                neo4j_trust_all=False,
                pggraph_dsn=pggraph_dsn or settings.pggraph_dsn,
                embedded_graph_path=Path(settings.embedded_graph_path),
                latticedb_path=Path(settings.latticedb_path),
                embedding_dimension=settings.embedding_dimension,
            )
            zaxy_retriever, graph = await build_live_zaxy_retriever(
                eventlog,
                provider,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                reset_graph=reset_graph,
                lexical_retriever=_build_source_lane_retriever(corpus, provider),
                reuse_projection=reuse_projection,
                projection_cache_key=projection_cache_key,
                scope_resolver=benchmark_query_scope_resolver(cases),
                projection_backend_config=projection_backend_config,
            )
            try:
                checkout_retriever: Any | None = None
                zaxy_backend_name = zaxy_backend.casefold()
                if zaxy_backend_name not in {"graph", "checkout", "both"}:
                    raise typer.BadParameter("--zaxy-backend must be graph, checkout, or both")
                if zaxy_backend_name in {"checkout", "both"}:
                    checkout_retriever = zaxy_retriever.as_checkout_retriever()
                retrievers = _build_benchmark_baselines(
                    corpus,
                    provider,
                    selected_baselines,
                )
                report = await benchmark_live_retrievers(
                    retrievers,
                    zaxy_retriever,
                    cases,
                    runs=runs,
                    limit=limit,
                    embedding_provider=provider_label,
                    workload=benchmark_workload,
                    external_results=_load_external_results(external_results),
                    checkout_retriever=checkout_retriever,
                    include_zaxy=zaxy_backend_name in {"graph", "both"},
                    progress_callback=(
                        lambda item: typer.echo(
                            (
                                f"progress {item['completed']}/{item['total']} "
                                f"{item['backend']} {item['case']} run={item['run']}"
                            ),
                            err=True,
                        )
                        if progress
                        else None
                    ),
                )
            finally:
                await graph.close()

        written = write_benchmark_report(report, output_dir)
        typer.echo(report_to_markdown(report))
        typer.echo(f"Wrote JSON report: {written.json_path}")
        typer.echo(f"Wrote Markdown report: {written.markdown_path}")

    try:
        asyncio.run(_run())
    finally:
        provider.flush()


@app.command("harvey-lab-benchmark")
def harvey_lab_benchmark(
    zaxy_results: Path = typer.Option(  # noqa: B008
        ...,
        "--zaxy-results",
        help="Harvey memory-ablation normalized-result JSON containing Zaxy rows",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/harvey-lab-memory-ablation"),
        "--output-dir",
        help="Directory for Harvey LAB benchmark JSON and Markdown reports",
    ),
) -> None:
    """Compare externally run Zaxy Harvey LAB rows with article-scored systems."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    build_harvey_lab_report = harvey_module.build_harvey_lab_report
    load_harvey_zaxy_results = harvey_module.load_harvey_zaxy_results
    report_to_markdown = harvey_module.report_to_markdown
    write_harvey_lab_report = harvey_module.write_harvey_lab_report

    try:
        zaxy_rows = load_harvey_zaxy_results(zaxy_results)
        report = build_harvey_lab_report(
            zaxy_rows,
            result_provenance={
                "source": "harvey-lab-benchmark",
                "zaxy_results_json_path": str(zaxy_results.resolve()),
            },
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    written = write_harvey_lab_report(report, output_dir)
    typer.echo(report_to_markdown(report))
    typer.echo(f"Harvey LAB external memory benchmark: {report.status}")
    typer.echo(f"Wrote JSON report: {written.json_path}")
    typer.echo(f"Wrote Markdown report: {written.markdown_path}")


@app.command("harvey-lab-import")
def harvey_lab_import(
    roots: list[Path] = typer.Argument(  # noqa: B008
        ...,
        help="External Harvey worktree, result directory, or normalized-result.json path",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/harvey-lab-memory-ablation"),
        "--output-dir",
        help="Directory for Harvey LAB benchmark JSON and Markdown reports",
    ),
    allow_baseline_only: bool = typer.Option(
        False,
        "--allow-baseline-only",
        help="Write a partial handoff report when only Harvey-native baseline comparison reports are present",
    ),
) -> None:
    """Import external Harvey normalized-result.json files and write Zaxy comparisons."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    build_harvey_lab_report = harvey_module.build_harvey_lab_report
    build_harvey_result_provenance = harvey_module.build_harvey_result_provenance
    import_harvey_zaxy_results = harvey_module.import_harvey_zaxy_results
    report_to_markdown = harvey_module.report_to_markdown
    write_harvey_lab_report = harvey_module.write_harvey_lab_report

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_roots = [*roots, output_dir]
    try:
        try:
            zaxy_rows = import_harvey_zaxy_results(roots)
        except ValueError:
            provenance = build_harvey_result_provenance(
                provenance_roots,
                source="harvey-lab-import",
            )
            if not allow_baseline_only or not provenance.get("external_baseline_report_paths"):
                raise
            zaxy_rows = ()
            baseline_only = True
        else:
            provenance = build_harvey_result_provenance(
                provenance_roots,
                source="harvey-lab-import",
            )
            baseline_only = False
        report = build_harvey_lab_report(
            zaxy_rows,
            result_provenance=provenance,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    written = write_harvey_lab_report(report, output_dir)
    typer.echo(report_to_markdown(report))
    typer.echo(f"Imported Zaxy Harvey LAB normalized results: {len(zaxy_rows)}")
    if baseline_only:
        typer.echo("Baseline-only handoff report: no Zaxy normalized results were imported.")
    typer.echo(f"Harvey LAB external memory benchmark: {report.status}")
    typer.echo(f"Wrote JSON report: {written.json_path}")
    typer.echo(f"Wrote Markdown report: {written.markdown_path}")


@app.command("harvey-lab-index")
def harvey_lab_index(
    normalized_corpus_root: Path = typer.Option(  # noqa: B008
        ...,
        "--normalized-corpus-root",
        help="Harvey normalized text corpus root, typically .ingestion/corpora/<hash>/txt",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path(".ingestion/indexes/zaxy"),
        "--output-dir",
        help="Directory for the Zaxy Eventloom index and manifest",
    ),
    source_map: Path | None = typer.Option(  # noqa: B008
        None,
        "--source-map",
        help="Optional Harvey source-map.json for original source path citations",
    ),
    max_lines: int = typer.Option(80, "--max-lines", min=1, help="Lines per indexed document chunk"),
) -> None:
    """Build an Eventloom-backed Zaxy memory index for Harvey LAB tools."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    build_harvey_zaxy_memory_index = harvey_module.build_harvey_zaxy_memory_index

    try:
        manifest = build_harvey_zaxy_memory_index(
            normalized_corpus_root,
            output_dir,
            source_map_path=source_map,
            max_lines=max_lines,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))
    typer.echo(f"Wrote Harvey LAB Zaxy manifest: {output_dir / 'manifest.json'}")


@app.command("harvey-lab-adapter-kit")
def harvey_lab_adapter_kit(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/harvey-lab-adapter-kit"),
        "--output-dir",
        help="Directory for Harvey LAB Zaxy adapter shim files",
    ),
) -> None:
    """Export a Harvey-compatible Zaxy memory adapter kit."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    export_harvey_adapter_kit = harvey_module.export_harvey_adapter_kit

    written = export_harvey_adapter_kit(output_dir)
    typer.echo(json.dumps(written, indent=2, sort_keys=True))


@app.command("harvey-lab-ready")
def harvey_lab_ready(
    harvey_worktree: Path = typer.Argument(  # noqa: B008
        ...,
        help="External Harvey worktree to check before launching model-backed Zaxy runs",
    ),
    generator: str = typer.Option(
        "HARVEY_GENERATOR_MODEL",
        "--generator",
        help="Generator model planned for Harvey harness runs",
    ),
    judge: str = typer.Option(
        "HARVEY_JUDGE_MODEL",
        "--judge",
        help="Judge model planned for Harvey evaluation runs",
    ),
    task_filter: str | None = typer.Option(
        None,
        "--task-filter",
        help="Optional task id, slug, or run id for filtered external runs",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Check external Harvey run prerequisites without launching model calls."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    build_harvey_external_run_readiness = harvey_module.build_harvey_external_run_readiness

    _ = json_output
    readiness = build_harvey_external_run_readiness(
        harvey_worktree,
        generator=generator,
        judge=judge,
        task_filter=task_filter,
    )
    typer.echo(json.dumps(readiness, indent=2, sort_keys=True))
    if readiness["status"] != "ready_for_external_runs":
        raise typer.Exit(1)


@app.command("harvey-lab-plan")
def harvey_lab_plan(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/harvey-lab-memory-ablation"),
        "--output-dir",
        help="Directory for the external Harvey run manifest",
    ),
    generator: str = typer.Option(
        "HARVEY_GENERATOR_MODEL",
        "--generator",
        help="Generator model to record in the external Harvey plan",
    ),
    judge: str = typer.Option(
        "HARVEY_JUDGE_MODEL",
        "--judge",
        help="Judge model to record in the external Harvey plan",
    ),
    reasoning_effort: str | None = typer.Option(  # noqa: B008
        "low",
        "--reasoning-effort",
        help="Generator reasoning effort to record; use empty string for none",
    ),
) -> None:
    """Write a reproducible external Harvey LAB run manifest."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    build_harvey_external_run_manifest = harvey_module.build_harvey_external_run_manifest
    write_harvey_external_run_manifest = harvey_module.write_harvey_external_run_manifest

    manifest = build_harvey_external_run_manifest(
        generator=generator,
        judge=judge,
        reasoning_effort=reasoning_effort or None,
    )
    written = write_harvey_external_run_manifest(manifest, output_dir)
    typer.echo(f"Wrote Harvey LAB external run JSON: {written.json_path}")
    typer.echo(f"Wrote Harvey LAB external run Markdown: {written.markdown_path}")
    typer.echo(f"Wrote Harvey LAB external run script: {written.script_path}")


@app.command("harvey-lab-normalize-run")
def harvey_lab_normalize_run(
    harvey_worktree: Path = typer.Option(  # noqa: B008
        ...,
        "--harvey-worktree",
        help="External Harvey worktree containing results/<run-id>",
    ),
    run_id: str = typer.Option(..., "--run-id", help="Harvey run id to normalize"),
    task_id: str = typer.Option(..., "--task-id", help="Harvey task id for the run"),
    manifest: Path = typer.Option(  # noqa: B008
        ...,
        "--manifest",
        help="Zaxy Harvey memory manifest used for the run",
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        help="Optional output path; defaults to .ingestion/runs/<run-id>/normalized-result.json",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        help="Override judge model if scores.json does not record it",
    ),
    judge_reasoning_effort: str | None = typer.Option(  # noqa: B008
        None,
        "--judge-reasoning-effort",
        help="Judge reasoning effort to record",
    ),
) -> None:
    """Write Harvey normalized-result.json from one external Zaxy run."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    build_harvey_normalized_result_from_run = harvey_module.build_harvey_normalized_result_from_run

    try:
        normalized = build_harvey_normalized_result_from_run(
            harvey_worktree,
            run_id=run_id,
            task_id=task_id,
            manifest_path=manifest,
            judge_model=judge_model,
            judge_reasoning_effort=judge_reasoning_effort,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_path = output or harvey_worktree / ".ingestion" / "runs" / run_id / "normalized-result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"Wrote Harvey LAB normalized result: {output_path}")


@app.command("harvey-lab-gate")
def harvey_lab_gate(
    report_path: Path = typer.Argument(..., help="harvey-lab-benchmark.json report"),  # noqa: B008
) -> None:
    """Gate public Harvey LAB claims on complete external Zaxy results."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    check_harvey_lab_completion = harvey_module.check_harvey_lab_completion
    load_harvey_lab_report = harvey_module.load_harvey_lab_report

    try:
        report = load_harvey_lab_report(report_path)
        gate = check_harvey_lab_completion(report)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(gate, indent=2, sort_keys=True))
    if gate["status"] != "passed":
        raise typer.Exit(1)


@app.command("harvey-lab-validate")
def harvey_lab_validate(
    report_path: Path = typer.Argument(..., help="harvey-lab-benchmark.json report"),  # noqa: B008
    require_complete: bool = typer.Option(
        False,
        "--require-complete",
        help="Require all ten article tasks in addition to evidence validation",
    ),
) -> None:
    """Validate Harvey LAB report evidence and local artifact availability."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    load_harvey_lab_report = harvey_module.load_harvey_lab_report
    validate_harvey_lab_report = harvey_module.validate_harvey_lab_report

    try:
        report = load_harvey_lab_report(report_path)
        validation = validate_harvey_lab_report(
            report,
            require_complete=require_complete,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(validation, indent=2, sort_keys=True))
    if validation["status"] != "valid":
        raise typer.Exit(1)


@app.command("harvey-lab-publish")
def harvey_lab_publish(
    report_path: Path = typer.Argument(..., help="harvey-lab-benchmark.json report"),  # noqa: B008
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        help="Optional Markdown output path for publishable comparative statistics",
    ),
) -> None:
    """Render publishable Harvey LAB statistics after the strict gate passes."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    load_harvey_lab_report = harvey_module.load_harvey_lab_report
    render_harvey_publication_markdown = harvey_module.render_harvey_publication_markdown

    try:
        report = load_harvey_lab_report(report_path)
        markdown = render_harvey_publication_markdown(report)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if output is None:
        typer.echo(markdown)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    typer.echo(f"Wrote Harvey LAB publishable statistics: {output}")


@app.command("longmembench-bootstrap")
def longmembench_bootstrap(
    worktree: Path = typer.Option(  # noqa: B008
        Path(".cache/zaxy/benchmarks/LongMemEval"),
        "--worktree",
        help="Official LongMemEval checkout path to create or reuse",
    ),
    repo_url: str = typer.Option(
        "https://github.com/xiaowu0162/LongMemEval",
        "--repo-url",
        help="Official LongMemEval repository URL",
    ),
    ref: str | None = typer.Option(
        None,
        "--ref",
        help="Optional git ref to checkout after clone",
    ),
    dataset_source: Path | None = typer.Option(  # noqa: B008
        None,
        "--dataset-source",
        help="Optional local longmemeval_oracle.json to copy instead of downloading",
    ),
    force_dataset: bool = typer.Option(
        False,
        "--force-dataset",
        help="Overwrite existing oracle dataset",
    ),
) -> None:
    """Clone official LongMemEval and install the oracle dataset."""
    longmembench_module = _benchmark_module("longmembench")
    bootstrap_longmemeval_official_suite = longmembench_module.bootstrap_longmemeval_official_suite

    try:
        result = bootstrap_longmemeval_official_suite(
            worktree=worktree,
            repo_url=repo_url,
            ref=ref,
            dataset_source=dataset_source,
            force_dataset=force_dataset,
        )
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "ready":
        raise typer.Exit(1)


@app.command("longmembench-adapter-kit")
def longmembench_adapter_kit(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-adapter-kit"),
        "--output-dir",
        help="Directory for LongMemBench adapter-kit files",
    ),
) -> None:
    """Export the official LongMemEval hypothesis/evaluation adapter kit."""
    longmembench_module = _benchmark_module("longmembench")
    export_longmembench_adapter_kit = longmembench_module.export_longmembench_adapter_kit

    written = export_longmembench_adapter_kit(output_dir)
    typer.echo(json.dumps(written, indent=2, sort_keys=True))


@app.command("longmembench-plan")
def longmembench_plan(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external"),
        "--output-dir",
        help="Directory for LongMemBench external run manifest",
    ),
    dataset: str = typer.Option(
        "data/longmemeval_oracle.json",
        "--dataset",
        help="Official LongMemEval dataset path inside the external worktree",
    ),
    evaluator_model: str = typer.Option(
        "gpt-4o",
        "--evaluator-model",
        help="Official LongMemEval evaluator model to record",
    ),
) -> None:
    """Write a reproducible external LongMemBench run manifest."""
    longmembench_module = _benchmark_module("longmembench")
    build_longmembench_external_run_manifest = (
        longmembench_module.build_longmembench_external_run_manifest
    )
    write_longmembench_external_run_manifest = (
        longmembench_module.write_longmembench_external_run_manifest
    )

    manifest = build_longmembench_external_run_manifest(
        dataset=dataset,
        evaluator_model=evaluator_model,
        output_dir=str(output_dir),
    )
    written = write_longmembench_external_run_manifest(manifest, output_dir)
    typer.echo(f"Wrote LongMemBench external run JSON: {written.json_path}")
    typer.echo(f"Wrote LongMemBench external run Markdown: {written.markdown_path}")
    typer.echo(f"Wrote LongMemBench external run script: {written.script_path}")


@app.command("longmembench-ready")
def longmembench_ready(
    longmemeval_worktree: Path | None = typer.Option(  # noqa: B008
        None,
        "--longmemeval-worktree",
        help="External official LongMemEval worktree",
    ),
    dataset: Path | None = typer.Option(  # noqa: B008
        None,
        "--dataset",
        help="Official LongMemEval dataset path",
    ),
    hypotheses: Path | None = typer.Option(  # noqa: B008
        None,
        "--hypotheses",
        help="Generated official hypothesis JSONL",
    ),
    official_eval_log: Path | None = typer.Option(  # noqa: B008
        None,
        "--official-eval-log",
        help="Official LongMemEval evaluate_qa.py JSONL log",
    ),
    diagnostic_report: Path | None = typer.Option(  # noqa: B008
        None,
        "--diagnostic-report",
        help="Optional Zaxy LongMemEval-compatible live-benchmark.json report",
    ),
    sota_baseline: Path | None = typer.Option(  # noqa: B008
        None,
        "--sota-baseline",
        help="External SOTA baseline JSON for strict SOTA claims",
    ),
    answer_mode: str = typer.Option(
        "openai-compatible",
        "--answer-mode",
        help="Planned hypothesis answer mode",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key presence override; defaults to OPENAI_API_KEY",
    ),
    require_sota_baseline: bool = typer.Option(
        True,
        "--require-sota-baseline/--no-require-sota-baseline",
        help="Require an external SOTA baseline in readiness",
    ),
) -> None:
    """Check readiness for official LongMemBench launch and SOTA claims."""

    longmembench_module = _benchmark_module("longmembench")
    build_longmembench_readiness = longmembench_module.build_longmembench_readiness

    readiness = build_longmembench_readiness(
        longmemeval_worktree=longmemeval_worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=official_eval_log,
        diagnostic_report_path=diagnostic_report,
        sota_baseline_path=sota_baseline,
        answer_mode=answer_mode,
        api_key_present=bool(api_key or os.getenv("OPENAI_API_KEY")),
        require_sota_baseline=require_sota_baseline,
    )
    typer.echo(json.dumps(readiness, indent=2, sort_keys=True))
    if readiness["status"] != "ready":
        raise typer.Exit(1)


@app.command("longmembench-import")
def longmembench_import(
    longmemeval_worktree: Path = typer.Option(  # noqa: B008
        ...,
        "--longmemeval-worktree",
        help="External official LongMemEval worktree",
    ),
    dataset: Path = typer.Option(  # noqa: B008
        ...,
        "--dataset",
        help="Official LongMemEval dataset used by the run",
    ),
    hypotheses: Path | None = typer.Option(  # noqa: B008
        None,
        "--hypotheses",
        help="Official hypothesis JSONL with question_id and hypothesis rows",
    ),
    official_eval_log: Path | None = typer.Option(  # noqa: B008
        None,
        "--official-eval-log",
        help="Official LongMemEval evaluate_qa.py JSONL log",
    ),
    diagnostic_report: Path | None = typer.Option(  # noqa: B008
        None,
        "--diagnostic-report",
        help="Optional Zaxy LongMemEval-compatible live-benchmark.json report",
    ),
    sota_baseline: Path | None = typer.Option(  # noqa: B008
        None,
        "--sota-baseline",
        help="External SOTA baseline JSON for strict SOTA comparison",
    ),
    evaluator_model: str | None = typer.Option(
        None,
        "--evaluator-model",
        help="Official evaluator model used by evaluate_qa.py",
    ),
    official_eval_command: str | None = typer.Option(
        None,
        "--official-eval-command",
        help="Exact official evaluate_qa.py command used by the validator",
    ),
    validator_evidence: Path | None = typer.Option(  # noqa: B008
        None,
        "--validator-evidence",
        help="Completed validator-evidence-template.json from an independent validator",
    ),
    validator_name: str | None = typer.Option(
        None,
        "--validator-name",
        help="Independent validator name for externally validated SOTA claims",
    ),
    validator_evidence_url: str | None = typer.Option(
        None,
        "--validator-evidence-url",
        help="Reviewable external validation URL",
    ),
    validator_run_id: str | None = typer.Option(
        None,
        "--validator-run-id",
        help="Independent validator run identifier",
    ),
    validator_relation: str | None = typer.Option(
        None,
        "--validator-relation",
        help="Relationship to Zaxy, for example independent-third-party",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external"),
        "--output-dir",
        help="Directory for LongMemBench report artifacts",
    ),
) -> None:
    """Import official LongMemEval QA evidence and Zaxy diagnostics."""
    longmembench_module = _benchmark_module("longmembench")
    build_longmembench_report = longmembench_module.build_longmembench_report
    load_validator_evidence = longmembench_module.load_validator_evidence
    report_to_markdown = longmembench_module.report_to_markdown
    validate_validator_evidence_matches_report = (
        longmembench_module.validate_validator_evidence_matches_report
    )
    validator_official_evaluation_metadata = (
        longmembench_module.validator_official_evaluation_metadata
    )
    validator_provenance_from_evidence = longmembench_module.validator_provenance_from_evidence
    write_longmembench_report = longmembench_module.write_longmembench_report

    try:
        evidence_payload = load_validator_evidence(validator_evidence) if validator_evidence else None
        if evidence_payload is not None:
            metadata = validator_official_evaluation_metadata(evidence_payload)
            evaluator_model = evaluator_model or metadata["evaluator_model"]
            official_eval_command = official_eval_command or metadata["official_eval_command"]
            validator_provenance = validator_provenance_from_evidence(
                evidence_payload,
                validator_name=validator_name,
                validator_evidence_url=validator_evidence_url,
                validator_run_id=validator_run_id,
                validator_relation=validator_relation,
            )
        elif any(
            value is not None
            for value in (validator_name, validator_evidence_url, validator_run_id, validator_relation)
        ):
            validator_provenance = validator_provenance_from_evidence(
                {"validator": {}},
                validator_name=validator_name,
                validator_evidence_url=validator_evidence_url,
                validator_run_id=validator_run_id,
                validator_relation=validator_relation,
            )
        else:
            validator_provenance = {"validator": None}
        validator_evidence_verified = False
        report = build_longmembench_report(
            longmemeval_worktree=longmemeval_worktree,
            dataset_path=dataset,
            hypotheses_path=hypotheses,
            official_eval_log_path=official_eval_log,
            diagnostic_report_path=diagnostic_report,
            sota_baseline_path=sota_baseline,
            evaluator_model=evaluator_model,
            official_eval_command=official_eval_command,
            result_provenance={
                "source": "longmembench-import",
                "dataset": str(dataset.resolve()),
                "hypotheses": str(hypotheses.resolve()) if hypotheses else None,
                "official_eval_log": (
                    str(official_eval_log.resolve()) if official_eval_log else None
                ),
                "diagnostic_report": (
                    str(diagnostic_report.resolve()) if diagnostic_report else None
                ),
                "sota_baseline": str(sota_baseline.resolve()) if sota_baseline else None,
                "validator_evidence": str(validator_evidence.resolve()) if validator_evidence else None,
                "validator_evidence_verified": validator_evidence_verified,
                **validator_provenance,
            },
        )
        if evidence_payload is not None:
            evidence_failures = validate_validator_evidence_matches_report(evidence_payload, report)
            if evidence_failures:
                raise ValueError("; ".join(evidence_failures))
            report.result_provenance["validator_evidence_verified"] = True
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    written = write_longmembench_report(report, output_dir)
    typer.echo(report_to_markdown(report))
    typer.echo(f"LongMemBench external validation: {report.status}")
    typer.echo(f"Wrote JSON report: {written.json_path}")
    typer.echo(f"Wrote Markdown report: {written.markdown_path}")


@app.command("longmembench-generate-hypotheses")
def longmembench_generate_hypotheses(
    dataset: Path = typer.Option(  # noqa: B008
        ...,
        "--dataset",
        help="Official LongMemEval dataset used to generate hypotheses",
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl"),
        "--output",
        help="Output official hypothesis JSONL path",
    ),
    report: Path | None = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/zaxy-hypotheses-report.json"),
        "--report",
        help="Optional machine-readable hypothesis-generation report",
    ),
    questions: int | None = typer.Option(  # noqa: B008
        None,
        "--questions",
        min=1,
        help="Optional question limit for smoke runs; omit for full official set",
    ),
    limit: int = typer.Option(10, "--limit", min=1, max=50, help="Checkout contexts per question"),
    answer_mode: str = typer.Option(
        "extractive",
        "--answer-mode",
        help="Answer mode: extractive or openai-compatible",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="OpenAI-compatible chat model for answer generation",
    ),
    base_url: str = typer.Option(
        "https://api.openai.com/v1",
        "--base-url",
        help="OpenAI-compatible chat-completions base URL",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="OpenAI-compatible API key; defaults to OPENAI_API_KEY when omitted",
    ),
    embedding_provider: str = typer.Option(
        "hash",
        "--embedding-provider",
        help="Embedding provider for Zaxy retrieval: hash or openai",
    ),
    embedding_cache: Path | None = typer.Option(  # noqa: B008
        None,
        "--embedding-cache",
        help="Optional embedding cache path",
    ),
    projection_backend: str = typer.Option(
        "embedded",
        "--projection-backend",
        help="Projection backend for Zaxy retrieval",
    ),
    reuse_projection: bool = typer.Option(
        False,
        "--reuse-projection",
        help="Reuse projection cache when supported",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Append missing hypotheses and skip existing question_id rows in the output JSONL",
    ),
    fsync_rows: bool = typer.Option(
        False,
        "--fsync-rows",
        help="Fsync each generated hypothesis row for crash-resistant external runs",
    ),
    provider_retries: int = typer.Option(
        3,
        "--provider-retries",
        min=0,
        max=10,
        help="Retries for transient OpenAI-compatible 429/5xx answer-generation failures",
    ),
    prefer_checkout_candidate: bool = typer.Option(
        False,
        "--prefer-checkout-candidate",
        help="Use checkout answer candidates directly before falling back to the answer model",
    ),
    filter_answer_contexts: bool = typer.Option(
        False,
        "--filter-answer-contexts",
        help="Experimental: remove checkout diagnostics from answer-model prompts",
    ),
) -> None:
    """Generate official LongMemEval hypothesis JSONL rows through Zaxy checkout."""
    import asyncio

    longmembench_module = _benchmark_module("longmembench")
    generate_longmembench_hypotheses = longmembench_module.generate_longmembench_hypotheses

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def _run() -> None:
        try:
            generated = await generate_longmembench_hypotheses(
                dataset_path=dataset,
                output_path=output,
                report_path=report,
                questions=questions,
                limit=limit,
                answer_mode=answer_mode,
                model=model,
                base_url=base_url,
                api_key=resolved_api_key,
                embedding_provider=embedding_provider,
                embedding_cache=embedding_cache,
                projection_backend=projection_backend,
                reuse_projection=reuse_projection,
                resume=resume,
                fsync_rows=fsync_rows,
                provider_retries=provider_retries,
                prefer_checkout_candidate=prefer_checkout_candidate,
                filter_answer_contexts=filter_answer_contexts,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(
            json.dumps(
                {
                    "status": "complete",
                    "question_count": generated.question_count,
                    "output": generated.output_path,
                    "report": str(report) if report is not None else None,
                    "answer_mode": generated.answer_mode,
                    "model": generated.model,
                },
                indent=2,
                sort_keys=True,
            )
        )

    asyncio.run(_run())


@app.command("longmembench-evaluate-official")
def longmembench_evaluate_official(
    longmemeval_worktree: Path = typer.Option(  # noqa: B008
        ...,
        "--longmemeval-worktree",
        help="External official LongMemEval worktree",
    ),
    hypotheses: Path = typer.Option(  # noqa: B008
        ...,
        "--hypotheses",
        help="Generated official hypothesis JSONL",
    ),
    dataset: Path = typer.Option(  # noqa: B008
        ...,
        "--dataset",
        help="Official LongMemEval dataset path",
    ),
    evaluator_model: str = typer.Option(
        "gpt-4o",
        "--evaluator-model",
        help="Official evaluator model passed to evaluate_qa.py",
    ),
    output_log: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-log",
        help="Optional path to copy the official evaluator JSONL log",
    ),
    run_report: Path | None = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/official-eval-run.json"),
        "--run-report",
        help="Optional JSON report for the official evaluator subprocess",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key presence override; defaults to OPENAI_API_KEY",
    ),
    require_api_key: bool = typer.Option(
        True,
        "--require-api-key/--no-require-api-key",
        help="Require API key before running official evaluator",
    ),
) -> None:
    """Run LongMemEval's official evaluate_qa.py over Zaxy hypotheses."""

    longmembench_module = _benchmark_module("longmembench")
    run_longmemeval_official_eval = longmembench_module.run_longmemeval_official_eval

    try:
        result = run_longmemeval_official_eval(
            worktree=longmemeval_worktree,
            hypotheses_path=hypotheses,
            dataset_path=dataset,
            evaluator_model=evaluator_model,
            output_log=output_log,
            require_api_key=require_api_key,
            api_key_present=bool(api_key or os.getenv("OPENAI_API_KEY")),
            api_key=api_key,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = asdict(result)
    if run_report is not None:
        run_report.parent.mkdir(parents=True, exist_ok=True)
        run_report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    if result.status != "complete":
        raise typer.Exit(1)


GRAPH_SCALE_LANE_NAMES: tuple[str, ...] = ("graph-walk", "vector-scale")


def _parse_graph_scale_lanes(value: str) -> tuple[str, ...]:
    """Parse the graph/scale lane selection string."""
    normalized = value.strip().casefold()
    if normalized == "all":
        return GRAPH_SCALE_LANE_NAMES
    selected = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not selected:
        raise typer.BadParameter("--lanes must include at least one lane")
    invalid = sorted(set(selected) - set(GRAPH_SCALE_LANE_NAMES))
    if invalid:
        valid_text = ", ".join((*GRAPH_SCALE_LANE_NAMES, "all"))
        raise typer.BadParameter(
            f"Unsupported graph/scale lane(s): {', '.join(invalid)}. Allowed: {valid_text}"
        )
    return selected


def _parse_vector_scale_sizes(value: str) -> tuple[int, ...]:
    """Parse the vector-scale corpus-size sweep string."""
    sizes: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            size = int(token)
        except ValueError as exc:
            raise typer.BadParameter(
                f"invalid corpus size {token!r}; use positive integers"
            ) from exc
        if size <= 0:
            raise typer.BadParameter("corpus sizes must be positive integers")
        sizes.append(size)
    if not sizes:
        raise typer.BadParameter("--scale-sizes must include at least one corpus size")
    return tuple(sizes)


@app.command("graph-scale-lanes")
def graph_scale_lanes(
    lanes: str = typer.Option(
        "all",
        "--lanes",
        help="Comma-separated internal lanes to run: graph-walk, vector-scale, or all",
    ),
    scale_sizes: str = typer.Option(
        "1000,10000",
        "--scale-sizes",
        help=(
            "Vector-scale corpus sizes, comma-separated positive integers. "
            "100000 is opt-in: the HNSW shadow sync takes minutes at 10^5."
        ),
    ),
    scale_dimension: int = typer.Option(
        64,
        "--scale-dimension",
        min=1,
        help="Embedding dimension for the vector-scale corpora",
    ),
    scale_distribution: str = typer.Option(
        "hash",
        "--scale-distribution",
        help=(
            "Vector distribution for the vector-scale corpora: hash "
            "(deterministic hash embeddings, the comparability baseline) or "
            "gaussian (seeded unit-normalized standard normal, the "
            "realistic-distribution control used for high-dimension gates)"
        ),
    ),
    ann_threshold: int = typer.Option(
        256,
        "--ann-threshold",
        min=1,
        help="Lowered vector_ann_threshold so the HNSW path engages at lane sizes",
    ),
    query_count: int = typer.Option(
        32,
        "--query-count",
        min=1,
        help="Fixed query-set size for vector-scale recall and latency",
    ),
    latency_passes: int = typer.Option(
        3,
        "--latency-passes",
        min=1,
        help="Timed passes over the query set per mode (latency samples)",
    ),
    walk_top_k: int = typer.Option(
        5,
        "--walk-top-k",
        min=1,
        help="Top-k window for graph-walk bridge and direct-hit membership",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Optional directory for the graph-scale-lanes.json report",
    ),
) -> None:
    """Run the internal PPR graph-walk and vector-scale lanes."""
    selected_lanes = _parse_graph_scale_lanes(lanes)
    sizes = _parse_vector_scale_sizes(scale_sizes)
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="zaxy-graph-scale-lanes-") as tmp:
        try:
            if "graph-walk" in selected_lanes:
                graph_walk_module = _benchmark_module("graph_walk_lane")
                results["graph_walk"] = graph_walk_module.run_graph_walk_lane(
                    Path(tmp) / "graph-walk",
                    top_k=walk_top_k,
                )
            if "vector-scale" in selected_lanes:
                vector_scale_module = _benchmark_module("vector_scale_lane")
                results["vector_scale"] = vector_scale_module.run_vector_scale_lane(
                    Path(tmp) / "vector-scale",
                    sizes=sizes,
                    dimension=scale_dimension,
                    distribution=scale_distribution,
                    ann_threshold=ann_threshold,
                    query_count=query_count,
                    latency_passes=latency_passes,
                )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    payload = {
        "version": "graph-scale-lanes-v1",
        "validation": "internal",
        "lanes": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(rendered)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "graph-scale-lanes.json"
        report_path.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"Wrote JSON report: {report_path}", err=True)
