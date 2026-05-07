"""CLI entrypoint for Zaxy.

Commands:
    serve       Start the MCP server over stdio or SSE.
    replay      Replay an Eventloom log and print integrity report.
    compact     Compact an Eventloom log (create snapshot).
    status      Check connectivity to Neo4j and Pathlight.

Example::

    python -m zaxy serve
    python -m zaxy replay .eventloom/work.jsonl
    python -m zaxy status
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import typer

from zaxy.benchmark import build_competitive_event_log, competitive_cases
from zaxy.embedding import EmbeddingProvider, HashEmbeddingProvider, OpenAIEmbeddingProvider
from zaxy.event import EventLog
from zaxy.graph import GraphStore
from zaxy.live_benchmark import (
    MarkdownRetriever,
    MarkdownVectorRetriever,
    VectorRetriever,
    benchmark_live_retrievers,
    build_live_zaxy_retriever,
    build_statistical_event_log,
    corpus_from_event_log,
    report_to_markdown,
    write_benchmark_report,
)
from zaxy.mcp_server import main as mcp_main

app = typer.Typer(help="Zaxy: Event-sourced temporal knowledge graph fabric")


@app.command()
def serve(
    eventloom_path: str | None = typer.Option(None, help="Directory for event logs"),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    transport: str = typer.Option("stdio", help="Transport: stdio or sse"),
    host: str = typer.Option("127.0.0.1", help="Host for SSE transport"),
    port: int = typer.Option(8080, help="Port for SSE transport"),
) -> None:
    """Start the MCP server (stdio or sse)."""
    import asyncio

    from zaxy import mcp_server

    # Configure the module-level server instance from CLI overrides
    mcp_server.server = mcp_server.ZaxyMCPServer(
        eventloom_path=eventloom_path,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    if transport == "sse":
        asyncio.run(mcp_server.main_sse(port=port, host=host))
    else:
        asyncio.run(mcp_main())


@app.command()
def replay(
    log_path: Path = typer.Argument(..., help="Path to Eventloom JSONL file"),  # noqa: B008
    from_seq: int = typer.Option(1, help="Start sequence number"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Replay an Eventloom log and print integrity report + events."""
    log = EventLog(str(log_path))
    result = log.replay(from_seq=from_seq)

    if json_output:
        output = {
            "integrity": result.integrity.model_dump(),
            "events": [e.model_dump() for e in result.events],
        }
        print(json.dumps(output, indent=2))
        return

    typer.echo(f"Integrity: {'OK' if result.integrity.ok else 'FAILED'}")
    typer.echo(f"Total events: {result.integrity.total_events}")
    if result.integrity.broken_at_seq:
        typer.echo(f"Broken at seq: {result.integrity.broken_at_seq}")
        typer.echo(f"Reason: {result.integrity.broken_reason}")

    for ev in result.events:
        typer.echo(f"  [{ev.seq}] {ev.timestamp} {ev.type} by {ev.actor}")

    summary = log.handoff_summary()
    typer.echo("\nHandoff summary:")
    typer.echo(f"  Goals: {summary['goals']}")
    typer.echo(f"  Open tasks: {len(summary['open_tasks'])}")
    typer.echo(f"  Last actor: {summary['last_actor']}")


@app.command()
def compact(
    log_path: Path = typer.Argument(..., help="Path to Eventloom JSONL file"),  # noqa: B008
    snapshot_every: int = typer.Option(10000, help="Create snapshot every N events"),
    output: Path = typer.Option(None, help="Output path (default: in-place)"),  # noqa: B008
) -> None:
    """Compact an Eventloom log and optionally create snapshots."""
    log = EventLog(str(log_path))
    events = log.read_all()
    total = len(events)

    if total == 0:
        typer.echo("Log is empty. Nothing to compact.")
        raise typer.Exit(0)

    out_path = output or log_path
    with open(out_path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(ev.model_dump_json() + "\n")

    typer.echo(f"Compacted {total} events -> {out_path}")

    if total >= snapshot_every:
        snapshot_path = log_path.with_suffix(f".snapshot-{total}.json")
        with open(snapshot_path, "w", encoding="utf-8") as fh:
            for ev in events[-snapshot_every:]:
                fh.write(ev.model_dump_json() + "\n")
        typer.echo(f"Created snapshot: {snapshot_path}")


@app.command()
def status(
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    pathlight_url: str | None = typer.Option(None, help="Pathlight collector URL"),
) -> None:
    """Check connectivity to external services."""
    import asyncio

    from zaxy.config import get_settings

    settings = get_settings()

    async def _check() -> None:
        ok = True
        _uri = neo4j_uri or settings.neo4j_uri
        _user = neo4j_user or settings.neo4j_user
        _password = neo4j_password or settings.neo4j_password
        _pathlight = pathlight_url or settings.pathlight_url

        # Neo4j
        try:
            gs = GraphStore(_uri, _user, _password)
            await gs.connect()
            assert gs._driver is not None
            await gs._driver.execute_query("RETURN 1 AS n")
            await gs.close()
            typer.echo(f"Neo4j:     OK ({_uri})")
        except Exception as exc:
            typer.echo(f"Neo4j:     FAIL ({exc})")
            ok = False

        # Pathlight is optional. Only fail health checks when explicitly enabled.
        if settings.pathlight_enabled:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{_pathlight}/health")
                    if resp.status_code == 200:
                        typer.echo(f"Pathlight: OK ({_pathlight})")
                    else:
                        typer.echo(f"Pathlight: FAIL (HTTP {resp.status_code})")
                        ok = False
            except Exception as exc:
                typer.echo(f"Pathlight: FAIL ({exc})")
                ok = False
        else:
            typer.echo("Pathlight: SKIP (disabled)")

        raise typer.Exit(0 if ok else 1)

    asyncio.run(_check())


@app.command()
def benchmark(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks"),
        help="Directory for JSON and Markdown benchmark reports",
    ),
    embedding_provider: str = typer.Option("openai", help="Embedding provider: openai or hash"),
    runs: int = typer.Option(5, min=1, help="Measured runs per backend/case"),
    limit: int = typer.Option(10, min=1, max=50, help="Returned contexts per query"),
    neo4j_uri: str = typer.Option("bolt://localhost:7688", help="Benchmark Neo4j Bolt URI"),
    neo4j_user: str = typer.Option("neo4j", help="Benchmark Neo4j username"),
    neo4j_password: str = typer.Option("testpassword", help="Benchmark Neo4j password"),
    reset_graph: bool = typer.Option(
        False,
        help="Delete Entity nodes in the benchmark Neo4j database before ingestion",
    ),
    workload: str = typer.Option("fixture", help="Workload: fixture or statistical"),
    subjects: int = typer.Option(
        100,
        min=1,
        help="Subject count for the statistical workload; each subject creates 3 queries",
    ),
) -> None:
    """Run live retrieval benchmarks against md/vector/md+vector/Zaxy."""
    import asyncio

    from zaxy.config import get_settings

    settings = get_settings()
    provider_name = embedding_provider.casefold()
    provider: EmbeddingProvider
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
    else:
        raise typer.BadParameter("embedding provider must be 'openai' or 'hash'")

    async def _run() -> None:
        with tempfile.TemporaryDirectory(prefix="zaxy-live-benchmark-") as tmp:
            if workload == "fixture":
                eventlog = build_competitive_event_log(Path(tmp) / "bench.jsonl")
                cases = competitive_cases()
            elif workload == "statistical":
                eventlog, cases = build_statistical_event_log(
                    Path(tmp) / "bench.jsonl",
                    subjects=subjects,
                )
            else:
                raise typer.BadParameter("workload must be 'fixture' or 'statistical'")
            corpus = corpus_from_event_log(eventlog)
            zaxy_retriever, graph = await build_live_zaxy_retriever(
                eventlog,
                provider,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                reset_graph=reset_graph,
            )
            try:
                report = await benchmark_live_retrievers(
                    {
                        "md": MarkdownRetriever(corpus),
                        "vector": VectorRetriever(corpus, provider),
                        "md+vector": MarkdownVectorRetriever(corpus, provider),
                    },
                    zaxy_retriever,
                    cases,
                    runs=runs,
                    limit=limit,
                    embedding_provider=provider_label,
                )
            finally:
                await graph.close()

        written = write_benchmark_report(report, output_dir)
        typer.echo(report_to_markdown(report))
        typer.echo(f"Wrote JSON report: {written.json_path}")
        typer.echo(f"Wrote Markdown report: {written.markdown_path}")

    asyncio.run(_run())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
