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
from pathlib import Path

import typer

from zaxy.event import EventLog
from zaxy.graph import GraphStore
from zaxy.mcp_server import main as mcp_main

app = typer.Typer(help="Zaxy: Event-sourced temporal knowledge graph fabric")


@app.command()
def serve(
    eventloom_path: str | None = typer.Option(None, help="Directory for event logs"),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    transport: str = typer.Option("stdio", help="Transport: stdio or sse"),
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
        asyncio.run(mcp_server.main_sse(port=port))
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
