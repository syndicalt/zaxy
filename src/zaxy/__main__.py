"""CLI entrypoint for Zaxy.

Commands:
    serve       Start the MCP stdio server.
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
from zaxy.trace import MemoryTracer

app = typer.Typer(help="Zaxy: Event-sourced temporal knowledge graph fabric")


@app.command()
def serve(
    eventloom_path: str = typer.Option(".eventloom", help="Directory for event logs"),
    neo4j_uri: str = typer.Option("bolt://localhost:7687", help="Neo4j Bolt URI"),
    neo4j_user: str = typer.Option("neo4j", help="Neo4j username"),
    neo4j_password: str = typer.Option("testpassword", help="Neo4j password"),
) -> None:
    """Start the MCP stdio server."""
    import asyncio

    # Patch server config before starting
    from zaxy import mcp_server

    mcp_server.ZaxyMCPServer.__init__ = lambda self: None  # type: ignore[method-assign, assignment, misc]
    obj = mcp_server.ZaxyMCPServer.__new__(mcp_server.ZaxyMCPServer)
    obj.eventloom_path = eventloom_path
    obj.graph = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
    obj.tracer = MemoryTracer()
    mcp_server.server = obj  # type: ignore[attr-defined]

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
    neo4j_uri: str = typer.Option("bolt://localhost:7687", help="Neo4j Bolt URI"),
    neo4j_user: str = typer.Option("neo4j", help="Neo4j username"),
    neo4j_password: str = typer.Option("testpassword", help="Neo4j password"),
    pathlight_url: str = typer.Option("http://localhost:4100", help="Pathlight collector URL"),
) -> None:
    """Check connectivity to external services."""
    import asyncio

    async def _check() -> None:
        ok = True

        # Neo4j
        try:
            gs = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
            await gs.connect()
            assert gs._driver is not None
            await gs._driver.execute_query("RETURN 1 AS n")
            await gs.close()
            typer.echo(f"Neo4j:     OK ({neo4j_uri})")
        except Exception as exc:
            typer.echo(f"Neo4j:     FAIL ({exc})")
            ok = False

        # Pathlight
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{pathlight_url}/health")
                if resp.status_code == 200:
                    typer.echo(f"Pathlight: OK ({pathlight_url})")
                else:
                    typer.echo(f"Pathlight: FAIL (HTTP {resp.status_code})")
                    ok = False
        except Exception as exc:
            typer.echo(f"Pathlight: FAIL ({exc})")
            ok = False

        raise typer.Exit(0 if ok else 1)

    asyncio.run(_check())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
