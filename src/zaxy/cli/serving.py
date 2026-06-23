"""Split from cli.py (mechanical decomposition)."""


from __future__ import annotations

import json
import os
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from zaxy.cli import runtime as _runtime

if TYPE_CHECKING:
    from zaxy.config import Settings
from zaxy.cli.runtime import (
    _format_causal_results_text,
    _format_reasoning_result_text,
    _parse_source_event,
    _project_packet_result_to_graph,
    _trace_payload_jsonl,
    _validate_causal_relation_type_option,
    _validate_reasoning_phase_option,
    app,
    memory_app,
    memory_causal_app,
    memory_consolidation_app,
    memory_reasoning_app,
    trace_app,
)
from zaxy.cli.workspace import (
    _checkout_activity_metadata,
    _is_embedded_projection_lock_error,
    _resolve_cli_projection_backend,
)


@trace_app.command("export")
def trace_export(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session: list[str] | None = typer.Option(None, "--session", help="Session ID to include"),  # noqa: B008
    output_format: str = typer.Option("json", "--format", help="Output format: json or jsonl"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write trace export to this file"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Export provider-neutral trace correlation from replayed Eventloom events."""
    from zaxy.event import EventLog
    from zaxy.trace import build_trace_correlation

    session_filter = set(session or [])
    events = []
    sessions = []
    for path in sorted(eventloom_path.glob("*.jsonl")):
        if session_filter and path.stem not in session_filter:
            continue
        replay = EventLog(path).replay()
        replay_integrity = replay.integrity
        if replay_integrity is None:
            raise RuntimeError("trace export replay requires integrity verification")
        sessions.append(
            {
                "session_id": path.stem,
                "event_count": len(replay.events),
                "integrity_ok": replay_integrity.ok,
                "latest_seq": replay.events[-1].seq if replay.events else None,
                "latest_hash": replay.events[-1].hash if replay.events else None,
            }
        )
        if not replay_integrity.ok:
            raise typer.BadParameter(
                f"Eventloom integrity failed for {path.name}: {replay_integrity.broken_reason}",
                param_hint="--eventloom-path",
            )
        events.extend(replay.events)
    payload = build_trace_correlation(events).to_dict()
    payload["sessions"] = sessions
    normalized_format = "json" if json_output else output_format.strip().lower()
    if normalized_format not in {"json", "jsonl"}:
        raise typer.BadParameter("--format must be json or jsonl", param_hint="--format")
    rendered = (
        json.dumps(payload, indent=2, sort_keys=True)
        if normalized_format == "json"
        else _trace_payload_jsonl(payload)
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"Wrote trace export: {output}")
        return
    if json_output or normalized_format == "jsonl":
        typer.echo(rendered)
        return
    summary = payload["summary"]
    typer.echo(
        "Trace correlation: "
        f"spans={summary['span_count']} edges={summary['edge_count']} "
        f"missions={summary['mission_count']} model_calls={summary['model_call_count']} "
        f"tool_calls={summary['tool_call_count']}"
    )


@app.command()
def compact(
    log_path: Path = typer.Argument(..., help="Path to Eventloom JSONL file"),  # noqa: B008
    snapshot_every: int = typer.Option(10000, help="Create snapshot every N events"),
    output: Path = typer.Option(None, help="Output path (default: in-place)"),  # noqa: B008
    audit: bool = typer.Option(False, "--audit", help="Run compaction safety audit only"),
    json_output: bool = typer.Option(False, "--json", help="Output audit report as JSON"),
    projection_output: Path | None = typer.Option(  # noqa: B008
        None,
        "--projection-output",
        help="Write source-backed compaction projection JSON without rewriting the log",
    ),
    strategy: str = typer.Option("medoid", help="Projection strategy: medoid or exemplar"),
    max_records: int = typer.Option(5, min=1, help="Maximum exemplar records to store"),
    purpose: str | None = typer.Option(
        None,
        "--purpose",
        help="Optional purpose profile for projection policy, e.g. coordinate",
    ),
) -> None:
    """Compact an Eventloom log and optionally create snapshots."""
    from zaxy.compaction import (
        audit_event_log,
        build_compaction_projection,
        write_compaction_projection,
    )
    from zaxy.event import EventLog

    log = EventLog(str(log_path))
    if audit:
        report = audit_event_log(log)
        if json_output:
            typer.echo(json.dumps(asdict(report), indent=2, sort_keys=True))
        else:
            typer.echo(f"Compaction audit: {'SAFE' if report.safe else 'UNSAFE'}")
            typer.echo(f"Events: {report.event_count}")
            typer.echo(f"Integrity: {'OK' if report.integrity_ok else 'FAILED'}")
            if report.integrity_reason:
                typer.echo(f"Integrity reason: {report.integrity_reason}")
            typer.echo(f"Identities: {report.identity_count}")
            typer.echo(f"Identity recall: {report.identity_recall:.3f}")
            typer.echo(f"Citation coverage: {report.citation_coverage:.3f}")
            typer.echo(
                "Mean within-cluster distance: "
                f"{report.mean_within_cluster_distance:.3f}"
            )
            if report.unsafe_reasons:
                typer.echo("Unsafe reasons:")
                for reason in report.unsafe_reasons:
                    typer.echo(f"  - {reason}")
            if report.missing_identities:
                typer.echo("Missing identities:")
                for identity in report.missing_identities[:10]:
                    typer.echo(f"  - {identity}")
                if len(report.missing_identities) > 10:
                    typer.echo(f"  - ... {len(report.missing_identities) - 10} more")
        raise typer.Exit(0 if report.safe else 1)

    if projection_output is not None:
        try:
            projection = build_compaction_projection(
                log,
                strategy=strategy,
                max_records=max_records,
                purpose=purpose,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        written = write_compaction_projection(projection, projection_output)
        typer.echo(f"Wrote compaction projection: {written}")
        typer.echo(f"Strategy: {projection.strategy}")
        typer.echo(f"Records: {len(projection.records)}")
        typer.echo(f"Source identities: {len(projection.source_identities)}")
        raise typer.Exit(0)

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

    snapshot_path = None
    if total >= snapshot_every:
        snapshot_path = log_path.with_suffix(f".snapshot-{total}.json")
        with open(snapshot_path, "w", encoding="utf-8") as fh:
            for ev in events[-snapshot_every:]:
                fh.write(ev.model_dump_json() + "\n")
        typer.echo(f"Created snapshot: {snapshot_path}")
    from zaxy.lifecycle import build_compaction_completed_event

    lifecycle = build_compaction_completed_event(
        session_id=log_path.stem,
        mode="rewrite",
        status="succeeded",
        log_path=str(log_path),
        event_count=total,
        output_path=str(out_path),
        snapshot_path=str(snapshot_path) if snapshot_path is not None else None,
    )
    EventLog(out_path).append(
        lifecycle["event_type"],
        actor=lifecycle["actor"],
        payload=lifecycle["payload"],
        thread=log_path.stem,
    )


def _format_status_memory_activation(activation: dict[str, Any]) -> str:
    """Format memory activation posture for top-level status output."""
    lines = [
        f"Memory activation: {str(activation['status']).upper()} ({activation['message']})",
        f"  stale after: {activation['stale_after_minutes']} minutes",
    ]
    latest_checkout = activation.get("latest_checkout")
    if isinstance(latest_checkout, dict):
        lines.append(
            f"  latest checkout: seq={latest_checkout['seq']} "
            f"session={latest_checkout['thread']} at {latest_checkout['timestamp']}"
        )
        token_efficiency = latest_checkout.get("token_efficiency")
        if isinstance(token_efficiency, dict):
            prompt_tokens = token_efficiency.get("prompt_tokens")
            facts_per_1k = token_efficiency.get("facts_per_1k_prompt_tokens")
            if isinstance(prompt_tokens, int | float) and isinstance(facts_per_1k, int | float):
                lines.append(
                    f"  checkout tokens: {int(prompt_tokens)} prompt, "
                    f"{float(facts_per_1k):.1f} facts/1k prompt tokens"
                )
    latest_capture = activation.get("latest_capture")
    if isinstance(latest_capture, dict):
        lines.append(
            f"  latest capture: {latest_capture['type']} seq={latest_capture['seq']} "
            f"session={latest_capture['thread']} source={latest_capture['source']}"
        )
    latest_reminder = activation.get("latest_reminder")
    if isinstance(latest_reminder, dict):
        lines.append(
            f"  latest reminder: seq={latest_reminder['seq']} "
            f"session={latest_reminder['thread']} at {latest_reminder['timestamp']}"
        )
    remediations = activation.get("remediations", [])
    if remediations:
        lines.append("Memory next steps:")
        for index, remediation in enumerate(remediations, start=1):
            if not isinstance(remediation, dict):
                continue
            message = remediation.get("message")
            command = remediation.get("command")
            if message:
                lines.append(f"  {index}. {message}")
            if command:
                lines.append(f"     {command}")
    return "\n".join(lines)


def _profile_root_for_eventloom_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name == ".eventloom":
        return candidate.parent
    if candidate.suffix == ".jsonl" and candidate.parent.name == ".eventloom":
        return candidate.parent.parent
    return Path(".")


def _read_env_profile(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _status_settings(root: Path = Path(".")) -> Settings:
    """Load status settings, honoring repo-local .env.local written by zaxy init."""
    from zaxy.config import Settings

    profile = root / ".env.local"
    if not profile.is_file():
        return Settings()
    values = _read_env_profile(profile)
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        if key in os.environ:
            continue
        field_name = key.casefold().lower()
        if field_name in Settings.model_fields:
            kwargs[field_name] = value
    return Settings(**kwargs)


@memory_app.command("checkout")
def memory_checkout(
    query: str = typer.Argument(..., help="Question or task to checkout memory for"),  # noqa: B008
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to checkout"),
    ref: str | None = typer.Option(None, help="Memory ref to checkout, e.g. HEAD or refs/heads/main"),
    replay_from_seq: int = typer.Option(1, min=1, help="Replay start sequence"),
    limit: int = typer.Option(10, min=1, help="Maximum retrieved context items"),
    max_recent_events: int = typer.Option(20, min=1, help="Maximum recent replay events"),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        min=0,
        help="Optional prompt token budget; elided sections are reported in diagnostics",
    ),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    neo4j_ca_cert: str | None = typer.Option(None, help="Neo4j CA certificate path; pass an empty value to disable TLS CA override"),
    neo4j_trust_all: bool | None = typer.Option(None, help="Trust all Neo4j TLS certificates"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Checkout current, cited memory state for an agent turn."""
    import asyncio

    async def _checkout_with_path(
        embedded_graph_path: Path, *, projection_backend_override: str | None = None
    ) -> dict[str, object]:
        settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
        projection_backend = projection_backend_override or _resolve_cli_projection_backend(
            None,
            settings,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
        fabric = _runtime.MemoryFabric(
            eventloom_path=str(eventloom_path),
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_ca_cert=neo4j_ca_cert,
            neo4j_trust_all=neo4j_trust_all,
            projection_backend=projection_backend,
            pggraph_dsn=settings.pggraph_dsn,
            embedded_graph_path=embedded_graph_path,
            latticedb_path=Path(settings.latticedb_path),
        )
        try:
            await fabric.connect()
            checkout = await fabric.checkout_memory(
                query,
                session_id=session_id,
                ref=ref,
                replay_from_seq=replay_from_seq,
                limit=limit,
                max_recent_events=max_recent_events,
            )
            return cast(dict[str, object], checkout.to_dict())
        finally:
            with suppress(Exception):
                await fabric.close()

    async def _checkout() -> dict[str, object]:
        settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
        embedded_graph_path = Path(settings.embedded_graph_path)
        try:
            return await _checkout_with_path(embedded_graph_path)
        except RuntimeError as exc:
            if not _is_embedded_projection_lock_error(exc):
                raise
            # The embedded projection is held by another process (typically the
            # long-lived MCP server's exclusive lock). Rather than stand up a
            # throwaway empty projection — which pays full schema/index setup
            # (~10s) yet returns no graph results anyway — run the checkout
            # graph-degraded on the verbatim + verified-replay lanes. Same
            # retrieval outcome a locked-out checkout produced before, far faster.
            payload = await _checkout_with_path(
                embedded_graph_path, projection_backend_override="null"
            )
            diagnostics = payload.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
                payload["diagnostics"] = diagnostics
            diagnostics["projection_fallback"] = {
                "status": "graph_degraded",
                "reason": "embedded_projection_locked",
                "original_path": str(embedded_graph_path),
                "detail": "graph lane disabled; verbatim + verified replay only",
            }
            return payload

    payload = asyncio.run(_checkout())
    from zaxy.checkout import apply_checkout_budget
    from zaxy.memory_persistence import record_memory_activity

    payload = apply_checkout_budget(payload, max_tokens=max_tokens)
    record_memory_activity(
        eventloom_path,
        session_id=session_id,
        activity="checkout",
        source="cli",
        query=query,
        metadata=_checkout_activity_metadata(payload),
    )
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("projection_fallback"), dict):
            fallback = diagnostics["projection_fallback"]
            typer.echo(
                "Zaxy checkout warning: embedded projection was locked; "
                f"used isolated projection {fallback.get('fallback_path')}",
                err=True,
            )
        typer.echo(payload["prompt"])


def _resolve_append_payload(
    payload_json: str | None, payload_file: Path | None
) -> dict[str, Any]:
    """Resolve the append payload from --payload-json, --payload-file, or stdin.

    Exactly one source is honored. The parsed value MUST be a JSON object so it
    matches the MCP ``memory_append`` contract; non-objects are rejected.
    """
    if payload_json is not None and payload_file is not None:
        raise ValueError("provide only one of --payload-json or --payload-file")
    if payload_json is not None:
        source, origin = payload_json, "--payload-json"
    elif payload_file is not None:
        try:
            source = payload_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"--payload-file could not be read: {exc}") from exc
        origin = "--payload-file"
    else:
        source = sys.stdin.read()
        origin = "stdin"
    if not source.strip():
        raise ValueError(
            "payload is required via --payload-json, --payload-file, or stdin"
        )
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{origin} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{origin} must be a JSON object")
    return parsed


@memory_app.command("append")
def memory_append(
    event_type: str = typer.Argument(..., help="Dotted lowercase event type, e.g. decision.made"),  # noqa: B008
    actor: str = typer.Option(..., "--actor", help="Actor id that produced the event, e.g. ainix-agent"),
    payload_json: str | None = typer.Option(None, "--payload-json", help="Event payload as a JSON object"),  # noqa: B008
    payload_file: Path | None = typer.Option(None, "--payload-file", help="Read the JSON payload object from this file"),  # noqa: B008
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to append into"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Append one cited event through the shared MemoryFabric pipeline (CLI twin of MCP memory_append)."""
    import asyncio

    from zaxy.security import validate_event_text, validate_payload, validate_session_id

    try:
        safe_event_type = validate_event_text(event_type, "event_type")
        safe_actor = validate_event_text(actor, "actor")
        safe_session_id = validate_session_id(session_id)
        safe_payload = validate_payload(_resolve_append_payload(payload_json, payload_file))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def _append_with_path(
        embedded_graph_path: Path, *, projection_backend_override: str | None = None
    ) -> Any:
        settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
        projection_backend = projection_backend_override or _resolve_cli_projection_backend(
            None, settings
        )
        fabric = _runtime._memory_fabric(
            eventloom_path=str(eventloom_path),
            projection_backend=projection_backend,
            pggraph_dsn=settings.pggraph_dsn,
            embedded_graph_path=embedded_graph_path,
            latticedb_path=Path(settings.latticedb_path),
        )
        try:
            await fabric.connect()
            return await fabric.append(
                safe_event_type, safe_actor, safe_payload, session_id=safe_session_id
            )
        finally:
            with suppress(Exception):
                await fabric.close()

    async def _append() -> Any:
        settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
        embedded_graph_path = Path(settings.embedded_graph_path)
        try:
            return await _append_with_path(embedded_graph_path)
        except RuntimeError as exc:
            if not _is_embedded_projection_lock_error(exc):
                raise
            # A server (typically the long-lived MCP daemon) holds the embedded
            # projection's single-owner lock. The append must stay durable, so
            # write the event with the graph lane degraded instead of hard-failing
            # on projection contention — same degraded behavior `memory checkout`
            # uses when locked out.
            return await _append_with_path(
                embedded_graph_path, projection_backend_override="null"
            )

    try:
        event = asyncio.run(_append())
    except (RuntimeError, OSError) as exc:
        # Lock/fsync errors, unreadable dir, etc. Exit non-zero with a message on
        # stderr and no partial JSON on stdout so a trusted shim can degrade.
        typer.echo(f"zaxy memory append failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    citation = f"eventloom://{event.thread}/events/{event.seq}#{event.hash[:12]}"
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "seq": event.seq,
                    "hash": event.hash,
                    "event_id": event.id,
                    "session_id": event.thread,
                    "event_type": event.type,
                    "citation": citation,
                },
                sort_keys=True,
            )
        )
    else:
        typer.echo(f"Recorded {event.type} seq={event.seq}")


def _resolve_ingest_items(items_file: Path | None) -> list[dict[str, Any]]:
    """Resolve a batch of ingest items from a JSONL --file or stdin.

    Each non-empty line MUST parse to a JSON object (one event per line), so the
    CLI matches the MCP ``memory_ingest`` per-event contract.
    """
    if items_file is not None:
        try:
            source = items_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"--file could not be read: {exc}") from exc
        origin = "--file"
    else:
        source = sys.stdin.read()
        origin = "stdin"
    items: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{origin} line {line_number} is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{origin} line {line_number} must be a JSON object")
        items.append(parsed)
    if not items:
        raise ValueError(
            "no ingest items provided via --file or stdin (one JSON object per line)"
        )
    return items


@memory_app.command("ingest")
def memory_ingest(
    items_file: Path | None = typer.Option(None, "--file", help="JSONL file of events (one JSON object per line); reads stdin if omitted"),  # noqa: B008
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to ingest into"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Batch-ingest external-producer events through the shared MemoryFabric pipeline (CLI twin of MCP memory_ingest)."""
    import asyncio

    from zaxy.security import validate_event_text, validate_payload, validate_session_id

    try:
        safe_session_id = validate_session_id(session_id)
        items = _resolve_ingest_items(items_file)
        for index, item in enumerate(items):
            validate_event_text(item.get("event_type"), "event_type")
            validate_event_text(item.get("actor"), "actor")
            validate_payload(item.get("payload") or {})
            for field in ("producer_ref", "parent_event_id", "id"):
                value = item.get(field)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"item {index} {field} must be a string")
            caused_by = item.get("caused_by")
            if caused_by is not None and (
                not isinstance(caused_by, list) or not all(isinstance(c, str) for c in caused_by)
            ):
                raise ValueError(f"item {index} caused_by must be a list of strings")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    async def _ingest_with_path(
        embedded_graph_path: Path, *, projection_backend_override: str | None = None
    ) -> Any:
        settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
        projection_backend = projection_backend_override or _resolve_cli_projection_backend(
            None, settings
        )
        fabric = _runtime._memory_fabric(
            eventloom_path=str(eventloom_path),
            projection_backend=projection_backend,
            pggraph_dsn=settings.pggraph_dsn,
            embedded_graph_path=embedded_graph_path,
            latticedb_path=Path(settings.latticedb_path),
        )
        try:
            await fabric.connect()
            return await fabric.append_batch(items, session_id=safe_session_id)
        finally:
            with suppress(Exception):
                await fabric.close()

    async def _ingest() -> Any:
        settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
        embedded_graph_path = Path(settings.embedded_graph_path)
        try:
            return await _ingest_with_path(embedded_graph_path)
        except RuntimeError as exc:
            if not _is_embedded_projection_lock_error(exc):
                raise
            # A server holds the embedded projection's single-owner lock; keep the
            # ingest durable by writing with the graph lane degraded (null backend),
            # mirroring `memory append`.
            return await _ingest_with_path(
                embedded_graph_path, projection_backend_override="null"
            )

    try:
        events = asyncio.run(_ingest())
    except (RuntimeError, OSError) as exc:
        typer.echo(f"zaxy memory ingest failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    imported = len(events)
    deduped = len(items) - imported
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "imported": imported,
                    "deduped": deduped,
                    "session_id": safe_session_id,
                    "events": [
                        {"seq": event.seq, "hash": event.hash, "event_id": event.id}
                        for event in events
                    ],
                },
                sort_keys=True,
            )
        )
    else:
        seq_range = f"{events[0].seq}..{events[-1].seq}" if events else "-"
        typer.echo(f"imported={imported} deduped={deduped} seq={seq_range}")


async def _run_reasoning_primitive(
    *,
    method_name: str,
    eventloom_path: Path,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
    fabric = _runtime._memory_fabric(
        eventloom_path=str(eventloom_path),
        projection_backend=_resolve_cli_projection_backend(None, settings),
        pggraph_dsn=settings.pggraph_dsn,
        embedded_graph_path=Path(settings.embedded_graph_path),
        latticedb_path=Path(settings.latticedb_path),
    )
    try:
        await fabric.connect()
        result = await getattr(fabric, method_name)(*args, **kwargs)
        return cast(dict[str, Any], result)
    finally:
        with suppress(Exception):
            await fabric.close()


async def _query_causal_memory(
    *,
    direction: str,
    entity_name: str,
    relation_type: str | None,
    session_id: str,
    depth: int,
    eventloom_path: Path,
) -> list[dict[str, object]]:
    settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
    fabric = _runtime._memory_fabric(
        eventloom_path=str(eventloom_path),
        projection_backend=_resolve_cli_projection_backend(None, settings),
        pggraph_dsn=settings.pggraph_dsn,
        embedded_graph_path=Path(settings.embedded_graph_path),
        latticedb_path=Path(settings.latticedb_path),
    )
    try:
        await fabric.connect()
        if direction == "successors":
            results = await fabric.query_causal_successors(
                entity_name,
                relation_type=relation_type,
                depth=depth,
                session_id=session_id,
            )
        else:
            results = await fabric.query_causal_predecessors(
                entity_name,
                relation_type=relation_type,
                depth=depth,
                session_id=session_id,
            )
        return [cast(dict[str, object], result.to_dict()) for result in results]
    finally:
        with suppress(Exception):
            await fabric.close()


@memory_causal_app.command("successors")
def memory_causal_successors(
    entity_name: str = typer.Argument(..., help="Entity name to inspect"),  # noqa: B008
    entity_type: str | None = typer.Option(None, help="Entity type label for output context"),
    relation_type: str | None = typer.Option(
        None,
        callback=_validate_causal_relation_type_option,
        help="Causal relation type to filter",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    depth: int = typer.Option(2, min=1, help="Traversal depth"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List directed causal effects of an entity."""
    import asyncio

    results = asyncio.run(
        _query_causal_memory(
            direction="successors",
            entity_name=entity_name,
            relation_type=relation_type,
            session_id=session_id,
            depth=depth,
            eventloom_path=eventloom_path,
        )
    )
    payload = {
        "direction": "successors",
        "entity": {"name": entity_name, "entity_type": entity_type},
        "results": results,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(_format_causal_results_text(direction="successors", entity_name=entity_name, results=results))


@memory_causal_app.command("predecessors")
def memory_causal_predecessors(
    entity_name: str = typer.Argument(..., help="Entity name to inspect"),  # noqa: B008
    entity_type: str | None = typer.Option(None, help="Entity type label for output context"),
    relation_type: str | None = typer.Option(
        None,
        callback=_validate_causal_relation_type_option,
        help="Causal relation type to filter",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    depth: int = typer.Option(2, min=1, help="Traversal depth"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List directed causal causes of an entity."""
    import asyncio

    results = asyncio.run(
        _query_causal_memory(
            direction="predecessors",
            entity_name=entity_name,
            relation_type=relation_type,
            session_id=session_id,
            depth=depth,
            eventloom_path=eventloom_path,
        )
    )
    payload = {
        "direction": "predecessors",
        "entity": {"name": entity_name, "entity_type": entity_type},
        "results": results,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(_format_causal_results_text(direction="predecessors", entity_name=entity_name, results=results))


@memory_reasoning_app.command("explain-outcome")
def memory_reasoning_explain_outcome(
    outcome: str = typer.Argument(..., help="Outcome to explain from memory context"),  # noqa: B008
    phase: str = typer.Option(  # noqa: B008
        "planning",
        callback=_validate_reasoning_phase_option,
        help="Reasoning phase: planning, execution, review, or reflection",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    depth: int = typer.Option(2, min=1, help="Causal traversal depth"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Explain an outcome through cited MemoryFabric reasoning context."""
    import asyncio

    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="explain_outcome",
            eventloom_path=eventloom_path,
            args=(outcome,),
            kwargs={"phase": phase, "session_id": session_id, "depth": depth},
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


@memory_reasoning_app.command("propose-belief-update")
def memory_reasoning_propose_belief_update(
    claim: str = typer.Argument(..., help="Claim to propose for review"),  # noqa: B008
    rationale: str = typer.Option(..., help="Cited rationale for the proposal"),
    confidence: float = typer.Option(..., min=0.0, max=1.0, help="Proposal confidence from 0.0 to 1.0"),
    source_event: list[str] = typer.Option(..., "--source-event", help="Cited source event as SEQ:HASH"),  # noqa: B008
    phase: str = typer.Option(  # noqa: B008
        "reflection",
        callback=_validate_reasoning_phase_option,
        help="Reasoning phase: planning, execution, review, or reflection",
    ),
    actor: str = typer.Option("zaxy-reasoning", help="Actor recording the proposal"),
    session_id: str = typer.Option("default", help="Session ID to append to"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Propose a non-authoritative belief update through MemoryFabric."""
    import asyncio

    source_events = [_parse_source_event(value) for value in source_event]
    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="propose_belief_update",
            eventloom_path=eventloom_path,
            args=(claim,),
            kwargs={
                "rationale": rationale,
                "confidence": confidence,
                "source_events": source_events,
                "phase": phase,
                "session_id": session_id,
                "actor": actor,
            },
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


@memory_reasoning_app.command("claim-confidence")
def memory_reasoning_claim_confidence(
    claim: str = typer.Argument(..., help="Claim to score against cited memory evidence"),  # noqa: B008
    phase: str = typer.Option(  # noqa: B008
        "review",
        callback=_validate_reasoning_phase_option,
        help="Reasoning phase: planning, execution, review, or reflection",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    limit: int = typer.Option(5, min=1, help="Maximum evidence items"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Score claim confidence through MemoryFabric evidence retrieval."""
    import asyncio

    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="get_claim_confidence",
            eventloom_path=eventloom_path,
            args=(claim,),
            kwargs={"phase": phase, "session_id": session_id, "limit": limit},
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


@memory_reasoning_app.command("record-unknown")
def memory_reasoning_record_unknown(
    question: str = typer.Argument(..., help="Known-unknown question to track"),  # noqa: B008
    reason: str = typer.Option(..., help="Reason this uncertainty was recorded"),
    source_event: list[str] = typer.Option(..., "--source-event", help="Cited source event as SEQ:HASH"),  # noqa: B008
    claim_key: str = typer.Option(..., help="Stable claim or uncertainty key"),
    gap_type: str = typer.Option("missing_evidence", help="Uncertainty gap type"),
    reverify_query: str | None = typer.Option(None, help="Suggested query for re-verification"),
    phase: str = typer.Option(  # noqa: B008
        "review",
        callback=_validate_reasoning_phase_option,
        help="Reasoning phase: planning, execution, review, or reflection",
    ),
    actor: str = typer.Option("zaxy-reasoning", help="Actor recording the known unknown"),
    session_id: str = typer.Option("default", help="Session ID to append to"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Record a cited, open, non-authoritative known unknown."""
    import asyncio

    source_events = [_parse_source_event(value) for value in source_event]
    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="record_known_unknown",
            eventloom_path=eventloom_path,
            args=(question,),
            kwargs={
                "reason": reason,
                "source_events": source_events,
                "claim_key": claim_key,
                "gap_type": gap_type,
                "reverify_query": reverify_query,
                "phase": phase,
                "session_id": session_id,
                "actor": actor,
            },
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


@memory_reasoning_app.command("confidence-trajectory")
def memory_reasoning_confidence_trajectory(
    claim: str = typer.Argument(..., help="Claim or claim key to inspect"),  # noqa: B008
    phase: str = typer.Option(  # noqa: B008
        "review",
        callback=_validate_reasoning_phase_option,
        help="Accepted for reasoning command consistency; trajectory queries are replay-derived",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    limit: int = typer.Option(10, min=1, help="Maximum trajectory points"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List append-only confidence assessments for a claim."""
    import asyncio

    _ = phase
    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="list_confidence_trajectory",
            eventloom_path=eventloom_path,
            args=(claim,),
            kwargs={"session_id": session_id, "limit": limit},
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


@memory_reasoning_app.command("reverify-needed")
def memory_reasoning_reverify_needed(
    query: str | None = typer.Option(None, help="Optional query filter"),
    phase: str = typer.Option(  # noqa: B008
        "review",
        callback=_validate_reasoning_phase_option,
        help="Accepted for reasoning command consistency; reverify queries are replay-derived",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    limit: int = typer.Option(10, min=1, help="Maximum re-verification needs"),
    min_confidence: float = typer.Option(0.7, min=0.0, max=1.0, help="Confidence threshold"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List open unknowns, unresolved conflicts, and low-confidence claims."""
    import asyncio

    _ = phase
    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="list_reverification_needs",
            eventloom_path=eventloom_path,
            args=(),
            kwargs={
                "query": query,
                "session_id": session_id,
                "limit": limit,
                "min_confidence": min_confidence,
            },
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


@memory_reasoning_app.command("plan-from-procedures")
def memory_reasoning_plan_from_procedures(
    goal: str = typer.Argument(..., help="Planning goal"),  # noqa: B008
    phase: str = typer.Option(  # noqa: B008
        "planning",
        callback=_validate_reasoning_phase_option,
        help="Reasoning phase: planning, execution, review, or reflection",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    limit: int = typer.Option(5, min=1, help="Maximum plan steps/source procedures"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Build a non-authoritative planning packet from applicable procedures."""
    import asyncio

    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="plan_from_procedures",
            eventloom_path=eventloom_path,
            args=(goal,),
            kwargs={"phase": phase, "session_id": session_id, "limit": limit},
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


@memory_reasoning_app.command("similar-procedures")
def memory_reasoning_similar_procedures(
    query: str = typer.Argument(..., help="Procedure retrieval query"),  # noqa: B008
    phase: str = typer.Option(  # noqa: B008
        "planning",
        callback=_validate_reasoning_phase_option,
        help="Reasoning phase: planning, execution, review, or reflection",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    limit: int = typer.Option(5, min=1, help="Maximum procedure candidates"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Retrieve similar procedures through MemoryFabric."""
    import asyncio

    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="retrieve_similar_procedures",
            eventloom_path=eventloom_path,
            args=(query,),
            kwargs={"phase": phase, "session_id": session_id, "limit": limit},
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


async def _append_consolidation_event(event: dict[str, Any], *, eventloom_path: Path) -> None:
    settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
    fabric = _runtime._memory_fabric(
        eventloom_path=str(eventloom_path),
        projection_backend=_resolve_cli_projection_backend(None, settings),
        pggraph_dsn=settings.pggraph_dsn,
        embedded_graph_path=Path(settings.embedded_graph_path),
        latticedb_path=Path(settings.latticedb_path),
    )
    try:
        await fabric.connect()
        await fabric.append(**event)
    finally:
        with suppress(Exception):
            await fabric.close()


async def _propose_consolidation_from_log(
    *,
    session_id: str,
    actor: str,
    purpose: str | None,
    window_size: int,
    eventloom_path: Path,
) -> dict[str, Any]:
    settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
    fabric = _runtime._memory_fabric(
        eventloom_path=str(eventloom_path),
        projection_backend=_resolve_cli_projection_backend(None, settings),
        pggraph_dsn=settings.pggraph_dsn,
        embedded_graph_path=Path(settings.embedded_graph_path),
        latticedb_path=Path(settings.latticedb_path),
    )
    try:
        await fabric.connect()
        result = await fabric.propose_consolidation_candidates(
            session_id=session_id,
            actor=actor,
            purpose=purpose,
            window_size=window_size,
        )
        return cast(dict[str, Any], result)
    finally:
        with suppress(Exception):
            await fabric.close()


async def _read_consolidation_status(*, session_id: str, eventloom_path: Path) -> dict[str, Any]:
    settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
    fabric = _runtime._memory_fabric(
        eventloom_path=str(eventloom_path),
        projection_backend=_resolve_cli_projection_backend(None, settings),
        pggraph_dsn=settings.pggraph_dsn,
        embedded_graph_path=Path(settings.embedded_graph_path),
        latticedb_path=Path(settings.latticedb_path),
    )
    try:
        await fabric.connect()
        result = await fabric.consolidation_status(session_id=session_id)
        return cast(dict[str, Any], result)
    finally:
        with suppress(Exception):
            await fabric.close()


@memory_consolidation_app.command("propose")
def memory_consolidation_propose(
    candidate_type: str = typer.Option(..., help="Candidate type: episode, claim, or procedure"),
    title: str = typer.Option(..., help="Candidate title"),
    summary: str = typer.Option(..., help="Candidate summary"),
    source_event: list[str] = typer.Option(..., "--source-event", help="Cited source event as SEQ:HASH"),  # noqa: B008
    confidence: float = typer.Option(..., min=0.0, max=1.0, help="Candidate confidence"),
    method: str = typer.Option(..., help="Consolidation method"),
    purpose: str | None = typer.Option(None, help="Optional candidate purpose"),
    actor: str = typer.Option("zaxy", help="Actor writing the event"),
    session_id: str = typer.Option("default", help="Session ID to append to"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Append a review-pending consolidation candidate event."""
    import asyncio

    from zaxy.consolidation import build_consolidation_candidate_event

    try:
        source_events = [_parse_source_event(value) for value in source_event]
        event = build_consolidation_candidate_event(
            actor=actor,
            session_id=session_id,
            candidate_type=candidate_type,
            title=title,
            summary=summary,
            source_events=source_events,
            confidence=confidence,
            method=method,
            purpose=purpose,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    asyncio.run(_append_consolidation_event(event, eventloom_path=eventloom_path))
    if json_output:
        typer.echo(json.dumps(event, indent=2, sort_keys=True))
    else:
        typer.echo(f"Created {event['payload']['candidate_id']} ({event['payload']['review_status']})")


@memory_consolidation_app.command("propose-from-log")
def memory_consolidation_propose_from_log(
    session_id: str = typer.Option("default", help="Session ID to replay for proposal windows"),
    actor: str = typer.Option("zaxy-consolidation", help="Actor writing candidate events"),
    purpose: str | None = typer.Option(None, help="Optional consolidation purpose"),
    window_size: int = typer.Option(5, min=1, max=200, help="Number of source events per proposal window"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Create review-pending consolidation candidates from Eventloom log segments."""
    import asyncio

    result = asyncio.run(
        _propose_consolidation_from_log(
            session_id=session_id,
            actor=actor,
            purpose=purpose,
            window_size=window_size,
            eventloom_path=eventloom_path,
        )
    )
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        candidates_created = int(result.get("candidate_count", 0))
        segments_considered = int(result.get("segment_count", 0))
        typer.echo(
            "Created "
            f"{candidates_created} non-authoritative consolidation candidates from "
            f"{segments_considered} log segments for {session_id}."
        )


@memory_consolidation_app.command("status")
def memory_consolidation_status(
    session_id: str = typer.Option("default", help="Session ID to inspect"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show review-gated consolidation candidate status."""
    import asyncio

    result = asyncio.run(_read_consolidation_status(session_id=session_id, eventloom_path=eventloom_path))
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        pending = int(result.get("pending_count", 0))
        accepted = int(result.get("accepted_count", 0))
        rejected = int(result.get("rejected_count", 0))
        deferred = int(result.get("deferred_count", 0))
        conflicted = int(result.get("conflicted_count", 0))
        typer.echo(
            f"Consolidation status for {session_id}: "
            f"pending={pending}, accepted={accepted}, rejected={rejected}, "
            f"deferred={deferred}, conflicted={conflicted}"
        )


@memory_app.command("mine-procedures")
def memory_mine_procedures(
    session_id: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--session-id",
        help="Session ID (Eventloom thread) to mine; repeatable, default all sessions",
    ),
    min_support: int = typer.Option(
        2,
        "--min-support",
        min=2,
        help="Minimum distinct supporting sessions per mined procedure",
    ),
    max_length: int = typer.Option(
        8,
        "--max-length",
        min=2,
        help="Maximum mined procedure length in steps",
    ),
    actor: str = typer.Option("zaxy-procedure-miner", "--actor", help="Actor writing candidate events"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Mine recurring successful tool sequences into review-pending procedure candidates."""
    from zaxy.event import EventLog
    from zaxy.procedure_mining import build_procedure_proposal, mine_and_propose

    log_paths = sorted(eventloom_path.glob("*.jsonl"))
    logs_payload: list[dict[str, Any]] = []
    candidate_lines: list[str] = []
    total_mined = 0
    total_appended = 0
    total_skipped = 0
    for log_path in log_paths:
        try:
            summary = mine_and_propose(
                EventLog(log_path),
                session_ids=session_id or None,
                min_support=min_support,
                max_length=max_length,
                actor=actor,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        appended_by_id = {proposal.candidate_id: proposal for proposal in summary.appended}
        skipped_ids = set(summary.skipped_candidate_ids)
        candidates: list[dict[str, Any]] = []
        for procedure in summary.mined:
            proposal = build_procedure_proposal(procedure)
            payload = proposal.to_candidate_event(actor=actor)["payload"]
            candidate_id = str(payload["candidate_id"])
            if candidate_id in appended_by_id:
                status = "proposed"
            elif candidate_id in skipped_ids:
                status = "skipped_duplicate"
            else:  # pragma: no cover - mine_and_propose appends or skips every candidate
                status = "unknown"
            candidate: dict[str, Any] = {
                "candidate_id": candidate_id,
                "title": proposal.title,
                "status": status,
                "support": procedure.support,
                "support_sessions": list(procedure.support_sessions),
            }
            appended = appended_by_id.get(candidate_id)
            if appended is not None:
                candidate["seq"] = appended.seq
                candidate["hash"] = appended.hash
                candidate["session_id"] = appended.session_id
            candidates.append(candidate)
            candidate_lines.append(
                f"  - [{status}] {proposal.title} (support={procedure.support})"
            )
        logs_payload.append(
            {
                "log": log_path.name,
                "session_ids": list(summary.session_ids),
                "mined_count": summary.mined_count,
                "appended_count": summary.appended_count,
                "skipped_duplicate_count": summary.skipped_duplicate_count,
                "candidates": candidates,
            }
        )
        total_mined += summary.mined_count
        total_appended += summary.appended_count
        total_skipped += summary.skipped_duplicate_count

    if json_output:
        payload = {
            "eventloom_path": str(eventloom_path),
            "actor": actor,
            "min_support": min_support,
            "max_length": max_length,
            "session_ids": list(session_id) if session_id else None,
            "logs": logs_payload,
            "totals": {
                "mined": total_mined,
                "proposed": total_appended,
                "skipped_duplicates": total_skipped,
            },
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not log_paths:
        typer.echo(f"No Eventloom logs found in {eventloom_path}")
        return
    noun = "log" if len(log_paths) == 1 else "logs"
    typer.echo(
        f"Mined {total_mined} procedure candidates from {len(log_paths)} {noun}: "
        f"proposed={total_appended}, skipped_duplicates={total_skipped}"
    )
    for line in candidate_lines:
        typer.echo(line)


@app.command("refresh-context")
def refresh_context(
    path: Path = typer.Argument(..., help="Document/codebase root to refresh"),  # noqa: B008
    kind: str = typer.Option("documents", "--kind", help="Context kind: documents or codebase"),
    session_id: str = typer.Option("default", help="Session ID to append refresh events into"),  # noqa: B008
    eventloom_path: Path = typer.Option(Path(".eventloom"), help="Eventloom directory for refresh state"),  # noqa: B008
    projection_backend: str | None = typer.Option(None, "--projection-backend", help="Projection backend: embedded, neo4j, pggraph, or latticedb"),  # noqa: B008
    pggraph_dsn: str | None = typer.Option(None, "--pggraph-dsn", help="pgGraph/PostgreSQL DSN"),  # noqa: B008
    max_lines: int = typer.Option(80, "--max-lines", min=1, help="Maximum document lines per chunk"),
    max_bytes: int = typer.Option(512 * 1024, "--max-bytes", min=1, help="Maximum source file size to refresh"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Incrementally refresh document or codebase context from changed sources."""
    import asyncio

    async def _run() -> dict[str, object]:
        settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
        fabric = _runtime.MemoryFabric(
            eventloom_path=str(eventloom_path),
            projection_backend=projection_backend or settings.projection_backend,
            pggraph_dsn=pggraph_dsn or settings.pggraph_dsn,
            embedded_graph_path=Path(settings.embedded_graph_path),
            latticedb_path=Path(settings.latticedb_path),
            tracer_disabled=False,
        )
        try:
            report = await fabric.refresh_context(
                path,
                kind=kind,
                session_id=session_id,
                max_lines=max_lines,
                max_bytes=max_bytes,
            )
            return cast(dict[str, object], report.to_dict())
        finally:
            await fabric.close()

    report = asyncio.run(_run())
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    summary = cast(dict[str, object], report["summary"])
    typer.echo(
        f"Refreshed {report['kind']} context for session {report['session_id']}: "
        f"{summary['indexed']} indexed, {summary['unchanged']} unchanged, "
        f"{summary['deleted']} deleted, {report['event_count']} events"
    )


@app.command("dashboard")
def dashboard(
    workspace: Path | None = typer.Option(None, help="Workspace root to inspect"),  # noqa: B008
    eventloom_path: Path | None = typer.Option(None, help="Eventloom directory to inspect"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Session ID to select by default"),  # noqa: B008
    domain: str | None = typer.Option(None, help="Domain scope to display"),  # noqa: B008
    host: str = typer.Option("127.0.0.1", help="Dashboard bind host"),
    port: int = typer.Option(8765, min=1, max=65535, help="Dashboard bind port"),
    projection_backend: str | None = typer.Option(None, "--projection-backend", help="Projection backend for graph visualization: embedded, neo4j, pggraph, or latticedb"),  # noqa: B008
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI for graph visualization"),  # noqa: B008
    neo4j_user: str | None = typer.Option(None, help="Neo4j username for graph visualization"),  # noqa: B008
    neo4j_password: str | None = typer.Option(None, help="Neo4j password for graph visualization"),  # noqa: B008
    pggraph_dsn: str | None = typer.Option(None, "--pggraph-dsn", help="pgGraph/PostgreSQL DSN for graph visualization"),  # noqa: B008
    embedded_graph_path: Path | None = typer.Option(None, "--embedded-graph-path", help="Embedded graph projection path for graph visualization"),  # noqa: B008
    enable_coordinate_review: bool = typer.Option(False, "--enable-coordinate-review", help="Enable local dashboard review and promotion controls"),
) -> None:
    """Start the local runtime dashboard."""
    from zaxy.dashboard import DashboardConfig, resolve_dashboard_scope, run_dashboard

    settings = _status_settings(workspace or Path("."))
    scope = resolve_dashboard_scope(
        DashboardConfig(
            workspace=workspace,
            eventloom_path=eventloom_path,
            session_id=session_id,
            domain=domain,
            host=host,
            port=port,
            projection_backend=_resolve_cli_projection_backend(
                projection_backend,
                settings,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                pggraph_dsn=pggraph_dsn,
                embedded_graph_path=embedded_graph_path,
            ),
            neo4j_uri=neo4j_uri or settings.neo4j_uri,
            neo4j_user=neo4j_user or settings.neo4j_user,
            neo4j_password=neo4j_password or settings.neo4j_password,
            pggraph_dsn=pggraph_dsn or settings.pggraph_dsn,
            embedded_graph_path=embedded_graph_path or Path(settings.embedded_graph_path),
            coordinate_review_enabled=enable_coordinate_review,
        )
    )
    typer.echo(f"Zaxy dashboard listening on http://{scope.host}:{scope.port}")
    typer.echo(f"Workspace: {scope.workspace}")
    typer.echo(f"Eventloom: {scope.eventloom_path}")
    typer.echo(f"Coordinate review: {'enabled' if scope.coordinate_review_enabled else 'disabled'}")
    run_dashboard(scope)


@app.command()
def serve(
    eventloom_path: str | None = typer.Option(None, help="Directory for event logs"),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    transport: str = typer.Option("stdio", help="Transport: stdio or sse"),
    host: str = typer.Option("127.0.0.1", help="Host for SSE transport"),
    port: int = typer.Option(8080, help="Port for SSE transport"),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="MCP tool listing profile: core or full (default from MCP_TOOL_PROFILE)",
    ),
) -> None:
    """Start the MCP server (stdio or sse)."""
    import asyncio

    from zaxy import mcp_server
    from zaxy.domain import derive_domain, domain_default_session
    from zaxy.mcp_runtime import EmbeddedMcpRuntimeCoordinator
    from zaxy.tool_profiles import resolve_profile

    workspace_root = Path.cwd()
    resolved_eventloom_path = eventloom_path or os.getenv("EVENTLOOM_PATH") or str(workspace_root / ".eventloom")
    resolved_session_id = os.getenv("EVENTLOOM_THREAD") or domain_default_session(derive_domain(workspace_root))
    settings = _status_settings(workspace_root)
    resolved_tool_profile = profile or settings.mcp_tool_profile
    try:
        resolve_profile(resolved_tool_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--profile") from exc
    embedded_graph_path = Path(settings.embedded_graph_path)
    if eventloom_path is not None and embedded_graph_path == Path(".eventloom/projections/embedded.kuzu"):
        embedded_graph_path = Path(resolved_eventloom_path) / "projections" / "embedded.kuzu"

    projection_backend = _resolve_cli_projection_backend(
        None,
        settings,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    owner_claim = None
    embedded_stdio_coordinator = None
    if transport == "stdio" and projection_backend.casefold().strip() == "embedded":
        # Key the owner lock on the STORE path (not the eventloom path) so two
        # processes opening the same embedded store always coordinate, even if
        # they resolved their eventloom path differently.
        embedded_stdio_coordinator = EmbeddedMcpRuntimeCoordinator.from_embedded_graph_path(
            embedded_graph_path
        )
        owner_claim = embedded_stdio_coordinator.try_claim_owner()
        if owner_claim is None:
            # A live-but-broken owner (lock held, socket dead) would wedge every
            # client that tries to proxy to it. Attempt a verified reap-and-reclaim
            # before falling back to proxy; a healthy owner is never reaped.
            repair = embedded_stdio_coordinator.repair_stale_runtime(
                reap=True, expected_graph_path=embedded_graph_path
            )
            if repair.get("repaired"):
                owner_claim = embedded_stdio_coordinator.try_claim_owner()
        if owner_claim is None:
            asyncio.run(mcp_server.proxy_main(embedded_stdio_coordinator))
            return

    # Configure the module-level server instance from CLI overrides
    mcp_server.server = mcp_server.ZaxyMCPServer(
        eventloom_path=resolved_eventloom_path,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        projection_backend=projection_backend,
        pggraph_dsn=settings.pggraph_dsn,
        embedded_graph_path=embedded_graph_path,
        latticedb_path=Path(settings.latticedb_path),
        workspace_root=workspace_root,
        default_session_id=resolved_session_id,
        tool_profile=resolved_tool_profile,
    )

    if transport == "sse":
        asyncio.run(mcp_server.main_sse(port=port, host=host))
    else:
        asyncio.run(mcp_server.main(owner_claim=owner_claim))


@app.command()
def reproject(
    log_path: Path = typer.Argument(..., help="Path to Eventloom JSONL file"),  # noqa: B008
    from_seq: int = typer.Option(1, help="Start sequence number"),
    session_id: str = typer.Option("default", help="Graph session ID to project into"),
    projection_backend: str | None = typer.Option(
        None,
        "--projection-backend",
        help="Projection backend to rebuild: embedded, neo4j, pggraph, or latticedb",
    ),
    pggraph_dsn: str | None = typer.Option(  # noqa: B008
        None,
        "--pggraph-dsn",
        help="Experimental pgGraph/PostgreSQL DSN for --projection-backend pggraph",
    ),
    embedded_graph_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--embedded-graph-path",
        help="Embedded graph projection path for --projection-backend embedded",
    ),
    reset_projection: bool = typer.Option(
        False,
        "--reset-projection",
        help="Clear backend projection tables before replaying the log",
    ),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
) -> None:
    """Replay an Eventloom log and rebuild its graph projection."""
    import asyncio

    profile_root = _profile_root_for_eventloom_path(log_path)
    settings = _status_settings(profile_root)
    backend = _resolve_cli_projection_backend(
        projection_backend,
        settings,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        pggraph_dsn=pggraph_dsn,
        embedded_graph_path=embedded_graph_path,
    )

    async def _run() -> int:
        from zaxy.event import EventLog
        from zaxy.extract import extract
        from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store

        store = build_projection_store(
            ProjectionBackendConfig(
                backend=backend,
                neo4j_uri=neo4j_uri or settings.neo4j_uri,
                neo4j_user=neo4j_user or settings.neo4j_user,
                neo4j_password=neo4j_password or settings.neo4j_password,
                neo4j_ca_cert=settings.neo4j_ca_cert,
                neo4j_trust_all=settings.neo4j_trust_all,
                pggraph_dsn=pggraph_dsn or settings.pggraph_dsn,
                embedded_graph_path=embedded_graph_path or Path(settings.embedded_graph_path),
                latticedb_path=Path(settings.latticedb_path),
                embedding_dimension=settings.embedding_dimension,
            )
        )
        await store.connect()
        try:
            await store.init_schema()
            if reset_projection:
                reset_backend = getattr(store, "reset_benchmark_projection", None)
                if reset_backend is None:
                    raise typer.BadParameter("--reset-projection requires backend reset support")
                await reset_backend()
            replay_result = EventLog(str(log_path)).replay(from_seq=from_seq)
            replay_integrity = replay_result.integrity
            if replay_integrity is None:
                raise RuntimeError("reprojection replay requires integrity verification")
            if not replay_integrity.ok:
                reason = replay_integrity.broken_reason or "unknown integrity failure"
                raise typer.BadParameter(f"Eventloom integrity failed: {reason}")
            begin_bulk = getattr(store, "begin_bulk_projection", None)
            commit_bulk = getattr(store, "commit_bulk_projection", None)
            rollback_bulk = getattr(store, "rollback_bulk_projection", None)
            use_bulk = callable(begin_bulk) and callable(commit_bulk)
            if use_bulk:
                await cast(Callable[[], Awaitable[None]], begin_bulk)()
            try:
                for event in replay_result.events:
                    await store.upsert_extraction(extract(event), session_id=session_id)
            except Exception:
                if use_bulk and callable(rollback_bulk):
                    await cast(Callable[[], Awaitable[None]], rollback_bulk)()
                raise
            if use_bulk:
                await cast(Callable[[], Awaitable[None]], commit_bulk)()
            return len(replay_result.events)
        finally:
            await store.close()

    count = asyncio.run(_run())
    typer.echo(f"Reprojected {count} events into session {session_id} using {backend}")


@app.command()
def packet_analyzer(
    upstream_base_url: str = typer.Option(
        ...,
        "--upstream-base-url",
        help="Upstream OpenAI-compatible base URL, for example https://api.openai.com/v1",
    ),
    upstream_api_key: str | None = typer.Option(
        None,
        "--upstream-api-key",
        help="Optional upstream bearer token; inbound Authorization is forwarded when omitted",
    ),
    eventloom_path: Path = typer.Option(  # noqa: B008
        Path(".eventloom"),
        "--eventloom-path",
        help="Eventloom directory for packet capture events",
    ),
    session_id: str = typer.Option(
        "default",
        "--session-id",
        help="Session ID to append packet events into",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Local interface for the packet analyzer",
    ),
    port: int = typer.Option(
        8787,
        "--port",
        min=1,
        max=65535,
        help="Local port for the packet analyzer",
    ),
) -> None:
    """Run an observe-only OpenAI-compatible packet analyzer."""
    from zaxy.packet_analyzer import PacketAnalyzerConfig, run_packet_analyzer

    config = PacketAnalyzerConfig(
        eventloom_path=eventloom_path,
        session_id=session_id,
        upstream_base_url=upstream_base_url,
        upstream_api_key=upstream_api_key,
    )
    typer.echo(
        f"Zaxy packet analyzer listening on http://{host}:{port} "
        f"-> {upstream_base_url}"
    )
    run_packet_analyzer(host=host, port=port, config=config)


@app.command("packet-project")
def packet_project(
    eventloom_path: Path = typer.Option(  # noqa: B008
        Path(".eventloom"),
        "--eventloom-path",
        help="Eventloom directory containing captured packet events",
    ),
    session_id: str = typer.Option(
        "default",
        "--session-id",
        help="Session ID containing packet events",
    ),
    from_seq: int = typer.Option(
        1,
        "--from-seq",
        min=1,
        help="First Eventloom sequence number to inspect",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum completed packet events to inspect",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        help="Continuously poll for new packet captures",
    ),
    interval_seconds: float = typer.Option(
        2.0,
        "--interval-seconds",
        min=0.0,
        help="Seconds between watch-mode projection passes",
    ),
    watch_iterations: int | None = typer.Option(
        None,
        "--watch-iterations",
        min=1,
        help="Optional bounded watch pass count for supervisors and tests",
    ),
    graph: bool = typer.Option(
        False,
        "--graph",
        help="Best-effort upsert newly projected packet memory into Neo4j",
    ),
    neo4j_uri: str = typer.Option("bolt://localhost:7687", help="Neo4j Bolt URI"),
    neo4j_user: str = typer.Option("neo4j", help="Neo4j username"),
    neo4j_password: str = typer.Option("testpassword", help="Neo4j password"),
) -> None:
    """Project captured LLM packets into compact memory events."""
    import asyncio

    from zaxy.packet_projection import project_packet_events, watch_packet_events

    graph_projected = 0
    graph_failed = 0

    def project_watch_result_to_graph(result: Any) -> None:
        nonlocal graph_projected, graph_failed
        graph_result = asyncio.run(
            _project_packet_result_to_graph(
                result,
                session_id=session_id,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
            )
        )
        graph_projected += graph_result.projected
        graph_failed += graph_result.failed

    if watch:
        watch_result = watch_packet_events(
            eventloom_path=eventloom_path,
            session_id=session_id,
            from_seq=from_seq,
            limit=limit,
            interval_seconds=interval_seconds,
            max_iterations=watch_iterations,
            on_projected=project_watch_result_to_graph if graph else None,
        )
        noun = "pass" if watch_result.iterations == 1 else "passes"
        graph_text = (
            f", graph_projected={graph_projected}, graph_failed={graph_failed}" if graph else ""
        )
        typer.echo(
            f"Watched {watch_result.iterations} projection {noun} "
            f"(read={watch_result.read}, projected={watch_result.projected}, "
            f"skipped={watch_result.skipped}{graph_text})"
        )
        return
    project_result = project_packet_events(
        eventloom_path=eventloom_path,
        session_id=session_id,
        from_seq=from_seq,
        limit=limit,
    )
    if graph:
        graph_result = asyncio.run(
            _project_packet_result_to_graph(
                project_result,
                session_id=session_id,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
            )
        )
        graph_projected = graph_result.projected
        graph_failed = graph_result.failed
    noun = "event" if project_result.projected == 1 else "events"
    graph_text = (
        f", graph_projected={graph_projected}, graph_failed={graph_failed}" if graph else ""
    )
    typer.echo(
        f"Projected {project_result.projected} packet {noun} "
        f"(read={project_result.read}, skipped={project_result.skipped}{graph_text})"
    )
