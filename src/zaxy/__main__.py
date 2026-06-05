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

import sys

if len(sys.argv) > 1 and sys.argv[1] == "--version":
    from zaxy.release import package_version

    print(f"zaxy {package_version()}")
    raise SystemExit(0)

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

if TYPE_CHECKING:
    from zaxy.config import Settings


def _memory_fabric(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for CLI commands that construct MemoryFabric."""
    from zaxy.core import MemoryFabric as _MemoryFabric

    return _MemoryFabric(*args, **kwargs)


MemoryFabric = _memory_fabric


def _graph_store(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for CLI commands that construct GraphStore."""
    from zaxy.graph import GraphStore as _GraphStore

    return _GraphStore(*args, **kwargs)


GraphStore = _graph_store


def capture_codex_sessions(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for local Codex capture."""
    from zaxy.codex_capture import capture_codex_sessions as _capture_codex_sessions

    return _capture_codex_sessions(*args, **kwargs)


def _local_embedded_graph_runtime(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for embedded runtime checks."""
    from zaxy.runtime import LocalEmbeddedGraphRuntime as _LocalEmbeddedGraphRuntime

    return _LocalEmbeddedGraphRuntime(*args, **kwargs)


LocalEmbeddedGraphRuntime = _local_embedded_graph_runtime


def _local_pggraph_runtime(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for pgGraph runtime checks."""
    from zaxy.runtime import LocalPgGraphRuntime as _LocalPgGraphRuntime

    return _LocalPgGraphRuntime(*args, **kwargs)


LocalPgGraphRuntime = _local_pggraph_runtime


def apply_onboarding_preset(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for onboarding preset expansion."""
    from zaxy.onboarding import apply_onboarding_preset as _apply_onboarding_preset

    return _apply_onboarding_preset(*args, **kwargs)


async def run_onboarding(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for the async onboarding orchestrator."""
    from zaxy.onboarding import run_onboarding as _run_onboarding

    return await _run_onboarding(*args, **kwargs)


def format_onboarding_result(*args: Any, **kwargs: Any) -> str:
    """Patchable lazy seam for onboarding result rendering."""
    from zaxy.onboarding import format_onboarding_result as _format_onboarding_result

    return str(_format_onboarding_result(*args, **kwargs))

app = typer.Typer(help="Zaxy: Event-sourced temporal knowledge graph fabric")
memory_app = typer.Typer(help="Inspect Eventloom-backed agent memory")
memory_purpose_app = typer.Typer(help="Inspect replay-backed purpose control-plane diagnostics")
capture_app = typer.Typer(help="Manage deterministic capture watchers")
coordinate_app = typer.Typer(help="Coordinate parent missions and worker sessions")
coordinate_worker_app = typer.Typer(help="Manage worker sessions for a mission")
coordinate_template_app = typer.Typer(help="Inspect and apply Coordinate mission templates")
coordinate_benchmark_adapter_app = typer.Typer(help="Validate and export CoordinationBench adapter contracts")
trace_app = typer.Typer(help="Export neutral trace correlations from Eventloom")
experimental_app = typer.Typer(help="Run isolated experimental memory research commands")
app.add_typer(memory_app, name="memory")
memory_app.add_typer(memory_purpose_app, name="purpose")
app.add_typer(capture_app, name="capture")
app.add_typer(coordinate_app, name="coordinate")
app.add_typer(trace_app, name="trace")
app.add_typer(experimental_app, name="experimental")
coordinate_app.add_typer(coordinate_worker_app, name="worker")
coordinate_app.add_typer(coordinate_template_app, name="template")
coordinate_app.add_typer(coordinate_benchmark_adapter_app, name="benchmark-adapter")


def _version_callback(value: bool) -> None:
    if value:
        from zaxy.release import package_version

        typer.echo(f"zaxy {package_version()}")
        raise typer.Exit()


@app.callback()
def _main_callback(
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed Zaxy version and exit.",
    ),
) -> None:
    """Zaxy command line interface."""


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
        sessions.append(
            {
                "session_id": path.stem,
                "event_count": len(replay.events),
                "integrity_ok": replay.integrity.ok,
                "latest_seq": replay.events[-1].seq if replay.events else None,
                "latest_hash": replay.events[-1].hash if replay.events else None,
            }
        )
        if not replay.integrity.ok:
            raise typer.BadParameter(
                f"Eventloom integrity failed for {path.name}: {replay.integrity.broken_reason}",
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


def _trace_payload_jsonl(payload: dict[str, Any]) -> str:
    records: list[dict[str, Any]] = [
        {
            "record_type": "summary",
            "format": "zaxy.trace.v0.8.jsonl",
            **cast(dict[str, Any], payload.get("summary", {})),
        }
    ]
    records.extend({"record_type": "session", **session} for session in cast(list[dict[str, Any]], payload.get("sessions", [])))
    records.extend({"record_type": "span", **span} for span in cast(list[dict[str, Any]], payload.get("spans", [])))
    records.extend({"record_type": "edge", **edge} for edge in cast(list[dict[str, Any]], payload.get("edges", [])))
    return "\n".join(json.dumps(record, sort_keys=True) for record in records)


@coordinate_app.command("start")
def coordinate_start(
    objective: str = typer.Argument(..., help="Mission objective"),
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Start a parent coordination mission."""
    from zaxy.coordination import CoordinationManager

    result = CoordinationManager(eventloom_path=eventloom_path).start_mission(mission, objective=objective, actor=actor)
    payload = {
        "mission_id": result.mission_id,
        "event_seq": result.event.seq,
        "event_hash": result.event.hash,
        "objective": objective,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Mission {result.mission_id} started")


def _coordinate_evidence_item(reference: str) -> dict[str, str]:
    command_markers = ("pytest", "unittest", "npm test", "pnpm test", "yarn test", "go test", "cargo test")
    normalized = reference.strip().lower()
    kind = "command" if any(marker in normalized for marker in command_markers) else "source"
    return {"kind": kind, "reference": reference}


@coordinate_worker_app.command("create")
def coordinate_worker_create(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    worker: str = typer.Option(..., "--worker", help="Worker session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Register a worker session under a mission."""
    from zaxy.coordination import CoordinationManager

    result = CoordinationManager(eventloom_path=eventloom_path).create_worker(mission, worker, actor=actor)
    payload = {
        "mission_id": result.mission_id,
        "worker_id": result.worker_id,
        "event_seq": result.event.seq,
        "event_hash": result.event.hash,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Worker {result.worker_id} registered")


@coordinate_app.command("assign")
def coordinate_assign(
    assignment: str = typer.Argument(..., help="Assignment text"),
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    worker: str = typer.Option(..., "--worker", help="Worker session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Assign scoped work to a worker."""
    from zaxy.coordination import CoordinationManager

    result = CoordinationManager(eventloom_path=eventloom_path).assign(
        mission,
        worker,
        assignment,
        actor=actor,
    )
    payload = {
        "mission_id": result.mission_id,
        "worker_id": result.worker_id,
        "assignment": assignment,
        "event_seq": result.event.seq,
        "event_hash": result.event.hash,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Assignment recorded for {result.worker_id}")


@coordinate_template_app.command("list")
def coordinate_template_list(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List built-in Coordinate mission templates."""
    from zaxy.coordination_templates import list_mission_templates

    templates = [template.to_dict() for template in list_mission_templates()]
    if json_output:
        typer.echo(json.dumps({"templates": templates}, indent=2, sort_keys=True))
        return
    for template in templates:
        typer.echo(f"{template['name']}: {template['title']}")


@coordinate_template_app.command("show")
def coordinate_template_show(
    name: str = typer.Argument(..., help="Template name"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show a built-in Coordinate mission template."""
    from zaxy.coordination_templates import get_mission_template

    try:
        template = get_mission_template(name).to_dict()
    except KeyError as exc:
        raise typer.BadParameter(str(exc), param_hint="name") from exc
    if json_output:
        typer.echo(json.dumps(template, indent=2, sort_keys=True))
        return
    typer.echo(f"{template['title']} ({template['name']})")
    typer.echo(f"Objective: {template['objective']}")
    for worker in template["workers"]:
        typer.echo(f"- {worker['worker_id']}: {worker['assignment']}")


@coordinate_template_app.command("apply")
def coordinate_template_apply(
    name: str = typer.Argument(..., help="Template name"),
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the events"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Create a mission from a built-in template."""
    from zaxy.coordination import CoordinationManager
    from zaxy.coordination_templates import apply_mission_template

    manager = CoordinationManager(eventloom_path=eventloom_path)
    try:
        payload = apply_mission_template(manager, name, mission_id=mission, actor=actor)
    except KeyError as exc:
        raise typer.BadParameter(str(exc), param_hint="name") from exc
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Mission {payload['mission_id']} created from {payload['template']} "
            f"with {payload['worker_count']} workers"
        )


@coordinate_app.command("report")
def coordinate_report(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    worker: str = typer.Option(..., "--worker", help="Worker session ID"),
    summary: str = typer.Option(..., "--summary", help="Finding summary"),
    evidence: list[str] | None = typer.Option(None, "--evidence", help="Evidence reference"),  # noqa: B008
    capture_git: bool = typer.Option(False, "--capture-git", help="Attach read-only git branch/worktree metadata as evidence"),
    git_workspace: Path = typer.Option(".", "--git-workspace", help="Workspace used for --capture-git"),  # noqa: B008
    git_metadata: Path | None = typer.Option(None, "--git-metadata", help="Attach read-only git metadata from this workspace"),  # noqa: B008
    test_command: str | None = typer.Option(None, "--test-command", help="Test command to attach as structured evidence"),
    test_status: str | None = typer.Option(None, "--test-status", help="Test status to attach with --test-command"),
    test_summary: str | None = typer.Option(None, "--test-summary", help="Optional test-result summary"),
    test_result_json: str | None = typer.Option(None, "--test-result-json", help="Structured test result JSON to attach as evidence"),
    claim_key: str | None = typer.Option(None, "--claim-key", help="Deterministic claim key for conflict checks"),
    claim_value: str | None = typer.Option(None, "--claim-value", help="Claim value for conflict checks"),
    confidence: float | None = typer.Option(None, "--confidence", help="Finding confidence from 0.0 to 1.0"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("worker", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Report a worker-local finding."""
    from zaxy.coordination import CoordinationManager
    from zaxy.coordination_git import build_test_result_evidence, capture_git_metadata

    evidence_items = [_coordinate_evidence_item(item) for item in evidence or []]
    test_results = []
    if test_result_json:
        test_results.append(_coordinate_test_result_from_json(test_result_json))
    if test_command:
        if not test_status:
            raise typer.BadParameter("--test-status is required with --test-command")
        test_results.append(
            build_test_result_evidence(test_command, status=test_status, summary=test_summary)
        )
    git_target = git_metadata or (git_workspace if capture_git else None)
    if git_target is not None:
        evidence_items.append(capture_git_metadata(git_target, test_results=test_results))
    evidence_items.extend(test_results)
    result = CoordinationManager(eventloom_path=eventloom_path).report_finding(
        mission,
        worker,
        summary=summary,
        actor=actor,
        evidence=evidence_items,
        confidence=confidence,
        claim_key=claim_key,
        claim_value=claim_value,
    )
    payload = {
        "mission_id": result.mission_id,
        "worker_id": result.worker_id,
        "finding_id": result.finding_id,
        "event_seq": result.event.seq,
        "event_hash": result.event.hash,
        "summary": summary,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Finding {result.finding_id} reported")


@coordinate_app.command("decide")
def coordinate_decide(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    finding: str = typer.Option(..., "--finding", help="Finding ID"),
    status: str = typer.Option(..., "--status", help="accepted, rejected, deferred, or conflicted"),
    rationale: str | None = typer.Option(None, "--rationale", help="Decision rationale"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Review a worker finding."""
    from zaxy.coordination import CoordinationManager

    result = CoordinationManager(eventloom_path=eventloom_path).review_finding(
        mission,
        finding,
        status=status,
        actor=actor,
        rationale=rationale,
    )
    payload = {
        "mission_id": result.mission_id,
        "worker_id": result.worker_id,
        "finding_id": result.finding_id,
        "status": status,
        "event_seq": result.event.seq,
        "event_hash": result.event.hash,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Finding {result.finding_id} reviewed as {status}")


@coordinate_app.command("promote")
def coordinate_promote(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    finding: str = typer.Option(..., "--finding", help="Finding ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Promote a finding into accepted parent mission state."""
    from zaxy.coordination import CoordinationManager

    result = CoordinationManager(eventloom_path=eventloom_path).promote_finding(mission, finding, actor=actor)
    payload = {
        "mission_id": result.mission_id,
        "worker_id": result.worker_id,
        "finding_id": result.finding_id,
        "summary": result.summary,
        "event_seq": result.event.seq,
        "event_hash": result.event.hash,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Finding {result.finding_id} promoted")


@coordinate_app.command("brief")
def coordinate_brief(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    semantic_conflicts: str = typer.Option(
        "none",
        "--semantic-conflicts",
        help="Semantic conflict detector: none, lexical, or http",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Print a governed mission brief."""

    brief = _coordinate_manager(
        eventloom_path,
        semantic_conflicts=semantic_conflicts,
    ).brief(mission)
    payload = brief.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"Mission {brief.mission_id}: {brief.objective or '-'}")
    typer.echo(f"Workers: {len(brief.workers)}")
    typer.echo(f"Accepted findings: {len(brief.accepted_findings)}")
    typer.echo(f"Pending findings: {len(brief.pending_findings)}")
    typer.echo(f"Conflicts: {len(brief.conflicts)}")


@coordinate_app.command("inspect")
def coordinate_inspect(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    semantic_conflicts: str = typer.Option(
        "none",
        "--semantic-conflicts",
        help="Semantic conflict detector: none, lexical, or http",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Inspect the full replayed mission state for operators."""

    inspection = _coordinate_manager(
        eventloom_path,
        semantic_conflicts=semantic_conflicts,
    ).inspect_mission(mission)
    payload = inspection.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(_format_coordinate_inspection(payload))


@coordinate_app.command("checkout")
def coordinate_checkout(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    include_diagnostics: bool = typer.Option(False, "--include-diagnostics", help="Include pending and conflict diagnostics"),
    semantic_conflicts: str = typer.Option(
        "none",
        "--semantic-conflicts",
        help="Semantic conflict detector for diagnostics: none, lexical, or http",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Print accepted mission state for prompt injection."""

    checkout = _coordinate_manager(
        eventloom_path,
        semantic_conflicts=semantic_conflicts,
    ).checkout(
        mission,
        include_diagnostics=include_diagnostics,
    )
    payload = checkout.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(checkout.prompt)


@coordinate_app.command("ledger")
def coordinate_ledger(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Print worker-level coordination outcome metrics."""
    from zaxy.coordination import CoordinationManager

    ledger = CoordinationManager(eventloom_path=eventloom_path).performance_ledger(mission)
    payload = ledger.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"Mission {ledger.mission_id}: {ledger.objective or '-'}")
    typer.echo(f"Workers: {ledger.worker_count}")
    typer.echo(f"Findings: {ledger.total_findings}")
    for worker in ledger.workers:
        typer.echo(
            f"{worker.worker_id}: accepted={worker.accepted_findings} "
            f"rejected={worker.rejected_findings} missing_evidence={worker.missing_evidence_count}"
        )


def _format_coordinate_inspection(payload: dict[str, Any]) -> str:
    findings = cast(dict[str, list[Any]], payload.get("findings")) if isinstance(payload.get("findings"), dict) else {}
    worker_ledgers = cast(list[Any], payload.get("worker_ledgers")) if isinstance(payload.get("worker_ledgers"), list) else []
    promoted_state = (
        cast(list[Any], payload.get("promoted_state")) if isinstance(payload.get("promoted_state"), list) else []
    )
    handoffs = cast(list[Any], payload.get("handoffs")) if isinstance(payload.get("handoffs"), list) else []
    conflicts = cast(list[Any], payload.get("conflicts")) if isinstance(payload.get("conflicts"), list) else []
    decisions = cast(list[Any], payload.get("decisions")) if isinstance(payload.get("decisions"), list) else []
    lines = [
        f"Mission {payload.get('mission_id')}: {payload.get('objective') or '-'}",
        f"Workers: {len(worker_ledgers)}",
        "Worker ledgers:",
    ]
    for worker in worker_ledgers:
        if not isinstance(worker, dict):
            continue
        lines.append(
            f"- {worker.get('worker_id')}: accepted={worker.get('accepted_findings', 0)} "
            f"pending={worker.get('pending_findings', 0)} "
            f"rejected={worker.get('rejected_findings', 0)} "
            f"missing_evidence={worker.get('missing_evidence_count', 0)}"
        )
    lines.extend(
        [
            "Findings: "
            f"accepted={len(findings.get('accepted', []))} "
            f"pending={len(findings.get('pending', []))} "
            f"rejected={len(findings.get('rejected', []))} "
            f"deferred={len(findings.get('deferred', []))} "
            f"conflicted={len(findings.get('conflicted', []))} "
            f"stale={len(findings.get('stale', []))}",
            f"Conflicts: {len(conflicts)}",
            f"Decisions: {len(decisions)}",
            f"Promoted state: {_finding_ids(promoted_state) or '-'}",
            f"Handoffs: {len(handoffs)}",
        ]
    )
    return "\n".join(lines)


def _finding_ids(findings: list[Any]) -> str:
    return ", ".join(
        str(finding.get("finding_id"))
        for finding in findings
        if isinstance(finding, dict) and finding.get("finding_id")
    )


@coordinate_app.command("handoff")
def coordinate_handoff(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    summary: str = typer.Option(..., "--summary", help="Final handoff summary"),
    next_step: list[str] | None = typer.Option(None, "--next-step", help="Next step to include in the handoff"),  # noqa: B008
    risk: list[str] | None = typer.Option(None, "--risk", help="Risk to include in the handoff"),  # noqa: B008
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Create a final replayable coordination handoff."""
    from zaxy.coordination import CoordinationManager

    result = CoordinationManager(eventloom_path=eventloom_path).create_handoff(
        mission,
        summary=summary,
        next_steps=next_step,
        risks=risk,
        actor=actor,
    )
    payload = {
        "event_type": result.event.type,
        "mission_id": result.mission_id,
        "handoff_id": result.handoff_id,
        "summary": result.summary,
        "next_steps": result.event.payload.get("next_steps", []),
        "risks": result.event.payload.get("risks", []),
        "event_seq": result.event.seq,
        "event_hash": result.event.hash,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"Handoff {result.handoff_id} recorded")


@coordinate_app.command("detect-conflicts")
def coordinate_detect_conflicts(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    semantic_conflicts: str = typer.Option(
        "none",
        "--semantic-conflicts",
        help="Semantic conflict detector to materialize: none, lexical, or http",
    ),
    actor: str = typer.Option("zaxy", help="Actor recording detected conflict events"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Materialize deterministic coordination conflicts as Eventloom facts."""

    results = _coordinate_manager(
        eventloom_path,
        semantic_conflicts=semantic_conflicts,
    ).record_detected_conflicts(
        mission,
        actor=actor,
    )
    payload = {
        "mission_id": mission,
        "recorded_count": len(results),
        "events": [
            {
                "event_type": result.event.type,
                "event_seq": result.event.seq,
                "event_hash": result.event.hash,
                "conflict_id": result.event.payload.get("conflict_id"),
                "conflict_type": result.event.payload.get("conflict_type"),
                "claim_key": result.event.payload.get("claim_key"),
                "source_reference": result.event.payload.get("source_reference"),
                "finding_ids": result.event.payload.get("finding_ids", []),
            }
            for result in results
        ],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"Recorded {len(results)} conflict event(s)")


@coordinate_app.command("approval-packet")
def coordinate_approval_packet(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Export a portable packet for remote finding approval."""
    from zaxy.coordination import CoordinationManager

    packet = CoordinationManager(eventloom_path=eventloom_path).approval_packet(mission)
    payload = packet.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"Approval packet {packet.packet_id}")
    typer.echo(f"Findings needing review: {len(packet.findings)}")


@coordinate_app.command("review-export")
def coordinate_review_export(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Export a static Markdown review artifact for humans."""
    from zaxy.coordination import CoordinationManager

    review_export = CoordinationManager(eventloom_path=eventloom_path).review_export(mission)
    if json_output:
        typer.echo(json.dumps(review_export.to_dict(), indent=2, sort_keys=True))
        return
    typer.echo(review_export.markdown)


@coordinate_app.command("audit-report")
def coordinate_audit_report(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Generate a replay-only mission audit report with Eventloom citations."""
    from zaxy.coordination import CoordinationManager

    report = CoordinationManager(eventloom_path=eventloom_path).audit_report(mission)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    typer.echo(report.markdown)


@coordinate_app.command("apply-approval")
def coordinate_apply_approval(
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    decisions_json: str = typer.Option(..., "--decisions-json", help="JSON array of approval decisions"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording review events"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Apply remote approval decisions to the parent mission."""
    from zaxy.coordination import CoordinationManager

    decisions = _coordinate_decisions_from_json(decisions_json)
    result = CoordinationManager(eventloom_path=eventloom_path).apply_approval_decisions(
        mission,
        decisions,
        actor=actor,
    )
    payload = result.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"Applied {result.reviewed_count} approval decision(s); promoted {result.promoted_count}")


def _coordinate_decisions_from_json(value: str) -> list[dict[str, Any]]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise typer.BadParameter("decisions JSON must be an array")
    decisions: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise typer.BadParameter("each approval decision must be an object")
        decisions.append(item)
    return decisions


def _coordinate_test_result_from_json(value: str) -> dict[str, Any]:
    from zaxy.coordination_git import build_test_result_evidence

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("test result JSON must be a valid object") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter("test result JSON must be an object")
    command = str(parsed.get("command") or "").strip()
    status = str(parsed.get("status") or "").strip()
    if not command or not status:
        raise typer.BadParameter("test result JSON requires command and status")
    exit_code = parsed.get("exit_code")
    if exit_code is not None:
        try:
            exit_code = int(str(exit_code))
        except ValueError as exc:
            raise typer.BadParameter("test result JSON exit_code must be an integer") from exc
    return build_test_result_evidence(
        command,
        status=status,
        summary=str(parsed.get("summary") or "") or None,
        exit_code=exit_code,
    )


@coordinate_app.command("benchmark")
def coordinate_benchmark(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for CoordinationBench reports"),  # noqa: B008
    missions: int = typer.Option(1, help="Number of missions to generate"),
    workers: int = typer.Option(3, help="Workers per mission, between 3 and 10"),
    workload: Path | None = typer.Option(  # noqa: B008
        None,
        "--workload",
        help="Frozen CoordinationBench workload JSON to run instead of generating the seed workload",
    ),
    competitor_result: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--competitor-result",
        help="Pinned competitor result as NAME=PATH; may be repeated",
    ),
    competitor_runner: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--competitor-runner",
        help="Pinned competitor runner manifest as NAME=PATH; may be repeated",
    ),
    require_competitor_claim: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--require-competitor-claim",
        help="Require named competitor adapters to have completed same-harness local scoring; may be repeated",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Run CoordinationBench against a generated or frozen workload."""
    from zaxy.coordination_benchmark import (
        coordination_competitor_claim_gate,
        run_coordination_benchmark,
    )

    competitor_results = _coordinate_competitor_results_from_options(competitor_result or [])
    competitor_runners = _coordinate_competitor_results_from_options(competitor_runner or [])
    duplicate_adapters = sorted(set(competitor_results) & set(competitor_runners))
    if duplicate_adapters:
        raise typer.BadParameter(f"duplicate competitor adapter: {', '.join(duplicate_adapters)}")
    report = run_coordination_benchmark(
        output_dir,
        missions=missions,
        workers=workers,
        workload_path=workload,
        competitor_results=competitor_results,
        competitor_runners=competitor_runners,
    )
    required_claims = tuple(require_competitor_claim or ())
    if required_claims:
        gate = coordination_competitor_claim_gate(report, required_adapters=required_claims)
        if gate.status != "passed":
            blockers = "; ".join(
                f"{name}: {reason}" for name, reason in sorted(gate.blocked_adapters.items())
            )
            raise typer.BadParameter(f"competitor claim gate blocked: {blockers}")
    payload = report.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"CoordinationBench report written to {output_dir}")
    typer.echo(f"accepted_finding_precision={report.metrics.accepted_finding_precision}")
    typer.echo(f"conflict_recall={report.metrics.conflict_recall}")


@experimental_app.command("pattern-completion")
def experimental_pattern_completion(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for experimental reports"),  # noqa: B008
    workload: Path | None = typer.Option(  # noqa: B008
        None,
        "--workload",
        help="Frozen PatternCompletionBench workload JSON to replay instead of generating the seed workload",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Run the isolated associative pattern-completion benchmark."""
    from zaxy.associative_memory import run_pattern_completion_benchmark

    report = run_pattern_completion_benchmark(output_dir, workload_path=workload)
    payload = report.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"PatternCompletionBench report written to {output_dir}")
    typer.echo(f"associative_latent_state_recall={report.metrics.latent_state_recall}")
    typer.echo(f"direct_lexical_latent_state_recall={report.baselines['direct_lexical'].latent_state_recall}")


@experimental_app.command("state-recovery")
def experimental_state_recovery(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for StateRecoveryBench reports"),  # noqa: B008
    workload: Path | None = typer.Option(  # noqa: B008
        None,
        "--workload",
        help="Frozen StateRecoveryBench workload JSON to replay instead of generating the seed workload",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Run StateRecoveryBench against experimental baselines."""
    _run_state_recovery_benchmark_command(
        output_dir=output_dir,
        workload=workload,
        json_output=json_output,
        fail_on_guardrail=False,
    )


@app.command("state-recovery-benchmark")
def state_recovery_benchmark(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/state-recovery-v1"),
        "--output-dir",
        help="Directory for official StateRecoveryBench reports",
    ),
    workload: Path | None = typer.Option(  # noqa: B008
        None,
        "--workload",
        help="Frozen StateRecoveryBench workload JSON to replay instead of generating the seed workload",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    allow_failures: bool = typer.Option(
        False,
        "--allow-failures",
        help="Write the report without failing the process when production guardrails fail",
    ),
) -> None:
    """Run the official StateRecoveryBench production guardrail lane."""
    _run_state_recovery_benchmark_command(
        output_dir=output_dir,
        workload=workload,
        json_output=json_output,
        fail_on_guardrail=not allow_failures,
    )


def _run_state_recovery_benchmark_command(
    *,
    output_dir: Path,
    workload: Path | None,
    json_output: bool,
    fail_on_guardrail: bool,
) -> None:
    from zaxy.associative_memory import run_state_recovery_benchmark

    report = run_state_recovery_benchmark(output_dir, workload_path=workload)
    payload = report.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"StateRecoveryBench report written to {output_dir}")
        typer.echo(f"status={report.status}")
        typer.echo(f"production_baseline={report.production_baseline}")
        for name, metrics in report.baselines.items():
            typer.echo(
                f"{name}: state_accuracy={metrics.state_accuracy:.3f} "
                f"minimal_evidence_recall={metrics.minimal_evidence_recall:.3f} "
                f"stale_rejection={metrics.stale_rejection:.3f} "
                f"distractor_resistance={metrics.distractor_resistance:.3f} "
                f"abstention_accuracy={metrics.abstention_accuracy:.3f} "
                f"token_cost={metrics.token_cost} latency_ms={metrics.latency_ms:.3f} "
                f"citation_coverage={metrics.citation_coverage:.3f}"
            )
    if fail_on_guardrail and report.status != "pass":
        raise typer.Exit(1)


@coordinate_app.command("adapter-template")
def coordinate_adapter_template(
    adapter: str = typer.Argument(..., help="Adapter target: codex, langgraph, crewai, or mcp"),
    mission: str = typer.Option(..., "--mission", help="Parent mission session ID"),
    worker: str = typer.Option(..., "--worker", help="Worker session ID"),
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory for the template"),
) -> None:
    """Print a dependency-light Zaxy Coordinate adapter starter."""
    from zaxy.integrations import render_coordination_adapter_template

    try:
        template = render_coordination_adapter_template(
            adapter,
            mission_id=mission,
            worker_id=worker,
            eventloom_path=eventloom_path,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(template, nl=False)


@coordinate_benchmark_adapter_app.command("export-kit")
def coordinate_benchmark_adapter_export_kit(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for the CoordinationBench adapter kit"),  # noqa: B008
    missions: int = typer.Option(1, help="Number of missions to generate"),
    workers: int = typer.Option(3, help="Workers per mission, between 3 and 10"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Export schemas, workload, and templates for external CoordinationBench adapters."""
    from zaxy.coordination_benchmark import export_coordination_benchmark_adapter_kit

    payload = export_coordination_benchmark_adapter_kit(
        output_dir,
        missions=missions,
        workers=workers,
    )
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"CoordinationBench adapter kit written to {output_dir}")
    typer.echo(f"workload_fingerprint={payload['workload_fingerprint']}")


@coordinate_benchmark_adapter_app.command("validate-manifest")
def coordinate_benchmark_adapter_validate_manifest(
    adapter: str = typer.Argument(..., help="Adapter manifest as NAME=PATH"),
    workload: Path = typer.Option(..., "--workload", help="Frozen CoordinationBench workload JSON"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Validate a runner manifest without executing it."""
    from zaxy.coordination_benchmark import (
        load_coordination_workload,
        validate_coordination_competitor_runner_manifest,
    )

    adapters = _coordinate_competitor_results_from_options([adapter])
    name, path = next(iter(adapters.items()))
    payload = validate_coordination_competitor_runner_manifest(
        name,
        load_coordination_workload(workload),
        path,
    )
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"{name} manifest valid for {payload['workload_fingerprint']}")


@coordinate_benchmark_adapter_app.command("validate-result")
def coordinate_benchmark_adapter_validate_result(
    adapter: str = typer.Argument(..., help="Adapter result as NAME=PATH"),
    workload: Path = typer.Option(..., "--workload", help="Frozen CoordinationBench workload JSON"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Validate and locally score a competitor result file."""
    from zaxy.coordination_benchmark import (
        load_coordination_workload,
        validate_coordination_competitor_result,
    )

    adapters = _coordinate_competitor_results_from_options([adapter])
    name, path = next(iter(adapters.items()))
    payload = validate_coordination_competitor_result(
        name,
        load_coordination_workload(workload),
        path,
    )
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"{name} result valid for {payload['workload_fingerprint']}")
    typer.echo(f"accepted_finding_precision={payload['metrics']['accepted_finding_precision']}")


def _coordinate_competitor_results_from_options(values: list[str]) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter("competitor result must use NAME=PATH")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise typer.BadParameter("competitor result must use NAME=PATH")
        if name in results:
            raise typer.BadParameter(f"duplicate competitor result: {name}")
        results[name] = Path(path)
    return results


def _coordinate_manager(eventloom_path: Path, *, semantic_conflicts: str = "none") -> Any:
    from zaxy.config import Settings
    from zaxy.coordination import CoordinationManager
    from zaxy.coordination_semantic import build_semantic_conflict_detector

    detector = build_semantic_conflict_detector(
        Settings(
            coordination_semantic_conflict_provider=semantic_conflicts,
            coordination_semantic_min_shared_subject_tokens=2,
        )
    )
    return CoordinationManager(
        eventloom_path=eventloom_path,
        semantic_conflict_detector=detector,
    )


async def _project_packet_result_to_graph(
    result: Any,
    *,
    session_id: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> Any:
    """Best-effort graph projection for newly appended packet memory events."""
    from zaxy.packet_projection import PacketGraphProjectionResult, project_packet_events_to_graph

    if not result.projected_events:
        return PacketGraphProjectionResult(projected=0, failed=0)

    graph = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
    try:
        await graph.connect()
        await graph.init_schema()
        return await project_packet_events_to_graph(
            result.projected_events,
            graph=graph,
            session_id=session_id,
        )
    except Exception:
        return PacketGraphProjectionResult(projected=0, failed=len(result.projected_events))
    finally:
        with suppress(Exception):
            await graph.close()


@memory_app.command("status")
def memory_status(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory or JSONL log"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    graph: bool = typer.Option(False, "--graph", help="Also inspect graph projection integrity"),
    projection_backend: str | None = typer.Option(
        None,
        "--projection-backend",
        help="Projection backend to inspect: embedded, neo4j, pggraph, or latticedb",
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
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
) -> None:
    """Show read-only Eventloom memory status."""
    from zaxy.memory_status import format_memory_status, inspect_memory_status

    status = inspect_memory_status(eventloom_path)
    graph_status: dict[str, object] | None = None
    if graph:
        import asyncio

        from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store

        async def _inspect_graph() -> dict[str, object]:
            settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
            backend = _resolve_cli_projection_backend(
                projection_backend,
                settings,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                pggraph_dsn=pggraph_dsn,
                embedded_graph_path=embedded_graph_path,
            )
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
                projections = []
                for session in status.sessions:
                    projection = await store.inspect_event_projection_status(
                        session.session_id,
                        eventloom_latest_seq=session.latest_seq,
                        eventloom_latest_hash=session.latest_hash,
                    )
                    projections.append(projection.to_dict())
                return {"backend": backend, "sessions": projections}
            finally:
                await store.close()

        graph_status = asyncio.run(_inspect_graph())
    if json_output:
        payload = status.to_dict()
        if graph:
            payload["graph"] = graph_status
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output = format_memory_status(status)
        if graph and graph_status is not None:
            output = "\n".join([output, "", _format_memory_graph_status(graph_status)])
        typer.echo(output)


def _format_memory_graph_status(graph_status: dict[str, object]) -> str:
    """Format graph projection status for humans."""
    backend = graph_status.get("backend", "unknown")
    graph_sessions = cast(list[dict[str, object]], graph_status.get("sessions", []))
    lines = [f"Graph projection (backend={backend}):"]
    if not graph_sessions:
        lines.append("  no Eventloom sessions to inspect")
        return "\n".join(lines)
    for session in graph_sessions:
        session_id = session.get("session_id", "-")
        integrity = "OK" if session.get("integrity_ok") is True else "FAILED"
        latest = session.get("latest_seq") or "-"
        lag = session.get("projection_lag")
        lag_text = "-" if lag is None else str(lag)
        missing = session.get("missing_chain_links", 0)
        lines.append(
            f"  {session_id}: graph_latest={latest} lag={lag_text} "
            f"missing_chain_links={missing} integrity={integrity}"
        )
    return "\n".join(lines)


@memory_purpose_app.command("status")
def memory_purpose_status(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory or JSONL log"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Session ID to inspect"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show replay-only purpose checkout, evidence, feedback, and Coordinate diagnostics."""
    from zaxy.purpose_control import build_purpose_status, format_purpose_status

    payload = build_purpose_status(eventloom_path, session_id=session_id)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(format_purpose_status(payload))


@memory_purpose_app.command("lanes")
def memory_purpose_lanes(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory or JSONL log"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Session ID to inspect"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show per-profile purpose lanes from Eventloom replay only."""
    from zaxy.purpose_control import build_purpose_lanes, format_purpose_lanes

    payload = build_purpose_lanes(eventloom_path, session_id=session_id)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(format_purpose_lanes(payload))


@memory_purpose_app.command("feedback")
def memory_purpose_feedback(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory or JSONL log"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Session ID to inspect"),  # noqa: B008
    profile: str | None = typer.Option(None, help="Purpose profile to filter by"),  # noqa: B008
    outcome: str = typer.Option("all", help="Outcome filter: all, positive, or negative"),
    limit: int = typer.Option(20, min=1, max=250, help="Maximum feedback targets to show"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show retained purpose feedback and consequence history."""
    from zaxy.purpose_control import build_purpose_feedback, format_purpose_feedback

    normalized_outcome = outcome.strip().casefold()
    if normalized_outcome not in {"all", "positive", "negative"}:
        raise typer.BadParameter("--outcome must be all, positive, or negative")
    payload = build_purpose_feedback(
        eventloom_path,
        session_id=session_id,
        profile=profile,
        outcome=normalized_outcome,
        limit=limit,
    )
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(format_purpose_feedback(payload))


@memory_app.command("inferred-status")
def memory_inferred_status(
    session_id: str = typer.Option("default", help="Session ID to inspect"),
    limit: int = typer.Option(10, help="Number of representative inferred edges to show"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    projection_backend: str | None = typer.Option(
        None,
        "--projection-backend",
        help="Projection backend to inspect: embedded, neo4j, pggraph, or latticedb",
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
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
) -> None:
    """Show read-only graph audit status for inferred relationships."""
    import asyncio

    from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store

    async def _inspect_graph() -> dict[str, object]:
        settings = _status_settings()
        backend = _resolve_cli_projection_backend(
            projection_backend,
            settings,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            pggraph_dsn=pggraph_dsn,
            embedded_graph_path=embedded_graph_path,
        )
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
            status = await store.inspect_inferred_edge_status(session_id, limit=limit)
            payload = status.to_dict()
            payload["backend"] = backend
            return payload
        finally:
            await store.close()

    payload = asyncio.run(_inspect_graph())
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(_format_memory_inferred_status(payload))


def _format_memory_inferred_status(status: dict[str, object]) -> str:
    """Format inferred-edge audit status for humans."""
    session_id = status.get("session_id", "-")
    backend = status.get("backend", "unknown")
    total_edges = _format_int_value(status.get("total_edges"))
    method_count = _format_int_value(status.get("method_count"))
    evidence_coverage = _format_float_value(status.get("evidence_coverage"))
    missing_evidence = _format_int_value(status.get("missing_evidence_count"))
    missing_source_events = _format_int_value(status.get("missing_source_event_count"))
    lines = [
        f"Inferred edges: {session_id} (backend={backend})",
        (
            f"  total={total_edges} methods={method_count} "
            f"evidence_coverage={evidence_coverage:.1%} "
            f"missing_evidence={missing_evidence} "
            f"missing_source_events={missing_source_events}"
        ),
    ]
    methods = status.get("methods")
    if isinstance(methods, list) and methods:
        lines.append("Methods:")
        for method in methods:
            if not isinstance(method, dict):
                continue
            relation_types = method.get("relation_types")
            relation_text = ", ".join(str(value) for value in relation_types or [])
            avg = _format_optional_float(method.get("average_confidence"))
            minimum = _format_optional_float(method.get("minimum_confidence"))
            lines.append(
                "  "
                f"{method.get('method', 'unknown')}: "
                f"edges={method.get('edge_count', 0)} "
                f"avg_confidence={avg} min_confidence={minimum} "
                f"missing_evidence={method.get('missing_evidence_count', 0)} "
                f"relations={relation_text or '-'}"
            )
    samples = status.get("samples")
    if isinstance(samples, list) and samples:
        lines.append("Samples:")
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            confidence = _format_optional_float(sample.get("confidence"))
            source_event_seq = sample.get("source_event_seq") or "-"
            evidence_keys = sample.get("evidence_keys")
            evidence_text = ", ".join(str(value) for value in evidence_keys or [])
            lines.append(
                "  "
                f"{sample.get('source', '')} -[{sample.get('relation_type', '')}]-> "
                f"{sample.get('target', '')} "
                f"confidence={confidence} method={sample.get('method', 'unknown')} "
                f"event_seq={source_event_seq} evidence={evidence_text or '-'}"
            )
    return "\n".join(lines)


def _format_optional_float(value: object) -> str:
    """Format an optional float-ish value for compact CLI output."""
    if isinstance(value, bool):
        return "-"
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "-"


def _format_int_value(value: object) -> int:
    """Return an integer value from trusted formatter payloads."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _format_float_value(value: object) -> float:
    """Return a float value from trusted formatter payloads."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


@memory_app.command("capabilities")
def memory_capabilities(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to describe"),
    current_task: str | None = typer.Option(None, help="Current task or question to seed checkout guidance"),  # noqa: B008
    workspace_root: Path = typer.Option(Path("."), help="Workspace root for hook/status discovery"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show model-facing Zaxy memory capabilities and usage guidance."""
    from zaxy.capabilities import build_memory_capabilities, format_memory_capabilities

    manifest = build_memory_capabilities(
        eventloom_path=eventloom_path,
        session_id=session_id,
        workspace_root=workspace_root,
        current_task=current_task,
    )
    if json_output:
        typer.echo(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        typer.echo(format_memory_capabilities(manifest))


@memory_app.command("bootstrap")
def memory_bootstrap(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to bootstrap"),
    current_task: str | None = typer.Option(None, help="Current task or question to seed checkout guidance"),  # noqa: B008
    workspace_root: Path = typer.Option(Path("."), help="Workspace root for capture/status discovery"),  # noqa: B008
    launch: bool = typer.Option(False, "--launch", help="Start the agent client with activation context"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the launch command without starting the client"),
    codex_executable: str = typer.Option("codex", help="Codex executable for --launch"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show compact session-start Zaxy memory bootstrap guidance."""
    from zaxy.capabilities import build_memory_bootstrap, format_memory_bootstrap

    bootstrap = build_memory_bootstrap(
        eventloom_path=eventloom_path,
        session_id=session_id,
        workspace_root=workspace_root,
        current_task=current_task,
    )
    from zaxy.memory_persistence import record_memory_activity

    record_memory_activity(
        eventloom_path,
        session_id=session_id,
        activity="bootstrap",
        source="cli",
        query=current_task,
    )
    if json_output:
        typer.echo(json.dumps(bootstrap, indent=2, sort_keys=True))
    else:
        typer.echo(format_memory_bootstrap(bootstrap))


@memory_app.command("checkout")
def memory_checkout(
    query: str = typer.Argument(..., help="Question or task to checkout memory for"),  # noqa: B008
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to checkout"),
    ref: str | None = typer.Option(None, help="Memory ref to checkout, e.g. HEAD or refs/heads/main"),
    replay_from_seq: int = typer.Option(1, min=1, help="Replay start sequence"),
    limit: int = typer.Option(10, min=1, help="Maximum retrieved context items"),
    max_recent_events: int = typer.Option(20, min=1, help="Maximum recent replay events"),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    neo4j_ca_cert: str | None = typer.Option(None, help="Neo4j CA certificate path; pass an empty value to disable TLS CA override"),
    neo4j_trust_all: bool | None = typer.Option(None, help="Trust all Neo4j TLS certificates"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Checkout current, cited memory state for an agent turn."""
    import asyncio

    async def _checkout() -> dict[str, object]:
        settings = _status_settings(_profile_root_for_eventloom_path(eventloom_path))
        fabric = MemoryFabric(
            eventloom_path=str(eventloom_path),
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_ca_cert=neo4j_ca_cert,
            neo4j_trust_all=neo4j_trust_all,
            projection_backend=_resolve_cli_projection_backend(
                None,
                settings,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
            ),
            pggraph_dsn=settings.pggraph_dsn,
            embedded_graph_path=Path(settings.embedded_graph_path),
            latticedb_path=Path(settings.latticedb_path),
        )
        await fabric.connect()
        try:
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
            await fabric.close()

    payload = asyncio.run(_checkout())
    from zaxy.memory_persistence import record_memory_activity

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
        typer.echo(payload["prompt"])


@memory_app.command("log")
def memory_log(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory or JSONL log"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Session ID to inspect"),  # noqa: B008
    limit: int = typer.Option(20, min=0, help="Maximum events to print"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show recent Eventloom memory events."""
    from zaxy.memory_status import format_memory_log, inspect_memory_log

    try:
        memory = inspect_memory_log(eventloom_path, session_id=session_id, limit=limit)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(json.dumps(memory.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(format_memory_log(memory))


@memory_app.command("diff")
def memory_diff(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory or JSONL log"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Session ID to inspect"),  # noqa: B008
    from_seq: int = typer.Option(..., "--from-seq", min=1, help="First sequence number"),
    to_seq: int = typer.Option(..., "--to-seq", min=1, help="Last sequence number"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show event-level Eventloom changes in an inclusive sequence range."""
    from zaxy.memory_status import format_memory_diff, inspect_memory_diff

    try:
        diff = inspect_memory_diff(
            eventloom_path,
            session_id=session_id,
            from_seq=from_seq,
            to_seq=to_seq,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(json.dumps(diff.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(format_memory_diff(diff))


@memory_app.command("ref")
def memory_ref_update(
    name: str = typer.Argument(..., help="Memory ref name, e.g. refs/heads/main"),  # noqa: B008
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option(..., help="Session ID the ref points to"),  # noqa: B008
    target_seq: int = typer.Option(..., min=1, help="Target Eventloom sequence"),
    target_hash: str = typer.Option(..., help="Target Eventloom hash"),
    ref_type: str = typer.Option("ref", "--type", help="Ref type, e.g. branch, tag, checkpoint"),
    actor: str = typer.Option("zaxy", help="Actor writing the ref"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Create or update a durable memory ref."""
    from zaxy.refs import MemoryRefStore

    store = MemoryRefStore(eventloom_path)
    try:
        event = store.update_ref(
            name,
            session_id=session_id,
            target_seq=target_seq,
            target_hash=target_hash,
            ref_type=ref_type,
            actor=actor,
        )
        ref = store.resolve(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {"event": event.model_dump(), "ref": ref.to_dict() if ref is not None else None}
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"{name} -> {session_id}@{target_seq} {target_hash[:12]}")


@memory_app.command("refs")
def memory_refs_list(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List durable memory refs."""
    from zaxy.refs import MemoryRefStore

    refs = MemoryRefStore(eventloom_path).list_refs()
    payload = {
        "eventloom_path": str(eventloom_path.resolve()),
        "refs": [ref.to_dict() for ref in refs],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    elif refs:
        typer.echo(
            "\n".join(
                f"{ref.name} -> {ref.session_id}@{ref.target_seq} {ref.target_hash[:12]}"
                for ref in refs
            )
        )
    else:
        typer.echo("No memory refs")


@app.command("activate")
def activate(
    client: str = typer.Argument(..., help="Agent client to activate: codex"),  # noqa: B008
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to activate"),
    current_task: str | None = typer.Option(None, help="Current task or question to seed checkout guidance"),  # noqa: B008
    workspace_root: Path = typer.Option(Path("."), help="Workspace root for capture/status discovery"),  # noqa: B008
    launch: bool = typer.Option(False, "--launch", help="Start the agent client with activation context"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the launch command without starting the client"),
    codex_executable: str = typer.Option("codex", help="Codex executable for --launch"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Emit a prompt-ready memory activation packet for an agent client."""
    normalized_client = client.casefold().strip().replace("_", "-")
    if normalized_client != "codex":
        raise typer.BadParameter("activate currently supports: codex", param_hint="client")
    from zaxy.capabilities import build_memory_bootstrap

    bootstrap = build_memory_bootstrap(
        eventloom_path=eventloom_path,
        session_id=session_id,
        workspace_root=workspace_root,
        current_task=current_task,
    )
    from zaxy.memory_persistence import record_memory_activity

    record_memory_activity(
        eventloom_path,
        session_id=session_id,
        activity="bootstrap",
        source="activate-codex",
        query=current_task,
    )
    packet = {
        "client": normalized_client,
        "mode": "session_start_injection",
        "session_id": session_id,
        "workspace": str(workspace_root.resolve()),
        "bootstrap": bootstrap,
        "injection_text": bootstrap["prompt"],
        "next_step": "Start Codex with this activation packet in session-start context, then run memory_checkout.",
    }
    if launch:
        command = _codex_activation_command(codex_executable, workspace_root.resolve(), str(packet["injection_text"]))
        if dry_run:
            typer.echo(_shell_join(command))
            return
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise typer.Exit(result.returncode)
        return
    if json_output:
        typer.echo(json.dumps(packet, indent=2, sort_keys=True))
    else:
        typer.echo(_format_activation_packet(packet))


@app.command("ide-config")
def ide_config(
    client: str = typer.Argument(..., help="MCP client: claude-desktop, claude-code, codex, cursor, hermes, or vscode"),  # noqa: B008
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory for this client"),
    transport: str = typer.Option("stdio", help="Transport: stdio or sse"),
    host: str = typer.Option("127.0.0.1", help="SSE host when transport=sse"),
    port: int = typer.Option(8080, help="SSE port when transport=sse"),
    domain: str | None = typer.Option(None, help="Project/domain used for default session scoping"),  # noqa: B008
    zaxy_executable: str | None = typer.Option(None, help="Executable path MCP clients should invoke"),  # noqa: B008
    install: bool = typer.Option(False, "--install", help="Merge into the verified project-local client config"),  # noqa: B008
    workspace: Path = typer.Option(Path("."), help="Workspace root for --install"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Replace an existing zaxy server during --install"),  # noqa: B008
    codex_config_scope: str | None = typer.Option(None, help="Codex direct config scope: project or user"),  # noqa: B008
    codex_home: Path | None = typer.Option(None, help="CODEX_HOME override for Codex user config"),  # noqa: B008
    codex_trusted_project: bool = typer.Option(False, "--codex-trusted-project", help="Acknowledge that Codex trusts this project config"),  # noqa: B008
    hermes_config: Path | None = typer.Option(None, help="Hermes config.yaml path for --install"),  # noqa: B008
) -> None:
    """Print or install a first-run MCP client configuration fragment."""
    from zaxy.integrations import (
        render_codex_mcp_add_command,
        render_mcp_client_config,
        write_codex_mcp_config,
        write_hermes_mcp_config,
        write_project_mcp_client_config,
    )

    try:
        if install:
            if client.casefold().replace("_", "-") == "codex":
                if codex_config_scope is not None:
                    written = write_codex_mcp_config(
                        scope=codex_config_scope,
                        workspace=workspace,
                        eventloom_path=eventloom_path,
                        domain=domain,
                        zaxy_executable=zaxy_executable,
                        force=force,
                        trusted_project=codex_trusted_project,
                        codex_home=codex_home,
                    )
                    typer.echo(f"Wrote Codex MCP config to {written}")
                    return
                command = render_codex_mcp_add_command(
                    eventloom_path=eventloom_path,
                    domain=domain,
                    zaxy_executable=zaxy_executable,
                )
                typer.echo("Run this Codex MCP install command:")
                typer.echo(_shell_join(command))
                return
            if client.casefold().replace("_", "-") == "hermes":
                written = write_hermes_mcp_config(
                    config_path=hermes_config,
                    zaxy_executable=zaxy_executable,
                    force=force,
                    domain=domain,
                )
                typer.echo(f"Wrote Hermes MCP config to {written}")
                return
            written = write_project_mcp_client_config(
                client,
                workspace=workspace,
                eventloom_path=eventloom_path,
                transport=transport,
                host=host,
                port=port,
                domain=domain,
                zaxy_executable=zaxy_executable,
                force=force,
            )
            typer.echo(f"Installed {client} MCP config to {written}")
            return
        if client.casefold().replace("_", "-") == "codex":
            command = render_codex_mcp_add_command(
                eventloom_path=eventloom_path,
                domain=domain,
                zaxy_executable=zaxy_executable,
            )
            typer.echo("Run this Codex MCP install command:")
            typer.echo(_shell_join(command))
            return
        config = render_mcp_client_config(
            client,
            eventloom_path=eventloom_path,
            transport=transport,
            host=host,
            port=port,
            domain=domain,
            zaxy_executable=zaxy_executable,
        )
    except (FileExistsError, PermissionError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if client.casefold().replace("_", "-") == "hermes":
        import yaml

        typer.echo(yaml.safe_dump(config, sort_keys=False).rstrip())
        return
    typer.echo(json.dumps(config, indent=2, sort_keys=True))


def _shell_join(command: list[str]) -> str:
    """Return a POSIX-shell-safe command string."""
    import shlex

    return shlex.join(command)


@app.command("integration-template")
def integration_template(
    framework: str = typer.Argument(..., help="Agent framework: langgraph, crewai, or autogen"),
    session_id: str = typer.Option("default", help="Session ID used by the template"),
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory for the template"),
    install_hint: bool = typer.Option(False, "--install-hint", help="Print the optional framework extra install command before the template"),  # noqa: B008
) -> None:
    """Print a direct Python framework integration starter."""
    from zaxy.integrations import (
        render_agent_integration_template,
        render_framework_install_command,
    )

    try:
        template = render_agent_integration_template(
            framework,
            session_id=session_id,
            eventloom_path=eventloom_path,
        )
        if install_hint:
            typer.echo(_shell_join(render_framework_install_command(framework)))
            typer.echo()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(template, nl=False)


@app.command("integrations")
def integrations(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable framework metadata"),
    recommendation: bool = typer.Option(False, "--recommendation", help="Print the next maintained integration target"),
) -> None:
    """List direct framework integration support and install extras."""
    from zaxy.integrations import (
        list_framework_integration_specs,
        recommend_framework_integration_target,
        render_framework_install_command,
    )

    if recommendation:
        decision = recommend_framework_integration_target()
        payload = asdict(decision)
        if json_output:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        typer.echo(
            f"{decision.target}: {decision.track} "
            f"(evidence={', '.join(decision.evidence_frameworks)}; "
            f"hold={', '.join(decision.hold_frameworks)})"
        )
        typer.echo(decision.rationale)
        return

    rows = []
    for spec in list_framework_integration_specs():
        command = _shell_join(render_framework_install_command(spec.framework))
        rows.append(
            {
                "framework": spec.framework,
                "display_name": spec.display_name,
                "package": spec.package,
                "extra": spec.extra,
                "install": command,
                "template_function": spec.template_function,
                "maturity": spec.maturity,
                "native_adapter": spec.native_adapter,
            }
        )
    if json_output:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        typer.echo(
            f"{row['display_name']}: {row['install']} "
            f"({row['maturity']}, native_adapter={row['native_adapter']})"
        )


@app.command("hooks")
def hooks(
    client: str = typer.Argument(..., help="Hook client: claude-code, codex, or generic"),  # noqa: B008
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory for hook events"),
    domain: str | None = typer.Option(None, help="Project/domain used for default session scoping"),  # noqa: B008
    source: str | None = typer.Option(None, help="Override hook source name"),  # noqa: B008
    output: Path | None = typer.Option(None, "--output", "-o", help="Write hook config to this file"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite an existing output file"),  # noqa: B008
) -> None:
    """Print or write observer hook configuration for supported clients."""
    from zaxy.hooks import (
        render_hook_config,
        write_claude_code_hook_config,
        write_hook_config,
    )

    try:
        config = render_hook_config(
            client,
            eventloom_path=eventloom_path,
            domain=domain,
            source=source,
        )
        if output is not None:
            normalized_client = client.casefold().strip().replace("_", "-")
            if normalized_client in {"claude", "claude-code"}:
                written = write_claude_code_hook_config(output, config, force=force)
            else:
                written = write_hook_config(output, config, force=force)
            typer.echo(f"Wrote hook config to {written}")
            return
        typer.echo(config, nl=False)
    except (FileExistsError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("hook-status")
def hook_status(
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory or JSONL log to inspect"),
    workspace_root: Path = typer.Option(Path("."), help="Workspace root to scan for hook config"),  # noqa: B008
    max_checkout_stale_minutes: int = typer.Option(120, help="Warn when the latest memory checkout is older than this many minutes"),
    min_activation_rate: float | None = typer.Option(
        None,
        "--min-activation-rate",
        help="Fail when fresh checkout rate for high-context sessions is below this 0.0-1.0 floor",
    ),
    max_checkout_prompt_tokens: int | None = typer.Option(
        None,
        "--max-checkout-prompt-tokens",
        min=1,
        help="Fail when the latest checkout prompt token estimate exceeds this ceiling",
    ),
    min_checkout_facts_per_1k_tokens: float | None = typer.Option(
        None,
        "--min-checkout-facts-per-1k-tokens",
        min=0.0,
        help="Fail when latest checkout current facts per 1k prompt tokens is below this floor",
    ),
    now: str | None = typer.Option(None, help="Override current time for deterministic status checks"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Inspect observer hook installation and recent lifecycle activity."""
    from zaxy.hooks import format_hook_status, inspect_hook_status

    if min_activation_rate is not None and not 0.0 <= min_activation_rate <= 1.0:
        raise typer.BadParameter("must be between 0.0 and 1.0", param_hint="--min-activation-rate")
    try:
        parsed_now = _parse_cli_datetime(now) if now else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--now") from exc
    report = inspect_hook_status(
        eventloom_path=eventloom_path,
        workspace_root=workspace_root,
        max_checkout_stale_minutes=max_checkout_stale_minutes,
        now=parsed_now,
    )
    guardrail = _activation_guardrail(report, threshold=min_activation_rate)
    if guardrail is not None:
        report["activation_guardrail"] = guardrail
    token_guardrail = _checkout_token_guardrail(
        report,
        max_prompt_tokens=max_checkout_prompt_tokens,
        min_facts_per_1k_prompt_tokens=min_checkout_facts_per_1k_tokens,
    )
    if token_guardrail is not None:
        report["checkout_token_guardrail"] = token_guardrail
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(format_hook_status(report))
        if guardrail is not None:
            typer.echo(_format_activation_guardrail(guardrail))
        if token_guardrail is not None:
            typer.echo(_format_checkout_token_guardrail(token_guardrail))
    if (guardrail is not None and guardrail["status"] != "ok") or (
        token_guardrail is not None and token_guardrail["status"] != "ok"
    ):
        raise typer.Exit(1)


def _activation_guardrail(report: dict[str, object], *, threshold: float | None) -> dict[str, object] | None:
    if threshold is None:
        return None
    memory_activation = report.get("memory_activation")
    if not isinstance(memory_activation, dict):
        return {
            "status": "fail",
            "threshold": threshold,
            "fresh_checkout_rate": None,
            "message": "activation efficiency is unavailable",
        }
    efficiency = memory_activation.get("activation_efficiency")
    if not isinstance(efficiency, dict):
        return {
            "status": "fail",
            "threshold": threshold,
            "fresh_checkout_rate": None,
            "message": "activation efficiency is unavailable",
        }
    rate = efficiency.get("fresh_checkout_rate")
    if not isinstance(rate, int | float):
        return {
            "status": "fail",
            "threshold": threshold,
            "fresh_checkout_rate": None,
            "message": "activation efficiency is unavailable",
        }
    status = "ok" if float(rate) >= threshold else "fail"
    comparison = ">=" if status == "ok" else "is below required"
    return {
        "status": status,
        "threshold": threshold,
        "fresh_checkout_rate": float(rate),
        "message": f"activation efficiency {float(rate) * 100:.1f}% {comparison} {threshold * 100:.1f}%",
    }


def _format_activation_guardrail(guardrail: dict[str, object]) -> str:
    status = str(guardrail["status"]).upper()
    rate = guardrail.get("fresh_checkout_rate")
    threshold_value = guardrail.get("threshold")
    threshold = float(threshold_value) if isinstance(threshold_value, int | float) else 0.0
    if isinstance(rate, int | float):
        comparator = ">=" if guardrail["status"] == "ok" else "<"
        return f"Activation guardrail: {status} ({float(rate) * 100:.1f}% {comparator} {threshold * 100:.1f}%)"
    return f"Activation guardrail: {status} ({guardrail['message']})"


def _checkout_token_guardrail(
    report: dict[str, object],
    *,
    max_prompt_tokens: int | None,
    min_facts_per_1k_prompt_tokens: float | None,
) -> dict[str, object] | None:
    if max_prompt_tokens is None and min_facts_per_1k_prompt_tokens is None:
        return None
    memory_activation = report.get("memory_activation")
    if not isinstance(memory_activation, dict):
        return _missing_checkout_token_guardrail(
            max_prompt_tokens=max_prompt_tokens,
            min_facts_per_1k_prompt_tokens=min_facts_per_1k_prompt_tokens,
        )
    latest_checkout = memory_activation.get("latest_checkout")
    if not isinstance(latest_checkout, dict):
        return _missing_checkout_token_guardrail(
            max_prompt_tokens=max_prompt_tokens,
            min_facts_per_1k_prompt_tokens=min_facts_per_1k_prompt_tokens,
        )
    token_efficiency = latest_checkout.get("token_efficiency")
    if not isinstance(token_efficiency, dict):
        return _missing_checkout_token_guardrail(
            max_prompt_tokens=max_prompt_tokens,
            min_facts_per_1k_prompt_tokens=min_facts_per_1k_prompt_tokens,
        )
    prompt_tokens = token_efficiency.get("prompt_tokens")
    facts_per_1k = token_efficiency.get("facts_per_1k_prompt_tokens")
    if not isinstance(prompt_tokens, int | float) or not isinstance(facts_per_1k, int | float):
        return _missing_checkout_token_guardrail(
            max_prompt_tokens=max_prompt_tokens,
            min_facts_per_1k_prompt_tokens=min_facts_per_1k_prompt_tokens,
        )
    messages: list[str] = []
    if max_prompt_tokens is not None and int(prompt_tokens) > max_prompt_tokens:
        messages.append(f"checkout prompt tokens {int(prompt_tokens)} exceed maximum {max_prompt_tokens}")
    if min_facts_per_1k_prompt_tokens is not None and float(facts_per_1k) < min_facts_per_1k_prompt_tokens:
        messages.append(
            f"checkout facts per 1k prompt tokens {float(facts_per_1k)} "
            f"below required {min_facts_per_1k_prompt_tokens}"
        )
    return {
        "status": "fail" if messages else "ok",
        "max_prompt_tokens": max_prompt_tokens,
        "min_facts_per_1k_prompt_tokens": min_facts_per_1k_prompt_tokens,
        "prompt_tokens": int(prompt_tokens),
        "facts_per_1k_prompt_tokens": float(facts_per_1k),
        "messages": messages,
    }


def _missing_checkout_token_guardrail(
    *,
    max_prompt_tokens: int | None,
    min_facts_per_1k_prompt_tokens: float | None,
) -> dict[str, object]:
    return {
        "status": "fail",
        "max_prompt_tokens": max_prompt_tokens,
        "min_facts_per_1k_prompt_tokens": min_facts_per_1k_prompt_tokens,
        "prompt_tokens": None,
        "facts_per_1k_prompt_tokens": None,
        "messages": ["checkout token efficiency is unavailable"],
    }


def _format_checkout_token_guardrail(guardrail: dict[str, object]) -> str:
    status = str(guardrail["status"]).upper()
    prompt_tokens = guardrail.get("prompt_tokens")
    facts_per_1k = guardrail.get("facts_per_1k_prompt_tokens")
    if isinstance(prompt_tokens, int | float) and isinstance(facts_per_1k, int | float):
        return (
            f"Checkout token guardrail: {status} "
            f"({int(prompt_tokens)} prompt tokens, {float(facts_per_1k)} facts/1k prompt tokens)"
        )
    messages = guardrail.get("messages")
    if isinstance(messages, list) and messages:
        return f"Checkout token guardrail: {status} ({'; '.join(str(message) for message in messages)})"
    return f"Checkout token guardrail: {status}"


def _checkout_activity_metadata(payload: dict[str, object]) -> dict[str, object]:
    token_efficiency = payload.get("token_efficiency")
    if isinstance(token_efficiency, dict):
        return {"token_efficiency": token_efficiency}
    return {}


@app.command("capture-soak")
def capture_soak(
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory or JSONL log to inspect"),
    workspace_root: Path = typer.Option(Path("."), help="Workspace root to scan for capture config"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Session ID to inspect when eventloom path is a directory"),  # noqa: B008
    max_stale_minutes: int = typer.Option(30, help="Maximum allowed age for active capture lanes"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Report whether deterministic capture satisfies beta soak criteria."""
    from zaxy.capture_soak import build_capture_soak_report, format_capture_soak_report

    report = build_capture_soak_report(
        eventloom_path=eventloom_path,
        workspace_root=workspace_root,
        session_id=session_id,
        max_stale_minutes=max_stale_minutes,
    )
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(format_capture_soak_report(report))
    if report["beta_criteria"]["status"] != "pass":
        raise typer.Exit(1)


@app.command("hook-event")
def hook_event(
    trigger: str = typer.Argument(..., help="Hook trigger: session-start, stop, precompact, checkpoint, heartbeat, command, file-edit, tool-call, or transcript-turn"),  # noqa: B008
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory for hook events"),
    session_id: str = typer.Option("default", help="Session ID to append hook events into"),
    source: str = typer.Option("generic", help="Client or adapter that emitted the hook"),
    workspace: str | None = typer.Option(None, help="Workspace path associated with the hook"),  # noqa: B008
    transcript_path: str | None = typer.Option(None, help="Transcript path associated with the hook"),  # noqa: B008
    summary: str | None = typer.Option(None, help="Checkpoint summary for retrieval"),  # noqa: B008
    reason: str | None = typer.Option(None, help="Checkpoint reason, e.g. manual, interval, precompact, shutdown"),  # noqa: B008
    turn_count: int | None = typer.Option(None, help="Turn count associated with the checkpoint"),  # noqa: B008
    command: str | None = typer.Option(None, help="Observed shell command for command hooks"),  # noqa: B008
    exit_code: int | None = typer.Option(None, help="Observed shell command exit code"),  # noqa: B008
    duration_ms: int | None = typer.Option(None, help="Observed command duration in milliseconds"),  # noqa: B008
    stdout: str = typer.Option("", help="Observed command stdout excerpt"),
    stderr: str = typer.Option("", help="Observed command stderr excerpt"),
    path: str | None = typer.Option(None, "--path", help="Observed edited file path"),  # noqa: B008
    operation: str = typer.Option("modified", help="Observed file operation"),
    line_count: int | None = typer.Option(None, help="Observed changed line count"),  # noqa: B008
    tool_name: str | None = typer.Option(None, help="Observed tool name for tool-call hooks"),  # noqa: B008
    tool_status: str = typer.Option("ok", help="Observed tool-call status"),
    call_id: str | None = typer.Option(None, help="Observed tool-call identifier"),  # noqa: B008
    arguments_json: str | None = typer.Option(None, help="Observed tool-call arguments as a JSON object"),  # noqa: B008
    result_summary: str | None = typer.Option(None, help="Observed tool-call result summary"),  # noqa: B008
    role: str | None = typer.Option(None, help="Transcript turn role for transcript-turn hooks"),  # noqa: B008
    content: str | None = typer.Option(None, help="Transcript turn content for transcript-turn hooks"),  # noqa: B008
    turn_index: int | None = typer.Option(None, help="Transcript turn index"),  # noqa: B008
) -> None:
    """Append a lightweight observer hook event without requiring Neo4j."""
    from zaxy.hooks import build_hook_payload, hook_event_type
    from zaxy.observation import (
        build_command_observation,
        build_file_edit_observation,
        build_tool_call_observation,
        build_transcript_turn_observation,
    )
    from zaxy.session import SessionManager

    eventlog = SessionManager(base_path=eventloom_path).get(session_id).eventlog
    normalized_trigger = trigger.casefold().strip().replace("_", "-")
    if normalized_trigger == "command":
        if command is None or exit_code is None:
            raise typer.BadParameter("command hooks require --command and --exit-code")
        event_input = build_command_observation(
            command=command,
            exit_code=exit_code,
            session_id=session_id,
            source=source,
            workspace=workspace,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
        )
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=event_input["payload"],
            thread=session_id,
        )
        if duration_ms is not None and duration_ms >= 30_000:
            from zaxy.memory_persistence import append_memory_reminder_if_needed

            append_memory_reminder_if_needed(
                eventloom_path,
                session_id=session_id,
                trigger="long-tool-run",
                source=source,
                reason="long_tool_run",
                current_task=summary or command,
            )
        typer.echo(f"Recorded observation {event_input['event_type']} seq={event.seq}")
        return
    if normalized_trigger == "file-edit":
        if path is None:
            raise typer.BadParameter("file-edit hooks require --path")
        event_input = build_file_edit_observation(
            path=path,
            operation=operation,
            session_id=session_id,
            source=source,
            workspace=workspace,
            summary=summary,
            line_count=line_count,
        )
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=event_input["payload"],
            thread=session_id,
        )
        typer.echo(f"Recorded observation {event_input['event_type']} seq={event.seq}")
        return
    if normalized_trigger == "tool-call":
        if tool_name is None:
            raise typer.BadParameter("tool-call hooks require --tool-name")
        arguments = _parse_json_object(arguments_json, option="--arguments-json") if arguments_json else None
        event_input = build_tool_call_observation(
            tool_name=tool_name,
            status=tool_status,
            session_id=session_id,
            source=source,
            workspace=workspace,
            call_id=call_id,
            arguments=arguments,
            result_summary=result_summary,
        )
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=event_input["payload"],
            thread=session_id,
        )
        typer.echo(f"Recorded observation {event_input['event_type']} seq={event.seq}")
        return
    if normalized_trigger == "transcript-turn":
        if role is None or content is None:
            raise typer.BadParameter("transcript-turn hooks require --role and --content")
        event_input = build_transcript_turn_observation(
            role=role,
            content=content,
            session_id=session_id,
            source=source,
            turn_index=turn_index,
        )
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=event_input["payload"],
            thread=session_id,
        )
        typer.echo(f"Recorded observation {event_input['event_type']} seq={event.seq}")
        return
    try:
        event_type = hook_event_type(trigger)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = build_hook_payload(
        trigger=trigger,
        source=source,
        workspace=workspace,
        transcript_path=transcript_path,
        summary=summary,
        reason=reason,
        turn_count=turn_count,
    )
    event = eventlog.append(event_type, actor="zaxy-hook", payload=payload, thread=session_id)
    from zaxy.memory_persistence import append_memory_reminder_if_needed

    reminder = append_memory_reminder_if_needed(
        eventloom_path,
        session_id=session_id,
        trigger=payload["trigger"],
        source=source,
        reason=reason,
        turn_count=turn_count,
        current_task=summary,
    )
    typer.echo(f"Recorded hook {payload['trigger']} as {event_type} seq={event.seq}")
    if reminder is not None:
        typer.echo(f"Suggested memory reminder seq={reminder.seq}")


@app.command("codex-capture")
def codex_capture(
    workspace: Path = typer.Option(Path("."), help="Workspace root whose Codex sessions should be captured"),  # noqa: B008
    codex_home: Path | None = typer.Option(None, help="Codex home directory; defaults to CODEX_HOME or ~/.codex"),  # noqa: B008
    eventloom_path: Path = typer.Option(Path(".eventloom"), help="Eventloom directory for captured observations"),  # noqa: B008
    session_id: str = typer.Option("default", help="Zaxy Eventloom session ID to append into"),
    source: str = typer.Option("codex-local", help="Capture source label"),
    max_records_per_file: int = typer.Option(
        1000,
        "--max-records-per-file",
        min=1,
        help="Maximum recent records to scan from each Codex session log per pass",
    ),
    graph: bool = typer.Option(False, "--graph", help="Project captured observations into Neo4j"),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    watch: bool = typer.Option(False, "--watch", help="Continuously poll Codex session logs"),
    interval_seconds: float = typer.Option(2.0, "--interval-seconds", min=0.25, help="Watch poll interval"),
    watch_iterations: int | None = typer.Option(
        None,
        "--watch-iterations",
        min=1,
        help="Optional bounded watch pass count for supervisors and tests",
    ),
) -> None:
    """Capture local Codex session JSONL records into Eventloom without proxying model traffic."""
    import asyncio

    from zaxy.config import get_settings

    async def project_events(events: tuple[Any, ...]) -> int:
        if not events:
            return 0
        from zaxy.extract import extract

        settings = get_settings()
        store = GraphStore(
            neo4j_uri or settings.neo4j_uri,
            neo4j_user or settings.neo4j_user,
            neo4j_password or settings.neo4j_password,
        )
        await store.connect()
        try:
            await store.init_schema()
            for event in events:
                await store.upsert_extraction(extract(event), session_id=session_id)
            return len(events)
        finally:
            await store.close()

    def run_once() -> None:
        result = capture_codex_sessions(
            workspace=workspace,
            codex_home=codex_home,
            eventloom_path=eventloom_path,
            session_id=session_id,
            source=source,
            max_records_per_file=max_records_per_file,
        )
        plural = "" if result.scanned_files == 1 else "s"
        typer.echo(
            f"Imported {result.imported} Codex observations from "
            f"{result.scanned_files} session log{plural} ({result.skipped} skipped)"
        )
        if graph:
            projected = asyncio.run(project_events(result.events))
            typer.echo(f"Projected {projected} captured observations into graph")

    if not watch:
        run_once()
        return
    typer.echo("Watching Codex session logs for deterministic Zaxy capture. Press Ctrl-C to stop.")
    try:
        iterations = 0
        while watch_iterations is None or iterations < watch_iterations:
            run_once()
            iterations += 1
            if watch_iterations is None or iterations < watch_iterations:
                time.sleep(interval_seconds)
    except KeyboardInterrupt:
        typer.echo("Stopped Codex capture.")


@capture_app.command("status")
def capture_status(
    workspace: Path = typer.Option(Path("."), help="Workspace root with capture config"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Inspect managed deterministic capture runtime state."""
    from zaxy.capture_manager import inspect_codex_capture

    report = inspect_codex_capture(workspace=workspace)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    configured = "configured" if report["configured"] else "not configured"
    running = "running" if report["running"] else "not running"
    typer.echo(f"Codex capture: {configured}, {running}")
    if report["pids"]:
        typer.echo("pids: " + ", ".join(str(pid) for pid in report["pids"]))
    typer.echo(f"state: {report['state_file']}")
    latest = report.get("latest_observation")
    if latest:
        typer.echo(
            f"latest: {latest['type']} seq={latest['seq']} "
            f"session={latest['thread']} source={latest['source']}"
        )


@capture_app.command("start")
def capture_start(
    workspace: Path = typer.Option(Path("."), help="Workspace root with .codex/zaxy-capture.json"),  # noqa: B008
    graph: bool = typer.Option(False, "--graph", help="Project captured observations into Neo4j"),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    max_records_per_file: int = typer.Option(
        500,
        "--max-records-per-file",
        min=1,
        help="Maximum recent records to scan from each Codex session log per pass",
    ),
) -> None:
    """Start a managed deterministic Codex capture watcher."""
    from zaxy.capture_manager import start_codex_capture

    try:
        result = start_codex_capture(
            workspace=workspace,
            graph=graph,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            max_records_per_file=max_records_per_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(result["message"])
    typer.echo(f"state: {result['state_file']}")


@capture_app.command("stop")
def capture_stop(
    workspace: Path = typer.Option(Path("."), help="Workspace root with managed capture state"),  # noqa: B008
) -> None:
    """Stop the managed deterministic Codex capture watcher."""
    from zaxy.capture_manager import stop_codex_capture

    result = stop_codex_capture(workspace=workspace)
    typer.echo(result["message"])


def _parse_json_object(value: str, *, option: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{option} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{option} must be a JSON object")
    return parsed


def _resolve_cli_projection_backend(
    projection_backend: str | None,
    settings: Settings,
    *,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    pggraph_dsn: str | None = None,
    embedded_graph_path: Path | None = None,
) -> str:
    """Resolve CLI backend intent while keeping bare commands embedded-first."""
    if projection_backend:
        return projection_backend.casefold().strip()
    if pggraph_dsn:
        return "pggraph"
    if embedded_graph_path:
        return "embedded"
    if neo4j_uri or neo4j_user or neo4j_password:
        return "neo4j"
    return settings.projection_backend.casefold().strip()


@app.command("local-profile")
def local_profile(
    output: Path | None = typer.Option(None, "--output", "-o", help="Write profile to this file"),  # noqa: B008
    projection_backend: str = typer.Option("embedded", "--projection-backend", help="Projection backend: embedded, neo4j, pggraph, or latticedb"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite an existing output file"),  # noqa: B008
    check: bool = typer.Option(False, "--check", help="Validate deterministic local providers"),  # noqa: B008
) -> None:
    """Print, write, or check an offline local retrieval profile."""
    from zaxy.local_profile import check_local_profile, render_local_profile, write_local_profile

    if check:
        typer.echo(json.dumps(check_local_profile(), indent=2, sort_keys=True))
        return
    if output is None:
        typer.echo(render_local_profile(projection_backend=projection_backend), nl=False)
        return
    try:
        written = write_local_profile(output, projection_backend=projection_backend, force=force)
    except (FileExistsError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote local profile to {written}")


@app.command("doctor")
def doctor(
    eventloom_path: str | None = typer.Option(None, help="Override Eventloom path for this check"),
    project_root: Path | None = typer.Option(  # noqa: B008
        None,
        "--project-root",
        help="Repository root for release and beta readiness checks",
    ),
    release_smoke: bool = typer.Option(
        False,
        "--release-smoke",
        help="Run local release metadata checks instead of onboarding checks",
    ),
    beta_readiness: bool = typer.Option(
        False,
        "--beta-readiness",
        help="Run beta release readiness checks instead of onboarding checks",
    ),
    external_validation_report: Path | None = typer.Option(  # noqa: B008
        None,
        "--external-validation-report",
        help="External-validation report JSON for beta readiness checks",
    ),
    require_external_validation: bool = typer.Option(
        False,
        "--require-external-validation",
        help="Fail beta readiness when external validation evidence is missing",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Run local setup and onboarding checks."""
    from zaxy.config import get_settings
    from zaxy.doctor import format_doctor_report, run_doctor

    if release_smoke and beta_readiness:
        raise typer.BadParameter("--release-smoke and --beta-readiness are mutually exclusive")
    if not beta_readiness and (external_validation_report is not None or require_external_validation):
        raise typer.BadParameter("external validation options require --beta-readiness")

    if release_smoke:
        from zaxy.release import run_release_smoke

        report = run_release_smoke(project_root=project_root)
        if json_output:
            typer.echo(json.dumps(report, indent=2, sort_keys=True))
        else:
            typer.echo(format_doctor_report(report))
        if report["status"] == "error":
            raise typer.Exit(1)
        return

    if beta_readiness:
        from zaxy.release import run_beta_readiness

        report = run_beta_readiness(
            project_root=project_root,
            external_validation_report=external_validation_report,
            require_external_validation=require_external_validation,
        )
        if json_output:
            typer.echo(json.dumps(report, indent=2, sort_keys=True))
        else:
            typer.echo(format_doctor_report(report))
        if report["status"] == "error":
            raise typer.Exit(1)
        return

    settings = get_settings()
    if eventloom_path is not None:
        settings = settings.model_copy(update={"eventloom_path": eventloom_path})
    report = run_doctor(settings=settings)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(format_doctor_report(report))


@app.command("packet-status")
def packet_status(
    eventloom_path: Path = typer.Option(  # noqa: B008
        Path(".eventloom"),
        "--eventloom-path",
        help="Eventloom directory containing packet memory events",
    ),
    session_id: str = typer.Option(
        "default",
        "--session-id",
        help="Session ID to inspect",
    ),
    analyzer_host: str = typer.Option("127.0.0.1", "--analyzer-host", help="Packet analyzer host to probe"),
    analyzer_port: int = typer.Option(8787, "--analyzer-port", min=1, max=65535, help="Packet analyzer port to probe"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Inspect the LLM packet-memory pipeline for one session."""
    from zaxy.doctor import format_packet_memory_report, packet_memory_report

    report = packet_memory_report(
        eventloom_path=eventloom_path,
        session_id=session_id,
        analyzer_host=analyzer_host,
        analyzer_port=analyzer_port,
    )
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(format_packet_memory_report(report))


@app.command("index-codebase")
def index_codebase(
    path: Path = typer.Argument(..., help="Repository or directory to index"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to append codebase events into"),  # noqa: B008
    max_bytes: int = typer.Option(512 * 1024, help="Maximum source file size to index"),  # noqa: B008
) -> None:
    """Append codebase file, symbol, and import mapping events."""
    import asyncio

    async def _run() -> int:
        fabric = MemoryFabric()
        try:
            return int(await fabric.ingest_codebase(path, session_id=session_id, max_bytes=max_bytes))
        finally:
            await fabric.close()

    count = asyncio.run(_run())
    typer.echo(f"Indexed {count} codebase events into session {session_id}")


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
        fabric = MemoryFabric(
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


@app.command("init-session")
def init_session(
    path: Path = typer.Argument(..., help="Workspace root to initialize"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to append genesis into"),  # noqa: B008
) -> None:
    """Append a workspace session genesis event."""
    import asyncio

    from zaxy.workspace import WorkspaceProfile

    async def _run() -> WorkspaceProfile:
        fabric = MemoryFabric()
        try:
            return cast(WorkspaceProfile, await fabric.initialize_session(path, session_id=session_id))
        finally:
            await fabric.close()

    profile = asyncio.run(_run())
    typer.echo(
        f"Initialized {session_id} as {profile.workspace_type} workspace "
        f"(confidence {profile.confidence})"
    )


@app.command("init")
def init(
    path: Path = typer.Argument(Path("."), help="Workspace root to initialize"),  # noqa: B008
    preset: str | None = typer.Option(None, help="Onboarding preset: local-claude, local-codex, or local-embedded-codex"),  # noqa: B008
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory for this workspace"),
    domain: str | None = typer.Option(None, help="Project/domain used for default session scoping"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Explicit session ID; defaults to <domain>-default"),  # noqa: B008
    mcp_client: str | None = typer.Option(None, help="MCP client config to render/write"),  # noqa: B008
    mcp_output: Path | None = typer.Option(None, help="Write MCP config JSON to this file"),  # noqa: B008
    hook_client: str | None = typer.Option(None, help="Hook client config to render/write"),  # noqa: B008
    hook_output: Path | None = typer.Option(None, help="Write hook config to this file"),  # noqa: B008
    local_profile_output: Path | None = typer.Option(None, help="Write local retrieval profile to this file"),  # noqa: B008
    infra: str = typer.Option("none", help="Local infra action: none, check, or start"),  # noqa: B008
    projection_backend: str | None = typer.Option(None, "--projection-backend", help="Projection backend for infra bootstrap: embedded, neo4j, pggraph, or latticedb"),  # noqa: B008
    pggraph_dsn: str | None = typer.Option(None, "--pggraph-dsn", help="pgGraph/PostgreSQL DSN for --projection-backend pggraph"),  # noqa: B008
    pggraph_repo: Path | None = typer.Option(None, "--pggraph-repo", help="Local pgGraph checkout containing scripts/quickstart.sh"),  # noqa: B008
    capture_mode: str = typer.Option("deterministic", help="Capture mode: deterministic, packet, or hybrid"),  # noqa: B008
    capture_action: str = typer.Option("none", "--capture", help="Capture action after init: none or start"),  # noqa: B008
    packet_capture: bool = typer.Option(False, "--packet-capture", help="Include packet analyzer/projector activation steps"),  # noqa: B008
    packet_upstream_base_url: str = typer.Option("https://api.openai.com/v1", help="Packet analyzer upstream OpenAI-compatible base URL"),  # noqa: B008
    packet_port: int = typer.Option(8787, "--packet-port", min=1, max=65535, help="Local packet analyzer port"),  # noqa: B008
    zaxy_executable: str | None = typer.Option(None, help="Executable path MCP clients should invoke"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite generated output files"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),  # noqa: B008
) -> None:
    """Bare zaxy init uses the local embedded Codex path for MCP config, infra, and hook status."""
    import asyncio

    async def _run() -> Any:
        effective_preset = preset
        if (
            effective_preset is None
            and mcp_client is None
            and mcp_output is None
            and hook_client is None
            and hook_output is None
            and local_profile_output is None
            and projection_backend is None
        ):
            effective_preset = "local-embedded-codex"
        preset_options = apply_onboarding_preset(
            effective_preset,
            workspace=path,
            mcp_client=mcp_client,
            mcp_output=mcp_output,
            hook_client=hook_client,
            hook_output=hook_output,
            local_profile_output=local_profile_output,
            infra=infra,
            capture_mode="hybrid" if packet_capture else capture_mode,
        )
        return await run_onboarding(
            path,
            eventloom_path=eventloom_path,
            domain=domain,
            session_id=session_id,
            mcp_client=preset_options["mcp_client"],
            mcp_output=preset_options["mcp_output"],
            hook_client=preset_options["hook_client"],
            hook_output=preset_options["hook_output"],
            local_profile_output=preset_options["local_profile_output"],
            infra=preset_options["infra"],
            projection_backend=projection_backend or preset_options["projection_backend"] or "embedded",
            pggraph_dsn=pggraph_dsn,
            pggraph_repo=pggraph_repo,
            capture_mode=preset_options["capture_mode"],
            packet_capture=packet_capture,
            packet_upstream_base_url=packet_upstream_base_url,
            packet_port=packet_port,
            capture_action=capture_action,
            zaxy_executable=zaxy_executable,
            force=force,
        )

    try:
        result = asyncio.run(_run())
    except (FileExistsError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        typer.echo(format_onboarding_result(result))


@app.command("viewer")
def viewer(
    path: Path = typer.Argument(..., help="Eventloom JSONL log or directory to inspect"),  # noqa: B008
    output: Path = typer.Option("eventloom-viewer.html", "--output", "-o", help="HTML output path"),  # noqa: B008
) -> None:
    """Write a standalone HTML viewer for Eventloom sessions."""
    from zaxy.viewer import write_viewer_html

    written = write_viewer_html(path, output)
    typer.echo(f"Wrote Eventloom viewer: {written}")


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


@app.command("schema-plan")
def schema_plan() -> None:
    """Print the current Neo4j schema migration plan."""
    from zaxy.schema import render_schema_plan

    typer.echo(render_schema_plan())


@app.command("schema-recovery-plan")
def schema_recovery_plan() -> None:
    """Inspect Neo4j migration records and print recovery guidance."""
    import asyncio

    from zaxy.config import get_settings
    from zaxy.graph import GraphStore as SchemaGraphStore
    from zaxy.schema import (
        fetch_schema_migration_records,
        render_schema_recovery_plan,
        schema_migration_status,
    )

    settings = get_settings()

    async def _run() -> str:
        graph = SchemaGraphStore(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            ca_cert=settings.neo4j_ca_cert,
            trust_all=settings.neo4j_trust_all,
        )
        try:
            await graph.connect()
            records = await fetch_schema_migration_records(graph._driver)
        finally:
            await graph.close()
        return render_schema_recovery_plan(
            schema_migration_status(records=records)
        )

    typer.echo(asyncio.run(_run()))


@app.command("extractor-template")
def extractor_template(
    event_type: str = typer.Argument(..., help="Typed Eventloom event, e.g. decision.recorded"),  # noqa: B008
    entity_type: str = typer.Option(..., "--entity-type", help="Graph entity type to extract"),
    name_key: str = typer.Option(..., "--name-key", help="Payload key used as the entity name"),
    summary_key: str | None = typer.Option(None, "--summary-key", help="Payload key used as summary"),
    actor_relation: str | None = typer.Option(
        None,
        "--actor-relation",
        help="Optional relation from event actor to extracted entity",
    ),
) -> None:
    """Print a validated rule-extractor starter for a new event type."""
    from zaxy.extract_templates import ExtractorTemplateSpec, render_extractor_template

    try:
        spec = ExtractorTemplateSpec(
            event_type=event_type,
            entity_type=entity_type,
            entity_name_payload_key=name_key,
            summary_payload_key=summary_key,
            actor_relation_type=actor_relation,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(render_extractor_template(spec))


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
    from zaxy.domain import derive_domain, domain_default_session
    from zaxy.mcp_runtime import EmbeddedMcpRuntimeCoordinator

    workspace_root = Path.cwd()
    resolved_eventloom_path = eventloom_path or os.getenv("EVENTLOOM_PATH") or str(workspace_root / ".eventloom")
    resolved_session_id = os.getenv("EVENTLOOM_THREAD") or domain_default_session(derive_domain(workspace_root))
    settings = _status_settings(workspace_root)
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
        embedded_stdio_coordinator = EmbeddedMcpRuntimeCoordinator.from_eventloom_path(resolved_eventloom_path)
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
    )

    if transport == "sse":
        asyncio.run(mcp_server.main_sse(port=port, host=host))
    else:
        asyncio.run(mcp_server.main(owner_claim=owner_claim))


@app.command()
def replay(
    log_path: Path = typer.Argument(..., help="Path to Eventloom JSONL file"),  # noqa: B008
    from_seq: int = typer.Option(1, help="Start sequence number"),
    to_seq: int | None = typer.Option(None, help="Inclusive end sequence number"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Replay an Eventloom log and print integrity report + events."""
    from zaxy.event import EventLog

    log = EventLog(str(log_path))
    try:
        result = log.replay(from_seq=from_seq, to_seq=to_seq)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        output = {
            "from_seq": from_seq,
            "to_seq": to_seq,
            "integrity": result.integrity.model_dump(),
            "events": [e.model_dump() for e in result.events],
        }
        print(json.dumps(output, indent=2))
        return

    typer.echo(f"Integrity: {'OK' if result.integrity.ok else 'FAILED'}")
    typer.echo(f"Total events: {result.integrity.total_events}")
    window = f"{from_seq}.." + (str(to_seq) if to_seq is not None else "HEAD")
    typer.echo(f"Replay window: {window}")
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
            if not replay_result.integrity.ok:
                reason = replay_result.integrity.broken_reason or "unknown integrity failure"
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


@app.command()
def status(
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    projection_backend: str | None = typer.Option(None, "--projection-backend", help="Projection backend to check: embedded, neo4j, pggraph, or latticedb"),
    pggraph_dsn: str | None = typer.Option(None, "--pggraph-dsn", help="pgGraph/PostgreSQL DSN"),
    embedded_graph_path: Path | None = typer.Option(None, "--embedded-graph-path", help="Embedded graph projection path"),  # noqa: B008
    eventloom_path: Path | None = typer.Option(None, "--eventloom-path", help="Eventloom directory for memory activation status"),  # noqa: B008
    max_checkout_stale_minutes: int = typer.Option(120, "--max-checkout-stale-minutes", help="Warn when latest memory checkout is older than this many minutes"),
    now: str | None = typer.Option(None, "--now", help="Override current time for deterministic status checks"),
    pathlight_url: str | None = typer.Option(None, help="Pathlight collector URL"),
) -> None:
    """Check connectivity to external services and local projection posture."""
    import asyncio

    from zaxy.hooks import inspect_memory_activation

    settings = _status_settings()
    try:
        parsed_now = _parse_cli_datetime(now) if now else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--now") from exc

    async def _check() -> None:
        ok = True
        _uri = neo4j_uri or settings.neo4j_uri
        _user = neo4j_user or settings.neo4j_user
        _password = neo4j_password or settings.neo4j_password
        _pathlight = pathlight_url or settings.pathlight_url
        _eventloom_path = eventloom_path or Path(settings.eventloom_path)
        backend = _resolve_cli_projection_backend(
            projection_backend,
            settings,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            pggraph_dsn=pggraph_dsn,
            embedded_graph_path=embedded_graph_path,
        )

        if backend == "embedded":
            embedded_runtime = LocalEmbeddedGraphRuntime(path=embedded_graph_path or Path(settings.embedded_graph_path))
            check = embedded_runtime.check()
            typer.echo(f"embedded graph: {check.status.upper()} ({check.message})")
            if check.status == "error":
                ok = False
        elif backend == "pggraph":
            pggraph_runtime = LocalPgGraphRuntime(
                dsn=pggraph_dsn or settings.pggraph_dsn,
                enabled=settings.pggraph_auto_start and settings.zaxy_env.lower() != "production",
            )
            check = pggraph_runtime.check()
            typer.echo(f"pgGraph: {check.status.upper()} ({check.message})")
            if check.status == "error":
                ok = False
        elif backend == "neo4j":
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
        else:
            typer.echo(f"Projection: FAIL (status supports neo4j, pggraph, or embedded; got {backend})")
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

        activation = inspect_memory_activation(
            eventloom_path=_eventloom_path,
            max_checkout_stale_minutes=max_checkout_stale_minutes,
            now=parsed_now,
        )
        typer.echo(_format_status_memory_activation(activation))

        raise typer.Exit(0 if ok else 1)

    asyncio.run(_check())


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
    from zaxy.live_benchmark import (
        BM25Retriever,
        CentroidConsolidationRetriever,
        MarkdownRetriever,
        MarkdownVectorRetriever,
        VectorRetriever,
    )

    retrievers: dict[str, Any] = {}
    for backend in selected:
        if backend == "md":
            retrievers[backend] = MarkdownRetriever(corpus)
        elif backend == "bm25":
            retrievers[backend] = BM25Retriever(corpus)
        elif backend == "vector":
            retrievers[backend] = VectorRetriever(corpus, provider)
        elif backend == "md+vector":
            retrievers[backend] = MarkdownVectorRetriever(corpus, provider)
        elif backend == "centroid":
            retrievers[backend] = CentroidConsolidationRetriever(corpus, provider)
    return retrievers


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
        help=(
            "Workload: fixture, statistical, frozen, suite, consolidation, "
            "context-collapse, graph-traversal, source-recall, temporal-recall, or longmemeval"
        ),
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

    from zaxy.benchmark import build_competitive_event_log, competitive_cases
    from zaxy.config import get_settings
    from zaxy.embedding import (
        HashEmbeddingProvider,
        LocalHTTPEmbeddingProvider,
        OpenAIEmbeddingProvider,
        SentenceTransformersEmbeddingProvider,
    )
    from zaxy.live_benchmark import (
        BenchmarkWorkload,
        CachedEmbeddingProvider,
        ZaxyCheckoutRetriever,
        _build_source_lane_retriever,
        benchmark_live_retrievers,
        benchmark_projection_cache_key,
        benchmark_query_scope_resolver,
        build_benchmark_suite_workload,
        build_consolidation_collapse_workload,
        build_context_collapse_workload,
        build_frozen_statistical_workload,
        build_graph_traversal_workload,
        build_live_zaxy_retriever,
        build_longmemeval_workload,
        build_source_recall_workload,
        build_statistical_event_log,
        build_temporal_recall_workload,
        corpus_from_event_log,
        report_to_markdown,
        write_benchmark_report,
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
    provider = CachedEmbeddingProvider(provider, cache_path=embedding_cache)

    async def _run() -> None:
        with tempfile.TemporaryDirectory(prefix="zaxy-live-benchmark-") as tmp:
            benchmark_workload: BenchmarkWorkload
            if workload == "fixture":
                eventlog = build_competitive_event_log(Path(tmp) / "bench.jsonl")
                cases = competitive_cases()
                benchmark_workload = BenchmarkWorkload.from_event_log(
                    eventlog,
                    cases,
                    version="fixture-v1",
                )
            elif workload == "statistical":
                eventlog, cases = build_statistical_event_log(
                    Path(tmp) / "bench.jsonl",
                    subjects=subjects,
                )
                benchmark_workload = BenchmarkWorkload.from_event_log(
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
                raise typer.BadParameter(
                    "workload must be 'fixture', 'statistical', 'frozen', "
                    "'suite', 'consolidation', 'context-collapse', "
                    "'graph-traversal', 'source-recall', 'temporal-recall', "
                    "or 'longmemeval'"
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
                checkout_retriever: ZaxyCheckoutRetriever | None = None
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


def _load_external_results(path: Path | None) -> tuple[Any, ...]:
    """Load operator-supplied external benchmark rows from JSON."""
    from zaxy.live_benchmark import ExternalBenchmarkResult

    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise typer.BadParameter("external results JSON must be a list")
    results: list[ExternalBenchmarkResult] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise typer.BadParameter(f"external result {idx} must be an object")
        try:
            results.append(ExternalBenchmarkResult(**item))
        except TypeError as exc:
            raise typer.BadParameter(f"invalid external result {idx}: {exc}") from exc
    return tuple(results)


@app.command("benchmark-inventory")
def benchmark_inventory(
    output_dir: Path = typer.Option(  # noqa: B008
        Path(".zaxy-benchmark-inventory"),
        help="Directory where reproducible inventory workload logs are written",
    ),
    subjects: int = typer.Option(100, min=1, help="Subject count for temporal and graph lanes"),
    documents: int = typer.Option(100, min=1, help="Document count for source-recall lane"),
    sessions: int = typer.Option(50, min=1, help="Session count for context-collapse lane"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List MemPalace-comparable benchmark lanes, fingerprints, and required metrics."""
    from zaxy.live_benchmark import (
        build_mempalace_workload_inventory,
        format_mempalace_workload_inventory,
    )

    inventory = build_mempalace_workload_inventory(
        output_dir,
        subjects=subjects,
        documents=documents,
        sessions=sessions,
    )
    if json_output:
        typer.echo(json.dumps([asdict(entry) for entry in inventory], indent=2, sort_keys=True))
    else:
        typer.echo(format_mempalace_workload_inventory(inventory), nl=False)


@app.command("purpose-benchmark")
def purpose_benchmark(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/purpose-v1"),
        help="Directory for JSON and Markdown purpose benchmark reports",
    ),
    holdout_pack: list[Path] | None = typer.Option(None, "--holdout-pack", help="Purpose holdout pack JSON to include"),  # noqa: B008
    include_holdouts: bool = typer.Option(False, "--include-holdouts", help="Include the packaged public-derived purpose holdout pack"),
    require_holdout_fingerprint: str | None = typer.Option(None, "--require-holdout-fingerprint", help="Require an included holdout pack fingerprint"),
    json_output: bool = typer.Option(False, "--json", help="Print report JSON instead of text summary"),
) -> None:
    """Run deterministic purpose-conditioned memory benchmark gates."""
    from zaxy.purpose_benchmark import run_purpose_benchmark, write_purpose_benchmark_report

    packs = list(holdout_pack or [])
    if include_holdouts:
        packs.append(Path("reports/benchmarks/purpose-v1/holdouts/public-derived-purpose-v1/holdout-pack.json"))
    report = run_purpose_benchmark(holdout_packs=tuple(packs))
    if require_holdout_fingerprint:
        fingerprints = {
            str(holdout.get("pack_fingerprint") or "")
            for holdout in report.holdout_reports.values()
        }
        if require_holdout_fingerprint not in fingerprints:
            typer.echo("Error: --require-holdout-fingerprint did not match an included holdout pack", err=True)
            raise typer.Exit(2)
    written = write_purpose_benchmark_report(report, output_dir)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(f"Purpose benchmark: {report.status}")
        typer.echo(f"Lanes: {report.passed_lanes}/{report.lane_count} passed")
        typer.echo(f"JSON: {written['json']}")
        typer.echo(f"Markdown: {written['markdown']}")
    raise typer.Exit(0 if report.status == "passed" else 1)


@app.command("benchmark-compare")
def benchmark_compare(
    candidate: Path = typer.Argument(..., help="Candidate live-benchmark.json report"),  # noqa: B008
    baseline: Path | None = typer.Option(  # noqa: B008
        None,
        "--baseline",
        help="Optional baseline live-benchmark.json report for regression checks",
    ),
    backend: str = typer.Option("zaxy", help="Backend to guard, usually zaxy"),
    min_mean_score: float = typer.Option(0.95, help="Minimum acceptable mean score"),
    min_answer_recall_at_5: float | None = typer.Option(
        0.95,
        help="Minimum acceptable Answer@5 when reported",
    ),
    min_recall_at_5: float | None = typer.Option(
        0.99,
        help="Minimum acceptable Recall@5 when reported",
    ),
    min_citation_coverage: float = typer.Option(
        0.95,
        help="Minimum acceptable citation coverage when reported",
    ),
    max_p95_ms: float = typer.Option(500.0, help="Maximum acceptable p95 latency in ms"),
    max_p99_ms: float = typer.Option(750.0, help="Maximum acceptable p99 latency in ms"),
    max_latency_regression_ratio: float = typer.Option(
        0.25,
        help="Allowed latency regression ratio versus the baseline report",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Compare benchmark reports against beta quality and latency guardrails."""
    from zaxy.live_benchmark import (
        compare_benchmark_reports,
        format_benchmark_comparison,
        load_benchmark_report,
    )

    baseline_report = load_benchmark_report(baseline) if baseline is not None else None
    candidate_report = load_benchmark_report(candidate)
    comparison = compare_benchmark_reports(
        baseline_report,
        candidate_report,
        backend=backend,
        min_mean_score=min_mean_score,
        min_answer_recall_at_5=min_answer_recall_at_5,
        min_recall_at_5=min_recall_at_5,
        min_citation_coverage=min_citation_coverage,
        max_p95_ms=max_p95_ms,
        max_p99_ms=max_p99_ms,
        max_latency_regression_ratio=max_latency_regression_ratio,
    )
    if json_output:
        from dataclasses import asdict

        typer.echo(json.dumps(asdict(comparison), indent=2, sort_keys=True))
    else:
        typer.echo(format_benchmark_comparison(comparison))
    if not comparison.passed:
        raise typer.Exit(code=1)


def main() -> None:
    app()


def _parse_cli_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_activation_packet(packet: dict[str, object]) -> str:
    bootstrap = packet["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise TypeError("activation packet bootstrap must be a dictionary")
    injection_text = str(packet["injection_text"])
    return "\n".join(
        [
            "# Zaxy Codex Activation",
            f"Session: {packet['session_id']}",
            f"Workspace: {packet['workspace']}",
            "",
            "Inject this at Codex session start, then follow the startup sequence:",
            "",
            injection_text,
            "",
            str(packet["next_step"]),
        ]
    )


def _codex_activation_command(executable: str, workspace: Path, prompt: str) -> list[str]:
    return [executable, "--cd", str(workspace), prompt]


if __name__ == "__main__":
    main()
