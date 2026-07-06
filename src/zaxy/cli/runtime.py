"""Split from cli.py (mechanical decomposition)."""


from __future__ import annotations

import importlib
import json
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from typer.core import TyperGroup

from zaxy.cli.plugins import plugin_app

if TYPE_CHECKING:
    pass


def _memory_fabric(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for CLI commands that construct MemoryFabric."""
    from zaxy.core import MemoryFabric as _MemoryFabric

    return _MemoryFabric(*args, **kwargs)


MemoryFabric = _memory_fabric


def _benchmark_module(module_name: str) -> Any:
    """Import source-checkout benchmark/eval modules with a clear runtime error."""
    try:
        return importlib.import_module(f"zaxy_benchmarks.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name == "zaxy_benchmarks" or str(exc.name).startswith("zaxy_benchmarks."):
            raise typer.BadParameter(
                "Benchmark and external-evaluation commands require the optional "
                "source-checkout eval package `zaxy_benchmarks`. Run this command "
                "from a Zaxy source checkout or install the eval tooling package."
            ) from exc
        raise


def _graph_store(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for CLI commands that construct GraphStore."""
    from zaxy.graph import GraphStore as _GraphStore

    return _GraphStore(*args, **kwargs)


GraphStore = _graph_store


def _embedded_graph_store(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for CLI commands that construct EmbeddedGraphStore."""
    from zaxy.embedded_graph_store import EmbeddedGraphStore as _EmbeddedGraphStore

    return _EmbeddedGraphStore(*args, **kwargs)


EmbeddedGraphStore = _embedded_graph_store


def capture_codex_sessions(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for local Codex capture."""
    from zaxy.codex_capture import capture_codex_sessions as _capture_codex_sessions

    return _capture_codex_sessions(*args, **kwargs)


def capture_claude_sessions(*args: Any, **kwargs: Any) -> Any:
    """Patchable lazy seam for local Claude Code capture."""
    from zaxy.claude_capture import capture_claude_sessions as _capture_claude_sessions

    return _capture_claude_sessions(*args, **kwargs)


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


def onboarding_result_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Patchable lazy seam for onboarding JSON output."""
    from zaxy.onboarding import onboarding_result_payload as _onboarding_result_payload

    return dict(_onboarding_result_payload(*args, **kwargs))


def resolve_zaxy_executable(*args: Any, **kwargs: Any) -> str:
    """Patchable lazy seam for executable resolution."""
    from zaxy.install import resolve_zaxy_executable as _resolve_zaxy_executable

    return str(_resolve_zaxy_executable(*args, **kwargs))


# ---------------------------------------------------------------------------
# `zaxy --help` organization.
#
# The top-level command list is large. Group it into ordered, labeled panels so
# high-value commands lead and testing/benchmark commands trail. Typer renders
# command panels in `list_commands()` order, so the custom group below assigns
# each command a ``rich_help_panel`` from this map and lists commands grouped by
# panel in the order declared here (top panel first).
# ---------------------------------------------------------------------------
_COMMAND_PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Essentials",
        ("init", "serve", "dashboard", "activate", "doctor", "status"),
    ),
    (
        "Memory",
        (
            "memory",
            "coordinate",
            "fleet",
            "replay",
            "index-codebase",
            "refresh-context",
            "compact",
            "crystallize",
            "reproject",
        ),
    ),
    (
        "Setup & integrations",
        (
            "ide-config",
            "integrations",
            "integration-template",
            "local-profile",
            "init-session",
            "extractor-template",
            "plugin",
        ),
    ),
    (
        "Capture & observability",
        (
            "capture",
            "codex-capture",
            "claude-capture",
            "capture-soak",
            "hooks",
            "hook-event",
            "hook-status",
            "packet-analyzer",
            "packet-project",
            "packet-status",
            "trace",
        ),
    ),
    (
        "Export & verification",
        (
            "export",
            "export-keygen",
            "verify-export",
            "export-disclose",
            "verify-export-subset",
            "export-push",
        ),
    ),
    (
        "Inspection & maintenance",
        ("schema-plan", "schema-recovery-plan", "viewer", "offload-get"),
    ),
    (
        "Benchmarks & evaluation",
        (
            "benchmark",
            "fleet-benchmark",
            "benchmark-inventory",
            "benchmark-compare",
            "benchmark-freeze",
            "purpose-benchmark",
            "harvey-lab-benchmark",
            "harvey-lab-import",
            "harvey-lab-index",
            "harvey-lab-adapter-kit",
            "harvey-lab-ready",
            "harvey-lab-plan",
            "harvey-lab-normalize-run",
            "harvey-lab-gate",
            "harvey-lab-validate",
            "harvey-lab-publish",
            "harvey-lab-doctor",
            "harvey-lab-preflight",
            "harvey-lab-status",
            "longmembench-bootstrap",
            "longmembench-adapter-kit",
            "longmembench-plan",
            "longmembench-ready",
            "longmembench-import",
            "longmembench-generate-hypotheses",
            "longmembench-evaluate-official",
            "longmembench-validator-evidence",
            "longmembench-validate",
            "longmembench-gate",
            "longmembench-audit",
            "longmembench-publish",
            "longmembench-doctor",
        ),
    ),
    (
        "Internal & experimental lanes",
        (
            "state-recovery-benchmark",
            "agent-experience-lanes",
            "cognitive-lanes",
            "graph-scale-lanes",
            "experimental",
        ),
    ),
)

_FALLBACK_PANEL = "Other commands"

_COMMAND_PANEL: dict[str, str] = {}
_COMMAND_RANK: dict[str, tuple[int, int]] = {}
for _panel_index, (_panel_name, _panel_cmds) in enumerate(_COMMAND_PANELS):
    for _cmd_index, _cmd_name in enumerate(_panel_cmds):
        _COMMAND_PANEL[_cmd_name] = _panel_name
        _COMMAND_RANK[_cmd_name] = (_panel_index, _cmd_index)

# Panels whose commands are benchmark/evaluation/experimental internals: they
# stay fully runnable but are hidden from the root ``zaxy --help`` so the everyday
# surface stays small. A newcomer sees the essentials and the everyday groups,
# not ~55 benchmark and release-gate commands. Reach them via the command name
# directly (e.g. ``zaxy benchmark --help``) — they are omitted from the listing,
# not removed.
_HIDDEN_PANELS: frozenset[str] = frozenset(
    {"Benchmarks & evaluation", "Internal & experimental lanes"}
)
# Individual power/maintenance commands hidden from the root listing while their
# everyday panel stays visible.
_HIDDEN_EXTRA_COMMANDS: frozenset[str] = frozenset(
    {"crystallize", "reproject", "capture-soak", "offload-get"}
)
_HIDDEN_COMMANDS: frozenset[str] = (
    frozenset(name for name, panel in _COMMAND_PANEL.items() if panel in _HIDDEN_PANELS)
    | _HIDDEN_EXTRA_COMMANDS
)


class _PanelOrderedGroup(TyperGroup):
    """Render top-level ``zaxy`` help in ordered, labeled command panels.

    Each command is assigned a ``rich_help_panel`` from ``_COMMAND_PANEL`` and
    listed grouped by panel in ``_COMMAND_PANELS`` order, so high-value commands
    lead and testing/benchmark commands trail. Commands missing from the map
    fall back to a trailing "Other commands" panel rather than disappearing,
    keeping new commands visible until they are categorized.
    """

    def list_commands(self, ctx: Any) -> list[str]:
        names = super().list_commands(ctx)
        fallback_rank = (len(_COMMAND_PANELS), 0)

        def sort_key(name: str) -> tuple[int, int, int]:
            panel_index, cmd_index = _COMMAND_RANK.get(name, fallback_rank)
            return (panel_index, cmd_index, names.index(name))

        ordered = sorted(names, key=sort_key)
        for name in ordered:
            command = self.commands.get(name)
            if command is not None:
                cast(Any, command).rich_help_panel = _COMMAND_PANEL.get(name, _FALLBACK_PANEL)
                # Hidden commands stay runnable (invocation does not consult
                # ``hidden``); they are only dropped from the rendered help and
                # shell completion.
                cast(Any, command).hidden = name in _HIDDEN_COMMANDS
        return ordered


app = typer.Typer(cls=_PanelOrderedGroup, help="Zaxy: Event-sourced temporal knowledge graph fabric")


memory_app = typer.Typer(help="Inspect Eventloom-backed agent memory")


memory_purpose_app = typer.Typer(help="Inspect replay-backed purpose control-plane diagnostics")


memory_causal_app = typer.Typer(help="Inspect causal memory graph relationships")


memory_consolidation_app = typer.Typer(help="Create and review consolidation candidates")


memory_reasoning_app = typer.Typer(help="Run MemoryFabric reasoning-loop primitives")


capture_app = typer.Typer(help="Manage deterministic capture watchers")


coordinate_app = typer.Typer(help="Coordinate parent missions and worker sessions")


coordinate_worker_app = typer.Typer(help="Manage worker sessions for a mission")


coordinate_template_app = typer.Typer(help="Inspect and apply Coordinate mission templates")


coordinate_benchmark_adapter_app = typer.Typer(help="Validate and export CoordinationBench adapter contracts")


trace_app = typer.Typer(help="Export neutral trace correlations from Eventloom")


experimental_app = typer.Typer(help="Run isolated experimental memory research commands")


fleet_app = typer.Typer(help="Govern the fleet memory plane (cross-agent/cross-session propagation)")


app.add_typer(memory_app, name="memory")


memory_app.add_typer(memory_purpose_app, name="purpose")


memory_app.add_typer(memory_causal_app, name="causal")


memory_app.add_typer(memory_consolidation_app, name="consolidation")


memory_app.add_typer(memory_reasoning_app, name="reasoning")


app.add_typer(capture_app, name="capture")


app.add_typer(coordinate_app, name="coordinate")


app.add_typer(trace_app, name="trace")


app.add_typer(experimental_app, name="experimental")


app.add_typer(fleet_app, name="fleet")


app.add_typer(plugin_app, name="plugin")


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


def _finding_ids(findings: list[Any]) -> str:
    return ", ".join(
        str(finding.get("finding_id"))
        for finding in findings
        if isinstance(finding, dict) and finding.get("finding_id")
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


@app.command("crystallize")
def crystallize(
    session_id: str = typer.Option("default", "--session-id", help="Session ID (Eventloom thread) to crystallize"),
    eventloom_path: Path = typer.Option(Path(".eventloom"), "--eventloom-path", help="Eventloom directory"),  # noqa: B008
    consolidation: bool = typer.Option(
        True, "--consolidation/--no-consolidation", help="Propose review-pending consolidation candidates"
    ),
    procedure_mining: bool = typer.Option(
        True, "--procedure-mining/--no-procedure-mining", help="Mine recurring procedures into candidates"
    ),
    compaction: bool = typer.Option(
        False, "--compaction/--no-compaction", help="Run the additive compaction audit/projection diagnostic"
    ),
    metacognition: bool = typer.Option(
        True, "--metacognition/--no-metacognition", help="Run the autonomous metacognition monitor"
    ),
    auto_apply: bool = typer.Option(
        True, "--auto-apply/--no-auto-apply", help="Let the I4 gate auto-accept high-confidence candidates"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Run one governed sleep-time crystallization pass (operator/cron-triggered; no daemon)."""
    import asyncio

    from zaxy.cli import serving as _serving
    from zaxy.cli.workspace import (
        _is_embedded_projection_lock_error,
        _resolve_cli_projection_backend,
    )
    from zaxy.crystallization import run_crystallization_pass

    settings = _serving._status_settings(_serving._profile_root_for_eventloom_path(eventloom_path))
    if not getattr(settings, "crystallization_enabled", False):
        typer.echo(
            "zaxy crystallize is disabled; set crystallization_enabled=true "
            "(env CRYSTALLIZATION_ENABLED=1) to enable the governed crystallization runner.",
            err=True,
        )
        raise typer.Exit(code=1)

    async def _crystallize_with_path(
        embedded_graph_path: Path, *, projection_backend_override: str | None = None
    ) -> Any:
        projection_backend = projection_backend_override or _resolve_cli_projection_backend(None, settings)
        fabric = _memory_fabric(
            eventloom_path=str(eventloom_path),
            projection_backend=projection_backend,
            pggraph_dsn=settings.pggraph_dsn,
            embedded_graph_path=embedded_graph_path,
            latticedb_path=Path(settings.latticedb_path),
        )
        try:
            await fabric.connect()
            return await run_crystallization_pass(
                fabric,
                session_id=session_id,
                consolidation=consolidation,
                procedure_mining=procedure_mining,
                compaction=compaction,
                metacognition=metacognition,
                auto_apply=auto_apply,
            )
        finally:
            with suppress(Exception):
                await fabric.close()

    async def _crystallize() -> Any:
        embedded_graph_path = Path(settings.embedded_graph_path)
        try:
            return await _crystallize_with_path(embedded_graph_path)
        except RuntimeError as exc:
            if not _is_embedded_projection_lock_error(exc):
                raise
            # A server holds the embedded projection's single-owner lock; keep the
            # crystallization appends durable by running with the graph lane degraded
            # (null backend), mirroring `memory outcome`.
            return await _crystallize_with_path(embedded_graph_path, projection_backend_override="null")

    try:
        report = asyncio.run(_crystallize())
    except (RuntimeError, OSError) as exc:
        typer.echo(f"zaxy crystallize failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(report.to_dict(), sort_keys=True))
    else:
        typer.echo(
            f"Crystallization pass for {report.session_id}: "
            f"consolidation={report.consolidation_candidates}, "
            f"procedure={report.procedure_candidates}, "
            f"auto_accepted={report.auto_accepted}, pending={report.left_pending}, "
            f"reverify_requested={report.reverify_requested}"
        )


@app.command("fleet-benchmark")
def fleet_benchmark(
    output_dir: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/fleet-v1"),
        "--output-dir",
        help="Directory for FleetBench reports and run artifacts",
    ),
    worker_counts: str = typer.Option(
        "3,5,8", "--worker-counts", help="Comma-separated worker counts to scale across"
    ),
    missions: int = typer.Option(1, help="Missions per scale point"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Run the FleetBench scaling scaffold (fleet axes over real CoordinationBench runs)."""
    try:
        counts = tuple(int(part) for part in worker_counts.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter("worker-counts must be comma-separated integers") from exc
    if not counts:
        raise typer.BadParameter("worker-counts must contain at least one integer")
    fleet_benchmark_module = _benchmark_module("fleet_benchmark")
    report = fleet_benchmark_module.run_fleet_benchmark(
        output_dir, worker_counts=counts, missions=missions
    )
    payload = report.to_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"FleetBench complete: fingerprint={payload['fingerprint']}")
    typer.echo(f"scale_points={[r['worker_count'] for r in payload['results']]}")
    typer.echo(f"cross_agent_transfer_scope={payload['cross_agent_transfer_scope']}")


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
    coordination_benchmark_module = _benchmark_module("coordination_benchmark")
    export_coordination_benchmark_adapter_kit = (
        coordination_benchmark_module.export_coordination_benchmark_adapter_kit
    )

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
    coordination_benchmark_module = _benchmark_module("coordination_benchmark")
    coordination_competitor_claim_gate = coordination_benchmark_module.coordination_competitor_claim_gate
    run_coordination_benchmark = coordination_benchmark_module.run_coordination_benchmark

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


@coordinate_benchmark_adapter_app.command("validate-manifest")
def coordinate_benchmark_adapter_validate_manifest(
    adapter: str = typer.Argument(..., help="Adapter manifest as NAME=PATH"),
    workload: Path = typer.Option(..., "--workload", help="Frozen CoordinationBench workload JSON"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Validate a runner manifest without executing it."""
    coordination_benchmark_module = _benchmark_module("coordination_benchmark")
    load_coordination_workload = coordination_benchmark_module.load_coordination_workload
    validate_coordination_competitor_runner_manifest = (
        coordination_benchmark_module.validate_coordination_competitor_runner_manifest
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
    coordination_benchmark_module = _benchmark_module("coordination_benchmark")
    load_coordination_workload = coordination_benchmark_module.load_coordination_workload
    validate_coordination_competitor_result = (
        coordination_benchmark_module.validate_coordination_competitor_result
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


def _fleet_manager(eventloom_path: Path) -> Any:
    from zaxy.config import Settings
    from zaxy.coordination_semantic import build_semantic_conflict_detector
    from zaxy.fleet import FleetManager

    settings = Settings()
    return FleetManager(
        eventloom_path=eventloom_path,
        settings=settings,
        semantic_conflict_detector=build_semantic_conflict_detector(settings),
    )


def _fleet_source_events(source_event: list[str] | None) -> list[dict[str, object]]:
    if not source_event:
        raise typer.BadParameter("at least one --source-event SEQ:HASH is required")
    return [_parse_source_event(value) for value in source_event]


def _emit_fleet_promotion(result: Any, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    elif result.rejected:
        typer.echo(f"Promotion rejected: {result.reason}")
    else:
        typer.echo(
            f"Promotion {result.promotion_id} recorded "
            f"(review_status={result.review_status}, auto_applied={result.auto_applied})"
        )
    if result.rejected:
        raise typer.Exit(code=1)


@fleet_app.command("create")
def fleet_create(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    summary: str = typer.Option(..., "--summary", help="Human summary of the fleet"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Create a governed fleet thread."""
    result = _fleet_manager(eventloom_path).create_fleet(fleet_id, summary=summary, actor=actor)
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(f"Fleet {result.fleet_id} created")


@fleet_app.command("enroll")
def fleet_enroll(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    agent: str = typer.Option(..., "--agent", help="Agent identifier to enroll"),
    trust_tier: str | None = typer.Option(None, "--trust-tier", help="Trust tier (default: configured fleet_default_trust_tier)"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    actor: str = typer.Option("coordinator", help="Actor recording the event"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Enroll an agent into a fleet (first enrollee of a steward-less fleet becomes steward)."""
    from zaxy.config import Settings
    from zaxy.fleet import validate_trust_tier

    tier = validate_trust_tier(trust_tier or Settings().fleet_default_trust_tier)
    result = _fleet_manager(eventloom_path).enroll_agent(fleet_id, agent, actor=actor, trust_tier=tier)
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        suffix = " (first-steward bootstrap)" if result.bootstrap_steward else ""
        typer.echo(f"Agent {result.agent_id} enrolled as {result.trust_tier}{suffix}")


@fleet_app.command("assign-trust")
def fleet_assign_trust(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    agent: str = typer.Option(..., "--agent", help="Agent identifier"),
    trust_tier: str = typer.Option(..., "--trust-tier", help="New trust tier"),
    actor: str = typer.Option(..., "--actor", help="Steward actor authorizing the assignment"),
    rationale: str | None = typer.Option(None, "--rationale", help="Reason for the assignment"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Assign a trust tier to an enrolled agent (steward authority)."""
    result = _fleet_manager(eventloom_path).assign_trust(
        fleet_id, agent, trust_tier=trust_tier, actor=actor, rationale=rationale
    )
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(f"Agent {result.agent_id} assigned trust tier {result.trust_tier}")


@fleet_app.command("promote-skill")
def fleet_promote_skill(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    skill_id: str = typer.Option(..., "--skill-id", help="Skill identifier"),
    skill_version: str = typer.Option(..., "--skill-version", help="Skill version"),
    origin_session: str = typer.Option(..., "--origin-session", help="Originating session/thread"),
    source_event: list[str] | None = typer.Option(None, "--source-event", help="Cited source event SEQ:HASH (repeatable)"),  # noqa: B008
    confidence: float = typer.Option(..., "--confidence", help="Promotion confidence in [0,1]"),
    actor: str = typer.Option(..., "--actor", help="Proposing actor (must be an enrolled agent)"),
    origin_actor: str | None = typer.Option(None, "--origin-actor", help="Agent that originally learned the skill"),
    visibility_scope: str = typer.Option("fleet", "--visibility-scope", help="Target scope: fleet or global"),
    keystone: bool = typer.Option(False, "--keystone", help="Mark as a fleet keystone"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Propose a skill promotion to fleet scope through the I4 gate."""
    result = _fleet_manager(eventloom_path).promote_skill(
        fleet_id,
        skill_id=skill_id,
        skill_version=skill_version,
        origin_session=origin_session,
        source_events=_fleet_source_events(source_event),
        confidence=confidence,
        actor=actor,
        origin_actor=origin_actor,
        visibility_scope=visibility_scope,
        keystone=keystone,
    )
    _emit_fleet_promotion(result, json_output)


@fleet_app.command("promote-outcome")
def fleet_promote_outcome(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    outcome: str = typer.Option(..., "--outcome", help="Outcome: success, failure, or partial"),
    summary: str = typer.Option(..., "--summary", help="Outcome summary"),
    origin_session: str = typer.Option(..., "--origin-session", help="Originating session/thread"),
    source_event: list[str] | None = typer.Option(None, "--source-event", help="Cited source event SEQ:HASH (repeatable)"),  # noqa: B008
    confidence: float = typer.Option(..., "--confidence", help="Promotion confidence in [0,1]"),
    actor: str = typer.Option(..., "--actor", help="Proposing actor (must be an enrolled agent)"),
    origin_actor: str | None = typer.Option(None, "--origin-actor", help="Agent that originally learned the outcome"),
    claim_key: str | None = typer.Option(None, "--claim-key", help="Claim key for conflict detection"),
    visibility_scope: str = typer.Option("fleet", "--visibility-scope", help="Target scope: fleet or global"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Propose an outcome propagation to fleet scope through the I4 gate."""
    result = _fleet_manager(eventloom_path).propagate_outcome(
        fleet_id,
        outcome=outcome,
        summary=summary,
        origin_session=origin_session,
        source_events=_fleet_source_events(source_event),
        confidence=confidence,
        actor=actor,
        origin_actor=origin_actor,
        claim_key=claim_key,
        visibility_scope=visibility_scope,
    )
    _emit_fleet_promotion(result, json_output)


@fleet_app.command("promote-rule")
def fleet_promote_rule(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    rule: str = typer.Option(..., "--rule", help="Preventive rule text"),
    trigger: str = typer.Option(..., "--trigger", help="Trigger condition for the rule"),
    origin_session: str = typer.Option(..., "--origin-session", help="Originating session/thread"),
    source_event: list[str] | None = typer.Option(None, "--source-event", help="Cited source event SEQ:HASH (repeatable)"),  # noqa: B008
    confidence: float = typer.Option(..., "--confidence", help="Promotion confidence in [0,1]"),
    actor: str = typer.Option(..., "--actor", help="Proposing actor (must be an enrolled agent)"),
    origin_actor: str | None = typer.Option(None, "--origin-actor", help="Agent that originally learned the rule"),
    visibility_scope: str = typer.Option("fleet", "--visibility-scope", help="Target scope: fleet or global"),
    keystone: bool = typer.Option(False, "--keystone", help="Mark as a fleet keystone"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Propose a preventive-rule propagation to fleet scope through the I4 gate."""
    result = _fleet_manager(eventloom_path).propagate_rule(
        fleet_id,
        rule=rule,
        trigger=trigger,
        origin_session=origin_session,
        source_events=_fleet_source_events(source_event),
        confidence=confidence,
        actor=actor,
        origin_actor=origin_actor,
        visibility_scope=visibility_scope,
        keystone=keystone,
    )
    _emit_fleet_promotion(result, json_output)


@fleet_app.command("review")
def fleet_review(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    promotion: str = typer.Option(..., "--promotion", help="Promotion id to review"),
    decision: str = typer.Option(..., "--decision", help="accepted, rejected, or deferred"),
    actor: str = typer.Option(..., "--actor", help="Steward actor recording the review"),
    rationale: str | None = typer.Option(None, "--rationale", help="Review rationale"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Steward review of a held promotion (accepted activates a pending memory)."""
    result = _fleet_manager(eventloom_path).review_promotion(
        fleet_id, promotion, decision=decision, actor=actor, rationale=rationale
    )
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(f"Promotion {result.promotion_id} reviewed: {decision}")


@fleet_app.command("rollback")
def fleet_rollback(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    promotion: str = typer.Option(..., "--promotion", help="Promotion id to roll back"),
    reason: str = typer.Option(..., "--reason", help="Reason for the rollback"),
    actor: str = typer.Option(..., "--actor", help="Actor recording the rollback"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Reversibly un-share a promotion (lowers effective scope additively)."""
    result = _fleet_manager(eventloom_path).rollback_promotion(
        fleet_id, promotion, reason=reason, actor=actor
    )
    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(f"Promotion {result.promotion_id} rolled back")


@fleet_app.command("status")
def fleet_status(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Print a governed fleet brief (active promotions, enrolled agents, trust tiers)."""
    brief = _fleet_manager(eventloom_path).fleet_brief(fleet_id)
    if json_output:
        typer.echo(json.dumps(brief.to_dict(), indent=2, sort_keys=True))
        return
    typer.echo(f"Fleet {brief.fleet_id}: {brief.summary or '-'}")
    typer.echo(f"Agents: {len(brief.agents)}")
    typer.echo(f"Active promotions: {len(brief.active_promotions)}")
    typer.echo(f"Pending promotions: {len(brief.pending_promotions)}")


@fleet_app.command("audit")
def fleet_audit(
    fleet_id: str = typer.Argument(..., help="Fleet identifier"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Print full provenance for every fleet memory (replay-only)."""
    report = _fleet_manager(eventloom_path).fleet_audit(fleet_id)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    typer.echo(f"Fleet {report.fleet_id}: {len(report.records)} memory record(s)")
    for record in report.records:
        typer.echo(
            f"- {record.promotion_id} [{record.review_status}] {record.kind} "
            f"by {record.origin_actor} from {len(record.source_events)} source event(s)"
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


def _parse_source_event(value: str) -> dict[str, object]:
    seq_text, separator, event_hash = value.partition(":")
    if separator != ":" or not seq_text or not event_hash:
        raise typer.BadParameter("source event must be formatted as SEQ:HASH")
    try:
        seq = int(seq_text)
    except ValueError as exc:
        raise typer.BadParameter("source event sequence must be an integer") from exc
    if seq <= 0:
        raise typer.BadParameter("source event sequence must be a positive integer")
    if len(event_hash) != 64 or any(char not in "0123456789abcdef" for char in event_hash):
        raise typer.BadParameter("source event hash must be exactly 64 lowercase hex characters")
    return {"seq": seq, "hash": event_hash}


def _validate_causal_relation_type_option(value: str | None) -> str | None:
    if value is None:
        return None
    from zaxy.causal import causal_relation_to_graph_relation

    try:
        causal_relation_to_graph_relation(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return value


REASONING_PHASES = {"planning", "execution", "review", "reflection"}


def _validate_reasoning_phase_option(value: str) -> str:
    phase = value.strip()
    if phase not in REASONING_PHASES:
        raise typer.BadParameter(
            "reasoning phase must be one of: " + ", ".join(sorted(REASONING_PHASES))
        )
    return phase


def _format_reasoning_result_text(result: dict[str, Any]) -> str:
    primitive = result.get("primitive") or "reasoning"
    session_id = result.get("session_id") or "default"
    result_count = result.get("result_count")
    if result_count is None:
        for key in ("explanations", "evidence", "procedures", "results"):
            values = result.get(key)
            if isinstance(values, list):
                result_count = len(values)
                break
    if result_count is not None:
        return f"{primitive} for {session_id}: result_count={int(result_count)}"
    return f"{primitive} for {session_id}"


def _format_causal_results_text(
    *,
    direction: str,
    entity_name: str,
    results: list[dict[str, object]],
) -> str:
    if not results:
        return f"No causal {direction} found for {entity_name}."

    lines = []
    for result in results:
        source = result.get("source")
        target = result.get("target")
        source_name = source.get("name") if isinstance(source, dict) else None
        target_name = target.get("name") if isinstance(target, dict) else None
        relation_type = result.get("relation_type")
        confidence = result.get("confidence")
        citation = result.get("citation")
        path_length = result.get("path_length")
        pieces = [str(relation_type)]
        if isinstance(confidence, int | float):
            pieces.append(f"confidence={confidence:.3f}")
        if isinstance(path_length, int):
            pieces.append(f"path_length={path_length}")
        if citation:
            pieces.append(f"citation={citation}")
        lines.append(f"{source_name or '?'} -> {target_name or '?'} [{', '.join(pieces)}]")
    return "\n".join(lines)


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
