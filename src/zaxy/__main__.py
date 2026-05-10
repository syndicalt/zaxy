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
from dataclasses import asdict
from pathlib import Path

import typer

from zaxy.benchmark import build_competitive_event_log, competitive_cases
from zaxy.compaction import (
    audit_event_log,
    build_compaction_projection,
    write_compaction_projection,
)
from zaxy.core import MemoryFabric
from zaxy.doctor import format_doctor_report, run_doctor
from zaxy.embedding import EmbeddingProvider, HashEmbeddingProvider, OpenAIEmbeddingProvider
from zaxy.event import EventLog
from zaxy.extract import extract
from zaxy.extract_templates import ExtractorTemplateSpec, render_extractor_template
from zaxy.graph import GraphStore
from zaxy.hooks import (
    build_hook_payload,
    format_hook_status,
    hook_event_type,
    inspect_hook_status,
    render_hook_config,
    write_claude_code_hook_config,
    write_hook_config,
)
from zaxy.integrations import (
    list_framework_integration_specs,
    render_agent_integration_template,
    render_codex_mcp_add_command,
    render_framework_install_command,
    render_mcp_client_config,
    write_codex_mcp_config,
    write_project_mcp_client_config,
)
from zaxy.lifecycle import build_compaction_completed_event
from zaxy.live_benchmark import (
    BenchmarkWorkload,
    BM25Retriever,
    CachedEmbeddingProvider,
    CentroidConsolidationRetriever,
    ExternalBenchmarkResult,
    MarkdownRetriever,
    MarkdownVectorRetriever,
    VectorRetriever,
    benchmark_live_retrievers,
    build_benchmark_suite_workload,
    build_consolidation_collapse_workload,
    build_frozen_statistical_workload,
    build_live_zaxy_retriever,
    build_longmemeval_workload,
    build_statistical_event_log,
    corpus_from_event_log,
    report_to_markdown,
    write_benchmark_report,
)
from zaxy.local_profile import check_local_profile, render_local_profile, write_local_profile
from zaxy.mcp_server import main as mcp_main
from zaxy.memory_status import (
    format_memory_diff,
    format_memory_log,
    format_memory_status,
    inspect_memory_diff,
    inspect_memory_log,
    inspect_memory_status,
)
from zaxy.observation import (
    build_command_observation,
    build_file_edit_observation,
    build_tool_call_observation,
    build_transcript_turn_observation,
)
from zaxy.onboarding import (
    OnboardingResult,
    apply_onboarding_preset,
    format_onboarding_result,
    run_onboarding,
)
from zaxy.packet_analyzer import PacketAnalyzerConfig, run_packet_analyzer
from zaxy.schema import render_schema_plan
from zaxy.viewer import write_viewer_html

app = typer.Typer(help="Zaxy: Event-sourced temporal knowledge graph fabric")
memory_app = typer.Typer(help="Inspect Eventloom-backed agent memory")
app.add_typer(memory_app, name="memory")


@memory_app.command("status")
def memory_status(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory or JSONL log"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show read-only Eventloom memory status."""
    status = inspect_memory_status(eventloom_path)
    if json_output:
        typer.echo(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(format_memory_status(status))


@memory_app.command("log")
def memory_log(
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory or JSONL log"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Session ID to inspect"),  # noqa: B008
    limit: int = typer.Option(20, min=0, help="Maximum events to print"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show recent Eventloom memory events."""
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


@app.command("ide-config")
def ide_config(
    client: str = typer.Argument(..., help="MCP client: claude-desktop, claude-code, codex, cursor, or vscode"),  # noqa: B008
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
) -> None:
    """Print or install a first-run MCP client configuration fragment."""
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
) -> None:
    """List direct framework integration support and install extras."""
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
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Inspect observer hook installation and recent lifecycle activity."""
    report = inspect_hook_status(eventloom_path=eventloom_path, workspace_root=workspace_root)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(format_hook_status(report))


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
    typer.echo(f"Recorded hook {payload['trigger']} as {event_type} seq={event.seq}")


def _parse_json_object(value: str, *, option: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{option} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{option} must be a JSON object")
    return parsed


@app.command("local-profile")
def local_profile(
    output: Path | None = typer.Option(None, "--output", "-o", help="Write profile to this file"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite an existing output file"),  # noqa: B008
    check: bool = typer.Option(False, "--check", help="Validate deterministic local providers"),  # noqa: B008
) -> None:
    """Print, write, or check an offline local retrieval profile."""
    if check:
        typer.echo(json.dumps(check_local_profile(), indent=2, sort_keys=True))
        return
    if output is None:
        typer.echo(render_local_profile(), nl=False)
        return
    try:
        written = write_local_profile(output, force=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote local profile to {written}")


@app.command("doctor")
def doctor(
    eventloom_path: str | None = typer.Option(None, help="Override Eventloom path for this check"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Run local setup and onboarding checks."""
    from zaxy.config import get_settings

    settings = get_settings()
    if eventloom_path is not None:
        settings = settings.model_copy(update={"eventloom_path": eventloom_path})
    report = run_doctor(settings=settings)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(format_doctor_report(report))


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
            return await fabric.ingest_codebase(path, session_id=session_id, max_bytes=max_bytes)
        finally:
            await fabric.close()

    count = asyncio.run(_run())
    typer.echo(f"Indexed {count} codebase events into session {session_id}")


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
            return await fabric.initialize_session(path, session_id=session_id)
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
    preset: str | None = typer.Option(None, help="Onboarding preset: local-claude"),  # noqa: B008
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory for this workspace"),
    domain: str | None = typer.Option(None, help="Project/domain used for default session scoping"),  # noqa: B008
    session_id: str | None = typer.Option(None, help="Explicit session ID; defaults to <domain>-default"),  # noqa: B008
    mcp_client: str | None = typer.Option(None, help="MCP client config to render/write"),  # noqa: B008
    mcp_output: Path | None = typer.Option(None, help="Write MCP config JSON to this file"),  # noqa: B008
    hook_client: str | None = typer.Option(None, help="Hook client config to render/write"),  # noqa: B008
    hook_output: Path | None = typer.Option(None, help="Write hook config to this file"),  # noqa: B008
    local_profile_output: Path | None = typer.Option(None, help="Write local retrieval profile to this file"),  # noqa: B008
    infra: str = typer.Option("none", help="Local infra action: none, check, or start"),  # noqa: B008
    zaxy_executable: str | None = typer.Option(None, help="Executable path MCP clients should invoke"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite generated output files"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),  # noqa: B008
) -> None:
    """Run first-run onboarding: MCP config, hooks, infra, genesis, heartbeat, doctor, hook status."""
    import asyncio

    async def _run() -> OnboardingResult:
        preset_options = apply_onboarding_preset(
            preset,
            workspace=path,
            mcp_client=mcp_client,
            mcp_output=mcp_output,
            hook_client=hook_client,
            hook_output=hook_output,
            local_profile_output=local_profile_output,
            infra=infra,
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
    written = write_viewer_html(path, output)
    typer.echo(f"Wrote Eventloom viewer: {written}")


@app.command("schema-plan")
def schema_plan() -> None:
    """Print the current Neo4j schema migration plan."""
    typer.echo(render_schema_plan())


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
def reproject(
    log_path: Path = typer.Argument(..., help="Path to Eventloom JSONL file"),  # noqa: B008
    from_seq: int = typer.Option(1, help="Start sequence number"),
    session_id: str = typer.Option("default", help="Graph session ID to project into"),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
) -> None:
    """Replay an Eventloom log and rebuild its Neo4j graph projection."""
    import asyncio

    from zaxy.config import get_settings

    async def _run() -> int:
        settings = get_settings()
        store = GraphStore(
            neo4j_uri or settings.neo4j_uri,
            neo4j_user or settings.neo4j_user,
            neo4j_password or settings.neo4j_password,
        )
        await store.connect()
        try:
            await store.init_schema()
            replay_result = EventLog(str(log_path)).replay(from_seq=from_seq)
            if not replay_result.integrity.ok:
                reason = replay_result.integrity.broken_reason or "unknown integrity failure"
                raise typer.BadParameter(f"Eventloom integrity failed: {reason}")
            for event in replay_result.events:
                await store.upsert_extraction(extract(event), session_id=session_id)
            return len(replay_result.events)
        finally:
            await store.close()

    count = asyncio.run(_run())
    typer.echo(f"Reprojected {count} events into session {session_id}")


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
) -> None:
    """Compact an Eventloom log and optionally create snapshots."""
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
    workload: str = typer.Option(
        "fixture",
        help="Workload: fixture, statistical, frozen, suite, consolidation, or longmemeval",
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
) -> None:
    """Run live retrieval benchmarks against md/BM25/vector/md+vector/Zaxy."""
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
                    "'suite', 'consolidation', or 'longmemeval'"
                )
            corpus = corpus_from_event_log(eventlog)
            zaxy_retriever, graph = await build_live_zaxy_retriever(
                eventlog,
                provider,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                reset_graph=reset_graph,
                lexical_retriever=BM25Retriever(corpus),
            )
            try:
                report = await benchmark_live_retrievers(
                    {
                        "md": MarkdownRetriever(corpus),
                        "bm25": BM25Retriever(corpus),
                        "vector": VectorRetriever(corpus, provider),
                        "md+vector": MarkdownVectorRetriever(corpus, provider),
                        **(
                            {"centroid": CentroidConsolidationRetriever(corpus, provider)}
                            if workload == "consolidation"
                            else {}
                        ),
                    },
                    zaxy_retriever,
                    cases,
                    runs=runs,
                    limit=limit,
                    embedding_provider=provider_label,
                    workload=benchmark_workload,
                    external_results=_load_external_results(external_results),
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


def _load_external_results(path: Path | None) -> tuple[ExternalBenchmarkResult, ...]:
    """Load operator-supplied external benchmark rows from JSON."""
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
