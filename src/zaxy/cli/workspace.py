"""Split from cli.py (mechanical decomposition)."""


from __future__ import annotations

import json
import os
import shlex
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from zaxy.cli import runtime as _runtime

if TYPE_CHECKING:
    from zaxy.config import Settings
from zaxy.cli.runtime import (
    app,
    apply_onboarding_preset,
    capture_app,
    format_onboarding_result,
    memory_app,
    onboarding_result_payload,
    resolve_zaxy_executable,
)


def _shell_join(command: list[str]) -> str:
    """Return a POSIX-shell-safe command string."""
    import shlex

    return shlex.join(command)


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


def _checkout_activity_metadata(payload: dict[str, object]) -> dict[str, object]:
    token_efficiency = payload.get("token_efficiency")
    if isinstance(token_efficiency, dict):
        return {"token_efficiency": token_efficiency}
    return {}


def _is_embedded_projection_lock_error(exc: RuntimeError) -> bool:
    from zaxy.embedded_graph_internals import is_embedded_projection_lock_error

    return is_embedded_projection_lock_error(exc)


def _checkout_fallback_embedded_graph_path(*, eventloom_path: Path, session_id: str) -> Path:
    from zaxy.security import validate_session_id

    base = eventloom_path if eventloom_path.suffix != ".jsonl" else eventloom_path.parent
    sid = validate_session_id(session_id)
    return base / "projections" / f"checkout-{sid}-{os.getpid()}.kuzu"


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
        store = _runtime.GraphStore(
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
        result = _runtime.capture_codex_sessions(
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


@app.command("claude-capture")
def claude_capture(
    workspace: Path = typer.Option(Path("."), help="Workspace root whose Claude Code sessions should be captured"),  # noqa: B008
    claude_home: Path | None = typer.Option(None, help="Claude config home; defaults to CLAUDE_CONFIG_DIR or ~/.claude"),  # noqa: B008
    eventloom_path: Path = typer.Option(Path(".eventloom"), help="Eventloom directory for captured observations"),  # noqa: B008
    session_id: str = typer.Option("default", help="Zaxy Eventloom session ID to append into"),
    source: str = typer.Option("claude-local", help="Capture source label"),
    max_records_per_file: int = typer.Option(
        1000,
        "--max-records-per-file",
        min=1,
        help="Maximum recent records to scan from each Claude session log per pass",
    ),
    graph: bool = typer.Option(False, "--graph", help="Project captured observations into Neo4j"),
    neo4j_uri: str | None = typer.Option(None, help="Neo4j Bolt URI"),
    neo4j_user: str | None = typer.Option(None, help="Neo4j username"),
    neo4j_password: str | None = typer.Option(None, help="Neo4j password"),
    watch: bool = typer.Option(False, "--watch", help="Continuously poll Claude session logs"),
    interval_seconds: float = typer.Option(2.0, "--interval-seconds", min=0.25, help="Watch poll interval"),
    watch_iterations: int | None = typer.Option(
        None,
        "--watch-iterations",
        min=1,
        help="Optional bounded watch pass count for supervisors and tests",
    ),
) -> None:
    """Capture local Claude Code session JSONL records into Eventloom without proxying model traffic."""
    import asyncio

    from zaxy.config import get_settings

    async def project_events(events: tuple[Any, ...]) -> int:
        if not events:
            return 0
        from zaxy.extract import extract

        settings = get_settings()
        store = _runtime.GraphStore(
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
        result = _runtime.capture_claude_sessions(
            workspace=workspace,
            claude_home=claude_home,
            eventloom_path=eventloom_path,
            session_id=session_id,
            source=source,
            max_records_per_file=max_records_per_file,
        )
        plural = "" if result.scanned_files == 1 else "s"
        typer.echo(
            f"Imported {result.imported} Claude observations from "
            f"{result.scanned_files} session log{plural} ({result.skipped} skipped)"
        )
        if graph:
            projected = asyncio.run(project_events(result.events))
            typer.echo(f"Projected {projected} captured observations into graph")

    if not watch:
        run_once()
        return
    typer.echo("Watching Claude session logs for deterministic Zaxy capture. Press Ctrl-C to stop.")
    try:
        iterations = 0
        while watch_iterations is None or iterations < watch_iterations:
            run_once()
            iterations += 1
            if watch_iterations is None or iterations < watch_iterations:
                time.sleep(interval_seconds)
    except KeyboardInterrupt:
        typer.echo("Stopped Claude capture.")


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


@app.command("hook-event")
def hook_event(
    trigger: str = typer.Argument(..., help="Hook trigger: session-start, resume, session-resumed, stop, precompact, checkpoint, heartbeat, command, file-edit, tool-call, or transcript-turn"),  # noqa: B008
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
    if normalized_trigger == "user-prompt-submit":
        # Deterministic recall lever: re-inject terse-prose memory state into the
        # model's context when memory is stale, and stay silent when fresh. For a
        # Claude Code UserPromptSubmit hook, stdout is injected into the prompt, so
        # this branch emits ONLY the structured JSON (no human-readable lines).
        from zaxy.memory_persistence import build_injection_context

        context = build_injection_context(eventloom_path, session_id=session_id)
        if context:
            typer.echo(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "UserPromptSubmit",
                            "additionalContext": context,
                        }
                    }
                )
            )
        return
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
            eventloom_path=eventloom_path,
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
            eventloom_path=eventloom_path,
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
    if payload["trigger"] == "session-resumed":
        from zaxy.recovery import assemble_recovery_packet, render_recovery_packet

        packet = assemble_recovery_packet(eventlog, session_id=session_id)
        typer.echo(render_recovery_packet(packet))


@app.command("offload-get")
def offload_get(
    sha256: str = typer.Argument(..., help="sha256 id from a full_io_ref pointer"),  # noqa: B008
    eventloom_path: str = typer.Option(".eventloom", help="Eventloom directory holding refs/"),
    fenced: bool = typer.Option(False, "--fenced", help="Wrap as injection-resistant untrusted data (when feeding the blob back to a model)"),  # noqa: B008
) -> None:
    """Drill down to the full offloaded tool I/O behind a `full_io_ref` sha256.

    Raw by default (for tooling); `--fenced` applies injection-resistant rehydration
    so captured (attacker-influenceable) content is safe to re-inject into a model.
    """
    from zaxy.offload import read_offload_ref

    content = read_offload_ref(eventloom_path, sha256)
    if content is None:
        raise typer.BadParameter(
            f"no offload blob for {sha256} under {eventloom_path}/refs (missing or integrity mismatch)"
        )
    if fenced:
        from zaxy.portable.rehydration import rehydrate

        typer.echo(rehydrate(content, origin="offload", label="offloaded tool I/O")["text"])
    else:
        typer.echo(content, nl=False)


@app.command("export-keygen")
def export_keygen(
    out_private: Path = typer.Option(..., "--out-private", help="Write PKCS8 PEM private key here (chmod 600)"),  # noqa: B008
    out_public: Path = typer.Option(..., "--out-public", help="Write hex public key here"),  # noqa: B008
    algorithm: str | None = typer.Option(None, "--algorithm", help="ml-dsa-65 (default if available) or ed25519"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite an existing keypair"),  # noqa: B008
) -> None:
    """Generate a self-sovereign signing keypair for portable memory export."""
    from zaxy.portable import generate_keypair

    if not force and (out_private.exists() or out_public.exists()):
        existing = out_private if out_private.exists() else out_public
        raise typer.BadParameter(f"{existing} already exists (use --force to overwrite)")
    if force and out_private.exists():
        # Drop any pre-existing file (which may carry a stale, permissive
        # mode) so the create-below always starts from a fresh inode.
        out_private.unlink()

    keypair = generate_keypair(algorithm)
    # Create the private key file pre-restricted to 0600 via the mode on
    # O_CREAT, so the plaintext signing key is never briefly world/group
    # readable at the process umask between write and chmod (TOCTOU window).
    fd = os.open(out_private, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(keypair["private_pem"])
    finally:
        out_private.chmod(0o600)  # belt-and-suspenders if the inode above wasn't freshly created
    out_public.write_text(keypair["public_key"].hex(), encoding="utf-8")
    typer.echo(
        f"keypair {keypair['algorithm']}: private -> {out_private} (chmod 600); public -> {out_public}"
    )


@app.command("export")
def export_memory(
    out: Path = typer.Option(..., "--out", help="Write the bundle JSON here"),  # noqa: B008
    private_key: Path | None = typer.Option(None, "--private-key", help="PKCS8 PEM private key file (omit for an unsigned bundle)"),  # noqa: B008
    public_key: Path | None = typer.Option(None, "--public-key", help="hex public key file (omit for an unsigned bundle)"),  # noqa: B008
    algorithm: str = typer.Option("ml-dsa-65", "--algorithm", help="signature algorithm of the key"),
    eventloom_path: str = typer.Option(".eventloom", "--eventloom-path"),
    session_id: str = typer.Option("default", "--session-id"),
    types: str = typer.Option("decision.made,goal.created,task.completed", "--types", help="comma-separated event types to export (empty for all)"),
    grains: str = typer.Option("event", "--grains", help="comma-separated grains: event, semantic"),
    since_seq: int | None = typer.Option(None, "--since", help="exclusive delta cursor: export events with seq > this"),
    max_seq: int | None = typer.Option(None, "--max-seq", help="inclusive upper bound on seq"),
    since_time: str | None = typer.Option(None, "--since-time", help="inclusive ISO-8601 lower time bound"),
    until_time: str | None = typer.Option(None, "--until-time", help="inclusive ISO-8601 upper time bound"),
    query: str | None = typer.Option(None, "--query", help="lexical pre-filter via the verbatim index"),
    exclude_sensitivities: str = typer.Option("", "--exclude-sensitivities", help="comma-separated sensitivity tiers to drop"),
    limit: int | None = typer.Option(None, "--limit", help="cap to the most recent N matching events"),
) -> None:
    """Build a verifiable export bundle from a session's memory.

    Signs the bundle when --private-key/--public-key are supplied; otherwise
    writes an unsigned canonical bundle. Uses the shared export contract, so the
    entries match the memory_export MCP tool exactly.
    """
    from zaxy.config import get_settings
    from zaxy.export_view import ExportSelector, build_memory_export, load_signing_key
    from zaxy.forgetting import build_vault
    from zaxy.retrieval_cache import SessionRetrievalCache
    from zaxy.session import SessionManager

    kinds = frozenset(t.strip() for t in types.split(",") if t.strip())
    selector = ExportSelector(
        grains=frozenset(g.strip() for g in grains.split(",") if g.strip()),
        kinds=kinds or None,
        since_seq=since_seq,
        max_seq=max_seq,
        since_time=since_time,
        until_time=until_time,
        query=query,
        exclude_sensitivities=frozenset(
            s.strip() for s in exclude_sensitivities.split(",") if s.strip()
        ),
        limit=limit,
    )

    signing_key = None
    if private_key is not None or public_key is not None:
        if private_key is None or public_key is None:
            raise typer.BadParameter("signing requires both --private-key and --public-key")
        signing_key = load_signing_key(
            private_key_path=private_key, public_key_path=public_key, algorithm=algorithm
        )

    cache = SessionRetrievalCache(SessionManager(base_path=eventloom_path))
    vault = build_vault(get_settings(), eventloom_path)
    bundle = build_memory_export(
        session_id, selector, retrieval_cache=cache, signing_key=signing_key, vault=vault
    )
    if not bundle["entries"]:
        raise typer.BadParameter(f"no matching memory to export in session {session_id}")
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    if signing_key is None:
        typer.echo(f"exported {len(bundle['entries'])} entries -> {out} (unsigned)")
    else:
        typer.echo(
            f"exported {len(bundle['entries'])} entries -> {out} "
            f"(alg={algorithm}, root={bundle['merkle_root'][:16]})"
        )


@app.command("verify-export")
def verify_export_cmd(
    bundle: Path = typer.Argument(..., help="signed bundle JSON file"),  # noqa: B008
    expect_public_key: str | None = typer.Option(None, "--expect-public-key", help="pin: hex public key the bundle must be signed by"),  # noqa: B008
) -> None:
    """Verify a signed export bundle (integrity + authenticity); exit 1 on failure."""
    from zaxy.portable import verify_export

    result = verify_export(
        json.loads(bundle.read_text(encoding="utf-8")), expect_public_key=expect_public_key
    )
    suffix = f" - {result['reason']}" if result.get("reason") else ""
    typer.echo(f"verify: {'OK' if result['ok'] else 'FAIL'}{suffix}")
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command("export-disclose")
def export_disclose(
    bundle: Path = typer.Argument(..., help="signed bundle JSON file"),  # noqa: B008
    out: Path = typer.Option(..., "--out", help="Write the disclosed subset JSON here"),  # noqa: B008
    grains: str = typer.Option("event,semantic", "--grains", help="comma-separated grains to disclose"),
    kinds: str = typer.Option("", "--kinds", help="comma-separated entry kinds to disclose (event type, or entity:/edge: kind)"),
    since_seq: int | None = typer.Option(None, "--since", help="exclusive lower bound: disclose entries with seq > this"),
    max_seq: int | None = typer.Option(None, "--max-seq", help="inclusive upper bound on seq"),
    since_time: str | None = typer.Option(None, "--since-time", help="inclusive ISO-8601 lower time bound"),
    until_time: str | None = typer.Option(None, "--until-time", help="inclusive ISO-8601 upper time bound"),
) -> None:
    """Disclose a verifiable subset of a signed export bundle, selected by predicate.

    Reveals only entries matching the selector (with Merkle inclusion proofs);
    undisclosed entries stay hidden. Verify the output with verify-export-subset.
    """
    from zaxy.export_view import ExportSelector, disclose_export_bundle

    selector = ExportSelector(
        grains=frozenset(g.strip() for g in grains.split(",") if g.strip()),
        kinds=frozenset(k.strip() for k in kinds.split(",") if k.strip()) or None,
        since_seq=since_seq,
        max_seq=max_seq,
        since_time=since_time,
        until_time=until_time,
    )
    signed = json.loads(bundle.read_text(encoding="utf-8"))
    subset = disclose_export_bundle(signed, selector)
    out.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(
        f"disclosed {len(subset['disclosed'])} of {len(signed['entries'])} entries -> {out}"
    )


@app.command("verify-export-subset")
def verify_export_subset_cmd(
    subset: Path = typer.Argument(..., help="disclosed subset JSON file"),  # noqa: B008
    expect_public_key: str | None = typer.Option(None, "--expect-public-key", help="pin: hex public key the subset must be signed by"),  # noqa: B008
) -> None:
    """Verify a disclosed subset (signature + each entry's inclusion proof); exit 1 on failure."""
    from zaxy.export_view import verify_memory_export_subset

    result = verify_memory_export_subset(
        json.loads(subset.read_text(encoding="utf-8")), expect_public_key=expect_public_key
    )
    suffix = f" - {result['reason']}" if result.get("reason") else ""
    typer.echo(f"verify-subset: {'OK' if result['ok'] else 'FAIL'}{suffix}")
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command("export-push")
def export_push(
    sink: str = typer.Option(..., "--sink", help="destination type: file or webhook"),
    dest: str = typer.Option(..., "--dest", help="file path (file sink) or http(s) URL (webhook sink)"),
    eventloom_path: str = typer.Option(".eventloom", "--eventloom-path"),
    session_id: str = typer.Option("default", "--session-id"),
    types: str = typer.Option("decision.made,goal.created,task.completed", "--types", help="comma-separated event types (empty for all)"),
    grains: str = typer.Option("event", "--grains", help="comma-separated grains: event, semantic"),
    since_seq: int | None = typer.Option(None, "--since", help="exclusive delta cursor: export events with seq > this"),
    max_seq: int | None = typer.Option(None, "--max-seq", help="inclusive upper bound on seq"),
    since_time: str | None = typer.Option(None, "--since-time", help="inclusive ISO-8601 lower time bound"),
    until_time: str | None = typer.Option(None, "--until-time", help="inclusive ISO-8601 upper time bound"),
    query: str | None = typer.Option(None, "--query", help="lexical pre-filter via the verbatim index"),
    exclude_sensitivities: str = typer.Option("", "--exclude-sensitivities", help="comma-separated sensitivity tiers to drop"),
    limit: int | None = typer.Option(None, "--limit", help="cap to the most recent N matching events"),
    private_key: Path | None = typer.Option(None, "--private-key", help="PKCS8 PEM private key file (omit for an unsigned bundle)"),  # noqa: B008
    public_key: Path | None = typer.Option(None, "--public-key", help="hex public key file (omit for an unsigned bundle)"),  # noqa: B008
    algorithm: str = typer.Option("ml-dsa-65", "--algorithm", help="signature algorithm of the key"),
    auth_token_file: Path | None = typer.Option(None, "--auth-token-file", help="file with the webhook bearer token"),  # noqa: B008
) -> None:
    """Build an export bundle and push it to a sink (one-shot; cron for recurring).

    Uses the same shared export path as `zaxy export`, so a pushed bundle is
    identical to the same export pulled. Signs when key files are supplied.
    """
    from zaxy.config import get_settings
    from zaxy.export_sinks import FileSink, WebhookSink, push_memory_export
    from zaxy.export_view import ExportSelector, load_signing_key
    from zaxy.forgetting import build_vault
    from zaxy.retrieval_cache import SessionRetrievalCache
    from zaxy.session import SessionManager

    selector = ExportSelector(
        grains=frozenset(g.strip() for g in grains.split(",") if g.strip()),
        kinds=frozenset(t.strip() for t in types.split(",") if t.strip()) or None,
        since_seq=since_seq,
        max_seq=max_seq,
        since_time=since_time,
        until_time=until_time,
        query=query,
        exclude_sensitivities=frozenset(
            s.strip() for s in exclude_sensitivities.split(",") if s.strip()
        ),
        limit=limit,
    )

    signing_key = None
    if private_key is not None or public_key is not None:
        if private_key is None or public_key is None:
            raise typer.BadParameter("signing requires both --private-key and --public-key")
        signing_key = load_signing_key(
            private_key_path=private_key, public_key_path=public_key, algorithm=algorithm
        )

    sink_impl: FileSink | WebhookSink
    if sink == "file":
        sink_impl = FileSink(dest)
    elif sink == "webhook":
        token = auth_token_file.read_text(encoding="utf-8").strip() if auth_token_file else None
        sink_impl = WebhookSink(dest, token=token)
    else:
        raise typer.BadParameter("--sink must be 'file' or 'webhook'")

    cache = SessionRetrievalCache(SessionManager(base_path=eventloom_path))
    vault = build_vault(get_settings(), eventloom_path)
    bundle = push_memory_export(
        session_id,
        selector,
        retrieval_cache=cache,
        signing_key=signing_key,
        vault=vault,
        sink=sink_impl,
    )
    signed = "signed" if signing_key is not None else "unsigned"
    typer.echo(f"pushed {len(bundle['entries'])} entries ({signed}) to {sink} {dest}")


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
    repair: bool = typer.Option(
        False,
        "--repair",
        help="Reap a broken embedded MCP owner (lock held but no healthy socket) for this workspace",
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
        update: dict[str, str] = {"eventloom_path": eventloom_path}
        # Keep the embedded store (and thus the owner-runtime lock, which keys on
        # the store) aligned with the overridden eventloom path, mirroring serve.
        if settings.embedded_graph_path == ".eventloom/projections/embedded.kuzu":
            update["embedded_graph_path"] = str(
                Path(eventloom_path) / "projections" / "embedded.kuzu"
            )
        settings = settings.model_copy(update=update)
    report = run_doctor(settings=settings, repair=repair)
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


@memory_app.command("re-embed")
def memory_re_embed(
    session_id: str = typer.Option("default", "--session-id", "--session", help="Session whose projected vectors should be re-embedded"),  # noqa: B008
    eventloom_path: str | None = typer.Option(None, help="Eventloom directory whose embedded projection should be migrated"),  # noqa: B008
    embedded_graph_path: Path | None = typer.Option(None, help="Embedded graph projection path override"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Re-embed projected vectors onto the active embedding provider version.

    Stale-version vectors never match searches embedded with the active
    provider; this batch migration upserts them to the active version tag.
    Eventloom events are never rewritten — only projection state changes.
    """
    import asyncio

    from zaxy.config import get_settings
    from zaxy.embedding import build_embedding_provider, provider_version_tag
    from zaxy.retrieval_profile import apply_retrieval_profile, resolve_retrieval_profile

    settings = get_settings()
    settings = apply_retrieval_profile(settings, resolve_retrieval_profile(settings))
    backend = settings.projection_backend.casefold().strip()
    if embedded_graph_path is None and backend != "embedded":
        raise typer.BadParameter(
            f"memory re-embed supports the embedded projection backend, not {backend}; "
            "pass --embedded-graph-path to migrate a specific embedded projection"
        )
    try:
        provider = build_embedding_provider(settings)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if provider is None:
        raise typer.BadParameter("embeddings are disabled (EMBEDDING_ENABLED=false)")
    version_tag = provider_version_tag(provider)
    if version_tag is None:
        raise typer.BadParameter(
            f"embedding provider {settings.embedding_provider} does not expose a version tag"
        )
    if embedded_graph_path is not None:
        projection_path = embedded_graph_path
    elif eventloom_path is not None:
        projection_path = Path(eventloom_path) / "projections" / "embedded.kuzu"
    else:
        projection_path = Path(settings.embedded_graph_path)
    if not projection_path.exists():
        raise typer.BadParameter(f"no embedded projection found at {projection_path}")

    async def _run() -> dict[str, int]:
        store = _runtime.EmbeddedGraphStore(projection_path)
        await store.connect()
        try:
            report = await store.re_embed_session(
                session_id=session_id,
                provider=provider,
                version_tag=version_tag,
            )
            return cast(dict[str, int], report)
        finally:
            await store.close()

    try:
        report = asyncio.run(_run())
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "version_tag": version_tag,
            "projection_path": str(projection_path),
            **report,
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(
        f"Re-embedded {report['re_embedded']} of {report['scanned']} projected vectors "
        f"in session {session_id} to {version_tag} "
        f"({report['already_current']} already current)"
    )


@app.command("index-codebase")
def index_codebase(
    path: Path = typer.Argument(..., help="Repository or directory to index"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to append codebase events into"),  # noqa: B008
    max_bytes: int = typer.Option(512 * 1024, help="Maximum source file size to index"),  # noqa: B008
) -> None:
    """Append codebase file, symbol, and import mapping events."""
    import asyncio

    async def _run() -> int:
        fabric = _runtime.MemoryFabric()
        try:
            return int(await fabric.ingest_codebase(path, session_id=session_id, max_bytes=max_bytes))
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
        fabric = _runtime.MemoryFabric()
        try:
            return cast(WorkspaceProfile, await fabric.initialize_session(path, session_id=session_id))
        finally:
            await fabric.close()

    profile = asyncio.run(_run())
    typer.echo(
        f"Initialized {session_id} as {profile.workspace_type} workspace "
        f"(confidence {profile.confidence})"
    )


def _render_init_verbose_command(
    *,
    path: Path,
    eventloom_path: str | Path,
    domain: str | None,
    session_id: str | None,
    preset: str | None,
    mcp_client: str | None,
    mcp_output: Path | None,
    hook_client: str | None,
    hook_output: Path | None,
    local_profile_output: Path | None,
    infra: str,
    projection_backend: str | None,
    pggraph_dsn: str | None,
    pggraph_repo: Path | None,
    capture_mode: str,
    capture_action: str,
    packet_capture: bool,
    packet_upstream_base_url: str,
    packet_port: int,
    codex_mcp_install: str,
    codex_trusted_project: bool,
    codex_home: Path | None,
    agent_instructions: bool,
    zaxy_executable: str | None,
    force: bool,
) -> str:
    """Render a copyable command that repeats init with verbose human output."""
    args = ["zaxy", "init", str(path)]
    if str(eventloom_path) != ".eventloom":
        args.extend(["--eventloom-path", str(eventloom_path)])
    if domain is not None:
        args.extend(["--domain", domain])
    if session_id is not None:
        args.extend(["--session-id", session_id])
    if preset is not None:
        args.extend(["--preset", preset])
    if mcp_client is not None:
        args.extend(["--mcp-client", mcp_client])
    if mcp_output is not None:
        args.extend(["--mcp-output", str(mcp_output)])
    if hook_client is not None:
        args.extend(["--hook-client", hook_client])
    if hook_output is not None:
        args.extend(["--hook-output", str(hook_output)])
    if local_profile_output is not None:
        args.extend(["--local-profile-output", str(local_profile_output)])
    if infra != "none":
        args.extend(["--infra", infra])
    if projection_backend is not None:
        args.extend(["--projection-backend", projection_backend])
    if pggraph_dsn is not None:
        args.extend(["--pggraph-dsn", pggraph_dsn])
    if pggraph_repo is not None:
        args.extend(["--pggraph-repo", str(pggraph_repo)])
    if capture_mode != "deterministic":
        args.extend(["--capture-mode", capture_mode])
    if capture_action != "none":
        args.extend(["--capture", capture_action])
    if packet_capture:
        args.append("--packet-capture")
    if packet_upstream_base_url != "https://api.openai.com/v1":
        args.extend(["--packet-upstream-base-url", packet_upstream_base_url])
    if packet_port != 8787:
        args.extend(["--packet-port", str(packet_port)])
    if codex_mcp_install != "auto":
        args.extend(["--codex-mcp-install", codex_mcp_install])
    if codex_trusted_project:
        args.append("--codex-trusted-project")
    if codex_home is not None:
        args.extend(["--codex-home", str(codex_home)])
    if not agent_instructions:
        args.append("--no-agent-instructions")
    if zaxy_executable is not None:
        args.extend(["--zaxy-executable", zaxy_executable])
    if force:
        args.append("--force")
    args.append("--verbose")
    return shlex.join(args)


def _codex_user_config_path_for_cli(codex_home: Path | None) -> Path:
    if codex_home is not None:
        return codex_home / "config.toml"
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser() / "config.toml"


def _codex_user_config_has_zaxy_entry(config_path: Path) -> bool:
    import tomllib

    try:
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    servers = document.get("mcp_servers", {})
    return isinstance(servers, dict) and "zaxy" in servers


def _codex_user_config_accepts_auto_install(
    config_path: Path,
    *,
    zaxy_executable: str | None,
) -> bool:
    """Return whether auto mode can merge Zaxy into Codex config without overwrite."""
    import tomllib

    try:
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    servers = document.get("mcp_servers", {})
    if not isinstance(servers, dict):
        return False
    existing = servers.get("zaxy")
    if existing is None:
        return True
    if not isinstance(existing, dict):
        return False
    expected_env = {
        "LOG_LEVEL": "ERROR",
        "ZAXY_ENV": "development",
    }
    existing_env = existing.get("env", {})
    if not isinstance(existing_env, dict):
        return False
    return (
        str(existing.get("command", "")) == resolve_zaxy_executable(zaxy_executable)
        and [str(arg) for arg in existing.get("args", [])] == ["serve"]
        and {str(key): str(value) for key, value in existing_env.items()} == expected_env
        and int(existing.get("startup_timeout_sec", 0)) == 90
    )


def _resolve_cli_codex_mcp_install_mode(
    mode: str,
    *,
    mcp_client: str | None,
    codex_home: Path | None,
    zaxy_executable: str | None,
) -> str:
    """Resolve the CLI-only auto Codex MCP install mode."""
    normalized = mode.casefold().strip().replace("_", "-")
    if normalized not in {"auto", "command", "user", "project"}:
        raise ValueError("codex_mcp_install must be one of: auto, command, user, project")
    if normalized != "auto":
        return normalized
    if mcp_client is None or mcp_client.casefold().strip().replace("_", "-") != "codex":
        return "command"
    if codex_home is not None:
        config_path = codex_home / "config.toml"
        if config_path.exists() and not _codex_user_config_accepts_auto_install(
            config_path,
            zaxy_executable=zaxy_executable,
        ):
            return "command"
        return "user"
    configured_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    config_path = configured_home / "config.toml"
    if config_path.exists() and _codex_user_config_accepts_auto_install(
        config_path,
        zaxy_executable=zaxy_executable,
    ):
        return "user"
    return "command"


def _codex_auto_conflict_path(
    mode: str,
    *,
    resolved_mode: str,
    mcp_client: str | None,
    codex_home: Path | None,
    zaxy_executable: str | None,
) -> Path | None:
    """Return the Codex config path when auto mode found an existing conflicting zaxy entry."""
    normalized = mode.casefold().strip().replace("_", "-")
    if normalized != "auto" or resolved_mode != "command":
        return None
    if mcp_client is None or mcp_client.casefold().strip().replace("_", "-") != "codex":
        return None
    config_path = _codex_user_config_path_for_cli(codex_home)
    if not config_path.exists():
        return None
    if _codex_user_config_accepts_auto_install(config_path, zaxy_executable=zaxy_executable):
        return None
    if not _codex_user_config_has_zaxy_entry(config_path):
        return None
    return config_path


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
    codex_mcp_install: str = typer.Option("auto", help="Codex MCP install mode: auto, command, user, or project"),  # noqa: B008
    codex_trusted_project: bool = typer.Option(False, "--codex-trusted-project", help="Allow project-scoped Codex MCP config writes"),  # noqa: B008
    codex_home: Path | None = typer.Option(None, help="CODEX_HOME override for user-scoped Codex MCP config"),  # noqa: B008
    agent_instructions: bool = typer.Option(
        True,
        "--agent-instructions/--no-agent-instructions",
        help="Install bounded Zaxy activation instructions into AGENTS.md",
    ),
    zaxy_executable: str | None = typer.Option(None, help="Executable path MCP clients should invoke"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite generated output files"),  # noqa: B008
    verbose: bool = typer.Option(False, "--verbose", help="Print full setup diagnostics in human output"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),  # noqa: B008
) -> None:
    """Bare zaxy init uses the local embedded Codex path for MCP config, infra, and hook status."""
    import asyncio

    verbose_codex_home = codex_home

    async def _run() -> Any:
        nonlocal verbose_codex_home
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
        resolved_codex_mcp_install = _resolve_cli_codex_mcp_install_mode(
            codex_mcp_install,
            mcp_client=preset_options["mcp_client"],
            codex_home=codex_home,
            zaxy_executable=zaxy_executable,
        )
        codex_mcp_conflict_path = _codex_auto_conflict_path(
            codex_mcp_install,
            resolved_mode=resolved_codex_mcp_install,
            mcp_client=preset_options["mcp_client"],
            codex_home=codex_home,
            zaxy_executable=zaxy_executable,
        )
        if (
            (resolved_codex_mcp_install == "user" or codex_mcp_conflict_path is not None)
            and verbose_codex_home is None
            and "CODEX_HOME" in os.environ
        ):
            verbose_codex_home = Path(os.environ["CODEX_HOME"])
        return await _runtime.run_onboarding(
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
            codex_mcp_install=resolved_codex_mcp_install,
            codex_mcp_conflict_path=codex_mcp_conflict_path,
            codex_trusted_project=codex_trusted_project,
            codex_home=codex_home,
            agent_instructions=agent_instructions,
            zaxy_executable=zaxy_executable,
            force=force,
        )

    try:
        result = asyncio.run(_run())
    except (FileExistsError, PermissionError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(json.dumps(onboarding_result_payload(result), indent=2, sort_keys=True))
    else:
        verbose_command = _render_init_verbose_command(
            path=path,
            eventloom_path=eventloom_path,
            domain=domain,
            session_id=session_id,
            preset=preset,
            mcp_client=mcp_client,
            mcp_output=mcp_output,
            hook_client=hook_client,
            hook_output=hook_output,
            local_profile_output=local_profile_output,
            infra=infra,
            projection_backend=projection_backend,
            pggraph_dsn=pggraph_dsn,
            pggraph_repo=pggraph_repo,
            capture_mode=capture_mode,
            capture_action=capture_action,
            packet_capture=packet_capture,
            packet_upstream_base_url=packet_upstream_base_url,
            packet_port=packet_port,
            codex_mcp_install=codex_mcp_install,
            codex_trusted_project=codex_trusted_project,
            codex_home=verbose_codex_home,
            agent_instructions=agent_instructions,
            zaxy_executable=zaxy_executable,
            force=force,
        )
        typer.echo(format_onboarding_result(result, verbose=verbose, verbose_command=verbose_command))


@app.command("viewer")
def viewer(
    path: Path = typer.Argument(..., help="Eventloom JSONL log or directory to inspect"),  # noqa: B008
    output: Path = typer.Option("eventloom-viewer.html", "--output", "-o", help="HTML output path"),  # noqa: B008
) -> None:
    """Write a standalone HTML viewer for Eventloom sessions."""
    from zaxy.viewer import write_viewer_html

    written = write_viewer_html(path, output)
    typer.echo(f"Wrote Eventloom viewer: {written}")


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
    integrity = result.integrity
    if integrity is None:
        raise RuntimeError("CLI replay requires integrity verification")

    if json_output:
        output = {
            "from_seq": from_seq,
            "to_seq": to_seq,
            "integrity": integrity.model_dump(),
            "events": [e.model_dump() for e in result.events],
        }
        print(json.dumps(output, indent=2))
        return

    typer.echo(f"Integrity: {'OK' if integrity.ok else 'FAILED'}")
    typer.echo(f"Total events: {integrity.total_events}")
    window = f"{from_seq}.." + (str(to_seq) if to_seq is not None else "HEAD")
    typer.echo(f"Replay window: {window}")
    if integrity.broken_at_seq:
        typer.echo(f"Broken at seq: {integrity.broken_at_seq}")
        typer.echo(f"Reason: {integrity.broken_reason}")

    for ev in result.events:
        typer.echo(f"  [{ev.seq}] {ev.timestamp} {ev.type} by {ev.actor}")

    summary = log.handoff_summary()
    typer.echo("\nHandoff summary:")
    typer.echo(f"  Goals: {summary['goals']}")
    typer.echo(f"  Open tasks: {len(summary['open_tasks'])}")
    typer.echo(f"  Last actor: {summary['last_actor']}")
