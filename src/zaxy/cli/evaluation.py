"""Split from cli.py (mechanical decomposition)."""


from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import typer

from zaxy.cli import runtime as _runtime
from zaxy.cli.runtime import (
    _benchmark_module,
    _coordinate_evidence_item,
    _format_float_value,
    _format_int_value,
    _format_memory_graph_status,
    _format_optional_float,
    _format_reasoning_result_text,
    _validate_reasoning_phase_option,
    app,
    coordinate_app,
    memory_app,
    memory_consolidation_app,
    memory_reasoning_app,
)
from zaxy.cli.serving import (
    _append_consolidation_event,
    _format_status_memory_activation,
    _profile_root_for_eventloom_path,
    _run_reasoning_primitive,
    _status_settings,
)
from zaxy.cli.workspace import (
    _checkout_token_guardrail,
    _resolve_cli_projection_backend,
    _shell_join,
)


@app.command("longmembench-validator-evidence")
def longmembench_validator_evidence(
    longmemeval_worktree: Path = typer.Option(  # noqa: B008
        ...,
        "--longmemeval-worktree",
        help="External official LongMemEval worktree",
    ),
    dataset: Path = typer.Option(  # noqa: B008
        ...,
        "--dataset",
        help="Official LongMemEval dataset path",
    ),
    hypotheses: Path = typer.Option(  # noqa: B008
        ...,
        "--hypotheses",
        help="Generated official hypothesis JSONL",
    ),
    official_eval_log: Path = typer.Option(  # noqa: B008
        ...,
        "--official-eval-log",
        help="Official LongMemEval evaluate_qa.py JSONL log",
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/validator-evidence.json"),
        "--output",
        help="Completed validator evidence JSON path",
    ),
    evaluator_model: str = typer.Option(
        "gpt-4o",
        "--evaluator-model",
        help="Official evaluator model used by evaluate_qa.py",
    ),
    official_eval_command: str = typer.Option(
        ...,
        "--official-eval-command",
        help="Exact official evaluate_qa.py command used by the validator",
    ),
    print_metrics_command: str | None = typer.Option(
        None,
        "--print-metrics-command",
        help="Exact official print_qa_metrics.py command used by the validator",
    ),
    validator_name: str = typer.Option(
        ...,
        "--validator-name",
        help="Independent validator name",
    ),
    validator_evidence_url: str = typer.Option(
        ...,
        "--validator-evidence-url",
        help="Reviewable external validation URL",
    ),
    validator_run_id: str = typer.Option(
        ...,
        "--validator-run-id",
        help="Independent validator run identifier",
    ),
    validator_relation: str = typer.Option(
        ...,
        "--validator-relation",
        help="Relationship to Zaxy, for example independent-third-party",
    ),
    zaxy_worktree: Path = typer.Option(  # noqa: B008
        Path("."),
        "--zaxy-worktree",
        help="Zaxy source checkout validated by the run",
    ),
) -> None:
    """Write completed validator evidence from official LongMemEval artifacts."""
    longmembench_module = _benchmark_module("longmembench")
    build_validator_evidence_record = longmembench_module.build_validator_evidence_record
    write_validator_evidence_record = longmembench_module.write_validator_evidence_record
    from zaxy.release import package_version

    try:
        record = build_validator_evidence_record(
            longmemeval_worktree=longmemeval_worktree,
            dataset_path=dataset,
            hypotheses_path=hypotheses,
            official_eval_log_path=official_eval_log,
            evaluator_model=evaluator_model,
            official_eval_command=official_eval_command,
            print_metrics_command=print_metrics_command,
            validator_name=validator_name,
            validator_evidence_url=validator_evidence_url,
            validator_run_id=validator_run_id,
            validator_relation=validator_relation,
            zaxy_worktree=zaxy_worktree,
            zaxy_version=package_version(),
            longmembench_report_json=output.parent / "longmembench-report.json",
            longmembench_report_md=output.parent / "longmembench-report.md",
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    written = write_validator_evidence_record(record, output)
    typer.echo(json.dumps(record, indent=2, sort_keys=True))
    typer.echo(f"Wrote validator evidence: {written}")


@app.command("longmembench-validate")
def longmembench_validate(
    report_path: Path = typer.Argument(..., help="longmembench-report.json report"),  # noqa: B008
    require_official_full: bool = typer.Option(
        False,
        "--require-official-full",
        help="Require official evaluator evidence over all 500 questions",
    ),
) -> None:
    """Validate LongMemBench external validation evidence."""
    longmembench_module = _benchmark_module("longmembench")
    load_longmembench_report = longmembench_module.load_longmembench_report
    validate_longmembench_report = longmembench_module.validate_longmembench_report

    try:
        report = load_longmembench_report(report_path)
        validation = validate_longmembench_report(
            report,
            require_official_full=require_official_full,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(validation, indent=2, sort_keys=True))
    if validation["status"] != "valid":
        raise typer.Exit(1)


@app.command("longmembench-gate")
def longmembench_gate(
    report_path: Path = typer.Argument(..., help="longmembench-report.json report"),  # noqa: B008
    require_official_sota_candidate: bool = typer.Option(
        False,
        "--require-official-sota-candidate",
        help="Require full official LongMemEval QA evidence before SOTA-candidate claims",
    ),
    require_official_sota: bool = typer.Option(
        False,
        "--require-official-sota",
        help="Require full official QA evidence and a beaten external SOTA baseline",
    ),
    require_external_validator: bool = typer.Option(
        False,
        "--require-external-validator",
        help="Require independent validator provenance",
    ),
    min_accuracy: float | None = typer.Option(  # noqa: B008
        None,
        "--min-accuracy",
        min=0.0,
        max=1.0,
        help="Optional minimum official QA accuracy",
    ),
) -> None:
    """Gate publishable LongMemBench and official SOTA-candidate claims."""
    longmembench_module = _benchmark_module("longmembench")
    check_longmembench_gate = longmembench_module.check_longmembench_gate
    load_longmembench_report = longmembench_module.load_longmembench_report

    try:
        report = load_longmembench_report(report_path)
        gate = check_longmembench_gate(
            report,
            require_official_sota_candidate=require_official_sota_candidate,
            require_official_sota=require_official_sota,
            require_external_validator=require_external_validator,
            min_accuracy=min_accuracy,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(gate, indent=2, sort_keys=True))
    if gate["status"] != "passed":
        raise typer.Exit(1)


@app.command("longmembench-audit")
def longmembench_audit(
    longmemeval_worktree: Path = typer.Option(  # noqa: B008
        ...,
        "--longmemeval-worktree",
        help="External official LongMemEval worktree",
    ),
    dataset: Path = typer.Option(  # noqa: B008
        ...,
        "--dataset",
        help="Official LongMemEval dataset path",
    ),
    hypotheses: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl"),
        "--hypotheses",
        help="Generated official hypothesis JSONL",
    ),
    official_eval_log: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o"),
        "--official-eval-log",
        help="Official LongMemEval evaluate_qa.py JSONL log",
    ),
    diagnostic_report: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/diagnostic/live-benchmark.json"),
        "--diagnostic-report",
        help="Zaxy LongMemEval-compatible live-benchmark.json report",
    ),
    sota_baseline: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/sota-baseline.json"),
        "--sota-baseline",
        help="External SOTA baseline JSON",
    ),
    validator_evidence: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/validator-evidence.json"),
        "--validator-evidence",
        help="Cross-checked validator evidence JSON",
    ),
    report_path: Path = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/longmembench-report.json"),
        "--report",
        help="Imported longmembench-report.json",
    ),
    hypothesis_report: Path | None = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/zaxy-hypotheses-report.json"),
        "--hypothesis-report",
        help="Optional hypothesis generation report JSON",
    ),
    official_eval_run_report: Path | None = typer.Option(  # noqa: B008
        Path("reports/benchmarks/longmembench-external/official-eval-run.json"),
        "--official-eval-run-report",
        help="Optional official evaluator subprocess report JSON",
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        help="Optional path to write the audit JSON result",
    ),
) -> None:
    """Audit a completed external LongMemBench artifact set."""
    longmembench_module = _benchmark_module("longmembench")
    audit_longmembench_artifacts = longmembench_module.audit_longmembench_artifacts

    audit = audit_longmembench_artifacts(
        longmemeval_worktree=longmemeval_worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=official_eval_log,
        diagnostic_report_path=diagnostic_report,
        sota_baseline_path=sota_baseline,
        validator_evidence_path=validator_evidence,
        report_path=report_path,
        hypothesis_report_path=hypothesis_report,
        official_eval_run_report_path=official_eval_run_report,
    )
    audit_json = json.dumps(audit, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(audit_json + "\n", encoding="utf-8")
    typer.echo(audit_json)
    if audit["status"] != "passed":
        raise typer.Exit(1)


@app.command("longmembench-publish")
def longmembench_publish(
    report_path: Path = typer.Argument(..., help="longmembench-report.json report"),  # noqa: B008
    audit: Path | None = typer.Option(  # noqa: B008
        None,
        "--audit",
        help="Passing longmembench-audit.json artifact; defaults to report directory",
    ),
    allow_unaudited: bool = typer.Option(
        False,
        "--allow-unaudited",
        help="Render draft output without requiring a passing audit artifact",
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        help="Optional Markdown output path for publishable LongMemBench statistics",
    ),
) -> None:
    """Render publishable LongMemBench statistics after the strict gate and audit pass."""
    longmembench_module = _benchmark_module("longmembench")
    load_longmembench_report = longmembench_module.load_longmembench_report
    render_longmembench_publication_markdown = (
        longmembench_module.render_longmembench_publication_markdown
    )
    validate_longmembench_audit_for_report = (
        longmembench_module.validate_longmembench_audit_for_report
    )

    try:
        if not allow_unaudited:
            audit_path = audit or report_path.parent / "longmembench-audit.json"
            validate_longmembench_audit_for_report(audit_path, report_path)
        report = load_longmembench_report(report_path)
        markdown = render_longmembench_publication_markdown(report)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if output is None:
        typer.echo(markdown)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    typer.echo(f"Wrote LongMemBench publishable statistics: {output}")


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
    live_benchmark_module = _benchmark_module("live_benchmark")
    build_mempalace_workload_inventory = live_benchmark_module.build_mempalace_workload_inventory
    format_mempalace_workload_inventory = live_benchmark_module.format_mempalace_workload_inventory

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
    purpose_benchmark_module = _benchmark_module("purpose_benchmark")
    run_purpose_benchmark = purpose_benchmark_module.run_purpose_benchmark
    write_purpose_benchmark_report = purpose_benchmark_module.write_purpose_benchmark_report

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
    live_benchmark_module = _benchmark_module("live_benchmark")
    compare_benchmark_reports = live_benchmark_module.compare_benchmark_reports
    format_benchmark_comparison = live_benchmark_module.format_benchmark_comparison
    load_benchmark_report = live_benchmark_module.load_benchmark_report

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


@app.command("benchmark-freeze")
def benchmark_freeze(
    root: Path = typer.Option(Path("."), "--root", help="Repository root containing frozen reports"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Validate the 2.0 RC.1 benchmark-freeze evidence contract."""
    rc_benchmark_freeze_module = _benchmark_module("rc_benchmark_freeze")
    build_rc1_benchmark_freeze_report = (
        rc_benchmark_freeze_module.build_rc1_benchmark_freeze_report
    )
    format_rc1_benchmark_freeze_report = (
        rc_benchmark_freeze_module.format_rc1_benchmark_freeze_report
    )

    report = build_rc1_benchmark_freeze_report(root)
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(format_rc1_benchmark_freeze_report(report), nl=False)
    if not report.passed:
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
            embedded_runtime = _runtime.LocalEmbeddedGraphRuntime(path=embedded_graph_path or Path(settings.embedded_graph_path))
            check = embedded_runtime.check()
            typer.echo(f"embedded graph: {check.status.upper()} ({check.message})")
            if check.status == "error":
                ok = False
        elif backend == "pggraph":
            pggraph_runtime = _runtime.LocalPgGraphRuntime(
                dsn=pggraph_dsn or settings.pggraph_dsn,
                enabled=settings.pggraph_auto_start and settings.zaxy_env.lower() != "production",
            )
            check = pggraph_runtime.check()
            typer.echo(f"pgGraph: {check.status.upper()} ({check.message})")
            if check.status == "error":
                ok = False
        elif backend == "neo4j":
            try:
                gs = _runtime.GraphStore(_uri, _user, _password)
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
    from zaxy.coordination_git import (
        GitCaptureError,
        build_test_result_evidence,
        capture_git_metadata,
    )

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
        try:
            evidence_items.append(capture_git_metadata(git_target, test_results=test_results))
        except GitCaptureError as exc:
            raise typer.BadParameter(str(exc)) from exc
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


@memory_reasoning_app.command("known-unknowns")
def memory_reasoning_known_unknowns(
    status: str = typer.Option("open", help="Known-unknown status filter or all"),
    phase: str = typer.Option(  # noqa: B008
        "review",
        callback=_validate_reasoning_phase_option,
        help="Accepted for reasoning command consistency; list queries are replay-derived",
    ),
    session_id: str = typer.Option("default", help="Session ID to query"),
    limit: int = typer.Option(10, min=1, help="Maximum known unknowns"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List replay-derived known unknowns."""
    import asyncio

    _ = phase
    result = asyncio.run(
        _run_reasoning_primitive(
            method_name="list_known_unknowns",
            eventloom_path=eventloom_path,
            args=(),
            kwargs={"session_id": session_id, "status": status, "limit": limit},
        )
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True) if json_output else _format_reasoning_result_text(result))


@memory_consolidation_app.command("review")
def memory_consolidation_review(
    candidate_id: str = typer.Option(..., help="Consolidation candidate ID"),
    status: str = typer.Option(..., help="Review status: accepted, rejected, deferred, or conflicted"),
    rationale: str = typer.Option(..., help="Review rationale"),
    actor: str = typer.Option("zaxy", help="Actor writing the event"),
    session_id: str = typer.Option("default", help="Session ID to append to"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Append a consolidation candidate review event."""
    import asyncio

    from zaxy.consolidation import build_consolidation_review_event

    try:
        event = build_consolidation_review_event(
            actor=actor,
            session_id=session_id,
            candidate_id=candidate_id,
            status=status,
            rationale=rationale,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    asyncio.run(_append_consolidation_event(event, eventloom_path=eventloom_path))
    if json_output:
        typer.echo(json.dumps(event, indent=2, sort_keys=True))
    else:
        typer.echo(f"Reviewed {candidate_id} as {status}")


def _capture_runtime_guardrail(
    report: dict[str, object],
    *,
    required: bool,
    workspace_root: Path,
) -> dict[str, object] | None:
    if not required:
        return None
    clients = report.get("clients")
    codex = clients.get("codex") if isinstance(clients, dict) else None
    if not isinstance(codex, dict) or not codex.get("installed", False):
        return {
            "status": "ok",
            "required": True,
            "configured": False,
            "running": False,
            "message": "Codex capture config is not installed; no configured watcher is required",
        }
    runtime = codex.get("runtime")
    running = bool(runtime.get("running", False)) if isinstance(runtime, dict) else False
    status = "ok" if running else "fail"
    return {
        "status": status,
        "required": True,
        "configured": True,
        "running": running,
        "message": (
            "Codex capture watcher is running"
            if running
            else "Codex capture config is installed, but the managed watcher is not running"
        ),
        "action": None if running else f"zaxy capture start --workspace {workspace_root}",
    }


def _format_capture_runtime_guardrail(guardrail: dict[str, object]) -> str:
    status = str(guardrail["status"]).upper()
    message = str(guardrail["message"])
    action = guardrail.get("action")
    if action:
        return f"Capture runtime guardrail: {status} ({message}; action: {action})"
    return f"Capture runtime guardrail: {status} ({message})"


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
    require_capture_running: bool = typer.Option(
        False,
        "--require-capture-running",
        help="Fail when installed Codex capture config does not have a running watcher",
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
    capture_guardrail = _capture_runtime_guardrail(
        report,
        required=require_capture_running,
        workspace_root=workspace_root,
    )
    if capture_guardrail is not None:
        report["capture_runtime_guardrail"] = capture_guardrail
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(format_hook_status(report))
        if guardrail is not None:
            typer.echo(_format_activation_guardrail(guardrail))
        if token_guardrail is not None:
            typer.echo(_format_checkout_token_guardrail(token_guardrail))
        if capture_guardrail is not None:
            typer.echo(_format_capture_runtime_guardrail(capture_guardrail))
    if (guardrail is not None and guardrail["status"] != "ok") or (
        token_guardrail is not None and token_guardrail["status"] != "ok"
    ) or (
        capture_guardrail is not None and capture_guardrail["status"] != "ok"
    ):
        raise typer.Exit(1)


@app.command("harvey-lab-doctor")
def harvey_lab_doctor(
    harvey_worktree: Path = typer.Argument(  # noqa: B008
        ...,
        help="External Harvey worktree to validate",
    ),
) -> None:
    """Validate that a Harvey checkout matches the external article suite."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    check_harvey_external_suite = harvey_module.check_harvey_external_suite

    status = check_harvey_external_suite(harvey_worktree)
    typer.echo(json.dumps(status, indent=2, sort_keys=True))
    if status["status"] != "valid":
        raise typer.Exit(1)


@app.command("harvey-lab-preflight")
def harvey_lab_preflight(
    harvey_worktree: Path = typer.Argument(  # noqa: B008
        ...,
        help="External Harvey worktree to normalize and index before model runs",
    ),
    max_lines: int = typer.Option(80, "--max-lines", min=1, help="Lines per indexed document chunk"),
    task_filter: str | None = typer.Option(
        None,
        "--task-filter",
        help="Optional Harvey task id, slug, or Zaxy run id for single-task preflight",
    ),
) -> None:
    """Normalize and Zaxy-index pinned Harvey LAB tasks without scoring."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    build_harvey_external_index_preflight = harvey_module.build_harvey_external_index_preflight

    try:
        status = build_harvey_external_index_preflight(
            harvey_worktree,
            max_lines=max_lines,
            task_filter=task_filter,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(status, indent=2, sort_keys=True))
    if status["status"] != "ready_for_external_runs":
        raise typer.Exit(1)


@app.command("harvey-lab-status")
def harvey_lab_status(
    harvey_worktree: Path = typer.Argument(  # noqa: B008
        ...,
        help="External Harvey worktree to scan for Zaxy run artifacts",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Report per-task readiness for the external Harvey Zaxy run pipeline."""
    harvey_module = _benchmark_module("harvey_lab_benchmark")
    build_harvey_external_run_status = harvey_module.build_harvey_external_run_status

    _ = json_output
    status = build_harvey_external_run_status(harvey_worktree)
    typer.echo(json.dumps(status, indent=2, sort_keys=True))
    if status["status"] != "complete":
        raise typer.Exit(1)


@app.command("longmembench-doctor")
def longmembench_doctor(
    longmemeval_worktree: Path = typer.Argument(  # noqa: B008
        ...,
        help="External official LongMemEval worktree to validate",
    ),
) -> None:
    """Validate an official LongMemEval checkout for external LongMemBench runs."""
    longmembench_module = _benchmark_module("longmembench")
    check_longmemeval_official_suite = longmembench_module.check_longmemeval_official_suite

    status = check_longmemeval_official_suite(longmemeval_worktree)
    typer.echo(json.dumps(status, indent=2, sort_keys=True))
    if status["status"] != "valid":
        raise typer.Exit(1)


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
            f"Capture action: {cast(dict[str, object], packet['capture_start']).get('status', 'unknown')}",
            str(cast(dict[str, object], packet["capture_start"]).get("message", "")),
            "",
            str(packet["next_step"]),
        ]
    )


def _activation_injection_text(bootstrap_prompt: str, capture_start: dict[str, object]) -> str:
    status = str(capture_start.get("status", "unknown"))
    message = str(capture_start.get("message", ""))
    action = capture_start.get("action")
    lines = [
        bootstrap_prompt,
        "",
        f"Capture action: {status}",
        f"- status: {status}",
        f"- message: {message}",
    ]
    if action:
        lines.append(f"- action: {action}")
    if status in {"degraded", "skipped"}:
        lines.extend(
            [
                "",
                "Degraded memory state: do not assume this Codex session is being captured until the action is completed.",
            ]
        )
    return "\n".join(lines)


def _ensure_codex_capture(*, workspace_root: Path) -> dict[str, object]:
    from zaxy.capture_manager import start_codex_capture

    try:
        result = start_codex_capture(workspace=workspace_root)
    except FileNotFoundError as exc:
        return {
            "status": "degraded",
            "reason": "not_configured",
            "message": "Managed Codex capture is not configured for this workspace.",
            "error": str(exc),
            "action": "Run zaxy init --preset local-codex or zaxy init --capture start, then restart activation.",
        }
    except (OSError, ValueError) as exc:
        return {
            "status": "degraded",
            "reason": "start_failed",
            "message": "Managed Codex capture could not be started.",
            "error": str(exc),
            "action": f"Run zaxy capture start --workspace {workspace_root.resolve()} before substantial work.",
        }
    status = "started" if result.get("started") else "running"
    return {
        "status": status,
        "reason": "started" if result.get("started") else "already_running",
        "message": str(result.get("message", "Managed Codex capture is available.")),
        "pid": result.get("pid"),
        "state_file": result.get("state_file"),
    }


def _codex_activation_command(executable: str, workspace: Path, prompt: str) -> list[str]:
    return [executable, "--cd", str(workspace), prompt]


@app.command("activate")
def activate(
    client: str = typer.Argument(..., help="Agent client to activate: codex"),  # noqa: B008
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),  # noqa: B008
    session_id: str = typer.Option("default", help="Session ID to activate"),
    current_task: str | None = typer.Option(None, help="Current task or question to seed checkout guidance"),  # noqa: B008
    workspace_root: Path = typer.Option(Path("."), help="Workspace root for capture/status discovery"),  # noqa: B008
    launch: bool = typer.Option(False, "--launch", help="Start the agent client with activation context"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the launch command without starting the client"),
    ensure_capture: bool = typer.Option(True, "--ensure-capture/--no-ensure-capture", help="Start managed Codex capture when configured"),
    codex_executable: str = typer.Option("codex", help="Codex executable for --launch"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Emit a prompt-ready memory activation packet for an agent client."""
    normalized_client = client.casefold().strip().replace("_", "-")
    if normalized_client != "codex":
        raise typer.BadParameter("activate currently supports: codex", param_hint="client")
    from zaxy.capabilities import build_memory_bootstrap

    capture_start: dict[str, object] = (
        _ensure_codex_capture(workspace_root=workspace_root)
        if ensure_capture and not dry_run
        else {
            "status": "skipped",
            "reason": "disabled" if not ensure_capture else "dry_run",
            "message": "Managed Codex capture was not started for this activation.",
            "action": "Run zaxy capture start --workspace . before substantial work.",
        }
    )
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
    packet: dict[str, object] = {
        "client": normalized_client,
        "mode": "session_start_injection",
        "session_id": session_id,
        "workspace": str(workspace_root.resolve()),
        "bootstrap": bootstrap,
        "capture_start": capture_start,
        "injection_text": _activation_injection_text(str(bootstrap["prompt"]), capture_start),
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
