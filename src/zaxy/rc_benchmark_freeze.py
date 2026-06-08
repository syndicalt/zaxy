"""Zaxy 2.0 RC.1 benchmark-freeze evidence contract.

This module validates release evidence and claim boundaries. It does not run
benchmarks, score answers, or modify retrieval behavior.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RC1_RELEASE = "2.0.0-rc.1"
RC1_MANIFEST = Path("reports/benchmarks/2.0.0-rc.1/manifest.json")
HEADLINE_REPORT = Path("reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json")
HEADLINE_RUN_CONFIG = Path("reports/benchmarks/longmemeval-500-publish-20260607/run-config.md")
HARVEY_DIR = Path("reports/benchmarks/harvey-lab-memory-ablation")
HARVEY_ARTIFACTS = (
    "harvey-lab-benchmark.json",
    "harvey-lab-external-run.json",
    "harvey-lab-ready.json",
    "harvey-lab-status.json",
)
HEADLINE_BACKEND = "zaxy-checkout"

HEADLINE_FLOORS = {
    "case_count": 500,
    "mean_score": 0.95,
    "mean_answer_recall_at_5": 0.90,
    "mean_recall_at_5": 0.99,
    "mean_citation_coverage": 1.0,
}
HEADLINE_BUDGETS = {
    "latency_ms_p95": 2500.0,
    "latency_ms_p99": 3000.0,
}

INTERNAL_LANE_SCOPE = "project_defined_internal"
EXTERNAL_ANCHOR_SCOPE = "external_anchor"
HEADLINE_SCOPE = "longmemeval_compatible_checkout"
REQUIRED_INTERNAL_LANES = ("causal", "consolidation", "procedural", "metacognition")


@dataclass(frozen=True)
class RcBenchmarkFreezeCheck:
    """One release-freeze validation result."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class RcBenchmarkFreezeReport:
    """Machine-readable 2.0 RC.1 benchmark-freeze report."""

    release: str
    passed: bool
    headline_500: dict[str, Any]
    harvey_lab: dict[str, Any]
    project_benchmarks: dict[str, Any]
    internal_lanes: list[dict[str, Any]]
    checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


def build_rc1_benchmark_freeze_report(
    root: str | Path = ".",
    *,
    internal_lane_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> RcBenchmarkFreezeReport:
    """Validate frozen RC.1 benchmark artifacts and claim boundaries."""
    root_path = Path(root)
    checks: list[RcBenchmarkFreezeCheck] = []
    git_tracking = _git_tracking_enabled(root_path)
    manifest = _manifest_evidence(root_path, checks)
    headline_500 = _headline_evidence(root_path, checks, manifest)
    harvey_lab = _harvey_evidence(root_path, checks, manifest)
    internal_lanes = _internal_lanes(manifest, internal_lane_overrides)
    checks.extend(_check_internal_lanes(internal_lanes))
    project_benchmarks = _project_benchmark_evidence(root_path, checks, manifest)
    if git_tracking:
        _check_tracked_release_artifacts(
            root_path,
            checks,
            headline_500=headline_500,
            harvey_lab=harvey_lab,
            project_benchmarks=project_benchmarks,
        )
    check_dicts = [asdict(check) for check in checks]
    return RcBenchmarkFreezeReport(
        release=RC1_RELEASE,
        passed=all(check.passed for check in checks),
        headline_500=headline_500,
        harvey_lab=harvey_lab,
        project_benchmarks=project_benchmarks,
        internal_lanes=internal_lanes,
        checks=check_dicts,
    )


def _check_tracked_release_artifacts(
    root: Path,
    checks: list[RcBenchmarkFreezeCheck],
    *,
    headline_500: Mapping[str, Any],
    harvey_lab: Mapping[str, Any],
    project_benchmarks: Mapping[str, Any],
) -> None:
    paths = {
        RC1_MANIFEST,
        Path(str(headline_500.get("artifact") or "")),
        Path(str(headline_500.get("run_config") or "")),
    }
    for artifact in harvey_lab.get("artifacts", []):
        if isinstance(artifact, Mapping):
            paths.add(Path(str(artifact.get("path") or "")))
    for benchmark in project_benchmarks.values():
        if isinstance(benchmark, Mapping):
            for key in ("artifact", "workload", "markdown", "holdout_pack", "source_disclosures"):
                value = benchmark.get(key)
                if value:
                    paths.add(Path(str(value)))
    for path in sorted(paths, key=str):
        if not str(path):
            continue
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"git_tracked:{path}",
                passed=_git_tracked(root, path),
                message=str(path),
            )
        )


def format_rc1_benchmark_freeze_report(report: RcBenchmarkFreezeReport) -> str:
    """Render a concise human-readable RC.1 freeze report."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "# Zaxy 2.0 RC.1 Benchmark Freeze",
        "",
        f"- Status: `{status}`",
        f"- Release: `{report.release}`",
        f"- Headline scope: `{report.headline_500.get('claim_scope', '')}`",
        f"- Harvey LAB scope: `{report.harvey_lab.get('claim_scope', '')}`",
        f"- Internal lanes: `{len(report.internal_lanes)}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.checks:
        check_status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- `{check['name']}`: {check_status} - {check['message']}")
    return "\n".join(lines) + "\n"


def _manifest_evidence(root: Path, checks: list[RcBenchmarkFreezeCheck]) -> Mapping[str, Any]:
    manifest_path = root / RC1_MANIFEST
    checks.append(
        RcBenchmarkFreezeCheck(
            name="rc1_manifest",
            passed=manifest_path.is_file(),
            message=str(RC1_MANIFEST),
        )
    )
    if not manifest_path.is_file():
        return {}
    try:
        manifest = _read_json(manifest_path)
    except ValueError as exc:
        checks.append(
            RcBenchmarkFreezeCheck(
                name="rc1_manifest_json",
                passed=False,
                message=str(exc),
            )
        )
        return {}
    checks.append(
        RcBenchmarkFreezeCheck(
            name="rc1_manifest_release",
            passed=manifest.get("release") == RC1_RELEASE,
            message=f"{manifest.get('release')!r} == {RC1_RELEASE}",
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="rc1_manifest_schema",
            passed=manifest.get("schema_version") == "zaxy.rc1-benchmark-freeze.v1",
            message="schema_version must be zaxy.rc1-benchmark-freeze.v1",
        )
    )
    return manifest


def _headline_evidence(
    root: Path,
    checks: list[RcBenchmarkFreezeCheck],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_headline = manifest.get("headline_500")
    manifest_headline = manifest_headline if isinstance(manifest_headline, Mapping) else {}
    manifest_artifact = Path(str(manifest_headline.get("artifact") or HEADLINE_REPORT))
    manifest_run_config = Path(str(manifest_headline.get("run_config") or HEADLINE_RUN_CONFIG))
    report_path = root / manifest_artifact
    run_config_path = root / manifest_run_config
    checks.append(
        RcBenchmarkFreezeCheck(
            name="headline_manifest_artifact",
            passed=manifest_artifact == HEADLINE_REPORT,
            message=str(manifest_artifact),
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="headline_manifest_run_config",
            passed=manifest_run_config == HEADLINE_RUN_CONFIG,
            message=str(manifest_run_config),
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="headline_manifest_scope",
            passed=manifest_headline.get("claim_scope") == HEADLINE_SCOPE,
            message=f"{manifest_headline.get('claim_scope')!r} == {HEADLINE_SCOPE}",
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="headline_report",
            passed=report_path.is_file(),
            message=str(manifest_artifact),
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="headline_run_config",
            passed=run_config_path.is_file(),
            message=str(manifest_run_config),
        )
    )
    evidence: dict[str, Any] = {
        "claim_scope": HEADLINE_SCOPE,
        "backend": HEADLINE_BACKEND,
        "artifact": str(manifest_artifact),
        "manifest": str(RC1_MANIFEST),
        "run_config": str(manifest_run_config),
        "floors": HEADLINE_FLOORS,
        "budgets": HEADLINE_BUDGETS,
    }
    if not report_path.is_file():
        return evidence

    try:
        report = _read_json(report_path)
    except ValueError as exc:
        checks.append(
            RcBenchmarkFreezeCheck(
                name="headline_report_json",
                passed=False,
                message=str(exc),
            )
        )
        return evidence
    summary = _summary_for_backend(report, HEADLINE_BACKEND)
    checks.append(
        RcBenchmarkFreezeCheck(
            name="headline_backend",
            passed=summary is not None,
            message=f"required backend={HEADLINE_BACKEND}",
        )
    )
    if summary is None:
        return evidence
    workload = report.get("workload")
    workload_hash = workload.get("sha256") if isinstance(workload, Mapping) else None
    manifest_workload_hash = manifest_headline.get("workload_sha256")
    checks.append(
        RcBenchmarkFreezeCheck(
            name="headline_workload_hash",
            passed=workload_hash == manifest_workload_hash,
            message=f"{workload_hash!r} == {manifest_workload_hash!r}",
        )
    )
    evidence.update(
        {
            "generated_at": report.get("generated_at"),
            "workload": workload,
            "metrics": {
                key: summary.get(key)
                for key in (
                    "case_count",
                    "mean_score",
                    "mean_answer_recall_at_5",
                    "mean_recall_at_5",
                    "mean_citation_coverage",
                    "latency_ms_p95",
                    "latency_ms_p99",
                    "mean_approx_tokens",
                )
            },
        }
    )
    for metric, floor in HEADLINE_FLOORS.items():
        value = summary.get(metric)
        passed = _numeric(value) >= floor
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"headline_floor:{metric}",
                passed=passed,
                message=f"{value!r} >= {floor}",
            )
        )
    for metric, budget in HEADLINE_BUDGETS.items():
        value = summary.get(metric)
        passed = _numeric(value) <= budget
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"headline_budget:{metric}",
                passed=passed,
                message=f"{value!r} <= {budget}",
            )
        )
    return evidence


def _harvey_evidence(
    root: Path,
    checks: list[RcBenchmarkFreezeCheck],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_harvey = manifest.get("harvey_lab")
    manifest_harvey = manifest_harvey if isinstance(manifest_harvey, Mapping) else {}
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_manifest_scope",
            passed=manifest_harvey.get("claim_scope") == EXTERNAL_ANCHOR_SCOPE,
            message=f"{manifest_harvey.get('claim_scope')!r} == {EXTERNAL_ANCHOR_SCOPE}",
        )
    )
    artifacts: list[dict[str, Any]] = []
    artifact_payloads: dict[str, Mapping[str, Any]] = {}
    for filename in HARVEY_ARTIFACTS:
        path = HARVEY_DIR / filename
        full_path = root / path
        exists = full_path.is_file()
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"harvey_artifact:{filename}",
                passed=exists,
                message=str(path),
            )
        )
        artifacts.append({"path": str(path), "present": exists})
        if exists:
            try:
                artifact_payloads[filename] = _read_json(full_path)
            except ValueError as exc:
                checks.append(
                    RcBenchmarkFreezeCheck(
                        name=f"harvey_json:{filename}",
                        passed=False,
                        message=str(exc),
                    )
                )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_claim_scope",
            passed=True,
            message=f"Harvey LAB classified as {EXTERNAL_ANCHOR_SCOPE}, not external validation",
        )
    )
    _check_harvey_payloads(artifact_payloads, manifest_harvey, checks)
    return {
        "claim_scope": EXTERNAL_ANCHOR_SCOPE,
        "artifact_dir": str(HARVEY_DIR),
        "artifacts": artifacts,
    }


def _check_harvey_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    manifest_harvey: Mapping[str, Any],
    checks: list[RcBenchmarkFreezeCheck],
) -> None:
    benchmark = payloads.get("harvey-lab-benchmark.json", {})
    summary = benchmark.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_benchmark_status",
            passed=benchmark.get("status") == "complete" and summary.get("status") == "complete",
            message="benchmark and summary status must be complete",
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_benchmark_task_count",
            passed=summary.get("zaxy_task_count") == 10 and summary.get("article_task_count") == 10,
            message="benchmark must cover 10 Zaxy tasks and 10 article tasks",
        )
    )

    manifest_commit = manifest_harvey.get("harvey_commit")
    zaxy_results = benchmark.get("zaxy_results")
    zaxy_results = zaxy_results if isinstance(zaxy_results, list) else []
    zaxy_result_rows_ok = len(zaxy_results) == 10 and all(
        _valid_harvey_zaxy_result(row, manifest_commit) for row in zaxy_results
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_zaxy_result_rows",
            passed=zaxy_result_rows_ok,
            message="benchmark must include 10 row-level Zaxy results with commit, score, paths, and memory-call evidence",
        )
    )

    task_rows = benchmark.get("task_rows")
    task_rows = task_rows if isinstance(task_rows, Mapping) else {}
    task_rows_ok = len(task_rows) == 10 and all(
        _valid_harvey_task_row(row) for row in task_rows.values()
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_task_rows",
            passed=task_rows_ok,
            message="benchmark must include 10 task comparison rows with Zaxy, article-best, baseline, and memory-call fields",
        )
    )

    provenance = benchmark.get("result_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    normalized_paths = provenance.get("normalized_result_paths")
    normalized_paths = normalized_paths if isinstance(normalized_paths, list) else []
    baseline_reports = provenance.get("external_baseline_reports")
    baseline_reports = baseline_reports if isinstance(baseline_reports, list) else []
    provenance_ok = (
        provenance.get("harvey_git_commit") == manifest_commit
        and len(normalized_paths) == 10
        and all(isinstance(path, str) and path for path in normalized_paths)
        and bool(baseline_reports)
        and all(_valid_harvey_baseline_report(report) for report in baseline_reports)
        and _nonempty_string_list(provenance.get("external_baseline_report_paths"))
        and _nonempty_string_list(provenance.get("external_readiness_report_paths"))
        and _nonempty_string_list(provenance.get("external_run_manifest_paths"))
        and _nonempty_string_list(provenance.get("external_status_report_paths"))
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_result_provenance",
            passed=provenance_ok,
            message="benchmark must preserve normalized-result paths, external report paths, and Harvey commit provenance",
        )
    )

    status = payloads.get("harvey-lab-status.json", {})
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_status_complete",
            passed=status.get("status") == "complete" and status.get("ready_task_count") == 10,
            message="status artifact must be complete with 10 ready tasks",
        )
    )

    external_run = payloads.get("harvey-lab-external-run.json", {})
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_external_run_task_count",
            passed=external_run.get("task_count") == 10,
            message="external run contract must cover 10 tasks",
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_external_run_report_path",
            passed=external_run.get("report_json_path")
            == "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json",
            message="external run must point at the frozen Harvey benchmark report",
        )
    )

    ready = payloads.get("harvey-lab-ready.json", {})
    blocking_reasons = ready.get("blocking_reasons")
    blocking_reasons = blocking_reasons if isinstance(blocking_reasons, list) else []
    readiness_complete = (
        ready.get("expected_task_count") == 10
        and ready.get("ready_task_count") == 10
        and ready.get("normalized_ready_count") == 10
        and ready.get("run_ready_count") == 10
    )
    acceptable_ready_status = ready.get("status") == "ready" or (
        ready.get("status") == "not_ready" and set(blocking_reasons) == {"results_already_complete"}
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_ready_artifact",
            passed=readiness_complete and acceptable_ready_status,
            message="ready artifact must be ready or blocked only because results are already complete",
        )
    )

    status_commit = status.get("harvey_git_commit")
    ready_commit = ready.get("harvey_git_commit")
    checks.append(
        RcBenchmarkFreezeCheck(
            name="harvey_commit_consistency",
            passed=bool(manifest_commit)
            and manifest_commit == status_commit
            and manifest_commit == ready_commit,
            message=f"{manifest_commit!r} == {status_commit!r} == {ready_commit!r}",
        )
    )


def _valid_harvey_zaxy_result(row: Any, manifest_commit: Any) -> bool:
    if not isinstance(row, Mapping):
        return False
    required_paths = (
        "answer_path",
        "judge_path",
        "run_metrics_path",
        "tool_log_path",
        "results_run_dir",
    )
    return (
        row.get("framework") == "zaxy"
        and row.get("commit") == manifest_commit
        and _number(row.get("score")) is not None
        and _positive_int(row.get("memory_search_calls"))
        and _positive_int(row.get("memory_read_calls"))
        and all(isinstance(row.get(key), str) and bool(row.get(key)) for key in required_paths)
    )


def _valid_harvey_task_row(row: Any) -> bool:
    if not isinstance(row, Mapping):
        return False
    return (
        _number(row.get("zaxy_score")) is not None
        and _number(row.get("article_best_score")) is not None
        and _number(row.get("regular_no_memory_score")) is not None
        and _positive_int(row.get("zaxy_memory_search_calls"))
        and _positive_int(row.get("zaxy_memory_read_calls"))
    )


def _valid_harvey_baseline_report(report: Any) -> bool:
    if not isinstance(report, Mapping):
        return False
    return (
        isinstance(report.get("path"), str)
        and bool(report.get("path"))
        and _positive_int(report.get("normalized_result_count"))
        and _positive_int(report.get("framework_count"))
    )


def _nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _internal_lanes(
    manifest: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    lanes = {
        "causal": {
            "lane": "causal",
            "claim_scope": INTERNAL_LANE_SCOPE,
            "evidence_kind": "contract_scorer",
            "module": "zaxy.causal_benchmark",
        },
        "consolidation": {
            "lane": "consolidation",
            "claim_scope": INTERNAL_LANE_SCOPE,
            "evidence_kind": "contract_scorer",
            "module": "zaxy.consolidation_benchmark",
        },
        "procedural": {
            "lane": "procedural",
            "claim_scope": INTERNAL_LANE_SCOPE,
            "evidence_kind": "contract_scorer",
            "module": "zaxy.reasoning_benchmark",
        },
        "metacognition": {
            "lane": "metacognition",
            "claim_scope": INTERNAL_LANE_SCOPE,
            "evidence_kind": "contract_scorer",
            "module": "zaxy.reasoning_benchmark",
        },
    }
    manifest_lanes = manifest.get("internal_lanes")
    if isinstance(manifest_lanes, list):
        lanes = {}
        for lane in manifest_lanes:
            if isinstance(lane, Mapping):
                lane_name = str(lane.get("lane", ""))
                if lane_name:
                    lanes[lane_name] = dict(lane)
    for lane, override in (overrides or {}).items():
        merged = dict(lanes.get(lane, {"lane": lane}))
        merged.update(dict(override))
        lanes[lane] = merged
    return [lanes[name] for name in sorted(lanes)]


def _check_internal_lanes(lanes: list[dict[str, Any]]) -> list[RcBenchmarkFreezeCheck]:
    checks: list[RcBenchmarkFreezeCheck] = []
    lane_names = {str(lane.get("lane", "")) for lane in lanes}
    for required in REQUIRED_INTERNAL_LANES:
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"internal_lane_present:{required}",
                passed=required in lane_names,
                message="required 2.0 project-defined internal lane",
            )
        )
    for lane in lanes:
        lane_name = str(lane.get("lane", ""))
        scope = str(lane.get("claim_scope", ""))
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"internal_lane_scope:{lane_name}",
                passed=scope == INTERNAL_LANE_SCOPE,
                message=f"{scope or '<missing>'} == {INTERNAL_LANE_SCOPE}",
            )
        )
    return checks


def _project_benchmark_evidence(
    root: Path,
    checks: list[RcBenchmarkFreezeCheck],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    raw = manifest.get("project_benchmarks")
    benchmarks = raw if isinstance(raw, Mapping) else {}
    state = _project_benchmark_entry(benchmarks, "state_recovery")
    coordination = _project_benchmark_entry(benchmarks, "coordination")
    purpose = _project_benchmark_entry(benchmarks, "purpose")
    evidence = {
        "state_recovery": _validate_state_recovery_benchmark(root, checks, state),
        "coordination": _validate_coordination_benchmark(root, checks, coordination),
        "purpose": _validate_purpose_benchmark(root, checks, purpose),
    }
    return evidence


def _project_benchmark_entry(manifest_benchmarks: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    entry = manifest_benchmarks.get(name)
    return entry if isinstance(entry, Mapping) else {}


def _validate_state_recovery_benchmark(
    root: Path,
    checks: list[RcBenchmarkFreezeCheck],
    manifest_entry: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = Path(str(manifest_entry.get("artifact") or ""))
    workload_path = Path(str(manifest_entry.get("workload") or ""))
    markdown_path = Path(str(manifest_entry.get("markdown") or ""))
    _check_manifest_scope(checks, "state_recovery", manifest_entry)
    report = _load_required_report(root, checks, "state_recovery", report_path)
    workload = _load_required_report(root, checks, "state_recovery_workload", workload_path)
    _check_file(checks, root, "state_recovery_markdown", markdown_path)
    checks.append(
        RcBenchmarkFreezeCheck(
            name="state_recovery_status",
            passed=report.get("status") == "pass",
            message="StateRecoveryBench status must be pass",
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="state_recovery_version",
            passed=report.get("version") == manifest_entry.get("version") == "state-recovery-v0",
            message="StateRecoveryBench version must match manifest",
        )
    )
    manifest_fingerprint = manifest_entry.get("workload_fingerprint")
    report_fingerprint = report.get("workload_fingerprint")
    workload_fingerprint = workload.get("fingerprint")
    checks.append(
        RcBenchmarkFreezeCheck(
            name="state_recovery_workload_fingerprint",
            passed=manifest_fingerprint == report_fingerprint == workload_fingerprint,
            message=f"{manifest_fingerprint!r} == {report_fingerprint!r} == {workload_fingerprint!r}",
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="state_recovery_production_baseline",
            passed=report.get("production_baseline") == "memory_fabric_checkout",
            message="production baseline must be memory_fabric_checkout",
        )
    )
    checks_obj = report.get("checks")
    checks_obj = checks_obj if isinstance(checks_obj, Mapping) else {}
    for metric in (
        "state_accuracy",
        "minimal_evidence_recall",
        "stale_rejection",
        "distractor_resistance",
        "abstention_accuracy",
        "citation_coverage",
    ):
        row = checks_obj.get(metric)
        row = row if isinstance(row, Mapping) else {}
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"state_recovery_check:{metric}",
                passed=row.get("status") == "pass",
                message=f"{metric} guardrail must pass",
            )
        )
    return {
        "artifact": str(report_path),
        "claim_scope": INTERNAL_LANE_SCOPE,
        "workload": str(workload_path),
        "workload_fingerprint": manifest_fingerprint,
    }


def _validate_coordination_benchmark(
    root: Path,
    checks: list[RcBenchmarkFreezeCheck],
    manifest_entry: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = Path(str(manifest_entry.get("artifact") or ""))
    workload_path = Path(str(manifest_entry.get("workload") or ""))
    markdown_path = Path(str(manifest_entry.get("markdown") or ""))
    _check_manifest_scope(checks, "coordination", manifest_entry)
    report = _load_required_report(root, checks, "coordination", report_path)
    _load_required_report(root, checks, "coordination_workload", workload_path)
    _check_file(checks, root, "coordination_markdown", markdown_path)
    checks.append(
        RcBenchmarkFreezeCheck(
            name="coordination_version",
            passed=report.get("version") == manifest_entry.get("version")
            and str(report.get("version") or "") in {"coordination-v1", "coordination-real-v1"},
            message="CoordinationBench version must match manifest",
        )
    )
    manifest_fingerprint = manifest_entry.get("workload_fingerprint")
    report_fingerprint = report.get("workload_fingerprint")
    checks.append(
        RcBenchmarkFreezeCheck(
            name="coordination_workload_fingerprint",
            passed=manifest_fingerprint == report_fingerprint,
            message=f"{manifest_fingerprint!r} == {report_fingerprint!r}",
        )
    )
    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    for metric in (
        "accepted_finding_precision",
        "accepted_finding_recall",
        "citation_coverage",
        "evidence_coverage",
        "stale_claim_rejection",
        "duplicate_consolidation",
        "non_authoritative_leakage",
        "parent_checkout_answerability",
        "purpose_feedback_coverage",
    ):
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"coordination_metric:{metric}",
                passed=_numeric(metrics.get(metric)) >= 1.0,
                message=f"{metric} must be 1.0",
            )
        )
    return {
        "artifact": str(report_path),
        "claim_scope": INTERNAL_LANE_SCOPE,
        "workload": str(workload_path),
        "workload_fingerprint": manifest_fingerprint,
    }


def _validate_purpose_benchmark(
    root: Path,
    checks: list[RcBenchmarkFreezeCheck],
    manifest_entry: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = Path(str(manifest_entry.get("artifact") or ""))
    markdown_path = Path(str(manifest_entry.get("markdown") or ""))
    holdout_pack_path = Path(str(manifest_entry.get("holdout_pack") or ""))
    source_disclosures_path = Path(str(manifest_entry.get("source_disclosures") or ""))
    _check_manifest_scope(checks, "purpose", manifest_entry)
    report = _load_required_report(root, checks, "purpose", report_path)
    _check_file(checks, root, "purpose_markdown", markdown_path)
    holdout_pack = _load_required_report(root, checks, "purpose_holdout_pack", holdout_pack_path)
    _load_required_report(root, checks, "purpose_source_disclosures", source_disclosures_path)
    checks.append(
        RcBenchmarkFreezeCheck(
            name="purpose_version",
            passed=report.get("version") == manifest_entry.get("version") == "purpose-v1",
            message="PurposeBench version must match manifest",
        )
    )
    lane_count = report.get("lane_count")
    passed_lanes = report.get("passed_lanes")
    manifest_lane_count = manifest_entry.get("lane_count")
    checks.append(
        RcBenchmarkFreezeCheck(
            name="purpose_status",
            passed=report.get("status") == "passed",
            message="PurposeBench status must be passed",
        )
    )
    checks.append(
        RcBenchmarkFreezeCheck(
            name="purpose_lanes",
            passed=lane_count == passed_lanes == manifest_lane_count == 10,
            message=f"{passed_lanes!r}/{lane_count!r} lanes must pass",
        )
    )
    holdout_reports = report.get("holdout_reports")
    holdout_reports = holdout_reports if isinstance(holdout_reports, Mapping) else {}
    public_holdout = holdout_reports.get("public-derived-purpose-v1")
    public_holdout = public_holdout if isinstance(public_holdout, Mapping) else {}
    manifest_holdout_fingerprint = manifest_entry.get("holdout_fingerprint")
    pack_fingerprint = holdout_pack.get("pack_fingerprint") or holdout_pack.get("fingerprint")
    report_fingerprint = public_holdout.get("pack_fingerprint")
    checks.append(
        RcBenchmarkFreezeCheck(
            name="purpose_holdout_fingerprint",
            passed=manifest_holdout_fingerprint == pack_fingerprint == report_fingerprint,
            message=f"{manifest_holdout_fingerprint!r} == {pack_fingerprint!r} == {report_fingerprint!r}",
        )
    )
    metrics = public_holdout.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    checks.append(
        RcBenchmarkFreezeCheck(
            name="purpose_holdout_diagnostic",
            passed=public_holdout.get("gate_status") == "diagnostic"
            and public_holdout.get("claim_status") == "public_derived_holdout"
            and metrics.get("case_count") == 5,
            message="public-derived-purpose-v1 holdout must be diagnostic with five cases",
        )
    )
    return {
        "artifact": str(report_path),
        "claim_scope": INTERNAL_LANE_SCOPE,
        "holdout_pack": str(holdout_pack_path),
        "lane_count": manifest_lane_count,
        "markdown": str(markdown_path),
        "source_disclosures": str(source_disclosures_path),
    }


def _check_manifest_scope(
    checks: list[RcBenchmarkFreezeCheck],
    name: str,
    manifest_entry: Mapping[str, Any],
) -> None:
    scope = manifest_entry.get("claim_scope")
    checks.append(
        RcBenchmarkFreezeCheck(
            name=f"{name}_claim_scope",
            passed=scope == INTERNAL_LANE_SCOPE,
            message=f"{scope!r} == {INTERNAL_LANE_SCOPE}",
        )
    )


def _check_file(
    checks: list[RcBenchmarkFreezeCheck],
    root: Path,
    name: str,
    path: Path,
) -> None:
    checks.append(
        RcBenchmarkFreezeCheck(
            name=name,
            passed=bool(str(path)) and (root / path).is_file(),
            message=str(path),
        )
    )


def _load_required_report(
    root: Path,
    checks: list[RcBenchmarkFreezeCheck],
    name: str,
    path: Path,
) -> Mapping[str, Any]:
    _check_file(checks, root, f"{name}_artifact", path)
    if not str(path) or not (root / path).is_file():
        return {}
    try:
        return _read_json(root / path)
    except ValueError as exc:
        checks.append(
            RcBenchmarkFreezeCheck(
                name=f"{name}_json",
                passed=False,
                message=str(exc),
            )
        )
        return {}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _summary_for_backend(report: Mapping[str, Any], backend: str) -> Mapping[str, Any] | None:
    summaries = report.get("summaries")
    if not isinstance(summaries, list):
        return None
    for summary in summaries:
        if isinstance(summary, Mapping) and summary.get("backend") == backend:
            return summary
    return None


def _numeric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return float("nan")
    return float(value)


def _git_tracking_enabled(root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _git_tracked(root: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0
