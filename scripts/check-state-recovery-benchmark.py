#!/usr/bin/env python3
"""Validate a StateRecoveryBench report for release use."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA_VERSION = "state-recovery-report-v1"
EXPECTED_WORKLOAD_VERSION = "state-recovery-v0"
EXPECTED_PRODUCTION_BASELINE = "memory_fabric_checkout"
EXPECTED_BASELINES = (
    "direct_lexical",
    "hash_vector",
    "graph_traversal",
    "zaxy_core_proxy",
    "memory_fabric_checkout",
    "associative_projection",
    "authority_resolved_associative",
)
EXPECTED_THRESHOLDS = {
    "state_accuracy": 0.818,
    "minimal_evidence_recall": 0.90,
    "stale_rejection": 1.0,
    "distractor_resistance": 0.80,
    "abstention_accuracy": 1.0,
    "citation_coverage": 1.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="state-recovery-benchmark.json")
    parser.add_argument("--workload", type=Path, required=True, help="state-recovery-workload.json")
    parser.add_argument(
        "--require-git-tracked-inputs",
        action="store_true",
        help="Require the supplied report and workload to be tracked by git",
    )
    args = parser.parse_args()

    failures: list[str] = []
    report = _load_json(args.report, failures, label="report")
    workload = _load_json(args.workload, failures, label="workload")
    if failures:
        return _finish(failures)
    assert isinstance(report, dict)
    assert isinstance(workload, dict)

    _check_workload(workload, failures)
    _check_report(report, workload, failures)
    if args.require_git_tracked_inputs:
        _check_git_tracked(args.report, failures)
        _check_git_tracked(args.workload, failures)
    return _finish(failures)


def _load_json(path: Path, failures: list[str], *, label: str) -> Any:
    if not path.exists():
        failures.append(f"missing {label}: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"invalid {label} JSON: {path}: {exc}")
        return None


def _check_workload(workload: dict[str, Any], failures: list[str]) -> None:
    if workload.get("version") != EXPECTED_WORKLOAD_VERSION:
        failures.append(f"unexpected workload version: {workload.get('version')!r}")
    cases = workload.get("cases")
    if not isinstance(cases, list) or not cases:
        failures.append("workload cases must be a non-empty list")
        return
    expected = _workload_fingerprint(workload)
    if workload.get("fingerprint") != expected:
        failures.append("workload fingerprint does not match workload contents")


def _check_report(report: dict[str, Any], workload: dict[str, Any], failures: list[str]) -> None:
    if report.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        failures.append(f"unexpected report schema_version: {report.get('schema_version')!r}")
    if report.get("version") != EXPECTED_WORKLOAD_VERSION:
        failures.append(f"unexpected report version: {report.get('version')!r}")
    if report.get("workload_fingerprint") != workload.get("fingerprint"):
        failures.append("report workload_fingerprint does not match workload fingerprint")
    if report.get("case_count") != len(workload.get("cases", [])):
        failures.append("report case_count does not match workload case count")
    if tuple(report.get("baseline_names", ())) != EXPECTED_BASELINES:
        failures.append("report baseline_names do not match the official StateRecoveryBench contract")
    if report.get("production_baseline") != EXPECTED_PRODUCTION_BASELINE:
        failures.append(f"unexpected production_baseline: {report.get('production_baseline')!r}")
    if report.get("thresholds") != EXPECTED_THRESHOLDS:
        failures.append("report thresholds do not match the release guardrail contract")
    if report.get("status") != "pass":
        failures.append(f"report status is not pass: {report.get('status')!r}")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        failures.append("report checks must be an object")
        return
    for metric, threshold in EXPECTED_THRESHOLDS.items():
        check = checks.get(metric)
        if not isinstance(check, dict):
            failures.append(f"missing guardrail check: {metric}")
            continue
        if check.get("baseline") != EXPECTED_PRODUCTION_BASELINE:
            failures.append(f"{metric} guardrail does not target {EXPECTED_PRODUCTION_BASELINE}")
        observed = check.get("observed")
        if not isinstance(observed, int | float) or isinstance(observed, bool):
            failures.append(f"{metric} observed value is not numeric")
            continue
        if float(observed) < threshold:
            failures.append(f"{metric}={observed} is below threshold {threshold}")
        if check.get("status") != "pass":
            failures.append(f"{metric} guardrail status is not pass")
    baselines = report.get("baselines")
    if not isinstance(baselines, dict):
        failures.append("report baselines must be an object")
        return
    if EXPECTED_PRODUCTION_BASELINE not in baselines:
        failures.append(f"missing production baseline row: {EXPECTED_PRODUCTION_BASELINE}")


def _workload_fingerprint(workload: dict[str, Any]) -> str:
    body = {"version": workload.get("version"), "cases": workload.get("cases", [])}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _check_git_tracked(path: Path, failures: list[str]) -> None:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        failures.append(f"not tracked by git: {path}")


def _finish(failures: list[str]) -> int:
    if failures:
        for failure in failures:
            print(f"StateRecoveryBench guardrail failed: {failure}", file=sys.stderr)
        return 1
    print("StateRecoveryBench guardrail passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
