#!/usr/bin/env python3
"""Validate backend shootout reports for release-gate use."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeGuard

EVENTLOOM_CITATION_RE = re.compile(r"^eventloom://[^/\s]+/events/[1-9][0-9]*#[0-9A-Fa-f]+$")
KNOWN_BACKENDS = {"embedded", "latticedb", "neo4j", "pggraph", "bm25"}
GRAPH_BACKENDS = {"embedded", "latticedb", "neo4j", "pggraph"}
KNOWN_TOP_CONTEXT_SOURCES = {"bm25", "exact", "keyword", "traversal", "vector", "verbatim"}
KNOWN_CONTRACTS = {"answer_ready", "retrieve"}
KNOWN_STATUSES = {"error", "ok"}
ERROR_STATUS_EMPTY_METRICS = {
    "answer_at_5",
    "answer_at_5_per_1k_injected_tokens",
    "answer_at_5_per_1k_returned_tokens",
    "citation_coverage",
    "mean_quality",
    "quality_per_1k_injected_tokens",
    "quality_per_1k_returned_tokens",
    "recall_at_5",
}
INTEGER_SUMMARY_METRICS = {
    "dashboard_graph_nodes",
    "dashboard_graph_edges",
    "memory_footprint_bytes",
    "on_disk_footprint_bytes",
    "query_count",
    "resident_memory_delta_bytes",
}
NON_NEGATIVE_SUMMARY_METRICS = {
    "cold_bootstrap_ms",
    "first_useful_init_ms",
    "first_checkout_ms",
    "append_to_projection_p95_ms",
    "projection_events_per_second",
    "checkout_p95_ms",
    "checkout_p99_ms",
    "exact_p50_ms",
    "exact_p95_ms",
    "exact_p99_ms",
    "keyword_p50_ms",
    "keyword_p95_ms",
    "keyword_p99_ms",
    "vector_p50_ms",
    "vector_p95_ms",
    "vector_p99_ms",
    "traversal_p50_ms",
    "traversal_p95_ms",
    "traversal_p99_ms",
    "dashboard_graph_load_ms",
    "dashboard_graph_nodes",
    "dashboard_graph_edges",
    "mean_returned_tokens",
    "quality_per_1k_returned_tokens",
    "answer_at_5_per_1k_returned_tokens",
    "mean_injected_tokens",
    "quality_per_1k_injected_tokens",
    "answer_at_5_per_1k_injected_tokens",
    "memory_footprint_bytes",
    "on_disk_footprint_bytes",
    "query_count",
    "rebuild_recovery_ms",
    "resident_memory_delta_bytes",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a backend shootout JSON report.")
    parser.add_argument("report", type=Path, help="backend-shootout JSON report")
    parser.add_argument("--require-backends", default="", help="Comma-separated backends that must be present and ok")
    parser.add_argument(
        "--forbid-backends",
        default="",
        help="Comma-separated parked candidate backends that must not appear in active release reports",
    )
    parser.add_argument(
        "--require-report-metadata",
        action="store_true",
        help="Require schema version, generation timestamp, and source/workload fingerprints",
    )
    parser.add_argument(
        "--require-markdown-report",
        action="store_true",
        help="Require the generated Markdown sidecar to include matching report provenance",
    )
    parser.add_argument(
        "--require-query-results",
        action="store_true",
        help="Require per-query diagnostics for each backend contract summary",
    )
    parser.add_argument(
        "--verify-report-fingerprints",
        action="store_true",
        help="Recompute source/workload fingerprints from the report inputs and reject stale reports",
    )
    parser.add_argument(
        "--require-git-tracked-inputs",
        action="store_true",
        help="Require the report Eventloom and query input paths to be tracked by git",
    )
    parser.add_argument(
        "--require-labeled-metrics",
        action="store_true",
        help="Require answer_at_5 and recall_at_5 for required ok backends",
    )
    parser.add_argument(
        "--min-answer-at-5",
        type=_non_negative_float,
        default=None,
        help="Minimum Answer@5 for required backends",
    )
    parser.add_argument(
        "--min-recall-at-5",
        type=_non_negative_float,
        default=None,
        help="Minimum Recall@5 for required backends",
    )
    parser.add_argument(
        "--min-citation-coverage",
        type=_non_negative_float,
        default=None,
        help="Minimum citation coverage for required backends",
    )
    parser.add_argument(
        "--require-dashboard-source",
        action="append",
        default=[],
        metavar="BACKEND=SOURCE",
        help="Require a backend to report a specific dashboard_graph_source",
    )
    parser.add_argument(
        "--min-projection-events-per-second",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to meet a minimum projection_events_per_second",
    )
    parser.add_argument(
        "--max-cold-bootstrap-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum cold_bootstrap_ms",
    )
    parser.add_argument(
        "--max-first-useful-init-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum first_useful_init_ms",
    )
    parser.add_argument(
        "--max-first-checkout-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum first_checkout_ms",
    )
    parser.add_argument(
        "--max-append-to-projection-p95-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum append_to_projection_p95_ms",
    )
    parser.add_argument(
        "--max-resident-memory-delta-bytes",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum resident_memory_delta_bytes",
    )
    parser.add_argument(
        "--max-on-disk-footprint-bytes",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum on_disk_footprint_bytes",
    )
    parser.add_argument(
        "--max-rebuild-recovery-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum rebuild_recovery_ms",
    )
    parser.add_argument(
        "--max-checkout-p95-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum checkout_p95_ms",
    )
    parser.add_argument(
        "--max-checkout-p99-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum checkout_p99_ms",
    )
    parser.add_argument(
        "--max-dashboard-graph-load-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum dashboard_graph_load_ms",
    )
    parser.add_argument(
        "--min-quality-per-1k-returned-tokens",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to meet a minimum quality_per_1k_returned_tokens",
    )
    parser.add_argument(
        "--min-answer-at-5-per-1k-returned-tokens",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to meet a minimum answer_at_5_per_1k_returned_tokens",
    )
    parser.add_argument(
        "--min-quality-per-1k-injected-tokens",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to meet a minimum quality_per_1k_injected_tokens",
    )
    parser.add_argument(
        "--min-answer-at-5-per-1k-injected-tokens",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to meet a minimum answer_at_5_per_1k_injected_tokens",
    )
    parser.add_argument(
        "--max-exact-p95-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum exact_p95_ms",
    )
    parser.add_argument(
        "--max-exact-p99-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum exact_p99_ms",
    )
    parser.add_argument(
        "--max-keyword-p95-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum keyword_p95_ms",
    )
    parser.add_argument(
        "--max-keyword-p99-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum keyword_p99_ms",
    )
    parser.add_argument(
        "--max-vector-p95-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum vector_p95_ms",
    )
    parser.add_argument(
        "--max-vector-p99-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum vector_p99_ms",
    )
    parser.add_argument(
        "--max-traversal-p95-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum traversal_p95_ms",
    )
    parser.add_argument(
        "--max-traversal-p99-ms",
        action="append",
        default=[],
        metavar="BACKEND=FLOAT",
        help="Require a backend to stay below a maximum traversal_p99_ms",
    )
    args = parser.parse_args()

    errors = validate_report(args)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print("Backend shootout guardrail passed.")


def validate_report(args: argparse.Namespace) -> list[str]:
    try:
        report = _load_report(args.report)
    except ValueError as exc:
        return [f"report: {exc}"]
    summaries = report.get("summaries")
    if not isinstance(summaries, list):
        return ["report: summaries must be a list"]
    if not summaries:
        return ["report: summaries must contain at least one backend row"]
    report_schema_errors = _report_schema_errors(report)
    report_query_count = report.get("query_count")
    summary_shape_errors = _summary_shape_errors(
        summaries,
        require_query_count=report.get("report_schema_version") is not None,
        report_query_count=report_query_count if _non_negative_int(report_query_count) else None,
    )
    duplicate_errors = _duplicate_backend_errors(summaries)
    by_backend_contract = _summaries_by_backend_contract(summaries)
    by_backend = {
        backend: _preferred_quality_summary(contract_rows)
        for backend, contract_rows in by_backend_contract.items()
    }
    retrieve_by_backend = {
        backend: _preferred_retrieval_summary(contract_rows)
        for backend, contract_rows in by_backend_contract.items()
    }
    required_backends = _csv(args.require_backends)
    forbidden_backends = _csv(args.forbid_backends)
    dashboard_sources = _dashboard_source_requirements(args.require_dashboard_source)
    min_projection_eps = _metric_requirements(args.min_projection_events_per_second, "--min-projection-events-per-second")
    max_cold_bootstrap_ms = _metric_requirements(args.max_cold_bootstrap_ms, "--max-cold-bootstrap-ms")
    max_first_init_ms = _metric_requirements(args.max_first_useful_init_ms, "--max-first-useful-init-ms")
    max_first_checkout_ms = _metric_requirements(args.max_first_checkout_ms, "--max-first-checkout-ms")
    max_append_projection_p95_ms = _metric_requirements(
        args.max_append_to_projection_p95_ms,
        "--max-append-to-projection-p95-ms",
    )
    max_resident_memory_delta_bytes = _metric_requirements(
        args.max_resident_memory_delta_bytes,
        "--max-resident-memory-delta-bytes",
    )
    max_on_disk_footprint_bytes = _metric_requirements(
        args.max_on_disk_footprint_bytes,
        "--max-on-disk-footprint-bytes",
    )
    max_rebuild_ms = _metric_requirements(args.max_rebuild_recovery_ms, "--max-rebuild-recovery-ms")
    max_checkout_p95_ms = _metric_requirements(args.max_checkout_p95_ms, "--max-checkout-p95-ms")
    max_checkout_p99_ms = _metric_requirements(args.max_checkout_p99_ms, "--max-checkout-p99-ms")
    max_dashboard_graph_load_ms = _metric_requirements(
        args.max_dashboard_graph_load_ms,
        "--max-dashboard-graph-load-ms",
    )
    min_quality_per_1k_tokens = _metric_requirements(
        args.min_quality_per_1k_returned_tokens,
        "--min-quality-per-1k-returned-tokens",
    )
    min_answer_at_5_per_1k_tokens = _metric_requirements(
        args.min_answer_at_5_per_1k_returned_tokens,
        "--min-answer-at-5-per-1k-returned-tokens",
    )
    min_quality_per_1k_injected_tokens = _metric_requirements(
        args.min_quality_per_1k_injected_tokens,
        "--min-quality-per-1k-injected-tokens",
    )
    min_answer_at_5_per_1k_injected_tokens = _metric_requirements(
        args.min_answer_at_5_per_1k_injected_tokens,
        "--min-answer-at-5-per-1k-injected-tokens",
    )
    max_exact_p95_ms = _metric_requirements(args.max_exact_p95_ms, "--max-exact-p95-ms")
    max_exact_p99_ms = _metric_requirements(args.max_exact_p99_ms, "--max-exact-p99-ms")
    max_keyword_p95_ms = _metric_requirements(args.max_keyword_p95_ms, "--max-keyword-p95-ms")
    max_keyword_p99_ms = _metric_requirements(args.max_keyword_p99_ms, "--max-keyword-p99-ms")
    max_vector_p95_ms = _metric_requirements(args.max_vector_p95_ms, "--max-vector-p95-ms")
    max_vector_p99_ms = _metric_requirements(args.max_vector_p99_ms, "--max-vector-p99-ms")
    max_traversal_p95_ms = _metric_requirements(args.max_traversal_p95_ms, "--max-traversal-p95-ms")
    max_traversal_p99_ms = _metric_requirements(args.max_traversal_p99_ms, "--max-traversal-p99-ms")
    errors: list[str] = [*report_schema_errors, *summary_shape_errors, *duplicate_errors]
    errors.extend(_unknown_backend_errors(list(by_backend), "report summaries"))
    errors.extend(_duplicate_backend_policy_errors(required_backends, "--require-backends"))
    errors.extend(_duplicate_backend_policy_errors(forbidden_backends, "--forbid-backends"))
    errors.extend(_unknown_backend_errors(required_backends, "--require-backends"))
    errors.extend(_unknown_backend_errors(forbidden_backends, "--forbid-backends"))
    errors.extend(_unknown_backend_errors(list(dashboard_sources), "--require-dashboard-source"))
    for option, requirements in [
        ("--min-projection-events-per-second", min_projection_eps),
        ("--max-cold-bootstrap-ms", max_cold_bootstrap_ms),
        ("--max-first-useful-init-ms", max_first_init_ms),
        ("--max-first-checkout-ms", max_first_checkout_ms),
        ("--max-append-to-projection-p95-ms", max_append_projection_p95_ms),
        ("--max-resident-memory-delta-bytes", max_resident_memory_delta_bytes),
        ("--max-on-disk-footprint-bytes", max_on_disk_footprint_bytes),
        ("--max-rebuild-recovery-ms", max_rebuild_ms),
        ("--max-checkout-p95-ms", max_checkout_p95_ms),
        ("--max-checkout-p99-ms", max_checkout_p99_ms),
        ("--max-dashboard-graph-load-ms", max_dashboard_graph_load_ms),
        ("--min-quality-per-1k-returned-tokens", min_quality_per_1k_tokens),
        ("--min-answer-at-5-per-1k-returned-tokens", min_answer_at_5_per_1k_tokens),
        ("--min-quality-per-1k-injected-tokens", min_quality_per_1k_injected_tokens),
        ("--min-answer-at-5-per-1k-injected-tokens", min_answer_at_5_per_1k_injected_tokens),
        ("--max-exact-p95-ms", max_exact_p95_ms),
        ("--max-exact-p99-ms", max_exact_p99_ms),
        ("--max-keyword-p95-ms", max_keyword_p95_ms),
        ("--max-keyword-p99-ms", max_keyword_p99_ms),
        ("--max-vector-p95-ms", max_vector_p95_ms),
        ("--max-vector-p99-ms", max_vector_p99_ms),
        ("--max-traversal-p95-ms", max_traversal_p95_ms),
        ("--max-traversal-p99-ms", max_traversal_p99_ms),
    ]:
        errors.extend(_unknown_backend_errors(list(requirements), option))
    errors.extend(_summary_metric_shape_errors(summaries))
    errors.extend(_query_result_shape_errors(report, summaries, require_query_results=args.require_query_results))
    errors.extend(_backend_policy_conflicts(required_backends, forbidden_backends))
    verified_query_count = _verified_query_count(report) if args.verify_report_fingerprints else None
    if args.require_report_metadata:
        errors.extend(_validate_report_metadata(report))
    if args.require_markdown_report:
        errors.extend(_validate_markdown_report(args.report.with_suffix(".md"), report))
    if args.verify_report_fingerprints:
        errors.extend(_verify_report_fingerprints(report))
        errors.extend(_verify_query_result_workload(report))
    if args.require_git_tracked_inputs:
        errors.extend(_validate_git_tracked_inputs(report, args.report))
    for backend in forbidden_backends:
        if backend in by_backend:
            errors.append(f"{backend}: forbidden backend present in report")
    for backend in required_backends:
        quality_summary = by_backend.get(backend)
        retrieval_summary = retrieve_by_backend.get(backend)
        contract_rows = by_backend_contract.get(backend, {})
        if quality_summary is None:
            errors.append(f"{backend}: missing required backend summary")
            continue
        if retrieval_summary is None:
            errors.append(f"{backend}: missing required retrieve contract summary")
            continue
        if (
            backend in GRAPH_BACKENDS
            and "answer_ready" not in contract_rows
            and _has_explicit_contract_rows(contract_rows)
        ):
            errors.append(f"{backend}: missing required answer_ready contract summary")
        selected_summaries = _unique_summaries(retrieval_summary, quality_summary)
        bad_statuses = [
            summary
            for summary in selected_summaries
            if summary.get("status") != "ok"
        ]
        if bad_statuses:
            for summary in bad_statuses:
                errors.append(f"{backend}: status is {summary.get('status')!r}, expected 'ok'")
            continue
        if verified_query_count is not None:
            for summary in selected_summaries:
                if summary.get("query_count") != verified_query_count:
                    errors.append(
                        f"{backend}: query_count={summary.get('query_count')} "
                        f"does not match current input count {verified_query_count}"
                    )
        if args.require_labeled_metrics:
            errors.extend(_require_metric(quality_summary, backend, "answer_at_5"))
            errors.extend(_require_metric(retrieval_summary, backend, "recall_at_5"))
        errors.extend(_check_floor(quality_summary, backend, "answer_at_5", args.min_answer_at_5))
        errors.extend(_check_floor(retrieval_summary, backend, "recall_at_5", args.min_recall_at_5))
        errors.extend(_check_floor(quality_summary, backend, "citation_coverage", args.min_citation_coverage))
        expected_source = dashboard_sources.get(backend)
        if expected_source is not None and retrieval_summary.get("dashboard_graph_source") != expected_source:
            errors.append(
                f"{backend}: dashboard_graph_source is {retrieval_summary.get('dashboard_graph_source')!r}, "
                f"expected {expected_source!r}"
            )
        errors.extend(_check_backend_floor(retrieval_summary, backend, "projection_events_per_second", min_projection_eps))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "cold_bootstrap_ms", max_cold_bootstrap_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "first_useful_init_ms", max_first_init_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "first_checkout_ms", max_first_checkout_ms))
        errors.extend(
            _check_backend_ceiling(
                retrieval_summary,
                backend,
                "append_to_projection_p95_ms",
                max_append_projection_p95_ms,
            )
        )
        errors.extend(
            _check_backend_ceiling(
                retrieval_summary,
                backend,
                "resident_memory_delta_bytes",
                max_resident_memory_delta_bytes,
            )
        )
        errors.extend(
            _check_backend_ceiling(
                retrieval_summary,
                backend,
                "on_disk_footprint_bytes",
                max_on_disk_footprint_bytes,
            )
        )
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "rebuild_recovery_ms", max_rebuild_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "checkout_p95_ms", max_checkout_p95_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "checkout_p99_ms", max_checkout_p99_ms))
        errors.extend(
            _check_backend_ceiling(
                retrieval_summary,
                backend,
                "dashboard_graph_load_ms",
                max_dashboard_graph_load_ms,
            )
        )
        errors.extend(
            _check_backend_floor(
                quality_summary,
                backend,
                "quality_per_1k_returned_tokens",
                min_quality_per_1k_tokens,
            )
        )
        errors.extend(
            _check_backend_floor(
                quality_summary,
                backend,
                "answer_at_5_per_1k_returned_tokens",
                min_answer_at_5_per_1k_tokens,
            )
        )
        errors.extend(
            _check_backend_floor(
                quality_summary,
                backend,
                "quality_per_1k_injected_tokens",
                min_quality_per_1k_injected_tokens,
            )
        )
        errors.extend(
            _check_backend_floor(
                quality_summary,
                backend,
                "answer_at_5_per_1k_injected_tokens",
                min_answer_at_5_per_1k_injected_tokens,
            )
        )
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "exact_p95_ms", max_exact_p95_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "exact_p99_ms", max_exact_p99_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "keyword_p95_ms", max_keyword_p95_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "keyword_p99_ms", max_keyword_p99_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "vector_p95_ms", max_vector_p95_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "vector_p99_ms", max_vector_p99_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "traversal_p95_ms", max_traversal_p95_ms))
        errors.extend(_check_backend_ceiling(retrieval_summary, backend, "traversal_p99_ms", max_traversal_p99_ms))
    return errors


def _load_report(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains non-standard numeric constant {value}")

    report = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(report, dict):
        raise ValueError("top-level report must be an object")
    return report


def _summaries_by_backend_contract(summaries: list[Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Group valid summary rows by backend and benchmark contract."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        if not isinstance(summary, dict) or summary.get("backend") is None:
            continue
        backend = str(summary["backend"])
        contract = str(summary.get("contract") or "retrieve")
        grouped.setdefault(backend, {})[contract] = summary
    return grouped


def _preferred_quality_summary(contract_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the row used for answer-ready quality gates."""
    return contract_rows.get("answer_ready") or contract_rows.get("retrieve") or next(iter(contract_rows.values()))


def _preferred_retrieval_summary(contract_rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Return the row used for raw retrieval and operational latency gates."""
    return contract_rows.get("retrieve")


def _has_explicit_contract_rows(contract_rows: dict[str, dict[str, Any]]) -> bool:
    """Return whether a backend report opted into contract-specific rows."""
    return any("contract" in summary for summary in contract_rows.values())


def _unique_summaries(*summaries: dict[str, Any]) -> list[dict[str, Any]]:
    """Deduplicate summary row objects while preserving order."""
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for summary in summaries:
        identity = id(summary)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(summary)
    return unique


def _summary_shape_errors(
    summaries: list[Any],
    *,
    require_query_count: bool = False,
    report_query_count: int | None = None,
) -> list[str]:
    errors: list[str] = []
    for index, summary in enumerate(summaries):
        if not isinstance(summary, dict):
            errors.append(f"report: summaries[{index}] must be an object")
            continue
        if not _non_empty_string(summary.get("backend")):
            errors.append(f"report: summaries[{index}].backend is missing")
        if "contract" in summary and summary.get("contract") not in KNOWN_CONTRACTS:
            expected = ", ".join(sorted(KNOWN_CONTRACTS))
            errors.append(
                f"report: summaries[{index}].contract is {summary.get('contract')!r}, expected one of: {expected}"
            )
        if not _non_empty_string(summary.get("status")):
            errors.append(f"report: summaries[{index}].status is missing")
        elif summary.get("status") not in KNOWN_STATUSES:
            expected = ", ".join(sorted(KNOWN_STATUSES))
            errors.append(f"report: summaries[{index}].status is {summary.get('status')!r}, expected one of: {expected}")
        elif summary.get("status") == "error" and not _non_empty_string(summary.get("error")):
            errors.append(f"report: summaries[{index}].error is missing for error status")
        elif summary.get("status") == "error":
            errors.extend(_error_status_metric_errors(summary))
        elif summary.get("status") == "ok" and _non_empty_string(summary.get("error")):
            errors.append(f"report: summaries[{index}].error must be empty for ok status")
        if require_query_count:
            query_count = summary.get("query_count")
            if query_count is None:
                errors.append(f"report: summaries[{index}].query_count is missing")
            elif not _non_negative_int(query_count):
                errors.append(f"report: summaries[{index}].query_count must be a non-negative integer")
            elif report_query_count is not None and query_count != report_query_count:
                errors.append(
                    f"{summary['backend']}: query_count={query_count} "
                    f"does not match report query_count {report_query_count}"
                )
    return errors


def _error_status_metric_errors(summary: dict[str, Any]) -> list[str]:
    backend = summary.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        backend = f"report: backend {backend!r}"
    return [
        f"{backend}: {metric} must be empty for error status"
        for metric in sorted(ERROR_STATUS_EMPTY_METRICS)
        if summary.get(metric) is not None
    ]


def _report_schema_errors(report: dict[str, Any]) -> list[str]:
    if report.get("report_schema_version") is None:
        return []
    errors: list[str] = []
    if not _schema_version_one(report.get("report_schema_version")):
        errors.append("report: report_schema_version must be 1")
    if not _non_negative_int(report.get("event_count")):
        errors.append("report: event_count must be a non-negative integer")
    if not _non_negative_int(report.get("query_count")):
        errors.append("report: query_count must be a non-negative integer")
    if not _positive_int(report.get("limit")):
        errors.append("report: limit must be a positive integer")
    return errors


def _summary_metric_shape_errors(summaries: list[Any]) -> list[str]:
    errors: list[str] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        backend = summary.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            continue
        errors.extend(_check_rate_metric(summary, backend, "answer_at_5"))
        errors.extend(_check_rate_metric(summary, backend, "recall_at_5"))
        errors.extend(_check_rate_metric(summary, backend, "citation_coverage"))
        errors.extend(_check_rate_metric(summary, backend, "mean_quality"))
        for metric in sorted(NON_NEGATIVE_SUMMARY_METRICS):
            errors.extend(_check_non_negative_metric(summary, backend, metric))
        for metric in sorted(INTEGER_SUMMARY_METRICS):
            errors.extend(_check_integer_metric(summary, backend, metric))
        errors.extend(_check_non_empty_string_metric(summary, backend, "dashboard_graph_source"))
    return errors


def _query_result_shape_errors(report: dict[str, Any], summaries: list[Any], *, require_query_results: bool = False) -> list[str]:
    query_results = report.get("query_results")
    if query_results is None:
        if require_query_results:
            return ["report: query_results are required"]
        return []
    if not isinstance(query_results, dict):
        return ["report: query_results must be an object"]

    errors: list[str] = []
    expected_counts = _summary_result_query_counts(summaries)
    expected_keys = set(expected_counts)
    actual_keys = set(query_results)
    for key in sorted(expected_keys - actual_keys):
        errors.append(f"report: query_results missing diagnostics for {key}")
    for key in sorted(actual_keys - expected_keys):
        errors.append(f"report: query_results contains diagnostics for {key} without matching summary")
    for key, value in query_results.items():
        if not isinstance(value, list):
            errors.append(f"report: query_results[{key!r}] must be a list")
            continue
        expected_count = expected_counts.get(key)
        if expected_count is not None and len(value) != expected_count:
            errors.append(
                f"report: query_results[{key!r}] has {len(value)} diagnostics, expected {expected_count}"
            )
        errors.extend(_query_diagnostic_item_errors(key, value))
    errors.extend(_query_result_summary_metric_errors(query_results, summaries))
    return errors


def _query_result_summary_metric_errors(query_results: dict[Any, Any], summaries: list[Any]) -> list[str]:
    errors: list[str] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        backend = summary.get("backend")
        if backend is None:
            continue
        key = f"{backend}:{summary.get('contract') or 'retrieve'}"
        diagnostics = query_results.get(key)
        if not isinstance(diagnostics, list) or not _query_diagnostics_ready_for_aggregation(diagnostics):
            continue
        expected = _query_result_aggregates(diagnostics)
        for metric, expected_value in expected.items():
            actual = summary.get(metric)
            if expected_value is None:
                continue
            if actual is None:
                errors.append(f"{key}: {metric} is missing; query_results aggregate is {expected_value}")
            elif not _number(actual):
                errors.append(f"{key}: {metric} must be a number to compare with query_results aggregate {expected_value}")
            elif round(float(actual), 4) != expected_value:
                errors.append(f"{key}: {metric}={actual} does not match query_results aggregate {expected_value}")
    return errors


def _query_diagnostics_ready_for_aggregation(diagnostics: list[Any]) -> TypeGuard[list[dict[str, Any]]]:
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            return False
        if not _number(diagnostic.get("quality")) or not _number(diagnostic.get("recall_quality")):
            return False
        if not _number(diagnostic.get("latency_ms")):
            return False
        if not _non_negative_int(diagnostic.get("returned_tokens")):
            return False
        if not _non_negative_int(diagnostic.get("injected_tokens")):
            return False
        if not isinstance(diagnostic.get("citation_hit"), bool):
            return False
        if not isinstance(diagnostic.get("expected_terms"), list):
            return False
        if not isinstance(diagnostic.get("retrieval_terms"), list):
            return False
    return True


def _query_result_aggregates(diagnostics: list[dict[str, Any]]) -> dict[str, float | None]:
    answer_labeled = [diagnostic for diagnostic in diagnostics if diagnostic["expected_terms"]]
    recall_labeled = [diagnostic for diagnostic in diagnostics if diagnostic["retrieval_terms"]]
    mean_quality = _mean_numbers([float(diagnostic["quality"]) for diagnostic in diagnostics])
    answer_at_5 = (
        round(sum(1.0 for diagnostic in answer_labeled if float(diagnostic["quality"]) >= 1.0) / len(answer_labeled), 4)
        if answer_labeled
        else None
    )
    mean_returned_tokens = _mean_numbers([float(diagnostic["returned_tokens"]) for diagnostic in diagnostics])
    mean_injected_tokens = _mean_numbers([float(diagnostic["injected_tokens"]) for diagnostic in diagnostics])
    latencies = [float(diagnostic["latency_ms"]) for diagnostic in diagnostics]
    return {
        "mean_quality": mean_quality,
        "answer_at_5": answer_at_5,
        "recall_at_5": _mean_numbers([float(diagnostic["recall_quality"]) for diagnostic in recall_labeled]),
        "citation_coverage": (
            round(sum(1.0 for diagnostic in diagnostics if diagnostic["citation_hit"]) / len(diagnostics), 4)
            if diagnostics
            else None
        ),
        "mean_returned_tokens": mean_returned_tokens,
        "quality_per_1k_returned_tokens": _per_1k_tokens(mean_quality, mean_returned_tokens),
        "answer_at_5_per_1k_returned_tokens": _per_1k_tokens(answer_at_5, mean_returned_tokens),
        "mean_injected_tokens": mean_injected_tokens,
        "quality_per_1k_injected_tokens": _per_1k_tokens(mean_quality, mean_injected_tokens),
        "answer_at_5_per_1k_injected_tokens": _per_1k_tokens(answer_at_5, mean_injected_tokens),
        "first_checkout_ms": round(latencies[0], 3) if latencies else None,
        "checkout_p95_ms": _nearest_rank_percentile(latencies, 95),
        "checkout_p99_ms": _nearest_rank_percentile(latencies, 99),
    }


def _mean_numbers(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _per_1k_tokens(metric: float | None, mean_tokens: float | None) -> float | None:
    if metric is None or mean_tokens is None or mean_tokens <= 0:
        return None
    return round(metric / mean_tokens * 1000, 4)


def _nearest_rank_percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil((percentile / 100) * len(ordered)) - 1)
    return round(ordered[index], 3)


def _query_diagnostic_item_errors(key: str, diagnostics: list[Any]) -> list[str]:
    errors: list[str] = []
    for index, diagnostic in enumerate(diagnostics):
        label = f"report: query_results[{key!r}][{index}]"
        if not isinstance(diagnostic, dict):
            errors.append(f"{label} must be an object")
            continue
        if not _non_empty_string(diagnostic.get("query")):
            errors.append(f"{label}.query must be a non-empty string")
        for metric in ("quality", "recall_quality"):
            value = diagnostic.get(metric)
            if not _number(value) or not 0 <= float(value) <= 1:
                errors.append(f"{label}.{metric} must be a rate between 0 and 1")
        for metric in ("answer_hit", "recall_hit", "citation_hit"):
            if not isinstance(diagnostic.get(metric), bool):
                errors.append(f"{label}.{metric} must be a boolean")
        errors.extend(_query_hit_consistency_errors(label, diagnostic))
        for metric in ("latency_ms",):
            value = diagnostic.get(metric)
            if not _number(value) or float(value) < 0:
                errors.append(f"{label}.{metric} must be a non-negative number")
        for metric in ("returned_tokens", "injected_tokens"):
            if not _non_negative_int(diagnostic.get(metric)):
                errors.append(f"{label}.{metric} must be a non-negative integer")
        for metric in (
            "expected_terms",
            "identity_terms",
            "source_terms",
            "retrieval_terms",
            "missing_expected_terms",
            "missing_retrieval_terms",
            "top_contexts",
        ):
            if not isinstance(diagnostic.get(metric), list):
                errors.append(f"{label}.{metric} must be a list")
        for metric in (
            "expected_terms",
            "identity_terms",
            "source_terms",
            "retrieval_terms",
            "missing_expected_terms",
            "missing_retrieval_terms",
        ):
            errors.extend(_string_list_item_errors(label, diagnostic, metric))
        errors.extend(_missing_term_consistency_errors(label, diagnostic))
        top_contexts = diagnostic.get("top_contexts")
        if isinstance(top_contexts, list):
            errors.extend(_top_context_diagnostic_errors(label, top_contexts))
            errors.extend(_citation_hit_context_errors(label, diagnostic, top_contexts))
    return errors


def _citation_hit_context_errors(label: str, diagnostic: dict[str, Any], top_contexts: list[Any]) -> list[str]:
    expected_hit = any(isinstance(context, dict) and _non_empty_string(context.get("citation")) for context in top_contexts)
    citation_hit = diagnostic.get("citation_hit")
    if citation_hit == expected_hit:
        return []
    if citation_hit is True:
        return [f"{label}.citation_hit requires at least one cited top_context"]
    return [f"{label}.citation_hit must equal presence of a cited top_context"]


def _query_hit_consistency_errors(label: str, diagnostic: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    quality = diagnostic.get("quality")
    answer_hit = diagnostic.get("answer_hit")
    if _number(quality) and isinstance(answer_hit, bool) and answer_hit != (float(quality) >= 1.0):
        errors.append(f"{label}.answer_hit must equal quality >= 1.0")
    recall_quality = diagnostic.get("recall_quality")
    recall_hit = diagnostic.get("recall_hit")
    if _number(recall_quality) and isinstance(recall_hit, bool) and recall_hit != (float(recall_quality) >= 1.0):
        errors.append(f"{label}.recall_hit must equal recall_quality >= 1.0")
    return errors


def _missing_term_consistency_errors(label: str, diagnostic: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(
        _missing_term_subset_errors(
            label,
            diagnostic,
            missing_metric="missing_expected_terms",
            source_metric="expected_terms",
        )
    )
    errors.extend(
        _missing_term_subset_errors(
            label,
            diagnostic,
            missing_metric="missing_retrieval_terms",
            source_metric="retrieval_terms",
        )
    )
    return errors


def _missing_term_subset_errors(
    label: str,
    diagnostic: dict[str, Any],
    *,
    missing_metric: str,
    source_metric: str,
) -> list[str]:
    missing = diagnostic.get(missing_metric)
    source = diagnostic.get(source_metric)
    if not _string_list(missing) or not _string_list(source):
        return []
    source_terms = set(source)
    return [
        f"{label}.{missing_metric}[{index}] is not present in {source_metric}"
        for index, term in enumerate(missing)
        if term not in source_terms
    ]


def _string_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _string_list_item_errors(label: str, payload: dict[str, Any], metric: str) -> list[str]:
    value = payload.get(metric)
    if not isinstance(value, list):
        return []
    return [
        f"{label}.{metric}[{index}] must be a string"
        for index, item in enumerate(value)
        if not isinstance(item, str)
    ]


def _top_context_diagnostic_errors(query_label: str, top_contexts: list[Any]) -> list[str]:
    errors: list[str] = []
    ranks: list[int] = []
    for index, context in enumerate(top_contexts):
        label = f"{query_label}.top_contexts[{index}]"
        if not isinstance(context, dict):
            errors.append(f"{label} must be an object")
            continue
        rank = context.get("rank")
        if not _positive_int(rank):
            errors.append(f"{label}.rank must be a positive integer")
        elif isinstance(rank, int):
            ranks.append(rank)
        else:
            errors.append(f"{label}.rank must be a positive integer")
        source = context.get("source")
        if not _non_empty_string(source):
            errors.append(f"{label}.source must be a non-empty string")
        elif source not in KNOWN_TOP_CONTEXT_SOURCES:
            errors.append(f"{label}.source is not recognized")
        score = context.get("score")
        if not _number(score) or float(score) < 0:
            errors.append(f"{label}.score must be a non-negative number")
        elif context.get("source") == "vector" and float(score) <= 0:
            errors.append(f"{label}.score must be positive for vector source")
        citation = context.get("citation")
        if citation is not None and not _non_empty_string(citation):
            errors.append(f"{label}.citation must be a non-empty string or null")
        elif isinstance(citation, str) and not _supported_context_citation(citation):
            errors.append(f"{label}.citation must start with eventloom:// or file://")
        elif isinstance(citation, str) and citation.startswith("eventloom://") and not _valid_eventloom_citation(citation):
            errors.append(f"{label}.citation must match eventloom://<thread>/events/<seq>#<hash>")
        elif isinstance(citation, str) and citation.startswith("file://") and not _valid_file_citation(citation):
            errors.append(f"{label}.citation must include a file:// path")
        if not _non_empty_string(context.get("snippet")):
            errors.append(f"{label}.snippet must be a non-empty string")
    if ranks and ranks != list(range(1, len(top_contexts) + 1)):
        errors.append(f"{query_label}.top_contexts ranks must be contiguous from 1")
    return errors


def _supported_context_citation(citation: str) -> bool:
    return citation.startswith(("eventloom://", "file://"))


def _valid_eventloom_citation(citation: str) -> bool:
    return EVENTLOOM_CITATION_RE.fullmatch(citation) is not None


def _valid_file_citation(citation: str) -> bool:
    return bool(citation.removeprefix("file://").strip())


def _summary_result_query_counts(summaries: list[Any]) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        backend = summary.get("backend")
        if backend is None:
            continue
        contract = str(summary.get("contract") or "retrieve")
        query_count = summary.get("query_count")
        key = f"{backend}:{contract}"
        if key in counts:
            continue
        counts[key] = query_count if _non_negative_int(query_count) else None
    return counts


def _schema_version_one(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


def _duplicate_backend_policy_errors(backends: list[str], option: str) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for backend in backends:
        if backend in seen and backend not in duplicates:
            duplicates.append(backend)
        seen.add(backend)
    return [f"{backend}: duplicate backend in {option}" for backend in duplicates]


def _unknown_backend_errors(backends: list[str], option: str) -> list[str]:
    return [f"{backend}: unknown backend in {option}" for backend in backends if backend not in KNOWN_BACKENDS]


def _backend_policy_conflicts(required_backends: list[str], forbidden_backends: list[str]) -> list[str]:
    forbidden = set(forbidden_backends)
    return [
        f"{backend}: backend cannot be both required and forbidden"
        for backend in required_backends
        if backend in forbidden
    ]


def _duplicate_backend_errors(summaries: list[Any]) -> list[str]:
    counts: dict[tuple[str, str], int] = {}
    for summary in summaries:
        if isinstance(summary, dict) and summary.get("backend") is not None:
            backend = str(summary["backend"])
            contract = str(summary.get("contract") or "retrieve")
            key = (backend, contract)
            counts[key] = counts.get(key, 0) + 1
    return [
        f"{backend}:{contract}: duplicate backend summary rows found"
        for (backend, contract), count in counts.items()
        if count > 1
    ]


def _validate_report_metadata(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("report_schema_version") is None:
        errors.append("report: report_schema_version must be 1")
    if not _non_empty_string(report.get("harness")):
        errors.append("report: harness is missing")
    if not _non_empty_string(report.get("generated_at_utc")):
        errors.append("report: generated_at_utc is missing")
    errors.extend(_require_fingerprint(report, "source_fingerprints", "eventloom_sha256"))
    errors.extend(_require_fingerprint(report, "source_fingerprints", "queries_sha256"))
    errors.extend(_require_fingerprint(report, "workload_fingerprints", "events_sha256"))
    errors.extend(_require_fingerprint(report, "workload_fingerprints", "queries_sha256"))
    return errors


def _validate_markdown_report(path: Path, report: dict[str, Any]) -> list[str]:
    try:
        markdown = path.read_text(encoding="utf-8")
    except OSError:
        return [f"report: Markdown sidecar {path} is missing"]
    required = {
        "Report schema version": f"Report schema version: `{report.get('report_schema_version')}`",
        "Harness": f"Harness: `{report.get('harness')}`",
        "Generated at UTC": f"Generated at UTC: `{report.get('generated_at_utc')}`",
        "Eventloom path": f"Eventloom path: `{report.get('eventloom_path')}`",
        "Queries file": f"Queries file: `{report.get('queries_file')}`",
        "Queries": f"Queries: `{report.get('query_count')}`",
        "Events": f"Events: `{report.get('event_count')}`",
        "Limit": f"Limit: `{report.get('limit')}`",
        "Source Eventloom SHA-256": _nested_string(report, "source_fingerprints", "eventloom_sha256"),
        "Source queries SHA-256": _nested_string(report, "source_fingerprints", "queries_sha256"),
        "Workload events SHA-256": _nested_string(report, "workload_fingerprints", "events_sha256"),
        "Workload queries SHA-256": _nested_string(report, "workload_fingerprints", "queries_sha256"),
    }
    return [
        f"report: Markdown sidecar missing {label}"
        for label, snippet in required.items()
        if not snippet or snippet not in markdown
    ] + _missing_markdown_backend_rows(markdown, report)


def _missing_markdown_backend_rows(markdown: str, report: dict[str, Any]) -> list[str]:
    summaries = report.get("summaries")
    if not isinstance(summaries, list):
        return []
    errors: list[str] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        backend = summary.get("backend")
        status = summary.get("status")
        if backend is None or status is None:
            continue
        contract = str(summary.get("contract") or "retrieve")
        row_prefix = f"| {backend} | {contract} | {status} |"
        row = _markdown_row(markdown, row_prefix)
        if row is None and ("contract" not in summary or "| Contract |" not in markdown):
            row = _markdown_row(markdown, f"| {backend} | {status} |")
        if row is None:
            backend_label = f"{backend}:{contract}" if "contract" in summary else str(backend)
            errors.append(f"report: Markdown sidecar missing backend row for {backend_label}")
            continue
        header = _markdown_header_for_row(markdown, row)
        backend_label = f"{backend}:{contract}" if "contract" in summary else str(backend)
        for metric in _MARKDOWN_METRIC_COLUMNS:
            value = summary.get(metric)
            if value is not None and _markdown_metric_value(row, header, metric) != str(value):
                errors.append(f"report: Markdown sidecar row for {backend_label} missing {metric}={value}")
    return errors


def _markdown_row(markdown: str, prefix: str) -> str | None:
    return next((line for line in markdown.splitlines() if line.startswith(prefix)), None)


def _markdown_header_for_row(markdown: str, row: str) -> str | None:
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line != row:
            continue
        for separator_index in range(index - 1, 0, -1):
            if _markdown_separator_row(lines[separator_index]) and lines[separator_index - 1].startswith("|"):
                return lines[separator_index - 1]
    return None


def _markdown_separator_row(row: str) -> bool:
    return all(char in "| :-" for char in row.strip())


def _markdown_metric_value(row: str, header: str | None, metric: str) -> str | None:
    if header is None:
        return None
    header_cells = _markdown_cells(header)
    row_cells = _markdown_cells(row)
    column_name = _MARKDOWN_METRIC_COLUMNS[metric]
    try:
        index = header_cells.index(column_name)
    except ValueError:
        return None
    if index >= len(row_cells):
        return None
    return row_cells[index]


def _markdown_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


_MARKDOWN_METRIC_COLUMNS = {
    "cold_bootstrap_ms": "Cold bootstrap ms",
    "first_useful_init_ms": "First useful init ms",
    "first_checkout_ms": "First checkout ms",
    "append_to_projection_p95_ms": "Append projection p95 ms",
    "projection_events_per_second": "Projection eps",
    "checkout_p95_ms": "Checkout p95 ms",
    "checkout_p99_ms": "Checkout p99 ms",
    "exact_p50_ms": "Exact p50 ms",
    "exact_p95_ms": "Exact p95 ms",
    "exact_p99_ms": "Exact p99 ms",
    "keyword_p50_ms": "Keyword p50 ms",
    "keyword_p95_ms": "Keyword p95 ms",
    "keyword_p99_ms": "Keyword p99 ms",
    "vector_p50_ms": "Vector p50 ms",
    "vector_p95_ms": "Vector p95 ms",
    "vector_p99_ms": "Vector p99 ms",
    "traversal_p50_ms": "Traversal p50 ms",
    "traversal_p95_ms": "Traversal p95 ms",
    "traversal_p99_ms": "Traversal p99 ms",
    "dashboard_graph_load_ms": "Dashboard graph load ms",
    "dashboard_graph_source": "Dashboard source",
    "dashboard_graph_nodes": "Dashboard nodes",
    "dashboard_graph_edges": "Dashboard edges",
    "mean_returned_tokens": "Returned tokens",
    "quality_per_1k_returned_tokens": "Quality / 1k tokens",
    "answer_at_5_per_1k_returned_tokens": "Answer@5 / 1k tokens",
    "mean_injected_tokens": "Injected tokens",
    "quality_per_1k_injected_tokens": "Quality / 1k injected",
    "answer_at_5_per_1k_injected_tokens": "Answer@5 / 1k injected",
    "answer_at_5": "Answer@5",
    "recall_at_5": "Recall@5",
    "citation_coverage": "Citation coverage",
    "mean_quality": "Mean quality",
    "memory_footprint_bytes": "Memory bytes",
    "resident_memory_delta_bytes": "Resident memory delta bytes",
    "on_disk_footprint_bytes": "On-disk footprint bytes",
    "rebuild_recovery_ms": "Rebuild recovery ms",
}


def _nested_string(report: dict[str, Any], section: str, key: str) -> str | None:
    payload = report.get(section)
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _verified_query_count(report: dict[str, Any]) -> int | None:
    value = report.get("query_count")
    return value if _non_negative_int(value) else None


def _verify_report_fingerprints(report: dict[str, Any]) -> list[str]:
    harness = _load_backend_shootout_harness()
    eventloom_path = report.get("eventloom_path")
    queries_file = report.get("queries_file")
    session_id = report.get("session_id")
    if not isinstance(eventloom_path, str) or not eventloom_path:
        return ["report: eventloom_path is missing"]
    if not isinstance(queries_file, str) or not queries_file:
        return ["report: queries_file is missing"]
    if not isinstance(session_id, str) or not session_id:
        return ["report: session_id is missing"]
    errors: list[str] = []
    eventloom = Path(eventloom_path)
    queries = Path(queries_file)
    errors.extend(
        _compare_fingerprint(
            report,
            section="source_fingerprints",
            key="eventloom_sha256",
            current=harness._path_fingerprint(eventloom),
        )
    )
    errors.extend(
        _compare_fingerprint(
            report,
            section="source_fingerprints",
            key="queries_sha256",
            current=harness._path_fingerprint(queries),
        )
    )
    events = harness._load_events(eventloom, session_id)
    query_specs = harness._load_queries(queries)
    errors.extend(_compare_count(report, key="event_count", current=len(events)))
    errors.extend(_compare_count(report, key="query_count", current=len(query_specs)))
    errors.extend(
        _compare_fingerprint(
            report,
            section="workload_fingerprints",
            key="events_sha256",
            current=harness._events_fingerprint(events),
        )
    )
    errors.extend(
        _compare_fingerprint(
            report,
            section="workload_fingerprints",
            key="queries_sha256",
            current=harness._queries_fingerprint(query_specs),
        )
    )
    return errors


def _verify_query_result_workload(report: dict[str, Any]) -> list[str]:
    query_results = report.get("query_results")
    queries_file = report.get("queries_file")
    if not isinstance(query_results, dict) or not isinstance(queries_file, str) or not queries_file:
        return []
    harness = _load_backend_shootout_harness()
    query_specs = harness._load_queries(Path(queries_file))
    expected_queries = [spec.query for spec in query_specs]
    expected_terms = [list(spec.expected_terms) for spec in query_specs]
    expected_identity_terms = [list(spec.identity_terms) for spec in query_specs]
    expected_source_terms = [list(spec.source_terms) for spec in query_specs]
    errors: list[str] = []
    for key, diagnostics in query_results.items():
        if not isinstance(diagnostics, list):
            continue
        for index, diagnostic in enumerate(diagnostics[: len(query_specs)]):
            if not isinstance(diagnostic, dict):
                continue
            label = f"report: query_results[{key!r}][{index}]"
            if diagnostic.get("query") != expected_queries[index]:
                errors.append(f"{label}.query does not match current query workload")
            if diagnostic.get("expected_terms") != expected_terms[index]:
                errors.append(f"{label}.expected_terms does not match current query workload")
            if diagnostic.get("identity_terms") != expected_identity_terms[index]:
                errors.append(f"{label}.identity_terms does not match current query workload")
            if diagnostic.get("source_terms") != expected_source_terms[index]:
                errors.append(f"{label}.source_terms does not match current query workload")
    return errors


def _validate_git_tracked_inputs(report: dict[str, Any], report_path: Path) -> list[str]:
    errors: list[str] = []
    root = _git_root(report_path)
    if root is None:
        return ["report: git root could not be resolved"]
    for label in ("eventloom_path", "queries_file"):
        value = report.get(label)
        if not isinstance(value, str) or not value:
            errors.append(f"report: {label} is missing")
            continue
        path = Path(value)
        candidate = path if path.is_absolute() else root / path
        if not _git_path_is_tracked(root, candidate):
            errors.append(f"report: {label} {value} is not tracked by git")
    return errors


def _git_root(path: Path) -> Path | None:
    for cwd in (path.parent if path.parent.exists() else Path.cwd(), Path.cwd()):
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).resolve()
    return None


def _git_path_is_tracked(root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _load_backend_shootout_harness() -> Any:
    path = Path(__file__).with_name("backend-shootout.py")
    spec = importlib.util.spec_from_file_location("backend_shootout", path)
    if spec is None or spec.loader is None:
        raise SystemExit("could not load scripts/backend-shootout.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["backend_shootout"] = module
    spec.loader.exec_module(module)
    return module


def _compare_fingerprint(report: dict[str, Any], *, section: str, key: str, current: str) -> list[str]:
    payload = report.get(section)
    if not isinstance(payload, dict) or payload.get(key) != current:
        return [f"report: {section}.{key} does not match current input"]
    return []


def _compare_count(report: dict[str, Any], *, key: str, current: int) -> list[str]:
    value = report.get(key)
    if not _non_negative_int(value) or value != current:
        return [f"report: {key}={value} does not match current input count {current}"]
    return []


def _require_fingerprint(report: dict[str, Any], section: str, key: str) -> list[str]:
    payload = report.get(section)
    if not isinstance(payload, dict) or not _sha256_string(payload.get(key)):
        return [f"report: {section}.{key} is missing"]
    return []


def _sha256_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _dashboard_source_requirements(values: list[str]) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--require-dashboard-source must use BACKEND=SOURCE")
        backend, source = value.split("=", 1)
        backend = backend.strip()
        source = source.strip()
        if not backend or not source:
            raise SystemExit("--require-dashboard-source must use BACKEND=SOURCE")
        if backend in requirements:
            raise SystemExit(f"{backend}: duplicate backend requirement in --require-dashboard-source")
        requirements[backend] = source
    return requirements


def _metric_requirements(values: list[str], option: str) -> dict[str, float]:
    requirements: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} must use BACKEND=FLOAT")
        backend, raw_threshold = value.split("=", 1)
        backend = backend.strip()
        raw_threshold = raw_threshold.strip()
        if not backend or not raw_threshold:
            raise SystemExit(f"{option} must use BACKEND=FLOAT")
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise SystemExit(f"{option} threshold must be a number") from exc
        if not math.isfinite(threshold):
            raise SystemExit(f"{option} threshold must be a finite number")
        if threshold < 0:
            raise SystemExit(f"{option} threshold must be non-negative")
        if backend in requirements:
            raise SystemExit(f"{backend}: duplicate backend threshold in {option}")
        requirements[backend] = threshold
    return requirements


def _non_negative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("minimum must be a number") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("minimum must be finite")
    if value < 0:
        raise argparse.ArgumentTypeError("minimum must be non-negative")
    if value > 1:
        raise argparse.ArgumentTypeError("minimum must be at most 1.0")
    return value


def _require_metric(summary: dict[str, Any], backend: str, metric: str) -> list[str]:
    value = summary.get(metric)
    if _number(value):
        return []
    return [f"{backend}: {metric} is missing"]


def _check_floor(
    summary: dict[str, Any],
    backend: str,
    metric: str,
    floor: float | None,
) -> list[str]:
    if floor is None:
        return []
    value = summary.get(metric)
    if not _number(value):
        return [f"{backend}: {metric} is missing"]
    if float(value) < floor:
        return [f"{backend}: {metric}={value} is below {floor}"]
    return []


def _check_rate_metric(summary: dict[str, Any], backend: str, metric: str) -> list[str]:
    value = summary.get(metric)
    if value is None:
        return []
    if not _number(value):
        return [f"{backend}: {metric} is missing"]
    if not 0 <= float(value) <= 1:
        return [f"{backend}: {metric}={value} must be between 0 and 1"]
    return []


def _check_non_negative_metric(summary: dict[str, Any], backend: str, metric: str) -> list[str]:
    value = summary.get(metric)
    if value is None:
        return []
    if not _number(value):
        return [f"{backend}: {metric} must be a non-negative number"]
    if float(value) < 0:
        return [f"{backend}: {metric}={value} must be non-negative"]
    return []


def _check_integer_metric(summary: dict[str, Any], backend: str, metric: str) -> list[str]:
    value = summary.get(metric)
    if value is None:
        return []
    if not isinstance(value, int) or isinstance(value, bool):
        return [f"{backend}: {metric}={value} must be an integer"]
    return []


def _check_non_empty_string_metric(summary: dict[str, Any], backend: str, metric: str) -> list[str]:
    value = summary.get(metric)
    if value is None:
        return []
    if not _non_empty_string(value):
        return [f"{backend}: {metric} must be a non-empty string"]
    return []


def _check_backend_floor(
    summary: dict[str, Any],
    backend: str,
    metric: str,
    floors: dict[str, float],
) -> list[str]:
    floor = floors.get(backend)
    if floor is None:
        return []
    return _check_floor(summary, backend, metric, floor)


def _check_backend_ceiling(
    summary: dict[str, Any],
    backend: str,
    metric: str,
    ceilings: dict[str, float],
) -> list[str]:
    ceiling = ceilings.get(backend)
    if ceiling is None:
        return []
    value = summary.get(metric)
    if not _number(value):
        return [f"{backend}: {metric} is missing"]
    if float(value) > ceiling:
        return [f"{backend}: {metric}={value} is above {ceiling}"]
    return []


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


if __name__ == "__main__":
    main()
