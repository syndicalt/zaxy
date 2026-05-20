#!/usr/bin/env python3
"""Validate backend shootout reports for release-gate use."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

KNOWN_BACKENDS = {"embedded", "latticedb", "neo4j", "pggraph", "bm25"}
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
        "--verify-report-fingerprints",
        action="store_true",
        help="Recompute source/workload fingerprints from the report inputs and reject stale reports",
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
    by_backend = {
        str(summary.get("backend")): summary
        for summary in summaries
        if isinstance(summary, dict) and summary.get("backend") is not None
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
    errors.extend(_backend_policy_conflicts(required_backends, forbidden_backends))
    verified_query_count = _verified_query_count(report) if args.verify_report_fingerprints else None
    if args.require_report_metadata:
        errors.extend(_validate_report_metadata(report))
    if args.require_markdown_report:
        errors.extend(_validate_markdown_report(args.report.with_suffix(".md"), report))
    if args.verify_report_fingerprints:
        errors.extend(_verify_report_fingerprints(report))
    for backend in forbidden_backends:
        if backend in by_backend:
            errors.append(f"{backend}: forbidden backend present in report")
    for backend in required_backends:
        summary = by_backend.get(backend)
        if summary is None:
            errors.append(f"{backend}: missing required backend summary")
            continue
        if summary.get("status") != "ok":
            errors.append(f"{backend}: status is {summary.get('status')!r}, expected 'ok'")
            continue
        if verified_query_count is not None and summary.get("query_count") != verified_query_count:
            errors.append(
                f"{backend}: query_count={summary.get('query_count')} "
                f"does not match current input count {verified_query_count}"
            )
        if args.require_labeled_metrics:
            errors.extend(_require_metric(summary, backend, "answer_at_5"))
            errors.extend(_require_metric(summary, backend, "recall_at_5"))
        errors.extend(_check_floor(summary, backend, "answer_at_5", args.min_answer_at_5))
        errors.extend(_check_floor(summary, backend, "recall_at_5", args.min_recall_at_5))
        errors.extend(_check_floor(summary, backend, "citation_coverage", args.min_citation_coverage))
        expected_source = dashboard_sources.get(backend)
        if expected_source is not None and summary.get("dashboard_graph_source") != expected_source:
            errors.append(
                f"{backend}: dashboard_graph_source is {summary.get('dashboard_graph_source')!r}, "
                f"expected {expected_source!r}"
            )
        errors.extend(_check_backend_floor(summary, backend, "projection_events_per_second", min_projection_eps))
        errors.extend(_check_backend_ceiling(summary, backend, "cold_bootstrap_ms", max_cold_bootstrap_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "first_useful_init_ms", max_first_init_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "first_checkout_ms", max_first_checkout_ms))
        errors.extend(
            _check_backend_ceiling(
                summary,
                backend,
                "append_to_projection_p95_ms",
                max_append_projection_p95_ms,
            )
        )
        errors.extend(
            _check_backend_ceiling(
                summary,
                backend,
                "resident_memory_delta_bytes",
                max_resident_memory_delta_bytes,
            )
        )
        errors.extend(
            _check_backend_ceiling(
                summary,
                backend,
                "on_disk_footprint_bytes",
                max_on_disk_footprint_bytes,
            )
        )
        errors.extend(_check_backend_ceiling(summary, backend, "rebuild_recovery_ms", max_rebuild_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "checkout_p95_ms", max_checkout_p95_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "checkout_p99_ms", max_checkout_p99_ms))
        errors.extend(
            _check_backend_ceiling(
                summary,
                backend,
                "dashboard_graph_load_ms",
                max_dashboard_graph_load_ms,
            )
        )
        errors.extend(
            _check_backend_floor(
                summary,
                backend,
                "quality_per_1k_returned_tokens",
                min_quality_per_1k_tokens,
            )
        )
        errors.extend(
            _check_backend_floor(
                summary,
                backend,
                "answer_at_5_per_1k_returned_tokens",
                min_answer_at_5_per_1k_tokens,
            )
        )
        errors.extend(
            _check_backend_floor(
                summary,
                backend,
                "quality_per_1k_injected_tokens",
                min_quality_per_1k_injected_tokens,
            )
        )
        errors.extend(
            _check_backend_floor(
                summary,
                backend,
                "answer_at_5_per_1k_injected_tokens",
                min_answer_at_5_per_1k_injected_tokens,
            )
        )
        errors.extend(_check_backend_ceiling(summary, backend, "exact_p95_ms", max_exact_p95_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "exact_p99_ms", max_exact_p99_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "keyword_p95_ms", max_keyword_p95_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "keyword_p99_ms", max_keyword_p99_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "vector_p95_ms", max_vector_p95_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "vector_p99_ms", max_vector_p99_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "traversal_p95_ms", max_traversal_p95_ms))
        errors.extend(_check_backend_ceiling(summary, backend, "traversal_p99_ms", max_traversal_p99_ms))
    return errors


def _load_report(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON contains non-standard numeric constant {value}")

    report = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(report, dict):
        raise ValueError("top-level report must be an object")
    return report


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
    counts: dict[str, int] = {}
    for summary in summaries:
        if isinstance(summary, dict) and summary.get("backend") is not None:
            backend = str(summary["backend"])
            counts[backend] = counts.get(backend, 0) + 1
    return [f"{backend}: duplicate backend summary rows found" for backend, count in counts.items() if count > 1]


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
        row_prefix = f"| {backend} | {status} |"
        row = _markdown_row(markdown, row_prefix)
        if row is None:
            errors.append(f"report: Markdown sidecar missing backend row for {backend}")
            continue
        for metric in (
            "answer_at_5",
            "recall_at_5",
            "citation_coverage",
            "quality_per_1k_injected_tokens",
            "answer_at_5_per_1k_injected_tokens",
        ):
            value = summary.get(metric)
            if value is not None and f"| {value} |" not in row:
                errors.append(f"report: Markdown sidecar row for {backend} missing {metric}={value}")
    return errors


def _markdown_row(markdown: str, prefix: str) -> str | None:
    return next((line for line in markdown.splitlines() if line.startswith(prefix)), None)


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


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


if __name__ == "__main__":
    main()
