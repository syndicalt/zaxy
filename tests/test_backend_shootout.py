"""Tests for the backend shootout harness."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from zaxy.event import EventLog


def _load_backend_shootout_module():
    spec = importlib.util.spec_from_file_location("backend_shootout", "scripts/backend-shootout.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["backend_shootout"] = module
    spec.loader.exec_module(module)
    return module


def _load_backend_workload_builder_module():
    spec = importlib.util.spec_from_file_location(
        "build_backend_shootout_workload",
        "scripts/build-backend-shootout-workload.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_backend_shootout_workload"] = module
    spec.loader.exec_module(module)
    return module


def _load_backend_shootout_check_module():
    spec = importlib.util.spec_from_file_location("check_backend_shootout", "scripts/check-backend-shootout.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["check_backend_shootout"] = module
    spec.loader.exec_module(module)
    return module


def test_backend_shootout_default_active_backends_exclude_latticedb_candidate() -> None:
    """The default active shootout should not spend time on a parked candidate backend."""
    module = _load_backend_shootout_module()

    assert "latticedb" in module.SUPPORTED_BACKENDS
    assert "latticedb" not in module.DEFAULT_BACKENDS


def test_backend_shootout_parses_linux_statm_current_rss_bytes() -> None:
    """Resident memory deltas should use current RSS, not the process high-water mark."""
    module = _load_backend_shootout_module()

    assert module._rss_bytes_from_linux_statm("123 45 0 0 0 0 0\n", page_size=4096) == 184320
    assert module._rss_bytes_from_linux_statm("", page_size=4096) is None
    assert module._rss_bytes_from_linux_statm("123 not-a-number", page_size=4096) is None


def test_backend_shootout_tracks_max_current_rss_delta() -> None:
    """Backend reports should preserve the largest observed current-RSS delta."""
    module = _load_backend_shootout_module()
    tracker = module._ResidentMemoryTracker(start_bytes=1_000)

    tracker.observe(1_500)
    tracker.observe(1_250)
    tracker.observe(2_100)

    assert tracker.delta_bytes() == 1_100
    assert module._ResidentMemoryTracker(start_bytes=None).delta_bytes() is None


def test_backend_shootout_dashboard_count_coercion_rejects_booleans() -> None:
    """Dashboard graph counts should not treat booleans as integer node or edge counts."""
    module = _load_backend_shootout_module()

    assert module._int_or_none(True) is None
    assert module._int_or_none(False) is None


def test_backend_shootout_dashboard_count_coercion_rejects_negative_counts() -> None:
    """Dashboard graph counts should be non-negative."""
    module = _load_backend_shootout_module()

    assert module._int_or_none(-1) is None
    assert module._int_or_none("-2") is None
    assert module._int_or_none(0) == 0


def test_backend_shootout_dashboard_count_coercion_rejects_fractional_counts() -> None:
    """Dashboard graph counts should not truncate fractional values."""
    module = _load_backend_shootout_module()

    assert module._int_or_none(1.9) is None
    assert module._int_or_none("2.5") is None
    assert module._int_or_none(2.0) == 2


def test_backend_shootout_report_json_writer_rejects_non_finite_values() -> None:
    """The report generator should not emit non-standard JSON constants."""
    module = _load_backend_shootout_module()

    with pytest.raises(ValueError):
        module._strict_json_dumps({"answer_at_5": float("nan")})


def test_backend_shootout_runs_bm25_and_writes_reports(tmp_path: Path) -> None:
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "agent-1.jsonl")
    log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use embedded graph for zero-friction local memory."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "embedded graph memory", "expected_terms": ["embedded graph"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["report_schema_version"] == 1
    assert payload["harness"] == "zaxy-backend-shootout"
    assert payload["generated_at_utc"].endswith("Z")
    assert payload["queries_file"] == str(queries)
    assert payload["source_fingerprints"]["eventloom_sha256"]
    assert payload["source_fingerprints"]["queries_sha256"]
    assert payload["workload_fingerprints"]["events_sha256"]
    assert payload["workload_fingerprints"]["queries_sha256"]
    assert len(payload["workload_fingerprints"]["events_sha256"]) == 64
    assert len(payload["workload_fingerprints"]["queries_sha256"]) == 64
    assert payload["event_count"] == 1
    assert payload["query_count"] == 1
    assert payload["summaries"][0]["backend"] == "bm25"
    assert payload["summaries"][0]["status"] == "ok"
    assert payload["summaries"][0]["mean_quality"] == 1.0
    assert payload["summaries"][0]["cold_bootstrap_ms"] is not None
    assert payload["summaries"][0]["first_checkout_ms"] is not None
    assert payload["summaries"][0]["append_to_projection_p95_ms"] is None
    assert payload["summaries"][0]["quality_per_1k_returned_tokens"] is not None
    assert payload["summaries"][0]["answer_at_5_per_1k_returned_tokens"] is not None
    assert payload["summaries"][0]["mean_injected_tokens"] is not None
    assert payload["summaries"][0]["quality_per_1k_injected_tokens"] is not None
    assert payload["summaries"][0]["answer_at_5_per_1k_injected_tokens"] is not None
    assert payload["summaries"][0]["exact_p50_ms"] is None
    assert payload["summaries"][0]["keyword_p95_ms"] is None
    assert payload["summaries"][0]["vector_p99_ms"] is None
    assert payload["summaries"][0]["projection_events_per_second"] is None
    assert payload["summaries"][0]["traversal_p95_ms"] is None
    assert payload["summaries"][0]["dashboard_graph_load_ms"] is None
    assert payload["summaries"][0]["dashboard_graph_source"] is None
    assert payload["summaries"][0]["dashboard_graph_nodes"] is None
    assert payload["summaries"][0]["dashboard_graph_edges"] is None
    assert payload["summaries"][0]["memory_footprint_bytes"] is not None
    assert payload["summaries"][0]["memory_footprint_bytes"] > 0
    assert payload["summaries"][0]["resident_memory_delta_bytes"] is not None
    assert payload["summaries"][0]["on_disk_footprint_bytes"] == 0
    assert payload["summaries"][0]["rebuild_recovery_ms"] == 0.0
    assert payload["summaries"][0]["answer_at_5"] == 1.0
    assert payload["summaries"][0]["recall_at_5"] == 1.0
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "Report schema version: `1`" in markdown
    assert "Harness: `zaxy-backend-shootout`" in markdown
    assert f"Queries file: `{queries}`" in markdown
    assert "Queries: `1`" in markdown
    assert "Events: `1`" in markdown
    assert "Limit: `5`" in markdown
    assert "Source Eventloom SHA-256:" in markdown
    assert "Source queries SHA-256:" in markdown
    assert "Workload events SHA-256:" in markdown
    assert "Workload queries SHA-256:" in markdown
    assert "| bm25 | ok |" in markdown
    assert "Checkout p95 ms" in markdown
    assert "Cold bootstrap ms" in markdown
    assert "First checkout ms" in markdown
    assert "Append projection p95 ms" in markdown
    assert "Projection eps" in markdown
    assert "Memory bytes" in markdown
    assert "Resident memory delta bytes" in markdown
    assert "On-disk footprint bytes" in markdown
    assert "Dashboard source" in markdown
    assert "Quality / 1k tokens" in markdown
    assert "Answer@5 / 1k tokens" in markdown
    assert "Injected tokens" in markdown
    assert "Quality / 1k injected" in markdown
    assert "Answer@5 / 1k injected" in markdown
    assert "Exact p50 ms" in markdown
    assert "Keyword p95 ms" in markdown
    assert "Vector p99 ms" in markdown
    assert "| bm25 | ok |" in markdown
    assert "| 1.0 | 1.0 |" in markdown


@pytest.mark.skipif(importlib.util.find_spec("latticedb") is None, reason="latticedb is not installed")
def test_backend_shootout_runs_latticedb_candidate_when_installed(tmp_path: Path) -> None:
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Evaluate LatticeDB as an embedded backend."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.txt"
    queries.write_text("LatticeDB embedded backend\n", encoding="utf-8")
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "latticedb",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))["summaries"][0]
    assert summary["backend"] == "latticedb"
    assert summary["status"] == "ok"
    assert summary["query_count"] == 1
    assert summary["projection_events_per_second"] is not None
    assert summary["traversal_p95_ms"] is not None
    assert summary["dashboard_graph_load_ms"] is not None
    assert summary["memory_footprint_bytes"] is not None
    assert summary["resident_memory_delta_bytes"] is not None
    assert summary["on_disk_footprint_bytes"] is not None
    assert summary["rebuild_recovery_ms"] is not None
    assert summary["answer_at_5"] is None
    assert summary["recall_at_5"] is None


def test_backend_shootout_checked_sample_workload_runs_bm25(tmp_path: Path) -> None:
    """The checked sample workload should stay runnable from a clean checkout."""
    output = tmp_path / "sample-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            "reports/backend-shootout/sample.eventloom",
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            "reports/backend-shootout/queries.json",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["event_count"] >= 4
    assert payload["query_count"] >= 3
    assert payload["summaries"][0]["status"] == "ok"
    assert payload["summaries"][0]["mean_quality"] is not None
    assert payload["summaries"][0]["answer_at_5"] is not None
    assert payload["summaries"][0]["recall_at_5"] is not None


def test_backend_shootout_rejects_non_positive_limit(tmp_path: Path) -> None:
    """Backend shootout runs should reject limits that cannot produce useful context."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject non-positive backend shootout limits."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "limit validation", "expected_terms": ["limit"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
            "--limit",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "limit must be a positive integer" in result.stderr


def test_backend_shootout_rejects_empty_backend_selection(tmp_path: Path) -> None:
    """Backend shootout runs should produce at least one backend summary."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject empty backend shootout selections."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "backend validation", "expected_terms": ["backend"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "At least one backend must be selected" in result.stderr


def test_backend_shootout_rejects_duplicate_backend_selection(tmp_path: Path) -> None:
    """Backend shootout runs should not produce duplicate backend summary rows."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject duplicate backend shootout selections."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "backend duplication", "expected_terms": ["backend"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25,bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Duplicate backend selection(s): bm25" in result.stderr


def test_backend_shootout_rejects_non_standard_json_query_constants(tmp_path: Path) -> None:
    """Backend shootout query JSON should not accept Python-specific numeric constants."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject non-standard query JSON constants."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        '[{"query": "strict query json", "expected_terms": ["strict"], "score": NaN}]',
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "queries file contains non-standard numeric constant NaN" in result.stderr


def test_backend_shootout_rejects_malformed_json_query_files(tmp_path: Path) -> None:
    """Malformed JSON-looking query files should not fall back to line-delimited mode."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject malformed JSON query files."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text('[{"query": "broken json"', encoding="utf-8")
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "queries file contains malformed JSON" in result.stderr


def test_backend_shootout_rejects_empty_json_query_entries(tmp_path: Path) -> None:
    """Backend shootout query JSON should not materialize empty benchmark queries."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject empty benchmark query entries."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "   ", "expected_terms": []}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "query entries must not be empty" in result.stderr


def test_backend_shootout_rejects_empty_query_workload(tmp_path: Path) -> None:
    """Backend shootout runs should not produce zero-query benchmark reports."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject empty benchmark workloads."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text("[]", encoding="utf-8")
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "queries file must contain at least one query" in result.stderr


def test_backend_shootout_rejects_empty_expected_terms(tmp_path: Path) -> None:
    """Backend shootout expected terms should be meaningful non-empty strings."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject empty expected benchmark terms."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "strict expected terms", "expected_terms": ["  "]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "expected_terms must contain non-empty strings" in result.stderr


@pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")
def test_backend_shootout_measures_real_embedded_dashboard_graph_load(tmp_path: Path) -> None:
    """Dashboard graph load metrics should come from the dashboard provider, not projection-status proxies."""
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "agent-1.jsonl")
    log.append(
        "task.proposed",
        actor="assistant",
        payload={
            "taskId": "embedded-dashboard-task",
            "title": "Wire embedded dashboard graph rendering",
            "goalTitle": "Embedded Dashboard Goal",
        },
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            [
                {
                    "query": "embedded dashboard graph rendering",
                    "expected_terms": ["embedded dashboard"],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "embedded",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))["summaries"][0]
    assert summary["backend"] == "embedded"
    assert summary["status"] == "ok"
    assert summary["dashboard_graph_load_ms"] is not None
    assert summary["dashboard_graph_source"] == "embedded"
    assert summary["dashboard_graph_nodes"] >= 2
    assert summary["dashboard_graph_edges"] >= 1
    assert summary["cold_bootstrap_ms"] is not None
    assert summary["first_checkout_ms"] is not None
    assert summary["append_to_projection_p95_ms"] is not None
    assert summary["resident_memory_delta_bytes"] is not None
    assert summary["on_disk_footprint_bytes"] is not None
    assert summary["exact_p50_ms"] is not None
    assert summary["keyword_p95_ms"] is not None
    assert summary["vector_p99_ms"] is not None
    assert summary["traversal_p50_ms"] is not None
    assert summary["traversal_p99_ms"] is not None
    assert summary["answer_at_5"] == 1.0
    assert summary["recall_at_5"] == 1.0


def test_backend_shootout_guardrail_rejects_missing_labeled_metrics(tmp_path: Path) -> None:
    """The guardrail should fail when active backend quality fields are unmeasured."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": None,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "dashboard_graph_source": "embedded",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--require-labeled-metrics",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: answer_at_5 is missing" in result.stderr


def test_backend_shootout_guardrail_rejects_negative_global_metric_floor(tmp_path: Path) -> None:
    """A negative global quality floor should not silently weaken a guardrail."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 0.0,
                        "recall_at_5": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--min-answer-at-5",
            "-1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "minimum must be non-negative" in result.stderr


def test_backend_shootout_guardrail_rejects_impossible_global_metric_floor(tmp_path: Path) -> None:
    """Global rate floors should not exceed 1.0."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--min-answer-at-5",
            "1.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "minimum must be at most 1.0" in result.stderr


def test_backend_shootout_guardrail_rejects_impossible_summary_rate_metric(tmp_path: Path) -> None:
    """Finite report rates should still stay within the possible 0..1 interval."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 1.2,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--require-labeled-metrics",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: answer_at_5=1.2 must be between 0 and 1" in result.stderr


def test_backend_shootout_guardrail_rejects_impossible_rate_metric_on_non_required_row(tmp_path: Path) -> None:
    """Malformed quality metrics should invalidate any backend row in the report."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                    },
                    {
                        "backend": "bm25",
                        "status": "ok",
                        "answer_at_5": 1.0,
                        "recall_at_5": -0.1,
                        "citation_coverage": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--require-labeled-metrics",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25: recall_at_5=-0.1 must be between 0 and 1" in result.stderr


def test_backend_shootout_guardrail_rejects_impossible_mean_quality_metric(tmp_path: Path) -> None:
    """Mean quality is generated from bounded quality scores and should stay within 0..1."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "mean_quality": 1.2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: mean_quality=1.2 must be between 0 and 1" in result.stderr


def test_backend_shootout_guardrail_rejects_negative_summary_performance_metric(tmp_path: Path) -> None:
    """Malformed performance metrics should invalidate any backend row in the report."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                    },
                    {
                        "backend": "bm25",
                        "status": "ok",
                        "checkout_p95_ms": -0.01,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25: checkout_p95_ms=-0.01 must be non-negative" in result.stderr


def test_backend_shootout_guardrail_rejects_fractional_summary_count_metric(tmp_path: Path) -> None:
    """Generated count metrics should not accept fractional numeric values."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "dashboard_graph_nodes": 2.5,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: dashboard_graph_nodes=2.5 must be an integer" in result.stderr


def test_backend_shootout_guardrail_rejects_negative_legacy_summary_query_count(tmp_path: Path) -> None:
    """Legacy report rows should not accept negative query counts."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "query_count": -1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: query_count=-1 must be non-negative" in result.stderr


def test_backend_shootout_guardrail_rejects_non_string_dashboard_graph_source(tmp_path: Path) -> None:
    """Dashboard graph source metadata should be string-shaped when present."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "dashboard_graph_source": 123,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: dashboard_graph_source must be a non-empty string" in result.stderr


def test_backend_shootout_guardrail_rejects_boolean_labeled_metrics(tmp_path: Path) -> None:
    """JSON booleans should not satisfy numeric quality metric requirements."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": True,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "dashboard_graph_source": "embedded",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--require-labeled-metrics",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: answer_at_5 is missing" in result.stderr


def test_backend_shootout_guardrail_rejects_non_finite_labeled_metrics(tmp_path: Path) -> None:
    """Non-finite report values should be rejected as non-standard JSON constants."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": float("nan"),
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "dashboard_graph_source": "embedded",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--require-labeled-metrics",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: JSON contains non-standard numeric constant NaN" in result.stderr


def test_backend_shootout_guardrail_rejects_forbidden_candidate_backend(tmp_path: Path) -> None:
    """Release evidence should be able to reject parked candidate backends."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                    {"backend": "latticedb", "status": "error"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--forbid-backends",
            "latticedb",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "latticedb: forbidden backend present in report" in result.stderr


def test_backend_shootout_guardrail_rejects_conflicting_required_and_forbidden_backend(tmp_path: Path) -> None:
    """Release policy should not both require and forbid the same backend."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "latticedb", "status": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "latticedb",
            "--forbid-backends",
            "latticedb",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "latticedb: backend cannot be both required and forbidden" in result.stderr


def test_backend_shootout_guardrail_rejects_unknown_forbidden_backend_name(tmp_path: Path) -> None:
    """A typo in parked-candidate policy should not silently pass release evidence."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--forbid-backends",
            "lattice-db",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "lattice-db: unknown backend in --forbid-backends" in result.stderr


def test_backend_shootout_guardrail_rejects_duplicate_backend_policy_name(tmp_path: Path) -> None:
    """Duplicate comma-list backend policy entries should fail instead of being ignored."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded,embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: duplicate backend in --require-backends" in result.stderr


def test_backend_shootout_guardrail_rejects_unknown_report_backend_row(tmp_path: Path) -> None:
    """Release evidence should not contain backend rows outside the supported backend set."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                    {"backend": "surprise", "status": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "surprise: unknown backend in report summaries" in result.stderr


def test_backend_shootout_guardrail_rejects_non_object_summary_row(tmp_path: Path) -> None:
    """Release evidence should reject malformed summary rows instead of ignoring them."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                    "not-a-summary",
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[1] must be an object" in result.stderr


def test_backend_shootout_guardrail_rejects_empty_summary_rows(tmp_path: Path) -> None:
    """Release evidence should contain at least one backend summary row."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(json.dumps({"summaries": []}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/check-backend-shootout.py", str(report)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries must contain at least one backend row" in result.stderr


def test_backend_shootout_guardrail_rejects_non_standard_json_constants(tmp_path: Path) -> None:
    """Release evidence should be strict JSON, not Python-specific JSON extensions."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        '{"summaries": [{"backend": "embedded", "status": "ok"}], "ignored_metric": NaN}',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: JSON contains non-standard numeric constant NaN" in result.stderr


def test_backend_shootout_guardrail_rejects_summary_row_without_backend(tmp_path: Path) -> None:
    """Release evidence should reject summary rows that cannot be attributed to a backend."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                    {"status": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[1].backend is missing" in result.stderr


def test_backend_shootout_guardrail_rejects_summary_row_without_status(tmp_path: Path) -> None:
    """Release evidence should reject summary rows that cannot report backend status."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                    {"backend": "bm25"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[1].status is missing" in result.stderr


def test_backend_shootout_guardrail_rejects_unknown_summary_status(tmp_path: Path) -> None:
    """Release evidence should reject status values the harness does not emit."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                    {"backend": "bm25", "status": "okay"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[1].status is 'okay', expected one of: error, ok" in result.stderr


def test_backend_shootout_guardrail_rejects_error_status_without_error_message(tmp_path: Path) -> None:
    """Failed backend rows should carry the diagnostic that explains the failure."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                    {"backend": "neo4j", "status": "error"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[1].error is missing for error status" in result.stderr


def test_backend_shootout_guardrail_rejects_error_status_with_success_metric(tmp_path: Path) -> None:
    """Failed backend rows should not carry success-only quality metrics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                    {
                        "backend": "neo4j",
                        "status": "error",
                        "error": "connection refused",
                        "answer_at_5": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "neo4j: answer_at_5 must be empty for error status" in result.stderr


def test_backend_shootout_guardrail_rejects_ok_status_with_error_message(tmp_path: Path) -> None:
    """Successful backend rows should not carry stale failure diagnostics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok", "error": "previous failure"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[0].error must be empty for ok status" in result.stderr


def test_backend_shootout_guardrail_rejects_schema_report_summary_without_query_count(tmp_path: Path) -> None:
    """Schema-versioned report rows should preserve per-backend query counts."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "summaries": [
                    {"backend": "embedded", "status": "ok"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[0].query_count is missing" in result.stderr


def test_backend_shootout_guardrail_rejects_invalid_schema_report_summary_query_count(tmp_path: Path) -> None:
    """Schema-versioned per-backend query counts should be non-negative integers."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "summaries": [
                    {"backend": "embedded", "status": "ok", "query_count": -1},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[0].query_count must be a non-negative integer" in result.stderr


def test_backend_shootout_guardrail_rejects_invalid_schema_report_query_count(tmp_path: Path) -> None:
    """Schema-versioned report-level query_count should be a non-negative integer."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "query_count": -1,
                "summaries": [
                    {"backend": "embedded", "status": "ok", "query_count": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: query_count must be a non-negative integer" in result.stderr


def test_backend_shootout_guardrail_rejects_invalid_schema_report_event_count(tmp_path: Path) -> None:
    """Schema-versioned report-level event_count should be a non-negative integer."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "event_count": -1,
                "query_count": 0,
                "summaries": [
                    {"backend": "embedded", "status": "ok", "query_count": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: event_count must be a non-negative integer" in result.stderr


def test_backend_shootout_guardrail_rejects_invalid_schema_report_limit(tmp_path: Path) -> None:
    """Schema-versioned report-level limit should be a positive integer."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "event_count": 0,
                "query_count": 0,
                "limit": 0,
                "summaries": [
                    {"backend": "embedded", "status": "ok", "query_count": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: limit must be a positive integer" in result.stderr


def test_backend_shootout_guardrail_rejects_unsupported_schema_version_without_metadata_mode(tmp_path: Path) -> None:
    """A report that declares an unsupported schema version should fail basic validation."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 2,
                "event_count": 0,
                "query_count": 0,
                "limit": 5,
                "summaries": [
                    {"backend": "embedded", "status": "ok", "query_count": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: report_schema_version must be 1" in result.stderr


def test_backend_shootout_guardrail_rejects_boolean_schema_version(tmp_path: Path) -> None:
    """JSON booleans must not satisfy the integer schema-version contract."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": True,
                "event_count": 0,
                "query_count": 0,
                "limit": 5,
                "summaries": [
                    {"backend": "embedded", "status": "ok", "query_count": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: report_schema_version must be 1" in result.stderr


def test_backend_shootout_guardrail_rejects_schema_summary_query_count_mismatch(tmp_path: Path) -> None:
    """Schema-versioned backend query counts should match the report-level query count."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "event_count": 0,
                "query_count": 2,
                "limit": 5,
                "summaries": [
                    {"backend": "embedded", "status": "ok", "query_count": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: query_count=1 does not match report query_count 2" in result.stderr


def test_backend_shootout_guardrail_rejects_unknown_backend_scoped_metric_name(tmp_path: Path) -> None:
    """A typo in backend-scoped thresholds should not silently disable a guardrail."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok", "checkout_p95_ms": 5.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--max-checkout-p95-ms",
            "emebdded=10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "emebdded: unknown backend in --max-checkout-p95-ms" in result.stderr


def test_backend_shootout_guardrail_rejects_duplicate_backend_scoped_metric(tmp_path: Path) -> None:
    """A repeated backend threshold should not silently overwrite the stricter value."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok", "checkout_p95_ms": 50.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--max-checkout-p95-ms",
            "embedded=10",
            "--max-checkout-p95-ms",
            "embedded=100",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: duplicate backend threshold in --max-checkout-p95-ms" in result.stderr


def test_backend_shootout_guardrail_rejects_non_finite_backend_scoped_metric_threshold(tmp_path: Path) -> None:
    """A non-finite backend threshold should not silently disable a guardrail."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok", "checkout_p95_ms": 50.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--max-checkout-p95-ms",
            "embedded=nan",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "--max-checkout-p95-ms threshold must be a finite number" in result.stderr


def test_backend_shootout_guardrail_rejects_negative_backend_scoped_metric_threshold(tmp_path: Path) -> None:
    """A negative backend threshold should not silently weaken a guardrail."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok", "quality_per_1k_returned_tokens": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--min-quality-per-1k-returned-tokens",
            "embedded=-1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "--min-quality-per-1k-returned-tokens threshold must be non-negative" in result.stderr


def test_backend_shootout_guardrail_rejects_duplicate_dashboard_source_requirement(tmp_path: Path) -> None:
    """A repeated dashboard source expectation should not silently overwrite release evidence."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok", "dashboard_graph_source": "embedded"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--require-dashboard-source",
            "embedded=neo4j",
            "--require-dashboard-source",
            "embedded=embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: duplicate backend requirement in --require-dashboard-source" in result.stderr


def test_backend_shootout_guardrail_help_describes_parked_candidate_policy() -> None:
    """The guardrail CLI should explain how to exclude parked candidate backends."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--forbid-backends" in result.stdout
    assert "parked candidate" in result.stdout


def test_backend_shootout_guardrail_rejects_missing_report_metadata(tmp_path: Path) -> None:
    """Release evidence should optionally require self-describing report provenance."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-report-metadata",
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: report_schema_version must be 1" in result.stderr
    assert "report: generated_at_utc is missing" in result.stderr
    assert "report: source_fingerprints.eventloom_sha256 is missing" in result.stderr
    assert "report: workload_fingerprints.queries_sha256 is missing" in result.stderr


def test_backend_shootout_guardrail_rejects_stale_markdown_sidecar(tmp_path: Path) -> None:
    """Release evidence should include a Markdown sidecar with matching provenance."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Render backend shootout provenance."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "render provenance", "expected_terms": ["provenance"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    output.with_suffix(".md").write_text("# Backend Shootout\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(output),
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: Markdown sidecar missing Report schema version" in result.stderr
    assert "report: Markdown sidecar missing Generated at UTC" in result.stderr
    assert "report: Markdown sidecar missing Queries" in result.stderr
    assert "report: Markdown sidecar missing Events" in result.stderr
    assert "report: Markdown sidecar missing Limit" in result.stderr
    assert "report: Markdown sidecar missing Workload queries SHA-256" in result.stderr


def test_backend_shootout_guardrail_rejects_markdown_missing_backend_rows(tmp_path: Path) -> None:
    """Markdown release evidence should include the backend rows from the JSON report."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Render backend rows."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "backend rows", "expected_terms": ["backend"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    output.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Backend Shootout",
                "",
                f"- Report schema version: `{report['report_schema_version']}`",
                f"- Harness: `{report['harness']}`",
                f"- Eventloom path: `{report['eventloom_path']}`",
                f"- Queries file: `{report['queries_file']}`",
                f"- Source Eventloom SHA-256: `{report['source_fingerprints']['eventloom_sha256']}`",
                f"- Source queries SHA-256: `{report['source_fingerprints']['queries_sha256']}`",
                f"- Workload events SHA-256: `{report['workload_fingerprints']['events_sha256']}`",
                f"- Workload queries SHA-256: `{report['workload_fingerprints']['queries_sha256']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(output),
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: Markdown sidecar missing backend row for bm25" in result.stderr


def test_backend_shootout_guardrail_rejects_markdown_backend_row_without_metrics(tmp_path: Path) -> None:
    """Markdown backend rows should expose the release-critical quality metrics."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Render backend row metrics."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "backend row metrics", "expected_terms": ["backend"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    output.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Backend Shootout",
                "",
                f"- Report schema version: `{report['report_schema_version']}`",
                f"- Harness: `{report['harness']}`",
                f"- Eventloom path: `{report['eventloom_path']}`",
                f"- Queries file: `{report['queries_file']}`",
                f"- Source Eventloom SHA-256: `{report['source_fingerprints']['eventloom_sha256']}`",
                f"- Source queries SHA-256: `{report['source_fingerprints']['queries_sha256']}`",
                f"- Workload events SHA-256: `{report['workload_fingerprints']['events_sha256']}`",
                f"- Workload queries SHA-256: `{report['workload_fingerprints']['queries_sha256']}`",
                "",
                "| Backend | Status |",
                "|---------|--------|",
                "| bm25 | ok |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(output),
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: Markdown sidecar row for bm25 missing answer_at_5=1.0" in result.stderr
    assert "report: Markdown sidecar row for bm25 missing recall_at_5=1.0" in result.stderr
    assert "report: Markdown sidecar row for bm25 missing quality_per_1k_injected_tokens" in result.stderr
    assert "report: Markdown sidecar row for bm25 missing answer_at_5_per_1k_injected_tokens" in result.stderr


def test_backend_shootout_guardrail_rejects_stale_report_fingerprints(tmp_path: Path) -> None:
    """Release evidence should fail when input files no longer match the checked report."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Fingerprint backend shootout reports."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "fingerprint reports", "expected_terms": ["Fingerprint"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr

    queries.write_text(
        json.dumps([{"query": "changed query", "expected_terms": ["changed"]}]),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(output),
            "--require-report-metadata",
            "--verify-report-fingerprints",
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: source_fingerprints.queries_sha256 does not match current input" in result.stderr
    assert "report: workload_fingerprints.queries_sha256 does not match current input" in result.stderr


def test_backend_shootout_guardrail_rejects_stale_report_counts(tmp_path: Path) -> None:
    """Fingerprint verification should also catch stale report count metadata."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Count backend shootout inputs."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "count reports", "expected_terms": ["Count"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    report["event_count"] = 999
    report["query_count"] = 999
    output.write_text(json.dumps(report), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(output),
            "--require-report-metadata",
            "--verify-report-fingerprints",
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: event_count=999 does not match current input count 1" in result.stderr
    assert "report: query_count=999 does not match current input count 1" in result.stderr


def test_backend_shootout_guardrail_rejects_boolean_report_counts_during_fingerprint_check(
    tmp_path: Path,
) -> None:
    """Fingerprint verification should not treat JSON booleans as integer counts."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Reject boolean backend shootout counts."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "boolean counts", "expected_terms": ["boolean"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    report.pop("report_schema_version")
    report["event_count"] = True
    report["query_count"] = True
    output.write_text(json.dumps(report), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(output),
            "--verify-report-fingerprints",
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: event_count=True does not match current input count 1" in result.stderr
    assert "report: query_count=True does not match current input count 1" in result.stderr


def test_backend_shootout_guardrail_verified_query_count_rejects_booleans() -> None:
    """Fingerprint count comparison should not treat JSON booleans as integers."""
    module = _load_backend_shootout_check_module()

    assert module._verified_query_count({"query_count": True}) is None


def test_backend_shootout_guardrail_rejects_stale_backend_query_counts(tmp_path: Path) -> None:
    """Fingerprint verification should catch stale per-backend query counts."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Count backend summary inputs."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "count backend summaries", "expected_terms": ["Count"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            "agent-1",
            "--backends",
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    report["summaries"][0]["query_count"] = 999
    output.write_text(json.dumps(report), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(output),
            "--require-report-metadata",
            "--verify-report-fingerprints",
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25: query_count=999 does not match current input count 1" in result.stderr


def test_backend_shootout_guardrail_rejects_duplicate_backend_summaries(tmp_path: Path) -> None:
    """Required backend evidence should not silently accept duplicate summary rows."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "error", "query_count": 1},
                    {"backend": "embedded", "status": "ok", "query_count": 1},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: duplicate backend summary rows found" in result.stderr


def test_backend_shootout_guardrail_rejects_slow_embedded_projection(tmp_path: Path) -> None:
    """The guardrail should fail when embedded projection performance regresses."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 0.5,
                        "recall_at_5": 0.5,
                        "citation_coverage": 1.0,
                        "dashboard_graph_source": "embedded",
                        "projection_events_per_second": 8.0,
                        "cold_bootstrap_ms": 900.0,
                        "first_useful_init_ms": 120000.0,
                        "first_checkout_ms": 250.0,
                        "append_to_projection_p95_ms": 40.0,
                        "resident_memory_delta_bytes": 200_000_000,
                        "on_disk_footprint_bytes": 120_000_000,
                        "rebuild_recovery_ms": 121000.0,
                        "checkout_p95_ms": 90.0,
                        "dashboard_graph_load_ms": 350.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--min-projection-events-per-second",
            "embedded=40",
            "--max-cold-bootstrap-ms",
            "embedded=500",
            "--max-first-useful-init-ms",
            "embedded=15000",
            "--max-first-checkout-ms",
            "embedded=200",
            "--max-append-to-projection-p95-ms",
            "embedded=25",
            "--max-resident-memory-delta-bytes",
            "embedded=100000000",
            "--max-on-disk-footprint-bytes",
            "embedded=100000000",
            "--max-rebuild-recovery-ms",
            "embedded=15000",
            "--max-checkout-p95-ms",
            "embedded=85",
            "--max-dashboard-graph-load-ms",
            "embedded=100",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: projection_events_per_second=8.0 is below 40.0" in result.stderr
    assert "embedded: cold_bootstrap_ms=900.0 is above 500.0" in result.stderr
    assert "embedded: first_useful_init_ms=120000.0 is above 15000.0" in result.stderr
    assert "embedded: first_checkout_ms=250.0 is above 200.0" in result.stderr
    assert "embedded: append_to_projection_p95_ms=40.0 is above 25.0" in result.stderr
    assert "embedded: resident_memory_delta_bytes=200000000 is above 100000000.0" in result.stderr
    assert "embedded: on_disk_footprint_bytes=120000000 is above 100000000.0" in result.stderr
    assert "embedded: rebuild_recovery_ms=121000.0 is above 15000.0" in result.stderr
    assert "embedded: checkout_p95_ms=90.0 is above 85.0" in result.stderr
    assert "embedded: dashboard_graph_load_ms=350.0 is above 100.0" in result.stderr


def test_backend_shootout_guardrail_rejects_low_token_efficiency_and_slow_lanes(tmp_path: Path) -> None:
    """The guardrail should fail when embedded token efficiency or lane latencies regress."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "status": "ok",
                        "answer_at_5": 0.5,
                        "recall_at_5": 0.5,
                        "citation_coverage": 1.0,
                        "dashboard_graph_source": "embedded",
                        "quality_per_1k_returned_tokens": 0.04,
                        "answer_at_5_per_1k_returned_tokens": 0.03,
                        "mean_injected_tokens": 2000.0,
                        "quality_per_1k_injected_tokens": 0.02,
                        "answer_at_5_per_1k_injected_tokens": 0.015,
                        "exact_p95_ms": 12.0,
                        "exact_p99_ms": 15.0,
                        "keyword_p95_ms": 75.0,
                        "keyword_p99_ms": 90.0,
                        "vector_p95_ms": 20.0,
                        "vector_p99_ms": 25.0,
                        "traversal_p95_ms": 5.0,
                        "traversal_p99_ms": 9.0,
                        "checkout_p99_ms": 150.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "embedded",
            "--min-quality-per-1k-returned-tokens",
            "embedded=0.1",
            "--min-answer-at-5-per-1k-returned-tokens",
            "embedded=0.1",
            "--min-quality-per-1k-injected-tokens",
            "embedded=0.1",
            "--min-answer-at-5-per-1k-injected-tokens",
            "embedded=0.1",
            "--max-exact-p95-ms",
            "embedded=10",
            "--max-keyword-p95-ms",
            "embedded=50",
            "--max-vector-p95-ms",
            "embedded=10",
            "--max-traversal-p95-ms",
            "embedded=1",
            "--max-checkout-p99-ms",
            "embedded=100",
            "--max-exact-p99-ms",
            "embedded=12",
            "--max-keyword-p99-ms",
            "embedded=80",
            "--max-vector-p99-ms",
            "embedded=20",
            "--max-traversal-p99-ms",
            "embedded=5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: quality_per_1k_returned_tokens=0.04 is below 0.1" in result.stderr
    assert "embedded: answer_at_5_per_1k_returned_tokens=0.03 is below 0.1" in result.stderr
    assert "embedded: quality_per_1k_injected_tokens=0.02 is below 0.1" in result.stderr
    assert "embedded: answer_at_5_per_1k_injected_tokens=0.015 is below 0.1" in result.stderr
    assert "embedded: exact_p95_ms=12.0 is above 10.0" in result.stderr
    assert "embedded: keyword_p95_ms=75.0 is above 50.0" in result.stderr
    assert "embedded: vector_p95_ms=20.0 is above 10.0" in result.stderr
    assert "embedded: traversal_p95_ms=5.0 is above 1.0" in result.stderr
    assert "embedded: checkout_p99_ms=150.0 is above 100.0" in result.stderr
    assert "embedded: exact_p99_ms=15.0 is above 12.0" in result.stderr
    assert "embedded: keyword_p99_ms=90.0 is above 80.0" in result.stderr
    assert "embedded: vector_p99_ms=25.0 is above 20.0" in result.stderr
    assert "embedded: traversal_p99_ms=9.0 is above 5.0" in result.stderr


def test_backend_shootout_guardrail_accepts_current_sample_report() -> None:
    """The checked sample report should pass the active-backend guardrail."""
    report = json.loads(Path("reports/backend-shootout/backend-shootout.json").read_text(encoding="utf-8"))
    ok_summaries = [summary for summary in report["summaries"] if summary["status"] == "ok"]

    assert ok_summaries
    for summary in ok_summaries:
        assert summary["mean_injected_tokens"] is not None
        assert summary["quality_per_1k_injected_tokens"] is not None
        assert summary["answer_at_5_per_1k_injected_tokens"] is not None
        assert summary["resident_memory_delta_bytes"] is not None
        assert summary["on_disk_footprint_bytes"] is not None

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            "reports/backend-shootout/backend-shootout.json",
            "--require-report-metadata",
            "--require-markdown-report",
            "--verify-report-fingerprints",
            "--require-backends",
            "embedded,bm25",
            "--require-labeled-metrics",
            "--require-dashboard-source",
            "embedded=embedded",
            "--min-answer-at-5",
            "0.5",
            "--min-recall-at-5",
            "0.5",
            "--min-citation-coverage",
            "1.0",
            "--min-quality-per-1k-injected-tokens",
            "embedded=1.0",
            "--min-answer-at-5-per-1k-injected-tokens",
            "embedded=1.0",
            "--max-cold-bootstrap-ms",
            "embedded=250",
            "--max-first-checkout-ms",
            "embedded=25",
            "--max-append-to-projection-p95-ms",
            "embedded=50",
            "--max-resident-memory-delta-bytes",
            "embedded=256000000",
            "--max-on-disk-footprint-bytes",
            "embedded=256000000",
            "--max-dashboard-graph-load-ms",
            "embedded=250",
            "--max-checkout-p99-ms",
            "embedded=25",
            "--max-exact-p99-ms",
            "embedded=10",
            "--max-keyword-p99-ms",
            "embedded=5",
            "--max-vector-p99-ms",
            "embedded=5",
            "--max-traversal-p99-ms",
            "embedded=5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend shootout guardrail passed" in result.stdout


def test_backend_shootout_guardrail_accepts_current_longmemeval_40_performance_report() -> None:
    """The medium-scale embedded report should protect transaction-backed projection performance."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            "reports/backend-shootout/longmemeval-40-backend-shootout.json",
            "--require-report-metadata",
            "--require-markdown-report",
            "--verify-report-fingerprints",
            "--require-backends",
            "embedded,bm25",
            "--require-labeled-metrics",
            "--require-dashboard-source",
            "embedded=embedded",
            "--min-citation-coverage",
            "1.0",
            "--min-projection-events-per-second",
            "embedded=40",
            "--max-cold-bootstrap-ms",
            "embedded=200",
            "--max-first-useful-init-ms",
            "embedded=15000",
            "--max-first-checkout-ms",
            "embedded=50",
            "--max-append-to-projection-p95-ms",
            "embedded=30",
            "--max-resident-memory-delta-bytes",
            "embedded=768000000",
            "--max-on-disk-footprint-bytes",
            "embedded=256000000",
            "--max-dashboard-graph-load-ms",
            "embedded=500",
            "--max-rebuild-recovery-ms",
            "embedded=15000",
            "--max-checkout-p95-ms",
            "embedded=100",
            "--max-checkout-p99-ms",
            "embedded=75",
            "--min-quality-per-1k-returned-tokens",
            "embedded=0.10",
            "--min-answer-at-5-per-1k-returned-tokens",
            "embedded=0.10",
            "--min-quality-per-1k-injected-tokens",
            "embedded=0.10",
            "--min-answer-at-5-per-1k-injected-tokens",
            "embedded=0.10",
            "--max-exact-p95-ms",
            "embedded=15",
            "--max-exact-p99-ms",
            "embedded=10",
            "--max-keyword-p95-ms",
            "embedded=75",
            "--max-keyword-p99-ms",
            "embedded=40",
            "--max-vector-p95-ms",
            "embedded=25",
            "--max-vector-p99-ms",
            "embedded=35",
            "--max-traversal-p95-ms",
            "embedded=10",
            "--max-traversal-p99-ms",
            "embedded=10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend shootout guardrail passed" in result.stdout


def test_backend_shootout_guardrail_accepts_current_longmemeval_100_scale_report() -> None:
    """The 100-query scale report should include injected-token efficiency evidence."""
    report = json.loads(Path("reports/backend-shootout/longmemeval-100-backend-shootout.json").read_text(encoding="utf-8"))
    summaries = {summary["backend"]: summary for summary in report["summaries"]}

    assert summaries["embedded"]["mean_injected_tokens"] == 1892.97
    assert summaries["embedded"]["quality_per_1k_injected_tokens"] == 0.1849
    assert summaries["embedded"]["answer_at_5_per_1k_injected_tokens"] == 0.1849
    assert summaries["embedded"]["cold_bootstrap_ms"] == 105.955
    assert summaries["embedded"]["first_checkout_ms"] == 66.316
    assert summaries["embedded"]["append_to_projection_p95_ms"] == 25.241
    assert summaries["embedded"]["resident_memory_delta_bytes"] == 1447043072
    assert summaries["embedded"]["on_disk_footprint_bytes"] == 57270272
    assert summaries["embedded"]["resident_memory_delta_bytes"] is not None
    assert summaries["embedded"]["on_disk_footprint_bytes"] is not None
    assert summaries["bm25"]["mean_injected_tokens"] == 4179.5
    assert summaries["bm25"]["quality_per_1k_injected_tokens"] == 0.0813
    assert summaries["bm25"]["answer_at_5_per_1k_injected_tokens"] == 0.0813

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            "reports/backend-shootout/longmemeval-100-backend-shootout.json",
            "--require-report-metadata",
            "--require-markdown-report",
            "--verify-report-fingerprints",
            "--require-backends",
            "embedded,bm25",
            "--require-labeled-metrics",
            "--require-dashboard-source",
            "embedded=embedded",
            "--min-citation-coverage",
            "1.0",
            "--min-projection-events-per-second",
            "embedded=40",
            "--max-cold-bootstrap-ms",
            "embedded=200",
            "--max-first-useful-init-ms",
            "embedded=40000",
            "--max-first-checkout-ms",
            "embedded=100",
            "--max-append-to-projection-p95-ms",
            "embedded=40",
            "--max-resident-memory-delta-bytes",
            "embedded=1536000000",
            "--max-on-disk-footprint-bytes",
            "embedded=512000000",
            "--max-dashboard-graph-load-ms",
            "embedded=500",
            "--max-rebuild-recovery-ms",
            "embedded=40000",
            "--max-checkout-p95-ms",
            "embedded=125",
            "--max-checkout-p99-ms",
            "embedded=175",
            "--min-quality-per-1k-returned-tokens",
            "embedded=0.15",
            "--min-answer-at-5-per-1k-returned-tokens",
            "embedded=0.15",
            "--min-quality-per-1k-injected-tokens",
            "embedded=0.15",
            "--min-answer-at-5-per-1k-injected-tokens",
            "embedded=0.15",
            "--max-exact-p95-ms",
            "embedded=10",
            "--max-exact-p99-ms",
            "embedded=12",
            "--max-keyword-p95-ms",
            "embedded=60",
            "--max-keyword-p99-ms",
            "embedded=80",
            "--max-vector-p95-ms",
            "embedded=15",
            "--max-vector-p99-ms",
            "embedded=20",
            "--max-traversal-p95-ms",
            "embedded=10",
            "--max-traversal-p99-ms",
            "embedded=10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend shootout guardrail passed" in result.stdout


def test_backend_shootout_workload_builder_exports_longmemeval_queries(tmp_path: Path) -> None:
    """LongMemEval should be materializable into backend-shootout Eventloom/query artifacts."""
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "What degree did I graduate with?",
                    "answer": "Business Administration",
                    "answer_session_ids": ["answer-1"],
                    "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                    "haystack_session_ids": ["answer-1"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "I graduated with a Business Administration degree.",
                                "has_answer": True,
                            }
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    eventloom_output = tmp_path / "longmemeval.jsonl"
    queries_output = tmp_path / "queries.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build-backend-shootout-workload.py",
            "--dataset",
            str(dataset),
            "--questions",
            "1",
            "--eventloom-output",
            str(eventloom_output),
            "--queries-output",
            str(queries_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert eventloom_output.exists()
    payload = json.loads(queries_output.read_text(encoding="utf-8"))
    assert payload == [
        {
            "query": "What degree did I graduate with?",
            "expected_terms": ["Business Administration"],
        }
    ]
    assert "Wrote 1 backend-shootout queries" in result.stdout


def test_backend_shootout_workload_builder_rejects_non_positive_question_limit(tmp_path: Path) -> None:
    """Workload materialization should not emit empty query artifacts from --questions 0."""
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text("[]", encoding="utf-8")
    eventloom_output = tmp_path / "longmemeval.jsonl"
    queries_output = tmp_path / "queries.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build-backend-shootout-workload.py",
            "--dataset",
            str(dataset),
            "--questions",
            "0",
            "--eventloom-output",
            str(eventloom_output),
            "--queries-output",
            str(queries_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "questions must be a positive integer" in result.stderr


def test_backend_shootout_workload_builder_rejects_empty_materialized_workload(tmp_path: Path) -> None:
    """Workload materialization should fail when no benchmark queries are produced."""
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text("[]", encoding="utf-8")
    eventloom_output = tmp_path / "longmemeval.jsonl"
    queries_output = tmp_path / "queries.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build-backend-shootout-workload.py",
            "--dataset",
            str(dataset),
            "--eventloom-output",
            str(eventloom_output),
            "--queries-output",
            str(queries_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "No backend-shootout queries were materialized" in result.stderr
    assert not queries_output.exists()


def test_backend_shootout_workload_builder_rejects_malformed_dataset_json(tmp_path: Path) -> None:
    """Workload materialization should report malformed dataset JSON without a traceback."""
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text('{"broken": true', encoding="utf-8")
    eventloom_output = tmp_path / "longmemeval.jsonl"
    queries_output = tmp_path / "queries.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build-backend-shootout-workload.py",
            "--dataset",
            str(dataset),
            "--eventloom-output",
            str(eventloom_output),
            "--queries-output",
            str(queries_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "dataset contains malformed JSON" in result.stderr
    assert "Traceback" not in result.stderr


def test_backend_shootout_workload_builder_rejects_non_list_dataset_json(tmp_path: Path) -> None:
    """Workload materialization should report invalid dataset shape without a traceback."""
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text('{"records": []}', encoding="utf-8")
    eventloom_output = tmp_path / "longmemeval.jsonl"
    queries_output = tmp_path / "queries.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build-backend-shootout-workload.py",
            "--dataset",
            str(dataset),
            "--eventloom-output",
            str(eventloom_output),
            "--queries-output",
            str(queries_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "LongMemEval dataset must be a JSON list" in result.stderr
    assert "Traceback" not in result.stderr


def test_backend_shootout_workload_builder_rejects_non_standard_dataset_json_constants(tmp_path: Path) -> None:
    """Workload materialization should reject non-portable JSON constants in datasets."""
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(
        """
        [
          {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What degree did I graduate with?",
            "answer": "Business Administration",
            "answer_session_ids": ["answer-1"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_session_ids": ["answer-1"],
            "haystack_sessions": [[
              {
                "role": "user",
                "content": "I graduated with a Business Administration degree.",
                "has_answer": true
              }
            ]],
            "ignored_score": NaN
          }
        ]
        """,
        encoding="utf-8",
    )
    eventloom_output = tmp_path / "longmemeval.jsonl"
    queries_output = tmp_path / "queries.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build-backend-shootout-workload.py",
            "--dataset",
            str(dataset),
            "--eventloom-output",
            str(eventloom_output),
            "--queries-output",
            str(queries_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "dataset contains non-standard numeric constant NaN" in result.stderr
    assert not queries_output.exists()


def test_backend_shootout_workload_builder_query_json_rejects_non_finite_values() -> None:
    """Generated query artifacts should be strict JSON."""
    module = _load_backend_workload_builder_module()

    with pytest.raises(ValueError):
        module._strict_json_dumps([{"query": "bad", "score": float("nan")}])


def test_backend_shootout_projects_events_inside_supported_bulk_transaction() -> None:
    """Backend shootout projection should use backend bulk transaction hooks when available."""
    spec = importlib.util.spec_from_file_location("backend_shootout", "scripts/backend-shootout.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["backend_shootout"] = module
    spec.loader.exec_module(module)

    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def begin_bulk_projection(self) -> None:
            self.calls.append("begin")

        async def commit_bulk_projection(self) -> None:
            self.calls.append("commit")

        async def rollback_bulk_projection(self) -> None:
            self.calls.append("rollback")

    class FakeFabric:
        def __init__(self) -> None:
            self.graph = FakeGraph()

        async def _project_event(self, event: object, *, session_id: str) -> None:
            self.graph.calls.append(f"project:{event}:{session_id}")

    fake = FakeFabric()

    import asyncio

    asyncio.run(module._project_events(fake, ["a", "b"], "agent-1"))

    assert fake.graph.calls == [
        "begin",
        "project:a:agent-1",
        "project:b:agent-1",
        "commit",
    ]


@pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")
def test_backend_shootout_embedded_accepts_single_eventloom_file(tmp_path: Path) -> None:
    """Graph backend shootouts should honor the documented JSONL-file Eventloom input."""
    log_path = tmp_path / "agent-1.jsonl"
    EventLog(log_path).append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use embedded file input for backend shootouts."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "embedded file input", "expected_terms": ["embedded file"]}]),
        encoding="utf-8",
    )
    output = tmp_path / "backend-shootout.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/backend-shootout.py",
            "--eventloom-path",
            str(log_path),
            "--session-id",
            "agent-1",
            "--backends",
            "embedded",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))["summaries"][0]
    assert summary["backend"] == "embedded"
    assert summary["status"] == "ok"
