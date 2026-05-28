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


def _guardrail_summary(**overrides: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "backend": "bm25",
        "contract": "retrieve",
        "status": "ok",
        "query_count": 1,
        "mean_quality": 1.0,
        "citation_coverage": 1.0,
        "mean_returned_tokens": 1.0,
        "quality_per_1k_returned_tokens": 1000.0,
        "mean_injected_tokens": 1.0,
        "quality_per_1k_injected_tokens": 1000.0,
        "first_checkout_ms": 0.0,
        "checkout_p95_ms": 0.0,
        "checkout_p99_ms": 0.0,
    }
    summary.update(overrides)
    return summary


def _guardrail_diagnostic(**overrides: object) -> dict[str, object]:
    diagnostic: dict[str, object] = {
        "query": "What happened?",
        "expected_terms": [],
        "identity_terms": [],
        "source_terms": [],
        "retrieval_terms": [],
        "quality": 1.0,
        "answer_hit": True,
        "recall_quality": 1.0,
        "recall_hit": True,
        "missing_expected_terms": [],
        "missing_retrieval_terms": [],
        "latency_ms": 0.0,
        "returned_tokens": 1,
        "injected_tokens": 1,
        "citation_hit": True,
        "top_contexts": [],
    }
    diagnostic.update(overrides)
    return diagnostic


def _write_guardrail_report(
    tmp_path: Path,
    *,
    summary: dict[str, object] | None = None,
    diagnostic: dict[str, object] | None = None,
) -> Path:
    summary = summary or _guardrail_summary()
    diagnostic = diagnostic or _guardrail_diagnostic()
    backend = str(summary["backend"])
    contract = str(summary.get("contract") or "retrieve")
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [summary],
                "query_results": {f"{backend}:{contract}": [diagnostic]},
            }
        ),
        encoding="utf-8",
    )
    return report


def test_backend_shootout_default_active_backends_are_embedded_and_bm25_only() -> None:
    """Default shootouts should run without optional sidecar infrastructure."""
    module = _load_backend_shootout_module()

    assert "latticedb" in module.SUPPORTED_BACKENDS
    assert "neo4j" in module.SUPPORTED_BACKENDS
    assert "pggraph" in module.SUPPORTED_BACKENDS
    assert module.DEFAULT_BACKENDS == ("embedded", "bm25")


def test_backend_shootout_help_does_not_import_runtime_or_dashboard() -> None:
    """Help output should avoid importing heavy runtime modules."""
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "scripts/backend-shootout.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "zaxy.core" not in result.stderr
    assert "zaxy.dashboard" not in result.stderr


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


def test_backend_shootout_scores_acceptable_answer_alternatives() -> None:
    """Backend Answer@5 should share benchmark answer-surface matching."""
    module = _load_backend_shootout_module()

    quality = module._expected_term_quality(
        "zaxy_synthesis_bundle=true duration_answer=14 days",
        ("14 days. 15 days (including the last day) is also acceptable.",),
    )

    assert quality == 1.0


def test_backend_shootout_scores_compact_count_answer_surfaces() -> None:
    """LongMemEval prose answers should match compact synthesized count answers."""
    module = _load_backend_shootout_module()

    quality = module._expected_term_quality(
        "count_answer_text=I currently own four musical instruments.",
        (
            "I currently own 4 musical instruments. "
            "I've had the Fender Stratocaster electric guitar for 5 years.",
        ),
    )

    assert quality == 1.0


def test_backend_shootout_materializes_fabric_eventloom_for_source_lane(tmp_path: Path) -> None:
    """Graph backend diagnostics should give MemoryFabric the same Eventloom source."""
    module = _load_backend_shootout_module()
    source_log = EventLog(tmp_path / "source.jsonl")
    source_event = source_log.append(
        "document.indexed",
        actor="longmemeval",
        payload={
            "path": "longmemeval/answer-1/salient-turn-0001.md",
            "content": "I started using NebulaStream last week.",
            "start_line": 1,
            "end_line": 1,
        },
        thread="answer-1",
    )

    eventloom_path = module._prepare_fabric_eventloom(
        [source_event],
        output_parent=tmp_path / "reports",
        backend="embedded",
        session_id="default",
    )

    materialized = EventLog(eventloom_path / "default.jsonl").read_all()
    assert [event.model_dump() for event in materialized] == [source_event.model_dump()]


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
    assert payload["summaries"][0]["contract"] == "retrieve"
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
    assert "| bm25 | retrieve | ok |" in markdown
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
    assert "| bm25 | retrieve | ok |" in markdown
    assert "| 1.0 | 1.0 |" in markdown


def test_backend_shootout_default_run_is_sidecar_free(tmp_path: Path) -> None:
    """Omitting --backends should run embedded plus BM25 without optional services."""
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "agent-1.jsonl")
    log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Embedded Kuzu is the default local projection."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "embedded Kuzu projection", "expected_terms": ["Kuzu"]}]),
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
    assert [(summary["backend"], summary["contract"]) for summary in payload["summaries"]] == [
        ("embedded", "retrieve"),
        ("embedded", "answer_ready"),
        ("bm25", "retrieve"),
    ]
    assert {summary["backend"] for summary in payload["summaries"]} == {"embedded", "bm25"}
    assert all(summary["status"] == "ok" for summary in payload["summaries"])
    assert not (output.parent / "embedded.eventloom").exists()


def test_backend_shootout_can_emit_per_query_diagnostics(tmp_path: Path) -> None:
    """Quality-parity work needs miss diagnostics, not just aggregate scores."""
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "agent-1.jsonl")
    log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Embedded Kuzu should match Neo4j quality."},
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            [
                {"query": "embedded kuzu quality", "expected_terms": ["Kuzu"]},
                {
                    "query": "embedded quality missing answer",
                    "expected_terms": ["97 days"],
                    "identity_terms": ["missing-agent", "agent-1"],
                },
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
            "bm25",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
            "--include-query-results",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    diagnostics = payload["query_results"]["bm25:retrieve"]
    assert payload["summaries"][0]["answer_at_5"] == 0.5
    assert payload["summaries"][0]["recall_at_5"] == 1.0
    assert payload["summaries"][0]["citation_coverage"] == 1.0
    assert diagnostics == [
        {
            "query": "embedded kuzu quality",
            "expected_terms": ["Kuzu"],
            "identity_terms": [],
            "source_terms": [],
            "retrieval_terms": ["Kuzu"],
            "quality": 1.0,
            "answer_hit": True,
            "recall_quality": 1.0,
            "recall_hit": True,
            "missing_expected_terms": [],
            "missing_retrieval_terms": [],
            "latency_ms": diagnostics[0]["latency_ms"],
            "returned_tokens": diagnostics[0]["returned_tokens"],
            "injected_tokens": diagnostics[0]["injected_tokens"],
            "citation_hit": True,
            "top_contexts": diagnostics[0]["top_contexts"],
        },
        {
            "query": "embedded quality missing answer",
            "expected_terms": ["97 days"],
            "identity_terms": ["missing-agent", "agent-1"],
            "source_terms": [],
            "retrieval_terms": ["missing-agent", "agent-1"],
            "quality": 0.0,
            "answer_hit": False,
            "recall_quality": 1.0,
            "recall_hit": True,
            "missing_expected_terms": ["97 days"],
            "missing_retrieval_terms": ["missing-agent"],
            "latency_ms": diagnostics[1]["latency_ms"],
            "returned_tokens": diagnostics[1]["returned_tokens"],
            "injected_tokens": diagnostics[1]["injected_tokens"],
            "citation_hit": True,
            "top_contexts": diagnostics[1]["top_contexts"],
        },
    ]
    assert diagnostics[0]["returned_tokens"] > 0
    assert diagnostics[1]["returned_tokens"] > 0
    assert diagnostics[0]["latency_ms"] >= 0.0
    assert diagnostics[1]["latency_ms"] >= 0.0
    assert diagnostics[0]["top_contexts"] == [
        {
            "rank": 1,
            "source": "bm25",
            "score": diagnostics[0]["top_contexts"][0]["score"],
            "citation": diagnostics[0]["top_contexts"][0]["citation"],
            "snippet": diagnostics[0]["top_contexts"][0]["snippet"],
        }
    ]
    assert diagnostics[0]["top_contexts"][0]["citation"].startswith("eventloom://agent-1/events/1#")
    assert diagnostics[0]["top_contexts"][0]["score"] > 0
    assert diagnostics[0]["top_contexts"][0]["snippet"] == (
        '{"decision": "Embedded Kuzu should match Neo4j quality."}'
    )


def test_backend_workload_builder_preserves_longmemeval_identity_terms(tmp_path: Path) -> None:
    """Backend parity reports need LongMemEval retrieval targets, not only final answers."""
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "Which issue happened first?",
                    "answer": "GPS system not functioning correctly",
                    "answer_session_ids": ["answer-session"],
                    "haystack_session_ids": ["answer-session"],
                    "haystack_dates": ["2024/01/01 (Mon) 10:00"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "My car GPS system failed after the first service.",
                                "has_answer": True,
                            }
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    eventloom = tmp_path / "longmemeval.eventloom.jsonl"
    queries = tmp_path / "queries.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build-backend-shootout-workload.py",
            "--dataset",
            str(dataset),
            "--eventloom-output",
            str(eventloom),
            "--queries-output",
            str(queries),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(queries.read_text(encoding="utf-8"))
    assert payload == [
        {
            "query": "Which issue happened first?",
            "expected_terms": ["GPS system not functioning correctly"],
            "identity_terms": ["answer-session"],
            "source_terms": [],
        }
    ]


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
                        "backend": "bm25",
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
                        "backend": "bm25",
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
    assert "report: Markdown sidecar row for bm25:retrieve missing answer_at_5=1.0" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing recall_at_5=1.0" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing quality_per_1k_injected_tokens" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing answer_at_5_per_1k_injected_tokens" in result.stderr


def test_backend_shootout_guardrail_rejects_markdown_metric_values_in_wrong_columns(tmp_path: Path) -> None:
    """Markdown metric checks should validate the column, not just row membership."""
    report = tmp_path / "backend-shootout.json"
    eventloom_sha = "a" * 64
    queries_sha = "b" * 64
    workload_events_sha = "c" * 64
    workload_queries_sha = "d" * 64
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "harness": "zaxy-backend-shootout",
                "generated_at_utc": "2026-05-21T00:00:00Z",
                "eventloom_path": "events.jsonl",
                "queries_file": "queries.json",
                "session_id": "default",
                "query_count": 1,
                "event_count": 1,
                "limit": 5,
                "source_fingerprints": {
                    "eventloom_sha256": eventloom_sha,
                    "queries_sha256": queries_sha,
                },
                "workload_fingerprints": {
                    "events_sha256": workload_events_sha,
                    "queries_sha256": workload_queries_sha,
                },
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "answer_ready",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
                        "recall_at_5": 0.5,
                        "citation_coverage": 1.0,
                        "quality_per_1k_injected_tokens": 0.25,
                        "answer_at_5_per_1k_injected_tokens": 0.75,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Backend Shootout",
                "",
                "- Report schema version: `1`",
                "- Harness: `zaxy-backend-shootout`",
                "- Generated at UTC: `2026-05-21T00:00:00Z`",
                "- Eventloom path: `events.jsonl`",
                "- Queries file: `queries.json`",
                "- Session ID: `default`",
                "- Queries: `1`",
                "- Events: `1`",
                "- Limit: `5`",
                f"- Source Eventloom SHA-256: `{eventloom_sha}`",
                f"- Source queries SHA-256: `{queries_sha}`",
                f"- Workload events SHA-256: `{workload_events_sha}`",
                f"- Workload queries SHA-256: `{workload_queries_sha}`",
                "",
                "| Backend | Contract | Status | Answer@5 | Recall@5 | Citation coverage | Quality / 1k injected | Answer@5 / 1k injected |",
                "|---------|----------|--------|----------|----------|-------------------|------------------------|-------------------------|",
                "| embedded | answer_ready | ok | 0.5 | 1.0 | 1.0 | 0.75 | 0.25 |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: Markdown sidecar row for embedded:answer_ready missing answer_at_5=1.0" in result.stderr
    assert "report: Markdown sidecar row for embedded:answer_ready missing recall_at_5=0.5" in result.stderr
    assert (
        "report: Markdown sidecar row for embedded:answer_ready missing quality_per_1k_injected_tokens=0.25"
        in result.stderr
    )
    assert (
        "report: Markdown sidecar row for embedded:answer_ready missing answer_at_5_per_1k_injected_tokens=0.75"
        in result.stderr
    )


def test_backend_shootout_guardrail_rejects_markdown_operational_metric_mismatch(tmp_path: Path) -> None:
    """Markdown sidecars should expose latency and token-efficiency metrics from JSON."""
    report = tmp_path / "backend-shootout.json"
    eventloom_sha = "a" * 64
    queries_sha = "b" * 64
    workload_events_sha = "c" * 64
    workload_queries_sha = "d" * 64
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "harness": "zaxy-backend-shootout",
                "generated_at_utc": "2026-05-21T00:00:00Z",
                "eventloom_path": "events.jsonl",
                "queries_file": "queries.json",
                "session_id": "default",
                "query_count": 1,
                "event_count": 1,
                "limit": 5,
                "source_fingerprints": {
                    "eventloom_sha256": eventloom_sha,
                    "queries_sha256": queries_sha,
                },
                "workload_fingerprints": {
                    "events_sha256": workload_events_sha,
                    "queries_sha256": workload_queries_sha,
                },
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "first_checkout_ms": 1.234,
                        "checkout_p95_ms": 2.345,
                        "checkout_p99_ms": 3.456,
                        "mean_returned_tokens": 10.0,
                        "quality_per_1k_returned_tokens": 20.0,
                        "answer_at_5_per_1k_returned_tokens": 30.0,
                        "mean_injected_tokens": 40.0,
                        "quality_per_1k_injected_tokens": 50.0,
                        "answer_at_5_per_1k_injected_tokens": 60.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Backend Shootout",
                "",
                "- Report schema version: `1`",
                "- Harness: `zaxy-backend-shootout`",
                "- Generated at UTC: `2026-05-21T00:00:00Z`",
                "- Eventloom path: `events.jsonl`",
                "- Queries file: `queries.json`",
                "- Session ID: `default`",
                "- Queries: `1`",
                "- Events: `1`",
                "- Limit: `5`",
                f"- Source Eventloom SHA-256: `{eventloom_sha}`",
                f"- Source queries SHA-256: `{queries_sha}`",
                f"- Workload events SHA-256: `{workload_events_sha}`",
                f"- Workload queries SHA-256: `{workload_queries_sha}`",
                "",
                "| Backend | Contract | Status | First checkout ms | Checkout p95 ms | Checkout p99 ms | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected |",
                "|---------|----------|--------|-------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|",
                "| embedded | retrieve | ok | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: Markdown sidecar row for embedded:retrieve missing first_checkout_ms=1.234" in result.stderr
    assert "report: Markdown sidecar row for embedded:retrieve missing checkout_p95_ms=2.345" in result.stderr
    assert "report: Markdown sidecar row for embedded:retrieve missing checkout_p99_ms=3.456" in result.stderr
    assert "report: Markdown sidecar row for embedded:retrieve missing mean_returned_tokens=10.0" in result.stderr
    assert "report: Markdown sidecar row for embedded:retrieve missing quality_per_1k_returned_tokens=20.0" in result.stderr
    assert "report: Markdown sidecar row for embedded:retrieve missing answer_at_5_per_1k_returned_tokens=30.0" in result.stderr


def test_backend_shootout_guardrail_rejects_markdown_remaining_operational_metric_mismatch(
    tmp_path: Path,
) -> None:
    """Markdown sidecars should match all rendered operational metric groups."""
    report = tmp_path / "backend-shootout.json"
    eventloom_sha = "a" * 64
    queries_sha = "b" * 64
    workload_events_sha = "c" * 64
    workload_queries_sha = "d" * 64
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "harness": "zaxy-backend-shootout",
                "generated_at_utc": "2026-05-21T00:00:00Z",
                "eventloom_path": "events.jsonl",
                "queries_file": "queries.json",
                "session_id": "default",
                "query_count": 1,
                "event_count": 1,
                "limit": 5,
                "source_fingerprints": {
                    "eventloom_sha256": eventloom_sha,
                    "queries_sha256": queries_sha,
                },
                "workload_fingerprints": {
                    "events_sha256": workload_events_sha,
                    "queries_sha256": workload_queries_sha,
                },
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "cold_bootstrap_ms": 1.0,
                        "first_useful_init_ms": 2.0,
                        "append_to_projection_p95_ms": 3.0,
                        "projection_events_per_second": 4.0,
                        "exact_p50_ms": 5.0,
                        "exact_p95_ms": 6.0,
                        "exact_p99_ms": 7.0,
                        "keyword_p50_ms": 8.0,
                        "keyword_p95_ms": 9.0,
                        "keyword_p99_ms": 10.0,
                        "vector_p50_ms": 11.0,
                        "vector_p95_ms": 12.0,
                        "vector_p99_ms": 13.0,
                        "traversal_p50_ms": 14.0,
                        "traversal_p95_ms": 15.0,
                        "traversal_p99_ms": 16.0,
                        "dashboard_graph_load_ms": 17.0,
                        "dashboard_graph_source": "embedded",
                        "dashboard_graph_nodes": 18,
                        "dashboard_graph_edges": 19,
                        "memory_footprint_bytes": 20,
                        "resident_memory_delta_bytes": 21,
                        "on_disk_footprint_bytes": 22,
                        "rebuild_recovery_ms": 23.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Backend Shootout",
                "",
                "- Report schema version: `1`",
                "- Harness: `zaxy-backend-shootout`",
                "- Generated at UTC: `2026-05-21T00:00:00Z`",
                "- Eventloom path: `events.jsonl`",
                "- Queries file: `queries.json`",
                "- Session ID: `default`",
                "- Queries: `1`",
                "- Events: `1`",
                "- Limit: `5`",
                f"- Source Eventloom SHA-256: `{eventloom_sha}`",
                f"- Source queries SHA-256: `{queries_sha}`",
                f"- Workload events SHA-256: `{workload_events_sha}`",
                f"- Workload queries SHA-256: `{workload_queries_sha}`",
                "",
                "| Backend | Contract | Status | Cold bootstrap ms | First useful init ms | Append projection p95 ms | Projection eps | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |",
                "|---------|----------|--------|-------------------|----------------------|--------------------------|----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|--------------|-----------------------------|-------------------------|---------------------|",
                "| bm25 | retrieve | ok | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | stale | 0 | 0 | 0 | 0 | 0 | 0.0 |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
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
    assert "report: Markdown sidecar row for bm25:retrieve missing cold_bootstrap_ms=1.0" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing exact_p95_ms=6.0" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing keyword_p99_ms=10.0" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing vector_p99_ms=13.0" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing traversal_p99_ms=16.0" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing dashboard_graph_source=embedded" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing memory_footprint_bytes=20" in result.stderr
    assert "report: Markdown sidecar row for bm25:retrieve missing rebuild_recovery_ms=23.0" in result.stderr


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
    assert "embedded:retrieve: duplicate backend summary rows found" in result.stderr


def test_backend_shootout_guardrail_treats_legacy_rows_as_retrieve_contract(tmp_path: Path) -> None:
    """Rows without contract should collide with explicit retrieve rows."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "status": "ok", "query_count": 1},
                    {"backend": "embedded", "contract": "retrieve", "status": "ok", "query_count": 1},
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
    assert "embedded:retrieve: duplicate backend summary rows found" in result.stderr


def test_backend_shootout_guardrail_rejects_unknown_contract_name(tmp_path: Path) -> None:
    """Typoed contract lanes must not satisfy required backend evidence."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "retrive",
                        "status": "ok",
                        "query_count": 1,
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
            "--require-labeled-metrics",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: summaries[0].contract is 'retrive', expected one of: answer_ready, retrieve" in result.stderr


def test_backend_shootout_guardrail_requires_retrieve_contract_evidence(tmp_path: Path) -> None:
    """Answer-ready rows must not stand in for raw retrieval evidence."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "answer_ready",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
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
    assert "embedded: missing required retrieve contract summary" in result.stderr


def test_backend_shootout_guardrail_requires_answer_ready_graph_contract(tmp_path: Path) -> None:
    """Explicit graph-backend contract reports must include answer-ready evidence."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                    },
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
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
            "embedded,bm25",
            "--require-labeled-metrics",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "embedded: missing required answer_ready contract summary" in result.stderr
    assert "bm25: missing required answer_ready contract summary" not in result.stderr


def test_backend_shootout_guardrail_rejects_query_result_contract_mismatch(tmp_path: Path) -> None:
    """Per-query diagnostics must use the same backend-contract keys as summaries."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                    },
                    {
                        "backend": "embedded",
                        "contract": "answer_ready",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                    },
                ],
                "query_results": {
                    "embedded:retrieve": [],
                    "embedded:answer-redy": [],
                },
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
    assert "report: query_results missing diagnostics for embedded:answer_ready" in result.stderr
    assert "report: query_results contains diagnostics for embedded:answer-redy without matching summary" in result.stderr


def test_backend_shootout_guardrail_requires_query_results_when_requested(tmp_path: Path) -> None:
    """Release gates should be able to require auditable per-query diagnostics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps({"summaries": [_guardrail_summary()]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
            "--require-query-results",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: query_results are required" in result.stderr


def test_backend_shootout_guardrail_requires_git_tracked_inputs_when_requested(tmp_path: Path) -> None:
    """Release evidence should not depend on untracked local benchmark inputs."""
    module = _load_backend_shootout_module()
    eventloom = tmp_path / "events"
    event = EventLog(eventloom / "default.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Track benchmark source inputs."},
        thread="default",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([{"query": "benchmark source inputs"}]), encoding="utf-8")
    query_specs = module._load_queries(queries)
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "harness": "zaxy-backend-shootout",
                "generated_at_utc": "2026-05-21T00:00:00Z",
                "eventloom_path": str(eventloom),
                "queries_file": str(queries),
                "session_id": "default",
                "query_count": 1,
                "event_count": 1,
                "limit": 5,
                "source_fingerprints": {
                    "eventloom_sha256": module._path_fingerprint(eventloom),
                    "queries_sha256": module._path_fingerprint(queries),
                },
                "workload_fingerprints": {
                    "events_sha256": module._events_fingerprint([event]),
                    "queries_sha256": module._queries_fingerprint(query_specs),
                },
                "summaries": [_guardrail_summary()],
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
            "--verify-report-fingerprints",
            "--require-git-tracked-inputs",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert f"report: eventloom_path {eventloom} is not tracked by git" in result.stderr
    assert f"report: queries_file {queries} is not tracked by git" in result.stderr


def test_backend_shootout_guardrail_accepts_git_tracked_sample_inputs() -> None:
    """The tracked smoke report should be eligible for stricter reproducibility checks."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            "reports/backend-shootout/backend-shootout.json",
            "--require-report-metadata",
            "--verify-report-fingerprints",
            "--require-git-tracked-inputs",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_backend_shootout_guardrail_rejects_stale_query_diagnostics(
    tmp_path: Path,
) -> None:
    """Per-query diagnostics should align with the current query workload order."""
    module = _load_backend_shootout_module()
    eventloom = tmp_path / "events"
    event = EventLog(eventloom / "default.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Pin diagnostic queries to workload inputs."},
        thread="default",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "current query", "expected_terms": ["current"]}]),
        encoding="utf-8",
    )
    query_specs = module._load_queries(queries)
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "harness": "zaxy-backend-shootout",
                "generated_at_utc": "2026-05-21T00:00:00Z",
                "eventloom_path": str(eventloom),
                "queries_file": str(queries),
                "session_id": "default",
                "query_count": 1,
                "event_count": 1,
                "limit": 5,
                "source_fingerprints": {
                    "eventloom_sha256": module._path_fingerprint(eventloom),
                    "queries_sha256": module._path_fingerprint(queries),
                },
                "workload_fingerprints": {
                    "events_sha256": module._events_fingerprint([event]),
                    "queries_sha256": module._queries_fingerprint(query_specs),
                },
                "summaries": [
                    _guardrail_summary(
                        backend="bm25",
                        contract="retrieve",
                        mean_quality=1.0,
                        answer_at_5=1.0,
                        recall_at_5=1.0,
                        citation_coverage=0.0,
                        mean_returned_tokens=1.0,
                        quality_per_1k_returned_tokens=1000.0,
                        answer_at_5_per_1k_returned_tokens=1000.0,
                        mean_injected_tokens=1.0,
                        quality_per_1k_injected_tokens=1000.0,
                        answer_at_5_per_1k_injected_tokens=1000.0,
                    )
                ],
                "query_results": {
                    "bm25:retrieve": [
                        _guardrail_diagnostic(
                            query="stale query",
                            expected_terms=["current"],
                            retrieval_terms=["current"],
                            quality=1.0,
                            answer_hit=True,
                            recall_quality=1.0,
                            recall_hit=True,
                            citation_hit=False,
                        )
                    ]
                },
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
            "--verify-report-fingerprints",
            "--require-query-results",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: query_results['bm25:retrieve'][0].query does not match current query workload" in result.stderr


def test_backend_shootout_guardrail_rejects_non_list_query_results(tmp_path: Path) -> None:
    """Per-query diagnostics values should stay machine-readable lists."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                    }
                ],
                "query_results": {"bm25:retrieve": {"query": "not a list"}},
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: query_results['bm25:retrieve'] must be a list" in result.stderr


def test_backend_shootout_guardrail_rejects_zero_score_vector_contexts(tmp_path: Path) -> None:
    """Vector diagnostics should not preserve stale zero-similarity padding."""
    report = _write_guardrail_report(
        tmp_path,
        summary=_guardrail_summary(citation_coverage=0.0),
        diagnostic=_guardrail_diagnostic(
            citation_hit=False,
            top_contexts=[
                {
                    "rank": 1,
                    "source": "vector",
                    "score": 0.0,
                    "citation": None,
                    "snippet": "unrelated padded vector result",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
            "--require-query-results",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: query_results['bm25:retrieve'][0].top_contexts[0].score must be positive for vector source" in result.stderr


def test_backend_shootout_guardrail_rejects_query_result_count_mismatch(tmp_path: Path) -> None:
    """Per-query diagnostics should match each summary row's measured query count."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 2,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                    },
                    {
                        "backend": "embedded",
                        "contract": "answer_ready",
                        "status": "ok",
                        "query_count": 2,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                    },
                ],
                "query_results": {
                    "embedded:retrieve": [{"query": "one"}],
                    "embedded:answer_ready": [{"query": "one"}, {"query": "two"}],
                },
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
    assert "report: query_results['embedded:retrieve'] has 1 diagnostics, expected 2" in result.stderr


def test_backend_shootout_guardrail_checks_query_result_count_before_duplicate_overwrite(
    tmp_path: Path,
) -> None:
    """Duplicate summaries should not hide stale query diagnostic counts."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "bm25", "contract": "retrieve", "status": "ok", "query_count": 2},
                    {"backend": "bm25", "contract": "retrieve", "status": "ok", "query_count": 1},
                ],
                "query_results": {"bm25:retrieve": [{"query": "one"}]},
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25:retrieve: duplicate backend summary rows found" in result.stderr
    assert "report: query_results['bm25:retrieve'] has 1 diagnostics, expected 2" in result.stderr


def test_backend_shootout_guardrail_rejects_malformed_query_diagnostic_item(tmp_path: Path) -> None:
    """Per-query diagnostics should keep the generated machine-readable schema."""
    report = _write_guardrail_report(
        tmp_path,
        diagnostic={
            "query": "",
            "quality": "1.0",
            "answer_hit": "yes",
            "latency_ms": -1.0,
            "returned_tokens": -2,
            "injected_tokens": 3,
            "top_contexts": {},
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: query_results['bm25:retrieve'][0].query must be a non-empty string" in result.stderr
    assert "report: query_results['bm25:retrieve'][0].quality must be a rate between 0 and 1" in result.stderr
    assert "report: query_results['bm25:retrieve'][0].answer_hit must be a boolean" in result.stderr
    assert "report: query_results['bm25:retrieve'][0].latency_ms must be a non-negative number" in result.stderr
    assert "report: query_results['bm25:retrieve'][0].returned_tokens must be a non-negative integer" in result.stderr
    assert "report: query_results['bm25:retrieve'][0].top_contexts must be a list" in result.stderr


def test_backend_shootout_guardrail_rejects_malformed_top_context_diagnostic(tmp_path: Path) -> None:
    """Top-context diagnostics should keep stable rank, source, score, and snippet fields."""
    report = _write_guardrail_report(
        tmp_path,
        summary=_guardrail_summary(citation_coverage=0.0),
        diagnostic=_guardrail_diagnostic(
            citation_hit=False,
            top_contexts=[
                {
                    "rank": 0,
                    "source": "",
                    "score": "1.0",
                    "snippet": "",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    prefix = "report: query_results['bm25:retrieve'][0].top_contexts[0]"
    assert f"{prefix}.rank must be a positive integer" in result.stderr
    assert f"{prefix}.source must be a non-empty string" in result.stderr
    assert f"{prefix}.score must be a non-negative number" in result.stderr
    assert f"{prefix}.snippet must be a non-empty string" in result.stderr


def test_backend_shootout_guardrail_rejects_unknown_top_context_source(tmp_path: Path) -> None:
    """Top-context source labels should stay tied to measured retrieval lanes."""
    report = _write_guardrail_report(
        tmp_path,
        summary=_guardrail_summary(citation_coverage=0.0),
        diagnostic=_guardrail_diagnostic(
            citation_hit=False,
            top_contexts=[
                {
                    "rank": 1,
                    "source": "vectro",
                    "score": 1.0,
                    "citation": None,
                    "snippet": "typoed source lane",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: query_results['bm25:retrieve'][0].top_contexts[0].source is not recognized" in result.stderr


def test_backend_shootout_guardrail_rejects_non_string_query_diagnostic_terms(tmp_path: Path) -> None:
    """Term arrays in query diagnostics should contain only strings."""
    report = _write_guardrail_report(
        tmp_path,
        diagnostic=_guardrail_diagnostic(
            expected_terms=["alpha", 42],
            identity_terms=[None],
            source_terms=["source"],
            retrieval_terms=["alpha", 42, None],
            missing_expected_terms=[False],
            missing_retrieval_terms=["ok"],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    prefix = "report: query_results['bm25:retrieve'][0]"
    assert f"{prefix}.expected_terms[1] must be a string" in result.stderr
    assert f"{prefix}.identity_terms[0] must be a string" in result.stderr
    assert f"{prefix}.retrieval_terms[1] must be a string" in result.stderr
    assert f"{prefix}.retrieval_terms[2] must be a string" in result.stderr
    assert f"{prefix}.missing_expected_terms[0] must be a string" in result.stderr


def test_backend_shootout_guardrail_rejects_blank_top_context_citation(tmp_path: Path) -> None:
    """Top-context citation values should be null or useful non-empty strings."""
    report = _write_guardrail_report(
        tmp_path,
        diagnostic=_guardrail_diagnostic(
            top_contexts=[
                {
                    "rank": 1,
                    "source": "bm25",
                    "score": 1.0,
                    "citation": "  ",
                    "snippet": "memory",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    prefix = "report: query_results['bm25:retrieve'][0].top_contexts[0]"
    assert f"{prefix}.citation must be a non-empty string or null" in result.stderr


def test_backend_shootout_guardrail_rejects_unsupported_top_context_citation_scheme(
    tmp_path: Path,
) -> None:
    """Top-context citations should be auditable Eventloom or source-file references."""
    report = _write_guardrail_report(
        tmp_path,
        diagnostic=_guardrail_diagnostic(
            top_contexts=[
                {
                    "rank": 1,
                    "source": "bm25",
                    "score": 1.0,
                    "citation": "opaque-reference",
                    "snippet": "memory",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    prefix = "report: query_results['bm25:retrieve'][0].top_contexts[0]"
    assert f"{prefix}.citation must start with eventloom:// or file://" in result.stderr


def test_backend_shootout_guardrail_rejects_malformed_eventloom_top_context_citation(
    tmp_path: Path,
) -> None:
    """Eventloom top-context citations should include thread, positive seq, and hash."""
    report = _write_guardrail_report(
        tmp_path,
        diagnostic=_guardrail_diagnostic(
            top_contexts=[
                {
                    "rank": 1,
                    "source": "bm25",
                    "score": 1.0,
                    "citation": "eventloom://agent-1/events/not-a-seq#abc",
                    "snippet": "memory",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    prefix = "report: query_results['bm25:retrieve'][0].top_contexts[0]"
    assert f"{prefix}.citation must match eventloom://<thread>/events/<seq>#<hash>" in result.stderr


def test_backend_shootout_guardrail_rejects_malformed_file_top_context_citation(
    tmp_path: Path,
) -> None:
    """File top-context citations should include a concrete source path."""
    report = _write_guardrail_report(
        tmp_path,
        diagnostic=_guardrail_diagnostic(
            top_contexts=[
                {
                    "rank": 1,
                    "source": "bm25",
                    "score": 1.0,
                    "citation": "file://",
                    "snippet": "memory",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    prefix = "report: query_results['bm25:retrieve'][0].top_contexts[0]"
    assert f"{prefix}.citation must include a file:// path" in result.stderr


def test_backend_shootout_guardrail_rejects_citation_hit_without_cited_context(
    tmp_path: Path,
) -> None:
    """Citation-hit diagnostics should expose at least one cited top context."""
    report = _write_guardrail_report(
        tmp_path,
        diagnostic=_guardrail_diagnostic(
            citation_hit=True,
            top_contexts=[
                {
                    "rank": 1,
                    "source": "bm25",
                    "score": 1.0,
                    "citation": None,
                    "snippet": "memory",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert (
        "report: query_results['bm25:retrieve'][0].citation_hit requires at least one cited top_context"
        in result.stderr
    )


def test_backend_shootout_guardrail_rejects_cited_context_without_citation_hit(
    tmp_path: Path,
) -> None:
    """Citation-hit diagnostics should agree with cited top-context evidence."""
    report = _write_guardrail_report(
        tmp_path,
        summary=_guardrail_summary(citation_coverage=0.0),
        diagnostic=_guardrail_diagnostic(
            citation_hit=False,
            top_contexts=[
                {
                    "rank": 1,
                    "source": "bm25",
                    "score": 1.0,
                    "citation": "eventloom://agent-1/events/1#abc",
                    "snippet": "memory",
                }
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert (
        "report: query_results['bm25:retrieve'][0].citation_hit must equal presence of a cited top_context"
        in result.stderr
    )


def test_backend_shootout_guardrail_rejects_unstable_top_context_ranks(tmp_path: Path) -> None:
    """Top-context diagnostics should use deterministic contiguous ranks."""
    report = _write_guardrail_report(
        tmp_path,
        diagnostic=_guardrail_diagnostic(
            top_contexts=[
                {
                    "rank": 1,
                    "source": "bm25",
                    "score": 1.0,
                    "citation": None,
                    "snippet": "first",
                },
                {
                    "rank": 1,
                    "source": "bm25",
                    "score": 0.9,
                    "citation": None,
                    "snippet": "duplicate",
                },
            ],
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-backends",
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert (
        "report: query_results['bm25:retrieve'][0].top_contexts ranks must be contiguous from 1"
        in result.stderr
    )


def test_backend_shootout_guardrail_rejects_inconsistent_query_hit_flags(tmp_path: Path) -> None:
    """Hit booleans should agree with quality and recall_quality values."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "bm25", "contract": "retrieve", "status": "ok", "query_count": 2}
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "What happened?",
                            "expected_terms": [],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": [],
                            "quality": 1.0,
                            "answer_hit": False,
                            "recall_quality": 0.5,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 0.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        },
                        {
                            "query": "What happened next?",
                            "expected_terms": [],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": [],
                            "quality": 0.5,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": False,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 0.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        },
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    prefix = "report: query_results['bm25:retrieve']"
    assert f"{prefix}[0].answer_hit must equal quality >= 1.0" in result.stderr
    assert f"{prefix}[0].recall_hit must equal recall_quality >= 1.0" in result.stderr
    assert f"{prefix}[1].answer_hit must equal quality >= 1.0" in result.stderr
    assert f"{prefix}[1].recall_hit must equal recall_quality >= 1.0" in result.stderr


def test_backend_shootout_guardrail_rejects_missing_terms_outside_term_sets(tmp_path: Path) -> None:
    """Missing-term diagnostics should refer to terms tracked for the query."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "bm25", "contract": "retrieve", "status": "ok", "query_count": 1}
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "What happened?",
                            "expected_terms": ["alpha"],
                            "identity_terms": [],
                            "source_terms": ["source-a"],
                            "retrieval_terms": ["alpha", "source-a"],
                            "quality": 0.0,
                            "answer_hit": False,
                            "recall_quality": 0.0,
                            "recall_hit": False,
                            "missing_expected_terms": ["beta"],
                            "missing_retrieval_terms": ["source-b"],
                            "latency_ms": 0.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        }
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    prefix = "report: query_results['bm25:retrieve'][0]"
    assert f"{prefix}.missing_expected_terms[0] is not present in expected_terms" in result.stderr
    assert f"{prefix}.missing_retrieval_terms[0] is not present in retrieval_terms" in result.stderr


def test_backend_shootout_guardrail_rejects_summary_metrics_that_disagree_with_query_results(
    tmp_path: Path,
) -> None:
    """Summary quality metrics should agree with included per-query diagnostics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 2,
                        "mean_quality": 1.0,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                    }
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "first",
                            "expected_terms": ["alpha"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["alpha"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 0.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        },
                        {
                            "query": "second",
                            "expected_terms": ["beta"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["beta"],
                            "quality": 0.0,
                            "answer_hit": False,
                            "recall_quality": 0.0,
                            "recall_hit": False,
                            "missing_expected_terms": ["beta"],
                            "missing_retrieval_terms": ["beta"],
                            "latency_ms": 0.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": False,
                            "top_contexts": [],
                        },
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25:retrieve: mean_quality=1.0 does not match query_results aggregate 0.5" in result.stderr
    assert "bm25:retrieve: answer_at_5=1.0 does not match query_results aggregate 0.5" in result.stderr
    assert "bm25:retrieve: recall_at_5=1.0 does not match query_results aggregate 0.5" in result.stderr
    assert "bm25:retrieve: citation_coverage=1.0 does not match query_results aggregate 0.5" in result.stderr


def test_backend_shootout_guardrail_allows_unlabeled_query_result_aggregates(tmp_path: Path) -> None:
    """Unlabeled diagnostics should still validate mean quality and citation coverage."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "mean_quality": 0.0,
                        "answer_at_5": None,
                        "recall_at_5": None,
                        "citation_coverage": 1.0,
                        "mean_returned_tokens": 1.0,
                        "quality_per_1k_returned_tokens": 0.0,
                        "mean_injected_tokens": 1.0,
                        "quality_per_1k_injected_tokens": 0.0,
                        "first_checkout_ms": 0.0,
                        "checkout_p95_ms": 0.0,
                        "checkout_p99_ms": 0.0,
                    }
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "unlabeled",
                            "expected_terms": [],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": [],
                            "quality": 0.0,
                            "answer_hit": False,
                            "recall_quality": 0.0,
                            "recall_hit": False,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 0.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [
                                {
                                    "rank": 1,
                                    "source": "keyword",
                                    "score": 1.0,
                                    "citation": "eventloom://default/events/1#abcdef",
                                    "snippet": "cited unlabeled context",
                                }
                            ],
                        }
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend shootout guardrail passed" in result.stdout


def test_backend_shootout_guardrail_rejects_token_efficiency_summary_mismatch(
    tmp_path: Path,
) -> None:
    """Token-efficiency summaries should agree with included per-query diagnostics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 2,
                        "mean_quality": 0.5,
                        "answer_at_5": 0.5,
                        "recall_at_5": 0.5,
                        "citation_coverage": 0.5,
                        "mean_returned_tokens": 100.0,
                        "quality_per_1k_returned_tokens": 10.0,
                        "answer_at_5_per_1k_returned_tokens": 10.0,
                        "mean_injected_tokens": 200.0,
                        "quality_per_1k_injected_tokens": 10.0,
                        "answer_at_5_per_1k_injected_tokens": 10.0,
                    }
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "first",
                            "expected_terms": ["alpha"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["alpha"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 0.0,
                            "returned_tokens": 100,
                            "injected_tokens": 200,
                            "citation_hit": True,
                            "top_contexts": [],
                        },
                        {
                            "query": "second",
                            "expected_terms": ["beta"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["beta"],
                            "quality": 0.0,
                            "answer_hit": False,
                            "recall_quality": 0.0,
                            "recall_hit": False,
                            "missing_expected_terms": ["beta"],
                            "missing_retrieval_terms": ["beta"],
                            "latency_ms": 0.0,
                            "returned_tokens": 100,
                            "injected_tokens": 200,
                            "citation_hit": False,
                            "top_contexts": [],
                        },
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25:retrieve: quality_per_1k_returned_tokens=10.0 does not match query_results aggregate 5.0" in result.stderr
    assert "bm25:retrieve: answer_at_5_per_1k_returned_tokens=10.0 does not match query_results aggregate 5.0" in result.stderr
    assert "bm25:retrieve: quality_per_1k_injected_tokens=10.0 does not match query_results aggregate 2.5" in result.stderr
    assert "bm25:retrieve: answer_at_5_per_1k_injected_tokens=10.0 does not match query_results aggregate 2.5" in result.stderr


def test_backend_shootout_guardrail_allows_zero_token_efficiency_aggregates(tmp_path: Path) -> None:
    """Zero-token diagnostics should not force impossible per-1k summary metrics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "mean_quality": 1.0,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "mean_returned_tokens": 0.0,
                        "quality_per_1k_returned_tokens": None,
                        "answer_at_5_per_1k_returned_tokens": None,
                        "mean_injected_tokens": 0.0,
                        "quality_per_1k_injected_tokens": None,
                        "answer_at_5_per_1k_injected_tokens": None,
                        "first_checkout_ms": 0.0,
                        "checkout_p95_ms": 0.0,
                        "checkout_p99_ms": 0.0,
                    }
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "first",
                            "expected_terms": ["alpha"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["alpha"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 0.0,
                            "returned_tokens": 0,
                            "injected_tokens": 0,
                            "citation_hit": True,
                            "top_contexts": [
                                {
                                    "rank": 1,
                                    "source": "keyword",
                                    "score": 1.0,
                                    "citation": "eventloom://default/events/1#abcdef",
                                    "snippet": "cited zero-token context",
                                }
                            ],
                        }
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend shootout guardrail passed" in result.stdout


def test_backend_shootout_guardrail_rejects_latency_summary_mismatch(
    tmp_path: Path,
) -> None:
    """Latency summaries should agree with included per-query diagnostics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 3,
                        "mean_quality": 1.0,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "first_checkout_ms": 99.0,
                        "checkout_p95_ms": 99.0,
                        "checkout_p99_ms": 99.0,
                    }
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "first",
                            "expected_terms": ["alpha"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["alpha"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 1.234,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        },
                        {
                            "query": "second",
                            "expected_terms": ["beta"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["beta"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 2.345,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        },
                        {
                            "query": "third",
                            "expected_terms": ["gamma"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["gamma"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 3.456,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        },
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25:retrieve: first_checkout_ms=99.0 does not match query_results aggregate 1.234" in result.stderr
    assert "bm25:retrieve: checkout_p95_ms=99.0 does not match query_results aggregate 3.456" in result.stderr
    assert "bm25:retrieve: checkout_p99_ms=99.0 does not match query_results aggregate 3.456" in result.stderr


def test_backend_shootout_guardrail_allows_empty_query_results_with_null_latency_summaries(
    tmp_path: Path,
) -> None:
    """Empty diagnostic lanes should allow nullable latency summaries."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 0,
                        "first_checkout_ms": None,
                        "checkout_p95_ms": None,
                        "checkout_p99_ms": None,
                    }
                ],
                "query_results": {"bm25:retrieve": []},
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend shootout guardrail passed" in result.stdout


def test_backend_shootout_guardrail_rejects_null_summary_metric_with_query_results(
    tmp_path: Path,
) -> None:
    """Query-derived aggregates should not silently ignore missing summary metrics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "mean_quality": None,
                        "answer_at_5": None,
                        "first_checkout_ms": None,
                    }
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "first",
                            "expected_terms": ["alpha"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["alpha"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 1.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        }
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25:retrieve: mean_quality is missing; query_results aggregate is 1.0" in result.stderr
    assert "bm25:retrieve: answer_at_5 is missing; query_results aggregate is 1.0" in result.stderr
    assert "bm25:retrieve: first_checkout_ms is missing; query_results aggregate is 1.0" in result.stderr


def test_backend_shootout_guardrail_rejects_non_numeric_summary_metric_with_query_results(
    tmp_path: Path,
) -> None:
    """Query-derived summary metrics should not silently ignore non-numeric summary values."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "mean_quality": "1.0",
                        "answer_at_5": "1.0",
                        "recall_at_5": "1.0",
                        "citation_coverage": "1.0",
                        "first_checkout_ms": "1.0",
                    }
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "first",
                            "expected_terms": ["alpha"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["alpha"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 1.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        }
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25:retrieve: mean_quality must be a number to compare with query_results aggregate 1.0" in result.stderr
    assert "bm25:retrieve: answer_at_5 must be a number to compare with query_results aggregate 1.0" in result.stderr
    assert "bm25:retrieve: first_checkout_ms must be a number to compare with query_results aggregate 1.0" in result.stderr


def test_backend_shootout_guardrail_rejects_boolean_summary_metric_with_query_results(
    tmp_path: Path,
) -> None:
    """JSON booleans should not satisfy query-derived numeric summary metrics."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "bm25",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "mean_quality": True,
                        "answer_at_5": True,
                        "recall_at_5": True,
                        "citation_coverage": True,
                    }
                ],
                "query_results": {
                    "bm25:retrieve": [
                        {
                            "query": "first",
                            "expected_terms": ["alpha"],
                            "identity_terms": [],
                            "source_terms": [],
                            "retrieval_terms": ["alpha"],
                            "quality": 1.0,
                            "answer_hit": True,
                            "recall_quality": 1.0,
                            "recall_hit": True,
                            "missing_expected_terms": [],
                            "missing_retrieval_terms": [],
                            "latency_ms": 1.0,
                            "returned_tokens": 1,
                            "injected_tokens": 1,
                            "citation_hit": True,
                            "top_contexts": [],
                        }
                    ]
                },
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
            "bm25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bm25:retrieve: mean_quality must be a number to compare with query_results aggregate 1.0" in result.stderr
    assert "bm25:retrieve: citation_coverage must be a number to compare with query_results aggregate 1.0" in result.stderr


def test_backend_shootout_guardrail_names_duplicate_backend_contract(tmp_path: Path) -> None:
    """Duplicate diagnostics should identify the colliding benchmark contract."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {"backend": "embedded", "contract": "answer_ready", "status": "ok", "query_count": 1},
                    {"backend": "embedded", "contract": "answer_ready", "status": "ok", "query_count": 1},
                    {"backend": "embedded", "contract": "retrieve", "status": "ok", "query_count": 1},
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
    assert "embedded:answer_ready: duplicate backend summary rows found" in result.stderr
    assert "embedded:retrieve: duplicate backend summary rows found" not in result.stderr


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


def test_backend_shootout_guardrail_splits_answer_ready_quality_from_retrieve_latency(tmp_path: Path) -> None:
    """Answer-ready rows should carry answer quality without weakening retrieval latency gates."""
    report = tmp_path / "backend-shootout.json"
    report.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 0.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "dashboard_graph_source": "embedded",
                        "checkout_p95_ms": 20.0,
                        "answer_at_5_per_1k_injected_tokens": 0.0,
                    },
                    {
                        "backend": "embedded",
                        "contract": "answer_ready",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "dashboard_graph_source": "embedded",
                        "checkout_p95_ms": 2_000.0,
                        "answer_at_5_per_1k_injected_tokens": 0.2,
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
            "--require-dashboard-source",
            "embedded=embedded",
            "--min-answer-at-5",
            "1.0",
            "--min-recall-at-5",
            "1.0",
            "--min-citation-coverage",
            "1.0",
            "--min-answer-at-5-per-1k-injected-tokens",
            "embedded=0.1",
            "--max-checkout-p95-ms",
            "embedded=100",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend shootout guardrail passed" in result.stdout


def test_backend_shootout_guardrail_validates_contract_scoped_markdown_rows(tmp_path: Path) -> None:
    """Markdown sidecars should match backend and contract when multiple rows exist."""
    report = tmp_path / "backend-shootout.json"
    eventloom_sha = "a" * 64
    queries_sha = "b" * 64
    workload_events_sha = "c" * 64
    workload_queries_sha = "d" * 64
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "harness": "zaxy-backend-shootout",
                "generated_at_utc": "2026-05-21T00:00:00Z",
                "eventloom_path": "events.jsonl",
                "queries_file": "queries.json",
                "session_id": "default",
                "query_count": 1,
                "event_count": 1,
                "limit": 5,
                "source_fingerprints": {
                    "eventloom_sha256": eventloom_sha,
                    "queries_sha256": queries_sha,
                },
                "workload_fingerprints": {
                    "events_sha256": workload_events_sha,
                    "queries_sha256": workload_queries_sha,
                },
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 0.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "quality_per_1k_injected_tokens": 0.0,
                        "answer_at_5_per_1k_injected_tokens": 0.0,
                    },
                    {
                        "backend": "embedded",
                        "contract": "answer_ready",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "quality_per_1k_injected_tokens": 0.2,
                        "answer_at_5_per_1k_injected_tokens": 0.2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Backend Shootout",
                "",
                "- Report schema version: `1`",
                "- Harness: `zaxy-backend-shootout`",
                "- Generated at UTC: `2026-05-21T00:00:00Z`",
                "- Eventloom path: `events.jsonl`",
                "- Queries file: `queries.json`",
                "- Session ID: `default`",
                "- Queries: `1`",
                "- Events: `1`",
                "- Limit: `5`",
                f"- Source Eventloom SHA-256: `{eventloom_sha}`",
                f"- Source queries SHA-256: `{queries_sha}`",
                f"- Workload events SHA-256: `{workload_events_sha}`",
                f"- Workload queries SHA-256: `{workload_queries_sha}`",
                "",
                "| Backend | Contract | Status | Answer@5 | Recall@5 | Citation coverage | Quality / 1k injected | Answer@5 / 1k injected |",
                "|---------|----------|--------|----------|----------|-------------------|------------------------|-------------------------|",
                "| embedded | retrieve | ok | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 |",
                "| embedded | answer_ready | ok | 1.0 | 1.0 | 1.0 | 0.2 | 0.2 |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend shootout guardrail passed" in result.stdout


def test_backend_shootout_guardrail_names_missing_contract_scoped_markdown_row(tmp_path: Path) -> None:
    """Markdown sidecar errors should identify the missing backend contract."""
    report = tmp_path / "backend-shootout.json"
    eventloom_sha = "a" * 64
    queries_sha = "b" * 64
    workload_events_sha = "c" * 64
    workload_queries_sha = "d" * 64
    report.write_text(
        json.dumps(
            {
                "report_schema_version": 1,
                "harness": "zaxy-backend-shootout",
                "generated_at_utc": "2026-05-21T00:00:00Z",
                "eventloom_path": "events.jsonl",
                "queries_file": "queries.json",
                "session_id": "default",
                "query_count": 1,
                "event_count": 1,
                "limit": 5,
                "source_fingerprints": {
                    "eventloom_sha256": eventloom_sha,
                    "queries_sha256": queries_sha,
                },
                "workload_fingerprints": {
                    "events_sha256": workload_events_sha,
                    "queries_sha256": workload_queries_sha,
                },
                "summaries": [
                    {
                        "backend": "embedded",
                        "contract": "retrieve",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 0.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "quality_per_1k_injected_tokens": 0.0,
                        "answer_at_5_per_1k_injected_tokens": 0.0,
                    },
                    {
                        "backend": "embedded",
                        "contract": "answer_ready",
                        "status": "ok",
                        "query_count": 1,
                        "answer_at_5": 1.0,
                        "recall_at_5": 1.0,
                        "citation_coverage": 1.0,
                        "quality_per_1k_injected_tokens": 0.2,
                        "answer_at_5_per_1k_injected_tokens": 0.2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Backend Shootout",
                "",
                "- Report schema version: `1`",
                "- Harness: `zaxy-backend-shootout`",
                "- Generated at UTC: `2026-05-21T00:00:00Z`",
                "- Eventloom path: `events.jsonl`",
                "- Queries file: `queries.json`",
                "- Session ID: `default`",
                "- Queries: `1`",
                "- Events: `1`",
                "- Limit: `5`",
                f"- Source Eventloom SHA-256: `{eventloom_sha}`",
                f"- Source queries SHA-256: `{queries_sha}`",
                f"- Workload events SHA-256: `{workload_events_sha}`",
                f"- Workload queries SHA-256: `{workload_queries_sha}`",
                "",
                "| Backend | Contract | Status | Answer@5 | Recall@5 | Citation coverage | Quality / 1k injected | Answer@5 / 1k injected |",
                "|---------|----------|--------|----------|----------|-------------------|------------------------|-------------------------|",
                "| embedded | retrieve | ok | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            str(report),
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-backends",
            "embedded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "report: Markdown sidecar missing backend row for embedded:answer_ready" in result.stderr


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
            "--require-query-results",
            "--verify-report-fingerprints",
            "--require-backends",
            "embedded,bm25",
            "--forbid-backends",
            "neo4j,pggraph,latticedb",
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
            "--require-query-results",
            "--verify-report-fingerprints",
            "--require-backends",
            "embedded,bm25",
            "--forbid-backends",
            "neo4j,pggraph,latticedb",
            "--require-labeled-metrics",
            "--require-dashboard-source",
            "embedded=embedded",
            "--min-citation-coverage",
            "1.0",
            "--min-projection-events-per-second",
            "embedded=40",
            "--max-cold-bootstrap-ms",
            "embedded=250",
            "--max-first-useful-init-ms",
            "embedded=15000",
            "--max-first-checkout-ms",
            "embedded=50",
            "--max-append-to-projection-p95-ms",
            "embedded=35",
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
            "embedded=85",
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
    summaries = {(summary["backend"], summary["contract"]): summary for summary in report["summaries"]}

    embedded_retrieve = summaries[("embedded", "retrieve")]
    embedded_answer = summaries[("embedded", "answer_ready")]
    bm25 = summaries[("bm25", "retrieve")]

    assert embedded_retrieve["recall_at_5"] == 0.99
    assert embedded_retrieve["cold_bootstrap_ms"] == 421.649
    assert embedded_retrieve["checkout_p95_ms"] == 19.915
    assert embedded_retrieve["checkout_p99_ms"] == 21.937
    assert embedded_retrieve["mean_injected_tokens"] == 1492.24
    assert embedded_retrieve["quality_per_1k_injected_tokens"] == 0.3485
    assert embedded_answer["answer_at_5"] == 0.99
    assert embedded_answer["recall_at_5"] == 1.0
    assert embedded_answer["first_checkout_ms"] == 37.615
    assert embedded_answer["checkout_p95_ms"] == 90.478
    assert embedded_answer["mean_injected_tokens"] == 3426.8
    assert embedded_answer["quality_per_1k_injected_tokens"] == 0.2889
    assert embedded_answer["answer_at_5_per_1k_injected_tokens"] == 0.2889
    assert embedded_answer["resident_memory_delta_bytes"] == 1604280320
    assert embedded_answer["on_disk_footprint_bytes"] == 57298944
    assert bm25["mean_injected_tokens"] == 4179.5
    assert bm25["quality_per_1k_injected_tokens"] == 0.1244
    assert bm25["answer_at_5_per_1k_injected_tokens"] == 0.1244

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check-backend-shootout.py",
            "reports/backend-shootout/longmemeval-100-backend-shootout.json",
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-query-results",
            "--verify-report-fingerprints",
            "--require-backends",
            "embedded,bm25",
            "--forbid-backends",
            "neo4j,pggraph,latticedb",
            "--require-labeled-metrics",
            "--require-dashboard-source",
            "embedded=embedded",
            "--min-recall-at-5",
            "0.90",
            "--min-citation-coverage",
            "1.0",
            "--min-projection-events-per-second",
            "embedded=35",
            "--max-cold-bootstrap-ms",
            "embedded=600",
            "--max-first-useful-init-ms",
            "embedded=45000",
            "--max-first-checkout-ms",
            "embedded=150",
            "--max-append-to-projection-p95-ms",
            "embedded=40",
            "--max-resident-memory-delta-bytes",
            "embedded=1700000000",
            "--max-on-disk-footprint-bytes",
            "embedded=512000000",
            "--max-dashboard-graph-load-ms",
            "embedded=500",
            "--max-rebuild-recovery-ms",
            "embedded=45000",
            "--max-checkout-p95-ms",
            "embedded=200",
            "--max-checkout-p99-ms",
            "embedded=250",
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
            "embedded=20",
            "--max-keyword-p99-ms",
            "embedded=15",
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
            "identity_terms": ["answer-1"],
            "source_terms": [],
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


def test_backend_shootout_rolls_back_when_bulk_commit_fails() -> None:
    """Failed bulk commits should leave backend transactions closed cleanly."""
    module = _load_backend_shootout_module()

    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def begin_bulk_projection(self) -> None:
            self.calls.append("begin")

        async def commit_bulk_projection(self) -> None:
            self.calls.append("commit")
            raise RuntimeError("commit failed")

        async def rollback_bulk_projection(self) -> None:
            self.calls.append("rollback")

    class FakeFabric:
        def __init__(self) -> None:
            self.graph = FakeGraph()

        async def _project_event(self, event: object, *, session_id: str) -> None:
            self.graph.calls.append(f"project:{event}:{session_id}")

    fake = FakeFabric()

    import asyncio

    with pytest.raises(RuntimeError, match="commit failed"):
        asyncio.run(module._project_events(fake, ["a"], "agent-1"))

    assert fake.graph.calls == [
        "begin",
        "project:a:agent-1",
        "commit",
        "rollback",
    ]


def test_backend_shootout_preserves_projection_error_when_rollback_fails() -> None:
    """Rollback cleanup failures should not mask the benchmark root cause."""
    module = _load_backend_shootout_module()

    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def begin_bulk_projection(self) -> None:
            self.calls.append("begin")

        async def commit_bulk_projection(self) -> None:
            self.calls.append("commit")

        async def rollback_bulk_projection(self) -> None:
            self.calls.append("rollback")
            raise RuntimeError("rollback failed")

    class FakeFabric:
        def __init__(self) -> None:
            self.graph = FakeGraph()

        async def _project_event(self, event: object, *, session_id: str) -> None:
            self.graph.calls.append(f"project:{event}:{session_id}")
            raise RuntimeError("projection failed")

    fake = FakeFabric()

    import asyncio

    with pytest.raises(RuntimeError, match="projection failed"):
        asyncio.run(module._project_events(fake, ["a"], "agent-1"))

    assert fake.graph.calls == [
        "begin",
        "project:a:agent-1",
        "rollback",
    ]


def test_backend_shootout_graph_backend_error_preserves_contract_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph backend failures should keep retrieve and answer-ready rows distinct."""
    module = _load_backend_shootout_module()

    class FailingFabric:
        def __init__(self, **_kwargs: object) -> None:
            self.graph = object()

        async def connect(self) -> None:
            raise RuntimeError("projection unavailable")

        async def close(self) -> None:
            return None

    import zaxy.core

    monkeypatch.setattr(zaxy.core, "MemoryFabric", FailingFabric)
    args = module.argparse.Namespace(
        eventloom_path=tmp_path / ".eventloom",
        output=tmp_path / "backend-shootout.json",
        session_id="agent-1",
        limit=5,
    )

    import asyncio

    runs = asyncio.run(module._run_graph_backend("embedded", [], [], args))

    assert [run.metrics.contract for run in runs] == ["retrieve", "answer_ready"]
    assert {run.metrics.status for run in runs} == {"error"}
    assert {run.metrics.error for run in runs} == {"projection unavailable"}
    assert all(run.query_results == [] for run in runs)


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


@pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")
def test_backend_shootout_embedded_reports_retrieve_and_answer_ready_contracts(tmp_path: Path) -> None:
    """Graph backend reports should separate retrieval quality from answer-ready assembly quality."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "document.indexed",
        actor="assistant",
        payload={
            "path": "memory.md",
            "content": "I spent $25 on a bike chain and $40 on bike lights.",
            "start_line": 1,
            "end_line": 1,
        },
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            [
                {
                    "query": "How much total money have I spent on bike-related expenses?",
                    "expected_terms": ["65"],
                    "identity_terms": ["bike chain"],
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
    summaries = json.loads(output.read_text(encoding="utf-8"))["summaries"]
    contracts = {summary["contract"] for summary in summaries}
    assert contracts == {"retrieve", "answer_ready"}
    assert {summary["backend"] for summary in summaries} == {"embedded"}


@pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")
def test_backend_shootout_query_results_are_contract_scoped(tmp_path: Path) -> None:
    """Per-query diagnostics must not collapse retrieve and answer-ready contracts."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent-1.jsonl").append(
        "document.indexed",
        actor="assistant",
        payload={
            "path": "memory.md",
            "content": "I adopted a local-first embedded graph contract.",
            "start_line": 1,
            "end_line": 1,
        },
        thread="agent-1",
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"query": "embedded graph contract", "expected_terms": ["embedded graph"]}]),
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
            "--include-query-results",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    query_results = json.loads(output.read_text(encoding="utf-8"))["query_results"]
    assert set(query_results) == {"embedded:retrieve", "embedded:answer_ready"}
