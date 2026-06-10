"""Tests for Zaxy CLI helper commands."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.coordination import CoordinationManager
from zaxy.event import EventLog
from zaxy.release import package_version


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _cli_command_params(*path: str) -> tuple[set[str], set[str]]:
    """Return option strings and argument names for a nested Typer command."""
    command = get_command(app)
    for name in path:
        command = command.commands[name]
    options = {opt for param in command.params for opt in getattr(param, "opts", []) if opt.startswith("--")}
    arguments = {param.name for param in command.params if not any(opt.startswith("--") for opt in param.opts)}
    return options, arguments


def _write_rc1_freeze_artifacts(root: Path) -> None:
    _write_rc1_project_benchmark_artifacts(root)
    manifest_dir = root / "reports/benchmarks/2.0.0-rc.1"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "release": "2.0.0-rc.1",
                "schema_version": "zaxy.rc1-benchmark-freeze.v1",
                "headline_500": {
                    "artifact": "reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json",
                    "run_config": "reports/benchmarks/longmemeval-500-publish-20260607/run-config.md",
                    "claim_scope": "longmemeval_compatible_checkout",
                    "workload_sha256": "90fb2307195d7e16b963a2b8a30f03b375bd42a45d41aeaa55423029dd84e3fc",
                },
                "harvey_lab": {
                    "claim_scope": "external_anchor",
                    "harvey_commit": "29748828133dff83ad2263af353fb035504f8f77",
                },
                "internal_lanes": [
                    {"lane": "causal", "claim_scope": "project_defined_internal"},
                    {"lane": "consolidation", "claim_scope": "project_defined_internal"},
                    {"lane": "procedural", "claim_scope": "project_defined_internal"},
                    {"lane": "metacognition", "claim_scope": "project_defined_internal"},
                ],
                "project_benchmarks": {
                    "state_recovery": {
                        "artifact": "reports/benchmarks/state-recovery-v1/state-recovery-benchmark.json",
                        "workload": "reports/benchmarks/state-recovery-v1/state-recovery-workload.json",
                        "markdown": "reports/benchmarks/state-recovery-v1/state-recovery-benchmark.md",
                        "claim_scope": "project_defined_internal",
                        "version": "state-recovery-v0",
                        "workload_fingerprint": "state-fingerprint",
                    },
                    "coordination": {
                        "artifact": "reports/benchmarks/coordination-v1/coordination-benchmark.json",
                        "workload": "reports/benchmarks/coordination-v1/coordination-workload.json",
                        "markdown": "reports/benchmarks/coordination-v1/coordination-benchmark.md",
                        "claim_scope": "project_defined_internal",
                        "version": "coordination-v1",
                        "workload_fingerprint": "coordination-fingerprint",
                    },
                    "purpose": {
                        "artifact": "reports/benchmarks/purpose-v1/purpose-benchmark.json",
                        "markdown": "reports/benchmarks/purpose-v1/purpose-benchmark.md",
                        "holdout_pack": "reports/benchmarks/purpose-v1/holdouts/public-derived-purpose-v1/holdout-pack.json",
                        "holdout_fingerprint": "holdout-fingerprint",
                        "source_disclosures": "reports/benchmarks/purpose-v1/holdouts/public-derived-purpose-v1/source-disclosures.json",
                        "claim_scope": "project_defined_internal",
                        "version": "purpose-v1",
                        "lane_count": 10,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    headline_dir = root / "reports/benchmarks/longmemeval-500-publish-20260607"
    headline_dir.mkdir(parents=True)
    (headline_dir / "run-config.md").write_text("frozen config\n", encoding="utf-8")
    (headline_dir / "live-benchmark.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-07T16:20:10Z",
                "workload": {"sha256": "90fb2307195d7e16b963a2b8a30f03b375bd42a45d41aeaa55423029dd84e3fc"},
                "summaries": [
                    {
                        "backend": "zaxy-checkout",
                        "case_count": 500,
                        "mean_score": 0.956,
                        "mean_answer_recall_at_5": 0.91,
                        "mean_recall_at_5": 1.0,
                        "mean_citation_coverage": 1.0,
                        "latency_ms_p95": 1966.65,
                        "latency_ms_p99": 2495.07,
                        "mean_approx_tokens": 10192,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    harvey_dir = root / "reports/benchmarks/harvey-lab-memory-ablation"
    harvey_dir.mkdir(parents=True)
    (harvey_dir / "harvey-lab-benchmark.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": "zaxy.harvey-lab-benchmark.v1",
                "summary": {
                    "status": "complete",
                    "article_task_count": 10,
                    "zaxy_task_count": 10,
                },
                "result_provenance": {
                    "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
                    "normalized_result_paths": [
                        f"/tmp/harvey/.ingestion/runs/zaxy-task-{index}/normalized-result.json"
                        for index in range(10)
                    ],
                    "external_baseline_reports": [
                        {
                            "path": "/tmp/harvey/.ingestion/reports/comparison-zaxy.json",
                            "schema_version": "0.1",
                            "normalized_result_count": 10,
                            "framework_count": 1,
                        }
                    ],
                    "external_baseline_report_paths": [
                        "/tmp/harvey/.ingestion/reports/comparison-zaxy.json"
                    ],
                    "external_readiness_report_paths": [
                        "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-ready.json"
                    ],
                    "external_run_manifest_paths": [
                        "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-external-run.json"
                    ],
                    "external_status_report_paths": [
                        "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-status.json"
                    ],
                },
                "task_rows": {
                    f"task-{index}": {
                        "task_id": f"task-{index}",
                        "zaxy_score": 0.8,
                        "article_best_score": 0.7,
                        "regular_no_memory_score": 0.6,
                        "zaxy_memory_read_calls": 1,
                        "zaxy_memory_search_calls": 3,
                    }
                    for index in range(10)
                },
                "zaxy_results": [
                    {
                        "task_id": f"task-{index}",
                        "run_id": f"zaxy-task-{index}",
                        "framework": "zaxy",
                        "commit": "29748828133dff83ad2263af353fb035504f8f77",
                        "score": 0.8,
                        "memory_read_calls": 1,
                        "memory_search_calls": 3,
                        "answer_path": f"results/zaxy-task-{index}/output.docx",
                        "judge_path": f"results/zaxy-task-{index}/scores.json",
                        "run_metrics_path": f"results/zaxy-task-{index}/metrics.json",
                        "tool_log_path": f"results/zaxy-task-{index}/transcript.jsonl",
                        "results_run_dir": f"results/zaxy-task-{index}",
                    }
                    for index in range(10)
                ],
            }
        ),
        encoding="utf-8",
    )
    (harvey_dir / "harvey-lab-external-run.json").write_text(
        json.dumps(
            {
                "schema_version": "zaxy.harvey-lab-external-run.v1",
                "task_count": 10,
                "report_json_path": "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json",
            }
        ),
        encoding="utf-8",
    )
    (harvey_dir / "harvey-lab-ready.json").write_text(
        json.dumps(
            {
                "schema_version": "zaxy.harvey-lab-run-readiness.v1",
                "status": "not_ready",
                "blocking_reasons": ["results_already_complete"],
                "expected_task_count": 10,
                "ready_task_count": 10,
                "normalized_ready_count": 10,
                "run_ready_count": 10,
                "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            }
        ),
        encoding="utf-8",
    )
    (harvey_dir / "harvey-lab-status.json").write_text(
        json.dumps(
            {
                "schema_version": "zaxy.harvey-lab-run-status.v1",
                "status": "complete",
                "expected_task_count": 10,
                "ready_task_count": 10,
                "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            }
        ),
        encoding="utf-8",
    )


def _write_rc1_project_benchmark_artifacts(root: Path) -> None:
    state_dir = root / "reports/benchmarks/state-recovery-v1"
    state_dir.mkdir(parents=True)
    (state_dir / "state-recovery-workload.json").write_text(
        json.dumps({"version": "state-recovery-v0", "fingerprint": "state-fingerprint", "cases": [{}]}),
        encoding="utf-8",
    )
    (state_dir / "state-recovery-benchmark.md").write_text("# StateRecoveryBench\n", encoding="utf-8")
    (state_dir / "state-recovery-benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": "state-recovery-report-v1",
                "version": "state-recovery-v0",
                "status": "pass",
                "workload_fingerprint": "state-fingerprint",
                "production_baseline": "memory_fabric_checkout",
                "checks": {
                    "state_accuracy": {"status": "pass"},
                    "minimal_evidence_recall": {"status": "pass"},
                    "stale_rejection": {"status": "pass"},
                    "distractor_resistance": {"status": "pass"},
                    "abstention_accuracy": {"status": "pass"},
                    "citation_coverage": {"status": "pass"},
                },
            }
        ),
        encoding="utf-8",
    )

    coordination_dir = root / "reports/benchmarks/coordination-v1"
    coordination_dir.mkdir(parents=True)
    (coordination_dir / "coordination-workload.json").write_text(
        json.dumps({"version": "coordination-v1"}), encoding="utf-8"
    )
    (coordination_dir / "coordination-benchmark.md").write_text("# CoordinationBench\n", encoding="utf-8")
    (coordination_dir / "coordination-benchmark.json").write_text(
        json.dumps(
            {
                "version": "coordination-v1",
                "workload_fingerprint": "coordination-fingerprint",
                "metrics": {
                    "accepted_finding_precision": 1.0,
                    "accepted_finding_recall": 1.0,
                    "citation_coverage": 1.0,
                    "evidence_coverage": 1.0,
                    "stale_claim_rejection": 1.0,
                    "duplicate_consolidation": 1.0,
                    "non_authoritative_leakage": 1.0,
                    "parent_checkout_answerability": 1.0,
                    "purpose_feedback_coverage": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    purpose_dir = root / "reports/benchmarks/purpose-v1"
    purpose_dir.mkdir(parents=True)
    holdout_dir = purpose_dir / "holdouts/public-derived-purpose-v1"
    holdout_dir.mkdir(parents=True)
    (holdout_dir / "holdout-pack.json").write_text(
        json.dumps({"fingerprint": "holdout-fingerprint", "cases": [{}, {}, {}, {}, {}]}),
        encoding="utf-8",
    )
    (holdout_dir / "source-disclosures.json").write_text(
        json.dumps({"sources": []}),
        encoding="utf-8",
    )
    (purpose_dir / "purpose-benchmark.md").write_text("# PurposeBench\n", encoding="utf-8")
    (purpose_dir / "purpose-benchmark.json").write_text(
        json.dumps(
            {
                "version": "purpose-v1",
                "status": "passed",
                "lane_count": 10,
                "passed_lanes": 10,
                "holdout_reports": {
                    "public-derived-purpose-v1": {
                        "pack_fingerprint": "holdout-fingerprint",
                        "gate_status": "diagnostic",
                        "claim_status": "public_derived_holdout",
                        "metrics": {"case_count": 5},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_version_option_reports_project_version() -> None:
    """The installed CLI should expose the packaged Zaxy version."""
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"zaxy {package_version()}"


def test_benchmark_freeze_json_passes_with_required_rc1_artifacts(tmp_path: Path) -> None:
    """benchmark-freeze should expose the RC.1 release evidence contract."""
    _write_rc1_freeze_artifacts(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["benchmark-freeze", "--root", str(tmp_path), "--json"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["release"] == "2.0.0-rc.1"
    assert report["passed"] is True
    assert report["headline_500"]["claim_scope"] == "longmemeval_compatible_checkout"
    assert report["harvey_lab"]["claim_scope"] == "external_anchor"


def test_benchmark_freeze_fails_when_required_rc1_artifact_is_missing(tmp_path: Path) -> None:
    """benchmark-freeze should fail closed when frozen evidence is incomplete."""
    _write_rc1_freeze_artifacts(tmp_path)
    (
        tmp_path
        / "reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json"
    ).unlink()
    runner = CliRunner()

    result = runner.invoke(app, ["benchmark-freeze", "--root", str(tmp_path), "--json"])

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["passed"] is False
    assert any(
        check["name"] == "headline_report" and check["passed"] is False
        for check in report["checks"]
    )


def test_memory_status_prints_eventloom_sessions(tmp_path: Path) -> None:
    """memory status should summarize Eventloom sessions without Neo4j."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    first = log.append("goal.created", actor="user", payload={"title": "Ship it"}, thread="agent-1")
    second = log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use source-aware assembly."},
        thread="agent-1",
    )
    assert first.seq == 1

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["memory", "status", "--eventloom-path", str(tmp_path / ".eventloom")],
    )

    assert result.exit_code == 0
    assert "Eventloom: " in result.output
    assert "Sessions: 1" in result.output
    assert "Total events: 2" in result.output
    assert "agent-1" in result.output
    assert "events=2" in result.output
    assert "latest=2" in result.output
    assert second.hash[:12] in result.output
    assert "integrity=OK" in result.output


def test_memory_status_json_output(tmp_path: Path) -> None:
    """memory status --json should expose stable machine-readable fields."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "transcript.turn",
        actor="assistant",
        payload={"content": "Recorded source recall."},
        thread="agent",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["memory", "status", "--eventloom-path", str(tmp_path / ".eventloom"), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["eventloom_path"] == str((tmp_path / ".eventloom").resolve())
    assert payload["session_count"] == 1
    assert payload["total_events"] == 1
    assert payload["sessions"][0]["session_id"] == "agent"
    assert payload["sessions"][0]["latest_seq"] == event.seq
    assert payload["sessions"][0]["latest_hash"] == event.hash


def test_memory_purpose_commands_report_status_lanes_and_feedback(tmp_path: Path) -> None:
    """memory purpose should expose replay-only purpose diagnostics without graph services."""
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "default.jsonl")
    log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        thread="default",
        payload={
            "purpose": {"profile": "legal", "evidence_policy": "exact_quote_and_citation_required"},
            "retention": {
                "purpose_policy": {
                    "suppressed_count": 1,
                    "suppressed_reasons": {"unsupported_legal_claim": 1},
                }
            },
            "diagnostics": {
                "evidence_policy": {
                    "status": "missing",
                    "missing": ["exact_quote"],
                    "suggested_queries": ["refresh exact contract clause"],
                }
            },
            "quality": {
                "required_action": {
                    "type": "memory_checkout",
                    "query": "refresh exact contract clause",
                }
            },
        },
    )
    log.append(
        "memory.feedback",
        actor="assistant",
        thread="default",
        payload={"purpose": {"profile": "legal"}, "citation": "event:default:1", "feedback": "rejected"},
    )

    runner = CliRunner()
    status = runner.invoke(app, ["memory", "purpose", "status", "--eventloom-path", str(eventloom)])
    lanes = runner.invoke(app, ["memory", "purpose", "lanes", "--eventloom-path", str(eventloom), "--json"])
    feedback = runner.invoke(
        app,
        [
            "memory",
            "purpose",
            "feedback",
            "--eventloom-path",
            str(eventloom),
            "--profile",
            "legal",
            "--outcome",
            "negative",
            "--json",
        ],
    )

    assert status.exit_code == 0
    assert "active profile: legal" in status.output
    assert "suppressed rows: 1" in status.output
    assert lanes.exit_code == 0
    lanes_payload = json.loads(lanes.output)
    assert lanes_payload["lanes"][0]["profile"] == "legal"
    assert lanes_payload["lanes"][0]["evidence_policy_fail_count"] == 1
    assert feedback.exit_code == 0
    feedback_payload = json.loads(feedback.output)
    assert feedback_payload["targets"][0]["target"] == "citation:event:default:1"
    assert feedback_payload["targets"][0]["negative_count"] == 1


def test_trace_export_json_correlates_eventloom_sessions(tmp_path: Path) -> None:
    """trace export should expose neutral spans and edges from replayed Eventloom logs."""
    eventloom_path = tmp_path / ".eventloom"
    parent = EventLog(eventloom_path / "auth-main.jsonl")
    worker = EventLog(eventloom_path / "auth-api.jsonl")
    mission = parent.append(
        "coordination.mission.created",
        actor="lead",
        payload={"mission_id": "auth-main", "objective": "Ship auth refactor"},
        thread="auth-main",
    )
    model_call = parent.append(
        "model.call.requested",
        actor="openai-compatible",
        payload={"provider": "openai-compatible", "model": "gpt-test", "mission_id": "auth-main"},
        thread="auth-main",
    )
    finding = worker.append(
        "coordination.finding.reported",
        actor="auth-api-agent",
        payload={"mission_id": "auth-main", "worker_id": "auth-api", "finding_id": "finding-auth-api-1"},
        thread="auth-api",
    )
    parent.append(
        "coordination.finding.promoted",
        actor="lead",
        payload={"mission_id": "auth-main", "finding_id": "finding-auth-api-1"},
        thread="auth-main",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["trace", "export", "--eventloom-path", str(eventloom_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["format"] == "zaxy.trace.v0.8"
    assert payload["summary"]["span_count"] == 4
    assert payload["summary"]["model_call_count"] == 1
    assert any(span["event_hash"] == mission.hash for span in payload["spans"])
    edges = {(edge["source"], edge["target"], edge["relation"]) for edge in payload["edges"]}
    assert (
        f"event:{mission.thread}:{mission.seq}",
        f"event:{model_call.thread}:{model_call.seq}",
        "contains",
    ) in edges
    assert (
        "event:auth-main:3",
        f"event:{finding.thread}:{finding.seq}",
        "promotes",
    ) in edges
    assert payload["sessions"][0]["integrity_ok"] is True


def test_trace_export_jsonl_writes_ingestion_records(tmp_path: Path) -> None:
    """trace export --format jsonl should emit one ingestion record per trace object."""
    eventloom_path = tmp_path / ".eventloom"
    parent = EventLog(eventloom_path / "auth-main.jsonl")
    mission = parent.append(
        "coordination.mission.created",
        actor="lead",
        payload={"mission_id": "auth-main", "objective": "Ship auth refactor"},
        thread="auth-main",
    )
    model_call = parent.append(
        "model.call.requested",
        actor="openai-compatible",
        payload={"provider": "openai-compatible", "model": "gpt-test", "mission_id": "auth-main"},
        thread="auth-main",
    )
    output = tmp_path / "trace.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "trace",
            "export",
            "--eventloom-path",
            str(eventloom_path),
            "--format",
            "jsonl",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Wrote trace export: {output}" in result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["record_type"] for record in records] == [
        "summary",
        "session",
        "span",
        "span",
        "edge",
    ]
    assert records[0]["format"] == "zaxy.trace.v0.8.jsonl"
    assert records[1]["session_id"] == "auth-main"
    assert records[2]["event_hash"] == mission.hash
    assert records[3]["event_hash"] == model_call.hash
    assert records[4] == {
        "record_type": "edge",
        "source": "event:auth-main:1",
        "target": "event:auth-main:2",
        "relation": "contains",
    }


def test_replay_cli_supports_inclusive_sequence_window(tmp_path: Path) -> None:
    """replay --to-seq should avoid dumping the tail of long Eventloom logs."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append("goal.created", actor="user", payload={"title": "Goal"}, thread="agent")
    middle = log.append("task.proposed", actor="assistant", payload={"summary": "Middle"}, thread="agent")
    log.append("task.completed", actor="assistant", payload={"summary": "Done"}, thread="agent")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["replay", str(log.path), "--from-seq", "2", "--to-seq", "2", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["from_seq"] == 2
    assert payload["to_seq"] == 2
    assert payload["integrity"]["total_events"] == 3
    assert [event["type"] for event in payload["events"]] == ["task.proposed"]
    assert payload["events"][0]["hash"] == middle.hash


def test_replay_cli_rejects_invalid_sequence_window(tmp_path: Path) -> None:
    """replay should reject inverted windows before presenting partial output."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append("goal.created", actor="user", payload={"title": "Goal"}, thread="agent")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["replay", str(log.path), "--from-seq", "3", "--to-seq", "2", "--json"],
    )

    assert result.exit_code == 2
    assert "from_seq must be <= to_seq" in result.output


def test_status_command_reports_embedded_projection_without_neo4j(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Top-level status should support the no-sidecar embedded projection path."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "status",
            "--projection-backend",
            "embedded",
            "--embedded-graph-path",
            str(tmp_path / ".eventloom" / "projections" / "embedded.kuzu"),
        ],
    )

    assert result.exit_code == 0
    assert "embedded graph: OK" in result.output
    assert "will be created lazily" in result.output
    assert "Neo4j:" not in result.output


@patch("zaxy.cli.runtime.LocalPgGraphRuntime")
def test_status_command_can_check_pggraph_projection_backend(
    mock_runtime_cls: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Top-level status should support pgGraph runtime posture checks."""
    monkeypatch.chdir(tmp_path)
    runtime = MagicMock()
    runtime.check.return_value.status = "warning"
    runtime.check.return_value.message = "pgGraph is not reachable; Docker is unavailable"
    mock_runtime_cls.return_value = runtime
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "status",
            "--projection-backend",
            "pggraph",
            "--pggraph-dsn",
            "postgresql://postgres:postgres@localhost:5432/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "pgGraph: WARNING (pgGraph is not reachable; Docker is unavailable)" in result.output
    mock_runtime_cls.assert_called_once_with(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        enabled=False,
    )
    assert "Neo4j:" not in result.output


def test_status_command_uses_repo_local_profile_for_bare_init(monkeypatch, tmp_path: Path) -> None:
    """After bare init, status should read .env.local and avoid probing Neo4j."""
    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu\n"
        "NEO4J_AUTO_START=false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "embedded graph: OK" in result.output
    assert "will be created lazily" in result.output
    assert "Neo4j:" not in result.output


def test_status_command_reports_memory_activation_remediation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Top-level status should surface stale checkout posture with a runnable fix."""
    monkeypatch.chdir(tmp_path)
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    checkout = log.append(
        "memory.checkout.completed",
        actor="assistant",
        payload={
            "token_efficiency": {
                "prompt_tokens": 180,
                "current_fact_count": 2,
                "evidence_count": 2,
                "facts_per_1k_prompt_tokens": 11.1,
            }
        },
        thread="agent-1",
        timestamp=now - timedelta(hours=3),
    )
    capture = log.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "codex", "role": "assistant"},
        thread="agent-1",
        timestamp=now - timedelta(minutes=10),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "status",
            "--projection-backend",
            "embedded",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--max-checkout-stale-minutes",
            "60",
            "--now",
            now.isoformat(),
        ],
    )

    assert result.exit_code == 0
    assert "embedded graph: OK" in result.output
    assert "Memory activation: WARNING (Latest memory checkout is stale)" in result.output
    assert f"latest checkout: seq={checkout.seq} session=agent-1" in result.output
    assert f"latest capture: transcript.turn seq={capture.seq} session=agent-1 source=codex" in result.output
    assert "checkout tokens: 180 prompt, 11.1 facts/1k prompt tokens" in result.output
    assert "Memory next steps:" in result.output
    assert "zaxy memory checkout" in result.output
    assert "--eventloom-path" in result.output


def test_coordinate_cli_brief_reports_mission_worker_and_findings(tmp_path: Path) -> None:
    """coordinate commands should expose a parent mission plus isolated worker findings."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"

    result = runner.invoke(
        app,
        [
            "coordinate",
            "start",
            "Ship auth refactor",
            "--mission",
            "auth-main",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    assert result.exit_code == 0
    assert "Mission auth-main started" in result.output

    result = runner.invoke(
        app,
        [
            "coordinate",
            "worker",
            "create",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    assert result.exit_code == 0
    assert "Worker auth-api registered" in result.output

    result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "API failures trace to expired JWKS cache handling.",
            "--evidence",
            "pytest tests/test_auth.py -q",
            "--claim-key",
            "auth.failure.cause",
            "--claim-value",
            "expired-jwks-cache",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    assert result.exit_code == 0
    assert "Finding " in result.output
    finding_id = result.output.strip().split()[1]

    result = runner.invoke(
        app,
        [
            "coordinate",
            "decide",
            "--mission",
            "auth-main",
            "--finding",
            finding_id,
            "--status",
            "accepted",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "coordinate",
            "promote",
            "--mission",
            "auth-main",
            "--finding",
            finding_id,
            "--eventloom-path",
            str(eventloom),
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "coordinate",
            "brief",
            "--mission",
            "auth-main",
            "--eventloom-path",
            str(eventloom),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mission_id"] == "auth-main"
    assert payload["workers"][0]["worker_id"] == "auth-api"
    assert payload["accepted_findings"][0]["summary"] == "API failures trace to expired JWKS cache handling."
    assert payload["accepted_findings"][0]["evidence"][0]["reference"] == "pytest tests/test_auth.py -q"


def test_coordinate_template_list_and_show_common_workflows() -> None:
    """Coordinate should expose built-in mission templates for common workflows."""
    runner = CliRunner()

    result = runner.invoke(app, ["coordinate", "template", "list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [template["name"] for template in payload["templates"]] == [
        "software-delivery",
        "research-review",
        "benchmark-investigation",
        "release-validation",
    ]

    result = runner.invoke(app, ["coordinate", "template", "show", "software-delivery", "--json"])

    assert result.exit_code == 0
    template = json.loads(result.output)
    assert template["name"] == "software-delivery"
    assert "Ship a production change" in template["objective"]
    assert [worker["worker_id"] for worker in template["workers"]] == [
        "implementation",
        "verification",
        "review",
    ]
    assert all(worker["assignment"] for worker in template["workers"])


def test_coordinate_template_apply_creates_mission_workers_and_assignments(tmp_path: Path) -> None:
    """Applying a template should create a replayable mission with assigned workers."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"

    result = runner.invoke(
        app,
        [
            "coordinate",
            "template",
            "apply",
            "release-validation",
            "--mission",
            "release-1",
            "--eventloom-path",
            str(eventloom),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["template"] == "release-validation"
    assert payload["mission_id"] == "release-1"
    assert payload["worker_count"] == 4
    assert payload["assignment_count"] == 4
    assert [worker["worker_id"] for worker in payload["workers"]] == [
        "release-gates",
        "docs-packaging",
        "runtime-smoke",
        "risk-audit",
    ]
    assert all(event["event_hash"] for event in payload["events"])

    result = runner.invoke(
        app,
        [
            "coordinate",
            "brief",
            "--mission",
            "release-1",
            "--eventloom-path",
            str(eventloom),
            "--json",
        ],
    )

    assert result.exit_code == 0
    brief = json.loads(result.output)
    assert brief["objective"].startswith("Validate a release candidate")
    workers = {worker["worker_id"]: worker for worker in brief["workers"]}
    assert workers["release-gates"]["assignment"].startswith("Run and record")
    assert workers["docs-packaging"]["assignment"].startswith("Validate")
    assert workers["runtime-smoke"]["assignment"].startswith("Exercise")
    assert workers["risk-audit"]["assignment"].startswith("Review")


def test_coordinate_report_cli_can_attach_git_and_test_metadata(tmp_path: Path) -> None:
    """coordinate report should optionally attach branch/worktree and test-result evidence."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Zaxy Test")
    (repo / "auth.py").write_text("TOKEN_TTL = 10\n", encoding="utf-8")
    _git(repo, "add", "auth.py")
    _git(repo, "commit", "-m", "initial")
    (repo / "auth.py").write_text("TOKEN_TTL = 20\n", encoding="utf-8")

    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-api", "--eventloom-path", str(eventloom)])

    result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "API failures trace to expired JWKS cache handling.",
            "--git-metadata",
            str(repo),
            "--test-result-json",
            json.dumps({
                "command": "pytest tests/test_auth.py -q",
                "status": "passed",
                "summary": "auth tests passed",
                "exit_code": 0,
            }),
            "--eventloom-path",
            str(eventloom),
        ],
    )

    assert result.exit_code == 0
    brief = CoordinationManager(eventloom_path=eventloom).brief("auth-main")
    evidence = brief.pending_findings[0].evidence
    git_evidence = next(item for item in evidence if item["kind"] == "git")
    test_evidence = next(item for item in evidence if item["kind"] == "test_result")
    assert git_evidence["worktree"] == str(repo.resolve())
    assert git_evidence["dirty"] is True
    assert {"path": "auth.py", "status": "M", "operation": "modified"} in git_evidence["changed_files"]
    assert test_evidence == {
        "kind": "test_result",
        "reference": "pytest tests/test_auth.py -q",
        "command": "pytest tests/test_auth.py -q",
        "status": "passed",
        "summary": "auth tests passed",
        "exit_code": 0,
    }


def test_coordinate_report_cli_rejects_malformed_test_result_json(tmp_path: Path) -> None:
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-api", "--eventloom-path", str(eventloom)])

    result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "Finding with bad test metadata.",
            "--test-result-json",
            "{not-json",
            "--eventloom-path",
            str(eventloom),
        ],
    )

    assert result.exit_code != 0
    assert "test result JSON" in result.output
    assert CoordinationManager(eventloom_path=eventloom).brief("auth-main").pending_findings == []


def test_coordinate_cli_checkout_returns_accepted_state_with_optional_diagnostics(tmp_path: Path) -> None:
    """coordinate checkout should keep worker scratch state out of default prompt context."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-api", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-ui", "--eventloom-path", str(eventloom)])
    accepted_result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "API failures trace to expired JWKS cache handling.",
            "--claim-key",
            "auth.failure.cause",
            "--claim-value",
            "expired-jwks-cache",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    pending_result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-ui",
            "--summary",
            "UI refresh handling is missing retry state.",
            "--claim-key",
            "auth.failure.cause",
            "--claim-value",
            "missing-browser-refresh",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    assert accepted_result.exit_code == 0
    assert pending_result.exit_code == 0
    accepted_id = accepted_result.output.strip().split()[1]
    pending_id = pending_result.output.strip().split()[1]
    runner.invoke(app, ["coordinate", "decide", "--mission", "auth-main", "--finding", accepted_id, "--status", "accepted", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "promote", "--mission", "auth-main", "--finding", accepted_id, "--eventloom-path", str(eventloom)])

    result = runner.invoke(
        app,
        ["coordinate", "checkout", "--mission", "auth-main", "--eventloom-path", str(eventloom), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [finding["finding_id"] for finding in payload["accepted_findings"]] == [accepted_id]
    assert payload["pending_findings"] == []
    assert payload["conflicts"] == []
    assert payload["excluded_pending_count"] == 1
    assert pending_id not in payload["prompt"]

    result = runner.invoke(
        app,
        [
            "coordinate",
            "checkout",
            "--mission",
            "auth-main",
            "--eventloom-path",
            str(eventloom),
            "--include-diagnostics",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [finding["finding_id"] for finding in payload["pending_findings"]] == [pending_id]
    assert payload["conflicts"][0]["claim_key"] == "auth.failure.cause"


def test_coordinate_cli_detect_conflicts_records_source_state_events(tmp_path: Path) -> None:
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    manager = CoordinationManager(eventloom_path=eventloom)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API worker saw one auth config snapshot.",
        actor="auth-api-agent",
        evidence=[{"kind": "file", "reference": "src/auth/config.py", "source_sha256": "a" * 64}],
    )
    manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="UI worker saw another auth config snapshot.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/auth/config.py", "source_sha256": "b" * 64}],
    )

    result = runner.invoke(
        app,
        [
            "coordinate",
            "detect-conflicts",
            "--mission",
            "auth-main",
            "--eventloom-path",
            str(eventloom),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["recorded_count"] == 1
    assert payload["events"][0]["conflict_type"] == "source_state"
    assert payload["events"][0]["source_reference"] == "src/auth/config.py"


def test_coordinate_cli_brief_can_enable_local_semantic_conflicts(tmp_path: Path) -> None:
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-api", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-ui", "--eventloom-path", str(eventloom)])
    runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "Token refresh retry is enabled in auth middleware.",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-ui",
            "--summary",
            "Token refresh retry is disabled in browser session handling.",
            "--eventloom-path",
            str(eventloom),
        ],
    )

    default_result = runner.invoke(
        app,
        ["coordinate", "brief", "--mission", "auth-main", "--eventloom-path", str(eventloom), "--json"],
    )
    semantic_result = runner.invoke(
        app,
        [
            "coordinate",
            "brief",
            "--mission",
            "auth-main",
            "--eventloom-path",
            str(eventloom),
            "--semantic-conflicts",
            "lexical",
            "--json",
        ],
    )

    assert default_result.exit_code == 0
    assert json.loads(default_result.output)["conflicts"] == []
    assert semantic_result.exit_code == 0
    payload = json.loads(semantic_result.output)
    assert payload["conflicts"][0]["conflict_type"] == "semantic"
    assert payload["conflicts"][0]["reason"] == "local_lexical_contradiction:disabled/enabled"


def test_coordinate_cli_ledger_reports_worker_quality_metrics(tmp_path: Path) -> None:
    """coordinate ledger should expose worker-level outcome metrics."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-api", "--eventloom-path", str(eventloom)])
    result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "API failures trace to expired JWKS cache handling.",
            "--evidence",
            "pytest tests/test_auth.py -q",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    finding_id = result.output.strip().split()[1]
    runner.invoke(app, ["coordinate", "decide", "--mission", "auth-main", "--finding", finding_id, "--status", "accepted", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "promote", "--mission", "auth-main", "--finding", finding_id, "--eventloom-path", str(eventloom)])

    result = runner.invoke(
        app,
        ["coordinate", "ledger", "--mission", "auth-main", "--eventloom-path", str(eventloom), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mission_id"] == "auth-main"
    assert payload["workers"][0]["worker_id"] == "auth-api"
    assert payload["workers"][0]["accepted_findings"] == 1
    assert payload["workers"][0]["test_backed_rate"] == 1.0


def test_coordinate_cli_handoff_records_replayable_final_event(tmp_path: Path) -> None:
    """coordinate handoff should create a parent mission handoff event."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])

    result = runner.invoke(
        app,
        [
            "coordinate",
            "handoff",
            "--mission",
            "auth-main",
            "--summary",
            "Auth mission complete.",
            "--next-step",
            "Release branch",
            "--risk",
            "Token cache metrics are sparse",
            "--eventloom-path",
            str(eventloom),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["event_type"] == "coordination.handoff.created"
    assert payload["mission_id"] == "auth-main"
    assert payload["handoff_id"].startswith("auth-main:handoff:")
    assert payload["summary"] == "Auth mission complete."
    assert payload["next_steps"] == ["Release branch"]
    assert payload["risks"] == ["Token cache metrics are sparse"]


def test_coordinate_cli_inspect_combines_mission_state_without_eventloom_jsonl(tmp_path: Path) -> None:
    """coordinate inspect should show the full replayed mission state in one operator view."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(
        app,
        [
            "coordinate",
            "start",
            "Ship auth refactor",
            "--mission",
            "auth-main",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    runner.invoke(
        app,
        [
            "coordinate",
            "worker",
            "create",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    runner.invoke(
        app,
        [
            "coordinate",
            "assign",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "trace API auth failures",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    accepted_result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "API failures trace to expired JWKS cache handling.",
            "--evidence",
            "pytest tests/test_auth.py -q",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    rejected_result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "Legacy token theory is unsupported.",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    accepted_id = accepted_result.output.strip().split()[1]
    rejected_id = rejected_result.output.strip().split()[1]
    runner.invoke(
        app,
        [
            "coordinate",
            "decide",
            "--mission",
            "auth-main",
            "--finding",
            accepted_id,
            "--status",
            "accepted",
            "--rationale",
            "Command-backed.",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    runner.invoke(
        app,
        [
            "coordinate",
            "promote",
            "--mission",
            "auth-main",
            "--finding",
            accepted_id,
            "--eventloom-path",
            str(eventloom),
        ],
    )
    runner.invoke(
        app,
        [
            "coordinate",
            "decide",
            "--mission",
            "auth-main",
            "--finding",
            rejected_id,
            "--status",
            "rejected",
            "--rationale",
            "No evidence.",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    runner.invoke(
        app,
        [
            "coordinate",
            "handoff",
            "--mission",
            "auth-main",
            "--summary",
            "Auth mission ready for release.",
            "--next-step",
            "Release branch",
            "--risk",
            "Metrics are sparse",
            "--eventloom-path",
            str(eventloom),
        ],
    )

    result = runner.invoke(
        app,
        ["coordinate", "inspect", "--mission", "auth-main", "--eventloom-path", str(eventloom), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mission_id"] == "auth-main"
    assert payload["brief"]["workers"][0]["assignment"] == "trace API auth failures"
    assert payload["worker_ledgers"][0]["accepted_findings"] == 1
    assert payload["worker_ledgers"][0]["rejected_findings"] == 1
    assert payload["findings"]["accepted"][0]["finding_id"] == accepted_id
    assert payload["findings"]["rejected"][0]["finding_id"] == rejected_id
    assert payload["evidence"][accepted_id][0]["reference"] == "pytest tests/test_auth.py -q"
    assert payload["decisions"][0]["finding_id"] == accepted_id
    assert payload["decisions"][0]["rationale"] == "Command-backed."
    assert payload["promoted_state"][0]["finding_id"] == accepted_id
    assert payload["handoffs"][0]["summary"] == "Auth mission ready for release."
    assert payload["handoffs"][0]["next_steps"] == ["Release branch"]

    text_result = runner.invoke(
        app,
        ["coordinate", "inspect", "--mission", "auth-main", "--eventloom-path", str(eventloom)],
    )

    assert text_result.exit_code == 0
    assert "Mission auth-main: Ship auth refactor" in text_result.output
    assert "Worker ledgers:" in text_result.output
    assert "Findings: accepted=1 pending=0 rejected=1 deferred=0 conflicted=0 stale=0" in text_result.output
    assert f"Promoted state: {accepted_id}" in text_result.output
    assert "Handoffs: 1" in text_result.output


def test_coordinate_cli_approval_packet_and_apply_decisions(tmp_path: Path) -> None:
    """coordinate approval commands should export and apply human review decisions."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-api", "--eventloom-path", str(eventloom)])
    result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "Expired JWKS cache causes API failures.",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    finding_id = result.output.strip().split()[1]

    packet_result = runner.invoke(
        app,
        ["coordinate", "approval-packet", "--mission", "auth-main", "--eventloom-path", str(eventloom), "--json"],
    )
    assert packet_result.exit_code == 0
    packet = json.loads(packet_result.output)
    assert packet["findings"][0]["finding_id"] == finding_id
    assert packet["findings"][0]["next_actions"][0] == {
        "code": "add_evidence",
        "label": "Attach evidence before accepting or promoting this finding.",
        "recommended_status": "deferred",
    }

    decisions = json.dumps([{"finding_id": finding_id, "status": "accepted", "rationale": "Reviewed remotely.", "promote": True}])
    apply_result = runner.invoke(
        app,
        [
            "coordinate",
            "apply-approval",
            "--mission",
            "auth-main",
            "--decisions-json",
            decisions,
            "--eventloom-path",
            str(eventloom),
            "--json",
        ],
    )

    assert apply_result.exit_code == 0
    payload = json.loads(apply_result.output)
    assert payload["reviewed_count"] == 1
    assert payload["promoted_count"] == 1


def test_coordinate_cli_review_export_outputs_markdown_and_json(tmp_path: Path) -> None:
    """coordinate review-export should provide a static human review artifact."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-api", "--eventloom-path", str(eventloom)])
    result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "Expired JWKS cache causes API failures.",
            "--evidence",
            "pytest tests/test_auth.py -q",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    finding_id = result.output.strip().split()[1]

    markdown_result = runner.invoke(
        app,
        ["coordinate", "review-export", "--mission", "auth-main", "--eventloom-path", str(eventloom)],
    )
    json_result = runner.invoke(
        app,
        ["coordinate", "review-export", "--mission", "auth-main", "--eventloom-path", str(eventloom), "--json"],
    )

    assert markdown_result.exit_code == 0
    assert "# Zaxy Coordinate Review: auth-main" in markdown_result.output
    assert f"## {finding_id}" in markdown_result.output
    assert "- Evidence: command `pytest tests/test_auth.py -q`" in markdown_result.output
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["read_only"] is True
    assert payload["packet"]["findings"][0]["finding_id"] == finding_id
    assert payload["markdown"] == markdown_result.output.rstrip()


def test_coordinate_cli_audit_report_outputs_eventloom_citations(tmp_path: Path) -> None:
    """coordinate audit-report should expose replayed Eventloom seq/hash citations."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runner.invoke(app, ["coordinate", "start", "Ship auth refactor", "--mission", "auth-main", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "worker", "create", "--mission", "auth-main", "--worker", "auth-api", "--eventloom-path", str(eventloom)])
    finding_result = runner.invoke(
        app,
        [
            "coordinate",
            "report",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
            "--summary",
            "Expired JWKS cache causes API failures.",
            "--evidence",
            "pytest tests/test_auth.py -q",
            "--eventloom-path",
            str(eventloom),
        ],
    )
    finding_id = finding_result.output.strip().split()[1]
    runner.invoke(app, ["coordinate", "decide", "--mission", "auth-main", "--finding", finding_id, "--status", "accepted", "--eventloom-path", str(eventloom)])
    runner.invoke(app, ["coordinate", "promote", "--mission", "auth-main", "--finding", finding_id, "--eventloom-path", str(eventloom)])

    result = runner.invoke(
        app,
        ["coordinate", "audit-report", "--mission", "auth-main", "--eventloom-path", str(eventloom), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mission_id"] == "auth-main"
    assert payload["read_only"] is True
    assert payload["summary"]["event_count"] == 5
    assert payload["events"][0]["event_type"] == "coordination.mission.created"
    assert payload["events"][0]["event_seq"] == 1
    assert len(payload["events"][0]["event_hash"]) == 64
    assert payload["events"][-1]["event_type"] == "coordination.finding.promoted"
    assert "## Eventloom Audit Trail" in payload["markdown"]
    assert f"finding={finding_id}" in payload["markdown"]

    text_result = runner.invoke(
        app,
        ["coordinate", "audit-report", "--mission", "auth-main", "--eventloom-path", str(eventloom)],
    )

    assert text_result.exit_code == 0
    assert "# Zaxy Coordinate Audit: auth-main" in text_result.output
    assert "seq=1" in text_result.output
    assert "hash=" in text_result.output


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_status_graph_json_reports_projection_health(
    mock_build_projection_store: MagicMock,
    tmp_path: Path,
) -> None:
    """memory status --graph should compare Eventloom and projection backend state."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="assistant",
        payload={"summary": "Projected graph chain."},
        thread="agent",
    )
    graph = AsyncMock()
    projection = MagicMock()
    projection.to_dict.return_value = {
        "session_id": "agent",
        "event_count": 1,
        "latest_seq": 1,
        "latest_hash": event.hash,
        "eventloom_latest_seq": 1,
        "eventloom_latest_hash": event.hash,
        "projection_lag": 0,
        "latest_hash_matches": True,
        "next_event_edges": 0,
        "previous_event_edges": 0,
        "missing_chain_links": 0,
        "integrity_ok": True,
    }
    graph.inspect_event_projection_status.return_value = projection
    mock_build_projection_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--graph",
            "--json",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["graph"]["backend"] == "neo4j"
    assert payload["graph"]["sessions"][0]["session_id"] == "agent"
    assert payload["graph"]["sessions"][0]["integrity_ok"] is True
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "neo4j"
    assert config.neo4j_uri == "bolt://test:7687"
    assert config.neo4j_user == "neo4j"
    assert config.neo4j_password == "testpassword"
    graph.connect.assert_awaited_once()
    graph.inspect_event_projection_status.assert_awaited_once_with(
        "agent",
        eventloom_latest_seq=event.seq,
        eventloom_latest_hash=event.hash,
    )
    graph.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_status_graph_can_use_pggraph_backend(
    mock_build_projection_store: MagicMock,
    tmp_path: Path,
) -> None:
    """Graph status should be backend-selectable for pgGraph operational checks."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="assistant",
        payload={"summary": "Projected graph chain."},
        thread="agent",
    )
    graph = AsyncMock()
    projection = MagicMock()
    projection.to_dict.return_value = {
        "session_id": "agent",
        "event_count": 1,
        "latest_seq": 1,
        "latest_hash": event.hash,
        "eventloom_latest_seq": 1,
        "eventloom_latest_hash": event.hash,
        "projection_lag": 0,
        "latest_hash_matches": True,
        "next_event_edges": 0,
        "previous_event_edges": 0,
        "missing_chain_links": 0,
        "integrity_ok": True,
    }
    graph.inspect_event_projection_status.return_value = projection
    mock_build_projection_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--graph",
            "--projection-backend",
            "pggraph",
            "--pggraph-dsn",
            "postgresql://postgres:postgres@localhost:5432/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "Graph projection (backend=pggraph):" in result.output
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "pggraph"
    assert config.pggraph_dsn == "postgresql://postgres:postgres@localhost:5432/zaxy"
    graph.inspect_event_projection_status.assert_awaited_once_with(
        "agent",
        eventloom_latest_seq=event.seq,
        eventloom_latest_hash=event.hash,
    )
    graph.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_status_graph_can_use_embedded_backend(
    mock_build_projection_store: MagicMock,
    tmp_path: Path,
) -> None:
    """Graph status should support the no-sidecar embedded projection backend."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="assistant",
        payload={"summary": "Projected embedded graph chain."},
        thread="agent",
    )
    embedded_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    graph = AsyncMock()
    projection = MagicMock()
    projection.to_dict.return_value = {
        "session_id": "agent",
        "event_count": 1,
        "latest_seq": 1,
        "latest_hash": event.hash,
        "eventloom_latest_seq": 1,
        "eventloom_latest_hash": event.hash,
        "projection_lag": 0,
        "latest_hash_matches": True,
        "next_event_edges": 0,
        "previous_event_edges": 0,
        "missing_chain_links": 0,
        "integrity_ok": True,
    }
    graph.inspect_event_projection_status.return_value = projection
    mock_build_projection_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--graph",
            "--projection-backend",
            "embedded",
            "--embedded-graph-path",
            str(embedded_path),
        ],
    )

    assert result.exit_code == 0
    assert "Graph projection (backend=embedded):" in result.output
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path == embedded_path
    assert config.neo4j_uri != "bolt://test:7687"
    graph.inspect_event_projection_status.assert_awaited_once_with(
        "agent",
        eventloom_latest_seq=event.seq,
        eventloom_latest_hash=event.hash,
    )
    graph.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_status_graph_uses_repo_local_profile_for_bare_init(
    mock_build_projection_store: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """After bare init, memory status --graph should use the embedded profile by default."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "hook.heartbeat",
        actor="zaxy-init",
        payload={"source": "zaxy-init"},
        thread="agent",
    )
    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu\n"
        "NEO4J_AUTO_START=false\n",
        encoding="utf-8",
    )
    graph = AsyncMock()
    projection = MagicMock()
    projection.to_dict.return_value = {
        "session_id": "agent",
        "event_count": 1,
        "latest_seq": event.seq,
        "latest_hash": event.hash,
        "eventloom_latest_seq": event.seq,
        "eventloom_latest_hash": event.hash,
        "projection_lag": 0,
        "latest_hash_matches": True,
        "next_event_edges": 0,
        "previous_event_edges": 0,
        "missing_chain_links": 0,
        "integrity_ok": True,
    }
    graph.inspect_event_projection_status.return_value = projection
    mock_build_projection_store.return_value = graph
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "status", "--eventloom-path", ".eventloom", "--graph"])

    assert result.exit_code == 0
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path == Path(".eventloom/projections/embedded.kuzu")
    assert "Graph projection (backend=embedded):" in result.output


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_status_graph_uses_profile_next_to_absolute_eventloom_path(
    mock_build_projection_store: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Absolute Eventloom paths should still resolve the inspected repo profile."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    event = EventLog(workspace / ".eventloom" / "agent.jsonl").append(
        "hook.heartbeat",
        actor="zaxy-init",
        payload={"source": "zaxy-init"},
        thread="agent",
    )
    embedded_path = workspace / ".eventloom" / "projections" / "embedded.kuzu"
    (workspace / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n",
        encoding="utf-8",
    )
    graph = AsyncMock()
    projection = MagicMock()
    projection.to_dict.return_value = {
        "session_id": "agent",
        "event_count": 1,
        "latest_seq": event.seq,
        "latest_hash": event.hash,
        "eventloom_latest_seq": event.seq,
        "eventloom_latest_hash": event.hash,
        "projection_lag": 0,
        "latest_hash_matches": True,
        "next_event_edges": 0,
        "previous_event_edges": 0,
        "missing_chain_links": 0,
        "integrity_ok": True,
    }
    graph.inspect_event_projection_status.return_value = projection
    mock_build_projection_store.return_value = graph
    monkeypatch.chdir(outside)
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "status", "--eventloom-path", str(workspace / ".eventloom"), "--graph"])

    assert result.exit_code == 0
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path == embedded_path


def test_memory_status_handles_empty_eventloom_directory(tmp_path: Path) -> None:
    """memory status should be useful before any memory has been written."""
    eventloom_dir = tmp_path / ".eventloom"
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "status", "--eventloom-path", str(eventloom_dir)])

    assert result.exit_code == 0
    assert "Sessions: 0" in result.output
    assert "Total events: 0" in result.output


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_inferred_status_json_reports_graph_audit(
    mock_build_projection_store: MagicMock,
) -> None:
    """memory inferred-status --json should expose inferred-edge audit metadata."""
    graph = AsyncMock()
    status = MagicMock()
    status.to_dict.return_value = {
        "session_id": "agent",
        "total_edges": 3,
        "method_count": 1,
        "evidence_count": 2,
        "missing_evidence_count": 1,
        "missing_source_event_count": 0,
        "evidence_coverage": 0.666667,
        "methods": [
            {
                "method": "task_completed_decision_citation_v1",
                "edge_count": 3,
                "relation_types": ["likely_implemented_decision"],
                "average_confidence": 0.86,
                "minimum_confidence": 0.86,
                "evidence_count": 2,
                "missing_evidence_count": 1,
                "missing_source_event_count": 0,
            }
        ],
        "samples": [],
    }
    graph.inspect_inferred_edge_status.return_value = status
    mock_build_projection_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "inferred-status",
            "--session-id",
            "agent",
            "--limit",
            "7",
            "--json",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "agent"
    assert payload["total_edges"] == 3
    assert payload["methods"][0]["method"] == "task_completed_decision_citation_v1"
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "neo4j"
    assert config.neo4j_uri == "bolt://test:7687"
    assert config.neo4j_user == "neo4j"
    assert config.neo4j_password == "testpassword"
    graph.connect.assert_awaited_once()
    graph.inspect_inferred_edge_status.assert_awaited_once_with("agent", limit=7)
    graph.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_inferred_status_can_use_pggraph_backend(
    mock_build_projection_store: MagicMock,
) -> None:
    """Inferred-edge audit status should be available through pgGraph."""
    graph = AsyncMock()
    status = MagicMock()
    status.to_dict.return_value = {
        "session_id": "agent",
        "total_edges": 0,
        "method_count": 0,
        "evidence_count": 0,
        "missing_evidence_count": 0,
        "missing_source_event_count": 0,
        "evidence_coverage": 1.0,
        "methods": [],
        "samples": [],
    }
    graph.inspect_inferred_edge_status.return_value = status
    mock_build_projection_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "inferred-status",
            "--session-id",
            "agent",
            "--projection-backend",
            "pggraph",
            "--pggraph-dsn",
            "postgresql://postgres:postgres@localhost:5432/zaxy",
            "--json",
        ],
    )

    assert result.exit_code == 0
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "pggraph"
    assert config.pggraph_dsn == "postgresql://postgres:postgres@localhost:5432/zaxy"
    graph.inspect_inferred_edge_status.assert_awaited_once_with("agent", limit=10)
    graph.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_inferred_status_uses_repo_local_profile_for_bare_init(
    mock_build_projection_store: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """After bare init, inferred-edge status should use the embedded profile by default."""
    embedded_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n"
        "NEO4J_AUTO_START=false\n",
        encoding="utf-8",
    )
    graph = AsyncMock()
    status = MagicMock()
    status.to_dict.return_value = {
        "session_id": "agent",
        "total_edges": 0,
        "method_count": 0,
        "evidence_count": 0,
        "missing_evidence_count": 0,
        "missing_source_event_count": 0,
        "evidence_coverage": 1.0,
        "methods": [],
        "samples": [],
    }
    graph.inspect_inferred_edge_status.return_value = status
    mock_build_projection_store.return_value = graph
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "inferred-status", "--session-id", "agent", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["backend"] == "embedded"
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path == embedded_path
    graph.inspect_inferred_edge_status.assert_awaited_once_with("agent", limit=10)
    graph.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_memory_inferred_status_text_reports_evidence_gaps(
    mock_build_projection_store: MagicMock,
) -> None:
    """The human inferred-edge status should call out evidence coverage and gaps."""
    graph = AsyncMock()
    status = MagicMock()
    status.to_dict.return_value = {
        "session_id": "agent",
        "total_edges": 2,
        "method_count": 1,
        "evidence_count": 1,
        "missing_evidence_count": 1,
        "missing_source_event_count": 0,
        "evidence_coverage": 0.5,
        "methods": [
            {
                "method": "task_completed_decision_citation_v1",
                "edge_count": 2,
                "relation_types": ["likely_implemented_decision"],
                "average_confidence": 0.86,
                "minimum_confidence": 0.86,
                "evidence_count": 1,
                "missing_evidence_count": 1,
                "missing_source_event_count": 0,
            }
        ],
        "samples": [
            {
                "source": "task-7",
                "target": "decision:Use graph audit",
                "relation_type": "likely_implemented_decision",
                "confidence": 0.86,
                "method": "task_completed_decision_citation_v1",
                "source_event_seq": 12,
                "source_event_hash": "a" * 64,
                "evidence_keys": ["evidence_source_event_seq", "evidence_reason"],
            }
        ],
    }
    graph.inspect_inferred_edge_status.return_value = status
    mock_build_projection_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "inferred-status",
            "--session-id",
            "agent",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    assert "Inferred edges: agent" in result.output
    assert "total=2" in result.output
    assert "evidence_coverage=50.0%" in result.output
    assert "task_completed_decision_citation_v1" in result.output
    assert "missing_evidence=1" in result.output
    assert "task-7 -[likely_implemented_decision]-> decision:Use graph audit" in result.output


def test_memory_capabilities_json_output(tmp_path: Path) -> None:
    """memory capabilities should expose a session-scoped model contract."""
    EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="codex",
        payload={"summary": "Capability manifest target."},
        thread="agent",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "capabilities",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--current-task",
            "make zaxy invisible",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "agent"
    assert payload["current_task"] == "make zaxy invisible"
    assert payload["recommended_next_call"]["tool"] == "memory_checkout"
    assert payload["status"]["eventloom"]["latest_seq"] == 1
    assert payload["status"]["mcp_tools"]["status"] == "runtime_unverified"
    assert "zaxy memory checkout" in payload["status"]["mcp_tools"]["fallback_command"]


def test_memory_capabilities_text_output(tmp_path: Path) -> None:
    """The text form should be prompt-ready for model session bootstrap."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "capabilities",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
        ],
    )

    assert result.exit_code == 0
    assert "# Zaxy Memory Contract" in result.output
    assert "Session: agent" in result.output
    assert "memory_checkout" in result.output
    assert "CLI fallback: zaxy memory checkout" in result.output
    assert "mcp_tools=runtime_unverified" in result.output


def test_memory_bootstrap_json_output(tmp_path: Path) -> None:
    """memory bootstrap should expose a model-facing session-start handoff."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "bootstrap",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--current-task",
            "ship the next sprint",
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "session_start"
    assert payload["session_id"] == "agent"
    assert payload["startup_sequence"][1]["tool"] == "memory_checkout"
    assert payload["startup_sequence"][1]["arguments"]["query"] == "ship the next sprint"
    assert payload["capture"]["configured"] is False
    assert payload["capabilities"]["status"]["mcp_tools"]["status"] == "runtime_unverified"


def test_memory_bootstrap_text_output(tmp_path: Path) -> None:
    """The text form should be compact enough to inject into model startup context."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "bootstrap",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--workspace-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "# Zaxy Session Bootstrap" in result.output
    assert "1. memory_capabilities" in result.output
    assert "2. memory_checkout" in result.output
    assert "If MCP memory tools are absent after resume" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent.jsonl").read_all()
    assert events[-1].type == "memory.bootstrap.shown"
    assert events[-1].payload["source"] == "cli"


def test_activate_codex_outputs_prompt_ready_startup_packet(tmp_path: Path) -> None:
    """zaxy activate codex should expose a low-friction session-start injection packet."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "activate",
            "codex",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--current-task",
            "continue the roadmap",
            "--workspace-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "# Zaxy Codex Activation" in result.output
    assert "continue the roadmap" in result.output
    assert "memory_checkout" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent.jsonl").read_all()
    assert events[-1].type == "memory.bootstrap.shown"
    assert events[-1].payload["source"] == "activate-codex"


def test_activate_codex_json_output(tmp_path: Path) -> None:
    """The activation packet should be machine-readable for launchers."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "activate",
            "codex",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--current-task",
            "continue the roadmap",
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["client"] == "codex"
    assert payload["mode"] == "session_start_injection"
    assert payload["session_id"] == "agent"
    assert payload["bootstrap"]["startup_sequence"][1]["tool"] == "memory_checkout"
    assert "# Zaxy Session Bootstrap" in payload["injection_text"]


@patch("zaxy.cli.evaluation.subprocess.run")
@patch("zaxy.capture_manager.subprocess.Popen")
def test_activate_codex_launches_codex_with_injected_prompt(
    mock_popen: MagicMock,
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    """--launch should start capture and Codex with activation context as the initial prompt."""
    config = tmp_path / ".codex" / "zaxy-capture.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "codex_home": str(tmp_path / ".codex-home"),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent",
                "source": "codex-local",
                "workspace": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    process = MagicMock()
    process.pid = 321
    mock_popen.return_value = process
    mock_run.return_value.returncode = 0
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "activate",
            "codex",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--current-task",
            "continue the roadmap",
            "--workspace-root",
            str(tmp_path),
            "--launch",
            "--codex-executable",
            "codex-test",
        ],
    )

    assert result.exit_code == 0
    command = mock_run.call_args.args[0]
    assert command[:3] == ["codex-test", "--cd", str(tmp_path.resolve())]
    assert "# Zaxy Session Bootstrap" in command[-1]
    assert "memory_checkout" in command[-1]
    capture_command = mock_popen.call_args.args[0]
    assert capture_command[:3] == [sys.executable, "-m", "zaxy"]
    assert "codex-capture" in capture_command
    assert "--watch" in capture_command


def test_activate_codex_reports_capture_degraded_when_config_missing(tmp_path: Path) -> None:
    """Activation should make missing managed capture visible instead of looking fully healthy."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "activate",
            "codex",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["capture_start"]["status"] == "degraded"
    assert payload["capture_start"]["reason"] == "not_configured"
    assert "zaxy init" in payload["capture_start"]["action"]
    assert "Capture action: degraded" in payload["injection_text"]


def test_activate_codex_dry_run_prints_launch_command(tmp_path: Path) -> None:
    """Dry-run should expose the launcher command without starting Codex."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "activate",
            "codex",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--workspace-root",
            str(tmp_path),
            "--launch",
            "--dry-run",
            "--codex-executable",
            "codex-test",
        ],
    )

    assert result.exit_code == 0
    assert "codex-test --cd" in result.output
    assert "Zaxy Session Bootstrap" in result.output


@patch("zaxy.cli.runtime.MemoryFabric")
def test_memory_checkout_json_output(mock_fabric_cls: MagicMock, tmp_path: Path) -> None:
    """memory checkout --json should expose the Memory Checkout contract."""
    checkout = MagicMock()
    checkout.to_dict.return_value = {
        "session_id": "agent-1",
        "query": "current project direction",
        "prompt": "# Memory Checkout\nUse Memory Checkout.",
        "current_facts": [{"content": "Use Memory Checkout.", "citation": "eventloom://agent-1/events/1#abc"}],
        "evidence": [{"citation": "eventloom://agent-1/events/1#abc"}],
        "provenance": [{"event_seq": 1, "event_hash": "abc"}],
        "token_efficiency": {
            "prompt_tokens": 8,
            "current_fact_count": 1,
            "evidence_count": 1,
            "facts_per_1k_prompt_tokens": 125.0,
        },
        "warnings": [],
    }
    fabric = AsyncMock()
    fabric.checkout_memory.return_value = checkout
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "checkout",
            "current project direction",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--ref",
            "refs/heads/main",
            "--neo4j-uri",
            "bolt://localhost:7687",
            "--neo4j-user",
            "neo4j",
            "--neo4j-password",
            "testpassword",
            "--neo4j-ca-cert",
            "",
            "--neo4j-trust-all",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "agent-1"
    assert payload["current_facts"][0]["citation"] == "eventloom://agent-1/events/1#abc"
    mock_fabric_cls.assert_called_once_with(
        eventloom_path=str(tmp_path / ".eventloom"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="testpassword",
        neo4j_ca_cert="",
        neo4j_trust_all=True,
        projection_backend="neo4j",
        pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        embedded_graph_path=Path(".eventloom/projections/embedded.kuzu"),
        latticedb_path=Path(".eventloom/projections/memory.latticedb"),
    )
    fabric.connect.assert_awaited_once()
    fabric.checkout_memory.assert_awaited_once_with(
        "current project direction",
        session_id="agent-1",
        ref="refs/heads/main",
        replay_from_seq=1,
        limit=10,
        max_recent_events=20,
    )
    fabric.close.assert_awaited_once()
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[-1].type == "memory.checkout.completed"
    assert events[-1].payload["query"] == "current project direction"
    assert events[-1].payload["token_efficiency"] == {
        "prompt_tokens": 8,
        "current_fact_count": 1,
        "evidence_count": 1,
        "facts_per_1k_prompt_tokens": 125.0,
    }


def test_memory_causal_and_consolidation_help_commands_are_registered() -> None:
    """Nested memory causal and consolidation commands should expose command help."""
    runner = CliRunner()

    successors = runner.invoke(app, ["memory", "causal", "successors", "--help"])
    propose = runner.invoke(app, ["memory", "consolidation", "propose", "--help"])
    propose_from_log = runner.invoke(
        app,
        ["memory", "consolidation", "propose-from-log", "--help"],
    )
    status = runner.invoke(app, ["memory", "consolidation", "status", "--help"])
    successor_options, successor_arguments = _cli_command_params("memory", "causal", "successors")
    propose_options, _ = _cli_command_params("memory", "consolidation", "propose")
    propose_from_log_options, _ = _cli_command_params("memory", "consolidation", "propose-from-log")
    status_options, _ = _cli_command_params("memory", "consolidation", "status")

    assert successors.exit_code == 0
    assert "entity_name" in successor_arguments
    assert "--relation-type" in successor_options
    assert propose.exit_code == 0
    assert "--source-event" in propose_options
    assert "--candidate-type" in propose_options
    assert propose_from_log.exit_code == 0
    assert "--window-size" in propose_from_log_options
    assert "--purpose" in propose_from_log_options
    assert status.exit_code == 0
    assert "--session-id" in status_options


def test_memory_reasoning_help_commands_are_registered() -> None:
    """Nested memory reasoning commands should expose primitive help."""
    runner = CliRunner()

    explain = runner.invoke(app, ["memory", "reasoning", "explain-outcome", "--help"])
    belief = runner.invoke(app, ["memory", "reasoning", "propose-belief-update", "--help"])
    confidence = runner.invoke(app, ["memory", "reasoning", "claim-confidence", "--help"])
    procedures = runner.invoke(app, ["memory", "reasoning", "similar-procedures", "--help"])
    record_unknown = runner.invoke(app, ["memory", "reasoning", "record-unknown", "--help"])
    known_unknowns = runner.invoke(app, ["memory", "reasoning", "known-unknowns", "--help"])
    trajectory = runner.invoke(app, ["memory", "reasoning", "confidence-trajectory", "--help"])
    reverify = runner.invoke(app, ["memory", "reasoning", "reverify-needed", "--help"])
    plan = runner.invoke(app, ["memory", "reasoning", "plan-from-procedures", "--help"])
    explain_options, explain_arguments = _cli_command_params("memory", "reasoning", "explain-outcome")
    belief_options, _ = _cli_command_params("memory", "reasoning", "propose-belief-update")
    confidence_options, confidence_arguments = _cli_command_params("memory", "reasoning", "claim-confidence")
    procedures_options, procedures_arguments = _cli_command_params("memory", "reasoning", "similar-procedures")
    record_unknown_options, _ = _cli_command_params("memory", "reasoning", "record-unknown")
    known_unknowns_options, _ = _cli_command_params("memory", "reasoning", "known-unknowns")
    _, trajectory_arguments = _cli_command_params("memory", "reasoning", "confidence-trajectory")
    reverify_options, _ = _cli_command_params("memory", "reasoning", "reverify-needed")
    plan_options, plan_arguments = _cli_command_params("memory", "reasoning", "plan-from-procedures")

    assert explain.exit_code == 0
    assert "outcome" in explain_arguments
    assert "--phase" in explain_options
    assert belief.exit_code == 0
    assert "--source-event" in belief_options
    assert "--confidence" in belief_options
    assert confidence.exit_code == 0
    assert "claim" in confidence_arguments
    assert "--limit" in confidence_options
    assert procedures.exit_code == 0
    assert "query" in procedures_arguments
    assert "--limit" in procedures_options
    assert record_unknown.exit_code == 0
    assert "--source-event" in record_unknown_options
    assert "--claim-key" in record_unknown_options
    assert known_unknowns.exit_code == 0
    assert "--status" in known_unknowns_options
    assert trajectory.exit_code == 0
    assert "claim" in trajectory_arguments
    assert reverify.exit_code == 0
    assert "--min-confidence" in reverify_options
    assert plan.exit_code == 0
    assert "goal" in plan_arguments
    assert "--phase" in plan_options


@pytest.mark.parametrize(
    ("command", "arguments", "method_name", "expected_kwargs"),
    [
        (
            "explain-outcome",
            ["Test failed", "--depth", "3"],
            "explain_outcome",
            {"phase": "review", "session_id": "agent", "depth": 3},
        ),
        (
            "claim-confidence",
            ["Projection is stale", "--limit", "4"],
            "get_claim_confidence",
            {"phase": "review", "session_id": "agent", "limit": 4},
        ),
        (
            "similar-procedures",
            ["Fix stale projection", "--limit", "6"],
            "retrieve_similar_procedures",
            {"phase": "review", "session_id": "agent", "limit": 6},
        ),
        (
            "known-unknowns",
            ["--limit", "3"],
            "list_known_unknowns",
            {"session_id": "agent", "status": "open", "limit": 3},
        ),
        (
            "confidence-trajectory",
            ["Projection is stale", "--limit", "4"],
            "list_confidence_trajectory",
            {"session_id": "agent", "limit": 4},
        ),
        (
            "reverify-needed",
            ["--query", "projection", "--limit", "7", "--min-confidence", "0.8"],
            "list_reverification_needs",
            {"session_id": "agent", "limit": 7, "min_confidence": 0.8},
        ),
        (
            "plan-from-procedures",
            ["Fix stale projection", "--limit", "6"],
            "plan_from_procedures",
            {"phase": "review", "session_id": "agent", "limit": 6},
        ),
    ],
)
@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_reasoning_read_commands_json_delegate_to_fabric(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
    command: str,
    arguments: list[str],
    method_name: str,
    expected_kwargs: dict[str, object],
) -> None:
    """Reasoning read commands should call configured MemoryFabric methods."""
    expected = {"primitive": method_name, "session_id": "agent", "results": []}
    fabric = AsyncMock()
    getattr(fabric, method_name).return_value = expected
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "reasoning",
            command,
            *arguments,
            "--phase",
            "review",
            "--session-id",
            "agent",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == expected
    fabric.connect.assert_awaited_once()
    method = getattr(fabric, method_name)
    if command == "known-unknowns":
        method.assert_awaited_once_with(**expected_kwargs)
    elif command == "reverify-needed":
        method.assert_awaited_once_with(query="projection", **expected_kwargs)
    else:
        method.assert_awaited_once_with(arguments[0], **expected_kwargs)
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_reasoning_propose_belief_update_json_delegates_to_fabric(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Belief update CLI should delegate proposal creation without authority promotion."""
    source_hash = "d" * 64
    expected = {
        "primitive": "propose_belief_update",
        "session_id": "agent",
        "event_type": "belief.update.proposed",
        "authority_status": "non_authoritative",
    }
    fabric = AsyncMock()
    fabric.propose_belief_update.return_value = expected
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "reasoning",
            "propose-belief-update",
            "Projection is stale",
            "--rationale",
            "Cited outcome points to stale projection.",
            "--confidence",
            "0.74",
            "--source-event",
            f"9:{source_hash}",
            "--phase",
            "reflection",
            "--actor",
            "reviewer",
            "--session-id",
            "agent",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == expected
    fabric.connect.assert_awaited_once()
    fabric.propose_belief_update.assert_awaited_once_with(
        "Projection is stale",
        rationale="Cited outcome points to stale projection.",
        confidence=0.74,
        source_events=[{"seq": 9, "hash": source_hash}],
        phase="reflection",
        session_id="agent",
        actor="reviewer",
    )
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_reasoning_record_unknown_json_delegates_to_fabric(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Known-unknown CLI should append cited non-authoritative uncertainty state."""
    source_hash = "e" * 64
    expected = {
        "event_type": "metacognition.unknown.recorded",
        "payload": {"authority_status": "non_authoritative"},
    }
    fabric = AsyncMock()
    fabric.record_known_unknown.return_value = expected
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "reasoning",
            "record-unknown",
            "Which backend caused latency?",
            "--reason",
            "Evidence conflicted.",
            "--claim-key",
            "backend-latency",
            "--gap-type",
            "conflicting_evidence",
            "--reverify-query",
            "latest backend latency cause",
            "--source-event",
            f"11:{source_hash}",
            "--phase",
            "review",
            "--actor",
            "reviewer",
            "--session-id",
            "agent",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == expected
    fabric.connect.assert_awaited_once()
    fabric.record_known_unknown.assert_awaited_once_with(
        "Which backend caused latency?",
        reason="Evidence conflicted.",
        source_events=[{"seq": 11, "hash": source_hash}],
        claim_key="backend-latency",
        gap_type="conflicting_evidence",
        reverify_query="latest backend latency cause",
        phase="review",
        session_id="agent",
        actor="reviewer",
    )
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_causal_successors_json_queries_fabric(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """memory causal successors --json should call the fabric causal read API."""
    causal_result = MagicMock()
    causal_result.to_dict.return_value = {
        "source": {"name": "Plan", "entity_type": "task"},
        "target": {"name": "Implementation", "entity_type": "task"},
        "relation_type": "enabled",
        "citation": "eventloom://agent/events/3#abc",
    }
    fabric = AsyncMock()
    fabric.query_causal_successors.return_value = [causal_result]
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "causal",
            "successors",
            "Plan",
            "--entity-type",
            "task",
            "--relation-type",
            "enabled",
            "--session-id",
            "agent",
            "--depth",
            "3",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "direction": "successors",
        "entity": {"name": "Plan", "entity_type": "task"},
        "results": [causal_result.to_dict.return_value],
    }
    fabric.connect.assert_awaited_once()
    fabric.query_causal_successors.assert_awaited_once_with(
        "Plan",
        relation_type="enabled",
        depth=3,
        session_id="agent",
    )
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_causal_successors_rejects_invalid_relation_before_fabric(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """--relation-type should reject labels outside the causal taxonomy at the CLI boundary."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "causal",
            "successors",
            "Plan",
            "--relation-type",
            "enables",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
        ],
    )

    assert result.exit_code != 0
    assert "causal relation_type must be one of:" in result.output
    mock_fabric_cls.assert_not_called()


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_consolidation_propose_appends_candidate_event(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """memory consolidation propose should append the cited candidate event."""
    from zaxy.consolidation import build_consolidation_candidate_event

    source_hash = "a" * 64
    fabric = AsyncMock()
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "consolidation",
            "propose",
            "--candidate-type",
            "claim",
            "--title",
            "Retry policy",
            "--summary",
            "Retries should preserve original citations.",
            "--source-event",
            f"7:{source_hash}",
            "--confidence",
            "0.82",
            "--method",
            "manual-review",
            "--purpose",
            "release audit",
            "--actor",
            "assistant",
            "--session-id",
            "agent",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    expected = build_consolidation_candidate_event(
        actor="assistant",
        session_id="agent",
        candidate_type="claim",
        title="Retry policy",
        summary="Retries should preserve original citations.",
        source_events=[{"seq": 7, "hash": source_hash}],
        confidence=0.82,
        method="manual-review",
        purpose="release audit",
    )
    payload = json.loads(result.output)
    assert payload["event_type"] == "consolidation.candidate.created"
    assert payload["payload"] == expected["payload"]
    fabric.connect.assert_awaited_once()
    fabric.append.assert_awaited_once_with(**expected)
    fabric.close.assert_awaited_once()


def test_memory_consolidation_propose_rejects_invalid_source_event(tmp_path: Path) -> None:
    """--source-event must be strict SEQ:HASH input."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "consolidation",
            "propose",
            "--candidate-type",
            "claim",
            "--title",
            "Retry policy",
            "--summary",
            "Retries should preserve original citations.",
            "--source-event",
            "not-a-citation",
            "--confidence",
            "0.82",
            "--method",
            "manual-review",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
        ],
    )

    assert result.exit_code != 0
    assert "source event must be formatted as SEQ:HASH" in result.output


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_consolidation_propose_from_log_json_delegates_to_fabric(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """memory consolidation propose-from-log should delegate segment proposal to MemoryFabric."""
    expected = {
        "session_id": "agent",
        "segment_count": 2,
        "candidate_count": 3,
        "events": [{"candidate_id": "consolidation:claim:" + ("a" * 24)}],
    }
    fabric = AsyncMock()
    fabric.propose_consolidation_candidates.return_value = expected
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "consolidation",
            "propose-from-log",
            "--session-id",
            "agent",
            "--actor",
            "review-bot",
            "--purpose",
            "release audit",
            "--window-size",
            "4",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == expected
    fabric.connect.assert_awaited_once()
    fabric.propose_consolidation_candidates.assert_awaited_once_with(
        session_id="agent",
        actor="review-bot",
        purpose="release audit",
        window_size=4,
    )
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_consolidation_propose_from_log_text_reports_segments(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Text output should make clear candidates came from reviewed log segments."""
    fabric = AsyncMock()
    fabric.propose_consolidation_candidates.return_value = {
        "session_id": "agent",
        "segment_count": 2,
        "candidate_count": 3,
    }
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "consolidation",
            "propose-from-log",
            "--session-id",
            "agent",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
        ],
    )

    assert result.exit_code == 0
    assert "Created 3 non-authoritative consolidation candidates from 2 log segments for agent." in result.output


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_consolidation_status_json_delegates_to_fabric(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """memory consolidation status should read review-gated status through MemoryFabric."""
    expected = {
        "session_id": "agent",
        "pending_count": 2,
        "accepted_count": 1,
        "rejected_count": 0,
    }
    fabric = AsyncMock()
    fabric.consolidation_status.return_value = expected
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "consolidation",
            "status",
            "--session-id",
            "agent",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == expected
    fabric.connect.assert_awaited_once()
    fabric.consolidation_status.assert_awaited_once_with(session_id="agent")
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime._memory_fabric")
def test_memory_consolidation_review_appends_review_event(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """memory consolidation review should append the review event contract."""
    from zaxy.consolidation import build_consolidation_review_event

    candidate_id = "consolidation:claim:" + ("b" * 24)
    fabric = AsyncMock()
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "consolidation",
            "review",
            "--candidate-id",
            candidate_id,
            "--status",
            "accepted",
            "--rationale",
            "Citations match the source events.",
            "--actor",
            "reviewer",
            "--session-id",
            "agent",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    expected = build_consolidation_review_event(
        actor="reviewer",
        session_id="agent",
        candidate_id=candidate_id,
        status="accepted",
        rationale="Citations match the source events.",
    )
    assert json.loads(result.output) == expected
    fabric.connect.assert_awaited_once()
    fabric.append.assert_awaited_once_with(**expected)
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime.MemoryFabric")
def test_memory_checkout_uses_repo_local_embedded_profile(
    mock_fabric_cls: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Checkout should use the repo-local embedded backend selected by bare init."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    embedded_path = eventloom_path / "projections" / "embedded.kuzu"
    (workspace / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n",
        encoding="utf-8",
    )
    checkout = MagicMock()
    checkout.to_dict.return_value = {"session_id": "agent-1", "query": "current project direction"}
    fabric = AsyncMock()
    fabric.checkout_memory.return_value = checkout
    mock_fabric_cls.return_value = fabric
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "checkout",
            "current project direction",
            "--eventloom-path",
            str(eventloom_path),
            "--session-id",
            "agent-1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    kwargs = mock_fabric_cls.call_args.kwargs
    assert kwargs["projection_backend"] == "embedded"
    assert kwargs["embedded_graph_path"] == embedded_path


@patch("os.getpid", return_value=4321)
@patch("zaxy.cli.runtime.MemoryFabric")
def test_memory_checkout_retries_locked_embedded_projection_with_isolated_path(
    mock_fabric_cls: MagicMock,
    _mock_getpid: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Checkout should not fail closed-loop memory use when the shared Kuzu projection is locked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    eventloom_path = workspace / ".eventloom"
    embedded_path = eventloom_path / "projections" / "embedded.kuzu"
    (workspace / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n",
        encoding="utf-8",
    )
    locked = AsyncMock()
    locked.connect.side_effect = RuntimeError(f"Could not set lock on file : {embedded_path}")
    checkout = MagicMock()
    checkout.to_dict.return_value = {
        "session_id": "agent-1",
        "query": "current project direction",
        "prompt": "# Memory Checkout\nUse cited memory.",
        "diagnostics": {},
    }
    fallback = AsyncMock()
    fallback.checkout_memory.return_value = checkout
    mock_fabric_cls.side_effect = [locked, fallback]
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "checkout",
            "current project direction",
            "--eventloom-path",
            str(eventloom_path),
            "--session-id",
            "agent-1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["diagnostics"]["projection_fallback"] == {
        "status": "used",
        "reason": "embedded_projection_locked",
        "original_path": str(embedded_path),
        "fallback_path": str(eventloom_path / "projections" / "checkout-agent-1-4321.kuzu"),
    }
    assert mock_fabric_cls.call_args_list[1].kwargs["embedded_graph_path"] == (
        eventloom_path / "projections" / "checkout-agent-1-4321.kuzu"
    )
    locked.close.assert_awaited_once()
    fallback.connect.assert_awaited_once()
    fallback.checkout_memory.assert_awaited_once()
    fallback.close.assert_awaited_once()


def test_packet_analyzer_cli_help_exposes_observe_only_gateway() -> None:
    """packet-analyzer should expose the low-latency observe-only gateway."""
    runner = CliRunner()

    result = runner.invoke(app, ["packet-analyzer", "--help"])
    command = get_command(app).commands["packet-analyzer"]
    option_names = {option for parameter in command.params for option in getattr(parameter, "opts", [])}

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "packet-analyzer" in result.output
    assert "--upstream-base-url" in option_names
    assert "--eventloom-path" in option_names
    assert "--session-id" in option_names


def test_packet_project_cli_projects_completed_packets(tmp_path: Path) -> None:
    """packet-project should run the cold-path packet projection worker."""
    eventloom_dir = tmp_path / ".eventloom"
    EventLog(eventloom_dir / "agent-1.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the codename is Atlas."}},
            "response": {"body": {"output_text": "I will remember Atlas."}},
        },
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["packet-project", "--eventloom-path", str(eventloom_dir), "--session-id", "agent-1"],
    )

    assert result.exit_code == 0
    assert "Projected 1 packet event" in result.output
    events = EventLog(eventloom_dir / "agent-1.jsonl").read_all()
    assert events[-1].type == "llm.packet.projected"
    assert "Atlas" in events[-1].payload["summary"]


@patch("zaxy.cli.runtime.GraphStore")
def test_packet_project_cli_can_project_new_packets_to_graph(
    mock_graph_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """packet-project --graph should upsert newly projected packets into Neo4j."""
    eventloom_dir = tmp_path / ".eventloom"
    EventLog(eventloom_dir / "agent-1.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the product owner is Nia."}},
            "response": {"body": {"output_text": "Product owner Nia recorded."}},
        },
    )
    mock_graph = AsyncMock()
    mock_graph_cls.return_value = mock_graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "packet-project",
            "--eventloom-path",
            str(eventloom_dir),
            "--session-id",
            "agent-1",
            "--graph",
            "--neo4j-uri",
            "bolt://localhost:7687",
            "--neo4j-user",
            "neo4j",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    assert "Projected 1 packet event" in result.output
    assert "graph_projected=1" in result.output
    assert "graph_failed=0" in result.output
    mock_graph.connect.assert_awaited_once()
    mock_graph.init_schema.assert_awaited_once()
    mock_graph.upsert_extraction.assert_awaited_once()
    mock_graph.close.assert_awaited_once()


def test_packet_project_cli_supports_bounded_watch_mode(tmp_path: Path) -> None:
    """packet-project watch mode should support bounded runs for supervisors/tests."""
    eventloom_dir = tmp_path / ".eventloom"
    EventLog(eventloom_dir / "agent-1.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the on-call is Dev."}},
            "response": {"body": {"output_text": "On-call Dev recorded."}},
        },
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "packet-project",
            "--eventloom-path",
            str(eventloom_dir),
            "--session-id",
            "agent-1",
            "--watch",
            "--watch-iterations",
            "2",
            "--interval-seconds",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert "Watched 2 projection pass" in result.output
    assert "projected=1" in result.output


def test_memory_log_prints_recent_events(tmp_path: Path) -> None:
    """memory log should print recent events in compact git-style form."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use memory log."},
        thread="agent",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["memory", "log", "--eventloom-path", str(tmp_path / ".eventloom")],
    )

    assert result.exit_code == 0
    assert f"agent [{event.seq}] {event.hash[:12]}" in result.output
    assert "decision.recorded by assistant" in result.output
    assert "Use memory log." in result.output


def test_memory_log_json_filters_session_and_limit(tmp_path: Path) -> None:
    """memory log --json should expose stable event entries with filtering."""
    agent_log = EventLog(tmp_path / ".eventloom" / "agent.jsonl")
    agent_log.append(
        "goal.created",
        actor="user",
        payload={"title": "Older"},
        thread="agent",
    )
    event = agent_log.append(
        "task.completed",
        actor="assistant",
        payload={"summary": "Newest"},
        thread="agent",
    )
    EventLog(tmp_path / ".eventloom" / "other.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Skip"},
        thread="other",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "memory",
            "log",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--limit",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["eventloom_path"] == str((tmp_path / ".eventloom").resolve())
    assert payload["limit"] == 1
    assert payload["session_id"] == "agent"
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["seq"] == event.seq
    assert payload["entries"][0]["hash"] == event.hash
    assert payload["entries"][0]["summary"] == "Newest"


def test_memory_diff_prints_event_range(tmp_path: Path) -> None:
    """memory diff should print added events in the requested sequence range."""
    log = EventLog(tmp_path / ".eventloom" / "agent.jsonl")
    log.append("goal.created", actor="user", payload={"title": "Older"}, thread="agent")
    event = log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Add diff."},
        thread="agent",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "memory",
            "diff",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--from-seq",
            "2",
            "--to-seq",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert f"agent +[{event.seq}] {event.hash[:12]} decision.recorded by assistant" in result.output
    assert "Add diff." in result.output


def test_memory_diff_json_output(tmp_path: Path) -> None:
    """memory diff --json should expose stable added event entries."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="codex",
        payload={"summary": "Added diff CLI."},
        thread="agent",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "memory",
            "diff",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--from-seq",
            "1",
            "--to-seq",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "agent"
    assert payload["from_seq"] == 1
    assert payload["to_seq"] == 1
    assert payload["integrity_ok"] is True
    assert payload["added"][0]["seq"] == event.seq
    assert payload["added"][0]["summary"] == "Added diff CLI."


def test_memory_ref_update_and_list(tmp_path: Path) -> None:
    """memory ref should create durable git-style refs and list latest pointers."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="codex",
        payload={"summary": "Ref target."},
        thread="agent",
    )
    runner = CliRunner()

    update = runner.invoke(
        app,
        [
            "memory",
            "ref",
            "refs/heads/main",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--target-seq",
            str(event.seq),
            "--target-hash",
            event.hash,
            "--type",
            "branch",
        ],
    )
    listed = runner.invoke(
        app,
        ["memory", "refs", "--eventloom-path", str(tmp_path / ".eventloom"), "--json"],
    )

    assert update.exit_code == 0
    assert "refs/heads/main -> agent@1" in update.output
    assert listed.exit_code == 0
    payload = json.loads(listed.output)
    assert payload["refs"][0]["name"] == "refs/heads/main"
    assert payload["refs"][0]["target_hash"] == event.hash


def test_ide_config_command_prints_copyable_mcp_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "claude-desktop",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert '"mcpServers"' in result.output
    assert '"zaxy"' in result.output
    assert '"command": "/opt/zaxy/bin/zaxy"' in result.output
    assert '"args": [' in result.output
    assert '"EVENTLOOM_THREAD": "zaxy-default"' in result.output
    assert '"ZAXY_DOMAIN": "zaxy"' in result.output
    assert '"ZAXY_ENV": "development"' in result.output
    assert '"PROJECTION_BACKEND": "embedded"' in result.output
    assert '"EMBEDDED_GRAPH_PATH": ".eventloom/projections/embedded.kuzu"' in result.output
    assert '"NEO4J_URI": "bolt://localhost:7687"' in result.output
    assert '"NEO4J_AUTO_START": "false"' in result.output
    assert '"NEO4J_CA_CERT": ""' in result.output
    assert '"NEO4J_PASSWORD_FILE": ""' in result.output
    assert '"PGGRAPH_AUTO_START": "false"' in result.output
    assert '"MCP_ADMIN_TOKEN_FILE": ""' in result.output
    assert '"MCP_REMOTE_AUTH_TOKEN_FILE": ""' in result.output
    assert '"OPENAI_API_KEY_FILE": ""' in result.output
    assert '"PATHLIGHT_ACCESS_TOKEN_FILE": ""' in result.output
    assert "testpassword" not in result.output.casefold()


def test_ide_config_command_installs_project_cursor_config(tmp_path: Path) -> None:
    """ide-config --install should merge into the verified project-local target."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "cursor",
            "--install",
            "--workspace",
            str(tmp_path),
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "Installed cursor MCP config" in result.output
    config = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["zaxy"]["command"] == "/opt/zaxy/bin/zaxy"


def test_ide_config_command_prints_codex_cli_install_command() -> None:
    """Codex install should keep workspace state out of global MCP config."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "codex",
            "--install",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "Run this Codex MCP install command:" in result.output
    assert "codex mcp add zaxy" in result.output
    assert "--env EVENTLOOM_THREAD" not in result.output
    assert "ZAXY_DOMAIN" not in result.output
    assert "NEO4J_URI" not in result.output
    assert "NEO4J_CA_CERT" not in result.output
    assert "NEO4J_PASSWORD_FILE" not in result.output
    assert "-- /opt/zaxy/bin/zaxy serve" in result.output
    assert "--eventloom-path" not in result.output


def test_ide_config_command_prints_codex_cli_command_without_install_flag() -> None:
    """Codex print mode should be useful and should not expose internal helper names."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "codex",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "codex mcp add zaxy" in result.output
    assert "render_codex_mcp_add_command" not in result.output


def test_ide_config_command_prints_hermes_yaml_config() -> None:
    """Hermes print mode should emit the config.yaml shape without repo-local state."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "hermes",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "mcp_servers:" in result.output
    assert "zaxy:" in result.output
    assert "command: /opt/zaxy/bin/zaxy" in result.output
    assert "- serve" in result.output
    assert "memory_checkout" in result.output
    assert "EVENTLOOM_PATH" not in result.output
    assert "EVENTLOOM_THREAD" not in result.output
    assert "ZAXY_DOMAIN" not in result.output


def test_ide_config_command_writes_hermes_config(tmp_path: Path) -> None:
    """Hermes install should merge into an explicit config.yaml path."""
    runner = CliRunner()
    target = tmp_path / "config.yaml"
    target.write_text("model:\n  default: anthropic/claude-opus-4.6\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ide-config",
            "hermes",
            "--install",
            "--hermes-config",
            str(target),
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert f"Wrote Hermes MCP config to {target}" in result.output
    config = target.read_text(encoding="utf-8")
    assert "mcp_servers:" in config
    assert "zaxy:" in config
    assert "command: /opt/zaxy/bin/zaxy" in config
    assert "EVENTLOOM_PATH" not in config


def test_ide_config_command_writes_trusted_project_codex_config(tmp_path: Path) -> None:
    """Codex direct config writes should require explicit project trust acknowledgement."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "codex",
            "--install",
            "--codex-config-scope",
            "project",
            "--codex-trusted-project",
            "--workspace",
            str(tmp_path),
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote Codex MCP config" in result.output
    config = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "[mcp_servers.zaxy]" in config
    assert 'command = "/opt/zaxy/bin/zaxy"' in config
    assert 'args = ["serve"]' in config
    assert "EVENTLOOM_PATH" not in config
    assert "NEO4J_URI" not in config
    assert "NEO4J_CA_CERT" not in config
    assert "NEO4J_PASSWORD_FILE" not in config


def test_ide_config_command_rejects_project_codex_config_without_trust(tmp_path: Path) -> None:
    """Project-scoped Codex writes should fail before touching disk without trust acknowledgement."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "codex",
            "--install",
            "--codex-config-scope",
            "project",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "trusted" in result.output
    assert "project" in result.output
    assert not (tmp_path / ".codex" / "config.toml").exists()


@patch("zaxy.mcp_server.main", new_callable=AsyncMock)
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_derives_workspace_defaults_when_not_overridden(
    mock_server_cls: MagicMock,
    mock_mcp_main: AsyncMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A bare `zaxy serve` should scope memory to the process workspace."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["serve"],
        catch_exceptions=False,
        obj=None,
        env={},
        color=False,
        prog_name="zaxy",
    )

    assert result.exit_code == 0
    mock_mcp_main.assert_awaited_once()
    mock_server_cls.assert_called_once()
    kwargs = mock_server_cls.call_args.kwargs
    assert kwargs["eventloom_path"] == str(Path.cwd() / ".eventloom")
    assert kwargs["workspace_root"] == Path.cwd()
    assert kwargs["default_session_id"] == f"{tmp_path.name}-default"


@patch("zaxy.mcp_server.main", new_callable=AsyncMock)
@patch("zaxy.mcp_runtime.EmbeddedMcpRuntimeCoordinator")
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_embedded_stdio_claims_runtime_owner(
    mock_server_cls: MagicMock,
    mock_coordinator_cls: MagicMock,
    mock_mcp_main: AsyncMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Embedded stdio serve should claim one repo-local runtime owner before Kuzu opens."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    owner = MagicMock()
    coordinator = MagicMock()
    coordinator.try_claim_owner.return_value = owner
    mock_coordinator_cls.from_eventloom_path.return_value = coordinator
    runner = CliRunner()

    result = runner.invoke(app, ["serve"], catch_exceptions=False, env={}, color=False, prog_name="zaxy")

    assert result.exit_code == 0
    mock_coordinator_cls.from_eventloom_path.assert_called_once_with(str(tmp_path / ".eventloom"))
    coordinator.try_claim_owner.assert_called_once_with()
    mock_server_cls.assert_called_once()
    mock_mcp_main.assert_awaited_once_with(owner_claim=owner)


@patch("zaxy.mcp_server.proxy_main", new_callable=AsyncMock)
@patch("zaxy.mcp_runtime.EmbeddedMcpRuntimeCoordinator")
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_embedded_stdio_proxies_when_owner_exists(
    mock_server_cls: MagicMock,
    mock_coordinator_cls: MagicMock,
    mock_proxy_main: AsyncMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Duplicate embedded stdio serve processes should proxy instead of opening Kuzu."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    coordinator = MagicMock()
    coordinator.try_claim_owner.return_value = None
    mock_coordinator_cls.from_eventloom_path.return_value = coordinator
    runner = CliRunner()

    result = runner.invoke(app, ["serve"], catch_exceptions=False, env={}, color=False, prog_name="zaxy")

    assert result.exit_code == 0
    mock_server_cls.assert_not_called()
    mock_proxy_main.assert_awaited_once_with(coordinator)


@patch("zaxy.mcp_server.main", new_callable=AsyncMock)
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_uses_repo_local_embedded_profile(
    mock_server_cls: MagicMock,
    mock_mcp_main: AsyncMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A bare `zaxy serve` should pass the repo-local projection profile to MCP."""
    embedded_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    runner = CliRunner()

    result = runner.invoke(app, ["serve"], catch_exceptions=False, env={}, color=False, prog_name="zaxy")

    assert result.exit_code == 0
    mock_mcp_main.assert_awaited_once()
    kwargs = mock_server_cls.call_args.kwargs
    assert kwargs["projection_backend"] == "embedded"
    assert kwargs["embedded_graph_path"] == embedded_path


@patch("zaxy.mcp_server.main", new_callable=AsyncMock)
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_eventloom_override_uses_matching_embedded_projection(
    mock_server_cls: MagicMock,
    mock_mcp_main: AsyncMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """An explicit Eventloom path should not share the workspace default Kuzu lock."""
    eventloom_path = tmp_path / "isolated.eventloom"
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["serve", "--eventloom-path", str(eventloom_path)],
        catch_exceptions=False,
        env={},
        color=False,
        prog_name="zaxy",
    )

    assert result.exit_code == 0
    mock_mcp_main.assert_awaited_once()
    kwargs = mock_server_cls.call_args.kwargs
    assert kwargs["eventloom_path"] == str(eventloom_path)
    assert kwargs["embedded_graph_path"] == eventloom_path / "projections" / "embedded.kuzu"


def test_integration_template_command_prints_framework_starter() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["integration-template", "langgraph", "--session-id", "zaxy-default"],
    )

    assert result.exit_code == 0
    assert "async def zaxy_langgraph_memory_node" in result.output
    assert "from zaxy import MemoryFabric" in result.output
    assert "session_id='zaxy-default'" in result.output
    assert "import langgraph" not in result.output.casefold()


def test_integration_template_command_can_print_install_hint() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["integration-template", "crewai", "--install-hint"],
    )

    assert result.exit_code == 0
    assert "python -m pip install 'zaxy-memory[crewai]'" in result.output
    assert "async def zaxy_crewai_memory_step" in result.output


def test_coordinate_adapter_template_command_prints_coordination_starter() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "coordinate",
            "adapter-template",
            "codex",
            "--mission",
            "auth-main",
            "--worker",
            "auth-api",
        ],
    )

    assert result.exit_code == 0
    assert "CoordinationAdapter" in result.output
    assert "mission_id='auth-main'" in result.output
    assert "worker_id='auth-api'" in result.output
    assert "adapter.report_finding" in result.output


def test_integrations_command_lists_framework_registry() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["integrations"])

    assert result.exit_code == 0
    assert "LangGraph" in result.output
    assert "zaxy-memory[langgraph]" in result.output
    assert "native-preview" in result.output
    assert "zaxy.adapters.langgraph" in result.output


def test_integrations_command_can_emit_json() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["integrations", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["framework"] == "langgraph"
    assert payload[0]["install"] == "python -m pip install 'zaxy-memory[langgraph]'"
    assert payload[0]["maturity"] == "native-beta"
    assert payload[0]["native_adapter"] == "zaxy.adapters.langgraph"


def test_integrations_command_can_emit_recommendation_json() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["integrations", "--recommendation", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["target"] == "common-native-preview-contract"
    assert payload["track"] == "model-facing-ux"
    assert payload["evidence_frameworks"] == ["langgraph", "crewai"]
    assert payload["hold_frameworks"] == ["autogen"]
    assert "AutoGen" in payload["rationale"]


def test_hooks_command_prints_claude_code_settings(tmp_path: Path) -> None:
    """hooks should render copyable observer hook config."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hooks",
            "claude-code",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--domain",
            "zaxy",
        ],
    )

    assert result.exit_code == 0
    assert '"hooks"' in result.output
    assert '"Stop"' in result.output
    assert '"PreCompact"' in result.output
    assert "zaxy hook-event stop" in result.output
    assert "zaxy hook-event precompact" in result.output
    assert "--session-id zaxy-default" in result.output


def test_hooks_command_writes_output_file(tmp_path: Path) -> None:
    """hooks --output should write config instead of printing it."""
    runner = CliRunner()
    output = tmp_path / ".claude" / "settings.local.json"

    result = runner.invoke(
        app,
        [
            "hooks",
            "claude-code",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote hook config" in result.output
    assert output.is_file()
    assert '"PreCompact"' in output.read_text(encoding="utf-8")
    assert '"hooks"' not in result.output


def test_hooks_command_merges_claude_local_settings(tmp_path: Path) -> None:
    """Claude hook install should preserve unrelated local settings and hooks."""
    runner = CliRunner()
    output = tmp_path / ".claude" / "settings.local.json"
    output.parent.mkdir()
    output.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(pytest)"]},
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Edit",
                            "hooks": [{"type": "command", "command": "ruff check ."}],
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "hooks",
            "claude-code",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    settings = json.loads(output.read_text(encoding="utf-8"))
    assert settings["permissions"]["allow"] == ["Bash(pytest)"]
    assert settings["hooks"]["PostToolUse"][0]["hooks"][0]["command"] == "ruff check ."
    assert "zaxy hook-event stop" in json.dumps(settings["hooks"]["Stop"])
    assert "zaxy hook-event precompact" in json.dumps(settings["hooks"]["PreCompact"])


def test_hooks_command_refuses_duplicate_claude_zaxy_hooks_without_force(tmp_path: Path) -> None:
    """Claude hook install should not duplicate existing Zaxy hook handlers."""
    runner = CliRunner()
    output = tmp_path / ".claude" / "settings.local.json"
    output.parent.mkdir()
    output.write_text(
        '{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "zaxy hook-event stop"}]}]}}\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["hooks", "claude-code", "--output", str(output)],
    )

    assert result.exit_code != 0
    assert "already contains Zaxy hook handlers" in result.output
    assert output.read_text(encoding="utf-8").count("zaxy hook-event") == 1


def test_hooks_command_force_replaces_claude_zaxy_hooks(tmp_path: Path) -> None:
    """Claude --force should replace Zaxy handlers while preserving unrelated settings."""
    runner = CliRunner()
    output = tmp_path / ".claude" / "settings.local.json"
    output.parent.mkdir()
    output.write_text(
        json.dumps(
            {
                "env": {"KEEP": "1"},
                "hooks": {
                    "Stop": [
                        {"hooks": [{"type": "command", "command": "zaxy hook-event stop --source old"}]},
                        {"hooks": [{"type": "command", "command": "echo keep"}]},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["hooks", "claude-code", "--domain", "zaxy", "--output", str(output), "--force"],
    )

    assert result.exit_code == 0
    settings = json.loads(output.read_text(encoding="utf-8"))
    assert settings["env"] == {"KEEP": "1"}
    serialized = json.dumps(settings)
    assert "--source old" not in serialized
    assert "echo keep" in serialized
    assert "zaxy hook-event stop" in serialized


def test_hook_status_ignores_non_hook_text_in_claude_settings(tmp_path: Path) -> None:
    """Hook detection should inspect command handlers, not arbitrary JSON text."""
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir()
    settings_path.write_text('{"notes": "zaxy hook-event is mentioned here"}\n', encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["hook-status", "--eventloom-path", str(tmp_path / ".eventloom"), "--workspace-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Claude Code hook config: missing" in result.output


def test_hooks_command_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    """hooks --output should be non-destructive by default."""
    runner = CliRunner()
    output = tmp_path / "hooks.sh"
    output.write_text("existing\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["hooks", "generic", "--output", str(output)],
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_hooks_command_generic_output_documents_observation_sinks() -> None:
    """Generic hook output should advertise every first-class observation sink."""
    runner = CliRunner()

    result = runner.invoke(app, ["hooks", "generic", "--domain", "zaxy"])

    assert result.exit_code == 0
    assert "zaxy hook-event command" in result.output
    assert "zaxy hook-event file-edit" in result.output
    assert "zaxy hook-event tool-call" in result.output
    assert "zaxy hook-event transcript-turn" in result.output


def test_hooks_command_force_overwrites_output_file(tmp_path: Path) -> None:
    """hooks --force should replace an existing output file."""
    runner = CliRunner()
    output = tmp_path / "hooks.sh"
    output.write_text("existing\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["hooks", "generic", "--domain", "zaxy", "--output", str(output), "--force"],
    )

    assert result.exit_code == 0
    assert "Wrote hook config" in result.output
    assert "zaxy hook-event session-start" in output.read_text(encoding="utf-8")


def test_hook_event_command_appends_eventloom_event(tmp_path: Path) -> None:
    """hook-event should append lightweight lifecycle observations without Neo4j."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "precompact",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded hook precompact" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert len(events) == 2
    assert events[0].type == "hook.precompact"
    assert events[0].actor == "zaxy-hook"
    assert events[0].thread == "agent-1"
    assert events[0].payload["source"] == "codex"
    assert events[1].type == "memory.reminder.suggested"
    assert events[1].payload["recommended_tool"] == "memory_checkout"


def test_hook_event_resume_suggests_fresh_checkout_reminder(tmp_path: Path) -> None:
    """Resume hooks should reintroduce checkout guidance even after recent memory use."""
    runner = CliRunner()
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={"activity": "checkout", "source": "test"},
        thread="agent-1",
    )

    result = runner.invoke(
        app,
        [
            "hook-event",
            "resume",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--summary",
            "resume after Codex update",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded hook resume as hook.resumed" in result.output
    assert "Suggested memory reminder" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert [event.type for event in events] == [
        "memory.checkout.completed",
        "hook.resumed",
        "memory.reminder.suggested",
    ]
    assert events[1].payload["trigger"] == "resume"
    assert events[1].payload["summary"] == "resume after Codex update"
    assert events[2].payload["trigger"] == "resume"
    assert events[2].payload["query"] == "resume after Codex update"
    assert events[2].payload["reasons"] == ["context_boundary"]


def test_hook_event_checkpoint_carries_summary_and_reason(tmp_path: Path) -> None:
    """checkpoint hooks should carry retrieval-useful checkpoint metadata."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "checkpoint",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--summary",
            "Finished hook install mode.",
            "--reason",
            "manual",
            "--turn-count",
            "7",
        ],
    )

    assert result.exit_code == 0
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "hook.checkpoint"
    assert events[0].payload["summary"] == "Finished hook install mode."
    assert events[0].payload["reason"] == "manual"
    assert events[0].payload["turn_count"] == 7


def test_hook_event_heartbeat_appends_health_event(tmp_path: Path) -> None:
    """heartbeat hooks should prove the observer path can write Eventloom."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "heartbeat",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "claude-code",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded hook heartbeat" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "hook.heartbeat"
    assert events[0].payload["trigger"] == "heartbeat"
    assert events[0].payload["source"] == "claude-code"


def test_hook_event_suppresses_reminder_after_recent_checkout(tmp_path: Path) -> None:
    """Hooks should not spam reminders when checkout was just used."""
    runner = CliRunner()
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={"activity": "checkout", "source": "test"},
        thread="agent-1",
    )

    result = runner.invoke(
        app,
        [
            "hook-event",
            "checkpoint",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--summary",
            "Routine checkpoint.",
            "--reason",
            "interval",
        ],
    )

    assert result.exit_code == 0
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert [event.type for event in events] == ["memory.checkout.completed", "hook.checkpoint"]


def test_hook_event_long_command_suggests_memory_reminder(tmp_path: Path) -> None:
    """Long tool runs should reintroduce Zaxy when memory has gone stale."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "command",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--command",
            "pytest",
            "--exit-code",
            "0",
            "--duration-ms",
            "45000",
        ],
    )

    assert result.exit_code == 0
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert [event.type for event in events] == ["command.completed", "memory.reminder.suggested"]
    assert "context_boundary" in events[1].payload["reasons"]


def test_hook_event_command_observation_appends_normalized_event(tmp_path: Path) -> None:
    """hook-event command should write first-class command.completed observations."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "command",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--workspace",
            "/repo",
            "--command",
            "pytest --token secret",
            "--exit-code",
            "1",
            "--stdout",
            "ok",
            "--stderr",
            "failed",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded observation command.completed" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "command.completed"
    assert events[0].actor == "zaxy-observer"
    assert events[0].payload["command"] == "pytest [REDACTED]"
    assert events[0].payload["source"] == "codex"
    assert events[0].payload["workspace"] == "/repo"


def test_hook_event_file_edit_observation_appends_normalized_event(tmp_path: Path) -> None:
    """hook-event file-edit should write first-class file.edit.applied observations."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "file-edit",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--workspace",
            "/repo",
            "--path",
            "src/zaxy/core.py",
            "--operation",
            "modified",
            "--summary",
            "Updated context assembly.",
            "--line-count",
            "12",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded observation file.edit.applied" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "file.edit.applied"
    assert events[0].actor == "zaxy-observer"
    assert events[0].payload["path"] == "src/zaxy/core.py"
    assert events[0].payload["summary"] == "Updated context assembly."
    assert "content" not in events[0].payload


def test_hook_event_tool_call_observation_appends_redacted_event(tmp_path: Path) -> None:
    """hook-event tool-call should write first-class tool.call.completed observations."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "tool-call",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--workspace",
            "/repo",
            "--tool-name",
            "functions.exec_command",
            "--tool-status",
            "ok",
            "--call-id",
            "call-123",
            "--arguments-json",
            '{"cmd": "pytest", "token": "secret"}',
            "--result-summary",
            "3 passed",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded observation tool.call.completed" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "tool.call.completed"
    assert events[0].actor == "zaxy-observer"
    assert events[0].payload["tool_name"] == "functions.exec_command"
    assert events[0].payload["argument_keys"] == ["cmd", "token"]
    assert events[0].payload["arguments_redacted"] is True
    assert "arguments" not in events[0].payload
    assert events[0].payload["source"] == "codex"
    assert events[0].payload["workspace"] == "/repo"


def test_hook_event_transcript_turn_observation_appends_sanitized_event(tmp_path: Path) -> None:
    """hook-event transcript-turn should write sanitized transcript.turn observations."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "transcript-turn",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--role",
            "assistant",
            "--content",
            "Use token sk-test-secret for the demo.",
            "--turn-index",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded observation transcript.turn" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "transcript.turn"
    assert events[0].actor == "assistant"
    assert events[0].payload["source"] == "codex"
    assert events[0].payload["turn_index"] == 7
    assert events[0].payload["role"] == "assistant"
    assert "sk-test-secret" not in events[0].payload["content"]
    assert events[0].payload["redacted_paths"]


def test_hook_status_reports_observation_type_coverage(tmp_path: Path) -> None:
    """hook-status should show which automatic capture types are active."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append(
        "hook.heartbeat",
        actor="zaxy-hook",
        payload={"trigger": "heartbeat", "source": "codex"},
        thread="agent-1",
    )
    command = log.append(
        "command.completed",
        actor="zaxy-observer",
        payload={"source": "codex", "command": "pytest", "exit_code": 0},
        thread="agent-1",
    )
    log.append(
        "file.edit.applied",
        actor="zaxy-observer",
        payload={"source": "codex", "path": "src/zaxy/core.py", "operation": "modified"},
        thread="agent-1",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["observation_coverage"]["command.completed"]["count"] == 1
    assert payload["observation_coverage"]["command.completed"]["latest"]["seq"] == command.seq
    assert payload["observation_coverage"]["file.edit.applied"]["count"] == 1
    assert payload["observation_coverage"]["transcript.turn"]["count"] == 0
    assert "transcript.turn" in payload["missing_observation_types"]
    assert payload["capture_readiness"] == {
        "status": "warning",
        "message": "2 of 4 high-value automatic capture lanes are active",
        "active_observation_types": ["command.completed", "file.edit.applied"],
        "missing_observation_types": ["tool.call.completed", "transcript.turn"],
        "actions": [
            "Wire hooks or adapter sinks for: tool.call.completed, transcript.turn.",
        ],
    }


def test_hook_status_reports_complete_observation_coverage(tmp_path: Path) -> None:
    """hook-status should clear missing coverage once every high-value type is captured."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    log.append("command.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("file.edit.applied", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("tool.call.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("transcript.turn", actor="assistant", payload={"source": "codex"}, thread="agent-1")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["observation_coverage"]["tool.call.completed"]["count"] == 1
    assert payload["observation_coverage"]["transcript.turn"]["count"] == 1
    assert payload["missing_observation_types"] == []
    assert payload["capture_readiness"] == {
        "status": "ok",
        "message": "4 of 4 high-value automatic capture lanes are active",
        "active_observation_types": [
            "command.completed",
            "file.edit.applied",
            "tool.call.completed",
            "transcript.turn",
        ],
        "missing_observation_types": [],
        "actions": [],
    }


def test_hook_status_reports_memory_activation_posture(tmp_path: Path) -> None:
    """hook-status should show whether the model has actually used memory checkout."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    checkout = log.append(
        "memory.checkout.completed",
        actor="assistant",
        payload={
            "activity": "checkout",
            "query": "current roadmap",
            "token_efficiency": {
                "prompt_tokens": 200,
                "current_fact_count": 3,
                "evidence_count": 4,
                "facts_per_1k_prompt_tokens": 15.0,
            },
        },
        thread="agent-1",
        timestamp=now - timedelta(minutes=20),
    )
    capture = log.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "codex", "role": "assistant"},
        thread="agent-1",
        timestamp=now - timedelta(minutes=5),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--max-checkout-stale-minutes",
            "60",
            "--now",
            now.isoformat(),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["memory_activation"]["activation_efficiency"] == {
        "high_context_session_count": 1,
        "fresh_checkout_session_count": 1,
        "stale_checkout_session_count": 0,
        "missing_checkout_session_count": 0,
        "fresh_checkout_rate": 1.0,
        "sessions": [
            {
                "session_id": "agent-1",
                "status": "fresh_checkout",
                "first_substantive_event": {
                    "seq": capture.seq,
                    "hash": capture.hash,
                    "timestamp": capture.timestamp,
                    "type": "transcript.turn",
                    "thread": "agent-1",
                    "source": "codex",
                },
                "checkout": {
                    "seq": checkout.seq,
                    "hash": checkout.hash,
                    "timestamp": checkout.timestamp,
                    "type": "memory.checkout.completed",
                    "thread": "agent-1",
                    "source": "unknown",
                    "token_efficiency": {
                        "prompt_tokens": 200,
                        "current_fact_count": 3,
                        "evidence_count": 4,
                        "facts_per_1k_prompt_tokens": 15.0,
                    },
                },
            }
        ],
    }
    assert payload["memory_activation"] | {"activation_efficiency": None} == {
        "status": "ok",
        "message": "Latest memory checkout is fresh",
        "stale_after_minutes": 60,
        "latest_checkout": {
            "seq": checkout.seq,
            "hash": checkout.hash,
            "timestamp": checkout.timestamp,
            "type": "memory.checkout.completed",
            "thread": "agent-1",
            "source": "unknown",
            "token_efficiency": {
                "prompt_tokens": 200,
                "current_fact_count": 3,
                "evidence_count": 4,
                "facts_per_1k_prompt_tokens": 15.0,
            },
        },
        "latest_capture": {
            "seq": capture.seq,
            "hash": capture.hash,
            "timestamp": capture.timestamp,
            "type": "transcript.turn",
            "thread": "agent-1",
            "source": "codex",
        },
        "latest_reminder": None,
        "activation_efficiency": None,
        "actions": [],
        "remediations": [],
    }

    stale = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--max-checkout-stale-minutes",
            "10",
            "--now",
            now.isoformat(),
        ],
    )

    assert stale.exit_code == 0
    assert "Memory activation" in stale.output
    assert "status: warning" in stale.output
    assert "Latest memory checkout is stale" in stale.output
    assert "activation efficiency: 0.0% (0/1 high-context sessions)" in stale.output
    assert "checkout tokens: 200 prompt, 15.0 facts/1k prompt tokens" in stale.output
    assert "Run memory checkout before relying on Zaxy context." in stale.output
    assert (
        "zaxy memory checkout 'current project memory and next useful action' "
        f"--eventloom-path {tmp_path / '.eventloom'} --session-id agent-1"
    ) in stale.output


def test_hook_status_warns_when_memory_checkout_has_never_run(tmp_path: Path) -> None:
    """hook-status should distinguish capture-only setups from active memory use."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    capture = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "command.completed",
        actor="zaxy-observer",
        payload={"source": "codex", "command": "pytest"},
        thread="agent-1",
        timestamp=now,
    )
    reminder = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "memory.reminder.suggested",
        actor="zaxy-memory",
        payload={"recommended_tool": "memory_checkout", "source": "codex"},
        thread="agent-1",
        timestamp=now,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--now",
            now.isoformat(),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["memory_activation"]["status"] == "warning"
    assert payload["memory_activation"]["message"] == "No memory checkout events found"
    assert payload["memory_activation"]["latest_checkout"] is None
    assert payload["memory_activation"]["latest_capture"]["seq"] == capture.seq
    assert payload["memory_activation"]["latest_reminder"]["seq"] == reminder.seq
    assert payload["memory_activation"]["latest_reminder"]["type"] == "memory.reminder.suggested"
    assert payload["memory_activation"]["actions"] == [
        "Run memory checkout before relying on Zaxy context.",
    ]


def test_hook_status_memory_activation_remediation_includes_checkout_command(tmp_path: Path) -> None:
    """hook-status JSON should give clients a directly runnable checkout remediation."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "command.completed",
        actor="zaxy-observer",
        payload={"source": "codex", "command": "pytest"},
        thread="agent-1",
        timestamp=now,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--now",
            now.isoformat(),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["memory_activation"]["remediations"] == [
        {
            "code": "missing_checkout",
            "message": "Run Memory Checkout before the next model or task call.",
            "command": (
                "zaxy memory checkout 'current project memory and next useful action' "
                f"--eventloom-path {tmp_path / '.eventloom'} --session-id agent-1"
            ),
        }
    ]


def test_hook_status_reports_activation_efficiency_by_session(tmp_path: Path) -> None:
    """hook-status should quantify how often high-context sessions start with fresh checkout."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    ready = EventLog(tmp_path / ".eventloom" / "ready.jsonl")
    checkout = ready.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={"activity": "checkout", "source": "test"},
        thread="ready",
        timestamp=now - timedelta(minutes=10),
    )
    first_ready = ready.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "codex", "role": "assistant", "content": "Ship the roadmap."},
        thread="ready",
        timestamp=now - timedelta(minutes=5),
    )
    stale = EventLog(tmp_path / ".eventloom" / "stale.jsonl")
    stale.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={"activity": "checkout", "source": "test"},
        thread="stale",
        timestamp=now - timedelta(minutes=90),
    )
    first_stale = stale.append(
        "command.completed",
        actor="zaxy-observer",
        payload={"source": "codex", "command": "pytest"},
        thread="stale",
        timestamp=now - timedelta(minutes=1),
    )
    missing = EventLog(tmp_path / ".eventloom" / "missing.jsonl")
    first_missing = missing.append(
        "file.edit.applied",
        actor="zaxy-observer",
        payload={"source": "codex", "path": "src/zaxy/core.py"},
        thread="missing",
        timestamp=now,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--max-checkout-stale-minutes",
            "60",
            "--now",
            now.isoformat(),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    efficiency = payload["memory_activation"]["activation_efficiency"]
    assert efficiency["high_context_session_count"] == 3
    assert efficiency["fresh_checkout_session_count"] == 1
    assert efficiency["stale_checkout_session_count"] == 1
    assert efficiency["missing_checkout_session_count"] == 1
    assert efficiency["fresh_checkout_rate"] == 1 / 3
    assert payload["memory_activation"]["status"] == "warning"
    assert payload["memory_activation"]["message"] == "Some high-context sessions lack fresh memory checkout"
    assert payload["memory_activation"]["actions"] == [
        "Run memory checkout before continuing sessions without fresh Zaxy context.",
    ]
    assert efficiency["sessions"] == [
        {
            "session_id": "missing",
            "status": "missing_checkout",
            "first_substantive_event": {
                "seq": first_missing.seq,
                "hash": first_missing.hash,
                "timestamp": first_missing.timestamp,
                "type": "file.edit.applied",
                "thread": "missing",
                "source": "codex",
            },
            "checkout": None,
        },
        {
            "session_id": "ready",
            "status": "fresh_checkout",
            "first_substantive_event": {
                "seq": first_ready.seq,
                "hash": first_ready.hash,
                "timestamp": first_ready.timestamp,
                "type": "transcript.turn",
                "thread": "ready",
                "source": "codex",
            },
            "checkout": {
                "seq": checkout.seq,
                "hash": checkout.hash,
                "timestamp": checkout.timestamp,
                "type": "memory.checkout.completed",
                "thread": "ready",
                "source": "test",
            },
        },
        {
            "session_id": "stale",
            "status": "stale_checkout",
            "first_substantive_event": {
                "seq": first_stale.seq,
                "hash": first_stale.hash,
                "timestamp": first_stale.timestamp,
                "type": "command.completed",
                "thread": "stale",
                "source": "codex",
            },
            "checkout": {
                "seq": 1,
                "hash": stale.read_all()[0].hash,
                "timestamp": stale.read_all()[0].timestamp,
                "type": "memory.checkout.completed",
                "thread": "stale",
                "source": "test",
            },
        },
    ]


def test_hook_status_can_fail_activation_efficiency_guardrail(tmp_path: Path) -> None:
    """hook-status should be usable as a release gate for fresh checkout adoption."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    ready = EventLog(tmp_path / ".eventloom" / "ready.jsonl")
    ready.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={"activity": "checkout", "source": "test"},
        thread="ready",
        timestamp=now - timedelta(minutes=5),
    )
    ready.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "codex"},
        thread="ready",
        timestamp=now - timedelta(minutes=1),
    )
    EventLog(tmp_path / ".eventloom" / "missing.jsonl").append(
        "command.completed",
        actor="zaxy-observer",
        payload={"source": "codex"},
        thread="missing",
        timestamp=now,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--now",
            now.isoformat(),
            "--min-activation-rate",
            "0.8",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["activation_guardrail"] == {
        "status": "fail",
        "threshold": 0.8,
        "fresh_checkout_rate": 0.5,
        "message": "activation efficiency 50.0% is below required 80.0%",
    }

    passing = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--now",
            now.isoformat(),
            "--min-activation-rate",
            "0.5",
        ],
    )

    assert passing.exit_code == 0
    assert "Activation guardrail: OK (50.0% >= 50.0%)" in passing.output


def test_hook_status_can_fail_checkout_token_efficiency_guardrail(tmp_path: Path) -> None:
    """hook-status should gate whether fresh checkout is compact enough to be useful."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    log = EventLog(tmp_path / ".eventloom" / "agent.jsonl")
    log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={
            "activity": "checkout",
            "source": "test",
            "token_efficiency": {
                "prompt_tokens": 1400,
                "current_fact_count": 2,
                "evidence_count": 3,
                "facts_per_1k_prompt_tokens": 1.43,
            },
        },
        thread="agent",
        timestamp=now - timedelta(minutes=5),
    )
    log.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "codex"},
        thread="agent",
        timestamp=now - timedelta(minutes=1),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--now",
            now.isoformat(),
            "--max-checkout-prompt-tokens",
            "1000",
            "--min-checkout-facts-per-1k-tokens",
            "2.0",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["checkout_token_guardrail"] == {
        "status": "fail",
        "max_prompt_tokens": 1000,
        "min_facts_per_1k_prompt_tokens": 2.0,
        "prompt_tokens": 1400,
        "facts_per_1k_prompt_tokens": 1.43,
        "messages": [
            "checkout prompt tokens 1400 exceed maximum 1000",
            "checkout facts per 1k prompt tokens 1.43 below required 2.0",
        ],
    }

    passing = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--now",
            now.isoformat(),
            "--max-checkout-prompt-tokens",
            "1500",
            "--min-checkout-facts-per-1k-tokens",
            "1.0",
        ],
    )

    assert passing.exit_code == 0
    assert "Checkout token guardrail: OK (1400 prompt tokens, 1.43 facts/1k prompt tokens)" in passing.output


@patch("zaxy.hooks._iter_process_cmdlines")
def test_hook_status_reports_codex_capture_watcher_runtime(
    mock_processes: MagicMock,
    tmp_path: Path,
) -> None:
    """hook-status should distinguish installed Codex capture config from a running watcher."""
    capture_config = tmp_path / ".codex" / "zaxy-capture.json"
    capture_config.parent.mkdir()
    capture_config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "workspace": str(tmp_path),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent-1",
            }
        ),
        encoding="utf-8",
    )
    mock_processes.return_value = [
        (
            123,
            [
                "python",
                "-m",
                "zaxy",
                "codex-capture",
                "--workspace",
                str(tmp_path),
                "--eventloom-path",
                str(tmp_path / ".eventloom"),
                "--session-id",
                "agent-1",
                "--watch",
            ],
        )
    ]
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["clients"]["codex"]["installed"] is True
    assert payload["clients"]["codex"]["runtime"] == {
        "running": True,
        "pids": [123],
        "message": "Codex capture watcher is running",
    }

    text = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
        ],
    )

    assert text.exit_code == 0
    assert "Codex capture config: installed (.codex/zaxy-capture.json)" in text.output
    assert "Codex capture watcher: running pid=123" in text.output


@patch("zaxy.hooks._iter_process_cmdlines")
def test_hook_status_warns_when_codex_capture_configured_but_not_running(
    mock_processes: MagicMock,
    tmp_path: Path,
) -> None:
    """hook-status should not treat stale Codex coverage as an active watcher."""
    capture_config = tmp_path / ".codex" / "zaxy-capture.json"
    capture_config.parent.mkdir()
    capture_config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "workspace": str(tmp_path),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent-1",
            }
        ),
        encoding="utf-8",
    )
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    log.append("command.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("file.edit.applied", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("tool.call.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("transcript.turn", actor="assistant", payload={"source": "codex"}, thread="agent-1")
    mock_processes.return_value = []
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "warning"
    assert payload["clients"]["codex"]["runtime"]["running"] is False
    assert payload["capture_readiness"]["status"] == "warning"
    assert payload["capture_readiness"]["actions"] == [
        f"Start managed deterministic Codex capture: zaxy capture start --workspace {tmp_path}."
    ]

    gated = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--require-capture-running",
            "--json",
        ],
    )

    assert gated.exit_code == 1
    gated_payload = json.loads(gated.output)
    assert gated_payload["capture_runtime_guardrail"] == {
        "status": "fail",
        "required": True,
        "configured": True,
        "running": False,
        "message": "Codex capture config is installed, but the managed watcher is not running",
        "action": f"zaxy capture start --workspace {tmp_path}",
    }


@patch("zaxy.hooks._iter_process_cmdlines")
def test_capture_status_reports_configured_codex_watcher_runtime(
    mock_processes: MagicMock,
    tmp_path: Path,
) -> None:
    """capture status should expose managed deterministic capture posture."""
    config = tmp_path / ".codex" / "zaxy-capture.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "codex_home": str(tmp_path / ".codex-home"),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent-1",
                "source": "codex-local",
                "workspace": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    mock_processes.return_value = [
        (
            321,
            [
                "python",
                "-m",
                "zaxy",
                "codex-capture",
                "--workspace",
                str(tmp_path),
                "--eventloom-path",
                str(tmp_path / ".eventloom"),
                "--session-id",
                "agent-1",
                "--watch",
            ],
        )
    ]
    runner = CliRunner()

    result = runner.invoke(app, ["capture", "status", "--workspace", str(tmp_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["client"] == "codex"
    assert payload["configured"] is True
    assert payload["running"] is True
    assert payload["pids"] == [321]
    assert payload["state_file"] == str(tmp_path / ".eventloom" / "runtime" / "codex-capture.json")


@patch.object(subprocess, "Popen")
def test_capture_start_launches_managed_codex_watcher(
    mock_popen: MagicMock,
    tmp_path: Path,
) -> None:
    """capture start should launch a watcher from repo-local Codex capture config."""
    config = tmp_path / ".codex" / "zaxy-capture.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "codex_home": str(tmp_path / ".codex-home"),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent-1",
                "source": "codex-local",
                "workspace": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    process = MagicMock()
    process.pid = 321
    mock_popen.return_value = process
    runner = CliRunner()

    result = runner.invoke(app, ["capture", "start", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Started Codex capture watcher pid=321" in result.output
    command = mock_popen.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "zaxy"]
    assert "codex-capture" in command
    assert "--watch" in command
    assert mock_popen.call_args.kwargs["start_new_session"] is True
    state = json.loads((tmp_path / ".eventloom" / "runtime" / "codex-capture.json").read_text(encoding="utf-8"))
    assert state["pid"] == 321
    assert state["client"] == "codex"
    assert state["workspace"] == str(tmp_path)


@patch("os.kill")
@patch("zaxy.hooks._iter_process_cmdlines")
def test_capture_stop_only_stops_matching_managed_codex_watcher(
    mock_processes: MagicMock,
    mock_kill: MagicMock,
    tmp_path: Path,
) -> None:
    """capture stop should stop the managed watcher without targeting unrelated processes."""
    runtime = tmp_path / ".eventloom" / "runtime"
    runtime.mkdir(parents=True)
    state_file = runtime / "codex-capture.json"
    state_file.write_text(
        json.dumps(
            {
                "client": "codex",
                "pid": 321,
                "workspace": str(tmp_path),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "command": ["python", "-m", "zaxy", "codex-capture", "--watch"],
            }
        ),
        encoding="utf-8",
    )
    mock_processes.return_value = [
        (
            321,
            [
                "python",
                "-m",
                "zaxy",
                "codex-capture",
                "--workspace",
                str(tmp_path),
                "--eventloom-path",
                str(tmp_path / ".eventloom"),
                "--session-id",
                "agent-1",
                "--watch",
            ],
        )
    ]
    runner = CliRunner()

    result = runner.invoke(app, ["capture", "stop", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Stopped Codex capture watcher pid=321" in result.output
    mock_kill.assert_called_once()
    assert mock_kill.call_args.args[0] == 321
    assert not state_file.exists()


def test_hooks_status_reports_installed_clients_and_recent_activity(tmp_path: Path) -> None:
    """hook-status should answer whether Zaxy is observing this workspace."""
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir()
    settings_path.write_text('{"hooks": {"Stop": [{"hooks": [{"command": "zaxy hook-event stop"}]}]}}', encoding="utf-8")
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "hook.heartbeat",
        actor="zaxy-hook",
        payload={"trigger": "heartbeat", "source": "claude-code"},
        thread="agent-1",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["hook-status", "--eventloom-path", str(tmp_path / ".eventloom"), "--workspace-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Zaxy hooks: warning" in result.output
    assert "Client setup" in result.output
    assert "Claude Code hook config: installed" in result.output
    assert "Codex capture config: missing" in result.output
    assert "Last observed event" in result.output
    assert "type: hook.heartbeat" in result.output
    assert "Capture readiness" in result.output
    assert "active lanes: 0 of 4" in result.output
    assert "Memory activation" in result.output
    assert "No memory checkout events found" in result.output
    assert "[ ] command.completed" in result.output
    assert "agent-1" in result.output


def test_schema_plan_command_prints_migration_plan() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["schema-plan"])

    assert result.exit_code == 0
    assert "Current schema version:" in result.output
    assert "entity_version_identity" in result.output


def test_schema_recovery_plan_command_prints_recovery_guidance() -> None:
    runner = CliRunner()

    with (
        patch("zaxy.graph.GraphStore") as mock_store_cls,
        patch("zaxy.schema.fetch_schema_migration_records", new_callable=AsyncMock) as mock_fetch,
    ):
        store = AsyncMock()
        mock_store_cls.return_value = store
        mock_fetch.return_value = {
            "001_entity_version_identity": {
                "checksum": "wrong",
                "statement_count": 4,
                "applied_at": "2026-05-11T00:00:00Z",
            }
        }
        result = runner.invoke(app, ["schema-recovery-plan"])

    assert result.exit_code == 0
    store.connect.assert_awaited_once()
    store.close.assert_awaited_once()
    assert "Schema recovery plan:" in result.output
    assert "001_entity_version_identity: checksum_mismatch" in result.output


def test_extractor_template_command_prints_safe_starter() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "extractor-template",
            "decision.recorded",
            "--entity-type",
            "decision",
            "--name-key",
            "title",
            "--summary-key",
            "rationale",
            "--actor-relation",
            "recorded_decision",
        ],
    )

    assert result.exit_code == 0
    assert '@register("decision.recorded")' in result.output
    assert 'relation_type="recorded_decision"' in result.output


def test_local_profile_command_prints_offline_env() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["local-profile"])

    assert result.exit_code == 0
    assert "PROJECTION_BACKEND=embedded" in result.output
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" in result.output
    assert "EMBEDDING_PROVIDER=hash" in result.output
    assert "RERANKER_PROVIDER=lexical" in result.output
    assert "NEO4J_AUTO_START=false" in result.output
    assert "NEO4J_URI=bolt://localhost:7687" in result.output
    assert "NEO4J_USER=neo4j" in result.output
    assert "NEO4J_PASSWORD=testpassword" in result.output
    assert "NEO4J_CA_CERT=" in result.output
    assert "NEO4J_PASSWORD_FILE=" in result.output
    assert "NEO4J_TRUST_ALL=false" in result.output
    assert "OPENAI_API_KEY" not in result.output


def test_local_profile_command_writes_output_file(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / ".env.local"

    result = runner.invoke(app, ["local-profile", "--output", str(target)])

    assert result.exit_code == 0
    assert "Wrote local profile" in result.output
    profile = target.read_text(encoding="utf-8")
    assert "PROJECTION_BACKEND=embedded" in profile
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" in profile
    assert "RERANKER_PROVIDER=lexical" in profile
    assert "NEO4J_AUTO_START=false" in profile
    assert "NEO4J_URI=bolt://localhost:7687" in profile
    assert "NEO4J_CA_CERT=" in profile
    assert "NEO4J_PASSWORD_FILE=" in profile


def test_local_profile_command_can_render_embedded_projection_profile(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / ".env.local"

    result = runner.invoke(app, ["local-profile", "--projection-backend", "embedded", "--output", str(target)])

    assert result.exit_code == 0
    profile = target.read_text(encoding="utf-8")
    assert "PROJECTION_BACKEND=embedded" in profile
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" in profile
    assert "NEO4J_AUTO_START=false" in profile


def test_local_profile_check_reports_success() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["local-profile", "--check"])

    assert result.exit_code == 0
    assert '"status": "ok"' in result.output
    assert '"reranker_provider": "lexical"' in result.output


def test_doctor_command_reports_text_summary(tmp_path: Path) -> None:
    runner = CliRunner()
    EventLog(tmp_path / ".eventloom" / "default.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        payload={"session_id": "default"},
        thread="default",
    )

    result = runner.invoke(app, ["doctor", "--eventloom-path", str(tmp_path / ".eventloom")])

    assert result.exit_code == 0
    assert "Zaxy doctor:" in result.output
    assert "eventloom: ok" in result.output
    assert "viewer: ok" in result.output
    assert "captured=1 projected=0 unprojected=1 reinforced=0 eligible=0" in result.output


def test_doctor_command_reports_json(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["doctor", "--eventloom-path", str(tmp_path / ".eventloom"), "--json"],
    )

    assert result.exit_code == 0
    assert '"status": "ok"' in result.output
    assert '"name": "eventloom"' in result.output


def test_doctor_repairs_stale_embedded_mcp_runtime(tmp_path: Path) -> None:
    """Doctor should clean stale embedded owner metadata before the next MCP startup."""
    runner = CliRunner()
    eventloom = tmp_path / ".eventloom"
    runtime = eventloom / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "zaxy-embedded-owner.json").write_text(
        '{"pid": 999999999, "socket_path": "/tmp/missing-zaxy.sock"}',
        encoding="utf-8",
    )
    (runtime / "zaxy-embedded-owner.sock").write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        ["doctor", "--eventloom-path", str(eventloom), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["embedded_mcp_runtime"]["status"] == "ok"
    assert checks["embedded_mcp_runtime"]["details"]["repaired"] is True
    assert not (runtime / "zaxy-embedded-owner.json").exists()
    assert not (runtime / "zaxy-embedded-owner.sock").exists()


def test_doctor_release_smoke_reports_packaging_readiness() -> None:
    """Release smoke mode should verify local release metadata without external services."""
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--release-smoke", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["package_version"]["status"] == "ok"
    assert checks["changelog"]["status"] == "ok"
    assert checks["trusted_publishing"]["status"] == "ok"
    assert checks["release_workflow"]["status"] == "ok"
    assert checks["langgraph_example"]["status"] == "ok"
    assert "examples/langgraph_memory.py" in checks["langgraph_example"]["message"]
    assert checks["openai_compatible_example"]["status"] == "ok"
    assert "examples/openai_compatible_memory.py" in checks["openai_compatible_example"]["message"]
    assert checks["claude_compatible_example"]["status"] == "ok"
    assert "examples/claude_compatible_memory.py" in checks["claude_compatible_example"]["message"]


def test_doctor_beta_readiness_reports_release_and_uat_gates() -> None:
    """Beta readiness should summarize the release, UAT, docs, and capture gates."""
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--beta-readiness", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["release_smoke"]["status"] == "ok"
    assert checks["release_gate"]["status"] == "ok"
    assert "optional backend exclusion" in checks["release_gate"]["message"]
    assert checks["coordination_competitor_claims"]["status"] == "ok"
    assert "Quarq/Hybi posture is guarded" in checks["coordination_competitor_claims"]["message"]
    assert checks["purpose_benchmark_gate"]["status"] == "ok"
    assert "purpose-v1 benchmark passes all purpose-memory lanes" in checks[
        "purpose_benchmark_gate"
    ]["message"]
    assert checks["purpose_evidence_policy"]["status"] == "ok"
    assert "support, product, sales, legal, and executive evidence-policy fixtures" in checks[
        "purpose_evidence_policy"
    ]["message"]
    assert checks["external_validation_evidence"]["status"] == "ok"
    assert "external validation is optional for v1.0 release" in checks["external_validation_evidence"]["message"]
    assert checks["clean_repo_uat"]["status"] == "ok"
    assert checks["docs_happy_path"]["status"] == "ok"
    assert checks["capture_happy_path"]["status"] == "ok"
    assert checks["first_run_timing"]["status"] == "ok"
    assert "300 seconds" in checks["first_run_timing"]["message"]
    assert "scripts/beta-uat.sh" in checks["clean_repo_uat"]["message"]


def test_doctor_beta_readiness_accepts_external_validation_report_path(tmp_path: Path) -> None:
    """Beta readiness CLI should accept explicit external-validation evidence."""
    runner = CliRunner()
    report_path = tmp_path / "external-validation-report.json"
    report_path.write_text(
        json.dumps(
            {
                "contract": "zaxy.v1.external-validation-report",
                "status": "validated",
                "validator": {
                    "name": "Independent Validation Project",
                    "external_to_implementation_session": True,
                },
                "date": "2026-05-31",
                "zaxy_version_or_commit": "v1.0.0-rc",
                "environment": {
                    "operating_system": "Linux",
                    "shell": "bash",
                    "python_version": "3.13",
                    "install_source": "pipx install zaxy-memory",
                },
                "validation_path": "first_run_local",
                "commands": [
                    "zaxy init",
                    "zaxy memory bootstrap --eventloom-path .eventloom",
                    "zaxy memory checkout current project memory --eventloom-path .eventloom",
                    "zaxy doctor --beta-readiness",
                ],
                "time_to_first_useful_checkout_seconds": 120,
                "unexpected_sidecar_or_credential_required": False,
                "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
                "friction_or_failure": "No blocking friction.",
                "release_decision": "pass",
                "supports_positioning": True,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "doctor",
            "--beta-readiness",
            "--external-validation-report",
            str(report_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["external_validation_evidence"]["status"] == "ok"
    assert str(report_path) in checks["external_validation_evidence"]["message"]


def test_doctor_beta_readiness_can_require_external_validation() -> None:
    """Strict beta readiness should fail when external validation is still pending."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["doctor", "--beta-readiness", "--require-external-validation", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["external_validation_evidence"]["status"] == "error"
    assert "external validation is required" in checks["external_validation_evidence"]["message"]


def test_doctor_rejects_external_validation_options_without_beta_readiness(tmp_path: Path) -> None:
    """External-validation options only apply to beta readiness checks."""
    runner = CliRunner()
    report_path = tmp_path / "external-validation-report.json"
    report_path.write_text("{}", encoding="utf-8")

    normal = runner.invoke(app, ["doctor", "--external-validation-report", str(report_path)])
    strict = runner.invoke(app, ["doctor", "--require-external-validation"])
    release = runner.invoke(
        app,
        ["doctor", "--release-smoke", "--external-validation-report", str(report_path)],
    )

    for result in (normal, strict, release):
        assert result.exit_code == 2
        assert "Invalid value" in result.output
        assert "external validation options require" in result.output
        assert "readiness" in result.output


def test_doctor_beta_readiness_fails_nonzero_for_unready_project(tmp_path: Path) -> None:
    """Beta readiness should be shell-gatable when a project is missing beta gates."""
    runner = CliRunner()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["doctor", "--beta-readiness", "--project-root", str(tmp_path), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["clean_repo_uat"]["status"] == "error"


def test_doctor_release_smoke_uses_explicit_project_root(tmp_path: Path) -> None:
    """Release smoke should support checking a repo root different from cwd."""
    runner = CliRunner()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 0.2.0 - 2026-05-11\n\n- Stable release.\n",
        encoding="utf-8",
    )
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    (tmp_path / ".github" / "workflows" / "publish.yml").write_text(
        "on:\n"
        "  release:\n"
        "    types: [published]\n"
        "  workflow_dispatch:\n"
        "permissions:\n"
        "  id-token: write\n"
        "steps:\n"
        "  - run: python -m build --sdist --wheel\n"
        "  - run: python -m twine check dist/*\n"
        "  - uses: pypa/gh-action-pypi-publish@release/v1\n",
        encoding="utf-8",
    )
    (tmp_path / "examples" / "langgraph_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'langgraph-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )
    (tmp_path / "examples" / "openai_compatible_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'openai-compatible-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )
    (tmp_path / "examples" / "claude_compatible_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'claude-compatible-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["doctor", "--release-smoke", "--project-root", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"


def test_packet_status_command_reports_text_summary(tmp_path: Path) -> None:
    runner = CliRunner()
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    packet = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        payload={"session_id": "agent-1"},
        thread="agent-1",
    )
    log.append(
        "llm.packet.projected",
        actor="zaxy-packet-projector",
        payload={"source_event_hash": packet.hash, "source_event_seq": packet.seq},
        thread="agent-1",
    )

    result = runner.invoke(
        app,
        ["packet-status", "--eventloom-path", str(tmp_path / ".eventloom"), "--session-id", "agent-1"],
    )

    assert result.exit_code == 0
    assert "Zaxy packet memory: ok" in result.output
    assert "captured=1 projected=1 unprojected=0 reinforced=0 eligible=1" in result.output


def test_packet_status_command_reports_activation_steps_when_inactive(tmp_path: Path) -> None:
    """packet-status should tell operators how to activate capture when none exists."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "packet-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--analyzer-port",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Zaxy packet memory: warning" in result.output
    assert "analyzer: inactive (http://127.0.0.1:1/v1)" in result.output
    assert "Start packet analyzer: zaxy packet-analyzer" in result.output
    assert "Start packet projector: zaxy packet-project" in result.output
    assert "http://127.0.0.1:1/v1" in result.output


def test_packet_status_command_reports_json(tmp_path: Path) -> None:
    runner = CliRunner()
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        payload={"session_id": "agent-1"},
        thread="agent-1",
    )

    result = runner.invoke(
        app,
        [
            "packet-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"status": "warning"' in result.output
    assert '"unprojected": 1' in result.output


@patch("zaxy.cli.runtime.capture_codex_sessions")
def test_codex_capture_command_imports_local_codex_records(mock_capture: MagicMock, tmp_path: Path) -> None:
    """codex-capture should expose deterministic local Codex observation import."""
    mock_capture.return_value.imported = 4
    mock_capture.return_value.scanned_files = 1
    mock_capture.return_value.skipped = 2
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "codex-capture",
            "--workspace",
            str(tmp_path),
            "--codex-home",
            str(tmp_path / "codex-home"),
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "repo-default",
        ],
    )

    assert result.exit_code == 0
    assert "Imported 4 Codex observations from 1 session log" in result.output
    mock_capture.assert_called_once_with(
        workspace=tmp_path,
        codex_home=tmp_path / "codex-home",
        eventloom_path=tmp_path / ".eventloom",
        session_id="repo-default",
        source="codex-local",
        max_records_per_file=1000,
    )


@patch("zaxy.cli.workspace.time.sleep")
@patch("zaxy.cli.runtime.capture_codex_sessions")
def test_codex_capture_watch_mode_supports_bounded_iterations(
    mock_capture: MagicMock,
    mock_sleep: MagicMock,
    tmp_path: Path,
) -> None:
    """codex-capture --watch should support bounded supervisor/test runs."""
    first = MagicMock(imported=2, scanned_files=1, skipped=0)
    second = MagicMock(imported=0, scanned_files=1, skipped=2)
    mock_capture.side_effect = [first, second]
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "codex-capture",
            "--workspace",
            str(tmp_path),
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "repo-default",
            "--watch",
            "--watch-iterations",
            "2",
            "--interval-seconds",
            "0.25",
        ],
    )

    assert result.exit_code == 0
    assert "Watching Codex session logs" in result.output
    assert result.output.count("Imported ") == 2
    assert mock_capture.call_count == 2
    mock_sleep.assert_called_once_with(0.25)


@patch("zaxy.cli.runtime.GraphStore")
@patch("zaxy.cli.runtime.capture_codex_sessions")
def test_codex_capture_can_project_captured_events_to_graph(
    mock_capture: MagicMock,
    mock_graph_store: MagicMock,
    tmp_path: Path,
) -> None:
    """codex-capture --graph should project only events captured in that pass."""
    event = EventLog(tmp_path / ".eventloom" / "repo-default.jsonl").append(
        "transcript.turn",
        actor="assistant",
        payload={"content": "Remember bounded capture."},
        thread="repo-default",
    )
    mock_capture.return_value.imported = 1
    mock_capture.return_value.scanned_files = 1
    mock_capture.return_value.skipped = 0
    mock_capture.return_value.events = (event,)
    store = AsyncMock()
    mock_graph_store.return_value = store
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "codex-capture",
            "--workspace",
            str(tmp_path),
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "repo-default",
            "--graph",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    assert "Projected 1 captured observations into graph" in result.output
    mock_graph_store.assert_called_once_with("bolt://test:7687", "neo4j", "testpassword")
    store.connect.assert_awaited_once()
    store.init_schema.assert_awaited_once()
    store.upsert_extraction.assert_awaited_once()
    assert store.upsert_extraction.await_args.kwargs == {"session_id": "repo-default"}
    store.close.assert_awaited_once()


@patch("zaxy.cli.runtime.MemoryFabric")
def test_index_codebase_command_reports_indexed_count(mock_fabric_cls: MagicMock, tmp_path: Path) -> None:
    """index-codebase should append codebase mapping events through MemoryFabric."""
    fabric = AsyncMock()
    fabric.ingest_codebase.return_value = 3
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["index-codebase", str(tmp_path), "--session-id", "agent-1", "--max-bytes", "1024"],
    )

    assert result.exit_code == 0
    assert "Indexed 3 codebase events into session agent-1" in result.output
    fabric.ingest_codebase.assert_awaited_once_with(tmp_path, session_id="agent-1", max_bytes=1024)
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime.MemoryFabric")
def test_refresh_context_command_uses_backend_aware_fabric(
    mock_fabric_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """refresh-context should expose backend selection while refreshing source context."""
    fabric = MagicMock()
    report = MagicMock()
    report.to_dict.return_value = {
        "session_id": "agent-1",
        "kind": "documents",
        "event_count": 3,
        "summary": {
            "kind": "documents",
            "discovered": 1,
            "changed": 0,
            "unchanged": 0,
            "deleted": 0,
            "indexed": 1,
            "retired": 0,
        },
    }
    fabric.refresh_context = AsyncMock(return_value=report)
    fabric.close = AsyncMock()
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "refresh-context",
            str(tmp_path),
            "--kind",
            "documents",
            "--session-id",
            "agent-1",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--projection-backend",
            "pggraph",
            "--pggraph-dsn",
            "postgresql://postgres:postgres@localhost:5432/zaxy",
            "--max-lines",
            "20",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["summary"]["indexed"] == 1
    mock_fabric_cls.assert_called_once_with(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="pggraph",
        pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        embedded_graph_path=Path(".eventloom/projections/embedded.kuzu"),
        latticedb_path=Path(".eventloom/projections/memory.latticedb"),
        tracer_disabled=False,
    )
    fabric.refresh_context.assert_awaited_once_with(
        tmp_path,
        kind="documents",
        session_id="agent-1",
        max_lines=20,
        max_bytes=512 * 1024,
    )
    fabric.close.assert_awaited_once()


@patch("zaxy.cli.runtime.MemoryFabric")
def test_refresh_context_command_uses_repo_local_embedded_profile(
    mock_fabric_cls: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """refresh-context should use the repo-local embedded profile by default."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    embedded_path = workspace / ".eventloom" / "projections" / "embedded.kuzu"
    (workspace / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n",
        encoding="utf-8",
    )
    fabric = MagicMock()
    report = MagicMock()
    report.to_dict.return_value = {
        "session_id": "agent-1",
        "kind": "documents",
        "event_count": 0,
        "summary": {
            "kind": "documents",
            "discovered": 0,
            "changed": 0,
            "unchanged": 0,
            "deleted": 0,
            "indexed": 0,
            "retired": 0,
        },
    }
    fabric.refresh_context = AsyncMock(return_value=report)
    fabric.close = AsyncMock()
    mock_fabric_cls.return_value = fabric
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "refresh-context",
            str(workspace),
            "--eventloom-path",
            str(workspace / ".eventloom"),
            "--session-id",
            "agent-1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    kwargs = mock_fabric_cls.call_args.kwargs
    assert kwargs["projection_backend"] == "embedded"
    assert kwargs["embedded_graph_path"] == embedded_path


@patch("zaxy.cli.runtime.MemoryFabric")
def test_init_session_command_reports_workspace_profile(mock_fabric_cls: MagicMock, tmp_path: Path) -> None:
    """init-session should append a genesis event through MemoryFabric."""
    fabric = AsyncMock()
    fabric.initialize_session.return_value.workspace_type = "codebase"
    fabric.initialize_session.return_value.confidence = 0.8
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(app, ["init-session", str(tmp_path), "--session-id", "agent-1"])

    assert result.exit_code == 0
    assert "Initialized agent-1 as codebase workspace (confidence 0.8)" in result.output
    fabric.initialize_session.assert_awaited_once_with(tmp_path, session_id="agent-1")
    fabric.close.assert_awaited_once()


def test_init_command_runs_first_run_onboarding(tmp_path: Path) -> None:
    """init should expose the unified first-run onboarding orchestrator."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(workspace),
            "--domain",
            "demo",
            "--mcp-client",
            "claude-desktop",
            "--mcp-output",
            str(workspace / "mcp.json"),
            "--hook-client",
            "claude-code",
            "--hook-output",
            str(workspace / ".claude" / "settings.local.json"),
            "--local-profile-output",
            str(workspace / ".env.local"),
        ],
    )

    assert result.exit_code == 0
    assert "Zaxy init complete: ok" in result.output
    assert "Session: demo-default" in result.output
    assert "Setup:" in result.output
    assert "[OK] mcp_config" not in result.output
    assert "[OK] hook_status" not in result.output
    assert (workspace / "mcp.json").is_file()
    assert (workspace / ".claude" / "settings.local.json").is_file()
    assert (workspace / ".eventloom" / "demo-default.jsonl").is_file()
    local_profile = (workspace / ".env.local").read_text(encoding="utf-8")
    assert "NEO4J_URI=bolt://localhost:7687" in local_profile
    assert "NEO4J_CA_CERT=" in local_profile
    assert "NEO4J_PASSWORD_FILE=" in local_profile
    assert "NEO4J_TRUST_ALL=false" in local_profile


def test_init_command_verbose_prints_full_setup_diagnostics(tmp_path: Path) -> None:
    """init --verbose should keep full setup rows available without making them the default."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(workspace),
            "--domain",
            "demo",
            "--mcp-client",
            "claude-desktop",
            "--mcp-output",
            str(workspace / "mcp.json"),
            "--hook-client",
            "claude-code",
            "--hook-output",
            str(workspace / ".claude" / "settings.local.json"),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "Setup:" in result.output
    assert "[OK] mcp_config" in result.output
    assert "[OK] hook_status" in result.output


def test_init_command_compact_more_hint_preserves_rerun_context(tmp_path: Path) -> None:
    """Compact init output should show a copyable verbose command for the same invocation context."""
    workspace = tmp_path / "repo with spaces"
    workspace.mkdir()
    codex_home = tmp_path / "codex home"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(workspace),
            "--domain",
            "demo project",
            "--preset",
            "local-codex",
            "--eventloom-path",
            str(workspace / ".events with spaces"),
            "--codex-mcp-install",
            "user",
            "--codex-home",
            str(codex_home),
            "--no-agent-instructions",
        ],
    )

    assert result.exit_code == 0
    expected = (
        "More: run zaxy init "
        f"{shlex.quote(str(workspace))} "
        f"--eventloom-path {shlex.quote(str(workspace / '.events with spaces'))} "
        "--domain 'demo project' "
        "--preset local-codex "
        "--codex-mcp-install user "
        f"--codex-home {shlex.quote(str(codex_home))} "
        "--no-agent-instructions "
        "--verbose "
        "to show checks, fallbacks, later commands, and notes."
    )
    assert expected in result.output


def test_init_command_compact_more_hint_preserves_env_codex_home(tmp_path: Path) -> None:
    """Compact rerun guidance should preserve env-derived Codex config targets."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text("", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(workspace),
            "--domain",
            "demo",
            "--preset",
            "local-codex",
            "--no-agent-instructions",
        ],
        env={"CODEX_HOME": str(codex_home)},
    )

    assert result.exit_code == 0
    expected = (
        "More: run zaxy init "
        f"{shlex.quote(str(workspace))} "
        "--domain demo "
        "--preset local-codex "
        f"--codex-home {shlex.quote(str(codex_home))} "
        "--no-agent-instructions "
        "--verbose "
        "to show checks, fallbacks, later commands, and notes."
    )
    assert expected in result.output


def test_init_command_conflict_more_hint_preserves_env_codex_home(tmp_path: Path) -> None:
    """Conflict rerun guidance should preserve env-derived Codex config targets."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "codex home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        '[mcp_servers.zaxy]\ncommand = "/custom/zaxy"\nargs = ["serve"]\n',
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(workspace),
            "--domain",
            "demo",
            "--preset",
            "local-codex",
            "--no-agent-instructions",
        ],
        env={"CODEX_HOME": str(codex_home)},
    )

    assert result.exit_code == 0
    expected = (
        "More: run zaxy init "
        f"{shlex.quote(str(workspace))} "
        "--domain demo "
        "--preset local-codex "
        f"--codex-home {shlex.quote(str(codex_home))} "
        "--no-agent-instructions "
        "--verbose "
        "to show checks, fallbacks, later commands, and notes."
    )
    assert expected in result.output


def test_init_command_rejects_mcp_output_without_client(tmp_path: Path) -> None:
    """init should reject renderer output paths without the matching client option."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(workspace), "--mcp-output", str(workspace / "mcp.json")])

    assert result.exit_code != 0
    assert "mcp_client is required" in result.output


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_passes_infra_action(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --infra should pass explicit infra action into the orchestrator."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--infra", "check"])

    assert result.exit_code == 0
    assert mock_run_onboarding.await_args.kwargs["infra"] == "check"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_passes_pggraph_bootstrap_options(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init should expose pgGraph bootstrap inputs to the onboarding orchestrator."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    repo = tmp_path / "pggraph"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--infra",
            "check",
            "--projection-backend",
            "pggraph",
            "--pggraph-dsn",
            "postgresql://postgres:postgres@localhost:5432/zaxy",
            "--pggraph-repo",
            str(repo),
        ],
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["projection_backend"] == "pggraph"
    assert kwargs["pggraph_dsn"] == "postgresql://postgres:postgres@localhost:5432/zaxy"
    assert kwargs["pggraph_repo"] == repo


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_passes_embedded_projection_backend(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init should expose embedded projection selection to onboarding infra checks."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--infra",
            "check",
            "--projection-backend",
            "embedded",
        ],
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["infra"] == "check"
    assert kwargs["projection_backend"] == "embedded"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_expands_local_embedded_codex_preset(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --preset local-embedded-codex should select embedded without extra flags."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--preset", "local-embedded-codex"])

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["projection_backend"] == "embedded"
    assert kwargs["infra"] == "check"
    assert kwargs["mcp_client"] == "codex"
    assert kwargs["hook_client"] == "codex"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_defaults_to_local_embedded_codex_onboarding(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """Bare init should be the one-command no-sidecar local onboarding path."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["projection_backend"] == "embedded"
    assert kwargs["infra"] == "check"
    assert kwargs["mcp_client"] == "codex"
    assert kwargs["mcp_output"] is None
    assert kwargs["hook_client"] == "codex"
    assert kwargs["hook_output"] == tmp_path / ".codex" / "zaxy-capture.json"
    assert kwargs["local_profile_output"] == tmp_path / ".env.local"
    assert kwargs["capture_mode"] == "deterministic"
    assert kwargs["capture_action"] == "none"
    assert kwargs["agent_instructions"] is True


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_can_skip_agent_instruction_install(
    mock_run_onboarding: AsyncMock,
    tmp_path: Path,
) -> None:
    """Users should be able to opt out of AGENTS.md activation block writes."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--no-agent-instructions"])

    assert result.exit_code == 0
    assert mock_run_onboarding.await_args.kwargs["agent_instructions"] is False


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_expands_local_claude_preset(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --preset local-claude should pass expanded explicit options."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--preset", "local-claude"])

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["mcp_client"] == "claude-desktop"
    assert kwargs["mcp_output"] == tmp_path / "zaxy-mcp.json"
    assert kwargs["hook_client"] == "claude-code"
    assert kwargs["hook_output"] == tmp_path / ".claude" / "settings.local.json"
    assert kwargs["local_profile_output"] == tmp_path / ".env.local"
    assert kwargs["infra"] == "check"
    assert kwargs["capture_mode"] == "deterministic"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_expands_local_codex_preset(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --preset local-codex should install safe repo-local capture config."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--preset", "local-codex"])

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["mcp_client"] == "codex"
    assert kwargs["mcp_output"] is None
    assert kwargs["hook_client"] == "codex"
    assert kwargs["hook_output"] == tmp_path / ".codex" / "zaxy-capture.json"
    assert kwargs["local_profile_output"] == tmp_path / ".env.local"
    assert kwargs["infra"] == "check"
    assert kwargs["capture_mode"] == "deterministic"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_passes_packet_capture_options(
    mock_run_onboarding: AsyncMock,
    tmp_path: Path,
) -> None:
    """init --packet-capture should pass packet activation settings to onboarding."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--packet-capture",
            "--packet-upstream-base-url",
            "https://api.openai.com/v1",
            "--packet-port",
            "8788",
        ],
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["packet_capture"] is True
    assert kwargs["capture_mode"] == "hybrid"
    assert kwargs["packet_upstream_base_url"] == "https://api.openai.com/v1"
    assert kwargs["packet_port"] == 8788


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_accepts_capture_mode_packet(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --capture-mode packet should explicitly opt into packet-capture guidance."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--capture-mode", "packet"])

    assert result.exit_code == 0
    assert mock_run_onboarding.await_args.kwargs["capture_mode"] == "packet"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_accepts_capture_start_action(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --capture start should ask onboarding to start deterministic capture."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--preset", "local-codex", "--capture", "start"])

    assert result.exit_code == 0
    assert mock_run_onboarding.await_args.kwargs["capture_action"] == "start"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_accepts_codex_mcp_install_options(
    mock_run_onboarding: AsyncMock,
    tmp_path: Path,
) -> None:
    """init should expose the no-copy-paste Codex MCP install path."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--preset",
            "local-codex",
            "--codex-mcp-install",
            "user",
            "--codex-trusted-project",
            "--codex-home",
            str(tmp_path / "codex-home"),
        ],
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["codex_mcp_install"] == "user"
    assert kwargs["codex_trusted_project"] is True
    assert kwargs["codex_home"] == tmp_path / "codex-home"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_auto_codex_mcp_install_uses_existing_user_config(
    mock_run_onboarding: AsyncMock,
    tmp_path: Path,
) -> None:
    """Default Codex onboarding should avoid copy/paste when a Codex config already exists."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text("", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["init", str(tmp_path), "--preset", "local-codex"],
        env={"CODEX_HOME": str(codex_home)},
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["codex_mcp_install"] == "user"
    assert kwargs["codex_home"] is None


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_auto_codex_mcp_install_keeps_command_without_existing_config(
    mock_run_onboarding: AsyncMock,
    tmp_path: Path,
) -> None:
    """Default Codex onboarding should stay non-invasive when no Codex config target exists."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    codex_home = tmp_path / "missing-codex-home"
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["init", str(tmp_path), "--preset", "local-codex"],
        env={"CODEX_HOME": str(codex_home)},
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["codex_mcp_install"] == "command"
    assert kwargs["codex_home"] is None


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_auto_codex_mcp_install_keeps_command_for_existing_zaxy_entry(
    mock_run_onboarding: AsyncMock,
    tmp_path: Path,
) -> None:
    """Auto install should not overwrite or error on an existing Codex zaxy server."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        '[mcp_servers.zaxy]\ncommand = "/custom/zaxy"\nargs = ["serve"]\n',
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["init", str(tmp_path), "--preset", "local-codex"],
        env={"CODEX_HOME": str(codex_home)},
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["codex_mcp_install"] == "command"
    assert kwargs["codex_home"] is None
    assert kwargs["codex_mcp_conflict_path"] == codex_home / "config.toml"


@patch("zaxy.cli.runtime.run_onboarding")
def test_init_command_auto_codex_mcp_install_uses_matching_existing_zaxy_entry(
    mock_run_onboarding: AsyncMock,
    tmp_path: Path,
) -> None:
    """Auto install should treat an existing compatible zaxy server as installed."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "\n".join(
            [
                "[mcp_servers.zaxy]",
                'command = "/opt/zaxy/bin/zaxy"',
                'args = ["serve"]',
                "startup_timeout_sec = 90",
                "",
                "[mcp_servers.zaxy.env]",
                'LOG_LEVEL = "ERROR"',
                'ZAXY_ENV = "development"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--preset",
            "local-codex",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
        env={"CODEX_HOME": str(codex_home)},
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["codex_mcp_install"] == "user"
    assert kwargs["codex_home"] is None


def test_init_command_help_describes_full_onboarding_path() -> None:
    """init help should describe the full golden-path onboarding behavior."""
    runner = CliRunner()

    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert "MCP config" in result.output
    assert "infra" in result.output
    assert "hook status" in result.output
    assert "Bare zaxy init uses the local embedded Codex path" in result.output


def test_init_command_json_includes_next_steps_and_capture_summary(tmp_path: Path) -> None:
    """init --json should expose next_steps and capture state for client UIs and automation."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(workspace), "--domain", "demo", "--preset", "local-codex", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "demo-default"
    assert any(step.startswith("Data lives in") for step in payload["next_steps"])
    assert any(step.startswith("Run zaxy hook-status") for step in payload["next_steps"])
    assert payload["capture"]["configured"] is True
    assert payload["capture"]["running"] is False
    assert payload["capture"]["doctor_status"] in {"ok", "warning"}
    assert payload["setup"]["status"] in {"ok", "warning"}
    assert payload["setup"]["counts"]["ok"] >= 1
    assert isinstance(payload["setup"]["issues"], list)
    assert isinstance(payload["setup"]["pending"], list)
    assert payload["readiness"]["status"] in {"ready", "needs_action"}
    assert payload["readiness"]["setup_status"] == payload["status"]
    assert isinstance(payload["readiness"]["reasons"], list)
    assert isinstance(payload["readiness"]["actions"], list)
    assert payload["readiness"]["capture"]["configured"] is True


def test_init_command_json_separates_setup_success_from_readiness_actions(tmp_path: Path) -> None:
    """Machine output should distinguish setup completion from readiness actions."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    codex_home = tmp_path / "missing-codex-home"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(workspace),
            "--domain",
            "demo",
            "--preset",
            "local-codex",
            "--no-agent-instructions",
            "--json",
        ],
        env={"CODEX_HOME": str(codex_home)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["setup"]["status"] == "ok"
    assert payload["setup"]["issues"] == []
    assert any(step["name"] == "mcp_config" and step["status"] == "preview" for step in payload["setup"]["pending"])
    assert payload["readiness"]["status"] == "needs_action"
    assert any("Start or restart Codex through the activation launcher" in action for action in payload["readiness"]["actions"])
    assert not any("Start managed deterministic Codex capture" in action for action in payload["readiness"]["actions"])
    assert not any("agent_instructions" in reason for reason in payload["readiness"]["reasons"])
    assert not any("mcp_config preview" in reason for reason in payload["readiness"]["reasons"])
    assert not any("capture not running" in reason for reason in payload["readiness"]["reasons"])
    assert payload["readiness"]["capture"] == payload["capture"]
    assert any(step.startswith("Smoke test recent memory: zaxy memory log") for step in payload["next_steps"])
    assert payload["capture"]["configured"] is True
    assert payload["capture"]["running"] is False
    assert payload["capture"]["pids"] == []
    assert payload["capture"]["doctor_status"] == "warning"


@patch("zaxy.projection_backends.build_projection_store")
def test_reproject_command_replays_log_into_graph(mock_build_projection_store: MagicMock, tmp_path: Path) -> None:
    """reproject should rebuild graph projections from an Eventloom log."""
    log_path = tmp_path / "default.jsonl"
    log = EventLog(log_path)
    log.append(
        "decision.made",
        actor="assistant",
        payload={
            "decision": "Use structured Eventloom trace.",
            "rationale": ["Supports replayable memory."],
        },
        thread="default",
    )
    store = AsyncMock()
    mock_build_projection_store.return_value = store
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "reproject",
            str(log_path),
            "--session-id",
            "default",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    assert "Reprojected 1 events into session default using neo4j" in result.output
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "neo4j"
    assert config.neo4j_uri == "bolt://test:7687"
    assert config.neo4j_user == "neo4j"
    assert config.neo4j_password == "testpassword"
    store.connect.assert_awaited_once()
    store.init_schema.assert_awaited_once()
    store.upsert_extraction.assert_awaited_once()
    extraction = store.upsert_extraction.await_args.args[0]
    assert extraction.entities[0].entity_type == "decision"
    assert store.upsert_extraction.await_args.kwargs == {"session_id": "default"}
    store.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_reproject_command_can_reset_and_rebuild_pggraph_backend(
    mock_build_projection_store: MagicMock,
    tmp_path: Path,
) -> None:
    """reproject should operationally cover pgGraph bootstrap, reset, and rebuild."""
    log_path = tmp_path / "default.jsonl"
    EventLog(log_path).append(
        "goal.created",
        actor="assistant",
        payload={"title": "Evaluate pgGraph"},
        thread="default",
    )
    store = AsyncMock()
    mock_build_projection_store.return_value = store
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "reproject",
            str(log_path),
            "--session-id",
            "default",
            "--projection-backend",
            "pggraph",
            "--pggraph-dsn",
            "postgresql://postgres:postgres@localhost:5432/zaxy",
            "--reset-projection",
        ],
    )

    assert result.exit_code == 0
    assert "Reprojected 1 events into session default using pggraph" in result.output
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "pggraph"
    assert config.pggraph_dsn == "postgresql://postgres:postgres@localhost:5432/zaxy"
    store.connect.assert_awaited_once()
    store.init_schema.assert_awaited_once()
    store.reset_benchmark_projection.assert_awaited_once()
    store.begin_bulk_projection.assert_awaited_once()
    store.upsert_extraction.assert_awaited_once()
    store.commit_bulk_projection.assert_awaited_once()
    store.rollback_bulk_projection.assert_not_awaited()
    store.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_reproject_command_can_reset_and_rebuild_embedded_backend(
    mock_build_projection_store: MagicMock,
    tmp_path: Path,
) -> None:
    """reproject should operationally cover embedded graph reset and rebuild."""
    log_path = tmp_path / "default.jsonl"
    EventLog(log_path).append(
        "goal.created",
        actor="assistant",
        payload={"title": "Evaluate embedded graph"},
        thread="default",
    )
    store = AsyncMock()
    mock_build_projection_store.return_value = store
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "reproject",
            str(log_path),
            "--session-id",
            "default",
            "--projection-backend",
            "embedded",
            "--reset-projection",
        ],
    )

    assert result.exit_code == 0
    assert "Reprojected 1 events into session default using embedded" in result.output
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path.name == "embedded.kuzu"
    store.connect.assert_awaited_once()
    store.init_schema.assert_awaited_once()
    store.reset_benchmark_projection.assert_awaited_once()
    store.upsert_extraction.assert_awaited_once()
    store.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_reproject_command_uses_repo_local_profile_for_bare_init(
    mock_build_projection_store: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """After bare init, reproject should rebuild the repo-local embedded graph by default."""
    log_path = tmp_path / ".eventloom" / "default.jsonl"
    EventLog(log_path).append(
        "goal.created",
        actor="assistant",
        payload={"title": "Use embedded projection"},
        thread="default",
    )
    embedded_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n"
        "NEO4J_AUTO_START=false\n",
        encoding="utf-8",
    )
    store = AsyncMock()
    mock_build_projection_store.return_value = store
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["reproject", str(log_path), "--session-id", "default"])

    assert result.exit_code == 0
    assert "Reprojected 1 events into session default using embedded" in result.output
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path == embedded_path
    store.connect.assert_awaited_once()
    store.init_schema.assert_awaited_once()
    store.upsert_extraction.assert_awaited_once()
    store.close.assert_awaited_once()


@patch("zaxy.projection_backends.build_projection_store")
def test_reproject_command_uses_profile_next_to_absolute_eventloom_log(
    mock_build_projection_store: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Absolute Eventloom logs should rebuild the projection configured by their repo."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    log_path = workspace / ".eventloom" / "default.jsonl"
    EventLog(log_path).append(
        "goal.created",
        actor="assistant",
        payload={"title": "Use embedded projection"},
        thread="default",
    )
    embedded_path = workspace / ".eventloom" / "projections" / "embedded.kuzu"
    (workspace / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n",
        encoding="utf-8",
    )
    store = AsyncMock()
    mock_build_projection_store.return_value = store
    monkeypatch.chdir(outside)
    runner = CliRunner()

    result = runner.invoke(app, ["reproject", str(log_path), "--session-id", "default"])

    assert result.exit_code == 0
    assert "using embedded" in result.output
    config = mock_build_projection_store.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path == embedded_path


@patch("zaxy.projection_backends.build_projection_store")
def test_reproject_command_closes_pggraph_backend_after_projection_failure(
    mock_build_projection_store: MagicMock,
    tmp_path: Path,
) -> None:
    """reproject failure recovery should close experimental backend resources."""
    log_path = tmp_path / "default.jsonl"
    EventLog(log_path).append(
        "goal.created",
        actor="assistant",
        payload={"title": "Evaluate pgGraph"},
        thread="default",
    )
    store = AsyncMock()
    store.upsert_extraction.side_effect = RuntimeError("pgGraph unavailable")
    mock_build_projection_store.return_value = store
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "reproject",
            str(log_path),
            "--session-id",
            "default",
            "--projection-backend",
            "pggraph",
            "--pggraph-dsn",
            "postgresql://postgres:postgres@localhost:5432/zaxy",
        ],
    )

    assert result.exit_code != 0
    assert "pgGraph unavailable" in str(result.exception)
    store.connect.assert_awaited_once()
    store.init_schema.assert_awaited_once()
    store.close.assert_awaited_once()


def test_compact_audit_reports_identity_safety_without_rewriting_log(tmp_path: Path) -> None:
    """compact --audit should report safety findings and leave the log untouched."""
    log_path = tmp_path / "work.jsonl"
    log = EventLog(log_path)
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/a.md",
            "start_line": 1,
            "end_line": 3,
            "content": "Runbook source records identity-code-0001.",
        },
    )
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/b.md",
            "start_line": 1,
            "end_line": 3,
            "content": "Runbook source records identity-code-0002.",
        },
    )
    before = Path(log_path).read_text(encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["compact", str(log_path), "--audit"])

    assert result.exit_code == 1
    assert "Compaction audit: UNSAFE" in result.output
    assert "Identity recall:" in result.output
    assert "Missing identities:" in result.output
    assert Path(log_path).read_text(encoding="utf-8") == before


def test_compact_audit_json_output(tmp_path: Path) -> None:
    """compact --audit --json should emit machine-readable audit results."""
    log_path = tmp_path / "work.jsonl"
    EventLog(log_path).append("goal.created", actor="user", payload={"title": "Ship"})
    runner = CliRunner()

    result = runner.invoke(app, ["compact", str(log_path), "--audit", "--json"])

    assert result.exit_code == 0
    assert '"safe": true' in result.output
    assert '"identity_recall": 1.0' in result.output


def test_compact_writes_projection_without_rewriting_log(tmp_path: Path) -> None:
    """compact --projection-output should store backpointer projections only."""
    log_path = tmp_path / "work.jsonl"
    projection_path = tmp_path / "work.compaction.json"
    EventLog(log_path).append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/context.md",
            "start_line": 10,
            "end_line": 12,
            "content": "Context note records identity-code-0001.",
        },
    )
    before = log_path.read_text(encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "compact",
            str(log_path),
            "--projection-output",
            str(projection_path),
            "--strategy",
            "medoid",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote compaction projection:" in result.output
    assert projection_path.exists()
    assert '"strategy": "medoid"' in projection_path.read_text(encoding="utf-8")
    assert log_path.read_text(encoding="utf-8") == before


def test_compact_projection_accepts_coordinate_purpose_policy(tmp_path: Path) -> None:
    """compact --purpose coordinate should write authoritative-only Coordinate projections."""
    log_path = tmp_path / "coordinate.jsonl"
    projection_path = tmp_path / "coordinate.compaction.json"
    log = EventLog(log_path)
    log.append(
        "coordination.finding.reported",
        actor="worker",
        payload={
            "mission_id": "mission-1",
            "finding_id": "finding-pending",
            "claim_key": "release.package",
            "claim_value": "pending",
            "coordination_status": "pending",
            "summary": "Pending row should not become compact authority.",
        },
    )
    log.append(
        "coordination.finding.promoted",
        actor="coordinator",
        payload={
            "mission_id": "mission-1",
            "finding_id": "finding-promoted",
            "claim_key": "release.package",
            "claim_value": "ready",
            "coordination_status": "promoted",
            "summary": "Promoted row is compact authority.",
        },
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "compact",
            str(log_path),
            "--projection-output",
            str(projection_path),
            "--purpose",
            "coordinate",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(projection_path.read_text(encoding="utf-8"))
    assert payload["purpose"]["profile"] == "coordinate"
    assert payload["strategy"] == "coordinate_authoritative"
    assert payload["records"][0]["kind"] == "coordinate_authoritative"
    assert payload["records"][0]["authority_scope"] == "authoritative"
    assert payload["consolidation_policy"]["diagnostic_event_seqs"] == [1]
    assert payload["consolidation_policy"]["authoritative_event_seqs"] == [2]


def test_compact_rewrite_appends_lifecycle_event(tmp_path: Path) -> None:
    """compact rewrite should record a compaction.completed lifecycle event."""
    log_path = tmp_path / "work.jsonl"
    EventLog(log_path).append("goal.created", actor="user", payload={"title": "Ship"})
    runner = CliRunner()

    result = runner.invoke(app, ["compact", str(log_path)])

    assert result.exit_code == 0
    events = EventLog(log_path).read_all()
    assert events[-1].type == "compaction.completed"
    assert events[-1].payload["mode"] == "rewrite"
    assert events[-1].payload["status"] == "succeeded"
    assert events[-1].payload["event_count"] == 1


def test_viewer_command_writes_static_html(tmp_path: Path) -> None:
    """viewer should write a standalone Eventloom inspection page."""
    log_path = tmp_path / "default.jsonl"
    output = tmp_path / "viewer.html"
    EventLog(log_path).append(
        "session.genesis",
        actor="zaxy",
        payload={"session_id": "default", "workspace_type": "codebase"},
        thread="default",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["viewer", str(log_path), "--output", str(output)])

    assert result.exit_code == 0
    assert f"Wrote Eventloom viewer: {output}" in result.output
    assert output.exists()
    assert "Eventloom Session Viewer" in output.read_text(encoding="utf-8")


def test_dashboard_cli_help_exposes_localhost_default() -> None:
    """dashboard should expose the local read-only web app command."""
    runner = CliRunner()
    command = get_command(app).commands["dashboard"]
    options = {option: parameter for parameter in command.params for option in parameter.opts}
    result = runner.invoke(app, ["dashboard", "--help"])

    assert result.exit_code == 0
    assert "dashboard" in result.output
    assert options["--host"].default == "127.0.0.1"
    assert options["--port"].default == 8765
    assert options["--projection-backend"].default is None
    assert options["--projection-backend"].help == (
        "Projection backend for graph visualization: embedded, neo4j, pggraph, or latticedb"
    )
    assert options["--pggraph-dsn"].help == "pgGraph/PostgreSQL DSN for graph visualization"
    assert options["--embedded-graph-path"].help == "Embedded graph projection path for graph visualization"


def test_status_cli_help_lists_all_projection_backends() -> None:
    """status help should match the supported projection backend registry."""
    command = get_command(app).commands["status"]
    options = {option: parameter for parameter in command.params for option in parameter.opts}

    assert options["--projection-backend"].default is None
    assert options["--projection-backend"].help == (
        "Projection backend to check: embedded, neo4j, pggraph, or latticedb"
    )


@patch("zaxy.dashboard.run_dashboard")
def test_dashboard_command_uses_repo_local_profile_for_bare_init(
    mock_run_dashboard: MagicMock,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """After bare init, dashboard should inspect the embedded graph profile by default."""
    embedded_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        f"EMBEDDED_GRAPH_PATH={embedded_path}\n"
        "NEO4J_AUTO_START=false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["dashboard"])

    assert result.exit_code == 0
    scope = mock_run_dashboard.call_args.args[0]
    assert scope.projection_backend == "embedded"
    assert scope.embedded_graph_path == embedded_path
    assert "Zaxy dashboard listening on http://127.0.0.1:8765" in result.output
