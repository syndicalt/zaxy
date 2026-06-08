"""Tests for the 2.0 RC.1 benchmark-freeze evidence gate."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.rc_benchmark_freeze import (
    build_rc1_benchmark_freeze_report,
    format_rc1_benchmark_freeze_report,
)


def test_rc1_benchmark_freeze_passes_with_required_evidence(tmp_path: Path) -> None:
    _write_required_artifacts(tmp_path)

    report = build_rc1_benchmark_freeze_report(tmp_path)

    assert report.passed is True
    assert report.release == "2.0.0-rc.1"
    assert report.headline_500["manifest"] == "reports/benchmarks/2.0.0-rc.1/manifest.json"
    assert report.headline_500["claim_scope"] == "longmemeval_compatible_checkout"
    assert report.harvey_lab["claim_scope"] == "external_anchor"
    assert report.project_benchmarks["state_recovery"]["claim_scope"] == "project_defined_internal"
    assert report.project_benchmarks["coordination"]["claim_scope"] == "project_defined_internal"
    assert report.project_benchmarks["purpose"]["claim_scope"] == "project_defined_internal"
    assert {lane["lane"] for lane in report.internal_lanes} == {
        "causal",
        "consolidation",
        "procedural",
        "metacognition",
    }
    assert all(lane["claim_scope"] == "project_defined_internal" for lane in report.internal_lanes)
    assert all(check["passed"] is True for check in report.checks)

    rendered = format_rc1_benchmark_freeze_report(report)
    assert "# Zaxy 2.0 RC.1 Benchmark Freeze" in rendered
    assert "- Status: `PASS`" in rendered
    assert "`rc1_manifest`" in rendered


def test_rc1_benchmark_freeze_fails_when_headline_run_config_missing(tmp_path: Path) -> None:
    _write_required_artifacts(tmp_path)
    (
        tmp_path
        / "reports/benchmarks/longmemeval-500-publish-20260607/run-config.md"
    ).unlink()

    report = build_rc1_benchmark_freeze_report(tmp_path)

    assert report.passed is False
    assert any(
        check["name"] == "headline_run_config" and check["passed"] is False
        for check in report.checks
    )


def test_rc1_benchmark_freeze_fails_when_internal_lane_claims_external_scope(
    tmp_path: Path,
) -> None:
    _write_required_artifacts(tmp_path)

    report = build_rc1_benchmark_freeze_report(
        tmp_path,
        internal_lane_overrides={
            "causal": {
                "lane": "causal",
                "claim_scope": "external_validation",
                "score": 1.0,
                "citation_coverage": 1.0,
            }
        },
    )

    assert report.passed is False
    assert any(
        check["name"] == "internal_lane_scope:causal" and check["passed"] is False
        for check in report.checks
    )


def test_rc1_benchmark_freeze_fails_when_harvey_ready_artifact_is_incomplete(
    tmp_path: Path,
) -> None:
    _write_required_artifacts(tmp_path)
    ready_path = tmp_path / "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-ready.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["blocking_reasons"] = ["missing_normalized_results"]
    ready_path.write_text(json.dumps(ready), encoding="utf-8")

    report = build_rc1_benchmark_freeze_report(tmp_path)

    assert report.passed is False
    assert any(
        check["name"] == "harvey_ready_artifact" and check["passed"] is False
        for check in report.checks
    )


def test_rc1_benchmark_freeze_fails_when_harvey_benchmark_is_summary_only(
    tmp_path: Path,
) -> None:
    _write_required_artifacts(tmp_path)
    benchmark_path = tmp_path / "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark.pop("result_provenance")
    benchmark.pop("task_rows")
    benchmark.pop("zaxy_results")
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

    report = build_rc1_benchmark_freeze_report(tmp_path)

    assert report.passed is False
    assert any(
        check["name"] == "harvey_zaxy_result_rows" and check["passed"] is False
        for check in report.checks
    )
    assert any(
        check["name"] == "harvey_result_provenance" and check["passed"] is False
        for check in report.checks
    )


def test_rc1_benchmark_freeze_fails_on_malformed_manifest_json(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "reports/benchmarks/2.0.0-rc.1"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text("{not-json", encoding="utf-8")

    report = build_rc1_benchmark_freeze_report(tmp_path)

    assert report.passed is False
    assert any(check["name"] == "rc1_manifest_json" for check in report.checks)
    assert report.headline_500["artifact"] == (
        "reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json"
    )


def test_rc1_benchmark_freeze_fails_on_non_object_json_artifacts(tmp_path: Path) -> None:
    _write_required_artifacts(tmp_path)
    headline = tmp_path / "reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json"
    headline.write_text("[]", encoding="utf-8")
    harvey = tmp_path / "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-status.json"
    harvey.write_text("[]", encoding="utf-8")

    report = build_rc1_benchmark_freeze_report(tmp_path)

    assert report.passed is False
    assert any(check["name"] == "headline_report_json" for check in report.checks)
    assert any(check["name"] == "harvey_json:harvey-lab-status.json" for check in report.checks)


def test_rc1_benchmark_freeze_fails_when_headline_backend_is_missing(tmp_path: Path) -> None:
    _write_required_artifacts(tmp_path)
    headline = tmp_path / "reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json"
    payload = json.loads(headline.read_text(encoding="utf-8"))
    payload["summaries"] = [{"backend": "bm25", "mean_score": 0.1}]
    headline.write_text(json.dumps(payload), encoding="utf-8")

    report = build_rc1_benchmark_freeze_report(tmp_path)

    assert report.passed is False
    assert any(
        check["name"] == "headline_backend" and check["passed"] is False
        for check in report.checks
    )


def test_rc1_benchmark_freeze_fails_malformed_harvey_result_rows(tmp_path: Path) -> None:
    _write_required_artifacts(tmp_path)
    benchmark_path = tmp_path / "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["zaxy_results"][0]["memory_read_calls"] = 0
    benchmark["task_rows"]["task-0"]["zaxy_memory_search_calls"] = True
    benchmark["result_provenance"]["external_baseline_reports"][0]["framework_count"] = 0
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

    report = build_rc1_benchmark_freeze_report(tmp_path)

    assert report.passed is False
    assert any(
        check["name"] == "harvey_zaxy_result_rows" and check["passed"] is False
        for check in report.checks
    )
    assert any(
        check["name"] == "harvey_task_rows" and check["passed"] is False
        for check in report.checks
    )
    assert any(
        check["name"] == "harvey_result_provenance" and check["passed"] is False
        for check in report.checks
    )


def _write_required_artifacts(root: Path) -> None:
    _write_rc1_manifest(root)
    _write_project_benchmark_artifacts(root)
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


def _write_rc1_manifest(root: Path) -> None:
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
                    {
                        "lane": "causal",
                        "claim_scope": "project_defined_internal",
                        "module": "zaxy.causal_benchmark",
                    },
                    {
                        "lane": "consolidation",
                        "claim_scope": "project_defined_internal",
                        "module": "zaxy.consolidation_benchmark",
                    },
                    {
                        "lane": "procedural",
                        "claim_scope": "project_defined_internal",
                        "module": "zaxy.reasoning_benchmark",
                    },
                    {
                        "lane": "metacognition",
                        "claim_scope": "project_defined_internal",
                        "module": "zaxy.reasoning_benchmark",
                    },
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


def _write_project_benchmark_artifacts(root: Path) -> None:
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
