"""Tests for the experimental associative pattern-completion projection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.associative_memory import (
    AssociativeProjection,
    StateRecoveryMetrics,
    build_pattern_completion_workload,
    build_state_recovery_workload,
    evaluate_state_recovery_guardrails,
    load_pattern_completion_workload,
    load_state_recovery_workload,
    run_pattern_completion_benchmark,
    run_state_recovery_benchmark,
)
from zaxy.event import EventLog


def test_pattern_completion_workload_is_frozen_and_replayable(tmp_path: Path) -> None:
    workload_path = tmp_path / "pattern-completion-workload.json"

    workload = build_pattern_completion_workload(workload_path)
    reloaded = load_pattern_completion_workload(workload_path)

    assert reloaded.fingerprint == workload.fingerprint
    assert reloaded.version == "pattern-completion-v0"
    assert len(reloaded.cases) == 2
    assert {
        case.gold.latent_state
        for case in reloaded.cases
    } == {"expired-jwks-cache", "cited-event-provenance-required"}


def test_pattern_completion_workload_rejects_fingerprint_drift(tmp_path: Path) -> None:
    workload_path = tmp_path / "pattern-completion-workload.json"
    workload = build_pattern_completion_workload(workload_path)
    payload = workload.to_dict()
    payload["cases"][0]["query"] = "mutated query"
    workload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        load_pattern_completion_workload(workload_path)


def test_associative_projection_recovers_latent_cause_from_partial_cue(tmp_path: Path) -> None:
    workload = build_pattern_completion_workload(tmp_path / "workload.json")
    case = workload.cases[0]
    eventlog = EventLog(tmp_path / "events.jsonl")
    for event in case.events:
        eventlog.append(
            str(event["type"]),
            actor=str(event.get("actor", "benchmark")),
            thread=str(event.get("thread", case.case_id)),
            payload=dict(event["payload"]),
        )
    replay = eventlog.replay()
    projection = AssociativeProjection.from_events(replay.events)

    direct_event_ids = projection.direct_event_ids(case.query, top_k=1)
    candidate = projection.complete(case.query, top_k=3, seed_k=1, propagation_k=3, iterations=2)[0]
    candidate_event_ids = {f"{ref.thread}:{ref.seq}" for ref in candidate.evidence}

    assert direct_event_ids == ("pcase-auth-hidden-cause:1",)
    assert "pcase-auth-hidden-cause:2" in candidate_event_ids
    assert "expired-jwks-cache" in candidate.support_terms
    assert all(ref.hash for ref in candidate.evidence)


def test_pattern_completion_benchmark_scores_associative_gain(tmp_path: Path) -> None:
    report = run_pattern_completion_benchmark(tmp_path)

    assert report.metrics.latent_state_recall == 1.0
    assert report.metrics.evidence_recall > report.baselines["direct_lexical"].evidence_recall
    assert report.metrics.citation_coverage == 1.0
    assert report.baselines["direct_lexical"].latent_state_recall < report.metrics.latent_state_recall
    assert (tmp_path / "pattern-completion-benchmark.json").exists()
    assert (tmp_path / "pattern-completion-benchmark.md").exists()


def test_experimental_pattern_completion_cli_writes_report(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["experimental", "pattern-completion", "--output-dir", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["version"] == "pattern-completion-v0"
    assert payload["metrics"]["latent_state_recall"] == 1.0
    assert payload["baselines"]["direct_lexical"]["latent_state_recall"] < 1.0


def test_state_recovery_workload_is_frozen_and_has_adversarial_cases(tmp_path: Path) -> None:
    workload_path = tmp_path / "state-recovery-workload.json"

    workload = build_state_recovery_workload(workload_path)
    reloaded = load_state_recovery_workload(workload_path)

    assert reloaded.fingerprint == workload.fingerprint
    assert reloaded.version == "state-recovery-v0"
    assert len(reloaded.cases) == 33
    assert all(case.gold.minimal_evidence_event_ids for case in reloaded.cases if not case.gold.should_abstain)
    assert any(case.gold.should_abstain for case in reloaded.cases)
    assert any(case.gold.stale_event_ids for case in reloaded.cases)
    assert all(case.gold.distractor_event_ids for case in reloaded.cases)


def test_state_recovery_workload_rejects_fingerprint_drift(tmp_path: Path) -> None:
    workload_path = tmp_path / "state-recovery-workload.json"
    workload = build_state_recovery_workload(workload_path)
    payload = workload.to_dict()
    payload["cases"][0]["gold"]["latent_state"] = "mutated"
    workload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        load_state_recovery_workload(workload_path)


def test_state_recovery_benchmark_reports_required_metrics_and_baselines(tmp_path: Path) -> None:
    report = run_state_recovery_benchmark(tmp_path)

    assert report.schema_version == "state-recovery-report-v1"
    assert report.case_count == 33
    assert report.production_baseline == "memory_fabric_checkout"
    assert report.status == "pass"
    assert report.baseline_names == (
        "direct_lexical",
        "hash_vector",
        "graph_traversal",
        "zaxy_core_proxy",
        "memory_fabric_checkout",
        "associative_projection",
        "authority_resolved_associative",
    )
    assert set(report.checks) == {
        "state_accuracy",
        "minimal_evidence_recall",
        "stale_rejection",
        "distractor_resistance",
        "abstention_accuracy",
        "citation_coverage",
    }
    assert all(check["baseline"] == "memory_fabric_checkout" for check in report.checks.values())
    assert all(check["status"] == "pass" for check in report.checks.values())
    assert set(report.baselines) == {
        "direct_lexical",
            "hash_vector",
            "graph_traversal",
            "zaxy_core_proxy",
            "memory_fabric_checkout",
            "associative_projection",
            "authority_resolved_associative",
        }
    for metrics in report.baselines.values():
        payload = metrics.to_dict()
        assert set(payload) == {
            "state_accuracy",
            "minimal_evidence_recall",
            "stale_rejection",
            "distractor_resistance",
            "token_cost",
            "latency_ms",
            "citation_coverage",
            "abstention_accuracy",
        }
    assert report.baselines["associative_projection"].state_accuracy >= report.baselines["direct_lexical"].state_accuracy
    assert report.baselines["associative_projection"].minimal_evidence_recall >= report.baselines["direct_lexical"].minimal_evidence_recall
    assert report.baselines["memory_fabric_checkout"].citation_coverage == 1.0
    assert report.baselines["memory_fabric_checkout"].minimal_evidence_recall > 0.0
    assert report.baselines["memory_fabric_checkout"].stale_rejection > report.baselines["zaxy_core_proxy"].stale_rejection
    assert report.baselines["memory_fabric_checkout"].abstention_accuracy == 1.0
    assert (
        report.baselines["authority_resolved_associative"].distractor_resistance
        > report.baselines["associative_projection"].distractor_resistance
    )
    assert (
        report.baselines["authority_resolved_associative"].stale_rejection
        > report.baselines["associative_projection"].stale_rejection
    )
    assert report.baselines["authority_resolved_associative"].abstention_accuracy > report.baselines["associative_projection"].abstention_accuracy
    assert report.baselines["authority_resolved_associative"].token_cost < report.baselines["associative_projection"].token_cost
    assert report.baselines["associative_projection"].citation_coverage == 1.0
    assert report.baselines["authority_resolved_associative"].citation_coverage == 1.0
    assert (tmp_path / "state-recovery-benchmark.json").exists()
    assert (tmp_path / "state-recovery-benchmark.md").exists()
    payload = json.loads((tmp_path / "state-recovery-benchmark.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "state-recovery-report-v1"
    assert payload["case_count"] == 33
    assert payload["production_baseline"] == "memory_fabric_checkout"
    assert payload["status"] == "pass"
    markdown = (tmp_path / "state-recovery-benchmark.md").read_text(encoding="utf-8")
    assert "StateRecoveryBench is an official Zaxy benchmark lane" in markdown
    assert "Associative projection rows remain diagnostic research baselines" in markdown


def test_state_recovery_guardrails_fail_missing_or_weak_production_baseline() -> None:
    missing = evaluate_state_recovery_guardrails({})

    assert set(missing) == {
        "state_accuracy",
        "minimal_evidence_recall",
        "stale_rejection",
        "distractor_resistance",
        "abstention_accuracy",
        "citation_coverage",
    }
    assert all(check["status"] == "fail" for check in missing.values())
    assert all(check["reason"] == "missing production baseline" for check in missing.values())

    weak = evaluate_state_recovery_guardrails(
        {
            "memory_fabric_checkout": StateRecoveryMetrics(
                state_accuracy=0.80,
                minimal_evidence_recall=0.89,
                stale_rejection=1.0,
                distractor_resistance=0.79,
                token_cost=40,
                latency_ms=1.0,
                citation_coverage=1.0,
                abstention_accuracy=1.0,
            )
        }
    )

    assert weak["state_accuracy"]["status"] == "fail"
    assert weak["minimal_evidence_recall"]["status"] == "fail"
    assert weak["distractor_resistance"]["status"] == "fail"
    assert weak["stale_rejection"]["status"] == "pass"


def test_state_recovery_cli_prints_scores(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["experimental", "state-recovery", "--output-dir", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["version"] == "state-recovery-v0"
    assert "associative_projection" in payload["baselines"]
    assert payload["baselines"]["associative_projection"]["citation_coverage"] == 1.0


def test_state_recovery_benchmark_cli_is_official_guardrail_lane(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["state-recovery-benchmark", "--output-dir", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "state-recovery-report-v1"
    assert payload["production_baseline"] == "memory_fabric_checkout"
    assert payload["status"] == "pass"
    assert payload["baseline_names"][4] == "memory_fabric_checkout"
