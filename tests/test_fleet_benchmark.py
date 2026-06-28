"""Tests for the FleetBench scaling scaffold (zaxy_benchmarks.fleet_benchmark).

These exercise REAL CoordinationBench runs (a real ``CoordinationManager``
mission per scale point). Scored axes are exact and deterministic; ``latency_ms``
is real wall-clock and is asserted to be a positive float, never a fixed value.
"""

from __future__ import annotations

from pathlib import Path

from zaxy_benchmarks.coordination_benchmark import (
    _run_case,
    build_coordination_workload,
    flat_eventlog_baseline_metrics,
)
from zaxy_benchmarks.fleet_benchmark import (
    CROSS_AGENT_TRANSFER_NOTE,
    CROSS_AGENT_TRANSFER_SCOPE,
    FLEET_WORKLOAD_VERSION,
    FleetBenchReport,
    _governance_correctness,
    _governance_ok,
    run_fleet_benchmark,
)

WORKER_COUNTS = (3, 4)


def _run(base: Path, name: str) -> FleetBenchReport:
    return run_fleet_benchmark(base / name, worker_counts=WORKER_COUNTS)


def test_one_result_per_scale_point_with_real_axes(tmp_path: Path) -> None:
    report = _run(tmp_path, "run")
    assert report.version == FLEET_WORKLOAD_VERSION
    assert [metrics.worker_count for metrics in report.results] == list(WORKER_COUNTS)
    assert len(report.results) == len(WORKER_COUNTS)
    for metrics in report.results:
        assert metrics.cross_agent_transfer_scope == CROSS_AGENT_TRANSFER_SCOPE
        assert metrics.mission_count == 1
        assert 0.0 <= metrics.coordination_quality <= 1.0
        assert 0.0 <= metrics.governance_correctness <= 1.0
        assert 0.0 <= metrics.cross_agent_transfer <= 1.0
        assert metrics.returned_tokens > 0
        assert metrics.injected_tokens > 0
        assert 0.0 <= metrics.token_efficiency <= 1.0


def test_scored_axes_are_exact_and_deterministic_across_runs(tmp_path: Path) -> None:
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")
    # Quality and governance are exact-scored and reproducible.
    for left, right in zip(first.results, second.results, strict=True):
        assert left.coordination_quality == right.coordination_quality
        assert left.governance_correctness == right.governance_correctness
        assert left.cross_agent_transfer == right.cross_agent_transfer
        assert left.returned_tokens == right.returned_tokens
        assert left.injected_tokens == right.injected_tokens
        assert left.token_efficiency == right.token_efficiency
        assert left.scored_dict() == right.scored_dict()
    assert first.mean_metrics.scored_dict() == second.mean_metrics.scored_dict()


def test_fingerprint_excludes_latency(tmp_path: Path) -> None:
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")
    # Identical fingerprint across two runs proves latency (the only
    # non-deterministic field) is excluded from the scored fingerprint.
    assert first.fingerprint == second.fingerprint
    assert "latency_ms" not in first.results[0].scored_dict()
    for metrics in (*first.results, first.mean_metrics):
        assert isinstance(metrics.latency_ms, float)
        assert metrics.latency_ms > 0.0  # real wall-clock, positive, not fixed


def test_token_fields_consistent_and_scale_with_worker_count(tmp_path: Path) -> None:
    report = _run(tmp_path, "run")
    for metrics in report.results:
        # token_efficiency is exactly 1 - injected/returned over the governed brief.
        expected = round(1.0 - (metrics.injected_tokens / metrics.returned_tokens), 6)
        assert metrics.token_efficiency == expected
        assert metrics.injected_tokens <= metrics.returned_tokens
    # The scaling axis is real: more workers -> more raw worker-log tokens, and a
    # bounded governed brief -> token_efficiency improves with fleet size.
    returned = [metrics.returned_tokens for metrics in report.results]
    efficiency = [metrics.token_efficiency for metrics in report.results]
    assert returned == sorted(returned)
    assert returned[-1] > returned[0]
    assert efficiency == sorted(efficiency)


def test_clean_run_governance_is_perfect(tmp_path: Path) -> None:
    report = _run(tmp_path, "run")
    for metrics in report.results:
        assert metrics.governance_correctness == 1.0
    assert report.mean_metrics.governance_correctness == 1.0


def test_governance_drops_with_non_authoritative_leak(tmp_path: Path) -> None:
    # Build the frozen workload and a real clean run.
    workload = build_coordination_workload(tmp_path / "wl.json", workers=3)
    case = workload.cases[0]
    clean = _run_case(tmp_path / "clean", case).metrics
    # The flat-eventlog baseline (existing scoring input) injects raw worker
    # findings wholesale: it leaks a non-authoritative row and drops citations.
    leaky = flat_eventlog_baseline_metrics(case)
    assert leaky.non_authoritative_leakage == 0.0
    assert _governance_ok(clean) is True
    assert _governance_ok(leaky) is False
    # The gate is wired to the existing leakage/citation/replay signals.
    assert _governance_correctness([clean]) == 1.0
    assert _governance_correctness([leaky]) == 0.0
    assert _governance_correctness([clean, leaky]) < 1.0


def test_cross_agent_transfer_is_labeled_within_mission_proxy(tmp_path: Path) -> None:
    report = _run(tmp_path, "run")
    payload = report.to_dict()
    assert payload["cross_agent_transfer_scope"] == CROSS_AGENT_TRANSFER_SCOPE
    for metrics_payload in payload["results"]:
        assert metrics_payload["cross_agent_transfer_scope"] == CROSS_AGENT_TRANSFER_SCOPE
        assert metrics_payload["scaffold"]["cross_agent_transfer"] == CROSS_AGENT_TRANSFER_NOTE
    assert payload["mean_metrics"]["scaffold"]["cross_agent_transfer"] == CROSS_AGENT_TRANSFER_NOTE
    notes = " ".join(payload["scaffold_notes"])
    assert "within_mission_proxy" in notes
    assert "I7" in notes
    assert any("proxy" in note.lower() for note in payload["scaffold_notes"])
    assert any("real" in note.lower() for note in payload["scaffold_notes"])


def test_markdown_has_scale_table_and_scaffold_caveats(tmp_path: Path) -> None:
    out = tmp_path / "run"
    run_fleet_benchmark(out, worker_counts=WORKER_COUNTS)
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "## Scaling axes" in markdown
    assert "| worker_count |" in markdown
    for worker_count in WORKER_COUNTS:
        assert f"| {worker_count} |" in markdown
    assert "## Scaffold caveats" in markdown
    assert "within_mission_proxy" in markdown
    assert "pending I7" in markdown
    assert (out / "report.json").exists()
