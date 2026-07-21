"""Tests for FleetBench (zaxy_benchmarks.fleet_benchmark).

These exercise REAL runs: a real ``CoordinationManager`` mission per scale point
and real ``FleetManager`` propagation scored through real ``checkout_memory``
calls. Scored axes are exact and deterministic; ``latency_ms`` is real wall-clock
and is asserted to be a positive float, never a fixed value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zaxy.fleet import FleetManager
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
    _measure_fleet_transfer,
    build_fleet_scaling_workload,
    run_fleet_benchmark,
)

WORKER_COUNTS = (3, 4)


def _run(base: Path, name: str) -> FleetBenchReport:
    return run_fleet_benchmark(base / name, worker_counts=WORKER_COUNTS)


def _mission(tmp_path: Path, workers: int = 3) -> tuple[Path, list[Any]]:
    """Run one real mission at ``workers`` scale and return its dir and case results."""
    workload = build_fleet_scaling_workload(tmp_path / "wl.json", workers=workers)
    scale_dir = tmp_path / f"scale-w{workers}"
    return scale_dir, [_run_case(scale_dir, case) for case in workload.cases]


# ---------------------------------------------------------------------------
# Scale-point shape, determinism, tokens, governance
# ---------------------------------------------------------------------------


def test_one_result_per_scale_point_with_real_axes(tmp_path: Path) -> None:
    """Every scale point yields one row whose axes are all in range and real."""
    report = _run(tmp_path, "run")
    assert report.version == FLEET_WORKLOAD_VERSION
    assert [metrics.worker_count for metrics in report.results] == list(WORKER_COUNTS)
    for metrics in report.results:
        assert metrics.cross_agent_transfer_scope == CROSS_AGENT_TRANSFER_SCOPE
        assert metrics.mission_count == 1
        assert 0.0 <= metrics.coordination_quality <= 1.0
        assert 0.0 <= metrics.governance_correctness <= 1.0
        assert 0.0 <= metrics.cross_agent_transfer <= 1.0
        assert 0.0 <= metrics.cross_agent_transfer_control <= 1.0
        assert metrics.returned_tokens > 0
        assert metrics.injected_tokens > 0


def test_scored_axes_are_exact_and_deterministic_across_runs(tmp_path: Path) -> None:
    """Two independent runs produce byte-identical scored fields."""
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")
    for left, right in zip(first.results, second.results, strict=True):
        assert left.scored_dict() == right.scored_dict()
    assert first.mean_metrics.scored_dict() == second.mean_metrics.scored_dict()


def test_fingerprint_excludes_latency(tmp_path: Path) -> None:
    """The fingerprint is stable across runs because wall-clock latency is excluded."""
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")
    assert first.fingerprint == second.fingerprint
    assert "latency_ms" not in first.results[0].scored_dict()
    for metrics in (*first.results, first.mean_metrics):
        assert isinstance(metrics.latency_ms, float)
        assert metrics.latency_ms > 0.0  # real wall-clock, positive, not fixed


def test_token_fields_are_consistent_and_returned_grows_with_workers(tmp_path: Path) -> None:
    """token_efficiency is exactly 1-injected/returned and raw logs grow with worker count."""
    report = _run(tmp_path, "run")
    for metrics in report.results:
        expected = round(1.0 - (metrics.injected_tokens / metrics.returned_tokens), 6)
        assert metrics.token_efficiency == expected
        assert metrics.injected_tokens <= metrics.returned_tokens
    returned = [metrics.returned_tokens for metrics in report.results]
    assert returned == sorted(returned)
    assert returned[-1] > returned[0]


def test_clean_run_governance_is_perfect(tmp_path: Path) -> None:
    """A clean governed run passes the governance gate at every scale point."""
    report = _run(tmp_path, "run")
    for metrics in report.results:
        assert metrics.governance_correctness == 1.0


def test_governance_drops_with_non_authoritative_leak(tmp_path: Path) -> None:
    """The governance gate fails a baseline that leaks a non-authoritative row."""
    workload = build_coordination_workload(tmp_path / "wl.json", workers=3)
    case = workload.cases[0]
    clean = _run_case(tmp_path / "clean", case).metrics
    leaky = flat_eventlog_baseline_metrics(case)
    assert leaky.non_authoritative_leakage == 0.0
    assert _governance_ok(clean) is True
    assert _governance_ok(leaky) is False
    assert _governance_correctness([clean]) == 1.0
    assert _governance_correctness([leaky]) == 0.0


# ---------------------------------------------------------------------------
# Real fleet-wide cross-agent transfer
# ---------------------------------------------------------------------------


def test_cross_agent_transfer_is_labeled_fleet_wide_not_a_proxy(tmp_path: Path) -> None:
    """The retired within_mission_proxy label is gone and the scope reads fleet_wide."""
    report = _run(tmp_path, "run")
    payload = report.to_dict()
    assert payload["cross_agent_transfer_scope"] == "fleet_wide"
    assert "within_mission_proxy" not in report.to_dict().__str__()
    for metrics_payload in payload["results"]:
        assert metrics_payload["cross_agent_transfer_scope"] == "fleet_wide"
    notes = " ".join(payload["notes"])
    assert "fleet_wide" in notes
    assert "negative control" in notes
    assert "fleet_wide" in CROSS_AGENT_TRANSFER_NOTE


def test_transfer_measures_real_delivery_to_every_enrolled_agent(tmp_path: Path) -> None:
    """Enrolled receivers really receive the origin agent's propagated memory."""
    scale_dir, results = _mission(tmp_path)
    outcome = _measure_fleet_transfer(scale_dir, results, worker_count=3)
    assert outcome.promotion_count > 0  # something really propagated
    assert outcome.receiver_count == 2
    assert outcome.transfer == 1.0


def test_negative_control_discriminates_enrolled_agent_from_stranger(tmp_path: Path) -> None:
    """Decisive: the never-enrolled stranger receives nothing while receivers receive all."""
    scale_dir, results = _mission(tmp_path)
    outcome = _measure_fleet_transfer(scale_dir, results, worker_count=3)
    assert outcome.control == 1.0  # stranger got nothing
    assert outcome.transfer > outcome.control - 1.0  # enrolled agents did get memory
    assert outcome.transfer == 1.0


def test_transfer_collapses_when_receivers_are_not_enrolled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-reward-hacking: breaking enrollment drops transfer to 0, so the metric is real."""
    real_enroll = FleetManager.enroll_agent

    def only_origin(self: FleetManager, fleet_id: str, agent_id: str, **kwargs: Any) -> Any:
        # Enrol nobody but the origin: if the metric still scored 1.0 it would be
        # measuring the propagation call rather than delivery to other agents.
        if agent_id.endswith("-000"):
            return real_enroll(self, fleet_id, agent_id, **kwargs)
        return None

    monkeypatch.setattr(FleetManager, "enroll_agent", only_origin)
    scale_dir, results = _mission(tmp_path)
    outcome = _measure_fleet_transfer(scale_dir, results, worker_count=3)
    assert outcome.promotion_count > 0  # propagation still happened
    assert outcome.transfer == 0.0  # but nothing reached the un-enrolled receivers


def test_control_fails_when_the_stranger_is_enrolled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control is not a constant: enrolling the stranger drives it to 0."""
    real_enroll = FleetManager.enroll_agent

    def also_stranger(self: FleetManager, fleet_id: str, agent_id: str, **kwargs: Any) -> Any:
        result = real_enroll(self, fleet_id, agent_id, **kwargs)
        if agent_id.endswith("-000"):
            real_enroll(self, fleet_id, "fleet-stranger", **kwargs)
        return result

    monkeypatch.setattr(FleetManager, "enroll_agent", also_stranger)
    scale_dir, results = _mission(tmp_path)
    outcome = _measure_fleet_transfer(scale_dir, results, worker_count=3)
    assert outcome.control == 0.0  # the stranger received fleet memory
    assert outcome.transfer == 1.0  # while genuine delivery still worked


def test_transfer_is_zero_without_receivers(tmp_path: Path) -> None:
    """A fleet with no other agents scores 0 transfer rather than a vacuous 1.0."""
    scale_dir, results = _mission(tmp_path)
    outcome = _measure_fleet_transfer(scale_dir, results, worker_count=1)
    assert outcome.transfer == 0.0


# ---------------------------------------------------------------------------
# Scaling workload
# ---------------------------------------------------------------------------


def test_scaling_workload_gives_every_added_worker_real_findings(tmp_path: Path) -> None:
    """Unlike the padded coordination case, no worker is empty-finding filler."""
    workload = build_fleet_scaling_workload(tmp_path / "wl.json", workers=8)
    case = workload.cases[0]
    assert len(case.workers) == 8
    for worker in case.workers:
        assert worker["findings"], f"{worker['worker_id']} is filler"


def test_scaling_workload_pressure_grows_with_worker_count(tmp_path: Path) -> None:
    """Conflicts, duplicates and accepted claims all grow as workers are added."""
    small = build_fleet_scaling_workload(tmp_path / "s.json", workers=3).cases[0]
    large = build_fleet_scaling_workload(tmp_path / "l.json", workers=8).cases[0]
    assert len(large.gold.expected_conflict_pairs) > len(small.gold.expected_conflict_pairs)
    assert len(large.gold.expected_duplicate_groups) > len(small.gold.expected_duplicate_groups)
    assert len(large.gold.expected_accepted_claims) > len(small.gold.expected_accepted_claims)
    assert len(large.gold.expected_missing_evidence) > len(small.gold.expected_missing_evidence)


def test_scaling_metrics_actually_move_with_worker_count(tmp_path: Path) -> None:
    """The scaling axes vary across scale points; the old workload held them constant."""
    report = run_fleet_benchmark(tmp_path / "run", worker_counts=(3, 5))
    quality = [metrics.coordination_quality for metrics in report.results]
    injected = [metrics.injected_tokens for metrics in report.results]
    receivers = [metrics.fleet_receiver_count for metrics in report.results]
    assert len(set(quality)) > 1
    assert len(set(injected)) > 1
    assert receivers == [2, 4]


def test_scaling_workload_is_deterministic_and_fingerprinted(tmp_path: Path) -> None:
    """The same worker count yields the same workload fingerprint every time."""
    first = build_fleet_scaling_workload(tmp_path / "a.json", workers=5)
    second = build_fleet_scaling_workload(tmp_path / "b.json", workers=5)
    other = build_fleet_scaling_workload(tmp_path / "c.json", workers=6)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != other.fingerprint


def test_scaling_workload_rejects_unsupported_shapes(tmp_path: Path) -> None:
    """Worker counts outside 3..10 and multi-mission requests are rejected."""
    with pytest.raises(ValueError):
        build_fleet_scaling_workload(tmp_path / "x.json", workers=2)
    with pytest.raises(ValueError):
        build_fleet_scaling_workload(tmp_path / "x.json", workers=11)
    with pytest.raises(ValueError):
        build_fleet_scaling_workload(tmp_path / "x.json", workers=3, missions=2)


def test_run_rejects_empty_worker_counts(tmp_path: Path) -> None:
    """An empty scale-point list is an error rather than an empty report."""
    with pytest.raises(ValueError):
        run_fleet_benchmark(tmp_path / "run", worker_counts=())


def test_markdown_has_scale_table_and_caveats(tmp_path: Path) -> None:
    """The rendered report carries the scale table, provenance and the scope limit."""
    out = tmp_path / "run"
    run_fleet_benchmark(out, worker_counts=WORKER_COUNTS)
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "## Scaling axes" in markdown
    assert "| worker_count |" in markdown
    for worker_count in WORKER_COUNTS:
        assert f"| {worker_count} |" in markdown
    assert "## Caveats" in markdown
    assert "fleet_wide" in markdown
    assert "within_mission_proxy" not in markdown
    assert "NOT retrieval ranking" in markdown
    assert (out / "report.json").exists()
