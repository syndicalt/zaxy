"""Tests for the warm-checkout latency lane (§6.8 of the 2026-07-21 gap audit)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zaxy_benchmarks.checkout_latency_lane import (
    CHECKOUT_LATENCY_LANE_VERSION,
    CheckoutLatencyPoint,
    CheckoutLatencyReport,
    measure_checkout_latency,
)


def test_report_is_labeled_internal_and_versioned() -> None:
    """The lane must declare itself internal so its numbers are never read as published."""
    payload = CheckoutLatencyReport().to_dict()
    assert payload["validation"] == "internal"
    assert payload["version"] == CHECKOUT_LATENCY_LANE_VERSION


def test_scaling_note_refuses_to_claim_beyond_the_measured_size() -> None:
    """The note must name the largest size measured and disclaim anything past it."""
    report = CheckoutLatencyReport(
        points=[
            CheckoutLatencyPoint(events=500, p50_ms=40.0, p95_ms=60.0, samples=20),
            CheckoutLatencyPoint(events=8000, p50_ms=80.0, p95_ms=120.0, samples=20),
        ]
    )
    note = report.to_dict()["scaling_note"]
    assert "8000" in note
    assert "makes no claim beyond" in note


def test_scaling_note_declines_with_a_single_point() -> None:
    """One size cannot support a scaling statement, so the lane must not make one."""
    report = CheckoutLatencyReport(
        points=[CheckoutLatencyPoint(events=500, p50_ms=40.0, p95_ms=60.0, samples=20)]
    )
    assert "no scaling claim" in report.to_dict()["scaling_note"]


@pytest.mark.parametrize(
    "kwargs",
    [{"repeats": 0}, {"warmup": -1}, {"sizes": (0,)}, {"sizes": (-5,)}],
)
def test_rejects_invalid_measurement_parameters(tmp_path: Path, kwargs: dict[str, object]) -> None:
    """Invalid sampling parameters are refused rather than silently producing a number."""
    with pytest.raises(ValueError):
        asyncio.run(measure_checkout_latency(tmp_path, **kwargs))  # type: ignore[arg-type]


def test_measures_a_real_checkout_at_each_requested_size(tmp_path: Path) -> None:
    """Every requested size yields a positive p50 from real checkouts, ordered by size."""
    report = asyncio.run(
        measure_checkout_latency(tmp_path, sizes=(40, 120), repeats=3, warmup=1)
    )
    assert [point.events for point in report.points] == [40, 120]
    assert all(point.p50_ms > 0 for point in report.points)
    assert all(point.samples == 3 for point in report.points)
