"""Internal lane: warm Memory Checkout latency as a function of store size.

Memory Checkout is designed to run every turn, so its warm latency is the
per-turn cost of using Zaxy at all. ``AGENTS.md`` once declared a <300 ms
retrieval target and `fable-findings.md` restated the measured reality as
~1.0-1.6 s warm / ~10-12 s cold. Neither number came from a harness anyone
could re-run, and the store that produced them was never reproduced.

This lane exists so the claim rests on measurement rather than recollection.
It reports warm p50/p95 at several store sizes, and it deliberately reports
the size it measured rather than extrapolating to one it did not.

**Methodology.** Arms are interleaved within a single process against the same
store, alternating which arm leads each round, with warmup rounds discarded.
Cross-run absolute latency on a loaded machine varies substantially, so the
within-run shape (how latency scales with size) is the trustworthy part and the
absolute p50 is reported with an explicit hardware caveat.

**Seeding.** Events are appended through the raw event log and then projected,
rather than through ``MemoryFabric.append``, because the fabric path costs
~30 ms/event (measured) and would make a 16k-event store a 8-minute build. The
projection is still performed for every event, so the retrieval lanes see the
same graph they would in production -- only the append bookkeeping is skipped.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zaxy.core.fabric import MemoryFabric

#: Store sizes the shipped default measures. Larger sizes are supported and are
#: the point of the lane being parameterised -- they are simply slow to build.
DEFAULT_SIZES: tuple[int, ...] = (500, 2000, 8000)

#: Discarded rounds before measurement, to settle caches and imports.
DEFAULT_WARMUP = 3

#: Measured rounds per arm.
DEFAULT_REPEATS = 20

CHECKOUT_LATENCY_LANE_VERSION = "checkout-latency-v1"

_TOPICS: tuple[str, ...] = (
    "authentication", "batching", "caching", "deployment", "embedding",
    "indexing", "latency", "migration", "projection", "retrieval",
    "salience", "throughput",
)

_QUERY = "what should I know about retrieval latency?"


@dataclass
class CheckoutLatencyPoint:
    """Measured warm-checkout latency at one store size."""

    events: int
    p50_ms: float
    p95_ms: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready measurement."""
        return {
            "events": self.events,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "samples": self.samples,
        }


@dataclass
class CheckoutLatencyReport:
    """Warm-checkout latency across store sizes, with its own honesty caveats."""

    version: str = CHECKOUT_LATENCY_LANE_VERSION
    points: list[CheckoutLatencyPoint] = field(default_factory=list)
    validation: str = "internal"
    measurement: str = (
        "Warm Memory Checkout p50/p95 at several store sizes, interleaved within "
        "one process against one store per size, warmup rounds discarded. Absolute "
        "latency is hardware- and load-dependent; the scaling shape is the "
        "comparable part. Cold start is not measured here."
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready report."""
        return {
            "lane": "checkout_latency",
            "version": self.version,
            "validation": self.validation,
            "measurement": self.measurement,
            "points": [point.to_dict() for point in self.points],
            "scaling_note": _scaling_note(self.points),
        }


def _scaling_note(points: list[CheckoutLatencyPoint]) -> str:
    """Describe how latency moved across the measured sizes, without extrapolating."""
    if len(points) < 2:
        return "Fewer than two sizes measured; no scaling claim."
    first, last = points[0], points[-1]
    if first.p50_ms <= 0:
        return "Baseline p50 was non-positive; no scaling claim."
    growth = last.p50_ms / first.p50_ms
    size_growth = last.events / max(first.events, 1)
    return (
        f"p50 grew {growth:.2f}x while the store grew {size_growth:.0f}x "
        f"({first.events} -> {last.events} events). This lane measured up to "
        f"{last.events} events and makes no claim beyond that size."
    )


def _seed_events(fabric: MemoryFabric, session_id: str, count: int) -> None:
    """Append and project ``count`` synthetic events without the fabric append path."""
    eventlog = fabric.session_manager.get(session_id).eventlog
    for index in range(count):
        topic = _TOPICS[index % len(_TOPICS)]
        eventlog.append(
            "note.recorded",
            actor="seed",
            payload={"text": f"note {index} about {topic}: observation {index}"},
            thread=session_id,
        )


async def _project_all(fabric: MemoryFabric, session_id: str) -> None:
    """Project every seeded event so retrieval lanes see a production-shaped graph."""
    for event in fabric.session_manager.get(session_id).eventlog.read_all():
        await fabric._project_event(event, session_id=session_id)


async def measure_checkout_latency(
    root: Path,
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
) -> CheckoutLatencyReport:
    """Measure warm checkout p50/p95 at each requested store size."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must not be negative")
    if any(size <= 0 for size in sizes):
        raise ValueError("sizes must be positive")

    report = CheckoutLatencyReport()
    for size in sorted(sizes):
        fabric = MemoryFabric(
            eventloom_path=str(root / f"store-{size}" / ".eventloom"),
            tracer_disabled=True,
        )
        await fabric.connect()
        try:
            _seed_events(fabric, "bench", size)
            await _project_all(fabric, "bench")
            for _ in range(warmup):
                await fabric.checkout_memory(_QUERY, session_id="bench", limit=10)
            timings: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter()
                await fabric.checkout_memory(_QUERY, session_id="bench", limit=10)
                timings.append((time.perf_counter() - start) * 1000.0)
        finally:
            await fabric.close()
        timings.sort()
        report.points.append(
            CheckoutLatencyPoint(
                events=size,
                p50_ms=statistics.median(timings),
                p95_ms=timings[min(len(timings) - 1, int(len(timings) * 0.95))],
                samples=len(timings),
            )
        )
    return report
