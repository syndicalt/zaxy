"""FleetBench: scaffold for fleet/coordination scaling axes over CoordinationBench.

FleetBench measures the axes that compound with agent count for Zaxy's category
claim ("governed active memory for agent fleets"), one row per scale point
(worker count), over REAL CoordinationBench runs (a real ``CoordinationManager``
mission per scale point):

- coordination_quality    REAL   exact-scored CoordinationBench quality aggregate
- governance_correctness  REAL   exact gate: no non-authoritative row leaked AND
                                 every accepted finding is promotion/citation
                                 backed AND the eventlog stays replayable
- returned_tokens /       REAL   deterministic ``_approx_tokens`` estimates of the
  injected_tokens /                raw worker logs (returned) vs the governed
  token_efficiency                 accepted-state brief (injected)
- cross_agent_transfer    PROXY  SCAFFOLD. Within-mission worker->parent promotion
  (+ _scope)                       propagation, scope="within_mission_proxy". It is
                                 NOT fleet-wide cross-agent transfer; fleet-wide
                                 transfer is realized in I7.
- mission_count /         REAL   the scaling point
  worker_count
- latency_ms              REAL   wall-clock per scale point (non-deterministic);
                                 EXCLUDED from the fingerprint and equality checks

Determinism mirrors ``coordination_benchmark``: the workload is frozen and every
scored field is exact and reproducible. ``latency_ms`` is the only
non-deterministic field and is kept out of the fingerprint and ``scored_dict``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zaxy.coordination import CoordinationBrief
from zaxy_benchmarks.coordination_benchmark import (
    CoordinationBenchCase,
    CoordinationBenchCaseResult,
    CoordinationBenchMetrics,
    _approx_tokens,
    _fingerprint,
    _mean,
    _ratio,
    _run_case,
    build_coordination_workload,
)

FLEET_WORKLOAD_VERSION = "fleet-v1"
CROSS_AGENT_TRANSFER_SCOPE = "within_mission_proxy"
CROSS_AGENT_TRANSFER_NOTE = "within_mission_proxy; fleet-wide pending I7"

# Exact-scored CoordinationBench signals aggregated into coordination_quality.
_COORDINATION_QUALITY_FIELDS = (
    "accepted_finding_precision",
    "accepted_finding_recall",
    "conflict_precision",
    "conflict_recall",
    "stale_claim_rejection",
    "duplicate_consolidation",
    "evidence_coverage",
    "parent_checkout_answerability",
    "citation_coverage",
)


@dataclass(frozen=True)
class FleetBenchMetrics:
    """Fleet/coordination scaling axes for one scale point.

    Every field is REAL and deterministic except:
    - ``cross_agent_transfer`` is a SCAFFOLD within-mission proxy (see
      ``cross_agent_transfer_scope``); fleet-wide transfer lands in I7.
    - ``latency_ms`` is real wall-clock and non-deterministic; it is excluded
      from ``scored_dict`` (and therefore from any fingerprint/equality check).
    """

    coordination_quality: float
    governance_correctness: float
    returned_tokens: int
    injected_tokens: int
    token_efficiency: float
    cross_agent_transfer: float
    mission_count: int
    worker_count: int
    latency_ms: float
    cross_agent_transfer_scope: str = CROSS_AGENT_TRANSFER_SCOPE

    def scored_dict(self) -> dict[str, Any]:
        """Deterministic scored fields only (``latency_ms`` excluded)."""
        return {
            "coordination_quality": self.coordination_quality,
            "governance_correctness": self.governance_correctness,
            "returned_tokens": self.returned_tokens,
            "injected_tokens": self.injected_tokens,
            "token_efficiency": self.token_efficiency,
            "cross_agent_transfer": self.cross_agent_transfer,
            "cross_agent_transfer_scope": self.cross_agent_transfer_scope,
            "mission_count": self.mission_count,
            "worker_count": self.worker_count,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.scored_dict()
        body["latency_ms"] = self.latency_ms
        body["scaffold"] = {"cross_agent_transfer": CROSS_AGENT_TRANSFER_NOTE}
        return body


@dataclass(frozen=True)
class FleetBenchReport:
    """FleetBench scaling report across scale points.

    ``fingerprint`` covers the SCORED fields only (latency excluded), so two runs
    in different environments produce an identical fingerprint.
    """

    version: str
    results: list[FleetBenchMetrics]
    mean_metrics: FleetBenchMetrics
    fingerprint: str
    scaffold_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "cross_agent_transfer_scope": CROSS_AGENT_TRANSFER_SCOPE,
            "mean_metrics": self.mean_metrics.to_dict(),
            "results": [metrics.to_dict() for metrics in self.results],
            "scaffold_notes": list(self.scaffold_notes),
        }


def run_fleet_benchmark(
    output_dir: Path,
    *,
    worker_counts: Sequence[int] = (3, 5, 8),
    missions: int = 1,
) -> FleetBenchReport:
    """Run FleetBench over real CoordinationBench cases, one scale point per worker count.

    Each scale point builds the frozen CoordinationBench workload for ``workers``
    and runs a real ``CoordinationManager`` mission (via ``_run_case``) over an
    isolated temp eventloom, measures wall-clock latency with ``perf_counter``,
    and computes the fleet axes. The assembled scaling report shows how each axis
    behaves as worker count grows. Writes ``report.json`` and ``report.md`` into
    ``output_dir``.
    """
    if not worker_counts:
        raise ValueError("worker_counts must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[FleetBenchMetrics] = []
    for worker_count in worker_counts:
        workload_path = output_dir / f"fleet-workload-w{worker_count}.json"
        workload = build_coordination_workload(workload_path, missions=missions, workers=worker_count)
        scale_dir = output_dir / f"scale-w{worker_count}"
        start = time.perf_counter()
        case_results = [_run_case(scale_dir, case) for case in workload.cases]
        latency_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            _fleet_metrics_for_scale(
                case_results,
                worker_count=worker_count,
                mission_count=len(workload.cases),
                latency_ms=latency_ms,
            )
        )
    mean_metrics = _mean_fleet_metrics(results)
    fingerprint = _fingerprint(
        {
            "version": FLEET_WORKLOAD_VERSION,
            "results": [metrics.scored_dict() for metrics in results],
            "mean": mean_metrics.scored_dict(),
        }
    )
    report = FleetBenchReport(
        version=FLEET_WORKLOAD_VERSION,
        results=results,
        mean_metrics=mean_metrics,
        fingerprint=fingerprint,
        scaffold_notes=_scaffold_notes(),
    )
    write_fleet_benchmark_report(report, output_dir)
    return report


def write_fleet_benchmark_report(report: FleetBenchReport, output_dir: Path) -> None:
    """Write the FleetBench JSON and markdown reports into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")


def _fleet_metrics_for_scale(
    results: list[CoordinationBenchCaseResult],
    *,
    worker_count: int,
    mission_count: int,
    latency_ms: float,
) -> FleetBenchMetrics:
    if not results:
        raise ValueError("results must not be empty")
    coord_metrics = [result.metrics for result in results]
    returned_tokens = sum(_raw_worker_log_tokens(result.workload_case) for result in results)
    injected_tokens = sum(_governed_brief_tokens(result.brief) for result in results)
    cross_agent_transfer = _mean(
        [
            _cross_agent_transfer_proxy(
                result.workload_case,
                result.brief,
                eventloom_replayable=result.metrics.eventloom_replayable,
            )
            for result in results
        ]
    )
    return FleetBenchMetrics(
        coordination_quality=_mean([_coordination_quality(metrics) for metrics in coord_metrics]),
        governance_correctness=_governance_correctness(coord_metrics),
        returned_tokens=returned_tokens,
        injected_tokens=injected_tokens,
        token_efficiency=_token_efficiency(returned_tokens, injected_tokens),
        cross_agent_transfer=cross_agent_transfer,
        mission_count=mission_count,
        worker_count=worker_count,
        latency_ms=round(latency_ms, 3),
    )


def _coordination_quality(metrics: CoordinationBenchMetrics) -> float:
    """Exact-scored aggregate of CoordinationBench quality signals (higher is better)."""
    return _mean([float(getattr(metrics, name)) for name in _COORDINATION_QUALITY_FIELDS])


def _governance_correctness(metrics_list: list[CoordinationBenchMetrics]) -> float:
    """Fraction of cases passing the governance gate (exact, deterministic)."""
    if not metrics_list:
        return 0.0
    passed = sum(1 for metrics in metrics_list if _governance_ok(metrics))
    return _ratio(passed, len(metrics_list))


def _governance_ok(metrics: CoordinationBenchMetrics) -> bool:
    """A case is governance-correct when, per existing CoordinationBench signals:

    - no non-authoritative row leaked (``non_authoritative_leakage == 1.0``),
    - every accepted finding is promotion/citation backed (``citation_coverage == 1.0``),
    - the eventlog stays replayable (``eventloom_replayable``).
    """
    return (
        metrics.non_authoritative_leakage == 1.0
        and metrics.citation_coverage == 1.0
        and bool(metrics.eventloom_replayable)
    )


def _cross_agent_transfer_proxy(
    case: CoordinationBenchCase,
    brief: CoordinationBrief,
    *,
    eventloom_replayable: bool,
) -> float:
    """SCAFFOLD: within-mission worker->parent promotion propagation.

    Scores the fraction of legitimately parent-accepted findings (claim matches
    gold) that originated from a worker session and remain replayable/cited
    (source event seq + hash present). This is a LOCAL PROXY
    (scope=``within_mission_proxy``) for fleet-wide cross-agent transfer: it
    measures memory moving worker->parent inside ONE mission, NOT propagation
    across agents in a fleet. Fleet-wide cross-agent transfer is realized in I7.
    """
    worker_ids = {str(worker["worker_id"]) for worker in case.workers}
    required = {
        finding.finding_id
        for finding in brief.accepted_findings
        if finding.claim_key
        and case.gold.expected_accepted_claims.get(finding.claim_key) == finding.claim_value
    }
    if not required or not eventloom_replayable:
        return 0.0
    transferred = {
        finding.finding_id
        for finding in brief.accepted_findings
        if finding.finding_id in required
        and finding.worker_id in worker_ids
        and finding.source_event_seq is not None
        and bool(finding.source_event_hash)
    }
    return _ratio(len(transferred), len(required))


def _raw_worker_log_tokens(case: CoordinationBenchCase) -> int:
    """Token cost of the raw worker logs a naive flat system would carry (grows with workers)."""
    return _approx_tokens(json.dumps(case.workers, sort_keys=True))


def _governed_brief_tokens(brief: CoordinationBrief) -> int:
    """Token cost of the governed accepted-state brief the parent injects downstream (bounded)."""
    accepted_state = {
        "mission_id": brief.mission_id,
        "objective": brief.objective,
        "accepted_findings": [finding.to_dict() for finding in brief.accepted_findings],
    }
    return _approx_tokens(json.dumps(accepted_state, sort_keys=True))


def _token_efficiency(returned_tokens: int, injected_tokens: int) -> float:
    """Fraction of raw worker-log tokens NOT injected into the governed brief.

    ``1 - injected/returned``; higher is better. As worker count grows the raw
    worker logs (returned) grow while the governed brief (injected) stays
    bounded, so token_efficiency rises with fleet size.
    """
    if returned_tokens <= 0:
        return 0.0
    return round(1.0 - (injected_tokens / returned_tokens), 6)


def _mean_fleet_metrics(results: list[FleetBenchMetrics]) -> FleetBenchMetrics:
    if not results:
        raise ValueError("results must not be empty")
    return FleetBenchMetrics(
        coordination_quality=_mean([metrics.coordination_quality for metrics in results]),
        governance_correctness=_mean([metrics.governance_correctness for metrics in results]),
        returned_tokens=int(round(_mean([metrics.returned_tokens for metrics in results]))),
        injected_tokens=int(round(_mean([metrics.injected_tokens for metrics in results]))),
        token_efficiency=_mean([metrics.token_efficiency for metrics in results]),
        cross_agent_transfer=_mean([metrics.cross_agent_transfer for metrics in results]),
        mission_count=int(round(_mean([metrics.mission_count for metrics in results]))),
        worker_count=int(round(_mean([metrics.worker_count for metrics in results]))),
        latency_ms=round(_mean([metrics.latency_ms for metrics in results]), 3),
    )


def _scaffold_notes() -> list[str]:
    return [
        "cross_agent_transfer is a SCAFFOLD within-mission PROXY "
        f"(scope={CROSS_AGENT_TRANSFER_SCOPE}): it scores worker->parent promotion "
        "propagation inside a single mission, NOT fleet-wide cross-agent transfer. "
        "Fleet-wide cross-agent transfer is realized in I7.",
        "latency_ms is REAL wall-clock and environment-dependent; it is EXCLUDED "
        "from the fingerprint and from equality/determinism checks.",
        "returned_tokens and injected_tokens are _approx_tokens estimates (len//4), "
        "not tokenizer-exact counts.",
        "coordination_quality and governance_correctness are REAL exact-scored, "
        "deterministic aggregates of CoordinationBench signals.",
    ]


def _render_markdown(report: FleetBenchReport) -> str:
    lines = [
        "# FleetBench (scaffold)",
        "",
        f"- version: `{report.version}`",
        f"- fingerprint: `{report.fingerprint}` (scored fields only; latency excluded)",
        f"- cross_agent_transfer scope: `{CROSS_AGENT_TRANSFER_SCOPE}` "
        "(within-mission proxy; fleet-wide pending I7)",
        "",
        "Fleet/coordination scaling axes measured over real CoordinationBench runs. "
        "Each row is one scale point (worker count); every axis is exact-scored and "
        "deterministic except `latency_ms` (real wall-clock).",
        "",
        "## Scaling axes",
        "",
        "| worker_count | mission_count | coordination_quality | governance_correctness | "
        "cross_agent_transfer (proxy) | returned_tokens | injected_tokens | token_efficiency | latency_ms |",
        "|--------------|---------------|----------------------|------------------------|"
        "------------------------------|-----------------|-----------------|------------------|------------|",
    ]
    for metrics in report.results:
        lines.append(_scale_row(metrics))
    mean = report.mean_metrics
    lines.append(
        f"| **mean** | {mean.mission_count} | {mean.coordination_quality} | "
        f"{mean.governance_correctness} | {mean.cross_agent_transfer} | {mean.returned_tokens} | "
        f"{mean.injected_tokens} | {mean.token_efficiency} | {mean.latency_ms} |"
    )
    lines.extend(
        [
            "",
            "Axis directions: `coordination_quality`, `governance_correctness`, "
            "`cross_agent_transfer`, and `token_efficiency` are higher-is-better in [0, 1]. "
            "`token_efficiency` = fraction of raw worker-log tokens NOT injected into the "
            "governed brief (`1 - injected/returned`); `returned_tokens` = raw worker logs, "
            "`injected_tokens` = governed accepted-state brief.",
            "",
            "## Metric provenance",
            "",
            "| Metric | Status |",
            "|--------|--------|",
            "| coordination_quality | REAL (exact, deterministic) |",
            "| governance_correctness | REAL (exact, deterministic) |",
            "| returned_tokens / injected_tokens / token_efficiency | REAL (deterministic `_approx_tokens` estimates) |",
            "| cross_agent_transfer | PROXY / SCAFFOLD (within_mission_proxy; fleet-wide pending I7) |",
            "| mission_count / worker_count | REAL (scaling point) |",
            "| latency_ms | REAL wall-clock (non-deterministic; excluded from fingerprint) |",
            "",
            "## Scaffold caveats",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in report.scaffold_notes)
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "Regenerate this scaffold report over real CoordinationBench cases:",
            "",
            "```bash",
            "env PYTHONPATH=src EMBEDDING_ENABLED=true EMBEDDING_PROVIDER=hash EMBEDDING_DIMENSION=1536 \\",
            "  python -c 'from pathlib import Path; "
            "from zaxy_benchmarks.fleet_benchmark import run_fleet_benchmark; "
            "run_fleet_benchmark(Path(\"reports/experimental/fleet-benchmark-scaffold\"))'",
            "```",
            "",
            "The `zaxy` CLI command for FleetBench is wired separately by the orchestrator.",
            "",
        ]
    )
    return "\n".join(lines)


def _scale_row(metrics: FleetBenchMetrics) -> str:
    return (
        f"| {metrics.worker_count} | {metrics.mission_count} | "
        f"{metrics.coordination_quality} | {metrics.governance_correctness} | "
        f"{metrics.cross_agent_transfer} | {metrics.returned_tokens} | "
        f"{metrics.injected_tokens} | {metrics.token_efficiency} | {metrics.latency_ms} |"
    )
