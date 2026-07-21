"""FleetBench: fleet/coordination scaling axes over REAL CoordinationBench runs.

FleetBench measures the axes that compound with agent count for Zaxy's category
claim ("governed active memory for agent fleets"), one row per scale point
(worker count), over a real ``CoordinationManager`` mission per scale point:

- coordination_quality    REAL   exact-scored CoordinationBench quality aggregate
- governance_correctness  REAL   exact gate: no non-authoritative row leaked AND
                                 every accepted finding is promotion/citation
                                 backed AND the eventlog stays replayable
- returned_tokens /       REAL   deterministic ``_approx_tokens`` estimates of the
  injected_tokens /                raw worker logs (returned) vs the governed
  token_efficiency                 accepted-state brief (injected)
- cross_agent_transfer    REAL   fleet-wide delivery: the mission's accepted
  (+ _control, _scope)             findings are propagated by the origin agent
                                 through the I4 gate and every OTHER enrolled
                                 agent's real ``checkout_memory`` is scored on
                                 whether it received them. A never-enrolled
                                 stranger is the mandatory negative control.
- mission_count /         REAL   the scaling point
  worker_count
- latency_ms              REAL   wall-clock per scale point (non-deterministic);
                                 EXCLUDED from the fingerprint and equality checks

The workload is FleetBench's own (``build_fleet_scaling_workload``), not the
frozen CoordinationBench case: the shared case pads worker counts above three
with empty-finding filler, so every quality axis was constant across scale points
and only the raw-log byte count moved. FleetBench's workload gives each added
worker real adjudication pressure (a contested subsystem claim, a cross-worker
corroboration, and periodic unbacked noise) so the axes actually vary with scale.

Determinism: the workload is deterministic and every scored field is exact and
reproducible. ``latency_ms`` is the only non-deterministic field and is kept out
of the fingerprint and ``scored_dict``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zaxy.config import Settings
from zaxy.coordination import CoordinationBrief
from zaxy.core import MemoryFabric
from zaxy.fleet import FleetManager
from zaxy.session import SessionManager
from zaxy_benchmarks.coordination_benchmark import (
    CoordinationBenchCase,
    CoordinationBenchCaseResult,
    CoordinationBenchGold,
    CoordinationBenchMetrics,
    CoordinationBenchWorkload,
    _approx_tokens,
    _fingerprint,
    _mean,
    _ratio,
    _run_case,
)

FLEET_WORKLOAD_VERSION = "fleet-v2"
CROSS_AGENT_TRANSFER_SCOPE = "fleet_wide"
CROSS_AGENT_TRANSFER_NOTE = (
    "fleet_wide: real FleetManager propagation through the I4 gate, scored on "
    "enrolled agents' real checkout_memory, with a never-enrolled negative control"
)

# The fleet the benchmark builds per scale point, and the confidence the origin
# agent propagates with. The confidence is the mission finding's own confidence,
# never a constant chosen to clear the gate.
FLEET_ID = "fleetbench"

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

    Every field is REAL and deterministic except ``latency_ms``, which is real
    wall-clock and is excluded from ``scored_dict`` (and therefore from any
    fingerprint/equality check).
    """

    coordination_quality: float
    governance_correctness: float
    returned_tokens: int
    injected_tokens: int
    token_efficiency: float
    cross_agent_transfer: float
    cross_agent_transfer_control: float
    fleet_receiver_count: int
    fleet_promotion_count: int
    fleet_promotion_gated_count: int
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
            "cross_agent_transfer_control": self.cross_agent_transfer_control,
            "cross_agent_transfer_scope": self.cross_agent_transfer_scope,
            "fleet_receiver_count": self.fleet_receiver_count,
            "fleet_promotion_count": self.fleet_promotion_count,
            "fleet_promotion_gated_count": self.fleet_promotion_gated_count,
            "mission_count": self.mission_count,
            "worker_count": self.worker_count,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.scored_dict()
        body["latency_ms"] = self.latency_ms
        return body


@dataclass(frozen=True)
class FleetBenchReport:
    """FleetBench scaling report across scale points.

    ``fingerprint`` covers the SCORED fields only (latency excluded), so two runs
    in different environments produce an identical fingerprint.
    """

    version: str
    workload_fingerprint: str
    results: list[FleetBenchMetrics]
    mean_metrics: FleetBenchMetrics
    fingerprint: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "workload_fingerprint": self.workload_fingerprint,
            "fingerprint": self.fingerprint,
            "cross_agent_transfer_scope": CROSS_AGENT_TRANSFER_SCOPE,
            "mean_metrics": self.mean_metrics.to_dict(),
            "results": [metrics.to_dict() for metrics in self.results],
            "notes": list(self.notes),
        }


def run_fleet_benchmark(
    output_dir: Path,
    *,
    worker_counts: Sequence[int] = (3, 5, 8),
    missions: int = 1,
) -> FleetBenchReport:
    """Run FleetBench over real CoordinationBench cases, one scale point per worker count.

    Each scale point builds FleetBench's scaling workload for ``workers``, runs a
    real ``CoordinationManager`` mission over an isolated temp eventloom, then
    propagates that mission's accepted findings fleet-wide through a real
    ``FleetManager`` and scores every other enrolled agent's real checkout.
    Writes ``report.json`` and ``report.md`` into ``output_dir``.
    """
    if not worker_counts:
        raise ValueError("worker_counts must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[FleetBenchMetrics] = []
    workload_fingerprints: list[str] = []
    for worker_count in worker_counts:
        workload_path = output_dir / f"fleet-workload-w{worker_count}.json"
        workload = build_fleet_scaling_workload(workload_path, missions=missions, workers=worker_count)
        workload_fingerprints.append(workload.fingerprint)
        scale_dir = output_dir / f"scale-w{worker_count}"
        start = time.perf_counter()
        case_results = [_run_case(scale_dir, case) for case in workload.cases]
        transfer = _measure_fleet_transfer(scale_dir, case_results, worker_count=worker_count)
        latency_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            _fleet_metrics_for_scale(
                case_results,
                transfer=transfer,
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
        workload_fingerprint=_fingerprint({"workloads": workload_fingerprints}),
        results=results,
        mean_metrics=mean_metrics,
        fingerprint=fingerprint,
        notes=_notes(),
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


# ---------------------------------------------------------------------------
# Fleet-wide cross-agent transfer (REAL)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetTransferOutcome:
    """Result of one real fleet-wide transfer measurement at a scale point."""

    transfer: float
    control: float
    receiver_count: int
    promotion_count: int
    gated_count: int


def _measure_fleet_transfer(
    scale_dir: Path,
    results: list[CoordinationBenchCaseResult],
    *,
    worker_count: int,
) -> FleetTransferOutcome:
    """Score real fleet-wide transfer of a mission's accepted findings.

    The origin agent appends real source-memory events for each accepted finding
    and propagates them to fleet scope through the real I4 gate; every other
    enrolled agent then performs a real ``checkout_memory`` and is scored on
    whether the active promotions reached it. A never-enrolled stranger runs the
    identical checkout as the negative control: a transfer number is only
    creditable when the stranger receives nothing.
    """
    return asyncio.run(_measure_fleet_transfer_async(scale_dir, results, worker_count=worker_count))


async def _measure_fleet_transfer_async(
    scale_dir: Path,
    results: list[CoordinationBenchCaseResult],
    *,
    worker_count: int,
) -> FleetTransferOutcome:
    base = scale_dir / "fleet-eventloom"
    settings = Settings(fleet_enabled=True)
    origin = "fleet-agent-000"
    receivers = [f"fleet-agent-{index:03d}" for index in range(1, worker_count)]
    stranger = "fleet-stranger"

    manager = FleetManager(eventloom_path=base, settings=settings)
    manager.create_fleet(FLEET_ID, summary="FleetBench scale fleet", actor="coordinator")
    manager.enroll_agent(FLEET_ID, origin, actor="coordinator")
    for receiver in receivers:
        manager.enroll_agent(FLEET_ID, receiver, actor="coordinator")

    active: set[str] = set()
    gated = 0
    for result in results:
        for finding in result.brief.accepted_findings:
            if not finding.claim_key:
                continue
            source = _seed_source_event(base, origin, finding.summary)
            promotion = manager.propagate_outcome(
                FLEET_ID,
                outcome="failure",
                summary=finding.summary,
                origin_session=origin,
                origin_actor=origin,
                source_events=[source],
                confidence=float(finding.confidence or 0.0),
                actor=origin,
                claim_key=finding.claim_key,
            )
            # Promotions the I4 gate holds for review are NOT expected to reach
            # receivers: withholding them is the gate working, not a transfer
            # failure, so they are counted separately and excluded from the
            # denominator rather than scored as misses.
            promotion_id = promotion.promotion_id
            if promotion.review_status == "active" and promotion_id:
                active.add(promotion_id)
            else:
                gated += 1

    if not active or not receivers:
        return FleetTransferOutcome(0.0, 0.0, len(receivers), len(active), gated)

    fabric = MemoryFabric(eventloom_path=str(base))
    fabric.settings = settings
    await fabric.connect()
    try:
        scores = [
            _ratio(len(active & await _delivered(fabric, agent)), len(active)) for agent in receivers
        ]
        stranger_received = await _delivered(fabric, stranger)
    finally:
        await fabric.close()

    return FleetTransferOutcome(
        transfer=_mean(scores),
        control=1.0 if not stranger_received else 0.0,
        receiver_count=len(receivers),
        promotion_count=len(active),
        gated_count=gated,
    )


async def _delivered(fabric: MemoryFabric, agent_id: str) -> set[str]:
    """Promotion IDs the fleet lane of ``agent_id``'s real checkout actually returned."""
    checkout = await fabric.checkout_memory(
        "what has the fleet learned about this system",
        session_id=f"fleetbench-{agent_id}",
        fleet_ids=[FLEET_ID],
        agent_id=agent_id,
        record_reinforcement=False,
    )
    lane = checkout.diagnostics.get("fleet")
    if not isinstance(lane, dict):
        return set()
    return {str(item.get("promotion_id")) for item in lane.get("items", [])}


def _seed_source_event(base: Path, agent_id: str, summary: str) -> dict[str, Any]:
    """Append a real origin-agent outcome event and return its {seq, hash} citation."""
    manager = SessionManager(base_path=str(base))
    event = manager.get(agent_id).eventlog.append(
        "memory.outcome.recorded",
        actor=agent_id,
        payload={"outcome": "failure", "summary": summary},
        thread=agent_id,
    )
    return {"seq": event.seq, "hash": event.hash}


# ---------------------------------------------------------------------------
# FleetBench scaling workload
# ---------------------------------------------------------------------------


def build_fleet_scaling_workload(
    path: Path,
    *,
    missions: int = 1,
    workers: int = 3,
) -> CoordinationBenchWorkload:
    """Write and return FleetBench's deterministic scaling workload."""
    if workers < 3 or workers > 10:
        raise ValueError("workers must be between 3 and 10")
    if missions != 1:
        raise ValueError("FleetBench supports exactly one mission per scale point")
    cases = [_fleet_scaling_case(workers=workers)]
    body = {"version": FLEET_WORKLOAD_VERSION, "cases": [case.to_dict() for case in cases]}
    workload = CoordinationBenchWorkload(
        version=FLEET_WORKLOAD_VERSION,
        cases=cases,
        fingerprint=_fingerprint(body),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return workload


def _fleet_scaling_case(*, workers: int) -> CoordinationBenchCase:
    """Build a case whose adjudication pressure grows with ``workers``.

    The first three workers are the auth investigation; every worker beyond that
    owns a distinct subsystem and contributes real signal rather than filler: a
    well-sourced current claim, a live competing theory on the same key (a
    conflict the parent must resolve), a cross-worker corroboration of the
    previous subsystem (a duplicate the parent must consolidate), and on every
    other worker an unbacked claim that must be rejected for missing evidence.
    """
    workers_specs = _base_workers()
    accepted: dict[str, str] = {"auth.failure.cause": "expired-jwks-cache"}
    conflicts: set[tuple[str, str]] = {("finding-api-jwks", "finding-ui-refresh")}
    duplicates: dict[str, tuple[str, ...]] = {
        "auth.failure.cause=expired-jwks-cache": ("finding-api-jwks", "finding-docs-jwks")
    }
    stale: list[str] = ["finding-api-stale-flag"]
    missing: list[str] = ["finding-no-evidence"]
    forbidden: list[str] = ["missing-browser-refresh", "flag-missing"]

    for extra in range(workers - len(workers_specs)):
        subsystem = f"sub{extra}"
        key = f"{subsystem}.failure.cause"
        winner = f"{subsystem}-token-replay"
        loser = f"{subsystem}-clock-skew"
        findings = [
            _finding(
                f"finding-{subsystem}-primary",
                f"Current CI evidence points at {winner} in the {subsystem} path.",
                key,
                winner,
                [{"kind": "command", "reference": f"ci://runs/{subsystem}/verify"}],
                0.91,
            ),
            _finding(
                f"finding-{subsystem}-contra",
                f"Earlier legacy theory blamed {loser} in {subsystem}.",
                key,
                loser,
                [{"kind": "file", "reference": f"src/{subsystem}/clock.py:18"}],
                0.54,
            ),
        ]
        conflicts.add((f"finding-{subsystem}-primary", f"finding-{subsystem}-contra"))
        accepted[key] = winner
        forbidden.append(loser)
        if extra > 0:
            previous = f"sub{extra - 1}"
            corroboration = f"finding-{subsystem}-corroborates-{previous}"
            findings.append(
                _finding(
                    corroboration,
                    f"Independent doc review confirms {previous}-token-replay.",
                    f"{previous}.failure.cause",
                    f"{previous}-token-replay",
                    [{"kind": "document", "reference": f"docs/{previous}.md:7"}],
                    0.86,
                )
            )
            duplicates[f"{previous}.failure.cause={previous}-token-replay"] = (
                f"finding-{previous}-primary",
                corroboration,
            )
        if extra % 2 == 1:
            unbacked = f"finding-{subsystem}-unbacked"
            findings.append(
                _finding(
                    unbacked,
                    f"Unbacked claim that scope drift degrades {subsystem}.",
                    f"{subsystem}.failure.secondary",
                    f"{subsystem}-scope-drift",
                    [],
                    0.19,
                )
            )
            missing.append(unbacked)
        workers_specs.append(
            {
                "worker_id": f"worker-{subsystem}",
                "assignment": f"Investigate the {subsystem} subsystem",
                "findings": findings,
            }
        )

    gold = CoordinationBenchGold(
        expected_accepted_claims=accepted,
        expected_conflict_pairs=conflicts,
        expected_duplicate_groups=duplicates,
        expected_stale_findings=tuple(stale),
        expected_missing_evidence=tuple(missing),
        final_questions=(
            {
                "query": "What is the accepted cause of auth failures?",
                "expected_terms": ("expired-jwks-cache",),
                "forbidden_terms": tuple(forbidden),
            },
        ),
    )
    return CoordinationBenchCase(
        case_id="fleet-scaling-case-1",
        mission_id="fleet-scaling-case-1",
        objective="Resolve conflicting findings across a scaling worker fleet.",
        workers=workers_specs,
        gold=gold,
    )


def _base_workers() -> list[dict[str, Any]]:
    """The three-worker auth investigation FleetBench's scaling case starts from."""
    return [
        {
            "worker_id": "worker-api",
            "assignment": "Trace API auth failures",
            "findings": [
                _finding(
                    "finding-api-jwks",
                    "JWKS cache expiry is the accepted auth failure cause.",
                    "auth.failure.cause",
                    "expired-jwks-cache",
                    [{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
                    0.95,
                ),
                _finding(
                    "finding-api-stale-flag",
                    "Stale flag-missing theory from old branch.",
                    "auth.failure.cause",
                    "flag-missing",
                    [
                        {
                            "kind": "transcript",
                            "reference": "eventloom://old/events/3#abc",
                            "stale": True,
                            "superseded_by": "decision:jwks-cache",
                        }
                    ],
                    0.31,
                ),
            ],
        },
        {
            "worker_id": "worker-ui",
            "assignment": "Check browser refresh flow",
            "findings": [
                _finding(
                    "finding-ui-refresh",
                    "Browser refresh is the auth failure cause.",
                    "auth.failure.cause",
                    "missing-browser-refresh",
                    [{"kind": "file", "reference": "src/ui/session.ts:42"}],
                    0.64,
                )
            ],
        },
        {
            "worker_id": "worker-docs",
            "assignment": "Verify docs and historical notes",
            "findings": [
                _finding(
                    "finding-docs-jwks",
                    "Docs independently confirm expired JWKS cache.",
                    "auth.failure.cause",
                    "expired-jwks-cache",
                    [{"kind": "document", "reference": "docs/auth.md:12"}],
                    0.88,
                ),
                _finding(
                    "finding-no-evidence",
                    "Unbacked claim that OAuth scope drift caused failures.",
                    "auth.failure.secondary",
                    "oauth-scope-drift",
                    [],
                    0.20,
                ),
            ],
        },
    ]


def _finding(
    finding_id: str,
    summary: str,
    claim_key: str,
    claim_value: str,
    evidence: list[dict[str, Any]],
    confidence: float,
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "summary": summary,
        "claim_key": claim_key,
        "claim_value": claim_value,
        "evidence": evidence,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _fleet_metrics_for_scale(
    results: list[CoordinationBenchCaseResult],
    *,
    transfer: FleetTransferOutcome,
    worker_count: int,
    mission_count: int,
    latency_ms: float,
) -> FleetBenchMetrics:
    if not results:
        raise ValueError("results must not be empty")
    coord_metrics = [result.metrics for result in results]
    returned_tokens = sum(_raw_worker_log_tokens(result.workload_case) for result in results)
    injected_tokens = sum(_governed_brief_tokens(result.brief) for result in results)
    return FleetBenchMetrics(
        coordination_quality=_mean([_coordination_quality(metrics) for metrics in coord_metrics]),
        governance_correctness=_governance_correctness(coord_metrics),
        returned_tokens=returned_tokens,
        injected_tokens=injected_tokens,
        token_efficiency=_token_efficiency(returned_tokens, injected_tokens),
        cross_agent_transfer=transfer.transfer,
        cross_agent_transfer_control=transfer.control,
        fleet_receiver_count=transfer.receiver_count,
        fleet_promotion_count=transfer.promotion_count,
        fleet_promotion_gated_count=transfer.gated_count,
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


def _raw_worker_log_tokens(case: CoordinationBenchCase) -> int:
    """Token cost of the raw worker logs a naive flat system would carry (grows with workers)."""
    return _approx_tokens(json.dumps(case.workers, sort_keys=True))


def _governed_brief_tokens(brief: CoordinationBrief) -> int:
    """Token cost of the governed accepted-state brief the parent injects downstream."""
    accepted_state = {
        "mission_id": brief.mission_id,
        "objective": brief.objective,
        "accepted_findings": [finding.to_dict() for finding in brief.accepted_findings],
    }
    return _approx_tokens(json.dumps(accepted_state, sort_keys=True))


def _token_efficiency(returned_tokens: int, injected_tokens: int) -> float:
    """Fraction of raw worker-log tokens NOT injected into the governed brief.

    ``1 - injected/returned``; higher is better.
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
        cross_agent_transfer_control=_mean(
            [metrics.cross_agent_transfer_control for metrics in results]
        ),
        fleet_receiver_count=int(round(_mean([metrics.fleet_receiver_count for metrics in results]))),
        fleet_promotion_count=int(round(_mean([metrics.fleet_promotion_count for metrics in results]))),
        fleet_promotion_gated_count=int(
            round(_mean([metrics.fleet_promotion_gated_count for metrics in results]))
        ),
        mission_count=int(round(_mean([metrics.mission_count for metrics in results]))),
        worker_count=int(round(_mean([metrics.worker_count for metrics in results]))),
        latency_ms=round(_mean([metrics.latency_ms for metrics in results]), 3),
    )


def _notes() -> list[str]:
    return [
        "cross_agent_transfer is REAL fleet-wide transfer (scope=fleet_wide): the "
        "mission's accepted findings are propagated by the origin agent through the "
        "real I4 gate and every OTHER enrolled agent's real checkout_memory is scored "
        "on whether it received them.",
        "cross_agent_transfer_control is the mandatory negative control: a "
        "never-enrolled stranger runs the identical checkout. It is 1.0 only when the "
        "stranger receives NOTHING. A transfer number with control < 1.0 is not "
        "creditable, because a metric that cannot tell an enrolled agent from a "
        "stranger measures nothing.",
        "SCOPE LIMIT: the checkout fleet lane returns every ACTIVE promotion an "
        "enrolled agent is entitled to, and is not filtered by query relevance. "
        "cross_agent_transfer therefore measures governed DELIVERY (does propagated "
        "memory reach the right agents and only them), NOT retrieval ranking or "
        "relevance. No relevance claim may cite this number.",
        "fleet_promotion_gated_count counts propagations the I4 gate held for review; "
        "they are excluded from the transfer denominator, because withholding them is "
        "the gate working rather than a transfer failure.",
        "latency_ms is REAL wall-clock and environment-dependent; it is EXCLUDED "
        "from the fingerprint and from equality/determinism checks.",
        "returned_tokens and injected_tokens are _approx_tokens estimates (len//4), "
        "not tokenizer-exact counts.",
        "coordination_quality and governance_correctness are REAL exact-scored, "
        "deterministic aggregates of CoordinationBench signals.",
    ]


def _render_markdown(report: FleetBenchReport) -> str:
    lines = [
        "# FleetBench",
        "",
        f"- version: `{report.version}`",
        f"- workload fingerprint: `{report.workload_fingerprint}`",
        f"- fingerprint: `{report.fingerprint}` (scored fields only; latency excluded)",
        f"- cross_agent_transfer scope: `{CROSS_AGENT_TRANSFER_SCOPE}` "
        "(real fleet-wide transfer with a never-enrolled negative control)",
        "",
        "Fleet/coordination scaling axes measured over real CoordinationBench missions "
        "and real FleetManager propagation. Each row is one scale point (worker count); "
        "every axis is exact-scored and deterministic except `latency_ms`.",
        "",
        "## Scaling axes",
        "",
        "| worker_count | mission_count | coordination_quality | governance_correctness | "
        "cross_agent_transfer | control | receivers | promotions | gated | returned_tokens | "
        "injected_tokens | token_efficiency | latency_ms |",
        "|--------------|---------------|----------------------|------------------------|"
        "----------------------|---------|-----------|------------|-------|-----------------|"
        "-----------------|------------------|------------|",
    ]
    lines.extend(_scale_row(metrics) for metrics in report.results)
    mean = report.mean_metrics
    lines.append(
        f"| **mean** | {mean.mission_count} | {mean.coordination_quality} | "
        f"{mean.governance_correctness} | {mean.cross_agent_transfer} | "
        f"{mean.cross_agent_transfer_control} | {mean.fleet_receiver_count} | "
        f"{mean.fleet_promotion_count} | {mean.fleet_promotion_gated_count} | "
        f"{mean.returned_tokens} | {mean.injected_tokens} | {mean.token_efficiency} | "
        f"{mean.latency_ms} |"
    )
    lines.extend(
        [
            "",
            "Axis directions: `coordination_quality`, `governance_correctness`, "
            "`cross_agent_transfer`, `cross_agent_transfer_control`, and "
            "`token_efficiency` are higher-is-better in [0, 1]. `token_efficiency` = "
            "fraction of raw worker-log tokens NOT injected into the governed brief "
            "(`1 - injected/returned`).",
            "",
            "## Metric provenance",
            "",
            "| Metric | Status |",
            "|--------|--------|",
            "| coordination_quality | REAL (exact, deterministic) |",
            "| governance_correctness | REAL (exact, deterministic) |",
            "| returned_tokens / injected_tokens / token_efficiency | REAL (deterministic `_approx_tokens` estimates) |",
            "| cross_agent_transfer | REAL (fleet_wide; governed delivery, not relevance) |",
            "| cross_agent_transfer_control | REAL (never-enrolled negative control) |",
            "| mission_count / worker_count | REAL (scaling point) |",
            "| latency_ms | REAL wall-clock (non-deterministic; excluded from fingerprint) |",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in report.notes)
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "env PYTHONPATH=src EMBEDDING_ENABLED=true EMBEDDING_PROVIDER=hash EMBEDDING_DIMENSION=1536 \\",
            "  python -c 'from pathlib import Path; "
            "from zaxy_benchmarks.fleet_benchmark import run_fleet_benchmark; "
            "run_fleet_benchmark(Path(\"reports/benchmarks/fleet-transfer-v1\"))'",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _scale_row(metrics: FleetBenchMetrics) -> str:
    return (
        f"| {metrics.worker_count} | {metrics.mission_count} | "
        f"{metrics.coordination_quality} | {metrics.governance_correctness} | "
        f"{metrics.cross_agent_transfer} | {metrics.cross_agent_transfer_control} | "
        f"{metrics.fleet_receiver_count} | {metrics.fleet_promotion_count} | "
        f"{metrics.fleet_promotion_gated_count} | {metrics.returned_tokens} | "
        f"{metrics.injected_tokens} | {metrics.token_efficiency} | {metrics.latency_ms} |"
    )
