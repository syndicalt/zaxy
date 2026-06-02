"""CoordinationBench: exact-scored benchmark for multi-agent coordination."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import math
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from zaxy.coordination import ConflictState, CoordinationBrief, CoordinationManager, FindingState

COORDINATION_WORKLOAD_VERSION = "coordination-v1"
COORDINATION_BASELINE_DESCRIPTIONS = {
    "flat_transcript": "Concatenate every worker finding and treat the combined transcript as accepted state.",
    "markdown_notes": "Render worker findings as shared markdown notes without promotion or conflict semantics.",
    "bm25_worker_logs": "Rank rendered worker findings with local BM25 and score the retrieved worker context.",
}
COORDINATION_COMPETITOR_ADAPTERS = {
    "mem0": {
        "display_name": "Mem0",
        "adapter_contract": "coordinationbench-v1",
        "blockers": [
            "No pinned adapter package/version and same-harness workload replay contract has been configured.",
        ],
    },
    "agent_memory": {
        "display_name": "Agent Memory",
        "adapter_contract": "coordinationbench-v1",
        "blockers": [
            "No pinned adapter package/version and same-harness workload replay contract has been configured.",
        ],
    },
    "activegraph": {
        "display_name": "ActiveGraph",
        "adapter_contract": "coordinationbench-v1",
        "blockers": [
            "No pinned adapter package/version and same-harness workload replay contract has been configured.",
        ],
    },
    "quarq": {
        "display_name": "Quarq",
        "adapter_contract": "coordinationbench-v1",
        "blockers": [
            "No pinned Quarq runner manifest, source ref, and same-harness workload replay contract has been configured.",
        ],
    },
    "hybi": {
        "display_name": "Semantic Reach / HyperBinder / Hybi",
        "adapter_contract": "coordinationbench-v1",
        "blockers": [
            "No pinned HyperBinder/Hybi server/runtime, source ref, and same-harness workload replay contract has been configured.",
        ],
    },
}
COORDINATION_COMPETITOR_MANIFEST_FIELDS = (
    "name",
    "display_name",
    "adapter_contract",
    "adapter_version",
    "install_command",
    "run_command",
    "source_url",
    "source_ref",
)
COORDINATION_COMPETITOR_RUNNER_PLACEHOLDER = "__REPLACE_WITH_PINNED_RUNNER_ARGV__"


@dataclass(frozen=True)
class CoordinationBenchGold:
    """Gold labels for one coordination benchmark case."""

    expected_accepted_claims: dict[str, str]
    expected_conflict_pairs: set[tuple[str, str]]
    expected_duplicate_groups: dict[str, tuple[str, ...]]
    expected_stale_findings: tuple[str, ...]
    expected_missing_evidence: tuple[str, ...]
    final_questions: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CoordinationBenchGold:
        return cls(
            expected_accepted_claims=dict(value["expected_accepted_claims"]),
            expected_conflict_pairs={tuple(item) for item in value["expected_conflict_pairs"]},
            expected_duplicate_groups={
                str(key): tuple(group) for key, group in value["expected_duplicate_groups"].items()
            },
            expected_stale_findings=tuple(value["expected_stale_findings"]),
            expected_missing_evidence=tuple(value["expected_missing_evidence"]),
            final_questions=tuple(value["final_questions"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_accepted_claims": self.expected_accepted_claims,
            "expected_conflict_pairs": [list(pair) for pair in sorted(self.expected_conflict_pairs)],
            "expected_duplicate_groups": {
                key: list(group) for key, group in sorted(self.expected_duplicate_groups.items())
            },
            "expected_stale_findings": list(self.expected_stale_findings),
            "expected_missing_evidence": list(self.expected_missing_evidence),
            "final_questions": list(self.final_questions),
        }


@dataclass(frozen=True)
class CoordinationBenchCase:
    """One deterministic multi-worker coordination case."""

    case_id: str
    mission_id: str
    objective: str
    workers: list[dict[str, Any]]
    gold: CoordinationBenchGold

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CoordinationBenchCase:
        return cls(
            case_id=str(value["case_id"]),
            mission_id=str(value["mission_id"]),
            objective=str(value["objective"]),
            workers=list(value["workers"]),
            gold=CoordinationBenchGold.from_dict(value["gold"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mission_id": self.mission_id,
            "objective": self.objective,
            "workers": self.workers,
            "gold": self.gold.to_dict(),
        }


@dataclass(frozen=True)
class CoordinationBenchWorkload:
    """Frozen CoordinationBench workload."""

    version: str
    cases: list[CoordinationBenchCase]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CoordinationBenchWorkload:
        return cls(
            version=str(value["version"]),
            fingerprint=str(value["fingerprint"]),
            cases=[CoordinationBenchCase.from_dict(case) for case in value["cases"]],
        )


@dataclass(frozen=True)
class CoordinationBenchMetrics:
    """Exact-scored coordination quality metrics."""

    accepted_finding_precision: float
    accepted_finding_recall: float
    conflict_precision: float
    conflict_recall: float
    stale_claim_rejection: float
    duplicate_consolidation: float
    evidence_coverage: float
    parent_checkout_answerability: float
    citation_coverage: float
    eventloom_replayable: bool
    returned_tokens: int
    injected_tokens: int
    brief_latency_ms: float
    promotion_latency_ms: float
    accepted_state_synthesis_quality: float = 1.0
    non_authoritative_leakage: float = 1.0
    purpose_feedback_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_finding_precision": self.accepted_finding_precision,
            "accepted_finding_recall": self.accepted_finding_recall,
            "conflict_precision": self.conflict_precision,
            "conflict_recall": self.conflict_recall,
            "stale_claim_rejection": self.stale_claim_rejection,
            "duplicate_consolidation": self.duplicate_consolidation,
            "evidence_coverage": self.evidence_coverage,
            "parent_checkout_answerability": self.parent_checkout_answerability,
            "citation_coverage": self.citation_coverage,
            "eventloom_replayable": self.eventloom_replayable,
            "returned_tokens": self.returned_tokens,
            "injected_tokens": self.injected_tokens,
            "brief_latency_ms": self.brief_latency_ms,
            "promotion_latency_ms": self.promotion_latency_ms,
            "accepted_state_synthesis_quality": self.accepted_state_synthesis_quality,
            "non_authoritative_leakage": self.non_authoritative_leakage,
            "purpose_feedback_coverage": self.purpose_feedback_coverage,
        }


@dataclass(frozen=True)
class CoordinationCompetitorAdapterDisclosure:
    """Disclosure record for an external same-harness adapter."""

    name: str
    display_name: str
    adapter_contract: str
    status: str
    claim_status: str
    blockers: tuple[str, ...]
    metrics: CoordinationBenchMetrics | None = None
    result_audit: CoordinationCompetitorResultAudit | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "adapter_contract": self.adapter_contract,
            "status": self.status,
            "claim_status": self.claim_status,
            "blockers": list(self.blockers),
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "result_audit": self.result_audit.to_dict() if self.result_audit is not None else None,
        }


@dataclass(frozen=True)
class CoordinationCompetitorResultAudit:
    """Provenance for a pinned external result file scored by Zaxy."""

    result_fingerprint: str
    generated_at_utc: str | None
    case_count: int
    manifest: dict[str, Any]
    runner_command: list[str] | None = None
    runner_returncode: int | None = None
    runner_stdout_path: str | None = None
    runner_stderr_path: str | None = None
    runner_stdout_sha256: str | None = None
    runner_stderr_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_fingerprint": self.result_fingerprint,
            "generated_at_utc": self.generated_at_utc,
            "case_count": self.case_count,
            "manifest": dict(self.manifest),
            "runner_command": list(self.runner_command) if self.runner_command is not None else None,
            "runner_returncode": self.runner_returncode,
            "runner_stdout_path": self.runner_stdout_path,
            "runner_stderr_path": self.runner_stderr_path,
            "runner_stdout_sha256": self.runner_stdout_sha256,
            "runner_stderr_sha256": self.runner_stderr_sha256,
        }


@dataclass(frozen=True)
class CoordinationCompetitorClaimGate:
    """Machine-readable guardrail for public same-harness competitor claims."""

    status: str
    required_adapters: tuple[str, ...]
    completed_adapters: tuple[str, ...]
    blocked_adapters: dict[str, str]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "required_adapters": list(self.required_adapters),
            "completed_adapters": list(self.completed_adapters),
            "blocked_adapters": dict(self.blocked_adapters),
            "message": self.message,
        }


@dataclass(frozen=True)
class CoordinationPurposeSynthesisGate:
    """Machine-readable guardrail for Coordinate purpose/synthesis claims."""

    status: str
    required_metrics: dict[str, float | bool]
    blocked_metrics: dict[str, str]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "required_metrics": dict(self.required_metrics),
            "blocked_metrics": dict(self.blocked_metrics),
            "message": self.message,
        }


CoordinationCompetitorAdapter = Callable[[CoordinationBenchCase], CoordinationBenchMetrics]


@dataclass(frozen=True)
class CoordinationBenchCaseResult:
    """Per-case benchmark output."""

    workload_case: CoordinationBenchCase
    gold: CoordinationBenchGold
    brief: CoordinationBrief
    metrics: CoordinationBenchMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.workload_case.to_dict(),
            "brief": self.brief.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class CoordinationBenchReport:
    """Full CoordinationBench report."""

    version: str
    workload_fingerprint: str
    metrics: CoordinationBenchMetrics
    cases: list[CoordinationBenchCaseResult]
    baselines: dict[str, CoordinationBenchMetrics]
    competitor_adapters: dict[str, CoordinationCompetitorAdapterDisclosure]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "workload_fingerprint": self.workload_fingerprint,
            "metrics": self.metrics.to_dict(),
            "baselines": {
                name: metrics.to_dict() for name, metrics in sorted(self.baselines.items())
            },
            "competitor_adapters": {
                name: disclosure.to_dict()
                for name, disclosure in sorted(self.competitor_adapters.items())
            },
            "coordinate_purpose_synthesis_gate": coordination_purpose_synthesis_gate(self).to_dict(),
            "competitor_claim_gate": coordination_competitor_claim_gate(self).to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


def build_coordination_workload(path: Path, *, missions: int = 1, workers: int = 3) -> CoordinationBenchWorkload:
    """Write and return the deterministic CoordinationBench workload."""
    if workers < 3 or workers > 10:
        raise ValueError("workers must be between 3 and 10")
    if missions != 1:
        raise ValueError("CoordinationBench MVP supports exactly one mission")
    cases = [_coordination_case(workers=workers)]
    body = {"version": COORDINATION_WORKLOAD_VERSION, "cases": [case.to_dict() for case in cases]}
    fingerprint = _fingerprint(body)
    workload = CoordinationBenchWorkload(
        version=COORDINATION_WORKLOAD_VERSION,
        cases=cases,
        fingerprint=fingerprint,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return workload


def load_coordination_workload(path: Path) -> CoordinationBenchWorkload:
    """Load and verify a frozen CoordinationBench workload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CoordinationBench workload must be a JSON object")
    workload = CoordinationBenchWorkload.from_dict(payload)
    body = {"version": workload.version, "cases": [case.to_dict() for case in workload.cases]}
    if workload.fingerprint != _fingerprint(body):
        raise ValueError("CoordinationBench workload fingerprint does not match contents")
    return workload


def run_coordination_benchmark(
    output_dir: Path,
    *,
    missions: int = 1,
    workers: int = 3,
    workload_path: Path | None = None,
    competitor_results: dict[str, Path] | None = None,
    competitor_runners: dict[str, Path] | None = None,
) -> CoordinationBenchReport:
    """Run Zaxy Coordinate against a generated or frozen workload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_output = output_dir / "coordination-workload.json"
    if workload_path is None:
        workload = build_coordination_workload(workload_output, missions=missions, workers=workers)
    else:
        workload = load_coordination_workload(workload_path)
        workload_output.write_text(
            json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    case_results = [_run_case(output_dir, case) for case in workload.cases]
    metrics = _mean_metrics([case.metrics for case in case_results])
    competitor_adapters = coordination_competitor_adapter_disclosures()
    for name, result_path in (competitor_results or {}).items():
        competitor_adapters[name] = run_coordination_competitor_adapter(
            name,
            workload,
            result_path=result_path,
        )
    for name, manifest_path in (competitor_runners or {}).items():
        competitor_adapters[name] = run_coordination_competitor_runner(
            name,
            workload,
            manifest_path=manifest_path,
            output_dir=output_dir / "competitor-runners",
        )
    report = CoordinationBenchReport(
        version=workload.version,
        workload_fingerprint=workload.fingerprint,
        metrics=metrics,
        cases=case_results,
        baselines=_baseline_report_metrics(workload.cases),
        competitor_adapters=competitor_adapters,
    )
    write_coordination_benchmark_report(report, output_dir)
    return report


def score_coordination_brief(
    brief: CoordinationBrief,
    gold: CoordinationBenchGold,
    *,
    eventloom_replayable: bool = True,
    brief_latency_ms: float = 0.0,
    promotion_latency_ms: float = 0.0,
) -> CoordinationBenchMetrics:
    """Score a coordination brief against exact gold labels."""
    accepted_claims = {
        finding.claim_key: finding.claim_value
        for finding in brief.accepted_findings
        if finding.claim_key and finding.claim_value
    }
    accepted_expected = {
        key: value
        for key, value in accepted_claims.items()
        if gold.expected_accepted_claims.get(key) == value
    }
    accepted_total = len(accepted_claims)
    expected_total = len(gold.expected_accepted_claims)
    conflict_pairs = {
        tuple(sorted((left.finding_id, right.finding_id)))
        for conflict in brief.conflicts
        for left in conflict.findings
        for right in conflict.findings
        if left.finding_id < right.finding_id
    }
    expected_pairs = {tuple(sorted(pair)) for pair in gold.expected_conflict_pairs}
    expected_stale = set(gold.expected_stale_findings)
    stale_accepted = {
        finding.finding_id
        for finding in brief.accepted_findings
        if finding.finding_id in expected_stale
    }
    stale_detected = {finding.finding_id for finding in brief.stale_findings}
    duplicate_ok = _duplicate_consolidation(brief, gold)
    accepted_with_evidence = [finding for finding in brief.accepted_findings if finding.evidence]
    accepted_with_citations = [
        finding for finding in brief.accepted_findings if finding.source_event_hash and finding.source_event_seq
    ]
    answerability = _answerability(accepted_claims, gold)
    returned_text = json.dumps(brief.to_dict(), sort_keys=True)
    return CoordinationBenchMetrics(
        accepted_finding_precision=_ratio(len(accepted_expected), accepted_total),
        accepted_finding_recall=_ratio(len(accepted_expected), expected_total),
        conflict_precision=_ratio(len(conflict_pairs & expected_pairs), len(conflict_pairs)),
        conflict_recall=_ratio(len(conflict_pairs & expected_pairs), len(expected_pairs)),
        stale_claim_rejection=1.0 if not stale_accepted and expected_stale <= stale_detected else 0.0,
        duplicate_consolidation=duplicate_ok,
        evidence_coverage=_ratio(len(accepted_with_evidence), len(brief.accepted_findings)),
        parent_checkout_answerability=answerability,
        citation_coverage=_ratio(len(accepted_with_citations), len(brief.accepted_findings)),
        eventloom_replayable=eventloom_replayable,
        returned_tokens=_approx_tokens(returned_text),
        injected_tokens=_approx_tokens(returned_text),
        brief_latency_ms=round(brief_latency_ms, 3),
        promotion_latency_ms=round(promotion_latency_ms, 3),
    )


def flat_eventlog_baseline_metrics(case: CoordinationBenchCase) -> CoordinationBenchMetrics:
    """Score a naive baseline that treats every reported worker finding as accepted."""
    findings = [finding for worker in case.workers for finding in worker["findings"]]
    accepted_claims = {
        str(finding.get("claim_key")): str(finding.get("claim_value"))
        for finding in findings
        if finding.get("claim_key") and finding.get("claim_value")
    }
    correct = sum(
        1 for key, value in accepted_claims.items() if case.gold.expected_accepted_claims.get(key) == value
    )
    stale_accepted = any(finding["finding_id"] in case.gold.expected_stale_findings for finding in findings)
    return CoordinationBenchMetrics(
        accepted_finding_precision=_ratio(correct, len(findings)),
        accepted_finding_recall=_ratio(correct, len(case.gold.expected_accepted_claims)),
        conflict_precision=0.0,
        conflict_recall=0.0,
        stale_claim_rejection=0.0 if stale_accepted else 1.0,
        duplicate_consolidation=0.0,
        evidence_coverage=_ratio(sum(1 for finding in findings if finding.get("evidence")), len(findings)),
        parent_checkout_answerability=0.0,
        citation_coverage=0.0,
        eventloom_replayable=True,
        returned_tokens=_approx_tokens(json.dumps(findings, sort_keys=True)),
        injected_tokens=_approx_tokens(json.dumps(findings, sort_keys=True)),
        brief_latency_ms=0.0,
        promotion_latency_ms=0.0,
        accepted_state_synthesis_quality=0.0,
        non_authoritative_leakage=0.0,
        purpose_feedback_coverage=0.0,
    )


def coordination_baseline_metrics(case: CoordinationBenchCase, baseline_name: str) -> CoordinationBenchMetrics:
    """Score a named same-harness local CoordinationBench baseline."""
    if baseline_name == "flat_transcript":
        return flat_eventlog_baseline_metrics(case)
    if baseline_name == "markdown_notes":
        return _markdown_notes_baseline_metrics(case)
    if baseline_name == "bm25_worker_logs":
        return _bm25_worker_logs_baseline_metrics(case)
    raise ValueError(f"Unknown CoordinationBench baseline: {baseline_name}")


def coordination_competitor_adapter_disclosures() -> dict[str, CoordinationCompetitorAdapterDisclosure]:
    """Return stable disclosures for external adapters not run in this harness."""
    return {
        name: CoordinationCompetitorAdapterDisclosure(
            name=name,
            display_name=str(spec["display_name"]),
            adapter_contract=str(spec["adapter_contract"]),
            status="not_run",
            claim_status="disclosure_only",
            blockers=tuple(str(blocker) for blocker in spec["blockers"]),
            metrics=None,
        )
        for name, spec in sorted(COORDINATION_COMPETITOR_ADAPTERS.items())
    }


def coordination_competitor_runner_manifest_templates(
    workload: CoordinationBenchWorkload,
) -> dict[str, dict[str, Any]]:
    """Return fingerprint-bound runner manifest templates for external adapters.

    These templates are intentionally not executable. Adapter authors must
    replace the placeholder run command and remove the template marker before
    Zaxy will run a competitor adapter.
    """
    manifests: dict[str, dict[str, Any]] = {}
    for name, spec in sorted(COORDINATION_COMPETITOR_ADAPTERS.items()):
        manifests[name] = {
            "name": name,
            "display_name": str(spec["display_name"]),
            "adapter_contract": "coordinationbench-v1",
            "adapter_version": "replace-with-pinned-version",
            "install_command": "replace-with-reproducible-install-command",
            "run_command": [COORDINATION_COMPETITOR_RUNNER_PLACEHOLDER],
            "source_url": "replace-with-adapter-source-url",
            "source_ref": "replace-with-pinned-source-ref",
            "workload_fingerprint": workload.fingerprint,
            "workload_file": "coordination-workload.json",
            "result_file": f"{name}-coordination-result.json",
            "template": True,
        }
    return manifests


def validate_coordination_competitor_runner_manifest(
    name: str,
    workload: CoordinationBenchWorkload,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate a runner manifest without executing the adapter."""
    manifest = _load_competitor_runner_manifest(name, workload, manifest_path)
    return {
        "valid": True,
        "name": name,
        "adapter_contract": manifest["adapter_contract"],
        "workload_fingerprint": manifest["workload_fingerprint"],
        "run_command": manifest["run_command"],
        "manifest": manifest,
    }


def validate_coordination_competitor_result(
    name: str,
    workload: CoordinationBenchWorkload,
    result_path: Path,
) -> dict[str, Any]:
    """Validate and locally score a competitor result file without trusting supplied metrics."""
    metrics, audit = _score_competitor_result_file(name, workload, result_path)
    return {
        "valid": True,
        "name": name,
        "adapter_contract": "coordinationbench-v1",
        "workload_fingerprint": workload.fingerprint,
        "metrics": metrics.to_dict(),
        "audit": audit.to_dict(),
    }


def coordination_competitor_claim_gate(
    report: CoordinationBenchReport,
    *,
    required_adapters: tuple[str, ...] = ("quarq", "hybi"),
) -> CoordinationCompetitorClaimGate:
    """Return whether public same-harness claims are supported for adapters.

    Disclosure-only rows remain valid benchmark metadata, but they are not
    evidence for a public same-harness comparison claim. This guardrail requires
    each named adapter to be completed, locally scored by Zaxy, and backed by a
    result audit with pinned manifest provenance.
    """
    completed: list[str] = []
    blocked: dict[str, str] = {}
    for name in required_adapters:
        disclosure = report.competitor_adapters.get(name)
        if disclosure is None:
            blocked[name] = "adapter row is missing from the report"
            continue
        if disclosure.status != "completed" or disclosure.claim_status != "same_harness":
            blocked[name] = (
                f"adapter status is {disclosure.status}/{disclosure.claim_status}; "
                "same-harness public claims require completed locally scored results"
            )
            continue
        if disclosure.metrics is None:
            blocked[name] = "completed adapter row has no locally scored metrics"
            continue
        audit = disclosure.result_audit
        if audit is None:
            blocked[name] = "completed adapter row has no result audit provenance"
            continue
        manifest = audit.manifest
        missing_manifest_fields = [
            field
            for field in COORDINATION_COMPETITOR_MANIFEST_FIELDS
            if not str(manifest.get(field) or "").strip()
        ]
        if missing_manifest_fields:
            blocked[name] = "result audit manifest is missing: " + ", ".join(missing_manifest_fields)
            continue
        if not audit.result_fingerprint:
            blocked[name] = "result audit is missing result_fingerprint"
            continue
        completed.append(name)
    if blocked:
        return CoordinationCompetitorClaimGate(
            status="blocked",
            required_adapters=tuple(required_adapters),
            completed_adapters=tuple(completed),
            blocked_adapters=blocked,
            message=(
                "Public same-harness competitor claims are blocked until required "
                "adapters have completed, locally scored, fingerprinted results."
            ),
        )
    return CoordinationCompetitorClaimGate(
        status="passed",
        required_adapters=tuple(required_adapters),
        completed_adapters=tuple(completed),
        blocked_adapters={},
        message="Public same-harness competitor claims are supported by completed local scoring.",
    )


def coordination_purpose_synthesis_gate(report: CoordinationBenchReport) -> CoordinationPurposeSynthesisGate:
    """Return whether Coordinate purpose-conditioned synthesis is proof-backed."""
    required: dict[str, float | bool] = {
        "accepted_state_synthesis_quality": 1.0,
        "non_authoritative_leakage": 1.0,
        "purpose_feedback_coverage": 1.0,
        "citation_coverage": 1.0,
        "parent_checkout_answerability": 1.0,
        "eventloom_replayable": True,
    }
    values = report.metrics.to_dict()
    blocked: dict[str, str] = {}
    for key, expected in required.items():
        actual = values.get(key)
        if isinstance(expected, bool):
            if actual is not expected:
                blocked[key] = f"expected {expected}, got {actual}"
            continue
        if not isinstance(actual, int | float) or float(actual) < expected:
            blocked[key] = f"expected >= {expected}, got {actual}"
    if not report.cases:
        blocked["cases"] = "report has no scored cases"
    if blocked:
        return CoordinationPurposeSynthesisGate(
            status="blocked",
            required_metrics=required,
            blocked_metrics=blocked,
            message=(
                "Coordinate purpose/synthesis claims are blocked until accepted-state "
                "synthesis, purpose feedback, citations, replayability, parent checkout "
                "answerability, and non-authoritative leakage gates all pass."
            ),
        )
    return CoordinationPurposeSynthesisGate(
        status="passed",
        required_metrics=required,
        blocked_metrics={},
        message=(
            "Coordinate accepted-state synthesis is proof-backed with citations, "
            "Coordinate-purpose feedback, replayable Eventloom provenance, parent "
            "checkout answerability, and no non-authoritative worker-row leakage."
        ),
    )


def export_coordination_benchmark_adapter_kit(
    output_dir: Path,
    *,
    missions: int = 1,
    workers: int = 3,
) -> dict[str, Any]:
    """Export the CoordinationBench adapter contract kit for external runner authors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workload = build_coordination_workload(
        output_dir / "coordination-workload.json",
        missions=missions,
        workers=workers,
    )
    _copy_coordinationbench_resource("README.md", output_dir / "README.md")
    _copy_coordinationbench_resource(
        "schemas/runner-manifest.schema.json",
        output_dir / "schemas" / "runner-manifest.schema.json",
    )
    _copy_coordinationbench_resource(
        "schemas/result.schema.json",
        output_dir / "schemas" / "result.schema.json",
    )
    templates_dir = output_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in coordination_competitor_runner_manifest_templates(workload).items():
        (templates_dir / f"{name}.runner-manifest.template.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (templates_dir / f"{name}-result.template.json").write_text(
            json.dumps(_competitor_result_template(name, workload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "output_dir": str(output_dir),
        "workload_path": str(output_dir / "coordination-workload.json"),
        "workload_fingerprint": workload.fingerprint,
        "schemas": [
            str(output_dir / "schemas" / "runner-manifest.schema.json"),
            str(output_dir / "schemas" / "result.schema.json"),
        ],
        "templates": [
            str(path)
            for path in sorted((output_dir / "templates").glob("*.template.json"))
        ],
    }


def run_coordination_competitor_adapter(
    name: str,
    workload: CoordinationBenchWorkload,
    *,
    adapter: CoordinationCompetitorAdapter | None = None,
    result_path: Path | None = None,
) -> CoordinationCompetitorAdapterDisclosure:
    """Run a pinned same-harness competitor adapter or return disclosure-only status."""
    disclosures = coordination_competitor_adapter_disclosures()
    if name not in disclosures:
        raise ValueError(f"Unknown CoordinationBench competitor adapter: {name}")
    disclosure = disclosures[name]
    if result_path is not None:
        metrics, audit = _score_competitor_result_file(name, workload, result_path)
        return CoordinationCompetitorAdapterDisclosure(
            name=disclosure.name,
            display_name=disclosure.display_name,
            adapter_contract=disclosure.adapter_contract,
            status="completed",
            claim_status="same_harness",
            blockers=(),
            metrics=metrics,
            result_audit=audit,
        )
    if adapter is None:
        return disclosure
    metrics = _mean_metrics([adapter(case) for case in workload.cases])
    return CoordinationCompetitorAdapterDisclosure(
        name=disclosure.name,
        display_name=disclosure.display_name,
        adapter_contract=disclosure.adapter_contract,
        status="completed",
        claim_status="same_harness",
        blockers=(),
        metrics=metrics,
    )


def run_coordination_competitor_runner(
    name: str,
    workload: CoordinationBenchWorkload,
    *,
    manifest_path: Path,
    output_dir: Path,
    timeout_seconds: int = 300,
) -> CoordinationCompetitorAdapterDisclosure:
    """Execute a pinned competitor runner and score its generated result locally."""
    disclosures = coordination_competitor_adapter_disclosures()
    if name not in disclosures:
        raise ValueError(f"Unknown CoordinationBench competitor adapter: {name}")
    disclosure = disclosures[name]
    manifest = _load_competitor_runner_manifest(name, workload, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_path = output_dir / f"{name}-coordination-workload.json"
    result_path = output_dir / f"{name}-coordination-result.json"
    workload_path.write_text(
        json.dumps(workload.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = [*manifest["run_command"], "--workload", str(workload_path), "--output", str(result_path)]
    try:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"CoordinationBench competitor runner timed out for {name} after {timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"CoordinationBench competitor runner failed for {name}: {stderr}")
    stdout_path = output_dir / f"{name}-runner.stdout.txt"
    stderr_path = output_dir / f"{name}-runner.stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if not result_path.is_file():
        raise RuntimeError(f"CoordinationBench competitor runner did not write result: {result_path}")
    metrics, audit = _score_competitor_result_file(name, workload, result_path)
    return CoordinationCompetitorAdapterDisclosure(
        name=disclosure.name,
        display_name=disclosure.display_name,
        adapter_contract=disclosure.adapter_contract,
        status="completed",
        claim_status="same_harness",
        blockers=(),
        metrics=metrics,
        result_audit=replace(
            audit,
            runner_command=list(manifest["run_command"]),
            runner_returncode=completed.returncode,
            runner_stdout_path=str(stdout_path),
            runner_stderr_path=str(stderr_path),
            runner_stdout_sha256=_sha256_text(completed.stdout),
            runner_stderr_sha256=_sha256_text(completed.stderr),
        ),
    )


def _load_competitor_runner_manifest(
    name: str,
    workload: CoordinationBenchWorkload,
    manifest_path: Path,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("competitor runner manifest must be a JSON object")
    if str(payload.get("name") or "") != name:
        raise ValueError("competitor runner manifest name does not match requested adapter")
    if str(payload.get("adapter_contract") or "") != "coordinationbench-v1":
        raise ValueError("competitor runner manifest adapter_contract must be coordinationbench-v1")
    if str(payload.get("workload_fingerprint") or "") != workload.fingerprint:
        raise ValueError("competitor runner manifest workload_fingerprint does not match workload")
    if bool(payload.get("template")):
        raise ValueError("competitor runner template manifest must be finalized before execution")
    for field in COORDINATION_COMPETITOR_MANIFEST_FIELDS:
        if field == "run_command":
            continue
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"competitor runner manifest {field} must be nonempty")
    run_command = payload.get("run_command")
    if not isinstance(run_command, list) or not run_command or not all(isinstance(item, str) and item for item in run_command):
        raise ValueError("competitor runner manifest run_command must be a nonempty argv array")
    if any(item == COORDINATION_COMPETITOR_RUNNER_PLACEHOLDER for item in run_command):
        raise ValueError("competitor runner manifest contains placeholder run_command")
    return {**payload, "run_command": list(run_command)}


def _score_competitor_result_file(
    name: str,
    workload: CoordinationBenchWorkload,
    result_path: Path,
) -> tuple[CoordinationBenchMetrics, CoordinationCompetitorResultAudit]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("competitor result must be a JSON object")
    if str(payload.get("name") or "") != name:
        raise ValueError("competitor result name does not match requested adapter")
    if str(payload.get("adapter_contract") or "") != "coordinationbench-v1":
        raise ValueError("competitor result adapter_contract must be coordinationbench-v1")
    if str(payload.get("workload_fingerprint") or "") != workload.fingerprint:
        raise ValueError("competitor result workload_fingerprint does not match workload")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("competitor result manifest must be an object")
    if str(manifest.get("name") or "") != name:
        raise ValueError("competitor result manifest name does not match requested adapter")
    if str(manifest.get("adapter_contract") or "") != "coordinationbench-v1":
        raise ValueError("competitor result manifest adapter_contract must be coordinationbench-v1")
    for field in COORDINATION_COMPETITOR_MANIFEST_FIELDS:
        if not str(manifest.get(field) or "").strip():
            raise ValueError(f"competitor result manifest {field} must be nonempty")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("competitor result cases must be an array")
    by_case: dict[str, dict[str, Any]] = {}
    for index, case_payload in enumerate(cases):
        if not isinstance(case_payload, dict):
            raise ValueError(f"competitor result case output must be an object at index {index}")
        case_id = str(case_payload.get("case_id") or "")
        if case_id in by_case:
            raise ValueError(f"competitor result duplicate case output: {case_id}")
        by_case[case_id] = case_payload
    expected_case_ids = {case.case_id for case in workload.cases}
    unexpected = sorted(set(by_case) - expected_case_ids)
    if unexpected:
        raise ValueError(f"competitor result unexpected case output: {', '.join(unexpected)}")
    if set(by_case) != expected_case_ids:
        missing = sorted(expected_case_ids - set(by_case))
        raise ValueError(f"competitor result missing case output: {', '.join(missing)}")
    metrics = [
        _score_competitor_case_output(case, by_case[case.case_id])
        for case in workload.cases
    ]
    audit = CoordinationCompetitorResultAudit(
        result_fingerprint=_fingerprint(payload),
        generated_at_utc=str(payload.get("generated_at_utc") or "") or None,
        case_count=len(by_case),
        manifest=dict(manifest),
    )
    return _mean_metrics(metrics), audit


def _score_competitor_case_output(
    case: CoordinationBenchCase,
    payload: dict[str, Any],
) -> CoordinationBenchMetrics:
    known_finding_ids = _known_finding_ids(case)
    accepted = [
        _finding_state_from_competitor(case, finding, known_finding_ids)
        for finding in _dict_list(payload.get("accepted_findings"))
    ]
    stale = [
        _finding_state_from_competitor(case, finding, known_finding_ids)
        for finding in _dict_list(payload.get("stale_findings"))
    ]
    findings_by_id = {finding.finding_id: finding for finding in [*accepted, *stale]}
    conflicts = []
    for conflict_payload in _dict_list(payload.get("conflicts")):
        conflict_ids = _string_list(conflict_payload.get("finding_ids"))
        unknown_conflict_ids = sorted(set(conflict_ids) - known_finding_ids)
        if unknown_conflict_ids:
            raise ValueError(f"competitor result unknown finding_id: {', '.join(unknown_conflict_ids)}")
        conflict_findings = [
            findings_by_id.get(finding_id) or _stub_finding(case, finding_id)
            for finding_id in conflict_ids
        ]
        conflicts.append(
            ConflictState(
                claim_key=str(conflict_payload.get("claim_key") or ""),
                findings=conflict_findings,
                conflict_type=str(conflict_payload.get("conflict_type") or "exact_claim"),
                reason=str(conflict_payload.get("reason") or "") or None,
                source_reference=str(conflict_payload.get("source_reference") or "") or None,
            )
        )
    brief = CoordinationBrief(
        mission_id=case.mission_id,
        objective=case.objective,
        workers=[],
        accepted_findings=accepted,
        pending_findings=[],
        rejected_findings=[],
        deferred_findings=[],
        conflicted_findings=[],
        stale_findings=stale,
        conflicts=conflicts,
    )
    metrics = score_coordination_brief(
        brief,
        case.gold,
        eventloom_replayable=False,
        brief_latency_ms=_optional_float(payload.get("latency_ms")) or 0.0,
        promotion_latency_ms=0.0,
    )
    return replace(
        metrics,
        returned_tokens=_approx_tokens(str(payload.get("returned_text") or "")),
        injected_tokens=_approx_tokens(str(payload.get("injected_text") or "")),
        accepted_state_synthesis_quality=_accepted_state_synthesis_quality(case, payload),
        non_authoritative_leakage=_non_authoritative_leakage(case, payload),
        purpose_feedback_coverage=_purpose_feedback_coverage(case, payload),
    )


def _finding_state_from_competitor(
    case: CoordinationBenchCase,
    payload: dict[str, Any],
    known_finding_ids: set[str],
) -> FindingState:
    finding_id = str(payload.get("finding_id") or "")
    if finding_id not in known_finding_ids:
        raise ValueError(f"competitor result unknown finding_id: {finding_id}")
    source_event_seq = _optional_int(payload.get("source_event_seq"))
    source_event_hash = str(payload.get("source_event_hash") or "") or None
    if not _valid_event_citation(source_event_seq, source_event_hash):
        source_event_seq = None
        source_event_hash = None
    return FindingState(
        finding_id=finding_id,
        mission_id=case.mission_id,
        worker_id=str(payload.get("worker_id") or "competitor"),
        summary=str(payload.get("summary") or ""),
        evidence=_dict_list(payload.get("evidence")),
        confidence=_optional_float(payload.get("confidence")),
        status=str(payload.get("status") or "accepted"),  # type: ignore[arg-type]
        claim_key=str(payload.get("claim_key") or "") or None,
        claim_value=str(payload.get("claim_value") or "") or None,
        stale=bool(payload.get("stale", False)),
        superseded_by=str(payload.get("superseded_by") or "") or None,
        source_event_seq=source_event_seq,
        source_event_hash=source_event_hash,
    )


def _known_finding_ids(case: CoordinationBenchCase) -> set[str]:
    return {
        str(finding["finding_id"])
        for worker in case.workers
        for finding in worker["findings"]
    }


def _valid_event_citation(source_event_seq: int | None, source_event_hash: str | None) -> bool:
    return (
        source_event_seq is not None
        and source_event_seq > 0
        and source_event_hash is not None
        and re.fullmatch(r"[0-9a-f]{64}", source_event_hash) is not None
    )


def _stub_finding(case: CoordinationBenchCase, finding_id: str) -> FindingState:
    return FindingState(
        finding_id=finding_id,
        mission_id=case.mission_id,
        worker_id="competitor",
        summary="",
        evidence=[],
    )


def write_coordination_benchmark_report(report: CoordinationBenchReport, output_dir: Path) -> None:
    """Write JSON and markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coordination-benchmark.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_competitor_runner_manifest_templates(report, output_dir)
    metrics = report.metrics.to_dict()
    lines = ["# CoordinationBench", "", f"- version: `{report.version}`", f"- workload: `{report.workload_fingerprint}`", ""]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    synthesis_gate = coordination_purpose_synthesis_gate(report)
    lines.extend(["", "## Coordinate Purpose/Synthesis Gate", ""])
    lines.append(f"- status: `{synthesis_gate.status}`")
    lines.append(f"- message: {synthesis_gate.message}")
    lines.append("")
    lines.append("| Required metric | Required value |")
    lines.append("|-----------------|----------------|")
    for key, value in synthesis_gate.required_metrics.items():
        lines.append(f"| {key} | {value} |")
    if synthesis_gate.blocked_metrics:
        lines.append("")
        lines.append("| Metric | Blocker |")
        lines.append("|--------|---------|")
        for key, reason in sorted(synthesis_gate.blocked_metrics.items()):
            lines.append(f"| {key} | {reason} |")
    claim_gate = coordination_competitor_claim_gate(report)
    lines.extend(["", "## Competitor Claim Gate", ""])
    lines.append(f"- status: `{claim_gate.status}`")
    lines.append(f"- required adapters: `{', '.join(claim_gate.required_adapters)}`")
    lines.append(f"- completed adapters: `{', '.join(claim_gate.completed_adapters) or 'none'}`")
    lines.append(f"- message: {claim_gate.message}")
    if claim_gate.blocked_adapters:
        lines.append("")
        lines.append("| Adapter | Blocker |")
        lines.append("|---------|---------|")
        for name, reason in sorted(claim_gate.blocked_adapters.items()):
            display_name = COORDINATION_COMPETITOR_ADAPTERS.get(name, {}).get("display_name", name)
            lines.append(f"| {display_name} | {reason} |")
    lines.extend(["", "## Baselines", ""])
    lines.append(
        "| Baseline | Description | accepted_finding_precision | conflict_recall | "
        "stale_claim_rejection | duplicate_consolidation | parent_checkout_answerability | "
        "citation_coverage | injected_tokens | returned_tokens |"
    )
    lines.append("|----------|-------------|----------------------------|-----------------|-----------------------|"
                 "-------------------------|-------------------------------|-------------------|-----------------|"
                 "-----------------|")
    for name, baseline_metrics in sorted(report.baselines.items()):
        values = baseline_metrics.to_dict()
        description = COORDINATION_BASELINE_DESCRIPTIONS.get(name, "")
        lines.append(
            f"| {name} | {description} | {values['accepted_finding_precision']} | "
            f"{values['conflict_recall']} | {values['stale_claim_rejection']} | "
            f"{values['duplicate_consolidation']} | {values['parent_checkout_answerability']} | "
            f"{values['citation_coverage']} | {values['injected_tokens']} | {values['returned_tokens']} |"
        )
    completed = [item for item in report.competitor_adapters.values() if item.status == "completed"]
    if completed:
        lines.extend(["", "## Competitor Adapter Runs", ""])
        lines.append(
            "| Adapter | Contract | Status | Claim status | accepted_finding_precision | "
            "conflict_recall | citation_coverage | injected_tokens | returned_tokens | "
            "result_fingerprint | source_ref |"
        )
        lines.append("|---------|----------|--------|--------------|----------------------------|"
                     "-----------------|-------------------|-----------------|-----------------|"
                     "--------------------|------------|")
        for disclosure in sorted(completed, key=lambda item: item.name):
            values = disclosure.metrics.to_dict() if disclosure.metrics is not None else {}
            audit = disclosure.result_audit
            result_fingerprint = audit.result_fingerprint if audit is not None else ""
            source_ref = str(audit.manifest.get("source_ref") or "") if audit is not None else ""
            lines.append(
                f"| {disclosure.display_name} | {disclosure.adapter_contract} | {disclosure.status} | "
                f"{disclosure.claim_status} | {values.get('accepted_finding_precision')} | "
                f"{values.get('conflict_recall')} | {values.get('citation_coverage')} | "
                f"{values.get('injected_tokens')} | {values.get('returned_tokens')} | "
                f"{result_fingerprint} | {source_ref} |"
            )
    lines.extend(["", "## Competitor Adapter Disclosures", ""])
    lines.append("| Adapter | Contract | Status | Claim status | Blockers |")
    lines.append("|---------|----------|--------|--------------|----------|")
    for disclosure in sorted(report.competitor_adapters.values(), key=lambda item: item.name):
        if disclosure.status == "completed":
            continue
        blockers = "; ".join(disclosure.blockers)
        lines.append(
            f"| {disclosure.display_name} | {disclosure.adapter_contract} | "
            f"{disclosure.status} | {disclosure.claim_status} | {blockers} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "CoordinationBench is a coordination-specific benchmark, not a universal memory benchmark.",
            "It measures accepted-state precision, conflict handling, stale-claim rejection, evidence grounding, "
            "parent checkout answerability, and replayability for multi-agent coordination workflows.",
            "The report should not be used as a claim about generic document RAG, open-domain QA, or all memory systems.",
            "Competitor rows marked `disclosure_only` or `not_run` are adapter-status disclosures, not scores.",
            "",
            "## Reproduction",
            "",
            "Regenerate this report with the CoordinationBench CLI:",
            "",
            "```bash",
            f"zaxy coordinate benchmark --output-dir {output_dir.as_posix()} --workload "
            f"{(output_dir / 'coordination-workload.json').as_posix()} --json",
            "```",
            "",
            "For generated seed workloads, omit `--workload` and pass `--missions 1 --workers 3`.",
            "For external systems, replace disclosure-only templates with pinned runner manifests or pinned result files.",
            "",
        ]
    )
    (output_dir / "coordination-benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def _write_competitor_runner_manifest_templates(report: CoordinationBenchReport, output_dir: Path) -> None:
    workload = CoordinationBenchWorkload(
        version=report.version,
        cases=[case.workload_case for case in report.cases],
        fingerprint=report.workload_fingerprint,
    )
    manifest_dir = output_dir / "competitor-runner-manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in coordination_competitor_runner_manifest_templates(workload).items():
        (manifest_dir / f"{name}.runner-manifest.template.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _copy_coordinationbench_resource(relative_path: str, destination: Path) -> None:
    root = importlib.resources.files("zaxy.resources.coordinationbench")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text((root / relative_path).read_text(encoding="utf-8"), encoding="utf-8")


def _competitor_result_template(name: str, workload: CoordinationBenchWorkload) -> dict[str, Any]:
    spec = COORDINATION_COMPETITOR_ADAPTERS[name]
    return {
        "name": name,
        "adapter_contract": "coordinationbench-v1",
        "workload_fingerprint": workload.fingerprint,
        "generated_at_utc": "replace-with-generation-time",
        "manifest": {
            "name": name,
            "display_name": str(spec["display_name"]),
            "adapter_contract": "coordinationbench-v1",
            "adapter_version": "replace-with-pinned-version",
            "install_command": "replace-with-reproducible-install-command",
            "run_command": "replace-with-runner-entrypoint",
            "source_url": "replace-with-adapter-source-url",
            "source_ref": "replace-with-pinned-source-ref",
        },
        "cases": [
            {
                "case_id": case.case_id,
                "accepted_findings": [],
                "stale_findings": [],
                "conflicts": [],
                "returned_text": "",
                "injected_text": "",
                "latency_ms": 0.0,
            }
            for case in workload.cases
        ],
    }


def _baseline_report_metrics(cases: list[CoordinationBenchCase]) -> dict[str, CoordinationBenchMetrics]:
    return {
        name: _mean_metrics([coordination_baseline_metrics(case, name) for case in cases])
        for name in COORDINATION_BASELINE_DESCRIPTIONS
    }


def _markdown_notes_baseline_metrics(case: CoordinationBenchCase) -> CoordinationBenchMetrics:
    documents = _worker_finding_documents(case)
    findings = [document["finding"] for document in documents]
    rendered = _render_markdown_notes(case, documents)
    return _score_retrieved_worker_context(
        case,
        findings,
        injected_text=rendered,
        returned_text=rendered,
        eventloom_replayable=False,
    )


def _bm25_worker_logs_baseline_metrics(case: CoordinationBenchCase) -> CoordinationBenchMetrics:
    documents = _worker_finding_documents(case)
    query = " ".join(str(question["query"]) for question in case.gold.final_questions)
    ranked = _rank_worker_documents_bm25(documents, query)
    selected = ranked[: min(3, len(ranked))]
    rendered = _render_markdown_notes(case, selected)
    return _score_retrieved_worker_context(
        case,
        [document["finding"] for document in selected],
        injected_text=rendered,
        returned_text=rendered,
        eventloom_replayable=False,
    )


def _worker_finding_documents(case: CoordinationBenchCase) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for worker in case.workers:
        for finding in worker["findings"]:
            text = " ".join(
                [
                    str(worker["worker_id"]),
                    str(worker["assignment"]),
                    str(finding.get("summary", "")),
                    str(finding.get("claim_key", "")),
                    str(finding.get("claim_value", "")),
                    json.dumps(finding.get("evidence") or [], sort_keys=True),
                ]
            )
            documents.append({"worker": worker, "finding": finding, "text": text})
    return documents


def _render_markdown_notes(case: CoordinationBenchCase, documents: list[dict[str, Any]]) -> str:
    lines = [f"# {case.objective}", ""]
    for document in documents:
        worker = document["worker"]
        finding = document["finding"]
        evidence = finding.get("evidence") or []
        lines.extend(
            [
                f"## {worker['worker_id']} / {finding['finding_id']}",
                f"- assignment: {worker['assignment']}",
                f"- summary: {finding.get('summary', '')}",
                f"- claim: {finding.get('claim_key', '')}={finding.get('claim_value', '')}",
                f"- confidence: {finding.get('confidence', '')}",
                f"- evidence_count: {len(evidence)}",
            ]
        )
        for item in evidence:
            lines.append(f"- evidence: {json.dumps(item, sort_keys=True)}")
        lines.append("")
    return "\n".join(lines)


def _rank_worker_documents_bm25(documents: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_terms = _tokenize(query)
    if not query_terms:
        return documents
    tokenized = [_tokenize(str(document["text"])) for document in documents]
    document_count = len(tokenized)
    average_length = sum(len(terms) for terms in tokenized) / document_count if document_count else 0.0
    document_frequency = Counter(term for terms in tokenized for term in set(terms))
    query_counts = Counter(query_terms)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, terms in enumerate(tokenized):
        term_counts = Counter(terms)
        length = len(terms)
        score = 0.0
        for term, query_weight in query_counts.items():
            if term_counts[term] == 0:
                continue
            idf = math.log(1.0 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = term_counts[term] + 1.2 * (1.0 - 0.75 + 0.75 * (length / average_length))
            score += float(query_weight) * idf * ((term_counts[term] * 2.2) / denominator)
        scored.append((score, -index, documents[index]))
    return [document for _, _, document in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)]


def _score_retrieved_worker_context(
    case: CoordinationBenchCase,
    findings: list[dict[str, Any]],
    *,
    injected_text: str,
    returned_text: str,
    eventloom_replayable: bool,
) -> CoordinationBenchMetrics:
    correct_findings = [
        finding
        for finding in findings
        if case.gold.expected_accepted_claims.get(str(finding.get("claim_key"))) == str(finding.get("claim_value"))
    ]
    expected_claims_found = {
        str(finding.get("claim_key")): str(finding.get("claim_value"))
        for finding in correct_findings
    }
    stale_selected = any(str(finding["finding_id"]) in case.gold.expected_stale_findings for finding in findings)
    findings_with_evidence = [finding for finding in findings if finding.get("evidence")]
    return CoordinationBenchMetrics(
        accepted_finding_precision=_ratio(len(correct_findings), len(findings)),
        accepted_finding_recall=_ratio(len(expected_claims_found), len(case.gold.expected_accepted_claims)),
        conflict_precision=0.0,
        conflict_recall=0.0,
        stale_claim_rejection=0.0 if stale_selected else 1.0,
        duplicate_consolidation=1.0 if not case.gold.expected_duplicate_groups else 0.0,
        evidence_coverage=_ratio(len(findings_with_evidence), len(findings)),
        parent_checkout_answerability=_retrieved_answerability(findings, case.gold),
        citation_coverage=0.0,
        eventloom_replayable=eventloom_replayable,
        returned_tokens=_approx_tokens(returned_text),
        injected_tokens=_approx_tokens(injected_text),
        brief_latency_ms=0.0,
        promotion_latency_ms=0.0,
        accepted_state_synthesis_quality=0.0,
        non_authoritative_leakage=0.0,
    )


def _run_case(output_dir: Path, case: CoordinationBenchCase) -> CoordinationBenchCaseResult:
    manager = CoordinationManager(eventloom_path=output_dir / "eventloom")
    manager.start_mission(case.mission_id, objective=case.objective, actor="benchmark")
    for worker in case.workers:
        worker_id = str(worker["worker_id"])
        manager.create_worker(case.mission_id, worker_id, actor="benchmark")
        manager.assign(case.mission_id, worker_id, str(worker["assignment"]), actor="benchmark")
        for finding in worker["findings"]:
            result = manager.report_finding(
                case.mission_id,
                worker_id,
                summary=str(finding["summary"]),
                actor=worker_id,
                evidence=list(finding.get("evidence") or []),
                confidence=float(finding["confidence"]) if finding.get("confidence") is not None else None,
                claim_key=str(finding.get("claim_key")) if finding.get("claim_key") else None,
                claim_value=str(finding.get("claim_value")) if finding.get("claim_value") else None,
                finding_id=str(finding["finding_id"]),
            )
            if result.finding_id != finding["finding_id"]:
                raise RuntimeError("coordination finding ID mismatch")
    promotion_start = time.perf_counter()
    for finding_id in _accepted_finding_ids(case):
        manager.review_finding(case.mission_id, finding_id, status="accepted", actor="benchmark")
        manager.promote_finding(case.mission_id, finding_id, actor="benchmark")
    promotion_latency_ms = (time.perf_counter() - promotion_start) * 1000.0
    brief_start = time.perf_counter()
    brief = manager.brief(case.mission_id)
    brief_latency_ms = (time.perf_counter() - brief_start) * 1000.0
    metrics = score_coordination_brief(
        brief,
        case.gold,
        eventloom_replayable=_eventloom_replayable(manager, case),
        brief_latency_ms=brief_latency_ms,
        promotion_latency_ms=promotion_latency_ms,
    )
    metrics = replace(
        metrics,
        purpose_feedback_coverage=_brief_purpose_feedback_coverage(case, brief),
    )
    return CoordinationBenchCaseResult(workload_case=case, gold=case.gold, brief=brief, metrics=metrics)


def _coordination_case(*, workers: int) -> CoordinationBenchCase:
    worker_specs = [
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
    selected = worker_specs[:workers]
    for index in range(len(selected), workers):
        selected.append(
            {
                "worker_id": f"worker-extra-{index + 1}",
                "assignment": "Investigate adjacent auth signals",
                "findings": [],
            }
        )
    gold = CoordinationBenchGold(
        expected_accepted_claims={"auth.failure.cause": "expired-jwks-cache"},
        expected_conflict_pairs={("finding-api-jwks", "finding-ui-refresh")},
        expected_duplicate_groups={
            "auth.failure.cause=expired-jwks-cache": ("finding-api-jwks", "finding-docs-jwks")
        },
        expected_stale_findings=("finding-api-stale-flag",),
        expected_missing_evidence=("finding-no-evidence",),
        final_questions=(
            {
                "query": "What is the accepted cause of auth failures?",
                "expected_terms": ("expired-jwks-cache",),
                "forbidden_terms": ("missing-browser-refresh", "flag-missing"),
            },
        ),
    )
    return CoordinationBenchCase(
        case_id="coordination-case-1",
        mission_id="coordination-case-1",
        objective="Resolve conflicting auth failure findings.",
        workers=selected,
        gold=gold,
    )


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


def _accepted_finding_ids(case: CoordinationBenchCase) -> list[str]:
    accepted_values = set(case.gold.expected_accepted_claims.items())
    accepted_seen: set[tuple[Any, Any]] = set()
    ids: list[str] = []
    for worker in case.workers:
        for finding in worker["findings"]:
            claim = (finding.get("claim_key"), finding.get("claim_value"))
            if claim in accepted_values and claim not in accepted_seen:
                ids.append(str(finding["finding_id"]))
                accepted_seen.add(claim)
    return ids


def _duplicate_consolidation(brief: CoordinationBrief, gold: CoordinationBenchGold) -> float:
    accepted_ids = {finding.finding_id for finding in brief.accepted_findings}
    if not gold.expected_duplicate_groups:
        return 1.0
    good = 0
    for group in gold.expected_duplicate_groups.values():
        if len(accepted_ids & set(group)) == 1:
            good += 1
    return _ratio(good, len(gold.expected_duplicate_groups))


def _answerability(accepted_claims: dict[str, str], gold: CoordinationBenchGold) -> float:
    if any(accepted_claims.get(key) != value for key, value in gold.expected_accepted_claims.items()):
        return 0.0
    forbidden = {
        term
        for question in gold.final_questions
        for term in question.get("forbidden_terms", ())
    }
    if any(value in forbidden for value in accepted_claims.values()):
        return 0.0
    return 1.0


def _retrieved_answerability(findings: list[dict[str, Any]], gold: CoordinationBenchGold) -> float:
    claims = [
        (str(finding.get("claim_key")), str(finding.get("claim_value")))
        for finding in findings
        if finding.get("claim_key") and finding.get("claim_value")
    ]
    for key, value in gold.expected_accepted_claims.items():
        if (key, value) not in claims:
            return 0.0
    forbidden = {
        term
        for question in gold.final_questions
        for term in question.get("forbidden_terms", ())
    }
    if any(value in forbidden for _, value in claims):
        return 0.0
    return 1.0


def _accepted_state_synthesis_quality(case: CoordinationBenchCase, payload: dict[str, Any]) -> float:
    if not _has_supported_synthesis_proof(case, payload):
        return 0.0
    text = _synthesis_answer_text(payload)
    if not text:
        return 0.0
    normalized = text.casefold()
    expected_terms = _final_question_terms(case.gold, "expected_terms")
    forbidden_terms = _final_question_terms(case.gold, "forbidden_terms")
    if expected_terms and not all(term in normalized for term in expected_terms):
        return 0.0
    if any(term in normalized for term in forbidden_terms):
        return 0.0
    return 1.0


def _has_supported_synthesis_proof(case: CoordinationBenchCase, payload: dict[str, Any]) -> bool:
    accepted_ids = {
        str(finding.get("finding_id") or "")
        for finding in _dict_list(payload.get("accepted_findings"))
    }
    expected_claim_ids = {
        str(finding.get("finding_id") or "")
        for finding in _dict_list(payload.get("accepted_findings"))
        if case.gold.expected_accepted_claims.get(str(finding.get("claim_key"))) == str(finding.get("claim_value"))
    }
    support_ids = set(_string_list(payload.get("support_source_ids")))
    answer_candidate = payload.get("answer_candidate")
    if isinstance(answer_candidate, dict):
        support_ids.update(_string_list(answer_candidate.get("support_source_ids")))
    artifact = payload.get("synthesis_artifact")
    ledger_support_ids: set[str] = set()
    if isinstance(artifact, dict):
        for candidate in _dict_list(artifact.get("answer_candidates")):
            support_ids.update(_string_list(candidate.get("support_source_ids")))
        for row in _dict_list(artifact.get("ledger_rows")):
            source_group = str(row.get("source_group") or "")
            if source_group and not str(row.get("exclude_reason") or ""):
                ledger_support_ids.add(source_group)
    if not support_ids:
        return False
    required_ids = expected_claim_ids or accepted_ids
    if not required_ids:
        return False
    if not support_ids <= accepted_ids:
        return False
    if not (support_ids & required_ids):
        return False
    return not ledger_support_ids or support_ids <= ledger_support_ids


def _non_authoritative_leakage(case: CoordinationBenchCase, payload: dict[str, Any]) -> float:
    injected_count = _optional_int(payload.get("non_authoritative_rows_injected")) or 0
    if injected_count > 0:
        return 0.0
    text = " ".join(
        [
            _synthesis_answer_text(payload),
            str(payload.get("injected_text") or ""),
            str(payload.get("returned_text") or ""),
        ]
    ).casefold()
    forbidden_terms = _final_question_terms(case.gold, "forbidden_terms")
    if any(term in text for term in forbidden_terms):
        return 0.0
    return 1.0


def _purpose_feedback_coverage(case: CoordinationBenchCase, payload: dict[str, Any]) -> float:
    """Return whether result feedback ties accepted state to Coordinate purpose."""
    required_ids = _expected_accepted_finding_ids(case, _dict_list(payload.get("accepted_findings")))
    if not required_ids:
        return 0.0
    covered: set[str] = set()
    for event in _dict_list(payload.get("feedback_events")) + _dict_list(payload.get("memory_feedback")):
        finding_id = str(event.get("finding_id") or "")
        if finding_id not in required_ids:
            continue
        if not _feedback_has_coordinate_purpose(event):
            continue
        if not _feedback_has_accepted_authority(event):
            continue
        feedback = str(event.get("feedback") or "used").casefold().strip()
        if feedback and feedback not in {"used", "helpful"}:
            continue
        covered.add(finding_id)
    return _ratio(len(covered), len(required_ids))


def _brief_purpose_feedback_coverage(case: CoordinationBenchCase, brief: CoordinationBrief) -> float:
    """Return coverage for Zaxy's accepted-state feedback-ready parent rows."""
    required_ids = {
        finding.finding_id
        for finding in brief.accepted_findings
        if finding.claim_key and case.gold.expected_accepted_claims.get(finding.claim_key) == finding.claim_value
    }
    if not required_ids:
        return 0.0
    covered = {
        finding.finding_id
        for finding in brief.accepted_findings
        if finding.finding_id in required_ids
        and finding.source_event_seq is not None
        and bool(finding.source_event_hash)
    }
    return _ratio(len(covered), len(required_ids))


def _expected_accepted_finding_ids(
    case: CoordinationBenchCase,
    accepted_findings: list[dict[str, Any]],
) -> set[str]:
    return {
        str(finding.get("finding_id") or "")
        for finding in accepted_findings
        if case.gold.expected_accepted_claims.get(str(finding.get("claim_key"))) == str(finding.get("claim_value"))
    }


def _feedback_has_coordinate_purpose(event: dict[str, Any]) -> bool:
    purpose = event.get("purpose")
    if isinstance(purpose, dict):
        return str(purpose.get("profile") or "").casefold().strip() == "coordinate"
    return str(event.get("purpose_profile") or purpose or "").casefold().strip() == "coordinate"


def _feedback_has_accepted_authority(event: dict[str, Any]) -> bool:
    authority = str(
        event.get("authority_scope")
        or event.get("authority")
        or event.get("coordination_status")
        or ""
    ).casefold().strip().replace("_", "-")
    return authority in {"accepted", "parent-accepted", "parent-accepted-state", "promoted"}


def _synthesis_answer_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    answer_candidate = payload.get("answer_candidate")
    if isinstance(answer_candidate, dict):
        parts.append(str(answer_candidate.get("answer") or ""))
    artifact = payload.get("synthesis_artifact")
    if isinstance(artifact, dict):
        for candidate in _dict_list(artifact.get("answer_candidates")):
            parts.append(str(candidate.get("answer") or ""))
        parts.append(str(artifact.get("answer") or ""))
    return " ".join(part for part in parts if part)


def _final_question_terms(gold: CoordinationBenchGold, key: str) -> set[str]:
    return {
        str(term).casefold()
        for question in gold.final_questions
        for term in question.get(key, ())
        if str(term)
    }


def _eventloom_replayable(manager: CoordinationManager, case: CoordinationBenchCase) -> bool:
    session_ids = [case.mission_id, *(str(worker["worker_id"]) for worker in case.workers)]
    return all(manager.session_manager.replay(session_id).integrity.ok for session_id in session_ids)


def _mean_metrics(metrics: list[CoordinationBenchMetrics]) -> CoordinationBenchMetrics:
    if not metrics:
        raise ValueError("metrics must not be empty")
    return CoordinationBenchMetrics(
        accepted_finding_precision=_mean([item.accepted_finding_precision for item in metrics]),
        accepted_finding_recall=_mean([item.accepted_finding_recall for item in metrics]),
        conflict_precision=_mean([item.conflict_precision for item in metrics]),
        conflict_recall=_mean([item.conflict_recall for item in metrics]),
        stale_claim_rejection=_mean([item.stale_claim_rejection for item in metrics]),
        duplicate_consolidation=_mean([item.duplicate_consolidation for item in metrics]),
        evidence_coverage=_mean([item.evidence_coverage for item in metrics]),
        parent_checkout_answerability=_mean([item.parent_checkout_answerability for item in metrics]),
        citation_coverage=_mean([item.citation_coverage for item in metrics]),
        eventloom_replayable=all(item.eventloom_replayable for item in metrics),
        returned_tokens=round(_mean([item.returned_tokens for item in metrics])),
        injected_tokens=round(_mean([item.injected_tokens for item in metrics])),
        brief_latency_ms=_mean([item.brief_latency_ms for item in metrics]),
        promotion_latency_ms=_mean([item.promotion_latency_ms for item in metrics]),
        accepted_state_synthesis_quality=_mean([item.accepted_state_synthesis_quality for item in metrics]),
        non_authoritative_leakage=_mean([item.non_authoritative_leakage for item in metrics]),
        purpose_feedback_coverage=_mean([item.purpose_feedback_coverage for item in metrics]),
    )


def _fingerprint(body: dict[str, Any]) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(numerator / denominator, 6)


def _mean(values: list[float | int]) -> float:
    return round(sum(float(value) for value in values) / len(values), 6)


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
