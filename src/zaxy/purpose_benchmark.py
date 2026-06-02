"""Deterministic benchmark gates for purpose-conditioned memory."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from zaxy.checkout import build_checkout_feedback_payload
from zaxy.compaction import build_compaction_projection
from zaxy.event import EventLog
from zaxy.purpose import PurposeProfile, purpose_profile, purpose_retrieval_policy
from zaxy.query import build_retention_policy

PURPOSE_BENCHMARK_VERSION = "purpose-v1"
PURPOSE_PROFILES = ("coding", "review", "release", "security", "research", "coordinate")
PURPOSE_BENCHMARK_LANES = (
    "Purpose Recall",
    "Ontology Shift",
    "Consequence Retention",
    "Governed Forgetting",
    "Action Outcome Loop",
    "Cross-Role Citation",
    "Accepted-State Discipline",
)


@dataclass(frozen=True)
class PurposeBenchmarkLane:
    """One exact-scored purpose-memory benchmark lane."""

    name: str
    score: float
    threshold: float
    status: str
    measurement: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PurposeBenchmarkReport:
    """Machine-readable report for purpose-conditioned memory gates."""

    version: str
    generated_at: str
    status: str
    lane_count: int
    passed_lanes: int
    lanes: tuple[PurposeBenchmarkLane, ...]
    competitor_claim_status: str
    competitor_claim_blockers: tuple[str, ...]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "status": self.status,
            "lane_count": self.lane_count,
            "passed_lanes": self.passed_lanes,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "competitor_claim_status": self.competitor_claim_status,
            "competitor_claim_blockers": list(self.competitor_claim_blockers),
            "elapsed_ms": self.elapsed_ms,
        }


def run_purpose_benchmark() -> PurposeBenchmarkReport:
    """Run the exact purpose-memory benchmark contract."""
    started = time.perf_counter()
    lanes = (
        _purpose_recall_lane(),
        _ontology_shift_lane(),
        _consequence_retention_lane(),
        _governed_forgetting_lane(),
        _action_outcome_loop_lane(),
        _cross_role_citation_lane(),
        _accepted_state_discipline_lane(),
    )
    passed = sum(1 for lane in lanes if lane.status == "passed")
    status = "passed" if passed == len(lanes) else "failed"
    return PurposeBenchmarkReport(
        version=PURPOSE_BENCHMARK_VERSION,
        generated_at=datetime.now(UTC).isoformat(),
        status=status,
        lane_count=len(lanes),
        passed_lanes=passed,
        lanes=lanes,
        competitor_claim_status="blocked",
        competitor_claim_blockers=(
            "Semantic Reach and Quarq require pinned same-harness adapters before "
            "Zaxy can publish comparative SOTA claims.",
        ),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def write_purpose_benchmark_report(report: PurposeBenchmarkReport, output_dir: str | Path) -> dict[str, Path]:
    """Write JSON and Markdown purpose benchmark reports."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "purpose-benchmark.json"
    md_path = directory / "purpose-benchmark.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_purpose_benchmark_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def render_purpose_benchmark_markdown(report: PurposeBenchmarkReport) -> str:
    """Render the report in the docs benchmark style."""
    lines = [
        "# Purpose Benchmark",
        "",
        f"- Version: `{report.version}`",
        f"- Generated: `{report.generated_at}`",
        f"- Status: `{report.status}`",
        f"- Lanes: `{report.passed_lanes}/{report.lane_count}` passed",
        f"- Elapsed: `{report.elapsed_ms:.3f} ms`",
        "",
        "| Lane | Status | Score | Threshold | Measurement |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for lane in report.lanes:
        lines.append(
            f"| {lane.name} | {lane.status} | {lane.score:.3f} | "
            f"{lane.threshold:.3f} | {lane.measurement} |"
        )
    lines.extend(
        [
            "",
            "## Competitor Claim Status",
            "",
            f"- Status: `{report.competitor_claim_status}`",
        ]
    )
    for blocker in report.competitor_claim_blockers:
        lines.append(f"- Blocker: {blocker}")
    return "\n".join(lines) + "\n"


def _purpose_recall_lane() -> PurposeBenchmarkLane:
    profiles = [_profile(name) for name in PURPOSE_PROFILES]
    passed_profiles = [
        profile.profile
        for profile in profiles
        if _policy_for(profile).applied
        and set(profile.required_evidence).issubset(set(profile.to_dict().get("required_evidence", ())))
        and any(term in _policy_for(profile).emphasis_terms for term in profile.ontology_lens)
    ]
    return _lane(
        "Purpose Recall",
        len(passed_profiles) / len(profiles),
        1.0,
        "all purpose profiles apply recall floors and ontology evidence terms",
        {"passed_profiles": passed_profiles},
    )


def _ontology_shift_lane() -> PurposeBenchmarkLane:
    signatures = {
        profile.profile: (
            _policy_for(profile).scoring_profile,
            tuple(_policy_for(profile).emphasis_terms[:8]),
        )
        for profile in (_profile(name) for name in PURPOSE_PROFILES)
    }
    unique_signatures = set(signatures.values())
    return _lane(
        "Ontology Shift",
        len(unique_signatures) / len(signatures),
        0.75,
        "same source query resolves to distinct purpose-specific retrieval lenses",
        {"signatures": signatures},
    )


def _consequence_retention_lane() -> PurposeBenchmarkLane:
    required_terms = {
        "coding": {"failed_fixes", "test_results"},
        "review": {"blocking_risks", "review_decisions"},
        "release": {"gate_failures", "external_blockers"},
        "security": {"security_findings", "risk_acceptance"},
        "research": {"contradictions", "open_questions"},
        "coordinate": {"accepted_parent_state", "proof_packets"},
    }
    retained = {
        name: sorted(required_terms[name] & set(_profile(name).retain)) for name in PURPOSE_PROFILES
    }
    passed_profiles = [name for name, terms in retained.items() if len(terms) == len(required_terms[name])]
    return _lane(
        "Consequence Retention",
        len(passed_profiles) / len(PURPOSE_PROFILES),
        1.0,
        "profiles retain prior failures, accepted decisions, risks, and proof outcomes",
        {"retained_terms": retained},
    )


def _governed_forgetting_lane() -> PurposeBenchmarkLane:
    policy = build_retention_policy(
        SimpleNamespace(
            retention_policy="decay",
            retention_decay_half_life_days=30,
            retention_expired_weight=0.0,
        )
    )
    protected = {
        name: policy.purpose_decay_half_life_days.get(name, policy.decay_half_life_days)
        for name in PURPOSE_PROFILES
    }
    passed = (
        protected["coordinate"] >= 180
        and protected["security"] >= 180
        and protected["release"] >= 120
        and protected["review"] >= 90
        and policy.purpose_expired_weights["coordinate"] > policy.expired_weight
        and policy.purpose_expired_weights["security"] > policy.expired_weight
    )
    return _lane(
        "Governed Forgetting",
        1.0 if passed else 0.0,
        1.0,
        "decay mode downweights noise while protecting obligations and risk memory",
        {
            "purpose_decay_half_life_days": protected,
            "purpose_expired_weights": dict(policy.purpose_expired_weights),
        },
    )


def _action_outcome_loop_lane() -> PurposeBenchmarkLane:
    fact = {
        "content": "Retry the migration with lock timeout disabled",
        "entity_name": "migration retry",
        "entity_type": "decision",
        "citation": "eventloom://purpose/events/7#abc123",
        "source": "eventloom",
        "score": 0.91,
    }
    payload = build_checkout_feedback_payload(fact, "ship migration", purpose="coding")
    purpose = payload.get("purpose") if isinstance(payload, dict) else None
    passed = (
        isinstance(payload, dict)
        and payload.get("feedback") == "used"
        and isinstance(purpose, dict)
        and purpose.get("profile") == "coding"
        and purpose.get("expected_action") == "implement_or_verify"
    )
    return _lane(
        "Action Outcome Loop",
        1.0 if passed else 0.0,
        1.0,
        "feedback payload records useful-for-purpose outcome metadata",
        {"feedback_payload": payload or {}},
    )


def _cross_role_citation_lane() -> PurposeBenchmarkLane:
    fact = {
        "content": "Auth token rotation was accepted after review",
        "entity_name": "auth token rotation",
        "entity_type": "decision",
        "citation": "eventloom://purpose/events/9#def456",
        "source": "eventloom",
        "score": 0.88,
    }
    payloads = {
        name: build_checkout_feedback_payload(fact, "auth token rotation", purpose=name)
        for name in ("release", "security", "review")
    }
    citations = {payload.get("citation") for payload in payloads.values() if isinstance(payload, dict)}
    profiles = {
        payload.get("purpose", {}).get("profile")
        for payload in payloads.values()
        if isinstance(payload, dict) and isinstance(payload.get("purpose"), dict)
    }
    passed = len(citations) == 1 and profiles == {"release", "security", "review"}
    return _lane(
        "Cross-Role Citation",
        1.0 if passed else 0.0,
        1.0,
        "same cited evidence can create distinct role-specific useful memory",
        {"profiles": sorted(profiles), "citations": sorted(citations), "payloads": payloads},
    )


def _accepted_state_discipline_lane() -> PurposeBenchmarkLane:
    with tempfile.TemporaryDirectory(prefix="zaxy-purpose-bench-") as tmp:
        log = EventLog(Path(tmp) / "coordinate.jsonl")
        log.append(
            "coordination.finding.promoted",
            actor="lead",
            payload={
                "mission_id": "mission-auth",
                "finding_id": "finding-accepted",
                "status": "accepted",
                "summary": "Accepted parent state: keep token rotation migration.",
                "source": "docs/auth.md",
                "turn_index": 4,
            },
            thread="mission-auth",
        )
        log.append(
            "coordination.finding.reported",
            actor="worker",
            payload={
                "mission_id": "mission-auth",
                "worker_id": "worker-1",
                "finding_id": "finding-pending",
                "status": "pending",
                "summary": "Pending local claim: revert token rotation migration.",
                "source": "worker.md",
                "turn_index": 2,
            },
            thread="worker-1",
        )
        projection = build_compaction_projection(log, purpose="coordinate", max_records=1)
    accepted_ids = {identity for record in projection.records for identity in record.identities}
    passed = (
        projection.strategy == "coordinate_authoritative"
        and projection.consolidation_policy["suppressed_count"] == 1
        and "finding-accepted" in accepted_ids
        and "finding-pending" not in accepted_ids
    )
    return _lane(
        "Accepted-State Discipline",
        1.0 if passed else 0.0,
        1.0,
        "Coordinate projection keeps accepted parent state and suppresses pending worker rows",
        {
            "strategy": projection.strategy,
            "suppressed_count": projection.consolidation_policy["suppressed_count"],
            "record_identities": sorted(accepted_ids),
        },
    )


def _profile(name: str) -> PurposeProfile:
    return purpose_profile(name)


def _policy_for(profile: PurposeProfile) -> Any:
    return purpose_retrieval_policy(
        profile,
        "ship auth token rotation",
        prompt_limit=4,
        base_recall_limit=8,
    )


def _lane(
    name: str,
    score: float,
    threshold: float,
    measurement: str,
    evidence: dict[str, Any],
) -> PurposeBenchmarkLane:
    rounded = round(score, 6)
    return PurposeBenchmarkLane(
        name=name,
        score=rounded,
        threshold=threshold,
        status="passed" if rounded >= threshold else "failed",
        measurement=measurement,
        evidence=evidence,
    )
