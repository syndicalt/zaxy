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
from zaxy.context import Context
from zaxy.core import _apply_purpose_outcome_learning, _purpose_outcome_aggregates
from zaxy.event import EventLog
from zaxy.evidence import evaluate_evidence_policy
from zaxy.extract import extract
from zaxy.neutral import audit_ingestion_purpose_labels, build_purpose_projection_record
from zaxy.purpose import (
    PurposeProfile,
    purpose_ontology_lens,
    purpose_profile,
    purpose_retrieval_policy,
)
from zaxy.query import build_retention_policy

PURPOSE_BENCHMARK_VERSION = "purpose-v1"
PURPOSE_PROFILES = (
    "coding",
    "review",
    "release",
    "security",
    "research",
    "support",
    "product",
    "sales",
    "legal",
    "executive",
    "coordinate",
)
PURPOSE_BENCHMARK_LANES = (
    "Purpose Recall",
    "Ontology Shift",
    "Consequence Retention",
    "Governed Forgetting",
    "Action Outcome Loop",
    "Evidence Policy Discipline",
    "Broader Profile Fixtures",
    "Neutral Substrate Projection",
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
        _evidence_policy_discipline_lane(),
        _broader_profile_fixtures_lane(),
        _neutral_substrate_projection_lane(),
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
    source_paths = {
        "coding": ("failed_attempt", "tests_symbol"),
        "release": ("fails_release_gate", "has_changelog"),
        "security": ("uses_credential", "risk_accepted_by"),
        "coordinate": ("mission_has_proof_packet", "artifact_has_ledger_row"),
    }
    signatures = {
        profile.profile: (
            _policy_for(profile).scoring_profile,
            tuple(_policy_for(profile).emphasis_terms[:8]),
            tuple(purpose_ontology_lens(profile).relationship_roles[:4]),
        )
        for profile in (_profile(name) for name in PURPOSE_PROFILES)
    }
    path_overlays = {
        profile: {
            "multiplier": purpose_ontology_lens(profile).path_multiplier(path),
            "relationship_roles": list(purpose_ontology_lens(profile).matched_relationship_roles(path)),
        }
        for profile, path in source_paths.items()
    }
    unique_signatures = set(signatures.values())
    distinct_path_multipliers = {
        overlay["multiplier"]
        for overlay in path_overlays.values()
    }
    path_roles_present = all(overlay["relationship_roles"] for overlay in path_overlays.values())
    score = min(
        len(unique_signatures) / len(signatures),
        len(distinct_path_multipliers) / len(path_overlays),
        1.0 if path_roles_present else 0.0,
    )
    return _lane(
        "Ontology Shift",
        score,
        0.75,
        "same source query resolves to distinct purpose-specific retrieval lenses and graph path roles",
        {"signatures": signatures, "path_overlays": path_overlays},
    )


def _consequence_retention_lane() -> PurposeBenchmarkLane:
    required_terms = {
        "coding": {"failed_fixes", "test_results"},
        "review": {"blocking_risks", "review_decisions"},
        "release": {"gate_failures", "external_blockers"},
        "security": {"security_findings", "risk_acceptance"},
        "research": {"contradictions", "open_questions"},
        "support": {"workaround_history", "customer_impact"},
        "product": {"roadmap_signals", "experiment_outcomes"},
        "sales": {"buyer_commitments", "renewal_blockers"},
        "legal": {"legal_obligations", "deadlines"},
        "executive": {"strategic_exceptions", "risk_summaries"},
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
        and protected["support"] >= 90
        and protected["product"] >= 120
        and protected["sales"] >= 120
        and protected["legal"] >= 365
        and protected["executive"] >= 180
        and policy.purpose_expired_weights["coordinate"] > policy.expired_weight
        and policy.purpose_expired_weights["security"] > policy.expired_weight
        and policy.purpose_expired_weights["legal"] >= 0.2
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
    citation = "eventloom://purpose/events/7#abc123"
    fact = {
        "content": "Retry the migration with lock timeout disabled",
        "entity_name": "migration retry",
        "entity_type": "decision",
        "citation": citation,
        "source": "eventloom",
        "score": 0.91,
    }
    payload = build_checkout_feedback_payload(fact, "ship migration", purpose="coding")
    purpose = payload.get("purpose") if isinstance(payload, dict) else None
    profile = purpose_profile("coding")
    events = [
        SimpleNamespace(
            seq=1,
            type="memory.reinforced",
            payload={
                "entity_name": "migration retry",
                "entity_type": "decision",
                "citation": citation,
                "purpose": {"profile": "coding"},
                "outcome": "avoided_failed_path",
            },
        ),
        SimpleNamespace(
            seq=2,
            type="memory.reinforced",
            payload={
                "entity_name": "migration retry",
                "entity_type": "decision",
                "citation": citation,
                "purpose": {"profile": "coding"},
                "outcome": "prevented_redundant_investigation",
            },
        ),
        SimpleNamespace(
            seq=3,
            type="memory.feedback",
            payload={
                "entity_name": "stale migration retry",
                "entity_type": "decision",
                "citation": "eventloom://purpose/events/8#def456",
                "purpose": {"profile": "coding"},
                "feedback": "irrelevant",
                "outcome": "caused_regression",
            },
        ),
        SimpleNamespace(
            seq=4,
            type="memory.feedback",
            payload={
                "entity_name": "stale migration retry",
                "entity_type": "decision",
                "citation": "eventloom://purpose/events/8#def456",
                "purpose": {"profile": "coding"},
                "feedback": "irrelevant",
                "outcome": "failed",
            },
        ),
    ]
    aggregates = _purpose_outcome_aggregates(events, profile)
    learned = _apply_purpose_outcome_learning(
        [
            Context(
                content="Stale migration retry advice.",
                source="keyword",
                score=0.95,
                metadata={
                    "entity_name": "stale migration retry",
                    "entity_type": "decision",
                    "citation": "eventloom://purpose/events/8#def456",
                },
            ),
            Context(
                content="Retry the migration with lock timeout disabled.",
                source="keyword",
                score=0.9,
                metadata={
                    "entity_name": "migration retry",
                    "entity_type": "decision",
                    "citation": citation,
                },
            ),
        ],
        aggregates,
    )
    learned_explanations = [
        (context.metadata or {}).get("score_explanation", {}).get("purpose_outcome", {})
        for context in learned
    ]
    boosted = learned[0].metadata and learned[0].metadata.get("entity_name") == "migration retry"
    suppression_candidate = any(
        isinstance(explanation, dict) and explanation.get("suppression_candidate")
        for explanation in learned_explanations
    )
    passed = (
        isinstance(payload, dict)
        and payload.get("feedback") == "used"
        and isinstance(purpose, dict)
        and purpose.get("profile") == "coding"
        and purpose.get("expected_action") == "implement_or_verify"
        and boosted
        and suppression_candidate
    )
    return _lane(
        "Action Outcome Loop",
        1.0 if passed else 0.0,
        1.0,
        "purpose outcome history changes future rank and warning candidates",
        {
            "feedback_payload": payload or {},
            "boosted_context": learned[0].metadata.get("entity_name") if learned[0].metadata else None,
            "outcome_explanations": learned_explanations,
        },
    )


def _evidence_policy_discipline_lane() -> PurposeBenchmarkLane:
    fixtures = {
        "security": {
            "unsupported": "Credential exposure found in auth config.",
            "supported": "Credential exposure has source citation, mitigation, and risk owner accepted risk.",
            "missing": "mitigation_or_risk_owner",
        },
        "release": {
            "unsupported": "Release gate is green for the current candidate.",
            "supported": "Release gate is green with pytest test result, changelog entry, package build, and twine check.",
            "missing": "verification_refs",
        },
        "coordinate": {
            "unsupported": "Worker-local finding says auth cache is stale.",
            "supported": "Accepted parent state was promoted after review with source_event_seq and source_event_hash.",
            "missing": "promotion_or_review_ref",
        },
        "support": {
            "unsupported": "Customer case says the dashboard is broken.",
            "supported": "Customer ticket report has cited impact severity and a documented workaround resolution.",
            "missing": "workaround_or_resolution_ref",
        },
        "product": {
            "unsupported": "Roadmap should prioritize dashboard export.",
            "supported": "Roadmap signal from customer feedback includes tradeoff, experiment outcome, and customer promise.",
            "missing": "tradeoff_ref",
        },
        "sales": {
            "unsupported": "The account wants a follow-up.",
            "supported": "Buyer account stakeholder recorded commitment, next step followup, objection, renewal blocker, and budget risk.",
            "missing": "commitment_ref",
        },
        "legal": {
            "unsupported": "The contract allows redistribution.",
            "supported": "Exact quote from clause section is approved by counsel authority with effective date and deadline.",
            "missing": "exact_quote_ref",
        },
        "executive": {
            "unsupported": "There is a strategic exception.",
            "supported": "Executive decision approved strategic exception with owner, source, risk metric, market trend, and accountable sponsor.",
            "missing": "risk_or_metric_ref",
        },
    }
    evidence: dict[str, Any] = {}
    passed = 0
    for profile, fixture in fixtures.items():
        unsupported = _policy_fixture_result(profile, str(fixture["unsupported"]))
        supported = _policy_fixture_result(profile, str(fixture["supported"]))
        evidence[profile] = {
            "unsupported": unsupported,
            "supported": supported,
        }
        if (
            unsupported["satisfied"] is False
            and str(fixture["missing"]) in unsupported["missing_requirements"]
            and unsupported["mode"] in {"block_checkout", "require_refresh", "warn"}
            and unsupported["suggested_queries"]
            and supported["satisfied"] is True
        ):
            passed += 1
    return _lane(
        "Evidence Policy Discipline",
        passed / len(fixtures),
        1.0,
        "purpose fixtures enforce missing and supported evidence policies",
        evidence,
    )


def _broader_profile_fixtures_lane() -> PurposeBenchmarkLane:
    profiles = ("support", "product", "sales", "legal", "executive")
    compaction: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="zaxy-purpose-profile-bench-") as tmp:
        root = Path(tmp)
        for profile in profiles:
            log = EventLog(root / f"{profile}.jsonl")
            for index in range(3):
                log.append(
                    "document.indexed",
                    actor=f"{profile}-operator",
                    payload={
                        "path": f"{profile}/fixture-{index}.md",
                        "start_line": index + 1,
                        "end_line": index + 2,
                        "content": (
                            f"{profile} purpose fixture identity-code-{index:04d} "
                            f"records {' '.join(purpose_profile(profile).retain)}."
                        ),
                    },
                )
            projection = build_compaction_projection(log, purpose=profile, max_records=1)
            compaction[profile] = {
                "purpose": projection.purpose.get("profile"),
                "strategy": projection.strategy,
                "record_kinds": sorted({record.kind for record in projection.records}),
                "retain": projection.consolidation_policy.get("retain", []),
                "suppress": projection.consolidation_policy.get("suppress", []),
            }
    profile_payloads = {
        profile: purpose_profile(profile).to_dict()
        for profile in profiles
    }
    checkout_ready = {
        profile: {
            "has_evidence_policy": bool(payload.get("evidence_policy")),
            "has_retention_policy": bool(payload.get("retention_policy")),
            "has_retain": bool(payload.get("retain")),
            "has_suppress": bool(payload.get("suppress")),
            "lens_applied": purpose_ontology_lens(profile).applied,
        }
        for profile, payload in profile_payloads.items()
    }
    local_positioning = all(
        payload.get("permission_scope") == "project-local"
        for payload in profile_payloads.values()
    )
    passed_profiles = [
        profile
        for profile in profiles
        if all(checkout_ready[profile].values())
        and compaction[profile]["purpose"] == profile
        and compaction[profile]["record_kinds"]
    ]
    score = len(passed_profiles) / len(profiles)
    return _lane(
        "Broader Profile Fixtures",
        score if local_positioning else 0.0,
        1.0,
        "support product sales legal and executive profiles have checkout compaction and benchmark fixtures",
        {
            "passed_profiles": passed_profiles,
            "checkout_ready": checkout_ready,
            "compaction": compaction,
            "local_project_memory_positioning": local_positioning,
        },
    )


def _neutral_substrate_projection_lane() -> PurposeBenchmarkLane:
    with tempfile.TemporaryDirectory(prefix="zaxy-neutral-substrate-bench-") as tmp:
        log = EventLog(Path(tmp) / "customer-email.jsonl")
        event = log.append(
            "document.indexed",
            actor="support-agent",
            payload={
                "path": "customers/acme-email.txt",
                "start_line": 1,
                "end_line": 4,
                "content": (
                    "ACME reports dashboard export failures affecting renewal. "
                    "They ask whether the export clause applies to dashboard data "
                    "and want the roadmap promise reviewed by Friday."
                ),
                "permission_scope": "project-local",
                "uncertainty": "customer email requires purpose-specific review",
            },
            thread="customer-acme",
        )
        result = extract(event)
    neutral = next(entity for entity in result.entities if entity.entity_type == "neutral_substrate")
    audit = audit_ingestion_purpose_labels(event.payload, source_event_ref=f"eventloom://{event.thread}/events/{event.seq}#{event.hash}")
    projections = {
        "support": build_purpose_projection_record(
            {"substrate_id": neutral.name, **(neutral.properties or {})},
            purpose_profile="support",
            purpose_label="customer_escalation",
        ).to_dict(),
        "product": build_purpose_projection_record(
            {"substrate_id": neutral.name, **(neutral.properties or {})},
            purpose_profile="product",
            purpose_label="roadmap_commitment",
        ).to_dict(),
        "legal": build_purpose_projection_record(
            {"substrate_id": neutral.name, **(neutral.properties or {})},
            purpose_profile="legal",
            purpose_label="legal_obligation",
        ).to_dict(),
        "executive": build_purpose_projection_record(
            {"substrate_id": neutral.name, **(neutral.properties or {})},
            purpose_profile="executive",
            purpose_label="churn_risk",
        ).to_dict(),
    }
    source_refs = {projection["source_event_ref"] for projection in projections.values()}
    backpointers = {projection["source_backpointer"] for projection in projections.values()}
    labels = {projection["purpose_label"] for projection in projections.values()}
    passed = (
        audit.safe
        and neutral.properties is not None
        and neutral.properties.get("permission_scope") == "project-local"
        and source_refs == {f"eventloom://{event.thread}/events/{event.seq}#{event.hash}"}
        and backpointers == {"customers/acme-email.txt:1-4"}
        and labels == {"customer_escalation", "roadmap_commitment", "legal_obligation", "churn_risk"}
    )
    return _lane(
        "Neutral Substrate Projection",
        1.0 if passed else 0.0,
        1.0,
        "one neutral customer artifact can rebuild distinct cited purpose projections",
        {
            "neutral_substrate": {
                "name": neutral.name,
                "properties": neutral.properties,
            },
            "ingestion_audit": audit.to_dict(),
            "purpose_projections": projections,
        },
    )


def _policy_fixture_result(profile: str, content: str) -> dict[str, Any]:
    fact = {
        "content": content,
        "source": "graph",
        "citation": f"eventloom://purpose-policy/events/{profile}#abcdefabcdef",
    }
    result = evaluate_evidence_policy(
        profile=profile,
        query=f"{profile} evidence policy fixture",
        current_facts=[fact],
        evidence=[fact],
    )
    if result is None:
        return {"satisfied": False, "mode": "missing", "missing_requirements": [], "suggested_queries": []}
    diagnostics = result.to_diagnostics()
    return {
        "satisfied": diagnostics["satisfied"],
        "mode": diagnostics["mode"],
        "missing_requirements": diagnostics["missing_requirements"],
        "suggested_queries": diagnostics["suggested_queries"],
    }


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
