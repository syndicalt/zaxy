"""Purpose-conditioned memory policy profiles.

Purpose profiles make the retrieval-time ontology explicit without changing the
Eventloom source of truth. They are intentionally small, deterministic objects
that checkout, feedback, coordination, and future retention policy can share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_TEXT_FIELDS = (
    "profile",
    "role",
    "task",
    "risk",
    "time_horizon",
    "expected_action",
    "permission_scope",
    "evidence_policy",
    "retention_policy",
)
_LIST_FIELDS = ("ontology_lens", "required_evidence", "retain", "suppress", "warnings")


@dataclass(frozen=True)
class PurposeProfile:
    """A stable retrieval-time lens for memory checkout and feedback."""

    profile: str = "general"
    role: str = "agent"
    task: str = "answer"
    risk: str = "normal"
    time_horizon: str = "turn"
    expected_action: str = "answer_or_continue"
    permission_scope: str = "session"
    evidence_policy: str = "cite_relevant_evidence"
    retention_policy: str = "preserve_relevant_current_state"
    ontology_lens: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    retain: tuple[str, ...] = ()
    suppress: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON-serializable purpose profile."""
        payload: dict[str, Any] = {
            "profile": self.profile,
            "role": self.role,
            "task": self.task,
            "risk": self.risk,
            "time_horizon": self.time_horizon,
            "expected_action": self.expected_action,
            "permission_scope": self.permission_scope,
            "evidence_policy": self.evidence_policy,
            "retention_policy": self.retention_policy,
        }
        for key in _LIST_FIELDS:
            values = list(getattr(self, key))
            if values:
                payload[key] = values
        return payload


@dataclass(frozen=True)
class PurposeRetrievalPolicy:
    """Deterministic retrieval-time controls derived from a purpose profile."""

    profile: str
    retrieval_query: str
    emphasis_terms: tuple[str, ...] = ()
    scoring_profile: str = "balanced"
    min_recall_limit: int = 0
    recall_multiplier: int = 1

    @property
    def applied(self) -> bool:
        """Return whether the policy changes retrieval behavior."""
        return bool(self.emphasis_terms or self.min_recall_limit)

    def to_diagnostics(self, *, base_recall_limit: int, resolved_recall_limit: int) -> dict[str, Any]:
        """Return an auditable retrieval policy payload."""
        return {
            "profile": self.profile,
            "applied": self.applied,
            "emphasis_terms": list(self.emphasis_terms),
            "scoring_profile": self.scoring_profile,
            "recall_multiplier": self.recall_multiplier,
            "min_recall_limit": self.min_recall_limit,
            "base_recall_limit": base_recall_limit,
            "resolved_recall_limit": resolved_recall_limit,
        }


_PRESETS: dict[str, PurposeProfile] = {
    "general": PurposeProfile(),
    "coding": PurposeProfile(
        profile="coding",
        role="coding-agent",
        task="implementation",
        risk="correctness-critical",
        time_horizon="current-change",
        expected_action="implement_or_verify",
        permission_scope="repo-local",
        evidence_policy="cite_current_facts_and_verification",
        retention_policy="preserve_invariants_failed_attempts_and_test_evidence",
        ontology_lens=("invariant", "regression", "prior_fix", "open_blocker", "test_evidence"),
        required_evidence=("current_fact_citation", "verification_or_source_citation"),
        retain=("architectural_constraints", "failed_fixes", "accepted_decisions", "test_results"),
        suppress=("superseded_context", "uncited_claim"),
    ),
    "review": PurposeProfile(
        profile="review",
        role="reviewer",
        task="review",
        risk="high",
        time_horizon="release",
        expected_action="approve_or_block",
        permission_scope="repo-local",
        evidence_policy="cited_current_facts_required",
        retention_policy="preserve_risks_decisions_and_verification",
        ontology_lens=("risk", "regression", "missing_test", "accepted_decision", "blocker"),
        required_evidence=("accepted_or_cited_fact", "verification_evidence"),
        retain=("blocking_risks", "accepted_findings", "test_results", "review_decisions"),
        suppress=("pending_unreviewed_claim", "superseded_context", "low_trust_inference"),
    ),
    "release": PurposeProfile(
        profile="release",
        role="release-manager",
        task="release-readiness",
        risk="high",
        time_horizon="release",
        expected_action="ship_block_or_defer",
        permission_scope="repo-local",
        evidence_policy="release_gate_evidence_required",
        retention_policy="preserve_release_gates_regressions_and_external_blockers",
        ontology_lens=(
            "release_gate",
            "regression",
            "changelog",
            "packaging",
            "deployment",
            "external_validation",
        ),
        required_evidence=("release_check_ref", "test_result_ref", "changelog_ref"),
        retain=("release_decisions", "gate_failures", "packaging_evidence", "external_blockers"),
        suppress=("draft_claim", "uncited_release_assertion", "stale_gate_result"),
        warnings=("Treat stale release-gate evidence as blocking until refreshed.",),
    ),
    "security": PurposeProfile(
        profile="security",
        role="security-reviewer",
        task="security-assessment",
        risk="critical",
        time_horizon="incident-or-release",
        expected_action="block_or_accept_risk",
        permission_scope="repo-local",
        evidence_policy="cited_security_evidence_required",
        retention_policy="preserve_threats_controls_findings_and_risk_acceptance",
        ontology_lens=(
            "secret",
            "credential",
            "auth",
            "authorization",
            "exposure",
            "vulnerability",
            "risk_acceptance",
        ),
        required_evidence=("source_citation", "mitigation_ref", "risk_owner_ref"),
        retain=("security_findings", "mitigations", "risk_acceptance", "secret_handling"),
        suppress=("uncited_safe_claim", "low_trust_inference", "stale_mitigation"),
        warnings=("Do not treat absence of retrieved findings as evidence of safety.",),
    ),
    "research": PurposeProfile(
        profile="research",
        role="researcher",
        task="evidence-synthesis",
        risk="normal",
        time_horizon="investigation",
        expected_action="synthesize_or_identify_gap",
        permission_scope="session",
        evidence_policy="multi_source_citations_preferred",
        retention_policy="preserve_claims_sources_contradictions_and_open_questions",
        ontology_lens=(
            "claim",
            "source",
            "contradiction",
            "method",
            "benchmark",
            "open_question",
        ),
        required_evidence=("source_citation", "method_ref", "contradiction_ref"),
        retain=("research_claims", "source_quality", "contradictions", "open_questions"),
        suppress=("single_source_overclaim", "uncited_claim", "stale_source"),
        warnings=("Separate confirmed findings from hypotheses and source gaps.",),
    ),
    "coordinate": PurposeProfile(
        profile="coordinate",
        role="coordinator",
        task="mission-state",
        risk="high",
        time_horizon="handoff",
        expected_action="brief_promote_or_handoff",
        permission_scope="mission-parent",
        evidence_policy="accepted_parent_state_with_citations_required",
        retention_policy="preserve_accepted_findings_conflicts_handoffs_and_proof_packets",
        ontology_lens=(
            "accepted_finding",
            "pending_diagnostic",
            "conflict",
            "stale_evidence",
            "handoff",
            "proof_packet",
        ),
        required_evidence=("promotion_event_ref", "review_event_ref", "source_event_ref"),
        retain=("accepted_parent_state", "proof_packets", "handoff_refs", "conflicts"),
        suppress=("worker_local_pending", "rejected_finding", "stale_unpromoted_finding"),
        warnings=("Pending worker-local rows are diagnostic only unless explicitly included.",),
    ),
}


_PURPOSE_RECALL_FLOORS = {
    "coding": 16,
    "review": 16,
    "release": 24,
    "security": 24,
    "research": 20,
    "coordinate": 24,
}

_PURPOSE_RECALL_MULTIPLIERS = {
    "coding": 3,
    "review": 3,
    "release": 4,
    "security": 4,
    "research": 3,
    "coordinate": 4,
}

_PURPOSE_SCORING_PROFILES = {
    "coding": "recall",
    "review": "precision",
    "release": "temporal",
    "security": "precision",
    "research": "recall",
    "coordinate": "recall",
}


def purpose_profile(value: PurposeProfile | dict[str, Any] | str | None = None) -> PurposeProfile:
    """Normalize caller input into a stable purpose profile."""
    if isinstance(value, PurposeProfile):
        return value
    if value is None:
        return _PRESETS["general"]
    if isinstance(value, str):
        preset = _PRESETS.get(_clean_text(value))
        if preset is None:
            return PurposeProfile(profile=_clean_text(value) or "custom")
        return preset
    if not isinstance(value, dict):
        raise TypeError("purpose must be a profile name, object, or mapping")
    base_name = _clean_text(value.get("profile") or value.get("preset") or "general")
    base = _PRESETS.get(base_name, _PRESETS["general"])
    payload = base.to_dict()
    for key in _TEXT_FIELDS:
        if key in value:
            payload[key] = _clean_text(value.get(key)) or str(payload.get(key) or "")
    for key in _LIST_FIELDS:
        if key in value:
            payload[key] = _string_tuple(value.get(key))
    return PurposeProfile(
        profile=str(payload["profile"]),
        role=str(payload["role"]),
        task=str(payload["task"]),
        risk=str(payload["risk"]),
        time_horizon=str(payload["time_horizon"]),
        expected_action=str(payload["expected_action"]),
        permission_scope=str(payload["permission_scope"]),
        evidence_policy=str(payload["evidence_policy"]),
        retention_policy=str(payload["retention_policy"]),
        ontology_lens=tuple(payload.get("ontology_lens", ())),
        required_evidence=tuple(payload.get("required_evidence", ())),
        retain=tuple(payload.get("retain", ())),
        suppress=tuple(payload.get("suppress", ())),
        warnings=tuple(payload.get("warnings", ())),
    )


def coordinate_purpose_profile() -> PurposeProfile:
    """Return the default purpose lens for Coordinate parent state."""
    return _PRESETS["coordinate"]


def purpose_retrieval_policy(
    profile: PurposeProfile,
    query: str,
    *,
    prompt_limit: int,
    base_recall_limit: int,
) -> PurposeRetrievalPolicy:
    """Build deterministic retrieval controls for a purpose-conditioned checkout."""
    if profile.profile == "general":
        return PurposeRetrievalPolicy(
            profile=profile.profile,
            retrieval_query=query,
        )
    terms = _purpose_retrieval_terms(profile)
    recall_multiplier = _PURPOSE_RECALL_MULTIPLIERS.get(profile.profile, 2)
    min_recall_limit = max(
        base_recall_limit,
        prompt_limit * recall_multiplier,
        _PURPOSE_RECALL_FLOORS.get(profile.profile, prompt_limit * 2),
    )
    retrieval_query = query
    if terms:
        retrieval_query = f"{query} purpose:{profile.profile} " + " ".join(terms)
    return PurposeRetrievalPolicy(
        profile=profile.profile,
        retrieval_query=retrieval_query,
        emphasis_terms=terms,
        scoring_profile=_PURPOSE_SCORING_PROFILES.get(profile.profile, "recall"),
        min_recall_limit=min_recall_limit,
        recall_multiplier=recall_multiplier,
    )


def _purpose_retrieval_terms(profile: PurposeProfile) -> tuple[str, ...]:
    values: list[str] = [
        profile.task,
        profile.expected_action,
        profile.evidence_policy,
        profile.retention_policy,
        *profile.ontology_lens,
        *profile.required_evidence,
        *profile.retain,
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        for raw in str(value or "").replace("-", "_").split():
            term = raw.strip(" :,;.").casefold()
            if not term or term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return tuple(terms[:24])


def _clean_text(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "-")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list | tuple):
        raw_items = list(value)
    else:
        return ()
    items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return tuple(items)
