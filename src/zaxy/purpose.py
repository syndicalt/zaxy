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


@dataclass(frozen=True)
class PurposeOntologyLens:
    """Retrieval-time graph ontology overlay for a purpose profile."""

    profile: str
    entity_roles: tuple[str, ...] = ()
    relationship_roles: tuple[str, ...] = ()
    edge_trust_multipliers: dict[str, float] | None = None
    suppress_rules: tuple[str, ...] = ()
    required_source_groups: tuple[str, ...] = ()

    @property
    def applied(self) -> bool:
        """Return whether the lens changes graph interpretation."""
        return bool(
            self.entity_roles
            or self.relationship_roles
            or self.edge_trust_multipliers
            or self.suppress_rules
            or self.required_source_groups
        )

    def path_multiplier(self, relation_types: tuple[str, ...] | list[str]) -> float:
        """Return a bounded trust multiplier for a traversal path."""
        multipliers = self.edge_trust_multipliers or {}
        if not multipliers or not relation_types:
            return 1.0
        multiplier = 1.0
        for relation in relation_types:
            normalized = _clean_text(relation).replace("-", "_")
            multiplier *= multipliers.get(normalized, 1.0)
        return max(0.25, min(2.5, multiplier))

    def matched_relationship_roles(self, relation_types: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Return lens relationship roles that matched a traversal path."""
        if not relation_types or not self.relationship_roles:
            return ()
        path_tokens = {
            token
            for relation in relation_types
            for token in str(relation or "").casefold().replace("-", "_").split("_")
            if token
        }
        matches = [
            role
            for role in self.relationship_roles
            if any(token in path_tokens for token in str(role).casefold().replace("-", "_").split("_"))
        ]
        return tuple(dict.fromkeys(matches))

    def matched_entity_roles(self, text: str) -> tuple[str, ...]:
        """Return lens entity roles that match text for checkout diagnostics."""
        if not text or not self.entity_roles:
            return ()
        tokens = {
            token
            for token in text.casefold().replace("-", "_").replace(":", "_").split("_")
            if token
        }
        words = set(text.casefold().replace("-", " ").replace("_", " ").split())
        matches = []
        for role in self.entity_roles:
            role_tokens = set(str(role).casefold().replace("-", "_").split("_"))
            if role_tokens & tokens or role_tokens & words:
                matches.append(role)
        return tuple(dict.fromkeys(matches))

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a JSON-serializable lens diagnostic payload."""
        return {
            "profile": self.profile,
            "applied": self.applied,
            "entity_roles": list(self.entity_roles),
            "relationship_roles": list(self.relationship_roles),
            "edge_trust_multipliers": dict(self.edge_trust_multipliers or {}),
            "suppress_rules": list(self.suppress_rules),
            "required_source_groups": list(self.required_source_groups),
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
    "support": PurposeProfile(
        profile="support",
        role="support-agent",
        task="customer-support",
        risk="customer-impact",
        time_horizon="case-or-escalation",
        expected_action="triage_escalate_or_apply_workaround",
        permission_scope="project-local",
        evidence_policy="customer_impact_and_workaround_evidence_required",
        retention_policy="preserve_escalations_workarounds_customer_impact_and_resolutions",
        ontology_lens=(
            "customer_impact",
            "escalation",
            "workaround",
            "resolution",
            "incident",
            "account_context",
        ),
        required_evidence=("customer_report_ref", "workaround_or_resolution_ref", "impact_ref"),
        retain=("escalation_context", "workaround_history", "customer_impact", "resolution_evidence"),
        suppress=("uncited_claim", "stale_workaround", "superseded_context"),
        warnings=("Do not treat a workaround as resolved without cited customer-impact evidence.",),
    ),
    "product": PurposeProfile(
        profile="product",
        role="product-operator",
        task="roadmap-planning",
        risk="strategic",
        time_horizon="planning-cycle",
        expected_action="prioritize_or_defer_with_evidence",
        permission_scope="project-local",
        evidence_policy="roadmap_signal_and_tradeoff_evidence_required",
        retention_policy="preserve_roadmap_signals_promises_tradeoffs_and_experiment_outcomes",
        ontology_lens=(
            "roadmap_signal",
            "customer_promise",
            "tradeoff",
            "experiment",
            "decision",
            "constraint",
        ),
        required_evidence=("roadmap_signal_ref", "tradeoff_ref", "experiment_or_customer_ref"),
        retain=("roadmap_signals", "customer_promises", "tradeoffs", "experiment_outcomes"),
        suppress=("single_source_overclaim", "uncited_claim", "stale_source"),
        warnings=("Separate roadmap commitments from hypotheses and exploratory signals.",),
    ),
    "sales": PurposeProfile(
        profile="sales",
        role="account-operator",
        task="account-followup",
        risk="commitment",
        time_horizon="account-cycle",
        expected_action="prepare_commitment_or_followup",
        permission_scope="project-local",
        evidence_policy="buyer_commitment_and_objection_evidence_required",
        retention_policy="preserve_commitments_objections_renewal_risks_and_account_context",
        ontology_lens=(
            "buyer_commitment",
            "objection",
            "renewal_risk",
            "account_context",
            "stakeholder",
            "followup",
        ),
        required_evidence=("buyer_ref", "commitment_ref", "objection_or_renewal_ref"),
        retain=("buyer_commitments", "buyer_objections", "renewal_blockers", "account_context"),
        suppress=("uncited_claim", "stale_commitment", "superseded_context"),
        warnings=("Do not present a buyer commitment without cited account-context evidence.",),
    ),
    "legal": PurposeProfile(
        profile="legal",
        role="legal-operator",
        task="obligation-review",
        risk="critical",
        time_horizon="obligation-window",
        expected_action="quote_obligation_or_require_review",
        permission_scope="project-local",
        evidence_policy="exact_wording_authority_and_date_required",
        retention_policy="preserve_obligations_approvals_deadlines_and_exceptions",
        ontology_lens=(
            "legal_obligation",
            "approval",
            "deadline",
            "exception",
            "authority",
            "exact_quote",
        ),
        required_evidence=("exact_quote_ref", "authority_ref", "date_or_deadline_ref"),
        retain=("legal_obligations", "approvals", "deadlines", "exceptions"),
        suppress=("uncited_claim", "paraphrased_obligation", "stale_authority", "low_trust_inference"),
        warnings=("Do not paraphrase legal obligations as final authority without exact cited wording.",),
    ),
    "executive": PurposeProfile(
        profile="executive",
        role="executive-operator",
        task="strategic-brief",
        risk="high",
        time_horizon="strategy-cycle",
        expected_action="summarize_risk_exception_or_pattern",
        permission_scope="project-local",
        evidence_policy="strategic_exception_and_risk_evidence_required",
        retention_policy="preserve_strategic_exceptions_market_patterns_risks_and_decisions",
        ontology_lens=(
            "strategic_exception",
            "market_pattern",
            "risk_summary",
            "decision",
            "metric",
            "owner",
        ),
        required_evidence=("decision_ref", "risk_or_metric_ref", "owner_or_source_ref"),
        retain=("strategic_exceptions", "market_patterns", "risk_summaries", "executive_decisions"),
        suppress=("single_source_overclaim", "uncited_claim", "stale_source", "low_trust_inference"),
        warnings=("Keep executive summaries tied to cited project memory, not broad company-brain claims.",),
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
    "support": 20,
    "product": 20,
    "sales": 20,
    "legal": 24,
    "executive": 24,
    "coordinate": 24,
}

_PURPOSE_RECALL_MULTIPLIERS = {
    "coding": 3,
    "review": 3,
    "release": 4,
    "security": 4,
    "research": 3,
    "support": 3,
    "product": 3,
    "sales": 3,
    "legal": 4,
    "executive": 4,
    "coordinate": 4,
}

_PURPOSE_SCORING_PROFILES = {
    "coding": "recall",
    "review": "precision",
    "release": "temporal",
    "security": "precision",
    "research": "recall",
    "support": "recall",
    "product": "temporal",
    "sales": "precision",
    "legal": "precision",
    "executive": "precision",
    "coordinate": "recall",
}

_PURPOSE_ONTOLOGY_LENSES: dict[str, PurposeOntologyLens] = {
    "coding": PurposeOntologyLens(
        profile="coding",
        entity_roles=("artifact", "symbol", "test", "task", "decision", "failure"),
        relationship_roles=("invariant", "regression", "prior_fix", "test_evidence", "failed_attempt"),
        edge_trust_multipliers={
            "tests_symbol": 1.25,
            "calls_symbol": 1.15,
            "implements_decision": 1.2,
            "failed_attempt": 1.35,
            "superseded_by": 0.6,
        },
        suppress_rules=("superseded_context", "uncited_claim"),
        required_source_groups=("source_citation", "verification_or_source_citation"),
    ),
    "review": PurposeOntologyLens(
        profile="review",
        entity_roles=("risk", "finding", "test", "decision", "regression"),
        relationship_roles=("risk", "regression", "missing_test", "accepted_decision", "blocker"),
        edge_trust_multipliers={
            "blocks_release": 1.4,
            "missing_test": 1.3,
            "accepted_decision": 1.2,
            "low_trust_inference": 0.55,
        },
        suppress_rules=("pending_unreviewed_claim", "superseded_context", "low_trust_inference"),
        required_source_groups=("accepted_or_cited_fact", "verification_evidence"),
    ),
    "release": PurposeOntologyLens(
        profile="release",
        entity_roles=("release_gate", "package", "test", "changelog", "deployment", "blocker"),
        relationship_roles=("release_gate", "packaging", "deployment", "external_validation", "blocker"),
        edge_trust_multipliers={
            "passes_release_gate": 1.35,
            "fails_release_gate": 1.45,
            "has_changelog": 1.2,
            "external_validation": 1.25,
            "stale_gate_result": 0.5,
        },
        suppress_rules=("draft_claim", "uncited_release_assertion", "stale_gate_result"),
        required_source_groups=("release_check_ref", "test_result_ref", "changelog_ref"),
    ),
    "security": PurposeOntologyLens(
        profile="security",
        entity_roles=("secret", "credential", "auth", "authorization", "vulnerability", "mitigation"),
        relationship_roles=("exposure", "mitigation", "risk_acceptance", "authorization", "credential"),
        edge_trust_multipliers={
            "exposes_secret": 1.6,
            "uses_credential": 1.35,
            "mitigates_risk": 1.25,
            "risk_accepted_by": 1.2,
            "stale_mitigation": 0.45,
        },
        suppress_rules=("uncited_safe_claim", "low_trust_inference", "stale_mitigation"),
        required_source_groups=("source_citation", "mitigation_ref", "risk_owner_ref"),
    ),
    "research": PurposeOntologyLens(
        profile="research",
        entity_roles=("claim", "source", "method", "benchmark", "contradiction", "question"),
        relationship_roles=("supports_claim", "contradicts_claim", "uses_method", "open_question"),
        edge_trust_multipliers={
            "supports_claim": 1.15,
            "contradicts_claim": 1.3,
            "uses_method": 1.15,
            "single_source_overclaim": 0.55,
        },
        suppress_rules=("single_source_overclaim", "uncited_claim", "stale_source"),
        required_source_groups=("source_citation", "method_ref", "contradiction_ref"),
    ),
    "support": PurposeOntologyLens(
        profile="support",
        entity_roles=("customer", "case", "impact", "workaround", "resolution", "incident"),
        relationship_roles=("customer_impact", "escalation", "workaround", "resolution", "incident"),
        edge_trust_multipliers={
            "records_customer_impact": 1.3,
            "has_workaround": 1.25,
            "resolves_case": 1.25,
            "stale_workaround": 0.45,
        },
        suppress_rules=("uncited_claim", "stale_workaround", "superseded_context"),
        required_source_groups=("customer_report_ref", "workaround_or_resolution_ref", "impact_ref"),
    ),
    "product": PurposeOntologyLens(
        profile="product",
        entity_roles=("roadmap", "promise", "tradeoff", "experiment", "decision", "constraint"),
        relationship_roles=("roadmap_signal", "customer_promise", "tradeoff", "experiment", "decision"),
        edge_trust_multipliers={
            "records_roadmap_signal": 1.25,
            "commits_customer_promise": 1.3,
            "captures_tradeoff": 1.2,
            "experiment_result": 1.25,
            "single_source_overclaim": 0.55,
        },
        suppress_rules=("single_source_overclaim", "uncited_claim", "stale_source"),
        required_source_groups=("roadmap_signal_ref", "tradeoff_ref", "experiment_or_customer_ref"),
    ),
    "sales": PurposeOntologyLens(
        profile="sales",
        entity_roles=("buyer", "commitment", "objection", "renewal", "account", "stakeholder"),
        relationship_roles=("buyer_commitment", "objection", "renewal_risk", "account_context", "followup"),
        edge_trust_multipliers={
            "records_buyer_commitment": 1.3,
            "raises_objection": 1.25,
            "blocks_renewal": 1.35,
            "stale_commitment": 0.45,
        },
        suppress_rules=("uncited_claim", "stale_commitment", "superseded_context"),
        required_source_groups=("buyer_ref", "commitment_ref", "objection_or_renewal_ref"),
    ),
    "legal": PurposeOntologyLens(
        profile="legal",
        entity_roles=("obligation", "approval", "deadline", "exception", "authority", "quote"),
        relationship_roles=("legal_obligation", "approval", "deadline", "exception", "authority"),
        edge_trust_multipliers={
            "quotes_obligation": 1.5,
            "approved_by_authority": 1.35,
            "sets_deadline": 1.35,
            "paraphrased_obligation": 0.35,
            "stale_authority": 0.35,
        },
        suppress_rules=("uncited_claim", "paraphrased_obligation", "stale_authority", "low_trust_inference"),
        required_source_groups=("exact_quote_ref", "authority_ref", "date_or_deadline_ref"),
    ),
    "executive": PurposeOntologyLens(
        profile="executive",
        entity_roles=("strategy", "exception", "market", "risk", "decision", "metric"),
        relationship_roles=("strategic_exception", "market_pattern", "risk_summary", "decision", "owner"),
        edge_trust_multipliers={
            "records_strategic_exception": 1.3,
            "summarizes_risk": 1.25,
            "uses_metric": 1.2,
            "single_source_overclaim": 0.55,
            "low_trust_inference": 0.55,
        },
        suppress_rules=("single_source_overclaim", "uncited_claim", "stale_source", "low_trust_inference"),
        required_source_groups=("decision_ref", "risk_or_metric_ref", "owner_or_source_ref"),
    ),
    "coordinate": PurposeOntologyLens(
        profile="coordinate",
        entity_roles=("mission", "finding", "conflict", "handoff", "proof_packet", "accepted_state"),
        relationship_roles=("accepted_state", "proof_packet", "handoff", "conflict", "promotion"),
        edge_trust_multipliers={
            "mission_has_proof_packet": 1.45,
            "proof_links_synthesis_artifact": 1.25,
            "artifact_has_answer_candidate": 1.2,
            "artifact_has_ledger_row": 1.2,
            "promotes_finding": 1.35,
            "worker_local_pending": 0.45,
            "rejected_finding": 0.35,
        },
        suppress_rules=("worker_local_pending", "rejected_finding", "stale_unpromoted_finding"),
        required_source_groups=("promotion_event_ref", "review_event_ref", "source_event_ref"),
    ),
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


def purpose_ontology_lens(
    profile: PurposeProfile | dict[str, Any] | str | None = None,
) -> PurposeOntologyLens:
    """Return the retrieval-time ontology overlay for a purpose profile."""
    normalized = purpose_profile(profile)
    if normalized.profile == "general":
        return PurposeOntologyLens(profile="general")
    preset = _PURPOSE_ONTOLOGY_LENSES.get(normalized.profile)
    if preset is not None:
        return preset
    return PurposeOntologyLens(
        profile=normalized.profile,
        entity_roles=normalized.ontology_lens,
        relationship_roles=normalized.ontology_lens,
        suppress_rules=normalized.suppress,
        required_source_groups=normalized.required_evidence,
    )


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
