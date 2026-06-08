"""Reasoning-loop memory primitive contracts.

These helpers define observable, phase-conditioned memory calls for agent
planning, execution, review, and reflection. They deliberately do not promote
generated claims to authority; belief updates are proposal events only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from zaxy.purpose import PurposeProfile

REASONING_PHASES = {"planning", "execution", "review", "reflection"}

_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENTLOOM_CITATION_RE = re.compile(
    r"^eventloom://[^/\s]+/events/[1-9][0-9]*#(?:[0-9a-f]{12}|[0-9a-f]{64})$"
)
_STATUSES = {"succeeded", "failed"}


def validate_reasoning_phase(value: object) -> str:
    """Return a normalized reasoning phase or raise for unknown phases."""
    if not isinstance(value, str):
        raise TypeError("reasoning phase must be a string")
    phase = value.strip().casefold()
    if phase not in REASONING_PHASES:
        raise ValueError("reasoning phase must be one of: " + ", ".join(sorted(REASONING_PHASES)))
    return phase


def phase_purpose_profile(phase: str) -> PurposeProfile:
    """Return the purpose profile used to condition retrieval for a phase."""
    normalized = validate_reasoning_phase(phase)
    profiles = {
        "planning": PurposeProfile(
            profile="reasoning-planning",
            role="reasoning-agent",
            task="planning",
            risk="normal",
            time_horizon="current-task",
            expected_action="choose_next_steps",
            permission_scope="session",
            evidence_policy="cite_relevant_precedents_and_constraints",
            retention_policy="preserve_reusable_plans_and_open_constraints",
            ontology_lens=("goal", "constraint", "procedure", "prior_outcome"),
            required_evidence=("source_event_ref", "citation"),
            retain=("successful_procedures", "constraints", "open_blockers"),
            suppress=("rejected_candidate", "stale_context", "conflicted_candidate"),
        ),
        "execution": PurposeProfile(
            profile="reasoning-execution",
            role="reasoning-agent",
            task="execution",
            risk="normal",
            time_horizon="current-step",
            expected_action="act_or_adjust",
            permission_scope="session",
            evidence_policy="cite_current_state_and_recent_results",
            retention_policy="preserve_commands_results_and_state_changes",
            ontology_lens=("current_state", "command_result", "file_edit", "blocker"),
            required_evidence=("current_state_citation",),
            retain=("recent_actions", "verification_results", "blockers"),
            suppress=("superseded_context", "rejected_candidate"),
        ),
        "review": PurposeProfile(
            profile="reasoning-review",
            role="reasoning-reviewer",
            task="review",
            risk="high",
            time_horizon="release-or-handoff",
            expected_action="approve_block_or_request_evidence",
            permission_scope="session",
            evidence_policy="cited_support_and_conflict_required",
            retention_policy="preserve_risks_conflicts_and_verification",
            ontology_lens=("claim", "conflict", "risk", "verification", "source"),
            required_evidence=("supporting_citation", "conflict_citation"),
            retain=("blocking_risks", "conflicts", "verification_results"),
            suppress=("uncited_claim", "pending_unreviewed_claim", "stale_context"),
        ),
        "reflection": PurposeProfile(
            profile="reasoning-reflection",
            role="reasoning-agent",
            task="reflection",
            risk="normal",
            time_horizon="cross-turn",
            expected_action="revise_or_record_learning",
            permission_scope="session",
            evidence_policy="cite_source_events_for_learning_proposals",
            retention_policy="preserve_review_pending_learning_without_authority",
            ontology_lens=("lesson", "belief_proposal", "procedure", "outcome"),
            required_evidence=("source_event_ref",),
            retain=("lessons", "procedure_candidates", "belief_proposals"),
            suppress=("authority_promotion", "rejected_candidate", "conflicted_candidate"),
            warnings=("Belief updates remain non-authoritative until reviewed.",),
        ),
    }
    return profiles[normalized]


@dataclass(frozen=True)
class ReasoningPrimitiveCall:
    """Observable Eventloom event contract for a reasoning primitive call."""

    primitive: str
    phase: str
    session_id: str
    query: str
    result_count: int
    evidence: list[dict[str, Any]] = field(default_factory=list)
    status: str = "succeeded"

    def __post_init__(self) -> None:
        _validate_text(self.primitive, field_name="primitive")
        object.__setattr__(self, "phase", validate_reasoning_phase(self.phase))
        _validate_text(self.session_id, field_name="session_id")
        _validate_text(self.query, field_name="query")
        if self.result_count < 0:
            raise ValueError("result_count must be non-negative")
        if self.status not in _STATUSES:
            raise ValueError("status must be one of: " + ", ".join(sorted(_STATUSES)))
        object.__setattr__(self, "evidence", [_validated_evidence(item) for item in self.evidence])

    def to_event(self, *, actor: str) -> dict[str, Any]:
        """Return an Eventloom append spec for the primitive call."""
        _validate_text(actor, field_name="actor")
        citations = [
            str(item["citation"])
            for item in self.evidence
            if isinstance(item.get("citation"), str) and item.get("citation")
        ]
        payload: dict[str, Any] = {
            "primitive": self.primitive,
            "phase": self.phase,
            "query": self.query,
            "status": self.status,
            "result_count": self.result_count,
            "evidence_count": len(self.evidence),
        }
        if citations:
            payload["citations"] = citations
        if self.evidence:
            payload["evidence"] = [dict(item) for item in self.evidence]
        return {
            "event_type": "reasoning.primitive.called",
            "actor": actor,
            "payload": payload,
            "thread": self.session_id,
        }


def build_belief_update_proposal_event(
    *,
    actor: str,
    session_id: str,
    claim: str,
    rationale: str,
    confidence: float,
    source_events: list[dict[str, Any]],
    phase: str = "reflection",
) -> dict[str, Any]:
    """Build a review-pending, non-authoritative belief proposal event."""
    _validate_text(actor, field_name="actor")
    _validate_text(session_id, field_name="session_id")
    _validate_text(claim, field_name="claim")
    _validate_text(rationale, field_name="rationale")
    normalized_phase = validate_reasoning_phase(phase)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    validated_sources = [_validated_source_event(item) for item in source_events]
    if not validated_sources:
        raise ValueError("source_events must include at least one cited Eventloom event")
    return {
        "event_type": "belief.update.proposed",
        "actor": actor,
        "thread": session_id,
        "payload": {
            "claim": claim.strip(),
            "rationale": rationale.strip(),
            "confidence": float(confidence),
            "phase": normalized_phase,
            "source_events": validated_sources,
            "authority_status": "non_authoritative",
            "review_status": "pending",
        },
    }


def _validate_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validated_source_event(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("source event must be a mapping")
    seq = value.get("seq")
    event_hash = value.get("hash")
    if not isinstance(seq, int) or seq < 1:
        raise ValueError("source event seq must be a positive integer")
    if not isinstance(event_hash, str) or _EVENT_HASH_RE.fullmatch(event_hash) is None:
        raise ValueError("source event hash must be 64 lowercase hex characters")
    return {"seq": seq, "hash": event_hash}


def _validated_evidence(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("evidence must be a mapping")
    citation = value.get("citation")
    if not isinstance(citation, str) or _EVENTLOOM_CITATION_RE.fullmatch(citation) is None:
        raise ValueError("evidence citation must be an Eventloom event citation")
    return dict(value)
