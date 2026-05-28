"""High-level coordination state for multi-agent projects.

The coordination layer packages existing Eventloom session sharding into a
governed parent/worker workflow. Worker sessions can report findings freely;
the parent mission session records reviews and promotions that become accepted
project state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from zaxy.event import Event
from zaxy.security import validate_payload, validate_session_id
from zaxy.session import SessionManager

FindingStatus = Literal["pending", "accepted", "rejected", "deferred", "conflicted"]
REVIEW_STATUSES: set[str] = {"accepted", "rejected", "deferred", "conflicted"}


@dataclass(frozen=True)
class CoordinationEventResult:
    """A sealed coordination event plus useful operator-facing fields."""

    event: Event
    mission_id: str
    worker_id: str | None = None
    finding_id: str | None = None
    handoff_id: str | None = None
    summary: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class WorkerState:
    """Registered worker session under a mission."""

    worker_id: str
    mission_id: str
    assignment: str | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "mission_id": self.mission_id,
            "assignment": self.assignment,
            "status": self.status,
        }


@dataclass(frozen=True)
class FindingState:
    """A worker finding with review/promotion state."""

    finding_id: str
    mission_id: str
    worker_id: str
    summary: str
    evidence: list[dict[str, Any]]
    confidence: float | None = None
    status: FindingStatus = "pending"
    claim_key: str | None = None
    claim_value: str | None = None
    rationale: str | None = None
    stale: bool = False
    superseded_by: str | None = None
    source_event_seq: int | None = None
    source_event_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "mission_id": self.mission_id,
            "worker_id": self.worker_id,
            "summary": self.summary,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "status": self.status,
            "claim_key": self.claim_key,
            "claim_value": self.claim_value,
            "rationale": self.rationale,
            "stale": self.stale,
            "superseded_by": self.superseded_by,
            "source_event_seq": self.source_event_seq,
            "source_event_hash": self.source_event_hash,
        }


@dataclass(frozen=True)
class ConflictState:
    """Deterministic conflict between findings about the same claim key."""

    claim_key: str
    findings: list[FindingState]
    conflict_type: str = "exact_claim"
    reason: str | None = None
    source_reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_key": self.claim_key,
            "conflict_type": self.conflict_type,
            "reason": self.reason,
            "source_reference": self.source_reference,
            "findings": [finding.to_dict() for finding in self.findings],
        }


SemanticConflictDetector = Callable[[list[FindingState]], list[ConflictState]]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
_LOCAL_CONTRADICTION_GROUPS: tuple[tuple[str, str], ...] = (
    ("enabled", "disabled"),
    ("enable", "disable"),
    ("present", "missing"),
    ("passing", "failing"),
    ("passed", "failed"),
    ("available", "unavailable"),
    ("supported", "unsupported"),
    ("required", "optional"),
    ("true", "false"),
)


class LocalSemanticConflictDetector:
    """Deterministic local lexical detector for obvious semantic contradictions.

    This detector intentionally avoids LLM inference. It only emits a semantic
    conflict when two findings share enough non-stopword subject tokens and use
    one of a small set of explicit opposite state terms.
    """

    def __init__(self, *, min_shared_subject_tokens: int = 2) -> None:
        if min_shared_subject_tokens < 1:
            raise ValueError("min_shared_subject_tokens must be positive")
        self.min_shared_subject_tokens = min_shared_subject_tokens

    def __call__(self, findings: list[FindingState]) -> list[ConflictState]:
        conflicts: list[ConflictState] = []
        for left_index, left in enumerate(findings):
            left_tokens = _semantic_subject_tokens(left)
            if not left_tokens:
                continue
            for right in findings[left_index + 1:]:
                right_tokens = _semantic_subject_tokens(right)
                shared = sorted(left_tokens & right_tokens)
                if len(shared) < self.min_shared_subject_tokens:
                    continue
                contradiction = _local_contradiction(left, right)
                if contradiction is None:
                    continue
                conflicts.append(
                    ConflictState(
                        claim_key=f"semantic:{'-'.join(shared[:4])}",
                        findings=[left, right],
                        conflict_type="semantic",
                        reason=f"local_lexical_contradiction:{contradiction}",
                    )
                )
        return conflicts


@dataclass(frozen=True)
class CoordinationBrief:
    """Prompt- and operator-ready state for a mission."""

    mission_id: str
    objective: str | None
    workers: list[WorkerState]
    accepted_findings: list[FindingState]
    pending_findings: list[FindingState]
    rejected_findings: list[FindingState]
    deferred_findings: list[FindingState]
    conflicted_findings: list[FindingState]
    stale_findings: list[FindingState]
    conflicts: list[ConflictState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "workers": [worker.to_dict() for worker in self.workers],
            "accepted_findings": [finding.to_dict() for finding in self.accepted_findings],
            "pending_findings": [finding.to_dict() for finding in self.pending_findings],
            "rejected_findings": [finding.to_dict() for finding in self.rejected_findings],
            "deferred_findings": [finding.to_dict() for finding in self.deferred_findings],
            "conflicted_findings": [finding.to_dict() for finding in self.conflicted_findings],
            "stale_findings": [finding.to_dict() for finding in self.stale_findings],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
        }


@dataclass(frozen=True)
class CoordinationCheckout:
    """Accepted mission state suitable for prompt injection."""

    mission_id: str
    objective: str | None
    accepted_findings: list[FindingState]
    pending_findings: list[FindingState]
    conflicted_findings: list[FindingState]
    stale_findings: list[FindingState]
    conflicts: list[ConflictState]
    excluded_pending_count: int
    excluded_conflict_count: int
    excluded_stale_count: int
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "accepted_findings": [finding.to_dict() for finding in self.accepted_findings],
            "pending_findings": [finding.to_dict() for finding in self.pending_findings],
            "conflicted_findings": [finding.to_dict() for finding in self.conflicted_findings],
            "stale_findings": [finding.to_dict() for finding in self.stale_findings],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "excluded_pending_count": self.excluded_pending_count,
            "excluded_conflict_count": self.excluded_conflict_count,
            "excluded_stale_count": self.excluded_stale_count,
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class WorkerPerformance:
    """Outcome metrics for one coordination worker."""

    worker_id: str
    mission_id: str
    assignment: str | None
    total_findings: int
    accepted_findings: int
    promoted_findings: int
    rejected_findings: int
    deferred_findings: int
    conflicted_findings: int
    pending_findings: int
    missing_evidence_count: int
    duplicate_finding_count: int
    test_backed_findings: int
    stale_claim_count: int | None
    duplicate_finding_ids: list[str]

    @property
    def acceptance_rate(self) -> float:
        return _rate(self.accepted_findings, self.total_findings)

    @property
    def missing_evidence_rate(self) -> float:
        return _rate(self.missing_evidence_count, self.total_findings)

    @property
    def duplicate_finding_rate(self) -> float:
        return _rate(self.duplicate_finding_count, self.total_findings)

    @property
    def test_backed_rate(self) -> float:
        return _rate(self.test_backed_findings, self.total_findings)

    @property
    def stale_claim_rate(self) -> float | None:
        if self.stale_claim_count is None:
            return None
        return _rate(self.stale_claim_count, self.total_findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "mission_id": self.mission_id,
            "assignment": self.assignment,
            "total_findings": self.total_findings,
            "accepted_findings": self.accepted_findings,
            "promoted_findings": self.promoted_findings,
            "rejected_findings": self.rejected_findings,
            "deferred_findings": self.deferred_findings,
            "conflicted_findings": self.conflicted_findings,
            "pending_findings": self.pending_findings,
            "missing_evidence_count": self.missing_evidence_count,
            "duplicate_finding_count": self.duplicate_finding_count,
            "test_backed_findings": self.test_backed_findings,
            "stale_claim_count": self.stale_claim_count,
            "duplicate_finding_ids": self.duplicate_finding_ids,
            "acceptance_rate": self.acceptance_rate,
            "missing_evidence_rate": self.missing_evidence_rate,
            "duplicate_finding_rate": self.duplicate_finding_rate,
            "test_backed_rate": self.test_backed_rate,
            "stale_claim_rate": self.stale_claim_rate,
        }


@dataclass(frozen=True)
class CoordinationPerformanceLedger:
    """Mission-level worker outcome ledger."""

    mission_id: str
    objective: str | None
    workers: list[WorkerPerformance]

    @property
    def worker_count(self) -> int:
        return len(self.workers)

    @property
    def total_findings(self) -> int:
        return sum(worker.total_findings for worker in self.workers)

    def worker(self, worker_id: str) -> WorkerPerformance:
        for performance in self.workers:
            if performance.worker_id == worker_id:
                return performance
        raise KeyError(f"Unknown worker_id for mission {self.mission_id}: {worker_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "worker_count": self.worker_count,
            "total_findings": self.total_findings,
            "workers": [worker.to_dict() for worker in self.workers],
        }


@dataclass(frozen=True)
class CoordinationApprovalFinding:
    """Finding entry exported for remote approval."""

    finding: FindingState
    conflict_keys: list[str]

    @property
    def finding_id(self) -> str:
        return self.finding.finding_id

    @property
    def requires_evidence(self) -> bool:
        return not self.finding.evidence

    @property
    def stale(self) -> bool:
        return self.finding.stale

    def to_dict(self) -> dict[str, Any]:
        payload = self.finding.to_dict()
        payload.update(
            {
                "allowed_statuses": sorted(REVIEW_STATUSES),
                "requires_evidence": self.requires_evidence,
                "conflict_keys": self.conflict_keys,
            }
        )
        return payload


@dataclass(frozen=True)
class CoordinationApprovalPacket:
    """Portable packet for remote human review."""

    packet_id: str
    mission_id: str
    objective: str | None
    findings: list[CoordinationApprovalFinding]
    pending_count: int
    conflict_count: int

    @property
    def decisions_template(self) -> list[dict[str, Any]]:
        return [
            {
                "finding_id": finding.finding_id,
                "status": "deferred",
                "rationale": "",
                "promote": False,
            }
            for finding in self.findings
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "mission_id": self.mission_id,
            "objective": self.objective,
            "pending_count": self.pending_count,
            "conflict_count": self.conflict_count,
            "findings": [finding.to_dict() for finding in self.findings],
            "decisions_template": self.decisions_template,
        }


@dataclass(frozen=True)
class CoordinationReviewExport:
    """Static human-review artifact derived from a coordination approval packet."""

    mission_id: str
    packet: CoordinationApprovalPacket
    markdown: str
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "packet": self.packet.to_dict(),
            "markdown": self.markdown,
            "read_only": self.read_only,
        }


@dataclass(frozen=True)
class CoordinationApprovalDecisionResult:
    """Result of applying a remote approval decision packet."""

    mission_id: str
    events: list[Event]
    reviewed_count: int
    promoted_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "reviewed_count": self.reviewed_count,
            "promoted_count": self.promoted_count,
            "events": [
                {
                    "event_type": event.type,
                    "seq": event.seq,
                    "hash": event.hash,
                    "finding_id": event.payload.get("finding_id"),
                }
                for event in self.events
            ],
        }


class CoordinationManager:
    """Manage mission/worker coordination state over Eventloom sessions."""

    def __init__(
        self,
        eventloom_path: str | Path = ".eventloom",
        *,
        semantic_conflict_detector: SemanticConflictDetector | None = None,
    ) -> None:
        self.session_manager = SessionManager(base_path=str(eventloom_path))
        self._semantic_conflict_detector = semantic_conflict_detector

    def start_mission(self, mission_id: str, *, objective: str, actor: str = "coordinator") -> CoordinationEventResult:
        """Create a parent mission session."""
        mission_sid = validate_session_id(mission_id)
        payload = validate_payload({"mission_id": mission_sid, "objective": objective, "status": "active"})
        event = self.session_manager.get(mission_sid).eventlog.append(
            "coordination.mission.created",
            actor=actor,
            payload=payload,
            thread=mission_sid,
        )
        return CoordinationEventResult(event=event, mission_id=mission_sid, summary=objective)

    def create_worker(
        self,
        mission_id: str,
        worker_id: str,
        *,
        actor: str = "coordinator",
    ) -> CoordinationEventResult:
        """Register a worker session under the parent mission."""
        mission_sid = validate_session_id(mission_id)
        worker_sid = validate_session_id(worker_id)
        payload = validate_payload({"mission_id": mission_sid, "worker_id": worker_sid, "status": "active"})
        event = self.session_manager.get(mission_sid).eventlog.append(
            "coordination.worker.created",
            actor=actor,
            payload=payload,
            thread=mission_sid,
        )
        self.session_manager.get(worker_sid)
        return CoordinationEventResult(event=event, mission_id=mission_sid, worker_id=worker_sid)

    def assign(
        self,
        mission_id: str,
        worker_id: str,
        assignment: str,
        *,
        actor: str = "coordinator",
    ) -> CoordinationEventResult:
        """Assign a scoped task to a worker."""
        mission_sid = validate_session_id(mission_id)
        worker_sid = validate_session_id(worker_id)
        payload = validate_payload(
            {
                "mission_id": mission_sid,
                "worker_id": worker_sid,
                "assignment": assignment,
                "status": "assigned",
            }
        )
        event = self.session_manager.get(mission_sid).eventlog.append(
            "coordination.assignment.created",
            actor=actor,
            payload=payload,
            thread=mission_sid,
        )
        return CoordinationEventResult(event=event, mission_id=mission_sid, worker_id=worker_sid, summary=assignment)

    def report_finding(
        self,
        mission_id: str,
        worker_id: str,
        *,
        summary: str,
        actor: str,
        evidence: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        claim_key: str | None = None,
        claim_value: str | None = None,
        finding_id: str | None = None,
    ) -> CoordinationEventResult:
        """Append a worker-local finding."""
        mission_sid = validate_session_id(mission_id)
        worker_sid = validate_session_id(worker_id)
        eventlog = self.session_manager.get(worker_sid).eventlog
        finding_id = finding_id or f"{worker_sid}:finding:{len(eventlog.read_all()) + 1}"
        safe_evidence = [_normalize_evidence(item) for item in evidence or []]
        payload = validate_payload(
            {
                "mission_id": mission_sid,
                "worker_id": worker_sid,
                "finding_id": finding_id,
                "summary": summary,
                "evidence": safe_evidence,
                "confidence": confidence,
                "status": "pending",
                "claim_key": claim_key,
                "claim_value": claim_value,
            }
        )
        event = eventlog.append(
            "coordination.finding.reported",
            actor=actor,
            payload=payload,
            thread=worker_sid,
        )
        return CoordinationEventResult(
            event=event,
            mission_id=mission_sid,
            worker_id=worker_sid,
            finding_id=finding_id,
            summary=summary,
            evidence=safe_evidence,
        )

    def review_finding(
        self,
        mission_id: str,
        finding_id: str,
        *,
        status: str,
        actor: str = "coordinator",
        rationale: str | None = None,
    ) -> CoordinationEventResult:
        """Record a coordinator review decision for a finding."""
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Invalid finding status: {status}")
        mission_sid = validate_session_id(mission_id)
        finding = self._find_finding(mission_sid, finding_id)
        payload = validate_payload(
            {
                "mission_id": mission_sid,
                "worker_id": finding.worker_id,
                "finding_id": finding.finding_id,
                "status": status,
                "rationale": rationale,
            }
        )
        event = self.session_manager.get(mission_sid).eventlog.append(
            "coordination.finding.reviewed",
            actor=actor,
            payload=payload,
            thread=mission_sid,
        )
        return CoordinationEventResult(
            event=event,
            mission_id=mission_sid,
            worker_id=finding.worker_id,
            finding_id=finding.finding_id,
            summary=rationale,
        )

    def promote_finding(
        self,
        mission_id: str,
        finding_id: str,
        *,
        actor: str = "coordinator",
    ) -> CoordinationEventResult:
        """Promote a finding into accepted parent mission state."""
        mission_sid = validate_session_id(mission_id)
        finding = self._find_finding(mission_sid, finding_id)
        payload = validate_payload(
            {
                "mission_id": mission_sid,
                "worker_id": finding.worker_id,
                "finding_id": finding.finding_id,
                "summary": finding.summary,
                "evidence": finding.evidence,
                "confidence": finding.confidence,
                "claim_key": finding.claim_key,
                "claim_value": finding.claim_value,
                "source_event_seq": finding.source_event_seq,
                "source_event_hash": finding.source_event_hash,
                "status": "accepted",
            }
        )
        event = self.session_manager.get(mission_sid).eventlog.append(
            "coordination.finding.promoted",
            actor=actor,
            payload=payload,
            thread=mission_sid,
        )
        return CoordinationEventResult(
            event=event,
            mission_id=mission_sid,
            worker_id=finding.worker_id,
            finding_id=finding.finding_id,
            summary=finding.summary,
            evidence=finding.evidence,
        )

    def brief(self, mission_id: str) -> CoordinationBrief:
        """Build a governed mission brief from parent and worker sessions."""
        mission_sid = validate_session_id(mission_id)
        parent_events = self.session_manager.replay(mission_sid).events
        objective = _mission_objective(parent_events)
        workers = self._workers(parent_events, mission_sid)
        reviews = _reviews(parent_events)
        promoted_ids = {
            str(event.payload.get("finding_id"))
            for event in parent_events
            if event.type == "coordination.finding.promoted"
        }
        findings = [
            finding
            for worker in workers
            for finding in self._worker_findings(worker.worker_id, mission_sid, reviews, promoted_ids)
        ]
        conflicts = _detect_conflicts(findings, semantic_conflict_detector=self._semantic_conflict_detector)
        conflict_ids = {finding.finding_id for conflict in conflicts for finding in conflict.findings}
        accepted = [finding for finding in findings if finding.finding_id in promoted_ids]
        pending = [
            finding
            for finding in findings
            if finding.status == "pending" and finding.finding_id not in promoted_ids
        ]
        rejected = [finding for finding in findings if finding.status == "rejected"]
        deferred = [finding for finding in findings if finding.status == "deferred"]
        conflicted = [finding for finding in findings if finding.status == "conflicted" or finding.finding_id in conflict_ids]
        stale = [finding for finding in findings if finding.stale]
        return CoordinationBrief(
            mission_id=mission_sid,
            objective=objective,
            workers=workers,
            accepted_findings=accepted,
            pending_findings=pending,
            rejected_findings=rejected,
            deferred_findings=deferred,
            conflicted_findings=conflicted,
            stale_findings=stale,
            conflicts=conflicts,
        )

    def checkout(self, mission_id: str, *, include_diagnostics: bool = False) -> CoordinationCheckout:
        """Return accepted parent mission state for model prompt context."""
        brief = self.brief(mission_id)
        pending = brief.pending_findings if include_diagnostics else []
        conflicted = brief.conflicted_findings if include_diagnostics else []
        stale = brief.stale_findings if include_diagnostics else []
        conflicts = brief.conflicts if include_diagnostics else []
        return CoordinationCheckout(
            mission_id=brief.mission_id,
            objective=brief.objective,
            accepted_findings=brief.accepted_findings,
            pending_findings=pending,
            conflicted_findings=conflicted,
            stale_findings=stale,
            conflicts=conflicts,
            excluded_pending_count=0 if include_diagnostics else len(brief.pending_findings),
            excluded_conflict_count=0 if include_diagnostics else len(brief.conflicts),
            excluded_stale_count=0 if include_diagnostics else len(brief.stale_findings),
            prompt=_checkout_prompt(brief, include_diagnostics=include_diagnostics),
        )

    def performance_ledger(self, mission_id: str) -> CoordinationPerformanceLedger:
        """Return replay-backed worker outcome metrics for a mission."""
        brief = self.brief(mission_id)
        all_findings = [
            *brief.accepted_findings,
            *brief.pending_findings,
            *brief.rejected_findings,
            *brief.deferred_findings,
        ]
        known_ids = {finding.finding_id for finding in all_findings}
        for finding in brief.conflicted_findings:
            if finding.finding_id not in known_ids:
                all_findings.append(finding)
                known_ids.add(finding.finding_id)
        promoted_ids = {finding.finding_id for finding in brief.accepted_findings}
        conflict_ids = {finding.finding_id for conflict in brief.conflicts for finding in conflict.findings}
        duplicate_ids = _duplicate_finding_ids(all_findings)
        worker_metrics = []
        for worker in brief.workers:
            findings = [finding for finding in all_findings if finding.worker_id == worker.worker_id]
            worker_metrics.append(
                WorkerPerformance(
                    worker_id=worker.worker_id,
                    mission_id=brief.mission_id,
                    assignment=worker.assignment,
                    total_findings=len(findings),
                    accepted_findings=sum(1 for finding in findings if finding.status == "accepted"),
                    promoted_findings=sum(1 for finding in findings if finding.finding_id in promoted_ids),
                    rejected_findings=sum(1 for finding in findings if finding.status == "rejected"),
                    deferred_findings=sum(1 for finding in findings if finding.status == "deferred"),
                    conflicted_findings=sum(
                        1 for finding in findings if finding.status == "conflicted" or finding.finding_id in conflict_ids
                    ),
                    pending_findings=sum(1 for finding in findings if finding.status == "pending"),
                    missing_evidence_count=sum(1 for finding in findings if not finding.evidence),
                    duplicate_finding_count=sum(1 for finding in findings if finding.finding_id in duplicate_ids),
                    test_backed_findings=sum(1 for finding in findings if _is_test_backed(finding.evidence)),
                    stale_claim_count=sum(1 for finding in findings if finding.stale),
                    duplicate_finding_ids=[
                        finding.finding_id for finding in findings if finding.finding_id in duplicate_ids
                    ],
                )
            )
        return CoordinationPerformanceLedger(
            mission_id=brief.mission_id,
            objective=brief.objective,
            workers=worker_metrics,
        )

    def create_handoff(
        self,
        mission_id: str,
        *,
        summary: str,
        actor: str = "coordinator",
        next_steps: list[str] | None = None,
        risks: list[str] | None = None,
        handoff_id: str | None = None,
    ) -> CoordinationEventResult:
        """Append a final handoff event to the parent mission session."""
        mission_sid = validate_session_id(mission_id)
        eventlog = self.session_manager.get(mission_sid).eventlog
        safe_next_steps = [str(item) for item in next_steps or [] if str(item)]
        safe_risks = [str(item) for item in risks or [] if str(item)]
        handoff_sid = handoff_id or f"{mission_sid}:handoff:{len(eventlog.read_all()) + 1}"
        payload = validate_payload(
            {
                "mission_id": mission_sid,
                "handoff_id": handoff_sid,
                "summary": summary,
                "next_steps": safe_next_steps,
                "risks": safe_risks,
                "status": "created",
            }
        )
        event = eventlog.append(
            "coordination.handoff.created",
            actor=actor,
            payload=payload,
            thread=mission_sid,
        )
        evidence = [
            *[{"kind": "next_step", "reference": item} for item in safe_next_steps],
            *[{"kind": "risk", "reference": item} for item in safe_risks],
        ]
        return CoordinationEventResult(
            event=event,
            mission_id=mission_sid,
            handoff_id=handoff_sid,
            summary=summary,
            evidence=evidence,
        )

    def approval_packet(self, mission_id: str) -> CoordinationApprovalPacket:
        """Return a portable packet of findings that need remote approval."""
        brief = self.brief(mission_id)
        conflict_keys_by_finding: dict[str, list[str]] = {}
        for conflict in brief.conflicts:
            for finding in conflict.findings:
                conflict_keys_by_finding.setdefault(finding.finding_id, []).append(conflict.claim_key)
        findings_by_id: dict[str, FindingState] = {}
        for finding in [*brief.pending_findings, *brief.conflicted_findings, *brief.stale_findings]:
            if finding.status == "accepted":
                continue
            findings_by_id[finding.finding_id] = finding
        approval_findings = [
            CoordinationApprovalFinding(
                finding=finding,
                conflict_keys=sorted(conflict_keys_by_finding.get(finding.finding_id, [])),
            )
            for finding in findings_by_id.values()
        ]
        return CoordinationApprovalPacket(
            packet_id=f"{brief.mission_id}:approval:{len(approval_findings)}:{len(brief.conflicts)}",
            mission_id=brief.mission_id,
            objective=brief.objective,
            findings=approval_findings,
            pending_count=len(brief.pending_findings),
            conflict_count=len(brief.conflicts),
        )

    def review_export(self, mission_id: str) -> CoordinationReviewExport:
        """Return a static Markdown review artifact without appending events."""
        packet = self.approval_packet(mission_id)
        return CoordinationReviewExport(
            mission_id=packet.mission_id,
            packet=packet,
            markdown=_review_export_markdown(packet),
        )

    def record_detected_conflicts(
        self,
        mission_id: str,
        *,
        actor: str = "zaxy",
    ) -> list[CoordinationEventResult]:
        """Append deterministic conflict facts for graph projection without duplicating them."""
        mission_sid = validate_session_id(mission_id)
        brief = self.brief(mission_sid)
        parent_log = self.session_manager.get(mission_sid).eventlog
        existing = _recorded_conflict_signatures(parent_log.read_all())
        results: list[CoordinationEventResult] = []
        for conflict in brief.conflicts:
            signature = _conflict_signature(conflict)
            if signature in existing:
                continue
            finding_ids = [finding.finding_id for finding in conflict.findings]
            payload = validate_payload(
                {
                    "mission_id": mission_sid,
                    "conflict_id": f"{mission_sid}:conflict:{signature[:16]}",
                    "conflict_signature": signature,
                    "claim_key": conflict.claim_key,
                    "conflict_type": conflict.conflict_type,
                    "reason": conflict.reason,
                    "source_reference": conflict.source_reference,
                    "finding_ids": finding_ids,
                    "summary": _conflict_summary(conflict),
                }
            )
            event = parent_log.append(
                "coordination.conflict.detected",
                actor=actor,
                payload=payload,
                thread=mission_sid,
            )
            existing.add(signature)
            results.append(
                CoordinationEventResult(
                    event=event,
                    mission_id=mission_sid,
                    summary=str(payload["summary"]),
                    evidence=[
                        {"kind": "finding", "reference": finding_id}
                        for finding_id in finding_ids
                    ],
                )
            )
        return results

    def apply_approval_decisions(
        self,
        mission_id: str,
        decisions: list[dict[str, Any]],
        *,
        actor: str = "coordinator",
    ) -> CoordinationApprovalDecisionResult:
        """Apply remote approval decisions as normal review and promotion events."""
        mission_sid = validate_session_id(mission_id)
        events: list[Event] = []
        reviewed_count = 0
        promoted_count = 0
        for decision in decisions:
            finding_id = str(decision.get("finding_id") or "")
            status = str(decision.get("status") or "")
            rationale = _optional_str(decision.get("rationale"))
            promote = bool(decision.get("promote", False))
            review = self.review_finding(
                mission_sid,
                finding_id,
                status=status,
                actor=actor,
                rationale=rationale,
            )
            events.append(review.event)
            reviewed_count += 1
            if status == "accepted" and promote:
                promotion = self.promote_finding(mission_sid, finding_id, actor=actor)
                events.append(promotion.event)
                promoted_count += 1
        return CoordinationApprovalDecisionResult(
            mission_id=mission_sid,
            events=events,
            reviewed_count=reviewed_count,
            promoted_count=promoted_count,
        )

    def _workers(self, parent_events: list[Event], mission_id: str) -> list[WorkerState]:
        assignments: dict[str, str] = {}
        workers: dict[str, WorkerState] = {}
        for event in parent_events:
            if event.type == "coordination.assignment.created":
                worker_id = str(event.payload.get("worker_id") or "")
                if worker_id:
                    assignments[worker_id] = str(event.payload.get("assignment") or "")
            if event.type == "coordination.worker.created":
                worker_id = str(event.payload.get("worker_id") or "")
                if worker_id:
                    workers[worker_id] = WorkerState(
                        worker_id=worker_id,
                        mission_id=mission_id,
                        assignment=assignments.get(worker_id),
                        status=str(event.payload.get("status") or "active"),
                    )
        for worker_id, assignment in assignments.items():
            if worker_id in workers:
                workers[worker_id] = WorkerState(
                    worker_id=worker_id,
                    mission_id=mission_id,
                    assignment=assignment,
                    status=workers[worker_id].status,
                )
        return list(workers.values())

    def _worker_findings(
        self,
        worker_id: str,
        mission_id: str,
        reviews: dict[str, dict[str, str | None]],
        promoted_ids: set[str],
    ) -> list[FindingState]:
        findings: list[FindingState] = []
        for event in self.session_manager.replay(worker_id).events:
            if event.type != "coordination.finding.reported":
                continue
            if event.payload.get("mission_id") != mission_id:
                continue
            finding_id = str(event.payload.get("finding_id") or "")
            review = reviews.get(finding_id, {})
            status = str(review.get("status") or event.payload.get("status") or "pending")
            if finding_id in promoted_ids:
                status = "accepted"
            findings.append(
                FindingState(
                    finding_id=finding_id,
                    mission_id=mission_id,
                    worker_id=worker_id,
                    summary=str(event.payload.get("summary") or ""),
                    evidence=_evidence_list(event.payload.get("evidence")),
                    confidence=_optional_float(event.payload.get("confidence")),
                    status=_finding_status(status),
                    claim_key=_optional_str(event.payload.get("claim_key")),
                    claim_value=_optional_str(event.payload.get("claim_value")),
                    rationale=review.get("rationale"),
                    stale=_is_stale_evidence(_evidence_list(event.payload.get("evidence"))),
                    superseded_by=_superseded_by(_evidence_list(event.payload.get("evidence"))),
                    source_event_seq=event.seq,
                    source_event_hash=event.hash,
                )
            )
        return findings

    def _find_finding(self, mission_id: str, finding_id: str) -> FindingState:
        parent_events = self.session_manager.replay(mission_id).events
        reviews = _reviews(parent_events)
        promoted_ids = {
            str(event.payload.get("finding_id"))
            for event in parent_events
            if event.type == "coordination.finding.promoted"
        }
        for worker in self._workers(parent_events, mission_id):
            for finding in self._worker_findings(worker.worker_id, mission_id, reviews, promoted_ids):
                if finding.finding_id == finding_id:
                    return finding
        raise ValueError(f"Unknown finding_id for mission {mission_id}: {finding_id}")


def _mission_objective(events: list[Event]) -> str | None:
    for event in events:
        if event.type == "coordination.mission.created":
            objective = event.payload.get("objective")
            return str(objective) if objective is not None else None
    return None


def _reviews(events: list[Event]) -> dict[str, dict[str, str | None]]:
    reviews: dict[str, dict[str, str | None]] = {}
    for event in events:
        if event.type != "coordination.finding.reviewed":
            continue
        finding_id = str(event.payload.get("finding_id") or "")
        if finding_id:
            reviews[finding_id] = {
                "status": _optional_str(event.payload.get("status")),
                "rationale": _optional_str(event.payload.get("rationale")),
            }
    return reviews


def _recorded_conflict_signatures(events: list[Event]) -> set[str]:
    signatures: set[str] = set()
    for event in events:
        if event.type != "coordination.conflict.detected":
            continue
        signature = _optional_str(event.payload.get("conflict_signature"))
        if signature is not None:
            signatures.add(signature)
    return signatures


def _conflict_signature(conflict: ConflictState) -> str:
    body = {
        "claim_key": conflict.claim_key,
        "conflict_type": conflict.conflict_type,
        "finding_ids": sorted(finding.finding_id for finding in conflict.findings),
        "reason": conflict.reason,
        "source_reference": conflict.source_reference,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _conflict_summary(conflict: ConflictState) -> str:
    if conflict.conflict_type == "source_state" and conflict.source_reference:
        return f"Findings cite incompatible source snapshots for {conflict.source_reference}."
    return f"Findings disagree on {conflict.claim_key}."


def _detect_conflicts(
    findings: list[FindingState],
    *,
    semantic_conflict_detector: SemanticConflictDetector | None = None,
) -> list[ConflictState]:
    conflicts: list[ConflictState] = []
    by_claim: dict[str, list[FindingState]] = {}
    for finding in findings:
        if finding.claim_key and finding.claim_value:
            by_claim.setdefault(finding.claim_key, []).append(finding)
    for claim_key, scoped_findings in by_claim.items():
        values = {finding.claim_value for finding in scoped_findings}
        if len(values) > 1:
            conflicts.append(
                ConflictState(
                    claim_key=claim_key,
                    findings=scoped_findings,
                    conflict_type="exact_claim",
                    reason="conflicting_claim_values",
                )
            )
    conflicts.extend(_detect_source_state_conflicts(findings))
    if semantic_conflict_detector is not None:
        conflicts.extend(_semantic_conflicts(findings, semantic_conflict_detector))
    return conflicts


def _semantic_conflicts(
    findings: list[FindingState],
    detector: SemanticConflictDetector,
) -> list[ConflictState]:
    known_ids = {finding.finding_id for finding in findings}
    conflicts: list[ConflictState] = []
    for conflict in detector(list(findings)):
        unknown_ids = sorted(
            finding.finding_id for finding in conflict.findings if finding.finding_id not in known_ids
        )
        if unknown_ids:
            raise ValueError(f"unknown semantic conflict finding_id: {', '.join(unknown_ids)}")
        conflicts.append(
            ConflictState(
                claim_key=conflict.claim_key,
                findings=_unique_findings(conflict.findings),
                conflict_type="semantic",
                reason=conflict.reason or "semantic_adapter",
                source_reference=conflict.source_reference,
            )
        )
    return conflicts


def _semantic_subject_tokens(finding: FindingState) -> set[str]:
    text_parts = [finding.summary]
    if finding.claim_key:
        text_parts.append(finding.claim_key)
    if finding.claim_value:
        text_parts.append(finding.claim_value)
    return {
        token
        for token in _semantic_tokens(" ".join(text_parts))
        if token not in _SEMANTIC_STOPWORDS and not _is_contradiction_token(token)
    }


def _local_contradiction(left: FindingState, right: FindingState) -> str | None:
    left_tokens = _semantic_tokens(left.summary)
    right_tokens = _semantic_tokens(right.summary)
    for positive, negative in _LOCAL_CONTRADICTION_GROUPS:
        if positive in left_tokens and negative in right_tokens:
            return f"{negative}/{positive}"
        if negative in left_tokens and positive in right_tokens:
            return f"{negative}/{positive}"
    return None


def _is_contradiction_token(token: str) -> bool:
    return any(token in pair for pair in _LOCAL_CONTRADICTION_GROUPS)


def _semantic_tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}


def _detect_source_state_conflicts(findings: list[FindingState]) -> list[ConflictState]:
    by_source: dict[str, dict[str, list[FindingState]]] = {}
    for finding in findings:
        for item in finding.evidence:
            reference = _optional_str(item.get("reference"))
            source_hash = _optional_str(item.get("source_sha256"))
            if reference is None or source_hash is None:
                continue
            if str(item.get("kind") or "").casefold() not in {"file", "source", "document"}:
                continue
            by_source.setdefault(reference, {}).setdefault(source_hash, []).append(finding)
    conflicts: list[ConflictState] = []
    for reference, findings_by_hash in sorted(by_source.items()):
        if len(findings_by_hash) <= 1:
            continue
        scoped_findings = _unique_findings(
            finding
            for source_hash in sorted(findings_by_hash)
            for finding in findings_by_hash[source_hash]
        )
        conflicts.append(
            ConflictState(
                claim_key=f"source:{reference}",
                findings=scoped_findings,
                conflict_type="source_state",
                reason="conflicting_source_snapshots",
                source_reference=reference,
            )
        )
    return conflicts


def _unique_findings(findings: Any) -> list[FindingState]:
    unique: dict[str, FindingState] = {}
    for finding in findings:
        unique.setdefault(finding.finding_id, finding)
    return list(unique.values())


def _normalize_evidence(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("kind") or "source")
    if kind == "git":
        return validate_payload(
            {
                "kind": "git",
                "reference": str(item.get("reference") or item.get("branch") or ""),
                "repo_root": item.get("repo_root"),
                "worktree": item.get("worktree"),
                "branch": item.get("branch"),
                "head": item.get("head"),
                "detached": bool(item.get("detached", False)),
                "dirty": bool(item.get("dirty", False)),
                "changed_files": item.get("changed_files") if isinstance(item.get("changed_files"), list) else [],
                "diff_summary": item.get("diff_summary"),
                "worktrees": item.get("worktrees") if isinstance(item.get("worktrees"), list) else [],
                "test_results": item.get("test_results") if isinstance(item.get("test_results"), list) else [],
            }
        )
    if kind == "test_result":
        return validate_payload(
            {
                "kind": "test_result",
                "reference": str(item.get("reference") or item.get("command") or ""),
                "command": item.get("command"),
                "status": item.get("status"),
                "summary": item.get("summary"),
                "exit_code": item.get("exit_code"),
            }
        )
    return validate_payload(
        {
            "kind": kind,
            "reference": str(item.get("reference") or ""),
            "summary": item.get("summary"),
            "status": item.get("status"),
            "stale": bool(item.get("stale", False)),
            "superseded_by": item.get("superseded_by"),
            "source_sha256": item.get("source_sha256"),
        }
    )


def _evidence_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _finding_status(value: str) -> FindingStatus:
    if value in {"accepted", "rejected", "deferred", "conflicted"}:
        return value  # type: ignore[return-value]
    return "pending"


def _duplicate_finding_ids(findings: list[FindingState]) -> set[str]:
    seen: set[tuple[str, str]] = set()
    duplicates: set[str] = set()
    ordered = sorted(findings, key=lambda finding: finding.source_event_seq or 0)
    for finding in ordered:
        if not finding.claim_key or not finding.claim_value:
            continue
        signature = (finding.claim_key, finding.claim_value)
        if signature in seen:
            duplicates.add(finding.finding_id)
        else:
            seen.add(signature)
    return duplicates


def _is_test_backed(evidence: list[dict[str, Any]]) -> bool:
    test_markers = ("pytest", "unittest", "npm test", "pnpm test", "yarn test", "go test", "cargo test")
    for item in evidence:
        kind = str(item.get("kind") or "").lower()
        if kind == "git" and item.get("test_results"):
            return True
        if kind not in {"command", "test", "test_result"}:
            continue
        reference = str(item.get("reference") or item.get("command") or "").lower()
        if any(marker in reference for marker in test_markers):
            return True
    return False


def _is_stale_evidence(evidence: list[dict[str, Any]]) -> bool:
    for item in evidence:
        if item.get("stale") is True:
            return True
        marker = str(item.get("status") or item.get("kind") or "").casefold()
        if marker in {"stale", "superseded"}:
            return True
        if _optional_str(item.get("superseded_by")) is not None:
            return True
    return False


def _superseded_by(evidence: list[dict[str, Any]]) -> str | None:
    for item in evidence:
        value = _optional_str(item.get("superseded_by"))
        if value is not None:
            return value
    return None


def _review_export_markdown(packet: CoordinationApprovalPacket) -> str:
    lines = [
        f"# Zaxy Coordinate Review: {packet.mission_id}",
        "",
        f"Objective: {packet.objective or '-'}",
        f"Packet: `{packet.packet_id}`",
        f"Findings needing review: {len(packet.findings)}",
        f"Pending findings: {packet.pending_count}",
        f"Conflicts: {packet.conflict_count}",
        "",
    ]
    if not packet.findings:
        lines.append("No findings currently require review.")
    for approval_finding in packet.findings:
        finding = approval_finding.finding
        lines.extend(
            [
                f"## {finding.finding_id}",
                "",
                f"- Worker: `{finding.worker_id}`",
                f"- Status: {finding.status}",
                f"- Confidence: {_markdown_value(finding.confidence)}",
                f"- Summary: {_markdown_text(finding.summary)}",
                f"- Status options: {', '.join(sorted(REVIEW_STATUSES))}",
                f"- Requires evidence: {str(approval_finding.requires_evidence).lower()}",
                f"- Stale: {str(finding.stale).lower()}",
            ]
        )
        if finding.superseded_by:
            lines.append(f"- Superseded by: `{_markdown_text(finding.superseded_by)}`")
        if approval_finding.conflict_keys:
            lines.append(f"- Conflict keys: {', '.join(approval_finding.conflict_keys)}")
        if finding.evidence:
            for item in finding.evidence:
                kind = _markdown_text(str(item.get("kind") or "source"))
                reference = _markdown_text(str(item.get("reference") or ""))
                lines.append(f"- Evidence: {kind} `{reference}`")
        else:
            lines.append("- Evidence: missing")
        lines.append("")
    lines.extend(
        [
            "## Decisions Template",
            "",
            "```json",
            json.dumps(packet.decisions_template, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines)


def _markdown_value(value: object) -> str:
    if value is None:
        return "-"
    return _markdown_text(str(value))


def _markdown_text(value: str) -> str:
    return value.replace("\n", " ").strip()


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _checkout_prompt(brief: CoordinationBrief, *, include_diagnostics: bool) -> str:
    lines = [
        f"Coordination checkout for mission {brief.mission_id}.",
        f"Objective: {brief.objective or '-'}",
        "Accepted findings:",
    ]
    if brief.accepted_findings:
        for finding in brief.accepted_findings:
            lines.append(f"- {finding.finding_id} ({finding.worker_id}): {finding.summary}")
            for item in finding.evidence:
                reference = item.get("reference")
                if reference:
                    lines.append(f"  evidence: {reference}")
    else:
        lines.append("- none")
    if include_diagnostics:
        lines.append("Diagnostics:")
        lines.append(f"- pending findings: {len(brief.pending_findings)}")
        for finding in brief.pending_findings:
            lines.append(f"  - {finding.finding_id} ({finding.worker_id}): {finding.summary}")
        lines.append(f"- stale findings: {len(brief.stale_findings)}")
        for finding in brief.stale_findings:
            superseded = f" superseded_by={finding.superseded_by}" if finding.superseded_by else ""
            lines.append(f"  - {finding.finding_id} ({finding.worker_id}): {finding.summary}{superseded}")
        lines.append(f"- conflicts: {len(brief.conflicts)}")
        for conflict in brief.conflicts:
            ids = ", ".join(finding.finding_id for finding in conflict.findings)
            source = f" source={conflict.source_reference}" if conflict.source_reference else ""
            reason = f" reason={conflict.reason}" if conflict.reason else ""
            lines.append(f"  - {conflict.conflict_type} {conflict.claim_key}{source}{reason}: {ids}")
    return "\n".join(lines)
