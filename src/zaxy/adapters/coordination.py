"""Dependency-light adapter contract for Zaxy Coordinate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zaxy.coordination import CoordinationEventResult, CoordinationManager


@dataclass(frozen=True)
class CoordinationAdapter:
    """JSON-friendly wrapper around the replay-backed coordination manager.

    Framework integrations should pass explicit structured fields into this
    adapter. It intentionally does not infer findings from transcripts or spawn
    worker processes.
    """

    eventloom_path: str | Path = ".eventloom"
    actor: str = "coordinator"

    def _manager(self) -> CoordinationManager:
        return CoordinationManager(eventloom_path=self.eventloom_path)

    def start_mission(self, mission_id: str, *, objective: str) -> dict[str, Any]:
        return _event_result_payload(
            self._manager().start_mission(mission_id, objective=objective, actor=self.actor)
        )

    def create_worker(self, mission_id: str, worker_id: str) -> dict[str, Any]:
        return _event_result_payload(
            self._manager().create_worker(mission_id, worker_id, actor=self.actor)
        )

    def assign(self, mission_id: str, worker_id: str, assignment: str) -> dict[str, Any]:
        return _event_result_payload(
            self._manager().assign(mission_id, worker_id, assignment, actor=self.actor)
        )

    def report_finding(
        self,
        mission_id: str,
        worker_id: str,
        *,
        summary: str,
        evidence: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        claim_key: str | None = None,
        claim_value: str | None = None,
        finding_id: str | None = None,
    ) -> dict[str, Any]:
        safe_evidence = _validate_evidence(evidence or [])
        return _event_result_payload(
            self._manager().report_finding(
                mission_id,
                worker_id,
                summary=summary,
                actor=self.actor,
                evidence=safe_evidence,
                confidence=confidence,
                claim_key=claim_key,
                claim_value=claim_value,
                finding_id=finding_id,
            )
        )

    def brief(self, mission_id: str) -> dict[str, Any]:
        return self._manager().brief(mission_id).to_dict()

    def checkout(self, mission_id: str, *, include_diagnostics: bool = False) -> dict[str, Any]:
        return self._manager().checkout(mission_id, include_diagnostics=include_diagnostics).to_dict()

    def approval_packet(self, mission_id: str) -> dict[str, Any]:
        return self._manager().approval_packet(mission_id).to_dict()

    def apply_approval(self, mission_id: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        return self._manager().apply_approval_decisions(
            mission_id,
            decisions,
            actor=self.actor,
        ).to_dict()

    def handoff(
        self,
        mission_id: str,
        *,
        summary: str,
        next_steps: list[str] | None = None,
        risks: list[str] | None = None,
    ) -> dict[str, Any]:
        return _event_result_payload(
            self._manager().create_handoff(
                mission_id,
                summary=summary,
                actor=self.actor,
                next_steps=next_steps,
                risks=risks,
            )
        )


def _event_result_payload(result: CoordinationEventResult) -> dict[str, Any]:
    return {
        "event_type": result.event.type,
        "event_seq": result.event.seq,
        "event_hash": result.event.hash,
        "mission_id": result.mission_id,
        "worker_id": result.worker_id,
        "finding_id": result.finding_id,
        "handoff_id": result.handoff_id,
        "summary": result.summary,
        "evidence": list(result.evidence),
    }


def _validate_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or not item:
            raise ValueError(f"evidence item {index} must be a nonempty object")
        safe.append(dict(item))
    return safe
