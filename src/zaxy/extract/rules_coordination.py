"""Rule extractors: coordination.* and fleet.* events, plus shared fleet promotion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zaxy.event import Event
from zaxy.extract.core import (
    ExtractedEdge,
    ExtractedEntity,
    ExtractionResult,
    _bounded_float,
    _compact_properties,
    _coordination_finding_id,
    _coordination_mission_id,
    _coordination_proof_row_id,
    _coordination_worker_id,
    _dict_list,
    _join_summary,
    _optional_text,
    _required_text,
    _string_list,
    register,
)


@register("coordination.mission.created")
def _extract_coordination_mission_created(event: Event) -> ExtractionResult:
    """Extract a high-level coordinator mission."""
    mission_id = _coordination_mission_id(event)
    status = _optional_text(event.payload.get("status")) or "active"
    mission = ExtractedEntity(
        name=mission_id,
        entity_type="mission",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("objective")),
        properties=_compact_properties({"status": status}),
    )
    actor = ExtractedEntity(name=event.actor, entity_type="actor", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=mission_id,
        relation_type="started_mission",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, actor], edges=[edge], source_event_seq=event.seq)


@register("coordination.worker.created")
def _extract_coordination_worker_created(event: Event) -> ExtractionResult:
    """Extract a registered worker under a mission."""
    mission_id = _coordination_mission_id(event)
    worker_id = _coordination_worker_id(event)
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    worker = ExtractedEntity(
        name=worker_id,
        entity_type="worker",
        observed_at=event.timestamp,
        properties=_compact_properties(
            {
                "mission_id": mission_id,
                "status": _optional_text(event.payload.get("status")) or "active",
            }
        ),
    )
    edge = ExtractedEdge(
        source=mission_id,
        target=worker_id,
        relation_type="mission_has_worker",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, worker], edges=[edge], source_event_seq=event.seq)


@register("coordination.assignment.created")
def _extract_coordination_assignment_created(event: Event) -> ExtractionResult:
    """Extract a worker assignment."""
    mission_id = _coordination_mission_id(event)
    worker_id = _coordination_worker_id(event)
    assignment_id = (
        _optional_text(event.payload.get("assignment_id"))
        or f"{mission_id}:{worker_id}:assignment:{event.seq}"
    )
    assignment = ExtractedEntity(
        name=assignment_id,
        entity_type="assignment",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("assignment")),
        properties=_compact_properties(
            {
                "status": _optional_text(event.payload.get("status")) or "assigned",
                "mission_id": mission_id,
                "worker_id": worker_id,
            }
        ),
    )
    worker = ExtractedEntity(name=worker_id, entity_type="worker", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=worker_id,
        target=assignment_id,
        relation_type="worker_has_assignment",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[worker, assignment], edges=[edge], source_event_seq=event.seq)


@register("coordination.finding.reported")
def _extract_coordination_finding_reported(event: Event) -> ExtractionResult:
    """Extract a worker-local finding with evidence metadata."""
    mission_id = _coordination_mission_id(event)
    worker_id = _coordination_worker_id(event)
    finding_id = _coordination_finding_id(event)
    evidence = event.payload.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    finding = ExtractedEntity(
        name=finding_id,
        entity_type="finding",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties=_compact_properties(
            {
                "status": _optional_text(event.payload.get("status")) or "pending",
                "mission_id": mission_id,
                "worker_id": worker_id,
                "evidence_count": evidence_count,
                "confidence": _bounded_float(event.payload.get("confidence")),
                "claim_key": _optional_text(event.payload.get("claim_key")),
                "claim_value": _optional_text(event.payload.get("claim_value")),
            }
        ),
    )
    worker = ExtractedEntity(name=worker_id, entity_type="worker", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=worker_id,
        target=finding_id,
        relation_type="worker_reported_finding",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[worker, finding], edges=[edge], source_event_seq=event.seq)


@register("coordination.finding.reviewed")
def _extract_coordination_finding_reviewed(event: Event) -> ExtractionResult:
    """Extract a coordinator review of a worker finding."""
    mission_id = _coordination_mission_id(event)
    worker_id = _coordination_worker_id(event)
    finding_id = _coordination_finding_id(event)
    status = _optional_text(event.payload.get("status")) or "reviewed"
    review_id = f"{mission_id}:{finding_id}:review:{event.seq}"
    review = ExtractedEntity(
        name=review_id,
        entity_type="finding_review",
        observed_at=event.timestamp,
        summary=_join_summary(status, event.payload.get("rationale")),
        properties=_compact_properties({"status": status, "mission_id": mission_id, "worker_id": worker_id}),
    )
    finding = ExtractedEntity(name=finding_id, entity_type="finding", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=event.actor,
        target=finding_id,
        relation_type="coordinator_reviewed_finding",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[finding, review], edges=[edge], source_event_seq=event.seq)


@register("coordination.finding.promoted")
def _extract_coordination_finding_promoted(event: Event) -> ExtractionResult:
    """Extract accepted parent state promoted from a worker finding."""
    mission_id = _coordination_mission_id(event)
    finding_id = _coordination_finding_id(event)
    promotion_id = f"{mission_id}:{finding_id}:promotion:{event.seq}"
    promotion = ExtractedEntity(
        name=promotion_id,
        entity_type="promotion",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties=_compact_properties(
            {
                "status": _optional_text(event.payload.get("status")) or "accepted",
                "mission_id": mission_id,
                "worker_id": _optional_text(event.payload.get("worker_id")),
                "finding_id": finding_id,
                "source_event_seq": event.payload.get("source_event_seq"),
                "source_event_hash": _optional_text(event.payload.get("source_event_hash")),
                "claim_key": _optional_text(event.payload.get("claim_key")),
                "claim_value": _optional_text(event.payload.get("claim_value")),
            }
        ),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=finding_id,
        target=promotion_id,
        relation_type="finding_promoted_to_parent",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, promotion], edges=[edge], source_event_seq=event.seq)


@register("coordination.conflict.detected")
def _extract_coordination_conflict_detected(event: Event) -> ExtractionResult:
    """Extract a deterministic conflict between worker findings."""
    mission_id = _coordination_mission_id(event)
    conflict_id = _optional_text(event.payload.get("conflict_id")) or f"{mission_id}:conflict:{event.seq}"
    finding_ids = _string_list(event.payload.get("finding_ids"))
    conflict = ExtractedEntity(
        name=conflict_id,
        entity_type="conflict",
        observed_at=event.timestamp,
        summary=_optional_text(event.payload.get("summary")),
        properties=_compact_properties(
            {
                "mission_id": mission_id,
                "claim_key": _optional_text(event.payload.get("claim_key")),
                "conflict_type": _optional_text(event.payload.get("conflict_type")) or "exact_claim",
                "reason": _optional_text(event.payload.get("reason")),
                "source_reference": _optional_text(event.payload.get("source_reference")),
                "finding_count": len(finding_ids),
            }
        ),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    entities = [mission, conflict]
    edges = [
        ExtractedEdge(
            source=mission_id,
            target=conflict_id,
            relation_type="mission_has_conflict",
            valid_from=event.timestamp,
        )
    ]
    for finding_id in finding_ids:
        entities.append(ExtractedEntity(name=finding_id, entity_type="finding", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=finding_id,
                target=conflict_id,
                relation_type="finding_conflicts_with",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


@register("coordination.decision.recorded")
def _extract_coordination_decision_recorded(event: Event) -> ExtractionResult:
    """Extract a coordinator decision."""
    mission_id = _coordination_mission_id(event)
    decision_id = _optional_text(event.payload.get("decision_id")) or f"{mission_id}:decision:{event.seq}"
    decision = ExtractedEntity(
        name=decision_id,
        entity_type="decision",
        observed_at=event.timestamp,
        summary=_join_summary(event.payload.get("decision"), event.payload.get("rationale")),
        properties=_compact_properties({"mission_id": mission_id}),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=mission_id,
        target=decision_id,
        relation_type="mission_has_decision",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, decision], edges=[edge], source_event_seq=event.seq)


@register("coordination.handoff.created")
def _extract_coordination_handoff_created(event: Event) -> ExtractionResult:
    """Extract a final mission handoff."""
    mission_id = _coordination_mission_id(event)
    handoff_id = _optional_text(event.payload.get("handoff_id")) or f"{mission_id}:handoff:{event.seq}"
    handoff = ExtractedEntity(
        name=handoff_id,
        entity_type="handoff",
        observed_at=event.timestamp,
        summary=_join_summary(event.payload.get("summary"), event.payload.get("next_steps"), event.payload.get("risks")),
        properties=_compact_properties({"mission_id": mission_id, "status": "created"}),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    edge = ExtractedEdge(
        source=mission_id,
        target=handoff_id,
        relation_type="mission_has_handoff",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[mission, handoff], edges=[edge], source_event_seq=event.seq)


@register("coordination.proof_packet.created")
def _extract_coordination_proof_packet_created(event: Event) -> ExtractionResult:
    """Extract a mission-scoped synthesis proof packet for coordinator memory."""
    mission_id = _coordination_mission_id(event)
    artifact_id = _optional_text(event.payload.get("artifact_id"))
    proof_id = artifact_id or f"{mission_id}:proof_packet:{event.seq}"
    accepted_ids = _string_list(event.payload.get("accepted_finding_ids"))
    diagnostic_pending_ids = _string_list(event.payload.get("diagnostic_pending_ids"))
    conflict_ids = _string_list(event.payload.get("conflict_ids"))
    non_authoritative_rows = _dict_list(event.payload.get("non_authoritative_rows"))
    excluded_row_reasons = _dict_list(event.payload.get("excluded_row_reasons"))
    handoff_ref = event.payload.get("handoff_event_ref")
    handoff_id = _optional_text(handoff_ref.get("handoff_id")) if isinstance(handoff_ref, dict) else None
    proof = ExtractedEntity(
        name=proof_id,
        entity_type="coordination_proof_packet",
        observed_at=event.timestamp,
        summary=_join_summary(event.payload.get("query"), event.payload.get("decision_scope")),
        properties=_compact_properties(
            {
                "mission_id": mission_id,
                "schema_version": _optional_text(event.payload.get("schema_version")),
                "artifact_id": artifact_id,
                "decision_scope": _optional_text(event.payload.get("decision_scope")),
                "authority_scope": _optional_text(event.payload.get("authority_scope")),
                "accepted_finding_count": len(accepted_ids),
                "diagnostic_pending_count": len(diagnostic_pending_ids),
                "conflict_count": len(conflict_ids),
                "non_authoritative_row_count": len(non_authoritative_rows),
                "excluded_row_reason_count": len(excluded_row_reasons),
                "handoff_id": handoff_id,
            }
        ),
    )
    mission = ExtractedEntity(name=mission_id, entity_type="mission", observed_at=event.timestamp)
    entities = [mission, proof]
    edges = [
        ExtractedEdge(
            source=mission_id,
            target=proof_id,
            relation_type="mission_has_proof_packet",
            valid_from=event.timestamp,
        )
    ]
    if artifact_id:
        entities.append(ExtractedEntity(name=artifact_id, entity_type="synthesis_artifact", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=artifact_id,
                relation_type="proof_links_synthesis_artifact",
                valid_from=event.timestamp,
            )
        )
    for finding_id in accepted_ids:
        entities.append(ExtractedEntity(name=finding_id, entity_type="finding", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=finding_id,
                relation_type="proof_uses_accepted_finding",
                valid_from=event.timestamp,
            )
        )
    for row in non_authoritative_rows:
        row_id = _coordination_proof_row_id(proof_id, row)
        if row_id is None:
            continue
        entities.append(
            ExtractedEntity(
                name=row_id,
                entity_type="coordination_non_authoritative_row",
                observed_at=event.timestamp,
                summary=_join_summary(row.get("status"), row.get("include_reason"), row.get("exclude_reason")),
                properties=_compact_properties(
                    {
                        "mission_id": mission_id,
                        "proof_packet_id": proof_id,
                        "source_group": _optional_text(row.get("source_group")),
                        "fact_id": _optional_text(row.get("fact_id")),
                        "status": _optional_text(row.get("status")),
                        "include_reason": _optional_text(row.get("include_reason")),
                        "exclude_reason": _optional_text(row.get("exclude_reason")),
                    }
                ),
            )
        )
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=row_id,
                relation_type="proof_excludes_non_authoritative_row",
                valid_from=event.timestamp,
            )
        )
    for conflict_id in conflict_ids:
        entities.append(ExtractedEntity(name=conflict_id, entity_type="conflict", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=conflict_id,
                relation_type="proof_diagnoses_conflict",
                valid_from=event.timestamp,
            )
        )
    if handoff_id:
        entities.append(ExtractedEntity(name=handoff_id, entity_type="handoff", observed_at=event.timestamp))
        edges.append(
            ExtractedEdge(
                source=proof_id,
                target=handoff_id,
                relation_type="proof_binds_handoff",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(entities=entities, edges=edges, source_event_seq=event.seq)


_FLEET_AUTHORITY_STATUS = "non_authoritative"


def _fleet_source_event_refs(source_events: list[dict[str, Any]]) -> list[str]:
    """Return ``"<seq>:<hash>"`` citation strings for projected fleet sources."""
    refs: list[str] = []
    for source_event in source_events:
        seq = source_event.get("seq")
        event_hash = _optional_text(source_event.get("hash"))
        if isinstance(seq, int) and not isinstance(seq, bool) and event_hash:
            refs.append(f"{seq}:{event_hash}")
    return refs


def _fleet_promotion_result(
    event: Event,
    *,
    kind: str,
    summary: str | None,
    extra_properties: dict[str, Any],
) -> ExtractionResult:
    """Project a cited, non-authoritative fleet promotion into the queryable index.

    Every promotion entity carries ``visibility_scope`` + ``fleet_id`` +
    ``non_authoritative`` and cites its source events; ``review_status`` is the
    status recorded on the promotion event (``active``/``pending``) and is later
    updated in place by the review / supersede / rollback extractors, exactly as
    the consolidation-candidate lifecycle updates its candidate entity. Only
    ``active`` promotions are surfaced by the checkout fleet lane (which resolves
    live status from :class:`zaxy.fleet.FleetManager` replay).
    """
    payload = event.payload
    promotion_id = _required_text(
        payload.get("promotion_id"), field="promotion_id", event_seq=event.seq, event_type=event.type
    )
    fleet_id = _required_text(
        payload.get("fleet_id"), field="fleet_id", event_seq=event.seq, event_type=event.type
    )
    source_events = _dict_list(payload.get("source_events"))
    visibility_scope = _optional_text(payload.get("visibility_scope")) or "fleet"
    review_status = _optional_text(payload.get("review_status")) or "pending"
    authority_status = _optional_text(payload.get("authority_status")) or _FLEET_AUTHORITY_STATUS
    gate_event = payload.get("gate_event")
    properties = _compact_properties(
        {
            "promotion_id": promotion_id,
            "fleet_id": fleet_id,
            "kind": kind,
            "review_status": review_status,
            "visibility_scope": visibility_scope,
            "authority_status": authority_status,
            "non_authoritative": authority_status == _FLEET_AUTHORITY_STATUS,
            "confidence": _bounded_float(payload.get("confidence")),
            "origin_actor": _optional_text(payload.get("origin_actor")),
            "origin_session": _optional_text(payload.get("origin_session")),
            "keystone": bool(payload.get("keystone", False)),
            "gate_event": dict(gate_event) if isinstance(gate_event, Mapping) else None,
            "source_event_refs": _fleet_source_event_refs(source_events),
            "source_events": source_events,
            **extra_properties,
        }
    )
    promotion = ExtractedEntity(
        name=promotion_id,
        entity_type="fleet_promotion",
        observed_at=event.timestamp,
        summary=summary,
        properties=properties,
    )
    fleet = ExtractedEntity(
        name=f"fleet:{fleet_id}",
        entity_type="fleet",
        observed_at=event.timestamp,
        properties={"fleet_id": fleet_id},
    )
    edge = ExtractedEdge(
        source=f"fleet:{fleet_id}",
        target=promotion_id,
        relation_type=f"fleet_promoted_{kind}",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[fleet, promotion], edges=[edge], source_event_seq=event.seq)


@register("fleet.skill.promoted")
def _extract_fleet_skill_promoted(event: Event) -> ExtractionResult:
    """Project a fleet-promoted skill as cited, non-authoritative fleet memory."""
    skill_id = _optional_text(event.payload.get("skill_id"))
    skill_version = _optional_text(event.payload.get("skill_version"))
    summary = f"skill {skill_id}@{skill_version}" if skill_id else _optional_text(event.payload.get("summary"))
    return _fleet_promotion_result(
        event,
        kind="skill",
        summary=summary,
        extra_properties={"skill_id": skill_id, "skill_version": skill_version},
    )


@register("fleet.rule.propagated")
def _extract_fleet_rule_propagated(event: Event) -> ExtractionResult:
    """Project a fleet-propagated preventive rule as cited, non-authoritative memory."""
    rule = _optional_text(event.payload.get("rule"))
    trigger = _optional_text(event.payload.get("trigger"))
    return _fleet_promotion_result(
        event,
        kind="rule",
        summary=rule,
        extra_properties={
            "rule_id": _optional_text(event.payload.get("rule_id")),
            "rule": rule,
            "trigger": trigger,
        },
    )


@register("fleet.outcome.propagated")
def _extract_fleet_outcome_propagated(event: Event) -> ExtractionResult:
    """Project a fleet-propagated outcome as cited, non-authoritative memory."""
    return _fleet_promotion_result(
        event,
        kind="outcome",
        summary=_optional_text(event.payload.get("summary")),
        extra_properties={
            "outcome": _optional_text(event.payload.get("outcome")),
            "claim_key": _optional_text(event.payload.get("claim_key")),
        },
    )


def _fleet_review_status_update(event: Event, *, promotion_id: str | None, review_status: str) -> ExtractionResult:
    """Update a projected fleet promotion's ``review_status`` in place (additive)."""
    if not promotion_id:
        return ExtractionResult(entities=[], edges=[], source_event_seq=event.seq)
    promotion = ExtractedEntity(
        name=promotion_id,
        entity_type="fleet_promotion",
        observed_at=event.timestamp,
        properties={
            "review_status": review_status,
            "authority_status": _FLEET_AUTHORITY_STATUS,
        },
    )
    return ExtractionResult(entities=[promotion], edges=[], source_event_seq=event.seq)


@register("fleet.promotion.reviewed")
def _extract_fleet_promotion_reviewed(event: Event) -> ExtractionResult:
    """Project a steward review outcome onto the promotion it cites (no new authority)."""
    decision = _optional_text(event.payload.get("decision"))
    review_status = {"accepted": "active", "rejected": "rejected", "deferred": "deferred"}.get(
        decision or "", "pending"
    )
    return _fleet_review_status_update(
        event,
        promotion_id=_optional_text(event.payload.get("promotion_id")),
        review_status=review_status,
    )


@register("fleet.memory.superseded")
def _extract_fleet_memory_superseded(event: Event) -> ExtractionResult:
    """Project additive supersession: the prior promotion is retained as superseded."""
    return _fleet_review_status_update(
        event,
        promotion_id=_optional_text(event.payload.get("superseded_promotion_id")),
        review_status="superseded",
    )


@register("fleet.promotion.rolled_back")
def _extract_fleet_promotion_rolled_back(event: Event) -> ExtractionResult:
    """Project a reversible un-share: the promotion is retained as rolled_back."""
    return _fleet_review_status_update(
        event,
        promotion_id=_optional_text(event.payload.get("promotion_id")),
        review_status="rolled_back",
    )
