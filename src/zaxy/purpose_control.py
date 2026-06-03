"""Replay-only purpose control-plane diagnostics.

This module intentionally reads Eventloom logs directly. It must stay usable
when graph projection backends are unavailable so operators can inspect purpose
policy effects from the durable event stream alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zaxy.event import Event, EventLog
from zaxy.purpose import PurposeProfile, purpose_profile

PURPOSE_CHECKOUT_EVENTS = {"memory.checkout.completed"}
PURPOSE_FEEDBACK_EVENTS = {
    "memory.reinforced",
    "memory.feedback",
    "memory.evidence.reinforced",
    "memory.evidence.excluded",
}
POSITIVE_OUTCOMES = {"accepted", "helpful", "kept", "promoted", "reinforced", "used"}
NEGATIVE_OUTCOMES = {"contradicted", "excluded", "failed", "irrelevant", "rejected"}


@dataclass
class PurposeEventRef:
    """Stable pointer to the Eventloom event that produced a diagnostic row."""

    seq: int
    hash: str
    event_type: str
    session_id: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "hash": self.hash,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }


@dataclass
class PurposeLane:
    """Aggregated operator diagnostics for one purpose profile."""

    profile: str
    role: str
    evidence_policy: str
    checkout_count: int = 0
    evidence_policy_pass_count: int = 0
    evidence_policy_fail_count: int = 0
    suppressed_count: int = 0
    suppressed_reasons: Counter[str] = field(default_factory=Counter)
    warning_count: int = 0
    refresh_suggestions: list[dict[str, Any]] = field(default_factory=list)
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    latest_checkout: PurposeEventRef | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "role": self.role,
            "evidence_policy": self.evidence_policy,
            "checkout_count": self.checkout_count,
            "evidence_policy_pass_count": self.evidence_policy_pass_count,
            "evidence_policy_fail_count": self.evidence_policy_fail_count,
            "suppressed_count": self.suppressed_count,
            "suppressed_reasons": dict(sorted(self.suppressed_reasons.items())),
            "warning_count": self.warning_count,
            "refresh_suggestions": self.refresh_suggestions,
            "positive_feedback_count": self.positive_feedback_count,
            "negative_feedback_count": self.negative_feedback_count,
            "latest_checkout": self.latest_checkout.to_dict() if self.latest_checkout else None,
        }


@dataclass
class PurposeFeedbackTarget:
    """Outcome history for one cited memory target."""

    target: str
    profile: str
    positive_count: int = 0
    negative_count: int = 0
    outcomes: list[str] = field(default_factory=list)
    latest_event: PurposeEventRef | None = None

    @property
    def suppression_candidate(self) -> bool:
        return self.negative_count >= 2 and self.negative_count >= self.positive_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "profile": self.profile,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "outcomes": self.outcomes,
            "suppression_candidate": self.suppression_candidate,
            "latest_event": self.latest_event.to_dict() if self.latest_event else None,
        }


def build_purpose_status(
    path: str | Path,
    *,
    session_id: str | None = None,
    feedback_limit: int = 10,
) -> dict[str, Any]:
    """Build the maintainer purpose status view from Eventloom replay only."""
    events = _load_events(path, session_id=session_id)
    lanes = _lanes_from_events(events)
    feedback = _feedback_from_events(events)
    coordinate = _coordinate_summary(path, session_id=session_id)
    latest_lane = _latest_lane(lanes)
    latest_checkout = latest_lane.latest_checkout if latest_lane else None
    active_profile = latest_lane.profile if latest_lane else None
    payload = {
        "available": True,
        "read_only": True,
        "source_path": str(Path(path)),
        "session_id": session_id,
        "active_profile": active_profile,
        "latest_checkout": latest_checkout.to_dict() if latest_checkout else None,
        "evidence_policy_status": _evidence_policy_status(latest_lane),
        "suppression": _suppression_status(lanes),
        "refresh_suggestions": _bounded_suggestions(lanes),
        "consequence_history": {
            "positive_count": sum(item.positive_count for item in feedback),
            "negative_count": sum(item.negative_count for item in feedback),
            "targets": [item.to_dict() for item in feedback[:feedback_limit]],
        },
        "coordinate": coordinate,
        "lanes": [lane.to_dict() for lane in lanes],
    }
    return payload


def build_purpose_lanes(path: str | Path, *, session_id: str | None = None) -> dict[str, Any]:
    """Return per-profile purpose lanes from Eventloom replay only."""
    events = _load_events(path, session_id=session_id)
    lanes = _lanes_from_events(events)
    return {
        "available": True,
        "read_only": True,
        "source_path": str(Path(path)),
        "session_id": session_id,
        "lanes": [lane.to_dict() for lane in lanes],
    }


def build_purpose_feedback(
    path: str | Path,
    *,
    session_id: str | None = None,
    profile: str | None = None,
    outcome: str = "all",
    limit: int = 20,
) -> dict[str, Any]:
    """Return retained purpose consequence history from Eventloom replay only."""
    events = _load_events(path, session_id=session_id)
    targets = _feedback_from_events(events)
    if profile:
        normalized = purpose_profile(profile).profile
        targets = [target for target in targets if target.profile == normalized]
    if outcome != "all":
        targets = [
            target
            for target in targets
            if (outcome == "positive" and target.positive_count)
            or (outcome == "negative" and target.negative_count)
        ]
    return {
        "available": True,
        "read_only": True,
        "source_path": str(Path(path)),
        "session_id": session_id,
        "profile": profile,
        "outcome": outcome,
        "targets": [target.to_dict() for target in targets[:limit]],
    }


def format_purpose_status(status: dict[str, Any]) -> str:
    """Format the purpose status view for operators."""
    lines = ["Purpose control plane:"]
    active = status.get("active_profile") or "none"
    lines.append(f"  active profile: {active}")
    latest = status.get("latest_checkout")
    if isinstance(latest, dict):
        lines.append(
            "  latest checkout: "
            f"seq={latest.get('seq')} session={latest.get('session_id')} type={latest.get('event_type')}"
        )
    evidence = status.get("evidence_policy_status")
    if isinstance(evidence, dict):
        lines.append(
            "  evidence policy: "
            f"{evidence.get('status')} ({evidence.get('policy') or 'unknown'})"
        )
        missing = evidence.get("missing")
        if isinstance(missing, list) and missing:
            lines.append(f"  missing evidence: {', '.join(str(item) for item in missing)}")
    suppression = status.get("suppression")
    if isinstance(suppression, dict):
        lines.append(f"  suppressed rows: {suppression.get('count', 0)}")
        reasons = suppression.get("reasons")
        if isinstance(reasons, dict) and reasons:
            lines.append(
                "  suppressed reasons: "
                + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items()))
            )
    suggestions = status.get("refresh_suggestions")
    if isinstance(suggestions, list) and suggestions:
        lines.append("Refresh suggestions:")
        for item in suggestions[:5]:
            lines.append(f"  - {item.get('query') or item.get('reason') or item.get('type')}")
    consequence = status.get("consequence_history")
    if isinstance(consequence, dict):
        lines.append(
            "Consequence history: "
            f"+{consequence.get('positive_count', 0)} / -{consequence.get('negative_count', 0)}"
        )
    coordinate = status.get("coordinate")
    if isinstance(coordinate, dict) and coordinate.get("missions"):
        lines.append("Coordinate:")
        for mission in coordinate["missions"][:5]:
            lines.append(
                "  "
                f"{mission.get('mission_id')}: accepted={mission.get('accepted_count', 0)} "
                f"pending={mission.get('pending_count', 0)} stale={mission.get('stale_count', 0)} "
                f"conflicts={mission.get('conflict_count', 0)} proof_packets={mission.get('proof_packet_count', 0)}"
            )
    return "\n".join(lines)


def format_purpose_lanes(payload: dict[str, Any]) -> str:
    """Format purpose lanes for operators."""
    lines = ["Purpose lanes:"]
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        return "Purpose lanes:\n  none"
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lines.append(
            "  "
            f"{lane.get('profile')}: checkouts={lane.get('checkout_count', 0)} "
            f"evidence_failures={lane.get('evidence_policy_fail_count', 0)} "
            f"suppressed={lane.get('suppressed_count', 0)} "
            f"feedback=+{lane.get('positive_feedback_count', 0)}/-{lane.get('negative_feedback_count', 0)}"
        )
        reasons = lane.get("suppressed_reasons")
        if isinstance(reasons, dict) and reasons:
            lines.append("    reasons: " + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items())))
    return "\n".join(lines)


def format_purpose_feedback(payload: dict[str, Any]) -> str:
    """Format purpose feedback history for operators."""
    lines = ["Purpose feedback:"]
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        return "Purpose feedback:\n  none"
    for target in targets:
        if not isinstance(target, dict):
            continue
        marker = " suppression-candidate" if target.get("suppression_candidate") else ""
        lines.append(
            "  "
            f"{target.get('target')}: profile={target.get('profile')} "
            f"+{target.get('positive_count', 0)}/-{target.get('negative_count', 0)}{marker}"
        )
        outcomes = target.get("outcomes")
        if isinstance(outcomes, list) and outcomes:
            lines.append("    outcomes: " + ", ".join(str(item) for item in outcomes))
    return "\n".join(lines)


def _load_events(path: str | Path, *, session_id: str | None = None) -> list[Event]:
    events: list[Event] = []
    for log_path in _eventlog_paths(Path(path)):
        events.extend(EventLog(log_path).replay().events)
    if session_id:
        events = [event for event in events if _event_session_id(event) == session_id or event.thread == session_id]
    return sorted(events, key=lambda event: (event.timestamp, event.seq, event.hash))


def _eventlog_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(candidate for candidate in path.glob("*.jsonl") if candidate.is_file())
    return [path]


def _lanes_from_events(events: list[Event]) -> list[PurposeLane]:
    lanes: dict[str, PurposeLane] = {}
    for event in events:
        payload = _checkout_payload(event)
        if payload is None:
            continue
        profile = _payload_profile(payload)
        lane = lanes.setdefault(profile.profile, _lane_for_profile(profile))
        lane.checkout_count += 1
        lane.latest_checkout = _event_ref(event)
        _add_evidence_status(lane, payload)
        _add_suppression(lane, payload)
        warnings = payload.get("warnings")
        if isinstance(warnings, list):
            lane.warning_count += len(warnings)
        for suggestion in _refresh_suggestions(payload):
            _append_unique(lane.refresh_suggestions, suggestion)
    for target in _feedback_from_events(events):
        lane = lanes.setdefault(target.profile, _lane_for_profile(purpose_profile(target.profile)))
        lane.positive_feedback_count += target.positive_count
        lane.negative_feedback_count += target.negative_count
    return sorted(lanes.values(), key=lambda lane: lane.profile)


def _checkout_payload(event: Event) -> dict[str, Any] | None:
    if event.type not in PURPOSE_CHECKOUT_EVENTS:
        return None
    payload = event.payload
    nested = payload.get("payload")
    if isinstance(nested, dict):
        return nested
    return payload


def _payload_profile(payload: dict[str, Any]) -> PurposeProfile:
    purpose = payload.get("purpose")
    diagnostics = payload.get("diagnostics")
    if not purpose and isinstance(diagnostics, dict):
        purpose = diagnostics.get("purpose")
    return purpose_profile(purpose)


def _lane_for_profile(profile: PurposeProfile) -> PurposeLane:
    return PurposeLane(profile=profile.profile, role=profile.role, evidence_policy=profile.evidence_policy)


def _add_evidence_status(lane: PurposeLane, payload: dict[str, Any]) -> None:
    quality = payload.get("quality")
    required = quality.get("required_action") if isinstance(quality, dict) else None
    diagnostics = payload.get("diagnostics")
    evidence_policy = diagnostics.get("evidence_policy") if isinstance(diagnostics, dict) else None
    missing = evidence_policy.get("missing") if isinstance(evidence_policy, dict) else None
    status = evidence_policy.get("status") if isinstance(evidence_policy, dict) else None
    if required or (isinstance(missing, list) and missing) or status in {"failed", "missing", "blocked"}:
        lane.evidence_policy_fail_count += 1
    else:
        lane.evidence_policy_pass_count += 1


def _add_suppression(lane: PurposeLane, payload: dict[str, Any]) -> None:
    retention = payload.get("retention")
    diagnostics = payload.get("diagnostics")
    policy = None
    if isinstance(retention, dict):
        policy = retention.get("purpose_policy")
    if policy is None and isinstance(diagnostics, dict):
        policy = diagnostics.get("purpose_policy")
    if not isinstance(policy, dict):
        return
    count = _int(policy.get("suppressed_count"))
    lane.suppressed_count += count
    reasons = policy.get("suppressed_reasons")
    if isinstance(reasons, dict):
        for reason, value in reasons.items():
            lane.suppressed_reasons[str(reason)] += _int(value)
    examples = policy.get("suppressed_examples")
    if isinstance(examples, list):
        for example in examples:
            if isinstance(example, dict) and example.get("reason"):
                lane.suppressed_reasons[str(example["reason"])] += 1


def _refresh_suggestions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    quality = payload.get("quality")
    if isinstance(quality, dict) and isinstance(quality.get("required_action"), dict):
        suggestions.append(dict(quality["required_action"]))
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        evidence_policy = diagnostics.get("evidence_policy")
        if isinstance(evidence_policy, dict):
            for query in evidence_policy.get("suggested_queries") or []:
                suggestions.append({"type": "memory_checkout", "query": str(query)})
        evidence_plan = diagnostics.get("evidence_plan")
        if isinstance(evidence_plan, dict):
            for query in evidence_plan.get("suggested_queries") or []:
                suggestions.append({"type": "memory_checkout", "query": str(query)})
    guidance = payload.get("guidance")
    if isinstance(guidance, dict) and isinstance(guidance.get("recommended_next_call"), dict):
        suggestions.append(dict(guidance["recommended_next_call"]))
    return suggestions


def _feedback_from_events(events: list[Event]) -> list[PurposeFeedbackTarget]:
    targets: dict[tuple[str, str], PurposeFeedbackTarget] = {}
    for event in events:
        if event.type not in PURPOSE_FEEDBACK_EVENTS:
            continue
        payload = event.payload
        profile = _feedback_profile(payload)
        if profile is None:
            continue
        outcome = _feedback_outcome(event.type, payload)
        polarity = _outcome_polarity(outcome)
        if polarity is None:
            continue
        keys = _feedback_target_keys(payload)
        for key in keys:
            target = targets.setdefault((profile, key), PurposeFeedbackTarget(target=key, profile=profile))
            if polarity == "positive":
                target.positive_count += 1
            else:
                target.negative_count += 1
            if outcome and outcome not in target.outcomes:
                target.outcomes.append(outcome)
            target.latest_event = _event_ref(event)
    return sorted(
        targets.values(),
        key=lambda item: (
            -item.negative_count,
            -item.positive_count,
            item.latest_event.seq if item.latest_event else 0,
            item.target,
        ),
    )


def _feedback_profile(payload: dict[str, Any]) -> str | None:
    purpose = payload.get("purpose")
    if isinstance(purpose, dict):
        return purpose_profile(purpose).profile
    if isinstance(purpose, str) and purpose.strip():
        return purpose_profile(purpose).profile
    return None


def _feedback_outcome(event_type: str, payload: dict[str, Any]) -> str | None:
    for key in ("outcome", "feedback"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold().replace(" ", "_")
    if event_type in {"memory.reinforced", "memory.evidence.reinforced"}:
        return "used"
    if event_type == "memory.evidence.excluded":
        return "excluded"
    return None


def _outcome_polarity(outcome: str | None) -> str | None:
    if outcome in POSITIVE_OUTCOMES:
        return "positive"
    if outcome in NEGATIVE_OUTCOMES:
        return "negative"
    return None


def _feedback_target_keys(payload: dict[str, Any]) -> list[str]:
    for key, prefix in (
        ("citation", "citation"),
        ("source_event_hash", "hash"),
        ("source_group", "source_group"),
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return [f"{prefix}:{value.strip()}"]
    entity_name = payload.get("entity_name")
    entity_type = payload.get("entity_type")
    if isinstance(entity_name, str) and isinstance(entity_type, str) and entity_name.strip() and entity_type.strip():
        return [f"entity:{entity_type.strip()}:{entity_name.strip()}"]
    target = payload.get("target") or payload.get("target_id")
    if isinstance(target, str) and target.strip():
        return [f"target:{target.strip()}"]
    return []


def _coordinate_summary(path: str | Path, *, session_id: str | None) -> dict[str, Any]:
    try:
        from zaxy.coordination import CoordinationManager
    except ImportError:
        return {"available": False, "missions": []}
    source = Path(path)
    eventloom_path = source if source.is_dir() else source.parent
    manager = CoordinationManager(eventloom_path=eventloom_path)
    mission_ids = sorted(_mission_ids(_load_events(path, session_id=session_id)))
    missions = []
    for mission_id in mission_ids:
        try:
            brief = manager.brief(mission_id)
            packet = manager.approval_packet(mission_id)
        except ValueError:
            continue
        proof_packets = [
            event
            for event in _load_events(path, session_id=mission_id)
            if event.type == "coordination.proof_packet.created"
        ]
        missions.append(
            {
                "mission_id": mission_id,
                "objective": brief.objective,
                "accepted_count": len(brief.accepted_findings),
                "pending_count": len(brief.pending_findings),
                "rejected_count": len(brief.rejected_findings),
                "deferred_count": len(brief.deferred_findings),
                "conflicted_count": len(brief.conflicted_findings),
                "stale_count": len(brief.stale_findings),
                "conflict_count": len(brief.conflicts),
                "proof_packet_count": len(proof_packets),
                "approval_packet_count": len(packet.findings),
                "accepted_findings": [finding.to_dict() for finding in brief.accepted_findings],
                "pending_findings": [finding.to_dict() for finding in brief.pending_findings],
                "stale_findings": [finding.to_dict() for finding in brief.stale_findings],
            }
        )
    return {"available": True, "missions": missions}


def _mission_ids(events: list[Event]) -> set[str]:
    mission_ids: set[str] = set()
    for event in events:
        if event.type.startswith("coordination."):
            mission_id = event.payload.get("mission_id") or event.thread
            if isinstance(mission_id, str) and mission_id.strip():
                mission_ids.add(mission_id.strip())
    return mission_ids


def _latest_lane(lanes: list[PurposeLane]) -> PurposeLane | None:
    checkout_lanes = [lane for lane in lanes if lane.latest_checkout is not None]
    if not checkout_lanes:
        return None
    return max(checkout_lanes, key=lambda lane: lane.latest_checkout.seq if lane.latest_checkout else 0)


def _evidence_policy_status(lane: PurposeLane | None) -> dict[str, Any]:
    if lane is None:
        return {"status": "missing", "policy": None, "missing": ["no purpose checkout events found"]}
    status = "ok" if lane.evidence_policy_fail_count == 0 else "needs_refresh"
    return {
        "status": status,
        "policy": lane.evidence_policy,
        "passed": lane.evidence_policy_pass_count,
        "failed": lane.evidence_policy_fail_count,
        "missing": [] if status == "ok" else ["checkout has missing or stale evidence requirements"],
    }


def _suppression_status(lanes: list[PurposeLane]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    count = 0
    for lane in lanes:
        count += lane.suppressed_count
        reasons.update(lane.suppressed_reasons)
    return {"count": count, "reasons": dict(sorted(reasons.items()))}


def _bounded_suggestions(lanes: list[PurposeLane], *, limit: int = 10) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for lane in lanes:
        for suggestion in lane.refresh_suggestions:
            _append_unique(suggestions, suggestion)
            if len(suggestions) >= limit:
                return suggestions
    return suggestions


def _event_ref(event: Event) -> PurposeEventRef:
    return PurposeEventRef(
        seq=event.seq,
        hash=event.hash,
        event_type=event.type,
        session_id=_event_session_id(event),
        timestamp=event.timestamp,
    )


def _event_session_id(event: Event) -> str:
    value = event.payload.get("session_id") or event.payload.get("subagent_session_id") or event.thread
    return str(value)


def _append_unique(items: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if item not in items:
        items.append(item)


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0
