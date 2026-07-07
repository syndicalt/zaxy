"""Fleet memory plane: governed cross-agent/cross-session propagation (Zaxy 3 / I7).

An outcome, preventive rule, or skill learned by one agent becomes governed,
cited, replayable knowledge for an entire fleet. A fleet memory is never a copy
or mutation of the source; it is a new, **non-authoritative** event appended to a
dedicated Eventloom thread (``fleet.<fleet_id>``) that **cites** the originating
events and the I4 evolution gate (op ``promote``) it routed through.

Three independent controls compose on every crossing (design §1b/§3):

1. **Trust tier** — *authorization*: caps the visibility scope an agent may
   *propose* into. Assigned by audited, reversible ``fleet.trust.assigned`` events.
2. **I4 gate** — *autonomy*: ``evaluate_evolution_gate("promote", confidence)``
   decides whether a *valid* proposal auto-applies or is held for review.
3. **Steward review** — a steward accepts/rejects *held* proposals.

Promotion raises *visibility scope*, never *authority*: every ``fleet.*`` event
carries ``authority_status="non_authoritative"``. Conflicts are resolved by
**additive supersession** (never delete/overwrite); forgetting is a reversible
``fleet.promotion.rolled_back``. Provenance ("which agent taught the fleet this,
from what evidence") is a deterministic replay query, not a side log.

The pure builders mirror :mod:`zaxy.outcome_learning`; :class:`FleetManager` is the
sync twin of :class:`zaxy.coordination.CoordinationManager`, reusing the shipped
:mod:`zaxy.evolution_policy` gate and the :mod:`zaxy.coordination` conflict
detectors rather than inventing a second convention.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zaxy.coordination import (
    FindingState,
    SemanticConflictDetector,
    _detect_conflicts,
)
from zaxy.event import Event
from zaxy.evolution_policy import (
    build_evolution_gate_event,
    evaluate_evolution_gate,
    resolve_evolution_policy,
)
from zaxy.outcome_learning import validate_outcome
from zaxy.security import validate_payload, validate_session_id
from zaxy.session import SessionManager

_AUTHORITY_STATUS = "non_authoritative"
_HASH_RE = "0123456789abcdef"

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Visibility scope ladder (conservative-first). Controls *which projection
#: replays a memory* — metadata, not authority.
VISIBILITY_SCOPES: tuple[str, ...] = ("private", "session", "mission", "fleet", "global")
DEFAULT_VISIBILITY_SCOPE = "session"

#: Agent trust tiers. Authorization for *what scope an agent may propose into*.
TRUST_TIERS: tuple[str, ...] = ("untrusted", "member", "trusted", "steward")
DEFAULT_TRUST_TIER = "member"

#: The ``EVOLUTION_OPS`` member every fleet crossing routes through (I4 gate).
PROMOTE_OP = "promote"

#: Reserved deployment-wide fleet id (single Eventloom root; federation out of scope).
GLOBAL_FLEET_ID = "global"

#: Promotion kinds carried on the plane.
PROMOTION_KINDS: tuple[str, ...] = ("skill", "outcome", "rule")

#: Review-status domain for a fleet memory.
REVIEW_STATUSES: tuple[str, ...] = (
    "pending",
    "active",
    "rejected",
    "deferred",
    "superseded",
    "rolled_back",
)

#: Steward review decisions.
REVIEW_DECISIONS: tuple[str, ...] = ("accepted", "rejected", "deferred")

# Registration / governance event types.
FLEET_CREATED_EVENT_TYPE = "fleet.created"
FLEET_AGENT_ENROLLED_EVENT_TYPE = "fleet.agent.enrolled"
FLEET_TRUST_ASSIGNED_EVENT_TYPE = "fleet.trust.assigned"

# Propagation (the plane) event types.
FLEET_SKILL_PROMOTED_EVENT_TYPE = "fleet.skill.promoted"
FLEET_OUTCOME_PROPAGATED_EVENT_TYPE = "fleet.outcome.propagated"
FLEET_RULE_PROPAGATED_EVENT_TYPE = "fleet.rule.propagated"

# Lifecycle (governance, conflict, reversal) event types.
FLEET_PROMOTION_REVIEWED_EVENT_TYPE = "fleet.promotion.reviewed"
FLEET_PROMOTION_ROLLED_BACK_EVENT_TYPE = "fleet.promotion.rolled_back"
FLEET_MEMORY_SUPERSEDED_EVENT_TYPE = "fleet.memory.superseded"

#: The three promotion event types that carry a fleet memory.
FLEET_PROMOTION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        FLEET_SKILL_PROMOTED_EVENT_TYPE,
        FLEET_OUTCOME_PROPAGATED_EVENT_TYPE,
        FLEET_RULE_PROPAGATED_EVENT_TYPE,
    }
)

_EVENT_TYPE_KIND: dict[str, str] = {
    FLEET_SKILL_PROMOTED_EVENT_TYPE: "skill",
    FLEET_OUTCOME_PROPAGATED_EVENT_TYPE: "outcome",
    FLEET_RULE_PROPAGATED_EVENT_TYPE: "rule",
}

#: Highest visibility scope each trust tier may *propose* a crossing into.
_MAX_PROPOSABLE_SCOPE: dict[str, str] = {
    "untrusted": "session",
    "member": "fleet",
    "trusted": "global",
    "steward": "global",
}


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_visibility_scope(scope: object) -> str:
    """Return a valid visibility scope or raise ``ValueError``."""
    if scope not in VISIBILITY_SCOPES:
        raise ValueError(f"visibility_scope must be one of: {', '.join(VISIBILITY_SCOPES)}")
    return str(scope)


def validate_trust_tier(tier: object) -> str:
    """Return a valid trust tier or raise ``ValueError``."""
    if tier not in TRUST_TIERS:
        raise ValueError(f"trust_tier must be one of: {', '.join(TRUST_TIERS)}")
    return str(tier)


def max_proposable_scope(trust_tier: str) -> str:
    """Return the highest visibility scope ``trust_tier`` may propose a crossing into."""
    return _MAX_PROPOSABLE_SCOPE[validate_trust_tier(trust_tier)]


def scope_permits_proposal(trust_tier: str, scope: str) -> bool:
    """Return whether ``trust_tier`` is authorized to propose a crossing to ``scope``."""
    ceiling = VISIBILITY_SCOPES.index(max_proposable_scope(trust_tier))
    return VISIBILITY_SCOPES.index(validate_visibility_scope(scope)) <= ceiling


def fleet_thread(fleet_id: str) -> str:
    """Return the Eventloom thread for a fleet (``':'`` is illegal in session ids)."""
    return f"fleet.{validate_session_id(fleet_id)}"


def promotion_id(
    *,
    fleet_id: str,
    kind: str,
    origin_session: str,
    source_events: Sequence[Mapping[str, Any]],
) -> str:
    """Deterministic promotion id: same source promoted twice is idempotent."""
    identity = {
        "fleet_id": validate_session_id(fleet_id),
        "kind": _validate_kind(kind),
        "origin_session": validate_session_id(origin_session),
        "source_events": _cite_source_events(source_events),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"fleetpromo:{digest}"


def _validate_kind(kind: object) -> str:
    if kind not in PROMOTION_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(PROMOTION_KINDS)}")
    return str(kind)


def _validate_review_decision(decision: object) -> str:
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of: {', '.join(REVIEW_DECISIONS)}")
    return str(decision)


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_unit_interval(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return float(value)


def _snapshot_event_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(ref, Mapping):
        raise ValueError("event ref must be a mapping with seq and hash")
    seq = ref.get("seq")
    event_hash = ref.get("hash")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ValueError("event ref seq must be a positive integer")
    if (
        not isinstance(event_hash, str)
        or len(event_hash) != 64
        or any(c not in _HASH_RE for c in event_hash)
    ):
        raise ValueError("event ref hash must be a 64-character hex digest")
    return {"seq": seq, "hash": event_hash}


def _cite_source_events(source_events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cited = [_snapshot_event_ref(ref) for ref in source_events]
    if not cited:
        raise ValueError("source_events must cite at least one event")
    return cited


def _fleet_rule_id(rule: str, trigger: str, source_events: Sequence[Mapping[str, Any]]) -> str:
    identity = {"rule": rule, "trigger": trigger, "source_events": list(source_events)}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"rule:{digest}"


# ---------------------------------------------------------------------------
# Pure builders (mirror outcome_learning.build_rule_event)
# ---------------------------------------------------------------------------


def build_fleet_created_event(*, actor: str, fleet_id: str, summary: str) -> dict[str, Any]:
    """Build a ``fleet.created`` event spec."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    summary = _require_str(summary, "summary")
    return {
        "event_type": FLEET_CREATED_EVENT_TYPE,
        "actor": actor,
        "payload": {
            "fleet_id": fid,
            "summary": summary,
            "authority_status": _AUTHORITY_STATUS,
        },
        "thread": fleet_thread(fid),
    }


def build_fleet_agent_enrolled_event(
    *, actor: str, fleet_id: str, agent_id: str, trust_tier: str = DEFAULT_TRUST_TIER
) -> dict[str, Any]:
    """Build a ``fleet.agent.enrolled`` event spec."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    aid = validate_session_id(agent_id)
    tier = validate_trust_tier(trust_tier)
    return {
        "event_type": FLEET_AGENT_ENROLLED_EVENT_TYPE,
        "actor": actor,
        "payload": {
            "fleet_id": fid,
            "agent_id": aid,
            "trust_tier": tier,
            "authority_status": _AUTHORITY_STATUS,
        },
        "thread": fleet_thread(fid),
    }


def build_fleet_trust_assigned_event(
    *,
    actor: str,
    fleet_id: str,
    agent_id: str,
    trust_tier: str,
    prior_tier: str,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Build a ``fleet.trust.assigned`` event spec (steward authority)."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    aid = validate_session_id(agent_id)
    tier = validate_trust_tier(trust_tier)
    prior = validate_trust_tier(prior_tier)
    payload: dict[str, Any] = {
        "fleet_id": fid,
        "agent_id": aid,
        "trust_tier": tier,
        "prior_tier": prior,
        "authority_status": _AUTHORITY_STATUS,
    }
    if rationale is not None:
        payload["rationale"] = _require_str(rationale, "rationale")
    return {
        "event_type": FLEET_TRUST_ASSIGNED_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": fleet_thread(fid),
    }


def build_fleet_skill_promoted_event(
    *,
    actor: str,
    fleet_id: str,
    skill_id: str,
    skill_version: str,
    origin_session: str,
    origin_actor: str,
    source_events: Sequence[Mapping[str, Any]],
    gate_event: Mapping[str, Any],
    confidence: float,
    auto_applied: bool,
    visibility_scope: str = "fleet",
    keystone: bool = False,
) -> dict[str, Any]:
    """Build a ``fleet.skill.promoted`` event spec citing source + gate events."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    skill_id = _require_str(skill_id, "skill_id")
    skill_version = _require_str(skill_version, "skill_version")
    origin = validate_session_id(origin_session)
    origin_actor = _require_str(origin_actor, "origin_actor")
    scope = validate_visibility_scope(visibility_scope)
    conf = _validate_unit_interval(confidence, field_name="confidence")
    cited = _cite_source_events(source_events)
    gate = _snapshot_event_ref(gate_event)
    payload = {
        "promotion_id": promotion_id(
            fleet_id=fid, kind="skill", origin_session=origin, source_events=cited
        ),
        "fleet_id": fid,
        "skill_id": skill_id,
        "skill_version": skill_version,
        "origin_session": origin,
        "origin_actor": origin_actor,
        "visibility_scope": scope,
        "confidence": conf,
        "source_events": cited,
        "gate_event": gate,
        "review_status": "active" if auto_applied else "pending",
        "keystone": bool(keystone),
        "authority_status": _AUTHORITY_STATUS,
    }
    return {
        "event_type": FLEET_SKILL_PROMOTED_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": fleet_thread(fid),
    }


def build_fleet_outcome_propagated_event(
    *,
    actor: str,
    fleet_id: str,
    outcome: str,
    summary: str,
    origin_session: str,
    origin_actor: str,
    source_events: Sequence[Mapping[str, Any]],
    gate_event: Mapping[str, Any],
    confidence: float,
    auto_applied: bool,
    claim_key: str | None = None,
    visibility_scope: str = "fleet",
) -> dict[str, Any]:
    """Build a ``fleet.outcome.propagated`` event spec citing source + gate events."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    outcome = validate_outcome(outcome)
    summary = _require_str(summary, "summary")
    origin = validate_session_id(origin_session)
    origin_actor = _require_str(origin_actor, "origin_actor")
    scope = validate_visibility_scope(visibility_scope)
    conf = _validate_unit_interval(confidence, field_name="confidence")
    cited = _cite_source_events(source_events)
    gate = _snapshot_event_ref(gate_event)
    payload: dict[str, Any] = {
        "promotion_id": promotion_id(
            fleet_id=fid, kind="outcome", origin_session=origin, source_events=cited
        ),
        "fleet_id": fid,
        "outcome": outcome,
        "summary": summary,
        "origin_session": origin,
        "origin_actor": origin_actor,
        "visibility_scope": scope,
        "confidence": conf,
        "source_events": cited,
        "gate_event": gate,
        "review_status": "active" if auto_applied else "pending",
        "authority_status": _AUTHORITY_STATUS,
    }
    if claim_key is not None:
        payload["claim_key"] = _require_str(claim_key, "claim_key")
    return {
        "event_type": FLEET_OUTCOME_PROPAGATED_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": fleet_thread(fid),
    }


def build_fleet_rule_propagated_event(
    *,
    actor: str,
    fleet_id: str,
    rule: str,
    trigger: str,
    origin_session: str,
    origin_actor: str,
    source_events: Sequence[Mapping[str, Any]],
    gate_event: Mapping[str, Any],
    confidence: float,
    auto_applied: bool,
    visibility_scope: str = "fleet",
    keystone: bool = False,
) -> dict[str, Any]:
    """Build a ``fleet.rule.propagated`` event spec citing source + gate events."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    rule = _require_str(rule, "rule")
    trigger = _require_str(trigger, "trigger")
    origin = validate_session_id(origin_session)
    origin_actor = _require_str(origin_actor, "origin_actor")
    scope = validate_visibility_scope(visibility_scope)
    conf = _validate_unit_interval(confidence, field_name="confidence")
    cited = _cite_source_events(source_events)
    gate = _snapshot_event_ref(gate_event)
    payload = {
        "promotion_id": promotion_id(
            fleet_id=fid, kind="rule", origin_session=origin, source_events=cited
        ),
        "fleet_id": fid,
        "rule_id": _fleet_rule_id(rule, trigger, cited),
        "rule": rule,
        "trigger": trigger,
        "origin_session": origin,
        "origin_actor": origin_actor,
        "visibility_scope": scope,
        "confidence": conf,
        "source_events": cited,
        "gate_event": gate,
        "review_status": "active" if auto_applied else "pending",
        "keystone": bool(keystone),
        "authority_status": _AUTHORITY_STATUS,
    }
    return {
        "event_type": FLEET_RULE_PROPAGATED_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": fleet_thread(fid),
    }


def build_fleet_promotion_reviewed_event(
    *,
    actor: str,
    fleet_id: str,
    promotion_id: str,
    decision: str,
    source_events: Sequence[Mapping[str, Any]],
    rationale: str | None = None,
) -> dict[str, Any]:
    """Build a ``fleet.promotion.reviewed`` event spec (steward decision)."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    pid = _require_str(promotion_id, "promotion_id")
    decision = _validate_review_decision(decision)
    cited = _cite_source_events(source_events)
    payload: dict[str, Any] = {
        "fleet_id": fid,
        "promotion_id": pid,
        "decision": decision,
        "source_events": cited,
        "authority_status": _AUTHORITY_STATUS,
    }
    if rationale is not None:
        payload["rationale"] = _require_str(rationale, "rationale")
    return {
        "event_type": FLEET_PROMOTION_REVIEWED_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": fleet_thread(fid),
    }


def build_fleet_promotion_rolled_back_event(
    *,
    actor: str,
    fleet_id: str,
    promotion_id: str,
    reason: str,
    within_rollback_window: bool,
    source_events: Sequence[Mapping[str, Any]],
    gate_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``fleet.promotion.rolled_back`` event spec (reversible un-share)."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    pid = _require_str(promotion_id, "promotion_id")
    reason = _require_str(reason, "reason")
    cited = _cite_source_events(source_events)
    payload: dict[str, Any] = {
        "fleet_id": fid,
        "promotion_id": pid,
        "reason": reason,
        "within_rollback_window": bool(within_rollback_window),
        "source_events": cited,
        "authority_status": _AUTHORITY_STATUS,
    }
    if gate_event is not None:
        payload["gate_event"] = _snapshot_event_ref(gate_event)
    return {
        "event_type": FLEET_PROMOTION_ROLLED_BACK_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": fleet_thread(fid),
    }


def build_fleet_memory_superseded_event(
    *,
    actor: str,
    fleet_id: str,
    superseded_promotion_id: str,
    superseding_promotion_id: str,
    reason: str,
    source_events: Sequence[Mapping[str, Any]],
    claim_key: str | None = None,
    skill_id: str | None = None,
) -> dict[str, Any]:
    """Build a ``fleet.memory.superseded`` event spec (additive, retains both)."""
    actor = _require_str(actor, "actor")
    fid = validate_session_id(fleet_id)
    superseded = _require_str(superseded_promotion_id, "superseded_promotion_id")
    superseding = _require_str(superseding_promotion_id, "superseding_promotion_id")
    reason = _require_str(reason, "reason")
    cited = _cite_source_events(source_events)
    payload: dict[str, Any] = {
        "fleet_id": fid,
        "superseded_promotion_id": superseded,
        "superseding_promotion_id": superseding,
        "reason": reason,
        "source_events": cited,
        "authority_status": _AUTHORITY_STATUS,
    }
    if claim_key is not None:
        payload["claim_key"] = _require_str(claim_key, "claim_key")
    if skill_id is not None:
        payload["skill_id"] = _require_str(skill_id, "skill_id")
    if "claim_key" not in payload and "skill_id" not in payload:
        raise ValueError("fleet.memory.superseded requires claim_key or skill_id")
    return {
        "event_type": FLEET_MEMORY_SUPERSEDED_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": fleet_thread(fid),
    }


# ---------------------------------------------------------------------------
# Replay state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetAgentState:
    """An enrolled fleet agent with its current trust tier."""

    agent_id: str
    trust_tier: str

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "trust_tier": self.trust_tier}


@dataclass(frozen=True)
class FleetMemoryState:
    """A replayed fleet memory with its effective review status and provenance."""

    promotion_id: str
    fleet_id: str
    kind: str
    review_status: str
    visibility_scope: str
    confidence: float | None
    summary: str
    origin_session: str
    origin_actor: str
    actor: str
    source_events: list[dict[str, Any]]
    gate_event: dict[str, Any] | None
    keystone: bool
    conflict_key: str | None
    conflict_value: str | None
    event_seq: int
    event_hash: str
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def skill_id(self) -> str | None:
        return self.details.get("skill_id")

    @property
    def claim_key(self) -> str | None:
        return self.details.get("claim_key")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "promotion_id": self.promotion_id,
            "fleet_id": self.fleet_id,
            "kind": self.kind,
            "review_status": self.review_status,
            "visibility_scope": self.visibility_scope,
            "confidence": self.confidence,
            "summary": self.summary,
            "origin_session": self.origin_session,
            "origin_actor": self.origin_actor,
            "actor": self.actor,
            "source_events": self.source_events,
            "gate_event": self.gate_event,
            "keystone": self.keystone,
            "event_seq": self.event_seq,
            "event_hash": self.event_hash,
            "timestamp": self.timestamp,
        }
        payload.update(self.details)
        return payload


@dataclass(frozen=True)
class FleetProjection:
    """Replay-derived fleet state: created summary, enrolled agents, all memories."""

    fleet_id: str | None
    summary: str | None
    created_actor: str | None
    agents: list[FleetAgentState]
    memories: list[FleetMemoryState]

    def trust_tier(self, agent_id: str) -> str | None:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent.trust_tier
        return None

    def memory(self, promotion_id: str) -> FleetMemoryState | None:
        for memory in self.memories:
            if memory.promotion_id == promotion_id:
                return memory
        return None

    def active_memories(self) -> list[FleetMemoryState]:
        return [memory for memory in self.memories if memory.review_status == "active"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "summary": self.summary,
            "created_actor": self.created_actor,
            "agents": [agent.to_dict() for agent in self.agents],
            "memories": [memory.to_dict() for memory in self.memories],
        }


def _ev_type(event: Any) -> str:
    if isinstance(event, Mapping):
        return str(event.get("type") or event.get("event_type") or "")
    return str(event.type)


def _ev_payload(event: Any) -> Mapping[str, Any]:
    payload = event.get("payload") if isinstance(event, Mapping) else event.payload
    return payload or {}


def _ev_attr(event: Any, key: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(key)
    return getattr(event, key, None)


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _parse_promotion(event: Any, kind: str) -> dict[str, Any]:
    payload = _ev_payload(event)
    details: dict[str, Any]
    if kind == "skill":
        skill_id = _opt_str(payload.get("skill_id"))
        skill_version = _opt_str(payload.get("skill_version"))
        details = {"skill_id": skill_id, "skill_version": skill_version}
        conflict_key = f"skill:{skill_id}" if skill_id else None
        conflict_value = skill_version
        summary = f"skill {skill_id}@{skill_version}"
    elif kind == "outcome":
        outcome = _opt_str(payload.get("outcome"))
        claim_key = _opt_str(payload.get("claim_key"))
        details = {"outcome": outcome, "claim_key": claim_key}
        conflict_key = claim_key
        conflict_value = outcome
        summary = _opt_str(payload.get("summary")) or ""
    else:  # rule
        rule = _opt_str(payload.get("rule"))
        trigger = _opt_str(payload.get("trigger"))
        details = {
            "rule_id": _opt_str(payload.get("rule_id")),
            "rule": rule,
            "trigger": trigger,
        }
        conflict_key = f"rule:{trigger}" if trigger else None
        conflict_value = rule
        summary = rule or ""
    gate_event = payload.get("gate_event")
    return {
        "promotion_id": str(payload.get("promotion_id") or ""),
        "fleet_id": str(payload.get("fleet_id") or ""),
        "kind": kind,
        "visibility_scope": str(payload.get("visibility_scope") or "fleet"),
        "confidence": payload.get("confidence"),
        "summary": summary,
        "origin_session": str(payload.get("origin_session") or ""),
        "origin_actor": str(payload.get("origin_actor") or ""),
        "actor": str(_ev_attr(event, "actor") or ""),
        "source_events": _dict_list(payload.get("source_events")),
        "gate_event": dict(gate_event) if isinstance(gate_event, Mapping) else None,
        "keystone": bool(payload.get("keystone", False)),
        "conflict_key": conflict_key,
        "conflict_value": conflict_value,
        "event_seq": int(_ev_attr(event, "seq") or 0),
        "event_hash": str(_ev_attr(event, "hash") or ""),
        "timestamp": str(_ev_attr(event, "timestamp") or ""),
        "details": details,
    }


def summarize_fleet_events(
    events: Iterable[Any], *, fleet_id: str | None = None
) -> FleetProjection:
    """Project a fleet thread's events into replay-derived fleet state.

    Honors review/supersession/rollback lifecycle: a rolled-back promotion that
    had superseded an earlier memory re-activates that earlier memory.
    """
    summary: str | None = None
    created_actor: str | None = None
    agents: dict[str, str] = {}
    enroll_order: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    status: dict[str, str] = {}
    superseded_by: dict[str, str] = {}

    for event in events:
        event_type = _ev_type(event)
        payload = _ev_payload(event)
        if fleet_id is None:
            candidate = payload.get("fleet_id")
            if isinstance(candidate, str) and candidate:
                fleet_id = candidate
        if event_type == FLEET_CREATED_EVENT_TYPE:
            summary = _opt_str(payload.get("summary"))
            created_actor = _opt_str(_ev_attr(event, "actor"))
        elif event_type in (FLEET_AGENT_ENROLLED_EVENT_TYPE, FLEET_TRUST_ASSIGNED_EVENT_TYPE):
            agent_id = str(payload.get("agent_id") or "")
            if agent_id:
                if agent_id not in agents:
                    enroll_order.append(agent_id)
                agents[agent_id] = str(payload.get("trust_tier") or DEFAULT_TRUST_TIER)
        elif event_type in FLEET_PROMOTION_EVENT_TYPES:
            kind = _EVENT_TYPE_KIND[event_type]
            record = _parse_promotion(event, kind)
            pid = record["promotion_id"]
            if not pid:
                continue
            if pid not in records:
                order.append(pid)
                records[pid] = record
                status[pid] = "active" if payload.get("review_status") == "active" else "pending"
            # A duplicate promotion event for an already-known promotion_id is
            # inert: status transitions only through review / supersede / rollback
            # events, so a re-cited source can never demote a memory that already
            # reached `active` (its initial auto-apply or a steward acceptance);
            # the first (canonical) record is retained.
        elif event_type == FLEET_PROMOTION_REVIEWED_EVENT_TYPE:
            pid = str(payload.get("promotion_id") or "")
            decision = str(payload.get("decision") or "")
            if pid in status:
                if decision == "accepted":
                    status[pid] = "active"
                elif decision == "rejected":
                    status[pid] = "rejected"
                elif decision == "deferred":
                    status[pid] = "deferred"
        elif event_type == FLEET_MEMORY_SUPERSEDED_EVENT_TYPE:
            old = str(payload.get("superseded_promotion_id") or "")
            new = str(payload.get("superseding_promotion_id") or "")
            if old in status:
                status[old] = "superseded"
                superseded_by[old] = new
        elif event_type == FLEET_PROMOTION_ROLLED_BACK_EVENT_TYPE:
            pid = str(payload.get("promotion_id") or "")
            if pid in status:
                status[pid] = "rolled_back"
                for old, winner in superseded_by.items():
                    if winner == pid and status.get(old) == "superseded":
                        status[old] = "active"

    memories = [_memory_from_record(records[pid], status[pid]) for pid in order]
    agent_states = [FleetAgentState(agent_id=a, trust_tier=agents[a]) for a in enroll_order]
    return FleetProjection(
        fleet_id=fleet_id,
        summary=summary,
        created_actor=created_actor,
        agents=agent_states,
        memories=memories,
    )


def _memory_from_record(record: dict[str, Any], review_status: str) -> FleetMemoryState:
    return FleetMemoryState(
        promotion_id=record["promotion_id"],
        fleet_id=record["fleet_id"],
        kind=record["kind"],
        review_status=review_status,
        visibility_scope=record["visibility_scope"],
        confidence=record["confidence"],
        summary=record["summary"],
        origin_session=record["origin_session"],
        origin_actor=record["origin_actor"],
        actor=record["actor"],
        source_events=record["source_events"],
        gate_event=record["gate_event"],
        keystone=record["keystone"],
        conflict_key=record["conflict_key"],
        conflict_value=record["conflict_value"],
        event_seq=record["event_seq"],
        event_hash=record["event_hash"],
        timestamp=record["timestamp"],
        details=record["details"],
    )


def resolve_fleet_skills(
    events: Iterable[Any], *, fleet_id: str | None = None
) -> list[FleetMemoryState]:
    """Return the active promoted skill memories of a fleet (honoring lifecycle)."""
    projection = summarize_fleet_events(events, fleet_id=fleet_id)
    return [
        memory
        for memory in projection.active_memories()
        if memory.kind == "skill"
    ]


def _memory_to_finding(memory: FleetMemoryState) -> FindingState:
    return FindingState(
        finding_id=memory.promotion_id,
        mission_id=memory.fleet_id,
        worker_id=memory.origin_actor or "fleet",
        summary=memory.summary or "",
        evidence=[],
        confidence=memory.confidence,
        claim_key=memory.conflict_key,
        claim_value=memory.conflict_value,
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _event_ref(event: Event | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {"seq": event.seq, "hash": event.hash}


@dataclass(frozen=True)
class FleetEventResult:
    """A sealed governance/lifecycle fleet event plus operator-facing fields."""

    event: Event
    fleet_id: str
    actor: str
    agent_id: str | None = None
    trust_tier: str | None = None
    promotion_id: str | None = None
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "actor": self.actor,
            "agent_id": self.agent_id,
            "trust_tier": self.trust_tier,
            "promotion_id": self.promotion_id,
            "summary": self.summary,
            "event_seq": self.event.seq,
            "event_hash": self.event.hash,
        }


@dataclass(frozen=True)
class FleetEnrollmentResult:
    """An agent enrollment at its requested trust tier.

    ``bootstrap_steward`` / ``bootstrap_event`` are retained for result-shape
    stability and are always ``False`` / ``None``: enrollment never escalates
    trust (the fleet creator is the implicit steward, recorded at creation).
    """

    event: Event
    fleet_id: str
    agent_id: str
    actor: str
    trust_tier: str
    bootstrap_steward: bool = False
    bootstrap_event: Event | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "agent_id": self.agent_id,
            "actor": self.actor,
            "trust_tier": self.trust_tier,
            "bootstrap_steward": self.bootstrap_steward,
            "event_seq": self.event.seq,
            "event_hash": self.event.hash,
            "bootstrap_event": _event_ref(self.bootstrap_event),
        }


@dataclass(frozen=True)
class FleetPromotionResult:
    """The outcome of a gated cross-boundary promotion proposal."""

    fleet_id: str
    kind: str
    promotion_id: str | None
    rejected: bool
    review_status: str | None
    auto_applied: bool
    reason: str | None = None
    actor: str | None = None
    gate_event: Event | None = None
    promotion_event: Event | None = None
    supersessions: list[Event] = field(default_factory=list)
    gate_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "kind": self.kind,
            "promotion_id": self.promotion_id,
            "rejected": self.rejected,
            "reason": self.reason,
            "review_status": self.review_status,
            "auto_applied": self.auto_applied,
            "actor": self.actor,
            "gate_event": _event_ref(self.gate_event),
            "promotion_event": _event_ref(self.promotion_event),
            "supersessions": [
                {
                    "superseded_promotion_id": event.payload.get("superseded_promotion_id"),
                    "superseding_promotion_id": event.payload.get("superseding_promotion_id"),
                    "event_seq": event.seq,
                    "event_hash": event.hash,
                }
                for event in self.supersessions
            ],
            "gate_decision": self.gate_decision,
        }


@dataclass(frozen=True)
class FleetBrief:
    """Operator- and prompt-ready governed fleet state."""

    fleet_id: str
    summary: str | None
    agents: list[FleetAgentState]
    active_promotions: list[FleetMemoryState]
    pending_promotions: list[FleetMemoryState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "summary": self.summary,
            "agents": [agent.to_dict() for agent in self.agents],
            "active_promotions": [memory.to_dict() for memory in self.active_promotions],
            "pending_promotions": [memory.to_dict() for memory in self.pending_promotions],
        }


@dataclass(frozen=True)
class FleetAuditRecord:
    """Full provenance for one fleet memory: who taught it, from what evidence."""

    promotion_id: str
    kind: str
    review_status: str
    visibility_scope: str
    origin_actor: str
    origin_session: str
    actor: str
    confidence: float | None
    summary: str
    source_events: list[dict[str, Any]]
    gate_event: dict[str, Any] | None
    keystone: bool
    event_seq: int
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion_id": self.promotion_id,
            "kind": self.kind,
            "review_status": self.review_status,
            "visibility_scope": self.visibility_scope,
            "origin_actor": self.origin_actor,
            "origin_session": self.origin_session,
            "actor": self.actor,
            "confidence": self.confidence,
            "summary": self.summary,
            "source_events": self.source_events,
            "gate_event": self.gate_event,
            "keystone": self.keystone,
            "event_seq": self.event_seq,
            "event_hash": self.event_hash,
        }


@dataclass(frozen=True)
class FleetAuditReport:
    """Replay-only fleet audit report with Eventloom citations."""

    fleet_id: str
    records: list[FleetAuditRecord]

    def active_records(self) -> list[FleetAuditRecord]:
        return [record for record in self.records if record.review_status == "active"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "records": [record.to_dict() for record in self.records],
        }


def _audit_record(memory: FleetMemoryState) -> FleetAuditRecord:
    return FleetAuditRecord(
        promotion_id=memory.promotion_id,
        kind=memory.kind,
        review_status=memory.review_status,
        visibility_scope=memory.visibility_scope,
        origin_actor=memory.origin_actor,
        origin_session=memory.origin_session,
        actor=memory.actor,
        confidence=memory.confidence,
        summary=memory.summary,
        source_events=memory.source_events,
        gate_event=memory.gate_event,
        keystone=memory.keystone,
        event_seq=memory.event_seq,
        event_hash=memory.event_hash,
    )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class FleetManager:
    """Manage governed fleet propagation over an Eventloom fleet thread.

    The sync twin of :class:`zaxy.coordination.CoordinationManager`: every
    cross-boundary promotion routes through the shipped I4 gate
    (:func:`zaxy.evolution_policy.evaluate_evolution_gate`, op ``promote``), the
    gate event is appended to the fleet thread and cited by the ``fleet.*`` event,
    and conflicts are detected with the :mod:`zaxy.coordination` claim-key matcher.
    """

    def __init__(
        self,
        eventloom_path: str | Path = ".eventloom",
        *,
        settings: Any | None = None,
        semantic_conflict_detector: SemanticConflictDetector | None = None,
    ) -> None:
        self.session_manager = SessionManager(base_path=str(eventloom_path))
        self._settings = settings
        self._semantic_conflict_detector = semantic_conflict_detector

    # -- internal append -------------------------------------------------

    def _append(self, spec: Mapping[str, Any]) -> Event:
        thread = spec["thread"]
        return self.session_manager.get(thread).eventlog.append(
            spec["event_type"],
            actor=spec["actor"],
            payload=validate_payload(dict(spec["payload"])),
            thread=thread,
        )

    def _project(self, fleet_id: str) -> FleetProjection:
        events = self.session_manager.replay(fleet_thread(fleet_id)).events
        return summarize_fleet_events(events, fleet_id=fleet_id)

    # -- registration & governance --------------------------------------

    def create_fleet(self, fleet_id: str, *, summary: str, actor: str = "coordinator") -> FleetEventResult:
        """Create a fleet thread; the creator is recorded as its implicit steward.

        Enrolling the creator at tier ``steward`` guarantees a steward always
        exists from creation, so :meth:`enroll_agent` never needs to escalate
        trust (no first-enrollee bootstrap).
        """
        fid = validate_session_id(fleet_id)
        creator = validate_session_id(actor)
        event = self._append(build_fleet_created_event(actor=actor, fleet_id=fid, summary=summary))
        self._append(
            build_fleet_agent_enrolled_event(
                actor=actor, fleet_id=fid, agent_id=creator, trust_tier="steward"
            )
        )
        return FleetEventResult(event=event, fleet_id=fid, actor=actor, summary=summary)

    def enroll_agent(
        self,
        fleet_id: str,
        agent_id: str,
        *,
        actor: str = "coordinator",
        trust_tier: str = DEFAULT_TRUST_TIER,
    ) -> FleetEnrollmentResult:
        """Enroll an agent at its requested trust tier (steward authority required).

        Enrollment never escalates trust: an ``untrusted`` enrollee stays
        ``untrusted`` and a ``member`` stays ``member``. Enrollment is a
        governance act: once a fleet has any enrolled agents, only a steward
        may enroll — otherwise any actor could mint itself (or an accomplice)
        a ``steward``. An empty fleet keeps the trust-on-first-use bootstrap
        posture of :meth:`create_fleet`, whose creator is the implicit steward.
        """
        fid = validate_session_id(fleet_id)
        aid = validate_session_id(agent_id)
        tier = validate_trust_tier(trust_tier)
        actor = _require_str(actor, "actor")
        projection = self._project(fid)
        if projection.agents and projection.trust_tier(actor) != "steward":
            raise ValueError(
                f"enroll_agent requires a steward actor; '{actor}' is not a steward of fleet '{fid}'"
            )
        event = self._append(
            build_fleet_agent_enrolled_event(actor=actor, fleet_id=fid, agent_id=aid, trust_tier=tier)
        )
        return FleetEnrollmentResult(
            event=event,
            fleet_id=fid,
            agent_id=aid,
            actor=actor,
            trust_tier=tier,
        )

    def assign_trust(
        self,
        fleet_id: str,
        agent_id: str,
        *,
        trust_tier: str,
        actor: str,
        rationale: str | None = None,
    ) -> FleetEventResult:
        """Assign a trust tier to an enrolled agent (steward authority required)."""
        fid = validate_session_id(fleet_id)
        aid = validate_session_id(agent_id)
        tier = validate_trust_tier(trust_tier)
        actor = _require_str(actor, "actor")
        projection = self._project(fid)
        if projection.trust_tier(actor) != "steward":
            raise ValueError(
                f"assign_trust requires a steward actor; '{actor}' is not a steward of fleet '{fid}'"
            )
        prior = projection.trust_tier(aid)
        if prior is None:
            raise ValueError(f"cannot assign trust to '{aid}': not enrolled in fleet '{fid}'")
        event = self._append(
            build_fleet_trust_assigned_event(
                actor=actor,
                fleet_id=fid,
                agent_id=aid,
                trust_tier=tier,
                prior_tier=prior,
                rationale=rationale,
            )
        )
        return FleetEventResult(
            event=event, fleet_id=fid, actor=actor, agent_id=aid, trust_tier=tier, summary=rationale
        )

    # -- propagation (the gated crossing) -------------------------------

    def promote_skill(
        self,
        fleet_id: str,
        *,
        skill_id: str,
        skill_version: str,
        origin_session: str,
        source_events: Sequence[Mapping[str, Any]],
        confidence: float,
        actor: str,
        origin_actor: str | None = None,
        visibility_scope: str = "fleet",
        keystone: bool = False,
    ) -> FleetPromotionResult:
        """Propose a skill promotion to fleet scope through the I4 gate."""
        skill_id = _require_str(skill_id, "skill_id")
        skill_version = _require_str(skill_version, "skill_version")

        def build_event(
            *,
            gate_event: Mapping[str, Any],
            auto_applied: bool,
            origin_session: str,
            origin_actor: str,
            source_events: Sequence[Mapping[str, Any]],
            confidence: float,
            visibility_scope: str,
            fleet_id: str,
        ) -> dict[str, Any]:
            return build_fleet_skill_promoted_event(
                actor=actor,
                fleet_id=fleet_id,
                skill_id=skill_id,
                skill_version=skill_version,
                origin_session=origin_session,
                origin_actor=origin_actor,
                source_events=source_events,
                gate_event=gate_event,
                confidence=confidence,
                auto_applied=auto_applied,
                visibility_scope=visibility_scope,
                keystone=keystone,
            )

        return self._propose(
            fleet_id=fleet_id,
            kind="skill",
            actor=actor,
            origin_session=origin_session,
            origin_actor=origin_actor,
            source_events=source_events,
            confidence=confidence,
            visibility_scope=visibility_scope,
            conflict_key=f"skill:{skill_id}",
            conflict_value=skill_version,
            candidate_name=f"skill {skill_id}@{skill_version}",
            build_event=build_event,
        )

    def propagate_outcome(
        self,
        fleet_id: str,
        *,
        outcome: str,
        summary: str,
        origin_session: str,
        source_events: Sequence[Mapping[str, Any]],
        confidence: float,
        actor: str,
        origin_actor: str | None = None,
        claim_key: str | None = None,
        visibility_scope: str = "fleet",
    ) -> FleetPromotionResult:
        """Propose an outcome propagation to fleet scope through the I4 gate."""
        outcome_value = validate_outcome(outcome)
        summary = _require_str(summary, "summary")
        claim_key_value = _require_str(claim_key, "claim_key") if claim_key is not None else None

        def build_event(
            *,
            gate_event: Mapping[str, Any],
            auto_applied: bool,
            origin_session: str,
            origin_actor: str,
            source_events: Sequence[Mapping[str, Any]],
            confidence: float,
            visibility_scope: str,
            fleet_id: str,
        ) -> dict[str, Any]:
            return build_fleet_outcome_propagated_event(
                actor=actor,
                fleet_id=fleet_id,
                outcome=outcome_value,
                summary=summary,
                origin_session=origin_session,
                origin_actor=origin_actor,
                source_events=source_events,
                gate_event=gate_event,
                confidence=confidence,
                auto_applied=auto_applied,
                claim_key=claim_key_value,
                visibility_scope=visibility_scope,
            )

        return self._propose(
            fleet_id=fleet_id,
            kind="outcome",
            actor=actor,
            origin_session=origin_session,
            origin_actor=origin_actor,
            source_events=source_events,
            confidence=confidence,
            visibility_scope=visibility_scope,
            conflict_key=claim_key_value,
            conflict_value=outcome_value,
            candidate_name=summary,
            build_event=build_event,
        )

    def propagate_rule(
        self,
        fleet_id: str,
        *,
        rule: str,
        trigger: str,
        origin_session: str,
        source_events: Sequence[Mapping[str, Any]],
        confidence: float,
        actor: str,
        origin_actor: str | None = None,
        visibility_scope: str = "fleet",
        keystone: bool = False,
    ) -> FleetPromotionResult:
        """Propose a preventive-rule propagation to fleet scope through the I4 gate."""
        rule_value = _require_str(rule, "rule")
        trigger_value = _require_str(trigger, "trigger")

        def build_event(
            *,
            gate_event: Mapping[str, Any],
            auto_applied: bool,
            origin_session: str,
            origin_actor: str,
            source_events: Sequence[Mapping[str, Any]],
            confidence: float,
            visibility_scope: str,
            fleet_id: str,
        ) -> dict[str, Any]:
            return build_fleet_rule_propagated_event(
                actor=actor,
                fleet_id=fleet_id,
                rule=rule_value,
                trigger=trigger_value,
                origin_session=origin_session,
                origin_actor=origin_actor,
                source_events=source_events,
                gate_event=gate_event,
                confidence=confidence,
                auto_applied=auto_applied,
                visibility_scope=visibility_scope,
                keystone=keystone,
            )

        return self._propose(
            fleet_id=fleet_id,
            kind="rule",
            actor=actor,
            origin_session=origin_session,
            origin_actor=origin_actor,
            source_events=source_events,
            confidence=confidence,
            visibility_scope=visibility_scope,
            conflict_key=f"rule:{trigger_value}",
            conflict_value=rule_value,
            candidate_name=rule_value,
            build_event=build_event,
        )

    def propose_promotion(self, fleet_id: str, kind: str, **fields: Any) -> FleetPromotionResult:
        """Dispatch a promotion by kind (``skill`` / ``outcome`` / ``rule``)."""
        kind = _validate_kind(kind)
        if kind == "skill":
            return self.promote_skill(fleet_id, **fields)
        if kind == "outcome":
            return self.propagate_outcome(fleet_id, **fields)
        return self.propagate_rule(fleet_id, **fields)

    def _propose(
        self,
        *,
        fleet_id: str,
        kind: str,
        actor: str,
        origin_session: str,
        origin_actor: str | None,
        source_events: Sequence[Mapping[str, Any]],
        confidence: float,
        visibility_scope: str,
        conflict_key: str | None,
        conflict_value: str | None,
        candidate_name: str,
        build_event: Any,
    ) -> FleetPromotionResult:
        """Compose the independent controls on a single crossing.

        1. trust-tier authorization (reject -> NO event on the fleet thread);
        2. idempotency: an already-active promotion is returned unchanged, never
           re-appended (a re-cited source must not demote it);
        3. the I4 gate (append the gate event, cite it);
        4. conflict posture: a conflict against an active keystone holds the new
           promotion ``pending`` for a steward (a live mandatory rule is never
           auto-flipped);
        5. emit the ``fleet.*`` event citing source + gate, then run additive
           supersession of non-keystone conflicts when the promotion auto-applies.
        """
        fid = validate_session_id(fleet_id)
        thread = fleet_thread(fid)
        actor = _require_str(actor, "actor")
        origin = validate_session_id(origin_session)
        origin_actor_value = _require_str(origin_actor or actor, "origin_actor")
        scope = validate_visibility_scope(visibility_scope)
        if scope not in ("fleet", "global"):
            raise ValueError("fleet promotion visibility_scope must be 'fleet' or 'global'")
        conf = _validate_unit_interval(confidence, field_name="confidence")
        cited = _cite_source_events(source_events)

        # 1. trust-tier authorization
        projection = self._project(fid)
        tier = projection.trust_tier(actor)
        if tier is None or not scope_permits_proposal(tier, scope):
            reason = (
                f"insufficient trust: actor '{actor}' "
                + (f"(tier '{tier}')" if tier else "(not enrolled)")
                + f" may not propose visibility_scope '{scope}'"
                + (f"; max proposable scope is '{max_proposable_scope(tier)}'" if tier else "")
            )
            return FleetPromotionResult(
                fleet_id=fid,
                kind=kind,
                promotion_id=None,
                rejected=True,
                review_status=None,
                auto_applied=False,
                reason=reason,
                actor=actor,
            )

        pid = promotion_id(fleet_id=fid, kind=kind, origin_session=origin, source_events=cited)

        # 1b. Idempotency / status-lifecycle integrity. A promotion that already
        # reached `active` — via its initial auto-apply or a steward acceptance —
        # is NOT re-appended: a duplicate plain promotion event (a re-cited source)
        # would otherwise demote it. Return the existing active state without
        # mutating the log (no gate event, no demoting duplicate).
        existing = projection.memory(pid)
        if existing is not None and existing.review_status == "active":
            return FleetPromotionResult(
                fleet_id=fid,
                kind=kind,
                promotion_id=pid,
                rejected=False,
                review_status="active",
                auto_applied=True,
                reason="idempotent: promotion already active; re-cited source does not demote it",
                actor=actor,
            )

        # 2. the I4 gate — append to the fleet thread, then cite it
        candidate_ref = {
            "candidate_id": pid,
            "name": candidate_name,
            "entity_type": f"fleet.{kind}",
            "seq": cited[0]["seq"],
            "hash": cited[0]["hash"],
        }
        policy = resolve_evolution_policy(self._settings)
        decision = evaluate_evolution_gate(PROMOTE_OP, conf, policy=policy)

        # 2b. Conflict posture: a conflict against an ACTIVE keystone is never
        # auto-flipped. Even when the gate would auto-apply, the new promotion is
        # held `pending` for a steward and the live keystone is left untouched
        # (no supersession). Detected BEFORE deciding auto-apply -> active.
        held_for_keystone = False
        if decision.auto_apply and conflict_key is not None:
            candidate_finding = FindingState(
                finding_id=pid,
                mission_id=fid,
                worker_id=origin_actor_value or "fleet",
                summary=candidate_name or "",
                evidence=[],
                confidence=conf,
                claim_key=conflict_key,
                claim_value=conflict_value,
            )
            conflicting = self._conflicting_actives(
                projection,
                candidate_pid=pid,
                candidate_finding=candidate_finding,
                candidate_value=conflict_value,
            )
            held_for_keystone = any(memory.keystone for memory in conflicting)

        auto_apply = decision.auto_apply and not held_for_keystone

        gate_event = self._append(
            build_evolution_gate_event(
                actor=actor, session_id=thread, decision=decision, candidate_ref=candidate_ref
            )
        )
        gate_ref = {"seq": gate_event.seq, "hash": gate_event.hash}

        # 3. emit the fleet.* event citing source + gate
        spec = build_event(
            gate_event=gate_ref,
            auto_applied=auto_apply,
            origin_session=origin,
            origin_actor=origin_actor_value,
            source_events=cited,
            confidence=conf,
            visibility_scope=scope,
            fleet_id=fid,
        )
        promotion_event = self._append(spec)

        supersessions: list[Event] = []
        if auto_apply and conflict_key is not None:
            supersessions = self._supersede_conflicts(
                fid, pid, conflict_key, conflict_value, actor
            )

        return FleetPromotionResult(
            fleet_id=fid,
            kind=kind,
            promotion_id=pid,
            rejected=False,
            review_status="active" if auto_apply else "pending",
            auto_applied=auto_apply,
            reason=(
                "held pending steward review: conflicts with an active keystone"
                if held_for_keystone
                else None
            ),
            actor=actor,
            gate_event=gate_event,
            promotion_event=promotion_event,
            supersessions=supersessions,
            gate_decision=decision.to_payload(),
        )

    def _conflicting_actives(
        self,
        projection: FleetProjection,
        *,
        candidate_pid: str,
        candidate_finding: FindingState,
        candidate_value: str | None,
    ) -> list[FleetMemoryState]:
        """Active fleet memories the candidate exactly conflicts with.

        An ``exact_claim`` conflict — same ``conflict_key``, differing
        ``conflict_value`` — found with the shared coordination conflict matcher,
        returned deterministically ordered by ``promotion_id``.
        """
        actives = [
            memory
            for memory in projection.active_memories()
            if memory.promotion_id != candidate_pid and memory.conflict_key is not None
        ]
        if not actives:
            return []
        findings = [_memory_to_finding(memory) for memory in actives]
        findings.append(candidate_finding)
        conflicts = _detect_conflicts(
            findings, semantic_conflict_detector=self._semantic_conflict_detector
        )
        conflicting: dict[str, FleetMemoryState] = {}
        for conflict in conflicts:
            if conflict.conflict_type != "exact_claim":
                continue
            ids = {finding.finding_id for finding in conflict.findings}
            if candidate_pid not in ids:
                continue
            for memory in actives:
                if memory.promotion_id in ids and memory.conflict_value != candidate_value:
                    conflicting[memory.promotion_id] = memory
        return [conflicting[pid] for pid in sorted(conflicting)]

    def _supersede_conflicts(
        self,
        fleet_id: str,
        candidate_pid: str,
        conflict_key: str | None,
        conflict_value: str | None,
        actor: str,
    ) -> list[Event]:
        """Emit additive ``fleet.memory.superseded`` for conflicting active memories.

        An active **keystone** is never auto-superseded — a live mandatory rule
        only ever changes through explicit steward action — so keystone conflicts
        are skipped here (and held ``pending`` upstream in :meth:`_propose`).
        """
        if conflict_key is None:
            return []
        projection = self._project(fleet_id)
        candidate = projection.memory(candidate_pid)
        if candidate is None or candidate.review_status != "active":
            return []
        conflicting = self._conflicting_actives(
            projection,
            candidate_pid=candidate_pid,
            candidate_finding=_memory_to_finding(candidate),
            candidate_value=candidate.conflict_value,
        )
        events: list[Event] = []
        for old in conflicting:
            if old.keystone:
                continue
            events.append(
                self._append(
                    build_fleet_memory_superseded_event(
                        actor=actor,
                        fleet_id=fleet_id,
                        superseded_promotion_id=old.promotion_id,
                        superseding_promotion_id=candidate_pid,
                        reason=f"superseded by {candidate_pid} on conflicting {old.conflict_key}",
                        source_events=[
                            {"seq": candidate.event_seq, "hash": candidate.event_hash},
                            {"seq": old.event_seq, "hash": old.event_hash},
                        ],
                        claim_key=None if old.kind == "skill" else old.conflict_key,
                        skill_id=old.skill_id,
                    )
                )
            )
        return events

    # -- review / rollback ----------------------------------------------

    def review_promotion(
        self,
        fleet_id: str,
        promotion_id: str,
        *,
        decision: str,
        actor: str,
        rationale: str | None = None,
    ) -> FleetEventResult:
        """Steward review of a held promotion (``accepted`` activates a pending memory)."""
        fid = validate_session_id(fleet_id)
        actor = _require_str(actor, "actor")
        decision_value = _validate_review_decision(decision)
        projection = self._project(fid)
        if projection.trust_tier(actor) != "steward":
            raise ValueError(
                f"review_promotion requires a steward actor; '{actor}' is not a steward of fleet '{fid}'"
            )
        memory = projection.memory(promotion_id)
        if memory is None:
            raise ValueError(f"Unknown promotion_id for fleet {fid}: {promotion_id}")
        event = self._append(
            build_fleet_promotion_reviewed_event(
                actor=actor,
                fleet_id=fid,
                promotion_id=promotion_id,
                decision=decision_value,
                rationale=rationale,
                source_events=[{"seq": memory.event_seq, "hash": memory.event_hash}],
            )
        )
        if decision_value == "accepted" and memory.conflict_key is not None:
            self._supersede_conflicts(
                fid, promotion_id, memory.conflict_key, memory.conflict_value, actor
            )
        return FleetEventResult(
            event=event, fleet_id=fid, actor=actor, promotion_id=promotion_id, summary=rationale
        )

    def rollback_promotion(
        self, fleet_id: str, promotion_id: str, *, reason: str, actor: str
    ) -> FleetEventResult:
        """Reversibly un-share a promotion: lowers effective scope additively.

        Authorized for a steward or the promotion's own actor (self-retraction).
        Anyone else un-sharing another agent's promotion would be an
        unauthorized governance action, so it is rejected.
        """
        fid = validate_session_id(fleet_id)
        actor = _require_str(actor, "actor")
        reason = _require_str(reason, "reason")
        projection = self._project(fid)
        memory = projection.memory(promotion_id)
        if memory is None:
            raise ValueError(f"Unknown promotion_id for fleet {fid}: {promotion_id}")
        if actor != memory.actor and projection.trust_tier(actor) != "steward":
            raise ValueError(
                "rollback_promotion requires a steward actor or the promotion's original "
                f"actor; '{actor}' is neither for promotion '{promotion_id}' in fleet '{fid}'"
            )
        event = self._append(
            build_fleet_promotion_rolled_back_event(
                actor=actor,
                fleet_id=fid,
                promotion_id=promotion_id,
                reason=reason,
                within_rollback_window=self._within_rollback_window(memory),
                source_events=[{"seq": memory.event_seq, "hash": memory.event_hash}],
                gate_event=memory.gate_event,
            )
        )
        return FleetEventResult(
            event=event, fleet_id=fid, actor=actor, promotion_id=promotion_id, summary=reason
        )

    def _within_rollback_window(self, memory: FleetMemoryState) -> bool:
        policy = resolve_evolution_policy(self._settings)
        try:
            stamped = datetime.fromisoformat(memory.timestamp.replace("Z", "+00:00"))
        except ValueError:
            return False
        if stamped.tzinfo is None:
            stamped = stamped.replace(tzinfo=UTC)
        return (datetime.now(UTC) - stamped).total_seconds() <= policy.rollback_window_seconds

    # -- replay projections ---------------------------------------------

    def fleet_brief(self, fleet_id: str) -> FleetBrief:
        """Active promotions, enrolled agents, and trust tiers for a fleet."""
        fid = validate_session_id(fleet_id)
        projection = self._project(fid)
        pending = [memory for memory in projection.memories if memory.review_status == "pending"]
        return FleetBrief(
            fleet_id=fid,
            summary=projection.summary,
            agents=projection.agents,
            active_promotions=projection.active_memories(),
            pending_promotions=pending,
        )

    def resolve_fleet_skills(self, fleet_id: str) -> list[FleetMemoryState]:
        """Return the fleet's active promoted skills (replay-derived)."""
        fid = validate_session_id(fleet_id)
        events = self.session_manager.replay(fleet_thread(fid)).events
        return resolve_fleet_skills(events, fleet_id=fid)

    def fleet_audit(self, fleet_id: str) -> FleetAuditReport:
        """Full provenance for every fleet memory: origin actor/session, source + gate citations."""
        fid = validate_session_id(fleet_id)
        projection = self._project(fid)
        return FleetAuditReport(
            fleet_id=fid,
            records=[_audit_record(memory) for memory in projection.memories],
        )
