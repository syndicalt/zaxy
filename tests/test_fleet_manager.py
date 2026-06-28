"""Tests for FleetManager: gated propagation, trust, review, rollback, supersession."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.config import Settings
from zaxy.fleet import FleetManager, fleet_thread
from zaxy.session import SessionManager


def _manager(tmp_path: Path, *, settings: Settings | None = None) -> FleetManager:
    return FleetManager(eventloom_path=tmp_path / ".eventloom", settings=settings)


def _seed_source(tmp_path: Path, session_id: str = "agent-a") -> dict[str, object]:
    """Append a real source memory event and return its {seq, hash} citation."""
    manager = SessionManager(base_path=str(tmp_path / ".eventloom"))
    event = manager.get(session_id).eventlog.append(
        "memory.outcome.recorded",
        actor=session_id,
        payload={"outcome": "failure", "summary": "expired JWKS cache breaks token refresh"},
        thread=session_id,
    )
    return {"seq": event.seq, "hash": event.hash}


def _fleet_events(tmp_path: Path, fleet_id: str) -> list:
    return SessionManager(base_path=str(tmp_path / ".eventloom")).replay(fleet_thread(fleet_id)).events


# ---------------------------------------------------------------------------
# Bootstrap & governance
# ---------------------------------------------------------------------------


def test_creator_is_implicit_steward_and_enrollment_never_escalates(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha fleet", actor="founder")

    # the creator is the fleet's implicit, initial steward (a steward always exists)
    tiers = {agent.agent_id: agent.trust_tier for agent in manager.fleet_brief("alpha").agents}
    assert tiers == {"founder": "steward"}

    # enrollment records the REQUESTED tier with NO auto-escalation
    first = manager.enroll_agent("alpha", "agent-a", actor="founder")
    assert first.bootstrap_steward is False
    assert first.bootstrap_event is None
    assert first.trust_tier == "member"  # default tier; never escalated to steward

    second = manager.enroll_agent("alpha", "agent-b", actor="founder", trust_tier="trusted")
    assert second.bootstrap_steward is False
    assert second.trust_tier == "trusted"

    tiers = {agent.agent_id: agent.trust_tier for agent in manager.fleet_brief("alpha").agents}
    assert tiers == {"founder": "steward", "agent-a": "member", "agent-b": "trusted"}


def test_enroll_with_explicit_steward_skips_bootstrap(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    result = manager.enroll_agent("alpha", "lead", actor="coordinator", trust_tier="steward")
    assert result.bootstrap_steward is False
    assert result.trust_tier == "steward"


def test_assign_trust_requires_steward_actor(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "steward-agent", actor="coordinator", trust_tier="steward")  # steward
    manager.enroll_agent("alpha", "worker", actor="steward-agent")  # member

    promoted = manager.assign_trust(
        "alpha", "worker", trust_tier="trusted", actor="steward-agent", rationale="proven"
    )
    assert promoted.trust_tier == "trusted"
    assert promoted.event.payload["prior_tier"] == "member"

    with pytest.raises(ValueError, match="steward"):
        manager.assign_trust("alpha", "worker", trust_tier="steward", actor="worker")


# ---------------------------------------------------------------------------
# Gated propagation
# ---------------------------------------------------------------------------


def test_gated_propagation_appends_gate_then_cites_it(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "agent-a", actor="coordinator")  # member (may propose to fleet)
    source = _seed_source(tmp_path)

    result = manager.propagate_outcome(
        "alpha",
        outcome="failure",
        summary="Pre-warm the JWKS cache before first token refresh",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[source],
        confidence=0.92,
        actor="agent-a",
        claim_key="auth.jwks.cache",
    )

    assert result.rejected is False
    assert result.auto_applied is True
    assert result.review_status == "active"

    # The gate event precedes and is cited by the fleet.* event.
    assert result.gate_event.type == "evolution.gate.evaluated"
    assert result.promotion_event.type == "fleet.outcome.propagated"
    cited_gate = result.promotion_event.payload["gate_event"]
    assert cited_gate == {"seq": result.gate_event.seq, "hash": result.gate_event.hash}
    assert result.gate_event.seq < result.promotion_event.seq

    # The crossing cites its source events and is non-authoritative.
    assert result.promotion_event.payload["source_events"] == [source]
    assert result.promotion_event.payload["authority_status"] == "non_authoritative"

    # No fleet.* event without a preceding gate event.
    events = _fleet_events(tmp_path, "alpha")
    fleet_promotions = [e for e in events if e.type == "fleet.outcome.propagated"]
    gates = [e for e in events if e.type == "evolution.gate.evaluated"]
    assert len(fleet_promotions) == 1
    assert len(gates) == 1


def test_every_fleet_event_is_non_authoritative_and_cites(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "agent-a", actor="coordinator")  # member (may propose to fleet)
    source = _seed_source(tmp_path)
    result = manager.promote_skill(
        "alpha",
        skill_id="deploy",
        skill_version="1.0",
        origin_session="agent-a",
        source_events=[source],
        confidence=0.9,
        actor="agent-a",
    )
    manager.rollback_promotion("alpha", result.promotion_id, reason="just checking", actor="agent-a")

    for event in _fleet_events(tmp_path, "alpha"):
        if event.type.startswith("fleet."):
            assert event.payload.get("authority_status") == "non_authoritative", event.type
        if event.type == "fleet.skill.promoted":
            assert event.payload["source_events"]
            assert event.payload["gate_event"]


def test_trust_tier_rejection_emits_nothing_on_fleet_thread(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "steward-agent", actor="coordinator", trust_tier="steward")  # steward
    manager.enroll_agent("alpha", "sandboxed", actor="steward-agent")
    manager.assign_trust("alpha", "sandboxed", trust_tier="untrusted", actor="steward-agent")
    source = _seed_source(tmp_path)

    before = len(_fleet_events(tmp_path, "alpha"))
    result = manager.propagate_outcome(
        "alpha",
        outcome="success",
        summary="should never cross",
        origin_session="sandboxed",
        origin_actor="sandboxed",
        source_events=[source],
        confidence=0.99,
        actor="sandboxed",
        claim_key="x.y",
    )
    after = len(_fleet_events(tmp_path, "alpha"))

    assert result.rejected is True
    assert result.promotion_id is None
    assert "insufficient trust" in (result.reason or "")
    assert before == after  # the crossing never happened: no event, not even a gate


def test_unenrolled_actor_cannot_propose(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "agent-a", actor="coordinator")
    source = _seed_source(tmp_path)
    result = manager.propagate_outcome(
        "alpha",
        outcome="failure",
        summary="from a stranger",
        origin_session="stranger",
        origin_actor="stranger",
        source_events=[source],
        confidence=0.95,
        actor="stranger",
        claim_key="a.b",
    )
    assert result.rejected is True
    assert "not enrolled" in (result.reason or "")


# ---------------------------------------------------------------------------
# Autonomy: auto-apply vs require-review
# ---------------------------------------------------------------------------


def test_require_review_override_holds_pending_then_steward_accepts(tmp_path: Path) -> None:
    settings = Settings(evolution_op_autonomy="promote=require_review")
    manager = _manager(tmp_path, settings=settings)
    manager.create_fleet("beta", summary="Beta", actor="coordinator")
    manager.enroll_agent("beta", "steward-agent", actor="coordinator", trust_tier="steward")  # steward
    manager.enroll_agent("beta", "member-agent", actor="steward-agent")  # member
    source = _seed_source(tmp_path, "member-agent")

    held = manager.propagate_rule(
        "beta",
        rule="Pre-warm caches before first request",
        trigger="cold start",
        origin_session="member-agent",
        origin_actor="member-agent",
        source_events=[source],
        confidence=0.99,  # high confidence, but override forces review
        actor="member-agent",
    )
    assert held.rejected is False
    assert held.auto_applied is False
    assert held.review_status == "pending"
    # pending memory excluded from active projection.
    assert held.promotion_id not in {m.promotion_id for m in manager.fleet_brief("beta").active_promotions}

    accepted = manager.review_promotion(
        "beta", held.promotion_id, decision="accepted", actor="steward-agent", rationale="reviewed"
    )
    assert accepted.event.type == "fleet.promotion.reviewed"
    active_ids = {m.promotion_id for m in manager.fleet_brief("beta").active_promotions}
    assert held.promotion_id in active_ids


def test_low_confidence_holds_for_review_under_default_tier(tmp_path: Path) -> None:
    manager = _manager(tmp_path)  # default auto_with_rollback, threshold 0.85
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "agent-a", actor="coordinator")
    source = _seed_source(tmp_path)
    result = manager.propagate_outcome(
        "alpha",
        outcome="partial",
        summary="weak signal",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[source],
        confidence=0.5,
        actor="agent-a",
        claim_key="weak.claim",
    )
    assert result.review_status == "pending"
    assert result.auto_applied is False


# ---------------------------------------------------------------------------
# Conflict / supersession / rollback
# ---------------------------------------------------------------------------


def test_additive_supersession_retains_both(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("gamma", summary="Gamma", actor="coordinator")
    manager.enroll_agent("gamma", "agent-a", actor="coordinator")  # member (may propose to fleet)
    s1 = _seed_source(tmp_path, "agent-a")
    s2 = _seed_source(tmp_path, "agent-a")

    first = manager.promote_skill(
        "gamma", skill_id="build", skill_version="1.0", origin_session="agent-a",
        source_events=[s1], confidence=0.9, actor="agent-a",
    )
    second = manager.promote_skill(
        "gamma", skill_id="build", skill_version="2.0", origin_session="agent-a",
        source_events=[s2], confidence=0.9, actor="agent-a",
    )

    # the supersession is emitted, citing both promotions, and retains both.
    assert len(second.supersessions) == 1
    supersede_event = second.supersessions[0]
    assert supersede_event.type == "fleet.memory.superseded"
    assert supersede_event.payload["superseded_promotion_id"] == first.promotion_id
    assert supersede_event.payload["superseding_promotion_id"] == second.promotion_id

    projection_statuses = {
        m.promotion_id: m.review_status for m in manager.fleet_audit("gamma").records
    }
    assert projection_statuses[first.promotion_id] == "superseded"
    assert projection_statuses[second.promotion_id] == "active"
    # nothing deleted: the prior promotion event still in the log.
    promoted = [e for e in _fleet_events(tmp_path, "gamma") if e.type == "fleet.skill.promoted"]
    assert len(promoted) == 2

    active_skills = manager.resolve_fleet_skills("gamma")
    assert [m.promotion_id for m in active_skills] == [second.promotion_id]


def test_rollback_lowers_scope_additively_and_reactivates_prior(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("gamma", summary="Gamma", actor="coordinator")
    manager.enroll_agent("gamma", "agent-a", actor="coordinator")  # member (may propose to fleet)
    s1 = _seed_source(tmp_path, "agent-a")
    s2 = _seed_source(tmp_path, "agent-a")
    first = manager.promote_skill(
        "gamma", skill_id="build", skill_version="1.0", origin_session="agent-a",
        source_events=[s1], confidence=0.9, actor="agent-a",
    )
    second = manager.promote_skill(
        "gamma", skill_id="build", skill_version="2.0", origin_session="agent-a",
        source_events=[s2], confidence=0.9, actor="agent-a",
    )

    rollback = manager.rollback_promotion("gamma", second.promotion_id, reason="regression", actor="agent-a")
    assert rollback.event.type == "fleet.promotion.rolled_back"
    assert rollback.event.payload["within_rollback_window"] is True

    statuses = {m.promotion_id: m.review_status for m in manager.fleet_audit("gamma").records}
    assert statuses[second.promotion_id] == "rolled_back"
    assert statuses[first.promotion_id] == "active"  # rolling back the winner re-activates the prior

    active_ids = {m.promotion_id for m in manager.fleet_brief("gamma").active_promotions}
    assert second.promotion_id not in active_ids
    assert first.promotion_id in active_ids
    # the rolled-back promotion event is retained (reversible, never deleted).
    assert any(
        e.type == "fleet.skill.promoted" and e.payload["promotion_id"] == second.promotion_id
        for e in _fleet_events(tmp_path, "gamma")
    )


def test_idempotent_promotion_id(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "agent-a", actor="coordinator")
    source = _seed_source(tmp_path)
    first = manager.promote_skill(
        "alpha", skill_id="deploy", skill_version="1.0", origin_session="agent-a",
        source_events=[source], confidence=0.9, actor="agent-a",
    )
    second = manager.promote_skill(
        "alpha", skill_id="deploy", skill_version="1.0", origin_session="agent-a",
        source_events=[source], confidence=0.9, actor="agent-a",
    )
    assert first.promotion_id == second.promotion_id


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_fleet_audit_answers_which_agent_taught_this_from_what_evidence(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "agent-a", actor="coordinator")  # member (may propose to fleet)
    manager.enroll_agent("alpha", "agent-b", actor="agent-a")  # member proposer
    source = _seed_source(tmp_path, "agent-a")

    result = manager.propagate_rule(
        "alpha",
        rule="Always validate JWKS freshness before token refresh",
        trigger="token refresh",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[source],
        confidence=0.9,
        actor="agent-b",  # agent-b proposes a rule agent-a learned
    )

    audit = manager.fleet_audit("alpha")
    record = next(r for r in audit.records if r.promotion_id == result.promotion_id)
    assert record.origin_actor == "agent-a"  # who taught the fleet
    assert record.origin_session == "agent-a"
    assert record.actor == "agent-b"  # who proposed the crossing
    assert record.source_events == [source]  # from what evidence
    assert record.gate_event is not None  # through which gate
    assert record.review_status == "active"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_create_enroll_promote_status(tmp_path: Path) -> None:
    runner = CliRunner()
    eventloom = str(tmp_path / ".eventloom")

    created = runner.invoke(app, ["fleet", "create", "alpha", "--summary", "Alpha fleet", "--eventloom-path", eventloom])
    assert created.exit_code == 0, created.output

    enrolled = runner.invoke(
        app, ["fleet", "enroll", "alpha", "--agent", "agent-a", "--eventloom-path", eventloom, "--json"]
    )
    assert enrolled.exit_code == 0, enrolled.output
    assert '"bootstrap_steward": false' in enrolled.output

    source = _seed_source(tmp_path, "agent-a")
    promoted = runner.invoke(
        app,
        [
            "fleet", "promote-outcome", "alpha",
            "--outcome", "failure",
            "--summary", "Pre-warm caches",
            "--origin-session", "agent-a",
            "--source-event", f"{source['seq']}:{source['hash']}",
            "--confidence", "0.92",
            "--actor", "agent-a",
            "--claim-key", "auth.cache",
            "--eventloom-path", eventloom,
            "--json",
        ],
    )
    assert promoted.exit_code == 0, promoted.output
    assert '"review_status": "active"' in promoted.output

    status = runner.invoke(app, ["fleet", "status", "alpha", "--eventloom-path", eventloom])
    assert status.exit_code == 0, status.output
    assert "Active promotions: 1" in status.output


# ---------------------------------------------------------------------------
# Regression: keystone conflicts are held for a steward (never auto-superseded)
# ---------------------------------------------------------------------------


def test_keystone_conflict_is_held_for_steward_not_auto_superseded(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("delta", summary="Delta", actor="coordinator")  # coordinator = steward
    manager.enroll_agent("delta", "agent-a", actor="coordinator", trust_tier="trusted")
    s1 = _seed_source(tmp_path, "agent-a")
    s2 = _seed_source(tmp_path, "agent-a")

    keystone = manager.propagate_rule(
        "delta",
        rule="Always rotate credentials before deploy",
        trigger="deploy",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[s1],
        confidence=0.95,  # auto-applies -> active keystone
        actor="agent-a",
        keystone=True,
    )
    assert keystone.review_status == "active"
    assert keystone.auto_applied is True

    conflicting = manager.propagate_rule(
        "delta",
        rule="Never rotate credentials before deploy",  # same trigger, different rule -> conflict
        trigger="deploy",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[s2],
        confidence=0.95,  # would auto-apply, but the live keystone forces a hold
        actor="agent-a",
    )
    # recorded + gated as usual, but HELD pending for a steward (not active)
    assert conflicting.rejected is False
    assert conflicting.review_status == "pending"
    assert conflicting.auto_applied is False
    assert conflicting.gate_event is not None  # still routed through the I4 gate
    assert conflicting.promotion_event is not None  # still recorded + cited
    assert conflicting.supersessions == []  # the keystone is NOT superseded

    statuses = {m.promotion_id: m.review_status for m in manager.fleet_audit("delta").records}
    assert statuses[keystone.promotion_id] == "active"  # keystone STAYS active
    assert statuses[conflicting.promotion_id] == "pending"

    # no fleet.memory.superseded was emitted against the keystone
    supersessions = [
        e for e in _fleet_events(tmp_path, "delta") if e.type == "fleet.memory.superseded"
    ]
    assert supersessions == []


# ---------------------------------------------------------------------------
# Regression: enrollment never escalates trust (no untrusted -> steward)
# ---------------------------------------------------------------------------


def test_enrollment_never_escalates_untrusted_and_blocks_crossing(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")  # a steward already exists
    enrolled = manager.enroll_agent(
        "alpha", "sandboxed", actor="coordinator", trust_tier="untrusted"
    )
    # the untrusted enrollee stays untrusted (no first-enrollee -> steward bootstrap)
    assert enrolled.bootstrap_steward is False
    assert enrolled.bootstrap_event is None
    assert enrolled.trust_tier == "untrusted"
    tiers = {a.agent_id: a.trust_tier for a in manager.fleet_brief("alpha").agents}
    assert tiers["sandboxed"] == "untrusted"

    source = _seed_source(tmp_path, "sandboxed")
    before = len(_fleet_events(tmp_path, "alpha"))
    result = manager.propagate_outcome(
        "alpha",
        outcome="success",
        summary="should never cross",
        origin_session="sandboxed",
        origin_actor="sandboxed",
        source_events=[source],
        confidence=0.99,
        actor="sandboxed",
        claim_key="x.y",
    )
    after = len(_fleet_events(tmp_path, "alpha"))
    assert result.rejected is True
    assert result.promotion_id is None
    assert "insufficient trust" in (result.reason or "")
    assert before == after  # nothing appended to the fleet thread, not even a gate


# ---------------------------------------------------------------------------
# Regression: status-lifecycle integrity (re-promotion never demotes / reverts)
# ---------------------------------------------------------------------------


def test_repromotion_does_not_demote_active_memory(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")
    manager.enroll_agent("alpha", "agent-a", actor="coordinator")  # member
    source = _seed_source(tmp_path, "agent-a")

    first = manager.propagate_outcome(
        "alpha", outcome="failure", summary="cache fix",
        origin_session="agent-a", origin_actor="agent-a",
        source_events=[source], confidence=0.95, actor="agent-a", claim_key="x.y",
    )
    assert first.review_status == "active"

    # re-cite the SAME source at low confidence -> idempotent, must NOT demote
    repromo = manager.propagate_outcome(
        "alpha", outcome="failure", summary="cache fix",
        origin_session="agent-a", origin_actor="agent-a",
        source_events=[source], confidence=0.10, actor="agent-a", claim_key="x.y",
    )
    assert repromo.promotion_id == first.promotion_id
    assert repromo.rejected is False
    assert repromo.review_status == "active"  # idempotent active, never demoted to pending

    statuses = {m.promotion_id: m.review_status for m in manager.fleet_audit("alpha").records}
    assert statuses[first.promotion_id] == "active"  # still active

    # no demoting duplicate promotion event was appended
    promos = [e for e in _fleet_events(tmp_path, "alpha") if e.type == "fleet.outcome.propagated"]
    assert len(promos) == 1


def test_repromotion_cannot_undo_steward_acceptance(tmp_path: Path) -> None:
    settings = Settings(evolution_op_autonomy="promote=require_review")
    manager = _manager(tmp_path, settings=settings)
    manager.create_fleet("alpha", summary="Alpha", actor="coordinator")  # coordinator = steward
    manager.enroll_agent("alpha", "agent-a", actor="coordinator")  # member
    source = _seed_source(tmp_path, "agent-a")

    held = manager.propagate_outcome(
        "alpha", outcome="failure", summary="cache fix",
        origin_session="agent-a", origin_actor="agent-a",
        source_events=[source], confidence=0.99, actor="agent-a", claim_key="x.y",
    )
    assert held.review_status == "pending"

    accepted = manager.review_promotion(
        "alpha", held.promotion_id, decision="accepted", actor="coordinator", rationale="reviewed"
    )
    assert accepted.event.type == "fleet.promotion.reviewed"
    assert held.promotion_id in {
        m.promotion_id for m in manager.fleet_brief("alpha").active_promotions
    }

    # re-promote the SAME source -> must NOT revert the steward acceptance
    repromo = manager.propagate_outcome(
        "alpha", outcome="failure", summary="cache fix",
        origin_session="agent-a", origin_actor="agent-a",
        source_events=[source], confidence=0.10, actor="agent-a", claim_key="x.y",
    )
    assert repromo.promotion_id == held.promotion_id
    statuses = {m.promotion_id: m.review_status for m in manager.fleet_audit("alpha").records}
    assert statuses[held.promotion_id] == "active"  # steward acceptance preserved

    # only a rollback / reviewed(rejected) leaves active
    manager.rollback_promotion("alpha", held.promotion_id, reason="regression", actor="coordinator")
    statuses = {m.promotion_id: m.review_status for m in manager.fleet_audit("alpha").records}
    assert statuses[held.promotion_id] == "rolled_back"


def test_propose_promotion_dispatcher_and_memory_serialization(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.create_fleet("alpha", summary="Alpha", actor="founder")
    src = _seed_source(tmp_path)
    skill = manager.propose_promotion(
        "alpha", "skill", skill_id="s1", skill_version="v1",
        origin_session="agent-a", source_events=[src], confidence=0.95, actor="founder",
    )
    outcome = manager.propose_promotion(
        "alpha", "outcome", outcome="failure", summary="cache stale",
        origin_session="agent-a", source_events=[src], confidence=0.95, actor="founder",
    )
    rule = manager.propose_promotion(
        "alpha", "rule", rule="refresh JWKS", trigger="token refresh",
        origin_session="agent-a", source_events=[src], confidence=0.95, actor="founder",
    )
    assert skill.kind == "skill" and skill.review_status == "active"
    assert outcome.kind == "outcome"
    assert rule.kind == "rule"
    with pytest.raises(ValueError):
        manager.propose_promotion("alpha", "not-a-kind", source_events=[src])

    # FleetMemoryState.to_dict carries full cited provenance for each active promotion.
    dicts = {m.to_dict()["kind"]: m.to_dict() for m in manager.fleet_brief("alpha").active_promotions}
    assert set(dicts) == {"skill", "outcome", "rule"}
    s = dicts["skill"]
    assert s["review_status"] == "active"
    assert s["visibility_scope"] == "fleet"
    assert s["keystone"] is False
    assert s["origin_actor"] == "founder"
    assert s["origin_session"] == "agent-a"
    assert s["source_events"] == [src]
    assert s["gate_event"]["seq"] > 0
    assert s["promotion_id"] == skill.promotion_id


def test_visibility_scope_and_trust_tier_validators_reject_bad_input() -> None:
    from zaxy.fleet import validate_trust_tier, validate_visibility_scope

    assert validate_visibility_scope("fleet") == "fleet"
    assert validate_trust_tier("steward") == "steward"
    with pytest.raises(ValueError):
        validate_visibility_scope("galaxy")
    with pytest.raises(ValueError):
        validate_trust_tier("god")
