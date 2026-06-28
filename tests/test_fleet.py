"""Tests for the fleet propagation contracts: builders, validators, projections."""

from __future__ import annotations

import pytest

from zaxy.fleet import (
    DEFAULT_TRUST_TIER,
    DEFAULT_VISIBILITY_SCOPE,
    TRUST_TIERS,
    VISIBILITY_SCOPES,
    build_fleet_agent_enrolled_event,
    build_fleet_created_event,
    build_fleet_memory_superseded_event,
    build_fleet_outcome_propagated_event,
    build_fleet_promotion_reviewed_event,
    build_fleet_promotion_rolled_back_event,
    build_fleet_rule_propagated_event,
    build_fleet_skill_promoted_event,
    build_fleet_trust_assigned_event,
    fleet_thread,
    max_proposable_scope,
    promotion_id,
    resolve_fleet_skills,
    scope_permits_proposal,
    summarize_fleet_events,
    validate_trust_tier,
    validate_visibility_scope,
)


def _hash(n: int) -> str:
    return f"{n:064x}"


def _ref(seq: int) -> dict[str, object]:
    return {"seq": seq, "hash": _hash(seq)}


def _event(spec: dict[str, object], seq: int, *, ts: str = "2026-06-28T00:00:00Z") -> dict[str, object]:
    return {
        "type": spec["event_type"],
        "actor": spec["actor"],
        "payload": spec["payload"],
        "thread": spec["thread"],
        "seq": seq,
        "hash": _hash(seq),
        "timestamp": ts,
    }


# ---------------------------------------------------------------------------
# Constants & validators
# ---------------------------------------------------------------------------


def test_default_vocabulary() -> None:
    assert VISIBILITY_SCOPES == ("private", "session", "mission", "fleet", "global")
    assert DEFAULT_VISIBILITY_SCOPE == "session"
    assert TRUST_TIERS == ("untrusted", "member", "trusted", "steward")
    assert DEFAULT_TRUST_TIER == "member"


def test_validate_visibility_scope_round_trip_and_rejects_unknown() -> None:
    assert validate_visibility_scope("fleet") == "fleet"
    with pytest.raises(ValueError):
        validate_visibility_scope("organization")


def test_validate_trust_tier_round_trip_and_rejects_unknown() -> None:
    assert validate_trust_tier("steward") == "steward"
    with pytest.raises(ValueError):
        validate_trust_tier("admin")


def test_max_proposable_scope_ladder() -> None:
    assert max_proposable_scope("untrusted") == "session"
    assert max_proposable_scope("member") == "fleet"
    assert max_proposable_scope("trusted") == "global"
    assert max_proposable_scope("steward") == "global"


def test_scope_permits_proposal_enforces_ceiling() -> None:
    # member can reach fleet but never global; untrusted reaches neither.
    assert scope_permits_proposal("member", "fleet") is True
    assert scope_permits_proposal("member", "global") is False
    assert scope_permits_proposal("untrusted", "fleet") is False
    assert scope_permits_proposal("trusted", "global") is True
    assert scope_permits_proposal("steward", "global") is True


def test_fleet_thread_uses_dotted_namespace_and_rejects_colon() -> None:
    assert fleet_thread("alpha") == "fleet.alpha"
    with pytest.raises(ValueError):
        fleet_thread("bad:id")  # ':' is illegal in session ids


# ---------------------------------------------------------------------------
# promotion_id determinism
# ---------------------------------------------------------------------------


def test_promotion_id_is_deterministic_and_idempotent() -> None:
    first = promotion_id(fleet_id="alpha", kind="skill", origin_session="agent-a", source_events=[_ref(3)])
    second = promotion_id(fleet_id="alpha", kind="skill", origin_session="agent-a", source_events=[_ref(3)])
    assert first == second
    assert first.startswith("fleetpromo:")
    assert len(first.split(":", 1)[1]) == 24


def test_promotion_id_varies_with_inputs() -> None:
    base = promotion_id(fleet_id="alpha", kind="skill", origin_session="agent-a", source_events=[_ref(3)])
    other_source = promotion_id(fleet_id="alpha", kind="skill", origin_session="agent-a", source_events=[_ref(4)])
    other_kind = promotion_id(fleet_id="alpha", kind="outcome", origin_session="agent-a", source_events=[_ref(3)])
    other_fleet = promotion_id(fleet_id="beta", kind="skill", origin_session="agent-a", source_events=[_ref(3)])
    assert len({base, other_source, other_kind, other_fleet}) == 4


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def test_created_and_enrolled_builders_are_non_authoritative() -> None:
    created = build_fleet_created_event(actor="coordinator", fleet_id="alpha", summary="Alpha fleet")
    assert created["event_type"] == "fleet.created"
    assert created["thread"] == "fleet.alpha"
    assert created["payload"]["authority_status"] == "non_authoritative"
    assert created["payload"]["fleet_id"] == "alpha"

    enrolled = build_fleet_agent_enrolled_event(actor="coordinator", fleet_id="alpha", agent_id="agent-a")
    assert enrolled["payload"]["trust_tier"] == DEFAULT_TRUST_TIER
    assert enrolled["payload"]["authority_status"] == "non_authoritative"


def test_trust_assigned_builder_records_prior_tier() -> None:
    event = build_fleet_trust_assigned_event(
        actor="steward",
        fleet_id="alpha",
        agent_id="agent-a",
        trust_tier="trusted",
        prior_tier="member",
        rationale="proven contributor",
    )
    assert event["event_type"] == "fleet.trust.assigned"
    assert event["payload"]["trust_tier"] == "trusted"
    assert event["payload"]["prior_tier"] == "member"
    assert event["payload"]["rationale"] == "proven contributor"


def test_skill_promotion_builder_cites_source_and_gate() -> None:
    event = build_fleet_skill_promoted_event(
        actor="agent-b",
        fleet_id="alpha",
        skill_id="deploy",
        skill_version="1.2",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_ref(3)],
        gate_event=_ref(9),
        confidence=0.9,
        auto_applied=True,
    )
    payload = event["payload"]
    assert event["event_type"] == "fleet.skill.promoted"
    assert payload["authority_status"] == "non_authoritative"
    assert payload["source_events"] == [_ref(3)]
    assert payload["gate_event"] == _ref(9)
    assert payload["review_status"] == "active"
    assert payload["keystone"] is False  # carried, default off (deferred enforcement)
    assert payload["visibility_scope"] == "fleet"
    assert payload["promotion_id"].startswith("fleetpromo:")


def test_outcome_propagation_builder_holds_pending_when_not_auto_applied() -> None:
    event = build_fleet_outcome_propagated_event(
        actor="agent-b",
        fleet_id="alpha",
        outcome="failure",
        summary="JWKS cache expired",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_ref(3)],
        gate_event=_ref(9),
        confidence=0.7,
        auto_applied=False,
        claim_key="auth.jwks",
    )
    payload = event["payload"]
    assert event["event_type"] == "fleet.outcome.propagated"
    assert payload["review_status"] == "pending"
    assert payload["outcome"] == "failure"
    assert payload["claim_key"] == "auth.jwks"
    assert payload["gate_event"] == _ref(9)


def test_rule_propagation_builder_derives_rule_id() -> None:
    event = build_fleet_rule_propagated_event(
        actor="agent-b",
        fleet_id="alpha",
        rule="Pre-warm caches before first use",
        trigger="cold start",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_ref(3)],
        gate_event=_ref(9),
        confidence=0.9,
        auto_applied=True,
    )
    payload = event["payload"]
    assert event["event_type"] == "fleet.rule.propagated"
    assert payload["rule_id"].startswith("rule:")
    assert payload["trigger"] == "cold start"
    assert payload["keystone"] is False


def test_builders_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        build_fleet_created_event(actor="", fleet_id="alpha", summary="x")
    with pytest.raises(ValueError):
        build_fleet_agent_enrolled_event(actor="c", fleet_id="alpha", agent_id="a", trust_tier="root")
    with pytest.raises(ValueError):
        # empty source_events must raise (everything cites).
        build_fleet_skill_promoted_event(
            actor="b", fleet_id="alpha", skill_id="s", skill_version="1",
            origin_session="a", origin_actor="a", source_events=[], gate_event=_ref(9),
            confidence=0.9, auto_applied=True,
        )
    with pytest.raises(ValueError):
        # bad gate ref (short hash) must raise.
        build_fleet_skill_promoted_event(
            actor="b", fleet_id="alpha", skill_id="s", skill_version="1",
            origin_session="a", origin_actor="a", source_events=[_ref(3)],
            gate_event={"seq": 9, "hash": "abc"}, confidence=0.9, auto_applied=True,
        )
    with pytest.raises(ValueError):
        # outcome label out of domain.
        build_fleet_outcome_propagated_event(
            actor="b", fleet_id="alpha", outcome="maybe", summary="x",
            origin_session="a", origin_actor="a", source_events=[_ref(3)],
            gate_event=_ref(9), confidence=0.9, auto_applied=True,
        )
    with pytest.raises(ValueError):
        # confidence outside the unit interval.
        build_fleet_rule_propagated_event(
            actor="b", fleet_id="alpha", rule="r", trigger="t",
            origin_session="a", origin_actor="a", source_events=[_ref(3)],
            gate_event=_ref(9), confidence=1.5, auto_applied=True,
        )


def test_superseded_builder_requires_claim_key_or_skill_id() -> None:
    with pytest.raises(ValueError):
        build_fleet_memory_superseded_event(
            actor="steward", fleet_id="alpha",
            superseded_promotion_id="fleetpromo:a", superseding_promotion_id="fleetpromo:b",
            reason="newer", source_events=[_ref(3)],
        )
    ok = build_fleet_memory_superseded_event(
        actor="steward", fleet_id="alpha",
        superseded_promotion_id="fleetpromo:a", superseding_promotion_id="fleetpromo:b",
        reason="newer", source_events=[_ref(3)], skill_id="deploy",
    )
    assert ok["payload"]["skill_id"] == "deploy"
    assert ok["payload"]["authority_status"] == "non_authoritative"


# ---------------------------------------------------------------------------
# Replay projections (pure)
# ---------------------------------------------------------------------------


def _skill_event(version: str, *, seq: int, source_seq: int, auto: bool = True) -> dict[str, object]:
    spec = build_fleet_skill_promoted_event(
        actor="agent-a",
        fleet_id="alpha",
        skill_id="deploy",
        skill_version=version,
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_ref(source_seq)],
        gate_event=_ref(seq - 1),
        confidence=0.9,
        auto_applied=auto,
    )
    return _event(spec, seq)


def test_summarize_tracks_agents_and_trust_tiers() -> None:
    events = [
        _event(build_fleet_created_event(actor="c", fleet_id="alpha", summary="Alpha"), 1),
        _event(build_fleet_agent_enrolled_event(actor="c", fleet_id="alpha", agent_id="a", trust_tier="member"), 2),
        _event(
            build_fleet_trust_assigned_event(
                actor="c", fleet_id="alpha", agent_id="a", trust_tier="steward", prior_tier="member"
            ),
            3,
        ),
        _event(build_fleet_agent_enrolled_event(actor="c", fleet_id="alpha", agent_id="b", trust_tier="member"), 4),
    ]
    projection = summarize_fleet_events(events)
    assert projection.summary == "Alpha"
    assert projection.trust_tier("a") == "steward"  # latest assignment wins
    assert projection.trust_tier("b") == "member"
    assert projection.trust_tier("ghost") is None


def test_summarize_marks_superseded_prior_and_keeps_both() -> None:
    skill_v1 = _skill_event("1.0", seq=3, source_seq=1)
    skill_v2 = _skill_event("2.0", seq=5, source_seq=2)
    pid1 = skill_v1["payload"]["promotion_id"]
    pid2 = skill_v2["payload"]["promotion_id"]
    superseded = _event(
        build_fleet_memory_superseded_event(
            actor="steward", fleet_id="alpha",
            superseded_promotion_id=pid1, superseding_promotion_id=pid2,
            reason="newer version", source_events=[_ref(5), _ref(3)], skill_id="deploy",
        ),
        6,
    )
    projection = summarize_fleet_events([skill_v1, skill_v2, superseded])
    statuses = {memory.promotion_id: memory.review_status for memory in projection.memories}
    assert statuses[pid1] == "superseded"
    assert statuses[pid2] == "active"
    # both retained (additive, never deleted)
    assert len(projection.memories) == 2
    assert [m.promotion_id for m in resolve_fleet_skills([skill_v1, skill_v2, superseded])] == [pid2]


def test_rollback_of_winner_reactivates_superseded_prior() -> None:
    skill_v1 = _skill_event("1.0", seq=3, source_seq=1)
    skill_v2 = _skill_event("2.0", seq=5, source_seq=2)
    pid1 = skill_v1["payload"]["promotion_id"]
    pid2 = skill_v2["payload"]["promotion_id"]
    superseded = _event(
        build_fleet_memory_superseded_event(
            actor="steward", fleet_id="alpha",
            superseded_promotion_id=pid1, superseding_promotion_id=pid2,
            reason="newer", source_events=[_ref(5), _ref(3)], skill_id="deploy",
        ),
        6,
    )
    rolled = _event(
        build_fleet_promotion_rolled_back_event(
            actor="steward", fleet_id="alpha", promotion_id=pid2, reason="regression",
            within_rollback_window=True, source_events=[_ref(5)],
        ),
        7,
    )
    projection = summarize_fleet_events([skill_v1, skill_v2, superseded, rolled])
    statuses = {memory.promotion_id: memory.review_status for memory in projection.memories}
    assert statuses[pid2] == "rolled_back"
    assert statuses[pid1] == "active"  # winner rolled back -> prior re-activates


def test_review_decision_activates_pending_and_rejects() -> None:
    pending = _skill_event("1.0", seq=3, source_seq=1, auto=False)
    pid = pending["payload"]["promotion_id"]
    assert pending["payload"]["review_status"] == "pending"
    accepted = _event(
        build_fleet_promotion_reviewed_event(
            actor="steward", fleet_id="alpha", promotion_id=pid, decision="accepted",
            source_events=[_ref(3)],
        ),
        4,
    )
    projection = summarize_fleet_events([pending, accepted])
    assert projection.memory(pid).review_status == "active"

    rejected = _event(
        build_fleet_promotion_reviewed_event(
            actor="steward", fleet_id="alpha", promotion_id=pid, decision="rejected",
            source_events=[_ref(3)],
        ),
        5,
    )
    projection = summarize_fleet_events([pending, rejected])
    assert projection.memory(pid).review_status == "rejected"
    assert projection.active_memories() == []


def test_duplicate_promotion_event_never_downgrades_active() -> None:
    # A re-cited source (same promotion_id) must never demote an already-active
    # memory: status transitions only through review / supersede / rollback.
    active = _skill_event("1.0", seq=3, source_seq=1, auto=True)  # review_status=active
    duplicate = _skill_event("1.0", seq=5, source_seq=1, auto=False)  # same pid, review_status=pending
    pid = active["payload"]["promotion_id"]
    assert duplicate["payload"]["promotion_id"] == pid
    projection = summarize_fleet_events([active, duplicate])
    assert projection.memory(pid).review_status == "active"  # not demoted to pending
    assert len(projection.memories) == 1  # one canonical memory, not a duplicate


def test_duplicate_promotion_cannot_revert_steward_acceptance() -> None:
    # A steward acceptance (pending -> active) cannot be reverted by a later
    # plain promotion event for the same promotion_id.
    pending = _skill_event("1.0", seq=3, source_seq=1, auto=False)  # pending
    pid = pending["payload"]["promotion_id"]
    accepted = _event(
        build_fleet_promotion_reviewed_event(
            actor="steward", fleet_id="alpha", promotion_id=pid, decision="accepted",
            source_events=[_ref(3)],
        ),
        4,
    )
    duplicate = _skill_event("1.0", seq=5, source_seq=1, auto=False)  # same pid, pending
    projection = summarize_fleet_events([pending, accepted, duplicate])
    assert projection.memory(pid).review_status == "active"  # acceptance preserved
