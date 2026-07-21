"""Tests for the I7b fleet surface: extractors, MCP tools, checkout lane, bridge.

Covers the agent-facing surface of the Fleet Memory Plane: deterministic
extractors that project active fleet memory (and exclude pending/superseded), the
governed ``fleet_*`` MCP tools, the enrollment-gated checkout fleet lane (the
decisive non-enrolled-exclusion proof), and the coordination->fleet bridge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.config import Settings
from zaxy.coordination import CoordinationManager
from zaxy.core import MemoryFabric
from zaxy.embedded_graph_store import EmbeddedGraphStore
from zaxy.event import Event
from zaxy.extract import extract
from zaxy.fleet import FleetAgentState, FleetManager, FleetMemoryState, fleet_thread
from zaxy.mcp_server import ZaxyMCPServer
from zaxy.session import SessionManager


def _seed_source(base: Path, sid: str = "agent-a", summary: str = "expired JWKS cache breaks refresh") -> dict[str, Any]:
    """Append a real source memory event and return its {seq, hash} citation."""
    manager = SessionManager(base_path=str(base))
    event = manager.get(sid).eventlog.append(
        "memory.outcome.recorded",
        actor=sid,
        payload={"outcome": "failure", "summary": summary},
        thread=sid,
    )
    return {"seq": event.seq, "hash": event.hash}


def _fleet_events(base: Path, fleet_id: str) -> list[Any]:
    return SessionManager(base_path=str(base)).replay(fleet_thread(fleet_id)).events


_FLEET_PROJECTED_TYPES = (
    "fleet.skill.promoted",
    "fleet.rule.propagated",
    "fleet.outcome.propagated",
    "fleet.promotion.reviewed",
    "fleet.memory.superseded",
    "fleet.promotion.rolled_back",
)


async def _embedded_store(tmp_path: Path) -> EmbeddedGraphStore:
    store = EmbeddedGraphStore(tmp_path / "projections" / "embedded.kuzu")
    await store.connect()
    await store.init_schema()
    return store


async def _project_fleet_thread(base: Path, fleet_id: str, store: EmbeddedGraphStore) -> str:
    """Project every fleet promotion/lifecycle event under the fleet-thread session.

    Mirrors the production projection path (``mcp_server._project_fleet_event``):
    extract + ``upsert_extraction`` under ``fleet_thread(fleet_id)`` so create and
    lifecycle events land in the same graph session.
    """
    sid = fleet_thread(fleet_id)
    for event in _fleet_events(base, fleet_id):
        if event.type in _FLEET_PROJECTED_TYPES:
            await store.upsert_extraction(extract(event), session_id=sid)
    return sid


def _real_server(tmp_path: Path, **kwargs: Any) -> ZaxyMCPServer:
    """Return a server with a real tmp eventloom and mocked graph/tracer."""
    with (
        patch("zaxy.mcp_server.build_projection_store", return_value=AsyncMock()),
        patch("zaxy.mcp_server.MemoryTracer", return_value=AsyncMock()),
    ):
        return ZaxyMCPServer(
            eventloom_path=str(tmp_path / ".eventloom"),
            workspace_root=tmp_path,
            **kwargs,
        )


def _text(result: list[Any]) -> dict[str, Any]:
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# 1. Extractors
# ---------------------------------------------------------------------------


def test_fleet_extractors_project_active_and_exclude_pending(tmp_path: Path) -> None:
    base = tmp_path / ".eventloom"
    manager = FleetManager(eventloom_path=base)
    manager.create_fleet("alpha", summary="Alpha", actor="founder")
    manager.enroll_agent("alpha", "agent-a", actor="founder")  # member

    active = manager.propagate_outcome(
        "alpha",
        outcome="failure",
        summary="Pre-warm the JWKS cache before first refresh",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_seed_source(base)],
        confidence=0.95,
        actor="agent-a",
        claim_key="auth.jwks.cache",
    )
    assert active.review_status == "active"

    pending = manager.propagate_rule(
        "alpha",
        rule="Add a retry budget",
        trigger="429 storms",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_seed_source(base, summary="retry budget")],
        confidence=0.2,  # below the auto-apply threshold -> held pending
        actor="agent-a",
    )
    assert pending.review_status == "pending"

    projected: dict[str, Any] = {}
    for event in _fleet_events(base, "alpha"):
        if event.type in ("fleet.outcome.propagated", "fleet.rule.propagated"):
            result = extract(event)
            entity = next(e for e in result.entities if e.entity_type.startswith("fleet_") and e.entity_type != "fleet")
            projected[event.type] = entity

    outcome_entity = projected["fleet.outcome.propagated"]
    props = outcome_entity.properties or {}
    assert props["review_status"] == "active"
    assert props["visibility_scope"] == "fleet"
    assert props["fleet_id"] == "alpha"
    assert props["non_authoritative"] is True
    assert props["source_events"]  # cites its source events
    assert props["gate_event"]  # cites the I4 gate

    rule_props = projected["fleet.rule.propagated"].properties or {}
    assert rule_props["review_status"] == "pending"


def test_fleet_supersede_extractor_marks_prior_superseded(tmp_path: Path) -> None:
    base = tmp_path / ".eventloom"
    manager = FleetManager(eventloom_path=base)
    manager.create_fleet("alpha", summary="Alpha", actor="founder")
    manager.enroll_agent("alpha", "agent-a", actor="founder")

    first = manager.promote_skill(
        "alpha",
        skill_id="deploy",
        skill_version="1.0",
        origin_session="agent-a",
        source_events=[_seed_source(base, summary="v1")],
        confidence=0.95,
        actor="agent-a",
    )
    second = manager.promote_skill(
        "alpha",
        skill_id="deploy",
        skill_version="2.0",
        origin_session="agent-a",
        source_events=[_seed_source(base, summary="v2")],
        confidence=0.95,
        actor="agent-a",
    )
    assert second.supersessions  # v2 additively supersedes v1

    superseded: dict[str, str] = {}
    for event in _fleet_events(base, "alpha"):
        if event.type == "fleet.memory.superseded":
            entity = extract(event).entities[0]
            superseded[entity.name] = (entity.properties or {})["review_status"]
    assert superseded.get(first.promotion_id) == "superseded"


async def test_fleet_supersede_merges_review_status_on_same_graph_entity(tmp_path: Path) -> None:
    """Regression: a supersede must flip the ORIGINAL promotion's GRAPH review_status.

    The create extractor and the lifecycle extractor must key the SAME graph
    entity ``(entity_type, name)``; otherwise the supersede lands on a different
    entity and the prior promotion stays ``active`` in graph queries (the no-op
    the review found). This projects through the real extract path into a real
    embedded graph store and asserts the MERGED status on the create entity.
    """
    base = tmp_path / ".eventloom"
    manager = FleetManager(eventloom_path=base)
    manager.create_fleet("alpha", summary="Alpha", actor="founder")
    manager.enroll_agent("alpha", "agent-a", actor="founder")
    first = manager.promote_skill(
        "alpha",
        skill_id="deploy",
        skill_version="1.0",
        origin_session="agent-a",
        source_events=[_seed_source(base, summary="v1")],
        confidence=0.95,
        actor="agent-a",
    )
    second = manager.promote_skill(
        "alpha",
        skill_id="deploy",
        skill_version="2.0",
        origin_session="agent-a",
        source_events=[_seed_source(base, summary="v2")],
        confidence=0.95,
        actor="agent-a",
    )
    assert second.supersessions  # v2 additively supersedes v1

    store = await _embedded_store(tmp_path)
    try:
        sid = await _project_fleet_thread(base, "alpha", store)

        projected = await store.search_exact(first.promotion_id, session_id=sid)
        assert projected, "the prior promotion must be projected into the graph"
        # The bug left the create entity 'active' (the supersede merged onto a
        # different entity_type), so the active graph state would be both
        # {'active', 'superseded'}; the fix unifies onto one entity.
        statuses = {(entity.properties or {}).get("review_status") for entity in projected}
        assert statuses == {"superseded"}, statuses
        assert all(entity.entity_type == "fleet_promotion" for entity in projected)

        # The surviving winner keeps its own promotion entity, still active.
        winner = await store.search_exact(second.promotion_id, session_id=sid)
        assert {(entity.properties or {}).get("review_status") for entity in winner} == {"active"}
    finally:
        await store.close()


async def test_fleet_rollback_merges_review_status_on_same_graph_entity(tmp_path: Path) -> None:
    """A rollback must flip the promotion's GRAPH review_status to ``rolled_back``."""
    base = tmp_path / ".eventloom"
    manager = FleetManager(eventloom_path=base)
    manager.create_fleet("alpha", summary="Alpha", actor="founder")  # founder is steward
    manager.enroll_agent("alpha", "agent-a", actor="founder")
    promoted = manager.propagate_outcome(
        "alpha",
        outcome="failure",
        summary="Pre-warm the JWKS cache",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_seed_source(base)],
        confidence=0.95,
        actor="agent-a",
        claim_key="auth.jwks.cache",
    )
    assert promoted.review_status == "active"
    manager.rollback_promotion("alpha", promoted.promotion_id, reason="regression", actor="founder")

    store = await _embedded_store(tmp_path)
    try:
        sid = await _project_fleet_thread(base, "alpha", store)
        projected = await store.search_exact(promoted.promotion_id, session_id=sid)
        assert projected
        statuses = {(entity.properties or {}).get("review_status") for entity in projected}
        assert statuses == {"rolled_back"}, statuses
        assert all(entity.entity_type == "fleet_promotion" for entity in projected)
    finally:
        await store.close()


def _extract_one(event: Any) -> Any:
    """Return the projected fleet promotion entity for a fleet event."""
    result = extract(event)
    return next(e for e in result.entities if e.entity_type == "fleet_promotion")


def test_fleet_create_extractors_cover_all_three_kinds(tmp_path: Path) -> None:
    """skill / rule / outcome promotions all project a unified ``fleet_promotion`` entity."""
    base = tmp_path / ".eventloom"
    manager = FleetManager(eventloom_path=base)
    manager.create_fleet("alpha", summary="Alpha", actor="founder")
    manager.enroll_agent("alpha", "agent-a", actor="founder")
    manager.promote_skill(
        "alpha", skill_id="deploy", skill_version="1.0", origin_session="agent-a",
        source_events=[_seed_source(base, summary="skill")], confidence=0.95, actor="agent-a",
    )
    manager.propagate_rule(
        "alpha", rule="Pre-warm caches", trigger="cold start", origin_session="agent-a",
        origin_actor="agent-a", source_events=[_seed_source(base, summary="rule")],
        confidence=0.95, actor="agent-a",
    )
    manager.propagate_outcome(
        "alpha", outcome="failure", summary="cold start hurts", origin_session="agent-a",
        origin_actor="agent-a", source_events=[_seed_source(base, summary="outcome")],
        confidence=0.95, actor="agent-a", claim_key="cold.start",
    )

    by_kind: dict[str, Any] = {}
    for event in _fleet_events(base, "alpha"):
        if event.type in ("fleet.skill.promoted", "fleet.rule.propagated", "fleet.outcome.propagated"):
            entity = _extract_one(event)
            by_kind[(entity.properties or {})["kind"]] = entity
    assert set(by_kind) == {"skill", "rule", "outcome"}
    for entity in by_kind.values():
        props = entity.properties or {}
        assert entity.entity_type == "fleet_promotion"  # unified key for create + lifecycle
        assert props["review_status"] == "active"
        assert props["non_authoritative"] is True
        assert props["source_events"]  # cited
    assert by_kind["skill"].properties["skill_id"] == "deploy"
    assert by_kind["rule"].properties["rule"] == "Pre-warm caches"
    assert by_kind["outcome"].properties["outcome"] == "failure"


def test_fleet_review_extractor_maps_each_decision(tmp_path: Path) -> None:
    """Steward review decisions project the matching ``review_status`` onto the promotion."""
    base = tmp_path / ".eventloom"
    manager = FleetManager(eventloom_path=base)
    manager.create_fleet("alpha", summary="Alpha", actor="founder")  # founder is steward
    manager.enroll_agent("alpha", "agent-a", actor="founder")

    def _held() -> str:
        promoted = manager.propagate_rule(
            "alpha", rule=f"rule {len(_fleet_events(base, 'alpha'))}", trigger="t",
            origin_session="agent-a", origin_actor="agent-a",
            source_events=[_seed_source(base, summary="held")], confidence=0.2, actor="agent-a",
        )
        assert promoted.review_status == "pending"
        return promoted.promotion_id

    cases = {"accepted": "active", "rejected": "rejected", "deferred": "deferred"}
    expected: dict[str, str] = {}
    for decision, status in cases.items():
        pid = _held()
        result = manager.review_promotion("alpha", pid, decision=decision, actor="founder")
        entity = _extract_one(result.event)
        assert entity.name == pid
        assert (entity.properties or {})["review_status"] == status
        expected[decision] = status
    assert expected == cases


def test_fleet_review_status_update_ignores_missing_promotion_id() -> None:
    """A lifecycle event without a promotion_id projects nothing (defensive no-op)."""
    event = Event(
        seq=1,
        timestamp="2026-06-28T00:00:00Z",
        type="fleet.promotion.rolled_back",
        actor="steward",
        thread="fleet:alpha",
        payload={"fleet_id": "alpha"},  # no promotion_id
        hash="0" * 64,
    )
    result = extract(event)
    assert result.entities == []
    assert result.edges == []


# ---------------------------------------------------------------------------
# 2. MCP tools route through the governed FleetManager
# ---------------------------------------------------------------------------


async def test_fleet_promote_status_audit_route_through_manager(tmp_path: Path) -> None:
    server = _real_server(tmp_path)
    base = tmp_path / ".eventloom"
    await server.handle_fleet_create({"fleet_id": "beta", "summary": "Beta", "actor": "founder"})
    await server.handle_fleet_enroll({"fleet_id": "beta", "agent_id": "member-agent", "actor": "founder"})
    source = _seed_source(base, "member-agent", summary="cold start")

    promoted = _text(
        await server.handle_fleet_promote(
            {
                "fleet_id": "beta",
                "kind": "outcome",
                "outcome": "failure",
                "summary": "Pre-warm caches before first request",
                "origin_session": "member-agent",
                "origin_actor": "member-agent",
                "actor": "member-agent",
                "confidence": 0.95,
                "source_events": [source],
                "claim_key": "cold.start",
            }
        )
    )
    assert promoted["rejected"] is False
    assert promoted["review_status"] == "active"
    assert promoted["promotion_id"]
    # routed through the I4 gate (a gate event was appended and cited)
    assert promoted["gate_event"] is not None
    assert promoted["gate_decision"]["op"] == "promote"

    status = _text(await server.handle_fleet_status({"fleet_id": "beta"}))
    assert promoted["promotion_id"] in {m["promotion_id"] for m in status["active_promotions"]}

    audit = _text(await server.handle_fleet_audit({"fleet_id": "beta"}))
    record = next(r for r in audit["records"] if r["promotion_id"] == promoted["promotion_id"])
    assert record["source_events"] and record["gate_event"]
    assert record["review_status"] == "active"


async def test_fleet_promote_untrusted_rejected_via_tool(tmp_path: Path) -> None:
    server = _real_server(tmp_path)
    base = tmp_path / ".eventloom"
    await server.handle_fleet_create({"fleet_id": "beta", "summary": "Beta", "actor": "founder"})
    await server.handle_fleet_enroll(
        {"fleet_id": "beta", "agent_id": "sandboxed", "trust_tier": "untrusted", "actor": "founder"}
    )
    before = len(_fleet_events(base, "beta"))

    rejected = _text(
        await server.handle_fleet_promote(
            {
                "fleet_id": "beta",
                "kind": "outcome",
                "outcome": "success",
                "summary": "should never cross",
                "origin_session": "sandboxed",
                "origin_actor": "sandboxed",
                "actor": "sandboxed",
                "confidence": 0.99,
                "source_events": [_seed_source(base, "sandboxed")],
                "claim_key": "x.y",
            }
        )
    )
    assert rejected["rejected"] is True
    assert rejected["promotion_id"] is None
    assert "insufficient trust" in (rejected["reason"] or "")
    # the crossing never happened: no event (not even a gate) on the fleet thread
    assert len(_fleet_events(base, "beta")) == before


async def test_fleet_require_review_pending_then_steward_accepts(tmp_path: Path) -> None:
    server = _real_server(tmp_path)
    server._settings = Settings(evolution_op_autonomy="promote=require_review")
    base = tmp_path / ".eventloom"
    await server.handle_fleet_create({"fleet_id": "beta", "summary": "Beta", "actor": "founder"})  # founder is steward
    await server.handle_fleet_enroll({"fleet_id": "beta", "agent_id": "member-agent", "actor": "founder"})

    held = _text(
        await server.handle_fleet_promote(
            {
                "fleet_id": "beta",
                "kind": "rule",
                "rule": "Pre-warm caches before first request",
                "trigger": "cold start",
                "origin_session": "member-agent",
                "origin_actor": "member-agent",
                "actor": "member-agent",
                "confidence": 0.99,  # high, but the override forces review
                "source_events": [_seed_source(base, "member-agent")],
            }
        )
    )
    assert held["review_status"] == "pending"
    pid = held["promotion_id"]

    status = _text(await server.handle_fleet_status({"fleet_id": "beta"}))
    assert pid not in {m["promotion_id"] for m in status["active_promotions"]}
    assert pid in {m["promotion_id"] for m in status["pending_promotions"]}

    await server.handle_fleet_review(
        {"fleet_id": "beta", "promotion_id": pid, "decision": "accepted", "actor": "founder"}
    )
    status_after = _text(await server.handle_fleet_status({"fleet_id": "beta"}))
    assert pid in {m["promotion_id"] for m in status_after["active_promotions"]}


async def test_fleet_assign_trust_via_tool_and_validation(tmp_path: Path) -> None:
    server = _real_server(tmp_path)
    await server.handle_fleet_create({"fleet_id": "beta", "summary": "Beta", "actor": "founder"})
    await server.handle_fleet_enroll({"fleet_id": "beta", "agent_id": "member-agent", "actor": "founder"})

    assigned = _text(
        await server.handle_fleet_assign_trust(
            {
                "fleet_id": "beta",
                "agent_id": "member-agent",
                "trust_tier": "trusted",
                "actor": "founder",
                "rationale": "consistent high-quality findings",
            }
        )
    )
    assert assigned["agent_id"] == "member-agent"
    assert assigned["trust_tier"] == "trusted"

    status = _text(await server.handle_fleet_status({"fleet_id": "beta"}))
    tiers = {a["agent_id"]: a["trust_tier"] for a in status["agents"]}
    assert tiers["member-agent"] == "trusted"

    # arg-validation rejection: trust_tier is required.
    with pytest.raises(ValueError):
        await server.handle_fleet_assign_trust(
            {"fleet_id": "beta", "agent_id": "member-agent", "actor": "founder"}
        )


async def test_fleet_promote_skill_via_tool_supersedes_and_validates(tmp_path: Path) -> None:
    server = _real_server(tmp_path)
    base = tmp_path / ".eventloom"
    await server.handle_fleet_create({"fleet_id": "beta", "summary": "Beta", "actor": "founder"})
    await server.handle_fleet_enroll({"fleet_id": "beta", "agent_id": "member-agent", "actor": "founder"})
    first = _text(
        await server.handle_fleet_promote(
            {
                "fleet_id": "beta",
                "kind": "skill",
                "skill_id": "deploy",
                "skill_version": "1.0",
                "origin_session": "member-agent",
                "origin_actor": "member-agent",
                "actor": "member-agent",
                "confidence": 0.95,
                "source_events": [_seed_source(base, "member-agent", summary="v1")],
            }
        )
    )
    assert first["review_status"] == "active"
    second = _text(
        await server.handle_fleet_promote(
            {
                "fleet_id": "beta",
                "kind": "skill",
                "skill_id": "deploy",
                "skill_version": "2.0",
                "origin_session": "member-agent",
                "origin_actor": "member-agent",
                "actor": "member-agent",
                "confidence": 0.95,
                "source_events": [_seed_source(base, "member-agent", summary="v2")],
            }
        )
    )
    # v2 supersedes v1: the handler projects the supersession event too (graph mocked).
    assert second["supersessions"]
    assert server.graph.upsert_extraction.await_count >= 3  # v1 + v2 + supersede projections

    # arg-validation: unknown kind and non-numeric confidence are rejected.
    with pytest.raises(ValueError):
        await server.handle_fleet_promote(
            {
                "fleet_id": "beta",
                "kind": "policy",
                "origin_session": "member-agent",
                "actor": "member-agent",
                "confidence": 0.9,
                "source_events": [_seed_source(base, "member-agent")],
            }
        )
    with pytest.raises(ValueError):
        await server.handle_fleet_promote(
            {
                "fleet_id": "beta",
                "kind": "outcome",
                "outcome": "failure",
                "summary": "x",
                "origin_session": "member-agent",
                "actor": "member-agent",
                "confidence": "high",  # not a number
                "source_events": [_seed_source(base, "member-agent")],
            }
        )


async def test_fleet_handlers_reject_missing_required_args(tmp_path: Path) -> None:
    server = _real_server(tmp_path)
    with pytest.raises(ValueError):
        await server.handle_fleet_create({"fleet_id": "beta"})  # summary required
    with pytest.raises(ValueError):
        await server.handle_fleet_enroll({"fleet_id": "beta"})  # agent_id required
    with pytest.raises(ValueError):
        await server.handle_fleet_review({"fleet_id": "beta", "promotion_id": "p"})  # decision/actor


async def test_project_fleet_event_swallows_extraction_failure(tmp_path: Path) -> None:
    """A malformed fleet event must not crash projection (best-effort graph write)."""
    server = _real_server(tmp_path)
    bad_event = Event(
        seq=1,
        timestamp="2026-06-28T00:00:00Z",
        type="fleet.skill.promoted",
        actor="member-agent",
        thread="fleet:beta",
        payload={"fleet_id": "beta"},  # missing promotion_id -> extract raises
        hash="0" * 64,
    )
    await server._project_fleet_event(bad_event)
    server.graph.upsert_extraction.assert_not_awaited()
    # a None event is a no-op as well.
    await server._project_fleet_event(None)
    server.graph.upsert_extraction.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. Enrollment-gated checkout fleet lane (the decisive proof)
# ---------------------------------------------------------------------------


def _platform_fleet(base: Path, settings: Settings) -> FleetManager:
    manager = FleetManager(eventloom_path=base, settings=settings)
    manager.create_fleet("platform", summary="Platform fleet", actor="founder")
    manager.enroll_agent("platform", "enrolled-agent", actor="founder")  # member
    manager.propagate_outcome(
        "platform",
        outcome="failure",
        summary="Pre-warm the JWKS cache before the first token refresh",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_seed_source(base)],
        confidence=0.95,
        actor="founder",
        claim_key="auth.jwks.cache",
    )
    return manager


async def test_checkout_fleet_lane_enrollment_gating(tmp_path: Path) -> None:
    """Decisive: an enrolled agent receives the fleet lane; a non-enrolled agent does not."""
    base = tmp_path / ".eventloom"
    settings = Settings(fleet_enabled=True)
    _platform_fleet(base, settings)

    fabric = MemoryFabric(eventloom_path=str(base))
    fabric.settings = settings
    await fabric.connect()
    try:
        enrolled = await fabric.checkout_memory(
            "jwks cache",
            session_id="session-1",
            fleet_ids=["platform"],
            agent_id="enrolled-agent",
            record_reinforcement=False,
        )
        fleet_diag = enrolled.diagnostics.get("fleet")
        assert fleet_diag is not None
        assert fleet_diag["count"] == 1
        assert fleet_diag["non_authoritative"] is True
        item = fleet_diag["items"][0]
        assert item["fleet_id"] == "platform"
        assert item["review_status"] == "active"
        assert item["non_authoritative"] is True
        assert item["citation"]  # cited
        assert enrolled.diagnostics["source_lanes"].get("fleet") == 1

        not_enrolled = await fabric.checkout_memory(
            "jwks cache",
            session_id="session-2",
            fleet_ids=["platform"],
            agent_id="stranger",
            record_reinforcement=False,
        )
        assert not_enrolled.diagnostics.get("fleet") is None
        assert "fleet" not in not_enrolled.diagnostics.get("source_lanes", {})
    finally:
        await fabric.close()


async def test_checkout_fleet_lane_excludes_untrusted_and_pending(tmp_path: Path) -> None:
    base = tmp_path / ".eventloom"
    settings = Settings(fleet_enabled=True)
    manager = _platform_fleet(base, settings)
    manager.enroll_agent("platform", "sandboxed", actor="founder")
    manager.assign_trust("platform", "sandboxed", trust_tier="untrusted", actor="founder")
    active_pid = next(m.promotion_id for m in manager.fleet_brief("platform").active_promotions)
    pending = manager.propagate_rule(
        "platform",
        rule="held rule",
        trigger="t",
        origin_session="agent-a",
        origin_actor="agent-a",
        source_events=[_seed_source(base, summary="pending one")],
        confidence=0.2,
        actor="founder",
    )
    assert pending.review_status == "pending"

    fabric = MemoryFabric(eventloom_path=str(base))
    fabric.settings = settings
    await fabric.connect()
    try:
        enrolled = await fabric.checkout_memory(
            "fleet memory",
            session_id="session-1",
            fleet_ids=["platform"],
            agent_id="enrolled-agent",
            record_reinforcement=False,
        )
        surfaced = {item["promotion_id"] for item in enrolled.diagnostics["fleet"]["items"]}
        assert active_pid in surfaced
        assert pending.promotion_id not in surfaced  # pending never surfaced as active

        untrusted = await fabric.checkout_memory(
            "fleet memory",
            session_id="session-2",
            fleet_ids=["platform"],
            agent_id="sandboxed",
            record_reinforcement=False,
        )
        assert untrusted.diagnostics.get("fleet") is None
    finally:
        await fabric.close()


async def test_checkout_fleet_disabled_no_lane(tmp_path: Path) -> None:
    base = tmp_path / ".eventloom"
    _platform_fleet(base, Settings(fleet_enabled=True))

    fabric = MemoryFabric(eventloom_path=str(base))
    fabric.settings = Settings(fleet_enabled=False)
    await fabric.connect()
    try:
        checkout = await fabric.checkout_memory(
            "jwks cache",
            session_id="session-1",
            fleet_ids=["platform"],
            agent_id="enrolled-agent",
            record_reinforcement=False,
        )
        assert checkout.diagnostics.get("fleet") is None
        assert "fleet" not in checkout.diagnostics.get("source_lanes", {})
    finally:
        await fabric.close()


async def test_mcp_checkout_fleet_lane_enrollment_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over MCP: an enrolled agent_id receives the fleet lane; a stranger does not."""
    from zaxy.config import get_settings

    monkeypatch.setenv("FLEET_ENABLED", "true")
    get_settings.cache_clear()

    server = _real_server(tmp_path)
    base = tmp_path / ".eventloom"
    source = _seed_source(base, "enrolled-agent")
    await server.handle_fleet_create(
        {"fleet_id": "platform", "summary": "Platform fleet", "actor": "founder"}
    )
    await server.handle_fleet_enroll(
        {"fleet_id": "platform", "agent_id": "enrolled-agent", "actor": "founder"}
    )
    promoted = _text(
        await server.handle_fleet_promote(
            {
                "fleet_id": "platform",
                "kind": "outcome",
                "outcome": "failure",
                "summary": "Pre-warm the JWKS cache before the first token refresh",
                "origin_session": "enrolled-agent",
                "origin_actor": "enrolled-agent",
                "actor": "enrolled-agent",
                "confidence": 0.95,
                "source_events": [source],
                "claim_key": "auth.jwks.cache",
            }
        )
    )
    assert promoted["review_status"] == "active"

    enrolled = _text(
        await server.handle_memory_checkout(
            {
                "query": "jwks cache",
                "session_id": "session-1",
                "fleet_ids": ["platform"],
                "agent_id": "enrolled-agent",
            }
        )
    )
    fleet_diag = enrolled["diagnostics"].get("fleet")
    assert fleet_diag is not None
    assert fleet_diag["count"] == 1
    assert fleet_diag["non_authoritative"] is True
    item = fleet_diag["items"][0]
    assert item["fleet_id"] == "platform"
    assert item["promotion_id"] == promoted["promotion_id"]
    assert item["review_status"] == "active"
    assert item["citation"]
    assert enrolled["diagnostics"]["source_lanes"].get("fleet") == 1

    stranger = _text(
        await server.handle_memory_checkout(
            {
                "query": "jwks cache",
                "session_id": "session-2",
                "fleet_ids": ["platform"],
                "agent_id": "stranger",
            }
        )
    )
    assert stranger["diagnostics"].get("fleet") is None
    assert "fleet" not in stranger["diagnostics"].get("source_lanes", {})


async def test_mcp_checkout_omitting_fleet_arguments_yields_no_fleet_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting fleet_ids/agent_id keeps memory_checkout on its pre-fleet behavior."""
    from zaxy.config import get_settings

    monkeypatch.setenv("FLEET_ENABLED", "true")
    get_settings.cache_clear()

    server = _real_server(tmp_path)
    base = tmp_path / ".eventloom"
    _platform_fleet(base, Settings(fleet_enabled=True))

    checkout = _text(
        await server.handle_memory_checkout({"query": "jwks cache", "session_id": "session-1"})
    )
    assert checkout["diagnostics"].get("fleet") is None


def test_memory_checkout_schema_exposes_fleet_arguments() -> None:
    """memory_checkout should advertise optional fleet_ids and agent_id arguments."""
    from zaxy.mcp_server import TOOLS

    schema = next(t for t in TOOLS if t.name == "memory_checkout").inputSchema
    assert schema["properties"]["fleet_ids"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Fleet IDs whose promoted, non-authoritative memory should be "
            "considered; each lane is gated on agent_id enrollment and trust."
        ),
    }
    assert schema["properties"]["agent_id"]["type"] == "string"
    assert schema["required"] == ["query"]


def _fleet_memory_state(promotion_id: str, *, visibility_scope: str, review_status: str = "active") -> FleetMemoryState:
    return FleetMemoryState(
        promotion_id=promotion_id,
        fleet_id="platform",
        kind="outcome",
        review_status=review_status,
        visibility_scope=visibility_scope,
        confidence=0.9,
        summary="cited fleet memory",
        origin_session="agent-a",
        origin_actor="agent-a",
        actor="founder",
        source_events=[{"seq": 1, "hash": "a" * 64}],
        gate_event={"seq": 2, "hash": "b" * 64},
        keystone=False,
        conflict_key=None,
        conflict_value=None,
        event_seq=3,
        event_hash="c" * 64,
        timestamp="2026-06-28T00:00:00Z",
    )


async def test_fleet_lane_dedupes_and_excludes_below_fleet_scope(tmp_path: Path) -> None:
    """The fleet lane dedupes a promotion requested twice and skips below-fleet scope."""
    base = tmp_path / ".eventloom"
    fabric = MemoryFabric(eventloom_path=str(base))
    fabric.settings = Settings(fleet_enabled=True)
    await fabric.connect()
    try:
        brief = MagicMock()
        brief.agents = [FleetAgentState(agent_id="enrolled-agent", trust_tier="member")]
        brief.active_promotions = [
            _fleet_memory_state("promo-fleet", visibility_scope="fleet"),
            _fleet_memory_state("promo-mission", visibility_scope="mission"),  # below fleet -> skipped
        ]
        fake_manager = MagicMock()
        fake_manager.fleet_brief.return_value = brief
        fabric._fleet_manager = lambda: fake_manager  # type: ignore[method-assign]

        # Same fleet requested twice: the promotion is surfaced once (dedup), mission scope dropped.
        contexts = fabric._fleet_lane_contexts(["platform", "platform"], agent_id="enrolled-agent")
        ids = [c.metadata["promotion_id"] for c in contexts]
        assert ids == ["promo-fleet"]
        assert contexts[0].metadata["entity_type"] == "fleet_promotion"
    finally:
        await fabric.close()


async def test_fleet_lane_degrades_when_brief_unavailable(tmp_path: Path) -> None:
    """A fleet whose brief cannot be resolved is skipped (degraded), never crashes checkout."""
    base = tmp_path / ".eventloom"
    fabric = MemoryFabric(eventloom_path=str(base))
    fabric.settings = Settings(fleet_enabled=True)
    await fabric.connect()
    try:
        fake_manager = MagicMock()
        fake_manager.fleet_brief.side_effect = RuntimeError("replay unavailable")
        fabric._fleet_manager = lambda: fake_manager  # type: ignore[method-assign]
        metrics = MagicMock()
        with patch("zaxy.core.fabric.get_metrics", return_value=metrics):
            contexts = fabric._fleet_lane_contexts(["platform"], agent_id="enrolled-agent")
        assert contexts == []
        metrics.record_degraded_operation.assert_any_call("query", "fleet_lane_unavailable")
    finally:
        await fabric.close()


def test_checkout_fleet_summary_filters_and_dedupes() -> None:
    """``_checkout_fleet`` reads only fleet-lane contexts and dedupes by promotion_id."""
    from zaxy.context import Context
    from zaxy.core.checkout_build import _checkout_fleet

    def _ctx(lane: str, promotion_id: Any) -> Context:
        return Context(
            content="fleet memory",
            source="fleet" if lane == "fleet" else "graph",
            score=0.9,
            metadata={
                "assembly_lane": lane,
                "promotion_id": promotion_id,
                "fleet_id": "platform",
                "kind": "outcome",
                "review_status": "active",
                "visibility_scope": "fleet",
                "citation": "eventloom://fleet:platform/events/3#abc",
            },
        )

    items = _checkout_fleet(
        [
            _ctx("graph", "ignored"),  # non-fleet lane -> skipped
            _ctx("fleet", "promo-1"),
            _ctx("fleet", "promo-1"),  # duplicate -> deduped
            _ctx("fleet", None),  # non-str promotion_id -> skipped
        ]
    )
    assert [item["promotion_id"] for item in items] == ["promo-1"]
    assert items[0]["non_authoritative"] is True
    assert items[0]["authority_status"] == "non_authoritative"


# ---------------------------------------------------------------------------
# 4. Coordination -> fleet bridge
# ---------------------------------------------------------------------------


def test_escalate_finding_to_fleet_cites_the_promoted_finding(tmp_path: Path) -> None:
    base = tmp_path / ".eventloom"
    coordination = CoordinationManager(eventloom_path=base)
    coordination.start_mission("m1", objective="ship auth", actor="coordinator")
    coordination.create_worker("m1", "w1", actor="coordinator")
    finding = coordination.report_finding(
        "m1",
        "w1",
        summary="cache must be pre-warmed",
        actor="w1",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        confidence=0.9,
        claim_key="auth.jwks",
        claim_value="prewarm",
    )
    coordination.review_finding("m1", finding.finding_id, status="accepted", actor="coordinator")
    promotion = coordination.promote_finding("m1", finding.finding_id, actor="coordinator")

    fleet_manager = FleetManager(eventloom_path=base)
    fleet_manager.create_fleet("platform", summary="Platform", actor="coordinator")  # coordinator is steward

    result = coordination.escalate_finding_to_fleet(
        "m1", finding.finding_id, "platform", fleet_manager=fleet_manager, actor="coordinator"
    )
    assert result.rejected is False
    assert result.promotion_id
    # the fleet promotion cites the mission coordination.finding.promoted event
    assert result.promotion_event.payload["source_events"] == [
        {"seq": promotion.event.seq, "hash": promotion.event.hash}
    ]
    # and it is governed: routed through the I4 gate, non-authoritative
    assert result.gate_event is not None
    assert result.promotion_event.payload["authority_status"] == "non_authoritative"


def test_escalate_finding_as_skill_routes_through_gate(tmp_path: Path) -> None:
    """``as_skill=True`` proposes a governed fleet skill citing the promoted finding."""
    base = tmp_path / ".eventloom"
    coordination = CoordinationManager(eventloom_path=base)
    coordination.start_mission("m1", objective="ship auth", actor="coordinator")
    coordination.create_worker("m1", "w1", actor="coordinator")
    finding = coordination.report_finding(
        "m1", "w1", summary="cache must be pre-warmed", actor="w1",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        confidence=0.9, claim_key="auth.jwks", claim_value="prewarm",
    )
    coordination.review_finding("m1", finding.finding_id, status="accepted", actor="coordinator")
    promotion = coordination.promote_finding("m1", finding.finding_id, actor="coordinator")

    fleet_manager = FleetManager(eventloom_path=base)
    fleet_manager.create_fleet("platform", summary="Platform", actor="coordinator")

    result = coordination.escalate_finding_to_fleet(
        "m1", finding.finding_id, "platform",
        fleet_manager=fleet_manager, actor="coordinator", as_skill=True, skill_version="3",
    )
    assert result.rejected is False
    assert result.kind == "skill"
    assert result.promotion_event.payload["skill_id"] == f"finding:{finding.finding_id}"
    assert result.promotion_event.payload["skill_version"] == "3"
    assert result.promotion_event.payload["source_events"] == [
        {"seq": promotion.event.seq, "hash": promotion.event.hash}
    ]


def test_escalate_unpromoted_finding_raises(tmp_path: Path) -> None:
    """Escalating a finding with no accepted promotion is a hard error (no silent crossing)."""
    base = tmp_path / ".eventloom"
    coordination = CoordinationManager(eventloom_path=base)
    coordination.start_mission("m1", objective="ship auth", actor="coordinator")
    coordination.create_worker("m1", "w1", actor="coordinator")
    finding = coordination.report_finding(
        "m1", "w1", summary="not promoted", actor="w1",
        confidence=0.9, claim_key="k", claim_value="v",
    )  # reported but never promoted

    fleet_manager = FleetManager(eventloom_path=base)
    fleet_manager.create_fleet("platform", summary="Platform", actor="coordinator")
    with pytest.raises(ValueError, match="no accepted promotion"):
        coordination.escalate_finding_to_fleet(
            "m1", finding.finding_id, "platform", fleet_manager=fleet_manager, actor="coordinator"
        )


def test_promote_finding_reinforcement_failure_is_best_effort(tmp_path: Path) -> None:
    """A salience-reinforcement failure during promotion is swallowed and recorded as degraded."""
    base = tmp_path / ".eventloom"
    coordination = CoordinationManager(eventloom_path=base)
    coordination.start_mission("m1", objective="ship auth", actor="coordinator")
    coordination.create_worker("m1", "w1", actor="coordinator")
    finding = coordination.report_finding(
        "m1", "w1", summary="cache must be pre-warmed", actor="w1",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        confidence=0.9, claim_key="auth.jwks", claim_value="prewarm",
    )
    coordination.review_finding("m1", finding.finding_id, status="accepted", actor="coordinator")

    metrics = MagicMock()
    with (
        patch("zaxy.coordination.build_promoted_reinforcement_event", side_effect=RuntimeError("boom")),
        patch("zaxy.coordination.get_metrics", return_value=metrics),
    ):
        result = coordination.promote_finding("m1", finding.finding_id, actor="coordinator")
    # the promotion itself still succeeds; only the observability reinforcement degraded.
    assert result.finding_id == finding.finding_id
    metrics.record_degraded_operation.assert_any_call("append", "salience_reinforcement_unavailable")


# ---------------------------------------------------------------------------
# Admin gating of governance-structure mutations (MCP boundary)
# ---------------------------------------------------------------------------


async def test_fleet_governance_tools_admin_gated_when_token_configured(tmp_path: Path) -> None:
    """With an admin token configured, create/enroll/assign_trust demand it.

    The MCP ``actor`` argument is self-asserted, so the manager's trust-tier
    checks cannot authenticate a remote caller; the admin token is the
    operator-controlled gate for governance-structure mutations.
    """
    server = _real_server(tmp_path)
    server._admin_token = "fleet-admin-secret"

    with pytest.raises(PermissionError, match="admin_token"):
        await server.handle_fleet_create({"fleet_id": "gated", "summary": "Gated"})
    with pytest.raises(PermissionError, match="admin_token"):
        await server.handle_fleet_create(
            {"fleet_id": "gated", "summary": "Gated", "admin_token": "wrong"}
        )

    created = _text(
        await server.handle_fleet_create(
            {"fleet_id": "gated", "summary": "Gated", "actor": "founder", "admin_token": "fleet-admin-secret"}
        )
    )
    assert created["fleet_id"] == "gated"

    with pytest.raises(PermissionError, match="admin_token"):
        await server.handle_fleet_enroll({"fleet_id": "gated", "agent_id": "worker", "actor": "founder"})
    enrolled = _text(
        await server.handle_fleet_enroll(
            {
                "fleet_id": "gated",
                "agent_id": "worker",
                "actor": "founder",
                "admin_token": "fleet-admin-secret",
            }
        )
    )
    assert enrolled["trust_tier"] == "member"

    with pytest.raises(PermissionError, match="admin_token"):
        await server.handle_fleet_assign_trust(
            {"fleet_id": "gated", "agent_id": "worker", "trust_tier": "trusted", "actor": "founder"}
        )
    assigned = _text(
        await server.handle_fleet_assign_trust(
            {
                "fleet_id": "gated",
                "agent_id": "worker",
                "trust_tier": "trusted",
                "actor": "founder",
                "admin_token": "fleet-admin-secret",
            }
        )
    )
    assert assigned["trust_tier"] == "trusted"


async def test_fleet_enroll_via_tool_rejects_non_steward_actor(tmp_path: Path) -> None:
    """The manager's steward requirement surfaces through the MCP tool."""
    server = _real_server(tmp_path)
    await server.handle_fleet_create({"fleet_id": "beta", "summary": "Beta", "actor": "founder"})
    await server.handle_fleet_enroll({"fleet_id": "beta", "agent_id": "mallory", "actor": "founder"})

    with pytest.raises(ValueError, match="steward"):
        await server.handle_fleet_enroll(
            {"fleet_id": "beta", "agent_id": "accomplice", "trust_tier": "steward", "actor": "mallory"}
        )
