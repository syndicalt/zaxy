"""Tests for the outcome-driven learning loop (Zaxy 3 / I1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zaxy.core.fabric import MemoryFabric
from zaxy.outcome_learning import (
    OUTCOME_EVENT_TYPE,
    RULE_GENERATED_EVENT_TYPE,
    RULE_PROPOSED_EVENT_TYPE,
    build_outcome_event,
    build_rule_event,
    preventive_rule_confidence,
    validate_outcome,
)

_REF = {"seq": 1, "hash": "a" * 64}


class TestContracts:
    def test_validate_outcome(self) -> None:
        assert validate_outcome("failure") == "failure"
        with pytest.raises(ValueError):
            validate_outcome("explosion")

    def test_preventive_rule_confidence(self) -> None:
        assert preventive_rule_confidence("failure") == 0.9
        assert preventive_rule_confidence("partial") == 0.7
        assert preventive_rule_confidence("failure", 0.42) == 0.42
        with pytest.raises(ValueError):
            preventive_rule_confidence("failure", 1.5)

    def test_build_outcome_event_is_non_authoritative(self) -> None:
        spec = build_outcome_event(
            actor="agent", session_id="ext", outcome="success", summary="worked", target=dict(_REF)
        )
        assert spec["event_type"] == OUTCOME_EVENT_TYPE
        assert spec["payload"]["authority_status"] == "non_authoritative"
        assert spec["payload"]["outcome"] == "success"
        assert spec["payload"]["target"] == _REF

    def test_build_rule_event_auto_vs_proposed(self) -> None:
        generated = build_rule_event(
            actor="a", session_id="ext", auto_applied=True, rule="avoid X", trigger="when Y",
            confidence=0.9, outcome="failure", source_events=[dict(_REF)],
        )
        assert generated["event_type"] == RULE_GENERATED_EVENT_TYPE
        assert generated["payload"]["review_status"] == "active"
        assert generated["payload"]["authority_status"] == "non_authoritative"
        assert generated["payload"]["rule_id"].startswith("rule:")

        proposed = build_rule_event(
            actor="a", session_id="ext", auto_applied=False, rule="avoid X", trigger="when Y",
            confidence=0.7, outcome="partial", source_events=[dict(_REF)],
        )
        assert proposed["event_type"] == RULE_PROPOSED_EVENT_TYPE
        assert proposed["payload"]["review_status"] == "pending"
        # Same rule+trigger+sources → stable id regardless of auto/proposed.
        assert generated["payload"]["rule_id"] == proposed["payload"]["rule_id"]

    def test_build_rule_event_requires_citation(self) -> None:
        with pytest.raises(ValueError):
            build_rule_event(
                actor="a", session_id="ext", auto_applied=True, rule="r", trigger="t",
                confidence=0.9, outcome="failure", source_events=[],
            )

    def test_build_rule_event_rejects_bad_event_ref(self) -> None:
        with pytest.raises(ValueError):
            build_rule_event(
                actor="a", session_id="ext", auto_applied=True, rule="r", trigger="t",
                confidence=0.9, outcome="failure", source_events=[{"seq": 0, "hash": "x"}],
            )


class TestRecordOutcome:
    async def _seed_target(self, fabric: MemoryFabric) -> tuple[int, str]:
        event = await fabric.append(
            "goal.created", actor="user", payload={"title": "Ship I1"}, session_id="ext"
        )
        return event.seq, event.hash

    async def test_success_reinforces_without_rule(self, tmp_path: Path) -> None:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            seq, h = await self._seed_target(fabric)
            result = await fabric.record_outcome(
                outcome="success", summary="used the goal", target_seq=seq, target_hash=h, session_id="ext"
            )
        finally:
            await fabric.close()

        assert result["outcome"] == "success"
        assert result["reinforced"] == "confirmed"
        assert "rule" not in result
        types = [e.type for e in fabric.session_manager.get("ext").eventlog.read_all()]
        assert OUTCOME_EVENT_TYPE in types
        assert "memory.reinforcement" in types
        assert RULE_GENERATED_EVENT_TYPE not in types

    async def test_failure_with_lesson_auto_generates_governed_rule(self, tmp_path: Path) -> None:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            seq, h = await self._seed_target(fabric)
            result = await fabric.record_outcome(
                outcome="failure",
                summary="the cached token expired mid-call",
                lesson="refresh the token before calls when older than 10m",
                trigger="before an authed call",
                target_seq=seq,
                target_hash=h,
                session_id="ext",
            )
        finally:
            await fabric.close()

        assert result["reinforced"] == "invalidated"
        # Default auto_with_rollback + failure confidence 0.9 >= 0.85 -> auto-applied rule.
        assert result["rule"]["auto_applied"] is True
        assert result["rule"]["event_type"] == RULE_GENERATED_EVENT_TYPE
        assert result["rule"]["review_status"] == "active"

        events = fabric.session_manager.get("ext").eventlog.read_all()
        types = [e.type for e in events]
        assert OUTCOME_EVENT_TYPE in types
        assert "memory.reinforcement" in types
        assert "evolution.gate.evaluated" in types  # the governed decision is audited
        rule_events = [e for e in events if e.type == RULE_GENERATED_EVENT_TYPE]
        assert len(rule_events) == 1
        assert rule_events[0].payload["authority_status"] == "non_authoritative"
        assert "refresh the token" in rule_events[0].payload["rule"]

    async def test_partial_lesson_is_proposed_not_auto_applied(self, tmp_path: Path) -> None:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            result = await fabric.record_outcome(
                outcome="partial",
                summary="partly worked",
                lesson="prefer the batch path for >50 items",
                session_id="ext",
            )
        finally:
            await fabric.close()

        # partial confidence 0.7 < 0.85 -> held for review under the default tier.
        assert result["rule"]["auto_applied"] is False
        assert result["rule"]["event_type"] == RULE_PROPOSED_EVENT_TYPE
        assert result["rule"]["review_status"] == "pending"
        types = [e.type for e in fabric.session_manager.get("ext").eventlog.read_all()]
        assert RULE_PROPOSED_EVENT_TYPE in types
        assert RULE_GENERATED_EVENT_TYPE not in types

    async def test_tightened_policy_holds_failure_rule_for_review(self, tmp_path: Path) -> None:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.settings.evolution_op_autonomy = "rule_generate=propose_only"
        await fabric.connect()
        try:
            result = await fabric.record_outcome(
                outcome="failure",
                summary="broke prod",
                lesson="never deploy on friday",
                session_id="ext",
            )
        finally:
            await fabric.close()

        # Operators can still demand review for generated rules.
        assert result["rule"]["auto_applied"] is False
        assert result["rule"]["event_type"] == RULE_PROPOSED_EVENT_TYPE
