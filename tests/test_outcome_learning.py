"""Tests for the outcome-driven learning loop (Zaxy 3 / I1)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from zaxy.core.fabric import MemoryFabric
from zaxy.outcome_learning import (
    OUTCOME_ACTUAL,
    OUTCOME_EVENT_TYPE,
    RULE_GENERATED_EVENT_TYPE,
    RULE_PROPOSED_EVENT_TYPE,
    build_outcome_event,
    build_rule_event,
    prediction_error,
    preventive_rule_confidence,
    validate_outcome,
)
from zaxy.salience import SALIENCE_REINFORCEMENT_MULTIPLIERS

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


class TestPredictionError:
    def test_outcome_actual_anchors_success_failure_partial(self) -> None:
        assert OUTCOME_ACTUAL == {"success": 1.0, "failure": 0.0, "partial": 0.5}

    def test_success_surprise_is_distance_from_full_success(self) -> None:
        # actual 1.0: a confident, correct prior is barely surprising; a
        # confident wrong-direction prior is maximally surprising.
        assert prediction_error("success", 0.9) == pytest.approx(0.1)
        assert prediction_error("success", 0.0) == pytest.approx(1.0)

    def test_failure_surprise_equals_the_prior(self) -> None:
        # actual 0.0: a high prior on a memory that then failed is a big surprise.
        assert prediction_error("failure", 0.9) == pytest.approx(0.9)
        assert prediction_error("failure", 0.0) == pytest.approx(0.0)

    def test_partial_surprise_is_measured_from_the_midpoint(self) -> None:
        assert prediction_error("partial", 0.5) == pytest.approx(0.0)
        assert prediction_error("partial", 1.0) == pytest.approx(0.5)

    @pytest.mark.parametrize("prior", [-0.1, 1.5, math.nan, math.inf, True])
    def test_rejects_bad_prior(self, prior: object) -> None:
        with pytest.raises(ValueError, match="prior"):
            prediction_error("failure", prior)

    def test_rejects_bad_outcome(self) -> None:
        with pytest.raises(ValueError, match="outcome"):
            prediction_error("explosion", 0.5)


class TestOutcomeEventSurprise:
    def test_records_prior_and_prediction_error_when_given(self) -> None:
        spec = build_outcome_event(
            actor="agent",
            session_id="s",
            outcome="failure",
            summary="it broke",
            prior=0.9,
            prediction_error=0.9,
        )
        assert spec["payload"]["prior"] == pytest.approx(0.9)
        assert spec["payload"]["prediction_error"] == pytest.approx(0.9)

    def test_omits_prior_and_prediction_error_when_none(self) -> None:
        spec = build_outcome_event(
            actor="agent", session_id="s", outcome="failure", summary="it broke"
        )
        assert "prior" not in spec["payload"]
        assert "prediction_error" not in spec["payload"]

    @pytest.mark.parametrize("bad", [-0.1, 1.5, True])
    def test_rejects_out_of_range_prior(self, bad: object) -> None:
        with pytest.raises(ValueError, match="prior"):
            build_outcome_event(
                actor="agent", session_id="s", outcome="failure", summary="x", prior=bad
            )


class TestRecordOutcomeSurprise:
    async def _seed_target(self, fabric: MemoryFabric) -> tuple[int, str]:
        event = await fabric.append(
            "goal.created", actor="user", payload={"title": "Ship I1.3"}, session_id="ext"
        )
        return event.seq, event.hash

    def _reinforcement(self, fabric: MemoryFabric) -> dict[str, object]:
        events = fabric.session_manager.get("ext").eventlog.read_all()
        reinforcement = next(e for e in events if e.type == "memory.reinforcement")
        return reinforcement.payload

    def _outcome(self, fabric: MemoryFabric) -> dict[str, object]:
        events = fabric.session_manager.get("ext").eventlog.read_all()
        outcome = next(e for e in events if e.type == OUTCOME_EVENT_TYPE)
        return outcome.payload

    async def test_failure_high_prior_strengthens_attenuation_below_default(
        self, tmp_path: Path
    ) -> None:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            seq, h = await self._seed_target(fabric)
            result = await fabric.record_outcome(
                outcome="failure",
                summary="the recalled memory led us astray",
                target_seq=seq,
                target_hash=h,
                prior=0.9,
                session_id="ext",
            )
        finally:
            await fabric.close()

        assert result["reinforced"] == "invalidated"
        payload = self._reinforcement(fabric)
        # pe == 0.9 drives the weight to the floor, far stronger attenuation
        # than the fixed-table invalidated multiplier (0.2).
        assert payload["weight"] == pytest.approx(0.01)
        assert payload["weight"] < SALIENCE_REINFORCEMENT_MULTIPLIERS["invalidated"]
        outcome = self._outcome(fabric)
        assert outcome["prior"] == pytest.approx(0.9)
        assert outcome["prediction_error"] == pytest.approx(0.9)

    async def test_success_low_prior_amplifies_reinforcement_above_default(
        self, tmp_path: Path
    ) -> None:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            seq, h = await self._seed_target(fabric)
            result = await fabric.record_outcome(
                outcome="success",
                summary="the doubted memory actually worked",
                target_seq=seq,
                target_hash=h,
                prior=0.1,
                session_id="ext",
            )
        finally:
            await fabric.close()

        assert result["reinforced"] == "confirmed"
        payload = self._reinforcement(fabric)
        # pe == 0.9 lifts the weight above the fixed-table confirmed multiplier (1.5).
        assert payload["weight"] == pytest.approx(1.9)
        assert payload["weight"] > SALIENCE_REINFORCEMENT_MULTIPLIERS["confirmed"]
        outcome = self._outcome(fabric)
        assert outcome["prior"] == pytest.approx(0.1)
        assert outcome["prediction_error"] == pytest.approx(0.9)

    async def test_no_prior_is_byte_identical_to_today(self, tmp_path: Path) -> None:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            seq, h = await self._seed_target(fabric)
            await fabric.record_outcome(
                outcome="failure",
                summary="no prior reported",
                target_seq=seq,
                target_hash=h,
                session_id="ext",
            )
        finally:
            await fabric.close()

        # Default path: no weight override, no surprise keys on the outcome event.
        assert "weight" not in self._reinforcement(fabric)
        outcome = self._outcome(fabric)
        assert "prior" not in outcome
        assert "prediction_error" not in outcome
