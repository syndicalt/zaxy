"""Tests for the outcome-driven learning loop (Zaxy 3 / I1)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from zaxy.core.fabric import MemoryFabric
from zaxy.evolution_policy import evaluate_evolution_gate, resolve_evolution_policy
from zaxy.outcome_learning import (
    OUTCOME_ACTUAL,
    OUTCOME_EVENT_TYPE,
    RULE_GENERATED_EVENT_TYPE,
    RULE_PROPOSED_EVENT_TYPE,
    build_outcome_event,
    build_rule_event,
    prediction_error,
    preventive_rule_confidence,
    resolve_rule_confidence,
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

    def test_resolve_rule_confidence_defaults_preserve_the_builtin_table(self) -> None:
        """A settings object configuring nothing yields the historical 0.9/0.7 split."""
        table = resolve_rule_confidence(SimpleNamespace())
        assert table == {"failure": 0.9, "partial": 0.7}

    def test_resolve_rule_confidence_reads_configured_values(self) -> None:
        """Per-deployment settings retune each outcome's rule confidence independently."""
        table = resolve_rule_confidence(
            SimpleNamespace(
                outcome_rule_confidence_failure=0.6,
                outcome_rule_confidence_partial=0.95,
            )
        )
        assert table == {"failure": 0.6, "partial": 0.95}

    @pytest.mark.parametrize(
        "settings",
        [
            SimpleNamespace(outcome_rule_confidence_failure=1.5),
            SimpleNamespace(outcome_rule_confidence_failure=-0.1),
            SimpleNamespace(outcome_rule_confidence_partial="high"),
            SimpleNamespace(outcome_rule_confidence_partial=True),
        ],
    )
    def test_resolve_rule_confidence_rejects_malformed(self, settings: object) -> None:
        """Out-of-range and non-numeric rule confidences raise rather than silently default."""
        with pytest.raises(ValueError):
            resolve_rule_confidence(settings)

    def test_configured_confidence_flips_partial_rules_over_the_gate_threshold(self) -> None:
        """Raising partial confidence lets partial-outcome rules auto-apply at the 0.85 default."""
        policy = resolve_evolution_policy(SimpleNamespace())

        default_table = resolve_rule_confidence(SimpleNamespace())
        default_confidence = preventive_rule_confidence("partial", defaults=default_table)
        assert (
            evaluate_evolution_gate(
                "rule_generate", default_confidence, policy=policy
            ).auto_apply
            is False
        )

        tuned_table = resolve_rule_confidence(
            SimpleNamespace(outcome_rule_confidence_partial=0.95)
        )
        tuned_confidence = preventive_rule_confidence("partial", defaults=tuned_table)
        assert (
            evaluate_evolution_gate("rule_generate", tuned_confidence, policy=policy).auto_apply
            is True
        )

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


class TestWithheldRuleIsNotAssembled:
    """The I4 gate's decision must hold all the way through checkout assembly.

    A rule the gate declined to auto-apply is a proposal awaiting review. If it
    still reaches the assembled prompt, the model conditions on it exactly as if
    it had been approved and the gate is decorative. These cases differ only in
    the outcome that drives the gate decision -- same lesson text, same query --
    so a pass isolates the gate as the cause.
    """

    _LESSON = "CANARY-prefer-the-batch-path-when-batching"
    _QUERY = "what should I know about batching?"

    async def _record_and_checkout(
        self, tmp_path: Path, outcome: str
    ) -> tuple[dict[str, object], str]:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            result = await fabric.record_outcome(
                outcome=outcome,
                summary="the batch path only half worked",
                lesson=self._LESSON,
                trigger="when batching over 50 items",
                session_id="ext",
            )
            checkout = await fabric.checkout_memory(self._QUERY, session_id="ext", limit=25)
        finally:
            await fabric.close()
        return result["rule"], checkout.prompt

    async def test_withheld_rule_never_reaches_the_prompt(self, tmp_path: Path) -> None:
        """A gate-withheld (pending) rule is excluded from the assembled prompt."""
        rule, prompt = await self._record_and_checkout(tmp_path, "partial")
        assert rule["review_status"] == "pending"
        assert rule["auto_applied"] is False
        assert self._LESSON not in prompt

    async def test_auto_applied_rule_still_reaches_the_prompt(self, tmp_path: Path) -> None:
        """An auto-applied (active) rule still surfaces -- the filter is not a blanket drop."""
        rule, prompt = await self._record_and_checkout(tmp_path, "failure")
        assert rule["review_status"] == "active"
        assert rule["auto_applied"] is True
        assert self._LESSON in prompt

    async def test_withheld_rule_is_absent_from_current_facts(self, tmp_path: Path) -> None:
        """The withheld rule is excluded from current_facts, not merely deprioritised."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            await fabric.record_outcome(
                outcome="partial",
                summary="the batch path only half worked",
                lesson=self._LESSON,
                trigger="when batching over 50 items",
                session_id="ext",
            )
            checkout = await fabric.checkout_memory(self._QUERY, session_id="ext", limit=25)
        finally:
            await fabric.close()
        assert not any(self._LESSON in json.dumps(fact, default=str) for fact in checkout.current_facts)

    async def test_withheld_rule_remains_in_the_replayable_log(self, tmp_path: Path) -> None:
        """Exclusion is an assembly-time filter, never a deletion from the event log."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            await fabric.record_outcome(
                outcome="partial",
                summary="the batch path only half worked",
                lesson=self._LESSON,
                trigger="when batching over 50 items",
                session_id="ext",
            )
        finally:
            await fabric.close()
        events = fabric.session_manager.get("ext").eventlog.read_all()
        proposed = [e for e in events if e.type == RULE_PROPOSED_EVENT_TYPE]
        assert len(proposed) == 1
        assert proposed[0].payload["rule"] == self._LESSON
class TestRecordOutcomeTargetResolution:
    """`record_outcome` must resolve its target against the sealed log.

    A citation that resolves to nothing is worse than no citation: it lands in
    an append-only log and cannot be retracted. The sibling evolution ops
    (edit/rollback/forget) all reject unresolvable targets; this asserts
    `record_outcome` now matches them.
    """

    async def _fabric(self, tmp_path: Path) -> MemoryFabric:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        return fabric

    async def test_rejects_target_seq_that_does_not_exist(self, tmp_path: Path) -> None:
        """A (seq, hash) pointing past the end of the log is refused, not silently cited."""
        fabric = await self._fabric(tmp_path)
        try:
            await fabric.append("goal.created", actor="user", payload={"t": "x"}, session_id="ext")
            with pytest.raises(ValueError, match="no event at seq 999"):
                await fabric.record_outcome(
                    outcome="failure", summary="s",
                    target_seq=999, target_hash="f" * 64, session_id="ext",
                )
        finally:
            await fabric.close()

    async def test_rejects_hash_that_does_not_match_the_sealed_event(self, tmp_path: Path) -> None:
        """A correct seq with the wrong hash is refused -- the pair must match the seal."""
        fabric = await self._fabric(tmp_path)
        try:
            event = await fabric.append(
                "goal.created", actor="user", payload={"t": "x"}, session_id="ext"
            )
            with pytest.raises(ValueError, match="does not match the sealed event"):
                await fabric.record_outcome(
                    outcome="failure", summary="s",
                    target_seq=event.seq, target_hash="f" * 64, session_id="ext",
                )
        finally:
            await fabric.close()

    async def test_rejects_half_supplied_target(self, tmp_path: Path) -> None:
        """Supplying seq without hash is an error, not a silently dropped reinforcement."""
        fabric = await self._fabric(tmp_path)
        try:
            event = await fabric.append(
                "goal.created", actor="user", payload={"t": "x"}, session_id="ext"
            )
            with pytest.raises(ValueError, match="target_hash"):
                await fabric.record_outcome(
                    outcome="failure", summary="s",
                    target_seq=event.seq, target_hash=None, session_id="ext",
                )
        finally:
            await fabric.close()

    async def test_no_target_still_records_outcome_without_reinforcement(
        self, tmp_path: Path
    ) -> None:
        """Omitting the target entirely stays valid -- the loop is not made target-mandatory."""
        fabric = await self._fabric(tmp_path)
        try:
            result = await fabric.record_outcome(
                outcome="failure", summary="no target reported", session_id="ext"
            )
        finally:
            await fabric.close()
        assert result["outcome"] == "failure"
        assert "reinforced" not in result

    async def test_resolvable_target_still_reinforces(self, tmp_path: Path) -> None:
        """A real (seq, hash) reinforces exactly as before -- no behavioural regression."""
        fabric = await self._fabric(tmp_path)
        try:
            event = await fabric.append(
                "goal.created", actor="user", payload={"t": "x"}, session_id="ext"
            )
            result = await fabric.record_outcome(
                outcome="success", summary="worked",
                target_seq=event.seq, target_hash=event.hash, session_id="ext",
            )
        finally:
            await fabric.close()
        assert result["reinforced"] == "confirmed"


class TestRolledBackRuleIsNotAssembled:
    """Rolling back an auto-applied rule must actually stop it governing the agent.

    `auto_with_rollback` is the default autonomy tier (ZAXY-3 §11 decision 1) and
    its whole promise is that an auto-applied change is reversible. A rollback
    that reports success while the rule keeps reaching the prompt is worse than
    no rollback, because the operator believes the change is undone.
    """

    _LESSON = "CANARY-rollback-me-refresh-the-token-when-batching"
    _QUERY = "what should I know about batching tokens?"

    async def _generate_rule(self, fabric: MemoryFabric) -> tuple[int, str]:
        result = await fabric.record_outcome(
            outcome="failure",
            summary="token expired when batching",
            lesson=self._LESSON,
            trigger="when batching",
            session_id="ext",
        )
        assert result["rule"]["review_status"] == "active"
        events = fabric.session_manager.get("ext").eventlog.read_all()
        rule_event = next(e for e in events if e.type == RULE_GENERATED_EVENT_TYPE)
        return rule_event.seq, rule_event.hash

    async def test_rolled_back_rule_leaves_the_prompt(self, tmp_path: Path) -> None:
        """After rollback the rule no longer reaches the assembled prompt."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            seq, event_hash = await self._generate_rule(fabric)
            before = await fabric.checkout_memory(self._QUERY, session_id="ext", limit=25)
            assert self._LESSON in before.prompt, "the rule must surface first for this to mean anything"

            await fabric.rollback_memory(
                target_seq=seq, target_hash=event_hash, reason="undo it", session_id="ext"
            )
            after = await fabric.checkout_memory(self._QUERY, session_id="ext", limit=25)
        finally:
            await fabric.close()
        assert self._LESSON not in after.prompt

    async def test_rollback_does_not_delete_the_rule_event(self, tmp_path: Path) -> None:
        """Reversal is an assembly-time exclusion; the log keeps both events."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            seq, event_hash = await self._generate_rule(fabric)
            await fabric.rollback_memory(
                target_seq=seq, target_hash=event_hash, reason="undo it", session_id="ext"
            )
        finally:
            await fabric.close()
        types = [e.type for e in fabric.session_manager.get("ext").eventlog.read_all()]
        assert RULE_GENERATED_EVENT_TYPE in types
        assert "memory.rolled_back" in types
