"""Tests for the governed Memory Evolution Policy (Zaxy 3 / I4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zaxy.core.fabric import MemoryFabric
from zaxy.evolution_policy import (
    BEHAVIOR_PRESERVING_OP_TIERS,
    GATE_EVENT_TYPE,
    EvolutionGateDecision,
    MemoryEvolutionPolicy,
    build_evolution_gate_event,
    evaluate_evolution_gate,
    parse_op_autonomy,
    resolve_evolution_policy,
)


class TestPolicyResolution:
    def test_default_policy_is_propose_only(self) -> None:
        policy = MemoryEvolutionPolicy()
        assert policy.default_tier == "propose_only"
        assert policy.threshold_for("consolidate") == 0.85
        assert policy.tier_for("forget") == "propose_only"

    def test_resolve_from_settings_defaults_to_propose_only(self) -> None:
        policy = resolve_evolution_policy(SimpleNamespace())
        assert policy.default_tier == "propose_only"
        assert policy.rollback_window_seconds == 86400

    def test_resolve_honors_settings_overrides(self) -> None:
        settings = SimpleNamespace(
            evolution_autonomy_default="auto_with_rollback",
            evolution_rollback_window_seconds=3600,
        )
        policy = resolve_evolution_policy(settings)
        assert policy.default_tier == "auto_with_rollback"
        assert policy.rollback_window_seconds == 3600

    def test_per_op_tier_override(self) -> None:
        policy = MemoryEvolutionPolicy(
            default_tier="auto_with_rollback",
            op_tiers={"forget": "require_review"},
        )
        assert policy.tier_for("consolidate") == "auto_with_rollback"
        assert policy.tier_for("forget") == "require_review"

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValueError):
            MemoryEvolutionPolicy(default_tier="yolo")

    def test_invalid_op_override_rejected(self) -> None:
        with pytest.raises(ValueError):
            MemoryEvolutionPolicy(op_tiers={"teleport": "propose_only"})

    def test_negative_rollback_window_rejected(self) -> None:
        with pytest.raises(ValueError):
            MemoryEvolutionPolicy(rollback_window_seconds=-1)


class TestGateEvaluation:
    def test_propose_only_never_auto_applies_even_at_full_confidence(self) -> None:
        policy = MemoryEvolutionPolicy(default_tier="propose_only")
        decision = evaluate_evolution_gate("rule_generate", 1.0, policy=policy)
        assert decision.auto_apply is False
        assert decision.requires_review is True
        assert decision.decision == "requires_review"
        assert "propose_only" in decision.reason

    def test_require_review_holds_regardless_of_confidence(self) -> None:
        policy = MemoryEvolutionPolicy(default_tier="require_review")
        decision = evaluate_evolution_gate("forget", 0.99, policy=policy)
        assert decision.auto_apply is False
        assert decision.requires_review is True

    def test_auto_with_rollback_applies_at_or_above_threshold(self) -> None:
        policy = MemoryEvolutionPolicy(default_tier="auto_with_rollback")
        decision = evaluate_evolution_gate("consolidate", 0.85, policy=policy)
        assert decision.auto_apply is True
        assert decision.requires_review is False
        assert decision.decision == "auto_apply"
        assert decision.rollback_window_seconds == 86400

    def test_auto_with_rollback_holds_below_threshold(self) -> None:
        policy = MemoryEvolutionPolicy(default_tier="auto_with_rollback")
        decision = evaluate_evolution_gate("consolidate", 0.84, policy=policy)
        assert decision.auto_apply is False
        assert decision.requires_review is True

    def test_per_op_override_beats_default(self) -> None:
        policy = MemoryEvolutionPolicy(
            default_tier="auto_with_rollback",
            op_tiers={"forget": "require_review"},
        )
        # consolidate auto-applies, forget is held — same confidence.
        assert evaluate_evolution_gate("consolidate", 0.95, policy=policy).auto_apply is True
        assert evaluate_evolution_gate("forget", 0.95, policy=policy).auto_apply is False

    def test_invalid_op_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate_evolution_gate("teleport", 0.9, policy=MemoryEvolutionPolicy())

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, "high", True])
    def test_invalid_confidence_rejected(self, confidence: object) -> None:
        with pytest.raises(ValueError):
            evaluate_evolution_gate("consolidate", confidence, policy=MemoryEvolutionPolicy())  # type: ignore[arg-type]


class TestGateEvent:
    def _decision(self) -> EvolutionGateDecision:
        return evaluate_evolution_gate("consolidate", 0.5, policy=MemoryEvolutionPolicy())

    def test_event_shape_is_non_authoritative(self) -> None:
        spec = build_evolution_gate_event(
            actor="zaxy-evolution", session_id="ext", decision=self._decision()
        )
        assert spec["event_type"] == GATE_EVENT_TYPE
        assert spec["thread"] == "ext"
        payload = spec["payload"]
        assert payload["authority_status"] == "non_authoritative"
        assert payload["op"] == "consolidate"
        assert payload["tier"] == "propose_only"
        assert payload["decision"] == "requires_review"
        assert payload["auto_apply"] is False

    def test_candidate_ref_is_snapshotted(self) -> None:
        spec = build_evolution_gate_event(
            actor="a",
            session_id="ext",
            decision=self._decision(),
            candidate_ref={"candidate_id": "consolidation:claim:" + "0" * 24, "extra": "dropped"},
        )
        ref = spec["payload"]["candidate_ref"]
        assert ref == {"candidate_id": "consolidation:claim:" + "0" * 24}

    def test_empty_candidate_ref_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_evolution_gate_event(
                actor="a", session_id="ext", decision=self._decision(), candidate_ref={}
            )

    def test_blank_actor_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_evolution_gate_event(actor="  ", session_id="ext", decision=self._decision())


class TestFabricEvolutionGate:
    async def test_gate_records_auditable_event_and_defaults_to_propose_only(
        self, tmp_path: Path
    ) -> None:
        """The default policy holds for review and records a replayable gate event."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            decision = await fabric.evaluate_evolution_gate(
                "rule_generate", 0.99, session_id="ext"
            )
        finally:
            await fabric.close()

        # propose_only default: nothing auto-promotes, even at high confidence.
        assert decision.auto_apply is False
        assert decision.requires_review is True

        log = fabric.session_manager.get("ext").eventlog
        events = log.read_all()
        assert log.verify().ok is True
        gate_events = [e for e in events if e.type == GATE_EVENT_TYPE]
        assert len(gate_events) == 1
        assert gate_events[0].payload["authority_status"] == "non_authoritative"
        assert gate_events[0].payload["op"] == "rule_generate"
        assert gate_events[0].payload["auto_apply"] is False

    async def test_auto_with_rollback_tier_auto_applies(self, tmp_path: Path) -> None:
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.settings.evolution_autonomy_default = "auto_with_rollback"
        await fabric.connect()
        try:
            decision = await fabric.evaluate_evolution_gate(
                "consolidate", 0.9, session_id="ext"
            )
        finally:
            await fabric.close()

        assert decision.auto_apply is True
        assert decision.decision == "auto_apply"
        log = fabric.session_manager.get("ext").eventlog
        gate_events = [e for e in log.read_all() if e.type == GATE_EVENT_TYPE]
        assert gate_events[0].payload["auto_apply"] is True


class TestPerOpAutonomyConfig:
    def test_resolve_bakes_behavior_preserving_update_tier(self) -> None:
        policy = resolve_evolution_policy(SimpleNamespace())
        # inferred-edge generation (op "update") keeps auto-applying by default...
        assert policy.tier_for("update") == "auto_with_rollback"
        # ...while everything else stays conservative.
        assert policy.tier_for("consolidate") == "propose_only"
        assert policy.tier_for("forget") == "propose_only"

    def test_settings_override_can_tighten_update(self) -> None:
        settings = SimpleNamespace(evolution_op_autonomy="update=propose_only")
        policy = resolve_evolution_policy(settings)
        assert policy.tier_for("update") == "propose_only"

    def test_parse_op_autonomy(self) -> None:
        assert parse_op_autonomy(None) == {}
        assert parse_op_autonomy("") == {}
        assert parse_op_autonomy("update=propose_only, forget=require_review") == {
            "update": "propose_only",
            "forget": "require_review",
        }

    def test_parse_op_autonomy_rejects_malformed(self) -> None:
        for bad in ("update", "teleport=propose_only", "update=yolo", 5):
            with pytest.raises(ValueError):
                parse_op_autonomy(bad)  # type: ignore[arg-type]

    def test_behavior_preserving_constant_only_relaxes_update(self) -> None:
        assert BEHAVIOR_PRESERVING_OP_TIERS == {"update": "auto_with_rollback"}


class TestInferredEdgeGate:
    """The inferred-edge producer routes through the gate (I4.3, option A)."""

    _TASK_PAYLOAD = {
        "taskId": "task-7",
        "summary": "Implemented Memory Checkout.",
        "decision": "Use Memory Checkout as the contract",
        "decision_event_seq": 5,
        "decision_event_hash": "a" * 64,
    }

    async def test_default_auto_applies_inferred_edge_without_gate_event(
        self, tmp_path: Path
    ) -> None:
        """Default policy preserves behavior: the edge is generated, no gate event."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        await fabric.connect()
        try:
            await fabric.append("task.completed", actor="codex", payload=dict(self._TASK_PAYLOAD), session_id="agent-1")
        finally:
            await fabric.close()

        types = [e.type for e in fabric.session_manager.get("agent-1").eventlog.read_all()]
        assert "inference.edge.generated" in types
        assert GATE_EVENT_TYPE not in types

    async def test_tightened_policy_withholds_edge_and_records_gate(self, tmp_path: Path) -> None:
        """update=propose_only withholds the autonomous edge and records the decision."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.settings.evolution_op_autonomy = "update=propose_only"
        await fabric.connect()
        try:
            await fabric.append("task.completed", actor="codex", payload=dict(self._TASK_PAYLOAD), session_id="agent-1")
        finally:
            await fabric.close()

        events = fabric.session_manager.get("agent-1").eventlog.read_all()
        types = [e.type for e in events]
        assert "inference.edge.generated" not in types
        gate_events = [e for e in events if e.type == GATE_EVENT_TYPE]
        assert len(gate_events) == 1
        assert gate_events[0].payload["op"] == "update"
        assert gate_events[0].payload["auto_apply"] is False
        assert gate_events[0].payload["authority_status"] == "non_authoritative"
