"""Fabric-level tests for the opt-in cognitive retrieval profile (2.2-alpha.2).

Covers salience-blended ranking with the attenuation floor and its
authority/pinned exemptions, encoding-specificity cues, the write-time
encoding gate, and proactive interference detection. The plain/default
profile's byte-parity is the strongest regression guard and is asserted
directly against checkouts built with and without the new parameters.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.config import get_settings
from zaxy.core import MemoryFabric, build_memory_checkout
from zaxy.graph import GraphEntity
from zaxy.salience import (
    REINFORCEMENT_EVENT_TYPE,
    build_confirmed_reinforcement_event,
    build_invalidated_reinforcement_event,
)


def _build_fabric(tmp_path: Path) -> MemoryFabric:
    """Real Eventloom + verbatim lane, mocked graph projection lane."""
    with patch("zaxy.core.build_projection_store") as mock_store:
        mock_store.return_value = AsyncMock()
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
    fabric.query_router = MagicMock(query=AsyncMock(return_value=[]))
    fabric._connected = True
    return fabric


def _plain_fabric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryFabric:
    """Explicit plain (local_fast) fabric; the settings default flipped to
    cognitive in 2.1.0, so plain-profile behavior is pinned by name here."""
    monkeypatch.setenv("RETRIEVAL_PROFILE", "local_fast")
    get_settings.cache_clear()
    try:
        return _build_fabric(tmp_path)
    finally:
        get_settings.cache_clear()


def _cognitive_fabric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **env: str,
) -> MemoryFabric:
    monkeypatch.setenv("RETRIEVAL_PROFILE", "cognitive")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    try:
        return _build_fabric(tmp_path)
    finally:
        get_settings.cache_clear()


def _gate_fabric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **env: str,
) -> MemoryFabric:
    monkeypatch.setenv("ENCODING_GATE_ENABLED", "true")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    try:
        return _build_fabric(tmp_path)
    finally:
        get_settings.cache_clear()


async def _append_decision(fabric: MemoryFabric, text: str, *, session_id: str = "agent-1", **extra: Any) -> Any:
    return await fabric.append(
        "decision.made",
        actor="dev",
        payload={"decision": text, **extra},
        session_id=session_id,
    )


async def _confirm(fabric: MemoryFabric, event: Any, *, session_id: str = "agent-1") -> None:
    spec = build_confirmed_reinforcement_event(
        actor="tester",
        session_id=session_id,
        feedback_id="feedback:0001",
        targets=[{"seq": event.seq, "hash": event.hash}],
    )
    await fabric.append(
        spec["event_type"], actor=spec["actor"], payload=spec["payload"], session_id=session_id
    )


async def _invalidate_below_floor(
    fabric: MemoryFabric, event: Any, *, session_id: str = "agent-1"
) -> None:
    """Append two invalidations: factor 0.2 * 0.2 = 0.04 < the 0.15 floor."""
    for index in range(2):
        spec = build_invalidated_reinforcement_event(
            actor="tester",
            session_id=session_id,
            invalidation_id=f"invalidate:{event.seq}:{index}",
            targets=[{"seq": event.seq, "hash": event.hash}],
        )
        await fabric.append(
            spec["event_type"], actor=spec["actor"], payload=spec["payload"], session_id=session_id
        )


def _fact_position(checkout: Any, text: str) -> int:
    for index, fact in enumerate(checkout.current_facts):
        if text in str(fact.get("content", "")):
            return index
    raise AssertionError(f"{text!r} not found in current_facts")


def _has_fact(checkout: Any, text: str) -> bool:
    return any(text in str(fact.get("content", "")) for fact in checkout.current_facts)


def _belief_proposals(fabric: MemoryFabric, session_id: str = "agent-1") -> list[Any]:
    log = fabric.session_manager.get(session_id).eventlog
    return [event for event in log.read_all() if event.type == "belief.update.proposed"]


def _reinforcements(fabric: MemoryFabric, session_id: str = "agent-1") -> list[Any]:
    log = fabric.session_manager.get(session_id).eventlog
    return [event for event in log.read_all() if event.type == REINFORCEMENT_EVENT_TYPE]


class TestPlainProfileParity:
    """The plain (local_fast) profile is byte-identical with or without the new knobs."""

    async def test_checkout_payload_identical_with_inert_cognitive_parameters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-cognitive profiles ignore profile/cues/floor parameters entirely."""
        fabric = _plain_fabric(tmp_path, monkeypatch)
        first = await _append_decision(fabric, "Analytics storage decision alpha route")
        await _append_decision(fabric, "Analytics storage decision bravo route")
        await _confirm(fabric, first)

        assembly = await fabric.assemble_context(
            "analytics storage decision", session_id="agent-1", limit=5
        )
        now = datetime.now(UTC)
        baseline = build_memory_checkout(
            query="analytics storage decision", assembly=assembly, now=now
        )
        parameterized = build_memory_checkout(
            query="analytics storage decision",
            assembly=assembly,
            now=now,
            retrieval_profile=fabric.retrieval_profile,
            cues={"tool": "pytest", "workspace": "/repo"},
            salience_floor=0.15,
            salience_half_life_days=30.0,
        )

        assert fabric.retrieval_profile.salience_ranking is False
        assert parameterized.to_dict() == baseline.to_dict()
        assert "attenuation" not in baseline.diagnostics

    async def test_plain_fabric_checkout_ignores_cues_argument(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _plain_fabric(tmp_path, monkeypatch)
        await _append_decision(fabric, "Analytics storage decision alpha route")

        without_cues = await fabric.checkout_memory(
            "analytics storage decision",
            session_id="agent-1",
            record_reinforcement=False,
        )
        with_cues = await fabric.checkout_memory(
            "analytics storage decision",
            session_id="agent-1",
            record_reinforcement=False,
            cues={"tool": "pytest"},
        )

        assert with_cues.current_facts == without_cues.current_facts
        assert with_cues.evidence == without_cues.evidence
        assert with_cues.provenance == without_cues.provenance


class TestSalienceBlendedRanking:
    """Cognitive ranking multiplies relevance by normalized salience."""

    async def test_reinforced_memory_outranks_equally_relevant_unreinforced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _cognitive_fabric(tmp_path, monkeypatch)
        await _append_decision(fabric, "Analytics storage decision alpha route")
        reinforced = await _append_decision(fabric, "Analytics storage decision bravo route")
        await _confirm(fabric, reinforced)

        assembly = await fabric.assemble_context(
            "analytics storage decision", session_id="agent-1", limit=5
        )
        now = datetime.now(UTC)
        plain = build_memory_checkout(
            query="analytics storage decision", assembly=assembly, now=now
        )
        cognitive = build_memory_checkout(
            query="analytics storage decision",
            assembly=assembly,
            now=now,
            retrieval_profile=fabric.retrieval_profile,
            salience_floor=fabric._salience_floor,
            salience_half_life_days=fabric._salience_half_life_days,
        )

        # Plain ranking ties on relevance: the stable sort keeps append order.
        assert _fact_position(plain, "alpha route") < _fact_position(plain, "bravo route")
        # Cognitive ranking lifts the reinforced memory above its equal peer.
        assert _fact_position(cognitive, "bravo route") < _fact_position(cognitive, "alpha route")

    async def test_below_floor_memory_attenuated_but_reachable_explicitly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _cognitive_fabric(tmp_path, monkeypatch)
        attenuated = await _append_decision(
            fabric, "Legacy cache invalidation strategy for storage"
        )
        await _append_decision(fabric, "Current cache invalidation strategy for storage")
        await _invalidate_below_floor(fabric, attenuated)

        checkout = await fabric.checkout_memory(
            "cache invalidation strategy",
            session_id="agent-1",
            record_reinforcement=False,
        )

        assert not _has_fact(checkout, "Legacy cache invalidation")
        assert _has_fact(checkout, "Current cache invalidation")
        attenuation = checkout.diagnostics["attenuation"]
        assert attenuation["floor"] == pytest.approx(0.15)
        excluded = attenuation["excluded"]
        assert attenuation["excluded_count"] == len(excluded) == 1
        assert excluded[0]["seq"] == attenuated.seq
        assert excluded[0]["label"] == "attenuated"
        assert excluded[0]["salience_score"] < 0.15

        # Explicit memory_query never routes through the attenuation blend.
        contexts = await fabric.query("cache invalidation strategy", session_id="agent-1")
        assert any("Legacy cache invalidation" in context.content for context in contexts)
        # memory_replay reaches the raw event unconditionally.
        replay = await fabric.replay(session_id="agent-1")
        assert any(event.seq == attenuated.seq for event in replay.events)

    async def test_authority_and_pinned_memories_survive_below_floor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _cognitive_fabric(tmp_path, monkeypatch)
        pinned = await _append_decision(
            fabric, "Pinned rollback runbook for storage incidents", pinned=True
        )
        accepted = await _append_decision(
            fabric,
            "Accepted storage incident escalation policy",
            review_status="accepted",
        )
        await _invalidate_below_floor(fabric, pinned)
        await _invalidate_below_floor(fabric, accepted)

        checkout = await fabric.checkout_memory(
            "storage incident policy runbook",
            session_id="agent-1",
            record_reinforcement=False,
        )

        assert _has_fact(checkout, "Pinned rollback runbook")
        assert _has_fact(checkout, "Accepted storage incident escalation")
        attenuation = checkout.diagnostics["attenuation"]
        assert attenuation["excluded_count"] == 0
        reasons = {item["seq"]: item["exempt_reason"] for item in attenuation["exempt"]}
        assert reasons[pinned.seq] == "pinned"
        assert reasons[accepted.seq] == "authority"


class TestCues:
    """Encoding-specificity cues: preserved on append, blended at checkout."""

    async def test_append_preserves_cue_records_through_replay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _plain_fabric(tmp_path, monkeypatch)
        cues = {"tool": "pytest", "workspace": "/repo", "mission": "ship 2.2"}
        event = await _append_decision(fabric, "Cue carrying decision", cues=cues)

        replay = await fabric.replay(session_id="agent-1")
        replayed = next(item for item in replay.events if item.seq == event.seq)

        assert replayed.payload["cues"] == cues

    async def test_matching_cues_rank_cue_matched_memory_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _cognitive_fabric(tmp_path, monkeypatch)
        await _append_decision(
            fabric,
            "Analytics storage decision alpha route",
            cues={"tool": "pytest", "workspace": "/repo"},
        )
        await _append_decision(fabric, "Analytics storage decision bravo route")

        matched = await fabric.checkout_memory(
            "analytics storage decision",
            session_id="agent-1",
            record_reinforcement=False,
            cues={"tool": "pytest", "workspace": "/repo"},
        )
        unmatched = await fabric.checkout_memory(
            "analytics storage decision",
            session_id="agent-1",
            record_reinforcement=False,
        )

        # With matching query cues the cue-carrying memory wins...
        assert _fact_position(matched, "alpha route") < _fact_position(matched, "bravo route")
        # ...and with no cues the baseline ordering (bravo first: its shorter
        # cue-free payload scores higher lexically) is untouched.
        assert _fact_position(unmatched, "bravo route") < _fact_position(unmatched, "alpha route")


class TestEncodingGate:
    """Write-time gate: tag-only, always-appended, reversible by replay."""

    async def test_gate_disabled_leaves_payloads_untagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _plain_fabric(tmp_path, monkeypatch)
        event = await _append_decision(fabric, "Adopt rust for the parser rewrite")
        duplicate = await _append_decision(fabric, "Adopt rust for the parser rewrite")

        assert "encoding" not in event.payload
        assert "encoding" not in duplicate.payload
        assert _reinforcements(fabric) == []

    async def test_gate_classifies_novel_redundant_and_reinforcing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _gate_fabric(tmp_path, monkeypatch)
        novel = await _append_decision(fabric, "Adopt rust for the parser rewrite")
        redundant = await _append_decision(fabric, "Adopt rust for the parser rewrite")
        reinforcing = await _append_decision(fabric, "Adopt rust for the parser rewrite tooling")

        assert novel.payload["encoding"]["classification"] == "novel"
        assert redundant.payload["encoding"]["classification"] == "redundant"
        assert redundant.payload["encoding"]["duplicate_of"].startswith(
            f"eventloom://agent-1/events/{novel.seq}#"
        )
        assert reinforcing.payload["encoding"]["classification"] == "reinforcing"

    async def test_redundant_append_emits_weak_reinforcement_toward_duplicate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _gate_fabric(tmp_path, monkeypatch)
        original = await _append_decision(fabric, "Adopt rust for the parser rewrite")
        await _append_decision(fabric, "Adopt rust for the parser rewrite")

        reinforcements = _reinforcements(fabric)
        assert len(reinforcements) == 1
        payload = reinforcements[0].payload
        assert payload["kind"] == "surfaced"
        assert payload["targets"] == [{"seq": original.seq, "hash": original.hash}]
        assert "encoding_gate" in payload["source"]

    async def test_gate_tags_are_inert_for_ranking_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-projecting the same log with the gate off yields identical ranking."""
        gate_on = _gate_fabric(tmp_path, monkeypatch)
        await _append_decision(gate_on, "Adopt rust for the parser rewrite")
        await _append_decision(gate_on, "Adopt rust for the parser rewrite")
        await _append_decision(gate_on, "Keep python for orchestration glue")

        monkeypatch.setenv("ENCODING_GATE_ENABLED", "false")
        get_settings.cache_clear()
        gate_off = _build_fabric(tmp_path)
        get_settings.cache_clear()

        tagged = await gate_on.checkout_memory(
            "parser rewrite language decision",
            session_id="agent-1",
            record_reinforcement=False,
        )
        untagged = await gate_off.checkout_memory(
            "parser rewrite language decision",
            session_id="agent-1",
            record_reinforcement=False,
        )

        assert untagged.current_facts == tagged.current_facts
        assert untagged.evidence == tagged.evidence


class TestInterferenceDetection:
    """Novel-and-contradicting appends propose review-gated belief updates."""

    @staticmethod
    def _existing_task_entity(seed: Any) -> GraphEntity:
        return GraphEntity(
            name="task-9001",
            entity_type="task",
            valid_from="2026-06-01T00:00:00Z",
            valid_to=None,
            properties={
                "status": "verified",
                "source_event_seq": seed.seq,
                "source_event_hash": seed.hash,
            },
            session_id="agent-1",
        )

    async def _seed_conflict_fixture(
        self, fabric: MemoryFabric
    ) -> tuple[Any, Any]:
        seed = await fabric.append(
            "task.completed",
            actor="dev",
            payload={
                "task": "task-9001",
                "status": "verified",
                "summary": "Deployment pipeline verification for service alpha",
            },
            session_id="agent-1",
        )
        fabric.graph.search_exact = AsyncMock(return_value=[self._existing_task_entity(seed)])
        conflicting = await fabric.append(
            "task.completed",
            actor="dev",
            payload={
                "task": "task-9001",
                "status": "blocked",
                "summary": "Completely different remediation context after outage",
            },
            session_id="agent-1",
        )
        return seed, conflicting

    async def test_seeded_contradiction_emits_one_proposal_citing_both_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _gate_fabric(tmp_path, monkeypatch)
        seed, conflicting = await self._seed_conflict_fixture(fabric)

        proposals = _belief_proposals(fabric)
        assert len(proposals) == 1
        payload = proposals[0].payload
        assert payload["review_status"] == "pending"
        assert payload["authority_status"] == "non_authoritative"
        assert payload["source_events"] == [
            {"seq": seed.seq, "hash": seed.hash},
            {"seq": conflicting.seq, "hash": conflicting.hash},
        ]
        assert "status" in payload["claim"]
        assert conflicting.payload["encoding"]["classification"] == "novel"

    async def test_interference_runs_under_cognitive_profile_without_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _cognitive_fabric(tmp_path, monkeypatch)
        seed, conflicting = await self._seed_conflict_fixture(fabric)

        proposals = _belief_proposals(fabric)
        assert len(proposals) == 1
        assert proposals[0].payload["source_events"] == [
            {"seq": seed.seq, "hash": seed.hash},
            {"seq": conflicting.seq, "hash": conflicting.hash},
        ]
        # Gate off: classification ran for interference but payloads stay untagged.
        assert "encoding" not in conflicting.payload

    async def test_non_contradicting_novel_append_emits_no_proposal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _gate_fabric(tmp_path, monkeypatch)
        await _append_decision(fabric, "Adopt rust for the parser rewrite")
        fabric.graph.search_exact = AsyncMock(return_value=[])
        await _append_decision(fabric, "Schedule quarterly dependency audit")

        assert _belief_proposals(fabric) == []

    async def test_proposal_failure_never_fails_the_append(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fabric = _gate_fabric(tmp_path, monkeypatch)
        seed = await fabric.append(
            "task.completed",
            actor="dev",
            payload={
                "task": "task-9001",
                "status": "verified",
                "summary": "Deployment pipeline verification for service alpha",
            },
            session_id="agent-1",
        )
        fabric.graph.search_exact = AsyncMock(return_value=[self._existing_task_entity(seed)])
        monkeypatch.setattr(
            fabric,
            "propose_belief_update",
            AsyncMock(side_effect=RuntimeError("proposal path offline")),
        )

        conflicting = await fabric.append(
            "task.completed",
            actor="dev",
            payload={
                "task": "task-9001",
                "status": "blocked",
                "summary": "Completely different remediation context after outage",
            },
            session_id="agent-1",
        )

        assert conflicting.seq > seed.seq
        assert _belief_proposals(fabric) == []


class TestGraphWalkWiring:
    """The cognitive profile arms the router's graph-walk stage."""

    async def test_cognitive_profile_enables_router_graph_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RETRIEVAL_PROFILE", "cognitive")
        get_settings.cache_clear()
        try:
            with patch("zaxy.core.build_projection_store") as mock_store:
                mock_store.return_value = AsyncMock()
                fabric = MemoryFabric(
                    eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True
                )
        finally:
            get_settings.cache_clear()

        assert fabric.retrieval_profile.graph_walk is True
        assert fabric.query_router.graph_walk_enabled is True

    async def test_default_profile_arms_router_graph_walk(self, tmp_path: Path) -> None:
        """2.1.0 default flip: an unconfigured fabric runs the cognitive profile."""
        with patch("zaxy.core.build_projection_store") as mock_store:
            mock_store.return_value = AsyncMock()
            fabric = MemoryFabric(
                eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True
            )

        assert fabric.retrieval_profile.name == "cognitive"
        assert fabric.retrieval_profile.graph_walk is True
        assert fabric.query_router.graph_walk_enabled is True

    async def test_local_fast_profile_keeps_router_graph_walk_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The explicit local_fast opt-out keeps the plain (pre-2.1.0) router wiring."""
        monkeypatch.setenv("RETRIEVAL_PROFILE", "local_fast")
        get_settings.cache_clear()
        try:
            with patch("zaxy.core.build_projection_store") as mock_store:
                mock_store.return_value = AsyncMock()
                fabric = MemoryFabric(
                    eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True
                )
        finally:
            get_settings.cache_clear()

        assert fabric.retrieval_profile.graph_walk is False
        assert fabric.query_router.graph_walk_enabled is False
