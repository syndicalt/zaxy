"""Tests for transparency & controlled editability (Zaxy 3 / I5a).

Covers the edit -> re-ingest round-trip (``memory.corrected``) and the explicit
rollback of an evolution (``memory.rolled_back``): both additive, gated, cited,
non-authoritative, and -- crucially -- never mutating a sealed event, so
``EventLog.verify()`` stays green and a rolled-back consolidation acceptance is
reversed purely in replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from zaxy.consolidation import (
    build_consolidation_candidate_event,
    build_consolidation_review_event,
)
from zaxy.core.fabric import MemoryFabric
from zaxy.editable import (
    MEMORY_CORRECTED_EVENT_TYPE,
    MEMORY_ROLLBACK_EVENT_TYPE,
    build_memory_correction_event,
    build_memory_rollback_event,
    parse_editable,
    render_editable,
)
from zaxy.mcp_server import ZaxyMCPServer

_HASH = "a" * 64
_REF = {"seq": 3, "hash": _HASH}
_CANDIDATE_ID = "consolidation:claim:" + "0" * 24


# ---------------------------------------------------------------------------
# Pure contract + round-trip tests
# ---------------------------------------------------------------------------


class TestContracts:
    def test_build_correction_event_is_cited_non_authoritative(self) -> None:
        spec = build_memory_correction_event(
            actor="human",
            session_id="ext",
            target=dict(_REF),
            new_content="fixed content",
            reason="typo",
        )
        assert spec["event_type"] == MEMORY_CORRECTED_EVENT_TYPE
        assert spec["thread"] == "ext"
        payload = spec["payload"]
        assert payload["authority_status"] == "non_authoritative"
        assert payload["target"] == _REF
        assert payload["content"] == "fixed content"
        assert payload["reason"] == "typo"
        assert payload["correction_id"].startswith("correction:")

    def test_correction_id_is_deterministic_in_target_and_content(self) -> None:
        first = build_memory_correction_event(
            actor="h", session_id="ext", target=dict(_REF), new_content="c", reason="r"
        )
        # Same target/content/reason -> same id regardless of actor.
        second = build_memory_correction_event(
            actor="other", session_id="ext", target=dict(_REF), new_content="c", reason="r"
        )
        assert first["payload"]["correction_id"] == second["payload"]["correction_id"]
        # Different content -> different id.
        third = build_memory_correction_event(
            actor="h", session_id="ext", target=dict(_REF), new_content="different", reason="r"
        )
        assert third["payload"]["correction_id"] != first["payload"]["correction_id"]

    def test_build_rollback_event_is_cited_non_authoritative(self) -> None:
        spec = build_memory_rollback_event(
            actor="human",
            session_id="ext",
            target=dict(_REF),
            reason="undo",
            reverts={
                "event_type": "consolidation.candidate.reviewed",
                "candidate_id": _CANDIDATE_ID,
                "to_status": "pending",
            },
        )
        assert spec["event_type"] == MEMORY_ROLLBACK_EVENT_TYPE
        payload = spec["payload"]
        assert payload["authority_status"] == "non_authoritative"
        assert payload["target"] == _REF
        assert payload["rollback_id"].startswith("rollback:")
        assert payload["reverts"]["candidate_id"] == _CANDIDATE_ID
        assert payload["reverts"]["to_status"] == "pending"

    def test_builders_reject_bad_target(self) -> None:
        with pytest.raises(ValueError):
            build_memory_correction_event(
                actor="h", session_id="ext", target={"seq": 1, "hash": "xyz"},
                new_content="c", reason="r",
            )
        with pytest.raises(ValueError):
            build_memory_rollback_event(
                actor="h", session_id="ext", target={"seq": 0, "hash": _HASH}, reason="r"
            )

    def test_builders_reject_empty_fields(self) -> None:
        with pytest.raises(ValueError):
            build_memory_correction_event(
                actor="h", session_id="ext", target=dict(_REF), new_content="  ", reason="r"
            )
        with pytest.raises(ValueError):
            build_memory_rollback_event(
                actor="h", session_id="ext", target=dict(_REF), reason=""
            )


class TestEditableRoundTrip:
    def test_render_parse_round_trip_preserves_target_and_edit(self) -> None:
        memory = {
            "seq": 7,
            "hash": _HASH,
            "content": "original text",
            "session_id": "ext",
            "entity_name": "Foo",
        }
        block = render_editable(memory)
        assert "seq: 7" in block
        assert _HASH in block
        assert "original text" in block

        # A human edits the body in place; everything else round-trips.
        edited = block.replace("original text", "corrected text")
        parsed = parse_editable(edited)
        assert parsed["target"] == {"seq": 7, "hash": _HASH}
        assert parsed["content"] == "corrected text"
        assert parsed["session_id"] == "ext"
        assert parsed["entity_name"] == "Foo"

        # The parsed edit produces a valid correction spec.
        spec = build_memory_correction_event(
            actor="human",
            session_id=parsed["session_id"],
            target=parsed["target"],
            new_content=parsed["content"],
            reason="manual edit",
        )
        assert spec["payload"]["content"] == "corrected text"
        assert spec["payload"]["target"] == {"seq": 7, "hash": _HASH}

    def test_render_falls_back_to_summary(self) -> None:
        block = render_editable({"seq": 1, "hash": _HASH, "summary": "summ"})
        assert "summ" in block
        assert parse_editable(block)["content"] == "summ"

    def test_multiline_body_is_preserved(self) -> None:
        memory = {"seq": 2, "hash": _HASH, "content": "line1\nline2\nline3"}
        parsed = parse_editable(render_editable(memory))
        assert parsed["content"] == "line1\nline2\nline3"

    def test_parse_rejects_malformed_blocks(self) -> None:
        with pytest.raises(ValueError):
            parse_editable("no header here")
        with pytest.raises(ValueError):  # empty body
            parse_editable(f"--- zaxy:editable v1 ---\nseq: 1\nhash: {_HASH}\n---\n")
        with pytest.raises(ValueError):  # non-integer seq
            parse_editable(f"--- zaxy:editable v1 ---\nseq: notint\nhash: {_HASH}\n---\nbody")
        with pytest.raises(ValueError):  # short hash
            parse_editable("--- zaxy:editable v1 ---\nseq: 1\nhash: short\n---\nbody")


class TestEditableEdgeCases:
    """Validator + render/parse edge branches (build/render/parse reject paths)."""

    def test_correction_event_carries_optional_original_and_entity(self) -> None:
        spec = build_memory_correction_event(
            actor="h", session_id="ext", target=dict(_REF),
            new_content="new", reason="r",
            original_content="was here", entity_name="Foo",
        )
        payload = spec["payload"]
        assert payload["original_content"] == "was here"
        assert payload["entity_name"] == "Foo"

    def test_correction_event_rejects_empty_optional_fields(self) -> None:
        with pytest.raises(ValueError, match="original_content"):
            build_memory_correction_event(
                actor="h", session_id="ext", target=dict(_REF),
                new_content="new", reason="r", original_content="   ",
            )
        with pytest.raises(ValueError, match="entity_name"):
            build_memory_correction_event(
                actor="h", session_id="ext", target=dict(_REF),
                new_content="new", reason="r", entity_name="",
            )

    def test_builders_reject_missing_actor_and_session(self) -> None:
        with pytest.raises(ValueError, match="actor"):
            build_memory_correction_event(
                actor="", session_id="ext", target=dict(_REF), new_content="c", reason="r"
            )
        with pytest.raises(ValueError, match="session_id"):
            build_memory_rollback_event(
                actor="h", session_id="  ", target=dict(_REF), reason="r"
            )

    def test_builders_reject_non_mapping_target_and_reverts(self) -> None:
        with pytest.raises(ValueError, match="event ref must be a mapping"):
            build_memory_rollback_event(
                actor="h", session_id="ext", target=["not", "a", "mapping"], reason="r",
            )
        with pytest.raises(ValueError, match="reverts must be a mapping"):
            build_memory_rollback_event(
                actor="h", session_id="ext", target=dict(_REF), reason="r", reverts=123,
            )

    def test_render_rejects_non_mapping(self) -> None:
        with pytest.raises(ValueError, match="memory must be a mapping"):
            render_editable("not a mapping")

    def test_render_falls_back_to_text_field(self) -> None:
        block = render_editable({"seq": 4, "hash": _HASH, "text": "from text"})
        assert "from text" in block
        assert parse_editable(block)["content"] == "from text"

    def test_render_rejects_missing_or_blank_content(self) -> None:
        with pytest.raises(ValueError, match="non-empty content"):
            render_editable({"seq": 4, "hash": _HASH})
        with pytest.raises(ValueError, match="non-empty content"):
            render_editable({"seq": 4, "hash": _HASH, "content": "   "})

    def test_parse_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            parse_editable("   ")

    def test_parse_skips_leading_blank_lines(self) -> None:
        block = f"\n\n--- zaxy:editable v1 ---\nseq: 5\nhash: {_HASH}\n---\nbody"
        parsed = parse_editable(block)
        assert parsed["target"] == {"seq": 5, "hash": _HASH}
        assert parsed["content"] == "body"

    def test_parse_skips_blank_header_lines(self) -> None:
        block = f"--- zaxy:editable v1 ---\nseq: 5\n\nhash: {_HASH}\n---\nbody"
        assert parse_editable(block)["target"]["seq"] == 5

    def test_parse_rejects_header_without_colon(self) -> None:
        block = f"--- zaxy:editable v1 ---\nseq: 5\nnocolon\nhash: {_HASH}\n---\nbody"
        with pytest.raises(ValueError, match="key: value"):
            parse_editable(block)

    def test_parse_rejects_unknown_header_key(self) -> None:
        block = f"--- zaxy:editable v1 ---\nseq: 5\nbogus: x\nhash: {_HASH}\n---\nbody"
        with pytest.raises(ValueError, match="unknown editable header key"):
            parse_editable(block)

    def test_parse_rejects_missing_closing_delimiter(self) -> None:
        block = f"--- zaxy:editable v1 ---\nseq: 5\nhash: {_HASH}"
        with pytest.raises(ValueError, match="closing"):
            parse_editable(block)

    def test_parse_rejects_missing_seq_header(self) -> None:
        block = f"--- zaxy:editable v1 ---\nhash: {_HASH}\n---\nbody"
        with pytest.raises(ValueError, match="'seq' is required"):
            parse_editable(block)



# ---------------------------------------------------------------------------
# Fabric integration: edit_memory
# ---------------------------------------------------------------------------


def _fabric(tmp_path: Path) -> MemoryFabric:
    return MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)


class TestEditMemory:
    async def test_edit_appends_cited_correction_and_gate_without_mutation(
        self, tmp_path: Path
    ) -> None:
        fabric = _fabric(tmp_path)
        await fabric.connect()
        try:
            target = await fabric.append(
                "goal.created", actor="user", payload={"title": "Ship I5a"}, session_id="ext"
            )
            eventlog = fabric.session_manager.get("ext").eventlog
            before = {
                e.seq: (e.type, e.hash, dict(e.payload)) for e in eventlog.read_all()
            }
            assert eventlog.verify().ok is True

            result = await fabric.edit_memory(
                target_seq=target.seq,
                target_hash=target.hash,
                new_content="corrected goal title",
                reason="title was wrong",
                session_id="ext",
            )

            events = eventlog.read_all()
            types = [e.type for e in events]
            assert "evolution.gate.evaluated" in types
            assert MEMORY_CORRECTED_EVENT_TYPE in types

            corrected = [e for e in events if e.type == MEMORY_CORRECTED_EVENT_TYPE]
            assert len(corrected) == 1
            payload = corrected[0].payload
            assert payload["authority_status"] == "non_authoritative"
            assert payload["target"] == {"seq": target.seq, "hash": target.hash}
            assert payload["content"] == "corrected goal title"
            assert result["correction_event"]["seq"] == corrected[0].seq
            assert result["gate"]["op"] == "update"
            assert result["correction_id"] == payload["correction_id"]

            # The original event is byte-for-byte unchanged; the chain stays intact.
            after_target = next(e for e in events if e.seq == target.seq)
            assert (after_target.type, after_target.hash, dict(after_target.payload)) == before[
                target.seq
            ]
            assert eventlog.verify().ok is True
        finally:
            await fabric.close()

    async def test_edit_rejects_unknown_or_mismatched_target(self, tmp_path: Path) -> None:
        fabric = _fabric(tmp_path)
        await fabric.connect()
        try:
            target = await fabric.append(
                "goal.created", actor="user", payload={"title": "t"}, session_id="ext"
            )
            with pytest.raises(ValueError):
                await fabric.edit_memory(
                    target_seq=999, target_hash=_HASH, new_content="x", reason="r", session_id="ext"
                )
            with pytest.raises(ValueError):
                await fabric.edit_memory(
                    target_seq=target.seq, target_hash="b" * 64,
                    new_content="x", reason="r", session_id="ext",
                )
        finally:
            await fabric.close()

    async def test_edit_round_trip_via_render_parse(self, tmp_path: Path) -> None:
        fabric = _fabric(tmp_path)
        await fabric.connect()
        try:
            target = await fabric.append(
                "context.policy", actor="user",
                payload={"source": "team", "content": "deploy on Fridays"}, session_id="ext",
            )
            block = render_editable(
                {"seq": target.seq, "hash": target.hash, "content": "deploy on Fridays"}
            )
            parsed = parse_editable(block.replace("Fridays", "Mondays"))
            result = await fabric.edit_memory(
                target_seq=parsed["target"]["seq"],
                target_hash=parsed["target"]["hash"],
                new_content=parsed["content"],
                reason="policy changed",
                session_id="ext",
            )
            events = fabric.session_manager.get("ext").eventlog.read_all()
            corrected = next(e for e in events if e.type == MEMORY_CORRECTED_EVENT_TYPE)
            assert "Mondays" in corrected.payload["content"]
            assert result["target"] == {"seq": target.seq, "hash": target.hash}
        finally:
            await fabric.close()


# ---------------------------------------------------------------------------
# Fabric integration: rollback_memory (the reversal-in-replay keystone)
# ---------------------------------------------------------------------------


async def _accept_candidate(
    fabric: MemoryFabric, *, session_id: str = "ext", status: str = "accepted"
) -> tuple[str, object]:
    """Seed a consolidation candidate and a review with ``status``; return (id, review)."""
    src = await fabric.append(
        "transcript.turn", actor="u",
        payload={"source": "chat", "content": "a durable fact"}, session_id=session_id,
    )
    cand_spec = build_consolidation_candidate_event(
        actor="zaxy-consolidation",
        session_id=session_id,
        candidate_type="claim",
        title="A durable claim",
        summary="the claim summary",
        source_events=[{"seq": src.seq, "hash": src.hash}],
        confidence=0.9,
        method="test",
    )
    await fabric.append(
        cand_spec["event_type"], cand_spec["actor"],
        payload=cand_spec["payload"], session_id=session_id,
    )
    candidate_id = cand_spec["payload"]["candidate_id"]
    review_spec = build_consolidation_review_event(
        actor="reviewer", session_id=session_id, candidate_id=candidate_id,
        status=status, rationale="reviewed",
    )
    review = await fabric.append(
        review_spec["event_type"], review_spec["actor"],
        payload=review_spec["payload"], session_id=session_id,
    )
    return candidate_id, review


class TestRollbackMemory:
    async def test_rollback_reverses_consolidation_acceptance_in_replay(
        self, tmp_path: Path
    ) -> None:
        fabric = _fabric(tmp_path)
        await fabric.connect()
        try:
            candidate_id, review = await _accept_candidate(fabric)
            eventlog = fabric.session_manager.get("ext").eventlog

            # Before rollback: the candidate is accepted in replay.
            status_before = await fabric.consolidation_status(session_id="ext")
            assert status_before["accepted_count"] == 1
            assert status_before["pending_count"] == 0
            assert status_before["candidates"][0]["review_status"] == "accepted"

            review_payload_before = dict(review.payload)
            assert eventlog.verify().ok is True

            result = await fabric.rollback_memory(
                target_seq=review.seq,
                target_hash=review.hash,
                reason="acceptance was premature",
                session_id="ext",
            )
            assert result["reverts"]["event_type"] == "consolidation.candidate.reviewed"
            assert result["reverts"]["candidate_id"] == candidate_id
            assert result["reverts"]["to_status"] == "pending"
            assert result["gate"]["op"] == "update"

            # After rollback: the acceptance is REVERSED in replay -> back to pending.
            status_after = await fabric.consolidation_status(session_id="ext")
            assert status_after["accepted_count"] == 0
            assert status_after["pending_count"] == 1
            assert status_after["rollback_count"] == 1
            candidate_after = status_after["candidates"][0]
            assert candidate_after["review_status"] == "pending"
            assert candidate_after["rolled_back_review_count"] == 1

            events = eventlog.read_all()
            rolled = [e for e in events if e.type == MEMORY_ROLLBACK_EVENT_TYPE]
            assert len(rolled) == 1
            assert rolled[0].payload["authority_status"] == "non_authoritative"
            assert rolled[0].payload["target"] == {"seq": review.seq, "hash": review.hash}
            assert any(e.type == "evolution.gate.evaluated" for e in events)

            # The accepted review event is NEVER mutated; the hash chain stays intact.
            after_review = next(e for e in events if e.seq == review.seq)
            assert dict(after_review.payload) == review_payload_before
            assert after_review.payload["status"] == "accepted"
            assert eventlog.verify().ok is True
        finally:
            await fabric.close()

    async def test_rollback_reverts_to_prior_review_not_just_pending(
        self, tmp_path: Path
    ) -> None:
        fabric = _fabric(tmp_path)
        await fabric.connect()
        try:
            # deferred, then accepted: rolling back the acceptance reverts to deferred.
            candidate_id, _ = await _accept_candidate(fabric, status="deferred")
            accept_spec = build_consolidation_review_event(
                actor="reviewer", session_id="ext", candidate_id=candidate_id,
                status="accepted", rationale="now accepting",
            )
            accept = await fabric.append(
                accept_spec["event_type"], accept_spec["actor"],
                payload=accept_spec["payload"], session_id="ext",
            )
            assert (await fabric.consolidation_status(session_id="ext"))["accepted_count"] == 1

            result = await fabric.rollback_memory(
                target_seq=accept.seq, target_hash=accept.hash,
                reason="revert to deferred", session_id="ext",
            )
            assert result["reverts"]["to_status"] == "deferred"

            status_after = await fabric.consolidation_status(session_id="ext")
            assert status_after["accepted_count"] == 0
            assert status_after["deferred_count"] == 1
            assert status_after["candidates"][0]["review_status"] == "deferred"
            assert fabric.session_manager.get("ext").eventlog.verify().ok is True
        finally:
            await fabric.close()

    async def test_rollback_rejects_non_latest_consolidation_review(
        self, tmp_path: Path
    ) -> None:
        fabric = _fabric(tmp_path)
        await fabric.connect()
        try:
            # deferred (seq N) then accepted (seq N+1): the deferred review is
            # historically superseded. Rolling it back would project a stale
            # review status onto the graph entity (the projection reverts to the
            # pre-target status, ignoring the surviving acceptance) while the
            # authoritative replay stays 'accepted' -- a divergence we reject.
            candidate_id, deferred_review = await _accept_candidate(fabric, status="deferred")
            accept_spec = build_consolidation_review_event(
                actor="reviewer", session_id="ext", candidate_id=candidate_id,
                status="accepted", rationale="now accepting",
            )
            accept = await fabric.append(
                accept_spec["event_type"], accept_spec["actor"],
                payload=accept_spec["payload"], session_id="ext",
            )
            assert accept.seq > deferred_review.seq
            eventlog = fabric.session_manager.get("ext").eventlog
            events_before = eventlog.read_all()
            gate_before = sum(1 for e in events_before if e.type == "evolution.gate.evaluated")

            # Rolling back the NON-LATEST (deferred) review is rejected outright.
            with pytest.raises(ValueError, match="superseded consolidation review"):
                await fabric.rollback_memory(
                    target_seq=deferred_review.seq, target_hash=deferred_review.hash,
                    reason="stale projection attempt", session_id="ext",
                )

            # No rolled_back marker and no gate event were appended for the rejection.
            events_after_reject = eventlog.read_all()
            assert len(events_after_reject) == len(events_before)
            assert not any(e.type == MEMORY_ROLLBACK_EVENT_TYPE for e in events_after_reject)
            assert (
                sum(1 for e in events_after_reject if e.type == "evolution.gate.evaluated")
                == gate_before
            )
            # Authoritative replay is untouched: still accepted.
            assert (await fabric.consolidation_status(session_id="ext"))["accepted_count"] == 1

            # Rolling back the LATEST (accepted) review still works -> reverts to deferred.
            result = await fabric.rollback_memory(
                target_seq=accept.seq, target_hash=accept.hash,
                reason="revert to deferred", session_id="ext",
            )
            assert result["reverts"]["to_status"] == "deferred"
            status_after = await fabric.consolidation_status(session_id="ext")
            assert status_after["accepted_count"] == 0
            assert status_after["deferred_count"] == 1
            assert status_after["candidates"][0]["review_status"] == "deferred"
            assert eventlog.verify().ok is True
        finally:
            await fabric.close()

    async def test_rollback_rejects_non_reversible_target(self, tmp_path: Path) -> None:
        fabric = _fabric(tmp_path)
        await fabric.connect()
        try:
            event = await fabric.append(
                "goal.created", actor="user", payload={"title": "not reversible"}, session_id="ext"
            )
            with pytest.raises(ValueError, match="not a reversible evolution"):
                await fabric.rollback_memory(
                    target_seq=event.seq, target_hash=event.hash, reason="nope", session_id="ext"
                )
        finally:
            await fabric.close()


# ---------------------------------------------------------------------------
# MCP tools route through the fabric methods
# ---------------------------------------------------------------------------


def _server(tmp_path: Path) -> ZaxyMCPServer:
    server = ZaxyMCPServer(
        eventloom_path=str(tmp_path / ".eventloom"), default_session_id="agent-1"
    )
    server.graph = AsyncMock()
    server.tracer = AsyncMock()
    return server


class TestMcpRouting:
    async def test_memory_edit_tool_routes_through_fabric(self, tmp_path: Path) -> None:
        server = _server(tmp_path)
        eventlog = server.session_manager.get("agent-1").eventlog
        target = eventlog.append(
            "goal.created", actor="user", payload={"title": "edit me"}, thread="agent-1"
        )
        response = await server.handle_memory_edit(
            {"target_seq": target.seq, "target_hash": target.hash,
             "new_content": "fixed", "reason": "typo"}
        )
        payload = json.loads(response[0].text)
        assert payload["correction_event"]["seq"] > target.seq
        assert payload["gate"]["op"] == "update"

        events = eventlog.read_all()
        assert any(e.type == MEMORY_CORRECTED_EVENT_TYPE for e in events)
        assert any(e.type == "evolution.gate.evaluated" for e in events)
        assert eventlog.verify().ok is True

    async def test_memory_rollback_tool_routes_through_fabric(self, tmp_path: Path) -> None:
        server = _server(tmp_path)
        eventlog = server.session_manager.get("agent-1").eventlog
        candidate_id, review = await _accept_candidate(server._fabric, session_id="agent-1")

        response = await server.handle_memory_rollback(
            {"target_seq": review.seq, "target_hash": review.hash, "reason": "undo it"}
        )
        payload = json.loads(response[0].text)
        assert payload["reverts"]["candidate_id"] == candidate_id
        assert payload["reverts"]["to_status"] == "pending"

        events = eventlog.read_all()
        rolled = [e for e in events if e.type == MEMORY_ROLLBACK_EVENT_TYPE]
        assert len(rolled) == 1
        assert rolled[0].payload["target"] == {"seq": review.seq, "hash": review.hash}
        assert eventlog.verify().ok is True

    async def test_memory_edit_tool_rejects_missing_required_args(self, tmp_path: Path) -> None:
        server = _server(tmp_path)
        with pytest.raises(ValueError):
            await server.handle_memory_edit(
                {"target_seq": 1, "target_hash": _HASH, "reason": "r"}
            )
