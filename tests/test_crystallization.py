"""Tests for the governed sleep-time crystallization runner (Zaxy 3 / I2).

These exercise the real embedded :class:`MemoryFabric` (no mocks of the system
under test): a seeded session log is crystallized and the additive, cited,
review-gated, governed, idempotent contract is asserted against the actual
replayed events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from zaxy.cli.runtime import app
from zaxy.crystallization import (
    CRYSTALLIZATION_EVENT_TYPE,
    CrystallizationReport,
    run_crystallization_pass,
)
from zaxy.lifecycle import (
    build_file_edit_applied_event,
    build_tool_call_completed_event,
)
from zaxy.metacognition import build_conflict_cluster_event, build_known_unknown_event
from zaxy.salience import build_reinforcement_event

# A recurring successful tool sequence; nine supporting sessions push the mined
# procedure confidence to the 0.85 cap (0.50 + 0.05 * (9 - 2)), which is exactly
# the default auto_with_rollback threshold, so the mined skill auto-applies while
# the lower-confidence consolidation candidates (<= 0.68) stay pending.
_RECURRING = ["memory_query", "memory_checkout", "memory_feedback"]
_SUPPORT_SESSIONS = 9


def _append_spec(eventlog: Any, spec: dict[str, Any], *, thread: str) -> Any:
    return eventlog.append(
        spec["event_type"],
        actor=spec["actor"],
        payload=spec["payload"],
        thread=thread,
    )


def _seed_recurring(eventlog: Any, thread: str) -> list[Any]:
    return [
        _append_spec(
            eventlog,
            build_tool_call_completed_event(
                tool_name=name,
                status="succeeded",
                session_id=thread,
                result_summary=f"{name} ok",
            ),
            thread=thread,
        )
        for name in _RECURRING
    ]


def _seed_crystallization_inputs(
    fabric: Any,
    session_id: str,
    *,
    support: int = _SUPPORT_SESSIONS,
    reinforce: bool = True,
) -> Any:
    """Seed a session log with consolidation + cross-session procedure inputs."""
    eventlog = fabric.session_manager.get(session_id).eventlog
    main_steps = _seed_recurring(eventlog, session_id)
    _append_spec(
        eventlog,
        build_file_edit_applied_event(
            path="src/zaxy/checkout.py",
            operation="edit",
            session_id=session_id,
            summary="adjust checkout budget",
        ),
        thread=session_id,
    )
    if reinforce:
        first = main_steps[0]
        _append_spec(
            eventlog,
            build_reinforcement_event(
                actor="zaxy",
                session_id=session_id,
                kind="confirmed",
                targets=[{"seq": first.seq, "hash": first.hash}],
                source={"feedback_id": "seed-confirm"},
            ),
            thread=session_id,
        )
    for index in range(support - 1):
        _seed_recurring(eventlog, f"{session_id}-peer-{index}")
    return eventlog


async def _build_fabric(tmp_path: Path) -> Any:
    from zaxy.core import MemoryFabric

    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    await fabric.connect()
    return fabric


def _events_of_type(eventlog: Any, event_type: str) -> list[Any]:
    return [event for event in eventlog.read_all() if event.type == event_type]


@pytest.mark.asyncio
async def test_pass_emits_additive_cited_non_authoritative_candidates_and_summary(
    tmp_path: Path,
) -> None:
    fabric = await _build_fabric(tmp_path)
    try:
        eventlog = _seed_crystallization_inputs(fabric, "crystal")
        report = await run_crystallization_pass(fabric, session_id="crystal")
    finally:
        await fabric.close()

    assert isinstance(report, CrystallizationReport)
    assert report.consolidation_candidates == 3
    assert report.procedure_candidates == 1

    candidates = _events_of_type(eventlog, "consolidation.candidate.created")
    assert len(candidates) == 4
    for candidate in candidates:
        # Additive + non-authoritative + review-pending, stamped by the builders.
        assert candidate.payload["authority_status"] == "non_authoritative"
        assert candidate.payload["review_status"] == "pending"
        # Everything cites: seq + 64-hex hash.
        sources = candidate.payload["source_events"]
        assert sources
        for citation in sources:
            assert isinstance(citation["seq"], int) and citation["seq"] > 0
            assert len(citation["hash"]) == 64
            int(citation["hash"], 16)

    summaries = _events_of_type(eventlog, CRYSTALLIZATION_EVENT_TYPE)
    assert len(summaries) == 1
    summary_payload = summaries[0].payload
    assert summary_payload["authority_status"] == "non_authoritative"
    assert summary_payload["consolidation_candidates"] == 3
    assert summary_payload["procedure_candidates"] == 1
    assert {entry["candidate_id"] for entry in summary_payload["candidates"]} == {
        candidate.payload["candidate_id"] for candidate in candidates
    }

    assert report.latency_ms > 0.0


@pytest.mark.asyncio
async def test_second_pass_is_idempotent_and_adds_no_duplicate_candidates(
    tmp_path: Path,
) -> None:
    fabric = await _build_fabric(tmp_path)
    try:
        eventlog = _seed_crystallization_inputs(fabric, "crystal")
        first = await run_crystallization_pass(fabric, session_id="crystal")
        candidates_after_first = len(_events_of_type(eventlog, "consolidation.candidate.created"))
        second = await run_crystallization_pass(fabric, session_id="crystal")
        candidates_after_second = len(_events_of_type(eventlog, "consolidation.candidate.created"))
    finally:
        await fabric.close()

    assert first.consolidation_candidates == 3
    assert first.procedure_candidates == 1
    assert candidates_after_first == 4
    # Re-running over the same log proposes nothing new.
    assert second.consolidation_candidates == 0
    assert second.procedure_candidates == 0
    assert second.auto_accepted == 0
    assert candidates_after_second == candidates_after_first


@pytest.mark.asyncio
async def test_auto_apply_gating_accepts_high_confidence_under_default_policy(
    tmp_path: Path,
) -> None:
    fabric = await _build_fabric(tmp_path)
    try:
        eventlog = _seed_crystallization_inputs(fabric, "crystal")
        report = await run_crystallization_pass(fabric, session_id="crystal")
    finally:
        await fabric.close()

    # The 0.85 mined procedure clears the default auto_with_rollback threshold;
    # the <=0.68 consolidation candidates do not.
    assert report.auto_accepted == 1
    assert report.left_pending == 3
    assert len(report.gate_decisions) == 4
    assert sum(1 for decision in report.gate_decisions if decision["auto_apply"]) == 1
    assert all(decision["op"] == "consolidate" for decision in report.gate_decisions)

    # Each candidate routed through the I4 gate (auditable gate events recorded).
    gate_events = _events_of_type(eventlog, "evolution.gate.evaluated")
    assert len(gate_events) == 4

    accepted = [
        event
        for event in _events_of_type(eventlog, "consolidation.candidate.reviewed")
        if event.payload["status"] == "accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0].payload["authority_status"] == "non_authoritative"


@pytest.mark.asyncio
async def test_propose_only_policy_holds_everything_for_review(tmp_path: Path) -> None:
    fabric = await _build_fabric(tmp_path)
    fabric.settings.evolution_op_autonomy = "consolidate=propose_only"
    try:
        eventlog = _seed_crystallization_inputs(fabric, "crystal")
        report = await run_crystallization_pass(fabric, session_id="crystal")
    finally:
        await fabric.close()

    assert report.auto_accepted == 0
    assert report.left_pending == 4
    assert all(decision["auto_apply"] is False for decision in report.gate_decisions)
    assert all(decision["tier"] == "propose_only" for decision in report.gate_decisions)
    # Gate decisions are still recorded, but no accepted review is emitted.
    assert len(_events_of_type(eventlog, "evolution.gate.evaluated")) == 4
    assert not _events_of_type(eventlog, "consolidation.candidate.reviewed")


@pytest.mark.asyncio
async def test_no_auto_apply_leaves_everything_pending(tmp_path: Path) -> None:
    fabric = await _build_fabric(tmp_path)
    try:
        eventlog = _seed_crystallization_inputs(fabric, "crystal")
        report = await run_crystallization_pass(
            fabric, session_id="crystal", auto_apply=False
        )
    finally:
        await fabric.close()

    assert report.auto_accepted == 0
    assert report.left_pending == 4
    # Gate decisions are still recorded for visibility even with auto_apply off.
    assert len(report.gate_decisions) == 4
    assert any(decision["auto_apply"] for decision in report.gate_decisions)
    # ...but no accepted review is emitted.
    assert not _events_of_type(eventlog, "consolidation.candidate.reviewed")


@pytest.mark.asyncio
async def test_metacognition_monitor_fires_reverify_for_open_gap_idempotently(
    tmp_path: Path,
) -> None:
    fabric = await _build_fabric(tmp_path)
    try:
        eventlog = fabric.session_manager.get("crystal").eventlog
        support = _append_spec(
            eventlog,
            build_tool_call_completed_event(
                tool_name="memory_checkout", status="succeeded", session_id="crystal"
            ),
            thread="crystal",
        )
        conflict = _append_spec(
            eventlog,
            build_tool_call_completed_event(
                tool_name="memory_query", status="succeeded", session_id="crystal"
            ),
            thread="crystal",
        )
        cluster = build_conflict_cluster_event(
            actor="zaxy",
            session_id="crystal",
            claim_key="checkout.budget.limit",
            claim="The checkout budget limit is 4000 tokens",
            supporting_source_events=[{"seq": support.seq, "hash": support.hash}],
            conflicting_source_events=[{"seq": conflict.seq, "hash": conflict.hash}],
            confidence=0.5,
            reason="Two cited events disagree on the checkout budget limit.",
        )
        _append_spec(eventlog, cluster, thread="crystal")

        report = await run_crystallization_pass(
            fabric,
            session_id="crystal",
            consolidation=False,
            procedure_mining=False,
        )
        reverify_after_first = len(_events_of_type(eventlog, "metacognition.reverify.requested"))
        rerun = await run_crystallization_pass(
            fabric,
            session_id="crystal",
            consolidation=False,
            procedure_mining=False,
        )
        reverify_after_second = len(_events_of_type(eventlog, "metacognition.reverify.requested"))
    finally:
        await fabric.close()

    assert report.reverify_requested == 1
    assert report.metacognition["unresolved_conflict_cluster_count"] == 1
    assert reverify_after_first == 1
    emitted = _events_of_type(eventlog, "metacognition.reverify.requested")[0]
    assert emitted.payload["status"] == "open"
    assert emitted.payload["authority_status"] == "non_authoritative"
    assert emitted.payload["claim_key"] == "checkout.budget.limit"
    # The monitor does not re-fire for a gap that already has an open reverify.
    assert rerun.reverify_requested == 0
    assert reverify_after_second == 1


@pytest.mark.asyncio
async def test_report_counts_and_diagnostics_round_trip(tmp_path: Path) -> None:
    fabric = await _build_fabric(tmp_path)
    try:
        _seed_crystallization_inputs(fabric, "crystal")
        report = await run_crystallization_pass(
            fabric, session_id="crystal", compaction=True
        )
    finally:
        await fabric.close()

    payload = report.to_dict()
    # Counts are internally consistent.
    assert payload["auto_accepted"] + payload["left_pending"] == (
        payload["consolidation_candidates"] + payload["procedure_candidates"]
    )
    # Read-only salience diagnostic surfaced the reinforced memory.
    assert report.top_salient
    top = report.top_salient[0]
    assert set(top) == {"seq", "hash", "score"}
    assert top["score"] > 0.0
    # Compaction ran as an additive audit/projection diagnostic (report-only).
    assert payload["compaction"] is not None
    assert "safe" in payload["compaction"]
    # No projection file is written by the pass itself.
    projections = tmp_path / ".eventloom" / "projections"
    if projections.exists():
        assert not any(child.name.startswith("compaction") for child in projections.iterdir())
    assert payload["latency_ms"] > 0.0


@pytest.mark.asyncio
async def test_compaction_defaults_off(tmp_path: Path) -> None:
    fabric = await _build_fabric(tmp_path)
    try:
        _seed_crystallization_inputs(fabric, "crystal")
        report = await run_crystallization_pass(fabric, session_id="crystal")
    finally:
        await fabric.close()

    assert report.compaction is None


def test_cli_crystallize_json_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from zaxy.event import EventLog
    from zaxy.security import eventlog_path

    eventloom = tmp_path / ".eventloom"
    eventloom.mkdir(parents=True, exist_ok=True)
    log = EventLog(str(eventlog_path(eventloom, "default")))
    for name in _RECURRING:
        spec = build_tool_call_completed_event(
            tool_name=name, status="succeeded", session_id="default", result_summary=f"{name} ok"
        )
        log.append(spec["event_type"], actor=spec["actor"], payload=spec["payload"], thread="default")
    edit = build_file_edit_applied_event(
        path="src/zaxy/checkout.py", operation="edit", session_id="default", summary="adjust"
    )
    log.append(edit["event_type"], actor=edit["actor"], payload=edit["payload"], thread="default")

    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu\n"
        "NEO4J_AUTO_START=false\n"
        "CRYSTALLIZATION_ENABLED=true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["crystallize", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["session_id"] == "default"
    assert payload["consolidation_candidates"] == 3
    assert payload["latency_ms"] > 0.0
    assert "gate_decisions" in payload


def test_cli_crystallize_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".eventloom").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu\n"
        "NEO4J_AUTO_START=false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CRYSTALLIZATION_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["crystallize", "--json"])
    assert result.exit_code == 1
    assert "disabled" in result.output.lower()


@pytest.mark.asyncio
async def test_metacognition_monitor_fires_reverify_for_open_known_unknown(
    tmp_path: Path,
) -> None:
    fabric = await _build_fabric(tmp_path)
    try:
        eventlog = fabric.session_manager.get("crystal").eventlog
        source = _append_spec(
            eventlog,
            build_tool_call_completed_event(
                tool_name="memory_query", status="succeeded", session_id="crystal"
            ),
            thread="crystal",
        )
        unknown = build_known_unknown_event(
            actor="zaxy",
            session_id="crystal",
            claim_key="deploy.window.policy",
            question="What is the production token budget?",
            reason="No cited source establishes the production token budget.",
            source_events=[{"seq": source.seq, "hash": source.hash}],
            reverify_query="Re-verify the production token budget",
        )
        _append_spec(eventlog, unknown, thread="crystal")

        report = await run_crystallization_pass(
            fabric, session_id="crystal", consolidation=False, procedure_mining=False
        )
        reverify_after_first = len(_events_of_type(eventlog, "metacognition.reverify.requested"))
        rerun = await run_crystallization_pass(
            fabric, session_id="crystal", consolidation=False, procedure_mining=False
        )
        reverify_after_second = len(_events_of_type(eventlog, "metacognition.reverify.requested"))
    finally:
        await fabric.close()

    assert report.reverify_requested == 1
    assert reverify_after_first == 1
    emitted = _events_of_type(eventlog, "metacognition.reverify.requested")[0]
    assert emitted.payload["status"] == "open"
    assert emitted.payload["authority_status"] == "non_authoritative"
    assert emitted.payload["priority"] == "normal"
    assert "production token budget" in emitted.payload["query"]
    # Idempotent: the open known-unknown already carries an open reverify request.
    assert rerun.reverify_requested == 0
    assert reverify_after_second == 1


def test_plan_reverify_and_citation_helper_edge_branches() -> None:
    from zaxy.crystallization import (
        _merge_citations,
        _normalize_citations,
        _optional_str,
        _plan_reverify_requests,
    )

    cite = {"seq": 1, "hash": "a" * 64}
    summary = {
        "conflict_clusters": [
            # claim_key=None forces dedup through the reverify-id path, not claim-key.
            {
                "claim_key": None,
                "claim": "c",
                "supporting_source_events": [cite],
                "conflicting_source_events": [],
                "cluster_id": "cl",
            },
            # No sources -> skipped.
            {"claim_key": "k2", "supporting_source_events": [], "conflicting_source_events": []},
        ],
        "open_unknowns": [
            {"claim_key": None, "question": "q", "source_events": [cite], "unknown_id": "uk"},
            # No sources -> skipped.
            {"claim_key": "u2", "question": "q2", "source_events": []},
        ],
    }
    existing: set[str] = set()
    open_keys: set[str] = set()
    first = _plan_reverify_requests(
        summary, actor="t", session_id="s", existing_reverify_ids=existing, open_reverify_claim_keys=open_keys
    )
    assert len(first) == 2  # the two cited gaps; the two source-less gaps are skipped
    # Re-running with the now-populated id set exercises the reverify-id dedup skip.
    second = _plan_reverify_requests(
        summary, actor="t", session_id="s", existing_reverify_ids=existing, open_reverify_claim_keys=open_keys
    )
    assert second == []

    assert _optional_str(None) is None
    assert _optional_str("   ") is None
    assert _optional_str(123) is None
    assert _optional_str("x") == "x"
    assert _normalize_citations("not-a-list") == []
    assert _normalize_citations([{"seq": 0, "hash": "a" * 64}, {"hash": "a" * 64}, cite]) == [cite]
    assert _merge_citations(None, [cite], [cite]) == [cite]
