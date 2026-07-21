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
from zaxy.learned_context import (
    LEARNED_CONTEXT_DIRNAME,
    LEARNED_CONTEXT_EVENT_TYPE,
    load_learned_context,
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


@pytest.mark.asyncio
async def test_failing_stage_still_appends_the_summary_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage that raises must not prevent the crystallization.run.completed audit record."""
    fabric = await _build_fabric(tmp_path)
    try:
        eventlog = _seed_crystallization_inputs(fabric, "crystal")

        async def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("consolidation projection unavailable")

        monkeypatch.setattr(fabric, "propose_consolidation_candidates", boom)
        report = await run_crystallization_pass(fabric, session_id="crystal")
    finally:
        await fabric.close()

    summaries = _events_of_type(eventlog, CRYSTALLIZATION_EVENT_TYPE)
    assert len(summaries) == 1
    payload = summaries[0].payload
    assert payload["ok"] is False
    assert [entry["stage"] for entry in payload["stage_errors"]] == ["consolidation"]
    assert payload["stage_errors"][0]["error_type"] == "RuntimeError"
    assert "consolidation projection unavailable" in payload["stage_errors"][0]["error"]
    # The failure is isolated: later stages still ran and are honestly counted.
    assert payload["consolidation_candidates"] == 0
    assert report.procedure_candidates == 1
    assert report.ok is False


@pytest.mark.asyncio
async def test_successful_pass_reports_ok_with_no_stage_errors(tmp_path: Path) -> None:
    """A pass where every stage succeeds reports ok with an empty stage_errors list."""
    fabric = await _build_fabric(tmp_path)
    try:
        eventlog = _seed_crystallization_inputs(fabric, "crystal")
        report = await run_crystallization_pass(fabric, session_id="crystal")
    finally:
        await fabric.close()

    assert report.stage_errors == []
    assert report.ok is True
    assert report.to_dict()["ok"] is True
    payload = _events_of_type(eventlog, CRYSTALLIZATION_EVENT_TYPE)[0].payload
    assert payload["ok"] is True
    assert payload["stage_errors"] == []


@pytest.mark.asyncio
async def test_independent_stage_failures_are_each_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple failing stages are each recorded rather than the first aborting the pass."""
    fabric = await _build_fabric(tmp_path)
    try:
        _seed_crystallization_inputs(fabric, "crystal")

        def mining_boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("mining index corrupt")

        def salience_boom(*args: Any, **kwargs: Any) -> Any:
            raise KeyError("salience ledger missing")

        monkeypatch.setattr("zaxy.crystallization.mine_and_propose", mining_boom)
        monkeypatch.setattr("zaxy.crystallization._top_salient", salience_boom)
        report = await run_crystallization_pass(fabric, session_id="crystal")
    finally:
        await fabric.close()

    stages = {entry["stage"]: entry for entry in report.stage_errors}
    assert set(stages) == {"procedure_mining", "salience"}
    assert stages["procedure_mining"]["error_type"] == "ValueError"
    assert stages["salience"]["error_type"] == "KeyError"
    # The consolidation stage was untouched and still did its work.
    assert report.consolidation_candidates == 3
    assert report.top_salient == []


@pytest.mark.asyncio
async def test_gate_failure_is_isolated_per_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One un-gateable candidate is recorded without costing the rest their gate decisions."""
    fabric = await _build_fabric(tmp_path)
    original = fabric.evaluate_evolution_gate
    calls: list[str] = []

    try:
        _seed_crystallization_inputs(fabric, "crystal")

        async def flaky(op: Any, confidence: Any, **kwargs: Any) -> Any:
            candidate_id = str(kwargs["candidate_ref"]["candidate_id"])
            calls.append(candidate_id)
            if len(calls) == 1:
                raise RuntimeError(f"gate policy unreadable for {candidate_id}")
            return await original(op, confidence, **kwargs)

        monkeypatch.setattr(fabric, "evaluate_evolution_gate", flaky)
        report = await run_crystallization_pass(fabric, session_id="crystal")
    finally:
        await fabric.close()

    assert len(calls) == 4
    assert len(report.gate_decisions) == 3
    assert [entry["stage"] for entry in report.stage_errors] == ["gating"]
    assert report.stage_errors[0]["candidate_id"] == calls[0]
    # The failed candidate is counted neither accepted nor pending.
    assert report.auto_accepted + report.left_pending == 3


def test_cli_crystallize_exits_non_zero_when_a_stage_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage failure must surface to cron as a non-zero exit with the error on stderr."""
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

    (tmp_path / ".env.local").write_text(
        "PROJECTION_BACKEND=embedded\n"
        "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu\n"
        "NEO4J_AUTO_START=false\n"
        "CRYSTALLIZATION_ENABLED=true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def mining_boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("mining index corrupt")

    monkeypatch.setattr("zaxy.crystallization.mine_and_propose", mining_boom)

    result = CliRunner().invoke(app, ["crystallize", "--json"])

    assert result.exit_code == 1, result.output
    assert "procedure_mining" in result.output
    assert "mining index corrupt" in result.output


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


# --------------------------------------------------------------------------
# I2 learned-context persistence (gated; off by default).
# --------------------------------------------------------------------------


async def _crystallize_with_learned_context(tmp_path: Path, *, enabled: bool) -> tuple[Any, Any]:
    """Run a compaction-only pass over a log the audit judges SAFE.

    Only the compaction stage runs: the other stages append candidate events,
    which would grow the log past the point where a single representative still
    carries every identity, and the audit (rightly) refuses to compact that.
    """
    from zaxy.config import Settings

    fabric = await _build_fabric(tmp_path)
    fabric.settings = Settings(learned_context_enabled=enabled)
    try:
        eventlog = fabric.session_manager.get("crystal").eventlog
        eventlog.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": "docs/one.md",
                "start_line": 1,
                "end_line": 3,
                "content": "The single source carries identity-code-0001.",
            },
            thread="crystal",
        )
        report = await run_crystallization_pass(
            fabric,
            session_id="crystal",
            compaction=True,
            consolidation=False,
            procedure_mining=False,
            metacognition=False,
        )
        events = eventlog.read_all()
    finally:
        await fabric.close()
    return report, events


@pytest.mark.asyncio
async def test_crystallization_persists_the_learned_context_artifact_when_enabled(tmp_path: Path) -> None:
    """With the gate on, the pass writes the artifact and records the build in the log."""
    report, events = await _crystallize_with_learned_context(tmp_path, enabled=True)

    assert report.compaction is not None
    assert report.compaction["safe"] is True
    persisted = report.compaction["learned_context"]
    assert persisted["persisted"] is True

    artifact = Path(persisted["artifact_path"])
    assert artifact.exists()
    assert artifact.parent.name == LEARNED_CONTEXT_DIRNAME
    envelope = json.loads(artifact.read_text(encoding="utf-8"))
    assert envelope["covered_seq"] == persisted["covered_seq"]

    built = _events_of_type_in(events, LEARNED_CONTEXT_EVENT_TYPE)
    assert len(built) == 1
    payload = built[0].payload
    assert payload["authority_status"] == "non_authoritative"
    assert payload["projection_id"] == envelope["projection"]["projection_id"]
    assert payload["covered_head"] == {
        "seq": persisted["covered_seq"],
        "hash": persisted["covered_hash"],
    }
    assert payload["record_count"] == persisted["record_count"]

    # The covered head is the log tip as it stood BEFORE the build event was
    # appended, so the build event never invalidates its own projection.
    assert payload["covered_head"]["seq"] < built[0].seq

    # The artifact is a cache the log vouches for: it loads clean against this log.
    load = load_learned_context(artifact, events)
    assert load.projection is not None
    assert load.stale is False


@pytest.mark.asyncio
async def test_crystallization_writes_nothing_when_learned_context_is_disabled(tmp_path: Path) -> None:
    """With the gate off (the default), the compaction stage stays report-only."""
    report, events = await _crystallize_with_learned_context(tmp_path, enabled=False)

    assert report.compaction is not None
    assert report.compaction["safe"] is True
    assert "learned_context" not in report.compaction
    assert _events_of_type_in(events, LEARNED_CONTEXT_EVENT_TYPE) == []
    assert not (tmp_path / ".eventloom" / "projections" / LEARNED_CONTEXT_DIRNAME).exists()


@pytest.mark.asyncio
async def test_crystallization_deleting_the_artifact_loses_nothing(tmp_path: Path) -> None:
    """The artifact is a cache: deleting it degrades to "missing" and loses no evidence.

    Everything the artifact held is still derivable — the build event in the log
    records what was built, and the source events it compacted are untouched.
    """
    report, events = await _crystallize_with_learned_context(tmp_path, enabled=True)
    artifact = Path(report.compaction["learned_context"]["artifact_path"])
    assert load_learned_context(artifact, events).projection is not None

    artifact.unlink()

    load = load_learned_context(artifact, events)
    assert load.projection is None
    assert load.stale is False
    assert load.reason == "missing"
    # The evidence survived the cache: the build event still describes the projection.
    built = _events_of_type_in(events, LEARNED_CONTEXT_EVENT_TYPE)[0]
    assert built.payload["projection_id"]
    assert built.payload["record_count"] >= 1
    assert built.payload["covered_head"]["seq"] >= 1


@pytest.mark.asyncio
async def test_crystallization_projection_build_is_deterministic(tmp_path: Path) -> None:
    """Rebuilding over the same source events reproduces the same projection identity."""
    from zaxy.compaction import build_compaction_projection
    from zaxy.event import EventLog

    log = EventLog(tmp_path / "src.jsonl")
    log.append(
        "document.indexed",
        actor="indexer",
        payload={"path": "docs/one.md", "start_line": 1, "end_line": 3, "content": "identity-code-0001 here."},
    )
    first = build_compaction_projection(log)
    second = build_compaction_projection(log)

    assert first.projection_id == second.projection_id
    assert first.records == second.records


def _events_of_type_in(events: list[Any], event_type: str) -> list[Any]:
    return [event for event in events if event.type == event_type]
