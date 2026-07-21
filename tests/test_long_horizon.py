"""Tests for I3 two-tier (episodic + consolidated) long-horizon checkout assembly.

Covers the partition helper (``zaxy.long_horizon``) and the end-to-end fabric
wiring: a long thread gets a bounded, cited consolidated remote tier for older
history while the default-off path stays byte-identical to the single-tier
checkout. Uses the real embedded MemoryFabric, the real consolidation builders,
and accept-via-review-event, mirroring the consolidation pipeline tests.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from zaxy.compaction import (
    CompactionAuditReport,
    CompactionProjection,
    CompactionProjectionRecord,
    text_tokens,
)
from zaxy.config import Settings
from zaxy.consolidation import build_consolidation_review_event
from zaxy.context import Context
from zaxy.core import MemoryFabric
from zaxy.core.checkout_build import (
    _checkout_long_horizon,
    _checkout_recall_limit,
    build_memory_checkout,
)
from zaxy.learned_context import (
    LEARNED_CONTEXT_EVENT_TYPE,
    LearnedContextLoad,
    build_projection_built_payload,
    learned_context_path,
    write_learned_context,
)
from zaxy.long_horizon import CONSOLIDATED_LANE, build_long_horizon_plan, span_relevance

_CITATION_RE = re.compile(r"^eventloom://[^/\s]+/events/[1-9][0-9]*#[0-9a-f]{12}$")
_RECENT_WINDOW = 10
_QUERY = "what did we learn from the earlier refactor work"


# --------------------------------------------------------------------------
# Partition-helper unit tests (synthetic events) — branch coverage.
# --------------------------------------------------------------------------


def _ev(seq: int, event_type: str, payload: dict | None = None, thread: str = "thread-x") -> SimpleNamespace:
    return SimpleNamespace(seq=seq, type=event_type, payload=payload or {}, hash="a" * 64, thread=thread)


def _candidate_created(seq: int, candidate_id: str, *, review_status: str, source_seqs: list[int], summary: str = "summary") -> SimpleNamespace:
    return _ev(
        seq,
        "consolidation.candidate.created",
        {
            "candidate_id": candidate_id,
            "candidate_type": candidate_id.split(":")[1],
            "summary": summary,
            "confidence": 0.7,
            "review_status": review_status,
            "source_events": [{"seq": s, "hash": "b" * 64} for s in source_seqs],
        },
    )


def test_build_long_horizon_plan_short_session_is_episodic_only() -> None:
    events = [_ev(i, "tool.call.completed") for i in range(1, 4)]
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=10, budget=10)
    assert plan.enabled is True
    assert plan.horizon_split_seq is None
    assert plan.episodic_count == 3
    assert plan.consolidated_count == 0
    assert plan.consolidated_contexts == []
    assert plan.to_diagnostics() == {
        "enabled": True,
        "recent_window": 10,
        "episodic_count": 3,
        "horizon_split_seq": None,
    }


def test_build_long_horizon_plan_surfaces_accepted_old_candidates_only() -> None:
    events: list[SimpleNamespace] = [_ev(i, "tool.call.completed", {"summary": f"e{i}"}) for i in range(1, 6)]
    # Accepted via a later review event, sourced from old events (seqs 1-2).
    events.append(_candidate_created(6, "consolidation:episode:" + "0" * 24, review_status="pending", source_seqs=[1, 2]))
    events.append(
        _ev(7, "consolidation.candidate.reviewed", {"candidate_id": "consolidation:episode:" + "0" * 24, "status": "accepted"})
    )
    # Accepted but sourced from a RECENT event (seq 8 > split) -> excluded.
    events.append(_candidate_created(8, "consolidation:claim:" + "1" * 24, review_status="accepted", source_seqs=[8]))
    # Old sources but never accepted (pending) -> excluded.
    events.append(_candidate_created(9, "consolidation:procedure:" + "2" * 24, review_status="pending", source_seqs=[1]))

    # total=9, window=3 -> older = seqs 1..6, split_seq = 6.
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=3, budget=10)
    assert plan.horizon_split_seq == 6
    assert plan.episodic_count == 3
    assert plan.consolidated_count == 1
    ctx = plan.consolidated_contexts[0]
    assert ctx.source == CONSOLIDATED_LANE
    assert ctx.metadata["candidate_id"] == "consolidation:episode:" + "0" * 24
    assert ctx.metadata["assembly_lane"] == CONSOLIDATED_LANE
    assert ctx.metadata["review_status"] == "accepted"  # latest review disposition wins
    assert ctx.metadata["non_authoritative"] is True
    assert ctx.metadata["authority_status"] == "non_authoritative"
    assert ctx.metadata["source_event_citations"] == [
        "eventloom://thread-x/events/1#" + "b" * 12,
        "eventloom://thread-x/events/2#" + "b" * 12,
    ]
    # The candidate's own created-event citation is real and distinct from sources.
    assert ctx.metadata["citation"] == "eventloom://thread-x/events/6#" + "a" * 12


def test_build_long_horizon_plan_excludes_candidate_without_source_events() -> None:
    events = [_ev(i, "tool.call.completed") for i in range(1, 6)]
    events.append(_candidate_created(6, "consolidation:episode:" + "f" * 24, review_status="accepted", source_seqs=[]))
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=2, budget=10)
    # max_source_seq is None (no cited sources) -> not surfaced.
    assert plan.consolidated_count == 0


def test_build_long_horizon_plan_bounds_consolidated_by_budget() -> None:
    events = [_ev(i, "tool.call.completed") for i in range(1, 6)]
    for n in range(3):
        events.append(
            _candidate_created(10 + n, f"consolidation:episode:{str(n) * 24}", review_status="accepted", source_seqs=[1])
        )
    # total=8, window=2 -> split allows all 3; budget caps at 2.
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=2, budget=2)
    assert plan.consolidated_count == 2


def test_build_long_horizon_plan_tolerates_malformed_candidate_events() -> None:
    candidate_id = "consolidation:episode:" + "0" * 24
    # Null hash -> the candidate's own citation resolves to None; a non-dict source
    # row is skipped while a valid one is kept; a duplicate created event is ignored.
    null_hash_created = SimpleNamespace(
        seq=6,
        type="consolidation.candidate.created",
        payload={
            "candidate_id": candidate_id,
            "candidate_type": "episode",
            "summary": "episode with one valid source",
            "confidence": 0.7,
            "review_status": "accepted",
            "source_events": ["not-a-dict", {"seq": 1, "hash": "b" * 64}],
        },
        hash=None,
        thread="thread-x",
    )
    duplicate_created = _candidate_created(7, candidate_id, review_status="accepted", source_seqs=[2])
    events = [_ev(i, "tool.call.completed") for i in range(1, 6)] + [null_hash_created, duplicate_created]
    # total=7, window=2 -> older = seqs 1..5, split_seq = 5.
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=2, budget=10)
    assert plan.consolidated_count == 1
    ctx = plan.consolidated_contexts[0]
    assert ctx.metadata["citation"] is None
    assert ctx.metadata["source_event_citations"] == ["eventloom://thread-x/events/1#" + "b" * 12]


def test_checkout_long_horizon_summarizer_dedups_and_skips_unidentified() -> None:
    candidate_id = "consolidation:episode:" + "0" * 24
    citation = "eventloom://t/events/1#" + "b" * 12

    def _consolidated(cid: str | None) -> Context:
        return Context(
            content="consolidated summary",
            source=CONSOLIDATED_LANE,
            score=0.5,
            metadata={
                "assembly_lane": CONSOLIDATED_LANE,
                "candidate_id": cid,
                "candidate_type": "episode",
                "citation": citation,
                "source_event_citations": [citation, "", 7],
                "review_status": "accepted",
                "confidence": 0.7,
            },
        )

    non_lane = Context(content="x", source="graph", score=0.9, metadata={"assembly_lane": "graph"})
    missing_id = Context(content="y", source=CONSOLIDATED_LANE, score=0.1, metadata={"assembly_lane": CONSOLIDATED_LANE})
    items = _checkout_long_horizon([non_lane, _consolidated(candidate_id), _consolidated(candidate_id), missing_id])

    assert len(items) == 1  # non-lane skipped, duplicate deduped, missing id skipped
    item = items[0]
    assert item["candidate_id"] == candidate_id
    assert item["non_authoritative"] is True
    assert item["authority_status"] == "non_authoritative"
    assert item["source_event_citations"] == [citation]  # blank/non-str citations dropped
    assert item["source_event_count"] == 1


# --------------------------------------------------------------------------
# End-to-end fabric tests (real embedded MemoryFabric).
# --------------------------------------------------------------------------


async def _build_fabric(tmp_path: Path, *, enabled: bool, recent_window: int = _RECENT_WINDOW) -> MemoryFabric:
    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    fabric.settings = Settings(long_horizon_enabled=enabled, long_horizon_recent_window=recent_window)
    await fabric.connect()
    return fabric


async def _seed_old_actionable(fabric: MemoryFabric, sid: str, count: int) -> list:
    events = []
    for i in range(count):
        if i % 2 == 0:
            event = await fabric.append(
                "tool.call.completed",
                actor="agent",
                payload={
                    "tool_name": "pytest",
                    "status": "failed" if i % 4 == 0 else "succeeded",
                    "summary": f"early step {i} executed",
                },
                session_id=sid,
            )
        else:
            event = await fabric.append(
                "file.edit.applied",
                actor="agent",
                payload={"path": f"src/zaxy/module_{i}.py", "summary": f"edited module {i}"},
                session_id=sid,
            )
        events.append(event)
    return events


async def _seed_recent(fabric: MemoryFabric, sid: str, count: int) -> list:
    events = []
    for i in range(count):
        event = await fabric.append(
            "tool.call.completed",
            actor="agent",
            payload={"tool_name": "ruff", "status": "succeeded", "summary": f"recent activity {i}"},
            session_id=sid,
        )
        events.append(event)
    return events


async def _accept_all(fabric: MemoryFabric, sid: str, result: dict) -> None:
    for entry in result["events"]:
        review = build_consolidation_review_event(
            actor="reviewer",
            session_id=sid,
            candidate_id=entry["candidate_id"],
            status="accepted",
            rationale="Cited and useful consolidated knowledge for older history.",
        )
        await fabric.append(review["event_type"], actor=review["actor"], payload=review["payload"], session_id=sid)


async def _seed_long_thread_with_consolidation(fabric: MemoryFabric, sid: str) -> dict:
    """Seed an old actionable region, accept its consolidation candidates, then recent events."""
    await _seed_old_actionable(fabric, sid, 8)
    result = await fabric.propose_consolidation_candidates(session_id=sid, window_size=8)
    assert result["candidate_count"] >= 1
    await _accept_all(fabric, sid, result)
    await _seed_recent(fabric, sid, 20)
    return result


@pytest.mark.asyncio
async def test_long_horizon_engaged_surfaces_cited_consolidated_remote_tier(tmp_path) -> None:
    fabric = await _build_fabric(tmp_path, enabled=True)
    sid = "long-thread"
    try:
        result = await _seed_long_thread_with_consolidation(fabric, sid)
        checkout = await fabric.checkout_memory(
            _QUERY,
            session_id=sid,
            long_horizon=True,
            max_recent_events=_RECENT_WINDOW,
            record_reinforcement=False,
        )
    finally:
        await fabric.close()

    long_horizon = checkout.diagnostics["long_horizon"]
    assert long_horizon["enabled"] is True
    assert long_horizon["recent_window"] == _RECENT_WINDOW
    assert long_horizon["episodic_count"] == _RECENT_WINDOW
    split_seq = long_horizon["horizon_split_seq"]
    assert isinstance(split_seq, int) and split_seq >= 8
    assert long_horizon["consolidated_count"] >= 1
    assert long_horizon["non_authoritative"] is True

    items = long_horizon["items"]
    assert items
    surfaced_ids = {item["candidate_id"] for item in items}
    assert surfaced_ids
    assert surfaced_ids <= {entry["candidate_id"] for entry in result["events"]}
    for item in items:
        # Non-authoritative, cited consolidation artifact (not a raw old event).
        assert item["non_authoritative"] is True
        assert item["authority_status"] == "non_authoritative"
        assert item["candidate_id"].startswith("consolidation:")
        assert item["candidate_type"] in {"episode", "claim", "procedure"}
        assert _CITATION_RE.match(item["citation"])
        assert item["source_event_citations"]
        for citation in item["source_event_citations"]:
            assert _CITATION_RE.match(citation)
            source_seq = int(citation.split("/events/")[1].split("#")[0])
            # Every cited source is older history, beyond the episodic window.
            assert source_seq <= split_seq


@pytest.mark.asyncio
async def test_long_horizon_default_off_is_byte_identical(tmp_path) -> None:
    fabric = await _build_fabric(tmp_path, enabled=False)
    sid = "long-thread"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    recall_limit = _checkout_recall_limit(_QUERY, 10)

    async def _checkout(long_horizon):
        assembly = await fabric.assemble_context(
            _QUERY,
            session_id=sid,
            limit=10,
            recall_limit=recall_limit,
            max_recent_events=20,
            long_horizon=long_horizon,
        )
        checkout = build_memory_checkout(
            query=_QUERY,
            assembly=assembly,
            now=now,
            retrieval_profile=fabric.retrieval_profile,
            salience_floor=fabric._salience_floor,
            salience_half_life_days=fabric._salience_half_life_days,
        )
        return assembly, checkout

    try:
        await _seed_long_thread_with_consolidation(fabric, sid)
        assembly_default, checkout_default = await _checkout(None)
        assembly_off, checkout_off = await _checkout(False)
        assembly_on, checkout_on = await _checkout(True)
    finally:
        await fabric.close()

    # The disabled assembly is provably inert: both new fields hold their defaults,
    # so neither the consolidated merge nor the diagnostics branch can fire.
    assert assembly_default.long_horizon is None
    assert assembly_default.long_horizon_contexts == []
    assert assembly_off.long_horizon is None
    assert assembly_off.long_horizon_contexts == []

    # Default and explicit-False yield the identical single-tier checkout, with no
    # long_horizon diagnostics key — byte-identical to the pre-I3 contract.
    assert "long_horizon" not in checkout_default.diagnostics
    assert "long_horizon" not in checkout_off.diagnostics
    assert checkout_off.to_dict() == checkout_default.to_dict()
    # No consolidated-lane content leaks into the single-tier checkout.
    assert all("consolidation:" not in str(entry) for entry in checkout_off.provenance)
    assert all(fact.get("source_lane") != CONSOLIDATED_LANE for fact in checkout_off.current_facts)

    # Engaging the feature on the SAME seed adds the consolidated tier and changes
    # the packet — proving the two-tier path is doing real work, not a no-op.
    assert assembly_on.long_horizon is not None
    assert assembly_on.long_horizon_contexts
    assert "long_horizon" in checkout_on.diagnostics
    assert checkout_on.diagnostics["long_horizon"]["consolidated_count"] >= 1
    assert checkout_on.to_dict() != checkout_off.to_dict()


@pytest.mark.asyncio
async def test_long_horizon_short_session_is_graceful(tmp_path) -> None:
    fabric = await _build_fabric(tmp_path, enabled=True)
    sid = "short-thread"
    try:
        await _seed_old_actionable(fabric, sid, 4)  # below the recent window
        checkout = await fabric.checkout_memory(
            _QUERY, session_id=sid, long_horizon=True, record_reinforcement=False
        )
    finally:
        await fabric.close()

    long_horizon = checkout.diagnostics["long_horizon"]
    assert long_horizon["enabled"] is True
    assert long_horizon["horizon_split_seq"] is None
    assert long_horizon["episodic_count"] == 4
    assert long_horizon["consolidated_count"] == 0
    assert long_horizon["items"] == []
    assert all("consolidation:" not in str(entry) for entry in checkout.provenance)


@pytest.mark.asyncio
async def test_long_horizon_window_guarded_against_raw_recent_overlap(tmp_path) -> None:
    """A configured recent_window smaller than max_recent_events widens to the latter,
    so an event rendered in the raw Recent Events section is never ALSO consolidated."""
    fabric = await _build_fabric(tmp_path, enabled=True)
    sid = "long-thread"
    try:
        await _seed_long_thread_with_consolidation(fabric, sid)
        # configured long_horizon_recent_window (_RECENT_WINDOW=10) < max_recent_events (20)
        checkout = await fabric.checkout_memory(
            _QUERY,
            session_id=sid,
            long_horizon=True,
            max_recent_events=20,
            record_reinforcement=False,
        )
    finally:
        await fabric.close()
    long_horizon = checkout.diagnostics["long_horizon"]
    # the effective split widened to max_recent_events (20), not the configured 10
    assert long_horizon["recent_window"] == 20
    assert long_horizon["episodic_count"] <= 20


# --------------------------------------------------------------------------
# I3 long-span relevance scoring + I2 learned-context consumption.
# --------------------------------------------------------------------------


def _lc_record(seq: int, text: str) -> CompactionProjectionRecord:
    ref = f"eventloom://thread-x/events/{seq}#" + "c" * 12
    return CompactionProjectionRecord(
        kind="medoid",
        event_seq=seq,
        event_ref=ref,
        text=text,
        identities=(ref,),
        citations=(ref,),
    )


def _lc_projection(records: tuple[CompactionProjectionRecord, ...]) -> CompactionProjection:
    return CompactionProjection(
        projection_id="d" * 64,
        strategy="medoid",
        source_event_count=len(records),
        source_identities=(),
        records=records,
        audit=CompactionAuditReport(
            safe=True,
            event_count=len(records),
            integrity_ok=True,
            integrity_reason=None,
            identity_count=0,
            identity_recall=1.0,
            citation_coverage=1.0,
            mean_within_cluster_distance=0.0,
            identities=(),
            identity_hits=(),
            missing_identities=(),
            unsafe_reasons=(),
        ),
    )


def _loaded(records: tuple[CompactionProjectionRecord, ...]) -> LearnedContextLoad:
    return LearnedContextLoad(
        projection=_lc_projection(records),
        stale=False,
        reason=None,
        covered_seq=99,
        projection_id="d" * 64,
    )


def test_span_relevance_weights_every_term_and_reports_them() -> None:
    """The span score is the documented weighted sum and echoes every input term."""
    score, terms = span_relevance(
        text="earlier refactor work on the query engine",
        query_tokens=text_tokens("earlier refactor work"),
        source_event_count=8,
        max_source_seq=100,
        split_seq=100,
        confidence=1.0,
    )
    # All three query tokens hit, coverage saturates at 8, the newest source sits
    # exactly at the split, and the authoring prior is maximal -> every term is 1.0.
    assert terms["query_overlap"] == 1.0
    assert terms["span_coverage"] == 1.0
    assert terms["horizon_proximity"] == 1.0
    assert terms["authoring_prior"] == 1.0
    assert score == 1.0
    # The score is reconstructible from the reported terms alone.
    assert score == pytest.approx(
        sum(terms[name] * weight for name, weight in terms["weights"].items())
    )

    # A non-degenerate point: with the terms all distinct, the score can only be
    # the weighted sum — it cannot coincide with any single term.
    mixed, mixed_terms = span_relevance(
        text="earlier work",
        query_tokens=text_tokens("earlier refactor work landed cleanly"),
        source_event_count=2,
        max_source_seq=30,
        split_seq=100,
        confidence=0.8,
    )
    assert mixed_terms["query_overlap"] == pytest.approx(0.4)
    assert mixed_terms["span_coverage"] == pytest.approx(0.25)
    assert mixed_terms["horizon_proximity"] == pytest.approx(0.3)
    assert mixed_terms["authoring_prior"] == pytest.approx(0.8)
    assert mixed == pytest.approx(
        0.40 * 0.4 + 0.25 * 0.25 + 0.20 * 0.3 + 0.15 * 0.8
    )
    assert mixed not in {0.4, 0.25, 0.3, 0.8}


def test_span_relevance_is_not_the_authoring_confidence() -> None:
    """A high-confidence item with no query fit and deep history outranks nothing.

    This is the I3 defect in one assertion: the old score WAS the confidence, so
    these two items would have tied at 0.9. Query fit and horizon proximity must
    separate them.
    """
    deep_irrelevant, _ = span_relevance(
        text="unrelated ancient chatter",
        query_tokens=text_tokens("earlier refactor work"),
        source_event_count=1,
        max_source_seq=1,
        split_seq=1000,
        confidence=0.9,
    )
    near_relevant, _ = span_relevance(
        text="earlier refactor work landed",
        query_tokens=text_tokens("earlier refactor work"),
        source_event_count=1,
        max_source_seq=1000,
        split_seq=1000,
        confidence=0.9,
    )
    assert near_relevant > deep_irrelevant
    assert deep_irrelevant != 0.9


def test_span_relevance_uses_a_neutral_prior_when_confidence_is_absent() -> None:
    """Records with no authoring confidence get a neutral prior, not a zero."""
    _, terms = span_relevance(
        text="anything",
        query_tokens=set(),
        source_event_count=1,
        max_source_seq=None,
        split_seq=0,
        confidence=None,
    )
    assert terms["authoring_prior"] == 0.5
    assert terms["query_overlap"] == 0.0
    assert terms["horizon_proximity"] == 0.0


def test_span_relevance_rewards_items_covering_more_scrolled_out_history() -> None:
    """Span coverage rises with source-event count and saturates rather than dominating."""
    one, _ = span_relevance(text="t", query_tokens=set(), source_event_count=1, max_source_seq=1, split_seq=1)
    eight, _ = span_relevance(text="t", query_tokens=set(), source_event_count=8, max_source_seq=1, split_seq=1)
    huge, _ = span_relevance(text="t", query_tokens=set(), source_event_count=800, max_source_seq=1, split_seq=1)
    assert one < eight
    assert huge == eight


def test_long_horizon_plan_without_learned_context_omits_the_diagnostics_stanza() -> None:
    """With no learned context passed, the plan is the pre-I2 single-source plan."""
    events = [_ev(i, "tool.call.completed") for i in range(1, 6)]
    events.append(_candidate_created(6, "consolidation:episode:" + "0" * 24, review_status="accepted", source_seqs=[1, 2]))
    plan = build_long_horizon_plan(events, session_id="thread-x", recent_window=2, budget=10)
    assert plan.learned_context is None
    assert "learned_context" not in plan.to_diagnostics()
    assert plan.consolidated_count == 1


def test_long_horizon_plan_surfaces_cited_learned_context_records() -> None:
    """A verified projection adds cited, non-authoritative records as a second source."""
    events = [_ev(i, "tool.call.completed") for i in range(1, 8)]
    records = (_lc_record(2, "earlier refactor work on the parser"),)
    plan = build_long_horizon_plan(
        events,
        session_id="thread-x",
        recent_window=2,
        budget=10,
        learned_context=_loaded(records),
        query=_QUERY,
    )
    assert plan.consolidated_count == 1
    ctx = plan.consolidated_contexts[0]
    assert ctx.metadata["learned_context"] is True
    assert ctx.metadata["non_authoritative"] is True
    assert ctx.metadata["confidence"] is None
    assert ctx.metadata["projection_id"] == "d" * 64
    # Every surfaced record cites its source event.
    assert ctx.metadata["source_event_citations"]
    for citation in ctx.metadata["source_event_citations"]:
        assert _CITATION_RE.match(citation)
    assert plan.to_diagnostics()["learned_context"]["available"] is True


def test_long_horizon_plan_prefers_the_accepted_candidate_over_a_duplicate_record() -> None:
    """A candidate and a projection record over the same source event surface ONCE.

    The accepted candidate wins because it carries a human review decision that
    the mechanically-derived projection record does not.
    """
    candidate_id = "consolidation:episode:" + "0" * 24
    events = [_ev(i, "tool.call.completed") for i in range(1, 8)]
    events.append(
        _candidate_created(8, candidate_id, review_status="accepted", source_seqs=[2], summary="earlier refactor work")
    )
    # The projection record covers source event seq 2 — the same history.
    records = (_lc_record(2, "earlier refactor work on the parser"),)
    plan = build_long_horizon_plan(
        events,
        session_id="thread-x",
        recent_window=2,
        budget=10,
        learned_context=_loaded(records),
        query=_QUERY,
    )
    assert plan.consolidated_count == 1
    surviving = plan.consolidated_contexts[0]
    assert surviving.metadata["candidate_id"] == candidate_id
    assert surviving.metadata.get("learned_context") is None


def test_long_horizon_plan_keeps_a_record_covering_different_history() -> None:
    """Dedupe is per source event: a record over UNcovered history still surfaces."""
    candidate_id = "consolidation:episode:" + "0" * 24
    events = [_ev(i, "tool.call.completed") for i in range(1, 8)]
    events.append(
        _candidate_created(8, candidate_id, review_status="accepted", source_seqs=[2], summary="earlier refactor work")
    )
    records = (_lc_record(3, "earlier refactor work elsewhere"),)
    plan = build_long_horizon_plan(
        events,
        session_id="thread-x",
        recent_window=2,
        budget=10,
        learned_context=_loaded(records),
        query=_QUERY,
    )
    assert plan.consolidated_count == 2
    assert {ctx.metadata["candidate_id"] for ctx in plan.consolidated_contexts} == {
        candidate_id,
        "learned-context:" + "d" * 12 + ":3",
    }


def test_long_horizon_plan_applies_the_recent_window_guard_to_records() -> None:
    """A projection record whose source is still inside the episodic window is skipped."""
    events = [_ev(i, "tool.call.completed") for i in range(1, 8)]
    # total=7, window=2 -> split_seq=5; seq 7 is inside the recent window.
    records = (_lc_record(7, "earlier refactor work very recent"),)
    plan = build_long_horizon_plan(
        events,
        session_id="thread-x",
        recent_window=2,
        budget=10,
        learned_context=_loaded(records),
        query=_QUERY,
    )
    assert plan.consolidated_count == 0


def test_long_horizon_plan_ignores_a_stale_learned_context_but_reports_it() -> None:
    """A stale load contributes no records and surfaces the degradation in diagnostics."""
    events = [_ev(i, "tool.call.completed") for i in range(1, 8)]
    stale = LearnedContextLoad(projection=None, stale=True, reason="covered_head_mismatch", covered_seq=3)
    plan = build_long_horizon_plan(
        events,
        session_id="thread-x",
        recent_window=2,
        budget=10,
        learned_context=stale,
        query=_QUERY,
    )
    assert plan.consolidated_count == 0
    diagnostics = plan.to_diagnostics()["learned_context"]
    assert diagnostics["stale"] is True
    assert diagnostics["available"] is False
    assert diagnostics["reason"] == "covered_head_mismatch"


def test_long_horizon_plan_skips_records_when_there_is_no_query() -> None:
    """Projection consumption is query-routed: with no query there is no routing signal."""
    events = [_ev(i, "tool.call.completed") for i in range(1, 8)]
    plan = build_long_horizon_plan(
        events,
        session_id="thread-x",
        recent_window=2,
        budget=10,
        learned_context=_loaded((_lc_record(2, "earlier refactor work"),)),
        query="",
    )
    assert plan.consolidated_count == 0
    assert plan.to_diagnostics()["learned_context"]["available"] is True


def test_long_horizon_plan_budget_keeps_the_highest_scoring_items() -> None:
    """The budget bites AFTER span ranking, so it keeps the most relevant history."""
    events = [_ev(i, "tool.call.completed") for i in range(1, 12)]
    # IDENTICAL text, so query overlap and span coverage tie exactly and the
    # only thing separating them is horizon proximity — the I3 term. Replay
    # order puts the deep record first, so anything that ranks by the old
    # authoring confidence (a tie at the neutral prior) or skips ranking
    # altogether keeps seq 2 instead.
    records = (
        _lc_record(2, "earlier refactor work on the query engine"),
        _lc_record(8, "earlier refactor work on the query engine"),
    )
    plan = build_long_horizon_plan(
        events,
        session_id="thread-x",
        recent_window=2,
        budget=1,
        learned_context=_loaded(records),
        query=_QUERY,
    )
    assert plan.consolidated_count == 1
    # seq 8 is both nearer the split and a query match, so it must win the single slot.
    assert plan.consolidated_contexts[0].metadata["candidate_id"].endswith(":8")


def test_long_horizon_plan_learned_context_short_session_reports_diagnostics() -> None:
    """A session inside the window still reports the learned-context stanza."""
    events = [_ev(i, "tool.call.completed") for i in range(1, 3)]
    plan = build_long_horizon_plan(
        events,
        session_id="thread-x",
        recent_window=10,
        budget=10,
        learned_context=_loaded((_lc_record(1, "earlier refactor work"),)),
        query=_QUERY,
    )
    assert plan.consolidated_count == 0
    assert plan.to_diagnostics()["learned_context"]["available"] is True


@pytest.mark.asyncio
async def test_learned_context_default_off_is_byte_identical(tmp_path) -> None:
    """With learned_context off, a present and VALID artifact changes nothing at all.

    This is the load-bearing default-off guarantee: the gate is checked before
    the artifact is even looked at, so an operator who ran a crystallization pass
    sees byte-identical checkouts until they opt in.
    """
    fabric = await _build_fabric(tmp_path, enabled=True)
    sid = "long-thread"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    recall_limit = _checkout_recall_limit(_QUERY, 10)

    async def _checkout():
        assembly = await fabric.assemble_context(
            _QUERY,
            session_id=sid,
            limit=10,
            recall_limit=recall_limit,
            max_recent_events=20,
            long_horizon=True,
        )
        return assembly, build_memory_checkout(
            query=_QUERY,
            assembly=assembly,
            now=now,
            retrieval_profile=fabric.retrieval_profile,
            salience_floor=fabric._salience_floor,
            salience_half_life_days=fabric._salience_half_life_days,
        )

    try:
        # No consolidation history at all: without I2 the remote tier is empty,
        # which is exactly the degradation I2 exists to fix.
        await _seed_old_actionable(fabric, sid, 20)
        await _seed_recent(fabric, sid, 20)
        eventlog = fabric.session_manager.get(sid).eventlog
        head = eventlog.read_all()[-1]

        # A genuinely valid artifact: query-matching records over OLD history,
        # vouched for by a real build event, covering a head that still verifies.
        projection = _lc_projection((_lc_record(2, "earlier refactor work on the parser"),))
        artifact = learned_context_path(fabric.eventloom_path, sid)
        write_learned_context(
            projection, artifact, session_id=sid, covered_seq=head.seq, covered_hash=head.hash
        )
        await fabric.append(
            LEARNED_CONTEXT_EVENT_TYPE,
            "zaxy-crystallizer",
            payload=build_projection_built_payload(
                projection,
                session_id=sid,
                covered_seq=head.seq,
                covered_hash=head.hash,
                artifact_path=str(artifact),
            ),
            session_id=sid,
        )
        assert artifact.exists()

        fabric.settings = Settings(long_horizon_enabled=True, long_horizon_recent_window=_RECENT_WINDOW, learned_context_enabled=False)
        assembly_off, checkout_off = await _checkout()

        # Same fabric, same log, same artifact — only the gate moves.
        fabric.settings = Settings(long_horizon_enabled=True, long_horizon_recent_window=_RECENT_WINDOW, learned_context_enabled=True)
        assembly_on, checkout_on = await _checkout()
    finally:
        await fabric.close()

    # OFF: the artifact is invisible. No stanza, no learned-context lane items.
    assert "learned_context" not in checkout_off.diagnostics["long_horizon"]
    # Without I2 the remote tier has no source at all here.
    assert checkout_off.diagnostics["long_horizon"]["consolidated_count"] == 0
    assert assembly_off.long_horizon is not None
    assert "learned_context" not in assembly_off.long_horizon
    assert all(
        not (context.metadata or {}).get("learned_context")
        for context in assembly_off.long_horizon_contexts
    )

    # ON: the same artifact is consumed, proving OFF was suppressing real work
    # rather than there being nothing to suppress.
    stanza = checkout_on.diagnostics["long_horizon"]["learned_context"]
    assert stanza["available"] is True
    assert stanza["stale"] is False
    assert stanza["projection_id"] == projection.projection_id
    surfaced = [
        context
        for context in assembly_on.long_horizon_contexts
        if (context.metadata or {}).get("learned_context")
    ]
    assert surfaced
    for context in surfaced:
        assert context.metadata["non_authoritative"] is True
        for citation in context.metadata["source_event_citations"]:
            assert _CITATION_RE.match(citation)
    assert checkout_on.to_dict() != checkout_off.to_dict()


@pytest.mark.asyncio
async def test_learned_context_stale_artifact_falls_back_and_is_visible(tmp_path) -> None:
    """A tampered covered head makes checkout ignore the artifact and say so."""
    fabric = await _build_fabric(tmp_path, enabled=True)
    sid = "long-thread"
    try:
        await _seed_long_thread_with_consolidation(fabric, sid)
        eventlog = fabric.session_manager.get(sid).eventlog
        head = eventlog.read_all()[-1]

        projection = _lc_projection((_lc_record(2, "earlier refactor work on the parser"),))
        artifact = learned_context_path(fabric.eventloom_path, sid)
        # Covered head claims a hash the log does not carry at that seq.
        write_learned_context(
            projection, artifact, session_id=sid, covered_seq=head.seq, covered_hash="f" * 64
        )
        await fabric.append(
            LEARNED_CONTEXT_EVENT_TYPE,
            "zaxy-crystallizer",
            payload=build_projection_built_payload(
                projection,
                session_id=sid,
                covered_seq=head.seq,
                covered_hash="f" * 64,
                artifact_path=str(artifact),
            ),
            session_id=sid,
        )

        fabric.settings = Settings(long_horizon_enabled=True, long_horizon_recent_window=_RECENT_WINDOW, learned_context_enabled=True)
        checkout = await fabric.checkout_memory(
            _QUERY,
            session_id=sid,
            long_horizon=True,
            max_recent_events=_RECENT_WINDOW,
            record_reinforcement=False,
        )
    finally:
        await fabric.close()

    stanza = checkout.diagnostics["long_horizon"]["learned_context"]
    assert stanza["stale"] is True
    assert stanza["available"] is False
    assert stanza["reason"] == "covered_head_mismatch"
    assert stanza["record_count"] == 0
    # Fell back to today's behaviour: the accepted-candidate tier still works.
    assert checkout.diagnostics["long_horizon"]["consolidated_count"] >= 1
    assert all(
        not item.get("learned_context")
        for item in checkout.diagnostics["long_horizon"]["items"]
    )
