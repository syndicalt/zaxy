from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.core import MemoryFabric
from zaxy.event import Event, EventLog
from zaxy.extract import extract
from zaxy.salience import (
    ENCODING_ENTITY_CORROBORATION_MIN_OVERLAP,
    ENCODING_REDUNDANT_MIN_OVERLAP,
    ENCODING_REINFORCING_MIN_OVERLAP,
    MAX_REINFORCEMENT_WEIGHT,
    REINFORCEMENT_EVENT_TYPE,
    SALIENCE_BASE,
    SALIENCE_MAX,
    SALIENCE_MIN,
    SALIENCE_REINFORCEMENT_MULTIPLIERS,
    EncodingDecision,
    EventRef,
    SalienceLedger,
    build_confirmed_reinforcement_event,
    build_cue_record,
    build_invalidated_reinforcement_event,
    build_promoted_reinforcement_event,
    build_reinforcement_event,
    build_surfaced_reinforcement_event,
    classify_append,
    cue_overlap,
    cue_pairs,
    event_ref_index,
    reinforcement_targets_from_citations,
    resolve_citation_target,
    target_ref,
)
from zaxy.verbatim import VerbatimIndex

T0 = datetime(2026, 6, 1, tzinfo=UTC)
TARGET = {"seq": 7, "hash": "a" * 64}
OTHER_TARGET = {"seq": 8, "hash": "b" * 64}
SOURCE = {"checkout_id": "checkout:0001"}
REF = EventRef(seq=7, hash="a" * 64)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _logged(spec: dict[str, Any], *, at: datetime) -> dict[str, Any]:
    """Wrap a builder spec as a replayed log event mapping."""
    return {
        "type": spec["event_type"],
        "actor": spec["actor"],
        "thread": spec["thread"],
        "payload": spec["payload"],
        "timestamp": _iso(at),
    }


def _confirmed_at(at: datetime, targets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    spec = build_confirmed_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        feedback_id="feedback:0001",
        targets=targets or [TARGET],
    )
    return _logged(spec, at=at)


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def test_reinforcement_event_spec_is_append_ready_and_non_authoritative() -> None:
    event = build_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        kind="surfaced",
        targets=[TARGET, OTHER_TARGET],
        source=SOURCE,
    )

    assert event["event_type"] == "memory.reinforcement"
    assert event["thread"] == "agent-1"
    assert event["payload"]["kind"] == "surfaced"
    assert event["payload"]["targets"] == [TARGET, OTHER_TARGET]
    assert event["payload"]["source"] == SOURCE
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert "weight" not in event["payload"]


def test_reinforcement_event_normalizes_kind_and_keeps_weight_override() -> None:
    event = build_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        kind=" Confirmed ",
        targets=[TARGET],
        source={"feedback_id": "feedback:0001"},
        weight=2.5,
    )

    assert event["payload"]["kind"] == "confirmed"
    assert event["payload"]["weight"] == 2.5


@pytest.mark.parametrize("kind", ["boosted", "", " ", 3])
def test_reinforcement_event_rejects_invalid_kind(kind: object) -> None:
    with pytest.raises(ValueError, match="kind"):
        build_reinforcement_event(
            actor="agent",
            session_id="agent-1",
            kind=kind,  # type: ignore[arg-type]
            targets=[TARGET],
            source=SOURCE,
        )


@pytest.mark.parametrize("targets", ["not-a-list", []])
def test_reinforcement_event_rejects_non_sequence_or_empty_targets(targets: object) -> None:
    with pytest.raises(ValueError, match="targets"):
        build_reinforcement_event(
            actor="agent",
            session_id="agent-1",
            kind="surfaced",
            targets=targets,  # type: ignore[arg-type]
            source=SOURCE,
        )


@pytest.mark.parametrize(
    "target",
    [
        {"seq": 0, "hash": "a" * 64},
        {"seq": -1, "hash": "a" * 64},
        {"seq": True, "hash": "a" * 64},
        {"seq": "7", "hash": "a" * 64},
        {"seq": 7, "hash": "A" * 64},
        {"seq": 7, "hash": "a" * 63},
        {"seq": 7},
        "eventloom://agent-1/events/7#aaaaaaaaaaaa",
    ],
)
def test_reinforcement_event_rejects_malformed_target_refs(target: object) -> None:
    with pytest.raises(ValueError, match="targets"):
        build_reinforcement_event(
            actor="agent",
            session_id="agent-1",
            kind="surfaced",
            targets=[target],  # type: ignore[list-item]
            source=SOURCE,
        )


def test_reinforcement_event_rejects_duplicate_target_refs() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        build_reinforcement_event(
            actor="agent",
            session_id="agent-1",
            kind="surfaced",
            targets=[TARGET, dict(TARGET)],
            source=SOURCE,
        )


@pytest.mark.parametrize(
    "source",
    [
        {},
        "checkout:0001",
        {"checkout_id": ""},
        {"checkout_id": 7},
        {"": "checkout:0001"},
    ],
)
def test_reinforcement_event_rejects_invalid_source(source: object) -> None:
    with pytest.raises(ValueError, match="source"):
        build_reinforcement_event(
            actor="agent",
            session_id="agent-1",
            kind="surfaced",
            targets=[TARGET],
            source=source,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "weight",
    [0.0, -1.0, math.inf, math.nan, True, MAX_REINFORCEMENT_WEIGHT + 0.1],
)
def test_reinforcement_event_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        build_reinforcement_event(
            actor="agent",
            session_id="agent-1",
            kind="surfaced",
            targets=[TARGET],
            source=SOURCE,
            weight=weight,
        )


def test_reinforcement_event_rejects_empty_required_text() -> None:
    with pytest.raises(ValueError, match="actor"):
        build_reinforcement_event(
            actor=" ",
            session_id="agent-1",
            kind="surfaced",
            targets=[TARGET],
            source=SOURCE,
        )


def test_builders_snapshot_mutable_targets_and_source() -> None:
    targets = [{"seq": 7, "hash": "a" * 64}]
    source = {"checkout_id": "checkout:0001"}

    event = build_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        kind="surfaced",
        targets=targets,
        source=source,
    )
    targets[0]["hash"] = "b" * 64
    source["checkout_id"] = "checkout:0002"

    assert event["payload"]["targets"] == [TARGET]
    assert event["payload"]["source"] == SOURCE


def test_kind_specific_builders_fill_kind_and_source() -> None:
    surfaced = build_surfaced_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        checkout_id="checkout:0001",
        targets=[TARGET, OTHER_TARGET],
    )
    confirmed = build_confirmed_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        feedback_id="feedback:0001",
        targets=[TARGET],
    )
    promoted = build_promoted_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        promotion_id="promotion:0001",
        targets=[TARGET],
    )
    invalidated = build_invalidated_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        invalidation_id="invalidation:0001",
        targets=[TARGET],
    )

    assert surfaced["payload"]["kind"] == "surfaced"
    assert surfaced["payload"]["source"] == {"checkout_id": "checkout:0001"}
    assert surfaced["payload"]["targets"] == [TARGET, OTHER_TARGET]
    assert confirmed["payload"]["kind"] == "confirmed"
    assert confirmed["payload"]["source"] == {"feedback_id": "feedback:0001"}
    assert promoted["payload"]["kind"] == "promoted"
    assert promoted["payload"]["source"] == {"promotion_id": "promotion:0001"}
    assert invalidated["payload"]["kind"] == "invalidated"
    assert invalidated["payload"]["source"] == {"invalidation_id": "invalidation:0001"}


def test_built_spec_round_trips_through_event_log(tmp_path: Path) -> None:
    spec = build_surfaced_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        checkout_id="checkout:0001",
        targets=[TARGET, OTHER_TARGET],
    )
    log = EventLog(tmp_path / "agent.jsonl")
    log.append(
        spec["event_type"],
        actor=spec["actor"],
        payload=spec["payload"],
        thread=spec["thread"],
        timestamp=T0,
    )

    assert log.verify().ok
    sealed = log.read_all()
    assert sealed[0].type == "memory.reinforcement"
    assert sealed[0].payload["targets"] == [TARGET, OTHER_TARGET]

    state = SalienceLedger().replay(sealed, now=T0)
    assert set(state) == {REF, EventRef(seq=8, hash="b" * 64)}
    assert state[REF].score == pytest.approx(SALIENCE_REINFORCEMENT_MULTIPLIERS["surfaced"])


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------


def test_replay_is_deterministic_for_same_events_and_now() -> None:
    events = [
        _confirmed_at(T0),
        _logged(
            build_surfaced_reinforcement_event(
                actor="agent",
                session_id="agent-1",
                checkout_id="checkout:0001",
                targets=[TARGET, OTHER_TARGET],
            ),
            at=T0 + timedelta(days=1),
        ),
    ]
    ledger = SalienceLedger()
    now = T0 + timedelta(days=3)

    assert ledger.replay(events, now=now) == ledger.replay(list(events), now=now)


def test_batched_surfaced_event_applies_to_every_listed_target() -> None:
    third = {"seq": 9, "hash": "c" * 64}
    spec = build_surfaced_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        checkout_id="checkout:0001",
        targets=[TARGET, OTHER_TARGET, third],
    )
    state = SalienceLedger().replay([_logged(spec, at=T0)], now=T0)

    assert len(state) == 3
    for target in (TARGET, OTHER_TARGET, third):
        entry = state[EventRef(seq=target["seq"], hash=target["hash"])]  # type: ignore[arg-type]
        assert entry.reinforcement_counts["surfaced"] == 1
        assert entry.score == pytest.approx(SALIENCE_REINFORCEMENT_MULTIPLIERS["surfaced"])


def test_score_halves_per_half_life() -> None:
    ledger = SalienceLedger()
    events = [_confirmed_at(T0)]

    fresh = ledger.replay(events, now=T0)[REF].score
    after_one = ledger.replay(events, now=T0 + timedelta(days=30))[REF].score
    after_two = ledger.replay(events, now=T0 + timedelta(days=60))[REF].score

    assert fresh == pytest.approx(1.5)
    assert after_one == pytest.approx(fresh / 2, rel=1e-9)
    assert after_two == pytest.approx(fresh / 4, rel=1e-9)


def test_half_life_is_overridable() -> None:
    ledger = SalienceLedger(half_life_days=1.0)
    events = [_confirmed_at(T0)]

    assert ledger.replay(events, now=T0 + timedelta(days=1))[REF].score == pytest.approx(
        0.75, rel=1e-9
    )


@pytest.mark.parametrize("half_life_days", [0.0, -1.0, math.nan, math.inf, True])
def test_ledger_rejects_invalid_half_life(half_life_days: float) -> None:
    with pytest.raises(ValueError, match="half_life_days"):
        SalienceLedger(half_life_days=half_life_days)


def test_invalidation_dominates_prior_confirmation() -> None:
    spec = build_invalidated_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        invalidation_id="invalidation:0001",
        targets=[TARGET],
    )
    events = [_confirmed_at(T0), _logged(spec, at=T0)]
    state = SalienceLedger().replay(events, now=T0)

    assert state[REF].score == pytest.approx(1.5 * 0.2)
    assert state[REF].score < SALIENCE_BASE


def test_score_is_clamped_at_the_ceiling() -> None:
    events = [
        _logged(
            build_promoted_reinforcement_event(
                actor="agent",
                session_id="agent-1",
                promotion_id=f"promotion:{index:04d}",
                targets=[TARGET],
            ),
            at=T0,
        )
        for index in range(10)
    ]
    state = SalienceLedger().replay(events, now=T0)

    assert state[REF].reinforcement_factor == SALIENCE_MAX
    assert state[REF].score == SALIENCE_MAX


def test_score_is_clamped_at_the_floor_under_decay() -> None:
    state = SalienceLedger().replay(
        [_confirmed_at(T0)],
        now=T0 + timedelta(days=30_000),
    )

    assert state[REF].score == SALIENCE_MIN


def test_repeated_invalidations_floor_the_reinforcement_factor() -> None:
    events = [
        _logged(
            build_invalidated_reinforcement_event(
                actor="agent",
                session_id="agent-1",
                invalidation_id=f"invalidation:{index:04d}",
                targets=[TARGET],
            ),
            at=T0,
        )
        for index in range(5)
    ]
    state = SalienceLedger().replay(events, now=T0)

    assert state[REF].reinforcement_factor == SALIENCE_MIN
    assert state[REF].score == SALIENCE_MIN
    assert state[REF].reinforcement_counts["invalidated"] == 5


def test_now_before_last_reinforcement_never_inflates_the_score() -> None:
    skewed_now = T0 - timedelta(days=5)
    state = SalienceLedger().replay([_confirmed_at(T0)], now=skewed_now)

    assert state[REF].recency_factor == 1.0
    assert state[REF].score == pytest.approx(1.5)


def test_replay_and_apply_require_timezone_aware_now() -> None:
    ledger = SalienceLedger()
    naive = datetime(2026, 6, 1)  # noqa: DTZ001 - intentionally naive

    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.replay([], now=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.apply({}, _confirmed_at(T0), now=naive)


def test_replay_ignores_unrelated_and_malformed_events() -> None:
    bad_targets = _confirmed_at(T0)
    bad_targets["payload"] = dict(bad_targets["payload"])
    bad_targets["payload"]["targets"] = [{"seq": 7, "hash": "not-hex"}]
    bad_weight = _confirmed_at(T0)
    bad_weight["payload"] = dict(bad_weight["payload"])
    bad_weight["payload"]["weight"] = "heavy"
    duplicate_targets = _confirmed_at(T0)
    duplicate_targets["payload"] = dict(duplicate_targets["payload"])
    duplicate_targets["payload"]["targets"] = [TARGET, dict(TARGET)]
    bad_timestamp = _confirmed_at(T0)
    bad_timestamp["timestamp"] = "yesterday"

    events: list[object] = [
        object(),
        {"type": "memory.feedback", "payload": {"memory": "m"}, "timestamp": _iso(T0)},
        {"type": "memory.reinforcement", "payload": "not-a-payload", "timestamp": _iso(T0)},
        bad_targets,
        bad_weight,
        duplicate_targets,
        bad_timestamp,
    ]

    assert SalienceLedger().replay(events, now=T0) == {}


def test_apply_skips_malformed_events_identically_to_replay() -> None:
    ledger = SalienceLedger()
    state = ledger.replay([_confirmed_at(T0)], now=T0)
    malformed = {
        "type": "memory.reinforcement",
        "payload": {"kind": "boosted"},
        "timestamp": _iso(T0),
    }

    assert ledger.apply(state, malformed, now=T0) == state


def test_apply_does_not_mutate_the_input_state() -> None:
    ledger = SalienceLedger()
    state = ledger.replay([_confirmed_at(T0)], now=T0)
    snapshot = dict(state)

    advanced = ledger.apply(
        state, _confirmed_at(T0 + timedelta(days=1)), now=T0 + timedelta(days=1)
    )

    assert state == snapshot
    assert advanced[REF].reinforcement_counts["confirmed"] == 2


def test_weight_override_replaces_the_table_multiplier() -> None:
    spec = build_surfaced_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        checkout_id="checkout:0001",
        targets=[TARGET],
        weight=3.0,
    )
    state = SalienceLedger().replay([_logged(spec, at=T0)], now=T0)

    assert state[REF].score == pytest.approx(3.0)


def test_counts_and_last_reinforced_at_track_the_latest_event() -> None:
    surfaced = build_surfaced_reinforcement_event(
        actor="agent",
        session_id="agent-1",
        checkout_id="checkout:0001",
        targets=[TARGET],
    )
    later = T0 + timedelta(days=1)
    state = SalienceLedger().replay(
        [_logged(surfaced, at=T0), _confirmed_at(later)],
        now=later,
    )

    assert state[REF].reinforcement_counts == {
        "surfaced": 1,
        "confirmed": 1,
        "promoted": 0,
        "invalidated": 0,
    }
    assert state[REF].last_reinforced_at == later


def test_composition_breakdown_explains_the_score() -> None:
    later = T0 + timedelta(days=15)
    state = SalienceLedger().replay([_confirmed_at(T0)], now=later)
    composition = state[REF].composition()

    recomposed = composition["base"] * composition["reinforcement_factor"]
    recomposed *= composition["recency_factor"]
    assert composition["score"] == pytest.approx(min(max(recomposed, SALIENCE_MIN), SALIENCE_MAX))
    assert composition["last_reinforced_at"] == "2026-06-01T00:00:00Z"
    assert composition["reinforcement_counts"]["confirmed"] == 1


# ----------------------------------------------------------------------
# Replay == incremental fold
# ----------------------------------------------------------------------


def _seeded_event_sequence(rng: random.Random, count: int) -> list[dict[str, Any]]:
    """Build a randomized-but-seeded reinforcement sequence over 8 targets."""
    targets = [{"seq": index + 1, "hash": format(index + 1, "064x")} for index in range(8)]
    builders_by_kind = ("surfaced", "confirmed", "promoted", "invalidated")
    events: list[dict[str, Any]] = []
    for index in range(count):
        chosen = rng.sample(targets, rng.randint(1, 4))
        weight = rng.choice([None, None, None, round(rng.uniform(0.1, 9.9), 3)])
        spec = build_reinforcement_event(
            actor="agent",
            session_id="agent-1",
            kind=rng.choice(builders_by_kind),
            targets=chosen,
            source={"origin_id": f"origin:{index:04d}"},
            weight=weight,
        )
        events.append(_logged(spec, at=T0 + timedelta(hours=index)))
    return events


def test_replay_equals_fold_of_applies_on_seeded_random_sequence() -> None:
    rng = random.Random(20260610)
    events = _seeded_event_sequence(rng, count=60)
    ledger = SalienceLedger()
    now = T0 + timedelta(days=10)

    folded: dict[EventRef, Any] = {}
    for event in events:
        folded = ledger.apply(folded, event, now=now)

    assert ledger.replay(events, now=now) == folded


def test_incremental_applies_extend_a_replayed_prefix() -> None:
    rng = random.Random(424242)
    events = _seeded_event_sequence(rng, count=40)
    ledger = SalienceLedger()
    now = T0 + timedelta(days=10)

    state = ledger.replay(events[:25], now=now)
    for event in events[25:]:
        state = ledger.apply(state, event, now=now)

    assert state == ledger.replay(events, now=now)


# ----------------------------------------------------------------------
# Citation canonicalization helpers
# ----------------------------------------------------------------------


def _sealed_log(tmp_path: Path, name: str = "agent-1") -> EventLog:
    return EventLog(tmp_path / ".eventloom" / f"{name}.jsonl")


def test_event_ref_index_maps_seq_to_full_hash_and_type(tmp_path: Path) -> None:
    log = _sealed_log(tmp_path)
    first = log.append("decision.made", actor="dev", payload={"decision": "Adopt salience"})
    second = log.append("task.completed", actor="dev", payload={"task": "wire emitters"})

    index = event_ref_index(log.read_all())

    assert index == {
        1: (first.hash, "decision.made"),
        2: (second.hash, "task.completed"),
    }


def test_resolve_citation_target_resolves_short_and_full_hash_fragments(tmp_path: Path) -> None:
    log = _sealed_log(tmp_path)
    event = log.append("decision.made", actor="dev", payload={"decision": "Adopt salience"})
    index = event_ref_index(log.read_all())
    short = f"eventloom://agent-1/events/{event.seq}#{event.hash[:12]}"
    full = f"eventloom://agent-1/events/{event.seq}#{event.hash}"

    assert resolve_citation_target(short, event_index=index) == EventRef(event.seq, event.hash)
    assert resolve_citation_target(full, event_index=index) == EventRef(event.seq, event.hash)


def test_resolve_citation_target_rejects_unresolvable_and_mismatched_refs(tmp_path: Path) -> None:
    log = _sealed_log(tmp_path)
    event = log.append("decision.made", actor="dev", payload={"decision": "Adopt salience"})
    index = event_ref_index(log.read_all())
    wrong_fragment = "f" * 12 if not event.hash.startswith("f" * 12) else "0" * 12

    assert resolve_citation_target(None, event_index=index) is None
    assert resolve_citation_target("file://notes.md:3", event_index=index) is None
    assert resolve_citation_target("eventloom://agent-1/events/99#" + "a" * 12, event_index=index) is None
    assert (
        resolve_citation_target(
            f"eventloom://agent-1/events/{event.seq}#{wrong_fragment}",
            event_index=index,
        )
        is None
    )


def test_reinforcement_events_are_never_reinforcement_targets(tmp_path: Path) -> None:
    log = _sealed_log(tmp_path)
    memory = log.append("decision.made", actor="dev", payload={"decision": "Adopt salience"})
    spec = build_surfaced_reinforcement_event(
        actor="zaxy-memory",
        session_id="agent-1",
        checkout_id="checkout:0001",
        targets=[{"seq": memory.seq, "hash": memory.hash}],
    )
    reinforcement = log.append(
        spec["event_type"], actor=spec["actor"], payload=spec["payload"], thread=spec["thread"]
    )
    index = event_ref_index(log.read_all())
    reinforcement_citation = f"eventloom://agent-1/events/{reinforcement.seq}#{reinforcement.hash}"

    assert resolve_citation_target(reinforcement_citation, event_index=index) is None
    assert reinforcement_targets_from_citations(
        [reinforcement_citation, f"eventloom://agent-1/events/{memory.seq}#{memory.hash[:12]}"],
        event_index=index,
    ) == [{"seq": memory.seq, "hash": memory.hash}]


def test_reinforcement_targets_from_citations_dedupes_across_fragment_lengths(tmp_path: Path) -> None:
    log = _sealed_log(tmp_path)
    event = log.append("decision.made", actor="dev", payload={"decision": "Adopt salience"})
    index = event_ref_index(log.read_all())

    targets = reinforcement_targets_from_citations(
        [
            f"eventloom://agent-1/events/{event.seq}#{event.hash[:12]}",
            f"eventloom://agent-1/events/{event.seq}#{event.hash}",
            "not-a-citation",
        ],
        event_index=index,
    )

    assert targets == [{"seq": event.seq, "hash": event.hash}]


def test_target_ref_requires_sealed_full_hash_provenance() -> None:
    assert target_ref(7, "a" * 64) == {"seq": 7, "hash": "a" * 64}
    assert target_ref(0, "a" * 64) is None
    assert target_ref(7, "a" * 12) is None
    assert target_ref(None, None) is None


# ----------------------------------------------------------------------
# Wiring: emitters and diagnostics (fabric-level, no ranking change)
# ----------------------------------------------------------------------


def _wired_fabric(tmp_path: Path) -> MemoryFabric:
    """Real Eventloom + verbatim lane, mocked graph projection lane."""
    with patch("zaxy.core.build_projection_store") as mock_store:
        mock_store.return_value = AsyncMock()
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
    fabric.query_router = MagicMock(query=AsyncMock(return_value=[]))
    fabric._connected = True
    return fabric


def _reinforcement_events(log: EventLog) -> list[Event]:
    return [event for event in log.read_all() if event.type == REINFORCEMENT_EVENT_TYPE]


async def _seed_decision(fabric: MemoryFabric, session_id: str = "agent-1") -> Any:
    return await fabric.append(
        "decision.made",
        actor="dev",
        payload={"decision": "Adopt the salience ledger for memory reinforcement."},
        session_id=session_id,
    )


class TestSurfacedWiring:
    """Checkout emits one batched surfaced event without changing ranking."""

    async def test_checkout_appends_one_batched_surfaced_event_for_packet_refs(
        self, tmp_path: Path
    ) -> None:
        fabric = _wired_fabric(tmp_path)
        seeded = await _seed_decision(fabric)

        checkout = await fabric.checkout_memory(
            "salience ledger decision", session_id="agent-1", limit=3
        )

        log = fabric.session_manager.get("agent-1").eventlog
        reinforcements = _reinforcement_events(log)
        assert len(reinforcements) == 1
        payload = reinforcements[0].payload
        assert payload["kind"] == "surfaced"
        assert payload["authority_status"] == "non_authoritative"
        assert payload["source"]["checkout_id"].startswith("eventloom://agent-1/events/")
        # Targets are exactly the packet's resolvable refs — here the seeded decision.
        assert {"seq": seeded.seq, "hash": seeded.hash} in payload["targets"]
        packet_index = event_ref_index(log.read_all())
        packet_refs = {
            resolve_citation_target(item.get("citation"), event_index=packet_index)
            for item in [*checkout.current_facts, *checkout.evidence]
        }
        packet_refs.discard(None)
        assert {EventRef(t["seq"], t["hash"]) for t in payload["targets"]} == packet_refs

    async def test_repeated_checkouts_accumulate_events_and_converge_scores(
        self, tmp_path: Path
    ) -> None:
        fabric = _wired_fabric(tmp_path)
        seeded = await _seed_decision(fabric)

        await fabric.checkout_memory("salience ledger decision", session_id="agent-1", limit=3)
        await fabric.checkout_memory("salience ledger decision", session_id="agent-1", limit=3)

        log = fabric.session_manager.get("agent-1").eventlog
        reinforcements = _reinforcement_events(log)
        assert len(reinforcements) == 2
        state = SalienceLedger().replay(log.read_all(), now=datetime.now(UTC))
        entry = state[EventRef(seq=seeded.seq, hash=seeded.hash)]
        assert entry.reinforcement_counts["surfaced"] == 2
        assert entry.reinforcement_factor == pytest.approx(1.05**2)

    async def test_checkout_ranking_and_selection_unchanged_by_reinforcement_events(
        self, tmp_path: Path
    ) -> None:
        """The salience-events-absent baseline (first checkout) must match later packets."""
        fabric = _wired_fabric(tmp_path)
        await _seed_decision(fabric)
        await fabric.append(
            "task.completed",
            actor="dev",
            payload={"task": "Document the salience reinforcement multipliers."},
            session_id="agent-1",
        )

        baseline = await fabric.checkout_memory(
            "salience reinforcement", session_id="agent-1", limit=3
        )
        assert _reinforcement_events(fabric.session_manager.get("agent-1").eventlog)
        with_salience_events = await fabric.checkout_memory(
            "salience reinforcement", session_id="agent-1", limit=3
        )

        assert with_salience_events.current_facts == baseline.current_facts
        assert with_salience_events.evidence == baseline.evidence
        assert with_salience_events.provenance == baseline.provenance
        assert "salience" not in baseline.diagnostics
        assert "salience" in with_salience_events.diagnostics

    async def test_no_emission_when_nothing_resolvable_surfaced(self, tmp_path: Path) -> None:
        fabric = _wired_fabric(tmp_path)

        await fabric.checkout_memory("anything at all", session_id="agent-1", limit=3)

        assert _reinforcement_events(fabric.session_manager.get("agent-1").eventlog) == []

    async def test_read_only_checkout_opts_out_of_reinforcement(self, tmp_path: Path) -> None:
        """Read-only inspection surfaces (dashboard) must not write to the log."""
        fabric = _wired_fabric(tmp_path)
        await _seed_decision(fabric)
        log = fabric.session_manager.get("agent-1").eventlog
        before = len(log.read_all())

        checkout = await fabric.checkout_memory(
            "salience ledger decision",
            session_id="agent-1",
            limit=3,
            record_reinforcement=False,
        )

        assert checkout.current_facts
        assert len(log.read_all()) == before
        assert _reinforcement_events(log) == []


class TestDiagnosticsWiring:
    """Salience composition appears in checkout diagnostics only."""

    async def test_checkout_diagnostics_expose_salience_composition_for_surfaced_refs(
        self, tmp_path: Path
    ) -> None:
        fabric = _wired_fabric(tmp_path)
        seeded = await _seed_decision(fabric)

        await fabric.checkout_memory("salience ledger decision", session_id="agent-1", limit=3)
        checkout = await fabric.checkout_memory(
            "salience ledger decision", session_id="agent-1", limit=3
        )

        salience = checkout.diagnostics["salience"]
        assert salience["authority_status"] == "non_authoritative"
        assert salience["half_life_days"] == 30.0
        assert salience["scored_count"] == len(salience["items"]) >= 1
        item = next(entry for entry in salience["items"] if entry["seq"] == seeded.seq)
        assert item["hash"] == seeded.hash
        composition = item["composition"]
        assert composition["reinforcement_counts"]["surfaced"] == 1
        assert composition["base"] == SALIENCE_BASE
        assert composition["score"] == pytest.approx(
            min(
                max(
                    composition["base"]
                    * composition["reinforcement_factor"]
                    * composition["recency_factor"],
                    SALIENCE_MIN,
                ),
                SALIENCE_MAX,
            )
        )
        assert "- Salience:" in checkout.prompt

    async def test_first_checkout_diagnostics_omit_salience_until_events_exist(
        self, tmp_path: Path
    ) -> None:
        fabric = _wired_fabric(tmp_path)
        await _seed_decision(fabric)

        checkout = await fabric.checkout_memory(
            "salience ledger decision", session_id="agent-1", limit=3
        )

        assert "salience" not in checkout.diagnostics


class TestConfirmedWiring:
    """Positive feedback emits confirmed reinforcement; negative emits none."""

    async def test_positive_feedback_appends_confirmed_reinforcement(self, tmp_path: Path) -> None:
        from zaxy.core import Context

        fabric = _wired_fabric(tmp_path)
        seeded = await _seed_decision(fabric)
        context = Context(
            content="Adopt the salience ledger for memory reinforcement.",
            source="verbatim",
            score=0.9,
            metadata={
                "entity_name": "salience ledger",
                "entity_type": "decision",
                "citation": f"eventloom://agent-1/events/{seeded.seq}#{seeded.hash[:12]}",
            },
        )

        await fabric.record_context_feedback([context], feedback="used", session_id="agent-1")

        log = fabric.session_manager.get("agent-1").eventlog
        events = log.read_all()
        reinforcements = _reinforcement_events(log)
        assert [event.type for event in events[-2:]] == ["memory.reinforced", REINFORCEMENT_EVENT_TYPE]
        assert len(reinforcements) == 1
        payload = reinforcements[0].payload
        assert payload["kind"] == "confirmed"
        assert payload["targets"] == [{"seq": seeded.seq, "hash": seeded.hash}]
        feedback_event = events[-2]
        assert payload["source"]["feedback_id"] == (
            f"eventloom://agent-1/events/{feedback_event.seq}#{feedback_event.hash[:12]}"
        )

    @pytest.mark.parametrize("feedback", ["irrelevant", "Irrelevant "])
    async def test_non_positive_feedback_appends_no_reinforcement(
        self, tmp_path: Path, feedback: str
    ) -> None:
        from zaxy.core import Context

        fabric = _wired_fabric(tmp_path)
        seeded = await _seed_decision(fabric)
        context = Context(
            content="Adopt the salience ledger for memory reinforcement.",
            source="verbatim",
            score=0.9,
            metadata={"citation": f"eventloom://agent-1/events/{seeded.seq}#{seeded.hash[:12]}"},
        )

        await fabric.record_context_feedback([context], feedback=feedback, session_id="agent-1")

        log = fabric.session_manager.get("agent-1").eventlog
        assert log.read_all()[-1].type == "memory.feedback"
        assert _reinforcement_events(log) == []


class TestInvalidatedWiring:
    """Explicit invalidation emits negative reinforcement at the fabric level."""

    async def test_invalidate_appends_invalidated_reinforcement(self, tmp_path: Path) -> None:
        fabric = _wired_fabric(tmp_path)
        seeded = await _seed_decision(fabric)
        fabric.graph.search_exact = AsyncMock(
            return_value=[
                SimpleNamespace(
                    properties={
                        "source_event_seq": seeded.seq,
                        "source_event_hash": seeded.hash,
                    }
                )
            ]
        )

        await fabric.invalidate(
            "salience ledger", "decision", "2026-06-10T00:00:00Z", session_id="agent-1"
        )

        fabric.graph.invalidate_entity.assert_awaited_once()
        log = fabric.session_manager.get("agent-1").eventlog
        reinforcements = _reinforcement_events(log)
        assert len(reinforcements) == 1
        payload = reinforcements[0].payload
        assert payload["kind"] == "invalidated"
        assert payload["targets"] == [{"seq": seeded.seq, "hash": seeded.hash}]
        assert payload["source"]["invalidation_id"] == (
            "invalidate:decision:salience ledger@2026-06-10T00:00:00Z"
        )

    async def test_invalidate_without_provenance_emits_nothing_and_still_invalidates(
        self, tmp_path: Path
    ) -> None:
        fabric = _wired_fabric(tmp_path)
        await _seed_decision(fabric)
        fabric.graph.search_exact = AsyncMock(return_value=[])

        await fabric.invalidate(
            "salience ledger", "decision", "2026-06-10T00:00:00Z", session_id="agent-1"
        )

        fabric.graph.invalidate_entity.assert_awaited_once()
        assert _reinforcement_events(fabric.session_manager.get("agent-1").eventlog) == []


class TestReinforcementIsolation:
    """Reinforcement events never pollute extraction or source recall."""

    def test_extraction_creates_no_entities_from_reinforcement_events(self, tmp_path: Path) -> None:
        log = _sealed_log(tmp_path)
        memory = log.append("decision.made", actor="dev", payload={"decision": "Adopt salience"})
        spec = build_surfaced_reinforcement_event(
            actor="zaxy-memory",
            session_id="agent-1",
            checkout_id="checkout:0001",
            targets=[{"seq": memory.seq, "hash": memory.hash}],
        )
        sealed = log.append(
            spec["event_type"], actor=spec["actor"], payload=spec["payload"], thread=spec["thread"]
        )

        extraction = extract(sealed)

        assert extraction.entities == []
        assert extraction.edges == []
        assert extraction.source_event_hash == sealed.hash

    def test_verbatim_index_skips_reinforcement_events(self, tmp_path: Path) -> None:
        log = _sealed_log(tmp_path)
        memory = log.append("decision.made", actor="dev", payload={"decision": "Adopt salience"})
        spec = build_surfaced_reinforcement_event(
            actor="zaxy-memory",
            session_id="agent-1",
            checkout_id="checkout:0001",
            targets=[{"seq": memory.seq, "hash": memory.hash}],
        )
        log.append(
            spec["event_type"], actor=spec["actor"], payload=spec["payload"], thread=spec["thread"]
        )

        index = VerbatimIndex.from_event_logs([log])

        hits = index.query("surfaced checkout reinforcement targets", limit=10)
        assert all(hit.metadata.get("event_type") != REINFORCEMENT_EVENT_TYPE for hit in hits)
        decision_hits = index.query("Adopt salience decision", limit=10)
        assert decision_hits and decision_hits[0].metadata["event_seq"] == memory.seq

    async def test_emitted_reinforcement_events_never_target_each_other(
        self, tmp_path: Path
    ) -> None:
        fabric = _wired_fabric(tmp_path)
        await _seed_decision(fabric)

        await fabric.checkout_memory("salience ledger decision", session_id="agent-1", limit=3)
        await fabric.checkout_memory("salience ledger decision", session_id="agent-1", limit=3)

        log = fabric.session_manager.get("agent-1").eventlog
        index = event_ref_index(log.read_all())
        for reinforcement in _reinforcement_events(log):
            for target in reinforcement.payload["targets"]:
                _target_hash, target_type = index[target["seq"]]
                assert target_type != REINFORCEMENT_EVENT_TYPE


# ----------------------------------------------------------------------
# Encoding-specificity cues (2.2-alpha.2)
# ----------------------------------------------------------------------


def test_build_cue_record_keeps_only_non_empty_string_fields() -> None:
    record = build_cue_record(
        mission="ship 2.2",
        workspace="  /repo  ",
        tool="",
        phase=None,
    )

    assert record == {"mission": "ship 2.2", "workspace": "/repo"}
    assert build_cue_record() == {}


def test_cue_pairs_normalizes_and_ignores_malformed_records() -> None:
    pairs = cue_pairs({"tool": " pytest ", "workspace": "/repo", "unknown": "x", "phase": 3})

    assert pairs == frozenset({"tool=pytest", "workspace=/repo"})
    assert cue_pairs(None) == frozenset()
    assert cue_pairs("tool=pytest") == frozenset()
    assert cue_pairs({"tool": "   "}) == frozenset()


def test_cue_overlap_is_jaccard_and_empty_sets_contribute_zero() -> None:
    query = frozenset({"tool=pytest", "workspace=/repo"})
    stored = frozenset({"tool=pytest", "phase=review"})

    assert cue_overlap(query, stored) == pytest.approx(1 / 3)
    assert cue_overlap(query, query) == 1.0
    assert cue_overlap(frozenset(), stored) == 0.0
    assert cue_overlap(query, frozenset()) == 0.0


# ----------------------------------------------------------------------
# Write-time encoding gate classification (2.2-alpha.2)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content_overlap", "entity_overlap", "expected"),
    [
        (0.95, 0.0, "redundant"),
        (ENCODING_REDUNDANT_MIN_OVERLAP, 0.0, "redundant"),
        (0.7, 0.0, "reinforcing"),
        (ENCODING_REINFORCING_MIN_OVERLAP, 0.0, "reinforcing"),
        (0.5, 1.0, "reinforcing"),
        (ENCODING_ENTITY_CORROBORATION_MIN_OVERLAP, 1.0, "reinforcing"),
        (0.5, 0.5, "novel"),
        (0.39, 1.0, "novel"),
        (0.0, 1.0, "novel"),
        (0.0, 0.0, "novel"),
    ],
)
def test_classify_append_thresholds(
    content_overlap: float, entity_overlap: float, expected: str
) -> None:
    """Contradicting content stays novel even when every entity is already known."""
    assert (
        classify_append(content_overlap=content_overlap, entity_overlap=entity_overlap)
        == expected
    )


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf"), True])
def test_classify_append_rejects_out_of_range_signals(bad: float) -> None:
    with pytest.raises(ValueError, match="content_overlap"):
        classify_append(content_overlap=bad)
    with pytest.raises(ValueError, match="entity_overlap"):
        classify_append(content_overlap=0.5, entity_overlap=bad)


def test_encoding_decision_tag_payload_is_compact_and_replayable() -> None:
    decision = EncodingDecision(
        classification="redundant",
        content_overlap=0.91234,
        entity_overlap=1.0,
        duplicate_of="eventloom://agent-1/events/3#" + "a" * 64,
    )

    tag = decision.tag_payload()

    assert tag == {
        "classification": "redundant",
        "content_overlap": 0.9123,
        "entity_overlap": 1.0,
        "duplicate_of": "eventloom://agent-1/events/3#" + "a" * 64,
    }
    novel = EncodingDecision(classification="novel", content_overlap=0.1, entity_overlap=0.0)
    assert "duplicate_of" not in novel.tag_payload()
