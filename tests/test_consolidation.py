from __future__ import annotations

import pytest

import zaxy.consolidation as consolidation
from zaxy.consolidation import (
    CONSOLIDATION_CANDIDATE_TYPES,
    CONSOLIDATION_INITIAL_REVIEW_STATUS,
    CONSOLIDATION_REVIEW_STATUSES,
    build_consolidation_candidate_event,
    build_consolidation_review_event,
)


def test_build_episode_candidate_is_review_pending_and_cited() -> None:
    event = build_consolidation_candidate_event(
        actor="zaxy-consolidation",
        session_id="agent-1",
        candidate_type="episode",
        title="Pytest failure investigation",
        summary="The agent ran pytest, saw a failure, and identified the cause.",
        source_events=[
            {"seq": 10, "hash": "a" * 64},
            {"seq": 11, "hash": "b" * 64},
        ],
        confidence=0.74,
        method="event_segment_cluster_v1",
        purpose="coding",
    )

    assert event["event_type"] == "consolidation.candidate.created"
    assert event["thread"] == "agent-1"
    assert event["payload"]["candidate_type"] == "episode"
    assert event["payload"]["review_status"] == "pending"
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert event["payload"]["source_events"] == [
        {"seq": 10, "hash": "a" * 64},
        {"seq": 11, "hash": "b" * 64},
    ]


def test_consolidation_candidate_rejects_missing_source_events() -> None:
    with pytest.raises(ValueError, match="source_events"):
        build_consolidation_candidate_event(
            actor="zaxy-consolidation",
            session_id="agent-1",
            candidate_type="claim",
            title="Unsupported claim",
            summary="No citation.",
            source_events=[],
            confidence=0.5,
            method="event_segment_cluster_v1",
        )


def test_build_review_event_cannot_promote_to_authority_in_alpha_1() -> None:
    event = build_consolidation_review_event(
        actor="reviewer",
        session_id="agent-1",
        candidate_id="consolidation:episode:" + "c" * 24,
        status="accepted",
        rationale="Cited and useful, but alpha.1 keeps authority separate.",
    )

    assert event == {
        "event_type": "consolidation.candidate.reviewed",
        "actor": "reviewer",
        "payload": {
            "candidate_id": "consolidation:episode:" + "c" * 24,
            "status": "accepted",
            "authority_status": "non_authoritative",
            "rationale": "Cited and useful, but alpha.1 keeps authority separate.",
        },
        "thread": "agent-1",
    }


def test_candidate_type_taxonomy_is_stable() -> None:
    assert {"episode", "claim", "procedure"} == CONSOLIDATION_CANDIDATE_TYPES


def test_public_candidate_id_validator_follows_candidate_type_taxonomy() -> None:
    assert hasattr(consolidation, "validate_consolidation_candidate_id")

    for candidate_type in CONSOLIDATION_CANDIDATE_TYPES:
        consolidation.validate_consolidation_candidate_id(
            f"consolidation:{candidate_type}:{'c' * 24}"
        )

    with pytest.raises(ValueError, match="candidate_id"):
        consolidation.validate_consolidation_candidate_id("consolidation:memory:" + "c" * 24)


def test_review_status_taxonomy_separates_initial_pending_from_review_outcomes() -> None:
    assert CONSOLIDATION_INITIAL_REVIEW_STATUS == "pending"
    assert {"accepted", "rejected", "deferred", "conflicted"} == CONSOLIDATION_REVIEW_STATUSES
    assert CONSOLIDATION_INITIAL_REVIEW_STATUS not in CONSOLIDATION_REVIEW_STATUSES


def test_build_candidate_rejects_unsupported_candidate_type() -> None:
    with pytest.raises(ValueError, match="candidate_type"):
        build_consolidation_candidate_event(
            actor="zaxy-consolidation",
            session_id="agent-1",
            candidate_type="memory",
            title="Unsupported type",
            summary="Unsupported consolidation type.",
            source_events=[{"seq": 1, "hash": "a" * 64}],
            confidence=0.5,
            method="event_segment_cluster_v1",
        )


def test_build_review_event_rejects_invalid_review_status() -> None:
    with pytest.raises(ValueError, match="status"):
        build_consolidation_review_event(
            actor="reviewer",
            session_id="agent-1",
            candidate_id="consolidation:episode:" + "c" * 24,
            status="pending",
            rationale="Pending is only for candidate creation.",
        )


@pytest.mark.parametrize("confidence", [True, False, "0.5"])
def test_build_candidate_rejects_bool_and_string_confidence(confidence: object) -> None:
    with pytest.raises(ValueError, match="confidence"):
        build_consolidation_candidate_event(
            actor="zaxy-consolidation",
            session_id="agent-1",
            candidate_type="claim",
            title="Confidence validation",
            summary="Confidence must be numeric.",
            source_events=[{"seq": 1, "hash": "a" * 64}],
            confidence=confidence,
            method="event_segment_cluster_v1",
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_build_candidate_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        build_consolidation_candidate_event(
            actor="zaxy-consolidation",
            session_id="agent-1",
            candidate_type="claim",
            title="Confidence validation",
            summary="Confidence must be in range.",
            source_events=[{"seq": 1, "hash": "a" * 64}],
            confidence=confidence,
            method="event_segment_cluster_v1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "   "),
        ("summary", "   "),
        ("method", "   "),
    ],
)
def test_build_candidate_rejects_whitespace_text_fields(field: str, value: str) -> None:
    kwargs = {
        "actor": "zaxy-consolidation",
        "session_id": "agent-1",
        "candidate_type": "procedure",
        "title": "Incident workflow",
        "summary": "Steps observed during the incident workflow.",
        "source_events": [{"seq": 1, "hash": "a" * 64}],
        "confidence": 0.5,
        "method": "event_segment_cluster_v1",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        build_consolidation_candidate_event(**kwargs)


@pytest.mark.parametrize("field", ["title", "summary", "method"])
def test_build_candidate_rejects_non_string_text_fields(field: str) -> None:
    kwargs = {
        "actor": "zaxy-consolidation",
        "session_id": "agent-1",
        "candidate_type": "procedure",
        "title": "Incident workflow",
        "summary": "Steps observed during the incident workflow.",
        "source_events": [{"seq": 1, "hash": "a" * 64}],
        "confidence": 0.5,
        "method": "event_segment_cluster_v1",
    }
    kwargs[field] = 123

    with pytest.raises(ValueError, match=field):
        build_consolidation_candidate_event(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor", "   "),
        ("actor", 123),
        ("session_id", "   "),
        ("session_id", 123),
        ("purpose", "   "),
        ("purpose", 123),
    ],
)
def test_build_candidate_rejects_invalid_actor_session_and_purpose(
    field: str,
    value: object,
) -> None:
    kwargs = {
        "actor": "zaxy-consolidation",
        "session_id": "agent-1",
        "candidate_type": "procedure",
        "title": "Incident workflow",
        "summary": "Steps observed during the incident workflow.",
        "source_events": [{"seq": 1, "hash": "a" * 64}],
        "confidence": 0.5,
        "method": "event_segment_cluster_v1",
        "purpose": "coding",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        build_consolidation_candidate_event(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor", "   "),
        ("actor", 123),
        ("session_id", "   "),
        ("session_id", 123),
    ],
)
def test_build_review_event_rejects_invalid_actor_and_session(
    field: str,
    value: object,
) -> None:
    kwargs = {
        "actor": "reviewer",
        "session_id": "agent-1",
        "candidate_id": "consolidation:episode:" + "c" * 24,
        "status": "accepted",
        "rationale": "Cited and useful.",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        build_consolidation_review_event(**kwargs)


@pytest.mark.parametrize("rationale", ["   ", 123])
def test_build_review_event_rejects_invalid_rationale(rationale: object) -> None:
    with pytest.raises(ValueError, match="rationale"):
        build_consolidation_review_event(
            actor="reviewer",
            session_id="agent-1",
            candidate_id="consolidation:episode:" + "c" * 24,
            status="accepted",
            rationale=rationale,
        )


@pytest.mark.parametrize(
    "candidate_id",
    [
        "   ",
        123,
        "episode:" + "c" * 24,
        "consolidation:memory:" + "c" * 24,
        "consolidation:episode:" + "C" * 24,
        "consolidation:episode:" + "g" * 24,
        "consolidation:episode:" + "c" * 23,
        "consolidation:episode:" + "c" * 25,
        "consolidation:episode:" + "c" * 24 + ":extra",
    ],
)
def test_build_review_event_rejects_invalid_candidate_id(candidate_id: object) -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        build_consolidation_review_event(
            actor="reviewer",
            session_id="agent-1",
            candidate_id=candidate_id,
            status="accepted",
            rationale="Cited and useful.",
        )


@pytest.mark.parametrize("source_event", [{"seq": "1", "hash": "a" * 64}, {"seq": 0, "hash": "a" * 64}])
def test_build_candidate_rejects_invalid_source_event_seq(source_event: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="seq"):
        build_consolidation_candidate_event(
            actor="zaxy-consolidation",
            session_id="agent-1",
            candidate_type="claim",
            title="Invalid citation",
            summary="Citation sequence is invalid.",
            source_events=[source_event],
            confidence=0.5,
            method="event_segment_cluster_v1",
        )


@pytest.mark.parametrize(
    "event_hash",
    [
        "A" * 64,
        "g" * 64,
        "a" * 63,
    ],
)
def test_build_candidate_rejects_invalid_source_event_hash(event_hash: str) -> None:
    with pytest.raises(ValueError, match="hash"):
        build_consolidation_candidate_event(
            actor="zaxy-consolidation",
            session_id="agent-1",
            candidate_type="claim",
            title="Invalid citation",
            summary="Citation hash is invalid.",
            source_events=[{"seq": 1, "hash": event_hash}],
            confidence=0.5,
            method="event_segment_cluster_v1",
        )


def test_build_candidate_snapshots_source_events() -> None:
    source_event = {"seq": 1, "hash": "a" * 64}
    source_events = [source_event]

    event = build_consolidation_candidate_event(
        actor="zaxy-consolidation",
        session_id="agent-1",
        candidate_type="claim",
        title="Snapshot citation",
        summary="Citation payload should be immutable from caller changes.",
        source_events=source_events,
        confidence=0.5,
        method="event_segment_cluster_v1",
    )
    source_event["seq"] = 2
    source_events.append({"seq": 3, "hash": "b" * 64})

    assert event["payload"]["source_events"] == [{"seq": 1, "hash": "a" * 64}]


def test_build_candidate_id_is_deterministic_for_type_title_and_sources() -> None:
    kwargs = {
        "actor": "zaxy-consolidation",
        "session_id": "agent-1",
        "candidate_type": "episode",
        "title": "Deterministic candidate",
        "summary": "First summary.",
        "source_events": [{"seq": 7, "hash": "a" * 64}],
        "confidence": 0.5,
        "method": "event_segment_cluster_v1",
    }

    first = build_consolidation_candidate_event(**kwargs)
    second = build_consolidation_candidate_event(
        **{
            **kwargs,
            "actor": "other-actor",
            "session_id": "other-session",
            "summary": "Changed summary.",
            "confidence": 0.9,
            "method": "other_method_v1",
        }
    )

    assert first["payload"]["candidate_id"] == second["payload"]["candidate_id"]
