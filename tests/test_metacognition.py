from __future__ import annotations

import pytest

from zaxy.event import Event
from zaxy.extract import extract
from zaxy.metacognition import (
    build_confidence_assessment_event,
    build_conflict_cluster_event,
    build_known_unknown_event,
    build_reverify_request_event,
    summarize_metacognition_events,
)

SOURCE = [{"seq": 7, "hash": "a" * 64}]


def test_known_unknown_event_is_open_cited_and_non_authoritative() -> None:
    event = build_known_unknown_event(
        actor="agent",
        session_id="agent-1",
        question="Which projection backend caused the latency spike?",
        reason="Checkout had conflicting backend evidence.",
        source_events=SOURCE,
        claim_key="projection-latency-cause",
        gap_type="conflicting_evidence",
        reverify_query="latest cited projection latency cause",
    )

    assert event["event_type"] == "metacognition.unknown.recorded"
    assert event["thread"] == "agent-1"
    assert event["payload"]["status"] == "open"
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert event["payload"]["source_events"] == SOURCE


def test_confidence_assessment_event_tracks_append_only_point() -> None:
    event = build_confidence_assessment_event(
        actor="zaxy-reasoning",
        session_id="agent-1",
        claim="Projection stale caused failure",
        confidence=0.42,
        support_count=1,
        conflict_count=2,
        evidence=[
            {"citation": "eventloom://agent-1/events/7#aaaaaaaaaaaa", "stance": "support"},
            {"citation": "eventloom://agent-1/events/8#bbbbbbbbbbbb", "stance": "conflict"},
        ],
        method="deterministic_token_overlap_v1",
        requires_reverify=True,
    )

    assert event["event_type"] == "metacognition.confidence.assessed"
    assert event["payload"]["confidence"] == 0.42
    assert event["payload"]["requires_reverify"] is True
    assert event["payload"]["authority_status"] == "non_authoritative"


def test_conflict_cluster_event_preserves_support_and_conflict_sources() -> None:
    event = build_conflict_cluster_event(
        actor="zaxy-reasoning",
        session_id="agent-1",
        claim_key="projection-latency-cause",
        claim="Projection stale caused failure",
        supporting_source_events=[{"seq": 7, "hash": "a" * 64}],
        conflicting_source_events=[{"seq": 8, "hash": "b" * 64}],
        confidence=0.5,
        reason="Support and conflict evidence both present.",
    )

    assert event["event_type"] == "metacognition.conflict.clustered"
    assert event["payload"]["resolution_status"] == "unresolved"
    assert event["payload"]["authority_status"] == "non_authoritative"


def test_reverify_request_event_is_open_and_cited() -> None:
    event = build_reverify_request_event(
        actor="zaxy-reasoning",
        session_id="agent-1",
        query="Re-check cited projection latency cause",
        reason="Low confidence and conflict count above zero.",
        source_events=SOURCE,
        priority="high",
        claim_key="projection-latency-cause",
    )

    assert event["event_type"] == "metacognition.reverify.requested"
    assert event["payload"]["reverify_id"].startswith("metacognition:reverify:")
    assert event["payload"]["status"] == "open"
    assert event["payload"]["priority"] == "high"


def test_reverify_request_builder_payload_projects_to_typed_entity() -> None:
    event = build_reverify_request_event(
        actor="zaxy-reasoning",
        session_id="agent-1",
        query="Re-check cited projection latency cause",
        reason="Low confidence and conflict count above zero.",
        source_events=SOURCE,
        priority="high",
        claim_key="projection-latency-cause",
    )
    extracted = extract(
        Event(
            seq=1,
            timestamp="2026-06-08T00:00:00Z",
            type=event["event_type"],
            actor=event["actor"],
            payload=event["payload"],
            hash="f" * 64,
        )
    )

    request = next(entity for entity in extracted.entities if entity.entity_type == "reverify_request")
    assert request.name == event["payload"]["reverify_id"]
    assert request.properties["authority_status"] == "non_authoritative"


def test_summarize_metacognition_events_returns_open_counts() -> None:
    events = [
        object(),
        {"event_type": "metacognition.unknown.recorded", "payload": "not-a-payload"},
        build_known_unknown_event(
            actor="agent",
            session_id="agent-1",
            question="What changed?",
            reason="No cited answer.",
            source_events=SOURCE,
            claim_key="change",
            gap_type="missing_evidence",
            reverify_query="what changed",
        ),
        build_reverify_request_event(
            actor="agent",
            session_id="agent-1",
            query="what changed",
            reason="missing evidence",
            source_events=SOURCE,
            priority="normal",
            claim_key="change",
        ),
        build_confidence_assessment_event(
            actor="agent",
            session_id="agent-1",
            claim="The release gate needs cited evidence.",
            confidence=0.8,
            support_count=2,
            conflict_count=0,
            evidence=[{"citation": "eventloom://agent-1/events/9#cccccccccccc"}],
            method="deterministic_confidence_v1",
        ),
        build_conflict_cluster_event(
            actor="agent",
            session_id="agent-1",
            claim_key="release-gate",
            claim="The release gate needs cited evidence.",
            supporting_source_events=[{"seq": 9, "hash": "c" * 64}],
            conflicting_source_events=[{"seq": 10, "hash": "d" * 64}],
            confidence=0.4,
            reason="Support and conflict evidence are both present.",
        ),
    ]

    summary = summarize_metacognition_events(events)

    assert summary["unknown_count"] == 1
    assert summary["open_unknown_count"] == 1
    assert summary["confidence_assessment_count"] == 1
    assert summary["conflict_cluster_count"] == 1
    assert summary["unresolved_conflict_cluster_count"] == 1
    assert summary["reverify_needed_count"] == 1
    assert summary["conflict_clusters"][0]["claim_key"] == "release-gate"


def test_builder_ids_are_deterministic_from_inputs() -> None:
    first = build_known_unknown_event(
        actor="agent",
        session_id="agent-1",
        question="What changed?",
        reason="No cited answer.",
        source_events=SOURCE,
        claim_key="change",
        gap_type="missing_evidence",
        reverify_query="what changed",
    )
    second = build_known_unknown_event(
        actor="other-agent",
        session_id="agent-1",
        question="What changed?",
        reason="No cited answer.",
        source_events=SOURCE,
        claim_key="change",
        gap_type="missing_evidence",
        reverify_query="what changed",
    )

    assert first["payload"]["unknown_id"] == second["payload"]["unknown_id"]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, True])
def test_confidence_assessment_rejects_invalid_confidence(confidence: object) -> None:
    with pytest.raises(ValueError, match="confidence"):
        build_confidence_assessment_event(
            actor="zaxy-reasoning",
            session_id="agent-1",
            claim="Projection stale caused failure",
            confidence=confidence,  # type: ignore[arg-type]
            support_count=1,
            conflict_count=0,
            evidence=[{"citation": "eventloom://agent-1/events/7#aaaaaaaaaaaa"}],
            method="deterministic_token_overlap_v1",
        )


@pytest.mark.parametrize(
    "source_event",
    [
        {"seq": 0, "hash": "a" * 64},
        {"seq": 1, "hash": "A" * 64},
        {"seq": True, "hash": "a" * 64},
    ],
)
def test_known_unknown_rejects_invalid_source_events(source_event: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="source_events"):
        build_known_unknown_event(
            actor="agent",
            session_id="agent-1",
            question="What changed?",
            reason="No cited answer.",
            source_events=[source_event],
            claim_key="change",
            gap_type="missing_evidence",
        )


@pytest.mark.parametrize(
    "citation",
    [
        "eventloom://agent-1/events/7#aaaaaaaaaaa",
        "eventloom://agent-1/events/7#aaaaaaaaaaaaa",
        "eventloom://agent-1/events/7#AAAAAAAAAAAA",
        "file://agent-1/events/7#aaaaaaaaaaaa",
    ],
)
def test_confidence_assessment_rejects_invalid_eventloom_citations(citation: str) -> None:
    with pytest.raises(ValueError, match="citation"):
        build_confidence_assessment_event(
            actor="zaxy-reasoning",
            session_id="agent-1",
            claim="Projection stale caused failure",
            confidence=0.42,
            support_count=1,
            conflict_count=0,
            evidence=[{"citation": citation}],
            method="deterministic_token_overlap_v1",
        )


def test_reverify_request_rejects_invalid_priority() -> None:
    with pytest.raises(ValueError, match="priority"):
        build_reverify_request_event(
            actor="agent",
            session_id="agent-1",
            query="what changed",
            reason="missing evidence",
            source_events=SOURCE,
            priority="soon",
        )


def test_builders_snapshot_mutable_sources() -> None:
    source_events = [{"seq": 7, "hash": "a" * 64}]

    event = build_reverify_request_event(
        actor="agent",
        session_id="agent-1",
        query="what changed",
        reason="missing evidence",
        source_events=source_events,
        priority="normal",
    )
    source_events[0]["hash"] = "b" * 64

    assert event["payload"]["source_events"] == SOURCE


def test_confidence_assessment_rejects_non_boolean_reverify_flag() -> None:
    with pytest.raises(ValueError, match="requires_reverify"):
        build_confidence_assessment_event(
            actor="zaxy-reasoning",
            session_id="agent-1",
            claim="Projection stale caused failure",
            confidence=0.42,
            support_count=1,
            conflict_count=0,
            evidence=[{"citation": "eventloom://agent-1/events/7#aaaaaaaaaaaa"}],
            method="deterministic_token_overlap_v1",
            requires_reverify="yes",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(("support_count", "conflict_count"), [("1", 0), (1, -1), (True, 0)])
def test_confidence_assessment_rejects_invalid_counts(
    support_count: object,
    conflict_count: object,
) -> None:
    with pytest.raises(ValueError, match="count"):
        build_confidence_assessment_event(
            actor="zaxy-reasoning",
            session_id="agent-1",
            claim="Projection stale caused failure",
            confidence=0.42,
            support_count=support_count,  # type: ignore[arg-type]
            conflict_count=conflict_count,  # type: ignore[arg-type]
            evidence=[{"citation": "eventloom://agent-1/events/7#aaaaaaaaaaaa"}],
            method="deterministic_token_overlap_v1",
        )


@pytest.mark.parametrize(
    "explicit_id",
    [
        "metacognition:unknown:abc",
        "metacognition:unknown:AAAAAAAAAAAAAAAAAAAAAAAA",
    ],
)
def test_known_unknown_rejects_invalid_explicit_id(explicit_id: str) -> None:
    with pytest.raises(ValueError, match="unknown_id"):
        build_known_unknown_event(
            actor="agent",
            session_id="agent-1",
            question="What changed?",
            reason="No cited answer.",
            source_events=SOURCE,
            unknown_id=explicit_id,
        )


def test_confidence_assessment_rejects_non_sequence_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        build_confidence_assessment_event(
            actor="zaxy-reasoning",
            session_id="agent-1",
            claim="Projection stale caused failure",
            confidence=0.42,
            support_count=1,
            conflict_count=0,
            evidence="eventloom://agent-1/events/7#aaaaaaaaaaaa",  # type: ignore[arg-type]
            method="deterministic_token_overlap_v1",
        )


def test_confidence_assessment_trims_evidence_string_fields() -> None:
    event = build_confidence_assessment_event(
        actor="zaxy-reasoning",
        session_id="agent-1",
        claim="Projection stale caused failure",
        confidence=0.42,
        support_count=1,
        conflict_count=0,
        evidence=[
            {
                "citation": "eventloom://agent-1/events/7#aaaaaaaaaaaa",
                "stance": " support ",
            }
        ],
        method="deterministic_token_overlap_v1",
    )

    assert event["payload"]["evidence"][0]["stance"] == "support"
