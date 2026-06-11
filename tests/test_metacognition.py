from __future__ import annotations

import time

import pytest

from zaxy.event import Event
from zaxy.extract import extract
from zaxy.metacognition import (
    FOK_BLOOM_FALSE_POSITIVE_RATE,
    FOK_BLOOM_WEIGHT,
    FOK_CUE_WEIGHT,
    FOK_LIKELY_THRESHOLD,
    FOK_POSSIBLE_THRESHOLD,
    FOK_SALIENCE_BUCKET_UPPER_BOUNDS,
    FOK_SALIENCE_WEIGHT,
    FeelingOfKnowingIndex,
    build_confidence_assessment_event,
    build_conflict_cluster_event,
    build_feeling_of_knowing_index,
    build_known_unknown_event,
    build_reverify_request_event,
    feeling_of_knowing,
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


def test_metacognition_builders_reject_empty_required_text() -> None:
    with pytest.raises(ValueError, match="actor"):
        build_known_unknown_event(
            actor=" ",
            session_id="agent-1",
            question="What changed?",
            reason="No cited answer.",
            source_events=SOURCE,
        )


def test_reverify_request_rejects_non_string_priority() -> None:
    with pytest.raises(ValueError, match="priority"):
        build_reverify_request_event(
            actor="agent",
            session_id="agent-1",
            query="what changed",
            reason="missing evidence",
            source_events=SOURCE,
            priority=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("source_events", ["not-a-list", []])
def test_known_unknown_rejects_non_sequence_or_empty_source_events(source_events: object) -> None:
    with pytest.raises(ValueError, match="source_events"):
        build_known_unknown_event(
            actor="agent",
            session_id="agent-1",
            question="What changed?",
            reason="No cited answer.",
            source_events=source_events,  # type: ignore[arg-type]
        )


def test_conflict_cluster_rejects_non_mapping_source_event() -> None:
    with pytest.raises(ValueError, match="supporting_source_events"):
        build_conflict_cluster_event(
            actor="zaxy-reasoning",
            session_id="agent-1",
            claim_key="projection-latency-cause",
            claim="Projection stale caused failure",
            supporting_source_events=["eventloom://agent-1/events/7#aaaaaaaaaaaa"],  # type: ignore[list-item]
            conflicting_source_events=[{"seq": 8, "hash": "b" * 64}],
            confidence=0.5,
            reason="Support and conflict evidence both present.",
        )


def test_confidence_assessment_rejects_non_mapping_evidence_item() -> None:
    with pytest.raises(ValueError, match="evidence"):
        build_confidence_assessment_event(
            actor="zaxy-reasoning",
            session_id="agent-1",
            claim="Projection stale caused failure",
            confidence=0.42,
            support_count=1,
            conflict_count=0,
            evidence=["eventloom://agent-1/events/7#aaaaaaaaaaaa"],  # type: ignore[list-item]
            method="deterministic_token_overlap_v1",
        )


def test_known_unknown_accepts_valid_explicit_id() -> None:
    event = build_known_unknown_event(
        actor="agent",
        session_id="agent-1",
        question="What changed?",
        reason="No cited answer.",
        source_events=SOURCE,
        unknown_id="metacognition:unknown:" + ("a" * 24),
    )

    assert event["payload"]["unknown_id"] == "metacognition:unknown:" + ("a" * 24)


# --- Feeling-of-knowing pre-check -------------------------------------------

_FOK_CORPUS_SIZE = 10_000


def _seeded_corpus_index() -> FeelingOfKnowingIndex:
    return build_feeling_of_knowing_index(
        f"corpusname{position:05d}" for position in range(_FOK_CORPUS_SIZE)
    )


def test_fok_known_present_terms_always_hit_the_bloom() -> None:
    """A bloom filter never produces false negatives for inserted tokens."""
    index = _seeded_corpus_index()
    for position in range(0, _FOK_CORPUS_SIZE, 7):
        verdict = feeling_of_knowing(index, f"corpusname{position:05d}")
        assert verdict.signals.bloom_hits == 1
        assert verdict.verdict == "likely"


def test_fok_bloom_false_positive_rate_stays_near_design_rate() -> None:
    """Known-absent terms should hit at most ~3x the design FP rate."""
    index = _seeded_corpus_index()
    probes = 10_000
    false_positives = sum(
        feeling_of_knowing(index, f"absenttoken{position:05d}").signals.bloom_hits
        for position in range(probes)
    )
    assert false_positives / probes <= 3.0 * FOK_BLOOM_FALSE_POSITIVE_RATE


def test_fok_verdict_thresholds_partition_the_score_range() -> None:
    index = build_feeling_of_knowing_index(["alpha beta", "gamma delta"])

    full_match = feeling_of_knowing(index, "alpha beta")
    assert full_match.score == pytest.approx(FOK_BLOOM_WEIGHT)
    assert full_match.score >= FOK_LIKELY_THRESHOLD
    assert full_match.verdict == "likely"

    half_match = feeling_of_knowing(index, "alpha zzqq")
    assert half_match.score == pytest.approx(FOK_BLOOM_WEIGHT / 2)
    assert FOK_POSSIBLE_THRESHOLD <= half_match.score < FOK_LIKELY_THRESHOLD
    assert half_match.verdict == "possible"

    no_match = feeling_of_knowing(index, "zzqq yyxx")
    assert no_match.score == 0.0
    assert no_match.verdict == "unlikely"


def test_fok_one_third_bloom_hit_lands_on_possible_boundary() -> None:
    """Regression: 0.6 * (1/3) is exactly the possible threshold, not below it.

    In binary floating point ``0.6 * (1/3)`` evaluates to
    0.19999999999999998 < 0.2, which misclassified the designed boundary
    case ("roughly one third of the query terms are known names") as
    "unlikely" before the epsilon-tolerant threshold comparison.
    """
    index = build_feeling_of_knowing_index(["amber dynamo gateway"])

    verdict = feeling_of_knowing(index, "amber crater turbine")

    assert verdict.signals.query_term_count == 3
    assert verdict.signals.bloom_hits == 1
    assert verdict.score == pytest.approx(FOK_BLOOM_WEIGHT / 3)
    assert verdict.verdict == "possible"


def test_fok_signal_breakdown_reports_exact_components() -> None:
    index = build_feeling_of_knowing_index(
        ["alpha beta"],
        cue_counts={"Alpha": 2, "repo": 1},
        salience_by_name={"alpha beta": 2.0},
    )

    verdict = feeling_of_knowing(index, "alpha gammaz")

    signals = verdict.signals
    assert signals.query_term_count == 2
    assert signals.bloom_hits == 1
    assert signals.bloom_hit_ratio == pytest.approx(0.5)
    assert signals.cue_hits == 1  # cue keys are casefolded at build time
    assert signals.cue_hit_total == 2
    assert signals.cue_hit_ratio == pytest.approx(0.5)
    assert signals.matched_salience_mass == pytest.approx(2.0)
    assert signals.total_salience_mass == pytest.approx(2.0)
    assert signals.salience_mass_ratio == pytest.approx(1.0)
    expected_score = (
        FOK_BLOOM_WEIGHT * 0.5 + FOK_CUE_WEIGHT * 0.5 + FOK_SALIENCE_WEIGHT * 1.0
    )
    assert verdict.score == pytest.approx(expected_score)
    assert verdict.verdict == "likely"
    payload = verdict.to_dict()
    assert payload["authority_status"] == "non_authoritative"
    assert payload["signals"]["bloom_hits"] == 1


def test_fok_salience_histogram_uses_fixed_buckets() -> None:
    index = build_feeling_of_knowing_index(
        ["a1", "b2", "c3", "d4", "e5"],
        salience_by_name={"a1": 0.3, "b2": 0.9, "c3": 1.5, "d4": 3.0, "e5": 7.0},
    )

    assert len(index.salience_histogram) == len(FOK_SALIENCE_BUCKET_UPPER_BOUNDS) + 1
    assert index.salience_histogram == (1, 1, 1, 1, 1)
    assert index.total_salience_mass == pytest.approx(12.7)


def test_fok_empty_index_is_always_unlikely() -> None:
    index = build_feeling_of_knowing_index([])

    verdict = feeling_of_knowing(index, "anything alpha beta")

    assert index.entity_count == 0
    assert index.token_count == 0
    assert verdict.verdict == "unlikely"
    assert verdict.score == 0.0
    assert verdict.signals.bloom_hits == 0


def test_fok_stop_word_only_query_is_unlikely_not_an_error() -> None:
    index = build_feeling_of_knowing_index(["alpha beta"])

    verdict = feeling_of_knowing(index, "what was the")

    assert verdict.signals.query_term_count == 0
    assert verdict.verdict == "unlikely"


def test_fok_index_and_verdicts_are_deterministic() -> None:
    names = ["alpha beta", "Gamma Service", "delta-pipeline"]
    cues = {"alpha": 3, "pipeline": 1}
    salience = {"alpha beta": 1.5, "Gamma Service": 0.4}

    first = build_feeling_of_knowing_index(names, cue_counts=cues, salience_by_name=salience)
    second = build_feeling_of_knowing_index(names, cue_counts=cues, salience_by_name=salience)

    assert first == second
    assert feeling_of_knowing(first, "alpha pipeline") == feeling_of_knowing(
        second, "alpha pipeline"
    )


def test_fok_builder_validates_inputs() -> None:
    with pytest.raises(ValueError, match="entity_names"):
        build_feeling_of_knowing_index([42])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="cue_counts"):
        build_feeling_of_knowing_index([], cue_counts={"alpha": -1})
    with pytest.raises(ValueError, match="cue_counts"):
        build_feeling_of_knowing_index([], cue_counts={"alpha": True})
    with pytest.raises(ValueError, match="salience_by_name"):
        build_feeling_of_knowing_index([], salience_by_name={"alpha": float("nan")})
    with pytest.raises(ValueError, match="salience_by_name"):
        build_feeling_of_knowing_index([], salience_by_name={"alpha": -0.1})


def test_fok_query_validation() -> None:
    index = build_feeling_of_knowing_index(["alpha"])
    with pytest.raises(ValueError, match="query"):
        feeling_of_knowing(index, "   ")
    with pytest.raises(ValueError, match="FeelingOfKnowingIndex"):
        feeling_of_knowing("not-an-index", "alpha")  # type: ignore[arg-type]


def test_fok_performance_smoke_10k_names_1k_queries() -> None:
    """10k-name index build plus 1k verdicts should stay comfortably fast."""
    started = time.perf_counter()
    index = _seeded_corpus_index()
    built = time.perf_counter()
    for position in range(1_000):
        feeling_of_knowing(index, f"corpusname{position:05d} absentterm{position:05d}")
    finished = time.perf_counter()

    # Generous bounds to avoid flaky CI; locally this runs in well under 1s.
    assert built - started < 5.0
    assert finished - built < 5.0
