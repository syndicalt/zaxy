from __future__ import annotations

from typing import Any

import pytest

from zaxy.causal import CausalQueryResult
from zaxy.context import Context
from zaxy.core import MemoryFabric
from zaxy.reasoning_primitives import (
    REASONING_PHASES,
    ReasoningPrimitiveCall,
    build_belief_update_proposal_event,
    phase_purpose_profile,
    validate_reasoning_phase,
)


def test_reasoning_phase_taxonomy_is_stable() -> None:
    assert {"planning", "execution", "review", "reflection"} == REASONING_PHASES


def test_phase_purpose_profiles_are_distinct() -> None:
    assert phase_purpose_profile("planning").task == "planning"
    assert phase_purpose_profile("execution").task == "execution"
    assert phase_purpose_profile("review").risk == "high"
    assert phase_purpose_profile("reflection").expected_action == "revise_or_record_learning"
    assert phase_purpose_profile("planning") != phase_purpose_profile("execution")


def test_validate_reasoning_phase_rejects_unknown_phase() -> None:
    assert validate_reasoning_phase("planning") == "planning"
    with pytest.raises(ValueError, match="reasoning phase"):
        validate_reasoning_phase("benchmark")


def test_reasoning_call_event_is_observable_and_cited() -> None:
    call = ReasoningPrimitiveCall(
        primitive="explain_outcome",
        phase="planning",
        session_id="agent-1",
        query="Why did the test fail?",
        result_count=2,
        evidence=[
            {
                "citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa",
                "content": "failure cause",
            }
        ],
        status="succeeded",
    )

    event = call.to_event(actor="zaxy-reasoning")

    assert event["event_type"] == "reasoning.primitive.called"
    assert event["actor"] == "zaxy-reasoning"
    assert event["thread"] == "agent-1"
    assert event["payload"]["primitive"] == "explain_outcome"
    assert event["payload"]["phase"] == "planning"
    assert event["payload"]["status"] == "succeeded"
    assert event["payload"]["result_count"] == 2
    assert event["payload"]["evidence_count"] == 1
    assert event["payload"]["citations"] == ["eventloom://agent-1/events/42#aaaaaaaaaaaa"]


def test_reasoning_call_event_accepts_full_eventloom_hash_citation() -> None:
    full_hash = "b" * 64
    call = ReasoningPrimitiveCall(
        primitive="get_claim_confidence",
        phase="review",
        session_id="agent-1",
        query="Projection stale caused failure",
        result_count=1,
        evidence=[
            {
                "citation": f"eventloom://agent-1/events/42#{full_hash}",
                "content": "Projection stale caused failure.",
            }
        ],
        status="succeeded",
    )

    event = call.to_event(actor="zaxy-reasoning")

    assert event["payload"]["evidence_count"] == 1
    assert event["payload"]["citations"] == [f"eventloom://agent-1/events/42#{full_hash}"]


def test_reasoning_call_event_rejects_malformed_hash_fragment_length() -> None:
    with pytest.raises(ValueError, match="citation"):
        ReasoningPrimitiveCall(
            primitive="get_claim_confidence",
            phase="review",
            session_id="agent-1",
            query="Projection stale caused failure",
            result_count=1,
            evidence=[
                {
                    "citation": "eventloom://agent-1/events/42#bbbbbbbbbbbbb",
                    "content": "Projection stale caused failure.",
                }
            ],
            status="succeeded",
        )


def test_reasoning_call_event_requires_cited_evidence() -> None:
    with pytest.raises(ValueError, match="citation"):
        ReasoningPrimitiveCall(
            primitive="explain_outcome",
            phase="planning",
            session_id="agent-1",
            query="Why did the test fail?",
            result_count=1,
            evidence=[{"content": "uncited cause"}],
            status="succeeded",
        )


def test_belief_update_proposal_is_never_authoritative() -> None:
    event = build_belief_update_proposal_event(
        actor="agent",
        session_id="agent-1",
        claim="The failure was caused by a stale projection.",
        rationale="Cited causal predecessor indicates stale projection.",
        confidence=0.74,
        source_events=[{"seq": 42, "hash": "a" * 64}],
        phase="reflection",
    )

    assert event["event_type"] == "belief.update.proposed"
    assert event["actor"] == "agent"
    assert event["thread"] == "agent-1"
    assert event["payload"]["claim"] == "The failure was caused by a stale projection."
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert event["payload"]["review_status"] == "pending"
    assert event["payload"]["source_events"] == [{"seq": 42, "hash": "a" * 64}]


def test_belief_update_proposal_requires_strict_source_events() -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        build_belief_update_proposal_event(
            actor="agent",
            session_id="agent-1",
            claim="Projection is stale.",
            rationale="Needs review.",
            confidence=0.6,
            source_events=[{"seq": 42, "hash": "A" * 64}],
            phase="reflection",
        )


@pytest.mark.asyncio
async def test_memory_fabric_explain_outcome_records_cited_primitive_call(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric = _fabric(tmp_path, monkeypatch)
    causal_result = CausalQueryResult(
        source={"name": "stale projection", "entity_type": "issue"},
        target={"name": "test failure", "entity_type": "outcome"},
        relation_type="caused",
        graph_relation_type="causal_caused",
        confidence=0.81,
        method="unit-test",
        citation="eventloom://agent-1/events/7#aaaaaaaaaaaa",
        review_status="proposed",
        authority_status="non_authoritative",
        evidence={"summary": "stale projection caused the failure"},
    )

    async def fake_predecessors(*args: Any, **kwargs: Any) -> list[CausalQueryResult]:
        assert args == ("test failure",)
        assert kwargs["depth"] == 2
        assert kwargs["session_id"] == "agent-1"
        return [causal_result]

    monkeypatch.setattr(fabric, "query_causal_predecessors", fake_predecessors)

    result = await fabric.explain_outcome("test failure", phase="planning", session_id="agent-1")

    assert result["outcome"] == "test failure"
    assert result["phase"] == "planning"
    assert result["result_count"] == 1
    assert result["evidence"][0]["citation"] == "eventloom://agent-1/events/7#aaaaaaaaaaaa"
    events = fabric.session_manager.get("agent-1").eventlog.read_all()
    assert events[-1].type == "reasoning.primitive.called"
    assert events[-1].payload["primitive"] == "explain_outcome"
    assert events[-1].payload["evidence_count"] == 1


@pytest.mark.asyncio
async def test_memory_fabric_propose_belief_update_appends_only_proposal_and_call(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric = _fabric(tmp_path, monkeypatch)

    result = await fabric.propose_belief_update(
        "The stale projection caused the failure.",
        rationale="Reviewed causal predecessor event.",
        confidence=0.72,
        source_events=[{"seq": 7, "hash": "a" * 64}],
        phase="reflection",
        session_id="agent-1",
        actor="agent",
    )

    assert result["event_type"] == "belief.update.proposed"
    assert result["payload"]["authority_status"] == "non_authoritative"
    events = fabric.session_manager.get("agent-1").eventlog.read_all()
    assert [event.type for event in events] == [
        "belief.update.proposed",
        "reasoning.primitive.called",
    ]
    assert events[0].payload["review_status"] == "pending"
    assert events[1].payload["primitive"] == "propose_belief_update"


@pytest.mark.asyncio
async def test_memory_fabric_get_claim_confidence_scores_cited_support_and_conflict(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric = _fabric(tmp_path, monkeypatch)

    async def fake_checkout_memory(*args: Any, **kwargs: Any) -> Any:
        assert args == ("Projection stale caused failure",)
        assert kwargs["purpose"].task == "review"
        return _Checkout(
            evidence=[
                {
                    "content": "Projection stale caused failure during replay.",
                    "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
                },
                {
                    "content": "Projection stale did cause the failure in the replay worker.",
                    "citation": "eventloom://agent-1/events/8#888888888888",
                },
                {
                    "content": "Network timeout did not cause the failure.",
                    "citation": "eventloom://agent-1/events/3#cccccccccccc",
                },
            ]
        )

    monkeypatch.setattr(fabric, "checkout_memory", fake_checkout_memory)

    result = await fabric.get_claim_confidence(
        "Projection stale caused failure",
        phase="review",
        session_id="agent-1",
        limit=5,
    )

    assert result["claim"] == "Projection stale caused failure"
    assert result["support_count"] == 2
    assert result["conflict_count"] == 1
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["evidence"]) == 3
    events = fabric.session_manager.get("agent-1").eventlog.read_all()
    assert events[-1].type == "reasoning.primitive.called"
    assert events[-1].payload["primitive"] == "get_claim_confidence"
    assert events[-1].payload["result_count"] == 3


@pytest.mark.asyncio
async def test_memory_fabric_claim_confidence_excludes_pending_proposals_and_trace_observations(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric = _fabric(tmp_path, monkeypatch)

    async def fake_checkout_memory(*args: Any, **kwargs: Any) -> Any:
        return _Checkout(
            evidence=[
                {
                    "content": "Projection stale caused failure during replay.",
                    "citation": "eventloom://agent-1/events/2#" + ("b" * 64),
                    "entity_type": "finding",
                    "review_status": "accepted",
                    "status": "accepted",
                },
                {
                    "content": "Projection stale caused failure during replay.",
                    "citation": "eventloom://agent-1/events/3#cccccccccccc",
                    "event_type": "belief.update.proposed",
                    "entity_type": "belief_update_proposal",
                    "authority_status": "non_authoritative",
                    "review_status": "pending",
                },
                {
                    "content": "Projection stale caused failure during replay.",
                    "citation": "eventloom://agent-1/events/4#dddddddddddd",
                    "event_type": "reasoning.primitive.called",
                    "entity_type": "reasoning_primitive_observation",
                    "authority_status": "non_authoritative",
                    "primitive": "propose_belief_update",
                },
            ]
        )

    monkeypatch.setattr(fabric, "checkout_memory", fake_checkout_memory)

    result = await fabric.get_claim_confidence(
        "Projection stale caused failure",
        phase="review",
        session_id="agent-1",
        limit=5,
    )

    assert result["support_count"] == 1
    assert result["conflict_count"] == 0
    assert result["confidence"] == 1.0
    assert [item["citation"] for item in result["evidence"]] == [
        "eventloom://agent-1/events/2#" + ("b" * 64)
    ]
    events = fabric.session_manager.get("agent-1").eventlog.read_all()
    assert events[-1].payload["evidence_count"] == 1
    assert events[-1].payload["citations"] == ["eventloom://agent-1/events/2#" + ("b" * 64)]


@pytest.mark.asyncio
async def test_memory_fabric_claim_confidence_records_reverify_for_missing_evidence(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-confidence no-evidence assessments should still create a cited reverify event."""
    fabric = _fabric(tmp_path, monkeypatch)

    async def fake_checkout_memory(*args: Any, **kwargs: Any) -> Any:
        return _Checkout(evidence=[])

    monkeypatch.setattr(fabric, "checkout_memory", fake_checkout_memory)

    result = await fabric.get_claim_confidence(
        "Projection stale caused failure",
        phase="review",
        session_id="agent-1",
        limit=5,
        min_confidence=0.7,
    )

    assert result["confidence"] == 0.0
    events = fabric.session_manager.get("agent-1").eventlog.read_all()
    assert [event.type for event in events] == [
        "metacognition.confidence.assessed",
        "metacognition.reverify.requested",
        "reasoning.primitive.called",
    ]
    assert events[1].payload["source_events"] == [{"seq": events[0].seq, "hash": events[0].hash}]
    assert events[1].payload["status"] == "open"


@pytest.mark.asyncio
async def test_memory_fabric_metacognition_query_surfaces_record_primitive_observations(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model-callable metacognition query surfaces should remain observable."""
    fabric = _fabric(tmp_path, monkeypatch)
    await fabric.record_known_unknown(
        "Which backend caused latency?",
        reason="Evidence conflicted.",
        source_events=[{"seq": 7, "hash": "a" * 64}],
        claim_key="backend-latency",
        phase="review",
        session_id="agent-1",
    )

    await fabric.list_known_unknowns(session_id="agent-1", limit=3)
    await fabric.list_confidence_trajectory("backend-latency", session_id="agent-1", limit=3)
    await fabric.list_reverification_needs("backend", session_id="agent-1", limit=3)

    primitives = [
        event.payload["primitive"]
        for event in fabric.session_manager.get("agent-1").eventlog.read_all()
        if event.type == "reasoning.primitive.called"
    ]
    assert primitives == [
        "record_known_unknown",
        "list_known_unknowns",
        "list_confidence_trajectory",
        "list_reverification_needs",
    ]


@pytest.mark.asyncio
async def test_memory_fabric_retrieve_similar_procedures_filters_review_state(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric = _fabric(tmp_path, monkeypatch)

    async def fake_query(*args: Any, **kwargs: Any) -> list[Context]:
        assert args == ("replay projection failure",)
        assert kwargs["session_id"] == "agent-1"
        return [
            Context(
                content="Procedure: rerun projection then replay tests",
                source="skill_memory",
                score=0.9,
                metadata={
                    "kind": "procedure",
                    "review_status": "accepted",
                    "citation": "eventloom://agent-1/events/4#dddddddddddd",
                },
            ),
            Context(
                content="Procedure: stale workaround",
                source="consolidation",
                score=0.8,
                metadata={
                    "candidate_type": "procedure",
                    "review_status": "rejected",
                    "citation": "eventloom://agent-1/events/5#eeeeeeeeeeee",
                },
            ),
            Context(
                content="Claim: projection was stale",
                source="consolidation",
                score=0.7,
                metadata={
                    "candidate_type": "claim",
                    "review_status": "accepted",
                    "citation": "eventloom://agent-1/events/6#ffffffffffff",
                },
            ),
        ]

    monkeypatch.setattr(fabric, "query", fake_query)

    result = await fabric.retrieve_similar_procedures(
        "replay projection failure",
        phase="planning",
        session_id="agent-1",
        limit=5,
    )

    assert result["procedure_count"] == 1
    assert result["procedures"][0]["content"] == "Procedure: rerun projection then replay tests"
    events = fabric.session_manager.get("agent-1").eventlog.read_all()
    assert events[-1].type == "reasoning.primitive.called"
    assert events[-1].payload["primitive"] == "retrieve_similar_procedures"
    assert events[-1].payload["result_count"] == 1


class _Checkout:
    def __init__(self, *, evidence: list[dict[str, Any]]) -> None:
        self.evidence = evidence
        self.current_facts: list[dict[str, Any]] = []


def _fabric(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> MemoryFabric:
    fabric = MemoryFabric(eventloom_path=str(tmp_path / "eventloom"))

    async def no_project(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(fabric, "_project_event", no_project)
    monkeypatch.setattr(fabric, "_append_generated_inferences", no_project)
    fabric._connected = True
    return fabric
