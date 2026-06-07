"""Tests for conservative inferred-edge event producers."""

from __future__ import annotations

from zaxy.event import Event
from zaxy.inference import build_inferred_edge_events


def _event(event_type: str, payload: dict) -> Event:
    return Event(
        seq=7,
        timestamp="2024-01-01T00:00:00Z",
        type=event_type,
        actor="codex",
        thread="agent-1",
        payload=payload,
        hash="b" * 64,
    )


def _outcome_event(event_type: str, payload: dict[str, object]) -> Event:
    return Event(
        seq=9,
        timestamp="2026-06-07T12:00:00Z",
        type=event_type,
        actor="assistant",
        payload=payload,
        prev_hash="0" * 64,
        hash="f" * 64,
        thread="agent-1",
    )


def test_task_completion_with_cited_decision_emits_inferred_edge_event() -> None:
    """Task completions can produce inferred edges only with explicit decision citation."""
    event = _event(
        "task.completed",
        {
            "taskId": "task-7",
            "summary": "Implemented Memory Checkout.",
            "decision": "Use Memory Checkout as the model-facing state contract",
            "decision_event_seq": 5,
            "decision_event_hash": "a" * 64,
        },
    )

    generated = build_inferred_edge_events(event)

    assert generated == [
        {
            "event_type": "inference.edge.generated",
            "actor": "zaxy-inference",
            "payload": {
                "source": {
                    "name": "task-7",
                    "entity_type": "task",
                    "summary": "Implemented Memory Checkout.",
                },
                "target": {
                    "name": "Use Memory Checkout as the model-facing state contract",
                    "entity_type": "decision",
                },
                "relation_type": "likely_implemented_decision",
                "confidence": 0.86,
                "inference_method": "task_completed_decision_citation_v1",
                "evidence": {
                    "source_event_seq": 7,
                    "source_event_hash": "b" * 64,
                    "decision_event_seq": 5,
                    "decision_event_hash": "a" * 64,
                    "reason": "task.completed explicitly cited a decision Eventloom event",
                },
            },
            "thread": "agent-1",
        }
    ]


def test_outcome_explained_event_generates_cited_causal_edge() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": 0.92,
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                    "reason": "The command output contained the failure.",
                },
            },
        )
    )

    assert generated == [
        {
            "event_type": "causal.edge.generated",
            "actor": "zaxy-causal",
            "payload": {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_caused",
                "confidence": 0.92,
                "causal_method": "explicit_outcome_explanation_v1",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                    "reason": "The command output contained the failure.",
                },
            },
            "thread": "agent-1",
        }
    ]


def test_outcome_explained_event_without_citation_generates_nothing() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": 0.92,
                "evidence": {"reason": "No Eventloom citation."},
            },
        )
    )

    assert generated == []


def test_outcome_explained_event_uses_default_evidence_reason() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": 0.92,
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                },
            },
        )
    )

    assert generated[0]["payload"]["evidence"]["reason"] == (
        "outcome.explained explicitly cited Eventloom evidence"
    )


def test_outcome_explained_event_does_not_infer_from_text() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "summary": "The pytest command caused a test failure.",
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                },
            },
        )
    )

    assert generated == []


def test_outcome_explained_event_with_unsupported_relation_generates_nothing() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "likely_caused",
                "confidence": 0.92,
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                },
            },
        )
    )

    assert generated == []


def test_outcome_explained_event_with_bool_confidence_generates_nothing() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": True,
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                },
            },
        )
    )

    assert generated == []


def test_outcome_explained_event_with_string_confidence_generates_nothing() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": "0.92",
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                },
            },
        )
    )

    assert generated == []


def test_outcome_explained_event_with_malformed_entity_refs_generates_nothing() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": 0.92,
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                },
            },
        )
    )

    assert generated == []


def test_outcome_explained_event_with_invalid_source_hash_generates_nothing() -> None:
    generated = build_inferred_edge_events(
        _outcome_event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": 0.92,
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "F" * 64,
                },
            },
        )
    )

    assert generated == []


def test_task_completion_without_decision_citation_emits_no_inference() -> None:
    """The producer should not infer links from uncited task text."""
    event = _event(
        "task.completed",
        {
            "taskId": "task-7",
            "summary": "Implemented Memory Checkout after the architecture decision.",
            "decision": "Use Memory Checkout as the model-facing state contract",
        },
    )

    assert build_inferred_edge_events(event) == []


def test_inference_contradiction_emits_retraction_event() -> None:
    """Contradicting evidence should produce a retraction event for inferred edges."""
    event = _event(
        "inference.edge.contradicted",
        {
            "source": {"name": "task-7", "entity_type": "task"},
            "target": {"name": "Use Memory Checkout", "entity_type": "decision"},
            "relation_type": "likely_implemented_decision",
            "valid_from": "2024-01-01T00:00:00Z",
            "original_event_seq": 8,
            "original_event_hash": "a" * 64,
            "reason": "Later source showed the task implemented a different decision.",
        },
    )

    generated = build_inferred_edge_events(event)

    assert generated == [
        {
            "event_type": "inference.edge.retracted",
            "actor": "zaxy-inference",
            "payload": {
                "source": {"name": "task-7", "entity_type": "task"},
                "target": {"name": "Use Memory Checkout", "entity_type": "decision"},
                "relation_type": "likely_implemented_decision",
                "valid_from": "2024-01-01T00:00:00Z",
                "valid_to": "2024-01-01T00:00:00Z",
                "confidence": 0.0,
                "inference_method": "contradicting_evidence_retraction_v1",
                "evidence": {
                    "source_event_seq": 7,
                    "source_event_hash": "b" * 64,
                    "original_event_seq": 8,
                    "original_event_hash": "a" * 64,
                    "reason": "Later source showed the task implemented a different decision.",
                },
            },
            "thread": "agent-1",
        }
    ]
