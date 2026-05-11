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
