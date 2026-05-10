"""Tests for deterministic memory working-set projection."""

from __future__ import annotations

from types import SimpleNamespace

from zaxy.context import Context
from zaxy.working_set import build_working_set, format_working_set


def test_build_working_set_projects_replay_events_and_source_anchors() -> None:
    """Working set should expose semantic memory without dumping raw history."""
    events = [
        SimpleNamespace(
            seq=1,
            type="goal.created",
            actor="user",
            payload={"title": "Ship automatic capture"},
            hash="a" * 64,
            thread="agent",
        ),
        SimpleNamespace(
            seq=2,
            type="decision.recorded",
            actor="assistant",
            payload={"decision": "Use Eventloom as source of truth."},
            hash="b" * 64,
            thread="agent",
        ),
        SimpleNamespace(
            seq=3,
            type="transcript.turn",
            actor="assistant",
            payload={"content": "This full turn should stay out of working-set items."},
            hash="c" * 64,
            thread="agent",
        ),
    ]
    contexts = [
        Context(
            content="assistant: Exact source mentions hook capture.",
            source="verbatim",
            score=1.0,
            metadata={
                "citation": "eventloom://agent/events/4#dddd",
                "assembly_lane": "verbatim",
            },
        )
    ]

    working_set = build_working_set(events, contexts, max_items=5)

    assert [item.category for item in working_set.items] == [
        "goal",
        "decision",
        "source_anchor",
    ]
    assert working_set.items[0].summary == "Ship automatic capture"
    assert working_set.items[0].citation == "eventloom://agent/events/1#aaaaaaaaaaaa"
    assert working_set.items[1].summary == "Use Eventloom as source of truth."
    assert working_set.items[2].citation == "eventloom://agent/events/4#dddd"
    assert working_set.truncated is False


def test_build_working_set_is_bounded() -> None:
    """Working set should cap projected items to protect the context window."""
    events = [
        SimpleNamespace(
            seq=idx,
            type="task.completed",
            actor="assistant",
            payload={"summary": f"Task {idx}"},
            hash=str(idx) * 64,
            thread="agent",
        )
        for idx in range(1, 6)
    ]

    working_set = build_working_set(events, [], max_items=3)

    assert [item.summary for item in working_set.items] == ["Task 1", "Task 2", "Task 3"]
    assert working_set.truncated is True


def test_format_working_set_renders_compact_section() -> None:
    """Formatted working set should be prompt-ready and citation-aware."""
    event = SimpleNamespace(
        seq=1,
        type="goal.created",
        actor="user",
        payload={"title": "Reduce context collapse"},
        hash="f" * 64,
        thread="agent",
    )

    output = format_working_set(build_working_set([event], [], max_items=5))

    assert "# Active Memory Working Set" in output
    assert "- goal: Reduce context collapse (eventloom://agent/events/1#ffffffffffff)" in output
