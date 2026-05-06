"""Tests for zaxy.extract — hybrid extraction engine.

Tests cover rule-based extractors, the registry, and the generic fallback.
Every registered extractor gets exercised."""

from __future__ import annotations

from zaxy.event import Event
from zaxy.extract import (
    ExtractionResult,
    extract,
    register,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_event(event_type: str, payload: dict, actor: str = "test") -> Event:
    """Build an Event with a dummy hash for extraction tests."""
    return Event(
        seq=1,
        timestamp="2024-01-01T00:00:00Z",
        type=event_type,
        actor=actor,
        payload=payload,
        hash="a" * 64,
    )


# ------------------------------------------------------------------
# Registry tests
# ------------------------------------------------------------------

class TestRegistry:
    """Tests for the extractor registry."""

    def test_register_and_extract(self) -> None:
        """Registered extractors should be called for matching event types."""
        calls: list[str] = []

        @register("custom.test")
        def _extract(e: Event) -> ExtractionResult:
            calls.append(e.type)
            return ExtractionResult(entities=[], edges=[], source_event_seq=e.seq)

        ev = _make_event("custom.test", {})
        result = extract(ev)
        assert calls == ["custom.test"]
        assert result.source_event_seq == 1

    def test_unknown_event_uses_fallback(self) -> None:
        """Unregistered event types should fall back to generic identity."""
        ev = _make_event("unknown.event", {"foo": "bar"})
        result = extract(ev)
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "event"
        assert result.edges == []

    def test_fallback_entity_name_includes_seq(self) -> None:
        """Fallback entity name should be deterministic."""
        ev = _make_event("x.y", {})
        ev2 = ev.model_copy(update={"seq": 42})
        result = extract(ev2)
        assert result.entities[0].name == "event:x.y:42"


# ------------------------------------------------------------------
# Built-in extractor tests
# ------------------------------------------------------------------

class TestGoalCreated:
    """Tests for goal.created extractor."""

    def test_extracts_goal_and_actor(self) -> None:
        """Should create goal and actor entities plus a created_goal edge."""
        ev = _make_event("goal.created", {"title": "Ship MVP"}, actor="user")
        result = extract(ev)
        names = {e.name for e in result.entities}
        assert "Ship MVP" in names
        assert "user" in names

    def test_default_title(self) -> None:
        """Missing title should default to 'untitled'."""
        ev = _make_event("goal.created", {}, actor="user")
        result = extract(ev)
        assert any(e.name == "untitled" for e in result.entities)

    def test_edge(self) -> None:
        """Edge should link actor to goal."""
        ev = _make_event("goal.created", {"title": "T"}, actor="alice")
        result = extract(ev)
        assert len(result.edges) == 1
        edge = result.edges[0]
        assert edge.source == "alice"
        assert edge.target == "T"
        assert edge.relation_type == "created_goal"

    def test_summary_uses_description(self) -> None:
        """Goal summaries should preserve descriptive context for embeddings."""
        ev = _make_event(
            "goal.created",
            {"title": "Ship MVP", "description": "Get product to market"},
            actor="alice",
        )
        result = extract(ev)
        goal = next(e for e in result.entities if e.entity_type == "goal")
        assert goal.summary == "Get product to market"


class TestTaskProposed:
    """Tests for task.proposed extractor."""

    def test_extracts_task_and_actor(self) -> None:
        ev = _make_event("task.proposed", {"taskId": "t1", "title": "Do it"}, actor="codex")
        result = extract(ev)
        names = {e.name for e in result.entities}
        assert "t1" in names
        assert "codex" in names

    def test_default_task_id(self) -> None:
        """Missing taskId should use task_{seq}."""
        ev = _make_event("task.proposed", {}, actor="codex")
        result = extract(ev)
        assert any(e.name == "task_1" for e in result.entities)

    def test_edge(self) -> None:
        ev = _make_event("task.proposed", {"taskId": "t1"}, actor="bot")
        result = extract(ev)
        assert result.edges[0].relation_type == "proposed_task"

    def test_summary_uses_task_summary(self) -> None:
        ev = _make_event(
            "task.proposed",
            {"taskId": "t1", "summary": "Design landing page"},
            actor="bot",
        )
        result = extract(ev)
        task = next(e for e in result.entities if e.entity_type == "task")
        assert task.summary == "Design landing page"


class TestTaskClaimed:
    """Tests for task.claimed extractor."""

    def test_links_actor_to_task(self) -> None:
        ev = _make_event("task.claimed", {"taskId": "t1"}, actor="agent-a")
        result = extract(ev)
        assert len(result.edges) == 1
        assert result.edges[0].source == "agent-a"
        assert result.edges[0].target == "t1"
        assert result.edges[0].relation_type == "claimed_task"


class TestTaskCompleted:
    """Tests for task.completed extractor."""

    def test_links_actor_to_task(self) -> None:
        ev = _make_event("task.completed", {"taskId": "t1"}, actor="agent-b")
        result = extract(ev)
        assert result.edges[0].relation_type == "completed_task"


class TestPreferenceChanged:
    """Tests for user.preference_changed extractor."""

    def test_extracts_user_and_preference(self) -> None:
        ev = _make_event(
            "user.preference_changed",
            {"userId": "u42", "key": "theme", "value": "dark"},
            actor="u42",
        )
        result = extract(ev)
        names = {e.name for e in result.entities}
        assert "u42" in names
        assert "u42:theme" in names

    def test_default_user_id(self) -> None:
        """Missing userId should fall back to actor."""
        ev = _make_event("user.preference_changed", {"key": "lang"}, actor="alice")
        result = extract(ev)
        assert any(e.name == "alice" for e in result.entities)

    def test_preference_edge(self) -> None:
        ev = _make_event(
            "user.preference_changed",
            {"userId": "u1", "key": "theme"},
            actor="u1",
        )
        result = extract(ev)
        edge = result.edges[0]
        assert edge.source == "u1"
        assert edge.target == "u1:theme"
        assert edge.relation_type == "has_theme"

    def test_summary_includes_preference_value(self) -> None:
        ev = _make_event(
            "user.preference_changed",
            {"userId": "u1", "key": "theme", "value": "dark"},
            actor="u1",
        )
        result = extract(ev)
        preference = next(e for e in result.entities if e.entity_type == "preference")
        assert preference.summary == "theme=dark"


# ------------------------------------------------------------------
# Integration / sanity tests
# ------------------------------------------------------------------

class TestExtractionSanity:
    """Cross-cutting sanity checks."""

    def test_all_results_have_source_event_seq(self) -> None:
        """Every result should preserve the originating event sequence."""
        for event_type, payload in [
            ("goal.created", {"title": "X"}),
            ("task.proposed", {"taskId": "t1"}),
            ("task.claimed", {"taskId": "t1"}),
            ("task.completed", {"taskId": "t1"}),
            ("user.preference_changed", {"key": "k"}),
            ("unknown.type", {}),
        ]:
            ev = _make_event(event_type, payload)
            ev2 = ev.model_copy(update={"seq": 99})
            result = extract(ev2)
            assert result.source_event_seq == 99

    def test_observed_at_matches_event_timestamp(self) -> None:
        """Extracted entities should inherit the event timestamp."""
        ev = _make_event("goal.created", {"title": "T"})
        ev2 = ev.model_copy(update={"timestamp": "2024-06-15T12:00:00Z"})
        result = extract(ev2)
        for entity in result.entities:
            assert entity.observed_at == "2024-06-15T12:00:00Z"

    def test_no_duplicate_entities_in_result(self) -> None:
        """A single event should not produce duplicate entity names in result.

        Note: different events *can* produce the same name (e.g. same actor),
        but graph.py handles idempotency / upsert.
        """
        ev = _make_event("goal.created", {"title": "T"}, actor="alice")
        result = extract(ev)
        names = [e.name for e in result.entities]
        assert len(names) == len(set(names))

    def test_edges_reference_existing_entities(self) -> None:
        """Every edge source/target should appear in the entities list."""
        ev = _make_event("goal.created", {"title": "T"}, actor="alice")
        result = extract(ev)
        entity_names = {e.name for e in result.entities}
        for edge in result.edges:
            assert edge.source in entity_names
            assert edge.target in entity_names
