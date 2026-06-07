"""Tests for zaxy.extract — hybrid extraction engine.

Tests cover rule-based extractors, the registry, and the generic fallback.
Every registered extractor gets exercised."""

from __future__ import annotations

import pytest

from zaxy.event import Event
from zaxy.extract import (
    ExtractedEdge,
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
# Edge model tests
# ------------------------------------------------------------------

class TestExtractedEdge:
    """Tests for edge provenance metadata."""

    def test_defaults_to_deterministic(self) -> None:
        """Edges should be deterministic unless an extractor marks them inferred."""
        edge = ExtractedEdge(
            source="Alice",
            target="Goal1",
            relation_type="created_goal",
            valid_from="2024-01-01T00:00:00Z",
        )

        assert edge.inferred is False
        assert edge.confidence == 1.0
        assert edge.inference_method is None
        assert edge.evidence == {}


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

    def test_unknown_event_preserves_authority_metadata(self) -> None:
        """Generic event projection should keep checkout-policy metadata structured."""
        ev = _make_event(
            "decision.accepted",
            {
                "summary": "Worker-local accepted-looking row should stay diagnostic.",
                "authority_scope": "worker",
                "status": "accepted",
                "promoted": False,
                "stale": True,
                "superseded_by": "mission:4",
            },
        )

        result = extract(ev)

        assert result.entities[0].properties is not None
        assert result.entities[0].properties["authority_scope"] == "worker"
        assert result.entities[0].properties["status"] == "accepted"
        assert result.entities[0].properties["promoted"] is False
        assert result.entities[0].properties["stale"] is True
        assert result.entities[0].properties["superseded_by"] == "mission:4"
        assert result.edges == []

    def test_fallback_entity_name_includes_seq(self) -> None:
        """Fallback entity name should be deterministic."""
        ev = _make_event("x.y", {})
        ev2 = ev.model_copy(update={"seq": 42})
        result = extract(ev2)
        assert result.entities[0].name == "event:x.y:42"

    def test_unknown_event_summary_includes_safe_payload_text(self) -> None:
        """Fallback summaries should keep unknown typed events keyword-searchable."""
        ev = _make_event(
            "tool.result",
            {
                "tool": "pytest",
                "status": "failed",
                "findings": ["Neo4j unavailable", "Pathlight trace recovered"],
                "nested": {"ignored": "not flattened"},
                "secret": "sk-123",
            },
            actor="assistant",
        )
        result = extract(ev)
        summary = result.entities[0].summary
        assert "assistant emitted tool.result" in summary
        assert "tool=pytest" in summary
        assert "status=failed" in summary
        assert "Neo4j unavailable" in summary
        assert "Pathlight trace recovered" in summary
        assert "nested" not in summary
        assert "sk-123" not in summary

    def test_unknown_event_carries_safe_retention_metadata(self) -> None:
        """Fallback event entities should keep retention metadata query policies can consume."""
        ev = _make_event(
            "unknown.event",
            {
                "summary": "temporary note",
                "expires_at": "2024-02-01T00:00:00Z",
                "importance": 0.4,
                "last_reinforced_at": "2024-01-15T00:00:00Z",
                "reinforcement_count": 2,
            },
        )

        result = extract(ev)

        assert result.entities[0].properties == {
            "expires_at": "2024-02-01T00:00:00Z",
            "importance": 0.4,
            "last_reinforced_at": "2024-01-15T00:00:00Z",
            "reinforcement_count": 2,
        }

    def test_extract_attaches_previous_event_hash(self) -> None:
        """Extraction provenance should preserve Eventloom's linear hash-chain link."""
        ev = _make_event("unknown.event", {"summary": "Project hash-chain edges."})
        ev = ev.model_copy(update={"prev_hash": "a" * 64, "hash": "b" * 64})

        result = extract(ev)

        assert result.source_event_hash == "b" * 64
        assert result.source_event_prev_hash == "a" * 64


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
        assert task.properties == {"taskId": "t1"}

    def test_links_task_to_goal_when_goal_title_present(self) -> None:
        """Structured task proposals should preserve task-goal graph links."""
        ev = _make_event(
            "task.proposed",
            {"taskId": "t1", "summary": "Design landing page", "goalTitle": "Ship MVP"},
            actor="bot",
        )

        result = extract(ev)

        assert any(e.name == "Ship MVP" and e.entity_type == "goal" for e in result.entities)
        assert any(
            edge.source == "Ship MVP"
            and edge.target == "t1"
            and edge.relation_type == "has_task"
            for edge in result.edges
        )


class TestSkillMemory:
    """Tests for skill lifecycle event extractors."""

    def test_skill_proposed_projects_skill_and_version(self) -> None:
        """skill.proposed should create a reusable procedure and its first version."""
        ev = _make_event(
            "skill.proposed",
            {
                "skill_id": "python-test-first",
                "name": "Python test-first implementation",
                "version": "1",
                "summary": "Write the failing pytest before implementation.",
                "procedure": ["Write focused failing test", "Run pytest", "Implement minimum code"],
                "applicability": ["Python feature work", "bug fixes"],
                "citations": ["eventloom://zaxy-default/events/12#abc"],
            },
            actor="agent",
        )

        result = extract(ev)

        skill = next(entity for entity in result.entities if entity.entity_type == "skill")
        version = next(entity for entity in result.entities if entity.entity_type == "skill_version")
        assert skill.name == "skill:python-test-first"
        assert skill.summary == "Python test-first implementation"
        assert skill.properties == {"skill_id": "python-test-first"}
        assert version.name == "skill:python-test-first:v1"
        assert version.summary == "Write the failing pytest before implementation."
        assert version.properties == {
            "skill_id": "python-test-first",
            "version": "1",
            "procedure": ["Write focused failing test", "Run pytest", "Implement minimum code"],
            "applicability": ["Python feature work", "bug fixes"],
            "citations": ["eventloom://zaxy-default/events/12#abc"],
            "status": "proposed",
        }
        assert any(
            edge.source == "skill:python-test-first"
            and edge.target == "skill:python-test-first:v1"
            and edge.relation_type == "has_version"
            for edge in result.edges
        )

    def test_skill_outcome_records_application_metrics(self) -> None:
        """skill.outcome_recorded should connect a skill version to audited outcome data."""
        ev = _make_event(
            "skill.outcome_recorded",
            {
                "skill_id": "python-test-first",
                "version": "1",
                "task": "fix retrieval scoring",
                "success_score": 0.95,
                "feedback": "used",
                "evidence": ["pytest tests/test_query.py::test_scoring -q"],
            },
            actor="agent",
        )

        result = extract(ev)

        outcome = next(entity for entity in result.entities if entity.entity_type == "skill_outcome")
        assert outcome.name == "skill:python-test-first:v1:outcome:1"
        assert outcome.properties == {
            "skill_id": "python-test-first",
            "version": "1",
            "task": "fix retrieval scoring",
            "success_score": 0.95,
            "feedback": "used",
            "evidence": ["pytest tests/test_query.py::test_scoring -q"],
        }
        assert any(
            edge.source == "skill:python-test-first:v1"
            and edge.target == "skill:python-test-first:v1:outcome:1"
            and edge.relation_type == "recorded_outcome"
            for edge in result.edges
        )

    def test_skill_contradicted_preserves_rollback_metadata(self) -> None:
        """skill.contradicted should keep rollback and contradiction evidence on the version."""
        ev = _make_event(
            "skill.contradicted",
            {
                "skill_id": "deploy-cache-check",
                "version": "1",
                "summary": "Old cache validation process misses invalidation races.",
                "failure_modes": ["misses cache invalidation race"],
                "rollback": "Use deploy-cache-check v0 until cache race is resolved.",
                "contradiction_reason": "Failed release validation after cache race.",
                "evidence": ["pytest tests/test_release.py::test_cache_race -q"],
            },
            actor="agent",
        )

        result = extract(ev)

        version = next(entity for entity in result.entities if entity.entity_type == "skill_version")
        assert version.properties["status"] == "contradicted"
        assert version.properties["failure_modes"] == ["misses cache invalidation race"]
        assert version.properties["rollback"] == "Use deploy-cache-check v0 until cache race is resolved."
        assert version.properties["contradiction_reason"] == "Failed release validation after cache race."
        assert version.properties["evidence"] == ["pytest tests/test_release.py::test_cache_race -q"]


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
        task = next(e for e in result.entities if e.entity_type == "task")
        assert task.properties == {"taskId": "t1"}

    def test_uses_task_and_summary_payload(self) -> None:
        ev = _make_event(
            "task.completed",
            {
                "task": "Debug and repair Zaxy MCP startup in Codex.",
                "summary": "Fixed startup_timeout_sec and Pathlight async trace compatibility.",
            },
            actor="assistant",
        )
        result = extract(ev)
        task = next(e for e in result.entities if e.entity_type == "task")
        assert task.name == "Debug and repair Zaxy MCP startup in Codex."
        assert task.summary == "Fixed startup_timeout_sec and Pathlight async trace compatibility."

    def test_carries_retention_metadata(self) -> None:
        ev = _make_event(
            "task.completed",
            {
                "taskId": "t1",
                "summary": "Done",
                "expires_at": "2024-03-01T00:00:00Z",
                "importance": "0.8",
            },
        )

        result = extract(ev)

        task = next(e for e in result.entities if e.entity_type == "task")
        assert task.properties == {
            "expires_at": "2024-03-01T00:00:00Z",
            "importance": 0.8,
            "taskId": "t1",
        }


class TestDecisionMade:
    """Tests for decision.made extractor."""

    def test_extracts_decision_with_rationale(self) -> None:
        ev = _make_event(
            "decision.made",
            {
                "decision": "Preserve the previous chat as a structured Eventloom trace.",
                "rationale": [
                    "Typed events are easier to replay.",
                    "Raw transcript remains available in resumed chat.",
                ],
            },
            actor="assistant",
        )
        result = extract(ev)
        decision = next(e for e in result.entities if e.entity_type == "decision")
        assert decision.name == "Preserve the previous chat as a structured Eventloom trace."
        assert "Typed events are easier to replay." in decision.summary
        assert "Raw transcript remains available in resumed chat." in decision.summary
        assert any(e.name == "assistant" and e.entity_type == "actor" for e in result.entities)
        assert result.edges[0].relation_type == "made_decision"


class TestContextPolicy:
    """Tests for context.policy extractor."""

    def test_extracts_policy_with_instruction_summary(self) -> None:
        ev = _make_event(
            "context.policy",
            {
                "source": "AGENTS.md",
                "project": "Zaxy",
                "instructions": [
                    "Use Eventloom append-only JSONL as immutable source of truth.",
                    "Write tests first for public functions and behavior changes.",
                ],
            },
            actor="user",
        )
        result = extract(ev)
        policy = next(e for e in result.entities if e.entity_type == "context_policy")
        assert policy.name == "Zaxy:AGENTS.md"
        assert "Use Eventloom append-only JSONL" in policy.summary
        assert policy.properties == {"source": "AGENTS.md", "project": "Zaxy"}
        assert result.edges[0].relation_type == "set_context_policy"


class TestMemoryReinforced:
    """Tests for memory.reinforced extractor."""

    def test_extracts_reinforced_memory_metadata(self) -> None:
        ev = _make_event(
            "memory.reinforced",
            {
                "entity_name": "Use retention metadata",
                "entity_type": "decision",
                "summary": "Still relevant after review.",
                "importance": 0.9,
                "reinforcement_count": 4,
                "purpose": {
                    "profile": "coordinate",
                    "expected_action": "brief_promote_or_handoff",
                    "evidence_policy": "accepted_parent_state_with_citations_required",
                },
                "authority_scope": "parent-accepted",
                "mission_id": "auth-main",
                "finding_id": "finding-api",
                "outcome": "supported_handoff",
            },
            actor="assistant",
        )

        result = extract(ev)

        entity = next(e for e in result.entities if e.entity_type == "decision")
        assert entity.name == "Use retention metadata"
        assert entity.summary == "Still relevant after review."
        assert entity.properties == {
            "importance": 0.9,
            "last_reinforced_at": "2024-01-01T00:00:00Z",
            "purpose_profile": "coordinate",
            "purpose_expected_action": "brief_promote_or_handoff",
            "purpose_evidence_policy": "accepted_parent_state_with_citations_required",
            "authority_scope": "parent-accepted",
            "mission_id": "auth-main",
            "finding_id": "finding-api",
            "outcome": "supported_handoff",
            "reinforcement_count": 4,
        }
        assert result.edges[0].relation_type == "reinforced_memory"


class TestMemoryFeedback:
    """Tests for memory.feedback extractor."""

    def test_memory_feedback_projects_negative_outcome_metadata(self) -> None:
        """Negative feedback should remain auditable and linked to target memory."""
        event = _make_event(
            "memory.feedback",
            {
                "entity_name": "migration retry",
                "entity_type": "decision",
                "feedback": "irrelevant",
                "outcome": "caused_regression",
                "citation": "eventloom://agent-1/events/9#bbbb",
                "purpose": {
                    "profile": "coding",
                    "expected_action": "implement_or_verify",
                },
            },
            actor="assistant",
        )

        result = extract(event)

        feedback = next(entity for entity in result.entities if entity.entity_type == "memory_feedback")
        target = next(entity for entity in result.entities if entity.name == "migration retry")
        assert feedback.properties == {
            "purpose_profile": "coding",
            "purpose_expected_action": "implement_or_verify",
            "outcome": "caused_regression",
            "feedback": "irrelevant",
            "citation": "eventloom://agent-1/events/9#bbbb",
            "last_feedback_at": "2024-01-01T00:00:00Z",
        }
        assert target.properties["feedback"] == "irrelevant"
        assert any(
            edge.source == feedback.name
            and edge.target == "migration retry"
            and edge.relation_type == "feedback_about_memory"
            for edge in result.edges
        )


class TestMemoryEvidenceFeedback:
    """Tests for row-level synthesis evidence feedback extractors."""

    def test_extracts_reinforced_synthesis_evidence_metadata(self) -> None:
        ev = _make_event(
            "memory.evidence.reinforced",
            {
                "query": "How much did I spend on bike expenses?",
                "outcome": "used",
                "fact_id": "currency:0:0",
                "source_group": "answer-1",
                "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                "reason": "row supported arithmetic",
            },
            actor="assistant",
        )

        result = extract(ev)

        entity = next(e for e in result.entities if e.entity_type == "synthesis_evidence")
        assert entity.name == "currency:0:0"
        assert entity.summary == "row supported arithmetic"
        assert entity.properties == {
            "outcome": "used",
            "source_group": "answer-1",
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "last_reinforced_at": "2024-01-01T00:00:00Z",
        }
        assert result.edges[0].relation_type == "reinforced_synthesis_evidence"
        assert result.edges[1].relation_type == "cites_source_group"

    def test_extracts_excluded_synthesis_evidence_metadata(self) -> None:
        ev = _make_event(
            "memory.evidence.excluded",
            {
                "outcome": "excluded",
                "fact_id": "currency:duplicate",
                "source_group": "answer-4",
                "reason": "duplicate source row",
            },
            actor="assistant",
        )

        result = extract(ev)

        entity = next(e for e in result.entities if e.entity_type == "synthesis_evidence")
        assert entity.name == "currency:duplicate"
        assert entity.properties == {
            "outcome": "excluded",
            "source_group": "answer-4",
            "last_excluded_at": "2024-01-01T00:00:00Z",
        }
        assert result.edges[0].relation_type == "excluded_synthesis_evidence"


class TestMemorySynthesisArtifacts:
    """Tests for synthesis artifact and candidate outcome extractors."""

    def test_extracts_synthesis_artifact_candidates_and_ledger_rows(self) -> None:
        ev = _make_event(
            "memory.synthesis.artifact.created",
            {
                "schema_version": "synthesis_artifact_v1",
                "artifact_id": "sha256:artifact",
                "session_id": "release-rc1",
                "query": "Compose accepted release findings.",
                "answer_candidates": [
                    {
                        "rank": 1,
                        "type": "coordinate_handoff",
                        "answer_key": "coordinate_handoff_answer",
                        "answer": "Accepted cause: expired JWKS cache.",
                        "confidence": 0.9,
                        "support_source_ids": ["auth-api:finding:1"],
                    }
                ],
                "ledger_rows": [
                    {
                        "fact_id": "auth-api:finding:1",
                        "source_group": "auth-api:finding:1",
                        "citation": "eventloom://release-rc1/events/4#aaaaaaaaaaaa",
                        "include_reason": "accepted_parent_state",
                    }
                ],
                "support_packet": {"source_groups": ["auth-api:finding:1"]},
            },
            actor="coordinator",
        )

        result = extract(ev)

        artifact = next(e for e in result.entities if e.entity_type == "synthesis_artifact")
        assert artifact.name == "sha256:artifact"
        assert artifact.properties == {
            "schema_version": "synthesis_artifact_v1",
            "session_id": "release-rc1",
            "answer_candidate_count": 1,
            "ledger_row_count": 1,
            "support_source_group_count": 1,
        }
        candidate = next(e for e in result.entities if e.entity_type == "synthesis_answer_candidate")
        assert candidate.name == "sha256:artifact:candidate:coordinate_handoff_answer"
        row = next(e for e in result.entities if e.entity_type == "synthesis_ledger_row")
        assert row.name == "sha256:artifact:ledger:auth-api:finding:1"
        assert [edge.relation_type for edge in result.edges] == [
            "created_synthesis_artifact",
            "artifact_has_answer_candidate",
            "candidate_supported_by_source_group",
            "artifact_has_ledger_row",
        ]

    def test_extracts_synthesis_candidate_outcome_feedback(self) -> None:
        ev = _make_event(
            "memory.synthesis.used",
            {
                "query": "Compose accepted release findings.",
                "outcome": "used",
                "answer_candidate": {
                    "rank": 1,
                    "type": "coordinate_handoff",
                    "answer_key": "coordinate_handoff_answer",
                    "answer": "Accepted cause: expired JWKS cache.",
                    "support_source_ids": ["auth-api:finding:1"],
                },
                "support_source_ids": ["auth-api:finding:1"],
            },
            actor="coordinator",
        )

        result = extract(ev)

        candidate = next(e for e in result.entities if e.entity_type == "synthesis_answer_candidate")
        assert candidate.name == "candidate_feedback:candidate:coordinate_handoff_answer"
        assert candidate.properties["outcome"] == "used"
        assert [edge.relation_type for edge in result.edges] == [
            "recorded_synthesis_used",
            "candidate_supported_by_source_group",
        ]


class TestHookCheckpoint:
    """Tests for hook.checkpoint extractor."""

    def test_extracts_checkpoint_with_summary_and_session_edge(self) -> None:
        ev = _make_event(
            "hook.checkpoint",
            {
                "trigger": "checkpoint",
                "source": "codex",
                "summary": "Finished hook install mode.",
                "reason": "manual",
                "turn_count": 7,
                "workspace": "/repo",
            },
            actor="zaxy-hook",
        )
        ev = ev.model_copy(update={"thread": "agent-1"})

        result = extract(ev)

        checkpoint = next(e for e in result.entities if e.entity_type == "hook_checkpoint")
        assert checkpoint.name == "agent-1:checkpoint:1"
        assert checkpoint.summary == "Finished hook install mode."
        assert checkpoint.properties == {
            "session_id": "agent-1",
            "source": "codex",
            "reason": "manual",
            "turn_count": 7,
            "workspace": "/repo",
        }
        assert result.edges[0].source == "agent-1"
        assert result.edges[0].target == "agent-1:checkpoint:1"
        assert result.edges[0].relation_type == "recorded_checkpoint"

    def test_checkpoint_links_explicit_task_id(self) -> None:
        """Checkpoint observations should link to an explicit task id when present."""
        ev = _make_event(
            "hook.checkpoint",
            {"summary": "Task checkpoint.", "task_id": "task-7"},
            actor="zaxy-hook",
        )
        ev = ev.model_copy(update={"thread": "agent-1"})

        result = extract(ev)

        assert any(e.name == "task-7" and e.entity_type == "task" for e in result.entities)
        assert any(
            edge.source == "task-7"
            and edge.target == "agent-1:checkpoint:1"
            and edge.relation_type == "has_checkpoint"
            for edge in result.edges
        )


class TestInferredEdgeGenerated:
    """Tests for explicit inferred-edge event projection."""

    def test_projects_auditable_inferred_edge(self) -> None:
        """Generated inference events should project inferred edges with evidence."""
        ev = _make_event(
            "inference.edge.generated",
            {
                "source": {
                    "name": "task-7",
                    "entity_type": "task",
                    "summary": "Implement memory checkout",
                },
                "target": {
                    "name": "Use Memory Checkout as the model-facing state contract",
                    "entity_type": "decision",
                    "summary": "Checkout is the killer product decision.",
                },
                "relation_type": "likely_informed",
                "confidence": 0.82,
                "inference_method": "operator_review_v1",
                "evidence": {
                    "source_event_seq": 6676,
                    "reason": "Operator confirmed the task led to the decision.",
                },
            },
            actor="zaxy",
        )

        result = extract(ev)

        assert [(entity.name, entity.entity_type) for entity in result.entities] == [
            ("task-7", "task"),
            ("Use Memory Checkout as the model-facing state contract", "decision"),
        ]
        assert result.entities[0].summary == "Implement memory checkout"
        assert result.entities[1].summary == "Checkout is the killer product decision."
        edge = result.edges[0]
        assert edge.source == "task-7"
        assert edge.target == "Use Memory Checkout as the model-facing state contract"
        assert edge.relation_type == "likely_informed"
        assert edge.inferred is True
        assert edge.confidence == 0.82
        assert edge.inference_method == "operator_review_v1"
        assert edge.evidence == {
            "source_event_seq": 6676,
            "reason": "Operator confirmed the task led to the decision.",
        }

    def test_projects_inferred_edge_retraction(self) -> None:
        """Retraction events should close the inferred edge validity interval."""
        ev = _make_event(
            "inference.edge.retracted",
            {
                "source": {"name": "task-7", "entity_type": "task"},
                "target": {"name": "Use Memory Checkout", "entity_type": "decision"},
                "relation_type": "likely_informed",
                "valid_from": "2024-01-01T00:00:00Z",
                "valid_to": "2024-01-02T00:00:00Z",
                "confidence": 0.0,
                "inference_method": "contradicting_evidence_retraction_v1",
                "evidence": {
                    "original_event_seq": 7,
                    "reason": "Contradicting evidence arrived.",
                },
            },
        )

        result = extract(ev)

        edge = result.edges[0]
        assert edge.source == "task-7"
        assert edge.target == "Use Memory Checkout"
        assert edge.relation_type == "likely_informed"
        assert edge.valid_from == "2024-01-01T00:00:00Z"
        assert edge.valid_to == "2024-01-02T00:00:00Z"
        assert edge.inferred is True
        assert edge.confidence == 0.0
        assert edge.inference_method == "contradicting_evidence_retraction_v1"


class TestCausalEdgeGenerated:
    """Tests for explicit causal-edge event projection."""

    def test_extract_causal_edge_generated_projects_inferred_causal_relation(self) -> None:
        event = _make_event(
            "causal.edge.generated",
            {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_caused",
                "confidence": 0.91,
                "causal_method": "explicit_outcome_citation_v1",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {
                    "source_event_seq": 42,
                    "source_event_hash": "a" * 64,
                    "reason": "The command output contained the failure.",
                },
            },
        )

        result = extract(event)

        assert {entity.name for entity in result.entities} == {"command:pytest", "test failure"}
        assert result.edges == [
            ExtractedEdge(
                source="command:pytest",
                target="test failure",
                relation_type="causal_caused",
                valid_from=event.timestamp,
                inferred=True,
                confidence=0.91,
                inference_method="explicit_outcome_citation_v1",
                evidence={
                    "causal_relation_type": "caused",
                    "review_status": "proposed",
                    "authority_status": "non_authoritative",
                    "source_event_seq": 42,
                    "source_event_hash": "a" * 64,
                    "reason": "The command output contained the failure.",
                },
            )
        ]

    @pytest.mark.parametrize(
        "source_event_hash",
        ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64],
    )
    def test_causal_edge_generated_rejects_invalid_source_event_hash(
        self,
        source_event_hash: str,
    ) -> None:
        event = _make_event(
            "causal.edge.generated",
            {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_caused",
                "confidence": 0.91,
                "causal_method": "explicit_outcome_citation_v1",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {
                    "source_event_seq": 42,
                    "source_event_hash": source_event_hash,
                },
            },
        )

        with pytest.raises(ValueError, match="source_event_hash"):
            extract(event)

    def test_causal_edge_generated_rejects_missing_source_event_hash(self) -> None:
        event = _make_event(
            "causal.edge.generated",
            {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_caused",
                "confidence": 0.91,
                "causal_method": "explicit_outcome_citation_v1",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {"source_event_seq": 42},
            },
        )

        with pytest.raises(ValueError, match="source_event_hash"):
            extract(event)

    def test_causal_edge_generated_rejects_mismatched_graph_relation_type(self) -> None:
        event = _make_event(
            "causal.edge.generated",
            {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_enabled",
                "confidence": 0.91,
                "causal_method": "explicit_outcome_citation_v1",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {"source_event_seq": 42, "source_event_hash": "a" * 64},
            },
        )

        with pytest.raises(ValueError, match="graph_relation_type"):
            extract(event)

    def test_causal_edge_generated_rejects_unsupported_causal_relation_type(self) -> None:
        event = _make_event(
            "causal.edge.generated",
            {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "likely_informed",
                "graph_relation_type": "causal_likely_informed",
                "confidence": 0.91,
                "causal_method": "explicit_outcome_citation_v1",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {"source_event_seq": 42, "source_event_hash": "a" * 64},
            },
        )

        with pytest.raises(ValueError, match="causal relation_type"):
            extract(event)

    def test_causal_edge_generated_rejects_authoritative_status(self) -> None:
        event = _make_event(
            "causal.edge.generated",
            {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_caused",
                "confidence": 0.91,
                "causal_method": "explicit_outcome_citation_v1",
                "review_status": "proposed",
                "authority_status": "authoritative",
                "evidence": {"source_event_seq": 42, "source_event_hash": "a" * 64},
            },
        )

        with pytest.raises(ValueError, match="authority_status"):
            extract(event)

    def test_causal_edge_generated_rejects_string_confidence(self) -> None:
        event = _make_event(
            "causal.edge.generated",
            {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_caused",
                "confidence": "0.91",
                "causal_method": "explicit_outcome_citation_v1",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {"source_event_seq": 42, "source_event_hash": "a" * 64},
            },
        )

        with pytest.raises(ValueError, match="confidence"):
            extract(event)

    def test_causal_edge_generated_rejects_non_string_causal_method(self) -> None:
        event = _make_event(
            "causal.edge.generated",
            {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_caused",
                "confidence": 0.91,
                "causal_method": 123,
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {"source_event_seq": 42, "source_event_hash": "a" * 64},
            },
        )

        with pytest.raises(ValueError, match="causal method"):
            extract(event)

    def test_causal_edge_generated_snapshots_nested_evidence(self) -> None:
        payload = {
            "source": {"name": "command:pytest", "entity_type": "command"},
            "target": {"name": "test failure", "entity_type": "outcome"},
            "relation_type": "caused",
            "graph_relation_type": "causal_caused",
            "confidence": 0.91,
            "causal_method": "explicit_outcome_citation_v1",
            "review_status": "proposed",
            "authority_status": "non_authoritative",
            "evidence": {
                "source_event_seq": 42,
                "source_event_hash": "a" * 64,
                "details": {"line": 12, "message": "failed"},
            },
        }
        event = _make_event("causal.edge.generated", payload)

        result = extract(event)
        payload["evidence"]["details"]["message"] = "mutated"

        assert result.edges[0].evidence["details"] == {"line": 12, "message": "failed"}


class TestIssueDiagnosed:
    """Tests for issue.diagnosed extractor."""

    def test_extracts_issue_with_root_cause_and_evidence(self) -> None:
        ev = _make_event(
            "issue.diagnosed",
            {
                "issue": "memory_query returned no results for recent decisions",
                "root_cause": "decision payload text was not projected into graph summaries",
                "evidence": [
                    "memory_replay showed the event existed",
                    "exact query for event:decision.made:3 worked",
                ],
                "fix": "add typed extractors and reproject Eventloom",
            },
            actor="assistant",
        )
        result = extract(ev)
        issue = next(e for e in result.entities if e.entity_type == "issue")
        assert issue.name == "memory_query returned no results for recent decisions"
        assert "decision payload text was not projected" in issue.summary
        assert "exact query for event:decision.made:3 worked" in issue.summary
        assert issue.properties == {"status": "diagnosed"}
        assert result.edges[0].relation_type == "diagnosed_issue"


class TestVerificationRecorded:
    """Tests for verification.recorded extractor."""

    def test_extracts_verification_with_command_and_outcome(self) -> None:
        ev = _make_event(
            "verification.recorded",
            {
                "command": "pytest --no-cov -m 'not integration'",
                "outcome": "passed",
                "summary": "382 passed, 5 deselected",
                "evidence": ["exit code 0", "ruff clean"],
            },
            actor="assistant",
        )
        result = extract(ev)
        verification = next(e for e in result.entities if e.entity_type == "verification")
        assert verification.name == "pytest --no-cov -m 'not integration'"
        assert "passed" in verification.summary
        assert "382 passed, 5 deselected" in verification.summary
        assert "ruff clean" in verification.summary
        assert verification.properties == {"outcome": "passed"}
        assert result.edges[0].relation_type == "recorded_verification"


class TestHandoffCreated:
    """Tests for handoff.created extractor."""

    def test_extracts_handoff_with_next_steps_and_risks(self) -> None:
        ev = _make_event(
            "handoff.created",
            {
                "summary": "Zaxy MCP is online with temporal memory.",
                "next_steps": [
                    "Add remote MCP rate limiting",
                    "Add local-first embedding setup",
                ],
                "risks": ["Pathlight traces are currently sparse"],
            },
            actor="assistant",
        )
        result = extract(ev)
        handoff = next(e for e in result.entities if e.entity_type == "handoff")
        assert handoff.name == "handoff:1"
        assert "Zaxy MCP is online" in handoff.summary
        assert "Add remote MCP rate limiting" in handoff.summary
        assert "Pathlight traces are currently sparse" in handoff.summary
        assert handoff.properties == {"status": "created"}
        assert result.edges[0].relation_type == "created_handoff"


class TestCoordinationEvents:
    """Tests for multi-agent coordination event extractors."""

    def test_mission_created_projects_mission_and_actor(self) -> None:
        ev = _make_event(
            "coordination.mission.created",
            {"mission_id": "auth-main", "objective": "Ship auth refactor", "status": "active"},
            actor="lead",
        )

        result = extract(ev)

        mission = next(e for e in result.entities if e.entity_type == "mission")
        assert mission.name == "auth-main"
        assert mission.summary == "Ship auth refactor"
        assert mission.properties == {"status": "active"}
        assert result.edges[0].relation_type == "started_mission"

    def test_worker_created_links_mission_to_worker(self) -> None:
        ev = _make_event(
            "coordination.worker.created",
            {"mission_id": "auth-main", "worker_id": "auth-api", "status": "active"},
            actor="lead",
        )

        result = extract(ev)

        names = {entity.name for entity in result.entities}
        assert {"auth-main", "auth-api"} <= names
        assert next(e for e in result.entities if e.name == "auth-api").entity_type == "worker"
        assert result.edges[0].source == "auth-main"
        assert result.edges[0].target == "auth-api"
        assert result.edges[0].relation_type == "mission_has_worker"

    def test_assignment_created_links_worker_to_assignment(self) -> None:
        ev = _make_event(
            "coordination.assignment.created",
            {
                "mission_id": "auth-main",
                "worker_id": "auth-api",
                "assignment": "Trace API auth failures",
                "status": "assigned",
            },
            actor="lead",
        )

        result = extract(ev)

        assignment = next(e for e in result.entities if e.entity_type == "assignment")
        assert assignment.name == "auth-main:auth-api:assignment:1"
        assert assignment.summary == "Trace API auth failures"
        assert assignment.properties == {"status": "assigned", "mission_id": "auth-main", "worker_id": "auth-api"}
        assert result.edges[0].relation_type == "worker_has_assignment"

    def test_finding_reported_projects_finding_with_evidence_and_confidence(self) -> None:
        ev = _make_event(
            "coordination.finding.reported",
            {
                "mission_id": "auth-main",
                "worker_id": "auth-api",
                "finding_id": "auth-api:finding:1",
                "summary": "JWKS cache is expired.",
                "evidence": [{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
                "confidence": 0.91,
                "claim_key": "auth.failure.cause",
                "claim_value": "expired-jwks-cache",
            },
            actor="auth-api-agent",
        )

        result = extract(ev)

        finding = next(e for e in result.entities if e.entity_type == "finding")
        assert finding.name == "auth-api:finding:1"
        assert finding.summary == "JWKS cache is expired."
        assert finding.properties["status"] == "pending"
        assert finding.properties["evidence_count"] == 1
        assert finding.properties["confidence"] == 0.91
        assert result.edges[0].relation_type == "worker_reported_finding"

    def test_finding_reviewed_preserves_review_status_and_rationale(self) -> None:
        ev = _make_event(
            "coordination.finding.reviewed",
            {
                "mission_id": "auth-main",
                "worker_id": "auth-api",
                "finding_id": "auth-api:finding:1",
                "status": "accepted",
                "rationale": "Command-backed evidence.",
            },
            actor="lead",
        )

        result = extract(ev)

        review = next(e for e in result.entities if e.entity_type == "finding_review")
        assert review.name == "auth-main:auth-api:finding:1:review:1"
        assert review.summary == "accepted Command-backed evidence."
        assert review.properties == {"status": "accepted", "mission_id": "auth-main", "worker_id": "auth-api"}
        assert result.edges[0].relation_type == "coordinator_reviewed_finding"

    def test_finding_promoted_projects_parent_accepted_state_with_source_citation(self) -> None:
        ev = _make_event(
            "coordination.finding.promoted",
            {
                "mission_id": "auth-main",
                "worker_id": "auth-api",
                "finding_id": "auth-api:finding:1",
                "summary": "JWKS cache is accepted cause.",
                "source_event_seq": 7,
                "source_event_hash": "b" * 64,
                "status": "accepted",
            },
            actor="lead",
        )

        result = extract(ev)

        promotion = next(e for e in result.entities if e.entity_type == "promotion")
        assert promotion.name == "auth-main:auth-api:finding:1:promotion:1"
        assert promotion.properties["status"] == "accepted"
        assert promotion.properties["source_event_seq"] == 7
        assert result.edges[0].relation_type == "finding_promoted_to_parent"

    def test_conflict_detected_links_conflicting_findings(self) -> None:
        ev = _make_event(
            "coordination.conflict.detected",
            {
                "mission_id": "auth-main",
                "conflict_id": "conflict-1",
                "claim_key": "auth.failure.cause",
                "conflict_type": "exact_claim",
                "source_reference": None,
                "finding_ids": ["auth-api:finding:1", "auth-ui:finding:1"],
                "summary": "Workers disagree on auth failure cause.",
            },
            actor="zaxy",
        )

        result = extract(ev)

        conflict = next(e for e in result.entities if e.entity_type == "conflict")
        assert conflict.name == "conflict-1"
        assert conflict.properties["claim_key"] == "auth.failure.cause"
        assert conflict.properties["conflict_type"] == "exact_claim"
        assert [edge.relation_type for edge in result.edges] == [
            "mission_has_conflict",
            "finding_conflicts_with",
            "finding_conflicts_with",
        ]

    def test_source_state_conflict_detected_preserves_source_reference(self) -> None:
        ev = _make_event(
            "coordination.conflict.detected",
            {
                "mission_id": "auth-main",
                "conflict_id": "conflict-source-1",
                "claim_key": "source:src/auth/config.py",
                "conflict_type": "source_state",
                "source_reference": "src/auth/config.py",
                "reason": "conflicting_source_snapshots",
                "finding_ids": ["auth-api:finding:1", "auth-ui:finding:1"],
                "summary": "Workers cite incompatible source snapshots.",
            },
            actor="zaxy",
        )

        result = extract(ev)

        conflict = next(e for e in result.entities if e.entity_type == "conflict")
        assert conflict.properties["conflict_type"] == "source_state"
        assert conflict.properties["source_reference"] == "src/auth/config.py"
        assert conflict.properties["reason"] == "conflicting_source_snapshots"
        assert conflict.properties["finding_count"] == 2

    def test_decision_recorded_links_mission_to_decision(self) -> None:
        ev = _make_event(
            "coordination.decision.recorded",
            {
                "mission_id": "auth-main",
                "decision_id": "decision-1",
                "decision": "Promote JWKS cache finding.",
                "rationale": "Only command-backed claim.",
            },
            actor="lead",
        )

        result = extract(ev)

        decision = next(e for e in result.entities if e.entity_type == "decision")
        assert decision.name == "decision-1"
        assert "Promote JWKS cache finding." in decision.summary
        assert result.edges[0].relation_type == "mission_has_decision"

    def test_handoff_created_links_mission_to_handoff(self) -> None:
        ev = _make_event(
            "coordination.handoff.created",
            {
                "mission_id": "auth-main",
                "handoff_id": "handoff-1",
                "summary": "Auth mission complete.",
                "next_steps": ["Release branch"],
            },
            actor="lead",
        )

        result = extract(ev)

        handoff = next(e for e in result.entities if e.entity_type == "handoff")
        assert handoff.name == "handoff-1"
        assert "Auth mission complete." in handoff.summary
        assert "Release branch" in handoff.summary
        assert result.edges[0].relation_type == "mission_has_handoff"

    def test_proof_packet_created_projects_authority_and_diagnostics(self) -> None:
        ev = _make_event(
            "coordination.proof_packet.created",
            {
                "schema_version": "coordination_proof_packet_v1",
                "mission_id": "release-rc1",
                "artifact_id": "sha256:proof",
                "query": "Compose accepted release findings.",
                "decision_scope": "handoff",
                "authority_scope": "parent_accepted_state",
                "accepted_finding_ids": ["auth-api:finding:1"],
                "diagnostic_pending_ids": ["auth-ui:finding:1"],
                "conflict_ids": ["conflict:abc123"],
                "excluded_row_reasons": [{"source_group": "auth-ui:finding:1", "exclude_reason": "pending"}],
                "non_authoritative_rows": [
                    {
                        "source_group": "auth-ui:finding:1",
                        "fact_id": "auth-ui:finding:1",
                        "status": "pending",
                        "include_reason": "diagnostic_pending",
                    }
                ],
                "handoff_event_ref": {
                    "handoff_id": "release-rc1:handoff:9",
                    "seq": 9,
                    "hash": "b" * 64,
                },
            },
            actor="coordinator",
        )

        result = extract(ev)

        proof = next(e for e in result.entities if e.entity_type == "coordination_proof_packet")
        assert proof.name == "sha256:proof"
        assert proof.summary == "Compose accepted release findings. handoff"
        assert proof.properties == {
            "mission_id": "release-rc1",
            "schema_version": "coordination_proof_packet_v1",
            "artifact_id": "sha256:proof",
            "decision_scope": "handoff",
            "authority_scope": "parent_accepted_state",
            "accepted_finding_count": 1,
            "diagnostic_pending_count": 1,
            "conflict_count": 1,
            "non_authoritative_row_count": 1,
            "excluded_row_reason_count": 1,
            "handoff_id": "release-rc1:handoff:9",
        }
        artifact = next(e for e in result.entities if e.entity_type == "synthesis_artifact")
        assert artifact.name == "sha256:proof"
        diagnostic_row = next(e for e in result.entities if e.entity_type == "coordination_non_authoritative_row")
        assert diagnostic_row.name == "sha256:proof:row:pending:auth-ui:finding:1"
        assert diagnostic_row.properties["status"] == "pending"
        assert [edge.relation_type for edge in result.edges] == [
            "mission_has_proof_packet",
            "proof_links_synthesis_artifact",
            "proof_uses_accepted_finding",
            "proof_excludes_non_authoritative_row",
            "proof_diagnoses_conflict",
            "proof_binds_handoff",
        ]


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


class TestDocumentIndexed:
    """Tests for document.indexed extractor."""

    def test_extracts_document_chunk_with_source_properties(self) -> None:
        ev = _make_event(
            "document.indexed",
            {
                "path": "docs/guide.md",
                "start_line": 4,
                "end_line": 8,
                "content": "Alpha context\nBeta context",
                "sha256": "abc123",
            },
            actor="indexer",
        )

        result = extract(ev)

        doc = next(entity for entity in result.entities if entity.entity_type == "document")
        assert doc.name == "docs/guide.md:4-8"
        assert doc.entity_type == "document"
        assert doc.summary == "Alpha context\nBeta context"
        assert doc.properties == {
            "source_path": "docs/guide.md",
            "source_start_line": 4,
            "source_end_line": 8,
            "source_sha256": "abc123",
        }
        neutral = next(entity for entity in result.entities if entity.entity_type == "neutral_substrate")
        assert neutral.name == "neutral:document:docs/guide.md:4-8"
        assert neutral.properties == {
            "actor": "indexer",
            "artifact": "docs/guide.md:4-8",
            "action": "document_indexed",
            "time": "2024-01-01T00:00:00Z",
            "source": "docs/guide.md",
            "quote": "Alpha context Beta context",
            "uncertainty": "unspecified",
            "permission_scope": "project-local",
            "candidate_claim": "Alpha context Beta context",
            "source_event_ref": f"eventloom://default/events/1#{'a' * 64}",
        }
        assert ExtractedEdge(
            source=neutral.name,
            target=doc.name,
            relation_type="neutral_substrate_cites_source",
            valid_from="2024-01-01T00:00:00Z",
        ) in result.edges
        assert not any(entity.entity_type == "neutral_projection_audit" for entity in result.entities)

    def test_flags_ingestion_time_purpose_labels_without_rewriting_document(self) -> None:
        ev = _make_event(
            "document.indexed",
            {
                "path": "customers/acme-email.txt",
                "start_line": 1,
                "end_line": 2,
                "content": "ACME asks whether the export clause applies to dashboard data.",
                "labels": ["legal_obligation"],
                "permission_scope": "project-local",
            },
            actor="support-agent",
        )

        result = extract(ev)

        doc = next(entity for entity in result.entities if entity.entity_type == "document")
        neutral = next(entity for entity in result.entities if entity.entity_type == "neutral_substrate")
        audit = next(entity for entity in result.entities if entity.entity_type == "neutral_projection_audit")
        assert doc.properties["source_path"] == "customers/acme-email.txt"
        assert "legal_obligation" not in doc.properties.values()
        assert neutral.properties["permission_scope"] == "project-local"
        assert audit.properties == {
            "safe": False,
            "forbidden_labels": ("legal_obligation",),
            "source_event_ref": f"eventloom://default/events/1#{'a' * 64}",
            "action": "flag_ingestion_purpose_label",
        }

    def test_extracts_document_retrieval_salience(self) -> None:
        ev = _make_event(
            "document.indexed",
            {
                "path": "memory/turn.md",
                "content": "I take classes at Serenity Yoga.",
                "retrieval_salience": 3.5,
            },
            actor="indexer",
        )

        result = extract(ev)

        assert result.entities[0].properties["retrieval_salience"] == 3.5

    def test_extracts_longmemeval_salient_turn_as_retrieval_salience(self) -> None:
        ev = _make_event(
            "document.indexed",
            {
                "path": "longmemeval/example/salient-turn-0005.md",
                "content": "I take classes at Serenity Yoga.",
                "longmemeval_salient_memory_turn": True,
            },
            actor="longmemeval",
        )

        result = extract(ev)

        assert result.entities[0].properties["retrieval_salience"] == 4.0

    def test_extracts_longmemeval_salient_memory_entity(self) -> None:
        ev = _make_event(
            "document.indexed",
            {
                "path": "longmemeval/user/session-1/salient-turn-0005.md",
                "start_line": 1,
                "end_line": 3,
                "content": "I started yoga classes at Serenity Yoga on March 3.",
                "longmemeval_session_id": "session-1",
                "longmemeval_session_date": "2023/03/03 (Fri) 18:00",
                "longmemeval_salient_memory_turn": True,
                "turn_index": 5,
                "role": "user",
            },
            actor="longmemeval",
        )

        result = extract(ev)

        salient = next(entity for entity in result.entities if entity.entity_type == "longmemeval_memory")
        assert salient.name == "longmemeval:memory:session-1:5"
        assert salient.summary == "I started yoga classes at Serenity Yoga on March 3."
        assert salient.properties == {
            "longmemeval_session_id": "session-1",
            "longmemeval_session_date": "2023/03/03 (Fri) 18:00",
            "turn_index": 5,
            "role": "user",
            "source_path": "longmemeval/user/session-1/salient-turn-0005.md",
            "source_start_line": 1,
            "source_end_line": 3,
        }
        assert ExtractedEdge(
            source="longmemeval:session:session-1",
            target="longmemeval:memory:session-1:5",
            relation_type="has_salient_memory",
            valid_from="2024-01-01T00:00:00Z",
        ) in result.edges
        assert ExtractedEdge(
            source="longmemeval:memory:session-1:5",
            target="longmemeval/user/session-1/salient-turn-0005.md:1-3",
            relation_type="derived_from_document",
            valid_from="2024-01-01T00:00:00Z",
        ) in result.edges

    def test_extracts_longmemeval_session_document_relationship(self) -> None:
        ev = _make_event(
            "document.indexed",
            {
                "path": "longmemeval/user/session-1/chunk-0001.md",
                "start_line": 1,
                "end_line": 10,
                "content": "longmemeval_session_id=session-1\nI take classes at Serenity Yoga.",
                "longmemeval_session_id": "session-1",
                "longmemeval_chunk_index": 1,
                "longmemeval_chunk_count": 3,
            },
            actor="longmemeval",
        )

        result = extract(ev)

        session = next(entity for entity in result.entities if entity.entity_type == "longmemeval_session")
        assert session.name == "longmemeval:session:session-1"
        assert session.properties == {"longmemeval_session_id": "session-1"}
        document = next(entity for entity in result.entities if entity.entity_type == "document")
        assert document.properties["longmemeval_session_id"] == "session-1"
        assert document.properties["longmemeval_chunk_index"] == 1
        assert document.properties["longmemeval_chunk_count"] == 3
        assert ExtractedEdge(
            source="longmemeval:session:session-1",
            target="longmemeval/user/session-1/chunk-0001.md:1-10",
            relation_type="has_document_chunk",
            valid_from="2024-01-01T00:00:00Z",
        ) in result.edges


class TestCodeFileIndexed:
    """Tests for code.file.indexed extractor."""

    def test_extracts_code_file_with_metadata_and_actor_edge(self) -> None:
        ev = _make_event(
            "code.file.indexed",
            {
                "path": "src/zaxy/core.py",
                "language": "python",
                "sha256": "abc123",
                "bytes": 2048,
                "lines": 80,
            },
            actor="zaxy-codebase-indexer",
        )

        result = extract(ev)

        code_file = next(e for e in result.entities if e.entity_type == "code_file")
        assert code_file.name == "src/zaxy/core.py"
        assert code_file.summary == "python source file with 80 lines"
        assert code_file.properties == {
            "source_path": "src/zaxy/core.py",
            "language": "python",
            "source_sha256": "abc123",
            "bytes": 2048,
            "lines": 80,
        }
        edge = result.edges[0]
        assert edge.relation_type == "indexed_code_file"
        assert edge.source == "zaxy-codebase-indexer"
        assert edge.target == "src/zaxy/core.py"


class TestSourceRefreshExtractors:
    """Tests for context refresh source/projection lifecycle extraction."""

    def test_extracts_source_changed_with_freshness_metadata(self) -> None:
        ev = _make_event(
            "source.changed",
            {
                "path": "docs/guide.md",
                "source_kind": "documents",
                "sha256": "a" * 64,
                "previous_sha256": "b" * 64,
                "bytes": 42,
            },
            actor="zaxy-context-refresh",
        )

        result = extract(ev)

        assert result.entities[0].name == "docs/guide.md"
        assert result.entities[0].entity_type == "source"
        assert result.entities[0].properties == {
            "source_path": "docs/guide.md",
            "source_kind": "documents",
            "source_sha256": "a" * 64,
            "previous_sha256": "b" * 64,
            "bytes": 42,
            "refresh_status": "changed",
        }

    def test_extracts_projection_retired_for_deleted_source(self) -> None:
        ev = _make_event(
            "projection.retired",
            {
                "path": "src/old.py",
                "source_kind": "codebase",
                "reason": "source_deleted",
            },
            actor="zaxy-context-refresh",
        )

        result = extract(ev)

        assert [entity.entity_type for entity in result.entities] == ["source", "projection"]
        assert result.edges[0].source == "src/old.py"
        assert result.edges[0].target == "projection:codebase:src/old.py"
        assert result.edges[0].relation_type == "projection_retired"


class TestMemoryPersistenceExtractors:
    """Tests for memory persistence and reminder lifecycle extraction."""

    def test_extracts_memory_reminder_suggestion(self) -> None:
        ev = _make_event(
            "memory.reminder.suggested",
            {
                "trigger": "checkpoint",
                "recommended_tool": "memory_checkout",
                "query": "what is left",
                "reasons": ["stale_memory_activity"],
            },
            actor="zaxy-memory",
        )

        result = extract(ev)

        assert result.entities[0].entity_type == "memory_reminder"
        assert result.entities[0].properties["recommended_tool"] == "memory_checkout"
        assert result.entities[0].properties["trigger"] == "checkpoint"

    def test_extracts_memory_checkout_activity(self) -> None:
        ev = _make_event(
            "memory.checkout.completed",
            {"activity": "checkout", "source": "mcp", "query": "roadmap"},
            actor="zaxy-memory",
        )

        result = extract(ev)

        assert result.entities[0].entity_type == "memory_activity"
        assert result.entities[0].properties["activity"] == "checkout"


class TestCodeSymbolIndexed:
    """Tests for code.symbol.indexed extractor."""

    def test_extracts_symbol_with_file_definition_edge(self) -> None:
        ev = _make_event(
            "code.symbol.indexed",
            {
                "path": "src/zaxy/core.py",
                "language": "python",
                "name": "MemoryFabric",
                "qualified_name": "MemoryFabric",
                "kind": "class",
                "start_line": 45,
                "end_line": 180,
            },
            actor="zaxy-codebase-indexer",
        )

        result = extract(ev)

        symbol = next(e for e in result.entities if e.entity_type == "code_symbol")
        assert symbol.name == "src/zaxy/core.py::MemoryFabric"
        assert symbol.summary == "python class MemoryFabric defined in src/zaxy/core.py:45-180"
        assert symbol.properties == {
            "source_path": "src/zaxy/core.py",
            "language": "python",
            "symbol_name": "MemoryFabric",
            "qualified_name": "MemoryFabric",
            "symbol_kind": "class",
            "source_start_line": 45,
            "source_end_line": 180,
        }
        assert any(e.name == "src/zaxy/core.py" and e.entity_type == "code_file" for e in result.entities)
        edge = result.edges[0]
        assert edge.source == "src/zaxy/core.py"
        assert edge.target == "src/zaxy/core.py::MemoryFabric"
        assert edge.relation_type == "defines_symbol"


class TestCodeImportIndexed:
    """Tests for code.import.indexed extractor."""

    def test_extracts_import_with_file_import_edge(self) -> None:
        ev = _make_event(
            "code.import.indexed",
            {
                "path": "src/zaxy/core.py",
                "language": "python",
                "module": "pathlib",
                "name": "Path",
                "kind": "from_import",
                "start_line": 23,
            },
            actor="zaxy-codebase-indexer",
        )

        result = extract(ev)

        imported = next(e for e in result.entities if e.entity_type == "code_import")
        assert imported.name == "import:pathlib:Path"
        assert imported.summary == "python from_import Path from pathlib in src/zaxy/core.py:23"
        assert imported.properties == {
            "source_path": "src/zaxy/core.py",
            "language": "python",
            "module": "pathlib",
            "import_name": "Path",
            "import_kind": "from_import",
            "source_start_line": 23,
        }
        assert any(e.name == "src/zaxy/core.py" and e.entity_type == "code_file" for e in result.entities)
        edge = result.edges[0]
        assert edge.source == "src/zaxy/core.py"
        assert edge.target == "import:pathlib:Path"
        assert edge.relation_type == "imports"


class TestCodeDependencyIndexed:
    """Tests for code.dependency.indexed extractor."""

    def test_extracts_dependency_between_code_files(self) -> None:
        ev = _make_event(
            "code.dependency.indexed",
            {
                "source_path": "src/zaxy/mcp_server.py",
                "target_path": "src/zaxy/core.py",
                "language": "python",
                "module": "zaxy.core",
                "import_name": "MemoryFabric",
                "start_line": 31,
                "resolution": "module_file",
            },
            actor="zaxy-codebase-indexer",
        )

        result = extract(ev)

        files = [e for e in result.entities if e.entity_type == "code_file"]
        assert [file.name for file in files] == ["src/zaxy/mcp_server.py", "src/zaxy/core.py"]
        assert files[0].properties == {
            "source_path": "src/zaxy/mcp_server.py",
            "language": "python",
        }
        edge = result.edges[0]
        assert edge.source == "src/zaxy/mcp_server.py"
        assert edge.target == "src/zaxy/core.py"
        assert edge.relation_type == "depends_on_file"


class TestCodeCallIndexed:
    """Tests for code.call.indexed extractor."""

    def test_extracts_call_with_resolved_symbol_edge(self) -> None:
        ev = _make_event(
            "code.call.indexed",
            {
                "path": "src/zaxy/workflow.py",
                "language": "python",
                "caller": "run",
                "callee": "MemoryFabric",
                "callee_qualified_name": "MemoryFabric",
                "target_path": "src/zaxy/core.py",
                "target_qualified_name": "MemoryFabric",
                "start_line": 8,
                "resolution": "imported_symbol",
            },
            actor="zaxy-codebase-indexer",
        )

        result = extract(ev)

        symbols = [e for e in result.entities if e.entity_type == "code_symbol"]
        assert [symbol.name for symbol in symbols] == [
            "src/zaxy/workflow.py::run",
            "src/zaxy/core.py::MemoryFabric",
        ]
        call = next(e for e in result.entities if e.entity_type == "code_call")
        assert call.name == "src/zaxy/workflow.py::run->MemoryFabric:8"
        assert call.properties == {
            "source_path": "src/zaxy/workflow.py",
            "language": "python",
            "caller": "run",
            "callee": "MemoryFabric",
            "callee_qualified_name": "MemoryFabric",
            "target_path": "src/zaxy/core.py",
            "target_qualified_name": "MemoryFabric",
            "source_start_line": 8,
            "resolution": "imported_symbol",
        }
        edge = result.edges[0]
        assert edge.source == "src/zaxy/workflow.py::run"
        assert edge.target == "src/zaxy/core.py::MemoryFabric"
        assert edge.relation_type == "calls_symbol"


class TestCodeCoverageIndexed:
    """Tests for code.coverage.indexed extractor."""

    def test_extracts_test_coverage_edge(self) -> None:
        ev = _make_event(
            "code.coverage.indexed",
            {
                "test_path": "tests/test_core.py",
                "test_name": "test_memory_fabric_starts",
                "test_qualified_name": "TestCore.test_memory_fabric_starts",
                "target_path": "src/zaxy/core.py",
                "target_name": "MemoryFabric",
                "target_qualified_name": "MemoryFabric",
                "language": "python",
                "start_line": 14,
                "resolution": "imported_symbol",
            },
            actor="zaxy-codebase-indexer",
        )

        result = extract(ev)

        symbols = [e for e in result.entities if e.entity_type == "code_symbol"]
        assert [symbol.name for symbol in symbols] == [
            "tests/test_core.py::TestCore.test_memory_fabric_starts",
            "src/zaxy/core.py::MemoryFabric",
        ]
        coverage = next(e for e in result.entities if e.entity_type == "code_coverage")
        assert coverage.name == "tests/test_core.py::TestCore.test_memory_fabric_starts=>src/zaxy/core.py::MemoryFabric:14"
        assert coverage.properties == {
            "test_path": "tests/test_core.py",
            "test_name": "test_memory_fabric_starts",
            "test_qualified_name": "TestCore.test_memory_fabric_starts",
            "target_path": "src/zaxy/core.py",
            "target_name": "MemoryFabric",
            "target_qualified_name": "MemoryFabric",
            "language": "python",
            "source_start_line": 14,
            "resolution": "imported_symbol",
        }
        edge = result.edges[0]
        assert edge.source == "tests/test_core.py::TestCore.test_memory_fabric_starts"
        assert edge.target == "src/zaxy/core.py::MemoryFabric"
        assert edge.relation_type == "tests_symbol"


class TestSessionGenesis:
    """Tests for session.genesis extractor."""

    def test_extracts_workspace_profile_and_session_edge(self) -> None:
        ev = _make_event(
            "session.genesis",
            {
                "root": "/repo",
                "workspace_type": "codebase",
                "confidence": 0.91,
                "signals": ["pyproject.toml", "src/"],
                "instructions_profile": "codebase",
                "session_id": "zaxy-default",
            },
            actor="zaxy",
        )

        result = extract(ev)

        workspace = next(e for e in result.entities if e.entity_type == "workspace")
        assert workspace.name == "/repo"
        assert workspace.summary == "codebase workspace profile codebase"
        assert workspace.properties == {
            "root": "/repo",
            "workspace_type": "codebase",
            "confidence": 0.91,
            "signals": ["pyproject.toml", "src/"],
            "instructions_profile": "codebase",
            "session_id": "zaxy-default",
        }
        session = next(e for e in result.entities if e.entity_type == "session")
        assert session.name == "zaxy-default"
        edge = result.edges[0]
        assert edge.source == "zaxy-default"
        assert edge.target == "/repo"
        assert edge.relation_type == "initialized_workspace"


class TestSessionProfileCorrected:
    """Tests for session.profile.corrected extractor."""

    def test_extracts_profile_correction_decision(self) -> None:
        ev = _make_event(
            "session.profile.corrected",
            {
                "session_id": "demo",
                "root": "/repo",
                "from": "generic_workspace",
                "to": "codebase",
                "reason": "pyproject.toml detected",
            },
            actor="user",
        )

        result = extract(ev)

        correction = next(e for e in result.entities if e.entity_type == "workspace_profile_correction")
        assert correction.name == "demo:generic_workspace->codebase"
        assert correction.summary == "pyproject.toml detected"
        edge = result.edges[0]
        assert edge.source == "demo"
        assert edge.target == correction.name
        assert edge.relation_type == "corrected_workspace_profile"


class TestWorkspaceInstructionsDiscovered:
    """Tests for workspace.instructions.discovered extractor."""

    def test_extracts_instruction_snapshot_and_session_edge(self) -> None:
        ev = _make_event(
            "workspace.instructions.discovered",
            {
                "session_id": "demo",
                "root": "/repo",
                "summary": "Project Rules: Use tests first.",
                "signature": "abc123",
                "files": [{"path": "AGENTS.md", "kind": "agents"}],
            },
            actor="zaxy",
        )

        result = extract(ev)

        instruction = next(e for e in result.entities if e.entity_type == "workspace_instructions")
        assert instruction.name == "/repo:instructions:abc123"
        assert instruction.summary == "Project Rules: Use tests first."
        assert instruction.properties == {
            "session_id": "demo",
            "root": "/repo",
            "signature": "abc123",
            "file_count": 1,
            "file_paths": ["AGENTS.md"],
            "file_kinds": ["agents"],
        }
        edge = result.edges[0]
        assert edge.source == "demo"
        assert edge.target == "/repo:instructions:abc123"
        assert edge.relation_type == "uses_workspace_instructions"


class TestLifecycleEvents:
    """Tests for lifecycle event extractors."""

    def test_extracts_tool_call_completed(self) -> None:
        ev = _make_event(
            "tool.call.completed",
            {
                "tool_name": "shell",
                "status": "succeeded",
                "session_id": "demo",
                "result_summary": "443 passed",
            },
            actor="zaxy",
        )

        result = extract(ev)

        tool = next(e for e in result.entities if e.entity_type == "tool_call")
        assert tool.name == "demo:shell:1"
        assert tool.summary == "succeeded 443 passed"
        assert tool.properties == {
            "tool_name": "shell",
            "status": "succeeded",
            "session_id": "demo",
            "call_id": None,
        }

    def test_tool_call_links_explicit_task_id(self) -> None:
        """Tool observations should link to an explicit task id when present."""
        ev = _make_event(
            "tool.call.completed",
            {
                "tool_name": "shell",
                "status": "succeeded",
                "session_id": "demo",
                "task_id": "task-7",
            },
            actor="zaxy",
        )

        result = extract(ev)

        assert any(e.name == "task-7" and e.entity_type == "task" for e in result.entities)
        assert any(
            edge.source == "task-7"
            and edge.target == "demo:shell:1"
            and edge.relation_type == "observed_tool_call"
            for edge in result.edges
        )

    def test_extracts_command_completed(self) -> None:
        ev = _make_event(
            "command.completed",
            {
                "command": "pytest",
                "exit_code": 0,
                "outcome": "passed",
                "session_id": "demo",
            },
            actor="zaxy",
        )

        result = extract(ev)

        command = next(e for e in result.entities if e.entity_type == "command_run")
        assert command.name == "demo:pytest:1"
        assert command.summary == "passed pytest"
        assert command.properties["exit_code"] == 0

    def test_command_links_explicit_task_id(self) -> None:
        """Command observations should link to an explicit task id when present."""
        ev = _make_event(
            "command.completed",
            {
                "command": "pytest",
                "exit_code": 0,
                "outcome": "passed",
                "session_id": "demo",
                "taskId": "task-7",
            },
            actor="zaxy",
        )

        result = extract(ev)

        assert any(e.name == "task-7" and e.entity_type == "task" for e in result.entities)
        assert any(
            edge.source == "task-7"
            and edge.target == "demo:pytest:1"
            and edge.relation_type == "observed_command"
            for edge in result.edges
        )

    def test_extracts_file_edit_applied(self) -> None:
        ev = _make_event(
            "file.edit.applied",
            {
                "path": "src/zaxy/core.py",
                "operation": "modified",
                "session_id": "demo",
                "summary": "Added lifecycle hook.",
            },
            actor="zaxy",
        )

        result = extract(ev)

        edit = next(e for e in result.entities if e.entity_type == "file_edit")
        assert edit.name == "demo:src/zaxy/core.py:1"
        assert edit.summary == "modified Added lifecycle hook."
        assert edit.properties["path"] == "src/zaxy/core.py"

    def test_file_edit_links_explicit_task_id(self) -> None:
        """File-edit observations should link to an explicit task id when present."""
        ev = _make_event(
            "file.edit.applied",
            {
                "path": "src/zaxy/core.py",
                "operation": "modified",
                "session_id": "demo",
                "task_id": "task-7",
            },
            actor="zaxy",
        )

        result = extract(ev)

        assert any(e.name == "task-7" and e.entity_type == "task" for e in result.entities)
        assert any(
            edge.source == "task-7"
            and edge.target == "demo:src/zaxy/core.py:1"
            and edge.relation_type == "observed_file_edit"
            for edge in result.edges
        )

    def test_extracts_compaction_completed(self) -> None:
        ev = _make_event(
            "compaction.completed",
            {
                "session_id": "demo",
                "mode": "rewrite",
                "status": "succeeded",
                "log_path": ".eventloom/demo.jsonl",
                "event_count": 7,
            },
            actor="zaxy",
        )

        result = extract(ev)

        compaction = next(e for e in result.entities if e.entity_type == "compaction_run")
        assert compaction.name == "demo:compaction:1"
        assert compaction.summary == "succeeded rewrite .eventloom/demo.jsonl"
        assert compaction.properties["event_count"] == 7
        assert result.edges[0].relation_type == "completed_compaction"

    def test_extracts_subagent_completed(self) -> None:
        ev = _make_event(
            "subagent.completed",
            {
                "parent_session_id": "main",
                "subagent_session_id": "worker-1",
                "status": "succeeded",
                "summary": "Worker finished retrieval.",
            },
            actor="zaxy",
        )

        result = extract(ev)

        subagent = next(e for e in result.entities if e.entity_type == "subagent_run")
        assert subagent.name == "main:worker-1:1"
        assert subagent.summary == "succeeded Worker finished retrieval."
        assert subagent.properties["parent_session_id"] == "main"
        assert result.edges[0].relation_type == "completed_subagent"

    def test_extracts_session_ended(self) -> None:
        ev = _make_event(
            "session.ended",
            {
                "session_id": "demo",
                "reason": "teardown",
                "status": "succeeded",
            },
            actor="zaxy",
        )

        result = extract(ev)

        ended = next(e for e in result.entities if e.entity_type == "session_end")
        assert ended.name == "demo:session-ended:1"
        assert ended.summary == "succeeded teardown"
        assert ended.properties["session_id"] == "demo"
        assert result.edges[0].relation_type == "ended_session"


class TestTranscriptTurn:
    """Tests for transcript.turn extractor."""

    def test_extracts_transcript_turn_with_role_properties(self) -> None:
        ev = _make_event(
            "transcript.turn",
            {
                "source": "codex",
                "turn_index": 7,
                "role": "assistant",
                "content": "We decided to ship the retrieval sprint.",
                "redacted_paths": [],
            },
            actor="assistant",
        )

        result = extract(ev)

        turn = next(entity for entity in result.entities if entity.entity_type == "transcript_turn")
        assert turn.name == "codex:turn-7"
        assert turn.entity_type == "transcript_turn"
        assert turn.summary == "assistant: We decided to ship the retrieval sprint."
        assert turn.properties == {
            "transcript_source": "codex",
            "transcript_role": "assistant",
            "transcript_turn_index": 7,
            "redacted_paths": [],
        }
        neutral = next(entity for entity in result.entities if entity.entity_type == "neutral_substrate")
        assert neutral.name == "neutral:transcript:codex:turn-7"
        assert neutral.properties == {
            "actor": "assistant",
            "artifact": "codex:turn-7",
            "action": "transcript_turn",
            "time": "2024-01-01T00:00:00Z",
            "source": "codex",
            "quote": "We decided to ship the retrieval sprint.",
            "uncertainty": "sanitized",
            "permission_scope": "session",
            "candidate_claim": "We decided to ship the retrieval sprint",
            "source_event_ref": f"eventloom://default/events/1#{'a' * 64}",
        }
        assert ExtractedEdge(
            source=neutral.name,
            target=turn.name,
            relation_type="neutral_substrate_cites_source",
            valid_from="2024-01-01T00:00:00Z",
        ) in result.edges


class TestLlmPacketProjection:
    """Tests for llm.packet.projected extractor."""

    def test_extracts_packet_projection_with_source_properties(self) -> None:
        ev = _make_event(
            "llm.packet.projected",
            {
                "session_id": "agent-1",
                "source_event_seq": 12,
                "source_event_hash": "b" * 64,
                "provider_path": "/v1/chat/completions",
                "status_code": 200,
                "model": "gpt-test",
                "usage_counts": {"prompt": 3, "completion": 2, "total": 5},
                "summary": "LLM packet /v1/chat/completions gpt-test status 200.",
            },
            actor="zaxy-packet-projector",
        )

        result = extract(ev)

        assert len(result.entities) == 2
        packet = next(e for e in result.entities if e.entity_type == "llm_packet_projection")
        assert packet.name == "agent-1:llm-packet:12"
        assert packet.summary == "LLM packet /v1/chat/completions gpt-test status 200."
        assert packet.properties == {
            "session_id": "agent-1",
            "source_event_seq": 12,
            "source_event_hash": "b" * 64,
            "provider_path": "/v1/chat/completions",
            "status_code": 200,
            "model": "gpt-test",
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        assert any(
            edge.source == "agent-1"
            and edge.target == "agent-1:llm-packet:12"
            and edge.relation_type == "projected_llm_packet"
            for edge in result.edges
        )


# ------------------------------------------------------------------
# Integration / sanity tests
# ------------------------------------------------------------------

class TestExtractionSanity:
    """Cross-cutting sanity checks."""

    def test_all_results_have_source_event_seq(self) -> None:
        """Every result should preserve originating event provenance."""
        for event_type, payload in [
            ("goal.created", {"title": "X"}),
            ("task.proposed", {"taskId": "t1"}),
            ("task.claimed", {"taskId": "t1"}),
            ("task.completed", {"taskId": "t1"}),
            (
                "inference.edge.generated",
                {
                    "source": {"name": "t1", "entity_type": "task"},
                    "target": {"name": "d1", "entity_type": "decision"},
                    "relation_type": "likely_informed",
                    "confidence": 0.8,
                    "inference_method": "operator_review_v1",
                },
            ),
            ("user.preference_changed", {"key": "k"}),
            ("document.indexed", {"path": "README.md", "content": "hello"}),
            ("transcript.turn", {"role": "user", "content": "hello"}),
            ("unknown.type", {}),
        ]:
            ev = _make_event(event_type, payload)
            ev2 = ev.model_copy(update={"seq": 99})
            result = extract(ev2)
            assert result.source_event_seq == 99
            assert result.source_event_hash == "a" * 64
            assert result.source_event_type == event_type
            assert result.source_thread == "default"

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
