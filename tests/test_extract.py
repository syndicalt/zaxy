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

        assert len(result.entities) == 1
        doc = result.entities[0]
        assert doc.name == "docs/guide.md:4-8"
        assert doc.entity_type == "document"
        assert doc.summary == "Alpha context\nBeta context"
        assert doc.properties == {
            "source_path": "docs/guide.md",
            "source_start_line": 4,
            "source_end_line": 8,
            "source_sha256": "abc123",
        }


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

        assert len(result.entities) == 1
        turn = result.entities[0]
        assert turn.name == "codex:turn-7"
        assert turn.entity_type == "transcript_turn"
        assert turn.summary == "assistant: We decided to ship the retrieval sprint."
        assert turn.properties == {
            "transcript_source": "codex",
            "transcript_role": "assistant",
            "transcript_turn_index": 7,
            "redacted_paths": [],
        }


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
