"""Tests for zaxy.core — MemoryFabric orchestrator.

Tests cover the full orchestration pipeline: event → extract → graph → query,
with all external dependencies mocked."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.compaction import build_compaction_projection, write_compaction_projection
from zaxy.core import Context, HandoffBundle, MemoryFabric
from zaxy.embedding import HashEmbeddingProvider
from zaxy.event import EventLog
from zaxy.query import ContextChunk


class BrokenEmbeddingProvider:
    """Embedding provider test double that simulates provider outages."""

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding down")


async def test_memory_fabric_queries_verbatim_eventloom_sources(tmp_path: Path) -> None:
    """Fabric should expose verbatim source recall without requiring Neo4j."""
    fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
    session = fabric.session_manager.get("agent")
    event = session.eventlog.append(
        "transcript.turn",
        actor="assistant",
        payload={
            "source": "codex",
            "turn_index": 3,
            "role": "assistant",
            "content": "The replay adapter uses identity-preserving chunks.",
        },
        thread="agent",
    )

    contexts = await fabric.query_verbatim(
        "Which adapter uses identity-preserving chunks?",
        session_id="agent",
        limit=1,
    )

    assert contexts[0].source == "verbatim"
    assert contexts[0].content == "assistant: The replay adapter uses identity-preserving chunks."
    assert contexts[0].metadata == {
        "citation": f"eventloom://agent/events/{event.seq}#{event.hash}",
        "source_kind": "transcript",
        "event_seq": event.seq,
        "event_type": "transcript.turn",
        "event_thread": "agent",
        "event_timestamp": event.timestamp,
        "transcript_source": "codex",
        "transcript_turn_index": 3,
        "transcript_role": "assistant",
    }

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def fabric() -> MemoryFabric:
    """Return a MemoryFabric with mocked dependencies."""
    with (
        patch("zaxy.core.EventLog") as mock_log_cls,
        patch("zaxy.core.GraphStore") as mock_graph_cls,
        patch("zaxy.core.QueryRouter") as mock_router_cls,
        patch("zaxy.core.build_reranker") as mock_build_reranker,
        patch("zaxy.core.MemoryTracer") as mock_tracer_cls,
        patch("zaxy.core.SessionManager") as mock_session_cls,
    ):
        log = MagicMock()
        log.append.return_value = MagicMock(seq=1, hash="a" * 64, type="x", actor="y", timestamp="2024-01-01T00:00:00Z")
        mock_log_cls.return_value = log

        session_mgr = MagicMock()
        session_mgr.get.return_value.eventlog = log
        mock_session_cls.return_value = session_mgr

        graph = AsyncMock()
        mock_graph_cls.return_value = graph

        router = AsyncMock()
        mock_router_cls.return_value = router
        mock_build_reranker.return_value = None

        tracer = AsyncMock()
        mock_tracer_cls.return_value = tracer

        f = MemoryFabric()
        f.session_manager = session_mgr
        f.eventloom = log
        f.graph = graph
        f.query_router = router
        f.tracer = tracer
        yield f


# ------------------------------------------------------------------
# Lifecycle tests
# ------------------------------------------------------------------

class TestLifecycle:
    """Tests for connect/close behavior."""

    def test_initializes_query_router_with_configured_reranker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MemoryFabric should wire configured scoring/reranking into QueryRouter."""
        monkeypatch.setenv("QUERY_SCORING_PROFILE", "precision")
        monkeypatch.setenv("RERANKER_PROVIDER", "lexical")
        from zaxy.config import get_settings

        get_settings.cache_clear()
        with (
            patch("zaxy.core.GraphStore"),
            patch("zaxy.core.QueryRouter") as mock_router_cls,
            patch("zaxy.core.build_reranker") as mock_build_reranker,
            patch("zaxy.core.MemoryTracer"),
            patch("zaxy.core.SessionManager") as mock_session_cls,
        ):
            mock_build_reranker.return_value = object()
            mock_session_cls.return_value.get.return_value.eventlog = MagicMock()

            MemoryFabric()

        assert mock_router_cls.call_args.kwargs["scoring_profile"] == "precision"
        assert mock_router_cls.call_args.kwargs["reranker"] is mock_build_reranker.return_value

    async def test_connect_initializes_graph_and_tracer(self, fabric: MemoryFabric) -> None:
        """connect() should init schema and connect tracer."""
        await fabric.connect()
        fabric.graph.connect.assert_awaited_once()
        fabric.graph.init_schema.assert_awaited_once()
        fabric.tracer.connect.assert_awaited_once()
        assert fabric._connected is True

    async def test_connect_is_idempotent(self, fabric: MemoryFabric) -> None:
        """Multiple connect() calls should not re-initialize."""
        await fabric.connect()
        await fabric.connect()
        fabric.graph.connect.assert_awaited_once()

    async def test_close_closes_all(self, fabric: MemoryFabric) -> None:
        """close() should close graph and tracer."""
        await fabric.connect()
        await fabric.close()
        fabric.graph.close.assert_awaited_once()
        fabric.tracer.close.assert_awaited_once()
        assert fabric._connected is False


# ------------------------------------------------------------------
# Append tests
# ------------------------------------------------------------------

class TestAppend:
    """Tests for the write path."""

    async def test_appends_event(self, fabric: MemoryFabric) -> None:
        """append() should write to Eventloom."""
        await fabric.append("goal.created", actor="user", payload={"title": "T"})
        fabric.session_manager.get.assert_any_call("default")
        log = fabric.session_manager.get.return_value.eventlog
        log.append.assert_called_once_with(
            "goal.created", actor="user", payload={"title": "T"}, thread="default"
        )

    async def test_appends_with_session_id(self, fabric: MemoryFabric) -> None:
        """append() should route to the correct session."""
        await fabric.append("goal.created", actor="user", payload={"title": "T"}, session_id="agent-1")
        fabric.session_manager.get.assert_any_call("agent-1")
        log = fabric.session_manager.get.return_value.eventlog
        log.append.assert_called_once_with(
            "goal.created", actor="user", payload={"title": "T"}, thread="agent-1"
        )

    async def test_extracts_and_upserts(self, fabric: MemoryFabric) -> None:
        """append() should extract entities and upsert to graph."""
        await fabric.append("goal.created", actor="user", payload={"title": "T"})
        fabric.graph.upsert_extraction.assert_awaited_once()
        assert fabric.graph.upsert_extraction.await_args.kwargs["session_id"] == "default"
        extraction = fabric.graph.upsert_extraction.await_args.args[0]
        assert extraction.entities[0].embedding is not None

    async def test_append_keeps_event_when_embedding_provider_fails(self, fabric: MemoryFabric) -> None:
        """Embedding failures should not block the durable event append."""
        fabric.embedding_provider = BrokenEmbeddingProvider()
        metrics = MagicMock()

        with patch("zaxy.core.get_metrics", return_value=metrics):
            await fabric.append("goal.created", actor="user", payload={"title": "T"})

        log = fabric.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        fabric.graph.upsert_extraction.assert_awaited_once()
        extraction = fabric.graph.upsert_extraction.await_args.args[0]
        assert extraction.entities[0].embedding is None
        metrics.record_degraded_operation.assert_any_call("append", "embedding_provider_unavailable")

    async def test_append_keeps_event_when_graph_projection_fails(self, fabric: MemoryFabric) -> None:
        """Graph projection failures should not roll back Eventloom writes."""
        fabric.graph.upsert_extraction.side_effect = RuntimeError("graph down")
        metrics = MagicMock()

        with patch("zaxy.core.get_metrics", return_value=metrics):
            await fabric.append("goal.created", actor="user", payload={"title": "T"})

        log = fabric.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        metrics.record_degraded_operation.assert_any_call("append", "graph_projection_unavailable")

    async def test_append_keeps_event_when_graph_connect_fails(self, fabric: MemoryFabric) -> None:
        """Graph connection outages should not block durable Eventloom writes."""
        fabric.graph.connect.side_effect = RuntimeError("graph down")
        metrics = MagicMock()

        with patch("zaxy.core.get_metrics", return_value=metrics):
            await fabric.append("goal.created", actor="user", payload={"title": "T"})

        log = fabric.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        metrics.record_degraded_operation.assert_any_call("append", "graph_connect_unavailable")

    async def test_traces_append(self, fabric: MemoryFabric) -> None:
        """append() should emit a Pathlight trace."""
        await fabric.append("goal.created", actor="user", payload={"title": "T"})
        fabric.tracer.trace_append.assert_awaited_once()

    async def test_auto_connects(self, fabric: MemoryFabric) -> None:
        """append() should auto-connect if not already connected."""
        await fabric.append("x", actor="y")
        fabric.graph.connect.assert_awaited_once()


class TestContextFeedback:
    """Tests for retrieval feedback event capture."""

    async def test_positive_feedback_reinforces_context_entity(self, fabric: MemoryFabric) -> None:
        """Used context should append a memory.reinforced event."""
        context = Context(
            content="Use retention metadata (decision)",
            source="keyword",
            score=0.9,
            metadata={
                "entity_name": "Use retention metadata",
                "entity_type": "decision",
                "citation": "eventloom://default/events/1#abc",
            },
        )

        count = await fabric.record_context_feedback(
            [context],
            feedback="used",
            session_id="agent-1",
            actor="assistant",
            importance=0.8,
        )

        assert count == 1
        log = fabric.session_manager.get.return_value.eventlog
        call = log.append.call_args_list[-1]
        assert call.args == ("memory.reinforced",)
        assert call.kwargs["actor"] == "assistant"
        assert call.kwargs["thread"] == "agent-1"
        assert call.kwargs["payload"]["entity_name"] == "Use retention metadata"
        assert call.kwargs["payload"]["entity_type"] == "decision"
        assert call.kwargs["payload"]["importance"] == 0.8

    async def test_positive_feedback_falls_back_to_content_identity(self, fabric: MemoryFabric) -> None:
        """Context without entity metadata should still produce a stable reinforcement event."""
        context = Context(content="Fallback note", source="eventloom", score=0.5)

        await fabric.record_context_feedback([context], feedback="helpful", session_id="agent-1")

        payload = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1].kwargs["payload"]
        assert payload["entity_name"] == "Fallback note"
        assert payload["entity_type"] == "memory"

    async def test_negative_feedback_is_audit_only(self, fabric: MemoryFabric) -> None:
        """Irrelevant context should be recorded without mutating retention metadata."""
        context = Context(
            content="Stale note (decision)",
            source="keyword",
            score=0.2,
            metadata={"entity_name": "Stale note", "entity_type": "decision"},
        )

        count = await fabric.record_context_feedback([context], feedback="irrelevant", session_id="agent-1")

        assert count == 1
        call = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1]
        assert call.args == ("memory.feedback",)
        assert call.kwargs["payload"]["feedback"] == "irrelevant"
        assert call.kwargs["payload"]["entity_name"] == "Stale note"


class TestDocumentIngestion:
    """Tests for filesystem document ingestion orchestration."""

    async def test_ingest_documents_appends_document_events(
        self,
        fabric: MemoryFabric,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        """ingest_documents() should append collected document chunks."""
        doc = tmp_path / "README.md"
        doc.write_text("Alpha\nBeta\n", encoding="utf-8")

        await fabric.ingest_documents(tmp_path, session_id="agent-1", max_lines=1)

        log = fabric.session_manager.get.return_value.eventlog
        assert log.append.call_count == 2
        assert log.append.call_args_list[0].args == ("document.indexed",)
        assert log.append.call_args_list[0].kwargs["actor"] == "zaxy-doc-ingest"
        assert log.append.call_args_list[0].kwargs["payload"]["path"] == "README.md"
        assert log.append.call_args_list[0].kwargs["payload"]["start_line"] == 1
        assert log.append.call_args_list[0].kwargs["thread"] == "agent-1"


class TestCodebaseIngestion:
    """Tests for codebase mapping ingestion orchestration."""

    async def test_ingest_codebase_appends_codebase_mapping_events(
        self,
        fabric: MemoryFabric,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        """ingest_codebase() should append collected file and symbol events."""
        source = tmp_path / "src" / "app.py"
        source.parent.mkdir()
        source.write_text("def main():\n    return 42\n", encoding="utf-8")

        count = await fabric.ingest_codebase(tmp_path, session_id="agent-1")

        assert count == 2
        log = fabric.session_manager.get.return_value.eventlog
        assert [call.args[0] for call in log.append.call_args_list] == [
            "code.file.indexed",
            "code.symbol.indexed",
        ]
        first_call = log.append.call_args_list[0]
        assert first_call.kwargs["actor"] == "zaxy-codebase-indexer"
        assert first_call.kwargs["payload"]["path"] == "src/app.py"
        assert first_call.kwargs["payload"]["language"] == "python"
        assert first_call.kwargs["thread"] == "agent-1"


class TestSessionInitialization:
    """Tests for session genesis orchestration."""

    async def test_initialize_session_appends_genesis_event(
        self,
        fabric: MemoryFabric,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        """initialize_session() should append a session.genesis event."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        (tmp_path / "src").mkdir()

        profile = await fabric.initialize_session(tmp_path, session_id="agent-1")

        assert profile.workspace_type == "codebase"
        log = fabric.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        assert log.append.call_args.args == ("session.genesis",)
        assert log.append.call_args.kwargs["actor"] == "zaxy"
        assert log.append.call_args.kwargs["payload"]["workspace_type"] == "codebase"
        assert log.append.call_args.kwargs["payload"]["session_id"] == "agent-1"
        assert log.append.call_args.kwargs["thread"] == "agent-1"

    async def test_ensure_session_initialized_skips_existing_genesis(
        self,
        fabric: MemoryFabric,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        """ensure_session_initialized() should not duplicate an existing genesis event."""
        root = str(tmp_path.resolve())
        existing = SimpleNamespace(
            type="session.genesis",
            payload={
                "root": root,
                "workspace_type": "codebase",
                "confidence": 0.91,
                "signals": ["pyproject.toml"],
                "instructions_profile": "codebase",
                "session_id": "agent-1",
            },
        )
        log = fabric.session_manager.get.return_value.eventlog
        log.read_all.return_value = [existing]

        profile = await fabric.ensure_session_initialized(tmp_path, session_id="agent-1")

        assert profile.workspace_type == "codebase"
        assert profile.confidence == 0.91
        log.append.assert_not_called()

    async def test_ensure_session_initialized_appends_instruction_discovery(
        self,
        fabric: MemoryFabric,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        """ensure_session_initialized() should write instruction summaries beside genesis."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("# Rules\n\nUse pytest.\n", encoding="utf-8")
        log = fabric.session_manager.get.return_value.eventlog
        log.read_all.return_value = []

        await fabric.ensure_session_initialized(tmp_path, session_id="agent-1")

        assert log.append.call_count == 2
        assert log.append.call_args_list[0].args == ("session.genesis",)
        instruction_call = log.append.call_args_list[1]
        assert instruction_call.args == ("workspace.instructions.discovered",)
        assert instruction_call.kwargs["actor"] == "zaxy"
        assert instruction_call.kwargs["payload"]["summary"] == "Rules: Use pytest."
        assert instruction_call.kwargs["thread"] == "agent-1"

    async def test_ensure_session_initialized_appends_instruction_update_on_drift(
        self,
        fabric: MemoryFabric,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        """ensure_session_initialized() should write an update when instruction hashes change."""
        root = str(tmp_path.resolve())
        (tmp_path / "AGENTS.md").write_text("# Rules\n\nUse pytest.\n", encoding="utf-8")
        existing_genesis = SimpleNamespace(
            type="session.genesis",
            payload={
                "root": root,
                "workspace_type": "codebase",
                "confidence": 0.91,
                "signals": ["pyproject.toml"],
                "instructions_profile": "codebase",
                "session_id": "agent-1",
            },
        )
        existing_instructions = SimpleNamespace(
            type="workspace.instructions.discovered",
            payload={
                "root": root,
                "session_id": "agent-1",
                "signature": "previous-signature",
            },
        )
        log = fabric.session_manager.get.return_value.eventlog
        log.read_all.return_value = [existing_genesis, existing_instructions]

        await fabric.ensure_session_initialized(tmp_path, session_id="agent-1")

        log.append.assert_called_once()
        assert log.append.call_args.args == ("workspace.instructions.updated",)
        payload = log.append.call_args.kwargs["payload"]
        assert payload["previous_signature"] == "previous-signature"
        assert payload["signature"] != "previous-signature"


class TestTranscriptIngestion:
    """Tests for transcript ingestion orchestration."""

    async def test_ingest_transcript_appends_sanitized_turn_events(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """ingest_transcript() should append sanitized transcript turns."""
        await fabric.ingest_transcript(
            [{"role": "user", "content": "remember sk-abcdefghijklmnop"}],
            source="codex",
            session_id="agent-1",
        )

        log = fabric.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        assert log.append.call_args.args == ("transcript.turn",)
        assert log.append.call_args.kwargs["actor"] == "user"
        assert log.append.call_args.kwargs["payload"]["content"] == "[REDACTED]"
        assert log.append.call_args.kwargs["payload"]["source"] == "codex"
        assert log.append.call_args.kwargs["thread"] == "agent-1"


# ------------------------------------------------------------------
# Query tests
# ------------------------------------------------------------------

class TestQuery:
    """Tests for the read path."""

    async def test_queries_router(self, fabric: MemoryFabric) -> None:
        """query() should delegate to QueryRouter."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Alice (user)",
                source="exact",
                score=1.0,
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                citation="eventloom://default/events/1#aaaaaaaaaaaa",
                score_explanation={"source": "exact", "weighted_score": 1.0},
            )
        ]
        results = await fabric.query("Alice")
        args = fabric.query_router.query.await_args
        assert args.args == ("Alice",)
        assert args.kwargs["temporal_point"] is None
        assert args.kwargs["limit"] == 10
        assert args.kwargs["embedding"] is not None
        assert len(results) == 1
        assert isinstance(results[0], Context)
        assert results[0].metadata == {
            "citation": "eventloom://default/events/1#aaaaaaaaaaaa",
            "score_explanation": {"source": "exact", "weighted_score": 1.0},
        }

    async def test_passes_temporal_filter(self, fabric: MemoryFabric) -> None:
        """query() should forward temporal_point to router."""
        fabric.query_router.query.return_value = []
        await fabric.query("x", temporal_point="2024-03-01T00:00:00Z", limit=5)
        args = fabric.query_router.query.await_args
        assert args.args == ("x",)
        assert args.kwargs["temporal_point"] == "2024-03-01T00:00:00Z"
        assert args.kwargs["limit"] == 5
        assert args.kwargs["session_id"] == "default"
        assert args.kwargs["embedding"] is not None

    async def test_query_with_session_id(self, fabric: MemoryFabric) -> None:
        """query() should scope graph retrieval to the requested session."""
        fabric.query_router.query.return_value = []
        await fabric.query("x", session_id="agent-1")
        assert fabric.query_router.query.await_args.kwargs["session_id"] == "agent-1"

    async def test_explicit_query_embedding_wins(self, fabric: MemoryFabric) -> None:
        """query() should not overwrite a caller-provided embedding."""
        explicit = [0.1, 0.2, 0.3]
        fabric.query_router.query.return_value = []
        await fabric.query("x", embedding=explicit)
        assert fabric.query_router.query.await_args.kwargs["embedding"] is explicit

    async def test_query_continues_when_embedding_provider_fails(self, fabric: MemoryFabric) -> None:
        """Embedding outages should degrade to non-vector retrieval."""
        fabric.embedding_provider = BrokenEmbeddingProvider()
        fabric.query_router.query.return_value = []
        metrics = MagicMock()

        with patch("zaxy.core.get_metrics", return_value=metrics):
            await fabric.query("x")

        assert fabric.query_router.query.await_args.kwargs["embedding"] is None
        metrics.record_degraded_operation.assert_any_call("query", "embedding_provider_unavailable")

    async def test_query_falls_back_to_eventlog_when_graph_unavailable(self, fabric: MemoryFabric) -> None:
        """Graph outages should still return matching durable Eventloom context."""
        fabric.graph.connect.side_effect = RuntimeError("graph down")
        metrics = MagicMock()
        event = MagicMock(
            seq=7,
            type="goal.created",
            actor="user",
            payload={"title": "Ship offline retrieval"},
            timestamp="2024-01-01T00:00:00Z",
        )
        fabric.session_manager.replay.return_value = SimpleNamespace(
            events=[event],
        )

        with patch("zaxy.core.get_metrics", return_value=metrics):
            results = await fabric.query("offline retrieval")

        assert results[0].source == "eventloom"
        assert "Ship offline retrieval" in results[0].content
        fabric.query_router.query.assert_not_awaited()
        metrics.record_degraded_operation.assert_any_call("query", "graph_unavailable")
        assert metrics.record_query.call_args.kwargs["source"] == "eventloom"

    async def test_query_falls_back_to_eventlog_when_router_fails(self, fabric: MemoryFabric) -> None:
        """Graph retrieval failures after connect should use durable Eventloom context."""
        fabric.query_router.query.side_effect = RuntimeError("graph query down")
        metrics = MagicMock()
        event = MagicMock(
            seq=8,
            type="task.proposed",
            actor="assistant",
            payload={"title": "Repair degraded retrieval"},
            timestamp="2024-01-02T00:00:00Z",
        )
        fabric.session_manager.replay.return_value = SimpleNamespace(
            events=[event],
        )

        with patch("zaxy.core.get_metrics", return_value=metrics):
            results = await fabric.query("degraded retrieval")

        assert results[0].source == "eventloom"
        assert results[0].metadata is not None
        assert results[0].metadata["reason"] == "graph retrieval unavailable"
        metrics.record_degraded_operation.assert_any_call("query", "graph_retrieval_unavailable")

    async def test_traces_query(self, fabric: MemoryFabric) -> None:
        """query() should emit a Pathlight trace with result count."""
        fabric.query_router.query.return_value = []
        await fabric.query("x")
        fabric.tracer.trace_query.assert_awaited_once()
        args = fabric.tracer.trace_query.await_args
        assert args.args[0] == "x"
        assert args.args[1] == 0

    async def test_auto_connects(self, fabric: MemoryFabric) -> None:
        """query() should auto-connect if not already connected."""
        fabric.query_router.query.return_value = []
        await fabric.query("x")
        fabric.graph.connect.assert_awaited_once()


class TestContextAssembly:
    """Tests for prompt-ready context assembly."""

    async def test_assemble_context_combines_replay_and_retrieval(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """assemble_context() should include recent replay and retrieved context."""
        event = MagicMock(
            seq=3,
            type="transcript.turn",
            actor="assistant",
            payload={"role": "assistant", "content": "We chose MMR."},
            hash="b" * 64,
        )
        fabric.session_manager.replay.return_value = MagicMock(
            events=[event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="MMR diversity (decision)",
                source="keyword",
                score=0.8,
                valid_from=None,
                valid_to=None,
                citation="eventloom://agent-1/events/2#aaaaaaaaaaaa",
            )
        ]

        assembly = await fabric.assemble_context(
            "What did we decide about retrieval?",
            session_id="agent-1",
            replay_from_seq=3,
            limit=1,
        )

        assert assembly.session_id == "agent-1"
        assert assembly.replay_event_count == 1
        assert assembly.contexts[0].content == "MMR diversity (decision)"
        assert "[3] transcript.turn by assistant" in assembly.prompt
        assert "MMR diversity (decision)" in assembly.prompt

    async def test_assemble_context_includes_verbatim_source_lane(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """assemble_context() should reserve room for exact Eventloom source recall."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Graph summary of identity decision",
                source="keyword",
                score=0.8,
                valid_from=None,
                valid_to=None,
                citation="eventloom://agent-1/events/1#aaaaaaaaaaaa",
            ),
            ContextChunk(
                content="Lower-priority graph context",
                source="traversal",
                score=0.5,
                valid_from=None,
                valid_to=None,
                citation="eventloom://agent-1/events/2#bbbbbbbbbbbb",
            ),
        ]
        with patch.object(
            fabric,
            "query_verbatim",
            return_value=[
                Context(
                    content="assistant: Exact source mentions identity-code-0042.",
                    source="verbatim",
                    score=1.2,
                    metadata={
                        "citation": "eventloom://agent-1/events/3#cccccccccccc",
                        "source_kind": "transcript",
                    },
                )
            ],
        ):
            assembly = await fabric.assemble_context(
                "identity-code-0042",
                session_id="agent-1",
                limit=2,
            )

        assert [context.source for context in assembly.contexts] == ["keyword", "verbatim"]
        assert assembly.contexts[0].metadata is not None
        assert assembly.contexts[0].metadata["assembly_lane"] == "graph"
        assert assembly.contexts[1].metadata is not None
        assert assembly.contexts[1].metadata["assembly_lane"] == "verbatim"
        assert "Exact source mentions identity-code-0042" in assembly.prompt
        assert "eventloom://agent-1/events/3#cccccccccccc" in assembly.prompt
        assert assembly.assembly_policy == {
            "verbatim_enabled": True,
            "verbatim_slots": 1,
        }
        assert assembly.context_counts == {"graph": 1, "verbatim": 1, "replay": 0}
        assert assembly.working_set["items"][0]["category"] == "source_anchor"
        assert "# Active Memory Working Set" in assembly.prompt

    async def test_assemble_context_skips_verbatim_when_policy_disabled(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Disabled source recall should not read Eventloom verbatim hits."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Graph summary",
                source="keyword",
                score=0.8,
                valid_from=None,
                valid_to=None,
            )
        ]
        fabric.context_assembly_policy = fabric.context_assembly_policy.with_verbatim_enabled(False)
        with patch.object(fabric, "query_verbatim") as mock_query_verbatim:
            assembly = await fabric.assemble_context("identity", session_id="agent-1", limit=2)

        mock_query_verbatim.assert_not_called()
        assert [context.source for context in assembly.contexts] == ["keyword"]
        assert assembly.assembly_policy == {
            "verbatim_enabled": False,
            "verbatim_slots": 1,
        }
        assert assembly.context_counts == {"graph": 1, "verbatim": 0, "replay": 0}

    async def test_after_turn_appends_turn_and_returns_compacted_context(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """after_turn() should persist a turn and return a compact lifecycle bundle."""
        old_event = MagicMock(
            seq=3,
            type="transcript.turn",
            actor="user",
            payload={"role": "user", "content": "Older context."},
            hash="b" * 64,
        )
        event = MagicMock(
            seq=4,
            type="transcript.turn",
            actor="assistant",
            payload={"role": "assistant", "content": "Use graceful fallback."},
            hash="c" * 64,
        )
        fabric.session_manager.replay.return_value = MagicMock(
            events=[old_event, event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Graceful fallback (decision)",
                source="keyword",
                score=0.9,
                valid_from=None,
                valid_to=None,
            )
        ]

        assembly = await fabric.after_turn(
            role="assistant",
            content="Use graceful fallback.",
            session_id="agent-1",
            query="fallback decision",
            max_recent_events=1,
            limit=1,
        )

        log = fabric.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        assert log.append.call_args.args == ("transcript.turn",)
        assert log.append.call_args.kwargs["payload"]["content"] == "Use graceful fallback."
        assert assembly.compacted is True
        assert assembly.replay_event_count == 1
        assert "Older context." not in assembly.prompt
        assert "Graceful fallback (decision)" in assembly.prompt

    async def test_assemble_context_warns_for_uncited_projection_context(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Projection-derived context without source citations should be explicit."""
        event = MagicMock(
            seq=7,
            type="document.indexed",
            actor="indexer",
            payload={"content": "Projection candidate."},
            hash="e" * 64,
        )
        fabric.session_manager.replay.return_value = MagicMock(
            events=[event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Compacted projection summary",
                source="projection",
                score=0.7,
                valid_from=None,
                valid_to=None,
            )
        ]

        assembly = await fabric.assemble_context("projection", session_id="agent-1")

        assert assembly.warnings == [
            "projection context 'Compacted projection summary' lacks source-level citation"
        ]
        assert "# Context Warnings" in assembly.prompt
        assert "lacks source-level citation" in assembly.prompt

    async def test_assemble_context_accepts_cited_projection_context(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Projection context with an Eventloom citation should not warn."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Source-backed projection record",
                source="projection",
                score=0.8,
                valid_from=None,
                valid_to=None,
                citation="eventloom://agent-1/events/9#aaaaaaaaaaaa",
            )
        ]

        assembly = await fabric.assemble_context("projection", session_id="agent-1")

        assert assembly.warnings == []
        assert "# Context Warnings" not in assembly.prompt

    async def test_assemble_context_warns_when_replay_is_truncated_without_retrieval(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Replay truncation without retrieved support should be visible to callers."""
        events = [
            MagicMock(
                seq=idx,
                type="transcript.turn",
                actor="assistant",
                payload={"content": f"Turn {idx}"},
                hash=str(idx) * 64,
            )
            for idx in range(1, 4)
        ]
        fabric.session_manager.replay.return_value = MagicMock(
            events=events,
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []

        assembly = await fabric.assemble_context(
            "handoff",
            session_id="agent-1",
            max_recent_events=1,
        )

        assert assembly.compacted is True
        assert assembly.warnings == [
            "recent replay was truncated and no retrieved source context was available"
        ]
        assert "Turn 1" not in assembly.prompt
        assert "recent replay was truncated" in assembly.prompt

    async def test_query_merges_projection_records_with_source_citations(
        self,
        tmp_path: Path,
    ) -> None:
        """MemoryFabric should use projection artifacts as cited routing candidates."""
        log = EventLog(tmp_path / "projection-source.jsonl")
        log.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": "docs/cache.md",
                "start_line": 2,
                "end_line": 6,
                "content": "Cache routing note records identity-code-0001.",
            },
        )
        projection = build_compaction_projection(
            log,
            provider=HashEmbeddingProvider(dimension=64),
            strategy="medoid",
        )
        projection_path = write_compaction_projection(
            projection,
            tmp_path / "projection.compaction.json",
        )

        with (
            patch("zaxy.core.GraphStore") as mock_graph_cls,
            patch("zaxy.core.QueryRouter") as mock_router_cls,
            patch("zaxy.core.build_reranker") as mock_build_reranker,
            patch("zaxy.core.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.core.SessionManager") as mock_session_cls,
        ):
            session_mgr = MagicMock()
            session_mgr.get.return_value.eventlog = MagicMock()
            mock_session_cls.return_value = session_mgr
            mock_graph_cls.return_value = AsyncMock()
            router = AsyncMock()
            router.query.return_value = []
            mock_router_cls.return_value = router
            mock_build_reranker.return_value = None
            mock_tracer_cls.return_value = AsyncMock()
            fabric = MemoryFabric(projection_paths=[projection_path])

        contexts = await fabric.query("cache identity-code-0001", limit=3)

        assert len(contexts) == 1
        assert contexts[0].source == "projection"
        assert contexts[0].metadata is not None
        assert contexts[0].metadata["citation"].startswith("eventloom://default/events/1#")
        assert "docs/cache.md:2-6" in contexts[0].metadata["citations"]

    async def test_query_auto_discovers_projection_records_under_eventloom_path(
        self,
        tmp_path: Path,
    ) -> None:
        """MemoryFabric should auto-load colocated compaction projections."""
        eventloom_dir = tmp_path / ".eventloom"
        eventloom_dir.mkdir()
        log = EventLog(tmp_path / "projection-source.jsonl")
        log.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": "docs/session.md",
                "start_line": 4,
                "end_line": 9,
                "content": "Session projection note records identity-code-0002.",
            },
        )
        projection = build_compaction_projection(
            log,
            provider=HashEmbeddingProvider(dimension=64),
            strategy="medoid",
        )
        write_compaction_projection(
            projection,
            eventloom_dir / "session.compaction.json",
        )

        with (
            patch("zaxy.core.GraphStore") as mock_graph_cls,
            patch("zaxy.core.QueryRouter") as mock_router_cls,
            patch("zaxy.core.build_reranker") as mock_build_reranker,
            patch("zaxy.core.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.core.SessionManager") as mock_session_cls,
        ):
            session_mgr = MagicMock()
            session_mgr.get.return_value.eventlog = MagicMock()
            mock_session_cls.return_value = session_mgr
            mock_graph_cls.return_value = AsyncMock()
            router = AsyncMock()
            router.query.return_value = []
            mock_router_cls.return_value = router
            mock_build_reranker.return_value = None
            mock_tracer_cls.return_value = AsyncMock()
            fabric = MemoryFabric(eventloom_path=str(eventloom_dir))

        contexts = await fabric.query("session identity-code-0002", limit=3)

        assert len(contexts) == 1
        assert contexts[0].source == "projection"
        assert contexts[0].metadata is not None
        assert "docs/session.md:4-9" in contexts[0].metadata["citations"]

    async def test_handoff_bundle_combines_summary_replay_and_context(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """handoff_bundle() should produce a portable session handoff object."""
        event = MagicMock(
            seq=5,
            type="goal.created",
            actor="user",
            payload={"title": "Ship lifecycle hooks"},
            hash="d" * 64,
        )
        fabric.session_manager.handoff_summary.return_value = {
            "event_count": 5,
            "goals": ["Ship lifecycle hooks"],
            "open_tasks": [],
            "last_actor": "user",
        }
        fabric.session_manager.replay.return_value = MagicMock(
            events=[event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []

        bundle = await fabric.handoff_bundle(
            session_id="agent-1",
            query="lifecycle hooks",
            replay_from_seq=5,
        )

        assert isinstance(bundle, HandoffBundle)
        assert bundle.session_id == "agent-1"
        assert bundle.summary["event_count"] == 5
        assert bundle.integrity_ok is True
        assert bundle.replay_event_count == 1
        assert "Ship lifecycle hooks" in bundle.prompt

    async def test_cleanup_subagent_appends_cleanup_event_and_bundle(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """cleanup_subagent() should mark a subagent session as cleaned up."""
        event = MagicMock(
            seq=9,
            type="subagent.cleaned",
            actor="zaxy",
            payload={"summary": "Subagent finished retrieval."},
            hash="e" * 64,
        )
        fabric.session_manager.handoff_summary.return_value = {"event_count": 9}
        fabric.session_manager.replay.return_value = MagicMock(
            events=[event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []

        bundle = await fabric.cleanup_subagent(
            parent_session_id="main",
            subagent_session_id="worker-1",
            summary="Subagent finished retrieval.",
        )

        log = fabric.session_manager.get.return_value.eventlog
        appended_types = [call.args[0] for call in log.append.call_args_list]
        assert appended_types == ["subagent.cleaned", "subagent.completed"]
        assert log.append.call_args_list[0].kwargs["thread"] == "worker-1"
        assert log.append.call_args_list[0].kwargs["payload"]["parent_session_id"] == "main"
        assert log.append.call_args_list[1].kwargs["payload"]["status"] == "succeeded"
        assert bundle.session_id == "worker-1"
        assert bundle.summary["event_count"] == 9


# ------------------------------------------------------------------
# Replay tests
# ------------------------------------------------------------------

class TestReplay:
    """Tests for event replay."""

    async def test_replay_delegates_to_session_manager(self, fabric: MemoryFabric) -> None:
        """replay() should delegate to SessionManager.replay()."""
        mock_result = MagicMock()
        fabric.session_manager.replay.return_value = mock_result
        result = await fabric.replay(from_seq=5)
        fabric.session_manager.replay.assert_called_once_with("default", from_seq=5)
        assert result is mock_result

    async def test_replay_with_session_id(self, fabric: MemoryFabric) -> None:
        """replay() should forward session_id to SessionManager."""
        mock_result = MagicMock()
        fabric.session_manager.replay.return_value = mock_result
        result = await fabric.replay(from_seq=1, session_id="agent-1")
        fabric.session_manager.replay.assert_called_once_with("agent-1", from_seq=1)
        assert result is mock_result


# ------------------------------------------------------------------
# Invalidation tests
# ------------------------------------------------------------------

class TestInvalidate:
    """Tests for bi-temporal invalidation."""

    async def test_invalidate_entity(self, fabric: MemoryFabric) -> None:
        """invalidate() should call graph.invalidate_entity."""
        await fabric.invalidate("OldFact", "fact", "2024-06-01T00:00:00Z")
        fabric.graph.invalidate_entity.assert_awaited_once_with(
            "OldFact", "fact", "2024-06-01T00:00:00Z", session_id="default"
        )

    async def test_auto_connects(self, fabric: MemoryFabric) -> None:
        """invalidate() should auto-connect if not already connected."""
        await fabric.invalidate("x", "y", "2024-01-01T00:00:00Z")
        fabric.graph.connect.assert_awaited_once()


# ------------------------------------------------------------------
# Handoff tests
# ------------------------------------------------------------------

class TestHandoff:
    """Tests for handoff summary generation."""

    async def test_handoff_delegates_to_session_manager(self, fabric: MemoryFabric) -> None:
        """handoff_summary() should delegate to SessionManager.handoff_summary()."""
        fabric.session_manager.handoff_summary.return_value = {"event_count": 42}
        summary = await fabric.handoff_summary()
        fabric.session_manager.handoff_summary.assert_called_once_with("default")
        assert summary["event_count"] == 42

    async def test_handoff_with_session_id(self, fabric: MemoryFabric) -> None:
        """handoff_summary() should forward session_id to SessionManager."""
        fabric.session_manager.handoff_summary.return_value = {"event_count": 7}
        summary = await fabric.handoff_summary(session_id="agent-2")
        fabric.session_manager.handoff_summary.assert_called_once_with("agent-2")
        assert summary["event_count"] == 7
