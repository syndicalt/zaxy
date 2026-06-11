"""Tests for zaxy.core — MemoryFabric orchestrator.

Tests cover the full orchestration pipeline: event → extract → graph → query,
with all external dependencies mocked."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.compaction import build_compaction_projection, write_compaction_projection
from zaxy.config import Settings
from zaxy.coordination import CoordinationBrief
from zaxy.core import (
    QUERY_PAGE_CACHE_TTL_SECONDS,
    Context,
    ContextAssembly,
    ContextRefreshReport,
    HandoffBundle,
    MemoryCheckout,
    MemoryFabric,
    QueryPage,
    build_memory_checkout,
)
from zaxy.embedding import HashEmbeddingProvider
from zaxy.event import Event, EventLog, IntegrityReport, ReplayResult
from zaxy.query import ContextChunk
from zaxy.refs import MemoryRef
from zaxy.retrieval_profile import resolve_retrieval_profile
from zaxy.verbatim import VerbatimIndex


class BrokenEmbeddingProvider:
    """Embedding provider test double that simulates provider outages."""

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding down")


def test_memory_fabric_constructs_projection_store_through_factory(tmp_path: Path) -> None:
    """MemoryFabric should use the backend-neutral projection factory."""
    with (
        patch("zaxy.core.build_projection_store") as mock_build,
        patch("zaxy.core.QueryRouter"),
        patch("zaxy.core.build_reranker", return_value=None),
        patch("zaxy.core.MemoryTracer"),
    ):
        mock_build.return_value = AsyncMock()

        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)

    assert fabric.graph is mock_build.return_value
    assert mock_build.call_args.args[0].backend == "embedded"


def test_memory_fabric_accepts_explicit_pggraph_projection_backend(tmp_path: Path) -> None:
    """Framework integrations should be able to select pgGraph without env mutation."""
    with (
        patch("zaxy.core.build_projection_store") as mock_build,
        patch("zaxy.core.QueryRouter"),
        patch("zaxy.core.build_reranker", return_value=None),
        patch("zaxy.core.MemoryTracer"),
    ):
        mock_build.return_value = AsyncMock()

        MemoryFabric(
            eventloom_path=str(tmp_path / ".eventloom"),
            projection_backend="pggraph",
            pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
            tracer_disabled=True,
        )

    config = mock_build.call_args.args[0]
    assert config.backend == "pggraph"
    assert config.pggraph_dsn == "postgresql://postgres:postgres@localhost:5432/zaxy"


def test_memory_fabric_accepts_explicit_embedded_projection_backend(tmp_path: Path) -> None:
    """Framework integrations should be able to select the embedded graph backend."""
    embedded_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    with (
        patch("zaxy.core.build_projection_store") as mock_build,
        patch("zaxy.core.QueryRouter"),
        patch("zaxy.core.build_reranker", return_value=None),
        patch("zaxy.core.MemoryTracer"),
    ):
        mock_build.return_value = AsyncMock()

        MemoryFabric(
            eventloom_path=str(tmp_path / ".eventloom"),
            projection_backend="embedded",
            embedded_graph_path=embedded_path,
            tracer_disabled=True,
        )

    config = mock_build.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path == embedded_path


def test_memory_fabric_accepts_explicit_latticedb_projection_backend(tmp_path: Path) -> None:
    """Framework integrations should be able to select the LatticeDB candidate backend."""
    latticedb_path = tmp_path / ".eventloom" / "projections" / "memory.latticedb"
    with (
        patch("zaxy.core.build_projection_store") as mock_build,
        patch("zaxy.core.QueryRouter"),
        patch("zaxy.core.build_reranker", return_value=None),
        patch("zaxy.core.MemoryTracer"),
    ):
        mock_build.return_value = AsyncMock()

        MemoryFabric(
            eventloom_path=str(tmp_path / ".eventloom"),
            projection_backend="latticedb",
            latticedb_path=latticedb_path,
            tracer_disabled=True,
        )

    config = mock_build.call_args.args[0]
    assert config.backend == "latticedb"
    assert config.latticedb_path == latticedb_path


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


async def test_memory_fabric_reuses_verbatim_index_until_eventloom_changes(tmp_path: Path) -> None:
    """Repeated source-lane calls should not rebuild unchanged Eventloom indexes."""
    fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
    session = fabric.session_manager.get("agent")
    session.eventlog.append(
        "transcript.turn",
        actor="assistant",
        payload={
            "source": "codex",
            "turn_index": 1,
            "role": "assistant",
            "content": "Cached verbatim source recall keeps answer assembly fast.",
        },
        thread="agent",
    )

    with patch("zaxy.core.VerbatimIndex.from_event_logs", wraps=VerbatimIndex.from_event_logs) as build_index:
        await fabric.query_verbatim("cached source recall", session_id="agent", limit=1)
        await fabric.query_verbatim("answer assembly fast", session_id="agent", limit=1)
        session.eventlog.append(
            "transcript.turn",
            actor="assistant",
            payload={
                "source": "codex",
                "turn_index": 2,
                "role": "assistant",
                "content": "A new event should invalidate the cached verbatim index.",
            },
            thread="agent",
        )
        await fabric.query_verbatim("new event invalidate", session_id="agent", limit=1)

    assert build_index.call_count == 2


async def test_memory_fabric_queries_packet_projection_sources(tmp_path: Path) -> None:
    """Packet projections should enter source-aware context retrieval."""
    fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
    session = fabric.session_manager.get("agent")
    event = session.eventlog.append(
        "llm.packet.projected",
        actor="zaxy-packet-projector",
        payload={
            "session_id": "agent",
            "source_event_seq": 2,
            "source_event_hash": "b" * 64,
            "provider_path": "/v1/responses",
            "summary": "LLM packet /v1/responses status 200. User: Mira owns dashboards.",
        },
        thread="agent",
    )

    contexts = await fabric.query_verbatim("Who owns dashboards?", session_id="agent", limit=1)

    assert contexts[0].content == "LLM packet /v1/responses status 200. User: Mira owns dashboards."
    assert contexts[0].metadata == {
        "citation": f"eventloom://agent/events/{event.seq}#{event.hash}",
        "source_kind": "packet_projection",
        "event_seq": event.seq,
        "event_type": "llm.packet.projected",
        "event_thread": "agent",
        "event_timestamp": event.timestamp,
        "session_id": "agent",
        "source_event_seq": 2,
        "source_event_hash": "b" * 64,
        "provider_path": "/v1/responses",
    }


def _feedback_event(
    seq: int,
    event_type: str,
    *,
    citation: str,
    purpose: str,
    outcome: str,
    entity_name: str,
    entity_type: str,
    feedback: str | None = None,
) -> Event:
    payload: dict[str, object] = {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "citation": citation,
        "purpose": {"profile": purpose},
        "outcome": outcome,
    }
    if feedback is not None:
        payload["feedback"] = feedback
    return Event(
        seq=seq,
        timestamp=f"2026-06-02T00:00:0{seq}Z",
        type=event_type,
        actor="assistant",
        thread="agent-1",
        payload=payload,
        hash=str(seq) * 64,
    )

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def fabric() -> MemoryFabric:
    """Return a MemoryFabric with mocked dependencies."""
    with (
        patch("zaxy.core.EventLog") as mock_log_cls,
        patch("zaxy.core.build_projection_store") as mock_build_projection_store,
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
        mock_build_projection_store.return_value = graph

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
            patch("zaxy.core.build_projection_store"),
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

    def test_initializes_query_router_from_named_retrieval_profile(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Named retrieval profiles should resolve before provider wiring."""
        monkeypatch.setenv("RETRIEVAL_PROFILE", "local_sota")
        from zaxy.config import get_settings

        get_settings.cache_clear()
        with (
            patch("zaxy.core.build_projection_store") as mock_build_projection_store,
            patch("zaxy.core.QueryRouter") as mock_router_cls,
            patch("zaxy.core.build_embedding_provider") as mock_build_embedding_provider,
            patch("zaxy.core.build_reranker") as mock_build_reranker,
            patch("zaxy.core.MemoryTracer"),
            patch("zaxy.core.SessionManager") as mock_session_cls,
        ):
            mock_build_embedding_provider.return_value = object()
            mock_build_reranker.return_value = object()
            mock_session_cls.return_value.get.return_value.eventlog = MagicMock()

            fabric = MemoryFabric()

        profile = resolve_retrieval_profile(fabric.settings)
        projection_config = mock_build_projection_store.call_args.args[0]
        assert profile.name == "local_sota"
        assert fabric.retrieval_profile == profile
        assert projection_config.embedding_dimension == 1024
        assert mock_build_embedding_provider.call_args.args[0].embedding_provider == "sentence-transformers"
        assert mock_build_embedding_provider.call_args.args[0].embedding_dimension == 1024
        assert mock_router_cls.call_args.kwargs["scoring_profile"] == "recall"
        assert mock_router_cls.call_args.kwargs["reranker"] is mock_build_reranker.return_value

    async def test_connect_initializes_graph_and_tracer(self, fabric: MemoryFabric) -> None:
        """connect() should init schema and connect tracer."""
        await fabric.connect()
        fabric.graph.connect.assert_awaited_once()
        fabric.graph.init_schema.assert_awaited_once()
        fabric.tracer.connect.assert_awaited_once()
        assert fabric._connected is True

    async def test_connect_warms_projection_session_when_supported(self, fabric: MemoryFabric) -> None:
        """connect() should prewarm embedded projection indexes before first checkout."""
        fabric.graph.warm_session = AsyncMock()
        fabric.settings.eventloom_thread = "agent-1"

        await fabric.connect()

        fabric.graph.warm_session.assert_awaited_once_with(session_id="agent-1")

    async def test_connect_warms_source_index_for_configured_session(self, tmp_path: Path) -> None:
        """connect() should prewarm Eventloom source recall before first answer-ready checkout."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        session = fabric.session_manager.get("agent-1")
        session.eventlog.append(
            "document.indexed",
            actor="assistant",
            payload={
                "path": "memory.md",
                "content": "Warm source recall before the first answer-ready checkout.",
                "start_line": 1,
                "end_line": 1,
            },
            thread="agent-1",
        )
        fabric.settings.eventloom_thread = "agent-1"

        with patch("zaxy.core.VerbatimIndex.from_event_logs", wraps=VerbatimIndex.from_event_logs) as build_index:
            await fabric.connect()
            assert build_index.call_count == 1
            await fabric.query_verbatim("answer-ready checkout", session_id="agent-1", limit=1)

        assert build_index.call_count == 1

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

    async def test_append_projects_generated_inferred_edge_events(self, fabric: MemoryFabric) -> None:
        """append() should persist high-confidence inferred edges as Eventloom events."""
        task_event = Event(
            seq=1,
            timestamp="2024-01-01T00:00:00Z",
            type="task.completed",
            actor="codex",
            thread="agent-1",
            payload={
                "taskId": "task-7",
                "summary": "Implemented Memory Checkout.",
                "decision": "Use Memory Checkout as the model-facing state contract",
                "decision_event_seq": 5,
                "decision_event_hash": "a" * 64,
            },
            hash="b" * 64,
        )
        inferred_payload = {
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
                "source_event_seq": 1,
                "source_event_hash": "b" * 64,
                "decision_event_seq": 5,
                "decision_event_hash": "a" * 64,
                "reason": "task.completed explicitly cited a decision Eventloom event",
            },
        }
        inferred_event = Event(
            seq=2,
            timestamp="2024-01-01T00:00:01Z",
            type="inference.edge.generated",
            actor="zaxy-inference",
            thread="agent-1",
            payload=inferred_payload,
            prev_hash="b" * 64,
            hash="c" * 64,
        )
        log = fabric.session_manager.get.return_value.eventlog
        log.append.side_effect = [task_event, inferred_event]

        await fabric.append(
            "task.completed",
            actor="codex",
            payload=task_event.payload,
            session_id="agent-1",
        )

        assert log.append.call_count == 2
        inferred_call = log.append.call_args_list[1]
        assert inferred_call.args == ("inference.edge.generated",)
        assert inferred_call.kwargs["actor"] == "zaxy-inference"
        assert inferred_call.kwargs["thread"] == "agent-1"
        assert inferred_call.kwargs["payload"]["source"]["name"] == "task-7"
        assert inferred_call.kwargs["payload"]["target"]["entity_type"] == "decision"
        assert inferred_call.kwargs["payload"]["confidence"] == 0.86
        assert fabric.graph.upsert_extraction.await_count == 2


class TestQueryPagination:
    """Tests for stable query continuation pages."""

    async def test_query_page_returns_next_cursor_and_continues_without_repeating(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Continuation cursors should page through one ranked result set."""
        fabric._connected = True
        fabric.query_router.query.return_value = [
            ContextChunk(content="alpha", source="keyword", score=0.9, valid_from=None, valid_to=None),
            ContextChunk(content="beta", source="keyword", score=0.8, valid_from=None, valid_to=None),
            ContextChunk(content="gamma", source="keyword", score=0.7, valid_from=None, valid_to=None),
        ]

        first = await fabric.query_page("roadmap", limit=2, session_id="agent-1")
        second = await fabric.query_page(
            "roadmap",
            limit=2,
            session_id="agent-1",
            cursor=first.next_cursor,
        )

        assert isinstance(first, QueryPage)
        assert [context.content for context in first.contexts] == ["alpha", "beta"]
        assert first.next_cursor is not None
        assert first.has_more is True
        assert [context.content for context in second.contexts] == ["gamma"]
        assert second.next_cursor is None
        assert second.has_more is False

    async def test_query_page_serves_repeat_page_from_cache(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """An identical page request inside the TTL should not re-run retrieval."""
        fabric._connected = True
        fabric.query_router.query.return_value = [
            ContextChunk(content="alpha", source="keyword", score=0.9, valid_from=None, valid_to=None),
            ContextChunk(content="beta", source="keyword", score=0.8, valid_from=None, valid_to=None),
        ]

        first = await fabric.query_page("roadmap", limit=2, session_id="agent-1")
        repeat = await fabric.query_page("roadmap", limit=2, session_id="agent-1")

        assert [context.content for context in first.contexts] == ["alpha", "beta"]
        assert [context.content for context in repeat.contexts] == ["alpha", "beta"]
        assert fabric.query_router.query.await_count == 1

    async def test_query_page_append_invalidates_cached_pages_for_session(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """New session events must drop cached pages so results stay fresh."""
        fabric._connected = True
        fabric.query_router.query.return_value = [
            ContextChunk(content="alpha", source="keyword", score=0.9, valid_from=None, valid_to=None),
        ]

        await fabric.query_page("roadmap", limit=2, session_id="agent-1")
        await fabric.append("note.created", actor="tester", payload={"text": "hi"}, session_id="agent-1")
        await fabric.query_page("roadmap", limit=2, session_id="agent-1")

        assert fabric.query_router.query.await_count == 2

    async def test_query_page_cache_detects_writers_that_bypass_fabric_append(
        self,
        fabric: MemoryFabric,
        tmp_path: Path,
    ) -> None:
        """Direct EventLog appends must invalidate cached pages via the log signature."""
        log_path = tmp_path / "agent-1.jsonl"
        log_path.write_text("seed\n")
        fabric.session_manager.get.return_value.eventlog.path = str(log_path)
        key = ("roadmap", "agent-1", None, None)

        signature = fabric._query_page_log_signature("agent-1")
        assert signature is not None
        fabric._store_query_page_contexts(key, 3, signature, [])
        assert (
            fabric._cached_query_page_contexts(key, 3, fabric._query_page_log_signature("agent-1"))
            == []
        )

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("direct-writer-event\n")

        assert (
            fabric._cached_query_page_contexts(key, 3, fabric._query_page_log_signature("agent-1"))
            is None
        )

    async def test_query_page_cache_expires_after_ttl(
        self,
        fabric: MemoryFabric,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cached pages must not outlive the freshness window."""
        import time

        fabric._connected = True
        fabric.query_router.query.return_value = [
            ContextChunk(content="alpha", source="keyword", score=0.9, valid_from=None, valid_to=None),
        ]
        clock = {"now": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

        await fabric.query_page("roadmap", limit=2, session_id="agent-1")
        clock["now"] += QUERY_PAGE_CACHE_TTL_SECONDS + 1.0
        await fabric.query_page("roadmap", limit=2, session_id="agent-1")

        assert fabric.query_router.query.await_count == 2

    async def test_query_page_rejects_cursor_for_different_query(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Cursor state should be bound to the original query."""
        fabric._connected = True
        fabric.query_router.query.return_value = [
            ContextChunk(content="alpha", source="keyword", score=0.9, valid_from=None, valid_to=None),
            ContextChunk(content="beta", source="keyword", score=0.8, valid_from=None, valid_to=None),
        ]

        first = await fabric.query_page("roadmap", limit=1, session_id="agent-1")

        with pytest.raises(ValueError, match="cursor does not match query"):
            await fabric.query_page(
                "different roadmap",
                limit=1,
                session_id="agent-1",
                cursor=first.next_cursor,
            )


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

    async def test_context_feedback_preserves_purpose_profile(self, fabric: MemoryFabric) -> None:
        """Feedback should preserve the purpose that made retrieved context useful."""
        context = Context(
            content="Accepted parent state: API failures trace to expired JWKS cache handling.",
            source="keyword",
            score=0.93,
            metadata={
                "entity_name": "expired JWKS cache handling",
                "entity_type": "accepted_finding",
                "citation": "eventloom://auth-main/events/8#hhhhhhhhhhhh",
            },
        )

        count = await fabric.record_context_feedback(
            [context],
            feedback="used",
            session_id="auth-main",
            actor="coordinator",
            importance=0.9,
            purpose="coordinate",
            outcome="supported_handoff",
        )

        assert count == 1
        payload = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1].kwargs["payload"]
        assert payload["purpose"]["profile"] == "coordinate"
        assert payload["purpose"]["expected_action"] == "brief_promote_or_handoff"
        assert payload["outcome"] == "supported_handoff"

    async def test_positive_feedback_falls_back_to_content_identity(self, fabric: MemoryFabric) -> None:
        """Context without entity metadata should still produce a stable reinforcement event."""
        context = Context(content="Fallback note", source="eventloom", score=0.5)

        await fabric.record_context_feedback([context], feedback="helpful", session_id="agent-1")

        payload = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1].kwargs["payload"]
        assert payload["entity_name"] == "Fallback note"
        assert payload["entity_type"] == "memory"


class TestSalienceReinforcementWiring:
    """Salience emitters ride the fabric append path without changing behavior."""

    @staticmethod
    def _checkout_setup(fabric: MemoryFabric) -> None:
        event = MagicMock(
            seq=3,
            type="decision.recorded",
            actor="assistant",
            payload={"decision": "Memory checkout should be the agent context contract."},
            hash="c" * 64,
            thread="agent-1",
            timestamp="2026-06-09T12:00:00Z",
        )
        fabric.session_manager.replay.return_value = MagicMock(
            events=[event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Use Memory Checkout as the prompt-ready context contract.",
                source="keyword",
                score=0.95,
                valid_from="2026-05-10T12:00:00Z",
                valid_to=None,
                citation=f"eventloom://agent-1/events/3#{'c' * 12}",
                entity_name="memory checkout",
                entity_type="decision",
            ),
        ]

    async def test_checkout_memory_appends_batched_surfaced_reinforcement(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Checkout should append one surfaced event through the event-spec path."""
        self._checkout_setup(fabric)

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "What memory contract should the model use?",
                session_id="agent-1",
                limit=3,
            )

        call = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1]
        assert call.args == ("memory.reinforcement",)
        assert call.kwargs["thread"] == "agent-1"
        assert call.kwargs["actor"] == "zaxy-memory"
        assert call.kwargs["payload"]["kind"] == "surfaced"
        assert call.kwargs["payload"]["targets"] == [{"seq": 3, "hash": "c" * 64}]
        assert call.kwargs["payload"]["source"]["checkout_id"] == (
            f"eventloom://agent-1/events/3#{'c' * 12}"
        )
        assert call.kwargs["payload"]["authority_status"] == "non_authoritative"
        assert checkout.current_facts

    async def test_failed_reinforcement_append_never_fails_the_checkout(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Reinforcement is best-effort observability state."""
        self._checkout_setup(fabric)

        with (
            patch.object(fabric, "query_verbatim", return_value=[]),
            patch.object(
                fabric,
                "_append_event_spec",
                AsyncMock(side_effect=RuntimeError("append unavailable")),
            ),
        ):
            checkout = await fabric.checkout_memory(
                "What memory contract should the model use?",
                session_id="agent-1",
                limit=3,
            )

        assert checkout.current_facts
        assert checkout.warnings is not None

    async def test_invalidate_appends_invalidated_reinforcement_with_entity_provenance(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Fabric-level invalidation should emit negative reinforcement for CLI and MCP."""
        fabric._connected = True
        fabric.graph.search_exact.return_value = [
            SimpleNamespace(
                properties={"source_event_seq": 9, "source_event_hash": "e" * 64}
            )
        ]

        await fabric.invalidate(
            "salience ledger",
            "decision",
            "2026-06-10T00:00:00Z",
            session_id="agent-1",
        )

        fabric.graph.invalidate_entity.assert_awaited_once_with(
            "salience ledger",
            "decision",
            "2026-06-10T00:00:00Z",
            session_id="agent-1",
        )
        call = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1]
        assert call.args == ("memory.reinforcement",)
        assert call.kwargs["payload"]["kind"] == "invalidated"
        assert call.kwargs["payload"]["targets"] == [{"seq": 9, "hash": "e" * 64}]
        assert call.kwargs["payload"]["source"]["invalidation_id"] == (
            "invalidate:decision:salience ledger@2026-06-10T00:00:00Z"
        )

    async def test_invalidate_emission_failure_never_fails_the_invalidation(
        self,
        fabric: MemoryFabric,
    ) -> None:
        fabric._connected = True
        fabric.graph.search_exact.side_effect = RuntimeError("projection down")

        await fabric.invalidate(
            "salience ledger",
            "decision",
            "2026-06-10T00:00:00Z",
            session_id="agent-1",
        )

        fabric.graph.invalidate_entity.assert_awaited_once()
        appended_types = [
            call.args[0]
            for call in fabric.session_manager.get.return_value.eventlog.append.call_args_list
        ]
        assert "memory.reinforcement" not in appended_types


class TestPurposeOutcomeLearning:
    """Tests for replay-derived purpose outcome effects on future checkout."""

    async def test_checkout_memory_repeated_positive_purpose_outcomes_boost_rank(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Repeated successful outcome history should boost matching memory in the next checkout."""
        fabric._connected = True
        fabric.session_manager.replay.return_value = ReplayResult(
            events=[
                _feedback_event(
                    1,
                    "memory.reinforced",
                    citation=f"eventloom://agent-1/events/9#{'b' * 64}",
                    purpose="coding",
                    outcome="avoided_failed_path",
                    entity_name="migration retry",
                    entity_type="decision",
                ),
                _feedback_event(
                    2,
                    "memory.reinforced",
                    citation=f"eventloom://agent-1/events/9#{'b' * 64}",
                    purpose="coding",
                    outcome="prevented_redundant_investigation",
                    entity_name="migration retry",
                    entity_type="decision",
                ),
            ],
            integrity=IntegrityReport(ok=True, total_events=2),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Migration retry can keep the legacy lock timeout.",
                source="keyword",
                score=0.96,
                valid_from="2026-06-02T00:00:00Z",
                valid_to=None,
                citation=f"eventloom://agent-1/events/8#{'a' * 64}",
                entity_name="legacy migration retry",
                entity_type="decision",
            ),
            ContextChunk(
                content="Retry the migration with lock timeout disabled.",
                source="keyword",
                score=0.9,
                valid_from="2026-06-02T00:00:00Z",
                valid_to=None,
                citation=f"eventloom://agent-1/events/9#{'b' * 64}",
                entity_name="migration retry",
                entity_type="decision",
            ),
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory("migration retry", session_id="agent-1", purpose="coding")

        assert checkout.current_facts[0]["entity_name"] == "migration retry"
        purpose_outcome = checkout.current_facts[0]["score_explanation"]["purpose_outcome"]
        assert purpose_outcome["positive_count"] == 2
        assert purpose_outcome["score_boost"] == 0.12

    async def test_checkout_memory_positive_outcomes_do_not_cross_purpose_profiles(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Outcome boosts are purpose-scoped and must not bleed between task profiles."""
        fabric._connected = True
        fabric.session_manager.replay.return_value = ReplayResult(
            events=[
                _feedback_event(
                    1,
                    "memory.reinforced",
                    citation=f"eventloom://agent-1/events/9#{'b' * 64}",
                    purpose="coordinate",
                    outcome="supported_handoff",
                    entity_name="migration retry",
                    entity_type="decision",
                )
            ],
            integrity=IntegrityReport(ok=True, total_events=1),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Migration retry can keep the legacy lock timeout.",
                source="keyword",
                score=0.96,
                valid_from="2026-06-02T00:00:00Z",
                valid_to=None,
                citation=f"eventloom://agent-1/events/8#{'a' * 64}",
                entity_name="legacy migration retry",
                entity_type="decision",
            ),
            ContextChunk(
                content="Retry the migration with lock timeout disabled.",
                source="keyword",
                score=0.9,
                valid_from="2026-06-02T00:00:00Z",
                valid_to=None,
                citation=f"eventloom://agent-1/events/9#{'b' * 64}",
                entity_name="migration retry",
                entity_type="decision",
            ),
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory("migration retry", session_id="agent-1", purpose="coding")

        assert checkout.current_facts[0]["entity_name"] == "legacy migration retry"
        assert "purpose_outcome" not in checkout.current_facts[1].get("score_explanation", {})

    async def test_checkout_memory_repeated_negative_outcomes_surface_warning_candidate(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Repeated negative outcome history should warn without deleting memory."""
        fabric._connected = True
        citation = f"eventloom://agent-1/events/9#{'b' * 64}"
        fabric.session_manager.replay.return_value = ReplayResult(
            events=[
                _feedback_event(
                    1,
                    "memory.feedback",
                    citation=citation,
                    purpose="coding",
                    outcome="caused_regression",
                    feedback="irrelevant",
                    entity_name="migration retry",
                    entity_type="decision",
                ),
                _feedback_event(
                    2,
                    "memory.feedback",
                    citation=citation,
                    purpose="coding",
                    outcome="failed",
                    feedback="irrelevant",
                    entity_name="migration retry",
                    entity_type="decision",
                ),
            ],
            integrity=IntegrityReport(ok=True, total_events=2),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Retry the migration with lock timeout disabled.",
                source="keyword",
                score=0.9,
                valid_from="2026-06-02T00:00:00Z",
                valid_to=None,
                citation=citation,
                entity_name="migration retry",
                entity_type="decision",
            )
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory("migration retry", session_id="agent-1", purpose="coding")

        assert checkout.diagnostics["warning_count"] == 1
        assert "Checkout contains warnings that reduce confidence." in checkout.quality["reasons"]
        candidates = checkout.diagnostics["purpose_policy"]["suppression_candidates"]
        assert candidates == [
            {
                "entity_name": "migration retry",
                "entity_type": "decision",
                "citation": citation,
                "negative_count": 2,
                "positive_count": 0,
                "outcomes": ["caused_regression", "failed"],
            }
        ]

    async def test_synthesis_candidate_use_writes_eventloom_artifact(self, fabric: MemoryFabric) -> None:
        """Used answer candidates should write a cited synthesis artifact event."""
        checkout = MemoryCheckout(
            session_id="agent-1",
            query="How much did I spend on bike expenses in total?",
            prompt="# Memory Checkout",
            working_set={},
            ref=None,
            current_facts=[],
            evidence=[
                {
                    "content": "session_id=answer-1 I spent $120 on a bike helmet.",
                    "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                    "source_lane": "verbatim",
                },
                {
                    "content": "session_id=answer-2 I spent $45 on a bike tune-up.",
                    "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
                    "source_lane": "verbatim",
                },
                {
                    "content": "session_id=answer-3 I spent $20 on bike lights.",
                    "citation": "eventloom://agent-1/events/3#cccccccccccc",
                    "source_lane": "verbatim",
                },
            ],
            provenance=[],
            retention={},
            warnings=[],
            guidance={},
            quality={"answerability": "answer_from_memory", "confidence": 0.86},
            diagnostics={
                "slot_plan": {
                    "version": "slot_plan_v1",
                    "required_slots": ["source", "numeric"],
                    "optional_slots": ["exact", "semantic"],
                },
                "synthesis": {
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "currency",
                            "confidence": 0.83,
                            "answer_key": "currency_total_answer",
                            "answer": "$185",
                            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
                            "excluded_source_ids": ["answer-4"],
                        }
                    ]
                },
            },
            context_counts={},
            replay_event_count=0,
            compacted=False,
            assembly_policy={},
        )

        event = await fabric.record_synthesis_candidate(
            checkout,
            candidate=checkout.diagnostics["synthesis"]["answer_candidates"][0],
            outcome="used",
            actor="assistant",
        )

        log = fabric.session_manager.get.return_value.eventlog
        call = log.append.call_args_list[-1]
        assert call.args == ("memory.synthesis.used",)
        assert call.kwargs["actor"] == "assistant"
        assert call.kwargs["thread"] == "agent-1"
        assert call.kwargs["payload"] == {
            "query": "How much did I spend on bike expenses in total?",
            "outcome": "used",
            "answer_candidate": {
                "rank": 1,
                "type": "currency",
                "confidence": 0.83,
                "answer_key": "currency_total_answer",
                "answer": "$185",
                "support_source_ids": ["answer-1", "answer-2", "answer-3"],
                "excluded_source_ids": ["answer-4"],
            },
            "quality": {"answerability": "answer_from_memory", "confidence": 0.86},
            "slot_plan": {
                "version": "slot_plan_v1",
                "required_slots": ["source", "numeric"],
                "optional_slots": ["exact", "semantic"],
            },
            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
            "excluded_source_ids": ["answer-4"],
            "citations": [
                "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                "eventloom://agent-1/events/2#bbbbbbbbbbbb",
                "eventloom://agent-1/events/3#cccccccccccc",
            ],
        }
        assert event.seq == 1

    async def test_synthesis_candidate_rejection_writes_audit_event(self, fabric: MemoryFabric) -> None:
        """Rejected answer candidates should be auditable without reinforcement semantics."""
        checkout = MemoryCheckout(
            session_id="agent-1",
            query="How many weddings did I attend?",
            prompt="# Memory Checkout",
            working_set={},
            ref=None,
            current_facts=[],
            evidence=[],
            provenance=[],
            retention={},
            warnings=[],
            guidance={},
            quality={"answerability": "refresh_recommended", "confidence": 0.52},
            diagnostics={
                "synthesis": {
                    "answer_candidates": [
                        {"rank": 1, "type": "count", "answer": "4"}
                    ]
                }
            },
            context_counts={},
            replay_event_count=0,
            compacted=False,
            assembly_policy={},
        )

        await fabric.record_synthesis_candidate(
            checkout,
            candidate={"rank": 1, "type": "count", "answer": "4"},
            outcome="rejected",
            actor="assistant",
            reason="supporting sources were incomplete",
        )

        call = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1]
        assert call.args == ("memory.synthesis.rejected",)
        assert call.kwargs["payload"]["reason"] == "supporting sources were incomplete"
        assert call.kwargs["payload"]["outcome"] == "rejected"

    async def test_synthesis_candidate_rejects_foreign_checkout_candidate(self, fabric: MemoryFabric) -> None:
        """Candidate feedback should fail closed when it is not from the checkout."""
        checkout = MemoryCheckout(
            session_id="agent-1",
            query="How many weddings did I attend?",
            prompt="# Memory Checkout",
            working_set={},
            ref=None,
            current_facts=[],
            evidence=[],
            provenance=[],
            retention={},
            warnings=[],
            guidance={},
            quality={"answerability": "refresh_recommended", "confidence": 0.52},
            diagnostics={
                "synthesis": {
                    "answer_candidates": [
                        {"rank": 1, "type": "count", "answer": "4", "support_source_ids": ["answer-1"]}
                    ]
                }
            },
            context_counts={},
            replay_event_count=0,
            compacted=False,
            assembly_policy={},
        )

        with pytest.raises(ValueError, match="diagnostics.synthesis.answer_candidates"):
            await fabric.record_synthesis_candidate(
                checkout,
                candidate={"rank": 1, "type": "count", "answer": "4", "support_source_ids": ["answer-99"]},
                outcome="used",
                actor="assistant",
            )

        fabric.session_manager.get.return_value.eventlog.append.assert_not_called()

    async def test_synthesis_evidence_feedback_writes_row_event(self, fabric: MemoryFabric) -> None:
        """Used evidence rows should write auditable row-level reinforcement events."""
        checkout = MemoryCheckout(
            session_id="agent-1",
            query="How much did I spend on bike expenses in total?",
            prompt="# Memory Checkout",
            working_set={},
            ref=None,
            current_facts=[],
            evidence=[],
            provenance=[],
            retention={},
            warnings=[],
            guidance={},
            quality={"answerability": "answer_from_memory", "confidence": 0.86},
            diagnostics={
                "slot_plan": {"version": "slot_plan_v1", "operation": "sum_values"},
                "synthesis": {
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "currency",
                            "answer": "$145",
                            "support_source_ids": ["answer-1"],
                        }
                    ],
                    "ledger_rows": [
                        {
                            "fact_id": "currency:0:0",
                            "source_group": "answer-1",
                            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                            "kind": "currency",
                            "value": "120",
                            "include_reason": "currency_amount",
                        }
                    ],
                },
            },
            context_counts={},
            replay_event_count=0,
            compacted=False,
            assembly_policy={},
        )

        event = await fabric.record_synthesis_evidence(
            checkout,
            row=checkout.diagnostics["synthesis"]["ledger_rows"][0],
            outcome="used",
            candidate=checkout.diagnostics["synthesis"]["answer_candidates"][0],
            actor="assistant",
            reason="row supported arithmetic",
        )

        call = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1]
        assert call.args == ("memory.evidence.reinforced",)
        assert call.kwargs["actor"] == "assistant"
        assert call.kwargs["thread"] == "agent-1"
        assert call.kwargs["payload"]["source_group"] == "answer-1"
        assert call.kwargs["payload"]["fact_id"] == "currency:0:0"
        assert call.kwargs["payload"]["reason"] == "row supported arithmetic"
        assert event.seq == 1

    async def test_synthesis_evidence_exclusion_writes_evidence_excluded_event(self, fabric: MemoryFabric) -> None:
        """Excluded synthesis rows should write evidence exclusion audit events."""
        checkout = MemoryCheckout(
            session_id="agent-1",
            query="How much did I spend on bike expenses in total?",
            prompt="# Memory Checkout",
            working_set={},
            ref=None,
            current_facts=[],
            evidence=[],
            provenance=[],
            retention={},
            warnings=[],
            guidance={},
            quality={"answerability": "answer_from_memory", "confidence": 0.86},
            diagnostics={},
            context_counts={},
            replay_event_count=0,
            compacted=False,
            assembly_policy={},
        )

        await fabric.record_synthesis_evidence(
            checkout,
            row={
                "fact_id": "currency:duplicate",
                "source_group": "answer-4",
                "citation": "eventloom://agent-1/events/4#dddddddddddd",
                "kind": "currency",
                "value": "40",
                "exclude_reason": "duplicate_identity",
            },
            outcome="excluded",
            actor="assistant",
            reason="duplicate source row",
        )

        call = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1]
        assert call.args == ("memory.evidence.excluded",)
        assert call.kwargs["payload"]["outcome"] == "excluded"
        assert call.kwargs["payload"]["source_group"] == "answer-4"
        assert call.kwargs["payload"]["reason"] == "duplicate source row"

    async def test_synthesis_artifact_created_writes_deterministic_payload(self, fabric: MemoryFabric) -> None:
        """Checkout answer candidates should be persisted as synthesis artifacts."""
        checkout = MemoryCheckout(
            session_id="agent-1",
            query="How much did I spend on bike expenses in total?",
            prompt="# Memory Checkout",
            working_set={},
            ref=None,
            current_facts=[],
            evidence=[
                {
                    "content": "session_id=answer-1 I spent $120 on a bike helmet.",
                    "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                    "source_lane": "verbatim",
                }
            ],
            provenance=[],
            retention={},
            warnings=[],
            guidance={},
            quality={"answerability": "answer_from_memory", "confidence": 0.86},
            diagnostics={
                "synthesis": {
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "currency",
                            "answer": "$120",
                            "support_source_ids": ["answer-1"],
                        }
                    ]
                }
            },
            context_counts={},
            replay_event_count=0,
            compacted=False,
            assembly_policy={},
        )

        event = await fabric.record_synthesis_artifact(checkout, actor="assistant")

        call = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1]
        assert call.args == ("memory.synthesis.artifact.created",)
        assert call.kwargs["actor"] == "assistant"
        assert call.kwargs["thread"] == "agent-1"
        assert call.kwargs["payload"]["schema_version"] == "synthesis_artifact_v1"
        assert call.kwargs["payload"]["artifact_id"].startswith("sha256:")
        assert call.kwargs["payload"]["answer_candidates"][0]["answer"] == "$120"
        assert call.kwargs["payload"]["support_packet"]["citations"] == [
            "eventloom://agent-1/events/1#aaaaaaaaaaaa"
        ]
        assert event.seq == 1

    async def test_positive_feedback_preserves_packet_memory_provenance(self, fabric: MemoryFabric) -> None:
        """Packet-memory feedback should preserve source packet identifiers."""
        context = Context(
            content="LLM packet /v1/responses status 200. User: Mira owns dashboards.",
            source="packet_memory",
            score=0.6,
            metadata={
                "citation": "eventloom://agent-1/events/6#ffffffffffff",
                "source_kind": "packet_projection",
                "source_event_seq": 5,
                "source_event_hash": "b" * 64,
                "provider_path": "/v1/responses",
                "model": "gpt-test",
            },
        )

        await fabric.record_context_feedback([context], feedback="used", session_id="agent-1")

        payload = fabric.session_manager.get.return_value.eventlog.append.call_args_list[-1].kwargs["payload"]
        assert payload["entity_name"] == "LLM packet /v1/responses status 200. User: Mira owns dashboards."
        assert payload["entity_type"] == "packet_memory"
        assert payload["source_event_seq"] == 5
        assert payload["source_event_hash"] == "b" * 64
        assert payload["provider_path"] == "/v1/responses"
        assert payload["model"] == "gpt-test"

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


class TestContextRefresh:
    """Tests for incremental context refresh orchestration."""

    async def test_refresh_context_appends_delta_and_index_events(
        self,
        fabric: MemoryFabric,
        tmp_path,
    ) -> None:  # type: ignore[no-untyped-def]
        """refresh_context() should append source and projection lifecycle events."""
        fabric.eventloom_path = tmp_path / ".eventloom"
        doc = tmp_path / "README.md"
        doc.write_text("Alpha\n", encoding="utf-8")

        report = await fabric.refresh_context(tmp_path, kind="documents", session_id="agent-1", max_lines=20)

        assert isinstance(report, ContextRefreshReport)
        assert report.summary == {
            "kind": "documents",
            "discovered": 1,
            "changed": 0,
            "unchanged": 0,
            "deleted": 0,
            "indexed": 1,
            "retired": 0,
            "transform_changed": 0,
        }
        log = fabric.session_manager.get.return_value.eventlog
        assert [call.args[0] for call in log.append.call_args_list] == [
            "source.discovered",
            "document.indexed",
            "projection.updated",
        ]
        assert log.append.call_args_list[0].kwargs["thread"] == "agent-1"

    async def test_refresh_context_persists_state_and_skips_unchanged(
        self,
        tmp_path,
    ) -> None:
        """A second refresh should skip re-indexing unchanged sources using persisted state."""
        eventloom = tmp_path / ".eventloom"
        doc = tmp_path / "README.md"
        doc.write_text("Alpha\n", encoding="utf-8")
        fabric = MemoryFabric(eventloom_path=str(eventloom), tracer_disabled=True)
        fabric.graph = AsyncMock()

        first = await fabric.refresh_context(tmp_path, kind="documents", session_id="agent-1")
        second = await fabric.refresh_context(tmp_path, kind="documents", session_id="agent-1")

        assert first.summary["indexed"] == 1
        assert second.summary["unchanged"] == 1
        assert second.summary["indexed"] == 0
        events = [event.type for event in fabric.session_manager.get("agent-1").eventlog.read_all()]
        assert events == [
            "source.discovered",
            "document.indexed",
            "projection.updated",
            "source.unchanged",
        ]

    async def test_refresh_context_records_replayable_retirement_for_changed_sources(
        self,
        tmp_path,
    ) -> None:
        """Changed sources should emit replayable retirement before current projections."""
        eventloom = tmp_path / ".eventloom"
        doc = tmp_path / "README.md"
        doc.write_text("Alpha\n", encoding="utf-8")
        fabric = MemoryFabric(eventloom_path=str(eventloom), tracer_disabled=True)
        fabric.graph = AsyncMock()

        await fabric.refresh_context(tmp_path, kind="documents", session_id="agent-1")
        doc.write_text("Beta\n", encoding="utf-8")

        report = await fabric.refresh_context(tmp_path, kind="documents", session_id="agent-1")

        assert report.summary["changed"] == 1
        assert report.summary["retired"] == 1
        events = [event.type for event in fabric.session_manager.get("agent-1").eventlog.read_all()]
        assert events[-4:] == [
            "projection.retired",
            "source.changed",
            "document.indexed",
            "projection.updated",
        ]


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

    async def test_retrieve_returns_graph_context_without_source_assembly(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """retrieve() should expose backend evidence without answer-ready source synthesis."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Graph evidence only",
                source="keyword",
                score=1.0,
                valid_from=None,
                valid_to=None,
                citation="eventloom://agent-1/events/1#aaaaaaaaaaaa",
            )
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]) as mock_query_verbatim:
            results = await fabric.retrieve("Graph evidence", session_id="agent-1", limit=5)

        mock_query_verbatim.assert_not_called()
        assert len(results) == 1
        assert results[0].content == "Graph evidence only"
        assert results[0].source == "keyword"

    async def test_retrieve_warms_requested_projection_session_once(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """retrieve() should prewarm non-default embedded sessions before router search."""
        fabric.graph.warm_session = AsyncMock()
        fabric.query_router.query.return_value = []
        fabric._connected = True
        fabric._warmed_projection_sessions = {"default"}

        await fabric.retrieve("Graph evidence", session_id="agent-1", limit=5)
        await fabric.retrieve("Graph evidence", session_id="agent-1", limit=5)

        fabric.graph.warm_session.assert_awaited_once_with(session_id="agent-1")

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

    async def test_query_reserves_source_lane_for_evidence_heavy_questions(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Raw query should recover cited source evidence before final prompt truncation."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content=f"graph candidate {index}",
                source="vector",
                score=2.0 - index / 10,
                valid_from=None,
                valid_to=None,
                citation=f"eventloom://agent-1/events/{index}#{str(index) * 12}",
            )
            for index in range(5)
        ]
        source_context = Context(
            content="answer-session-42 contains the house Rachel helped find.",
            source="verbatim",
            score=10.0,
            metadata={
                "citation": "eventloom://agent-1/events/42#cccccccccccc",
                "source_kind": "document",
            },
        )

        with patch.object(fabric, "query_verbatim", return_value=[source_context]) as mock_query_verbatim:
            results = await fabric.query(
                "How many days did it take to find a house after starting to work with Rachel?",
                session_id="agent-1",
                limit=5,
            )

        mock_query_verbatim.assert_any_call(
            "How many days did it take to find a house after starting to work with Rachel?",
            limit=48,
            session_id="agent-1",
        )
        assert len(results) == 5
        assert results[-1].source == "verbatim"
        assert results[-1].metadata is not None
        assert results[-1].metadata["assembly_lane"] == "verbatim"
        assert "answer-session-42" in results[-1].content

    async def test_query_expands_source_lane_with_graph_answer_concepts(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Graph-derived answer concepts should recover verbatim evidence on raw query."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="graph says Max is a Golden Retriever",
                source="keyword",
                score=2.0,
                valid_from=None,
                valid_to=None,
                citation="eventloom://agent-1/events/1#aaaaaaaaaaaa",
            )
        ]
        distractor = Context(
            content="longmemeval_session_id=distractor my dog enjoys trail walks.",
            source="verbatim",
            score=1.0,
            metadata={"citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb"},
        )
        answer = Context(
            content="longmemeval_session_id=answer Max is a Golden Retriever.",
            source="verbatim",
            score=1.0,
            metadata={"citation": "eventloom://agent-1/events/3#cccccccccccc"},
        )

        async def query_verbatim_side_effect(
            query: str,
            *,
            limit: int,
            session_id: str,
        ) -> list[Context]:
            del limit, session_id
            if "Golden Retriever" in query:
                return [answer]
            return [distractor]

        with patch.object(
            fabric,
            "query_verbatim",
            side_effect=query_verbatim_side_effect,
        ) as mock_query_verbatim:
            results = await fabric.query("What breed is my dog?", session_id="agent-1", limit=5)

        called_queries = [call.args[0] for call in mock_query_verbatim.call_args_list]
        assert called_queries[0] == "What breed is my dog?"
        assert any("Golden Retriever" in query for query in called_queries[1:])
        assert any("longmemeval_session_id=answer" in result.content for result in results)

    async def test_query_prioritizes_aggregation_source_expansions(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Aggregation expansions should beat noisy raw source matches."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="graph found a general travel memory",
                source="keyword",
                score=2.0,
                valid_from=None,
                valid_to=None,
                citation="eventloom://agent-1/events/1#aaaaaaaaaaaa",
            )
        ]
        distractor = Context(
            content="longmemeval_session_id=distractor I visited Munich.",
            source="verbatim",
            score=20.0,
            metadata={"citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb"},
        )
        answer = Context(
            content="longmemeval_session_id=answer_55a6940c_1 I saw a dermatologist.",
            source="verbatim",
            score=10.0,
            metadata={"citation": "eventloom://agent-1/events/3#cccccccccccc"},
        )

        async def query_verbatim_side_effect(
            query: str,
            *,
            limit: int,
            session_id: str,
        ) -> list[Context]:
            del limit, session_id
            if query == "doctor physician dermatologist ent visited saw appointment":
                return [answer]
            return [distractor]

        with patch.object(
            fabric,
            "query_verbatim",
            side_effect=query_verbatim_side_effect,
        ) as mock_query_verbatim:
            results = await fabric.query(
                "How many different doctors did I visit?",
                session_id="agent-1",
                limit=5,
            )

        called_queries = [call.args[0] for call in mock_query_verbatim.call_args_list]
        assert called_queries[0] == "doctor physician dermatologist ent visited saw appointment"
        assert any("longmemeval_session_id=answer_55a6940c_1" in result.content for result in results)

    async def test_query_leaves_typed_source_ordering_to_synthesis_bundle(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Raw query should not evidence-rank sources twice before assembly."""
        answer = Context(
            content="longmemeval_session_id=answer I bought bike lights for $40.",
            source="verbatim",
            score=10.0,
            metadata={"citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb"},
        )
        distractor = Context(
            content="longmemeval_session_id=distractor I bought hiking boots for $90.",
            source="verbatim",
            score=8.0,
            metadata={"citation": "eventloom://agent-1/events/3#cccccccccccc"},
        )

        ordered = fabric._order_source_contexts_for_assembly(
            "How much total money have I spent on bike-related expenses?",
            [distractor, answer],
        )

        assert ordered == [distractor, answer]

    async def test_query_adds_source_synthesis_bundle_for_aggregation(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Raw aggregation query should compact multiple source groups into one context."""
        fabric.query_router.query.return_value = []
        source_texts = [
            "longmemeval_session_id=answer_1 I visited a dermatologist for a skin check.",
            "longmemeval_session_id=answer_2 I saw an ENT specialist about sinus pain.",
            "longmemeval_session_id=answer_3 I had an appointment with my primary care physician.",
        ]
        source_contexts = [
            Context(
                content=content,
                source="verbatim",
                score=10.0 - index,
                metadata={"citation": f"eventloom://agent-1/events/{index}#cccccccccccc"},
            )
            for index, content in enumerate(source_texts, start=1)
        ]

        with patch.object(fabric, "query_verbatim", return_value=source_contexts):
            results = await fabric.query(
                "How many different doctors did I visit?",
                session_id="agent-1",
                limit=5,
            )

        assert results[0].source == "verbatim"
        assert results[0].metadata is not None
        assert results[0].metadata["source_kind"] == "source_synthesis"
        assert results[0].metadata["synthesis_packet"]["schema_version"] == "synthesis_packet_v1"
        assert results[0].metadata["synthesis_packet"]["answer_candidates"]
        assert "zaxy_synthesis_bundle=true" in results[0].content
        assert "answer_1" in results[0].content
        assert "answer_2" in results[0].content
        assert "answer_3" in results[0].content

    async def test_query_synthesis_context_uses_typed_bundle_packet(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Synthetic contexts should preserve typed packet data instead of reparsing text."""
        fabric.query_router.query.return_value = []
        typed_packet = {
            "schema_version": "synthesis_packet_v1",
            "answer_candidates": [
                {
                    "rank": 1,
                    "type": "currency",
                    "confidence": 0.91,
                    "answer_key": "currency_total_answer",
                    "answer": "$145",
                    "support_source_ids": ["answer-1", "answer-2"],
                    "excluded_source_ids": [],
                }
            ],
            "ledger_rows": [
                {
                    "fact_id": "typed:currency:1",
                    "source_group": "answer-1",
                    "citation": "eventloom://agent/events/1#aaaaaaaaaaaa",
                    "kind": "currency",
                    "value": "120",
                    "include_reason": "currency_amount",
                }
            ],
            "content": "zaxy_synthesis_bundle=true",
        }
        source_contexts = [
            Context(
                content="longmemeval_session_id=answer-1 I bought a helmet for $120.",
                source="verbatim",
                score=10.0,
                metadata={"citation": "eventloom://agent-1/events/1#cccccccccccc"},
            ),
            Context(
                content="longmemeval_session_id=answer-2 I bought a chain for $25.",
                source="verbatim",
                score=9.0,
                metadata={"citation": "eventloom://agent-1/events/2#cccccccccccc"},
            ),
        ]

        with (
            patch.object(fabric, "query_verbatim", return_value=source_contexts),
            patch(
                "zaxy.core.source_synthesis_bundle_result",
                return_value=SimpleNamespace(
                    content="\n".join(
                        [
                            "zaxy_synthesis_bundle=true",
                            "candidate_rank=1 candidate_type=currency candidate_confidence=0.10",
                            "currency_total_answer=$1",
                        ]
                    ),
                    packet=typed_packet,
                ),
            ),
        ):
            results = await fabric.query(
                "How much total money have I spent on bike-related expenses?",
                session_id="agent-1",
                limit=5,
            )

        packet = results[0].metadata["synthesis_packet"]
        assert packet["answer_candidates"][0]["answer"] == "$145"
        assert packet["ledger_rows"][0]["fact_id"] == "typed:currency:1"

    async def test_query_adds_absence_bundle_when_source_synthesis_defers(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Missing target evidence should produce an absence bundle from cited sources."""
        fabric.query_router.query.return_value = []
        source_contexts = [
            Context(
                content=(
                    "content=longmemeval_session_id=answer_e5131a1b_abs_1 "
                    "I've been working professionally for 9 years and currently use a physical notebook."
                ),
                source="verbatim",
                score=8.0,
                metadata={"citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa"},
            ),
            Context(
                content=(
                    "content=longmemeval_session_id=answer_e5131a1b_abs_2 "
                    "I've been working at NovaTech for about 4 years and 3 months now."
                ),
                source="verbatim",
                score=7.5,
                metadata={"citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb"},
            ),
        ]

        with patch.object(fabric, "query_verbatim", return_value=source_contexts):
            results = await fabric.query(
                "How long have I been working before I started my current job at Google?",
                session_id="agent-1",
                limit=5,
            )

        assert results[0].source == "verbatim"
        assert results[0].metadata is not None
        assert results[0].metadata["source_kind"] == "source_absence"
        assert "zaxy_absence_check=true" in results[0].content
        assert "answer_e5131a1b_abs_1" in results[0].content
        assert "answer_e5131a1b_abs_2" in results[0].content

    async def test_query_synthesizes_elapsed_duration_at_prior_event(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Answer-ready query should subtract event age from current duration evidence."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content=(
                    "longmemeval_session_id=answer_436d4309_1 "
                    "I've been taking weekly guitar lessons with a new instructor, Alex, "
                    "for six weeks now."
                ),
                source="keyword",
                score=3.0,
                valid_from=None,
                valid_to=None,
                citation="file://longmemeval/answer_436d4309_1.md:1",
            ),
            ContextChunk(
                content=(
                    "longmemeval_session_id=answer_436d4309_2 "
                    "summary references the new guitar amp source."
                ),
                source="keyword",
                score=2.0,
                valid_from=None,
                valid_to=None,
                citation="file://longmemeval/answer_436d4309_2.md:1",
            ),
        ]
        source_contexts = [
            Context(
                content=(
                    "longmemeval_session_id=answer_436d4309_1 "
                    "I've been taking weekly guitar lessons with a new instructor, Alex, "
                    "for six weeks now."
                ),
                source="verbatim",
                score=9.0,
                metadata={"citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa"},
            ),
            Context(
                content=(
                    "longmemeval_session_id=answer_436d4309_2 "
                    "I just got a new amp two weeks ago."
                ),
                source="verbatim",
                score=8.0,
                metadata={"citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb"},
            ),
        ]

        with patch.object(fabric, "query_verbatim", return_value=source_contexts):
            results = await fabric.query(
                "How long had I been taking guitar lessons when I bought the new guitar amp?",
                session_id="agent-1",
                limit=5,
            )

        bundle = next(result for result in results if (result.metadata or {}).get("source_kind") == "source_synthesis")
        assert "elapsed_at_event_answer=Four weeks" in bundle.content

    async def test_query_backfills_graph_source_groups_for_elapsed_synthesis(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Graph-ranked provenance groups should get verbatim backfill before synthesis."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content=(
                    "longmemeval_session_id=answer_436d4309_1 "
                    "I've been taking weekly guitar lessons with a new instructor, Alex, "
                    "for six weeks now."
                ),
                source="keyword",
                score=3.0,
                valid_from=None,
                valid_to=None,
                citation="file://longmemeval/answer_436d4309_1.md:1",
            ),
            ContextChunk(
                content=(
                    "longmemeval_session_id=answer_436d4309_2 "
                    "summary references the new guitar amp source."
                ),
                source="keyword",
                score=2.0,
                valid_from=None,
                valid_to=None,
                citation="file://longmemeval/answer_436d4309_2.md:1",
            ),
        ]
        noisy_sources = [
            Context(
                content=(
                    f"longmemeval_session_id=distractor_{index} "
                    "I've been playing guitar for six weeks now."
                ),
                source="verbatim",
                score=20.0 - index,
                metadata={"citation": f"eventloom://agent-1/events/{index}#aaaaaaaaaaaa"},
            )
            for index in range(20)
        ]
        amp_source = Context(
            content=(
                "longmemeval_session_id=answer_436d4309_2 "
                "I just got a new amp two weeks ago."
            ),
            source="verbatim",
            score=4.0,
            metadata={"citation": "eventloom://agent-1/events/42#bbbbbbbbbbbb"},
        )

        async def query_verbatim_side_effect(
            query: str,
            *,
            limit: int,
            session_id: str,
        ) -> list[Context]:
            del limit, session_id
            if query == "answer_436d4309_2":
                return [amp_source]
            return noisy_sources

        with patch.object(
            fabric,
            "query_verbatim",
            side_effect=query_verbatim_side_effect,
        ):
            results = await fabric.query(
                "How long had I been taking guitar lessons when I bought the new guitar amp?",
                session_id="agent-1",
                limit=5,
            )

        bundle = next(result for result in results if (result.metadata or {}).get("source_kind") == "source_synthesis")
        assert "elapsed_at_event_answer=Four weeks" in bundle.content
        assert "source_id=answer_436d4309_2" in bundle.content

    async def test_query_synthesizes_from_graph_evidence_when_source_lane_drifts(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Answer-ready synthesis should use cited graph evidence, not only source-lane hits."""
        fabric.query_router.query.return_value = [
            ContextChunk(
                content=(
                    "longmemeval/gpt4/session/answer_1/salient-turn-0003.md:1-6 "
                    "(document) — summary=longmemeval_session_id=answer_1 "
                    "I recently had an issue with my car's GPS system on 3/22, "
                    "and I had to take it back to the dealership to get it fixed."
                ),
                source="keyword",
                score=3.0,
                valid_from=None,
                valid_to=None,
                citation="file://longmemeval/gpt4/session/answer_1/salient-turn-0003.md:1",
            ),
            ContextChunk(
                content=(
                    "longmemeval/gpt4/session/answer_2/salient-turn-0001.md:1-6 "
                    "(document) — summary=longmemeval_session_id=answer_2 "
                    "I just got my car serviced for the first time on March 15th."
                ),
                source="keyword",
                score=2.0,
                valid_from=None,
                valid_to=None,
                citation="file://longmemeval/gpt4/session/answer_2/salient-turn-0001.md:1",
            ),
        ]
        source_contexts = [
            Context(
                content=(
                    "longmemeval_session_id=distractor "
                    "I am researching a newer Toyota Corolla hybrid."
                ),
                source="verbatim",
                score=9.0,
                metadata={"citation": "eventloom://agent-1/events/1#cccccccccccc"},
            )
        ]

        with patch.object(fabric, "query_verbatim", return_value=source_contexts):
            results = await fabric.query(
                "What was the first issue I had with my new car after its first service?",
                session_id="agent-1",
                limit=5,
            )

        bundle = next(result for result in results if result.metadata.get("source_kind") == "source_synthesis")
        assert "issue_candidate=GPS system not functioning correctly" in bundle.content

    async def test_query_promotes_later_high_evidence_source_hits(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Source-lane hits from expanded queries should be evidence-ordered before assembly."""
        fabric.query_router.query.return_value = []
        source_contexts = [
            Context(
                content=(
                    "longmemeval_session_id=distractor "
                    "For a hike, a prime lens like the 50mm can be a good choice for photography."
                ),
                source="verbatim",
                score=30.0,
                metadata={"citation": "eventloom://agent-1/events/1#cccccccccccc"},
            ),
            Context(
                content=(
                    "longmemeval_session_id=answer_b9d9150e_2 "
                    "I recently got a new 50mm f/1.8 prime lens that I'm still getting used to."
                ),
                source="verbatim",
                score=20.0,
                metadata={"citation": "eventloom://agent-1/events/2#dddddddddddd"},
            ),
        ]

        with patch.object(fabric, "query_verbatim", return_value=source_contexts):
            results = await fabric.query(
                "Which event happened first, the road trip to the coast or the arrival of the new prime lens?",
                session_id="agent-1",
                limit=5,
            )

        assert any("answer_b9d9150e_2" in result.content for result in results[:5])
        distractor_rank = next(
            index for index, result in enumerate(results) if "longmemeval_session_id=distractor" in result.content
        )
        answer_rank = next(
            index for index, result in enumerate(results) if "answer_b9d9150e_2" in result.content
        )
        assert answer_rank <= distractor_rank + 1

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

    async def test_checkout_memory_returns_current_cited_working_state(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """checkout_memory() should expose current facts, evidence, and prompt state."""
        event = MagicMock(
            seq=4,
            type="decision.recorded",
            actor="assistant",
            payload={"decision": "Memory checkout should be the agent context contract."},
            hash="d" * 64,
        )
        fabric.session_manager.replay.return_value = MagicMock(
            events=[event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Use Memory Checkout as the prompt-ready context contract.",
                source="keyword",
                score=0.95,
                valid_from="2026-05-10T12:00:00Z",
                valid_to=None,
                citation="eventloom://agent-1/events/3#cccccccccccc",
                entity_name="memory checkout",
                entity_type="decision",
            ),
            ContextChunk(
                content="Use raw replay only for model context.",
                source="keyword",
                score=0.7,
                valid_from="2026-05-09T12:00:00Z",
                valid_to="2026-05-10T12:00:00Z",
                citation="eventloom://agent-1/events/2#bbbbbbbbbbbb",
                entity_name="raw replay",
                entity_type="decision",
            ),
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "What memory contract should the model use?",
                session_id="agent-1",
                limit=3,
            )

        assert isinstance(checkout, MemoryCheckout)
        assert checkout.session_id == "agent-1"
        assert checkout.current_facts == [
            {
                "content": "Use Memory Checkout as the prompt-ready context contract.",
                "source": "keyword",
                "score": 0.95,
                "citation": "eventloom://agent-1/events/3#cccccccccccc",
                "valid_from": "2026-05-10T12:00:00Z",
                "valid_to": None,
                "entity_name": "memory checkout",
                "entity_type": "decision",
                "source_lane": "graph",
            }
        ]
        assert checkout.evidence[0]["citation"] == "eventloom://agent-1/events/3#cccccccccccc"
        assert checkout.evidence[0]["source_lane"] == "graph"
        assert checkout.provenance[0]["event_seq"] == 3
        diagnostics = dict(checkout.diagnostics)
        slot_plan = diagnostics.pop("slot_plan")
        assert slot_plan["version"] == "slot_plan_v1"
        assert slot_plan["answer_type"] == "direct_fact"
        assert slot_plan["operation"] == "select_fact"
        assert slot_plan["required_slots"] == []
        assert diagnostics == {
            "source_lanes": {"graph": 2},
            "citation_count": 2,
            "current_citation_count": 1,
            "current_fact_count": 1,
            # The cognitive default profile (2.1.0 flip) always reports the
            # attenuation block; with no reinforcement events it is empty and
            # ranking is identical to plain (cold-start parity).
            "attenuation": {
                "authority_status": "non_authoritative",
                "floor": 0.15,
                "label": "attenuated",
                "excluded_count": 0,
                "excluded": [],
                "exempt_count": 0,
                "exempt": [],
            },
            "evidence_plan": {
                "mode": "direct_fact",
                "needs_source_lane": False,
                "source_lane_slots": 0,
                "required_source_groups": 0,
                "promote_cited_sources": False,
                "reasons": [],
            },
            "evidence_set": {
                "groups": [
                    {
                        "source_id": "eventloom://agent-1/events/3#cccccccccccc",
                        "evidence_count": 1,
                        "citation_count": 1,
                        "citations": ["eventloom://agent-1/events/3#cccccccccccc"],
                        "source_lanes": ["graph"],
                        "top_score": 0.95,
                        "snippet": "Use Memory Checkout as the prompt-ready context contract.",
                    },
                    {
                        "source_id": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
                        "evidence_count": 1,
                        "citation_count": 1,
                        "citations": ["eventloom://agent-1/events/2#bbbbbbbbbbbb"],
                        "source_lanes": ["graph"],
                        "top_score": 0.7,
                        "snippet": "Use raw replay only for model context.",
                    },
                ]
            },
            "superseded_contexts_excluded": 1,
            "warning_count": 0,
            "feedback_recommended": True,
            "feedback_tool": "memory_feedback",
            "feedback_reason": "Reinforce cited context if it materially informed the next response.",
            # 2.1.0 default flip: unconfigured fabrics report the cognitive
            # profile (same local_fast stack plus the cognitive flags).
            "retrieval_profile": {
                "name": "cognitive",
                "embedding_provider": "hash",
                "embedding_model": None,
                "embedding_dimension": 1536,
                "reranker_provider": "lexical",
                "scoring_profile": "balanced",
                "lanes": [
                    "bm25",
                    "hash_vector",
                    "verbatim",
                    "graph",
                    "graph_walk",
                    "lexical_rerank",
                ],
                "hosted": False,
                "experimental": False,
                "cognitive": {
                    "salience_ranking": True,
                    "cue_blending": True,
                    "graph_walk": True,
                },
            },
            "purpose_retrieval_policy": {
                "profile": "general",
                "applied": False,
                "emphasis_terms": [],
                "scoring_profile": "balanced",
                "recall_multiplier": 1,
                "min_recall_limit": 0,
                "base_recall_limit": 3,
                "resolved_recall_limit": 3,
            },
        }
        assert checkout.guidance["recommended_next_call"] == {
            "tool": "memory_checkout",
            "query": "current decisions, blockers, and next actions for: What memory contract should the model use?",
            "reason": "Refresh memory before major follow-up work, after compaction/resume, or when task scope changes.",
        }
        assert checkout.guidance["feedback"]["tool"] == "memory_feedback"
        assert checkout.guidance["feedback"]["payloads"] == [
            {
                "entity_name": "memory checkout",
                "entity_type": "decision",
                "feedback": "used",
                "actor": "assistant",
                "query": "What memory contract should the model use?",
                "source": "keyword",
                "score": 0.95,
                "citation": "eventloom://agent-1/events/3#cccccccccccc",
                "importance": 0.6,
            }
        ]
        assert "Do not treat superseded contexts as current facts." in checkout.guidance["ignore"]
        assert checkout.quality == {
            "answerability": "answer_from_memory",
            "confidence": 0.82,
            "reasons": [
                "Retrieved current facts with Eventloom citations.",
                "Superseded contexts were excluded from current facts.",
            ],
            "required_action": None,
        }
        assert "# Memory Checkout" in checkout.prompt
        assert "## Checkout Quality" in checkout.prompt
        assert "answer_from_memory" in checkout.prompt
        assert "## Checkout Guidance" in checkout.prompt
        assert "## Checkout Diagnostics" in checkout.prompt
        assert "current decisions, blockers, and next actions" in checkout.prompt
        assert "memory_feedback" in checkout.prompt
        assert "Use raw replay only" not in "\n".join(fact["content"] for fact in checkout.current_facts)
        assert checkout.context_counts["graph"] == 2

    async def test_checkout_memory_asks_user_when_no_current_facts(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """checkout_memory() should not imply answerability when retrieval is empty."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "What did we decide about missing memory?",
                session_id="agent-1",
                limit=3,
            )

        assert checkout.current_facts == []
        assert checkout.diagnostics["current_fact_count"] == 0
        assert checkout.diagnostics["current_citation_count"] == 0
        assert checkout.quality["answerability"] == "ask_user"
        assert checkout.quality["confidence"] == 0.25
        assert checkout.quality["required_action"] == {
            "type": "ask_user",
            "reason": "No current facts were retrieved; ask the user for the missing context before answering from memory.",
        }
        assert "No current facts were retrieved." in checkout.prompt
        assert "ask_user" in checkout.prompt

    async def test_checkout_memory_asks_user_when_only_superseded_context_is_retrieved(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """checkout_memory() should keep superseded evidence auditable but not answerable."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Raw replay used to be the model context contract.",
                source="keyword",
                score=0.8,
                valid_from="2026-05-09T12:00:00Z",
                valid_to="2026-05-10T12:00:00Z",
                citation="eventloom://agent-1/events/2#bbbbbbbbbbbb",
                entity_name="raw replay",
                entity_type="decision",
            )
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "What memory contract should the model use?",
                session_id="agent-1",
                limit=3,
            )

        assert checkout.current_facts == []
        assert checkout.evidence[0]["citation"] == "eventloom://agent-1/events/2#bbbbbbbbbbbb"
        assert checkout.diagnostics["citation_count"] == 1
        assert checkout.diagnostics["current_citation_count"] == 0
        assert checkout.retention["superseded_contexts_excluded"] == 1
        assert checkout.quality["answerability"] == "ask_user"
        assert checkout.quality["confidence"] == 0.25
        assert "Superseded contexts were excluded from current facts." in checkout.quality["reasons"]

    async def test_checkout_memory_refreshes_when_current_facts_lack_citations(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """checkout_memory() should request refresh before relying on uncited current facts."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="The model should use Memory Checkout.",
                source="keyword",
                score=0.74,
                valid_from="2026-05-10T12:00:00Z",
                valid_to=None,
                citation=None,
                entity_name="memory checkout",
                entity_type="decision",
            )
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "What memory contract should the model use?",
                session_id="agent-1",
                limit=3,
            )

        assert checkout.current_facts[0]["citation"] is None
        assert checkout.diagnostics["citation_count"] == 0
        assert checkout.diagnostics["current_citation_count"] == 0
        assert checkout.warnings == ["Checkout contains current facts without Eventloom citations."]
        assert checkout.quality["answerability"] == "refresh_recommended"
        assert checkout.quality["confidence"] == 0.29
        assert checkout.quality["required_action"] == checkout.guidance["recommended_next_call"]
        assert "Retrieved current facts, but they lack Eventloom citations." in checkout.quality["reasons"]

    def test_checkout_memory_refreshes_when_checkout_has_warnings(self) -> None:
        """build_memory_checkout() should force refresh when compaction or warnings reduce confidence."""
        assembly = ContextAssembly(
            session_id="agent-1",
            prompt="# Active Memory Working Set\n- Memory checkout is current.",
            contexts=[
                Context(
                    content="Memory checkout is the current contract.",
                    source="keyword",
                    score=0.91,
                    metadata={"citation": "eventloom://agent-1/events/8#hhhhhhhhhhhh"},
                    valid_from="2026-05-10T12:00:00Z",
                    valid_to=None,
                ),
                Context(
                    content="The feedback loop should reinforce used checkout context.",
                    source="keyword",
                    score=0.9,
                    metadata={"citation": "eventloom://agent-1/events/9#iiiiiiiiiiii"},
                    valid_from="2026-05-10T12:05:00Z",
                    valid_to=None,
                ),
            ],
            working_set={"items": []},
            context_counts={"graph": 2},
            replay_event_count=12,
            compacted=True,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(
            query="What is the current memory contract?",
            assembly=assembly,
        )

        assert checkout.diagnostics["current_fact_count"] == 2
        assert checkout.diagnostics["current_citation_count"] == 2
        assert checkout.warnings == ["Recent replay was compacted to fit the checkout budget."]
        assert checkout.quality["answerability"] == "refresh_recommended"
        assert checkout.quality["confidence"] == 0.77
        assert checkout.quality["required_action"] == checkout.guidance["recommended_next_call"]
        assert "Checkout contains warnings that reduce confidence." in checkout.quality["reasons"]

    def test_memory_checkout_serializes_purpose_profile(self) -> None:
        """build_memory_checkout() should preserve the purpose-conditioned contract."""
        assembly = ContextAssembly(
            session_id="agent-1",
            prompt="# Active Memory Working Set\n- JWKS cache risk is current.",
            contexts=[
                Context(
                    content="JWKS cache risk is current.",
                    source="keyword",
                    score=0.91,
                    metadata={"citation": "eventloom://agent-1/events/8#hhhhhhhhhhhh"},
                    valid_from="2026-05-10T12:00:00Z",
                    valid_to=None,
                )
            ],
            working_set={"items": []},
            context_counts={"graph": 1},
            replay_event_count=8,
            compacted=False,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(
            query="review current auth risks",
            assembly=assembly,
            purpose={"profile": "review", "expected_action": "approve_or_block"},
        )
        payload = checkout.to_dict()

        assert payload["purpose"]["profile"] == "review"
        assert payload["purpose"]["expected_action"] == "approve_or_block"
        assert payload["diagnostics"]["purpose"]["evidence_policy"] == "cited_current_facts_required"
        assert "## Purpose Profile" in payload["prompt"]

    def test_checkout_feedback_payloads_include_purpose_profile(self) -> None:
        """Checkout feedback templates should preserve useful-for-what metadata."""
        assembly = ContextAssembly(
            session_id="agent-1",
            prompt="# Active Memory Working Set\n- JWKS cache risk is current.",
            contexts=[
                Context(
                    content="JWKS cache risk is current.",
                    source="keyword",
                    score=0.91,
                    metadata={
                        "citation": "eventloom://agent-1/events/8#hhhhhhhhhhhh",
                        "entity_name": "JWKS cache risk",
                        "entity_type": "risk",
                    },
                    valid_from="2026-05-10T12:00:00Z",
                    valid_to=None,
                )
            ],
            working_set={"items": []},
            context_counts={"graph": 1},
            replay_event_count=8,
            compacted=False,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(
            query="review current auth risks",
            assembly=assembly,
            purpose={"profile": "review", "expected_action": "approve_or_block"},
        )

        feedback_payload = checkout.guidance["feedback"]["payloads"][0]
        assert feedback_payload["purpose"]["profile"] == "review"
        assert feedback_payload["purpose"]["expected_action"] == "approve_or_block"

    def test_memory_checkout_applies_coordinate_purpose_suppression(self) -> None:
        """Coordinate purpose should prevent worker-local pending rows becoming current memory."""
        assembly = ContextAssembly(
            session_id="auth-main",
            prompt="# Active Memory Working Set\n- Accepted and pending findings were retrieved.",
            contexts=[
                Context(
                    content="Accepted parent state: API failures trace to expired JWKS cache handling.",
                    source="keyword",
                    score=0.91,
                    metadata={
                        "citation": "eventloom://auth-main/events/8#hhhhhhhhhhhh",
                        "mission_id": "auth-main",
                        "finding_id": "finding-api",
                        "coordination_status": "accepted",
                        "authority_scope": "parent-accepted",
                    },
                    valid_from="2026-05-10T12:00:00Z",
                    valid_to=None,
                ),
                Context(
                    content="Pending worker-local claim: UI refresh handling is missing retry state.",
                    source="keyword",
                    score=0.99,
                    metadata={
                        "citation": "eventloom://auth-main/events/9#iiiiiiiiiiii",
                        "mission_id": "auth-main",
                        "worker_id": "auth-ui",
                        "finding_id": "finding-ui",
                        "coordination_status": "pending",
                        "authority_scope": "worker-local",
                    },
                    valid_from="2026-05-10T12:05:00Z",
                    valid_to=None,
                ),
            ],
            working_set={"items": []},
            context_counts={"graph": 2},
            replay_event_count=9,
            compacted=False,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(
            query="current accepted coordination state",
            assembly=assembly,
            purpose="coordinate",
        )

        assert [fact["finding_id"] for fact in checkout.current_facts] == ["finding-api"]
        assert [item["finding_id"] for item in checkout.evidence] == ["finding-api"]
        assert checkout.diagnostics["purpose_policy"]["suppressed_count"] == 1
        assert checkout.diagnostics["purpose_policy"]["suppressed_reasons"] == {"worker_local_pending": 1}
        assert checkout.retention["purpose_policy"]["suppress"] == [
            "worker_local_pending",
            "rejected_finding",
            "stale_unpromoted_finding",
        ]
        assert checkout.guidance["feedback"]["payloads"] == [
            {
                "entity_name": "Accepted parent state: API failures trace to expired JWKS cache handling.",
                "entity_type": "memory",
                "feedback": "used",
                "actor": "assistant",
                "query": "current accepted coordination state",
                "source": "keyword",
                "score": 0.91,
                "citation": "eventloom://auth-main/events/8#hhhhhhhhhhhh",
                "importance": 0.6,
                "mission_id": "auth-main",
                "finding_id": "finding-api",
                "coordination_status": "accepted",
                "authority_scope": "parent-accepted",
                "purpose": checkout.purpose,
            }
        ]
        assert "UI refresh handling" not in checkout.prompt
        assert "Purpose policy suppressed non-matching retrieved rows before projection." in checkout.quality["reasons"]

    def test_memory_checkout_suppresses_worker_and_stale_generic_event_metadata(self) -> None:
        """Coordinate purpose should suppress generic graph rows carrying authority metadata."""
        assembly = ContextAssembly(
            session_id="auth-main",
            prompt="# Active Memory Working Set\n- Accepted, worker-local, and stale rows were retrieved.",
            contexts=[
                Context(
                    content="Parent accepted state: expired JWKS cache caused auth failures.",
                    source="keyword",
                    score=0.91,
                    metadata={
                        "citation": "eventloom://auth-main/events/8#hhhhhhhhhhhh",
                        "authority_scope": "parent-accepted",
                        "status": "accepted",
                    },
                    valid_from="2026-05-10T12:00:00Z",
                    valid_to=None,
                ),
                Context(
                    content="Worker accepted-looking state: database pool caused auth failures.",
                    source="keyword",
                    score=0.99,
                    metadata={
                        "citation": "eventloom://auth-main/events/9#iiiiiiiiiiii",
                        "authority_scope": "worker",
                        "status": "accepted",
                        "promoted": False,
                    },
                    valid_from="2026-05-10T12:05:00Z",
                    valid_to=None,
                ),
                Context(
                    content="Deprecated policy: stale worker rows could be promoted directly.",
                    source="keyword",
                    score=0.98,
                    metadata={
                        "citation": "eventloom://auth-main/events/10#jjjjjjjjjjjj",
                        "authority_scope": "policy",
                        "status": "superseded",
                        "superseded_by": "auth-main:8",
                    },
                    valid_from="2026-05-10T12:06:00Z",
                    valid_to=None,
                ),
                Context(
                    content="Unsupported observation: auth failures might be caused by UI refresh.",
                    source="keyword",
                    score=0.97,
                    metadata={
                        "citation": "eventloom://auth-main/events/11#kkkkkkkkkkkk",
                        "authority_scope": "observation",
                        "status": "unsupported",
                    },
                    valid_from="2026-05-10T12:07:00Z",
                    valid_to=None,
                ),
            ],
            working_set={"items": []},
            context_counts={"graph": 4},
            replay_event_count=11,
            compacted=False,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(
            query="current accepted coordination state",
            assembly=assembly,
            purpose="coordinate",
        )

        assert [fact["content"] for fact in checkout.current_facts] == [
            "Parent accepted state: expired JWKS cache caused auth failures."
        ]
        assert checkout.diagnostics["purpose_policy"]["suppressed_reasons"] == {
            "rejected_finding": 1,
            "stale_unpromoted_finding": 1,
            "worker_local_pending": 1,
        }
        assert checkout.diagnostics["accepted_state"] == {
            "diagnostic_count": 0,
            "mode": "coordinate_accepted_state",
            "selected_citations": ["eventloom://auth-main/events/8#hhhhhhhhhhhh"],
            "selected_count": 1,
        }
        assert "database pool caused auth failures" not in checkout.prompt
        assert "stale worker rows" not in checkout.prompt
        assert "UI refresh" not in checkout.prompt

    def test_memory_checkout_keeps_bridge_evidence_for_accepted_state(self) -> None:
        """Accepted-state checkout should retain current bridge evidence that supports promoted state."""
        assembly = ContextAssembly(
            session_id="auth-main",
            prompt="# Active Memory Working Set\n- Parent state and cited logs were retrieved.",
            contexts=[
                Context(
                    content="Parent-accepted diagnosis: jwks-cache-refresh-regression caused the auth incident.",
                    source="verbatim",
                    score=0.95,
                    metadata={
                        "citation": "eventloom://auth-main/events/4#dddddddddddd",
                        "authority_scope": "parent",
                        "status": "current",
                        "promoted": True,
                    },
                    valid_from="2026-05-10T12:00:00Z",
                    valid_to=None,
                ),
                Context(
                    content="Auth edge logs show JWKS cache refresh regression after key rotation.",
                    source="verbatim",
                    score=0.9,
                    metadata={
                        "citation": "eventloom://auth-main/events/3#cccccccccccc",
                        "authority_scope": "observation",
                        "status": "current",
                    },
                    valid_from="2026-05-10T12:01:00Z",
                    valid_to=None,
                ),
                Context(
                    content="Current but unrelated observation: dashboard route caching improved.",
                    source="verbatim",
                    score=0.89,
                    metadata={
                        "citation": "eventloom://auth-main/events/5#eeeeeeeeeeee",
                        "authority_scope": "observation",
                        "status": "current",
                    },
                    valid_from="2026-05-10T12:02:00Z",
                    valid_to=None,
                ),
            ],
            working_set={"items": []},
            context_counts={"verbatim": 3},
            replay_event_count=5,
            compacted=False,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(
            query="Which accepted auth diagnosis should the responder trust?",
            assembly=assembly,
            purpose="coordinate",
        )

        assert [fact["citation"] for fact in checkout.current_facts] == [
            "eventloom://auth-main/events/4#dddddddddddd",
            "eventloom://auth-main/events/3#cccccccccccc",
        ]
        assert checkout.diagnostics["accepted_state"]["diagnostic_count"] == 1
        assert "dashboard route caching" not in checkout.prompt

    def test_checkout_memory_reports_inferred_context_dependency(self) -> None:
        """build_memory_checkout() should expose inferred-path reliance to the model."""
        assembly = ContextAssembly(
            session_id="agent-1",
            prompt="# Active Memory Working Set\n- Task 7 likely implemented a decision.",
            contexts=[
                Context(
                    content="Task 7 likely implemented the Memory Checkout decision.",
                    source="traversal",
                    score=0.94,
                    metadata={
                        "citation": "eventloom://agent-1/events/12#aaaaaaaaaaaa",
                        "score_explanation": {
                            "inferred_edge_count": 1,
                            "inferred_edge_trust": 0.86,
                            "inferred_edge_trust_multiplier": 1.08,
                            "inferred_edge_method_coverage": 1.0,
                            "inferred_edge_source_coverage": 1.0,
                            "inferred_edge_evidence_coverage": 1.0,
                        },
                    },
                    valid_from="2026-05-10T12:00:00Z",
                    valid_to=None,
                ),
            ],
            working_set={"items": []},
            context_counts={"graph": 1},
            replay_event_count=12,
            compacted=False,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(
            query="What decision did task 7 implement?",
            assembly=assembly,
        )

        assert checkout.current_facts[0]["score_explanation"]["inferred_edge_trust"] == 0.86
        assert checkout.diagnostics["inferred_context"]["context_count"] == 1
        assert checkout.diagnostics["inferred_context"]["average_trust"] == 0.86
        assert "Checkout includes inferred graph paths." in checkout.quality["reasons"]
        assert "Inferred graph context: contexts=1, edges=1, average_trust=0.86" in checkout.prompt

    def test_memory_checkout_surfaces_applicable_skills(self) -> None:
        """build_memory_checkout() should expose applicable procedural memory separately."""
        assembly = ContextAssembly(
            session_id="agent-1",
            prompt="# Active Memory Working Set",
            contexts=[
                Context(
                    content="Skill Python test-first implementation applies to Python feature work.",
                    source="graph",
                    score=0.95,
                    valid_from="2026-05-17T00:00:00Z",
                    valid_to=None,
                    metadata={
                        "entity_name": "skill:python-test-first:v1",
                        "entity_type": "skill_version",
                        "citation": "eventloom://agent-1/events/4#abcd",
                        "skill_id": "python-test-first",
                        "procedure": ["Write failing test", "Run pytest", "Implement minimum code"],
                        "applicability": ["Python feature work"],
                        "status": "validated",
                    },
                ),
            ],
            working_set={"items": []},
            context_counts={"graph": 1},
            replay_event_count=0,
            compacted=False,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(query="implement a Python feature", assembly=assembly)

        assert checkout.diagnostics["skills"]["count"] == 1
        assert checkout.diagnostics["skills"]["items"][0]["skill_id"] == "python-test-first"
        assert "## Applicable Skills" in checkout.prompt
        assert "Write failing test" in checkout.prompt

    def test_memory_checkout_reports_skill_outcome_analytics(self) -> None:
        """Skill Memory checkout should summarize promotion, rollback, and contradiction signals."""
        assembly = ContextAssembly(
            session_id="agent-1",
            prompt="# Active Memory Working Set",
            contexts=[
                Context(
                    content="Skill Python test-first implementation applies to Python feature work.",
                    source="graph",
                    score=0.95,
                    metadata={
                        "entity_name": "skill:python-test-first:v2",
                        "entity_type": "skill_version",
                        "citation": "eventloom://agent-1/events/10#aaaa",
                        "skill_id": "python-test-first",
                        "version": "2",
                        "status": "validated",
                        "procedure": ["Write failing test", "Run pytest"],
                        "applicability": ["Python feature work"],
                    },
                ),
                Context(
                    content="Outcome for python-test-first passed with high confidence.",
                    source="graph",
                    score=0.9,
                    metadata={
                        "entity_name": "skill:python-test-first:v2:outcome:12",
                        "entity_type": "skill_outcome",
                        "citation": "eventloom://agent-1/events/12#bbbb",
                        "skill_id": "python-test-first",
                        "version": "2",
                        "success_score": 0.96,
                        "feedback": "used",
                        "task": "fix benchmark regression",
                    },
                ),
                Context(
                    content="Skill deploy-cache-check was contradicted after a failed rollout.",
                    source="graph",
                    score=0.88,
                    metadata={
                        "entity_name": "skill:deploy-cache-check:v1",
                        "entity_type": "skill_version",
                        "citation": "eventloom://agent-1/events/18#cccc",
                        "skill_id": "deploy-cache-check",
                        "version": "1",
                        "status": "contradicted",
                        "failure_modes": ["misses cache invalidation race"],
                        "rollback": "Use deploy-cache-check v0 until cache race is resolved.",
                    },
                ),
                Context(
                    content="Outcome for deploy-cache-check failed during release validation.",
                    source="graph",
                    score=0.85,
                    metadata={
                        "entity_name": "skill:deploy-cache-check:v1:outcome:19",
                        "entity_type": "skill_outcome",
                        "citation": "eventloom://agent-1/events/19#dddd",
                        "skill_id": "deploy-cache-check",
                        "version": "1",
                        "success_score": 0.2,
                        "feedback": "failed",
                        "task": "release cache validation",
                    },
                ),
            ],
            working_set={"items": []},
            context_counts={"graph": 4},
            replay_event_count=0,
            compacted=False,
            warnings=[],
            assembly_policy={},
        )

        checkout = build_memory_checkout(query="implement a Python feature", assembly=assembly)

        analytics = checkout.diagnostics["skill_analytics"]
        assert analytics["outcome_count"] == 2
        assert analytics["contradiction_count"] == 1
        assert analytics["promotion_candidates"] == [
            {
                "skill_id": "python-test-first",
                "version": "2",
                "status": "validated",
                "success_count": 1,
                "failure_count": 0,
                "average_success_score": 0.96,
                "latest_citation": "eventloom://agent-1/events/12#bbbb",
            }
        ]
        assert analytics["rollback_candidates"][0]["skill_id"] == "deploy-cache-check"
        assert analytics["rollback_candidates"][0]["reason"] == "contradicted"
        assert "## Skill Analytics" in checkout.prompt
        assert "promotion_candidate=python-test-first v2" in checkout.prompt
        assert "rollback_candidate=deploy-cache-check v1" in checkout.prompt

    async def test_checkout_memory_prioritizes_exact_recent_task_context(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """checkout_memory() should rank the turn-relevant memory above noisy older context."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="A memory capture gap was recorded during benchmark debugging.",
                source="keyword",
                score=0.91,
                valid_from="2026-05-10T06:42:06Z",
                valid_to=None,
                citation="eventloom://agent-1/events/1832#gap",
                entity_name="memory capture gap",
                entity_type="event",
            ),
            ContextChunk(
                content="Implemented first-class Memory Checkout for Zaxy.",
                source="keyword",
                score=0.8,
                valid_from="2026-05-10T20:55:40Z",
                valid_to=None,
                citation="eventloom://agent-1/events/1882#checkout",
                entity_name="memory checkout",
                entity_type="task",
            ),
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "memory checkout implementation",
                session_id="agent-1",
                limit=2,
            )

        assert checkout.current_facts[0]["citation"] == "eventloom://agent-1/events/1882#checkout"
        assert checkout.provenance[0]["event_seq"] == 1882

    async def test_checkout_memory_filters_to_resolved_ref(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """checkout_memory(ref=...) should not surface facts after the ref target."""
        resolved_ref = MemoryRef(
            name="refs/heads/main",
            session_id="agent-1",
            target_seq=4,
            target_hash="d" * 64,
            ref_type="branch",
            updated_at="2026-05-10T12:00:00Z",
        )
        old_event = MagicMock(
            seq=4,
            type="task.completed",
            actor="codex",
            payload={"summary": "Available at ref."},
            hash="d" * 64,
        )
        future_event = MagicMock(
            seq=5,
            type="task.completed",
            actor="codex",
            payload={"summary": "Future work."},
            hash="e" * 64,
        )
        fabric.session_manager.replay.return_value = MagicMock(
            events=[old_event, future_event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Available at ref.",
                source="keyword",
                score=0.8,
                valid_from="2026-05-10T12:00:00Z",
                valid_to=None,
                citation="eventloom://agent-1/events/4#dddddddddddd",
                entity_name="available",
                entity_type="task",
            ),
            ContextChunk(
                content="Future work.",
                source="keyword",
                score=0.9,
                valid_from="2026-05-10T13:00:00Z",
                valid_to=None,
                citation="eventloom://agent-1/events/5#eeeeeeeeeeee",
                entity_name="future",
                entity_type="task",
            ),
        ]

        with (
            patch.object(fabric.refs, "resolve", return_value=resolved_ref),
            patch.object(fabric, "query_verbatim", return_value=[]),
        ):
            checkout = await fabric.checkout_memory(
                "what was available",
                session_id="agent-1",
                ref="refs/heads/main",
                limit=2,
            )

        assert checkout.ref is not None
        assert checkout.ref["name"] == "refs/heads/main"
        assert [fact["content"] for fact in checkout.current_facts] == ["Available at ref."]
        assert "Future work." not in checkout.prompt

    def test_checkout_head_ref_uses_eventlog_tail(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """MemoryFabric HEAD refs should resolve from the Eventloom tail."""
        latest = MagicMock(
            seq=9,
            hash="f" * 64,
            timestamp="2026-06-09T12:00:00Z",
        )
        fabric.session_manager.get.return_value.eventlog.last_event.return_value = latest

        ref = fabric._resolve_checkout_ref("HEAD", session_id="agent-1")

        assert ref == MemoryRef(
            name="HEAD",
            session_id="agent-1",
            target_seq=9,
            target_hash="f" * 64,
            ref_type="head",
            updated_at="2026-06-09T12:00:00Z",
        )
        fabric.session_manager.get.return_value.eventlog.last_event.assert_called_once_with()
        fabric.session_manager.replay.assert_not_called()

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
            "packet_memory_enabled": True,
            "packet_memory_slots": 1,
        }
        assert assembly.context_counts == {"graph": 1, "verbatim": 1, "packet_memory": 0, "replay": 0}
        assert assembly.working_set["items"][0]["category"] == "source_anchor"
        assert "# Active Memory Working Set" in assembly.prompt

    async def test_assemble_context_overfetches_source_candidates_for_aggregation(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """assemble_context() should fetch enough sources for evidence planning."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []
        with patch.object(fabric, "query_verbatim", return_value=[]) as mock_query_verbatim:
            await fabric.assemble_context(
                "How many properties did I visit before making an offer?",
                session_id="agent-1",
                limit=8,
            )

        mock_query_verbatim.assert_called_once_with(
            "How many properties did I visit before making an offer?",
            limit=72,
            session_id="agent-1",
        )

    async def test_assemble_context_keeps_recall_set_separate_from_prompt_budget(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Recall overfetch should not inflate the prompt-ready context surface."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content=f"Recall candidate {index}",
                source="keyword",
                score=1.0 - (index / 100),
                valid_from=None,
                valid_to=None,
                citation=f"eventloom://agent-1/events/{index}#{str(index) * 12}",
            )
            for index in range(1, 7)
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            assembly = await fabric.assemble_context(
                "What should I remember?",
                session_id="agent-1",
                limit=2,
                recall_limit=6,
            )

        assert [context.content for context in assembly.contexts] == [
            "Recall candidate 1",
            "Recall candidate 2",
        ]
        assert len(assembly.recall.candidates) == 6
        assert assembly.recall.to_diagnostics() == {
            "candidate_count": 6,
            "evidence_count": 6,
            "source_group_count": 6,
            "budget": 6,
            "lanes": {"graph": 6},
        }
        assert "Recall candidate 6" not in assembly.prompt

    async def test_checkout_memory_uses_internal_recall_without_benchmark_branching(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Checkout should select cited evidence from recall, not only visible prompt contexts."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content=(
                    f"longmemeval_session_id=answer-{index} "
                    f"Wedding memory source {index}."
                ),
                source="keyword",
                score=1.0 - (index / 100),
                valid_from=None,
                valid_to=None,
                citation=f"eventloom://agent-1/events/{index}#{str(index) * 12}",
            )
            for index in range(1, 5)
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "How many weddings did I attend?",
                session_id="agent-1",
                limit=2,
            )

        assert fabric.query_router.query.await_args.kwargs["limit"] > 2
        assert checkout.diagnostics["recall"]["candidate_count"] == 4
        assert checkout.diagnostics["recall"]["source_group_count"] == 4
        # 2.1.0 default flip: the unconfigured fabric reports the cognitive
        # profile (local_fast stack plus the graph_walk lane).
        assert checkout.diagnostics["retrieval_profile"]["name"] == "cognitive"
        assert checkout.diagnostics["retrieval_profile"]["lanes"] == [
            "bm25",
            "hash_vector",
            "verbatim",
            "graph",
            "graph_walk",
            "lexical_rerank",
        ]
        assert [item["citation"] for item in checkout.evidence[:3]] == [
            "eventloom://agent-1/events/1#111111111111",
            "eventloom://agent-1/events/2#222222222222",
            "eventloom://agent-1/events/3#333333333333",
        ]
        assert "required_source_groups=2" in checkout.prompt

    async def test_checkout_memory_general_purpose_does_not_rewrite_retrieval(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """General checkout should keep retrieval query semantics unchanged."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "current release state",
                session_id="agent-1",
                limit=3,
            )

        assert fabric.query_router.query.await_args.args[0] == "current release state"
        assert checkout.diagnostics["purpose_retrieval_policy"] == {
            "profile": "general",
            "applied": False,
            "emphasis_terms": [],
            "scoring_profile": "balanced",
            "recall_multiplier": 1,
            "min_recall_limit": 0,
            "base_recall_limit": 3,
            "resolved_recall_limit": 3,
        }

    async def test_checkout_memory_applies_coordinate_purpose_retrieval_policy(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Coordinate checkout should retrieve with mission-state terms before projection."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = [
            ContextChunk(
                content="Accepted parent state requires proof packet citations.",
                source="keyword",
                score=0.9,
                valid_from=None,
                valid_to=None,
                citation="eventloom://agent-1/events/9#999999999999",
                metadata={
                    "coordination_status": "accepted",
                    "authority_scope": "mission-parent",
                },
            )
        ]

        with patch.object(fabric, "query_verbatim", return_value=[]):
            checkout = await fabric.checkout_memory(
                "current mission state",
                session_id="agent-1",
                limit=3,
                purpose="coordinate",
            )

        retrieval_query = fabric.query_router.query.await_args.args[0]
        assert retrieval_query.startswith("current mission state purpose:coordinate")
        assert "accepted_finding" in retrieval_query
        assert "proof_packet" in retrieval_query
        assert fabric.query_router.query.await_args.kwargs["limit"] == 24
        assert fabric.query_router.query.await_args.kwargs["scoring_profile"] == "recall"
        assert checkout.diagnostics["purpose_retrieval_policy"]["applied"] is True
        assert checkout.diagnostics["purpose_retrieval_policy"]["profile"] == "coordinate"
        assert checkout.diagnostics["purpose_retrieval_policy"]["scoring_profile"] == "recall"
        assert checkout.diagnostics["purpose_retrieval_policy"]["base_recall_limit"] == 3
        assert checkout.diagnostics["purpose_retrieval_policy"]["resolved_recall_limit"] == 24
        assert checkout.current_facts[0]["authority_scope"] == "mission-parent"

    async def test_checkout_memory_synthesizes_top_recall_sources_without_expanding_prompt_budget(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Checkout synthesis should use top recall sources even beyond visible prompt slots."""
        fabric.session_manager.replay.return_value = MagicMock(
            events=[],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []
        source_contexts = [
            ContextChunk(
                content="session_id=distractor-1 I researched bike routes with no expenses.",
                source="verbatim",
                score=0.99,
                valid_from=None,
                valid_to=None,
                metadata={
                    "citation": "eventloom://agent-1/events/1#111111111111",
                    "assembly_lane": "verbatim",
                },
            ),
            ContextChunk(
                content="session_id=distractor-2 My bike commute took 20 minutes.",
                source="verbatim",
                score=0.98,
                valid_from=None,
                valid_to=None,
                metadata={
                    "citation": "eventloom://agent-1/events/2#222222222222",
                    "assembly_lane": "verbatim",
                },
            ),
            ContextChunk(
                content="session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
                source="verbatim",
                score=0.97,
                valid_from=None,
                valid_to=None,
                metadata={
                    "citation": "eventloom://agent-1/events/3#333333333333",
                    "assembly_lane": "verbatim",
                },
            ),
            ContextChunk(
                content="session_id=answer-2 I replaced the bike chain and it cost me $25.",
                source="verbatim",
                score=0.96,
                valid_from=None,
                valid_to=None,
                metadata={
                    "citation": "eventloom://agent-1/events/4#444444444444",
                    "assembly_lane": "verbatim",
                },
            ),
            ContextChunk(
                content="session_id=answer-3 I got a new set of bike lights installed, which were $40.",
                source="verbatim",
                score=0.95,
                valid_from=None,
                valid_to=None,
                metadata={
                    "citation": "eventloom://agent-1/events/5#555555555555",
                    "assembly_lane": "verbatim",
                },
            ),
        ]

        with patch.object(fabric, "query_verbatim", return_value=source_contexts):
            checkout = await fabric.checkout_memory(
                "How much total money have I spent on bike-related expenses?",
                session_id="agent-1",
                limit=2,
            )

        assert fabric.query_router.query.await_args.kwargs["limit"] > 2
        assert "bike lights installed" not in checkout.prompt.split("# Memory Checkout", 1)[0]
        assert checkout.diagnostics["synthesis"]["answer_candidates"][0]["answer"] == "$185"
        assert checkout.diagnostics["synthesis"]["answer_candidates"][0]["support_source_ids"] == [
            "answer-1",
            "answer-2",
            "answer-3",
        ]
        assert list(
            dict.fromkeys(
                row["source_group"]
                for row in checkout.diagnostics["synthesis"]["ledger_rows"]
                if row.get("include_reason") == "currency_amount"
                and not row.get("exclude_reason")
            )
        ) == ["answer-1", "answer-2", "answer-3"]

    async def test_assemble_context_includes_recent_packet_memory_lane(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Recent packet projections should be proactively available as context."""
        packet_event = MagicMock(
            seq=6,
            type="llm.packet.projected",
            actor="zaxy-packet-projector",
            payload={
                "summary": "LLM packet /v1/responses status 200. User: Mira owns dashboards.",
                "source_event_seq": 5,
                "source_event_hash": "b" * 64,
                "provider_path": "/v1/responses",
            },
            hash="f" * 64,
            thread="agent-1",
            timestamp="2024-01-01T00:00:00Z",
        )
        fabric.session_manager.replay.return_value = MagicMock(
            events=[packet_event],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []
        with patch.object(fabric, "query_verbatim", return_value=[]):
            assembly = await fabric.assemble_context(
                "current operating context",
                session_id="agent-1",
                limit=1,
            )

        assert [context.source for context in assembly.contexts] == ["packet_memory"]
        assert assembly.contexts[0].metadata is not None
        assert assembly.contexts[0].metadata["assembly_lane"] == "packet_memory"
        assert assembly.contexts[0].metadata["citation"] == "eventloom://agent-1/events/6#ffffffffffff"
        assert "Mira owns dashboards" in assembly.prompt
        assert assembly.context_counts == {"graph": 0, "verbatim": 0, "packet_memory": 1, "replay": 1}

    async def test_assemble_context_prioritizes_reinforced_packet_memory(
        self,
        fabric: MemoryFabric,
    ) -> None:
        """Reinforced packet memories should outrank merely recent packet memories."""
        reinforced_packet = MagicMock(
            seq=4,
            type="llm.packet.projected",
            actor="zaxy-packet-projector",
            payload={
                "summary": "LLM packet /v1/responses status 200. User: Mira owns dashboards.",
                "source_event_seq": 3,
                "source_event_hash": "b" * 64,
            },
            hash="d" * 64,
            thread="agent-1",
            timestamp="2024-01-01T00:00:00Z",
        )
        newer_packet = MagicMock(
            seq=6,
            type="llm.packet.projected",
            actor="zaxy-packet-projector",
            payload={
                "summary": "LLM packet /v1/responses status 200. User: Alex owns billing.",
                "source_event_seq": 5,
                "source_event_hash": "c" * 64,
            },
            hash="f" * 64,
            thread="agent-1",
            timestamp="2024-01-01T00:05:00Z",
        )
        reinforcement = MagicMock(
            seq=7,
            type="memory.reinforced",
            actor="assistant",
            payload={
                "entity_type": "packet_memory",
                "source_event_hash": "b" * 64,
                "importance": 0.9,
            },
            hash="g" * 64,
            thread="agent-1",
            timestamp="2024-01-01T00:06:00Z",
        )
        fabric.session_manager.replay.return_value = MagicMock(
            events=[reinforced_packet, newer_packet, reinforcement],
            integrity=MagicMock(ok=True),
        )
        fabric.query_router.query.return_value = []
        with patch.object(fabric, "query_verbatim", return_value=[]):
            assembly = await fabric.assemble_context("owner context", session_id="agent-1", limit=1)

        assert len(assembly.contexts) == 1
        assert "Mira owns dashboards" in assembly.contexts[0].content
        assert assembly.contexts[0].score > 0.6
        assert assembly.contexts[0].metadata is not None
        assert assembly.contexts[0].metadata["reinforcement_count"] == 1
        assert assembly.contexts[0].metadata["importance"] == 0.9

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
            "packet_memory_enabled": True,
            "packet_memory_slots": 1,
        }
        assert assembly.context_counts == {"graph": 1, "verbatim": 0, "packet_memory": 0, "replay": 0}

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
            patch("zaxy.core.build_projection_store") as mock_build_projection_store,
            patch("zaxy.core.QueryRouter") as mock_router_cls,
            patch("zaxy.core.build_reranker") as mock_build_reranker,
            patch("zaxy.core.build_embedding_provider") as mock_build_embedding_provider,
            patch("zaxy.core.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.core.SessionManager") as mock_session_cls,
        ):
            session_mgr = MagicMock()
            session_mgr.get.return_value.eventlog = MagicMock()
            mock_session_cls.return_value = session_mgr
            mock_build_projection_store.return_value = AsyncMock()
            router = AsyncMock()
            router.query.return_value = []
            mock_router_cls.return_value = router
            mock_build_reranker.return_value = None
            mock_build_embedding_provider.return_value = None
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
            patch("zaxy.core.build_projection_store") as mock_build_projection_store,
            patch("zaxy.core.QueryRouter") as mock_router_cls,
            patch("zaxy.core.build_reranker") as mock_build_reranker,
            patch("zaxy.core.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.core.SessionManager") as mock_session_cls,
        ):
            session_mgr = MagicMock()
            session_mgr.get.return_value.eventlog = MagicMock()
            mock_session_cls.return_value = session_mgr
            mock_build_projection_store.return_value = AsyncMock()
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


class TestCoordinationAPI:
    """Tests for MemoryFabric coordination convenience methods."""

    async def test_coordinate_methods_write_projected_parent_and_worker_events(
        self,
        tmp_path: Path,
    ) -> None:
        """MemoryFabric should expose the high-level coordination workflow."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("auth-main", objective="Ship auth refactor", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-api", actor="lead")
        finding = await fabric.coordinate_report_finding(
            "auth-main",
            "auth-api",
            summary="API failures trace to expired JWKS cache handling.",
            actor="auth-api-agent",
            evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
            claim_key="auth.failure.cause",
            claim_value="expired-jwks-cache",
        )
        await fabric.coordinate_review_finding("auth-main", finding.finding_id or "", status="accepted", actor="lead")
        await fabric.coordinate_promote_finding("auth-main", finding.finding_id or "", actor="lead")

        brief = await fabric.coordinate_brief("auth-main")

        assert isinstance(brief, CoordinationBrief)
        assert brief.accepted_findings[0].finding_id == finding.finding_id
        assert fabric.graph.upsert_extraction.await_count == 5

    async def test_coordinate_checkout_returns_accepted_state_only(self, tmp_path: Path) -> None:
        """MemoryFabric should expose accepted coordination checkout for prompt injection."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("auth-main", objective="Ship auth refactor", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-api", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-ui", actor="lead")
        finding = await fabric.coordinate_report_finding(
            "auth-main",
            "auth-api",
            summary="API failures trace to expired JWKS cache handling.",
            actor="auth-api-agent",
        )
        await fabric.coordinate_report_finding(
            "auth-main",
            "auth-ui",
            summary="UI refresh handling is missing retry state.",
            actor="auth-ui-agent",
        )
        await fabric.coordinate_review_finding("auth-main", finding.finding_id or "", status="accepted", actor="lead")
        await fabric.coordinate_promote_finding("auth-main", finding.finding_id or "", actor="lead")

        checkout = await fabric.coordinate_checkout("auth-main")

        assert checkout.accepted_findings[0].finding_id == finding.finding_id
        assert checkout.pending_findings == []
        assert checkout.excluded_pending_count == 1
        assert checkout.purpose["profile"] == "coordinate"
        assert "Purpose profile: coordinate" in checkout.prompt

    async def test_coordinate_brief_uses_configured_local_semantic_conflict_provider(
        self,
        tmp_path: Path,
    ) -> None:
        """MemoryFabric should pass the configured semantic adapter into coordination state."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.settings = Settings(_env_file=None, coordination_semantic_conflict_provider="lexical")
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("auth-main", objective="Ship auth refactor", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-api", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-ui", actor="lead")
        await fabric.coordinate_report_finding(
            "auth-main",
            "auth-api",
            summary="Token refresh retry is enabled in auth middleware.",
            actor="auth-api-agent",
        )
        await fabric.coordinate_report_finding(
            "auth-main",
            "auth-ui",
            summary="Token refresh retry is disabled in browser session handling.",
            actor="auth-ui-agent",
        )

        brief = await fabric.coordinate_brief("auth-main")

        assert brief.conflicts[0].conflict_type == "semantic"
        assert brief.conflicts[0].reason == "local_lexical_contradiction:disabled/enabled"

    async def test_coordinate_performance_ledger_returns_worker_metrics(self, tmp_path: Path) -> None:
        """MemoryFabric should expose replay-backed coordination worker metrics."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("auth-main", objective="Ship auth refactor", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-api", actor="lead")
        finding = await fabric.coordinate_report_finding(
            "auth-main",
            "auth-api",
            summary="API failures trace to expired JWKS cache handling.",
            actor="auth-api-agent",
            evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        )
        await fabric.coordinate_review_finding("auth-main", finding.finding_id or "", status="accepted", actor="lead")
        await fabric.coordinate_promote_finding("auth-main", finding.finding_id or "", actor="lead")

        ledger = await fabric.coordinate_performance_ledger("auth-main")

        assert ledger.worker("auth-api").accepted_findings == 1
        assert ledger.worker("auth-api").test_backed_findings == 1

    async def test_coordinate_create_handoff_projects_parent_event(self, tmp_path: Path) -> None:
        """MemoryFabric should expose final coordination handoff creation."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("auth-main", objective="Ship auth refactor", actor="lead")
        handoff = await fabric.coordinate_create_handoff(
            "auth-main",
            summary="Auth mission complete.",
            next_steps=["Release branch"],
            risks=["Token cache metrics are sparse"],
            actor="lead",
        )

        assert handoff.event.type == "coordination.handoff.created"
        assert handoff.event.payload["next_steps"] == ["Release branch"]
        assert handoff.event.payload["risks"] == ["Token cache metrics are sparse"]
        assert fabric.graph.upsert_extraction.await_count == 2

    async def test_coordinate_approval_packet_and_apply_decisions(self, tmp_path: Path) -> None:
        """MemoryFabric should expose remote approval packet and application helpers."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("auth-main", objective="Ship auth refactor", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-api", actor="lead")
        finding = await fabric.coordinate_report_finding(
            "auth-main",
            "auth-api",
            summary="Expired JWKS cache causes API failures.",
            actor="auth-api-agent",
            evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        )

        packet = await fabric.coordinate_approval_packet("auth-main")
        result = await fabric.coordinate_apply_approval_decisions(
            "auth-main",
            [{"finding_id": finding.finding_id, "status": "accepted", "rationale": "Command-backed.", "promote": True}],
            actor="reviewer",
        )

        assert packet.findings[0].finding_id == finding.finding_id
        assert result.reviewed_count == 1
        assert result.promoted_count == 1
        assert fabric.graph.upsert_extraction.await_count == 5

    async def test_coordinate_record_synthesis_artifact_writes_proof_packet(self, tmp_path: Path) -> None:
        """Coordinate synthesis should persist a mission-scoped proof packet."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("release-rc1", objective="Ship release", actor="lead")
        await fabric.coordinate_create_worker("release-rc1", "auth-api", actor="lead")
        finding = await fabric.coordinate_report_finding(
            "release-rc1",
            "auth-api",
            summary="Expired JWKS cache is the accepted auth failure cause.",
            actor="auth-api-agent",
            evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
            claim_key="auth.failure.cause",
            claim_value="expired-jwks-cache",
        )
        review = await fabric.coordinate_review_finding(
            "release-rc1",
            finding.finding_id or "",
            status="accepted",
            actor="lead",
            rationale="Command-backed.",
        )
        promotion = await fabric.coordinate_promote_finding("release-rc1", finding.finding_id or "", actor="lead")
        handoff = await fabric.coordinate_create_handoff(
            "release-rc1",
            summary="Release handoff ready.",
            actor="lead",
        )
        checkout = MemoryCheckout(
            session_id="release-rc1",
            query="Compose accepted release findings into the handoff answer.",
            prompt="# Memory Checkout",
            working_set={},
            ref=None,
            current_facts=[],
            evidence=[
                {
                    "content": "finding-api accepted the JWKS cache cause.",
                    "citation": f"eventloom://release-rc1/events/{promotion.event.seq}#{promotion.event.hash[:12]}",
                    "source_lane": "graph",
                }
            ],
            provenance=[],
            retention={},
            warnings=[],
            guidance={},
            quality={"answerability": "answer_from_memory", "confidence": 0.9},
            diagnostics={
                "synthesis": {
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "coordinate_handoff",
                            "answer": "Accepted cause: expired JWKS cache.",
                            "support_source_ids": [finding.finding_id],
                        }
                    ],
                    "ledger_rows": [
                        {
                            "fact_id": finding.finding_id,
                            "source_group": finding.finding_id,
                            "citation": f"eventloom://release-rc1/events/{promotion.event.seq}#{promotion.event.hash[:12]}",
                            "include_reason": "accepted_parent_state",
                        }
                    ],
                }
            },
            context_counts={},
            replay_event_count=0,
            compacted=False,
            assembly_policy={},
        )

        result = await fabric.coordinate_record_synthesis_artifact(
            "release-rc1",
            checkout,
            decision_scope="handoff",
            handoff_id=handoff.handoff_id,
            actor="coordinator",
        )

        assert result["artifact_event"]["event_type"] == "memory.synthesis.artifact.created"
        assert result["proof_event"]["event_type"] == "coordination.proof_packet.created"
        proof = result["proof_packet"]
        assert proof["authority_scope"] == "parent_accepted_state"
        assert proof["artifact_id"] == result["artifact_id"]
        assert proof["accepted_finding_ids"] == [finding.finding_id]
        assert proof["review_event_refs"] == [
            {"seq": review.event.seq, "hash": review.event.hash, "finding_id": finding.finding_id}
        ]
        assert proof["promotion_event_refs"] == [
            {"seq": promotion.event.seq, "hash": promotion.event.hash, "finding_id": finding.finding_id}
        ]
        assert proof["worker_source_event_refs"][0]["worker_id"] == "auth-api"
        assert proof["worker_source_event_refs"][0]["finding_id"] == finding.finding_id
        assert proof["handoff_event_ref"] == {
            "handoff_id": handoff.handoff_id,
            "seq": handoff.event.seq,
            "hash": handoff.event.hash,
        }
        assert proof["non_authoritative_rows"] == []
        proof_events = [
            event
            for event in fabric.session_manager.replay("release-rc1").events
            if event.type == "coordination.proof_packet.created"
        ]
        assert proof_events[-1].payload["handoff_event_ref"] == proof["handoff_event_ref"]

    async def test_coordinate_record_synthesis_artifact_rejects_unknown_handoff_without_appending(
        self,
        tmp_path: Path,
    ) -> None:
        """Invalid handoff-scoped proof calls should fail before writing audit events."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("release-rc1", objective="Ship release", actor="lead")
        checkout = MemoryCheckout(
            session_id="release-rc1",
            query="Compose accepted release findings into the handoff answer.",
            prompt="# Memory Checkout",
            working_set={},
            ref=None,
            current_facts=[],
            evidence=[],
            provenance=[],
            retention={},
            warnings=[],
            guidance={},
            quality={"answerability": "answer_from_memory", "confidence": 0.9},
            diagnostics={
                "synthesis": {
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "coordinate_handoff",
                            "answer": "No accepted findings.",
                            "support_source_ids": [],
                        }
                    ],
                    "ledger_rows": [],
                }
            },
            context_counts={},
            replay_event_count=0,
            compacted=False,
            assembly_policy={},
        )
        before = len(fabric.session_manager.replay("release-rc1").events)

        with pytest.raises(ValueError, match="Unknown handoff_id"):
            await fabric.coordinate_record_synthesis_artifact(
                "release-rc1",
                checkout,
                decision_scope="handoff",
                handoff_id="release-rc1:handoff:missing",
            )

        events = fabric.session_manager.replay("release-rc1").events
        assert len(events) == before
        assert not any(event.type == "memory.synthesis.artifact.created" for event in events)
        assert not any(event.type == "coordination.proof_packet.created" for event in events)

    async def test_coordinate_record_detected_conflicts_projects_source_state_conflict(
        self,
        tmp_path: Path,
    ) -> None:
        """MemoryFabric should materialize deterministic conflicts for graph projection."""
        fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
        fabric.graph = AsyncMock()
        fabric.tracer = AsyncMock()

        await fabric.coordinate_start_mission("auth-main", objective="Ship auth refactor", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-api", actor="lead")
        await fabric.coordinate_create_worker("auth-main", "auth-ui", actor="lead")
        await fabric.coordinate_report_finding(
            "auth-main",
            "auth-api",
            summary="API worker saw one auth config snapshot.",
            actor="auth-api-agent",
            evidence=[{"kind": "file", "reference": "src/auth/config.py", "source_sha256": "a" * 64}],
        )
        await fabric.coordinate_report_finding(
            "auth-main",
            "auth-ui",
            summary="UI worker saw another auth config snapshot.",
            actor="auth-ui-agent",
            evidence=[{"kind": "file", "reference": "src/auth/config.py", "source_sha256": "b" * 64}],
        )

        results = await fabric.coordinate_record_detected_conflicts("auth-main", actor="zaxy")

        assert len(results) == 1
        assert results[0].event.payload["conflict_type"] == "source_state"
        extraction = fabric.graph.upsert_extraction.await_args_list[-1].args[0]
        conflict = next(entity for entity in extraction.entities if entity.entity_type == "conflict")
        assert conflict.properties["source_reference"] == "src/auth/config.py"
        assert fabric.graph.upsert_extraction.await_count == 6
