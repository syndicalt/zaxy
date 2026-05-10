"""Tests for zaxy.mcp_server — MCP protocol compliance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import zaxy.mcp_server
from zaxy.event import EventLog
from zaxy.mcp_server import (
    TOOLS,
    MCPTransportAuth,
    ZaxyMCPServer,
    main,
    remote_session_scope,
)


def json_loads(value: str) -> Any:
    return json.loads(value)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def server() -> ZaxyMCPServer:
    """Return a server with mocked graph, tracer, and session manager."""
    with (
        patch("zaxy.mcp_server.GraphStore") as mock_graph_cls,
        patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
        patch("zaxy.mcp_server.SessionManager") as mock_session_cls,
    ):
        mock_graph = AsyncMock()
        mock_graph_cls.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        mock_log = MagicMock()
        mock_event = MagicMock(seq=1, hash="a" * 64)
        mock_log.append.return_value = mock_event
        mock_session_mgr = MagicMock()
        mock_session_mgr.get.return_value.eventlog = mock_log
        mock_session_cls.return_value = mock_session_mgr

        srv = ZaxyMCPServer()
        srv.graph = mock_graph
        srv.tracer = mock_tracer
        srv.session_manager = mock_session_mgr
        yield srv


# ------------------------------------------------------------------
# Tool schema tests
# ------------------------------------------------------------------

class TestToolSchema:
    """Tests for MCP tool definitions."""

    def test_tools_list_length(self) -> None:
        """Should expose the memory and context lifecycle tools."""
        assert len(TOOLS) == 11

    def test_tool_names(self) -> None:
        """Tool names should match the expected contract."""
        names = {t.name for t in TOOLS}
        assert names == {
            "memory_append",
            "memory_query",
            "memory_verbatim",
            "memory_feedback",
            "memory_replay",
            "memory_invalidate",
            "memory_capabilities",
            "memory_checkout",
            "context_assemble",
            "context_after_turn",
            "subagent_cleanup",
        }

    def test_memory_verbatim_has_query_and_limit(self) -> None:
        """memory_verbatim should expose source-recall retrieval."""
        tool = next(t for t in TOOLS if t.name == "memory_verbatim")

        assert tool.inputSchema["required"] == ["query"]
        assert "limit" in tool.inputSchema["properties"]
        assert "session_id" in tool.inputSchema["properties"]

    def test_memory_append_has_required_fields(self) -> None:
        """memory_append schema should require event_type, actor, payload."""
        tool = next(t for t in TOOLS if t.name == "memory_append")
        assert tool.inputSchema["required"] == ["event_type", "actor", "payload"]

    def test_memory_query_has_optional_temporal(self) -> None:
        """memory_query temporal_filter should be optional."""
        tool = next(t for t in TOOLS if t.name == "memory_query")
        assert "temporal_filter" in tool.inputSchema["properties"]
        assert "temporal_filter" not in (tool.inputSchema.get("required") or [])

    def test_memory_feedback_has_required_identity_and_feedback(self) -> None:
        """memory_feedback should require a target entity and feedback value."""
        tool = next(t for t in TOOLS if t.name == "memory_feedback")
        assert tool.inputSchema["required"] == ["entity_name", "entity_type", "feedback"]
        assert "importance" in tool.inputSchema["properties"]

    def test_context_after_turn_has_required_fields(self) -> None:
        """context_after_turn should require role and content."""
        tool = next(t for t in TOOLS if t.name == "context_after_turn")
        assert tool.inputSchema["required"] == ["role", "content"]

    def test_memory_checkout_has_query_schema(self) -> None:
        """memory_checkout should expose the prompt-ready memory checkout contract."""
        tool = next(t for t in TOOLS if t.name == "memory_checkout")
        assert tool.inputSchema["required"] == ["query"]
        assert "session_id" in tool.inputSchema["properties"]
        assert "ref" in tool.inputSchema["properties"]
        assert "max_recent_events" in tool.inputSchema["properties"]

    def test_memory_capabilities_has_optional_query_schema(self) -> None:
        """memory_capabilities should expose model-facing Zaxy usage guidance."""
        tool = next(t for t in TOOLS if t.name == "memory_capabilities")
        assert tool.inputSchema["required"] == []
        assert "session_id" in tool.inputSchema["properties"]
        assert "current_task" in tool.inputSchema["properties"]


# ------------------------------------------------------------------
# Handler tests
# ------------------------------------------------------------------

class TestMemoryAppend:
    """Tests for memory_append handler."""

    async def test_appends_event_and_projects(self, server: ZaxyMCPServer) -> None:
        """Should append to Eventloom, extract, upsert to graph, and trace."""
        result = await server.handle_memory_append({
            "event_type": "goal.created",
            "actor": "user",
            "payload": {"title": "Ship it"},
            "session_id": "session-1",
        })

        server.session_manager.get.assert_called_once_with("session-1")
        log = server.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        server.graph.upsert_extraction.assert_awaited_once()
        server.tracer.trace_append.assert_awaited_once()
        assert len(result) == 1
        assert "1" in result[0].text


class TestMemoryCapabilities:
    """Tests for the model-facing memory capability manifest."""

    async def test_returns_capability_manifest_for_default_session(
        self,
        server: ZaxyMCPServer,
        tmp_path: Path,
    ) -> None:
        """memory_capabilities should make Zaxy's ambient memory loop visible to the model."""
        eventloom = tmp_path / ".eventloom"
        EventLog(eventloom / "agent-1.jsonl").append(
            "task.completed",
            actor="codex",
            payload={"summary": "Manifest source."},
            thread="agent-1",
        )
        server._eventloom_path = str(eventloom)
        server._workspace_root = tmp_path

        result = await server.handle_memory_capabilities(
            {"session_id": "agent-1", "current_task": "smooth memory UX"}
        )

        payload = json_loads(result[0].text)
        assert payload["session_id"] == "agent-1"
        assert payload["current_task"] == "smooth memory UX"
        assert payload["recommended_next_call"]["tool"] == "memory_checkout"
        assert payload["ambient_loop"]["after_compaction_or_resume"]["tool"] == "memory_checkout"
        assert payload["status"]["eventloom"]["latest_seq"] == 1
        assert "memory_checkout" in payload["prompt"]

    async def test_uses_default_session(self, server: ZaxyMCPServer) -> None:
        """Missing session_id should default to 'default'."""
        await server.handle_memory_append({
            "event_type": "x",
            "actor": "y",
            "payload": {},
        })

        server.session_manager.get.assert_called_once_with("default")

    async def test_falls_back_to_thread(self, server: ZaxyMCPServer) -> None:
        """thread should be used as fallback when session_id is missing."""
        await server.handle_memory_append({
            "event_type": "x",
            "actor": "y",
            "payload": {},
            "thread": "legacy-thread",
        })

        server.session_manager.get.assert_called_once_with("legacy-thread")

    async def test_remote_scope_supplies_default_session(self, server: ZaxyMCPServer) -> None:
        """Remote SSE clients should default to their request session scope."""
        token = remote_session_scope.set("client-session")
        try:
            await server.handle_memory_append({
                "event_type": "x",
                "actor": "y",
                "payload": {},
            })
        finally:
            remote_session_scope.reset(token)

        server.session_manager.get.assert_called_once_with("client-session")

    async def test_remote_scope_rejects_cross_session_append(self, server: ZaxyMCPServer) -> None:
        """Remote SSE clients should not write outside their request session scope."""
        token = remote_session_scope.set("client-session")
        try:
            with pytest.raises(PermissionError, match="session scope"):
                await server.handle_memory_append({
                    "event_type": "x",
                    "actor": "y",
                    "payload": {},
                    "session_id": "other-session",
                })
        finally:
            remote_session_scope.reset(token)

        server.session_manager.get.assert_not_called()

    async def test_rejects_large_payload(self, server: ZaxyMCPServer) -> None:
        """Oversized payloads should be rejected before writing to Eventloom."""
        with pytest.raises(ValueError, match="payload"):
            await server.handle_memory_append({
                "event_type": "x",
                "actor": "y",
                "payload": {"blob": "x" * (1024 * 1024 + 1)},
            })

        server.session_manager.get.assert_not_called()


class TestMemoryQuery:
    """Tests for memory_query handler."""

    async def test_returns_context_chunks(self, server: ZaxyMCPServer) -> None:
        """Should return formatted context chunks."""
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            mock_router = AsyncMock()
            mock_router.query.return_value = [
                MagicMock(
                    content="Alice (user)",
                    source="exact",
                    score=1.0,
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    citation="eventloom://default/events/1#aaaaaaaaaaaa",
                    score_explanation={"source": "exact", "weighted_score": 1.0},
                )
            ]
            mock_router_cls.return_value = mock_router

            result = await server.handle_memory_query({
                "query": "Alice",
                "limit": 5,
            })

            mock_router_cls.assert_called_once_with(
                server.graph,
                session_id="default",
                retention_policy=server._retention_policy,
            )
            assert len(result) == 1
            data = result[0].text
            assert "Alice" in data
            assert "exact" in data
            assert "eventloom://default/events/1#aaaaaaaaaaaa" in data
            assert "score_explanation" in data
            assert "weighted_score" in data

    async def test_memory_verbatim_returns_eventloom_citations(
        self,
        tmp_path: Path,
    ) -> None:
        """memory_verbatim should return exact source chunks without graph retrieval."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        eventlog = server.session_manager.get("agent").eventlog
        event = eventlog.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": "docs/design.md",
                "start_line": 4,
                "end_line": 8,
                "content": "Git for agent memory needs verbatim source recall.",
            },
            thread="agent",
        )

        result = await server.handle_memory_verbatim(
            {"query": "source recall", "session_id": "agent", "limit": 1}
        )

        payload = json.loads(result[0].text)
        assert payload[0]["content"] == "Git for agent memory needs verbatim source recall."
        assert payload[0]["source"] == "verbatim"
        assert payload[0]["citation"] == f"eventloom://agent/events/{event.seq}#{event.hash}"
        assert payload[0]["metadata"]["source_path"] == "docs/design.md"

    async def test_passes_temporal_filter(self, server: ZaxyMCPServer) -> None:
        """temporal_filter should be forwarded to the router."""
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            mock_router = AsyncMock()
            mock_router.query.return_value = []
            mock_router_cls.return_value = mock_router

            await server.handle_memory_query({
                "query": "x",
                "temporal_filter": "2024-03-01T00:00:00Z",
            })

            mock_router_cls.assert_called_once_with(
                server.graph,
                session_id="default",
                retention_policy=server._retention_policy,
            )
            call = mock_router.query.await_args
            assert call.kwargs["temporal_point"] == "2024-03-01T00:00:00Z"

    async def test_remote_scope_passes_session_to_router(self, server: ZaxyMCPServer) -> None:
        """Remote SSE queries should search only within their request session scope."""
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            mock_router = AsyncMock()
            mock_router.query.return_value = []
            mock_router_cls.return_value = mock_router

            token = remote_session_scope.set("client-session")
            try:
                await server.handle_memory_query({"query": "x"})
            finally:
                remote_session_scope.reset(token)

            mock_router_cls.assert_called_once_with(
                server.graph,
                session_id="client-session",
                retention_policy=server._retention_policy,
            )

    async def test_remote_scope_rejects_cross_session_query(self, server: ZaxyMCPServer) -> None:
        """Remote SSE clients should not query another explicit session."""
        token = remote_session_scope.set("client-session")
        try:
            with pytest.raises(PermissionError, match="session scope"):
                await server.handle_memory_query({
                    "query": "x",
                    "session_id": "other-session",
                })
        finally:
            remote_session_scope.reset(token)

    async def test_rejects_invalid_limit(self, server: ZaxyMCPServer) -> None:
        """Query limits should be bounded to prevent expensive fan-out."""
        with pytest.raises(ValueError, match="limit"):
            await server.handle_memory_query({"query": "x", "limit": 100000})

    async def test_rejects_long_query(self, server: ZaxyMCPServer) -> None:
        """Very large queries should be rejected before database work."""
        with pytest.raises(ValueError, match="query"):
            await server.handle_memory_query({"query": "x" * 4097})


class TestMemoryFeedback:
    """Tests for memory_feedback handler."""

    async def test_positive_feedback_appends_reinforcement_event(self, server: ZaxyMCPServer) -> None:
        """Used context should reinforce the target memory entity."""
        result = await server.handle_memory_feedback({
            "entity_name": "Use retention metadata",
            "entity_type": "decision",
            "feedback": "used",
            "actor": "assistant",
            "session_id": "agent-1",
            "importance": 0.8,
            "query": "retention decisions",
            "citation": "eventloom://agent-1/events/1#abc",
            "score": 0.91,
        })

        log = server.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        call = log.append.call_args
        assert call.args == ("memory.reinforced",)
        assert call.kwargs["actor"] == "assistant"
        assert call.kwargs["thread"] == "agent-1"
        assert call.kwargs["payload"] == {
            "entity_name": "Use retention metadata",
            "entity_type": "decision",
            "query": "retention decisions",
            "source": "mcp",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/1#abc",
            "importance": 0.8,
        }
        server.graph.upsert_extraction.assert_awaited_once()
        server.tracer.trace_append.assert_awaited_once_with("memory.reinforced", "assistant", 1)
        assert json_loads(result[0].text)["event_type"] == "memory.reinforced"

    async def test_negative_feedback_appends_audit_event(self, server: ZaxyMCPServer) -> None:
        """Irrelevant context should be recorded without reinforcement metadata."""
        result = await server.handle_memory_feedback({
            "entity_name": "Stale note",
            "entity_type": "decision",
            "feedback": "irrelevant",
            "reason": "Superseded by later decision",
        })

        call = server.session_manager.get.return_value.eventlog.append.call_args
        assert call.args == ("memory.feedback",)
        assert call.kwargs["actor"] == "zaxy"
        assert call.kwargs["payload"]["feedback"] == "irrelevant"
        assert call.kwargs["payload"]["reason"] == "Superseded by later decision"
        assert "importance" not in call.kwargs["payload"]
        assert json_loads(result[0].text)["event_type"] == "memory.feedback"

    async def test_rejects_unknown_feedback(self, server: ZaxyMCPServer) -> None:
        """Feedback values should stay constrained to known retrieval outcomes."""
        with pytest.raises(ValueError, match="feedback"):
            await server.handle_memory_feedback({
                "entity_name": "x",
                "entity_type": "memory",
                "feedback": "maybe",
            })

        server.session_manager.get.assert_not_called()


class TestServerSetup:
    """Tests for MCP server startup orchestration."""

    async def test_setup_bootstraps_local_neo4j_before_graph_schema(self) -> None:
        """Local stdio startup should make its Neo4j dependency transparent."""
        with (
            patch("zaxy.mcp_server.GraphStore") as mock_graph_cls,
            patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.mcp_server.SessionManager"),
            patch("zaxy.mcp_server.LocalNeo4jRuntime") as mock_runtime_cls,
        ):
            mock_graph = AsyncMock()
            mock_graph_cls.return_value = mock_graph
            mock_tracer = AsyncMock()
            mock_tracer_cls.return_value = mock_tracer
            mock_runtime = MagicMock()
            mock_runtime_cls.return_value = mock_runtime

            srv = ZaxyMCPServer()
            await srv.setup()

        mock_runtime.ensure_available.assert_called_once()
        mock_graph.connect.assert_awaited_once()
        mock_graph.init_schema.assert_awaited_once()
        mock_tracer.connect.assert_awaited_once()

    async def test_setup_appends_workspace_genesis_once(self, tmp_path: Path) -> None:
        """setup() should bootstrap the default session with one workspace genesis event."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("# Rules\n\nUse pytest.\n", encoding="utf-8")
        eventlog = EventLog(tmp_path / "events.jsonl")
        mock_log = MagicMock()
        mock_log.read_all.return_value = []
        mock_log.append.side_effect = eventlog.append
        with (
            patch("zaxy.mcp_server.GraphStore") as mock_graph_cls,
            patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.mcp_server.SessionManager") as mock_session_cls,
            patch("zaxy.mcp_server.LocalNeo4jRuntime"),
        ):
            mock_graph = AsyncMock()
            mock_graph_cls.return_value = mock_graph
            mock_tracer = AsyncMock()
            mock_tracer_cls.return_value = mock_tracer
            mock_session_mgr = MagicMock()
            mock_session_mgr.get.return_value.eventlog = mock_log
            mock_session_cls.return_value = mock_session_mgr

            srv = ZaxyMCPServer(workspace_root=tmp_path)
            await srv.setup()
            await srv.setup()

        mock_session_mgr.get.assert_called_with("default")
        assert mock_log.append.call_count == 2
        assert mock_log.append.call_args_list[0].args == ("session.genesis",)
        assert mock_log.append.call_args_list[0].kwargs["payload"]["root"] == str(tmp_path.resolve())
        assert mock_log.append.call_args_list[1].args == ("workspace.instructions.discovered",)
        assert mock_log.append.call_args_list[1].kwargs["payload"]["summary"] == "Rules: Use pytest."
        assert mock_graph.upsert_extraction.await_count == 2


class TestContextLifecycleTools:
    """Tests for MCP context lifecycle handlers."""

    async def test_context_assemble_returns_prompt_and_contexts(self, server: ZaxyMCPServer) -> None:
        """context_assemble should combine replay with retrieved context."""
        event = MagicMock(
            seq=2,
            type="transcript.turn",
            actor="assistant",
            payload={"content": "Use MMR."},
        )
        replay = MagicMock(events=[event])
        server.session_manager.replay.return_value = replay
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            router = AsyncMock()
            router.query.return_value = [
                MagicMock(
                    content="MMR diversity (decision)",
                    source="keyword",
                    score=0.9,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation=None,
                )
            ]
            mock_router_cls.return_value = router

            result = await server.handle_context_assemble({
                "query": "retrieval decision",
                "session_id": "agent-1",
                "max_recent_events": 1,
            })

        output = json_loads(result[0].text)
        assert output["session_id"] == "agent-1"
        assert output["replay_event_count"] == 1
        assert "MMR diversity" in output["prompt"]

    async def test_memory_checkout_returns_current_facts_and_evidence(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """memory_checkout should package assembled context as a cited working state."""
        event = MagicMock(
            seq=2,
            type="decision.recorded",
            actor="assistant",
            payload={"decision": "Use memory checkout."},
            hash="c" * 64,
        )
        server.session_manager.replay.return_value = MagicMock(events=[event])
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            router = AsyncMock()
            router.query.return_value = [
                MagicMock(
                    content="A memory capture gap was recorded during benchmark debugging.",
                    source="keyword",
                    score=0.91,
                    valid_from="2026-05-10T06:42:06Z",
                    valid_to=None,
                    citation="eventloom://agent-1/events/1832#gap",
                    score_explanation=None,
                    entity_name="memory capture gap",
                    entity_type="event",
                ),
                MagicMock(
                    content="Memory checkout is the context contract.",
                    source="keyword",
                    score=0.8,
                    valid_from="2026-05-10T20:55:40Z",
                    valid_to=None,
                    citation="eventloom://agent-1/events/1882#checkout",
                    score_explanation=None,
                    entity_name="memory checkout",
                    entity_type="task",
                ),
            ]
            mock_router_cls.return_value = router

            result = await server.handle_memory_checkout({
                "query": "What context contract should the model use?",
                "session_id": "agent-1",
                "limit": 3,
            })

        output = json_loads(result[0].text)
        assert output["session_id"] == "agent-1"
        assert output["current_facts"][0]["content"] == "Memory checkout is the context contract."
        assert output["current_facts"][0]["citation"] == "eventloom://agent-1/events/1882#checkout"
        assert output["evidence"][0]["citation"] == "eventloom://agent-1/events/1882#checkout"
        assert output["provenance"][0]["event_seq"] == 1882
        assert "# Memory Checkout" in output["prompt"]
        assert all(fact["valid_to"] is None for fact in output["current_facts"])

    async def test_context_assemble_includes_verbatim_source_lane(
        self,
        tmp_path: Path,
    ) -> None:
        """context_assemble should include exact Eventloom source hits by default."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        event = server.session_manager.get("agent-1").eventlog.append(
            "transcript.turn",
            actor="assistant",
            payload={
                "source": "codex",
                "turn_index": 9,
                "role": "assistant",
                "content": "The audit trail uses identity-code-0042.",
            },
            thread="agent-1",
        )
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            router = AsyncMock()
            router.query.return_value = [
                MagicMock(
                    content="Graph summary of audit trail",
                    source="keyword",
                    score=0.7,
                    valid_from=None,
                    valid_to=None,
                    citation="eventloom://agent-1/events/1#graph",
                    score_explanation=None,
                ),
                MagicMock(
                    content="Lower-priority graph context",
                    source="traversal",
                    score=0.4,
                    valid_from=None,
                    valid_to=None,
                    citation="eventloom://agent-1/events/2#graph",
                    score_explanation=None,
                ),
            ]
            mock_router_cls.return_value = router

            result = await server.handle_context_assemble({
                "query": "identity-code-0042",
                "session_id": "agent-1",
                "limit": 2,
            })

        output = json_loads(result[0].text)
        assert [context["source"] for context in output["contexts"]] == ["keyword", "verbatim"]
        assert output["contexts"][0]["metadata"]["assembly_lane"] == "graph"
        assert output["contexts"][1]["metadata"]["assembly_lane"] == "verbatim"
        assert output["contexts"][1]["citation"] == f"eventloom://agent-1/events/{event.seq}#{event.hash}"
        assert "identity-code-0042" in output["prompt"]
        assert output["assembly_policy"] == {
            "packet_memory_enabled": True,
            "packet_memory_slots": 1,
            "verbatim_enabled": True,
            "verbatim_slots": 1,
        }
        assert output["context_counts"] == {
            "graph": 1,
            "packet_memory": 0,
            "replay": 1,
            "verbatim": 1,
        }
        assert output["working_set"]["items"][0]["category"] == "source_anchor"
        assert "# Active Memory Working Set" in output["prompt"]

    async def test_context_assemble_uses_configured_default_session(self, server: ZaxyMCPServer) -> None:
        """Omitted session_id should use the configured domain-separated default."""
        server._default_session_id = "zaxy-default"
        server.session_manager.replay.return_value = MagicMock(events=[])
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            router = AsyncMock()
            router.query.return_value = []
            mock_router_cls.return_value = router

            result = await server.handle_context_assemble({"query": "retrieval decision"})

        output = json_loads(result[0].text)
        assert output["session_id"] == "zaxy-default"
        server.session_manager.replay.assert_called_with("zaxy-default", from_seq=1)

    async def test_context_after_turn_appends_and_assembles(self, server: ZaxyMCPServer) -> None:
        """context_after_turn should persist the latest turn before assembly."""
        server.session_manager.replay.return_value = MagicMock(events=[])
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            router = AsyncMock()
            router.query.return_value = []
            mock_router_cls.return_value = router

            result = await server.handle_context_after_turn({
                "role": "assistant",
                "content": "Use lifecycle hooks.",
                "session_id": "agent-1",
            })

        log = server.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        assert log.append.call_args.args == ("transcript.turn",)
        output = json_loads(result[0].text)
        assert output["session_id"] == "agent-1"

    async def test_subagent_cleanup_appends_cleanup_event(self, server: ZaxyMCPServer) -> None:
        """subagent_cleanup should finalize the subagent session with a cleanup event."""
        server.session_manager.handoff_summary.return_value = {"event_count": 3}
        replay = MagicMock(events=[], integrity=MagicMock(ok=True))
        server.session_manager.replay.return_value = replay
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            router = AsyncMock()
            router.query.return_value = []
            mock_router_cls.return_value = router

            result = await server.handle_subagent_cleanup({
                "parent_session_id": "main",
                "subagent_session_id": "worker-1",
                "summary": "Worker finished.",
            })

        log = server.session_manager.get.return_value.eventlog
        appended_types = [call.args[0] for call in log.append.call_args_list]
        assert appended_types == ["subagent.cleaned", "subagent.completed"]
        assert log.append.call_args_list[0].kwargs["payload"]["parent_session_id"] == "main"
        assert log.append.call_args_list[1].kwargs["payload"]["status"] == "succeeded"
        output = json_loads(result[0].text)
        assert output["summary"]["event_count"] == 3


class TestMemoryReplay:
    """Tests for memory_replay handler."""

    async def test_replays_from_seq(self, server: ZaxyMCPServer) -> None:
        """Should replay events from the given sequence."""
        mock_replay = MagicMock()
        mock_replay.integrity.model_dump.return_value = {"ok": True, "total_events": 2}
        mock_replay.events = []
        server.session_manager.replay.return_value = mock_replay

        result = await server.handle_memory_replay({
            "session_id": "session-1",
            "from_seq": 5,
        })

        server.session_manager.replay.assert_called_once_with("session-1", from_seq=5)
        assert "ok" in result[0].text

    async def test_default_from_seq(self, server: ZaxyMCPServer) -> None:
        """Missing from_seq should default to 1."""
        mock_replay = MagicMock()
        mock_replay.integrity.model_dump.return_value = {"ok": True, "total_events": 0}
        mock_replay.events = []
        server.session_manager.replay.return_value = mock_replay

        await server.handle_memory_replay({"session_id": "s1"})
        server.session_manager.replay.assert_called_once_with("s1", from_seq=1)

    async def test_rejects_invalid_from_seq(self, server: ZaxyMCPServer) -> None:
        """Replay from_seq should be positive."""
        with pytest.raises(ValueError, match="from_seq"):
            await server.handle_memory_replay({"session_id": "s1", "from_seq": 0})

        server.session_manager.replay.assert_not_called()

    async def test_remote_scope_rejects_cross_session_replay(self, server: ZaxyMCPServer) -> None:
        """Remote SSE clients should not replay sessions outside their scope."""
        mock_replay = MagicMock()
        mock_replay.integrity.model_dump.return_value = {"ok": True, "total_events": 0}
        mock_replay.events = []
        server.session_manager.replay.return_value = mock_replay

        token = remote_session_scope.set("client-session")
        try:
            with pytest.raises(PermissionError, match="session scope"):
                await server.handle_memory_replay({"session_id": "other-session"})
        finally:
            remote_session_scope.reset(token)

        server.session_manager.replay.assert_not_called()


class TestTransportAuth:
    """Tests for remote MCP/SSE request authentication."""

    def test_dev_without_token_allows_request_and_validates_session(self) -> None:
        """Development mode remains usable without configuring remote auth."""
        auth = MCPTransportAuth(token=None)

        session_id = auth.authorize({"x-zaxy-session-id": "agent-1"})

        assert session_id == "agent-1"

    def test_configured_token_rejects_missing_authorization(self) -> None:
        """Configured remote auth should require an Authorization header."""
        auth = MCPTransportAuth(token="secret")

        with pytest.raises(PermissionError, match="Authorization"):
            auth.authorize({"x-zaxy-session-id": "agent-1"})

    def test_configured_token_rejects_wrong_bearer(self) -> None:
        """Bearer token mismatch should reject the request."""
        auth = MCPTransportAuth(token="secret")

        with pytest.raises(PermissionError, match="Authorization"):
            auth.authorize({
                "authorization": "Bearer wrong",
                "x-zaxy-session-id": "agent-1",
            })

    def test_configured_token_accepts_bearer_and_session(self) -> None:
        """A valid bearer token should return the request session scope."""
        auth = MCPTransportAuth(token="secret")

        session_id = auth.authorize({
            "authorization": "Bearer secret",
            "x-zaxy-session-id": "agent-1",
        })

        assert session_id == "agent-1"

    def test_rejects_invalid_session_header(self) -> None:
        """Remote session scope should use the same session validation as tools."""
        auth = MCPTransportAuth(token=None)

        with pytest.raises(ValueError, match="session_id"):
            auth.authorize({"x-zaxy-session-id": "../escape"})

    def test_remote_request_guard_allows_and_audits_request(self, tmp_path: Path) -> None:
        """Remote request guard should authorize, rate-limit, and audit allowed requests."""
        from zaxy.mcp_server import RemoteRequestGuard

        audit_path = tmp_path / "audit.jsonl"
        guard = RemoteRequestGuard(
            auth=MCPTransportAuth(token="secret"),
            rate_limit_enabled=True,
            rate_limit_requests=2,
            rate_limit_window_seconds=60,
            audit_enabled=True,
            audit_path=audit_path,
        )

        session_id = guard.authorize(
            {
                "authorization": "Bearer secret",
                "x-zaxy-session-id": "tenant-1",
            },
            route="/messages/",
            method="POST",
            client_host="127.0.0.1",
        )

        assert session_id == "tenant-1"
        assert '"outcome":"allowed"' in audit_path.read_text(encoding="utf-8")

    def test_remote_request_guard_denies_after_session_limit(self, tmp_path: Path) -> None:
        """Remote request guard should return a rate-limit error for excess session traffic."""
        from zaxy.mcp_server import RemoteRateLimitError, RemoteRequestGuard

        audit_path = tmp_path / "audit.jsonl"
        guard = RemoteRequestGuard(
            auth=MCPTransportAuth(token=None),
            rate_limit_enabled=True,
            rate_limit_requests=1,
            rate_limit_window_seconds=60,
            audit_enabled=True,
            audit_path=audit_path,
        )

        guard.authorize(
            {"x-zaxy-session-id": "tenant-1"},
            route="/messages/",
            method="POST",
            client_host=None,
        )
        with pytest.raises(RemoteRateLimitError) as exc:
            guard.authorize(
                {"x-zaxy-session-id": "tenant-1"},
                route="/messages/",
                method="POST",
                client_host=None,
            )

        assert exc.value.retry_after_seconds == 60
        assert '"outcome":"denied_rate_limit"' in audit_path.read_text(encoding="utf-8")

    def test_remote_request_guard_audits_auth_denial(self, tmp_path: Path) -> None:
        """Remote request guard should audit authentication failures without secrets."""
        from zaxy.mcp_server import RemoteRequestGuard

        audit_path = tmp_path / "audit.jsonl"
        guard = RemoteRequestGuard(
            auth=MCPTransportAuth(token="secret"),
            rate_limit_enabled=True,
            rate_limit_requests=2,
            rate_limit_window_seconds=60,
            audit_enabled=True,
            audit_path=audit_path,
        )

        with pytest.raises(PermissionError):
            guard.authorize(
                {"authorization": "Bearer wrong", "x-zaxy-session-id": "tenant-1"},
                route="/sse",
                method="GET",
                client_host="127.0.0.1",
            )

        text = audit_path.read_text(encoding="utf-8")
        assert '"outcome":"denied_auth"' in text
        assert "wrong" not in text

    def test_oidc_accepts_valid_token_scope_and_session_claim(self) -> None:
        """OIDC mode should validate JWT claims and scope the request from the token."""
        captured: dict[str, Any] = {}

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, token: str) -> Any:
                captured["token"] = token
                return MagicMock(key="public-key")

        def fake_decode(token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
            captured["decode"] = {"token": token, "key": key, **kwargs}
            return {
                "iss": "https://idp.example",
                "aud": "zaxy",
                "scope": "profile zaxy:mcp",
                "zaxy_session": "tenant-1",
            }

        auth = MCPTransportAuth(
            token=None,
            oidc_issuer="https://idp.example",
            oidc_audience="zaxy",
            oidc_jwks_url="https://idp.example/.well-known/jwks.json",
            oidc_required_scope="zaxy:mcp",
            jwt_client=FakeJwksClient(),
            jwt_decoder=fake_decode,
        )

        session_id = auth.authorize({"authorization": "Bearer oidc-token"})

        assert session_id == "tenant-1"
        assert captured["token"] == "oidc-token"
        assert captured["decode"]["audience"] == "zaxy"
        assert captured["decode"]["issuer"] == "https://idp.example"

    def test_oidc_rejects_missing_required_scope(self) -> None:
        """OIDC mode should reject tokens that lack the configured MCP scope."""

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, token: str) -> Any:
                return MagicMock(key="public-key")

        def fake_decode(token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "iss": "https://idp.example",
                "aud": "zaxy",
                "scope": "profile",
                "zaxy_session": "tenant-1",
            }

        auth = MCPTransportAuth(
            token=None,
            oidc_issuer="https://idp.example",
            oidc_audience="zaxy",
            oidc_jwks_url="https://idp.example/.well-known/jwks.json",
            oidc_required_scope="zaxy:mcp",
            jwt_client=FakeJwksClient(),
            jwt_decoder=fake_decode,
        )

        with pytest.raises(PermissionError, match="scope"):
            auth.authorize({"authorization": "Bearer oidc-token"})

    def test_oidc_rejects_missing_session_claim(self) -> None:
        """OIDC mode should require a tenant/session claim for scoping."""

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, token: str) -> Any:
                return MagicMock(key="public-key")

        def fake_decode(token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "iss": "https://idp.example",
                "aud": "zaxy",
                "scope": "zaxy:mcp",
            }

        auth = MCPTransportAuth(
            token=None,
            oidc_issuer="https://idp.example",
            oidc_audience="zaxy",
            oidc_jwks_url="https://idp.example/.well-known/jwks.json",
            jwt_client=FakeJwksClient(),
            jwt_decoder=fake_decode,
        )

        with pytest.raises(PermissionError, match="session claim"):
            auth.authorize({"authorization": "Bearer oidc-token"})


class TestMemoryInvalidate:
    """Tests for memory_invalidate handler."""

    async def test_invalidates_entity(self, server: ZaxyMCPServer) -> None:
        """Should call graph.invalidate_entity with correct args."""
        result = await server.handle_memory_invalidate({
            "entity_name": "OldFact",
            "entity_type": "fact",
            "invalid_at": "2024-06-01T00:00:00Z",
        })

        server.graph.invalidate_entity.assert_awaited_once_with(
            "OldFact", "fact", "2024-06-01T00:00:00Z", session_id="default"
        )
        assert "invalidated" in result[0].text


# ------------------------------------------------------------------
# Entrypoint tests
# ------------------------------------------------------------------

class TestEntrypoint:
    """Tests for the MCP stdio server main() function."""

    @patch("zaxy.mcp_server.stdio_server")
    @patch("zaxy.mcp_server.GraphStore")
    @patch("zaxy.mcp_server.MemoryTracer")
    async def test_main_setup_and_teardown(
        self,
        mock_tracer_cls: MagicMock,
        mock_graph_cls: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """main() should setup server, register handlers, and teardown on exit."""
        mock_graph = AsyncMock()
        mock_graph_cls.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("zaxy.mcp_server.app.run", new_callable=AsyncMock) as mock_run:
            await main()

        mock_graph.connect.assert_awaited_once()
        mock_graph.init_schema.assert_awaited_once()
        mock_tracer.connect.assert_awaited_once()
        mock_run.assert_awaited_once()
        mock_graph.close.assert_awaited_once()
        mock_tracer.close.assert_awaited_once()

    @patch("zaxy.mcp_server.ZaxyMCPServer")
    @patch("zaxy.mcp_server.stdio_server")
    async def test_unknown_tool_raises_value_error(
        self,
        mock_stdio: MagicMock,
        mock_server_cls: MagicMock,
    ) -> None:
        """Calling an unknown tool should raise ValueError."""
        mock_server = AsyncMock()
        mock_server_cls.return_value = mock_server

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        captured_handlers: dict[str, Any] = {}

        def make_capture_decorator(name: str) -> Any:
            def decorator(fn: Any) -> Any:
                captured_handlers[name] = fn
                return fn
            return decorator

        with (
            patch("zaxy.mcp_server.app.run", new_callable=AsyncMock),
            patch.object(
                zaxy.mcp_server.app,
                "list_tools",
                return_value=make_capture_decorator("list_tools"),
            ),
            patch.object(
                zaxy.mcp_server.app,
                "call_tool",
                return_value=make_capture_decorator("call_tool"),
            ),
        ):
            await main()

        assert "call_tool" in captured_handlers
        with pytest.raises(ValueError, match="Unknown tool"):
            await captured_handlers["call_tool"]("unknown_tool", {})

    @patch("zaxy.mcp_server.ZaxyMCPServer")
    @patch("zaxy.mcp_server.stdio_server")
    async def test_call_tool_records_lifecycle_capture_without_raw_arguments(
        self,
        mock_stdio: MagicMock,
        mock_server_cls: MagicMock,
    ) -> None:
        """The MCP dispatcher should record a redacted tool.call.completed event."""
        mock_server = AsyncMock()
        mock_server.handle_memory_query.return_value = [MagicMock(text="[]")]
        mock_server.capture_tool_call_completed = AsyncMock()
        mock_server._default_session_id = "default"
        mock_server._session_id_from_arguments = MagicMock(return_value="agent-1")
        mock_server_cls.return_value = mock_server

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        captured_handlers: dict[str, Any] = {}

        def make_capture_decorator(name: str) -> Any:
            def decorator(fn: Any) -> Any:
                captured_handlers[name] = fn
                return fn
            return decorator

        with (
            patch("zaxy.mcp_server.app.run", new_callable=AsyncMock),
            patch.object(
                zaxy.mcp_server.app,
                "list_tools",
                return_value=make_capture_decorator("list_tools"),
            ),
            patch.object(
                zaxy.mcp_server.app,
                "call_tool",
                return_value=make_capture_decorator("call_tool"),
            ),
        ):
            await main()

        await captured_handlers["call_tool"](
            "memory_query",
            {
                "query": "roadmap",
                "session_id": "agent-1",
                "api_key": "secret",
            },
        )

        mock_server.capture_tool_call_completed.assert_awaited_once()
        call = mock_server.capture_tool_call_completed.await_args
        assert call.kwargs["tool_name"] == "memory_query"
        assert call.kwargs["status"] == "succeeded"
        assert call.kwargs["session_id"] == "agent-1"
        assert call.kwargs["arguments"] == {
            "query": "roadmap",
            "session_id": "agent-1",
            "api_key": "secret",
        }
        assert "secret" not in call.kwargs["result_summary"]

    @patch("zaxy.mcp_server.ZaxyMCPServer")
    @patch("zaxy.mcp_server.stdio_server")
    async def test_call_tool_records_failed_lifecycle_capture(
        self,
        mock_stdio: MagicMock,
        mock_server_cls: MagicMock,
    ) -> None:
        """Failed MCP dispatch should record a failed lifecycle event and re-raise."""
        mock_server = AsyncMock()
        mock_server.capture_tool_call_completed = AsyncMock()
        mock_server._default_session_id = "default"
        mock_server._session_id_from_arguments = MagicMock(return_value="agent-1")
        mock_server_cls.return_value = mock_server

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        captured_handlers: dict[str, Any] = {}

        def make_capture_decorator(name: str) -> Any:
            def decorator(fn: Any) -> Any:
                captured_handlers[name] = fn
                return fn
            return decorator

        with (
            patch("zaxy.mcp_server.app.run", new_callable=AsyncMock),
            patch.object(
                zaxy.mcp_server.app,
                "list_tools",
                return_value=make_capture_decorator("list_tools"),
            ),
            patch.object(
                zaxy.mcp_server.app,
                "call_tool",
                return_value=make_capture_decorator("call_tool"),
            ),
        ):
            await main()

        with pytest.raises(ValueError, match="Unknown tool"):
            await captured_handlers["call_tool"](
                "unknown_tool",
                {"session_id": "agent-1", "api_key": "secret"},
            )

        mock_server.capture_tool_call_completed.assert_awaited_once()
        call = mock_server.capture_tool_call_completed.await_args
        assert call.kwargs["tool_name"] == "unknown_tool"
        assert call.kwargs["status"] == "failed"
        assert call.kwargs["session_id"] == "agent-1"
        assert call.kwargs["result_summary"] is None

    @patch("zaxy.mcp_server.ZaxyMCPServer")
    @patch("zaxy.mcp_server.stdio_server")
    async def test_call_tool_skips_lifecycle_capture_when_disabled(
        self,
        mock_stdio: MagicMock,
        mock_server_cls: MagicMock,
    ) -> None:
        """The dispatcher should honor disabled lifecycle capture config."""
        mock_server = AsyncMock()
        mock_server.handle_memory_query.return_value = [MagicMock(text="[]")]
        mock_server.capture_tool_call_completed = AsyncMock()
        mock_server._default_session_id = "default"
        mock_server._lifecycle_capture_enabled = False
        mock_server._session_id_from_arguments = MagicMock(return_value="agent-1")
        mock_server_cls.return_value = mock_server

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        captured_handlers: dict[str, Any] = {}

        def make_capture_decorator(name: str) -> Any:
            def decorator(fn: Any) -> Any:
                captured_handlers[name] = fn
                return fn
            return decorator

        with (
            patch("zaxy.mcp_server.app.run", new_callable=AsyncMock),
            patch.object(
                zaxy.mcp_server.app,
                "list_tools",
                return_value=make_capture_decorator("list_tools"),
            ),
            patch.object(
                zaxy.mcp_server.app,
                "call_tool",
                return_value=make_capture_decorator("call_tool"),
            ),
        ):
            await main()

        await captured_handlers["call_tool"](
            "memory_query",
            {"query": "roadmap", "session_id": "agent-1"},
        )

        mock_server.capture_tool_call_completed.assert_not_awaited()


# ------------------------------------------------------------------
# Lifecycle tests
# ------------------------------------------------------------------

class TestLifecycle:
    """Tests for server setup/teardown."""

    async def test_capture_tool_call_completed_appends_redacted_event(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """capture_tool_call_completed() should append and project redacted lifecycle metadata."""
        await server.capture_tool_call_completed(
            tool_name="memory_query",
            status="succeeded",
            session_id="agent-1",
            arguments={"query": "roadmap", "api_key": "secret"},
            result_summary="1 result",
        )

        log = server.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        assert log.append.call_args.args == ("tool.call.completed",)
        payload = log.append.call_args.kwargs["payload"]
        assert payload["tool_name"] == "memory_query"
        assert payload["argument_keys"] == ["api_key", "query"]
        assert payload["arguments_redacted"] is True
        assert "api_key" in payload["argument_keys"]
        assert "secret" not in str(payload)
        server.graph.upsert_extraction.assert_awaited_once()

    async def test_teardown_records_session_end_when_lifecycle_capture_enabled(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """teardown should record a best-effort session.ended lifecycle event."""
        await server.teardown()

        log = server.session_manager.get.return_value.eventlog
        assert log.append.call_args.args == ("session.ended",)
        assert log.append.call_args.kwargs["payload"]["reason"] == "teardown"

    @patch("zaxy.mcp_server.GraphStore")
    @patch("zaxy.mcp_server.MemoryTracer")
    @patch("zaxy.mcp_server.SessionManager")
    async def test_setup_connects_all(
        self,
        mock_session_cls: MagicMock,
        mock_tracer_cls: MagicMock,
        mock_graph_cls: MagicMock,
    ) -> None:
        """setup() should connect graph and tracer."""
        mock_graph = AsyncMock()
        mock_graph_cls.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        srv = ZaxyMCPServer()
        await srv.setup()
        mock_graph.connect.assert_awaited_once()
        mock_graph.init_schema.assert_awaited_once()
        mock_tracer.connect.assert_awaited_once()

    @patch("zaxy.mcp_server.GraphStore")
    @patch("zaxy.mcp_server.MemoryTracer")
    @patch("zaxy.mcp_server.SessionManager")
    async def test_teardown_closes_all(
        self,
        mock_session_cls: MagicMock,
        mock_tracer_cls: MagicMock,
        mock_graph_cls: MagicMock,
    ) -> None:
        """teardown() should close graph and tracer."""
        mock_graph = AsyncMock()
        mock_graph_cls.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        srv = ZaxyMCPServer()
        await srv.setup()
        await srv.teardown()
        mock_graph.close.assert_awaited_once()
        mock_tracer.close.assert_awaited_once()


# ------------------------------------------------------------------
# SSE transport tests
# ------------------------------------------------------------------

class TestSSEEntrypoint:
    """Tests for the MCP SSE server main_sse() function."""

    @patch("uvicorn.Server")
    @patch("zaxy.mcp_server.GraphStore")
    @patch("zaxy.mcp_server.MemoryTracer")
    @patch("zaxy.mcp_server.SessionManager")
    async def test_main_sse_setup_and_teardown(
        self,
        mock_session_cls: MagicMock,
        mock_tracer_cls: MagicMock,
        mock_graph_cls: MagicMock,
        mock_uvicorn_cls: MagicMock,
    ) -> None:
        """main_sse() should setup server, run uvicorn, and teardown."""
        mock_graph = AsyncMock()
        mock_graph_cls.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        mock_uvicorn_server = AsyncMock()
        mock_uvicorn_cls.return_value = mock_uvicorn_server

        from zaxy.mcp_server import main_sse

        await main_sse(port=9999)

        mock_graph.connect.assert_awaited_once()
        mock_graph.init_schema.assert_awaited_once()
        mock_tracer.connect.assert_awaited_once()
        mock_uvicorn_server.serve.assert_awaited_once()
        mock_graph.close.assert_awaited_once()
        mock_tracer.close.assert_awaited_once()
