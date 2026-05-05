"""Tests for zaxy.mcp_server — MCP protocol compliance."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import zaxy.mcp_server
from zaxy.mcp_server import TOOLS, ZaxyMCPServer, main

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
        """Should expose exactly 4 tools."""
        assert len(TOOLS) == 4

    def test_tool_names(self) -> None:
        """Tool names should match the expected contract."""
        names = {t.name for t in TOOLS}
        assert names == {"memory_append", "memory_query", "memory_replay", "memory_invalidate"}

    def test_memory_append_has_required_fields(self) -> None:
        """memory_append schema should require event_type, actor, payload."""
        tool = next(t for t in TOOLS if t.name == "memory_append")
        assert tool.inputSchema["required"] == ["event_type", "actor", "payload"]

    def test_memory_query_has_optional_temporal(self) -> None:
        """memory_query temporal_filter should be optional."""
        tool = next(t for t in TOOLS if t.name == "memory_query")
        assert "temporal_filter" in tool.inputSchema["properties"]
        assert "temporal_filter" not in (tool.inputSchema.get("required") or [])


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
                )
            ]
            mock_router_cls.return_value = mock_router

            result = await server.handle_memory_query({
                "query": "Alice",
                "limit": 5,
            })

            assert len(result) == 1
            data = result[0].text
            assert "Alice" in data
            assert "exact" in data

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

            call = mock_router.query.await_args
            assert call.kwargs["temporal_point"] == "2024-03-01T00:00:00Z"

    async def test_rejects_invalid_limit(self, server: ZaxyMCPServer) -> None:
        """Query limits should be bounded to prevent expensive fan-out."""
        with pytest.raises(ValueError, match="limit"):
            await server.handle_memory_query({"query": "x", "limit": 100000})

    async def test_rejects_long_query(self, server: ZaxyMCPServer) -> None:
        """Very large queries should be rejected before database work."""
        with pytest.raises(ValueError, match="query"):
            await server.handle_memory_query({"query": "x" * 4097})


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
            "OldFact", "fact", "2024-06-01T00:00:00Z"
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

        with patch("zaxy.mcp_server.app.run", new_callable=AsyncMock) as mock_run:
            with patch.object(
                zaxy.mcp_server.app, "list_tools", return_value=make_capture_decorator("list_tools")
            ), patch.object(
                zaxy.mcp_server.app, "call_tool", return_value=make_capture_decorator("call_tool")
            ):
                await main()

        assert "call_tool" in captured_handlers
        with pytest.raises(ValueError, match="Unknown tool"):
            await captured_handlers["call_tool"]("unknown_tool", {})


# ------------------------------------------------------------------
# Lifecycle tests
# ------------------------------------------------------------------

class TestLifecycle:
    """Tests for server setup/teardown."""

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
