"""Tests for zaxy.trace — Pathlight observability hooks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.event import EventLog
from zaxy.trace import MemoryTracer, build_trace_correlation

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def tracer() -> MemoryTracer:
    """Return a tracer with a mock Pathlight dependency."""
    with patch("zaxy.trace._HAS_PATHLIGHT", True):
        yield MemoryTracer(disabled=False)


@pytest.fixture
def mock_pathlight() -> MagicMock:
    """Return a mock Pathlight client class."""
    return MagicMock()


# ------------------------------------------------------------------
# Lifecycle tests
# ------------------------------------------------------------------

class TestLifecycle:
    """Tests for connect/close behavior."""

    @patch("zaxy.trace.AsyncPathlight")
    async def test_connect_creates_client(self, mock_cls: MagicMock) -> None:
        """connect() should instantiate AsyncPathlight."""
        with patch("zaxy.trace._HAS_PATHLIGHT", True):
            t = MemoryTracer(base_url="http://x:4100", project_id="p1")
            await t.connect()
            mock_cls.assert_called_once_with(base_url="http://x:4100", project_id="p1")

    async def test_disabled_tracer_is_noop(self) -> None:
        """A disabled tracer should do nothing on connect/close."""
        t = MemoryTracer(disabled=True)
        await t.connect()
        assert t._client is None
        await t.close()

    async def test_close_clears_client(self, tracer: MemoryTracer) -> None:
        """close() should clear the client reference."""
        tracer._client = AsyncMock()
        await tracer.close()
        assert tracer._client is None


# ------------------------------------------------------------------
# Span context manager tests
# ------------------------------------------------------------------

class TestSpan:
    """Tests for the span context manager."""

    @patch("zaxy.trace.AsyncPathlight")
    async def test_span_uses_keyword_metadata_for_pathlight_trace(self, mock_cls: MagicMock) -> None:
        """trace() metadata should be passed as a keyword for current Pathlight SDKs."""
        with patch("zaxy.trace._HAS_PATHLIGHT", True):
            t = MemoryTracer()
            mock_trace = MagicMock()
            mock_trace.end = AsyncMock()
            mock_span = AsyncMock()
            mock_trace.span.return_value = mock_span

            def trace(
                name: str,
                *,
                input: object | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, object] | None = None,
            ) -> MagicMock:
                return mock_trace

            mock_cls.return_value.trace.side_effect = trace
            await t.connect()

            async with t.span("test_op", metadata={"k": "v"}):
                pass

            mock_cls.return_value.trace.assert_called_once_with(
                "zaxy-memory",
                metadata={"k": "v"},
            )

    @patch("zaxy.trace.AsyncPathlight")
    async def test_span_uses_async_pathlight_api(self, mock_cls: MagicMock) -> None:
        """Current Pathlight async clients return trace and span objects from coroutines."""
        with patch("zaxy.trace._HAS_PATHLIGHT", True):
            t = MemoryTracer()
            mock_trace = MagicMock()
            mock_trace.end = AsyncMock()
            mock_span = MagicMock()
            mock_span.end = AsyncMock()
            mock_trace.span = AsyncMock(return_value=mock_span)
            mock_cls.return_value.trace = AsyncMock(return_value=mock_trace)
            await t.connect()

            async with t.span("test_op", span_type="tool", metadata={"k": "v"}) as result:
                result["output"] = {"ok": True}

            mock_cls.return_value.trace.assert_awaited_once_with(
                "zaxy-memory",
                metadata={"k": "v"},
            )
            mock_trace.span.assert_awaited_once_with(
                "test_op",
                type="tool",
                input={"k": "v"},
            )
            mock_span.end.assert_awaited_once_with(output={"ok": True}, error=None)
            mock_trace.end.assert_awaited_once()

    @patch("zaxy.trace.AsyncPathlight")
    async def test_span_yields_result_dict(self, mock_cls: MagicMock) -> None:
        """The context manager should yield a mutable result dict."""
        with patch("zaxy.trace._HAS_PATHLIGHT", True):
            t = MemoryTracer()
            mock_trace = MagicMock()
            mock_trace.end = AsyncMock()
            mock_span = AsyncMock()
            mock_trace.span.return_value = mock_span
            mock_cls.return_value.trace.return_value = mock_trace
            await t.connect()

            async with t.span("test_op") as result:
                result["output"] = {"x": 1}

            mock_span.end.assert_awaited_once()
            args = mock_span.end.await_args.kwargs
            assert args["output"] == {"x": 1}

    @patch("zaxy.trace.AsyncPathlight")
    async def test_span_captures_exception(self, mock_cls: MagicMock) -> None:
        """Exceptions inside the span should be recorded and re-raised."""
        with patch("zaxy.trace._HAS_PATHLIGHT", True):
            t = MemoryTracer()
            mock_trace = MagicMock()
            mock_trace.end = AsyncMock()
            mock_span = AsyncMock()
            mock_trace.span.return_value = mock_span
            mock_cls.return_value.trace.return_value = mock_trace
            await t.connect()

            with pytest.raises(ValueError):
                async with t.span("fail") as _result:
                    raise ValueError("boom")

            args = mock_span.end.await_args.kwargs
            assert "boom" in args["error"]

    async def test_disabled_span_yields_empty_dict(self) -> None:
        """When disabled, span should yield an empty dict and skip tracing."""
        t = MemoryTracer(disabled=True)
        async with t.span("x") as result:
            result["output"] = 1
        # Should not raise


# ------------------------------------------------------------------
# Fire-and-forget trace helpers
# ------------------------------------------------------------------

class TestTraceHelpers:
    """Tests for trace_append and trace_query."""

    @patch("zaxy.trace.AsyncPathlight")
    async def test_trace_append(self, mock_cls: MagicMock) -> None:
        """trace_append should emit a span with event metadata."""
        with patch("zaxy.trace._HAS_PATHLIGHT", True):
            t = MemoryTracer()
            mock_trace = MagicMock()
            mock_trace.end = AsyncMock()
            mock_span = AsyncMock()
            mock_trace.span.return_value = mock_span
            mock_cls.return_value.trace.return_value = mock_trace
            await t.connect()

            await t.trace_append("goal.created", "user", seq=5)
            mock_trace.span.assert_called_once()
            assert mock_trace.span.call_args.kwargs["input"]["event_type"] == "goal.created"
            mock_span.end.assert_awaited_once()

    @patch("zaxy.trace.AsyncPathlight")
    async def test_trace_append_with_error(self, mock_cls: MagicMock) -> None:
        """trace_append should record error when provided."""
        with patch("zaxy.trace._HAS_PATHLIGHT", True):
            t = MemoryTracer()
            mock_trace = MagicMock()
            mock_trace.end = AsyncMock()
            mock_span = AsyncMock()
            mock_trace.span.return_value = mock_span
            mock_cls.return_value.trace.return_value = mock_trace
            await t.connect()

            await t.trace_append("goal.created", "user", seq=5, success=False, error="connection refused")
            args = mock_span.end.await_args.kwargs
            assert args["error"] == "connection refused"

    @patch("zaxy.trace.AsyncPathlight")
    async def test_trace_query(self, mock_cls: MagicMock) -> None:
        """trace_query should emit a span with query metadata."""
        with patch("zaxy.trace._HAS_PATHLIGHT", True):
            t = MemoryTracer()
            mock_trace = MagicMock()
            mock_trace.end = AsyncMock()
            mock_span = AsyncMock()
            mock_trace.span.return_value = mock_span
            mock_cls.return_value.trace.return_value = mock_trace
            await t.connect()

            await t.trace_query("What is X?", result_count=3, duration_ms=42.0)
            mock_span.end.assert_awaited_once()
            metadata = mock_trace.span.call_args.kwargs["input"]
            assert metadata["query_hash"]
            assert "What is X?" not in str(metadata)
            output = mock_span.end.await_args.kwargs["output"]
            assert output["result_count"] == 3

    async def test_trace_helpers_noop_when_disabled(self) -> None:
        """When disabled, trace helpers should not raise."""
        t = MemoryTracer(disabled=True)
        await t.trace_append("x", "y", 1)
        await t.trace_query("q", 0, 0.0)


def test_build_trace_correlation_links_mission_model_tool_and_promoted_finding(tmp_path) -> None:
    """Neutral trace output should follow a mission through model work to accepted state."""
    eventloom_path = tmp_path / ".eventloom"
    parent = EventLog(eventloom_path / "auth-main.jsonl")
    worker = EventLog(eventloom_path / "auth-api.jsonl")

    mission = parent.append(
        "coordination.mission.created",
        actor="lead",
        payload={"mission_id": "auth-main", "objective": "Ship auth refactor"},
        thread="auth-main",
    )
    checkout = parent.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={"query": "Ship auth refactor", "source": "cli", "mission_id": "auth-main"},
        thread="auth-main",
    )
    model_call = parent.append(
        "model.call.requested",
        actor="openai-compatible",
        payload={
            "provider": "openai-compatible",
            "model": "gpt-compatible",
            "query": "Ship auth refactor",
            "mission_id": "auth-main",
            "message_count": 2,
            "injected_memory": True,
        },
        thread="auth-main",
    )
    tool_call = parent.append(
        "tool.call.completed",
        actor="zaxy-observer",
        payload={
            "tool_name": "pytest",
            "status": "ok",
            "session_id": "auth-main",
            "argument_keys": ["target"],
            "arguments_redacted": True,
        },
        thread="auth-main",
    )
    finding = worker.append(
        "coordination.finding.reported",
        actor="auth-api-agent",
        payload={
            "mission_id": "auth-main",
            "worker_id": "auth-api",
            "finding_id": "finding-auth-api-1",
            "summary": "Expired JWKS cache caused API failures.",
            "evidence": [{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        },
        thread="auth-api",
    )
    review = parent.append(
        "coordination.finding.reviewed",
        actor="lead",
        payload={
            "mission_id": "auth-main",
            "finding_id": "finding-auth-api-1",
            "status": "accepted",
            "rationale": "Command backed.",
        },
        thread="auth-main",
    )
    promotion = parent.append(
        "coordination.finding.promoted",
        actor="lead",
        payload={"mission_id": "auth-main", "finding_id": "finding-auth-api-1"},
        thread="auth-main",
    )
    handoff = parent.append(
        "coordination.handoff.created",
        actor="lead",
        payload={"mission_id": "auth-main", "handoff_id": "auth-main:handoff:8", "summary": "Ready."},
        thread="auth-main",
    )

    trace = build_trace_correlation([mission, checkout, model_call, tool_call, finding, review, promotion, handoff])
    payload = trace.to_dict()

    assert payload["format"] == "zaxy.trace.v0.8"
    assert payload["summary"] == {
        "span_count": 8,
        "edge_count": 10,
        "mission_count": 1,
        "finding_count": 1,
        "model_call_count": 1,
        "tool_call_count": 1,
    }
    span_by_type = {span["event_type"]: span for span in payload["spans"]}
    assert span_by_type["coordination.mission.created"]["event_hash"] == mission.hash
    assert span_by_type["model.call.requested"]["attributes"] == {
        "provider": "openai-compatible",
        "model": "gpt-compatible",
        "query": "Ship auth refactor",
        "message_count": 2,
        "injected_memory": True,
        "mission_id": "auth-main",
    }
    assert "summary" not in span_by_type["coordination.finding.reported"]["attributes"]
    edges = {(edge["source"], edge["target"], edge["relation"]) for edge in payload["edges"]}
    assert (
        f"event:{mission.thread}:{mission.seq}",
        f"event:{model_call.thread}:{model_call.seq}",
        "contains",
    ) in edges
    assert (
        f"event:{model_call.thread}:{model_call.seq}",
        f"event:{tool_call.thread}:{tool_call.seq}",
        "observed_tool_call",
    ) in edges
    assert (
        f"event:{review.thread}:{review.seq}",
        f"event:{finding.thread}:{finding.seq}",
        "reviews",
    ) in edges
    assert (
        f"event:{promotion.thread}:{promotion.seq}",
        f"event:{finding.thread}:{finding.seq}",
        "promotes",
    ) in edges
    assert (
        f"event:{mission.thread}:{mission.seq}",
        f"event:{handoff.thread}:{handoff.seq}",
        "contains",
    ) in edges


# ------------------------------------------------------------------
# Fallback tests
# ------------------------------------------------------------------

class TestFallback:
    """Tests for when Pathlight is not installed or unavailable."""

    async def test_missing_pathlight_package_sets_disabled(self) -> None:
        """If pathlight is not installed, _HAS_PATHLIGHT should be False."""
        with patch("zaxy.trace._HAS_PATHLIGHT", False):
            t = MemoryTracer(disabled=False)
            assert t.disabled is True

    async def test_env_default_url(self) -> None:
        """Default base_url should read from PATHLIGHT_URL env var."""
        with patch.dict("os.environ", {"PATHLIGHT_URL": "http://custom:4100"}), patch(
            "zaxy.trace._HAS_PATHLIGHT", True
        ):
            t = MemoryTracer()
            assert t.base_url == "http://custom:4100"
