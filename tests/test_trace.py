"""Tests for zaxy.trace — Pathlight observability hooks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.trace import MemoryTracer

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
            output = mock_span.end.await_args.kwargs["output"]
            assert output["result_count"] == 3

    async def test_trace_helpers_noop_when_disabled(self) -> None:
        """When disabled, trace helpers should not raise."""
        t = MemoryTracer(disabled=True)
        await t.trace_append("x", "y", 1)
        await t.trace_query("q", 0, 0.0)


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
