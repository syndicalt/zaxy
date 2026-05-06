"""Pathlight observability hooks.

Wraps the Pathlight Python SDK to emit spans for every memory operation.
If Pathlight is unavailable (collector down), tracing fails silently so
that memory operations are never blocked by observability.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from zaxy.config import get_settings
from zaxy.security import query_hash

try:
    from pathlight import AsyncPathlight
    _HAS_PATHLIGHT = True
except ImportError:
    _HAS_PATHLIGHT = False


class MemoryTracer:
    """Tracer for memory fabric operations.

    Args:
        base_url: Pathlight collector URL. Defaults to PATHLIGHT_URL env var
            or http://localhost:4100.
        project_id: Optional project identifier.
        disabled: If True, all tracing is no-ops.
    """

    def __init__(
        self,
        base_url: str | None = None,
        project_id: str | None = None,
        disabled: bool = False,
        trace_raw_queries: bool | None = None,
    ) -> None:
        self.disabled = disabled or not _HAS_PATHLIGHT
        self._client: Any | None = None
        settings = get_settings()
        self.trace_raw_queries = (
            settings.trace_raw_queries if trace_raw_queries is None else trace_raw_queries
        )
        if not self.disabled:
            self.base_url = base_url or os.getenv("PATHLIGHT_URL", "http://localhost:4100")
            self.project_id = project_id

    async def connect(self) -> None:
        """Initialize the Pathlight client."""
        if self.disabled:
            return
        self._client = AsyncPathlight(
            base_url=self.base_url or "http://localhost:4100",
            project_id=self.project_id,
        )

    async def close(self) -> None:
        """Close the Pathlight client."""
        if self._client and hasattr(self._client, "close"):
            await self._client.close()
        self._client = None

    @asynccontextmanager
    async def span(
        self,
        name: str,
        span_type: str = "custom",
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Context manager that traces a block of code.

        Yields a mutable dict where callers can set ``output``, ``error``,
        and ``duration_ms`` before exiting the context.
        """
        if self.disabled or self._client is None:
            yield {}
            return

        import time

        trace = self._client.trace("zaxy-memory", metadata or {})
        sp = trace.span(name, span_type, input=metadata)
        result: dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield result
        except Exception as exc:
            result["error"] = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            await sp.end(
                output=result.get("output"),
                error=result.get("error"),
                duration_ms=duration_ms,
            )
            await trace.end()

    async def trace_append(
        self,
        event_type: str,
        actor: str,
        seq: int,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Fire-and-forget trace for an append operation."""
        if self.disabled:
            return
        async with self.span("memory_append", metadata={"event_type": event_type, "actor": actor, "seq": seq}) as result:
            result["output"] = {"seq": seq, "success": success}
            if error:
                result["error"] = error

    async def trace_query(
        self,
        query: str,
        result_count: int,
        duration_ms: float,
        temporal_filter: str | None = None,
    ) -> None:
        """Fire-and-forget trace for a query operation."""
        if self.disabled:
            return
        metadata = {"query_hash": query_hash(query), "temporal_filter": temporal_filter}
        if self.trace_raw_queries:
            metadata["query"] = query
        async with self.span("memory_query", metadata=metadata) as result:
            result["output"] = {"result_count": result_count, "duration_ms": duration_ms}
