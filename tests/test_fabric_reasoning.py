"""Seam-contract tests for ReasoningOps (fabric decomposition phase 1).

The behavioral surface is covered end-to-end by tests/test_reasoning_primitives.py
through MemoryFabric; these tests pin the extraction's load-bearing properties —
late-bound host lookups and failure accounting — against a minimal fake host,
with no fabric, projection store, or event log involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from zaxy.core.fabric_reasoning import ReasoningOps


class _RecordingHost:
    """Minimal ReasoningHost double that records every append."""

    def __init__(self) -> None:
        self.appended: list[dict[str, Any]] = []
        self.graph = _FakeGraph([])
        self.session_manager = None  # not exercised by these tests

    async def append(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None = None,
        thread: str = "default",
        session_id: str | None = None,
        *,
        forgettable: bool = False,
    ) -> Any:
        self.appended.append({"event_type": event_type, "actor": actor, "payload": payload or {}})
        return None

    async def _append_event_spec(self, event: dict[str, Any], *, session_id: str) -> Any:
        self.appended.append(dict(event))
        return None

    async def checkout_memory(self, query: str, **kwargs: Any) -> Any:
        raise AssertionError("not exercised")

    async def query(self, query: str, **kwargs: Any) -> list[Any]:
        raise AssertionError("not exercised")

    async def query_causal_predecessors(self, entity_name: str, **kwargs: Any) -> list[Any]:
        raise AssertionError("not exercised")

    async def retrieve_similar_procedures(self, query: str, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("not exercised")


class _FakeGraph:
    def __init__(self, neighbors: list[Any]) -> None:
        self.neighbors = neighbors
        self.calls: list[dict[str, Any]] = []

    async def search_causal_neighbors(self, entity_name: str, **kwargs: Any) -> list[Any]:
        self.calls.append({"entity_name": entity_name, **kwargs})
        return self.neighbors


async def test_host_graph_is_late_bound_across_runtime_swaps() -> None:
    """Swapping host.graph AFTER construction must be honored (degraded fallback)."""
    host = _RecordingHost()
    ops = ReasoningOps(host=host)
    first = host.graph

    await ops.query_causal_successors("deploy pipeline")
    assert len(first.calls) == 1

    replacement = _FakeGraph([])  # e.g. graph-degraded fallback swapped the store
    host.graph = replacement
    await ops.query_causal_successors("deploy pipeline")
    assert len(replacement.calls) == 1
    assert len(first.calls) == 1  # the stale store was not used again


async def test_failed_primitive_appends_failure_observation_and_reraises() -> None:
    """A raising host still yields a cited 'failed' primitive-call event."""
    host = _RecordingHost()

    async def broken_checkout(query: str, **kwargs: Any) -> Any:
        raise RuntimeError("projection unavailable")

    host.checkout_memory = broken_checkout  # type: ignore[method-assign]
    ops = ReasoningOps(host=host)

    with pytest.raises(RuntimeError, match="projection unavailable"):
        await ops.get_claim_confidence("the cache is warm", session_id="default")

    calls = [e for e in host.appended if e["event_type"] == "reasoning.primitive.called"]
    assert len(calls) == 1
    assert calls[0]["payload"]["status"] == "failed"
    assert calls[0]["payload"]["primitive"] == "get_claim_confidence"
    assert calls[0]["payload"]["result_count"] == 0


async def test_intra_cluster_public_calls_route_through_the_host() -> None:
    """explain_outcome must use host.query_causal_predecessors, so fabric-level
    delegation (and any instance patch on it) stays the single dispatch point."""
    host = _RecordingHost()
    seen: list[str] = []

    class _Result:
        def to_dict(self) -> dict[str, Any]:
            return {
                "entity": "root-cause",
                "citation": "eventloom://default/events/1#abcdef123456",
            }

    async def fake_predecessors(entity_name: str, **kwargs: Any) -> list[Any]:
        seen.append(entity_name)
        return [_Result()]

    host.query_causal_predecessors = fake_predecessors  # type: ignore[method-assign]
    ops = ReasoningOps(host=host)

    result = await ops.explain_outcome("deploy failed")
    assert seen == ["deploy failed"]
    assert result["fallback_used"] is False
    assert result["result_count"] == 1
