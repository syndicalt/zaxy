"""Tests for the graph-degraded null projection backend."""

from __future__ import annotations

import pytest

from zaxy.core import MemoryFabric
from zaxy.null_projection_store import NullProjectionStore
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store


def _null_config() -> ProjectionBackendConfig:
    return ProjectionBackendConfig(
        backend="null",
        neo4j_uri="",
        neo4j_user="",
        neo4j_password="",
        neo4j_ca_cert=None,
        neo4j_trust_all=False,
    )


def test_build_projection_store_returns_null_backend() -> None:
    store = build_projection_store(_null_config())
    assert isinstance(store, NullProjectionStore)


@pytest.mark.asyncio
async def test_null_store_reads_are_empty_and_writes_are_noops() -> None:
    store = NullProjectionStore()
    await store.connect()
    await store.init_schema()
    # Writes never raise.
    await store.invalidate_entity("x", "concept", "2026-06-16T00:00:00Z")
    await store.retire_source_projections(source_path="p", invalid_at="2026-06-16T00:00:00Z")
    # Every read lane is empty.
    assert await store.search_exact("x") == []
    assert await store.search_keyword("x") == []
    assert await store.search_vector([0.0, 1.0]) == []
    assert await store.search_traversal("x") == []
    assert (
        await store.search_causal_neighbors("x", direction="successors") == []
    )
    assert await store.has_traversal_edges() is False
    status = await store.inspect_event_projection_status("agent")
    assert status.event_count == 0 and status.session_id == "agent"
    inferred = await store.inspect_inferred_edge_status("agent")
    assert inferred.total_edges == 0
    await store.close()


@pytest.mark.asyncio
async def test_fabric_checkout_is_graph_degraded_with_null_backend(tmp_path) -> None:  # noqa: ANN001
    """A fabric on the null backend still appends, replays, and checks out via the
    verbatim + verified-replay lanes — no graph, no error.
    """
    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="null",
        tracer_disabled=True,
    )
    await fabric.connect()
    try:
        await fabric.append(
            "transcript.turn",
            actor="assistant",
            payload={"role": "assistant", "content": "Graph-degraded checkout still recalls this."},
            thread="agent",
        )
        checkout = await fabric.checkout_memory("what does degraded recall", session_id="agent")
        # The checkout succeeds and the verbatim lane still surfaces the content.
        assert checkout is not None
        blob = str(checkout.to_dict())
        assert "degraded" in blob.casefold()
    finally:
        await fabric.close()
