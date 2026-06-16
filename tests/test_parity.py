"""Parity tests: the MCP surface and the Python API (MemoryFabric) share one path.

These run a real embedded backend and assert that append/query/checkout and the
context lifecycle tools (context_assemble / context_after_turn / subagent_cleanup)
produce equivalent effects/results whether driven through the MCP handlers or the
fabric directly — the guarantee behind unifying the two surfaces onto one path.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from zaxy.mcp_server import ZaxyMCPServer

pytestmark = pytest.mark.filterwarnings("ignore")


async def _embedded_server(path: Path) -> ZaxyMCPServer:
    server = ZaxyMCPServer(eventloom_path=str(path / ".eventloom"), projection_backend="embedded")
    await server.setup()
    return server


def _json(result: list[Any]) -> Any:
    return json.loads(result[0].text)


async def test_query_parity_mcp_matches_fabric(tmp_path: Path) -> None:
    """handle_memory_query returns exactly the fabric.query_page contexts, flattened."""
    server = await _embedded_server(tmp_path)
    try:
        await server.handle_memory_append(
            {"event_type": "goal.created", "actor": "u", "payload": {"title": "Ship the export contract"}, "session_id": "s"}
        )
        await server.handle_memory_append(
            {"event_type": "decision.made", "actor": "u", "payload": {"decision": "Adopt the export contract"}, "session_id": "s"}
        )

        handler = _json(await server.handle_memory_query({"query": "export contract", "session_id": "s", "limit": 5}))
        page = await server._fabric.query_page("export contract", session_id="s", limit=5)

        assert [row["content"] for row in handler] == [c.content for c in page.contexts]
        assert [row["citation"] for row in handler] == [(c.metadata or {}).get("citation") for c in page.contexts]
        assert [row["score"] for row in handler] == [c.score for c in page.contexts]
    finally:
        await server.teardown()


async def test_checkout_parity_mcp_matches_fabric(tmp_path: Path) -> None:
    """handle_memory_checkout's cited facts match fabric.checkout_memory."""
    server = await _embedded_server(tmp_path)
    try:
        await server.handle_memory_append(
            {"event_type": "decision.made", "actor": "u", "payload": {"decision": "Adopt the export contract"}, "session_id": "s"}
        )
        await server.handle_memory_append(
            {"event_type": "goal.created", "actor": "u", "payload": {"title": "Ship the export contract"}, "session_id": "s"}
        )

        # Fabric first (read-only, no reinforcement) so the handler's reinforcement
        # append can't perturb the comparison.
        fabric_checkout = (
            await server._fabric.checkout_memory("export contract", session_id="s", limit=5, record_reinforcement=False)
        ).to_dict()
        handler = _json(await server.handle_memory_checkout({"query": "export contract", "session_id": "s", "limit": 5}))

        assert handler["session_id"] == fabric_checkout["session_id"]
        assert [f["citation"] for f in handler["current_facts"]] == [
            f["citation"] for f in fabric_checkout["current_facts"]
        ]
        assert handler["quality"]["answerability"] == fabric_checkout["quality"]["answerability"]
    finally:
        await server.teardown()


async def test_append_parity_projects_equivalently(tmp_path: Path) -> None:
    """An event appended via the MCP handler and via the Python API projects the
    same way (both retrievable), and the MCP append now emits the fabric's
    inferred edges (parity it previously lacked)."""
    from zaxy.core import MemoryFabric

    server = await _embedded_server(tmp_path / "mcp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fabric = MemoryFabric(
            eventloom_path=str(tmp_path / "fab" / ".eventloom"),
            projection_backend="embedded",
            embedded_graph_path=str(tmp_path / "fab" / "graph.kuzu"),
            tracer_disabled=True,
        )
    await fabric.connect()
    try:
        payload = {"title": "Ship the export contract"}
        await server.handle_memory_append(
            {"event_type": "goal.created", "actor": "u", "payload": payload, "session_id": "s"}
        )
        await fabric.append("goal.created", "u", dict(payload), session_id="s")

        mcp_hits = _json(await server.handle_memory_query({"query": "export contract", "session_id": "s", "limit": 5}))
        fabric_hits = await fabric.query_page("export contract", session_id="s", limit=5)

        # Both surfaces projected the event so it is retrievable.
        assert mcp_hits, "MCP append did not project a retrievable entity"
        assert fabric_hits.contexts, "fabric append did not project a retrievable entity"

        # The MCP append went through the shared fabric pipeline: the two logs hold
        # the same event types (incl. any generated inferences), proving the MCP
        # path no longer skips inference/projection the Python API performs.
        mcp_types = [e.type for e in server.session_manager.get("s").eventlog.read_all()]
        fab_types = [e.type for e in fabric.session_manager.get("s").eventlog.read_all()]
        assert mcp_types == fab_types
    finally:
        await fabric.close()
        await server.teardown()


async def test_context_assemble_parity_mcp_matches_fabric(tmp_path: Path) -> None:
    """handle_context_assemble returns exactly the serialized fabric assembly."""
    from zaxy.mcp_server import _context_assembly_payload

    server = await _embedded_server(tmp_path)
    try:
        await server.handle_memory_append(
            {"event_type": "goal.created", "actor": "u", "payload": {"title": "Ship the export contract"}, "session_id": "s"}
        )
        # Both read-only over the same state -> identical payloads.
        fabric_payload = _context_assembly_payload(
            await server._fabric.assemble_context("export contract", session_id="s", replay_from_seq=1, limit=5, max_recent_events=20)
        )
        handler = _json(await server.handle_context_assemble({"query": "export contract", "session_id": "s", "limit": 5}))
        assert handler["session_id"] == fabric_payload["session_id"]
        assert [c["content"] for c in handler["contexts"]] == [c["content"] for c in fabric_payload["contexts"]]
        assert handler["context_counts"] == fabric_payload["context_counts"]
    finally:
        await server.teardown()


async def test_after_turn_and_cleanup_run_through_fabric(tmp_path: Path) -> None:
    """context_after_turn and subagent_cleanup go end-to-end through the fabric."""
    server = await _embedded_server(tmp_path)
    try:
        after = _json(await server.handle_context_after_turn(
            {"role": "user", "content": "Ship the export contract", "session_id": "s"}
        ))
        # after_turn appended the turn and assembled (fabric.after_turn shape).
        assert after["session_id"] == "s"
        assert "prompt" in after and "contexts" in after and "context_counts" in after

        cleanup = _json(await server.handle_subagent_cleanup(
            {"parent_session_id": "parent", "subagent_session_id": "sub", "summary": "did the work"}
        ))
        # Canonical HandoffBundle shape from fabric.cleanup_subagent.
        assert cleanup["session_id"] == "sub"
        assert set(cleanup) >= {"session_id", "summary", "prompt", "contexts", "replay_event_count", "integrity_ok"}
        sub_events = [e.type for e in server.session_manager.get("sub").eventlog.read_all()]
        assert "subagent.cleaned" in sub_events
    finally:
        await server.teardown()
