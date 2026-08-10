"""Tests for MCP tool profiles, the memory_checkout front door, and umbrella tools."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

import zaxy.mcp_server
from zaxy.__main__ import app
from zaxy.config import Settings
from zaxy.embedded_graph_store import EmbeddedGraphStore
from zaxy.event import EventLog
from zaxy.extract import ExtractedEntity, ExtractionResult
from zaxy.mcp_server import TOOLS, ZaxyMCPServer
from zaxy.metacognition import build_feeling_of_knowing_index
from zaxy.tool_profiles import CORE_TOOLS, resolve_profile
from zaxy.workspace import build_workspace_instruction_event

ALL_TOOL_NAMES = {tool.name for tool in TOOLS}
UMBRELLA_TOOL_NAMES = {"memory_consolidation", "memory_confidence"}
# The pre-profile (Zaxy 2.0.1) tool surface; profiles and umbrellas are additive.
LEGACY_TOOL_NAMES = {
    "memory_append",
    "memory_query",
    "memory_causal_successors",
    "memory_causal_predecessors",
    "memory_consolidation_candidate",
    "memory_consolidation_review",
    "memory_consolidation_propose_from_log",
    "memory_consolidation_status",
    "memory_explain_outcome",
    "memory_propose_belief_update",
    "memory_claim_confidence",
    "memory_similar_procedures",
    "memory_record_known_unknown",
    "memory_known_unknowns",
    "memory_confidence_trajectory",
    "memory_reverification_needs",
    "memory_plan_from_procedures",
    "memory_verbatim",
    "memory_feedback",
    "memory_synthesis_artifact",
    "memory_synthesis_evidence",
    "memory_skill",
    "memory_replay",
    "memory_invalidate",
    "memory_capabilities",
    "memory_bootstrap",
    "memory_checkout",
    "context_assemble",
    "context_after_turn",
    "subagent_cleanup",
    "coordination_start",
    "coordination_worker_create",
    "coordination_assign",
    "coordination_report_finding",
    "coordination_merge_brief",
    "coordination_checkout",
    "coordination_performance_ledger",
    "coordination_approval_packet",
    "coordination_apply_approval",
    "coordination_review_finding",
    "coordination_promote",
    "coordination_handoff",
    "coordination_record_synthesis_artifact",
    "coordination_proof_trace",
}


def json_loads(value: str) -> Any:
    return json.loads(value)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def make_server() -> Iterator[Callable[..., ZaxyMCPServer]]:
    """Return a factory for servers with mocked graph, tracer, and session manager."""
    with (
        patch("zaxy.mcp_server.build_projection_store") as mock_build_projection_store,
        patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
        patch("zaxy.mcp_server.SessionManager") as mock_session_cls,
    ):
        mock_graph = AsyncMock()
        mock_build_projection_store.return_value = mock_graph
        mock_tracer_cls.return_value = AsyncMock()

        mock_log = MagicMock()
        mock_log.append.return_value = MagicMock(seq=1, hash="a" * 64)
        mock_session_mgr = MagicMock()
        mock_session_mgr.get.return_value.eventlog = mock_log
        mock_session_cls.return_value = mock_session_mgr

        def _build(**kwargs: Any) -> ZaxyMCPServer:
            return ZaxyMCPServer(**kwargs)

        yield _build


def _real_server(tmp_path: Path, **kwargs: Any) -> ZaxyMCPServer:
    """Return a server with a real tmp eventloom and mocked graph/tracer.

    The graph/tracer are mocked *at construction* so the server's persistent
    fabric is wired to the mocks (not a real, never-connected embedded store).
    """
    with (
        patch("zaxy.mcp_server.build_projection_store", return_value=AsyncMock()),
        patch("zaxy.mcp_server.MemoryTracer", return_value=AsyncMock()),
    ):
        server = ZaxyMCPServer(
            eventloom_path=str(tmp_path / ".eventloom"),
            workspace_root=tmp_path,
            **kwargs,
        )
    return server


# ------------------------------------------------------------------
# Profile resolution
# ------------------------------------------------------------------

class TestResolveProfile:
    """Tests for the tool_profiles module."""

    def test_full_profile_resolves_to_no_filter(self) -> None:
        """full should disable listing filtering entirely."""
        assert resolve_profile("full") is None

    def test_core_profile_resolves_to_core_tools(self) -> None:
        """core should resolve to the front-door verb set."""
        assert resolve_profile("core") == CORE_TOOLS

    def test_unknown_profile_raises_clear_error(self) -> None:
        """Unknown profile names should fail with the valid choices."""
        with pytest.raises(ValueError, match="Unknown MCP tool profile: 'compact'.*core, full"):
            resolve_profile("compact")

    def test_core_tools_reserve_feeling_of_knowing(self) -> None:
        """The 2.2-reserved metamemory pre-check stays in the core set."""
        assert "memory_feeling_of_knowing" in CORE_TOOLS

    def test_settings_default_core_with_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MCP_TOOL_PROFILE should follow the SCREAMING_SNAKE settings convention.

        2.1.0 flipped the default listing profile to core;
        MCP_TOOL_PROFILE=full is the documented opt-out.
        """
        assert Settings().mcp_tool_profile == "core"
        monkeypatch.setenv("MCP_TOOL_PROFILE", "full")
        assert Settings().mcp_tool_profile == "full"
        monkeypatch.setenv("MCP_TOOL_PROFILE", "compact")
        with pytest.raises(ValueError, match="mcp_tool_profile"):
            Settings()


# ------------------------------------------------------------------
# Listing profiles
# ------------------------------------------------------------------

class TestToolListingProfiles:
    """Tests for profile-filtered tool listing."""

    def test_core_profile_lists_at_most_eight_tools(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """The core profile should list a small front-door verb set."""
        server = make_server(tool_profile="core")
        visible = server.visible_tools()

        assert len(visible) <= 8
        assert {tool.name for tool in visible} == CORE_TOOLS & ALL_TOOL_NAMES

    def test_core_profile_hides_umbrella_tools(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """Umbrella tools are full-profile-only listings."""
        server = make_server(tool_profile="core")

        assert UMBRELLA_TOOL_NAMES.isdisjoint({tool.name for tool in server.visible_tools()})

    def test_default_server_lists_the_core_profile(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """2.1.0 default flip: an unconfigured server lists the core verb set."""
        server = make_server()

        assert server._tool_profile_name == "core"
        assert {tool.name for tool in server.visible_tools()} <= CORE_TOOLS

    def test_full_profile_lists_every_existing_tool(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """The explicit full profile must keep listing every pre-profile tool name."""
        # Pinned explicitly since 2.1.0 flipped the default listing to core.
        server = make_server(tool_profile="full")
        visible_names = [tool.name for tool in server.visible_tools()]

        assert server._tool_profile_name == "full"
        assert visible_names == [tool.name for tool in TOOLS]
        assert set(visible_names) >= LEGACY_TOOL_NAMES

    def test_unknown_profile_rejected_at_server_construction(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """Server construction should validate the profile name."""
        with pytest.raises(ValueError, match="Unknown MCP tool profile"):
            make_server(tool_profile="compact")

    async def test_hidden_tool_still_dispatches_by_name(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """Profiles change listing, not capability: hidden tools stay callable."""
        server = make_server(tool_profile="core")
        assert "memory_consolidation_status" not in {tool.name for tool in server.visible_tools()}

        expected = {"session_id": "agent-1", "pending_count": 0}
        fabric = AsyncMock()
        fabric.consolidation_status.return_value = expected
        server._fabric = fabric  # handlers delegate to the persistent fabric
        result = await zaxy.mcp_server._dispatch_tool_call(
            server,
            "memory_consolidation_status",
            {"session_id": "agent-1"},
        )

        assert json_loads(result[0].text) == expected
        fabric.consolidation_status.assert_awaited_once_with(session_id="agent-1")

    def test_feeling_of_knowing_listed_in_both_profiles(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """The experimental pre-check ships listed in core and full alike."""
        for profile in ("core", "full"):
            server = make_server(tool_profile=profile)
            assert "memory_feeling_of_knowing" in {tool.name for tool in server.visible_tools()}

    def test_listing_leads_with_memory_checkout_in_both_profiles(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """The front door must be the first listed tool under every profile."""
        for profile in ("core", "full"):
            server = make_server(tool_profile=profile)
            assert server.visible_tools()[0].name == "memory_checkout"

    def test_core_profile_lists_front_door_verbs_in_order(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """The core listing should be exactly the ordered front-door verb set."""
        server = make_server(tool_profile="core")

        assert [tool.name for tool in server.visible_tools()] == [
            "memory_checkout",
            "memory_append",
            "memory_query",
            "context_assemble",
            "memory_feedback",
            "memory_invalidate",
            "memory_capabilities",
            "memory_feeling_of_knowing",
        ]

    def test_checkout_description_is_the_front_door(self) -> None:
        """memory_checkout should be positioned as the single entry point."""
        description = next(t for t in TOOLS if t.name == "memory_checkout").description
        assert description is not None

        assert "front door" in description.lower()
        assert "call this first" in description.lower()
        assert "before substantial work" in description.lower()

    def test_capabilities_description_is_the_discovery_surface(self) -> None:
        """memory_capabilities should advertise profile-aware discovery."""
        description = next(t for t in TOOLS if t.name == "memory_capabilities").description
        assert description is not None

        assert "discover" in description.lower()
        assert "profile" in description.lower()
        assert "front door" in description.lower()

    def test_workspace_instruction_block_names_front_door(self, tmp_path: Path) -> None:
        """The emitted workspace instruction block should point at memory_checkout."""
        (tmp_path / "AGENTS.md").write_text("# Project\nUse the memory.\n", encoding="utf-8")

        event = build_workspace_instruction_event(tmp_path, session_id="agent-1")

        assert event is not None
        front_door = event["payload"]["memory_front_door"]
        assert front_door["tool"] == "memory_checkout"
        assert "front door" in front_door["guidance"]


# ------------------------------------------------------------------
# memory_capabilities profile block
# ------------------------------------------------------------------

class TestCapabilitiesProfileBlock:
    """Tests for the profile delta reported by memory_capabilities."""

    async def test_core_profile_reports_available_but_unlisted_tools(
        self,
        tmp_path: Path,
    ) -> None:
        """Core-profile capabilities should report the listing delta."""
        server = _real_server(tmp_path, tool_profile="core")

        result = await server.handle_memory_capabilities({"session_id": "agent-1"})

        payload = json_loads(result[0].text)
        profile = payload["profile"]
        assert profile["active"] == "core"
        assert set(profile["listed_tools"]) == CORE_TOOLS & ALL_TOOL_NAMES
        assert profile["available_but_unlisted"] == sorted(ALL_TOOL_NAMES - CORE_TOOLS)
        assert "callable by name" in profile["note"]

    async def test_full_profile_reports_no_unlisted_tools(self, tmp_path: Path) -> None:
        """Full-profile capabilities should report the unfiltered listing."""
        server = _real_server(tmp_path, tool_profile="full")

        result = await server.handle_memory_capabilities({"session_id": "agent-1"})

        profile = json_loads(result[0].text)["profile"]
        assert profile["active"] == "full"
        assert profile["listed_tools"] == [tool.name for tool in TOOLS]
        assert "available_but_unlisted" not in profile

    async def test_capabilities_report_vector_search_settings(self, tmp_path: Path) -> None:
        """Capabilities should surface the effective ANN engagement rule.

        2.2 G4 engagement: scopes at or below ann_max_dimension engage when
        count >= ann_threshold OR — with ann_byte_budget_engagement on and no
        int8 opt-in — when the exact float64 matrix would exceed the reported
        cache byte budget.
        """
        from zaxy.embedded_graph_store import VECTOR_INDEX_CACHE_MAX_BYTES

        server = _real_server(tmp_path)

        result = await server.handle_memory_capabilities({"session_id": "agent-1"})

        vector_search = json_loads(result[0].text)["vector_search"]
        assert vector_search == {
            "quantization": server._settings.vector_quantization,
            "ann_threshold": server._settings.vector_ann_threshold,
            "ann_max_dimension": server._settings.vector_ann_max_dimension,
            "ann_byte_budget_engagement": server._settings.vector_ann_byte_budget_engagement,
            "vector_index_cache_max_bytes": VECTOR_INDEX_CACHE_MAX_BYTES,
        }
        assert vector_search["quantization"] in {"none", "int8"}
        assert isinstance(vector_search["ann_threshold"], int)
        assert vector_search["ann_threshold"] >= 1
        assert isinstance(vector_search["ann_max_dimension"], int)
        assert vector_search["ann_max_dimension"] >= 1
        assert isinstance(vector_search["ann_byte_budget_engagement"], bool)
        assert vector_search["vector_index_cache_max_bytes"] == 256 * 1024 * 1024


# ------------------------------------------------------------------
# memory_feeling_of_knowing
# ------------------------------------------------------------------

def _fok_server(tmp_path: Path, entity_names: list[str]) -> ZaxyMCPServer:
    """Return a server whose projection store reports the given active entities."""
    server = _real_server(tmp_path)
    server.graph.active_entity_names = AsyncMock(return_value=list(entity_names))
    return server


class TestFeelingOfKnowing:
    """Tests for the experimental memory_feeling_of_knowing pre-check."""

    async def test_query_about_known_entity_predicts_likely_with_breakdown(
        self,
        tmp_path: Path,
    ) -> None:
        """A query naming a projected entity should predict likely with signals."""
        server = _fok_server(tmp_path, ["Kuzu", "EmbeddedGraphStore"])

        result = await server.handle_memory_feeling_of_knowing(
            {"query": "kuzu", "session_id": "agent-1"}
        )

        payload = json_loads(result[0].text)
        assert payload["verdict"] == "likely"
        assert payload["score"] == pytest.approx(0.6)
        assert payload["authority_status"] == "non_authoritative"
        assert payload["signals"]["query_term_count"] == 1
        assert payload["signals"]["bloom_hits"] == 1
        assert payload["signals"]["bloom_hit_ratio"] == pytest.approx(1.0)

    async def test_partial_entity_overlap_predicts_possible(self, tmp_path: Path) -> None:
        """A query that only partially touches known entities should be possible."""
        server = _fok_server(tmp_path, ["Kuzu"])

        result = await server.handle_memory_feeling_of_knowing(
            {"query": "kuzu migration", "session_id": "agent-1"}
        )

        payload = json_loads(result[0].text)
        assert payload["verdict"] == "possible"
        assert payload["signals"]["bloom_hits"] == 1
        assert payload["signals"]["query_term_count"] == 2

    async def test_absent_topic_predicts_unlikely(self, tmp_path: Path) -> None:
        """A query about a never-projected topic should predict unlikely."""
        server = _fok_server(tmp_path, ["Kuzu", "EmbeddedGraphStore"])

        result = await server.handle_memory_feeling_of_knowing(
            {"query": "quarterly payroll reconciliation", "session_id": "agent-1"}
        )

        payload = json_loads(result[0].text)
        assert payload["verdict"] == "unlikely"
        assert payload["signals"]["bloom_hits"] == 0

    async def test_empty_session_predicts_unlikely_without_error(self, tmp_path: Path) -> None:
        """An empty session is an honest unlikely prediction, not an error."""
        server = _fok_server(tmp_path, [])

        result = await server.handle_memory_feeling_of_knowing(
            {"query": "anything at all", "session_id": "agent-1"}
        )

        payload = json_loads(result[0].text)
        assert payload["verdict"] == "unlikely"
        assert payload["score"] == 0.0

    async def test_backend_without_entity_surface_degrades_to_unlikely(
        self,
        tmp_path: Path,
    ) -> None:
        """Backends without the in-memory accessor degrade honestly, not loudly."""
        server = _real_server(tmp_path)
        server.graph = AsyncMock(spec=[])

        result = await server.handle_memory_feeling_of_knowing(
            {"query": "kuzu", "session_id": "agent-1"}
        )

        assert json_loads(result[0].text)["verdict"] == "unlikely"

    async def test_cue_values_are_probed_alongside_query_terms(self, tmp_path: Path) -> None:
        """A cue naming a known entity should raise the verdict."""
        server = _fok_server(tmp_path, ["Kuzu"])

        without_cues = await server.handle_memory_feeling_of_knowing(
            {"query": "latency", "session_id": "agent-1"}
        )
        with_cues = await server.handle_memory_feeling_of_knowing(
            {"query": "latency", "session_id": "agent-1", "cues": {"workspace": "kuzu"}}
        )

        assert json_loads(without_cues[0].text)["verdict"] == "unlikely"
        cued = json_loads(with_cues[0].text)
        assert cued["verdict"] == "possible"
        assert cued["signals"]["query_term_count"] == 2
        assert cued["signals"]["bloom_hits"] == 1

    async def test_non_string_cue_values_are_rejected(self, tmp_path: Path) -> None:
        """Cue fields must be non-empty strings, matching the strict schema."""
        server = _fok_server(tmp_path, ["Kuzu"])

        with pytest.raises(ValueError, match=r"cues\['phase'\] must be a non-empty string"):
            await server.handle_memory_feeling_of_knowing(
                {"query": "kuzu", "session_id": "agent-1", "cues": {"phase": 3}}
            )
        with pytest.raises(ValueError, match="cues must be an object"):
            await server.handle_memory_feeling_of_knowing(
                {"query": "kuzu", "session_id": "agent-1", "cues": ["planning"]}
            )

    async def test_index_cache_reuses_until_the_session_log_changes(
        self,
        tmp_path: Path,
    ) -> None:
        """The per-session index is built once and rebuilt after a new append."""
        server = _fok_server(tmp_path, [])
        with patch(
            "zaxy.mcp_server.build_feeling_of_knowing_index",
            side_effect=build_feeling_of_knowing_index,
        ) as builder:
            first = await server.handle_memory_feeling_of_knowing(
                {"query": "kuzu", "session_id": "agent-1"}
            )
            await server.handle_memory_feeling_of_knowing(
                {"query": "kuzu", "session_id": "agent-1"}
            )
            assert builder.call_count == 1
            assert json_loads(first[0].text)["verdict"] == "unlikely"

            # A new projected entity advances the log tail, invalidating the
            # cached index; the rebuilt index sees the new entity name.
            server.graph.active_entity_names.return_value = ["Kuzu"]
            await server.handle_memory_append(
                {
                    "event_type": "decision.made",
                    "actor": "tester",
                    "payload": {"decision": "Adopt Kuzu for the embedded backend"},
                    "session_id": "agent-1",
                }
            )
            third = await server.handle_memory_feeling_of_knowing(
                {"query": "kuzu", "session_id": "agent-1"}
            )
            assert builder.call_count == 2
            assert json_loads(third[0].text)["verdict"] == "likely"

    async def test_calibration_marker_is_appended_with_stable_shape(
        self,
        tmp_path: Path,
    ) -> None:
        """Each prediction appends a non-authoritative calibration marker."""
        server = _fok_server(tmp_path, ["Kuzu"])

        result = await server.handle_memory_feeling_of_knowing(
            {"query": "kuzu retrieval", "session_id": "agent-1"}
        )

        verdict = json_loads(result[0].text)
        events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
        marker = events[-1]
        assert marker.type == "metacognition.fok.predicted"
        assert marker.actor == "zaxy-memory"
        assert marker.payload == {
            "query_hash": hashlib.sha256(b"kuzu retrieval").hexdigest(),
            "verdict": verdict["verdict"],
            "score": verdict["score"],
            "authority_status": "non_authoritative",
        }

    async def test_calibration_append_failure_never_fails_the_tool(
        self,
        tmp_path: Path,
    ) -> None:
        """A raising calibration append degrades silently; the verdict still returns."""
        server = _fok_server(tmp_path, ["Kuzu"])

        with patch.object(EventLog, "append", side_effect=OSError("disk full")):
            result = await server.handle_memory_feeling_of_knowing(
                {"query": "kuzu", "session_id": "agent-1"}
            )

        assert json_loads(result[0].text)["verdict"] == "likely"

    async def test_dispatch_routes_feeling_of_knowing(self, tmp_path: Path) -> None:
        """_dispatch_tool_call should route the new tool name."""
        server = _fok_server(tmp_path, ["Kuzu"])

        result = await zaxy.mcp_server._dispatch_tool_call(
            server,
            "memory_feeling_of_knowing",
            {"query": "kuzu", "session_id": "agent-1"},
        )

        assert json_loads(result[0].text)["verdict"] == "likely"

    @pytest.mark.skipif(
        importlib.util.find_spec("ladybug") is None,
        reason="ladybug is not installed",
    )
    async def test_end_to_end_against_real_embedded_projection(self, tmp_path: Path) -> None:
        """The accessor and handler agree against a real embedded projection."""
        store = EmbeddedGraphStore(tmp_path / "projections" / "embedded.kuzu")
        await store.connect()
        await store.init_schema()
        try:
            await store.upsert_extraction(
                ExtractionResult(
                    entities=[
                        ExtractedEntity(
                            name="Retrieval Roadmap",
                            entity_type="goal",
                            observed_at="2026-06-10T00:00:00Z",
                            summary="ship the metamemory pre-check",
                        )
                    ],
                    edges=[],
                    source_event_seq=1,
                    source_event_hash="a" * 64,
                ),
                session_id="agent-1",
            )
            server = _real_server(tmp_path)
            server.graph = store

            hit = await server.handle_memory_feeling_of_knowing(
                {"query": "retrieval roadmap", "session_id": "agent-1"}
            )
            miss = await server.handle_memory_feeling_of_knowing(
                {"query": "payroll reconciliation", "session_id": "agent-1"}
            )

            assert json_loads(hit[0].text)["verdict"] == "likely"
            assert json_loads(miss[0].text)["verdict"] == "unlikely"
        finally:
            await store.close()


# ------------------------------------------------------------------
# Umbrella tools
# ------------------------------------------------------------------

class TestUmbrellaTools:
    """Tests for the memory_consolidation and memory_confidence umbrellas."""

    async def test_memory_consolidation_status_matches_legacy_tool(
        self,
        tmp_path: Path,
    ) -> None:
        """Umbrella status should return the same payload as the legacy tool."""
        server = _real_server(tmp_path)
        source = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
            "decision.made",
            actor="tester",
            payload={"decision": "Adopt profile listing"},
            thread="agent-1",
        )
        await server.handle_memory_consolidation_candidate({
            "candidate_type": "claim",
            "title": "Profile listing adopted",
            "summary": "The team adopted profile-filtered tool listing.",
            "source_events": [{"seq": source.seq, "hash": source.hash}],
            "confidence": 0.9,
            "method": "manual-review",
            "session_id": "agent-1",
        })

        legacy = await server.handle_memory_consolidation_status({"session_id": "agent-1"})
        umbrella = await server.handle_memory_consolidation(
            {"operation": "status", "session_id": "agent-1"}
        )

        assert json_loads(umbrella[0].text) == json_loads(legacy[0].text)
        assert json_loads(legacy[0].text)["pending_count"] == 1

    async def test_memory_confidence_known_unknowns_matches_legacy_tool(
        self,
        tmp_path: Path,
    ) -> None:
        """Umbrella known_unknowns should return the same payload as the legacy tool."""
        server = _real_server(tmp_path)
        source = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
            "observation.recorded",
            actor="tester",
            payload={"text": "latency spike cause unknown"},
            thread="agent-1",
        )
        await server.handle_memory_record_known_unknown({
            "question": "Which backend caused latency?",
            "reason": "Evidence conflicted.",
            "source_events": [{"seq": source.seq, "hash": source.hash}],
            "claim_key": "backend-latency",
            "session_id": "agent-1",
        })

        legacy = await server.handle_memory_known_unknowns({"session_id": "agent-1"})
        umbrella = await server.handle_memory_confidence(
            {"operation": "known_unknowns", "session_id": "agent-1"}
        )

        assert json_loads(umbrella[0].text) == json_loads(legacy[0].text)
        assert json_loads(legacy[0].text)["unknown_count"] == 1

    @pytest.mark.parametrize(
        ("operation", "handler_name", "arguments"),
        [
            (
                "candidate",
                "handle_memory_consolidation_candidate",
                {
                    "candidate_type": "claim",
                    "title": "Retry policy",
                    "summary": "Retries should preserve original citations.",
                    "source_events": [{"seq": 7, "hash": "b" * 64}],
                    "confidence": 0.82,
                    "method": "manual-review",
                },
            ),
            ("propose_from_log", "handle_memory_consolidation_propose_from_log", {"session_id": "agent-1"}),
            ("status", "handle_memory_consolidation_status", {"session_id": "agent-1"}),
            (
                "review",
                "handle_memory_consolidation_review",
                {
                    "candidate_id": "consolidation:claim:" + ("c" * 24),
                    "status": "accepted",
                    "rationale": "Citations verified.",
                },
            ),
        ],
    )
    async def test_memory_consolidation_forwards_arguments_unchanged(
        self,
        make_server: Callable[..., ZaxyMCPServer],
        operation: str,
        handler_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Each consolidation operation should pass through to its legacy handler."""
        server = make_server()
        expected = [MagicMock()]
        handler = AsyncMock(return_value=expected)
        setattr(server, handler_name, handler)

        result = await server.handle_memory_consolidation({"operation": operation, **arguments})

        assert result == expected
        handler.assert_awaited_once_with(arguments)

    @pytest.mark.parametrize(
        ("operation", "handler_name", "arguments"),
        [
            ("claim", "handle_memory_claim_confidence", {"claim": "Projection is stale"}),
            ("trajectory", "handle_memory_confidence_trajectory", {"claim": "Projection is stale"}),
            ("reverification", "handle_memory_reverification_needs", {"limit": 5}),
            ("known_unknowns", "handle_memory_known_unknowns", {"status": "all"}),
            (
                "record_known_unknown",
                "handle_memory_record_known_unknown",
                {
                    "question": "Which backend caused latency?",
                    "reason": "Evidence conflicted.",
                    "source_events": [{"seq": 11, "hash": "e" * 64}],
                    "claim_key": "backend-latency",
                },
            ),
        ],
    )
    async def test_memory_confidence_forwards_arguments_unchanged(
        self,
        make_server: Callable[..., ZaxyMCPServer],
        operation: str,
        handler_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Each confidence operation should pass through to its legacy handler."""
        server = make_server()
        expected = [MagicMock()]
        handler = AsyncMock(return_value=expected)
        setattr(server, handler_name, handler)

        result = await server.handle_memory_confidence({"operation": operation, **arguments})

        assert result == expected
        handler.assert_awaited_once_with(arguments)

    @pytest.mark.parametrize(
        ("tool_name", "handler_name"),
        [
            ("memory_consolidation", "handle_memory_consolidation"),
            ("memory_confidence", "handle_memory_confidence"),
        ],
    )
    async def test_dispatch_routes_umbrella_tools(
        self,
        make_server: Callable[..., ZaxyMCPServer],
        tool_name: str,
        handler_name: str,
    ) -> None:
        """_dispatch_tool_call should route the umbrella tool names."""
        server = make_server()
        expected = [MagicMock()]
        handler = AsyncMock(return_value=expected)
        setattr(server, handler_name, handler)

        result = await zaxy.mcp_server._dispatch_tool_call(
            server, tool_name, {"operation": "status"}
        )

        assert result == expected
        handler.assert_awaited_once_with({"operation": "status"})

    async def test_unknown_consolidation_operation_raises_clear_error(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """Unknown operations should fail with the valid operation list."""
        server = make_server()

        with pytest.raises(
            ValueError,
            match=(
                "memory_consolidation requires 'operation' to be one of: "
                "candidate, propose_from_log, status, review; got 'bogus'"
            ),
        ):
            await server.handle_memory_consolidation({"operation": "bogus"})

    async def test_missing_confidence_operation_raises_clear_error(
        self,
        make_server: Callable[..., ZaxyMCPServer],
    ) -> None:
        """A missing operation should fail with the valid operation list."""
        server = make_server()

        with pytest.raises(
            ValueError,
            match=(
                "memory_confidence requires 'operation' to be one of: "
                "claim, trajectory, reverification, known_unknowns, record_known_unknown; "
                "got None"
            ),
        ):
            await server.handle_memory_confidence({"claim": "Projection is stale"})

    @pytest.mark.parametrize(
        ("handler_name", "arguments", "missing"),
        [
            (
                "handle_memory_consolidation",
                {"operation": "review", "session_id": "agent-1"},
                "candidate_id, status, rationale",
            ),
            (
                "handle_memory_consolidation",
                {"operation": "candidate", "title": "Retry policy"},
                "candidate_type, summary, source_events, confidence, method",
            ),
            ("handle_memory_confidence", {"operation": "claim"}, "claim"),
            (
                "handle_memory_confidence",
                {"operation": "record_known_unknown", "question": "Which backend?"},
                "reason, source_events, claim_key",
            ),
        ],
    )
    async def test_missing_required_operation_arguments_raise_clear_errors(
        self,
        make_server: Callable[..., ZaxyMCPServer],
        handler_name: str,
        arguments: dict[str, Any],
        missing: str,
    ) -> None:
        """Operation-specific required arguments should be validated up front."""
        server = make_server()

        with pytest.raises(ValueError, match=f"requires arguments: {missing}"):
            await getattr(server, handler_name)(arguments)

    def test_umbrella_schemas_compose_operation_enum_with_per_operation_requirements(self) -> None:
        """Umbrella input schemas should mark only truly required arguments per operation."""
        consolidation = next(t for t in TOOLS if t.name == "memory_consolidation")
        confidence = next(t for t in TOOLS if t.name == "memory_confidence")

        assert consolidation.input_schema["required"] == ["operation"]
        assert consolidation.input_schema["properties"]["operation"]["enum"] == [
            "candidate",
            "propose_from_log",
            "status",
            "review",
        ]
        consolidation_clauses = {
            clause["if"]["properties"]["operation"]["const"]: clause["then"]["required"]
            for clause in consolidation.input_schema["allOf"]
        }
        assert consolidation_clauses == {
            "candidate": ["candidate_type", "title", "summary", "source_events", "confidence", "method"],
            "review": ["candidate_id", "status", "rationale"],
        }

        assert confidence.input_schema["required"] == ["operation"]
        assert confidence.input_schema["properties"]["operation"]["enum"] == [
            "claim",
            "trajectory",
            "reverification",
            "known_unknowns",
            "record_known_unknown",
        ]
        confidence_clauses = {
            clause["if"]["properties"]["operation"]["const"]: clause["then"]["required"]
            for clause in confidence.input_schema["allOf"]
        }
        assert confidence_clauses == {
            "claim": ["claim"],
            "trajectory": ["claim"],
            "record_known_unknown": ["question", "reason", "source_events", "claim_key"],
        }


# ------------------------------------------------------------------
# serve --profile
# ------------------------------------------------------------------

@patch("zaxy.mcp_server.main", new_callable=AsyncMock)
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_threads_profile_into_server(
    mock_server_cls: MagicMock,
    mock_mcp_main: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`zaxy serve --profile core` should thread the profile into the MCP server."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["serve", "--profile", "core"],
        catch_exceptions=False,
        env={},
        color=False,
        prog_name="zaxy",
    )

    assert result.exit_code == 0
    mock_mcp_main.assert_awaited_once()
    mock_server_cls.assert_called_once()
    assert mock_server_cls.call_args.kwargs["tool_profile"] == "core"


@patch("zaxy.mcp_server.main", new_callable=AsyncMock)
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_defaults_profile_from_settings(
    mock_server_cls: MagicMock,
    mock_mcp_main: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bare `zaxy serve` should keep the core settings-default profile (2.1.0 flip)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["serve"],
        catch_exceptions=False,
        env={},
        color=False,
        prog_name="zaxy",
    )

    assert result.exit_code == 0
    mock_mcp_main.assert_awaited_once()
    # 2.1.0 flipped the settings default to core.
    assert mock_server_cls.call_args.kwargs["tool_profile"] == "core"


@patch("zaxy.mcp_server.main", new_callable=AsyncMock)
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_profile_full_restores_previous_listing(
    mock_server_cls: MagicMock,
    mock_mcp_main: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`zaxy serve --profile full` is the documented opt-out from the core default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["serve", "--profile", "full"],
        catch_exceptions=False,
        env={},
        color=False,
        prog_name="zaxy",
    )

    assert result.exit_code == 0
    mock_mcp_main.assert_awaited_once()
    assert mock_server_cls.call_args.kwargs["tool_profile"] == "full"


def test_serve_rejects_unknown_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`zaxy serve --profile bogus` should fail with the valid profile names."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["serve", "--profile", "bogus"],
        env={},
        color=False,
        prog_name="zaxy",
    )

    assert result.exit_code != 0
    assert "Unknown MCP tool profile" in result.output
