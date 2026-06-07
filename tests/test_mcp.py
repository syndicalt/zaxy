"""Tests for zaxy.mcp_server — MCP protocol compliance."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import zaxy.mcp_server
from zaxy.causal import CausalQueryResult
from zaxy.config import Settings
from zaxy.core import build_memory_checkout
from zaxy.event import EventLog
from zaxy.graph import GraphEntity
from zaxy.mcp_server import (
    TOOLS,
    MCPTransportAuth,
    ZaxyMCPServer,
    main,
    remote_session_scope,
)


def json_loads(value: str) -> Any:
    return json.loads(value)


def _mcp_response_snapshot(name: str, payload: Any) -> dict[str, Any]:
    """Return stable representative response fields for v0.6 MCP snapshots."""
    if name == "memory_bootstrap":
        return {
            "mode": payload["mode"],
            "session_id": payload["session_id"],
            "startup_sequence": [
                {"tool": step["tool"], "argument_keys": sorted(step.get("arguments", {}).keys())}
                for step in payload["startup_sequence"]
            ],
            "recommended_next_tool": payload["capabilities"]["recommended_next_call"]["tool"],
            "eventloom_latest_seq": payload["capabilities"]["status"]["eventloom"]["latest_seq"],
            "prompt_contains_checkout_rule": "memory_checkout" in payload["prompt"],
        }
    if name == "memory_checkout":
        return {
            "session_id": payload["session_id"],
            "current_fact_count": len(payload["current_facts"]),
            "first_current_fact": {
                "content": payload["current_facts"][0]["content"],
                "citation": payload["current_facts"][0]["citation"],
                "source_lane": payload["current_facts"][0]["source_lane"],
            },
            "diagnostics": {
                "current_citation_count": payload["diagnostics"]["current_citation_count"],
                "feedback_tool": payload["diagnostics"]["feedback_tool"],
                "warning_count": payload["diagnostics"]["warning_count"],
                "purpose_profile": payload["diagnostics"].get("purpose", {}).get("profile"),
            },
            "purpose_profile": payload.get("purpose", {}).get("profile"),
            "quality": payload["quality"],
            "feedback_template_keys": sorted(payload["guidance"]["feedback"]["payloads"][0].keys()),
            "token_efficiency_keys": sorted(payload["token_efficiency"].keys()),
            "prompt_sections": [
                section
                for section in (
                    "# Memory Checkout",
                    "## Checkout Quality",
                    "## Checkout Guidance",
                    "## Checkout Diagnostics",
                )
                if section in payload["prompt"]
            ],
        }
    if name == "context_assemble":
        return {
            "session_id": payload["session_id"],
            "replay_event_count": payload["replay_event_count"],
            "context_count": len(payload["contexts"]),
            "first_context": {
                "content": payload["contexts"][0]["content"],
                "source": payload["contexts"][0]["source"],
                "score": payload["contexts"][0]["score"],
            },
            "warning_count": len(payload.get("warnings", [])),
            "prompt_contains_context": payload["contexts"][0]["content"] in payload["prompt"],
        }
    if name == "memory_query":
        return {
            "result_count": len(payload),
            "first_result": {
                "content": payload[0]["content"],
                "source": payload[0]["source"],
                "score": payload[0]["score"],
                "citation": payload[0]["citation"],
                "score_explanation_keys": sorted((payload[0].get("score_explanation") or {}).keys()),
            },
        }
    if name == "memory_verbatim":
        citation_prefix, _, citation_hash = payload[0]["citation"].partition("#")
        return {
            "result_count": len(payload),
            "first_result": {
                "content": payload[0]["content"],
                "source": payload[0]["source"],
                "source_kind": payload[0]["source_kind"],
                "citation_prefix": citation_prefix,
                "citation_hash_length": len(citation_hash),
                "metadata_keys": sorted(payload[0].get("metadata", {}).keys()),
            },
        }
    if name == "memory_feedback":
        return {
            "seq": payload["seq"],
            "hash_length": len(payload["hash"]),
            "event_type": payload["event_type"],
        }
    if name == "memory_synthesis_artifact":
        return {
            "artifact_event_type": payload["artifact_event"]["event_type"],
            "artifact_hash_length": len(payload["artifact_event"]["hash"]),
            "artifact_id_prefix": payload["artifact_id"].split(":", 1)[0],
            "candidate_event_type": (
                payload["candidate_event"]["event_type"] if payload.get("candidate_event") else None
            ),
            "candidate_outcome": payload["candidate_event"].get("outcome") if payload.get("candidate_event") else None,
        }
    if name == "memory_synthesis_evidence":
        return {
            "event_type": payload["event_type"],
            "hash_length": len(payload["hash"]),
            "outcome": payload["outcome"],
            "source_group": payload["source_group"],
            "fact_id": payload["fact_id"],
        }
    if name == "coordination_checkout":
        return {
            "mission_id": payload["mission_id"],
            "accepted_count": len(payload["accepted_findings"]),
            "pending_count": len(payload["pending_findings"]),
            "conflict_count": len(payload["conflicts"]),
            "first_accepted": {
                "worker_id": payload["accepted_findings"][0]["worker_id"],
                "status": payload["accepted_findings"][0]["status"],
                "summary": payload["accepted_findings"][0]["summary"],
                "evidence_kinds": [
                    item["kind"] for item in payload["accepted_findings"][0]["evidence"]
                ],
            },
            "purpose_profile": payload.get("purpose", {}).get("profile"),
            "prompt_contains_accepted_state": "Accepted findings" in payload["prompt"],
        }
    raise ValueError(f"unknown MCP response snapshot: {name}")

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def server() -> ZaxyMCPServer:
    """Return a server with mocked graph, tracer, and session manager."""
    with (
        patch("zaxy.mcp_server.build_projection_store") as mock_build_projection_store,
        patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
        patch("zaxy.mcp_server.SessionManager") as mock_session_cls,
    ):
        mock_graph = AsyncMock()
        mock_build_projection_store.return_value = mock_graph
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
        assert len(TOOLS) == 35

    def test_tool_names(self) -> None:
        """Tool names should match the expected contract."""
        names = {t.name for t in TOOLS}
        assert names == {
            "memory_append",
            "memory_query",
            "memory_causal_successors",
            "memory_causal_predecessors",
            "memory_consolidation_candidate",
            "memory_consolidation_review",
            "memory_consolidation_propose_from_log",
            "memory_consolidation_status",
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

    def test_v06_mcp_tool_contract_matches_snapshot(self) -> None:
        """The public MCP tool surface should stay protected by a canonical snapshot."""
        snapshot_path = Path("docs/examples/mcp-tool-contract.json")
        expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
        actual = {
            "tool_count": len(TOOLS),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in TOOLS
            ],
        }

        assert actual == expected

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
        assert "cursor" in tool.inputSchema["properties"]
        assert "paged" in tool.inputSchema["properties"]
        assert "session_ids" in tool.inputSchema["properties"]
        assert "temporal_filter" not in (tool.inputSchema.get("required") or [])

    def test_memory_feedback_has_required_identity_and_feedback(self) -> None:
        """memory_feedback should require a target entity and feedback value."""
        tool = next(t for t in TOOLS if t.name == "memory_feedback")
        assert tool.inputSchema["required"] == ["entity_name", "entity_type", "feedback"]
        assert "importance" in tool.inputSchema["properties"]
        assert "purpose" in tool.inputSchema["properties"]
        assert "outcome" in tool.inputSchema["properties"]

    def test_memory_synthesis_artifact_has_checkout_schema(self) -> None:
        """memory_synthesis_artifact should persist checkout answer candidates."""
        tool = next(t for t in TOOLS if t.name == "memory_synthesis_artifact")
        assert tool.inputSchema["required"] == ["checkout"]
        assert "candidate" in tool.inputSchema["properties"]
        assert "outcome" in tool.inputSchema["properties"]
        assert "session_id" not in tool.inputSchema["properties"]
        assert tool.inputSchema["properties"]["outcome"]["enum"] == [
            "used",
            "helpful",
            "rejected",
            "corrected",
            "excluded",
        ]

    def test_memory_synthesis_evidence_has_row_feedback_schema(self) -> None:
        """memory_synthesis_evidence should record one synthesis ledger-row outcome."""
        tool = next(t for t in TOOLS if t.name == "memory_synthesis_evidence")
        assert tool.inputSchema["required"] == ["checkout", "row", "outcome"]
        assert "candidate" in tool.inputSchema["properties"]
        assert tool.inputSchema["properties"]["outcome"]["enum"] == ["used", "helpful", "excluded"]

    def test_memory_skill_has_lifecycle_action_schema(self) -> None:
        """memory_skill should expose validated skill lifecycle capture."""
        tool = next(t for t in TOOLS if t.name == "memory_skill")
        assert tool.inputSchema["required"] == ["action", "skill_id"]
        assert tool.inputSchema["properties"]["action"]["enum"] == [
            "proposed",
            "validated",
            "revised",
            "deprecated",
            "contradicted",
            "applied",
            "outcome_recorded",
        ]
        assert "procedure" in tool.inputSchema["properties"]

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
        assert "purpose" in tool.inputSchema["properties"]

    def test_memory_capabilities_has_optional_query_schema(self) -> None:
        """memory_capabilities should expose model-facing Zaxy usage guidance."""
        tool = next(t for t in TOOLS if t.name == "memory_capabilities")
        assert tool.inputSchema["required"] == []
        assert "session_id" in tool.inputSchema["properties"]
        assert "current_task" in tool.inputSchema["properties"]

    def test_memory_bootstrap_has_optional_query_schema(self) -> None:
        """memory_bootstrap should expose the session-start model handoff contract."""
        tool = next(t for t in TOOLS if t.name == "memory_bootstrap")
        assert tool.inputSchema["required"] == []
        assert "session_id" in tool.inputSchema["properties"]
        assert "current_task" in tool.inputSchema["properties"]

    def test_v05_tool_descriptions_are_model_actionable(self) -> None:
        """Core MCP tools should describe when a model should call them."""
        descriptions = {tool.name: tool.description for tool in TOOLS}

        assert "session start" in descriptions["memory_bootstrap"].lower()
        assert "before substantial work" in descriptions["memory_checkout"].lower()
        assert "after using retrieved context" in descriptions["memory_feedback"].lower()
        assert "worker-local" in descriptions["coordination_report_finding"].lower()
        assert "accepted coordination state" in descriptions["coordination_checkout"].lower()
        assert "remote reviewer" in descriptions["coordination_approval_packet"].lower()

    def test_coordination_report_finding_schema_requires_mission_worker_and_summary(self) -> None:
        """coordination_report_finding should expose structured worker-local finding capture."""
        tool = next(t for t in TOOLS if t.name == "coordination_report_finding")
        assert tool.inputSchema["required"] == ["mission_id", "worker_id", "summary"]
        assert "evidence" in tool.inputSchema["properties"]
        assert "claim_key" in tool.inputSchema["properties"]

    def test_coordination_review_finding_schema_has_status_enum(self) -> None:
        """coordination_review_finding should restrict review states."""
        tool = next(t for t in TOOLS if t.name == "coordination_review_finding")
        assert tool.inputSchema["required"] == ["mission_id", "finding_id", "status"]
        assert tool.inputSchema["properties"]["status"]["enum"] == [
            "accepted",
            "rejected",
            "deferred",
            "conflicted",
        ]

    def test_coordination_checkout_schema_has_optional_diagnostics(self) -> None:
        """coordination_checkout should default to accepted state with opt-in diagnostics."""
        tool = next(t for t in TOOLS if t.name == "coordination_checkout")

        assert tool.inputSchema["required"] == ["mission_id"]
        assert tool.inputSchema["properties"]["include_diagnostics"]["type"] == "boolean"

    def test_coordination_performance_ledger_schema_requires_mission_id(self) -> None:
        """coordination_performance_ledger should expose worker-level outcome metrics."""
        tool = next(t for t in TOOLS if t.name == "coordination_performance_ledger")

        assert tool.inputSchema["required"] == ["mission_id"]

    def test_coordination_approval_tool_schemas(self) -> None:
        """Approval tools should expose packet export and decision application."""
        packet = next(t for t in TOOLS if t.name == "coordination_approval_packet")
        apply = next(t for t in TOOLS if t.name == "coordination_apply_approval")

        assert packet.inputSchema["required"] == ["mission_id"]
        assert apply.inputSchema["required"] == ["mission_id", "decisions"]
        assert apply.inputSchema["properties"]["decisions"]["type"] == "array"

    def test_coordination_record_synthesis_artifact_schema(self) -> None:
        """Coordinate proof packet tool should require mission scope and checkout."""
        tool = next(t for t in TOOLS if t.name == "coordination_record_synthesis_artifact")
        assert tool.inputSchema["required"] == ["mission_id", "checkout"]
        assert "decision_scope" in tool.inputSchema["properties"]
        assert "candidate" in tool.inputSchema["properties"]
        assert "outcome" in tool.inputSchema["properties"]

    def test_coordination_proof_trace_schema(self) -> None:
        """Coordinate proof trace should resolve replay chains by stable refs."""
        tool = next(t for t in TOOLS if t.name == "coordination_proof_trace")
        assert tool.inputSchema["required"] == ["mission_id"]
        assert "artifact_id" in tool.inputSchema["properties"]
        assert "handoff_id" in tool.inputSchema["properties"]
        assert tool.inputSchema["properties"]["proof_seq"]["minimum"] == 1

    def test_causal_and_consolidation_tools_are_registered(self) -> None:
        """Causal and consolidation tools should expose stable public schemas."""
        tools = {tool.name: tool for tool in TOOLS}

        successors = tools["memory_causal_successors"]
        predecessors = tools["memory_causal_predecessors"]
        assert successors.inputSchema["required"] == ["entity_name"]
        assert predecessors.inputSchema["required"] == ["entity_name"]
        for tool in (successors, predecessors):
            assert "relation_type" in tool.inputSchema["properties"]
            assert tool.inputSchema["properties"]["depth"]["default"] == 2
            assert tool.inputSchema["properties"]["depth"]["minimum"] == 1
            assert "session_id" in tool.inputSchema["properties"]

        candidate = tools["memory_consolidation_candidate"]
        assert candidate.inputSchema["required"] == [
            "candidate_type",
            "title",
            "summary",
            "source_events",
            "confidence",
            "method",
        ]
        assert candidate.inputSchema["properties"]["candidate_type"]["enum"] == [
            "episode",
            "claim",
            "procedure",
        ]
        source_events = candidate.inputSchema["properties"]["source_events"]
        assert source_events["minItems"] == 1
        assert source_events["items"]["properties"]["hash"]["pattern"] == "^[0-9a-f]{64}$"
        assert candidate.inputSchema["properties"]["confidence"]["minimum"] == 0
        assert candidate.inputSchema["properties"]["confidence"]["maximum"] == 1
        assert candidate.inputSchema["properties"]["actor"]["default"] == "zaxy-consolidation"

        propose_from_log = tools["memory_consolidation_propose_from_log"]
        assert propose_from_log.inputSchema["required"] == ["session_id"]
        assert propose_from_log.inputSchema["properties"]["actor"]["default"] == "zaxy-consolidation"
        assert propose_from_log.inputSchema["properties"]["window_size"]["minimum"] == 1
        assert propose_from_log.inputSchema["properties"]["window_size"]["maximum"] == 200
        assert "purpose" in propose_from_log.inputSchema["properties"]

        status = tools["memory_consolidation_status"]
        assert status.inputSchema["required"] == ["session_id"]

        review = tools["memory_consolidation_review"]
        assert review.inputSchema["required"] == ["candidate_id", "status", "rationale"]
        assert (
            review.inputSchema["properties"]["candidate_id"]["pattern"]
            == "^consolidation:(episode|claim|procedure):[0-9a-f]{24}$"
        )
        assert review.inputSchema["properties"]["status"]["enum"] == [
            "accepted",
            "rejected",
            "deferred",
            "conflicted",
        ]
        assert review.inputSchema["properties"]["actor"]["default"] == "zaxy-reviewer"


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

    @pytest.mark.parametrize(
        "arguments",
        [
            {"event_type": "", "actor": "user", "payload": {}},
            {"event_type": "x" * 257, "actor": "user", "payload": {}},
            {"event_type": "goal.created", "actor": "", "payload": {}},
            {"event_type": "goal.created", "actor": "x" * 257, "payload": {}},
            {"event_type": "goal.created", "actor": "user", "payload": ["not", "object"]},
        ],
    )
    async def test_rejects_unbounded_append_inputs(
        self,
        server: ZaxyMCPServer,
        arguments: dict[str, object],
    ) -> None:
        """memory_append should bound direct handler inputs as tightly as advertised schemas."""
        with pytest.raises(ValueError):
            await server.handle_memory_append(arguments)


class TestCausalAndConsolidationTools:
    """Tests for causal reads and consolidation MCP handlers."""

    async def test_causal_successors_calls_graph_read_path_and_serializes_results(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """memory_causal_successors should query configured graph causal neighbors."""
        result = CausalQueryResult(
            source={"name": "Plan", "entity_type": "task"},
            target={"name": "Implementation", "entity_type": "task"},
            relation_type="enabled",
            graph_relation_type="causal_enabled",
            confidence=0.82,
            method="explicit_outcome_citation_v1",
            citation="eventloom://agent-1/events/7#" + ("a" * 12),
            review_status="proposed",
            authority_status="non_authoritative",
            evidence={"source_event_seq": 7, "source_event_hash": "a" * 64},
            path_length=1,
        )
        server.graph.search_causal_neighbors.return_value = [
            GraphEntity(
                name="Implementation",
                entity_type="task",
                valid_from="2026-06-07T00:00:00Z",
                valid_to=None,
                session_id="agent-1",
                properties={
                    "causal_source_name": "Plan",
                    "causal_source_type": "task",
                    "causal_target_name": "Implementation",
                    "causal_target_type": "task",
                    "causal_relation_type": "enabled",
                    "graph_relation_type": "causal_enabled",
                    "confidence": 0.82,
                    "method": "explicit_outcome_citation_v1",
                    "review_status": "proposed",
                    "authority_status": "non_authoritative",
                    "session_id": "agent-1",
                    "source_event_seq": 7,
                    "source_event_hash": "a" * 64,
                    "_path_length": 1,
                },
            )
        ]

        response = await server.handle_memory_causal_successors({
            "entity_name": "Plan",
            "relation_type": "enabled",
            "depth": 3,
            "session_id": "agent-1",
        })

        payload = json_loads(response[0].text)
        assert payload == {"results": [result.to_dict()]}
        server.graph.search_causal_neighbors.assert_awaited_once_with(
            "Plan",
            direction="successors",
            relation_type="causal_enabled",
            depth=3,
            temporal_point=None,
            session_id="agent-1",
        )

    async def test_causal_predecessors_calls_graph_read_path_and_serializes_results(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """memory_causal_predecessors should use incoming causal direction."""
        server.graph.search_causal_neighbors.return_value = [
            GraphEntity(
                name="Plan",
                entity_type="task",
                valid_from="2026-06-07T00:00:00Z",
                valid_to=None,
                session_id="agent-1",
                properties={
                    "causal_source_name": "Plan",
                    "causal_source_type": "task",
                    "causal_target_name": "Implementation",
                    "causal_target_type": "task",
                    "causal_relation_type": "enabled",
                    "graph_relation_type": "causal_enabled",
                    "confidence": 0.82,
                    "method": "explicit_outcome_citation_v1",
                    "review_status": "proposed",
                    "authority_status": "non_authoritative",
                    "session_id": "agent-1",
                    "source_event_seq": 7,
                    "source_event_hash": "a" * 64,
                },
            )
        ]

        response = await server.handle_memory_causal_predecessors({
            "entity_name": "Implementation",
            "session_id": "agent-1",
        })

        payload = json_loads(response[0].text)
        assert payload["results"][0]["source"]["name"] == "Plan"
        assert payload["results"][0]["target"]["name"] == "Implementation"
        server.graph.search_causal_neighbors.assert_awaited_once_with(
            "Implementation",
            direction="predecessors",
            relation_type=None,
            depth=2,
            temporal_point=None,
            session_id="agent-1",
        )

    @pytest.mark.parametrize(
        ("arguments", "match"),
        [
            ({"entity_name": "Plan", "depth": 0, "session_id": "agent-1"}, "depth must be between 1"),
            ({"entity_name": "Plan", "depth": True, "session_id": "agent-1"}, "depth must be an integer"),
            ({"entity_name": 42, "session_id": "agent-1"}, "query must be a non-empty string"),
            ({"entity_name": "x" * 4097, "session_id": "agent-1"}, "query exceeds 4096 characters"),
            (
                {"entity_name": "Plan", "relation_type": "enables", "session_id": "agent-1"},
                "causal relation_type",
            ),
        ],
    )
    async def test_causal_handlers_reject_invalid_relation_and_depth(
        self,
        server: ZaxyMCPServer,
        arguments: dict[str, object],
        match: str,
    ) -> None:
        """Causal handler validation should fail before graph access."""
        with pytest.raises(ValueError, match=match):
            await server.handle_memory_causal_successors(arguments)

        server.graph.search_causal_neighbors.assert_not_awaited()

    async def test_consolidation_candidate_appends_projects_traces_and_returns_event_ref(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """memory_consolidation_candidate should append and immediately project the candidate."""
        source_hash = "b" * 64
        extraction = MagicMock()
        with patch("zaxy.mcp_server.extract", return_value=extraction) as mock_extract:
            response = await server.handle_memory_consolidation_candidate({
                "candidate_type": "claim",
                "title": "Retry policy",
                "summary": "Retries should preserve original citations.",
                "source_events": [{"seq": 7, "hash": source_hash}],
                "confidence": 0.82,
                "method": "manual-review",
                "purpose": "release audit",
                "session_id": "agent-1",
                "actor": "assistant",
            })

        payload = json_loads(response[0].text)
        assert payload == {"seq": 1, "hash": "a" * 64}
        server.session_manager.get.assert_called_once_with("agent-1")
        append_call = server.session_manager.get.return_value.eventlog.append.call_args
        assert append_call.args == ("consolidation.candidate.created",)
        assert append_call.kwargs["actor"] == "assistant"
        assert append_call.kwargs["thread"] == "agent-1"
        event_payload = append_call.kwargs["payload"]
        assert event_payload["candidate_type"] == "claim"
        assert event_payload["review_status"] == "pending"
        assert event_payload["authority_status"] == "non_authoritative"
        assert event_payload["source_events"] == [{"seq": 7, "hash": source_hash}]
        mock_extract.assert_called_once_with(server.session_manager.get.return_value.eventlog.append.return_value)
        server.graph.upsert_extraction.assert_awaited_once_with(extraction, session_id="agent-1")
        server.tracer.trace_append.assert_awaited_once_with(
            "consolidation.candidate.created",
            "assistant",
            1,
        )

    async def test_consolidation_propose_from_log_uses_configured_fabric_path(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """memory_consolidation_propose_from_log should use the server's configured fabric path."""
        expected = {
            "session_id": "agent-1",
            "segments_considered": 2,
            "candidates_created": 3,
        }
        fabric = AsyncMock()
        fabric.propose_consolidation_candidates.return_value = expected

        with patch("zaxy.mcp_server.MemoryFabric", return_value=fabric) as fabric_cls:
            response = await server.handle_memory_consolidation_propose_from_log({
                "session_id": "agent-1",
                "actor": "assistant",
                "purpose": "release audit",
                "window_size": 5,
            })

        assert json_loads(response[0].text) == expected
        fabric_cls.assert_called_once()
        assert fabric_cls.call_args.kwargs["eventloom_path"] == server._eventloom_path
        fabric.connect.assert_awaited_once()
        fabric.propose_consolidation_candidates.assert_awaited_once_with(
            session_id="agent-1",
            actor="assistant",
            purpose="release audit",
            window_size=5,
        )
        fabric.close.assert_awaited_once()

    async def test_consolidation_status_uses_configured_fabric_path(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """memory_consolidation_status should read candidate review state through MemoryFabric."""
        expected = {"session_id": "agent-1", "pending": 2, "accepted": 1}
        fabric = AsyncMock()
        fabric.consolidation_status.return_value = expected

        with patch("zaxy.mcp_server.MemoryFabric", return_value=fabric) as fabric_cls:
            response = await server.handle_memory_consolidation_status({"session_id": "agent-1"})

        assert json_loads(response[0].text) == expected
        fabric_cls.assert_called_once()
        assert fabric_cls.call_args.kwargs["eventloom_path"] == server._eventloom_path
        fabric.connect.assert_awaited_once()
        fabric.consolidation_status.assert_awaited_once_with(session_id="agent-1")
        fabric.close.assert_awaited_once()

    async def test_consolidation_propose_from_log_rejects_invalid_window_before_fabric(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """window_size should stay bounded at the MCP handler boundary."""
        with (
            patch("zaxy.mcp_server.MemoryFabric") as fabric_cls,
            pytest.raises(ValueError, match="window_size must be between 1 and 200"),
        ):
            await server.handle_memory_consolidation_propose_from_log({
                "session_id": "agent-1",
                "window_size": 0,
            })

        fabric_cls.assert_not_called()

    async def test_consolidation_candidate_leaves_source_event_validation_to_builder(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """Invalid source_events should fail under the strict consolidation builder contract."""
        with pytest.raises(ValueError, match=r"source_events\[0\]\.hash"):
            await server.handle_memory_consolidation_candidate({
                "candidate_type": "claim",
                "title": "Retry policy",
                "summary": "Retries should preserve original citations.",
                "source_events": [{"seq": 7, "hash": "not-a-hash"}],
                "confidence": 0.82,
                "method": "manual-review",
                "session_id": "agent-1",
            })

        server.session_manager.get.assert_not_called()

    @pytest.mark.parametrize("field", ["actor", "candidate_type", "title", "summary", "method", "purpose"])
    async def test_consolidation_candidate_rejects_non_string_contract_fields(
        self,
        server: ZaxyMCPServer,
        field: str,
    ) -> None:
        """String fields should not be silently coerced before candidate event building."""
        arguments: dict[str, object] = {
            "candidate_type": "claim",
            "title": "Retry policy",
            "summary": "Retries should preserve original citations.",
            "source_events": [{"seq": 7, "hash": "b" * 64}],
            "confidence": 0.82,
            "method": "manual-review",
            "purpose": "release audit",
            "session_id": "agent-1",
            "actor": "assistant",
        }
        arguments[field] = {"not": "text"}

        with pytest.raises(ValueError, match=rf"{field} must be a string"):
            await server.handle_memory_consolidation_candidate(arguments)

        server.session_manager.get.assert_not_called()

    async def test_consolidation_review_appends_projects_traces_without_authority_promotion(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """memory_consolidation_review should record review lifecycle without promoting authority."""
        candidate_id = "consolidation:claim:" + ("c" * 24)
        extraction = MagicMock()
        with patch("zaxy.mcp_server.extract", return_value=extraction):
            response = await server.handle_memory_consolidation_review({
                "candidate_id": candidate_id,
                "status": "accepted",
                "rationale": "Citations match the source events.",
                "session_id": "agent-1",
                "actor": "reviewer",
            })

        assert json_loads(response[0].text) == {"seq": 1, "hash": "a" * 64}
        append_call = server.session_manager.get.return_value.eventlog.append.call_args
        assert append_call.args == ("consolidation.candidate.reviewed",)
        assert append_call.kwargs["actor"] == "reviewer"
        assert append_call.kwargs["thread"] == "agent-1"
        assert append_call.kwargs["payload"] == {
            "candidate_id": candidate_id,
            "status": "accepted",
            "authority_status": "non_authoritative",
            "rationale": "Citations match the source events.",
        }
        server.graph.upsert_extraction.assert_awaited_once_with(extraction, session_id="agent-1")
        server.tracer.trace_append.assert_awaited_once_with(
            "consolidation.candidate.reviewed",
            "reviewer",
            1,
        )

    @pytest.mark.parametrize("field", ["actor", "candidate_id", "status", "rationale"])
    async def test_consolidation_review_rejects_non_string_contract_fields(
        self,
        server: ZaxyMCPServer,
        field: str,
    ) -> None:
        """Review contract string fields should fail before append/project on non-string values."""
        arguments: dict[str, object] = {
            "candidate_id": "consolidation:claim:" + ("c" * 24),
            "status": "accepted",
            "rationale": "Citations match the source events.",
            "session_id": "agent-1",
            "actor": "reviewer",
        }
        arguments[field] = ["not", "text"]

        with pytest.raises(ValueError, match=rf"{field} must be a string"):
            await server.handle_memory_consolidation_review(arguments)

        server.session_manager.get.assert_not_called()

    @pytest.mark.parametrize(
        ("handler_name", "arguments"),
        [
            ("handle_memory_causal_successors", {"entity_name": "Plan", "session_id": "agent-2"}),
            ("handle_memory_consolidation_propose_from_log", {"session_id": "agent-2"}),
            ("handle_memory_consolidation_status", {"session_id": "agent-2"}),
            (
                "handle_memory_consolidation_review",
                {
                    "candidate_id": "consolidation:claim:" + ("c" * 24),
                    "status": "deferred",
                    "rationale": "Needs source review.",
                    "session_id": "agent-2",
                },
            ),
        ],
    )
    async def test_remote_scope_rejects_cross_session_causal_and_consolidation_calls(
        self,
        server: ZaxyMCPServer,
        handler_name: str,
        arguments: dict[str, object],
    ) -> None:
        """Remote scoped calls should not cross into a different session."""
        token = remote_session_scope.set("agent-1")
        try:
            with pytest.raises(PermissionError, match="session scope"):
                await getattr(server, handler_name)(arguments)
        finally:
            remote_session_scope.reset(token)

    @pytest.mark.parametrize(
        ("tool_name", "handler_name"),
        [
            ("memory_causal_successors", "handle_memory_causal_successors"),
            ("memory_causal_predecessors", "handle_memory_causal_predecessors"),
            ("memory_consolidation_candidate", "handle_memory_consolidation_candidate"),
            ("memory_consolidation_propose_from_log", "handle_memory_consolidation_propose_from_log"),
            ("memory_consolidation_status", "handle_memory_consolidation_status"),
            ("memory_consolidation_review", "handle_memory_consolidation_review"),
        ],
    )
    async def test_causal_and_consolidation_dispatch_routes_to_handlers(
        self,
        server: ZaxyMCPServer,
        tool_name: str,
        handler_name: str,
    ) -> None:
        """_dispatch_tool_call should route all new public tools."""
        expected = [MagicMock()]
        handler = AsyncMock(return_value=expected)
        setattr(server, handler_name, handler)

        result = await zaxy.mcp_server._dispatch_tool_call(server, tool_name, {"entity_name": "Plan"})

        assert result == expected
        handler.assert_awaited_once_with({"entity_name": "Plan"})


class TestCoordinationTools:
    """Tests for high-level coordination MCP tools."""

    async def test_coordination_tools_append_project_and_brief(
        self,
        tmp_path: Path,
    ) -> None:
        """MCP coordination tools should preserve parent/worker isolation and return briefs."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        server.graph = AsyncMock()
        server.tracer = AsyncMock()

        result = await server.handle_coordination_start({
            "mission_id": "auth-main",
            "objective": "Ship auth refactor",
            "actor": "lead",
        })
        assert json_loads(result[0].text)["event_type"] == "coordination.mission.created"

        await server.handle_coordination_worker_create({
            "mission_id": "auth-main",
            "worker_id": "auth-api",
            "actor": "lead",
        })
        result = await server.handle_coordination_report_finding({
            "mission_id": "auth-main",
            "worker_id": "auth-api",
            "summary": "API failures trace to expired JWKS cache handling.",
            "actor": "auth-api-agent",
            "evidence": [{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
            "claim_key": "auth.failure.cause",
            "claim_value": "expired-jwks-cache",
        })
        finding_id = json_loads(result[0].text)["finding_id"]
        await server.handle_coordination_review_finding({
            "mission_id": "auth-main",
            "finding_id": finding_id,
            "status": "accepted",
            "actor": "lead",
        })
        await server.handle_coordination_promote({
            "mission_id": "auth-main",
            "finding_id": finding_id,
            "actor": "lead",
        })

        result = await server.handle_coordination_merge_brief({"mission_id": "auth-main"})
        brief = json_loads(result[0].text)

        assert brief["mission_id"] == "auth-main"
        assert brief["accepted_findings"][0]["finding_id"] == finding_id
        assert brief["accepted_findings"][0]["worker_id"] == "auth-api"
        assert brief["pending_findings"] == []
        assert server.graph.upsert_extraction.await_count == 5

        result = await server.handle_coordination_checkout({"mission_id": "auth-main"})
        checkout = json_loads(result[0].text)
        assert checkout["accepted_findings"][0]["finding_id"] == finding_id
        assert checkout["pending_findings"] == []
        assert "API failures trace to expired JWKS cache handling." in checkout["prompt"]

        snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
        assert _mcp_response_snapshot("coordination_checkout", checkout) == snapshots["coordination_checkout"]

        result = await server.handle_coordination_performance_ledger({"mission_id": "auth-main"})
        ledger = json_loads(result[0].text)
        assert ledger["workers"][0]["worker_id"] == "auth-api"
        assert ledger["workers"][0]["accepted_findings"] == 1
        assert ledger["workers"][0]["test_backed_findings"] == 1

        result = await server.handle_coordination_approval_packet({"mission_id": "auth-main"})
        packet = json_loads(result[0].text)
        assert packet["mission_id"] == "auth-main"

        result = await server.handle_coordination_handoff({
            "mission_id": "auth-main",
            "summary": "Auth mission complete.",
            "next_steps": ["Release branch"],
            "risks": ["Token cache metrics are sparse"],
            "actor": "lead",
        })
        handoff = json_loads(result[0].text)
        assert handoff["event_type"] == "coordination.handoff.created"
        assert handoff["handoff_id"].startswith("auth-main:handoff:")
        assert handoff["next_steps"] == ["Release branch"]
        assert handoff["risks"] == ["Token cache metrics are sparse"]
        assert server.graph.upsert_extraction.await_count == 6

    async def test_coordination_apply_approval_reviews_and_promotes(self, tmp_path: Path) -> None:
        """MCP approval application should append review and promotion events."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        server.graph = AsyncMock()
        server.tracer = AsyncMock()

        await server.handle_coordination_start({"mission_id": "auth-main", "objective": "Ship auth refactor"})
        await server.handle_coordination_worker_create({"mission_id": "auth-main", "worker_id": "auth-api"})
        result = await server.handle_coordination_report_finding({
            "mission_id": "auth-main",
            "worker_id": "auth-api",
            "summary": "Expired JWKS cache causes API failures.",
            "evidence": [{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        })
        finding_id = json_loads(result[0].text)["finding_id"]

        result = await server.handle_coordination_apply_approval({
            "mission_id": "auth-main",
            "decisions": [{"finding_id": finding_id, "status": "accepted", "rationale": "Remote approval.", "promote": True}],
            "actor": "reviewer",
        })

        payload = json_loads(result[0].text)
        assert payload["reviewed_count"] == 1
        assert payload["promoted_count"] == 1
        assert server.graph.upsert_extraction.await_count == 5

    async def test_coordination_record_synthesis_artifact_returns_proof_packet(self, tmp_path: Path) -> None:
        """MCP Coordinate synthesis should append artifact and proof packet events."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        server.graph = AsyncMock()
        server.tracer = AsyncMock()

        await server.handle_coordination_start({"mission_id": "release-rc1", "objective": "Ship release"})
        await server.handle_coordination_worker_create({"mission_id": "release-rc1", "worker_id": "auth-api"})
        result = await server.handle_coordination_report_finding({
            "mission_id": "release-rc1",
            "worker_id": "auth-api",
            "summary": "Expired JWKS cache causes API failures.",
            "evidence": [{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        })
        finding_id = json_loads(result[0].text)["finding_id"]
        await server.handle_coordination_review_finding({
            "mission_id": "release-rc1",
            "finding_id": finding_id,
            "status": "accepted",
            "actor": "lead",
        })
        await server.handle_coordination_promote({
            "mission_id": "release-rc1",
            "finding_id": finding_id,
            "actor": "lead",
        })
        handoff_result = await server.handle_coordination_handoff({
            "mission_id": "release-rc1",
            "summary": "Release handoff ready.",
            "actor": "lead",
        })
        handoff_id = json_loads(handoff_result[0].text)["handoff_id"]

        result = await server.handle_coordination_record_synthesis_artifact({
            "mission_id": "release-rc1",
            "decision_scope": "handoff",
            "handoff_id": handoff_id,
            "actor": "coordinator",
            "checkout": {
                "session_id": "release-rc1",
                "query": "Compose accepted release findings.",
                "prompt": "# Memory Checkout",
                "working_set": {},
                "ref": None,
                "current_facts": [],
                "evidence": [],
                "provenance": [],
                "retention": {},
                "warnings": [],
                "guidance": {},
                "quality": {"answerability": "answer_from_memory", "confidence": 0.9},
                "diagnostics": {
                    "synthesis": {
                        "answer_candidates": [
                            {
                                "rank": 1,
                                "type": "coordinate_handoff",
                                "answer": "Accepted cause: expired JWKS cache.",
                                "support_source_ids": [finding_id],
                            }
                        ],
                        "ledger_rows": [{"fact_id": finding_id, "source_group": finding_id}],
                    }
                },
                "context_counts": {},
                "replay_event_count": 0,
                "compacted": False,
                "assembly_policy": {},
            },
        })

        payload = json_loads(result[0].text)
        assert payload["artifact_event"]["event_type"] == "memory.synthesis.artifact.created"
        assert payload["proof_event"]["event_type"] == "coordination.proof_packet.created"
        assert payload["proof_packet"]["mission_id"] == "release-rc1"
        assert payload["proof_packet"]["decision_scope"] == "handoff"
        assert payload["proof_packet"]["accepted_finding_ids"] == [finding_id]
        assert payload["proof_packet"]["handoff_event_ref"]["handoff_id"] == handoff_id
        assert payload["proof_packet"]["non_authoritative_rows"] == []

        trace_result = await server.handle_coordination_proof_trace({
            "mission_id": "release-rc1",
            "handoff_id": handoff_id,
        })
        trace = json_loads(trace_result[0].text)
        assert trace["proof_event"]["seq"] == payload["proof_event"]["seq"]
        assert trace["artifact_event"]["seq"] == payload["artifact_event"]["seq"]
        assert trace["handoff_event"]["event_type"] == "coordination.handoff.created"
        assert trace["accepted_finding_ids"] == [finding_id]
        assert trace["answer_candidates"][0]["answer"] == "Accepted cause: expired JWKS cache."
        assert trace["ledger_rows"][0]["source_group"] == finding_id

    async def test_coordination_record_synthesis_artifact_rejects_unknown_handoff_without_appending(
        self,
        tmp_path: Path,
    ) -> None:
        """MCP handoff proof validation should run before artifact or outcome writes."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        server.graph = AsyncMock()
        server.tracer = AsyncMock()

        await server.handle_coordination_start({"mission_id": "release-rc1", "objective": "Ship release"})
        before = len(server.session_manager.replay("release-rc1").events)
        checkout = {
            "session_id": "release-rc1",
            "query": "Compose accepted release findings.",
            "prompt": "# Memory Checkout",
            "working_set": {},
            "ref": None,
            "current_facts": [],
            "evidence": [],
            "provenance": [],
            "retention": {},
            "warnings": [],
            "guidance": {},
            "quality": {"answerability": "answer_from_memory", "confidence": 0.9},
            "diagnostics": {
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
            "context_counts": {},
            "replay_event_count": 0,
            "compacted": False,
            "assembly_policy": {},
        }

        with pytest.raises(ValueError, match="Unknown handoff_id"):
            await server.handle_coordination_record_synthesis_artifact({
                "mission_id": "release-rc1",
                "decision_scope": "handoff",
                "handoff_id": "release-rc1:handoff:missing",
                "actor": "coordinator",
                "checkout": checkout,
                "candidate": checkout["diagnostics"]["synthesis"]["answer_candidates"][0],
                "outcome": "used",
            })

        events = server.session_manager.replay("release-rc1").events
        assert len(events) == before
        assert not any(event.type == "memory.synthesis.artifact.created" for event in events)
        assert not any(event.type == "memory.synthesis.used" for event in events)
        assert not any(event.type == "coordination.proof_packet.created" for event in events)

    async def test_coordination_record_synthesis_artifact_rejects_foreign_candidate_without_appending(
        self,
        tmp_path: Path,
    ) -> None:
        """MCP Coordinate proof calls should reject candidates outside checkout diagnostics."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        server.graph = AsyncMock()
        server.tracer = AsyncMock()

        await server.handle_coordination_start({"mission_id": "release-rc1", "objective": "Ship release"})
        before = len(server.session_manager.replay("release-rc1").events)
        checkout = {
            "session_id": "release-rc1",
            "query": "Compose accepted release findings.",
            "prompt": "# Memory Checkout",
            "working_set": {},
            "ref": None,
            "current_facts": [],
            "evidence": [],
            "provenance": [],
            "retention": {},
            "warnings": [],
            "guidance": {},
            "quality": {"answerability": "answer_from_memory", "confidence": 0.9},
            "diagnostics": {
                "synthesis": {
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "coordinate_handoff",
                            "answer": "Accepted cause: expired JWKS cache.",
                            "support_source_ids": ["finding-api"],
                        }
                    ],
                    "ledger_rows": [],
                }
            },
            "context_counts": {},
            "replay_event_count": 0,
            "compacted": False,
            "assembly_policy": {},
        }

        with pytest.raises(ValueError, match="diagnostics.synthesis.answer_candidates"):
            await server.handle_coordination_record_synthesis_artifact({
                "mission_id": "release-rc1",
                "checkout": checkout,
                "candidate": {
                    "rank": 1,
                    "type": "coordinate_handoff",
                    "answer": "Accepted cause: expired JWKS cache.",
                    "support_source_ids": ["foreign-finding"],
                },
                "outcome": "used",
            })

        events = server.session_manager.replay("release-rc1").events
        assert len(events) == before
        assert not any(event.type == "memory.synthesis.artifact.created" for event in events)
        assert not any(event.type == "memory.synthesis.used" for event in events)
        assert not any(event.type == "coordination.proof_packet.created" for event in events)

    async def test_coordination_manager_uses_configured_semantic_detector(self, tmp_path: Path) -> None:
        """MCP coordination briefs should use the configured semantic conflict factory."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        server.graph = AsyncMock()
        server.tracer = AsyncMock()
        server._settings = Settings(
            _env_file=None,
            coordination_semantic_conflict_provider="lexical",
        )

        await server.handle_coordination_start({"mission_id": "auth-main", "objective": "Ship auth refactor"})
        await server.handle_coordination_worker_create({"mission_id": "auth-main", "worker_id": "auth-api"})
        await server.handle_coordination_worker_create({"mission_id": "auth-main", "worker_id": "auth-ui"})
        await server.handle_coordination_report_finding({
            "mission_id": "auth-main",
            "worker_id": "auth-api",
            "summary": "Token refresh retry is enabled in auth middleware.",
        })
        await server.handle_coordination_report_finding({
            "mission_id": "auth-main",
            "worker_id": "auth-ui",
            "summary": "Token refresh retry is disabled in browser session handling.",
        })

        result = await server.handle_coordination_merge_brief({"mission_id": "auth-main"})

        brief = json_loads(result[0].text)
        assert brief["conflicts"][0]["conflict_type"] == "semantic"
        assert brief["conflicts"][0]["reason"] == "local_lexical_contradiction:disabled/enabled"

    async def test_coordination_remote_scope_rejects_cross_session_start(self, tmp_path: Path) -> None:
        """Remote MCP sessions must not write arbitrary parent mission sessions."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        token = remote_session_scope.set("client-session")
        try:
            with pytest.raises(PermissionError, match="session scope"):
                await server.handle_coordination_start({
                    "mission_id": "other-session",
                    "objective": "Not allowed",
                })
        finally:
            remote_session_scope.reset(token)


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


class TestMemoryBootstrap:
    """Tests for the compact session-start memory bootstrap."""

    async def test_returns_bootstrap_packet_for_session_start(
        self,
        server: ZaxyMCPServer,
        tmp_path: Path,
    ) -> None:
        """memory_bootstrap should make Zaxy natural to use at session start."""
        eventloom = tmp_path / ".eventloom"
        EventLog(eventloom / "agent-1.jsonl").append(
            "task.completed",
            actor="codex",
            payload={"summary": "Bootstrap source."},
            thread="agent-1",
        )
        server._eventloom_path = str(eventloom)
        server._workspace_root = tmp_path

        result = await server.handle_memory_bootstrap(
            {"session_id": "agent-1", "current_task": "resume roadmap"}
        )

        payload = json_loads(result[0].text)
        assert payload["mode"] == "session_start"
        assert payload["session_id"] == "agent-1"
        assert payload["startup_sequence"][0]["tool"] == "memory_capabilities"
        assert payload["startup_sequence"][1]["tool"] == "memory_checkout"
        assert payload["startup_sequence"][1]["arguments"]["query"] == "resume roadmap"
        assert payload["capabilities"]["status"]["eventloom"]["latest_seq"] == 1
        assert "Call memory_checkout before answering roadmap or implementation questions." in payload["prompt"]
        events = EventLog(eventloom / "agent-1.jsonl").read_all()
        assert events[-1].type == "memory.bootstrap.shown"
        assert events[-1].payload["source"] == "mcp"

    async def test_bootstrap_response_matches_v06_snapshot(
        self,
        server: ZaxyMCPServer,
        tmp_path: Path,
    ) -> None:
        """memory_bootstrap should keep a representative response snapshot stable."""
        eventloom = tmp_path / ".eventloom"
        EventLog(eventloom / "agent-1.jsonl").append(
            "task.completed",
            actor="codex",
            payload={"summary": "Bootstrap source."},
            thread="agent-1",
        )
        server._eventloom_path = str(eventloom)
        server._workspace_root = tmp_path

        result = await server.handle_memory_bootstrap(
            {"session_id": "agent-1", "current_task": "resume roadmap"}
        )

        snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
        payload = json_loads(result[0].text)
        assert _mcp_response_snapshot("memory_bootstrap", payload) == snapshots["memory_bootstrap"]


def test_mcp_server_constructs_projection_store_through_factory(tmp_path: Path) -> None:
    """ZaxyMCPServer should use the backend-neutral projection factory."""
    with (
        patch("zaxy.mcp_server.build_projection_store") as mock_build,
        patch("zaxy.mcp_server.MemoryTracer"),
        patch("zaxy.mcp_server.SessionManager"),
        patch("zaxy.mcp_server.LocalEmbeddedGraphRuntime"),
    ):
        mock_build.return_value = AsyncMock()

        srv = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))

    assert srv.graph is mock_build.return_value
    assert mock_build.call_args.args[0].backend == "embedded"
    assert mock_build.call_args.args[0].embedded_graph_path == tmp_path / ".eventloom" / "projections" / "embedded.kuzu"


def test_mcp_server_accepts_embedded_projection_overrides(tmp_path: Path) -> None:
    """ZaxyMCPServer should honor the embedded projection profile passed by zaxy serve."""
    embedded_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    with (
        patch("zaxy.mcp_server.build_projection_store") as mock_build,
        patch("zaxy.mcp_server.MemoryTracer"),
        patch("zaxy.mcp_server.SessionManager"),
    ):
        mock_build.return_value = AsyncMock()

        srv = ZaxyMCPServer(
            eventloom_path=str(tmp_path / ".eventloom"),
            projection_backend="embedded",
            embedded_graph_path=embedded_path,
        )

    assert srv.graph is mock_build.return_value
    config = mock_build.call_args.args[0]
    assert config.backend == "embedded"
    assert config.embedded_graph_path == embedded_path
    assert srv.local_projection_runtime.path == embedded_path


class TestSessionDefaults:
    """Tests for session default behavior."""

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
            payload = json.loads(data)
            snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
            assert _mcp_response_snapshot("memory_query", payload) == snapshots["memory_query"]

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
        snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
        assert _mcp_response_snapshot("memory_verbatim", payload) == snapshots["memory_verbatim"]

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

    async def test_paged_query_returns_continuation_cursor(self, server: ZaxyMCPServer) -> None:
        """Paged memory_query should return an object with an opaque continuation cursor."""
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            mock_router = AsyncMock()
            mock_router.query.return_value = [
                MagicMock(
                    content="alpha",
                    source="keyword",
                    score=0.9,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation={},
                ),
                MagicMock(
                    content="beta",
                    source="keyword",
                    score=0.8,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation={},
                ),
                MagicMock(
                    content="gamma",
                    source="keyword",
                    score=0.7,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation={},
                ),
            ]
            mock_router_cls.return_value = mock_router

            first = await server.handle_memory_query({"query": "roadmap", "limit": 2, "paged": True})
            payload = json.loads(first[0].text)

            assert [row["content"] for row in payload["contexts"]] == ["alpha", "beta"]
            assert payload["next_cursor"]
            assert payload["has_more"] is True
            call = mock_router.query.await_args
            assert call.kwargs["limit"] == 3

    async def test_paged_query_continues_without_repeating_results(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """A returned cursor should continue from the next ranked item."""
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            mock_router = AsyncMock()
            mock_router.query.return_value = [
                MagicMock(
                    content="alpha",
                    source="keyword",
                    score=0.9,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation={},
                ),
                MagicMock(
                    content="beta",
                    source="keyword",
                    score=0.8,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation={},
                ),
                MagicMock(
                    content="gamma",
                    source="keyword",
                    score=0.7,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation={},
                ),
            ]
            mock_router_cls.return_value = mock_router

            first = await server.handle_memory_query({"query": "roadmap", "limit": 2, "paged": True})
            cursor = json.loads(first[0].text)["next_cursor"]
            second = await server.handle_memory_query({"query": "roadmap", "limit": 2, "cursor": cursor})
            payload = json.loads(second[0].text)

            assert [row["content"] for row in payload["contexts"]] == ["gamma"]
            assert payload["next_cursor"] is None
            assert payload["has_more"] is False

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

    async def test_local_cross_session_query_merges_scoped_results(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """Local clients should be able to request an explicit cross-session scope."""
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            first_router = AsyncMock()
            first_router.query.return_value = [
                MagicMock(
                    content="agent one decision",
                    source="keyword",
                    score=0.7,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation={},
                )
            ]
            second_router = AsyncMock()
            second_router.query.return_value = [
                MagicMock(
                    content="agent two decision",
                    source="keyword",
                    score=0.9,
                    valid_from=None,
                    valid_to=None,
                    citation=None,
                    score_explanation={},
                )
            ]
            mock_router_cls.side_effect = [first_router, second_router]

            result = await server.handle_memory_query(
                {"query": "decision", "session_ids": ["agent-1", "agent-2"], "limit": 2}
            )
            payload = json.loads(result[0].text)

            assert [row["content"] for row in payload] == [
                "agent two decision",
                "agent one decision",
            ]
            assert [row["session_id"] for row in payload] == ["agent-2", "agent-1"]

    async def test_remote_scope_rejects_cross_session_ids(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """Remote clients must not fan out across arbitrary sessions."""
        token = remote_session_scope.set("client-session")
        try:
            with pytest.raises(PermissionError, match="cross-session"):
                await server.handle_memory_query(
                    {"query": "decision", "session_ids": ["client-session", "other-session"]}
                )
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

    async def test_feedback_preserves_purpose_and_outcome(self, server: ZaxyMCPServer) -> None:
        """MCP feedback should record what purpose the memory helped satisfy."""
        await server.handle_memory_feedback({
            "entity_name": "accepted release finding",
            "entity_type": "accepted_finding",
            "feedback": "used",
            "actor": "coordinator",
            "session_id": "release-rc1",
            "purpose": "coordinate",
            "outcome": "supported_handoff",
        })

        payload = server.session_manager.get.return_value.eventlog.append.call_args.kwargs["payload"]
        assert payload["purpose"]["profile"] == "coordinate"
        assert payload["purpose"]["expected_action"] == "brief_promote_or_handoff"
        assert payload["outcome"] == "supported_handoff"

    async def test_memory_feedback_response_matches_v06_snapshot(self, server: ZaxyMCPServer) -> None:
        """memory_feedback should keep its client-facing reinforcement response stable."""
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

        snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
        payload = json_loads(result[0].text)
        assert _mcp_response_snapshot("memory_feedback", payload) == snapshots["memory_feedback"]

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


class TestMemorySynthesisArtifact:
    """Tests for memory_synthesis_artifact handler."""

    def _checkout_payload(self) -> dict[str, object]:
        return {
            "session_id": "agent-1",
            "query": "How much did I spend on bike expenses in total?",
            "prompt": "# Memory Checkout",
            "working_set": {},
            "ref": None,
            "current_facts": [],
            "evidence": [
                {
                    "content": "session_id=answer-1 I spent $120 on a bike helmet.",
                    "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                    "source_lane": "verbatim",
                }
            ],
            "provenance": [],
            "retention": {},
            "warnings": [],
            "guidance": {},
            "quality": {"answerability": "answer_from_memory", "confidence": 0.86},
            "diagnostics": {
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
            "context_counts": {},
            "replay_event_count": 0,
            "compacted": False,
            "assembly_policy": {},
        }

    async def test_appends_synthesis_artifact_and_candidate_outcome(self, server: ZaxyMCPServer) -> None:
        """MCP should persist checkout answer candidates through Eventloom."""
        result = await server.handle_memory_synthesis_artifact({
            "checkout": self._checkout_payload(),
            "candidate": {"rank": 1, "type": "currency", "answer": "$120", "support_source_ids": ["answer-1"]},
            "outcome": "used",
            "actor": "assistant",
            "reason": "answer used in final response",
        })

        calls = server.session_manager.get.return_value.eventlog.append.call_args_list
        assert calls[0].args == ("memory.synthesis.artifact.created",)
        assert calls[0].kwargs["actor"] == "assistant"
        assert calls[0].kwargs["thread"] == "agent-1"
        assert calls[0].kwargs["payload"]["schema_version"] == "synthesis_artifact_v1"
        assert calls[1].args == ("memory.synthesis.used",)
        assert calls[1].kwargs["payload"]["reason"] == "answer used in final response"
        assert server.graph.upsert_extraction.await_count == 2
        output = json_loads(result[0].text)
        assert output["artifact_event"]["event_type"] == "memory.synthesis.artifact.created"
        assert output["candidate_event"]["event_type"] == "memory.synthesis.used"
        assert output["candidate_event"]["outcome"] == "used"
        assert output["artifact_id"].startswith("sha256:")

    async def test_response_matches_snapshot(self, server: ZaxyMCPServer) -> None:
        """memory_synthesis_artifact should keep a stable compact response."""
        result = await server.handle_memory_synthesis_artifact({
            "checkout": self._checkout_payload(),
            "candidate": {"rank": 1, "type": "currency", "answer": "$120", "support_source_ids": ["answer-1"]},
            "outcome": "used",
            "actor": "assistant",
        })

        snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
        payload = json_loads(result[0].text)
        assert _mcp_response_snapshot("memory_synthesis_artifact", payload) == snapshots["memory_synthesis_artifact"]

    async def test_rejects_checkout_without_answer_candidates(self, server: ZaxyMCPServer) -> None:
        """MCP should fail closed instead of writing empty synthesis artifacts."""
        checkout = self._checkout_payload()
        diagnostics = checkout["diagnostics"]
        assert isinstance(diagnostics, dict)
        diagnostics["synthesis"] = {"answer_candidates": []}

        with pytest.raises(ValueError, match="answer_candidates"):
            await server.handle_memory_synthesis_artifact({"checkout": checkout})

        server.session_manager.get.return_value.eventlog.append.assert_not_called()

    async def test_rejects_candidate_without_outcome(self, server: ZaxyMCPServer) -> None:
        """Candidate feedback must be explicit so clients do not record accidental usage."""
        with pytest.raises(ValueError, match="candidate and outcome"):
            await server.handle_memory_synthesis_artifact({
                "checkout": self._checkout_payload(),
                "candidate": {"rank": 1, "answer": "$120"},
            })

        server.session_manager.get.return_value.eventlog.append.assert_not_called()

    async def test_rejects_candidate_not_present_in_checkout(self, server: ZaxyMCPServer) -> None:
        """Candidate feedback should not disagree with checkout answer candidates."""
        with pytest.raises(ValueError, match="diagnostics.synthesis.answer_candidates"):
            await server.handle_memory_synthesis_artifact({
                "checkout": self._checkout_payload(),
                "candidate": {
                    "rank": 1,
                    "type": "currency",
                    "answer": "$120",
                    "support_source_ids": ["answer-99"],
                },
                "outcome": "used",
            })

        server.session_manager.get.return_value.eventlog.append.assert_not_called()


class TestMemorySynthesisEvidence:
    """Tests for memory_synthesis_evidence handler."""

    def _checkout_payload(self) -> dict[str, object]:
        return {
            "session_id": "agent-1",
            "query": "How much did I spend on bike expenses in total?",
            "prompt": "# Memory Checkout",
            "working_set": {},
            "ref": None,
            "current_facts": [],
            "evidence": [],
            "provenance": [],
            "retention": {},
            "warnings": [],
            "guidance": {},
            "quality": {"answerability": "answer_from_memory", "confidence": 0.86},
            "diagnostics": {
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
                        },
                        {
                            "fact_id": "currency:duplicate",
                            "source_group": "answer-4",
                            "citation": "eventloom://agent-1/events/4#dddddddddddd",
                            "kind": "currency",
                            "value": "40",
                            "exclude_reason": "duplicate_identity",
                        }
                    ],
                }
            },
            "context_counts": {},
            "replay_event_count": 0,
            "compacted": False,
            "assembly_policy": {},
        }

    async def test_appends_synthesis_evidence_feedback_event(self, server: ZaxyMCPServer) -> None:
        """MCP should persist row-level synthesis evidence feedback."""
        checkout = self._checkout_payload()
        synthesis = checkout["diagnostics"]["synthesis"]  # type: ignore[index]
        row = synthesis["ledger_rows"][0]  # type: ignore[index]
        candidate = synthesis["answer_candidates"][0]  # type: ignore[index]

        result = await server.handle_memory_synthesis_evidence({
            "checkout": checkout,
            "row": row,
            "candidate": candidate,
            "outcome": "used",
            "actor": "assistant",
            "reason": "row supported arithmetic",
        })

        call = server.session_manager.get.return_value.eventlog.append.call_args_list[-1]
        assert call.args == ("memory.evidence.reinforced",)
        assert call.kwargs["actor"] == "assistant"
        assert call.kwargs["thread"] == "agent-1"
        assert call.kwargs["payload"]["source_group"] == "answer-1"
        assert call.kwargs["payload"]["fact_id"] == "currency:0:0"
        assert call.kwargs["payload"]["reason"] == "row supported arithmetic"
        assert server.graph.upsert_extraction.await_count == 1
        output = json_loads(result[0].text)
        assert output["event_type"] == "memory.evidence.reinforced"
        assert output["outcome"] == "used"
        assert output["source_group"] == "answer-1"

    async def test_excluded_synthesis_evidence_response_matches_snapshot(self, server: ZaxyMCPServer) -> None:
        """memory_synthesis_evidence should keep a stable compact response."""
        checkout = self._checkout_payload()
        synthesis = checkout["diagnostics"]["synthesis"]  # type: ignore[index]
        row = synthesis["ledger_rows"][1]  # type: ignore[index]

        result = await server.handle_memory_synthesis_evidence({
            "checkout": checkout,
            "row": row,
            "outcome": "excluded",
            "actor": "assistant",
        })

        snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
        payload = json_loads(result[0].text)
        assert _mcp_response_snapshot("memory_synthesis_evidence", payload) == snapshots["memory_synthesis_evidence"]

    async def test_rejects_invalid_synthesis_evidence_inputs(self, server: ZaxyMCPServer) -> None:
        """Evidence feedback should fail closed for malformed payloads."""
        with pytest.raises(ValueError, match="row"):
            await server.handle_memory_synthesis_evidence({
                "checkout": self._checkout_payload(),
                "row": "not-a-row",
                "outcome": "used",
            })

        server.session_manager.get.return_value.eventlog.append.assert_not_called()

    async def test_rejects_empty_or_foreign_synthesis_evidence_row(self, server: ZaxyMCPServer) -> None:
        """Evidence feedback should refer to a specific row in the checkout."""
        with pytest.raises(ValueError, match="fact_id, source_group, or citation"):
            await server.handle_memory_synthesis_evidence({
                "checkout": self._checkout_payload(),
                "row": {},
                "outcome": "used",
            })

        with pytest.raises(ValueError, match="diagnostics.synthesis.ledger_rows"):
            await server.handle_memory_synthesis_evidence({
                "checkout": self._checkout_payload(),
                "row": {
                    "fact_id": "currency:foreign",
                    "source_group": "answer-99",
                    "citation": "eventloom://agent-1/events/99#ffffffffffff",
                },
                "outcome": "used",
            })

        server.session_manager.get.return_value.eventlog.append.assert_not_called()


class TestMemorySkill:
    """Tests for memory_skill handler."""

    async def test_appends_skill_lifecycle_event(self, server: ZaxyMCPServer) -> None:
        """Skill lifecycle helper should append and project deterministic skill events."""
        result = await server.handle_memory_skill({
            "action": "validated",
            "skill_id": "python-test-first",
            "name": "Python test-first implementation",
            "version": "2",
            "summary": "Write the failing pytest before implementation.",
            "procedure": ["Write focused failing test", "Run pytest"],
            "applicability": ["Python feature work"],
            "citations": ["eventloom://agent-1/events/4#abcd"],
            "actor": "assistant",
            "session_id": "agent-1",
        })

        log = server.session_manager.get.return_value.eventlog
        log.append.assert_called_once()
        call = log.append.call_args
        assert call.args == ("skill.validated",)
        assert call.kwargs["actor"] == "assistant"
        assert call.kwargs["thread"] == "agent-1"
        assert call.kwargs["payload"] == {
            "skill_id": "python-test-first",
            "version": "2",
            "name": "Python test-first implementation",
            "summary": "Write the failing pytest before implementation.",
            "procedure": ["Write focused failing test", "Run pytest"],
            "applicability": ["Python feature work"],
            "citations": ["eventloom://agent-1/events/4#abcd"],
        }
        server.graph.upsert_extraction.assert_awaited_once()
        server.tracer.trace_append.assert_awaited_once_with("skill.validated", "assistant", 1)
        payload = json_loads(result[0].text)
        assert payload["event_type"] == "skill.validated"
        assert payload["seq"] == 1

    async def test_rejects_unknown_skill_action(self, server: ZaxyMCPServer) -> None:
        """Skill helper should only allow known lifecycle event types."""
        with pytest.raises(ValueError, match="skill action"):
            await server.handle_memory_skill({
                "action": "invented",
                "skill_id": "python-test-first",
            })

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

    async def test_setup_bootstraps_default_embedded_runtime_before_graph_schema(self) -> None:
        """Local stdio startup should prepare the default embedded runtime."""
        with (
            patch("zaxy.mcp_server.build_projection_store") as mock_build_projection_store,
            patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.mcp_server.SessionManager"),
            patch("zaxy.mcp_server.LocalEmbeddedGraphRuntime") as mock_runtime_cls,
        ):
            mock_graph = AsyncMock()
            mock_build_projection_store.return_value = mock_graph
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

    async def test_setup_bootstraps_pggraph_runtime_when_backend_is_pggraph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MCP bootstrap should not require Neo4j when pgGraph is the selected backend."""
        monkeypatch.setenv("PROJECTION_BACKEND", "pggraph")
        monkeypatch.setenv("PGGRAPH_DSN", "postgresql://postgres:postgres@localhost:5432/zaxy")
        with (
            patch("zaxy.mcp_server.build_projection_store") as mock_build_projection_store,
            patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.mcp_server.SessionManager"),
            patch("zaxy.mcp_server.LocalNeo4jRuntime") as mock_neo4j_runtime_cls,
            patch("zaxy.mcp_server.LocalPgGraphRuntime") as mock_pggraph_runtime_cls,
        ):
            mock_graph = AsyncMock()
            mock_build_projection_store.return_value = mock_graph
            mock_tracer = AsyncMock()
            mock_tracer_cls.return_value = mock_tracer
            mock_pggraph_runtime = MagicMock()
            mock_pggraph_runtime_cls.return_value = mock_pggraph_runtime

            srv = ZaxyMCPServer()
            await srv.setup()

        mock_neo4j_runtime_cls.assert_not_called()
        mock_pggraph_runtime.ensure_available.assert_called_once()
        mock_graph.connect.assert_awaited_once()
        mock_graph.init_schema.assert_awaited_once()

    async def test_setup_bootstraps_embedded_runtime_when_backend_is_embedded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MCP bootstrap should not require Neo4j for the embedded graph backend."""
        monkeypatch.setenv("PROJECTION_BACKEND", "embedded")
        monkeypatch.setenv("EMBEDDED_GRAPH_PATH", ".eventloom/projections/embedded.kuzu")
        with (
            patch("zaxy.mcp_server.build_projection_store") as mock_build_projection_store,
            patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.mcp_server.SessionManager"),
            patch("zaxy.mcp_server.LocalNeo4jRuntime") as mock_neo4j_runtime_cls,
            patch("zaxy.mcp_server.LocalEmbeddedGraphRuntime") as mock_embedded_runtime_cls,
        ):
            mock_graph = AsyncMock()
            mock_build_projection_store.return_value = mock_graph
            mock_tracer = AsyncMock()
            mock_tracer_cls.return_value = mock_tracer
            mock_embedded_runtime = MagicMock()
            mock_embedded_runtime_cls.return_value = mock_embedded_runtime

            srv = ZaxyMCPServer()
            await srv.setup()

        mock_neo4j_runtime_cls.assert_not_called()
        mock_embedded_runtime.ensure_available.assert_called_once()
        assert mock_embedded_runtime_cls.call_args.kwargs["path"] == Path(".eventloom/projections/embedded.kuzu")
        mock_graph.connect.assert_awaited_once()
        mock_graph.init_schema.assert_awaited_once()

    async def test_setup_appends_workspace_genesis_once(self, tmp_path: Path) -> None:
        """setup() should bootstrap the default session with one workspace genesis event."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("# Rules\n\nUse pytest.\n", encoding="utf-8")
        eventlog = EventLog(tmp_path / "events.jsonl")
        mock_log = MagicMock()
        mock_log.read_all.return_value = []
        mock_log.append.side_effect = eventlog.append
        with (
            patch("zaxy.mcp_server.build_projection_store") as mock_build_projection_store,
            patch("zaxy.mcp_server.MemoryTracer") as mock_tracer_cls,
            patch("zaxy.mcp_server.SessionManager") as mock_session_cls,
            patch("zaxy.mcp_server.LocalNeo4jRuntime"),
        ):
            mock_graph = AsyncMock()
            mock_build_projection_store.return_value = mock_graph
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

        snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
        assert _mcp_response_snapshot("context_assemble", output) == snapshots["context_assemble"]

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

            with patch("zaxy.mcp_server.record_memory_activity") as record_activity:
                result = await server.handle_memory_checkout({
                    "query": "What context contract should the model use?",
                    "session_id": "agent-1",
                    "limit": 3,
                    "purpose": "review",
                })

        output = json_loads(result[0].text)
        assert output["session_id"] == "agent-1"
        assert output["current_facts"][0]["content"] == "Memory checkout is the context contract."
        assert output["current_facts"][0]["citation"] == "eventloom://agent-1/events/1882#checkout"
        assert output["current_facts"][0]["source_lane"] == "graph"
        assert output["evidence"][0]["citation"] == "eventloom://agent-1/events/1882#checkout"
        assert output["evidence"][0]["source_lane"] == "graph"
        assert output["provenance"][0]["event_seq"] == 1882
        assert output["purpose"]["profile"] == "review"
        purpose = output["diagnostics"].pop("purpose")
        assert purpose["profile"] == "review"
        assert purpose["evidence_policy"] == "cited_current_facts_required"
        slot_plan = output["diagnostics"].pop("slot_plan")
        assert slot_plan["version"] == "slot_plan_v1"
        assert slot_plan["answer_type"] == "direct_fact"
        assert slot_plan["operation"] == "select_fact"
        assert slot_plan["required_slots"] == []
        assert slot_plan["optional_slots"] == ["exact", "semantic"]
        assert output["diagnostics"] == {
            "source_lanes": {"graph": 2},
            "citation_count": 2,
            "current_citation_count": 2,
            "current_fact_count": 2,
            "superseded_contexts_excluded": 0,
            "warning_count": 0,
            "feedback_recommended": True,
            "feedback_tool": "memory_feedback",
            "feedback_reason": "Reinforce cited context if it materially informed the next response.",
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
                        "source_id": "eventloom://agent-1/events/1832#gap",
                        "evidence_count": 1,
                        "citation_count": 1,
                        "citations": ["eventloom://agent-1/events/1832#gap"],
                        "source_lanes": ["graph"],
                        "top_score": 0.91,
                        "snippet": "A memory capture gap was recorded during benchmark debugging.",
                    },
                    {
                        "source_id": "eventloom://agent-1/events/1882#checkout",
                        "evidence_count": 1,
                        "citation_count": 1,
                        "citations": ["eventloom://agent-1/events/1882#checkout"],
                        "source_lanes": ["graph"],
                        "top_score": 0.8,
                        "snippet": "Memory checkout is the context contract.",
                    },
                ]
            },
            "purpose_ontology_lens": {
                "applied": True,
                "profile": "review",
                "entity_roles": ["risk", "finding", "test", "decision", "regression"],
                "relationship_roles": [
                    "risk",
                    "regression",
                    "missing_test",
                    "accepted_decision",
                    "blocker",
                ],
                "current_fact_roles": [],
                "evidence_roles": [],
                "required_source_groups": ["accepted_or_cited_fact", "verification_evidence"],
                "suppress_rules": [
                    "pending_unreviewed_claim",
                    "superseded_context",
                    "low_trust_inference",
                ],
                "edge_trust_multipliers": {
                    "accepted_decision": 1.2,
                    "blocks_release": 1.4,
                    "low_trust_inference": 0.55,
                    "missing_test": 1.3,
                },
            },
        }
        assert output["guidance"]["recommended_next_call"] == {
            "tool": "memory_checkout",
            "query": "current decisions, blockers, and next actions for: What context contract should the model use?",
            "reason": "Refresh memory before major follow-up work, after compaction/resume, or when task scope changes.",
        }
        assert output["guidance"]["feedback"]["payloads"][0] == {
            "entity_name": "memory checkout",
            "entity_type": "task",
            "feedback": "used",
            "actor": "assistant",
            "query": "What context contract should the model use?",
            "source": "keyword",
            "score": 0.8,
            "citation": "eventloom://agent-1/events/1882#checkout",
            "importance": 0.6,
            "purpose": output["purpose"],
        }
        assert "Do not treat superseded contexts as current facts." in output["guidance"]["ignore"]
        assert output["guidance"]["purpose"]["profile"] == "review"
        assert "Use the purpose evidence policy: cited_current_facts_required." in output["guidance"]["trust"]
        record_activity.assert_called_once_with(
            server._eventloom_path,
            session_id="agent-1",
            activity="checkout",
            source="mcp",
            query="What context contract should the model use?",
            metadata={"token_efficiency": output["token_efficiency"]},
        )
        assert output["token_efficiency"]["prompt_tokens"] > 0
        assert output["token_efficiency"]["current_fact_count"] == 2
        assert output["token_efficiency"]["evidence_count"] == 2
        assert output["quality"] == {
            "answerability": "answer_from_memory",
            "confidence": 0.95,
            "reasons": [
                "Retrieved current facts with Eventloom citations.",
                "Applied purpose profile review with evidence policy cited_current_facts_required.",
            ],
            "required_action": None,
        }
        assert "# Memory Checkout" in output["prompt"]
        assert "## Purpose Profile" in output["prompt"]
        assert "## Checkout Quality" in output["prompt"]
        assert "answer_from_memory" in output["prompt"]
        assert "## Checkout Guidance" in output["prompt"]
        assert "## Checkout Diagnostics" in output["prompt"]
        assert "current decisions, blockers, and next actions" in output["prompt"]
        assert "memory_feedback" in output["prompt"]
        assert all(fact["valid_to"] is None for fact in output["current_facts"])

    async def test_memory_checkout_uses_core_checkout_builder(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """MCP memory_checkout should share the core MemoryFabric checkout policy."""
        server.session_manager.replay.return_value = MagicMock(events=[])
        with (
            patch("zaxy.mcp_server.QueryRouter") as mock_router_cls,
            patch("zaxy.mcp_server.build_memory_checkout", wraps=build_memory_checkout) as builder,
        ):
            router = AsyncMock()
            router.query.return_value = [
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
        assert output["current_facts"][0]["content"] == "Memory checkout is the context contract."
        builder.assert_called_once()

    async def test_memory_checkout_response_matches_v06_snapshot(
        self,
        server: ZaxyMCPServer,
    ) -> None:
        """memory_checkout should keep a representative response snapshot stable."""
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

        snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
        payload = json_loads(result[0].text)
        assert _mcp_response_snapshot("memory_checkout", payload) == snapshots["memory_checkout"]

    async def test_memory_checkout_asks_user_when_only_superseded_context_is_retrieved(
        self,
        tmp_path: Path,
    ) -> None:
        """memory_checkout should not treat superseded-only context as answerable."""
        server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
        with patch("zaxy.mcp_server.QueryRouter") as mock_router_cls:
            router = AsyncMock()
            router.query.return_value = [
                MagicMock(
                    content="Raw replay used to be the model context contract.",
                    source="keyword",
                    score=0.8,
                    valid_from="2026-05-09T12:00:00Z",
                    valid_to="2026-05-10T12:00:00Z",
                    citation="eventloom://agent-1/events/2#bbbbbbbbbbbb",
                    score_explanation=None,
                    entity_name="raw replay",
                    entity_type="decision",
                )
            ]
            mock_router_cls.return_value = router

            result = await server.handle_memory_checkout({
                "query": "What memory contract should the model use?",
                "session_id": "agent-1",
                "limit": 3,
            })

        output = json_loads(result[0].text)
        assert output["current_facts"] == []
        assert output["diagnostics"]["citation_count"] == 1
        assert output["diagnostics"]["current_citation_count"] == 0
        assert output["quality"]["answerability"] == "ask_user"
        assert output["quality"]["confidence"] == 0.25
        assert output["quality"]["required_action"] == {
            "type": "ask_user",
            "reason": "No current facts were retrieved; ask the user for the missing context before answering from memory.",
        }
        assert "ask_user" in output["prompt"]

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

    def test_configured_token_rejects_missing_session_header(self) -> None:
        """Authenticated remote transports should fail closed without session identity."""
        auth = MCPTransportAuth(token="secret")

        with pytest.raises(PermissionError, match="session header"):
            auth.authorize({"authorization": "Bearer secret"})

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
    @patch("zaxy.mcp_server.build_projection_store")
    @patch("zaxy.mcp_server.MemoryTracer")
    async def test_main_setup_and_teardown(
        self,
        mock_tracer_cls: MagicMock,
        mock_build_projection_store: MagicMock,
        mock_stdio: MagicMock,
    ) -> None:
        """main() should setup server, register handlers, and teardown on exit."""
        mock_graph = AsyncMock()
        mock_build_projection_store.return_value = mock_graph
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
    async def test_main_releases_owner_claim_when_setup_fails(
        self,
        mock_server_cls: MagicMock,
    ) -> None:
        """Embedded owner locks should not leak when MCP setup fails before serving."""
        mock_server = AsyncMock()
        mock_server.setup.side_effect = RuntimeError("kuzu failed")
        mock_server_cls.return_value = mock_server
        owner_claim = MagicMock()

        with pytest.raises(RuntimeError, match="kuzu failed"):
            await main(owner_claim=owner_claim)

        owner_claim.close.assert_called_once_with()

    @patch("zaxy.mcp_server.ZaxyMCPServer")
    @patch("zaxy.mcp_server.stdio_server")
    async def test_main_publishes_and_releases_embedded_owner_claim(
        self,
        mock_stdio: MagicMock,
        mock_server_cls: MagicMock,
    ) -> None:
        """Embedded owners should publish socket metadata and clean up on shutdown."""
        mock_server = AsyncMock()
        mock_server._workspace_root = Path("/tmp/workspace")
        mock_server._embedded_graph_path = Path("/tmp/workspace/.eventloom/projections/embedded.kuzu")
        mock_server_cls.return_value = mock_server

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        owner_claim = MagicMock()
        socket_server = MagicMock()
        socket_server.wait_closed = AsyncMock()

        with (
            patch("zaxy.mcp_server._start_owner_socket_server", new_callable=AsyncMock) as start_socket,
            patch("zaxy.mcp_server.app.run", new_callable=AsyncMock),
        ):
            start_socket.return_value = socket_server
            await main(owner_claim=owner_claim)

        start_socket.assert_awaited_once_with(owner_claim)
        owner_claim.write_ready_record.assert_called_once_with(
            workspace_root=mock_server._workspace_root,
            projection_backend="embedded",
            graph_path=mock_server._embedded_graph_path,
        )
        socket_server.close.assert_called_once_with()
        socket_server.wait_closed.assert_awaited_once_with()
        mock_server.teardown.assert_awaited_once_with()
        owner_claim.close.assert_called_once_with()

    async def test_proxy_main_forwards_stdio_to_owner_socket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Duplicate embedded workers should proxy JSON-RPC lines to the owner socket."""
        coordinator = MagicMock()
        coordinator.wait_for_owner_record.return_value = {"socket_path": "/tmp/zaxy-owner.sock"}
        socket_reader = AsyncMock()
        socket_reader.readline = AsyncMock(side_effect=[b'{"jsonrpc":"2.0","id":1,"result":{}}\n', b""])
        socket_writer = MagicMock()
        socket_writer.drain = AsyncMock()
        socket_writer.wait_closed = AsyncMock()

        stdin_buffer = MagicMock()
        stdin_buffer.readline = MagicMock(side_effect=[b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n', b""])
        stdout_buffer = MagicMock()
        stdout_buffer.write = MagicMock()
        stdout_buffer.flush = MagicMock()
        monkeypatch.setattr(zaxy.mcp_server.sys, "stdin", SimpleNamespace(buffer=stdin_buffer))
        monkeypatch.setattr(zaxy.mcp_server.sys, "stdout", SimpleNamespace(buffer=stdout_buffer))

        with patch(
            "zaxy.mcp_server.asyncio.open_unix_connection",
            new_callable=AsyncMock,
            return_value=(socket_reader, socket_writer),
        ) as open_socket:
            await zaxy.mcp_server.proxy_main(coordinator)

        coordinator.wait_for_owner_record.assert_called_once_with()
        open_socket.assert_awaited_once_with("/tmp/zaxy-owner.sock")
        socket_writer.write.assert_any_call(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
        socket_writer.write_eof.assert_called_once_with()
        socket_writer.close.assert_called_once_with()
        socket_writer.wait_closed.assert_awaited_once_with()

    async def test_socket_mcp_transport_bridges_socket_lines_and_mcp_messages(self) -> None:
        """The owner socket transport should decode and encode JSON-RPC line messages."""
        input_lines = iter([b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n', b""])

        class FakeSocketReader:
            async def readline(self) -> bytes:
                return next(input_lines, b"")

        class FakeSocketWriter:
            def __init__(self) -> None:
                self.lines: list[bytes] = []
                self.closed = False
                self.waited = False

            def write(self, line: bytes) -> None:
                self.lines.append(line)

            async def drain(self) -> None:
                return None

            def close(self) -> None:
                self.closed = True

            async def wait_closed(self) -> None:
                self.waited = True

        writer = FakeSocketWriter()

        async with zaxy.mcp_server._socket_mcp_transport(
            FakeSocketReader(),
            writer,
        ) as (read_stream, write_stream):
            inbound = await read_stream.receive()
            assert inbound.message.root.method == "notifications/initialized"

            outbound = zaxy.mcp_server.types.JSONRPCMessage.model_validate_json(
                '{"jsonrpc":"2.0","id":1,"result":{}}'
            )
            await write_stream.send(zaxy.mcp_server.SessionMessage(outbound))
            await write_stream.aclose()

        assert writer.lines == [b'{"jsonrpc":"2.0","id":1,"result":{}}\n']
        assert writer.closed is True
        assert writer.waited is True

    @patch("zaxy.mcp_server.ZaxyMCPServer")
    @patch("zaxy.mcp_server.stdio_server")
    async def test_unknown_tool_returns_structured_error(
        self,
        mock_stdio: MagicMock,
        mock_server_cls: MagicMock,
    ) -> None:
        """Calling an unknown tool should return a stable client-facing error."""
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
        result = await captured_handlers["call_tool"]("unknown_tool", {})
        payload = json.loads(result[0].text)
        assert payload["error"]["code"] == "unknown_tool"
        assert payload["error"]["message"] == "Unknown tool: unknown_tool"

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
        """Failed MCP dispatch should record a failed lifecycle event and return an error."""
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

        result = await captured_handlers["call_tool"](
            "unknown_tool",
            {"session_id": "agent-1", "api_key": "secret"},
        )
        payload = json.loads(result[0].text)
        assert payload["error"]["code"] == "unknown_tool"

        mock_server.capture_tool_call_completed.assert_awaited_once()
        call = mock_server.capture_tool_call_completed.await_args
        assert call.kwargs["tool_name"] == "unknown_tool"
        assert call.kwargs["status"] == "failed"
        assert call.kwargs["session_id"] == "agent-1"
        assert call.kwargs["result_summary"] == "unknown_tool: Unknown tool: unknown_tool"

    @patch("zaxy.mcp_server.ZaxyMCPServer")
    @patch("zaxy.mcp_server.stdio_server")
    async def test_call_tool_returns_structured_error_payload(
        self,
        mock_stdio: MagicMock,
        mock_server_cls: MagicMock,
    ) -> None:
        """MCP clients should receive stable error codes and remediation hints."""
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

        result = await captured_handlers["call_tool"](
            "unknown_tool",
            {"session_id": "agent-1", "api_key": "secret"},
        )

        payload = json.loads(result[0].text)
        assert payload == {
            "error": {
                "code": "unknown_tool",
                "message": "Unknown tool: unknown_tool",
                "remediation": "Call list_tools and retry with one of the advertised tool names.",
            }
        }
        mock_server.capture_tool_call_completed.assert_awaited_once()
        call = mock_server.capture_tool_call_completed.await_args
        assert call.kwargs["tool_name"] == "unknown_tool"
        assert call.kwargs["status"] == "failed"
        assert call.kwargs["result_summary"] == "unknown_tool: Unknown tool: unknown_tool"

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

    @patch("zaxy.mcp_server.build_projection_store")
    @patch("zaxy.mcp_server.MemoryTracer")
    @patch("zaxy.mcp_server.SessionManager")
    async def test_setup_connects_all(
        self,
        mock_session_cls: MagicMock,
        mock_tracer_cls: MagicMock,
        mock_build_projection_store: MagicMock,
    ) -> None:
        """setup() should connect graph and tracer."""
        mock_graph = AsyncMock()
        mock_build_projection_store.return_value = mock_graph
        mock_tracer = AsyncMock()
        mock_tracer_cls.return_value = mock_tracer

        srv = ZaxyMCPServer()
        await srv.setup()
        mock_graph.connect.assert_awaited_once()
        mock_graph.init_schema.assert_awaited_once()
        mock_tracer.connect.assert_awaited_once()

    @patch("zaxy.mcp_server.build_projection_store")
    @patch("zaxy.mcp_server.MemoryTracer")
    @patch("zaxy.mcp_server.SessionManager")
    async def test_teardown_closes_all(
        self,
        mock_session_cls: MagicMock,
        mock_tracer_cls: MagicMock,
        mock_build_projection_store: MagicMock,
    ) -> None:
        """teardown() should close graph and tracer."""
        mock_graph = AsyncMock()
        mock_build_projection_store.return_value = mock_graph
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
    @patch("zaxy.mcp_server.build_projection_store")
    @patch("zaxy.mcp_server.MemoryTracer")
    @patch("zaxy.mcp_server.SessionManager")
    async def test_main_sse_setup_and_teardown(
        self,
        mock_session_cls: MagicMock,
        mock_tracer_cls: MagicMock,
        mock_build_projection_store: MagicMock,
        mock_uvicorn_cls: MagicMock,
    ) -> None:
        """main_sse() should setup server, run uvicorn, and teardown."""
        mock_graph = AsyncMock()
        mock_build_projection_store.return_value = mock_graph
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
