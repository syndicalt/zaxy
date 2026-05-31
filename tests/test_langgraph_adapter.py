"""Tests for the native-beta LangGraph adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from zaxy.adapters.langgraph import (
    LangGraphMemoryAdapter,
    create_langgraph_coordination_node,
    create_langgraph_memory_node,
)
from zaxy.core import Context, ContextAssembly
from zaxy.integrations import list_framework_integration_specs


@dataclass
class FakeFabric:
    """Minimal async MemoryFabric test double."""

    calls: list[tuple[str, dict[str, Any]]]

    async def after_turn(self, **kwargs: Any) -> ContextAssembly:
        self.calls.append(("after_turn", kwargs))
        return _assembly()

    async def assemble_context(self, query: str, **kwargs: Any) -> ContextAssembly:
        self.calls.append(("assemble_context", {"query": query, **kwargs}))
        return _assembly()

    async def append(self, event_type: str, actor: str, payload: dict[str, Any], session_id: str) -> None:
        self.calls.append(
            (
                "append",
                {
                    "event_type": event_type,
                    "actor": actor,
                    "payload": payload,
                    "session_id": session_id,
                },
            )
        )

    async def record_context_feedback(self, contexts: list[Context], **kwargs: Any) -> int:
        self.calls.append(("record_context_feedback", {"contexts": contexts, **kwargs}))
        return len(contexts)

    async def checkout_memory(self, query: str, **kwargs: Any) -> Any:
        self.calls.append(("checkout_memory", {"query": query, **kwargs}))
        return _checkout()

    async def close(self) -> None:
        self.calls.append(("close", {}))


@dataclass
class FailingCheckoutFabric:
    """MemoryFabric test double that fails checkout but still needs closing."""

    calls: list[tuple[str, dict[str, Any]]]

    async def checkout_memory(self, query: str, **kwargs: Any) -> Any:
        self.calls.append(("checkout_memory", {"query": query, **kwargs}))
        raise RuntimeError("projection unavailable")

    async def close(self) -> None:
        self.calls.append(("close", {}))


def test_langgraph_registry_marks_native_preview() -> None:
    """LangGraph should advertise the maintained native-beta adapter."""
    specs = {spec.framework: spec for spec in list_framework_integration_specs()}

    assert specs["langgraph"].maturity == "native-beta"
    assert specs["langgraph"].native_adapter == "zaxy.adapters.langgraph"
    assert specs["crewai"].native_adapter == "zaxy.adapters.crewai"


@pytest.mark.asyncio
async def test_langgraph_adapter_captures_latest_user_turn_and_projects_context() -> None:
    """before_model should preserve a user turn and inject prompt-ready memory."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = LangGraphMemoryAdapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    state = await adapter.before_model({"messages": [{"role": "user", "content": "What did we decide?"}]})

    assert state["zaxy_context"] == "Use source-aware memory."
    assert state["zaxy"]["session_id"] == "agent-1"
    assert state["zaxy_contexts"][0].content == "source-aware memory"
    assert calls[0] == (
        "after_turn",
        {
            "role": "user",
            "content": "What did we decide?",
            "session_id": "agent-1",
            "query": "What did we decide?",
            "source": "langgraph",
            "max_recent_events": 20,
            "limit": 10,
        },
    )
    assert calls[-1] == ("close", {})


@pytest.mark.asyncio
async def test_langgraph_memory_node_uses_adapter_before_model() -> None:
    """Factory helper should return a LangGraph-compatible async node."""
    calls: list[tuple[str, dict[str, Any]]] = []
    node = create_langgraph_memory_node(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    state = await node({"latest_message": "Assemble context"})

    assert state["zaxy_context"] == "Use source-aware memory."
    assert calls[0][0] == "after_turn"


@pytest.mark.asyncio
async def test_langgraph_coordination_node_reports_explicit_worker_finding(tmp_path) -> None:
    """Coordinate node should append worker-local findings without framework imports."""
    node = create_langgraph_coordination_node(
        mission_id="auth-main",
        worker_id="auth-api",
        eventloom_path=str(tmp_path / ".eventloom"),
    )

    state = await node(
        {
            "coordination_summary": "API failures trace to expired JWKS cache handling",
            "coordination_evidence": [{"kind": "source", "reference": "src/auth.py:12"}],
            "coordination_claim_key": "auth.failure.cause",
            "coordination_claim_value": "expired-jwks-cache",
        }
    )

    finding = state["zaxy_coordination"]
    assert finding["event_type"] == "coordination.finding.reported"
    assert finding["mission_id"] == "auth-main"
    assert finding["worker_id"] == "auth-api"
    assert finding["finding_id"]


@pytest.mark.asyncio
async def test_langgraph_adapter_checkout_before_model_uses_memory_checkout() -> None:
    """Opinionated middleware should inject Memory Checkout, not just generic context."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = LangGraphMemoryAdapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    state = await adapter.checkout_before_model({"latest_message": "Where are we?"})

    assert state["zaxy_context"] == "# Memory Checkout\nUse cited memory."
    assert state["zaxy"]["kind"] == "memory_checkout"
    assert calls[0] == (
        "checkout_memory",
        {
            "query": "Where are we?",
            "session_id": "agent-1",
            "limit": 10,
            "max_recent_events": 20,
        },
    )


@pytest.mark.asyncio
async def test_langgraph_checkout_payload_exposes_v06_native_contract() -> None:
    """LangGraph checkout metadata should be stable enough for beta middleware."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = LangGraphMemoryAdapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    state = await adapter.checkout_before_model({"latest_message": "Where are we?"})

    assert state["zaxy"] == {
        "contract": "zaxy.native.v0.6",
        "framework": "langgraph",
        "operation": "before_model",
        "source": "langgraph",
        "kind": "memory_checkout",
        "status": "ok",
        "session_id": "agent-1",
        "query": "Where are we?",
        "current_fact_count": 1,
        "warning_count": 0,
        "diagnostics": {
            "current_fact_count": 1,
            "current_citation_count": 1,
            "feedback_tool": "memory_feedback",
        },
        "quality": {
            "answerability": "answer_from_memory",
            "confidence": 0.91,
            "required_action": None,
        },
        "feedback": {
            "tool": "memory_feedback",
            "payloads": [
                {
                    "entity_name": "memory checkout",
                    "entity_type": "workflow",
                    "feedback": "used",
                }
            ],
        },
        "error": None,
    }


@pytest.mark.asyncio
async def test_langgraph_checkout_failure_returns_stable_error_payload() -> None:
    """LangGraph checkout should fail closed with no context and actionable metadata."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = LangGraphMemoryAdapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FailingCheckoutFabric(calls),
    )

    state = await adapter.checkout_before_model({"latest_message": "Where are we?"})

    assert state["zaxy_context"] == ""
    assert state["zaxy_contexts"] == []
    assert state["zaxy"] == {
        "contract": "zaxy.native.v0.6",
        "framework": "langgraph",
        "operation": "before_model",
        "source": "langgraph",
        "kind": "memory_checkout",
        "status": "error",
        "session_id": "agent-1",
        "query": "Where are we?",
        "current_fact_count": 0,
        "warning_count": 1,
        "diagnostics": {},
        "quality": {
            "answerability": "refresh_recommended",
            "confidence": 0.0,
            "required_action": {
                "tool": "memory_checkout",
                "reason": "Projection unavailable during LangGraph checkout.",
            },
        },
        "feedback": None,
        "error": {
            "code": "checkout_failed",
            "message": "projection unavailable",
            "remediation": "Retry Memory Checkout before the next model call or run zaxy doctor.",
        },
    }
    assert calls[-1] == ("close", {})


@pytest.mark.asyncio
async def test_langgraph_adapter_records_tool_calls_without_argument_values() -> None:
    """record_tool_call should append redacted tool-call observations."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = LangGraphMemoryAdapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    await adapter.record_tool_call(
        tool_name="search",
        status="ok",
        arguments={"query": "zaxy", "token": "secret"},
        result_summary="1 hit",
    )

    append = calls[0]
    assert append[0] == "append"
    assert append[1]["event_type"] == "tool.call.completed"
    assert append[1]["actor"] == "zaxy-observer"
    assert append[1]["payload"]["argument_keys"] == ["query", "token"]
    assert "arguments" not in append[1]["payload"]


@pytest.mark.asyncio
async def test_langgraph_adapter_records_feedback_for_projected_contexts() -> None:
    """record_context_feedback should reinforce contexts carried in LangGraph state."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = LangGraphMemoryAdapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )
    state = {"zaxy_contexts": _assembly().contexts}

    count = await adapter.record_context_feedback(state, feedback="used", importance=0.8)

    assert count == 1
    assert calls[0][0] == "record_context_feedback"
    assert calls[0][1]["feedback"] == "used"
    assert calls[0][1]["importance"] == 0.8


def _assembly() -> ContextAssembly:
    context = Context(content="source-aware memory", source="verbatim", score=0.9)
    return ContextAssembly(
        session_id="agent-1",
        prompt="Use source-aware memory.",
        contexts=[context],
        replay_event_count=2,
        compacted=False,
        warnings=[],
    )


def _checkout() -> Any:
    class Checkout:
        def to_dict(self) -> dict[str, Any]:
            return {
                "session_id": "agent-1",
                "query": "Where are we?",
                "prompt": "# Memory Checkout\nUse cited memory.",
                "current_facts": [{"content": "Use cited memory.", "citation": "eventloom://agent-1/events/1#abc"}],
                "evidence": [],
                "diagnostics": {
                    "current_fact_count": 1,
                    "current_citation_count": 1,
                    "feedback_tool": "memory_feedback",
                },
                "quality": {
                    "answerability": "answer_from_memory",
                    "confidence": 0.91,
                    "required_action": None,
                },
                "guidance": {
                    "feedback": {
                        "tool": "memory_feedback",
                        "payloads": [
                            {
                                "entity_name": "memory checkout",
                                "entity_type": "workflow",
                                "feedback": "used",
                            }
                        ],
                    }
                },
                "warnings": [],
            }

    return Checkout()
