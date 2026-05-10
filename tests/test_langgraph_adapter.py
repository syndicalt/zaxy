"""Tests for the native-preview LangGraph adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from zaxy.adapters.langgraph import LangGraphMemoryAdapter, create_langgraph_memory_node
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

    async def close(self) -> None:
        self.calls.append(("close", {}))


def test_langgraph_registry_marks_native_preview() -> None:
    """LangGraph should advertise the maintained native-preview adapter."""
    specs = {spec.framework: spec for spec in list_framework_integration_specs()}

    assert specs["langgraph"].maturity == "native-preview"
    assert specs["langgraph"].native_adapter == "zaxy.adapters.langgraph"
    assert specs["crewai"].native_adapter == "planned-next"


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
