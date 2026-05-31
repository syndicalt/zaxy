"""Tests for the Claude-compatible model-call adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from zaxy.adapters.claude_compatible import (
    ClaudeCompatibleMemoryAdapter,
    create_claude_compatible_memory_adapter,
)


@dataclass
class FakeCheckout:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


@dataclass
class FakeFabric:
    calls: list[tuple[str, dict[str, Any]]]

    async def checkout_memory(self, query: str, **kwargs: Any) -> FakeCheckout:
        self.calls.append(("checkout_memory", {"query": query, **kwargs}))
        return FakeCheckout(
            {
                "session_id": kwargs["session_id"],
                "query": query,
                "prompt": "# Memory Checkout\nUse accepted release state.",
                "current_facts": [{"name": "release.state"}],
                "warnings": [],
                "diagnostics": {"citation_coverage": 1.0},
                "quality": {"answerability": "answerable", "confidence": 0.9},
                "guidance": {"feedback": {"tool": "memory_feedback"}},
            }
        )

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

    async def close(self) -> None:
        self.calls.append(("close", {}))


@dataclass
class FailingCheckoutFabric:
    calls: list[tuple[str, dict[str, Any]]]

    async def checkout_memory(self, query: str, **kwargs: Any) -> FakeCheckout:
        self.calls.append(("checkout_memory", {"query": query, **kwargs}))
        raise RuntimeError("projection unavailable")

    async def close(self) -> None:
        self.calls.append(("close", {}))


class FakeMessages:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self._captured["request"] = kwargs
        return {
            "id": "msg-test",
            "content": [{"type": "text", "text": "Use checkout context."}],
        }


class FakeClient:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.messages = FakeMessages(captured)


@pytest.mark.asyncio
async def test_claude_compatible_adapter_injects_checkout_and_captures_response() -> None:
    """messages_create should inject checkout as system text and capture bounded observations."""
    calls: list[tuple[str, dict[str, Any]]] = []
    captured: dict[str, Any] = {}
    adapter = ClaudeCompatibleMemoryAdapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    result = await adapter.messages_create(
        FakeClient(captured),
        model="claude-test",
        messages=[{"role": "user", "content": "What next?"}],
        max_tokens=256,
    )

    request = captured["request"]
    assert request["model"] == "claude-test"
    assert request["max_tokens"] == 256
    assert request["system"] == "# Memory Checkout\nUse accepted release state."
    assert request["messages"] == [{"role": "user", "content": "What next?"}]
    assert result["assistant_content"] == "Use checkout context."
    assert result["zaxy"]["contract"] == "zaxy.native.v0.6"
    assert result["zaxy"]["framework"] == "claude-compatible"
    assert result["zaxy"]["operation"] == "messages_create"
    append_events = [call[1] for call in calls if call[0] == "append"]
    assert [event["event_type"] for event in append_events] == [
        "model.call.requested",
        "transcript.turn",
    ]
    request_payload = append_events[0]["payload"]
    assert request_payload["provider"] == "claude-compatible"
    assert request_payload["message_count"] == 1
    assert request_payload["injected_memory"] is True
    assert "messages" not in request_payload
    assert append_events[1]["payload"]["content"] == "Use checkout context."
    assert calls[-1] == ("close", {})


@pytest.mark.asyncio
async def test_claude_compatible_adapter_records_tool_observation_and_feedback() -> None:
    """The adapter should expose the shared tool observation and feedback helpers."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = create_claude_compatible_memory_adapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    await adapter.record_tool_call(
        tool_name="search",
        status="ok",
        arguments={"query": "roadmap", "token": "secret"},
        result_summary="found roadmap",
    )
    await adapter.record_feedback(
        entity_name="release.state",
        entity_type="Decision",
        query="What next?",
        feedback="helpful",
        importance=0.8,
    )

    append_events = [call[1] for call in calls if call[0] == "append"]
    assert append_events[0]["event_type"] == "tool.call.completed"
    assert append_events[0]["payload"]["arguments_redacted"] is True
    assert append_events[1]["event_type"] == "memory.reinforced"
    assert append_events[1]["actor"] == "claude-compatible"
    assert append_events[1]["payload"] == {
        "entity_name": "release.state",
        "entity_type": "Decision",
        "query": "What next?",
        "source": "claude-compatible",
        "importance": 0.8,
    }


@pytest.mark.asyncio
async def test_claude_compatible_adapter_fails_closed_when_checkout_fails() -> None:
    """Checkout failure should not call the provider or inject stale context."""
    calls: list[tuple[str, dict[str, Any]]] = []
    captured: dict[str, Any] = {}
    adapter = ClaudeCompatibleMemoryAdapter(
        session_id="agent-1",
        fabric_factory=lambda eventloom_path: FailingCheckoutFabric(calls),
    )

    result = await adapter.messages_create(
        FakeClient(captured),
        model="claude-test",
        messages=[{"role": "user", "content": "What next?"}],
    )

    assert "request" not in captured
    assert result["messages"] == [{"role": "user", "content": "What next?"}]
    assert result["assistant_content"] == ""
    assert result["zaxy"]["status"] == "error"
    assert result["zaxy"]["error"]["code"] == "checkout_failed"
    assert calls == [
        (
            "checkout_memory",
            {
                "query": "What next?",
                "session_id": "agent-1",
                "limit": 10,
                "max_recent_events": 20,
            },
        ),
        ("close", {}),
    ]
