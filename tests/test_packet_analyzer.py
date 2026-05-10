"""Tests for the OpenAI-compatible LLM packet analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from zaxy.event import EventLog
from zaxy.packet_analyzer import LlmPacketAnalyzer, PacketAnalyzerConfig


def test_packet_analyzer_forwards_and_captures_completed_packet(tmp_path: Path) -> None:
    """The analyzer should pass through requests and append packet provenance."""
    eventloom_path = tmp_path / ".eventloom"
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "model": "gpt-test",
                "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            },
        )

    analyzer = LlmPacketAnalyzer(
        PacketAnalyzerConfig(
            eventloom_path=eventloom_path,
            session_id="agent-1",
            upstream_base_url="https://upstream.example/v1",
            upstream_api_key="upstream-key",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = analyzer.forward(
        "POST",
        "/v1/chat/completions",
        headers={"authorization": "Bearer client-key", "content-type": "application/json"},
        body=json.dumps(
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        ).encode(),
    )
    analyzer.close()

    assert response.status_code == 200
    assert seen["url"] == "https://upstream.example/v1/chat/completions"
    assert seen["authorization"] == "Bearer upstream-key"

    events = EventLog(eventloom_path / "agent-1.jsonl").read_all()
    assert [event.type for event in events] == ["llm.packet.completed"]
    payload = events[0].payload
    assert payload["provider_path"] == "/v1/chat/completions"
    assert payload["method"] == "POST"
    assert payload["status_code"] == 200
    assert payload["model"] == "gpt-test"
    assert payload["usage_counts"] == {"prompt": 11, "completion": 2, "total": 13}
    assert payload["request"]["body"]["messages"][0]["content"] == "Hi"
    assert payload["response"]["body"]["choices"][0]["message"]["content"] == "Hello"
    assert payload["request_hash"]
    assert payload["response_hash"]
    assert "authorization" not in payload["request"]["headers"]


def test_packet_analyzer_records_upstream_errors(tmp_path: Path) -> None:
    """Failed upstream responses should still be captured for audit."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    analyzer = LlmPacketAnalyzer(
        PacketAnalyzerConfig(
            eventloom_path=tmp_path / ".eventloom",
            session_id="agent-1",
            upstream_base_url="https://upstream.example",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = analyzer.forward(
        "POST",
        "/chat/completions",
        headers={"content-type": "application/json"},
        body=b'{"model":"gpt-test","messages":[]}',
    )
    analyzer.close()

    assert response.status_code == 429
    event = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()[0]
    assert event.type == "llm.packet.completed"
    assert event.payload["status_code"] == 429
    assert event.payload["response"]["body"]["error"]["message"] == "rate limited"
