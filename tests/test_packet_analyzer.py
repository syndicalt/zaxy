"""Tests for the OpenAI-compatible LLM packet analyzer."""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from zaxy.event import EventLog
from zaxy.packet_analyzer import (
    LlmPacketAnalyzer,
    PacketAnalyzerConfig,
    PacketStreamResponse,
    _captured_headers,
    _json_body,
    _model_from_packet,
    _response_headers,
    _usage_counts_from_response,
    run_packet_analyzer,
)


class ChunkedStream(httpx.SyncByteStream):
    """Simple sync stream that exposes chunk boundaries to tests."""

    def __init__(self, chunks: list[bytes], markers: list[str]) -> None:
        self._chunks = chunks
        self._markers = markers

    def __iter__(self) -> Iterator[bytes]:
        for index, chunk in enumerate(self._chunks):
            self._markers.append(f"yielded-{index}")
            yield chunk


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


def test_packet_analyzer_normalizes_urls_headers_and_non_json_bodies(tmp_path: Path) -> None:
    """Forwarding should avoid duplicate base paths and capture only safe packet metadata."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream", "transfer-encoding": "chunked"},
            content=b"\xff\xfeopaque",
        )

    analyzer = LlmPacketAnalyzer(
        PacketAnalyzerConfig(
            eventloom_path=tmp_path / ".eventloom",
            session_id="agent-1",
            upstream_base_url="https://upstream.example/v1",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = analyzer.forward(
        "POST",
        "/v1/responses",
        headers={
            "host": "localhost",
            "content-length": "999",
            "content-type": "application/json",
            "openai-project": "proj_123",
            "x-secret": "redacted",
        },
        body=b"\xff\xfe",
    )
    analyzer.close()

    assert seen["url"] == "https://upstream.example/v1/responses"
    assert seen["headers"]["host"] == "upstream.example"
    assert seen["headers"]["content-length"] == "2"
    assert response.headers == {"content-type": "application/octet-stream"}

    event = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()[0]
    payload = event.payload
    assert payload["model"] is None
    assert payload["usage_counts"] is None
    assert payload["request"]["body"]["bytes"] == 2
    assert payload["response"]["body"]["bytes"] == len(b"\xff\xfeopaque")
    assert payload["request"]["headers"] == {
        "content-type": "application/json",
        "openai-project": "proj_123",
    }


def test_packet_analyzer_helper_parsers_handle_optional_packet_shapes() -> None:
    """Packet helper parsers should support empty, malformed, and response-first metadata."""
    assert _json_body(b"") is None
    assert _json_body(b"not-json")["bytes"] == len(b"not-json")
    assert _captured_headers({"User-Agent": "codex", "Authorization": "secret"}) == {
        "user-agent": "codex"
    }
    assert _response_headers(httpx.Headers({"Connection": "close", "Content-Type": "text/plain"})) == {
        "content-type": "text/plain"
    }
    assert _model_from_packet({"model": "request-model"}, {"model": "response-model"}) == "response-model"
    assert _model_from_packet({"model": "request-model"}, {}) == "request-model"
    assert _model_from_packet([], []) is None
    assert _usage_counts_from_response({"usage": {"prompt_tokens": 3, "completion_tokens": 2}}) == {
        "prompt": 3,
        "completion": 2,
        "total": None,
    }
    assert _usage_counts_from_response({"usage": "unknown"}) is None


def test_run_packet_analyzer_serves_post_requests_and_closes_owned_analyzer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The blocking packet analyzer server should proxy POST bodies and close cleanly."""
    import http.server

    seen: dict[str, object] = {"closed": False}
    server_holder: dict[str, http.server.ThreadingHTTPServer] = {}

    class FakeAnalyzer:
        def __init__(self, config: PacketAnalyzerConfig) -> None:
            seen["config"] = config

        def forward_stream(
            self,
            method: str,
            path: str,
            *,
            headers: dict[str, str],
            body: bytes,
        ) -> PacketStreamResponse:
            seen["method"] = method
            seen["path"] = path
            seen["headers"] = headers
            seen["body"] = body
            return PacketStreamResponse(
                status_code=202,
                headers={"content-type": "application/json"},
                body_chunks=iter([b'{"ok":true}']),
            )

        def close(self) -> None:
            seen["closed"] = True

    class CapturingServer(http.server.ThreadingHTTPServer):
        def __init__(self, server_address: tuple[str, int], handler_class: type[http.server.BaseHTTPRequestHandler]):
            super().__init__(server_address, handler_class)
            server_holder["server"] = self

    monkeypatch.setattr("zaxy.packet_analyzer.LlmPacketAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(http.server, "ThreadingHTTPServer", CapturingServer)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    thread = threading.Thread(
        target=run_packet_analyzer,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "config": PacketAnalyzerConfig(
                eventloom_path=tmp_path / ".eventloom",
                session_id="agent-1",
                upstream_base_url="https://upstream.example",
            ),
        },
        daemon=True,
    )
    thread.start()
    while "server" not in server_holder:
        pass

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/v1/responses", body=b'{"model":"gpt"}', headers={"content-type": "application/json"})
    response = conn.getresponse()
    body = response.read()
    conn.close()
    server_holder["server"].shutdown()
    thread.join(timeout=5)

    assert response.status == 202
    assert body == b'{"ok":true}'
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/responses"
    assert seen["body"] == b'{"model":"gpt"}'
    assert seen["closed"] is True


def test_packet_analyzer_streams_response_before_finalizing_capture(tmp_path: Path) -> None:
    """Streaming responses should pass chunks through before packet capture finalizes."""
    eventloom_path = tmp_path / ".eventloom"
    markers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedStream([b"data: one\n\n", b"data: two\n\n"], markers),
        )

    analyzer = LlmPacketAnalyzer(
        PacketAnalyzerConfig(
            eventloom_path=eventloom_path,
            session_id="agent-1",
            upstream_base_url="https://upstream.example",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = analyzer.forward_stream(
        "POST",
        "/chat/completions",
        headers={"content-type": "application/json"},
        body=b'{"model":"gpt-test","stream":true,"messages":[]}',
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    iterator = iter(response.body_chunks)
    assert next(iterator) == b"data: one\n\n"
    assert markers == ["yielded-0"]
    assert not (eventloom_path / "agent-1.jsonl").exists()

    assert list(iterator) == [b"data: two\n\n"]
    analyzer.close()

    event = EventLog(eventloom_path / "agent-1.jsonl").read_all()[0]
    assert event.payload["response"]["body"] == {
        "raw_sha256": hashlib.sha256(b"data: one\n\ndata: two\n\n").hexdigest(),
        "bytes": 22,
    }
