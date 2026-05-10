"""OpenAI-compatible packet analyzer for LLM request provenance.

The analyzer is intentionally not a router. It forwards packets to one upstream
endpoint and records durable request/response provenance to Eventloom.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from zaxy.event import EventLog
from zaxy.security import eventlog_path, validate_session_id

_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

_CAPTURED_HEADER_ALLOWLIST = {
    "content-type",
    "openai-organization",
    "openai-project",
    "user-agent",
}


@dataclass(frozen=True)
class PacketAnalyzerConfig:
    """Runtime settings for observe-only packet capture."""

    eventloom_path: Path
    session_id: str
    upstream_base_url: str
    upstream_api_key: str | None = None
    source: str = "llm-packet-analyzer"


@dataclass(frozen=True)
class PacketResponse:
    """HTTP response returned by the upstream provider."""

    status_code: int
    headers: dict[str, str]
    body: bytes


class EventloomPacketSink:
    """Background Eventloom writer for packet events."""

    def __init__(self, eventloom_dir: Path, session_id: str) -> None:
        self._log = EventLog(eventlog_path(eventloom_dir, session_id))
        self._session_id = validate_session_id(session_id)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="zaxy-packet-sink", daemon=True)
        self._thread.start()

    def enqueue(self, payload: dict[str, Any]) -> None:
        """Queue a packet event for durable append."""
        self._queue.put(payload)

    def close(self) -> None:
        """Flush pending events and stop the background writer."""
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                self._log.append(
                    "llm.packet.completed",
                    actor="zaxy-packet-analyzer",
                    payload=payload,
                    thread=self._session_id,
                )
            finally:
                self._queue.task_done()


class LlmPacketAnalyzer:
    """Observe-only OpenAI-compatible pass-through analyzer."""

    def __init__(
        self,
        config: PacketAnalyzerConfig,
        *,
        client: httpx.Client | None = None,
        sink: EventloomPacketSink | None = None,
    ) -> None:
        self._config = config
        self._session_id = validate_session_id(config.session_id)
        self._client = client or httpx.Client(timeout=None)
        self._own_client = client is None
        self._sink = sink or EventloomPacketSink(config.eventloom_path, self._session_id)
        self._own_sink = sink is None

    def forward(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        body: bytes,
    ) -> PacketResponse:
        """Forward one HTTP request and enqueue packet provenance."""
        upstream_response = self._client.request(
            method,
            self._upstream_url(path),
            headers=self._forward_headers(headers),
            content=body,
        )
        response_body = upstream_response.content
        request_body = _json_body(body)
        response_json = _json_body(response_body)
        payload = {
            "source": self._config.source,
            "session_id": self._session_id,
            "method": method.upper(),
            "provider_path": path,
            "status_code": upstream_response.status_code,
            "model": _model_from_packet(request_body, response_json),
            "usage_counts": _usage_counts_from_response(response_json),
            "request_hash": _stable_hash_bytes(body),
            "response_hash": _stable_hash_bytes(response_body),
            "request": {
                "headers": _captured_headers(headers),
                "body": request_body,
            },
            "response": {
                "headers": _captured_headers(dict(upstream_response.headers)),
                "body": response_json,
            },
        }
        self._sink.enqueue(payload)
        return PacketResponse(
            status_code=upstream_response.status_code,
            headers=_response_headers(upstream_response.headers),
            body=response_body,
        )

    def close(self) -> None:
        """Close owned resources after flushing queued packet events."""
        if self._own_sink:
            self._sink.close()
        if self._own_client:
            self._client.close()

    def _upstream_url(self, path: str) -> str:
        base = self._config.upstream_base_url.rstrip("/") + "/"
        request_path = path.lstrip("/")
        base_path = httpx.URL(base).path.strip("/")
        if base_path and request_path.startswith(f"{base_path}/"):
            request_path = request_path[len(base_path) + 1 :]
        return urljoin(base, request_path)

    def _forward_headers(self, headers: dict[str, str]) -> dict[str, str]:
        forwarded = {
            key: value
            for key, value in headers.items()
            if key.casefold() not in _HOP_BY_HOP_HEADERS
        }
        if self._config.upstream_api_key:
            forwarded["authorization"] = f"Bearer {self._config.upstream_api_key}"
        return forwarded


def _json_body(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw_sha256": _stable_hash_bytes(body), "bytes": len(body)}


def _stable_hash_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _captured_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key.casefold(): value
        for key, value in headers.items()
        if key.casefold() in _CAPTURED_HEADER_ALLOWLIST
    }


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.casefold() not in _HOP_BY_HOP_HEADERS
    }


def _model_from_packet(request_body: Any, response_body: Any) -> str | None:
    if isinstance(response_body, dict) and isinstance(response_body.get("model"), str):
        return str(response_body["model"])
    if isinstance(request_body, dict) and isinstance(request_body.get("model"), str):
        return str(request_body["model"])
    return None


def _usage_counts_from_response(response_body: Any) -> dict[str, Any] | None:
    if isinstance(response_body, dict) and isinstance(response_body.get("usage"), dict):
        usage = response_body["usage"]
        return {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
            "total": usage.get("total_tokens"),
        }
    return None


def run_packet_analyzer(
    *,
    host: str,
    port: int,
    config: PacketAnalyzerConfig,
) -> None:
    """Run a blocking HTTP packet analyzer server."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    analyzer = LlmPacketAnalyzer(config)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            response = analyzer.forward(
                "POST",
                self.path,
                headers=dict(self.headers),
                body=body,
            )
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        analyzer.close()
