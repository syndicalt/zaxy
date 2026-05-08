"""Tests for remote MCP security helpers."""

from __future__ import annotations

import json

from zaxy.remote_security import AuditEventExporter, RemoteAuditEvent, SessionRateLimiter


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_session_rate_limiter_denies_after_limit() -> None:
    clock = FakeClock()
    limiter = SessionRateLimiter(max_requests=2, window_seconds=60, clock=clock)

    first = limiter.check("tenant-1")
    second = limiter.check("tenant-1")
    third = limiter.check("tenant-1")

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False
    assert third.retry_after_seconds == 60


def test_session_rate_limiter_resets_after_window() -> None:
    clock = FakeClock()
    limiter = SessionRateLimiter(max_requests=1, window_seconds=10, clock=clock)

    assert limiter.check("tenant-1").allowed is True
    assert limiter.check("tenant-1").allowed is False

    clock.now = 111.0

    assert limiter.check("tenant-1").allowed is True


def test_audit_event_exporter_writes_compact_jsonl_without_secrets(tmp_path) -> None:
    path = tmp_path / "remote_audit.jsonl"
    exporter = AuditEventExporter(path=path, enabled=True)

    exporter.write(
        RemoteAuditEvent(
            timestamp="2026-05-08T12:00:00Z",
            session_id="tenant-1",
            route="/messages/",
            method="POST",
            outcome="allowed",
            reason=None,
            client_host="127.0.0.1",
        )
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {
        "timestamp": "2026-05-08T12:00:00Z",
        "session_id": "tenant-1",
        "route": "/messages/",
        "method": "POST",
        "outcome": "allowed",
        "client_host": "127.0.0.1",
    }
    assert "authorization" not in record
    assert "token" not in record


def test_disabled_audit_exporter_does_not_create_file(tmp_path) -> None:
    path = tmp_path / "remote_audit.jsonl"
    exporter = AuditEventExporter(path=path, enabled=False)

    exporter.write(
        RemoteAuditEvent(
            timestamp="2026-05-08T12:00:00Z",
            session_id=None,
            route="/sse",
            method="GET",
            outcome="denied_auth",
            reason="Authorization bearer token is required",
            client_host=None,
        )
    )

    assert not path.exists()
