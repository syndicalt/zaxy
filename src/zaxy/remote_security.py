"""Remote MCP/SSE rate limiting and audit export helpers."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of a remote request rate-limit check."""

    allowed: bool
    retry_after_seconds: int


class SessionRateLimiter:
    """Fixed-window request limiter keyed by remote session ID."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: int,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._clock = clock
        self._buckets: dict[str, tuple[float, int]] = {}

    def check(self, session_id: str) -> RateLimitDecision:
        """Return whether the session may make another request."""
        if not self.enabled:
            return RateLimitDecision(allowed=True, retry_after_seconds=0)

        now = self._clock()
        window_start, count = self._buckets.get(session_id, (now, 0))
        elapsed = now - window_start
        if elapsed >= self.window_seconds:
            window_start = now
            count = 0

        if count >= self.max_requests:
            retry_after = max(1, math.ceil(self.window_seconds - elapsed))
            return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)

        self._buckets[session_id] = (window_start, count + 1)
        return RateLimitDecision(allowed=True, retry_after_seconds=0)


AuditOutcome = Literal["allowed", "denied_auth", "denied_rate_limit"]


@dataclass(frozen=True)
class RemoteAuditEvent:
    """Compact audit record for a remote MCP/SSE request decision."""

    timestamp: str
    session_id: str | None
    route: str
    method: str
    outcome: AuditOutcome
    reason: str | None
    client_host: str | None


class AuditEventExporter:
    """Append remote request audit events as newline-delimited JSON."""

    def __init__(self, *, path: Path, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled

    def write(self, event: RemoteAuditEvent) -> None:
        """Append one audit event unless export is disabled."""
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value
            for key, value in asdict(event).items()
            if value is not None
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
