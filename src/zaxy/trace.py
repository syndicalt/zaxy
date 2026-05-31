"""Pathlight observability hooks.

Wraps the Pathlight Python SDK to emit spans for every memory operation.
If Pathlight is unavailable (collector down), tracing fails silently so
that memory operations are never blocked by observability.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from zaxy.config import get_settings
from zaxy.event import Event
from zaxy.security import query_hash

AsyncPathlight: Any = None

try:
    from pathlight import AsyncPathlight as _AsyncPathlight

    AsyncPathlight = _AsyncPathlight
    _HAS_PATHLIGHT = True
except ImportError:
    _HAS_PATHLIGHT = False


class MemoryTracer:
    """Tracer for memory fabric operations.

    Args:
        base_url: Pathlight collector URL. Defaults to PATHLIGHT_URL env var
            or http://localhost:4100.
        project_id: Optional project identifier.
        disabled: If True, all tracing is no-ops.
    """

    def __init__(
        self,
        base_url: str | None = None,
        project_id: str | None = None,
        disabled: bool = False,
        trace_raw_queries: bool | None = None,
    ) -> None:
        self.disabled = disabled or not _HAS_PATHLIGHT
        self._client: Any | None = None
        settings = get_settings()
        self.trace_raw_queries = (
            settings.trace_raw_queries if trace_raw_queries is None else trace_raw_queries
        )
        if not self.disabled:
            self.base_url = base_url or os.getenv("PATHLIGHT_URL", "http://localhost:4100")
            self.project_id = project_id

    async def connect(self) -> None:
        """Initialize the Pathlight client."""
        if self.disabled:
            return
        self._client = AsyncPathlight(
            base_url=self.base_url or "http://localhost:4100",
            project_id=self.project_id,
        )

    async def close(self) -> None:
        """Close the Pathlight client."""
        if self._client and hasattr(self._client, "close"):
            await self._client.close()
        self._client = None

    @asynccontextmanager
    async def span(
        self,
        name: str,
        span_type: str = "custom",
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Context manager that traces a block of code.

        Yields a mutable dict where callers can set ``output``, ``error``,
        and ``duration_ms`` before exiting the context.
        """
        if self.disabled or self._client is None:
            yield {}
            return

        trace = self._client.trace("zaxy-memory", metadata=metadata or {})
        if inspect.isawaitable(trace):
            trace = await trace
        sp = trace.span(name, type=span_type, input=metadata)
        if inspect.isawaitable(sp):
            sp = await sp
        result: dict[str, Any] = {}
        try:
            yield result
        except Exception as exc:
            result["error"] = str(exc)
            raise
        finally:
            await sp.end(
                output=result.get("output"),
                error=result.get("error"),
            )
            await trace.end()

    async def trace_append(
        self,
        event_type: str,
        actor: str,
        seq: int,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Fire-and-forget trace for an append operation."""
        if self.disabled:
            return
        async with self.span("memory_append", metadata={"event_type": event_type, "actor": actor, "seq": seq}) as result:
            result["output"] = {"seq": seq, "success": success}
            if error:
                result["error"] = error

    async def trace_query(
        self,
        query: str,
        result_count: int,
        duration_ms: float,
        temporal_filter: str | None = None,
    ) -> None:
        """Fire-and-forget trace for a query operation."""
        if self.disabled:
            return
        metadata = {"query_hash": query_hash(query), "temporal_filter": temporal_filter}
        if self.trace_raw_queries:
            metadata["query"] = query
        async with self.span("memory_query", metadata=metadata) as result:
            result["output"] = {"result_count": result_count, "duration_ms": duration_ms}


@dataclass(frozen=True)
class TraceSpan:
    """Provider-neutral span derived from a durable Eventloom event."""

    span_id: str
    name: str
    kind: str
    event_type: str
    session_id: str
    event_seq: int
    event_hash: str
    timestamp: str
    actor: str
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "event_seq": self.event_seq,
            "event_hash": self.event_hash,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class TraceEdge:
    """Directed relationship between two neutral trace spans."""

    source: str
    target: str
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "relation": self.relation}


@dataclass(frozen=True)
class TraceCorrelation:
    """Neutral trace graph for local JSONL or provider-specific exporters."""

    spans: list[TraceSpan]
    edges: list[TraceEdge]

    @property
    def summary(self) -> dict[str, int]:
        return {
            "span_count": len(self.spans),
            "edge_count": len(self.edges),
            "mission_count": sum(1 for span in self.spans if span.kind == "mission"),
            "finding_count": sum(1 for span in self.spans if span.kind == "finding"),
            "model_call_count": sum(1 for span in self.spans if span.kind == "model_call"),
            "tool_call_count": sum(1 for span in self.spans if span.kind == "tool_call"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "zaxy.trace.v0.8",
            "summary": self.summary,
            "spans": [span.to_dict() for span in self.spans],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def build_trace_correlation(events: list[Event]) -> TraceCorrelation:
    """Build provider-neutral trace spans and edges from replayed Eventloom events."""
    ordered = sorted(events, key=lambda event: (event.timestamp, event.thread, event.seq))
    spans = [_trace_span(event) for event in ordered]
    span_by_event = {(event.thread, event.seq): span for event, span in zip(ordered, spans, strict=True)}
    mission_spans: dict[str, str] = {}
    finding_spans: dict[str, str] = {}
    last_model_call_by_session: dict[str, str] = {}
    edges: list[TraceEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for event in ordered:
        span_id = span_by_event[(event.thread, event.seq)].span_id
        mission_id = _payload_text(event, "mission_id")
        finding_id = _payload_text(event, "finding_id")
        if event.type == "coordination.mission.created" and mission_id:
            mission_spans[mission_id] = span_id
        if event.type == "coordination.finding.reported" and finding_id:
            finding_spans[finding_id] = span_id
        if event.type == "model.call.requested":
            last_model_call_by_session[event.thread] = span_id
        if event.type != "coordination.mission.created":
            mission_span = mission_spans.get(mission_id) or mission_spans.get(event.thread)
            if mission_span:
                _add_edge(edges, seen_edges, mission_span, span_id, "contains")
        if event.type == "tool.call.completed":
            model_span = last_model_call_by_session.get(event.thread)
            if model_span:
                _add_edge(edges, seen_edges, model_span, span_id, "observed_tool_call")
        if event.type == "transcript.turn":
            model_span = last_model_call_by_session.get(event.thread)
            if model_span:
                _add_edge(edges, seen_edges, model_span, span_id, "responded_with")
        if event.type == "coordination.finding.reviewed" and finding_id:
            finding_span = finding_spans.get(finding_id)
            if finding_span:
                _add_edge(edges, seen_edges, span_id, finding_span, "reviews")
        if event.type == "coordination.finding.promoted" and finding_id:
            finding_span = finding_spans.get(finding_id)
            if finding_span:
                _add_edge(edges, seen_edges, span_id, finding_span, "promotes")
    return TraceCorrelation(spans=spans, edges=edges)


def _trace_span(event: Event) -> TraceSpan:
    kind = _span_kind(event.type)
    return TraceSpan(
        span_id=f"event:{event.thread}:{event.seq}",
        name=_span_name(event, kind),
        kind=kind,
        event_type=event.type,
        session_id=event.thread,
        event_seq=event.seq,
        event_hash=event.hash,
        timestamp=event.timestamp,
        actor=event.actor,
        attributes=_safe_trace_attributes(event),
    )


def _span_kind(event_type: str) -> str:
    if event_type == "coordination.mission.created":
        return "mission"
    if event_type == "coordination.finding.reported":
        return "finding"
    if event_type == "coordination.finding.reviewed":
        return "review"
    if event_type == "coordination.finding.promoted":
        return "promotion"
    if event_type == "coordination.handoff.created":
        return "handoff"
    if event_type == "memory.checkout.completed":
        return "checkout"
    if event_type == "model.call.requested":
        return "model_call"
    if event_type == "tool.call.completed":
        return "tool_call"
    if event_type == "transcript.turn":
        return "transcript"
    if event_type.startswith("coordination."):
        return "coordination"
    return "event"


def _span_name(event: Event, kind: str) -> str:
    identifiers = [
        _payload_text(event, "mission_id"),
        _payload_text(event, "worker_id"),
        _payload_text(event, "finding_id"),
        _payload_text(event, "handoff_id"),
    ]
    suffix = next((value for value in identifiers if value), "")
    if suffix:
        return f"{kind}:{suffix}"
    return kind


def _safe_trace_attributes(event: Event) -> dict[str, Any]:
    allowed_by_type: dict[str, set[str]] = {
        "coordination.mission.created": {"mission_id", "objective"},
        "coordination.worker.created": {"mission_id", "worker_id", "role"},
        "coordination.assignment.created": {"mission_id", "worker_id", "assignment_id", "summary"},
        "coordination.finding.reported": {
            "mission_id",
            "worker_id",
            "finding_id",
            "claim_key",
            "claim_value",
            "confidence",
            "status",
            "stale",
            "superseded_by",
        },
        "coordination.finding.reviewed": {"mission_id", "finding_id", "status"},
        "coordination.finding.promoted": {"mission_id", "finding_id"},
        "coordination.handoff.created": {"mission_id", "handoff_id", "status"},
        "coordination.conflict.detected": {"mission_id", "claim_key", "conflict_type", "finding_ids"},
        "memory.checkout.completed": {"mission_id", "query", "source", "token_efficiency"},
        "model.call.requested": {
            "provider",
            "model",
            "query",
            "message_count",
            "injected_memory",
            "mission_id",
            "session_id",
        },
        "tool.call.completed": {"tool_name", "status", "session_id", "call_id", "argument_keys", "arguments_redacted"},
        "transcript.turn": {"source", "role", "session_id", "model", "query", "turn_index"},
    }
    allowed = allowed_by_type.get(event.type, {"mission_id", "worker_id", "finding_id", "handoff_id", "source"})
    return {
        key: value
        for key, value in event.payload.items()
        if key in allowed and _trace_value_is_safe(value)
    }


def _trace_value_is_safe(value: Any) -> bool:
    if isinstance(value, str | int | float | bool) or value is None:
        return True
    if isinstance(value, list):
        return all(isinstance(item, str | int | float | bool) or item is None for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and (isinstance(item, str | int | float | bool) or item is None)
            for key, item in value.items()
        )
    return False


def _payload_text(event: Event, key: str) -> str:
    value = event.payload.get(key)
    return value if isinstance(value, str) else ""


def _add_edge(
    edges: list[TraceEdge],
    seen: set[tuple[str, str, str]],
    source: str,
    target: str,
    relation: str,
) -> None:
    edge_key = (source, target, relation)
    if edge_key in seen or source == target:
        return
    seen.add(edge_key)
    edges.append(TraceEdge(source=source, target=target, relation=relation))
