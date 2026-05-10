"""Cold-path projection for captured LLM packets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zaxy.event import Event, EventLog
from zaxy.security import eventlog_path, validate_session_id

MAX_PACKET_EXCERPT_CHARS = 600


@dataclass(frozen=True)
class PacketProjectionResult:
    """Summary of one packet projection pass."""

    read: int
    projected: int
    skipped: int


@dataclass(frozen=True)
class PacketProjectionWatchResult:
    """Summary of a packet projection watch loop."""

    iterations: int
    read: int
    projected: int
    skipped: int


def project_packet_events(
    *,
    eventloom_path: Path,
    session_id: str,
    from_seq: int = 1,
    limit: int | None = None,
) -> PacketProjectionResult:
    """Project completed packet events into compact memory-ready summaries."""
    validated_session_id = validate_session_id(session_id)
    log = EventLog(eventlog_path(eventloom_path, validated_session_id))
    events = log.read_all()
    projected_hashes = {
        event.payload.get("source_event_hash")
        for event in events
        if event.type == "llm.packet.projected"
    }
    candidates = [
        event
        for event in events
        if event.type == "llm.packet.completed" and event.seq >= from_seq
    ]
    if limit is not None:
        candidates = candidates[: max(limit, 0)]

    projected = 0
    skipped = 0
    for event in candidates:
        if event.hash in projected_hashes:
            skipped += 1
            continue
        payload = build_packet_projection_payload(event)
        log.append(
            "llm.packet.projected",
            actor="zaxy-packet-projector",
            payload=payload,
            thread=validated_session_id,
        )
        projected_hashes.add(event.hash)
        projected += 1

    return PacketProjectionResult(read=len(candidates), projected=projected, skipped=skipped)


def watch_packet_events(
    *,
    eventloom_path: Path,
    session_id: str,
    interval_seconds: float = 2.0,
    from_seq: int = 1,
    limit: int | None = None,
    max_iterations: int | None = None,
) -> PacketProjectionWatchResult:
    """Continuously project completed packet events until bounded or interrupted."""
    iterations = 0
    total_read = 0
    total_projected = 0
    total_skipped = 0
    while max_iterations is None or iterations < max_iterations:
        result = project_packet_events(
            eventloom_path=eventloom_path,
            session_id=session_id,
            from_seq=from_seq,
            limit=limit,
        )
        iterations += 1
        total_read += result.read
        total_projected += result.projected
        total_skipped += result.skipped
        if max_iterations is not None and iterations >= max_iterations:
            break
        if interval_seconds > 0:
            time.sleep(interval_seconds)
    return PacketProjectionWatchResult(
        iterations=iterations,
        read=total_read,
        projected=total_projected,
        skipped=total_skipped,
    )


def build_packet_projection_payload(event: Event) -> dict[str, Any]:
    """Build a compact projection payload for one completed packet event."""
    request_body = _nested_dict(event.payload, "request", "body")
    response_body = _nested_dict(event.payload, "response", "body")
    request_summary = _request_summary(request_body)
    response_summary = _response_summary(response_body)
    provider_path = _optional_text(event.payload.get("provider_path")) or "unknown-provider-path"
    status_code = _optional_int(event.payload.get("status_code"), default=0)
    model = _optional_text(event.payload.get("model"))
    summary = _packet_summary(
        provider_path=provider_path,
        status_code=status_code,
        model=model,
        request_summary=request_summary,
        response_summary=response_summary,
    )
    return {
        "source": "llm-packet-analyzer",
        "session_id": _optional_text(event.payload.get("session_id")) or event.thread,
        "source_event_seq": event.seq,
        "source_event_hash": event.hash,
        "provider_path": provider_path,
        "status_code": status_code,
        "model": model,
        "usage_counts": event.payload.get("usage_counts"),
        "summary": summary,
        "request_summary": request_summary,
        "response_summary": response_summary,
    }


def _packet_summary(
    *,
    provider_path: str,
    status_code: int,
    model: str | None,
    request_summary: dict[str, Any],
    response_summary: dict[str, Any],
) -> str:
    model_text = f" {model}" if model else ""
    parts = [f"LLM packet {provider_path}{model_text} status {status_code}."]
    if request_summary.get("last_user_message"):
        parts.append(f"User: {request_summary['last_user_message']}")
    elif request_summary.get("input"):
        parts.append(f"Input: {request_summary['input']}")
    if response_summary.get("assistant_message"):
        parts.append(f"Assistant: {response_summary['assistant_message']}")
    elif response_summary.get("output_text"):
        parts.append(f"Output: {response_summary['output_text']}")
    elif response_summary.get("raw_response_bytes"):
        parts.append(f"Response body: {response_summary['raw_response_bytes']} bytes.")
    return " ".join(parts)


def _request_summary(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages")
    if isinstance(messages, list):
        last_user = _last_message_content(messages, role="user")
        return {
            "message_count": len(messages),
            "last_user_message": _excerpt(last_user) if last_user else None,
        }
    input_text = _text_from_value(body.get("input"))
    if input_text:
        return {"input": _excerpt(input_text)}
    return {}


def _response_summary(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if isinstance(choices, list):
        assistant = _assistant_choice_content(choices)
        if assistant:
            return {"assistant_message": _excerpt(assistant)}
    output_text = _text_from_value(body.get("output_text"))
    if output_text:
        return {"output_text": _excerpt(output_text)}
    raw_bytes = _optional_int(body.get("bytes"), default=0)
    if raw_bytes:
        return {"raw_response_bytes": raw_bytes}
    return {}


def _last_message_content(messages: list[Any], *, role: str) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        content = _text_from_value(message.get("content"))
        if content:
            return content
    return None


def _assistant_choice_content(choices: list[Any]) -> str | None:
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            content = _text_from_value(message.get("content"))
            if content:
                return content
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = _text_from_value(delta.get("content"))
            if content:
                return content
    return None


def _nested_dict(value: dict[str, Any], *path: str) -> dict[str, Any]:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _text_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        text = " ".join(part.strip() for part in parts if part.strip())
        return text or None
    return None


def _excerpt(value: str) -> str:
    text = " ".join(value.split())
    if len(text) <= MAX_PACKET_EXCERPT_CHARS:
        return text
    return text[: MAX_PACKET_EXCERPT_CHARS - 3].rstrip() + "..."


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
