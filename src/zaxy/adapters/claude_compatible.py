"""Dependency-light Claude-compatible model-call adapter for Zaxy memory."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from zaxy.adapters._common import (
    FabricFactory,
    default_fabric_factory,
    native_checkout_error_payload,
    native_checkout_payload,
)
from zaxy.observation import build_tool_call_observation, build_transcript_turn_observation


@dataclass(frozen=True)
class ClaudeCompatibleMemoryAdapter:
    """Inject Memory Checkout into Claude-style messages calls.

    The adapter intentionally avoids importing Anthropic or Claude SDKs. It
    accepts any object exposing ``client.messages.create(**request)`` and
    supports synchronous or asynchronous ``create`` methods.
    """

    session_id: str = "default"
    eventloom_path: str = ".eventloom"
    source: str = "claude-compatible"
    max_recent_events: int = 20
    limit: int = 10
    metadata_key: str = "zaxy"
    fabric_factory: FabricFactory = default_fabric_factory

    async def messages_create(
        self,
        client: Any,
        *,
        model: str,
        messages: list[dict[str, Any]],
        query: str | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a Claude-style messages call with Memory Checkout as system text."""
        resolved_query = query or _latest_user_content(messages) or "model context"
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            try:
                checkout = await fabric.checkout_memory(
                    resolved_query,
                    session_id=self.session_id,
                    limit=self.limit,
                    max_recent_events=self.max_recent_events,
                )
            except Exception as exc:
                original_messages = [dict(message) for message in messages]
                return {
                    "response": None,
                    "request": {"model": model, "messages": original_messages, **kwargs},
                    "messages": original_messages,
                    self.metadata_key: native_checkout_error_payload(
                        exc,
                        framework="claude-compatible",
                        operation="messages_create",
                        source=self.source,
                        session_id=self.session_id,
                        query=resolved_query,
                    ),
                    "checkout": None,
                    "assistant_content": "",
                }
            checkout_payload = checkout.to_dict()
            zaxy_payload = native_checkout_payload(
                checkout_payload,
                framework="claude-compatible",
                operation="messages_create",
                source=self.source,
                session_id=self.session_id,
            )
            system_prompt = _join_system_prompt(str(checkout_payload.get("prompt") or ""), system)
            request_messages = [dict(message) for message in messages]
            request = {
                "model": model,
                "system": system_prompt,
                "messages": request_messages,
                **kwargs,
            }
            await fabric.append(
                "model.call.requested",
                actor=self.source,
                payload={
                    "provider": "claude-compatible",
                    "model": model,
                    "query": resolved_query,
                    "session_id": self.session_id,
                    "message_count": len(request_messages),
                    "injected_memory": True,
                    self.metadata_key: zaxy_payload,
                },
                session_id=self.session_id,
            )
            response = await _maybe_await(client.messages.create(**request))
            assistant_content = _assistant_content(response)
            if assistant_content:
                event = build_transcript_turn_observation(
                    role="assistant",
                    content=assistant_content,
                    session_id=self.session_id,
                    source=self.source,
                )
                event["payload"]["model"] = model
                event["payload"]["query"] = resolved_query
                await fabric.append(
                    event["event_type"],
                    actor=event["actor"],
                    payload=event["payload"],
                    session_id=self.session_id,
                )
            return {
                "response": response,
                "request": request,
                "messages": request_messages,
                self.metadata_key: zaxy_payload,
                "checkout": checkout_payload,
                "assistant_content": assistant_content,
            }
        finally:
            await fabric.close()

    async def record_tool_call(
        self,
        *,
        tool_name: str,
        status: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
        result_summary: str | None = None,
    ) -> None:
        """Append a redacted tool-call observation for provider tool usage."""
        event = build_tool_call_observation(
            tool_name=tool_name,
            status=status,
            session_id=self.session_id,
            source=self.source,
            call_id=call_id,
            arguments=arguments,
            result_summary=result_summary,
        )
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            await fabric.append(
                event["event_type"],
                actor=event["actor"],
                payload=event["payload"],
                session_id=self.session_id,
            )
        finally:
            await fabric.close()

    async def record_feedback(
        self,
        *,
        entity_name: str,
        entity_type: str,
        feedback: str = "used",
        query: str | None = None,
        score: float | None = None,
        citation: str | None = None,
        reason: str | None = None,
        importance: float | None = None,
    ) -> None:
        """Append memory feedback for context used by a model call."""
        normalized = feedback.strip().lower() or "used"
        payload: dict[str, Any] = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "query": query,
            "source": self.source,
            "score": score,
            "citation": citation,
            "reason": reason,
        }
        event_type = "memory.feedback"
        if normalized in {"used", "helpful"}:
            event_type = "memory.reinforced"
            if importance is not None:
                payload["importance"] = max(0.0, min(1.0, float(importance)))
        else:
            payload["feedback"] = normalized
        payload = {key: value for key, value in payload.items() if value is not None}
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            await fabric.append(
                event_type,
                actor=self.source,
                payload=payload,
                session_id=self.session_id,
            )
        finally:
            await fabric.close()


def create_claude_compatible_memory_adapter(
    *,
    session_id: str = "default",
    eventloom_path: str = ".eventloom",
    source: str = "claude-compatible",
    max_recent_events: int = 20,
    limit: int = 10,
    fabric_factory: FabricFactory = default_fabric_factory,
) -> ClaudeCompatibleMemoryAdapter:
    """Return a dependency-light adapter for Claude-style messages clients."""
    return ClaudeCompatibleMemoryAdapter(
        session_id=session_id,
        eventloom_path=eventloom_path,
        source=source,
        max_recent_events=max_recent_events,
        limit=limit,
        fabric_factory=fabric_factory,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _latest_user_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _assistant_content(response: Any) -> str:
    content = _get(response, "content") or []
    parts: list[str] = []
    for block in content:
        text = _get(block, "text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)


def _join_system_prompt(memory_prompt: str, system: str | None) -> str:
    if system and system.strip():
        return "\n\n".join([memory_prompt, system.strip()])
    return memory_prompt


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
