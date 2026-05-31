"""Dependency-light LangGraph native-beta adapter for Zaxy memory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from zaxy.adapters._common import (
    FabricFactory,
    assembly_payload,
    default_fabric_factory,
    native_checkout_error_payload,
    native_checkout_payload,
)
from zaxy.adapters.coordination import CoordinationAdapter
from zaxy.context import Context
from zaxy.core import ContextAssembly
from zaxy.observation import build_tool_call_observation

LangGraphNode = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class LangGraphMemoryAdapter:
    """Small adapter that fits LangGraph's async node shape.

    The adapter intentionally avoids importing LangGraph so applications can use
    it with whichever LangGraph version owns their state schema.
    """

    session_id: str = "default"
    eventloom_path: str = ".eventloom"
    source: str = "langgraph"
    max_recent_events: int = 20
    limit: int = 10
    context_key: str = "zaxy_context"
    context_list_key: str = "zaxy_contexts"
    metadata_key: str = "zaxy"
    fabric_factory: FabricFactory = default_fabric_factory

    async def before_model(
        self,
        state: Mapping[str, Any],
        *,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Capture the latest user turn and inject assembled context."""
        role, content = _latest_message(state)
        resolved_query = query or content or "langgraph context"
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            if content:
                assembly = await fabric.after_turn(
                    role=role,
                    content=content,
                    session_id=self.session_id,
                    query=resolved_query,
                    source=self.source,
                    max_recent_events=self.max_recent_events,
                    limit=self.limit,
                )
            else:
                assembly = await fabric.assemble_context(
                    resolved_query,
                    session_id=self.session_id,
                    max_recent_events=self.max_recent_events,
                    limit=self.limit,
                )
            return self._with_context(state, assembly)
        finally:
            await fabric.close()

    async def checkout_before_model(
        self,
        state: Mapping[str, Any],
        *,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Inject Memory Checkout at model/lifecycle boundaries."""
        _role, content = _latest_message(state)
        resolved_query = query or content or "langgraph context"
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            checkout = await fabric.checkout_memory(
                resolved_query,
                session_id=self.session_id,
                limit=self.limit,
                max_recent_events=self.max_recent_events,
            )
            return self._with_checkout(state, checkout.to_dict())
        except Exception as exc:
            return self._with_checkout_error(state, resolved_query, exc)
        finally:
            await fabric.close()

    async def record_assistant_turn(
        self,
        content: str,
        *,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Persist an assistant turn and return prompt-ready context metadata."""
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            assembly = await fabric.after_turn(
                role="assistant",
                content=content,
                session_id=self.session_id,
                query=query or content or "assistant context",
                source=self.source,
                max_recent_events=self.max_recent_events,
                limit=self.limit,
            )
            return assembly_payload(assembly)
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
        """Append a redacted tool-call observation for LangGraph tool nodes."""
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

    async def record_context_feedback(
        self,
        state: Mapping[str, Any],
        *,
        feedback: str = "used",
        importance: float | None = None,
    ) -> int:
        """Record retrieval feedback for contexts projected into state."""
        contexts = [context for context in state.get(self.context_list_key, []) if isinstance(context, Context)]
        if not contexts:
            return 0
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            return int(
                await fabric.record_context_feedback(
                    contexts,
                    feedback=feedback,
                    session_id=self.session_id,
                    actor=self.source,
                    importance=importance,
                )
            )
        finally:
            await fabric.close()

    def _with_context(
        self,
        state: Mapping[str, Any],
        assembly: ContextAssembly,
    ) -> dict[str, Any]:
        updated = dict(state)
        updated[self.context_key] = assembly.prompt
        updated[self.context_list_key] = assembly.contexts
        updated[self.metadata_key] = assembly_payload(assembly)
        return updated

    def _with_checkout(
        self,
        state: Mapping[str, Any],
        checkout: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(state)
        updated[self.context_key] = str(checkout.get("prompt") or "")
        updated[self.context_list_key] = []
        updated[self.metadata_key] = native_checkout_payload(
            checkout,
            framework="langgraph",
            operation="before_model",
            source=self.source,
            session_id=self.session_id,
        )
        return updated

    def _with_checkout_error(
        self,
        state: Mapping[str, Any],
        query: str,
        exc: Exception,
    ) -> dict[str, Any]:
        updated = dict(state)
        updated[self.context_key] = ""
        updated[self.context_list_key] = []
        updated[self.metadata_key] = native_checkout_error_payload(
            exc,
            framework="langgraph",
            operation="before_model",
            source=self.source,
            session_id=self.session_id,
            query=query,
        )
        return updated


def create_langgraph_memory_node(
    *,
    session_id: str = "default",
    eventloom_path: str = ".eventloom",
    source: str = "langgraph",
    max_recent_events: int = 20,
    limit: int = 10,
    fabric_factory: FabricFactory = default_fabric_factory,
) -> LangGraphNode:
    """Return an async node that injects Zaxy context into LangGraph state."""
    adapter = LangGraphMemoryAdapter(
        session_id=session_id,
        eventloom_path=eventloom_path,
        source=source,
        max_recent_events=max_recent_events,
        limit=limit,
        fabric_factory=fabric_factory,
    )

    async def zaxy_langgraph_memory_node(state: dict[str, Any]) -> dict[str, Any]:
        return await adapter.before_model(state)

    return zaxy_langgraph_memory_node


def create_langgraph_memory_checkout_node(
    *,
    session_id: str = "default",
    eventloom_path: str = ".eventloom",
    source: str = "langgraph",
    max_recent_events: int = 20,
    limit: int = 10,
    fabric_factory: FabricFactory = default_fabric_factory,
) -> LangGraphNode:
    """Return an async node that injects Memory Checkout at model boundaries."""
    adapter = LangGraphMemoryAdapter(
        session_id=session_id,
        eventloom_path=eventloom_path,
        source=source,
        max_recent_events=max_recent_events,
        limit=limit,
        fabric_factory=fabric_factory,
    )

    async def zaxy_langgraph_memory_checkout_node(state: dict[str, Any]) -> dict[str, Any]:
        return await adapter.checkout_before_model(state)

    return zaxy_langgraph_memory_checkout_node


def create_langgraph_coordination_node(
    *,
    mission_id: str,
    worker_id: str,
    eventloom_path: str = ".eventloom",
    actor: str | None = None,
) -> LangGraphNode:
    """Return an async node that reports explicit Coordinate findings."""

    async def zaxy_langgraph_coordination_node(state: dict[str, Any]) -> dict[str, Any]:
        adapter = CoordinationAdapter(eventloom_path=eventloom_path, actor=actor or worker_id)
        finding = adapter.report_finding(
            mission_id,
            worker_id,
            summary=str(state.get("coordination_summary") or state.get("latest_message") or ""),
            evidence=list(state.get("coordination_evidence") or []),
            confidence=state.get("coordination_confidence"),
            claim_key=state.get("coordination_claim_key"),
            claim_value=state.get("coordination_claim_value"),
        )
        return {**state, "zaxy_coordination": finding}

    return zaxy_langgraph_coordination_node


def _latest_message(state: Mapping[str, Any]) -> tuple[str, str]:
    latest = state.get("latest_message")
    if latest is not None:
        return _message_role_content(latest)
    messages = state.get("messages")
    if isinstance(messages, list) and messages:
        return _message_role_content(messages[-1])
    return "user", ""


def _message_role_content(message: Any) -> tuple[str, str]:
    if isinstance(message, str):
        return "user", message
    if isinstance(message, Mapping):
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        return role, content
    role = str(getattr(message, "role", "user") or "user")
    content = str(getattr(message, "content", "") or "")
    return role, content
