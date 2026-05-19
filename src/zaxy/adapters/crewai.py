"""Dependency-light CrewAI adapter preview for Zaxy memory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from zaxy.adapters._common import FabricFactory, assembly_payload, default_fabric_factory
from zaxy.context import Context
from zaxy.core import ContextAssembly
from zaxy.observation import build_tool_call_observation

CrewAIMemoryStep = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class CrewAIMemoryAdapter:
    """Small adapter for CrewAI task lifecycle callbacks.

    The adapter intentionally avoids importing CrewAI. CrewAI applications can
    call these methods from before/after task hooks, callbacks, or custom task
    wrappers while keeping ownership of their runtime objects.
    """

    session_id: str = "default"
    eventloom_path: str = ".eventloom"
    source: str = "crewai"
    max_recent_events: int = 20
    limit: int = 10
    fabric_factory: FabricFactory = default_fabric_factory

    async def before_task(
        self,
        task_input: str,
        *,
        query: str | None = None,
        crew: str | None = None,
        agent: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Capture task input and return prompt-ready memory for the task."""
        resolved_query = query or task_input or "crew context"
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            if task_input:
                assembly = await fabric.after_turn(
                    role="user",
                    content=task_input,
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
            return self._task_payload(assembly, crew=crew, agent=agent, task_id=task_id)
        finally:
            await fabric.close()

    async def checkout_before_task(
        self,
        task_input: str,
        *,
        query: str | None = None,
        crew: str | None = None,
        agent: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Return Memory Checkout payload for task-boundary middleware."""
        resolved_query = query or task_input or "crew context"
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            checkout = await fabric.checkout_memory(
                resolved_query,
                session_id=self.session_id,
                limit=self.limit,
                max_recent_events=self.max_recent_events,
            )
            payload = self._checkout_payload(checkout.to_dict())
            if crew is not None:
                payload["zaxy"]["crew"] = crew
            if agent is not None:
                payload["zaxy"]["agent"] = agent
            if task_id is not None:
                payload["zaxy"]["task_id"] = task_id
            return payload
        finally:
            await fabric.close()

    async def after_task(
        self,
        result: str,
        *,
        query: str | None = None,
        crew: str | None = None,
        agent: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Capture task output and return updated prompt-ready memory."""
        fabric = self.fabric_factory(self.eventloom_path)
        try:
            assembly = await fabric.after_turn(
                role="assistant",
                content=result,
                session_id=self.session_id,
                query=query or result or "crew result",
                source=self.source,
                max_recent_events=self.max_recent_events,
                limit=self.limit,
            )
            return self._task_payload(assembly, crew=crew, agent=agent, task_id=task_id)
        finally:
            await fabric.close()

    async def record_tool_use(
        self,
        *,
        tool_name: str,
        status: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
        result_summary: str | None = None,
    ) -> None:
        """Append a redacted tool-call observation for CrewAI tool usage."""
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
        payload: Mapping[str, Any],
        *,
        feedback: str = "used",
        importance: float | None = None,
    ) -> int:
        """Record retrieval feedback for contexts returned by before_task."""
        contexts = [context for context in payload.get("contexts", []) if isinstance(context, Context)]
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

    def _task_payload(
        self,
        assembly: ContextAssembly,
        *,
        crew: str | None,
        agent: str | None,
        task_id: str | None,
    ) -> dict[str, Any]:
        metadata = assembly_payload(assembly)
        if crew is not None:
            metadata["crew"] = crew
        if agent is not None:
            metadata["agent"] = agent
        if task_id is not None:
            metadata["task_id"] = task_id
        return {
            "memory": assembly.prompt,
            "contexts": assembly.contexts,
            "zaxy": metadata,
        }

    def _checkout_payload(self, checkout: dict[str, Any]) -> dict[str, Any]:
        return {
            "memory": str(checkout.get("prompt") or ""),
            "contexts": [],
            "zaxy": {
                "kind": "memory_checkout",
                "session_id": checkout.get("session_id", self.session_id),
                "query": checkout.get("query"),
                "current_fact_count": len(checkout.get("current_facts", []) or []),
                "warning_count": len(checkout.get("warnings", []) or []),
            },
        }


def create_crewai_memory_step(
    *,
    session_id: str = "default",
    eventloom_path: str = ".eventloom",
    source: str = "crewai",
    max_recent_events: int = 20,
    limit: int = 10,
    fabric_factory: FabricFactory = default_fabric_factory,
) -> CrewAIMemoryStep:
    """Return an async task helper that yields prompt-ready CrewAI memory."""
    adapter = CrewAIMemoryAdapter(
        session_id=session_id,
        eventloom_path=eventloom_path,
        source=source,
        max_recent_events=max_recent_events,
        limit=limit,
        fabric_factory=fabric_factory,
    )

    async def zaxy_crewai_memory_step(message: str) -> str:
        payload = await adapter.before_task(message)
        return str(payload["memory"])

    return zaxy_crewai_memory_step


def create_crewai_memory_checkout_step(
    *,
    session_id: str = "default",
    eventloom_path: str = ".eventloom",
    source: str = "crewai",
    max_recent_events: int = 20,
    limit: int = 10,
    fabric_factory: FabricFactory = default_fabric_factory,
) -> CrewAIMemoryStep:
    """Return an async task helper that yields Memory Checkout prompt text."""
    adapter = CrewAIMemoryAdapter(
        session_id=session_id,
        eventloom_path=eventloom_path,
        source=source,
        max_recent_events=max_recent_events,
        limit=limit,
        fabric_factory=fabric_factory,
    )

    async def zaxy_crewai_memory_checkout_step(message: str) -> str:
        payload = await adapter.checkout_before_task(message)
        return str(payload["memory"])

    return zaxy_crewai_memory_checkout_step
