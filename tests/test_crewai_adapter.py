"""Tests for the native CrewAI adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from zaxy import CrewAIMemoryAdapter, create_crewai_coordination_step, create_crewai_memory_step
from zaxy.core import Context, ContextAssembly
from zaxy.integrations import list_framework_integration_specs


@dataclass
class FakeFabric:
    """Minimal async MemoryFabric test double."""

    calls: list[tuple[str, dict[str, Any]]]

    async def after_turn(self, **kwargs: Any) -> ContextAssembly:
        self.calls.append(("after_turn", kwargs))
        return _assembly()

    async def assemble_context(self, query: str, **kwargs: Any) -> ContextAssembly:
        self.calls.append(("assemble_context", {"query": query, **kwargs}))
        return _assembly()

    async def append(self, event_type: str, actor: str, payload: dict[str, Any], session_id: str) -> None:
        self.calls.append(
            (
                "append",
                {
                    "event_type": event_type,
                    "actor": actor,
                    "payload": payload,
                    "session_id": session_id,
                },
            )
        )

    async def record_context_feedback(self, contexts: list[Context], **kwargs: Any) -> int:
        self.calls.append(("record_context_feedback", {"contexts": contexts, **kwargs}))
        return len(contexts)

    async def checkout_memory(self, query: str, **kwargs: Any) -> Any:
        self.calls.append(("checkout_memory", {"query": query, **kwargs}))
        return _checkout()

    async def close(self) -> None:
        self.calls.append(("close", {}))


@dataclass
class FailingCheckoutFabric:
    """MemoryFabric test double that fails checkout but still needs closing."""

    calls: list[tuple[str, dict[str, Any]]]

    async def checkout_memory(self, query: str, **kwargs: Any) -> Any:
        self.calls.append(("checkout_memory", {"query": query, **kwargs}))
        raise RuntimeError("projection unavailable")

    async def close(self) -> None:
        self.calls.append(("close", {}))


def test_crewai_registry_marks_native_preview_adapter() -> None:
    """CrewAI should advertise the maintained native-preview adapter."""
    specs = {spec.framework: spec for spec in list_framework_integration_specs()}

    assert specs["crewai"].maturity == "native-preview"
    assert specs["crewai"].native_adapter == "zaxy.adapters.crewai"
    assert specs["crewai"].template_function == "create_crewai_memory_step"


@pytest.mark.asyncio
async def test_crewai_adapter_projects_task_context_before_execution() -> None:
    """before_task should capture the task input and return a prompt-ready payload."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = CrewAIMemoryAdapter(
        session_id="crew-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    payload = await adapter.before_task(
        "Summarize the beta roadmap",
        crew="release",
        agent="planner",
        task_id="task-7",
    )

    assert payload["memory"] == "Use CrewAI memory."
    assert payload["zaxy"]["session_id"] == "crew-1"
    assert payload["zaxy"]["crew"] == "release"
    assert payload["zaxy"]["agent"] == "planner"
    assert payload["zaxy"]["task_id"] == "task-7"
    assert payload["contexts"][0].content == "CrewAI memory"
    assert calls[0] == (
        "after_turn",
        {
            "role": "user",
            "content": "Summarize the beta roadmap",
            "session_id": "crew-1",
            "query": "Summarize the beta roadmap",
            "source": "crewai",
            "max_recent_events": 20,
            "limit": 10,
        },
    )
    assert calls[-1] == ("close", {})


@pytest.mark.asyncio
async def test_crewai_memory_step_returns_prompt_text() -> None:
    """Factory helper should return a CrewAI-friendly async callable."""
    calls: list[tuple[str, dict[str, Any]]] = []
    step = create_crewai_memory_step(
        session_id="crew-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    prompt = await step("Prepare release notes")

    assert prompt == "Use CrewAI memory."
    assert calls[0][0] == "after_turn"


@pytest.mark.asyncio
async def test_crewai_coordination_step_reports_explicit_worker_finding(tmp_path) -> None:
    """Coordinate step should report explicit task findings into worker-local state."""
    step = create_crewai_coordination_step(
        mission_id="auth-main",
        worker_id="auth-api",
        eventloom_path=str(tmp_path / ".eventloom"),
    )

    finding = await step(
        "API failures trace to expired JWKS cache handling",
        evidence=[{"kind": "source", "reference": "src/auth.py:12"}],
    )

    assert finding["event_type"] == "coordination.finding.reported"
    assert finding["mission_id"] == "auth-main"
    assert finding["worker_id"] == "auth-api"
    assert finding["finding_id"]


@pytest.mark.asyncio
async def test_crewai_adapter_checkout_before_task_uses_memory_checkout() -> None:
    """Opinionated CrewAI middleware should call Memory Checkout before tasks."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = CrewAIMemoryAdapter(
        session_id="crew-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    payload = await adapter.checkout_before_task("What is left?")

    assert payload["memory"] == "# Memory Checkout\nUse CrewAI checkout."
    assert payload["zaxy"]["kind"] == "memory_checkout"
    assert calls[0] == (
        "checkout_memory",
        {
            "query": "What is left?",
            "session_id": "crew-1",
            "limit": 10,
            "max_recent_events": 20,
        },
    )


@pytest.mark.asyncio
async def test_crewai_checkout_payload_exposes_v06_native_contract() -> None:
    """CrewAI checkout metadata should share the v0.6 native adapter contract."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = CrewAIMemoryAdapter(
        session_id="crew-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    payload = await adapter.checkout_before_task(
        "What is left?",
        crew="release",
        agent="planner",
        task_id="task-7",
    )

    assert payload["zaxy"] == {
        "contract": "zaxy.native.v0.6",
        "framework": "crewai",
        "operation": "before_task",
        "source": "crewai",
        "kind": "memory_checkout",
        "status": "ok",
        "session_id": "crew-1",
        "query": "What is left?",
        "current_fact_count": 1,
        "warning_count": 0,
        "diagnostics": {
            "current_fact_count": 1,
            "current_citation_count": 1,
            "feedback_tool": "memory_feedback",
        },
        "quality": {
            "answerability": "answer_from_memory",
            "confidence": 0.91,
            "required_action": None,
        },
        "feedback": {
            "tool": "memory_feedback",
            "payloads": [
                {
                    "entity_name": "memory checkout",
                    "entity_type": "workflow",
                    "feedback": "used",
                }
            ],
        },
        "error": None,
        "crew": "release",
        "agent": "planner",
        "task_id": "task-7",
    }


@pytest.mark.asyncio
async def test_crewai_checkout_failure_returns_stable_error_payload() -> None:
    """CrewAI checkout should fail closed with no task memory and actionable metadata."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = CrewAIMemoryAdapter(
        session_id="crew-1",
        fabric_factory=lambda eventloom_path: FailingCheckoutFabric(calls),
    )

    payload = await adapter.checkout_before_task("What is left?", crew="release")

    assert payload["memory"] == ""
    assert payload["contexts"] == []
    assert payload["zaxy"] == {
        "contract": "zaxy.native.v0.6",
        "framework": "crewai",
        "operation": "before_task",
        "source": "crewai",
        "kind": "memory_checkout",
        "status": "error",
        "session_id": "crew-1",
        "query": "What is left?",
        "current_fact_count": 0,
        "warning_count": 1,
        "diagnostics": {},
        "quality": {
            "answerability": "refresh_recommended",
            "confidence": 0.0,
            "required_action": {
                "tool": "memory_checkout",
                "reason": "Projection unavailable during CrewAI checkout.",
            },
        },
        "feedback": None,
        "error": {
            "code": "checkout_failed",
            "message": "projection unavailable",
            "remediation": "Retry Memory Checkout before the next model call or run zaxy doctor.",
        },
        "crew": "release",
    }
    assert calls[-1] == ("close", {})


@pytest.mark.asyncio
async def test_crewai_adapter_records_task_result_as_assistant_turn() -> None:
    """after_task should persist task output as an assistant observation."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = CrewAIMemoryAdapter(
        session_id="crew-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    payload = await adapter.after_task(
        "Release notes drafted.",
        query="release notes",
        crew="release",
        agent="writer",
        task_id="task-8",
    )

    assert payload["memory"] == "Use CrewAI memory."
    assert payload["zaxy"]["agent"] == "writer"
    assert calls[0][0] == "after_turn"
    assert calls[0][1]["role"] == "assistant"
    assert calls[0][1]["content"] == "Release notes drafted."
    assert calls[0][1]["query"] == "release notes"


@pytest.mark.asyncio
async def test_crewai_adapter_records_tool_observations_without_argument_values() -> None:
    """record_tool_use should append redacted tool-call observations."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = CrewAIMemoryAdapter(
        session_id="crew-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )

    await adapter.record_tool_use(
        tool_name="search",
        status="ok",
        arguments={"query": "zaxy", "api_key": "secret"},
        result_summary="2 hits",
    )

    append = calls[0]
    assert append[0] == "append"
    assert append[1]["event_type"] == "tool.call.completed"
    assert append[1]["actor"] == "zaxy-observer"
    assert append[1]["payload"]["source"] == "crewai"
    assert append[1]["payload"]["argument_keys"] == ["api_key", "query"]
    assert "arguments" not in append[1]["payload"]


@pytest.mark.asyncio
async def test_crewai_adapter_records_feedback_for_projected_contexts() -> None:
    """record_context_feedback should reinforce contexts returned by before_task."""
    calls: list[tuple[str, dict[str, Any]]] = []
    adapter = CrewAIMemoryAdapter(
        session_id="crew-1",
        fabric_factory=lambda eventloom_path: FakeFabric(calls),
    )
    payload = {"contexts": _assembly().contexts}

    count = await adapter.record_context_feedback(payload, feedback="used", importance=0.7)

    assert count == 1
    assert calls[0][0] == "record_context_feedback"
    assert calls[0][1]["feedback"] == "used"
    assert calls[0][1]["importance"] == 0.7


def _assembly() -> ContextAssembly:
    context = Context(content="CrewAI memory", source="verbatim", score=0.9)
    return ContextAssembly(
        session_id="crew-1",
        prompt="Use CrewAI memory.",
        contexts=[context],
        replay_event_count=2,
        compacted=False,
        warnings=[],
    )


def _checkout() -> Any:
    class Checkout:
        def to_dict(self) -> dict[str, Any]:
            return {
                "session_id": "crew-1",
                "query": "What is left?",
                "prompt": "# Memory Checkout\nUse CrewAI checkout.",
                "current_facts": [{"content": "Use CrewAI checkout.", "citation": "eventloom://crew-1/events/1#abc"}],
                "evidence": [],
                "diagnostics": {
                    "current_fact_count": 1,
                    "current_citation_count": 1,
                    "feedback_tool": "memory_feedback",
                },
                "quality": {
                    "answerability": "answer_from_memory",
                    "confidence": 0.91,
                    "required_action": None,
                },
                "guidance": {
                    "feedback": {
                        "tool": "memory_feedback",
                        "payloads": [
                            {
                                "entity_name": "memory checkout",
                                "entity_type": "workflow",
                                "feedback": "used",
                            }
                        ],
                    }
                },
                "warnings": [],
            }

    return Checkout()
