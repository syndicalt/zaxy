"""Shared helpers for dependency-light framework adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from zaxy.core import ContextAssembly, MemoryFabric

FabricFactory = Callable[[str], Any]


def assembly_payload(assembly: ContextAssembly) -> dict[str, Any]:
    """Return JSON-friendly metadata for a context assembly."""
    return {
        "session_id": assembly.session_id,
        "prompt": assembly.prompt,
        "contexts": [asdict(context) for context in assembly.contexts],
        "replay_event_count": assembly.replay_event_count,
        "compacted": assembly.compacted,
        "warnings": list(assembly.warnings),
    }


def native_checkout_payload(
    checkout: dict[str, Any],
    *,
    framework: str,
    operation: str,
    source: str,
    session_id: str,
) -> dict[str, Any]:
    """Return stable v0.6 metadata for native Memory Checkout adapters."""
    return {
        "contract": "zaxy.native.v0.6",
        "framework": framework,
        "operation": operation,
        "source": source,
        "kind": "memory_checkout",
        "status": "ok",
        "session_id": checkout.get("session_id", session_id),
        "query": checkout.get("query"),
        "current_fact_count": len(checkout.get("current_facts", []) or []),
        "warning_count": len(checkout.get("warnings", []) or []),
        "diagnostics": dict(checkout.get("diagnostics") or {}),
        "quality": dict(checkout.get("quality") or {}),
        "feedback": _checkout_feedback(checkout),
        "error": None,
    }


def native_checkout_error_payload(
    exc: Exception,
    *,
    framework: str,
    operation: str,
    source: str,
    session_id: str,
    query: str,
) -> dict[str, Any]:
    """Return stable v0.6 failure metadata without injecting stale context."""
    return {
        "contract": "zaxy.native.v0.6",
        "framework": framework,
        "operation": operation,
        "source": source,
        "kind": "memory_checkout",
        "status": "error",
        "session_id": session_id,
        "query": query,
        "current_fact_count": 0,
        "warning_count": 1,
        "diagnostics": {},
        "quality": {
            "answerability": "refresh_recommended",
            "confidence": 0.0,
            "required_action": {
                "tool": "memory_checkout",
                "reason": f"Projection unavailable during {framework_label(framework)} checkout.",
            },
        },
        "feedback": None,
        "error": {
            "code": "checkout_failed",
            "message": str(exc),
            "remediation": "Retry Memory Checkout before the next model call or run zaxy doctor.",
        },
    }


def framework_label(framework: str) -> str:
    labels = {"langgraph": "LangGraph", "crewai": "CrewAI"}
    return labels.get(framework, framework)


def _checkout_feedback(checkout: dict[str, Any]) -> dict[str, Any] | None:
    guidance = checkout.get("guidance")
    if not isinstance(guidance, dict):
        return None
    feedback = guidance.get("feedback")
    if not isinstance(feedback, dict):
        return None
    return dict(feedback)


def default_fabric_factory(eventloom_path: str) -> MemoryFabric:
    """Construct a MemoryFabric for adapter defaults."""
    return MemoryFabric(eventloom_path)
