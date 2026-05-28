"""Zaxy: Event-sourced temporal knowledge graph fabric for AI agent memory."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CoordinationAdapter": ("zaxy.adapters.coordination", "CoordinationAdapter"),
    "CrewAIMemoryAdapter": ("zaxy.adapters.crewai", "CrewAIMemoryAdapter"),
    "LangGraphMemoryAdapter": ("zaxy.adapters.langgraph", "LangGraphMemoryAdapter"),
    "Context": ("zaxy.context", "Context"),
    "ContextAssemblyPolicy": ("zaxy.context", "ContextAssemblyPolicy"),
    "MemoryCheckout": ("zaxy.core", "MemoryCheckout"),
    "MemoryFabric": ("zaxy.core", "MemoryFabric"),
    "QueryPage": ("zaxy.core", "QueryPage"),
    "CoordinationApprovalDecisionResult": ("zaxy.coordination", "CoordinationApprovalDecisionResult"),
    "CoordinationApprovalPacket": ("zaxy.coordination", "CoordinationApprovalPacket"),
    "CoordinationBrief": ("zaxy.coordination", "CoordinationBrief"),
    "CoordinationCheckout": ("zaxy.coordination", "CoordinationCheckout"),
    "CoordinationManager": ("zaxy.coordination", "CoordinationManager"),
    "CoordinationPerformanceLedger": ("zaxy.coordination", "CoordinationPerformanceLedger"),
    "CoordinationReviewExport": ("zaxy.coordination", "CoordinationReviewExport"),
    "HTTPSemanticConflictDetector": ("zaxy.coordination_semantic", "HTTPSemanticConflictDetector"),
    "LocalSemanticConflictDetector": ("zaxy.coordination", "LocalSemanticConflictDetector"),
    "build_semantic_conflict_detector": (
        "zaxy.coordination_semantic",
        "build_semantic_conflict_detector",
    ),
    "FrameworkIntegrationDecision": (
        "zaxy.integrations",
        "FrameworkIntegrationDecision",
    ),
    "FrameworkIntegrationSpec": ("zaxy.integrations", "FrameworkIntegrationSpec"),
    "ProjectionStore": ("zaxy.projection", "ProjectionStore"),
    "MemoryRef": ("zaxy.refs", "MemoryRef"),
    "MemoryRefStore": ("zaxy.refs", "MemoryRefStore"),
    "VerbatimHit": ("zaxy.verbatim", "VerbatimHit"),
    "VerbatimIndex": ("zaxy.verbatim", "VerbatimIndex"),
    "__version__": ("zaxy.release", "package_version"),
    "create_crewai_memory_step": ("zaxy.adapters.crewai", "create_crewai_memory_step"),
    "create_crewai_coordination_step": (
        "zaxy.adapters.crewai",
        "create_crewai_coordination_step",
    ),
    "create_crewai_memory_checkout_step": (
        "zaxy.adapters.crewai",
        "create_crewai_memory_checkout_step",
    ),
    "create_langgraph_memory_node": (
        "zaxy.adapters.langgraph",
        "create_langgraph_memory_node",
    ),
    "create_langgraph_coordination_node": (
        "zaxy.adapters.langgraph",
        "create_langgraph_coordination_node",
    ),
    "create_langgraph_memory_checkout_node": (
        "zaxy.adapters.langgraph",
        "create_langgraph_memory_checkout_node",
    ),
    "list_framework_integration_specs": (
        "zaxy.integrations",
        "list_framework_integration_specs",
    ),
    "recommend_framework_integration_target": (
        "zaxy.integrations",
        "recommend_framework_integration_target",
    ),
    "render_agent_integration_template": (
        "zaxy.integrations",
        "render_agent_integration_template",
    ),
    "render_coordination_adapter_template": (
        "zaxy.integrations",
        "render_coordination_adapter_template",
    ),
    "render_framework_install_command": (
        "zaxy.integrations",
        "render_framework_install_command",
    ),
    "render_handoff_adapter": ("zaxy.integrations", "render_handoff_adapter"),
    "render_mcp_client_config": ("zaxy.integrations", "render_mcp_client_config"),
}

__all__ = [
    "FrameworkIntegrationSpec",
    "FrameworkIntegrationDecision",
    "CoordinationAdapter",
    "CrewAIMemoryAdapter",
    "LangGraphMemoryAdapter",
    "Context",
    "ContextAssemblyPolicy",
    "MemoryCheckout",
    "MemoryFabric",
    "ProjectionStore",
    "QueryPage",
    "CoordinationApprovalDecisionResult",
    "CoordinationApprovalPacket",
    "CoordinationBrief",
    "CoordinationCheckout",
    "CoordinationManager",
    "CoordinationPerformanceLedger",
    "CoordinationReviewExport",
    "HTTPSemanticConflictDetector",
    "LocalSemanticConflictDetector",
    "MemoryRef",
    "MemoryRefStore",
    "VerbatimHit",
    "VerbatimIndex",
    "__version__",
    "create_crewai_coordination_step",
    "create_crewai_memory_step",
    "create_crewai_memory_checkout_step",
    "create_langgraph_coordination_node",
    "create_langgraph_memory_node",
    "create_langgraph_memory_checkout_node",
    "build_semantic_conflict_detector",
    "list_framework_integration_specs",
    "recommend_framework_integration_target",
    "render_agent_integration_template",
    "render_coordination_adapter_template",
    "render_framework_install_command",
    "render_handoff_adapter",
    "render_mcp_client_config",
]


def __getattr__(name: str) -> Any:
    """Load package-level public exports only when callers request them."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    if name == "__version__":
        value = value()
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return discoverable lazy public exports."""
    return sorted({*globals(), *__all__})
