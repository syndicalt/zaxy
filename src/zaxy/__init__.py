"""Zaxy: Event-sourced temporal knowledge graph fabric for AI agent memory."""

from zaxy.adapters.crewai import CrewAIMemoryAdapter, create_crewai_memory_step
from zaxy.adapters.langgraph import LangGraphMemoryAdapter, create_langgraph_memory_node
from zaxy.context import Context, ContextAssemblyPolicy
from zaxy.core import MemoryCheckout, MemoryFabric, QueryPage
from zaxy.integrations import (
    FrameworkIntegrationDecision,
    FrameworkIntegrationSpec,
    list_framework_integration_specs,
    recommend_framework_integration_target,
    render_agent_integration_template,
    render_framework_install_command,
    render_handoff_adapter,
    render_mcp_client_config,
)
from zaxy.projection import ProjectionStore
from zaxy.refs import MemoryRef, MemoryRefStore
from zaxy.release import package_version
from zaxy.verbatim import VerbatimHit, VerbatimIndex

__version__ = package_version()

__all__ = [
    "FrameworkIntegrationSpec",
    "FrameworkIntegrationDecision",
    "CrewAIMemoryAdapter",
    "LangGraphMemoryAdapter",
    "Context",
    "ContextAssemblyPolicy",
    "MemoryCheckout",
    "MemoryFabric",
    "ProjectionStore",
    "QueryPage",
    "MemoryRef",
    "MemoryRefStore",
    "VerbatimHit",
    "VerbatimIndex",
    "__version__",
    "create_crewai_memory_step",
    "create_langgraph_memory_node",
    "list_framework_integration_specs",
    "recommend_framework_integration_target",
    "render_agent_integration_template",
    "render_framework_install_command",
    "render_handoff_adapter",
    "render_mcp_client_config",
]
