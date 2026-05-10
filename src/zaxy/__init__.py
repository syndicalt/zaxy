"""Zaxy: Event-sourced temporal knowledge graph fabric for AI agent memory."""

from zaxy.adapters.langgraph import LangGraphMemoryAdapter, create_langgraph_memory_node
from zaxy.context import Context, ContextAssemblyPolicy
from zaxy.core import MemoryFabric
from zaxy.integrations import (
    FrameworkIntegrationSpec,
    list_framework_integration_specs,
    render_agent_integration_template,
    render_framework_install_command,
    render_handoff_adapter,
    render_mcp_client_config,
)
from zaxy.verbatim import VerbatimHit, VerbatimIndex

__all__ = [
    "FrameworkIntegrationSpec",
    "LangGraphMemoryAdapter",
    "Context",
    "ContextAssemblyPolicy",
    "MemoryFabric",
    "VerbatimHit",
    "VerbatimIndex",
    "create_langgraph_memory_node",
    "list_framework_integration_specs",
    "render_agent_integration_template",
    "render_framework_install_command",
    "render_handoff_adapter",
    "render_mcp_client_config",
]
