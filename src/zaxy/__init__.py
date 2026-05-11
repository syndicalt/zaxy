"""Zaxy: Event-sourced temporal knowledge graph fabric for AI agent memory."""

from zaxy.adapters.langgraph import LangGraphMemoryAdapter, create_langgraph_memory_node
from zaxy.context import Context, ContextAssemblyPolicy
from zaxy.core import MemoryCheckout, MemoryFabric
from zaxy.integrations import (
    FrameworkIntegrationSpec,
    list_framework_integration_specs,
    render_agent_integration_template,
    render_framework_install_command,
    render_handoff_adapter,
    render_mcp_client_config,
)
from zaxy.refs import MemoryRef, MemoryRefStore
from zaxy.release import package_version
from zaxy.verbatim import VerbatimHit, VerbatimIndex

__version__ = package_version()

__all__ = [
    "FrameworkIntegrationSpec",
    "LangGraphMemoryAdapter",
    "Context",
    "ContextAssemblyPolicy",
    "MemoryCheckout",
    "MemoryFabric",
    "MemoryRef",
    "MemoryRefStore",
    "VerbatimHit",
    "VerbatimIndex",
    "__version__",
    "create_langgraph_memory_node",
    "list_framework_integration_specs",
    "render_agent_integration_template",
    "render_framework_install_command",
    "render_handoff_adapter",
    "render_mcp_client_config",
]
