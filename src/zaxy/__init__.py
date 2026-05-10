"""Zaxy: Event-sourced temporal knowledge graph fabric for AI agent memory."""

from zaxy.core import MemoryFabric
from zaxy.integrations import (
    FrameworkIntegrationSpec,
    list_framework_integration_specs,
    render_agent_integration_template,
    render_framework_install_command,
    render_handoff_adapter,
    render_mcp_client_config,
)

__all__ = [
    "FrameworkIntegrationSpec",
    "MemoryFabric",
    "list_framework_integration_specs",
    "render_agent_integration_template",
    "render_framework_install_command",
    "render_handoff_adapter",
    "render_mcp_client_config",
]
