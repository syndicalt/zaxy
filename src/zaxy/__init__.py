"""Zaxy: Event-sourced temporal knowledge graph fabric for AI agent memory."""

from zaxy.core import MemoryFabric
from zaxy.integrations import (
    render_agent_integration_template,
    render_handoff_adapter,
    render_mcp_client_config,
)

__all__ = [
    "MemoryFabric",
    "render_agent_integration_template",
    "render_handoff_adapter",
    "render_mcp_client_config",
]
