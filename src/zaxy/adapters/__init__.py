"""Native-preview framework adapters for Zaxy memory."""

from zaxy.adapters.claude_compatible import (
    ClaudeCompatibleMemoryAdapter,
    create_claude_compatible_memory_adapter,
)
from zaxy.adapters.coordination import CoordinationAdapter
from zaxy.adapters.crewai import (
    CrewAIMemoryAdapter,
    create_crewai_coordination_step,
    create_crewai_memory_checkout_step,
    create_crewai_memory_step,
)
from zaxy.adapters.langgraph import (
    LangGraphMemoryAdapter,
    create_langgraph_coordination_node,
    create_langgraph_memory_checkout_node,
    create_langgraph_memory_node,
)
from zaxy.adapters.openai_compatible import (
    OpenAICompatibleMemoryAdapter,
    create_openai_compatible_memory_adapter,
)

__all__ = [
    "CoordinationAdapter",
    "CrewAIMemoryAdapter",
    "ClaudeCompatibleMemoryAdapter",
    "LangGraphMemoryAdapter",
    "OpenAICompatibleMemoryAdapter",
    "create_claude_compatible_memory_adapter",
    "create_crewai_coordination_step",
    "create_crewai_memory_step",
    "create_crewai_memory_checkout_step",
    "create_langgraph_coordination_node",
    "create_langgraph_memory_node",
    "create_langgraph_memory_checkout_node",
    "create_openai_compatible_memory_adapter",
]
