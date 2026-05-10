"""LangGraph integration example using Zaxy's native-preview adapter.

Prerequisites::

    pip install 'zaxy-memory[langgraph]'
    docker compose up -d neo4j

Run::

    python examples/langgraph_memory.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from zaxy.adapters.langgraph import LangGraphMemoryAdapter, create_langgraph_memory_node


class AgentState(TypedDict, total=False):
    """LangGraph state with messages and Zaxy-projected memory."""

    messages: list[dict[str, Any]]
    zaxy_context: str
    zaxy_contexts: list[Any]
    zaxy: dict[str, Any]


adapter = LangGraphMemoryAdapter(session_id="langgraph-demo", eventloom_path=".eventloom")


async def llm_node(state: AgentState) -> AgentState:
    """Placeholder LLM node that consumes Zaxy's projected context."""
    user_msg = state["messages"][-1]["content"] if state.get("messages") else ""
    context = state.get("zaxy_context", "")
    response = f"Context:\n{context}\n\nUser said: {user_msg}\n\n[LLM response would go here]"
    messages = [*state.get("messages", []), {"role": "assistant", "content": response}]
    await adapter.record_assistant_turn(response)
    await adapter.record_context_feedback(state, feedback="used", importance=0.7)
    return {**state, "messages": messages}


async def tool_node(state: AgentState) -> AgentState:
    """Example tool node that records redacted execution metadata."""
    await adapter.record_tool_call(
        tool_name="demo_search",
        status="ok",
        arguments={"query": state["messages"][-1]["content"]},
        result_summary="demo result",
    )
    return state


def build_graph() -> Any:
    """Build a LangGraph with Zaxy memory projection before model work."""
    workflow = StateGraph(AgentState)
    workflow.add_node("zaxy_memory", create_langgraph_memory_node(session_id="langgraph-demo"))
    workflow.add_node("tool", tool_node)
    workflow.add_node("llm", llm_node)
    workflow.set_entry_point("zaxy_memory")
    workflow.add_edge("zaxy_memory", "tool")
    workflow.add_edge("tool", "llm")
    workflow.add_edge("llm", END)
    return workflow.compile()


async def main() -> None:
    graph = build_graph()
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "What do we know about Zaxy memory?"}]}
    )
    print(result["messages"][-1]["content"])


if __name__ == "__main__":
    asyncio.run(main())
