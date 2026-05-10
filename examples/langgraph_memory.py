"""LangGraph integration example.

Demonstrates how to wire Zaxy into a LangGraph agent as a memory node.
The memory node appends events after each agent step and queries context
before LLM calls.

Prerequisites::

    pip install langgraph langchain-openai
    docker compose up -d neo4j

Run::

    python examples/langgraph_memory.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from zaxy.event import EventLog
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.query import QueryRouter


class AgentState(TypedDict):
    """LangGraph state with messages and retrieved context."""

    messages: list[dict[str, Any]]
    context: list[str]


# ------------------------------------------------------------------
# Zaxy memory node
# ------------------------------------------------------------------

class ZaxyMemoryNode:
    """LangGraph node that interfaces with Zaxy memory fabric."""

    def __init__(
        self,
        eventloom_path: str = ".eventloom/langgraph.jsonl",
        neo4j_uri: str = "bolt://localhost:7687",
    ) -> None:
        self.log = EventLog(eventloom_path)
        self.graph = GraphStore(neo4j_uri, "neo4j", "testpassword")
        self._connected = False

    async def connect(self) -> None:
        if not self._connected:
            await self.graph.connect()
            await self.graph.init_schema()
            self._connected = True

    async def query(self, state: AgentState) -> AgentState:
        """Retrieve context before the LLM call."""
        await self.connect()
        router = QueryRouter(self.graph)

        # Use the last user message as the query
        last_message = state["messages"][-1]["content"] if state["messages"] else ""
        chunks = await router.query(last_message, limit=5)
        state["context"] = [c.content for c in chunks]
        return state

    async def append(self, state: AgentState) -> AgentState:
        """Append the agent's response as an event."""
        last_message = state["messages"][-1] if state["messages"] else {}
        if last_message.get("role") == "assistant":
            event = self.log.append(
                "agent.responded",
                actor="assistant",
                payload={"content": last_message.get("content", "")},
            )
            extraction = extract(event)
            await self.graph.upsert_extraction(extraction)
        return state


# ------------------------------------------------------------------
# Dummy LLM node (replace with real LLM)
# ------------------------------------------------------------------

async def dummy_llm(state: AgentState) -> AgentState:
    """Placeholder LLM that echoes context + user message."""
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    context = "\n".join(state.get("context", []))
    response = f"Context:\n{context}\n\nUser said: {user_msg}\n\n[LLM response would go here]"
    state["messages"].append({"role": "assistant", "content": response})
    return state


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build a LangGraph with Zaxy memory nodes."""
    memory = ZaxyMemoryNode()
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("memory_query", lambda s: asyncio.run(memory.query(s)))
    workflow.add_node("llm", lambda s: asyncio.run(dummy_llm(s)))
    workflow.add_node("memory_append", lambda s: asyncio.run(memory.append(s)))

    # Edges
    workflow.set_entry_point("memory_query")
    workflow.add_edge("memory_query", "llm")
    workflow.add_edge("llm", "memory_append")
    workflow.add_edge("memory_append", END)

    return workflow.compile()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main() -> None:
    graph = build_graph()
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "What do we know about Alice?"}]}
    )
    print(result["messages"][-1]["content"])


if __name__ == "__main__":
    asyncio.run(main())
