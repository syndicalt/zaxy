"""Dependency-light LangGraph adapter example.

Run:

    python examples/langgraph_memory.py

The script uses Zaxy's LangGraph-shaped async node without importing LangGraph.
Applications can plug the same node into a real LangGraph workflow when the
optional `zaxy-memory[langgraph]` extra is installed.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from zaxy.adapters.langgraph import create_langgraph_memory_checkout_node
from zaxy.core import MemoryFabric


async def run_demo(eventloom_path: str | Path) -> dict[str, Any]:
    """Run the dependency-light checkout node and return a stable smoke payload."""
    session_id = "langgraph-demo"
    fabric = MemoryFabric(eventloom_path=eventloom_path)
    try:
        await fabric.append(
            "task.proposed",
            actor="planner",
            payload={
                "task_id": "demo-task",
                "summary": "LangGraph should call Memory Checkout before model work.",
                "status": "pending",
            },
            session_id=session_id,
        )
    finally:
        await fabric.close()

    node = create_langgraph_memory_checkout_node(
        session_id=session_id,
        eventloom_path=str(eventloom_path),
        limit=5,
    )
    state = await node(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What should this LangGraph worker remember before acting?",
                }
            ]
        }
    )
    return {
        "session_id": session_id,
        "has_zaxy_context": bool(state.get("zaxy_context")),
        "kind": state.get("zaxy", {}).get("kind", "context_assembly"),
    }


async def main() -> None:
    """Run the example in a temporary Eventloom directory and print JSON."""
    with tempfile.TemporaryDirectory(prefix="zaxy-langgraph-example-") as tmp:
        payload = await run_demo(Path(tmp) / ".eventloom")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
