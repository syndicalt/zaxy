"""Claude-compatible model-call memory example.

Run:

    python examples/claude_compatible_memory.py

The script uses a tiny fake Claude-style client, so it does not need network
access or a provider SDK. Real clients only need to expose
``client.messages.create(**request)``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from zaxy.adapters.claude_compatible import create_claude_compatible_memory_adapter
from zaxy.core import MemoryFabric


class FakeMessages:
    """Small Claude-compatible messages surface for local smoke tests."""

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        memory_prompt = kwargs["system"]
        return {
            "id": "msg-zaxy-demo",
            "content": [
                {
                    "type": "text",
                    "text": f"Memory was injected: {bool(memory_prompt)}",
                }
            ],
        }


class FakeClaudeCompatibleClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


async def run_demo(eventloom_path: str | Path) -> dict[str, Any]:
    """Run the model-call adapter and return a stable smoke payload."""
    session_id = "claude-compatible-demo"
    fabric = MemoryFabric(eventloom_path=str(eventloom_path))
    try:
        await fabric.append(
            "task.proposed",
            actor="planner",
            payload={
                "task_id": "claude-model-call-demo",
                "summary": "Claude-compatible calls should receive Memory Checkout outside MCP.",
                "status": "pending",
            },
            session_id=session_id,
        )
    finally:
        await fabric.close()

    adapter = create_claude_compatible_memory_adapter(
        session_id=session_id,
        eventloom_path=str(eventloom_path),
        limit=5,
    )
    result = await adapter.messages_create(
        FakeClaudeCompatibleClient(),
        model="claude-compatible-demo-model",
        messages=[
            {
                "role": "user",
                "content": "What should this model call remember before acting?",
            }
        ],
        max_tokens=256,
    )
    return {
        "session_id": session_id,
        "has_zaxy_context": result["request"]["system"].startswith("# Memory Checkout"),
        "kind": result["zaxy"]["kind"],
        "assistant_content": result["assistant_content"],
    }


async def main() -> None:
    """Run the example in a temporary Eventloom directory and print JSON."""
    with tempfile.TemporaryDirectory(prefix="zaxy-claude-compatible-example-") as tmp:
        payload = await run_demo(Path(tmp) / ".eventloom")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
