"""OpenAI-compatible model-call memory example.

Run:

    python examples/openai_compatible_memory.py

The script uses a tiny fake OpenAI-style client, so it does not need network
access or the OpenAI Python package. Real clients only need to expose
``client.chat.completions.create(**request)``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from zaxy.adapters.openai_compatible import create_openai_compatible_memory_adapter
from zaxy.core import MemoryFabric


class FakeChatCompletions:
    """Small OpenAI-compatible chat.completions surface for local smoke tests."""

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        memory_prompt = kwargs["messages"][0]["content"]
        return {
            "id": "chatcmpl-zaxy-demo",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Memory was injected: {bool(memory_prompt)}",
                    }
                }
            ],
        }


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeChatCompletions()


class FakeOpenAICompatibleClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


async def run_demo(eventloom_path: str | Path) -> dict[str, Any]:
    """Run the model-call adapter and return a stable smoke payload."""
    session_id = "openai-compatible-demo"
    fabric = MemoryFabric(eventloom_path=str(eventloom_path))
    try:
        await fabric.append(
            "task.proposed",
            actor="planner",
            payload={
                "task_id": "model-call-demo",
                "summary": "OpenAI-compatible calls should receive Memory Checkout outside MCP.",
                "status": "pending",
            },
            session_id=session_id,
        )
    finally:
        await fabric.close()

    adapter = create_openai_compatible_memory_adapter(
        session_id=session_id,
        eventloom_path=str(eventloom_path),
        limit=5,
    )
    result = await adapter.chat_completion(
        FakeOpenAICompatibleClient(),
        model="openai-compatible-demo-model",
        messages=[
            {
                "role": "user",
                "content": "What should this model call remember before acting?",
            }
        ],
    )
    return {
        "session_id": session_id,
        "has_zaxy_context": result["messages"][0]["content"].startswith("# Memory Checkout"),
        "kind": result["zaxy"]["kind"],
        "assistant_content": result["assistant_content"],
    }


async def main() -> None:
    """Run the example in a temporary Eventloom directory and print JSON."""
    with tempfile.TemporaryDirectory(prefix="zaxy-openai-compatible-example-") as tmp:
        payload = await run_demo(Path(tmp) / ".eventloom")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
