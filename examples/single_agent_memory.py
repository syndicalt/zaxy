"""Single-agent durable memory example with no sidecar services.

Run from the repository root:

    python examples/single_agent_memory.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from zaxy.capabilities import build_memory_bootstrap  # noqa: E402
from zaxy.core import MemoryFabric  # noqa: E402


SESSION_ID = "single-agent-demo"
QUESTION = "What should the agent do before major project work?"


async def run_example() -> dict[str, Any]:
    """Run a complete single-agent memory loop and return JSON evidence."""
    with TemporaryDirectory(prefix="zaxy-single-agent-") as tmp:
        eventloom_path = Path(tmp) / ".eventloom"
        fabric = MemoryFabric(
            eventloom_path=str(eventloom_path),
            embedded_graph_path=eventloom_path / "projections" / "embedded.kuzu",
            tracer_disabled=True,
        )
        try:
            await fabric.append(
                "task.proposed",
                actor="user",
                session_id=SESSION_ID,
                payload={
                    "taskId": "prepare-major-work",
                    "title": "Before major project work, the agent should refresh memory.",
                    "summary": "Use Memory Checkout before major project work.",
                },
            )
            await fabric.append(
                "transcript.turn",
                actor="assistant",
                session_id=SESSION_ID,
                payload={
                    "role": "assistant",
                    "content": "Before major project work, run memory_checkout and trust cited current facts.",
                },
            )

            bootstrap = build_memory_bootstrap(
                eventloom_path=eventloom_path,
                session_id=SESSION_ID,
                workspace_root=ROOT,
                current_task=QUESTION,
            )
            checkout = await fabric.checkout_memory(
                QUESTION,
                session_id=SESSION_ID,
                limit=5,
            )
            replay = await fabric.replay(session_id=SESSION_ID)
            return {
                "session_id": SESSION_ID,
                "bootstrap": bootstrap,
                "checkout": checkout.to_dict(),
                "event_count": len(replay.events),
            }
        finally:
            await fabric.close()


def main() -> None:
    """Print only JSON so validation can parse stdout directly."""
    print(json.dumps(asyncio.run(run_example()), sort_keys=True))


if __name__ == "__main__":
    main()
