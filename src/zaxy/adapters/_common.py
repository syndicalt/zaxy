"""Shared helpers for dependency-light framework adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from zaxy.core import ContextAssembly, MemoryFabric

FabricFactory = Callable[[str], Any]


def assembly_payload(assembly: ContextAssembly) -> dict[str, Any]:
    """Return JSON-friendly metadata for a context assembly."""
    return {
        "session_id": assembly.session_id,
        "prompt": assembly.prompt,
        "contexts": [asdict(context) for context in assembly.contexts],
        "replay_event_count": assembly.replay_event_count,
        "compacted": assembly.compacted,
        "warnings": list(assembly.warnings),
    }


def default_fabric_factory(eventloom_path: str) -> MemoryFabric:
    """Construct a MemoryFabric for adapter defaults."""
    return MemoryFabric(eventloom_path)
