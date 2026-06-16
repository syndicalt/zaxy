"""Core memory fabric API.

The MemoryFabric is the primary interface for agents to persist and query
context. It coordinates between Eventloom (immutable log), the temporal
selected projection graph, hybrid extraction, and optional tracing.

Public import surface unchanged: import from ``zaxy.core`` as before. The
implementation is split across ``zaxy.core.models`` (dataclasses),
``zaxy.core.checkout_build`` (the checkout builder), and ``zaxy.core.fabric``
(the MemoryFabric class).

Example::

    fabric = MemoryFabric(
        eventloom_path=".eventloom/agent.jsonl",
    )
    await fabric.connect()
    await fabric.append("goal.created", actor="user", payload={"title": "Ship it"})
    context = await fabric.query("What are our goals?")
    await fabric.close()
"""

from zaxy.context import Context
from zaxy.core.checkout_build import (
    _apply_purpose_outcome_learning,
    _purpose_outcome_aggregates,
    build_memory_checkout,
    entity_reinforcement_targets,
)
from zaxy.core.fabric import QUERY_PAGE_CACHE_TTL_SECONDS, MemoryFabric
from zaxy.core.models import (
    ContextAssembly,
    ContextRefreshReport,
    HandoffBundle,
    MemoryCheckout,
    QueryPage,
    checkout_token_efficiency,
)

__all__ = [
    "QUERY_PAGE_CACHE_TTL_SECONDS",
    "Context",
    "ContextAssembly",
    "ContextRefreshReport",
    "HandoffBundle",
    "MemoryCheckout",
    "MemoryFabric",
    "QueryPage",
    "_apply_purpose_outcome_learning",
    "_purpose_outcome_aggregates",
    "build_memory_checkout",
    "checkout_token_efficiency",
    "entity_reinforcement_targets",
]
