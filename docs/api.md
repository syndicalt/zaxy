# Python API

The primary Python entry point is `zaxy.core.MemoryFabric`. It wires the event
log, session manager, graph store, query router, embedding provider, metrics,
and tracer. The MCP server uses the same underlying orchestration, so Python
and MCP callers share behavior.

Create a fabric:

```python
from zaxy import MemoryFabric

fabric = MemoryFabric()
await fabric.connect()
```

Append an event:

```python
event = await fabric.append(
    event_type="goal.created",
    actor="planner",
    payload={"title": "Ship memory integration"},
    session_id="agent-1",
)
```

Query memory:

```python
context = await fabric.query(
    "Ship memory integration",
    temporal_filter=None,
    limit=5,
    session_id="agent-1",
)
```

Replay a session:

```python
replay = fabric.replay(session_id="agent-1", from_seq=0)
```

Invalidate a graph fact:

```python
await fabric.invalidate(
    entity_name="Ship memory integration",
    entity_type="Goal",
    invalid_at="2026-05-06T12:00:00Z",
)
```

Always close long-lived clients:

```python
await fabric.close()
```

`MemoryFabric` validates inputs before writing events or querying the graph.
Session IDs are passed through the same safety rules used by MCP. Payload size,
query length, traversal depth, and limits are bounded in `src/zaxy/security.py`.

Lower-level modules are public enough for advanced integration but should be
used carefully: `EventLog` for direct JSONL operations, `extract` for rule-based
projection, `GraphStore` for Neo4j operations, `QueryRouter` for retrieval, and
`MemoryTracer` for Pathlight spans. Prefer `MemoryFabric` unless you are
building tests, migrations, or specialized tooling.

Errors should be treated as operational signals. Validation errors normally mean
the caller sent an unsafe session ID, oversized payload, invalid limit, or empty
query. Graph errors usually mean Neo4j is unavailable, indexes are missing, or
credentials are wrong. Event log errors usually mean filesystem permissions,
lock contention, or integrity verification failed. In all cases, the Eventloom
log should remain the recovery anchor: fix the environment, replay the log, and
rebuild projections rather than inventing graph state by hand.

For long-running processes, create one fabric per service process and reuse it.
Avoid constructing a new fabric for every query because each instance owns graph
and tracer clients. Tests can instantiate directly with mocks, but production
callers should use the lifecycle methods consistently.

Configuration comes from [configuration.md](configuration.md). MCP behavior is
in [mcp.md](mcp.md). Graph details are in [graph-schema.md](graph-schema.md).
The quick install path is in [README.md](../README.md), and the public product
overview is [site/index.html](../site/index.html).
