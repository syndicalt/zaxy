# Architecture

Zaxy has four primary layers: Eventloom, extraction, Neo4j, and MCP. Each layer
has a narrow responsibility so the system remains replayable and auditable.
Eventloom stores append-only JSONL events with hash-chain integrity. The
extraction engine converts typed events into `ExtractedEntity` and
`ExtractedEdge` objects. Neo4j stores temporal graph projections. MCP exposes
the memory operations to agent frameworks and clients.

The data flow starts when an agent calls `memory_append` or a Python caller uses
`MemoryFabric.append`. Zaxy validates the payload, writes an Eventloom event,
runs extraction, optionally generates embeddings, upserts graph facts, emits
metrics, and traces the operation through Pathlight when enabled. Query calls
flow in the opposite direction: input validation, optional query embedding,
exact search, keyword search, vector similarity, traversal expansion, fusion,
and compact context chunk rendering.

Eventloom is deliberately the bottom layer. It must remain useful even if Neo4j
is unavailable or a projection bug is discovered. If the graph needs to be
rebuilt, replay the log and re-run extraction. This is the reason Zaxy does not
silently overwrite facts: graph entities carry `valid_from` and `valid_to`
windows, and reasserted facts become new versions.

Neo4j is used directly through the official driver. Zaxy does not delegate
schema control to a high-level graph-memory abstraction because temporal
validity, invalidation semantics, index management, and query fusion need to be
explicit and testable. The graph schema is documented in
[graph-schema.md](graph-schema.md).

Pathlight is observability, not storage. It records memory operation traces so
operators can see append/query/replay/invalidate spans and correlate them with
agent runs. Zaxy can run without Pathlight; missing tracing must not prevent
memory operations.

The remote interface is MCP over stdio or SSE. Stdio is best for local desktop
and framework integrations. SSE is best for daemon mode, but it requires bearer
auth and per-client session scoping in production. See [mcp.md](mcp.md) and
[security.md](security.md) for those contracts.

The public site at [site/index.html](../site/index.html) describes the system
for new users. The [README.md](../README.md) gives a quick start. The runbook at
[runbook.md](runbook.md) covers operations and incident response.
