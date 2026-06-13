# Architecture

Zaxy has four primary layers: Eventloom, extraction, projection backends, and
MCP. Each layer has a narrow responsibility so the system remains replayable and
auditable. Eventloom stores append-only JSONL events with hash-chain integrity.
The extraction engine converts typed events into `ExtractedEntity` and
`ExtractedEdge` objects. Projection backends store temporal graph views for
retrieval, dashboard inspection, and graph traversal. MCP exposes the memory
operations to agent frameworks and clients.

The data flow starts when an agent calls `memory_append`, a Python caller uses
`MemoryFabric.append`, a service calls `MemoryFabric.ingest_documents` for
local project material, or `MemoryFabric.ingest_transcript` imports sanitized
session turns. Zaxy validates the payload, writes an Eventloom event, runs
extraction, optionally generates embeddings, upserts graph facts, emits metrics,
and traces the operation through Pathlight when enabled. Query calls flow in
the opposite direction: input validation, optional query embedding, exact
search, keyword search, vector similarity, traversal expansion, fusion, and
compact context chunk rendering.

Context lifecycle APIs sit above replay and retrieval. `MemoryFabric.assemble_context`
replays recent session events, queries ranked graph memory, reserves a small
verbatim Eventloom source-recall lane, projects a bounded active memory working
set, and formats the result into a prompt-ready bundle. `after_turn` appends the
completed turn and returns bounded context for the next turn. `handoff_bundle`
packages summary, replay, integrity status, and retrieval into a portable resume
object. `cleanup_subagent` finalizes a subagent session with a cleanup event and
emits a handoff bundle. These APIs use Eventloom as the recovery anchor, the
graph as the relevance layer, verbatim retrieval as the exact-source lane, and
the working set as the model-facing memory projection instead of creating a
separate context cache.

Eventloom is deliberately the bottom layer. It must remain useful even if a
projection backend is unavailable or a projection bug is discovered. If a graph
needs to be rebuilt, replay the log and re-run extraction. This is the reason
Zaxy does not silently overwrite facts: graph entities carry `valid_from` and
`valid_to` windows, and reasserted facts become new versions.

The embedded LadybugDB projection is the default graph backend because it preserves
graph-native traversal, citations, temporal versioning, and replay without a
separate graph service. LadybugDB is the actively maintained fork of the
archived Kuzu engine; Zaxy 2.3 moved the embedded store to it (exact-pinned)
after verifying the fork against Zaxy's own defect reproductions and scale
lanes. Neo4j remains the quality and performance control
backend, and pgGraph remains an experimental Postgres-native projection backend.
See [zero-friction-runtime-roadmap.md](archive/zero-friction-runtime-roadmap.md) for the
embedded runtime gates and backend lifecycle.

Zaxy does not delegate schema control to a high-level graph-memory abstraction
because temporal validity, invalidation semantics, index management, and query
fusion need to be explicit and testable. The graph schema is documented in
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
