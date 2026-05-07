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

Returned context metadata includes Eventloom citations when available and
`score_explanation` details for retrieval debugging. Score explanations include
source weights, the matched query, query-expansion weights when applicable, and
temporal scoring fields for as-of queries.

Ingest local project documents:

```python
count = await fabric.ingest_documents(
    "docs",
    session_id="agent-1",
    max_lines=80,
)
```

Document ingestion currently supports Markdown, plain text, and reStructuredText
files. It skips hidden directories, chunks content by line range, writes
`document.indexed` Eventloom events, and projects each chunk as a `document`
entity. Retrieved document chunks cite their original file and starting line
with `file://path:line` citations.

Ingest a sanitized transcript:

```python
count = await fabric.ingest_transcript(
    [
        {"role": "user", "content": "What did we decide?"},
        {"role": "assistant", "content": "Use MMR for diversity."},
    ],
    source="codex",
    session_id="agent-1",
)
```

Transcript ingestion writes one `transcript.turn` event per non-empty turn.
Secret-looking content is redacted before the event is appended, and redaction
paths are retained on the event payload for auditability.

Assemble prompt-ready context:

```python
assembly = await fabric.assemble_context(
    "What did we decide about retrieval?",
    session_id="agent-1",
    replay_from_seq=1,
    limit=5,
)
print(assembly.prompt)
```

Context assembly combines recent replayed events with graph retrieval. It is the
first lifecycle API for building an LLM context window from both durable session
history and ranked memory.

Run lifecycle hooks after a turn or subagent handoff:

```python
next_context = await fabric.after_turn(
    role="assistant",
    content="Use MMR for diversity.",
    session_id="agent-1",
    query="retrieval decisions",
    max_recent_events=20,
)

handoff = await fabric.handoff_bundle(
    session_id="agent-1",
    query="current goals and open tasks",
)

subagent = await fabric.cleanup_subagent(
    parent_session_id="main",
    subagent_session_id="worker-1",
    summary="Indexed retrieval docs and found no blockers.",
)
```

`after_turn` preserves the turn as a `transcript.turn` event before assembling
bounded context. `handoff_bundle` includes summary data, prompt-ready context,
and Eventloom integrity status. `cleanup_subagent` records `subagent.cleaned`
in the subagent session and returns a bundle the parent can import or inspect.

Customize retrieval policy:

```python
from zaxy.query import HTTPReranker, LexicalReranker, QueryRouter

router = QueryRouter(
    fabric.graph,
    scoring_profile="precision",
    reranker=LexicalReranker(),
)
chunks = await router.query("auth decision rationale", session_id="agent-1")
```

Built-in scoring profiles are `balanced`, `precision`, `recall`, and
`temporal`. Rerankers implement an async `rerank(query, results, limit=...)`
method and receive fused, deduplicated graph candidates before final truncation.
Use `LexicalReranker` for deterministic local reranking, `HTTPReranker` for
local/self-hosted model endpoints, or `OpenAICompatibleReranker` for
OpenAI-compatible chat-completions reranking.

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
At the durable `EventLog.append` boundary, payloads are also classified and
common secret fields or secret-looking values are redacted before the event hash
is sealed.

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
