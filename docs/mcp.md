# MCP Interface

MCP is Zaxy's primary integration surface. The server exposes memory operations
as typed tools so LangGraph, Claude Desktop, custom agents, or any MCP-capable
client can use the same memory system without linking directly against Python
application code.

`memory_append(event_type, actor, payload, thread?)` appends a typed event to
the Eventloom log for the selected session, extracts graph entities and edges,
upserts the Neo4j projection, records metrics, and emits a Pathlight span when
tracing is enabled. Payload size is bounded and session IDs are validated before
they affect filesystem paths.

`memory_query(query, temporal_filter?, limit?)` returns ranked context chunks.
The query router validates the string and limit, optionally embeds the query,
runs exact/keyword/vector/traversal search, fuses scores, and returns compact
context suitable for an agent prompt. Temporal filters let a client ask what was
valid at a specific time. Remote SSE requests are constrained to the session
from the configured session header. Results include Eventloom citations when
available so clients can display or replay the source event. Results also
include `score_explanation` metadata for ranking diagnostics.

`memory_replay(session_id, from_seq?)` rebuilds session history from the
Eventloom log. This is useful for handoffs, audits, and debugging. In remote SSE
mode, the authenticated session scope is enforced so a client cannot replay a
different session.

`memory_invalidate(entity_name, entity_type, invalid_at)` closes the validity
window for a graph fact without deleting history. This lets agents correct
memory while preserving provenance.

Run stdio locally:

```bash
zaxy serve
```

Run SSE daemon mode:

```bash
zaxy serve --transport sse --host 127.0.0.1 --port 8080
```

Production SSE requires `MCP_REMOTE_AUTH_TOKEN` or
`MCP_REMOTE_AUTH_TOKEN_FILE`. Clients send `Authorization: Bearer <token>` and a
session header such as `x-zaxy-session-id: agent-1`. The header name is
configurable through `MCP_REMOTE_SESSION_HEADER`. Production also requires
`MCP_ADMIN_TOKEN` or `MCP_ADMIN_TOKEN_FILE` for replay and invalidation.

The MCP implementation lives in `src/zaxy/mcp_server.py`. Core orchestration
lives in `src/zaxy/core.py`. Security helpers live in `src/zaxy/security.py`.
See [api.md](api.md) for Python-level calls, [configuration.md](configuration.md)
for environment variables, and [security.md](security.md) for remote transport
hardening. The public overview is [site/index.html](../site/index.html), while
[README.md](../README.md) keeps the short command list.
