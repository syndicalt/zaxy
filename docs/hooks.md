# Hook Protocol

Zaxy hooks are observer adapters. Agents and tools execute normally; hooks only
record lifecycle checkpoints into Eventloom. This keeps Zaxy out of the
execution path while preserving durable provenance for session boundaries,
checkpoints, and compaction.

Generate client-specific adapter config:

```bash
zaxy hooks claude-code --eventloom-path .eventloom --domain my-project
zaxy hooks codex --eventloom-path .eventloom --domain my-project
```

The generated commands call the stable sink:

```bash
zaxy hook-event precompact \
  --eventloom-path .eventloom \
  --session-id my-project-default \
  --source claude-code
```

## Triggers

Supported triggers are:

| Trigger | Event Type | Purpose |
|---------|------------|---------|
| `session-start` | `hook.session_started` | Mark the start of a client session. |
| `stop` | `hook.stop` | Record a normal response/session checkpoint. |
| `precompact` | `hook.precompact` | Record that context compaction is about to happen. |
| `checkpoint` | `hook.checkpoint` | Record a manual or periodic save/checkpoint. |

Unknown triggers are rejected before writing.

## Payload

Hook events use actor `zaxy-hook` and append to the selected session log. The
payload is intentionally small:

```json
{
  "trigger": "precompact",
  "source": "claude-code",
  "workspace": "/path/to/project",
  "transcript_path": "/path/to/transcript.jsonl"
}
```

Required fields:

- `trigger`: normalized hook trigger.
- `source`: client or adapter name.

Optional fields:

- `workspace`: workspace root associated with the hook.
- `transcript_path`: transcript file associated with the hook.

## Failure Behavior

`zaxy hook-event` writes directly to Eventloom and does not require Neo4j,
Pathlight, embeddings, or a running MCP server. This is deliberate: stop and
pre-compaction hooks should remain fast and should still preserve provenance
when graph projection is unavailable. Clients should treat hook failures as
non-fatal and continue normal execution.

## Custom Adapters

Custom clients can implement hooks by invoking `zaxy hook-event` with one of the
supported triggers, or by appending equivalent typed events through
`memory_append`. Prefer `zaxy hook-event` for lifecycle hooks because it avoids
graph startup work and keeps the hook path deterministic.

Related pages: [README.md](../README.md), [mcp.md](mcp.md),
[eventloom.md](eventloom.md), [agent-events.md](agent-events.md),
[workspace-genesis.md](workspace-genesis.md).
