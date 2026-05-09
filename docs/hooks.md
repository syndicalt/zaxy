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

Write config directly during onboarding:

```bash
zaxy hooks claude-code \
  --eventloom-path .eventloom \
  --domain my-project \
  --output .claude/settings.local.json
```

`--output` creates parent directories and refuses to overwrite existing files.
Pass `--force` when replacing a generated hook config intentionally.

## Supported Clients

| Client | Generated Output | Install Detection | Notes |
|--------|------------------|-------------------|-------|
| Claude Code | JSON settings fragment | `.claude/settings.local.json`, `.claude/settings.json` | Preferred first target for repository-local hook config. |
| Codex | Shell snippet | `.codex/hooks.json` when present | Codex hook support exists behind feature/config paths, but project-local interactive hook behavior is still evolving. Use the generic snippet unless your Codex version documents a working `hooks.json` path. |
| Generic | Shell snippet | Any explicit file you wire manually | Use for clients that can run lifecycle shell commands. |

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

Checkpoint hooks can include retrieval-useful metadata:

```bash
zaxy hook-event checkpoint \
  --eventloom-path .eventloom \
  --session-id my-project-default \
  --source codex \
  --summary "Finished hook install mode." \
  --reason manual \
  --turn-count 7
```

`hook.checkpoint` events are projected into graph `hook_checkpoint` entities so
future retrieval can find durable session milestones.

## Payload

Hook events use actor `zaxy-hook` and append to the selected session log. The
payload is intentionally small:

```json
{
  "trigger": "precompact",
  "source": "claude-code",
  "workspace": "/path/to/project",
  "transcript_path": "/path/to/transcript.jsonl",
  "summary": "Finished hook install mode.",
  "reason": "manual",
  "turn_count": 7
}
```

Required fields:

- `trigger`: normalized hook trigger.
- `source`: client or adapter name.

Optional fields:

- `workspace`: workspace root associated with the hook.
- `transcript_path`: transcript file associated with the hook.
- `summary`: short checkpoint summary.
- `reason`: checkpoint reason such as `manual`, `interval`, `precompact`, or
  `shutdown`.
- `turn_count`: client turn count at the checkpoint.

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
