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
For Claude Code JSON settings, `--output` merges Zaxy's hook groups into the
existing settings file and preserves unrelated settings and hook handlers. It
refuses to add duplicate Zaxy handlers unless `--force` is passed, in which
case only existing Zaxy hook handlers are replaced. Generic hook outputs remain
whole-file writes and keep the no-overwrite default.

Inspect the current observer posture:

```bash
zaxy hook-status --eventloom-path .eventloom
zaxy hook-event heartbeat --eventloom-path .eventloom --session-id my-project-default --source manual
```

`hook-status` reports supported client install detection and the latest observed
hook event. Claude Code detection parses JSON hook command handlers rather than
matching arbitrary text, so comments or unrelated string fields do not count as
an installation. `heartbeat` is a health probe: it proves the Eventloom observer
path can write without pretending that a real task or compaction happened.

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
| `heartbeat` | `hook.heartbeat` | Prove the hook write path is healthy. |
| `command` | `command.completed` | Record a bounded, redacted shell command result. |
| `file-edit` | `file.edit.applied` | Record file edit metadata without source content. |

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

Command and file-edit hooks write first-class observation events instead of
generic `hook.*` checkpoints:

```bash
zaxy hook-event command \
  --eventloom-path .eventloom \
  --session-id my-project-default \
  --source codex \
  --workspace "$PWD" \
  --command "pytest" \
  --exit-code 0 \
  --stdout "557 passed"

zaxy hook-event file-edit \
  --eventloom-path .eventloom \
  --session-id my-project-default \
  --source codex \
  --workspace "$PWD" \
  --path src/zaxy/core.py \
  --operation modified \
  --summary "Updated context assembly"
```

Command observations redact common secret-bearing arguments and bound stdout and
stderr excerpts. File-edit observations persist path, operation, summary, and
line count metadata; they do not persist source content.

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
