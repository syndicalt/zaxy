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

For Codex, the preferred deterministic path is not provider packet capture and
not an assumed project-local hook schema. `zaxy init --preset local-codex`
writes `.codex/zaxy-capture.json`, then `zaxy capture start --workspace .`
starts a managed watcher that imports Codex's own local session JSONL into
Eventloom as normalized transcript, tool-call, command, and file-edit
observations. This keeps capture local, idempotent, and out of the model
request path. Use `zaxy capture status` and `zaxy capture stop` to inspect or
stop the managed watcher. Add `--graph` to `capture start` when Neo4j is
reachable and newly captured observations should be projected immediately.
The underlying `zaxy codex-capture --watch` command remains available for
supervisors that need direct control; use `--watch-iterations <n>` for bounded
health checks and tests.

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
zaxy hook-status --json
zaxy hook-status --eventloom-path .eventloom --json
zaxy capture-soak --eventloom-path .eventloom --session-id my-project-default
zaxy hook-event heartbeat --eventloom-path .eventloom --session-id my-project-default --source manual
```

`hook-status` reports supported client install detection, the latest observed
hook event, and observation coverage by high-value capture type. Missing
`command.completed`, `file.edit.applied`, `tool.call.completed`, or
`transcript.turn` coverage means Zaxy can see lifecycle checkpoints but is not
yet seeing the richer activity needed for durable session reconstruction.
For Codex, install detection and live capture are separate signals:
`.codex/zaxy-capture.json` means local capture is configured, while a running
managed `zaxy capture start` watcher means this session is actively importing
new observations. If Codex capture is configured but the watcher is stopped,
`hook-status` reports a warning and prints the managed command needed to resume
it. `zaxy activate codex` now attempts that managed start automatically when
capture is configured; if the watcher cannot be started or the config is
missing, the activation packet marks capture as degraded so the model does not
mistake a resumed session for an actively captured one.
The same report includes a capture readiness summary. In JSON output, inspect
`capture_readiness.status`, `active_observation_types`, and
`missing_observation_types` to decide whether automatic capture is healthy or
which adapter sinks still need to be wired.
Memory activation status also includes checkout token-efficiency diagnostics
when the latest checkout was produced by the CLI or MCP tool. Inspect
`memory_activation.latest_checkout.token_efficiency.prompt_tokens` and
`facts_per_1k_prompt_tokens` when you need to know whether the session is using
fresh memory without overloading the working context.
The model-facing capability manifest also treats MCP tool availability as
runtime-unverified after resume or process reload. If the active model context
does not expose `memory_checkout`, use the printed `zaxy memory checkout ...`
CLI fallback before substantial work.
`zaxy doctor` also surfaces the same signal as `capture_health`, making it the
single first-run health row for whether deterministic capture is installed,
running when needed, and producing usable observations.
For beta evidence, `zaxy capture-soak` turns the same observation data into a
release-gate report with latest seq/hash, stale lane detection, missing lane
remediation, and a pass/fail `beta_criteria` field.
Claude Code detection parses JSON hook command handlers rather than matching
arbitrary text, so comments or unrelated string fields do not count as an
installation. `heartbeat` is a health probe: it proves the Eventloom observer
path can write without pretending that a real task or compaction happened.

## Supported Clients

| Client | Generated Output | Install Detection | Notes |
|--------|------------------|-------------------|-------|
| Claude Code | JSON settings fragment | `.claude/settings.local.json`, `.claude/settings.json` | Preferred first target for repository-local hook config. |
| Codex | Local capture config plus optional shell snippet | `.codex/zaxy-capture.json`, valid `.codex/hooks.json` | Preferred path is `zaxy capture start`, which manages a local watcher without proxying provider traffic. |
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

Lifecycle hooks also run the memory persistence policy. If Zaxy has not been
used recently, or the hook marks a session start, resume, compaction, long
session, long tool run, or roadmap/status question, Zaxy appends
`memory.reminder.suggested`. That event is intentionally read-only guidance for
the agent: call `memory_bootstrap` if awareness is unclear, then call
`memory_checkout` and trust only cited current checkout facts.

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

Command, file-edit, tool-call, and transcript-turn hooks write first-class
observation events instead of generic `hook.*` checkpoints:

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

zaxy hook-event tool-call \
  --eventloom-path .eventloom \
  --session-id my-project-default \
  --source codex \
  --workspace "$PWD" \
  --tool-name memory_append \
  --tool-status ok \
  --arguments-json '{"event_type":"task.completed"}' \
  --result-summary "seq=42"

zaxy hook-event transcript-turn \
  --eventloom-path .eventloom \
  --session-id my-project-default \
  --source codex \
  --role assistant \
  --content "Recorded the implementation decision." \
  --turn-index 12
```

Command observations redact common secret-bearing arguments and bound stdout and
stderr excerpts. File-edit observations persist path, operation, summary, and
line count metadata; they do not persist source content. Tool-call observations
persist argument keys but not raw argument values. Transcript-turn observations
sanitize content before append so they can serve as source-recall material
without storing obvious secrets.

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
