# Workspace Genesis

Zaxy can initialize a session with a durable workspace profile before other
memory writes or indexing work. The entrypoint is:

```bash
zaxy init-session . --session-id zaxy-default
```

The command inspects lightweight filesystem signals and appends a
`session.genesis` event. The event records the root path, workspace type,
confidence, matched signals, instructions profile, session ID, and write
instructions.

The first profile is intentionally conservative. Codebase workspaces are
detected from signals such as `pyproject.toml`, `package.json`, `go.mod`,
`Cargo.toml`, `src/`, `tests/`, and `.git/`. If the root cannot be classified
confidently, Zaxy falls back to `generic_workspace` with primitive event types
such as `observation.recorded`, `decision.made`, `task.completed`, and
`artifact.indexed`.

Genesis events are auditable. If Zaxy classifies a workspace incorrectly, append
a `session.profile.corrected` event instead of rewriting history. That preserves
the provenance trail: what Zaxy believed at session start, which write
instructions that implied, and when the profile changed.

MCP startup also discovers workspace instruction files adjacent to genesis:
`AGENTS.md`, `CLAUDE.md`, `SOUL.md`, and `.github/copilot-instructions.md`.
Zaxy writes compact `workspace.instructions.discovered` events with per-file
hashes, citations, kinds, sizes, and short summaries; it does not store full
instruction file content. On later startup, if the discovered signature differs
from the latest recorded signature, Zaxy appends `workspace.instructions.updated`
with `previous_signature` to preserve instruction drift.

Related pages: [eventloom.md](eventloom.md), [codebase.md](codebase.md), and
[runbook.md](runbook.md).
