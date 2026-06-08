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
The `write_instructions.memory_activation` block is intentionally operational:
it records `memory_checkout` as required before roadmap, implementation, release,
review, resume, and high-context work, plus concrete command templates for
`zaxy activate codex`, `zaxy activate codex --launch`, `zaxy hook-event resume`,
and the CLI `zaxy memory checkout` fallback. It also marks MCP memory tools as
`runtime_unverified` and sets `fail_closed=true`, so agents that can still read
genesis after `/resume`, compaction, or MCP reload have a durable blocker instead
of a vague preference.
The activation templates include explicit `<eventloom_path>` and
`<workspace_root>` placeholders, matching `zaxy init` output so copied commands
stay tied to the initialized repository even when run from another shell.
`zaxy init` also mirrors the operational part of this policy into a
marker-managed `Zaxy Memory Activation` block in `AGENTS.md` by default. That
block is the model-visible fallback for Codex sessions that resume after MCP tool
reloads or context compaction. Use `--no-agent-instructions` to skip that write
when a repository manages agent instructions another way.

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
