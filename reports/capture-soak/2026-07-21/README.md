# Capture soak — 2026-07-21

**Status: FAIL — 0 of 4 deterministic capture lanes active.**

BETA.md's release criterion reads:

> A capture soak report shows deterministic capture staying active across a long
> session **or records the exact gaps that remain.**

This artifact is the second half of that sentence. It is not a passing soak, and
it is archived precisely so the gap stops being invisible: before this, the
criterion was simply unmet with nothing under `reports/` to show for it.

## What the run found

| Lane | Observed |
|---|---|
| `command.completed` | no |
| `file.edit.applied` | no |
| `tool.call.completed` | no |
| `transcript.turn` | no |

Codex capture watcher: not installed, not running. `latest_hook_event: null`.

## Why — and a correction to CLAUDE.md

`CLAUDE.md` states, under Local environment quirks:

> Zaxy's own memory hooks run in this repo (`.claude/settings.local.json`):
> session `zaxy-default`, with Stop-hook transcript capture.

**`.claude/settings.local.json` does not exist in this checkout.** So no hooks are
registered, nothing writes the deterministic capture lanes, and the
`zaxy-default` session contains only a `session.genesis` event created when a
fabric connected during unrelated work — not captured activity.

That is why no long-session capture data exists to soak: capture was never
running here, and the controlling doc says otherwise.

## What would close the criterion properly

1. Register the capture hooks in this workspace (`zaxy install`, which writes the
   harness config). **Deliberately not done here** — that mutates the developer's
   local environment, which is theirs to opt into, not something a session should
   switch on unilaterally.
2. Work a long real session with capture active.
3. Re-run `zaxy capture soak --json` and archive the passing report beside this
   one, leaving this file as the before-state.

Until step 1 happens, no amount of running the soak will produce evidence — and
manufacturing a synthetic "long session" to make the number green would be
exactly the kind of thing this repo's benchmark retraction exists to prevent.

## Reproduce

```bash
zaxy capture soak --eventloom-path .eventloom --workspace-root . --json
```
