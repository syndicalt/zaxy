# Eventloom Contract

Eventloom is Zaxy's immutable source of truth. Zaxy reads and writes local JSONL
logs where each line is a validated event. The graph can be rebuilt from these
logs, which means graph projection bugs are recoverable and memory history is
auditable.

An event has a type, actor, payload, sequence number, timestamp, hash, previous
hash, and optional security classification metadata. The exact Python model is
in `src/zaxy/event.py`. The hash chain makes the log tamper-evident: replay can
detect missing, reordered, or edited records. A corrupt projection should be
fixed by replaying the log rather than patching the projection backend directly.

Typed event names matter. Zaxy's extractor uses deterministic handlers for
known events such as `goal.created`, `task.proposed`, and related lifecycle
records. Deterministic extraction is cheaper and easier to test than broad LLM
extraction. New event types should come with a schema decision and extractor
tests before they are considered public. Use `zaxy extractor-template` to
generate a validated starter that follows the registry and provenance pattern:

```bash
zaxy extractor-template decision.recorded \
  --entity-type decision \
  --name-key title \
  --summary-key rationale \
  --actor-relation recorded_decision
```

The generator rejects unsafe event names and identifiers so template rendering
cannot become an injection path. The output is still starter code: add tests
for the specific payload contract before making the extractor public.

For common agent memory events such as decisions, diagnosed issues,
verification evidence, handoffs, and policies, use the payload shapes in
[agent-events.md](agent-events.md). These shapes are backed by built-in
extractors and are intended to keep future `memory_query` calls useful.

For local inspection, generate a standalone HTML viewer:

```bash
zaxy memory status --eventloom-path .eventloom
zaxy memory log --eventloom-path .eventloom --limit 20
zaxy memory diff --eventloom-path .eventloom --session-id default --from-seq 10 --to-seq 20
zaxy viewer .eventloom --output eventloom-viewer.html
```

`zaxy memory status` is the lightweight terminal inspection path. It summarizes
session logs, event counts, latest sequence/hash, latest event type, and
Eventloom integrity without requiring a graph service. Use `--json` when
scripts need the same fields in a stable machine-readable format. Add `--graph`
when the selected projection backend is available and you need to compare
Eventloom with its projection. Graph status reports latest projected sequence/hash, projection lag, missing
`NEXT_EVENT`/`PREVIOUS_EVENT` chain links, and whether the latest graph hash
matches the sealed log.
`zaxy memory log` prints recent memory events in compact reverse-chronological
form with session ID, sequence, short hash, timestamp, event type, actor, and a
payload summary. Use `--session-id` to inspect one session and `--json` for
stable machine-readable entries.
`zaxy memory diff` prints the immutable events added in an inclusive sequence
range for one session. The first version is deliberately event-level; it does
not claim semantic graph/fact diffs.

The viewer reads one JSONL log or every `*.jsonl` file in an Eventloom
directory. It highlights session bootstrap events, lifecycle events, integrity
status, and underlying payloads without requiring a graph service or an MCP
client.

Events are session-scoped. The `SessionManager` maps validated session IDs to
per-session JSONL paths under `EVENTLOOM_PATH`. Multi-agent deployments should
shard by session so a busy agent does not contend on the same file as every
other agent. The graph remains shared, letting retrieval cross relevant facts
while preserving provenance.

Project domains separate default sessions across repositories. Project-local MCP
configs set `ZAXY_DOMAIN` and `EVENTLOOM_THREAD` together, for example
`ZAXY_DOMAIN=zaxy` and `EVENTLOOM_THREAD=zaxy-default`. Codex user-level config
is workspace-neutral for Eventloom/session/domain values; `zaxy serve` derives
the domain default from the process workspace when no explicit Eventloom
environment is supplied. A session ID may still change per run or per agent,
but the domain should stay stable for a project.

Codebase inventory is represented as typed events, not as direct graph writes.
`code.file.indexed` records preserve relative path, language, hash, byte count,
and line count. Replaying Eventloom can regenerate the file map projection.

Context refresh is also event-backed. `zaxy refresh-context` compares current
file fingerprints with the prior refresh state for a session and source kind,
then appends source lifecycle events such as `source.discovered`,
`source.changed`, `source.unchanged`, and `source.deleted`. Derived rows are
tracked through `projection.updated` and `projection.retired`; the latter is
used for changed or deleted sources so the selected projection backend can close
active projection rows while the immutable source history remains replayable.
Refresh state also records the transform version, so a future
extractor/chunker version bump can force unchanged files through the same
retire-and-reindex path.

Replay is the operational escape hatch. If the selected graph projection is
unavailable, Eventloom still contains the history. If an extractor changes,
replay can regenerate the graph. If a handoff needs context, replay
reconstructs the sequence of events from a known point. This is why append
performance and file locking are treated as core requirements.

When a new extractor is added for an event type that already exists in a log,
rebuild the graph projection from Eventloom so retrieval can see the richer
entities and summaries:

```bash
zaxy reproject .eventloom/default.jsonl --session-id default
```

Use `--from-seq` to reproject only newer events after a known migration point.
Reprojection does not rewrite the Eventloom log; it rebuilds selected graph
projections from the immutable events using the current extractor registry.

Do not store secrets in Eventloom payloads. Payloads are durable and may be
exported to observability systems. Event appends redact common secret keys and
secret-looking values before the hash is sealed, and record the affected payload
paths under `security.redacted_paths`. Store references, summaries, or redacted
metadata instead. See [security.md](security.md) for data-handling guidance.

Related pages: [architecture.md](architecture.md), [graph-schema.md](graph-schema.md),
[mcp.md](mcp.md), and [runbook.md](runbook.md). The high-level product framing
is available in [site/index.html](../site/index.html) and [README.md](../README.md).
