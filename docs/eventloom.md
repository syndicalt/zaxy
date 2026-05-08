# Eventloom Contract

Eventloom is Zaxy's immutable source of truth. Zaxy reads and writes local JSONL
logs where each line is a validated event. The graph can be rebuilt from these
logs, which means graph projection bugs are recoverable and memory history is
auditable.

An event has a type, actor, payload, sequence number, timestamp, hash, previous
hash, and optional security classification metadata. The exact Python model is
in `src/zaxy/event.py`. The hash chain makes the log tamper-evident: replay can
detect missing, reordered, or edited records. A corrupt projection should be
fixed by replaying the log rather than patching Neo4j directly.

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

Events are session-scoped. The `SessionManager` maps validated session IDs to
per-session JSONL paths under `EVENTLOOM_PATH`. Multi-agent deployments should
shard by session so a busy agent does not contend on the same file as every
other agent. The graph remains shared, letting retrieval cross relevant facts
while preserving provenance.

Replay is the operational escape hatch. If Neo4j is unavailable, Eventloom still
contains the history. If an extractor changes, replay can regenerate the graph.
If a handoff needs context, replay reconstructs the sequence of events from a
known point. This is why append performance and file locking are treated as core
requirements.

When a new extractor is added for an event type that already exists in a log,
rebuild the graph projection from Eventloom so retrieval can see the richer
entities and summaries:

```bash
zaxy reproject .eventloom/default.jsonl --session-id default
```

Use `--from-seq` to reproject only newer events after a known migration point.
Reprojection does not rewrite the Eventloom log; it rebuilds Neo4j projections
from the immutable events using the current extractor registry.

Do not store secrets in Eventloom payloads. Payloads are durable and may be
exported to observability systems. Event appends redact common secret keys and
secret-looking values before the hash is sealed, and record the affected payload
paths under `security.redacted_paths`. Store references, summaries, or redacted
metadata instead. See [security.md](security.md) for data-handling guidance.

Related pages: [architecture.md](architecture.md), [graph-schema.md](graph-schema.md),
[mcp.md](mcp.md), and [runbook.md](runbook.md). The high-level product framing
is available in [site/index.html](../site/index.html) and [README.md](../README.md).
