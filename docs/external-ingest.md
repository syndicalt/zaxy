# External-Producer Ingest

Zaxy's normal write path is single-event: `MemoryFabric.append` →
MCP `memory_append` → CLI `zaxy memory append`. That is the right call when an
agent records one decision, task, or observation at a time.

An **external producer** is different. It already has its own typed event stream
— a tool, a bridge, or another runtime that emits many events with its own
identifiers and causal links — and wants those events to live in Zaxy's
hash-chained log, redaction, and graph projection instead of being written to a
raw JSONL file out of band. The batch-ingest path exists for exactly that:

- **Batch, atomic.** A whole batch is sealed under one exclusive lock; either
  every event is appended or none is (an invalid item rejects the batch with no
  partial writes).
- **Caller-id preserving.** Each event records its producer through the standard
  `actor` field. There is no Event-schema change.
- **Chain preserving.** The producer's causal links — `parent_event_id`,
  `caused_by`, and an external `id` — are threaded through onto each event. Zaxy
  always computes its own `seq`/`prev_hash`/`hash` from the locked tail (the
  producer's own integrity chain is never adopted), but the causal links
  round-trip on [replay](eventloom.md) and are hash-sealed when the event is
  written as an `eventloom.v1` envelope (the normal case for dotted-lowercase
  event types on a fresh or v1 session log).
- **Idempotent.** A reserved payload key, `__zaxy_producer_ref`, dedups
  re-ingest. Re-running the same batch is a no-op.
- **Immediately retrievable.** Every appended event is projected (extract →
  embed → graph upsert), so it shows up in [retrieval](retrieval.md) and
  `memory_checkout` right away.

## Item schema

Each ingest item is a JSON object:

| Field | Required | Meaning |
|---|---|---|
| `event_type` | yes | Dotted-lowercase type, e.g. `decision.made`, `task.created`. |
| `actor` | yes | The producer that emitted the event (the caller-id). |
| `payload` | no | Structured payload object (defaults to `{}`). |
| `producer_ref` | no | Stable per-session source reference used for idempotent dedup, e.g. `limina:evt:42`. Items without it are always appended. |
| `parent_event_id` | no | The producer's parent event id. |
| `caused_by` | no | List of the producer's causal predecessor ids. |
| `id` | no | The producer's external event id. When absent, Zaxy generates a deterministic v1 id. |

`producer_ref` is merged into the sealed payload under the reserved
`__zaxy_producer_ref` key. The `__zaxy_*` namespace is reserved; a payload that
already sets that key has it overwritten by the item's `producer_ref`.

Dedup is scoped to the target session log, so `producer_ref` values must be
unique per session. Point each external producer at its **own** `session_id`
(for example, a bridge writes to a dedicated session). A dedicated session also
guarantees v1 sealing of the causal links, because a fresh log always writes the
`eventloom.v1` envelope.

## Interfaces

### CLI

`zaxy memory ingest` reads one JSON object per line (JSONL) from `--file` or
stdin:

```bash
zaxy memory ingest --file batch.jsonl --session-id limina --json
# imported=2 deduped=0 seq=1..2

# Re-running the same batch is idempotent:
zaxy memory ingest --file batch.jsonl --session-id limina --json
# imported=0 deduped=2 seq=-
```

`batch.jsonl`:

```jsonl
{"event_type":"decision.made","actor":"limina","payload":{"title":"pick X"},"producer_ref":"limina:evt:1","id":"lim-1"}
{"event_type":"task.created","actor":"limina","payload":{"title":"do Y"},"producer_ref":"limina:evt:2","id":"lim-2","parent_event_id":"lim-1","caused_by":["lim-1"]}
```

### MCP

The `memory_ingest` tool (see [docs/mcp.md](mcp.md)) takes an `events` array of
the items above plus an optional `session_id`, and returns
`{"imported", "deduped", "events": [{"seq", "hash"}, ...]}`.

### Python

```python
appended = await fabric.append_batch(items, session_id="limina")
# appended is the list of events actually written (deduped items excluded)
```

`append_batch` deliberately skips the agent-turn-only extras of
`MemoryFabric.append` (encoding classification and generated-inference appends);
it appends, projects each event, and invalidates the query-page cache.

## Replacing a direct-file-write bridge

A producer that previously appended its own JSONL (its own integrity chain, its
own hashing) should instead POST/exec a batch through one of the interfaces
above. The producer keeps emitting its native events with its own ids; it maps
each to an ingest item (`actor` = producer, `producer_ref` = its source ref,
`parent_event_id`/`caused_by` = its causal edges) and lets Zaxy reseal them into
the durable log and graph. Because ingest is idempotent on `producer_ref`, the
bridge can safely replay its backlog without creating duplicates.

See also: [README.md](../README.md) · [Eventloom](eventloom.md) ·
[Retrieval](retrieval.md) · [MCP tools](mcp.md).
