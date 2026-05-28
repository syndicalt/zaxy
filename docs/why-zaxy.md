# Why Zaxy

Zaxy is heavier than a markdown file and heavier than a standalone vector
database on purpose. It is for agents that need memory to be replayable,
auditable, temporal, and connected across sessions. If the only requirement is
"remember a few preferences," markdown or a hosted memory feature is probably
enough. If the agent needs to explain what it knew, when it knew it, which
source supported it, and how a decision followed from prior work, flat notes
and vector chunks stop being a reliable substrate.

Markdown works well as human-authored notes, but it has weak operational
semantics. Updates overwrite prior wording unless every change is manually
preserved. Relationships are implicit. A model can read the file, but it cannot
replay the sequence of events that produced the file or verify a hash-linked
history. That makes markdown convenient for summaries and poor as the source of
truth for long-running autonomous work.

Vector memory solves a different problem: it retrieves text that is close to a
query. That is useful, and Zaxy can still use embeddings and rerankers, but
similarity is not the same thing as provenance, temporal validity, or multi-hop
reasoning. A vector hit can say "this chunk sounds related." It does not, by
itself, answer whether the fact was superseded, which session produced it, which
source line backs it, or which task and decision chain it belongs to.

Zaxy splits those responsibilities:

- Eventloom is the immutable append-only source of truth.
- Embedded Kuzu is the default temporal graph projection for traversal and
  retrieval without a sidecar service.
- Neo4j is the explicit quality-control and interoperability backend.
- pgGraph is an experimental projection backend for teams evaluating PostgreSQL
  plus graph semantics.
- MCP is the agent-facing interface, so clients call stable tools rather than
  reading private files directly.
- Memory Bootstrap and Memory Checkout give models explicit guidance about what
  to trust, refresh, ignore, and cite.

That architecture has an activation cost. A user must install a Python package,
wire an MCP client, and choose a local graph posture. The intended local entry
point keeps that cost visible rather than hiding it:

```bash
pipx install zaxy-memory
zaxy init
zaxy memory log --eventloom-path .eventloom --limit 5
zaxy memory bootstrap --eventloom-path .eventloom
zaxy doctor --eventloom-path .eventloom
```

After that flow, local data lives in `.eventloom/` as append-only JSONL, and
the generated MCP config or install command tells the client how to launch the
same `zaxy` executable. The graph backend can remain unchecked during the first
smoke test, or it can be started explicitly when graph-backed retrieval is
needed.

Use Zaxy when memory quality depends on temporal state, citations, operational
debugging, or graph traversal. Use markdown or simple vector recall when the
workload is casual, single-session, or easy to reconstruct by rereading notes.

See also [getting-started.md](getting-started.md), [retrieval.md](retrieval.md),
and [README.md](../README.md).
