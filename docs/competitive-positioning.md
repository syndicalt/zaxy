# Competitive Positioning

Zaxy's product thesis is "git for agent memory": an event-sourced, replayable,
auditable memory fabric that projects durable context into graph, lexical,
verbatim, and active working-set views.

## What Stands Out

- Eventloom append-only JSONL is the source of truth, with hash-chain integrity
  and deterministic replay.
- Bi-temporal graph projection preserves what was true then versus now.
- Retrieval uses graph, lexical, vector, verbatim source recall, and active
  working-set projection rather than only chunk similarity.
- Automatic capture writes typed lifecycle observations for transcript turns,
  tool calls, command results, file edits, compaction, subagents, and session
  boundaries.
- Context assembly returns prompt-ready memory with citations, source lanes,
  policy metadata, warnings, and feedback hooks.

## MemPalace Target

MemPalace is the current public target for LLM memory product comparison. Zaxy
should compete on trust and provenance rather than trying to match every UX
surface first. The benchmark lane should stay architecture-driven:

1. Temporal correctness: recover old and current facts without overwriting
   history.
2. Source recall: answer with verbatim Eventloom citations and transcript
   source anchors.
3. Relational recall: follow graph relationships across goals, tasks,
   decisions, files, symbols, and test coverage.
4. Context collapse resistance: preserve identity through compaction and active
   working-set projection.
5. Auditability: replay how a memory was written, projected, retrieved, and
   reinforced.

## Near-Term Roadmap

- Keep LangGraph as the first native-preview adapter and use it to learn the
  maintained adapter interface.
- Build CrewAI next if LangGraph's adapter shape holds; keep AutoGen template
  only until the right runtime hooks are clear.
- Add public comparison workloads that evaluate temporal, source-recall,
  graph-traversal, and context-collapse behavior against MemPalace-style memory.
- Keep claims reproducible: every comparison should publish workload hashes,
  retrieval settings, and exact scoring rules.

Related pages: [benchmark-review.md](benchmark-review.md),
[integrations.md](integrations.md), and [architecture.md](architecture.md).
