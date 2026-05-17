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

## Public Benchmark Posture

The current Zaxy public benchmark hub is [benchmarks.md](benchmarks.md). Zaxy's
same-harness evidence should distinguish the archived LongMemEval-compatible
100-question headline from the full 500-question archive. The 100-question run
remains the strongest headline: mean score 0.950, Answer@5 0.950, citation
coverage 1.000, and R@1/R@5/R@10 0.990. The full 500-question hash run is a
separate no-regression floor: Zaxy checkout mean score 0.626, Answer@5 0.608,
citation coverage 1.000, and R@1/R@5/R@10 of 0.944/0.956/0.956 versus BM25
mean score 0.560, Answer@5 0.516, and R@5 0.770.

Competitor numbers belong in an external-disclosure table, not a universal
leaderboard. MemPalace publicly reports 96.6% raw LongMemEval R@5 and 98.4%
held-out hybrid R@5. Agent Memory publicly reports 95.2% R@5 on LongMemEval-S.
Mem0 publicly reports +26% Accuracy over OpenAI Memory on LOCOMO, which is a
different metric family. These claims are important market context, but they are
not same-harness Zaxy results.

## Near-Term Roadmap

- Treat the archived full 500-question LongMemEval-compatible report as the
  no-regression floor, and work down synthesis misses without reducing citation
  coverage or R@5.
- Keep same-harness BM25 baselines in public benchmark reports so Zaxy claims
  show both retrieval quality and latency/token tradeoffs against a strong
  lexical baseline.
- Build or document feasible same-harness competitor adapters for MemPalace,
  Mem0, and Agent Memory. If a same-harness adapter is not practical, document
  the blocker and keep the competitor number in the external-disclosure table.
- Add Skill Memory as a first-class procedural layer: skill lifecycle events,
  skill-version graph projection, checkout skill routing, outcome tracking, and
  eval-gated promotion/rollback.
- Evaluate pgGraph as an experimental Postgres-backed projection backend behind
  a backend-neutral contract, with Neo4j remaining the default until same-harness
  benchmark and operations gates prove parity or better.
- Expand clean-repo onboarding and model-facing UAT across fresh Codex and
  Claude Code workspaces: `zaxy init`, Memory Bootstrap, deterministic capture,
  Memory Checkout, retrieval feedback, and doctor status.
- Keep public benchmark scripts reproducible from a clean checkout with cached
  datasets, archived reports, guardrail commands, and no hidden manual steps.
- Keep LangGraph and CrewAI native-preview adapters dependency-light and use
  them to learn the maintained adapter interface.
- Keep AutoGen template-only until the right runtime hooks are clear.
- Add public comparison workloads that evaluate temporal, source-recall,
  graph-traversal, and context-collapse behavior against MemPalace-style memory.
- Keep claims reproducible: every comparison should publish workload hashes,
  retrieval settings, and exact scoring rules.

Related pages: [benchmark-review.md](benchmark-review.md),
[benchmarks.md](benchmarks.md), [integrations.md](integrations.md), and
[architecture.md](architecture.md).
