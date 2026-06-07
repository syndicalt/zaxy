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

MemPalace and AgentGraph-style graph memory are the current public validation
points for Zaxy's thesis. The market is moving toward persistent, structured
memory rather than flat context files. Zaxy should compete on trust,
provenance, and replayability rather than trying to match every UX surface
first. The benchmark lane should stay architecture-driven:

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

The current Zaxy public benchmark hub is [benchmarks.md](../benchmarks.md). Zaxy's
same-harness evidence now leads with the full 500-question current74 checkout
archive: mean score 0.940, Answer@5 0.906, citation coverage 1.000,
R@1/R@5/R@10 of 0.906/1.000/1.000, p95 687.67 ms, and p99 969.10 ms on
workload SHA-256 `90fb2307195d7e16b963a2b8a30f03b375bd42a45d41aeaa55423029dd84e3fc`.
The 100-question BM25 comparison remains a useful tradeoff report: Zaxy mean
score 0.970, Answer@5 0.950, citation coverage 1.000, and R@1/R@5/R@10 1.000,
with BM25 in the same report at mean score 0.540, Answer@5 0.500, and R@5
0.840. The older BM25-included full 500-question hash run remains historical
legacy `limit=10` floor evidence, not the current headline. Projection-backend
work should use the current same-harness `limit=5` backend-evaluation floor
instead of mixing those older `limit=10` thresholds into backend comparisons.

Competitor numbers belong in an external-disclosure table, not a universal
leaderboard. MemPalace publicly reports 96.6% raw LongMemEval R@5 and 98.4%
held-out hybrid R@5, plus optional LLM-reranked full-set runs reported at
99%+ R@5. Agent Memory publicly reports 95.2% R@5 on LongMemEval-S. Mem0
publicly reports LongMemEval accuracy in the low-to-mid 90s and lower-token
memory retrieval, plus LoCoMo accuracy gains. Zep/Graphiti remains an important
graph-memory product reference point, but its public numbers should still be
kept separate unless a same-harness adapter is pinned. GBrain is the
benchmark-visible Harvey LAB comparison: Zaxy's audited Harvey run beats the
article-best task rows by +0.081 mean criterion pass rate and wins 9/10 tasks,
but that is article-relative evidence rather than a Zaxy-run GBrain adapter.
These claims are important market context, but they are not same-harness Zaxy
results.

## Same-Harness Adapter Feasibility

MemPalace is the strongest adapter candidate because its public repo documents
a local `benchmarks/longmemeval_bench.py` path and committed per-question
result files. A Zaxy adapter should wrap that command, pin the mode and top-k
settings, and import per-question retrieval hits into Zaxy's report schema.

Mem0 is a benchmark harness candidate rather than a drop-in retrieval adapter:
the public `mem0ai/memory-benchmarks` project can run LongMemEval, but the OSS
path requires Docker, Qdrant, model configuration, and LLM answer/judge choices.
The first Zaxy integration should document those inputs and separate
retrieval-only comparisons from judge-scored answer accuracy.

Agent Memory remains external disclosure only for now. Its product page reports
LongMemEval-S R@5 and a BM25/vector/graph retrieval stack, but the public page
does not provide a stable same-harness CLI/API contract for Zaxy to call. Keep
the number in the disclosure table until a reproducible command and result
export are available.

Zep/Graphiti should be treated as a larger graph-memory comparison target, not
as a thesis-adjacent ally to attack. A same-harness adapter would need to pin
the graph construction path, retrieval limit, source-citation mapping, and
latency/token accounting.

GBrain belongs in the Harvey LAB lane. The current publishable claim is that
Zaxy beats the article-best GBrain task rows in the audited Harvey artifact
comparison; a stronger claim requires a pinned GBrain runner.

## Near-Term Roadmap

- Treat the current74 full 500-question LongMemEval-compatible report as the
  public headline and no-regression floor: mean score 0.940, Answer@5 0.906,
  citation coverage 1.000, and R@5 1.000. Work down synthesis misses without
  reducing citation coverage or R@5.
- Keep same-harness BM25 baselines in public benchmark reports so Zaxy claims
  show both retrieval quality and latency/token tradeoffs against a strong
  lexical baseline.
- Build or document feasible same-harness competitor adapters for MemPalace,
  Mem0, Agent Memory, and Zep/Graphiti. Keep GBrain in the Harvey LAB
  article-relative lane until a pinned runner exists. If a same-harness adapter
  is not practical, document the blocker and keep the competitor number in the
  external-disclosure table.
- Add Skill Memory as a first-class procedural layer: skill lifecycle events,
  skill-version graph projection, checkout skill routing, outcome tracking, and
  eval-gated promotion/rollback.
- Evaluate pgGraph as an experimental Postgres-backed projection backend behind
  a backend-neutral contract, with embedded Kuzu remaining the default until a
  candidate backend proves same-harness benchmark and operations parity or better.
- Expand clean-repo onboarding and model-facing UAT across fresh Codex and
  Claude Code workspaces: `zaxy init`, Memory Bootstrap, deterministic capture,
  Memory Checkout, retrieval feedback, and doctor status.
- Keep public benchmark scripts reproducible from a clean checkout with cached
  datasets, archived reports, guardrail commands, and no hidden manual steps.
- Keep LangGraph native-beta, CrewAI native-preview, and OpenAI-compatible
  model-call adapters dependency-light on the shared native payload contract,
  `zaxy.native.v0.6`.
- Keep AutoGen template-only until the right runtime hooks are clear.
- Add public comparison workloads that evaluate temporal, source-recall,
  graph-traversal, and context-collapse behavior against MemPalace-style memory.
- Keep claims reproducible: every comparison should publish workload hashes,
  retrieval settings, and exact scoring rules.

Related pages: [archived benchmark-review.md](benchmark-review.md),
[benchmarks.md](../benchmarks.md), [integrations.md](../integrations.md), and
[architecture.md](../architecture.md).
