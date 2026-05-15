# Zaxy Suite-v1 Benchmark Review

This page preserves the suite-v1 representative context benchmark review. For
the current public LongMemEval-compatible headline, BM25 comparison, and
external MemPalace, Mem0, and Agent Memory disclosures, start with
[benchmarks.md](benchmarks.md).

Author: Nicholas Blanchard

Generated result reviewed: May 8, 2026

## Abstract

This review evaluates Zaxy on the `suite-v1` representative context benchmark:
an 850-event, 650-query paired retrieval workload covering current memory,
historical memory, graph traversal, indexed documents, session transcripts, and
mixed cross-lane context. Zaxy is compared against three baselines built over
the same Eventloom corpus: direct markdown-style token scanning, vector-only
retrieval, and markdown candidate generation with vector ranking.

Using OpenAI `text-embedding-3-small` embeddings, Zaxy achieved a mean retrieval
score of `1.000` across all 650 queries. The markdown+vector and vector-only
baselines both achieved `0.520`; direct markdown achieved `0.005`. The paired
mean score difference between Zaxy and markdown+vector/vector was `+0.4795`
with a 95% bootstrap confidence interval of `[0.4431, 0.5154]` and paired
randomization `p = 0.0001`.

The result supports a narrow conclusion: on this generated context-preservation
workload, Zaxy's event-sourced temporal graph architecture retrieves the target
facts more reliably and with fewer returned tokens than the tested markdown and
vector baselines. It does not establish universal superiority over all RAG
systems, all document search engines, or all memory frameworks.

## Research Question

The benchmark asks whether an event-sourced temporal knowledge graph can preserve
and retrieve agent context more accurately than flat markdown and vector
retrieval strategies when the task requires one or more of the following:

- recovering current facts after updates;
- recovering historical facts at a requested time;
- following graph relationships between goals and tasks;
- retrieving cited source-document chunks;
- retrieving sanitized transcript turns;
- assembling mixed context across memory, documents, and transcripts.

The central product question is not "Can Zaxy search text?" A generic vector
store can search text. The question is whether Zaxy can preserve durable working
state in a way that remains temporal, relational, replayable, and accountable.

## Systems Compared

Four retrieval systems were evaluated over the same generated Eventloom corpus.

`md`: A markdown-style baseline that scans serialized event chunks for query
tokens and returns matching chunks in corpus order.

`vector`: A vector-only baseline that embeds every serialized event chunk and
ranks chunks by cosine similarity to the embedded query.

`md+vector`: A hybrid baseline that first selects markdown/token candidates and
then ranks those candidates by vector similarity.

`zaxy`: Zaxy's graph-backed retrieval path. Events are appended to Eventloom,
extracted into typed entities and edges, embedded, projected into Neo4j, and
queried through exact lookup, full-text search, vector search, graph traversal,
identifier-aware keyword expansion, temporal filtering, fusion, and MMR-style
diversity.

## Workload

The reviewed run used `suite-v1`:

| Dimension | Value |
|-----------|-------|
| Events | 850 |
| Queries | 650 |
| Subjects | 100 |
| Documents | 250 |
| Sessions | 50 |
| Embedding model | OpenAI `text-embedding-3-small` |
| Workload SHA-256 | `fd3e2679e37b0953bb2c2ca90f5b98b803a3983b7f0661a6a706e0ef2b41acae` |

The workload lanes were:

| Lane | Query count | Intended pressure |
|------|-------------|-------------------|
| Current | 100 | retrieve the latest valid preference after an update |
| Temporal | 100 | retrieve the earlier valid preference at a historical time |
| Traversal | 100 | recover the task connected to a goal |
| Document | 250 | recover cited document chunks by durable markers |
| Transcript | 50 | recover session decisions from sanitized transcript turns |
| Mixed | 50 | recover context spanning current memory, documents, and transcripts |

## Scoring And Statistical Method

Each query defines expected terms and, where appropriate, forbidden terms. A run
receives credit when returned context contains expected terms and avoids forbidden
terms. This favors factual retrieval over generic semantic similarity.

The benchmark reports:

- mean score by backend;
- p50, p95, and p99 latency;
- returned bytes and approximate token count;
- mean score by lane;
- miss counts by lane;
- paired score deltas against baselines;
- bootstrap confidence intervals;
- paired randomization p-values.

Pairing matters because every backend receives the same query set. The primary
comparisons are within-query score differences between Zaxy and each baseline.
This reduces variance from query difficulty and makes the result more
interpretable than comparing unpaired aggregate means.

## Results

Overall result:

| Backend | Mean score | p50 ms | p95 ms | p99 ms | Approx tokens |
|---------|------------|--------|--------|--------|---------------|
| md | 0.005 | 0.01 | 0.22 | 0.22 | 593 |
| md+vector | 0.520 | 39.91 | 84.40 | 92.60 | 960 |
| vector | 0.520 | 260.24 | 559.89 | 2325.21 | 960 |
| zaxy | 1.000 | 94.01 | 126.26 | 133.60 | 404 |

Category result:

| Backend | Current | Temporal | Traversal | Document | Transcript | Mixed |
|---------|---------|----------|-----------|----------|------------|-------|
| md | 0.000 | 0.000 | 0.035 | 0.000 | 0.000 | 0.000 |
| md+vector | 0.510 | 0.160 | 0.930 | 0.456 | 0.980 | 0.307 |
| vector | 0.510 | 0.160 | 0.930 | 0.456 | 0.980 | 0.307 |
| zaxy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Paired comparisons:

| Comparison | Mean score delta | 95% CI | p-value |
|------------|------------------|--------|---------|
| Zaxy vs md | +0.9946 | [0.9885, 0.9992] | 0.0001 |
| Zaxy vs md+vector | +0.4795 | [0.4431, 0.5154] | 0.0001 |
| Zaxy vs vector | +0.4795 | [0.4431, 0.5154] | 0.0001 |

The current `reports/benchmarks/live-benchmark.*` files now hold the latest
LongMemEval-compatible release evidence. The suite-v1 figures above are retained
in this review as the audited representative-context result; regenerate them
with the suite workload described in [testing.md](testing.md) when a fresh
same-file report is needed.

## Why Zaxy Succeeds On This Workload

Zaxy succeeds because the benchmark rewards properties that are difficult for
flat chunk retrieval to represent.

First, Zaxy models temporal validity directly. A user preference update does not
overwrite history; it closes the previous fact's validity window and opens a new
one. Current and historical queries can therefore be answered by filtering graph
facts by time. Vector baselines see both old and new chunks as semantically
similar, so they often retrieve stale context.

Second, Zaxy preserves typed structure. Goals, tasks, users, preferences,
documents, and transcript turns are not just text fragments. They become graph
entities with typed metadata, citations, and relationships. This lets retrieval
combine exact entity lookup, keyword search, vector similarity, and traversal
rather than relying on embedding proximity alone.

Third, Zaxy uses identifiers as first-class retrieval anchors. Durable markers
such as `doc-code-0015`, `session-0001`, `Goal 0015`, and `user-0015:theme`
should dominate fuzzy semantic similarity. The benchmark exposed this as a
practical ranking requirement: when a machine identifier is present in the
query, the correct cited artifact must not be displaced by merely similar
neighbors.

Fourth, Zaxy returns compact projected facts instead of large raw chunks. In the
reviewed run, Zaxy returned roughly 404 approximate tokens on average, compared
with 960 for vector and markdown+vector baselines. That matters for agents
because memory quality is constrained by both correctness and context-window
budget.

Finally, Zaxy's Eventloom source log preserves replayability and provenance.
The graph is a projection, not the primary record. This is important for agent
debugging and compliance because retrieval can point back to source events or
source files rather than only returning opaque embedding matches.

## Assumptions

The benchmark assumes that generated workloads are useful proxies for recurring
agent-memory tasks. The data is deterministic and structured by design. That
makes it reproducible and diagnostically useful, but it may not capture the full
noise profile of real project repositories, human notes, or production agent
transcripts.

The benchmark assumes that expected-term scoring is a reasonable measure of
retrieval correctness. This is appropriate for facts, identifiers, and
provenance markers, but it is less suitable for open-ended explanatory answers.
Future evaluations should add human or model-graded relevance judgments for
ambiguous natural-language questions.

The benchmark compares Zaxy with local baselines implemented in the Zaxy harness.
It does not directly execute QMD/OpenClaw, Graphiti/Zep, Mem0, or a production
RAG stack with query rewriting, cross-encoder reranking, chunk-level citations,
and tuned ingestion. External systems should be included only when they can run
against the same workload under the same measurement protocol.

The benchmark uses OpenAI embeddings for vector comparisons and Zaxy vector
features. Results may vary with embedding model, reranker configuration, local
embedding providers, corpus size, and Neo4j deployment characteristics.

## Threats To Validity

The largest threat is benchmark specificity. Zaxy is designed for temporal,
relational, replayable agent context. The suite intentionally stresses those
capabilities. A benchmark centered purely on broad semantic document search
could favor a mature document-search sidecar with aggressive reranking and
chunking.

The second threat is synthetic data regularity. Identifiers and event types are
well-formed. Real agent logs contain ambiguity, malformed payloads, duplicate
facts, missing links, sensitive content, and inconsistent naming. Zaxy's
production value depends on robust extraction, normalization, redaction,
degraded-mode behavior, and operator tooling under those conditions.

The third threat is latency interpretation. Zaxy is slower than direct markdown
scan and slower than markdown+vector in p50/p95 latency on this run. It is far
faster than vector-only p95/p99, and it returns fewer tokens, but applications
with very tight retrieval latency budgets must consider graph deployment,
connection pooling, batching, and cache strategy.

The fourth threat is implementation locality. The baselines are intentionally
simple and transparent. They are useful controls, but they are not the strongest
possible implementations of document search. A future publication should run
the same workload against QMD/OpenClaw-style search, Graphiti/Zep-style memory,
Mem0-style memory, and a tuned vector RAG implementation.

## Discussion

The result is strong evidence that Zaxy's architecture is well matched to agent
context preservation. The win is not primarily caused by a better embedding
model; all vectorized systems use the same hosted embedding family. The win
comes from retaining structure that vector search alone discards: fact identity,
validity windows, graph relationships, source citations, and replayable event
history.

The most important qualitative finding from the benchmarking process was that
retrieval quality depends on respecting both semantic and symbolic signals.
OpenAI embeddings are good at semantic neighborhood search, but they can rank
`doc-code-0013` close to `doc-code-0015` because the surrounding language is
nearly identical. For agent memory, that is a material correctness bug. Zaxy's
retrieval layer must therefore treat durable identifiers as precision anchors,
not as ordinary text.

The benchmark also clarifies market positioning. Zaxy should not be framed as
"a better vector database." Its stronger claim is that agent memory needs an
accountable context substrate: append-only events, temporal graph projection,
hybrid retrieval, citations, and MCP-accessible operations. In that frame, Zaxy
is not merely another implementation of chunk RAG. It is an attempt to preserve
the working state that chunk RAG usually flattens away.

## Conclusions

On the reviewed `suite-v1` workload, Zaxy achieves statistically significant
retrieval improvements over markdown, vector, and markdown+vector baselines.
The effect is large, consistent across all evaluated lanes, and accompanied by
substantially smaller returned context than the vector baselines.

The result supports three conclusions:

1. Zaxy's temporal graph model is valuable for agent memory workloads that
   require current-vs-historical truth, graph relationships, citations, and
   mixed context assembly.
2. Flat markdown and simple vector retrieval are insufficient controls for
   durable agent memory because they do not represent validity windows,
   provenance, or multi-hop structure.
3. Zaxy is promising enough to justify stronger external benchmarking against
   QMD/OpenClaw, Graphiti/Zep, Mem0, and tuned production RAG stacks.

The responsible claim is therefore narrow but meaningful: Zaxy has demonstrated
excellent performance on a reproducible, statistically powered representative
agent-context benchmark. The next research step is adversarial and external
comparison, not broader marketing claims.

Related references: [testing.md](testing.md), [retrieval.md](retrieval.md),
[architecture.md](architecture.md), and [README.md](../README.md).
