# Retrieval

Zaxy retrieval is hybrid by design. Agent memory queries are rarely solved by a
single strategy. Exact lookup is excellent when a query names a known entity.
Full-text search is useful for names and summaries. Vector similarity helps
when the phrasing changes. Graph traversal brings in connected facts that a flat
retriever would miss.

The query router lives in `src/zaxy/query.py`. It validates the query and limit,
calls `GraphStore.search_exact`, `search_keyword`, and `search_vector` where
appropriate, expands from high-confidence hits through traversal, fuses scores,
applies MMR diversity, and returns `ContextChunk` objects. A context chunk
contains the content an agent should see and metadata about source entities,
scores, and provenance.

Ranking is intentionally explainable. Each chunk carries `score_explanation`
metadata with the retrieval source, raw backend score, source weight, weighted
score, matched query, query expansion weight, temporal scoring fields when an
as-of query is used, retention policy effects when configured, and final ranking
score. The ranking pass uses maximum marginal relevance so near-duplicate hits
do not crowd out adjacent context. Traversal hits get a small preservation bonus
because graph-neighbor evidence is often the difference between generic search
and relational memory.

Keyword search includes a deterministic expansion pass for terse agent queries.
For example, `auth decision` also searches known equivalents such as
`authentication`, `authorization`, and `rationale`. Expansion is bounded to one
additional query and receives a small query-weight discount so broadened matches
help recall without overpowering the user's literal query.

Scoring profiles make retrieval policy explicit. `balanced` is the default.
`precision` favors exact and literal evidence, `recall` keeps more vector and
graph-neighbor evidence in play, and `temporal` gives as-of freshness a stronger
role. Callers can also pass a custom `ScoringProfile` or override individual
fusion weights for advanced deployments.

Reranking is pluggable. `LexicalReranker` is a deterministic local provider
that promotes candidates with better query-token coverage over the fused graph
candidate set. Hosted or model-backed rerankers can implement the same async
interface and return candidates with `reranker` and `rerank_score` metadata.
Zaxy ships `HTTPReranker` for local/self-hosted rerank endpoints and
`OpenAICompatibleReranker` for OpenAI-compatible chat-completions models that
return JSON candidate scores. `build_reranker(settings)` wires configured
providers into `MemoryFabric`.

Retrieval degrades by strategy instead of failing the whole query. If vector
search is unavailable, exact, keyword, and traversal retrieval continue. If a
reranker endpoint fails, Zaxy returns the built-in MMR order and records a
`reranker unavailable` warning in score metadata. If Neo4j cannot be reached,
`MemoryFabric.query()` falls back to the durable Eventloom log and marks
returned contexts as degraded with the fallback reason. Embedding provider
outages disable only vector participation for that call.

Every graph-backed context chunk should cite its originating Eventloom event
when provenance is available. Citations use the form
`eventloom://<session>/events/<seq>#<hash-prefix>`. They let callers show why a
fact exists, replay the surrounding session, and distinguish retrieved context
from unsupported generated text.

Filesystem document chunks use file-line citations when source path metadata is
available: `file://docs/guide.md:42`. These chunks still originate from
Eventloom `document.indexed` events, but retrieval prefers the file citation
because it is the most useful pointer for human review and editor navigation.

Temporal filtering is a first-class part of retrieval. Without a temporal
filter, graph search returns current facts. With an `as_of` filter, the graph
returns facts whose validity window contains that time. This is what lets agents
answer questions like "what did we believe before the rollback?" without losing
newer corrections. As-of retrieval also applies a small temporal-proximity
score that prefers facts asserted closer to the requested point in time while
keeping old-but-still-valid facts eligible.

Retention is retrieval-side and non-destructive. Eventloom remains immutable and
Neo4j facts remain replayable; retention policies only filter or rescore
candidate context. `RETENTION_POLICY=filter_expired` hides results whose
`expires_at` metadata is at or before the query time. `RETENTION_POLICY=decay`
keeps results eligible but applies a half-life multiplier based on
`last_reinforced_at` or `valid_from`, with optional `importance` and
`reinforcement_count` metadata nudging the multiplier. Expired results under
decay use `RETENTION_EXPIRED_WEIGHT`. Goal, task, decision, context policy, fallback event, and `memory.reinforced` extractors project these fields into graph properties. These effects are exposed in
`score_explanation` and are not written back as memory facts.

The vector path depends on embeddings. Local deterministic embeddings are useful
for tests and offline development. Hosted embeddings are better for semantic
quality. Both feed the same vector index shape. See [embeddings.md](embeddings.md)
for provider configuration.

Fusion should remain conservative. Exact entity matches should not be drowned
out by vague semantic hits. Traversal should add connected evidence, not flood
the prompt. Limits are validated centrally, and traversal depth is bounded in
`src/zaxy/security.py` to avoid runaway graph expansion.

Document ingestion is intentionally source-preserving rather than a separate
chunk store. `MemoryFabric.ingest_documents()` reads supported local files
(`.md`, `.markdown`, `.txt`, `.rst`), skips hidden directories, chunks by line
range, appends `document.indexed` events, and projects those chunks into the
same graph as agent memory. This gives Zaxy generic project-material recall
without losing replayability or provenance.

Codebase indexing follows the same Eventloom-first shape. `zaxy index-codebase`
and `MemoryFabric.ingest_codebase()` append `code.file.indexed`,
`code.symbol.indexed`, `code.import.indexed`, and `code.dependency.indexed`
events plus Python, JavaScript/TypeScript, Go, Rust, and Java
`code.call.indexed` events and Python `code.coverage.indexed` events for
supported source files. The graph projection creates `code_file`,
`code_symbol`, `code_import`, `code_call`, and `code_coverage` entities plus
`depends_on_file`, `calls_symbol`, and `tests_symbol` edges so retrieval can
answer inventory, definition, import, local dependency, call graph, and static
test coverage questions without storing full source text. Go package-qualified
local calls can resolve across files when the imported package maps to a scanned
local package directory, and simple Rust `use crate::module::symbol` imports can
resolve calls to scanned sibling module files. Java package imports can resolve
class-qualified calls to scanned local `.java` files.

Transcript ingestion follows the same rule. `MemoryFabric.ingest_transcript()`
turns session messages into sanitized `transcript.turn` events and graph
`transcript_turn` entities. `MemoryFabric.assemble_context()` can then combine
recent replay with ranked retrieval, giving callers a single prompt-ready view
without throwing away replayability.

Benchmark coverage lives in `src/zaxy/benchmark.py`, `src/zaxy/live_benchmark.py`,
`tests/test_competitive_benchmarks.py`, and `tests/test_live_benchmark.py`. The
current live benchmark compares markdown, BM25, vector, markdown+vector, and Zaxy
retrieval on generated paired workloads. Use `--workload frozen` for publishable
statistical temporal-memory runs, `--workload temporal-recall` for the narrower
MemPalace-comparable as-of recall lane, `--workload source-recall` for exact
cited-source recall against target and distractor documents, `--workload
graph-traversal` for goal-task-completion path recall, `--workload
context-collapse` for noisy transcript plus checkpoint recovery, and `--workload
suite` for representative runs that add indexed documents, sanitized
transcripts, and mixed cross-lane queries. Reports include workload versions,
source recall, citation coverage, and SHA-256 fingerprints so results remain
comparable over time. Treat these as workload-specific signals, not universal
claims against every production-grade vector RAG or file memory system.

Use `zaxy benchmark-inventory` when the goal is release evidence rather than a
live retrieval run. It emits the four MemPalace-comparable lanes, frozen
versions, fingerprints, event/query counts, product claims, and required metrics
without requiring Neo4j or provider quota.

The current public LongMemEval-compatible evidence is summarized in
[benchmarks.md](benchmarks.md). The archived Zaxy-only 100-question report
shows mean score 0.950, Answer@5 0.950, citation coverage 1.000, and
R@1/R@5/R@10 0.990. The archived same-harness BM25 comparison shows BM25 R@5
0.840 versus Zaxy checkout R@5 0.990. The archived legacy `limit=10` full
500-question hash run is the broader checkout no-regression floor: Zaxy checkout
mean score 0.626, Answer@5 0.608, citation coverage 1.000, and R@5 0.956. The
current backend-evaluation floor is the same-harness `limit=5` Neo4j checkout
control with workload SHA-256
`0dc36a139bb9a4fdc7c6cd34400737a58a1eb7410517341f015e9fbfc76ed854`: mean
score 0.712, Answer@5 0.620, citation coverage 1.000, and R@5 0.958. External
MemPalace, Mem0, and Agent Memory numbers should be described as external
disclosures until those systems run through the same Zaxy harness.

The next retrieval-quality work should close the practical ergonomics gap with
QMD-style search sidecars: richer assembly lifecycle hooks, stronger local
embedding providers, and broader degraded-mode observability. These should
augment Zaxy's temporal/provenance layer rather than replace it with generic
chunk search.

Future compaction work should remain identity-preserving. Consolidated vectors
or summaries may route queries and reduce token load, but they should not become
the sole authority for event, document, transcript, or graph facts. Prompt
assembly emits warnings when compacted or projection-derived context lacks
source-level citations, and when replay truncation leaves no retrieved source
support. `MemoryFabric(eventloom_path=...)` auto-discovers
`*.compaction.json` artifacts under the Eventloom directory; explicit
`projection_paths=[...]` remain available for artifacts stored elsewhere.
Returned projection contexts carry `projection_id`, `event_ref`, and source
citations so prompt assembly can distinguish supported compact context from
degraded summaries. See [consolidation.md](consolidation.md) for the
geometry-aware consolidation roadmap and identity-preserving projection model.

Related references: [graph-schema.md](graph-schema.md), [mcp.md](mcp.md),
[configuration.md](configuration.md), [testing.md](testing.md),
[consolidation.md](consolidation.md), and [README.md](../README.md). The public
explanation is [site/index.html](../site/index.html).
