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
and relational memory. Traversal score explanations include
`path_relation_types`, so Coordinate proof queries can show chains such as
mission proof, synthesis artifact, answer candidate, and ledger-row links.

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

## The cognitive retrieval profile

`RETRIEVAL_PROFILE=cognitive` is a named profile that layers cognitive-memory
policy over the same local retrieval stack as `local_fast`. It is the default
since 2.1.0, promoted by the internal forgetting lane: exact cold-start
parity (zero-reinforcement corpora rank byte-identically to plain retrieval),
no-recall-loss 1.0 for attenuated memories, pin/authority exemptions 1.0, and
ranking lift 1.0 vs 0.0. The plain profiles remain available — set
`RETRIEVAL_PROFILE=local_fast` to restore the pre-2.1.0 default — and every
plain profile keeps byte-identical pre-cognitive behavior. The cognitive
profile enables three flags:

- **Salience-blended ranking** (`salience_ranking`). Checkout relevance is
  multiplied by normalized salience from the reinforcement ledger:
  `m = clamp(score / base, [0.01, 10.0])`. A never-reinforced memory keeps
  the implicit base salience `1.0` and ranks exactly as before; reinforced
  memories rise and invalidated memories sink. Memories whose replayed
  salience falls below `SALIENCE_FLOOR` (default `0.15`) leave default
  checkout ranking entirely and are listed — labeled `attenuated`, with
  their scores — in `diagnostics.attenuation`. Attenuation is projection
  policy only: the events are untouched, and the memories stay fully
  reachable through explicit `memory_query`/`memory_replay`, which never
  route through the blend. Authority-bearing state (accepted review status
  or accepted-authority scope) and events appended with `"pinned": true`
  payload metadata are exempt from the floor and reported as exempt.
- **Cue blending** (`cue_blending`). Appends may carry a `cues` payload
  record with optional `mission`, `workspace`, `tool`, and `phase` string
  fields (the Codex capture path records `workspace` and `tool`
  automatically). Checkout and `memory_query` accept an optional `cues`
  argument; the Jaccard overlap between query cues and stored cues is
  blended as a bounded bonus of `0.25 * jaccard` (at most `0.25` for a
  perfect match). Missing cues on either side contribute nothing.
- **Graph walk** (`graph_walk`). When the projection backend exposes an
  adjacency snapshot (`fetch_adjacency`), retrieval seeds a bounded
  personalized PageRank (`alpha=0.85`, 20 iterations) from query-matched
  entities and folds max-normalized walk mass into the fused candidate
  scores as `(1 - 0.2) * fused + 0.2 * walk`. The walk re-weights existing
  candidates (it never invents uncited ones), is deterministic, and is
  cached per `(snapshot signature, seed set)` so repeated queries replay
  the cached walk until the projected graph changes.

The write-time encoding gate (`ENCODING_GATE_ENABLED`, default off) and
interference detection complement the profile: gate tags
(`novel`/`reinforcing`/`redundant`) ride in event payload metadata without
ever dropping an event, redundant appends emit a weak reinforcement toward
the duplicated memory, and a novel append that contradicts an above-floor
memory on the same entity's scalar property emits a review-pending
`memory_propose_belief_update` proposal citing both events.

Purpose profiles make the retrieval-time ontology explicit. Memory Checkout
accepts `purpose` as a preset (`coding`, `review`, `release`, `security`,
`research`, or `coordinate`) or as a structured profile with role, task, risk,
time horizon, expected action, permission scope, evidence policy, retention
policy, and ontology lens fields. The profile is returned in checkout
diagnostics and prompt guidance, so the same source evidence can be treated as
an implementation invariant, release risk, security exposure, research
contradiction, pending diagnostic, accepted coordinator finding, or handoff
proof depending on the caller's intended action. Non-general profiles add
deterministic retrieval emphasis terms, purpose-specific recall floors, and a
purpose-selected scoring profile before checkout projection; the applied policy is reported in
`diagnostics.purpose_retrieval_policy`. Checkout also applies the profile's
purpose policy before retrieved rows become current memory: suppressed rows are
excluded from `current_facts` and cited evidence, while counts and reasons are
reported in `diagnostics.purpose_policy` and `retention.purpose_policy`. The
`coordinate` profile preserves accepted parent state and proof packets while
suppressing worker-local pending rows unless diagnostics are requested.

Reranking is pluggable. `LexicalReranker` is a deterministic local provider
that promotes candidates with better query-token coverage over the fused graph
candidate set. Hosted or model-backed rerankers can implement the same async
interface and return candidates with `reranker` and `rerank_score` metadata.
Zaxy ships `HTTPReranker` for local/self-hosted rerank endpoints and
`LateInteractionHTTPReranker` for ColBERT-style token-interaction endpoints that
accept tokenized query and candidate payloads. It also ships
`OpenAICompatibleReranker` for OpenAI-compatible chat-completions models that
return JSON candidate scores. Score diagnostics include `rerank_strategy` so
benchmark reports can separate deterministic lexical, cross-encoder, hosted, and
late-interaction reranking paths. `build_reranker(settings)` wires configured
providers into `MemoryFabric`.

Retrieval degrades by strategy instead of failing the whole query. If vector
search is unavailable, exact, keyword, and traversal retrieval continue. If a
reranker endpoint fails, Zaxy returns the built-in MMR order and records a
`reranker unavailable` warning in score metadata. If the selected graph
projection cannot be reached, `MemoryFabric.query()` falls back to the durable
Eventloom log and marks returned contexts as degraded with the fallback reason.
Embedding provider outages disable only vector participation for that call.

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
projected facts remain replayable; retention policies only filter or rescore
candidate context. `RETENTION_POLICY=filter_expired` hides results whose
`expires_at` metadata is at or before the query time. `RETENTION_POLICY=decay`
keeps results eligible but applies a half-life multiplier based on
`last_reinforced_at` or `valid_from`, with optional `importance` and
`reinforcement_count` metadata nudging the multiplier. Expired results under
decay use `RETENTION_EXPIRED_WEIGHT`. Purpose-scoped memories can override the
decay half-life without mutating storage: Coordinate and security memories use
at least a 180-day half-life, release memories at least 120 days, and review
memories at least 90 days, while coding and research use the configured default.
Expired Coordinate, security, release, and review memories under decay keep a
small bounded score floor; `filter_expired` still hides expired rows regardless
of purpose. Goal, task, decision, context policy, fallback event, and
`memory.reinforced` extractors project these fields into graph properties. These effects are exposed in
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
without requiring a graph sidecar or provider quota.

The current public benchmark evidence is summarized in
[benchmarks.md](benchmarks.md). Treat that page as the source of truth for
retrieval-quality claims: the active public surface is the headline
500-question LongMemEval-compatible checkout diagnostic plus the Harvey LAB
external memory-ablation report. Older BM25 slices, backend comparisons, and
suite/debug runs are archived development history, not current public claims.

The next retrieval-quality work should close the practical ergonomics gap with
QMD-style search sidecars: richer assembly lifecycle hooks, stronger local
embedding providers, and broader degraded-mode observability. These should
augment Zaxy's temporal/provenance layer rather than replace it with generic
chunk search.

Compaction remains identity-preserving. Consolidated vectors or summaries may
route queries and reduce token load, but they should not become the sole
authority for event, document, transcript, or graph facts. Purpose-aware
compaction projection is available with `zaxy compact --projection-output ...
--purpose <profile>`. Security, release, and review profiles preserve all
source-backed records because risks, gates, decisions, and mitigations should
not be collapsed into a single representative. Coding and research profiles use
bounded exemplar sets with purpose-specific record floors, so invariants,
failed attempts, tests, claims, sources, contradictions, and open questions
survive better than generic medoid collapse. Coordinate projections force
authoritative records, retain accepted/promoted parent state plus proof and
handoff events, and keep pending, rejected, deferred, stale, or unpromoted
worker rows out of searchable authoritative projection records while reporting
them in `consolidation_policy` diagnostics. Prompt assembly emits warnings when
compacted or projection-derived context lacks source-level citations, and when
replay truncation leaves no retrieved source support. `MemoryFabric(eventloom_path=...)` auto-discovers
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
