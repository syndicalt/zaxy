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
score, and final ranking score. The ranking pass uses maximum marginal
relevance so near-duplicate hits do not crowd out adjacent context. Traversal
hits get a small preservation bonus because graph-neighbor evidence is often
the difference between generic search and relational memory.

Every graph-backed context chunk should cite its originating Eventloom event
when provenance is available. Citations use the form
`eventloom://<session>/events/<seq>#<hash-prefix>`. They let callers show why a
fact exists, replay the surrounding session, and distinguish retrieved context
from unsupported generated text.

Temporal filtering is a first-class part of retrieval. Without a temporal
filter, graph search returns current facts. With an `as_of` filter, the graph
returns facts whose validity window contains that time. This is what lets agents
answer questions like "what did we believe before the rollback?" without losing
newer corrections.

The vector path depends on embeddings. Local deterministic embeddings are useful
for tests and offline development. Hosted embeddings are better for semantic
quality. Both feed the same vector index shape. See [embeddings.md](embeddings.md)
for provider configuration.

Fusion should remain conservative. Exact entity matches should not be drowned
out by vague semantic hits. Traversal should add connected evidence, not flood
the prompt. Limits are validated centrally, and traversal depth is bounded in
`src/zaxy/security.py` to avoid runaway graph expansion.

Benchmark coverage lives in `src/zaxy/benchmark.py`, `src/zaxy/live_benchmark.py`,
`tests/test_competitive_benchmarks.py`, and `tests/test_live_benchmark.py`. The
current live benchmark compares markdown, vector, markdown+vector, and Zaxy
retrieval on the same generated temporal event workload. Treat it as a workload-specific
signal, not a universal benchmark against production-grade vector RAG or file
memory systems.

The next retrieval-quality work should close the practical ergonomics gap with
QMD-style search sidecars: stronger reranking, query expansion,
document/file ingestion, transcript indexing, local embedding and reranking
providers, and graceful degradation when Neo4j, embeddings, or rerankers are
unavailable. These should augment Zaxy's temporal/provenance layer rather than
replace it with generic chunk search.

Related references: [graph-schema.md](graph-schema.md), [mcp.md](mcp.md),
[configuration.md](configuration.md), [testing.md](testing.md), and
[README.md](../README.md). The public explanation is [site/index.html](../site/index.html).
