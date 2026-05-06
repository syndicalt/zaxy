# Retrieval

Zaxy retrieval is hybrid by design. Agent memory queries are rarely solved by a
single strategy. Exact lookup is excellent when a query names a known entity.
Full-text search is useful for names and summaries. Vector similarity helps
when the phrasing changes. Graph traversal brings in connected facts that a flat
retriever would miss.

The query router lives in `src/zaxy/query.py`. It validates the query and limit,
calls `GraphStore.search_exact`, `search_keyword`, and `search_vector` where
appropriate, expands from high-confidence hits through traversal, fuses scores,
and returns `ContextChunk` objects. A context chunk contains the content an agent
should see and metadata about source entities, scores, and provenance.

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

Benchmark coverage lives in `src/zaxy/benchmark.py` and
`tests/test_competitive_benchmarks.py`. The current baseline compares Zaxy's
structured retrieval harness with flat JSONL context. Future external adapters
can compare vector RAG, hybrid RAG, file memory, and framework-native memories.

Related references: [graph-schema.md](graph-schema.md), [mcp.md](mcp.md),
[configuration.md](configuration.md), [testing.md](testing.md), and
[README.md](../README.md). The public explanation is [site/index.html](../site/index.html).
