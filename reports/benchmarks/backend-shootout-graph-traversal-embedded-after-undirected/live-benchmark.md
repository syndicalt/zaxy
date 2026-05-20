# Live Retrieval Benchmark

- Generated: `2026-05-20T03:29:01Z`
- Embedding provider: `hash:1536`
- Workload: `mempalace-graph-traversal-v1`
- Workload SHA-256: `03a01cf0300bebef46ff906798e23a4542b5b5a4d670a8e240505e744a402a54`
- Events: `40`
- Queries: `10`
- Subjects: `10`
- Lanes: `graph-traversal`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.000 | 1.000 |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 0.27 | 0.37 | 0.39 | 2074 | 519 |
| zaxy | 0.000 | 0.333 |  |  | 0.000 | 1.000 | 1.000 | 1.000 | 16.50 | 33.13 | 42.05 | 182 | 45 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | graph-traversal | 10 | 0.000 | 10 |  |  | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy | graph-traversal | 10 | 0.000 | 10 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | stale_or_forbidden_hit | 10 |
| zaxy | synthesis_miss | 10 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.0000 | [0.0000, 0.0000] | 1.0000 | no |
