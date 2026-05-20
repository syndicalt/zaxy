# Live Retrieval Benchmark

- Generated: `2026-05-20T03:31:25Z`
- Embedding provider: `hash:1536`
- Workload: `mempalace-graph-traversal-v1`
- Workload SHA-256: `03a01cf0300bebef46ff906798e23a4542b5b5a4d670a8e240505e744a402a54`
- Events: `40`
- Queries: `10`
- Subjects: `10`
- Lanes: `graph-traversal`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.000 | 1.000 |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 0.21 | 0.35 | 0.42 | 2074 | 519 |
| zaxy | 1.000 | 0.667 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 18.31 | 31.87 | 39.55 | 1014 | 252 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | graph-traversal | 10 | 0.000 | 10 |  |  | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy | graph-traversal | 10 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | stale_or_forbidden_hit | 10 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 1.0000 | [1.0000, 1.0000] | 0.0020 | yes |
