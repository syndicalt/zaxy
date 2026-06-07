# Live Retrieval Benchmark

- Generated: `2026-05-20T03:20:27Z`
- Embedding provider: `hash:1536`
- Workload: `mempalace-graph-traversal-v1`
- Workload SHA-256: `7561582cf2f3a8fde2f1b5b65d9aab367d61b166bc845695886abd7dc4c422bd`
- Events: `12`
- Queries: `3`
- Subjects: `3`
- Lanes: `graph-traversal`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.000 | 0.667 |  |  | 1.000 | 1.000 | 1.000 | 1.000 | 0.16 | 0.28 | 0.29 | 2104 | 526 |
| zaxy | 0.000 | 0.333 |  |  | 0.000 | 1.000 | 1.000 | 1.000 | 6425.34 | 6443.94 | 6445.59 | 181 | 45 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | graph-traversal | 3 | 0.000 | 3 |  |  | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy | graph-traversal | 3 | 0.000 | 3 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | stale_or_forbidden_hit | 3 |
| zaxy | synthesis_miss | 3 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.0000 | [0.0000, 0.0000] | 1.0000 | no |
