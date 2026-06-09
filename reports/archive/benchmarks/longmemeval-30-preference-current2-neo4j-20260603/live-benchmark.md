# Live Retrieval Benchmark

- Generated: `2026-06-03T05:49:21Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `57f6c2178f76202e68b695aa9bb18be68f3f3170a8e42e33b90e872a6be05612`
- Events: `223`
- Queries: `30`
- Subjects: `30`
- Sessions: `30`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.000 | 0.700 |  |  | 0.000 | 0.433 | 0.700 | 0.700 | 12.04 | 12.60 | 12.82 | 8246 | 2062 |
| zaxy | 0.000 | 1.000 |  |  | 0.000 | 1.000 | 1.000 | 1.000 | 201.29 | 304.63 | 322.25 | 27014 | 6753 |
| zaxy-checkout | 0.967 | 1.000 |  | 1.000 | 0.967 | 1.000 | 1.000 | 1.000 | 200.75 | 314.72 | 340.62 | 44610 | 11148 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.433 | 0.700 | 0.700 |
| zaxy | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.967 | 1 |  | 1.000 | 0.967 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | retrieval_miss | 9 |
| bm25 | synthesis_miss | 21 |
| zaxy | synthesis_miss | 30 |
| zaxy-checkout | synthesis_miss | 1 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.0000 | [0.0000, 0.0000] | 1.0000 | no |
| zaxy | zaxy-checkout | -0.9667 | [-1.0000, -0.9000] | 0.0001 | no |
