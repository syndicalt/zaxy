# Live Retrieval Benchmark

- Generated: `2026-05-18T16:48:39Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `84a401a669e62f5de0ef7fae124258e7cdf58ca9bfb42b5bd6ad8e85c3a858f9`
- Events: `1559`
- Queries: `100`
- Subjects: `100`
- Sessions: `265`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.540 | 0.721 |  | 1.000 | 0.500 | 0.710 | 0.840 | 0.870 | 79.96 | 85.77 | 90.47 | 21971 | 5493 |
| zaxy | 0.970 | 0.971 |  | 1.000 | 0.950 | 1.000 | 1.000 | 1.000 | 616.02 | 816.71 | 2449.74 | 44164 | 11038 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 40 | 0.550 | 18 |  | 1.000 | 0.525 | 0.675 | 0.775 | 0.775 |
| bm25 | longmemeval:temporal-reasoning | 60 | 0.533 | 28 |  | 1.000 | 0.483 | 0.733 | 0.883 | 0.933 |
| zaxy | longmemeval:multi-session | 40 | 0.975 | 1 |  | 1.000 | 0.975 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:temporal-reasoning | 60 | 0.967 | 2 |  | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 6 |
| bm25 | ranking_miss | 3 |
| bm25 | retrieval_miss | 7 |
| bm25 | synthesis_miss | 41 |
| zaxy | synthesis_miss | 5 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.4300 | [0.3300, 0.5300] | 0.0001 | yes |
