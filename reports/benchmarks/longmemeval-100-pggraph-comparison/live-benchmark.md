# Live Retrieval Benchmark

- Generated: `2026-05-18T17:24:36Z`
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
| bm25 | 0.500 | 0.638 |  | 1.000 | 0.500 | 0.710 | 0.840 | 0.840 | 92.22 | 98.24 | 101.13 | 10054 | 2514 |
| zaxy | 0.960 | 0.926 |  | 1.000 | 0.960 | 0.980 | 0.980 | 0.980 | 137.90 | 355.37 | 1452.80 | 23156 | 5789 |
| zaxy-checkout | 0.910 | 0.919 |  | 1.000 | 0.910 | 0.950 | 0.980 | 0.980 | 145.03 | 312.62 | 1455.09 | 20131 | 5033 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 40 | 0.525 | 19 |  | 1.000 | 0.525 | 0.675 | 0.775 | 0.775 |
| bm25 | longmemeval:temporal-reasoning | 60 | 0.483 | 31 |  | 1.000 | 0.483 | 0.733 | 0.883 | 0.883 |
| zaxy | longmemeval:multi-session | 40 | 0.975 | 1 |  | 1.000 | 0.975 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:temporal-reasoning | 60 | 0.950 | 3 |  | 1.000 | 0.950 | 0.967 | 0.967 | 0.967 |
| zaxy-checkout | longmemeval:multi-session | 40 | 0.875 | 5 |  | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 60 | 0.933 | 4 |  | 1.000 | 0.933 | 0.917 | 0.967 | 0.967 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 7 |
| bm25 | retrieval_miss | 9 |
| bm25 | synthesis_miss | 41 |
| zaxy | retrieval_miss | 2 |
| zaxy | synthesis_miss | 2 |
| zaxy-checkout | retrieval_miss | 2 |
| zaxy-checkout | synthesis_miss | 7 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.4600 | [0.3600, 0.5600] | 0.0001 | yes |
| zaxy | zaxy-checkout | 0.0500 | [0.0100, 0.1000] | 0.0652 | no |
