# Live Retrieval Benchmark

- Generated: `2026-06-03T05:07:56Z`
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
| bm25 | 0.490 | 0.638 |  | 1.000 | 0.490 | 0.710 | 0.840 | 0.840 | 118.25 | 136.60 | 151.40 | 10054 | 2514 |
| zaxy | 1.000 | 0.954 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 558.94 | 737.41 | 876.08 | 21788 | 5445 |
| zaxy-checkout | 1.000 | 0.954 |  | 1.000 | 0.950 | 0.910 | 1.000 | 1.000 | 554.27 | 792.80 | 843.48 | 44947 | 11233 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 40 | 0.500 | 20 |  | 1.000 | 0.500 | 0.675 | 0.775 | 0.775 |
| bm25 | longmemeval:temporal-reasoning | 60 | 0.483 | 31 |  | 1.000 | 0.483 | 0.733 | 0.883 | 0.883 |
| zaxy | longmemeval:multi-session | 40 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:temporal-reasoning | 60 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 40 | 1.000 | 0 |  | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 60 | 1.000 | 0 |  | 1.000 | 0.983 | 0.850 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 7 |
| bm25 | retrieval_miss | 9 |
| bm25 | synthesis_miss | 42 |
| zaxy-checkout | synthesis_miss | 5 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.5100 | [0.4100, 0.6100] | 0.0001 | yes |
| zaxy | zaxy-checkout | 0.0000 | [0.0000, 0.0000] | 1.0000 | no |
