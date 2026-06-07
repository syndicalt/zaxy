# Live Retrieval Benchmark

- Generated: `2026-05-15T14:06:14Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `0a0f41eb22b77ec8b07806c6745c62343ea1323bf77b001606605aa1f99d140b`
- Events: `1559`
- Queries: `100`
- Subjects: `100`
- Sessions: `265`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.500 | 0.638 |  | 1.000 | 0.500 | 0.710 | 0.840 | 0.840 | 79.39 | 84.66 | 90.77 | 10054 | 2514 |
| zaxy-checkout | 0.900 | 0.952 |  | 1.000 | 0.880 | 0.950 | 0.990 | 0.990 | 1533.78 | 2123.07 | 3207.61 | 29681 | 7418 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 40 | 0.525 | 19 |  | 1.000 | 0.525 | 0.675 | 0.775 | 0.775 |
| bm25 | longmemeval:temporal-reasoning | 60 | 0.483 | 31 |  | 1.000 | 0.483 | 0.733 | 0.883 | 0.883 |
| zaxy-checkout | longmemeval:multi-session | 40 | 0.875 | 5 |  | 1.000 | 0.875 | 0.975 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 60 | 0.917 | 5 |  | 1.000 | 0.883 | 0.933 | 0.983 | 0.983 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 7 |
| bm25 | retrieval_miss | 9 |
| bm25 | synthesis_miss | 41 |
| zaxy-checkout | retrieval_miss | 1 |
| zaxy-checkout | synthesis_miss | 11 |
