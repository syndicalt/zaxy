# Live Retrieval Benchmark

- Generated: `2026-06-03T05:49:39Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `e0d96f7e6eed85a3247e45d8e5cf0313a4261098d3d95a9d69d178d4cc0e4469`
- Events: `98`
- Queries: `10`
- Subjects: `10`
- Sessions: `16`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.200 | 0.450 |  | 1.000 | 0.200 | 0.400 | 0.600 | 0.600 | 5.81 | 6.07 | 6.15 | 26704 | 6676 |
| zaxy | 0.400 | 1.000 |  | 1.000 | 0.400 | 1.000 | 1.000 | 1.000 | 234.84 | 392.20 | 445.94 | 30067 | 7516 |
| zaxy-checkout | 0.700 | 1.000 |  | 1.000 | 0.600 | 1.000 | 1.000 | 1.000 | 260.17 | 354.77 | 368.34 | 47172 | 11790 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 2 | 0.000 | 2 |  |  | 0.000 | 0.500 | 0.500 | 0.500 |
| bm25 | longmemeval:single-session-preference | 2 | 0.000 | 2 |  |  | 0.000 | 0.500 | 0.500 | 0.500 |
| bm25 | longmemeval:single-session-user | 2 | 0.000 | 2 |  |  | 0.000 | 0.500 | 0.500 | 0.500 |
| bm25 | longmemeval:temporal-reasoning | 4 | 0.500 | 2 |  | 1.000 | 0.500 | 0.250 | 0.750 | 0.750 |
| zaxy | longmemeval:multi-session | 2 | 0.000 | 2 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:single-session-preference | 2 | 0.000 | 2 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:single-session-user | 2 | 0.000 | 2 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:temporal-reasoning | 4 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 2 | 0.500 | 1 |  | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 2 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 2 | 0.000 | 2 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 4 | 1.000 | 0 |  | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | retrieval_miss | 4 |
| bm25 | synthesis_miss | 4 |
| zaxy | synthesis_miss | 6 |
| zaxy-checkout | synthesis_miss | 4 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.2000 | [0.0000, 0.5000] | 0.5000 | no |
| zaxy | zaxy-checkout | -0.3000 | [-0.6000, 0.0000] | 0.2500 | no |
