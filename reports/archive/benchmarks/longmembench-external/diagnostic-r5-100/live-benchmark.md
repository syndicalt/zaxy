# Live Retrieval Benchmark

- Generated: `2026-06-07T07:21:15Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `fcd341be409fdb099166cbc37ef203a03609352299c6c9b2e4438fd38e920d63`
- Events: `1559`
- Queries: `100`
- Subjects: `100`
- Sessions: `265`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| zaxy-checkout | 0.990 | 0.982 |  | 1.000 | 0.960 | 0.650 | 1.000 | 1.000 | 255.73 | 662.69 | 1029.37 | 93791 | 23441 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:multi-session | 40 | 0.975 | 1 |  | 1.000 | 0.925 | 0.950 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 60 | 1.000 | 0 |  | 1.000 | 0.983 | 0.450 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | synthesis_miss | 4 |
