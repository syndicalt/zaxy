# Live Retrieval Benchmark

- Generated: `2026-06-04T04:54:44Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `0b20950cd69be3b5a74cdd8ea859dade5d15bfe0232abace1a0123ed7c32f2ee`
- Events: `17`
- Queries: `1`
- Subjects: `1`
- Sessions: `3`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| zaxy-checkout | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11277.13 | 11277.13 | 11277.13 | 88476 | 22114 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:temporal-reasoning | 1 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
