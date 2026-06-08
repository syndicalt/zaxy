# Live Retrieval Benchmark

- Generated: `2026-06-03T06:38:55Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `a438d482a6ad0913de691b8194dd5d98f7ffe277016a44f6089596ade7620026`
- Events: `74`
- Queries: `3`
- Subjects: `3`
- Sessions: `11`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.000 | 0.644 |  |  | 0.000 | 0.333 | 1.000 | 1.000 | 4.19 | 4.20 | 4.21 | 18319 | 4580 |
| zaxy-checkout | 1.000 | 1.000 |  | 1.000 | 0.667 | 1.000 | 1.000 | 1.000 | 335.83 | 390.82 | 395.71 | 67347 | 16833 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 3 | 0.000 | 3 |  |  | 0.000 | 0.333 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 3 | 1.000 | 0 |  | 1.000 | 0.667 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | synthesis_miss | 3 |
| zaxy-checkout | synthesis_miss | 1 |
