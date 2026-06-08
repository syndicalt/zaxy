# Live Retrieval Benchmark

- Generated: `2026-06-03T07:52:11Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `1f6bb0ad91d5e06cbca4e89d8cccb128e16eb5342656d249e3dbbe9334cfdaef`
- Events: `128`
- Queries: `6`
- Subjects: `6`
- Sessions: `21`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.333 | 0.655 |  | 1.000 | 0.333 | 0.667 | 1.000 | 1.000 | 6.71 | 6.85 | 6.86 | 12788 | 3197 |
| zaxy-checkout | 0.500 | 1.000 |  | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 301.94 | 403.36 | 426.74 | 52996 | 13244 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 6 | 0.333 | 4 |  | 1.000 | 0.333 | 0.667 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 6 | 0.500 | 3 |  | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | synthesis_miss | 4 |
| zaxy-checkout | synthesis_miss | 4 |
