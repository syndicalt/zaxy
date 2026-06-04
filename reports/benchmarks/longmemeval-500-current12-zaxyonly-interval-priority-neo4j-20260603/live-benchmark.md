# Live Retrieval Benchmark

- Generated: `2026-06-03T09:29:18Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `0dc36a139bb9a4fdc7c6cd34400737a58a1eb7410517341f015e9fbfc76ed854`
- Events: `5372`
- Queries: `500`
- Subjects: `500`
- Sessions: `948`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| zaxy-checkout | 0.808 | 0.948 |  | 1.000 | 0.716 | 0.890 | 0.992 | 0.992 | 613.55 | 868.83 | 1003.72 | 33226 | 8304 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.808 | 15 |  | 1.000 | 0.615 | 0.923 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.707 | 39 |  | 1.000 | 0.579 | 0.947 | 0.993 | 0.993 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.911 | 5 |  | 1.000 | 0.839 | 0.982 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.900 | 3 |  | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.914 | 0.971 | 0.971 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.789 | 28 |  | 1.000 | 0.714 | 0.737 | 0.993 | 0.993 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | retrieval_miss | 4 |
| zaxy-checkout | synthesis_miss | 138 |
