# Live Retrieval Benchmark

- Generated: `2026-06-03T07:14:03Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `6cd464ffca1f116dcc798817ae043160d081b5eb5be33036e25f39e7ccfe0f1d`
- Events: `22`
- Queries: `8`
- Subjects: `8`
- Sessions: `8`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.625 | 1.000 |  | 1.000 | 0.625 | 0.875 | 1.000 | 1.000 | 1.15 | 1.25 | 1.26 | 31348 | 7838 |
| zaxy-checkout | 0.750 | 1.000 |  | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 185.10 | 249.40 | 271.43 | 29290 | 7319 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:single-session-assistant | 8 | 0.625 | 3 |  | 1.000 | 0.625 | 0.875 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-assistant | 8 | 0.750 | 2 |  | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | synthesis_miss | 3 |
| zaxy-checkout | synthesis_miss | 8 |
