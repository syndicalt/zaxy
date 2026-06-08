# Live Retrieval Benchmark

- Generated: `2026-06-03T07:08:18Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `dba6b48e2c3cee4a1acd19a3f8a073d58d23f48b658de5017214c18823e37319`
- Events: `113`
- Queries: `10`
- Subjects: `10`
- Sessions: `21`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.600 | 0.917 |  | 1.000 | 0.600 | 1.000 | 1.000 | 1.000 | 6.94 | 7.37 | 7.37 | 20822 | 5206 |
| zaxy-checkout | 0.900 | 1.000 |  | 1.000 | 0.100 | 0.900 | 1.000 | 1.000 | 273.02 | 439.09 | 465.13 | 41131 | 10280 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 2 | 0.500 | 1 |  | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| bm25 | longmemeval:single-session-assistant | 4 | 0.750 | 1 |  | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| bm25 | longmemeval:temporal-reasoning | 4 | 0.500 | 2 |  | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 2 | 1.000 | 0 |  | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-assistant | 4 | 1.000 | 0 |  | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 4 | 0.750 | 1 |  | 1.000 | 0.000 | 0.750 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | synthesis_miss | 4 |
| zaxy-checkout | synthesis_miss | 9 |
