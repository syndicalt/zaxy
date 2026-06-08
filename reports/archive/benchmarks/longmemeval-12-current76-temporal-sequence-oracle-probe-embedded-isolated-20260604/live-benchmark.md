# Live Retrieval Benchmark

- Generated: `2026-06-04T10:59:13Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `c5949320d3b98975a871ce2d8ed01b0514a17e55ddba0205bbfe145bde2ca0e0`
- Events: `155`
- Queries: `12`
- Subjects: `12`
- Sessions: `28`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| zaxy-checkout | 0.667 | 1.000 |  | 1.000 | 0.167 | 1.000 | 1.000 | 1.000 | 131.49 | 273.15 | 364.13 | 91663 | 22910 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:temporal-reasoning | 12 | 0.667 | 4 |  | 1.000 | 0.167 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | synthesis_miss | 10 |
