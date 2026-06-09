# Live Retrieval Benchmark

- Generated: `2026-06-03T08:14:54Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `f8fb604ccdb6138a821156d70429d705a3dcc5d8265fde3cef993ffab577082e`
- Events: `32`
- Queries: `2`
- Subjects: `2`
- Sessions: `6`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| zaxy-checkout | 1.000 | 1.000 |  | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 355.97 | 462.39 | 471.85 | 100876 | 25217 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:multi-session | 2 | 1.000 | 0 |  | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | synthesis_miss | 1 |
