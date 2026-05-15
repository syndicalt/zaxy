# Live Retrieval Benchmark

- Generated: `2026-05-15T13:05:54Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `7bfb3ce4f3d5b87d1fc8045bb84e40b3195eb13ed711950365efbb721b9de10e`
- Events: `1559`
- Queries: `100`
- Subjects: `100`
- Sessions: `265`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| zaxy | 0.950 | 0.949 |  | 1.000 | 0.950 | 0.990 | 0.990 | 0.990 | 1415.99 | 2007.56 | 3171.29 | 15808 | 3951 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy | longmemeval:multi-session | 40 | 0.975 | 1 |  | 1.000 | 0.975 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:temporal-reasoning | 60 | 0.933 | 4 |  | 1.000 | 0.933 | 0.983 | 0.983 | 0.983 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy | retrieval_miss | 1 |
| zaxy | synthesis_miss | 4 |
