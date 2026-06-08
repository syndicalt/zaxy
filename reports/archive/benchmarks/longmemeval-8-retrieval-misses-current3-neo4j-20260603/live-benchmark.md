# Live Retrieval Benchmark

- Generated: `2026-06-03T06:12:02Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `9d6d34bda927f7b5a97c6c22248084d5c9cb4de93e7b1ef81db32241aaf62aaf`
- Events: `82`
- Queries: `8`
- Subjects: `8`
- Sessions: `14`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.375 | 0.708 |  | 1.000 | 0.375 | 0.125 | 0.875 | 0.875 | 4.95 | 5.01 | 5.02 | 35010 | 8753 |
| zaxy | 0.625 | 1.000 |  | 1.000 | 0.625 | 1.000 | 1.000 | 1.000 | 302.83 | 413.27 | 443.52 | 30666 | 7665 |
| zaxy-checkout | 0.625 | 1.000 |  | 1.000 | 0.375 | 1.000 | 1.000 | 1.000 | 309.58 | 426.23 | 441.55 | 55892 | 13969 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:multi-session | 1 | 0.000 | 1 |  |  | 0.000 | 0.000 | 1.000 | 1.000 |
| bm25 | longmemeval:single-session-user | 2 | 0.000 | 2 |  |  | 0.000 | 0.500 | 0.500 | 0.500 |
| bm25 | longmemeval:temporal-reasoning | 5 | 0.600 | 2 |  | 1.000 | 0.600 | 0.000 | 1.000 | 1.000 |
| zaxy | longmemeval:multi-session | 1 | 0.000 | 1 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:single-session-user | 2 | 0.000 | 2 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:temporal-reasoning | 5 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 1 | 0.000 | 1 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 2 | 0.000 | 2 |  |  | 0.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 5 | 1.000 | 0 |  | 1.000 | 0.600 | 1.000 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | retrieval_miss | 1 |
| bm25 | synthesis_miss | 4 |
| zaxy | synthesis_miss | 3 |
| zaxy-checkout | synthesis_miss | 5 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.2500 | [0.0000, 0.5000] | 0.5000 | no |
| zaxy | zaxy-checkout | 0.0000 | [0.0000, 0.0000] | 1.0000 | no |
