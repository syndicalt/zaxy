# Live Retrieval Benchmark

- Generated: `2026-06-07T16:20:10Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `90fb2307195d7e16b963a2b8a30f03b375bd42a45d41aeaa55423029dd84e3fc`
- Events: `5372`
- Queries: `500`
- Subjects: `500`
- Sessions: `948`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.520 | 0.669 |  | 1.000 | 0.520 | 0.592 | 0.770 | 0.770 | 330.72 | 352.27 | 370.69 | 10644 | 2661 |
| zaxy-checkout | 0.956 | 0.980 |  | 1.000 | 0.910 | 0.960 | 1.000 | 1.000 | 881.01 | 1966.65 | 2495.07 | 40783 | 10192 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.679 | 25 |  | 1.000 | 0.679 | 0.756 | 0.872 | 0.872 |
| bm25 | longmemeval:multi-session | 133 | 0.429 | 76 |  | 1.000 | 0.429 | 0.511 | 0.752 | 0.752 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.233 |
| bm25 | longmemeval:single-session-user | 70 | 0.843 | 11 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.814 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.346 | 87 |  | 1.000 | 0.346 | 0.519 | 0.752 | 0.752 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.923 | 6 |  | 1.000 | 0.846 | 0.949 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.947 | 7 |  | 1.000 | 0.865 | 0.985 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.933 | 2 |  | 1.000 | 0.933 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.986 | 1 |  | 1.000 | 0.986 | 0.943 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.955 | 6 |  | 1.000 | 0.910 | 0.925 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 23 |
| bm25 | retrieval_miss | 92 |
| bm25 | synthesis_miss | 148 |
| zaxy-checkout | synthesis_miss | 45 |
