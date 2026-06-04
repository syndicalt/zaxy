# Live Retrieval Benchmark

- Generated: `2026-06-03T07:41:41Z`
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
| bm25 | 0.520 | 0.669 |  | 1.000 | 0.520 | 0.592 | 0.770 | 0.770 | 307.54 | 327.24 | 336.56 | 10644 | 2661 |
| zaxy-checkout | 0.802 | 0.947 |  | 1.000 | 0.716 | 0.892 | 0.992 | 0.992 | 616.19 | 869.71 | 1015.19 | 33099 | 8272 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.679 | 25 |  | 1.000 | 0.679 | 0.756 | 0.872 | 0.872 |
| bm25 | longmemeval:multi-session | 133 | 0.429 | 76 |  | 1.000 | 0.429 | 0.511 | 0.752 | 0.752 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.233 |
| bm25 | longmemeval:single-session-user | 70 | 0.843 | 11 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.814 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.346 | 87 |  | 1.000 | 0.346 | 0.519 | 0.752 | 0.752 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.808 | 15 |  | 1.000 | 0.603 | 0.923 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.692 | 41 |  | 1.000 | 0.594 | 0.955 | 0.993 | 0.993 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.911 | 5 |  | 1.000 | 0.839 | 0.982 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.900 | 3 |  | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.914 | 0.971 | 0.971 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.782 | 29 |  | 1.000 | 0.707 | 0.737 | 0.993 | 0.993 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 23 |
| bm25 | retrieval_miss | 92 |
| bm25 | synthesis_miss | 148 |
| zaxy-checkout | retrieval_miss | 4 |
| zaxy-checkout | synthesis_miss | 138 |
