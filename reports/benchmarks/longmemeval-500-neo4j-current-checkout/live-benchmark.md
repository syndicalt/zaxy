# Live Retrieval Benchmark

- Generated: `2026-05-18T19:16:51Z`
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
| bm25 | 0.514 | 0.669 |  | 1.000 | 0.514 | 0.592 | 0.770 | 0.770 | 285.06 | 329.24 | 385.19 | 10644 | 2661 |
| zaxy-checkout | 0.712 | 0.923 |  | 1.000 | 0.620 | 0.946 | 0.958 | 0.958 | 808.39 | 1076.40 | 2436.42 | 53547 | 13383 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.679 | 25 |  | 1.000 | 0.679 | 0.756 | 0.872 | 0.872 |
| bm25 | longmemeval:multi-session | 133 | 0.429 | 76 |  | 1.000 | 0.429 | 0.511 | 0.752 | 0.752 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.233 |
| bm25 | longmemeval:single-session-user | 70 | 0.843 | 11 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.814 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.323 | 90 |  | 1.000 | 0.323 | 0.519 | 0.752 | 0.752 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.808 | 15 |  | 1.000 | 0.615 | 0.949 | 0.987 | 0.987 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.677 | 43 |  | 1.000 | 0.564 | 0.970 | 0.970 | 0.970 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.839 | 9 |  | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.767 | 0.767 | 0.767 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.900 | 0.971 | 0.971 | 0.971 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.692 | 41 |  | 1.000 | 0.617 | 0.925 | 0.947 | 0.947 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 22 |
| bm25 | retrieval_miss | 93 |
| bm25 | synthesis_miss | 150 |
| zaxy-checkout | retrieval_miss | 21 |
| zaxy-checkout | synthesis_miss | 169 |
