# Live Retrieval Benchmark

- Generated: `2026-05-17T23:09:22Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `a4d229ecd831abec57ac2a7c365fcf5903b1815ab84c383183e0a39512afe829`
- Events: `5372`
- Queries: `500`
- Subjects: `500`
- Sessions: `948`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|----------|----------|----------|-----------|--------|--------|--------|----------------|---------------|
| bm25 | 0.560 | 0.715 |  | 1.000 | 0.516 | 0.592 | 0.770 | 0.802 | 289.00 | 348.99 | 422.83 | 21983 | 5496 |
| zaxy-checkout | 0.626 | 0.922 |  | 1.000 | 0.608 | 0.944 | 0.956 | 0.956 | 11055.69 | 14686.65 | 22359.76 | 32975 | 8242 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.705 | 23 |  | 1.000 | 0.679 | 0.756 | 0.872 | 0.885 |
| bm25 | longmemeval:multi-session | 133 | 0.519 | 64 |  | 1.000 | 0.436 | 0.511 | 0.752 | 0.804 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.367 |
| bm25 | longmemeval:single-session-user | 70 | 0.857 | 10 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.843 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.384 | 82 |  | 1.000 | 0.323 | 0.519 | 0.752 | 0.767 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.654 | 27 |  | 1.000 | 0.641 | 0.962 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.587 | 55 |  | 1.000 | 0.579 | 0.985 | 0.985 | 0.985 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.750 | 14 |  | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.733 | 0.733 | 0.733 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.886 | 0.957 | 0.957 | 0.957 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.587 | 55 |  | 1.000 | 0.549 | 0.910 | 0.932 | 0.932 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 19 |
| bm25 | ranking_miss | 16 |
| bm25 | retrieval_miss | 80 |
| bm25 | synthesis_miss | 149 |
| zaxy-checkout | retrieval_miss | 22 |
| zaxy-checkout | synthesis_miss | 174 |
