# Live Retrieval Benchmark

- Generated: `2026-06-03T05:39:35Z`
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
| bm25 | 0.518 | 0.669 |  | 1.000 | 0.518 | 0.592 | 0.770 | 0.770 | 416.61 | 453.62 | 572.25 | 10644 | 2661 |
| zaxy | 0.720 | 0.937 |  | 1.000 | 0.720 | 0.980 | 0.980 | 0.980 | 860.94 | 1226.33 | 1544.83 | 17997 | 4498 |
| zaxy-checkout | 0.740 | 0.938 |  | 1.000 | 0.652 | 0.944 | 0.980 | 0.980 | 870.11 | 1238.75 | 1395.44 | 31459 | 7862 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.667 | 26 |  | 1.000 | 0.667 | 0.756 | 0.872 | 0.872 |
| bm25 | longmemeval:multi-session | 133 | 0.429 | 76 |  | 1.000 | 0.429 | 0.511 | 0.752 | 0.752 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.233 |
| bm25 | longmemeval:single-session-user | 70 | 0.843 | 11 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.814 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.346 | 87 |  | 1.000 | 0.346 | 0.519 | 0.752 | 0.752 |
| zaxy | longmemeval:knowledge-update | 78 | 0.782 | 17 |  | 1.000 | 0.782 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:multi-session | 133 | 0.707 | 39 |  | 1.000 | 0.707 | 0.985 | 0.985 | 0.985 |
| zaxy | longmemeval:single-session-assistant | 56 | 0.821 | 10 |  | 1.000 | 0.821 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.933 | 0.933 | 0.933 |
| zaxy | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.971 | 0.971 | 0.971 |
| zaxy | longmemeval:temporal-reasoning | 133 | 0.714 | 38 |  | 1.000 | 0.714 | 0.970 | 0.970 | 0.970 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.808 | 15 |  | 1.000 | 0.628 | 0.962 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.707 | 39 |  | 1.000 | 0.602 | 0.962 | 0.985 | 0.985 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.839 | 9 |  | 1.000 | 0.732 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.933 | 0.933 | 0.933 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.943 | 0.971 | 0.971 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.767 | 31 |  | 1.000 | 0.692 | 0.895 | 0.970 | 0.970 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 23 |
| bm25 | retrieval_miss | 92 |
| bm25 | synthesis_miss | 149 |
| zaxy | retrieval_miss | 10 |
| zaxy | synthesis_miss | 130 |
| zaxy-checkout | retrieval_miss | 10 |
| zaxy-checkout | synthesis_miss | 164 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.2020 | [0.1660, 0.2380] | 0.0001 | yes |
| zaxy | zaxy-checkout | -0.0200 | [-0.0360, -0.0040] | 0.0203 | no |
