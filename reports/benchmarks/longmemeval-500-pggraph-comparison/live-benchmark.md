# Live Retrieval Benchmark

- Generated: `2026-05-18T20:10:35Z`
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
| bm25 | 0.512 | 0.669 |  | 1.000 | 0.512 | 0.592 | 0.770 | 0.770 | 304.19 | 343.80 | 432.63 | 10644 | 2661 |
| zaxy | 0.698 | 0.917 |  | 1.000 | 0.698 | 0.958 | 0.958 | 0.958 | 735.35 | 1077.11 | 2760.11 | 16776 | 4193 |
| zaxy-checkout | 0.714 | 0.919 |  | 1.000 | 0.632 | 0.948 | 0.958 | 0.958 | 732.97 | 1020.22 | 2457.36 | 52077 | 13016 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.667 | 26 |  | 1.000 | 0.667 | 0.756 | 0.872 | 0.872 |
| bm25 | longmemeval:multi-session | 133 | 0.429 | 76 |  | 1.000 | 0.429 | 0.511 | 0.752 | 0.752 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.233 |
| bm25 | longmemeval:single-session-user | 70 | 0.843 | 11 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.814 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.323 | 90 |  | 1.000 | 0.323 | 0.519 | 0.752 | 0.752 |
| zaxy | longmemeval:knowledge-update | 78 | 0.795 | 16 |  | 1.000 | 0.795 | 0.987 | 0.987 | 0.987 |
| zaxy | longmemeval:multi-session | 133 | 0.669 | 44 |  | 1.000 | 0.669 | 0.970 | 0.970 | 0.970 |
| zaxy | longmemeval:single-session-assistant | 56 | 0.821 | 10 |  | 1.000 | 0.821 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.767 | 0.767 | 0.767 |
| zaxy | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.957 | 0.957 | 0.957 |
| zaxy | longmemeval:temporal-reasoning | 133 | 0.662 | 45 |  | 1.000 | 0.662 | 0.955 | 0.955 | 0.955 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.808 | 15 |  | 1.000 | 0.628 | 0.962 | 0.987 | 0.987 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.699 | 40 |  | 1.000 | 0.587 | 0.970 | 0.970 | 0.970 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.839 | 9 |  | 1.000 | 0.786 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.767 | 0.767 | 0.767 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.957 | 0.957 | 0.957 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.677 | 43 |  | 1.000 | 0.609 | 0.932 | 0.955 | 0.955 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 22 |
| bm25 | retrieval_miss | 93 |
| bm25 | synthesis_miss | 151 |
| zaxy | retrieval_miss | 21 |
| zaxy | synthesis_miss | 130 |
| zaxy-checkout | retrieval_miss | 21 |
| zaxy-checkout | synthesis_miss | 163 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.1860 | [0.1500, 0.2220] | 0.0001 | yes |
| zaxy | zaxy-checkout | -0.0160 | [-0.0280, -0.0060] | 0.0075 | no |
