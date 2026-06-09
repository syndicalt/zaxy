# Live Retrieval Benchmark

- Generated: `2026-06-03T04:26:11Z`
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
| bm25 | 0.524 | 0.669 |  | 1.000 | 0.524 | 0.592 | 0.770 | 0.770 | 304.41 | 423.46 | 433.60 | 10644 | 2661 |
| zaxy | 0.710 | 0.930 |  | 1.000 | 0.710 | 0.972 | 0.972 | 0.972 | 841.36 | 1186.05 | 1517.94 | 17916 | 4478 |
| zaxy-checkout | 0.732 | 0.931 |  | 1.000 | 0.646 | 0.962 | 0.972 | 0.972 | 847.52 | 1184.10 | 1301.00 | 31279 | 7817 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.679 | 25 |  | 1.000 | 0.679 | 0.756 | 0.872 | 0.872 |
| bm25 | longmemeval:multi-session | 133 | 0.444 | 74 |  | 1.000 | 0.444 | 0.511 | 0.752 | 0.752 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.233 |
| bm25 | longmemeval:single-session-user | 70 | 0.843 | 11 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.814 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.346 | 87 |  | 1.000 | 0.346 | 0.519 | 0.752 | 0.752 |
| zaxy | longmemeval:knowledge-update | 78 | 0.782 | 17 |  | 1.000 | 0.782 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:multi-session | 133 | 0.669 | 44 |  | 1.000 | 0.669 | 0.985 | 0.985 | 0.985 |
| zaxy | longmemeval:single-session-assistant | 56 | 0.821 | 10 |  | 1.000 | 0.821 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.833 | 0.833 | 0.833 |
| zaxy | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.971 | 0.971 | 0.971 |
| zaxy | longmemeval:temporal-reasoning | 133 | 0.714 | 38 |  | 1.000 | 0.714 | 0.962 | 0.962 | 0.962 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.808 | 15 |  | 1.000 | 0.628 | 0.974 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.677 | 43 |  | 1.000 | 0.571 | 0.985 | 0.985 | 0.985 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.839 | 9 |  | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.833 | 0.833 | 0.833 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.971 | 0.971 | 0.971 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.767 | 31 |  | 1.000 | 0.692 | 0.940 | 0.962 | 0.962 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 23 |
| bm25 | retrieval_miss | 92 |
| bm25 | synthesis_miss | 146 |
| zaxy | retrieval_miss | 14 |
| zaxy | synthesis_miss | 131 |
| zaxy-checkout | retrieval_miss | 14 |
| zaxy-checkout | synthesis_miss | 163 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.1860 | [0.1500, 0.2240] | 0.0001 | yes |
| zaxy | zaxy-checkout | -0.0220 | [-0.0400, -0.0060] | 0.0196 | no |
