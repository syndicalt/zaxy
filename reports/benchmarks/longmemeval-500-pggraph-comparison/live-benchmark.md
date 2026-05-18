# Live Retrieval Benchmark

- Generated: `2026-05-18T18:15:32Z`
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
| bm25 | 0.516 | 0.669 |  | 1.000 | 0.516 | 0.592 | 0.770 | 0.770 | 278.44 | 323.27 | 410.26 | 10644 | 2661 |
| zaxy | 0.696 | 0.910 |  | 1.000 | 0.696 | 0.950 | 0.950 | 0.950 | 797.23 | 1090.09 | 2523.17 | 16678 | 4168 |
| zaxy-checkout | 0.622 | 0.912 |  | 1.000 | 0.610 | 0.940 | 0.950 | 0.950 | 753.85 | 1037.51 | 2459.72 | 22724 | 5680 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.679 | 25 |  | 1.000 | 0.679 | 0.756 | 0.872 | 0.872 |
| bm25 | longmemeval:multi-session | 133 | 0.436 | 75 |  | 1.000 | 0.436 | 0.511 | 0.752 | 0.752 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.233 |
| bm25 | longmemeval:single-session-user | 70 | 0.843 | 11 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.814 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.323 | 90 |  | 1.000 | 0.323 | 0.519 | 0.752 | 0.752 |
| zaxy | longmemeval:knowledge-update | 78 | 0.795 | 16 |  | 1.000 | 0.795 | 0.974 | 0.974 | 0.974 |
| zaxy | longmemeval:multi-session | 133 | 0.684 | 42 |  | 1.000 | 0.684 | 0.970 | 0.970 | 0.970 |
| zaxy | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 1.000 | 1.000 | 1.000 |
| zaxy | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.667 | 0.667 | 0.667 |
| zaxy | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.957 | 0.957 | 0.957 |
| zaxy | longmemeval:temporal-reasoning | 133 | 0.647 | 47 |  | 1.000 | 0.647 | 0.955 | 0.955 | 0.955 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.679 | 25 |  | 1.000 | 0.628 | 0.949 | 0.974 | 0.974 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.579 | 56 |  | 1.000 | 0.571 | 0.970 | 0.970 | 0.970 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.750 | 14 |  | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.667 | 0.667 | 0.667 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.900 | 7 |  | 1.000 | 0.886 | 0.957 | 0.957 | 0.957 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.571 | 57 |  | 1.000 | 0.571 | 0.932 | 0.955 | 0.955 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 22 |
| bm25 | retrieval_miss | 93 |
| bm25 | synthesis_miss | 149 |
| zaxy | retrieval_miss | 25 |
| zaxy | synthesis_miss | 127 |
| zaxy-checkout | retrieval_miss | 25 |
| zaxy-checkout | synthesis_miss | 170 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.1800 | [0.1440, 0.2160] | 0.0001 | yes |
| zaxy | zaxy-checkout | 0.0740 | [0.0500, 0.1000] | 0.0001 | yes |
