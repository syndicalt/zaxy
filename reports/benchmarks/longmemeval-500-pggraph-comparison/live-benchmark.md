# Live Retrieval Benchmark

- Generated: `2026-05-18T17:40:44Z`
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
| bm25 | 0.514 | 0.669 |  | 1.000 | 0.514 | 0.592 | 0.770 | 0.770 | 323.82 | 387.49 | 465.21 | 10644 | 2661 |
| zaxy | 0.690 | 0.880 |  | 1.000 | 0.690 | 0.926 | 0.926 | 0.926 | 359.10 | 713.31 | 2317.79 | 17137 | 4284 |
| zaxy-checkout | 0.612 | 0.880 |  | 1.000 | 0.604 | 0.916 | 0.926 | 0.926 | 330.56 | 669.86 | 2097.44 | 14544 | 3636 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| bm25 | longmemeval:knowledge-update | 78 | 0.679 | 25 |  | 1.000 | 0.679 | 0.756 | 0.872 | 0.872 |
| bm25 | longmemeval:multi-session | 133 | 0.429 | 76 |  | 1.000 | 0.429 | 0.511 | 0.752 | 0.752 |
| bm25 | longmemeval:single-session-assistant | 56 | 0.804 | 11 |  | 1.000 | 0.804 | 0.893 | 0.946 | 0.946 |
| bm25 | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.133 | 0.233 | 0.233 |
| bm25 | longmemeval:single-session-user | 70 | 0.843 | 11 |  | 1.000 | 0.843 | 0.657 | 0.814 | 0.814 |
| bm25 | longmemeval:temporal-reasoning | 133 | 0.323 | 90 |  | 1.000 | 0.323 | 0.519 | 0.752 | 0.752 |
| zaxy | longmemeval:knowledge-update | 78 | 0.782 | 17 |  | 1.000 | 0.782 | 0.962 | 0.962 | 0.962 |
| zaxy | longmemeval:multi-session | 133 | 0.662 | 45 |  | 1.000 | 0.662 | 0.962 | 0.962 | 0.962 |
| zaxy | longmemeval:single-session-assistant | 56 | 0.768 | 13 |  | 1.000 | 0.768 | 0.964 | 0.964 | 0.964 |
| zaxy | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.467 | 0.467 | 0.467 |
| zaxy | longmemeval:single-session-user | 70 | 0.914 | 6 |  | 1.000 | 0.914 | 0.957 | 0.957 | 0.957 |
| zaxy | longmemeval:temporal-reasoning | 133 | 0.669 | 44 |  | 1.000 | 0.669 | 0.940 | 0.940 | 0.940 |
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.603 | 31 |  | 1.000 | 0.577 | 0.936 | 0.962 | 0.962 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.571 | 57 |  | 1.000 | 0.564 | 0.962 | 0.962 | 0.962 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 0.696 | 17 |  | 1.000 | 0.696 | 0.964 | 0.964 | 0.964 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.000 | 30 |  |  | 0.000 | 0.467 | 0.467 | 0.467 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.900 | 7 |  | 1.000 | 0.900 | 0.957 | 0.957 | 0.957 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.609 | 52 |  | 1.000 | 0.602 | 0.917 | 0.940 | 0.940 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| bm25 | projection_miss | 22 |
| bm25 | retrieval_miss | 93 |
| bm25 | synthesis_miss | 150 |
| zaxy | retrieval_miss | 37 |
| zaxy | synthesis_miss | 118 |
| zaxy-checkout | retrieval_miss | 37 |
| zaxy-checkout | synthesis_miss | 161 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.1760 | [0.1400, 0.2140] | 0.0001 | yes |
| zaxy | zaxy-checkout | 0.0780 | [0.0540, 0.1040] | 0.0001 | yes |
