# Live Retrieval Benchmark

- Generated: `2026-06-04T00:12:57Z`
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
| zaxy-checkout | 0.876 | 0.950 |  | 1.000 | 0.810 | 0.896 | 0.990 | 0.990 | 323.47 | 568.56 | 698.06 | 33971 | 8490 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.885 | 9 |  | 1.000 | 0.821 | 0.936 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.767 | 31 |  | 1.000 | 0.632 | 0.955 | 0.985 | 0.985 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.900 | 3 |  | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.957 | 3 |  | 1.000 | 0.943 | 0.900 | 0.971 | 0.971 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.880 | 16 |  | 1.000 | 0.812 | 0.744 | 0.993 | 0.993 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | retrieval_miss | 5 |
| zaxy-checkout | synthesis_miss | 90 |
