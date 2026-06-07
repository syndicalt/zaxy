# Live Retrieval Benchmark

- Generated: `2026-06-04T07:50:00Z`
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
| zaxy-checkout | 0.922 | 0.968 |  | 1.000 | 0.862 | 0.904 | 1.000 | 1.000 | 355.33 | 708.99 | 979.64 | 55948 | 13982 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.897 | 8 |  | 1.000 | 0.833 | 0.923 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.872 | 17 |  | 1.000 | 0.782 | 0.970 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 1.000 | 0 |  | 1.000 | 0.946 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.900 | 3 |  | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.971 | 2 |  | 1.000 | 0.957 | 0.929 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.932 | 9 |  | 1.000 | 0.865 | 0.752 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | synthesis_miss | 69 |
