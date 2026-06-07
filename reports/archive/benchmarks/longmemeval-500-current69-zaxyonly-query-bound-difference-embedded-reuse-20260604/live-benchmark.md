# Live Retrieval Benchmark

- Generated: `2026-06-04T09:46:39Z`
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
| zaxy-checkout | 0.932 | 0.968 |  | 1.000 | 0.886 | 0.906 | 1.000 | 1.000 | 356.41 | 700.82 | 968.67 | 56005 | 13996 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.936 | 5 |  | 1.000 | 0.923 | 0.936 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.880 | 16 |  | 1.000 | 0.797 | 0.970 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 1.000 | 0 |  | 1.000 | 0.946 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.900 | 3 |  | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.971 | 2 |  | 1.000 | 0.971 | 0.929 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.940 | 8 |  | 1.000 | 0.880 | 0.752 | 1.000 | 1.000 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | synthesis_miss | 57 |
