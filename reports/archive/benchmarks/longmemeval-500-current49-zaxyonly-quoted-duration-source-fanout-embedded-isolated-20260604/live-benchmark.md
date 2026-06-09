# Live Retrieval Benchmark

- Generated: `2026-06-04T03:18:30Z`
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
| zaxy-checkout | 0.898 | 0.950 |  | 1.000 | 0.844 | 0.896 | 0.990 | 0.990 | 483.96 | 855.28 | 1000.05 | 34975 | 8741 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.885 | 9 |  | 1.000 | 0.795 | 0.923 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.812 | 25 |  | 1.000 | 0.744 | 0.955 | 0.985 | 0.985 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 1.000 | 0 |  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.900 | 3 |  | 1.000 | 0.900 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.957 | 3 |  | 1.000 | 0.929 | 0.900 | 0.971 | 0.971 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.917 | 11 |  | 1.000 | 0.850 | 0.752 | 0.993 | 0.993 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | retrieval_miss | 5 |
| zaxy-checkout | synthesis_miss | 73 |
