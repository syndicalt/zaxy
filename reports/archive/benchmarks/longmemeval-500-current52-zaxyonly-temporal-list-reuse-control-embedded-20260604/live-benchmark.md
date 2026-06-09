# Live Retrieval Benchmark

- Generated: `2026-06-04T05:00:22Z`
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
| zaxy-checkout | 0.908 | 0.964 |  | 1.000 | 0.834 | 0.900 | 0.998 | 0.998 | 413.21 | 735.98 | 1936.07 | 54375 | 13589 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Answer@5 | Recall@1 | Recall@5 | Recall@10 |
|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|----------|-----------|
| zaxy-checkout | longmemeval:knowledge-update | 78 | 0.897 | 8 |  | 1.000 | 0.782 | 0.923 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:multi-session | 133 | 0.842 | 21 |  | 1.000 | 0.744 | 0.970 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-assistant | 56 | 1.000 | 0 |  | 1.000 | 0.929 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-preference | 30 | 0.867 | 4 |  | 1.000 | 0.867 | 1.000 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:single-session-user | 70 | 0.957 | 3 |  | 1.000 | 0.914 | 0.929 | 1.000 | 1.000 |
| zaxy-checkout | longmemeval:temporal-reasoning | 133 | 0.925 | 10 |  | 1.000 | 0.865 | 0.737 | 0.993 | 0.993 |

## Miss taxonomy

| Backend | Miss category | Count |
|---------|---------------|-------|
| zaxy-checkout | retrieval_miss | 1 |
| zaxy-checkout | synthesis_miss | 82 |
