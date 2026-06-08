# Backend Shootout

- Report schema version: `1`
- Harness: `zaxy-backend-shootout`
- Generated at UTC: `2026-05-20T13:20:19Z`
- Eventloom path: `reports/backend-shootout/longmemeval-40.eventloom.jsonl`
- Queries file: `reports/backend-shootout/longmemeval-40-queries-with-targets.json`
- Session ID: `default`
- Queries: `40`
- Events: `520`
- Limit: `5`
- Source Eventloom SHA-256: `2870768feac6aa5378edd22f4bc9480f24642c37fea785417589b020a516adb6`
- Source queries SHA-256: `8479ff4b7152f5d988ba70e8bb9c1c0f5e721f5a62bd40cede26db3298dfcad5`
- Workload events SHA-256: `146d2d262e951a711341aefefed13ac001b20d5c0b8d948801dd9a8bdd9583ea`
- Workload queries SHA-256: `e1df9a299c93d3168a0918e6d248db33ffae56e0c69983b9cd3447fa7eaa1dd4`

| Backend | Status | Cold bootstrap ms | First useful init ms | First checkout ms | Append projection p95 ms | Projection eps | Checkout p95 ms | Checkout p99 ms | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected | Citation coverage | Mean quality | Answer@5 | Recall@5 | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |
|---------|--------|-------------------|----------------------|-------------------|--------------------------|----------------|-----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|-------------------|--------------|----------|----------|--------------|-----------------------------|-------------------------|---------------------|
| embedded | ok | 97.941 | 10885.341 | 43.986 | 30.417 | 48.205 | 56.119 | 61.588 | 3.655 | 4.385 | 6.435 | 14.696 | 18.415 | 21.055 | 3.255 | 3.948 | 4.215 | 0.004 | 0.005 | 0.006 | 132.32 | embedded | 100 | 100 | 1514.575 | 0.1651 | 0.1651 | 1617.05 | 0.1546 | 0.1546 | 1.0 | 0.25 | 0.25 | 0.975 | 28762112 | 675287040 | 28762112 | 10737.323 |
| bm25 | ok | 5.162 | 5.162 | 153.592 | - | - | 167.846 | 227.391 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 3944.9 | 0.0634 | 0.0634 | 3944.9 | 0.0634 | 0.0634 | 1.0 | 0.25 | 0.25 | 1.0 | 1598781 | 28672 | 0 | 0.0 |
