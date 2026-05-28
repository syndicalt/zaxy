# Backend Shootout

- Report schema version: `1`
- Harness: `zaxy-backend-shootout`
- Generated at UTC: `2026-05-21T13:41:29Z`
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

| Backend | Contract | Status | Cold bootstrap ms | First useful init ms | First checkout ms | Append projection p95 ms | Projection eps | Checkout p95 ms | Checkout p99 ms | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected | Citation coverage | Mean quality | Answer@5 | Recall@5 | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |
|---------|----------|--------|-------------------|----------------------|-------------------|--------------------------|----------------|-----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|-------------------|--------------|----------|----------|--------------|-----------------------------|-------------------------|---------------------|
| embedded | retrieve | ok | 225.93 | 9347.717 | 8.431 | 24.674 | 57.007 | 10.55 | 11.528 | 0.005 | 0.007 | 1.892 | 2.334 | 3.285 | 4.02 | 2.049 | 2.614 | 3.513 | 0.004 | 0.005 | 0.009 | 124.683 | embedded | 100 | 100 | 1442.9 | 0.3985 | 0.3985 | 1545.45 | 0.3721 | 0.3721 | 1.0 | 0.575 | 0.575 | 1.0 | 28762112 | 652308480 | 28762112 | 10433.54 |
| embedded | answer_ready | ok | 225.93 | 9347.717 | 27.102 | 24.674 | 57.007 | 56.541 | 93.633 | 0.005 | 0.007 | 1.892 | 2.334 | 3.285 | 4.02 | 2.049 | 2.614 | 3.513 | 0.004 | 0.005 | 0.009 | 124.683 | embedded | 100 | 100 | 3377.175 | 0.2961 | 0.2961 | 3507.55 | 0.2851 | 0.2851 | 1.0 | 1.0 | 1.0 | 1.0 | 28762112 | 652308480 | 28762112 | 10433.54 |
| bm25 | retrieve | ok | 7.817 | 7.817 | 139.46 | - | - | 169.674 | 178.692 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 3944.9 | 0.1394 | 0.1394 | 3944.9 | 0.1394 | 0.1394 | 1.0 | 0.55 | 0.55 | 1.0 | 1598781 | 4096 | 0 | 0.0 |
