# Backend Shootout

- Report schema version: `1`
- Harness: `zaxy-backend-shootout`
- Generated at UTC: `2026-05-20T13:08:30Z`
- Eventloom path: `reports/backend-shootout/longmemeval-40.eventloom.jsonl`
- Queries file: `reports/backend-shootout/longmemeval-40-queries.json`
- Session ID: `default`
- Queries: `40`
- Events: `520`
- Limit: `5`
- Source Eventloom SHA-256: `2870768feac6aa5378edd22f4bc9480f24642c37fea785417589b020a516adb6`
- Source queries SHA-256: `0e46dd25ffe69f9ca41a3779ba58acce5d5fc71fc269ba2e559c9099fd903da0`
- Workload events SHA-256: `146d2d262e951a711341aefefed13ac001b20d5c0b8d948801dd9a8bdd9583ea`
- Workload queries SHA-256: `36618a0f823577d620fef966875835469a6f9173f8570213e2594af8070f7c3d`

| Backend | Status | Cold bootstrap ms | First useful init ms | First checkout ms | Append projection p95 ms | Projection eps | Checkout p95 ms | Checkout p99 ms | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected | Citation coverage | Mean quality | Answer@5 | Recall@5 | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |
|---------|--------|-------------------|----------------------|-------------------|--------------------------|----------------|-----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|-------------------|--------------|----------|----------|--------------|-----------------------------|-------------------------|---------------------|
| embedded | ok | 84.875 | 10722.284 | 37.872 | 29.953 | 48.885 | 48.512 | 61.538 | 3.356 | 3.776 | 5.056 | 13.628 | 15.807 | 16.776 | 2.956 | 3.299 | 3.42 | 0.004 | 0.005 | 0.005 | 126.279 | embedded | 100 | 100 | 1350.225 | 0.1852 | 0.1852 | 1453.15 | 0.172 | 0.172 | 1.0 | 0.25 | 0.25 | 0.25 | 28762112 | 680677376 | 28762112 | 11090.164 |
| bm25 | ok | 5.304 | 5.304 | 151.852 | - | - | 159.918 | 163.908 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 3944.9 | 0.0634 | 0.0634 | 3944.9 | 0.0634 | 0.0634 | 1.0 | 0.25 | 0.25 | 0.25 | 1598781 | 28672 | 0 | 0.0 |
