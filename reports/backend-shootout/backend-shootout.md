# Backend Shootout

- Report schema version: `1`
- Harness: `zaxy-backend-shootout`
- Generated at UTC: `2026-05-20T09:45:17Z`
- Eventloom path: `reports/backend-shootout/sample.eventloom`
- Queries file: `reports/backend-shootout/queries.json`
- Session ID: `agent-1`
- Queries: `4`
- Events: `4`
- Limit: `5`
- Source Eventloom SHA-256: `a0a31aaa54b946a5f3da26fbf61a2421c5cf2f91d12b2e2b42a4bcbe9d2f53c8`
- Source queries SHA-256: `cb36efd7b55a91966024bf3668da4c536c8759cafa3d7899a37012e03c838df8`
- Workload events SHA-256: `5df099c2442454347f574de253473fc776da8c2f55dc7bdbffb22f803c7851b9`
- Workload queries SHA-256: `7818dc2e3343c7558a3ebfd83458766d054e7be026017c359d4bf50acc35ab18`

| Backend | Status | Cold bootstrap ms | First useful init ms | First checkout ms | Append projection p95 ms | Projection eps | Checkout p95 ms | Checkout p99 ms | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected | Citation coverage | Mean quality | Answer@5 | Recall@5 | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |
|---------|--------|-------------------|----------------------|-------------------|--------------------------|----------------|-----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|-------------------|--------------|----------|----------|--------------|-----------------------------|-------------------------|---------------------|
| embedded | ok | 83.998 | 200.721 | 9.014 | 28.772 | 34.302 | 9.014 | 9.014 | 2.642 | 4.413 | 4.413 | 0.092 | 0.169 | 0.169 | 0.104 | 0.137 | 0.137 | 0.003 | 0.003 | 0.003 | 65.037 | embedded | 6 | 3 | 237.0 | 2.1097 | 2.1097 | 303.0 | 1.6502 | 1.6502 | 1.0 | 0.5 | 0.5 | 0.5 | 13205504 | 161452032 | 13205504 | 215.742 |
| bm25 | ok | 0.03 | 0.03 | 0.115 | - | - | 0.115 | 0.115 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 128.5 | 7.7821 | 7.7821 | 128.5 | 7.7821 | 7.7821 | 1.0 | 1.0 | 1.0 | 1.0 | 775 | 0 | 0 | 0.0 |
