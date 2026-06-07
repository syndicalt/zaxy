# Backend Shootout

- Report schema version: `1`
- Harness: `zaxy-backend-shootout`
- Generated at UTC: `2026-05-21T14:31:25Z`
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

| Backend | Contract | Status | Cold bootstrap ms | First useful init ms | First checkout ms | Append projection p95 ms | Projection eps | Checkout p95 ms | Checkout p99 ms | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected | Citation coverage | Mean quality | Answer@5 | Recall@5 | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |
|---------|----------|--------|-------------------|----------------------|-------------------|--------------------------|----------------|-----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|-------------------|--------------|----------|----------|--------------|-----------------------------|-------------------------|---------------------|
| embedded | retrieve | ok | 106.506 | 188.072 | 0.716 | 17.391 | 49.097 | 0.716 | 0.716 | 0.002 | 0.008 | 0.008 | 0.023 | 0.055 | 0.055 | 0.075 | 0.083 | 0.083 | 0.002 | 0.003 | 0.003 | 79.161 | embedded | 6 | 3 | 141.0 | 3.5461 | 3.5461 | 173.5 | 2.8818 | 2.8818 | 1.0 | 0.5 | 0.5 | 0.5 | 13205504 | 165064704 | 13205504 | 212.976 |
| embedded | answer_ready | ok | 106.506 | 188.072 | 2.296 | 17.391 | 49.097 | 2.296 | 2.296 | 0.002 | 0.008 | 0.008 | 0.023 | 0.055 | 0.055 | 0.075 | 0.083 | 0.083 | 0.002 | 0.003 | 0.003 | 79.161 | embedded | 6 | 3 | 152.0 | 6.5789 | 6.5789 | 204.25 | 4.896 | 4.896 | 1.0 | 1.0 | 1.0 | 1.0 | 13205504 | 165064704 | 13205504 | 212.976 |
| bm25 | retrieve | ok | 0.044 | 0.044 | 0.143 | - | - | 0.143 | 0.143 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 128.5 | 7.7821 | 7.7821 | 128.5 | 7.7821 | 7.7821 | 1.0 | 1.0 | 1.0 | 1.0 | 775 | 0 | 0 | 0.0 |
