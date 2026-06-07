# Backend Shootout

- Report schema version: `1`
- Harness: `zaxy-backend-shootout`
- Generated at UTC: `2026-05-21T13:40:14Z`
- Eventloom path: `reports/backend-shootout/longmemeval-100.eventloom.jsonl`
- Queries file: `reports/backend-shootout/longmemeval-100-queries-with-targets.json`
- Session ID: `default`
- Queries: `100`
- Events: `1559`
- Limit: `5`
- Source Eventloom SHA-256: `9385a1462713c3ea0b99ced904d4e8f335cacc9071ae797a27f587672cdfae93`
- Source queries SHA-256: `f146f23f47dc2edfddb2d050b46e7f7a4bbe675135050ae8f6df00e467408d89`
- Workload events SHA-256: `79018ca643e1dfa9f6843a2d80982ee3be5440f81be20a0faa62fc29799c36f8`
- Workload queries SHA-256: `e106380ec1fdfb6e4e4c63d3e53b662466ae0df2a9061f15687d179ac01436a4`

| Backend | Contract | Status | Cold bootstrap ms | First useful init ms | First checkout ms | Append projection p95 ms | Projection eps | Checkout p95 ms | Checkout p99 ms | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected | Citation coverage | Mean quality | Answer@5 | Recall@5 | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |
|---------|----------|--------|-------------------|----------------------|-------------------|--------------------------|----------------|-----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|-------------------|--------------|----------|----------|--------------|-----------------------------|-------------------------|---------------------|
| embedded | retrieve | ok | 421.649 | 29620.186 | 12.705 | 26.931 | 53.393 | 19.915 | 21.937 | 0.005 | 0.006 | 0.007 | 3.633 | 6.08 | 6.65 | 6.104 | 7.689 | 8.659 | 0.004 | 0.006 | 0.007 | 161.379 | embedded | 100 | 100 | 1389.64 | 0.3742 | 0.3742 | 1492.24 | 0.3485 | 0.3485 | 1.0 | 0.52 | 0.52 | 0.99 | 57298944 | 1604280320 | 57298944 | 27678.562 |
| embedded | answer_ready | ok | 421.649 | 29620.186 | 37.615 | 26.931 | 53.393 | 90.478 | 101.728 | 0.005 | 0.006 | 0.007 | 3.633 | 6.08 | 6.65 | 6.104 | 7.689 | 8.659 | 0.004 | 0.006 | 0.007 | 161.379 | embedded | 100 | 100 | 3290.64 | 0.3009 | 0.3009 | 3426.8 | 0.2889 | 0.2889 | 1.0 | 0.99 | 0.99 | 1.0 | 57298944 | 1604280320 | 57298944 | 27678.562 |
| bm25 | retrieve | ok | 14.576 | 14.576 | 411.017 | - | - | 439.913 | 451.698 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 4179.5 | 0.1244 | 0.1244 | 4179.5 | 0.1244 | 0.1244 | 1.0 | 0.52 | 0.52 | 0.9 | 4797838 | 40960 | 0 | 0.0 |
