# Backend Shootout

- Report schema version: `1`
- Harness: `zaxy-backend-shootout`
- Generated at UTC: `2026-05-21T03:05:47Z`
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
| embedded | retrieve | ok | 113.49 | 53437.96 | 114.252 | 50.165 | 29.236 | 215.993 | 252.528 | 8.449 | 11.746 | 15.698 | 62.797 | 74.292 | 83.297 | 12.951 | 16.693 | 18.933 | 0.006 | 0.009 | 0.011 | 230.965 | embedded | 100 | 100 | 1391.42 | 0.3737 | 0.3737 | 1493.97 | 0.3481 | 0.3481 | 1.0 | 0.52 | 0.52 | 0.99 | 57298944 | 1661026304 | 57298944 | 51995.545 |
| embedded | answer_ready | ok | 113.49 | 53437.96 | 2992.81 | 50.165 | 29.236 | 5268.202 | 25321.07 | 8.449 | 11.746 | 15.698 | 62.797 | 74.292 | 83.297 | 12.951 | 16.693 | 18.933 | 0.006 | 0.009 | 0.011 | 230.965 | embedded | 100 | 100 | 3416.0 | 0.2927 | 0.2927 | 3552.07 | 0.2815 | 0.2815 | 1.0 | 1.0 | 1.0 | 1.0 | 57298944 | 1661026304 | 57298944 | 51995.545 |
