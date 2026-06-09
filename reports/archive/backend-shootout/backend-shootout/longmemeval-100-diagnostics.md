# Backend Shootout

- Report schema version: `1`
- Harness: `zaxy-backend-shootout`
- Generated at UTC: `2026-05-20T12:56:41Z`
- Eventloom path: `reports/backend-shootout/longmemeval-100.eventloom.jsonl`
- Queries file: `reports/backend-shootout/longmemeval-100-queries.json`
- Session ID: `default`
- Queries: `100`
- Events: `1559`
- Limit: `5`
- Source Eventloom SHA-256: `9385a1462713c3ea0b99ced904d4e8f335cacc9071ae797a27f587672cdfae93`
- Source queries SHA-256: `c486d8b11c3e1f6013dc652011243f41443826b28d3e188578ac0fa36cc405a0`
- Workload events SHA-256: `79018ca643e1dfa9f6843a2d80982ee3be5440f81be20a0faa62fc29799c36f8`
- Workload queries SHA-256: `33d92ca63b130e542eebf4e104373a0edf6f90968d85ad60d49fbbb8bdce6fd1`

| Backend | Status | Cold bootstrap ms | First useful init ms | First checkout ms | Append projection p95 ms | Projection eps | Checkout p95 ms | Checkout p99 ms | Exact p50 ms | Exact p95 ms | Exact p99 ms | Keyword p50 ms | Keyword p95 ms | Keyword p99 ms | Vector p50 ms | Vector p95 ms | Vector p99 ms | Traversal p50 ms | Traversal p95 ms | Traversal p99 ms | Dashboard graph load ms | Dashboard source | Dashboard nodes | Dashboard edges | Returned tokens | Quality / 1k tokens | Answer@5 / 1k tokens | Injected tokens | Quality / 1k injected | Answer@5 / 1k injected | Citation coverage | Mean quality | Answer@5 | Recall@5 | Memory bytes | Resident memory delta bytes | On-disk footprint bytes | Rebuild recovery ms |
|---------|--------|-------------------|----------------------|-------------------|--------------------------|----------------|-----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|--------------|--------------|--------------|------------------|------------------|------------------|-------------------------|------------------|-----------------|-----------------|-----------------|----------------------|-----------------------|-----------------|------------------------|-------------------------|-------------------|--------------|----------|----------|--------------|-----------------------------|-------------------------|---------------------|
| embedded | ok | 108.659 | 31118.336 | 65.706 | 25.243 | 50.275 | 99.822 | 127.225 | 6.062 | 7.553 | 8.078 | 40.393 | 56.737 | 64.51 | 8.212 | 10.308 | 11.042 | 0.004 | 0.006 | 0.006 | 156.333 | embedded | 100 | 100 | 1791.53 | 0.1954 | 0.1954 | 1892.97 | 0.1849 | 0.1849 | 1.0 | 0.35 | 0.35 | 0.35 | 57270272 | 1446199296 | 57270272 | 34840.575 |
| bm25 | ok | 15.806 | 15.806 | 456.914 | - | - | 462.58 | 470.971 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 4179.5 | 0.0813 | 0.0813 | 4179.5 | 0.0813 | 0.0813 | 1.0 | 0.34 | 0.34 | 0.34 | 4797838 | 61440 | 0 | 0.0 |
