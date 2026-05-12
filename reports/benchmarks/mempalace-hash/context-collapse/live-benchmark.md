# Live Retrieval Benchmark

- Generated: `2026-05-12T03:22:55Z`
- Embedding provider: `hash:1536`
- Workload: `mempalace-context-collapse-v1`
- Workload SHA-256: `33d8d30864c605cb9665f56a353971926d301005bad05fd7239e09904e266249`
- Events: `1025`
- Queries: `25`
- Sessions: `25`
- Lanes: `context-collapse`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|--------|--------|--------|----------------|---------------|
| bm25 | 1.000 | 1.000 |  | 1.000 | 4.42 | 4.56 | 9.43 | 3552 | 889 |
| md | 0.000 | 0.020 |  |  | 0.00 | 0.01 | 0.02 | 3479 | 870 |
| md+vector | 0.000 | 0.120 |  |  | 114.55 | 122.17 | 122.93 | 3526 | 882 |
| vector | 0.000 | 0.120 |  |  | 116.66 | 125.19 | 125.50 | 3526 | 882 |
| zaxy | 1.000 | 1.000 |  | 1.000 | 36.75 | 56.28 | 84.50 | 1524 | 379 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage |
|---------|----------|---------|------------|--------|---------------|-------------------|
| bm25 | context-collapse | 25 | 1.000 | 0 |  | 1.000 |
| md | context-collapse | 25 | 0.000 | 25 |  |  |
| md+vector | context-collapse | 25 | 0.000 | 25 |  |  |
| vector | context-collapse | 25 | 0.000 | 25 |  |  |
| zaxy | context-collapse | 25 | 1.000 | 0 |  | 1.000 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.0000 | [0.0000, 0.0000] | 1.0000 | no |
| zaxy | md | 1.0000 | [1.0000, 1.0000] | 0.0001 | yes |
| zaxy | md+vector | 1.0000 | [1.0000, 1.0000] | 0.0001 | yes |
| zaxy | vector | 1.0000 | [1.0000, 1.0000] | 0.0001 | yes |
