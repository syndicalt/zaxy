# Live Retrieval Benchmark

- Generated: `2026-05-07T07:00:57Z`
- Embedding provider: `hash:1536`

| Backend | Mean score | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|--------|--------|--------|----------------|---------------|
| md | 0.060 | 0.00 | 0.01 | 0.01 | 1474 | 369 |
| md+vector | 0.080 | 28.56 | 47.57 | 48.22 | 1671 | 418 |
| vector | 0.070 | 46.33 | 46.92 | 49.25 | 1615 | 404 |
| zaxy | 0.983 | 29.70 | 41.50 | 56.90 | 452 | 110 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses |
|---------|----------|---------|------------|--------|
| md | current | 100 | 0.000 | 100 |
| md | temporal | 100 | 0.150 | 85 |
| md | traversal | 100 | 0.030 | 97 |
| md+vector | current | 100 | 0.030 | 97 |
| md+vector | temporal | 100 | 0.130 | 87 |
| md+vector | traversal | 100 | 0.080 | 92 |
| vector | current | 100 | 0.020 | 98 |
| vector | temporal | 100 | 0.130 | 87 |
| vector | traversal | 100 | 0.060 | 94 |
| zaxy | current | 100 | 0.970 | 3 |
| zaxy | temporal | 100 | 1.000 | 0 |
| zaxy | traversal | 100 | 0.980 | 4 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | md | 0.9233 | [0.8933, 0.9533] | 0.0001 | yes |
| zaxy | md+vector | 0.9033 | [0.8700, 0.9350] | 0.0001 | yes |
| zaxy | vector | 0.9133 | [0.8800, 0.9417] | 0.0001 | yes |
