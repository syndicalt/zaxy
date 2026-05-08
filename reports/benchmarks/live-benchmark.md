# Live Retrieval Benchmark

- Generated: `2026-05-08T00:24:39Z`
- Embedding provider: `openai:text-embedding-3-small`
- Workload: `statistical-v1`
- Workload SHA-256: `6fae8e60567830db9254e5c90dbc850210a1be60293255a4d11128b52b7139ce`
- Events: `500`
- Queries: `300`
- Subjects: `100`

| Backend | Mean score | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|--------|--------|--------|----------------|---------------|
| md | 0.060 | 0.01 | 0.01 | 0.01 | 1474 | 369 |
| md+vector | 0.507 | 28.70 | 48.01 | 48.80 | 1681 | 421 |
| vector | 0.507 | 214.97 | 419.56 | 1035.00 | 1681 | 421 |
| zaxy | 1.000 | 41.20 | 56.83 | 71.45 | 964 | 237 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses |
|---------|----------|---------|------------|--------|
| md | current | 100 | 0.000 | 100 |
| md | temporal | 100 | 0.150 | 85 |
| md | traversal | 100 | 0.030 | 97 |
| md+vector | current | 100 | 0.470 | 53 |
| md+vector | temporal | 100 | 0.120 | 88 |
| md+vector | traversal | 100 | 0.930 | 7 |
| vector | current | 100 | 0.470 | 53 |
| vector | temporal | 100 | 0.120 | 88 |
| vector | traversal | 100 | 0.930 | 7 |
| zaxy | current | 100 | 1.000 | 0 |
| zaxy | temporal | 100 | 1.000 | 0 |
| zaxy | traversal | 100 | 1.000 | 0 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | md | 0.9400 | [0.9133, 0.9667] | 0.0001 | yes |
| zaxy | md+vector | 0.4933 | [0.4367, 0.5500] | 0.0001 | yes |
| zaxy | vector | 0.4933 | [0.4367, 0.5500] | 0.0001 | yes |
