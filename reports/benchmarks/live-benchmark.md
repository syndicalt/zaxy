# Live Retrieval Benchmark

- Generated: `2026-05-08T04:29:10Z`
- Embedding provider: `openai:text-embedding-3-small`
- Workload: `suite-v1`
- Workload SHA-256: `fd3e2679e37b0953bb2c2ca90f5b98b803a3983b7f0661a6a706e0ef2b41acae`
- Events: `850`
- Queries: `650`
- Subjects: `100`
- Documents: `250`
- Sessions: `50`
- Lanes: `current, temporal, traversal, document, transcript, mixed`

| Backend | Mean score | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|--------|--------|--------|----------------|---------------|
| md | 0.005 | 0.01 | 0.22 | 0.22 | 2371 | 593 |
| md+vector | 0.520 | 39.91 | 84.40 | 92.60 | 3839 | 960 |
| vector | 0.520 | 260.24 | 559.89 | 2325.21 | 3839 | 960 |
| zaxy | 1.000 | 94.01 | 126.26 | 133.60 | 1634 | 404 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses |
|---------|----------|---------|------------|--------|
| md | current | 100 | 0.000 | 100 |
| md | document | 250 | 0.000 | 250 |
| md | mixed | 50 | 0.000 | 50 |
| md | temporal | 100 | 0.000 | 100 |
| md | transcript | 50 | 0.000 | 50 |
| md | traversal | 100 | 0.035 | 97 |
| md+vector | current | 100 | 0.510 | 49 |
| md+vector | document | 250 | 0.456 | 158 |
| md+vector | mixed | 50 | 0.307 | 50 |
| md+vector | temporal | 100 | 0.160 | 84 |
| md+vector | transcript | 50 | 0.980 | 1 |
| md+vector | traversal | 100 | 0.930 | 7 |
| vector | current | 100 | 0.510 | 49 |
| vector | document | 250 | 0.456 | 158 |
| vector | mixed | 50 | 0.307 | 50 |
| vector | temporal | 100 | 0.160 | 84 |
| vector | transcript | 50 | 0.980 | 1 |
| vector | traversal | 100 | 0.930 | 7 |
| zaxy | current | 100 | 1.000 | 0 |
| zaxy | document | 250 | 1.000 | 0 |
| zaxy | mixed | 50 | 1.000 | 0 |
| zaxy | temporal | 100 | 1.000 | 0 |
| zaxy | transcript | 50 | 1.000 | 0 |
| zaxy | traversal | 100 | 1.000 | 0 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | md | 0.9946 | [0.9885, 0.9992] | 0.0001 | yes |
| zaxy | md+vector | 0.4795 | [0.4431, 0.5154] | 0.0001 | yes |
| zaxy | vector | 0.4795 | [0.4431, 0.5154] | 0.0001 | yes |
