# Live Retrieval Benchmark

- Generated: `2026-05-12T03:22:32Z`
- Embedding provider: `hash:1536`
- Workload: `mempalace-source-recall-v1`
- Workload SHA-256: `f1e520fc9778012a707a9ba93e1c729fd3ea9f879021a49a46f52a472dc1f7dd`
- Events: `50`
- Queries: `25`
- Documents: `25`
- Lanes: `source-recall`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|--------|--------|--------|----------------|---------------|
| bm25 | 1.000 |  | 1.000 | 1.000 | 0.19 | 0.22 | 0.35 | 5031 | 1258 |
| md | 0.040 |  | 0.120 | 1.000 | 0.01 | 0.01 | 0.02 | 4909 | 1228 |
| md+vector | 0.200 |  | 0.200 | 1.000 | 5.52 | 5.63 | 5.97 | 5037 | 1260 |
| vector | 0.200 |  | 0.200 | 1.000 | 5.49 | 5.71 | 5.74 | 5037 | 1260 |
| zaxy | 1.000 |  | 1.000 | 1.000 | 31.09 | 35.97 | 84.89 | 4369 | 1092 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage |
|---------|----------|---------|------------|--------|---------------|-------------------|
| bm25 | source-recall | 25 | 1.000 | 0 | 1.000 | 1.000 |
| md | source-recall | 25 | 0.040 | 24 | 0.120 | 1.000 |
| md+vector | source-recall | 25 | 0.200 | 20 | 0.200 | 1.000 |
| vector | source-recall | 25 | 0.200 | 20 | 0.200 | 1.000 |
| zaxy | source-recall | 25 | 1.000 | 0 | 1.000 | 1.000 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.0000 | [0.0000, 0.0000] | 1.0000 | no |
| zaxy | md | 0.9600 | [0.8800, 1.0000] | 0.0001 | yes |
| zaxy | md+vector | 0.8000 | [0.6400, 0.9600] | 0.0001 | yes |
| zaxy | vector | 0.8000 | [0.6400, 0.9600] | 0.0001 | yes |
