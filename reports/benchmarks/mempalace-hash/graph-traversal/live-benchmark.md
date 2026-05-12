# Live Retrieval Benchmark

- Generated: `2026-05-12T03:22:38Z`
- Embedding provider: `hash:1536`
- Workload: `mempalace-graph-traversal-v1`
- Workload SHA-256: `c7972ceadf84dc32f51efa2bdd071a73ae69d590a35ea26f6749727677f81547`
- Events: `100`
- Queries: `25`
- Subjects: `25`
- Lanes: `graph-traversal`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|--------|--------|--------|----------------|---------------|
| bm25 | 0.920 | 0.667 |  | 1.000 | 0.28 | 0.29 | 0.40 | 2108 | 528 |
| md | 0.040 | 0.053 |  | 1.000 | 0.00 | 0.01 | 0.02 | 2023 | 506 |
| md+vector | 0.120 | 0.133 |  | 1.000 | 11.66 | 11.76 | 11.83 | 2113 | 529 |
| vector | 0.120 | 0.133 |  | 1.000 | 11.84 | 11.97 | 12.01 | 2113 | 529 |
| zaxy | 1.000 | 0.667 |  | 1.000 | 126.10 | 139.55 | 143.59 | 819 | 203 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage |
|---------|----------|---------|------------|--------|---------------|-------------------|
| bm25 | graph-traversal | 25 | 0.920 | 2 |  | 1.000 |
| md | graph-traversal | 25 | 0.040 | 24 |  | 1.000 |
| md+vector | graph-traversal | 25 | 0.120 | 24 |  | 1.000 |
| vector | graph-traversal | 25 | 0.120 | 24 |  | 1.000 |
| zaxy | graph-traversal | 25 | 1.000 | 0 |  | 1.000 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.0800 | [0.0000, 0.2000] | 0.4936 | no |
| zaxy | md | 0.9600 | [0.8800, 1.0000] | 0.0001 | yes |
| zaxy | md+vector | 0.8800 | [0.7800, 0.9600] | 0.0001 | yes |
| zaxy | vector | 0.8800 | [0.7800, 0.9600] | 0.0001 | yes |
