# Live Retrieval Benchmark

- Generated: `2026-05-12T03:22:29Z`
- Embedding provider: `hash:1536`
- Workload: `mempalace-temporal-recall-v1`
- Workload SHA-256: `9f3fd26449b6673d6437519a4c078699d77030327c56f7c72b30a85c9b970a97`
- Events: `75`
- Queries: `75`
- Subjects: `25`
- Lanes: `temporal-recall`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|--------|--------|--------|----------------|---------------|
| bm25 | 0.000 | 1.000 |  |  | 0.26 | 0.27 | 0.34 | 3169 | 793 |
| md | 0.013 | 0.080 |  |  | 0.00 | 0.01 | 0.01 | 3164 | 791 |
| md+vector | 0.047 | 0.107 |  | 1.000 | 8.40 | 8.75 | 9.57 | 3168 | 792 |
| vector | 0.047 | 0.107 |  | 1.000 | 8.55 | 8.96 | 9.09 | 3168 | 792 |
| zaxy | 1.000 | 1.000 |  | 1.000 | 20.31 | 26.70 | 34.12 | 383 | 95 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage |
|---------|----------|---------|------------|--------|---------------|-------------------|
| bm25 | temporal-recall | 75 | 0.000 | 75 |  |  |
| md | temporal-recall | 75 | 0.013 | 75 |  |  |
| md+vector | temporal-recall | 75 | 0.047 | 73 |  | 1.000 |
| vector | temporal-recall | 75 | 0.047 | 73 |  | 1.000 |
| zaxy | temporal-recall | 75 | 1.000 | 0 |  | 1.000 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 1.0000 | [1.0000, 1.0000] | 0.0001 | yes |
| zaxy | md | 0.9867 | [0.9667, 1.0000] | 0.0001 | yes |
| zaxy | md+vector | 0.9533 | [0.9067, 0.9867] | 0.0001 | yes |
| zaxy | vector | 0.9533 | [0.9067, 0.9867] | 0.0001 | yes |
