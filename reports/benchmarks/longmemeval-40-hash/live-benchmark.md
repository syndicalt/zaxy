# Live Retrieval Benchmark

- Generated: `2026-05-12T03:58:05Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `c042a16106a5907a80488aa83a76043ce85e366a8aa9d3e1e60260800f967a0b`
- Events: `7011`
- Queries: `40`
- Subjects: `40`
- Sessions: `1952`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | Source recall | Citation coverage | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|---------------|-------------------|--------|--------|--------|----------------|---------------|
| bm25 | 0.975 | 0.975 |  | 1.000 | 313.60 | 326.98 | 329.58 | 49633 | 12408 |
| md | 0.300 | 0.000 |  | 1.000 | 0.14 | 1.04 | 1.17 | 68449 | 17112 |
| md+vector | 0.450 | 0.400 |  | 1.000 | 864.65 | 1030.41 | 1045.37 | 57291 | 14313 |
| vector | 0.475 | 0.400 |  | 1.000 | 837.00 | 886.38 | 892.54 | 57283 | 14311 |
| zaxy | 0.825 | 0.750 |  | 1.000 | 91.99 | 116.56 | 217.18 | 31629 | 7901 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage |
|---------|----------|---------|------------|--------|---------------|-------------------|
| bm25 | longmemeval:single-session-user | 40 | 0.975 | 1 |  | 1.000 |
| md | longmemeval:single-session-user | 40 | 0.300 | 28 |  | 1.000 |
| md+vector | longmemeval:single-session-user | 40 | 0.450 | 22 |  | 1.000 |
| vector | longmemeval:single-session-user | 40 | 0.475 | 21 |  | 1.000 |
| zaxy | longmemeval:single-session-user | 40 | 0.825 | 7 |  | 1.000 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | -0.1500 | [-0.2750, -0.0500] | 0.0315 | no |
| zaxy | md | 0.5250 | [0.3500, 0.6750] | 0.0001 | yes |
| zaxy | md+vector | 0.3750 | [0.2000, 0.5250] | 0.0002 | yes |
| zaxy | vector | 0.3500 | [0.1750, 0.5000] | 0.0003 | yes |
