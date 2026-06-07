# Live Retrieval Benchmark

- Generated: `2026-05-10T01:52:21Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `63e0be12c95da6f61653d41c7b53904cddd81c168b191b32bc632688239f93ca`
- Events: `980`
- Queries: `20`
- Subjects: `20`
- Sessions: `980`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|--------|--------|--------|----------------|---------------|
| bm25 | 0.700 | 0.800 | 123.68 | 126.84 | 127.39 | 135054 | 33761 |
| md | 0.150 | 0.000 | 0.10 | 0.28 | 0.72 | 86508 | 21627 |
| md+vector | 0.100 | 0.000 | 142.33 | 174.59 | 178.59 | 61540 | 15379 |
| vector | 0.100 | 0.000 | 125.79 | 130.66 | 130.85 | 59329 | 14826 |
| zaxy | 0.700 | 0.800 | 245.95 | 289.92 | 306.25 | 135054 | 33761 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses |
|---------|----------|---------|------------|--------|
| bm25 | longmemeval:single-session-user | 20 | 0.700 | 6 |
| md | longmemeval:single-session-user | 20 | 0.150 | 17 |
| md+vector | longmemeval:single-session-user | 20 | 0.100 | 18 |
| vector | longmemeval:single-session-user | 20 | 0.100 | 18 |
| zaxy | longmemeval:single-session-user | 20 | 0.700 | 6 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | 0.0000 | [0.0000, 0.0000] | 1.0000 | no |
| zaxy | md | 0.5500 | [0.3500, 0.7500] | 0.0010 | yes |
| zaxy | md+vector | 0.6000 | [0.4000, 0.8000] | 0.0005 | yes |
| zaxy | vector | 0.6000 | [0.4000, 0.8000] | 0.0005 | yes |
