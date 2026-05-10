# Live Retrieval Benchmark

- Generated: `2026-05-10T01:41:31Z`
- Embedding provider: `hash:1536`
- Workload: `longmemeval-cleaned-v1`
- Workload SHA-256: `9328931d5371ad1c2efb281c0c76d292294c8a4af4bacbe7c83685ff488f455d`
- Events: `980`
- Queries: `20`
- Subjects: `20`
- Sessions: `980`
- Lanes: `longmemeval`

| Backend | Mean score | Identity recall | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |
|---------|------------|-----------------|--------|--------|--------|----------------|---------------|
| bm25 | 0.700 | 0.800 | 108.61 | 112.08 | 125.54 | 135054 | 33761 |
| md | 0.150 | 0.000 | 0.10 | 0.28 | 0.75 | 86508 | 21627 |
| md+vector | 0.100 | 0.000 | 142.31 | 173.88 | 174.76 | 61540 | 15379 |
| vector | 0.100 | 0.000 | 110.59 | 113.19 | 113.60 | 59329 | 14826 |
| zaxy | 0.600 | 0.600 | 136.46 | 240.72 | 547.70 | 62713 | 15673 |

## Category summaries

| Backend | Category | Queries | Mean score | Misses |
|---------|----------|---------|------------|--------|
| bm25 | longmemeval:single-session-user | 20 | 0.700 | 6 |
| md | longmemeval:single-session-user | 20 | 0.150 | 17 |
| md+vector | longmemeval:single-session-user | 20 | 0.100 | 18 |
| vector | longmemeval:single-session-user | 20 | 0.100 | 18 |
| zaxy | longmemeval:single-session-user | 20 | 0.600 | 8 |

## Paired comparisons

| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |
|--------|----------|------------------|--------|---------|-------------|
| zaxy | bm25 | -0.1000 | [-0.2500, 0.0000] | 0.5000 | no |
| zaxy | md | 0.4500 | [0.2500, 0.6500] | 0.0039 | yes |
| zaxy | md+vector | 0.5000 | [0.2500, 0.7000] | 0.0020 | yes |
| zaxy | vector | 0.5000 | [0.2500, 0.7000] | 0.0020 | yes |
