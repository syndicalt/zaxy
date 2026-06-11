# ANN Baseline — 2026-06-11, master @ post-2.1.0 (b68e1a9+)

Lane: `zaxy graph-scale-lanes --lanes vector-scale`, query_count 32, latency_passes 3,
ann_threshold 256 (lane-lowered), hash embeddings. Host: local dev machine.
Raw JSON: dim64/graph-scale-lanes.json, dim1536/graph-scale-lanes.json.
100k dim-64 reference (identical code, 2.1.0 lane run): recall 0.8969, ANN p50 37.9ms
vs exact 17.0ms, shadow sync ~20.6 min; int8 0.9938 / 9.5ms / 7.2MB.

## dim 64

| size | mode | recall@10 | p50 ms | p95 ms | first-query ms (build) |
|---|---|---|---|---|---|
| 1k | exact | 1.0 | 0.133 | 0.145 | 16.8 |
| 1k | ann | (run) | 4.562 | 5.968 | 2,408 |
| 1k | int8 | (pass) | 0.593 | 1.374 | 31.6 |
| 10k | exact | 1.0 | 6.128 | 21.658 | 130 |
| 10k | ann | 0.9062 FAIL | 9.809 | 11.110 | 53,328 |
| 10k | int8 | pass (all criteria) | 1.097 | 1.220 | 137 |

## dim 1536 (production-scale dimension)

| size | mode | recall@10 | p50 ms | p95 ms | first-query ms (build) |
|---|---|---|---|---|---|
| 1k | exact | 1.0 | 6.001 | 8.992 | 211 |
| 1k | ann | — | 10.489 | 12.384 | 18,140 |
| 1k | int8 | — | 8.629 | 10.878 | 220 |
| 10k | exact | 1.0 | 8.853 | 10.904 | 1,936 |
| 10k | ann | **0.5156 FAIL** | 26.502 | 30.455 | **196,290 (3.3 min @ 10k)** |
| 10k | int8 | **0.6094 FAIL** | 35.575 | 41.570 | 2,753 |

## Findings that reframe the 2.2 plan

1. **ANN recall collapses with dimension**: 0.906 (d64) → 0.516 (d1536) at 10k.
   Default index/query parameters are far off at high dimension.
2. **int8 quantization also collapses at d1536** (0.609): per-vector-scale int8
   candidate selection loses the true top-10 outside the 4x oversample at high
   dimension. The "promote quantization in 2.2" candidate FAILS at production
   dims on current evidence. (Caveat: hash-embedding value distribution may be
   adversarial for int8; verify against a realistic-embedding distribution
   before final conclusions.)
3. **Exact latency is fine at 10k/d1536 (8.9ms p50)** — but float64 memory is
   the real ceiling: 100k x 1536 x 8B ≈ 1.2GB vs the 256MB vector-cache budget.
   At production dims, approximate methods must win on MEMORY first.
4. **Build cost scales brutally**: ANN first-query 53s (10k/d64), 196s
   (10k/d1536), ~20min (100k/d64). Per-query p50 also includes the
   projected-graph create/drop overhead (research agent quantifying).
5. Quantized p50 at d1536 (35.6ms) is WORSE than exact (8.9ms) — int8 integer
   matmul + rerank in numpy loses to a single float64 BLAS matmul at high dim.

## Bars to beat (from the roadmap exit criteria, at 10^5)

- recall@10 ≥ 0.95 vs exact
- p50 latency better than exact
- resident bytes better than exact
- reproducible-enough builds for the deterministic lane block
