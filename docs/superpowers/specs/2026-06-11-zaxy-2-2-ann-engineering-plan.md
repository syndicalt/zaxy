# Zaxy 2.2 ANN Path Engineering Plan

## Purpose

The 2.1 vector-scale lane measured the embedded ANN (Kuzu HNSW) path below
exact brute-force search on every axis that matters, so 2.1 shipped with
`VECTOR_ANN_THRESHOLD` raised to keep ANN opt-in. This plan makes the ANN path
genuinely better than exact search where exact search stops being viable, and
lowers the threshold only when the lane proves it.

The guiding rule is unchanged from 2.1:

> Defaults move on lane evidence, not on assertion.

## Baseline (measured 2026-06-11, master @ post-2.1.0)

Full record: `/tmp/zaxy-ann-baseline/BASELINE.md` (lane runs at dim 64 and
dim 1536; 100k dim-64 reference from the identical released code).

| Metric | dim 64, 10k | dim 64, 100k | dim 1536, 10k |
| --- | --- | --- | --- |
| ANN recall@10 | 0.9062 | 0.8969 | **0.5156** |
| ANN p50 vs exact | 9.8ms vs 6.1ms | 37.9ms vs 17.0ms | 26.5ms vs 8.9ms |
| ANN build (first query) | 53s | ~20.6 min | 196s |
| int8 recall@10 | pass (0.99+) | 0.9938 | **0.6094** |
| int8 p50 vs exact | 1.1ms vs 6.1ms | 9.5ms vs 17.0ms | 35.6ms vs 8.9ms |

Three reframing findings versus the 2.1 understanding:

1. **Dimension is the dominant variable.** At production-scale dimension
   (1536), ANN recall collapses to 0.52 and int8 quantization — the 2.1
   promotion candidate — collapses to 0.61. The dim-64 lane was the easy case.
2. **At high dimension the fight is memory, not latency.** Exact float64 is
   8.9ms p50 at 10k/1536 — fine — but 100k x 1536 x 8B ≈ 1.2GB against a
   256MB vector-cache budget. Approximate methods must win on resident bytes
   first, recall second, latency third.
3. **int8's collapse is candidate selection, not rerank.** The float rerank is
   exact; at high dimension the true top-10 falls outside the fixed top-k×4
   int8 candidate set. Fix is adaptive oversampling and/or quantization with
   better high-dim behavior — measured against realistic embedding
   distributions, since hash-embedding value distributions may be adversarial
   for int8 specifically.

## Research Findings

### Kuzu reality (external research, fully sourced in /tmp/zaxy-ann-research/)

- **Kuzu is frozen.** 0.11.3 (2025-10-10) is the final release; the upstream
  repo is archived and official docs are offline. There is no upgrade path,
  only the LadybugDB fork (a dependency change, out of scope for the default
  path in 2.2). All engineering below targets the pinned 0.11.3, verified
  against its source.
- **`efs` is the primary recall knob** (query-time candidate list, default
  200; build-side `efc=200`, `ml=60`, `mu=30`, undocumented `alpha=1.1`).
  Kuzu's own NaviX paper methodology reaches 95% recall by raising `efs`
  per dataset rather than touching build parameters.
- **Per-query `PROJECT_GRAPH` create/drop is not the intended pattern.**
  Projected graphs are connection-scoped and long-lived by design. For
  unfiltered queries the table name can be passed directly — no projected
  graph at all. Caveat: the Python connection lifecycle must keep the
  projected graph's connection alive for reuse.
- **`COPY FROM` (in-memory Arrow, no file roundtrip) is the documented bulk
  path**; CREATE/MERGE (our batched UNWIND) is documented for "small sporadic
  additions" only. Index should be created after bulk load: `SET` on indexed
  properties is forbidden (#5965) and bulk mutation under a live index is the
  historically buggiest path upstream.
- **Build nondeterminism is by design** (acknowledged data races in the
  concurrent HNSW build; no seed). `CALL THREADS=1` before build is a
  plausible mitigation at build-time cost; otherwise the lane gates on
  measured recall with margin rather than byte-identical builds.
- **Frozen-upstream defects to design around**: #6047 (sequential HNSW load
  can block DB open — cold-start risk; measure), #6040 (DROP_VECTOR_INDEX
  metadata corruption — the rebuild cycle must be drop-free or tested
  hard against this), #6012 (prefer Arrow over Pandas for LOAD).
- **The bar is reachable**: comparable embedded systems serve this workload at
  ≥0.95 recall in 1–2ms; Kuzu's paper does 1M×960-dim filtered search at 95%
  recall in 5–40ms.

### Decisive experiments (10k, dim 64, this machine; raw JSON
/tmp/zaxy-ann-exp/results.json)

- **E1 bulk load**: UNWIND batches 8.16s vs `COPY FROM` parquet **0.08s —
  100x**. Index build after copy: 6.25s. The 20-minute 100k sync is fully
  explained and fully fixable. Caveat discovered en route: `COPY FROM` an
  in-memory Arrow table with a fixed-size-list column **segfaults** 0.11.3;
  COPY from a parquet file is the safe path. (Also: an unbound `$param` in
  any query segfaults rather than raising — production query construction
  must be audited for both.)
- **E2 query pattern**: direct-table 6.56ms p50 ≈ per-query projected graph
  6.95ms ≈ reused projected graph 7.93ms at 10k. Projected-graph create/drop
  is NOT the dominant overhead at this scale. The lane's 100k latency gap
  (37.9ms vs exact 17.0ms) most plausibly lives in the per-query prefilter
  mask scan over the session/version predicate — the experiment queried
  unfiltered, production always filters. Profiling that scan is the first
  task of the implementation wave.
- **E3 recall**: **1.0 at every efs (200/400/800)** with float32-consistent
  ground truth — where the production lane measures 0.906 on the same corpus
  scale. The recall deficit is NOT HNSW search quality. Prime suspect: the
  shadow table stores float32 while the lane's exact ground truth ranks in
  float64; near-tie flips at the precision boundary, amplified at high
  dimension by tie-dense hash embeddings (the 0.52 at dim 1536). The
  oversample + float64-rerank design (A3) eliminates this class entirely;
  `efs` tuning is demoted to a secondary lever.
- **E4 determinism**: inconclusive at saturated recall (all runs 1.0,
  single- and multi-threaded). Robustness comes from the rerank, not from
  build determinism.

## Design

### Workstream A — Query path (latency + recall)

Lever order revised by E2/E3: rerank first, filter-scan profiling second,
projected-graph hygiene third, `efs` last.

1. **Oversample + exact float64 rerank on the ANN path** (mirroring the int8
   design): fetch k×oversample from HNSW, rerank with exact float64 scores
   from the already-resident entity vectors. E3 says this eliminates the
   measured recall deficit class (float32-boundary tie flips) entirely and
   makes recall robust to build variance — the determinism mitigation that
   costs nothing at build time.
2. **Profile and fix the filtered-query cost at 10^5.** E2 shows the latency
   gap is not graph create/drop; the per-query prefilter mask scan over the
   session/version predicate is the suspect. Options in order: avoid the
   predicate entirely when one (session, version) group owns the whole shadow
   generation (make the shadow table per-group so unfiltered direct-table
   queries are the common case); otherwise one long-lived projected graph per
   (session, version) on the store's derived-cache pattern.
3. **Drop per-query projected graphs** regardless (hygiene + the design
   intent of connection-scoped graphs), with lifecycle tied to
   `_clear_read_caches`.
4. **Expose `efs`** as setting `VECTOR_ANN_EFS` (default from the lane sweep;
   secondary lever now — E3 hit 1.0 at the 200 default under clean
   conditions). Capabilities reports it.

### Workstream B — Build path (sync time + reproducibility)

1. **Replace batched UNWIND sync with `COPY FROM` an in-memory Arrow table**
   for full rebuilds; create the HNSW index after the copy completes.
2. **Rebuild without DROP_VECTOR_INDEX** (#6040): rebuild into a fresh shadow
   table generation (e.g. `ZaxyVectorAnnShadow{dim}_g{n}`), swap the active
   generation atomically in store state, and drop the old *table* (not the
   index) afterward. Test the full cycle hard.
3. **Incremental small deltas stay on the existing insert path** (inserts are
   reflected in queries — verified in 2.1); full COPY rebuilds trigger on the
   same lazy signature change that rebuilds the dense matrix today, with a
   delta-threshold to choose between incremental insert and generation swap.
4. **Cold-start guard** (#6047): measure index-load cost at 10^5 on DB open;
   if it blocks, document and gate index existence behind the threshold so
   default-path users never pay it.

### Post-ann.1 diagnostic findings (dim-1536 root cause)

The 2.2-ann.1 gate runs failed recall at dim 1536 (0.55/0.61) even with the
float64 rerank. Follow-up diagnostics isolated the cause and exonerate the
index:

- float32 brute-force ceiling on the lane corpus: **0.5344** — even exact
  float32 search cannot exceed it against float64 ground truth.
- float64-stored HNSW scores the same (~0.52–0.54): storage precision is not
  the cause either.
- Tie analysis: at dim 1536 the hash-embedding corpus has a **median of 210
  vectors exactly tied** with the true top-10; the float64 score gap between
  rank 10 and rank 40 is **0.0**. Recall@10 is ill-posed on this corpus — any
  top-10 among the tied set is equally correct.
- Gaussian-distribution control on the identical index: 0.8531 at efs 200,
  **0.9875 at efs 400, 1.0 at efs 800**. With realistic distributions the
  index is healthy; efs 400 is the evidence-backed high-dim default.

Consequences (folded into Workstream C below): the lane gains a tie-aware
recall metric (hit = retrieved score equals the k-th true score — standard
ANN-benchmarking practice), reported ALONGSIDE strict recall so nothing is
hidden; the realistic-distribution variant (C3) becomes the high-dim gate
corpus; and `VECTOR_ANN_EFS` default moves to 400 on the sweep evidence.

### Workstream C — Quantized path at high dimension

1. **Adaptive oversampling**: scale the int8 candidate multiplier with
   dimension (the fixed k×4 is the measured failure at 1536); sweep on the
   lane to find the recall/latency frontier.
2. **Evaluate int8 asymmetric scoring** (float query × int8 corpus, per-dim
   or per-block scales) if oversampling alone cannot reach 0.95 at 1536
   within latency budget.
3. **Realistic-distribution check**: add an optional lane variant using a
   realistic embedding distribution (e.g. normalized Gaussian mixture or
   downloadable real vectors kept out of the default path) so int8
   conclusions are not artifacts of hash-embedding value distributions.

### Workstream D — Consolidation

The store currently carries three parallel search paths (dense float64,
`_AnnVectorGroup`, `_QuantizedVectorGroup`) with duplicated selection,
scoring, and cache-accounting logic. Consolidate behind one internal strategy
interface (selection → candidates → exact rerank → results with `exact` flag),
so A/B/C land as strategy implementations rather than more branching. This is
the "consolidate where needed" mandate — done as part of the work, not as a
separate refactor pass.

## Decision Gates

| Gate | Evidence required | Action on pass | Action on fail |
| --- | --- | --- | --- |
| G1 query path | Lane at 10^5 dim 64: ANN recall ≥0.95 AND p50 < exact | proceed to G2 | ANN stays opt-in; record findings |
| G2 build path | Full rebuild at 10^5 in single-digit minutes; rebuild cycle survives generation-swap stress test | proceed to G3 | threshold stays; incremental-only posture documented |
| G3 high-dim | Lane at 10^4–10^5 dim 1536: at least one approximate mode (ANN or int8) recall ≥0.95 with bytes < exact | mode becomes the documented high-dim recommendation | exact remains the only recommendation at high dim; memory ceiling documented |
| G4 threshold | G1+G2 pass with margin on two consecutive lane runs | lower `VECTOR_ANN_THRESHOLD` with migration note | threshold unchanged |

No gate is judged on a single lane run; HNSW build variance means each gate
needs two consecutive passing runs (the lane's documented nondeterminism
posture).

## Non-Goals

- No new Python dependencies on the default path; no backend migration
  (LadybugDB or alternatives are a 2.3+ strategic decision, informed by this
  work's findings on frozen-Kuzu limits).
- No change to the exact path's behavior below the threshold.
- No Eventloom or projection-contract changes — this is entirely inside the
  embedded store's vector machinery.
- No public benchmark claims from lane numbers; internal validation labels
  throughout.

## Increment Plan

1. **2.2-ann.1 (Workstreams A + D)**: strategy consolidation, direct-table /
   reused projected graph, `VECTOR_ANN_EFS`, ANN oversample+rerank. Lane
   evidence against G1.
2. **2.2-ann.2 (Workstream B)**: COPY-based generation-swap rebuilds,
   delta-threshold incremental policy, cold-start measurement. Evidence
   against G2.
3. **2.2-ann.3 (Workstream C)**: adaptive int8 oversampling, high-dim sweep,
   realistic-distribution lane variant. Evidence against G3.
4. **2.2-ann.4**: G4 threshold decision, docs (configuration/embeddings/
   migration), capabilities reporting, release notes.

Each increment lands green (ruff, mypy strict, full pytest with coverage,
site freshness) and updates the lane before the next starts.

## G4 Outcome (2026-06-11)

G4 passed and the threshold moved, scoped to the dimension the evidence
covers. Decision as shipped in 2.2-ann.4:

- `VECTOR_ANN_THRESHOLD` lowered `1000000` → `100000`. Evidence: two
  consecutive vector-scale lane runs at exactly 10^5 vectors, dimension 64,
  with `status: pass` and ANN `all_criteria_met` — recall@10 of 1.0 on both
  the strict and tie-aware metrics, ANN p50 at-or-better than exact in-run
  (24.17ms vs 24.20ms, then 26.67ms vs 30.82ms), resident bytes improved
  (0 vs 51.2MB), full COPY builds 92–98s
  (`docs/research/artifacts/ann-2026-06/ann3-d64-100k-r1.json`, `-r2.json`).
- Engagement is two-clause within a dimension ceiling: a scope at or below
  `VECTOR_ANN_MAX_DIMENSION` (default `64`, the measured envelope of the
  double pass) engages when count >= threshold (clause a, inclusive) or when
  its exact float64 matrix (count × dimension × 8) would exceed
  `VECTOR_INDEX_CACHE_MAX_BYTES` (clause b; opt out with
  `VECTOR_ANN_BYTE_BUDGET_ENGAGEMENT=false`). The ceiling exists because the
  G4 evidence does not transfer upward: at dim 1536/50k gaussian the lane
  measured HNSW recall@10 of **0.6344** at efs 400 (latency better — 45.2ms
  vs exact 55.6ms — but recall disqualifying), while exact stayed serviceable
  at 25.8ms p50 despite sitting 2.29× over the byte budget (cache-of-one
  residency, 614MB)
  (`docs/research/artifacts/ann-2026-06/ann3-d1536-50k-gauss-crossover.json`,
  `ann4-d1536-50k-dimension-gated.json`).
- efs scaling does not rescue high-dim scale: efs 800 at the same corpus
  recovered recall only to 0.8438 while p50 rose to 69.8ms vs exact 37.5ms —
  tuning erodes the latency advantage faster than it repairs recall
  (`ann4-d1536-50k-gauss-efs800.json`). Within the ceiling, clause (b) is a
  defense-in-depth backstop (at dim 64 the count clause fires first; the byte
  clause protects if the count threshold is raised or the budget lowered).
- High-dim guidance as shipped: over-budget high-dimension corpora stay on
  exact search (measured serviceable at cache-of-one residency); int8 remains
  opt-in for memory-tightest deployments (recall 1.0 deterministic, 8× byte
  advantage, but 228ms p50 at 50k×1536 — latency-honest in docs).
- Strategic consequence recorded for 2.3: frozen-Kuzu HNSW is measured
  inadequate exactly where approximate search matters most (high-dim, large
  N). Evaluating the LadybugDB fork or an alternative embedded ANN backend is
  formally on the 2.3 agenda, with this corpus of lane evidence as the
  acceptance bar.
  (`docs/research/artifacts/ann-2026-06/ann3-d1536-50k-gauss-crossover.json`)
  — the LRU eviction keeps the newest matrix resident, so a single
  over-budget scope is a cache of one, not a thrash.
- An explicit `VECTOR_ANN_THRESHOLD` stays an absolute count override for
  clause (a); `VECTOR_QUANTIZATION=int8` keeps its pre-G4 precedence below
  the count threshold and is exempt from clause (b); quantized engagement is
  otherwise unchanged. `memory_capabilities` reports the effective rule
  (`ann_threshold`, `ann_max_dimension`, `ann_byte_budget_engagement`,
  `vector_index_cache_max_bytes`).
- Migration note: `docs/migration.md` ("2.2: ANN engagement defaults");
  release note: `CHANGELOG.md` Unreleased (2.2.0).
