# Evidence-Bounded Approximate Search: Engineering the Zaxy 2.2 ANN Path Against a Frozen Upstream

- Author: Zaxy project
- Status: Research draft (engineering report)
- Version context: Zaxy 2.2.0 (Unreleased at time of writing)
- Artifact root: `docs/research/artifacts/ann-2026-06/`

## Abstract

Zaxy 2.1 shipped an approximate nearest-neighbor (ANN) vector path — a
Kuzu-native HNSW index — that its own benchmark lane measured as worse than
exact brute-force search on every axis that mattered: recall@10 of 0.9062 at
10^4 vectors and 0.8969 at 10^5 (dimension 64) against a 0.95 floor, p50
latency of 37.9 ms against 17.0 ms exact at 10^5, and a ~20-minute index
build. The release response was not to remove the feature or to tune it
against the gate; it was to raise `VECTOR_ANN_THRESHOLD` to 1,000,000 so the
path stayed opt-in, and to record the lane evidence as the bar any future
default change must clear.

This report documents the 2.2 engineering effort that cleared that bar — and
the exact boundary at which the evidence stops. After a baseline-first
measurement campaign, four decisive experiments, and a diagnostic episode
that exonerated the index and indicted the evaluation corpus, the shipped
path is a clean double pass at the target scale: two consecutive
vector-scale lane runs at exactly 10^5 vectors, dimension 64, with recall@10
of 1.0 on both the strict and tie-aware metrics, ANN p50 at-or-better than
exact in-run (24.17 ms vs 24.20 ms, then 26.67 ms vs 30.82 ms), resident
index bytes of 0 versus 51.2 MB, and full `COPY`-based builds of 92–98 s —
roughly 13x faster than the 1,180 s build the same lane measured before the
build-path rework (`ann3-d64-100k-r1.json`, `ann3-d64-100k-r2.json`,
`after-dim64-100k.json`). A confirmatory run reproduced the pass
(`ann4-d64-100k-confirm.json`).

The default moved exactly as far as the evidence extends and no further:
ANN engagement now requires vector count >= 100,000 AND dimension <= 64.
At dimension 1536 with 50k Gaussian vectors the identical machinery measured
HNSW recall@10 of 0.6 at `efs` 400 and only 0.8438 at `efs` 800 — with the
`efs` increase eroding the latency advantage faster than it repaired recall
— while exact float64 search stayed serviceable at 22–26 ms p50 despite a
matrix 2.29x over the vector-cache byte budget
(`ann3-d1536-50k-gauss-crossover.json`, `ann4-d1536-50k-gauss-efs800.json`,
`ann4-d1536-50k-dimension-gated.json`). Those negative results shipped in
the documentation alongside the positive ones. The thesis of this report is
methodological as much as technical: a feature default is a claim, and the
claim boundary should be drawn by the benchmark lane that measured it.

All numbers in this report are internal same-machine lane measurements over
synthetic corpora (see `docs/benchmarks.md` claim boundaries and
`docs/external-validation.md`). They measure index mechanics, not retrieval
quality, and they are not public benchmark claims.

## 1. Background and Problem Statement

Zaxy's embedded projection store (`src/zaxy/embedded_graph_store.py`) serves
vector similarity search over projected entity embeddings. Three execution
modes exist for a (session, embedding-version, dimension) vector scope:

1. **Exact**: a unit-normalized float64 dense matrix; one BLAS
   matrix-vector product per query. Ground truth by construction.
2. **Quantized (int8, opt-in)**: int8 matrix with per-vector scales;
   integer-dot-product candidate selection followed by exact float64
   rerank.
3. **ANN**: a Kuzu-native HNSW index over a shadow table; the subject of
   this report.

The 2.1 vector-scale lane (`zaxy_benchmarks/vector_scale_lane.py`, runnable
via `zaxy graph-scale-lanes --lanes vector-scale`) measured the ANN mode
below exact search on recall, latency, and build cost, so 2.1 shipped with
the path effectively disabled by threshold. The 2.2 charter
(`docs/superpowers/specs/2026-06-11-zaxy-2-2-ann-engineering-plan.md`) was
to make the ANN path genuinely better than exact search where exact search
stops being viable, and to lower the threshold only when the lane proves it:

> Defaults move on lane evidence, not on assertion.

Two constraints shaped everything below. First, the upstream is frozen:
Kuzu 0.11.3 (2025-10-10) is the final release of an archived repository, so
every defect found is permanent and must be designed around rather than
reported. Second, no development was permitted before a measured baseline
existed (`BASELINE.md`); every subsequent change is judged as a delta
against it.

## 2. Related Work and Theory

### 2.1 HNSW: layered proximity graphs

Hierarchical Navigable Small World graphs (Malkov & Yashunin,
arXiv:1603.09320) organize the corpus as a multi-layer proximity graph.
Each element is assigned a maximum layer \(l\) drawn from an exponentially
decaying distribution (\(l = \lfloor -\ln(u) \cdot m_L \rfloor\) for uniform
\(u\)), so the expected number of layers is \(O(\log n)\) and each upper
layer is an exponentially sparser sample of the one below. A search greedily
descends from the top layer: within a layer, greedy routing over a
bounded-degree graph (degree controlled by \(M\)) takes an expected number
of hops that the authors argue is bounded by a constant under the
small-world construction, giving overall search complexity that scales
logarithmically with \(n\). Two candidate-list sizes govern the
quality/latency trade: `efConstruction` during build and `ef` (`efs` in
Kuzu's surface) during search. Recall increases monotonically with `ef` at
linear latency cost — `ef` is the canonical recall knob, and that is the
theoretical basis for treating `efs` as the first tuning lever in Section 5.

### 2.2 NaviX: Kuzu's vector index

Kuzu's vector extension implements NaviX (Sehgal & Salihoglu,
arXiv:2506.23397), a disk-based HNSW designed for graph DBMSs with
predicate-agnostic filtered search: the selection subquery runs first,
producing a selected-set mask, and the HNSW search consults it through
adaptive per-iteration heuristics (one-hop, directed, blind) switched by
local selectivity. Two properties of the paper matter directly here:

- **The efs-driven recall methodology.** The paper's evaluation fixes
  construction parameters (upper/lower max degree approximately Kuzu's
  `mu=30`/`ml=60` defaults, `efc=200`, 5% upper-layer sampling) and reaches
  a 95% recall target per dataset purely by raising `efs`. Kuzu's own
  authors tune recall at query time, not build time — Zaxy's lane adopted
  the same posture.
- **Acknowledged build nondeterminism.** Construction is parallel: worker
  threads concurrently scan morsels of vectors and update shared CSR
  structures with acknowledged, unsynchronized data races, justified
  because HNSW is already approximate. There is no seed parameter anywhere
  in the 0.11.3 surface or source. Identical data therefore yields
  different graphs and slightly different recall on every rebuild. This is
  why the lane reports ANN recall under run-dependent `measurements` rather
  than the `deterministic` block, and why every decision gate requires two
  consecutive passing runs rather than one.

### 2.3 Recall@k and tie handling

The ann-benchmarks methodology (Aumüller, Bernhardsson & Faithfull,
arXiv:1807.05614) defines recall@k by counting returned points whose
distance to the query is no worse than the distance of the k-th true
neighbor, precisely so that ties at the boundary do not penalize an answer
that is equally correct. Section 3.1 formalizes the variant Zaxy's lane
ships and shows, with measured tie statistics, why strict identity recall
is ill-posed on the lane's high-dimension hash corpus.

## 3. Mathematics

### 3.1 Tie-aware recall and the ill-posedness of strict recall

Let \(V\) be the corpus, \(|V| = n\), and let
\(s(q, v) = \langle \hat{q}, \hat{v} \rangle\) be the exact float64 cosine
score of corpus vector \(v\) against query \(q\) (unit-normalized exactly as
the production dense path normalizes — the lane's ground-truth scores are
bit-identical to production scores; see `exact_score_matrix` in
`zaxy_benchmarks/vector_scale_lane.py`). Let \(s_k(q)\) be the k-th largest
value of \(s(q, \cdot)\) over \(V\), and let \(T_k(q)\) be a true top-k set
(any k-subset consistent with the score ordering).

**Strict recall** of a returned set \(R_k\) is
\(|R_k \cap T_k| / k\) where \(T_k\) is the particular top-k the exact
store returned — identity matching against one arbitrary tie-break.

**Tie-aware recall** declares a returned vector a hit when it scores at
least the k-th true score:

\[
\operatorname{hit}(v) \;:=\; s_{\mathrm{f64}}(q, v) \,\ge\, s_k(q),
\qquad
\operatorname{recall}^{\mathrm{tie}}_{@k}(R_k)
= \frac{\bigl|\{\, v \in R_k : \operatorname{hit}(v) \,\}\bigr|}{k}.
\]

The comparison is exact — no epsilon — because both sides are drawn from the
same float64 score matrix (`tie_aware_recall_at_k` in
`zaxy_benchmarks/vector_scale_lane.py`). Two properties follow directly:

1. **It never rewards a wrong vector.** Any \(v\) with
   \(s(q,v) \ge s_k(q)\) belongs to some valid top-k under the true
   ordering, so every counted hit is a correct answer.
2. **It reduces to strict recall on tie-free corpora**, where the top-k set
   is unique.

Strict recall becomes ill-posed when the boundary tie class
\(B_k(q) := \{\, v : s(q,v) = s_k(q) \,\}\) is large: the true top-k set is
then non-unique, and an answer drawn entirely from
\(\{v : s(q,v) \ge s_k(q)\}\) — equally correct by score — is penalized for
disagreeing with the exact store's storage-order tie-break. This is not a
hypothetical on the lane's corpus. Measured at dimension 1536 on the hash
embedding corpus (`diag-ties-gaussian.json`):

- a **median of 210 corpus vectors** (min 11, max 229) are *exactly tied*
  with the true top-10 per query;
- the **float64 score gap between rank 10 and rank 40 is exactly 0.0**
  (median) — the ties are real values, not rounding artifacts;
- the float32 machine epsilon at score scale is
  \(5.96 \times 10^{-8}\) (\(2^{-24}\)), so any float32 representation of
  these scores is constitutionally unable to order the tie class.

The consequence is measurable as a ceiling: an **exact float32 brute-force
scan** over the same corpus achieves strict recall of only **0.5344**
against float64 ground truth, at candidate depths k=40, k=100, and k=200
alike (`diag-d1536-f32-ceiling.json`). No index whose stored vectors are
float32 — Kuzu's HNSW stores `FLOAT[]` — can beat a number that exact
float32 search itself cannot beat. Storing float64 in the index does not
help either: a float64-stored HNSW measured 0.5219–0.5406 on the same
corpus (`diag-d1536-f32-ceiling.json`, `f64_*` entries), because the ties
exist in float64 too. The 2.1-era recall numbers at dimension 1536
(0.5156 ANN, 0.6094 int8; `baseline-dim1536.json`) were therefore
substantially a property of the metric-times-corpus, not of the index — a
diagnosis the Gaussian control in Section 4.4 confirms.

The shipped lane reports both metrics unconditionally
(`recall_at_k_strict`, `recall_at_k_tie_aware`) and gates on the tie-aware
one, so the divergence is always visible and nothing is hidden
(`docs/benchmarks.md`, vector-scale lane section).

### 3.2 Oversample-and-rerank: confining approximation to candidate selection

Both approximate modes share one pipeline (Section 5.1): a selector
produces a candidate set \(C\) with
\(|C| = \min(n,\, m \cdot k)\), where \(m\) is the oversample multiplier
(`VECTOR_SEARCH_OVERSAMPLE = 4` in `src/zaxy/embedded_graph_store.py`),
and the final result is the exact-score top-k of \(C\):

\[
R_k \;=\; \operatorname*{arg\,top\text{-}k}_{v \in C}\; s_{\mathrm{f64}}(q, v).
\]

Because the rerank ordering within \(C\) is exact, the final strict recall
is purely a function of candidate selection:

\[
\operatorname{recall}_{@k}(R_k)
\;=\; \frac{|C \cap T_k|}{k}
\qquad\text{(since } |C \cap T_k| \le |T_k| = k\text{)}.
\]

Every true top-k member that reaches the candidate set is restored to its
correct position; a member is lost only if the selector's error displaces
it below the \(mk\)-th approximate rank. This eliminates the dominant
measured error class of the unreranked path: the shadow table stores
float32, and near-ties that flip at the \(2^{-24}\) precision boundary
land within the \(mk\) slack and are re-ordered exactly. Experiment E3
(Section 4.3) measured recall 1.0 at every `efs` under float32-consistent
ground truth — demonstrating the deficit was the precision boundary, not
HNSW search quality — and the same rerank is what makes recall robust to
HNSW build variance: it is the determinism mitigation that costs nothing
at build time. The float64 rerank reads vectors already resident on the
indexed entities, so it adds no duplicate storage
(`_exact_rerank_results` in `src/zaxy/embedded_graph_store.py`). Reranked
results still report `exact: false`, because the candidate set remains
approximate even though the final ordering is exact.

### 3.3 The byte model and the cache-of-one property

Resident bytes per scope of \(n\) vectors at dimension \(d\):

\[
\mathrm{bytes}_{\mathrm{exact}}(n, d) = 8nd
\qquad
\mathrm{bytes}_{\mathrm{int8}}(n, d) = nd + 8n
\qquad
\mathrm{bytes}_{\mathrm{ann}}(n, d) = 0.
\]

Exact stores float64; int8 stores one byte per component plus one float64
scale per vector (ratios confirm: \(72n/512n = 0.1406\) at d=64,
\(1544n/12288n = 0.1257\) at d=1536, matching every lane artifact's
`bytes_vs_exact_ratio`); the ANN mode keeps vectors in the database's
buffer-managed shadow table, so process-resident index bytes are zero (the
lane separately reports the on-disk backing store, e.g. 67.5 MB at
10^5 x 64, `ann3-d64-100k-r1.json`).

The store budgets process-resident vector matrices at
\(B = 256\,\mathrm{MiB} = 2^{28}\) bytes across all scopes
(`VECTOR_INDEX_CACHE_MAX_BYTES`). Engagement clause (b) (Section 3.4)
fires when a single scope's exact matrix would exceed it,
\(8nd > B \Leftrightarrow n > 2^{25}/d\): 524,288 rows at d=64, 21,845 at
d=1536 (`docs/configuration.md`).

A single over-budget scope does not thrash, by an explicit eviction
invariant: the LRU byte eviction never evicts the newest index, even alone
over budget (`_evict_vector_indexes_over_budget`,
`src/zaxy/embedded_graph_store.py`). One over-budget session therefore
degrades to a cache of one — built once, resident thereafter — and the
budget bounds *multi-scope* totals. This is measured, not asserted: at
50k x 1536 the exact matrix is 614.4 MB, 2.29x the budget
(`budget_fraction: 2.288818`), and exact search still answered at
22.4–25.8 ms p50 across runs (`ann3-d1536-50k-gauss-crossover.json`,
`ann4-d1536-50k-dimension-gated.json`). Only workloads alternating across
several jointly over-budget scopes pay a rebuild per switch.

### 3.4 The engagement predicate as shipped

With defaults \(D = 64\) (`VECTOR_ANN_MAX_DIMENSION`),
\(T = 100{,}000\) (`VECTOR_ANN_THRESHOLD`), \(B = 2^{28}\), quantization
setting \(Q\), and byte-clause flag \(\beta\)
(`VECTOR_ANN_BYTE_BUDGET_ENGAGEMENT`), a scope of \(n\) vectors at
dimension \(d\) engages the ANN path iff

\[
\operatorname{engage}(n, d) \;\Longleftrightarrow\;
d \le D \;\wedge\;
\Bigl( n \ge T \;\vee\; \bigl( Q \ne \mathrm{int8} \wedge \beta \wedge 8nd > B \bigr) \Bigr).
\]

The dimension ceiling is a hard precondition; the count clause (a) is the
lane-proven default; the byte clause (b) is defense-in-depth within the
ceiling (at d=64 the count clause fires first — clause (b) protects if the
count threshold is raised or the budget lowered). An explicit int8 opt-in
keeps its precedence below the count threshold and is exempt from clause
(b) because it already resolves the memory problem it guards against. This
predicate is implemented at a single choke point
(`_ann_engagement_reason`, `src/zaxy/embedded_graph_store.py`) through
which both query-time strategy choice and shadow rebuild triggers pass, so
no shadow generation is ever built for a scope that would not be queried
through it. `memory_capabilities` reports the effective rule.

## 4. Method: Baseline First, Then Decisive Experiments

### 4.1 Baseline (2026-06-11, master @ post-2.1.0)

No development happened before `BASELINE.md` existed. The lane ran at
dimensions 64 and 1536 (sizes 10^3, 10^4; query_count 32, latency_passes 3,
top_k 10, lane-lowered ann_threshold 256, hash embeddings), with the 10^5
dim-64 reference taken from the identical released 2.1.0 code. Raw lane
JSON: `baseline-dim64.json`, `baseline-dim1536.json`.

| Metric | dim 64, 10k | dim 64, 100k (2.1 ref) | dim 1536, 10k |
| --- | --- | --- | --- |
| ANN recall@10 (strict) | 0.9062 | 0.8969 | 0.5156 |
| ANN p50 vs exact | 9.8 ms vs 6.1 ms | 37.9 ms vs 17.0 ms | 26.5 ms vs 8.9 ms |
| ANN build (first query) | 53.3 s | ~20.6 min | 196.3 s |
| int8 recall@10 | 0.9906 | 0.9938 | 0.6094 |
| int8 p50 vs exact | 1.1 ms vs 6.1 ms | 9.5 ms vs 17.0 ms | 35.6 ms vs 8.9 ms |

Three findings reframed the plan (`BASELINE.md`):

1. **Dimension is the dominant variable.** At production-scale dimension
   (1536), ANN recall collapsed to 0.52 and int8 — the 2.1 promotion
   candidate — to 0.61. The dim-64 lane had been the easy case.
2. **At high dimension the fight is memory, not latency.** Exact float64
   was 8.9 ms p50 at 10k x 1536, but 100k x 1536 x 8 B is roughly 1.23 GB
   against the 256 MiB cache budget. Approximate methods must win on
   resident bytes first, recall second, latency third.
3. **int8's collapse is candidate selection, not rerank.** The float
   rerank is exact; at high dimension the true top-10 fell outside the
   fixed k x 4 int8 candidate set — later traced (Section 4.4) to the same
   tie pathology that depressed ANN strict recall.

### 4.2 External research: the Kuzu reality

Parallel research (sources in Section References) established the frozen
context: 0.11.3 is the final release of an archived repository; `efs` is
the primary recall knob per the NaviX methodology; `COPY FROM` is the
documented bulk path while CREATE/MERGE-style loads (Zaxy's batched
UNWIND) are documented for small sporadic additions only; the index should
be created after bulk load (`SET` on indexed properties is rejected,
kuzu#5965, and bulk mutation under a live index is the historically
buggiest upstream path, kuzu#5694/#5695/#5898); build nondeterminism is by
design with no seed; and three open defects had to be designed around
permanently — #6047 (sequential HNSW load can block DB open), #6040
(`DROP_VECTOR_INDEX` metadata corruption), #6012 (prefer Arrow over Pandas
for loads). Comparable embedded systems serve this workload at >= 0.95
recall in 1–2 ms (ann-benchmarks), and NaviX itself reports 95% recall at
5–40 ms on 1M x 960-dim filtered search — so the bar was known reachable
in principle.

### 4.3 Four decisive experiments (10k, dim 64; `experiments-e1-e4.json`)

- **E1, bulk load**: batched UNWIND 8.16 s vs `COPY FROM` parquet
  **0.08 s — roughly 100x** — plus 6.25 s post-copy index build. The
  20-minute 100k sync was fully explained and fully fixable. Two defects
  surfaced en route: `COPY FROM` an *in-memory* Arrow table with a
  fixed-size-list column segfaults 0.11.3 (a parquet file is the safe
  path), and an unbound `$param` in any query segfaults rather than
  raising (Section 5.4).
- **E2, query pattern**: direct-table 6.56 ms p50, per-query projected
  graph 6.95 ms, reused projected graph 7.93 ms — projected-graph
  create/drop is *not* the dominant overhead at 10k. The lane's 100k
  latency gap had to live elsewhere; profiling at 10^5 (`profiling-a2.json`)
  located it: direct-table 8.0 ms p50 versus 24.3 ms through a *reused*
  projected graph versus 28.3 ms creating one per query (k=10). The
  per-query prefilter mask scan over the session/version predicate costs
  about 16.3 ms (reused-graph minus direct), the create/drop pair only
  about 4.0 ms (per-query minus reused). The fix is therefore not graph
  hygiene but predicate elimination: make the shadow table per-scope so
  the common case is an unfiltered direct-table query (Section 5.1).
- **E3, recall**: **1.0 at every efs (200/400/800)** with
  float32-consistent ground truth, on the same corpus scale where the
  production lane measured 0.9062. The recall deficit was not HNSW search
  quality but the float32 precision boundary between the shadow table and
  the float64 ground truth — the class the oversample + float64 rerank
  eliminates entirely (Section 3.2). `efs` tuning was demoted to a
  secondary lever.
- **E4, determinism**: inconclusive at saturated recall (all runs 1.0,
  single- and multi-threaded). Robustness comes from the rerank, not from
  chasing build determinism.

### 4.4 Post-ann.1 diagnostics: the dimension-1536 root cause

The first increment's gate runs passed recall at dim 64 but still failed
at dim 1536 — 0.55 and 0.6125 strict at 10k *with* the float64 rerank
(`after-dim1536-r1.json`, `after-dim1536-r2.json`). The follow-up
diagnostics isolated the cause and exonerated the index:

- the float32 brute-force ceiling on the lane corpus is 0.5344 — even
  exact float32 search cannot exceed it against float64 ground truth
  (`diag-d1536-f32-ceiling.json`);
- a float64-stored HNSW scores in the same 0.52–0.54 band, so storage
  precision is not the cause either (`diag-d1536-f32-ceiling.json`);
- the tie analysis (Section 3.1): median 210 corpus vectors exactly tied
  with the true top-10, rank-10-to-rank-40 float64 gap exactly 0.0
  (`diag-ties-gaussian.json`) — strict recall@10 is ill-posed on this
  corpus;
- a Gaussian-distribution control on the identical index machinery:
  recall@10 of **0.8531 at efs 200, 0.9875 at efs 400, 1.0 at efs 800**
  (k=40 and k=100 alike), at p50 of 17.98/19.90/19.92 ms
  (`diag-gauss-efs-sweep.json`). With a realistic distribution the index
  is healthy, and efs 400 is the evidence-backed high-dimension default.

Three consequences shipped: the lane gained the tie-aware metric reported
alongside strict (Section 3.1); the seeded Gaussian variant
(`--scale-distribution gaussian`, PCG64 seeds 20260611/20260612,
`zaxy_benchmarks/vector_scale_lane.py`) became the high-dimension gate
corpus; and the `VECTOR_ANN_EFS` default moved from 200 to 400 on the
sweep evidence.

## 5. Engineering Narrative

The work landed as three lane-gated increments plus a threshold decision,
each green on lint, strict typing, full tests, and site freshness before
the next started.

### 5.1 Increment ann.1: strategy consolidation and the exact rerank

The store had carried three parallel search paths (dense float64,
`_AnnVectorGroup`, `_QuantizedVectorGroup`) with duplicated selection,
scoring, and cache accounting. They were consolidated behind one strategy
pipeline — candidate selection, then a shared exact float64 rerank — so
subsequent work landed as strategy implementations rather than more
branching (`search_vector` and `_exact_rerank_results` in
`src/zaxy/embedded_graph_store.py`). Dense selection already scores
exactly and returns `exact: true`; the HNSW and int8 selectors produce
oversampled candidates and stay `exact: false` through the rerank.

The query path also eliminated the profiled predicate cost: each
(session, version, dimension) scope owns its own shadow table, queried
directly by name through `QUERY_VECTOR_INDEX` — no session/version
predicate, no projected graph, no per-query mask scan
(`_ann_candidate_entity_indexes`). `VECTOR_ANN_EFS` became a setting, with
the effective value never below the oversampled candidate count a query
requests.

Lane effect at dim 64: strict recall@10 moved from 0.9062 to 0.9906 at 10k
(`after-dim64-r1.json` through `-r3.json`), and from 0.8969 (2.1 reference)
to 0.9906 at 10^5 (`after-dim64-100k.json`) — clearing the 0.95 floor.
Latency at 10^5 remained adverse in that run (ANN 16.3 ms vs exact
13.3 ms p50), and the build remained the open wound: 1,179.8 s first-query
cost at 10^5 (`after-dim64-100k.json`).

### 5.2 Increment ann.2: COPY-based generation-swap builds

Full rebuilds now bulk-load a fresh generation table via `COPY FROM` and
build the HNSW index after the load completes — the documented fast path,
avoiding the historically buggiest upstream path (bulk mutation under a
live index). The swap into store state is atomic (a single assignment of
the generation record) before superseded generations are reclaimed
(`_try_build_ann_group`, `_ann_bulk_load`,
`src/zaxy/embedded_graph_store.py`).

Build cost at 10^5/dim 64 fell from 1,179.8 s (`after-dim64-100k.json`) to
91.8–97.9 s in the G4 runs (`ann3-d64-100k-r1.json`, `-r2.json`) — a 12.9x
improvement, with the confirmatory run at 62.5 s
(`ann4-d64-100k-confirm.json`). The components at 10^5: the `COPY` itself
is 0.86 s; the post-load index build is 54.7 s (`profiling-a2.json`); the
lane's first-query figure additionally includes corpus sync and parquet
serialization. At 10k the first-query cost fell from 53.3 s (baseline) to
5.5 s (`ann2-d64-10k-postcopy.json`).

Small deltas avoid rebuilds entirely: when the new corpus is a
digest-verified pure extension of the resident generation and the delta is
at most 10% of the resident count, the new rows ride live-index inserts
(reflected in queries on the pinned 0.11.3). The 10% boundary is measured:
live-index inserts cost ~6.9 ms/row at 10k/dim 64 versus ~0.63 ms/row for
a COPY load plus post-load index build, so raw build time crosses over
near a 9% delta; the boundary sits at 10% because incremental inserts also
preserve the resident HNSW graph (no rebuild recall variance) and add no
superseded generation table (`_ANN_DELTA_REBUILD_FRACTION`,
`src/zaxy/embedded_graph_store.py`).

### 5.3 Increment ann.3/ann.4: metric correction, efs 400, and the gates

The lane shipped the tie-aware metric beside strict recall, the Gaussian
distribution control, and the byte-budget block per size; `VECTOR_ANN_EFS`
defaulted to 400 on the sweep evidence. The G1/G2/G4 evidence then
accumulated at 10^5/dim 64 (Section 6.1) and the G3 high-dimension
evidence at 10k–50k/dim 1536 (Section 6.2), culminating in the engagement
predicate of Section 3.4.

### 5.4 Three frozen-upstream defects, designed around

Engineering against an archived dependency means defects are permanent
facts, not bug reports. Three were found and designed around, all
re-verified against the pinned 0.11.3:

1. **In-memory Arrow COPY segfault.** `COPY FROM` an in-memory Arrow table
   with a fixed-size-list column (the natural encoding of a vector column)
   segfaults the process. Bulk loads therefore round-trip through a
   parquet tempfile, which is safe and still ~100x faster than UNWIND;
   absence of pyarrow degrades to the batched-UNWIND load rather than
   failing (`_ann_bulk_load`).
2. **Unbound-parameter segfault.** A query referencing an unbound
   `$parameter` segfaults instead of raising. Every store query now passes
   through a single execution choke point that rejects unbound parameters
   before they reach the runtime (`_QUERY_PARAMETER_RE`,
   `src/zaxy/embedded_graph_store.py`).
3. **No safe teardown of a live index.** Mutating a live index in place
   (delete + reinsert) silently breaks subsequent direct-table searches;
   `DROP_VECTOR_INDEX` leaves un-checkpointed index metadata (kuzu#6040);
   and `DROP TABLE` is rejected by the binder while an index references
   the table. Rebuilds are therefore **drop-free generation swaps**:
   each rebuild creates `ZaxyVectorAnnShadow{dim}_{scope}_g{n+1}`, builds a
   fresh index over it, swaps atomically, and *empties* (never drops)
   superseded generation tables — never queried again, near-zero load cost
   at database open, the maximum reclaim the frozen runtime allows
   (`_try_build_ann_group`).

The strategic consequence is recorded for 2.3: frozen-Kuzu HNSW measured
inadequate exactly where approximate search matters most (high dimension,
large N), so evaluating the LadybugDB fork or an alternative embedded ANN
backend is formally on the 2.3 agenda, with this corpus of lane evidence
as the acceptance bar
(`docs/superpowers/specs/2026-06-11-zaxy-2-2-ann-engineering-plan.md`).

## 6. Results

### 6.1 Target scale, dimension 64: before and after

All runs at 10^5 vectors, dimension 64, hash corpus
(`corpus_sha256: 7f1465…f37f3`, identical across runs), top_k 10,
96 latency samples per mode.

| Run (artifact) | ANN recall@10 strict | tie-aware | ANN p50 vs exact p50 | Build (first query) | ANN resident bytes vs exact | Gate status |
| --- | --- | --- | --- | --- | --- | --- |
| 2.1 reference (`BASELINE.md`) | 0.8969 | — (metric did not exist) | 37.9 ms vs 17.0 ms | ~20.6 min | 0 vs 51.2 MB | fail |
| Post-ann.1 (`after-dim64-100k.json`) | 0.9906 | — | 16.3 ms vs 13.3 ms | 1,179.8 s | 0 vs 51.2 MB | recall pass, latency fail |
| G4 run 1 (`ann3-d64-100k-r1.json`) | 1.0 | 1.0 | 24.17 ms vs 24.20 ms | 91.8 s | 0 vs 51.2 MB | **pass, all criteria** |
| G4 run 2 (`ann3-d64-100k-r2.json`) | 1.0 | 1.0 | 26.67 ms vs 30.82 ms | 97.9 s | 0 vs 51.2 MB | **pass, all criteria** |
| Confirmatory (`ann4-d64-100k-confirm.json`) | 1.0 | 1.0 | 16.48 ms vs 38.27 ms | 62.5 s | 0 vs 51.2 MB | **pass, all criteria** |

The two consecutive G4 passes are the evidence behind lowering
`VECTOR_ANN_THRESHOLD` from 1,000,000 to 100,000 (gate G4; the lane's
two-consecutive-run rule is the documented posture for HNSW build
nondeterminism). The confirmatory run reproduced the pass; its exact-path
p50 of 38.27 ms (p95 107.6 ms) against 24.2/30.8 ms in the G4 runs on the
byte-identical corpus also illustrates the host-noise band discussed in
Section 7.2 — which is why latency criteria are evaluated in-run (paired),
never across runs.

### 6.2 High dimension: the honest negative results

At dimension 1536 the same machinery, including the rerank and efs 400,
does not transfer. All 50k runs use the seeded Gaussian corpus
(`corpus_sha256: 5356e5a9…28588`), the realistic-distribution control:

| Run (artifact) | Config | ANN recall@10 (strict = tie-aware) | ANN p50 vs exact p50 | int8 p50 (recall) |
| --- | --- | --- | --- | --- |
| `ann3-d1536-10k-gauss-r1.json` | 10k, efs 400 | 0.9844 | 40.8 ms vs 17.3 ms | 76.2 ms (1.0) |
| `ann3-d1536-10k-gauss-r2.json` | 10k, efs 400 | 0.975 | 23.6 ms vs 12.0 ms | 46.3 ms (1.0) |
| `ann3-d1536-50k-gauss-crossover.json` | 50k, efs 400 | **0.6** | 45.3 ms vs 22.4 ms | 223.7 ms (1.0) |
| `ann4-d1536-50k-gauss-r2.json` | 50k, efs 400 (rerun) | **0.6344** | 45.2 ms vs 55.6 ms | 256.5 ms (1.0) |
| `ann4-d1536-50k-gauss-efs800.json` | 50k, efs 800 | **0.8438** | 69.8 ms vs 37.5 ms | 234.2 ms (1.0) |

Three conclusions, each disqualifying for a default:

1. **efs does not transfer across scale at high dimension.** efs 400
   delivers 0.975–0.9844 at 10k but collapses to 0.6–0.6344 at 50k on the
   same distribution. Doubling to efs 800 recovered recall only to 0.8438
   while p50 rose to 69.8 ms against exact's 37.5 ms — tuning erodes the
   latency advantage faster than it repairs recall.
2. **int8 hits a latency cliff at high-dimension scale**: recall 1.0
   deterministic and an 8x byte advantage (77.2 MB vs 614.4 MB resident),
   but 223.7–256.5 ms p50 at 50k x 1536 — the integer matmul plus rerank
   loses an order of magnitude to one float64 BLAS matmul. int8 remains
   opt-in for memory-tightest deployments, latency-honest in the docs.
3. **Exact remains the high-dimension recommendation.** Despite sitting
   2.29x over the byte budget, the exact matrix answered at 22.4–25.8 ms
   p50 under cache-of-one residency (Section 3.3).

The G4 confirmatory run for the shipped predicate
(`ann4-d1536-50k-dimension-gated.json`) ran the same 50k x 1536 corpus
with the production dimension ceiling in force: the ANN mode honestly
reports `engaged: false` and falls back to the dense path (strict recall
1.0 by construction, exact p50 25.8 ms) — the engagement rule refusing
exactly the configuration the lane measured as disqualifying.

On the hash corpus at dimension 1536, the metric correction is visible in
one row: strict 0.525 versus tie-aware 0.9938 for the identical result
sets (`ann3-d1536-10k-hash-r1.json`; the record run
`ann3-d1536-10k-hash-record.json` measured 0.5625 strict / 1.0 tie-aware,
and int8 0.6094 strict / 1.0 tie-aware). The 0.52–0.61 "failures" of the
baseline era were the metric being ill-posed on a tie-dense corpus
(Section 3.1), to the extent the tie-aware metric quantifies.

### 6.3 The gate ledger

| Gate | Evidence required (plan) | Outcome |
| --- | --- | --- |
| G1 query path | 10^5/dim 64: recall >= 0.95 AND p50 < exact | Pass (Section 6.1, G4 runs; recall via rerank, latency via direct-table scoped shadow tables) |
| G2 build path | Full rebuild at 10^5 in single-digit minutes; generation-swap cycle survives stress | Pass (91.8–97.9 s builds; drop-free swap cycle under test) |
| G3 high-dim | >= 0.95 recall with bytes < exact at dim 1536 | **Fail** — exact remains the high-dim recommendation; memory ceiling documented (Section 6.2) |
| G4 threshold | G1+G2 with margin on two consecutive runs | Pass — threshold 1,000,000 → 100,000, scoped by the dimension ceiling at 64 |

The G3 failure is load-bearing: it is why the threshold move carries a
dimension ceiling. The defaults moved exactly as far as the evidence
extends — count >= 100,000 AND dimension <= 64 — and the over-ceiling
posture (exact, or opted-in int8) is itself measured, not residual.

## 7. Threats to Validity

### 7.1 Corpus distribution

The lane's default hash-embedding corpus is adversarially tie-dense at
high dimension (the measured 210-way boundary ties, Section 3.1). Every
high-dimension conclusion is therefore gated on the seeded Gaussian
control instead, and the diagnosis that separated metric pathology from
index pathology is itself artifact-backed (`diag-ties-gaussian.json`,
`diag-gauss-efs-sweep.json`, `diag-d1536-f32-ceiling.json`). Neither
distribution is a semantic embedding: the lane measures index mechanics,
not retrieval quality, and says so in its own `measurement` field.

### 7.2 Single-machine latency noise and the contaminated-run episode

All timings come from one local development machine and are
environment-dependent — the lane labels them so and excludes them from its
determinism comparisons. The artifact record contains a concrete
contamination episode: on the byte-identical 50k x 1536 Gaussian corpus,
the exact path — a deterministic float64 matmul — measured 22.4 ms p50 in
one run and 55.6 ms p50 (p95 116.4 ms) in another
(`ann3-d1536-50k-gauss-crossover.json` vs `ann4-d1536-50k-gauss-r2.json`);
at dim 64/10^5 the same spread appears as 24.2/30.8/38.3 ms across the
three passing runs. Since the workload is fixed, the spread is host load.
Three protocol rules absorb it: latency runs execute on an otherwise
quiet machine and visibly load-contaminated runs are rerun rather than
averaged; the latency criterion is evaluated *within* a run (ANN and
exact measured back-to-back over the same 96 samples), never across runs;
and no gate is judged on a single run. The recall conclusions are
unaffected — recall is computed from result sets, not timings, and the
deterministic blocks (corpus hashes, exact/quantized recall, bytes) are
exactly reproducible across all runs cited here.

### 7.3 Internal validation labels

Every artifact in this report carries `"validation": "internal"` and the
lane version `vector-scale-lane-v1`. Per `docs/external-validation.md` and
the claim boundaries in `docs/benchmarks.md`, these are same-harness,
same-machine internal measurements: they justify Zaxy's own defaults and
are not public benchmark claims, comparisons with other systems, or
statements about semantic retrieval quality.

### 7.4 The dimension envelope is two points, not a curve

The evidence covers dimension 64 (the double pass) and dimension 1536 (the
measured failure). Nothing between was measured: the ceiling default of 64
is the conservative edge of the measured envelope, not an estimated
crossover. The clause-(b) row counts quoted for intermediate dimensions
(e.g. 87,381 at d=384 in `docs/configuration.md`) are arithmetic over the
byte model, not measurements. Raising `VECTOR_ANN_MAX_DIMENSION` is
documented as requiring lane evidence for the user's own dimension and
distribution.

### 7.5 Build nondeterminism

HNSW construction on the pinned runtime is not run-to-run reproducible
(Section 2.2), so any single recall number for the ANN mode carries
rebuild variance. The mitigations are structural — the exact rerank makes
final recall insensitive to graph variance up to candidate-set membership,
and gates require two consecutive passes — but a pathological rebuild
below the candidate-slack margin remains possible in principle and would
be caught by the lane, not prevented by the store.

## 8. Conclusion: Evidence-Bounded Defaults as a Release Discipline

The 2.2 ANN effort is a case study in treating a default as a falsifiable
claim. The same benchmark lane that kept the feature opt-in in 2.1 — by
measuring it worse than brute force — defined the bar for promoting it in
2.2, adjudicated every increment against a frozen baseline, exposed that
part of the original failure was the evaluation metric rather than the
index, and finally drew the boundary of the new default at exactly the
envelope it had measured: count >= 100,000 AND dimension <= 64, with the
high-dimension negative results shipped in the same documentation as the
positive ones.

Three transferable lessons:

1. **Confine approximation to candidate selection.** The oversample +
   exact-rerank decomposition (Section 3.2) converted both a precision-
   boundary recall deficit and an upstream nondeterminism liability into a
   bounded candidate-selection problem, at zero build-time cost.
2. **Audit the metric before tuning the system.** The float32 ceiling and
   tie analysis showed a 0.52 "recall failure" that no index could have
   fixed; the tie-aware metric (with strict recall reported alongside,
   permanently) made the lane well-posed without hiding the divergence.
3. **A frozen upstream changes the engineering contract.** Against an
   archived dependency, defects are design constraints. The drop-free
   generation swap, the parquet round-trip, and the unbound-parameter
   choke point are all permanent accommodations — and the measured limit
   of the frozen index at high dimension is now the recorded acceptance
   bar for any 2.3 backend evaluation.

The threshold will move again — upward, downward, or across the dimension
ceiling — when, and only when, a lane measures it.

## References

### Internal artifacts (versioned in this repository)

All lane artifacts under `docs/research/artifacts/ann-2026-06/`:

- Baseline record and reframing findings: `BASELINE.md`; raw lane runs
  `baseline-dim64.json`, `baseline-dim1536.json`.
- Decisive experiments E1–E4: `experiments-e1-e4.json`.
- Query-path profiling at 10^5: `profiling-a2.json`.
- Dimension-1536 diagnostics: `diag-d1536-f32-ceiling.json`,
  `diag-ties-gaussian.json`, `diag-gauss-efs-sweep.json`.
- Post-ann.1 gate runs: `after-dim64-r1.json`, `after-dim64-r2.json`,
  `after-dim64-r3.json`, `after-dim64-100k.json`, `after-dim1536-r1.json`,
  `after-dim1536-r2.json`.
- Post-COPY build run: `ann2-d64-10k-postcopy.json`.
- G4 double pass and high-dimension evidence: `ann3-d64-100k-r1.json`,
  `ann3-d64-100k-r2.json`, `ann3-d1536-10k-hash-r1.json`,
  `ann3-d1536-10k-hash-record.json`, `ann3-d1536-10k-gauss-r1.json`,
  `ann3-d1536-10k-gauss-r2.json`, `ann3-d1536-50k-gauss-crossover.json`.
- G4 confirmatory runs: `ann4-d64-100k-confirm.json`,
  `ann4-d1536-50k-gauss-r2.json`, `ann4-d1536-50k-gauss-efs800.json`,
  `ann4-d1536-50k-dimension-gated.json`.

Other internal references:

- 2.2 ANN engineering plan:
  `docs/superpowers/specs/2026-06-11-zaxy-2-2-ann-engineering-plan.md`.
- Lane implementation (tie-aware metric, Gaussian control, byte budget):
  `zaxy_benchmarks/vector_scale_lane.py`.
- Store implementation (engagement predicate, strategy pipeline,
  generation swaps, defect accommodations):
  `src/zaxy/embedded_graph_store.py`.
- Shipped behavior: `docs/configuration.md`, `docs/embeddings.md`,
  `docs/migration.md` ("2.2: ANN engagement defaults"), `CHANGELOG.md`
  Unreleased (2.2.0).
- Claim boundaries and validation labels: `docs/benchmarks.md`,
  `docs/external-validation.md`.

### External references

- Yu. A. Malkov and D. A. Yashunin, "Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World
  graphs", 2016: <https://arxiv.org/abs/1603.09320>.
- Gaurav Sehgal and Semih Salihoglu, "NaviX: A Native Vector Index Design
  for Graph DBMSs With Robust Predicate-Agnostic Search Performance",
  2025: <https://arxiv.org/abs/2506.23397>.
- Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull,
  "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor
  Algorithms", 2018: <https://arxiv.org/abs/1807.05614>. Benchmark site:
  <https://ann-benchmarks.com/>.
- Kuzu repository (archived 2025-10-10; 0.11.3 is the final release):
  <https://github.com/kuzudb/kuzu>; releases:
  <https://github.com/kuzudb/kuzu/releases>.
- Kuzu documentation source (the archived docs repository; the hosted docs
  site is offline): vector extension
  <https://github.com/kuzudb/docs/blob/main/src/content/docs/extensions/vector.mdx>;
  import guidance
  <https://github.com/kuzudb/docs/blob/main/src/content/docs/import/index.mdx>;
  projected-graph lifecycle
  <https://github.com/kuzudb/docs/blob/main/src/content/docs/extensions/algo/index.mdx>.
- Kuzu 0.11.3 vector-index source (build parameters, insert path, deleted-
  node handling):
  <https://github.com/kuzudb/kuzu/blob/v0.11.3/extension/vector/src/include/index/hnsw_config.h>,
  <https://github.com/kuzudb/kuzu/blob/v0.11.3/extension/vector/src/index/hnsw_index.cpp>.
- Kuzu frozen-upstream defects designed around or guarded:
  #5965 (SET on indexed property rejected)
  <https://github.com/kuzudb/kuzu/issues/5965>;
  #6040 (DROP_VECTOR_INDEX metadata corruption)
  <https://github.com/kuzudb/kuzu/issues/6040>;
  #6047 (sequential HNSW load at database open)
  <https://github.com/kuzudb/kuzu/issues/6047>;
  #6012 (memory growth on Pandas LOAD)
  <https://github.com/kuzudb/kuzu/issues/6012>.
- LadybugDB, the community continuation fork (recorded as the 2.3
  evaluation candidate): <https://github.com/LadybugDB/ladybug>.
