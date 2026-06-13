# Zaxy 2.3 Promotion Pass — Evidence Plan

Date: 2026-06-11. Author: benchmarks research pass (read-only against
`/home/cheapseatsecon/Projects/Personal/zaxy`, master @ post-2.2).
Scratch artifacts: `/tmp/zaxy-23-research/` (`int8_bench.py`,
`int8_bench_results.json`, `int8_bench_e.py`, `int8_bench_e_results.json`).

Three deferred promotions from the 2.2 plan
(`docs/superpowers/specs/2026-06-11-zaxy-2-2-ann-engineering-plan.md`):
int8 quantization default, `ENCODING_GATE_ENABLED` flip, FoK organic
calibration.

---

## 1. int8 latency cliff — diagnosis and prototype verdict

### Diagnosis (profiled, deterministic seed 20260611, numpy 2.4.4 / OpenBLAS,
Gaussian unit corpora matching the lane's gaussian distribution, dim 1536,
limit 10, oversample 4)

The current quantized selector
(`_quantized_candidate_entity_indexes`, `src/zaxy/embedded_graph_store.py:2499`):

```python
integer_scores = group.matrix.astype(np.int32) @ quantized_query.astype(np.int32)
```

Component breakdown per query:

| Component | 10k x 1536 | 50k x 1536 |
| --- | --- | --- |
| `matrix.astype(np.int32)` (fresh 4nd temp) | 26.4 ms | **125.3 ms** (307 MB temp) |
| int32 matmul (non-BLAS ufunc loop) | 9.5 ms | **48.2 ms** |
| scale multiply | 0.01 ms | 0.03 ms |
| argpartition | 0.03 ms | 0.16 ms |

The cliff is fully explained: **~72% is the per-query full-matrix int8→int32
cast** (a fresh 4nd allocation + page faults every query), the rest is
numpy's integer matmul, which never touches BLAS. Nothing else matters.

### Prototype matrix (select + exact float64 rerank, p50 over 48 timings,
recall vs float64 ground truth; resident bytes vs exact 8nd = 614.4 MB at 50k)

| Approach | p50 10k | p50 50k | resident vs exact (50k) | recall strict/tie | bar (beat 17–22 ms, bytes < 8nd) |
| --- | --- | --- | --- | --- | --- |
| exact float64 (the bar) | 25.0 | **21.6** | 1.0 (614 MB) | 1.0 / 1.0 | — |
| current int8 (int32 cast+matmul) | 38.9 | 182.1 | 0.126 (77 MB) | 1.0 / 1.0 | FAIL (latency) |
| (a) per-query int8→f32 cast + sgemv | 95.8 | 213.4 | 0.126 | 1.0 / 1.0 | FAIL — cast cost eats the BLAS win |
| (b) precast f32 copy resident alongside int8 + sgemv | 5.7 | **12.4** | 0.626 (384 MB = 5nd+8n) | 1.0 / 1.0 | **PASS** |
| (b2) pure f32 unit matrix, no int8 at all | 6.0 | **12.0** | 0.500 (307 MB = 4nd) | 1.0 / 1.0 | **PASS** |
| (c1) block-wise int32 accumulation (8k rows) | 34.5 | 95.5 | 0.126 | 1.0 / 1.0 | FAIL (best int8-resident, still 4.4x exact) |
| (c2) `np.matmul(..., dtype=int32)` (no temp) | 53.1 | 197.8 | 0.126 | 1.0 / 1.0 | FAIL |
| (d) f32 query x int8 corpus via promotion | 92.8 | 214.4 | 0.126 | 1.0 / 1.0 | FAIL — numpy promotes via a full 4nd temp anyway |
| (d2) tensordot variant | 52.4 | 231.6 | 0.126 | 1.0 / 1.0 | FAIL |
| (e) block-wise cast into reusable f32 buffer + BLAS gemv | 27.0 | 107.0 | 0.208 (128 MB incl. 50 MB buffer) | 1.0 / 1.0 | FAIL |
| (e2) full reusable f32 scratch (cast with `out=`) | 12.9 | 57.8 | 0.626 | 1.0 / 1.0 | FAIL — cast *compute* alone is ~45 ms; dominated by (b) |

Recall notes: c1/c2/e compute bit-identical integer scores to the current
path — recall identical by construction. The f32 variants (a/b/b2/d/e) were
measured: 1.0 strict and tie-aware at both sizes (int8 values are exactly
representable in f32; accumulation rounding never flipped a 40-candidate
boundary on this corpus; the exact float64 rerank owns final ordering
regardless). Host had background load (three codex-capture processes at
~72% CPU); p50s are robust, p95s noisy. Our measured exact (21.6/25.0 ms)
and current-int8 (182 ms) bracket the lane's 22.4/223.7 ms — same machine
class, lane includes full query-path machinery.

### Verdict

**No approach clears the bar while preserving int8's memory posture.** The
only paths that beat exact latency (12 ms vs 17–22 ms) are the precast-f32
family, which carries 0.50–0.63x of exact bytes — that is a *float32 mode*,
not int8, and it defeats int8's reason to exist: at 50k x 1536 the 256 MB
`VECTOR_INDEX_CACHE_MAX_BYTES` budget admits int8 (77 MB) but not f32
(307 MB) or f32+int8 (384 MB). The best int8-resident numpy play (c1,
block-wise int32) halves the cliff (182 → 96 ms) but still loses 4.4x to
exact; variant (e) shows the int8→f32 cast compute itself (~45 ms at 50k) is
irreducible in pure numpy, so no int8-resident numpy kernel can reach 22 ms.

**Recommendation: do not promote int8 to default in 2.3.** Three concrete
actions instead:

1. **Land (c1) block-wise accumulation as a repair to the opt-in path** —
   free 1.9x latency improvement (182 → 96 ms at 50k; 39 → 35 ms at 10k),
   zero memory change, recall identical by construction. Honest docs update:
   int8 remains the memory-tightest option, latency ~4x exact at high-dim
   scale.
2. **Evaluate float32 selection (b2) as a new lane-gated mode**, not as int8
   promotion: 1.8x faster than exact, 2x fewer bytes, measured recall 1.0
   with the float64 rerank absorbing the f32 tie-flip class the 2.2
   diagnostics identified. It is the only measured mode that wins latency
   AND bytes simultaneously at 50k x 1536.
3. **Fold the "int8 fast kernel" question into the 2.3 ANN-backend agenda**
   (LadybugDB / alternative embedded ANN): a real int8 GEMV kernel (SIMD
   dot-product instructions) is a dependency decision, which 2.2 already
   deferred to 2.3 strategy.

---

## 2. Encoding-gate lane spec (`ENCODING_GATE_ENABLED` flip) — spec only

### What the gate does when enabled (read from `core.py:1188-1220`,
`salience.py:752-789`)

Classification is tag-only — every event is still appended and hash-chained —
but enabled-state consequences are: (i) payloads carry the `encoding` tag;
(ii) `redundant` appends emit a weak reinforcement toward `duplicate_of`
(the *old* memory); (iii) interference detection runs only for `novel`
classifications. Thresholds (`salience.py`): `redundant` at token-Jaccard
`content_overlap >= 0.9`; `reinforcing` at `>= 0.6`, or `>= 0.4` with
`entity_overlap >= 1.0`; else `novel`.

The dangerous error is **false-redundant**: genuinely novel content
classified redundant ⇒ reinforcement credit misdirected to the old memory,
interference detection silently skipped, and the new fact treated as a
duplicate under the cognitive profile.

### Corpus construction (deterministic, no LLM)

Reuse the `fok_calibration_lane.py` word-table pattern (disjoint
adjective/noun/domain tables, hash embeddings, embedded backend, fixed
session). Base corpus: N `decision.made` appends (sizes 200 / 1000). For a
base chunk of T tokens, replacing r tokens yields Jaccard
J = (T-r)/(T+r), so r = T(1-J)/(1+J) gives controlled overlap levels.
Probe families, each with a construction-time intent label and the measured
Jaccard recorded per probe:

| Family | Construction | Target J | Expected class |
| --- | --- | --- | --- |
| exact-duplicate | byte-identical re-append | 1.0 | redundant |
| near-duplicate paraphrase | replace 1 of ~20 tokens (filler word) | ~0.905 | redundant |
| reinforcing restatement | replace ~3 of 20 tokens, same facts | ~0.74 | reinforcing |
| corroborated moderate overlap | ~0.45 overlap + all entities known | 0.4–0.6 | reinforcing |
| uncorroborated moderate overlap | same J, one new entity | 0.4–0.6 | novel |
| **similar-but-genuinely-novel** | same entities + template skeleton, semantic payload flipped: negation, changed value/date/outcome tokens | engineered into 0.55–0.85, with a sub-band pushed to 0.88–0.92 to stress the redundant boundary | **novel (by construction: carries new facts)** |
| novel control | disjoint vocabulary | < 0.4 | novel |

The genuinely-novel family is the heart of the lane: token-level Jaccard is
blind to negation and value changes ("adopt the amber anchor pipeline
rollout" vs "abandon the amber anchor pipeline rollout" shares ~5/6 tokens,
J ≈ 0.71 — and a 20-token version with one flipped token sits at 0.905,
*above* the redundant threshold). The lane must measure exactly how often
that happens.

### Metrics

- **false_redundant_rate** (dangerous): fraction of novel-by-construction
  probes classified `redundant`, overall and per Jaccard band.
- **false_novel_rate**: fraction of exact/near-duplicate probes classified
  `novel` (cost: missed dedup, salience inflation — mild).
- Full per-family confusion matrix + threshold-margin histogram (distance of
  measured J from the nearest threshold, so boundary fragility is visible).
- **duplicate_of accuracy**: redundant classifications whose citation
  resolves to the true constructed source chunk.
- **End-to-end recall impact**: two fabrics over the identical mixed stream,
  gate on vs gate off, cognitive profile
  (`ZAXY_RETRIEVAL_PROFILE=cognitive`); query set targets the
  novel-by-construction payloads (the flipped facts); report recall@10
  gate-on minus gate-off, plus interference-detection counts (novel probes
  must still trigger detection when gated).

Deterministic by construction (fixed tables, hash embeddings, no wall
clock); `"validation": "internal"`.

### Exit criteria for flipping the default

1. **false_redundant_rate = 0** on the genuinely-novel family at every size,
   on **two consecutive lane runs** (zero tolerance: the harm is silent
   suppression of new facts; this mirrors the lane-evidence rule).
2. false_novel_rate ≤ 0.05 on exact + near-duplicate families.
3. End-to-end recall@10 for novel-fact queries, gate on ≥ gate off (no
   regression), and interference detections for novel probes unchanged.
4. duplicate_of accuracy ≥ 0.95 on true duplicates.
5. Boundary report published: classification behavior within ±0.02 of each
   threshold documented (informational, not gated).

If criterion 1 fails (likely for the negation sub-band at J ≥ 0.9), the lane
has earned a threshold/feature change (e.g. content-token vs filler-token
weighting or a negation guard) *before* any flip — defaults move on lane
evidence, not on assertion.

---

## 3. FoK promotion — data status, time-to-data, join design

### Marker count: zero, everywhere

`metacognition.fok.predicted` is emitted only by the MCP
`memory_feeling_of_knowing` handler (`mcp_server.py:2788-2816`; shipped
2.1.0, deployed `zaxy 2.2.0` is serving). Read-only scan of all local
Eventloom stores (`~/Projects/Personal/*/.eventloom` — 24 workspaces
including zaxy, the-hollow-lodge, codex-game, jedi-devproject-0):
**0 markers**. The string "fok" appears in zaxy-default.jsonl only inside
captured agent-transcript content, not as events.

Meanwhile checkout volume is healthy — `memory.checkout.completed` by
workspace: zaxy 644 (last 7 days: 17/45/24/47/64/97/54 ≈ **50–100/day**),
the-hollow-lodge 140 (bursts to 88/day), codex-game 24, jedi-devproject-0 3.
Agents check out constantly; **no agent ever calls the FoK tool first**. At
the current organic rate (0/day), time-to-calibration is **infinite**. This
is an instrumentation problem, not a patience problem.

### Power assumption and N

Target: a Brier-score CI that distinguishes FoK skill from the base-rate
predictor (the existing lane's exit criterion, now on organic data). Paired
per-query squared-error differences have sd ≈ 0.25 (scores in [0,1],
outcome rate ~0.5–0.7); to detect a Brier improvement of 0.05 at 95%
confidence: N ≈ (1.96 × 0.25 / 0.05)² ≈ **100–200 joined pairs**. Add per-
verdict bucket calibration (≥ 30 pairs in each of likely/possible/unlikely)
and a realistic join-success rate of ~0.7 ⇒ **target ~400 emitted
predictions**.

Time-to-data: if 2.3 auto-emits a FoK prediction at every `memory_checkout`
(prediction computed against the pre-retrieval index, then the same checkout
provides the outcome — a natural prediction/resolution pair), the zaxy
workspace alone produces ~60/day ⇒ **~400 predictions in roughly 7 days; a
meaningful calibration join after ~1–2 weeks of normal use**. Without that
change, never.

### Join design

- **Prediction side**: `metacognition.fok.predicted` — payload
  `{query_hash: sha256(query), verdict, score, authority_status}`, keys
  `thread` (= session_id) and `seq`.
- **Outcome side**: `memory.checkout.completed` — payload carries the
  **plaintext** `query` plus `token_efficiency {prompt_tokens,
  current_fact_count, evidence_count, facts_per_1k_prompt_tokens}`.
- **Match**: same `thread`; `sha256(checkout.payload.query) ==
  marker.query_hash`; `checkout.seq > marker.seq` within a window
  (next 50 events or 15 minutes); nearest-following wins.
- **Outcome label**: positive iff `token_efficiency.evidence_count >= 1`
  (proxy for "retrieval surfaced something"); upgrade with
  `current_fact_count >= 1` as a stricter variant and with
  `memory.feedback.recorded` / `memory.reinforced` events in the window as
  ground-truth-quality labels when present (currently rare: ~1–46 per
  workspace). Report both labelings — the proxy's optimism is itself a
  finding.
- **Known gap + 2.3 fix**: agents paraphrase between a check and the
  checkout, breaking exact hash equality. Two cheap remedies: (i) record
  `query_hash` in the checkout payload too, (ii) allow a sequence-adjacency
  fallback join (nearest following checkout in-window, flagged
  `weak_match`) reported separately. The auto-emit-at-checkout design makes
  the join trivial (same event pair, same query string by construction) and
  is the recommended path.

---

## Promotion-pass sequencing recommendation for 2.3

1. **First increment — FoK instrumentation** (smallest change, calendar-time
   bound): auto-emit `metacognition.fok.predicted` inside
   `handle_memory_checkout` before retrieval (plus `query_hash` on checkout
   payloads). Every day this isn't deployed is a day of calibration data
   lost; landing it first lets organic data accrue during the whole cycle.
2. **Encoding-gate lane** (cheap, deterministic, spec above is buildable in
   one increment): build, run, and either flip `ENCODING_GATE_ENABLED` on a
   double pass or harvest the threshold findings. This is the most likely
   promotion to actually land in 2.3.
3. **int8: demote from promotion candidate to repair + reframe**: land the
   block-wise accumulation fix on the opt-in path (free ~2x), add the
   float32-selection mode to the vector-scale lane as the new
   latency+memory candidate, and hand the "true int8 kernel" question to
   the already-scheduled ANN-backend evaluation. The promotion-pass verdict
   on int8-as-default is **no** — supported by the prototype matrix above.
4. **End of cycle — FoK calibration join lane** over the accumulated organic
   markers (~2+ weeks of data by then), evaluated against the
   Brier-vs-base-rate criterion with both outcome labelings.
