# Benchmarks

Zaxy keeps the public benchmark surface intentionally small. Active benchmark
evidence is limited to:

- the current headline 500-question LongMemEval-compatible checkout report; and
- the Harvey LAB external legal-agent memory-ablation report.

Older backend shootouts, partial slices, experimental LongMemEval iterations,
LongMemBench adapter artifacts, and debug reports are archived under
`reports/archive/` and `docs/archive/`. They are development history, not
current public claims.

## Headline 500

The current headline LongMemEval-compatible result is:

[reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.md](../reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.md)

Frozen run config:
[reports/benchmarks/longmemeval-500-publish-20260607/run-config.md](../reports/benchmarks/longmemeval-500-publish-20260607/run-config.md)

This is a Zaxy same-harness checkout diagnostic over the cleaned
LongMemEval-compatible workload. It is not an official LongMemEval end-to-end
assistant score.

| Metric | Value |
|--------|------:|
| Generated | `2026-06-07T16:20:10Z` |
| Workload SHA-256 | `90fb2307195d7e16b963a2b8a30f03b375bd42a45d41aeaa55423029dd84e3fc` |
| Events | `5,372` |
| Questions | `500` |
| Sessions | `948` |
| Backend | `zaxy-checkout` |
| Mean score | `0.956` |
| Answer@5 | `0.910` |
| Recall@1 | `0.960` |
| Recall@5 | `1.000` |
| Recall@10 | `1.000` |
| Identity recall | `0.980` |
| Citation coverage | `1.000` |
| p50 latency | `881.01 ms` |
| p95 latency | `1,966.65 ms` |
| p99 latency | `2,495.07 ms` |
| Approx tokens | `10,192` |

Interpretation: retrieval and citation are at ceiling in this adapted checkout
protocol. The remaining reported misses are synthesis-side (`45`
`synthesis_miss` cases). The same report includes a BM25 baseline with mean
`0.520`, Answer@5 `0.520`, Recall@5 `0.770`, and citation coverage `1.000`.

## Harvey LAB

The current Harvey LAB external memory-ablation evidence is:

[reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md](../reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md)

Primary report artifacts:

- [harvey-lab-benchmark.md](../reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.md)
- [harvey-lab-benchmark.json](../reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json)
- [harvey-lab-external-run.md](../reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-external-run.md)
- [harvey-lab-external-run.json](../reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-external-run.json)
- [harvey-lab-ready.json](../reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-ready.json)
- [harvey-lab-status.json](../reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-status.json)

| Metric | Value |
|--------|------:|
| External suite | Harvey LAB memory retrieval ablation |
| Harvey commit | `29748828133dff83ad2263af353fb035504f8f77` |
| Tasks completed | `10/10` |
| Mean criterion pass rate | `0.788` |
| Delta vs regular/no-memory | `+0.184` |
| Delta vs article-best task rows | `+0.081` |
| Wins vs article-best task rows | `9/10` |
| Mean total seconds | `138.786` |
| Total tokens | `5,951,174` |
| Memory search calls | `30` |
| Memory read calls | `10` |

Interpretation: Harvey LAB is external downstream work-product evidence. The
metric is criterion pass rate, not binary task pass/fail.

## Zaxy 2.0 RC.1 Benchmark Freeze

The `2.0.0-rc.1` freeze gate validates the current release evidence without
changing retrieval, synthesis, or benchmark scoring behavior:

```bash
zaxy benchmark-freeze --json
```

The tracked freeze manifest is
[reports/benchmarks/2.0.0-rc.1/manifest.json](../reports/benchmarks/2.0.0-rc.1/manifest.json).

The gate is a claim-boundary and artifact-integrity check. It requires the
headline 500-question LongMemEval-compatible checkout report, the frozen
headline run config, Harvey LAB external-anchor artifacts, and the
project-defined RC lanes for StateRecoveryBench, CoordinationBench,
PurposeBench, causal, consolidation, procedural, and metacognitive behavior.

RC.1 evidence is interpreted in three separate buckets:

- `longmemeval_compatible_checkout`: the same-harness 500-question checkout
  diagnostic listed above. It is the headline public benchmark artifact, not an
  official LongMemEval end-to-end assistant score.
- `external_anchor`: Harvey LAB legal-agent memory-ablation evidence. It is
  external downstream work-product evidence, not a general outside-user
  validation report.
- `project_defined_internal`: StateRecoveryBench, CoordinationBench,
  PurposeBench, causal, consolidation, procedural, and metacognitive
  guardrails. These lanes protect product contracts and must not be merged into
  the headline 500 or Harvey LAB numbers.

The active RC.1 project-defined artifacts are:

- [StateRecoveryBench](../reports/benchmarks/state-recovery-v1/state-recovery-benchmark.md)
- [CoordinationBench](../reports/benchmarks/coordination-real-v1/coordination-benchmark.md)
- [PurposeBench](../reports/benchmarks/purpose-v1/purpose-benchmark.md)

CoordinationBench has a conservative competitor-claim boundary. The
`competitor_claim_gate` remains blocked for Quarq and Hybi unless a future
same-harness run supplies tracked, auditable runner outputs. Release checks use
explicit flags such as `--require-competitor-claim quarq` and
`--require-competitor-claim hybi` only to prove that missing competitor evidence
is disclosed and cannot silently become a public claim. The current
CoordinationBench artifact is a Zaxy project-defined internal guardrail for
accepted parent state, stale-claim rejection, duplicate consolidation,
non-authoritative leakage, evidence coverage, purpose feedback, and checkout
answerability.

PurposeBench follows the same disclosure posture for purpose-conditioned memory
claims. Publicly derived purpose examples that mention systems such as Quarq or
Semantic Reach are diagnostic holdouts only; they are not head-to-head benchmark
claims and they do not establish competitor performance. The active PurposeBench
report proves Zaxy's purpose profiles and evidence policies on tracked internal
lanes, while the holdout pack documents source boundaries and claim status.

The RC.1 gate fails closed when required artifacts are missing, when the
headline 500 falls below the frozen quality or latency floors, or when a 2.0
internal or project-defined lane is classified as external validation. This is
intentionally a release-readiness gate, not a reward function.

## Zaxy 2.0 Alpha Causal And Consolidation Lane

Zaxy 2.0 alpha includes a project-defined internal guardrail lane for causal
projection and review-gated consolidation. This lane is not external
validation, is not part of the headline LongMemEval-compatible checkout claim,
and must not be reported as a public benchmark number unless a future release
explicitly publishes a full report with its own claim boundary.

The alpha lane checks behavior that is specific to the causal and
consolidation contracts:

- causal predecessor and successor queries preserve expected endpoint and
  relation matching;
- causal results retain Eventloom citation coverage and expose review and
  authority metadata;
- alpha.2 consolidation segment selection is deterministic and event-sourced
  from replayed Eventloom ranges, with stable session-scoped segment identity;
- authority-boundary preservation keeps inferred causal edges and
  consolidation candidates non-authoritative unless a separate gate promotes
  them;
- stale or distractor-supported causal paths do not outrank cited target paths;
- consolidation candidate scoring verifies source-event fidelity and rejects
  candidates that omit required source references or imply authority promotion;
- generated episode, claim, and procedure candidates remain review material,
  not authoritative memory, even when a review disposition is `accepted`;
- stale, conflicted, rejected, superseded, and `valid_to`-closed consolidation
  candidates are diagnosed so checkout and status surfaces do not present them
  as current authoritative memory.

Use this lane as an engineering regression guardrail for the alpha causal
and consolidation surface. The consolidation guardrail is internal and
project-defined: it measures source-event fidelity, review gating, stale
rejection, and authority-boundary preservation. Do not combine it with the
headline 500 metrics, Harvey LAB evidence, or external-validation language.

## Zaxy 2.0 Beta.1 Reasoning-Loop Guardrail

Beta.1 adds an internal guardrail scorer for reasoning-loop memory primitives.
This is an engineering contract check, not a public benchmark claim and not a
LongMemBench-tailored lane. It does not score final answers or tune retrieval.

The guardrail reports five transparent fields:

- `observable_call`: primitive and belief proposal activity must be represented
  by replayable Eventloom event types such as `reasoning.primitive.called` or
  `belief.update.proposed`.
- `phase_match`: the recorded phase must match deterministic routing for
  `planning`, `execution`, `review`, or `reflection`.
- `citation_presence`: trace evidence must carry Eventloom citations.
- `authority_boundary`: primitive observations and belief proposals must remain
  `non_authoritative`; belief proposals remain pending until reviewed.
- `score`: the simple mean of the four contract ratios.

Use this lane to catch regressions in observability, phase routing, citation
coverage, and authority boundaries for beta.1 primitives. Do not report it as
external validation, do not combine it with the headline 500 or Harvey LAB
numbers, and do not use it to reward answer phrasing.

Beta.2 extends the internal guardrail to metacognition and procedural planning
contracts. The scorer inspects contract fields only; it does not score final
answers, expected benchmark labels, or answer phrasing. The beta.2 fields are:

- `observable_metacognition`: known unknowns, confidence assessments, conflict
  clusters, and reverify requests must be replayable Eventloom event types.
- `open_reverify_status`: re-verification needs stay open until a separate
  resolution path changes state.
- `procedural_citation_presence`: applicable procedures must carry Eventloom
  citations.
- `planning_phase_match`: procedure-derived planning packets must remain in the
  planning phase unless explicitly routed otherwise.
- `authority_boundary`: metacognition and procedures remain
  `non_authoritative`; they are diagnostic or planning guidance, not accepted
  facts.
- `score`: the simple mean of the beta.2 contract ratios.

This beta.2 guardrail is an internal release-quality check and readiness signal.
It is not external validation and must not be merged into the headline
LongMemEval-compatible or Harvey LAB results.

## Agent Experience Lanes (Internal)

Zaxy 2.1 Phase 1 adds three deterministic agent-experience lanes in
`zaxy_benchmarks/agent_experience_lanes.py`. They are project-defined internal
lanes labeled `"validation": "internal"` in every report. Run them with:

```bash
zaxy agent-experience-lanes --lanes all
```

- **Tool-adoption lane**: static MCP listing-surface metrics for the `core`
  versus `full` tool profiles — listed tool count, serialized schema bytes and
  estimated tokens (`estimate_tokens`), the 1-based listing rank of the
  `memory_checkout` front door, and the fraction of other listed tool
  descriptions that reference it. It does NOT simulate agent transcripts and
  makes no claim about tool-selection accuracy or turns-to-first-checkout;
  those require scripted-agent runs that have not been performed.
- **Budget lane**: seeds a real in-temp-dir fabric through the production
  write path, runs one real `memory_checkout`, and sweeps
  `apply_checkout_budget` across token budgets (default
  `256,512,1024,2048,4096,8192,unlimited`). It reports the raw curve
  (`budget_used`, elided section count and kinds, packed prompt tokens) and
  passes or fails the graceful-degradation contract: citation-bearing payload
  fields (`evidence`, `current_facts`, `provenance`, citation counts) survive
  every budget, and elisions are monotone non-increasing as the budget grows.
  It does NOT measure recall@k or answer quality under budgets.
- **Cache lane**: same seeded fabric; repeats checkouts without agent appends
  and verifies the consolidated stable prefix is byte-identical across
  repeats, then appends a consolidated-tier validated skill and verifies the
  prefix changes. Full-prompt identity across repeats is reported as an
  informational bool only: checkout records salience reinforcement events
  whose replay lands in the volatile tail. The lane reports stable-prefix
  length, prefix/total ratio, and an `estimated_provider_cache_hit_fraction`
  (`stable_prefix_tokens / prompt_tokens`). That fraction is arithmetic over
  Zaxy's own token estimator — it is NOT a measured provider cache-hit rate.

All three lanes are deterministic (hash embeddings, embedded projection, fixed
seed content, no LLM calls). Do not report them as external validation and do
not combine them with the headline 500 or Harvey LAB numbers.

## Cognitive Lanes (Internal)

Zaxy 2.2 adds two deterministic cognitive-memory lanes in
`zaxy_benchmarks/forgetting_lane.py` and
`zaxy_benchmarks/fok_calibration_lane.py`. They produce the mechanism-level
evidence behind the 2.3-rc.1 default-flip decisions (cognitive retrieval
profile, salience-on) and are labeled `"validation": "internal"` in every
report. Run them with:

```bash
zaxy cognitive-lanes --lanes all
```

- **Forgetting lane**: seeds real embedded fabrics through the production
  write path and synthesizes reinforcement histories by appending real
  `memory.reinforcement` events (built with the `zaxy.salience` builders) at
  fixed timestamps, with a fixed `now` for every salience replay. It measures
  four flip-safety properties of salience attenuation under the cognitive
  retrieval profile versus plain: cold-start parity (zero reinforcement
  events must leave checkout facts/evidence content and order identical, at
  the checkout-ranking layer and end to end), no recall loss (every
  below-floor attenuated memory stays reachable via explicit `memory_query`
  and `memory_replay` and is labeled `attenuated` in diagnostics), ranking
  lift (the confirmed-reinforced member of an equally relevant pair ranks
  first under cognitive, not under plain), and exemption correctness (pinned
  and authority-accepted below-floor memories still surface). It does NOT use
  LongMemBench-style probes and makes no claim about retrieval quality on
  organic usage.
- **FoK calibration lane**: seeds deterministic word-composition corpora at
  several sizes (default `50,200`, larger sweeps via `--fok-sizes`), builds
  the feeling-of-knowing index exactly like the MCP
  `memory_feeling_of_knowing` handler (projected active entity names only),
  and scores raw FoK predictions against ground truth produced by the real
  explicit-query retrieval path — a query is positive only if retrieval
  returns a context containing one of its topic terms. It reports the FoK
  Brier score against the base-rate predictor's Brier score (the roadmap exit
  criterion), verdict-bucket hit/miss rates, and false-positive/negative
  rates per corpus size. Template query phrasings over synthetic corpora —
  this is NOT a measurement of organic-usage calibration.

Both lanes are deterministic (hash embeddings, embedded projection, fixed
word tables and timestamps, no LLM calls): two runs produce identical
reports. They are mechanism-level engineering evidence, not external
validation, and must not be merged into the headline 500 or Harvey LAB
numbers.

## Graph-walk and vector-scale lanes (internal)

Zaxy 2.2/2.3 adds two more deterministic lanes in
`zaxy_benchmarks/graph_walk_lane.py` and
`zaxy_benchmarks/vector_scale_lane.py`, labeled `"validation": "internal"` in
every report. Run them with:

```bash
zaxy graph-scale-lanes --lanes all
```

- **Graph-walk (PPR) lane**: seeds one real embedded fabric through the
  production write path with multi-hop entity-bridge clusters: the correct
  memory is connected to the query-matched anchor only via 1-2 intermediate
  hops, while a distractor memory carries an *exactly tied* lexical signal
  (same matched query terms, same token counts) and no graph path to the
  anchor. It compares two identical `QueryRouter` arms over the same store,
  differing only in `graph_walk_enabled` — the seam the cognitive retrieval
  profile arms. Because the lexical scores tie exactly, the plain arm's
  target/distractor score margin is zero (the tie resolves by storage order);
  the walk arm must produce a strictly positive margin, attributable to graph
  evidence alone. The lane also checks single-hop non-regression (direct hits
  keep their rank), two-pass ranking identity, and that repeated walks are
  served from the signature-keyed walk cache. It does NOT measure full
  cognitive checkouts (salience decay is wall-clock-dependent) and makes no
  claim about organic multi-hop questions: the bridges, ties, and vocabulary
  are synthetic and constructed to sit inside the walk blend's operating
  window.
- **Vector-scale lane**: builds deterministic hash-embedded synthetic corpora
  directly against `EmbeddedGraphStore` (default sizes `1000,10000`;
  `100000` opt-in via `--scale-sizes` because the Kuzu HNSW shadow sync is
  insert-bound and takes minutes at 10^5) and measures, per size, the exact
  float64 path, the Kuzu-native HNSW path (with a lowered
  `vector_ann_threshold`), and the int8-quantized path: recall@10 versus the
  exact ground truth, p50/p95 query latency, and resident index bytes.
  Corpus hashes, exact/quantized recall, and bytes are two-run reproducible;
  ANN recall is reported under `measurements` because Kuzu's HNSW graph
  construction is not run-to-run reproducible, and all timings are
  environment-dependent. The roadmap exit criterion (>= 0.95 recall@10 with
  latency and byte improvements) is defined at 10^5 vectors; smaller runs
  report `not_evaluated_at_target_scale`. Hash embeddings are not semantic
  embeddings — this lane measures index mechanics, not retrieval quality.

Both lanes use synthetic corpora and the deterministic hash embedding
provider. Do not report them as external validation and do not combine them
with the headline 500 or Harvey LAB numbers.

## Claim Boundaries

- Use **LongMemEval-compatible checkout** for the headline 500 diagnostic.
- Use **Harvey LAB external** for the legal-agent work-product result.
- Do not describe the LongMemEval-compatible checkout run as an official
  LongMemEval score.
- Do not cite archived partial runs as current benchmark claims.
- Before publishing a new full 500, update this page to point at one new
  headline report and keep the previous headline under `reports/archive/`.

Related docs: [testing.md](testing.md), [external-validation.md](external-validation.md),
and [README.md](../README.md).
