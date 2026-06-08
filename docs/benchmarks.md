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
