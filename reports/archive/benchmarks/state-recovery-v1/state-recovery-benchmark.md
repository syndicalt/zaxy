# StateRecoveryBench

- schema: `state-recovery-report-v1`
- version: `state-recovery-v0`
- workload: `916201f70da9d058aee80a31f8cf59d92dad59f5fd645f3dfbd3a1b23e7dddad`
- generated at: `2026-06-05T02:50:25.450993Z`
- cases: `33`
- baselines: `direct_lexical, hash_vector, graph_traversal, zaxy_core_proxy, memory_fabric_checkout, associative_projection, authority_resolved_associative`
- status: `pass`
- production baseline: `memory_fabric_checkout`

## Guardrails

| metric | observed | threshold | status |
| --- | ---: | ---: | --- |
| state_accuracy | 0.818 | 0.818 | pass |
| minimal_evidence_recall | 0.909 | 0.900 | pass |
| stale_rejection | 1.000 | 1.000 | pass |
| distractor_resistance | 0.818 | 0.800 | pass |
| abstention_accuracy | 1.000 | 1.000 | pass |
| citation_coverage | 1.000 | 1.000 | pass |

## Baseline Scores

| baseline | state accuracy | minimal evidence recall | stale rejection | distractor resistance | abstention accuracy | token cost | latency ms | citation coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_lexical | 0.697 | 0.606 | 0.394 | 0.121 | 0.848 | 69 | 0.049 | 1.000 |
| hash_vector | 0.697 | 0.606 | 0.394 | 0.121 | 0.848 | 69 | 0.037 | 1.000 |
| graph_traversal | 0.818 | 0.742 | 0.515 | 0.061 | 0.848 | 71 | 0.064 | 1.000 |
| zaxy_core_proxy | 0.697 | 0.652 | 0.424 | 0.091 | 0.848 | 71 | 0.045 | 1.000 |
| memory_fabric_checkout | 0.818 | 0.909 | 1.000 | 0.818 | 1.000 | 34 | 436.638 | 1.000 |
| associative_projection | 1.000 | 0.803 | 0.485 | 0.212 | 0.848 | 56 | 0.328 | 1.000 |
| authority_resolved_associative | 1.000 | 0.985 | 1.000 | 0.909 | 1.000 | 28 | 0.348 | 1.000 |

## Scope

StateRecoveryBench is an official Zaxy benchmark lane for partial-cue accepted-state recovery under stale, distracting, incomplete, and no-safe-answer event histories.
`memory_fabric_checkout` is the production guardrail baseline. Associative projection rows remain diagnostic research baselines and are not product claims.
This benchmark does not replace LongMemEval or CoordinationBench.
