# Experimental Associative Memory Projection

This branch explores whether a neuron-astrocyte-inspired associative projection
can help Zaxy recover latent agent state from partial event-history cues.

The experiment is based on:

- Kozachkov, Slotine, and Krotov, *Neuron-Astrocyte Associative Memory*:
  <https://arxiv.org/abs/2311.08135>
- The authors' NAAM reference repository:
  <https://github.com/kozleo/naam>

## Scope

This is not a production checkout feature and does not change Zaxy core.
Eventloom remains the immutable source of truth. The associative projection is a
derived, replayable, discardable layer that can only propose candidates after
resolving them back to cited Eventloom event refs.

The target is not LongMemEval answer synthesis. Zaxy already has strong
LongMemEval-compatible results. The target is partial-cue pattern completion:

- recover a hidden root cause from scattered symptoms;
- recover an unstated coordination constraint from indirect handoff failures;
- surface distributed state that is not directly named in the user's query;
- preserve citations for every recalled pattern.

## Current Baseline

The first implementation is intentionally small and dependency-free:

- `src/zaxy/associative_memory.py`
- `zaxy experimental pattern-completion`
- `tests/test_associative_memory.py`

It uses a deterministic token projection:

1. Tokenize Eventloom events into event packets.
2. Seed active process terms from direct lexical matches.
3. Reinforce terms through shared event packets over a few iterations.
4. Rank final candidate events by direct and associative support.
5. Score candidates only after Eventloom refs and hashes are present.

This approximates the useful software lesson from the paper without importing
the research model directly: a slower higher-order process layer can propagate
weak cues across multiple local events.

## Run

```bash
zaxy experimental pattern-completion --output-dir reports/benchmarks/pattern-completion-smoke
```

For machine-readable output:

```bash
zaxy experimental pattern-completion --output-dir reports/benchmarks/pattern-completion-smoke --json
```

The command writes:

- `pattern-completion-workload.json`
- `pattern-completion-benchmark.json`
- `pattern-completion-benchmark.md`
- one Eventloom JSONL file per case

## StateRecoveryBench

StateRecoveryBench is the broader falsification harness for this branch. It
uses adversarial event histories for hidden causes, release constraints, stale
runtime state, release-fixture drift, user quality bars, coordination metric
gaps, generated authority-shaped negatives, incomplete authority metadata,
bridge-evidence requirements, and no-safe-answer abstention cases.

Run it with:

```bash
zaxy experimental state-recovery --output-dir reports/benchmarks/state-recovery-smoke
```

For machine-readable output:

```bash
zaxy experimental state-recovery --output-dir reports/benchmarks/state-recovery-smoke --json
```

The built-in baselines are:

- `direct_lexical`: direct query-token overlap.
- `hash_vector`: deterministic hashed token-vector cosine similarity.
- `graph_traversal`: lexical seed plus shared-term traversal.
- `zaxy_core_proxy`: deterministic source-aware proxy for current Zaxy retrieval
  posture; this is not a full `MemoryFabric` checkout run.
- `memory_fabric_checkout`: appends each case through `MemoryFabric`, runs the
  model-facing Memory Checkout contract with the `coordinate` purpose profile,
  and scores selected cited checkout facts/evidence.
- `associative_projection`: the derived process-term propagation baseline.
- `authority_resolved_associative`: associative projection followed by explicit
  current/promoted/non-rejected authority filtering.

Current official StateRecoveryBench scores over the expanded 33-case workload
are tracked in
`reports/benchmarks/state-recovery-v1/state-recovery-benchmark.md`:

| baseline | state accuracy | minimal evidence recall | stale rejection | distractor resistance | abstention accuracy | token cost | latency ms | citation coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| associative_projection | 1.000 | 0.803 | 0.485 | 0.212 | 0.848 | 56 | 0.328 | 1.000 |
| authority_resolved_associative | 1.000 | 0.985 | 1.000 | 0.909 | 1.000 | 28 | 0.348 | 1.000 |
| direct_lexical | 0.697 | 0.606 | 0.394 | 0.121 | 0.848 | 69 | 0.049 | 1.000 |
| graph_traversal | 0.818 | 0.742 | 0.515 | 0.061 | 0.848 | 71 | 0.064 | 1.000 |
| hash_vector | 0.697 | 0.606 | 0.394 | 0.121 | 0.848 | 69 | 0.037 | 1.000 |
| memory_fabric_checkout | 0.818 | 0.909 | 1.000 | 0.818 | 1.000 | 34 | 436.638 | 1.000 |
| zaxy_core_proxy | 0.697 | 0.652 | 0.424 | 0.091 | 0.848 | 71 | 0.045 | 1.000 |

The first StateRecoveryBench run showed that associative propagation recovered
latent state but pulled in nearby symptom/distractor observations. The expanded
workload makes that failure mode explicit: raw association is strong at latent
state recovery, but weak at authority, stale-row rejection, and abstention.

The authority-resolved baseline keeps broad propagation for recall, then
resolves from the associative support set using explicit current/promoted/non-
rejected authority metadata and current observation bridge rows. This preserves
latent state recall, improves minimal evidence recall, rejects stale rows,
handles no-safe-answer cases, and cuts the average token budget roughly in half.
It is still branch-local research code, not a production checkout policy.

The MemoryFabric checkout-backed run is the important product signal. It is not
the proxy: every case is appended through `MemoryFabric`, projected through the
embedded graph backend, assembled through Memory Checkout, filtered by the
`coordinate` purpose profile, and scored only from selected cited checkout
facts/evidence. That full path clears the 1.1 production guardrails for state
accuracy, minimal evidence recall, stale rejection, distractor resistance,
abstention accuracy, and citation coverage. The guardrail is deliberately scoped
to accepted-state recovery under this 33-case workload; it does not replace
LongMemEval or CoordinationBench.

This run also found and fixed a real metadata propagation gap. Generic
Eventloom rows now preserve authority metadata such as `authority_scope`,
`status`, `stale`, `promoted`, and `superseded_by` through verbatim and graph
checkout lanes, so purpose policies can suppress stale, rejected, unsupported,
and worker-local rows consistently.

The non-perfect distractor score is useful. The remaining misses are old
hand-authored cases where distractors lack enough authority metadata to reject
them, or where a bridge event is not cleanly distinguishable from nearby
support. That argues for authority/provenance metadata as the real Zaxy thesis,
not for adding a heavier neural associative model just because the direction is
interesting.

## Metrics

- `latent_state_recall`: whether expected latent-state terms are recovered.
- `evidence_recall`: whether expected support events are retrieved.
- `citation_coverage`: whether returned evidence has Eventloom hashes.
- `injected_tokens`: approximate support-term count.
- `latency_ms`: projection completion latency.

The same harness reports a direct lexical baseline. The experiment only earns
attention if the associative projection beats direct lexical retrieval on
partial-cue state recovery while keeping citation coverage at `1.0`.

StateRecoveryBench adds:

- `state_accuracy`: exact all-terms latent-state recovery.
- `minimal_evidence_recall`: required proof events returned.
- `stale_rejection`: stale/superseded events avoided.
- `distractor_resistance`: distractor events avoided.
- `abstention_accuracy`: no-answer cases return no evidence instead of a
  plausible but unsupported answer.
- `token_cost`: retrieved/support term budget.
- `latency_ms`: baseline runtime.
- `citation_coverage`: returned evidence has Eventloom hashes.

## Next Research Gate

The current branch is a baseline. The next useful gate is not a heavier Modern
Hopfield or NAAM-like implementation yet. The next gate is an accepted-state
checkout resolver that can choose the promoted state packet from a broad
Memory Checkout result while keeping the broader evidence auditable in
provenance.

Only if that benchmark shows a remaining class of failures that event-sourced
authority/provenance cannot solve should Zaxy consider learned/tensor process
state.
