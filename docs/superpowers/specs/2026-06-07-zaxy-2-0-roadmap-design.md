# Zaxy 2.0 Roadmap Design

## Purpose

Zaxy 2.0 is the causal and consolidated cognitive-substrate release. The
existing 1.x line remains the stable, benchmark-winning event-sourced memory
fabric. Zaxy 2.0 adds rebuildable intelligence projections on top of that core:
causal structure, reviewable consolidation, metacognitive state, procedural
memory, and reasoning-loop primitives.

The release must not trade away the published 1.x strengths. Existing checkout
quality, recall, citation coverage, and latency remain regression gates for all
2.0 work.

## Release Lines

The versioning split is:

- `1.1.x`: stable published baseline, patch releases only.
- `1.2.x`: optional adoption and hardening release for existing concepts.
- `2.0.0-alpha.*`: experimental cognitive-substrate features behind explicit
  gates.
- `2.0.0-beta.*`: integrated but still pre-release reasoning and metacognitive
  features.
- `2.0.0-rc.*`: frozen release candidates with benchmark and regression
  evidence.
- `2.0.0`: public cognitive-substrate release.

The 1.x tree protects the proven product. The 2.0 tree carries architecture that
changes what Zaxy can do, while still depending on the 1.x event-sourced core as
the source of truth.

## Core Boundary

Eventloom remains the immutable source of truth. Causal and consolidation layers
are discardable projections that can be rebuilt from the log. They may propose,
cite, rank, and explain higher-order memory, but they do not become trusted
state unless they pass the existing review and authority path.

Generated abstractions must always carry:

- source Eventloom event references;
- citation paths;
- confidence;
- extraction or inference method;
- projection version;
- review status;
- authority status;
- stale or supersession diagnostics when applicable.

Memory Checkout may expose causal and consolidated context, but it must clearly
distinguish accepted state from inferred or review-pending context.

## Increment Plan

### 2.0.0-alpha.1: Causal Projection and Consolidation Scaffold

This is the first 2.0 architecture slice.

Scope:

- Add a rebuildable causal graph projection from Eventloom events.
- Add typed causal edge schema with provenance, confidence, method, and source
  event references.
- Add causal read APIs for predecessor, successor, path, and outcome-explanation
  queries.
- Add Memory Checkout diagnostics for causal context.
- Add consolidation candidate objects for episodes, claims, and procedures.
- Add a review-gated promotion path for consolidation candidates.
- Keep promotion conservative: no automatic promotion to authoritative memory.
- Add benchmark and regression gates that prove no regression against the
  frozen 1.x checkout behavior.

Non-scope:

- Learned retrieval policy.
- Automatic belief revision.
- Autonomous consolidation-to-authority.
- Broad synthesis rewrites.
- Benchmark-specific tuning.

### 2.0.0-alpha.2: Review-Gated Consolidation MVP

This release turns the scaffold into a usable consolidation layer.

Scope:

- Cluster related event segments into reviewable episodes.
- Synthesize cited candidate claims from episodes.
- Synthesize cited candidate procedures from successful or failed workflows.
- Preserve source event backpointers for every abstraction.
- Track review status, authority status, confidence, scope, and purpose.
- Add consolidation-specific diagnostics to checkout and status surfaces.
- Add rejection, stale, supersession, and conflict behavior for generated
  abstractions.

Non-scope:

- Fully autonomous long-term learning.
- Silent authority promotion.
- Cross-project global procedure sharing.

### 2.0.0-beta.1: Reasoning-Loop Memory Primitives

This release makes memory callable during planning, execution, and reflection
rather than only before generation.

Scope:

- Expose primitives such as causal predecessor lookup, causal successor lookup,
  explain-outcome, propose-belief-update, get-claim-confidence, and retrieve
  similar procedures.
- Make primitive calls observable and replayable.
- Apply purpose profiles differently for planning, execution, review, and
  reflection.
- Keep updates as proposals unless existing authority gates accept them.

### 2.0.0-beta.2: Metacognitive and Procedural Hardening

This release strengthens the agent-facing intelligence layer.

Scope:

- Track known unknowns.
- Track conflicting evidence clusters.
- Track confidence trajectories over time.
- Add query surfaces for high-uncertainty claims and re-verification needs.
- Promote procedural memory as a first-class retrieval lane for planning.
- Add rollback and contradiction diagnostics for procedural memory.

### 2.0.0-rc.1: Benchmark Freeze

This release candidate freezes behavior and evidence for publication.

Scope:

- Freeze benchmark config.
- Run the full LongMemEval-compatible 500-question checkout benchmark.
- Run Harvey/LAB external-anchor checks where available.
- Run StateRecoveryBench, CoordinationBench, and PurposeBench.
- Add 2.0-specific causal, consolidation, procedural, and metacognitive lanes.
- Publish benchmark artifacts with workload hashes and config.
- Block release on regression against 1.x public numbers unless the regression is
  explicitly justified and accepted.

## Benchmark Gates

Zaxy 2.0 must protect existing public numbers before claiming new intelligence.
At minimum, every release candidate must report:

- LongMemEval-compatible 500-question mean score.
- Answer@5.
- Recall@5.
- citation coverage.
- latency percentiles.
- token budgets.
- benchmark config and workload hash.

The published 1.x checkout behavior is the regression floor. The intended
guardrail is:

- Recall@5 remains at the published floor.
- citation coverage remains at the published floor.
- Answer@5 and mean do not regress materially.
- latency and token budgets remain within documented tolerances.

Implementation must not tailor code to individual benchmark questions. Fixes
must address general classes of memory behavior or add missing product
capability.

## New 2.0 Evaluation Lanes

The 2.0 thesis needs benchmarks that measure more than recall. Proposed lanes:

- causal predecessor and successor accuracy;
- outcome explanation quality;
- causal distractor resistance;
- stale causal edge rejection;
- consolidation citation fidelity;
- consolidation authority gating accuracy;
- procedural reuse lift;
- task-success lift with and without Zaxy memory;
- metacognitive uncertainty detection;
- belief revision accuracy under conflicting evidence.

These lanes should be documented as project-defined until external validation is
available. Claims must distinguish internal benchmark results from external or
same-harness comparisons.

## Production Constraints

All 2.0 features must follow the existing architectural discipline:

- event-sourced source of truth;
- rebuildable projections;
- no uncited trusted memory;
- authority and purpose controls;
- fail-open behavior for optional infrastructure;
- deterministic rule-based extraction where possible;
- LLM-dependent extraction or synthesis only where it adds real capability;
- observable replayable memory operations;
- tests before implementation;
- benchmark evidence before release claims.

## Open Implementation Questions

These are intentionally deferred to implementation planning:

- the exact causal edge type taxonomy;
- whether causal projections live inside the existing graph backend contract or
  a parallel projection interface;
- how consolidation candidates are stored before review;
- whether alpha.1 should expose causal APIs through CLI, MCP, or both;
- the smallest useful causal benchmark lane for alpha.1;
- latency budgets for causal path expansion inside checkout.

## Acceptance Criteria

The roadmap design is accepted when:

- Zaxy 1.x remains the stable benchmark-winning line;
- Zaxy 2.0 is explicitly scoped as causal and consolidated cognitive substrate;
- `2.0.0-alpha.1` includes causal projection plus consolidation scaffold;
- all generated abstractions remain review-gated before authority;
- existing benchmark quality is protected by regression gates;
- new 2.0 claims require new 2.0 evaluation lanes.
