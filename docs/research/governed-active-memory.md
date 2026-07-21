# Governed Active Memory

> Zaxy 3 position paper (I8p — "Proof & Category Definition"). The category
> claim, the three differentiating claims, the now-real demo, and the honest
> evidence boundary for Zaxy's event-sourced agent-memory system. Every Zaxy
> capability cited below maps to shipped code on `master`; every number comes
> from a committed benchmark artifact and carries its caveat inline. Competitor
> figures are labeled vendor- or self-reported and are never restated as Zaxy's
> own. The grounding plan is [ZAXY-3.md](../../ZAXY-3.md); the metrics index is
> [AGENTS.md](../../AGENTS.md); the published benchmark surface is
> [benchmarks.md](../benchmarks.md).

## Summary

The agent-memory field has converged on one idea: memory should be *active* — it
should reflect on experience, distill skills, prevent repeated mistakes, and get
better over time. The two strongest category rivals make memory active by
letting it **mutate itself**: caura-memclaw runs an autonomous nightly
"Crystallizer" over a mutable store; Letta (formerly MemGPT) gives the model
self-editing memory blocks under last-write-wins concurrency. Both ship working
active-learning loops. Both, by construction, give up the ability to answer
"what changed, why, on whose evidence, and can we take it back?"

Zaxy's thesis — **Governed Active Memory** — is the inverse:

> Other systems make memory active by letting it mutate itself. Zaxy makes
> memory active while keeping every change a **gated, replayable, cited event**.

This is not a slogan retrofitted onto a vector database. Zaxy's substrate is an
append-only, hash-chained Eventloom log; the entire memory state is a pure
*replay function* of that log. Every learning step — a reinforcement, a generated
rule, a consolidation, a cross-agent promotion, a forget — is a sealed event with
a citation and a hash-chain position. The 2026 governance literature now argues
this is mandatory rather than optional. This paper states the category, maps the
three claims to shipped code and event types, walks the demo that no
mutable-store or last-write-wins system can reproduce, and draws a deliberately
conservative line between what is proven today and what is not yet independently
validated.

## The category, and where the rivals sit

The competitive teardown in [ZAXY-3.md](../../ZAXY-3.md) §3 is the source of
truth; the factual positions are summarized here.

**caura-memclaw (caura-ai)** — the direct category rival. "Fleet memory for AI
agents — governed, shared, self-improving." Architecture: a *mutable* PostgreSQL
+ pgvector + Redis store with an async event bus — **not** event-sourced. Its
active-learning loop (the "Karpathy Loop") has agents report success/fail/partial
outcomes; winners are reinforced and **failures auto-generate preventive `rule`
memories**. A nightly **Crystallizer** LLM-merges near-duplicates and retires
stale data autonomously, *not* review-gated. memclaw added a tamper-evident
hash-chained **audit log** in v2.17.0 — but the hash-chaining is on the audit log
only; the memory store itself remains mutable. Its proof points are
**[vendor-claimed]** (eToro: 300+ agents, 26.5k memories, 1,372 shared skills,
23 ms p50) and **[self-reported]** (LoCoMo 77.6%, LongMemEval 72.5%, 96–98% token
savings). These are memclaw's figures on memclaw's harness; they are not Zaxy
results and are reproduced here only to characterize the rival.

**Letta (letta-ai, formerly MemGPT)** — the category's research engine. An OS
metaphor (MemGPT, arXiv [2310.08560](https://arxiv.org/abs/2310.08560)) with core
in-context **memory blocks**, recall, and archival tiers, plus **self-editing
memory** (insert/replace/rethink tools). Its active loop is **sleep-time agents**:
a background agent shares the primary's memory blocks and asynchronously rewrites
them into "learned context" (*Sleep-time Compute*, Lin et al. 2025, arXiv
[2504.13171](https://arxiv.org/abs/2504.13171); the paper reports **[vendor/
self-reported]** ~5× less test-time compute). Edits are largely autonomous;
concurrency is **last-write-wins**; there is no tamper-evident log and no
first-class citations or grounded checkout.

Both rivals are **ahead on the active axis** (a shipped loop plus background
reflection in production) and **behind on provenance/governance-by-construction**
(mutable stores, autonomous overwrite, last-write-wins). Zaxy is the inverse, and
Zaxy 3 closes the active gap *without* surrendering the substrate that makes the
provenance claims true. The category Zaxy defines and intends to own is
**Governed Active Memory**: active learning where every mutation is a gated,
replayable, cited event.

## The three claims, mapped to shipped code

Each claim below names the shipped module and event type that makes it true. The
load-bearing invariant under all three (ZAXY-3 §9): the log is the source of
truth, every derived artifact is `authority_status=non_authoritative` until it
passes an explicit gate, nothing rewrites history, and everything cites an
`eventloom://<thread>/events/<seq>#<hash>` source.

### Claim 1 — Provable evolution

*Every reinforcement, rule, consolidation, promotion, and forget is a cited,
hash-chained event, and the whole memory state is a replay of the log.*

- **Outcome → governed preventive rule (I1).** `outcome_learning.py` builds the
  preventive-rule events; `MemoryFabric.generate_preventive_rule` routes a
  failure outcome through the evolution gate (op `rule_generate`) and, above
  threshold under the default tier, appends a `memory.rule.generated` event
  (otherwise it is held as `memory.rule.proposed`). This is the governed twin of
  memclaw's auto-rule: same trigger, but the rule is non-authoritative, cited,
  and reversible.
- **The single gate (I4).** `evolution_policy.py` + `MemoryFabric.evaluate_evolution_gate`
  implement one configurable autonomy policy with three tiers —
  `propose_only`, `auto_with_rollback`, and `require_review` — selectable
  globally (`evolution_autonomy_default`, shipped default `auto_with_rollback`)
  and per-op (`evolution_op_autonomy`, e.g. `forget=propose_only`). Every gate
  decision records a non-authoritative, replayable `evolution.gate.evaluated`
  event. I1, I2, and I7 producers all route through this one gate.
- **Fleet propagation (I7).** `fleet.py` emits `fleet.skill.promoted`,
  `fleet.outcome.propagated`, and `fleet.rule.propagated` for cross-agent
  sharing; lifecycle is `fleet.promotion.reviewed`, `fleet.promotion.rolled_back`,
  and `fleet.memory.superseded` (additive supersession, never delete/overwrite).
  "Which agent taught the fleet this, from what evidence" is a deterministic
  replay query, not a side log.
- **Replay.** Because the Eventloom log is append-only and hash-chained,
  `EventLog.verify()` validates the chain and the projection is a pure replay
  function; `memory_replay` (MCP) and `zaxy replay` / `zaxy memory log` / `zaxy
  memory diff` (CLI) reconstruct exactly how any state was reached.

This is precisely the property the 2026 security literature now calls mandatory:
long-term-memory security "*cannot be retrofitted at retrieval or execution time
alone, but must be anchored in storage-time provenance, versioning, and
policy-aware retention from the outset*" — Verifiable Memory Governance, Lin et
al. 2026 (arXiv [2604.16548](https://arxiv.org/abs/2604.16548)). Zaxy is that
from the substrate up.

### Claim 2 — Drift-resistant consolidation

*Compaction is additive, source-backed, and audited; the log is never rewritten
by a summarize-and-overwrite step.*

Competitors crystallize by LLM summarize-and-overwrite. SSGM (Lam et al. 2026,
arXiv [2603.11768](https://arxiv.org/abs/2603.11768)) names the failure mode:
**semantic drift — knowledge degrades through iterative summarization.** Zaxy
treats this warning as a hard constraint.

- **Additive, audited compaction.** Consolidation produces cited, review-pending
  candidates that are always non-authoritative; `compaction.py`'s
  `audit_event_log` checks identity-recall and citation-coverage before any
  additive medoid/exemplar projection is built, and it never rewrites the log.
- **Governed sleep-time crystallization (I2).** `crystallization.py` is a
  config-gated (`crystallization_enabled`, off by default), operator/cron-fired
  **one-shot** reflection pass — no always-on daemon and no MCP tool, so the MCP
  surface stays pull-only. Each pass *schedules existing primitives*
  (consolidation, procedure mining, the metacognition monitor, the compaction
  audit, a read-only salience replay), routes every fresh candidate through the
  I4 gate, and appends one `crystallization.run.completed` summary event citing
  the candidate ids it produced. "Auto-apply" means an *accepted review* that is
  still non-authoritative and reversible within the rollback window — never a
  promotion to authority, and never a destructive overwrite. Details:
  [crystallization.md](../crystallization.md), [consolidation.md](../consolidation.md).
- **Long-horizon two-tier assembly (I3).** For never-ending threads,
  [consolidation.md](../consolidation.md) describes an explicit episodic (recent)
  vs consolidated (remote) checkout (`long_horizon_enabled`,
  `long_horizon_recent_window`): older history is represented by its *cited*
  consolidation candidates, never by raw re-summarization, and the consolidated
  tier stays cited back to source events.

### Claim 3 — Reversible and verified forgetting

*Forgetting is reversible attenuation by default; hard deletion is governed
cryptographic erasure that leaves the hash chain verifiable.*

VMG (2604.16548) requires storage-time provenance, versioning, **rollbackability,
and verified-forgetting**. Zaxy ships all four.

- **Reversible attenuation (default).** Salience decay (half-life 30 days,
  clamped at ≥ 0.01) lets a memory fade from default ranking while staying one
  explicit query away, with a replayable record of why it faded. "Forgotten"
  never means "gone."
- **Edit → re-ingest and rollback (I5a).** `editable.py` +
  `MemoryFabric.edit_memory` re-ingest a human edit as a cited
  `memory.corrected` event (the original is never mutated); `rollback_memory`
  reverses a prior evolution with a cited `memory.rolled_back` event that, on
  replay, undoes the cited evolution's effect (a rolled-back consolidation
  acceptance reverts the candidate to its pre-acceptance status). Both route
  through the I4 `update` gate and leave `EventLog.verify().ok == True`. MCP:
  `memory_edit`, `memory_rollback`; CLI: `zaxy memory edit`, `zaxy memory
  rollback`. See [editability.md](../editability.md).
- **Verified forgetting / crypto-erasure (I5b).** Because a payload is sealed
  into the event hash, you cannot scrub plaintext without breaking `verify()`. So
  forgettable memories are encrypted at append time (an `append(...,
  forgettable=True)` seals the payload as a `__zaxy_cipher` ciphertext cell);
  the data-encryption key is wrapped under a key-encryption key stored only in an
  out-of-log erasure vault. `MemoryFabric.verified_forget` (event
  `memory.forgotten`, via `forgetting.py`, routed through the I4 `forget` gate)
  destroys the wrapped key and appends a cited tombstone. The on-disk ciphertext
  and its hash are untouched — `EventLog.verify()` stays green — while the
  plaintext is permanently unrecoverable and readers see a `[FORGOTTEN]`
  sentinel. MCP: `memory_forget`; CLI: `zaxy memory forget`.
  **Honest caveat (carried from [editability.md](../editability.md)):** the
  erasure crypto reuses Zaxy's portable-bundle envelope, which is **experimental
  and unaudited** — do not rely on it for high-value-secret or compliance
  guarantees without an independent cryptographic review.

## The demo (now real)

The differentiation that survives a demo (ZAXY-3 §10) is a four-step sequence
that exercises one Eventloom session end to end. Every step below maps to shipped
code; none of it is possible on a mutable store (memclaw) or under last-write-wins
(Letta).

1. **Evolve — generate a preventive rule from a failure.** An agent reports a
   failure outcome on a recalled memory; `MemoryFabric.generate_preventive_rule`
   routes it through the I4 gate (`evolution.gate.evaluated`, op `rule_generate`)
   and appends a cited `memory.rule.generated` event
   (`outcome_learning.py`).
2. **Replay — reconstruct exactly how the rule came to be.** The rule cites its
   source events; `memory_replay` / `zaxy replay` replays the hash-chained log to
   show the precise failure observations and the gate decision that produced it,
   and `EventLog.verify()` confirms the chain is intact.
3. **Roll it back — reverse the evolution.** `MemoryFabric.rollback_memory`
   appends a cited `memory.rolled_back` event (`editable.py`) that, on replay,
   undoes the rule's effect — additively, without mutating the original rule
   event.
4. **Verified-forget — crypto-erase a payload while keeping `verify()` green.**
   `MemoryFabric.verified_forget` destroys the wrapped key for a forgettable
   payload and appends a cited `memory.forgotten` tombstone (`forgetting.py`);
   the ciphertext and its hash are untouched, so the hash chain still verifies,
   yet the plaintext is gone for good.

A mutable store cannot replay *how* a rule was formed because it overwrote the
evidence; a last-write-wins store cannot guarantee a rollback or a verified erase
leaves an intact, tamper-evident chain. Governed Active Memory can do all four
because each step is just one more sealed, cited event.

## Evidence (honest)

Two committed artifacts back the claims. Both carry their caveats inline; neither
is overstated.

### FleetBench (governance, token efficiency, real fleet-wide transfer)

Source: `reports/benchmarks/fleet-transfer-v1/` (`report.md`, `report.json`,
`manifest.json`), version `fleet-v2`, result fingerprint `868b0b96…7a079d9d`
(scored fields only; latency excluded). Measured over real CoordinationBench
missions and real `FleetManager` propagation at three scale points.

| worker_count | coordination_quality | governance_correctness | cross_agent_transfer | control | token_efficiency |
|---|---|---|---|---|---|
| 3 | 0.907407 | 1.0 | 1.0 | 1.0 | 0.4825 |
| 5 | 0.922222 | 1.0 | 1.0 | 1.0 | 0.336504 |
| 8 | 0.923977 | 1.0 | 1.0 | 1.0 | 0.313291 |
| **mean** | 0.917869 | 1.0 | 1.0 | 1.0 | 0.377432 |

What this shows, with caveats:

- **`cross_agent_transfer = 1.0` is now REAL fleet-wide transfer**
  (`scope = fleet_wide`), replacing the retired `within_mission_proxy`. The
  origin agent propagates the mission's accepted findings through the real I4
  gate; every other *enrolled* agent is scored by a real `checkout_memory` call.
  A never-enrolled stranger is the mandatory negative control and receives
  nothing (`control = 1.0`). The retired proxy also read 1.0, but the two are
  **not comparable** — the proxy never exercised enrollment, trust tiers, the
  gate, or another agent's checkout.
- **It measures governed *delivery*, not relevance.** The checkout fleet lane
  returns every active promotion an enrolled agent is entitled to and is **not**
  filtered by query relevance (verified by issuing an unrelated query and
  receiving the same promotions). No retrieval-quality claim may cite it.
- **The transfer number is falsifiable, not decorative.** Un-enrolling the
  receivers drops it 1.0 → 0.0 while propagation still succeeds; enrolling the
  stranger drops the control 1.0 → 0.0. Both are asserted in
  `tests/test_fleet_benchmark.py`.
- **`governance_correctness = 1.0` and `coordination_quality`** are REAL,
  exact-scored, deterministic aggregates of CoordinationBench signals. Quality
  now varies with worker count, but the variation is driven **entirely** by
  `conflict_precision`; the other eight sub-axes stay saturated at 1.0, and
  `governance_correctness` does not vary with scale at all. Neither is on its
  own evidence of scaling behaviour.
- **`token_efficiency` now *falls* with worker count** (0.483 → 0.313), reversing
  the retired scaffold's rising curve. That earlier rise was an artifact: the old
  workload padded worker counts above three with **empty-finding filler**, so raw
  logs grew on padding while the governed brief stayed fixed at 202 tokens. Real
  added workers contribute real accepted findings, so the brief genuinely grows.
- **`latency_ms` is REAL wall-clock and environment-dependent**; excluded from
  the fingerprint and from determinism checks. Not a portable performance claim.
- **Not cleared for external publication** (`claim_scope.publishable: false`):
  one synthetic workload, no baseline or competitor run, and transfer observed
  only at its 1.0 ceiling.

### LongMemEval 500 hash report (plumbing and recall parity)

Source: [AGENTS.md](../../AGENTS.md) Metrics table. The full 500-question
LongMemEval-compatible **hash** checkout: mean **0.724**, Answer@5 **0.628**,
Recall@5 **0.972**, citation coverage **1.000**, p95 **1472.11 ms**, p99
**2652.55 ms**.

Caveats, stated plainly:

- This run uses the **`hash` embedding provider**. A hash embedding is *not* a
  semantic embedding, so these numbers prove **plumbing and recall parity** — the
  retrieval/citation path returns the right events with full citation coverage
  (Recall@5 0.972, citation coverage 1.000) — they do **not** measure semantic
  retrieval quality. The lower mean/Answer@5 versus recall reflects synthesis-side
  behavior, not a retrieval failure.
- It is a Zaxy **same-harness checkout diagnostic**, **not** an official
  LongMemEval end-to-end assistant score (the standing claim-boundary in
  [benchmarks.md](../benchmarks.md)).

## Current evidence boundary

This is the section the honesty discipline exists for. Drawing the line clearly
is part of the claim.

**Proven today (on committed artifacts):**

- Governance/provenance by construction: append-only hash-chained log, replayable
  state, non-authoritative-by-default artifacts, a single auditable evolution
  gate, reversible rollback, and verified forgetting that keeps `verify()` green
  — all shipped and event-typed.
- Governance correctness and coordination quality on the FleetBench scaffold
  (`governance_correctness = 1.0`, `coordination_quality ≈ 0.907`, exact and
  deterministic).
- Plumbing and recall parity on the LongMemEval hash harness (Recall@5 0.972,
  citation coverage 1.000), with monotone token-efficiency gains as worker count
  grows on the scaffold.

**Not yet independently validated (do not claim):**

- **Semantic retrieval quality on a real embedding provider.** The headline
  evidence above uses hash embeddings; semantic quality at scale is unproven here.
- **Fleet-scale cross-agent transfer.** The scaffold's transfer number is a
  within-mission proxy; genuine fleet-wide propagation is implemented (I7) but not
  yet measured at fleet scale.
- **Head-to-head comparisons with memclaw or Letta.** Competitor figures in this
  paper are vendor- or self-reported on their own harnesses; Zaxy has not run a
  same-harness head-to-head, and CoordinationBench keeps competitor claims
  disclosure-only ([benchmarks.md](../benchmarks.md)).
- **Cryptographic assurance of verified forgetting.** The erasure envelope is
  experimental and unaudited (see Claim 3).

No number in this document is a substitute for any of the above. The category
claim rests on the governance/provenance properties, which are proven; the
performance and head-to-head claims are explicitly deferred.

## Research foundation

The Zaxy 3 design borrows mechanism from neuroscience and the agent-memory
literature, and the 2026 governance papers validate the substrate bet. The full
mapping is in [ZAXY-3.md](../../ZAXY-3.md) §5; the essentials follow with real
links.

**Neuroscience → mechanism.**

- Complementary Learning Systems — McClelland, McNaughton & O'Reilly 1995
  ([PMC](https://pubmed.ncbi.nlm.nih.gov/7624455/)); updated for AI in Kumaran,
  Hassabis & McClelland 2016 ([TICS](https://pubmed.ncbi.nlm.nih.gov/27315762/)).
  Fast episodic store + slow structured store trained by replay → the two-tier
  assembly (I3): Eventloom as the hippocampal episodic log, the consolidated
  projection as neocortical structure.
- Systems consolidation — Frankland & Bontempi 2005
  ([nrn1607](https://www.nature.com/articles/nrn1607)). Recent→remote
  reorganization → recent vs remote tiers and time-dependent governed promotion
  (I3, I7).
- Hippocampal replay / sleep consolidation — Wilson & McNaughton 1994
  ([Science](https://www.science.org/doi/10.1126/science.8036517)); Diekelmann &
  Born 2010 ([nrn2762](https://www.nature.com/articles/nrn2762)). Offline
  reactivation reorganizes memory → the background crystallization runner (I2).
- Schema-based assimilation — Tse et al. 2007
  ([Science](https://www.science.org/doi/10.1126/science.1135935)). Fast-assimilate
  schema-consistent facts, gate novel/conflicting ones (I2/I4).
- Synaptic tagging & salience — Frey & Morris 1997
  ([Nature 385:533](https://www.nature.com/articles/385533a0)). Salient/surprising
  events are preferentially stabilized → outcome/prediction-error-weighted
  reinforcement (I1).
- Reconsolidation — Nader, Schafe & LeDoux 2000
  ([Nature](https://www.nature.com/articles/35021052)). Retrieval makes a memory
  labile and prediction-error-gated for update → the basis for governed
  update-on-recall (I4) and outcome-gated learning (I1).
- Adaptive forgetting — Richards & Frankland 2017, *The Persistence and Transience
  of Memory*, Neuron (https://www.cell.com/neuron/fulltext/S0896-6273(17)30365-3).
  Forgetting is regulated and adaptive → decay-aware retrieval + reversible
  attenuation (Claim 3).

**Agent-memory systems → what to match/beat.**

- MemGPT ([2310.08560](https://arxiv.org/abs/2310.08560)) → tiered/paged
  long-horizon context (I3).
- Generative Agents — Park et al. 2023
  ([2304.03442](https://arxiv.org/abs/2304.03442)) → reflection into higher-level
  memories that **cite their source observations**, a precedent for cited
  consolidation (I2, I5).
- Reflexion — Shinn et al. 2023
  ([2303.11366](https://arxiv.org/abs/2303.11366)) → outcome feedback → verbal
  self-reflection (I1).
- Voyager — Wang et al. 2023
  ([2305.16291](https://arxiv.org/abs/2305.16291)) → an ever-growing library of
  verified executable skills (I6, I1).
- A-MEM — Xu et al. 2025
  ([2502.12110](https://arxiv.org/abs/2502.12110)) → "memory evolution" updates to
  linked notes (I4).
- Mem0 — Chhikara et al. 2025
  ([2504.19413](https://arxiv.org/abs/2504.19413)) → extract-then-update with
  ADD/UPDATE/DELETE/NOOP ops (I3, governance vocabulary for I4).
- HippoRAG — Gutiérrez et al. 2024
  ([2405.14831](https://arxiv.org/abs/2405.14831)) → Personalized PageRank over a
  KG (I3; Zaxy already ships a `graph_walk` lane).
- MemoryBank — Zhong et al. 2023
  ([2305.10250](https://arxiv.org/abs/2305.10250)) → Ebbinghaus-curve
  decay/reinforce (I1/I4).
- Larimar — Das et al. 2024
  ([2403.11901](https://arxiv.org/abs/2403.11901)) → one-shot edit + selective
  fact-forgetting (I5).
- Self-RAG — Asai et al. 2023
  ([2310.11511](https://arxiv.org/abs/2310.11511)) → reflection tokens + per-segment
  citations (I5, I1).
- Sleep-time Compute — Lin et al. 2025
  ([2504.13171](https://arxiv.org/abs/2504.13171)) → background pre-computation of
  learned context (I2).
- Surveys/vocabulary: Zhang et al. 2024
  ([2404.13501](https://arxiv.org/abs/2404.13501)); Du et al. 2025
  ([2505.00675](https://arxiv.org/abs/2505.00675)), whose six atomic ops
  (Consolidation, Updating, Indexing, Forgetting, Retrieval, Condensation) Zaxy
  adopts as its governance vocabulary; Shi et al. 2024
  ([2404.16789](https://arxiv.org/abs/2404.16789)).

**2026 governance → why the substrate bet is right.**

- SSGM — Lam et al. 2026 ([2603.11768](https://arxiv.org/abs/2603.11768)): names
  semantic drift from iterative summarization; validates Claim 2.
- Verifiable Memory Governance — Lin et al. 2026
  ([2604.16548](https://arxiv.org/abs/2604.16548)): storage-time provenance,
  versioning, rollbackability, verified-forgetting; validates Claims 1 and 3.
- Provenance / evidence-tracing — Wang et al. 2026
  ([2606.04990](https://arxiv.org/abs/2606.04990)): every recalled/consolidated
  item should carry its evidence trail; validates the citation invariant.

## Reproducibility

Every number above is checkable from committed artifacts.

**FleetBench scaffold.** Regenerate over real CoordinationBench cases with the
shipped CLI:

```bash
zaxy fleet-benchmark --output-dir reports/benchmarks/fleet-v1 \
  --worker-counts 3,5,8 --missions 1
```

The exact command that produced the committed scaffold artifact
(`reports/experimental/fleet-benchmark-scaffold/report.md`) is:

```bash
env PYTHONPATH=src EMBEDDING_ENABLED=true EMBEDDING_PROVIDER=hash EMBEDDING_DIMENSION=1536 \
  python -c 'from pathlib import Path; from zaxy_benchmarks.fleet_benchmark import run_fleet_benchmark; run_fleet_benchmark(Path("reports/experimental/fleet-benchmark-scaffold"))'
```

The scored fields are deterministic and fingerprinted; `latency_ms` is excluded
because it is wall-clock and environment-dependent.

**LongMemEval 500.** The hash-report metrics live in the
[AGENTS.md](../../AGENTS.md) Metrics table; the published headline checkout report
and its claim boundary live in [benchmarks.md](../benchmarks.md) (the headline 500
artifact under `reports/benchmarks/`). The benchmark page is the authority on
which artifact is the current public claim.

## Related documentation

- Plan and category source: [ZAXY-3.md](../../ZAXY-3.md)
- Architecture: [architecture.md](../architecture.md)
- Evolution gate, crystallization, consolidation:
  [crystallization.md](../crystallization.md),
  [consolidation.md](../consolidation.md)
- Editability, rollback, verified forgetting: [editability.md](../editability.md)
- Fleet memory plane: [agent-events.md](../agent-events.md), [mcp.md](../mcp.md)
- Plugins: [plugins.md](../plugins.md)
- Benchmarks and claim boundaries: [benchmarks.md](../benchmarks.md)
- Project overview: [README.md](../../README.md)
