# Zaxy 2.1–2.3 Agent Experience and Cognitive Memory Roadmap Design

## Purpose

Zaxy 2.1 through 2.3 make memory effortless for agents to adopt, economical to
consult, cognitively informed in how it ranks and forgets, and scalable in its
embedding layer. These releases land before the 2.5/3.0 latent memory roadmap
(see `2026-06-09-zaxy-2-5-3-0-latent-memory-roadmap-design.md`) and several
items feed directly into its lanes.

The motivating observations:

- The primary user of Zaxy is the agent, and the agent-facing surface is wide:
  44 MCP tools with no signal about which one is the entry point. Tool-selection
  accuracy degrades as tool count grows, so the surface itself is a usability
  cost.
- Memory consultation has a token price and a latency price, and Zaxy currently
  offers no budget contract: checkout returns what ranking selects, not the best
  packet under a caller-stated token budget, and packet ordering is indifferent
  to provider prompt caching.
- Zaxy never forgets. The append-only design is correct for truth, but ranking
  treats a memory retrieved and confirmed useful fifty times the same as one
  never retrieved at all. Human memory research is unambiguous that retrieval
  strengthens, disuse decays, and surprise gates encoding — and the
  event-sourced design makes all three implementable as pure projection policy
  with zero loss of audit.
- The embedded vector index is exact brute force over float64 matrices. That is
  the right default for small corpora and the wrong ceiling for large ones.

The guiding rule is:

> Memory should be cheap to consult, honest about what it knows, and quiet
> about what no longer matters — without ever deleting truth.

## Research Anchors

This roadmap stays close to the memory literature — human and agent — without
copying any one paper's assumptions into production.

- **Complementary Learning Systems** (McClelland, McNaughton, O'Reilly 1995;
  updated Kumaran et al. 2016): a fast episodic hippocampal store paired with a
  slow consolidating neocortical store. Zaxy already is this architecture —
  Eventloom is the episodic store, the consolidated graph and (in 2.5) latent
  artifacts are the semantic store, and the consolidation-review pipeline is
  systems consolidation with a provenance guarantee the brain does not have.
  This framing should be adopted explicitly in the memory fabric paper and
  product docs.
- **Recency × importance × relevance scoring** (Generative Agents, Park et al.
  2023, <https://arxiv.org/abs/2304.03442>): a simple, legible composite
  retrieval score. Zaxy adopts the composite but replaces the LLM-assigned
  importance scalar with outcome-grounded reinforcement from `memory_feedback`
  and the coordination performance ledger.
- **The testing effect** (Roediger & Karpicke 2006): retrieval practice
  strengthens memory more than re-exposure. Zaxy analog: a memory surfaced by
  checkout and confirmed useful gains salience; surfacing alone is weaker
  reinforcement than confirmed use.
- **Prediction-error gated encoding** (Greve et al. 2017): humans
  preferentially encode what violates expectation. Zaxy analog: a write-time
  gate that classifies appends as novel, reinforcing, or redundant against what
  checkout would already return — tagging, never dropping.
- **Encoding specificity** (Tulving & Thomson 1973): retrieval succeeds when
  cues at recall match cues at encoding. Zaxy analog: store context cues
  (active task, repository, tool, session phase) alongside memories and weight
  cue overlap at checkout, not just semantic similarity.
- **Reconsolidation** (Nader, Schafe, LeDoux 2000): retrieved memories become
  labile and are re-encoded. Zaxy analog: retrospective credit assignment (3.0
  lane) should target memories that were actually checked out into the failing
  or succeeding session, not the whole store.
- **Metamemory and feeling-of-knowing** (Hart 1965; Nelson & Narens 1990):
  organisms cheaply estimate whether they know something before attempting
  recall. Zaxy already ships `memory_known_unknowns`; this roadmap adds the
  positive half — a sub-millisecond "probably/possibly/unlikely" pre-check so
  agents can decide whether checkout is worth its cost.
- **HippoRAG** (Gutiérrez et al. 2024, <https://arxiv.org/abs/2405.14831>):
  personalized PageRank over a knowledge graph as an implementation of
  hippocampal indexing theory. Zaxy already maintains the graph; PPR from
  query-matched entities is a natural multi-hop retrieval upgrade over pure
  vector similarity.
- **Voyager** (Wang et al. 2023, <https://arxiv.org/abs/2305.16291>): an
  ever-growing skill library mined from successful execution traces. Zaxy
  already ships `memory_skill` and tool-call capture; the missing piece is the
  loop that proposes skills from successful traces through the existing
  consolidation-review pipeline.
- **MemGPT / Letta** (Packer et al. 2023, <https://arxiv.org/abs/2310.08560>):
  OS-style memory hierarchy with explicit token budgets and paging. Zaxy adopts
  the budget contract (checkout under a caller-stated token ceiling) without
  the self-editing memory model, which conflicts with the authority contract.

## Release Positioning

### Zaxy 2.1: Agent Experience

The agent-facing surface gets a front door, a budget contract, and a closed
compaction loop.

Primary thesis:

> An agent should be able to adopt Zaxy with one tool, state a token budget,
> and survive a context compaction without losing what mattered.

### Zaxy 2.2: Cognitive Memory

Retrieval ranking adopts salience, forgetting, cue-matched recall, metamemory,
and graph-walk retrieval — all as projection-level policy over the unchanged
immutable log.

Primary thesis:

> Zaxy can forget the way organisms forget — by attenuation, not erasure —
> and every attenuation is replayable, inspectable, and reversible.

### Zaxy 2.3: Embedding Scale

The embedding layer gets an approximate-nearest-neighbor path, quantization,
and a versioned migration story.

Primary thesis:

> Exact search stays the default where it is exact and fast; scale is opt-in,
> measured, and never silently lossy.

### Contributions forward to 2.5/3.0

- Cache-stable packet ordering (2.1) is the delivery vehicle for 2.5 latent
  artifacts: stable consolidated prefixes are exactly what provider prompt
  caching rewards.
- Reconsolidation-targeted credit assignment (2.2 design, 3.0 delivery)
  sharpens the 3.0-alpha.2 retrospective loop: attribute outcomes to memories
  that were checked out, not to the store at large.
- The salience ledger (2.2) becomes a training signal for the 3.0-beta.1
  continual learning policy.

## Current Architecture Fit

Nothing in this roadmap weakens the existing invariants:

- Eventloom remains the immutable source of truth. Forgetting is a ranking
  policy in projections; no event is deleted or rewritten.
- The write-time encoding gate tags events; it never drops them. A redundant
  event is still appended, hash-chained, and replayable.
- Salience scores, cue records, and skill candidates are projection state or
  ordinary Eventloom events — rebuildable, discardable, review-gated where they
  propose abstractions.
- Memory Checkout remains the model-facing trust contract. Budgeted checkout
  changes how much is returned, never what is citable.
- Tool profiles change which tools are listed, not what the server can do; the
  full surface remains available to callers that request it.

## Designs

### Tool Surface Profiles and Verb Consolidation

The MCP server gains a profile parameter (`zaxy serve --profile core|full`,
`MCP_TOOL_PROFILE` setting). The core profile lists a small verb set:
`memory_checkout`, `memory_append`, `memory_query`, `context_assemble`,
`memory_feedback`, `memory_invalidate`, plus `memory_capabilities` as the
discovery escape hatch. The full profile lists everything, unchanged.

In parallel, the long tail of single-purpose tools is consolidated behind
`operation`-enum tools where the grouping is natural (consolidation lifecycle,
confidence/metacognition reads). Consolidated tools are additive; the existing
names remain available in the full profile through at least 2.x per the
stability commitment.

`memory_capabilities` becomes the canonical "what else can I do" tool so a
core-profile agent can discover and request the full surface.

### Single Front Door

Tool descriptions and docs converge on one message: call `memory_checkout`
first; everything else is plumbing or power use. The checkout description gets
rewritten as the entry point; quickstarts, MCP docs, and the workspace
instruction block emitted by `ensure_session_initialized` all point at it.

### Doctor Deepening

`zaxy doctor` (existing preflight in `src/zaxy/doctor.py`) gains checks for:
hash-chain integrity over the active log, projection freshness vs. log
signature, embedding provider availability and dimension agreement, vector
index cache budget headroom, MCP profile sanity, and hook installation status —
with one-line remediation per failure.

### Token-Budget Checkout

`context_assemble` and `memory_checkout` accept `max_tokens`. Packing is a
greedy salience-per-token knapsack over candidate packet sections with a
deterministic tokenizer estimate (chars/4 fallback; provider tokenizer when
configured). The response reports `budget_requested`, `budget_used`, and what
was elided, so the agent knows recall was truncated rather than empty.

### Cache-Stable Packet Ordering

Checkout output is reordered into stability tiers: (1) consolidated/accepted
facts and procedures that change rarely, (2) session-scoped state, (3)
query-specific evidence. Tier-1 content is serialized canonically (stable sort
keys, no timestamps in the rendered prefix) so repeated checkouts in one
session produce byte-identical prefixes and provider prompt caches hit.
A diagnostics field reports the stable-prefix length so cache efficiency is
measurable.

### Compaction Recovery Loop

The precompact hook already records compaction. The loop closes with a
`session-resumed` hook event: after the harness compacts, Zaxy assembles a
recovery packet — checkout against pre-compaction state, diffed against what a
summary plausibly preserves (recent verbatim, open tasks, accepted findings,
known unknowns) — and emits it for re-injection. `src/zaxy/compaction.py`
(identity-preserving compaction audits) provides the safety rails: the recovery
packet must cite only Eventloom-backed state.

### Salience and Projection-Level Forgetting

Each projected memory carries a salience score:

```
salience = recency_decay(last_use) × reinforcement(use_history) × base_importance
```

- Reinforcement events: checkout surfacing (weak), positive `memory_feedback`
  or coordination promotion (strong), explicit invalidation (negative).
- Reinforcement is recorded as ordinary Eventloom events, so salience is
  rebuildable by replay — forgetting is replayable.
- Checkout ranking multiplies existing relevance scores by salience; below a
  floor, memories leave default ranking but remain reachable via
  `memory_query`/`memory_replay` and are labeled "attenuated", never hidden
  from explicit search.
- Interference detection: when a new append contradicts an above-floor memory,
  both are flagged and a `memory_propose_belief_update` proposal is generated
  proactively, review-gated as today.

### Write-Time Encoding Gate

At append time, an optional gate classifies the event against current
checkout state: `novel` (contradicts or extends), `reinforcing` (confirms),
`redundant` (duplicates). The tag rides in event metadata. Projection treats
`redundant` events as reinforcement signals rather than new ranked entries.
The gate is off by default in 2.2-alpha and opt-in until measurement shows no
recall regression.

### Encoding-Specificity Cues

Appends capture a cue record when available: active task or mission, repo or
workspace identity, originating tool, session phase. Checkout computes a cue
overlap term and blends it with semantic similarity. Cues are plain event
metadata — no schema migration of the log, only projection changes.

### Feeling-of-Knowing Pre-Check

A new lightweight tool `memory_feeling_of_knowing` (core profile) answers
"would checkout likely return something for this query?" in O(1)-ish time using
only in-memory state: cue-index hit counts, entity-name bloom filters, and
salience histograms — no embedding call, no graph query. Returns
`likely | possible | unlikely` plus the signal breakdown. Honest calibration is
the acceptance bar: measured against actual checkout outcomes in the benchmark
lane.

### Graph-Walk Retrieval (Personalized PageRank)

Retrieval gains a multi-hop stage: seed nodes from query-matched entities
(vector + name match), run bounded personalized PageRank over the projected
graph (Kuzu first; same algorithm against Neo4j/Postgres backends), blend PPR
mass into candidate ranking. This is the HippoRAG result applied to a graph
Zaxy already maintains. Behind a retrieval-profile flag until the internal
benchmark lane shows lift.

### Procedure Mining

`capture_tool_call_completed` traces feed a miner that detects successful
multi-step tool sequences recurring across sessions, and proposes them as
skill candidates through the existing consolidation-review pipeline (proposal
events, review, acceptance). Accepted candidates become `memory_skill` entries
with citations to the source traces. No skill becomes authoritative without
review — identical to every other abstraction path.

### Embedding Scale

- **ANN**: when a session's vector corpus crosses a size threshold, the
  embedded backend builds a Kuzu-native HNSW index instead of the dense numpy
  matrix; below the threshold, exact brute force remains the default (it is
  exact, simple, and already budget-capped). Search results carry an
  `exact: bool` flag.
- **Quantization**: opt-in int8 (later binary) storage with float rerank of an
  oversampled candidate set (top-k×4). Memory budget accounting (existing
  byte-budget eviction) counts quantized bytes.
- **Versioning**: vectors store `embedding_model@version`. A lazy migration
  path re-embeds on read when versions mismatch and a provider is available;
  `zaxy doctor` reports mixed-version corpora; checkout never silently compares
  vectors across incompatible versions.

## Evaluation Plan

Each behavior change lands with a measurement lane before it changes defaults:

- **Tool-adoption lane**: scripted agent transcripts measuring tool-selection
  accuracy and turns-to-first-successful-checkout under core vs. full profile.
- **Budget lane**: recall@k and citation coverage as a function of `max_tokens`;
  the budget contract must degrade gracefully, never cliff.
- **Cache lane**: stable-prefix byte length and simulated provider cache hit
  rate across repeated checkouts in long sessions.
- **Forgetting lane**: precision/recall on LongMemBench-style probes with
  salience on vs. off; attenuated memories must remain reachable by explicit
  query with zero loss.
- **FoK calibration lane**: Brier score of feeling-of-knowing predictions
  against actual checkout outcomes.
- **PPR lane**: multi-hop retrieval probes (entity-bridge questions) embedded
  vs. PPR-blended ranking.
- **Scale lane**: recall@k, p50/p99 latency, and resident bytes for exact vs.
  HNSW vs. quantized at 10^4–10^6 vectors.

Public claims follow the external-validation policy: internal lanes are labeled
internal; no headline numbers from unverified lanes.

## Non-Goals

- No deletion, rewriting, or expiry of Eventloom events. Forgetting is ranking.
- No self-editing memory in the MemGPT sense; the authority contract stands.
- No removal of existing MCP tool names in 2.x; profiles change listing, not
  existence.
- No mandatory new dependencies for the default path: HNSW uses Kuzu's native
  index; quantization uses numpy already present.
- No LLM-in-the-loop scoring inside the hot retrieval path; salience and FoK
  are computed from recorded events and in-memory state.

## Risks

### Ranking Opacity

Salience-modulated ranking can surprise users who expect deterministic
recency ordering. Mitigation: salience contributions appear in checkout
diagnostics; `--retrieval-profile plain` restores pre-2.2 ranking.

### Attenuation of Load-Bearing Memories

A rarely-retrieved memory may still be critical (credentials policy, standing
constraint). Mitigation: authority-bearing and pinned memories are exempt from
the floor; `memory_record_known_unknown` and pinning paths are documented.

### Gate False Negatives

The encoding gate may tag genuinely novel events as redundant. Mitigation:
tags are metadata only and reversible by replay; the gate ships off by default
and is promoted only after the forgetting lane shows no recall regression.

### Profile Fragmentation

Two tool profiles risk divergent agent behavior and support burden.
Mitigation: profiles share one handler table; the core profile is a strict
subset; `memory_capabilities` documents the delta at runtime.

### Cache-Ordering Staleness

Optimizing for byte-stable prefixes risks serving stale consolidated state.
Mitigation: stability tiers are invalidated by the same log-signature checks
as the query-page cache; staleness is impossible by construction, only
reordering is new.

### Benchmark Drift

New lanes must not silently replace existing public numbers. Mitigation: lanes
are additive; the release gate diffs public claims against lane provenance.

## Increment Plan

### 2.1-alpha.1: Front Door and Profiles

Scope:

- Add `MCP_TOOL_PROFILE` setting and `--profile` serve option.
- Define the core tool set; rewrite `memory_checkout` and
  `memory_capabilities` descriptions as entry point and discovery surface.
- Update quickstarts and workspace instruction block.

Exit criteria:

- core profile lists ≤ 8 tools; full profile is unchanged;
- every core-profile tool description names checkout as the front door;
- tool-adoption lane baseline recorded.

### 2.1-alpha.2: Budgeted, Cache-Stable Checkout

Scope:

- Add `max_tokens` to `context_assemble` and `memory_checkout` with knapsack
  packing and elision reporting.
- Implement stability-tier packet ordering with canonical serialization.
- Add budget and cache diagnostics fields.

Exit criteria:

- budget lane shows graceful degradation;
- repeated same-session checkouts produce byte-identical stable prefixes;
- citation coverage unchanged at every budget level.

### 2.1-beta.1: Compaction Recovery and Doctor

Scope:

- Add `session-resumed` hook event and recovery packet assembly.
- Extend `zaxy doctor` with chain, freshness, embedding, cache, profile, and
  hook checks.

Exit criteria:

- recovery packet cites only Eventloom-backed state;
- a scripted compact-then-resume scenario recovers open tasks and accepted
  findings;
- doctor failures each print a remediation line.

### 2.2-alpha.1: Salience Ledger

Scope:

- Define reinforcement event taxonomy; record surfacing/feedback/promotion
  reinforcement.
- Compute salience in projection; expose in checkout diagnostics only (no
  ranking change yet).

Exit criteria:

- salience is fully rebuildable by replay;
- diagnostics show per-memory salience composition;
- zero ranking behavior change.

### 2.2-alpha.2: Forgetting, Cues, and the Encoding Gate

Scope:

- Blend salience into checkout ranking behind a retrieval profile; add the
  attenuation floor with authority/pinned exemptions.
- Capture cue records on append; blend cue overlap at checkout.
- Ship the write-time encoding gate, off by default.
- Proactive interference detection emitting belief-update proposals.

Exit criteria:

- forgetting lane shows no recall loss for explicit queries;
- attenuated memories are labeled, reachable, and restorable;
- gate tags are reversible by replay.

### 2.2-beta.1: Metamemory and Graph-Walk Retrieval

Scope:

- Add `memory_feeling_of_knowing` with calibration lane.
- Add bounded personalized PageRank stage behind a retrieval-profile flag.
- Add procedure mining proposals through consolidation review.

Exit criteria:

- FoK Brier score beats a base-rate predictor;
- PPR lane shows multi-hop lift without single-hop regression;
- mined skills carry trace citations and require review.

### 2.3-alpha.1: Embedding Scale

Scope:

- Kuzu-native HNSW path above a corpus-size threshold; `exact` flag in
  results.
- Opt-in int8 quantization with float rerank.
- `embedding_model@version` stamping, mixed-version detection in doctor,
  lazy re-embedding migration.

Exit criteria:

- scale lane: ≥ 0.95 recall@10 vs. exact at 10^5 vectors with p99 latency and
  resident-byte improvements;
- quantization is opt-in and reported in capabilities;
- no silent cross-version vector comparison.

### 2.3-rc.1: Defaults and Freeze

Scope:

- Promote measured winners to defaults (profile default, salience-on default,
  gate default) per lane evidence.
- Full release gates, docs, migration notes, claim review.

Exit criteria:

- all lanes green with promoted defaults;
- public claims labeled per validation policy;
- migration doc covers every default change with the opt-out.

## Acceptance Criteria

This roadmap is accepted when:

- the core profile makes checkout the single front door without removing any
  capability;
- checkout honors a caller token budget and reports what was elided;
- forgetting exists only as replayable, reversible projection policy over an
  unchanged immutable log;
- attenuated, gated, and mined state is always inspectable and review-gated
  where it proposes abstractions;
- every default change is promoted by a measurement lane, not by assertion;
- the 2.5/3.0 latent roadmap inherits cache-stable ordering, the salience
  ledger, and reconsolidation-targeted credit assignment as ready inputs.
