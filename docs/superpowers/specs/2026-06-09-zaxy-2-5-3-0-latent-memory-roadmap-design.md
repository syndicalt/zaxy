# Zaxy 2.5 and 3.0 Latent Memory Roadmap Design

## Purpose

Zaxy 2.5 and 3.0 move agent memory deeper toward long-term contextual
understanding, continual learning, and memory-coupled reasoning while preserving
Zaxy's core architectural claim: Eventloom-backed, cited, temporal memory is the
authority layer.

The goal is not to pretend that an external memory system can rewrite a frontier
model's hidden activations or run end-to-end gradient updates through OpenAI,
Anthropic, DeepSeek, or similar closed providers. The goal is to build the
strongest production-grade outside-the-model approximation of durable latent
memory:

- derived latent artifacts that compact long histories into reusable reasoning
  state;
- prospective passes that prepare memory before action;
- retrospective passes that assign credit after outcomes;
- agent-callable memory operations inside planning, execution, review, and
  reflection loops;
- continual learning signals that update memory projections, procedures,
  confidence, and retrieval policy without silently changing authoritative
  truth.

The guiding rule is:

> Latent memory may influence reasoning. Event-sourced Memory Checkout
> authorizes claims.

## Research Anchors

This roadmap should stay close to current academic work without copying any one
paper's assumptions into production.

- **Reflective Memory Management (RMM)**: RMM separates forward-looking
  prospective reflection from backward-looking retrospective reflection for
  long-term dialogue memory. Zaxy should adopt this split as prospective and
  retrospective memory passes, but ground every generated memory in Eventloom
  citations and authority status. Reference:
  <https://arxiv.org/abs/2503.08026>.
- **MEM1**: MEM1 learns a compact shared state that jointly supports memory
  consolidation and reasoning across long-horizon tasks. Zaxy should not assume
  it can train the base model in the same way, but it can build compact
  rebuildable latent artifacts that serve the same role externally. Reference:
  <https://arxiv.org/abs/2506.15841>.
- **Agentic Memory / AgeMem**: AgeMem treats memory operations as part of the
  agent's action policy. Zaxy should expose memory operations as first-class,
  observable, replayable tools rather than a hidden pre-retrieval heuristic.
  Reference: <https://arxiv.org/abs/2601.01885>.
- **Latent memory and cross-layer memory surveys**: Current surveys describe
  latent memory as reusable internal state, memory tokens, cross-layer token
  pools, KV-like state, and generated latent units. Zaxy should use these ideas
  selectively while preserving inspectability, rollback, and provenance.
  Reference: <https://openreview.net/pdf/180d26775b5edf368b1aea4bcf724855acc29c14.pdf>.

These anchors imply a practical direction: Zaxy can move toward latent
participation in reasoning from the outside, but it must keep the latent layer
derived, versioned, reviewable, and discardable.

## Release Positioning

### Zaxy 2.5: Cited Latent Projection Layer

Zaxy 2.5 introduces latent artifacts as rebuildable projections. This is the
production-realistic release. It adds dense and compact memory state, but keeps
the model-facing trust boundary unchanged.

Primary thesis:

> Zaxy can synthesize compact latent memory from cited temporal state and use it
> to improve reasoning context without making latent memory authoritative.

### Zaxy 3.0: Memory-Coupled Reasoning Runtime

Zaxy 3.0 makes memory management an active part of the agent reasoning loop.
This is the deeper cognitive-runtime release. It adds prospective simulation,
retrospective credit assignment, policy feedback, and optional local-model
experiments.

Primary thesis:

> Agents should learn when and how to use memory through observable memory
> actions, while Zaxy records, evaluates, and constrains those actions with
> replayable provenance.

## Current Architecture Fit

The roadmap does not require replacing the current architecture.

Current Zaxy properties that must remain true:

- Eventloom is the immutable source of truth.
- Embedded Kuzu is the default local projection backend.
- Graph projections are rebuildable and discardable.
- Memory Checkout is the model-facing trust contract.
- Citations, temporal validity, authority status, and review status are
  first-class.
- Generated abstractions do not become authoritative without review or an
  existing authority path.

The latent layer fits as another projection layer:

- Eventloom events and accepted graph state produce latent artifacts.
- Latent artifacts are projected into the graph for provenance and routing.
- Dense payloads live in a sidecar artifact store when they do not fit cleanly
  in the graph.
- Memory Checkout may use latent artifacts for ranking, grouping, planning, and
  context construction.
- Memory Checkout must not cite a latent artifact as truth unless it also emits
  the underlying Eventloom-backed evidence.

## Current Embedded Kuzu Shape

The current embedded Kuzu implementation uses a compact physical schema:

- `Entity`: versioned memory facts with `session_id`, `name`, `entity_type`,
  `summary`, `properties_json`, `valid_from`, `valid_to`,
  `source_event_seq`, and `source_event_hash`.
- `Event`: Eventloom projection spine with `seq`, `hash`, `prev_hash`,
  `event_type`, and `source_thread`.
- `BenchmarkProjection`: benchmark projection reuse markers.
- `RELATES`: the main edge table between entities, with `relation_type`,
  temporal validity, inferred-edge audit fields, source-event provenance, and
  `evidence_json`.
- `NEXT_EVENT` and `PREVIOUS_EVENT`: bidirectional Eventloom hash-chain
  projection.

The embedded backend currently represents semantic edge types through
`RELATES.relation_type` rather than many physical relationship tables. Keyword,
vector, and traversal indexes are built primarily as Python-side read indexes
over Kuzu rows.

This means the first latent projection can use the existing `Entity` +
`RELATES` pattern without a risky schema expansion. A later stable version can
add dedicated tables if evidence shows the generic projection shape is too
opaque or too slow.

## Zaxy 2.5 Architecture

### Latent Artifact Contract

A latent artifact is a derived memory object that may influence retrieval,
ranking, planning, or reflection. It is not authoritative truth.

Minimum fields:

- `artifact_id`: stable content-addressed or deterministic identifier.
- `artifact_type`: one of the approved artifact types.
- `session_id`: memory scope.
- `summary`: short human-readable explanation.
- `source_event_refs`: ordered source Eventloom sequence/hash references.
- `source_entity_refs`: optional graph entity references.
- `valid_from` and `valid_to`: temporal validity window.
- `created_at`: projection time.
- `projection_version`: artifact producer version.
- `method`: deterministic, LLM-assisted, learned scorer, or hybrid method.
- `confidence`: bounded confidence score.
- `review_status`: proposed, accepted, rejected, deferred, or conflicted.
- `authority_status`: non-authoritative by default.
- `purpose_profile`: planning, execution, review, reflection, synthesis, or
  general.
- `reasoning_phase`: phase where the artifact is intended to activate.
- `payload_ref`: reference to dense payload storage.
- `payload_kind`: embedding, centroid, memory_weave, sparse_feature_vector,
  adapter_training_example, or token_candidate.
- `supersedes`: prior artifact IDs replaced by this artifact.
- `stale_at`: optional stale marker.
- `diagnostics`: compact quality, citation, and token-budget metadata.

### Artifact Types

`episode_latent`

- Represents a compact memory state for a session, task, milestone, or
  conversation segment.
- Used for long-horizon continuity and fast reconstruction of relevant
  background.

`procedure_latent`

- Represents a reusable workflow learned from successful or failed traces.
- Used during planning and execution.

`failure_latent`

- Represents a known failure mode, blocker pattern, or repeated mistake.
- Used for prospective risk checks and retrospective diagnosis.

`causal_latent`

- Represents a dense or compact view of a causal neighborhood.
- Used for outcome explanation, counterfactual planning, and distractor
  resistance.

`preference_latent`

- Represents stable user, project, or agent preferences.
- Must be especially strict about temporal validity and supersession because
  preferences change over time.

`memory_weave`

- Represents a purpose-conditioned bundle assembled for one reasoning step.
- This is the main model-facing latent artifact in 2.5.
- It must include the artifact IDs and cited evidence that produced the bundle.

`retrieval_policy_latent`

- Represents learned or scored retrieval preferences for a purpose, domain, or
  agent.
- It may adjust ranking but cannot suppress required warnings, citations, or
  authority checks.

### Projection Shape

For 2.5, latent artifacts should be projected through the existing compact Kuzu
shape:

- create an `Entity` with `entity_type = "latent_artifact"`;
- put stable artifact metadata in `properties_json`;
- link it with `RELATES` edges such as:
  - `derived_from_event`;
  - `covers_entity`;
  - `covers_claim`;
  - `covers_procedure`;
  - `activates_for_purpose`;
  - `supports_memory_weave`;
  - `supersedes_latent_artifact`;
  - `contradicts_latent_artifact`;
  - `derived_from_outcome`.

The sidecar artifact store should hold dense payloads when payloads are too
large or backend-specific:

- embeddings;
- centroids;
- sparse feature vectors;
- serialized memory-weave candidates;
- local adapter training examples;
- token candidate payloads for experimental runtime paths.

The artifact store must be rebuildable from Eventloom plus projection config.
If it cannot be rebuilt, it is cache, not memory.

### Prospective Pass

The prospective pass runs before a planning or execution step.

Inputs:

- current goal or query;
- reasoning phase;
- purpose profile;
- active working set;
- accepted Memory Checkout state;
- causal, procedural, metacognitive, and latent projections.

Outputs:

- candidate memory weave;
- risk and failure-prior notes;
- relevant procedure priors;
- causal predecessor/successor context;
- explicit missing-evidence warnings;
- token budget estimate;
- Eventloom-backed citation bundle.

Rules:

- The pass may use latent artifacts to find and rank context.
- It must surface source citations for any factual claim.
- It must mark inferred, review-pending, stale, or low-confidence latent context.
- It must never silently drop high-authority contradictory evidence.

### Retrospective Pass

The retrospective pass runs after a task, answer, failure, or checkpoint.

Inputs:

- task outcome;
- recent tool and model trace;
- memory weave used during the task;
- citations actually relied upon;
- warnings emitted by checkout;
- user feedback or benchmark result when available.

Outputs:

- retrieval feedback events;
- outcome attribution records;
- proposed causal edges;
- proposed consolidation candidates;
- proposed procedure or failure latent artifacts;
- confidence trajectory updates;
- known-unknown or re-verification records.

Rules:

- The pass writes proposals and feedback, not silent authoritative facts.
- Every update must cite the trace or Eventloom events that caused it.
- Failed tasks are first-class learning signals.
- If attribution is uncertain, the pass should record uncertainty rather than
  inventing a cause.

### Memory Weave Contract

A memory weave is the 2.5 bridge between retrieval and latent participation. It
is a bounded model-facing bundle constructed from latent artifacts plus cited
evidence.

Required fields:

- `weave_id`;
- `goal`;
- `reasoning_phase`;
- `purpose_profile`;
- `artifact_ids`;
- `source_event_refs`;
- `accepted_evidence`;
- `inferred_context`;
- `warnings`;
- `token_budget`;
- `expected_use`;
- `prohibited_use`;
- `created_at`;
- `projection_version`.

The model-facing guidance must be explicit:

- use the weave for orientation, planning, and recall;
- use cited checkout evidence for factual claims;
- treat latent-only context as suggestive;
- re-query when a latent artifact conflicts with accepted evidence.

## Zaxy 3.0 Architecture

### Memory Actions as Agent Policy Surface

Zaxy 3.0 exposes memory operations as first-class actions that an agent can call
during reasoning:

- `weave_memory(goal, phase, purpose)`;
- `simulate_forward(state, horizon, constraints)`;
- `reflect_backward(outcome, trace_ref)`;
- `retrieve_procedure_prior(task)`;
- `retrieve_failure_prior(task)`;
- `query_causal_neighborhood(outcome_or_action)`;
- `propose_memory_update(claim, evidence)`;
- `revise_confidence(claim, evidence, outcome)`;
- `record_known_unknown(question, reason)`;
- `score_memory_action(action, outcome)`.

Every action must append or reference Eventloom events so the memory policy is
auditable and replayable.

### Forward-Pass Thinking

Forward-pass thinking is not a literal gradient forward pass through the base
model. It is an external prospective reasoning pass over memory state before
the agent acts.

Capabilities:

- plan simulation over causal and procedural memory;
- trajectory comparison against success and failure priors;
- retrieval of relevant latent artifacts for each candidate plan;
- expected evidence needs for each plan;
- warning generation for stale, missing, or low-confidence memory;
- memory-weave construction for the chosen plan.

3.0 should make forward thinking iterative:

1. propose candidate plan;
2. weave memory for that plan;
3. check causal/procedural/failure priors;
4. revise or accept the plan;
5. record the memory actions that influenced the decision.

### Backward-Pass Thinking

Backward-pass thinking is not a full backward pass through a closed model's
parameters. It is outcome-driven credit assignment over the agent trace and
memory decisions.

Capabilities:

- identify which retrieved memories were used;
- identify which memories were missing;
- identify stale or misleading artifacts;
- attribute failures to retrieval, synthesis, planning, tool use, missing
  evidence, or bad assumptions;
- update confidence trajectories;
- propose causal edges and procedure revisions;
- generate training examples for optional local scorers or adapters.

3.0 should make backward thinking structured:

1. bind the outcome to a trace and memory weave;
2. classify success, partial success, failure, or unknown;
3. compare cited evidence against final answer or task result;
4. assign credit or blame to memory actions with confidence;
5. emit reviewable updates and replayable feedback.

### Continual Learning Loop

The 3.0 continual learning loop is event-sourced and reversible:

1. observe outcome;
2. run retrospective pass;
3. append feedback and proposed updates;
4. update latent projections;
5. optionally train or tune local scorers/adapters from accepted replay data;
6. evaluate against regression gates;
7. promote only when review and benchmark criteria pass.

Allowed learning targets:

- retrieval policy weights;
- artifact activation policies;
- procedure priors;
- failure priors;
- confidence scoring;
- memory action selection;
- local reranker or small adapter models.

Disallowed by default:

- silent mutation of accepted facts;
- uncited preference updates;
- authority promotion from latent confidence alone;
- benchmark-question-specific tuning;
- irreversible parametric updates without replay data and rollback.

### Optional Local-Model Integration

3.0 may include experimental local-model paths because Zaxy users can control
Ollama or custom inference stacks.

Possible paths:

- memory-token candidates inserted into local prompts;
- model-specific KV or cache experiments when supported by the runtime;
- LoRA or adapter training on accepted replay buffers;
- lightweight scoring models for memory action selection;
- test-time adaptation experiments behind explicit flags.

These must remain optional and clearly labeled experimental. They must not be
required for the core Zaxy trust contract.

## Evaluation Plan

Existing gates remain mandatory:

- LongMemEval-compatible 500-question checkout result;
- Recall@5;
- Answer@5;
- citation coverage;
- latency percentiles;
- token budgets;
- StateRecoveryBench;
- CoordinationBench;
- PurposeBench;
- release check and coverage ratchet.

New 2.5 evaluation lanes:

- latent artifact citation fidelity;
- stale latent rejection;
- memory-weave usefulness;
- token reduction with no citation loss;
- retrieval precision lift from latent projection;
- procedure/failure-prior retrieval accuracy;
- retrospective attribution accuracy on labeled traces.

New 3.0 evaluation lanes:

- task-success lift with memory actions enabled versus disabled;
- forward-plan quality lift from prospective passes;
- failure recovery lift from retrospective passes;
- memory action calibration;
- confidence trajectory calibration;
- continual learning improvement over repeated task families;
- rollback correctness after bad latent updates;
- agent autonomy without authority leakage.

Claims must be labeled carefully:

- **internal benchmark**: project-defined harness;
- **same-harness comparison**: another system or baseline run through the same
  harness;
- **externally anchored**: public benchmark or outside artifact used for
  alignment, not full validation;
- **externally validated**: independent outside run or report.

## Non-Goals

Zaxy 2.5 and 3.0 do not aim to:

- replace Eventloom with dense memory;
- make embeddings authoritative;
- hide memory state inside non-replayable caches;
- train frontier model parameters;
- claim true end-to-end differentiable memory for closed providers;
- optimize for one benchmark at the expense of real memory behavior;
- make latent artifacts mandatory for simple local memory use cases.

## Risks

### Authority Leakage

Latent artifacts may cause agents to rely on plausible but uncited context.
Mitigation: Memory Checkout must surface citations for claims and label
latent-only material as suggestive.

### Temporal Flattening

Dense vectors can blur old and current state. Mitigation: every artifact carries
validity windows, supersession metadata, and stale diagnostics.

### Non-Replayable Learning

Adapters or policy updates can become opaque. Mitigation: learned changes must
reference replay data, config, model versions, and rollback state.

### Context Bloat

Latent memory can become another pile of context. Mitigation: memory weaves have
strict token budgets and must prove token-efficiency lift.

### Benchmark Drift

New layers can accidentally tailor behavior to known benchmark cases.
Mitigation: only class-level functionality changes are allowed, with documented
general behavior and regression tests.

### User Trust

More powerful memory can feel invasive or uncontrollable. Mitigation: expose
artifact inspection, invalidation, stale markers, and authority boundaries in
CLI, MCP, and dashboard surfaces.

## Increment Plan

### 2.5-alpha.1: Latent Artifact Contract

Scope:

- Add latent artifact payload schema.
- Project latent artifacts as `Entity(entity_type="latent_artifact")`.
- Add source-event, validity, method, confidence, review, and authority fields.
- Add artifact store interface with a local file-backed implementation.
- Add inspection CLI for artifacts.

Exit criteria:

- artifacts are rebuildable from Eventloom;
- artifact inspection shows all source events;
- no checkout behavior changes unless explicitly enabled.

### 2.5-alpha.2: Prospective and Retrospective Pass MVP

Scope:

- Add prospective pass that builds a cited `memory_weave`.
- Add retrospective pass that writes feedback and proposed updates.
- Add Memory Checkout diagnostics for memory weaves.
- Add tests for stale, uncited, and low-confidence artifacts.

Exit criteria:

- weave construction is cited and bounded;
- retrospective output is proposal-only;
- no authority leakage.

### 2.5-beta.1: Latent Retrieval and Policy Scoring

Scope:

- Use latent artifacts for retrieval ranking and clustering.
- Add retrieval policy latent artifacts.
- Add token-efficiency diagnostics.
- Add internal latent projection benchmark lane.

Exit criteria:

- measurable lift in precision or token efficiency;
- no citation coverage regression;
- stale latent artifacts are rejected.

### 2.5-rc.1: Production Freeze

Scope:

- Freeze 2.5 latent projection config.
- Run full release gates and 2.5 evaluation lanes.
- Publish claim boundaries and artifact schema.

Exit criteria:

- public numbers do not regress materially;
- latent claims are evidence-backed;
- operational docs are complete.

### 3.0-alpha.1: Memory Actions Runtime

Scope:

- Expose agent-callable memory actions.
- Record action traces in Eventloom.
- Add memory action diagnostics to checkout and dashboard.

Exit criteria:

- every memory action is replayable;
- actions can be disabled without breaking core memory;
- no silent memory mutation.

### 3.0-alpha.2: Forward and Backward Reasoning Loops

Scope:

- Add iterative prospective planning loop.
- Add structured retrospective credit assignment loop.
- Emit outcome attribution records.
- Add task-success and failure-recovery evaluation lanes.

Exit criteria:

- memory actions improve at least one real task-success lane;
- bad attribution is visible and reversible;
- feedback updates remain review-gated.

### 3.0-beta.1: Continual Learning Policy

Scope:

- Train or tune local retrieval/action scorers from accepted replay data.
- Add policy versioning and rollback.
- Add repeated-task-family evaluation.

Exit criteria:

- policy updates improve repeated-task performance;
- rollback restores prior behavior;
- learned policy never overrides authority gates.

### 3.0-beta.2: Optional Local-Model Experiments

Scope:

- Add experimental memory-token or adapter integration for local models.
- Keep provider-agnostic core unchanged.
- Document model/runtime compatibility.

Exit criteria:

- local experiments are isolated behind flags;
- release gates pass with experiments disabled;
- experimental claims are not mixed with stable product claims.

## Acceptance Criteria

This roadmap is accepted when:

- 2.5 is scoped as cited latent projection, not hidden model memory;
- 3.0 is scoped as memory-coupled reasoning runtime, not frontier-model
  retraining;
- Eventloom remains the source of truth;
- Kuzu remains the projection control plane for local default deployments;
- dense payload storage is rebuildable and subordinate to citations;
- prospective and retrospective passes are explicit, observable, and bounded;
- continual learning updates are replayable and reversible;
- all claims require benchmark evidence and careful validation labels.
