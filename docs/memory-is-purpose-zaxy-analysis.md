# Zaxy vs. "Memory Is Purpose"

## Executive Summary

Zaxy is closer to the article's desired memory substrate than most AI-memory
systems, but it is not yet the full product the article describes.

Zaxy already has the hard foundation:

- exact episodic record
- hash-chained Eventloom provenance
- temporal graph projection
- deterministic replay
- cited retrieval
- Memory Checkout
- feedback reinforcement
- coordination workflows
- embedded local runtime

The main gap is the article's central thesis:

> Memory is retained consequence. Purpose is the utility function that decides
> what survives.

Today, Zaxy has purpose signals, but they are distributed across query text,
scoring profiles, retention metadata, checkout policy, coordination status, and
feedback events. Zaxy does not yet have a first-class purpose layer that governs
what gets preserved, retrieved, consolidated, forgotten, surfaced, or acted on.

The strongest product direction is not to chase a broad "Company Brain"
immediately. The better wedge is:

> Zaxy as a purpose-conditioned memory control plane for agent work.

Coordinate and Memory Checkout are the right primitives to build around.

## 1. The Article's Core Thesis

The article argues that memory is not the same as storage, retrieval,
summarization, profile persistence, graphing, or larger context windows.

Its shortest claim is:

> Memory is retained consequence.

A past event becomes memory only when it changes future behavior.

### Key Distinctions

| Concept | Meaning |
| --- | --- |
| Knowledge | What was present. |
| Context | What makes stored material usable now. |
| Memory | What survives because it should alter future work. |
| Purpose | The utility function deciding what survives. |
| Forgetting | A governed part of intelligence, not always failure. |

The boulder example makes this concrete:

| Actor | Same Boulder Becomes |
| --- | --- |
| Tired hiker | Seat |
| Navigator | Landmark |
| Trail maintainer | Obstacle |
| Geologist | Sample |
| Poet | Image |
| Passerby | Nothing worth retaining |

The object does not change. The purpose changes. Therefore, the memory changes.

## 2. What a Real Memory Substrate Requires

The implied architecture has five major layers.

| Layer | Requirement |
| --- | --- |
| Exact episodic record | Source events, documents, meetings, traces, tickets, approvals, corrections, outcomes. |
| Shared semantic state | Actors, artifacts, actions, timestamps, sources, evidence, uncertainty, permissions. |
| Purpose layer | Role, goal, task, risk, time horizon, permission boundary, expected action. |
| Governed consolidation and forgetting | Some memories exact, some summarized, some rules, some decayed, some archived. |
| Action feedback | Outcomes and corrections change future memory behavior. |

The article's most important architectural claim is:

> Semantics at ingestion. Ontology at retrieval.

That means the system should preserve rich source structure early, but avoid
prematurely deciding whether an artifact is a complaint, obligation, churn risk,
roadmap signal, invariant, or warning. The task should supply the ontology
later.

## 3. Where Zaxy Is Strong

### 3.1 Exact Episodic Grounding

Zaxy is very strong here.

It has:

- Eventloom append-only JSONL
- hash-chain integrity
- deterministic replay
- Eventloom citations
- source-path and line citations
- transcript and document ingestion
- graph facts tied back to source events
- Memory Checkout diagnostics

This directly addresses the article's warning that semantic memory needs an
exact grounding layer to prevent similarity from being mistaken for
significance.

### 3.2 Graphs as Projections

The article says graphs are useful but dangerous if treated as final meaning.

Zaxy already mostly follows the right philosophy:

- Eventloom is the source of truth.
- Graph backends are projections.
- Embedded Kuzu, Neo4j, pgGraph, and LatticeDB are projection/runtime choices.
- Reprojection from Eventloom is a core capability.
- Backend changes are benchmark-guarded.

This maps well to the article's claim that graphs are projections. Zaxy is not
simply a graph memory. It is closer to an event-sourced memory fabric with
graph-backed retrieval.

### 3.3 Temporal State

Zaxy is strong on temporal memory:

- facts have validity windows
- reasserted facts become new versions
- invalidation closes validity rather than deleting history
- temporal retrieval supports "what was true then vs. now?"

This is directly aligned with the article's claim that agents operate across
decisions, corrections, commitments, dependencies, and changing truth.

### 3.4 Provenance and Auditability

Zaxy's provenance model is one of its strongest differentiators.

The article says enterprise memory needs evidence, permissions, contradictions,
corrections, and outcomes. Zaxy already supports much of the evidence and
provenance side:

- Eventloom citations
- replay
- hash-linked event chains
- source citations
- Checkout quality diagnostics
- inferred-edge audit metadata
- synthesis artifacts
- Coordinate proof packets

This is a real advantage over vector-only memory systems and generic context
stores.

### 3.5 Agent Coordination Memory

Zaxy Coordinate is especially aligned with the article's retained-consequence
thesis.

Coordinate distinguishes:

- worker-local claims
- evidence-backed findings
- conflicts
- reviews
- accepted findings
- promoted parent state
- handoffs

That is much closer to real memory than just storing a transcript.

A worker finding does not automatically become authoritative memory. It becomes
durable project memory only after review and promotion. That is exactly the
kind of consequence-aware memory the article is describing.

## 4. Where Zaxy Falls Short

### 4.1 Purpose Is Not First-Class Enough

This is the central gap.

Zaxy has purpose-like signals:

- query text
- scoring profiles
- temporal filters
- retention metadata
- feedback reinforcement
- Memory Checkout quality policy
- coordination mission state
- accepted vs. pending findings
- Skill Memory applicability

But there is no unified purpose object that says:

- Who is asking?
- What role are they acting in?
- What are they trying to do?
- What risk attaches to error?
- What time horizon matters?
- What action will follow?
- What evidence standard is required?
- What should be retained, decayed, pinned, or ignored?

The article's standard is higher than query-aware retrieval. It wants purpose to
govern memory formation and retrieval.

### 4.2 Ingestion Still Commits to Some Ontology Early

Zaxy's deterministic extractors are a strength for auditability and speed, but
they also introduce a risk the article warns about: premature labeling.

For agent-event memory, this is acceptable and often desirable. Typed events
like `coordination.finding.reported` or `memory.reinforced` should have
deterministic structure.

But for broader company-memory use cases, the same artifact may need different
interpretations:

| Artifact | Possible Memories |
| --- | --- |
| Customer email | Churn risk, legal obligation, feature request, escalation context. |
| Code comment | Onboarding clue, security warning, implicit contract, historical note. |
| Slack debate | Decision rationale, unresolved tension, roadmap signal, customer commitment. |

Zaxy needs more separation between role-neutral semantic substrate and
purpose-specific memory projection.

### 4.3 Retrieval-Time Ontology Is Partial

Memory Checkout is a strong model-facing contract, but it is not yet full
retrieval-time ontology.

Today, different users can ask different queries, but Zaxy does not yet deeply
project different relationship structures for different purposes. A coding
agent, legal reviewer, support agent, and CEO should be able to retrieve
different memories from the same evidence.

Zaxy can approximate this through query, scoring, and profile choices, but it
does not yet make the ontology lens explicit.

### 4.4 Forgetting Is Safe, But Not Purpose-Governed Enough

Zaxy has good safety around forgetting:

- non-destructive retention policies
- decay scoring
- expiration filtering
- identity-preserving compaction
- compaction audit
- projection warnings

But the article's claim is stronger:

> Forgetting should be governed by purpose.

For example:

| Purpose | What Should Survive |
| --- | --- |
| Legal | Exact wording, obligations, dates, approvals. |
| Engineering | Invariants, failed fixes, architectural constraints. |
| Sales | Commitments, renewal blockers, buyer objections. |
| Support | Escalation context, workaround history. |
| Executive | Market patterns, strategic exceptions. |

Zaxy has the primitives for this, but not yet the explicit
purpose-conditioned policy layer.

### 4.5 Zaxy Is Not Yet a Company Brain

Zaxy should not yet claim the full Company Brain category.

It does not yet sit broadly across:

- email
- Slack or Teams
- meeting transcripts
- CRM
- support tools
- legal systems
- product management systems
- enterprise permission models
- workflow automation outcomes

Zaxy is much stronger as agent/project memory than as whole-company memory.

## 5. Deep Comparison

| Article Concept | Zaxy Today | Evaluation |
| --- | --- | --- |
| Memory is retained consequence | Feedback, reinforcement, accepted findings, handoffs. | Directionally aligned, not unified. |
| Exact episodic record | Eventloom, replay, hash chains, citations. | Very strong. |
| Shared semantic state | Extracted entities/edges, docs, transcripts, code, coordination graph. | Strong for agent/project memory. |
| Purpose layer | Query, scoring profiles, retention, coordination status. | Present but fragmented. |
| Semantics at ingestion | Typed extraction, source metadata, temporal windows. | Strong, but sometimes too ontology-committed. |
| Ontology at retrieval | Checkout, retrieval planning, temporal filters, synthesis slots. | Partial. |
| Governed forgetting | Decay/filter policies, compaction audit. | Safe, not purpose-rich enough. |
| Action feedback | `memory_feedback`, synthesis artifacts, Coordinate review/promotion. | Strong early loop. |
| Company Brain | Local/project/agent memory fabric. | Not yet. |
| Dynamic projection | Backend projection plus checkout assembly. | Strong backend story, weaker purpose-ontology story. |

## 6. Strategic Interpretation

The article is a warning against positioning Zaxy as merely:

- a better vector DB
- a temporal graph
- a context manager
- a replayable event log
- an MCP memory server

Those are architectural advantages, but they are not the highest-level product
story.

The better positioning is:

> Zaxy turns agent work into cited, purpose-conditioned state that changes
> future runs.

Or more directly:

> Zaxy is the memory control plane for agent work.

That positioning preserves Zaxy's real strengths:

- replayable state
- temporal truth
- coordination
- evidence
- accepted findings
- checkout
- feedback
- agent activation

It also avoids overclaiming full enterprise Company Brain status too early.

## 7. Recommended Product Direction

### Direction: Build the Purpose Layer

Zaxy should add a first-class `PurposeProfile` or `MemoryPurpose`.

Example shape:

```json
{
  "role": "coding-agent",
  "task": "review",
  "risk": "correctness-critical",
  "time_horizon": "release",
  "expected_action": "approve_or_block",
  "permission_scope": "repo-local",
  "evidence_policy": "cited_current_facts_required",
  "retention_policy": "preserve_invariants_and_failed_attempts",
  "ontology_lens": ["invariant", "regression", "prior_fix", "open_blocker"]
}
```

This should become an input to Memory Checkout:

```text
memory_checkout(
  query="current auth refactor risks",
  purpose={
    role: "reviewer",
    task: "release-review",
    risk: "high",
    expected_action: "block_or_approve"
  }
)
```

Status: the first contract slice is implemented. `Memory Checkout` accepts a
purpose preset or object, returns the normalized profile in the checkout
payload and diagnostics, includes purpose guidance in the prompt, and preserves
the profile on synthesis artifacts plus candidate/evidence feedback. Coordinate
checkout now carries the `coordinate` purpose profile by default, making
accepted parent state authoritative and pending worker-local rows diagnostic
unless explicitly included.

### What This Unlocks

The same evidence could produce different memory for different purposes.

| Purpose | Retrieved Memory |
| --- | --- |
| Coding agent | Prior fixes, invariants, test coverage, failed attempts. |
| Security reviewer | Attack surface, auth assumptions, unresolved risks. |
| Release manager | Blockers, owners, approval status, stale evidence. |
| Product lead | Customer promises, roadmap implications. |
| Legal reviewer | Obligations, exact wording, deadlines, exceptions. |

This would make Zaxy directly embody the article's boulder thesis.

## 8. Proposed Architecture Evolution

### 8.1 Two-Tier Ingestion

Zaxy should separate ingestion into:

| Tier | Function |
| --- | --- |
| Neutral semantic substrate | Actor, artifact, action, source, timestamp, evidence, uncertainty, permission. |
| Purpose projection | Obligation, churn risk, invariant, blocker, commitment, accepted finding. |

This keeps deterministic extraction while reducing premature business labeling.

### 8.2 Purpose-Conditioned Checkout

Memory Checkout should become the main surface where purpose is applied.

New checkout responsibilities:

- select relevant purpose profile **implemented for explicit presets and
  caller-supplied profiles**
- apply role-specific retrieval weights **implemented as deterministic purpose
  query emphasis, profile-specific internal recall floors, and purpose-selected
  scoring profiles for coding, review, release, security, research, and
  coordinate checkouts**
- enforce evidence policy **implemented as diagnostics and guidance; purpose
  suppress rules now filter checkout facts and evidence before projection**
- choose ontology lens **implemented as a model-facing contract; graph
  projection variants remain pending**
- apply purpose-aware retention and decay **partially implemented through
  checkout suppression diagnostics and Coordinate purpose-aware compaction
  projections, plus purpose-specific retrieval decay floors for Coordinate,
  security, release, and review**
- surface action-specific warnings **implemented for Coordinate authority
  boundaries**
- recommend next memory action **existing checkout guidance remains the base
  path**

### 8.3 Purpose-Aware Consolidation

Compaction should not be globally safe or unsafe. It should be safe or unsafe
under a purpose.

| Purpose | Consolidation Rule |
| --- | --- |
| Legal | Preserve exact source language. |
| Engineering | Preserve invariant, source line, and test evidence. |
| Coordination | Preserve accepted parent state and suppress pending worker noise. |
| Support | Preserve workaround history and escalation status. |
| Executive | Preserve trend and exception summaries with evidence links. |

Status: Coordinate compaction projection now accepts `purpose="coordinate"` and
forces an authoritative exemplar policy. Accepted/promoted parent state, proof
packets, and handoff events remain searchable authoritative records; pending,
rejected, deferred, stale, or unpromoted coordination rows are preserved only in
`consolidation_policy` diagnostics with event sequences and identities. This
keeps compaction from turning worker-local diagnostics into parent memory while
still preserving Eventloom as the replayable source of truth.
Other purpose profiles now have explicit consolidation policies too: security,
release, and review preserve all source-backed records; coding and research use
bounded exemplar projections with larger purpose-specific record floors.

### 8.4 Purpose Feedback Loop

Feedback should capture not only whether memory was useful, but useful for what.

Example:

```json
{
  "entity_name": "auth module changed",
  "feedback": "used",
  "purpose": {
    "role": "coding-agent",
    "task": "debug",
    "risk": "high"
  },
  "outcome": "prevented_redundant_investigation",
  "citation": "eventloom://..."
}
```

This turns feedback into training data for future purpose-conditioned retrieval.

## 9. Product Wedge Recommendation

### Do Not Start With Full Company Brain

A full Company Brain would require:

- enterprise integrations
- complex ACLs
- email, chat, and meeting capture
- workflow system integration
- CRM, support, legal, and product connectors
- org-wide governance
- admin surfaces
- enterprise sales motion

That is too broad for the next step.

### Start With Agent Work Memory

The sharper wedge:

> Zaxy is the memory layer for serious agent work: coding, research, release
> validation, multi-agent coordination, and long-running technical projects.

This wedge is:

- closer to the current product
- easier to benchmark
- easier to adopt locally
- more differentiated from enterprise search
- more credible than broad Company Brain claims

Coordinate should be the flagship workflow because it already demonstrates
retained consequence.

## 10. Benchmark Direction

Zaxy now includes a deterministic `purpose-v1` benchmark gate that tests the
article's thesis directly. Run it with:

```bash
python -m zaxy purpose-benchmark --output-dir reports/benchmarks/purpose-v1
```

The archived report at
`reports/benchmarks/purpose-v1/purpose-benchmark.json` passes all ten
internal lanes. This is an internal Zaxy gate, not a public competitor
leaderboard: Semantic Reach and Quarq remain blocked until pinned same-harness
adapters are available.

| Benchmark | What It Measures |
| --- | --- |
| Purpose Recall | Same artifact, different role, correct retained consequence. |
| Ontology Shift | Retrieval lens changes without reingesting source. |
| Consequence Retention | Future run avoids rediscovering prior outcome. |
| Governed Forgetting | Noise decays while obligations and invariants survive. |
| Action Outcome Loop | Corrections and failed attempts alter next execution. |
| Evidence Policy Discipline | High-risk purposes enforce missing and supported evidence fixtures. |
| Broader Profile Fixtures | Support, product, sales, legal, and executive stay local/project-scoped while gaining checkout, compaction, and benchmark fixtures. |
| Neutral Substrate Projection | One neutral customer artifact rebuilds distinct cited support, product, legal, and executive purpose projections. |
| Cross-Role Citation | Same evidence supports different role-specific memories. |
| Accepted-State Discipline | Pending claims do not contaminate authoritative memory. |

These lanes differentiate Zaxy from vector memory, generic graph memory, and
long-context systems by requiring purpose-specific behavior across retrieval,
feedback, retention, consolidation, and accepted-state discipline.

## 11. Suggested Roadmap

### Phase 1: Purpose Profile Contract

Add a typed purpose object used by:

- Memory Checkout
- context assembly
- retrieval scoring
- feedback
- retention policy
- coordination checkout

Goal: make purpose visible and auditable.

### Phase 2: Purpose-Conditioned Retrieval

Implement purpose-specific retrieval policies:

- coding **implemented**
- review **implemented**
- release **implemented**
- coordination **implemented as the `coordinate` preset**
- security **implemented**
- research **implemented**

Each profile should define:

- retrieval weights
- evidence requirements
- retention rules
- warning rules
- ontology lens

### Phase 3: Purpose Feedback Events

Extend memory feedback so outcomes are tied to purpose.

Status: the first feedback slice is implemented. Memory Checkout feedback
templates include the normalized purpose profile for non-general profiles, and
both `memory_feedback` and `record_context_feedback(..., purpose=..., outcome=...)`
persist useful-for-what metadata. Reinforced memory projection stores compact
purpose and authority fields so future checkout can distinguish accepted
Coordinate state used for a handoff from the same source evidence used for a
different action.

Track whether retrieved memory:

- prevented rediscovery
- avoided a failed path
- changed an answer
- blocked an unsafe action
- supported a handoff
- resolved a conflict

### Phase 4: Purpose-Aware Consolidation

Make compaction and decay depend on role, task, and risk.

Do not compact legal obligations the same way as coding hints or support
context.

Status: implemented for the Coordinate production path through purpose-aware
compaction projections and for retrieval-time decay through purpose-specific
half-life and expired-weight floors. Broader consolidation strategies for every
purpose profile now have explicit preserve-all or exemplar policies; future
work is expanding the profile set beyond the current presets.

### Phase 5: Purpose Benchmarks

Status: implemented as the deterministic `purpose-v1` gate. The report covers
Purpose Recall, Ontology Shift, Consequence Retention, Governed Forgetting,
Action Outcome Loop, Evidence Policy Discipline, Broader Profile Fixtures,
Neutral Substrate Projection, Cross-Role Citation, and Accepted-State
Discipline. The gate must pass before purpose-conditioned memory claims are
treated as releaseable. Comparative Semantic Reach or Quarq claims remain
blocked until same-harness adapters are pinned and scored.

Maintain benchmark lanes that show Zaxy can retrieve and preserve different
memories from the same evidence depending on purpose.

This is a stronger internal product claim than generic token savings, but it
does not become a comparative SOTA claim until external products are evaluated
on the same harness.

## 12. Implementation Roadmap for Remaining Weak Areas

Zaxy is now aligned with the article's core direction for agent-work memory:
purpose is typed, visible in checkout, preserved in feedback, used in retrieval
policy, used in retention/compaction policy, and covered by the `purpose-v1`
gate.

The remaining roadmap should focus on the areas where Zaxy is still weaker
than the strongest version of the "memory is purpose" thesis and where Semantic
Reach / Quarq-style claims could pressure the product.

### Priority 1: Purpose Ontology Projection

Problem: Zaxy applies purpose at retrieval and checkout, but graph structure is
still mostly role-neutral. The same evidence should project different
relationship structures for coding, security, release, support, legal, product,
and coordination without reingesting the source.

Implementation:

- Add a `PurposeOntologyLens` contract that maps a `PurposeProfile` to allowed
  entity roles, relationship roles, edge trust multipliers, suppression rules,
  and required source groups. **Implemented for current purpose presets as a
  retrieval-time overlay contract.**
- Store purpose projection metadata as overlay artifacts, not as rewritten
  source facts. **Implemented in checkout diagnostics and the strengthened
  `purpose-v1` Ontology Shift lane.**
- Add purpose-aware graph expansion in checkout so traversal can ask for
  `risk`, `obligation`, `invariant`, `commitment`, `blocker`, `workaround`, or
  `accepted_state` views over the same Eventloom evidence. **Partially
  implemented as checkout/path overlay diagnostics; backend traversal APIs are
  still future work.**
- Add diagnostics showing which purpose lens changed the retrieved graph path.
  **Implemented for checkout overlay role matches and benchmark path
  multipliers.**

Exit criteria:

- One source artifact can produce distinct cited checkout paths for coding,
  security, release, and coordination. **Partially satisfied through overlay
  role/path diagnostics; full cited backend traversal-path projection remains
  pending.**
- `purpose-v1` gains a stronger Ontology Shift lane that verifies edge/path
  differences, not only query-term and scoring-profile differences.
  **Implemented.**
- No Eventloom source event is rewritten to satisfy a purpose projection.
  **Preserved.**

### Priority 2: Evidence Policy Enforcement

Problem: Purpose evidence policy is visible and diagnostic, but some policies
still behave as guidance instead of hard checkout gates. High-risk purposes
need enforceable standards.

Implementation:

- Create a shared `EvidencePolicy` evaluator for checkout, MCP, synthesis
  artifacts, Coordinate promotion, and purpose benchmarks. **Implemented for
  checkout diagnostics, quality, synthesis promotion, and purpose benchmarks;
  MCP and Coordinate promotion wiring remain pending.**
- Support policy modes: `warn`, `suppress`, `block_checkout`, and
  `require_refresh`. **Implemented in the evaluator contract.**
- Make high-risk profiles enforce stricter defaults:
  - security: source citation plus mitigation or risk-owner reference
    **implemented**
  - release: release gate plus changelog/test/package evidence **implemented**
  - coordinate: promotion/review/source event reference **implemented**
  - legal future profile: exact wording plus date/authority references
- Add machine-readable failure reasons and suggested follow-up checkout
  queries. **Implemented.**

Exit criteria:

- Checkout can refuse or mark non-actionable high-risk answers that lack the
  required evidence. **Implemented as `refresh_recommended` with policy
  failure diagnostics and required-action queries.**
- Synthesis artifacts cannot promote unsupported answer candidates for high-risk
  purposes. **Implemented for positive `used` synthesis candidate feedback;
  unsupported high-risk artifacts now preserve promotion-gate failures.**
- Beta readiness includes an evidence-policy fixture for at least security,
  release, and coordinate. **Implemented through an executable beta-readiness
  check and archived `purpose-v1` Evidence Policy Discipline lane.**

### Priority 3: Outcome Learning Loop

Problem: Feedback captures useful-for-what metadata, and Zaxy now uses the
Eventloom outcome history to adjust future checkout ranking and warnings. The
remaining work is to broaden the same outcome categories across every product
surface and compaction policy.

Implementation:

- Normalize outcome categories across MCP feedback, checkout feedback,
  synthesis evidence feedback, Coordinate reviews, and Skill Memory outcomes.
- Add purpose/outcome aggregates:
  - prevented redundant investigation
  - avoided failed path
  - blocked unsafe action
  - changed answer
  - supported handoff
  - resolved conflict
  - caused regression
- Feed aggregates into retrieval ranking as bounded, explainable score features.
- Feed negative outcomes into purpose warnings and suppression candidates.
- Keep all learning non-destructive and replayable from Eventloom.

Exit criteria:

- Repeated positive outcomes increase retrieval priority for that purpose with
  score explanations. **Implemented for replay-derived checkout context
  learning.**
- Repeated negative outcomes produce warnings or suppression candidates.
  **Implemented for replay-derived checkout context learning.**
- `purpose-v1` Action Outcome Loop verifies that outcome history changes a
  future checkout ranking or warning, not only that feedback is persisted.
  **Implemented in the archived `purpose-v1` report.**

### Priority 4: Broader Purpose Profiles Without Company-Brain Overclaiming

Problem: Zaxy is strong for agent/project work but still not a full Company
Brain. It needs more purpose profiles before it can credibly cover wider
organizational memory.

Implementation:

- Add first-class profiles for:
  - support: escalation context, workaround history, customer-impact evidence **implemented**
  - product: roadmap signals, customer promises, tradeoffs, experiment outcomes **implemented**
  - sales: commitments, buyer objections, renewal blockers, account context **implemented**
  - legal: exact obligations, approvals, deadlines, exceptions **implemented**
  - executive: strategic exceptions, market patterns, risk summaries **implemented**
- Keep these profiles local/project-memory oriented first. Do not require broad
  enterprise connectors before the policy model is proven. **Preserved through
  project-local profile scopes and docs positioning tests.**
- Extend purpose-aware compaction rules for each new profile. **Implemented.**
- Add fixtures from documents/transcripts/tickets rather than only agent traces.
  **Implemented through document-backed compaction and `purpose-v1` broader
  profile fixtures.**

Exit criteria:

- Each new profile has explicit retain/suppress/evidence/retention policies.
  **Implemented.**
- Each new profile has at least one checkout, compaction, and benchmark fixture.
  **Implemented.**
- Public positioning still says "agent work memory" unless enterprise
  integrations and permissions are actually implemented. **Preserved.**

### Priority 5: Neutral Substrate for Documents and Transcripts

Problem: Typed agent events are safe to extract deterministically, but broader
documents can be over-labeled too early. Zaxy should preserve role-neutral
semantic substrate first and apply stronger business ontology at retrieval.

Implementation:

- Add neutral extraction records for documents/transcripts:
  `actor`, `artifact`, `action`, `time`, `source`, `quote`, `uncertainty`,
  `permission_scope`, and `candidate_claim`. **Implemented as
  `neutral_substrate` graph records for `document.indexed` and
  `transcript.turn`.**
- Move purpose-specific labels like `legal_obligation`, `churn_risk`,
  `roadmap_commitment`, or `security_warning` into purpose projections with
  source backpointers. **Implemented through neutral purpose projection records
  and the `purpose-v1` Neutral Substrate Projection lane.**
- Add a projection audit that flags irreversible purpose labels written during
  ingestion. **Implemented for document and transcript ingestion payloads.**

Exit criteria:

- Document ingestion can preserve a customer email without choosing exactly one
  meaning at ingest time. **Implemented.**
- Product, support, legal, and executive checkouts can derive different cited
  memories from the same artifact. **Implemented as benchmarked purpose
  projections from one neutral customer artifact.**
- Reprojection can rebuild purpose views from neutral substrate and Eventloom.
  **Implemented for Eventloom-backed neutral extraction and deterministic
  purpose projection records.**

### Priority 6: Purpose Control Plane UX

Problem: The core product should not be only a library/API feature. Operators
need to see which purpose profile was used, what was suppressed, why a checkout
was trusted, and what action the memory changed.

Implementation:

- Add dashboard or static-viewer panels for:
  - active purpose profile
  - checkout evidence policy status
  - suppressed rows and reasons
  - retained consequence history
  - stale or missing evidence refresh suggestions
  - Coordinate accepted-state versus worker-local diagnostics
  **Implemented through the local dashboard Purpose tab/API and the static
  Eventloom viewer Purpose panel, both backed by replay-only
  `purpose_control` summaries.**
- Add CLI inspection:
  - `zaxy memory purpose status`
  - `zaxy memory purpose lanes`
  - `zaxy memory purpose feedback`
  **Implemented as replay-only `memory purpose` subcommands.**
- Keep Coordinate first in the UI, because it is the strongest flagship proof.
  **Implemented by surfacing Coordinate accepted parent rows, pending/stale
  worker diagnostics, approval counts, and proof packet counts in the purpose
  status surface.**

Exit criteria:

- A maintainer can inspect why a checkout changed future work without reading
  raw JSON. **Implemented with human-readable CLI summaries and dashboard
  metrics.**
- Coordinate users can see accepted parent memory, pending worker diagnostics,
  and proof packets in one surface. **Implemented in `zaxy memory purpose
  status` and the dashboard Purpose tab.**
- Purpose diagnostics are available without requiring Neo4j. **Implemented by
  direct Eventloom replay; no projection backend is constructed.**

### Priority 7: Same-Harness Semantic Reach and Quarq Evaluation

Problem: Zaxy now blocks comparative claims correctly, but it still lacks
scored same-harness adapters for Semantic Reach / HyperBinder / Hybi and Quarq.

Implementation:

- Fill the packaged Quarq and Hybi runner manifests with pinned source refs,
  install commands, runtime commands, dataset contracts, and result export
  schemas. **Implemented as pinned unsupported runner manifests: Quarq pins
  `quarqlabs/agent-oss` at
  `b68386048795765d46c87bef5bd88ecfb1301337`; Hybi pins `hybi==0.1.1`
  by PyPI wheel hash.**
- Add strict result audits:
  - exact source ref
  - command argv
  - workload fingerprint
  - result fingerprint
  - locally scored metrics
  - failure logs when unsupported
  **Implemented for runner manifests/results; nonzero runner failures now
  archive stdout/stderr paths before raising.**
- Run them against CoordinationBench and the purpose benchmark where adapter
  capabilities allow. **Not claimable yet: packaged Quarq/Hybi commands are
  explicit unsupported runners until real workload replay adapters exist.**
- Keep disclosure-only status until results are locally reproducible.
  **Implemented; the competitor claim gate still requires completed locally
  scored Quarq and Hybi rows.**

Exit criteria:

- `competitor_claim_gate` can pass only with completed Quarq and Hybi rows.
  **Implemented.**
- Public benchmark docs distinguish unsupported adapter gaps from scored
  performance gaps. **Implemented in `docs/benchmarks.md`.**
- No public copy uses SOTA or superiority language without same-harness results.
  **Maintained by disclosure-only docs and release checks.**

### Priority 8: Representative Purpose Holdouts

Problem: `purpose-v1` is a deterministic internal gate. It proves the contract,
but it is not yet a representative public benchmark.

Implementation:

- Create frozen holdout packs for:
  - coding/review/release over real repo histories
  - security over redacted vulnerability/fix histories
  - support/product/legal over synthetic but realistic document transcripts
  - Coordinate over multi-agent missions with unseen worker conflicts
  **Implemented as
  `reports/benchmarks/purpose-v1/holdouts/public-derived-purpose-v1/`, a
  diagnostic public-derived pack with source disclosures and five frozen
  representative cases.**
- Add metrics for answerability, citation coverage, ontology shift, consequence
  retention, governed forgetting, and action outcome effect.
  **Implemented in `holdout_reports.public-derived-purpose-v1.metrics`, kept
  separate from `passed_lanes`.**
- Archive hash fingerprints and forbid tuning on holdout answers.
  **Implemented with canonical pack fingerprint
  `0d8217bb4e905164305970050ef34c987d7e9b287ce648a1730685f3dd0e61f6` and
  no-tuning limitations in the pack README/report.**

Exit criteria:

- Internal deterministic gates remain fast. **Implemented; holdouts are
  optional/diagnostic and do not affect deterministic lane count.**
- Public-derived holdouts are reported separately from first-party smoke lanes.
  **Implemented as `holdout_reports` and a separate Markdown section.**
- Zaxy can state where it is strong and where it still fails without
  overstating the result. **Implemented through forbidden-overclaim fields,
  source disclosures, and diagnostic holdout status.**

### Recommended Order

1. Purpose Ontology Projection.
2. Evidence Policy Enforcement.
3. Outcome Learning Loop.
4. Broader Purpose Profiles.
5. Neutral Document/Transcript Substrate.
6. Purpose Control Plane UX.
7. Same-Harness Semantic Reach / Quarq Evaluation.
8. Representative Purpose Holdouts.

This order keeps Coordinate and Memory Checkout front-and-center while building
toward broader organizational memory without prematurely claiming full Company
Brain coverage.

## 13. Positioning Recommendation

### Current Positioning

Zaxy is often described as:

> Event-sourced temporal knowledge graph fabric for AI agent memory.

That is accurate, but infrastructure-heavy.

### Better Positioning

> Zaxy is the memory control plane for agent work.

### Expanded Version

> Zaxy turns traces, documents, code, decisions, corrections, findings, and
> outcomes into cited state that changes what future agents do.

### Product-Specific Version

> Zaxy Coordinate lets agent teams investigate independently, resolve conflicts,
> promote only accepted findings, and carry forward one durable project memory.

This is more aligned with the article than generic persistent memory.

## 14. Final Assessment

Zaxy is already strong where the article says most memory systems are weak:

- exact grounding
- provenance
- temporal state
- replay
- graph projection
- coordination
- feedback
- cited checkout

But the article's deepest critique still applies:

> Zaxy needs a stronger purpose layer.

The next product move should be to make purpose explicit, typed, auditable, and
operational.

The best direction is:

> Build Zaxy into a purpose-conditioned memory layer for agent work, with
> Coordinate as the flagship proof case and Memory Checkout as the model-facing
> contract.

That keeps Zaxy focused, technically defensible, and aligned with the article's
most important insight:

> Memory is not what was stored. Memory is what should change the next action.
