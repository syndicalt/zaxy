# Zaxy: Event-Sourced Memory Fabric for Cited, Temporal, Multi-Agent AI Systems

- Author: Zaxy project
- Status: Research draft
- Version context: Zaxy 1.1.2

## Abstract

Long-running AI agents do not merely need more context. They need memory that is
replayable, temporal, cited, purpose-aware, and safe to share across agents.
Zaxy was developed to test that thesis in a production-oriented memory system.
The resulting architecture treats memory as an event-sourced temporal knowledge
graph fabric: immutable Eventloom JSONL logs provide the source of truth,
extractors project typed events into graph and source-recall views, retrieval
combines exact, lexical, vector, verbatim, traversal, and active working-set
lanes, and Memory Checkout returns a bounded, cited prompt contract rather than
a raw context dump.

This paper summarizes the theory, mathematics, implementation lessons,
benchmarks, failure modes, and research directions that emerged while building
Zaxy through the 1.1.2 release. The strongest internal same-harness result is
the current74 full 500-question LongMemEval-compatible checkout run: mean score
0.940, Answer@5 0.906, citation coverage 1.000, Recall@1/5/10 of
0.906/1.000/1.000, p95 latency 687.67 ms, and p99 latency 969.10 ms over 5,372
Eventloom events, 948 sessions, and a frozen workload hash
`90fb2307195d7e16b963a2b8a30f03b375bd42a45d41aeaa55423029dd84e3fc`.
The important interpretation is not simply that retrieval is strong. It is that
within Zaxy's event-sourced adaptation of this workload, retrieval and citation
reached a plateau: 47 `synthesis_miss` cases remained and no retrieval-miss
class was observed. That shifted the research agenda from "find more context"
to "compose, rank, verify, and cite the right answer from already-retrieved
evidence." External legal-agent evidence from the Harvey LAB memory-ablation
suite provides a separate downstream validation signal: Zaxy completed 10/10
pinned tasks with mean criterion pass rate 0.788, +0.184 over the
regular/no-memory article row, +0.081 over article-best task rows, and 9/10 task
wins versus the article-best rows.

Other benchmark lanes sharpened the thesis. StateRecoveryBench showed that raw
associative propagation can recover latent state but fails at authority, stale
row rejection, and distractor resistance unless resolved through explicit
provenance metadata. CoordinationBench showed that flat transcripts, markdown
notes, and BM25 worker logs can retrieve text but cannot represent accepted
parent state, conflict review, duplicate consolidation, purpose feedback, or
replayable citation chains. PurposeBench showed that memory quality depends on
the intended action: coding, release, security, research, support, legal, and
coordinate work require different evidence policies even when they query the
same underlying event log.

The central lesson is that agent memory is not only a vector database problem, a
prompt-length problem, or a summary-management problem. It is primarily a
state-management problem under temporal, evidential, and organizational
constraints.

## 1. Motivation

The default approach to agent memory is still a mixture of markdown notes,
conversation summaries, vector indexes, and larger context windows. Each is
useful, but each collapses a different part of the memory problem.

Markdown is easy for humans to edit, but weak as a system-of-record. Updates
overwrite prior state unless history is manually preserved. Relationships are
implicit. The model can read the note, but cannot replay how the note was
formed or verify the sequence of accepted decisions that led to the current
state.

Vector memory retrieves text that is close to a query. This is an important
capability, but similarity is not provenance. A vector hit does not by itself
tell the agent whether a fact is current, stale, accepted, rejected,
superseded, worker-local, parent-approved, cited, or safe to act on.

Long context windows reduce pressure, but do not solve context selection.
Research on long-context models shows that relevant information can be
underused when buried inside long prompts. In Zaxy's language, a larger prompt
is not the same thing as a better checkout. The question is not "how much can
we stuff into the model?" but "what state should the model be allowed to rely
on, and why?"

The Zaxy project began from a stronger claim:

> Agent memory should be treated like a replayable, cited, temporal state
> system. Retrieval is a view over that state, not the state itself.

That claim led to four design principles:

1. The source of truth must be immutable and replayable.
2. Projection layers must be derived, discardable, and rebuildable.
3. Model-facing memory must carry citations, warnings, and answerability
   diagnostics.
4. Shared multi-agent memory must distinguish worker-local findings from
   accepted parent state.

## 2. Related Work

Zaxy sits at the intersection of event sourcing, temporal databases, graph RAG,
retrieval-augmented generation, memory-augmented agents, and associative memory.

### Event Sourcing and Temporal State

Event sourcing stores changes to application state as a sequence of events and
allows state to be rebuilt by replaying those events. This pattern directly
influenced Zaxy's Eventloom-first design: the graph is not the memory; it is a
projection of the memory. Microsoft and Martin Fowler both describe event
sourcing as a design that makes the event log the durable source of state and
enables rebuilding derived views.

Temporal and bi-temporal database ideas influence the way Zaxy models
validity. A fact can be asserted at one transaction time while describing a
valid interval in the world. Zaxy's graph projection therefore preserves
`valid_from` and `valid_to` windows rather than overwriting facts in place.
This lets memory answer both "what is true now?" and "what did we believe then?"

### Retrieval-Augmented Generation and Graph RAG

Classic RAG combines parametric model knowledge with non-parametric retrieved
documents. Zaxy adopts the non-parametric-memory premise but adds stricter
state and provenance requirements. GraphRAG showed that graph-structured
representations can answer questions that are poorly served by flat retrieval,
especially when relationships and summaries over groups matter. Zaxy uses a
similar insight, but applies it to agent events, decisions, files, symbols,
missions, handoffs, and temporal facts rather than only document corpora.

### Long-Term Agent Memory Benchmarks

LongMemEval evaluates five long-term memory abilities for chat assistants:
information extraction, multi-session reasoning, temporal reasoning, knowledge
updates, and abstention. It also frames memory design as indexing, retrieval,
and reading. Zaxy's results support that decomposition but add a fourth stage:
checkout synthesis. Once answer-bearing evidence is retrieved and cited,
the remaining problem is constructing an answerable, auditable context packet.

The LongMemEval-compatible Zaxy reports in this paper should be read as adapted
retrieval/checkout evidence, not as official LongMemEval end-to-end assistant
scores. Zaxy maps the cleaned benchmark sessions into Eventloom document and
memory events, projects them through its graph/source-recall stack, and evaluates
whether checkout retrieves and cites answer-bearing session identities and
answer surfaces. That protocol is useful for testing Zaxy's memory fabric, but
it differs from the original chat-history assistant-interaction setup and is
therefore not directly comparable to published LongMemEval leaderboard or
end-to-end answer-accuracy numbers.

MemGPT frames agent context as a virtual memory problem: keep main context
small and page in external memory deliberately. Zaxy agrees with the virtual
context framing but replaces opaque paging with a cited Memory Checkout
contract. A-MEM and related agentic memory systems emphasize dynamically
organized memory. Zaxy's version of dynamic organization is event-backed:
reinforcement, feedback, purpose profiles, and projection artifacts are new
events or derived views, not silent mutation of the source of truth.

### External Legal-Agent Memory Ablations

Zaxy's Harvey LAB evidence is a separate external-validation lane. Harvey LAB is
a legal-agent work-product benchmark built around task deliverables and judged
rubric criteria, not a long-term chat-memory retrieval benchmark. The current
Zaxy report imports externally generated Harvey normalized-result artifacts,
audits memory-search/read calls and judge score files, and compares Zaxy's rows
against the article-published regular/no-memory and task-best rows. This makes
the Harvey result a stronger external downstream signal than an internal
synthetic benchmark, while still requiring careful interpretation: the metric is
criterion pass rate, article-relative rows are published disclosures, and
non-Zaxy systems were not all rerun by Zaxy in the same process.

### Ranking, Diversity, and Late Interaction

Zaxy uses BM25 as a strong lexical baseline and as part of hybrid retrieval.
BM25 remains difficult to beat for short, exact, named-entity queries.
Maximum Marginal Relevance (MMR) motivates Zaxy's diversity pass: near-duplicate
hits should not crowd out adjacent evidence. ColBERT-style late interaction
motivates the future reranker lane: single-vector similarity loses token-level
alignment, especially for "which item" and source recall questions.

### Associative Memory and Neuroscience-Inspired Directions

Modern Hopfield Networks connect energy-based associative memory and attention.
The neuron-astrocyte associative memory work of Kozachkov, Slotine, and Krotov
suggests that higher-order process layers can improve associative capacity and
partial-cue completion. Zaxy explored this direction experimentally, but
StateRecoveryBench showed the crucial product lesson: raw association recovers
latent state while often pulling in stale or authority-shaped distractors.
The useful direction is not "add a neural memory layer because it is elegant."
It is "use broad association to propose candidates, then resolve them through
explicit authority and provenance."

## 3. Formal Model

This section gives the mathematical shape of Zaxy's memory fabric. The notation
is intentionally implementation-adjacent: each term corresponds to a concrete
system component.

### 3.1 Event Log

Let a session-scoped Eventloom log be an ordered sequence

\[
\mathcal{E}_s = \langle e_1, e_2, \ldots, e_n \rangle .
\]

where each event is

\[
e_i = \bigl(i, t_i, \tau_i, a_i, p_i, h_i, h_{i-1}\bigr)
\]

with sequence number \(i\), timestamp \(t_i\), type \(\tau_i\), actor \(a_i\),
payload \(p_i\), hash \(h_i\), and previous hash \(h_{i-1}\). Hash-chain
integrity is:

\[
h_i = \operatorname{Hash}\!\left(i, t_i, \tau_i, a_i, p_i, h_{i-1}\right)
\]

for a canonical serialization and collision-resistant cryptographic hash
\(\operatorname{Hash}\). A valid log
satisfies:

\[
\forall i \in \{2,\ldots,n\},\qquad
\operatorname{prev}(e_i) = h(e_{i-1}).
\]

In addition, each \(h(e_i)\) must recompute from the sealed event fields and
\(\operatorname{prev}(e_i)\). This gives Zaxy a concrete audit invariant:
missing, reordered, or edited rows are detectable without trusting the graph
projection.

### 3.2 Extraction

Extraction maps events into entities and edges:

\[
\mathcal{X}(e_i) = \bigl(\mathcal{V}_i, \mathcal{R}_i\bigr)
\]

where \(\mathcal{V}_i\) is a set of extracted entities and
\(\mathcal{R}_i\) a set of extracted
relationships. For typed events, Zaxy uses deterministic extractors:

\[
\mathcal{X}(e_i) = \mathcal{X}_{\tau_i}(p_i).
\]

The extractor registry is deliberately schema-disciplined. Adding a new public
event type requires deciding its payload contract and writing extractor tests.
LLM extraction can be useful for unstructured events, but Zaxy's default
posture is deterministic because reproducibility and replay matter more than
maximal recall at ingestion time.

### 3.3 Temporal Projection

The graph projection is a derived view:

\[
\mathcal{G}_t = \mathcal{P}\!\left(\mathcal{E}_{\leq t}\right)
\]

where \(\mathcal{P}\) is a projection function over replayed events. Each
projected fact
has an event provenance pointer:

\[
\pi(f) = \bigl(s, i, h_i\bigr)
\]

and a validity interval:

\[
\operatorname{valid}(f) = [v_{\mathrm{from}}, v_{\mathrm{to}})
\]

Current-state retrieval is:

\[
\mathcal{F}_{\mathrm{now}}(q) =
\left\{f \in \mathcal{G} \mid
v_{\mathrm{from}}(f) \leq t_{\mathrm{now}} < v_{\mathrm{to}}(f)
\right\}.
\]

As-of retrieval is:

\[
\mathcal{F}_{\theta}(q) =
\left\{f \in \mathcal{G} \mid
v_{\mathrm{from}}(f) \leq \theta < v_{\mathrm{to}}(f)
\right\}.
\]

This separates two notions that flat memory often confuses: what was stored in
the log and what was valid in the world at query time.

### 3.4 Hybrid Retrieval

For query \(q\), Zaxy computes candidates from multiple lanes:

- exact entity lookup \(\mathcal{C}_E(q)\)
- keyword/BM25 search \(\mathcal{C}_K(q)\)
- vector similarity \(\mathcal{C}_V(q)\)
- graph traversal \(\mathcal{C}_T(q)\)
- verbatim Eventloom source recall \(\mathcal{C}_S(q)\)
- active working-set projection \(\mathcal{C}_W(q)\)

The candidate pool is:

\[
\mathcal{C}(q) =
\mathcal{C}_E(q) \cup
\mathcal{C}_K(q) \cup
\mathcal{C}_V(q) \cup
\mathcal{C}_T(q) \cup
\mathcal{C}_S(q) \cup
\mathcal{C}_W(q).
\]

Each candidate \(c\) receives lane-specific scores \(s_l(c, q)\) and a fused
score:

\[
S(c, q) =
\sum_{\ell \in \mathcal{L}} w_\ell\,\hat{s}_\ell(c, q)
+ \Delta_{\mathrm{time}}(c, q)
+ \Delta_{\mathrm{purpose}}(c, q)
+ \Delta_{\mathrm{retention}}(c, q).
\]

where \(w_\ell\) are lane weights, \(\hat{s}_\ell\) denotes a normalized
lane score, and the deltas represent temporal proximity,
purpose-conditioned policy, and retention/reinforcement effects.
When lane scores are incommensurable or only ordinally meaningful, Zaxy uses
rank-based fusion such as reciprocal-rank fusion:

\[
S_{\mathrm{RRF}}(c, q) =
\sum_{\ell \in \mathcal{L}_c}
\frac{1}{k + \operatorname{rank}_{\ell}(c, q)} ,
\]

where \(\mathcal{L}_c\) is the set of lanes that returned \(c\) and \(k\) is a
stability constant. This is the practical form that made lexical BM25 and graph
retrieval cooperate in early identity-recall runs.

Zaxy then applies diversity-aware selection. A simplified MMR objective for
selecting the next candidate is:

\[
c^\star =
\arg\max_{c \in \mathcal{C}(q) \setminus \mathcal{A}}
\left[
\lambda S(c, q)
- (1-\lambda)\max_{a \in \mathcal{A}}\operatorname{sim}(c, a)
\right],
\]

where \(\mathcal{A}\) is the already selected set. This prevents repeated
near-identical
rows from filling the context budget while leaving room for adjacent evidence.

### 3.5 Memory Checkout

Memory Checkout is the model-facing projection:

\[
\mathcal{M}(q, p, B) =
\operatorname{Pack}_B\!\left(
  \operatorname{Filter}_{p}\!\left(
    \operatorname{Rank}\!\left(\mathcal{C}(q)\right)
  \right)
\right)
\]

where \(p\) is a purpose profile and \(B\) is a prompt budget. The checkout
packet contains:

- current facts
- citations
- evidence rows
- warnings
- answerability status
- required actions
- active working set
- synthesis candidates and diagnostics when available

Checkout quality can be modeled as:

\[
Q(\mathcal{M}) =
\alpha A(\mathcal{M})
+ \beta\,\operatorname{Cit}(\mathcal{M})
+ \gamma\,\operatorname{Rec}(\mathcal{M})
- \delta\,\operatorname{Warn}(\mathcal{M})
\]

where \(A\) is answerability, \(Cit\) citation coverage, \(Rec\) evidence
recall, and \(Warn\) warning severity. Zaxy uses explicit diagnostics rather
than hiding low-confidence states: missing citations, superseded-only context,
warnings, and degraded projection paths are surfaced to the model.

### 3.6 Authority Filtering

For multi-agent coordination, not all retrieved evidence is equal. Let a row
\(r\) have authority metadata:

\[
\operatorname{auth}(r) =
\bigl(\operatorname{scope}(r), \operatorname{status}(r),
\operatorname{promoted}(r), \operatorname{stale}(r),
\operatorname{superseded\_by}(r)\bigr).
\]

For coordinate-purpose checkout, a row is allowed into accepted parent state
only if:

\[
\operatorname{allow}(r) =
\left(\operatorname{scope}(r) \in
\{\mathrm{parent}, \mathrm{accepted}\}\right)
\land
\left(\operatorname{status}(r) \notin
\{\mathrm{rejected}, \mathrm{deferred}, \mathrm{unsupported}\}\right)
\land
\neg \operatorname{stale}(r)
\land
\operatorname{superseded\_by}(r) = \varnothing .
\]

This is why raw associative retrieval is insufficient. A broad memory mechanism
can propose the right neighborhood, but accepted state requires policy over
authority, freshness, and provenance.

## 4. Architecture

Zaxy's architecture has four primary layers.

### 4.1 Eventloom Source of Truth

Eventloom is the bottom layer. In Zaxy 1.1.2, the adapter targets Eventloom v1
JSONL envelopes from `@eventloom/runtime@1.0.0`: `id`, `type`, `actorId`,
`threadId`, `parentEventId`, `causedBy`, `timestamp`, `payload`, and nested
`integrity.hash` / `integrity.previousHash` fields. Zaxy normalizes these to
its internal `Event` API while preserving legacy top-level Zaxy log replay.

This matters because source compatibility should not force a core semantic
rewrite. Eventloom remains the durable substrate; the graph, checkout, and MCP
interfaces keep stable contracts.

### 4.2 Extraction and Projection

The extraction layer converts typed event payloads into structured entities and
edges. The projection layer stores temporal graph views used for retrieval,
dashboard inspection, graph traversal, inferred-edge diagnostics, source
citations, and codebase mapping. Embedded Kuzu is the default local graph
backend; Neo4j remains a quality and interoperability control backend; pgGraph
and LatticeDB are experimental backend candidates.

The engineering rule is:

> Projections may fail, lag, or be rebuilt. Eventloom must remain useful.

That rule drove the operational design. `zaxy memory status` can inspect logs
without a graph service. `zaxy reproject` rebuilds graph state from immutable
events. `zaxy refresh-context` appends source lifecycle events rather than
patching graph rows directly.

### 4.3 Retrieval and Context Assembly

Retrieval is hybrid. Exact search handles named entities. BM25 handles literal
terms and short queries. Vector search handles phrasing changes. Traversal
adds connected facts. Verbatim Eventloom recall preserves exact source lines.
The active working set keeps the prompt focused on high-value current context.

Context assembly is not a cache. It is a prompt-ready projection over replay
and retrieval. `MemoryFabric.assemble_context` replays recent session events,
queries ranked graph memory, reserves a verbatim source-recall lane, projects a
bounded active working set, and formats the result with citations and warnings.

### 4.4 MCP Interface

MCP gives agents a standard way to discover and call memory tools. Zaxy exposes
append, query, verbatim retrieval, checkout, feedback, replay, invalidation,
coordination, and context lifecycle tools through stdio or SSE. The MCP
specification's JSON-RPC structure maps well to Zaxy's need for typed tools and
stable schemas.

The user-facing lesson from development was that memory is only real if the
agent reliably uses it. Zaxy 1.1.1 added activation persistence: model-visible
AGENTS.md instructions, resume reminders, deterministic capture, CLI checkout
fallbacks, and doctor checks for missing capture. This was not cosmetic. It
addressed a real failure mode observed during development: after resume,
compaction, or tool reload, agents could stop using Zaxy unless the memory
contract was visible and fail-closed.

### 4.5 Observability Without Control-Plane Coupling

Pathlight is observability, not storage. Zaxy 1.1.2 fixed a class of failures
where an unavailable Pathlight collector could block MCP startup. The corrected
contract is that tracing degrades to no-op when the collector is down. Memory
operations must not depend on optional observability infrastructure.

## 5. Applied Engineering Lessons

### 5.1 Event-Sourced Memory Made Bugs Recoverable

The most important engineering choice was making Eventloom the source of truth.
When projection bugs appeared, the fix was not to repair graph rows by hand. It
was to change extraction/projection code and replay. This kept mistakes
auditable and made backend experiments possible without changing the memory
semantics.

### 5.2 Lexical Retrieval Was a Breakthrough, Not a Baseline Detail

An early LongMemEval smoke run showed that vector-only and markdown-vector
paths could miss obvious identity matches. Adding reciprocal-rank fusion and
BM25 lexical candidates raised the 20-question smoke path so Zaxy matched BM25
at 0.800 identity recall@5 and 0.700 mean score. This was a reminder that
"semantic" retrieval is not automatically better for memory. Names, dates,
numbers, and exact phrases are often the highest-signal features.

### 5.3 Citations Became a Product Constraint

Once citations were required, many easy fixes were disallowed. A summary that
looks right but cannot point back to an Eventloom event is not trustworthy
memory. This forced source-aware graph projection, verbatim retrieval lanes,
file-line citations for documents, source nodes, `CITES_SOURCE` edges, and
checkout diagnostics that warn on uncited context.

### 5.4 The Retrieval Plateau Was Real

The current74 LongMemEval-compatible run hit Recall@5 1.000, Recall@10 1.000,
and citation coverage 1.000 within Zaxy's event-sourced checkout adaptation.
The remaining 47 misses were `synthesis_miss`, not retrieval misses. This
changed the roadmap. Adding more retrievers would mostly increase prompt load
unless synthesis improved. The next layer had to be a typed synthesis ledger:
answer type, evidence requirements, normalized values, units, support groups,
exclusions, answer candidates, and proof packets.

### 5.5 Authority Metadata Beat Raw Association

The neuron-astrocyte associative memory direction was intellectually attractive.
StateRecoveryBench made it falsifiable. Raw associative projection reached
state accuracy 1.000 but stale rejection 0.485 and distractor resistance 0.212.
Authority-resolved associative projection kept state accuracy 1.000 while
raising minimal evidence recall to 0.985, stale rejection to 1.000, distractor
resistance to 0.909, and abstention accuracy to 1.000. The lesson was sharp:
association finds neighborhoods; authority resolves state.

### 5.6 Coordination Needed a New Benchmark

LongMemEval was not enough for coordinated agent memory. It evaluates long-term
interactive recall, but not whether worker-local findings remain separate from
accepted parent mission state. Zaxy therefore introduced CoordinationBench and
StateRecoveryBench. This was a methodological breakthrough: once the product
thesis moved beyond single-agent recall, the benchmark had to move too.

### 5.7 Purpose Is Part of Memory

The same fact should not always be ranked or trusted the same way. A security
review, release validation, coding task, legal response, and coordination
handoff need different evidence policies. Purpose profiles became the
retrieval-time ontology: they condition scoring, suppression, retention,
answerability, and diagnostics. PurposeBench's full-lane pass supports the
claim that memory quality is action-conditioned, not merely query-conditioned.

## 6. Empirical Results

Internal metrics in this section are same-harness Zaxy artifacts unless
explicitly marked as external evidence or article-relative disclosure. The
LongMemEval-compatible numbers evaluate Zaxy's event-sourced memory checkout
adaptation, not official LongMemEval end-to-end assistant accuracy.

### 6.1 LongMemEval-Compatible Current74

The current public Zaxy headline is the current74 full 500-question
LongMemEval-compatible checkout run.

Protocol boundary. The cleaned LongMemEval-compatible dataset is transformed
into Eventloom-backed memory events and queried through Zaxy's checkout
protocol. The report measures whether the memory fabric retrieves and cites
answer-bearing identities and answer surfaces. It does not replay the official
LongMemEval assistant-interaction protocol, and should not be compared directly
to published LongMemEval end-to-end answer accuracy. In this paper, "current74"
is the internal archived run label for the gated relative-temporal-anchor
embedded Kuzu report, not a separate benchmark version.

| Metric | Value |
| --- | ---: |
| Cases | 500 |
| Event count | 5,372 |
| Sessions | 948 |
| Subjects | 500 |
| Embedding provider | `hash:1536` |
| Projection backend | embedded Kuzu |
| Mean score | 0.940 |
| Answer@1 | 0.696 |
| Answer@5 | 0.906 |
| Answer@10 | 0.932 |
| Citation coverage | 1.000 |
| Recall@1 | 0.906 |
| Recall@5 | 1.000 |
| Recall@10 | 1.000 |
| Mean identity recall | 0.9682 |
| Mean latency | 413.765 ms |
| p50 latency | 358.434 ms |
| p95 latency | 687.671 ms |
| p99 latency | 969.099 ms |
| Mean approximate tokens | 14,033.496 |
| Miss taxonomy | 47 `synthesis_miss` |

Category summaries show the remaining weak areas are not uniform. Multi-session
queries had mean score 0.9023 and Answer@5 0.8421; knowledge-update queries
had mean score 0.9359 and Answer@5 0.9231; temporal reasoning had mean score
0.9474 and Answer@5 0.9098. All categories retained citation coverage 1.000
and Recall@5 1.000.

Interpretation: within this adapted memory-checkout protocol, Zaxy's evidence
retrieval and citation machinery cleared the top-k recall requirement. The
bottleneck moved to answer synthesis: ordering, arithmetic, temporal interval
composition, direct answer promotion, and answer surface placement.

### 6.2 StateRecoveryBench

StateRecoveryBench evaluates partial-cue accepted-state recovery under stale
rows, authority-shaped distractors, incomplete bridge evidence, and no-safe
answer cases. The official 33-case report uses workload fingerprint
`916201f70da9d058aee80a31f8cf59d92dad59f5fd645f3dfbd3a1b23e7dddad`.

| Baseline | State accuracy | Minimal evidence recall | Stale rejection | Distractor resistance | Abstention accuracy | Citation coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| memory_fabric_checkout | 0.818 | 0.909 | 1.000 | 0.818 | 1.000 | 1.000 |
| associative_projection | 1.000 | 0.803 | 0.485 | 0.212 | 0.848 | 1.000 |
| authority_resolved_associative | 1.000 | 0.985 | 1.000 | 0.909 | 1.000 | 1.000 |
| direct_lexical | 0.697 | 0.606 | 0.394 | 0.121 | 0.848 | 1.000 |
| graph_traversal | 0.818 | 0.742 | 0.515 | 0.061 | 0.848 | 1.000 |
| hash_vector | 0.697 | 0.606 | 0.394 | 0.121 | 0.848 | 1.000 |

Interpretation: raw association is strong at latent state recall and poor at
authority. The production MemoryFabric checkout path is slower than diagnostic
baselines but meets the release guardrails with full citation coverage. The
research path is to combine broad associative proposal with explicit
authority-resolved checkout, not to replace the event-sourced substrate.

### 6.3 CoordinationBench

CoordinationBench evaluates multi-agent mission memory: accepted findings,
conflicts, duplicate consolidation, stale claim rejection, citations, parent
checkout answerability, purpose feedback, and replayability. In the
coordination-real-v1 report, flat and lexical baselines could retrieve evidence
but failed the coordination semantics.

| Baseline | Accepted precision | Accepted recall | Evidence coverage | Citation coverage | Replayable | Accepted-state synthesis |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| markdown_notes | 0.636 | 1.000 | 0.909 | 0.000 | false | 0.000 |
| flat_transcript | 0.273 | 0.600 | 0.909 | 0.000 | true | 0.000 |
| bm25_worker_logs | 0.667 | 0.200 | 1.000 | 0.000 | false | 0.000 |

Interpretation: text retrieval can find claims, but it does not know whether
the parent mission accepted them. Coordination requires a memory model with
worker-local state, review events, promotions, proof packets, and accepted
checkout boundaries.

### 6.4 PurposeBench

PurposeBench evaluates whether purpose profiles apply recall floors,
ontology-specific evidence terms, consequence retention, governed forgetting,
evidence-policy discipline, forbidden-overclaim rejection, and citation
coverage. The tracked purpose-v1 report shows 10 lanes passed and citation
coverage 1.000 in the public-derived holdout metadata. Competitor claims remain
blocked until same-harness adapters exist.

Interpretation: purpose is not metadata decoration. It changes what memory rows
are eligible as current facts, which evidence is required, and which risks must
be surfaced.

### 6.5 Harvey LAB External Legal-Agent Memory Ablation

The Harvey LAB memory-ablation result is the strongest external downstream
validation artifact currently in the repository. It is separate from
LongMemEval: Harvey LAB evaluates legal-agent work products with rubric-based
judging, and the public metric is mean criterion pass rate rather than binary
task all-pass success.

The run used the external Harvey checkout at commit
`29748828133dff83ad2263af353fb035504f8f77`, `gpt-5.5` as generator,
`gpt-5.4-mini` as judge, generator reasoning effort `low`, and temperature
0.000. Zaxy completed all 10 pinned article tasks through Harvey's harness and
judge. The report reloaded 10 Zaxy normalized-result artifacts, checked that
judge score artifacts matched normalized scores, checked memory-call metrics,
and verified that transcripts contained memory-tool evidence.

| System | Tasks | Mean criterion pass rate | Delta vs regular/no-memory | Delta vs article best | Wins vs article best |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zaxy | 10 | 0.788 | +0.184 | +0.081 | 9 |

Runtime and memory-use evidence from the same report:

| Mean total seconds | Total tokens | Memory search calls | Memory read calls |
| ---: | ---: | ---: | ---: |
| 138.786 | 5,951,174 | 30 | 10 |

Task-level results show one loss and nine article-best wins:

| Task | Regular | Article best | Zaxy | Delta vs best |
| --- | ---: | --- | ---: | ---: |
| FTC noncompete | 0.807 | Graphiti 0.790 | 0.895 | +0.105 |
| Change-of-control | 0.667 | GBrain keyword 0.737 | 0.860 | +0.123 |
| Acquisition diligence | 0.469 | raw-rg 0.641 | 0.797 | +0.156 |
| Data-room red flags | 0.520 | LightRAG 0.600 | 0.660 | +0.060 |
| Privacy program | 0.532 | ActiveGraph 0.661 | 0.790 | +0.129 |
| Litigation timeline | 0.652 | GBrain keyword 0.758 | 0.894 | +0.136 |
| Relevance / privilege | 0.701 | GBrain keyword 0.791 | 0.687 | -0.104 |
| Attorney production review | 0.583 | GBrain Gemma / LightRAG 0.708 | 0.750 | +0.042 |
| Privilege log | 0.402 | GBrain keyword 0.598 | 0.634 | +0.036 |
| Subpoena comparison | 0.702 | raw-rg 0.790 | 0.912 | +0.122 |

Interpretation. Harvey LAB gives Zaxy an external, legal-domain, work-product
validation signal. It does not eliminate the need for same-process external
adapters across memory systems, because the article-best and framework rows are
published external disclosures rather than Zaxy-controlled reruns of every
competitor. It does, however, address the reviewer's concern that the paper had
only internal benchmarks: Zaxy has now run through a third-party task suite with
audited artifacts and article-relative baselines.

### 6.6 Release and Quality Gates

As of the 1.1.2 release, the local release gate passed with 2,403 tests, 1
skipped test, 11 deselected tests, and 92.18% coverage. The PR and post-merge
GitHub CI passed lint, integration, package, and Python 3.11/3.12/3.13 jobs.
The PyPI publish workflow succeeded and `zaxy-memory 1.1.2` became the latest
published package.

## 7. Pitfalls, Plateaus, and Breakthroughs

### Pitfall: Treating Memory as Search

Search returns candidates. Memory returns state. The difference appears when a
retrieved row is stale, contradicted, pending, rejected, worker-local, or
uncited. Zaxy repeatedly found that retrieval-only improvements could raise
recall while lowering trust.

### Pitfall: Benchmark-Specific Narrow Fixes

During LongMemEval development, the temptation was to add narrow patterns for
missed answer strings. The sustainable fixes were classes of issue:
query-bound arithmetic, latest-state promotion, structured scalar scoring,
direct boolean evidence, aggregate answer priority, and gated relative temporal
anchors. Each class had to preserve citation coverage and Recall@5.

### Pitfall: Projection as Source of Truth

Graph stores are operationally convenient, but if the graph becomes the source
of truth, bugs become history. Zaxy's replay discipline prevented that. The
graph can be rebuilt; Eventloom cannot be silently rewritten.

### Pitfall: Optional Infrastructure Blocking Core Memory

The 1.1.2 Pathlight fix is a small but important example. Observability should
not block memory operations. Optional layers must fail open or degrade visibly,
not become hidden dependencies of the control path.

### Plateau: Retrieval Saturation

The current74 plateau is the clearest empirical result. Once Recall@5 and
citation coverage reached 1.000, further broad retrieval would mostly increase
the cognitive load on the model. The work shifted to structured synthesis.

### Plateau: LongMemEval Did Not Measure Coordination

LongMemEval was valuable, but it could not tell whether a worker finding was
accepted by a parent coordinator. This led to CoordinationBench and
StateRecoveryBench. The broader lesson is that benchmarks must match the
architecture claim. If the claim is coordinated agent memory, single-agent
recall is necessary but not sufficient.

### Breakthrough: Memory Checkout

Memory Checkout turned memory from a backend query into a model-facing
contract. It tells the model what to trust, what to ignore, what is cited, what
is stale, what requires refresh, and what answer candidates exist. This is the
main abstraction that made the system usable by agents rather than only by
humans.

### Breakthrough: Authority-Resolved Association

The astrocyte-inspired branch did not justify adding a heavy neural layer to
core. It did produce a better insight: broad association is useful for proposal,
but accepted memory requires authority resolution. This is likely a general
principle for agent memory systems.

### Breakthrough: Activation Persistence

An agent memory product fails if the agent stops using it after resume. Zaxy's
activation work made memory use visible and checkable through bootstrap,
checkout, AGENTS.md instructions, capture status, doctor warnings, and CLI
fallbacks. This is an applied systems lesson more than an algorithmic one:
memory must be operationally attached to the agent lifecycle.

## 8. Design Principles Learned

1. Use immutable events as the memory substrate.
2. Treat graph, vector, lexical, and summary layers as projections.
3. Preserve citations at every boundary.
4. Separate recall from authority.
5. Separate retrieval from synthesis.
6. Make purpose part of retrieval policy.
7. Prefer typed deterministic extraction for known events.
8. Keep degraded modes visible.
9. Benchmark the claim you are actually making.
10. Do not add complexity merely because a research direction is elegant.

## 9. Future Research Directions

### 9.1 Structured Synthesis Working Memory

The highest-priority research direction is a typed synthesis ledger. The ledger
should represent answer type, evidence requirements, normalized values, units,
source groups, include/exclude reasons, candidate answer surfaces, and support
packets. Operations such as count, sum, difference, average, argmax, absence,
temporal interval, and graph path explanation should be pure functions over
ledger rows.

This directly targets the current74 miss taxonomy: 47 synthesis misses with no
retrieval-miss class.

### 9.2 Two-Phase Evidence Assembly

Current top-k retrieval and final prompt assembly are too tightly coupled.
Future checkout should overfetch candidates, then build a smaller evidence
packet according to the synthesis plan. Candidate recall and evidence assembly
quality should be measured separately.

### 9.3 Late Interaction and Local Reranking

ColBERT-style late interaction is promising for source recall and answer-item
selection. Zaxy should keep deterministic local defaults, but support optional
late-interaction rerankers through the existing reranker interface.

### 9.4 Authority-Resolved Associative Projection

The StateRecoveryBench result suggests a future hybrid:

1. associative propagation proposes latent state neighborhoods;
2. authority metadata filters stale, rejected, unsupported, and worker-local
   rows;
3. Memory Checkout returns cited accepted state and diagnostics.

This is a better research target than importing a full neuron-astrocyte model
into core prematurely.

### 9.5 Better Benchmarks for Agent Teams

CoordinationBench should expand beyond accepted-state recovery into real team
behaviors: duplicate work avoidance, conflict arbitration, stale handoff
recovery, reviewer latency, cross-agent proof transfer, and failure-driven
learning. LongMemEval-style recall remains useful, but team memory needs team
benchmarks. The current StateRecoveryBench, CoordinationBench, and PurposeBench
workloads are committed as reproducibility artifacts; the next step is external
critique, larger hidden splits, and adapter-ready runner definitions.

### 9.6 Memory Activation and Lifecycle Reliability

The agent lifecycle remains a product and research challenge. Future work
should make checkout/capture/feedback default across Codex, Claude Code,
Cursor, LangGraph, CrewAI, and other runtimes even after resume, compaction,
tool reload, or client update. Memory should not depend on the model remembering
to use memory.

### 9.7 Backend Portability Without Semantic Drift

Embedded Kuzu reduced sidecar friction while preserving graph-native semantics.
pgGraph, LatticeDB, Neo4j, and other backends remain useful, but each must pass
the same projection, temporal, citation, traversal, dashboard, rebuild, and
benchmark gates before becoming default.

### 9.8 External Same-Harness Validation

Zaxy now has Harvey LAB external work-product evidence, but the broader
scientific step remains same-process competitor adapters for systems such as
MemPalace, Mem0, Agent Memory, Graphiti/Zep, and A-MEM-style systems. The
Harvey article-relative rows should remain disclosures unless those systems are
rerun through a common adapter protocol with pinned commits, model settings,
judges, and artifact validation.

## 10. Limitations

Zaxy is heavier than markdown and heavier than a standalone vector database.
The architecture pays for provenance, replay, temporal validity, graph
traversal, and coordination semantics. For casual preference memory, this may
be unnecessary.

The current headline LongMemEval-compatible run uses deterministic local hash
embeddings and a cleaned workload transformed into Eventloom memory events. It
is a strong internal same-harness Zaxy result, not an official LongMemEval
assistant-interaction score and not a universal claim against every external
memory system. Competitor disclosures must remain separate until adapters run
under the same protocol.

The Harvey LAB result is external and downstream, but it is also
article-relative: Zaxy rows were generated through the Harvey harness, while
most non-Zaxy rows are published article disclosures or Harvey-native
comparison artifacts rather than Zaxy-controlled reruns of every framework.

The current synthesis layer is still evolving. Retrieval and citation are
strong, but answer composition remains the main benchmark miss class.

StateRecoveryBench, CoordinationBench, and PurposeBench are project-defined
benchmarks. Their definitions and artifacts are committed, which helps
reproducibility, but they still require external scrutiny, broader workloads,
and hidden or independently generated splits before they can serve as community
standards.

The associative-memory branch is experimental. It shows promise as a proposal
mechanism, but core Zaxy should not absorb heavier learned memory machinery
until a benchmark demonstrates a class of failures that event-sourced
authority/provenance cannot solve.

## 11. Conclusion

The development of Zaxy supports a specific theory of agent memory:

> Durable agent memory is not the largest possible prompt. It is a replayable,
> cited, temporal state fabric that can project the right working set for the
> current purpose.

The strongest evidence is not a single benchmark number, but the pattern across
benchmarks and failures. Within Zaxy's adapted checkout protocol,
LongMemEval-compatible retrieval reached Recall@5 1.000 with citation coverage
1.000, revealing synthesis as the next bottleneck. Harvey LAB added an external
legal-agent work-product signal while preserving the need for pinned,
same-process adapters before making framework-wide leaderboard claims.
StateRecoveryBench showed that association without authority is unsafe.
CoordinationBench showed that text retrieval without accepted-state semantics
cannot support agent teams. PurposeBench showed that memory quality depends on
the action being taken.

Zaxy's architecture therefore converged on Eventloom as the immutable source,
temporal graph projections as derived reasoning views, hybrid retrieval as a
candidate generator, Memory Checkout as the model-facing trust contract, and
purpose/authority policies as the guardrails that make shared agent memory
usable.

The next frontier is not merely better retrieval. It is synthesis working
memory: typed, cited, compact, auditable answer construction over already
retrieved evidence.

## References

Internal Zaxy artifacts:

- Zaxy architecture: `docs/architecture.md`.
- Zaxy retrieval design and benchmark interpretation: `docs/retrieval.md`.
- Zaxy benchmark hub and current74 report: `docs/benchmarks.md`.
- Current74 LongMemEval-compatible report:
  `reports/benchmarks/longmemeval-500-current74-zaxyonly-gated-relative-temporal-anchor-embedded-reuse-20260604/live-benchmark.json`.
- StateRecoveryBench report:
  `reports/benchmarks/state-recovery-v1/state-recovery-benchmark.json`.
- CoordinationBench report:
  `reports/benchmarks/coordination-real-v1/coordination-benchmark.json`.
- PurposeBench report:
  `reports/benchmarks/purpose-v1/purpose-benchmark.json`.
- Harvey LAB external legal-agent report:
  `reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json`.
- Harvey LAB publishable statistics:
  `reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md`.
- Eventloom contract: `docs/eventloom.md`.
- Synthesis research note: `docs/synthesis-context-research.md`.
- Experimental associative memory note: `docs/experimental-associative-memory.md`.
- Competitive positioning: `docs/competitive-positioning.md`.
- Release history and version context: `CHANGELOG.md`.

External references:

- Martin Fowler, "Event Sourcing", 2005:
  <https://www.martinfowler.com/eaaDev/EventSourcing.html>.
- Microsoft Azure Architecture Center, "Event Sourcing Pattern":
  <https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing>.
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks", 2020: <https://arxiv.org/abs/2005.11401>.
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and
  Beyond", 2009: <https://doi.org/10.1561/1500000019>.
- Carbonell and Goldstein, "The Use of MMR, Diversity-Based Reranking for
  Reordering Documents and Producing Summaries", SIGIR 1998:
  <https://doi.org/10.1145/290941.291025>. See also the TIPSTER workshop
  version in ACL Anthology: <https://aclanthology.org/X98-1025/>.
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts",
  2023: <https://arxiv.org/abs/2307.03172>.
- Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization", 2024: <https://arxiv.org/abs/2404.16130>.
- Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through
  Self-Reflection", 2023: <https://openreview.net/forum?id=hSyW5go0v8>.
- Khattab and Zaharia, "ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT", 2020:
  <https://arxiv.org/abs/2004.12832>.
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels",
  2022: <https://arxiv.org/abs/2212.10496>.
- Packer et al., "MemGPT: Towards LLMs as Operating Systems", 2023:
  <https://arxiv.org/abs/2310.08560>.
- Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
  Interactive Memory", 2024: <https://arxiv.org/abs/2410.10813>.
- Xu et al., "A-MEM: Agentic Memory for LLM Agents", 2025:
  <https://arxiv.org/abs/2502.12110>.
- Ramsauer et al., "Hopfield Networks is All You Need", 2020:
  <https://arxiv.org/abs/2008.02217>.
- Kozachkov, Slotine, and Krotov, "Neuron-Astrocyte Associative Memory", 2024:
  <https://arxiv.org/abs/2311.08135>.
- Harvey LAB memory retrieval ablation repository:
  <https://github.com/rushilchugh01/harvey-labs-ablations-and-benchmarks>.
- Chugh, "What Agent Memory Actually Fixes in Legal Workflows", 2026:
  <https://rushilchugh.substack.com/p/what-agent-memory-actually-fixes>.
- Model Context Protocol specification, version 2025-03-26:
  <https://modelcontextprotocol.io/specification/2025-03-26/basic/index>.
