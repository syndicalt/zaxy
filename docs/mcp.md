# MCP Interface

For a concise local setup path, see [MCP Quickstart](mcp-quickstart.md).

MCP is Zaxy's primary integration surface. The server exposes memory operations
as typed tools so LangGraph, Claude Desktop, custom agents, or any MCP-capable
client can use the same memory system without linking directly against Python
application code.

## Model Call Rhythm

`memory_checkout` is the front door to Zaxy memory; everything else is
plumbing or power use. Use Zaxy tools in this order:

1. At session start, call `memory_bootstrap`.
2. Before substantial work, call `memory_checkout`.
3. For multi-agent missions, use Coordinate tools to record worker-local
   findings and promote only accepted state.
4. After using retrieved context, call `memory_feedback`.
5. When checkout returns synthesis answer candidates and one materially shapes
   the answer, call `memory_synthesis_artifact`.
6. When an individual synthesis ledger row is used or excluded, call
   `memory_synthesis_evidence`.

## Tool Profiles

The server lists tools through a profile selected by `MCP_TOOL_PROFILE` or
`zaxy serve --profile core|full`. The `core` profile is the default since
2.1.0; restore the previous every-tool listing with `MCP_TOOL_PROFILE=full`
or `zaxy serve --profile full`.
The `core` profile lists only the front-door verb set — `memory_checkout`,
`memory_append`, `memory_query`, `context_assemble`, `memory_feedback`,
`memory_invalidate`, `memory_capabilities`, and the experimental
`memory_feeling_of_knowing` — so models facing a long tool registry select the
right entry point more reliably. Both profiles list `memory_checkout` first:
the front door leads the listing order. Profiles change listing,
not capability: any tool remains callable by name under either profile, and
`memory_capabilities` returns a `profile` block with the active profile name
and, under `core`, the available-but-unlisted tool names so a core-profile
agent can discover and request the full surface.

`memory_append(event_type, actor, payload, thread?)` appends a typed event to
the Eventloom log for the selected session, extracts graph entities and edges,
upserts the selected graph projection, records metrics, and emits a Pathlight
span when tracing is enabled. Payload size is bounded and session IDs are
validated before they affect filesystem paths.

`memory_query(query, temporal_filter?, limit?)` returns ranked context chunks.
The query router validates the string and limit, optionally embeds the query,
runs exact/keyword/vector/traversal search, fuses scores, and returns compact
context suitable for an agent prompt. Temporal filters let a client ask what was
valid at a specific time. Remote SSE requests are constrained to the session
from the configured session header. Results include Eventloom citations when
available so clients can display or replay the source event. Results also
include `score_explanation` metadata for ranking diagnostics.

`memory_verbatim(query, session_id?, limit?)` returns exact source chunks from
the Eventloom log without requiring a graph service. It is the source-recall lane for
questions that need raw transcript turns, document chunks, identifiers, quoted
phrases, or file/source citations. Results include the raw content, BM25 score,
`eventloom://...` citation, source kind, and source metadata such as document
path/line range or transcript source/turn index.

`memory_feedback(entity_name, entity_type, feedback, ...)` records whether a
retrieved graph entity was useful. Positive feedback values, `used` and
`helpful`, append a `memory.reinforced` event with optional `importance`,
`query`, `source`, `score`, `citation`, and `reason` fields. Negative feedback,
`irrelevant`, appends an audit-only `memory.feedback` event and does not delete
or decay existing memory. Fabric-level feedback for assembled `packet_memory`
context also preserves source packet projection metadata. The tool uses the same
session scoping rules as append and query. Pass `purpose` and `outcome` when a
memory item changes a purpose-specific action, such as accepted Coordinate state
supporting a handoff; reinforced memory projection preserves compact purpose
profile, evidence-policy, expected-action, authority, and mission metadata for
future checkout and audit.

`memory_synthesis_artifact(checkout, candidate?, outcome?, ...)` persists the
answer candidates from a Memory Checkout response as a deterministic
`memory.synthesis.artifact.created` event. If a selected `candidate` is
provided, `outcome` must also be provided; Zaxy records the normalized outcome
as `memory.synthesis.used`, `memory.synthesis.rejected`,
`memory.synthesis.corrected`, or `memory.evidence.excluded`. The checkout
payload owns the session scope, so this tool intentionally has no separate
`session_id` override. Candidate feedback must match one of
`diagnostics.synthesis.answer_candidates`; Zaxy canonicalizes the persisted
candidate from checkout diagnostics before appending the outcome event.
Artifacts and candidate outcomes project into graph memory as synthesis
artifact, answer-candidate, ledger-row, and source-group relationships.

`memory_synthesis_evidence(checkout, row, outcome, candidate?, reason?, actor?)`
records synthesis evidence row feedback for one
`diagnostics.synthesis.ledger_rows` item. `used` and `helpful` outcomes append
`memory.evidence.reinforced`; `excluded` appends `memory.evidence.excluded`.
The event preserves the row fact id, source group, citation, support source ids,
optional selected answer candidate, checkout quality, slot plan, and reason.
Use this when a composed answer depends on or explicitly rejects a ledger row
instead of only recording the aggregate candidate outcome.

In Coordinate missions, use `coordination_record_synthesis_artifact` instead of
the generic memory tool when the answer candidate is part of a merge brief,
approval packet, accepted checkout, or handoff. It writes the generic synthesis
artifact and then appends a mission-scoped `coordination.proof_packet.created`
event with accepted finding ids, review refs, promotion refs, worker source
refs, optional handoff event refs, diagnostic pending ids, conflict ids,
excluded row reasons, and non-authoritative row diagnostics. For
`decision_scope="handoff"`, provide `handoff_id` so the proof packet cites the
exact `coordination.handoff.created` sequence and hash.

`memory_skill(action, skill_id, ...)` is a typed helper for procedural memory.
It appends one of the deterministic skill lifecycle events
`skill.proposed`, `skill.validated`, `skill.revised`, `skill.deprecated`,
`skill.contradicted`, `skill.applied`, or `skill.outcome_recorded`, then
projects the event through the same extractor and graph path as
`memory_append`. Use it when an agent learns, validates, revises, applies, or
records outcomes for a reusable procedure. The helper accepts `version`, `name`,
`summary`, `procedure`, `applicability`, `citations`, `task`, `success_score`,
`feedback`, `evidence`, `reason`, `failure_modes`, `rollback`,
`contradiction_reason`, and `supersedes_version` as relevant to the action.
Skill updates are never implicit checkout side effects.

## Alpha Causal And Consolidation Tools

Zaxy 2.0 alpha.1 exposes a narrow causal and consolidation MCP surface.
Alpha.2 extends consolidation with deterministic proposal generation from
Eventloom history while keeping the same non-authoritative review boundary:

`memory_causal_successors(entity_name, relation_type?, depth?, session_id?)`
reads graph-backed causal effects that start at `entity_name`. When
`relation_type` is provided, it must use the causal taxonomy value such as
`caused`, `enabled`, `blocked`, `prevented`, `regressed`, `fixed`, or
`explained`; the server maps that to the corresponding `causal_...` graph
relation label. Results include endpoint references, relation labels,
confidence, method, citation, review status, authority status, evidence, and
path length when available.

`memory_causal_predecessors(entity_name, relation_type?, depth?, session_id?)`
uses the same contract in the incoming causal direction, returning cited causes
that lead to `entity_name`.

`memory_consolidation_candidate(candidate_type, title, summary, source_events,
confidence, method, purpose?, session_id?, actor?)` appends a cited
`consolidation.candidate.created` event and immediately projects it. The
candidate type must be `episode`, `claim`, or `procedure`; `source_events` must
cite Eventloom events with sequence and hash; and the returned event reference
is the durable audit handle for the candidate.

`memory_consolidation_propose_from_log(session_id?, actor?, purpose?, limit?,
max_events?, window_size?)` replays Eventloom history for the selected session,
selects deterministic event segments, and appends generated episode, claim, or
procedure candidates through the same `consolidation.candidate.created`
contract. It is a proposal tool, not an authority tool. It must not mark
generated summaries as current facts, bypass review, or promote accepted-looking
content into authoritative memory. Its output should be presented as cited
review material with Eventloom source-event sequence and hash coverage.

`memory_consolidation_review(candidate_id, status, rationale, session_id?,
actor?)` appends a `consolidation.candidate.reviewed` event. Review status is a
lifecycle disposition, not an authority promotion. Valid review statuses are
`accepted`, `rejected`, `deferred`, and `conflicted`. An `accepted` review
means the candidate received that disposition; it does not make the generated
episode, claim, or procedure authoritative without a separate promotion event.

`memory_consolidation_status(session_id?)` reports proposal and review counts
for the selected session, including pending, accepted, rejected, conflicted,
stale, superseded, and `valid_to`-closed candidates when those states are
projected. Use this for review queues and diagnostics, not as evidence of
external validation.

`memory_consolidation(operation, ...)` is an additive umbrella over the four
tools above with `operation` set to `candidate`, `propose_from_log`, `status`,
or `review`. Remaining arguments pass through unchanged to the corresponding
single-purpose handler, and each operation validates its own required
arguments. The single-purpose tool names remain available and unchanged; the
umbrella is listed in the full profile only.

Procedure mining feeds the same review pipeline from the command line.
`zaxy memory mine-procedures` reads captured `tool.call.completed` and
`command.completed` lifecycle events, mines successful multi-step tool
sequences that recur across at least `--min-support` (default 2) distinct
sessions, and appends each as a review-pending `procedure`
`consolidation.candidate.created` candidate with citations to its source
traces. Mined candidates surface through the existing consolidation status and
review tools and never become authoritative skills without review.

The trust contract is intentionally conservative. Causal edges and
consolidation candidates are proposed, cited, non-authoritative memory surfaces
in alpha. They should be used as diagnostics and operator review material, not
as hidden facts. Memory Checkout keeps this boundary visible by reporting
causal context and consolidation candidates in diagnostics and prompt guidance
separate from deterministic current facts. Checkout diagnostics include
candidate counts by type and review/staleness state so clients can distinguish
pending, accepted, rejected, conflicted, stale, superseded, and closed
candidates. Clients should preserve that separation in their own prompts and
UI, especially when a candidate is pending, stale, conflicted, rejected, or
superseded, when a causal edge is inferred, or when citations do not cover the
claim.

Beta.1 reasoning-loop primitives follow the same boundary. Primitive calls are
observable Eventloom records, not hidden chain-of-thought state: a call records
the primitive name, deterministic phase (`planning`, `execution`, `review`, or
`reflection`), status, result count, and cited evidence count. Belief updates
are appended only as `belief.update.proposed` events with
`authority_status=non_authoritative` and `review_status=pending`. A belief
proposal can be inspected, replayed, and reviewed, but it does not update
current facts unless a separate authority path promotes supported state.
Memory Checkout reports these surfaces under
`diagnostics.reasoning_primitives` and
`diagnostics.belief_update_proposals` so clients can show trace evidence
without granting authority.

Beta.2 adds metacognitive and procedural planning tools on the same boundary.
`memory_record_known_unknown(question, reason, source_events, claim_key, ...)`
records an open, cited, non-authoritative uncertainty item.
`memory_known_unknowns`, `memory_confidence_trajectory`, and
`memory_reverification_needs` replay the Eventloom log to surface open
unknowns, append-only confidence assessments, unresolved conflict clusters, and
explicit re-verification requests. Confidence trajectory events do not overwrite
truth, and re-verification requests remain open until a separate workflow
records a resolution. `memory_plan_from_procedures(goal, ...)` builds a
non-authoritative planning packet from applicable procedural memory; rejected,
conflicted, deprecated, contradicted, stale, superseded, closed, or uncited
procedures are diagnostic or excluded instead of being returned as operational
instructions.

`memory_confidence(operation, ...)` is an additive umbrella over the
confidence and metacognition tools with `operation` set to `claim`
(`memory_claim_confidence`), `trajectory` (`memory_confidence_trajectory`),
`reverification` (`memory_reverification_needs`), `known_unknowns`
(`memory_known_unknowns`), or `record_known_unknown`
(`memory_record_known_unknown`). Remaining arguments pass through unchanged,
each operation validates its own required arguments, and the single-purpose
tool names remain available. The umbrella is listed in the full profile only.

`memory_replay(session_id, from_seq?)` rebuilds session history from the
Eventloom log. This is useful for handoffs, audits, and debugging. In remote SSE
mode, the authenticated session scope is enforced so a client cannot replay a
different session. For long sessions, the CLI replay tool can inspect bounded
Eventloom ranges with `zaxy replay .eventloom/work.jsonl --from-seq N --to-seq M`.

`memory_invalidate(entity_name, entity_type, invalid_at)` closes the validity
window for a graph fact without deleting history. This lets agents correct
memory while preserving provenance.

`memory_capabilities(session_id?, current_task?)` returns the model-facing
memory contract for the active session. It tells the model what Zaxy is, which
tools are available, when to refresh context, which capture paths appear
healthy, and which call should normally happen next. Models should call this at
session start or whenever tool awareness is unclear, then call
`memory_checkout` for the current task. The contract explicitly treats memory
as an ambient loop: session-start awareness is not enough, so models should
refresh memory before major work, after compaction/resume, and before roadmap or
architecture decisions.

`memory_feeling_of_knowing(query, session_id?, cues?)` is an **experimental**
metamemory pre-check: it predicts whether `memory_checkout` would likely
return something for a query, in roughly a millisecond, from in-memory session
state only — no embedding call and no graph query. The response is the
non-authoritative verdict `likely`, `possible`, or `unlikely` plus the raw
score and the signal breakdown behind it (query term count, entity-name bloom
hits, cue hits, salience mass). What it does *not* claim: a verdict is never a
memory answer, never evidence, and never authority — it is a cheap prediction
about a checkout you have not run yet, built from the session's active
entity names alone. Optional `cues` string fields (mission, workspace, tool,
phase) are probed alongside the query terms. Each call appends a
non-authoritative `metacognition.fok.predicted` marker (query hash, verdict,
raw score) so the internal calibration lane can score predictions against
actual checkout outcomes; until that lane reports, treat verdicts as hints
and call `memory_checkout` whenever the answer matters.

`memory_bootstrap(session_id?, current_task?)` is the shorter startup packet for
clients that want one model-facing handoff. It embeds the capabilities manifest,
the first recommended checkout call, deterministic capture status, and a trust
policy for cited current facts, unsupported context, and feedback recording.
Both `memory_bootstrap` and `memory_checkout` record lightweight activity
markers so hooks and dashboards can tell whether Zaxy is still visible in a long
session.

Zaxy also records `memory.reminder.suggested` when lifecycle hooks detect a
session boundary, resume, compaction, long session, long tool run, or
roadmap/status question after stale memory activity. Treat that event as a
runtime nudge: call `memory_bootstrap` if tool awareness is unclear, then call
`memory_checkout` for the current task before answering.

`memory_checkout(query, session_id?, ref?, replay_from_seq?, limit?, max_recent_events?, max_tokens?, purpose?)`
returns the high-level contract an agent should condition on before a turn. It
wraps context assembly with a `# Memory Checkout` prompt, current facts that
exclude superseded context, cited evidence, provenance parsed from
`eventloom://...` citations, retention metadata, warnings, the active working
set, and Checkout diagnostics. Diagnostics include source lane counts, total
citation count, current-fact citation count, current fact count, excluded
superseded context count, warning count, and a `memory_feedback` recommendation
when cited context is returned. Optional `max_tokens` applies a deterministic
prompt token budget: checkout sections are greedily packed, citation-bearing
fields always survive, and diagnostics report `budget_requested`,
`budget_used`, and an `elided` summary so the agent knows recall was truncated
rather than empty. Checkout prompts render in stability tiers (consolidated,
session-scoped, volatile) with a canonical consolidated prefix, and
diagnostics report `stable_prefix_chars` so provider prompt-cache efficiency
is measurable. The same budget is available on the CLI as
`zaxy memory checkout --max-tokens`. `purpose` may be a preset such as `coding`,
`review`, `release`, `security`, `research`, or `coordinate`, or an object with
role, task, risk, time horizon, expected action, evidence policy, retention
policy, and ontology lens fields. The normalized profile is returned in
`purpose`, diagnostics, prompt guidance, and later synthesis feedback payloads
so memory can be conditioned by the intended action instead of query text alone.
Non-general purpose profiles condition retrieval before checkout projection by
adding deterministic profile emphasis terms, purpose-specific recall floors, and
a purpose-selected scoring profile; the applied policy is reported in
`diagnostics.purpose_retrieval_policy`. Purpose suppress rules are
then enforced before checkout projection, and any excluded rows are summarized in
`diagnostics.purpose_policy` and `retention.purpose_policy` so clients can audit
why retrieved material did not become current memory.

### Skill Analytics

When applicable Skill Memory is retrieved,
diagnostics also include a `skills` block and the prompt includes an
`Applicable Skills` section with cited procedure steps. This lane is read-only:
models may follow the guidance, but revisions require a new `memory_skill` or
`memory_append` event. When retrieved skill versions and outcomes include enough
history, diagnostics also include `skill_analytics` and
`procedural_memory`, and the prompt includes skill and procedural diagnostics.
Those sections report applicable, diagnostic, and excluded procedural memory;
read-only promotion candidates; rollback candidates; contradiction counts;
outcome counts; scores; citations; and exclusion reasons so the model can decide
whether to apply, avoid, or explicitly revise a skill.
This is the preferred tool when a model needs a
bounded, auditable working state rather than a raw list of retrieval hits. The
response also includes `guidance` with
model-facing trust and ignore instructions, a recommended follow-up
`memory_checkout` call, and concrete `memory_feedback` payload templates for
cited facts that materially influence the next response. The `quality` block
adds an answerability decision (`answer_from_memory`, `refresh_recommended`, or
`ask_user`), a bounded confidence score, reasons, and any required next action.
Checkout only returns `answer_from_memory` when current facts have current
Eventloom citations and the checkout has no warnings; missing, superseded-only,
uncited, or compacted checkout states ask the model to refresh memory or ask the
user instead of answering from stale context.
When reasoning-loop primitive observations or belief proposals are present,
checkout guidance explicitly tells the model to treat primitive observations as
replayable reasoning trace evidence rather than authority, and to treat belief
updates as proposals until reviewed and promoted by a separate authority path.
When metacognitive state is present, checkout reports
`diagnostics.metacognition` and tells the model to use known unknowns,
confidence trajectories, conflict clusters, and reverify requests as diagnostic
uncertainty signals, not as accepted facts.
When `ref` is supplied, checkout resolves a Git-style memory ref such as `HEAD`
or `refs/heads/main` and filters replay/context to the target event identity.
MCP clients discover this tool through the standard `tools/list` handshake, so
an already-running client must reconnect before the new tool appears in the
model-visible tool registry.

Example checkout response fragment:

The canonical docs/test fixture for this contract is
`docs/examples/memory-checkout-contract.json`; keep examples aligned with that
fixture when changing the tool response.

```json
{
  "current_facts": [
    {
      "content": "Memory Checkout is the model-facing context contract.",
      "citation": "eventloom://zaxy-default/events/1882#abc123def456",
      "source_lane": "graph",
      "entity_name": "memory checkout",
      "entity_type": "decision"
    }
  ],
  "diagnostics": {
    "source_lanes": {"graph": 1},
    "citation_count": 1,
    "current_citation_count": 1,
    "current_fact_count": 1,
    "superseded_contexts_excluded": 0,
    "warning_count": 0,
    "feedback_recommended": true,
    "feedback_tool": "memory_feedback",
    "consolidation_candidates": {
      "candidate_count": 1,
      "candidate_types": ["episode"],
      "pending_count": 1,
      "accepted_count": 0,
      "rejected_count": 0,
      "conflicted_count": 0,
      "stale_count": 0,
      "superseded_count": 0,
      "valid_to_count": 0,
      "authority_status": "non_authoritative"
    },
    "reasoning_primitives": {
      "context_count": 2,
      "phase_counts": {"planning": 1, "review": 1},
      "primitive_counts": {"explain_outcome": 1, "get_claim_confidence": 1},
      "authority_status": "non_authoritative"
    },
    "belief_update_proposals": {
      "proposal_count": 1,
      "pending_count": 1,
      "authority_status": "non_authoritative"
    },
    "skills": {
      "count": 1,
      "items": [
        {
          "skill_id": "python-test-first",
          "version": "1",
          "status": "validated",
          "procedure": ["Write failing test", "Run pytest"],
          "citation": "eventloom://zaxy-default/events/1883#def456abc123"
        }
      ]
    }
  },
  "quality": {
    "answerability": "answer_from_memory",
    "confidence": 0.75,
    "reasons": ["Retrieved current facts with Eventloom citations."],
    "required_action": null
  },
  "guidance": {
    "feedback": {
      "tool": "memory_feedback",
      "payloads": [
        {
          "entity_name": "memory checkout",
          "entity_type": "decision",
          "feedback": "used",
          "actor": "assistant",
          "citation": "eventloom://zaxy-default/events/1882#abc123def456"
        }
      ]
    }
  }
}
```

Model consumption rule: answer from memory only when `quality.answerability` is
`answer_from_memory`. If it is `refresh_recommended`, call the
`quality.required_action` tool before relying on the checkout. If it is
`ask_user`, ask for missing context instead of inventing continuity. When cited
facts materially influence the response, call `memory_feedback` with one of the
listed payloads so Zaxy can reinforce useful context.

For composed questions, checkout diagnostics also include a `slot_plan` contract
that names required retrieval slots such as `source`, `numeric`, and `temporal`,
plus optional `exact` and `semantic` slots. When evidence is incomplete,
`quality.required_action` includes `missing_slots` and `suggested_queries` so the
model can retry checkout before answering. Numeric and aggregation synthesis
bundles are projected into `diagnostics.synthesis.answer_candidates`, including
rank, candidate type, answer value, confidence, supporting source IDs, and
excluded source IDs. Generated synthesis bundles also expose
`diagnostics.synthesis.ledger_rows` with fact IDs, source groups, citations,
values, labels, and include/exclude reasons. After using one of those candidates
in a response, record the checkout and candidate outcome through
`memory_synthesis_artifact`; this keeps composed answers auditable without
mutating source memory.
When a specific ledger row materially supports or is excluded from that answer,
record the row-level outcome through `memory_synthesis_evidence` so future
checkout can reinforce or diagnose the exact cited fact.

In Coordinate missions, synthesis bundles are the proof format for merge
briefs, approval packets, accepted checkout, and handoff answers. Each ledger
row must preserve whether it came from accepted parent state, pending
worker-local findings, rejected findings, stale evidence, or conflict
diagnostics. Accepted checkout synthesis must default to parent-accepted state;
pending worker-local rows may appear only in diagnostics and must be labeled
non-authoritative.

Degraded checkout response fragment:

```json
{
  "current_facts": [
    {
      "content": "Memory Checkout is current.",
      "citation": null,
      "source_lane": "graph"
    }
  ],
  "diagnostics": {
    "citation_count": 0,
    "current_citation_count": 0,
    "current_fact_count": 1,
    "warning_count": 1
  },
  "quality": {
    "answerability": "refresh_recommended",
    "confidence": 0.29,
    "reasons": [
      "Retrieved current facts, but they lack Eventloom citations.",
      "Checkout contains warnings that reduce confidence."
    ],
    "required_action": {
      "tool": "memory_checkout",
      "query": "current decisions, blockers, and next actions for: Memory Checkout is current.",
      "reason": "Refresh memory before major follow-up work, after compaction/resume, or when task scope changes."
    }
  }
}
```

`context_assemble(query, session_id?, replay_from_seq?, limit?, max_recent_events?, max_tokens?)`
returns a prompt-ready bundle containing bounded recent replay plus ranked
retrieval. Optional `max_tokens` applies the same greedy token-budget packing
as `memory_checkout`, reporting `budget_requested`, `budget_used`, and the
`elided` summary in the response's `budget` payload. The assembled retrieval set includes a reserved verbatim Eventloom
source-recall lane by default, and each context item includes `metadata` with an
`assembly_lane` value such as `graph` or `verbatim`.
The response also includes `assembly_policy` and `context_counts` fields so
clients can inspect the active policy and the number of graph, verbatim, and
replay items returned. The prompt begins with an `# Active Memory Working Set`
section, and the JSON response includes `working_set` items for bounded goals,
decisions, tasks, artifacts, blockers, and source anchors.
`context_after_turn(role, content, ...)` first appends a `transcript.turn` event,
then assembles context for the next model call.
`subagent_cleanup(parent_session_id, subagent_session_id, summary, ...)` records
`subagent.cleaned` in the subagent session and returns a handoff bundle with
summary and integrity status. These lifecycle tools are session-scoped under
remote SSE auth just like query and append.

Coordination tools package the parent mission plus worker session workflow:
`coordination_start`, `coordination_worker_create`, `coordination_assign`,
`coordination_report_finding`, `coordination_merge_brief`,
`coordination_checkout`, `coordination_performance_ledger`,
`coordination_approval_packet`, `coordination_apply_approval`,
`coordination_review_finding`, `coordination_promote`, and
`coordination_handoff`. `coordination_record_synthesis_artifact` binds a
mission-scoped Memory Checkout synthesis artifact to a Coordinate proof packet
for briefs, approvals, accepted checkout, and handoffs. Handoff-scoped proof
packets require `handoff_id` and return `handoff_event_ref` with the cited
Eventloom sequence and hash. They preserve
accepted finding ids, non-authoritative diagnostic rows, conflict ids, and
handoff refs as graph-projected coordinator memory, so future checkout can
retrieve both the accepted support and the excluded worker-local rows.
Selected answer-candidate feedback must also match the checkout candidates, so
the proof packet, artifact, and outcome event cannot disagree about support ids.
Coordinate keeps worker-local findings isolated until a coordinator review
promotes accepted state into the parent mission history. The first merge brief,
accepted checkout, inspection handoff records, and performance ledger are
replay-backed and do not require graph availability.
`coordination_checkout` returns promoted parent state by default and includes
the `coordinate` purpose profile: accepted parent state is authoritative,
worker-local pending rows are suppressed unless diagnostics are requested, and
proof packets/handoff refs are retained as coordinator memory. Set
`include_diagnostics` to include pending findings and conflicts as
non-authoritative diagnostics. `coordination_performance_ledger`
returns per-worker outcome metrics such as accepted, rejected, duplicate,
missing-evidence, and test-backed finding rates. `coordination_handoff` appends
a replayable parent mission event with summary, next steps, and risks; later
handoff-scoped proof packets are linked back during mission inspection through
their cited `handoff_event_ref`. `coordination_proof_trace` resolves a proof by
`artifact_id`, `handoff_id`, or `proof_seq` and returns the replayed proof
event, synthesis artifact event, handoff event, accepted finding refs,
review/promotion refs, answer candidates, ledger rows, and non-authoritative
diagnostics. The
approval tools export pending/conflicted findings for remote review and apply
returned decisions as ordinary review events, with promotion only when a
decision explicitly sets `promote`.

MCP dispatch also performs automatic lifecycle capture by default. After each
tool call, Zaxy appends a `tool.call.completed` event to the resolved session
with the tool name, status, argument keys, and a bounded result summary. Raw
argument values are not persisted in the lifecycle payload. Capture is
best-effort: failures while recording metadata do not fail the original MCP
tool call. Set `MCP_LIFECYCLE_CAPTURE_ENABLED=false` to disable this automatic
capture. Server shutdown also records a best-effort `session.ended` event for
the default session when lifecycle capture is enabled, and subagent cleanup
records `subagent.completed` alongside the existing `subagent.cleaned` event.

Run stdio locally:

```bash
zaxy serve
```

At session start, clients or agents should bootstrap model awareness:

```bash
zaxy memory bootstrap --session-id zaxy-default
zaxy memory checkout "current task, project direction, and recent decisions" --session-id zaxy-default
```

`memory_bootstrap` returns the compact session-start handoff: the active
capability manifest, recommended first checkout call, capture status, and trust
policy. During a long session, repeat checkout before important work and after
compaction or resume. Capture meaningful completions with `context_after_turn`
or typed `memory_append`, and reinforce cited context that was actually used
with `memory_feedback`.

Generate first-run MCP client config:

```bash
zaxy ide-config claude-desktop --eventloom-path .eventloom
zaxy ide-config claude-code --install --workspace .
zaxy ide-config cursor --eventloom-path .eventloom
zaxy ide-config cursor --install --workspace .
zaxy ide-config vscode --eventloom-path .eventloom
zaxy ide-config vscode --install --workspace .
zaxy ide-config codex --install --domain zaxy
zaxy ide-config codex --install --codex-config-scope project --codex-trusted-project --workspace .
zaxy ide-config hermes --install
zaxy ide-config claude-desktop --eventloom-path .eventloom --domain zaxy
```

By default, `ide-config` prints copyable config. The `--install` flag is
limited to verified targets: project-local `.mcp.json` for Claude Code,
`.cursor/mcp.json` for Cursor, `.vscode/mcp.json` for VS Code, explicit Codex
TOML scopes, and Hermes Agent's global `config.yaml`. Install mode merges the
`zaxy` server entry into the documented root object, preserves unrelated
servers, and refuses to replace an existing `zaxy` entry unless `--force` is
passed.

Codex is CLI-assisted by default: `zaxy ide-config codex --install` prints the
official `codex mcp add zaxy -- zaxy serve` command. Codex config is kept
workspace-neutral because user-level Codex MCP servers can be reused across
repositories: it does not write graph-backend variables such as `NEO4J_URI`,
`PROJECTION_BACKEND`, or repo identity values such as `EVENTLOOM_PATH`,
`EVENTLOOM_THREAD`, or `ZAXY_DOMAIN`. A bare `zaxy serve` resolves `.eventloom`,
the projection backend, and the domain-prefixed default session from the process
workspace at startup. Direct TOML writes are opt-in through
`--codex-config-scope project|user`. Project-scoped writes target
`.codex/config.toml` and require `--codex-trusted-project` because Codex only
loads project config from trusted projects. User-scoped writes target
`CODEX_HOME/config.toml` or `~/.codex/config.toml`.

`zaxy ide-config codex` without `--install` also prints the Codex CLI command,
because Codex does not use the JSON config shapes consumed by Claude, Cursor, or
VS Code.

Hermes Agent uses a global YAML config, normally `~/.hermes/config.yaml` or
`HERMES_HOME/config.yaml`, with MCP servers under `mcp_servers`. Zaxy keeps the
Hermes server workspace-neutral for the same reason as Codex: the global config
may be reused across repositories. `zaxy ide-config hermes` prints the YAML
fragment. `zaxy ide-config hermes --install` merges it into the Hermes config,
or `--hermes-config /path/to/config.yaml` can target an explicit file. The
generated entry exposes the model-facing memory tools without writing
graph-backend variables or repo-specific `EVENTLOOM_PATH`, `EVENTLOOM_THREAD`,
or `ZAXY_DOMAIN`; `zaxy serve` derives those from the current workspace at
startup.

Embedded runtime ownership: when `PROJECTION_BACKEND=embedded`, one workspace
`zaxy serve` process owns the repo-local LadybugDB graph in read-write mode. Additional
stdio `zaxy serve` processes, including worker/subagent MCP launches, proxy to
that owner through `.eventloom/runtime/zaxy-embedded-owner.sock` instead of
opening LadybugDB themselves. This preserves full graph-backed checkout without
starting in degraded mode.

`zaxy init` and `zaxy doctor` clean stale embedded owner records when no live
owner lock is held. If the owner lock is held but no healthy proxy socket exists,
`doctor` reports `embedded_mcp_runtime` with an action to fully exit stale
Codex/Zaxy processes before retrying. Verify the local process set with:

```bash
ps -ef | awk '/[z]axy serve/ {print}'
```

Install the `zaxy` CLI before generating MCP config:

```bash
pipx install zaxy-memory
# or: pip install zaxy-memory
```

MCP clients cannot launch a server command that does not exist yet. Generated
stdio config uses the resolved executable path by default, preferring the
installed `zaxy` console script and falling back to the current process path.
This is more reliable for GUI clients than assuming they inherit your shell
`PATH`.

Generate observer hook adapters:

```bash
zaxy hooks claude-code --eventloom-path .eventloom --domain zaxy
zaxy hooks codex --eventloom-path .eventloom --domain zaxy
```

For Codex, the default local preset also writes `.codex/zaxy-capture.json`.
Starting Codex through the printed `zaxy activate codex ... --launch` command
ensures that managed watcher is running; explicit supervisors can still call
`zaxy capture start --workspace .`. The watcher reads Codex's local session JSONL
and appends normalized Eventloom observations. It does not proxy provider
traffic or require an OpenAI API key.
For supervised checks, add `--watch-iterations <n>` to run a bounded number of
capture passes to the underlying `zaxy codex-capture --watch` command. Add
`--graph` to `zaxy capture start` when the selected projection backend should
receive newly captured observations during the same pass.

Hook adapters do not proxy tool execution. Agents and tools continue to execute
normally while hooks append lightweight Eventloom observations through
`zaxy hook-event`. The sink is intentionally graph-independent so session stop
and pre-compaction hooks can record provenance even when the selected graph
projection is unavailable.
Custom clients can implement the same contract by emitting normalized triggers
such as `session-start`, `stop`, `precompact`, `checkpoint`, `command`, and
`file-edit`. Command and file-edit triggers become first-class
`command.completed` and `file.edit.applied` events, which lets automatic capture
feed retrieval and working-set projection without storing raw file content.

This deterministic capture path is the default Zaxy onboarding posture. Packet
capture is optional and should be enabled only when raw provider request/response
audit is worth the provider quota and transport-compatibility cost.

These commands print copyable JSON fragments and do not include bearer tokens,
passwords, or admin secrets. Keep remote SSE credentials in the client secret
store or environment, not in committed config.

The verified write targets for future client-specific installers are tracked in
[mcp-install-targets.md](mcp-install-targets.md). That matrix is the guardrail
for deciding when `zaxy init` may write or merge config directly versus printing
copyable instructions.

Generated project-local stdio configs include `ZAXY_DOMAIN` and a
domain-prefixed `EVENTLOOM_THREAD`, such as `zaxy-default`. This prevents
clients that omit `session_id` from accidentally sharing the global `default`
session across different projects. Codex user-level config is the exception: it
must keep Eventloom/session/domain state out of config and let `zaxy serve`
derive those values from the active project. For remote SSE configs, the same
default is sent through the session header.

For local stdio clients, the generated config is intentionally self-contained:
it forces development-mode local settings and defaults to the embedded LadybugDB
projection. Generated stdio configs set `startup_timeout_sec` to `90` so MCP
clients do not kill startup while local indexes are opening or optional
sidecars are warming. If you explicitly choose the optional Neo4j sidecar, Zaxy
can reuse `bolt://localhost:7687` or start a named Docker container,
`zaxy-neo4j`, with the default local credentials. Leave `NEO4J_AUTO_START=false`
when you manage Neo4j yourself; set `NEO4J_AUTO_START=true` only when you want
Zaxy to start that local sidecar.

Run SSE daemon mode:

```bash
zaxy serve --transport sse --host 127.0.0.1 --port 8080
```

Production SSE requires either static bearer auth or OIDC. Static bearer auth
uses `MCP_REMOTE_AUTH_TOKEN` or `MCP_REMOTE_AUTH_TOKEN_FILE`; clients send
`Authorization: Bearer <token>` and a session header such as
`x-zaxy-session-id: agent-1`. OIDC uses `MCP_OIDC_ISSUER`,
`MCP_OIDC_AUDIENCE`, and `MCP_OIDC_JWKS_URL`; clients send an access token and
Zaxy scopes the request from `MCP_OIDC_SESSION_CLAIM`. Production also requires
`MCP_ADMIN_TOKEN` or `MCP_ADMIN_TOKEN_FILE` for replay and invalidation.

The MCP implementation lives in `src/zaxy/mcp_server.py`. Core orchestration
lives in `src/zaxy/core.py`. Security helpers live in `src/zaxy/security.py`.
See [api.md](api.md) for Python-level calls, [configuration.md](configuration.md)
for environment variables, and [security.md](security.md) for remote transport
hardening. The public overview is [site/index.html](../site/index.html), while
[README.md](../README.md) keeps the short command list.

## MCP Tool Contract Snapshot

The v0.6 public-surface guardrail is
`docs/examples/mcp-tool-contract.json`. That MCP tool contract snapshot records
the tool count, tool names, descriptions, required fields, and full input schema
for every public MCP tool exposed by `src/zaxy/mcp_server.py`.

Treat changes to that fixture as public contract changes. A schema change should
be paired with docs, tests, and a changelog note explaining whether the affected
tool is stable, beta, experimental, or internal. The snapshot is intentionally
plain JSON so MCP clients and external adapter authors can inspect it without
importing Python.

## Representative Response Snapshots

The v0.6 response-shape guardrail is
`docs/examples/mcp-response-snapshots.json`. It stores representative,
normalized responses for `memory_bootstrap`, `memory_checkout`,
`memory_query`, `memory_verbatim`, `context_assemble`, `memory_feedback`, and
`memory_synthesis_artifact`, `memory_synthesis_evidence`, and
`coordination_checkout`, covering startup,
checkout, graph retrieval, verbatim source recall, prompt context assembly,
feedback reinforcement, synthesis artifact writes, synthesis evidence row
feedback, and accepted coordination state.

The snapshot deliberately preserves stable client fields rather than full prompt
text. It covers startup sequence shape, recommended next tool, Eventloom status,
checkout fact/citation diagnostics, quality status, purpose profile keys,
feedback template keys, token-efficiency keys, required prompt sections, graph retrieval score
explanation keys, verbatim source metadata keys, assembled context counts,
feedback event identity, synthesis artifact event identity and normalized
candidate outcome, and accepted coordination checkout counts. Update it
only when the client-facing response contract intentionally changes.

## Structured Error Payloads

MCP tool dispatch returns structured JSON error content instead of exposing raw
Python exceptions to clients. The stable shape is:

```json
{
  "error": {
    "code": "unknown_tool",
    "message": "Unknown tool: unknown_tool",
    "remediation": "Call list_tools and retry with one of the advertised tool names."
  }
}
```

The current error codes are:

- `unknown_tool`: the requested tool name is not in the advertised MCP tool
  list.
- `invalid_request`: the tool name exists, but the request payload fails
  validation or does not match the input schema.
- `internal_error`: the request reached server execution and failed for an
  unexpected service, projection, or runtime reason.

Every error payload includes a human-readable `message` and `remediation` so
MCP clients can surface actionable recovery text without parsing logs. Failed
tool calls are still captured as lifecycle observations with the stable error
summary, but secrets from arguments remain subject to the existing redaction
path.
