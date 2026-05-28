# Zaxy Coordinate Roadmap

Zaxy's next product direction is high-level coordination memory for multi-agent
projects. The core claim is simple: isolated worker agents should be able to
investigate, report, and hand off work without contaminating the authoritative
project history. The coordinator promotes only accepted, cited findings into the
parent mission session.

This extends the existing Zaxy architecture instead of replacing it. Eventloom
remains the append-only source of truth, projection backends remain queryable
views, Memory Checkout remains the model-facing context contract, and MCP
remains the primary integration surface. The new package gives those primitives
a product-level workflow for agent teams.

## Current Implementation Status

The first production slice is implemented:

- `src/zaxy/coordination.py` provides replay-backed mission, worker, assignment,
  finding, review, promotion, and brief state over session-sharded Eventloom logs.
- `MemoryFabric` exposes thin async coordination methods so Python callers can
  use the workflow without going through the CLI.
- `zaxy coordinate start`, `worker create`, `assign`, `report`, `decide`,
  `promote`, `brief`, `checkout`, `ledger`, `approval-packet`,
  `apply-approval`, `handoff`, and `benchmark` are available as CLI commands.
- MCP exposes coordination tools for start, worker create, assignment, finding
  report, merge brief, accepted checkout, performance ledger, approval packet,
  approval application, review, promote, and handoff.
- Deterministic extractors project the coordination event taxonomy into missions,
  workers, assignments, findings, reviews, promotions, conflicts, decisions, and
  handoffs.
- `src/zaxy/coordination_benchmark.py` implements the first deterministic
  CoordinationBench workload, exact scoring, explicit stale-claim detection, a
  flat-eventlog contamination baseline, and JSON/markdown report writing.
- The local dashboard exposes a read-only Coordinate mission view backed by
  Eventloom replay, including mission brief, accepted checkout diagnostics,
  worker ledger, and approval packet payloads.
- Static review export is available as replay-only Markdown with a JSON wrapper
  through `zaxy coordinate review-export` and the dashboard review-export API.
- Dashboard human review controls are available behind explicit opt-in. The
  dashboard remains read-only by default, but `--enable-coordinate-review`
  enables single-finding review/promote actions and JSON approval packet
  application through the same Eventloom-backed coordination review path.
- CoordinationBench now reports same-harness local baselines for flat transcript
  concatenation, markdown notes, and BM25 over worker logs.
- CoordinationBench includes a competitor adapter disclosure contract for Mem0,
  Agent Memory, and ActiveGraph. Until a pinned adapter and workload replay
  contract is configured, these entries are reported as `not_run` with
  `disclosure_only` claim status rather than fake scores. Pinned external
  result files can be ingested with `--competitor-result NAME=PATH`, but Zaxy
  recomputes public metrics locally from case outputs and rejects fingerprint
  mismatches, duplicate case outputs, and adapter manifest mismatches. Completed
  same-harness result ingests now carry a result fingerprint, generated time,
  case count, and adapter manifest in JSON, with the key provenance surfaced in
  the Markdown report.
- CoordinationBench can execute pinned same-harness competitor runner manifests
  with `--competitor-runner NAME=PATH`. The runner contract writes the frozen
  workload, executes a manifest `run_command` argv list without a shell, requires
  the adapter to write the standard result file, and still recomputes all public
  metrics locally through the same strict scorer. Duplicate adapter names across
  result ingestion and runner execution are rejected.
- Each benchmark run now writes a `competitor-runner-manifests/` manifest pack
  for Mem0, Agent Memory, and ActiveGraph. These JSON templates are bound to the
  frozen workload fingerprint and include the exact result path each adapter must
  write, but they are marked `template: true` and contain a placeholder
  `run_command`; Zaxy refuses to execute unfinalized templates, preventing
  placeholder manifests from turning into public same-harness claims.
- Zaxy also ships an installable CoordinationBench Adapter Contract Kit with
  packaged runner-manifest and result JSON schemas, non-executable competitor
  templates, workload export, and validation commands:
  `zaxy coordinate benchmark-adapter export-kit`,
  `zaxy coordinate benchmark-adapter validate-manifest`, and
  `zaxy coordinate benchmark-adapter validate-result`.
- Deterministic source-state conflict detection is implemented. Findings that
  cite the same source/file reference with different `source_sha256` values are
  surfaced as `source_state` conflicts in briefs, diagnostic checkout, approval
  packets, and prompt diagnostics. `zaxy coordinate detect-conflicts` can
  materialize those conflicts as idempotent `coordination.conflict.detected`
  events for graph projection.
- Semantic conflict detection now has an explicit opt-in adapter hook in the
  coordination manager. Adapter output is labeled `semantic`, validated against
  current mission finding IDs, and remains disabled unless configured by the
  caller. A deterministic local lexical polarity adapter is available via
  `LocalSemanticConflictDetector`, `build_semantic_conflict_detector`, and
  `zaxy coordinate brief --semantic-conflicts lexical`. An optional hosted HTTP
  adapter is available through the same factory with a strict
  `zaxy-coordination-semantic-v1` response contract, bounded evidence payloads,
  bearer-token secret support, timeout control, min-confidence filtering, and
  local validation of duplicate or unknown finding IDs before adapter output can
  affect mission state.
- Optional git and test-result metadata capture is available for worker
  findings. `zaxy coordinate report --git-metadata <path>` records read-only
  branch, head, worktree, changed-file, worktree-list, dirty-state, and
  bounded diff-stat evidence using non-locking git commands, while
  `--test-result-json` or `--test-command`/`--test-status` attach structured
  test-result evidence without running tests inside Zaxy.
- A dependency-light Coordinate adapter contract is available through
  `zaxy.adapters.coordination.CoordinationAdapter`. It wraps mission, worker,
  assignment, finding, brief, checkout, approval, and handoff operations as
  JSON-friendly payloads without spawning workers or inferring findings from
  transcripts. LangGraph and CrewAI now expose thin Coordinate helper
  nodes/steps, and `zaxy coordinate adapter-template` prints starters for
  Codex-style local agents, LangGraph, CrewAI, and generic MCP clients.
- `examples/coordinate_three_worker_project.py` demonstrates a complete
  three-worker mission with conflicting worker claims, accepted promotion,
  clean accepted-state checkout, and final handoff.

Still pending:

- External Mem0, Agent Memory, and ActiveGraph adapter packages that replace the
  generated templates with real pinned `run_command` entries and publish
  reproducible runner packages.

## Product Position

Zaxy Coordinate is the coordinator memory layer for agent teams:

- Worker sessions stay isolated.
- Findings carry evidence, confidence, source, and status.
- Conflicts are first-class objects, not buried in transcripts.
- Accepted findings are promoted into one parent mission history.
- The parent session becomes the durable project memory that future agents
  should trust.

The product should stop leading with "persistent memory" or "temporal graph
fabric" in user-facing surfaces. Those are architectural advantages. The
market-facing wedge is governed coordination: run many agents without losing the
plot.

## Architecture

```text
Mission session
  -> objectives
  -> worker assignments
  -> worker sessions
  -> reported findings
  -> evidence ledger
  -> conflicts
  -> coordinator decisions
  -> promoted accepted state
  -> final handoff
```

The parent mission session is the only source for accepted state. Worker
sessions are scoped workspaces for exploration and evidence gathering. The graph
connects them through explicit coordination edges rather than a shared scratchpad.

## Event Taxonomy

Add deterministic event types before adding orchestration UI:

| Event | Purpose |
|-------|---------|
| `coordination.mission.created` | Create the parent mission session and objective. |
| `coordination.worker.created` | Register a worker session under a mission. |
| `coordination.assignment.created` | Assign a scoped question or task to a worker. |
| `coordination.finding.reported` | Record a worker finding with evidence and confidence. |
| `coordination.finding.reviewed` | Mark a finding as accepted, rejected, deferred, or conflicted. |
| `coordination.finding.promoted` | Copy accepted state into the parent mission session. |
| `coordination.conflict.detected` | Link contradictory findings, stale claims, or incompatible decisions. |
| `coordination.decision.recorded` | Preserve the coordinator decision and rationale. |
| `coordination.handoff.created` | Produce the final mission handoff bundle. |

Payloads should stay deterministic and extractor-friendly. Free-form summaries
are allowed, but status, actor, session IDs, evidence references, files, commands,
tests, citations, and confidence should be structured fields.

## Graph Model

Project the event taxonomy into graph objects:

- `Mission`
- `Worker`
- `Assignment`
- `Finding`
- `Evidence`
- `Conflict`
- `Decision`
- `Promotion`
- `Handoff`

Required edges:

- `MISSION_HAS_WORKER`
- `WORKER_HAS_ASSIGNMENT`
- `WORKER_REPORTED_FINDING`
- `FINDING_HAS_EVIDENCE`
- `FINDING_CONFLICTS_WITH`
- `COORDINATOR_REVIEWED_FINDING`
- `FINDING_PROMOTED_TO_PARENT`
- `MISSION_HAS_DECISION`
- `MISSION_HAS_HANDOFF`

Every projected finding and promotion must preserve Eventloom citation metadata.
Checkout must be able to distinguish worker-local claims from parent-accepted
state.

## CLI Surface

The first package should be usable from a shell without a custom orchestrator:

```bash
zaxy coordinate start "ship auth refactor" --mission auth-main
zaxy coordinate worker create --mission auth-main --worker auth-api
zaxy coordinate assign --mission auth-main --worker auth-api "trace API auth failures"
zaxy coordinate report --mission auth-main --worker auth-api --summary "API failures trace to expired JWKS cache handling"
zaxy coordinate brief --mission auth-main
zaxy coordinate decide --mission auth-main --finding finding-id --status accepted
zaxy coordinate promote --mission auth-main --finding finding-id
zaxy coordinate checkout --mission auth-main
zaxy coordinate ledger --mission auth-main
zaxy coordinate approval-packet --mission auth-main
zaxy coordinate apply-approval --mission auth-main --decisions-json '[{"finding_id":"finding-id","status":"accepted","promote":true}]'
zaxy coordinate handoff --mission auth-main
```

The CLI should emit JSON with `--json` and concise operator text by default.
No command should require a graph service when Eventloom replay can answer the
basic request. Graph-backed conflict detection and ranking can degrade with
explicit warnings.

## MCP Surface

Add model-facing MCP tools that mirror the CLI:

- `coordination_start`
- `coordination_worker_create`
- `coordination_assign`
- `coordination_report_finding`
- `coordination_merge_brief`
- `coordination_checkout`
- `coordination_performance_ledger`
- `coordination_approval_packet`
- `coordination_apply_approval`
- `coordination_review_finding`
- `coordination_promote`
- `coordination_handoff`

The existing `memory_append`, `memory_checkout`, `memory_replay`,
`context_assemble`, and `subagent_cleanup` tools remain lower-level primitives.
The coordination tools should use those primitives internally and return a
bounded, cited state object that a high-level coordinator can act on.

## Coordinator Brief

`zaxy coordinate brief` is the flagship daily-use command. It should answer:

- What is the mission?
- Which workers exist and what are they assigned?
- What changed since the last brief?
- Which findings are new?
- Which findings are ready to promote?
- Which findings conflict or duplicate each other?
- Which findings lack evidence?
- Which files, commands, tests, transcripts, or tool outputs support each claim?
- What coordinator decision is needed next?

The brief must separate:

- accepted parent state
- pending worker-local findings
- rejected findings
- deferred findings
- conflicts
- stale or superseded claims

This is the main UX difference between shared memory and governed coordination.

## Accepted-State Checkout

`zaxy coordinate checkout` returns only parent-accepted state by default:

```bash
zaxy coordinate checkout --mission auth-main
```

Worker-local findings should never leak into accepted project memory unless the
coordinator explicitly promotes them. Operators can request non-authoritative
pending and conflict diagnostics explicitly:

```bash
zaxy coordinate checkout --mission auth-main --include-diagnostics
```

This feature is a direct guard against memory contamination.

## Evidence Ledger

Every durable finding should cite evidence. Evidence can include:

- Eventloom event citations
- transcript turns
- tool call summaries
- command output observations
- file edit observations
- source file and line citations
- test command and result metadata
- human review decisions
- external URLs when explicitly captured

Findings without evidence are allowed, but they should be low-trust and ineligible
for automatic promotion.

## Conflict And Drift Detection

Add conflict detection in increasing strength:

1. Exact entity/status conflicts, such as two workers reporting incompatible
   current owners, statuses, or test results.
2. Temporal conflicts, such as a finding based on a superseded decision.
3. Source conflicts, such as two findings citing incompatible file states.
4. Semantic conflicts using local reranking or hosted adapters when configured.

The first release implements deterministic conflict detection first. Semantic
conflict analysis is available only through explicit opt-in adapters, is labeled
`semantic`, and rejects conflicts that reference unknown mission findings. The
first adapter is local lexical polarity detection: it requires shared subject
tokens and an explicit opposite pair such as enabled/disabled, present/missing,
or passing/failing. The hosted HTTP adapter is also available but remains
disabled by default; it accepts only a bounded finding schema and a strict
response schema, then Zaxy revalidates returned finding IDs locally.

## Agent Performance Ledger

Track useful coordination outcomes by worker:

- accepted findings
- rejected findings
- conflicted findings
- findings promoted after revision
- missing-evidence rate
- duplicate-finding rate
- test-backed finding rate
- stale-claim rate

`zaxy coordinate ledger` and `coordination_performance_ledger` now expose the
replay-backed first pass: accepted, promoted, rejected, deferred, conflicted,
pending, missing-evidence, duplicate, test-backed, and explicitly stale metrics
per worker. Stale/superseded claims are detected from deterministic evidence
metadata and surfaced in briefs, diagnostic checkout, approval packets, and the
ledger. Source-state drift is detected when evidence for the same source
reference carries incompatible `source_sha256` values. Semantic drift detection
now has opt-in local lexical and hosted HTTP adapters, but no semantic provider
is enabled by default.

This gives teams a practical reason to keep Zaxy in the loop: it can tell which
agents, prompts, and workflows produce useful work.

## CoordinationBench

CoordinationBench is the public benchmark for the new product direction. It
should test whether a memory/coordinator system can turn parallel worker outputs
into one reliable project history.

### Workload Shape

Each benchmark case should include:

- a mission objective
- 3 to 10 worker sessions
- worker assignments with partial overlap
- true findings with citations
- duplicate findings
- stale findings
- false findings
- conflicting findings
- missing-evidence findings
- final questions that require accepted parent state

### Metrics

Report:

- accepted-finding precision
- accepted-finding recall
- conflict recall
- conflict precision
- stale-claim rejection
- duplicate consolidation
- evidence coverage
- parent-checkout answerability
- citation coverage
- replayability from Eventloom only
- injected tokens
- returned tokens
- brief latency
- promotion latency

### Baselines

Implemented same-harness local baselines:

- flat transcript concatenation
- markdown notes
- BM25 over worker logs

Still gated until they can run through a non-toy pinned adapter contract:

- vector retrieval over worker logs
- Zaxy retrieval without coordination semantics

Competitor adapters should be added only when they can run through a pinned,
reproducible contract. External public claims should stay in disclosure tables
until they are same-harness.
The first disclosure, result-ingestion, and pinned runner execution contract is
implemented for Mem0, Agent Memory, and ActiveGraph; all three remain `not_run`
until a real adapter result or runner manifest is provided.
Every benchmark run writes `competitor-runner-manifests/*.template.json` files
for those three adapters. The files are workload-fingerprint-bound templates for
adapter authors, not executable claims; unedited templates are rejected by the
runner.

The adapter contract kit is exportable with:

```bash
zaxy coordinate benchmark-adapter export-kit --output-dir coordinationbench-kit
zaxy coordinate benchmark-adapter validate-manifest mem0=coordinationbench-kit/templates/mem0.runner-manifest.json --workload coordinationbench-kit/coordination-workload.json
zaxy coordinate benchmark-adapter validate-result mem0=coordinationbench-kit/templates/mem0-result.json --workload coordinationbench-kit/coordination-workload.json
```

This makes the remaining competitor work an external adapter authoring and
publishing task rather than an ambiguous benchmark harness gap.

Pinned result files must be strict JSON objects:

- top-level `name`, `adapter_contract`, and `workload_fingerprint` must match
  the requested adapter and generated workload;
- `manifest` must include nonempty `name`, `display_name`, `adapter_contract`,
  `adapter_version`, `install_command`, `run_command`, `source_url`, and
  `source_ref`;
- `cases` must contain exactly one object per workload case, with no duplicate,
  missing, unexpected, or non-object case outputs;
- every accepted, stale, or conflicted finding ID must exist in the workload;
- public metrics are recomputed locally from accepted findings, stale findings,
  conflicts, returned text, and injected text;
- citation coverage only counts Eventloom-style citations with a positive
  `source_event_seq` and 64-character lowercase hex `source_event_hash`;
- competitor-provided `metrics` fields are ignored.

Pinned runner manifests use the same adapter identity fields and add:

- `workload_fingerprint`, which must match the frozen workload generated by
  Zaxy before execution;
- `run_command`, a nonempty argv array. String commands are rejected; Zaxy does
  not invoke a shell, expand environment variables, or accept adapter-supplied
  output paths;
- runner execution always appends `--workload <path> --output <path>` and then
  scores only the generated result file through the same local scorer.

## Roadmap Phases

### Phase 1: Coordination Core

- Add event schemas and extractors.
- Add graph projection for missions, workers, assignments, findings, evidence,
  conflicts, decisions, promotions, and handoffs.
- Add Python API methods in `MemoryFabric`.
- Add CLI commands for start, worker create, assign, report, brief, decide,
  promote, and handoff.
- Add MCP tools for the same workflow.
- Add unit tests for event construction, extraction, projection, and session
  isolation.

### Phase 2: Coordinator Brief And Accepted Checkout

- Build the merge brief engine.
- Add accepted-state checkout.
- Separate accepted, pending, rejected, deferred, and conflicted facts in prompt
  output.
- Add deterministic conflict detection.
- Add missing-evidence and stale-claim warnings.
- Add docs and examples for a 3-worker project.
  The first runnable example is `examples/coordinate_three_worker_project.py`.

### Phase 3: CoordinationBench

- Define the workload schema.
- Generate deterministic benchmark fixtures.
- Add scorer for accepted findings, conflicts, evidence, citations, replay, and
  token cost.
- Add CLI commands for benchmark generation and execution.
- Publish a first same-harness report against local baselines.

### Phase 4: Coordinator UX

- Add dashboard mission view.
- Show worker sessions, findings, evidence, decisions, and conflicts. The first
  slice is read-only and surfaces brief, checkout diagnostics, ledger, and
  approval packet data without adding a second mutation path.
- Add accept/reject/defer controls for local review. These controls are
  implemented as explicit opt-in dashboard mutations and reuse approval
  decision application so review, optional promotion, and event hashes stay
  audit-compatible with CLI and MCP behavior.
- Add mobile-friendly review export or static brief. The first export is static
  Markdown generated from the approval packet, with JSON available for machines.

### Phase 5: Orchestrator Adapters

- Provide thin adapters for Codex-style local agents, LangGraph, CrewAI, and
  generic MCP clients. The first production slice is a shared
  `CoordinationAdapter`, LangGraph/CrewAI Coordinate helper factories, generic
  MCP tool-call template output, and `zaxy coordinate adapter-template`.
- Keep spawning and worktree/container management outside the core package until
  the coordination state contract is stable.
- Add optional git metadata capture for branches, worktrees, changed files, diff
  summaries, and test results. The first slice is explicit CLI capture on
  `coordinate report`; it uses read-only `git --no-optional-locks` commands and
  stores the result as finding evidence rather than managing branches or
  worktrees.

## Pruning For Product Focus

To make Zaxy tighter:

- Lead with coordination memory, not generic persistent memory.
- Keep embedded graph as the default runtime story. Move Neo4j, pgGraph, and
  LatticeDB into advanced backend documentation.
- Keep LongMemEval as table stakes, but make CoordinationBench the flagship
  proof.
- Keep Pathlight optional and do not make it part of the core product pitch.
- Treat Skill Memory as a supporting lane inside checkout, not a separate
  headline product.
- Keep framework adapters dependency-light. MCP is the primary interface.
- Avoid marketing every backend experiment. Backend shootouts should support the
  product, not become the product.

## Additional High-Demand Features

These features fit the coordination wedge and should be considered after the MVP:

- **Memory Diff / Conflict View**: compare worker beliefs, parent state, and
  accepted decisions.
- **Graph-Backed Drift Detector**: extend the current explicit stale/superseded
  metadata path with source-state and temporal graph checks, then warn when
  stale worker-local facts appear in a parent checkout or final answer.
- **Evidence Ledger**: make every durable claim traceable to commands, files,
  transcript turns, tool results, or human decisions.
- **Remote Approval Loop**: export a review packet where a human can accept,
  reject, or defer findings. The first implementation exposes
  `zaxy coordinate approval-packet`, `zaxy coordinate apply-approval`,
  `coordination_approval_packet`, and `coordination_apply_approval`; promotion
  remains explicit through `promote: true` decisions.
- **Memory Ownership Contract**: export missions, findings, evidence, and
  decisions as portable Eventloom-backed artifacts.
- **Agent Performance Ledger**: track which agents produce useful, cited,
  accepted work.
- **Trust-Gated Autopromotion**: allow automatic promotion only for findings that
  meet evidence, test, citation, and non-conflict policies.

## Success Criteria

Zaxy Coordinate is ready to headline the product when:

- A user can create a mission and three worker sessions from CLI or MCP.
- Workers can report findings with citations and evidence.
- The coordinator can produce a cited brief that separates accepted, pending,
  rejected, deferred, and conflicted state.
- Accepted-state checkout excludes unpromoted worker findings by default.
- The final handoff is replayable from Eventloom only.
- CoordinationBench shows Zaxy Coordinate beating implemented flat transcript,
  markdown, and BM25 baselines on conflict recall, accepted state precision,
  citation coverage, and token efficiency.
- Vector, non-coordinate Zaxy, and external competitor comparisons appear only
  when their pinned adapters have run through the same harness; otherwise they
  remain disclosure-only and are not used for public superiority claims.

Related references: [announcements/zaxy-coordinate.md](announcements/zaxy-coordinate.md),
[benchmarks.md](benchmarks.md), [integrations.md](integrations.md), and
[site/index.html](../site/index.html).
