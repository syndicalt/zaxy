# Changelog

All notable Zaxy release changes are recorded here.

## 2.0.1 - 2026-06-10

- Decomposed the largest internal modules into dependency-layered packages while
  preserving the original `zaxy.retrieval_plan`, `zaxy.synthesis`, and CLI
  import surfaces for existing callers.
- Replaced the embedded projection backend's pure-Python vector scoring loop
  with numpy-backed unit-vector matrix ranking plus bounded LRU cache budgets.
- Added session-scoped paged-query caching with Eventloom log freshness
  detection so continuation pages avoid redundant retrieval without serving
  stale direct-writer results.
- Expanded CI lint and strict type checking to cover source-checkout benchmark
  modules in `zaxy_benchmarks`.
- Updated stale v0.9-era documentation wording and regenerated rendered site
  pages.

## 2.0.0 - 2026-06-09

- Added the Zaxy 2.0 cognitive-substrate release-candidate surface with causal
  memory contracts, auditable causal edge projection, causal checkout
  diagnostics, causal CLI/MCP read APIs, and benchmark helpers.
- Added review-gated consolidation contracts and pipeline support so raw
  Eventloom traces can produce cited, reviewable higher-level candidates
  without replacing the immutable source of truth.
- Added reasoning-loop, metacognitive, and procedural-planning primitives for
  first-class memory participation during planning, execution, review, and
  reflection.
- Added the 2.0 RC.1 benchmark freeze manifest, refreshed release guardrails,
  and tracked benchmark evidence for backend shootout, StateRecoveryBench,
  PurposeBench, CoordinationBench, and LongMemBench/LongMemEval development
  history.
- Hardened first-run onboarding and Codex activation so `zaxy init` produces a
  compact setup/readiness summary, path-stable activation commands, safer Codex
  MCP config handling, structured JSON action items, and clearer capture
  guidance.
- Added first-class LongMemBench adapter support plus archived externally
  anchored run artifacts, while keeping generated benchmark projection
  databases out of git.
- Hardened high-value codebase review findings by tail-reading Eventloom hot
  paths, protecting dashboard state-changing endpoints, preserving projection
  caches on no-op writes, moving provider calls off blocking async paths, and
  keeping benchmark/eval implementation code out of the production wheel.

## 1.1.2 - 2026-06-05

- Updated the Eventloom adapter for `@eventloom/runtime@1.0.0` v1 JSONL
  envelopes with `id`, `actorId`, `threadId`, `parentEventId`, `causedBy`, and
  nested `integrity.hash` / `integrity.previousHash` fields.
- Preserved Zaxy's internal `Event` API and legacy top-level Zaxy log replay so
  existing graph, checkout, MCP, and recovery paths continue to work.
- Promoted native Eventloom v1 logs from skipped foreign JSONL to first-class
  read-only memory status/log inputs, while keeping malformed v1-looking logs
  diagnostic and non-fatal.
- Made optional Pathlight tracing degrade to no-op when the collector is
  unavailable so MCP startup and memory operations are not blocked by
  observability.
- Documented the v1 envelope boundary, legacy fallback behavior, and
  dot-delimited event-type requirement.

## 1.1.1 - 2026-06-05

- Hardened Codex activation persistence across session starts, `/resume`,
  compaction, MCP tool reloads, and capture watcher restarts.
- Added model-visible `AGENTS.md` Zaxy Memory Activation instructions during
  `zaxy init`, with a marker-managed block and `--no-agent-instructions`
  opt-out.
- Made managed Codex capture startup part of `zaxy activate codex`, with
  degraded activation packets when capture is missing or cannot start.
- Added `zaxy hook-event resume`, fresh-checkout reminders for resumed sessions,
  and `zaxy hook-status --require-capture-running` as a failing capture guardrail.
- Surfaced runtime-unverified MCP tool availability with CLI checkout fallbacks,
  and retried embedded Kuzu checkout lock failures with session-local projection
  fallback diagnostics.
- Extended `zaxy doctor` to hard-warn on configured-but-stopped Codex capture and
  missing model-visible activation instructions.

## 1.1.0 - 2026-06-05

- Promoted StateRecoveryBench as an official benchmark lane for partial-cue
  accepted-state recovery under stale, distracting, incomplete, and
  no-safe-answer event histories.
- Added the `zaxy state-recovery-benchmark` release command, a canonical
  tracked workload/report artifact, and a release guardrail checker for the
  production `memory_fabric_checkout` baseline.
- Added report schema metadata, workload fingerprints, case/baseline counts,
  production-baseline thresholds, Markdown guardrails, and release-check wiring
  for StateRecoveryBench.
- Added replay-derived Coordinate accepted-state resolution so Coordinate
  checkout and proof packets share the same parent-promoted state, diagnostic
  row classification, review refs, promotion refs, and worker source refs.
- Kept associative projection rows diagnostic and experimental; 1.1.0 product
  claims are gated on MemoryFabric checkout, Eventloom citations, and explicit
  Coordinate authority metadata.

## 1.0.4 - 2026-06-05

- Fixed authority metadata propagation for generic Eventloom rows so
  `authority_scope`, `status`, `stale`, `promoted`, and `superseded_by` survive
  through verbatim source recall and generic graph checkout lanes.
- Hardened Coordinate-purpose Memory Checkout suppression so worker-scoped
  unpromoted rows, unsupported or rejected rows, and superseded or deprecated
  stale rows stay out of current facts and cited evidence while remaining
  auditable in provenance.
- Preserved the existing Memory Checkout contract and release posture; this is
  a patch release, not the accepted-state StateRecoveryBench feature release.

## 1.0.3 - 2026-06-04

- Promoted the current74 full 500-question LongMemEval-compatible report as the
  public benchmark headline: mean score 0.940, Answer@5 0.906, citation
  coverage 1.000, R@1/R@5/R@10 0.906/1.000/1.000, p95 687.67 ms, and p99
  969.10 ms.
- Archived the current74 report, reproduction command, and benchmark-compare
  guardrail while preserving separate legacy `limit=10` and same-harness
  backend-evaluation floors.
- Added deterministic evidence-program tracing and broader answer-candidate
  synthesis coverage for preference, temporal, scalar, arithmetic, and
  source-cited answer assembly.
- Updated public benchmark, retrieval, testing, competitive-positioning, README,
  and generated static-site documentation to match the released benchmark
  posture and external disclosure rules.
- Added preference synthesis and rendered-packet coverage so the 92% coverage
  ratchet remains enforced across Python 3.11, 3.12, and 3.13.

## 1.0.2 - 2026-06-02

- Fixed Memory Capabilities and read-only memory status/log inspection for
  repositories that contain native Eventloom JSONL files next to Zaxy session
  logs. Zaxy now skips incompatible top-level JSONL logs with diagnostics
  instead of treating native Eventloom `events.jsonl` as a Zaxy event log and
  failing MCP startup with missing `seq`, `actor`, or `hash` fields.
- Added the `memory_synthesis_artifact` MCP tool and deterministic synthesis
  artifact payloads with auditable ledger rows so answer candidates preserve
  support, exclusion, and source-citation decisions from Memory Checkout.
- Added the `memory_synthesis_evidence` MCP tool so clients can reinforce or
  exclude individual synthesis ledger rows with cited fact ids, source groups,
  answer candidates, and reasons.
- Projected synthesis artifacts, answer candidates, ledger rows, candidate
  outcomes, and Coordinate proof packets into graph memory, and made candidate
  feedback canonicalize against checkout answer candidates before writing.
- Hardened synthesis bundles so elapsed-duration, social-media break, and
  road-trip duration fields carry ledger-row provenance, while currency-only
  synthesis no longer emits unrelated duration fallback totals.
- Added ledger-row provenance for age-at-event, career-prior-duration, and
  family-age-average synthesis fields.
- Added ledger-row provenance for relative week/month intervals, anniversary
  month subtraction, parent-order, recency, and temporal-order synthesis fields.
- Added optional late-interaction HTTP reranking with tokenized candidate
  payloads and `rerank_strategy` score diagnostics while keeping lexical local
  reranking as the deterministic default.
- Moved Memory Checkout answer candidates to the top of the full prompt
  contract so composed answers appear before raw facts and evidence.
- Added reusable synthesis operation objects for sum, difference, average, list,
  and temporal interval projection, and routed aggregate candidate assembly
  through the operation layer without changing answer-line compatibility.
- Replaced synthesis artifact verification placeholders with deterministic
  missing-evidence, dedupe-decision, warning, and skill-memory contradiction
  diagnostics from Memory Checkout.
- Added first-class purpose profiles for Memory Checkout and Coordinate so
  callers can condition memory by role, task, risk, evidence policy, retention
  policy, ontology lens, and expected action. Synthesis artifacts and feedback
  now preserve the checkout purpose profile for future outcome learning.
- Added purpose-conditioned retrieval scoring so non-general checkout purposes
  apply deterministic query emphasis, profile-specific recall floors, and a
  purpose-selected scoring profile without mutating the global router policy.
- Enforced purpose suppress rules at the Memory Checkout boundary so
  purpose-incompatible rows do not become current facts or cited evidence, with
  suppressed counts and reasons exposed in checkout diagnostics and retention
  metadata.
- Added purpose-aware Coordinate compaction projections. `zaxy compact
  --projection-output ... --purpose coordinate` now keeps accepted/promoted
  parent state, proof packets, and handoffs authoritative while preserving
  pending, rejected, deferred, stale, and unpromoted worker rows only as
  consolidation diagnostics.
- Added generalized purpose-aware compaction policies: security, release, and
  review preserve all source-backed records, while coding and research use
  bounded exemplar projections with purpose-specific record floors.
- Added purpose-aware retrieval decay floors so Coordinate, security, release,
  and review memories resist generic staleness decay without mutating Eventloom
  or graph facts. Score explanations now expose the applied purpose profile and
  retention half-life.
- Added purpose-scoped feedback for Memory Checkout and MCP so
  `memory_feedback`/`record_context_feedback` can preserve useful-for-what
  purpose profiles, outcomes, Coordinate authority metadata, and projected
  purpose audit fields on reinforced memory.
- Added the deterministic `purpose-v1` benchmark gate and `zaxy
  purpose-benchmark` command. The archived report covers Purpose Recall,
  Ontology Shift, Consequence Retention, Governed Forgetting, Action Outcome
  Loop, Cross-Role Citation, and Accepted-State Discipline while blocking
  Semantic Reach/Quarq comparative claims until same-harness adapters are
  pinned and scored.
- Added the first `PurposeOntologyLens` overlay contract and high-risk
  `EvidencePolicy` evaluator. Checkout diagnostics now expose purpose role
  matches, lens metadata, missing evidence requirements, failure reasons, and
  refresh queries for security, release, and Coordinate profiles without
  rewriting Eventloom or graph facts. The `purpose-v1` Ontology Shift lane now
  verifies purpose-specific graph path roles and edge multipliers.
- Added synthesis promotion gating for high-risk purpose evidence failures:
  synthesis artifacts now preserve promotion-gate/evidence-policy failures, and
  positive `used` candidate feedback is rejected until answerability, required
  evidence, and cited support-source checks pass. The `purpose-v1` benchmark now
  includes an Evidence Policy Discipline lane, and beta readiness executes
  security, release, and Coordinate policy fixtures.
- Added replay-derived purpose outcome learning for Memory Checkout. Repeated
  positive outcomes now apply bounded, explainable rank boosts for the matching
  purpose, repeated negative outcomes surface suppression candidates and warning
  pressure without deleting memory, and `memory.feedback` now projects auditable
  feedback metadata into the graph.
- Added broader project-local purpose profiles for support, product, sales,
  legal, and executive work. Each profile now has explicit retrieval, ontology,
  evidence, retention, suppression, compaction, checkout, and `purpose-v1`
  benchmark coverage while preserving the agent-work-memory positioning and
  avoiding full Company Brain claims.
- Added neutral document/transcript substrate projection. `document.indexed`
  and `transcript.turn` now emit `neutral_substrate` records with source
  backpointers, ingestion audits flag irreversible purpose labels, and
  `purpose-v1` proves one customer artifact can rebuild distinct support,
  product, legal, and executive purpose projections.
- Added the replay-only purpose control plane. `zaxy memory purpose status`,
  `zaxy memory purpose lanes`, `zaxy memory purpose feedback`, the local
  dashboard Purpose tab/API, and the static Eventloom viewer now expose active
  profiles, evidence-policy failures, suppressed rows, refresh suggestions,
  retained consequence history, and Coordinate accepted-state versus worker
  diagnostics without requiring Neo4j.
- Hardened CoordinationBench Quarq/Hybi same-harness posture. Packaged Quarq
  and Hybi manifests now include pinned public source/package refs, install
  commands, workload/result contracts, explicit unsupported runner commands, and
  archived stdout/stderr on runner failure while keeping public competitor
  claims blocked until completed local scoring exists.
- Added `purpose_feedback_coverage` to CoordinationBench so Zaxy Coordinate and
  same-harness adapters can prove accepted parent-state feedback is tied to the
  `coordinate` purpose profile instead of generic retrieval usefulness.
- Added a CoordinationBench competitor claim gate for Quarq and Semantic
  Reach/Hybi. Reports now expose a machine-readable blocked/passed verdict, the
  CLI can fail public claim runs with `--require-competitor-claim`, and the
  archived `coordination-real-v1` report includes disclosure-only Quarq/Hybi
  rows plus manifest templates.
- Added a Coordinate purpose/synthesis gate to CoordinationBench reports so
  Coordinate product claims require proof-backed accepted-state synthesis,
  Coordinate-purpose feedback, citation coverage, parent-checkout answerability,
  replayability, and no non-authoritative worker-row leakage.
- Added the `coordination_competitor_claims` beta-readiness check so release
  readiness fails if Quarq/Hybi public docs or archived CoordinationBench
  artifacts drift into unsafe same-harness claims without locally scored result
  audits.

## 1.0.1 - 2026-05-31

- Fixed embedded MCP worker startup for long-running multi-agent sessions by
  adding a workspace owner/proxy runtime so duplicate `zaxy serve` processes
  proxy to the single Kuzu graph owner instead of opening the embedded graph
  concurrently.
- Added `zaxy doctor` and `zaxy init` cleanup for stale embedded MCP owner
  metadata, including an `embedded_mcp_runtime` doctor check for actionable
  runtime repair.

## 1.0.0 - 2026-05-31

- Added the v1.0 stability commitment covering public API surfaces, Eventloom
  data model compatibility, migration events, and non-commitments.
- Added the v1.0 release announcement and release validation checklist
  artifacts, with external validation kept as an optional post-release evidence path.
- Added the v1.0 public release article, launch header image, and scripted
  Zaxy Coordinate/Collaborate demo media package for the public docs site.
- Added the v0.9 gate audit recording local release gates and the now-optional
  external-user feedback evidence path so v1.0 readiness does not overclaim.
- Added the v1.0 gate audit mapping every final release gate to command-level
  local evidence and optional external validation evidence.
- Added the external validation packet and GitHub issue template for collecting
  outside-user evidence when it becomes available without blocking the release.

## 0.9.0 - Release Candidate

- Added the v0.9 API inventory documenting MCP tools, Python exports, CLI
  commands, Eventloom events, projection backend contracts, and benchmark
  artifact schemas with stability labels.
- Added the v0.9 Migration guide for upgrades from 0.4 through 0.9, including
  compatibility test expectations and non-destructive rollback guidance.
- Added contributor guidance, GitHub issue templates, and benchmark
  contribution rules for tracked inputs, query diagnostics, citation coverage,
  and release guardrails.
- Hardened Eventloom and MCP validation with fuzz-style tests for non-object or
  oversized payloads, hash-chain sequence tampering, and bounded
  `memory_append` inputs.
- Expanded the release gate surface inventory with named public examples, MCP,
  LangGraph, Coordinate mission, docs, benchmark, and beta UAT commands plus
  explicit `SKIP:<reason>` handling.
- Added the v1 schema-freeze manifest and schema migration event taxonomy for
  stable or beta contract changes after the v0.9 freeze candidate.

## 0.8.0 - Unreleased

- Added a dependency-light OpenAI-compatible model-call adapter that injects
  Memory Checkout into `chat.completions.create` requests outside MCP, captures
  bounded request metadata, records sanitized assistant turns, and returns the
  shared `zaxy.native.v0.6` metadata contract.
- Added a dependency-light Claude-compatible model-call adapter that injects
  Memory Checkout through Claude-style `messages.create` system text, captures
  bounded request metadata, records sanitized assistant turns, and shares the
  same native checkout contract.
- Added OpenAI-compatible adapter helpers for redacted tool-call observations
  and direct memory feedback events, with matching Claude-compatible helpers.
- Added no-network OpenAI-compatible and Claude-compatible examples using fake
  provider clients to demonstrate model-call memory activation without MCP or
  provider SDK dependencies.
- Added the OpenAI-compatible and Claude-compatible examples to
  `zaxy doctor --release-smoke` so direct model-call activation is release-gated.
- Added provider-neutral `zaxy.trace.v0.8` trace correlation from replayed
  Eventloom events plus `zaxy trace export --json` and
  `--format jsonl --output ...` for local JSONL or external tracing-provider
  ingestion.
- Added inclusive `zaxy replay --from-seq/--to-seq` windows for bounded
  inspection of long-running Eventloom logs.
- Added an explicit beta-readiness benchmark no-regression gate for checkout
  quality, citation coverage, and p95/p99 latency budgets across smoke,
  performance, and scale backend reports.

## 0.7.0 - Unreleased

- Added built-in Coordinate mission templates for software delivery, research
  review, benchmark investigation, and release validation, with CLI support for
  `zaxy coordinate template list`, `show`, and `apply`.
- Added explicit approval next-action metadata for pending, conflicted, stale,
  and evidence-poor findings in Coordinate approval packets and review exports.
- Added `zaxy coordinate inspect` as a replay-only mission viewer combining
  brief state, worker ledgers, findings, evidence, decisions, promoted state,
  conflicts, approval packets, and handoffs.
- Added `zaxy coordinate audit-report` for read-only mission audit reports with
  Eventloom session, sequence, and hash citations across mission and worker
  replay.
- Expanded the three-worker Coordinate example to include approval packet
  export, approval decision application, accepted promotion, conflict/defer
  decisions, mission inspection, audit reporting, checkout, and handoff.
- Published the `coordination-real-v1` CoordinationBench report with local
  baselines, disclosure-only adapter status, limitations, and reproduction
  commands from a tracked workload.
- Added conflict materialization to the dependency-light `CoordinationAdapter`
  so direct native helpers cover the full v0.7 mission workflow.

## 0.6.0 - Unreleased

- Added a canonical MCP tool contract snapshot for tool names, descriptions,
  required fields, and full input schemas.
- Added representative MCP response snapshots for `memory_bootstrap`,
  `memory_checkout`, `memory_query`, and `memory_verbatim`.
- Standardized MCP tool-dispatch error payloads with stable `unknown_tool`,
  `invalid_request`, and `internal_error` codes plus remediation hints.
- Added structured memory activation remediations to `zaxy hook-status` and a
  matching `memory_activation` doctor check with runnable checkout commands.
- Added top-level `zaxy status` memory activation output so local runtime checks
  also show stale checkout, latest capture, token efficiency, and checkout
  remediation commands.
- Added the dependency-light LangGraph example to `zaxy doctor --release-smoke`
  so release validation runs the native-beta checkout path.
- Published `docs/examples/native-integration-contract.json` for the shared
  `zaxy.native.v0.6` non-MCP adapter lifecycle and payload keys.
- Added a beta-readiness first-run timing check backed by
  `docs/examples/first-run-timing-report.json` to keep the clean local path
  under the five-minute budget.
- Raised the configured coverage ratchet and pytest coverage gate to the v0.6
  roadmap floor of 92%.
- Stabilized LangGraph checkout metadata around the `zaxy.native.v0.6` native
  adapter contract, including diagnostics, quality, feedback guidance, and
  fail-closed checkout error payloads.
- Applied the same `zaxy.native.v0.6` checkout contract and fail-closed error
  behavior to the CrewAI native-preview task middleware.
- Expanded the MCP Quickstart with one recommended local route each for Codex,
  Claude Code, Claude Desktop, Cursor, and generic MCP clients.
- Extended representative MCP response snapshots to cover `context_assemble`,
  `memory_feedback`, and `coordination_checkout` alongside bootstrap,
  checkout, graph retrieval, and verbatim retrieval.

## 0.5.0 - Unreleased

- Repositioned Zaxy around **Coordinator Memory for Agent Teams** across package
  metadata, README, docs, and the static site.
- Added first-run validation docs so new users can report install, init,
  bootstrap, checkout, doctor, and example timing.
- Added MCP Quickstart and Coordinate Quickstart docs for the v0.5 public path.
- Added single-agent, LangGraph, and Coordinate example smoke coverage.
- Improved MCP tool descriptions so model-facing clients know when to call
  bootstrap, checkout, feedback, and coordination tools.

## 0.4.0 - 2026-05-28

- Added Zaxy Coordinate, a replay-backed parent/worker coordination layer for
  multi-agent projects with mission briefs, worker assignments, structured
  findings, approvals, promoted parent state, handoffs, stale/conflict
  diagnostics, and performance ledgers.
- Exposed Coordinate through CLI, MCP, dashboard review controls, framework
  adapter templates, and a dependency-light `CoordinationAdapter` with
  LangGraph and CrewAI helpers.
- Added the CoordinationBench standard with frozen schemas, runner manifest
  templates for Mem0, Agent Memory, and ActiveGraph comparisons, local
  baselines, competitor disclosure validation, and report generation.
- Published the Coordinate roadmap, announcement article, header image,
  generated site pages, and a three-worker project example.
- Hardened source-lane synthesis and context assembly so absence bundles are
  labeled correctly, graph summaries do not crowd out verbatim source evidence
  for the same provenance group, and redundant source expansions are avoided
  when salient source hits are already present.
- Updated development extras and coverage tests so optional Neo4j and LatticeDB
  integration paths are exercised in CI while remaining optional runtime
  installs.

## 0.3.1 - 2026-05-19

- Exposed `--projection-backend pggraph` and `--pggraph-dsn` on the read-only
  local dashboard so pgGraph projection evaluation is visible from the same
  runtime graph UI as Neo4j.
- Added a read-only pgGraph dashboard graph provider over the projection
  contract tables with Eventloom fallback and explicit backend validation.

## 0.3.0 - 2026-05-19

- Added Memory Persistence / Agent Recall Hardening so Zaxy reintroduces itself
  across session start, resume, compaction, long sessions, long tool runs, and
  roadmap/status questions.
- Added `memory.reminder.suggested`, memory bootstrap/checkout/feedback
  activity markers, graph extractors, hook coverage, and dashboard visibility
  for stale memory state.
- Added opinionated LangGraph, CrewAI, and AutoGen checkout paths so framework
  integrations call Memory Checkout at model/task/reply boundaries.
- Added backend-aware context refresh and source projection retirement for
  changed documents, transcripts, and codebase indexes.
- Expanded pgGraph experimental backend coverage for projection, retrieval,
  invalidation, traversal, integrity status, and release-safe operational
  diagnostics.

## 0.2.3 - 2026-05-18

- Added an explicit local pgGraph bootstrap path for `zaxy init --infra`.
- Fixed MCP startup so `PROJECTION_BACKEND=pggraph` bootstraps pgGraph instead
  of trying to start Neo4j.
- Documented the `PGGRAPH_REPO` installer requirement so Zaxy does not silently
  run plain PostgreSQL without graph traversal support.

## 0.2.2 - 2026-05-18

- Added pgGraph projection integrity and inferred-edge audit status support.
- Routed read-only memory graph status commands through the backend selector so
  pgGraph can use the same operator diagnostics as Neo4j.
- Expanded dashboard and pgGraph test coverage to preserve the release coverage
  ratchet.

## 0.2.1 - 2026-05-15

- Added first-class Hermes Agent MCP config rendering and explicit `config.yaml`
  merge support through `zaxy ide-config hermes`.
- Kept Hermes Agent onboarding workspace-neutral so global MCP config does not
  pin `EVENTLOOM_PATH`, `EVENTLOOM_THREAD`, or `ZAXY_DOMAIN` to one repository.
- Added PyYAML packaging support and documentation for Hermes Agent MCP install
  targets.

## 0.2.0 - 2026-05-15

- Promoted the beta release to a stable package so default `pip install zaxy-memory` resolves to the current Zaxy release without prerelease flags.
- Preserved the 0.2.0 beta release evidence and benchmark claims while making the same production-ready memory, capture, checkout, graph, and benchmark hardening available as the latest stable PyPI version.

## 0.2.0b1 - 2026-05-15

- Promoted Zaxy to its first beta packaging track with clean CI, release smoke, beta readiness, and trusted PyPI publishing gates.
- Hardened model-facing memory UX with Memory Bootstrap, Memory Checkout diagnostics, feedback guidance, source-aware context assembly, and shared checkout policy across core and MCP paths.
- Expanded deterministic capture and onboarding with local Codex capture, hook status coverage, leak detection, happy-path infrastructure profiles, and clean-repo UAT.
- Improved graph projection and auditability with hash-linked Eventloom event paths, source citation edges, temporal entity version edges, inferred-edge audit metadata, and graph projection integrity checks.
- Added and archived MemPalace-comparable benchmark evidence, including guardrails for mean score, Answer@5, Recall@5, citation coverage, and latency budgets.
- Hardened long-memory retrieval and synthesis to reach the current archived beta benchmark report: mean 0.950, Answer@5 0.950, citation coverage 1.000, and R@1/R@5/R@10 0.990.

## 0.1.0 - 2026-05-11

- Published the first public `zaxy-memory` package on PyPI.
- Added the `zaxy` console script for local onboarding, memory inspection, MCP serving, capture, projection, benchmarking, and release operations.
- Switched the publish workflow to PyPI Trusted Publishing so future releases use GitHub OIDC instead of long-lived PyPI API tokens.
- Shipped the current alpha memory substrate: Eventloom-backed provenance, Neo4j projection, Memory Checkout, deterministic capture, local onboarding, hooks, packet capture as an optional path, and benchmark tooling.
