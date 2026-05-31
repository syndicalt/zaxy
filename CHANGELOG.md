# Changelog

All notable Zaxy release changes are recorded here.

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
