# Changelog

All notable Zaxy release changes are recorded here.

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
