# Zaxy v0.5 to v1.0 Release-Gate Roadmap

Zaxy is moving from a technically strong memory substrate into a stable product:
**coordinator memory for multi-agent projects**.

The v1.0 goal is not to add every possible memory feature. It is to make the
existing thesis reliable, legible, and useful enough that external users can
adopt Zaxy with confidence:

> Zaxy is the leading open-source system for auditable, replayable, and
> coordinated memory in multi-agent projects.

## Current Baseline

This roadmap starts from the current 0.4.x codebase, not from an empty plan.
Several capabilities that normally belong in later phases are already present:

- README positioning already leads with "Coordinator memory for multi-agent
  projects".
- `zaxy init` defaults to the local embedded Codex path with no Neo4j sidecar.
- Embedded Kuzu is the default projection backend; Neo4j, pgGraph, and LatticeDB
  remain optional or experimental paths.
- Memory Bootstrap and Memory Checkout are the model-facing context contracts.
- MCP exposes memory, context, lifecycle, and coordination tools.
- Zaxy Coordinate includes mission, worker, assignment, finding, review,
  promotion, approval packet, handoff, conflict, and benchmark flows.
- LangGraph has a dependency-light native-beta adapter; CrewAI remains a
  dependency-light native-preview adapter.
- CoordinationBench, LongMemEval-compatible reports, backend shootout tooling,
  release smoke checks, beta UAT, and docs validation already exist.
- Test coverage is already near the v1.0 bar, so 92% is a ratchet target rather
  than a future aspiration.

The work from v0.5 to v1.0 should therefore be release-gate driven: each release
must make a narrower public promise, prove it with docs, examples, tests, and
benchmarks, and avoid broad speculative expansion.

## v1.0 Success Criteria

Zaxy v1.0 is ready when these statements are true:

- Positioning is clear: **Coordinator Memory for Agent Teams**.
- A new user can install, initialize, inspect, and run a meaningful example in
  less than five minutes on a normal local development machine.
- MCP is polished enough that clients can discover tools, understand errors, and
  use structured outputs without reading source code.
- LangGraph is first-class and documented as the primary native framework path.
- At least one direct model-call integration path works outside MCP.
- Coordinate supports real mission workflows: isolated workers, cited findings,
  conflict review, approval, promotion, checkout, audit, and handoff.
- Public benchmarks are transparent about methodology, baselines, limitations,
  latency, citation coverage, and token tradeoffs.
- Public APIs, MCP schemas, CLI commands, and event payloads have a documented
  v1.0 stability contract.
- Coverage stays at or above 92%, with release gates preventing regressions in
  core memory, coordination, and onboarding flows.
- External validation collection remains available as optional post-release
  evidence when an outside user, project, or case study is available.

## Release Principles

- Minor releases before v1.0 may introduce breaking changes, but every breaking
  change must be documented in the changelog and migration guide.
- v1.0 freezes the public API, MCP schemas, stable CLI surfaces, and durable
  event payload contracts.
- Every release includes a changelog entry, updated docs, release smoke output,
  and at least one polished example or benchmark artifact.
- Eventloom remains the source of truth. Projection backends remain derived
  views.
- MCP remains the primary framework-neutral interface, but native integrations
  must share the same Memory Bootstrap, Memory Checkout, capture, and feedback
  contracts.
- Backend experiments do not become defaults unless they preserve temporal
  semantics, citations, inferred-edge metadata, session isolation, dashboard
  rendering, rebuild behavior, and published benchmark guardrails.

## v0.5: Public Positioning and First-Run Trust

**Theme:** Make the product legible and trustworthy from the first command.

**Target:** 4 weeks.

**Implementation status:** completed against
`docs/superpowers/plans/2026-05-30-v0-5-public-positioning-first-run-trust.md`;
the public positioning, first-run docs, quickstarts, examples, changelog, and
static site coverage are in place.

### Ship

- Align package metadata, README, docs homepage, site copy, and examples around
  "Coordinator Memory for Agent Teams".
- Keep architectural language available, but stop making "temporal knowledge
  graph fabric" the first public explanation.
- Turn the current quick start into a measured first-run path:
  install, `zaxy init`, memory bootstrap, memory checkout, doctor, and one
  example.
- Polish three examples:
  - single-agent durable memory;
  - LangGraph memory checkout before model work;
  - three-worker Coordinate mission with conflict review and handoff.
- Improve MCP tool descriptions for model-facing clarity and client discovery.
- Publish public docs for "Why Zaxy", "Getting Started", "MCP Quickstart",
  "Coordinate Quickstart", and "Architecture".
- Create a lightweight external validation script for one person outside the
  project to run and report time-to-first-success.

### Gates

- Clean install to successful `zaxy doctor` completes in less than five minutes
  on a normal local development machine.
- `zaxy init` leaves a user with clear next steps and no required sidecar.
- All polished examples run from a clean checkout.
- Documentation links validate.
- Coverage remains at or above 92%.
- Release artifact and release smoke checks pass.

### Explicit Non-Goals

- Do not expand backend scope.
- Do not make hosted or multi-tenant memory a v0.5 deliverable.
- Do not add another framework adapter before the first-run story is clean.

## v0.6: MCP and Native Runtime DX

**Theme:** Become the best MCP-native memory backend while preparing native
runtime integrations.

**Target:** 4 to 5 weeks after v0.5.

**Implementation status:** completed with the canonical MCP tool contract
snapshot at `docs/examples/mcp-tool-contract.json`, a matching schema test in
`tests/test_mcp.py`, representative response snapshots at
`docs/examples/mcp-response-snapshots.json` for bootstrap, checkout, graph
retrieval, verbatim retrieval, context assembly, feedback, and coordination
checkout, and structured MCP error payloads with stable error codes and
remediation hints. `zaxy hook-status` now reports
structured memory activation remediations with runnable checkout commands,
`zaxy doctor` surfaces the same missing or stale checkout state as a
`memory_activation` check, and top-level `zaxy status` now reports memory
activation freshness, latest checkout/capture details, token efficiency, and
checkout remediation commands. LangGraph checkout middleware now emits the
`zaxy.native.v0.6` metadata contract with diagnostics, quality, feedback
guidance, and fail-closed error payloads, and `zaxy doctor --release-smoke` now
runs `examples/langgraph_memory.py` as the LangGraph release-validation smoke.
The shared non-MCP adapter lifecycle and payload keys are published at
`docs/examples/native-integration-contract.json`. CrewAI checkout middleware
shares that same non-MCP native contract while remaining native-preview. The MCP
Quickstart now documents one recommended local setup route each for Codex,
Claude Code, Claude Desktop, Cursor, and generic MCP clients. Beta readiness now
checks `docs/examples/first-run-timing-report.json` so clean first-run doctor
and example timings remain under the five-minute budget. The configured
coverage ratchet and pytest coverage gate now enforce the 92% floor.

### Ship

- Harden MCP structured outputs for memory, checkout, context assembly,
  feedback, and coordination tools.
- Standardize MCP error payloads with stable error codes, human messages, and
  remediation hints.
- Add MCP contract tests that snapshot tool names, schemas, required fields, and
  representative responses.
- Improve `memory_checkout` output for client use: stable summary fields,
  diagnostics, citation coverage, stale-state warnings, required actions, and
  feedback templates.
- Improve `zaxy status`, `zaxy doctor`, and `zaxy hook-status` around last
  checkout, last capture, stale memory, missing hooks, and degraded projection
  states.
- Promote LangGraph from native-preview toward beta by stabilizing payload keys,
  examples, and failure behavior. **Status:** LangGraph now reports
  `native-beta` and emits the `zaxy.native.v0.6` checkout contract.
- Define the shared native integration contract for non-MCP runtimes:
  - before model/task call: bootstrap or checkout;
  - after model/task call: capture assistant or task output;
  - after tool call: capture redacted observation;
  - after context use: record feedback.
  **Status:** published in `docs/examples/native-integration-contract.json`.
- Document Claude Desktop, Claude Code, Cursor, Codex, and generic MCP setup
  paths with one recommended local route each. **Status:** covered in
  `docs/mcp-quickstart.md` and rendered to the static docs.

### Gates

- MCP schema snapshot tests protect the public tool surface.
- LangGraph example runs as part of release validation. **Status:** covered by
  the `langgraph_example` check in `zaxy doctor --release-smoke`.
- Status and doctor commands return actionable remediation for common local
  failures.
- Clean first-run time improves or remains below the v0.5 threshold.
  **Status:** covered by the `first_run_timing` check in
  `zaxy doctor --beta-readiness`.
- Coverage remains at or above 92%.
  **Status:** enforced by `pyproject.toml` and `scripts/check-coverage.py`.

### Explicit Non-Goals

- Do not promise full API stability yet.
- Do not promote CrewAI or AutoGen beyond current maturity until LangGraph and
  the shared contract are stable.

## v0.7: Coordination Workflows

**Theme:** Make Coordinate production-useful, not only demonstrable.

**Target:** 5 to 6 weeks after v0.6.

### Ship

- Add mission templates for common workflows such as software delivery,
  research review, benchmark investigation, and release validation.
  **Status:** built-in templates are available through
  `zaxy coordinate template list`, `show`, and `apply`; applying a template
  creates replayable mission, worker, and assignment events.
- Improve approval flows so pending, conflicted, stale, and evidence-poor
  findings have obvious next actions. **Status:** approval packets and static
  review exports now include explicit next-action guidance for normal review,
  conflict resolution, stale evidence refresh, and missing evidence.
- Improve conflict review:
  - deterministic source-state conflicts remain default;
  - lexical semantic conflicts remain opt-in;
  - hosted semantic conflict adapters remain bounded and auditable.
  **Status:** deterministic source-state conflicts are the default, local
  lexical semantic conflicts are opt-in through `--semantic-conflicts lexical`,
  and hosted HTTP semantic conflict adapters remain opt-in with bounded request
  payloads and locally validated response IDs.
- Improve the dashboard or CLI mission viewer so users can inspect mission
  state, worker ledgers, findings, evidence, decisions, and promoted state
  without reading Eventloom JSONL. **Status:** `zaxy coordinate inspect`
  provides a replay-only CLI viewer with mission brief, worker ledgers,
  findings by status, evidence, review decisions, promoted state, conflicts,
  approval packet, and handoff records.
- Add audit report generation for a mission from Eventloom replay.
  **Status:** `zaxy coordinate audit-report` generates a read-only Markdown or
  JSON report from mission and worker Eventloom replay with sequence/hash
  citations for every audited event.
- Publish a CoordinationBench report with clear baselines, adapter status,
  limitations, and reproduction commands. **Status:** the
  `coordination-real-v1` report is archived at
  `reports/benchmarks/coordination-real-v1/coordination-benchmark.md` with
  Zaxy metrics, local baselines, disclosure-only adapter status, limitations,
  and reproduction commands.
- Add a polished multi-agent example that includes review and approval steps,
  not just worker reporting. **Status:**
  `examples/coordinate_three_worker_project.py` now demonstrates a
  three-worker mission with approval packet export, approval application,
  accepted promotion, conflict/defer decisions, mission inspection, audit
  reporting, checkout, and handoff.

### Gates

- A user can complete a full mission workflow from CLI and MCP:
  start mission, create workers, assign work, report findings, detect conflicts,
  review findings, promote accepted state, checkout accepted memory, and create
  handoff. **Status:** covered across Coordinate CLI tests and MCP coordination
  tool tests, including conflict materialization, approval application,
  checkout, and handoff.
- The same workflow is demonstrated through LangGraph or a direct native helper.
  **Status:** `CoordinationAdapter` now covers start, worker creation,
  assignment, finding report, conflict detection, approval application,
  checkout, and handoff as JSON-friendly native helper calls.
- CoordinationBench report generation is reproducible from tracked inputs.
  **Status:** `coordination-real-v1` regenerates from
  `benchmarks/coordination-real-v1/coordination-workload.json` through
  `zaxy coordinate benchmark --workload ...`.
- Audit report output cites Eventloom sequence and hash metadata.
  **Status:** covered by `CoordinationManager.audit_report` and
  `zaxy coordinate audit-report --json` tests.
- Coverage remains at or above 92%. **Status:** enforced by the configured
  pytest coverage gate and coverage ratchet.

### Explicit Non-Goals

- Zaxy does not spawn or manage workers itself unless a separate orchestrator
  design is approved.
- Zaxy does not infer accepted findings from raw transcripts without explicit
  evidence and review.

## v0.8: Model-Native Integrations and Observability

**Theme:** Make memory activation work where model calls actually happen, not
only through MCP.

**Target:** 5 to 6 weeks after v0.7.

### Ship

- Add direct model-call integration modules outside MCP, sharing the same native
  contract defined in v0.6. **Status:** dependency-light OpenAI-compatible and
  Claude-compatible adapters now run Memory Checkout before provider-shaped
  model calls, inject checkout context, record bounded model-call metadata,
  capture sanitized assistant turns, and return the shared `zaxy.native.v0.6`
  metadata.
- Prioritize two paths:
  - OpenAI-compatible model-call wrapper for request/response capture,
    checkout injection, tool observation, and feedback; **Status:** available
    in `zaxy.adapters.openai_compatible` with redacted tool-call observation
    and direct memory feedback helpers.
  - Anthropic or Claude-style wrapper if the local usage path and API boundary
    can be kept stable. **Status:** available in
    `zaxy.adapters.claude_compatible` as a duck-typed
    `client.messages.create(**request)` helper with the same capture,
    observation, feedback, and fail-closed checkout contract.
- Keep direct integrations dependency-light and optional. Core install remains
  small. **Status:** both direct model-call adapters are duck-typed and import
  no provider SDK.
- Add examples showing model-call memory activation without MCP. **Status:**
  `examples/openai_compatible_memory.py` and
  `examples/claude_compatible_memory.py` run with fake provider-shaped clients
  and no network or provider dependency.
- Add trace correlation from mission, checkout, context assembly, model call,
  tool call, finding, review, and handoff. **Status:** `zaxy.trace` now builds
  provider-neutral `zaxy.trace.v0.8` spans and edges from replayed Eventloom
  events with sequence/hash citations, and `zaxy trace export --json` exposes
  correlated mission, checkout, model-call, tool-call, finding, review,
  promotion, and handoff paths without requiring a tracing provider.
- Improve Pathlight integration or add neutral trace hooks that can feed
  LangSmith, Phoenix, or local JSONL traces without making any one provider a
  hard dependency. **Status:** the neutral trace export is JSON-friendly and
  provider-independent; `zaxy trace export --format jsonl --output trace.jsonl`
  writes ingestion records for local JSONL pipelines while Pathlight remains
  optional.
- Improve replay tools for long-running missions, including branch/fork design
  if it can be implemented without destabilizing Eventloom semantics. **Status:**
  `zaxy replay` now supports inclusive `--from-seq/--to-seq` windows so
  operators can inspect bounded ranges in long Eventloom logs without rewriting
  history.
- Continue performance work on compaction, context assembly, projection rebuild,
  and query latency based on benchmark evidence. **Status:** beta readiness now
  includes a named `benchmark_no_regression` check over the release benchmark
  commands so performance work stays tied to reproducible smoke, 40-query
  performance, and 100-query scale reports.

### Gates

- At least one outside-MCP direct model integration runs in a documented
  example. **Status:** `zaxy doctor --release-smoke` now runs both
  `examples/openai_compatible_memory.py` and
  `examples/claude_compatible_memory.py`.
- Model-call capture is redacted, bounded, and opt-in where provider cost or
  privacy could surprise users. **Status:** direct model adapters capture
  bounded request metadata and sanitized assistant turns only when applications
  explicitly call the adapter helpers.
- Trace output can follow a useful path from mission to model call to promoted
  finding. **Status:** covered by trace correlation and CLI tests that link
  mission spans to model calls, tool observations, reviewed findings, promoted
  findings, and handoffs.
- Benchmarks show no regression in checkout quality, citation coverage, and
  latency budgets. **Status:** `zaxy doctor --beta-readiness` reports
  `benchmark_no_regression` and requires checkout quality floors, citation
  coverage at `1.0`, and p95/p99 checkout latency budgets in
  `scripts/release-check.sh`.
- Coverage remains at or above 92%. **Status:** enforced by the configured
  pytest coverage gate and coverage ratchet.

### Explicit Non-Goals

- Do not turn the packet analyzer into a required router.
- Do not make direct provider integrations the only recommended path; MCP
  remains the primary framework-neutral route.

## v0.9: Hardening and API Freeze Candidate

**Theme:** Prepare for a stable 1.0 contract.

**Target:** 4 to 5 weeks after v0.8.

### Ship

- Publish an API inventory covering:
  - MCP tool names, schemas, and response contracts;
  - Python SDK public classes and functions;
  - stable CLI commands and options;
  - durable Eventloom event types and payload fields;
  - projection backend contract;
  - benchmark artifact schemas. **Status:** `docs/api-inventory.md` now
    inventories these surfaces and links each category to its current contract
    authority.
- Mark each surface as stable, beta, experimental, or internal. **Status:**
  the API inventory uses explicit `Stable`, `Beta`, `Experimental`, and
  `Internal` labels and keeps candidate backends and Skill Memory out of the
  stable release surface.
- Add migration tests and docs for upgrades from 0.4 through 0.9. **Status:**
  `docs/migration.md` now covers each release band from 0.4 to 0.9, the
  compatibility test expectations, and the non-destructive rollback policy.
- Add fuzz tests for Eventloom payload validation, hash-chain replay, and
  bounded MCP inputs. **Status:** parametrized validation tests now reject
  non-object and oversized Eventloom payloads, sequence-tampered hash chains,
  and unbounded direct `memory_append` MCP inputs.
- Add failure-injection or chaos tests for projection rebuild, corrupted
  projection artifacts, missing hooks, stale checkout, and degraded backends.
  **Status:** projection rebuild failure recovery is covered by
  `test_reproject_command_closes_pggraph_backend_after_projection_failure`;
  corrupted projection artifacts are handled through integrity-first replay,
  reset/rebuild, and rollback paths; missing hooks and stale checkout are
  covered by `hook-status` activation and capture tests; degraded backends are
  surfaced through fallback metrics including `zaxy_degraded_operations_total`
  and documented alerting guidance.
- Expand release gates so the public examples, MCP smoke, LangGraph smoke,
  Coordinate mission smoke, benchmark comparison, docs validation, and beta UAT
  all run or fail with explicit skip reasons. **Status:** `scripts/release-check.sh`
  now has named commands for public examples, MCP smoke, LangGraph smoke,
  Coordinate mission smoke, docs validation, benchmark comparison, and beta UAT;
  `zaxy doctor --beta-readiness` reports `release_gate_surface_coverage` and
  accepts intentional omissions only as `SKIP:<reason>`.
- Add contributor guide, issue templates, and benchmark contribution guidance.
  **Status:** `CONTRIBUTING.md`, GitHub issue templates, and
  `docs/benchmark-contributions.md` now document the production-code bar,
  release checks, issue evidence, and reproducible benchmark contribution
  requirements.
- Freeze v1.0 candidate schemas and begin treating changes as migration events.
  **Status:** `docs/examples/v1-schema-freeze.json` now binds the v1 freeze
  candidate surfaces, and `docs/agent-events.md` defines
  `schema.migration.proposed` and `schema.migration.applied` for non-additive
  stable or beta contract changes.

### Gates

- Full release gate passes on Python 3.11, 3.12, and 3.13.
- Coverage remains at or above 92%.
- No public benchmark claim depends on untracked or local-only inputs.
- API inventory has no undocumented stable surfaces.
- Migration guide covers 0.4 to 0.9.
  **Status:** current evidence for these v0.9 gates is recorded in
  `docs/v09-gate-audit.md`.
- External-user feedback is tracked as optional post-release evidence. **Status:**
  collection packet and evidence shape are documented in
  `docs/v09-gate-audit.md`.

### Explicit Non-Goals

- Do not add major new feature families after v0.9 unless they are required to
  fix a release blocker.
- Do not change event schemas casually after the freeze candidate.

## v1.0: Stable Coordinator Memory Release

**Theme:** Stable, documented, benchmarked, and positioned release.

**Target:** 2 to 3 weeks after v0.9.

### Ship

- Final API and data model stability commitment. **Status:**
  `docs/stability-commitment.md` defines the v1.0 public API and data model
  compatibility promise, migration-event requirement, and non-commitments for
  experimental/internal surfaces.
- Comprehensive changelog from 0.4 to 1.0. **Status:** `CHANGELOG.md` now has
  explicit `0.9.0 - Release Candidate` and `1.0.0 - 2026-05-31`
  sections above the existing 0.4+ release history.
- Migration guide from 0.4. **Status:** `docs/migration.md` covers upgrades
  from 0.4 through the v0.9 freeze candidate and links migration events for
  post-freeze stable or beta schema changes.
- Public v1.0 announcement with positioning, examples, benchmark evidence,
  limitations, and roadmap beyond 1.0. **Status:** the stable 1.0.0
  announcement is in `docs/announcements/zaxy-v1.0.md`; it keeps external
  validation as optional post-release evidence and links the public
  verification request.
- Updated website and docs. **Status:** generated static site pages include the
  v1.0 announcement, stability commitment, release validation checklist, gate
  audit, API inventory, and migration guide.
- Final release smoke command and release validation checklist. **Status:**
  `docs/release-validation-checklist.md` records the v1.0 gates, aggregate
  commands, artifact review steps, and optional external validation path.
- Final release gate evidence audit. **Status:** `docs/v10-gate-audit.md`
  maps clean-repo UAT, MCP smoke, LangGraph smoke, direct model integration
  smoke, Coordinate mission smoke, benchmark guardrails, docs validation,
  release smoke, coverage, public stability tags, and optional external
  validation to proof or evidence guidance.
- External validation note, case study, or public user feedback if available.
  **Status:** collection packet and report template are published in
  `docs/external-validation.md`; outside-user evidence is optional for v1.0.

### Gates

- Clean-repo UAT passes. **Status:** command-level evidence is recorded in
  `docs/v10-gate-audit.md`; final release execution is tracked by
  `docs/release-validation-checklist.md`.
- MCP smoke passes. **Status:** command-level evidence is recorded in
  `docs/v10-gate-audit.md`.
- LangGraph smoke passes. **Status:** command-level evidence is recorded in
  `docs/v10-gate-audit.md`.
- At least one direct model integration smoke passes. **Status:** command-level
  evidence is recorded in `docs/v10-gate-audit.md`.
- Coordinate mission smoke passes. **Status:** command-level evidence is
  recorded in `docs/v10-gate-audit.md`.
- Benchmark guardrails pass. **Status:** command-level evidence is recorded in
  `docs/v10-gate-audit.md`.
- Docs validation passes. **Status:** command-level evidence is recorded in
  `docs/v10-gate-audit.md`.
- Release smoke passes. **Status:** command-level evidence is recorded in
  `docs/v10-gate-audit.md`.
- Coverage remains at or above 92%. **Status:** command-level evidence is
  recorded in `docs/v10-gate-audit.md`.
- Public surfaces are tagged with their stability level. **Status:** command-level
  evidence is recorded in `docs/v10-gate-audit.md`.
- External validation is optional post-release evidence. **Status:** the
  evidence shape is documented in `docs/v10-gate-audit.md`.

## Cross-Cutting Workstreams

### Benchmarks

Run and publish benchmark evidence regularly. Zaxy benchmark claims should show
the workload, input fingerprint, baseline, metrics, latency, citation coverage,
token tradeoffs, and limitations.

Priority benchmark lanes:

- LongMemEval-compatible memory retrieval;
- CoordinationBench;
- backend shootout for embedded, Neo4j, pgGraph, LatticeDB, and BM25;
- first-run onboarding time;
- activation efficiency for high-context sessions;
- audit and replay completeness.

### Documentation and Content

Every release needs docs and public communication. Content should focus on
auditable coordination, replayable memory, MCP-native adoption, LangGraph, and
real examples.

Recommended content sequence:

- v0.5: "Why coordinator memory is different from vector memory";
- v0.6: "Using Zaxy as an MCP memory backend";
- v0.7: "Coordinating agent teams with accepted state and audit trails";
- v0.8: "Native model-call memory outside MCP";
- v1.0: launch announcement with benchmark and case-study evidence.

### Testing and CI

The v1.0 quality bar is coverage at or above 92%, with stable tests for public
contracts. The release process should continue to include ruff, mypy, pytest,
coverage ratchet, docs validation, release smoke, and benchmark guardrails.

### Community and External Validation

External validation is optional post-release evidence, not a v1.0 release
blocker. Start with one or two users who can run the first-run path and one
example when available. Capture where they get stuck, and turn that into docs,
doctor checks, or CLI improvements.

### Competitive Awareness

Track Mem0, Zep, Letta, MemPalace, Agent Memory, ActiveGraph, and the MCP
ecosystem. Avoid fake apples-to-apples claims. Keep competitor comparisons tied
to reproducible adapter contracts, public disclosures, or clearly labeled
limitations.

## v1.1: Accepted-State Recovery and Benchmark Guardrails

**Theme:** Make Zaxy's accepted-state memory thesis falsifiable under stale,
distracting, incomplete, and no-safe-answer histories.

**Implementation status:** in progress on the 1.1.0 release line.

### Ship

- Promote StateRecoveryBench as an official benchmark lane with a canonical
  tracked workload, report schema, Markdown report, guardrail checker, and
  release-check wiring.
- Add `zaxy state-recovery-benchmark` as the public command. Keep
  `zaxy experimental state-recovery` as a research/debug path.
- Use `memory_fabric_checkout` as the production guardrail baseline. Treat
  associative projection rows as diagnostic research baselines only.
- Resolve Coordinate accepted state from parent mission replay so checkout and
  proof packets agree on accepted findings, diagnostic rows, review refs,
  promotion refs, and worker source refs.
- Update public docs, API inventory, migration notes, testing guidance,
  changelog, and benchmark docs without broadening product claims beyond the
  measured lane.

### Gates

- `python scripts/check-state-recovery-benchmark.py
  reports/benchmarks/state-recovery-v1/state-recovery-benchmark.json --workload
  reports/benchmarks/state-recovery-v1/state-recovery-workload.json
  --require-git-tracked-inputs`
- `zaxy state-recovery-benchmark --output-dir /tmp/zaxy-state-recovery
  --workload reports/benchmarks/state-recovery-v1/state-recovery-workload.json`
- Focused tests for StateRecoveryBench, Coordinate accepted-state resolution,
  Memory Checkout coordinate suppression, CLI behavior, ruff, mypy, and release
  smoke.

### Explicit Non-Goals

- Do not promote astro-associative projection to core product behavior in 1.1.
- Do not claim StateRecoveryBench replaces LongMemEval or CoordinationBench.
- Do not tune behavior to benchmark labels; only authority, review, promotion,
  provenance, stale, and citation metadata may determine accepted state.

## Primary Risks

| Risk | Mitigation |
|------|------------|
| Scope creep | Treat auditable coordination, MCP, LangGraph, native model-call activation, and DX as the only v1.0 pillars. |
| Low external feedback | Keep v0.5 first-run validation and v0.7 Coordinate example feedback as optional post-release evidence, while local release gates prove the documented paths. |
| Benchmark credibility | Require tracked inputs, fingerprints, baselines, methodology, and limitation notes. |
| Breaking changes | Inventory public surfaces in v0.9 and document every migration. |
| Backend distraction | Keep embedded Kuzu as default unless another backend beats the same gates without sidecar friction or quality loss. |
| Native integration fragmentation | Force MCP and direct integrations through the same Memory Bootstrap, Memory Checkout, capture, and feedback contracts. |

## Minimum Viable v1.0

If timeline or feedback pressure requires a narrower release, preserve only
these pillars:

1. Polished positioning and docs.
2. Excellent MCP experience.
3. First-class LangGraph path.
4. Production-useful Coordinate workflow.
5. One outside-MCP model integration.
6. Transparent benchmark evidence.
7. Stable public API and data model contracts.

Everything else can move past v1.0.

Related references: [README.md](../../README.md), [site/index.html](../../site/index.html),
[Coordinate roadmap](../coordinate-roadmap.md), and [benchmarks](../benchmarks.md).
