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
- LangGraph and CrewAI have dependency-light native-preview adapters.
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
- At least one external user, project, or case study has validated the workflow.

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
  examples, and failure behavior.
- Define the shared native integration contract for non-MCP runtimes:
  - before model/task call: bootstrap or checkout;
  - after model/task call: capture assistant or task output;
  - after tool call: capture redacted observation;
  - after context use: record feedback.
- Document Claude Desktop, Claude Code, Cursor, Codex, and generic MCP setup
  paths with one recommended local route each.

### Gates

- MCP schema snapshot tests protect the public tool surface.
- LangGraph example runs as part of release validation.
- Status and doctor commands return actionable remediation for common local
  failures.
- Clean first-run time improves or remains below the v0.5 threshold.
- Coverage remains at or above 92%.

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
- Improve approval flows so pending, conflicted, stale, and evidence-poor
  findings have obvious next actions.
- Improve conflict review:
  - deterministic source-state conflicts remain default;
  - lexical semantic conflicts remain opt-in;
  - hosted semantic conflict adapters remain bounded and auditable.
- Improve the dashboard or CLI mission viewer so users can inspect mission
  state, worker ledgers, findings, evidence, decisions, and promoted state
  without reading Eventloom JSONL.
- Add audit report generation for a mission from Eventloom replay.
- Publish a CoordinationBench report with clear baselines, adapter status,
  limitations, and reproduction commands.
- Add a polished multi-agent example that includes review and approval steps,
  not just worker reporting.

### Gates

- A user can complete a full mission workflow from CLI and MCP:
  start mission, create workers, assign work, report findings, detect conflicts,
  review findings, promote accepted state, checkout accepted memory, and create
  handoff.
- The same workflow is demonstrated through LangGraph or a direct native helper.
- CoordinationBench report generation is reproducible from tracked inputs.
- Audit report output cites Eventloom sequence and hash metadata.
- Coverage remains at or above 92%.

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
  contract defined in v0.6.
- Prioritize two paths:
  - OpenAI-compatible model-call wrapper for request/response capture,
    checkout injection, tool observation, and feedback;
  - Anthropic or Claude-style wrapper if the local usage path and API boundary
    can be kept stable.
- Keep direct integrations dependency-light and optional. Core install remains
  small.
- Add examples showing model-call memory activation without MCP.
- Add trace correlation from mission, checkout, context assembly, model call,
  tool call, finding, review, and handoff.
- Improve Pathlight integration or add neutral trace hooks that can feed
  LangSmith, Phoenix, or local JSONL traces without making any one provider a
  hard dependency.
- Improve replay tools for long-running missions, including branch/fork design
  if it can be implemented without destabilizing Eventloom semantics.
- Continue performance work on compaction, context assembly, projection rebuild,
  and query latency based on benchmark evidence.

### Gates

- At least one outside-MCP direct model integration runs in a documented example.
- Model-call capture is redacted, bounded, and opt-in where provider cost or
  privacy could surprise users.
- Trace output can follow a useful path from mission to model call to promoted
  finding.
- Benchmarks show no regression in checkout quality, citation coverage, and
  latency budgets.
- Coverage remains at or above 92%.

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
  - benchmark artifact schemas.
- Mark each surface as stable, beta, experimental, or internal.
- Add migration tests and docs for upgrades from 0.4 through 0.9.
- Add fuzz tests for Eventloom payload validation, hash-chain replay, and
  bounded MCP inputs.
- Add failure-injection or chaos tests for projection rebuild, corrupted
  projection artifacts, missing hooks, stale checkout, and degraded backends.
- Expand release gates so the public examples, MCP smoke, LangGraph smoke,
  Coordinate mission smoke, benchmark comparison, docs validation, and beta UAT
  all run or fail with explicit skip reasons.
- Add contributor guide, issue templates, and benchmark contribution guidance.
- Freeze v1.0 candidate schemas and begin treating changes as migration events.

### Gates

- Full release gate passes on Python 3.11, 3.12, and 3.13.
- Coverage remains at or above 92%.
- No public benchmark claim depends on untracked or local-only inputs.
- API inventory has no undocumented stable surfaces.
- Migration guide covers 0.4 to 0.9.
- At least one external user has run the first-run path or a Coordinate example
  and provided feedback.

### Explicit Non-Goals

- Do not add major new feature families after v0.9 unless they are required to
  fix a release blocker.
- Do not change event schemas casually after the freeze candidate.

## v1.0: Stable Coordinator Memory Release

**Theme:** Stable, documented, benchmarked, and positioned release.

**Target:** 2 to 3 weeks after v0.9.

### Ship

- Final API and data model stability commitment.
- Comprehensive changelog from 0.4 to 1.0.
- Migration guide from 0.4.
- Public v1.0 announcement with positioning, examples, benchmark evidence,
  limitations, and roadmap beyond 1.0.
- Updated website and docs.
- Final release smoke command and release validation checklist.
- External validation note, case study, or public user feedback if available.

### Gates

- Clean-repo UAT passes.
- MCP smoke passes.
- LangGraph smoke passes.
- At least one direct model integration smoke passes.
- Coordinate mission smoke passes.
- Benchmark guardrails pass.
- Docs validation passes.
- Release smoke passes.
- Coverage remains at or above 92%.
- Public surfaces are tagged with their stability level.

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

External validation is a release requirement, not a marketing extra. Start with
one or two users who can run the first-run path and one example. Capture where
they get stuck, and turn that into docs, doctor checks, or CLI improvements.

### Competitive Awareness

Track Mem0, Zep, Letta, MemPalace, Agent Memory, ActiveGraph, and the MCP
ecosystem. Avoid fake apples-to-apples claims. Keep competitor comparisons tied
to reproducible adapter contracts, public disclosures, or clearly labeled
limitations.

## Primary Risks

| Risk | Mitigation |
|------|------------|
| Scope creep | Treat auditable coordination, MCP, LangGraph, native model-call activation, and DX as the only v1.0 pillars. |
| Low external feedback | Make v0.5 first-run validation and v0.7 Coordinate example feedback explicit release gates. |
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
