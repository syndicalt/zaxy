# Migration Guide

This guide covers the public upgrade path from Zaxy 0.4 through 2.0. Eventloom
logs are append-only; migrations should add events,
projection metadata, indexes, docs, or compatibility shims instead of rewriting
historical records.

## Upgrade Checklist

Before and after each upgrade:

1. Back up `.eventloom/`, projection directories, and any sidecar database.
2. Run `zaxy memory status --graph` to verify Eventloom integrity and graph
   projection lag before changing code.
3. Run `zaxy doctor --beta-readiness` from the target checkout.
4. Rebuild or refresh projections with `zaxy refresh-context` or
   `zaxy reproject` when the release notes mention projection changes.
5. Re-run the documented examples or mission smoke that your integration uses.
6. Capture the final command evidence in Eventloom with
   `verification.recorded` or your normal lifecycle capture path.

## From 0.4 to 0.5

0.5 reframed the public path around Coordinator Memory for Agent Teams. The
memory substrate remains compatible with 0.4 Eventloom logs, but docs and
examples now expect the first-run flow to prove install, init, bootstrap,
checkout, doctor, and example timing.

Required actions:

- Read the updated getting-started and MCP quickstart docs before changing
  client configuration.
- Keep existing Eventloom files in place; do not flatten them into markdown or
  vector-only stores.
- Run `zaxy init` or the appropriate preset only after confirming it will not
  overwrite local MCP config unexpectedly.
- Validate public examples with the release smoke path if you package Zaxy for
  other users.

## From 0.5 to 0.6

0.6 introduced the shared native integration contract and stronger MCP response
shape checks. Existing MCP clients should keep working, but code that parses
checkout metadata should expect the `zaxy.native.v0.6` contract in native
adapter responses.

Required actions:

- Update LangGraph or CrewAI adapters to read the `zaxy.native.v0.6` metadata
  envelope instead of relying on ad hoc keys.
- Prefer `memory_bootstrap`, `memory_checkout`, and `context_assemble` for
  model-facing state instead of assembling prompt memory manually.
- Check `docs/examples/mcp-tool-contract.json` and
  `docs/examples/mcp-response-snapshots.json` when updating client tests.
- Run the MCP and native example smoke tests after changing integration code.

## From 0.6 to 0.7

0.7 added Zaxy Coordinate mission workflows. This is additive for single-agent
memory users, but multi-agent clients should move from informal transcript-only
coordination to explicit mission, worker, assignment, finding, review,
promotion, approval, checkout, and handoff events.

Required actions:

- Use Coordinate APIs or MCP tools for team state instead of storing accepted
  findings as unstructured transcript text.
- Inspect migrated mission state with `zaxy coordinate inspect`.
- Preserve finding evidence and review decisions so promoted memory remains
  auditable.
- If you use benchmark claims, regenerate CoordinationBench reports from
  tracked workloads instead of local scratch files.

## From 0.7 to 0.8

0.8 added outside-MCP model-call wrappers and provider-neutral trace export.
These paths are optional; MCP remains the framework-neutral integration route.

Required actions:

- Use the OpenAI-compatible or Claude-compatible adapters only when model calls
  happen directly in application code and you want checkout injection there.
- Keep model-call capture opt-in and bounded. Do not record raw provider
  secrets or full private request bodies in Eventloom.
- Use `zaxy trace export` when a run needs replay-derived mission, checkout,
  model-call, tool-call, review, promotion, or handoff correlation without a
  tracing vendor.
- Use `zaxy replay --from-seq` and `--to-seq` for bounded inspection of long
  sessions rather than splitting or rewriting logs.

## From 1.0 to 1.1

1.1 adds StateRecoveryBench and replay-derived Coordinate accepted-state
resolution. This is additive for existing memory users, but Coordinate clients
should treat parent-promoted accepted state as the only answerable mission state
by default.

Required actions:

- Use `zaxy state-recovery-benchmark` for accepted-state recovery claims.
  `zaxy experimental state-recovery` remains a research/debug command.
- Keep benchmark claims scoped to the production `memory_fabric_checkout`
  baseline. Associative projection rows are diagnostic only.
- If a client consumes Coordinate proof packets, expect `accepted_finding_ids`,
  review refs, promotion refs, worker source refs, and non-authoritative row
  labels to come from one replay-derived resolver.
- Preserve review and promotion events in the parent mission session; worker
  findings alone are diagnostic until promoted.

## From 0.8 to 0.9

0.9 was the API freeze candidate. The main migration task is to compare your
client usage with `docs/api-inventory.md` and remove dependencies on surfaces
marked `Internal`. Surfaces marked `Experimental` can remain in prototypes, but
they should not be required for a production 1.0 rollout.

Required actions:

- Compare MCP tools, Python imports, CLI commands, Eventloom event types,
  projection backend usage, and benchmark artifacts against
  `docs/api-inventory.md`.
- For `Stable` or `Beta` surfaces, add compatibility tests before changing
  payload keys, command options, event names, or report schemas.
- Treat pgGraph and LatticeDB backend use as experimental unless the current
  release gate explicitly promotes them.
- Do not make public benchmark claims from untracked Eventloom, query, or
  diagnostics files.

## From 0.9 to 2.0

The frozen v0.9 surfaces carry forward; 1.x and 2.0 changes are additive or
follow the freeze policy in `docs/api-inventory.md`. Notable upgrade items:

- The PyPI distribution is `zaxy-memory`; the import package and console
  command remain `zaxy`.
- 1.1.x adopted Eventloom v1 JSONL envelopes while preserving legacy log
  replay; see the changelog for envelope details.
- 2.0 added causal memory contracts, review-gated consolidation, and
  reasoning-loop primitives as new (additive) surfaces.
- Benchmark and evaluation modules moved out of the runtime package into
  `zaxy_benchmarks`; production imports of `zaxy.*benchmark*` modules should
  be removed (they were `Internal`/`Experimental` surfaces).

## Compatibility Tests

The compatibility suite should prove that old public behavior still works or
that a migration path is documented:

- MCP schema tests compare advertised tool names and input schemas.
- Response snapshot tests cover `memory_bootstrap`, `memory_checkout`,
  `memory_query`, `memory_verbatim`, `context_assemble`, `memory_feedback`, and
  coordination checkout outputs.
- Event and graph tests cover hash-chain replay, projection provenance,
  invalidation, temporal versions, inferred-edge metadata, and source
  citations.
- CLI tests cover release smoke, beta readiness, replay windows, trace export,
  Coordinate inspection, and schema migration planning.
- Benchmark guardrail tests reject reports without tracked inputs, query
  diagnostics, citation coverage, latency budgets, and fingerprint validation.

When a compatibility break is intentional, update this guide, the changelog,
the API inventory, and the narrow test that proves the old behavior is rejected
or migrated. For stable or beta surfaces listed in
`docs/examples/v1-schema-freeze.json`, record the change with
`schema.migration.proposed` before implementation and
`schema.migration.applied` after verification.

## Rollback Policy

Rollback should restore code and projections without rewriting Eventloom:

- Revert the application code or package version.
- Restore projection directories or sidecar database snapshots if projection
  schema changed.
- Keep new Eventloom records as audit history. If a fact is no longer valid,
  append `memory_invalidate` or a superseding typed event.
- Run `zaxy memory status --graph` after rollback to confirm the log and graph
  agree.
- Record rollback verification with `verification.recorded` so future checkout
  can cite the operational decision.

Related references: [api-inventory.md](api-inventory.md),
[testing.md](testing.md), [runbook.md](runbook.md), and
[README.md](../README.md).
