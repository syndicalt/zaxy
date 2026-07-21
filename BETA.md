# Zaxy Beta Roadmap

> **Last reviewed against 3.2.0 (2026-07-21).** This file was previously edited
> when `pyproject.toml` said `version = "0.1.0"` — before 3.0.0, before the 3.1.0
> CLI restructure, and before 3.2.0. The restatement below corrects claims that
> had drifted; where a claim turned out not to hold on the default backend it is
> now marked a blocker rather than quietly left as prose.

Zaxy's beta goal is to prove the product thesis: **Git for LLM memory**. A session with Zaxy should have durable, cited, replayable memory that materially improves long-running work compared with a session without Zaxy.

## Current Beta Posture

- Clean-repo onboarding is gated by `scripts/beta-uat.sh`.
- `zaxy doctor --beta-readiness` verifies 15 release/beta gates (not the 7 once
  listed here) — run `zaxy doctor --beta-readiness --json` for the current set.
  Caveat: its `_check_beta_roadmap` check is a keyword grep against *this file*,
  so it is self-satisfying and proves nothing about the code.
- Deterministic Codex capture is the default local path. Packet capture remains optional diagnostics, not the default memory path.
- Memory Bootstrap and Memory Checkout are the model-facing contracts for discovering capabilities and retrieving cited current context.

## Remaining Work

1. **MemPalace-comparable benchmark lanes**
   - The temporal recall lane is implemented as `--workload temporal-recall` with a frozen workload fingerprint and citation-coverage reporting.
   - The source recall lane is implemented as `--workload source-recall` with target/distractor source paths and source-recall reporting.
   - The graph traversal lane is implemented as `--workload graph-traversal` with goal-task-completion path cases.
   - The context-collapse lane is implemented as `--workload context-collapse` with noisy transcript turns plus compact checkpoint recovery cases.
   - `zaxy benchmark-inventory` emits the frozen workload versions, SHA-256 fingerprints, event/query counts, product claims, and required metrics for all four lanes without requiring Neo4j or provider quota.
   - Benchmark guardrails are implemented with `zaxy benchmark-compare` for mean score, citation coverage, p95, p99, and baseline latency regressions.
   - Keep workload fingerprints frozen and disclose where synthetic workloads do or do not compare to external systems.
   - **Not yet enforced.** Fingerprints are computed deterministically, but no
     test pins the four live SHA-256 values, so editing a workload generator
     changes its fingerprint silently. A pinned-SHA drift guard is open work.

2. **Maintained adapter expansion**
   - CrewAI native-preview is implemented as a dependency-light adapter for task
     lifecycle callbacks. It is duck-typed and never imports `crewai`, and CI
     installs only `[dev]`, so the framework contact surface is unverified. The
     `crewai` handoff branch is also untested and undocumented.
   - Use LangGraph and CrewAI adapter usage to decide whether AutoGen or model-facing Codex/Claude Code UX is the next maintained path.

3. **Capture soak**
   - `zaxy capture soak` (the flat `capture-soak` is a deprecated alias) reports
     deterministic capture coverage, latest seq/hash, stale lanes, missing lanes, and beta pass/fail status.
   - Run long local sessions with deterministic capture enabled and archive capture-soak output as release evidence.
   - Turn repeated capture gaps into concrete doctor or hook-status checks.

4. **Memory quality hardening**
   - **In the optional Neo4j projection only:** source-backed projection creates
     `Source` nodes and deterministic `CITES_SOURCE` edges, and entity
     reassertions create `SUPERSEDED_BY`/`PREVIOUS_VERSION` edges. The default
     embedded backend has neither — it models supersession via `valid_to` alone.
     Embedded parity is open work.
   - Lifecycle observations with explicit task ids now create task-to-observation edges for commands, file edits, tool calls, and checkpoints.
   - The graph projection contract now has first-class inferred-edge audit metadata: `inferred`, bounded `confidence`, `inference_method`, and namespaced evidence properties.
   - Explicit `inference.edge.generated` events now project auditable inferred edges without free-text relationship guessing.
   - Task completions that explicitly cite a decision Eventloom event now generate `likely_implemented_decision` inferred-edge events.
   - `zaxy memory inferred-status` now reports inferred-edge totals, method distribution, confidence statistics, evidence coverage, source-event gaps, and representative samples.
   - **BETA BLOCKER — these two hold only on the optional Neo4j sidecar, not on
     the default embedded backend.** Source-aware inferred-edge trust scoring and
     the checkout `inferred_context` diagnostic both key off `_path_inferred_*`
     properties that only the Neo4j traversal emits, so on a default install
     uncited inferred paths are never downweighted and the diagnostic never
     attaches. No test caught this because every test hand-injects those
     properties instead of getting them from a real embedded traversal.
   - Local Neo4j UAT proves inferred edges flow from Eventloom append through
     graph traversal into Memory Checkout with relation labels and inference
     methods intact, and it runs in CI. **There is no embedded-backend
     equivalent**, which is why the blocker above went unnoticed.
   - Continue improving graph traversal density by adding generated inferred edges only when provenance and confidence are defensible.
   - Keep Eventloom provenance as the source of truth; do not fake graph density.

5. **Beta documentation pass**
   - Keep the install -> init -> bootstrap -> capture -> checkout path short and deterministic.
   - Move optional packet analyzer setup into clearly marked advanced diagnostics.
   - Keep troubleshooting focused on real failure states: graph projection
     (embedded default; Neo4j sidecar optional), capture watcher, MCP process
     duplication, and missing observation lanes. `docs/runbook.md` is already
     embedded-first; of these four, only graph projection currently has a
     troubleshooting section.

## Beta release criteria

Before beta, all of the following must be true:

- `scripts/beta-uat.sh` passes from a fresh throwaway workspace.
- `zaxy doctor --beta-readiness` reports `ok`.
- CI is green for Python 3.11, 3.12, and 3.13.
- Coverage remains above the configured ratchet and above 90%.
- All four MemPalace-comparable benchmark lanes beyond identity recall are implemented, inventoried with `zaxy benchmark-inventory`, and documented.
- At least one maintained non-LangGraph adapter path has docs, tests, and a working handoff pattern.
- A capture soak report shows deterministic capture staying active across a long
  session or records the exact gaps that remain. **Not met:** the tool exists but
  no soak report has been archived under `reports/`.

## Non-Goals For Beta

- Hosted multi-tenant memory storage. Remote storage can live in a separate product or repository later.
- Making packet capture mandatory. It can increase provider cost and should stay opt-in.
- Hiding uncertainty. Checkout quality, citations, warnings, and degraded states should remain visible to the model.
