# Zaxy Beta Roadmap

Zaxy's beta goal is to prove the product thesis: **Git for LLM memory**. A session with Zaxy should have durable, cited, replayable memory that materially improves long-running work compared with a session without Zaxy.

## Current Beta Posture

- Clean-repo onboarding is gated by `scripts/beta-uat.sh`.
- `zaxy doctor --beta-readiness` verifies release metadata, release gates, clean-repo UAT coverage, happy-path docs, deterministic capture docs, this roadmap, and beta release criteria.
- Deterministic Codex capture is the default local path. Packet capture remains optional diagnostics, not the default memory path.
- Memory Bootstrap and Memory Checkout are the model-facing contracts for discovering capabilities and retrieving cited current context.

## Remaining Work

1. **MemPalace-comparable benchmark lanes**
   - The temporal recall lane is implemented as `--workload temporal-recall` with a frozen workload fingerprint and citation-coverage reporting.
   - The source recall lane is implemented as `--workload source-recall` with target/distractor source paths and source-recall reporting.
   - The graph traversal lane is implemented as `--workload graph-traversal` with goal-task-completion path cases.
   - The context-collapse lane is implemented as `--workload context-collapse` with noisy transcript turns plus compact checkpoint recovery cases.
   - Benchmark guardrails are implemented with `zaxy benchmark-compare` for mean score, citation coverage, p95, p99, and baseline latency regressions.
   - Keep workload fingerprints frozen and disclose where synthetic workloads do or do not compare to external systems.

2. **Maintained adapter expansion**
   - CrewAI native-preview is implemented as a dependency-light adapter for task lifecycle callbacks.
   - Use LangGraph and CrewAI adapter usage to decide whether AutoGen or model-facing Codex/Claude Code UX is the next maintained path.

3. **Capture soak**
   - `zaxy capture-soak` reports deterministic capture coverage, latest seq/hash, stale lanes, missing lanes, and beta pass/fail status.
   - Run long local sessions with deterministic capture enabled and archive capture-soak output as release evidence.
   - Turn repeated capture gaps into concrete doctor or hook-status checks.

4. **Memory quality hardening**
   - Source-backed graph projection now creates `Source` nodes and deterministic `CITES_SOURCE` edges from projected entities and Eventloom events.
   - Continue improving graph traversal density with typed relation layers, temporal successor edges, and clearly marked inferred edges.
   - Keep Eventloom provenance as the source of truth; do not fake graph density.

5. **Beta documentation pass**
   - Keep the install -> init -> bootstrap -> capture -> checkout path short and deterministic.
   - Move optional packet analyzer setup into clearly marked advanced diagnostics.
   - Keep troubleshooting focused on real failure states: Neo4j, capture watcher, MCP process duplication, and missing observation lanes.

## Beta release criteria

Before beta, all of the following must be true:

- `scripts/beta-uat.sh` passes from a fresh throwaway workspace.
- `zaxy doctor --beta-readiness` reports `ok`.
- CI is green for Python 3.11, 3.12, and 3.13.
- Coverage remains above the configured ratchet and above 90%.
- At least one MemPalace-comparable benchmark lane beyond identity recall is implemented and documented.
- At least one maintained non-LangGraph adapter path has docs, tests, and a working handoff pattern.
- A capture soak report shows deterministic capture staying active across a long session or records the exact gaps that remain.

## Non-Goals For Beta

- Hosted multi-tenant memory storage. Remote storage can live in a separate product or repository later.
- Making packet capture mandatory. It can increase provider cost and should stay opt-in.
- Hiding uncertainty. Checkout quality, citations, warnings, and degraded states should remain visible to the model.
