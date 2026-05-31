# API Inventory

This inventory is the v0.9 freeze-candidate map for public Zaxy surfaces. It
does not make every surface stable. It records which contracts are stable now,
which are beta, which are experimental, and which remain internal so v1.0 work
can avoid accidental breaking changes.

## Stability Labels

- `Stable`: supported public contract; breaking changes require migration notes
  and compatibility tests.
- `Beta`: public enough for users, but payload keys or ergonomics may still
  change before v1.0 with changelog coverage.
- `Experimental`: opt-in or candidate functionality; do not build release
  promises around it.
- `Internal`: implementation detail; no compatibility promise.

## MCP Tool Contracts

MCP remains the primary framework-neutral API. Tool names and JSON input schemas
are tested through `docs/examples/mcp-tool-contract.json` and
`tests/test_mcp.py`.

| Surface | Status | Contract authority |
| --- | --- | --- |
| `memory_bootstrap`, `memory_capabilities` | `Beta` | Model-facing bootstrap and capability manifest. |
| `memory_checkout`, `context_assemble`, `context_after_turn` | `Beta` | Prompt-state and context assembly payloads. |
| `memory_query`, `memory_verbatim`, `memory_feedback` | `Beta` | Graph, source-recall, and feedback tools. |
| `memory_append`, `memory_replay`, `memory_invalidate` | `Beta` | Eventloom and temporal graph operations. |
| `memory_skill` | `Experimental` | Procedural Skill Memory lifecycle helper. |
| `coordination_start`, `coordination_worker_create`, `coordination_assign` | `Beta` | Coordinate mission setup tools. |
| `coordination_report_finding`, `coordination_review_finding`, `coordination_promote` | `Beta` | Finding review and promotion workflow. |
| `coordination_merge_brief`, `coordination_checkout`, `coordination_handoff` | `Beta` | Mission state, accepted memory, and handoff. |
| `coordination_performance_ledger`, `coordination_approval_packet`, `coordination_apply_approval` | `Beta` | Audit and approval surfaces. |

## Python SDK Public Exports

Package-level exports in `zaxy.__all__` are the public Python SDK inventory.
Module-private helpers and unexported functions are `Internal`.

| Surface | Status | Notes |
| --- | --- | --- |
| `MemoryFabric`, `MemoryCheckout`, `QueryPage` | `Beta` | Primary orchestration and retrieval result surfaces. |
| `Context`, `ContextAssemblyPolicy` | `Beta` | Prompt assembly primitives shared by MCP and adapters. |
| `CoordinationManager`, `CoordinationBrief`, `CoordinationCheckout` | `Beta` | Coordinate replay and checkout API. |
| `CoordinationApprovalPacket`, `CoordinationApprovalDecisionResult`, `CoordinationReviewExport`, `CoordinationPerformanceLedger` | `Beta` | Coordinate review and audit payloads. |
| `LangGraphMemoryAdapter`, `create_langgraph_memory_node`, `create_langgraph_memory_checkout_node`, `create_langgraph_coordination_node` | `Beta` | Native LangGraph integration path. |
| `CrewAIMemoryAdapter`, `create_crewai_memory_step`, `create_crewai_memory_checkout_step`, `create_crewai_coordination_step` | `Beta` | Native-preview CrewAI integration path. |
| `CoordinationAdapter` | `Beta` | Dependency-light direct Coordinate helper. |
| `ProjectionStore` | `Beta` | Backend contract used by query and projection code. |
| `MemoryRef`, `MemoryRefStore`, `VerbatimHit`, `VerbatimIndex` | `Beta` | Eventloom-backed refs and verbatim source recall. |
| `FrameworkIntegrationSpec`, `FrameworkIntegrationDecision`, integration render helpers | `Beta` | Config and handoff rendering helpers. |
| `HTTPSemanticConflictDetector`, `LocalSemanticConflictDetector`, `build_semantic_conflict_detector` | `Experimental` | Optional semantic conflict detection. |
| `zaxy.adapters.openai_compatible`, `zaxy.adapters.claude_compatible` | `Beta` | Direct model-call helpers outside MCP; dependency-light and opt-in. |

## Stable CLI Commands

The CLI is beta as a whole until v1.0, with the following command families
treated as public release-candidate surfaces. Internal helper functions in
`src/zaxy/__main__.py` remain `Internal`.

| Command family | Status | Notes |
| --- | --- | --- |
| `zaxy init`, `zaxy activate`, `zaxy doctor`, `zaxy status` | `Beta` | First-run, activation, and readiness checks. |
| `zaxy serve`, `zaxy replay`, `zaxy compact`, `zaxy refresh-context`, `zaxy reproject` | `Beta` | Runtime, replay, compaction, and projection operations. |
| `zaxy memory bootstrap`, `zaxy memory checkout`, `zaxy memory status`, `zaxy memory log`, `zaxy memory diff`, `zaxy memory refs` | `Beta` | Model-facing memory and Eventloom inspection. |
| `zaxy hook-status`, `zaxy hook-event`, `zaxy capture start`, `zaxy capture stop`, `zaxy capture status`, `zaxy capture-soak`, `zaxy codex-capture` | `Beta` | Deterministic capture and lifecycle observation. |
| `zaxy coordinate ...` | `Beta` | Coordinate mission, worker, finding, approval, handoff, audit, and benchmark commands. |
| `zaxy integration-template`, `zaxy integrations`, `zaxy ide-config`, `zaxy local-profile` | `Beta` | Integration discovery and config rendering. |
| `zaxy trace export` | `Beta` | Provider-neutral Eventloom-derived trace export. |
| `zaxy benchmark-inventory`, `zaxy benchmark-compare` | `Beta` | Benchmark report inventory and guardrail checks. |
| Backend candidate options such as `PROJECTION_BACKEND=pggraph` or `latticedb` | `Experimental` | Candidate backends stay behind explicit configuration. |

## Durable Eventloom Events

Eventloom JSONL is the immutable source of truth. Event type names listed here
are durable enough for migrations and replay tests; new payload fields should be
additive unless a migration event documents the change.

| Event family | Status | Examples |
| --- | --- | --- |
| Core memory and context | `Beta` | `memory.checkout.completed`, `memory.reinforced`, `memory.feedback`, `context.policy`. |
| Agent work records | `Beta` | `goal.created`, `task.proposed`, `task.completed`, `decision.made`, `verification.recorded`, `issue.diagnosed`, `handoff.created`. |
| Transcript and observations | `Beta` | `transcript.turn`, `command.completed`, `file.edit.applied`, `tool_call.completed`, `hook.checkpoint`. |
| Document and code indexing | `Beta` | `document.indexed`, code file/symbol/import/call/coverage projection events. |
| Coordinate lifecycle | `Beta` | `coordination.mission.created`, `coordination.worker.created`, `coordination.assignment.created`, `coordination.finding.reported`, `coordination.finding.reviewed`, `coordination.finding.promoted`, `coordination.handoff.created`. |
| Inferred edges | `Beta` | `inference.edge.generated` with explicit confidence, method, and evidence metadata. |
| Skill Memory | `Experimental` | `skill.proposed`, `skill.validated`, `skill.revised`, `skill.deprecated`, `skill.contradicted`, `skill.applied`, `skill.outcome_recorded`. |

## Projection Backend Contract

`ProjectionStore` is the public backend contract. The default embedded graph
runtime and Neo4j sidecar implementation are release-quality paths; pgGraph and
LatticeDB remain candidates until they meet the same benchmark, projection, and
operation gates.

| Surface | Status | Notes |
| --- | --- | --- |
| Entity, relation, source, event, and session projection semantics | `Beta` | Eventloom provenance remains authoritative. |
| Exact, keyword, vector, traversal, invalidation, projection status, and inferred-edge diagnostics | `Beta` | Required for checkout and context assembly. |
| Neo4j backend | `Beta` | Optional sidecar and control backend. |
| Embedded graph backend | `Beta` | Default sidecar-free runtime path. |
| pgGraph and LatticeDB adapters | `Experimental` | Benchmark candidates, not default release promises. |
| Direct Cypher/Kuzu implementation helpers | `Internal` | Backend internals may change without public migration guarantees. |

## Benchmark Artifact Schemas

Benchmark claims must be reproducible from tracked inputs. The release gate
checks report metadata, Markdown sidecars, query diagnostics, workload
fingerprints, and git-tracked Eventloom/query inputs.

| Artifact | Status | Contract authority |
| --- | --- | --- |
| `reports/backend-shootout/backend-shootout.json` | `Beta` | Active backend smoke report. |
| `reports/backend-shootout/longmemeval-40-backend-shootout.json` | `Beta` | Medium-scale performance report. |
| `reports/backend-shootout/longmemeval-100-backend-shootout.json` | `Beta` | Scale guardrail report. |
| `reports/benchmarks/coordination-real-v1/coordination-benchmark.json` | `Beta` | Coordinate benchmark smoke artifact. |
| `reports/benchmarks/*-diagnostics.*` | `Internal` | Local diagnostics are ignored unless promoted to tracked evidence. |
| Competitor adapter disclosures | `Experimental` | Disclosure-only until run against the same harness and inputs. |

## Migration and Freeze Policy

From v0.9 onward, changes to `Stable` or `Beta` surfaces require one of:

- a compatibility-preserving additive change;
- a migration note covering affected versions from 0.4 through the current
  release candidate;
- a schema or contract test update that proves the old and new behavior are
  intentionally handled.

`Experimental` surfaces can still change, but release notes must say so when
the change affects documented setup, benchmark results, or user-visible output.
`Internal` surfaces should not be referenced by public docs or examples except
when explaining architecture.

The current freeze-candidate manifest is
[`docs/examples/v1-schema-freeze.json`](examples/v1-schema-freeze.json). For
non-additive stable or beta changes, append `schema.migration.proposed` before
implementation and `schema.migration.applied` after verification.

Upgrade guidance from 0.4 through 0.9 is tracked in
[Migration Guide](migration.md).

Related references: [api.md](api.md), [migration.md](migration.md),
[runbook.md](runbook.md), and [README.md](../README.md).
