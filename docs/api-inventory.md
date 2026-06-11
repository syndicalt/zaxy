# API Inventory

This inventory is the maintained map of public Zaxy surfaces, first frozen at
the v0.9 release candidate and kept current through 2.0. It does not make every
surface stable. It records which contracts are stable now, which are beta,
which are experimental, and which remain internal so ongoing release work can
avoid accidental breaking changes.

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
| `memory_checkout`, `context_assemble`, `context_after_turn` | `Beta` | Prompt-state and context assembly payloads, including purpose-conditioned checkout profiles. |
| `memory_query`, `memory_verbatim`, `memory_feedback`, `memory_synthesis_artifact`, `memory_synthesis_evidence` | `Beta` | Graph, source-recall, feedback, synthesis artifact, and row-level synthesis evidence tools. |
| `memory_append`, `memory_replay`, `memory_invalidate` | `Beta` | Eventloom and temporal graph operations. |
| `memory_skill` | `Experimental` | Procedural Skill Memory lifecycle helper. |
| `memory_consolidation`, `memory_confidence` | `Beta` | Umbrella tools dispatching to the existing consolidation and confidence handlers; the single-purpose tool names remain available. |
| `memory_feeling_of_knowing` | `Experimental` | Metamemory pre-check; verdicts are predictions, never memory answers or evidence. |
| `MCP_TOOL_PROFILE` listing profiles (`core` default since 2.1.0, `full`) | `Beta` | Profiles change tool listing only; every tool stays callable by name. `MCP_TOOL_PROFILE=full` restores the pre-2.1.0 listing. |
| `coordination_start`, `coordination_worker_create`, `coordination_assign` | `Beta` | Coordinate mission setup tools. |
| `coordination_report_finding`, `coordination_review_finding`, `coordination_promote` | `Beta` | Finding review and promotion workflow. |
| `coordination_merge_brief`, `coordination_checkout`, `coordination_handoff`, `coordination_record_synthesis_artifact`, `coordination_proof_trace` | `Beta` | Mission state, accepted memory, Coordinate purpose policy, proof packets, proof trace replay, and handoff. |
| `coordination_performance_ledger`, `coordination_approval_packet`, `coordination_apply_approval` | `Beta` | Audit and approval surfaces. |

## Python SDK Public Exports

Package-level exports in `zaxy.__all__` are the public Python SDK inventory.
Module-private helpers and unexported functions are `Internal`.

| Surface | Status | Notes |
| --- | --- | --- |
| `MemoryFabric`, `MemoryCheckout`, `QueryPage` | `Beta` | Primary orchestration and retrieval result surfaces. |
| `RETRIEVAL_PROFILE` named profiles (`cognitive` default since 2.1.0; `local_fast`, `local_sota`, `hosted_sota`, `custom`) | `Beta` | Named retrieval profile resolution. `RETRIEVAL_PROFILE=local_fast` restores the pre-2.1.0 plain ranking. |
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
| `zaxy benchmark-inventory`, `zaxy benchmark-compare`, `zaxy state-recovery-benchmark` | `Beta` | Benchmark report inventory and guardrail helpers. Public claims are scoped by [benchmarks.md](benchmarks.md). |
| `zaxy memory re-embed` | `Beta` | Batch migration of projected vectors to the active embedding version tag; Eventloom events are never rewritten. |
| `zaxy memory mine-procedures` | `Experimental` | Mines recurring successful tool sequences into review-pending procedure candidates. |
| `zaxy agent-experience-lanes` | `Beta` | Internal-validation lanes for tool-adoption, budget, and cache measurements. Claims are scoped by [benchmarks.md](benchmarks.md). |
| Backend candidate options such as `PROJECTION_BACKEND=pggraph` or `latticedb` | `Experimental` | Candidate backends stay behind explicit configuration. |
| `zaxy experimental pattern-completion`, `zaxy experimental state-recovery` | `Experimental` | Research commands for associative projection diagnostics; do not use for production claims. |

## Durable Eventloom Events

Eventloom JSONL is the immutable source of truth. Event type names listed here
are durable enough for migrations and replay tests; new payload fields should be
additive unless a migration event documents the change.

| Event family | Status | Examples |
| --- | --- | --- |
| Core memory and context | `Beta` | `memory.checkout.completed`, `memory.reinforced`, `memory.feedback`, `memory.synthesis.artifact.created`, `memory.synthesis.used`, `memory.synthesis.rejected`, `memory.synthesis.corrected`, `memory.evidence.reinforced`, `memory.evidence.excluded`, `context.policy`. |
| Agent work records | `Beta` | `goal.created`, `task.proposed`, `task.completed`, `decision.made`, `verification.recorded`, `issue.diagnosed`, `handoff.created`. |
| Transcript and observations | `Beta` | `transcript.turn`, `command.completed`, `file.edit.applied`, `tool_call.completed`, `hook.checkpoint`. |
| Document and code indexing | `Beta` | `document.indexed`, code file/symbol/import/call/coverage projection events. |
| Coordinate lifecycle | `Beta` | `coordination.mission.created`, `coordination.worker.created`, `coordination.assignment.created`, `coordination.finding.reported`, `coordination.finding.reviewed`, `coordination.finding.promoted`, `coordination.handoff.created`, `coordination.proof_packet.created`. |
| Inferred edges | `Beta` | `inference.edge.generated` with explicit confidence, method, and evidence metadata. |
| Skill Memory | `Experimental` | `skill.proposed`, `skill.validated`, `skill.revised`, `skill.deprecated`, `skill.contradicted`, `skill.applied`, `skill.outcome_recorded`. |
| Salience and metacognition markers | `Experimental` | `memory.reinforcement` (surfaced/confirmed/promoted/invalidated), `metacognition.fok.predicted`, `hook.session_resumed`; non-authoritative observability state, additive payload conventions `cues`, `pinned`, `encoding`. |

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

Benchmark claims must be reproducible from tracked inputs. The current public
benchmark surface is intentionally limited to the headline 500-question
LongMemEval-compatible checkout report and the Harvey LAB external
memory-ablation report. Older suite, backend, state-recovery, purpose, and debug
artifacts are archived development history.

| Artifact | Status | Contract authority |
| --- | --- | --- |
| `reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json` | `Beta` | Current headline 500-question checkout diagnostic. |
| `reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json` | `Beta` | Harvey LAB external legal-agent memory-ablation report. |
| `reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md` | `Beta` | Publishable Harvey LAB statistics summary. |
| `reports/archive/**` and `docs/archive/**` | `Internal` | Archived development history; do not cite as current public benchmark claims. |
| `reports/benchmarks/*-diagnostics.*` | `Internal` | Local diagnostics are ignored unless promoted through [benchmarks.md](benchmarks.md). |
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

Upgrade guidance from 0.4 through 2.0 is tracked in
[Migration Guide](migration.md).

Related references: [api.md](api.md), [migration.md](migration.md),
[runbook.md](runbook.md), and [README.md](../README.md).
