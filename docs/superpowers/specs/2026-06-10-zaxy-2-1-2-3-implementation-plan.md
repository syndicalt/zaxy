# Zaxy 2.1–2.3 Implementation Plan

Companion to
`2026-06-10-zaxy-2-1-2-3-agent-experience-cognitive-memory-roadmap-design.md`.
That document says what and why; this one says where, in what order, and how
each slice is verified. File references are to the current tree (2.0.1,
post-decomposition: `src/zaxy/cli/` package, `src/zaxy/retrieval_plan/` and
`src/zaxy/synthesis/` packages, 33-line `__main__.py` shim).

## Ground Rules

- Every increment merges green: `ruff check src tests zaxy_benchmarks`,
  `mypy src zaxy_benchmarks` (strict), full pytest with coverage
  `fail_under = 92`, docs-site freshness (`python scripts/build-site-docs.py`).
- New behavior ships behind a setting or retrieval profile and is promoted to
  default only by its evaluation lane (roadmap §Evaluation Plan).
- New event kinds are additive Eventloom payloads — no envelope changes, no
  migration of existing logs.
- New settings go in `src/zaxy/config.py` `Settings` with env-var names
  matching the existing `SCREAMING_SNAKE` convention, documented in
  `docs/configuration.md`.
- Existing MCP tool names are never removed or renamed (stability commitment).

## Existing Seams This Plan Builds On

| Seam | Location | Used by |
| --- | --- | --- |
| MCP tool table + handler dispatch | `src/zaxy/mcp_server.py` (Tool list ~L125–1056, dispatch ~L3190+) | profiles, FoK tool, budget params |
| Checkout policy, diagnostics, prompt formatting | `src/zaxy/checkout.py` | budget packing, ordering tiers, salience diagnostics |
| Context assembly policy | `src/zaxy/context.py` | budget knapsack, stability tiers |
| Named retrieval profiles | `src/zaxy/retrieval_profile.py` | salience blend, PPR stage, plain fallback |
| Preflight checks | `src/zaxy/doctor.py` | doctor deepening |
| Hook clients + status | `src/zaxy/hooks.py`, `zaxy hook-event` in `src/zaxy/cli/workspace.py:525` | session-resumed event |
| Compaction safety audits | `src/zaxy/compaction.py` | recovery packet citation rails |
| Review-gated consolidation | `src/zaxy/consolidation_pipeline.py` | procedure mining, interference proposals |
| Metacognition event contracts | `src/zaxy/metacognition.py` | FoK signals, known-unknown links |
| Tool-call capture | `MCPServer.capture_tool_call_completed` (`src/zaxy/mcp_server.py:1198`) | procedure mining input |
| Embedding providers | `src/zaxy/embedding.py` | version stamping, quantization |
| Vector index + LRU/byte budget | `src/zaxy/embedded_graph_store.py` (`_VectorIndex`, `_evict_vector_indexes_over_budget`) | HNSW path, quantized matrices |
| Query-page log-signature cache | `src/zaxy/core.py` (`_query_page_log_signature`) | stability-tier invalidation reuses the same pattern |

## Phase 1 — Zaxy 2.1

### 1.1 Tool profiles and front door (2.1-alpha.1)

New module `src/zaxy/tool_profiles.py`:

- `CORE_TOOLS: frozenset[str]` — `memory_checkout`, `memory_append`,
  `memory_query`, `context_assemble`, `memory_feedback`, `memory_invalidate`,
  `memory_capabilities`, `memory_feeling_of_knowing` (reserved; lands 2.2).
- `resolve_profile(settings) -> frozenset[str] | None` (None = full).

Touches:

- `src/zaxy/config.py`: `mcp_tool_profile: Literal["core", "full"] = "full"`
  (env `MCP_TOOL_PROFILE`). Default stays `full` until 2.3-rc.1 promotion.
- `src/zaxy/mcp_server.py`: `list_tools` filters the Tool table through the
  profile; dispatch is unfiltered (a caller may invoke any tool by name —
  profiles change listing, not capability). Rewrite `memory_checkout` and
  `memory_capabilities` descriptions; `memory_capabilities` response gains a
  `profile` block listing hidden tools.
- `src/zaxy/cli/serving.py`: `--profile` option on `serve`, threaded to
  settings overrides like existing serve options.
- `MCPServer._ensure_workspace_instructions`
  (`src/zaxy/mcp_server.py:1277`): instruction block names checkout as the
  front door.
- Docs: `docs/mcp-quickstart.md`, `docs/mcp.md`, `docs/getting-started.md`.

Tests: `tests/test_mcp_server.py` — core profile lists ≤ 8 tools; full listing
byte-identical to today; hidden tool still dispatches; capabilities reports
the delta. Existing tool-name regression tests unchanged.

### 1.2 Verb consolidation (2.1-alpha.1, additive)

Two new umbrella tools, full-profile only at first:

- `memory_consolidation` with `operation: candidate | propose_from_log |
  status | review` → existing handlers (`handle_memory_consolidation_*`).
- `memory_confidence` with `operation: claim | trajectory | reverification |
  known_unknowns | record_known_unknown` → existing handlers.

Implementation is a thin dispatch shim per umbrella tool; legacy tools keep
their handlers and tests. No handler logic moves.

Tests: umbrella dispatch parity — each operation produces the same payload as
its legacy tool against the same fixture log.

### 1.3 Budgeted checkout (2.1-alpha.2)

New module `src/zaxy/token_budget.py`:

- `estimate_tokens(text) -> int` — deterministic chars/4 with a provider
  tokenizer hook (`Settings.token_estimator`); never network in the hot path.
- `pack_sections(sections: list[BudgetSection], max_tokens) ->
  PackResult` — greedy salience-per-token (salience defaults to section
  priority until 2.2 wires real salience), with `elided: list[ElisionRecord]`.

Touches:

- `src/zaxy/context.py`: assembly accepts `max_tokens`, builds
  `BudgetSection`s from existing source-aware sections, calls the packer.
- `src/zaxy/checkout.py`: diagnostics gain `budget_requested`, `budget_used`,
  `elided` (count + section kinds).
- `src/zaxy/mcp_server.py`: `max_tokens` (optional int) on `context_assemble`
  and `memory_checkout` input schemas, threaded through
  `handle_context_assemble` (`:2293`) and the checkout handler.
- CLI: `--max-tokens` on the checkout command in `src/zaxy/cli/`.

Tests: monotonicity (raising budget never removes a previously included
section), determinism, elision reporting, zero-budget edge (header-only packet
with explicit elision), citation coverage invariance per budget level.

### 1.4 Cache-stable ordering (2.1-alpha.2)

In `src/zaxy/checkout.py` + `src/zaxy/context.py`:

- Tag each section with a stability tier: `consolidated` (accepted facts,
  skills, procedures), `session` (working set, open tasks), `volatile`
  (query-specific evidence). Render in tier order.
- Canonical serialization for tier-1: stable sort keys, no render-time
  timestamps, no per-call randomness. Invalidate tier-1 canonical form on the
  same `(st_mtime_ns, st_size)` log signature used by
  `_query_page_log_signature` in `src/zaxy/core.py` — staleness impossible by
  construction.
- Diagnostics: `stable_prefix_chars`.

Tests: two checkouts, no intervening append → byte-identical prefix; append →
prefix changes; tier ordering asserted on a mixed fixture.

### 1.5 Compaction recovery + doctor (2.1-beta.1)

Recovery:

- `src/zaxy/cli/workspace.py` `hook_event` (`:526`): accept
  `session-resumed`; append the lifecycle event (same path as `precompact`).
- New module `src/zaxy/recovery.py`: `assemble_recovery_packet(fabric,
  session_id, precompact_seq) -> RecoveryPacket` — open tasks, accepted
  findings, known unknowns, recent verbatim anchors since last consolidation;
  every line carries event citations; run `src/zaxy/compaction.py` audit
  helpers over the packet before emit.
- Emit on stdout from the hook (harness-injectable) and as a
  `memory_bootstrap`-style MCP surface for harnesses that pull instead.
- `docs/hooks.md`: Claude Code wiring example (`SessionStart`-equivalent hook
  → `zaxy hook-event session-resumed`).

Doctor (`src/zaxy/doctor.py`): add checks — hash-chain verify over active log
tail, projection freshness vs. log signature, embedding provider availability
and dimension agreement, vector-cache budget headroom
(`VECTOR_INDEX_CACHE_MAX_BYTES`), profile sanity, hook installation
(`inspect_hook_status` already exists). Each check returns
`(status, remediation)`; CLI renders one line per failure.

Tests: scripted compact-then-resume over a fixture log recovers a seeded open
task and accepted finding with citations; doctor check matrix with each
failure injected.

Phase 1 exit: tool-adoption, budget, and cache lanes recorded in
`zaxy_benchmarks` (new lane modules follow the existing benchmark layout);
release notes; site rebuilt.

## Phase 2 — Zaxy 2.2

### 2.1 Salience ledger (2.2-alpha.1)

New module `src/zaxy/salience.py`:

- Reinforcement event taxonomy (additive Eventloom payloads):
  `memory.reinforcement` with `kind: surfaced | confirmed | promoted |
  invalidated`, target event refs, source (checkout id, feedback id,
  promotion id).
- `SalienceLedger.replay(events) -> dict[target, SalienceState]` — pure
  function of the log; exponential recency decay (half-life setting
  `SALIENCE_HALF_LIFE_DAYS`, default 30), reinforcement multipliers
  (surfaced 1.05, confirmed 1.5, promoted 2.0, invalidated 0.2 — constants in
  one table, tuned by the forgetting lane).

Emitters:

- checkout surfacing → weak reinforcement append (batched, one event per
  checkout listing the surfaced refs — keeps log volume O(checkouts));
- `handle_memory_feedback` → confirmed; coordination promotion → promoted;
  `memory_invalidate` → invalidated.

Projection: salience computed during projection rebuild and incrementally on
append; exposed in checkout diagnostics only. **No ranking change in this
increment.**

Tests: replay determinism (same log → same scores), rebuild-equals-incremental,
diagnostics composition shows each factor.

### 2.2 Forgetting, cues, gate, interference (2.2-alpha.2)

- `src/zaxy/retrieval_profile.py`: new named profile `cognitive` — relevance ×
  salience blend with attenuation floor (`SALIENCE_FLOOR`, default 0.15);
  `plain` profile preserves today's ranking exactly. Authority-bearing and
  pinned memories exempt from the floor; pinning is a small additive event
  kind + `memory_append` metadata flag.
- Attenuated results: excluded from default checkout ranking, still returned
  by `memory_query`/`memory_replay`, labeled `attenuated` in any surface that
  shows them.
- Cues: append paths capture `{mission, workspace, tool, phase}` when the
  caller provides them (MCP arguments + capture manager enrichment); checkout
  computes Jaccard cue overlap, blended in the `cognitive` profile.
- Encoding gate: `src/zaxy/salience.py` `classify_append(fabric, event) ->
  novel | reinforcing | redundant` using verbatim-index and entity-name
  overlap (no embedding call); tag in event metadata; `redundant` projects as
  reinforcement, not a new ranked entry. Setting `ENCODING_GATE_ENABLED`,
  default false.
- Interference: on `novel`-contradicts classification, emit a
  `memory_propose_belief_update` proposal via the existing review pipeline.

Tests: plain-profile byte-parity with 2.1 ranking; floor exemptions; explicit
query reaches attenuated memories; gate tags reversible by replay (re-project
with gate off → identical ranking state); proposal emission on a seeded
contradiction fixture.

### 2.3 Metamemory + graph walk + mining (2.2-beta.1)

Feeling of knowing:

- `src/zaxy/metacognition.py`: `feeling_of_knowing(state, query) ->
  FoKVerdict` over in-memory signals only — entity-name bloom filter
  (built at projection load), cue-index hit counts, salience histogram.
  Target < 1 ms; no embedding, no graph query.
- `src/zaxy/mcp_server.py`: `memory_feeling_of_knowing` tool (core profile),
  returning `likely | possible | unlikely` + signal breakdown.
- Calibration lane: log FoK verdicts vs. subsequent checkout non-emptiness;
  Brier score in the benchmark lane.

Personalized PageRank:

- New module `src/zaxy/graph_walk.py`: `personalized_pagerank(adjacency,
  seeds, alpha=0.85, iters=20, top_n)` — pure numpy power iteration over an
  adjacency snapshot fetched per session from the projection backend
  (Kuzu first; the fetch is a backend-interface method in
  `src/zaxy/projection.py` so Neo4j/Postgres implement the same contract).
  Snapshot cached with the log-signature pattern.
- Wire as an optional stage in `src/zaxy/retrieval_plan/` candidate scoring,
  enabled by the `cognitive` profile flag `graph_walk: true`.

Procedure mining:

- New module `src/zaxy/procedure_mining.py`: consume captured tool-call
  traces (from `capture_tool_call_completed` events), detect recurring
  successful sequences (normalized tool-name n-grams, ≥ 2 sessions,
  ≥ `MINING_MIN_SUPPORT`), emit skill-candidate proposals through
  `consolidation_pipeline` with trace citations.
- Surface: proposals appear in existing consolidation status/review tools; a
  `zaxy memory mine-procedures` CLI command triggers a batch pass.

Tests: FoK calibration harness with seeded corpora (known-hit, known-miss,
ambiguous); PPR correctness on a hand-built graph (analytic stationary
distribution within tolerance) and determinism; mining support threshold,
citation completeness, and review-gating (no skill without acceptance).

Phase 2 exit: forgetting, FoK, and PPR lanes green; `cognitive` profile
documented in `docs/retrieval.md`; defaults unchanged.

## Phase 3 — Zaxy 2.3

### 3.1 Embedding scale (2.3-alpha.1)

Versioning first (it gates everything else):

- `src/zaxy/embedding.py`: providers expose `version_tag` (`name@semver` or
  content hash for the deterministic hash provider); vectors stored with the
  tag (additive property on `Entity` rows in
  `src/zaxy/embedded_graph_store.py`; absent = `legacy`).
- `_VectorIndex` groups become version-keyed; search never mixes tags;
  mixed-version corpora surface in `zaxy doctor` and capabilities.
- Lazy migration: on read, if provider tag ≠ stored tag and the provider is
  available, re-embed and upsert; CLI `zaxy memory re-embed --session` for
  batch migration.

HNSW:

- `src/zaxy/embedded_graph_store.py`: when a session's vector count exceeds
  `VECTOR_ANN_THRESHOLD` (default 50_000), build a Kuzu-native HNSW index for
  that session instead of the dense matrix; below it, the existing exact path
  (already LRU/byte-budget capped) is untouched. Results carry `exact: bool`.
  No new Python dependency.

Quantization:

- Opt-in `VECTOR_QUANTIZATION: none | int8` — int8 matrices with scale
  factors, float rerank of top-k×4 oversampled candidates; byte-budget
  eviction counts quantized bytes (extend `matrix_bytes`).

Tests: version isolation (cross-tag search returns nothing rather than
garbage), lazy migration round-trip, doctor mixed-version report; HNSW
recall@10 ≥ 0.95 vs. exact on a seeded 10^5 corpus (marked slow/integration),
threshold hysteresis; int8 rerank recall and byte accounting.

### 3.2 Defaults and freeze (2.3-rc.1)

- Promote lane-backed winners: candidate default flips are
  `MCP_TOOL_PROFILE=core`, `cognitive` retrieval profile, gate-on — each flips
  only if its lane is green, with the opt-out documented.
- `docs/migration.md`: a "From 2.0 to 2.x" section per default change.
- Full gates, site rebuild, claim review against the external-validation
  policy.

## Sequencing and Dependencies

```
1.1 profiles ──► 1.2 umbrellas
1.3 budget ──► 1.4 cache ordering          (packer feeds tier renderer)
1.5 recovery+doctor                        (independent of 1.1–1.4)
2.1 salience ledger ──► 2.2 forgetting/gate ──► 2.3 FoK (uses salience
                                                histograms)
2.3 PPR, mining                            (independent of salience)
3.1 versioning ──► HNSW, quantization      (within one increment)
3.2 freeze                                 (last)
```

Parallelizable pairs for solo work: (1.3+1.4) with (1.5); (2.3 PPR) with
(2.3 mining). Everything in Phase 1 is independent of Phase 2's event
taxonomy, so 2.1-alpha can ship while salience design is still settling.

## Estimated Shape

| Increment | New modules | Touched modules | Test surface |
| --- | --- | --- | --- |
| 2.1-alpha.1 | tool_profiles | mcp_server, config, cli/serving, docs | test_mcp_server |
| 2.1-alpha.2 | token_budget | context, checkout, mcp_server, cli | new test_token_budget + test_checkout |
| 2.1-beta.1 | recovery | cli/workspace, doctor, hooks docs | new test_recovery, test_doctor |
| 2.2-alpha.1 | salience | mcp_server (emitters), projection | new test_salience |
| 2.2-alpha.2 | — | retrieval_profile, retrieval_plan, salience, core | test_retrieval_plan, test_salience |
| 2.2-beta.1 | graph_walk, procedure_mining | metacognition, mcp_server, projection, cli | new test_graph_walk, test_procedure_mining, test_metacognition |
| 2.3-alpha.1 | — | embedding, embedded_graph_store, doctor, cli | test_embedded_graph_store, new scale lane |
| 2.3-rc.1 | — | config defaults, docs | full suite |

## Open Questions (decide at increment start, not now)

- Tokenizer hook: ship chars/4 only in 2.1 and add provider tokenizers later,
  or accept an optional `tiktoken`-style extra immediately. Leaning chars/4
  only — deterministic and dependency-free.
- Reinforcement batching: one event per checkout vs. per-memory events.
  Leaning per-checkout (log volume), with per-memory refs inside the payload.
- PPR seed selection: vector top-k seeds vs. exact name-match seeds vs. both.
  Decide from the PPR lane ablation.
- Whether `memory_feeling_of_knowing` belongs in the core profile from day one
  or graduates after the calibration lane. Leaning: ship in core, labeled
  experimental in the description.
