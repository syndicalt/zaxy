# Zaxy 2.3 Release-Readiness Research

Synthesis of four parallel research tracks completed 2026-06-11. Full
findings with evidence: `docs/research/artifacts/2-3-research/`
(backend-options.md, kuzu-usage-audit.md, promotion-pass.md,
external-validation.md, int8 benchmark JSON).

## The shape of 2.3

2.3 is a continuity-and-credibility release, not a feature release: replace
the frozen storage engine before its countdown expires, repair the
external-reproduction story before inviting anyone to use it, and run the
promotion pass where evidence (not assertion) says a default should move.

## Track 1 — Backend continuity: adopt LadybugDB, design the exit

**The countdown is already running.** Frozen kuzu 0.11.3 has no macOS or
Windows cp314 wheels today, and Python 3.15 (Oct 2026) is a hard cliff —
no compatible wheels will ever exist.

**The engine is barely load-bearing.** On a warm cache under default
config, effectively zero request-time retrieval work happens in Kuzu: 2.2
moved everything (traversal BFS, PageRank, BM25, vector selection + rerank)
to cached Python/numpy. What remains: durable bitemporal storage, per-event
writes, cold-start hydration, one uncached per-call scan
(`search_causal_neighbors` — repair candidate), and maintenance surfaces.
Every MATCH in the repo is depth ≤ 1.

**LadybugDB is viable, locally verified.** Active fork (monthly releases,
cp310–cp314 wheels on all platforms, MIT); three of our six documented
defects are fixed and verified by running our own reproductions against
0.17.1 (DROP_VECTOR_INDEX corruption, Arrow-COPY segfault, crash-on-unbound
parameter — now a silent NULL, so the choke point stays). SET-on-indexed
remains rejected; the vector index is unchanged since the fork (numpy stays
primary regardless). Storage files are incompatible but projections rebuild
by design — blast radius is two lazy import sites plus an exact pin
(`ladybug==`, NOT the stale `real-ladybug` name). Open counter-signal the
acceptance lane must clear: reported write-throughput collapse at ~60k
nodes on 0.16.x (LadybugDB#452).

**Bus factor ≈ 1** (one maintainer cut every release). Therefore the
decision is two-stage: LadybugDB as the 2.3 bridge, with a parallel
design study for a Zaxy-owned SQLite + numpy store (~1.2k LOC changed,
~600 deleted, kills the C++-wheel risk class permanently) as the structural
destination. A new full graph backend is dominated — nothing left uses a
graph-database-distinguishing capability.

**Prerequisite artifact for any path: the conformance suite.** Today there
is no backend-parametrized behavioral parity suite; embedded-only extras
degrade via silent `getattr`; the protocol has drifted in three places.
The suite converts every option's risk into mechanical verification.

## Track 2 — Promotion pass: one promotion, one demotion, one instrumentation

- **int8 default: DO NOT promote.** The high-dim latency cliff is a
  per-query `astype(int32)` allocation (125ms of the 224ms) plus non-BLAS
  int32 matmul (48ms). No int8-resident prototype clears the
  beat-exact bar. What does clear it is plain **float32 selection**
  (12.0ms vs exact 21.6ms at 50k×1536, recall 1.0 measured) at 0.5× the
  float64 bytes — a new lane-gated mode candidate, honestly labeled as not
  int8. Free repair regardless: block-wise int32 accumulation halves the
  opt-in int8 path (182→96ms) with bit-identical scores.
- **Encoding gate: the most promotable item.** Lane spec is written
  (promotion-pass.md): controlled-Jaccard probe families including the
  adversarial one — a single negation flip in a 20-token memory sits at
  J=0.905, ABOVE the redundant threshold, so the false-redundant risk is
  real and must measure 0 across two consecutive runs before any flip.
- **FoK: zero organic markers exist.** Across all 24 local workspaces, no
  agent has ever called `memory_feeling_of_knowing` — opt-in telemetry
  yields no telemetry. 2.3 must auto-emit a FoK prediction inside
  `handle_memory_checkout` (~7 days to the ~400 predictions needed for a
  powered calibration join) or the promotion question stays unanswerable
  forever. Join design (query_hash ⨝ in-thread checkout outcome) is
  specified.

## Track 3 — External validation: fix before inviting

- **The Harvey rerun would die at step one**: the adapter kit's runner
  still sets `ZAXY_PYTHONPATH` to the pre-extraction `src/` path while the
  shim imports `zaxy_benchmarks` (verified ModuleNotFoundError). Fix before
  any author contact; the email-ready ask is drafted.
- **Fresh-clone friction (P0)**: documented subset test runs exit red
  (coverage gate fires at 0.79% on partial runs); both lane commands fail
  from a fresh checkout (`zaxy_benchmarks` not packaged, cwd not on
  sys.path) with a misleading error. Workarounds exist; none documented.
- **The archived artifact corpus is not reproducible as published**: 2.2's
  determinism machinery proved bit-exact cross-machine (corpus hashes,
  bytes, recall), but the archive was generated mid-development under a
  pre-release schema with no per-artifact commit. Regenerate from released
  2.2.0 with provenance blocks (commit, version, host, schema).
- Then the reproduction packet: runner script with committed expected-output
  checksums (verified it would pass today), pinned eval environment,
  reproducible-vs-environment-dependent number separation.

## Proposed increment order

1. **2.3-a**: conformance suite + P0 reproduction fixes (lane entrypoint,
   Harvey runner, artifact regeneration with provenance) + FoK auto-emit
   instrumentation (starts the data clock) + the two free repairs
   (causal-neighbors caching, int8 block-wise).
2. **2.3-b**: LadybugDB adoption behind the conformance suite + full lane
   rerun acceptance (incl. the #452 write-throughput check); encoding-gate
   lane built and run.
3. **2.3-c**: promotion decisions on lane evidence (gate flip or hold;
   float32 selection mode); Harvey author invitation with the fixed kit;
   SQLite-store design study published for 2.4+.
4. **FoK calibration join** lands whenever its data matures (~2 weeks after
   2.3-a), independent of the release train.

## Readiness verdict

Research complete. No unknowns block planning; every 2.3 decision now has
either the evidence in hand or a specified instrument to produce it. The
two highest-risk discoveries (Harvey kit breakage, FoK zero-data) were both
silent failures that would have surfaced at the worst possible moments —
found at research cost instead.
