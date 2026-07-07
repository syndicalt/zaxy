# Fable findings — codebase assessment (2026-07-06)

Assessor: Claude Fable 5, after three parallel repo surveys (code conventions,
docs/process, git patterns), two prior code reviews (2026-06-16, 2026-07-03), and
three weeks of live incident history. Scale: 10 = best on all axes (for slop,
10 = slop-free). Every deduction is tied to a named file, incident, or unfixed
review finding — not vibes.

## Scores

| Criterion | Score |
|---|---|
| 1. Project architecture & code quality | **7** |
| 2. Runtime efficiency & code economy | **6** |
| 3. AI slop (10 = none) | **8** |

---

## 1. Architecture & code quality: 7

The *architecture* is a 9: append-only hash-chained log → replayable projections →
cited checkout is a genuinely correct core abstraction, and the discipline around
it is rare — `mypy --strict` over ~100k LOC, a 92% coverage ratchet, drift-guard
tests (frozen MCP tool contract, version guard, docs pins), design-first specs
with "Done-when" gates, and a CLI deprecation-alias policy most teams never
bother with.

**What keeps the whole from 9–10:**

- **The big-module problem.** `core/fabric.py` (4,120 lines), `mcp_server.py`
  (3,668), `extract/rules.py` (3,075), plus **106 flat top-level modules** in
  `src/zaxy`. Already finding #3 in the 2026-06-16 codebase review; it has grown
  since, not shrunk. God-modules are where the next subtle bug lives.
- **Hardening lags the surface.** Two store-corruption incidents in three weeks
  (orphan-WAL 2026-06-16, then the 2026-07-06 bloat variant that *bypasses* the
  shipped self-heal because a native segfault is not a catchable exception).
  **Correction (2026-07-07):** the original text here understated 3.0.2 — the
  `close()`/`_connected` bug and compact-truncation were fixed in its Phase 0
  (commits 824328d/eed2c73), alongside the contamination removal. The findings
  still open at assessment time were the fleet-governance auth gaps and the
  embedded-store ghost-thread/TOCTOU pair — both now fixed (see backlog below).
- **Brittle failure modes at the edges.** One missing `*_FILE` secret bricks every
  CLI command *and* every hook (`Settings()` validates at import). The capture
  pipeline was fully built but wired to nothing until 3.1.1.

## 2. Runtime efficiency & code economy: 6

**What keeps it from 9–10:**

- **It misses its own stated targets.** AGENTS.md declares retrieval <300ms;
  reality is ~1.6s warm checkout and ~10–12s cold (after dedicated optimization
  work in PRs #81–83; it was 18–22s). Fine for a human, heavy for a per-turn agent
  primitive — checkout is designed to run *every turn*.
- **Unbounded projection growth.** ~500KB of events became a 397MB projection —
  ~800× amplification with no compaction pressure and no size/open-timeout guard.
  Not a perf nit: it was the direct cause of the 2026-07-06 checkout segfault.
- **Code economy is the weak axis.** ~100k LOC in `src/zaxy` is a lot of code for
  the essential complexity. Visible responsibility overlap (`hooks.py` /
  `lifecycle.py` / `observation.py` / `transcripts.py` / `claude_capture.py` /
  `codex_capture.py` / `capture_soak.py`); dead code findable on first look
  (`_merge_codex_toml`, the dead `--cov-fail-under=90` addopts); 61 MCP tools and
  a 151-command CLI (before ~55 were hidden) is surface that costs maintenance
  forever.
- **Credit where due:** the warm-path engineering is genuinely good — lazy import
  seams, incremental retrieval cache, verbatim checkpoints. The deduction is
  volume and growth, not hot-loop sloppiness.

## 3. AI slop: 8 (notably clean, not pristine)

A heavily AI-coauthored codebase (Co-Authored-By on nearly every commit) that
mostly *doesn't* read like one: comment density ~2–3% and rationale-only,
one-line behavioral docstrings, uniform style under strict tooling, commit bodies
that cite evidence, tests that assert behavior rather than echo implementation —
and the strongest anti-slop signal, a **public retraction culture** that caught
and permanently documented its own worst incident.

**What keeps it from 9–10:**

- **Sprawl-before-consolidation** — the characteristic AI-era failure even when
  every individual piece is competent: 42 releases in 7 weeks, 37 docs with
  visible near-duplicates (`mcp.md` / `mcp-quickstart.md` / `mcp-install-targets.md`;
  17 docs unreferenced from the README), 15 benchmark report dirs, 106 modules.
  Generation is outrunning gardening.
- **Stale generated-feeling artifacts kept alive.** AGENTS.md's 175-item checklist
  cites retracted-era metrics and a pre-3.1 CLI — exactly the kind of
  confidently-wrong context that poisons future AI sessions.
- **The gold-answer table existed at all.** Fixed with unusual integrity, but
  "model hard-codes the benchmark answers into the product" is the purest form of
  AI slop, and it survived on a live path for a while.

---

## One-line summary

Conception and honesty are 9s; the gap to 9–10 everywhere is the same single
theme — **consolidation debt**: split the god-modules, garden the surface
(docs/tools/modules), and put growth bounds + real hardening on the embedded
store's lifecycle. That is one focused "gardening release," not a rewrite.

## Gardening backlog — statuses (updated 2026-07-07)

1. **Decompose the god-modules** — IN PROGRESS: `extract/rules.py` split by
   event family and `mcp_server.py` transport-auth/payload-codec extraction
   delegated and under review; `core/fabric.py` mixin split queued (highest
   risk, done last). Note: a first-level decomposition already happened in
   June (PRs #74–77); this is the second pass on the residue.
2. **2026-07-03 reliability findings** — DONE. Corrected scope: `close()` +
   compact were already fixed in 3.0.2; the remaining two are now fixed —
   fleet-governance auth (PR #125, merged) and embedded-store
   ghost-thread/TOCTOU (PR #126).
3. **Projection growth bounds** — DONE (PR #126): settings-tunable pre-open
   bloat guard routing to the move-aside self-heal *before* the uncatchable
   native open, plus a stat()-only `projection_store_size` doctor check.
   `max_db_size` deliberately not set (mmap cap, not a corruption guard).
4. **AGENTS.md** — DONE (this PR): rewritten to ~105 current, honest lines;
   the 175-item stale checklist retired in favor of CHANGELOG.md.
5. **Docs garden** — DONE (this PR), with a scope correction: the audit showed
   all 37 docs are in the site nav by design and the MCP trio must NOT merge
   (two are build-gate-pinned). Real fixes: hooks.md was factually wrong about
   3.1.1 Stop-hook capture (rewritten), deprecated flat commands normalized in
   5 docs, external-ingest linked, api-inventory restamped.
6. **Checkout latency** — RESTATED honestly (this PR, in AGENTS.md): measured
   warm ~1.0–1.6s / cold ~10–12s; the <300ms figure is a roadmap goal, not a
   claim. Engineering toward it (bounded checkout window, write-path caches)
   remains open, deliberately out of this gardening pass.
