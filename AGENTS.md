# AGENTS.md — working on Zaxy

Zaxy is an event-sourced memory runtime for agent fleets: an append-only,
hash-chained JSONL log (Eventloom) is the source of truth; an embedded
LadybugDB graph projection and every cache are replayable derivatives; recall
is a cited **Memory Checkout**. Published to PyPI as **`zaxy-memory`** (CLI
binary `zaxy`; the bare `zaxy` name on PyPI is an unrelated package). Docs
site: https://zaxy.io.

**Controlling docs:** `CLAUDE.md` (operating manual: conventions, named
failure modes, quality bars, escalation rules) + `docs/architecture.md` +
`README.md`. This file is the condensed cross-tool orientation; when it and
the code disagree, the code wins — fix this file.

## Architecture in five invariants

1. `.eventloom/*.jsonl` is append-only truth. Never edit, truncate, or delete
   it. Everything under `.eventloom/projections/` is derived and rebuildable —
   move it aside (never `rm`) and replay rebuilds it.
2. Embedded LadybugDB is the default backend: file-based, in-process,
   zero-daemon. Neo4j is an optional sidecar; pgGraph and LatticeDB are
   experimental. Nothing may assume Neo4j is primary.
3. One owner per store (exclusive write lock); concurrent owners corrupt the
   WAL. Owner locking, reaping, self-heal, and the pre-open bloat guard live
   in the embedded runtime — see `docs/runbook.md`.
4. MCP is the primary interface (`memory_checkout` is the front door); the CLI
   is the human/debug surface. The MCP tool contract is snapshot-pinned in
   `docs/examples/mcp-tool-contract.json` — changing a tool means regenerating
   the snapshot in the same PR.
5. Recall is cited or it doesn't count: current facts carry `eventloom://`
   citations, and agents are told to trust only cited facts.

## Decision record (condensed)

| ADR | Decision | Status |
|-----|----------|--------|
| 1 | Event-sourced JSONL over mutable state | active |
| 2 | Hybrid extraction: deterministic rules first, LLM fallback | active |
| 3 | Graph projection backend | **superseded: embedded LadybugDB default**; Neo4j sidecar opt-in |
| 4 | Hybrid retrieval (exact + BM25 + graph traversal + embeddings) | active |
| 5 | Pathlight observability integration | active, optional |
| 6 | MCP as the primary agent interface | active |

## Tech stack

Python 3.11–3.13 · Pydantic 2 · `mcp` · Typer · embedded LadybugDB (Kuzu
fork) · pytest (+ coverage ratchet) · ruff + mypy `--strict` · Astro docs site
(`web/`, mirrors canonical `docs/*.md` at build time — edit `docs/`, never
`web/src/content/docs/`).

## Development workflow

- **Test-first (Karpathy rule): every function gets a test**, shipped in the
  same commit, with a one-line behavioral docstring.
- Gates, in CI order: `ruff check src tests zaxy_benchmarks` → `mypy src
  zaxy_benchmarks` (strict) → pytest on 3.11/3.12/3.13 → **coverage ratchet:
  ≥ 92.00% over `src/zaxy` only** (`scripts/check-coverage.py`; the
  `--cov-fail-under=90` in pytest addopts is vestigial). Never lower the floor.
- Local full runs: exclude `tests/test_doctor.py` (known local native hang;
  CI is authoritative) and pass `--benchmark-disable`.
- Branches `<type>/<kebab-slug>`; conventional commits with evidence-citing
  bodies; **rebase-merge** PRs; PRs are the unit of work.
- CLI compatibility: moving/renaming a command requires
  `register_deprecated_alias` (see `src/zaxy/cli/runtime.py`); the only
  sanctioned exception is a group/leaf collision disclosed as **Breaking**.
- Releases: version lives in **three** fields (`pyproject.toml`, `server.json`
  top-level `version`, `server.json` `packages[0].version`); publishing fires
  from GitHub **Release published**, not tag push. See `.claude/skills/release`.

## Benchmark honesty (non-negotiable)

There is a public retraction on record (`docs/benchmarks.md`): the 0.956
headline was oracle-mode with a hardcoded gold-answer table, both removed.
Rules: no gold answers or answer hints anywhere on a live path; tests assert
cited evidence, never dataset gold strings; claims are full-haystack with the
official judge, held-out validated, variance disclosed; every published run
gets a reproducibility package under `reports/benchmarks/` with a
`manifest.json` pinning the dataset sha256 and commit.

## Current measured state (2026-07-06 — update when re-measured, never guess)

| Metric | Value |
|--------|-------|
| LongMemEval-S answer accuracy (full-haystack, official judge) | **~0.90** gpt-5 reader (full 500) · **0.777** gpt-4o (held-out 130) |
| LongMemEval-S retrieval Recall@5 | 0.99 |
| Memory Checkout latency, embedded backend | **~1.0–1.6 s warm** (measured; varies with store size) · ~10–12 s cold/restart |
| Coverage ratchet | 92.00% floor over `src/zaxy` |
| MCP tools | 61 (snapshot-pinned) |

The original `<300 ms retrieval` target was aspirational and has never been
met end-to-end; treat it as a roadmap goal, not a claim. The measured checkout
numbers above are the honest ones (post cold-checkout work, PRs #81–83).
Detailed methodology and the retraction history: `docs/benchmarks.md`.

## Status & roadmap

Shipped state lives in `CHANGELOG.md` (release-by-release, candid). Design
work lives in `docs/superpowers/specs/` (`YYYY-MM-DD-<topic>-<kind>.md`,
design-first: agree before code). The old in-file completion checklist was
retired — it drifted stale and misled agents; the changelog does not.
