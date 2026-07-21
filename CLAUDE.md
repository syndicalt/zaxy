# CLAUDE.md — Zaxy operating manual

Zaxy is an event-sourced memory runtime for agent fleets: append-only hash-chained
JSONL (Eventloom) → embedded LadybugDB graph projection → cited Memory Checkout.
Published to PyPI as **`zaxy-memory`** (CLI binary `zaxy`). Docs site: zaxy.io.

**Controlling docs:** this file + `docs/architecture.md` + `README.md`.
`AGENTS.md` is historical and stale (pre-3.1 CLI, retracted-era metrics) — do not
trust it over current code or `docs/benchmarks.md`.

## Architecture invariants (violating these is never a refactor)

- `.eventloom/*.jsonl` is the **source of truth**. Never edit, truncate, or delete
  it. Everything else — `embedded.kuzu`, caches, graph — is a replayable projection:
  safe to move aside and rebuild, never precious.
- Embedded LadybugDB is the **default backend** (file-based, in-process, zero-daemon).
  Neo4j is an optional sidecar; pggraph is experimental. Don't write code or docs
  that assume Neo4j is primary.
- One owner per store: LadybugDB takes an exclusive write lock. Multiple owners =
  corruption (see incident history in `docs/superpowers/specs/2026-06-16-embedded-single-owner-reaping-investigation.md`).
- MCP is the primary interface; the CLI is the human/debug surface. `memory_checkout`
  is the front door.

## Conventions (followed here; match them)

**Code**
- `mypy --strict` over `src` and `zaxy_benchmarks` — untyped defs are CI failures.
  Every module starts with `from __future__ import annotations`.
- ruff: `E,F,I,N,W,UP,B,C4,SIM`, line length 100, Google docstrings, double quotes.
  Run as CI does: `ruff check src tests zaxy_benchmarks && mypy src zaxy_benchmarks`.
- Comments are sparse; where present they are multi-line *rationale* at non-obvious
  seams, not narration. Docstrings: one behavioral line, sections only for real contracts.
- Heavy imports in the CLI go through patchable lazy seams (see `_graph_store`,
  `src/zaxy/cli/runtime.py:45`) so startup stays cheap and tests can monkeypatch.

**Tests**
- Karpathy rule: every function gets a test; tests ship in the same commit as src.
- Every test has a one-line behavioral docstring stating the guarantee.
- Idioms: `tmp_path`, `monkeypatch`, `typer.testing.CliRunner`; autouse
  `isolate_settings` in `tests/conftest.py` forces `ZAXY_ENV=test`.
- Drift guards are load-bearing: MCP tool contract snapshot
  (`docs/examples/mcp-tool-contract.json` + frozen tool count in `tests/test_mcp.py`),
  version guard, coverage-ratchet test, docs drift guard pinning export versions.

**Git / PRs**
- Conventional commits: `feat:|fix:|docs:|test:|release:|site:|bench:|cli:` — long
  em-dash subjects, bodies that cite evidence (numbers, file refs, before/after).
  End with `Co-Authored-By: Claude <model> <noreply@anthropic.com>`.
- Branches: `<type>/<kebab-slug>` (`fix/…`, `feat/…`, `release/…`, `bench/…`, `cli/…`).
- **Rebase-merge PRs** (`gh pr merge N --rebase --delete-branch`). Never squash: a
  squash once bundled unrelated unpushed local commits into a PR (#119). True merge
  commits only for long-lived branches needing reconciliation (like #121).
- PRs are the unit of work; the issue tracker is nearly unused.
- CHANGELOG (`Keep a Changelog` categories, candid tone — it admits mistakes) is
  updated in release commits, not every feature commit.

**CLI surface**
- Top-level commands are grouped into ordered panels (`_COMMAND_PANELS`,
  `src/zaxy/cli/runtime.py`). Bench/internal commands are hidden but runnable.
- Moving/renaming a command REQUIRES `register_deprecated_alias(old, new_path, func)`
  so scripts and agent configs never break. The only sanctioned exception is a
  group/leaf name collision, disclosed as **Breaking** in the CHANGELOG (`export` → `export bundle`).
- `hidden=True` removes a command from help/completion only — it still runs.

**Docs / site**
- Repo `docs/*.md` is canonical. `web/src/content/docs/` is a **generated mirror**
  (`web/scripts/sync-content.mjs`) — edits there are silently overwritten at build.
- Site deploys on push to master touching `web/**`, `docs/**`, `reports/**`.
- Design docs live in `docs/superpowers/specs/` as `YYYY-MM-DD-<topic>-<kind>.md`
  with a Status/Date/Owner header, numbered sections, explicit **Open questions**,
  **Done-when**, and a design-first gate ("agree before code").

**Benchmark honesty (non-negotiable; there is a public retraction on record)**
- Never hardcode gold answers or inject answer hints anywhere on a live path.
- Tests assert **cited evidence content**, never dataset gold strings — a test
  asserting a gold answer is asserting contamination.
- Claims must be: full-haystack (no oracle), official judge (`gpt-4o-2024-08-06`,
  temp 0), held-out validated, LLM-judge variance disclosed (report ~2 sig figs).
- Canonical numbers live in `docs/benchmarks.md` (currently gpt-5 **0.90** full-500,
  gpt-4o **0.777** held-out, Recall@5 **0.99**); every other surface must match it.
  The 0.956 retraction section stays forever.
- Runs get a reproducibility package in `reports/benchmarks/<name>/` with
  `manifest.json` (dataset sha256, commit, judge) + artifacts.

## Named mistakes a weaker model WILL make here — with the rule that prevents each

1. **Editing `web/src/content/docs/*.md`.** It's a build-time mirror; changes vanish.
   → Rule: docs edits go to repo `docs/*.md` only.
2. **Bumping one version field.** The version lives in **three** places:
   `pyproject.toml:3`, `server.json` top-level `version`, and `server.json`
   `packages[0].version` (that one silently sat at 3.0.2 for two releases).
   → Rule: grep both files for the old version after any bump; all three must match.
3. **Pushing a tag to publish.** `publish.yml` fires on `release: published`, not
   tag push. → Rule: publish = `gh release create vX.Y.Z --target master` — and only
   after explicit user go (PyPI is irreversible).
4. **`pip install zaxy` / assuming the `zaxy` PyPI name is ours.** It's squatted by
   an unrelated package. → Rule: the package is `zaxy-memory`, always.
5. **Trusting the local `zaxy` binary to be the repo.** It's the *installed* package
   (a stale MCP server once ran 2.2.0 against 2.5-era source for a whole session).
   → Rule: `pip show zaxy-memory` when behavior doesn't match source; reinstall
   before judging runtime behavior.
6. **Running the full local test suite and hanging.** `tests/test_doctor.py`
   segfaults/hangs locally (ladybug native lib quirk on this machine); CI is fine.
   → Rule: locally run `pytest -m "not integration" --ignore=tests/test_doctor.py
   --benchmark-disable -p no:randomly`; CI is the authority for doctor + coverage.
   **Then expect exactly 17 failures and treat them as the floor, not a signal:**
   `test_harvey_lab_benchmark.py` (11, needs podman + a host docx reader) and
   `test_packaging.py` (4) / `test_coverage_ratchet.py` (2), which shell out to
   bare `python` — absent here, only `python3` and the venv exist. All 17 pass in
   CI. → Rule: baseline the count *before* your change and compare; "17 failed"
   alone is not a regression, and chasing them wastes a session.
7. **Believing `--cov-fail-under=90` in pytest addopts.** Dead config. The real gate
   is the ratchet: `[tool.zaxy.coverage] min_total_percent = "92.00"` enforced by
   `scripts/check-coverage.py` on the 3.13 CI job — and it currently sits ~92.0x,
   razor-thin. → Rule: any new `src/zaxy` line needs a test in the same PR; coverage
   counts `src/zaxy` ONLY (tests/benchmarks don't move the number). Never lower the floor.
8. **Deleting or "fixing" `.eventloom` contents.** → Rule: JSONL is append-only truth.
   Projections (`projections/embedded.kuzu*`) may be MOVED ASIDE (never rm) to a
   `_corrupt-backup-<date>/` dir; the next checkout rebuilds them from the log.
9. **Opening a suspect store in-process.** A corrupt store can hang 2min then
   segfault, killing your shell/server. → Rule: diagnose in a `timeout`-wrapped
   subprocess; test a FRESH `ladybug.Database(tmpdir)` first to rule out the lib.
10. **Killing every `zaxy serve` process.** Some belong to other projects. → Rule:
    check `/proc/<pid>/environ` for `EVENTLOOM_PATH` and only touch processes bound
    to the store you're repairing. Prefer `zaxy doctor --repair`.
11. **Breaking a `--json` output test by invoking a deprecated alias.** Typer prints
    the deprecation notice; `CliRunner` mixes stderr into stdout. → Rule: tests
    exercise the canonical grouped command (`capture soak`, not `capture-soak`).
12. **Adding an MCP tool and failing the snapshot guard.** Tool count and contract
    are frozen in tests. → Rule: adding/changing a tool means regenerating
    `docs/examples/mcp-tool-contract.json` and the count assertion, in the same PR.
13. **Deleting a "secrets" file referenced by `.env`.** `Settings()` validates
    `*_FILE` paths at import — one missing file crashes every CLI command AND every
    hook. → Rule: comment out the `.env` line in the same change; never leave a
    dangling `*_FILE` reference.
14. **Trusting a green master check for the wrong commit.** CI runs are per-SHA; the
    "latest green" may be the previous tip. → Rule: before releasing, match the run's
    `headSha` to `git rev-parse HEAD`.
15. **Asserting dataset gold strings in tests** (e.g. `"hotels in Miami"`). That
    encodes contamination as a spec. → Rule: assert content quoted from the *cited
    evidence* in the test's own fixture.

## Quality bar per deliverable (checkable, in order)

**Any code PR**
- [ ] `ruff check src tests zaxy_benchmarks` clean; `mypy src zaxy_benchmarks` clean
- [ ] tests added/updated in the same commits; each has a behavioral docstring
- [ ] local suite green with the standard local exclusions (mistake #6)
- [ ] new/changed `src/zaxy` lines covered (ratchet, mistake #7)
- [ ] CLI moved/renamed? deprecated alias registered; MCP tool changed? snapshot regenerated
- [ ] conventional commit + `<type>/<slug>` branch + Co-Authored-By
- [ ] all CI checks green on the PR's latest SHA → rebase-merge

**A release**
- [ ] master CI green **on the exact tip SHA being released** (mistake #14)
- [ ] version identical in all three fields (mistake #2)
- [ ] CHANGELOG section for the version, categorized, dated
- [ ] explicit user go recorded → `gh release create vX.Y.Z --target master`
- [ ] `publish.yml` run concludes success; PyPI shows the new version
      (`curl -s https://pypi.org/pypi/zaxy-memory/json`)
- [ ] breaking changes: disclosed in CHANGELOG **Breaking** + migration note

**Docs / site change**
- [ ] edited under `docs/` (or `web/src/` for layout), never `web/src/content/docs/`
- [ ] every number matches `docs/benchmarks.md` canon; retraction untouched
- [ ] `cd web && npm run build` passes (it link-gates)

**A benchmark claim**
- [ ] full-haystack, official judge pinned (model + temp), no answer hints in any prompt
- [ ] held-out subset validated; judge variance stated; 2-sig-fig headline
- [ ] `reports/benchmarks/<run>/manifest.json` pins dataset sha256 + commit + artifacts

**A design/spec doc**
- [ ] `docs/superpowers/specs/YYYY-MM-DD-<topic>-<kind>.md`, Status/Date/Owner header
- [ ] has Open questions + Done-when; status says "agree before code" until the user agrees

## When uncertain — exact escalation rules

**STOP and get explicit user approval before:**
- `gh release create` / anything that publishes to PyPI (irreversible)
- posting anything outward-facing: issue/PR comments to third parties, publishing
  benchmark numbers, changing published claims (draft first, show, wait for go)
- deleting anything a user created; ANY write to `.eventloom` JSONL beyond appends
- force-push, history rewrites, or committing directly to master
- changing benchmark methodology, the coverage floor, or `docs/benchmarks.md` numbers

**Proceed without asking (then report):**
- branch + PR creation, test/lint/type fixes, CI triage, internal refactors with
  tests, canonical docs edits, moving a corrupt projection aside (it's rebuildable)

**Decision rules:**
- Test vs source disagree → find the documented intent (CHANGELOG, spec, commit
  body). Intent shipped ⇒ fix the test; behavior changed accidentally ⇒ fix the
  source; genuinely ambiguous ⇒ ask, presenting both readings with evidence.
- Docs/memory/AGENTS.md vs code disagree → code wins; verify, then fix the stale text.
- Destructive-looking operation with any doubt → move aside instead of delete.
- Native crash or hang → isolate in a timeout-wrapped subprocess before any theory.
- A claim you're about to make (a number, "X is published", "Y is the default") →
  verify with one command before asserting it; this repo's culture is cited claims.

## Local environment quirks

- **There may be no dev environment checked out.** `ruff`/`mypy`/`pytest` are not
  on PATH by default and the `zaxy` on PATH is a `uv` tool install, *not* this
  source tree (mistake #5). Nothing is verifiable until you build one:
  `uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev,embedded,export]"
  -c constraints/ci.txt`. The `-c` matches CI's pinned resolution.
- `secrets/` is gitignored, file-based creds; never commit, never echo contents.
- Zaxy's own memory hooks run in this repo (`.claude/settings.local.json`): session
  `zaxy-default`, with Stop-hook transcript capture. Breaking `Settings()` breaks them.
- Cloudflare access (`cf` CLI OAuth) is DNS+Workers-scoped only; zaxy.io redirects
  (docs., www.) are a Worker named `docs-redirect` — load-bearing, don't delete.
- GitHub Pages deploys from master; `web/public/CNAME` = `zaxy.io`.
