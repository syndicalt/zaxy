---
name: ci-triage
description: Diagnose and fix red CI on zaxy — fetch the right logs, reproduce locally despite the known local quirks (test_doctor hang, benchmark tables), do coverage-ratchet math correctly, and apply the test-vs-source intent rule. Use when CI is red, a check fails on a PR, "make master green", or coverage ratchet fails.
---

# CI triage for zaxy

CI jobs: `lint` (ruff+mypy), `test (3.11/3.12/3.13)` (pytest + coverage ratchet on
3.13 only), `integration` (Neo4j docker), `web` (astro build + link gate),
`package`. The authority is CI, not local runs — this machine has known quirks.

## 1. Identify what actually failed (per-SHA, not "latest")

```bash
gh run list --branch <branch> --workflow CI --limit 3 --json databaseId,headSha,conclusion,status
git rev-parse HEAD   # match headSha — a green run for an older SHA proves nothing
gh run view <RUN_ID> --json jobs -q '.jobs[] | "\(.conclusion)\t\(.name)"'
```

**Never pipe `gh pr checks` through awk/cut column hacks to decide a merge.**
Job names contain spaces (`test (3.13)`), so column extraction shows name
tokens where you expect statuses — a red matrix once read as green this way
and got merged (the 2026-07-06 doctor-map incident). Before any merge, read
the FULL untruncated table and require every row to literally say `pass`:
`gh pr checks <N>` with no pipeline.

Beware **fail-fast collateral**: matrix jobs showing `cancelled` were killed by a
sibling's failure — they are not independent failures. Find the job that says
`failure`, fix it, and the cancelled ones usually follow.

## 2. Fetch failure logs (the incantation that works)

`gh run view --log-failed` is unreliable for grep. Download the job log raw:

```bash
jid=$(gh run view <RUN_ID> --json jobs -q '.jobs[] | select(.name=="test (3.13)") | .databaseId')
gh api "repos/<owner>/<repo>/actions/jobs/$jid/logs" > /tmp/job.txt
grep -aE 'FAILED tests/|short test summary|[0-9]+ failed|Coverage ratchet' /tmp/job.txt | tail -30
```
(`-a`: logs contain binary escape bytes. If the tail only shows "operation was
canceled", it's collateral — go find the real failing job.)

## 3. Reproduce locally — with the local quirks

```bash
ruff check src tests zaxy_benchmarks
mypy src zaxy_benchmarks
pytest -m "not integration" --no-cov -q -p no:randomly \
  --ignore=tests/test_doctor.py --benchmark-disable
```
- `tests/test_doctor.py` **hangs/segfaults locally** (ladybug native lib); it is
  fine in CI. Always exclude it locally; never "fix" it locally.
- `--benchmark-disable` or the output drowns in perf tables.
- A native segfault mid-suite is an environment/store problem, not the code — see
  the `store-doctor` skill before chasing ghosts.

## 4. Classify and fix

**Lint/mypy** — mechanical; fix and move on. mypy is `--strict`: annotate, don't ignore.

**Test failures — apply the intent rule before touching anything:**
1. Read the failing assertion and the code it exercises.
2. Find documented intent: CHANGELOG entry, spec in `docs/superpowers/specs/`,
   commit body of the change that broke it.
3. Intent shipped deliberately (e.g. command re-grouped, contamination removed)
   ⇒ **update the test** to assert the new intended behavior.
4. Source behavior changed accidentally (e.g. a hardening dropped `Retry-After`
   handling) ⇒ **fix the source**; the test was right.
5. Ambiguous ⇒ stop and ask, presenting both readings with evidence.

Repo-specific test traps:
- Never assert dataset gold strings (benchmark contamination); assert content from
  the test's own cited fixture.
- `CliRunner` mixes stderr into stdout — deprecated-alias deprecation notices
  corrupt `--json` assertions; invoke the canonical grouped command.
- Changed an MCP tool? Regenerate `docs/examples/mcp-tool-contract.json` and the
  frozen tool-count assertion or the snapshot guard fails.

**Coverage ratchet** (`Coverage ratchet failed: observed X% is below floor 92.00%`):
- Only `src/zaxy` counts (`[tool.coverage.run] source`). Tests and
  `zaxy_benchmarks` **cannot** move the number. Ignore the dead
  `--cov-fail-under=90` in pytest addopts; 92.00 (`[tool.zaxy.coverage]`) is real.
- The floor is razor-thin. Order of operations:
  1. `coverage report --show-missing | grep <changed files>` — find YOUR uncovered lines.
  2. **Delete dead code first** (shrinks the denominator honestly) — verify it's
     truly unreferenced before deleting.
  3. Cover the remaining new branches with real behavioral tests.
  4. NEVER lower the floor; never add `pragma: no cover` to dodge it.
- Local totals understate CI (test_doctor excluded locally covers doctor.py in CI);
  if local ≥ 92.0 you're safe, if 91.9x it's a judgment call — push and let CI measure.

**Integration** — needs Docker + Neo4j; usually env, check container logs before code.
**Web** — `cd web && npm run build`; it link-gates, so a moved/renamed doc breaks it.

## 5. Verify + ship

```bash
ruff check src tests zaxy_benchmarks && mypy src zaxy_benchmarks
pytest -q -p no:randomly --no-cov <the failing tests> && pytest -q -p no:randomly --no-cov <their whole files>
```
Commit with a body that names root cause per failure (this repo's style), push,
`gh pr checks <N> --watch --interval 30`. Green means the run for YOUR sha — recheck headSha.
