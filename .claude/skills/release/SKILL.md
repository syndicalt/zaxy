---
name: release
description: Cut a zaxy-memory release end-to-end — version bump (all 3 fields), CHANGELOG, PR, CI watch, rebase-merge, master-CI verify on the exact tip, GitHub Release → PyPI publish, post-verify. Use when the user says "cut a release", "ship X.Y.Z", "release this", or after merged work needs to reach PyPI.
---

# Release zaxy-memory

Publishing is **irreversible**. Two hard gates in this flow require an explicit
user go: creating the GitHub Release (step 7) is the only one that publishes.

## 0. Decide the version

Patch = fixes only. Minor = features/CLI surface changes. Any **Breaking** entry
(even a disclosed one) still fits minor pre-1.0-style per this repo's history
(3.1.0 broke `export`). Confirm choice with the user if mixed.

## 1. Preflight

```bash
cd <repo> && git checkout master && git pull -q origin master
git status --porcelain --untracked-files=no   # must be empty
git log --oneline -3
```
- Working tree must be clean. Untracked scratch (reports dirs, .claude/) stays out.
- Identify what's shipping: `git log v<PREV>..HEAD --oneline`.

## 2. Bump the version — THREE fields, not one

```bash
grep -n '<OLD_VERSION>' pyproject.toml server.json
```
Edit all of:
1. `pyproject.toml` → `version = "X.Y.Z"`
2. `server.json` top-level `"version"`
3. `server.json` → `packages[0].version`  ← the one that historically gets missed

Re-grep for the old version in both files; zero hits or you're not done.

## 3. CHANGELOG

Add/finalize `## X.Y.Z - YYYY-MM-DD` at the top (rename `## Unreleased` if present).
Keep-a-Changelog categories (`### Added/Changed/Deprecated/Breaking/Fixed`),
bold-lead bullets, candid tone. Breaking changes get a migration line.

## 4. Release PR

```bash
git checkout -b release/X.Y.Z
git add pyproject.toml server.json CHANGELOG.md
git commit  # "release: X.Y.Z — <em-dash summary>" + body + Co-Authored-By trailer
git push -u origin release/X.Y.Z
gh pr create --base master --title "release: X.Y.Z — <summary>" --body "<what it bundles>"
gh pr checks <N> --watch --interval 30
```
All checks must pass. If coverage fails see the `ci-triage` skill — never lower the floor.

## 5. Merge (rebase, never squash)

```bash
gh pr merge <N> --rebase --delete-branch
git checkout master && git pull -q origin master
```

## 6. Verify master CI on the EXACT merged tip

CI runs are per-SHA — a green "latest" may be the previous tip.

```bash
git rev-parse HEAD
gh run list --branch master --workflow CI --limit 3 --json headSha,status,conclusion,databaseId
# the run whose headSha == HEAD must conclude success:
gh run watch <RUN_ID> --interval 30 --exit-status
```

## 7. HARD GATE — ask the user

State plainly: "master is green at <SHA>; creating the GitHub Release publishes
X.Y.Z to PyPI and cannot be undone. Go?" Do not proceed on silence or prior
approvals from other releases.

## 8. Publish (Release event fires publish.yml — tag push alone does NOT)

```bash
gh release create vX.Y.Z --target master \
  --title "vX.Y.Z — <summary>" --notes-file <notes.md>
```
Notes: condensed CHANGELOG section; lead with the install one-liner when relevant.

## 9. Watch the publish + confirm PyPI

```bash
gh run list --workflow publish.yml --limit 1 --json databaseId -q '.[0].databaseId'
gh run watch <RUN_ID> --interval 20 --exit-status
# PyPI index lags ~1-2 min; poll:
curl -s https://pypi.org/pypi/zaxy-memory/X.Y.Z/json | python -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"
```
The package is **zaxy-memory** (bare `zaxy` on PyPI is an unrelated squatter — never reference it).

## 10. Post-verify

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://zaxy.io          # 200
curl -s -o /dev/null -w '%{http_code}\n' https://zaxy.io/install.sh  # 200
```
If docs/site content changed, confirm the Pages deploy ran. Report the final state
with the release URL and PyPI version, and note anything intentionally NOT shipped.

## Failure handling

- PR CI red → fix on the release branch (ci-triage skill); never merge red.
- Publish workflow fails AFTER release creation → do NOT delete/recreate the
  release; rerun via `gh run rerun` or `workflow_dispatch`, and tell the user.
- Discovered a bad artifact post-publish → you cannot unpublish; the fix is a new
  patch release. Say so plainly.
