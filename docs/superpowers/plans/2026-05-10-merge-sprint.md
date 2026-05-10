# Zaxy Merge Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the accumulated Zaxy memory, onboarding, graph, and documentation work into a clean merge-ready release.

**Architecture:** Treat this as an integration sprint, not a feature sprint. First freeze the current working tree, then split commits by product capability, update the public site/docs narrative, run the release gate, and push/merge only after the repo tells one coherent story.

**Tech Stack:** Python 3.13, pytest, ruff, mypy, static HTML docs site, Eventloom JSONL, Neo4j, MCP.

---

### Task 1: Freeze And Inventory Current Work

**Files:**
- Inspect: `git status --short --branch`
- Inspect: `git diff --stat`
- Inspect: `git log --oneline --decorate --max-count=12`

- [ ] **Step 1: Confirm branch and dirty state**

Run:

```bash
git status --short --branch
```

Expected: `master` is ahead of `origin/master` and has tracked/untracked changes.

- [ ] **Step 2: Capture the changed surface area**

Run:

```bash
git diff --stat
git diff --name-only
```

Expected: changed files are grouped across local profile/onboarding, graph provenance, memory checkout/refs/capabilities, MCP, hooks, tests, and docs.

- [ ] **Step 3: Run the current verification baseline**

Run:

```bash
ruff check .
mypy src/zaxy
pytest -q
```

Expected: ruff clean, mypy clean, full test suite passes with coverage above 90%.

### Task 2: Split Work Into Coherent Commits

**Files:**
- Stage selectively from the current working tree.
- Do not include unrelated generated artifacts.

- [ ] **Step 1: Commit local infrastructure and onboarding cleanup**

Stage:

```bash
git add .env.example scripts/setup.sh src/zaxy/local_profile.py src/zaxy/onboarding.py tests/test_setup_script.py tests/test_onboarding.py docs/configuration.md docs/getting-started.md docs/packet-analyzer.md docs/hooks.md
git commit -m "feat: streamline local onboarding capture"
```

Expected: commit covers deterministic capture defaults, `local-codex`, localhost Neo4j profile, packet capture opt-in guidance, and Codex hook JSON safety.

- [ ] **Step 2: Commit graph provenance and projection integrity**

Stage:

```bash
git add scripts/setup_neo4j_indexes.cypher src/zaxy/graph.py src/zaxy/schema.py src/zaxy/extract.py tests/test_graph.py tests/test_extract.py tests/test_schema_migrations.py docs/eventloom.md docs/graph-schema.md
git commit -m "feat: add graph provenance integrity checks"
```

Expected: commit covers Eventloom hash-path projection, `NEXT_EVENT`/`PREVIOUS_EVENT`, graph status checks, and schema docs.

- [ ] **Step 3: Commit model-facing memory UX**

Stage:

```bash
git add src/zaxy/capabilities.py src/zaxy/refs.py src/zaxy/__init__.py src/zaxy/__main__.py src/zaxy/core.py src/zaxy/mcp_server.py src/zaxy/embedding.py tests/test_capabilities.py tests/test_refs.py tests/test_cli.py tests/test_core.py tests/test_mcp.py docs/mcp.md docs/getting-started.md AGENTS.md
git commit -m "feat: add model-facing memory checkout"
```

Expected: commit covers `memory_capabilities`, Memory Checkout, memory refs, checkout-by-ref, MCP exposure, and model-facing docs.

### Task 3: Update Public Site Narrative

**Files:**
- Modify: `site/index.html`
- Modify: `site/style.css` only if the content needs layout support.
- Test: `tests/test_docs_site.py`

- [ ] **Step 1: Add failing docs-site assertions**

Modify `tests/test_docs_site.py` so the public site is required to mention:

```python
assert "Git for LLM memory" in html
assert "Memory Checkout" in html
assert "memory_capabilities" in html
assert "deterministic capture" in html
assert "local-codex" in html
assert "NEXT_EVENT" in html
```

Run:

```bash
pytest tests/test_docs_site.py --no-cov -q
```

Expected: failure until `site/index.html` is updated.

- [ ] **Step 2: Update the site copy**

Modify `site/index.html` so the first screen and architecture sections explain:

```text
Zaxy is Git for LLM memory: an Eventloom-backed, hash-linked, replayable memory substrate with graph projection, Memory Checkout, deterministic capture, and MCP-native model awareness.
```

Expected: the site now reflects the current product direction rather than only the earlier temporal graph fabric positioning.

- [ ] **Step 3: Verify docs site**

Run:

```bash
pytest tests/test_docs_site.py --no-cov -q
bash scripts/validate-docs.sh --root .
```

Expected: docs-site tests and link validation pass.

- [ ] **Step 4: Commit public site update**

Run:

```bash
git add site/index.html site/style.css tests/test_docs_site.py
git commit -m "docs: update public memory product narrative"
```

Expected: commit is docs/site only.

### Task 4: Release Gate And Merge

**Files:**
- Inspect: full repo.

- [ ] **Step 1: Run release gate**

Run:

```bash
scripts/release-check.sh --root .
```

Expected: release checks pass. If the script requires services not currently running, capture the exact failing check and run the documented focused alternative only after confirming it is an environmental dependency.

- [ ] **Step 2: Confirm final history**

Run:

```bash
git status --short --branch
git log --oneline --decorate --max-count=12
```

Expected: clean working tree, local `master` ahead of `origin/master` by the new sprint commits plus the previous five local commits.

- [ ] **Step 3: Push merge result**

Run:

```bash
git push origin master
git status --short --branch
```

Expected: `master` is synchronized with `origin/master` and no uncommitted work remains.

