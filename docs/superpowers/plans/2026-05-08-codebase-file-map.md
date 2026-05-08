# Codebase File Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add file-level codebase indexing events and graph projection.

**Architecture:** Create a focused `zaxy.codebase` collector, add a `code.file.indexed` extractor, expose `MemoryFabric.ingest_codebase`, and wire a CLI command that appends events through the existing fabric.

**Tech Stack:** Python 3.11, pathlib, hashlib, Typer, pytest, ruff.

---

### Task 1: Codebase Collector

**Files:**
- Create: `src/zaxy/codebase.py`
- Create: `tests/test_codebase.py`

- [ ] **Step 1: Write failing tests**

Test that `collect_codebase_events()` indexes supported source files with language, sha256, bytes, and lines; skips hidden/cache/dependency directories; and rejects missing roots.

- [ ] **Step 2: Run red test**

Run: `pytest --no-cov tests/test_codebase.py -q`

Expected: fail because `zaxy.codebase` does not exist.

- [ ] **Step 3: Implement collector**

Add supported suffixes, excluded directory names, language inference, and event payload generation.

- [ ] **Step 4: Run green test**

Run: `pytest --no-cov tests/test_codebase.py -q`

Expected: pass.

### Task 2: Extractor

**Files:**
- Modify: `src/zaxy/extract.py`
- Modify: `tests/test_extract.py`

- [ ] **Step 1: Write failing extractor test**

Test `code.file.indexed` creates a `code_file` entity and `indexed_code_file` edge from actor to file.

- [ ] **Step 2: Run red test**

Run: `pytest --no-cov tests/test_extract.py -q`

Expected: fail because no extractor is registered.

- [ ] **Step 3: Implement extractor**

Register `code.file.indexed` and preserve code file metadata.

- [ ] **Step 4: Run green test**

Run: `pytest --no-cov tests/test_extract.py -q`

Expected: pass.

### Task 3: Fabric And CLI

**Files:**
- Modify: `src/zaxy/core.py`
- Modify: `src/zaxy/__main__.py`
- Modify: `tests/test_core.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Test `MemoryFabric.ingest_codebase()` appends collected code events to the requested session. Test `zaxy index-codebase PATH --session-id default` reports the indexed count with `MemoryFabric` mocked.

- [ ] **Step 2: Run red tests**

Run: `pytest --no-cov tests/test_core.py tests/test_cli.py -q`

Expected: fail because the method and command do not exist.

- [ ] **Step 3: Implement method and command**

Add `ingest_codebase()` beside document/transcript ingestion and a Typer command that calls it.

- [ ] **Step 4: Run green tests**

Run: `pytest --no-cov tests/test_codebase.py tests/test_extract.py tests/test_core.py tests/test_cli.py -q`

Expected: pass.

### Task 4: Docs And Verification

**Files:**
- Create: `docs/codebase.md`
- Modify: `docs/eventloom.md`
- Modify: `docs/retrieval.md`
- Modify: `docs/runbook.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Document codebase indexing**

Explain file-level scope, command usage, event shape, and future symbol extraction.

- [ ] **Step 2: Verify**

Run:

```bash
ruff check src/zaxy/codebase.py src/zaxy/extract.py src/zaxy/core.py src/zaxy/__main__.py tests/test_codebase.py tests/test_extract.py tests/test_core.py tests/test_cli.py
pytest --no-cov tests/test_codebase.py tests/test_extract.py tests/test_core.py tests/test_cli.py tests/test_docs_site.py -q
pytest --no-cov -m "not integration"
```

Expected: all checks pass.
