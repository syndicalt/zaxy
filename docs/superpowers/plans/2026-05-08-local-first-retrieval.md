# Local-First Retrieval Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `zaxy local-profile` command that prints, writes, and validates an offline deterministic retrieval profile.

**Architecture:** Put pure profile rendering and validation in `src/zaxy/local_profile.py`, then keep `src/zaxy/__main__.py` limited to Typer argument handling. Documentation explains the offline baseline and leaves local HTTP model runtime integration for a later slice.

**Tech Stack:** Python 3.11, Typer, Pydantic settings, pytest, ruff.

---

### Task 1: Local Profile Module

**Files:**
- Create: `src/zaxy/local_profile.py`
- Create: `tests/test_local_profile.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import pytest

from zaxy.local_profile import check_local_profile, render_local_profile, write_local_profile


def test_render_local_profile_outputs_offline_env_without_secrets() -> None:
    text = render_local_profile()

    assert "EMBEDDING_ENABLED=true" in text
    assert "EMBEDDING_PROVIDER=hash" in text
    assert "RERANKER_PROVIDER=lexical" in text
    assert "NEO4J_AUTO_START=true" in text
    assert "OPENAI_API_KEY" not in text


def test_write_local_profile_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    target = tmp_path / ".env.local"
    target.write_text("existing=true\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_local_profile(target)


def test_check_local_profile_accepts_deterministic_defaults() -> None:
    report = check_local_profile()

    assert report["embedding_provider"] == "hash"
    assert report["reranker_provider"] == "lexical"
    assert report["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest --no-cov tests/test_local_profile.py -q`

Expected: fail because `zaxy.local_profile` does not exist.

- [ ] **Step 3: Implement module**

Create `src/zaxy/local_profile.py` with `render_local_profile()`, `write_local_profile()`, and `check_local_profile()`. The checker should use `Settings(_env_file=None, embedding_provider="hash", reranker_provider="lexical")`, `build_embedding_provider()`, and `build_reranker()`.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest --no-cov tests/test_local_profile.py -q`

Expected: pass.

### Task 2: CLI Command

**Files:**
- Modify: `src/zaxy/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add tests that call `CliRunner` for:

```python
def test_local_profile_command_prints_offline_env() -> None:
    result = runner.invoke(app, ["local-profile"])
    assert result.exit_code == 0
    assert "EMBEDDING_PROVIDER=hash" in result.output
    assert "RERANKER_PROVIDER=lexical" in result.output


def test_local_profile_command_writes_output_file(tmp_path: Path) -> None:
    target = tmp_path / ".env.local"
    result = runner.invoke(app, ["local-profile", "--output", str(target)])
    assert result.exit_code == 0
    assert "Wrote local profile" in result.output
    assert "RERANKER_PROVIDER=lexical" in target.read_text(encoding="utf-8")


def test_local_profile_check_reports_success() -> None:
    result = runner.invoke(app, ["local-profile", "--check"])
    assert result.exit_code == 0
    assert '"status": "ok"' in result.output
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest --no-cov tests/test_cli.py -q`

Expected: fail because `local-profile` command does not exist.

- [ ] **Step 3: Implement CLI wiring**

Import local profile helpers and add `@app.command("local-profile")` with `--output`, `--force`, and `--check`.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest --no-cov tests/test_cli.py tests/test_local_profile.py -q`

Expected: pass.

### Task 3: Documentation And Verification

**Files:**
- Modify: `docs/embeddings.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/runbook.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Document local-first setup**

Add the `zaxy local-profile` flow to the embedding docs, getting started guide, runbook CLI table, and mark the roadmap item complete in `AGENTS.md`.

- [ ] **Step 2: Run focused verification**

Run:

```bash
ruff check src/zaxy/local_profile.py src/zaxy/__main__.py tests/test_local_profile.py tests/test_cli.py
pytest --no-cov tests/test_local_profile.py tests/test_cli.py tests/test_docs_site.py -q
```

Expected: all checks pass.
