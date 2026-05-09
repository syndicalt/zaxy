# Agent Integration Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add direct, dependency-light Python starter templates for common agent frameworks beyond MCP.

**Architecture:** Extend the existing pure integration helper module with a template renderer that emits copyable Python snippets for LangGraph, CrewAI, and AutoGen. Add a Typer command that prints those templates, and document the direct integration path while keeping MCP as the primary interface.

**Tech Stack:** Python 3.11, Typer, pytest, ruff, mypy.

---

### Task 1: Template Renderer

**Files:**
- Modify: `src/zaxy/integrations.py`
- Modify: `tests/test_integrations.py`

- [ ] Write failing tests for `render_agent_integration_template("langgraph", session_id="zaxy-default")`, `render_agent_integration_template("crewai")`, and invalid framework rejection.
- [ ] Run `pytest --no-cov tests/test_integrations.py -q` and confirm the tests fail because the function is not implemented.
- [ ] Add `AgentFramework` and `render_agent_integration_template()` to `src/zaxy/integrations.py`.
- [ ] Keep templates dependency-light: import `MemoryFabric`, call `after_turn()` and `handoff_bundle()`, and include framework-specific function names/comments without importing framework packages.
- [ ] Run `pytest --no-cov tests/test_integrations.py -q` and confirm it passes.

### Task 2: Public API And CLI

**Files:**
- Modify: `src/zaxy/__init__.py`
- Modify: `src/zaxy/__main__.py`
- Modify: `tests/test_cli.py`

- [ ] Write a failing CLI test for `zaxy integration-template langgraph --session-id zaxy-default`.
- [ ] Run that focused test and confirm it fails because the command is missing.
- [ ] Export `render_agent_integration_template` from `zaxy.__init__`.
- [ ] Add a Typer `integration-template` command that prints the rendered template and maps invalid frameworks to `BadParameter`.
- [ ] Run `pytest --no-cov tests/test_cli.py tests/test_integrations.py -q` and confirm it passes.

### Task 3: Documentation And Roadmap

**Files:**
- Create: `docs/integrations.md`
- Modify: `docs/api.md`
- Modify: `AGENTS.md`

- [ ] Document the Python direct integration template workflow and clarify that templates avoid hard runtime dependencies.
- [ ] Add `docs/integrations.md` to API related references where appropriate.
- [ ] Mark the roadmap item complete and update the next step to retention metadata extraction and reinforcement events.
- [ ] Run `scripts/validate-docs.sh`.

### Task 4: Full Verification And Commit

**Files:**
- All changed files.

- [ ] Run `ruff check src tests`.
- [ ] Run `mypy src`.
- [ ] Run `pytest -m "not integration" --benchmark-disable --no-cov -q`.
- [ ] Update `AGENTS.md` test count with the passing suite result.
- [ ] Re-run `scripts/validate-docs.sh`.
- [ ] Commit with `feat: add direct agent integration templates`.
