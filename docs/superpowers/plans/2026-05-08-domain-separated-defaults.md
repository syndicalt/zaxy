# Domain-Separated Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make omitted MCP `session_id` values resolve to project-domain defaults instead of global `default`.

**Architecture:** Add small domain helper functions, include domain/default session in generated MCP configs, add config fields, and replace MCP server hardcoded default fallbacks with the configured default thread.

**Tech Stack:** Python 3.11, Pydantic settings, Typer, pytest, ruff.

---

### Task 1: Domain Helpers

**Files:**
- Create: `src/zaxy/domain.py`
- Create: `tests/test_domain.py`

- [ ] Write tests for slug derivation and `domain_default_session`.
- [ ] Implement helper functions.
- [ ] Run `pytest --no-cov tests/test_domain.py -q`.

### Task 2: Integration Config

**Files:**
- Modify: `src/zaxy/integrations.py`
- Modify: `src/zaxy/__main__.py`
- Modify: `tests/test_integrations.py`
- Modify: `tests/test_cli.py`

- [ ] Write tests for stdio env and SSE headers containing domain-separated sessions.
- [ ] Implement `domain` parameter and CLI `--domain`.
- [ ] Run focused tests.

### Task 3: MCP Handler Defaults

**Files:**
- Modify: `src/zaxy/config.py`
- Modify: `src/zaxy/mcp_server.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_mcp.py`

- [ ] Add `ZAXY_DOMAIN` setting.
- [ ] Update omitted-session fallbacks to use `settings.eventloom_thread`.
- [ ] Test `context_assemble` without `session_id` uses configured thread.

### Task 4: Docs And Verification

**Files:**
- Modify: `docs/mcp.md`
- Modify: `docs/eventloom.md`
- Modify: `docs/configuration.md`
- Modify: `AGENTS.md`

- [ ] Document project domain vs session ID.
- [ ] Run ruff and non-integration tests.
