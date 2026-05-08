# Remote MCP Rate Limit And Audit Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session-scoped remote MCP/SSE rate limiting and append-only audit JSONL export.

**Architecture:** Add a focused `zaxy.remote_security` module with fixed-window rate limiting and audit export. Wire it into the existing SSE authorization path so stdio behavior remains unchanged.

**Tech Stack:** Python 3.11, Starlette request handlers, Typer/Pydantic settings, pytest, ruff.

---

### Task 1: Remote Security Module

**Files:**
- Create: `src/zaxy/remote_security.py`
- Create: `tests/test_remote_security.py`

- [ ] **Step 1: Write failing tests**

Add tests for `SessionRateLimiter` allowing two requests and denying the third, resetting after the time window, `AuditEventExporter` writing compact JSONL without secrets, and disabled exporter no-op behavior.

- [ ] **Step 2: Run red test**

Run: `pytest --no-cov tests/test_remote_security.py -q`

Expected: fail because `zaxy.remote_security` does not exist.

- [ ] **Step 3: Implement module**

Add `RateLimitDecision`, `SessionRateLimiter`, `RemoteAuditEvent`, and `AuditEventExporter`. Use injected clock functions for deterministic tests.

- [ ] **Step 4: Run green test**

Run: `pytest --no-cov tests/test_remote_security.py -q`

Expected: pass.

### Task 2: Config And Metrics

**Files:**
- Modify: `src/zaxy/config.py`
- Modify: `src/zaxy/metrics.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Assert default settings expose enabled rate limiting, default request/window values, disabled audit export, and default audit path. Assert metrics collector exposes a `record_rate_limit_denial(session_id)` no-op-safe method.

- [ ] **Step 2: Run red tests**

Run: `pytest --no-cov tests/test_config.py tests/test_metrics.py -q`

Expected: fail because settings and metric method are missing.

- [ ] **Step 3: Implement config and metric additions**

Add fields and metric counter without changing production auth validation.

- [ ] **Step 4: Run green tests**

Run: `pytest --no-cov tests/test_config.py tests/test_metrics.py -q`

Expected: pass.

### Task 3: MCP SSE Wiring

**Files:**
- Modify: `src/zaxy/mcp_server.py`
- Modify: `tests/test_mcp.py`

- [ ] **Step 1: Write failing tests**

Add direct tests around a request guard helper that returns session IDs for allowed requests, raises rate-limit errors on excess session requests, and audits auth-denied, allowed, and rate-limited outcomes.

- [ ] **Step 2: Run red tests**

Run: `pytest --no-cov tests/test_mcp.py -q`

Expected: fail because the request guard helper does not exist.

- [ ] **Step 3: Implement request guard**

Add a small helper around `MCPTransportAuth`, `SessionRateLimiter`, `AuditEventExporter`, and settings. Wire `/sse` and `/messages/` through it.

- [ ] **Step 4: Run green tests**

Run: `pytest --no-cov tests/test_mcp.py tests/test_remote_security.py -q`

Expected: pass.

### Task 4: Docs And Roadmap

**Files:**
- Modify: `docs/security.md`
- Modify: `docs/runbook.md`
- Modify: `docs/configuration.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Document settings and behavior**

Describe session-first rate limiting, audit JSONL export, no secret capture, and process-local limitations.

- [ ] **Step 2: Verify**

Run:

```bash
ruff check src/zaxy/remote_security.py src/zaxy/mcp_server.py src/zaxy/config.py src/zaxy/metrics.py tests/test_remote_security.py tests/test_mcp.py tests/test_config.py tests/test_metrics.py
pytest --no-cov tests/test_remote_security.py tests/test_mcp.py tests/test_config.py tests/test_metrics.py tests/test_docs_site.py -q
pytest --no-cov -m "not integration"
```

Expected: all checks pass.
