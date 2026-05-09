# Retrieval Feedback Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Let Python callers append feedback for retrieved context so useful context reinforces decay-aware memory.

**Architecture:** Add `MemoryFabric.record_context_feedback()` as a small API over existing `append()`. Positive feedback writes `memory.reinforced`; negative feedback writes audit-only `memory.feedback`. Query chunks also expose entity identity so feedback can target the retrieved graph entity.

**Tech Stack:** Python 3.11, pytest, ruff, mypy.

---

### Task 1: Feedback API Tests

**Files:**
- Modify: `tests/test_core.py`
- Modify: `tests/test_query.py`

- [x] Add failing tests for positive feedback writing `memory.reinforced`.
- [x] Add failing tests for fallback context identity.
- [x] Add failing tests for negative feedback writing `memory.feedback`.
- [x] Add failing tests for query chunks exposing entity name and type.

### Task 2: Feedback API Implementation

**Files:**
- Modify: `src/zaxy/core.py`
- Modify: `src/zaxy/query.py`

- [x] Add entity identity fields to `ContextChunk`.
- [x] Preserve entity identity in `MemoryFabric.query()` context metadata.
- [x] Add `record_context_feedback()` using the existing append path.
- [x] Add helper normalization and fallback identity parsing.
- [x] Run focused tests and ruff.

### Task 3: Docs, Verification, Commit

**Files:**
- Modify: `docs/api.md`
- Modify: `docs/agent-events.md`
- Modify: `AGENTS.md`

- [x] Document feedback events and Python API usage.
- [x] Run `ruff check src tests`.
- [x] Run `mypy src`.
- [x] Run `pytest -m "not integration" --benchmark-disable --no-cov -q`.
- [x] Update roadmap metrics and commit with `feat: add retrieval feedback events`.
