# Retention Metadata Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Project retention metadata and reinforcement events into graph entities so existing decay-aware retrieval policies have useful signals.

**Architecture:** Add a small extractor helper for safe retention fields, apply it to high-value built-in memory entities and fallback events, and add a typed `memory.reinforced` extractor. Retrieval logic remains unchanged and consumes the projected properties it already understands.

**Tech Stack:** Python 3.11, pytest, ruff, mypy.

---

### Task 1: Extractor Tests

**Files:**
- Modify: `tests/test_extract.py`

- [x] Add failing coverage for retention metadata on fallback events.
- [x] Add failing coverage for retention metadata on `task.completed`.
- [x] Add failing coverage for `memory.reinforced`.
- [x] Run focused tests and confirm they fail before implementation.

### Task 2: Extractor Implementation

**Files:**
- Modify: `src/zaxy/extract.py`

- [x] Add `_retention_properties()` for `expires_at`, `importance`, `last_reinforced_at`, and `reinforcement_count`.
- [x] Add `_merge_properties()` for existing extractor property maps.
- [x] Project retention metadata on fallback events and selected memory entities.
- [x] Add `memory.reinforced` projection with `last_reinforced_at` and `reinforcement_count`.
- [x] Run focused tests and confirm they pass.

### Task 3: Docs And Roadmap

**Files:**
- Modify: `docs/agent-events.md`
- Modify: `docs/retrieval.md`
- Modify: `AGENTS.md`

- [x] Document retention metadata projection and `memory.reinforced`.
- [x] Mark roadmap item complete and update metrics after verification.
- [x] Run docs validation.

### Task 4: Full Verification And Commit

**Files:**
- All changed files.

- [x] Run `ruff check src tests`.
- [x] Run `mypy src`.
- [x] Run `pytest -m "not integration" --benchmark-disable --no-cov -q`.
- [x] Commit with `feat: project retention metadata from agent events`.
