# Zaxy 2.0 RC.1 Benchmark Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `2.0.0-rc.1` release-freeze gate that validates the exact benchmark artifacts and claim boundaries allowed for the 2.0 release candidate.

**Architecture:** Add a small release evidence contract that reads archived benchmark reports and internal lane summaries without changing retrieval, synthesis, or benchmark scoring behavior. Expose the contract through a CLI command and docs so RC evidence is reproducible, machine-readable, and separated into headline, external-anchor, and project-defined internal lanes.

**Tech Stack:** Python 3.11+, Typer CLI, dataclasses, JSON artifacts, pytest, existing Zaxy benchmark modules.

---

### Task 1: RC.1 Evidence Contract

**Files:**
- Create: `src/zaxy/rc_benchmark_freeze.py`
- Test: `tests/test_rc_benchmark_freeze.py`

- [ ] **Step 1: Write failing tests**

Add tests that create temporary LongMemEval, Harvey LAB, and internal-lane artifacts. Assert that the contract passes when all required evidence is present and fails when a lane is missing or when an internal lane is marked external.

Run: `pytest tests/test_rc_benchmark_freeze.py --no-cov -q`

Expected: FAIL because `zaxy.rc_benchmark_freeze` does not exist.

- [ ] **Step 2: Implement evidence models and validation**

Create focused dataclasses for evidence checks and a `build_rc1_benchmark_freeze_report(root: Path) -> RcBenchmarkFreezeReport` function. Validate:

- headline 500 report exists and passes floors: mean `0.95`, Answer@5 `0.90`, Recall@5 `0.99`, citation coverage `1.0`, p95 `2500`, p99 `3000`;
- run config exists beside the headline report;
- Harvey external artifacts exist and remain classified as `external_anchor`;
- internal 2.0 lane definitions include causal, consolidation, procedural, and metacognition as `project_defined_internal`;
- no internal lane is classified as external validation.

Run: `pytest tests/test_rc_benchmark_freeze.py --no-cov -q`

Expected: PASS.

### Task 2: CLI Gate

**Files:**
- Modify: `src/zaxy/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add tests for `zaxy benchmark-freeze --json` using a temporary root. Assert exit code `0` for complete fixtures and exit code `1` when a required artifact is missing.

Run: `pytest tests/test_cli.py -k benchmark_freeze --no-cov -q`

Expected: FAIL because the command is not registered.

- [ ] **Step 2: Add Typer command**

Add `benchmark-freeze` as a top-level command. It should call `build_rc1_benchmark_freeze_report`, print JSON when `--json` is passed, print a concise text summary otherwise, and exit non-zero on failed gates.

Run: `pytest tests/test_cli.py -k benchmark_freeze --no-cov -q`

Expected: PASS.

### Task 3: Docs And Claim Boundary

**Files:**
- Modify: `docs/benchmarks.md`
- Modify: `docs/testing.md`

- [ ] **Step 1: Update benchmark docs**

Add a `Zaxy 2.0 RC.1 Benchmark Freeze` section that names the gate command and states that it validates existing artifacts without changing benchmark behavior. Keep LongMemEval-compatible, Harvey LAB, and project-defined internal lanes separate.

- [ ] **Step 2: Update testing docs**

Add the exact RC.1 freeze command to the release guardrail list.

Run: `scripts/validate-docs.sh --root .`

Expected: PASS.

### Task 4: Verification

**Files:**
- Existing test and report artifacts only.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_rc_benchmark_freeze.py tests/test_cli.py -k "benchmark_freeze or rc1" --no-cov -q
```

Expected: PASS.

- [ ] **Step 2: Run existing benchmark guardrail**

Run:

```bash
python -m zaxy benchmark-compare reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json --backend zaxy-checkout --min-mean-score 0.95 --min-answer-recall-at-5 0.90 --min-recall-at-5 0.99 --min-citation-coverage 1.0 --max-p95-ms 2500 --max-p99-ms 3000
```

Expected: PASS.

- [ ] **Step 3: Run RC.1 freeze gate**

Run:

```bash
python -m zaxy benchmark-freeze --json
```

Expected: JSON report with `"passed": true`.
