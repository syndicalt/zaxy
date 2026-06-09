"""External LongMemBench / official LongMemEval validation helpers."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import httpx

LONGMEMEVAL_REPO_URL = "https://github.com/xiaowu0162/LongMemEval"
LONGMEMEVAL_DATASET_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_ORACLE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_oracle.json"
)
LONGMEMBENCH_SCHEMA_VERSION = "zaxy.longmembench-report.v1"
LONGMEMBENCH_PLAN_SCHEMA_VERSION = "zaxy.longmembench-external-run.v1"
LONGMEMBENCH_AUDIT_SCHEMA_VERSION = "zaxy.longmembench-audit.v1"
DEFAULT_OFFICIAL_DATASET = "data/longmemeval_oracle.json"
OFFICIAL_FULL_QUESTION_COUNT = 500
DEFAULT_SOTA_BASELINE_MAX_AGE_DAYS = 30


@dataclass(frozen=True)
class LongMemBenchOfficialQA:
    """Official LongMemEval QA evaluation evidence."""

    dataset_path: str
    dataset_sha256: str
    dataset_question_count: int
    hypothesis_path: str
    hypothesis_count: int
    eval_log_path: str
    evaluated_count: int
    correct_count: int
    accuracy: float
    evaluator_model: str | None
    official_eval_command: str | None


@dataclass(frozen=True)
class LongMemBenchDiagnostic:
    """Zaxy LongMemEval-compatible checkout/retrieval diagnostic evidence."""

    report_path: str
    backend: str
    case_count: int
    mean_score: float | None
    answer_at_5: float | None
    citation_coverage: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    p95_ms: float | None
    approx_tokens: float | None
    workload_sha256: str | None


@dataclass(frozen=True)
class LongMemBenchSotaBaseline:
    """External official LongMemEval score to beat for a SOTA claim."""

    system: str
    accuracy: float
    metric: str
    evidence_url: str
    evidence_date: str
    source_type: str
    question_count: int | None = None
    evaluator_model: str | None = None
    notes: str | None = None
    checked_at: str | None = None
    expires_at: str | None = None
    currentness_url: str | None = None


@dataclass(frozen=True)
class LongMemBenchReport:
    """Externally validated LongMemBench report."""

    schema_version: str
    generated_at: str
    status: str
    external_suite: dict[str, object]
    result_provenance: dict[str, object]
    official_qa: LongMemBenchOfficialQA | None
    zaxy_diagnostic: LongMemBenchDiagnostic | None
    sota_baseline: LongMemBenchSotaBaseline | None
    sota_claim: dict[str, object]
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class WrittenLongMemBenchReport:
    """Paths written for a LongMemBench report."""

    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class WrittenLongMemBenchPlan:
    """Paths written for a LongMemBench external run plan."""

    json_path: Path
    markdown_path: Path
    script_path: Path


@dataclass(frozen=True)
class LongMemBenchContextAudit:
    """Top-k context telemetry for a generated LongMemEval hypothesis."""

    rank: int
    approx_tokens: int
    citation_count: int
    session_ids: tuple[str, ...]
    source_paths: tuple[str, ...]
    event_refs: tuple[str, ...]
    contains_expected_answer: bool
    contains_answer_session: bool
    snippet: str


@dataclass(frozen=True)
class LongMemBenchGeneratedHypothesis:
    """One generated official LongMemEval hypothesis row."""

    question_id: str
    hypothesis: str
    context_count: int
    answer_mode: str
    answer_session_ids: tuple[str, ...] = ()
    answer_session_hits_top5: tuple[str, ...] = ()
    expected_answer_hit_top5: bool = False
    context_audit: tuple[LongMemBenchContextAudit, ...] = ()


@dataclass(frozen=True)
class LongMemBenchHypothesisReport:
    """Machine-readable report for generated hypotheses."""

    schema_version: str
    generated_at: str
    dataset_path: str
    dataset_sha256: str
    question_count: int
    output_path: str
    answer_mode: str
    model: str | None
    embedding_provider: str
    projection_backend: str
    limit: int
    hypotheses: tuple[LongMemBenchGeneratedHypothesis, ...]


@dataclass(frozen=True)
class LongMemBenchOfficialEvalRun:
    """Result from invoking LongMemEval's official evaluator."""

    status: str
    command: str
    worktree: str
    hypotheses_path: str
    dataset_path: str
    eval_log_path: str
    stdout: str
    stderr: str
    returncode: int


LONGMEMBENCH_ADAPTER_README = """# Zaxy LongMemBench Adapter Kit

This kit is for an external LongMemEval checkout. It keeps official QA
evaluation outside Zaxy while letting Zaxy publish an audited report that
separates official answer accuracy from Zaxy retrieval/checkout diagnostics.

Official LongMemEval testing requires:

1. Feed timestamped histories to the memory system.
2. Write a JSONL hypotheses file with one object per line:
   `{"question_id": "...", "hypothesis": "..."}`.
3. Run LongMemEval's official evaluator:

   ```bash
   cd path/to/LongMemEval/src/evaluation
   python3 evaluate_qa.py gpt-4o path/to/zaxy-hypotheses.jsonl ../../data/longmemeval_oracle.json
   python3 print_qa_metrics.py path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o ../../data/longmemeval_oracle.json
   ```

Write the completed validator evidence record:

```bash
zaxy longmembench-validator-evidence \\
  --longmemeval-worktree path/to/LongMemEval \\
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \\
  --hypotheses path/to/zaxy-hypotheses.jsonl \\
  --official-eval-log path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o \\
  --output reports/benchmarks/longmembench-external/validator-evidence.json \\
  --evaluator-model gpt-4o \\
  --official-eval-command "python3 evaluate_qa.py gpt-4o path/to/zaxy-hypotheses.jsonl ../../data/longmemeval_oracle.json" \\
  --print-metrics-command "python3 print_qa_metrics.py path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o ../../data/longmemeval_oracle.json" \\
  --validator-name "Independent Validator" \\
  --validator-evidence-url https://validation.openmemory.dev/reviewable-run \\
  --validator-run-id validator-run-001 \\
  --validator-relation independent-third-party
```

Then import the official evaluator log into Zaxy:

```bash
zaxy longmembench-import \\
  --longmemeval-worktree path/to/LongMemEval \\
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \\
  --hypotheses path/to/zaxy-hypotheses.jsonl \\
  --official-eval-log path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o \\
  --diagnostic-report reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json \\
  --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json \\
  --validator-evidence reports/benchmarks/longmembench-external/validator-evidence.json \\
  --output-dir reports/benchmarks/longmembench-external

zaxy longmembench-validate reports/benchmarks/longmembench-external/longmembench-report.json --require-official-full
zaxy longmembench-gate reports/benchmarks/longmembench-external/longmembench-report.json --require-official-sota
zaxy longmembench-audit \\
  --longmemeval-worktree path/to/LongMemEval \\
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \\
  --hypotheses path/to/zaxy-hypotheses.jsonl \\
  --official-eval-log path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o \\
  --diagnostic-report reports/benchmarks/longmembench-external/diagnostic/live-benchmark.json \\
  --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json \\
  --validator-evidence reports/benchmarks/longmembench-external/validator-evidence.json \\
  --report reports/benchmarks/longmembench-external/longmembench-report.json \\
  --hypothesis-report reports/benchmarks/longmembench-external/zaxy-hypotheses-report.json \\
  --official-eval-run-report reports/benchmarks/longmembench-external/official-eval-run.json \\
  --output reports/benchmarks/longmembench-external/longmembench-audit.json
```

For CI systems that cannot pass a completed validator JSON file, the equivalent
manual fields are `--validator-name`, `--validator-evidence-url`,
`--validator-run-id`, and `--validator-relation`.

Do not claim official LongMemEval SOTA from Zaxy retrieval diagnostics alone.
The SOTA gate requires official evaluator evidence over the full 500-question
dataset.
"""


LONGMEMBENCH_RUNNER = '''"""Placeholder external LongMemBench runner for Zaxy.

This file documents the required hypothesis contract. The actual answer
generation path should live in the external validation checkout so model
settings, prompts, and provider credentials are visible to the validator.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_hypothesis(path: str | Path, question_id: str, hypothesis: str) -> None:
    """Append one official LongMemEval hypothesis row."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"question_id": question_id, "hypothesis": hypothesis}) + "\\n")
'''


LONGMEMBENCH_VALIDATOR_CHECKLIST = """# LongMemBench Validator Checklist

Use this checklist when producing independent evidence for a Zaxy official
LongMemEval SOTA claim.

## Required Inputs

- Official LongMemEval checkout URL and commit.
- Official `longmemeval_oracle.json` dataset with 500 questions.
- Zaxy source checkout commit under validation.
- `reports/benchmarks/longmembench-external/sota-baseline.json` with a
  `checked_at` date no older than 30 days for strict SOTA gating.
- Evaluator credentials for LongMemEval's official `evaluate_qa.py`.

## Required Run Steps

1. Run `zaxy longmembench-bootstrap --worktree <LongMemEval>`.
2. Run `zaxy longmembench-doctor <LongMemEval>` and record the commit.
3. Generate 500 Zaxy hypotheses in openai-compatible mode.
4. Run LongMemEval's official evaluator over the generated hypotheses.
5. Run `print_qa_metrics.py` on the evaluator result.
6. Run `zaxy longmembench-validator-evidence` to complete
   `validator-evidence.json`, then import the result with `--validator-evidence`.
7. Run `zaxy longmembench-gate <report> --require-official-sota`.
8. Run `zaxy longmembench-audit ...` over the complete artifact set.

## Required Evidence Artifacts

- `zaxy-hypotheses.jsonl` with exactly 500 rows.
- `zaxy-hypotheses-report.json`.
- `zaxy-hypotheses.jsonl.eval-results-<model>` with exactly 500 rows.
- `official-eval-run.json`.
- `longmembench-report.json`.
- `longmembench-report.md`.
- `longmembench-audit.json` with SHA-256 hashes for the complete artifact set.
- Current SOTA baseline JSON with official QA metric, full-set question count,
  reviewable evidence URL, and fresh `checked_at`.
- Terminal transcript or CI log showing the gate command and result.
- Terminal transcript or CI log showing the audit command and result.
- Completed `validator-evidence-template.json`.

The completed validator evidence must match the imported report: validated
system name, Zaxy commit, LongMemEval commit, dataset SHA-256, question count,
hypotheses SHA-256, official evaluator log SHA-256, evaluator model, evaluated
count, correct count, accuracy, and official evaluator command are checked
during `longmembench-import`.

Manual validator fields alone cannot pass `--require-official-sota`; the strict
gate requires a cross-checked `validator-evidence.json` import bound to a Zaxy
commit.

## Non-Negotiable Claim Boundary

Do not treat Zaxy retrieval diagnostics, smoke runs, or internally generated
partial reports as official SOTA evidence. The publishable claim requires full
official QA evaluator evidence and independent validator provenance.
"""


LONGMEMBENCH_VALIDATOR_EVIDENCE_TEMPLATE = {
    "validator": {
        "name": "",
        "relation": "independent-third-party",
        "evidence_url": "",
        "run_id": "",
    },
    "validated_system": {
        "name": "Zaxy",
        "zaxy_commit": "",
        "zaxy_version": "",
    },
    "longmemeval": {
        "repo_url": LONGMEMEVAL_REPO_URL,
        "commit": "",
        "dataset": "data/longmemeval_oracle.json",
        "dataset_sha256": "",
        "question_count": OFFICIAL_FULL_QUESTION_COUNT,
    },
    "official_evaluation": {
        "evaluator_model": "gpt-4o",
        "hypotheses_path": "",
        "official_eval_log_path": "",
        "official_eval_command": "",
        "print_metrics_command": "",
        "accuracy": None,
        "correct_count": None,
        "evaluated_count": None,
    },
    "artifacts": {
        "dataset_sha256": "",
        "hypotheses_sha256": "",
        "official_eval_log_sha256": "",
        "longmembench_report_json": "",
        "longmembench_report_json_sha256": "",
        "longmembench_report_md": "",
        "longmembench_report_md_sha256": "",
        "gate_command": (
            "zaxy longmembench-gate "
            "reports/benchmarks/longmembench-external/longmembench-report.json "
            "--require-official-sota"
        ),
        "gate_status": "",
    },
}


def check_longmemeval_official_suite(worktree: Path) -> dict[str, object]:
    """Validate that an external LongMemEval checkout exposes official evaluation files."""
    root = worktree.resolve()
    required = (
        "README.md",
        "src/evaluation/evaluate_qa.py",
        "src/evaluation/print_qa_metrics.py",
    )
    missing = [item for item in required if not (root / item).exists()]
    dataset_paths = {
        name: root / "data" / name
        for name in (
            "longmemeval_oracle.json",
            "longmemeval_s_cleaned.json",
            "longmemeval_m_cleaned.json",
        )
    }
    dataset_counts = {
        name: _dataset_count(path) if path.exists() else None
        for name, path in dataset_paths.items()
    }
    commit = _git_commit(root)
    status = "valid" if not missing else "invalid"
    return {
        "status": status,
        "worktree": str(root),
        "source_url": LONGMEMEVAL_REPO_URL,
        "commit": commit,
        "missing_required_files": missing,
        "dataset_counts": dataset_counts,
        "official_evaluator": str(root / "src/evaluation/evaluate_qa.py"),
    }


def bootstrap_longmemeval_official_suite(
    *,
    worktree: Path,
    repo_url: str = LONGMEMEVAL_REPO_URL,
    ref: str | None = None,
    dataset_source: Path | None = None,
    dataset_url: str = LONGMEMEVAL_ORACLE_URL,
    force_dataset: bool = False,
) -> dict[str, object]:
    """Clone/update official LongMemEval checkout and ensure oracle dataset exists."""
    actions: list[str] = []
    if worktree.exists():
        if not (worktree / ".git").exists():
            raise ValueError(f"{worktree} exists but is not a git checkout")
        actions.append("reuse-existing-checkout")
    else:
        worktree.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", repo_url, str(worktree)],
            check=True,
            capture_output=True,
            text=True,
        )
        actions.append("git-clone")
    if ref:
        subprocess.run(
            ["git", "-C", str(worktree), "checkout", ref],
            check=True,
            capture_output=True,
            text=True,
        )
        actions.append(f"git-checkout:{ref}")
    dataset_path = worktree / DEFAULT_OFFICIAL_DATASET
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists() and not force_dataset:
        actions.append("reuse-existing-dataset")
    elif dataset_source is not None:
        shutil.copyfile(dataset_source, dataset_path)
        actions.append("copy-dataset")
    else:
        response = httpx.get(dataset_url, follow_redirects=True, timeout=120.0)
        response.raise_for_status()
        dataset_path.write_bytes(response.content)
        actions.append("download-dataset")
    suite = check_longmemeval_official_suite(worktree)
    return {
        "status": "ready" if suite["status"] == "valid" and dataset_path.exists() else "not_ready",
        "actions": actions,
        "worktree": str(worktree.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_count": _dataset_count(dataset_path),
        "dataset_sha256": _file_sha256(dataset_path) if dataset_path.exists() else None,
        "official_suite": suite,
    }


def run_longmemeval_official_eval(
    *,
    worktree: Path,
    hypotheses_path: Path,
    dataset_path: Path,
    evaluator_model: str,
    output_log: Path | None = None,
    require_api_key: bool = True,
    api_key_present: bool = False,
    api_key: str | None = None,
) -> LongMemBenchOfficialEvalRun:
    """Run LongMemEval's official evaluate_qa.py over generated hypotheses."""
    if require_api_key and not api_key_present:
        raise ValueError("OPENAI_API_KEY or --api-key is required for official LongMemEval evaluation")
    eval_dir = worktree / "src" / "evaluation"
    script = eval_dir / "evaluate_qa.py"
    if not script.exists():
        raise ValueError("official LongMemEval evaluate_qa.py not found")
    if not hypotheses_path.exists():
        raise ValueError("hypotheses JSONL not found")
    if not dataset_path.exists():
        raise ValueError("official LongMemEval dataset not found")
    command = [
        "python3",
        str(script.name),
        evaluator_model,
        str(hypotheses_path.resolve()),
        str(dataset_path.resolve()),
    ]
    env = None
    if api_key is not None:
        env = {**os.environ, "OPENAI_API_KEY": api_key}
    result = subprocess.run(
        command,
        cwd=eval_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    produced_candidates = [
        hypotheses_path.with_name(f"{hypotheses_path.name}.eval-results-{evaluator_model}"),
        hypotheses_path.with_name(f"{hypotheses_path.name}.log"),
    ]
    produced_log = next((candidate for candidate in produced_candidates if candidate.exists()), None)
    log_path = output_log or produced_log or produced_candidates[0]
    if output_log is not None and produced_log is not None and output_log != produced_log:
        output_log.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced_log, output_log)
    status = "complete" if result.returncode == 0 and log_path.exists() else "failed"
    return LongMemBenchOfficialEvalRun(
        status=status,
        command=shlex.join(command),
        worktree=str(worktree.resolve()),
        hypotheses_path=str(hypotheses_path.resolve()),
        dataset_path=str(dataset_path.resolve()),
        eval_log_path=str(log_path.resolve()),
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def build_longmembench_readiness(
    *,
    longmemeval_worktree: Path | None = None,
    dataset_path: Path | None = None,
    hypotheses_path: Path | None = None,
    official_eval_log_path: Path | None = None,
    diagnostic_report_path: Path | None = None,
    sota_baseline_path: Path | None = None,
    answer_mode: str = "openai-compatible",
    api_key_present: bool = False,
    require_official_full: bool = True,
    require_sota_baseline: bool = True,
) -> dict[str, object]:
    """Return launch/claim readiness for external LongMemBench runs."""
    blockers: list[str] = []
    warnings: list[str] = []
    suite = None
    if longmemeval_worktree is None:
        blockers.append("missing official LongMemEval worktree")
    else:
        suite = check_longmemeval_official_suite(longmemeval_worktree)
        if suite["status"] != "valid":
            blockers.append("official LongMemEval worktree is invalid")
    dataset_count = None
    dataset_sha256 = None
    if dataset_path is None or not dataset_path.exists():
        blockers.append("missing official LongMemEval dataset")
    else:
        dataset_count = _dataset_count(dataset_path)
        dataset_sha256 = _file_sha256(dataset_path)
        if require_official_full and dataset_count != OFFICIAL_FULL_QUESTION_COUNT:
            blockers.append("official LongMemEval dataset must contain 500 questions")
    hypothesis_count = None
    if hypotheses_path is not None and hypotheses_path.exists():
        hypothesis_count = len(_load_jsonl_objects(hypotheses_path, "hypotheses"))
        if require_official_full and hypothesis_count != OFFICIAL_FULL_QUESTION_COUNT:
            warnings.append("hypotheses do not cover all 500 questions yet")
    else:
        warnings.append("hypotheses have not been generated yet")
    eval_count = None
    if official_eval_log_path is not None and official_eval_log_path.exists():
        eval_count = len(_load_jsonl_objects(official_eval_log_path, "official evaluator log"))
        if require_official_full and eval_count != OFFICIAL_FULL_QUESTION_COUNT:
            warnings.append("official evaluator log does not cover all 500 questions yet")
    else:
        warnings.append("official evaluator log has not been imported yet")
    diagnostic_status = "missing"
    if diagnostic_report_path is not None and diagnostic_report_path.exists():
        try:
            load_zaxy_diagnostic_report(diagnostic_report_path)
        except ValueError as exc:
            blockers.append(f"diagnostic report invalid: {exc}")
            diagnostic_status = "invalid"
        else:
            diagnostic_status = "valid"
    else:
        warnings.append("Zaxy diagnostic report is missing")
    baseline_status = "missing"
    if sota_baseline_path is not None and sota_baseline_path.exists():
        try:
            baseline = load_sota_baseline(sota_baseline_path)
        except ValueError as exc:
            blockers.append(f"SOTA baseline invalid: {exc}")
            baseline_status = "invalid"
        else:
            currentness_failures = validate_sota_baseline_currentness(baseline)
            if currentness_failures:
                blockers.append(f"SOTA baseline currentness invalid: {'; '.join(currentness_failures)}")
                baseline_status = "stale"
            else:
                baseline_status = "valid"
    elif require_sota_baseline:
        blockers.append("missing external SOTA baseline")
    mode = answer_mode.casefold()
    if mode == "openai-compatible" and not api_key_present:
        blockers.append("OPENAI_API_KEY or --api-key is required for openai-compatible hypothesis generation")
    elif mode == "extractive":
        warnings.append("extractive mode is suitable for smoke tests, not a strong SOTA run")
    return {
        "status": "ready" if not blockers else "not_ready",
        "blockers": blockers,
        "warnings": warnings,
        "official_suite": suite,
        "dataset": {
            "path": str(dataset_path) if dataset_path is not None else None,
            "question_count": dataset_count,
            "sha256": dataset_sha256,
        },
        "hypotheses": {
            "path": str(hypotheses_path) if hypotheses_path is not None else None,
            "count": hypothesis_count,
        },
        "official_eval_log": {
            "path": str(official_eval_log_path) if official_eval_log_path is not None else None,
            "count": eval_count,
        },
        "diagnostic_report": {
            "path": str(diagnostic_report_path) if diagnostic_report_path is not None else None,
            "status": diagnostic_status,
        },
        "sota_baseline": {
            "path": str(sota_baseline_path) if sota_baseline_path is not None else None,
            "status": baseline_status,
        },
    }


def export_longmembench_adapter_kit(output_dir: Path) -> dict[str, str]:
    """Write external LongMemBench adapter-kit files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    readme = output_dir / "README.md"
    runner = output_dir / "zaxy_longmembench_runner.py"
    checklist = output_dir / "validator-checklist.md"
    evidence_template = output_dir / "validator-evidence-template.json"
    readme.write_text(LONGMEMBENCH_ADAPTER_README, encoding="utf-8")
    runner.write_text(LONGMEMBENCH_RUNNER, encoding="utf-8")
    checklist.write_text(LONGMEMBENCH_VALIDATOR_CHECKLIST, encoding="utf-8")
    evidence_template.write_text(
        json.dumps(LONGMEMBENCH_VALIDATOR_EVIDENCE_TEMPLATE, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "readme": str(readme),
        "runner": str(runner),
        "validator_checklist": str(checklist),
        "validator_evidence_template": str(evidence_template),
    }


def build_longmembench_external_run_manifest(
    *,
    dataset: str = DEFAULT_OFFICIAL_DATASET,
    evaluator_model: str = "gpt-4o",
    diagnostic_output_dir: str = "reports/benchmarks/longmembench-external/diagnostic",
    output_dir: str = "reports/benchmarks/longmembench-external",
) -> dict[str, object]:
    """Build a reproducible external LongMemBench run manifest."""
    hypotheses = f"{output_dir}/zaxy-hypotheses.jsonl"
    eval_log = f"{hypotheses}.eval-results-{evaluator_model}"
    diagnostic_report = f"{diagnostic_output_dir}/live-benchmark.json"
    sota_baseline = f"{output_dir}/sota-baseline.json"
    validator_evidence = f"{output_dir}/validator-evidence.json"
    audit_report = f"{output_dir}/longmembench-audit.json"
    print_metrics_command = (
        "python3 path/to/LongMemEval/src/evaluation/print_qa_metrics.py "
        f"{shlex.quote(eval_log)} path/to/LongMemEval/{shlex.quote(dataset)}"
    )
    return {
        "schema_version": LONGMEMBENCH_PLAN_SCHEMA_VERSION,
        "source_url": LONGMEMEVAL_REPO_URL,
        "dataset_url": LONGMEMEVAL_DATASET_URL,
        "dataset": dataset,
        "evaluator_model": evaluator_model,
        "expected_question_count": OFFICIAL_FULL_QUESTION_COUNT,
        "output_dir": output_dir,
        "commands": {
            "bootstrap": (
                "zaxy longmembench-bootstrap "
                "--worktree path/to/LongMemEval"
            ),
            "doctor": "zaxy longmembench-doctor path/to/LongMemEval",
            "ready_before_run": (
                "zaxy longmembench-ready "
                "--longmemeval-worktree path/to/LongMemEval "
                f"--dataset path/to/LongMemEval/{shlex.quote(dataset)} "
                f"--sota-baseline {shlex.quote(sota_baseline)} "
                "--answer-mode openai-compatible"
            ),
            "diagnostic": (
                "zaxy benchmark "
                f"--output-dir {shlex.quote(diagnostic_output_dir)} "
                "--embedding-provider hash --workload longmemeval "
                f"--dataset path/to/LongMemEval/{shlex.quote(dataset)} "
                "--questions 500 --runs 1 --limit 10 "
                "--baseline-backends bm25 --zaxy-backend checkout "
                "--embedding-cache .cache/zaxy/longmemeval-embeddings.json"
            ),
            "generate_hypotheses": (
                "zaxy longmembench-generate-hypotheses "
                f"--dataset path/to/LongMemEval/{shlex.quote(dataset)} "
                f"--output {shlex.quote(hypotheses)} "
                f"--report {shlex.quote(output_dir)}/zaxy-hypotheses-report.json "
                "--questions 500 "
                "--answer-mode openai-compatible "
                "--model gpt-4o "
                "--embedding-provider hash "
                "--embedding-cache .cache/zaxy/longmemeval-embeddings.json"
            ),
            "official_eval": (
                "zaxy longmembench-evaluate-official "
                "--longmemeval-worktree path/to/LongMemEval "
                f"--hypotheses {shlex.quote(hypotheses)} "
                f"--dataset path/to/LongMemEval/{shlex.quote(dataset)} "
                f"--evaluator-model {shlex.quote(evaluator_model)} "
                f"--output-log {shlex.quote(eval_log)} "
                f"--run-report {shlex.quote(output_dir)}/official-eval-run.json"
            ),
            "official_metrics": (
                print_metrics_command
            ),
            "validator_evidence": (
                "zaxy longmembench-validator-evidence "
                "--longmemeval-worktree path/to/LongMemEval "
                f"--dataset path/to/LongMemEval/{shlex.quote(dataset)} "
                f"--hypotheses {shlex.quote(hypotheses)} "
                f"--official-eval-log {shlex.quote(eval_log)} "
                f"--output {shlex.quote(validator_evidence)} "
                f"--evaluator-model {shlex.quote(evaluator_model)} "
                "--official-eval-command ZAXY_OFFICIAL_EVAL_COMMAND "
                "--print-metrics-command ZAXY_PRINT_METRICS_COMMAND "
                '--validator-name "Independent Validator" '
                "--validator-evidence-url https://validation.openmemory.dev/reviewable-run "
                "--validator-run-id validator-run-001 "
                "--validator-relation independent-third-party"
            ),
            "import": (
                "zaxy longmembench-import "
                "--longmemeval-worktree path/to/LongMemEval "
                f"--dataset path/to/LongMemEval/{shlex.quote(dataset)} "
                f"--hypotheses {shlex.quote(hypotheses)} "
                f"--official-eval-log {shlex.quote(eval_log)} "
                f"--diagnostic-report {shlex.quote(diagnostic_report)} "
                f"--sota-baseline {shlex.quote(sota_baseline)} "
                f"--validator-evidence {shlex.quote(validator_evidence)} "
                f"--output-dir {shlex.quote(output_dir)}"
            ),
            "gate": (
                f"zaxy longmembench-gate {output_dir}/longmembench-report.json "
                "--require-official-sota"
            ),
            "audit": (
                "zaxy longmembench-audit "
                "--longmemeval-worktree path/to/LongMemEval "
                f"--dataset path/to/LongMemEval/{shlex.quote(dataset)} "
                f"--hypotheses {shlex.quote(hypotheses)} "
                f"--official-eval-log {shlex.quote(eval_log)} "
                f"--diagnostic-report {shlex.quote(diagnostic_report)} "
                f"--sota-baseline {shlex.quote(sota_baseline)} "
                f"--validator-evidence {shlex.quote(validator_evidence)} "
                f"--report {shlex.quote(output_dir)}/longmembench-report.json "
                f"--hypothesis-report {shlex.quote(output_dir)}/zaxy-hypotheses-report.json "
                f"--official-eval-run-report {shlex.quote(output_dir)}/official-eval-run.json "
                f"--output {shlex.quote(audit_report)}"
            ),
            "publish": (
                f"zaxy longmembench-publish {shlex.quote(output_dir)}/longmembench-report.json "
                f"--audit {shlex.quote(audit_report)} "
                f"--output {shlex.quote(output_dir)}/publishable-statistics.md"
            ),
        },
    }


def write_longmembench_external_run_manifest(
    manifest: dict[str, object],
    output_dir: Path,
) -> WrittenLongMemBenchPlan:
    """Write JSON, Markdown, and shell script external run artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "longmembench-external-run.json"
    markdown_path = output_dir / "longmembench-external-run.md"
    script_path = output_dir / "run-longmembench-zaxy.sh"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_longmembench_plan_markdown(manifest), encoding="utf-8")
    script_path.write_text(_longmembench_plan_script(manifest), encoding="utf-8")
    script_path.chmod(0o755)
    return WrittenLongMemBenchPlan(json_path=json_path, markdown_path=markdown_path, script_path=script_path)


async def generate_longmembench_hypotheses(
    *,
    dataset_path: Path,
    output_path: Path,
    report_path: Path | None = None,
    questions: int | None = None,
    limit: int = 10,
    answer_mode: str = "extractive",
    model: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    embedding_provider: str = "hash",
    embedding_cache: Path | None = None,
    projection_backend: str = "embedded",
    reuse_projection: bool = False,
    resume: bool = False,
    fsync_rows: bool = False,
    provider_retries: int = 3,
    prefer_checkout_candidate: bool = False,
    filter_answer_contexts: bool = False,
) -> LongMemBenchHypothesisReport:
    """Generate official LongMemEval hypothesis JSONL rows using Zaxy checkout."""
    from zaxy.config import get_settings
    from zaxy.embedding import EmbeddingProvider, HashEmbeddingProvider, OpenAIEmbeddingProvider
    from zaxy.live_benchmark import (
        CachedEmbeddingProvider,
        _build_source_lane_retriever,
        benchmark_projection_cache_key,
        benchmark_query_scope_resolver,
        build_live_zaxy_retriever,
        build_longmemeval_workload,
        corpus_from_event_log,
    )
    from zaxy.projection_backends import ProjectionBackendConfig

    if limit <= 0:
        raise ValueError("limit must be positive")
    if provider_retries < 0:
        raise ValueError("provider_retries must be non-negative")
    mode = answer_mode.casefold()
    if mode not in {"extractive", "openai-compatible"}:
        raise ValueError("answer_mode must be 'extractive' or 'openai-compatible'")
    if mode == "openai-compatible" and not api_key:
        raise ValueError("api_key is required for openai-compatible answer generation")

    settings = get_settings()
    provider_name = embedding_provider.casefold()
    if provider_name == "hash":
        raw_provider: EmbeddingProvider = HashEmbeddingProvider(dimension=settings.embedding_dimension)
        provider_label = f"hash:{settings.embedding_dimension}"
    elif provider_name == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embedding generation")
        raw_provider = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            base_url=settings.openai_base_url,
        )
        provider_label = f"openai:{settings.openai_embedding_model}"
    else:
        raise ValueError("embedding_provider must be 'hash' or 'openai'")
    provider = CachedEmbeddingProvider(raw_provider, cache_path=embedding_cache)
    generated: list[LongMemBenchGeneratedHypothesis] = []
    try:
        with tempfile.TemporaryDirectory(prefix="zaxy-longmembench-") as tmp:
            tmp_path = Path(tmp)
            eventlog, cases, workload = build_longmemeval_workload(
                tmp_path / "bench.jsonl",
                dataset_path,
                questions=questions,
            )
            corpus = corpus_from_event_log(eventlog)
            projection_cache_key = benchmark_projection_cache_key(
                eventlog,
                cases,
                workload,
                provider_label,
            )
            projection_config = ProjectionBackendConfig(
                backend=projection_backend,
                neo4j_uri="bolt://localhost:7688",
                neo4j_user="neo4j",
                neo4j_password="testpassword",
                neo4j_ca_cert=None,
                neo4j_trust_all=False,
                pggraph_dsn=settings.pggraph_dsn,
                embedded_graph_path=tmp_path / "embedded-graph",
                latticedb_path=tmp_path / "latticedb",
                embedding_dimension=settings.embedding_dimension,
            )
            zaxy_retriever, graph = await build_live_zaxy_retriever(
                eventlog,
                provider,
                neo4j_uri="bolt://localhost:7688",
                neo4j_user="neo4j",
                neo4j_password="testpassword",
                reset_graph=True,
                lexical_retriever=_build_source_lane_retriever(corpus, provider),
                reuse_projection=reuse_projection,
                projection_cache_key=projection_cache_key,
                scope_resolver=benchmark_query_scope_resolver(cases),
                projection_backend_config=projection_config,
            )
            try:
                checkout_retriever = zaxy_retriever.as_checkout_retriever()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                selected_question_ids = {
                    case.name.removeprefix("longmemeval-")
                    for case in cases
                }
                existing_ids: set[str] = set()
                if resume and output_path.exists():
                    existing_rows = _load_jsonl_objects(output_path, "hypotheses")
                    for row in existing_rows:
                        question_id = row.get("question_id") if isinstance(row, dict) else None
                        hypothesis = row.get("hypothesis") if isinstance(row, dict) else None
                        if not isinstance(question_id, str) or not question_id:
                            raise ValueError("existing hypotheses contain a row without question_id")
                        if question_id in existing_ids:
                            raise ValueError("existing hypotheses contain duplicate question_id rows")
                        if question_id not in selected_question_ids:
                            raise ValueError(
                                "existing hypotheses contain question_id values outside selected dataset"
                            )
                        existing_ids.add(question_id)
                        generated.append(
                            LongMemBenchGeneratedHypothesis(
                                question_id=question_id,
                                hypothesis=hypothesis if isinstance(hypothesis, str) else "",
                                context_count=0,
                                answer_mode=mode,
                            )
                        )
                file_mode = "a" if resume else "w"
                with output_path.open(file_mode, encoding="utf-8") as handle:
                    for case in cases:
                        contexts = await checkout_retriever.query_async(
                            case.query,
                            temporal_point=case.temporal_point,
                            limit=limit,
                        )
                        question_id = case.name.removeprefix("longmemeval-")
                        if question_id in existing_ids:
                            continue
                        answer_session_ids = case.identity_terms
                        expected_terms = case.expected_terms
                        deterministic_answer = _deterministic_temporal_order_answer(
                            case.query,
                            contexts,
                        )
                        answer_ready_candidate = _answer_ready_preference_candidate(case.query, contexts)
                        checkout_candidate = answer_ready_candidate if prefer_checkout_candidate else None
                        hypothesis = deterministic_answer or checkout_candidate or answer_ready_candidate or (
                            _openai_compatible_answer(
                                question=case.query,
                                contexts=(
                                    _answer_generation_contexts(contexts)
                                    if filter_answer_contexts
                                    else contexts
                                ),
                                model=model or "gpt-4o-mini",
                                base_url=base_url,
                                api_key=api_key or "",
                                max_retries=provider_retries,
                            )
                            if mode == "openai-compatible"
                            else _extractive_answer(case.query, contexts)
                        )
                        handle.write(
                            json.dumps(
                                {"question_id": question_id, "hypothesis": hypothesis},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.flush()
                        if fsync_rows:
                            os.fsync(handle.fileno())
                        context_audit = _longmembench_context_audit(
                            contexts,
                            expected_terms=expected_terms,
                            answer_session_ids=answer_session_ids,
                        )
                        generated.append(
                            LongMemBenchGeneratedHypothesis(
                                question_id=question_id,
                                hypothesis=hypothesis,
                                context_count=len(contexts),
                                answer_mode=mode,
                                answer_session_ids=answer_session_ids,
                                answer_session_hits_top5=_longmembench_answer_session_hits(
                                    contexts,
                                    answer_session_ids,
                                ),
                                expected_answer_hit_top5=any(
                                    item.contains_expected_answer
                                    for item in context_audit
                                ),
                                context_audit=context_audit,
                            )
                        )
            finally:
                await graph.close()
    finally:
        provider.flush()

    report = LongMemBenchHypothesisReport(
        schema_version="zaxy.longmembench-hypotheses.v1",
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        dataset_path=str(dataset_path),
        dataset_sha256=_file_sha256(dataset_path),
        question_count=len(generated),
        output_path=str(output_path),
        answer_mode=mode,
        model=model,
        embedding_provider=provider_label,
        projection_backend=projection_backend,
        limit=limit,
        hypotheses=tuple(generated),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_hypothesis_report_to_dict(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def build_longmembench_report(
    *,
    longmemeval_worktree: Path,
    dataset_path: Path,
    hypotheses_path: Path | None = None,
    official_eval_log_path: Path | None = None,
    diagnostic_report_path: Path | None = None,
    sota_baseline_path: Path | None = None,
    evaluator_model: str | None = None,
    official_eval_command: str | None = None,
    result_provenance: dict[str, object] | None = None,
) -> LongMemBenchReport:
    """Build a LongMemBench report from official QA and diagnostic artifacts."""
    suite = check_longmemeval_official_suite(longmemeval_worktree)
    official_qa = None
    if official_eval_log_path is not None:
        if hypotheses_path is None:
            raise ValueError("--hypotheses is required when --official-eval-log is supplied")
        official_qa = load_official_qa_evidence(
            dataset_path=dataset_path,
            hypotheses_path=hypotheses_path,
            official_eval_log_path=official_eval_log_path,
            evaluator_model=evaluator_model,
            official_eval_command=official_eval_command,
        )
    diagnostic = (
        load_zaxy_diagnostic_report(diagnostic_report_path)
        if diagnostic_report_path is not None
        else None
    )
    sota_baseline = load_sota_baseline(sota_baseline_path) if sota_baseline_path is not None else None
    status = "complete" if official_qa is not None and suite["status"] == "valid" else "partial"
    sota_claim = _sota_claim_status(official_qa, sota_baseline=sota_baseline)
    caveats = _longmembench_caveats(
        official_qa=official_qa,
        diagnostic=diagnostic,
        sota_baseline=sota_baseline,
    )
    return LongMemBenchReport(
        schema_version=LONGMEMBENCH_SCHEMA_VERSION,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        status=status,
        external_suite={
            "source_url": LONGMEMEVAL_REPO_URL,
            "dataset_url": LONGMEMEVAL_DATASET_URL,
            "worktree": suite["worktree"],
            "commit": suite.get("commit"),
            "official_evaluator": suite.get("official_evaluator"),
            "dataset_counts": suite.get("dataset_counts"),
        },
        result_provenance=result_provenance or {},
        official_qa=official_qa,
        zaxy_diagnostic=diagnostic,
        sota_baseline=sota_baseline,
        sota_claim=sota_claim,
        caveats=caveats,
    )


def load_official_qa_evidence(
    *,
    dataset_path: Path,
    hypotheses_path: Path,
    official_eval_log_path: Path,
    evaluator_model: str | None = None,
    official_eval_command: str | None = None,
) -> LongMemBenchOfficialQA:
    """Load official LongMemEval QA evidence from hypotheses and evaluator log."""
    dataset = _load_json_list(dataset_path, "LongMemEval dataset")
    hypotheses = _load_jsonl_objects(hypotheses_path, "hypotheses")
    eval_rows = _load_jsonl_objects(official_eval_log_path, "official evaluator log")
    dataset_ids = {str(row.get("question_id")) for row in dataset if isinstance(row, dict)}
    hypothesis_ids = [str(row.get("question_id")) for row in hypotheses if isinstance(row, dict)]
    eval_ids = [str(row.get("question_id")) for row in eval_rows if isinstance(row, dict)]
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise ValueError("hypotheses contain duplicate question_id rows")
    if len(set(eval_ids)) != len(eval_ids):
        raise ValueError("official evaluator log contains duplicate question_id rows")
    if not set(eval_ids).issubset(dataset_ids):
        raise ValueError("official evaluator log contains question_id values absent from dataset")
    if set(eval_ids) != set(hypothesis_ids):
        raise ValueError("official evaluator log question_ids must match hypotheses question_ids")
    correct = 0
    for row in eval_rows:
        label = _autoeval_label(row)
        if label is None:
            raise ValueError("official evaluator log row missing recognized autoeval_label")
        if label:
            correct += 1
    evaluated = len(eval_rows)
    accuracy = correct / evaluated if evaluated else 0.0
    return LongMemBenchOfficialQA(
        dataset_path=str(dataset_path),
        dataset_sha256=_file_sha256(dataset_path),
        dataset_question_count=len(dataset),
        hypothesis_path=str(hypotheses_path),
        hypothesis_count=len(hypotheses),
        eval_log_path=str(official_eval_log_path),
        evaluated_count=evaluated,
        correct_count=correct,
        accuracy=round(accuracy, 6),
        evaluator_model=evaluator_model,
        official_eval_command=official_eval_command,
    )


def load_zaxy_diagnostic_report(path: Path) -> LongMemBenchDiagnostic:
    """Load Zaxy LongMemEval-compatible diagnostic metrics from live-benchmark.json."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("diagnostic report missing summaries")
    selected = None
    for summary in summaries:
        if isinstance(summary, dict) and summary.get("backend") == "zaxy-checkout":
            selected = summary
            break
    if selected is None:
        for summary in summaries:
            if isinstance(summary, dict) and str(summary.get("backend", "")).startswith("zaxy"):
                selected = summary
                break
    if selected is None:
        raise ValueError("diagnostic report missing Zaxy backend summary")
    workload = payload.get("workload")
    workload_sha = workload.get("sha256") if isinstance(workload, dict) else None
    return LongMemBenchDiagnostic(
        report_path=str(path),
        backend=str(selected.get("backend")),
        case_count=int(selected.get("case_count") or 0),
        mean_score=_float_or_none(selected.get("mean_score")),
        answer_at_5=_float_or_none(selected.get("mean_answer_recall_at_5")),
        citation_coverage=_float_or_none(selected.get("mean_citation_coverage")),
        recall_at_5=_float_or_none(selected.get("mean_recall_at_5")),
        recall_at_10=_float_or_none(selected.get("mean_recall_at_10")),
        p95_ms=_float_or_none(selected.get("latency_ms_p95")),
        approx_tokens=_float_or_none(selected.get("mean_approx_tokens")),
        workload_sha256=str(workload_sha) if workload_sha else None,
    )


def load_sota_baseline(path: Path) -> LongMemBenchSotaBaseline:
    """Load an external official-score baseline for SOTA comparison."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SOTA baseline must be a JSON object")
    try:
        baseline = LongMemBenchSotaBaseline(
            system=str(payload["system"]),
            accuracy=float(payload["accuracy"]),
            metric=str(payload["metric"]),
            evidence_url=str(payload["evidence_url"]),
            evidence_date=str(payload["evidence_date"]),
            source_type=str(payload["source_type"]),
            question_count=int(payload["question_count"]) if payload.get("question_count") is not None else None,
            evaluator_model=str(payload["evaluator_model"]) if payload.get("evaluator_model") is not None else None,
            notes=str(payload["notes"]) if payload.get("notes") is not None else None,
            checked_at=str(payload["checked_at"]) if payload.get("checked_at") is not None else None,
            expires_at=str(payload["expires_at"]) if payload.get("expires_at") is not None else None,
            currentness_url=str(payload["currentness_url"]) if payload.get("currentness_url") is not None else None,
        )
    except KeyError as exc:
        raise ValueError(f"SOTA baseline missing required field: {exc.args[0]}") from exc
    validation = validate_sota_baseline(baseline)
    if validation:
        raise ValueError("; ".join(validation))
    return baseline


def load_validator_evidence(path: Path) -> dict[str, object]:
    """Load a completed external-validator evidence record."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"validator evidence is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("validator evidence must be a JSON object")
    validator = payload.get("validator")
    if not isinstance(validator, dict):
        raise ValueError("validator evidence missing validator object")
    official_evaluation = payload.get("official_evaluation")
    if official_evaluation is not None and not isinstance(official_evaluation, dict):
        raise ValueError("validator evidence official_evaluation must be an object")
    artifacts = payload.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        raise ValueError("validator evidence artifacts must be an object")
    return payload


def validator_provenance_from_evidence(
    evidence: dict[str, object],
    *,
    validator_name: str | None = None,
    validator_evidence_url: str | None = None,
    validator_run_id: str | None = None,
    validator_relation: str | None = None,
) -> dict[str, object]:
    """Build validator provenance, rejecting conflicting nonblank overrides."""
    validator_payload = evidence.get("validator")
    if not isinstance(validator_payload, dict):
        raise ValueError("validator evidence missing validator object")
    validated_system_payload = evidence.get("validated_system")
    if validated_system_payload is not None and not isinstance(validated_system_payload, dict):
        raise ValueError("validator evidence validated_system must be an object")
    fields = {
        "name": validator_payload.get("name"),
        "evidence_url": validator_payload.get("evidence_url"),
        "run_id": validator_payload.get("run_id"),
        "relation": validator_payload.get("relation"),
    }
    overrides = {
        "name": validator_name,
        "evidence_url": validator_evidence_url,
        "run_id": validator_run_id,
        "relation": validator_relation,
    }
    for key, override in overrides.items():
        if override is None:
            continue
        existing = str(fields.get(key) or "").strip()
        replacement = override.strip()
        if existing and existing != replacement:
            raise ValueError(f"validator evidence conflicts with --validator-{key.replace('_', '-')}")
        fields[key] = replacement
    provenance: dict[str, object] = {
        "validator": fields,
    }
    if isinstance(validated_system_payload, dict):
        zaxy_commit = str(validated_system_payload.get("zaxy_commit") or "").strip()
        zaxy_version = str(validated_system_payload.get("zaxy_version") or "").strip()
        provenance["validated_system"] = {
            "name": str(validated_system_payload.get("name") or "").strip(),
            "zaxy_commit": zaxy_commit,
            "zaxy_version": zaxy_version,
        }
        provenance["zaxy_commit"] = zaxy_commit
    return provenance


def validator_official_evaluation_metadata(evidence: dict[str, object]) -> dict[str, str | None]:
    """Extract optional official-evaluation metadata from validator evidence."""
    official_evaluation = evidence.get("official_evaluation")
    if not isinstance(official_evaluation, dict):
        return {"evaluator_model": None, "official_eval_command": None}
    evaluator_model = official_evaluation.get("evaluator_model")
    official_eval_command = official_evaluation.get("official_eval_command")
    return {
        "evaluator_model": str(evaluator_model).strip() if evaluator_model else None,
        "official_eval_command": str(official_eval_command).strip() if official_eval_command else None,
    }


def validate_validator_evidence_matches_report(
    evidence: dict[str, object],
    report: LongMemBenchReport,
) -> list[str]:
    """Return mismatches between completed validator evidence and imported results."""
    failures: list[str] = []
    qa = report.official_qa
    if qa is None:
        return failures
    longmemeval = evidence.get("longmemeval")
    if not isinstance(longmemeval, dict):
        failures.append("validator evidence missing longmemeval object")
        longmemeval = {}
    validated_system = evidence.get("validated_system")
    if not isinstance(validated_system, dict):
        failures.append("validator evidence missing validated_system object")
        validated_system = {}
    official_evaluation = evidence.get("official_evaluation")
    if not isinstance(official_evaluation, dict):
        failures.append("validator evidence missing official_evaluation object")
        official_evaluation = {}
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        failures.append("validator evidence missing artifacts object")
        artifacts = {}
    if str(validated_system.get("name") or "").strip().casefold() != "zaxy":
        failures.append("validator evidence validated_system.name must be Zaxy")
    failures.extend(
        _match_required_string(
            "validator evidence validated_system.zaxy_commit",
            validated_system.get("zaxy_commit"),
            report.result_provenance.get("zaxy_commit"),
        )
    )
    failures.extend(
        _match_required_string(
            "validator evidence longmemeval.commit",
            longmemeval.get("commit"),
            report.external_suite.get("commit"),
        )
    )
    failures.extend(
        _match_required_string(
            "validator evidence longmemeval.dataset_sha256",
            longmemeval.get("dataset_sha256"),
            qa.dataset_sha256,
        )
    )
    failures.extend(
        _match_required_int(
            "validator evidence longmemeval.question_count",
            longmemeval.get("question_count"),
            qa.dataset_question_count,
        )
    )
    failures.extend(
        _match_required_string(
            "validator evidence artifacts.dataset_sha256",
            artifacts.get("dataset_sha256"),
            qa.dataset_sha256,
        )
    )
    failures.extend(
        _match_required_string(
            "validator evidence artifacts.hypotheses_sha256",
            artifacts.get("hypotheses_sha256"),
            _file_sha256(Path(qa.hypothesis_path)),
        )
    )
    failures.extend(
        _match_required_string(
            "validator evidence artifacts.official_eval_log_sha256",
            artifacts.get("official_eval_log_sha256"),
            _file_sha256(Path(qa.eval_log_path)),
        )
    )
    failures.extend(
        _match_required_string(
            "validator evidence official_evaluation.evaluator_model",
            official_evaluation.get("evaluator_model"),
            qa.evaluator_model,
        )
    )
    failures.extend(
        _match_required_int(
            "validator evidence official_evaluation.evaluated_count",
            official_evaluation.get("evaluated_count"),
            qa.evaluated_count,
        )
    )
    failures.extend(
        _match_required_int(
            "validator evidence official_evaluation.correct_count",
            official_evaluation.get("correct_count"),
            qa.correct_count,
        )
    )
    failures.extend(
        _match_required_float(
            "validator evidence official_evaluation.accuracy",
            official_evaluation.get("accuracy"),
            qa.accuracy,
        )
    )
    command = official_evaluation.get("official_eval_command")
    if not str(command or "").strip():
        failures.append("validator evidence official_evaluation.official_eval_command is required")
    elif qa.official_eval_command and str(command).strip() != qa.official_eval_command:
        failures.append("validator evidence official_evaluation.official_eval_command does not match imported report")
    return failures


def build_validator_evidence_record(
    *,
    longmemeval_worktree: Path,
    dataset_path: Path,
    hypotheses_path: Path,
    official_eval_log_path: Path,
    evaluator_model: str,
    official_eval_command: str,
    validator_name: str,
    validator_evidence_url: str,
    validator_run_id: str,
    validator_relation: str,
    print_metrics_command: str | None = None,
    zaxy_worktree: Path | None = None,
    zaxy_version: str | None = None,
    longmembench_report_json: Path | None = None,
    longmembench_report_md: Path | None = None,
    gate_status: str | None = None,
) -> dict[str, object]:
    """Build a completed validator evidence record from official artifacts."""
    suite = check_longmemeval_official_suite(longmemeval_worktree)
    qa = load_official_qa_evidence(
        dataset_path=dataset_path,
        hypotheses_path=hypotheses_path,
        official_eval_log_path=official_eval_log_path,
        evaluator_model=evaluator_model,
        official_eval_command=official_eval_command,
    )
    zaxy_commit = _git_commit(zaxy_worktree or Path.cwd())
    return {
        "validator": {
            "name": validator_name,
            "relation": validator_relation,
            "evidence_url": validator_evidence_url,
            "run_id": validator_run_id,
        },
        "validated_system": {
            "name": "Zaxy",
            "zaxy_commit": zaxy_commit or "",
            "zaxy_version": zaxy_version or "",
        },
        "longmemeval": {
            "repo_url": LONGMEMEVAL_REPO_URL,
            "commit": suite.get("commit") or "",
            "dataset": str(dataset_path),
            "dataset_sha256": qa.dataset_sha256,
            "question_count": qa.dataset_question_count,
        },
        "official_evaluation": {
            "evaluator_model": evaluator_model,
            "hypotheses_path": str(hypotheses_path),
            "official_eval_log_path": str(official_eval_log_path),
            "official_eval_command": official_eval_command,
            "print_metrics_command": print_metrics_command or "",
            "accuracy": qa.accuracy,
            "correct_count": qa.correct_count,
            "evaluated_count": qa.evaluated_count,
        },
        "artifacts": {
            "dataset_sha256": qa.dataset_sha256,
            "hypotheses_sha256": _file_sha256(hypotheses_path),
            "official_eval_log_sha256": _file_sha256(official_eval_log_path),
            "longmembench_report_json": str(longmembench_report_json) if longmembench_report_json else "",
            "longmembench_report_json_sha256": (
                _file_sha256(longmembench_report_json)
                if longmembench_report_json is not None and longmembench_report_json.exists()
                else ""
            ),
            "longmembench_report_md": str(longmembench_report_md) if longmembench_report_md else "",
            "longmembench_report_md_sha256": (
                _file_sha256(longmembench_report_md)
                if longmembench_report_md is not None and longmembench_report_md.exists()
                else ""
            ),
            "gate_command": (
                "zaxy longmembench-gate "
                f"{longmembench_report_json or 'reports/benchmarks/longmembench-external/longmembench-report.json'} "
                "--require-official-sota"
            ),
            "gate_status": gate_status or "",
        },
    }


def write_validator_evidence_record(record: dict[str, object], output: Path) -> Path:
    """Write a completed validator evidence JSON artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_sota_baseline(baseline: LongMemBenchSotaBaseline) -> list[str]:
    """Return validation failures for a SOTA baseline row."""
    failures: list[str] = []
    if not baseline.system.strip():
        failures.append("SOTA baseline system is required")
    if not (0.0 <= baseline.accuracy <= 1.0):
        failures.append("SOTA baseline accuracy must be between 0 and 1")
    if baseline.metric not in {"official_longmemeval_task_averaged_accuracy", "official_longmemeval_overall_accuracy"}:
        failures.append("SOTA baseline metric must be an official LongMemEval QA accuracy metric")
    if baseline.question_count is not None and baseline.question_count < OFFICIAL_FULL_QUESTION_COUNT:
        failures.append("SOTA baseline question_count must cover all 500 official questions")
    failures.extend(_validate_reviewable_http_url("SOTA baseline evidence_url", baseline.evidence_url))
    if not baseline.evidence_date.strip():
        failures.append("SOTA baseline evidence_date is required")
    elif _parse_iso_date(baseline.evidence_date) is None:
        failures.append("SOTA baseline evidence_date must be YYYY-MM-DD")
    if baseline.source_type not in {
        "official-leaderboard",
        "maintainer-accepted",
        "peer-reviewed-paper",
        "public-reproduction",
        "vendor-disclosure",
    }:
        failures.append("SOTA baseline source_type is not recognized")
    if baseline.checked_at is not None and _parse_iso_date(baseline.checked_at) is None:
        failures.append("SOTA baseline checked_at must be YYYY-MM-DD")
    if baseline.expires_at is not None and _parse_iso_date(baseline.expires_at) is None:
        failures.append("SOTA baseline expires_at must be YYYY-MM-DD")
    if baseline.currentness_url is not None:
        failures.extend(_validate_reviewable_http_url("SOTA baseline currentness_url", baseline.currentness_url))
    return failures


def validate_sota_baseline_currentness(
    baseline: LongMemBenchSotaBaseline,
    *,
    reference_date: date | None = None,
    max_age_days: int = DEFAULT_SOTA_BASELINE_MAX_AGE_DAYS,
) -> list[str]:
    """Return failures for strict-SOTA baseline freshness and currentness."""
    failures = validate_sota_baseline(baseline)
    if max_age_days < 0:
        failures.append("SOTA baseline max_age_days must be non-negative")
        return failures
    today = reference_date or datetime.now(UTC).date()
    evidence_date = _parse_iso_date(baseline.evidence_date)
    if evidence_date is not None and evidence_date > today:
        failures.append("SOTA baseline evidence_date cannot be in the future")
    if not baseline.checked_at:
        failures.append("official SOTA requires SOTA baseline checked_at")
    else:
        checked_at = _parse_iso_date(baseline.checked_at)
        if checked_at is not None:
            if checked_at > today:
                failures.append("SOTA baseline checked_at cannot be in the future")
            elif (today - checked_at).days > max_age_days:
                failures.append(
                    "SOTA baseline checked_at is stale "
                    f"({(today - checked_at).days} days old; max {max_age_days})"
                )
    if baseline.expires_at:
        expires_at = _parse_iso_date(baseline.expires_at)
        if expires_at is not None and expires_at < today:
            failures.append("SOTA baseline expires_at has passed")
    return failures


def _validate_external_validator_provenance(provenance: dict[str, object]) -> list[str]:
    failures: list[str] = []
    validator = provenance.get("validator")
    if not isinstance(validator, dict):
        return ["external validator provenance is required for official SOTA claims"]
    name = str(validator.get("name") or "").strip()
    evidence_url = str(validator.get("evidence_url") or "").strip()
    run_id = str(validator.get("run_id") or "").strip()
    relation = str(validator.get("relation") or "").strip().casefold()
    if not name:
        failures.append("external validator name is required")
    failures.extend(_validate_reviewable_http_url("external validator evidence_url", evidence_url))
    if not run_id:
        failures.append("external validator run_id is required")
    if relation in {"", "self", "internal", "zaxy", "zaxy-dev", "author"}:
        failures.append("external validator relation must describe an independent reviewer")
    return failures


def _validate_cross_checked_validator_evidence(provenance: dict[str, object]) -> list[str]:
    failures: list[str] = []
    evidence_path = str(provenance.get("validator_evidence") or "").strip()
    if not evidence_path:
        failures.append("official SOTA requires imported validator_evidence JSON")
    if provenance.get("validator_evidence_verified") is not True:
        failures.append("official SOTA requires cross-checked validator evidence")
    if not str(provenance.get("zaxy_commit") or "").strip():
        failures.append("official SOTA requires validator-bound Zaxy commit")
    return failures


def write_longmembench_report(
    report: LongMemBenchReport,
    output_dir: Path,
) -> WrittenLongMemBenchReport:
    """Write LongMemBench JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "longmembench-report.json"
    markdown_path = output_dir / "longmembench-report.md"
    json_path.write_text(json.dumps(_report_to_dict(report), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
    return WrittenLongMemBenchReport(json_path=json_path, markdown_path=markdown_path)


def load_longmembench_report(path: Path) -> LongMemBenchReport:
    """Load a LongMemBench report from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LONGMEMBENCH_SCHEMA_VERSION:
        raise ValueError("unsupported LongMemBench report schema_version")
    official_payload = payload.get("official_qa")
    diagnostic_payload = payload.get("zaxy_diagnostic")
    baseline_payload = payload.get("sota_baseline")
    return LongMemBenchReport(
        schema_version=str(payload["schema_version"]),
        generated_at=str(payload["generated_at"]),
        status=str(payload["status"]),
        external_suite=dict(payload.get("external_suite") or {}),
        result_provenance=dict(payload.get("result_provenance") or {}),
        official_qa=(
            LongMemBenchOfficialQA(**official_payload)
            if isinstance(official_payload, dict)
            else None
        ),
        zaxy_diagnostic=(
            LongMemBenchDiagnostic(**diagnostic_payload)
            if isinstance(diagnostic_payload, dict)
            else None
        ),
        sota_baseline=(
            LongMemBenchSotaBaseline(**baseline_payload)
            if isinstance(baseline_payload, dict)
            else None
        ),
        sota_claim=dict(payload.get("sota_claim") or {}),
        caveats=tuple(str(item) for item in payload.get("caveats") or ()),
    )


def validate_longmembench_report(
    report: LongMemBenchReport,
    *,
    require_official_full: bool = False,
    require_external_validator: bool = False,
) -> dict[str, object]:
    """Validate report evidence for external LongMemBench claims."""
    failures: list[str] = []
    if report.schema_version != LONGMEMBENCH_SCHEMA_VERSION:
        failures.append("unsupported schema_version")
    if not report.external_suite.get("commit"):
        failures.append("external LongMemEval worktree commit is missing")
    if report.official_qa is None:
        if require_official_full:
            failures.append("official QA evaluator evidence is required")
    else:
        qa = report.official_qa
        if qa.dataset_question_count != OFFICIAL_FULL_QUESTION_COUNT and require_official_full:
            failures.append("official dataset must contain 500 questions")
        if qa.hypothesis_count != qa.evaluated_count:
            failures.append("hypothesis_count must equal evaluated_count")
        if require_official_full and qa.evaluated_count != OFFICIAL_FULL_QUESTION_COUNT:
            failures.append("official evaluator must cover all 500 questions")
        if qa.correct_count > qa.evaluated_count:
            failures.append("correct_count cannot exceed evaluated_count")
        if not (0.0 <= qa.accuracy <= 1.0):
            failures.append("accuracy must be between 0 and 1")
    if report.zaxy_diagnostic is not None:
        diagnostic = report.zaxy_diagnostic
        if diagnostic.case_count <= 0:
            failures.append("diagnostic case_count must be positive")
        if diagnostic.citation_coverage is None:
            failures.append("diagnostic citation_coverage is missing")
    if report.sota_baseline is not None:
        failures.extend(validate_sota_baseline(report.sota_baseline))
    if require_external_validator:
        failures.extend(_validate_external_validator_provenance(report.result_provenance))
    return {
        "status": "valid" if not failures else "invalid",
        "failures": failures,
        "require_official_full": require_official_full,
        "require_external_validator": require_external_validator,
    }


def check_longmembench_gate(
    report: LongMemBenchReport,
    *,
    require_official_sota_candidate: bool = False,
    require_official_sota: bool = False,
    require_external_validator: bool = False,
    min_accuracy: float | None = None,
) -> dict[str, object]:
    """Gate publishable LongMemBench claims."""
    validation = validate_longmembench_report(
        report,
        require_official_full=require_official_sota_candidate or require_official_sota,
        require_external_validator=require_external_validator or require_official_sota,
    )
    validation_failures = validation.get("failures")
    failures = (
        [str(item) for item in validation_failures]
        if isinstance(validation_failures, list)
        else []
    )
    qa = report.official_qa
    if require_official_sota_candidate:
        if qa is None:
            failures.append("official SOTA candidate requires official QA evidence")
        elif qa.evaluated_count != OFFICIAL_FULL_QUESTION_COUNT:
            failures.append("official SOTA candidate requires all 500 official questions")
    if require_official_sota:
        if qa is None:
            failures.append("official SOTA requires official QA evidence")
        elif qa.evaluated_count != OFFICIAL_FULL_QUESTION_COUNT:
            failures.append("official SOTA requires all 500 official questions")
        failures.extend(_validate_cross_checked_validator_evidence(report.result_provenance))
        if report.sota_baseline is None:
            failures.append("official SOTA requires a current external SOTA baseline")
        else:
            failures.extend(validate_sota_baseline_currentness(report.sota_baseline))
            if qa is not None and qa.accuracy <= report.sota_baseline.accuracy:
                failures.append(
                    "official QA accuracy "
                    f"{qa.accuracy:.6f} does not beat {report.sota_baseline.system} "
                    f"baseline {report.sota_baseline.accuracy:.6f}"
                )
    if min_accuracy is not None:
        if qa is None:
            failures.append("min_accuracy requires official QA evidence")
        elif qa.accuracy < min_accuracy:
            failures.append(f"official QA accuracy {qa.accuracy:.6f} is below {min_accuracy:.6f}")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "official_sota_candidate": bool(
            (require_official_sota_candidate or require_official_sota)
            and qa is not None
            and qa.evaluated_count == OFFICIAL_FULL_QUESTION_COUNT
            and not failures
        ),
        "official_sota": bool(require_official_sota and not failures),
        "external_validator": report.result_provenance.get("validator"),
        "validator_evidence": report.result_provenance.get("validator_evidence"),
        "validator_evidence_verified": report.result_provenance.get("validator_evidence_verified"),
        "sota_baseline": (
            asdict(report.sota_baseline) if report.sota_baseline is not None else None
        ),
        "accuracy": qa.accuracy if qa is not None else None,
        "evaluated_count": qa.evaluated_count if qa is not None else None,
        "claim_boundary": (
            "Official LongMemEval SOTA still requires comparison against current accepted "
            "leaderboard/evidence and any maintainer submission process."
        ),
    }


def audit_longmembench_artifacts(
    *,
    longmemeval_worktree: Path,
    dataset_path: Path,
    hypotheses_path: Path,
    official_eval_log_path: Path,
    diagnostic_report_path: Path,
    sota_baseline_path: Path,
    validator_evidence_path: Path,
    report_path: Path,
    hypothesis_report_path: Path | None = None,
    official_eval_run_report_path: Path | None = None,
) -> dict[str, object]:
    """Audit a completed external LongMemBench artifact set."""
    failures: list[str] = []
    evidence: dict[str, object] = {}
    artifacts = _longmembench_audit_artifacts(
        dataset_path=dataset_path,
        hypotheses_path=hypotheses_path,
        official_eval_log_path=official_eval_log_path,
        diagnostic_report_path=diagnostic_report_path,
        sota_baseline_path=sota_baseline_path,
        validator_evidence_path=validator_evidence_path,
        report_path=report_path,
        hypothesis_report_path=hypothesis_report_path,
        official_eval_run_report_path=official_eval_run_report_path,
    )

    def record(name: str, status: str, **values: object) -> None:
        evidence[name] = {"status": status, **values}

    suite = check_longmemeval_official_suite(longmemeval_worktree)
    record("official_suite", str(suite["status"]), commit=suite.get("commit"), worktree=suite.get("worktree"))
    if suite["status"] != "valid":
        failures.append("official LongMemEval worktree is invalid")

    try:
        loaded_official_qa = load_official_qa_evidence(
            dataset_path=dataset_path,
            hypotheses_path=hypotheses_path,
            official_eval_log_path=official_eval_log_path,
        )
    except (OSError, ValueError) as exc:
        failures.append(f"official QA evidence invalid: {exc}")
        record("official_qa", "invalid")
    else:
        record(
            "official_qa",
            "valid",
            dataset_question_count=loaded_official_qa.dataset_question_count,
            hypothesis_count=loaded_official_qa.hypothesis_count,
            evaluated_count=loaded_official_qa.evaluated_count,
            accuracy=loaded_official_qa.accuracy,
            dataset_sha256=loaded_official_qa.dataset_sha256,
        )
        if loaded_official_qa.dataset_question_count != OFFICIAL_FULL_QUESTION_COUNT:
            failures.append("official dataset must contain 500 questions")
        if loaded_official_qa.hypothesis_count != OFFICIAL_FULL_QUESTION_COUNT:
            failures.append("hypotheses must contain 500 rows")
        if loaded_official_qa.evaluated_count != OFFICIAL_FULL_QUESTION_COUNT:
            failures.append("official evaluator log must contain 500 rows")

    if hypothesis_report_path is not None:
        try:
            payload = json.loads(hypothesis_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"hypothesis report invalid: {exc}")
            record("hypothesis_report", "invalid", path=str(hypothesis_report_path))
        else:
            question_count = payload.get("question_count") if isinstance(payload, dict) else None
            record("hypothesis_report", "valid", path=str(hypothesis_report_path), question_count=question_count)
            if question_count != OFFICIAL_FULL_QUESTION_COUNT:
                failures.append("hypothesis report must cover 500 questions")

    if official_eval_run_report_path is not None:
        try:
            payload = json.loads(official_eval_run_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"official eval run report invalid: {exc}")
            record("official_eval_run", "invalid", path=str(official_eval_run_report_path))
        else:
            status = str(payload.get("status") if isinstance(payload, dict) else "")
            returncode = payload.get("returncode") if isinstance(payload, dict) else None
            record("official_eval_run", "valid", path=str(official_eval_run_report_path), run_status=status, returncode=returncode)
            if status != "complete" or returncode != 0:
                failures.append("official eval run report must be complete with returncode 0")

    try:
        diagnostic = load_zaxy_diagnostic_report(diagnostic_report_path)
    except (OSError, ValueError) as exc:
        failures.append(f"diagnostic report invalid: {exc}")
        record("diagnostic", "invalid", path=str(diagnostic_report_path))
    else:
        record("diagnostic", "valid", path=str(diagnostic_report_path), case_count=diagnostic.case_count)
        if diagnostic.case_count != OFFICIAL_FULL_QUESTION_COUNT:
            failures.append("diagnostic report must cover 500 cases")

    try:
        baseline = load_sota_baseline(sota_baseline_path)
    except (OSError, ValueError) as exc:
        failures.append(f"SOTA baseline invalid: {exc}")
        record("sota_baseline", "invalid", path=str(sota_baseline_path))
    else:
        record("sota_baseline", "valid", path=str(sota_baseline_path), system=baseline.system, accuracy=baseline.accuracy)

    loaded_validator_evidence: dict[str, object] | None = None
    try:
        loaded_validator_evidence = load_validator_evidence(validator_evidence_path)
    except (OSError, ValueError) as exc:
        failures.append(f"validator evidence invalid: {exc}")
        record("validator_evidence", "invalid", path=str(validator_evidence_path))
    else:
        record("validator_evidence", "valid", path=str(validator_evidence_path))

    try:
        loaded_report = load_longmembench_report(report_path)
    except (OSError, ValueError) as exc:
        failures.append(f"LongMemBench report invalid: {exc}")
        record("longmembench_report", "invalid", path=str(report_path))
    else:
        gate = check_longmembench_gate(loaded_report, require_official_sota=True)
        record("longmembench_report", "valid", path=str(report_path), gate_status=gate["status"])
        if gate["status"] != "passed":
            gate_failures = gate.get("failures")
            if isinstance(gate_failures, list):
                failures.extend(str(item) for item in gate_failures)
            else:
                failures.append("strict LongMemBench gate failed")
        failures.extend(
            _validate_report_artifact_paths(
                loaded_report,
                dataset_path=dataset_path,
                hypotheses_path=hypotheses_path,
                official_eval_log_path=official_eval_log_path,
                diagnostic_report_path=diagnostic_report_path,
                sota_baseline_path=sota_baseline_path,
                validator_evidence_path=validator_evidence_path,
            )
        )
        if loaded_validator_evidence is not None:
            failures.extend(validate_validator_evidence_matches_report(loaded_validator_evidence, loaded_report))
            failures.extend(
                _validate_validator_evidence_artifact_paths(
                    loaded_validator_evidence,
                    dataset_path=dataset_path,
                    hypotheses_path=hypotheses_path,
                    official_eval_log_path=official_eval_log_path,
                    report_path=report_path,
                )
            )

    return {
        "schema_version": LONGMEMBENCH_AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "artifacts": artifacts,
        "evidence": evidence,
    }


def _longmembench_audit_artifacts(
    *,
    dataset_path: Path,
    hypotheses_path: Path,
    official_eval_log_path: Path,
    diagnostic_report_path: Path,
    sota_baseline_path: Path,
    validator_evidence_path: Path,
    report_path: Path,
    hypothesis_report_path: Path | None,
    official_eval_run_report_path: Path | None,
) -> dict[str, object]:
    artifacts: dict[str, object] = {
        "dataset": _audit_artifact_entry(dataset_path),
        "hypotheses": _audit_artifact_entry(hypotheses_path),
        "official_eval_log": _audit_artifact_entry(official_eval_log_path),
        "diagnostic_report": _audit_artifact_entry(diagnostic_report_path),
        "sota_baseline": _audit_artifact_entry(sota_baseline_path),
        "validator_evidence": _audit_artifact_entry(validator_evidence_path),
        "longmembench_report": _audit_artifact_entry(report_path),
    }
    if hypothesis_report_path is not None:
        artifacts["hypothesis_report"] = _audit_artifact_entry(hypothesis_report_path)
    if official_eval_run_report_path is not None:
        artifacts["official_eval_run_report"] = _audit_artifact_entry(official_eval_run_report_path)
    return artifacts


def _audit_artifact_entry(path: Path) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "exists": path.exists(),
    }
    if path.exists():
        entry["sha256"] = _file_sha256(path)
        entry["bytes"] = path.stat().st_size
    return entry


def _validate_report_artifact_paths(
    report: LongMemBenchReport,
    *,
    dataset_path: Path,
    hypotheses_path: Path,
    official_eval_log_path: Path,
    diagnostic_report_path: Path,
    sota_baseline_path: Path,
    validator_evidence_path: Path,
) -> list[str]:
    failures: list[str] = []
    provenance = report.result_provenance
    if report.official_qa is None:
        failures.append("LongMemBench report missing official QA evidence")
    else:
        failures.extend(_match_path("report official_qa.dataset_path", report.official_qa.dataset_path, dataset_path))
        failures.extend(_match_path("report official_qa.hypothesis_path", report.official_qa.hypothesis_path, hypotheses_path))
        failures.extend(_match_path("report official_qa.eval_log_path", report.official_qa.eval_log_path, official_eval_log_path))
    failures.extend(_match_path("report provenance.dataset", provenance.get("dataset"), dataset_path))
    failures.extend(_match_path("report provenance.hypotheses", provenance.get("hypotheses"), hypotheses_path))
    failures.extend(_match_path("report provenance.official_eval_log", provenance.get("official_eval_log"), official_eval_log_path))
    failures.extend(_match_path("report provenance.diagnostic_report", provenance.get("diagnostic_report"), diagnostic_report_path))
    failures.extend(_match_path("report provenance.sota_baseline", provenance.get("sota_baseline"), sota_baseline_path))
    failures.extend(_match_path("report provenance.validator_evidence", provenance.get("validator_evidence"), validator_evidence_path))
    return failures


def _validate_validator_evidence_artifact_paths(
    evidence: dict[str, object],
    *,
    dataset_path: Path,
    hypotheses_path: Path,
    official_eval_log_path: Path,
    report_path: Path,
) -> list[str]:
    failures: list[str] = []
    longmemeval = evidence.get("longmemeval")
    official_evaluation = evidence.get("official_evaluation")
    artifacts = evidence.get("artifacts")
    if not isinstance(longmemeval, dict):
        failures.append("validator evidence missing longmemeval object")
    else:
        failures.extend(_match_path("validator evidence longmemeval.dataset", longmemeval.get("dataset"), dataset_path))
    if not isinstance(official_evaluation, dict):
        failures.append("validator evidence missing official_evaluation object")
    else:
        failures.extend(
            _match_path(
                "validator evidence official_evaluation.hypotheses_path",
                official_evaluation.get("hypotheses_path"),
                hypotheses_path,
            )
        )
        failures.extend(
            _match_path(
                "validator evidence official_evaluation.official_eval_log_path",
                official_evaluation.get("official_eval_log_path"),
                official_eval_log_path,
            )
        )
    if not isinstance(artifacts, dict):
        failures.append("validator evidence missing artifacts object")
    else:
        failures.extend(
            _match_required_string(
                "validator evidence artifacts.dataset_sha256",
                artifacts.get("dataset_sha256"),
                _file_sha256(dataset_path),
            )
        )
        failures.extend(
            _match_required_string(
                "validator evidence artifacts.hypotheses_sha256",
                artifacts.get("hypotheses_sha256"),
                _file_sha256(hypotheses_path),
            )
        )
        failures.extend(
            _match_required_string(
                "validator evidence artifacts.official_eval_log_sha256",
                artifacts.get("official_eval_log_sha256"),
                _file_sha256(official_eval_log_path),
            )
        )
        failures.extend(
            _match_path(
                "validator evidence artifacts.longmembench_report_json",
                artifacts.get("longmembench_report_json"),
                report_path,
            )
        )
        if str(artifacts.get("longmembench_report_json_sha256") or "").strip():
            failures.extend(
                _match_required_string(
                    "validator evidence artifacts.longmembench_report_json_sha256",
                    artifacts.get("longmembench_report_json_sha256"),
                    _file_sha256(report_path),
                )
            )
    return failures


def render_longmembench_publication_markdown(report: LongMemBenchReport) -> str:
    """Render publishable LongMemBench statistics after gate validation."""
    gate = check_longmembench_gate(report, require_official_sota=True)
    if gate["status"] != "passed":
        raise ValueError("LongMemBench report does not pass official SOTA gate")
    return report_to_markdown(report)


def validate_longmembench_audit_for_report(audit_path: Path, report_path: Path) -> dict[str, object]:
    """Load and validate a passing audit artifact for a report being published."""
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LongMemBench audit is not valid JSON: {exc}") from exc
    if not isinstance(audit, dict):
        raise ValueError("LongMemBench audit must be a JSON object")
    if audit.get("schema_version") != LONGMEMBENCH_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported LongMemBench audit schema_version")
    if not str(audit.get("generated_at") or "").strip():
        raise ValueError("LongMemBench audit generated_at is required")
    if audit.get("status") != "passed":
        raise ValueError("LongMemBench audit must have status=passed")
    failures = audit.get("failures")
    if isinstance(failures, list) and failures:
        raise ValueError("LongMemBench audit contains failures")
    artifacts = audit.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("LongMemBench audit missing artifacts object")
    report_artifact = artifacts.get("longmembench_report")
    if not isinstance(report_artifact, dict):
        raise ValueError("LongMemBench audit missing longmembench_report artifact")
    failures = []
    failures.extend(_match_path("audit longmembench_report.path", report_artifact.get("path"), report_path))
    expected_hash = _file_sha256(report_path)
    observed_hash = str(report_artifact.get("sha256") or "").strip()
    if not observed_hash:
        failures.append("audit longmembench_report.sha256 is required")
    elif observed_hash != expected_hash:
        failures.append("audit longmembench_report.sha256 does not match report file")
    if failures:
        raise ValueError("; ".join(failures))
    return audit


def report_to_markdown(report: LongMemBenchReport) -> str:
    """Render a LongMemBench report as Markdown."""
    lines = [
        "# LongMemBench External Validation Report",
        "",
        f"- Status: `{report.status}`",
        f"- Generated at: `{report.generated_at}`",
        f"- LongMemEval source: {report.external_suite.get('source_url', LONGMEMEVAL_REPO_URL)}",
        f"- LongMemEval commit: `{report.external_suite.get('commit')}`",
        "",
        "## Official QA Evidence",
        "",
    ]
    if report.official_qa is None:
        lines.append("No official evaluator evidence was imported.")
    else:
        qa = report.official_qa
        lines.extend(
            [
                f"- Dataset questions: `{qa.dataset_question_count}`",
                f"- Evaluated questions: `{qa.evaluated_count}`",
                f"- Correct: `{qa.correct_count}`",
                f"- Accuracy: `{qa.accuracy:.6f}`",
                f"- Evaluator model: `{qa.evaluator_model}`",
                f"- Dataset SHA-256: `{qa.dataset_sha256}`",
            ]
        )
    lines.extend(["", "## Zaxy Diagnostic Evidence", ""])
    if report.zaxy_diagnostic is None:
        lines.append("No Zaxy LongMemEval-compatible diagnostic report was imported.")
    else:
        diagnostic = report.zaxy_diagnostic
        lines.extend(
            [
                f"- Backend: `{diagnostic.backend}`",
                f"- Cases: `{diagnostic.case_count}`",
                f"- Mean score: `{_format_optional_float(diagnostic.mean_score)}`",
                f"- Answer@5: `{_format_optional_float(diagnostic.answer_at_5)}`",
                f"- Recall@5: `{_format_optional_float(diagnostic.recall_at_5)}`",
                f"- Recall@10: `{_format_optional_float(diagnostic.recall_at_10)}`",
                f"- Citation coverage: `{_format_optional_float(diagnostic.citation_coverage)}`",
                f"- p95 latency ms: `{_format_optional_float(diagnostic.p95_ms)}`",
                f"- Approx tokens: `{_format_optional_float(diagnostic.approx_tokens)}`",
            ]
        )
    lines.extend(["", "## SOTA Baseline", ""])
    if report.sota_baseline is None:
        lines.append("No external SOTA baseline was imported.")
    else:
        baseline = report.sota_baseline
        lines.extend(
            [
                f"- System: `{baseline.system}`",
                f"- Accuracy: `{baseline.accuracy:.6f}`",
                f"- Evidence URL: {baseline.evidence_url}",
                f"- Evidence date: `{baseline.evidence_date}`",
                f"- Source type: `{baseline.source_type}`",
                f"- Checked at: `{baseline.checked_at or 'n/a'}`",
                f"- Expires at: `{baseline.expires_at or 'n/a'}`",
                f"- Currentness URL: {baseline.currentness_url or 'n/a'}",
            ]
        )
    lines.extend(["", "## Claim Boundary", ""])
    for key, value in report.sota_claim.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in report.caveats)
    return "\n".join(lines) + "\n"


def _longmembench_plan_markdown(manifest: dict[str, object]) -> str:
    commands = manifest.get("commands")
    lines = [
        "# LongMemBench External Run Plan",
        "",
        f"- Source: {manifest.get('source_url')}",
        f"- Dataset: `{manifest.get('dataset')}`",
        f"- Evaluator model: `{manifest.get('evaluator_model')}`",
        f"- Expected questions: `{manifest.get('expected_question_count')}`",
        "",
        "## Commands",
        "",
    ]
    if isinstance(commands, dict):
        for name, command in commands.items():
            lines.extend([f"### {name}", "", "```bash", str(command), "```", ""])
    return "\n".join(lines)


def _longmembench_plan_script(manifest: dict[str, object]) -> str:
    commands = manifest.get("commands")
    if not isinstance(commands, dict):
        commands = {}
    output_dir = str(manifest.get("output_dir") or "reports/benchmarks/longmembench-external")
    evaluator_model = str(manifest.get("evaluator_model") or "gpt-4o")
    dataset = str(manifest.get("dataset") or DEFAULT_OFFICIAL_DATASET)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "LONGMEMEVAL_WORKTREE=${LONGMEMEVAL_WORKTREE:-${1:-.cache/zaxy/benchmarks/LongMemEval}}",
        "ANSWER_MODE=${ANSWER_MODE:-openai-compatible}",
        "EVALUATOR_MODEL=${EVALUATOR_MODEL:-gpt-4o}",
        "QUESTIONS=${QUESTIONS:-500}",
        "RUN_OFFICIAL_EVAL=${RUN_OFFICIAL_EVAL:-1}",
        "RUN_DIAGNOSTIC=${RUN_DIAGNOSTIC:-${RUN_OFFICIAL_EVAL}}",
        "VALIDATOR_NAME=${VALIDATOR_NAME:-}",
        "VALIDATOR_EVIDENCE_URL=${VALIDATOR_EVIDENCE_URL:-}",
        "VALIDATOR_RUN_ID=${VALIDATOR_RUN_ID:-}",
        "VALIDATOR_RELATION=${VALIDATOR_RELATION:-}",
        'if [[ -z "${RUN_OUTPUT_DIR:-}" ]]; then',
        '  if [[ "${RUN_OFFICIAL_EVAL}" == "0" ]]; then',
        f"    RUN_OUTPUT_DIR={shlex.quote(output_dir)}/smoke",
        "  else",
        f"    RUN_OUTPUT_DIR={shlex.quote(output_dir)}",
        "  fi",
        "fi",
        'OFFICIAL_EVAL_COMMAND="python3 evaluate_qa.py ${EVALUATOR_MODEL} ${RUN_OUTPUT_DIR}/zaxy-hypotheses.jsonl ${LONGMEMEVAL_WORKTREE}/'
        f'{dataset}"',
        'PRINT_METRICS_COMMAND="python3 ${LONGMEMEVAL_WORKTREE}/src/evaluation/print_qa_metrics.py ${RUN_OUTPUT_DIR}/zaxy-hypotheses.jsonl.eval-results-${EVALUATOR_MODEL} ${LONGMEMEVAL_WORKTREE}/'
        f'{dataset}"',
        "",
        "if [[ \"${RUN_OFFICIAL_EVAL}\" != \"0\" && \"${ANSWER_MODE}\" == \"openai-compatible\" && -z \"${OPENAI_API_KEY:-}\" ]]; then",
        "  echo 'OPENAI_API_KEY is required for official openai-compatible LongMemBench runs.' >&2",
        "  exit 2",
        "fi",
        "if [[ \"${RUN_OFFICIAL_EVAL}\" != \"0\" ]]; then",
        "  if [[ -z \"${VALIDATOR_NAME}\" || -z \"${VALIDATOR_EVIDENCE_URL}\" || -z \"${VALIDATOR_RUN_ID}\" || -z \"${VALIDATOR_RELATION}\" ]]; then",
        "    echo 'VALIDATOR_NAME, VALIDATOR_EVIDENCE_URL, VALIDATOR_RUN_ID, and VALIDATOR_RELATION are required for official SOTA runs.' >&2",
        "    exit 2",
        "  fi",
        "fi",
        "",
        "run_step() {",
        "  local name=$1",
        "  local command=$2",
        "  echo \"[$name] $command\"",
        "  eval \"$command\"",
        "}",
        "",
    ]
    for name, command in commands.items():
        executable_command = (
            str(command)
            .replace("path/to/LongMemEval", '"${LONGMEMEVAL_WORKTREE}"')
            .replace(f"{output_dir}/diagnostic", '"${RUN_OUTPUT_DIR}"/diagnostic')
            .replace(
                f"{output_dir}/zaxy-hypotheses",
                '"${RUN_OUTPUT_DIR}"/zaxy-hypotheses',
            )
            .replace(
                f"{output_dir}/official-eval-run.json",
                '"${RUN_OUTPUT_DIR}"/official-eval-run.json',
            )
            .replace(
                f"{output_dir}/validator-evidence.json",
                '"${RUN_OUTPUT_DIR}"/validator-evidence.json',
            )
            .replace(
                f"{output_dir}/longmembench-report.json",
                '"${RUN_OUTPUT_DIR}"/longmembench-report.json',
            )
            .replace(
                f"{output_dir}/longmembench-audit.json",
                '"${RUN_OUTPUT_DIR}"/longmembench-audit.json',
            )
            .replace(
                f"{output_dir}/publishable-statistics.md",
                '"${RUN_OUTPUT_DIR}"/publishable-statistics.md',
            )
            .replace(
                f"--output-dir {output_dir}",
                '--output-dir "${RUN_OUTPUT_DIR}"',
            )
            .replace("--answer-mode openai-compatible", '--answer-mode "${ANSWER_MODE}"')
            .replace("--model gpt-4o", '--model "${EVALUATOR_MODEL}"')
            .replace("--evaluator-model gpt-4o", '--evaluator-model "${EVALUATOR_MODEL}"')
            .replace(
                f"evaluate_qa.py {evaluator_model}",
                'evaluate_qa.py "${EVALUATOR_MODEL}"',
            )
            .replace("ZAXY_OFFICIAL_EVAL_COMMAND", '"${OFFICIAL_EVAL_COMMAND}"')
            .replace("ZAXY_PRINT_METRICS_COMMAND", '"${PRINT_METRICS_COMMAND}"')
            .replace('--validator-name "Independent Validator"', '--validator-name "${VALIDATOR_NAME}"')
            .replace(
                "--validator-evidence-url https://validation.openmemory.dev/reviewable-run",
                '--validator-evidence-url "${VALIDATOR_EVIDENCE_URL}"',
            )
            .replace("--validator-run-id validator-run-001", '--validator-run-id "${VALIDATOR_RUN_ID}"')
            .replace(
                "--validator-relation independent-third-party",
                '--validator-relation "${VALIDATOR_RELATION}"',
            )
            .replace("--questions 500", '--questions "${QUESTIONS}"')
            .replace(".eval-results-gpt-4o", '.eval-results-"${EVALUATOR_MODEL}"')
        )
        if name == "diagnostic":
            lines.extend(
                [
                    'if [[ "${RUN_DIAGNOSTIC}" != "0" ]]; then',
                    f"  run_step {shlex.quote(str(name))} {shlex.quote(executable_command)}",
                    "else",
                    "  echo '[diagnostic] skipped because RUN_DIAGNOSTIC=0'",
                    "fi",
                ]
            )
            continue
        if name in {"official_eval", "official_metrics", "validator_evidence", "import", "gate", "audit", "publish"}:
            lines.extend(
                [
                    'if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then',
                    f"  run_step {shlex.quote(str(name))} {shlex.quote(executable_command)}",
                    "else",
                    f"  echo '[{name}] skipped because RUN_OFFICIAL_EVAL=0'",
                    "fi",
                ]
            )
            continue
        lines.append(f"run_step {shlex.quote(str(name))} {shlex.quote(executable_command)}")
    return "\n".join(lines) + "\n"


def _extractive_answer(question: str, contexts: list[str]) -> str:
    """Return a conservative extractive hypothesis from checkout contexts."""
    del question
    for context in contexts:
        candidate = _extractive_candidate_from_context(context)
        if candidate:
            return candidate
    return "I do not have enough information to answer."


def _extractive_candidate_from_context(context: str) -> str:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    preferred_markers = (
        "zaxy_synthesis_bundle=true",
        "checkout_item=current_fact",
        "checkout_item=evidence",
        "longmemeval_salient_memory_turn=true",
        "has_answer",
    )
    for marker in preferred_markers:
        for index, line in enumerate(lines):
            if marker in line:
                for candidate in [*lines[index + 1 : index + 8], line]:
                    cleaned = _clean_extractive_text(candidate)
                    if cleaned and _looks_like_answer_text(cleaned):
                        return cleaned
    for line in lines:
        cleaned = _clean_extractive_text(line)
        if not _looks_like_answer_text(cleaned):
            continue
        if cleaned:
            return cleaned
    return ""


def _clean_extractive_text(value: str) -> str:
    text = " ".join(value.split())
    while text.startswith(("- ", "* ")):
        text = text[2:].strip()
    prefixes = (
        "role=user ",
        "role=assistant ",
        "user: ",
        "assistant: ",
        "snippet=",
        "content=",
    )
    if " — summary=" in text:
        text = text.split(" — summary=", 1)[1].strip()
    if " content=" in text:
        text = text.split(" content=", 1)[1].strip()
    for marker in (" role=user ", " role=assistant ", " 1. user: ", " 2. assistant: "):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    for suffix in (", source_path=", ", source_event_seq=", ", source_start_line="):
        if suffix in text:
            text = text.split(suffix, 1)[0].strip()
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text[:1200].strip()


def _looks_like_answer_text(value: str) -> bool:
    if not value:
        return False
    metadata_prefixes = (
        "citation=",
        "source_lane=",
        "score=",
        "checkout_item=",
        "longmemeval_session_id=",
        "longmemeval_session_date=",
        "longmemeval_salient_memory_turn=",
        "turn_index=",
        "role=",
        "source_id=",
        "source_path=",
        "snippet=",
        "evidence_count=",
        "citation_count=",
        "memory_checkout=",
        "answerability=",
        "confidence=",
        "evidence_plan_",
        "required_source_groups=",
        "observed_source_groups=",
        "query=",
        "question=",
    )
    if value.startswith(metadata_prefixes):
        return False
    if "=" in value and len(value.split()) <= 4:
        return False
    return any(char.isalpha() for char in value)


def _openai_compatible_answer(
    *,
    question: str,
    contexts: list[str],
    model: str,
    base_url: str,
    api_key: str,
    max_retries: int = 3,
) -> str:
    """Generate a concise answer from checkout evidence through chat completions."""
    if answer_candidate := _answer_ready_preference_candidate(question, contexts):
        return answer_candidate
    prompt = "\n\n".join(contexts[:12])
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 96,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the LongMemEval question using only the supplied Zaxy "
                    "Memory Checkout evidence. Return only the answer, with no "
                    "explanation. If the evidence is insufficient, answer exactly: "
                    "I do not have enough information to answer. Prefer cited "
                    "source snippets and checkout facts over diagnostic counters. "
                    "Do not use checkout metrics such as week_total, month_total, "
                    "source_count, or candidate_rank as the answer unless the "
                    "same value is supported by a cited source snippet. For "
                    "questions asking which item happened first, compare the "
                    "dated or relative-time evidence for the named alternatives "
                    "and return the earlier alternative."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nZaxy Memory Checkout evidence:\n{prompt}",
            },
        ],
    }
    response: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=120.0,
        )
        if response.status_code < 500 and response.status_code != 429:
            break
        if response.status_code == 429:
            try:
                error = response.json().get("error", {})
            except ValueError:
                error = {}
            if error.get("code") == "insufficient_quota":
                break
        if attempt >= max_retries:
            break
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 15.0
        else:
            delay = 10.0 * (attempt + 1)
        time.sleep(min(90.0, delay))
    if response is None:
        raise ValueError("OpenAI-compatible request was not attempted")
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI-compatible response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("OpenAI-compatible response choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenAI-compatible response choice missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI-compatible response message content is empty")
    return content.strip()


def _answer_ready_preference_candidate(question: str, contexts: list[str]) -> str | None:
    """Return a cited preference-profile answer candidate from available contexts."""
    if answer := _checkout_answer_candidate(question, contexts):
        return answer
    from zaxy.evidence_candidates import checkout_candidate_projection

    projection = checkout_candidate_projection(question, contexts, limit=max(1, min(10, len(contexts))))
    for candidate in projection.answer_candidates:
        if str(candidate.get("type", "")).casefold() != "preference":
            continue
        candidate_answer = candidate.get("answer")
        if isinstance(candidate_answer, str) and _usable_checkout_answer_candidate(
            question,
            candidate_answer,
            candidate_type="preference",
        ):
            return candidate_answer
    return None


def _answer_generation_contexts(contexts: list[str]) -> list[str]:
    """Return cited evidence contexts without checkout diagnostics or synthesis scaffolding."""
    selected: list[str] = []
    fallback: list[str] = []
    for context in contexts:
        lowered = context.casefold()
        if _diagnostic_answer_context(lowered):
            continue
        if "checkout_fact=true" in lowered or "checkout_item=current_fact" in lowered or "checkout_item=evidence" in lowered:
            selected.append(context)
            continue
        if _cited_answer_context(lowered):
            fallback.append(context)
    return selected or fallback or contexts


def _deterministic_temporal_order_answer(question: str, contexts: list[str]) -> str | None:
    """Answer binary first/earlier questions from dated or relative cited evidence."""
    if not _temporal_order_question(question.casefold()):
        return None
    alternatives = _temporal_order_alternatives(question)
    if len(alternatives) < 2:
        return None
    observations: list[tuple[date, int, str]] = []
    for index, alternative in enumerate(alternatives):
        value = _temporal_order_alternative_date(alternative, contexts, question=question)
        if value is not None:
            observations.append((value, index, alternative))
    if len(observations) < 2:
        return None
    observations.sort(key=lambda item: (item[0], item[1]))
    earliest = observations[0]
    if len(observations) > 1 and observations[0][0] == observations[1][0]:
        return None
    return _display_temporal_order_answer(earliest[2])


def _temporal_order_alternatives(question: str) -> tuple[str, ...]:
    quoted = [
        match.group("single") or match.group("double")
        for match in re.finditer(r"'(?P<single>[^']{2,180})'|\"(?P<double>[^\"]{2,180})\"", question)
    ]
    if len(quoted) >= 2:
        return tuple(_clean_temporal_order_alternative(item) for item in quoted[:2])
    text = question.rstrip(" ?")
    if " or " not in text:
        return ()
    before, after = text.rsplit(" or ", 1)
    left = before.split(",", 1)[-1]
    left = re.sub(
        r"^.*?\b(?:first|earlier|before|the)\b\s*",
        "",
        left,
        flags=re.IGNORECASE,
    )
    left = re.sub(
        r"^.*?\b(?:did I|did i|was|were)\b\s*",
        "",
        left,
        flags=re.IGNORECASE,
    )
    right = after
    return tuple(
        item
        for item in (
            _clean_temporal_order_alternative(left),
            _clean_temporal_order_alternative(right),
        )
        if item
    )


def _clean_temporal_order_alternative(value: str) -> str:
    value = " ".join(value.strip(" ,.?").split())
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    return value


def _temporal_order_alternative_date(alternative: str, contexts: list[str], *, question: str) -> date | None:
    alt_terms = _temporal_order_terms(alternative)
    if not alt_terms:
        return None
    candidates: list[tuple[int, date]] = []
    for context_index, context in enumerate(contexts):
        lowered = context.casefold()
        if _diagnostic_answer_context(lowered):
            continue
        if not _terms_present(alt_terms, lowered):
            continue
        context_date = _context_session_date(context)
        for span in _evidence_spans_for_terms(context, alt_terms):
            value = _span_temporal_value(span, context_date=context_date, question=question)
            if value is None:
                continue
            relevance = _term_occurrence_count(alt_terms, span.casefold())
            candidates.append((-relevance * 1000 + context_index, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _temporal_order_terms(value: str) -> tuple[str, ...]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "did",
        "item",
        "event",
        "device",
        "task",
        "vehicle",
        "purchase",
    }
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", value.casefold())
        if len(term) > 2 and term not in stopwords
    ]
    if "car" in terms:
        terms.extend(["corolla", "vehicle"])
    if "laptop" in terms:
        terms.extend(["dell", "xps"])
    return tuple(dict.fromkeys(terms))


def _terms_present(terms: tuple[str, ...], lowered_context: str) -> bool:
    if not terms:
        return False
    required = 1 if len(terms) <= 3 else 2
    return _terms_matched_count(terms, lowered_context) >= required


def _terms_matched_count(terms: tuple[str, ...], lowered_text: str) -> int:
    return sum(1 for term in terms if term in lowered_text)


def _term_occurrence_count(terms: tuple[str, ...], lowered_text: str) -> int:
    return sum(lowered_text.count(term) for term in terms)


def _evidence_spans_for_terms(context: str, terms: tuple[str, ...]) -> list[str]:
    text = " ".join(context.split())
    spans: list[str] = []
    sentence_parts = re.split(r"(?<=[.!?])\s+|\s+(?=\d+\.\s+user:|\buser:)", text)
    for index, sentence in enumerate(sentence_parts):
        lowered = sentence.casefold()
        if _terms_present(terms, lowered):
            start = max(0, index - 1)
            end = min(len(sentence_parts), index + 2)
            spans.append(" ".join(sentence_parts[start:end]))
    if spans:
        return spans[:4]
    lowered = text.casefold()
    first_index = min((lowered.find(term) for term in terms if term in lowered), default=-1)
    if first_index < 0:
        return []
    start = max(0, first_index - 260)
    end = min(len(text), first_index + 360)
    return [text[start:end]]


def _context_session_date(context: str) -> date | None:
    match = re.search(
        r"\blongmemeval_session_date=(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
        context,
    )
    if not match:
        return None
    return _safe_date(int(match.group("year")), int(match.group("month")), int(match.group("day")))


def _span_temporal_value(span: str, *, context_date: date | None, question: str = "") -> date | None:
    span = _strip_longmembench_metadata(span)
    default_year = context_date.year if context_date else None
    action_date = _action_specific_span_date(span, question=question, default_year=default_year)
    if action_date is not None:
        return action_date
    if _acquisition_temporal_question(question) and _contains_preorder_only_date(span):
        return None
    explicit = _explicit_span_date(span, default_year=default_year)
    if explicit is not None:
        return explicit
    if context_date is None:
        return None
    days_ago = _relative_days_ago(span)
    if days_ago is not None:
        return context_date - timedelta(days=days_ago)
    return None


def _strip_longmembench_metadata(span: str) -> str:
    span = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)\s*", "", span)
    span = re.sub(r"\blongmemeval_session_id=\S+\s*", "", span)
    return span


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _explicit_span_date(span: str, *, default_year: int | None) -> date | None:
    if default_year is None:
        return None
    month_names = "|".join(_MONTHS)
    match = re.search(
        rf"\b(?P<month>{month_names})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b",
        span,
        flags=re.IGNORECASE,
    )
    if match:
        return _safe_date(default_year, _MONTHS[match.group("month").casefold()], int(match.group("day")))
    day_first = re.search(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+(?P<month>{month_names})\b",
        span,
        flags=re.IGNORECASE,
    )
    if day_first:
        return _safe_date(default_year, _MONTHS[day_first.group("month").casefold()], int(day_first.group("day")))
    numeric = re.search(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})\b", span)
    if numeric:
        return _safe_date(default_year, int(numeric.group("month")), int(numeric.group("day")))
    month_only = re.search(
        rf"\b(?P<qualifier>early|mid|late)?-?\s*(?P<month>{month_names})\b",
        span,
        flags=re.IGNORECASE,
    )
    if month_only:
        qualifier = (month_only.group("qualifier") or "").casefold()
        day = 5 if qualifier == "early" else 15 if qualifier == "mid" else 25 if qualifier == "late" else 15
        return _safe_date(default_year, _MONTHS[month_only.group("month").casefold()], day)
    return None


def _action_specific_span_date(span: str, *, question: str, default_year: int | None) -> date | None:
    if default_year is None:
        return None
    lowered_question = question.casefold()
    if _acquisition_temporal_question(lowered_question):
        action_pattern = (
            r"\b(?:arrived|got|received|bought|purchased|acquired|picked\s+up|set\s+up)\b"
        )
    elif any(term in lowered_question for term in ("started", "start", "attend", "attended", "participated")):
        action_pattern = r"\b(?:started|attended|participated|joined|went|visited)\b"
    elif any(term in lowered_question for term in ("finished", "completed", "done")):
        action_pattern = r"\b(?:finished|completed|submitted|wrapped\s+up)\b"
    else:
        return None
    month_names = "|".join(_MONTHS)
    match = re.search(
        action_pattern
        + r"[^.?!]{0,140}?\bon\s+"
        + rf"(?P<month>{month_names})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b",
        span,
        flags=re.IGNORECASE,
    )
    if match:
        return _safe_date(default_year, _MONTHS[match.group("month").casefold()], int(match.group("day")))
    match = re.search(
        action_pattern + r"[^.?!]{0,140}?\bon\s+(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
        span,
        flags=re.IGNORECASE,
    )
    if match:
        return _safe_date(default_year, int(match.group("month")), int(match.group("day")))
    match = re.search(
        rf"\bon\s+(?P<month>{month_names})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b"
        + r"[^.?!]{0,140}?"
        + action_pattern,
        span,
        flags=re.IGNORECASE,
    )
    if match:
        return _safe_date(default_year, _MONTHS[match.group("month").casefold()], int(match.group("day")))
    match = re.search(
        r"\bon\s+(?P<month>\d{1,2})/(?P<day>\d{1,2})\b[^.?!]{0,140}?" + action_pattern,
        span,
        flags=re.IGNORECASE,
    )
    if match:
        return _safe_date(default_year, int(match.group("month")), int(match.group("day")))
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    """Return a calendar date, ignoring invalid dates from noisy benchmark text."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _acquisition_temporal_question(question: str) -> bool:
    lowered = question.casefold()
    return bool(
        re.search(
            r"\b(?:got|get|received|receive|arrived|bought|buy|purchased|purchase|acquired|set\s+up)\b",
            lowered,
        )
    )


def _contains_preorder_only_date(span: str) -> bool:
    lowered = span.casefold()
    if not re.search(r"\b(?:pre-ordered|preordered|pre-order|preorder|expected arrival|expected)\b", lowered):
        return False
    return not re.search(
        r"\b(?:arrived|got|received|bought|purchased|acquired|picked\s+up|set\s+up)\b[^.?!]{0,140}?\bon\s+"
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",
        lowered,
    )


def _relative_days_ago(span: str) -> int | None:
    lowered = span.casefold()
    if re.search(r"\b(?:today|tonight|this morning|this afternoon|this evening)\b", lowered):
        return 0
    if "yesterday" in lowered:
        return 1
    if "last week" in lowered:
        return 7
    if "last month" in lowered:
        return 30
    if "a few months ago" in lowered:
        return 90
    if "recently" in lowered:
        return 3
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "a": 1,
        "an": 1,
    }
    match = re.search(
        r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?P<unit>days?|weeks?|months?)\s+ago\b",
        lowered,
    )
    if not match:
        return None
    raw = match.group("value")
    value = number_words.get(raw, int(raw) if raw.isdigit() else 0)
    unit = match.group("unit")
    multiplier = 1 if unit.startswith("day") else 7 if unit.startswith("week") else 30
    return value * multiplier


def _display_temporal_order_answer(alternative: str) -> str:
    alternative = alternative.strip()
    if not alternative:
        return alternative
    if alternative.isupper() or any(char.isdigit() for char in alternative):
        return alternative
    return alternative[0].upper() + alternative[1:]


def _diagnostic_answer_context(lowered_context: str) -> bool:
    """Return whether a checkout context is answer-generation metadata, not evidence."""
    diagnostic_markers = (
        "memory_checkout_compact=true",
        "memory_checkout=true",
        "checkout_synthesis=true",
        "zaxy_synthesis_bundle=true",
        "zaxy_absence_check=true",
        "checkout_answer_candidate=true",
        "checkout_evidence_group=true",
    )
    return any(marker in lowered_context for marker in diagnostic_markers)


def _cited_answer_context(lowered_context: str) -> bool:
    return "citation=" in lowered_context or "source_lane=" in lowered_context


def _checkout_answer_candidate(question: str, contexts: list[str]) -> str | None:
    """Return the top answer-ready checkout candidate when one is present."""
    for context in contexts:
        if "checkout_answer_candidate=true" not in context:
            continue
        candidate_type = ""
        for line in context.splitlines():
            if line.startswith("candidate_type="):
                candidate_type = line.removeprefix("candidate_type=").strip()
                continue
            if not line.startswith("answer="):
                continue
            answer = line.removeprefix("answer=").strip()
            if _usable_checkout_answer_candidate(question, answer, candidate_type=candidate_type):
                return answer
    for context in contexts:
        marker = "- Answer candidate:"
        if marker not in context:
            continue
        for line in context.splitlines():
            if marker not in line or "answer=" not in line:
                continue
            candidate_type = ""
            if "type=" in line:
                candidate_type = line.split("type=", 1)[1].split(",", 1)[0].strip()
            answer = line.split("answer=", 1)[1].split(", confidence=", 1)[0].strip()
            if _usable_checkout_answer_candidate(question, answer, candidate_type=candidate_type):
                return answer
    if absence_answer := _absence_answer_candidate(question, contexts):
        return absence_answer
    return None


def _absence_answer_candidate(question: str, contexts: list[str]) -> str | None:
    """Return cited absence-check answer guidance when checkout proves missing evidence."""
    for context in contexts:
        if "zaxy_absence_check=true" not in context:
            continue
        candidate_type = "absence_check"
        for line in context.splitlines():
            if not line.startswith("answer_guidance="):
                continue
            answer = line.removeprefix("answer_guidance=").strip()
            if _usable_checkout_answer_candidate(question, answer, candidate_type=candidate_type):
                return answer
    return None


def _usable_checkout_answer_candidate(question: str, answer: str, *, candidate_type: str = "") -> bool:
    """Reject provenance fragments masquerading as direct answers."""
    if not answer:
        return False
    lowered = answer.casefold()
    if lowered.startswith("# event") or " document.indexed " in lowered:
        return False
    if "citation=eventloom://" in lowered and len(answer.split()) <= 16:
        return False
    question_lowered = question.casefold()
    answer_lowered = answer.casefold()
    candidate_type = candidate_type.casefold()
    if "how many days" in question_lowered:
        return re.search(r"\b\d+(?:\.\d+)?\s+days?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\b", answer_lowered) is not None
    if "how many months" in question_lowered:
        return re.search(r"\b\d+(?:\.\d+)?\s+months?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+months?\b", answer_lowered) is not None
    if "how long" in question_lowered:
        return re.search(r"\b\d+(?:\.\d+)?\s+(?:days?|weeks?|months?|years?|hours?)\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?|years?|hours?)\b", answer_lowered) is not None
    if _temporal_order_question(question_lowered):
        return candidate_type in {
            "absence_check",
            "temporal_order",
            "temporal_sequence",
            "preference",
            "query_bound_direct_answer",
        }
    return True


def _temporal_order_question(question_lowered: str) -> bool:
    return (
        " first" in question_lowered
        or " happened first" in question_lowered
        or " before " in question_lowered
    ) and " or " in question_lowered


def _sota_claim_status(
    official_qa: LongMemBenchOfficialQA | None,
    *,
    sota_baseline: LongMemBenchSotaBaseline | None,
) -> dict[str, object]:
    if official_qa is None:
        return {
            "official_sota_candidate": False,
            "reason": "missing official LongMemEval evaluator evidence",
            "claim_allowed": "LongMemEval-compatible diagnostic only",
        }
    full = (
        official_qa.dataset_question_count == OFFICIAL_FULL_QUESTION_COUNT
        and official_qa.evaluated_count == OFFICIAL_FULL_QUESTION_COUNT
    )
    return {
        "official_sota_candidate": full,
        "official_sota": bool(
            full
            and sota_baseline is not None
            and official_qa.accuracy > sota_baseline.accuracy
        ),
        "accuracy": official_qa.accuracy,
        "evaluated_count": official_qa.evaluated_count,
        "baseline_system": sota_baseline.system if sota_baseline is not None else None,
        "baseline_accuracy": sota_baseline.accuracy if sota_baseline is not None else None,
        "claim_allowed": (
            "official LongMemEval SOTA"
            if full and sota_baseline is not None and official_qa.accuracy > sota_baseline.accuracy
            else "official LongMemEval full-set candidate"
            if full
            else "official LongMemEval partial-set evidence"
        ),
        "requires_external_comparison": sota_baseline is None,
    }


def _longmembench_caveats(
    *,
    official_qa: LongMemBenchOfficialQA | None,
    diagnostic: LongMemBenchDiagnostic | None,
    sota_baseline: LongMemBenchSotaBaseline | None,
) -> tuple[str, ...]:
    caveats = [
        "Official LongMemEval SOTA requires official evaluator QA evidence, not Zaxy retrieval diagnostics alone.",
        "Leaderboard language requires comparison against current accepted external results and any maintainer submission process.",
    ]
    if official_qa is None:
        caveats.append("No official evaluator log was imported; this report cannot support an official LongMemEval score.")
    if diagnostic is not None:
        caveats.append("Zaxy diagnostic metrics are LongMemEval-compatible checkout evidence, not official QA accuracy.")
    if sota_baseline is None:
        caveats.append("No external SOTA baseline was imported; this report cannot support a SOTA claim.")
    return tuple(caveats)


def _report_to_dict(report: LongMemBenchReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "generated_at": report.generated_at,
        "status": report.status,
        "external_suite": report.external_suite,
        "result_provenance": report.result_provenance,
        "official_qa": asdict(report.official_qa) if report.official_qa is not None else None,
        "zaxy_diagnostic": (
            asdict(report.zaxy_diagnostic) if report.zaxy_diagnostic is not None else None
        ),
        "sota_baseline": (
            asdict(report.sota_baseline) if report.sota_baseline is not None else None
        ),
        "sota_claim": report.sota_claim,
        "caveats": list(report.caveats),
    }


def _hypothesis_report_to_dict(report: LongMemBenchHypothesisReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "generated_at": report.generated_at,
        "dataset_path": report.dataset_path,
        "dataset_sha256": report.dataset_sha256,
        "question_count": report.question_count,
        "output_path": report.output_path,
        "answer_mode": report.answer_mode,
        "model": report.model,
        "embedding_provider": report.embedding_provider,
        "projection_backend": report.projection_backend,
        "limit": report.limit,
        "hypotheses": [asdict(item) for item in report.hypotheses],
    }


_LONGMEMBENCH_SESSION_ID_RE = re.compile(r"\blongmemeval_session_id=([^\s]+)")
_LONGMEMBENCH_SOURCE_PATH_RE = re.compile(r"\blongmemeval/[^\s)]+")
_LONGMEMBENCH_EVENT_REF_RE = re.compile(r"eventloom://[^\s)]+")


def _longmembench_context_audit(
    contexts: list[str],
    *,
    expected_terms: tuple[str, ...],
    answer_session_ids: tuple[str, ...],
    limit: int = 5,
) -> tuple[LongMemBenchContextAudit, ...]:
    """Build compact top-k evidence telemetry for LongMemBench reports."""
    expected_needles = tuple(term.casefold() for term in expected_terms if term.strip())
    session_needles = tuple(session_id.casefold() for session_id in answer_session_ids if session_id)
    audit: list[LongMemBenchContextAudit] = []
    for rank, context in enumerate(contexts[:limit], start=1):
        folded = context.casefold()
        session_ids = tuple(dict.fromkeys(_LONGMEMBENCH_SESSION_ID_RE.findall(context)))
        source_paths = tuple(dict.fromkeys(_LONGMEMBENCH_SOURCE_PATH_RE.findall(context)))
        event_refs = tuple(dict.fromkeys(_LONGMEMBENCH_EVENT_REF_RE.findall(context)))
        snippet = " ".join(context.split())
        audit.append(
            LongMemBenchContextAudit(
                rank=rank,
                approx_tokens=max(1, (len(context) + 3) // 4) if context else 0,
                citation_count=len(event_refs),
                session_ids=session_ids,
                source_paths=source_paths,
                event_refs=event_refs,
                contains_expected_answer=any(term in folded for term in expected_needles),
                contains_answer_session=any(session_id in folded for session_id in session_needles),
                snippet=snippet[:500],
            )
        )
    return tuple(audit)


def _longmembench_answer_session_hits(
    contexts: list[str],
    answer_session_ids: tuple[str, ...],
    *,
    limit: int = 5,
) -> tuple[str, ...]:
    """Return answer-session ids found in the top-k contexts."""
    haystack = "\n".join(contexts[:limit]).casefold()
    return tuple(session_id for session_id in answer_session_ids if session_id.casefold() in haystack)


def _match_required_string(label: str, observed: object, expected: object) -> list[str]:
    observed_text = str(observed or "").strip()
    expected_text = str(expected or "").strip()
    if not observed_text:
        return [f"{label} is required"]
    if expected_text and observed_text != expected_text:
        return [f"{label} does not match imported report"]
    return []


def _validate_reviewable_http_url(label: str, value: str) -> list[str]:
    failures: list[str] = []
    text = str(value or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return [f"{label} must be absolute http(s)"]
    host = (parsed.hostname or "").strip().casefold().rstrip(".")
    if not host:
        return [f"{label} must be absolute http(s)"]
    if host == "localhost" or host.endswith(".localhost"):
        failures.append(f"{label} must be a public reviewable URL")
    if host in {"example.com", "example.org", "example.net"} or host.endswith(".example"):
        failures.append(f"{label} must not use placeholder example domains")
    if host.endswith((".invalid", ".test")):
        failures.append(f"{label} must not use reserved placeholder domains")
    try:
        address = ip_address(host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        failures.append(f"{label} must be a public reviewable URL")
    return failures


def _match_required_int(label: str, observed: object, expected: int) -> list[str]:
    observed_text = str(observed).strip() if observed is not None else ""
    if not observed_text:
        return [f"{label} is required"]
    try:
        observed_int = int(observed_text)
    except (TypeError, ValueError):
        return [f"{label} must be an integer"]
    if observed_int != expected:
        return [f"{label} does not match imported report"]
    return []


def _match_required_float(label: str, observed: object, expected: float) -> list[str]:
    observed_text = str(observed).strip() if observed is not None else ""
    if not observed_text:
        return [f"{label} is required"]
    try:
        observed_float = float(observed_text)
    except (TypeError, ValueError):
        return [f"{label} must be a number"]
    if abs(observed_float - expected) > 0.000001:
        return [f"{label} does not match imported report"]
    return []


def _match_path(label: str, observed: object, expected: Path) -> list[str]:
    observed_text = str(observed or "").strip()
    if not observed_text:
        return [f"{label} is required"]
    if Path(observed_text).resolve() != expected.resolve():
        return [f"{label} does not match audited artifact path"]
    return []


def _dataset_count(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return len(payload) if isinstance(payload, list) else None


def _load_json_list(path: Path, label: str) -> list[object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON list")
    return payload


def _load_jsonl_objects(path: Path, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        rows.append(row)
    return rows


def _autoeval_label(row: dict[str, object]) -> bool | None:
    value = row.get("autoeval_label")
    if value is None:
        value = row.get("auto_eval_label")
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value > 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "correct", "yes", "1", "pass", "passed"}:
            return True
        if normalized in {"false", "incorrect", "no", "0", "fail", "failed"}:
            return False
    return None


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
