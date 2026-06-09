"""Harvey LAB external memory-ablation benchmark reporting."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zaxy.event import EventLog
from zaxy.verbatim import VerbatimIndex

HARVEY_LAB_REPO_URL = "https://github.com/rushilchugh01/harvey-labs-ablations-and-benchmarks"
ARTICLE_URL = "https://rushilchugh.substack.com/p/what-agent-memory-actually-fixes"
SCHEMA_VERSION = "zaxy.harvey-lab-benchmark.v1"


@dataclass(frozen=True)
class HarveyArticleTask:
    """One published task row from the Harvey LAB memory-ablation article."""

    task_id: str
    article_label: str
    task_shape: str
    documents_read_by_regular: str
    document_count: int
    criteria_count: int
    best_framework: str
    best_score: float
    regular_no_memory_score: float
    published_delta: float
    article_note: str


@dataclass(frozen=True)
class HarveyFrameworkFit:
    """Article interpretation for one compared memory/search framework."""

    framework: str
    where_strongest: str
    example_tasks: str
    interpretation: str


@dataclass(frozen=True)
class HarveyFrameworkScorecard:
    """Framework-level statistics from article disclosures and Zaxy rows."""

    framework: str
    evidence_scope: str
    article_task_count: int
    mean_score: float | None
    mean_delta_vs_regular_no_memory: float | None
    task_wins: int
    zaxy_overlap_task_count: int
    zaxy_mean_on_overlap: float | None
    zaxy_delta_on_overlap: float | None


@dataclass(frozen=True)
class HarveyExternalBaselineScorecard:
    """Framework aggregate row from Harvey-native external comparison artifacts."""

    framework: str
    evidence_scope: str
    runs: int
    mean_score: float | None
    delta_vs_raw_rg: float | None
    mean_total_seconds: float | None
    source_report_path: str


@dataclass(frozen=True)
class HarveyExternalComparisonScorecard:
    """Combined leaderboard row for Zaxy and Harvey-native scored systems."""

    framework: str
    evidence_scope: str
    runs: int
    mean_score: float | None
    delta_vs_raw_rg: float | None
    delta_vs_best_external: float | None
    rank_by_mean_score: int | None
    mean_total_seconds: float | None
    source_report_path: str | None


@dataclass(frozen=True)
class HarveyZaxyResult:
    """Normalized Zaxy result from a Harvey LAB run."""

    task_id: str
    score: float
    run_id: str
    framework: str
    generator: str | None
    judge: str | None
    generator_reasoning_effort: str | None
    judge_reasoning_effort: str | None
    temperature: float | None
    total_seconds: float | None
    total_tokens: int | None
    memory_search_calls: int | None
    memory_read_calls: int | None
    corpus_hash: str | None
    commit: str | None
    results_run_dir: str | None
    answer_path: str | None
    tool_log_path: str | None
    judge_path: str | None
    run_metrics_path: str | None


@dataclass(frozen=True)
class HarveyTaskComparison:
    """One task-level comparison between Zaxy and published article rows."""

    task_id: str
    article_label: str
    task_shape: str
    article_best_framework: str
    article_best_score: float
    regular_no_memory_score: float
    zaxy_score: float | None
    zaxy_delta_vs_regular_no_memory: float | None
    zaxy_delta_vs_article_best: float | None
    zaxy_winner: bool | None
    zaxy_total_seconds: float | None
    zaxy_total_tokens: int | None
    zaxy_memory_search_calls: int | None
    zaxy_memory_read_calls: int | None


@dataclass(frozen=True)
class HarveySummary:
    """Aggregate comparison statistics for available Zaxy Harvey LAB rows."""

    status: str
    article_task_count: int
    zaxy_task_count: int
    zaxy_mean_score: float | None
    article_best_mean_for_zaxy_tasks: float | None
    regular_mean_for_zaxy_tasks: float | None
    mean_delta_vs_regular_no_memory: float | None
    mean_delta_vs_article_best: float | None
    zaxy_task_wins: int
    zaxy_task_losses: int
    zaxy_task_ties: int
    zaxy_mean_total_seconds: float | None
    zaxy_total_tokens: int | None
    zaxy_total_memory_search_calls: int | None
    zaxy_total_memory_read_calls: int | None


@dataclass(frozen=True)
class HarveyLabReport:
    """Complete Harvey LAB external benchmark report."""

    schema_version: str
    generated_at: str
    status: str
    external_suite: dict[str, str]
    result_provenance: dict[str, object]
    summary: HarveySummary
    task_rows: dict[str, HarveyTaskComparison]
    framework_scorecard: dict[str, HarveyFrameworkScorecard]
    external_baseline_scorecard: dict[str, HarveyExternalBaselineScorecard]
    external_comparison_scorecard: dict[str, HarveyExternalComparisonScorecard]
    framework_fit: dict[str, HarveyFrameworkFit]
    zaxy_results: tuple[HarveyZaxyResult, ...]
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class WrittenHarveyLabReport:
    """Paths written for a Harvey LAB benchmark report."""

    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class WrittenHarveyExternalRunManifest:
    """Paths written for the external Harvey run manifest."""

    json_path: Path
    markdown_path: Path
    script_path: Path


HarveyMemoryManifest = dict[str, object]

HARVEY_ADAPTER_MODULE = '''"""Harvey LAB memory adapter shim for Zaxy.

Copy this file to `scripts/memory_ablation/raw_rg_memory.py` in a Harvey LAB
worktree, set `HARVEY_MEMORY_MANIFEST` to a Zaxy manifest produced by
`zaxy harvey-lab-index`, and run the existing Harvey harness.
"""

from __future__ import annotations

from pathlib import Path

from zaxy_benchmarks.harvey_lab_benchmark import (
    build_harvey_zaxy_memory_index,
    harvey_memory_read,
    harvey_memory_search,
)


def scan_corpus(corpus_root):
    """Build a Zaxy Eventloom index for a Harvey corpus root."""
    root = Path(corpus_root)
    output_dir = root.parent / ".zaxy-harvey-index"
    return build_harvey_zaxy_memory_index(root, output_dir)


def search(manifest, query, limit=5):
    """Return Harvey-compatible memory_search JSON backed by Zaxy."""
    return harvey_memory_search(manifest, query, limit=limit)


def read(manifest, item_id, context_lines=8):
    """Return Harvey-compatible memory_read JSON backed by Zaxy."""
    return harvey_memory_read(manifest, item_id, context_lines=context_lines)
'''

HARVEY_ADAPTER_README = """# Zaxy Harvey LAB Adapter Kit

This kit is for an external Harvey LAB worktree. It keeps the benchmark external
to Zaxy while letting Harvey's existing `memory_search` and `memory_read` tool
names call a Zaxy Eventloom-backed memory index.

1. In the Harvey worktree, normalize a task corpus using Harvey's
   `scripts/memory_ablation/normalize_corpus.py`.
2. Build the Zaxy index:

   ```bash
   zaxy harvey-lab-index \\
     --normalized-corpus-root .ingestion/corpora/HASH/txt \\
     --source-map .ingestion/corpora/HASH/source-map.json \\
     --output-dir .ingestion/indexes/HASH/zaxy
   ```

3. Copy `raw_rg_memory.py` from this kit to
   `scripts/memory_ablation/raw_rg_memory.py` in the Harvey worktree, or apply
   the equivalent branch-local import change.
4. Run Harvey with an explicit Zaxy manifest:

   ```bash
   HARVEY_MEMORY_MANIFEST=.ingestion/indexes/HASH/zaxy/manifest.json \\
   uv run python -m harness.run --task TASK_ID --model MODEL
   ```

   The shim also exposes `scan_corpus(corpus_root)` for Harvey's fallback path.
   That path builds a `.zaxy-harvey-index` directory beside the corpus and
   returns the corpus metadata (`corpus_hash`, `corpus_root`, `files`) expected
   by the Harvey harness.

5. Judge with Harvey's normal evaluation scripts, then materialize the
   normalized-result contract:

   ```bash
   zaxy harvey-lab-normalize-run \\
     --harvey-worktree . \\
     --run-id RUN_ID \\
     --task-id TASK_ID \\
     --manifest .ingestion/indexes/HASH/zaxy/manifest.json
   ```
"""


ARTICLE_TASKS: tuple[HarveyArticleTask, ...] = (
    HarveyArticleTask(
        task_id="corporate-governance/assess-impact-of-ftc-noncompete-ban-on-existing-employment-agreements",
        article_label="FTC noncompete",
        task_shape="Compact legal-risk synthesis",
        documents_read_by_regular="15/22",
        document_count=22,
        criteria_count=57,
        best_framework="Graphiti",
        best_score=0.790,
        regular_no_memory_score=0.807,
        published_delta=-0.018,
        article_note="Counterexample where the regular run already found enough decisive evidence.",
    ),
    HarveyArticleTask(
        task_id="corporate-ma/analyze-change-of-control-provisions-across-targets-material-contracts",
        article_label="Change-of-control",
        task_shape="Sparse clause hunt",
        documents_read_by_regular="5/19",
        document_count=19,
        criteria_count=57,
        best_framework="GBrain keyword",
        best_score=0.737,
        regular_no_memory_score=0.667,
        published_delta=0.070,
        article_note="Sparse clause hunt where retrieval helped reach clause-bearing contracts.",
    ),
    HarveyArticleTask(
        task_id="corporate-ma/draft-acquisition-due-diligence",
        article_label="Acquisition diligence",
        task_shape="Broad diligence sweep",
        documents_read_by_regular="11/31",
        document_count=31,
        criteria_count=64,
        best_framework="raw-rg",
        best_score=0.641,
        regular_no_memory_score=0.469,
        published_delta=0.172,
        article_note="Direct lexical retrieval found more relevant source material.",
    ),
    HarveyArticleTask(
        task_id="corporate-ma/review-data-room-red-flag-review",
        article_label="Data-room red flags",
        task_shape="Red-flag spotting",
        documents_read_by_regular="13/13",
        document_count=13,
        criteria_count=50,
        best_framework="LightRAG",
        best_score=0.600,
        regular_no_memory_score=0.520,
        published_delta=0.080,
        article_note="Full document coverage was not enough; evidence focus mattered.",
    ),
    HarveyArticleTask(
        task_id="data-privacy-cybersecurity/compare-privacy-program-documentation-against-applicable-data-protection-regulations",
        article_label="Privacy program",
        task_shape="Compliance mapping",
        documents_read_by_regular="11/13",
        document_count=13,
        criteria_count=62,
        best_framework="ActiveGraph",
        best_score=0.661,
        regular_no_memory_score=0.532,
        published_delta=0.129,
        article_note="Structured state helped organize controls, obligations, and gaps.",
    ),
    HarveyArticleTask(
        task_id="litigation-dispute-resolution/build-litigation-case-timeline",
        article_label="Litigation timeline",
        task_shape="Event reconstruction",
        documents_read_by_regular="15/15",
        document_count=15,
        criteria_count=66,
        best_framework="GBrain keyword",
        best_score=0.758,
        regular_no_memory_score=0.652,
        published_delta=0.106,
        article_note="Retrieval helped pull dated facts into a more complete chronology.",
    ),
    HarveyArticleTask(
        task_id="litigation-dispute-resolution/categorize-document-production-set-by-relevance-and-privilege",
        article_label="Relevance / privilege",
        task_shape="Document-by-document coding",
        documents_read_by_regular="25/25",
        document_count=25,
        criteria_count=67,
        best_framework="GBrain keyword",
        best_score=0.791,
        regular_no_memory_score=0.701,
        published_delta=0.090,
        article_note="Retrieval formulation mattered even after all documents were read.",
    ),
    HarveyArticleTask(
        task_id="litigation-dispute-resolution/review-document-production-set-for-attorney",
        article_label="Attorney production review",
        task_shape="Production-set classification",
        documents_read_by_regular="18/18",
        document_count=18,
        criteria_count=48,
        best_framework="GBrain Gemma / LightRAG",
        best_score=0.708,
        regular_no_memory_score=0.583,
        published_delta=0.125,
        article_note="The issue was likely salience, not raw document exposure.",
    ),
    HarveyArticleTask(
        task_id="litigation-dispute-resolution/review-privilege-log-clawback-review",
        article_label="Privilege log",
        task_shape="Large log-heavy classification",
        documents_read_by_regular="3/55",
        document_count=55,
        criteria_count=82,
        best_framework="GBrain keyword",
        best_score=0.598,
        regular_no_memory_score=0.402,
        published_delta=0.195,
        article_note="The clearest coverage-insurance case in the article.",
    ),
    HarveyArticleTask(
        task_id="white-collar-defense-investigations/compare-document-production-set-against-subpoena-request-categories",
        article_label="Subpoena comparison",
        task_shape="Request matching",
        documents_read_by_regular="6/14",
        document_count=14,
        criteria_count=57,
        best_framework="raw-rg",
        best_score=0.790,
        regular_no_memory_score=0.702,
        published_delta=0.088,
        article_note="Direct matching problem where lexical search was a strong baseline.",
    ),
)

ARTICLE_FRAMEWORK_FIT: dict[str, HarveyFrameworkFit] = {
    "raw-rg": HarveyFrameworkFit(
        framework="raw-rg",
        where_strongest="Literal evidence finding",
        example_tasks="Acquisition diligence; subpoena comparison",
        interpretation=(
            "raw-rg is a retrieval/search baseline, not a no-memory baseline; "
            "it was strongest when direct lexical matches were enough."
        ),
    ),
    "GBrain keyword": HarveyFrameworkFit(
        framework="GBrain keyword",
        where_strongest="Clause hunts and classification-heavy tasks",
        example_tasks="Change-of-control; litigation timeline; relevance / privilege; privilege log",
        interpretation="Keyword retrieval preserved high-precision hooks from task prompts and rubrics.",
    ),
    "GBrain Gemma": HarveyFrameworkFit(
        framework="GBrain Gemma",
        where_strongest="Production-review style classification",
        example_tasks="Attorney production review",
        interpretation="The query layer appeared to bring useful classification cues into final review.",
    ),
    "LightRAG": HarveyFrameworkFit(
        framework="LightRAG",
        where_strongest="Red-flag spotting after full document coverage",
        example_tasks="Data-room red flags; attorney production review",
        interpretation="Graph/vector retrieval looked useful when selection and focus mattered.",
    ),
    "ActiveGraph": HarveyFrameworkFit(
        framework="ActiveGraph",
        where_strongest="Compliance/state mapping",
        example_tasks="Privacy program",
        interpretation="Structured state helped organize controls, obligations, and gaps.",
    ),
    "Graphiti": HarveyFrameworkFit(
        framework="Graphiti",
        where_strongest="Compact legal synthesis",
        example_tasks="FTC noncompete",
        interpretation="Episode/graph memory may organize actors and relationships, but regular already scored higher.",
    ),
    "Mem0": HarveyFrameworkFit(
        framework="Mem0",
        where_strongest="Native stored source chunk memory",
        example_tasks="Article comparison set",
        interpretation="Included as a native memory-search system in the article, but not a task winner in the published matrix.",
    ),
}


def load_harvey_zaxy_results(path: Path) -> tuple[HarveyZaxyResult, ...]:
    """Load Zaxy rows from Harvey memory-ablation normalized result artifacts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("normalized_results"), list):
        raw_results = payload["normalized_results"]
    elif isinstance(payload, list):
        raw_results = payload
    elif isinstance(payload, dict):
        raw_results = [payload]
    else:
        raise ValueError("Harvey Zaxy results JSON must be an object or list")
    results = tuple(_parse_zaxy_result(item) for item in raw_results)
    if not results:
        raise ValueError("Harvey Zaxy results JSON did not contain any results")
    return results


def import_harvey_zaxy_results(roots: list[Path] | tuple[Path, ...]) -> tuple[HarveyZaxyResult, ...]:
    """Discover and load Zaxy normalized-result.json files from external Harvey roots."""
    results = [
        result
        for _, result in _latest_harvey_result_items(roots)
    ]
    if not results:
        raise ValueError("No Harvey Zaxy normalized-result.json files found")
    return tuple(results)


def build_harvey_result_provenance(
    roots: list[Path] | tuple[Path, ...],
    *,
    source: str,
) -> dict[str, object]:
    """Build reviewable provenance metadata for external Harvey result roots."""
    result_paths: list[Path] = []
    resolved_roots = [root.resolve() for root in roots]
    selected_items = _latest_harvey_result_items(resolved_roots)
    result_paths.extend(path for path, _ in selected_items)
    git_commits = [
        commit
        for root in resolved_roots
        if (commit := _git_commit_or_none(root)) is not None
    ]
    provenance: dict[str, object] = {
        "source": source,
        "roots": [str(root) for root in resolved_roots],
        "normalized_result_paths": [
            str(path.resolve())
            for path in sorted(dict.fromkeys(result_paths))
        ],
    }
    baseline_reports = _discover_harvey_baseline_reports(resolved_roots)
    if baseline_reports:
        provenance["external_baseline_report_paths"] = [
            str(report["path"])
            for report in baseline_reports
        ]
        provenance["external_baseline_reports"] = baseline_reports
    run_gate_artifacts = _discover_harvey_run_gate_artifacts(resolved_roots)
    provenance.update(run_gate_artifacts)
    if git_commits:
        provenance["harvey_git_commit"] = git_commits[0]
    return provenance


def _discover_harvey_run_gate_artifacts(roots: list[Path]) -> dict[str, object]:
    artifact_specs = {
        "external_run_manifest_paths": (
            "harvey-lab-external-run.json",
            "zaxy.harvey-lab-external-run.v1",
        ),
        "external_readiness_report_paths": (
            "harvey-lab-ready.json",
            "zaxy.harvey-lab-run-readiness.v1",
        ),
        "external_status_report_paths": (
            "harvey-lab-status.json",
            "zaxy.harvey-lab-run-status.v1",
        ),
    }
    discovered: dict[str, object] = {}
    for key, (filename, schema_version) in artifact_specs.items():
        paths: list[str] = []
        seen: set[Path] = set()
        for root in roots:
            for path in _harvey_run_gate_artifact_candidates(root, filename):
                resolved = path.resolve()
                if resolved in seen or not _json_file_has_schema(resolved, schema_version):
                    continue
                seen.add(resolved)
                paths.append(str(resolved))
        if paths:
            discovered[key] = paths
    return discovered


def _harvey_run_gate_artifact_candidates(root: Path, filename: str) -> list[Path]:
    return [
        root / filename,
        root / "reports" / "benchmarks" / "harvey-lab-memory-ablation" / filename,
    ]


def _json_file_has_schema(path: Path, schema_version: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") == schema_version


def _discover_harvey_baseline_reports(roots: list[Path]) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root in roots:
        reports_dir = root / ".ingestion" / "reports"
        if not reports_dir.exists():
            continue
        for path in sorted(reports_dir.glob("comparison*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            summary = _harvey_baseline_report_summary(resolved)
            if summary is not None:
                reports.append(summary)
    return reports


def _harvey_baseline_report_summary(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    aggregate = payload.get("aggregate")
    frameworks: object = []
    if isinstance(aggregate, dict):
        frameworks = aggregate.get("frameworks", [])
    normalized_results = payload.get("normalized_results", [])
    return {
        "path": str(path),
        "schema_version": str(payload.get("schema_version", "")),
        "framework_count": len(frameworks) if isinstance(frameworks, list) else 0,
        "normalized_result_count": (
            len(normalized_results)
            if isinstance(normalized_results, list)
            else 0
        ),
    }


def build_harvey_zaxy_memory_index(
    normalized_corpus_root: Path,
    output_dir: Path,
    *,
    source_map_path: Path | None = None,
    max_lines: int = 80,
) -> HarveyMemoryManifest:
    """Index a Harvey normalized text corpus into Eventloom for memory tools."""
    corpus_root = normalized_corpus_root.resolve()
    if not corpus_root.exists():
        raise ValueError(f"normalized corpus root does not exist: {corpus_root}")
    if max_lines < 1:
        raise ValueError("max_lines must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    eventloom_path = output_dir / "zaxy-harvey-lab.eventloom.jsonl"
    if eventloom_path.exists():
        eventloom_path.unlink()
    eventlog = EventLog(eventloom_path)
    events: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    for path in sorted(corpus_root.rglob("*.txt")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        rel_path = path.relative_to(corpus_root).as_posix()
        digest = _sha256_text(content)
        files.append(
            {
                "relative_path": rel_path,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
        for start in range(0, len(lines), max_lines):
            chunk_lines = lines[start : start + max_lines]
            if not any(line.strip() for line in chunk_lines):
                continue
            events.append(
                {
                    "event_type": "document.indexed",
                    "actor": "zaxy-harvey-lab-adapter",
                    "payload": {
                        "path": rel_path,
                        "start_line": start + 1,
                        "end_line": start + len(chunk_lines),
                        "content": "\n".join(chunk_lines),
                        "sha256": digest,
                    },
                    "thread": "harvey-lab",
                }
            )
    eventlog.append_many(events)
    indexed_paths: set[str] = set()
    for item in events:
        payload = item.get("payload")
        if isinstance(payload, dict):
            indexed_paths.add(str(payload.get("path", "")))
    manifest: HarveyMemoryManifest = {
        "framework": "zaxy",
        "adapter_contract": "harvey-memory-ablation-zaxy-v1",
        "eventloom_path": str(eventloom_path),
        "corpus_root": str(corpus_root),
        "corpus_hash": _sha256_json(files),
        "files": files,
        "file_count": len(indexed_paths),
        "event_count": len(events),
        "normalized_text": {
            "version": "harvey-normalized-text-v1",
            "corpus_root": str(corpus_root),
            "source_map": str(source_map_path) if source_map_path is not None else None,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_harvey_index_artifacts(output_dir, manifest)
    return manifest


def export_harvey_adapter_kit(output_dir: Path) -> dict[str, str]:
    """Write a small Harvey-compatible adapter kit for external worktrees."""
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_module = output_dir / "raw_rg_memory.py"
    readme = output_dir / "README.md"
    memory_module.write_text(HARVEY_ADAPTER_MODULE, encoding="utf-8")
    readme.write_text(HARVEY_ADAPTER_README, encoding="utf-8")
    return {
        "memory_module": str(memory_module),
        "readme": str(readme),
    }


def check_harvey_external_suite(worktree: Path) -> dict[str, object]:
    """Validate that a Harvey checkout matches the external article suite shape."""
    root = worktree.resolve()
    required_files = (
        "harness/run.py",
        "evaluation/run_eval.py",
        "scripts/memory_ablation/normalize_corpus.py",
        "scripts/memory_ablation/validate_result.py",
        "scripts/memory_ablation/collect_results.py",
        "scripts/memory_ablation/render_report.py",
    )
    missing_files = [
        relative
        for relative in required_files
        if not (root / relative).is_file()
    ]
    missing_tasks: list[str] = []
    present_tasks: list[str] = []
    task_audits: dict[str, dict[str, object]] = {}
    task_mismatches: list[dict[str, object]] = []
    for task in ARTICLE_TASKS:
        task_dir = root / "tasks" / task.task_id
        task_json_path = task_dir / "task.json"
        documents_dir = task_dir / "documents"
        task_ok = task_json_path.is_file() and documents_dir.is_dir()
        if task_ok:
            present_tasks.append(task.task_id)
            document_count = sum(1 for path in documents_dir.iterdir() if path.is_file())
            criteria_count = _harvey_task_criteria_count(task_json_path)
            task_audits[task.task_id] = {
                "document_count": document_count,
                "expected_document_count": task.document_count,
                "criteria_count": criteria_count,
                "expected_criteria_count": task.criteria_count,
            }
            if document_count != task.document_count or criteria_count != task.criteria_count:
                task_mismatches.append(
                    {
                        "task_id": task.task_id,
                        "reason": "task_shape_mismatch",
                        "expected_document_count": task.document_count,
                        "actual_document_count": document_count,
                        "expected_criteria_count": task.criteria_count,
                        "actual_criteria_count": criteria_count,
                    }
                )
        else:
            missing_tasks.append(task.task_id)
    status = (
        "valid"
        if root.is_dir() and not missing_files and not missing_tasks and not task_mismatches
        else "invalid"
    )
    return {
        "schema_version": "zaxy.harvey-lab-external-suite-check.v1",
        "status": status,
        "worktree": str(root),
        "source_url": HARVEY_LAB_REPO_URL,
        "article_url": ARTICLE_URL,
        "task_count": len(present_tasks),
        "expected_task_count": len(ARTICLE_TASKS),
        "present_task_ids": present_tasks,
        "missing_task_ids": missing_tasks,
        "task_audits": task_audits,
        "task_mismatches": task_mismatches,
        "missing_required_files": missing_files,
        "harvey_git_commit": _git_commit_or_none(root),
        "message": (
            "Harvey external suite checkout contains the pinned article tasks and required memory-ablation scripts."
            if status == "valid"
            else "Harvey external suite checkout is missing pinned article tasks or required memory-ablation scripts."
        ),
    }


def build_harvey_external_index_preflight(
    worktree: Path,
    *,
    max_lines: int = 80,
    smoke_query: str = "evidence",
    smoke_limit: int = 3,
    use_external_environment: bool = True,
    task_filter: str | None = None,
) -> dict[str, object]:
    """Normalize and index pinned Harvey tasks without running model judges."""
    root = worktree.resolve()
    suite = check_harvey_external_suite(root)
    if suite["status"] != "valid":
        raise ValueError("Harvey external suite checkout is invalid; run harvey-lab-doctor first")
    filter_text = (task_filter or "").strip()
    resolved_task: HarveyArticleTask | None = None
    if filter_text:
        resolved_task = _resolve_harvey_task_filter(filter_text)
        if resolved_task is None:
            raise ValueError(f"Unknown Harvey task filter: {filter_text}")
    tasks_to_index = (resolved_task,) if resolved_task is not None else ARTICLE_TASKS
    normalizer: Any | None = None if use_external_environment else _load_harvey_normalizer(root)
    tasks: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    for task in tasks_to_index:
        slug = _harvey_task_slug(task.task_id)
        task_dir = root / "tasks" / task.task_id
        documents_dir = task_dir / "documents"
        ingestion_root = root / ".ingestion"
        if use_external_environment:
            normalization = _prepare_harvey_normalized_corpus_external(root, task.task_id)
        else:
            if normalizer is None:
                raise ValueError("Harvey normalizer was not loaded")
            normalization = normalizer.prepare_normalized_corpus(documents_dir, ingestion_root)
        normalized_root = Path(_str_required(normalization.get("normalized_corpus_root"), "normalized_corpus_root"))
        source_map = Path(_str_required(normalization.get("source_map_path"), "source_map_path"))
        index_dir = root / ".ingestion" / "indexes" / slug / "zaxy"
        manifest = build_harvey_zaxy_memory_index(
            normalized_root,
            index_dir,
            source_map_path=source_map,
            max_lines=max_lines,
        )
        task_smoke_query = _harvey_smoke_query_from_corpus(
            normalized_root,
            fallback=f"{task.article_label} {smoke_query}",
        )
        smoke = harvey_memory_search(manifest, task_smoke_query, limit=smoke_limit)
        hits = smoke.get("hits")
        hit_count = len(hits) if isinstance(hits, list) else 0
        if hit_count <= 0:
            failures.append({"task_id": task.task_id, "reason": "empty_smoke_search"})
        tasks[task.task_id] = {
            "task_id": task.task_id,
            "article_label": task.article_label,
            "status": "indexed" if hit_count > 0 else "failed",
            "document_count": task.document_count,
            "criteria_count": task.criteria_count,
            "normalized_corpus_root": str(normalized_root),
            "source_map_path": str(source_map),
            "index_dir": str(index_dir),
            "manifest_path": str(index_dir / "manifest.json"),
            "artifact_summary_path": str(index_dir / "artifact-summary.json"),
            "smoke_result_path": str(index_dir / "smoke-result.json"),
            "corpus_hash": manifest.get("corpus_hash"),
            "file_count": manifest.get("file_count"),
            "event_count": manifest.get("event_count"),
            "smoke_query": task_smoke_query,
            "smoke_search_hit_count": hit_count,
        }
    status = "ready_for_external_runs" if not failures else "failed"
    return {
        "schema_version": "zaxy.harvey-lab-index-preflight.v1",
        "status": status,
        "worktree": str(root),
        "source_url": HARVEY_LAB_REPO_URL,
        "article_url": ARTICLE_URL,
        "harvey_git_commit": suite.get("harvey_git_commit"),
        "task_filter": filter_text,
        "resolved_task_id": resolved_task.task_id if resolved_task is not None else None,
        "task_count": len(tasks),
        "expected_task_count": len(tasks_to_index),
        "tasks": tasks,
        "failures": failures,
        "message": (
            "All pinned Harvey LAB tasks have Zaxy index artifacts and smoke-search evidence; ready for external model runs."
            if status == "ready_for_external_runs"
            else "One or more Harvey LAB task indexes failed smoke validation; do not run comparative scoring yet."
        ),
    }


def _load_harvey_normalizer(root: Path) -> Any:
    normalizer_path = root / "scripts" / "memory_ablation" / "normalize_corpus.py"
    spec = importlib.util.spec_from_file_location("zaxy_harvey_external_normalize_corpus", normalizer_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load Harvey normalizer: {normalizer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "prepare_normalized_corpus"):
        raise ValueError("Harvey normalize_corpus.py does not define prepare_normalized_corpus")
    return module


def _prepare_harvey_normalized_corpus_external(root: Path, task_id: str) -> dict[str, object]:
    script = """
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(
    root / "tasks" / task_id / "documents",
    root / ".ingestion",
)
print(json.dumps(normalization))
""".lstrip()
    completed = subprocess.run(
        ["uv", "run", "python", "-", task_id],
        cwd=root,
        input=script,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "Harvey normalization failed in the external checkout environment: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Harvey normalization returned non-JSON output") from exc
    if not isinstance(payload, dict):
        raise ValueError("Harvey normalization must return a JSON object")
    return payload


def _str_required(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Harvey normalization result requires {name}")
    return value


def _harvey_smoke_query_from_corpus(normalized_root: Path, *, fallback: str) -> str:
    for path in sorted(normalized_root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\n\n" in text:
            text = text.split("\n\n", 1)[1]
        tokens = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
            if token.casefold() not in {"source", "path", "extractor", "original", "size", "bytes"}
        ]
        if tokens:
            return " ".join(tokens[:3])
    return fallback


def _harvey_task_criteria_count(task_json_path: Path) -> int | None:
    try:
        payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    criteria = payload.get("criteria")
    if isinstance(criteria, list):
        return len(criteria)
    return None


def _harvey_task_slug(task_id: str) -> str:
    return task_id.replace("/", "__")


def _resolve_harvey_task_filter(task_filter: str) -> HarveyArticleTask | None:
    """Resolve a Harvey task id, slug, or generated Zaxy run id to the pinned task."""
    for task in ARTICLE_TASKS:
        slug = _harvey_task_slug(task.task_id)
        run_id = f"zaxy-{slug}"
        if task_filter in {task.task_id, slug, run_id}:
            return task
    return None


def build_harvey_external_run_status(worktree: Path) -> dict[str, object]:
    """Return per-task readiness for the external Harvey Zaxy run pipeline."""
    root = worktree.resolve()
    task_statuses: dict[str, dict[str, object]] = {}
    ready_count = 0
    for task in ARTICLE_TASKS:
        slug = _harvey_task_slug(task.task_id)
        run_id = f"zaxy-{slug}"
        index_dir = root / ".ingestion" / "indexes" / slug / "zaxy"
        run_dir = root / "results" / run_id
        normalized_dir = root / ".ingestion" / "runs" / run_id
        run_answer_path = _harvey_run_answer_path(root, run_dir, task.task_id)
        expected_artifacts = {
            "index_manifest": index_dir / "manifest.json",
            "index_artifact_summary": index_dir / "artifact-summary.json",
            "index_smoke_result": index_dir / "smoke-result.json",
            "run_config": run_dir / "config.json",
            "run_metrics": run_dir / "metrics.json",
            "run_scores": run_dir / "scores.json",
            "run_transcript": run_dir / "transcript.jsonl",
            "run_answer": run_answer_path,
            "normalized_result": normalized_dir / "normalized-result.json",
        }
        missing_artifacts = [
            label
            for label, path in expected_artifacts.items()
            if not path.exists()
        ]
        index_ready = all(
            expected_artifacts[label].exists()
            for label in ("index_manifest", "index_artifact_summary", "index_smoke_result")
        )
        run_artifact_files_ready = all(
            expected_artifacts[label].exists()
            for label in ("run_config", "run_metrics", "run_scores", "run_transcript", "run_answer")
        )
        run_artifacts_ready, run_artifact_error = _harvey_status_run_artifacts_ready(
            run_artifact_files_ready,
            expected_artifacts["run_metrics"],
            expected_artifacts["run_transcript"],
        )
        normalized_result_ready, normalized_result_error = _harvey_status_normalized_result_ready(
            expected_artifacts["normalized_result"],
            task.task_id,
            run_id,
        )
        import_ready = index_ready and run_artifacts_ready and normalized_result_ready
        if import_ready:
            ready_count += 1
        task_statuses[task.task_id] = {
            "task_id": task.task_id,
            "article_label": task.article_label,
            "run_id": run_id,
            "index_dir": str(index_dir),
            "run_dir": str(run_dir),
            "normalized_result_path": str(expected_artifacts["normalized_result"]),
            "index_ready": index_ready,
            "run_artifacts_ready": run_artifacts_ready,
            "run_artifact_error": run_artifact_error,
            "normalized_result_ready": normalized_result_ready,
            "normalized_result_error": normalized_result_error,
            "import_ready": import_ready,
            "missing_artifacts": [
                *missing_artifacts,
                *([] if run_artifact_error is None else ["run_memory_evidence"]),
                *([] if normalized_result_error is None else ["normalized_result_contract"]),
            ],
        }
    status = "complete" if ready_count == len(ARTICLE_TASKS) else ("partial" if ready_count else "not_ready")
    evidence_audit = _harvey_external_status_evidence_audit(task_statuses)
    return {
        "schema_version": "zaxy.harvey-lab-run-status.v1",
        "status": status,
        "worktree": str(root),
        "harvey_git_commit": _git_commit_or_none(root),
        "ready_task_count": ready_count,
        "expected_task_count": len(ARTICLE_TASKS),
        "evidence_audit": evidence_audit,
        "tasks": task_statuses,
        "message": (
            "All pinned Harvey LAB article tasks have import-ready Zaxy artifacts."
            if status == "complete"
            else "External Harvey LAB Zaxy artifacts are incomplete; do not publish comparative statistics."
        ),
    }


def _harvey_external_status_evidence_audit(
    task_statuses: dict[str, dict[str, object]],
) -> dict[str, int]:
    return {
        "index_ready_count": sum(1 for item in task_statuses.values() if item.get("index_ready") is True),
        "run_artifacts_ready_count": sum(
            1 for item in task_statuses.values() if item.get("run_artifacts_ready") is True
        ),
        "normalized_result_ready_count": sum(
            1 for item in task_statuses.values() if item.get("normalized_result_ready") is True
        ),
        "memory_evidence_ready_count": sum(
            1
            for item in task_statuses.values()
            if item.get("run_artifacts_ready") is True and item.get("run_artifact_error") is None
        ),
        "import_ready_count": sum(1 for item in task_statuses.values() if item.get("import_ready") is True),
        "expected_task_count": len(task_statuses),
    }


def build_harvey_external_run_readiness(
    worktree: Path,
    *,
    generator: str,
    judge: str,
    env: dict[str, str] | None = None,
    task_filter: str | None = None,
) -> dict[str, object]:
    """Report whether the external Harvey checkout is ready for model-backed Zaxy runs."""
    root = worktree.resolve()
    env_map = _harvey_effective_env(root, env)
    suite = check_harvey_external_suite(root)
    status = build_harvey_external_run_status(root)
    filter_text = (task_filter or "").strip()
    task_items, resolved_task_id = _harvey_readiness_task_items(status, filter_text)
    expected_task_count = len(task_items) if filter_text else len(ARTICLE_TASKS)
    index_ready_count = sum(1 for item in task_items if isinstance(item, dict) and item.get("index_ready") is True)
    ready_task_count = sum(1 for item in task_items if isinstance(item, dict) and item.get("import_ready") is True)
    unresolved_models = [
        label
        for label, model in (("generator", generator), ("judge", judge))
        if model in {"HARVEY_GENERATOR_MODEL", "HARVEY_JUDGE_MODEL"}
    ]
    model_requirements = {
        "generator": _harvey_model_requirement(generator, env_map),
        "judge": _harvey_model_requirement(judge, env_map),
    }
    missing_credentials = _harvey_missing_credentials(model_requirements)
    sandbox_runtime = _harvey_sandbox_runtime_requirement(env_map)
    host_document_reader = _harvey_host_document_reader_requirement(env_map)
    blocking_reasons: list[str] = []
    if suite["status"] != "valid":
        blocking_reasons.append("invalid_harvey_suite")
    if filter_text and resolved_task_id is None:
        blocking_reasons.append("unknown_task_filter")
    if index_ready_count != expected_task_count:
        blocking_reasons.append("missing_zaxy_indexes")
    if unresolved_models:
        blocking_reasons.append("unresolved_model_placeholders")
    if missing_credentials:
        blocking_reasons.append("missing_model_credentials")
    if sandbox_runtime["status"] != "ready":
        blocking_reasons.append("missing_sandbox_runtime")
    if host_document_reader["status"] != "ready":
        blocking_reasons.append("missing_host_document_reader")
    if expected_task_count > 0 and ready_task_count == expected_task_count:
        blocking_reasons.append("results_already_complete")
    readiness_status = "ready_for_external_runs" if not blocking_reasons else "not_ready"
    return {
        "schema_version": "zaxy.harvey-lab-run-readiness.v1",
        "status": readiness_status,
        "worktree": str(root),
        "source_url": HARVEY_LAB_REPO_URL,
        "article_url": ARTICLE_URL,
        "harvey_git_commit": suite.get("harvey_git_commit"),
        "suite_valid": suite["status"] == "valid",
        "task_filter": filter_text,
        "resolved_task_id": resolved_task_id,
        "expected_task_count": expected_task_count,
        "index_ready_count": index_ready_count,
        "ready_task_count": ready_task_count,
        "run_ready_count": _harvey_status_count(status, "run_artifacts_ready", task_filter=filter_text),
        "normalized_ready_count": _harvey_status_count(status, "normalized_result_ready", task_filter=filter_text),
        "evidence_audit": _harvey_external_status_evidence_audit(
            {
                str(item.get("task_id") or index): item
                for index, item in enumerate(task_items)
                if isinstance(item, dict)
            }
        ),
        "unresolved_models": unresolved_models,
        "model_requirements": model_requirements,
        "sandbox_runtime": sandbox_runtime,
        "host_document_reader": host_document_reader,
        "missing_credentials": missing_credentials,
        "dotenv_path": str(root / ".env") if (root / ".env").exists() else None,
        "blocking_reasons": blocking_reasons,
        "message": (
            "Harvey checkout has Zaxy indexes and model prerequisites; run the generated external script to produce judged artifacts."
            if readiness_status == "ready_for_external_runs"
            else "Harvey external Zaxy run prerequisites are incomplete; do not start or publish comparative scoring yet."
        ),
    }


def _harvey_readiness_task_items(
    status: dict[str, object],
    task_filter: str,
) -> tuple[list[dict[str, object]], str | None]:
    tasks = status.get("tasks")
    if not isinstance(tasks, dict):
        return [], None
    if not task_filter:
        return [
            item
            for item in tasks.values()
            if isinstance(item, dict)
        ], None
    task = _resolve_harvey_task_filter(task_filter)
    if task is not None:
        item = tasks.get(task.task_id)
        return ([item] if isinstance(item, dict) else []), task.task_id
    return [], None


def _harvey_status_count(
    status: dict[str, object],
    field: str,
    *,
    task_filter: str = "",
) -> int:
    filtered_items, _ = _harvey_readiness_task_items(status, task_filter)
    if task_filter:
        return sum(1 for item in filtered_items if item.get(field) is True)
    tasks = status.get("tasks")
    if not isinstance(tasks, dict):
        return 0
    return sum(1 for item in tasks.values() if isinstance(item, dict) and item.get(field) is True)


def _harvey_effective_env(root: Path, env: dict[str, str] | None) -> dict[str, str]:
    env_map = dict(os.environ if env is None else env)
    dotenv_path = root / ".env"
    if not dotenv_path.exists():
        return env_map
    for line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key or key in env_map:
            continue
        env_map[key] = value.strip().strip('"').strip("'")
    return env_map


def _harvey_missing_credentials(
    model_requirements: dict[str, dict[str, object]],
) -> list[str]:
    missing: set[str] = set()
    for requirement in model_requirements.values():
        credentials = requirement.get("missing_credentials")
        if not isinstance(credentials, list):
            continue
        missing.update(credential for credential in credentials if isinstance(credential, str))
    return sorted(missing)


def _harvey_sandbox_runtime_requirement(env: dict[str, str]) -> dict[str, object]:
    """Report whether Harvey's required Podman sandbox runtime is launchable."""
    path = str(env.get("PATH") or os.environ.get("PATH") or "")
    executable = shutil.which("podman", path=path)
    if executable is None:
        return {
            "status": "not_ready",
            "required_by": "Harvey LAB harness sandbox",
            "runtime": "podman",
            "podman_status": "missing",
            "executable": None,
            "message": "Harvey LAB harness.run requires podman; no podman executable was found on PATH.",
        }
    try:
        result = subprocess.run(
            [executable, "info"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env={**os.environ, **env, "PATH": path},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "not_ready",
            "required_by": "Harvey LAB harness sandbox",
            "runtime": "podman",
            "podman_status": "unreachable",
            "executable": executable,
            "message": f"podman info could not complete: {type(exc).__name__}",
        }
    if result.returncode != 0:
        return {
            "status": "not_ready",
            "required_by": "Harvey LAB harness sandbox",
            "runtime": "podman",
            "podman_status": "unreachable",
            "executable": executable,
            "message": result.stderr.strip() or result.stdout.strip() or "podman info failed",
        }
    return {
        "status": "ready",
        "required_by": "Harvey LAB harness sandbox",
        "runtime": "podman",
        "podman_status": "present",
        "executable": executable,
        "message": "podman info succeeded.",
    }


def _harvey_host_document_reader_requirement(env: dict[str, str]) -> dict[str, object]:
    """Report whether Harvey's host-side evaluator can read docx deliverables."""
    path = str(env.get("PATH") or os.environ.get("PATH") or "")
    executable = shutil.which("pandoc", path=path)
    if executable is None:
        return {
            "status": "not_ready",
            "required_by": "Harvey LAB evaluator docx scoring",
            "reader": "pandoc",
            "pandoc_status": "missing",
            "executable": None,
            "message": (
                "Harvey evaluation.scoring reads .docx deliverables with host pandoc; "
                "without it, the judge scores conversion errors instead of agent output."
            ),
        }
    return {
        "status": "ready",
        "required_by": "Harvey LAB evaluator docx scoring",
        "reader": "pandoc",
        "pandoc_status": "present",
        "executable": executable,
        "message": "pandoc executable is available for Harvey docx scoring.",
    }


def _harvey_model_requirement(model: str, env: dict[str, str]) -> dict[str, object]:
    if model in {"HARVEY_GENERATOR_MODEL", "HARVEY_JUDGE_MODEL"}:
        return {
            "model": model,
            "provider": "unresolved",
            "credential_status": "unresolved",
            "missing_credentials": [],
        }
    provider = _harvey_model_provider(model)
    accepted_credentials = _harvey_provider_credentials(provider)
    missing_credentials = [
        credential
        for credential in accepted_credentials
        if not str(env.get(credential, "")).strip()
    ]
    credential_status = (
        "not_required"
        if not accepted_credentials
        else ("present" if len(missing_credentials) < len(accepted_credentials) else "missing")
    )
    if provider in {"openai-compatible", "baseten", "vllm"}:
        credential_status = (
            "present"
            if str(env.get("OPENAI_COMPATIBLE_API_KEY") or env.get("OPENAI_API_KEY") or "").strip()
            else "not_required"
        )
        missing_credentials = []
    return {
        "model": model,
        "provider": provider,
        "credential_status": credential_status,
        "accepted_credentials": accepted_credentials,
        "missing_credentials": missing_credentials if credential_status == "missing" else [],
        "endpoint": _harvey_model_endpoint(provider, env),
    }


def _harvey_model_provider(model: str) -> str:
    provider, model_id = model.split("/", 1) if "/" in model else (None, model)
    if provider in {"openai-compatible", "baseten", "vllm", "openai", "anthropic", "google", "mistral"}:
        return provider
    lowered = model_id.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "google"
    if lowered.startswith(("gpt", "o1", "o3", "o4", "o5")):
        return "openai"
    if lowered.startswith("mistral"):
        return "mistral"
    return "unknown"


def _harvey_provider_credentials(provider: str) -> list[str]:
    return {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "mistral": ["MISTRAL_API_KEY"],
    }.get(provider, [])


def _harvey_model_endpoint(provider: str, env: dict[str, str]) -> str | None:
    if provider in {"openai-compatible", "baseten", "vllm"}:
        return str(env.get("OPENAI_COMPATIBLE_BASE_URL") or "http://127.0.0.1:8318/v1")
    return None


def _harvey_status_run_artifacts_ready(
    files_ready: bool,
    metrics_path: Path,
    transcript_path: Path,
) -> tuple[bool, str | None]:
    if not files_ready:
        return False, None
    try:
        metrics = _json_object_from_file(metrics_path, "metrics")
    except ValueError:
        return False, "invalid_run_metrics"
    search_calls = _optional_int(metrics.get("memory_search_calls")) or 0
    if search_calls <= 0:
        return False, "memory_tools_not_used"
    if not _transcript_has_memory_tool_evidence(transcript_path):
        return False, "missing_transcript_memory_tool_evidence"
    return True, None


def _harvey_status_normalized_result_ready(
    path: Path,
    task_id: str,
    run_id: str,
) -> tuple[bool, str | None]:
    if not path.exists():
        return False, None
    try:
        loaded = load_harvey_zaxy_results(path)
    except ValueError:
        return False, "invalid_normalized_result"
    result = next((item for item in loaded if item.task_id == task_id), None)
    if result is None or result.run_id != run_id:
        return False, "normalized_result_mismatch"
    return True, None


def build_harvey_normalized_result_from_run(
    harvey_worktree: Path,
    *,
    run_id: str,
    task_id: str,
    manifest_path: Path,
    framework: str = "zaxy",
    judge_model: str | None = None,
    judge_reasoning_effort: str | None = None,
) -> dict[str, object]:
    """Build Harvey's normalized-result contract from one external run directory."""
    worktree = harvey_worktree.resolve()
    run_dir = worktree / "results" / run_id
    if not run_dir.exists():
        raise ValueError(f"Harvey run directory does not exist: {run_dir}")
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    scores_path = run_dir / "scores.json"
    transcript_path = run_dir / "transcript.jsonl"
    answer_path = _harvey_run_answer_path(worktree, run_dir, task_id)
    for required_path in (config_path, metrics_path, scores_path, transcript_path, answer_path):
        if not required_path.exists():
            raise ValueError(f"Harvey run artifact does not exist: {required_path}")
    manifest = _json_object_from_file(manifest_path, "manifest")
    config = _json_object_from_file(config_path, "config")
    metrics = _json_object_from_file(metrics_path, "metrics")
    scores = _json_object_from_file(scores_path, "scores")
    score = _score_fraction(scores)
    generator_model = _optional_str(config.get("model"))
    actual_task_id = str(config.get("task") or scores.get("task") or task_id)
    if actual_task_id != task_id:
        raise ValueError(f"Harvey run task mismatch: expected {task_id}, found {actual_task_id}")
    return {
        "schema_version": "harvey-memory-ablation-v1",
        "run_id": run_id,
        "framework": framework,
        "task_id": task_id,
        "corpus_hash": _optional_str(manifest.get("corpus_hash")),
        "branch": _git_branch_or_none(worktree),
        "commit": _git_commit_or_none(worktree),
        "models": {
            "generator": generator_model,
            "judge": judge_model or _optional_str(scores.get("judge_model")),
            "endpoint": None,
            "generator_reasoning_effort": _optional_str(config.get("reasoning_effort")),
            "judge_reasoning_effort": judge_reasoning_effort,
            "temperature": _optional_float(config.get("temperature")),
            "embedding": _optional_str(manifest.get("embedding_model")),
            "embedding_endpoint": _optional_str(manifest.get("embedding_endpoint")),
            "embedding_backend": _optional_str(manifest.get("embedding_backend")),
            "embedding_dimension": _optional_int(manifest.get("embedding_dimension")),
            "embedding_device": _optional_str(manifest.get("embedding_device")),
        },
        "paths": {
            "results_run_dir": _relative_path_string(run_dir, worktree),
            "answer": _relative_path_string(answer_path, worktree),
            "tool_log": _relative_path_string(transcript_path, worktree),
            "judge": _relative_path_string(scores_path, worktree),
            "run_metrics": _relative_path_string(metrics_path, worktree),
        },
        "scores": {
            "answer_correctness": score,
            "final_score": score,
        },
        "timing": {
            "ingest_seconds": None,
            "agent_runtime_seconds": _optional_float(metrics.get("wall_clock_seconds")),
            "judge_seconds": None,
            "total_seconds": _optional_float(metrics.get("wall_clock_seconds")),
        },
        "usage": {
            "generator_prompt_tokens": _optional_int(metrics.get("input_tokens")),
            "generator_completion_tokens": _optional_int(metrics.get("output_tokens")),
            "judge_prompt_tokens": None,
            "judge_completion_tokens": None,
            "embedding_tokens": None,
            "total_tokens": _optional_int(metrics.get("total_tokens")),
        },
        "cost": {
            "estimated_usd": None,
            "generator_estimated_usd": None,
            "judge_estimated_usd": None,
            "embedding_estimated_usd": None,
        },
        "tooling": {
            "tool_calls_total": _optional_int(metrics.get("tool_calls_total")),
            "memory_search_calls": _optional_int(metrics.get("memory_search_calls")),
            "memory_read_calls": _optional_int(metrics.get("memory_read_calls")),
            "empty_memory_searches": _optional_int(metrics.get("empty_memory_searches")),
        },
        "retrieval": {
            "citation_recall": None,
        },
        "failure_modes": [],
        "qualitative_notes": "Generated by zaxy harvey-lab-normalize-run from external Harvey run artifacts.",
    }


def _harvey_run_answer_path(worktree: Path, run_dir: Path, task_id: str) -> Path:
    output_dir = run_dir / "output"
    response_path = output_dir / "response.md"
    if response_path.exists():
        return response_path
    for expected_name in _harvey_task_expected_deliverables(worktree, task_id):
        expected_path = output_dir / expected_name
        if expected_path.exists():
            return expected_path
    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    if len(output_files) == 1:
        return output_files[0]
    return response_path


def _harvey_task_expected_deliverables(worktree: Path, task_id: str) -> tuple[str, ...]:
    task_path = worktree / "tasks" / task_id / "task.json"
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, dict):
        return ()
    names: list[str] = []
    deliverables = payload.get("deliverables")
    if isinstance(deliverables, dict):
        names.extend(
            value
            for value in deliverables.values()
            if isinstance(value, str) and value.strip()
        )
    elif isinstance(deliverables, list):
        names.extend(value for value in deliverables if isinstance(value, str) and value.strip())
    for criterion in payload.get("criteria", []):
        if not isinstance(criterion, dict):
            continue
        for value in criterion.get("deliverables", []):
            if isinstance(value, str) and value.strip():
                names.append(value)
    return tuple(dict.fromkeys(names))


def build_harvey_external_run_manifest(
    *,
    generator: str,
    judge: str,
    reasoning_effort: str | None = "low",
    output_root: str = "results/memory-ablation/zaxy",
) -> dict[str, object]:
    """Return the reproducible external Harvey run plan for all article tasks."""
    tasks: list[dict[str, object]] = []
    for task in ARTICLE_TASKS:
        slug = task.task_id.replace("/", "__")
        index_dir = f".ingestion/indexes/{slug}/zaxy"
        normalized_root = f".ingestion/corpora/{slug}/txt"
        source_map = f".ingestion/corpora/{slug}/source-map.json"
        run_id = f"zaxy-{slug}"
        tasks.append(
            {
                "task_id": task.task_id,
                "article_label": task.article_label,
                "task_shape": task.task_shape,
                "document_count": task.document_count,
                "criteria_count": task.criteria_count,
                "index_command": (
                    "zaxy harvey-lab-index "
                    f"--normalized-corpus-root {normalized_root} "
                    f"--source-map {source_map} "
                    f"--output-dir {index_dir}"
                ),
                "run_command": (
                    f"HARVEY_MEMORY_MANIFEST={index_dir}/manifest.json "
                    "uv run python -m harness.run "
                    f"--model {generator} --task {task.task_id} --run-id {run_id}"
                    + (f" --reasoning-effort {reasoning_effort}" if reasoning_effort else "")
                ),
                "judge_command": (
                    "uv run python -m evaluation.run_eval "
                    f"--judge-model {judge} --run-id {run_id} --task {task.task_id}"
                ),
                "normalize_command": (
                    "zaxy harvey-lab-normalize-run "
                    "--harvey-worktree . "
                    f"--run-id {run_id} "
                    f"--task-id {task.task_id} "
                    f"--manifest {index_dir}/manifest.json"
                ),
                "validate_command": (
                    "uv run python scripts/memory_ablation/validate_result.py "
                    f"--run-dir .ingestion/runs/{run_id} --worktree-root ."
                ),
                "expected_normalized_result": (
                    f".ingestion/runs/{run_id}/normalized-result.json"
                ),
            }
        )
    return {
        "schema_version": "zaxy.harvey-lab-external-run.v1",
        "source_url": HARVEY_LAB_REPO_URL,
        "article_url": ARTICLE_URL,
        "generator": generator,
        "judge": judge,
        "reasoning_effort": reasoning_effort,
        "task_count": len(tasks),
        "doctor_command": "zaxy harvey-lab-doctor path/to/harvey-zaxy-worktree",
        "adapter_command": "zaxy harvey-lab-adapter-kit --output-dir reports/benchmarks/harvey-lab-adapter-kit",
        "status_command": "zaxy harvey-lab-status path/to/harvey-zaxy-worktree",
        "collection_command": (
            "uv run python scripts/memory_ablation/collect_results.py "
            "--worktree . --dedupe-latest "
            "--output .ingestion/reports/comparison-zaxy.json"
        ),
        "comparison_command": (
            "zaxy harvey-lab-import path/to/harvey-zaxy-worktree "
            "--output-dir reports/benchmarks/harvey-lab-memory-ablation"
        ),
        "report_json_path": "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json",
        "publish_output_path": "reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md",
        "validation_command": (
            "zaxy harvey-lab-validate "
            "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json "
            "--require-complete"
        ),
        "gate_command": (
            "zaxy harvey-lab-gate "
            "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json"
        ),
        "publish_command": (
            "zaxy harvey-lab-publish "
            "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json "
            "--output reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md"
        ),
        "tasks": tasks,
    }


def write_harvey_external_run_manifest(
    manifest: dict[str, object],
    output_dir: Path,
) -> WrittenHarveyExternalRunManifest:
    """Write the external Harvey run manifest as JSON, Markdown, and a shell runner."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "harvey-lab-external-run.json"
    markdown_path = output_dir / "harvey-lab-external-run.md"
    script_path = output_dir / "run-harvey-lab-zaxy.sh"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(external_run_manifest_to_markdown(manifest), encoding="utf-8")
    script_path.write_text(build_harvey_external_run_script(manifest), encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | 0o111)
    return WrittenHarveyExternalRunManifest(
        json_path=json_path,
        markdown_path=markdown_path,
        script_path=script_path,
    )


def build_harvey_external_run_script(manifest: dict[str, object]) -> str:
    """Render an executable script for producing external Harvey Zaxy rows."""
    generator = _bash_default_value(str(manifest.get("generator") or "HARVEY_GENERATOR_MODEL"))
    judge = _bash_default_value(str(manifest.get("judge") or "HARVEY_JUDGE_MODEL"))
    reasoning_effort = manifest.get("reasoning_effort")
    reasoning_args = (
        f" --reasoning-effort {shlex.quote(str(reasoning_effort))}"
        if isinstance(reasoning_effort, str) and reasoning_effort.strip()
        else ""
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'HARVEY_WORKTREE="${1:-.}"',
        'OUTPUT_DIR="${2:-reports/benchmarks/harvey-lab-memory-ablation}"',
        'TASK_FILTER="${3:-${HARVEY_TASK_FILTER:-}}"',
        f'GENERATOR_MODEL="${{HARVEY_GENERATOR_MODEL:-{generator}}}"',
        f'JUDGE_MODEL="${{HARVEY_JUDGE_MODEL:-{judge}}}"',
        'JUDGE_PARALLEL="${HARVEY_JUDGE_PARALLEL:-1}"',
        'OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"',
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'ZAXY_WORKTREE="$(cd "$SCRIPT_DIR/../../.." && pwd)"',
        'ZAXY_PYTHONPATH="$ZAXY_WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"',
        'SOURCE_MANIFEST_JSON="$SCRIPT_DIR/harvey-lab-external-run.json"',
        'RUN_MANIFEST_JSON="$OUTPUT_DIR/harvey-lab-external-run.json"',
        'READY_JSON="$OUTPUT_DIR/harvey-lab-ready.json"',
        'STATUS_JSON="$OUTPUT_DIR/harvey-lab-status.json"',
        'HARVEY_COMPARISON_JSON=".ingestion/reports/comparison-zaxy.json"',
        'if [[ "$GENERATOR_MODEL" == "HARVEY_GENERATOR_MODEL" || "$JUDGE_MODEL" == "HARVEY_JUDGE_MODEL" ]]; then',
        '  echo "Unresolved Harvey model placeholders; set HARVEY_GENERATOR_MODEL and HARVEY_JUDGE_MODEL or generate the plan with explicit models." >&2',
        "  exit 2",
        "fi",
        'HARVEY_WORKTREE="$(cd "$HARVEY_WORKTREE" && pwd)"',
        'if [[ "$SOURCE_MANIFEST_JSON" != "$RUN_MANIFEST_JSON" ]]; then',
        '  cp "$SOURCE_MANIFEST_JSON" "$RUN_MANIFEST_JSON"',
        "fi",
        'ADAPTER_DIR="$(mktemp -d)"',
        'HARVEY_ADAPTER_PATH="$HARVEY_WORKTREE/scripts/memory_ablation/raw_rg_memory.py"',
        'HARVEY_ADAPTER_BACKUP="$ADAPTER_DIR/raw_rg_memory.py.original"',
        'HARVEY_ADAPTER_HAD_ORIGINAL=0',
        'restore_harvey_adapter() {',
        '  if [[ "$HARVEY_ADAPTER_HAD_ORIGINAL" == "1" && -f "$HARVEY_ADAPTER_BACKUP" ]]; then',
        '    cp "$HARVEY_ADAPTER_BACKUP" "$HARVEY_ADAPTER_PATH"',
        "  else",
        '    rm -f "$HARVEY_ADAPTER_PATH"',
        "  fi",
        '  rm -rf "$ADAPTER_DIR"',
        "}",
        'trap restore_harvey_adapter EXIT',
        "",
        'zaxy harvey-lab-doctor "$HARVEY_WORKTREE"',
        'zaxy harvey-lab-preflight "$HARVEY_WORKTREE" --task-filter "$TASK_FILTER"',
        'zaxy harvey-lab-ready "$HARVEY_WORKTREE" --generator "$GENERATOR_MODEL" --judge "$JUDGE_MODEL" --task-filter "$TASK_FILTER" --json | tee "$READY_JSON"',
        'zaxy harvey-lab-adapter-kit --output-dir "$ADAPTER_DIR"',
        'mkdir -p "$HARVEY_WORKTREE/scripts/memory_ablation"',
        'if [[ -f "$HARVEY_ADAPTER_PATH" ]]; then',
        '  cp "$HARVEY_ADAPTER_PATH" "$HARVEY_ADAPTER_BACKUP"',
        '  HARVEY_ADAPTER_HAD_ORIGINAL=1',
        "fi",
        'cp "$ADAPTER_DIR/raw_rg_memory.py" "$HARVEY_ADAPTER_PATH"',
        'cd "$HARVEY_WORKTREE"',
        "",
    ]
    tasks = manifest.get("tasks")
    if isinstance(tasks, list):
        for item in tasks:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or "")
            if not task_id:
                continue
            slug = _harvey_task_slug(task_id)
            run_id = str(item.get("run_id") or f"zaxy-{slug}")
            lines.extend(
                [
                    f"TASK_ID={shlex.quote(task_id)}",
                    f"RUN_ID={shlex.quote(run_id)}",
                    f"SLUG={shlex.quote(slug)}",
                    'if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then',
                    '  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"',
                    "else",
                    'INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"',
                    'NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<\'PY\'',
                    "import json",
                    "import sys",
                    "from pathlib import Path",
                    "import scripts.memory_ablation.normalize_corpus as normalize_corpus",
                    "",
                    "task_id = sys.argv[1]",
                    "root = Path.cwd()",
                    'normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")',
                    "print(json.dumps(normalization))",
                    "PY",
                    ')"',
                    'NORMALIZED_ROOT="$(printf \'%s\' "$NORMALIZATION_JSON" | python -c \'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])\')"',
                    'SOURCE_MAP="$(printf \'%s\' "$NORMALIZATION_JSON" | python -c \'import json, sys; print(json.load(sys.stdin)["source_map_path"])\')"',
                    'zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"',
                    (
                        'HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" '
                        'PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" '
                        f'--task "$TASK_ID" --run-id "$RUN_ID"{reasoning_args}'
                    ),
                    (
                        'uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" '
                        '--run-id "$RUN_ID" --task "$TASK_ID"'
                    ),
                    'zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"',
                    'uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"',
                    "fi",
                    "",
                ]
            )
    lines.extend(
        [
            'if [[ -n "$TASK_FILTER" ]]; then',
            '  zaxy harvey-lab-status "$HARVEY_WORKTREE" --json | tee "$STATUS_JSON" || true',
            '  uv run python scripts/memory_ablation/collect_results.py --worktree "$HARVEY_WORKTREE" --dedupe-latest --output "$HARVEY_COMPARISON_JSON" || true',
            "else",
            '  zaxy harvey-lab-status "$HARVEY_WORKTREE" --json | tee "$STATUS_JSON"',
            '  uv run python scripts/memory_ablation/collect_results.py --worktree "$HARVEY_WORKTREE" --dedupe-latest --output "$HARVEY_COMPARISON_JSON"',
            "fi",
            'zaxy harvey-lab-import "$HARVEY_WORKTREE" --output-dir "$OUTPUT_DIR"',
            'REPORT_JSON="$OUTPUT_DIR/harvey-lab-benchmark.json"',
            'PUBLISH_MD="$OUTPUT_DIR/publishable-statistics.md"',
            'if [[ -n "$TASK_FILTER" ]]; then',
            '  zaxy harvey-lab-validate "$REPORT_JSON" || true',
            '  echo "Filtered Harvey run imported; full publish gate is intentionally skipped until all tasks are complete."',
            "else",
            '  zaxy harvey-lab-validate "$REPORT_JSON" --require-complete',
            '  zaxy harvey-lab-gate "$REPORT_JSON"',
            '  zaxy harvey-lab-publish "$REPORT_JSON" --output "$PUBLISH_MD"',
            "fi",
            "",
        ]
    )
    return "\n".join(lines)


def _bash_default_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")


def external_run_manifest_to_markdown(manifest: dict[str, object]) -> str:
    """Render an external Harvey run manifest as Markdown."""
    lines = [
        "# Harvey LAB External Run Manifest",
        "",
        f"- Source: `{manifest.get('source_url')}`",
        f"- Article: `{manifest.get('article_url')}`",
        f"- Generator: `{manifest.get('generator')}`",
        f"- Judge: `{manifest.get('judge')}`",
        f"- Task count: `{manifest.get('task_count')}`",
        "",
        "## Doctor",
        "",
        f"```bash\n{manifest.get('doctor_command')}\n```",
        "",
        "## Adapter",
        "",
        f"```bash\n{manifest.get('adapter_command')}\n```",
        "",
        "## Tasks",
        "",
        "| Task | Shape | Index | Run | Judge | Normalize | Validate |",
        "|------|-------|-------|-----|-------|-----------|----------|",
    ]
    tasks = manifest.get("tasks")
    if isinstance(tasks, list):
        for item in tasks:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"{item.get('task_id')} | "
                f"{item.get('task_shape')} | "
                f"`{item.get('index_command')}` | "
                f"`{item.get('run_command')}` | "
                f"`{item.get('judge_command')}` | "
                f"`{item.get('normalize_command')}` | "
                f"`{item.get('validate_command')}` |"
            )
    lines.extend(
        [
            "",
            "## Collect",
            "",
            f"```bash\n{manifest.get('collection_command')}\n```",
            "",
            "## Status",
            "",
            f"```bash\n{manifest.get('status_command')}\n```",
            "",
            "## Compare",
            "",
            f"```bash\n{manifest.get('comparison_command')}\n```",
            "",
            "## Validate And Publish",
            "",
            f"Report JSON: `{manifest.get('report_json_path')}`",
            "",
            f"Publish output: `{manifest.get('publish_output_path')}`",
            "",
            f"```bash\n{manifest.get('validation_command')}\n```",
            "",
            f"```bash\n{manifest.get('gate_command')}\n```",
            "",
            f"```bash\n{manifest.get('publish_command')}\n```",
        ]
    )
    return "\n".join(lines) + "\n"


def load_harvey_lab_report(path: Path) -> HarveyLabReport:
    """Load a Zaxy Harvey LAB report JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Harvey LAB report JSON must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Harvey LAB report has unsupported schema_version")
    rows_payload = _dict(payload.get("task_rows"), "task_rows")
    fit_payload = _dict(payload.get("framework_fit"), "framework_fit")
    scorecard_payload = _dict(payload.get("framework_scorecard", {}), "framework_scorecard")
    external_baseline_payload = _dict(
        payload.get("external_baseline_scorecard", {}),
        "external_baseline_scorecard",
    )
    external_comparison_payload = _dict(
        payload.get("external_comparison_scorecard", {}),
        "external_comparison_scorecard",
    )
    return HarveyLabReport(
        schema_version=SCHEMA_VERSION,
        generated_at=str(payload.get("generated_at", "")),
        status=str(payload.get("status", "")),
        external_suite={
            str(key): str(value)
            for key, value in _dict(payload.get("external_suite"), "external_suite").items()
        },
        result_provenance=_dict(payload.get("result_provenance", {}), "result_provenance"),
        summary=_summary_from_payload(_dict(payload.get("summary"), "summary")),
        task_rows={
            str(key): _task_comparison_from_payload(_dict(value, f"task_rows.{key}"))
            for key, value in rows_payload.items()
        },
        framework_scorecard={
            str(key): _framework_scorecard_from_payload(_dict(value, f"framework_scorecard.{key}"))
            for key, value in scorecard_payload.items()
        },
        external_baseline_scorecard={
            str(key): _external_baseline_scorecard_from_payload(
                _dict(value, f"external_baseline_scorecard.{key}")
            )
            for key, value in external_baseline_payload.items()
        },
        external_comparison_scorecard={
            str(key): _external_comparison_scorecard_from_payload(
                _dict(value, f"external_comparison_scorecard.{key}")
            )
            for key, value in external_comparison_payload.items()
        },
        framework_fit={
            str(key): _framework_fit_from_payload(_dict(value, f"framework_fit.{key}"))
            for key, value in fit_payload.items()
        },
        zaxy_results=tuple(
            _zaxy_result_from_payload(_dict(item, "zaxy_results[]"))
            for item in payload.get("zaxy_results", [])
        ),
        caveats=tuple(str(item) for item in payload.get("caveats", [])),
    )


def check_harvey_lab_completion(report: HarveyLabReport) -> dict[str, object]:
    """Return a public-claim gate for completed external Harvey LAB evidence."""
    completed = sorted(
        task_id
        for task_id, row in report.task_rows.items()
        if row.zaxy_score is not None
    )
    expected = [task.task_id for task in ARTICLE_TASKS]
    missing = [task_id for task_id in expected if task_id not in completed]
    evidence_failures = [
        *_harvey_evidence_failures(report),
        *_harvey_provenance_failures(report, require_complete=True),
    ]
    status = (
        "passed"
        if not missing and not evidence_failures and report.status == "complete"
        else "blocked"
    )
    zaxy_external = report.external_comparison_scorecard.get("Zaxy")
    zaxy_fit = report.framework_fit.get("Zaxy")
    return {
        "schema_version": "zaxy.harvey-lab-completion-gate.v1",
        "status": status,
        "completed_task_ids": completed,
        "missing_task_ids": missing,
        "evidence_failures": evidence_failures,
        "evidence_audit": _harvey_evidence_audit(report, evidence_failures),
        "zaxy_mean_score": report.summary.zaxy_mean_score,
        "article_best_mean_for_zaxy_tasks": report.summary.article_best_mean_for_zaxy_tasks,
        "mean_delta_vs_regular_no_memory": report.summary.mean_delta_vs_regular_no_memory,
        "mean_delta_vs_article_best": report.summary.mean_delta_vs_article_best,
        "zaxy_external_rank": None if zaxy_external is None else zaxy_external.rank_by_mean_score,
        "zaxy_delta_vs_raw_rg": None if zaxy_external is None else zaxy_external.delta_vs_raw_rg,
        "zaxy_delta_vs_best_external": (
            None if zaxy_external is None else zaxy_external.delta_vs_best_external
        ),
        "zaxy_framework_fit": None if zaxy_fit is None else zaxy_fit.interpretation,
        "zaxy_run_configuration": _zaxy_run_configuration(report),
        "message": (
            "All ten Harvey LAB article tasks have externally judged Zaxy rows."
            if status == "passed"
            else "External Harvey LAB benchmark is incomplete or lacks reviewable memory-use evidence; do not publish Zaxy comparative claims."
        ),
    }


def _zaxy_run_configuration(report: HarveyLabReport) -> dict[str, object | None]:
    result = next(iter(report.zaxy_results), None)
    return {
        "generator": None if result is None else result.generator,
        "judge": None if result is None else result.judge,
        "generator_reasoning_effort": None if result is None else result.generator_reasoning_effort,
        "judge_reasoning_effort": None if result is None else result.judge_reasoning_effort,
        "temperature": None if result is None else result.temperature,
        "harvey_git_commit": _optional_str(report.result_provenance.get("harvey_git_commit")),
    }


def validate_harvey_lab_report(
    report: HarveyLabReport,
    *,
    require_complete: bool = False,
) -> dict[str, object]:
    """Validate Harvey LAB report evidence without necessarily requiring all tasks."""
    completed = sorted(
        task_id
        for task_id, row in report.task_rows.items()
        if row.zaxy_score is not None
    )
    expected = [task.task_id for task in ARTICLE_TASKS]
    missing = [task_id for task_id in expected if task_id not in completed]
    evidence_failures = [
        *_harvey_evidence_failures(report),
        *_harvey_provenance_failures(report, require_complete=require_complete),
    ]
    if require_complete and missing:
        evidence_failures.append(
            {
                "task_id": "*",
                "reason": "missing_required_tasks",
            }
        )
    status = "valid" if not evidence_failures else "invalid"
    return {
        "schema_version": "zaxy.harvey-lab-validation.v1",
        "status": status,
        "require_complete": require_complete,
        "completed_task_count": len(completed),
        "expected_task_count": len(expected),
        "completed_task_ids": completed,
        "missing_task_ids": missing,
        "evidence_failures": evidence_failures,
        "evidence_audit": _harvey_evidence_audit(report, evidence_failures),
        "message": (
            "Harvey LAB report evidence is valid for the requested validation scope."
            if status == "valid"
            else "Harvey LAB report evidence is incomplete or not reviewable."
        ),
    }


def _harvey_evidence_audit(
    report: HarveyLabReport,
    evidence_failures: list[dict[str, str]],
) -> dict[str, object]:
    failure_reasons = {
        str(failure.get("reason") or "")
        for failure in evidence_failures
    }
    normalized_result_count = len(
        _list_value(report.result_provenance.get("normalized_result_paths"))
    )
    external_comparison_count = len(
        _list_value(report.result_provenance.get("external_baseline_report_paths"))
    )
    external_run_manifest_count = len(
        _list_value(report.result_provenance.get("external_run_manifest_paths"))
    )
    external_readiness_report_count = len(
        _list_value(report.result_provenance.get("external_readiness_report_paths"))
    )
    external_status_report_count = len(
        _list_value(report.result_provenance.get("external_status_report_paths"))
    )
    has_zaxy_artifact_evidence = normalized_result_count > 0
    return {
        "normalized_result_artifacts": normalized_result_count,
        "external_comparison_artifacts": external_comparison_count,
        "judge_score_artifacts_match": has_zaxy_artifact_evidence
        and "judge_score_mismatch" not in failure_reasons,
        "run_metrics_artifacts_match": has_zaxy_artifact_evidence
        and "run_metrics_mismatch" not in failure_reasons,
        "transcript_memory_tool_evidence": has_zaxy_artifact_evidence
        and "missing_transcript_memory_tool_evidence" not in failure_reasons,
        "external_comparison_recomputed_from_non_zaxy_rows": external_comparison_count > 0
        and _external_comparison_has_full_suite_result_evidence(
            report.result_provenance,
            expected_run_config=_zaxy_run_configuration(report),
        )
        and (
            "missing_external_comparison_result_evidence" not in failure_reasons
            and "stale_external_comparison_result_evidence" not in failure_reasons
            and "external_comparison_not_full_suite" not in failure_reasons
            and "external_comparison_model_mismatch" not in failure_reasons
        ),
        "external_run_manifest_artifacts": external_run_manifest_count,
        "external_readiness_report_artifacts": external_readiness_report_count,
        "external_status_report_artifacts": external_status_report_count,
        "external_run_audit_artifacts_valid": _external_run_audit_artifact_failure(
            report.result_provenance,
            report=report,
        )
        is None,
    }


def harvey_memory_search(
    manifest: HarveyMemoryManifest,
    query: str,
    *,
    limit: int = 5,
) -> dict[str, object]:
    """Search a Zaxy Harvey Eventloom index using Harvey memory tool JSON."""
    index = VerbatimIndex.from_event_logs([EventLog(_manifest_eventloom_path(manifest))])
    hits: list[dict[str, object]] = []
    for hit in index.query(query, limit=limit):
        metadata = dict(hit.metadata)
        source_path = str(metadata.get("source_path") or "")
        display_path = _display_source_path(manifest, source_path)
        start_line = _int_or_none(metadata.get("source_start_line"))
        end_line = _int_or_none(metadata.get("source_end_line"))
        item_id = _memory_item_id(display_path, start_line, end_line)
        hits.append(
            {
                "id": item_id,
                "source_path": display_path,
                "start_line": start_line,
                "end_line": end_line,
                "score": hit.score,
                "snippet": hit.content,
                "citation": hit.citation,
                "metadata": {
                    **metadata,
                    "storage_source_path": source_path,
                    "source_kind": hit.source_kind,
                },
            }
        )
    return {
        "framework": "zaxy",
        "query": query,
        "hits": hits,
        "source_identity": _source_identity(manifest),
    }


def harvey_memory_read(
    manifest: HarveyMemoryManifest,
    item_id: str,
    *,
    context_lines: int = 8,
) -> dict[str, object]:
    """Read context for a Harvey memory hit id returned by `harvey_memory_search`."""
    source_path, start_line, end_line = _parse_memory_item_id(item_id)
    storage_path = _storage_source_path(manifest, source_path)
    corpus_root = Path(str(_normalized_text(manifest).get("corpus_root") or manifest["corpus_root"]))
    file_path = corpus_root / storage_path
    if not file_path.exists():
        raise ValueError(f"memory_read id not found: {item_id}")
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start_index = max(0, start_line - 1 - max(0, context_lines))
    end_index = min(len(lines), end_line + max(0, context_lines))
    if start_index >= len(lines):
        raise ValueError(f"memory_read id not found: {item_id}")
    display_path = _display_source_path(manifest, storage_path)
    return {
        "id": _memory_item_id(display_path, start_line, end_line),
        "source_path": display_path,
        "start_line": start_index + 1,
        "end_line": end_index,
        "content": "\n".join(lines[start_index:end_index]),
        "source_identity": _source_identity(manifest),
    }


def _write_harvey_index_artifacts(output_dir: Path, manifest: HarveyMemoryManifest) -> None:
    manifest_path = output_dir / "manifest.json"
    artifact_summary = {
        "schema_version": "zaxy.harvey-lab-artifact-summary.v1",
        "framework": "zaxy",
        "manifest_path": str(manifest_path),
        "corpus_hash": manifest.get("corpus_hash"),
        "adapter_contract": manifest.get("adapter_contract"),
        "artifact_counts": {
            "files": manifest.get("file_count"),
            "eventloom_events": manifest.get("event_count"),
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "eventloom": manifest.get("eventloom_path"),
            "artifact_summary": str(output_dir / "artifact-summary.json"),
            "smoke_result": str(output_dir / "smoke-result.json"),
        },
        "normalized_text": _normalized_text(manifest),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    smoke_result = _build_harvey_index_smoke_result(manifest)
    (output_dir / "artifact-summary.json").write_text(
        json.dumps(artifact_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "smoke-result.json").write_text(
        json.dumps(smoke_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_harvey_index_smoke_result(manifest: HarveyMemoryManifest) -> dict[str, object]:
    query = _smoke_query(manifest)
    try:
        search = harvey_memory_search(manifest, query, limit=1)
        hits = search.get("hits")
        if not isinstance(hits, list) or not hits:
            return {
                "schema_version": "zaxy.harvey-lab-smoke-result.v1",
                "framework": "zaxy",
                "ok": False,
                "query": query,
                "search": search,
                "error": "memory_search returned no hits",
            }
        first_hit = hits[0]
        if not isinstance(first_hit, dict) or not isinstance(first_hit.get("id"), str):
            return {
                "schema_version": "zaxy.harvey-lab-smoke-result.v1",
                "framework": "zaxy",
                "ok": False,
                "query": query,
                "search": search,
                "error": "memory_search returned malformed hit",
            }
        read = harvey_memory_read(manifest, first_hit["id"], context_lines=1)
        return {
            "schema_version": "zaxy.harvey-lab-smoke-result.v1",
            "framework": "zaxy",
            "ok": True,
            "query": query,
            "search": search,
            "read": read,
        }
    except Exception as exc:  # pragma: no cover - defensive artifact context
        return {
            "schema_version": "zaxy.harvey-lab-smoke-result.v1",
            "framework": "zaxy",
            "ok": False,
            "query": query,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _smoke_query(manifest: HarveyMemoryManifest) -> str:
    corpus_root_value = _normalized_text(manifest).get("corpus_root") or manifest.get("corpus_root")
    if not isinstance(corpus_root_value, str):
        return "document"
    corpus_root = Path(corpus_root_value)
    for path in sorted(corpus_root.rglob("*.txt")):
        if not path.is_file():
            continue
        words = [
            word.strip(".,;:()[]{}\"'").casefold()
            for word in path.read_text(encoding="utf-8", errors="replace").split()
        ]
        candidates = [
            word
            for word in words
            if len(word) >= 5 and any(char.isalpha() for char in word)
        ]
        if candidates:
            return " ".join(dict.fromkeys(candidates[:3]))
    return "document"


def build_harvey_lab_report(
    zaxy_results: list[dict[str, object]] | tuple[HarveyZaxyResult, ...],
    *,
    result_provenance: dict[str, object] | None = None,
) -> HarveyLabReport:
    """Build article-relative Harvey LAB statistics for externally run Zaxy rows."""
    parsed_results = tuple(
        result if isinstance(result, HarveyZaxyResult) else _parse_zaxy_result(result)
        for result in zaxy_results
    )
    latest_by_task: dict[str, HarveyZaxyResult] = {}
    for result in parsed_results:
        latest_by_task[result.task_id] = result
    rows: dict[str, HarveyTaskComparison] = {}
    zaxy_rows: list[HarveyTaskComparison] = []
    for task in ARTICLE_TASKS:
        task_result = latest_by_task.get(task.task_id)
        row = _build_task_row(task, task_result)
        rows[task.task_id] = row
        if task_result is not None:
            zaxy_rows.append(row)
    status = "complete" if len(zaxy_rows) == len(ARTICLE_TASKS) else "partial"
    summary = _build_summary(status, zaxy_rows)
    framework_scorecard = _build_framework_scorecard(zaxy_rows)
    provenance = result_provenance or {}
    caveats = (
        "Scores are criterion pass rates from Harvey LAB judging, not binary all-pass task success.",
        "Article comparison rows are published external disclosures, not Zaxy same-process reruns.",
        "Framework scorecard rows for article systems use published task-winner coverage, not full hidden per-framework averages.",
        "raw-rg is a retrieval/search baseline, not the no-memory baseline.",
        "Zaxy rows must come from Harvey normalized result artifacts; internal LongMemEval results are rejected.",
    )
    external_baseline_scorecard = _build_external_baseline_scorecard(provenance)
    return HarveyLabReport(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(UTC).isoformat(),
        status=status,
        external_suite={
            "name": "Harvey LAB memory retrieval ablation",
            "source_url": HARVEY_LAB_REPO_URL,
            "article_url": ARTICLE_URL,
        },
        result_provenance=provenance,
        summary=summary,
        task_rows=rows,
        framework_scorecard=framework_scorecard,
        external_baseline_scorecard=external_baseline_scorecard,
        external_comparison_scorecard=_build_external_comparison_scorecard(
            summary,
            external_baseline_scorecard,
        ),
        framework_fit=_build_framework_fit(zaxy_rows, summary),
        zaxy_results=parsed_results,
        caveats=caveats,
    )


def write_harvey_lab_report(report: HarveyLabReport, output_dir: Path) -> WrittenHarveyLabReport:
    """Write Harvey LAB benchmark JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "harvey-lab-benchmark.json"
    markdown_path = output_dir / "harvey-lab-benchmark.md"
    json_path.write_text(
        json.dumps(_report_payload(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
    return WrittenHarveyLabReport(json_path=json_path, markdown_path=markdown_path)


def report_to_markdown(report: HarveyLabReport) -> str:
    """Render a Harvey LAB benchmark report as Markdown."""
    summary = report.summary
    lines = [
        "# Harvey LAB External Memory Benchmark",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Status: `{report.status}`",
        f"- External suite: [{report.external_suite['name']}]({report.external_suite['source_url']})",
        f"- Article: [{ARTICLE_URL}]({ARTICLE_URL})",
        f"- Zaxy tasks scored: `{summary.zaxy_task_count}/{summary.article_task_count}`",
        "",
        "## Result Provenance",
        "",
        f"- Source: `{report.result_provenance.get('source', '')}`",
        f"- Harvey git commit: `{report.result_provenance.get('harvey_git_commit', '')}`",
        (
            "- Provenance roots: "
            f"`{len(_list_value(report.result_provenance.get('roots')))}` "
            "(exact paths are stored in the JSON report)"
        ),
        (
            "- Normalized result artifacts: "
            f"`{len(_list_value(report.result_provenance.get('normalized_result_paths')))}`"
        ),
        (
            "- External baseline reports: "
            f"`{_harvey_report_path_labels(report.result_provenance.get('external_baseline_report_paths'))}`"
        ),
        (
            "- External run manifests: "
            f"`{_harvey_report_path_labels(report.result_provenance.get('external_run_manifest_paths'))}`"
        ),
        (
            "- External readiness reports: "
            f"`{_harvey_report_path_labels(report.result_provenance.get('external_readiness_report_paths'))}`"
        ),
        (
            "- External status reports: "
            f"`{_harvey_report_path_labels(report.result_provenance.get('external_status_report_paths'))}`"
        ),
        "",
        "## Summary",
        "",
        "| System | Tasks | Mean criterion pass rate | Delta vs regular | Delta vs article best | Wins vs article best |",
        "|--------|------:|--------------------------:|-----------------:|----------------------:|---------------------:|",
        (
            "| Zaxy | "
            f"{summary.zaxy_task_count} | "
            f"{_fmt(summary.zaxy_mean_score)} | "
            f"{_fmt(summary.mean_delta_vs_regular_no_memory, signed=True)} | "
            f"{_fmt(summary.mean_delta_vs_article_best, signed=True)} | "
            f"{summary.zaxy_task_wins} |"
        ),
        "",
        "## Zaxy Runtime And Usage",
        "",
        "| Mean total seconds | Total tokens | Memory search calls | Memory read calls |",
        "|-------------------:|-------------:|--------------------:|------------------:|",
        (
            "| "
            f"{_fmt(summary.zaxy_mean_total_seconds)} | "
            f"{_fmt_int(summary.zaxy_total_tokens)} | "
            f"{_fmt_int(summary.zaxy_total_memory_search_calls)} | "
            f"{_fmt_int(summary.zaxy_total_memory_read_calls)} |"
        ),
        "",
        "## Task Comparison",
        "",
        "| Task | Shape | Regular | Article best | Zaxy | Delta vs regular | Delta vs best | Winner |",
        "|------|-------|--------:|--------------|------:|-----------------:|--------------:|--------|",
    ]
    for task in ARTICLE_TASKS:
        row = report.task_rows[task.task_id]
        winner = "" if row.zaxy_winner is None else ("Zaxy" if row.zaxy_winner else row.article_best_framework)
        lines.append(
            "| "
            f"{row.article_label} | "
            f"{row.task_shape} | "
            f"{row.regular_no_memory_score:.3f} | "
            f"{row.article_best_framework} {row.article_best_score:.3f} | "
            f"{_fmt(row.zaxy_score)} | "
            f"{_fmt(row.zaxy_delta_vs_regular_no_memory, signed=True)} | "
            f"{_fmt(row.zaxy_delta_vs_article_best, signed=True)} | "
            f"{winner} |"
        )
    lines.extend(
        [
            "",
            "## Framework Scorecard",
            "",
            "| Framework | Evidence scope | Tasks | Mean score | Delta vs regular | Zaxy overlap | Zaxy delta on overlap |",
            "|-----------|----------------|------:|-----------:|-----------------:|-------------:|----------------------:|",
        ]
    )
    for framework in _framework_scorecard_order():
        scorecard = report.framework_scorecard.get(framework)
        if scorecard is None:
            continue
        lines.append(
            "| "
            f"{scorecard.framework} | "
            f"{scorecard.evidence_scope} | "
            f"{scorecard.article_task_count} | "
            f"{_fmt(scorecard.mean_score)} | "
            f"{_fmt(scorecard.mean_delta_vs_regular_no_memory, signed=True)} | "
            f"{scorecard.zaxy_overlap_task_count} | "
            f"{_fmt(scorecard.zaxy_delta_on_overlap, signed=True)} |"
        )
    _append_external_baseline_aggregate_markdown(lines, report)
    _append_external_comparison_markdown(lines, report)
    lines.extend(
        [
            "",
            "## Framework Fit",
            "",
            "| Framework | Where strongest | Example tasks | Interpretation |",
            "|-----------|-----------------|---------------|----------------|",
        ]
    )
    for framework in [*ARTICLE_FRAMEWORK_FIT.keys(), "Zaxy"]:
        fit = report.framework_fit[framework]
        lines.append(
            "| "
            f"{fit.framework} | "
            f"{fit.where_strongest} | "
            f"{fit.example_tasks} | "
            f"{fit.interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in report.caveats:
        lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"


def render_harvey_publication_markdown(report: HarveyLabReport) -> str:
    """Render gated public Harvey LAB comparative statistics."""
    gate = check_harvey_lab_completion(report)
    if gate["status"] != "passed":
        raise ValueError("Harvey LAB report is not publishable; completion gate did not pass")
    summary = report.summary
    run_config = _dict(gate.get("zaxy_run_configuration", {}), "zaxy_run_configuration")
    evidence_audit = _dict(gate.get("evidence_audit", {}), "evidence_audit")
    external_comparison_evidence = (
        "Full-suite native non-Zaxy comparison rows recomputed for rank, score, run-count, and latency evidence"
        if evidence_audit.get("external_comparison_recomputed_from_non_zaxy_rows") is True
        else (
            "Native non-Zaxy comparison artifacts are partial context; full-suite article-relative "
            "comparison uses the published article scorecard"
        )
    )
    lines = [
        "# Harvey LAB Publishable Comparative Statistics",
        "",
        f"- External suite: [{report.external_suite['name']}]({report.external_suite['source_url']})",
        f"- Article: [{ARTICLE_URL}]({ARTICLE_URL})",
        f"- Zaxy task coverage: `{summary.zaxy_task_count}/{summary.article_task_count}`",
        "- Metric: criterion pass rate, not binary Harvey all-pass score.",
        "",
        "## Zaxy Summary",
        "",
        "| System | Tasks | Mean criterion pass rate | Delta vs regular | Delta vs article best | Wins vs article best |",
        "|--------|------:|--------------------------:|-----------------:|----------------------:|---------------------:|",
        (
            "| Zaxy | "
            f"{summary.zaxy_task_count} | "
            f"{_fmt(summary.zaxy_mean_score)} | "
            f"{_fmt(summary.mean_delta_vs_regular_no_memory, signed=True)} | "
            f"{_fmt(summary.mean_delta_vs_article_best, signed=True)} | "
            f"{summary.zaxy_task_wins} |"
        ),
        "",
        "## Zaxy Runtime And Usage",
        "",
        "| Mean total seconds | Total tokens | Memory search calls | Memory read calls |",
        "|-------------------:|-------------:|--------------------:|------------------:|",
        (
            "| "
            f"{_fmt(summary.zaxy_mean_total_seconds)} | "
            f"{_fmt_int(summary.zaxy_total_tokens)} | "
            f"{_fmt_int(summary.zaxy_total_memory_search_calls)} | "
            f"{_fmt_int(summary.zaxy_total_memory_read_calls)} |"
        ),
        "",
        "## Zaxy Run Configuration",
        "",
        "| Generator | Judge | Generator reasoning effort | Judge reasoning effort | Temperature | Harvey commit |",
        "|-----------|-------|----------------------------|------------------------|------------:|---------------|",
        (
            "| "
            f"{run_config.get('generator') or ''} | "
            f"{run_config.get('judge') or ''} | "
            f"{run_config.get('generator_reasoning_effort') or ''} | "
            f"{run_config.get('judge_reasoning_effort') or ''} | "
            f"{_fmt(_optional_float(run_config.get('temperature')))} | "
            f"{run_config.get('harvey_git_commit') or ''} |"
        ),
        "",
        "## Zaxy External Position",
        "",
        "| Rank vs external scored systems | Delta vs source raw-rg | Delta vs best external | Framework Fit |",
        "|--------------------------------:|-----------------------:|-----------------------:|---------------|",
        (
            "| "
            f"{gate['zaxy_external_rank']} | "
            f"{_fmt_na(_optional_float(gate.get('zaxy_delta_vs_raw_rg')), signed=True)} | "
            f"{_fmt_na(_optional_float(gate.get('zaxy_delta_vs_best_external')), signed=True)} | "
            f"{gate['zaxy_framework_fit']} |"
        ),
        "",
        "## Article-Relative Framework Scorecard",
        "",
        "| Framework | Evidence scope | Tasks | Mean score | Delta vs regular | Zaxy overlap | Zaxy delta on overlap |",
        "|-----------|----------------|------:|-----------:|-----------------:|-------------:|----------------------:|",
    ]
    for framework in _framework_scorecard_order():
        scorecard = report.framework_scorecard.get(framework)
        if scorecard is None:
            continue
        lines.append(
            "| "
            f"{scorecard.framework} | "
            f"{scorecard.evidence_scope} | "
            f"{scorecard.article_task_count} | "
            f"{_fmt(scorecard.mean_score)} | "
            f"{_fmt(scorecard.mean_delta_vs_regular_no_memory, signed=True)} | "
            f"{scorecard.zaxy_overlap_task_count} | "
            f"{_fmt(scorecard.zaxy_delta_on_overlap, signed=True)} |"
        )
    _append_external_baseline_aggregate_markdown(lines, report)
    _append_external_comparison_markdown(lines, report)
    lines.extend(
        [
            "",
            "## Task Results",
            "",
            "| Task | Regular | Article best | Zaxy | Delta vs best |",
            "|------|--------:|--------------|------:|--------------:|",
        ]
    )
    for task in ARTICLE_TASKS:
        row = report.task_rows[task.task_id]
        lines.append(
            "| "
            f"{row.article_label} | "
            f"{row.regular_no_memory_score:.3f} | "
            f"{row.article_best_framework} {row.article_best_score:.3f} | "
            f"{_fmt(row.zaxy_score)} | "
            f"{_fmt(row.zaxy_delta_vs_article_best, signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Framework Fit",
            "",
            "| Framework | Where strongest | Example tasks | Interpretation |",
            "|-----------|-----------------|---------------|----------------|",
        ]
    )
    for framework in [*ARTICLE_FRAMEWORK_FIT.keys(), "Zaxy"]:
        fit = report.framework_fit[framework]
        lines.append(
            "| "
            f"{fit.framework} | "
            f"{fit.where_strongest} | "
            f"{fit.example_tasks} | "
            f"{fit.interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Source: `{report.result_provenance.get('source', '')}`",
            f"- Harvey git commit: `{report.result_provenance.get('harvey_git_commit', '')}`",
            f"- Normalized result artifacts: `{len(_list_value(report.result_provenance.get('normalized_result_paths')))}`",
            f"- External comparison artifacts: `{len(_list_value(report.result_provenance.get('external_baseline_report_paths')))}`",
            "",
            "## Evidence Audit",
            "",
            "| Check | Evidence |",
            "|-------|----------|",
            (
                "| Zaxy normalized-result artifacts | "
                f"{len(_list_value(report.result_provenance.get('normalized_result_paths')))} reviewable artifacts reloaded and matched to report rows |"
            ),
            "| Judge score artifacts match normalized scores | Enforced for every completed Zaxy row |",
            "| Run metrics artifacts match memory-call totals | Enforced for every completed Zaxy row |",
            "| Transcripts contain memory tool evidence | Enforced for every completed Zaxy row |",
            f"| External comparison aggregates recomputed from non-Zaxy result rows | {external_comparison_evidence} |",
            (
                "| External run manifest artifacts | "
                f"{len(_list_value(report.result_provenance.get('external_run_manifest_paths')))} reviewable launch manifests |"
            ),
            (
                "| External readiness report artifacts | "
                f"{len(_list_value(report.result_provenance.get('external_readiness_report_paths')))} reviewable launch readiness reports |"
            ),
            (
                "| External status report artifacts | "
                f"{len(_list_value(report.result_provenance.get('external_status_report_paths')))} reviewable run status reports |"
            ),
            "| External run audit artifacts are complete and commit-consistent | Enforced before publishable statistics are rendered |",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_external_baseline_aggregate_markdown(
    lines: list[str],
    report: HarveyLabReport,
) -> None:
    if not report.external_baseline_scorecard:
        return
    lines.extend(
        [
            "",
            "## External Baseline Aggregate",
            "",
            "| Framework | Evidence scope | Runs | Mean score | Delta vs source raw-rg | Mean seconds | Source report |",
            "|-----------|----------------|-----:|-----------:|----------------:|-------------:|---------------|",
        ]
    )
    for baseline_scorecard in sorted(
        report.external_baseline_scorecard.values(),
        key=lambda item: (
            item.mean_score is None,
            -(item.mean_score or 0.0),
            item.framework,
        ),
    ):
        lines.append(
            "| "
            f"{baseline_scorecard.framework} | "
            f"{baseline_scorecard.evidence_scope} | "
            f"{baseline_scorecard.runs} | "
            f"{_fmt(baseline_scorecard.mean_score)} | "
            f"{_fmt(baseline_scorecard.delta_vs_raw_rg, signed=True)} | "
            f"{_fmt(baseline_scorecard.mean_total_seconds)} | "
            f"{_harvey_report_path_label(baseline_scorecard.source_report_path)} |"
        )


def _append_external_comparison_markdown(
    lines: list[str],
    report: HarveyLabReport,
) -> None:
    if not report.external_comparison_scorecard:
        return
    lines.extend(
        [
            "",
            "## Zaxy vs External Scored Systems",
            "",
            "| Framework | Evidence scope | Runs | Mean score | Delta vs source raw-rg | Delta vs best external | Mean seconds | Rank |",
            "|-----------|----------------|-----:|-----------:|-----------------------:|-----------------------:|-------------:|-----:|",
        ]
    )
    for comparison in sorted(
        report.external_comparison_scorecard.values(),
        key=lambda item: (
            item.rank_by_mean_score is None,
            item.rank_by_mean_score or 999_999,
            item.framework != "Zaxy",
            item.framework,
        ),
    ):
        lines.append(
            "| "
            f"{comparison.framework} | "
            f"{comparison.evidence_scope} | "
            f"{comparison.runs} | "
            f"{_fmt_na(comparison.mean_score)} | "
            f"{_fmt_na(comparison.delta_vs_raw_rg, signed=True)} | "
            f"{_fmt_na(comparison.delta_vs_best_external, signed=True)} | "
            f"{_fmt_na(comparison.mean_total_seconds)} | "
            f"{comparison.rank_by_mean_score if comparison.rank_by_mean_score is not None else 'n/a'} |"
        )


def _harvey_report_path_label(path_value: str | None) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if path.name:
        return path.name
    return path_value


def _harvey_report_path_labels(paths_value: object) -> str:
    labels = [
        _harvey_report_path_label(str(path_value))
        for path_value in _list_value(paths_value)
    ]
    return ", ".join(labels)


def _parse_zaxy_result(item: object) -> HarveyZaxyResult:
    if not isinstance(item, dict):
        raise ValueError("Harvey normalized result contract requires object rows")
    required = {
        "schema_version",
        "run_id",
        "framework",
        "task_id",
        "corpus_hash",
        "models",
        "paths",
        "scores",
        "timing",
        "usage",
        "tooling",
    }
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(
            "Harvey normalized result contract missing required key(s): "
            + ", ".join(missing)
        )
    framework = str(item["framework"])
    if not framework.casefold().startswith("zaxy"):
        raise ValueError("Harvey Zaxy results must have framework starting with 'zaxy'")
    task_id = str(item["task_id"])
    allowed_tasks = {task.task_id for task in ARTICLE_TASKS}
    if task_id not in allowed_tasks:
        raise ValueError(f"task_id {task_id!r} is not part of the Harvey LAB article suite")
    scores = _dict(item["scores"], "scores")
    score = _number(scores.get("final_score"), "scores.final_score")
    models = _dict(item["models"], "models")
    timing = _dict(item["timing"], "timing")
    usage = _dict(item["usage"], "usage")
    tooling = _dict(item["tooling"], "tooling")
    paths = _dict(item["paths"], "paths")
    return HarveyZaxyResult(
        task_id=task_id,
        score=round(score, 3),
        run_id=str(item["run_id"]),
        framework=framework,
        generator=_optional_str(models.get("generator")),
        judge=_optional_str(models.get("judge")),
        generator_reasoning_effort=_optional_str(models.get("generator_reasoning_effort")),
        judge_reasoning_effort=_optional_str(models.get("judge_reasoning_effort")),
        temperature=_optional_float(models.get("temperature")),
        total_seconds=_optional_float(timing.get("total_seconds")),
        total_tokens=_optional_int(usage.get("total_tokens")),
        memory_search_calls=_optional_int(tooling.get("memory_search_calls")),
        memory_read_calls=_optional_int(tooling.get("memory_read_calls")),
        corpus_hash=_optional_str(item.get("corpus_hash")),
        commit=_optional_str(item.get("commit")),
        results_run_dir=_optional_str(paths.get("results_run_dir")),
        answer_path=_optional_str(paths.get("answer")),
        tool_log_path=_optional_str(paths.get("tool_log")),
        judge_path=_optional_str(paths.get("judge")),
        run_metrics_path=_optional_str(paths.get("run_metrics")),
    )


def _build_task_row(
    task: HarveyArticleTask,
    result: HarveyZaxyResult | None,
) -> HarveyTaskComparison:
    zaxy_score = result.score if result is not None else None
    delta_regular = (
        None
        if zaxy_score is None
        else round(zaxy_score - task.regular_no_memory_score, 3)
    )
    delta_best = (
        None
        if zaxy_score is None
        else round(zaxy_score - task.best_score, 3)
    )
    return HarveyTaskComparison(
        task_id=task.task_id,
        article_label=task.article_label,
        task_shape=task.task_shape,
        article_best_framework=task.best_framework,
        article_best_score=task.best_score,
        regular_no_memory_score=task.regular_no_memory_score,
        zaxy_score=zaxy_score,
        zaxy_delta_vs_regular_no_memory=delta_regular,
        zaxy_delta_vs_article_best=delta_best,
        zaxy_winner=None if delta_best is None else delta_best > 0,
        zaxy_total_seconds=result.total_seconds if result is not None else None,
        zaxy_total_tokens=result.total_tokens if result is not None else None,
        zaxy_memory_search_calls=result.memory_search_calls if result is not None else None,
        zaxy_memory_read_calls=result.memory_read_calls if result is not None else None,
    )


def _build_summary(status: str, rows: list[HarveyTaskComparison]) -> HarveySummary:
    zaxy_scores = [row.zaxy_score for row in rows if row.zaxy_score is not None]
    article_best = [row.article_best_score for row in rows]
    regular = [row.regular_no_memory_score for row in rows]
    delta_regular = [
        row.zaxy_delta_vs_regular_no_memory
        for row in rows
        if row.zaxy_delta_vs_regular_no_memory is not None
    ]
    delta_best = [
        row.zaxy_delta_vs_article_best
        for row in rows
        if row.zaxy_delta_vs_article_best is not None
    ]
    zaxy_seconds = [
        row.zaxy_total_seconds
        for row in rows
        if row.zaxy_score is not None and row.zaxy_total_seconds is not None
    ]
    zaxy_tokens = [
        row.zaxy_total_tokens
        for row in rows
        if row.zaxy_score is not None and row.zaxy_total_tokens is not None
    ]
    zaxy_search_calls = [
        row.zaxy_memory_search_calls
        for row in rows
        if row.zaxy_score is not None and row.zaxy_memory_search_calls is not None
    ]
    zaxy_read_calls = [
        row.zaxy_memory_read_calls
        for row in rows
        if row.zaxy_score is not None and row.zaxy_memory_read_calls is not None
    ]
    return HarveySummary(
        status=status,
        article_task_count=len(ARTICLE_TASKS),
        zaxy_task_count=len(zaxy_scores),
        zaxy_mean_score=_mean(zaxy_scores),
        article_best_mean_for_zaxy_tasks=_mean(article_best),
        regular_mean_for_zaxy_tasks=_mean(regular),
        mean_delta_vs_regular_no_memory=_mean(delta_regular),
        mean_delta_vs_article_best=_mean(delta_best),
        zaxy_task_wins=sum(1 for row in rows if row.zaxy_winner is True),
        zaxy_task_losses=sum(1 for row in rows if row.zaxy_winner is False),
        zaxy_task_ties=sum(1 for row in rows if row.zaxy_delta_vs_article_best == 0),
        zaxy_mean_total_seconds=_mean(zaxy_seconds),
        zaxy_total_tokens=sum(zaxy_tokens) if zaxy_tokens else None,
        zaxy_total_memory_search_calls=sum(zaxy_search_calls) if zaxy_search_calls else None,
        zaxy_total_memory_read_calls=sum(zaxy_read_calls) if zaxy_read_calls else None,
    )


def _build_framework_scorecard(
    zaxy_rows: list[HarveyTaskComparison],
) -> dict[str, HarveyFrameworkScorecard]:
    zaxy_by_task = {row.task_id: row for row in zaxy_rows if row.zaxy_score is not None}
    scorecard: dict[str, HarveyFrameworkScorecard] = {}
    regular_rows = [
        (task.task_id, task.regular_no_memory_score, task.regular_no_memory_score)
        for task in ARTICLE_TASKS
    ]
    best_rows = [
        (task.task_id, task.best_score, task.regular_no_memory_score)
        for task in ARTICLE_TASKS
    ]
    scorecard["regular no-memory"] = _framework_scorecard_row(
        framework="regular no-memory",
        evidence_scope="article regular baseline across all ten tasks",
        article_rows=regular_rows,
        zaxy_by_task=zaxy_by_task,
        task_wins=0,
    )
    scorecard["article best observed"] = _framework_scorecard_row(
        framework="article best observed",
        evidence_scope="best published memory/search row per article task",
        article_rows=best_rows,
        zaxy_by_task=zaxy_by_task,
        task_wins=len(ARTICLE_TASKS),
    )
    for framework in ARTICLE_FRAMEWORK_FIT:
        article_rows: list[tuple[str, float, float]] = []
        for task in ARTICLE_TASKS:
            if framework in _published_winning_frameworks(task):
                article_rows.append((task.task_id, task.best_score, task.regular_no_memory_score))
        if article_rows:
            scorecard[framework] = _framework_scorecard_row(
                framework=framework,
                evidence_scope="article task-winner matrix only",
                article_rows=article_rows,
                zaxy_by_task=zaxy_by_task,
                task_wins=len(article_rows),
            )
        else:
            scorecard[framework] = HarveyFrameworkScorecard(
                framework=framework,
                evidence_scope="framework fit only; no published task-winning score",
                article_task_count=0,
                mean_score=None,
                mean_delta_vs_regular_no_memory=None,
                task_wins=0,
                zaxy_overlap_task_count=0,
                zaxy_mean_on_overlap=None,
                zaxy_delta_on_overlap=None,
            )
    zaxy_scores = [row.zaxy_score for row in zaxy_rows if row.zaxy_score is not None]
    zaxy_deltas = [
        row.zaxy_delta_vs_regular_no_memory
        for row in zaxy_rows
        if row.zaxy_delta_vs_regular_no_memory is not None
    ]
    scorecard["Zaxy"] = HarveyFrameworkScorecard(
        framework="Zaxy",
        evidence_scope="same-harness external Zaxy normalized results",
        article_task_count=len(zaxy_scores),
        mean_score=_mean(zaxy_scores),
        mean_delta_vs_regular_no_memory=_mean(zaxy_deltas),
        task_wins=sum(1 for row in zaxy_rows if row.zaxy_winner is True),
        zaxy_overlap_task_count=len(zaxy_scores),
        zaxy_mean_on_overlap=_mean(zaxy_scores),
        zaxy_delta_on_overlap=None,
    )
    return scorecard


def _build_external_baseline_scorecard(
    provenance: dict[str, object],
) -> dict[str, HarveyExternalBaselineScorecard]:
    scorecard: dict[str, HarveyExternalBaselineScorecard] = {}
    for path_value in _list_value(provenance.get("external_baseline_report_paths")):
        path = Path(str(path_value))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        aggregate = payload.get("aggregate")
        if not isinstance(aggregate, dict):
            continue
        frameworks = aggregate.get("frameworks")
        if not isinstance(frameworks, list):
            continue
        raw_rg_score = _external_baseline_raw_rg_score(frameworks)
        for item in frameworks:
            if not isinstance(item, dict):
                continue
            framework = str(item.get("framework") or "")
            if not framework:
                continue
            if framework.casefold() == "zaxy":
                continue
            mean_score = _optional_float(item.get("avg_final_score"))
            runs = _optional_int(item.get("runs")) or 0
            existing = scorecard.get(framework)
            if existing is not None and existing.runs > runs:
                continue
            scorecard[framework] = HarveyExternalBaselineScorecard(
                framework=framework,
                evidence_scope="Harvey-native comparison artifact",
                runs=runs,
                mean_score=mean_score,
                delta_vs_raw_rg=(
                    None
                    if mean_score is None or raw_rg_score is None
                    else round(mean_score - raw_rg_score, 3)
                ),
                mean_total_seconds=_optional_float(item.get("avg_total_seconds")),
                source_report_path=str(path),
            )
    return scorecard


def _build_external_comparison_scorecard(
    summary: HarveySummary,
    external_baseline_scorecard: dict[str, HarveyExternalBaselineScorecard],
) -> dict[str, HarveyExternalComparisonScorecard]:
    raw_rg_score = None
    raw_rg = external_baseline_scorecard.get("raw-rg")
    if raw_rg is not None:
        raw_rg_score = raw_rg.mean_score
    best_external_score = _mean(
        [
            row.mean_score
            for row in external_baseline_scorecard.values()
            if row.mean_score is not None
        ]
    )
    numeric_external_scores = [
        row.mean_score
        for row in external_baseline_scorecard.values()
        if row.mean_score is not None
    ]
    if numeric_external_scores:
        best_external_score = max(numeric_external_scores)
    rows: dict[str, HarveyExternalComparisonScorecard] = {}
    for framework, row in external_baseline_scorecard.items():
        rows[framework] = HarveyExternalComparisonScorecard(
            framework=row.framework,
            evidence_scope=row.evidence_scope,
            runs=row.runs,
            mean_score=row.mean_score,
            delta_vs_raw_rg=row.delta_vs_raw_rg,
            delta_vs_best_external=(
                None
                if row.mean_score is None or best_external_score is None
                else round(row.mean_score - best_external_score, 3)
            ),
            rank_by_mean_score=None,
            mean_total_seconds=row.mean_total_seconds,
            source_report_path=row.source_report_path,
        )
    zaxy_mean = summary.zaxy_mean_score
    rows["Zaxy"] = HarveyExternalComparisonScorecard(
        framework="Zaxy",
        evidence_scope=(
            "same-harness external Zaxy normalized results"
            if zaxy_mean is not None
            else "pending external Zaxy normalized results"
        ),
        runs=summary.zaxy_task_count,
        mean_score=zaxy_mean,
        delta_vs_raw_rg=(
            None
            if zaxy_mean is None or raw_rg_score is None
            else round(zaxy_mean - raw_rg_score, 3)
        ),
        delta_vs_best_external=(
            None
            if zaxy_mean is None or best_external_score is None
            else round(zaxy_mean - best_external_score, 3)
        ),
        rank_by_mean_score=None,
        mean_total_seconds=summary.zaxy_mean_total_seconds,
        source_report_path=None,
    )
    ranked = sorted(
        [row for row in rows.values() if row.mean_score is not None],
        key=lambda row: (-(row.mean_score or 0.0), row.framework),
    )
    ranks = {row.framework: index for index, row in enumerate(ranked, start=1)}
    return {
        framework: (
            HarveyExternalComparisonScorecard(
                framework=row.framework,
                evidence_scope=row.evidence_scope,
                runs=row.runs,
                mean_score=row.mean_score,
                delta_vs_raw_rg=row.delta_vs_raw_rg,
                delta_vs_best_external=row.delta_vs_best_external,
                rank_by_mean_score=ranks.get(row.framework),
                mean_total_seconds=row.mean_total_seconds,
                source_report_path=row.source_report_path,
            )
            if row.mean_score is not None
            else row
        )
        for framework, row in rows.items()
    }


def _external_baseline_raw_rg_score(frameworks: list[object]) -> float | None:
    for item in frameworks:
        if not isinstance(item, dict):
            continue
        if str(item.get("framework") or "").casefold() == "raw-rg":
            return _optional_float(item.get("avg_final_score"))
    return None


def _framework_scorecard_row(
    *,
    framework: str,
    evidence_scope: str,
    article_rows: list[tuple[str, float, float]],
    zaxy_by_task: dict[str, HarveyTaskComparison],
    task_wins: int,
) -> HarveyFrameworkScorecard:
    scores = [score for _, score, _ in article_rows]
    deltas = [round(score - regular, 3) for _, score, regular in article_rows]
    overlap_article_scores: list[float] = []
    overlap_zaxy_scores: list[float] = []
    for task_id, score, _ in article_rows:
        zaxy_row = zaxy_by_task.get(task_id)
        if zaxy_row is None or zaxy_row.zaxy_score is None:
            continue
        overlap_article_scores.append(score)
        overlap_zaxy_scores.append(zaxy_row.zaxy_score)
    zaxy_overlap_mean = _mean(overlap_zaxy_scores)
    article_overlap_mean = _mean(overlap_article_scores)
    return HarveyFrameworkScorecard(
        framework=framework,
        evidence_scope=evidence_scope,
        article_task_count=len(article_rows),
        mean_score=_mean(scores),
        mean_delta_vs_regular_no_memory=_mean(deltas),
        task_wins=task_wins,
        zaxy_overlap_task_count=len(overlap_zaxy_scores),
        zaxy_mean_on_overlap=zaxy_overlap_mean,
        zaxy_delta_on_overlap=(
            None
            if zaxy_overlap_mean is None or article_overlap_mean is None
            else round(zaxy_overlap_mean - article_overlap_mean, 3)
        ),
    )


def _published_winning_frameworks(task: HarveyArticleTask) -> tuple[str, ...]:
    return tuple(part.strip() for part in task.best_framework.split("/") if part.strip())


def _framework_scorecard_order() -> tuple[str, ...]:
    return (
        "regular no-memory",
        "article best observed",
        *ARTICLE_FRAMEWORK_FIT.keys(),
        "Zaxy",
    )


def _zaxy_fit_label(rows: list[HarveyTaskComparison]) -> str:
    winners = [row.task_shape for row in rows if row.zaxy_winner]
    if not winners:
        return "Pending external Harvey LAB runs"
    return "; ".join(dict.fromkeys(winners))


def _build_framework_fit(
    zaxy_rows: list[HarveyTaskComparison],
    summary: HarveySummary,
) -> dict[str, HarveyFrameworkFit]:
    return {
        **ARTICLE_FRAMEWORK_FIT,
        "Zaxy": HarveyFrameworkFit(
            framework="Zaxy",
            where_strongest=_zaxy_fit_label(zaxy_rows),
            example_tasks=_zaxy_example_tasks(zaxy_rows),
            interpretation=_zaxy_fit_interpretation(summary),
        ),
    }


def _zaxy_example_tasks(rows: list[HarveyTaskComparison]) -> str:
    winners = [row.article_label for row in rows if row.zaxy_winner]
    if not winners:
        return "No Zaxy task wins recorded yet"
    return "; ".join(winners)


def _zaxy_fit_interpretation(summary: HarveySummary) -> str:
    if summary.zaxy_task_count == 0:
        return "Zaxy has not yet been scored on this external Harvey LAB suite."
    if summary.mean_delta_vs_article_best is not None and summary.mean_delta_vs_article_best > 0:
        return "Zaxy is ahead of the published article-best rows on the scored subset."
    if summary.mean_delta_vs_regular_no_memory is not None and summary.mean_delta_vs_regular_no_memory > 0:
        return "Zaxy improves on the regular no-memory baseline on the scored subset."
    return "Zaxy did not improve on the published baselines for the scored subset."


def _report_payload(report: HarveyLabReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "generated_at": report.generated_at,
        "status": report.status,
        "external_suite": report.external_suite,
        "result_provenance": report.result_provenance,
        "summary": asdict(report.summary),
        "task_rows": {
            key: asdict(value)
            for key, value in report.task_rows.items()
        },
        "framework_scorecard": {
            key: asdict(value)
            for key, value in report.framework_scorecard.items()
        },
        "external_baseline_scorecard": {
            key: asdict(value)
            for key, value in report.external_baseline_scorecard.items()
        },
        "external_comparison_scorecard": {
            key: asdict(value)
            for key, value in report.external_comparison_scorecard.items()
        },
        "framework_fit": {
            key: asdict(value)
            for key, value in report.framework_fit.items()
        },
        "zaxy_results": [asdict(result) for result in report.zaxy_results],
        "caveats": list(report.caveats),
    }


def _summary_from_payload(payload: dict[str, object]) -> HarveySummary:
    return HarveySummary(
        status=str(payload.get("status", "")),
        article_task_count=_required_int(payload.get("article_task_count"), "summary.article_task_count"),
        zaxy_task_count=_required_int(payload.get("zaxy_task_count"), "summary.zaxy_task_count"),
        zaxy_mean_score=_optional_float(payload.get("zaxy_mean_score")),
        article_best_mean_for_zaxy_tasks=_optional_float(payload.get("article_best_mean_for_zaxy_tasks")),
        regular_mean_for_zaxy_tasks=_optional_float(payload.get("regular_mean_for_zaxy_tasks")),
        mean_delta_vs_regular_no_memory=_optional_float(payload.get("mean_delta_vs_regular_no_memory")),
        mean_delta_vs_article_best=_optional_float(payload.get("mean_delta_vs_article_best")),
        zaxy_task_wins=_required_int(payload.get("zaxy_task_wins"), "summary.zaxy_task_wins"),
        zaxy_task_losses=_required_int(payload.get("zaxy_task_losses"), "summary.zaxy_task_losses"),
        zaxy_task_ties=_required_int(payload.get("zaxy_task_ties"), "summary.zaxy_task_ties"),
        zaxy_mean_total_seconds=_optional_float(payload.get("zaxy_mean_total_seconds")),
        zaxy_total_tokens=_optional_int(payload.get("zaxy_total_tokens")),
        zaxy_total_memory_search_calls=_optional_int(payload.get("zaxy_total_memory_search_calls")),
        zaxy_total_memory_read_calls=_optional_int(payload.get("zaxy_total_memory_read_calls")),
    )


def _task_comparison_from_payload(payload: dict[str, object]) -> HarveyTaskComparison:
    return HarveyTaskComparison(
        task_id=str(payload.get("task_id", "")),
        article_label=str(payload.get("article_label", "")),
        task_shape=str(payload.get("task_shape", "")),
        article_best_framework=str(payload.get("article_best_framework", "")),
        article_best_score=_required_float(payload.get("article_best_score"), "task.article_best_score"),
        regular_no_memory_score=_required_float(
            payload.get("regular_no_memory_score"),
            "task.regular_no_memory_score",
        ),
        zaxy_score=_optional_float(payload.get("zaxy_score")),
        zaxy_delta_vs_regular_no_memory=_optional_float(payload.get("zaxy_delta_vs_regular_no_memory")),
        zaxy_delta_vs_article_best=_optional_float(payload.get("zaxy_delta_vs_article_best")),
        zaxy_winner=_optional_bool(payload.get("zaxy_winner")),
        zaxy_total_seconds=_optional_float(payload.get("zaxy_total_seconds")),
        zaxy_total_tokens=_optional_int(payload.get("zaxy_total_tokens")),
        zaxy_memory_search_calls=_optional_int(payload.get("zaxy_memory_search_calls")),
        zaxy_memory_read_calls=_optional_int(payload.get("zaxy_memory_read_calls")),
    )


def _framework_fit_from_payload(payload: dict[str, object]) -> HarveyFrameworkFit:
    return HarveyFrameworkFit(
        framework=str(payload.get("framework", "")),
        where_strongest=str(payload.get("where_strongest", "")),
        example_tasks=str(payload.get("example_tasks", "")),
        interpretation=str(payload.get("interpretation", "")),
    )


def _framework_scorecard_from_payload(payload: dict[str, object]) -> HarveyFrameworkScorecard:
    return HarveyFrameworkScorecard(
        framework=str(payload.get("framework", "")),
        evidence_scope=str(payload.get("evidence_scope", "")),
        article_task_count=_required_int(
            payload.get("article_task_count"),
            "framework_scorecard.article_task_count",
        ),
        mean_score=_optional_float(payload.get("mean_score")),
        mean_delta_vs_regular_no_memory=_optional_float(
            payload.get("mean_delta_vs_regular_no_memory")
        ),
        task_wins=_required_int(payload.get("task_wins"), "framework_scorecard.task_wins"),
        zaxy_overlap_task_count=_required_int(
            payload.get("zaxy_overlap_task_count"),
            "framework_scorecard.zaxy_overlap_task_count",
        ),
        zaxy_mean_on_overlap=_optional_float(payload.get("zaxy_mean_on_overlap")),
        zaxy_delta_on_overlap=_optional_float(payload.get("zaxy_delta_on_overlap")),
    )


def _external_baseline_scorecard_from_payload(payload: dict[str, object]) -> HarveyExternalBaselineScorecard:
    return HarveyExternalBaselineScorecard(
        framework=str(payload.get("framework", "")),
        evidence_scope=str(payload.get("evidence_scope", "")),
        runs=_required_int(payload.get("runs"), "external_baseline_scorecard.runs"),
        mean_score=_optional_float(payload.get("mean_score")),
        delta_vs_raw_rg=_optional_float(payload.get("delta_vs_raw_rg")),
        mean_total_seconds=_optional_float(payload.get("mean_total_seconds")),
        source_report_path=str(payload.get("source_report_path", "")),
    )


def _external_comparison_scorecard_from_payload(payload: dict[str, object]) -> HarveyExternalComparisonScorecard:
    return HarveyExternalComparisonScorecard(
        framework=str(payload.get("framework", "")),
        evidence_scope=str(payload.get("evidence_scope", "")),
        runs=_required_int(payload.get("runs"), "external_comparison_scorecard.runs"),
        mean_score=_optional_float(payload.get("mean_score")),
        delta_vs_raw_rg=_optional_float(payload.get("delta_vs_raw_rg")),
        delta_vs_best_external=_optional_float(payload.get("delta_vs_best_external")),
        rank_by_mean_score=_optional_int(payload.get("rank_by_mean_score")),
        mean_total_seconds=_optional_float(payload.get("mean_total_seconds")),
        source_report_path=_optional_str(payload.get("source_report_path")),
    )


def _zaxy_result_from_payload(payload: dict[str, object]) -> HarveyZaxyResult:
    return HarveyZaxyResult(
        task_id=str(payload.get("task_id", "")),
        score=_required_float(payload.get("score"), "zaxy_result.score"),
        run_id=str(payload.get("run_id", "")),
        framework=str(payload.get("framework", "")),
        generator=_optional_str(payload.get("generator")),
        judge=_optional_str(payload.get("judge")),
        generator_reasoning_effort=_optional_str(payload.get("generator_reasoning_effort")),
        judge_reasoning_effort=_optional_str(payload.get("judge_reasoning_effort")),
        temperature=_optional_float(payload.get("temperature")),
        total_seconds=_optional_float(payload.get("total_seconds")),
        total_tokens=_optional_int(payload.get("total_tokens")),
        memory_search_calls=_optional_int(payload.get("memory_search_calls")),
        memory_read_calls=_optional_int(payload.get("memory_read_calls")),
        corpus_hash=_optional_str(payload.get("corpus_hash")),
        commit=_optional_str(payload.get("commit")),
        results_run_dir=_optional_str(payload.get("results_run_dir")),
        answer_path=_optional_str(payload.get("answer_path")),
        tool_log_path=_optional_str(payload.get("tool_log_path")),
        judge_path=_optional_str(payload.get("judge_path")),
        run_metrics_path=_optional_str(payload.get("run_metrics_path")),
    )


def _harvey_evidence_failures(report: HarveyLabReport) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    if (report.external_baseline_scorecard or report.zaxy_results) and not report.external_comparison_scorecard:
        failures.append(
            {
                "task_id": "*",
                "reason": "missing_external_comparison_scorecard",
            }
        )
    elif report.external_comparison_scorecard and "Zaxy" not in report.external_comparison_scorecard:
        failures.append(
            {
                "task_id": "*",
                "reason": "missing_zaxy_external_comparison_row",
            }
        )
    external_baseline_failure = _external_baseline_scorecard_failure(report)
    if external_baseline_failure is not None:
        failures.append(external_baseline_failure)
    external_comparison_failure = _external_comparison_scorecard_failure(report)
    if external_comparison_failure is not None:
        failures.append(external_comparison_failure)
    framework_fit_failure = _framework_fit_failure(report)
    if framework_fit_failure is not None:
        failures.append(framework_fit_failure)
    model_configs = {
        (result.generator, result.judge)
        for result in report.zaxy_results
        if result.generator and result.judge
    }
    if len(model_configs) > 1:
        failures.append(
            {
                "task_id": "*",
                "reason": "mixed_zaxy_model_configuration",
            }
        )
    generation_settings = {
        (
            result.generator_reasoning_effort,
            result.judge_reasoning_effort,
            result.temperature,
        )
        for result in report.zaxy_results
        if result.generator and result.judge
    }
    if len(generation_settings) > 1:
        failures.append(
            {
                "task_id": "*",
                "reason": "mixed_zaxy_generation_settings",
            }
        )
    harvey_git_commit = _optional_str(report.result_provenance.get("harvey_git_commit"))
    if harvey_git_commit:
        for zaxy_result in report.zaxy_results:
            if zaxy_result.commit and zaxy_result.commit != harvey_git_commit:
                failures.append(
                    {
                        "task_id": zaxy_result.task_id,
                        "reason": "zaxy_run_commit_mismatch",
                    }
                )
                break
    results = {result.task_id: result for result in report.zaxy_results}
    for task in ARTICLE_TASKS:
        result = results.get(task.task_id)
        if result is None:
            continue
        if not result.generator or not result.judge or not result.corpus_hash:
            failures.append(
                {
                    "task_id": task.task_id,
                    "reason": "missing_model_or_corpus_metadata",
                }
            )
            continue
        if not result.memory_search_calls or result.memory_search_calls <= 0:
            failures.append(
                {
                    "task_id": task.task_id,
                    "reason": "memory_tools_not_used",
                }
            )
            continue
        required_paths = (
            result.answer_path,
            result.tool_log_path,
            result.judge_path,
            result.run_metrics_path,
        )
        if any(not path for path in required_paths):
            failures.append(
                {
                    "task_id": task.task_id,
                    "reason": "missing_reviewable_artifact_paths",
                }
            )
    return failures


def _external_baseline_scorecard_failure(report: HarveyLabReport) -> dict[str, str] | None:
    if not report.external_baseline_scorecard:
        return None
    baseline_paths = _list_value(report.result_provenance.get("external_baseline_report_paths"))
    if any(not Path(str(path_value)).exists() for path_value in baseline_paths):
        return None
    expected = _build_external_baseline_scorecard(report.result_provenance)
    if report.external_baseline_scorecard != expected:
        return {
            "task_id": "*",
            "reason": "stale_external_baseline_scorecard",
        }
    return None


def _framework_fit_failure(report: HarveyLabReport) -> dict[str, str] | None:
    zaxy_rows = [
        row
        for row in report.task_rows.values()
        if row.zaxy_score is not None
    ]
    expected = _build_framework_fit(zaxy_rows, report.summary)
    if report.framework_fit != expected:
        return {
            "task_id": "*",
            "reason": "stale_framework_fit",
        }
    return None


def _external_comparison_scorecard_failure(report: HarveyLabReport) -> dict[str, str] | None:
    if not report.external_comparison_scorecard:
        return None
    expected = _build_external_comparison_scorecard(
        report.summary,
        report.external_baseline_scorecard,
    )
    if set(report.external_comparison_scorecard) != set(expected):
        return {
            "task_id": "*",
            "reason": "stale_external_comparison_scorecard",
        }
    for framework, expected_row in expected.items():
        actual_row = report.external_comparison_scorecard[framework]
        if actual_row != expected_row:
            return {
                "task_id": "*",
                "reason": (
                    "stale_zaxy_external_comparison_row"
                    if framework == "Zaxy"
                    else "stale_external_comparison_scorecard"
                ),
            }
    return None

def _harvey_provenance_failures(
    report: HarveyLabReport,
    *,
    require_complete: bool,
) -> list[dict[str, str]]:
    if not report.zaxy_results:
        return []
    if not require_complete and not report.result_provenance:
        return []
    provenance = report.result_provenance
    source = provenance.get("source")
    roots = _list_value(provenance.get("roots"))
    normalized_paths = _list_value(provenance.get("normalized_result_paths"))
    zaxy_results_path = provenance.get("zaxy_results_json_path")
    if not source or (not roots and not normalized_paths and not zaxy_results_path):
        return [
            {
                "task_id": "*",
                "reason": "missing_external_result_provenance",
            }
        ]
    for path_value in _list_value(provenance.get("external_baseline_report_paths")):
        path = Path(str(path_value))
        if (
            path.name.startswith("comparison")
            and path.suffix == ".json"
            and path.exists()
            and _path_is_within_any_root(path, roots)
        ):
            continue
        return [
            {
                "task_id": "*",
                "reason": "missing_external_baseline_report_artifact",
            }
        ]
    completed_count = sum(1 for row in report.task_rows.values() if row.zaxy_score is not None)
    required_path_count = completed_count if require_complete else min(completed_count, len(normalized_paths))
    if len(normalized_paths) < required_path_count:
        return [
            {
                "task_id": "*",
                "reason": "missing_normalized_result_paths",
            }
        ]
    for path_value in normalized_paths:
        path = Path(str(path_value))
        if (
            path.name != "normalized-result.json"
            or not path.exists()
            or not _path_is_within_any_root(path, roots)
        ):
            return [
                {
                    "task_id": "*",
                    "reason": "missing_normalized_result_artifact",
                }
            ]
    artifact_results: dict[str, tuple[HarveyZaxyResult, Path]] = {}
    for path_value in normalized_paths:
        path = Path(str(path_value))
        try:
            loaded = load_harvey_zaxy_results(path)
        except ValueError:
            return [
                {
                    "task_id": "*",
                    "reason": "invalid_normalized_result_artifact",
                }
            ]
        for result in loaded:
            artifact_results[result.task_id] = (result, path)
    for result in report.zaxy_results:
        artifact_item = artifact_results.get(result.task_id)
        if artifact_item is None:
            return [
                {
                    "task_id": result.task_id,
                    "reason": "missing_normalized_result_for_task",
                }
            ]
        artifact_result, artifact_path = artifact_item
        if artifact_result.run_id != result.run_id or artifact_result.score != result.score:
            return [
                {
                    "task_id": result.task_id,
                    "reason": "normalized_result_mismatch",
                }
            ]
        if _zaxy_result_metadata_fingerprint(artifact_result) != _zaxy_result_metadata_fingerprint(result):
            return [
                {
                    "task_id": result.task_id,
                    "reason": "normalized_result_metadata_mismatch",
                }
            ]
        referenced_artifact_failure = _referenced_run_artifact_failure(
            artifact_result,
            artifact_path,
            roots,
        )
        if referenced_artifact_failure is not None:
            return [referenced_artifact_failure]
    if require_complete and not provenance.get("harvey_git_commit"):
        return [
            {
                "task_id": "*",
                "reason": "missing_harvey_suite_commit",
            }
        ]
    if require_complete and not any(
        framework != "Zaxy"
        for framework, row in report.external_comparison_scorecard.items()
        if row.mean_score is not None
    ):
        return [
            {
                "task_id": "*",
                "reason": "missing_external_scored_system_comparison",
            }
        ]
    if require_complete and (
        external_comparison_failure := _external_comparison_result_evidence_failure(
            provenance,
            expected_run_config=_zaxy_run_configuration(report),
        )
    ) is not None:
        return [
            {
                "task_id": "*",
                "reason": external_comparison_failure,
            }
        ]
    if require_complete and (
        run_audit_failure := _external_run_audit_artifact_failure(provenance, report=report)
    ) is not None:
        return [run_audit_failure]
    return []


def _path_is_within_any_root(path: Path, roots: list[object]) -> bool:
    if not roots:
        return False
    resolved_path = path.resolve()
    for root_value in roots:
        if str(root_value) and _path_is_relative_to(resolved_path, Path(str(root_value)).resolve()):
            return True
    return False


def _external_run_audit_artifact_failure(
    provenance: dict[str, object],
    *,
    report: HarveyLabReport | None = None,
) -> dict[str, str] | None:
    provenance_roots = _list_value(provenance.get("roots"))
    expected_worktree = (
        Path(str(provenance_roots[0])).resolve()
        if provenance_roots
        else None
    )
    specs = (
        (
            "external_run_manifest_paths",
            "harvey-lab-external-run.json",
            "zaxy.harvey-lab-external-run.v1",
        ),
        (
            "external_readiness_report_paths",
            "harvey-lab-ready.json",
            "zaxy.harvey-lab-run-readiness.v1",
        ),
        (
            "external_status_report_paths",
            "harvey-lab-status.json",
            "zaxy.harvey-lab-run-status.v1",
        ),
    )
    payloads: dict[str, dict[str, object]] = {}
    for key, filename, schema_version in specs:
        paths = _list_value(provenance.get(key))
        if not paths:
            return {"task_id": "*", "reason": "missing_external_run_audit_artifacts"}
        path = Path(str(paths[0]))
        if (
            path.name != filename
            or not path.exists()
            or not _path_is_within_any_root(path, provenance_roots)
        ):
            return {"task_id": "*", "reason": "missing_external_run_audit_artifact"}
        if expected_worktree is not None:
            expected_locations = {
                candidate.resolve()
                for root_value in provenance_roots
                for candidate in _harvey_run_gate_artifact_candidates(Path(str(root_value)), filename)
            }
            if path.resolve() not in expected_locations:
                return {"task_id": "*", "reason": "external_run_audit_root_mismatch"}
        try:
            payload = _json_object_from_file(path, filename)
        except ValueError:
            return {"task_id": "*", "reason": "invalid_external_run_audit_artifact"}
        if payload.get("schema_version") != schema_version:
            return {"task_id": "*", "reason": "invalid_external_run_audit_artifact"}
        payloads[key] = payload
    manifest = payloads["external_run_manifest_paths"]
    normalized_results_by_task = _normalized_results_by_task(provenance)
    if manifest.get("source_url") != HARVEY_LAB_REPO_URL or manifest.get("article_url") != ARTICLE_URL:
        return {"task_id": "*", "reason": "external_run_manifest_source_mismatch"}
    expected_worktree_str = str(expected_worktree) if expected_worktree is not None else None
    collection_command = _optional_str(manifest.get("collection_command")) or ""
    if (
        "uv run python scripts/memory_ablation/collect_results.py" not in collection_command
        or "--worktree ." not in collection_command
        or "--dedupe-latest" not in collection_command
        or "--output .ingestion/reports/comparison-zaxy.json" not in collection_command
    ):
        return {"task_id": "*", "reason": "external_run_manifest_command_mismatch"}
    comparison_command = _optional_str(manifest.get("comparison_command")) or ""
    if (
        not comparison_command.startswith("zaxy harvey-lab-import ")
        or "path/to/harvey-zaxy-worktree" not in comparison_command
        or "--output-dir reports/benchmarks/harvey-lab-memory-ablation" not in comparison_command
    ):
        return {"task_id": "*", "reason": "external_run_manifest_command_mismatch"}
    report_json_path = _optional_str(manifest.get("report_json_path")) or ""
    publish_output_path = _optional_str(manifest.get("publish_output_path")) or ""
    validation_command = _optional_str(manifest.get("validation_command")) or ""
    gate_command = _optional_str(manifest.get("gate_command")) or ""
    publish_command = _optional_str(manifest.get("publish_command")) or ""
    if (
        report_json_path != "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json"
        or publish_output_path
        != "reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md"
        or not validation_command.endswith(
            "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json --require-complete"
        )
        or not gate_command.endswith(
            "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json"
        )
        or not publish_command.endswith(
            "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json "
            "--output reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md"
        )
    ):
        return {"task_id": "*", "reason": "external_run_manifest_command_mismatch"}
    if _optional_int(manifest.get("task_count")) != len(ARTICLE_TASKS):
        return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
    manifest_tasks = manifest.get("tasks")
    if not isinstance(manifest_tasks, list):
        return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
    manifest_task_ids: list[str] = []
    for task, expected_task in zip(manifest_tasks, ARTICLE_TASKS, strict=True):
        if not isinstance(task, dict):
            return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
        task_id = _optional_str(task.get("task_id"))
        if task_id is None:
            return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
        manifest_task_ids.append(task_id)
        expected_run_id = f"zaxy-{expected_task.task_id.replace('/', '__')}"
        expected_normalized_result = f".ingestion/runs/{expected_run_id}/normalized-result.json"
        if _optional_str(task.get("expected_normalized_result")) != expected_normalized_result:
            return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
        run_command = _optional_str(task.get("run_command")) or ""
        if (
            f"--run-id {expected_run_id}" not in run_command
            or f"--task {expected_task.task_id}" not in run_command
            or f"--model {_optional_str(manifest.get('generator'))}" not in run_command
        ):
            return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
        manifest_reasoning_effort = _optional_str(manifest.get("reasoning_effort"))
        if manifest_reasoning_effort and f"--reasoning-effort {manifest_reasoning_effort}" not in run_command:
            return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
        judge_command = _optional_str(task.get("judge_command")) or ""
        if (
            f"--run-id {expected_run_id}" not in judge_command
            or f"--task {expected_task.task_id}" not in judge_command
            or f"--judge-model {_optional_str(manifest.get('judge'))}" not in judge_command
        ):
            return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
        normalize_command = _optional_str(task.get("normalize_command")) or ""
        if (
            f"--run-id {expected_run_id}" not in normalize_command
            or f"--task-id {expected_task.task_id}" not in normalize_command
        ):
            return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
        validate_command = _optional_str(task.get("validate_command")) or ""
        if f"--run-dir .ingestion/runs/{expected_run_id}" not in validate_command:
            return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
    if manifest_task_ids != [task.task_id for task in ARTICLE_TASKS]:
        return {"task_id": "*", "reason": "external_run_manifest_task_mismatch"}
    if report is not None:
        run_config = _zaxy_run_configuration(report)
        expected_manifest_config = {
            "generator": run_config.get("generator"),
            "judge": run_config.get("judge"),
            "reasoning_effort": run_config.get("generator_reasoning_effort"),
        }
        actual_manifest_config = {
            "generator": _optional_str(manifest.get("generator")),
            "judge": _optional_str(manifest.get("judge")),
            "reasoning_effort": _optional_str(manifest.get("reasoning_effort")),
        }
        if actual_manifest_config != expected_manifest_config:
            return {"task_id": "*", "reason": "external_run_manifest_config_mismatch"}
    readiness = payloads["external_readiness_report_paths"]
    if expected_worktree_str and _optional_str(readiness.get("worktree")) != expected_worktree_str:
        return {"task_id": "*", "reason": "external_run_audit_worktree_mismatch"}
    if readiness.get("source_url") != HARVEY_LAB_REPO_URL or readiness.get("article_url") != ARTICLE_URL:
        return {"task_id": "*", "reason": "external_readiness_source_mismatch"}
    readiness_audit = readiness.get("evidence_audit")
    readiness_index_ready_count = None
    readiness_run_ready_count = _optional_int(readiness.get("run_ready_count"))
    readiness_normalized_ready_count = _optional_int(readiness.get("normalized_ready_count"))
    readiness_import_ready_count = None
    if isinstance(readiness_audit, dict):
        readiness_index_ready_count = _optional_int(readiness_audit.get("index_ready_count"))
        readiness_import_ready_count = _optional_int(readiness_audit.get("import_ready_count"))
    readiness_blocking_reasons = _list_value(readiness.get("blocking_reasons"))
    readiness_is_post_run_complete = (
        readiness.get("status") == "not_ready"
        and readiness_blocking_reasons == ["results_already_complete"]
        and readiness_run_ready_count == len(ARTICLE_TASKS)
        and readiness_normalized_ready_count == len(ARTICLE_TASKS)
        and readiness_import_ready_count == len(ARTICLE_TASKS)
    )
    if readiness.get("status") != "ready_for_external_runs" and not readiness_is_post_run_complete:
        return {"task_id": "*", "reason": "external_readiness_not_ready"}
    if (
        readiness.get("suite_valid") is not True
        or _optional_str(readiness.get("task_filter")) not in (None, "")
        or readiness.get("resolved_task_id") is not None
        or _optional_int(readiness.get("expected_task_count")) != len(ARTICLE_TASKS)
        or _optional_int(readiness.get("index_ready_count")) != len(ARTICLE_TASKS)
        or readiness_index_ready_count != len(ARTICLE_TASKS)
    ):
        return {"task_id": "*", "reason": "external_readiness_not_full_suite"}
    if (readiness_blocking_reasons and not readiness_is_post_run_complete) or _list_value(
        readiness.get("missing_credentials")
    ):
        return {"task_id": "*", "reason": "external_readiness_has_blockers"}
    readiness_model_requirements = readiness.get("model_requirements")
    if not isinstance(readiness_model_requirements, dict):
        return {"task_id": "*", "reason": "external_readiness_config_mismatch"}
    readiness_generator = readiness_model_requirements.get("generator")
    readiness_judge = readiness_model_requirements.get("judge")
    if not isinstance(readiness_generator, dict) or not isinstance(readiness_judge, dict):
        return {"task_id": "*", "reason": "external_readiness_config_mismatch"}
    if (
        _optional_str(readiness_generator.get("model")) != _optional_str(manifest.get("generator"))
        or _optional_str(readiness_judge.get("model")) != _optional_str(manifest.get("judge"))
    ):
        return {"task_id": "*", "reason": "external_readiness_config_mismatch"}
    expected_commit = _optional_str(provenance.get("harvey_git_commit"))
    readiness_commit = _optional_str(readiness.get("harvey_git_commit"))
    if expected_commit:
        if not readiness_commit:
            return {"task_id": "*", "reason": "external_run_audit_commit_missing"}
        if readiness_commit != expected_commit:
            return {"task_id": "*", "reason": "external_run_audit_commit_mismatch"}
    status = payloads["external_status_report_paths"]
    if expected_worktree_str and _optional_str(status.get("worktree")) != expected_worktree_str:
        return {"task_id": "*", "reason": "external_run_audit_worktree_mismatch"}
    if status.get("status") != "complete":
        return {"task_id": "*", "reason": "external_run_status_incomplete"}
    status_commit = _optional_str(status.get("harvey_git_commit"))
    if expected_commit:
        if not status_commit:
            return {"task_id": "*", "reason": "external_run_audit_commit_missing"}
        if status_commit != expected_commit:
            return {"task_id": "*", "reason": "external_run_audit_commit_mismatch"}
    status_audit = status.get("evidence_audit")
    status_import_ready_count = None
    if isinstance(status_audit, dict):
        status_import_ready_count = _optional_int(status_audit.get("import_ready_count"))
    if (
        _optional_int(status.get("expected_task_count")) != len(ARTICLE_TASKS)
        or status_import_ready_count != len(ARTICLE_TASKS)
    ):
        return {"task_id": "*", "reason": "external_run_status_not_full_suite"}
    status_tasks = status.get("tasks")
    if not isinstance(status_tasks, dict):
        return {"task_id": "*", "reason": "external_run_status_task_evidence_missing"}
    expected_task_ids = [task.task_id for task in ARTICLE_TASKS]
    if set(status_tasks.keys()) != set(expected_task_ids):
        return {"task_id": "*", "reason": "external_run_status_task_evidence_missing"}
    for task_id in expected_task_ids:
        task_status = status_tasks.get(task_id)
        if not isinstance(task_status, dict):
            return {"task_id": task_id, "reason": "external_run_status_task_evidence_missing"}
        if _optional_str(task_status.get("task_id")) != task_id:
            return {"task_id": task_id, "reason": "external_run_status_task_evidence_missing"}
        expected_index_dir = (
            expected_worktree / ".ingestion" / "indexes" / task_id.replace("/", "__") / "zaxy"
            if expected_worktree is not None
            else None
        )
        status_index_dir = _optional_str(task_status.get("index_dir"))
        if (
            status_index_dir is None
            or expected_index_dir is None
            or Path(status_index_dir).resolve() != expected_index_dir.resolve()
        ):
            return {"task_id": task_id, "reason": "external_run_status_index_dir_mismatch"}
        expected_run_id = f"zaxy-{task_id.replace('/', '__')}"
        expected_run_dir = expected_worktree / "results" / expected_run_id if expected_worktree is not None else None
        status_run_dir = _optional_str(task_status.get("run_dir"))
        if (
            status_run_dir is None
            or expected_run_dir is None
            or Path(status_run_dir).resolve() != expected_run_dir.resolve()
        ):
            return {"task_id": task_id, "reason": "external_run_status_run_dir_mismatch"}
        if (
            task_status.get("index_ready") is not True
            or task_status.get("run_artifacts_ready") is not True
            or task_status.get("normalized_result_ready") is not True
            or task_status.get("import_ready") is not True
        ):
            return {"task_id": task_id, "reason": "external_run_status_task_not_ready"}
        imported_result = normalized_results_by_task.get(task_id)
        if imported_result is None:
            return {"task_id": task_id, "reason": "external_run_status_normalized_result_mismatch"}
        expected_normalized_path = imported_result[1]
        expected_manifest_normalized_path = (
            expected_worktree / ".ingestion" / "runs" / expected_run_id / "normalized-result.json"
            if expected_worktree is not None
            else None
        )
        status_normalized_path = _optional_str(task_status.get("normalized_result_path"))
        if status_normalized_path is None:
            return {"task_id": task_id, "reason": "external_run_status_normalized_result_mismatch"}
        if expected_manifest_normalized_path is None or expected_normalized_path != expected_manifest_normalized_path.resolve():
            return {"task_id": task_id, "reason": "external_run_status_normalized_result_mismatch"}
        if Path(status_normalized_path).resolve() != expected_normalized_path:
            return {"task_id": task_id, "reason": "external_run_status_normalized_result_mismatch"}
        if imported_result[0].run_id != expected_run_id:
            return {"task_id": task_id, "reason": "external_run_status_run_id_mismatch"}
        if _optional_str(task_status.get("run_id")) != expected_run_id:
            return {"task_id": task_id, "reason": "external_run_status_run_id_mismatch"}
    return None


def _normalized_results_by_task(provenance: dict[str, object]) -> dict[str, tuple[HarveyZaxyResult, Path]]:
    results_by_task: dict[str, tuple[HarveyZaxyResult, Path]] = {}
    for path_value in _list_value(provenance.get("normalized_result_paths")):
        path = Path(str(path_value))
        if path.name != "normalized-result.json" or not path.exists():
            continue
        try:
            results = load_harvey_zaxy_results(path)
        except ValueError:
            continue
        for result in results:
            results_by_task[result.task_id] = (result, path.resolve())
    return results_by_task


def _zaxy_result_metadata_fingerprint(result: HarveyZaxyResult) -> tuple[object, ...]:
    return (
        result.framework,
        result.generator,
        result.judge,
        result.generator_reasoning_effort,
        result.judge_reasoning_effort,
        result.temperature,
        result.total_seconds,
        result.total_tokens,
        result.memory_search_calls,
        result.memory_read_calls,
        result.corpus_hash,
        result.commit,
    )


def _external_comparison_result_evidence_failure(
    provenance: dict[str, object],
    *,
    expected_run_config: dict[str, object | None] | None = None,
) -> str | None:
    saw_matching_frameworks = False
    saw_full_suite_frameworks = False
    saw_model_mismatch = False
    for path_value in _list_value(provenance.get("external_baseline_report_paths")):
        path = Path(str(path_value))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        aggregate = payload.get("aggregate")
        frameworks = aggregate.get("frameworks") if isinstance(aggregate, dict) else None
        aggregate_frameworks: set[str] = set()
        full_suite_frameworks: set[str] = set()
        if isinstance(frameworks, list):
            for item in frameworks:
                if not isinstance(item, dict):
                    continue
                framework = str(item.get("framework") or "").casefold()
                if framework == "zaxy" or _optional_float(item.get("avg_final_score")) is None:
                    continue
                aggregate_frameworks.add(framework)
                runs = _optional_int(item.get("runs"))
                if runs == len(ARTICLE_TASKS):
                    full_suite_frameworks.add(framework)
                elif runs is not None and runs > len(ARTICLE_TASKS):
                    return "external_comparison_not_full_suite"
        normalized_results = payload.get("normalized_results")
        result_scores_by_framework: dict[str, list[float]] = {}
        result_seconds_by_framework: dict[str, list[float]] = {}
        result_task_ids_by_framework: dict[str, set[str]] = {}
        result_model_configs_by_framework: dict[str, set[tuple[object, ...] | None]] = {}
        if isinstance(normalized_results, list):
            for item in normalized_results:
                if not isinstance(item, dict):
                    continue
                framework = str(item.get("framework") or "").casefold()
                if framework and framework != "zaxy":
                    try:
                        score = _score_fraction(item)
                    except ValueError:
                        continue
                    result_scores_by_framework.setdefault(framework, []).append(score)
                    seconds = _external_result_total_seconds(item)
                    if seconds is not None:
                        result_seconds_by_framework.setdefault(framework, []).append(seconds)
                    task_id = str(item.get("task_id") or "")
                    if task_id:
                        result_task_ids_by_framework.setdefault(framework, set()).add(task_id)
                    result_model_configs_by_framework.setdefault(framework, set()).add(
                        _external_result_model_config(item)
                    )
        if not aggregate_frameworks:
            continue
        if not (aggregate_frameworks & set(result_scores_by_framework)):
            continue
        saw_matching_frameworks = True
        if not full_suite_frameworks:
            continue
        saw_full_suite_frameworks = True
        if _external_comparison_missing_article_tasks(
            full_suite_frameworks,
            result_task_ids_by_framework,
        ):
            return "external_comparison_not_full_suite"
        if expected_run_config is not None and _external_comparison_model_mismatch(
            full_suite_frameworks,
            result_model_configs_by_framework,
            expected_run_config,
        ):
            saw_model_mismatch = True
            continue
        if _external_aggregate_matches_results(
            _external_comparison_framework_rows(frameworks, full_suite_frameworks),
            result_scores_by_framework,
            result_seconds_by_framework,
        ):
            return None
    if saw_model_mismatch:
        return "external_comparison_model_mismatch"
    if saw_matching_frameworks and not saw_full_suite_frameworks:
        return None
    return (
        "stale_external_comparison_result_evidence"
        if saw_matching_frameworks and saw_full_suite_frameworks
        else "missing_external_comparison_result_evidence"
    )


def _external_comparison_framework_rows(
    frameworks: object,
    allowed_frameworks: set[str],
) -> list[dict[str, object]]:
    if not isinstance(frameworks, list):
        return []
    rows: list[dict[str, object]] = []
    for item in frameworks:
        if not isinstance(item, dict):
            continue
        framework = str(item.get("framework") or "").casefold()
        if framework in allowed_frameworks:
            rows.append(item)
    return rows


def _external_comparison_has_full_suite_result_evidence(
    provenance: dict[str, object],
    *,
    expected_run_config: dict[str, object | None] | None,
) -> bool:
    for path_value in _list_value(provenance.get("external_baseline_report_paths")):
        path = Path(str(path_value))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        aggregate = payload.get("aggregate")
        frameworks = aggregate.get("frameworks") if isinstance(aggregate, dict) else None
        if not isinstance(frameworks, list):
            continue
        full_suite_frameworks = {
            str(item.get("framework") or "").casefold()
            for item in frameworks
            if isinstance(item, dict)
            and str(item.get("framework") or "").casefold() != "zaxy"
            and _optional_float(item.get("avg_final_score")) is not None
            and _optional_int(item.get("runs")) == len(ARTICLE_TASKS)
        }
        if not full_suite_frameworks:
            continue
        if (
            _external_comparison_result_evidence_failure(
                {"external_baseline_report_paths": [str(path)]},
                expected_run_config=expected_run_config,
            )
            is None
        ):
            return True
    return False


def _external_comparison_model_mismatch(
    aggregate_frameworks: set[str],
    result_model_configs_by_framework: dict[str, set[tuple[object, ...] | None]],
    expected_run_config: dict[str, object | None],
) -> bool:
    expected_config = (
        expected_run_config.get("generator"),
        expected_run_config.get("judge"),
        expected_run_config.get("generator_reasoning_effort"),
        expected_run_config.get("temperature"),
    )
    for framework in aggregate_frameworks:
        configs = result_model_configs_by_framework.get(framework)
        if not configs or configs != {expected_config}:
            return True
    return False


def _external_result_model_config(item: dict[str, object]) -> tuple[object, ...] | None:
    models = item.get("models")
    if isinstance(models, dict):
        generator = _optional_str(models.get("generator"))
        judge = _optional_str(models.get("judge"))
        generator_reasoning_effort = _optional_str(models.get("generator_reasoning_effort"))
        temperature = _optional_float(models.get("temperature"))
    else:
        generator = _optional_str(item.get("generator"))
        judge = _optional_str(item.get("judge"))
        generator_reasoning_effort = _optional_str(item.get("generator_reasoning_effort"))
        temperature = _optional_float(item.get("temperature"))
    if not generator or not judge:
        return None
    return (generator, judge, generator_reasoning_effort, temperature)


def _external_comparison_missing_article_tasks(
    aggregate_frameworks: set[str],
    result_task_ids_by_framework: dict[str, set[str]],
) -> bool:
    expected_task_ids = {task.task_id for task in ARTICLE_TASKS}
    for framework in aggregate_frameworks:
        task_ids = result_task_ids_by_framework.get(framework)
        if task_ids is None or task_ids != expected_task_ids:
            return True
    return False


def _external_aggregate_matches_results(
    frameworks: object,
    result_scores_by_framework: dict[str, list[float]],
    result_seconds_by_framework: dict[str, list[float]],
) -> bool:
    if not isinstance(frameworks, list):
        return False
    for item in frameworks:
        if not isinstance(item, dict):
            continue
        framework = str(item.get("framework") or "").casefold()
        if not framework or framework == "zaxy":
            continue
        mean_score = _optional_float(item.get("avg_final_score"))
        runs = _optional_int(item.get("runs"))
        if mean_score is None:
            continue
        result_scores = result_scores_by_framework.get(framework)
        if not result_scores:
            return False
        if runs is not None and len(result_scores) != runs:
            return False
        if _mean(result_scores) != round(mean_score, 3):
            return False
        mean_seconds = _optional_float(item.get("avg_total_seconds"))
        if mean_seconds is not None:
            result_seconds = result_seconds_by_framework.get(framework)
            if result_seconds is not None and _mean(result_seconds) != round(mean_seconds, 3):
                return False
    return True


def _external_result_total_seconds(item: dict[str, object]) -> float | None:
    seconds = _optional_float(item.get("total_seconds"))
    if seconds is not None:
        return seconds
    timing = item.get("timing")
    if isinstance(timing, dict):
        return _optional_float(timing.get("total_seconds"))
    return None


def _referenced_run_artifact_failure(
    result: HarveyZaxyResult,
    normalized_result_path: Path,
    roots: list[object],
) -> dict[str, str] | None:
    for key, path_value in (
        ("answer", result.answer_path),
        ("tool_log", result.tool_log_path),
        ("judge", result.judge_path),
        ("run_metrics", result.run_metrics_path),
    ):
        if not path_value:
            return {
                "task_id": result.task_id,
                "reason": "missing_reviewable_artifact_paths",
                "path_key": key,
            }
        resolved_artifact = _resolve_harvey_result_artifact(
            path_value,
            normalized_result_path,
            roots,
        )
        if resolved_artifact is None:
            return {
                "task_id": result.task_id,
                "reason": "missing_referenced_run_artifact",
                "path_key": key,
                "path": path_value,
            }
        expected_artifact = _expected_harvey_run_artifact_path(
            result,
            key,
            normalized_result_path,
        )
        if expected_artifact is None or resolved_artifact != expected_artifact:
            return {
                "task_id": result.task_id,
                "reason": "missing_referenced_run_artifact",
                "path_key": key,
                "path": path_value,
            }
        if key == "tool_log" and not _transcript_has_memory_tool_evidence(resolved_artifact):
            return {
                "task_id": result.task_id,
                "reason": "missing_transcript_memory_tool_evidence",
                "path_key": key,
                "path": path_value,
            }
        if key == "judge" and not _judge_score_matches_result(resolved_artifact, result):
            return {
                "task_id": result.task_id,
                "reason": "judge_score_mismatch",
                "path_key": key,
                "path": path_value,
            }
        if key == "run_metrics" and not _run_metrics_match_result(resolved_artifact, result):
            return {
                "task_id": result.task_id,
                "reason": "run_metrics_mismatch",
                "path_key": key,
                "path": path_value,
            }
    return None


def _expected_harvey_run_artifact_path(
    result: HarveyZaxyResult,
    path_key: str,
    normalized_result_path: Path,
) -> Path | None:
    if len(normalized_result_path.parents) < 4:
        return None
    worktree = normalized_result_path.resolve().parents[3]
    if path_key == "answer":
        if result.answer_path is None:
            return None
        answer_name = Path(result.answer_path).name
        if not answer_name:
            return None
        return (worktree / "results" / result.run_id / "output" / answer_name).resolve()
    relative_paths = {
        "tool_log": Path("results") / result.run_id / "transcript.jsonl",
        "judge": Path("results") / result.run_id / "scores.json",
        "run_metrics": Path("results") / result.run_id / "metrics.json",
    }
    relative_path = relative_paths.get(path_key)
    if relative_path is None:
        return None
    return (worktree / relative_path).resolve()


def _judge_score_matches_result(path: Path, result: HarveyZaxyResult) -> bool:
    try:
        scores = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(scores, dict):
        return False
    try:
        score = _score_fraction(scores)
    except ValueError:
        return False
    return score == result.score


def _run_metrics_match_result(path: Path, result: HarveyZaxyResult) -> bool:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metrics, dict):
        return False
    for key, expected in (
        ("memory_search_calls", result.memory_search_calls),
        ("memory_read_calls", result.memory_read_calls),
    ):
        if expected is None:
            continue
        actual = _optional_int(metrics.get(key))
        if actual != expected:
            return False
    return True


def _transcript_has_memory_tool_evidence(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("tool_name") in {"memory_search", "memory_read"}:
            return True
        tool_calls = event.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict) and call.get("name") in {"memory_search", "memory_read"}:
                    return True
    return False


def _resolve_harvey_result_artifact(
    path_value: str,
    normalized_result_path: Path,
    roots: list[object],
) -> Path | None:
    artifact_path = Path(path_value)
    candidate_bases = [
        Path(str(root_value))
        for root_value in roots
        if str(root_value)
    ]
    candidate_bases.append(normalized_result_path.parent)
    if artifact_path.is_absolute():
        resolved_artifact = artifact_path.resolve()
        if not resolved_artifact.exists():
            return None
        for base in candidate_bases:
            if _path_is_relative_to(resolved_artifact, base.resolve()):
                return resolved_artifact
        return None
    for base in candidate_bases:
        resolved_base = base.resolve()
        candidate = (resolved_base / artifact_path).resolve()
        if not _path_is_relative_to(candidate, resolved_base):
            continue
        if candidate.exists():
            return candidate
    return None


def _path_is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _harvey_result_paths(root: Path) -> list[Path]:
    root = root.resolve()
    if root.is_file():
        if root.name != "normalized-result.json":
            raise ValueError(f"Harvey result file must be named normalized-result.json: {root}")
        return [root]
    if not root.exists():
        raise ValueError(f"Harvey result root does not exist: {root}")
    direct = root / "normalized-result.json"
    if direct.exists():
        return [direct]
    ingestion_runs = root / ".ingestion" / "runs"
    if ingestion_runs.exists():
        return sorted(ingestion_runs.glob("*/normalized-result.json"))
    return sorted(root.glob("**/normalized-result.json"))


def _latest_harvey_result_items(
    roots: list[Path] | tuple[Path, ...],
) -> list[tuple[Path, HarveyZaxyResult]]:
    result_paths: list[Path] = []
    for root in roots:
        result_paths.extend(_harvey_result_paths(root))
    deduped_paths = sorted(dict.fromkeys(path.resolve() for path in result_paths))
    latest_by_task: dict[str, tuple[Path, HarveyZaxyResult]] = {}
    for path in deduped_paths:
        try:
            loaded = load_harvey_zaxy_results(path)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        for result in loaded:
            current = latest_by_task.get(result.task_id)
            candidate_key = (path.stat().st_mtime_ns, str(path))
            current_key = (
                -1,
                "",
            ) if current is None else (current[0].stat().st_mtime_ns, str(current[0]))
            if current is None or candidate_key >= current_key:
                latest_by_task[result.task_id] = (path, result)
    return [
        item
        for _, item in sorted(latest_by_task.items())
    ]


def _git_commit_or_none(root: Path) -> str | None:
    candidates = [root] if root.is_dir() else [root.parent]
    if root.is_dir() and not (root / ".git").exists():
        candidates.extend(root.parents)
    for candidate in candidates:
        if not (candidate / ".git").exists():
            continue
        try:
            completed = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        commit = completed.stdout.strip()
        return commit or None
    return None


def _git_branch_or_none(root: Path) -> str | None:
    candidates = [root] if root.is_dir() else [root.parent]
    if root.is_dir() and not (root / ".git").exists():
        candidates.extend(root.parents)
    for candidate in candidates:
        if not (candidate / ".git").exists():
            continue
        try:
            completed = subprocess.run(
                ["git", "-C", str(candidate), "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        branch = completed.stdout.strip()
        return branch or None
    return None


def _json_object_from_file(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Harvey {label} JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Harvey {label} JSON must be an object: {path}")
    return payload


def _score_fraction(scores: dict[str, object]) -> float:
    nested_scores = scores.get("scores")
    if isinstance(nested_scores, dict):
        return _score_fraction(nested_scores)
    if "criterion_pass_rate" in scores:
        return round(_number(scores.get("criterion_pass_rate"), "scores.criterion_pass_rate"), 3)
    if "final_score" in scores:
        return round(_number(scores.get("final_score"), "scores.final_score"), 3)
    if "answer_correctness" in scores:
        return round(_number(scores.get("answer_correctness"), "scores.answer_correctness"), 3)
    score = _number(scores.get("score"), "scores.score")
    max_score = _number(scores.get("max_score"), "scores.max_score")
    if max_score <= 0:
        raise ValueError("scores.max_score must be positive")
    return round(score / max_score, 3)


def _relative_path_string(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _list_value(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _manifest_eventloom_path(manifest: HarveyMemoryManifest) -> Path:
    value = manifest.get("eventloom_path")
    if not isinstance(value, str):
        raise ValueError("Harvey Zaxy memory manifest requires eventloom_path")
    return Path(value)


def _normalized_text(manifest: HarveyMemoryManifest) -> dict[str, object]:
    value = manifest.get("normalized_text")
    if isinstance(value, dict):
        return value
    return {}


def _source_map(manifest: HarveyMemoryManifest) -> dict[str, object]:
    source_map_path = _normalized_text(manifest).get("source_map")
    if not isinstance(source_map_path, str) or not source_map_path:
        return {}
    try:
        payload = json.loads(Path(source_map_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _display_source_path(manifest: HarveyMemoryManifest, storage_path: str) -> str:
    by_normalized = _source_map(manifest).get("by_normalized_path")
    if isinstance(by_normalized, dict):
        entry = by_normalized.get(storage_path)
        if isinstance(entry, dict) and isinstance(entry.get("original_path"), str):
            return str(entry["original_path"])
    return storage_path


def _storage_source_path(manifest: HarveyMemoryManifest, display_path: str) -> str:
    by_original = _source_map(manifest).get("by_original_path")
    if isinstance(by_original, dict):
        entry = by_original.get(display_path)
        if isinstance(entry, dict) and isinstance(entry.get("normalized_path"), str):
            return str(entry["normalized_path"])
    return display_path


def _source_identity(manifest: HarveyMemoryManifest) -> dict[str, object]:
    normalized = _normalized_text(manifest)
    return {
        "indexed_corpus": "normalized_text" if normalized else "original",
        "corpus_root": normalized.get("corpus_root") or manifest.get("corpus_root"),
        "source_map": normalized.get("source_map"),
    }


def _memory_item_id(source_path: str, start_line: int | None, end_line: int | None) -> str:
    if start_line is None or end_line is None:
        return source_path
    return f"{source_path}:{start_line}-{end_line}"


def _parse_memory_item_id(item_id: str) -> tuple[str, int, int]:
    source_path, sep, suffix = item_id.partition(":")
    if not sep:
        raise ValueError(f"memory_read id not found: {item_id}")
    start_text, dash, end_text = suffix.partition("-")
    if not dash:
        raise ValueError(f"memory_read id not found: {item_id}")
    try:
        start_line = int(start_text)
        end_line = int(end_text)
    except ValueError as exc:
        raise ValueError(f"memory_read id not found: {item_id}") from exc
    return source_path, start_line, end_line


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Harvey normalized result contract requires {name} to be an object")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"Harvey normalized result contract requires numeric {name}")
    return float(value)


def _required_float(value: object, name: str) -> float:
    return _number(value, name)


def _required_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Harvey LAB report requires integer {name}")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return ""
    if signed:
        return f"{value:+.3f}"
    return f"{value:.3f}"


def _fmt_na(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return _fmt(value, signed=signed)


def _fmt_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)
