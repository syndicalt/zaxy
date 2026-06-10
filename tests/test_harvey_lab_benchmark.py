"""Tests for Harvey LAB external memory benchmark reporting."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy_benchmarks.harvey_lab_benchmark import (
    ARTICLE_FRAMEWORK_FIT,
    ARTICLE_TASKS,
    HARVEY_LAB_REPO_URL,
    _harvey_result_paths,
    _json_object_from_file,
    _score_fraction,
    build_harvey_external_index_preflight,
    build_harvey_external_run_manifest,
    build_harvey_external_run_readiness,
    build_harvey_external_run_script,
    build_harvey_external_run_status,
    build_harvey_lab_report,
    build_harvey_normalized_result_from_run,
    build_harvey_result_provenance,
    build_harvey_zaxy_memory_index,
    check_harvey_external_suite,
    check_harvey_lab_completion,
    export_harvey_adapter_kit,
    harvey_memory_read,
    harvey_memory_search,
    import_harvey_zaxy_results,
    load_harvey_zaxy_results,
    render_harvey_publication_markdown,
    report_to_markdown,
    validate_harvey_lab_report,
    write_harvey_lab_report,
)


def _zaxy_result(
    task_id: str,
    score: float,
    *,
    run_id: str | None = None,
    seconds: float = 100.0,
    tokens: int = 500_000,
    memory_search_calls: int = 4,
    memory_read_calls: int = 2,
    include_paths: bool = True,
    generator: str = "openai-compatible/gpt-5.5",
    judge: str = "gpt-5.4-mini",
    generator_reasoning_effort: str | None = "low",
    temperature: float = 0.0,
    commit: str = "29748828133dff83ad2263af353fb035504f8f77",
) -> dict[str, object]:
    result = {
        "schema_version": "harvey-memory-ablation-v1",
        "run_id": run_id or f"zaxy-{task_id.replace('/', '__')}",
        "framework": "zaxy",
        "task_id": task_id,
        "corpus_hash": "external-corpus-hash",
        "branch": "zaxy-harvey-lab",
        "commit": commit,
        "models": {
            "generator": generator,
            "judge": judge,
            "endpoint": "http://127.0.0.1:8318/v1",
            "generator_reasoning_effort": generator_reasoning_effort,
            "judge_reasoning_effort": None,
            "temperature": temperature,
            "embedding": "zaxy-hash",
            "embedding_endpoint": None,
            "embedding_backend": "zaxy",
            "embedding_dimension": 1536,
            "embedding_device": "cpu",
        },
        "scores": {
            "answer_correctness": score,
            "final_score": score,
        },
        "timing": {"total_seconds": seconds},
        "usage": {"total_tokens": tokens},
        "cost": {"estimated_usd": None},
        "tooling": {
            "memory_search_calls": memory_search_calls,
            "memory_read_calls": memory_read_calls,
            "empty_memory_searches": 0,
        },
        "retrieval": {"citation_recall": 1.0},
        "failure_modes": [],
        "qualitative_notes": "fixture",
    }
    if include_paths:
        resolved_run_id = run_id or f"zaxy-{task_id.replace('/', '__')}"
        result["paths"] = {
            "results_run_dir": f"results/{resolved_run_id}",
            "answer": f"results/{resolved_run_id}/output/response.md",
            "tool_log": f"results/{resolved_run_id}/transcript.jsonl",
            "judge": f"results/{resolved_run_id}/scores.json",
            "run_metrics": f"results/{resolved_run_id}/metrics.json",
        }
    return result


def _write_normalized_result_paths(
    root: Path,
    task_scores: dict[str, float],
    *,
    create_run_artifacts: bool = True,
) -> list[str]:
    paths: list[str] = []
    for task_id, score in task_scores.items():
        run_dir = root / f"zaxy-{task_id.replace('/', '__')}"
        run_dir.mkdir(parents=True)
        path = run_dir / "normalized-result.json"
        result = _zaxy_result(task_id, score)
        path.write_text(json.dumps(result), encoding="utf-8")
        if create_run_artifacts:
            worktree = root.parent.parent
            result_paths = result["paths"]
            assert isinstance(result_paths, dict)
            for key in ("answer", "tool_log", "judge", "run_metrics"):
                artifact_path = worktree / str(result_paths[key])
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                if key == "tool_log":
                    artifact_path.write_text(
                        json.dumps(
                            {
                                "turn": 1,
                                "role": "tool",
                                "tool_name": "memory_search",
                                "arguments": {"query": "consent"},
                                "result_preview": '{"hits": [{"id": "memo.txt:1-2"}]}',
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                elif key == "run_metrics":
                    tooling = result["tooling"]
                    usage = result["usage"]
                    timing = result["timing"]
                    assert isinstance(tooling, dict)
                    assert isinstance(usage, dict)
                    assert isinstance(timing, dict)
                    artifact_path.write_text(
                        json.dumps(
                            {
                                "memory_search_calls": tooling["memory_search_calls"],
                                "memory_read_calls": tooling["memory_read_calls"],
                                "total_tokens": usage["total_tokens"],
                                "total_seconds": timing["total_seconds"],
                            }
                        ),
                        encoding="utf-8",
                    )
                elif key == "judge":
                    scores = result["scores"]
                    assert isinstance(scores, dict)
                    artifact_path.write_text(json.dumps(scores), encoding="utf-8")
                else:
                    artifact_path.write_text(f"{task_id} {key}\n", encoding="utf-8")
        paths.append(str(path))
    return paths


def _write_harvey_run_gate_artifact_paths(
    root: Path,
    *,
    readiness_status: str = "ready_for_external_runs",
    run_status: str = "complete",
    task_count: int = 10,
    readiness_expected_task_count: int = 10,
    status_expected_task_count: int = 10,
    status_import_ready_count: int = 10,
    status_include_tasks: bool = True,
    status_task_import_ready_overrides: dict[str, bool] | None = None,
    status_normalized_result_path_overrides: dict[str, str] | None = None,
    status_index_dir_overrides: dict[str, str] | None = None,
    status_run_dir_overrides: dict[str, str] | None = None,
    status_run_id_overrides: dict[str, str] | None = None,
    manifest_generator: str = "openai-compatible/gpt-5.5",
    manifest_judge: str = "gpt-5.4-mini",
    manifest_reasoning_effort: str | None = "low",
    manifest_source_url: str = HARVEY_LAB_REPO_URL,
    manifest_article_url: str = "https://rushilchugh.substack.com/p/what-agent-memory-actually-fixes",
    manifest_task_ids: list[str] | None = None,
    manifest_expected_normalized_results: list[str] | None = None,
    manifest_run_commands: list[str] | None = None,
    manifest_judge_commands: list[str] | None = None,
    manifest_collection_command: str | None = None,
    manifest_comparison_command: str | None = None,
    readiness_worktree: str | None = None,
    readiness_generator: str | None = None,
    readiness_judge: str | None = None,
    readiness_blocking_reasons: list[str] | None = None,
    readiness_missing_credentials: list[str] | None = None,
    status_worktree: str | None = None,
    readiness_commit: str | None = "29748828133dff83ad2263af353fb035504f8f77",
    status_commit: str | None = "29748828133dff83ad2263af353fb035504f8f77",
) -> dict[str, list[str]]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "harvey-lab-external-run.json"
    ready = root / "harvey-lab-ready.json"
    status = root / "harvey-lab-status.json"
    task_ids = (
        manifest_task_ids
        if manifest_task_ids is not None
        else [task.task_id for task in ARTICLE_TASKS]
    )
    expected_normalized_results = (
        manifest_expected_normalized_results
        if manifest_expected_normalized_results is not None
        else [
            f".ingestion/runs/zaxy-{task_id.replace('/', '__')}/normalized-result.json"
            for task_id in task_ids
        ]
    )
    judge_commands = (
        manifest_judge_commands
        if manifest_judge_commands is not None
        else [
            (
                f"uv run python -m evaluation.run_eval --judge-model {manifest_judge} "
                f"--run-id zaxy-{task_id.replace('/', '__')} --task {task_id}"
            )
            for task_id in task_ids
        ]
    )
    run_commands = (
        manifest_run_commands
        if manifest_run_commands is not None
        else [
            (
                f"uv run python -m harness.run --model {manifest_generator} "
                f"--task {task_id} --run-id zaxy-{task_id.replace('/', '__')}"
                + (
                    f" --reasoning-effort {manifest_reasoning_effort}"
                    if manifest_reasoning_effort is not None
                    else ""
                )
            )
            for task_id in task_ids
        ]
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "zaxy.harvey-lab-external-run.v1",
                "source_url": manifest_source_url,
                "article_url": manifest_article_url,
                "task_count": task_count,
                "generator": manifest_generator,
                "judge": manifest_judge,
                "reasoning_effort": manifest_reasoning_effort,
                "collection_command": manifest_collection_command
                or (
                    "uv run python scripts/memory_ablation/collect_results.py "
                    "--worktree . --dedupe-latest --output .ingestion/reports/comparison-zaxy.json"
                ),
                "comparison_command": manifest_comparison_command
                or (
                    "zaxy harvey-lab-import path/to/harvey-zaxy-worktree "
                    "--output-dir reports/benchmarks/harvey-lab-memory-ablation"
                ),
                "report_json_path": "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json",
                "publish_output_path": (
                    "reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md"
                ),
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
                "tasks": [
                    {
                        "task_id": task_id,
                        "run_command": run_command,
                        "judge_command": judge_command,
                        "normalize_command": (
                            "zaxy harvey-lab-normalize-run --harvey-worktree . "
                            f"--run-id zaxy-{task_id.replace('/', '__')} --task-id {task_id}"
                        ),
                        "validate_command": (
                            "uv run python scripts/memory_ablation/validate_result.py "
                            f"--run-dir .ingestion/runs/zaxy-{task_id.replace('/', '__')} --worktree-root ."
                        ),
                        "expected_normalized_result": expected_normalized_result,
                    }
                    for task_id, expected_normalized_result, run_command, judge_command in zip(
                        task_ids,
                        expected_normalized_results,
                        run_commands,
                        judge_commands,
                        strict=True,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    readiness_payload = {
        "schema_version": "zaxy.harvey-lab-run-readiness.v1",
        "status": readiness_status,
        "worktree": readiness_worktree or str(root.resolve()),
        "source_url": HARVEY_LAB_REPO_URL,
        "article_url": "https://rushilchugh.substack.com/p/what-agent-memory-actually-fixes",
        "suite_valid": True,
        "task_filter": "",
        "resolved_task_id": None,
        "expected_task_count": readiness_expected_task_count,
        "index_ready_count": readiness_expected_task_count,
        "ready_task_count": 0,
        "run_ready_count": 0,
        "normalized_ready_count": 0,
        "evidence_audit": {
            "index_ready_count": readiness_expected_task_count,
            "run_artifacts_ready_count": 0,
            "normalized_result_ready_count": 0,
            "memory_evidence_ready_count": 0,
            "import_ready_count": 0,
            "expected_task_count": readiness_expected_task_count,
        },
        "model_requirements": {
            "generator": {
                "model": readiness_generator or manifest_generator,
                "provider": "openai-compatible",
                "credential_status": "not_required",
                "missing_credentials": [],
            },
            "judge": {
                "model": readiness_judge or manifest_judge,
                "provider": "openai",
                "credential_status": "present",
                "missing_credentials": [],
            },
        },
        "missing_credentials": readiness_missing_credentials or [],
        "blocking_reasons": readiness_blocking_reasons or [],
    }
    if readiness_commit is not None:
        readiness_payload["harvey_git_commit"] = readiness_commit
    ready.write_text(json.dumps(readiness_payload), encoding="utf-8")
    status_payload: dict[str, object] = {
        "schema_version": "zaxy.harvey-lab-run-status.v1",
        "status": run_status,
        "worktree": status_worktree or str(root.resolve()),
        "expected_task_count": status_expected_task_count,
        "evidence_audit": {"import_ready_count": status_import_ready_count},
    }
    if status_include_tasks:
        import_ready_overrides = status_task_import_ready_overrides or {}
        normalized_path_overrides = status_normalized_result_path_overrides or {}
        index_dir_overrides = status_index_dir_overrides or {}
        run_dir_overrides = status_run_dir_overrides or {}
        run_id_overrides = status_run_id_overrides or {}
        status_payload["tasks"] = {
            task.task_id: {
                "task_id": task.task_id,
                "run_id": run_id_overrides.get(task.task_id)
                or f"zaxy-{task.task_id.replace('/', '__')}",
                "index_dir": index_dir_overrides.get(task.task_id)
                or str(
                    root.resolve()
                    / ".ingestion"
                    / "indexes"
                    / task.task_id.replace("/", "__")
                    / "zaxy"
                ),
                "run_dir": run_dir_overrides.get(task.task_id)
                or str(
                    root.resolve()
                    / "results"
                    / (
                        run_id_overrides.get(task.task_id)
                        or f"zaxy-{task.task_id.replace('/', '__')}"
                    )
                ),
                "index_ready": True,
                "run_artifacts_ready": True,
                "normalized_result_ready": True,
                "import_ready": import_ready_overrides.get(task.task_id, True),
                "normalized_result_path": normalized_path_overrides.get(task.task_id) or str(
                    root.resolve()
                    / ".ingestion"
                    / "runs"
                    / f"zaxy-{task.task_id.replace('/', '__')}"
                    / "normalized-result.json"
                ),
            }
            for task in ARTICLE_TASKS
        }
    if status_commit is not None:
        status_payload["harvey_git_commit"] = status_commit
    status.write_text(json.dumps(status_payload), encoding="utf-8")
    return {
        "external_run_manifest_paths": [str(manifest)],
        "external_readiness_report_paths": [str(ready)],
        "external_status_report_paths": [str(status)],
    }


def _external_comparison_normalized_results() -> list[dict[str, object]]:
    return [
        *[
            {
                "framework": "mem0",
                "task_id": task.task_id,
                "final_score": 0.456,
                "total_seconds": 90.0,
                "models": {
                    "generator": "openai-compatible/gpt-5.5",
                    "judge": "gpt-5.4-mini",
                    "generator_reasoning_effort": "low",
                    "temperature": 0.0,
                },
            }
            for task in ARTICLE_TASKS
        ],
        *[
            {
                "framework": "raw-rg",
                "task_id": task.task_id,
                "final_score": 0.399,
                "total_seconds": 40.0,
                "models": {
                    "generator": "openai-compatible/gpt-5.5",
                    "judge": "gpt-5.4-mini",
                    "generator_reasoning_effort": "low",
                    "temperature": 0.0,
                },
            }
            for task in ARTICLE_TASKS
        ],
    ]


def _partial_external_comparison_normalized_results() -> list[dict[str, object]]:
    return [
        *[
            {
                "framework": "mem0",
                "task_id": task.task_id,
                "final_score": 0.456,
                "total_seconds": 90.0,
                "models": {
                    "generator": "openai-compatible/gpt-5.5",
                    "judge": "gpt-5.4-mini",
                    "generator_reasoning_effort": "low",
                    "temperature": 0.0,
                },
            }
            for task in ARTICLE_TASKS[:4]
        ],
        *[
            {
                "framework": "raw-rg",
                "task_id": task.task_id,
                "final_score": 0.399,
                "total_seconds": 40.0,
                "models": {
                    "generator": "openai-compatible/gpt-5.5",
                    "judge": "gpt-5.4-mini",
                    "generator_reasoning_effort": "low",
                    "temperature": 0.0,
                },
            }
            for task in ARTICLE_TASKS[:4]
        ],
    ]


def _write_valid_harvey_worktree(root: Path) -> Path:
    bin_dir = root / ".fixture-bin"
    bin_dir.mkdir(parents=True)
    uv = bin_dir / "uv"
    uv.write_text(
        """#!/usr/bin/env sh
set -eu
if [ "${1:-}" = "run" ]; then
  shift
fi
exec "$@"
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    path = os.environ.get("PATH", "")
    bin_prefix = str(bin_dir)
    if not path.split(os.pathsep) or path.split(os.pathsep)[0] != bin_prefix:
        os.environ["PATH"] = f"{bin_prefix}{os.pathsep}{path}" if path else bin_prefix
    for task in ARTICLE_TASKS:
        task_dir = root / "tasks" / task.task_id
        docs_dir = task_dir / "documents"
        docs_dir.mkdir(parents=True)
        criteria = [{"match_criteria": "PASS"} for _ in range(task.criteria_count)]
        (task_dir / "task.json").write_text(json.dumps({"criteria": criteria}), encoding="utf-8")
        for idx in range(task.document_count):
            (docs_dir / f"doc-{idx}.txt").write_text(
                f"{task.article_label} evidence document {idx}\nmaterial clause and timeline facts\n",
                encoding="utf-8",
            )
    scripts = root / "scripts" / "memory_ablation"
    scripts.mkdir(parents=True)
    for name in (
        "validate_result.py",
        "collect_results.py",
        "render_report.py",
    ):
        (scripts / name).write_text("# fixture\n", encoding="utf-8")
    (scripts / "normalize_corpus.py").write_text(
        """
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def prepare_normalized_corpus(corpus_root: Path, ingestion_root: Path) -> dict:
    files = sorted(path for path in corpus_root.iterdir() if path.is_file())
    encoded = json.dumps([path.name for path in files]).encode("utf-8")
    corpus_hash = hashlib.sha256(encoded).hexdigest()
    normalized_root = ingestion_root / "corpora" / corpus_hash / "txt"
    if normalized_root.exists():
        shutil.rmtree(normalized_root)
    normalized_root.mkdir(parents=True)
    source_map = {"by_normalized_path": {}, "by_original_path": {}}
    for path in files:
        normalized_name = f"{path.name}.txt"
        normalized_path = normalized_root / normalized_name
        normalized_path.write_text(
            f"Source-Path: {path.name}\\n\\n{path.read_text(encoding='utf-8')}",
            encoding="utf-8",
        )
        entry = {"original_path": path.name, "normalized_path": normalized_name}
        source_map["by_normalized_path"][normalized_name] = entry
        source_map["by_original_path"][path.name] = entry
    source_map_path = normalized_root.parent / "source-map.json"
    source_map_path.write_text(json.dumps(source_map), encoding="utf-8")
    return {
        "normalized_corpus_root": str(normalized_root),
        "source_map_path": str(source_map_path),
        "original_corpus_hash": corpus_hash,
        "normalized_corpus_hash": corpus_hash,
        "conversion_errors": [],
    }
""".lstrip(),
        encoding="utf-8",
    )
    (root / "harness").mkdir(parents=True)
    (root / "harness" / "run.py").write_text("# fixture\n", encoding="utf-8")
    (root / "evaluation").mkdir(parents=True)
    (root / "evaluation" / "run_eval.py").write_text("# fixture\n", encoding="utf-8")
    return root


def test_article_baselines_are_fixed_to_the_external_harvey_suite() -> None:
    """The built-in comparison set should be the article's external LAB suite."""
    assert HARVEY_LAB_REPO_URL == "https://github.com/rushilchugh01/harvey-labs-ablations-and-benchmarks"
    assert len(ARTICLE_TASKS) == 10
    assert {task.task_shape for task in ARTICLE_TASKS} == {
        "Compact legal-risk synthesis",
        "Sparse clause hunt",
        "Broad diligence sweep",
        "Red-flag spotting",
        "Compliance mapping",
        "Event reconstruction",
        "Document-by-document coding",
        "Production-set classification",
        "Large log-heavy classification",
        "Request matching",
    }
    assert any(task.best_framework == "raw-rg" for task in ARTICLE_TASKS)
    assert any(task.best_framework == "ActiveGraph" for task in ARTICLE_TASKS)
    assert ARTICLE_FRAMEWORK_FIT["raw-rg"].where_strongest == "Literal evidence finding"
    assert "not a no-memory baseline" in ARTICLE_FRAMEWORK_FIT["raw-rg"].interpretation


def test_harvey_report_compares_zaxy_against_article_task_winners() -> None:
    """Zaxy rows should be compared to regular, raw search, and best article rows."""
    zaxy_results = [
        _zaxy_result(
            "corporate-ma/review-data-room-red-flag-review",
            0.64,
            seconds=120.0,
            tokens=400,
            memory_search_calls=3,
            memory_read_calls=2,
        ),
        _zaxy_result(
            "litigation-dispute-resolution/review-privilege-log-clawback-review",
            0.62,
            seconds=180.0,
            tokens=600,
            memory_search_calls=5,
            memory_read_calls=4,
        ),
    ]

    report = build_harvey_lab_report(zaxy_results)

    assert report.status == "partial"
    assert report.summary.zaxy_task_count == 2
    assert report.summary.zaxy_mean_score == 0.63
    assert report.summary.mean_delta_vs_regular_no_memory == 0.169
    assert report.summary.mean_delta_vs_article_best == 0.031
    assert report.summary.zaxy_mean_total_seconds == 150.0
    assert report.summary.zaxy_total_tokens == 1000
    assert report.summary.zaxy_total_memory_search_calls == 8
    assert report.summary.zaxy_total_memory_read_calls == 6
    red_flag = report.task_rows["corporate-ma/review-data-room-red-flag-review"]
    assert red_flag.zaxy_score == 0.64
    assert red_flag.article_best_framework == "LightRAG"
    assert red_flag.zaxy_delta_vs_article_best == 0.04
    assert red_flag.zaxy_winner is True
    privilege = report.task_rows["litigation-dispute-resolution/review-privilege-log-clawback-review"]
    assert privilege.article_best_framework == "GBrain keyword"
    assert privilege.zaxy_delta_vs_regular_no_memory == 0.218


def test_harvey_publication_markdown_includes_zaxy_runtime_and_usage_statistics(tmp_path: Path) -> None:
    """Publishable statistics should include Zaxy timing, token, and memory-tool aggregates."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    zaxy_rows = [
        _zaxy_result(
            task_id,
            score,
            seconds=90.0 + index,
            tokens=1000 + index,
            memory_search_calls=2 + index,
            memory_read_calls=1 + index,
        )
        for index, (task_id, score) in enumerate(task_scores.items())
    ]
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    zaxy_by_task = {
        str(row["task_id"]): row
        for row in zaxy_rows
    }
    for normalized_path in normalized_paths:
        path = Path(normalized_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        zaxy_row = zaxy_by_task[str(payload["task_id"])]
        path.write_text(
            json.dumps(zaxy_row),
            encoding="utf-8",
        )
        result_paths = zaxy_row["paths"]
        assert isinstance(result_paths, dict)
        metrics_path = external / str(result_paths["run_metrics"])
        metrics_path.write_text(
            json.dumps(
                {
                    "memory_search_calls": zaxy_row["tooling"]["memory_search_calls"],
                    "memory_read_calls": zaxy_row["tooling"]["memory_read_calls"],
                    "total_tokens": zaxy_row["usage"]["total_tokens"],
                    "total_seconds": zaxy_row["timing"]["total_seconds"],
                }
            ),
            encoding="utf-8",
        )
    report = build_harvey_lab_report(
        zaxy_rows,
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external),
        },
    )

    markdown = render_harvey_publication_markdown(report)

    assert "| Mean total seconds | Total tokens | Memory search calls | Memory read calls |" in markdown
    assert "| 94.500 | 10045 | 65 | 55 |" in markdown
    assert "## Evidence Audit" in markdown
    assert "Zaxy normalized-result artifacts" in markdown
    assert "Judge score artifacts match normalized scores" in markdown
    assert "Run metrics artifacts match memory-call totals" in markdown
    assert "Transcripts contain memory tool evidence" in markdown
    assert "External comparison aggregates recomputed from non-Zaxy result rows" in markdown
    assert "External run manifest artifacts" in markdown
    assert "External readiness report artifacts" in markdown
    assert "External status report artifacts" in markdown
    assert "External run audit artifacts are complete and commit-consistent" in markdown


def test_harvey_report_can_record_external_result_provenance(tmp_path: Path) -> None:
    """Reports should preserve where external Harvey result artifacts came from."""
    root = tmp_path / "harvey"
    result_path = root / ".ingestion" / "runs" / "zaxy-red-flags" / "normalized-result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)),
        encoding="utf-8",
    )

    report = build_harvey_lab_report(
        import_harvey_zaxy_results([root]),
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(root)],
            "normalized_result_paths": [str(result_path)],
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
        },
    )
    written = write_harvey_lab_report(report, tmp_path / "out")
    payload = json.loads(written.json_path.read_text(encoding="utf-8"))

    assert payload["result_provenance"]["source"] == "harvey-lab-import"
    assert payload["result_provenance"]["harvey_git_commit"] == "29748828133dff83ad2263af353fb035504f8f77"
    assert str(result_path) in payload["result_provenance"]["normalized_result_paths"]
    assert "Result Provenance" in written.markdown_path.read_text(encoding="utf-8")


def test_harvey_report_includes_article_framework_scorecard() -> None:
    """Framework statistics should reflect published article score coverage."""
    zaxy_results = [
        _zaxy_result(task.task_id, task.best_score + 0.01)
        for task in ARTICLE_TASKS
    ]

    report = build_harvey_lab_report(zaxy_results)

    regular = report.framework_scorecard["regular no-memory"]
    assert regular.evidence_scope == "article regular baseline across all ten tasks"
    assert regular.article_task_count == 10
    assert regular.mean_score == 0.604

    article_best = report.framework_scorecard["article best observed"]
    assert article_best.article_task_count == 10
    assert article_best.mean_score == 0.707

    gbrain = report.framework_scorecard["GBrain keyword"]
    assert gbrain.evidence_scope == "article task-winner matrix only"
    assert gbrain.article_task_count == 4
    assert gbrain.mean_score == 0.721
    assert gbrain.zaxy_overlap_task_count == 4
    assert gbrain.zaxy_delta_on_overlap == 0.01

    mem0 = report.framework_scorecard["Mem0"]
    assert mem0.article_task_count == 0
    assert mem0.mean_score is None
    assert mem0.evidence_scope == "framework fit only; no published task-winning score"

    zaxy = report.framework_scorecard["Zaxy"]
    assert zaxy.evidence_scope == "same-harness external Zaxy normalized results"
    assert zaxy.article_task_count == 10
    assert zaxy.mean_score == 0.717


def test_harvey_loader_rejects_non_external_or_unknown_suite_results(tmp_path: Path) -> None:
    """Internal synthetic Zaxy benchmark rows must not masquerade as Harvey LAB."""
    payload_path = tmp_path / "zaxy-results.json"
    payload_path.write_text(
        json.dumps(
            {
                "normalized_results": [
                    _zaxy_result("longmemeval-001", 0.9),
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        load_harvey_zaxy_results(payload_path)
    except ValueError as exc:
        assert "not part of the Harvey LAB article suite" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected unknown task rejection")

    malformed_path = tmp_path / "malformed.json"
    malformed = _zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.6)
    malformed.pop("paths")
    malformed_path.write_text(json.dumps([malformed]), encoding="utf-8")

    try:
        load_harvey_zaxy_results(malformed_path)
    except ValueError as exc:
        assert "Harvey normalized result contract" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected contract rejection")


def test_harvey_loader_preserves_generation_settings(tmp_path: Path) -> None:
    """Normalized rows should preserve generator reasoning effort and temperature."""
    payload_path = tmp_path / "zaxy-results.json"
    payload_path.write_text(
        json.dumps(
            _zaxy_result(
                "corporate-ma/review-data-room-red-flag-review",
                0.64,
                generator_reasoning_effort="medium",
                temperature=0.2,
            )
        ),
        encoding="utf-8",
    )

    result = load_harvey_zaxy_results(payload_path)[0]

    assert result.generator_reasoning_effort == "medium"
    assert result.temperature == 0.2


def test_import_harvey_zaxy_results_discovers_external_normalized_result_tree(tmp_path: Path) -> None:
    """An external Harvey worktree should be importable without hand-built JSON."""
    runs = tmp_path / ".ingestion" / "runs"
    first = runs / "zaxy-one"
    second = runs / "zaxy-two"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "normalized-result.json").write_text(
        json.dumps(_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)),
        encoding="utf-8",
    )
    (second / "normalized-result.json").write_text(
        json.dumps(_zaxy_result("litigation-dispute-resolution/review-privilege-log-clawback-review", 0.62)),
        encoding="utf-8",
    )

    imported = import_harvey_zaxy_results([tmp_path])

    assert [result.task_id for result in imported] == [
        "corporate-ma/review-data-room-red-flag-review",
        "litigation-dispute-resolution/review-privilege-log-clawback-review",
    ]
    assert imported[0].framework == "zaxy"


def test_import_harvey_zaxy_results_dedupes_to_latest_task_artifact(tmp_path: Path) -> None:
    """Rerun artifacts for the same task should resolve to one latest row."""
    task_id = "corporate-ma/review-data-room-red-flag-review"
    old_run = tmp_path / ".ingestion" / "runs" / "zaxy-red-old"
    new_run = tmp_path / ".ingestion" / "runs" / "zaxy-red-new"
    old_run.mkdir(parents=True)
    new_run.mkdir(parents=True)
    old_path = old_run / "normalized-result.json"
    new_path = new_run / "normalized-result.json"
    old_path.write_text(json.dumps(_zaxy_result(task_id, 0.40)), encoding="utf-8")
    new_path.write_text(json.dumps(_zaxy_result(task_id, 0.64)), encoding="utf-8")
    old_time = 1_700_000_000
    new_time = old_time + 60
    old_path.touch()
    new_path.touch()
    os.utime(old_path, (old_time, old_time))
    os.utime(new_path, (new_time, new_time))

    imported = import_harvey_zaxy_results([tmp_path])
    provenance = build_harvey_result_provenance([tmp_path], source="harvey-lab-import")

    assert len(imported) == 1
    assert imported[0].task_id == task_id
    assert imported[0].score == 0.64
    assert provenance["normalized_result_paths"] == [str(new_path.resolve())]


def test_harvey_provenance_records_external_baseline_reports_without_unlocking_gate(tmp_path: Path) -> None:
    """Harvey-native framework reports are baseline provenance, not Zaxy result evidence."""
    reports_dir = tmp_path / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison-gpt54mini-flashlite.json"
    comparison_path.write_text(
        json.dumps(
            {
                "schema_version": "harvey.memory_comparison.v1",
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": [],
            }
        ),
        encoding="utf-8",
    )

    provenance = build_harvey_result_provenance([tmp_path], source="harvey-lab-import")
    report = build_harvey_lab_report([], result_provenance=provenance)
    validation = validate_harvey_lab_report(report)

    assert provenance["external_baseline_report_paths"] == [str(comparison_path.resolve())]
    assert provenance["external_baseline_reports"] == [
        {
            "path": str(comparison_path.resolve()),
            "schema_version": "harvey.memory_comparison.v1",
            "framework_count": 2,
            "normalized_result_count": 0,
        }
    ]
    assert check_harvey_lab_completion(report)["status"] == "blocked"
    assert validation["status"] == "valid"
    assert validation["evidence_audit"]["normalized_result_artifacts"] == 0
    assert validation["evidence_audit"]["judge_score_artifacts_match"] is False
    assert validation["evidence_audit"]["run_metrics_artifacts_match"] is False
    assert validation["evidence_audit"]["transcript_memory_tool_evidence"] is False
    markdown = report_to_markdown(report)
    assert "External baseline reports" in markdown
    assert "comparison-gpt54mini-flashlite.json" in markdown
    assert str(comparison_path.resolve()) not in markdown
    assert report.external_baseline_scorecard["mem0"].mean_score == 0.456
    assert report.external_baseline_scorecard["mem0"].delta_vs_raw_rg == 0.057
    assert report.external_baseline_scorecard["raw-rg"].delta_vs_raw_rg == 0.0
    assert report.external_comparison_scorecard["Zaxy"].evidence_scope == (
        "pending external Zaxy normalized results"
    )
    assert report.external_comparison_scorecard["Zaxy"].mean_score is None
    assert report.external_comparison_scorecard["Zaxy"].rank_by_mean_score is None
    assert "## External Baseline Aggregate" in markdown
    assert "## Zaxy vs External Scored Systems" in markdown
    assert "Delta vs source raw-rg" in markdown
    assert "Mean seconds" in markdown
    assert "| Zaxy | pending external Zaxy normalized results | 0 | n/a | n/a | n/a | n/a | n/a |" in markdown
    assert "| mem0 | Harvey-native comparison artifact | 10 | 0.456 | +0.057 |" in markdown


def test_harvey_provenance_records_external_run_gate_artifacts(tmp_path: Path) -> None:
    """Generated run manifests and gate JSON should travel with imported reports."""
    manifest_path = tmp_path / "harvey-lab-external-run.json"
    ready_path = tmp_path / "harvey-lab-ready.json"
    status_path = tmp_path / "harvey-lab-status.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "zaxy.harvey-lab-external-run.v1", "task_count": 10}),
        encoding="utf-8",
    )
    ready_path.write_text(
        json.dumps({"schema_version": "zaxy.harvey-lab-run-readiness.v1", "status": "not_ready"}),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps({"schema_version": "zaxy.harvey-lab-run-status.v1", "status": "not_ready"}),
        encoding="utf-8",
    )

    provenance = build_harvey_result_provenance([tmp_path], source="harvey-lab-import")
    report = build_harvey_lab_report([], result_provenance=provenance)
    markdown = report_to_markdown(report)

    assert provenance["external_run_manifest_paths"] == [str(manifest_path.resolve())]
    assert provenance["external_readiness_report_paths"] == [str(ready_path.resolve())]
    assert provenance["external_status_report_paths"] == [str(status_path.resolve())]
    assert "External run manifests" in markdown
    assert "harvey-lab-external-run.json" in markdown
    assert str(manifest_path.resolve()) not in markdown
    assert "harvey-lab-ready.json" in markdown
    assert "harvey-lab-status.json" in markdown
    assert str(ready_path.resolve()) not in markdown
    assert str(status_path.resolve()) not in markdown


def test_harvey_report_ranks_zaxy_against_external_scored_systems(tmp_path: Path) -> None:
    """Reports should expose Zaxy rank and deltas against Harvey-native scored aggregates."""
    reports_dir = tmp_path / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456, "avg_total_seconds": 90.0},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399, "avg_total_seconds": 40.0},
                    ]
                },
                "normalized_results": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_harvey_lab_report(
        [
            _zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64, seconds=120.0),
            _zaxy_result(
                "litigation-dispute-resolution/review-privilege-log-clawback-review",
                0.62,
                seconds=80.0,
            ),
        ],
        result_provenance=build_harvey_result_provenance([tmp_path], source="harvey-lab-import"),
    )
    markdown = report_to_markdown(report)

    zaxy = report.external_comparison_scorecard["Zaxy"]
    assert zaxy.evidence_scope == "same-harness external Zaxy normalized results"
    assert zaxy.runs == 2
    assert zaxy.mean_score == 0.63
    assert zaxy.delta_vs_raw_rg == 0.231
    assert zaxy.delta_vs_best_external == 0.174
    assert zaxy.mean_total_seconds == 100.0
    assert zaxy.rank_by_mean_score == 1
    assert report.external_comparison_scorecard["mem0"].rank_by_mean_score == 2
    assert (
        "| Zaxy | same-harness external Zaxy normalized results | 2 | "
        "0.630 | +0.231 | +0.174 | 100.000 | 1 |"
    ) in markdown


def test_harvey_report_excludes_zaxy_from_external_baseline_artifacts(tmp_path: Path) -> None:
    """Harvey collector aggregates should not make Zaxy its own external competitor."""
    reports_dir = tmp_path / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison-zaxy.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "zaxy", "runs": 2, "avg_final_score": 0.900},
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_harvey_lab_report(
        [
            _zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64),
            _zaxy_result("litigation-dispute-resolution/review-privilege-log-clawback-review", 0.62),
        ],
        result_provenance=build_harvey_result_provenance([tmp_path], source="harvey-lab-import"),
    )

    assert "zaxy" not in report.external_baseline_scorecard
    assert "zaxy" not in report.external_comparison_scorecard
    assert report.external_comparison_scorecard["Zaxy"].delta_vs_best_external == 0.174
    assert report.external_comparison_scorecard["Zaxy"].rank_by_mean_score == 1


def test_harvey_external_baseline_scorecard_keeps_highest_run_aggregate(tmp_path: Path) -> None:
    """When multiple Harvey comparison reports overlap, broader aggregates should win."""
    reports_dir = tmp_path / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "comparison-a.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "comparison-z.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": 1, "avg_final_score": 0.300},
                        {"framework": "raw-rg", "runs": 1, "avg_final_score": 0.250},
                    ]
                },
                "normalized_results": [{}],
            }
        ),
        encoding="utf-8",
    )

    report = build_harvey_lab_report(
        [],
        result_provenance=build_harvey_result_provenance([tmp_path], source="harvey-lab-import"),
    )

    mem0 = report.external_baseline_scorecard["mem0"]
    assert mem0.runs == len(ARTICLE_TASKS)
    assert mem0.mean_score == 0.456
    assert mem0.delta_vs_raw_rg == 0.057
    assert mem0.source_report_path.endswith("comparison-a.json")


def test_harvey_report_writes_json_and_markdown(tmp_path: Path) -> None:
    """The report writer should produce reviewable comparative statistics."""
    report = build_harvey_lab_report(
        [_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)]
    )

    written = write_harvey_lab_report(report, tmp_path)
    payload = json.loads(written.json_path.read_text(encoding="utf-8"))
    markdown = written.markdown_path.read_text(encoding="utf-8")

    assert payload["schema_version"] == "zaxy.harvey-lab-benchmark.v1"
    assert payload["external_suite"]["source_url"] == HARVEY_LAB_REPO_URL
    assert payload["summary"]["zaxy_task_count"] == 1
    assert "## Framework Fit" in markdown
    assert "| Zaxy |" in markdown
    assert "LightRAG" in markdown
    assert "raw-rg is a retrieval/search baseline" in markdown


def test_harvey_external_run_manifest_lists_all_article_tasks_and_commands() -> None:
    """The external manifest should define the full Harvey task execution plan."""
    manifest = build_harvey_external_run_manifest(
        generator="openai-compatible/gpt-5.5",
        judge="gpt-5.4-mini",
        reasoning_effort="low",
    )

    assert manifest["schema_version"] == "zaxy.harvey-lab-external-run.v1"
    assert manifest["task_count"] == 10
    assert manifest["doctor_command"] == "zaxy harvey-lab-doctor path/to/harvey-zaxy-worktree"
    assert manifest["status_command"] == "zaxy harvey-lab-status path/to/harvey-zaxy-worktree"
    assert len(manifest["tasks"]) == len(ARTICLE_TASKS)
    assert manifest["tasks"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert "--run-id zaxy-" in manifest["tasks"][0]["run_command"]
    assert "HARVEY_MEMORY_MANIFEST=" in manifest["tasks"][0]["run_command"]
    assert "evaluation.run_eval" in manifest["tasks"][0]["judge_command"]
    assert "zaxy harvey-lab-normalize-run" in manifest["tasks"][0]["normalize_command"]
    assert "scripts/memory_ablation/validate_result.py" in manifest["tasks"][0]["validate_command"]
    assert "scripts/memory_ablation/export_result.py" not in json.dumps(manifest)
    assert manifest["collection_command"].startswith("uv run python scripts/memory_ablation/collect_results.py")
    assert "--output .ingestion/reports/comparison-zaxy.json" in manifest["collection_command"]
    assert "zaxy-comparison.json" not in manifest["collection_command"]
    assert manifest["comparison_command"].startswith("zaxy harvey-lab-import")
    assert manifest["report_json_path"] == "reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json"
    assert manifest["publish_output_path"] == "reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md"
    assert manifest["validation_command"].endswith("harvey-lab-benchmark.json --require-complete")
    assert manifest["gate_command"].endswith("harvey-lab-benchmark.json")
    assert manifest["publish_command"].endswith(
        "harvey-lab-benchmark.json --output reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md"
    )
    assert manifest["tasks"][0]["expected_normalized_result"].startswith(".ingestion/runs/zaxy-")


def test_harvey_external_run_manifest_collection_output_is_import_discoverable(tmp_path: Path) -> None:
    """The generated collector output should match the importer's comparison*.json discovery rule."""
    manifest = build_harvey_external_run_manifest(
        generator="openai-compatible/gpt-5.5",
        judge="gpt-5.4-mini",
    )
    output_argument = str(manifest["collection_command"]).split("--output ", maxsplit=1)[1]
    output_path = tmp_path / output_argument
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "harvey.memory_ablation.comparison.v1",
                "aggregate": {"frameworks": []},
                "normalized_results": [],
            }
        ),
        encoding="utf-8",
    )

    provenance = build_harvey_result_provenance([tmp_path], source="harvey-lab-import")

    assert provenance["external_baseline_report_paths"] == [str(output_path.resolve())]


def test_harvey_external_run_script_runs_the_full_external_pipeline() -> None:
    """The generated script should execute the pinned Harvey suite in an external worktree."""
    manifest = build_harvey_external_run_manifest(
        generator="openai-compatible/gpt-5.5",
        judge="gpt-5.4-mini",
        reasoning_effort="low",
    )

    script = build_harvey_external_run_script(manifest)

    assert script.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in script
    assert 'HARVEY_WORKTREE="${1:-.}"' in script
    assert 'OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"' in script
    assert 'TASK_FILTER="${3:-${HARVEY_TASK_FILTER:-}}"' in script
    assert 'GENERATOR_MODEL="${HARVEY_GENERATOR_MODEL:-openai-compatible/gpt-5.5}"' in script
    assert 'JUDGE_MODEL="${HARVEY_JUDGE_MODEL:-gpt-5.4-mini}"' in script
    assert 'JUDGE_PARALLEL="${HARVEY_JUDGE_PARALLEL:-1}"' in script
    assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in script
    assert 'SOURCE_MANIFEST_JSON="$SCRIPT_DIR/harvey-lab-external-run.json"' in script
    assert 'RUN_MANIFEST_JSON="$OUTPUT_DIR/harvey-lab-external-run.json"' in script
    assert 'READY_JSON="$OUTPUT_DIR/harvey-lab-ready.json"' in script
    assert 'STATUS_JSON="$OUTPUT_DIR/harvey-lab-status.json"' in script
    assert 'ZAXY_WORKTREE="$(cd "$SCRIPT_DIR/../../.." && pwd)"' in script
    assert 'ZAXY_PYTHONPATH="$ZAXY_WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert 'if [[ "$SOURCE_MANIFEST_JSON" != "$RUN_MANIFEST_JSON" ]]; then' in script
    assert '  cp "$SOURCE_MANIFEST_JSON" "$RUN_MANIFEST_JSON"' in script
    assert "Unresolved Harvey model placeholders" in script
    assert "zaxy harvey-lab-doctor \"$HARVEY_WORKTREE\"" in script
    preflight_command = 'zaxy harvey-lab-preflight "$HARVEY_WORKTREE" --task-filter "$TASK_FILTER"'
    assert preflight_command in script
    ready_command = 'zaxy harvey-lab-ready "$HARVEY_WORKTREE" --generator "$GENERATOR_MODEL" --judge "$JUDGE_MODEL" --task-filter "$TASK_FILTER" --json | tee "$READY_JSON"'
    assert ready_command in script
    assert script.index('OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"') < script.index('READY_JSON="$OUTPUT_DIR/harvey-lab-ready.json"')
    assert script.index('if [[ "$SOURCE_MANIFEST_JSON" != "$RUN_MANIFEST_JSON" ]]; then') < script.index(ready_command)
    assert script.index(preflight_command) < script.index(ready_command)
    assert script.index(ready_command) < script.index("uv run python -m harness.run")
    assert script.index('OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"') < script.index('cd "$HARVEY_WORKTREE"')
    assert "zaxy harvey-lab-adapter-kit" in script
    assert 'HARVEY_ADAPTER_PATH="$HARVEY_WORKTREE/scripts/memory_ablation/raw_rg_memory.py"' in script
    assert 'HARVEY_ADAPTER_BACKUP="$ADAPTER_DIR/raw_rg_memory.py.original"' in script
    assert 'restore_harvey_adapter()' in script
    assert 'cp "$HARVEY_ADAPTER_PATH" "$HARVEY_ADAPTER_BACKUP"' in script
    assert 'rm -f "$HARVEY_ADAPTER_PATH"' in script
    assert 'trap restore_harvey_adapter EXIT' in script
    assert "cp \"$ADAPTER_DIR/raw_rg_memory.py\" \"$HARVEY_ADAPTER_PATH\"" in script
    assert script.count("prepare_normalized_corpus") == len(ARTICLE_TASKS)
    assert 'if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then' in script
    assert 'echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"' in script
    assert "continue" not in script
    assert "else\nINDEX_DIR=" in script
    assert script.count("zaxy harvey-lab-index") == len(ARTICLE_TASKS)
    assert script.count('PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL"') == len(ARTICLE_TASKS)
    assert script.count(
        'uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL"'
    ) == len(ARTICLE_TASKS)
    assert script.count("zaxy harvey-lab-normalize-run") == len(ARTICLE_TASKS)
    assert script.count("uv run python scripts/memory_ablation/validate_result.py") == len(ARTICLE_TASKS)
    assert 'zaxy harvey-lab-status "$HARVEY_WORKTREE" --json | tee "$STATUS_JSON"' in script
    assert 'if [[ -n "$TASK_FILTER" ]]; then' in script
    assert 'zaxy harvey-lab-status "$HARVEY_WORKTREE" --json | tee "$STATUS_JSON" || true' in script
    assert 'HARVEY_COMPARISON_JSON=".ingestion/reports/comparison-zaxy.json"' in script
    collect_command = 'uv run python scripts/memory_ablation/collect_results.py --worktree "$HARVEY_WORKTREE" --dedupe-latest'
    assert collect_command in script
    assert script.index(collect_command) < script.index('zaxy harvey-lab-import "$HARVEY_WORKTREE"')
    assert (
        'uv run python scripts/memory_ablation/collect_results.py --worktree "$HARVEY_WORKTREE" '
        '--dedupe-latest --output "$HARVEY_COMPARISON_JSON" || true'
    ) in script
    assert "zaxy harvey-lab-import \"$HARVEY_WORKTREE\"" in script
    assert 'REPORT_JSON="$OUTPUT_DIR/harvey-lab-benchmark.json"' in script
    assert 'PUBLISH_MD="$OUTPUT_DIR/publishable-statistics.md"' in script
    assert 'zaxy harvey-lab-validate "$REPORT_JSON" --require-complete' in script
    assert 'zaxy harvey-lab-gate "$REPORT_JSON"' in script
    assert 'zaxy harvey-lab-publish "$REPORT_JSON" --output "$PUBLISH_MD"' in script
    assert 'zaxy harvey-lab-validate "$REPORT_JSON" || true' in script
    assert 'echo "Filtered Harvey run imported; full publish gate is intentionally skipped until all tasks are complete."' in script


def test_harvey_external_suite_doctor_validates_required_checkout_shape(tmp_path: Path) -> None:
    """A Harvey worktree should be checked before Zaxy treats it as the external suite."""
    worktree = tmp_path / "harvey"
    for task in ARTICLE_TASKS:
        task_dir = worktree / "tasks" / task.task_id
        docs_dir = task_dir / "documents"
        docs_dir.mkdir(parents=True)
        criteria = [{"match_criteria": "PASS"} for _ in range(task.criteria_count)]
        (task_dir / "task.json").write_text(json.dumps({"criteria": criteria}), encoding="utf-8")
        for idx in range(task.document_count):
            (docs_dir / f"doc-{idx}.txt").write_text("fixture\n", encoding="utf-8")
    scripts = worktree / "scripts" / "memory_ablation"
    scripts.mkdir(parents=True)
    for name in (
        "normalize_corpus.py",
        "validate_result.py",
        "collect_results.py",
        "render_report.py",
    ):
        (scripts / name).write_text("# fixture\n", encoding="utf-8")
    (worktree / "harness").mkdir()
    (worktree / "harness" / "run.py").write_text("# fixture\n", encoding="utf-8")
    (worktree / "evaluation").mkdir()
    (worktree / "evaluation" / "run_eval.py").write_text("# fixture\n", encoding="utf-8")

    status = check_harvey_external_suite(worktree)

    assert status["status"] == "valid"
    assert status["task_count"] == 10
    assert status["missing_task_ids"] == []
    assert status["missing_required_files"] == []
    assert status["task_audits"][ARTICLE_TASKS[0].task_id]["document_count"] == ARTICLE_TASKS[0].document_count
    assert status["task_audits"][ARTICLE_TASKS[0].task_id]["criteria_count"] == ARTICLE_TASKS[0].criteria_count
    assert status["task_mismatches"] == []


def test_harvey_external_suite_doctor_reports_missing_tasks_and_scripts(tmp_path: Path) -> None:
    """The doctor should fail closed when the external suite checkout is incomplete."""
    worktree = tmp_path / "harvey"
    for task in ARTICLE_TASKS[1:]:
        task_dir = worktree / "tasks" / task.task_id
        task_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text("{}", encoding="utf-8")
        (task_dir / "documents").mkdir()
    (worktree / "scripts" / "memory_ablation").mkdir(parents=True)
    (worktree / "scripts" / "memory_ablation" / "normalize_corpus.py").write_text("# fixture\n", encoding="utf-8")

    status = check_harvey_external_suite(worktree)

    assert status["status"] == "invalid"
    assert status["missing_task_ids"] == [ARTICLE_TASKS[0].task_id]
    assert "scripts/memory_ablation/validate_result.py" in status["missing_required_files"]
    assert "harness/run.py" in status["missing_required_files"]


def test_harvey_external_suite_doctor_reports_task_count_drift(tmp_path: Path) -> None:
    """The doctor should detect task content drift against the pinned article matrix."""
    worktree = tmp_path / "harvey"
    for task in ARTICLE_TASKS:
        task_dir = worktree / "tasks" / task.task_id
        docs_dir = task_dir / "documents"
        docs_dir.mkdir(parents=True)
        criteria = [{"match_criteria": "PASS"} for _ in range(task.criteria_count)]
        (task_dir / "task.json").write_text(json.dumps({"criteria": criteria}), encoding="utf-8")
        doc_count = task.document_count - 1 if task == ARTICLE_TASKS[0] else task.document_count
        for idx in range(doc_count):
            (docs_dir / f"doc-{idx}.txt").write_text("fixture\n", encoding="utf-8")
    first_task = ARTICLE_TASKS[0]
    first_task_json = worktree / "tasks" / first_task.task_id / "task.json"
    first_task_json.write_text(json.dumps({"criteria": [{"match_criteria": "PASS"}]}), encoding="utf-8")
    scripts = worktree / "scripts" / "memory_ablation"
    scripts.mkdir(parents=True)
    for name in (
        "normalize_corpus.py",
        "validate_result.py",
        "collect_results.py",
        "render_report.py",
    ):
        (scripts / name).write_text("# fixture\n", encoding="utf-8")
    (worktree / "harness").mkdir()
    (worktree / "harness" / "run.py").write_text("# fixture\n", encoding="utf-8")
    (worktree / "evaluation").mkdir()
    (worktree / "evaluation" / "run_eval.py").write_text("# fixture\n", encoding="utf-8")

    status = check_harvey_external_suite(worktree)

    assert status["status"] == "invalid"
    assert status["task_mismatches"] == [
        {
            "task_id": first_task.task_id,
            "reason": "task_shape_mismatch",
            "expected_document_count": first_task.document_count,
            "actual_document_count": first_task.document_count - 1,
            "expected_criteria_count": first_task.criteria_count,
            "actual_criteria_count": 1,
        }
    ]


def test_harvey_lab_doctor_cli_reports_external_suite_status(tmp_path: Path) -> None:
    """The CLI should expose the external Harvey checkout validation."""
    worktree = tmp_path / "harvey"
    worktree.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-doctor",
            str(worktree),
        ],
    )

    assert result.exit_code == 1
    assert '"status": "invalid"' in result.output
    assert "missing_task_ids" in result.output


def test_harvey_external_index_preflight_builds_all_zaxy_indexes(tmp_path: Path) -> None:
    """Preflight should use Harvey normalization and build reviewable Zaxy indexes only."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")

    report = build_harvey_external_index_preflight(worktree)

    assert report["schema_version"] == "zaxy.harvey-lab-index-preflight.v1"
    assert report["status"] == "ready_for_external_runs"
    assert report["task_count"] == len(ARTICLE_TASKS)
    first = report["tasks"][ARTICLE_TASKS[0].task_id]
    assert first["status"] == "indexed"
    assert first["document_count"] == ARTICLE_TASKS[0].document_count
    assert first["manifest_path"].endswith("/manifest.json")
    assert first["event_count"] >= ARTICLE_TASKS[0].document_count
    assert first["smoke_search_hit_count"] > 0
    assert Path(str(first["manifest_path"])).exists()
    assert not (worktree / ".ingestion" / "runs").exists()


def test_harvey_external_index_preflight_smoke_uses_corpus_terms(tmp_path: Path) -> None:
    """Smoke validation should prove retrieval over indexed content, not a fixed phrase."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    for path in (worktree / "tasks").rglob("documents/*.txt"):
        path.write_text("alphaomega contractual schedule\n", encoding="utf-8")

    report = build_harvey_external_index_preflight(worktree)

    assert report["status"] == "ready_for_external_runs"
    assert report["failures"] == []
    first = report["tasks"][ARTICLE_TASKS[0].task_id]
    assert first["smoke_query"] == "alphaomega contractual schedule"
    assert first["smoke_search_hit_count"] > 0


def test_harvey_external_index_preflight_can_scope_to_task_filter(tmp_path: Path) -> None:
    """Filtered preflight should only normalize and index the requested task."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    selected = ARTICLE_TASKS[0]
    skipped = ARTICLE_TASKS[1]
    selected_slug = selected.task_id.replace("/", "__")
    skipped_slug = skipped.task_id.replace("/", "__")

    report = build_harvey_external_index_preflight(
        worktree,
        task_filter=selected_slug,
    )

    assert report["status"] == "ready_for_external_runs"
    assert report["task_filter"] == selected_slug
    assert report["resolved_task_id"] == selected.task_id
    assert report["task_count"] == 1
    assert report["expected_task_count"] == 1
    assert set(report["tasks"]) == {selected.task_id}
    assert (worktree / ".ingestion" / "indexes" / selected_slug / "zaxy" / "manifest.json").exists()
    assert not (worktree / ".ingestion" / "indexes" / skipped_slug).exists()


def test_harvey_external_index_preflight_supports_local_normalizer_fallback(tmp_path: Path) -> None:
    """Preflight should also support direct normalizer loading for deterministic local checks."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    selected = ARTICLE_TASKS[0]
    selected_slug = selected.task_id.replace("/", "__")

    report = build_harvey_external_index_preflight(
        worktree,
        use_external_environment=False,
        task_filter=selected_slug,
    )

    assert report["status"] == "ready_for_external_runs"
    assert report["resolved_task_id"] == selected.task_id
    assert report["tasks"][selected.task_id]["smoke_search_hit_count"] > 0


def test_harvey_external_index_preflight_rejects_local_normalizer_without_entrypoint(tmp_path: Path) -> None:
    """Local fallback should fail if Harvey's normalizer contract changes."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    selected_slug = ARTICLE_TASKS[0].task_id.replace("/", "__")
    normalizer = worktree / "scripts" / "memory_ablation" / "normalize_corpus.py"
    normalizer.write_text("NORMALIZER_CONTRACT_CHANGED = True\n", encoding="utf-8")

    try:
        build_harvey_external_index_preflight(
            worktree,
            use_external_environment=False,
            task_filter=selected_slug,
        )
    except ValueError as exc:
        assert "does not define prepare_normalized_corpus" in str(exc)
    else:
        raise AssertionError("expected local Harvey normalizer contract failure")


def test_harvey_external_index_preflight_reports_external_uv_failure(tmp_path: Path) -> None:
    """External normalization failures should surface the Harvey checkout error text."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    selected_slug = ARTICLE_TASKS[0].task_id.replace("/", "__")
    uv = worktree / ".fixture-bin" / "uv"
    uv.write_text(
        """#!/usr/bin/env sh
echo "normalizer exploded" >&2
exit 7
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    try:
        build_harvey_external_index_preflight(worktree, task_filter=selected_slug)
    except ValueError as exc:
        assert "Harvey normalization failed" in str(exc)
        assert "normalizer exploded" in str(exc)
    else:
        raise AssertionError("expected external Harvey normalizer failure")


def test_harvey_external_index_preflight_rejects_non_json_external_normalization(tmp_path: Path) -> None:
    """External normalization must return JSON that Zaxy can audit."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    selected_slug = ARTICLE_TASKS[0].task_id.replace("/", "__")
    uv = worktree / ".fixture-bin" / "uv"
    uv.write_text(
        """#!/usr/bin/env sh
echo "not-json"
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    try:
        build_harvey_external_index_preflight(worktree, task_filter=selected_slug)
    except ValueError as exc:
        assert "returned non-JSON output" in str(exc)
    else:
        raise AssertionError("expected non-JSON normalization rejection")


def test_harvey_external_index_preflight_rejects_non_object_external_normalization(tmp_path: Path) -> None:
    """External normalization JSON must be an object with path fields."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    selected_slug = ARTICLE_TASKS[0].task_id.replace("/", "__")
    uv = worktree / ".fixture-bin" / "uv"
    uv.write_text(
        """#!/usr/bin/env sh
echo "[]"
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    try:
        build_harvey_external_index_preflight(worktree, task_filter=selected_slug)
    except ValueError as exc:
        assert "must return a JSON object" in str(exc)
    else:
        raise AssertionError("expected non-object normalization rejection")


def test_harvey_external_index_preflight_rejects_invalid_external_suite(tmp_path: Path) -> None:
    """Preflight should not build indexes for a checkout that does not match the pinned suite."""
    worktree = tmp_path / "harvey"
    worktree.mkdir()

    try:
        build_harvey_external_index_preflight(worktree)
    except ValueError as exc:
        assert "Harvey external suite checkout is invalid" in str(exc)
    else:
        raise AssertionError("expected invalid Harvey checkout to be rejected")


def test_harvey_external_index_preflight_rejects_unknown_task_filter(tmp_path: Path) -> None:
    """Preflight should fail closed rather than accidentally indexing all tasks."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")

    try:
        build_harvey_external_index_preflight(worktree, task_filter="not-a-harvey-task")
    except ValueError as exc:
        assert "Unknown Harvey task filter: not-a-harvey-task" in str(exc)
    else:
        raise AssertionError("expected unknown Harvey task filter to be rejected")


def test_harvey_lab_preflight_cli_indexes_external_suite(tmp_path: Path) -> None:
    """The CLI should build all Zaxy index artifacts before external model runs."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-preflight",
            str(worktree),
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"status": "ready_for_external_runs"' in result.output
    assert '"task_count": 10' in result.output
    status = build_harvey_external_run_status(worktree)
    first = status["tasks"][ARTICLE_TASKS[0].task_id]
    assert first["index_ready"] is True
    assert first["run_artifacts_ready"] is False


def test_harvey_lab_preflight_cli_accepts_task_filter(tmp_path: Path) -> None:
    """The preflight CLI should support single-task external rerun setup."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    task = ARTICLE_TASKS[0]
    slug = task.task_id.replace("/", "__")

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-preflight",
            str(worktree),
            "--task-filter",
            slug,
        ],
    )

    assert result.exit_code == 0, result.output
    assert f'"task_filter": "{slug}"' in result.output
    assert f'"resolved_task_id": "{task.task_id}"' in result.output
    assert '"task_count": 1' in result.output
    status = build_harvey_external_run_status(worktree)
    assert status["tasks"][task.task_id]["index_ready"] is True
    assert status["tasks"][ARTICLE_TASKS[1].task_id]["index_ready"] is False


def test_harvey_external_run_readiness_blocks_unresolved_model_placeholders(tmp_path: Path) -> None:
    """Readiness should fail before external runs if plan placeholders remain unresolved."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    build_harvey_external_index_preflight(worktree)

    readiness = build_harvey_external_run_readiness(
        worktree,
        generator="HARVEY_GENERATOR_MODEL",
        judge="HARVEY_JUDGE_MODEL",
        env={},
    )

    assert readiness["schema_version"] == "zaxy.harvey-lab-run-readiness.v1"
    assert readiness["status"] == "not_ready"
    assert readiness["suite_valid"] is True
    assert readiness["index_ready_count"] == 10
    assert readiness["ready_task_count"] == 0
    assert readiness["unresolved_models"] == ["generator", "judge"]
    assert "unresolved_model_placeholders" in readiness["blocking_reasons"]


def test_harvey_external_run_readiness_reports_missing_provider_credentials(tmp_path: Path) -> None:
    """Readiness should identify provider credentials before expensive external runs."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    build_harvey_external_index_preflight(worktree)

    readiness = build_harvey_external_run_readiness(
        worktree,
        generator="openai/gpt-5.5",
        judge="gpt-5.4-mini",
        env={},
    )

    assert readiness["status"] == "not_ready"
    assert readiness["model_requirements"]["generator"]["provider"] == "openai"
    assert readiness["model_requirements"]["generator"]["credential_status"] == "missing"
    assert readiness["model_requirements"]["judge"]["credential_status"] == "missing"
    assert readiness["missing_credentials"] == ["OPENAI_API_KEY"]
    assert "missing_model_credentials" in readiness["blocking_reasons"]


def test_harvey_external_run_readiness_loads_worktree_dotenv_credentials(tmp_path: Path) -> None:
    """Readiness should mirror Harvey's runtime behavior and inspect worktree .env."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    build_harvey_external_index_preflight(worktree)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    podman = bin_dir / "podman"
    podman.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    podman.chmod(0o755)
    pandoc = bin_dir / "pandoc"
    pandoc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pandoc.chmod(0o755)
    pandoc = bin_dir / "pandoc"
    pandoc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pandoc.chmod(0o755)
    (worktree / ".env").write_text(
        "OPENAI_API_KEY=from-dotenv\n"
        "OPENAI_COMPATIBLE_BASE_URL=http://127.0.0.1:8318/v1\n",
        encoding="utf-8",
    )

    readiness = build_harvey_external_run_readiness(
        worktree,
        generator="openai-compatible/gpt-5.5",
        judge="gpt-5.4-mini",
        env={"PATH": str(bin_dir)},
    )

    assert readiness["status"] == "ready_for_external_runs"
    assert readiness["missing_credentials"] == []
    assert "missing_model_credentials" not in readiness["blocking_reasons"]
    assert readiness["model_requirements"]["judge"]["credential_status"] == "present"
    assert readiness["sandbox_runtime"]["podman_status"] == "present"
    assert readiness["host_document_reader"]["pandoc_status"] == "present"


def test_harvey_external_run_readiness_blocks_missing_podman_runtime(tmp_path: Path) -> None:
    """Readiness should fail before launch if Harvey's required sandbox runtime is missing."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    build_harvey_external_index_preflight(worktree)

    readiness = build_harvey_external_run_readiness(
        worktree,
        generator="openai-compatible/gpt-5.5",
        judge="gpt-5.4-mini",
        env={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:8318/v1",
            "PATH": str(tmp_path / "empty-bin"),
        },
    )

    assert readiness["status"] == "not_ready"
    assert "missing_sandbox_runtime" in readiness["blocking_reasons"]
    assert readiness["sandbox_runtime"]["podman_status"] == "missing"
    assert readiness["sandbox_runtime"]["required_by"] == "Harvey LAB harness sandbox"


def test_harvey_external_run_readiness_blocks_missing_host_docx_reader(tmp_path: Path) -> None:
    """Readiness should fail before launch if Harvey's host judge cannot read docx outputs."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    build_harvey_external_index_preflight(worktree)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    podman = bin_dir / "podman"
    podman.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    podman.chmod(0o755)

    readiness = build_harvey_external_run_readiness(
        worktree,
        generator="openai-compatible/gpt-5.5",
        judge="gpt-5.4-mini",
        env={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:8318/v1",
            "PATH": str(bin_dir),
        },
    )

    assert readiness["status"] == "not_ready"
    assert "missing_host_document_reader" in readiness["blocking_reasons"]
    assert readiness["host_document_reader"]["pandoc_status"] == "missing"
    assert readiness["host_document_reader"]["required_by"] == "Harvey LAB evaluator docx scoring"


def test_harvey_external_run_readiness_can_scope_indexes_to_task_filter(tmp_path: Path) -> None:
    """Filtered reruns should not require all ten Zaxy indexes before launch."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    podman = bin_dir / "podman"
    podman.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    podman.chmod(0o755)
    pandoc = bin_dir / "pandoc"
    pandoc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pandoc.chmod(0o755)
    task = ARTICLE_TASKS[0]
    slug = task.task_id.replace("/", "__")
    index_dir = worktree / ".ingestion" / "indexes" / slug / "zaxy"
    index_dir.mkdir(parents=True)
    for name in ("manifest.json", "artifact-summary.json", "smoke-result.json"):
        (index_dir / name).write_text("{}", encoding="utf-8")
    env = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:8318/v1",
        "PATH": str(bin_dir),
    }

    full_readiness = build_harvey_external_run_readiness(
        worktree,
        generator="openai-compatible/gpt-5.5",
        judge="gpt-5.4-mini",
        env=env,
    )
    filtered_readiness = build_harvey_external_run_readiness(
        worktree,
        generator="openai-compatible/gpt-5.5",
        judge="gpt-5.4-mini",
        env=env,
        task_filter=slug,
    )

    assert full_readiness["status"] == "not_ready"
    assert "missing_zaxy_indexes" in full_readiness["blocking_reasons"]
    assert filtered_readiness["status"] == "ready_for_external_runs"
    assert filtered_readiness["task_filter"] == slug
    assert filtered_readiness["expected_task_count"] == 1
    assert filtered_readiness["index_ready_count"] == 1
    assert filtered_readiness["evidence_audit"]["expected_task_count"] == 1
    assert filtered_readiness["evidence_audit"]["index_ready_count"] == 1
    assert filtered_readiness["evidence_audit"]["run_artifacts_ready_count"] == 0
    assert filtered_readiness["evidence_audit"]["normalized_result_ready_count"] == 0
    assert filtered_readiness["evidence_audit"]["import_ready_count"] == 0
    assert "missing_zaxy_indexes" not in filtered_readiness["blocking_reasons"]


def test_harvey_lab_ready_cli_reports_current_run_readiness(tmp_path: Path) -> None:
    """The CLI should expose run readiness without launching model calls."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    build_harvey_external_index_preflight(worktree)

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-ready",
            str(worktree),
            "--generator",
            "openai/gpt-5.5",
            "--judge",
            "gpt-5.4-mini",
        ],
    )

    assert result.exit_code == 1
    assert '"status": "not_ready"' in result.output
    assert '"missing_credentials": [' in result.output
    assert '"index_ready_count": 10' in result.output


def test_harvey_lab_ready_cli_accepts_json_flag_for_automation(tmp_path: Path) -> None:
    """External runners should be able to request machine-readable readiness explicitly."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-ready",
            str(worktree),
            "--generator",
            "openai-compatible/gpt-5.5",
            "--judge",
            "gpt-5.4-mini",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["schema_version"] == "zaxy.harvey-lab-run-readiness.v1"
    assert payload["status"] == "not_ready"


def test_harvey_lab_ready_cli_accepts_task_filter(tmp_path: Path) -> None:
    """The readiness CLI should support single-task external reruns."""
    worktree = _write_valid_harvey_worktree(tmp_path / "harvey")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    podman = bin_dir / "podman"
    podman.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    podman.chmod(0o755)
    pandoc = bin_dir / "pandoc"
    pandoc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pandoc.chmod(0o755)
    task = ARTICLE_TASKS[0]
    slug = task.task_id.replace("/", "__")
    index_dir = worktree / ".ingestion" / "indexes" / slug / "zaxy"
    index_dir.mkdir(parents=True)
    for name in ("manifest.json", "artifact-summary.json", "smoke-result.json"):
        (index_dir / name).write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-ready",
            str(worktree),
            "--generator",
            "openai-compatible/gpt-5.5",
            "--judge",
            "gpt-5.4-mini",
            "--task-filter",
            slug,
        ],
        env={
            "OPENAI_API_KEY": "test-key",
            "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:8318/v1",
            "PATH": str(bin_dir),
        },
    )

    assert result.exit_code == 0, result.output
    assert f'"task_filter": "{slug}"' in result.output
    assert '"expected_task_count": 1' in result.output
    assert '"index_ready_count": 1' in result.output


def test_harvey_external_run_status_reports_missing_pipeline_artifacts(tmp_path: Path) -> None:
    """The status scanner should show which external task artifacts are missing."""
    worktree = tmp_path / "harvey"
    worktree.mkdir()

    status = build_harvey_external_run_status(worktree)

    assert status["schema_version"] == "zaxy.harvey-lab-run-status.v1"
    assert status["status"] == "not_ready"
    assert status["ready_task_count"] == 0
    assert status["expected_task_count"] == 10
    first = status["tasks"][ARTICLE_TASKS[0].task_id]
    assert first["run_id"].startswith("zaxy-")
    assert first["index_ready"] is False
    assert first["run_artifacts_ready"] is False
    assert first["normalized_result_ready"] is False
    assert first["import_ready"] is False
    assert "index_manifest" in first["missing_artifacts"]


def test_harvey_external_run_status_records_harvey_git_commit(tmp_path: Path) -> None:
    """The status artifact should identify the external Harvey checkout commit."""
    worktree = tmp_path / "harvey"
    worktree.mkdir()
    subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "zaxy@example.com"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Zaxy Test"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    )
    (worktree / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=worktree, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    status = build_harvey_external_run_status(worktree)

    assert status["harvey_git_commit"] == commit


def test_harvey_external_run_status_reports_import_ready_task(tmp_path: Path) -> None:
    """A task with complete external artifacts should be marked import-ready."""
    worktree = tmp_path / "harvey"
    task = ARTICLE_TASKS[0]
    slug = task.task_id.replace("/", "__")
    run_id = f"zaxy-{slug}"
    index_dir = worktree / ".ingestion" / "indexes" / slug / "zaxy"
    index_dir.mkdir(parents=True)
    for name in ("manifest.json", "artifact-summary.json", "smoke-result.json"):
        (index_dir / name).write_text("{}", encoding="utf-8")
    run_dir = worktree / "results" / run_id
    (run_dir / "output").mkdir(parents=True)
    for relative in (
        "config.json",
        "scores.json",
        "output/response.md",
    ):
        (run_dir / relative).write_text("fixture\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"memory_search_calls": 2, "memory_read_calls": 1}),
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
        encoding="utf-8",
    )
    normalized_dir = worktree / ".ingestion" / "runs" / run_id
    normalized_dir.mkdir(parents=True)
    (normalized_dir / "normalized-result.json").write_text(
        json.dumps(_zaxy_result(task.task_id, 0.7, run_id=run_id)),
        encoding="utf-8",
    )

    status = build_harvey_external_run_status(worktree)

    first = status["tasks"][task.task_id]
    assert first["index_ready"] is True
    assert first["run_artifacts_ready"] is True
    assert first["normalized_result_ready"] is True
    assert first["import_ready"] is True
    assert first["missing_artifacts"] == []
    assert status["ready_task_count"] == 1
    assert status["status"] == "partial"
    assert status["evidence_audit"] == {
        "index_ready_count": 1,
        "run_artifacts_ready_count": 1,
        "normalized_result_ready_count": 1,
        "memory_evidence_ready_count": 1,
        "import_ready_count": 1,
        "expected_task_count": 10,
    }


def test_harvey_external_run_status_accepts_named_task_deliverable(tmp_path: Path) -> None:
    """Status should not require response.md when a Harvey task wrote its named deliverable."""
    worktree = tmp_path / "harvey"
    task = next(item for item in ARTICLE_TASKS if item.task_id == "corporate-ma/draft-acquisition-due-diligence")
    slug = task.task_id.replace("/", "__")
    run_id = f"zaxy-{slug}"
    task_dir = worktree / "tasks" / task.task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "deliverables": {
                    "novabright-diligence-memorandum.docx": "novabright-diligence-memorandum.docx"
                }
            }
        ),
        encoding="utf-8",
    )
    index_dir = worktree / ".ingestion" / "indexes" / slug / "zaxy"
    index_dir.mkdir(parents=True)
    for name in ("manifest.json", "artifact-summary.json", "smoke-result.json"):
        (index_dir / name).write_text("{}", encoding="utf-8")
    run_dir = worktree / "results" / run_id
    (run_dir / "output").mkdir(parents=True)
    for relative in (
        "config.json",
        "scores.json",
        "output/novabright-diligence-memorandum.docx",
    ):
        (run_dir / relative).write_text("fixture\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"memory_search_calls": 2, "memory_read_calls": 1}),
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
        encoding="utf-8",
    )
    normalized_dir = worktree / ".ingestion" / "runs" / run_id
    normalized_dir.mkdir(parents=True)
    result = _zaxy_result(task.task_id, 0.797, run_id=run_id)
    paths = result["paths"]
    assert isinstance(paths, dict)
    paths["answer"] = f"results/{run_id}/output/novabright-diligence-memorandum.docx"
    (normalized_dir / "normalized-result.json").write_text(json.dumps(result), encoding="utf-8")

    status = build_harvey_external_run_status(worktree)

    first = status["tasks"][task.task_id]
    assert first["run_artifacts_ready"] is True
    assert first["normalized_result_ready"] is True
    assert first["import_ready"] is True
    assert "run_answer" not in first["missing_artifacts"]


def test_harvey_external_run_status_rejects_mismatched_normalized_result(tmp_path: Path) -> None:
    """A present normalized-result file should not be import-ready if it names the wrong run."""
    worktree = tmp_path / "harvey"
    task = ARTICLE_TASKS[0]
    slug = task.task_id.replace("/", "__")
    run_id = f"zaxy-{slug}"
    index_dir = worktree / ".ingestion" / "indexes" / slug / "zaxy"
    index_dir.mkdir(parents=True)
    for name in ("manifest.json", "artifact-summary.json", "smoke-result.json"):
        (index_dir / name).write_text("{}", encoding="utf-8")
    run_dir = worktree / "results" / run_id
    (run_dir / "output").mkdir(parents=True)
    for relative in (
        "config.json",
        "metrics.json",
        "scores.json",
        "transcript.jsonl",
        "output/response.md",
    ):
        (run_dir / relative).write_text("fixture\n", encoding="utf-8")
    normalized_dir = worktree / ".ingestion" / "runs" / run_id
    normalized_dir.mkdir(parents=True)
    (normalized_dir / "normalized-result.json").write_text(
        json.dumps(_zaxy_result(task.task_id, 0.7, run_id="zaxy-wrong-run")),
        encoding="utf-8",
    )

    status = build_harvey_external_run_status(worktree)

    first = status["tasks"][task.task_id]
    assert first["normalized_result_ready"] is False
    assert first["import_ready"] is False
    assert first["normalized_result_error"] == "normalized_result_mismatch"
    assert "normalized_result_contract" in first["missing_artifacts"]
    assert status["ready_task_count"] == 0
    assert status["status"] == "not_ready"


def test_harvey_external_run_status_rejects_run_without_memory_tool_evidence(tmp_path: Path) -> None:
    """Existing run files should not be import-ready without memory metrics and transcript evidence."""
    worktree = tmp_path / "harvey"
    task = ARTICLE_TASKS[0]
    slug = task.task_id.replace("/", "__")
    run_id = f"zaxy-{slug}"
    index_dir = worktree / ".ingestion" / "indexes" / slug / "zaxy"
    index_dir.mkdir(parents=True)
    for name in ("manifest.json", "artifact-summary.json", "smoke-result.json"):
        (index_dir / name).write_text("{}", encoding="utf-8")
    run_dir = worktree / "results" / run_id
    (run_dir / "output").mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"memory_search_calls": 0, "memory_read_calls": 0}),
        encoding="utf-8",
    )
    (run_dir / "scores.json").write_text("{}", encoding="utf-8")
    (run_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "tool", "tool_name": "read", "result_preview": "plain read"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "output" / "response.md").write_text("Answer\n", encoding="utf-8")
    normalized_dir = worktree / ".ingestion" / "runs" / run_id
    normalized_dir.mkdir(parents=True)
    (normalized_dir / "normalized-result.json").write_text(
        json.dumps(_zaxy_result(task.task_id, 0.7, run_id=run_id)),
        encoding="utf-8",
    )

    status = build_harvey_external_run_status(worktree)

    first = status["tasks"][task.task_id]
    assert first["run_artifacts_ready"] is False
    assert first["run_artifact_error"] == "memory_tools_not_used"
    assert first["import_ready"] is False
    assert "run_memory_evidence" in first["missing_artifacts"]
    assert status["ready_task_count"] == 0
    assert status["status"] == "not_ready"


def test_harvey_lab_status_cli_reports_external_run_status(tmp_path: Path) -> None:
    """The CLI should expose per-task external pipeline readiness."""
    worktree = tmp_path / "harvey"
    worktree.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-status",
            str(worktree),
        ],
    )

    assert result.exit_code == 1
    assert '"status": "not_ready"' in result.output
    assert "ready_task_count" in result.output


def test_harvey_lab_status_cli_accepts_json_flag_for_automation(tmp_path: Path) -> None:
    """External runners should be able to request machine-readable status explicitly."""
    worktree = tmp_path / "harvey"
    worktree.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-status",
            str(worktree),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["schema_version"] == "zaxy.harvey-lab-run-status.v1"
    assert payload["status"] == "not_ready"


def test_harvey_completion_gate_blocks_missing_or_partial_results(tmp_path: Path) -> None:
    """Public comparative claims should require all ten external Zaxy rows."""
    partial_report = build_harvey_lab_report(
        [_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)]
    )

    partial_gate = check_harvey_lab_completion(partial_report)

    assert partial_gate["status"] == "blocked"
    assert "corporate-ma/review-data-room-red-flag-review" in partial_gate["completed_task_ids"]
    assert "corporate-ma/draft-acquisition-due-diligence" in partial_gate["missing_task_ids"]

    complete_report = build_harvey_lab_report(
        [
            _zaxy_result(task.task_id, task.best_score + 0.01)
            for task in ARTICLE_TASKS
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(tmp_path / "external-harvey-worktree")],
            "normalized_result_paths": _write_normalized_result_paths(
                tmp_path / "external-harvey-worktree" / ".ingestion" / "runs",
                {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS},
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
        },
    )

    complete_gate = check_harvey_lab_completion(complete_report)

    assert complete_gate["status"] == "blocked"
    assert complete_gate["evidence_failures"][0]["reason"] == "missing_external_scored_system_comparison"

    compared_external = tmp_path / "external-harvey-compared-worktree"
    reports_dir = compared_external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    externally_compared_report = build_harvey_lab_report(
        [
            _zaxy_result(task.task_id, task.best_score + 0.01)
            for task in ARTICLE_TASKS
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(compared_external)],
            "normalized_result_paths": _write_normalized_result_paths(
                compared_external / ".ingestion" / "runs",
                {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS},
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(compared_external),
        },
    )
    externally_compared_gate = check_harvey_lab_completion(externally_compared_report)
    externally_compared_validation = validate_harvey_lab_report(
        externally_compared_report,
        require_complete=True,
    )

    assert externally_compared_gate["status"] == "passed"
    assert externally_compared_validation["status"] == "valid"
    assert externally_compared_gate["missing_task_ids"] == []
    assert externally_compared_gate["zaxy_mean_score"] > externally_compared_gate["article_best_mean_for_zaxy_tasks"]
    assert externally_compared_gate["zaxy_external_rank"] == (
        externally_compared_report.external_comparison_scorecard["Zaxy"].rank_by_mean_score
    )
    assert externally_compared_gate["zaxy_delta_vs_raw_rg"] == (
        externally_compared_report.external_comparison_scorecard["Zaxy"].delta_vs_raw_rg
    )
    assert externally_compared_gate["zaxy_delta_vs_best_external"] == (
        externally_compared_report.external_comparison_scorecard["Zaxy"].delta_vs_best_external
    )
    assert externally_compared_gate["zaxy_framework_fit"] == (
        externally_compared_report.framework_fit["Zaxy"].interpretation
    )
    assert externally_compared_gate["zaxy_run_configuration"] == {
        "generator": "openai-compatible/gpt-5.5",
        "judge": "gpt-5.4-mini",
        "generator_reasoning_effort": "low",
        "judge_reasoning_effort": None,
        "temperature": 0.0,
        "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
    }
    for payload in (externally_compared_gate, externally_compared_validation):
        assert payload["evidence_audit"] == {
            "normalized_result_artifacts": 10,
            "external_comparison_artifacts": 1,
            "judge_score_artifacts_match": True,
            "run_metrics_artifacts_match": True,
            "transcript_memory_tool_evidence": True,
            "external_comparison_recomputed_from_non_zaxy_rows": True,
            "external_run_manifest_artifacts": 1,
            "external_readiness_report_artifacts": 1,
            "external_status_report_artifacts": 1,
            "external_run_audit_artifacts_valid": True,
        }


def test_harvey_completion_gate_blocks_external_aggregate_without_result_evidence(tmp_path: Path) -> None:
    """External system ranks need Harvey comparison artifacts with underlying result rows."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": [],
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "missing_external_comparison_result_evidence"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_comparison_result_evidence"


def test_harvey_completion_gate_reads_nested_external_comparison_scores(tmp_path: Path) -> None:
    """Harvey-native comparison artifacts store scores/timing in nested sections."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": [
                    {
                        "framework": item["framework"],
                        "task_id": item["task_id"],
                        "models": item["models"],
                        "scores": {"final_score": item["final_score"]},
                        "timing": {"total_seconds": 10.0},
                    }
                    for item in _external_comparison_normalized_results()
                ],
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "valid"
    assert gate["status"] == "passed"
    assert validation["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is True


def test_harvey_completion_gate_blocks_external_comparison_outside_external_root(tmp_path: Path) -> None:
    """External comparison reports should be reviewable under the cited Harvey worktree."""
    external = tmp_path / "external-harvey-worktree"
    outside = tmp_path / "outside-reports"
    outside.mkdir()
    comparison_path = outside / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "missing_external_baseline_report_artifact"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_baseline_report_artifact"


def test_harvey_completion_gate_blocks_external_comparison_partial_suite_rows(tmp_path: Path) -> None:
    """External comparison rows should cover every pinned article task before ranking Zaxy."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _partial_external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_comparison_not_full_suite"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_comparison_not_full_suite"
    assert validation["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False
    assert gate["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False


def test_harvey_completion_gate_allows_partial_native_comparison_context(tmp_path: Path) -> None:
    """Partial Harvey-native competitor reruns are context, not full-suite publication evidence."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": 1, "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": 1, "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _partial_external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "valid"
    assert gate["status"] == "passed"
    assert validation["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False
    assert gate["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False
    markdown = render_harvey_publication_markdown(report)
    assert "Native non-Zaxy comparison artifacts are partial context" in markdown
    assert "full-suite article-relative comparison uses the published article scorecard" in markdown


def test_harvey_completion_gate_accepts_output_dir_run_audit_artifacts(tmp_path: Path) -> None:
    """Archived launch/readiness/status files beside the report are valid cited evidence."""
    external = tmp_path / "external-harvey-worktree"
    output_dir = tmp_path / "reports" / "harvey-lab-memory-ablation"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": 1, "avg_final_score": 0.456},
                    ]
                },
                "normalized_results": _partial_external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external), str(output_dir)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                output_dir,
                readiness_worktree=str(external.resolve()),
                status_worktree=str(external.resolve()),
                status_index_dir_overrides={
                    task.task_id: str(
                        external
                        / ".ingestion"
                        / "indexes"
                        / task.task_id.replace("/", "__")
                        / "zaxy"
                    )
                    for task in ARTICLE_TASKS
                },
                status_run_dir_overrides={
                    task.task_id: str(
                        external
                        / "results"
                        / f"zaxy-{task.task_id.replace('/', '__')}"
                    )
                    for task in ARTICLE_TASKS
                },
                status_normalized_result_path_overrides={
                    task.task_id: str(
                        external
                        / ".ingestion"
                        / "runs"
                        / f"zaxy-{task.task_id.replace('/', '__')}"
                        / "normalized-result.json"
                    )
                    for task in ARTICLE_TASKS
                },
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "valid"
    assert gate["status"] == "passed"
    assert validation["evidence_audit"]["external_run_audit_artifacts_valid"] is True


def test_harvey_completion_gate_accepts_post_run_readiness_snapshot(tmp_path: Path) -> None:
    """After completion, readiness may report results_already_complete instead of ready."""
    external = tmp_path / "external-harvey-worktree"
    output_dir = tmp_path / "reports" / "harvey-lab-memory-ablation"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {"frameworks": [{"framework": "mem0", "runs": 1, "avg_final_score": 0.456}]},
                "normalized_results": _partial_external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    audit_paths = _write_harvey_run_gate_artifact_paths(
        output_dir,
        readiness_status="not_ready",
        readiness_blocking_reasons=["results_already_complete"],
        readiness_worktree=str(external.resolve()),
        status_worktree=str(external.resolve()),
        status_index_dir_overrides={
            task.task_id: str(
                external / ".ingestion" / "indexes" / task.task_id.replace("/", "__") / "zaxy"
            )
            for task in ARTICLE_TASKS
        },
        status_run_dir_overrides={
            task.task_id: str(external / "results" / f"zaxy-{task.task_id.replace('/', '__')}")
            for task in ARTICLE_TASKS
        },
        status_normalized_result_path_overrides={
            task.task_id: str(
                external
                / ".ingestion"
                / "runs"
                / f"zaxy-{task.task_id.replace('/', '__')}"
                / "normalized-result.json"
            )
            for task in ARTICLE_TASKS
        },
    )
    ready_path = Path(audit_paths["external_readiness_report_paths"][0])
    readiness = json.loads(ready_path.read_text(encoding="utf-8"))
    readiness["ready_task_count"] = len(ARTICLE_TASKS)
    readiness["run_ready_count"] = len(ARTICLE_TASKS)
    readiness["normalized_ready_count"] = len(ARTICLE_TASKS)
    readiness["evidence_audit"]["run_artifacts_ready_count"] = len(ARTICLE_TASKS)
    readiness["evidence_audit"]["normalized_result_ready_count"] = len(ARTICLE_TASKS)
    readiness["evidence_audit"]["memory_evidence_ready_count"] = len(ARTICLE_TASKS)
    readiness["evidence_audit"]["import_ready_count"] = len(ARTICLE_TASKS)
    ready_path.write_text(json.dumps(readiness), encoding="utf-8")
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external), str(output_dir)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **audit_paths,
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "valid"
    assert gate["status"] == "passed"
    assert validation["evidence_audit"]["external_run_audit_artifacts_valid"] is True


def test_harvey_completion_gate_blocks_external_comparison_model_mismatch(tmp_path: Path) -> None:
    """External scored-system rows must use the same article generator and judge setup."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    normalized_results = _external_comparison_normalized_results()
    first = normalized_results[0]
    models = first["models"]
    assert isinstance(models, dict)
    models["generator"] = "openai-compatible/not-the-article-generator"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": normalized_results,
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_comparison_model_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_comparison_model_mismatch"
    assert validation["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False
    assert gate["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False


def test_harvey_completion_gate_blocks_external_comparison_extra_task_rows(tmp_path: Path) -> None:
    """External scored-system rows must be exactly the ten pinned article tasks."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    normalized_results = _external_comparison_normalized_results()
    normalized_results.append(
        {
            "framework": "mem0",
            "task_id": "unpublished-suite/extra-task",
            "final_score": 0.456,
            "total_seconds": 90.0,
            "models": {
                "generator": "openai-compatible/gpt-5.5",
                "judge": "gpt-5.4-mini",
                "generator_reasoning_effort": "low",
                "temperature": 0.0,
            },
        }
    )
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS) + 1, "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": normalized_results,
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_comparison_not_full_suite"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_comparison_not_full_suite"
    assert validation["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False
    assert gate["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False


def test_harvey_completion_gate_blocks_complete_report_without_run_gate_artifacts(tmp_path: Path) -> None:
    """Publishable external claims should include the generated launch/status audit artifacts."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "missing_external_run_audit_artifacts"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_run_audit_artifacts"


def test_harvey_completion_gate_blocks_run_audit_artifacts_outside_external_root(tmp_path: Path) -> None:
    """External launch/readiness/status audit files should live under the cited Harvey worktree."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                tmp_path / "outside-audit",
                readiness_worktree=str(external.resolve()),
                status_worktree=str(external.resolve()),
                status_normalized_result_path_overrides=dict(
                    zip(task_scores, normalized_paths, strict=True)
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "missing_external_run_audit_artifact"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_run_audit_artifact"


def test_harvey_completion_gate_blocks_overbroad_external_root(tmp_path: Path) -> None:
    """Complete reports should not treat a broad parent directory as the Harvey worktree."""
    reports_dir = tmp_path / "outside-comparison"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        tmp_path / ".ingestion" / "runs",
        task_scores,
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(tmp_path)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                tmp_path / "outside-audit",
                readiness_worktree=str(tmp_path.resolve()),
                status_worktree=str(tmp_path.resolve()),
                status_normalized_result_path_overrides=dict(
                    zip(task_scores, normalized_paths, strict=True)
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_audit_root_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_audit_root_mismatch"


def test_harvey_completion_gate_blocks_not_ready_external_run_audit(tmp_path: Path) -> None:
    """Complete reports should not pass when the captured launch gate was not ready."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, readiness_status="not_ready"),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_readiness_not_ready"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_readiness_not_ready"


def test_harvey_completion_gate_blocks_readiness_model_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite readiness evidence for the same generator and judge."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                readiness_judge="gpt-4.1-mini",
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_readiness_config_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_readiness_config_mismatch"


def test_harvey_completion_gate_blocks_ready_status_with_readiness_blockers(tmp_path: Path) -> None:
    """Complete reports should not cite readiness evidence that still has blockers."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                readiness_blocking_reasons=["missing_model_credentials"],
                readiness_missing_credentials=["OPENAI_API_KEY"],
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_readiness_has_blockers"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_readiness_has_blockers"


def test_harvey_completion_gate_blocks_incomplete_external_run_status(tmp_path: Path) -> None:
    """Complete reports should not pass when the captured run status was incomplete."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, run_status="partial"),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_incomplete"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_incomplete"


def test_harvey_completion_gate_blocks_filtered_external_readiness_audit(tmp_path: Path) -> None:
    """Complete reports should not cite a single-task readiness artifact."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, readiness_expected_task_count=1),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_readiness_not_full_suite"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_readiness_not_full_suite"


def test_harvey_completion_gate_blocks_status_with_incomplete_import_count(tmp_path: Path) -> None:
    """Complete reports should cite status evidence showing all tasks import-ready."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, status_import_ready_count=9),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_not_full_suite"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_not_full_suite"


def test_harvey_completion_gate_blocks_status_without_task_level_evidence(tmp_path: Path) -> None:
    """Complete reports should not trust aggregate-only external run status."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, status_include_tasks=False),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_task_evidence_missing"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_task_evidence_missing"


def test_harvey_completion_gate_blocks_status_with_incomplete_task_evidence(tmp_path: Path) -> None:
    """Complete reports should require each pinned task to be import-ready in status evidence."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    incomplete_task = ARTICLE_TASKS[0].task_id
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_task_import_ready_overrides={incomplete_task: False},
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_task_not_ready"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_task_not_ready"


def test_harvey_completion_gate_blocks_status_normalized_result_path_mismatch(tmp_path: Path) -> None:
    """Complete reports should import the same normalized artifacts marked ready in status."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    mismatched_task = ARTICLE_TASKS[0].task_id
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_normalized_result_path_overrides={
                    mismatched_task: str(external / ".ingestion" / "runs" / "different" / "normalized-result.json")
                },
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_normalized_result_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_normalized_result_mismatch"


def test_harvey_completion_gate_blocks_non_manifest_normalized_result_path(tmp_path: Path) -> None:
    """Complete reports should import normalized results from the manifest run directory."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    wrong_task = ARTICLE_TASKS[0].task_id
    wrong_path = external / ".ingestion" / "runs" / "non-manifest-run" / "normalized-result.json"
    wrong_path.parent.mkdir(parents=True)
    first_path = Path(normalized_paths[0])
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert isinstance(first_payload, dict)
    wrong_path.write_text(json.dumps(first_payload), encoding="utf-8")
    normalized_paths[0] = str(wrong_path)
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_normalized_result_path_overrides=dict(
                    zip(task_scores, normalized_paths, strict=True)
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["task_id"] == wrong_task
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_normalized_result_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_normalized_result_mismatch"


def test_harvey_completion_gate_blocks_status_run_id_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite status evidence for the same imported run IDs."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    mismatched_task = ARTICLE_TASKS[0].task_id
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_run_id_overrides={mismatched_task: "zaxy-different-run"},
                status_run_dir_overrides={
                    mismatched_task: str(external / "results" / f"zaxy-{mismatched_task.replace('/', '__')}")
                },
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_run_id_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_run_id_mismatch"


def test_harvey_completion_gate_blocks_non_manifest_run_id(tmp_path: Path) -> None:
    """Complete reports should use the deterministic run IDs from the external run manifest."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    wrong_task = ARTICLE_TASKS[0].task_id
    wrong_run_id = "zaxy-non-manifest-run"
    first_path = Path(normalized_paths[0])
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert isinstance(first_payload, dict)
    first_payload["run_id"] = wrong_run_id
    paths = first_payload["paths"]
    assert isinstance(paths, dict)
    paths["results_run_dir"] = f"results/{wrong_run_id}"
    paths["answer"] = f"results/{wrong_run_id}/output/response.md"
    paths["tool_log"] = f"results/{wrong_run_id}/transcript.jsonl"
    paths["judge"] = f"results/{wrong_run_id}/scores.json"
    paths["run_metrics"] = f"results/{wrong_run_id}/metrics.json"
    first_path.write_text(json.dumps(first_payload), encoding="utf-8")
    wrong_run_dir = external / "results" / wrong_run_id
    (wrong_run_dir / "output").mkdir(parents=True)
    (wrong_run_dir / "output" / "response.md").write_text("Wrong run answer\n", encoding="utf-8")
    (wrong_run_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
        encoding="utf-8",
    )
    scores = first_payload["scores"]
    tooling = first_payload["tooling"]
    usage = first_payload["usage"]
    timing = first_payload["timing"]
    assert isinstance(scores, dict)
    assert isinstance(tooling, dict)
    assert isinstance(usage, dict)
    assert isinstance(timing, dict)
    (wrong_run_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (wrong_run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "memory_search_calls": tooling["memory_search_calls"],
                "memory_read_calls": tooling["memory_read_calls"],
                "total_tokens": usage["total_tokens"],
                "total_seconds": timing["total_seconds"],
            }
        ),
        encoding="utf-8",
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score, run_id=wrong_run_id if task_id == wrong_task else None)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_run_id_overrides={wrong_task: wrong_run_id},
                status_run_dir_overrides={
                    wrong_task: str(external / "results" / f"zaxy-{wrong_task.replace('/', '__')}")
                },
                status_normalized_result_path_overrides=dict(
                    zip(task_scores, normalized_paths, strict=True)
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["task_id"] == wrong_task
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_run_id_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_run_id_mismatch"


def test_harvey_completion_gate_blocks_readiness_worktree_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite readiness evidence from the imported external worktree."""
    external = tmp_path / "external-harvey-worktree"
    other_external = tmp_path / "other-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                readiness_worktree=str(other_external),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_audit_worktree_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_audit_worktree_mismatch"


def test_harvey_completion_gate_blocks_status_worktree_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite status evidence from the imported external worktree."""
    external = tmp_path / "external-harvey-worktree"
    other_external = tmp_path / "other-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_worktree=str(other_external),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_audit_worktree_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_audit_worktree_mismatch"


def test_harvey_completion_gate_blocks_status_run_dir_outside_worktree(tmp_path: Path) -> None:
    """Complete reports should cite Harvey status run directories under the external worktree."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    outside_run_dir = tmp_path / "outside-results" / "zaxy-red-flags"
    outside_run_dir.mkdir(parents=True)
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_run_dir_overrides={ARTICLE_TASKS[0].task_id: str(outside_run_dir)},
                status_normalized_result_path_overrides=dict(
                    zip(task_scores, normalized_paths, strict=True)
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_run_dir_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_run_dir_mismatch"


def test_harvey_completion_gate_blocks_non_manifest_status_run_dir(tmp_path: Path) -> None:
    """Complete reports should cite run directories from the manifest run path."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    wrong_run_dir = external / "results" / "non-manifest-run"
    wrong_run_dir.mkdir(parents=True)
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_run_dir_overrides={ARTICLE_TASKS[0].task_id: str(wrong_run_dir)},
                status_normalized_result_path_overrides=dict(
                    zip(task_scores, normalized_paths, strict=True)
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_run_dir_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_run_dir_mismatch"


def test_harvey_completion_gate_blocks_status_index_dir_outside_worktree(tmp_path: Path) -> None:
    """Complete reports should cite Harvey status index directories under the external worktree."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    outside_index_dir = tmp_path / "outside-indexes" / "zaxy-red-flags"
    outside_index_dir.mkdir(parents=True)
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_index_dir_overrides={ARTICLE_TASKS[0].task_id: str(outside_index_dir)},
                status_normalized_result_path_overrides=dict(
                    zip(task_scores, normalized_paths, strict=True)
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_index_dir_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_index_dir_mismatch"


def test_harvey_completion_gate_blocks_non_manifest_status_index_dir(tmp_path: Path) -> None:
    """Complete reports should cite index directories from the manifest index path."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    wrong_index_dir = external / ".ingestion" / "indexes" / "non-manifest-index" / "zaxy"
    wrong_index_dir.mkdir(parents=True)
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                status_index_dir_overrides={ARTICLE_TASKS[0].task_id: str(wrong_index_dir)},
                status_normalized_result_path_overrides=dict(
                    zip(task_scores, normalized_paths, strict=True)
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert validation["evidence_failures"][0]["reason"] == "external_run_status_index_dir_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_status_index_dir_mismatch"


def test_harvey_completion_gate_blocks_readiness_commit_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite readiness evidence from the same Harvey commit."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, readiness_commit="different-commit"),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_audit_commit_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_audit_commit_mismatch"


def test_harvey_completion_gate_blocks_status_commit_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite status evidence from the same Harvey commit."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, status_commit="different-commit"),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_audit_commit_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_audit_commit_mismatch"


def test_harvey_completion_gate_blocks_readiness_without_commit_metadata(tmp_path: Path) -> None:
    """Complete reports should not cite readiness evidence without checkout metadata."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, readiness_commit=None),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_audit_commit_missing"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_audit_commit_missing"


def test_harvey_completion_gate_blocks_status_without_commit_metadata(tmp_path: Path) -> None:
    """Complete reports should not cite status evidence without checkout metadata."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(external, status_commit=None),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_audit_commit_missing"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_audit_commit_missing"


def test_harvey_completion_gate_blocks_manifest_generator_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite an external run manifest matching Zaxy row config."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_generator="different-generator",
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_config_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_config_mismatch"


def test_harvey_completion_gate_blocks_manifest_task_list_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite a manifest listing the pinned article tasks."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    manifest_task_ids = [task.task_id for task in ARTICLE_TASKS]
    manifest_task_ids[-1] = "not/an-article-task"
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_task_ids=manifest_task_ids,
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"


def test_harvey_completion_gate_blocks_manifest_expected_result_path_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite manifest paths for the expected Zaxy run IDs."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    expected_paths = [
        f".ingestion/runs/zaxy-{task.task_id.replace('/', '__')}/normalized-result.json"
        for task in ARTICLE_TASKS
    ]
    expected_paths[0] = ".ingestion/runs/zaxy-wrong/normalized-result.json"
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_expected_normalized_results=expected_paths,
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"


def test_harvey_completion_gate_blocks_manifest_judge_run_id_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite judge commands for the expected Zaxy run IDs."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    judge_commands = [
        (
            "uv run python -m evaluation.run_eval --judge-model gpt-5.4-mini "
            f"--run-id zaxy-{task.task_id.replace('/', '__')} --task {task.task_id}"
        )
        for task in ARTICLE_TASKS
    ]
    judge_commands[0] = (
        "uv run python -m evaluation.run_eval --judge-model gpt-5.4-mini "
        f"--run-id zaxy-wrong --task {ARTICLE_TASKS[0].task_id}"
    )
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_judge_commands=judge_commands,
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"


def test_harvey_completion_gate_blocks_manifest_run_command_task_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite harness commands for the expected Harvey task IDs."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    run_commands = [
        (
            "uv run python -m harness.run --model openai-compatible/gpt-5.5 "
            f"--task {task.task_id} --run-id zaxy-{task.task_id.replace('/', '__')}"
        )
        for task in ARTICLE_TASKS
    ]
    run_commands[0] = (
        "uv run python -m harness.run --model openai-compatible/gpt-5.5 "
        f"--task not/an-article-task --run-id zaxy-{ARTICLE_TASKS[0].task_id.replace('/', '__')}"
    )
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_run_commands=run_commands,
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"


def test_harvey_completion_gate_blocks_manifest_judge_model_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite judge commands using the manifest judge model."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    judge_commands = [
        (
            "uv run python -m evaluation.run_eval --judge-model different-judge "
            f"--run-id zaxy-{task.task_id.replace('/', '__')} --task {task.task_id}"
        )
        for task in ARTICLE_TASKS
    ]
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_judge_commands=judge_commands,
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"


def test_harvey_completion_gate_blocks_manifest_reasoning_effort_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite harness commands using the manifest reasoning effort."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    run_commands = [
        (
            "uv run python -m harness.run --model openai-compatible/gpt-5.5 "
            f"--task {task.task_id} --run-id zaxy-{task.task_id.replace('/', '__')}"
        )
        for task in ARTICLE_TASKS
    ]
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_run_commands=run_commands,
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_task_mismatch"


def test_harvey_completion_gate_blocks_manifest_source_url_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite the expected external Harvey suite source."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_source_url="https://example.com/not-harvey",
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_source_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_source_mismatch"


def test_harvey_completion_gate_blocks_manifest_collection_command_mismatch(tmp_path: Path) -> None:
    """Complete reports should cite the expected external collection output."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_collection_command=(
                    "uv run python scripts/memory_ablation/collect_results.py "
                    "--framework zaxy --output /tmp/not-the-zaxy-comparison.json"
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_command_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_command_mismatch"


def test_harvey_completion_gate_blocks_manifest_comparison_command_mismatch(tmp_path: Path) -> None:
    """Complete reports should import from the external collection artifact into the report artifact."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(
                external,
                manifest_comparison_command=(
                    "zaxy harvey-lab-import /tmp/not-the-zaxy-comparison.json "
                    "--out harvey-lab-benchmark.json"
                ),
            ),
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "external_run_manifest_command_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "external_run_manifest_command_mismatch"


def test_harvey_completion_gate_blocks_external_aggregate_with_only_zaxy_result_evidence(tmp_path: Path) -> None:
    """External rank evidence should include result rows for non-Zaxy systems."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": [
                    {"framework": "zaxy", "task_id": ARTICLE_TASKS[0].task_id, "final_score": 0.64}
                ],
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "missing_external_comparison_result_evidence"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_comparison_result_evidence"


def test_harvey_completion_gate_blocks_external_aggregate_without_matching_result_frameworks(
    tmp_path: Path,
) -> None:
    """External rank evidence should tie aggregate frameworks to underlying result rows."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": [
                    {"framework": "graphiti", "task_id": ARTICLE_TASKS[0].task_id, "final_score": 0.5}
                ],
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "missing_external_comparison_result_evidence"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_comparison_result_evidence"


def test_harvey_completion_gate_blocks_external_aggregate_score_mismatch(tmp_path: Path) -> None:
    """External aggregate means and runs should be derived from cited result rows."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.900},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "stale_external_comparison_result_evidence"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "stale_external_comparison_result_evidence"


def test_harvey_completion_gate_blocks_external_aggregate_latency_mismatch(tmp_path: Path) -> None:
    """External aggregate latency should match cited result-row timings when present."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {
                            "framework": "mem0",
                            "runs": len(ARTICLE_TASKS),
                            "avg_final_score": 0.456,
                            "avg_total_seconds": 90.0,
                        },
                        {
                            "framework": "raw-rg",
                            "runs": len(ARTICLE_TASKS),
                            "avg_final_score": 0.399,
                            "avg_total_seconds": 40.0,
                        },
                    ]
                },
                "normalized_results": [
                        {
                            "framework": item["framework"],
                            "task_id": item["task_id"],
                            "final_score": item["final_score"],
                            "total_seconds": 10.0 if item["framework"] == "mem0" else item["total_seconds"],
                            "models": item["models"],
                        }
                        for item in _external_comparison_normalized_results()
                    ],
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "stale_external_comparison_result_evidence"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "stale_external_comparison_result_evidence"


def test_harvey_completion_gate_blocks_no_memory_or_missing_artifact_evidence() -> None:
    """Complete task coverage is not enough without external memory-use evidence."""
    no_memory_report = build_harvey_lab_report(
        [
            _zaxy_result(
                task.task_id,
                task.best_score + 0.01,
                memory_search_calls=0,
                memory_read_calls=0,
            )
            for task in ARTICLE_TASKS
        ]
    )

    no_memory_gate = check_harvey_lab_completion(no_memory_report)

    assert no_memory_gate["status"] == "blocked"
    assert no_memory_gate["evidence_failures"]
    assert no_memory_gate["evidence_failures"][0]["reason"] == "memory_tools_not_used"

    missing_paths = _zaxy_result(
        ARTICLE_TASKS[0].task_id,
        ARTICLE_TASKS[0].best_score + 0.01,
    )
    missing_paths["paths"] = {
        "answer": "results/zaxy/output/response.md",
        "tool_log": "",
        "judge": "",
        "run_metrics": "",
    }
    mixed_report = build_harvey_lab_report(
        [missing_paths]
        + [
            _zaxy_result(task.task_id, task.best_score + 0.01)
            for task in ARTICLE_TASKS[1:]
        ]
    )

    mixed_gate = check_harvey_lab_completion(mixed_report)

    assert mixed_gate["status"] == "blocked"
    assert mixed_gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert mixed_gate["evidence_failures"][0]["reason"] == "missing_reviewable_artifact_paths"


def test_harvey_completion_gate_blocks_mixed_zaxy_model_configurations(tmp_path: Path) -> None:
    """Publishable Zaxy rows should use one fixed generator and judge configuration."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    zaxy_rows = [
        _zaxy_result(
            task.task_id,
            task.best_score + 0.01,
            generator=(
                "openai-compatible/gpt-5.5"
                if index == 0
                else "openai-compatible/gpt-5.5-rerun"
            ),
        )
        for index, task in enumerate(ARTICLE_TASKS)
    ]
    report = build_harvey_lab_report(
        zaxy_rows,
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "mixed_zaxy_model_configuration"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "mixed_zaxy_model_configuration"


def test_harvey_completion_gate_blocks_mixed_zaxy_generation_settings(tmp_path: Path) -> None:
    """Publishable Zaxy rows should use one fixed reasoning effort and temperature."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    zaxy_rows = [
        _zaxy_result(
            task.task_id,
            task.best_score + 0.01,
            generator_reasoning_effort=("low" if index == 0 else "medium"),
        )
        for index, task in enumerate(ARTICLE_TASKS)
    ]
    report = build_harvey_lab_report(
        zaxy_rows,
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "mixed_zaxy_generation_settings"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "mixed_zaxy_generation_settings"


def test_harvey_completion_gate_blocks_zaxy_run_commit_mismatch(tmp_path: Path) -> None:
    """Publishable Zaxy rows should match the external Harvey suite commit in provenance."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    zaxy_rows = [
        _zaxy_result(
            task_id,
            score,
            commit="different-harvey-commit",
        )
        for task_id, score in task_scores.items()
    ]
    normalized_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    zaxy_by_task = {
        str(row["task_id"]): row
        for row in zaxy_rows
    }
    for normalized_path in normalized_paths:
        path = Path(normalized_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(zaxy_by_task[str(payload["task_id"])]),
            encoding="utf-8",
        )
    report = build_harvey_lab_report(
        zaxy_rows,
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "zaxy_run_commit_mismatch"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "zaxy_run_commit_mismatch"


def test_harvey_completion_gate_blocks_complete_results_without_external_provenance() -> None:
    """Complete task coverage still needs external result-source provenance."""
    report = build_harvey_lab_report(
        [
            _zaxy_result(task.task_id, task.best_score + 0.01)
            for task in ARTICLE_TASKS
        ]
    )

    gate = check_harvey_lab_completion(report)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_result_provenance"


def test_harvey_completion_gate_blocks_collated_json_without_normalized_result_paths() -> None:
    """A collated JSON file alone is not enough external proof for publication."""
    report = build_harvey_lab_report(
        [
            _zaxy_result(task.task_id, task.best_score + 0.01)
            for task in ARTICLE_TASKS
        ],
        result_provenance={
            "source": "harvey-lab-benchmark",
            "zaxy_results_json_path": "external/zaxy-normalized-results.json",
        },
    )

    gate = check_harvey_lab_completion(report)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_normalized_result_paths"


def test_harvey_completion_gate_blocks_missing_normalized_result_artifacts() -> None:
    """Normalized-result provenance must point at reviewable local artifacts."""
    report = build_harvey_lab_report(
        [
            _zaxy_result(task.task_id, task.best_score + 0.01)
            for task in ARTICLE_TASKS
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": ["external-harvey-worktree"],
            "normalized_result_paths": [f"missing/{idx}/normalized-result.json" for idx, _ in enumerate(ARTICLE_TASKS)],
        },
    )

    gate = check_harvey_lab_completion(report)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_normalized_result_artifact"


def test_harvey_completion_gate_blocks_normalized_results_outside_external_root(tmp_path: Path) -> None:
    """Normalized-result files should live under the cited external Harvey worktree."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    worktree = tmp_path / "external-harvey-worktree"
    outside = tmp_path / "outside-runs"
    normalized_paths = _write_normalized_result_paths(outside, task_scores)
    for path_value in normalized_paths:
        normalized_path = Path(path_value)
        payload = json.loads(normalized_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        paths = payload["paths"]
        assert isinstance(paths, dict)
        paths["answer"] = "response.md"
        paths["tool_log"] = "transcript.jsonl"
        paths["judge"] = "scores.json"
        paths["run_metrics"] = "metrics.json"
        normalized_path.write_text(json.dumps(payload), encoding="utf-8")
        normalized_dir = normalized_path.parent
        (normalized_dir / "response.md").write_text("Answer\n", encoding="utf-8")
        (normalized_dir / "transcript.jsonl").write_text(
            json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
            encoding="utf-8",
        )
        (normalized_dir / "scores.json").write_text(
            json.dumps({"final_score": payload["scores"]["final_score"]}),
            encoding="utf-8",
        )
        (normalized_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "memory_search_calls": payload["tooling"]["memory_search_calls"],
                    "memory_read_calls": payload["tooling"]["memory_read_calls"],
                    "total_tokens": payload["usage"]["total_tokens"],
                    "total_seconds": payload["timing"]["total_seconds"],
                }
            ),
            encoding="utf-8",
        )
    reports_dir = worktree / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(worktree)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(worktree),
        },
    )

    gate = check_harvey_lab_completion(report)
    validation = validate_harvey_lab_report(report, require_complete=True)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_normalized_result_artifact"
    assert validation["status"] == "invalid"


def test_harvey_completion_gate_blocks_normalized_result_mismatch(tmp_path: Path) -> None:
    """Provenance artifacts must match the Zaxy rows summarized by the report."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    artifact_scores = dict(task_scores)
    artifact_scores[ARTICLE_TASKS[0].task_id] = ARTICLE_TASKS[0].best_score - 0.2
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(tmp_path / "external-harvey-worktree")],
            "normalized_result_paths": _write_normalized_result_paths(
                tmp_path / "external-harvey-worktree" / ".ingestion" / "runs",
                artifact_scores,
            ),
        },
    )

    gate = check_harvey_lab_completion(report)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "normalized_result_mismatch"


def test_harvey_completion_gate_blocks_normalized_result_metadata_mismatch(tmp_path: Path) -> None:
    """Provenance artifacts must also match Zaxy model, corpus, usage, and tool metadata."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    artifact_paths = _write_normalized_result_paths(
        external / ".ingestion" / "runs",
        task_scores,
    )
    zaxy_rows = [
        _zaxy_result(
            task_id,
            score,
            generator="openai-compatible/gpt-5.5-reported",
        )
        for task_id, score in task_scores.items()
    ]
    report = build_harvey_lab_report(
        zaxy_rows,
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": artifact_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    gate = check_harvey_lab_completion(report)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "normalized_result_metadata_mismatch"


def test_harvey_completion_gate_blocks_missing_referenced_run_artifacts(tmp_path: Path) -> None:
    """A normalized-result file must point to existing answer, transcript, judge, and metric files."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(tmp_path / "external-harvey-worktree")],
            "normalized_result_paths": _write_normalized_result_paths(
                tmp_path / "external-harvey-worktree" / ".ingestion" / "runs",
                task_scores,
                create_run_artifacts=False,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
        },
    )

    gate = check_harvey_lab_completion(report)
    validation = validate_harvey_lab_report(report, require_complete=True)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "missing_referenced_run_artifact"
    assert validation["status"] == "invalid"


def test_harvey_completion_gate_blocks_non_manifest_answer_artifact_path(tmp_path: Path) -> None:
    """Normalized results should cite the answer from the manifest run artifact path."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    worktree = tmp_path / "external-harvey-worktree"
    normalized_paths = _write_normalized_result_paths(
        worktree / ".ingestion" / "runs",
        task_scores,
    )
    first_path = Path(normalized_paths[0])
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert isinstance(first_payload, dict)
    paths = first_payload["paths"]
    assert isinstance(paths, dict)
    alternate_answer = worktree / "results" / "alternate-run" / "output" / "response.md"
    alternate_answer.parent.mkdir(parents=True)
    alternate_answer.write_text("Alternate answer\n", encoding="utf-8")
    paths["answer"] = "results/alternate-run/output/response.md"
    first_path.write_text(json.dumps(first_payload), encoding="utf-8")
    reports_dir = worktree / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, score) for task_id, score in task_scores.items()],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(worktree)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(worktree),
        },
    )

    gate = check_harvey_lab_completion(report)
    validation = validate_harvey_lab_report(report, require_complete=True)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "missing_referenced_run_artifact"
    assert gate["evidence_failures"][0]["path_key"] == "answer"
    assert validation["status"] == "invalid"


def test_harvey_completion_gate_blocks_transcripts_without_memory_tool_events(tmp_path: Path) -> None:
    """Metrics counters are not enough if the transcript does not show memory tool calls."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    worktree = tmp_path / "external-harvey-worktree"
    normalized_paths = _write_normalized_result_paths(
        worktree / ".ingestion" / "runs",
        task_scores,
    )
    first_result = _zaxy_result(ARTICLE_TASKS[0].task_id, ARTICLE_TASKS[0].best_score + 0.01)
    result_paths = first_result["paths"]
    assert isinstance(result_paths, dict)
    tool_log = worktree / str(result_paths["tool_log"])
    tool_log.write_text(
        json.dumps(
            {
                "turn": 1,
                "role": "tool",
                "tool_name": "read",
                "arguments": {"file_path": "memo.txt"},
                "result_preview": "ordinary read output",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(worktree)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
        },
    )

    gate = check_harvey_lab_completion(report)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "missing_transcript_memory_tool_evidence"


def test_harvey_completion_gate_blocks_artifact_paths_outside_external_root(tmp_path: Path) -> None:
    """Normalized results should not cite arbitrary absolute artifacts outside the Harvey worktree."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    worktree = tmp_path / "external-harvey-worktree"
    normalized_paths = _write_normalized_result_paths(
        worktree / ".ingestion" / "runs",
        task_scores,
    )
    outside_transcript = tmp_path / "outside-transcript.jsonl"
    outside_transcript.write_text(
        json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
        encoding="utf-8",
    )
    first_path = Path(normalized_paths[0])
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert isinstance(first_payload, dict)
    paths = first_payload["paths"]
    assert isinstance(paths, dict)
    paths["tool_log"] = str(outside_transcript)
    first_path.write_text(json.dumps(first_payload), encoding="utf-8")
    reports_dir = worktree / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(worktree)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(worktree),
        },
    )

    gate = check_harvey_lab_completion(report)
    validation = validate_harvey_lab_report(report, require_complete=True)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "missing_referenced_run_artifact"
    assert gate["evidence_failures"][0]["path_key"] == "tool_log"
    assert validation["status"] == "invalid"


def test_harvey_completion_gate_blocks_relative_artifact_paths_escaping_external_root(
    tmp_path: Path,
) -> None:
    """Relative artifact paths should not escape the cited Harvey evidence roots."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    worktree = tmp_path / "external-harvey-worktree"
    normalized_paths = _write_normalized_result_paths(
        worktree / ".ingestion" / "runs",
        task_scores,
    )
    outside_transcript = tmp_path / "outside-transcript.jsonl"
    outside_transcript.write_text(
        json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
        encoding="utf-8",
    )
    first_path = Path(normalized_paths[0])
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert isinstance(first_payload, dict)
    paths = first_payload["paths"]
    assert isinstance(paths, dict)
    paths["tool_log"] = "../outside-transcript.jsonl"
    first_path.write_text(json.dumps(first_payload), encoding="utf-8")
    reports_dir = worktree / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(worktree)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(worktree),
        },
    )

    gate = check_harvey_lab_completion(report)
    validation = validate_harvey_lab_report(report, require_complete=True)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "missing_referenced_run_artifact"
    assert gate["evidence_failures"][0]["path_key"] == "tool_log"
    assert validation["status"] == "invalid"


def test_harvey_completion_gate_blocks_run_metrics_mismatch(tmp_path: Path) -> None:
    """Published memory-tool totals should match the cited external metrics artifact."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    worktree = tmp_path / "external-harvey-worktree"
    normalized_paths = _write_normalized_result_paths(
        worktree / ".ingestion" / "runs",
        task_scores,
    )
    first_result = _zaxy_result(ARTICLE_TASKS[0].task_id, ARTICLE_TASKS[0].best_score + 0.01)
    result_paths = first_result["paths"]
    assert isinstance(result_paths, dict)
    metrics_path = worktree / str(result_paths["run_metrics"])
    metrics_path.write_text(
        json.dumps({"memory_search_calls": 0, "memory_read_calls": 0}),
        encoding="utf-8",
    )
    reports_dir = worktree / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(worktree)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    gate = check_harvey_lab_completion(report)
    validation = validate_harvey_lab_report(report, require_complete=True)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "run_metrics_mismatch"
    assert validation["status"] == "invalid"


def test_harvey_completion_gate_blocks_judge_score_mismatch(tmp_path: Path) -> None:
    """Published Zaxy scores should match the cited external Harvey judge artifact."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    worktree = tmp_path / "external-harvey-worktree"
    normalized_paths = _write_normalized_result_paths(
        worktree / ".ingestion" / "runs",
        task_scores,
    )
    first_result = _zaxy_result(ARTICLE_TASKS[0].task_id, ARTICLE_TASKS[0].best_score + 0.01)
    result_paths = first_result["paths"]
    assert isinstance(result_paths, dict)
    judge_path = worktree / str(result_paths["judge"])
    judge_path.write_text(
        json.dumps({"final_score": ARTICLE_TASKS[0].best_score - 0.2}),
        encoding="utf-8",
    )
    reports_dir = worktree / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(worktree)],
            "normalized_result_paths": normalized_paths,
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )

    gate = check_harvey_lab_completion(report)
    validation = validate_harvey_lab_report(report, require_complete=True)

    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["task_id"] == ARTICLE_TASKS[0].task_id
    assert gate["evidence_failures"][0]["reason"] == "judge_score_mismatch"
    assert validation["status"] == "invalid"


def test_harvey_completion_gate_blocks_complete_report_without_harvey_commit(tmp_path: Path) -> None:
    """Publishable external results should identify the Harvey suite checkout."""
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(tmp_path / "external-harvey-worktree")],
            "normalized_result_paths": _write_normalized_result_paths(
                tmp_path / "external-harvey-worktree" / ".ingestion" / "runs",
                task_scores,
            ),
        },
    )

    validation = validate_harvey_lab_report(report)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "valid"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_harvey_suite_commit"


def test_harvey_completion_gate_blocks_missing_external_baseline_artifacts(tmp_path: Path) -> None:
    """Publishable reports should keep cited Harvey baseline aggregate artifacts reviewable."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )
    comparison_path.unlink()

    validation = validate_harvey_lab_report(report, require_complete=True)
    gate = check_harvey_lab_completion(report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "missing_external_baseline_report_artifact"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_baseline_report_artifact"


def test_harvey_completion_gate_blocks_missing_external_comparison_scorecard(tmp_path: Path) -> None:
    """Publishable reports should include the Zaxy-vs-scored-systems leaderboard."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )
    stale_report = replace(report, external_comparison_scorecard={})

    validation = validate_harvey_lab_report(stale_report, require_complete=True)
    gate = check_harvey_lab_completion(stale_report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "missing_external_comparison_scorecard"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "missing_external_comparison_scorecard"


def test_harvey_completion_gate_blocks_stale_zaxy_external_comparison_row(tmp_path: Path) -> None:
    """Publishable reports should not carry stale Zaxy leaderboard statistics."""
    external = tmp_path / "external-harvey-worktree"
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
        },
    )
    stale_zaxy_row = replace(
        report.external_comparison_scorecard["Zaxy"],
        runs=1,
        mean_score=0.1,
    )
    stale_report = replace(
        report,
        external_comparison_scorecard={
            **report.external_comparison_scorecard,
            "Zaxy": stale_zaxy_row,
        },
    )

    validation = validate_harvey_lab_report(stale_report, require_complete=True)
    gate = check_harvey_lab_completion(stale_report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "stale_zaxy_external_comparison_row"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "stale_zaxy_external_comparison_row"


def test_harvey_completion_gate_blocks_stale_external_scored_system_row(tmp_path: Path) -> None:
    """Publishable reports should not carry stale non-Zaxy leaderboard statistics."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )
    stale_mem0_row = replace(
        report.external_comparison_scorecard["mem0"],
        rank_by_mean_score=99,
    )
    stale_report = replace(
        report,
        external_comparison_scorecard={
            **report.external_comparison_scorecard,
            "mem0": stale_mem0_row,
        },
    )

    validation = validate_harvey_lab_report(stale_report, require_complete=True)
    gate = check_harvey_lab_completion(stale_report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "stale_external_comparison_scorecard"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "stale_external_comparison_scorecard"


def test_harvey_completion_gate_blocks_stale_external_baseline_scorecard(tmp_path: Path) -> None:
    """Publishable reports should recompute baseline rows from cited Harvey artifacts."""
    external = tmp_path / "external-harvey-worktree"
    reports_dir = external / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
        },
    )
    stale_mem0_baseline = replace(
        report.external_baseline_scorecard["mem0"],
        mean_score=0.999,
    )
    stale_report = replace(
        report,
        external_baseline_scorecard={
            **report.external_baseline_scorecard,
            "mem0": stale_mem0_baseline,
        },
    )

    validation = validate_harvey_lab_report(stale_report, require_complete=True)
    gate = check_harvey_lab_completion(stale_report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "stale_external_baseline_scorecard"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "stale_external_baseline_scorecard"


def test_harvey_completion_gate_blocks_stale_framework_fit(tmp_path: Path) -> None:
    """Publishable reports should keep Framework Fit aligned with scored Zaxy rows."""
    external = tmp_path / "external-harvey-worktree"
    task_scores = {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS}
    report = build_harvey_lab_report(
        [
            _zaxy_result(task_id, score)
            for task_id, score in task_scores.items()
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(external)],
            "normalized_result_paths": _write_normalized_result_paths(
                external / ".ingestion" / "runs",
                task_scores,
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
        },
    )
    stale_zaxy_fit = replace(
        report.framework_fit["Zaxy"],
        where_strongest="Pending external Harvey LAB runs",
    )
    stale_report = replace(
        report,
        framework_fit={
            **report.framework_fit,
            "Zaxy": stale_zaxy_fit,
        },
    )

    validation = validate_harvey_lab_report(stale_report, require_complete=True)
    gate = check_harvey_lab_completion(stale_report)

    assert validation["status"] == "invalid"
    assert validation["evidence_failures"][0]["reason"] == "stale_framework_fit"
    assert gate["status"] == "blocked"
    assert gate["evidence_failures"][0]["reason"] == "stale_framework_fit"


def test_harvey_report_validation_checks_partial_external_artifacts(tmp_path: Path) -> None:
    """Partial reports should still validate their existing external artifacts."""
    task_id = "corporate-ma/review-data-room-red-flag-review"
    result_paths = _write_normalized_result_paths(
        tmp_path / "external" / ".ingestion" / "runs",
        {task_id: 0.64},
    )
    report = build_harvey_lab_report(
        [_zaxy_result(task_id, 0.64)],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(tmp_path / "external")],
            "normalized_result_paths": result_paths,
        },
    )

    validation = validate_harvey_lab_report(report)

    assert validation["status"] == "valid"
    assert validation["completed_task_count"] == 1
    assert validation["require_complete"] is False
    assert validation["evidence_audit"]["normalized_result_artifacts"] == 1
    assert validation["evidence_audit"]["external_comparison_artifacts"] == 0
    assert validation["evidence_audit"]["judge_score_artifacts_match"] is True
    assert validation["evidence_audit"]["run_metrics_artifacts_match"] is True
    assert validation["evidence_audit"]["transcript_memory_tool_evidence"] is True
    assert validation["evidence_audit"]["external_comparison_recomputed_from_non_zaxy_rows"] is False
    assert validation["evidence_audit"]["external_run_manifest_artifacts"] == 0
    assert validation["evidence_audit"]["external_readiness_report_artifacts"] == 0
    assert validation["evidence_audit"]["external_status_report_artifacts"] == 0
    assert validation["evidence_audit"]["external_run_audit_artifacts_valid"] is False

    broken = build_harvey_lab_report(
        [_zaxy_result(task_id, 0.64)],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(tmp_path / "external")],
            "normalized_result_paths": [str(tmp_path / "missing" / "normalized-result.json")],
        },
    )

    broken_validation = validate_harvey_lab_report(broken)

    assert broken_validation["status"] == "invalid"
    assert broken_validation["evidence_failures"][0]["reason"] == "missing_normalized_result_artifact"


def test_harvey_publication_markdown_requires_passed_gate(tmp_path: Path) -> None:
    """Publishable statistics should only render after the strict external gate passes."""
    partial_report = build_harvey_lab_report(
        [_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)]
    )

    try:
        render_harvey_publication_markdown(partial_report)
    except ValueError as exc:
        assert "not publishable" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected publication renderer to reject partial report")

    reports_dir = tmp_path / "external-harvey-worktree" / ".ingestion" / "reports"
    reports_dir.mkdir(parents=True)
    comparison_path = reports_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "frameworks": [
                        {"framework": "mem0", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.456},
                        {"framework": "raw-rg", "runs": len(ARTICLE_TASKS), "avg_final_score": 0.399},
                    ]
                },
                "normalized_results": _external_comparison_normalized_results(),
            }
        ),
        encoding="utf-8",
    )
    complete_report = build_harvey_lab_report(
        [
            _zaxy_result(task.task_id, task.best_score + 0.01)
            for task in ARTICLE_TASKS
        ],
        result_provenance={
            "source": "harvey-lab-import",
            "roots": [str(tmp_path / "external-harvey-worktree")],
            "normalized_result_paths": _write_normalized_result_paths(
                tmp_path / "external-harvey-worktree" / ".ingestion" / "runs",
                {task.task_id: task.best_score + 0.01 for task in ARTICLE_TASKS},
            ),
            "harvey_git_commit": "29748828133dff83ad2263af353fb035504f8f77",
            "external_baseline_report_paths": [str(comparison_path)],
            **_write_harvey_run_gate_artifact_paths(tmp_path / "external-harvey-worktree"),
        },
    )

    markdown = render_harvey_publication_markdown(complete_report)

    assert "# Harvey LAB Publishable Comparative Statistics" in markdown
    assert "| Zaxy | 10 | 0.717 | +0.114 | +0.010 | 10 |" in markdown
    assert "## Zaxy External Position" in markdown
    assert "| Rank vs external scored systems | Delta vs source raw-rg | Delta vs best external | Framework Fit |" in markdown
    assert "| 1 | +0.318 | +0.261 | Zaxy is ahead of the published article-best rows on the scored subset. |" in markdown
    assert "## Zaxy Run Configuration" in markdown
    assert "| Generator | Judge | Generator reasoning effort | Judge reasoning effort | Temperature | Harvey commit |" in markdown
    assert "| openai-compatible/gpt-5.5 | gpt-5.4-mini | low |  | 0.000 | 29748828133dff83ad2263af353fb035504f8f77 |" in markdown
    assert "| article best observed | best published memory/search row per article task | 10 | 0.707 | +0.104 |" in markdown
    assert "## External Baseline Aggregate" in markdown
    assert "| mem0 | Harvey-native comparison artifact | 10 | 0.456 | +0.057 |" in markdown
    assert "/tmp/" not in markdown
    assert "## Framework Fit" in markdown
    assert "raw-rg is a retrieval/search baseline" in markdown


def test_harvey_benchmark_cli_writes_report(tmp_path: Path) -> None:
    """The public CLI should expose the external Harvey LAB benchmark report."""
    zaxy_results = tmp_path / "zaxy-results.json"
    zaxy_results.write_text(
        json.dumps([_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-benchmark",
            "--zaxy-results",
            str(zaxy_results),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Harvey LAB external memory benchmark: partial" in result.output
    assert (output_dir / "harvey-lab-benchmark.json").exists()
    assert (output_dir / "harvey-lab-benchmark.md").exists()
    assert "Framework Fit" in report_to_markdown(build_harvey_lab_report(load_harvey_zaxy_results(zaxy_results)))


def test_harvey_lab_import_cli_writes_report_from_external_result_tree(tmp_path: Path) -> None:
    """The CLI should consume a Harvey worktree-style result directory directly."""
    runs = tmp_path / "harvey" / ".ingestion" / "runs" / "zaxy-red-flags"
    runs.mkdir(parents=True)
    (runs / "normalized-result.json").write_text(
        json.dumps(_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-import",
            str(tmp_path / "harvey"),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Imported Zaxy Harvey LAB normalized results: 1" in result.output
    payload = json.loads((output_dir / "harvey-lab-benchmark.json").read_text(encoding="utf-8"))
    assert payload["summary"]["zaxy_task_count"] == 1
    assert payload["result_provenance"]["source"] == "harvey-lab-import"
    assert str(tmp_path / "harvey") in payload["result_provenance"]["roots"]
    assert "framework_scorecard" in payload


def test_harvey_lab_import_cli_records_output_dir_run_gate_artifacts(tmp_path: Path) -> None:
    """Generated runner audit files live beside the report and should be imported as provenance."""
    harvey = tmp_path / "harvey"
    runs = harvey / ".ingestion" / "runs" / "zaxy-red-flags"
    runs.mkdir(parents=True)
    (runs / "normalized-result.json").write_text(
        json.dumps(_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest = output_dir / "harvey-lab-external-run.json"
    ready = output_dir / "harvey-lab-ready.json"
    status = output_dir / "harvey-lab-status.json"
    manifest.write_text(
        json.dumps({"schema_version": "zaxy.harvey-lab-external-run.v1"}),
        encoding="utf-8",
    )
    ready.write_text(
        json.dumps({"schema_version": "zaxy.harvey-lab-run-readiness.v1"}),
        encoding="utf-8",
    )
    status.write_text(
        json.dumps({"schema_version": "zaxy.harvey-lab-run-status.v1"}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-import",
            str(harvey),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((output_dir / "harvey-lab-benchmark.json").read_text(encoding="utf-8"))
    provenance = payload["result_provenance"]
    assert str(output_dir.resolve()) in provenance["roots"]
    assert provenance["external_run_manifest_paths"] == [str(manifest.resolve())]
    assert provenance["external_readiness_report_paths"] == [str(ready.resolve())]
    assert provenance["external_status_report_paths"] == [str(status.resolve())]


def test_harvey_lab_import_cli_can_write_baseline_only_handoff_report(tmp_path: Path) -> None:
    """Baseline-only mode should expose Harvey comparison artifacts without Zaxy scores."""
    harvey = tmp_path / "harvey"
    reports = harvey / ".ingestion" / "reports"
    reports.mkdir(parents=True)
    comparison = reports / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "schema_version": "harvey.memory_comparison.v1",
                "aggregate": {"frameworks": [{"framework": "raw-rg"}]},
                "normalized_results": [{"framework": "raw-rg"}],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-import",
            str(harvey),
            "--allow-baseline-only",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Imported Zaxy Harvey LAB normalized results: 0" in result.output
    assert "Baseline-only handoff report" in result.output
    payload = json.loads((output_dir / "harvey-lab-benchmark.json").read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert payload["summary"]["zaxy_task_count"] == 0
    assert payload["result_provenance"]["external_baseline_report_paths"] == [str(comparison.resolve())]
    assert payload["external_baseline_scorecard"]["raw-rg"]["mean_score"] is None
    markdown = (output_dir / "harvey-lab-benchmark.md").read_text(encoding="utf-8")
    assert "External baseline reports" in markdown
    assert "comparison.json" in markdown
    assert str(comparison.resolve()) not in markdown


def test_harvey_lab_plan_cli_writes_external_run_manifest(tmp_path: Path) -> None:
    """The CLI should render the reproducible external Harvey run plan."""
    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-plan",
            "--output-dir",
            str(tmp_path),
            "--generator",
            "openai-compatible/gpt-5.5",
            "--judge",
            "gpt-5.4-mini",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "harvey-lab-external-run.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "harvey-lab-external-run.md").read_text(encoding="utf-8")
    runner = tmp_path / "run-harvey-lab-zaxy.sh"
    assert payload["task_count"] == 10
    assert payload["publish_output_path"].endswith("publishable-statistics.md")
    assert runner.exists()
    assert os.access(runner, os.X_OK)
    assert "zaxy harvey-lab-adapter-kit" in markdown
    assert "zaxy harvey-lab-doctor" in markdown
    assert "zaxy harvey-lab-status" in markdown
    assert "corporate-ma/draft-acquisition-due-diligence" in markdown
    assert "harvey-lab-normalize-run" in markdown
    assert "## Validate And Publish" in markdown
    assert "zaxy harvey-lab-publish" in markdown
    assert "export_result.py" not in markdown


def test_harvey_lab_plan_cli_defaults_to_explicit_model_placeholders(tmp_path: Path) -> None:
    """Default plan metadata should not invent current generator or judge model names."""
    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-plan",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "harvey-lab-external-run.json").read_text(encoding="utf-8"))
    assert payload["generator"] == "HARVEY_GENERATOR_MODEL"
    assert payload["judge"] == "HARVEY_JUDGE_MODEL"


def test_harvey_lab_gate_cli_rejects_partial_report(tmp_path: Path) -> None:
    """The CLI gate should fail when fewer than ten Zaxy rows are present."""
    report = build_harvey_lab_report(
        [_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)]
    )
    written = write_harvey_lab_report(report, tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-gate",
            str(written.json_path),
        ],
    )

    assert result.exit_code == 1
    assert "blocked" in result.output
    assert "missing_task_ids" in result.output


def test_harvey_lab_validate_cli_accepts_partial_imported_report(tmp_path: Path) -> None:
    """The validator CLI should check artifacts without requiring all ten tasks by default."""
    _write_normalized_result_paths(
        tmp_path / "harvey" / ".ingestion" / "runs",
        {"corporate-ma/review-data-room-red-flag-review": 0.64},
    )
    output_dir = tmp_path / "out"
    import_result = CliRunner().invoke(
        app,
        [
            "harvey-lab-import",
            str(tmp_path / "harvey"),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert import_result.exit_code == 0, import_result.output

    validate_result = CliRunner().invoke(
        app,
        [
            "harvey-lab-validate",
            str(output_dir / "harvey-lab-benchmark.json"),
        ],
    )

    assert validate_result.exit_code == 0, validate_result.output
    assert '"status": "valid"' in validate_result.output


def test_harvey_lab_publish_cli_rejects_partial_report(tmp_path: Path) -> None:
    """The publication CLI should refuse reports that have not passed the gate."""
    report = build_harvey_lab_report(
        [_zaxy_result("corporate-ma/review-data-room-red-flag-review", 0.64)]
    )
    written = write_harvey_lab_report(report, tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-publish",
            str(written.json_path),
            "--output",
            str(tmp_path / "public.md"),
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / "public.md").exists()


def test_harvey_lab_index_cli_writes_eventloom_manifest(tmp_path: Path) -> None:
    """The public CLI should build the Zaxy memory index for a Harvey corpus."""
    corpus = tmp_path / "txt"
    corpus.mkdir()
    (corpus / "memo.txt").write_text("Closing consent is required.\n", encoding="utf-8")
    output_dir = tmp_path / "index"

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-index",
            "--normalized-corpus-root",
            str(corpus),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["framework"] == "zaxy"
    assert manifest["event_count"] == 1
    assert manifest["corpus_hash"]
    assert manifest["files"][0]["relative_path"] == "memo.txt"
    assert Path(manifest["eventloom_path"]).exists()
    assert (output_dir / "artifact-summary.json").exists()
    assert (output_dir / "smoke-result.json").exists()


def test_harvey_index_writes_artifact_summary_and_smoke_result(tmp_path: Path) -> None:
    """The Zaxy index should satisfy Harvey's three-file ingestion contract."""
    corpus = tmp_path / "txt"
    corpus.mkdir()
    (corpus / "memo.txt").write_text(
        "The closing checklist requires buyer consent before assignment.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "index"

    manifest = build_harvey_zaxy_memory_index(corpus, output_dir)
    summary = json.loads((output_dir / "artifact-summary.json").read_text(encoding="utf-8"))
    smoke = json.loads((output_dir / "smoke-result.json").read_text(encoding="utf-8"))

    assert summary["framework"] == "zaxy"
    assert summary["corpus_hash"] == manifest["corpus_hash"]
    assert summary["manifest_path"] == str(output_dir / "manifest.json")
    assert summary["artifact_counts"]["eventloom_events"] == 1
    assert summary["normalized_text"]["corpus_root"] == str(corpus.resolve())
    assert smoke["framework"] == "zaxy"
    assert smoke["ok"] is True
    assert smoke["search"]["hits"]
    assert smoke["read"]["source_path"] == "memo.txt"


def test_harvey_normalize_run_builds_contract_from_external_run_artifacts(tmp_path: Path) -> None:
    """Zaxy should export Harvey's normalized-result contract from a real run directory."""
    task_id = "corporate-ma/review-data-room-red-flag-review"
    run_id = "zaxy-corporate-ma__review-data-room-red-flag-review"
    worktree = tmp_path / "harvey"
    run_dir = worktree / "results" / run_id
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "response.md").write_text("Answer\n", encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": "openai-compatible/gpt-5.5",
                "task": task_id,
                "run_id": run_id,
                "reasoning_effort": "low",
                "temperature": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "total_tokens": 1234,
                "wall_clock_seconds": 45.6,
                "memory_search_calls": 2,
                "memory_read_calls": 1,
                "empty_memory_searches": 0,
                "tool_calls_total": 9,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "scores.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task": task_id,
                "score": 32,
                "max_score": 50,
                "judge_model": "openai-compatible/gemini-3-flash-preview",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = worktree / ".ingestion" / "indexes" / "hash" / "zaxy" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "framework": "zaxy",
                "corpus_hash": "external-corpus-hash",
                "normalized_text": {"corpus_root": str(tmp_path / "txt")},
            }
        ),
        encoding="utf-8",
    )

    result = build_harvey_normalized_result_from_run(
        worktree,
        run_id=run_id,
        task_id=task_id,
        manifest_path=manifest_path,
    )

    assert result["schema_version"] == "harvey-memory-ablation-v1"
    assert result["framework"] == "zaxy"
    assert result["task_id"] == task_id
    assert result["corpus_hash"] == "external-corpus-hash"
    assert result["scores"]["final_score"] == 0.64
    assert result["models"]["generator"] == "openai-compatible/gpt-5.5"
    assert result["models"]["judge"] == "openai-compatible/gemini-3-flash-preview"
    assert result["tooling"]["memory_search_calls"] == 2
    assert result["paths"]["answer"] == f"results/{run_id}/output/response.md"


def test_harvey_normalize_run_uses_expected_task_deliverable_when_response_md_absent(tmp_path: Path) -> None:
    """Harvey tasks write named deliverables, so normalization must not assume response.md."""
    task_id = "corporate-ma/draft-acquisition-due-diligence"
    run_id = "zaxy-corporate-ma__draft-acquisition-due-diligence"
    worktree = tmp_path / "harvey"
    task_dir = worktree / "tasks" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "deliverables": {
                    "novabright-diligence-memorandum.docx": "novabright-diligence-memorandum.docx"
                }
            }
        ),
        encoding="utf-8",
    )
    run_dir = worktree / "results" / run_id
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "novabright-diligence-memorandum.docx").write_bytes(b"docx-bytes")
    (run_dir / "config.json").write_text(
        json.dumps({"model": "model-a", "task": task_id, "run_id": run_id, "reasoning_effort": "low", "temperature": 0.0}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"total_tokens": 10, "wall_clock_seconds": 1.2, "memory_search_calls": 1, "memory_read_calls": 1}),
        encoding="utf-8",
    )
    (run_dir / "scores.json").write_text(
        json.dumps({"run_id": run_id, "task": task_id, "score": 0, "max_score": 64, "judge_model": "judge-a"}),
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "tool", "tool_name": "memory_read", "result_preview": "{\"content\": \"memo\"}"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = worktree / ".ingestion" / "indexes" / "hash" / "zaxy" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"framework": "zaxy", "corpus_hash": "hash"}), encoding="utf-8")

    result = build_harvey_normalized_result_from_run(
        worktree,
        run_id=run_id,
        task_id=task_id,
        manifest_path=manifest_path,
    )

    assert result["paths"]["answer"] == (
        f"results/{run_id}/output/novabright-diligence-memorandum.docx"
    )


def test_harvey_normalize_run_prefers_criterion_pass_rate_over_all_pass_score(tmp_path: Path) -> None:
    """Harvey article comparisons use rubric pass rates, not the binary all-pass field."""
    task_id = "corporate-ma/draft-acquisition-due-diligence"
    run_id = "zaxy-corporate-ma__draft-acquisition-due-diligence"
    worktree = tmp_path / "harvey"
    run_dir = worktree / "results" / run_id
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "response.md").write_text("Answer\n", encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"model": "model-a", "task": task_id, "run_id": run_id, "reasoning_effort": "low", "temperature": 0.0}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"total_tokens": 10, "wall_clock_seconds": 1.2, "memory_search_calls": 1, "memory_read_calls": 1}),
        encoding="utf-8",
    )
    (run_dir / "scores.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task": task_id,
                "score": 0.0,
                "max_score": 1.0,
                "criterion_pass_rate": 0.7969,
                "judge_model": "judge-a",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = worktree / ".ingestion" / "indexes" / "hash" / "zaxy" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"framework": "zaxy", "corpus_hash": "hash"}), encoding="utf-8")

    result = build_harvey_normalized_result_from_run(
        worktree,
        run_id=run_id,
        task_id=task_id,
        manifest_path=manifest_path,
    )

    assert result["scores"]["final_score"] == 0.797
    assert result["scores"]["answer_correctness"] == 0.797


def test_harvey_lab_normalize_run_cli_writes_normalized_result(tmp_path: Path) -> None:
    """The CLI should make the external Harvey run contract materializable."""
    task_id = "corporate-ma/review-data-room-red-flag-review"
    run_id = "zaxy-red-flags"
    worktree = tmp_path / "harvey"
    run_dir = worktree / "results" / run_id
    (run_dir / "output").mkdir(parents=True)
    (run_dir / "output" / "response.md").write_text("Answer\n", encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"model": "model-a", "task": task_id, "run_id": run_id, "reasoning_effort": None, "temperature": 0.0}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"total_tokens": 10, "wall_clock_seconds": 1.2, "memory_search_calls": 1, "memory_read_calls": 0}),
        encoding="utf-8",
    )
    (run_dir / "scores.json").write_text(
        json.dumps({"run_id": run_id, "task": task_id, "score": 6, "max_score": 10, "judge_model": "judge-a"}),
        encoding="utf-8",
    )
    (run_dir / "transcript.jsonl").write_text(
        json.dumps({"role": "tool", "tool_name": "memory_search", "result_preview": "{\"hits\": []}"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = worktree / ".ingestion" / "indexes" / "hash" / "zaxy" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"framework": "zaxy", "corpus_hash": "hash"}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-normalize-run",
            "--harvey-worktree",
            str(worktree),
            "--run-id",
            run_id,
            "--task-id",
            task_id,
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0, result.output
    output_path = worktree / ".ingestion" / "runs" / run_id / "normalized-result.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["framework"] == "zaxy"
    assert payload["scores"]["final_score"] == 0.6


def test_harvey_index_manifest_is_compatible_with_harness_scan_fallback(tmp_path: Path) -> None:
    """Harvey's harness expects scan_corpus-like manifests to include corpus metadata."""
    corpus = tmp_path / "txt"
    corpus.mkdir()
    (corpus / "memo.txt").write_text("Closing consent is required.\n", encoding="utf-8")

    manifest = build_harvey_zaxy_memory_index(corpus, tmp_path / "index")

    assert manifest["framework"] == "zaxy"
    assert manifest["corpus_root"] == str(corpus.resolve())
    assert isinstance(manifest["corpus_hash"], str)
    assert manifest["files"] == [
        {
            "relative_path": "memo.txt",
            "sha256": manifest["files"][0]["sha256"],
            "size_bytes": len("Closing consent is required.\n"),
        }
    ]
    assert manifest["file_count"] == 1


def test_export_harvey_adapter_kit_writes_drop_in_memory_module(tmp_path: Path) -> None:
    """The adapter kit should provide Harvey-compatible search/read functions."""
    written = export_harvey_adapter_kit(tmp_path)

    module = (tmp_path / "raw_rg_memory.py").read_text(encoding="utf-8")
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert written["memory_module"] == str(tmp_path / "raw_rg_memory.py")
    assert "def search(manifest, query, limit=5):" in module
    assert "def read(manifest, item_id, context_lines=8):" in module
    assert "zaxy_benchmarks.harvey_lab_benchmark" in module
    assert "HARVEY_MEMORY_MANIFEST" in readme
    assert "zaxy harvey-lab-index" in readme


def test_harvey_adapter_kit_cli_writes_drop_in_files(tmp_path: Path) -> None:
    """The CLI should export files an external Harvey worktree can vendor."""
    result = CliRunner().invoke(
        app,
        [
            "harvey-lab-adapter-kit",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "raw_rg_memory.py" in result.output
    assert (tmp_path / "raw_rg_memory.py").exists()
    assert (tmp_path / "README.md").exists()


def test_harvey_zaxy_memory_adapter_indexes_normalized_text_with_original_citations(tmp_path: Path) -> None:
    """The Harvey adapter should expose Eventloom-backed search/read results."""
    corpus = tmp_path / "txt"
    corpus.mkdir()
    normalized = corpus / "contract.docx.txt"
    normalized.write_text(
        "\n".join(
            [
                "Source-Path: contract.docx",
                "Source-SHA256: original",
                "Extractor: harvey-normalized-text-v1",
                "",
                "The assignment clause requires buyer consent before closing.",
                "A separate schedule mentions ordinary-course notices.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "source-map.json"
    source_map.write_text(
        json.dumps(
            {
                "by_normalized_path": {
                    "contract.docx.txt": {
                        "original_path": "contract.docx",
                        "normalized_path": "contract.docx.txt",
                    }
                },
                "by_original_path": {
                    "contract.docx": {
                        "original_path": "contract.docx",
                        "normalized_path": "contract.docx.txt",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_harvey_zaxy_memory_index(
        corpus,
        tmp_path / "index",
        source_map_path=source_map,
        max_lines=4,
    )
    search = harvey_memory_search(manifest, "assignment consent", limit=3)

    assert manifest["framework"] == "zaxy"
    assert Path(str(manifest["eventloom_path"])).exists()
    assert search["framework"] == "zaxy"
    assert search["hits"][0]["source_path"] == "contract.docx"
    assert search["hits"][0]["id"].startswith("contract.docx:")
    assert "eventloom://" in search["hits"][0]["citation"]

    read = harvey_memory_read(manifest, search["hits"][0]["id"], context_lines=2)

    assert read["source_path"] == "contract.docx"
    assert "assignment clause requires buyer consent" in read["content"]
    assert read["source_identity"]["indexed_corpus"] == "normalized_text"


def test_harvey_memory_read_rejects_unknown_ids(tmp_path: Path) -> None:
    """Harvey memory_read should fail closed for ids not returned by search."""
    corpus = tmp_path / "txt"
    corpus.mkdir()
    (corpus / "memo.txt").write_text("Important closing consent language.\n", encoding="utf-8")
    manifest = build_harvey_zaxy_memory_index(corpus, tmp_path / "index")

    try:
        harvey_memory_read(manifest, "missing.txt:1-2")
    except ValueError as exc:
        assert "memory_read id not found" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected unknown id rejection")


def test_harvey_score_fraction_accepts_supported_score_shapes() -> None:
    """Score normalization should handle Harvey's observed result score variants."""
    assert _score_fraction({"scores": {"criterion_pass_rate": 0.7969}}) == 0.797
    assert _score_fraction({"answer_correctness": 0.621}) == 0.621
    assert _score_fraction({"score": 6, "max_score": 10}) == 0.6


def test_harvey_score_fraction_rejects_non_positive_max_score() -> None:
    """Fractional Harvey score parsing should fail closed on invalid denominators."""
    try:
        _score_fraction({"score": 1, "max_score": 0})
    except ValueError as exc:
        assert "scores.max_score must be positive" in str(exc)
    else:
        raise AssertionError("expected invalid max_score rejection")


def test_harvey_json_object_reader_rejects_invalid_json_and_non_objects(tmp_path: Path) -> None:
    """Report artifact readers should reject malformed or wrong-shaped JSON."""
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")

    try:
        _json_object_from_file(invalid, "score")
    except ValueError as exc:
        assert "invalid Harvey score JSON" in str(exc)
    else:
        raise AssertionError("expected invalid JSON rejection")

    try:
        _json_object_from_file(non_object, "score")
    except ValueError as exc:
        assert "must be an object" in str(exc)
    else:
        raise AssertionError("expected non-object JSON rejection")


def test_harvey_result_path_discovery_handles_file_and_directory_shapes(tmp_path: Path) -> None:
    """Importer path discovery should support Harvey's direct, ingestion, and nested layouts."""
    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    direct = direct_dir / "normalized-result.json"
    direct.write_text("{}", encoding="utf-8")
    ingestion = tmp_path / "worktree" / ".ingestion" / "runs" / "run-a"
    ingestion.mkdir(parents=True)
    ingestion_result = ingestion / "normalized-result.json"
    ingestion_result.write_text("{}", encoding="utf-8")
    nested = tmp_path / "nested" / "run-b"
    nested.mkdir(parents=True)
    nested_result = nested / "normalized-result.json"
    nested_result.write_text("{}", encoding="utf-8")

    assert _harvey_result_paths(direct) == [direct.resolve()]
    assert _harvey_result_paths(direct_dir) == [direct.resolve()]
    assert _harvey_result_paths(tmp_path / "worktree") == [ingestion_result.resolve()]
    assert _harvey_result_paths(tmp_path / "nested") == [nested_result.resolve()]

    try:
        _harvey_result_paths(tmp_path / "wrong-name.json")
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("expected missing path rejection")

    wrong_name = tmp_path / "wrong-name.json"
    wrong_name.write_text("{}", encoding="utf-8")
    try:
        _harvey_result_paths(wrong_name)
    except ValueError as exc:
        assert "must be named normalized-result.json" in str(exc)
    else:
        raise AssertionError("expected wrong filename rejection")
