"""Tests for external LongMemBench validation helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy_benchmarks.longmembench import (
    OFFICIAL_FULL_QUESTION_COUNT,
    _answer_generation_contexts,
    _checkout_answer_candidate,
    _deterministic_temporal_order_answer,
    _explicit_span_date,
    _openai_compatible_answer,
    bootstrap_longmemeval_official_suite,
    build_longmembench_external_run_manifest,
    build_longmembench_readiness,
    build_longmembench_report,
    check_longmembench_gate,
    check_longmemeval_official_suite,
    export_longmembench_adapter_kit,
    generate_longmembench_hypotheses,
    load_official_qa_evidence,
    load_sota_baseline,
    load_validator_evidence,
    load_zaxy_diagnostic_report,
    run_longmemeval_official_eval,
    validate_longmembench_report,
    validate_sota_baseline_currentness,
    validate_validator_evidence_matches_report,
    validator_provenance_from_evidence,
    write_longmembench_external_run_manifest,
    write_longmembench_report,
)


def _write_official_worktree(root: Path, *, questions: int = 2) -> Path:
    (root / "src" / "evaluation").mkdir(parents=True)
    (root / "README.md").write_text("# LongMemEval\n", encoding="utf-8")
    (root / "src" / "evaluation" / "evaluate_qa.py").write_text("print('eval')\n", encoding="utf-8")
    (root / "src" / "evaluation" / "print_qa_metrics.py").write_text("print('metrics')\n", encoding="utf-8")
    (root / "data").mkdir()
    dataset = [
        {
            "question_id": f"q{i}",
            "question": f"Question {i}?",
            "answer": f"Answer {i}",
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        for i in range(questions)
    ]
    dataset_path = root / "data" / "longmemeval_oracle.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=root, check=True, capture_output=True)
    return dataset_path


def _write_generation_dataset(path: Path, *, include_second: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "question_id": "q-memory-1",
            "question_type": "single-session-user",
            "question": "What is the user's project codename?",
            "answer": "Project Kestrel",
            "question_date": "2024-06-01",
            "haystack_session_ids": ["session-1"],
            "haystack_dates": ["2024-05-01"],
            "answer_session_ids": ["session-1"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "Please remember that my project codename is Project Kestrel.",
                        "has_answer": True,
                    },
                    {
                        "role": "assistant",
                        "content": "I will remember that the project codename is Project Kestrel.",
                    },
                ]
            ],
        }
    ]
    if include_second:
        payload.append(
            {
                "question_id": "q-memory-2",
                "question_type": "single-session-user",
                "question": "What city is the launch event in?",
                "answer": "Austin",
                "question_date": "2024-06-01",
                "haystack_session_ids": ["session-2"],
                "haystack_dates": ["2024-05-02"],
                "answer_session_ids": ["session-2"],
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": "The launch event city is Austin.",
                            "has_answer": True,
                        },
                        {
                            "role": "assistant",
                            "content": "I will remember that the launch event is in Austin.",
                        },
                    ]
                ],
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_preference_generation_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "question_id": "q-preference-1",
            "question_type": "single-session-preference",
            "question": "Can you suggest a hotel for my upcoming trip to Miami?",
            "answer": (
                "The user would prefer suggestions of hotels in Miami that offer great views, "
                "possibly of the ocean or the city skyline, and have unique features such as "
                "a rooftop pool or a hot tub on the balcony."
            ),
            "question_date": "2024-06-01",
            "haystack_session_ids": ["session-preference-1"],
            "haystack_dates": ["2024-05-01"],
            "answer_session_ids": ["session-preference-1"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": (
                            "For my Miami trip, I prefer hotels with ocean views, a rooftop pool, "
                            "or a hot tub on the balcony. Basic budget hotels are not what I want."
                        ),
                        "has_answer": True,
                    },
                    {
                        "role": "assistant",
                        "content": "I will keep those Miami hotel preferences in mind.",
                    },
                ]
            ],
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_official_rows(root: Path, *, questions: int = 2, correct: int = 2) -> tuple[Path, Path]:
    hypotheses = root / "zaxy-hypotheses.jsonl"
    eval_log = root / "zaxy-hypotheses.jsonl.log"
    hypotheses.write_text(
        "".join(
            json.dumps({"question_id": f"q{i}", "hypothesis": f"Answer {i}"}) + "\n"
            for i in range(questions)
        ),
        encoding="utf-8",
    )
    eval_log.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": f"q{i}",
                    "hypothesis": f"Answer {i}",
                    "autoeval_label": i < correct,
                }
            )
            + "\n"
            for i in range(questions)
        ),
        encoding="utf-8",
    )
    return hypotheses, eval_log


def _write_diagnostic_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workload": {"sha256": "abc123"},
        "summaries": [
            {
                "backend": "zaxy-checkout",
                "case_count": 500,
                "mean_score": 0.956,
                "mean_answer_recall_at_5": 0.91,
                "mean_citation_coverage": 1.0,
                "mean_recall_at_5": 1.0,
                "mean_recall_at_10": 1.0,
                "latency_ms_p95": 1966.65,
                "mean_approx_tokens": 10192.0,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_sota_baseline(path: Path, *, accuracy: float = 0.96) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date().isoformat()
    payload = {
        "system": "ExternalBest",
        "accuracy": accuracy,
        "metric": "official_longmemeval_task_averaged_accuracy",
        "evidence_url": "https://validation.openmemory.dev/baselines/longmemeval",
        "evidence_date": today,
        "source_type": "public-reproduction",
        "question_count": OFFICIAL_FULL_QUESTION_COUNT,
        "evaluator_model": "gpt-4o",
        "checked_at": today,
        "currentness_url": "https://validation.openmemory.dev/baselines/longmemeval",
        "notes": "fixture",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _validator_provenance() -> dict[str, object]:
    return {
        "validator": {
            "name": "Independent Validator",
            "evidence_url": "https://validation.openmemory.dev/validator/run-1",
            "run_id": "validator-run-1",
            "relation": "independent-third-party",
        },
        "validated_system": {
            "name": "Zaxy",
            "zaxy_commit": "fixture-zaxy-commit",
            "zaxy_version": "",
        },
        "zaxy_commit": "fixture-zaxy-commit",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _write_validator_evidence(
    path: Path,
    *,
    longmemeval_commit: str = "fixture-commit",
    dataset_sha256: str = "fixture-sha",
    question_count: int = OFFICIAL_FULL_QUESTION_COUNT,
    correct_count: int = OFFICIAL_FULL_QUESTION_COUNT,
    evaluated_count: int = OFFICIAL_FULL_QUESTION_COUNT,
    accuracy: float = 1.0,
    zaxy_commit: str = "fixture-zaxy-commit",
    hypotheses_sha256: str = "fixture-hypotheses-sha",
    official_eval_log_sha256: str = "fixture-official-eval-log-sha",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "validator": {
            "name": "Independent Validator",
            "evidence_url": "https://validation.openmemory.dev/validator/run-1",
            "run_id": "validator-run-1",
            "relation": "independent-third-party",
        },
        "validated_system": {
            "name": "Zaxy",
            "zaxy_commit": zaxy_commit,
            "zaxy_version": "",
        },
        "longmemeval": {
            "commit": longmemeval_commit,
            "dataset_sha256": dataset_sha256,
            "question_count": question_count,
        },
        "official_evaluation": {
            "evaluator_model": "gpt-4o",
            "official_eval_command": "python3 evaluate_qa.py gpt-4o zaxy-hypotheses.jsonl data/longmemeval_oracle.json",
            "accuracy": accuracy,
            "correct_count": correct_count,
            "evaluated_count": evaluated_count,
        },
        "artifacts": {
            "dataset_sha256": dataset_sha256,
            "hypotheses_sha256": hypotheses_sha256,
            "official_eval_log_sha256": official_eval_log_sha256,
            "longmembench_report_json": "",
            "longmembench_report_json_sha256": "",
            "longmembench_report_md": "",
            "longmembench_report_md_sha256": "",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_doctor_validates_official_worktree(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree)

    status = check_longmemeval_official_suite(worktree)

    assert status["status"] == "valid"
    assert status["commit"]
    dataset_counts = status["dataset_counts"]
    assert isinstance(dataset_counts, dict)
    assert dataset_counts["longmemeval_oracle.json"] == 2
    assert dataset.exists()


def test_bootstrap_reuses_checkout_and_copies_dataset(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    source_dataset = _write_generation_dataset(tmp_path / "source" / "longmemeval_oracle.json")
    _write_official_worktree(worktree)
    (worktree / "data" / "longmemeval_oracle.json").unlink()

    result = bootstrap_longmemeval_official_suite(
        worktree=worktree,
        dataset_source=source_dataset,
    )

    assert result["status"] == "ready"
    assert result["dataset_count"] == 1
    assert (worktree / "data" / "longmemeval_oracle.json").exists()
    actions = result["actions"]
    assert isinstance(actions, list)
    assert "copy-dataset" in actions


def test_bootstrap_clones_downloads_dataset_and_checks_out_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Official bootstrap should support first-time clone, ref checkout, and dataset download."""
    worktree = tmp_path / "LongMemEval"
    calls: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        if command[:2] == ["git", "clone"]:
            monkeypatch.setattr("zaxy_benchmarks.longmembench.subprocess.run", real_run)
            _write_official_worktree(worktree)
            monkeypatch.setattr("zaxy_benchmarks.longmembench.subprocess.run", fake_run)
            (worktree / "data" / "longmemeval_oracle.json").unlink()
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    def fake_get(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        return httpx.Response(
            200,
            content=json.dumps(
                [
                    {
                        "question_id": "q-download",
                        "question": "Downloaded?",
                        "answer": "Yes",
                        "haystack_sessions": [],
                        "answer_session_ids": [],
                    }
                ]
            ).encode(),
            request=httpx.Request("GET", "https://example.test/oracle.json"),
        )

    monkeypatch.setattr("zaxy_benchmarks.longmembench.subprocess.run", fake_run)
    monkeypatch.setattr("zaxy_benchmarks.longmembench.httpx.get", fake_get)

    result = bootstrap_longmemeval_official_suite(
        worktree=worktree,
        repo_url="https://example.test/LongMemEval.git",
        ref="abc123",
        dataset_url="https://example.test/oracle.json",
    )

    assert result["status"] == "ready"
    assert result["dataset_count"] == 1
    assert result["actions"] == ["git-clone", "git-checkout:abc123", "download-dataset"]
    assert calls[0][:3] == ["git", "clone", "https://example.test/LongMemEval.git"]
    assert ["git", "-C", str(worktree), "checkout", "abc123"] in calls


def test_bootstrap_rejects_existing_non_git_worktree(tmp_path: Path) -> None:
    """Bootstrap should not overwrite an arbitrary existing directory."""
    worktree = tmp_path / "LongMemEval"
    worktree.mkdir()

    with pytest.raises(ValueError, match="not a git checkout"):
        bootstrap_longmemeval_official_suite(worktree=worktree)


def test_official_evaluator_runner_invokes_fixture_script(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=2)
    hypotheses, _eval_log = _write_official_rows(tmp_path, questions=2, correct=2)
    script = worktree / "src" / "evaluation" / "evaluate_qa.py"
    script.write_text(
        "\n".join(
            [
                "import json, os, pathlib, sys",
                "hyp=pathlib.Path(sys.argv[2])",
                "rows=[json.loads(line) for line in hyp.read_text().splitlines() if line.strip()]",
                "out=hyp.with_name(hyp.name + '.eval-results-fixture-judge')",
                "out.write_text(''.join(json.dumps({**row, 'autoeval_label': True}) + '\\n' for row in rows))",
                "print('key=' + os.environ.get('OPENAI_API_KEY', ''))",
                "print('fixture official eval complete')",
            ]
        ),
        encoding="utf-8",
    )

    result = run_longmemeval_official_eval(
        worktree=worktree,
        hypotheses_path=hypotheses,
        dataset_path=dataset,
        evaluator_model="fixture-judge",
        require_api_key=True,
        api_key_present=True,
        api_key="fixture-key",
    )

    assert result.status == "complete"
    assert result.returncode == 0
    assert Path(result.eval_log_path).exists()
    assert result.eval_log_path.endswith(".eval-results-fixture-judge")
    assert "key=fixture-key" in result.stdout
    assert "fixture official eval complete" in result.stdout


def test_official_qa_evidence_scores_evaluator_log(tmp_path: Path) -> None:
    dataset = _write_official_worktree(tmp_path / "LongMemEval", questions=2)
    hypotheses, eval_log = _write_official_rows(tmp_path, questions=2, correct=1)

    evidence = load_official_qa_evidence(
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        evaluator_model="gpt-4o",
        official_eval_command="python3 evaluate_qa.py gpt-4o zaxy-hypotheses.jsonl data/longmemeval_oracle.json",
    )

    assert evidence.evaluated_count == 2
    assert evidence.correct_count == 1
    assert evidence.accuracy == 0.5


def test_report_gate_rejects_diagnostics_only_for_official_sota(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree)
    diagnostic = _write_diagnostic_report(tmp_path / "diag" / "live-benchmark.json")

    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        diagnostic_report_path=diagnostic,
    )
    validation = validate_longmembench_report(report, require_official_full=True)
    gate = check_longmembench_gate(report, require_official_sota_candidate=True)

    assert validation["status"] == "invalid"
    assert gate["status"] == "failed"
    failures = gate["failures"]
    assert isinstance(failures, list)
    assert "official QA" in " ".join(str(item) for item in failures)


def test_report_gate_accepts_full_official_evidence(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    diagnostic = _write_diagnostic_report(tmp_path / "diag" / "live-benchmark.json")
    baseline = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.99)

    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        diagnostic_report_path=diagnostic,
        sota_baseline_path=baseline,
        evaluator_model="gpt-4o",
        result_provenance=_validator_provenance(),
    )
    gate = check_longmembench_gate(
        report,
        require_official_sota_candidate=True,
        min_accuracy=0.99,
    )

    assert report.status == "complete"
    assert gate["status"] == "passed"
    assert gate["official_sota_candidate"] is True
    strict_gate = check_longmembench_gate(report, require_official_sota=True)
    assert strict_gate["status"] == "failed"
    assert strict_gate["official_sota"] is False
    assert strict_gate["external_validator"] == _validator_provenance()["validator"]
    strict_failures = strict_gate["failures"]
    assert isinstance(strict_failures, list)
    assert "official SOTA requires imported validator_evidence JSON" in strict_failures
    assert "official SOTA requires cross-checked validator evidence" in strict_failures


def test_strict_sota_gate_requires_external_validator(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    baseline = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.99)

    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        sota_baseline_path=baseline,
    )
    gate = check_longmembench_gate(report, require_official_sota=True)

    assert gate["status"] == "failed"
    failures = gate["failures"]
    assert isinstance(failures, list)
    assert "external validator provenance" in " ".join(str(item) for item in failures)


def test_strict_sota_gate_rejects_blank_validator_evidence(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    baseline = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.99)
    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        sota_baseline_path=baseline,
        result_provenance={"validator": {"name": "", "evidence_url": "", "run_id": "", "relation": ""}},
    )

    gate = check_longmembench_gate(report, require_official_sota=True)

    assert gate["status"] == "failed"
    failures = gate["failures"]
    assert isinstance(failures, list)
    assert "external validator name is required" in failures
    assert "external validator evidence_url must be absolute http(s)" in failures


def test_strict_sota_gate_rejects_placeholder_validator_url(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    baseline = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.99)
    provenance = _validator_provenance()
    validator_payload = provenance.get("validator")
    assert isinstance(validator_payload, dict)
    validator = dict(validator_payload)
    validator["evidence_url"] = "https://example.com/validator/run-1"
    provenance["validator"] = validator
    provenance["validator_evidence"] = str(tmp_path / "validator-evidence.json")
    provenance["validator_evidence_verified"] = True
    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        sota_baseline_path=baseline,
        result_provenance=provenance,
    )

    gate = check_longmembench_gate(report, require_official_sota=True)

    assert gate["status"] == "failed"
    failures = gate["failures"]
    assert isinstance(failures, list)
    assert "external validator evidence_url must not use placeholder example domains" in failures


def test_validator_evidence_builds_provenance_and_rejects_conflicts(tmp_path: Path) -> None:
    evidence = load_validator_evidence(_write_validator_evidence(tmp_path / "validator-evidence.json"))

    provenance = validator_provenance_from_evidence(evidence)

    assert provenance == _validator_provenance()
    with pytest.raises(ValueError, match="conflicts"):
        validator_provenance_from_evidence(evidence, validator_name="Different Validator")


def test_validator_evidence_must_match_imported_official_report(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        evaluator_model="gpt-4o",
        official_eval_command="python3 evaluate_qa.py gpt-4o zaxy-hypotheses.jsonl data/longmemeval_oracle.json",
        result_provenance=_validator_provenance(),
    )
    matching = load_validator_evidence(
        _write_validator_evidence(
            tmp_path / "matching-validator-evidence.json",
            longmemeval_commit=_git_commit(worktree),
            dataset_sha256=_sha256(dataset),
            hypotheses_sha256=_sha256(hypotheses),
            official_eval_log_sha256=_sha256(eval_log),
        )
    )
    mismatched = load_validator_evidence(
        _write_validator_evidence(
            tmp_path / "mismatched-validator-evidence.json",
            longmemeval_commit=_git_commit(worktree),
            dataset_sha256=_sha256(dataset),
            hypotheses_sha256=_sha256(hypotheses),
            official_eval_log_sha256=_sha256(eval_log),
            correct_count=OFFICIAL_FULL_QUESTION_COUNT - 1,
            accuracy=0.998,
        )
    )

    assert validate_validator_evidence_matches_report(matching, report) == []
    failures = validate_validator_evidence_matches_report(mismatched, report)
    assert "validator evidence official_evaluation.correct_count does not match imported report" in failures
    assert "validator evidence official_evaluation.accuracy does not match imported report" in failures


def test_validator_evidence_must_bind_validated_zaxy_commit(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        evaluator_model="gpt-4o",
        official_eval_command="python3 evaluate_qa.py gpt-4o zaxy-hypotheses.jsonl data/longmemeval_oracle.json",
        result_provenance=_validator_provenance(),
    )
    mismatched = load_validator_evidence(
        _write_validator_evidence(
            tmp_path / "mismatched-zaxy-commit-validator-evidence.json",
            longmemeval_commit=_git_commit(worktree),
            dataset_sha256=_sha256(dataset),
            hypotheses_sha256=_sha256(hypotheses),
            official_eval_log_sha256=_sha256(eval_log),
            zaxy_commit="different-zaxy-commit",
        )
    )

    failures = validate_validator_evidence_matches_report(mismatched, report)

    assert "validator evidence validated_system.zaxy_commit does not match imported report" in failures


def test_cli_validator_evidence_writes_completed_cross_checked_record(tmp_path: Path) -> None:
    runner = CliRunner()
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=2)
    hypotheses, eval_log = _write_official_rows(tmp_path, questions=2, correct=1)
    output = tmp_path / "validator-evidence.json"
    official_command = "python3 evaluate_qa.py gpt-4o zaxy-hypotheses.jsonl data/longmemeval_oracle.json"

    result = runner.invoke(
        app,
        [
            "longmembench-validator-evidence",
            "--longmemeval-worktree",
            str(worktree),
            "--dataset",
            str(dataset),
            "--hypotheses",
            str(hypotheses),
            "--official-eval-log",
            str(eval_log),
            "--output",
            str(output),
            "--evaluator-model",
            "gpt-4o",
            "--official-eval-command",
            official_command,
            "--validator-name",
            "Independent Validator",
            "--validator-evidence-url",
            "https://validation.openmemory.dev/validator/run-1",
            "--validator-run-id",
            "validator-run-1",
            "--validator-relation",
            "independent-third-party",
            "--zaxy-worktree",
            str(Path.cwd()),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["longmemeval"]["commit"] == _git_commit(worktree)
    assert payload["longmemeval"]["dataset_sha256"] == _sha256(dataset)
    assert payload["artifacts"]["dataset_sha256"] == _sha256(dataset)
    assert payload["artifacts"]["hypotheses_sha256"] == _sha256(hypotheses)
    assert payload["artifacts"]["official_eval_log_sha256"] == _sha256(eval_log)
    assert payload["official_evaluation"]["evaluated_count"] == 2
    assert payload["official_evaluation"]["correct_count"] == 1
    assert payload["official_evaluation"]["accuracy"] == 0.5


def test_cli_import_accepts_validator_evidence_json(tmp_path: Path) -> None:
    runner = CliRunner()
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    baseline = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.99)
    validator_evidence = _write_validator_evidence(
        tmp_path / "validator-evidence.json",
        longmemeval_commit=_git_commit(worktree),
        dataset_sha256=_sha256(dataset),
        hypotheses_sha256=_sha256(hypotheses),
        official_eval_log_sha256=_sha256(eval_log),
    )
    output_dir = tmp_path / "report"

    result = runner.invoke(
        app,
        [
            "longmembench-import",
            "--longmemeval-worktree",
            str(worktree),
            "--dataset",
            str(dataset),
            "--hypotheses",
            str(hypotheses),
            "--official-eval-log",
            str(eval_log),
            "--sota-baseline",
            str(baseline),
            "--validator-evidence",
            str(validator_evidence),
            "--output-dir",
            str(output_dir),
        ],
    )
    gate = runner.invoke(
        app,
        [
            "longmembench-gate",
            str(output_dir / "longmembench-report.json"),
            "--require-official-sota",
        ],
    )

    assert result.exit_code == 0, result.output
    assert gate.exit_code == 0, gate.output
    report_payload = json.loads((output_dir / "longmembench-report.json").read_text(encoding="utf-8"))
    assert report_payload["result_provenance"]["validator"] == _validator_provenance()["validator"]
    assert report_payload["result_provenance"]["validator_evidence_verified"] is True
    assert report_payload["official_qa"]["evaluator_model"] == "gpt-4o"


def test_cli_official_eval_to_strict_gate_chain_with_fixture_validator(tmp_path: Path) -> None:
    runner = CliRunner()
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, _unused_eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    evaluator = worktree / "src" / "evaluation" / "evaluate_qa.py"
    evaluator.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                "hypotheses=pathlib.Path(sys.argv[2])",
                "rows=[json.loads(line) for line in hypotheses.read_text().splitlines() if line.strip()]",
                "output=hypotheses.with_name(hypotheses.name + '.eval-results-' + sys.argv[1])",
                "output.write_text(''.join(json.dumps({**row, 'autoeval_label': True}) + '\\n' for row in rows))",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "external"
    eval_log = output_dir / "zaxy-hypotheses.jsonl.eval-results-gpt-4o"
    official_command = f"python3 evaluate_qa.py gpt-4o {hypotheses} {dataset}"
    diagnostic = _write_diagnostic_report(output_dir / "diagnostic" / "live-benchmark.json")
    baseline = _write_sota_baseline(output_dir / "sota-baseline.json", accuracy=0.99)
    validator_evidence = output_dir / "validator-evidence.json"
    hypothesis_report = output_dir / "zaxy-hypotheses-report.json"
    hypothesis_report.write_text(
        json.dumps(
            {
                "schema_version": "zaxy.longmembench-hypotheses.v1",
                "question_count": OFFICIAL_FULL_QUESTION_COUNT,
                "output_path": str(hypotheses),
            }
        ),
        encoding="utf-8",
    )

    official_eval = runner.invoke(
        app,
        [
            "longmembench-evaluate-official",
            "--longmemeval-worktree",
            str(worktree),
            "--hypotheses",
            str(hypotheses),
            "--dataset",
            str(dataset),
            "--evaluator-model",
            "gpt-4o",
            "--output-log",
            str(eval_log),
            "--run-report",
            str(output_dir / "official-eval-run.json"),
            "--api-key",
            "fixture-key",
        ],
    )
    write_evidence = runner.invoke(
        app,
        [
            "longmembench-validator-evidence",
            "--longmemeval-worktree",
            str(worktree),
            "--dataset",
            str(dataset),
            "--hypotheses",
            str(hypotheses),
            "--official-eval-log",
            str(eval_log),
            "--output",
            str(validator_evidence),
            "--evaluator-model",
            "gpt-4o",
            "--official-eval-command",
            official_command,
            "--print-metrics-command",
            f"python3 print_qa_metrics.py {eval_log} {dataset}",
            "--validator-name",
            "Independent Validator",
            "--validator-evidence-url",
            "https://validation.openmemory.dev/validator/run-1",
            "--validator-run-id",
            "validator-run-1",
            "--validator-relation",
            "independent-third-party",
            "--zaxy-worktree",
            str(Path.cwd()),
        ],
    )
    imported = runner.invoke(
        app,
        [
            "longmembench-import",
            "--longmemeval-worktree",
            str(worktree),
            "--dataset",
            str(dataset),
            "--hypotheses",
            str(hypotheses),
            "--official-eval-log",
            str(eval_log),
            "--diagnostic-report",
            str(diagnostic),
            "--sota-baseline",
            str(baseline),
            "--validator-evidence",
            str(validator_evidence),
            "--output-dir",
            str(output_dir),
        ],
    )
    gate = runner.invoke(
        app,
        [
            "longmembench-gate",
            str(output_dir / "longmembench-report.json"),
            "--require-official-sota",
        ],
    )
    audit = runner.invoke(
        app,
        [
            "longmembench-audit",
            "--longmemeval-worktree",
            str(worktree),
            "--dataset",
            str(dataset),
            "--hypotheses",
            str(hypotheses),
            "--official-eval-log",
            str(eval_log),
            "--diagnostic-report",
            str(diagnostic),
            "--sota-baseline",
            str(baseline),
            "--validator-evidence",
            str(validator_evidence),
            "--report",
            str(output_dir / "longmembench-report.json"),
            "--hypothesis-report",
            str(hypothesis_report),
            "--official-eval-run-report",
            str(output_dir / "official-eval-run.json"),
            "--output",
            str(output_dir / "longmembench-audit.json"),
        ],
    )

    assert official_eval.exit_code == 0, official_eval.output
    assert write_evidence.exit_code == 0, write_evidence.output
    assert imported.exit_code == 0, imported.output
    assert gate.exit_code == 0, gate.output
    assert audit.exit_code == 0, audit.output
    audit_payload = json.loads((output_dir / "longmembench-audit.json").read_text(encoding="utf-8"))
    assert audit_payload["schema_version"] == "zaxy.longmembench-audit.v1"
    assert audit_payload["generated_at"]
    assert audit_payload["status"] == "passed"
    assert audit_payload["artifacts"]["dataset"]["sha256"] == _sha256(dataset)
    assert audit_payload["artifacts"]["hypotheses"]["sha256"] == _sha256(hypotheses)
    assert audit_payload["artifacts"]["official_eval_log"]["sha256"] == _sha256(eval_log)
    assert audit_payload["artifacts"]["validator_evidence"]["sha256"] == _sha256(validator_evidence)
    report_path = output_dir / "longmembench-report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["official_qa"]["evaluated_count"] == OFFICIAL_FULL_QUESTION_COUNT
    assert report_payload["official_qa"]["accuracy"] == 1.0
    assert report_payload["result_provenance"]["validator_evidence"] == str(validator_evidence.resolve())
    publish = runner.invoke(
        app,
        [
            "longmembench-publish",
            str(report_path),
            "--audit",
            str(output_dir / "longmembench-audit.json"),
            "--output",
            str(output_dir / "publishable-statistics.md"),
        ],
    )
    assert publish.exit_code == 0, publish.output
    assert (output_dir / "publishable-statistics.md").exists()
    old_audit_payload = dict(audit_payload)
    old_audit_payload.pop("schema_version")
    old_audit_path = output_dir / "old-longmembench-audit.json"
    old_audit_path.write_text(json.dumps(old_audit_payload), encoding="utf-8")
    old_audit_publish = runner.invoke(
        app,
        [
            "longmembench-publish",
            str(report_path),
            "--audit",
            str(old_audit_path),
        ],
    )
    assert old_audit_publish.exit_code != 0
    assert "unsupported LongMemBench audit schema_version" in old_audit_publish.output
    report_payload["result_provenance"]["hypotheses"] = str(output_dir / "different-hypotheses.jsonl")
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    stale_publish = runner.invoke(
        app,
        [
            "longmembench-publish",
            str(report_path),
            "--audit",
            str(output_dir / "longmembench-audit.json"),
        ],
    )
    assert stale_publish.exit_code != 0
    assert "audit longmembench_report.sha256 does not match report file" in stale_publish.output
    mismatched_audit = runner.invoke(
        app,
        [
            "longmembench-audit",
            "--longmemeval-worktree",
            str(worktree),
            "--dataset",
            str(dataset),
            "--hypotheses",
            str(hypotheses),
            "--official-eval-log",
            str(eval_log),
            "--diagnostic-report",
            str(diagnostic),
            "--sota-baseline",
            str(baseline),
            "--validator-evidence",
            str(validator_evidence),
            "--report",
            str(report_path),
            "--hypothesis-report",
            str(hypothesis_report),
            "--official-eval-run-report",
            str(output_dir / "official-eval-run.json"),
        ],
    )
    assert mismatched_audit.exit_code == 1
    assert "report provenance.hypotheses does not match audited artifact path" in mismatched_audit.output
    report_payload["result_provenance"]["hypotheses"] = str(hypotheses.resolve())
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    validator_payload = json.loads(validator_evidence.read_text(encoding="utf-8"))
    validator_payload["official_evaluation"]["hypotheses_path"] = str(output_dir / "stale-hypotheses.jsonl")
    validator_evidence.write_text(json.dumps(validator_payload), encoding="utf-8")
    stale_validator_audit = runner.invoke(
        app,
        [
            "longmembench-audit",
            "--longmemeval-worktree",
            str(worktree),
            "--dataset",
            str(dataset),
            "--hypotheses",
            str(hypotheses),
            "--official-eval-log",
            str(eval_log),
            "--diagnostic-report",
            str(diagnostic),
            "--sota-baseline",
            str(baseline),
            "--validator-evidence",
            str(validator_evidence),
            "--report",
            str(report_path),
            "--hypothesis-report",
            str(hypothesis_report),
            "--official-eval-run-report",
            str(output_dir / "official-eval-run.json"),
        ],
    )
    assert stale_validator_audit.exit_code == 1
    assert (
        "validator evidence official_evaluation.hypotheses_path does not match audited artifact path"
        in stale_validator_audit.output
    )
    validator_payload["official_evaluation"]["hypotheses_path"] = str(hypotheses)
    validator_payload["artifacts"]["hypotheses_sha256"] = "0" * 64
    validator_evidence.write_text(json.dumps(validator_payload), encoding="utf-8")
    stale_hash_audit = runner.invoke(
        app,
        [
            "longmembench-audit",
            "--longmemeval-worktree",
            str(worktree),
            "--dataset",
            str(dataset),
            "--hypotheses",
            str(hypotheses),
            "--official-eval-log",
            str(eval_log),
            "--diagnostic-report",
            str(diagnostic),
            "--sota-baseline",
            str(baseline),
            "--validator-evidence",
            str(validator_evidence),
            "--report",
            str(report_path),
            "--hypothesis-report",
            str(hypothesis_report),
            "--official-eval-run-report",
            str(output_dir / "official-eval-run.json"),
        ],
    )
    assert stale_hash_audit.exit_code == 1
    assert (
        "validator evidence artifacts.hypotheses_sha256 does not match imported report"
        in stale_hash_audit.output
    )


def test_report_write_and_cli_validate(tmp_path: Path) -> None:
    runner = CliRunner()
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
    )
    written = write_longmembench_report(report, tmp_path / "report")

    result = runner.invoke(
        app,
        ["longmembench-validate", str(written.json_path), "--require-official-full"],
    )
    gate = runner.invoke(
        app,
        ["longmembench-gate", str(written.json_path), "--require-official-sota-candidate"],
    )

    assert result.exit_code == 0, result.output
    assert gate.exit_code == 0, gate.output


def test_strict_sota_gate_requires_baseline_and_beating_score(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=490,
    )
    report_without_baseline = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
    )
    assert check_longmembench_gate(
        report_without_baseline,
        require_official_sota=True,
    )["status"] == "failed"

    baseline = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.99)
    report_with_higher_baseline = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        sota_baseline_path=baseline,
    )
    gate = check_longmembench_gate(report_with_higher_baseline, require_official_sota=True)

    assert gate["status"] == "failed"
    failures = gate["failures"]
    assert isinstance(failures, list)
    assert "does not beat" in " ".join(str(item) for item in failures)


def test_readiness_reports_missing_launch_blockers(tmp_path: Path) -> None:
    dataset = _write_generation_dataset(tmp_path / "longmemeval_oracle.json")

    readiness = build_longmembench_readiness(
        longmemeval_worktree=tmp_path / "missing-LongMemEval",
        dataset_path=dataset,
        answer_mode="openai-compatible",
        api_key_present=False,
        require_sota_baseline=True,
    )

    assert readiness["status"] == "not_ready"
    blockers = readiness["blockers"]
    assert isinstance(blockers, list)
    assert any("worktree" in str(item) for item in blockers)
    assert any("OPENAI_API_KEY" in str(item) for item in blockers)
    assert any("SOTA baseline" in str(item) for item in blockers)


def test_readiness_accepts_full_external_run_inputs_and_warns_on_extractive_mode(
    tmp_path: Path,
) -> None:
    """Readiness should pass when official, diagnostic, and baseline artifacts are present."""
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    diagnostic = _write_diagnostic_report(tmp_path / "live-benchmark.json")
    baseline = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.95)

    readiness = build_longmembench_readiness(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        diagnostic_report_path=diagnostic,
        sota_baseline_path=baseline,
        answer_mode="extractive",
        api_key_present=False,
    )

    assert readiness["status"] == "ready"
    assert readiness["blockers"] == []
    assert any(
        str(warning).startswith("extractive mode is suitable for smoke tests")
        for warning in readiness["warnings"]
    )
    assert readiness["dataset"]["question_count"] == OFFICIAL_FULL_QUESTION_COUNT
    assert readiness["hypotheses"]["count"] == OFFICIAL_FULL_QUESTION_COUNT
    assert readiness["official_eval_log"]["count"] == OFFICIAL_FULL_QUESTION_COUNT
    assert readiness["diagnostic_report"]["status"] == "valid"
    assert readiness["sota_baseline"]["status"] == "valid"


def test_load_sota_baseline_validates_contract(tmp_path: Path) -> None:
    baseline_path = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.965)

    baseline = load_sota_baseline(baseline_path)

    assert baseline.system == "ExternalBest"
    assert baseline.accuracy == 0.965
    assert baseline.metric == "official_longmemeval_task_averaged_accuracy"
    assert baseline.question_count == OFFICIAL_FULL_QUESTION_COUNT


def test_sota_baseline_rejects_retrieval_metric(tmp_path: Path) -> None:
    baseline_path = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.966)
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["metric"] = "recall_at_5"
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="official LongMemEval QA accuracy"):
        load_sota_baseline(baseline_path)


def test_sota_baseline_rejects_placeholder_evidence_url(tmp_path: Path) -> None:
    baseline_path = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.966)
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["evidence_url"] = "https://example.com/reviewable-artifact"
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="placeholder example domains"):
        load_sota_baseline(baseline_path)


def test_strict_sota_baseline_currentness_rejects_stale_evidence(tmp_path: Path) -> None:
    baseline_path = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.90)
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["checked_at"] = "2026-01-01"
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    baseline = load_sota_baseline(baseline_path)

    failures = validate_sota_baseline_currentness(
        baseline,
        reference_date=date(2026, 3, 2),
        max_age_days=30,
    )

    assert any("stale" in item for item in failures)


def test_strict_sota_gate_rejects_stale_baseline_even_when_score_beats_it(tmp_path: Path) -> None:
    worktree = tmp_path / "LongMemEval"
    dataset = _write_official_worktree(worktree, questions=OFFICIAL_FULL_QUESTION_COUNT)
    hypotheses, eval_log = _write_official_rows(
        tmp_path,
        questions=OFFICIAL_FULL_QUESTION_COUNT,
        correct=OFFICIAL_FULL_QUESTION_COUNT,
    )
    stale_checked_at = (datetime.now(UTC).date() - timedelta(days=60)).isoformat()
    baseline = _write_sota_baseline(tmp_path / "sota-baseline.json", accuracy=0.90)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["checked_at"] = stale_checked_at
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    report = build_longmembench_report(
        longmemeval_worktree=worktree,
        dataset_path=dataset,
        hypotheses_path=hypotheses,
        official_eval_log_path=eval_log,
        sota_baseline_path=baseline,
        result_provenance={
            **_validator_provenance(),
            "validator_evidence": str(tmp_path / "validator-evidence.json"),
            "validator_evidence_verified": True,
        },
    )

    gate = check_longmembench_gate(report, require_official_sota=True)

    assert gate["status"] == "failed"
    failures = gate["failures"]
    assert isinstance(failures, list)
    assert any("stale" in str(item) for item in failures)


def test_adapter_kit_and_plan_are_written(tmp_path: Path) -> None:
    kit = export_longmembench_adapter_kit(tmp_path / "kit")
    manifest = build_longmembench_external_run_manifest(output_dir=str(tmp_path / "run"))
    written = write_longmembench_external_run_manifest(manifest, tmp_path / "run")

    assert Path(kit["readme"]).exists()
    assert Path(kit["runner"]).exists()
    assert Path(kit["validator_checklist"]).exists()
    assert Path(kit["validator_evidence_template"]).exists()
    checklist_text = Path(kit["validator_checklist"]).read_text(encoding="utf-8")
    evidence_template = json.loads(Path(kit["validator_evidence_template"]).read_text(encoding="utf-8"))
    assert "--require-official-sota" in checklist_text
    assert "Zaxy commit" in checklist_text
    assert evidence_template["validator"]["relation"] == "independent-third-party"
    assert evidence_template["validated_system"]["zaxy_commit"] == ""
    assert evidence_template["longmemeval"]["question_count"] == OFFICIAL_FULL_QUESTION_COUNT
    assert written.json_path.exists()
    assert written.markdown_path.exists()
    assert written.script_path.exists()
    plan_markdown = written.markdown_path.read_text(encoding="utf-8")
    assert "longmembench-import" in plan_markdown
    assert "longmembench-publish" in plan_markdown
    assert "--audit" in plan_markdown
    script_text = written.script_path.read_text(encoding="utf-8")
    assert "run_step" in script_text
    assert "OPENAI_API_KEY is required" in script_text
    assert "RUN_OFFICIAL_EVAL=${RUN_OFFICIAL_EVAL:-1}" in script_text
    assert "RUN_DIAGNOSTIC=${RUN_DIAGNOSTIC:-${RUN_OFFICIAL_EVAL}}" in script_text
    assert f"RUN_OUTPUT_DIR={tmp_path / 'run'}/smoke" in script_text
    assert '"${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl' in script_text
    assert "VALIDATOR_NAME=${VALIDATOR_NAME:-}" in script_text
    assert "VALIDATOR_NAME, VALIDATOR_EVIDENCE_URL, VALIDATOR_RUN_ID, and VALIDATOR_RELATION" in script_text
    assert "longmembench-validator-evidence" in script_text
    assert "longmembench-audit" in script_text
    assert "longmembench-publish" in script_text
    assert '"${RUN_OUTPUT_DIR}"/longmembench-audit.json' in script_text
    assert '"${RUN_OUTPUT_DIR}"/publishable-statistics.md' in script_text
    assert '"${RUN_OUTPUT_DIR}"/validator-evidence.json' in script_text
    assert '--validator-evidence "${RUN_OUTPUT_DIR}"/validator-evidence.json' in script_text
    assert '--validator-name "${VALIDATOR_NAME}"' in script_text
    assert "Independent Validator" not in script_text
    assert "skipped because RUN_DIAGNOSTIC=0" in script_text
    assert "skipped because RUN_OFFICIAL_EVAL=0" in script_text
    assert "[audit] skipped because RUN_OFFICIAL_EVAL=0" in script_text
    assert "[publish] skipped because RUN_OFFICIAL_EVAL=0" in script_text
    assert "QUESTIONS=${QUESTIONS:-500}" in script_text
    assert '--dataset "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json' in script_text
    assert "manifest carrier" not in script_text


def test_generate_hypotheses_writes_official_jsonl(tmp_path: Path) -> None:
    dataset = _write_generation_dataset(tmp_path / "longmemeval_oracle.json")
    output = tmp_path / "zaxy-hypotheses.jsonl"
    report_path = tmp_path / "zaxy-hypotheses-report.json"

    report = asyncio.run(
        generate_longmembench_hypotheses(
            dataset_path=dataset,
            output_path=output,
            report_path=report_path,
            questions=1,
            answer_mode="extractive",
            embedding_provider="hash",
        )
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report.question_count == 1
    assert report_path.exists()
    assert rows == [
        {
            "question_id": "q-memory-1",
            "hypothesis": rows[0]["hypothesis"],
        }
    ]
    assert "Project Kestrel" in rows[0]["hypothesis"]
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    hypothesis_payload = report_payload["hypotheses"][0]
    assert hypothesis_payload["answer_session_ids"] == ["session-1"]
    assert hypothesis_payload["answer_session_hits_top5"] == ["session-1"]
    assert hypothesis_payload["expected_answer_hit_top5"] is True
    assert hypothesis_payload["context_audit"]
    assert hypothesis_payload["context_audit"][0]["rank"] == 1
    assert "session-1" in hypothesis_payload["context_audit"][0]["session_ids"]


def test_generate_hypotheses_resume_skips_existing_rows(tmp_path: Path) -> None:
    dataset = _write_generation_dataset(tmp_path / "longmemeval_oracle.json", include_second=True)
    output = tmp_path / "zaxy-hypotheses.jsonl"
    report_path = tmp_path / "zaxy-hypotheses-report.json"
    output.write_text(
        json.dumps({"question_id": "q-memory-1", "hypothesis": "Project Kestrel"}) + "\n",
        encoding="utf-8",
    )

    report = asyncio.run(
        generate_longmembench_hypotheses(
            dataset_path=dataset,
            output_path=output,
            report_path=report_path,
            questions=2,
            answer_mode="extractive",
            embedding_provider="hash",
            resume=True,
        )
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report.question_count == 2
    assert [row["question_id"] for row in rows] == ["q-memory-1", "q-memory-2"]
    assert "Austin" in rows[1]["hypothesis"]


def test_generate_openai_hypotheses_uses_answer_ready_candidate_before_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context filtering should not strip answer-ready checkout candidates before generation."""
    calls: list[object] = []

    def fake_post(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("provider should not be called when checkout has an answer-ready candidate")

    monkeypatch.setattr("zaxy_benchmarks.longmembench.httpx.post", fake_post)
    dataset = _write_preference_generation_dataset(tmp_path / "longmemeval_oracle.json")
    output = tmp_path / "zaxy-hypotheses.jsonl"
    report_path = tmp_path / "zaxy-hypotheses-report.json"

    report = asyncio.run(
        generate_longmembench_hypotheses(
            dataset_path=dataset,
            output_path=output,
            report_path=report_path,
            questions=1,
            answer_mode="openai-compatible",
            api_key="test-key",
            embedding_provider="hash",
            filter_answer_contexts=True,
        )
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert calls == []
    assert report.question_count == 1
    assert rows[0]["question_id"] == "q-preference-1"
    # Honest candidate reflects the cited evidence ("hotels with ocean views, a
    # rooftop pool, ...") — never the gold `answer` field's "hotels in Miami".
    assert "hotels with ocean views" in rows[0]["hypothesis"]
    assert "ocean" in rows[0]["hypothesis"]


def test_checkout_answer_candidate_extracts_human_answer() -> None:
    contexts = [
        "\n".join(
            [
                "memory_checkout_compact=true",
                "checkout_answer_candidate=true",
                "candidate_type=temporal_order",
                "answer=Data Analysis using Python",
                "support_source_ids=answer_1",
            ]
        )
    ]

    assert _checkout_answer_candidate(
        "Which event did I attend first, the workshop or the webinar?",
        contexts,
    ) == "Data Analysis using Python"


def test_checkout_answer_candidate_rejects_event_header_fragment() -> None:
    contexts = [
        "\n".join(
            [
                "memory_checkout_compact=true",
                "checkout_answer_candidate=true",
                "candidate_type=temporal_order",
                "answer=# Event 31 citation=eventloom://benchmark/events/31#abc 07:17",
            ]
        )
    ]

    assert _checkout_answer_candidate(
        "Which event happened first, the purchase or the malfunction?",
        contexts,
    ) is None


def test_checkout_answer_candidate_rejects_mismatched_numeric_answer() -> None:
    contexts = [
        "\n".join(
            [
                "memory_checkout_compact=true",
                "checkout_answer_candidate=true",
                "candidate_type=temporal_order",
                "answer=Turbocharged Tuesdays",
            ]
        )
    ]

    assert _checkout_answer_candidate(
        "How many days before the 'Rack Fest' did I participate in the 'Turbocharged Tuesdays' event?",
        contexts,
    ) is None


def test_checkout_answer_candidate_uses_absence_guidance() -> None:
    contexts = [
        "\n".join(
            [
                "checkout_synthesis=true",
                "zaxy_absence_check=true",
                "synthesis_mode=absence_check",
                "query=What is the name of my hamster?",
                "not_mentioned_candidate=hamster",
                (
                    "answer_guidance=The information provided is not enough. "
                    "You did not mention this information. You did not mention hamster. "
                    "You mentioned cited evidence below, but not hamster."
                ),
                "- source_id=pet-1 citation=eventloom://agent/events/1#aa snippet=I adopted a cat named Luna.",
            ]
        )
    ]

    assert _checkout_answer_candidate("What is the name of my hamster?", contexts) == (
        "The information provided is not enough. "
        "You did not mention this information. You did not mention hamster. "
        "You mentioned cited evidence below, but not hamster."
    )


def test_openai_compatible_answer_uses_absence_guidance_without_provider(monkeypatch) -> None:
    """Cited absence bundles should be answer-ready for missing-evidence questions."""
    calls: list[object] = []

    def fake_post(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("provider should not be called for answer-ready absence evidence")

    monkeypatch.setattr("zaxy_benchmarks.longmembench.httpx.post", fake_post)

    answer = _openai_compatible_answer(
        question="Which task did I complete first, fixing the fence or buying a greenhouse?",
        contexts=[
            "\n".join(
                [
                    "checkout_synthesis=true",
                    "zaxy_absence_check=true",
                    "synthesis_mode=absence_check",
                    "query=Which task did I complete first, fixing the fence or buying a greenhouse?",
                    "not_mentioned_candidate=buying greenhouse",
                    (
                        "answer_guidance=The information provided is not enough. "
                        "You did not mention this information. You did not mention buying greenhouse. "
                        "You mentioned cited evidence below, but not buying greenhouse."
                    ),
                    "- source_id=task-1 citation=eventloom://agent/events/1#aa snippet=I finished fixing the fence.",
                ]
            )
        ],
        model="gpt-4o-mini",
        base_url="https://api.example.test/v1",
        api_key="test",
    )

    assert calls == []
    assert answer.startswith("The information provided is not enough.")
    assert "buying greenhouse" in answer


def test_answer_generation_contexts_exclude_checkout_diagnostics() -> None:
    contexts = [
        "memory_checkout_compact=true\ncheckout_answer_candidate=true\nanswer=One weeks",
        (
            "checkout_synthesis=true\nzaxy_synthesis_bundle=true\n"
            "temporal_order_answer=One weeks"
        ),
        (
            "checkout_fact=true\ncitation=eventloom://benchmark/events/1#abc\n"
            "source_lane=verbatim\n"
            "I started tomatoes indoors on February 20th."
        ),
    ]

    selected = _answer_generation_contexts(contexts)

    assert selected == [contexts[2]]


def test_openai_compatible_answer_uses_preference_candidate_without_provider(monkeypatch) -> None:
    """Preference-profile synthesis should use cited answer-ready candidates before LLM calls."""
    calls: list[object] = []

    def fake_post(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("provider should not be called for answer-ready preference evidence")

    monkeypatch.setattr("zaxy_benchmarks.longmembench.httpx.post", fake_post)

    answer = _openai_compatible_answer(
        question="What hotel recommendations would I prefer?",
        contexts=[
            (
                "source_id=hotel-1 citation=eventloom://agent/events/1#aa "
                "longmemeval_session_id=answer-hotel "
                "user: I prefer Miami hotel options with ocean views, a rooftop pool, "
                "or a hot tub on the balcony."
            )
        ],
        model="gpt-4o-mini",
        base_url="https://api.example.test/v1",
        api_key="test",
    )

    assert calls == []
    # Honest candidate quotes the cited preference ("Miami hotel options with
    # ocean views, a rooftop pool, ...") rather than a gold-answer string.
    assert "Miami hotel options" in answer
    assert "ocean" in answer
    assert "rooftop pool" in answer


def test_openai_compatible_answer_retries_provider_then_returns_message(monkeypatch) -> None:
    """OpenAI-compatible answer generation should retry transient provider failures."""
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    responses = [
        httpx.Response(500, headers={"retry-after": "0.25"}, json={"error": {"message": "busy"}}, request=request),
        httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Project Kestrel"}},
                ]
            },
            request=request,
        ),
    ]

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        calls.append({"args": args, "kwargs": kwargs})
        return responses.pop(0)

    monkeypatch.setattr("zaxy_benchmarks.longmembench.httpx.post", fake_post)
    monkeypatch.setattr("zaxy_benchmarks.longmembench.time.sleep", lambda delay: sleeps.append(delay))

    answer = _openai_compatible_answer(
        question="What is the project codename?",
        contexts=["checkout_fact=true citation=eventloom://agent/events/1#aa user: Project Kestrel."],
        model="gpt-4o-mini",
        base_url="https://api.example.test/v1/",
        api_key="test-key",
        max_retries=1,
    )

    assert answer == "Project Kestrel"
    assert len(calls) == 2
    assert calls[0]["args"] == ("https://api.example.test/v1/chat/completions",)
    assert calls[0]["kwargs"]["headers"] == {"Authorization": "Bearer test-key"}
    assert sleeps == [0.25]
    payload = calls[0]["kwargs"]["json"]
    assert payload["model"] == "gpt-4o-mini"
    assert "Project Kestrel" in payload["messages"][1]["content"]


def test_openai_compatible_answer_5xx_backs_off_when_retry_after_absent_or_invalid(monkeypatch) -> None:
    """5xx backoff: honor a numeric Retry-After, else fall back to linear backoff.

    Covers the non-numeric Retry-After (``except ValueError``) and missing-header
    (``else``) branches of the server-error retry path.
    """
    sleeps: list[float] = []
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    responses = [
        # Non-numeric Retry-After -> linear backoff (10.0 * attempt=1).
        httpx.Response(503, headers={"retry-after": "soon"}, json={"error": {}}, request=request),
        # No Retry-After header -> linear backoff (10.0 * attempt=2).
        httpx.Response(500, json={"error": {}}, request=request),
        httpx.Response(200, json={"choices": [{"message": {"content": "Project Kestrel"}}]}, request=request),
    ]

    monkeypatch.setattr("zaxy_benchmarks.longmembench.httpx.post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr("zaxy_benchmarks.longmembench.time.sleep", lambda delay: sleeps.append(delay))

    answer = _openai_compatible_answer(
        question="What is the project codename?",
        contexts=["checkout_fact=true citation=eventloom://agent/events/1#aa user: Project Kestrel."],
        model="gpt-4o-mini",
        base_url="https://api.example.test/v1/",
        api_key="test-key",
        max_retries=2,
    )

    assert answer == "Project Kestrel"
    assert sleeps == [10.0, 20.0]


def test_openai_compatible_answer_rejects_malformed_provider_responses(monkeypatch) -> None:
    """Malformed provider responses should fail before producing unsupported hypotheses."""
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    responses = [
        httpx.Response(200, json={}, request=request),
        httpx.Response(200, json={"choices": ["not-an-object"]}, request=request),
        httpx.Response(200, json={"choices": [{"message": "not-an-object"}]}, request=request),
        httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}, request=request),
    ]

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        return responses.pop(0)

    monkeypatch.setattr("zaxy_benchmarks.longmembench.httpx.post", fake_post)

    expected_messages = [
        "missing choices",
        "choice must be an object",
        "choice missing message",
        "message content is empty",
    ]
    for expected in expected_messages:
        with pytest.raises(ValueError, match=expected):
            _openai_compatible_answer(
                question="What is the project codename?",
                contexts=["checkout_fact=true citation=eventloom://agent/events/1#aa user: Project Kestrel."],
                model="gpt-4o-mini",
                base_url="https://api.example.test/v1",
                api_key="test-key",
                max_retries=0,
            )


def test_openai_compatible_answer_stops_retries_on_insufficient_quota(monkeypatch) -> None:
    """Insufficient quota should not burn retry budget or hide the provider error."""
    calls = 0
    sleeps: list[float] = []

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        nonlocal calls
        del args, kwargs
        calls += 1
        return httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.example.test/v1/chat/completions"),
            json={"error": {"code": "insufficient_quota"}},
        )

    monkeypatch.setattr("zaxy_benchmarks.longmembench.httpx.post", fake_post)
    monkeypatch.setattr("zaxy_benchmarks.longmembench.time.sleep", lambda delay: sleeps.append(delay))

    with pytest.raises(httpx.HTTPStatusError):
        _openai_compatible_answer(
            question="What is the project codename?",
            contexts=["checkout_fact=true citation=eventloom://agent/events/1#aa user: Project Kestrel."],
            model="gpt-4o-mini",
            base_url="https://api.example.test/v1",
            api_key="test-key",
            max_retries=3,
        )

    assert calls == 1
    assert sleeps == []


def test_deterministic_temporal_order_answer_uses_explicit_dates() -> None:
    answer = _deterministic_temporal_order_answer(
        "Which seeds were started first, the tomatoes or the marigolds?",
        [
            (
                "checkout_fact=true citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_date=2023/03/10 (Fri) "
                "user: I started tomatoes indoors on February 20th."
            ),
            (
                "checkout_fact=true citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_date=2023/03/10 (Fri) "
                "user: I started marigolds on March 3rd."
            ),
        ],
    )

    assert answer == "Tomatoes"


def test_deterministic_temporal_order_answer_uses_relative_dates() -> None:
    answer = _deterministic_temporal_order_answer(
        "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
        [
            (
                "checkout_fact=true citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_date=2023/05/28 (Sun) "
                "user: I attended an Effective Time Management workshop last week."
            ),
            (
                "checkout_fact=true citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_date=2023/05/28 (Sun) "
                "user: I participated in a Data Analysis using Python webinar two months ago."
            ),
        ],
    )

    assert answer == "Data Analysis using Python"


def test_deterministic_temporal_order_answer_uses_neighboring_date_sentence() -> None:
    answer = _deterministic_temporal_order_answer(
        "Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?",
        [
            (
                "checkout_fact=true citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_date=2023/03/15 (Wed) "
                "user: I recently got a new Samsung Galaxy S22 from Best Buy on February 20th."
            ),
            (
                "checkout_fact=true citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_date=2023/03/15 (Wed) "
                "user: My new laptop, Dell XPS 13, finally arrived. "
                "I pre-ordered the laptop on January 28th, and it arrived on February 25th."
            ),
        ],
    )

    assert answer == "Samsung Galaxy S22"


def test_longmembench_date_parser_ignores_invalid_calendar_dates() -> None:
    """Noisy date-like text should not abort full external hypothesis generation."""
    assert _explicit_span_date("I started the model on February 31st.", default_year=2024) is None
    assert _explicit_span_date("I started the model on 9/31.", default_year=2024) is None


def test_load_zaxy_diagnostic_report(tmp_path: Path) -> None:
    path = _write_diagnostic_report(tmp_path / "live-benchmark.json")

    diagnostic = load_zaxy_diagnostic_report(path)

    assert diagnostic.backend == "zaxy-checkout"
    assert diagnostic.recall_at_5 == 1.0
    assert diagnostic.citation_coverage == 1.0
