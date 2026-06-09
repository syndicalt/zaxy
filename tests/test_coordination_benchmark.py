"""Tests for the CoordinationBench harness."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner
from zaxy_benchmarks.coordination_benchmark import (
    COORDINATION_WORKLOAD_VERSION,
    CoordinationBenchMetrics,
    CoordinationCompetitorResultAudit,
    build_coordination_workload,
    coordination_baseline_metrics,
    coordination_competitor_adapter_disclosures,
    coordination_competitor_claim_gate,
    coordination_competitor_runner_manifest_templates,
    coordination_purpose_synthesis_gate,
    export_coordination_benchmark_adapter_kit,
    flat_eventlog_baseline_metrics,
    load_coordination_workload,
    run_coordination_benchmark,
    run_coordination_competitor_adapter,
    run_coordination_competitor_runner,
    score_coordination_brief,
    validate_coordination_competitor_result,
    validate_coordination_competitor_runner_manifest,
)

from zaxy.__main__ import app


def test_coordination_benchmark_workload_is_frozen_and_replayable(tmp_path: Path) -> None:
    """The deterministic workload should preserve mission/worker isolation and evidence."""
    workload_path = tmp_path / "coordination-workload.json"

    workload = build_coordination_workload(workload_path, missions=1, workers=3)

    payload = json.loads(workload_path.read_text(encoding="utf-8"))
    assert payload["version"] == COORDINATION_WORKLOAD_VERSION
    assert payload["fingerprint"] == workload.fingerprint
    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert case["mission_id"] == "coordination-case-1"
    assert len(case["workers"]) == 3
    assert case["gold"]["expected_accepted_claims"] == {"auth.failure.cause": "expired-jwks-cache"}
    assert "finding-api-stale-flag" in case["gold"]["expected_stale_findings"]
    stale = next(
        finding
        for worker in case["workers"]
        for finding in worker["findings"]
        if finding["finding_id"] == "finding-api-stale-flag"
    )
    assert stale["evidence"][0]["stale"] is True
    assert stale["evidence"][0]["superseded_by"] == "decision:jwks-cache"
    assert any(finding["evidence"] for worker in case["workers"] for finding in worker["findings"])
    assert workload.fingerprint == build_coordination_workload(tmp_path / "again.json", missions=1, workers=3).fingerprint


@pytest.mark.parametrize("workers", [2, 11])
def test_coordination_benchmark_rejects_invalid_worker_counts(tmp_path: Path, workers: int) -> None:
    """CoordinationBench should enforce the roadmap's 3-to-10 worker range."""
    with pytest.raises(ValueError, match="workers"):
        build_coordination_workload(tmp_path / "bad.json", missions=1, workers=workers)


def test_coordination_workload_rejects_unsupported_mission_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one mission"):
        build_coordination_workload(tmp_path / "bad.json", missions=2, workers=3)


def test_load_coordination_workload_rejects_invalid_json_shape_and_fingerprint(tmp_path: Path) -> None:
    invalid_shape = tmp_path / "invalid-shape.json"
    invalid_shape.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_coordination_workload(invalid_shape)

    workload_path = tmp_path / "coordination-workload.json"
    workload = build_coordination_workload(workload_path, missions=1, workers=3)
    payload = workload.to_dict()
    payload["fingerprint"] = "wrong"
    workload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_coordination_workload(workload_path)


def test_coordination_scorer_measures_accepted_conflict_stale_duplicate_and_evidence(tmp_path: Path) -> None:
    """Scoring should use exact IDs and claim keys rather than fuzzy text matching."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)
    case = report.cases[0]

    metrics = score_coordination_brief(case.brief, case.gold)

    assert metrics.accepted_finding_precision == 1.0
    assert metrics.accepted_finding_recall == 1.0
    assert metrics.conflict_recall == 1.0
    assert metrics.stale_claim_rejection == 1.0
    assert [finding.finding_id for finding in case.brief.stale_findings] == ["finding-api-stale-flag"]
    assert metrics.duplicate_consolidation == 1.0
    assert metrics.evidence_coverage == 1.0
    assert metrics.parent_checkout_answerability == 1.0
    assert metrics.citation_coverage == 1.0
    assert metrics.eventloom_replayable is True
    assert case.metrics.purpose_feedback_coverage == 1.0


def test_coordination_benchmark_runs_frozen_workload_file(tmp_path: Path) -> None:
    """A report run can replay an externally frozen workload without regenerating it."""
    source_path = tmp_path / "source-workload.json"
    workload = build_coordination_workload(source_path, missions=1, workers=3)
    payload = workload.to_dict()
    payload["version"] = "coordination-real-test"
    body = {"version": payload["version"], "cases": payload["cases"]}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = run_coordination_benchmark(tmp_path / "out", workload_path=source_path, missions=1, workers=10)

    copied = load_coordination_workload(tmp_path / "out" / "coordination-workload.json")
    assert report.version == "coordination-real-test"
    assert report.workload_fingerprint == payload["fingerprint"]
    assert copied.fingerprint == payload["fingerprint"]
    assert report.metrics.accepted_finding_recall == 1.0


def test_coordinate_purpose_synthesis_gate_requires_proof_backed_coordinate_state(
    tmp_path: Path,
) -> None:
    """Coordinate claims should require synthesis proof, purpose feedback, citations, and no leakage."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)

    gate = coordination_purpose_synthesis_gate(report)

    assert gate.status == "passed"
    assert gate.required_metrics == {
        "accepted_state_synthesis_quality": 1.0,
        "non_authoritative_leakage": 1.0,
        "purpose_feedback_coverage": 1.0,
        "citation_coverage": 1.0,
        "parent_checkout_answerability": 1.0,
        "eventloom_replayable": True,
    }
    assert gate.blocked_metrics == {}
    payload = report.to_dict()
    assert payload["coordinate_purpose_synthesis_gate"]["status"] == "passed"


def test_coordinate_purpose_synthesis_gate_blocks_weak_reports(tmp_path: Path) -> None:
    """A metrics table should not imply Coordinate proof claims when purpose evidence is missing."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)
    weak_report = replace(
        report,
        metrics=replace(
            report.metrics,
            accepted_state_synthesis_quality=0.0,
            purpose_feedback_coverage=0.0,
            non_authoritative_leakage=0.0,
        ),
    )

    gate = coordination_purpose_synthesis_gate(weak_report)

    assert gate.status == "blocked"
    assert gate.blocked_metrics["accepted_state_synthesis_quality"] == "expected >= 1.0, got 0.0"
    assert gate.blocked_metrics["purpose_feedback_coverage"] == "expected >= 1.0, got 0.0"
    assert gate.blocked_metrics["non_authoritative_leakage"] == "expected >= 1.0, got 0.0"


def test_flat_eventlog_baseline_gets_contaminated_by_worker_findings(tmp_path: Path) -> None:
    """A naive all-findings baseline should accept stale and conflicting worker-local claims."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)
    case = report.cases[0]

    baseline = flat_eventlog_baseline_metrics(case.workload_case)

    assert baseline.accepted_finding_precision < 1.0
    assert baseline.stale_claim_rejection < 1.0
    assert baseline.duplicate_consolidation < 1.0


def test_coordination_benchmark_reports_same_harness_local_baselines(tmp_path: Path) -> None:
    """CoordinationBench should expose roadmap baselines before external competitor claims."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)

    assert set(report.baselines) == {
        "flat_transcript",
        "markdown_notes",
        "bm25_worker_logs",
    }
    assert report.baselines["flat_transcript"].stale_claim_rejection < report.metrics.stale_claim_rejection
    assert report.baselines["markdown_notes"].accepted_finding_precision < report.metrics.accepted_finding_precision
    assert report.baselines["markdown_notes"].citation_coverage < report.metrics.citation_coverage
    assert report.baselines["markdown_notes"].duplicate_consolidation < report.metrics.duplicate_consolidation
    assert report.baselines["bm25_worker_logs"].conflict_recall == 0.0
    assert report.baselines["bm25_worker_logs"].citation_coverage == 0.0
    assert report.baselines["bm25_worker_logs"].duplicate_consolidation == 0.0
    assert report.baselines["bm25_worker_logs"].parent_checkout_answerability < report.metrics.parent_checkout_answerability
    payload = report.to_dict()
    assert payload["baselines"]["flat_transcript"]["accepted_finding_precision"] < 1.0


@pytest.mark.parametrize(
    "baseline_name",
    [
        "flat_transcript",
        "markdown_notes",
        "bm25_worker_logs",
    ],
)
def test_coordination_baseline_metrics_are_strict_and_reproducible(
    tmp_path: Path,
    baseline_name: str,
) -> None:
    """Every local baseline should run through the same exact metric contract."""
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    case = workload.cases[0]

    first = coordination_baseline_metrics(case, baseline_name)
    second = coordination_baseline_metrics(case, baseline_name)

    assert first.to_dict() == second.to_dict()
    json.dumps(first.to_dict(), allow_nan=False)
    assert first.returned_tokens > 0
    assert first.injected_tokens > 0


def test_coordination_baseline_metrics_reject_unknown_baseline(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)

    with pytest.raises(ValueError, match="Unknown CoordinationBench baseline"):
        coordination_baseline_metrics(workload.cases[0], "memory-magic")


def test_coordination_benchmark_reports_competitor_adapter_disclosures(tmp_path: Path) -> None:
    """External competitors should be disclosed without unpinned fake scores."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)

    disclosures = report.competitor_adapters
    assert set(disclosures) == {"mem0", "agent_memory", "activegraph", "quarq", "hybi"}
    assert all(disclosure.status == "not_run" for disclosure in disclosures.values())
    assert all(disclosure.metrics is None for disclosure in disclosures.values())
    assert disclosures["activegraph"].claim_status == "disclosure_only"
    assert "pinned adapter" in disclosures["mem0"].blockers[0]
    payload = report.to_dict()
    assert payload["competitor_adapters"]["activegraph"]["status"] == "not_run"
    assert payload["competitor_adapters"]["activegraph"]["metrics"] is None
    assert payload["competitor_adapters"]["activegraph"]["claim_status"] == "disclosure_only"
    assert payload["competitor_claim_gate"]["status"] == "blocked"
    assert "quarq" in payload["competitor_claim_gate"]["blocked_adapters"]
    assert "hybi" in payload["competitor_claim_gate"]["blocked_adapters"]


def test_coordination_competitor_claim_gate_requires_scored_quarq_and_hybi(tmp_path: Path) -> None:
    """Public Quarq/Hybi claims should fail until real same-harness rows exist."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)

    gate = coordination_competitor_claim_gate(report)

    assert gate.status == "blocked"
    assert gate.completed_adapters == ()
    assert set(gate.blocked_adapters) == {"quarq", "hybi"}
    assert "same-harness public claims require completed locally scored results" in gate.blocked_adapters["quarq"]
    payload = report.to_dict()
    assert payload["competitor_adapters"]["quarq"]["status"] == "not_run"
    assert payload["competitor_adapters"]["hybi"]["status"] == "not_run"


def test_coordination_competitor_claim_gate_reports_completed_row_defects(tmp_path: Path) -> None:
    """Completed competitor rows still need local metrics and result-audit provenance."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)
    manifest = json.loads(_competitor_runner_manifest(tmp_path, report.workload_fingerprint).read_text(encoding="utf-8"))
    audit = CoordinationCompetitorResultAudit(
        result_fingerprint="result-1",
        generated_at_utc="2026-06-02T00:00:00Z",
        case_count=len(report.cases),
        manifest=manifest,
    )
    complete = replace(
        report.competitor_adapters["quarq"],
        status="completed",
        claim_status="same_harness",
        metrics=report.metrics,
        result_audit=audit,
    )
    missing_metrics = replace(complete, metrics=None)
    missing_audit = replace(complete, result_audit=None)
    missing_manifest_field = replace(
        complete,
        result_audit=replace(complete.result_audit, manifest={**manifest, "source_ref": ""}),
    )
    missing_fingerprint = replace(
        complete,
        result_audit=replace(complete.result_audit, result_fingerprint=""),
    )
    broken_report = replace(
        report,
        competitor_adapters={
            "quarq": missing_metrics,
            "hybi": missing_audit,
            "mem0": missing_manifest_field,
            "agent_memory": missing_fingerprint,
        },
    )

    gate = coordination_competitor_claim_gate(
        broken_report,
        required_adapters=("missing", "quarq", "hybi", "mem0", "agent_memory"),
    )

    assert gate.status == "blocked"
    assert gate.blocked_adapters["missing"] == "adapter row is missing from the report"
    assert gate.blocked_adapters["quarq"] == "completed adapter row has no locally scored metrics"
    assert gate.blocked_adapters["hybi"] == "completed adapter row has no result audit provenance"
    assert gate.blocked_adapters["mem0"] == "result audit manifest is missing: source_ref"
    assert gate.blocked_adapters["agent_memory"] == "result audit is missing result_fingerprint"


def test_coordination_competitor_adapter_disclosures_are_strict_and_reproducible() -> None:
    """Adapter disclosures should be stable until a real same-harness adapter exists."""
    first = coordination_competitor_adapter_disclosures()
    second = coordination_competitor_adapter_disclosures()

    assert first == second
    encoded = json.dumps({name: item.to_dict() for name, item in first.items()}, allow_nan=False)
    assert "ActiveGraph" in encoded
    assert "Quarq" in encoded
    assert "Semantic Reach / HyperBinder / Hybi" in encoded
    assert "same-harness" in encoded


def test_coordination_competitor_runner_manifest_templates_are_fingerprint_bound(tmp_path: Path) -> None:
    """Published manifest templates should give adapter authors a pinned starting point."""
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)

    templates = coordination_competitor_runner_manifest_templates(workload)

    assert set(templates) == {"mem0", "agent_memory", "activegraph", "quarq", "hybi"}
    for name, manifest in templates.items():
        assert manifest["name"] == name
        assert manifest["adapter_contract"] == "coordinationbench-v1"
        assert manifest["workload_fingerprint"] == workload.fingerprint
        assert manifest["template"] is True
        assert manifest["dataset_contract"].startswith("coordinationbench-v1")
        assert manifest["result_export_schema"] == "schemas/result.schema.json; runner must write --output <path>"
        assert "coordination-workload.json" in manifest["workload_file"]
        assert manifest["result_file"].endswith(f"{name}-coordination-result.json")
    for name in {"mem0", "agent_memory", "activegraph"}:
        assert templates[name]["run_command"] == ["__REPLACE_WITH_PINNED_RUNNER_ARGV__"]
    assert templates["quarq"]["source_ref"] == "git:b68386048795765d46c87bef5bd88ecfb1301337"
    assert templates["quarq"]["run_command"][:4] == [
        "python",
        "-m",
        "zaxy.resources.coordinationbench.unsupported_runner",
        "--adapter",
    ]
    assert templates["hybi"]["source_ref"].startswith("pypi:hybi==0.1.1 sha256:")
    assert templates["hybi"]["capability_status"] == "unsupported_runner_required"


def test_write_coordination_benchmark_report_publishes_manifest_templates(tmp_path: Path) -> None:
    """A benchmark run should write a manifest-pack directory for external adapters."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)

    manifest_dir = tmp_path / "competitor-runner-manifests"
    manifest_paths = sorted(manifest_dir.glob("*.runner-manifest.template.json"))

    assert report.workload_fingerprint
    assert [path.name for path in manifest_paths] == [
        "activegraph.runner-manifest.template.json",
        "agent_memory.runner-manifest.template.json",
        "hybi.runner-manifest.template.json",
        "mem0.runner-manifest.template.json",
        "quarq.runner-manifest.template.json",
    ]
    payload = json.loads((manifest_dir / "mem0.runner-manifest.template.json").read_text(encoding="utf-8"))
    assert payload["template"] is True
    assert payload["workload_fingerprint"] == report.workload_fingerprint
    assert payload["run_command"] == ["__REPLACE_WITH_PINNED_RUNNER_ARGV__"]
    quarq = json.loads((manifest_dir / "quarq.runner-manifest.template.json").read_text(encoding="utf-8"))
    assert quarq["source_ref"] == "git:b68386048795765d46c87bef5bd88ecfb1301337"
    assert quarq["capability_status"] == "unsupported_runner_required"


def test_coordinationbench_contract_resources_are_packaged() -> None:
    """The adapter contract kit should ship installable schema/template assets."""
    root = importlib.resources.files("zaxy.resources.coordinationbench")

    assert (root / "README.md").is_file()
    assert (root / "schemas" / "runner-manifest.schema.json").is_file()
    assert (root / "schemas" / "result.schema.json").is_file()
    assert (root / "templates" / "mem0.runner-manifest.template.json").is_file()
    assert (root / "templates" / "agent_memory.runner-manifest.template.json").is_file()
    assert (root / "templates" / "activegraph.runner-manifest.template.json").is_file()
    assert (root / "templates" / "quarq.runner-manifest.template.json").is_file()
    assert (root / "templates" / "hybi.runner-manifest.template.json").is_file()


def test_export_coordination_benchmark_adapter_kit_writes_schemas_workload_and_templates(tmp_path: Path) -> None:
    """The public kit export should be enough for adapter authors to build against."""
    exported = export_coordination_benchmark_adapter_kit(tmp_path / "kit", missions=1, workers=3)

    output_dir = tmp_path / "kit"
    assert exported["workload_fingerprint"]
    assert (output_dir / "coordination-workload.json").exists()
    assert (output_dir / "schemas" / "runner-manifest.schema.json").exists()
    assert (output_dir / "schemas" / "result.schema.json").exists()
    assert (output_dir / "templates" / "mem0.runner-manifest.template.json").exists()
    assert (output_dir / "templates" / "quarq.runner-manifest.template.json").exists()
    assert (output_dir / "templates" / "hybi.runner-manifest.template.json").exists()
    assert (output_dir / "templates" / "mem0-result.template.json").exists()
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "coordinationbench-v1" in readme
    assert "--competitor-runner mem0=" in readme


def test_validate_coordination_competitor_runner_manifest_returns_audit_payload(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest_path = _competitor_runner_manifest(tmp_path, workload.fingerprint)

    payload = validate_coordination_competitor_runner_manifest("mem0", workload, manifest_path)

    assert payload["name"] == "mem0"
    assert payload["workload_fingerprint"] == workload.fingerprint
    assert payload["run_command"] == [sys.executable, str(tmp_path / "mem0_runner.py")]


def test_validate_coordination_competitor_result_returns_local_score_and_audit(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(_competitor_result_payload(workload.fingerprint)), encoding="utf-8")

    payload = validate_coordination_competitor_result("mem0", workload, result_path)

    assert payload["metrics"]["conflict_recall"] == 1.0
    assert payload["audit"]["manifest"]["source_ref"] == "abc123"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "wrong", "name does not match"),
        ("adapter_contract", "wrong", "adapter_contract"),
        ("source_ref", "", "source_ref must be nonempty"),
        ("run_command", [], "run_command"),
        ("run_command", ["__REPLACE_WITH_PINNED_RUNNER_ARGV__"], "placeholder"),
    ],
)
def test_validate_coordination_competitor_runner_manifest_rejects_malformed_contracts(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest_path = _competitor_runner_manifest(tmp_path, workload.fingerprint)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_coordination_competitor_runner_manifest("mem0", workload, manifest_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"name": "wrong"}), "name does not match"),
        (lambda payload: payload.update({"adapter_contract": "wrong"}), "adapter_contract"),
        (lambda payload: payload.update({"manifest": None}), "manifest must be an object"),
        (lambda payload: payload["manifest"].update({"source_url": ""}), "source_url must be nonempty"),
        (lambda payload: payload.update({"cases": {}}), "cases must be an array"),
        (lambda payload: payload["cases"].append("not-an-object"), "case output must be an object"),
    ],
)
def test_validate_coordination_competitor_result_rejects_malformed_contracts(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    result_path = tmp_path / "mem0-result.json"
    payload = _competitor_result_payload(workload.fingerprint)
    mutate(payload)
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_coordination_competitor_result("mem0", workload, result_path)


def test_competitor_runner_rejects_unedited_template_manifest(tmp_path: Path) -> None:
    """Placeholder manifests should never execute or produce same-harness claims."""
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest = coordination_competitor_runner_manifest_templates(workload)["mem0"]
    manifest_path = tmp_path / "mem0-template.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="template manifest"):
        run_coordination_competitor_runner(
            "mem0",
            workload,
            manifest_path=manifest_path,
            output_dir=tmp_path / "runner-output",
        )


def test_competitor_runner_rejects_placeholder_run_command_even_without_template_flag(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest = coordination_competitor_runner_manifest_templates(workload)["mem0"]
    manifest["template"] = False
    manifest_path = tmp_path / "mem0-placeholder.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="placeholder run_command"):
        run_coordination_competitor_runner(
            "mem0",
            workload,
            manifest_path=manifest_path,
            output_dir=tmp_path / "runner-output",
        )


def test_unregistered_coordination_competitor_adapter_cannot_emit_scores(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)

    disclosure = run_coordination_competitor_adapter("mem0", workload)

    assert disclosure.status == "not_run"
    assert disclosure.metrics is None
    assert disclosure.claim_status == "disclosure_only"


def test_registered_coordination_competitor_adapter_runs_same_metric_contract(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)

    def adapter(case) -> CoordinationBenchMetrics:  # type: ignore[no-untyped-def]
        return coordination_baseline_metrics(case, "markdown_notes")

    disclosure = run_coordination_competitor_adapter("mem0", workload, adapter=adapter)

    assert disclosure.status == "completed"
    assert disclosure.claim_status == "same_harness"
    assert disclosure.metrics is not None
    assert disclosure.metrics.to_dict() == coordination_baseline_metrics(workload.cases[0], "markdown_notes").to_dict()


def _competitor_result_payload(workload_fingerprint: str, *, include_metrics: bool = False) -> dict[str, object]:
    case_payload: dict[str, object] = {
        "case_id": "coordination-case-1",
        "accepted_findings": [
            {
                "finding_id": "finding-api-jwks",
                "worker_id": "worker-api",
                "summary": "JWKS cache expiry is the accepted auth failure cause.",
                "claim_key": "auth.failure.cause",
                "claim_value": "expired-jwks-cache",
                "evidence": [{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
                "source_event_seq": 1,
                "source_event_hash": "a" * 64,
            }
        ],
        "conflicts": [
            {
                "claim_key": "auth.failure.cause",
                "finding_ids": ["finding-api-jwks", "finding-ui-refresh"],
            }
        ],
        "stale_findings": [{"finding_id": "finding-api-stale-flag"}],
        "returned_text": "accepted cause: expired-jwks-cache",
        "injected_text": "accepted cause: expired-jwks-cache",
        "latency_ms": 12.5,
    }
    if include_metrics:
        case_payload["metrics"] = {"accepted_finding_precision": 0.0}
    return {
        "name": "mem0",
        "adapter_contract": "coordinationbench-v1",
        "workload_fingerprint": workload_fingerprint,
        "generated_at_utc": "2026-05-27T00:00:00Z",
        "manifest": {
            "name": "mem0",
            "display_name": "Mem0",
            "adapter_contract": "coordinationbench-v1",
            "adapter_version": "0.1.0",
            "install_command": "uv tool install mem0==0.1.0",
            "run_command": "mem0-coordinationbench",
            "source_url": "https://example.test/mem0-adapter",
            "source_ref": "abc123",
            "dataset_contract": "coordinationbench-v1 workload JSON; runner receives --workload <path>",
            "result_export_schema": "schemas/result.schema.json; runner must write --output <path>",
        },
        "cases": [case_payload],
    }


def test_competitor_result_path_is_scored_locally_without_trusting_supplied_metrics(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(
        json.dumps(_competitor_result_payload(workload.fingerprint, include_metrics=True)),
        encoding="utf-8",
    )

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.status == "completed"
    assert disclosure.claim_status == "same_harness"
    assert disclosure.metrics is not None
    assert disclosure.metrics.accepted_finding_precision == 1.0
    assert disclosure.metrics.conflict_recall == 1.0
    assert disclosure.metrics.stale_claim_rejection == 1.0
    assert disclosure.metrics.returned_tokens > 0
    assert disclosure.result_audit is not None
    assert disclosure.result_audit.case_count == 1
    assert disclosure.result_audit.result_fingerprint
    assert disclosure.result_audit.manifest["adapter_version"] == "0.1.0"
    assert disclosure.to_dict()["result_audit"]["manifest"]["source_ref"] == "abc123"


def test_competitor_result_scores_clean_synthesis_proof_packet(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint, include_metrics=True)
    case_payload = payload["cases"][0]  # type: ignore[index]
    assert isinstance(case_payload, dict)
    case_payload["answer_candidate"] = {
        "answer": "Accepted cause: expired-jwks-cache",
        "support_source_ids": ["finding-api-jwks"],
    }
    case_payload["synthesis_artifact"] = {
        "answer_candidates": [case_payload["answer_candidate"]],
        "ledger_rows": [{"source_group": "finding-api-jwks", "include_reason": "accepted_parent_state"}],
    }
    case_payload["non_authoritative_rows_injected"] = 0
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.metrics is not None
    assert disclosure.metrics.accepted_state_synthesis_quality == 1.0
    assert disclosure.metrics.non_authoritative_leakage == 1.0


def test_competitor_result_requires_synthesis_proof_for_synthesis_quality(tmp_path: Path) -> None:
    """Accepted-state synthesis quality should not pass from returned_text alone."""
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    case_payload = payload["cases"][0]  # type: ignore[index]
    assert isinstance(case_payload, dict)
    case_payload["returned_text"] = "Accepted cause: expired-jwks-cache"
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.metrics is not None
    assert disclosure.metrics.accepted_state_synthesis_quality == 0.0


def test_competitor_result_requires_supported_synthesis_proof_for_quality(tmp_path: Path) -> None:
    """Answer candidates without support ids or ledger rows should not get proof credit."""
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    case_payload = payload["cases"][0]  # type: ignore[index]
    assert isinstance(case_payload, dict)
    case_payload["answer_candidate"] = {"answer": "Accepted cause: expired-jwks-cache"}
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.metrics is not None
    assert disclosure.metrics.accepted_state_synthesis_quality == 0.0


def test_local_retrieval_baseline_has_no_proof_backed_synthesis_quality(tmp_path: Path) -> None:
    """Raw note baselines should not inherit perfect proof-backed synthesis metrics."""
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)

    metrics = coordination_baseline_metrics(workload.cases[0], "markdown_notes")

    assert metrics.accepted_state_synthesis_quality == 0.0
    assert metrics.non_authoritative_leakage == 0.0
    assert metrics.purpose_feedback_coverage == 0.0


def test_competitor_result_scores_coordinate_purpose_feedback_coverage(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    case_payload = payload["cases"][0]  # type: ignore[index]
    assert isinstance(case_payload, dict)
    case_payload["feedback_events"] = [
        {
            "finding_id": "finding-api-jwks",
            "feedback": "used",
            "purpose": {"profile": "coordinate"},
            "authority_scope": "parent-accepted",
            "outcome": "supported_handoff",
        }
    ]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.metrics is not None
    assert disclosure.metrics.purpose_feedback_coverage == 1.0


def test_competitor_result_requires_coordinate_purpose_feedback_for_coverage(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    case_payload = payload["cases"][0]  # type: ignore[index]
    assert isinstance(case_payload, dict)
    case_payload["feedback_events"] = [
        {
            "finding_id": "finding-api-jwks",
            "feedback": "used",
            "purpose": {"profile": "general"},
            "authority_scope": "worker-local",
        }
    ]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.metrics is not None
    assert disclosure.metrics.purpose_feedback_coverage == 0.0


def test_competitor_result_penalizes_forbidden_synthesis_answer(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    case_payload = payload["cases"][0]  # type: ignore[index]
    assert isinstance(case_payload, dict)
    case_payload["answer_candidate"] = {
        "answer": "Accepted cause: missing-browser-refresh",
        "support_source_ids": ["finding-ui-refresh"],
    }
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.metrics is not None
    assert disclosure.metrics.accepted_state_synthesis_quality == 0.0


def test_competitor_result_penalizes_non_authoritative_synthesis_leakage(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    case_payload = payload["cases"][0]  # type: ignore[index]
    assert isinstance(case_payload, dict)
    case_payload["answer_candidate"] = {"answer": "Accepted cause: expired-jwks-cache"}
    case_payload["injected_text"] = "accepted cause: expired-jwks-cache plus stale flag-missing"
    case_payload["non_authoritative_rows_injected"] = 2
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.metrics is not None
    assert disclosure.metrics.accepted_state_synthesis_quality == 0.0
    assert disclosure.metrics.non_authoritative_leakage == 0.0


def test_competitor_result_rejects_workload_fingerprint_mismatch(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(
        json.dumps(_competitor_result_payload("wrong-fingerprint")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="workload_fingerprint"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_requires_all_workload_cases(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    payload["cases"] = []
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing case output"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_rejects_manifest_name_mismatch(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    payload["manifest"]["name"] = "activegraph"  # type: ignore[index]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest name"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_requires_pinned_manifest_fields(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    del payload["manifest"]["source_ref"]  # type: ignore[index]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest source_ref"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_rejects_duplicate_case_outputs(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    payload["cases"] = [payload["cases"][0], payload["cases"][0]]  # type: ignore[index]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case output"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_rejects_non_object_case_output(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    payload["cases"] = [payload["cases"][0], "not-an-object"]  # type: ignore[index]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="case output must be an object"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_rejects_unexpected_case_id(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    payload["cases"][0]["case_id"] = "unknown-case"  # type: ignore[index]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected case output"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_rejects_unknown_finding_ids(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    payload["cases"][0]["accepted_findings"][0]["finding_id"] = "invented-finding"  # type: ignore[index]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown finding_id"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_rejects_unknown_conflict_finding_ids(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    payload["cases"][0]["conflicts"][0]["finding_ids"] = ["finding-api-jwks", "invented-finding"]  # type: ignore[index]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown finding_id"):
        run_coordination_competitor_adapter("mem0", workload, result_path=result_path)


def test_competitor_result_invalid_citations_do_not_raise_citation_coverage(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    payload = _competitor_result_payload(workload.fingerprint)
    payload["cases"][0]["accepted_findings"][0]["source_event_seq"] = 0  # type: ignore[index]
    payload["cases"][0]["accepted_findings"][0]["source_event_hash"] = "fake"  # type: ignore[index]
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    disclosure = run_coordination_competitor_adapter("mem0", workload, result_path=result_path)

    assert disclosure.metrics is not None
    assert disclosure.metrics.citation_coverage == 0.0


def test_coordinate_benchmark_cli_rejects_duplicate_competitor_result_names(tmp_path: Path) -> None:
    runner = CliRunner()
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark",
            "--output-dir",
            str(tmp_path / "out"),
            "--competitor-result",
            f"mem0={result_path}",
            "--competitor-result",
            f"mem0={result_path}",
        ],
    )

    assert result.exit_code != 0
    assert "duplicate competitor result" in result.output


def test_coordinate_benchmark_cli_accepts_frozen_workload(tmp_path: Path) -> None:
    source_path = tmp_path / "source-workload.json"
    build_coordination_workload(source_path, missions=1, workers=3)
    output_dir = tmp_path / "out"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark",
            "--output-dir",
            str(output_dir),
            "--workload",
            str(source_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["workload_fingerprint"] == json.loads(source_path.read_text(encoding="utf-8"))["fingerprint"]
    assert (output_dir / "coordination-benchmark.json").exists()


def _competitor_runner_script(tmp_path: Path, *, name: str = "mem0", display_name: str = "Mem0") -> Path:
    script = tmp_path / f"{name}_runner.py"
    script.write_text(
        "\n".join(
            [
                "import argparse, json",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--workload', required=True)",
                "parser.add_argument('--output', required=True)",
                "args = parser.parse_args()",
                "workload = json.load(open(args.workload, encoding='utf-8'))",
                "payload = {",
                f"  'name': {name!r},",
                "  'adapter_contract': 'coordinationbench-v1',",
                "  'workload_fingerprint': workload['fingerprint'],",
                "  'generated_at_utc': '2026-05-27T00:00:00Z',",
                "  'manifest': {",
                f"    'name': {name!r},",
                f"    'display_name': {display_name!r},",
                "    'adapter_contract': 'coordinationbench-v1',",
                "    'adapter_version': '0.1.0',",
                f"    'install_command': 'uv tool install {name}==0.1.0',",
                f"    'run_command': '{name}-coordinationbench',",
                f"    'source_url': 'https://example.test/{name}-adapter',",
                "    'source_ref': 'abc123',",
                "    'dataset_contract': 'coordinationbench-v1 workload JSON; runner receives --workload <path>',",
                "    'result_export_schema': 'schemas/result.schema.json; runner must write --output <path>',",
                "  },",
                "  'cases': [{",
                "    'case_id': workload['cases'][0]['case_id'],",
                "    'accepted_findings': [{",
                "      'finding_id': 'finding-api-jwks',",
                "      'worker_id': 'worker-api',",
                "      'summary': 'JWKS cache expiry is the accepted auth failure cause.',",
                "      'claim_key': 'auth.failure.cause',",
                "      'claim_value': 'expired-jwks-cache',",
                "      'evidence': [{'kind': 'command', 'reference': 'pytest tests/test_auth.py -q'}],",
                "      'source_event_seq': 1,",
                "      'source_event_hash': 'a' * 64,",
                "    }],",
                "    'conflicts': [{'claim_key': 'auth.failure.cause', 'finding_ids': ['finding-api-jwks', 'finding-ui-refresh']}],",
                "    'stale_findings': [{'finding_id': 'finding-api-stale-flag'}],",
                "    'returned_text': 'accepted cause: expired-jwks-cache',",
                "    'injected_text': 'accepted cause: expired-jwks-cache',",
                "    'latency_ms': 12.5,",
                "  }],",
                "}",
                "json.dump(payload, open(args.output, 'w', encoding='utf-8'))",
            ]
        ),
        encoding="utf-8",
    )
    return script


def _competitor_runner_manifest(
    tmp_path: Path,
    workload_fingerprint: str,
    *,
    name: str = "mem0",
    display_name: str = "Mem0",
) -> Path:
    script = _competitor_runner_script(tmp_path, name=name, display_name=display_name)
    manifest = {
        "name": name,
        "display_name": display_name,
        "adapter_contract": "coordinationbench-v1",
        "adapter_version": "0.1.0",
        "install_command": f"uv tool install {name}==0.1.0",
        "run_command": [sys.executable, str(script)],
        "source_url": f"https://example.test/{name}-adapter",
        "source_ref": "abc123",
        "dataset_contract": "coordinationbench-v1 workload JSON; runner receives --workload <path>",
        "result_export_schema": "schemas/result.schema.json; runner must write --output <path>",
        "workload_fingerprint": workload_fingerprint,
    }
    manifest_path = tmp_path / f"{name}-runner.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_competitor_runner_executes_pinned_manifest_and_scores_generated_result(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest_path = _competitor_runner_manifest(tmp_path, workload.fingerprint)

    disclosure = run_coordination_competitor_runner(
        "mem0",
        workload,
        manifest_path=manifest_path,
        output_dir=tmp_path / "runner-output",
    )

    assert disclosure.status == "completed"
    assert disclosure.claim_status == "same_harness"
    assert disclosure.metrics is not None
    assert disclosure.metrics.conflict_recall == 1.0
    assert disclosure.result_audit is not None
    assert disclosure.result_audit.manifest["source_ref"] == "abc123"
    assert disclosure.result_audit.runner_command == [sys.executable, str(tmp_path / "mem0_runner.py")]
    audit_payload = disclosure.result_audit.to_dict()
    assert audit_payload["runner_returncode"] == 0
    assert audit_payload["runner_stdout_sha256"] == hashlib.sha256(b"").hexdigest()
    assert audit_payload["runner_stderr_sha256"] == hashlib.sha256(b"").hexdigest()
    assert Path(audit_payload["runner_stdout_path"]).exists()
    assert Path(audit_payload["runner_stderr_path"]).exists()


@pytest.mark.parametrize(
    ("adapter_name", "display_name"),
    [
        ("quarq", "Quarq"),
        ("hybi", "Semantic Reach / HyperBinder / Hybi"),
    ],
)
def test_quarq_and_hybi_runners_execute_only_through_local_scoring(
    tmp_path: Path,
    adapter_name: str,
    display_name: str,
) -> None:
    """New competitor adapters should publish metrics only after local scoring."""
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest_path = _competitor_runner_manifest(
        tmp_path,
        workload.fingerprint,
        name=adapter_name,
        display_name=display_name,
    )

    disclosure = run_coordination_competitor_runner(
        adapter_name,
        workload,
        manifest_path=manifest_path,
        output_dir=tmp_path / f"{adapter_name}-runner-output",
    )

    assert disclosure.status == "completed"
    assert disclosure.claim_status == "same_harness"
    assert disclosure.metrics is not None
    assert disclosure.metrics.conflict_recall == 1.0
    assert disclosure.result_audit is not None
    assert disclosure.result_audit.manifest["name"] == adapter_name
    assert disclosure.result_audit.manifest["source_ref"] == "abc123"
    assert disclosure.result_audit.runner_command == [
        sys.executable,
        str(tmp_path / f"{adapter_name}_runner.py"),
    ]
    audit_payload = disclosure.result_audit.to_dict()
    assert audit_payload["runner_returncode"] == 0
    assert audit_payload["result_fingerprint"]
    assert audit_payload["runner_stdout_sha256"] == hashlib.sha256(b"").hexdigest()
    assert audit_payload["runner_stderr_sha256"] == hashlib.sha256(b"").hexdigest()
    assert Path(audit_payload["runner_stdout_path"]).exists()
    assert Path(audit_payload["runner_stderr_path"]).exists()


def test_competitor_claim_gate_passes_with_scored_quarq_and_hybi_runners(tmp_path: Path) -> None:
    """The public claim gate should pass only with audited same-harness rows."""
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    quarq_manifest = _competitor_runner_manifest(
        tmp_path,
        workload.fingerprint,
        name="quarq",
        display_name="Quarq",
    )
    hybi_manifest = _competitor_runner_manifest(
        tmp_path,
        workload.fingerprint,
        name="hybi",
        display_name="Semantic Reach / HyperBinder / Hybi",
    )
    report = run_coordination_benchmark(
        tmp_path / "report",
        workload_path=tmp_path / "coordination-workload.json",
        competitor_runners={"quarq": quarq_manifest, "hybi": hybi_manifest},
    )

    gate = coordination_competitor_claim_gate(report)

    assert gate.status == "passed"
    assert gate.completed_adapters == ("quarq", "hybi")
    assert gate.blocked_adapters == {}
    payload = report.to_dict()
    assert payload["competitor_claim_gate"]["status"] == "passed"


def test_competitor_runner_rejects_unpinned_workload_manifest(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest_path = _competitor_runner_manifest(tmp_path, "wrong-fingerprint")

    with pytest.raises(ValueError, match="workload_fingerprint"):
        run_coordination_competitor_runner(
            "mem0",
            workload,
            manifest_path=manifest_path,
            output_dir=tmp_path / "runner-output",
        )


def test_competitor_runner_rejects_shell_string_run_command(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest_path = _competitor_runner_manifest(tmp_path, workload.fingerprint)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["run_command"] = f"{sys.executable} {tmp_path / 'mem0_runner.py'}"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run_command"):
        run_coordination_competitor_runner(
            "mem0",
            workload,
            manifest_path=manifest_path,
            output_dir=tmp_path / "runner-output",
        )


def test_competitor_runner_nonzero_exit_reports_stderr_excerpt(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    failing = tmp_path / "fail_runner.py"
    failing.write_text("import sys\nprint('adapter exploded', file=sys.stderr)\nsys.exit(7)\n", encoding="utf-8")
    manifest_path = _competitor_runner_manifest(tmp_path, workload.fingerprint)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["run_command"] = [sys.executable, str(failing)]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    output_dir = tmp_path / "runner-output"
    with pytest.raises(RuntimeError, match="adapter exploded"):
        run_coordination_competitor_runner(
            "mem0",
            workload,
            manifest_path=manifest_path,
            output_dir=output_dir,
        )
    assert (output_dir / "mem0-runner.stdout.txt").exists()
    assert "adapter exploded" in (output_dir / "mem0-runner.stderr.txt").read_text(encoding="utf-8")


def test_competitor_runner_timeout_reports_adapter_name(tmp_path: Path) -> None:
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    slow = tmp_path / "slow_runner.py"
    slow.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    manifest_path = _competitor_runner_manifest(tmp_path, workload.fingerprint)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["run_command"] = [sys.executable, str(slow)]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TimeoutError, match="mem0"):
        run_coordination_competitor_runner(
            "mem0",
            workload,
            manifest_path=manifest_path,
            output_dir=tmp_path / "runner-output",
            timeout_seconds=1,
        )


def test_coordinate_benchmark_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """The CLI should write reproducible JSON and markdown CoordinationBench reports."""
    runner = CliRunner()
    output_dir = tmp_path / "coordination-v1"

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark",
            "--output-dir",
            str(output_dir),
            "--missions",
            "1",
            "--workers",
            "3",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["version"] == COORDINATION_WORKLOAD_VERSION
    assert payload["metrics"]["accepted_finding_precision"] == 1.0
    assert payload["metrics"]["purpose_feedback_coverage"] == 1.0
    assert "baselines" in payload
    assert "competitor_adapters" in payload
    assert "markdown_notes" in payload["baselines"]
    assert payload["competitor_adapters"]["mem0"]["status"] == "not_run"
    assert payload["competitor_adapters"]["quarq"]["status"] == "not_run"
    assert payload["competitor_adapters"]["hybi"]["status"] == "not_run"
    assert payload["coordinate_purpose_synthesis_gate"]["status"] == "passed"
    assert "purpose_feedback_coverage" in payload["coordinate_purpose_synthesis_gate"]["required_metrics"]
    assert payload["competitor_claim_gate"]["status"] == "blocked"
    assert (output_dir / "coordination-benchmark.json").exists()
    assert (
        output_dir / "competitor-runner-manifests" / "activegraph.runner-manifest.template.json"
    ).exists()
    assert (
        output_dir / "competitor-runner-manifests" / "quarq.runner-manifest.template.json"
    ).exists()
    assert (
        output_dir / "competitor-runner-manifests" / "hybi.runner-manifest.template.json"
    ).exists()
    markdown = (output_dir / "coordination-benchmark.md").read_text(encoding="utf-8")
    assert "# CoordinationBench" in markdown
    assert "accepted_finding_precision" in markdown
    assert "purpose_feedback_coverage" in markdown
    assert "## Coordinate Purpose/Synthesis Gate" in markdown
    assert "Coordinate accepted-state synthesis is proof-backed" in markdown
    assert "flat_transcript" in markdown
    assert "bm25_worker_logs" in markdown
    assert "## Competitor Claim Gate" in markdown
    assert "Quarq" in markdown
    assert "Semantic Reach / HyperBinder / Hybi" in markdown
    assert "## Competitor Adapter Disclosures" in markdown
    assert "ActiveGraph" in markdown
    assert "## Limitations" in markdown
    assert "not a universal memory benchmark" in markdown
    assert "## Reproduction" in markdown
    assert "zaxy coordinate benchmark" in markdown
    assert "--output-dir" in markdown
    assert "--workload" in markdown


def test_coordinate_benchmark_cli_ingests_competitor_result(tmp_path: Path) -> None:
    runner = CliRunner()
    output_dir = tmp_path / "coordination-v1"
    workload = build_coordination_workload(tmp_path / "workload.json", missions=1, workers=3)
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(
        json.dumps(_competitor_result_payload(workload.fingerprint)),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark",
            "--output-dir",
            str(output_dir),
            "--competitor-result",
            f"mem0={result_path}",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["competitor_adapters"]["mem0"]["status"] == "completed"
    assert payload["competitor_adapters"]["mem0"]["metrics"]["conflict_recall"] == 1.0
    markdown = (output_dir / "coordination-benchmark.md").read_text(encoding="utf-8")
    assert "## Competitor Adapter Runs" in markdown
    assert "| Mem0 | coordinationbench-v1 | completed | same_harness |" in markdown
    assert "abc123" in markdown
    assert "result_fingerprint" in markdown


def test_coordinate_benchmark_cli_runs_competitor_runner_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    output_dir = tmp_path / "coordination-v1"
    workload = build_coordination_workload(tmp_path / "workload.json", missions=1, workers=3)
    manifest_path = _competitor_runner_manifest(tmp_path, workload.fingerprint)

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark",
            "--output-dir",
            str(output_dir),
            "--competitor-runner",
            f"mem0={manifest_path}",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["competitor_adapters"]["mem0"]["status"] == "completed"
    assert payload["competitor_adapters"]["mem0"]["result_audit"]["runner_command"] == [
        sys.executable,
        str(tmp_path / "mem0_runner.py"),
    ]
    assert (output_dir / "competitor-runners" / "mem0-coordination-result.json").exists()


def test_coordinate_benchmark_cli_rejects_required_competitor_claim_without_scored_row(
    tmp_path: Path,
) -> None:
    """Public claim mode should fail if a required adapter remains disclosure-only."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark",
            "--output-dir",
            str(tmp_path / "coordination-v1"),
            "--require-competitor-claim",
            "quarq",
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert "competitor claim gate blocked" in result.output
    assert "quarq" in result.output


def test_coordinate_benchmark_cli_rejects_duplicate_competitor_result_and_runner_names(tmp_path: Path) -> None:
    runner = CliRunner()
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "mem0-runner.json"
    manifest_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark",
            "--output-dir",
            str(tmp_path / "out"),
            "--competitor-result",
            f"mem0={result_path}",
            "--competitor-runner",
            f"mem0={manifest_path}",
        ],
    )

    assert result.exit_code != 0
    assert "duplicate competitor adapter" in result.output


def test_coordinate_benchmark_adapter_export_kit_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    output_dir = tmp_path / "kit"

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark-adapter",
            "export-kit",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["workload_fingerprint"]
    assert (output_dir / "schemas" / "runner-manifest.schema.json").exists()
    assert (output_dir / "templates" / "activegraph.runner-manifest.template.json").exists()


def test_coordinate_benchmark_adapter_validate_manifest_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    manifest_path = _competitor_runner_manifest(tmp_path, workload.fingerprint)

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark-adapter",
            "validate-manifest",
            f"mem0={manifest_path}",
            "--workload",
            str(tmp_path / "coordination-workload.json"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "mem0"
    assert payload["valid"] is True


def test_coordinate_benchmark_adapter_validate_result_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    workload = build_coordination_workload(tmp_path / "coordination-workload.json", missions=1, workers=3)
    result_path = tmp_path / "mem0-result.json"
    result_path.write_text(json.dumps(_competitor_result_payload(workload.fingerprint)), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "coordinate",
            "benchmark-adapter",
            "validate-result",
            f"mem0={result_path}",
            "--workload",
            str(tmp_path / "coordination-workload.json"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["valid"] is True
    assert payload["metrics"]["accepted_finding_precision"] == 1.0


def test_coordination_benchmark_report_round_trips_without_non_finite_values(tmp_path: Path) -> None:
    """Report JSON should be strict and contain no NaN or Infinity values."""
    report = run_coordination_benchmark(tmp_path, missions=1, workers=3)
    encoded = json.dumps(report.to_dict(), allow_nan=False)

    decoded = json.loads(encoded)
    assert decoded["metrics"]["brief_latency_ms"] >= 0.0
    assert decoded["metrics"]["promotion_latency_ms"] >= 0.0
