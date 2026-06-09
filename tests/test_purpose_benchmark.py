import json
from pathlib import Path

from typer.testing import CliRunner
from zaxy_benchmarks.purpose_benchmark import (
    PURPOSE_BENCHMARK_LANES,
    purpose_holdout_fingerprint,
    run_purpose_benchmark,
)

from zaxy.__main__ import app

HOLDOUT_PACK = Path("reports/benchmarks/purpose-v1/holdouts/public-derived-purpose-v1/holdout-pack.json")


def test_purpose_benchmark_passes_all_research_lanes() -> None:
    report = run_purpose_benchmark()

    assert report.status == "passed"
    assert report.passed_lanes == len(PURPOSE_BENCHMARK_LANES)
    assert {lane.name for lane in report.lanes} == set(PURPOSE_BENCHMARK_LANES)
    assert all(lane.status == "passed" for lane in report.lanes)
    assert report.holdout_reports == {}
    assert report.competitor_claim_status == "blocked"
    assert "Semantic Reach" in report.competitor_claim_blockers[0]
    assert "Quarq" in report.competitor_claim_blockers[0]


def test_purpose_benchmark_includes_evidence_policy_fixtures() -> None:
    report = run_purpose_benchmark()
    lane = next(lane for lane in report.lanes if lane.name == "Evidence Policy Discipline")

    assert lane.status == "passed"
    assert set(lane.evidence) == {
        "security",
        "release",
        "coordinate",
        "support",
        "product",
        "sales",
        "legal",
        "executive",
    }
    for profile in lane.evidence:
        assert lane.evidence[profile]["unsupported"]["satisfied"] is False
        assert lane.evidence[profile]["unsupported"]["suggested_queries"]
        assert lane.evidence[profile]["supported"]["satisfied"] is True


def test_purpose_benchmark_includes_broader_profile_fixtures() -> None:
    report = run_purpose_benchmark()
    lane = next(lane for lane in report.lanes if lane.name == "Broader Profile Fixtures")

    assert lane.status == "passed"
    assert set(lane.evidence["passed_profiles"]) == {"support", "product", "sales", "legal", "executive"}
    assert lane.evidence["local_project_memory_positioning"] is True
    for profile in ("support", "product", "sales", "legal", "executive"):
        assert lane.evidence["checkout_ready"][profile]["has_evidence_policy"] is True
        assert lane.evidence["checkout_ready"][profile]["lens_applied"] is True
        assert lane.evidence["compaction"][profile]["purpose"] == profile
        assert lane.evidence["compaction"][profile]["record_kinds"]


def test_purpose_benchmark_includes_neutral_substrate_projection() -> None:
    report = run_purpose_benchmark()
    lane = next(lane for lane in report.lanes if lane.name == "Neutral Substrate Projection")

    assert lane.status == "passed"
    assert lane.evidence["ingestion_audit"]["safe"] is True
    projections = lane.evidence["purpose_projections"]
    assert set(projections) == {"support", "product", "legal", "executive"}
    assert {projection["neutral_substrate_id"] for projection in projections.values()} == {
        lane.evidence["neutral_substrate"]["name"]
    }
    assert {projection["source_backpointer"] for projection in projections.values()} == {
        "customers/acme-email.txt:1-4"
    }
    assert {
        projection["purpose_label"] for projection in projections.values()
    } == {"customer_escalation", "roadmap_commitment", "legal_obligation", "churn_risk"}


def test_purpose_benchmark_reports_representative_holdouts_separately() -> None:
    report = run_purpose_benchmark(holdout_packs=(HOLDOUT_PACK,))

    assert set(report.holdout_reports) == {"public-derived-purpose-v1"}
    assert report.passed_lanes == len(PURPOSE_BENCHMARK_LANES)
    holdout = report.holdout_reports["public-derived-purpose-v1"]
    assert holdout["claim_status"] == "public_derived_holdout"
    assert holdout["gate_status"] == "diagnostic"
    assert holdout["pack_fingerprint"] == "0d8217bb4e905164305970050ef34c987d7e9b287ce648a1730685f3dd0e61f6"
    assert holdout["metrics"]["case_count"] == 5
    assert holdout["metrics"]["citation_coverage"] == 1.0
    pack = json.loads(HOLDOUT_PACK.read_text(encoding="utf-8"))
    assert purpose_holdout_fingerprint(pack) == pack["fingerprint"]
    covered_profiles = {
        case["purpose_profile"]
        for case in pack["cases"]
    }
    assert {"release", "review", "security", "support", "coordinate"} <= covered_profiles


def test_purpose_benchmark_action_outcome_loop_proves_future_effect() -> None:
    report = run_purpose_benchmark()
    lane = next(lane for lane in report.lanes if lane.name == "Action Outcome Loop")

    assert lane.status == "passed"
    assert lane.evidence["boosted_context"] == "migration retry"
    assert any(
        explanation.get("suppression_candidate")
        for explanation in lane.evidence["outcome_explanations"]
    )


def test_purpose_benchmark_cli_writes_json_and_markdown(tmp_path) -> None:
    runner = CliRunner()
    output_dir = tmp_path / "purpose-v1"

    result = runner.invoke(
        app,
        [
            "purpose-benchmark",
            "--output-dir",
            str(output_dir),
            "--include-holdouts",
            "--require-holdout-fingerprint",
            "0d8217bb4e905164305970050ef34c987d7e9b287ce648a1730685f3dd0e61f6",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Purpose benchmark: passed" in result.output
    assert (output_dir / "purpose-benchmark.md").exists()
    payload = json.loads((output_dir / "purpose-benchmark.json").read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["passed_lanes"] == len(PURPOSE_BENCHMARK_LANES)
    assert payload["holdout_reports"]["public-derived-purpose-v1"]["gate_status"] == "diagnostic"
    assert "Accepted-State Discipline" in (output_dir / "purpose-benchmark.md").read_text(
        encoding="utf-8"
    )
    assert "Public-Derived Holdouts" in (output_dir / "purpose-benchmark.md").read_text(
        encoding="utf-8"
    )


def test_purpose_benchmark_cli_rejects_mismatched_holdout_fingerprint(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "purpose-benchmark",
            "--output-dir",
            str(tmp_path / "purpose-v1"),
            "--include-holdouts",
            "--require-holdout-fingerprint",
            "0" * 64,
        ],
    )

    assert result.exit_code != 0
    assert "did not match an included holdout pack" in result.output
