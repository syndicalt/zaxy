import json

from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.purpose_benchmark import PURPOSE_BENCHMARK_LANES, run_purpose_benchmark


def test_purpose_benchmark_passes_all_research_lanes() -> None:
    report = run_purpose_benchmark()

    assert report.status == "passed"
    assert report.passed_lanes == len(PURPOSE_BENCHMARK_LANES)
    assert {lane.name for lane in report.lanes} == set(PURPOSE_BENCHMARK_LANES)
    assert all(lane.status == "passed" for lane in report.lanes)
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

    result = runner.invoke(app, ["purpose-benchmark", "--output-dir", str(output_dir)])

    assert result.exit_code == 0, result.output
    assert "Purpose benchmark: passed" in result.output
    assert (output_dir / "purpose-benchmark.md").exists()
    payload = json.loads((output_dir / "purpose-benchmark.json").read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["passed_lanes"] == len(PURPOSE_BENCHMARK_LANES)
    assert "Accepted-State Discipline" in (output_dir / "purpose-benchmark.md").read_text(
        encoding="utf-8"
    )
