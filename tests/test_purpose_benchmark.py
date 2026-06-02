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
