"""Tests for the internal cognitive memory lanes: forgetting and FoK calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.cli import benchmarks as cli_benchmarks
from zaxy_benchmarks.fok_calibration_lane import (
    FOK_CALIBRATION_LANE_VERSION,
    run_fok_calibration_lane,
)
from zaxy_benchmarks.forgetting_lane import (
    FORGETTING_LANE_VERSION,
    run_forgetting_lane,
)


class TestForgettingLane:
    @pytest.fixture(scope="class")
    def lane_result(self, tmp_path_factory: pytest.TempPathFactory) -> dict:
        return run_forgetting_lane(tmp_path_factory.mktemp("forgetting-lane"))

    def test_lane_is_deterministic_across_runs(self, tmp_path: Path) -> None:
        """Two seeded-fabric runs must produce byte-identical lane results."""
        first = run_forgetting_lane(tmp_path / "run1")
        second = run_forgetting_lane(tmp_path / "run2")
        assert first == second

    def test_report_is_labeled_internal(self, lane_result: dict) -> None:
        assert lane_result["lane"] == "forgetting"
        assert lane_result["validation"] == "internal"
        assert lane_result["version"] == FORGETTING_LANE_VERSION

    def test_cold_start_parity_holds_with_zero_reinforcement(self, lane_result: dict) -> None:
        """The key flip-safety number: cognitive == plain without reinforcement."""
        parity = lane_result["checks"]["cold_start_parity"]
        assert parity["checkout_layer"]["identical_fraction"] == 1.0
        assert all(parity["checkout_layer"]["per_query_identical"])
        assert parity["full_path"]["identical_fraction"] == 1.0
        assert all(parity["full_path"]["per_query_identical"])
        assert parity["status"] == "pass"

    def test_no_recall_loss_for_below_floor_memories(self, lane_result: dict) -> None:
        """Spec exit criterion: attenuated memories labeled, reachable, restorable."""
        check = lane_result["checks"]["no_recall_loss"]
        assert check["below_floor_memory_count"] >= 4
        assert check["explicit_query_retrieved_fraction"] == 1.0
        assert check["labeled_attenuated_fraction"] == 1.0
        assert check["replay_reachable_fraction"] == 1.0
        assert check["status"] == "pass"
        # The reported scores must actually sit below the floor.
        floor = lane_result["fixture"]["salience_floor"]
        scores = check["below_floor_salience_scores"]
        assert len(scores) == check["below_floor_memory_count"]
        assert all(score < floor for score in scores.values())
        # Re-derive the fractions from the raw rows instead of trusting flags.
        for row in check["per_memory"]:
            assert row["explicit_query_retrieved"] is True
            assert row["labeled_attenuated_in_diagnostics"] is True
            assert row["replay_reachable"] is True

    def test_ranking_lift_separates_cognitive_from_plain(self, lane_result: dict) -> None:
        check = lane_result["checks"]["ranking_lift"]
        assert check["pair_count"] >= 6
        assert (
            check["cognitive_reinforced_first_fraction"]
            > check["plain_reinforced_first_fraction"]
        )
        for row in check["per_pair"]:
            assert row["plain_both_present"] is True
            assert row["cognitive_both_present"] is True

    def test_exemptions_keep_pinned_and_authority_memories(self, lane_result: dict) -> None:
        check = lane_result["checks"]["exemption_correctness"]
        assert check["exempt_memory_count"] >= 4
        assert check["surfaced_fraction"] == 1.0
        assert check["exempt_reason_correct_fraction"] == 1.0
        assert check["status"] == "pass"
        reasons = {row["expected_reason"] for row in check["per_memory"]}
        assert reasons == {"pinned", "authority"}

    def test_contract_summarizes_every_check(self, lane_result: dict) -> None:
        contract = lane_result["contract"]
        assert set(contract) == {
            "cold_start_parity",
            "no_recall_loss",
            "ranking_lift",
            "exemption_correctness",
            "status",
        }
        assert contract["status"] == (
            "pass"
            if all(value == "pass" for name, value in contract.items() if name != "status")
            else "fail"
        )

    def test_report_carries_no_wall_clock_values(self, lane_result: dict) -> None:
        """Only the fixed lane instants may appear; no run-time hashes/timestamps."""
        import re

        rendered = json.dumps(lane_result)
        assert lane_result["fixture"]["lane_now"] == "2026-03-01T00:00:00Z"
        # No sealed event hashes (they embed append wall-clock timestamps).
        assert re.search(r"[0-9a-f]{64}", rendered) is None
        # No timestamps beyond the two fixed lane instants.
        timestamps = set(re.findall(r"\d{4}-\d{2}-\d{2}T[0-9:.]+Z?", rendered))
        assert timestamps == {"2026-03-01T00:00:00Z", "2026-01-01T00:00:00Z"}


class TestFokCalibrationLane:
    @pytest.fixture(scope="class")
    def lane_result(self, tmp_path_factory: pytest.TempPathFactory) -> dict:
        return run_fok_calibration_lane(
            tmp_path_factory.mktemp("fok-lane"), sizes=(40,)
        )

    def test_lane_is_deterministic_across_runs(self, tmp_path: Path) -> None:
        first = run_fok_calibration_lane(tmp_path / "run1", sizes=(30,))
        second = run_fok_calibration_lane(tmp_path / "run2", sizes=(30,))
        assert first == second

    def test_report_is_labeled_internal(self, lane_result: dict) -> None:
        assert lane_result["lane"] == "fok_calibration"
        assert lane_result["validation"] == "internal"
        assert lane_result["version"] == FOK_CALIBRATION_LANE_VERSION

    def test_ground_truth_comes_from_real_retrieval(self, lane_result: dict) -> None:
        """Labels must vary across families: real retrieval, not assumptions."""
        corpus = lane_result["corpora"][0]
        labels_by_family: dict[str, set[int]] = {}
        for row in corpus["queries"]:
            labels_by_family.setdefault(row["family"], set()).add(row["label"])
        # Present-entity queries must actually retrieve; absent must not.
        assert labels_by_family["present"] == {1}
        assert labels_by_family["absent"] == {0}
        assert 0.0 < corpus["positive_rate"] < 1.0

    def test_brier_scores_are_consistent_with_raw_rows(self, lane_result: dict) -> None:
        corpus = lane_result["corpora"][0]
        rows = corpus["queries"]
        labels = [row["label"] for row in rows]
        scores = [row["score"] for row in rows]
        positive_rate = sum(labels) / len(labels)
        brier_fok = sum(
            (score - label) ** 2 for score, label in zip(scores, labels, strict=True)
        ) / len(rows)
        brier_base = sum((positive_rate - label) ** 2 for label in labels) / len(rows)
        assert corpus["brier_fok"] == pytest.approx(brier_fok, abs=1e-4)
        assert corpus["brier_base_rate"] == pytest.approx(brier_base, abs=1e-4)
        assert corpus["beats_base_rate"] == (corpus["brier_fok"] < corpus["brier_base_rate"])

    def test_exit_criterion_is_reported_per_corpus_size(self, lane_result: dict) -> None:
        contract = lane_result["contract"]
        per_size = contract["beats_base_rate_per_size"]
        assert set(per_size) == {
            str(corpus["corpus_size"]) for corpus in lane_result["corpora"]
        }
        expected = "pass" if all(per_size.values()) else "fail"
        assert contract["status"] == expected

    def test_verdict_buckets_and_error_rates_are_in_range(self, lane_result: dict) -> None:
        corpus = lane_result["corpora"][0]
        buckets = corpus["verdict_buckets"]
        assert sum(buckets[v]["count"] for v in ("likely", "possible", "unlikely")) == (
            corpus["query_count"]
        )
        for rate in (corpus["false_positive_rate"], corpus["false_negative_rate"]):
            assert rate is None or 0.0 <= rate <= 1.0

    def test_invalid_sizes_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive integers"):
            run_fok_calibration_lane(tmp_path, sizes=(0,))
        with pytest.raises(ValueError, match="at least one"):
            run_fok_calibration_lane(tmp_path, sizes=())


class TestCognitiveLanesCli:
    def test_cli_parses_lane_selection(self) -> None:
        assert cli_benchmarks._parse_cognitive_lanes("all") == (
            "forgetting",
            "fok-calibration",
        )
        assert cli_benchmarks._parse_cognitive_lanes("fok-calibration,forgetting") == (
            "fok-calibration",
            "forgetting",
        )
        with pytest.raises(Exception, match="Unsupported cognitive lane"):
            cli_benchmarks._parse_cognitive_lanes("cache")
        with pytest.raises(Exception, match="at least one lane"):
            cli_benchmarks._parse_cognitive_lanes(",")

    def test_cli_parses_fok_sizes(self) -> None:
        assert cli_benchmarks._parse_fok_corpus_sizes("50,200") == (50, 200)
        with pytest.raises(Exception, match="invalid corpus size"):
            cli_benchmarks._parse_fok_corpus_sizes("50,many")
        with pytest.raises(Exception, match="positive integers"):
            cli_benchmarks._parse_fok_corpus_sizes("0")
        with pytest.raises(Exception, match="at least one corpus size"):
            cli_benchmarks._parse_fok_corpus_sizes(",")

    def test_cli_runs_fok_lane_and_writes_report(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "reports"
        result = CliRunner().invoke(
            app,
            [
                "cognitive-lanes",
                "--lanes",
                "fok-calibration",
                "--fok-sizes",
                "30",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((output_dir / "cognitive-lanes.json").read_text(encoding="utf-8"))
        assert payload["validation"] == "internal"
        assert set(payload["lanes"]) == {"fok_calibration"}
        lane = payload["lanes"]["fok_calibration"]
        assert lane["validation"] == "internal"
        assert lane["corpora"][0]["corpus_size"] == 30

    def test_cli_rejects_unknown_lane(self) -> None:
        result = CliRunner().invoke(app, ["cognitive-lanes", "--lanes", "budget"])
        assert result.exit_code != 0
