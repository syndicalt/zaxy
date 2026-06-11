"""Tests for the internal PPR graph-walk and vector-scale lanes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.cli import benchmarks as cli_benchmarks
from zaxy_benchmarks.graph_walk_lane import (
    BRIDGE_CASES,
    DIRECT_CASES,
    GRAPH_WALK_LANE_VERSION,
    run_graph_walk_lane,
)
from zaxy_benchmarks.vector_scale_lane import (
    RECALL_FLOOR,
    TARGET_SCALE,
    VECTOR_SCALE_LANE_VERSION,
    run_vector_scale_lane,
)

# CI-friendly vector-scale settings: small corpus, ANN threshold below it so
# the HNSW path still engages, one latency pass.
_SCALE_TEST_SIZE = 600
_SCALE_TEST_KWARGS: dict[str, Any] = {
    "sizes": (_SCALE_TEST_SIZE,),
    "ann_threshold": 256,
    "query_count": 16,
    "latency_passes": 1,
}


@pytest.fixture(scope="module")
def graph_walk_results(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two full graph-walk lane runs in separate directories."""
    first = run_graph_walk_lane(tmp_path_factory.mktemp("walk-run1"))
    second = run_graph_walk_lane(tmp_path_factory.mktemp("walk-run2"))
    return first, second


@pytest.fixture(scope="module")
def vector_scale_results(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two small vector-scale lane runs in separate directories."""
    first = run_vector_scale_lane(tmp_path_factory.mktemp("scale-run1"), **_SCALE_TEST_KWARGS)
    second = run_vector_scale_lane(tmp_path_factory.mktemp("scale-run2"), **_SCALE_TEST_KWARGS)
    return first, second


class TestGraphWalkLane:
    def test_lane_is_deterministic_across_runs(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Two seeded-fabric runs must produce byte-identical lane results."""
        first, second = graph_walk_results
        assert first == second

    def test_lane_is_labeled_internal(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        result = graph_walk_results[0]
        assert result["lane"] == "graph_walk"
        assert result["validation"] == "internal"
        assert result["version"] == GRAPH_WALK_LANE_VERSION

    def test_fixture_projects_real_graph(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """The seeded fabric must project nodes, edges, and a stable signature."""
        fixture = graph_walk_results[0]["fixture"]
        assert fixture["bridge_case_count"] == len(BRIDGE_CASES)
        assert fixture["direct_case_count"] == len(DIRECT_CASES)
        assert fixture["adjacency"]["node_count"] > 0
        assert fixture["adjacency"]["edge_count"] > 0
        assert fixture["adjacency"]["signature"]

    def test_plain_arm_cannot_distinguish_bridge_pairs(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """The fixture's lexical tie must be exact: plain margin is zero."""
        for case in graph_walk_results[0]["bridge"]["cases"]:
            assert case["plain"]["target_distractor_margin"] == 0.0, case["case"]
            assert case["plain"]["target_score"] == case["plain"]["distractor_score"]

    def test_walk_arm_separates_every_bridge_pair_by_graph_evidence(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Walk margins must be strictly positive toward the bridged target."""
        cases = graph_walk_results[0]["bridge"]["cases"]
        for case in cases:
            assert case["walk"]["target_distractor_margin"] > 0.0, case["case"]
            assert case["walk"]["target_in_top_k"] is True
        # Re-derive the headline fractions instead of trusting the flags.
        fractions = graph_walk_results[0]["bridge"]["fractions"]
        assert fractions["positive_margin_plain"] == 0.0
        assert fractions["positive_margin_walk"] == 1.0
        assert fractions["margin_gained_fraction"] == 1.0

    def test_single_hop_queries_do_not_regress(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        single_hop = graph_walk_results[0]["single_hop"]
        assert single_hop["non_regression"] is True
        for case in single_hop["cases"]:
            assert case["retained_in_top_k"] is True
            assert case["rank_not_worse"] is True

    def test_walk_cache_serves_repeat_passes(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        determinism = graph_walk_results[0]["determinism"]
        assert determinism["repeat_pass_identical"] is True
        cache = determinism["walk_cache"]
        assert cache["first_pass_misses"] > 0
        assert cache["second_pass_served_from_cache"] is True

    def test_exit_criteria_pass(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        exit_criteria = graph_walk_results[0]["exit_criteria"]
        assert exit_criteria["multi_hop_lift"] is True
        assert exit_criteria["single_hop_non_regression"] is True
        assert exit_criteria["status"] == "pass"

    def test_embedding_variant_is_context_only(
        self, graph_walk_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """The hash-vector variant is reported but never gates exit criteria."""
        variant = graph_walk_results[0]["embedding_variant"]
        assert "note" in variant
        assert set(variant["bridge_fractions"]) == set(
            graph_walk_results[0]["bridge"]["fractions"]
        )

    def test_invalid_top_k_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="top_k"):
            run_graph_walk_lane(tmp_path, top_k=0)


class TestVectorScaleLane:
    def test_deterministic_block_is_reproducible(
        self, vector_scale_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Corpus hashes, exact/quantized recall, and bytes must match across runs."""
        first, second = vector_scale_results
        assert first["deterministic"] == second["deterministic"]
        assert first["config"] == second["config"]

    def test_lane_is_labeled_internal(
        self, vector_scale_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        result = vector_scale_results[0]
        assert result["lane"] == "vector_scale"
        assert result["validation"] == "internal"
        assert result["version"] == VECTOR_SCALE_LANE_VERSION

    def test_all_modes_engage_their_index_paths(
        self, vector_scale_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        modes = vector_scale_results[0]["deterministic"]["sizes"][str(_SCALE_TEST_SIZE)]["modes"]
        assert modes["exact"]["group_type"] == "dense"
        assert modes["ann"]["group_type"] == "ann"
        assert modes["ann"]["engaged"] is True
        assert modes["quantized"]["group_type"] == "quantized"
        assert modes["quantized"]["engaged"] is True

    def test_resident_byte_accounting_matches_index_layouts(
        self, vector_scale_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Bytes must reflect float64, int8+scales, and DB-resident layouts."""
        result = vector_scale_results[0]
        dimension = result["config"]["dimension"]
        modes = result["deterministic"]["sizes"][str(_SCALE_TEST_SIZE)]["modes"]
        assert modes["exact"]["resident_index_bytes"] == _SCALE_TEST_SIZE * dimension * 8
        assert modes["quantized"]["resident_index_bytes"] == (
            _SCALE_TEST_SIZE * dimension + _SCALE_TEST_SIZE * 8
        )
        # ANN vectors are resident in the Kuzu database, not in process memory.
        assert modes["ann"]["resident_index_bytes"] == 0
        assert modes["quantized"]["bytes_vs_exact_ratio"] < 1.0

    def test_quantized_recall_is_deterministic_and_meets_floor(
        self, vector_scale_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        modes = vector_scale_results[0]["deterministic"]["sizes"][str(_SCALE_TEST_SIZE)]["modes"]
        assert modes["quantized"]["recall_at_k"] >= RECALL_FLOOR
        assert modes["exact"]["recall_at_k"] == 1.0

    def test_ann_recall_is_reported_as_run_dependent_measurement(
        self, vector_scale_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """HNSW rebuilds are not reproducible, so ann recall lives in measurements."""
        result = vector_scale_results[0]
        size_key = str(_SCALE_TEST_SIZE)
        assert "recall_at_k" not in result["deterministic"]["sizes"][size_key]["modes"]["ann"]
        ann_recall = result["measurements"]["sizes"][size_key]["ann_recall_at_k"]
        assert 0.0 <= ann_recall <= 1.0

    def test_latency_measurements_are_present_and_labeled(
        self, vector_scale_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        result = vector_scale_results[0]
        assert "determinism" in result["measurements"]["note"]
        latency = result["measurements"]["sizes"][str(_SCALE_TEST_SIZE)]["latency_ms"]
        for mode in ("exact", "ann", "quantized"):
            assert latency[mode]["p50_ms"] > 0.0
            assert latency[mode]["p95_ms"] >= latency[mode]["p50_ms"]
            assert latency[mode]["samples"] == 16

    def test_exit_criteria_report_target_scale_honestly(
        self, vector_scale_results: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Small runs must never claim the 10^5 roadmap criterion."""
        exit_criteria = vector_scale_results[0]["exit_criteria"]
        assert exit_criteria["evaluated_at_size"] == _SCALE_TEST_SIZE
        assert exit_criteria["evaluated_at_target_scale"] is False
        assert exit_criteria["status"] == "not_evaluated_at_target_scale"
        assert TARGET_SCALE == 100_000
        for mode in ("ann", "quantized"):
            verdict = exit_criteria["modes"][mode]
            assert verdict["engaged"] is True
            assert isinstance(verdict["recall_pass"], bool)
            assert isinstance(verdict["bytes_improved"], bool)
            assert isinstance(verdict["latency_improved_p50"], bool)

    def test_invalid_parameters_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one corpus size"):
            run_vector_scale_lane(tmp_path, sizes=())
        with pytest.raises(ValueError, match="sizes must be positive"):
            run_vector_scale_lane(tmp_path, sizes=(0,))
        with pytest.raises(ValueError, match="query_count"):
            run_vector_scale_lane(tmp_path, sizes=(10,), query_count=0)
        with pytest.raises(ValueError, match="latency_passes"):
            run_vector_scale_lane(tmp_path, sizes=(10,), latency_passes=0)


class TestGraphScaleLanesCli:
    def test_cli_parses_lane_selection(self) -> None:
        valid = ("graph-walk", "vector-scale")
        assert cli_benchmarks._parse_graph_scale_lanes("all") == valid
        assert cli_benchmarks._parse_graph_scale_lanes("vector-scale,graph-walk,vector-scale") == (
            "vector-scale",
            "graph-walk",
        )
        with pytest.raises(Exception, match="Unsupported graph/scale lane"):
            cli_benchmarks._parse_graph_scale_lanes("forgetting")
        with pytest.raises(Exception, match="at least one lane"):
            cli_benchmarks._parse_graph_scale_lanes(",")

    def test_cli_parses_scale_sizes(self) -> None:
        assert cli_benchmarks._parse_vector_scale_sizes("1000,10000") == (1000, 10000)
        with pytest.raises(Exception, match="invalid corpus size"):
            cli_benchmarks._parse_vector_scale_sizes("1000,big")
        with pytest.raises(Exception, match="must be positive"):
            cli_benchmarks._parse_vector_scale_sizes("0")
        with pytest.raises(Exception, match="at least one corpus size"):
            cli_benchmarks._parse_vector_scale_sizes(",")

    def test_cli_runs_vector_scale_lane_and_writes_report(self, tmp_path: Path) -> None:
        """A small CLI run should print and persist one internally labeled report."""
        output_dir = tmp_path / "reports"
        result = CliRunner().invoke(
            app,
            [
                "graph-scale-lanes",
                "--lanes",
                "vector-scale",
                "--scale-sizes",
                "400",
                "--query-count",
                "8",
                "--latency-passes",
                "1",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads((output_dir / "graph-scale-lanes.json").read_text(encoding="utf-8"))
        assert payload["validation"] == "internal"
        assert payload["version"] == "graph-scale-lanes-v1"
        assert set(payload["lanes"]) == {"vector_scale"}
        lane = payload["lanes"]["vector_scale"]
        assert lane["validation"] == "internal"
        assert lane["exit_criteria"]["status"] == "not_evaluated_at_target_scale"

    def test_cli_rejects_unknown_lane(self) -> None:
        result = CliRunner().invoke(app, ["graph-scale-lanes", "--lanes", "budget"])
        assert result.exit_code != 0
