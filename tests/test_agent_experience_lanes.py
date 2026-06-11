"""Tests for the internal Phase 1 agent-experience lanes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.cli import benchmarks as cli_benchmarks
from zaxy.mcp_server import TOOLS
from zaxy.tool_profiles import CORE_TOOLS
from zaxy_benchmarks.agent_experience_lanes import (
    AGENT_EXPERIENCE_LANE_NAMES,
    AGENT_EXPERIENCE_LANES_VERSION,
    DEFAULT_BUDGET_SWEEP,
    FRONT_DOOR_TOOL,
    run_agent_experience_lanes,
    run_budget_lane,
    run_cache_lane,
    run_tool_adoption_lane,
)


class TestToolAdoptionLane:
    def test_lane_is_deterministic(self) -> None:
        """The static listing-surface lane must be exactly reproducible."""
        assert run_tool_adoption_lane() == run_tool_adoption_lane()

    def test_core_profile_is_strictly_smaller_than_full(self) -> None:
        """Core must list fewer tools and fewer estimated schema tokens than full."""
        result = run_tool_adoption_lane()
        core = result["profiles"]["core"]
        full = result["profiles"]["full"]

        assert result["validation"] == "internal"
        assert result["version"] == AGENT_EXPERIENCE_LANES_VERSION
        assert core["listed_tool_count"] < full["listed_tool_count"]
        assert core["schema_bytes"] < full["schema_bytes"]
        assert core["estimated_schema_tokens"] < full["estimated_schema_tokens"]
        assert result["deltas"]["listed_tool_count"] == (
            full["listed_tool_count"] - core["listed_tool_count"]
        )
        assert result["deltas"]["estimated_schema_tokens"] == (
            full["estimated_schema_tokens"] - core["estimated_schema_tokens"]
        )
        assert 0.0 < result["deltas"]["schema_token_reduction_fraction"] < 1.0

    def test_front_door_rank_is_reported_truthfully(self) -> None:
        """The reported rank must match the actual MCP listing order per profile."""
        result = run_tool_adoption_lane()

        full_names = [tool.name for tool in TOOLS]
        core_names = [name for name in full_names if name in CORE_TOOLS]
        assert result["front_door_tool"] == FRONT_DOOR_TOOL
        for profile, names in (("core", core_names), ("full", full_names)):
            metrics = result["profiles"][profile]
            assert metrics["front_door_listed"] is True
            assert metrics["front_door_rank"] == names.index(FRONT_DOOR_TOOL) + 1
            assert metrics["listed_tool_count"] == len(names)
            assert 0.0 <= metrics["front_door_reference_fraction"] <= 1.0

    def test_front_door_references_count_other_listed_descriptions(self) -> None:
        """Reference counts must match descriptions that actually name the front door."""
        result = run_tool_adoption_lane()
        expected_full = sum(
            1
            for tool in TOOLS
            if tool.name != FRONT_DOOR_TOOL and FRONT_DOOR_TOOL in (tool.description or "")
        )
        assert result["profiles"]["full"]["front_door_reference_count"] == expected_full


class TestBudgetLane:
    def test_lane_is_deterministic_across_runs(self, tmp_path: Path) -> None:
        """Two seeded-fabric runs must produce byte-identical lane results."""
        first = run_budget_lane(tmp_path / "run1")
        second = run_budget_lane(tmp_path / "run2")
        assert first == second

    def test_contract_holds_against_real_checkout_path(self, tmp_path: Path) -> None:
        """Citation preservation and monotone elision must pass on the real path."""
        result = run_budget_lane(tmp_path / "lane")

        assert result["validation"] == "internal"
        assert result["contract"]["status"] == "pass"
        assert result["contract"]["citation_fields_preserved_at_every_budget"] is True
        assert result["contract"]["elided_count_monotone_non_increasing"] is True
        assert result["contract"]["elided_kinds_monotone_non_increasing"] is True

        sweep = result["sweep"]
        assert len(sweep) == len(DEFAULT_BUDGET_SWEEP)
        assert all(point["citation_fields_preserved"] for point in sweep)
        # The fixture makes packing meaningful: the tightest budget elides
        # sections and the loosest budgets elide nothing.
        assert sweep[0]["budget_requested"] == 256
        assert sweep[0]["elided_count"] > 0
        assert sweep[-1]["budget_requested"] is None
        assert sweep[-1]["elided_count"] == 0
        # Re-derive monotonicity from the raw curve instead of trusting flags.
        finite_points = [point for point in sweep if point["budget_requested"] is not None]
        for previous, current in zip(finite_points, finite_points[1:], strict=False):
            assert current["elided_count"] <= previous["elided_count"]
            assert set(current["elided_kinds"]) <= set(previous["elided_kinds"])
            assert current["budget_used"] >= previous["budget_used"]

    def test_budget_used_is_reported_per_finite_budget(self, tmp_path: Path) -> None:
        """Each finite budget reports its packed-prompt token usage."""
        result = run_budget_lane(tmp_path / "lane", budgets=(0, 512, None))
        by_budget = {point["budget_requested"]: point for point in result["sweep"]}
        # Zero budget keeps only mandatory trust-contract sections; usage is
        # still positive and the contract still preserves citation fields.
        assert by_budget[0]["budget_used"] > 0
        assert by_budget[0]["citation_fields_preserved"] is True
        assert by_budget[512]["budget_used"] <= 512
        assert by_budget[None]["budget_used"] is None
        assert by_budget[None]["elided_count"] == 0

    def test_empty_budget_sweep_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one budget"):
            run_budget_lane(tmp_path / "lane", budgets=())


class TestCacheLane:
    def test_lane_is_deterministic_across_runs(self, tmp_path: Path) -> None:
        """Two seeded-fabric runs must produce byte-identical lane results."""
        first = run_cache_lane(tmp_path / "run1", repeats=3)
        second = run_cache_lane(tmp_path / "run2", repeats=3)
        assert first == second

    def test_stable_prefix_repeat_invariance_and_append_change(self, tmp_path: Path) -> None:
        """Repeats reuse one byte-identical prefix; a consolidated append changes it."""
        result = run_cache_lane(tmp_path / "lane", repeats=3)

        assert result["validation"] == "internal"
        assert result["contract"]["status"] == "pass"
        assert result["prefix_byte_identical_across_repeats"] is True
        # Full-prompt identity is informational only: checkout records salience
        # reinforcement events whose replay lands in the volatile tail.
        assert isinstance(result["prompt_byte_identical_across_repeats"], bool)
        assert result["stable_prefix_chars"] > len("# Memory Checkout")
        assert result["append"]["prefix_changed"] is True
        assert result["append"]["prefix_grew"] is True
        assert result["append"]["stable_prefix_chars"] > result["stable_prefix_chars"]

    def test_estimated_cache_hit_fraction_is_arithmetic(self, tmp_path: Path) -> None:
        """The estimated fraction must equal prefix tokens over prompt tokens."""
        result = run_cache_lane(tmp_path / "lane", repeats=2)
        expected = round(
            result["stable_prefix_estimated_tokens"] / result["prompt_estimated_tokens"], 4
        )
        assert result["estimated_provider_cache_hit_fraction"] == expected
        assert 0.0 < result["estimated_provider_cache_hit_fraction"] < 1.0
        assert result["stable_prefix_ratio"] == round(
            result["stable_prefix_chars"] / result["prompt_chars"], 4
        )

    def test_single_repeat_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="repeats must be >= 2"):
            run_cache_lane(tmp_path / "lane", repeats=1)


class TestRunner:
    def test_combined_runner_labels_every_lane_internal(self, tmp_path: Path) -> None:
        """The combined report and each lane must carry the internal label."""
        result = run_agent_experience_lanes(tmp_path, repeats=2)

        assert result["validation"] == "internal"
        assert result["version"] == AGENT_EXPERIENCE_LANES_VERSION
        assert set(result["lanes"]) == {"tool_adoption", "budget", "cache"}
        for lane in result["lanes"].values():
            assert lane["validation"] == "internal"

    def test_runner_supports_lane_selection(self, tmp_path: Path) -> None:
        result = run_agent_experience_lanes(tmp_path, lanes=("tool-adoption",))
        assert set(result["lanes"]) == {"tool_adoption"}

    def test_runner_rejects_unknown_lanes(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown agent-experience lane"):
            run_agent_experience_lanes(tmp_path, lanes=("tool-adoption", "forgetting"))
        with pytest.raises(ValueError, match="at least one"):
            run_agent_experience_lanes(tmp_path, lanes=())

    def test_lane_names_match_phase_one_scope(self) -> None:
        assert AGENT_EXPERIENCE_LANE_NAMES == ("tool-adoption", "budget", "cache")


class TestAgentExperienceLanesCli:
    def test_cli_parses_lane_selection(self) -> None:
        """The CLI lane parser should validate selection like other benchmark parsers."""
        valid = ("tool-adoption", "budget", "cache")
        assert cli_benchmarks._parse_agent_experience_lanes("all", valid) == valid
        assert cli_benchmarks._parse_agent_experience_lanes("cache,budget,cache", valid) == (
            "cache",
            "budget",
        )
        with pytest.raises(Exception, match="Unsupported agent-experience lane"):
            cli_benchmarks._parse_agent_experience_lanes("forgetting", valid)
        with pytest.raises(Exception, match="at least one lane"):
            cli_benchmarks._parse_agent_experience_lanes(",", valid)

    def test_cli_parses_budget_sweep(self) -> None:
        """The CLI budget parser should accept integers and 'unlimited'."""
        assert cli_benchmarks._parse_agent_experience_budgets("256,1024,unlimited") == (
            256,
            1024,
            None,
        )
        assert cli_benchmarks._parse_agent_experience_budgets("0,none") == (0, None)
        with pytest.raises(Exception, match="invalid budget"):
            cli_benchmarks._parse_agent_experience_budgets("256,not-a-number")
        with pytest.raises(Exception, match="must be >= 0"):
            cli_benchmarks._parse_agent_experience_budgets("-1")
        with pytest.raises(Exception, match="at least one budget"):
            cli_benchmarks._parse_agent_experience_budgets(",")

    def test_cli_tool_adoption_lane_prints_internal_json(self) -> None:
        """The subcommand should print one internally labeled JSON report."""
        result = CliRunner().invoke(app, ["agent-experience-lanes", "--lanes", "tool-adoption"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["validation"] == "internal"
        assert set(payload["lanes"]) == {"tool_adoption"}
        assert payload["lanes"]["tool_adoption"]["profiles"]["core"]["front_door_listed"] is True

    def test_cli_runs_all_lanes_and_writes_report(self, tmp_path: Path) -> None:
        """The full subcommand run should pass contracts and write the JSON report."""
        output_dir = tmp_path / "reports"
        result = CliRunner().invoke(
            app,
            [
                "agent-experience-lanes",
                "--repeats",
                "2",
                "--budgets",
                "256,2048,unlimited",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        report_path = output_dir / "agent-experience-lanes.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["validation"] == "internal"
        assert set(payload["lanes"]) == {"tool_adoption", "budget", "cache"}
        assert payload["lanes"]["budget"]["contract"]["status"] == "pass"
        assert payload["lanes"]["cache"]["contract"]["status"] == "pass"

    def test_cli_rejects_unknown_lane(self) -> None:
        result = CliRunner().invoke(app, ["agent-experience-lanes", "--lanes", "forgetting"])
        assert result.exit_code != 0
