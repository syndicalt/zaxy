"""Tests for deterministic token estimation and monotone section packing."""

from __future__ import annotations

import pytest

from zaxy.token_budget import BudgetSection, estimate_tokens, pack_sections


def _section(
    section_id: str,
    text: str,
    *,
    weight: float = 0.5,
    mandatory: bool = False,
    kind: str | None = None,
) -> BudgetSection:
    return BudgetSection(
        section_id=section_id,
        kind=kind or section_id,
        text=text,
        weight=weight,
        mandatory=mandatory,
    )


_MIXED_SECTIONS = [
    _section("header", "# Memory Checkout", mandatory=True),
    _section("facts", "- fact one\n- fact two\n- fact three", weight=1.0),
    _section("evidence", "- eventloom://s/events/1#abc: fact one", weight=0.95),
    _section("diagnostics", "- Source lanes: graph=3\n- Citations: 1\n- Current facts: 3", weight=0.3),
    _section("guidance", "- Trust: use current facts", mandatory=True),
    _section("retrieved", "- retrieved context line with quite a lot of extra words", weight=0.7),
]


def test_estimate_tokens_is_exact_ceil_chars_over_four() -> None:
    """The estimator must be exactly ceil(chars / 4) with no provider calls."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 8) == 2
    assert estimate_tokens("x" * 9) == 3
    assert estimate_tokens("x" * 4001) == 1001


def test_pack_sections_is_deterministic() -> None:
    """The same sections and budget must always produce the same packing."""
    first = pack_sections(_MIXED_SECTIONS, 20)
    second = pack_sections(_MIXED_SECTIONS, 20)

    assert first == second


def test_pack_sections_preserves_original_section_order() -> None:
    """Packing selects by weight-per-token but renders in caller order."""
    result = pack_sections(_MIXED_SECTIONS, 10_000)

    assert [section.section_id for section in result.sections] == [
        section.section_id for section in _MIXED_SECTIONS
    ]
    assert result.elided == []


def test_pack_sections_is_monotone_in_budget() -> None:
    """Raising the budget must never remove a previously included section."""
    previous_ids: set[str] = set()
    for budget in range(0, sum(estimate_tokens(s.text) for s in _MIXED_SECTIONS) + 2):
        result = pack_sections(_MIXED_SECTIONS, budget)
        included_ids = {section.section_id for section in result.sections}
        assert previous_ids <= included_ids, f"budget {budget} dropped {previous_ids - included_ids}"
        previous_ids = included_ids


def test_pack_sections_zero_budget_keeps_mandatory_and_reports_elisions() -> None:
    """A zero budget yields a mandatory-only packet with explicit elision records."""
    result = pack_sections(_MIXED_SECTIONS, 0)

    assert [section.section_id for section in result.sections] == ["header", "guidance"]
    assert result.budget_requested == 0
    assert result.budget_used == estimate_tokens("# Memory Checkout") + estimate_tokens(
        "- Trust: use current facts"
    )
    elided_by_id = {record.section_id: record for record in result.elided}
    assert set(elided_by_id) == {"facts", "evidence", "diagnostics", "retrieved"}
    for section in _MIXED_SECTIONS:
        if section.mandatory:
            continue
        record = elided_by_id[section.section_id]
        assert record.kind == section.kind
        assert record.estimated_tokens == estimate_tokens(section.text)


def test_pack_sections_elision_records_match_excluded_sections() -> None:
    """Elision records carry the id, kind, and exact token estimate of each cut."""
    sections = [
        _section("keep", "tiny", weight=1.0),
        _section("cut", "x" * 400, weight=0.1, kind="diagnostics"),
    ]

    result = pack_sections(sections, estimate_tokens("tiny"))

    assert [section.section_id for section in result.sections] == ["keep"]
    assert len(result.elided) == 1
    assert result.elided[0].section_id == "cut"
    assert result.elided[0].kind == "diagnostics"
    assert result.elided[0].estimated_tokens == 100
    assert result.elided[0].to_dict() == {
        "section_id": "cut",
        "kind": "diagnostics",
        "estimated_tokens": 100,
    }


def test_pack_sections_prefers_weight_per_token_density() -> None:
    """A small high-weight section wins over a large low-weight section."""
    sections = [
        _section("large_low", "x" * 200, weight=0.4),
        _section("small_high", "y" * 20, weight=0.9),
    ]

    result = pack_sections(sections, estimate_tokens("y" * 20))

    assert [section.section_id for section in result.sections] == ["small_high"]


def test_pack_sections_mandatory_overflow_still_includes_mandatory() -> None:
    """Mandatory sections survive even when they alone exceed the budget."""
    sections = [
        _section("contract", "z" * 100, mandatory=True),
        _section("optional", "tiny"),
    ]

    result = pack_sections(sections, 5)

    assert [section.section_id for section in result.sections] == ["contract"]
    assert result.budget_used == 25
    assert result.budget_used > result.budget_requested
    assert [record.section_id for record in result.elided] == ["optional"]


def test_pack_sections_rejects_negative_budget() -> None:
    """Negative budgets are caller errors, not silent empty packets."""
    with pytest.raises(ValueError, match="max_tokens must be >= 0"):
        pack_sections(_MIXED_SECTIONS, -1)


def test_pack_sections_budget_used_counts_every_included_section() -> None:
    """budget_used is the exact estimated total of the packed sections."""
    result = pack_sections(_MIXED_SECTIONS, 10_000)

    assert result.budget_used == sum(estimate_tokens(section.text) for section in _MIXED_SECTIONS)
    assert result.budget_requested == 10_000
