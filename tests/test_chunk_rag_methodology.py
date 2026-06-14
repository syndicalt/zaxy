"""Tests for the chunk-RAG token-reduction methodology (dev target #2).

These assert the METHODOLOGY is sound (equal-recall enforced; structured beats
raw when a fact is buried in noise; baseline pinned) -- not a specific headline
number, which requires a real QA dataset + the gated apparatus.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chunk_rag_token_compare import _self_test_corpus, compare_case, compare_corpus  # noqa: E402


def test_equal_recall_excludes_cases_where_an_arm_misses_gold() -> None:
    # gold appears in raw text but in NO structured fact -> structured can't reach it
    r = compare_case(
        query="where is the gold",
        gold="UNIQUEGOLD",
        facts=["[note] nothing relevant here", "[note] also irrelevant"],
        raw_text="filler filler UNIQUEGOLD filler",
    )
    assert r.both_reached is False
    assert r.reduction is None  # excluded from the ratio -> not gameable by asymmetric recall


def test_structured_beats_raw_when_fact_is_buried() -> None:
    out = compare_corpus(_self_test_corpus())
    assert out["cases_equal_recall"] == out["cases_total"]  # both arms find every gold
    # at equal recall, the compact cited card costs fewer tokens than chunk-RAG over noise
    assert out["median_reduction"] is not None and out["median_reduction"] > 0
    for case in out["per_case"]:
        assert case["structured_tok"] < case["chunkrag_tok"]


def test_baseline_is_pinned_and_reported() -> None:
    out = compare_corpus(_self_test_corpus(), chunk_tokens=256)
    assert out["baseline"]["chunk_tokens"] == 256
    assert "keyword" in out["baseline"]["ranking"]


def test_reduction_is_quality_controlled_definition() -> None:
    # reduction is defined only over equal-recall cases; a no-recall corpus yields None
    out = compare_corpus([
        {"query": "q", "gold": "ZZZ", "facts": ["irrelevant"], "raw_text": "also irrelevant"}
    ])
    assert out["cases_equal_recall"] == 0
    assert out["median_reduction"] is None
