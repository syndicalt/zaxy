"""Tests for the public-anchor interface (P4)."""

from __future__ import annotations

from zaxy.portable.anchor import anchor_bundle, bundle_commitment, verify_anchor

_BUNDLE = {"merkle_root": "aa" * 32, "signature": "bb" * 8, "public_key": "cc" * 16, "anchor": None}


def test_anchor_sets_commitment_and_verifies() -> None:
    anchored = anchor_bundle(_BUNDLE)
    assert anchored["anchor"]["commitment"] == bundle_commitment(_BUNDLE)
    assert verify_anchor(anchored) is True


def test_tamper_after_anchor_detected() -> None:
    anchored = anchor_bundle(_BUNDLE)
    anchored["merkle_root"] = "00" * 32  # alter signed core after anchoring
    assert verify_anchor(anchored) is False


def test_missing_anchor_is_unverified() -> None:
    assert verify_anchor(_BUNDLE) is False


def test_pluggable_anchor_fn() -> None:
    def fake(commitment: str) -> dict:
        return {"type": "opentimestamps", "commitment": commitment, "ots": "deadbeef"}

    anchored = anchor_bundle(_BUNDLE, anchor_fn=fake)
    assert anchored["anchor"]["type"] == "opentimestamps"
    assert verify_anchor(anchored) is True
