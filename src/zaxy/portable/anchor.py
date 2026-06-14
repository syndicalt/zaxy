"""Public-anchor interface for export bundles (P4).

Operator-independent verifiability: commit a bundle's signed core to an external
timestamp/ledger. The default is a deterministic OFFLINE STUB (no network); a real
OpenTimestamps / public-chain anchor is a pluggable hook (intentionally not run
here). Full W3C-spec conformance is blocked on the unfinalized standard (CG
proposed 2026-05-18) -- see docs/portable-export-conformance.md.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any


def bundle_commitment(bundle: dict[str, Any]) -> str:
    """A stable commitment over the bundle's signed core (root + signature + key)."""
    h = hashlib.sha256()
    for key in ("merkle_root", "signature", "public_key"):
        h.update(str(bundle[key]).encode("utf-8"))
    return h.hexdigest()


def stub_anchor(commitment: str) -> dict[str, Any]:
    return {
        "type": "stub",
        "commitment": commitment,
        "note": "offline stub; replace with OpenTimestamps / public-chain anchor",
    }


def anchor_bundle(
    bundle: dict[str, Any], anchor_fn: Callable[[str], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Return a copy of `bundle` with its `anchor` field set to an anchor receipt."""
    receipt = (anchor_fn or stub_anchor)(bundle_commitment(bundle))
    out = dict(bundle)
    out["anchor"] = receipt
    return out


def verify_anchor(bundle: dict[str, Any]) -> bool:
    """True if the bundle carries an anchor whose commitment matches the bundle."""
    anchor = bundle.get("anchor")
    if not isinstance(anchor, dict):
        return False
    return anchor.get("commitment") == bundle_commitment(bundle)
