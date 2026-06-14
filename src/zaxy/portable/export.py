"""Signed, verifiable memory export bundles with verifiable partial disclosure.

A bundle binds, under one signature, the Merkle root of the exported entries AND
all bundle metadata (version, algorithm, public key, session, timestamp, nonce) —
so neither the entries nor the metadata can be altered without breaking
verification. `disclose_subset` reveals only chosen entries plus Merkle inclusion
proofs, letting a recipient verify those entries are authentic members of the
signed set WITHOUT seeing the undisclosed ones.

Verification proves "signed by the holder of this public key, untampered". Trust
that the public key belongs to the expected party is established out-of-band
(pinning / DID); see `verify_export(..., expect_public_key=...)`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from zaxy.portable import merkle, signing

BUNDLE_VERSION = "zaxy.portable.v1"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def entry_id(content: Any) -> str:
    return hashlib.sha256(_canonical(content)).hexdigest()


def _signing_header(b: dict[str, Any]) -> bytes:
    # Exactly the fields the signature binds. Order-independent (canonical sort).
    return _canonical(
        {
            "version": b["version"],
            "algorithm": b["algorithm"],
            "public_key": b["public_key"],
            "session_id": b["session_id"],
            "created_at": b["created_at"],
            "nonce": b["nonce"],
            "merkle_root": b["merkle_root"],
        }
    )


def build_export(
    entries: list[dict[str, Any]],
    *,
    keypair: dict[str, Any],
    session_id: str,
    created_at: str,
    nonce: str,
) -> dict[str, Any]:
    """Build a signed bundle from content entries. `created_at`/`nonce` are caller-
    supplied (deterministic + anti-replay); each export should use a fresh nonce."""
    # Deep snapshot: the bundle must own its content, not alias the caller's
    # mutable data (correctness + an export is an immutable record).
    contents = [copy.deepcopy(e) for e in entries]
    leaves = [merkle.leaf_hash(_canonical(c)) for c in contents]
    root = merkle.merkle_root(leaves)
    bundle: dict[str, Any] = {
        "version": BUNDLE_VERSION,
        "algorithm": keypair["algorithm"],
        "public_key": keypair["public_key"].hex(),
        "session_id": session_id,
        "created_at": created_at,
        "nonce": nonce,
        "merkle_root": root.hex(),
        "anchor": None,  # reserved for OpenTimestamps / public-chain anchor (later phase)
        "entries": [{"id": entry_id(c), "content": c} for c in contents],
    }
    bundle["signature"] = signing.sign(keypair["private_pem"], _signing_header(bundle)).hex()
    return bundle


def verify_export(bundle: dict[str, Any], *, expect_public_key: str | None = None) -> dict[str, Any]:
    """Verify integrity + authenticity. If `expect_public_key` (hex) is given, also
    enforce that the bundle was signed by that pinned key (trust establishment)."""
    try:
        if expect_public_key is not None and bundle.get("public_key") != expect_public_key:
            return {"ok": False, "reason": "public key does not match the pinned/expected key"}
        contents = [e["content"] for e in bundle["entries"]]
        for e, c in zip(bundle["entries"], contents, strict=True):
            if e["id"] != entry_id(c):
                return {"ok": False, "reason": "entry id does not match its content (tampered)"}
        root = merkle.merkle_root([merkle.leaf_hash(_canonical(c)) for c in contents])
        if root.hex() != bundle["merkle_root"]:
            return {"ok": False, "reason": "merkle root mismatch (entries added/removed/altered)"}
        ok = signing.verify(
            bundle["algorithm"],
            bytes.fromhex(bundle["public_key"]),
            bytes.fromhex(bundle["signature"]),
            _signing_header(bundle),
        )
        return {"ok": ok, "reason": None if ok else "signature invalid (forged, tampered, or wrong key)"}
    except (KeyError, ValueError, TypeError) as exc:
        return {"ok": False, "reason": f"malformed bundle: {exc}"}


def disclose_subset(bundle: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    """Produce a partial disclosure: only the chosen entries + inclusion proofs.
    Undisclosed entry contents are NOT included (only their hashes, via proofs)."""
    contents = [e["content"] for e in bundle["entries"]]
    leaves = [merkle.leaf_hash(_canonical(c)) for c in contents]
    disclosed = []
    for i in indices:
        proof = merkle.inclusion_proof(leaves, i)
        disclosed.append(
            {"index": i, "content": contents[i], "proof": [[s.hex(), side] for s, side in proof]}
        )
    return {
        **{k: bundle[k] for k in
           ("version", "algorithm", "public_key", "session_id", "created_at", "nonce",
            "merkle_root", "signature")},
        "disclosed": disclosed,
    }


def verify_subset(subset: dict[str, Any], *, expect_public_key: str | None = None) -> dict[str, Any]:
    """Verify a partial disclosure: the signature over root+metadata, then each
    disclosed entry's Merkle inclusion against the signed root."""
    try:
        if expect_public_key is not None and subset.get("public_key") != expect_public_key:
            return {"ok": False, "reason": "public key does not match the pinned/expected key"}
        if not signing.verify(
            subset["algorithm"],
            bytes.fromhex(subset["public_key"]),
            bytes.fromhex(subset["signature"]),
            _signing_header(subset),
        ):
            return {"ok": False, "reason": "signature invalid"}
        root = bytes.fromhex(subset["merkle_root"])
        for d in subset["disclosed"]:
            leaf = merkle.leaf_hash(_canonical(d["content"]))
            proof = [(bytes.fromhex(s), side) for s, side in d["proof"]]
            if not merkle.verify_inclusion(leaf, proof, root):
                return {"ok": False, "reason": f"inclusion proof failed for index {d['index']}"}
        return {"ok": True, "reason": None}
    except (KeyError, ValueError, TypeError) as exc:
        return {"ok": False, "reason": f"malformed subset: {exc}"}
