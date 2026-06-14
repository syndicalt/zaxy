"""Portable, signed Zaxy memory export (dev target #3, Phase 1).

SECURITY-CRITICAL. Provides self-sovereign cryptographic *authenticity* (signing)
on top of Eventloom's existing hash-chain *integrity*, plus a Merkle tree enabling
verifiable PARTIAL disclosure of an exported memory subset.

Trust model: self-sovereign — the signer holds its own keypair; the public key is
the identity. No CA. Verifiers establish trust out-of-band (pin the public key /
DID, trust-on-first-use). Operator-independent timestamping (OpenTimestamps /
public-chain anchoring) is a later phase; the `anchor` bundle field is reserved.

NOT YET REVIEWED — Phase 1 reference implementation pending the independent
cryptographic review the goal (seq 78021) mandates before any release.
"""

import warnings

from zaxy.portable.export import (
    BUNDLE_VERSION,
    build_export,
    disclose_subset,
    verify_export,
    verify_subset,
)
from zaxy.portable.signing import default_algorithm, generate_keypair, mldsa_available

warnings.warn(
    "zaxy.portable is EXPERIMENTAL and UNAUDITED cryptographic code. Do not rely on it "
    "to protect high-value secrets or for compliance guarantees. "
    "See docs/experimental/portable-export-security.md.",
    category=UserWarning,
    stacklevel=2,
)

__all__ = [
    "BUNDLE_VERSION",
    "build_export",
    "disclose_subset",
    "verify_export",
    "verify_subset",
    "default_algorithm",
    "generate_keypair",
    "mldsa_available",
]
