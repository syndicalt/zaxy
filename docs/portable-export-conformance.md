# Portable Export — W3C Interop Conformance Mapping (dev target #3)

Status: **reference implementation, held for independent review. Not released.**

This maps Zaxy's signed portable-export implementation to the scope of the W3C
"AI Agent Memory Interoperability" Community Group (proposed 2026-05-18). **Full
conformance is intentionally deferred until the CG spec is finalized** — you
cannot conform to an unfinalized standard. The pieces below are built against the
*announced scope* and will be reconciled to exact field names / serialization once
the spec stabilizes.

| W3C CG scope item | Zaxy implementation | Status |
|---|---|---|
| Identity binding (post-quantum signatures, ML-DSA-65 / FIPS-204 reference) | `portable/signing.py` — ML-DSA-65 primary, Ed25519 classical fallback (vetted pyca) | ✅ built |
| Tamper-evident provenance | `portable/merkle.py` + `export.py` — domain-separated Merkle root, signed; verifiable subset disclosure | ✅ built |
| Memory cell shape (encrypted unit + metadata) | `portable/envelope.py` — per-cell AES-256-GCM | ⚠️ primitive built; canonical cell schema pending spec |
| Encryption envelope (per-cell DEK; KEK wrap; rotation) | `portable/envelope.py` — DEK + KEK wrap (local KEK; wallet-derived KEK deferred) | ⚠️ partial |
| Cryptographic erasure (DEK destroy + tombstone + content-address blacklist; GDPR Art.17) | `portable/envelope.py::ErasureVault` | ✅ built |
| Sharing contracts (temporary / permanent / syndicate + revocation) | `portable/envelope.py::Capability` | ✅ built (policy layer) |
| Audit anchor (public-chain receipts; operator-independent verifiability) | `portable/anchor.py` — pluggable; deterministic stub + OpenTimestamps hook | ⚠️ interface built; real public-chain anchor not wired |
| Serialization (JSON-first, optional CBOR) | JSON bundle implemented | ⚠️ CBOR deferred |

## Trust model (no external authority required)
Self-sovereign: the exporter holds its own keypair; the public key (or a DID) is
the identity. No CA. Verifiers pin the key / trust-on-first-use. Operator-
independent verification is provided by the public-chain anchor (OpenTimestamps),
not a private signing authority. A managed "notary" service is a future product
layer, never a dependency.

## Review gate
This is security-critical reference code using vetted primitives only. It must
pass independent cryptographic review before any release. Nothing here has been
published.
