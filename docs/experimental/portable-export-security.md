# Portable Export — Security Status & Review Requests

> ## ⚠️ EXPERIMENTAL · UNAUDITED
> The portable signed-export / rehydration / envelope code (`src/zaxy/portable/`)
> is a **reference implementation**. It has **not** had independent cryptographic
> review. **Do not rely on it to protect high-value secrets or for compliance
> guarantees.** No "audited" or "secure" claim is made. It lives on a feature
> branch and has not been released.

## What is and isn't assured

**Assured by design / tests:**
- Only **vetted primitives** (pyca/cryptography): ML-DSA-65, Ed25519, AES-256-GCM,
  SHA-256. No homegrown cryptographic primitives.
- Adversarial test suite (forge, tamper, drop, swapped-key, metadata-binding,
  subset non-leakage, forged-proof, GCM auth-fail, key-destruction erasure,
  capability expiry/revocation, anchor tamper-detection).
- An **adversarial self-review** of the protocol/composition logic
  (see `portable-export-conformance.md`). This is a *layer*, **not** a substitute
  for independent human review.

**Not assured (needs human review and/or later phases):**
- Protocol composition beyond the tested cases.
- Canonicalization is sorted-key compact JSON — not yet JCS (RFC 8785); floats
  are a known canonicalization hazard.
- KEK nonce management at high volume; wallet-derived KEK not implemented.
- The public anchor ships as an offline **stub**; real OpenTimestamps/public-chain
  anchoring is a hook, not wired. A stub anchor proves *consistency only*, not
  external timestamping.
- W3C interop conformance is deferred until the spec (CG proposed 2026-05-18)
  finalizes.

## What it provides (threat model summary)
- **Authenticity + integrity** of an exported memory bundle (signature binds the
  Merkle root and all metadata), with **self-sovereign** keys (no CA; pin the key
  or use a DID; trust-on-first-use).
- **Verifiable partial disclosure**: prove a subset of entries belongs to the
  signed set without revealing the rest.
- **Confidentiality + cryptographic erasure** of memory cells (per-cell DEK;
  destroy-the-key erasure). **Invariant:** key material must never enter the
  immutable Eventloom log, or erasure is void.
- **Injection-resistant rehydration**: defense-in-depth fencing of recalled
  untrusted content — explicitly **not** a guarantee against prompt injection.

**Non-goals (v1):** CA/PKI federation, threshold/multi-party signing, confidential
compute, replay/rollback prevention beyond per-bundle nonce + timestamp.

## Reporting issues
Found a flaw? Please open a private report (security contact / GitHub security
advisory) rather than a public issue, and allow reasonable time to fix. Since this
is pre-release experimental code, coordinated disclosure is appreciated but there
is no production deployment at risk.

---

## Review request — W3C AI Agent Memory Interoperability CG

> **Subject: Reference impl + review request — self-sovereign signed memory export (ML-DSA-65 + Merkle subset proofs)**
>
> Hi all — I've built an open, experimental reference implementation aligned with
> the CG's announced scope (identity binding via ML-DSA-65/FIPS-204, Merkle
> provenance with verifiable partial disclosure, per-cell encryption envelope,
> cryptographic erasure for GDPR Art.17, capability-scoped sharing). It's
> self-sovereign (no CA) with a public-chain anchor hook.
>
> It is **experimental and unaudited** — I'm a resource-constrained project and
> can't commission a formal audit, so I'd hugely value eyes on the *protocol/
> composition logic*: how the signature binds the Merkle root + metadata, the
> subset-disclosure proofs, and the crypto-erasure model in an append-only store.
> Vetted primitives only (pyca/cryptography); adversarial tests included.
>
> Repo + threat model: <link>. Happy to align field names/serialization to the
> spec as it stabilizes, and to contribute findings back. What have I gotten wrong?

## Review request — crypto.stackexchange

> **Title: Is this composition sound? Signed Merkle-root memory bundle with verifiable subset disclosure + crypto-erasure**
>
> I have an experimental protocol for portable, signed agent-memory bundles and
> want to sanity-check the *composition* (primitives are pyca: ML-DSA-65/Ed25519,
> AES-256-GCM, SHA-256 — I'm not rolling my own).
>
> 1. A bundle's signature is computed over a canonical header that includes the
>    Merkle root of the entries **plus** all metadata (version, alg, pubkey,
>    session id, created-at, nonce). Is binding metadata-with-root this way
>    sufficient to prevent metadata-swap / cross-bundle attacks?
> 2. Subset disclosure reveals chosen entries + Merkle inclusion proofs (leaf
>    prefix `0x00`, node prefix `0x01`, unpaired nodes promoted not duplicated).
>    Any forgery/second-preimage concern, given the proofs aren't position-bound?
> 3. "Cryptographic erasure" = destroy the (wrapped) DEK + tombstone + blacklist;
>    ciphertext is otherwise immutable. Are there pitfalls beyond "ensure no other
>    copy of the key survives"?
>
> Canonicalization is sorted-key compact JSON (I know JCS/RFC 8785 is the better
> answer). Threat model + code: <link>. Where does this break?
