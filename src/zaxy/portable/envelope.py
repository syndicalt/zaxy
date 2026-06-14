"""Encryption envelope, cryptographic erasure, and capability contracts (P3).

SECURITY-CRITICAL reference implementation. Uses ONLY vetted pyca/cryptography
primitives (AES-256-GCM); HELD FOR INDEPENDENT REVIEW before any release.

- Envelope: each memory cell is encrypted under its own random DEK (data
  encryption key); the DEK is wrapped by a KEK (key encryption key). Confidential
  per-cell; rotate by re-wrapping.
- Cryptographic erasure (GDPR Art.17 in an append-only world): destroy the wrapped
  DEK + write a tombstone + blacklist the content address. The (immutable)
  ciphertext remains but is permanently undecryptable.
- Capability contracts: scoped, revocable, optionally-expiring grants
  (temporary / permanent / syndicate).

Timestamps (`now`, `expires_at`) are passed in explicitly for deterministic,
testable behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_NONCE = 12  # AES-GCM standard nonce length


def new_dek() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def encrypt_cell(plaintext: bytes, dek: bytes | None = None) -> dict[str, Any]:
    key = dek or new_dek()
    nonce = os.urandom(_NONCE)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return {"dek": key, "nonce": nonce.hex(), "ciphertext": ciphertext.hex()}


def decrypt_cell(ciphertext_hex: str, nonce_hex: str, dek: bytes) -> bytes:
    return AESGCM(dek).decrypt(bytes.fromhex(nonce_hex), bytes.fromhex(ciphertext_hex), None)


def wrap_dek(dek: bytes, kek: bytes) -> dict[str, str]:
    nonce = os.urandom(_NONCE)
    return {"nonce": nonce.hex(), "wrapped": AESGCM(kek).encrypt(nonce, dek, None).hex()}


def unwrap_dek(wrapped: dict[str, str], kek: bytes) -> bytes:
    return AESGCM(kek).decrypt(bytes.fromhex(wrapped["nonce"]), bytes.fromhex(wrapped["wrapped"]), None)


class ErasureVault:
    """Holds wrapped DEKs; cryptographic erasure destroys the key, not the data."""

    def __init__(self) -> None:
        self._wrapped: dict[str, dict[str, Any]] = {}
        self._tombstones: dict[str, dict[str, Any]] = {}
        self._blacklist: set[str] = set()

    def store(self, cell_id: str, wrapped_dek: dict[str, str], content_address: str) -> None:
        self._wrapped[cell_id] = {"wrapped": wrapped_dek, "address": content_address}

    def get_dek(self, cell_id: str, kek: bytes) -> bytes:
        if cell_id in self._tombstones:
            raise KeyError(f"cell {cell_id} cryptographically erased")
        return unwrap_dek(self._wrapped[cell_id]["wrapped"], kek)

    def erase(self, cell_id: str, *, erased_at: str) -> None:
        entry = self._wrapped.pop(cell_id, None)  # destroy the only copy of the wrapped DEK
        address = entry["address"] if entry else None
        self._tombstones[cell_id] = {"erased_at": erased_at, "address": address}
        if address:
            self._blacklist.add(address)

    def is_erased(self, cell_id: str) -> bool:
        return cell_id in self._tombstones

    def is_blacklisted(self, content_address: str) -> bool:
        return content_address in self._blacklist


@dataclass
class Capability:
    """A scoped sharing grant. `cells` may contain '*' for all."""

    cap_id: str
    kind: str  # "temporary" | "permanent" | "syndicate"
    cells: tuple[str, ...] = field(default_factory=tuple)
    expires_at: float | None = None  # epoch seconds; None = no expiry
    revoked: bool = False

    def authorizes(self, cell_id: str, *, now: float) -> bool:
        if self.revoked:
            return False
        if self.kind == "temporary" and self.expires_at is not None and now > self.expires_at:
            return False
        return "*" in self.cells or cell_id in self.cells
