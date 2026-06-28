"""Verified forgetting via cryptographic erasure (Zaxy 3 / I5b).

The Eventloom log is append-only and hash-chained, and an event's payload is
sealed *into* its hash (see :meth:`zaxy.event.Event.canonical` and
``to_eventloom_v1_unsigned``). A sealed plaintext payload therefore can never be
scrubbed without breaking :meth:`zaxy.event.EventLog.verify`. The only sound
"right to be forgotten" in such a store is **cryptographic erasure**:

1. At append time, a *forgettable* payload is encrypted to a single AES-256-GCM
   cell. The **ciphertext** is what gets sealed into the hash, so the hash is a
   hash of ciphertext and never changes when the plaintext is forgotten.
2. The per-cell data-encryption key (DEK) is wrapped under a key-encryption key
   (KEK) and stored ONLY in a mutable :class:`PersistentErasureVault` that lives
   *outside* the immutable log. Neither plaintext nor unwrapped keys ever touch
   the JSONL.
3. To forget, the wrapped DEK is destroyed (erase-by-key-discard) and an audited,
   cited ``memory.forgotten`` tombstone is appended. The on-disk ciphertext and
   its hash are untouched, so ``verify()`` stays green with zero changes to the
   hash computation, while the plaintext becomes permanently unrecoverable.

This module wraps the vetted primitives in :mod:`zaxy.portable.envelope`
(``encrypt_cell``/``decrypt_cell``/``wrap_dek``/``unwrap_dek``/``ErasureVault``);
it never reimplements the crypto.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zaxy.config import Settings
    from zaxy.portable.envelope import ErasureVault


def _new_erasure_vault() -> ErasureVault:
    # Lazy import: ``zaxy.portable`` emits an EXPERIMENTAL-crypto warning on import.
    # Forgetting is opt-in, so the crypto stack (and its warning) is only pulled
    # when a forgettable memory is actually written or read -- never on the
    # default plaintext path.
    from zaxy.portable.envelope import ErasureVault

    return ErasureVault()

#: Payload key carrying the encrypted cell of a forgettable memory. The presence
#: of this key in a sealed payload is the marker that the event is forgettable.
CIPHER_PAYLOAD_KEY = "__zaxy_cipher"
#: Versioned cell envelope so a consumer can pin the on-disk cipher contract.
CIPHER_VERSION = 1
CIPHER_ALGORITHM = "AES-256-GCM"

#: Payload key flagging a decrypted view as a forgotten (erased) memory.
FORGOTTEN_MARKER_KEY = "__zaxy_forgotten"
#: Human-facing sentinel text surfaced by viewers/exports for a forgotten memory.
FORGOTTEN_TEXT = "[FORGOTTEN]"

#: Event type for the audited, cited cryptographic-erasure tombstone.
MEMORY_FORGOTTEN_EVENT_TYPE = "memory.forgotten"
_AUTHORITY_STATUS = "non_authoritative"

#: Files persisted beside the Eventloom logs (never inside the append-only JSONL).
VAULT_FILENAME = "__erasure_vault__.json"
KEK_FILENAME = "__erasure_kek__.key"
_VAULT_VERSION = 1
_KEK_BYTES = 32  # AES-256
_SECRET_FILE_MODE = 0o600

_HASH_RE_LEN = 64


def forgotten_sentinel() -> dict[str, Any]:
    """Return a fresh ``[FORGOTTEN]`` sentinel payload for an erased memory.

    A *fresh* dict each call so callers can never mutate shared state. The
    ``content`` field makes viewers/exports render ``[FORGOTTEN]`` and the
    marker key lets retrieval explicitly exclude the forgotten memory.
    """
    return {"content": FORGOTTEN_TEXT, FORGOTTEN_MARKER_KEY: True}


def _write_secret_file(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` with 0600 permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SECRET_FILE_MODE)
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    os.chmod(path, _SECRET_FILE_MODE)


class PersistentErasureVault:
    """A durable wrapped-DEK store for cryptographic erasure.

    Composes the in-memory :class:`~zaxy.portable.envelope.ErasureVault`
    (erase-by-key-discard semantics) and persists its state to a mutable JSON
    file beside the Eventloom logs. The DEK of each forgettable cell is wrapped
    under a KEK that lives in a separate 0600 key file (dev default; a production
    deployment SHOULD point ``kek_path`` at a KMS/secret-managed key — see
    ``docs/forgetting.md``).

    CRITICAL INVARIANT: only *wrapped* DEKs and ciphertext content addresses are
    ever written to disk here, and never to the append-only log. Plaintext and
    unwrapped keys never leave memory. Erasing a cell pops its only wrapped-DEK
    copy, so the immutable ciphertext in the log becomes permanently
    undecryptable.
    """

    def __init__(self, vault_path: str | Path, *, kek_path: str | Path) -> None:
        self._vault_path = Path(vault_path)
        self._kek_path = Path(kek_path)
        self._ev = _new_erasure_vault()
        self._wrapped: dict[str, dict[str, Any]] = {}
        self._tombstones: dict[str, dict[str, Any]] = {}
        self._blacklist: set[str] = set()
        self._kek: bytes | None = None
        self._loaded = False

    @classmethod
    def for_eventloom_dir(
        cls, base: str | Path, *, kek_path: str | Path | None = None
    ) -> PersistentErasureVault:
        """Build a vault rooted at an Eventloom directory.

        The wrapped-DEK store lives at ``<base>/__erasure_vault__.json`` and the
        KEK defaults to ``<base>/__erasure_kek__.key`` unless ``kek_path``
        overrides it (production: a KMS-managed key path).
        """
        base_path = Path(base)
        resolved_kek = Path(kek_path) if kek_path else base_path / KEK_FILENAME
        return cls(base_path / VAULT_FILENAME, kek_path=resolved_kek)

    # -- persistence ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._vault_path.exists():
            return
        data = json.loads(self._vault_path.read_text(encoding="utf-8"))
        for cell_id, entry in dict(data.get("wrapped", {})).items():
            self._wrapped[cell_id] = entry
            self._ev.store(cell_id, entry["wrapped"], entry.get("address", cell_id))
        for cell_id, tomb in dict(data.get("tombstones", {})).items():
            self._tombstones[cell_id] = tomb
            self._ev.erase(cell_id, erased_at=str(tomb.get("erased_at", "")))
        self._blacklist = set(data.get("blacklist", []))

    def _persist(self) -> None:
        payload = {
            "version": _VAULT_VERSION,
            "wrapped": self._wrapped,
            "tombstones": self._tombstones,
            "blacklist": sorted(self._blacklist),
        }
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._vault_path.with_name(f"{self._vault_path.name}.{os.getpid()}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SECRET_FILE_MODE)
        try:
            os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self._vault_path)
        os.chmod(self._vault_path, _SECRET_FILE_MODE)

    def _load_kek(self) -> bytes:
        if self._kek is not None:
            return self._kek
        if self._kek_path.exists():
            kek = bytes.fromhex(self._kek_path.read_text(encoding="utf-8").strip())
        else:
            from zaxy.portable.envelope import new_dek

            kek = new_dek()
            _write_secret_file(self._kek_path, kek.hex())
        if len(kek) != _KEK_BYTES:
            raise ValueError(f"KEK at {self._kek_path} must be {_KEK_BYTES} bytes (256-bit)")
        self._kek = kek
        return kek

    # -- public API ----------------------------------------------------

    def store_wrapped(self, cell_id: str, dek: bytes, content_address: str) -> None:
        """Wrap ``dek`` under the KEK and persist it keyed by ``cell_id``."""
        self._ensure_loaded()
        from zaxy.portable.envelope import wrap_dek

        wrapped = wrap_dek(dek, self._load_kek())
        self._wrapped[cell_id] = {"wrapped": wrapped, "address": content_address}
        self._ev.store(cell_id, wrapped, content_address)
        self._persist()

    def get_dek(self, cell_id: str) -> bytes | None:
        """Return the unwrapped DEK, or ``None`` if erased or never stored."""
        self._ensure_loaded()
        if self._ev.is_erased(cell_id) or cell_id not in self._wrapped:
            return None
        from zaxy.portable.envelope import unwrap_dek

        return unwrap_dek(self._wrapped[cell_id]["wrapped"], self._load_kek())

    def erase(self, cell_id: str, *, erased_at: str) -> bool:
        """Destroy the wrapped DEK for ``cell_id`` (cryptographic erasure).

        Returns whether a live wrapped DEK was present and destroyed. After this
        the cell's ciphertext in the log is permanently undecryptable.
        """
        self._ensure_loaded()
        entry = self._wrapped.pop(cell_id, None)
        address = entry["address"] if entry else self._tombstones.get(cell_id, {}).get("address")
        self._tombstones[cell_id] = {"erased_at": erased_at, "address": address}
        if address:
            self._blacklist.add(address)
        self._ev.erase(cell_id, erased_at=erased_at)
        self._persist()
        return entry is not None

    def is_erased(self, cell_id: str) -> bool:
        """Return whether ``cell_id`` has been cryptographically erased."""
        self._ensure_loaded()
        return self._ev.is_erased(cell_id)

    def is_blacklisted(self, content_address: str) -> bool:
        """Return whether a ciphertext content address is erasure-blacklisted."""
        self._ensure_loaded()
        return content_address in self._blacklist


def build_vault(settings: Settings, eventloom_path: str | Path) -> PersistentErasureVault | None:
    """Return the gated erasure vault for an Eventloom dir, or ``None`` when off.

    The single shared way the read/export surfaces obtain a wrapped-DEK store, so
    the CLI export, the push sink, and the MCP export handler never grow a second
    convention. When ``settings.forgetting_enabled`` is False the result is
    ``None`` -- a harmless no-op: :func:`decrypt_payload` / the export projection
    leave plaintext events untouched, so a forgetting-disabled export stays
    byte-identical. When enabled, the vault is rooted at ``eventloom_path`` with
    the configured KEK, exactly as :class:`SessionManager` builds it.
    """
    if not settings.forgetting_enabled:
        return None
    return PersistentErasureVault.for_eventloom_dir(
        eventloom_path, kek_path=settings.forgetting_kek_path
    )


def encrypt_forgettable_payload(
    payload: dict[str, Any], *, vault: PersistentErasureVault
) -> dict[str, Any]:
    """Encrypt a whole payload to one cipher cell and stash its wrapped DEK.

    Returns a sealed payload of the form ``{"__zaxy_cipher": {...}}`` whose only
    content is ciphertext; this is what gets hash-sealed at append time. The DEK
    is wrapped and stored in ``vault`` keyed by the cell id, never in the log.
    """
    from zaxy.portable.envelope import encrypt_cell

    plaintext = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cell = encrypt_cell(plaintext)
    cell_id = uuid4().hex
    content_address = hashlib.sha256(bytes.fromhex(cell["ciphertext"])).hexdigest()
    vault.store_wrapped(cell_id, cell["dek"], content_address)
    return {
        CIPHER_PAYLOAD_KEY: {
            "v": CIPHER_VERSION,
            "alg": CIPHER_ALGORITHM,
            "cell_id": cell_id,
            "nonce": cell["nonce"],
            "ciphertext": cell["ciphertext"],
            "content_address": content_address,
        }
    }


def cipher_cell(payload: Any) -> dict[str, Any] | None:
    """Return the ``__zaxy_cipher`` cell of a payload, or ``None`` if plaintext."""
    if not isinstance(payload, dict):
        return None
    cell = payload.get(CIPHER_PAYLOAD_KEY)
    return cell if isinstance(cell, dict) else None


def decrypt_payload(
    payload: dict[str, Any], *, vault: PersistentErasureVault | None
) -> dict[str, Any]:
    """Return the plaintext payload, the ``[FORGOTTEN]`` sentinel, or a passthrough.

    - No ``__zaxy_cipher`` cell -> the payload is plaintext; returned unchanged.
    - Cell present and its DEK still lives in ``vault`` -> decrypt to the original
      plaintext payload.
    - Cell present but the DEK is erased/missing (or ``vault`` is unavailable) ->
      the ``[FORGOTTEN]`` sentinel: the plaintext is permanently unrecoverable.
    """
    cell = cipher_cell(payload)
    if cell is None:
        return payload
    cell_id = cell.get("cell_id")
    nonce = cell.get("nonce")
    ciphertext = cell.get("ciphertext")
    if not (isinstance(cell_id, str) and isinstance(nonce, str) and isinstance(ciphertext, str)):
        return forgotten_sentinel()
    if vault is None:
        return forgotten_sentinel()
    try:
        # Resolve + decrypt under ONE guard: a memory whose DEK can no longer be
        # unwrapped (rotated/replaced/lost or length-invalid KEK, corrupted
        # wrapped-DEK blob) is effectively forgotten, never a crash. ``get_dek``
        # raises InvalidTag/ValueError in those cases, so we degrade to the same
        # ``[FORGOTTEN]`` sentinel as the erased path and keep every read/checkout
        # /export alive instead of propagating the exception up the read stack.
        dek = vault.get_dek(cell_id)
        if dek is None:
            return forgotten_sentinel()
        from zaxy.portable.envelope import decrypt_cell

        plaintext = decrypt_cell(ciphertext, nonce, dek)
        decoded = json.loads(plaintext.decode("utf-8"))
    except Exception:
        return forgotten_sentinel()
    return decoded if isinstance(decoded, dict) else forgotten_sentinel()


def is_forgotten_payload(payload: Any) -> bool:
    """Return whether a (decrypted) payload is the forgotten sentinel."""
    return isinstance(payload, dict) and bool(payload.get(FORGOTTEN_MARKER_KEY))


def _validate_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _snapshot_target(target: dict[str, Any]) -> dict[str, Any]:
    seq = target.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ValueError("target seq must be a positive integer")
    event_hash = target.get("hash")
    if not isinstance(event_hash, str) or len(event_hash) != _HASH_RE_LEN:
        raise ValueError("target hash must be a 64-character hex digest")
    return {"seq": seq, "hash": event_hash}


def _forget_id(target: dict[str, Any], cell_id: str, reason: str) -> str:
    identity = {"target": dict(target), "cell_id": cell_id, "reason": reason}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"forget:{digest}"


def build_memory_forgotten_event(
    *,
    actor: str,
    session_id: str,
    target: dict[str, Any],
    cell_id: str,
    reason: str,
) -> dict[str, Any]:
    """Build a non-authoritative, cited ``memory.forgotten`` tombstone spec.

    Cites the forgotten memory via ``target`` ({seq, 64-hex hash}) and its
    ``cell_id``, carries the reason and a deterministic ``forget_id``. The sealed
    memory event is never mutated; this is a new, additive audit record of the
    cryptographic erasure.
    """
    _validate_non_empty_string(actor, field_name="actor")
    _validate_non_empty_string(session_id, field_name="session_id")
    _validate_non_empty_string(cell_id, field_name="cell_id")
    _validate_non_empty_string(reason, field_name="reason")
    snapshot = _snapshot_target(target)
    payload: dict[str, Any] = {
        "forget_id": _forget_id(snapshot, cell_id, reason),
        "target": snapshot,
        "cell_id": cell_id,
        "reason": reason,
        "authority_status": _AUTHORITY_STATUS,
    }
    return {
        "event_type": MEMORY_FORGOTTEN_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": session_id,
    }
