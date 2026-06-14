"""Adversarial tests: encryption envelope, cryptographic erasure, capabilities (P3)."""

from __future__ import annotations

import pytest

from zaxy.portable.envelope import (
    Capability,
    ErasureVault,
    decrypt_cell,
    encrypt_cell,
    new_dek,
    unwrap_dek,
    wrap_dek,
)


def test_encrypt_decrypt_roundtrip() -> None:
    cell = encrypt_cell(b"secret memory")
    assert decrypt_cell(cell["ciphertext"], cell["nonce"], cell["dek"]) == b"secret memory"


def test_tampered_ciphertext_fails_auth() -> None:
    cell = encrypt_cell(b"secret memory")
    bad = ("ff" if cell["ciphertext"][:2] != "ff" else "00") + cell["ciphertext"][2:]
    with pytest.raises(Exception):  # noqa: B017 - GCM auth failure (InvalidTag)
        decrypt_cell(bad, cell["nonce"], cell["dek"])


def test_wrong_dek_fails() -> None:
    cell = encrypt_cell(b"secret memory")
    with pytest.raises(Exception):  # noqa: B017
        decrypt_cell(cell["ciphertext"], cell["nonce"], new_dek())


def test_dek_wrap_unwrap() -> None:
    dek, kek = new_dek(), new_dek()
    wrapped = wrap_dek(dek, kek)
    assert unwrap_dek(wrapped, kek) == dek
    with pytest.raises(Exception):  # noqa: B017 - wrong KEK
        unwrap_dek(wrapped, new_dek())


def test_cryptographic_erasure_destroys_key_not_data() -> None:
    kek = new_dek()
    cell = encrypt_cell(b"forget me")
    vault = ErasureVault()
    vault.store("c1", wrap_dek(cell["dek"], kek), content_address="addr-c1")
    # before erasure: key recoverable, decrypt works
    assert vault.get_dek("c1", kek) == cell["dek"]
    # erase: the wrapped DEK is destroyed
    vault.erase("c1", erased_at="2026-06-14T00:00:00Z")
    assert vault.is_erased("c1") is True
    assert vault.is_blacklisted("addr-c1") is True
    with pytest.raises(KeyError):
        vault.get_dek("c1", kek)  # key gone -> ciphertext permanently undecryptable


def test_capability_temporary_expiry() -> None:
    cap = Capability("cap1", "temporary", cells=("c1",), expires_at=100.0)
    assert cap.authorizes("c1", now=50.0) is True
    assert cap.authorizes("c1", now=150.0) is False  # expired


def test_capability_permanent_and_scope() -> None:
    cap = Capability("cap2", "permanent", cells=("c1", "c2"))
    assert cap.authorizes("c2", now=10_000.0) is True
    assert cap.authorizes("c3", now=10_000.0) is False  # out of scope


def test_capability_revocation_and_wildcard() -> None:
    cap = Capability("cap3", "syndicate", cells=("*",))
    assert cap.authorizes("anything", now=1.0) is True
    cap.revoked = True
    assert cap.authorizes("anything", now=1.0) is False
