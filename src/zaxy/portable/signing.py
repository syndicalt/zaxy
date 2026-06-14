"""Self-sovereign signer: post-quantum ML-DSA-65 primary, Ed25519 classical fallback.

Uses vetted pyca/cryptography primitives only (no homegrown crypto). ML-DSA needs
cryptography >= 46; when unavailable we fall back to Ed25519 (always present via
PyJWT[crypto]). Private keys are persisted as PKCS8 PEM (uniform across algs);
public keys as raw bytes (compact identity).
"""

from __future__ import annotations

from typing import Any

from cryptography.hazmat.primitives import serialization as _ser
from cryptography.hazmat.primitives.asymmetric import ed25519

ALG_MLDSA = "ml-dsa-65"
ALG_ED25519 = "ed25519"
ALGORITHMS = (ALG_MLDSA, ALG_ED25519)


def _mldsa() -> Any:
    from cryptography.hazmat.primitives.asymmetric import mldsa  # may be absent on old installs

    return mldsa


def mldsa_available() -> bool:
    try:
        _mldsa().MLDSA65PrivateKey.generate()
        return True
    except Exception:  # noqa: BLE001 - any import/runtime failure => unavailable
        return False


def default_algorithm() -> str:
    """Post-quantum when the installed cryptography supports it, else classical."""
    return ALG_MLDSA if mldsa_available() else ALG_ED25519


def generate_keypair(algorithm: str | None = None) -> dict[str, Any]:
    alg = algorithm or default_algorithm()
    if alg == ALG_MLDSA:
        key: Any = _mldsa().MLDSA65PrivateKey.generate()
    elif alg == ALG_ED25519:
        key = ed25519.Ed25519PrivateKey.generate()
    else:
        raise ValueError(f"unsupported algorithm {alg!r}; expected one of {ALGORITHMS}")
    private_pem = key.private_bytes(
        _ser.Encoding.PEM, _ser.PrivateFormat.PKCS8, _ser.NoEncryption()
    )
    return {
        "algorithm": alg,
        "private_pem": private_pem,
        "public_key": key.public_key().public_bytes_raw(),
    }


def sign(private_pem: bytes, message: bytes) -> bytes:
    key: Any = _ser.load_pem_private_key(private_pem, password=None)
    signature: bytes = key.sign(message)  # Ed25519/ML-DSA: single-arg sign
    return signature


def verify(algorithm: str, public_key_raw: bytes, signature: bytes, message: bytes) -> bool:
    """Constant-effort verify; returns False on any failure (never raises)."""
    try:
        if algorithm == ALG_MLDSA:
            pub: Any = _mldsa().MLDSA65PublicKey.from_public_bytes(public_key_raw)
        elif algorithm == ALG_ED25519:
            pub = ed25519.Ed25519PublicKey.from_public_bytes(public_key_raw)
        else:
            return False
        pub.verify(signature, message)
        return True
    except Exception:  # noqa: BLE001 - invalid signature/key/alg => not verified
        return False
