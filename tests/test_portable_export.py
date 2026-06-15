"""Adversarial tests for signed portable export (dev target #3, Phase 1).

Asserts the security invariants: no valid sig without the key; any byte change
breaks verification; metadata is bound by the signature; subset disclosure proves
membership without leaking undisclosed entries.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from zaxy.portable import (
    build_export,
    disclose_subset,
    generate_keypair,
    merkle,
    mldsa_available,
    signing,
    verify_export,
    verify_subset,
)

ALGS = [signing.ALG_ED25519] + ([signing.ALG_MLDSA] if mldsa_available() else [])

ENTRIES = [
    {"type": "decision.made", "seq": 77848, "payload": {"decision": "session = zaxy-default"}},
    {"type": "goal.created", "seq": 77889, "payload": {"title": "memory injection"}},
    {"type": "task.completed", "seq": 77957, "payload": {"task": "ship 2.3.3"}},
    {"type": "decision.made", "seq": 77885, "payload": {"decision": "reject glyph notation"}},
    {"type": "goal.created", "seq": 78021, "payload": {"title": "signed export"}},
]


def _bundle(alg: str):  # type: ignore[no-untyped-def]
    kp = generate_keypair(alg)
    return build_export(ENTRIES, keypair=kp, session_id="zaxy-default", created_at="2026-06-14T00:00:00Z", nonce="n1"), kp


# ---- Merkle correctness ----------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 9])
def test_merkle_inclusion_all_indices(n: int) -> None:
    leaves = [merkle.leaf_hash(f"leaf{i}".encode()) for i in range(n)]
    root = merkle.merkle_root(leaves)
    for i in range(n):
        proof = merkle.inclusion_proof(leaves, i)
        assert merkle.verify_inclusion(leaves[i], proof, root)
        # a wrong leaf must not verify against the same proof
        assert not merkle.verify_inclusion(merkle.leaf_hash(b"forged"), proof, root)


# ---- round trip + invariants ----------------------------------------------

@pytest.mark.parametrize("alg", ALGS)
def test_roundtrip_verifies(alg: str) -> None:
    bundle, _ = _bundle(alg)
    assert verify_export(bundle)["ok"] is True


@pytest.mark.parametrize("alg", ALGS)
def test_tampered_content_fails(alg: str) -> None:
    bundle, _ = _bundle(alg)
    bad = copy.deepcopy(bundle)
    bad["entries"][1]["content"]["payload"]["title"] = "ATTACKER EDIT"
    assert verify_export(bad)["ok"] is False


@pytest.mark.parametrize("alg", ALGS)
def test_tampered_entry_id_fails(alg: str) -> None:
    bundle, _ = _bundle(alg)
    bad = copy.deepcopy(bundle)
    bad["entries"][0]["id"] = "0" * 64
    assert verify_export(bad)["ok"] is False


@pytest.mark.parametrize("alg", ALGS)
def test_dropped_entry_fails(alg: str) -> None:
    bundle, _ = _bundle(alg)
    bad = copy.deepcopy(bundle)
    del bad["entries"][2]
    assert verify_export(bad)["ok"] is False


@pytest.mark.parametrize("alg", ALGS)
@pytest.mark.parametrize("field", ["session_id", "created_at", "nonce", "merkle_root"])
def test_metadata_is_signature_bound(alg: str, field: str) -> None:
    bundle, _ = _bundle(alg)
    bad = copy.deepcopy(bundle)
    bad[field] = bad[field] + "x" if field != "merkle_root" else "ab" + bad[field][2:]
    assert verify_export(bad)["ok"] is False


@pytest.mark.parametrize("alg", ALGS)
def test_signature_tamper_fails(alg: str) -> None:
    bundle, _ = _bundle(alg)
    bad = copy.deepcopy(bundle)
    flip = "f" if bad["signature"][0] != "f" else "0"
    bad["signature"] = flip + bad["signature"][1:]
    assert verify_export(bad)["ok"] is False


@pytest.mark.parametrize("alg", ALGS)
def test_forgery_with_swapped_key_fails(alg: str) -> None:
    bundle, _ = _bundle(alg)
    attacker = generate_keypair(alg)
    bad = copy.deepcopy(bundle)
    bad["public_key"] = attacker["public_key"].hex()  # claim attacker's key over original sig
    assert verify_export(bad)["ok"] is False


@pytest.mark.parametrize("alg", ALGS)
def test_pubkey_pinning(alg: str) -> None:
    bundle, kp = _bundle(alg)
    assert verify_export(bundle, expect_public_key=kp["public_key"].hex())["ok"] is True
    assert verify_export(bundle, expect_public_key="00" * 8)["ok"] is False


@pytest.mark.parametrize("alg", ALGS)
def test_unknown_version_rejected(alg: str) -> None:
    bundle, _ = _bundle(alg)
    bad = copy.deepcopy(bundle)
    bad["version"] = "zaxy.portable.v99"
    assert verify_export(bad)["ok"] is False  # version allow-list (downgrade/forward-confusion)


def test_unique_nonce_changes_bundle() -> None:
    kp = generate_keypair(signing.ALG_ED25519)
    a = build_export(ENTRIES, keypair=kp, session_id="s", created_at="t", nonce="n1")
    b = build_export(ENTRIES, keypair=kp, session_id="s", created_at="t", nonce="n2")
    assert a["signature"] != b["signature"]


# ---- verifiable partial disclosure ----------------------------------------

@pytest.mark.parametrize("alg", ALGS)
def test_subset_discloses_without_leaking(alg: str) -> None:
    bundle, kp = _bundle(alg)
    subset = disclose_subset(bundle, [1, 3])
    res = verify_subset(subset, expect_public_key=kp["public_key"].hex())
    assert res["ok"] is True
    # only the disclosed entries are present; undisclosed content does not leak
    disclosed_titles = [d["content"] for d in subset["disclosed"]]
    assert ENTRIES[1] in disclosed_titles and ENTRIES[3] in disclosed_titles
    assert ENTRIES[0] not in disclosed_titles  # undisclosed -> absent
    assert "entries" not in subset  # full entry list never shipped


@pytest.mark.parametrize("alg", ALGS)
def test_subset_tampered_disclosed_content_fails(alg: str) -> None:
    bundle, _ = _bundle(alg)
    subset = disclose_subset(bundle, [0])
    subset["disclosed"][0]["content"]["payload"]["decision"] = "FORGED"
    assert verify_subset(subset)["ok"] is False


@pytest.mark.parametrize("alg", ALGS)
def test_subset_forged_proof_fails(alg: str) -> None:
    bundle, _ = _bundle(alg)
    subset = disclose_subset(bundle, [2])
    if subset["disclosed"][0]["proof"]:
        sib, side = subset["disclosed"][0]["proof"][0]
        subset["disclosed"][0]["proof"][0] = ["00" * 32, side]
    assert verify_subset(subset)["ok"] is False


def test_default_algorithm_prefers_pq_when_available() -> None:
    expected = signing.ALG_MLDSA if mldsa_available() else signing.ALG_ED25519
    assert signing.default_algorithm() == expected


def test_cli_keygen_export_verify_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from typer.testing import CliRunner

    from zaxy.__main__ import app
    from zaxy.event import EventLog

    runner = CliRunner()
    el = tmp_path / ".eventloom"
    log = EventLog(el / "demo.jsonl")
    log.append("decision.made", actor="claude", payload={"decision": "x"}, thread="demo")
    log.append("goal.created", actor="claude", payload={"title": "y"}, thread="demo")

    priv, pub, bundle = tmp_path / "k.pem", tmp_path / "k.pub", tmp_path / "b.json"
    alg = "ed25519"  # CLI default flag is ml-dsa-65; pin ed25519 so the test is portable
    assert runner.invoke(app, ["export-keygen", "--out-private", str(priv), "--out-public", str(pub), "--algorithm", alg]).exit_code == 0
    assert priv.exists() and pub.exists()
    assert runner.invoke(app, ["export", "--out", str(bundle), "--private-key", str(priv), "--public-key", str(pub), "--algorithm", alg, "--eventloom-path", str(el), "--session-id", "demo"]).exit_code == 0

    pinned = pub.read_text(encoding="utf-8").strip()
    ok = runner.invoke(app, ["verify-export", str(bundle), "--expect-public-key", pinned])
    assert ok.exit_code == 0 and "OK" in ok.stdout

    # tamper the bundle file -> verification must fail (non-zero exit).
    # Each entry's content is a canonical export entry; mutate the nested payload.
    b = json.loads(bundle.read_text(encoding="utf-8"))
    b["entries"][0]["content"]["content"]["payload"]["decision"] = "TAMPERED"
    bundle.write_text(json.dumps(b), encoding="utf-8")
    assert runner.invoke(app, ["verify-export", str(bundle)]).exit_code == 1


def test_cli_export_unsigned_and_since_delta(tmp_path: Path) -> None:
    """CLI export with no keys writes an unsigned bundle matching the projector,
    and --since exports only newer events."""
    import json

    from typer.testing import CliRunner

    from zaxy.__main__ import app
    from zaxy.export_view import ExportSelector, build_memory_export_view
    from zaxy.retrieval_cache import SessionRetrievalCache
    from zaxy.session import SessionManager

    el = tmp_path / ".eventloom"
    log = SessionManager(base_path=str(el)).get("demo").eventlog
    log.append("decision.made", actor="a", payload={"decision": "d1"}, thread="demo")
    log.append("goal.created", actor="a", payload={"title": "g2"}, thread="demo")

    runner = CliRunner()
    out = tmp_path / "b.json"
    res = runner.invoke(
        app, ["export", "--out", str(out), "--eventloom-path", str(el), "--session-id", "demo"]
    )
    assert res.exit_code == 0, res.output
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["signed"] is False
    assert bundle["version"] == "zaxy.export.unsigned.v1"

    cache = SessionRetrievalCache(SessionManager(base_path=str(el)))
    expected = build_memory_export_view(
        "demo",
        ExportSelector(
            grains=frozenset({"event"}),
            kinds=frozenset({"decision.made", "goal.created", "task.completed"}),
        ),
        retrieval_cache=cache,
    )
    assert bundle["entries"] == expected  # CLI and projector agree

    out2 = tmp_path / "b2.json"
    res2 = runner.invoke(
        app,
        ["export", "--out", str(out2), "--eventloom-path", str(el), "--session-id", "demo", "--since", "1"],
    )
    assert res2.exit_code == 0, res2.output
    delta = json.loads(out2.read_text(encoding="utf-8"))
    assert {e["seq"] for e in delta["entries"]} == {2}
