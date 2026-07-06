"""Verified forgetting via cryptographic erasure (Zaxy 3 / I5b).

Crypto-meaningful, real-behavior tests: the SUT is exercised end to end (no
mocked crypto, no weakened assertions). The load-bearing invariant is that
``EventLog.verify()`` operates on the RAW on-disk ciphertext and NEVER decrypts,
so a ``verified_forget`` (which only destroys an out-of-log key) keeps the hash
chain green while making the plaintext permanently unrecoverable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zaxy.config import Settings
from zaxy.core.fabric import ForgetTombstoneUnauditedError, MemoryFabric
from zaxy.event import EventLog
from zaxy.forgetting import (
    CIPHER_PAYLOAD_KEY,
    FORGOTTEN_TEXT,
    MEMORY_FORGOTTEN_EVENT_TYPE,
    PersistentErasureVault,
    build_memory_forgotten_event,
    build_vault,
    cipher_cell,
    decrypt_payload,
    encrypt_forgettable_payload,
    forgotten_sentinel,
    is_forgotten_payload,
)

_SECRET = "SSN 123-45-6789"


# ---------------------------------------------------------------------------
# Unit: PersistentErasureVault + encrypt/decrypt + tombstone builder
# ---------------------------------------------------------------------------


def test_encrypt_seals_ciphertext_and_decrypt_recovers(tmp_path: Path) -> None:
    vault = PersistentErasureVault.for_eventloom_dir(tmp_path)
    payload = {"title": "plan", "content": _SECRET}
    sealed = encrypt_forgettable_payload(payload, vault=vault)

    assert set(sealed) == {CIPHER_PAYLOAD_KEY}
    assert _SECRET not in json.dumps(sealed)  # only ciphertext is sealed
    assert decrypt_payload(sealed, vault=vault) == payload


def test_vault_persists_across_instances_and_kek_is_0600(tmp_path: Path) -> None:
    vault = PersistentErasureVault.for_eventloom_dir(tmp_path)
    sealed = encrypt_forgettable_payload({"content": _SECRET}, vault=vault)

    # A fresh instance reads the wrapped DEK from disk (shared store).
    reborn = PersistentErasureVault.for_eventloom_dir(tmp_path)
    assert decrypt_payload(sealed, vault=reborn) == {"content": _SECRET}

    kek = tmp_path / "__erasure_kek__.key"
    assert kek.exists()
    assert oct(kek.stat().st_mode & 0o777) == "0o600"
    vault_file = tmp_path / "__erasure_vault__.json"
    assert vault_file.exists()
    assert _SECRET not in vault_file.read_text()  # never plaintext on disk


def test_erase_destroys_key_and_decrypt_returns_forgotten(tmp_path: Path) -> None:
    vault = PersistentErasureVault.for_eventloom_dir(tmp_path)
    sealed = encrypt_forgettable_payload({"content": _SECRET}, vault=vault)
    cell_id = sealed[CIPHER_PAYLOAD_KEY]["cell_id"]

    assert vault.erase(cell_id, erased_at="2026-06-28T00:00:00Z") is True
    assert vault.is_erased(cell_id) is True
    decrypted = decrypt_payload(sealed, vault=vault)
    assert is_forgotten_payload(decrypted)
    assert decrypted["content"] == FORGOTTEN_TEXT
    assert _SECRET not in json.dumps(decrypted)
    # Even a fresh instance can never recover it -- the only wrapped DEK is gone.
    reborn = PersistentErasureVault.for_eventloom_dir(tmp_path)
    assert is_forgotten_payload(decrypt_payload(sealed, vault=reborn))


def test_decrypt_passes_through_plaintext_payloads(tmp_path: Path) -> None:
    vault = PersistentErasureVault.for_eventloom_dir(tmp_path)
    assert decrypt_payload({"a": 1}, vault=vault) == {"a": 1}
    assert decrypt_payload({"a": 1}, vault=None) == {"a": 1}
    assert cipher_cell({"a": 1}) is None
    assert forgotten_sentinel() == forgotten_sentinel()
    # fresh dict each call -- callers cannot poison shared state
    assert forgotten_sentinel() is not forgotten_sentinel()


def test_build_memory_forgotten_event_is_cited_and_audited() -> None:
    spec = build_memory_forgotten_event(
        actor="zaxy-forgetter",
        session_id="s1",
        target={"seq": 4, "hash": "a" * 64},
        cell_id="cell-xyz",
        reason="gdpr erasure request",
    )
    assert spec["event_type"] == MEMORY_FORGOTTEN_EVENT_TYPE
    assert spec["thread"] == "s1"
    payload = spec["payload"]
    assert payload["target"] == {"seq": 4, "hash": "a" * 64}
    assert payload["cell_id"] == "cell-xyz"
    assert payload["reason"] == "gdpr erasure request"
    assert payload["authority_status"] == "non_authoritative"
    assert payload["forget_id"].startswith("forget:")


@pytest.mark.parametrize(
    "target",
    [{"seq": 0, "hash": "a" * 64}, {"seq": 1, "hash": "abc"}, {"hash": "a" * 64}],
)
def test_build_memory_forgotten_event_rejects_bad_target(target: dict) -> None:
    with pytest.raises(ValueError):
        build_memory_forgotten_event(
            actor="a", session_id="s", target=target, cell_id="c", reason="r"
        )


# ---------------------------------------------------------------------------
# EventLog: the critical verify()-over-ciphertext invariant
# ---------------------------------------------------------------------------


def test_forgettable_append_seals_ciphertext_and_verify_ok(tmp_path: Path) -> None:
    """(a) Forgettable append -> on-disk ciphertext; decrypted read recovers plaintext."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append("note.public", actor="user", payload={"title": "public"})
    log.append("note.secret", actor="user", payload={"content": _SECRET}, forgettable=True)

    on_disk = (tmp_path / "agent.jsonl").read_text()
    assert _SECRET not in on_disk
    raw = log.read_all()
    assert CIPHER_PAYLOAD_KEY in raw[1].payload
    assert log.verify().ok is True

    decrypted = log.read_all_decrypted()
    assert decrypted[0].payload == {"title": "public"}  # plaintext passthrough
    assert decrypted[1].payload == {"content": _SECRET}  # recovered via vault
    assert decrypted[1].hash == raw[1].hash  # sealed citation stable


def test_verify_never_decrypts_after_key_erased(tmp_path: Path) -> None:
    """(e) Erasing the key must not break verify() -- proof it operates on ciphertext."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append("note.secret", actor="user", payload={"content": _SECRET}, forgettable=True)
    cell_id = log.read_all()[0].payload[CIPHER_PAYLOAD_KEY]["cell_id"]

    before = (tmp_path / "agent.jsonl").read_bytes()
    log.vault.erase(cell_id, erased_at="2026-06-28T00:00:00Z")
    after = (tmp_path / "agent.jsonl").read_bytes()

    assert before == after  # the log bytes are untouched by erasure
    assert log.verify().ok is True  # would FAIL if verify decrypted
    assert is_forgotten_payload(log.read_all_decrypted()[0].payload)


def test_tampering_ciphertext_fails_verify(tmp_path: Path) -> None:
    """(f) Corrupting the sealed ciphertext fails verify (hash mismatch)."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append("note.secret", actor="user", payload={"content": _SECRET}, forgettable=True)
    path = tmp_path / "agent.jsonl"
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    cell = (rec.get("payload") or {}).get(CIPHER_PAYLOAD_KEY)
    assert isinstance(cell, dict)
    ct = cell["ciphertext"]
    cell["ciphertext"] = ("ff" if ct[:2] != "ff" else "00") + ct[2:]
    lines[0] = json.dumps(rec, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    assert EventLog(path).verify().ok is False


def test_plaintext_append_is_byte_identical_to_no_vault(tmp_path: Path) -> None:
    """(d) The default plaintext append path is byte-identical (forgetting unused)."""
    plain = EventLog(tmp_path / "plain.jsonl")
    plain.append("goal.created", actor="user", payload={"title": "ship it"}, timestamp=_ts())
    with_vault = EventLog(tmp_path / "vaulted.jsonl", vault=PersistentErasureVault.for_eventloom_dir(tmp_path))
    with_vault.append("goal.created", actor="user", payload={"title": "ship it"}, timestamp=_ts())

    assert (tmp_path / "plain.jsonl").read_bytes() == (tmp_path / "vaulted.jsonl").read_bytes()
    # no vault file is created by a plaintext-only append
    assert not (tmp_path / "__erasure_vault__.json").exists()


def _ts():
    from datetime import UTC, datetime

    return datetime(2026, 6, 28, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fabric: verified_forget end to end
# ---------------------------------------------------------------------------


def _fabric(tmp_path: Path) -> MemoryFabric:
    fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
    fabric.settings = Settings(forgetting_enabled=True)
    return fabric


@pytest.mark.asyncio
async def test_fabric_forgettable_append_then_verified_forget(tmp_path: Path) -> None:
    """(a)+(b) Full crypto-erasure: ciphertext sealed, key erased, verify stays green."""
    fabric = _fabric(tmp_path)
    await fabric.connect()
    try:
        event = await fabric.append(
            "memory.note",
            actor="user",
            payload={"content": _SECRET, "title": "sensitive"},
            session_id="agent-1",
            forgettable=True,
        )
        eventlog = fabric.session_manager.get("agent-1").eventlog
        log_path = eventlog.path

        # (a) on-disk ciphertext, verify ok, decrypted read recovers plaintext
        assert _SECRET not in log_path.read_text()
        cell = cipher_cell(eventlog.read_all()[event.seq - 1].payload)
        assert cell is not None
        assert eventlog.verify().ok is True
        recovered = eventlog.read_all_decrypted()[event.seq - 1].payload
        assert recovered == {"content": _SECRET, "title": "sensitive"}

        bytes_before = log_path.read_bytes()

        # (b) verified_forget
        result = await fabric.verified_forget(
            target_seq=event.seq,
            target_hash=event.hash,
            reason="gdpr erasure request",
            session_id="agent-1",
        )
        assert result["erased"] is True
        assert result["cell_id"] == cell["cell_id"]
        assert result["gate"]["op"] == "forget"

        # append-only: every pre-forget byte (incl. the target ciphertext line) is unchanged
        assert log_path.read_bytes().startswith(bytes_before)
        # verify STILL green (verify never decrypts)
        assert eventlog.verify().ok is True
        # plaintext permanently unrecoverable -> [FORGOTTEN]
        forgotten = eventlog.read_all_decrypted()[event.seq - 1].payload
        assert is_forgotten_payload(forgotten)
        assert _SECRET not in json.dumps(eventlog.read_all_decrypted()[event.seq - 1].payload)

        # tombstone + gate audit events appended
        types = [e.type for e in eventlog.read_all()]
        assert MEMORY_FORGOTTEN_EVENT_TYPE in types
        assert "evolution.gate.evaluated" in types
        tomb = next(e for e in eventlog.read_all() if e.type == MEMORY_FORGOTTEN_EVENT_TYPE)
        assert tomb.payload["target"] == {"seq": event.seq, "hash": event.hash}
        assert tomb.payload["cell_id"] == cell["cell_id"]
        assert tomb.payload["authority_status"] == "non_authoritative"
    finally:
        await fabric.close()


@pytest.mark.asyncio
async def test_verified_forget_tombstone_failure_is_flagged_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the DEK is destroyed but the tombstone append fails, surface it loudly.

    The erasure is irreversible, so a failed tombstone append leaves an
    erased-but-unaudited memory. That must raise a distinguishable
    ``ForgetTombstoneUnauditedError`` (carrying the ids needed to re-append the
    tombstone) rather than a generic error or a silent swallow.
    """
    fabric = _fabric(tmp_path)
    await fabric.connect()
    try:
        event = await fabric.append(
            "memory.note",
            actor="user",
            payload={"content": _SECRET, "title": "sensitive"},
            session_id="agent-1",
            forgettable=True,
        )
        eventlog = fabric.session_manager.get("agent-1").eventlog
        cell = cipher_cell(eventlog.read_all()[event.seq - 1].payload)
        assert cell is not None

        # Fail ONLY the tombstone append; the gate-audit append (a different
        # event type, appended before the key erasure) must still succeed so the
        # erasure actually happens and we exercise the erased-but-unaudited path.
        original_append = fabric.append

        async def failing_append(event_type: str, *args: object, **kwargs: object) -> object:
            if event_type == MEMORY_FORGOTTEN_EVENT_TYPE:
                raise RuntimeError("simulated disk-full during tombstone append")
            return await original_append(event_type, *args, **kwargs)

        monkeypatch.setattr(fabric, "append", failing_append)

        with pytest.raises(ForgetTombstoneUnauditedError) as excinfo:
            await fabric.verified_forget(
                target_seq=event.seq,
                target_hash=event.hash,
                reason="gdpr erasure request",
                session_id="agent-1",
            )

        # The alert carries what an operator needs to re-append the tombstone.
        assert excinfo.value.cell_id == cell["cell_id"]
        assert excinfo.value.target == {"seq": event.seq, "hash": event.hash}
        assert excinfo.value.forget_id
        # The original failure is chained, not hidden.
        assert isinstance(excinfo.value.__cause__, RuntimeError)

        # The key really was destroyed (plaintext is gone) ...
        forgotten = eventlog.read_all_decrypted()[event.seq - 1].payload
        assert is_forgotten_payload(forgotten)
        # ... but no tombstone landed: this is exactly the gap the error flags.
        assert MEMORY_FORGOTTEN_EVENT_TYPE not in [e.type for e in eventlog.read_all()]
    finally:
        await fabric.close()


@pytest.mark.asyncio
async def test_forgotten_memory_excluded_from_checkout_and_export(tmp_path: Path) -> None:
    """(c) After forget: checkout/retrieval exclude the plaintext; export shows [FORGOTTEN]."""
    from zaxy.export_view import ExportSelector, build_memory_export_view
    from zaxy.retrieval_cache import SessionRetrievalCache

    fabric = _fabric(tmp_path)
    await fabric.connect()
    try:
        event = await fabric.append(
            "memory.note",
            actor="user",
            payload={"content": f"the {_SECRET} is here", "title": "pii record"},
            session_id="agent-1",
            forgettable=True,
        )
        # while alive the plaintext is recoverable through the decrypted read path
        eventlog = fabric.session_manager.get("agent-1").eventlog
        assert eventlog.read_all_decrypted()[event.seq - 1].payload["content"] == f"the {_SECRET} is here"

        await fabric.verified_forget(
            target_seq=event.seq, target_hash=event.hash, reason="erase", session_id="agent-1"
        )

        # checkout never surfaces the forgotten plaintext; viewer shows [FORGOTTEN]
        checkout = await fabric.checkout_memory("pii record", session_id="agent-1")
        assert _SECRET not in checkout.prompt
        assert FORGOTTEN_TEXT in checkout.prompt

        # eventloom retrieval lane excludes the forgotten memory entirely
        fallback = fabric._query_eventlog_fallback(
            f"the {_SECRET}", "agent-1", limit=10, reason="test"
        )
        assert all(_SECRET not in ctx.content for ctx in fallback)

        # export shows [FORGOTTEN] and leaks no plaintext
        cache = SessionRetrievalCache(fabric.session_manager)
        entries = build_memory_export_view(
            "agent-1", ExportSelector(), retrieval_cache=cache, vault=fabric.session_manager.vault
        )
        blob = json.dumps(entries)
        assert _SECRET not in blob
        assert FORGOTTEN_TEXT in blob
    finally:
        await fabric.close()


@pytest.mark.asyncio
async def test_forgettable_append_disabled_without_flag(tmp_path: Path) -> None:
    """Opt-in guard: forgettable append + forget require FORGETTING_ENABLED."""
    fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)
    fabric.settings = Settings(forgetting_enabled=False)
    await fabric.connect()
    try:
        with pytest.raises(ValueError, match="FORGETTING_ENABLED"):
            await fabric.append(
                "memory.note", actor="u", payload={"content": _SECRET},
                session_id="agent-1", forgettable=True,
            )
        with pytest.raises(ValueError, match="disabled"):
            await fabric.verified_forget(
                target_seq=1, target_hash="a" * 64, reason="x", session_id="agent-1"
            )
    finally:
        await fabric.close()


@pytest.mark.asyncio
async def test_verified_forget_rejects_non_forgettable_target(tmp_path: Path) -> None:
    """verified_forget refuses a plaintext (non-cipher) memory."""
    fabric = _fabric(tmp_path)
    await fabric.connect()
    try:
        event = await fabric.append(
            "goal.created", actor="user", payload={"title": "public"}, session_id="agent-1"
        )
        with pytest.raises(ValueError, match="forgettable"):
            await fabric.verified_forget(
                target_seq=event.seq, target_hash=event.hash, reason="x", session_id="agent-1"
            )
    finally:
        await fabric.close()


# ---------------------------------------------------------------------------
# P1: ANY key failure degrades to [FORGOTTEN] -- never crashes the read path
# ---------------------------------------------------------------------------


def test_rotated_kek_degrades_to_forgotten_not_crash(tmp_path: Path) -> None:
    """A wrapped DEK whose KEK can no longer unwrap it degrades to [FORGOTTEN] on
    every read path; it must NEVER raise InvalidTag up through read_all_decrypted.
    """
    from zaxy.portable.envelope import new_dek

    log = EventLog(
        tmp_path / "agent.jsonl", vault=PersistentErasureVault.for_eventloom_dir(tmp_path)
    )
    log.append("note.secret", actor="user", payload={"content": _SECRET}, forgettable=True)
    sealed = log.read_all()[0].payload
    assert log.read_all_decrypted()[0].payload == {"content": _SECRET}  # intact -> recoverable

    # Rotate/lose the KEK: overwrite the key file with a DIFFERENT valid 32-byte key.
    kek_file = tmp_path / "__erasure_kek__.key"
    rotated = new_dek()
    assert bytes.fromhex(kek_file.read_text().strip()) != rotated
    kek_file.write_text(rotated.hex(), encoding="utf-8")

    # A fresh vault/log reads the NEW (wrong) KEK; unwrapping raises InvalidTag
    # inside get_dek. decrypt_payload + read_all_decrypted must degrade, not raise.
    reborn = EventLog(
        tmp_path / "agent.jsonl", vault=PersistentErasureVault.for_eventloom_dir(tmp_path)
    )
    decrypted = reborn.read_all_decrypted()  # must NOT raise
    assert is_forgotten_payload(decrypted[0].payload)
    assert _SECRET not in json.dumps(decrypted[0].payload)
    fresh_vault = PersistentErasureVault.for_eventloom_dir(tmp_path)
    assert decrypt_payload(sealed, vault=fresh_vault)["content"] == FORGOTTEN_TEXT

    # The raw hash chain is untouched -- verify never decrypts, so it stays green.
    assert reborn.verify().ok is True


def test_length_invalid_kek_degrades_to_forgotten(tmp_path: Path) -> None:
    """A corrupted/length-invalid KEK (not 32 bytes) degrades to [FORGOTTEN]
    instead of raising ValueError out of the read path."""
    vault = PersistentErasureVault.for_eventloom_dir(tmp_path)
    sealed = encrypt_forgettable_payload({"content": _SECRET}, vault=vault)

    (tmp_path / "__erasure_kek__.key").write_text("ab" * 16, encoding="utf-8")  # 16 bytes, not 32

    reborn = PersistentErasureVault.for_eventloom_dir(tmp_path)
    decrypted = decrypt_payload(sealed, vault=reborn)  # must NOT raise ValueError
    assert is_forgotten_payload(decrypted)
    assert decrypted["content"] == FORGOTTEN_TEXT


def test_decrypt_payload_edge_cases_degrade_safely() -> None:
    """Defensive branches: non-dict passthrough; a malformed cipher cell and an
    absent vault both yield the [FORGOTTEN] sentinel, never an exception."""
    assert cipher_cell("not-a-dict") is None  # non-dict -> not a cipher payload
    assert decrypt_payload({"plain": "text"}, vault=None) == {"plain": "text"}
    bad = {CIPHER_PAYLOAD_KEY: {"cell_id": 1, "nonce": None, "ciphertext": 2}}
    assert is_forgotten_payload(decrypt_payload(bad, vault=None))  # fields not all strings
    well_formed = {CIPHER_PAYLOAD_KEY: {"cell_id": "x", "nonce": "00", "ciphertext": "00"}}
    assert is_forgotten_payload(decrypt_payload(well_formed, vault=None))  # no vault available


def test_vault_blacklists_erased_content_address(tmp_path: Path) -> None:
    """The ciphertext content address is erasure-blacklisted after forget."""
    vault = PersistentErasureVault.for_eventloom_dir(tmp_path)
    sealed = encrypt_forgettable_payload({"content": _SECRET}, vault=vault)
    address = sealed[CIPHER_PAYLOAD_KEY]["content_address"]
    assert vault.is_blacklisted(address) is False
    assert vault.is_blacklisted("never-seen") is False
    vault.erase(sealed[CIPHER_PAYLOAD_KEY]["cell_id"], erased_at="2026-06-28T00:00:00Z")
    assert vault.is_blacklisted(address) is True


@pytest.mark.parametrize(
    "blank",
    [
        {"actor": "", "session_id": "s", "cell_id": "c", "reason": "r"},
        {"actor": "a", "session_id": "  ", "cell_id": "c", "reason": "r"},
        {"actor": "a", "session_id": "s", "cell_id": "", "reason": "r"},
        {"actor": "a", "session_id": "s", "cell_id": "c", "reason": ""},
    ],
)
def test_build_memory_forgotten_event_rejects_blank_fields(blank: dict) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        build_memory_forgotten_event(target={"seq": 1, "hash": "a" * 64}, **blank)


def test_build_vault_gates_on_settings(tmp_path: Path) -> None:
    """build_vault is the one gated constructor: None when forgetting is off (a
    no-op for the plaintext path), a real vault rooted at the dir when on."""
    assert build_vault(Settings(forgetting_enabled=False), tmp_path) is None
    vault = build_vault(Settings(forgetting_enabled=True), tmp_path)
    assert isinstance(vault, PersistentErasureVault)
    sealed = encrypt_forgettable_payload({"content": _SECRET}, vault=vault)
    reborn = build_vault(Settings(forgetting_enabled=True), tmp_path)
    assert decrypt_payload(sealed, vault=reborn) == {"content": _SECRET}


@pytest.mark.asyncio
async def test_push_export_decrypts_forgettable_then_forgotten(tmp_path: Path) -> None:
    """P2: the push sink must decrypt forgettable cells -- plaintext while live,
    [FORGOTTEN] after erase -- and NEVER emit the raw __zaxy_cipher ciphertext."""
    from zaxy.export_sinks import FileSink, push_memory_export
    from zaxy.export_view import ExportSelector
    from zaxy.retrieval_cache import SessionRetrievalCache

    fabric = _fabric(tmp_path)
    await fabric.connect()
    try:
        event = await fabric.append(
            "memory.note",
            actor="user",
            payload={"content": _SECRET, "title": "pii"},
            session_id="agent-1",
            forgettable=True,
        )
        vault = fabric.session_manager.vault

        # Pre-fix behavior (no vault threaded) leaks the raw ciphertext cell.
        leaky = tmp_path / "leak.json"
        push_memory_export(
            "agent-1",
            ExportSelector(),
            retrieval_cache=SessionRetrievalCache(fabric.session_manager),
            sink=FileSink(leaky),
        )
        assert CIPHER_PAYLOAD_KEY in leaky.read_text()

        # The fix threads the vault -> decrypted plaintext, no ciphertext.
        live = tmp_path / "live.json"
        push_memory_export(
            "agent-1",
            ExportSelector(),
            retrieval_cache=SessionRetrievalCache(fabric.session_manager),
            vault=vault,
            sink=FileSink(live),
        )
        live_text = live.read_text()
        assert CIPHER_PAYLOAD_KEY not in live_text
        assert _SECRET in live_text

        # After verified_forget the same push shows [FORGOTTEN], never plaintext.
        await fabric.verified_forget(
            target_seq=event.seq, target_hash=event.hash, reason="erase", session_id="agent-1"
        )
        gone = tmp_path / "gone.json"
        push_memory_export(
            "agent-1",
            ExportSelector(),
            retrieval_cache=SessionRetrievalCache(fabric.session_manager),
            vault=vault,
            sink=FileSink(gone),
        )
        gone_text = gone.read_text()
        assert CIPHER_PAYLOAD_KEY not in gone_text
        assert _SECRET not in gone_text
        assert FORGOTTEN_TEXT in gone_text
    finally:
        await fabric.close()
