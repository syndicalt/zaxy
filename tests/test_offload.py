"""Tests for opt-in Eventloom-owned tool-I/O offload (dev target #1)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.lifecycle import OUTPUT_EXCERPT_CHARS
from zaxy.observation import build_command_observation, build_tool_call_observation
from zaxy.offload import read_offload_ref, redact_secret_args, write_offload_ref

runner = CliRunner()

BIG = "X" * (OUTPUT_EXCERPT_CHARS * 4)  # comfortably past the inline excerpt


# ---- default (disabled) keeps the lean behavior unchanged ------------------

def test_command_lean_when_offload_disabled(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ZAXY_OFFLOAD_TOOL_IO", raising=False)
    ev = build_command_observation(
        command="echo hi", exit_code=0, session_id="s", source="test",
        stdout=BIG, stderr="", eventloom_path=str(tmp_path / ".eventloom"),
    )
    assert "full_io_ref" not in ev["payload"]
    assert len(ev["payload"]["stdout_excerpt"]) == OUTPUT_EXCERPT_CHARS


def test_tool_call_lean_when_offload_disabled(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ZAXY_OFFLOAD_TOOL_IO", raising=False)
    ev = build_tool_call_observation(
        tool_name="t", status="succeeded", session_id="s", source="test",
        arguments={"q": "secret-value"}, eventloom_path=str(tmp_path / ".eventloom"),
    )
    assert "full_io_ref" not in ev["payload"]
    assert ev["payload"]["arguments_redacted"] is True
    assert ev["payload"]["argument_keys"] == ["q"]


# ---- enabled: offload writes a recoverable, integrity-checked ref ----------

def test_command_offload_when_enabled(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ZAXY_OFFLOAD_TOOL_IO", "1")
    el = str(tmp_path / ".eventloom")
    ev = build_command_observation(
        command="run", exit_code=0, session_id="s", source="test",
        stdout=BIG, stderr="boom", eventloom_path=el,
    )
    ref = ev["payload"]["full_io_ref"]
    assert set(ref) == {"ref", "sha256", "bytes"}
    # summary stays lean
    assert len(ev["payload"]["stdout_excerpt"]) == OUTPUT_EXCERPT_CHARS
    # full content recoverable and complete
    full = read_offload_ref(el, ref["sha256"])
    assert full is not None and BIG in full and "boom" in full


def test_tool_args_offload_masks_secrets(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ZAXY_OFFLOAD_TOOL_IO", "true")
    el = str(tmp_path / ".eventloom")
    ev = build_tool_call_observation(
        tool_name="t", status="ok", session_id="s", source="test",
        arguments={"query": "keep me", "api_key": "sk-SECRET", "token": "abc"},
        eventloom_path=el,
    )
    ref = ev["payload"]["full_io_ref"]
    stored = json.loads(read_offload_ref(el, ref["sha256"]))
    assert stored["query"] == "keep me"          # non-secret preserved
    assert stored["api_key"] == "<redacted>"     # secret-keyed masked
    assert stored["token"] == "<redacted>"


def test_short_output_no_ref_even_when_enabled(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ZAXY_OFFLOAD_TOOL_IO", "1")
    ev = build_command_observation(
        command="echo hi", exit_code=0, session_id="s", source="test",
        stdout="short", stderr="", eventloom_path=str(tmp_path / ".eventloom"),
    )
    assert "full_io_ref" not in ev["payload"]  # nothing beyond the excerpt to keep


# ---- content addressing + tamper-evidence ----------------------------------

def test_content_addressing_idempotent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    el = str(tmp_path / ".eventloom")
    a = write_offload_ref(el, "same content")
    b = write_offload_ref(el, "same content")
    assert a == b  # identical content -> identical ref, written once


def test_tamper_evident_read(tmp_path) -> None:  # type: ignore[no-untyped-def]
    el = tmp_path / ".eventloom"
    ref = write_offload_ref(str(el), "trustworthy")
    blob = el / ref["ref"]
    blob.write_text("tampered", encoding="utf-8")  # corrupt the blob
    assert read_offload_ref(str(el), ref["sha256"]) is None  # id no longer matches content


def test_redact_secret_args_unit() -> None:
    out = redact_secret_args({"password": "p", "name": "ok", "AUTHORIZATION": "z"})
    assert out == {"password": "<redacted>", "name": "ok", "AUTHORIZATION": "<redacted>"}


# ---- drill-down CLI ---------------------------------------------------------

def test_offload_get_cli_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    el = str(tmp_path / ".eventloom")
    ref = write_offload_ref(el, "full output here")
    result = runner.invoke(app, ["offload-get", ref["sha256"], "--eventloom-path", el])
    assert result.exit_code == 0
    assert result.stdout == "full output here"


def test_offload_get_cli_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(
        app, ["offload-get", "0" * 64, "--eventloom-path", str(tmp_path / ".eventloom")]
    )
    assert result.exit_code != 0
