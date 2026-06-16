"""Tests for the export outbound/push layer."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from zaxy.export_sinks import FileSink, WebhookSink, push_memory_export
from zaxy.export_view import ExportSelector, build_memory_export
from zaxy.retrieval_cache import SessionRetrievalCache
from zaxy.session import SessionManager


def _cache(tmp_path: Path) -> SessionRetrievalCache:
    return SessionRetrievalCache(SessionManager(base_path=str(tmp_path / ".eventloom")))


def _append(cache: SessionRetrievalCache, session_id: str, event_type: str, payload: dict) -> None:
    cache.session_manager.get(session_id).eventlog.append(
        event_type, actor="a", payload=payload, thread=session_id
    )


class _FakeResponse:
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return b""


def test_file_sink_delivers_bundle_matching_helper(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    selector = ExportSelector(grains=frozenset({"event"}))
    out = tmp_path / "b.json"

    bundle = push_memory_export("s", selector, retrieval_cache=cache, sink=FileSink(out))

    expected = build_memory_export("s", selector, retrieval_cache=cache)
    assert bundle == expected
    assert json.loads(out.read_text(encoding="utf-8")) == expected  # delivered == projected


def test_webhook_sink_posts_json_with_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["data"] = request.data
        captured["content_type"] = request.get_header("Content-type")
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})

    bundle = push_memory_export(
        "s",
        ExportSelector(grains=frozenset({"event"})),
        retrieval_cache=cache,
        sink=WebhookSink("https://example.test/hook", token="secret"),
    )

    assert captured["url"] == "https://example.test/hook"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    assert captured["authorization"] == "Bearer secret"
    assert json.loads(captured["data"].decode("utf-8")) == bundle


def test_webhook_sink_without_token_sends_no_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})

    push_memory_export("s", retrieval_cache=cache, sink=WebhookSink("http://example.test/hook"))
    assert captured["authorization"] is None


def test_webhook_sink_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match="http"):
        WebhookSink("file:///etc/passwd")


def test_push_memory_export_signed_roundtrips(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from zaxy.portable import generate_keypair, verify_export

    keypair = generate_keypair()
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    out = tmp_path / "b.json"

    bundle = push_memory_export("s", retrieval_cache=cache, signing_key=keypair, sink=FileSink(out))
    assert "signature" in bundle and "merkle_root" in bundle
    assert verify_export(json.loads(out.read_text(encoding="utf-8")))["ok"] is True


def test_cli_export_push_file_roundtrip(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from zaxy.__main__ import app

    el = tmp_path / ".eventloom"
    SessionManager(base_path=str(el)).get("demo").eventlog.append(
        "goal.created", actor="a", payload={"title": "g"}, thread="demo"
    )
    out = tmp_path / "b.json"
    res = CliRunner().invoke(
        app,
        ["export-push", "--sink", "file", "--dest", str(out), "--eventloom-path", str(el), "--session-id", "demo"],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["signed"] is False
    assert data["entries"]


def test_cli_export_push_unknown_sink_errors(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from zaxy.__main__ import app

    el = tmp_path / ".eventloom"
    SessionManager(base_path=str(el)).get("demo").eventlog.append(
        "goal.created", actor="a", payload={"title": "g"}, thread="demo"
    )
    res = CliRunner().invoke(
        app,
        ["export-push", "--sink", "s3", "--dest", "x", "--eventloom-path", str(el), "--session-id", "demo"],
    )
    assert res.exit_code != 0
