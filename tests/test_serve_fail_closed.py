"""Tests for the fail-closed SSE bind guard in `zaxy serve` (fix B4).

`serve()` must refuse to bind the SSE transport to a non-loopback host unless
remote-transport authentication is configured, and it must refuse *before*
`mcp_server.main_sse` is ever reached (i.e. before anything actually binds a
socket). These tests exercise the real guard: `_host_is_loopback`, the
`Settings.remote_transport_auth_configured` property it consults, and the
`serve()` function itself with `mcp_server.main_sse`/`ZaxyMCPServer` stubbed
out so the check is exercised without standing up a real server.

Note: as of this fix, `serve()` is not currently registered as a Typer CLI
command (a `@app.command()` decorator that used to sit directly above `def
serve(...)` now decorates the newly-inserted `_host_is_loopback` helper
instead, so `zaxy serve` / `python -m zaxy serve` is unreachable and
`-host-is-loopback` is spuriously registered as a command). `serve` is still
a plain, directly importable function, so the behavioral test below calls it
directly rather than through `typer.testing.CliRunner`, per this task's
explicit fallback for cases where the CLI plumbing doesn't match. This is a
real regression worth fixing in source separately; it is reported alongside
these tests rather than patched here.
"""

from __future__ import annotations

import pytest
import typer

from zaxy.cli.serving import _host_is_loopback, serve
from zaxy.config import Settings, get_settings


class _SentinelReachedError(Exception):
    """Raised by the stubbed `main_sse` to prove the guard let execution through."""


class _StubZaxyMCPServer:
    """Lightweight stand-in for `mcp_server.ZaxyMCPServer` so no real store is built."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


def _call_serve(*, transport: str, host: str) -> None:
    serve(
        eventloom_path=None,
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        transport=transport,
        host=host,
        port=8080,
        profile=None,
    )


# ---- _host_is_loopback ------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "LOCALHOST", "127.5.5.5"])
def test_host_is_loopback_true_for_loopback_hosts(host: str) -> None:
    """Loopback IP literals (127.0.0.0/8, ::1) and the localhost name are loopback."""
    assert _host_is_loopback(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "example.com"])
def test_host_is_loopback_false_for_non_loopback_hosts(host: str) -> None:
    """Wildcard binds, non-loopback IPs, and unresolved hostnames fail closed as non-loopback."""
    assert _host_is_loopback(host) is False


# ---- Settings.remote_transport_auth_configured ------------------------------


def test_remote_transport_auth_configured_false_with_no_auth() -> None:
    """No bearer token and no OIDC config means auth is not configured."""
    settings = Settings()
    assert settings.remote_transport_auth_configured is False


def test_remote_transport_auth_configured_true_with_static_token() -> None:
    """A static MCP_REMOTE_AUTH_TOKEN alone is sufficient."""
    settings = Settings(mcp_remote_auth_token="s3cr3t")
    assert settings.remote_transport_auth_configured is True


def test_remote_transport_auth_configured_true_with_complete_oidc() -> None:
    """A complete OIDC issuer/audience/JWKS triple alone is sufficient."""
    settings = Settings(
        mcp_oidc_issuer="https://issuer.example.com",
        mcp_oidc_audience="zaxy-mcp",
        mcp_oidc_jwks_url="https://issuer.example.com/.well-known/jwks.json",
    )
    assert settings.remote_transport_auth_configured is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mcp_oidc_issuer": "https://issuer.example.com"},
        {"mcp_oidc_issuer": "https://issuer.example.com", "mcp_oidc_audience": "zaxy-mcp"},
        {"mcp_oidc_audience": "zaxy-mcp", "mcp_oidc_jwks_url": "https://issuer.example.com/jwks.json"},
    ],
)
def test_remote_transport_auth_configured_false_with_partial_oidc(kwargs: dict[str, str]) -> None:
    """A partial OIDC configuration must not count as configured auth."""
    settings = Settings(**kwargs)
    assert settings.remote_transport_auth_configured is False


# ---- serve() behavioral guard ------------------------------------------------


def test_serve_sse_refuses_unauthenticated_non_loopback_bind(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """serve(transport="sse", host="0.0.0.0") with no auth must refuse before binding."""
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    from zaxy import mcp_server

    main_sse_calls: list[tuple[int, str]] = []

    async def _tracking_main_sse(port: int = 8080, host: str = "127.0.0.1") -> None:
        main_sse_calls.append((port, host))
        raise _SentinelReachedError("main_sse should not have been called")

    monkeypatch.setattr(mcp_server, "ZaxyMCPServer", _StubZaxyMCPServer)
    monkeypatch.setattr(mcp_server, "main_sse", _tracking_main_sse)

    with pytest.raises(typer.BadParameter):
        _call_serve(transport="sse", host="0.0.0.0")

    assert main_sse_calls == []


@pytest.mark.parametrize(
    "host,configure_auth",
    [
        ("127.0.0.1", False),
        ("0.0.0.0", True),
    ],
)
def test_serve_sse_reaches_main_sse_when_loopback_or_authenticated(
    monkeypatch: pytest.MonkeyPatch, tmp_path, host: str, configure_auth: bool
) -> None:
    """A loopback host, or a non-loopback host with auth configured, passes the guard."""
    monkeypatch.chdir(tmp_path)
    if configure_auth:
        monkeypatch.setenv("MCP_REMOTE_AUTH_TOKEN", "s3cr3t")
    get_settings.cache_clear()

    from zaxy import mcp_server

    async def _raising_main_sse(port: int = 8080, host: str = "127.0.0.1") -> None:
        raise _SentinelReachedError("guard passed, main_sse reached")

    monkeypatch.setattr(mcp_server, "ZaxyMCPServer", _StubZaxyMCPServer)
    monkeypatch.setattr(mcp_server, "main_sse", _raising_main_sse)

    with pytest.raises(_SentinelReachedError):
        _call_serve(transport="sse", host=host)
