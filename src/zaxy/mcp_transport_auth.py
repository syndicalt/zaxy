"""Transport auth for remote MCP/SSE HTTP requests.

Authenticates bearer-token and OIDC requests, resolves the remote session
scope, and wraps the whole thing in a rate-limiting, audit-logging guard.

Extracted from :mod:`zaxy.mcp_server`, which re-exports every public name here
so `from zaxy.mcp_server import RemoteRequestGuard` (and friends) keeps working.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import jwt

from zaxy.metrics import get_metrics
from zaxy.remote_security import AuditEventExporter, RemoteAuditEvent, SessionRateLimiter
from zaxy.security import validate_session_id


class JWTDecoder(Protocol):
    def __call__(self, token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
        """Decode and validate a JWT."""


class JWKSClient(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any:
        """Return the signing key for a JWT."""


class MCPTransportAuth:
    """Authenticate and scope remote MCP/SSE HTTP requests."""

    def __init__(
        self,
        token: str | None,
        session_header: str = "x-zaxy-session-id",
        oidc_issuer: str | None = None,
        oidc_audience: str | None = None,
        oidc_jwks_url: str | None = None,
        oidc_required_scope: str = "zaxy:mcp",
        oidc_session_claim: str = "zaxy_session",
        jwt_client: JWKSClient | None = None,
        jwt_decoder: JWTDecoder | None = None,
    ) -> None:
        self._token = token
        self._session_header = session_header.casefold()
        self._oidc_issuer = oidc_issuer
        self._oidc_audience = oidc_audience
        self._oidc_jwks_url = oidc_jwks_url
        self._oidc_required_scope = oidc_required_scope
        self._oidc_session_claim = oidc_session_claim
        self._jwt_client = jwt_client
        self._jwt_decoder = jwt_decoder or jwt.decode

    def authorize(self, headers: Mapping[str, str]) -> str:
        """Validate request headers and return the remote session scope."""
        normalized = {key.casefold(): value for key, value in headers.items()}
        if self._oidc_enabled:
            return self._authorize_oidc(normalized)
        if self._token is not None:
            header = normalized.get("authorization")
            if not header or not header.startswith("Bearer "):
                raise PermissionError("Authorization bearer token is required")
            supplied = header.removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied, self._token):
                raise PermissionError("Authorization bearer token is invalid")
            session_id = normalized.get(self._session_header)
            if not session_id:
                raise PermissionError("remote session header is required")
            return validate_session_id(session_id)
        return validate_session_id(normalized.get(self._session_header, "default"))

    @property
    def _oidc_enabled(self) -> bool:
        return bool(self._oidc_issuer and self._oidc_audience and self._oidc_jwks_url)

    def _authorize_oidc(self, headers: Mapping[str, str]) -> str:
        header = headers.get("authorization")
        if not header or not header.startswith("Bearer "):
            raise PermissionError("Authorization bearer token is required")
        token = header.removeprefix("Bearer ").strip()
        if not token:
            raise PermissionError("Authorization bearer token is required")
        try:
            jwks_client = self._jwt_client or jwt.PyJWKClient(str(self._oidc_jwks_url))
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            claims = self._jwt_decoder(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=self._oidc_audience,
                issuer=self._oidc_issuer,
                options={"require": ["exp", "iat", "iss", "aud"]},
            )
        except Exception as exc:
            raise PermissionError("Authorization bearer token is invalid") from exc

        scopes = _claim_values(claims.get("scope")) | _claim_values(claims.get("scp"))
        if self._oidc_required_scope and self._oidc_required_scope not in scopes:
            raise PermissionError("Authorization bearer token missing required scope")

        session_claim = claims.get(self._oidc_session_claim)
        if not isinstance(session_claim, str) or not session_claim:
            raise PermissionError("Authorization bearer token missing session claim")
        return validate_session_id(session_claim)


class RemoteRateLimitError(PermissionError):
    """Raised when a remote session exceeds its request rate limit."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("remote MCP rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


class RemoteRequestGuard:
    """Authorize, rate-limit, and audit remote MCP/SSE HTTP requests."""

    def __init__(
        self,
        *,
        auth: MCPTransportAuth,
        rate_limit_enabled: bool,
        rate_limit_requests: int,
        rate_limit_window_seconds: int,
        audit_enabled: bool,
        audit_path: Path | str,
    ) -> None:
        self._auth = auth
        self._limiter = SessionRateLimiter(
            enabled=rate_limit_enabled,
            max_requests=rate_limit_requests,
            window_seconds=rate_limit_window_seconds,
        )
        self._audit = AuditEventExporter(path=Path(audit_path), enabled=audit_enabled)

    def authorize(
        self,
        headers: Mapping[str, str],
        *,
        route: str,
        method: str,
        client_host: str | None,
    ) -> str:
        """Return authorized session ID or raise an auth/rate-limit error."""
        try:
            session_id = self._auth.authorize(headers)
        except (PermissionError, ValueError) as exc:
            self._write_audit(
                session_id=None,
                route=route,
                method=method,
                outcome="denied_auth",
                reason=str(exc),
                client_host=client_host,
            )
            raise

        decision = self._limiter.check(session_id)
        if not decision.allowed:
            get_metrics().record_rate_limit_denial(session_id)
            self._write_audit(
                session_id=session_id,
                route=route,
                method=method,
                outcome="denied_rate_limit",
                reason="rate limit exceeded",
                client_host=client_host,
            )
            raise RemoteRateLimitError(decision.retry_after_seconds)

        self._write_audit(
            session_id=session_id,
            route=route,
            method=method,
            outcome="allowed",
            reason=None,
            client_host=client_host,
        )
        return session_id

    def _write_audit(
        self,
        *,
        session_id: str | None,
        route: str,
        method: str,
        outcome: str,
        reason: str | None,
        client_host: str | None,
    ) -> None:
        self._audit.write(
            RemoteAuditEvent(
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                session_id=session_id,
                route=route,
                method=method,
                outcome=outcome,  # type: ignore[arg-type]
                reason=reason,
                client_host=client_host,
            )
        )


def _claim_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {part for part in value.split() if part}
    if isinstance(value, list):
        return {str(part) for part in value if part}
    return set()
