"""Tests for zaxy.config production configuration handling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from zaxy.config import Settings
from zaxy.log import get_logger, setup_logging


def test_remote_rate_limit_and_audit_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.mcp_rate_limit_enabled is True
    assert settings.mcp_rate_limit_requests == 120
    assert settings.mcp_rate_limit_window_seconds == 60
    assert settings.mcp_audit_enabled is False
    assert settings.mcp_audit_path == ".eventloom/remote_audit.jsonl"
    assert settings.mcp_lifecycle_capture_enabled is True


def test_domain_default_is_optional() -> None:
    settings = Settings(_env_file=None)

    assert settings.zaxy_domain is None


def test_projection_backend_defaults_to_embedded() -> None:
    settings = Settings(_env_file=None)

    assert settings.projection_backend == "embedded"


def test_projection_backend_description_lists_embedded_first() -> None:
    """User-facing config metadata should present the default backend first."""
    description = Settings.model_fields["projection_backend"].description

    assert description == "Projection backend: embedded, neo4j, pggraph, or latticedb"


def test_pggraph_dsn_defaults_to_local_postgres() -> None:
    settings = Settings(_env_file=None)

    assert settings.pggraph_dsn == "postgresql://postgres:postgres@localhost:5432/zaxy"


def test_pggraph_bootstrap_defaults_are_local_and_explicit() -> None:
    settings = Settings(_env_file=None)

    assert settings.pggraph_auto_start is False
    assert settings.pggraph_auto_start_image == "pgvector/pgvector:pg17"
    assert settings.pggraph_auto_start_container == "zaxy-pggraph"
    assert settings.pggraph_repo is None


def test_sidecar_autostart_defaults_are_disabled_for_embedded_backend() -> None:
    settings = Settings(_env_file=None)

    assert settings.neo4j_auto_start is False
    assert settings.pggraph_auto_start is False


def test_embedded_graph_path_defaults_to_eventloom_projection_file() -> None:
    settings = Settings(_env_file=None)

    assert settings.embedded_graph_path == ".eventloom/projections/embedded.kuzu"


def test_latticedb_path_defaults_to_eventloom_projection_file() -> None:
    settings = Settings(_env_file=None)

    assert settings.latticedb_path == ".eventloom/projections/memory.latticedb"


def test_retention_policy_defaults_are_non_destructive() -> None:
    settings = Settings(_env_file=None)

    assert settings.retention_policy == "none"
    assert settings.retention_decay_half_life_days == 30
    assert settings.retention_expired_weight == 0.0


def test_context_assembly_defaults_include_source_recall() -> None:
    settings = Settings(_env_file=None)

    assert settings.context_verbatim_enabled is True
    assert settings.context_verbatim_slots == 1
    assert settings.context_packet_memory_enabled is True
    assert settings.context_packet_memory_slots == 1


def test_retrieval_profile_default_is_local_fast() -> None:
    settings = Settings(_env_file=None)

    assert settings.retrieval_profile == "local_fast"


def test_setup_logging_json_serializes_exception(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "zaxy.log.get_settings",
        lambda: SimpleNamespace(log_level="warning", log_format="json"),
    )

    setup_logging()
    logger = get_logger("config-test")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.warning("structured failure", exc_info=True)

    payload = json.loads(capsys.readouterr().err)
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "zaxy.config-test"
    assert payload["message"] == "structured failure"
    assert "RuntimeError: boom" in payload["exception"]


class TestSecretFiles:
    """Docker/Kubernetes secret file loading."""

    def test_loads_secret_file_when_env_value_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """*_FILE env vars should populate sensitive settings."""
        password_file = tmp_path / "neo4j_password"
        password_file.write_text("from-file\n", encoding="utf-8")
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.setenv("NEO4J_PASSWORD_FILE", str(password_file))

        settings = Settings(_env_file=None)

        assert settings.neo4j_password == "from-file"

    def test_direct_env_wins_over_secret_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct env vars should take precedence over *_FILE env vars."""
        password_file = tmp_path / "neo4j_password"
        password_file.write_text("from-file\n", encoding="utf-8")
        monkeypatch.setenv("NEO4J_PASSWORD", "from-env")
        monkeypatch.setenv("NEO4J_PASSWORD_FILE", str(password_file))

        settings = Settings(_env_file=None)

        assert settings.neo4j_password == "from-env"

    def test_loads_admin_and_pathlight_secret_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All production secret fields should support *_FILE env vars."""
        admin_file = tmp_path / "mcp_admin"
        remote_file = tmp_path / "mcp_remote"
        openai_file = tmp_path / "openai_key"
        reranker_file = tmp_path / "reranker_key"
        pathlight_file = tmp_path / "pathlight_token"
        admin_file.write_text("admin-secret\n", encoding="utf-8")
        remote_file.write_text("remote-secret\n", encoding="utf-8")
        openai_file.write_text("openai-secret\n", encoding="utf-8")
        reranker_file.write_text("reranker-secret\n", encoding="utf-8")
        pathlight_file.write_text("pathlight-secret\n", encoding="utf-8")
        monkeypatch.delenv("MCP_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("MCP_REMOTE_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("RERANKER_API_KEY", raising=False)
        monkeypatch.delenv("PATHLIGHT_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("MCP_ADMIN_TOKEN_FILE", str(admin_file))
        monkeypatch.setenv("MCP_REMOTE_AUTH_TOKEN_FILE", str(remote_file))
        monkeypatch.setenv("OPENAI_API_KEY_FILE", str(openai_file))
        monkeypatch.setenv("RERANKER_API_KEY_FILE", str(reranker_file))
        monkeypatch.setenv("PATHLIGHT_ACCESS_TOKEN_FILE", str(pathlight_file))

        settings = Settings(_env_file=None)

        assert settings.mcp_admin_token == "admin-secret"
        assert settings.mcp_remote_auth_token == "remote-secret"
        assert settings.openai_api_key == "openai-secret"
        assert settings.reranker_api_key == "reranker-secret"
        assert settings.pathlight_access_token == "pathlight-secret"

    def test_loads_oidc_client_secret_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OIDC client secrets should support *_FILE loading."""
        secret_file = tmp_path / "oidc_client_secret"
        secret_file.write_text("oidc-secret\n", encoding="utf-8")
        monkeypatch.delenv("MCP_OIDC_CLIENT_SECRET", raising=False)
        monkeypatch.setenv("MCP_OIDC_CLIENT_SECRET_FILE", str(secret_file))

        settings = Settings(_env_file=None)

        assert settings.mcp_oidc_client_secret == "oidc-secret"

    def test_missing_secret_file_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configured *_FILE path should fail loudly if it cannot be read."""
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.setenv("NEO4J_PASSWORD_FILE", "/does/not/exist")

        with pytest.raises(ValueError, match="NEO4J_PASSWORD_FILE"):
            Settings(_env_file=None)

    def test_loads_secret_file_declared_in_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Secret file references in `.env` should populate sensitive settings."""
        password_file = tmp_path / "neo4j_password.txt"
        remote_file = tmp_path / "mcp_remote_auth_token.txt"
        admin_file = tmp_path / "mcp_admin_token.txt"
        env_file = tmp_path / ".env"
        password_file.write_text("secure-password\n", encoding="utf-8")
        remote_file.write_text("remote-token\n", encoding="utf-8")
        admin_file.write_text("admin-token\n", encoding="utf-8")
        env_file.write_text(
            f"ZAXY_ENV=production\n"
            f"NEO4J_URI=bolt://neo4j:7687\n"
            f"NEO4J_CA_CERT=/ssl/bolt/trusted/public.crt\n"
            f"NEO4J_PASSWORD_FILE={password_file}\n"
            f"MCP_REMOTE_AUTH_TOKEN_FILE={remote_file}\n"
            f"MCP_ADMIN_TOKEN_FILE={admin_file}\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("ZAXY_ENV", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD_FILE", raising=False)
        monkeypatch.delenv("MCP_REMOTE_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MCP_REMOTE_AUTH_TOKEN_FILE", raising=False)
        monkeypatch.delenv("MCP_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("MCP_ADMIN_TOKEN_FILE", raising=False)

        settings = Settings(_env_file=env_file)

        assert settings.neo4j_password == "secure-password"
        assert settings.mcp_remote_auth_token == "remote-token"
        assert settings.mcp_admin_token == "admin-token"


class TestProductionValidation:
    """Production-mode security validation."""

    def test_production_requires_non_default_password(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production mode should reject the development password for Neo4j."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("PROJECTION_BACKEND", "neo4j")
        monkeypatch.setenv("NEO4J_URI", "bolt+s://neo4j:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")

        with pytest.raises(ValidationError, match="NEO4J_PASSWORD"):
            Settings(_env_file=None)

    def test_production_requires_tls_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Production mode should reject plaintext Neo4j URIs when Neo4j is selected."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("PROJECTION_BACKEND", "neo4j")
        monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
        monkeypatch.delenv("NEO4J_CA_CERT", raising=False)
        monkeypatch.setenv("NEO4J_PASSWORD", "secure-password")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")

        with pytest.raises(ValidationError, match="NEO4J_URI"):
            Settings(_env_file=None)

    def test_production_embedded_backend_does_not_require_neo4j_secrets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Embedded production deployments should not inherit sidecar Neo4j requirements."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("PROJECTION_BACKEND", "embedded")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")
        monkeypatch.setenv("MCP_REMOTE_AUTH_TOKEN", "remote-token")
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD_FILE", raising=False)
        monkeypatch.delenv("NEO4J_CA_CERT", raising=False)

        settings = Settings(_env_file=None)

        assert settings.projection_backend == "embedded"

    def test_production_requires_admin_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production mode should not expose admin tools without an admin token."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("NEO4J_URI", "bolt+s://neo4j:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "secure-password")
        monkeypatch.delenv("MCP_ADMIN_TOKEN", raising=False)
        monkeypatch.delenv("MCP_ADMIN_TOKEN_FILE", raising=False)

        with pytest.raises(ValidationError, match="MCP_ADMIN_TOKEN"):
            Settings(_env_file=None)

    def test_production_requires_remote_static_or_oidc_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production SSE should require either static bearer auth or OIDC."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("NEO4J_URI", "bolt+s://neo4j:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "secure-password")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")
        monkeypatch.delenv("MCP_REMOTE_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MCP_REMOTE_AUTH_TOKEN_FILE", raising=False)
        monkeypatch.delenv("MCP_OIDC_ISSUER", raising=False)
        monkeypatch.delenv("MCP_OIDC_AUDIENCE", raising=False)
        monkeypatch.delenv("MCP_OIDC_JWKS_URL", raising=False)

        with pytest.raises(ValidationError, match="MCP_REMOTE_AUTH_TOKEN"):
            Settings(_env_file=None)

    def test_production_allows_complete_oidc_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Complete OIDC configuration should satisfy remote auth in production."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("NEO4J_URI", "bolt+s://neo4j:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "secure-password")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")
        monkeypatch.delenv("MCP_REMOTE_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("MCP_OIDC_ISSUER", "https://idp.example")
        monkeypatch.setenv("MCP_OIDC_AUDIENCE", "zaxy")
        monkeypatch.setenv("MCP_OIDC_JWKS_URL", "https://idp.example/.well-known/jwks.json")

        settings = Settings(_env_file=None)

        assert settings.mcp_oidc_issuer == "https://idp.example"

    def test_production_allows_bolt_uri_with_custom_ca(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom CA config makes bolt:// use encrypted driver settings."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "secure-password")
        monkeypatch.setenv("NEO4J_CA_CERT", "/ssl/bolt/trusted/public.crt")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")
        monkeypatch.setenv("MCP_REMOTE_AUTH_TOKEN", "remote-token")

        settings = Settings(_env_file=None)

        assert settings.neo4j_uri == "bolt://neo4j:7687"


def test_logging_uses_stderr_for_mcp_stdio(capsys: pytest.CaptureFixture[str]) -> None:
    """Logs must not contaminate MCP stdio frames on stdout."""
    setup_logging()

    logging.getLogger("zaxy.mcp_server").error("startup failed")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "startup failed" in captured.err
