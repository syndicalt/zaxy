"""Tests for zaxy.config production configuration handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from zaxy.config import Settings


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
        """Production mode should reject the development password."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("NEO4J_URI", "bolt+s://neo4j:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")

        with pytest.raises(ValidationError, match="NEO4J_PASSWORD"):
            Settings(_env_file=None)

    def test_production_requires_tls_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Production mode should reject plaintext Neo4j URIs without CA trust."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
        monkeypatch.delenv("NEO4J_CA_CERT", raising=False)
        monkeypatch.setenv("NEO4J_PASSWORD", "secure-password")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")

        with pytest.raises(ValidationError, match="NEO4J_URI"):
            Settings(_env_file=None)

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

    def test_production_allows_bolt_uri_with_custom_ca(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom CA config makes bolt:// use encrypted driver settings."""
        monkeypatch.setenv("ZAXY_ENV", "production")
        monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "secure-password")
        monkeypatch.setenv("NEO4J_CA_CERT", "/ssl/bolt/trusted/public.crt")
        monkeypatch.setenv("MCP_ADMIN_TOKEN", "admin-token")

        settings = Settings(_env_file=None)

        assert settings.neo4j_uri == "bolt://neo4j:7687"
