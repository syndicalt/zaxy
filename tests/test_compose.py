"""Tests for Docker Compose production hardening."""

from __future__ import annotations

from pathlib import Path


def test_production_compose_uses_secret_files() -> None:
    """Production compose should mount secrets instead of plaintext credentials."""
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert "NEO4J_AUTH_FILE: /run/secrets/neo4j_auth" in compose
    assert "NEO4J_PASSWORD_FILE: /run/secrets/neo4j_password" in compose
    assert "MCP_ADMIN_TOKEN_FILE: /run/secrets/mcp_admin_token" in compose
    assert "PATHLIGHT_ACCESS_TOKEN_FILE: /run/secrets/pathlight_access_token" in compose
    assert "NEO4J_PASSWORD:" not in compose


def test_production_compose_requires_production_security_mode() -> None:
    """Production compose should run with strict config validation enabled."""
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert "ZAXY_ENV: production" in compose
    assert "NEO4J_URI: \"${NEO4J_URI:-bolt://neo4j:7687}\"" in compose
    assert "NEO4J_CA_CERT: \"${NEO4J_CA_CERT:-/ssl/bolt/trusted/public.crt}\"" in compose


def test_compose_has_tls_integration_service() -> None:
    """Base compose should include a TLS Neo4j test service."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "neo4j-tls:" in compose
    assert "127.0.0.1:7689:7687" in compose
    assert "NEO4J_server_bolt_tls__level: REQUIRED" in compose
    assert "NEO4J_dbms_ssl_policy_bolt_enabled: \"true\"" in compose
    assert "./.certs/neo4j:/ssl/bolt:ro" in compose


def test_production_compose_enables_neo4j_bolt_tls() -> None:
    """Production compose should enable server TLS and mount the CA for Zaxy."""
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert "NEO4J_server_bolt_tls__level: REQUIRED" in compose
    assert "NEO4J_dbms_ssl_policy_bolt_enabled: \"true\"" in compose
    assert "${NEO4J_TLS_CERTS_DIR:-./.certs/neo4j}:/ssl/bolt:ro" in compose
