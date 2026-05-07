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


def test_dockerfile_copies_package_sources_before_building_wheel() -> None:
    """The production image build should include files required by pyproject."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY README.md ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert dockerfile.index("COPY README.md ./") < dockerfile.index("python -m build --wheel")
    assert dockerfile.index("COPY src ./src") < dockerfile.index("python -m build --wheel")


def test_dockerfile_defaults_to_remote_sse_server_on_exposed_port() -> None:
    """The production image should serve the advertised HTTP/SSE port."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert 'EXPOSE 8080' in dockerfile
    assert 'CMD ["serve", "--transport", "sse", "--host", "0.0.0.0", "--port", "8080"]' in dockerfile


def test_sse_transport_host_is_configurable() -> None:
    """Container deployments need the SSE listener host to be explicit."""
    cli = Path("src/zaxy/__main__.py").read_text(encoding="utf-8")
    server = Path("src/zaxy/mcp_server.py").read_text(encoding="utf-8")

    assert "host: str = typer.Option" in cli
    assert "mcp_server.main_sse(port=port, host=host)" in cli
    assert "async def main_sse(port: int = 8080, host: str = \"127.0.0.1\")" in server
    assert "uvicorn.Config(starlette_app, host=host, port=port" in server
