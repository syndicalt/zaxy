"""Tests for Docker Compose production hardening."""

from __future__ import annotations

from pathlib import Path


def test_production_compose_uses_secret_files() -> None:
    """Production compose should mount secrets instead of plaintext credentials."""
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert "NEO4J_AUTH_FILE: /run/secrets/neo4j_auth" in compose
    assert "MCP_ADMIN_TOKEN_FILE: /run/secrets/mcp_admin_token" in compose
    assert "PATHLIGHT_ACCESS_TOKEN_FILE: /run/secrets/pathlight_access_token" in compose
    assert "NEO4J_PASSWORD:" not in compose


def test_production_compose_requires_production_security_mode() -> None:
    """Production compose should run with strict config validation enabled."""
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert "ZAXY_ENV: production" in compose
    assert "PROJECTION_BACKEND: \"${PROJECTION_BACKEND:-embedded}\"" in compose
    assert "EMBEDDED_GRAPH_PATH: \"${EMBEDDED_GRAPH_PATH:-/app/.eventloom/projections/embedded.kuzu}\"" in compose
    assert "NEO4J_AUTO_START: \"${NEO4J_AUTO_START:-false}\"" in compose
    assert "PGGRAPH_AUTO_START: \"${PGGRAPH_AUTO_START:-false}\"" in compose


def test_production_compose_embedded_service_does_not_depend_on_neo4j() -> None:
    """The default production service should not require optional Neo4j sidecars."""
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")
    zaxy_section = compose.split("  zaxy:", maxsplit=1)[1].split("  zaxy-neo4j:", maxsplit=1)[0]

    assert "depends_on:" not in zaxy_section
    assert "NEO4J_PASSWORD_FILE" not in zaxy_section
    assert "/ssl/bolt" not in zaxy_section


def test_production_compose_keeps_neo4j_sidecar_profile() -> None:
    """Neo4j should remain available as an explicit production sidecar profile."""
    compose = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert 'profiles: ["neo4j"]' in compose
    assert "NEO4J_PASSWORD_FILE: /run/secrets/neo4j_password" in compose
    assert "NEO4J_URI: \"${NEO4J_URI:-bolt://neo4j:7687}\"" in compose
    assert "NEO4J_CA_CERT: \"${NEO4J_CA_CERT:-/ssl/bolt/trusted/public.crt}\"" in compose


def test_compose_has_tls_integration_service() -> None:
    """Base compose should include a TLS Neo4j test service."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "neo4j-tls:" in compose
    assert 'profiles: ["integration"]' in compose
    assert "127.0.0.1:7689:7687" in compose
    assert "NEO4J_server_bolt_tls__level: REQUIRED" in compose
    assert "NEO4J_dbms_ssl_policy_bolt_enabled: \"true\"" in compose
    assert "./.certs/neo4j:/ssl/bolt:ro" in compose


def test_default_compose_zaxy_uses_embedded_without_neo4j_dependency() -> None:
    """Default development compose should start embedded Zaxy without Neo4j."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    zaxy_section = compose.split("  zaxy:", maxsplit=1)[1].split("\nvolumes:", maxsplit=1)[0]

    assert "PROJECTION_BACKEND: \"${PROJECTION_BACKEND:-embedded}\"" in zaxy_section
    assert "EMBEDDED_GRAPH_PATH: \"${EMBEDDED_GRAPH_PATH:-/app/.eventloom/projections/embedded.kuzu}\"" in zaxy_section
    assert "NEO4J_AUTO_START: \"${NEO4J_AUTO_START:-false}\"" in zaxy_section
    assert "PGGRAPH_AUTO_START: \"${PGGRAPH_AUTO_START:-false}\"" in zaxy_section
    assert "depends_on:" not in zaxy_section


def test_default_compose_keeps_neo4j_services_behind_profiles() -> None:
    """Neo4j services should be explicit sidecars, not default compose startup."""
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    neo4j_section = compose.split("  neo4j:", maxsplit=1)[1].split("  neo4j-test:", maxsplit=1)[0]
    neo4j_test_section = compose.split("  neo4j-test:", maxsplit=1)[1].split("  neo4j-tls:", maxsplit=1)[0]
    neo4j_tls_section = compose.split("  neo4j-tls:", maxsplit=1)[1].split("  zaxy:", maxsplit=1)[0]

    assert 'profiles: ["neo4j"]' in neo4j_section
    assert 'profiles: ["integration"]' in neo4j_test_section
    assert 'profiles: ["integration"]' in neo4j_tls_section


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


def test_dockerfile_creates_embedded_projection_directory_without_stale_volumes() -> None:
    """The production image should prepare embedded storage, not old sidecar volume paths."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "mkdir -p /app/.eventloom/projections" in dockerfile
    assert "/app/.volumes" not in dockerfile


def test_dockerfile_preseeds_ladybugdb_vector_extension_into_cache() -> None:
    """The image should bake the LadybugDB vector extension so containerized ANN
    works without a runtime network fetch (2.3 LadybugDB downloads it on first use)."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    # HOME anchored under the writable app tree so HOME/.lbdb is zaxy-owned.
    assert "ENV HOME=/app" in dockerfile
    # Extension pre-seeded at build time...
    assert "INSTALL vector" in dockerfile
    assert "LOAD vector" in dockerfile
    # ...before the chown, so the pre-seeded cache is handed to the non-root user.
    assert dockerfile.index("INSTALL vector") < dockerfile.index("chown -R zaxy:zaxy /app")
    # ...and the pre-seed happens after the package (which provides ladybug) installs.
    assert dockerfile.index("pip install --no-cache-dir /tmp/*.whl") < dockerfile.index("INSTALL vector")


def test_sse_transport_host_is_configurable() -> None:
    """Container deployments need the SSE listener host to be explicit."""
    cli = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(Path("src/zaxy/cli").glob("*.py"))
    )
    server = Path("src/zaxy/mcp_server.py").read_text(encoding="utf-8")

    assert "host: str = typer.Option" in cli
    assert "mcp_server.main_sse(port=port, host=host)" in cli
    assert "async def main_sse(port: int = 8080, host: str = \"127.0.0.1\")" in server
    assert "uvicorn.Config(starlette_app, host=host, port=port" in server
