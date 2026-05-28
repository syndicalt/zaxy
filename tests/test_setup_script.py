"""Tests for the setup script production secret scaffolding."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def test_setup_script_writes_production_secrets_outside_env() -> None:
    """Production setup should create secret files instead of putting secrets in .env."""
    script = Path("scripts/setup.sh").read_text(encoding="utf-8")

    assert 'printf "neo4j/%s\\n" "${NEO4J_PASSWORD}" > "${SECRETS_DIR}/neo4j_auth.txt"' in script
    assert 'printf "%s\\n" "${NEO4J_PASSWORD}" > "${SECRETS_DIR}/neo4j_password.txt"' in script
    assert 'printf "%s\\n" "${MCP_ADMIN_TOKEN}" > "${SECRETS_DIR}/mcp_admin_token.txt"' in script
    assert 'if [[ "${MODE}" != "--production" ]]; then' in script
    assert 'NEO4J_CA_CERT="/ssl/bolt/trusted/public.crt"' in script
    assert 'echo "NEO4J_PASSWORD=${NEO4J_PASSWORD}"' in script


def test_setup_script_references_secret_files_in_production_env() -> None:
    """Production env generation should point default config at required secret files."""
    script = Path("scripts/setup.sh").read_text(encoding="utf-8")

    assert "NEO4J_PASSWORD_FILE=secrets/neo4j_password.txt" not in script
    assert "MCP_ADMIN_TOKEN_FILE=secrets/mcp_admin_token.txt" in script
    assert "MCP_REMOTE_AUTH_TOKEN_FILE=secrets/mcp_remote_auth_token.txt" in script
    assert "OPENAI_API_KEY_FILE=secrets/openai_api_key.txt" in script
    assert "PATHLIGHT_ACCESS_TOKEN_FILE=secrets/pathlight_access_token.txt" in script


def test_setup_script_generates_embedded_only_production_env(tmp_path: Path) -> None:
    """Generated production env should not include optional Neo4j config by default."""
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(Path("scripts/setup.sh"), scripts / "setup.sh")

    subprocess.run(
        ["bash", "scripts/setup.sh", "--production"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    env = (project / ".env").read_text(encoding="utf-8")
    assert "PROJECTION_BACKEND=embedded\n" in env
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu\n" in env
    assert "NEO4J_AUTO_START=false\n" in env
    assert "PGGRAPH_AUTO_START=false\n" in env
    assert "NEO4J_URI=" not in env
    assert "NEO4J_CA_CERT=" not in env
    assert "NEO4J_PASSWORD_FILE=" not in env

    validation = subprocess.run(
        ["bash", str(Path.cwd() / "scripts/validate-deployment.sh"), "--root", str(project)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Deployment validation passed" in validation.stdout


def test_setup_script_dev_env_targets_local_plain_bolt() -> None:
    """Development setup should generate the same embedded posture used by onboarding."""
    script = Path("scripts/setup.sh").read_text(encoding="utf-8")

    assert "PROJECTION_BACKEND=embedded" in script
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" in script
    assert "NEO4J_AUTO_START=false" in script
    assert "PGGRAPH_AUTO_START=false" in script
    assert 'NEO4J_URI="bolt://localhost:7687"' in script
    assert 'NEO4J_CA_CERT=""' in script
    assert "NEO4J_PASSWORD_FILE=" in script
    assert "NEO4J_TRUST_ALL=false" in script


def test_setup_script_production_env_defaults_to_embedded_projection() -> None:
    """Production setup should not select optional sidecar projection backends by default."""
    script = Path("scripts/setup.sh").read_text(encoding="utf-8")

    assert "PROJECTION_BACKEND=embedded" in script
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" in script
    assert "NEO4J_AUTO_START=false" in script
    assert "PGGRAPH_AUTO_START=false" in script
    assert "docker compose -f docker-compose.prod.yml up -d   # embedded production service" in script
    assert "docker compose -f docker-compose.prod.yml --profile neo4j up -d zaxy-neo4j" in script


def test_setup_script_does_not_require_docker_for_embedded_setup() -> None:
    """Embedded setup should not hard-fail when optional sidecar tooling is absent."""
    script = Path("scripts/setup.sh").read_text(encoding="utf-8")

    assert "Docker unavailable; skipping optional sidecar validation" in script
    assert "Docker Compose unavailable; skipping optional sidecar validation" in script
    assert "Optional Docker sidecar tooling available" in script
    assert "docker compose --profile integration up -d neo4j-test neo4j-tls" in script


def test_setup_script_does_not_create_unused_neo4j_volume_directories() -> None:
    """Embedded setup should not create stale local Neo4j volume directories."""
    script = Path("scripts/setup.sh").read_text(encoding="utf-8")

    assert '.volumes/neo4j_data' not in script
    assert '.volumes/neo4j_logs' not in script


def test_env_example_defaults_to_embedded_without_sidecar_autostart() -> None:
    """The sample env file should advertise the embedded no-sidecar default."""
    example = Path(".env.example").read_text(encoding="utf-8")

    assert "PROJECTION_BACKEND=embedded" in example
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" in example
    assert "NEO4J_AUTO_START=false" in example
    assert "PGGRAPH_AUTO_START=false" in example


def test_generate_certs_script_writes_neo4j_tls_layout() -> None:
    """Generated certs should match Neo4j's Bolt SSL policy file names."""
    script = Path("scripts/generate-certs.sh").read_text(encoding="utf-8")

    assert '"${OUTPUT_DIR}/neo4j/private.key"' in script
    assert '"${OUTPUT_DIR}/neo4j/public.crt"' in script
    assert '"${OUTPUT_DIR}/neo4j/trusted/public.crt"' in script
    assert '"${OUTPUT_DIR}/neo4j/revoked"' in script
