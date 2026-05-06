"""Tests for the setup script production secret scaffolding."""

from __future__ import annotations

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
    """Production env generation should point config at generated secret files."""
    script = Path("scripts/setup.sh").read_text(encoding="utf-8")

    assert "NEO4J_PASSWORD_FILE=secrets/neo4j_password.txt" in script
    assert "MCP_ADMIN_TOKEN_FILE=secrets/mcp_admin_token.txt" in script
    assert "MCP_REMOTE_AUTH_TOKEN_FILE=secrets/mcp_remote_auth_token.txt" in script
    assert "OPENAI_API_KEY_FILE=secrets/openai_api_key.txt" in script
    assert "PATHLIGHT_ACCESS_TOKEN_FILE=secrets/pathlight_access_token.txt" in script


def test_generate_certs_script_writes_neo4j_tls_layout() -> None:
    """Generated certs should match Neo4j's Bolt SSL policy file names."""
    script = Path("scripts/generate-certs.sh").read_text(encoding="utf-8")

    assert '"${OUTPUT_DIR}/neo4j/private.key"' in script
    assert '"${OUTPUT_DIR}/neo4j/public.crt"' in script
    assert '"${OUTPUT_DIR}/neo4j/trusted/public.crt"' in script
    assert '"${OUTPUT_DIR}/neo4j/revoked"' in script
