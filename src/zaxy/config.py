"""Central configuration for Zaxy.

All settings are loaded from environment variables with sensible defaults.
In Docker, values are injected via compose environment or Docker secrets.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from zaxy.security import eventlog_path


class Settings(BaseSettings):
    """Production-ready configuration with env var support.

    Priority (highest first):
    1. Environment variables
    2. `.env` file
    3. Default values below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Allow extra env vars without error
    )

    # ------------------------------------------------------------------
    # Eventloom
    # ------------------------------------------------------------------
    eventloom_path: str = Field(
        default=".eventloom",
        description="Directory for append-only JSONL event logs",
    )
    eventloom_thread: str = Field(
        default="default",
        description="Default thread/session identifier",
    )

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: str = Field(
        default="testpassword",
        description="Neo4j password (override in production)",
    )
    neo4j_password_file: str | None = Field(
        default=None,
        description="Path to a file containing the Neo4j password",
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name",
    )
    neo4j_auto_start: bool = Field(
        default=True,
        description="Automatically start local Neo4j for localhost development MCP use",
    )
    neo4j_auto_start_image: str = Field(
        default="neo4j:5.26-community",
        description="Docker image used when automatically starting local Neo4j",
    )
    neo4j_auto_start_container: str = Field(
        default="zaxy-neo4j",
        description="Docker container name used for automatically started local Neo4j",
    )

    # ------------------------------------------------------------------
    # TLS / Security
    # ------------------------------------------------------------------
    neo4j_ca_cert: str | None = Field(
        default=None,
        description="Path to CA certificate for Neo4j TLS (bolt+s://)",
    )
    neo4j_trust_all: bool = Field(
        default=False,
        description="Trust all certificates (dev only, insecure)",
    )

    # ------------------------------------------------------------------
    # Pathlight
    # ------------------------------------------------------------------
    pathlight_url: str = Field(
        default="http://localhost:4100",
        description="Pathlight collector URL",
    )
    pathlight_enabled: bool = Field(
        default=False,
        description="Enable Pathlight trace emission and health checks",
    )
    pathlight_project_id: str | None = Field(
        default=None,
        description="Pathlight project identifier",
    )
    pathlight_access_token: str | None = Field(
        default=None,
        description="Optional Pathlight access token",
    )
    pathlight_access_token_file: str | None = Field(
        default=None,
        description="Path to a file containing the Pathlight access token",
    )
    trace_raw_queries: bool = Field(
        default=False,
        description="Emit raw query text to Pathlight traces (off by default)",
    )

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    server_name: str = Field(
        default="zaxy-memory",
        description="MCP server name",
    )
    zaxy_env: str = Field(
        default="development",
        description="Runtime environment: development, test, or production",
    )
    mcp_admin_token: str | None = Field(
        default=None,
        description="Optional admin token required for replay/invalidate tools",
    )
    mcp_admin_token_file: str | None = Field(
        default=None,
        description="Path to a file containing the MCP admin token",
    )
    mcp_remote_auth_token: str | None = Field(
        default=None,
        description="Bearer token required for remote MCP/SSE transport",
    )
    mcp_remote_auth_token_file: str | None = Field(
        default=None,
        description="Path to a file containing the remote MCP/SSE bearer token",
    )
    mcp_remote_session_header: str = Field(
        default="x-zaxy-session-id",
        description="HTTP header that scopes remote MCP/SSE clients to a session",
    )
    mcp_oidc_issuer: str | None = Field(
        default=None,
        description="OIDC issuer URL for remote MCP/SSE JWT validation",
    )
    mcp_oidc_audience: str | None = Field(
        default=None,
        description="Expected OIDC audience for remote MCP/SSE JWT validation",
    )
    mcp_oidc_jwks_url: str | None = Field(
        default=None,
        description="JWKS URL for remote MCP/SSE JWT signature validation",
    )
    mcp_oidc_required_scope: str = Field(
        default="zaxy:mcp",
        description="Required OAuth scope for remote MCP/SSE access",
    )
    mcp_oidc_session_claim: str = Field(
        default="zaxy_session",
        description="JWT claim containing the Zaxy session/tenant scope",
    )
    mcp_oidc_client_secret: str | None = Field(
        default=None,
        description="Optional OIDC client secret for future token introspection flows",
    )
    mcp_oidc_client_secret_file: str | None = Field(
        default=None,
        description="Path to a file containing the OIDC client secret",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Python logging level",
    )
    log_format: str = Field(
        default="console",
        description="Log format: console or json",
    )

    # ------------------------------------------------------------------
    # Query router
    # ------------------------------------------------------------------
    query_default_limit: int = Field(
        default=10,
        description="Default result limit for queries",
    )
    query_scoring_profile: str = Field(
        default="balanced",
        description="Query scoring profile: balanced, precision, recall, or temporal",
    )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_enabled: bool = Field(
        default=True,
        description="Generate embeddings for vector search",
    )
    embedding_provider: str = Field(
        default="hash",
        description="Embedding provider: hash or openai",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension; must match the Neo4j vector index",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for hosted embeddings",
    )
    openai_api_key_file: str | None = Field(
        default=None,
        description="Path to a file containing the OpenAI API key",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    reranker_provider: str = Field(
        default="none",
        description="Reranker provider: none, lexical, http, or openai",
    )
    reranker_url: str | None = Field(
        default=None,
        description="HTTP reranker endpoint URL",
    )
    reranker_api_key: str | None = Field(
        default=None,
        description="Optional bearer token for HTTP reranker endpoint",
    )
    reranker_api_key_file: str | None = Field(
        default=None,
        description="Path to a file containing the reranker bearer token",
    )
    openai_rerank_model: str = Field(
        default="gpt-5-mini",
        description="OpenAI-compatible chat model used for reranking",
    )

    def model_post_init(self, __context: Any) -> None:
        """Load Docker/Kubernetes-style secret files after env parsing."""
        self._load_secret_file("NEO4J_PASSWORD", "neo4j_password", "neo4j_password_file")
        self._load_secret_file("MCP_ADMIN_TOKEN", "mcp_admin_token", "mcp_admin_token_file")
        self._load_secret_file(
            "MCP_REMOTE_AUTH_TOKEN",
            "mcp_remote_auth_token",
            "mcp_remote_auth_token_file",
        )
        self._load_secret_file(
            "MCP_OIDC_CLIENT_SECRET",
            "mcp_oidc_client_secret",
            "mcp_oidc_client_secret_file",
        )
        self._load_secret_file("OPENAI_API_KEY", "openai_api_key", "openai_api_key_file")
        self._load_secret_file("RERANKER_API_KEY", "reranker_api_key", "reranker_api_key_file")
        self._load_secret_file(
            "PATHLIGHT_ACCESS_TOKEN",
            "pathlight_access_token",
            "pathlight_access_token_file",
        )

    def _load_secret_file(self, env_name: str, field_name: str, file_field_name: str) -> None:
        """Populate a sensitive field from ENV_NAME_FILE when direct env is absent."""
        file_env = f"{env_name}_FILE"
        file_path = os.getenv(file_env) or getattr(self, file_field_name)
        if not file_path:
            return
        current_value = getattr(self, field_name)
        default_value = type(self).model_fields[field_name].default
        if os.getenv(env_name) is not None or current_value not in (None, default_value):
            return
        try:
            value = Path(file_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"{file_env} could not be read: {file_path}") from exc
        object.__setattr__(self, field_name, value)

    @model_validator(mode="after")
    def _validate_production_security(self) -> Settings:
        """Reject known-insecure production defaults."""
        if self.zaxy_env.lower() == "production":
            if self.neo4j_password == "testpassword":
                raise ValueError("NEO4J_PASSWORD must be overridden in production")
            if self.neo4j_uri.startswith("bolt://") and not self.neo4j_ca_cert:
                raise ValueError("NEO4J_URI must use TLS or NEO4J_CA_CERT in production")
            if not self.mcp_admin_token:
                raise ValueError("MCP_ADMIN_TOKEN must be configured in production")
            has_static_remote_auth = bool(self.mcp_remote_auth_token)
            has_oidc_remote_auth = all(
                (self.mcp_oidc_issuer, self.mcp_oidc_audience, self.mcp_oidc_jwks_url)
            )
            if not has_static_remote_auth and not has_oidc_remote_auth:
                raise ValueError(
                    "MCP_REMOTE_AUTH_TOKEN or complete MCP_OIDC_ISSUER/"
                    "MCP_OIDC_AUDIENCE/MCP_OIDC_JWKS_URL must be configured in production"
                )
        return self

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def eventloom_dir(self) -> Path:
        """Return the Eventloom directory as a Path."""
        return Path(self.eventloom_path)

    def eventloom_log(self, thread: str | None = None) -> Path:
        """Return the JSONL path for a given thread."""
        name = thread or self.eventloom_thread
        return eventlog_path(self.eventloom_dir, name)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Use this in application code to avoid re-parsing env vars.
    """
    return Settings()
