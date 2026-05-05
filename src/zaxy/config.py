"""Central configuration for Zaxy.

All settings are loaded from environment variables with sensible defaults.
In Docker, values are injected via compose environment or Docker secrets.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name",
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
    pathlight_project_id: str | None = Field(
        default=None,
        description="Pathlight project identifier",
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

    @model_validator(mode="after")
    def _validate_production_security(self) -> "Settings":
        """Reject known-insecure production defaults."""
        if self.zaxy_env.lower() == "production":
            if self.neo4j_password == "testpassword":
                raise ValueError("NEO4J_PASSWORD must be overridden in production")
            if self.neo4j_uri.startswith("bolt://"):
                raise ValueError("NEO4J_URI must use TLS in production")
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
