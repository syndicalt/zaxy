"""Central configuration for Zaxy.

All settings are loaded from environment variables with sensible defaults.
In Docker, values are injected via compose environment or Docker secrets.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    server_name: str = Field(
        default="zaxy-memory",
        description="MCP server name",
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
        return self.eventloom_dir / f"{name}.jsonl"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Use this in application code to avoid re-parsing env vars.
    """
    return Settings()
