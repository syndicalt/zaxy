"""Projection backend construction.

Neo4j is the production default. pgGraph is exposed only as an explicit
experimental target until an adapter passes the same contract and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

from zaxy.graph import GraphStore
from zaxy.projection import ProjectionStore


@dataclass(frozen=True)
class ProjectionBackendConfig:
    """Configuration needed to construct a projection backend."""

    backend: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_ca_cert: str | None
    neo4j_trust_all: bool


def build_projection_store(config: ProjectionBackendConfig) -> ProjectionStore:
    """Build a projection store from a backend config."""
    backend = config.backend.casefold().strip()
    if backend == "neo4j":
        return GraphStore(
            config.neo4j_uri,
            config.neo4j_user,
            config.neo4j_password,
            ca_cert=config.neo4j_ca_cert,
            trust_all=config.neo4j_trust_all,
        )
    if backend == "pggraph":
        raise NotImplementedError(
            "pgGraph backend is experimental and has no adapter yet. "
            "Keep PROJECTION_BACKEND=neo4j until pgGraph passes the projection "
            "contract and benchmark gates."
        )
    raise ValueError("projection backend must be one of: neo4j, pggraph")
