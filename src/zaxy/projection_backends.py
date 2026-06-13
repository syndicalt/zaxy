"""Projection backend construction.

Embedded LadybugDB is the default local projection. Neo4j remains the sidecar control
backend, and pgGraph is exposed only as an explicit experimental target until it
passes the same contract and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zaxy.projection import ProjectionStore

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ProjectionBackendConfig:
    """Configuration needed to construct a projection backend."""

    backend: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_ca_cert: str | None
    neo4j_trust_all: bool
    pggraph_dsn: str | None = None
    embedded_graph_path: Path | None = None
    latticedb_path: Path | None = None
    embedding_dimension: int = 1536


def build_projection_store(config: ProjectionBackendConfig) -> ProjectionStore:
    """Build a projection store from a backend config."""
    backend = config.backend.casefold().strip()
    if backend == "neo4j":
        from zaxy.graph import GraphStore

        return GraphStore(
            config.neo4j_uri,
            config.neo4j_user,
            config.neo4j_password,
            ca_cert=config.neo4j_ca_cert,
            trust_all=config.neo4j_trust_all,
        )
    if backend == "pggraph":
        from zaxy.pggraph_store import PgGraphStore

        if not config.pggraph_dsn:
            raise ValueError("pgGraph backend requires pggraph_dsn")
        return PgGraphStore(config.pggraph_dsn)
    if backend == "embedded":
        from zaxy.embedded_graph_store import EmbeddedGraphStore

        if config.embedded_graph_path is None:
            raise ValueError("embedded backend requires embedded_graph_path")
        return EmbeddedGraphStore(config.embedded_graph_path)
    if backend == "latticedb":
        from zaxy.latticedb_store import LatticeDBStore

        if config.latticedb_path is None:
            raise ValueError("LatticeDB backend requires latticedb_path")
        return LatticeDBStore(config.latticedb_path, vector_dimensions=config.embedding_dimension)
    raise ValueError("projection backend must be one of: embedded, neo4j, pggraph, latticedb")
