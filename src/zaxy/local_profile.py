"""Offline local retrieval profile helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zaxy.config import Settings
from zaxy.embedding import build_embedding_provider
from zaxy.query import build_reranker

_LOCAL_PROFILE_VALUES = {
    "ZAXY_ENV": "development",
    "PROJECTION_BACKEND": "neo4j",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "testpassword",
    "NEO4J_DATABASE": "neo4j",
    "NEO4J_AUTO_START": "true",
    "NEO4J_CA_CERT": "",
    "NEO4J_PASSWORD_FILE": "",
    "NEO4J_TRUST_ALL": "false",
    "EMBEDDING_ENABLED": "true",
    "EMBEDDING_PROVIDER": "hash",
    "EMBEDDING_DIMENSION": "1536",
    "RERANKER_PROVIDER": "lexical",
}


def render_local_profile(*, projection_backend: str = "neo4j") -> str:
    """Return an .env-style offline retrieval profile."""
    normalized_backend = projection_backend.casefold().strip()
    if normalized_backend not in {"neo4j", "pggraph", "embedded", "latticedb"}:
        raise ValueError("projection_backend must be one of: neo4j, pggraph, embedded, latticedb")
    values = dict(_LOCAL_PROFILE_VALUES)
    values["PROJECTION_BACKEND"] = normalized_backend
    if normalized_backend == "embedded":
        values["NEO4J_AUTO_START"] = "false"
        values["PGGRAPH_AUTO_START"] = "false"
        values["EMBEDDED_GRAPH_PATH"] = ".eventloom/projections/embedded.kuzu"
    elif normalized_backend == "pggraph":
        values["NEO4J_AUTO_START"] = "false"
        values["PGGRAPH_AUTO_START"] = "true"
    lines = [
        "# Zaxy offline local retrieval profile",
        "# Deterministic embeddings and lexical reranking require no hosted secrets.",
        *[f"{key}={value}" for key, value in values.items()],
        "",
    ]
    return "\n".join(lines)


def write_local_profile(path: Path, *, projection_backend: str = "neo4j", force: bool = False) -> Path:
    """Write the offline retrieval profile to path."""
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    path.write_text(render_local_profile(projection_backend=projection_backend), encoding="utf-8")
    return path


def check_local_profile() -> dict[str, Any]:
    """Validate that deterministic local embedding and reranker providers build."""
    settings_values: dict[str, Any] = {
        "_env_file": None,
        "zaxy_env": "development",
        "embedding_enabled": True,
        "embedding_provider": "hash",
        "embedding_dimension": 1536,
        "reranker_provider": "lexical",
    }
    settings = Settings(**settings_values)
    embedding_provider = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "embedding_dimension": settings.embedding_dimension,
        "embedding_ready": embedding_provider is not None,
        "reranker_provider": settings.reranker_provider,
        "reranker_ready": reranker is not None,
        "hosted_secrets_required": False,
    }
