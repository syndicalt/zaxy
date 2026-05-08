"""Offline local retrieval profile helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zaxy.config import Settings
from zaxy.embedding import build_embedding_provider
from zaxy.query import build_reranker

_LOCAL_PROFILE_VALUES = {
    "ZAXY_ENV": "development",
    "EMBEDDING_ENABLED": "true",
    "EMBEDDING_PROVIDER": "hash",
    "EMBEDDING_DIMENSION": "1536",
    "RERANKER_PROVIDER": "lexical",
    "NEO4J_AUTO_START": "true",
}


def render_local_profile() -> str:
    """Return an .env-style offline retrieval profile."""
    lines = [
        "# Zaxy offline local retrieval profile",
        "# Deterministic embeddings and lexical reranking require no hosted secrets.",
        *[f"{key}={value}" for key, value in _LOCAL_PROFILE_VALUES.items()],
        "",
    ]
    return "\n".join(lines)


def write_local_profile(path: Path, *, force: bool = False) -> Path:
    """Write the offline retrieval profile to path."""
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    path.write_text(render_local_profile(), encoding="utf-8")
    return path


def check_local_profile() -> dict[str, Any]:
    """Validate that deterministic local embedding and reranker providers build."""
    settings = Settings(
        _env_file=None,
        zaxy_env="development",
        embedding_enabled=True,
        embedding_provider="hash",
        embedding_dimension=1536,
        reranker_provider="lexical",
    )
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
