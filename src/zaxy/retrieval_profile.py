"""Named retrieval profiles for auditable retrieval configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalProfile:
    """Resolved retrieval configuration used by core query paths."""

    name: str
    embedding_provider: str
    embedding_model: str | None
    embedding_dimension: int
    reranker_provider: str
    scoring_profile: str
    lanes: tuple[str, ...]
    hosted: bool = False
    experimental: bool = False

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a stable diagnostics representation."""
        return {
            "name": self.name,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "reranker_provider": self.reranker_provider,
            "scoring_profile": self.scoring_profile,
            "lanes": list(self.lanes),
            "hosted": self.hosted,
            "experimental": self.experimental,
        }


def resolve_retrieval_profile(settings: Any) -> RetrievalProfile:
    """Resolve configured retrieval knobs into a named profile."""
    name = str(getattr(settings, "retrieval_profile", "local_fast")).casefold().replace("-", "_")
    if name == "local_fast" and _has_explicit_non_profile_settings(settings):
        return _custom_profile(settings, name="custom")
    if name == "local_fast":
        return RetrievalProfile(
            name="local_fast",
            embedding_provider="hash",
            embedding_model=None,
            embedding_dimension=1536,
            reranker_provider="lexical",
            scoring_profile="balanced",
            lanes=("bm25", "hash_vector", "verbatim", "graph", "lexical_rerank"),
        )
    if name == "local_sota":
        return RetrievalProfile(
            name="local_sota",
            embedding_provider="sentence-transformers",
            embedding_model="BAAI/bge-m3",
            embedding_dimension=1024,
            reranker_provider="lexical",
            scoring_profile="recall",
            lanes=("bm25", "dense_vector", "verbatim", "graph", "lexical_rerank"),
            experimental=True,
        )
    if name == "hosted_sota":
        return RetrievalProfile(
            name="hosted_sota",
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            embedding_dimension=3072,
            reranker_provider="openai",
            scoring_profile="recall",
            lanes=("bm25", "hosted_dense_vector", "verbatim", "graph", "hosted_rerank"),
            hosted=True,
            experimental=True,
        )
    if name == "custom":
        return _custom_profile(settings, name="custom")
    raise ValueError(
        "RETRIEVAL_PROFILE must be one of: local_fast, local_sota, hosted_sota, custom"
    )


def apply_retrieval_profile(settings: Any, profile: RetrievalProfile) -> Any:
    """Return settings with profile-controlled retrieval fields resolved."""
    if profile.name == "custom":
        return settings
    updates = {
        "embedding_provider": profile.embedding_provider,
        "embedding_dimension": profile.embedding_dimension,
        "embedding_sentence_transformer_model": profile.embedding_model
        or getattr(settings, "embedding_sentence_transformer_model", ""),
        "openai_embedding_model": profile.embedding_model
        or getattr(settings, "openai_embedding_model", ""),
        "reranker_provider": profile.reranker_provider,
        "query_scoring_profile": profile.scoring_profile,
    }
    if hasattr(settings, "model_copy"):
        return settings.model_copy(update=updates)
    return settings.copy(update=updates)


def _custom_profile(settings: Any, *, name: str) -> RetrievalProfile:
    return RetrievalProfile(
        name=name,
        embedding_provider=str(getattr(settings, "embedding_provider", "hash")),
        embedding_model=_configured_embedding_model(settings),
        embedding_dimension=int(getattr(settings, "embedding_dimension", 1536)),
        reranker_provider=str(getattr(settings, "reranker_provider", "none")),
        scoring_profile=str(getattr(settings, "query_scoring_profile", "balanced")),
        lanes=_custom_lanes(settings),
        hosted=_custom_hosted(settings),
        experimental=True,
    )


def _has_explicit_non_profile_settings(settings: Any) -> bool:
    return any(
        getattr(settings, field, default) != default
        for field, default in (
            ("embedding_provider", "hash"),
            ("embedding_dimension", 1536),
            ("reranker_provider", "none"),
            ("query_scoring_profile", "balanced"),
        )
    )


def _configured_embedding_model(settings: Any) -> str | None:
    provider = str(getattr(settings, "embedding_provider", "")).casefold()
    if provider == "openai":
        return str(getattr(settings, "openai_embedding_model", ""))
    if provider in {"local-http", "local_http", "http"}:
        model = getattr(settings, "embedding_http_model", None)
        return str(model) if model else None
    if provider in {
        "sentence-transformers",
        "sentence_transformers",
        "sentence-transformer",
        "sentence_transformer",
        "local-model",
        "local_model",
    }:
        return str(getattr(settings, "embedding_sentence_transformer_model", ""))
    return None


def _custom_lanes(settings: Any) -> tuple[str, ...]:
    lanes = ["bm25"]
    provider = str(getattr(settings, "embedding_provider", "hash")).casefold()
    if provider == "hash":
        lanes.append("hash_vector")
    elif provider:
        lanes.append("dense_vector")
    if bool(getattr(settings, "context_verbatim_enabled", True)):
        lanes.append("verbatim")
    lanes.append("graph")
    reranker = str(getattr(settings, "reranker_provider", "none")).casefold()
    if reranker not in {"", "none"}:
        lanes.append(f"{reranker}_rerank")
    return tuple(lanes)


def _custom_hosted(settings: Any) -> bool:
    embedding_provider = str(getattr(settings, "embedding_provider", "")).casefold()
    reranker_provider = str(getattr(settings, "reranker_provider", "")).casefold()
    return embedding_provider == "openai" or reranker_provider == "openai"
