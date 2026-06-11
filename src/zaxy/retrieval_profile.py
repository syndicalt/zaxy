"""Named retrieval profiles for auditable retrieval configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalProfile:
    """Resolved retrieval configuration used by core query paths.

    The cognitive flags (``salience_ranking``, ``cue_blending``,
    ``graph_walk``) default to ``False`` on every plain profile so
    plain-profile behavior is byte-identical to the pre-cognitive contract;
    only the ``cognitive`` profile (the 2.1.0 default) enables them.
    """

    name: str
    embedding_provider: str
    embedding_model: str | None
    embedding_dimension: int
    reranker_provider: str
    scoring_profile: str
    lanes: tuple[str, ...]
    hosted: bool = False
    experimental: bool = False
    #: Multiply checkout relevance by replayed salience and apply the
    #: attenuation floor (cognitive profile only).
    salience_ranking: bool = False
    #: Blend encoding-specificity cue overlap into ranking (cognitive only).
    cue_blending: bool = False
    #: Run the bounded personalized-PageRank stage over graph candidates
    #: (cognitive profile only, and only when the store exposes adjacency).
    graph_walk: bool = False

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a stable diagnostics representation.

        The ``cognitive`` block is emitted only when at least one cognitive
        flag is enabled, keeping diagnostics for pre-existing profiles
        byte-identical to the pre-cognitive contract.
        """
        diagnostics: dict[str, Any] = {
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
        if self.salience_ranking or self.cue_blending or self.graph_walk:
            diagnostics["cognitive"] = {
                "salience_ranking": self.salience_ranking,
                "cue_blending": self.cue_blending,
                "graph_walk": self.graph_walk,
            }
        return diagnostics


def resolve_retrieval_profile(settings: Any) -> RetrievalProfile:
    """Resolve configured retrieval knobs into a named profile."""
    name = str(getattr(settings, "retrieval_profile", "local_fast")).casefold().replace("-", "_")
    if name == "local_fast" and _has_explicit_non_profile_settings(settings):
        return _custom_profile(settings, name="custom")
    if (
        name == "cognitive"
        and _retrieval_profile_defaulted(settings)
        and _has_explicit_non_profile_settings(settings)
    ):
        # The 2.1.0 default flip to "cognitive" must not override explicitly
        # configured embedding/reranker/scoring knobs. When the profile field
        # itself was left unset, customized knobs keep resolving to the
        # "custom" profile exactly as they did when the default was
        # local_fast. An explicit RETRIEVAL_PROFILE=cognitive still wins over
        # the knobs, matching every other named profile.
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
    if name == "cognitive":
        # Promoted to the settings default in 2.1.0 (and out of experimental
        # status) backed by the internal forgetting lane: exact cold-start
        # parity with local_fast, no-recall-loss 1.0, exemptions 1.0.
        return RetrievalProfile(
            name="cognitive",
            embedding_provider="hash",
            embedding_model=None,
            embedding_dimension=1536,
            reranker_provider="lexical",
            scoring_profile="balanced",
            lanes=("bm25", "hash_vector", "verbatim", "graph", "graph_walk", "lexical_rerank"),
            salience_ranking=True,
            cue_blending=True,
            graph_walk=True,
        )
    if name == "custom":
        return _custom_profile(settings, name="custom")
    raise ValueError(
        "RETRIEVAL_PROFILE must be one of: cognitive, local_fast, local_sota, hosted_sota, custom"
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


def _retrieval_profile_defaulted(settings: Any) -> bool:
    """Return True when the settings object left ``retrieval_profile`` unset.

    Pydantic models expose explicitly provided fields (including values from
    environment sources) through ``model_fields_set``. Duck-typed settings
    objects without that attribute are treated as explicit so a stated
    profile name is always honored.
    """
    fields_set = getattr(settings, "model_fields_set", None)
    if fields_set is None:
        return False
    return "retrieval_profile" not in fields_set


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
