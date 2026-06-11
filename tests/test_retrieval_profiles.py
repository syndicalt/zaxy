"""Tests for named retrieval profile resolution."""

from __future__ import annotations

import pytest

from zaxy.config import Settings
from zaxy.retrieval_profile import resolve_retrieval_profile


def test_default_settings_resolve_to_the_cognitive_profile() -> None:
    """2.1.0 default flip: unconfigured settings resolve to cognitive."""
    profile = resolve_retrieval_profile(Settings(_env_file=None))

    assert profile.name == "cognitive"


def test_local_fast_profile_resolves_existing_deterministic_defaults() -> None:
    """The explicit local_fast opt-out should describe the lightweight local path."""
    # Pinned explicitly since 2.1.0 flipped the default profile to cognitive.
    profile = resolve_retrieval_profile(Settings(_env_file=None, retrieval_profile="local_fast"))

    assert profile.name == "local_fast"
    assert profile.embedding_provider == "hash"
    assert profile.reranker_provider == "lexical"
    assert profile.scoring_profile == "balanced"
    assert profile.lanes == ("bm25", "hash_vector", "verbatim", "graph", "lexical_rerank")
    assert profile.hosted is False
    assert profile.experimental is False


def test_local_sota_profile_prefers_bge_m3_style_local_models() -> None:
    """The SOTA local profile should be explicit and dependency-gated."""
    settings = Settings(_env_file=None, retrieval_profile="local_sota")

    profile = resolve_retrieval_profile(settings)

    assert profile.name == "local_sota"
    assert profile.embedding_provider == "sentence-transformers"
    assert profile.embedding_model == "BAAI/bge-m3"
    assert profile.embedding_dimension == 1024
    assert profile.reranker_provider == "lexical"
    assert profile.scoring_profile == "recall"
    assert profile.lanes == (
        "bm25",
        "dense_vector",
        "verbatim",
        "graph",
        "lexical_rerank",
    )
    assert profile.hosted is False
    assert profile.experimental is True


def test_hosted_sota_profile_prefers_openai_compatible_high_quality_path() -> None:
    """Hosted SOTA profile should centralize hosted embedding/reranker defaults."""
    settings = Settings(_env_file=None, retrieval_profile="hosted_sota")

    profile = resolve_retrieval_profile(settings)

    assert profile.name == "hosted_sota"
    assert profile.embedding_provider == "openai"
    assert profile.embedding_model == "text-embedding-3-large"
    assert profile.embedding_dimension == 3072
    assert profile.reranker_provider == "openai"
    assert profile.scoring_profile == "recall"
    assert profile.hosted is True


def test_custom_profile_preserves_explicit_settings() -> None:
    """Custom profile should avoid overriding explicitly configured knobs."""
    settings = Settings(
        _env_file=None,
        retrieval_profile="custom",
        embedding_provider="local-http",
        embedding_dimension=768,
        embedding_http_model="custom-embed",
        reranker_provider="http",
        query_scoring_profile="temporal",
    )

    profile = resolve_retrieval_profile(settings)

    assert profile.name == "custom"
    assert profile.embedding_provider == "local-http"
    assert profile.embedding_model == "custom-embed"
    assert profile.embedding_dimension == 768
    assert profile.reranker_provider == "http"
    assert profile.scoring_profile == "temporal"
    assert profile.experimental is True


def test_unknown_retrieval_profile_is_rejected() -> None:
    """Unknown profiles should fail loudly instead of silently downgrading retrieval."""
    with pytest.raises(ValueError, match="RETRIEVAL_PROFILE"):
        resolve_retrieval_profile(Settings(_env_file=None, retrieval_profile="random"))


def test_cognitive_profile_enables_cognitive_flags_over_local_fast_retrieval() -> None:
    """The cognitive profile should layer flags on the local_fast stack."""
    settings = Settings(_env_file=None, retrieval_profile="cognitive")

    profile = resolve_retrieval_profile(settings)

    assert profile.name == "cognitive"
    assert profile.embedding_provider == "hash"
    assert profile.reranker_provider == "lexical"
    assert profile.scoring_profile == "balanced"
    # Promoted out of experimental status with the 2.1.0 default flip.
    assert profile.experimental is False
    assert profile.salience_ranking is True
    assert profile.cue_blending is True
    assert profile.graph_walk is True
    assert "graph_walk" in profile.lanes
    assert profile.to_diagnostics()["cognitive"] == {
        "salience_ranking": True,
        "cue_blending": True,
        "graph_walk": True,
    }


def test_existing_profiles_keep_cognitive_flags_off_and_diagnostics_unchanged() -> None:
    """Pre-cognitive profiles must stay byte-identical: flags off, no new diagnostics key."""
    for name in ("local_fast", "local_sota", "hosted_sota"):
        profile = resolve_retrieval_profile(Settings(_env_file=None, retrieval_profile=name))

        assert profile.salience_ranking is False
        assert profile.cue_blending is False
        assert profile.graph_walk is False
        assert "cognitive" not in profile.to_diagnostics()
