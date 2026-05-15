"""Tests for zaxy.embedding."""

from __future__ import annotations

import math
import types

import httpx
import pytest

from zaxy.config import Settings
from zaxy.embedding import (
    HashEmbeddingProvider,
    LocalHTTPEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
    _load_sentence_transformer,
    build_embedding_provider,
    embed_extraction,
    entity_embedding_text,
)
from zaxy.extract import ExtractedEntity, ExtractionResult


class TestHashEmbeddingProvider:
    """Tests for the deterministic local embedding provider."""

    def test_embed_is_deterministic(self) -> None:
        provider = HashEmbeddingProvider(dimension=32)
        assert provider.embed("Ship MVP") == provider.embed("Ship MVP")

    def test_embed_has_configured_dimension(self) -> None:
        provider = HashEmbeddingProvider(dimension=48)
        assert len(provider.embed("Ship MVP")) == 48

    def test_embed_is_unit_normalized(self) -> None:
        provider = HashEmbeddingProvider(dimension=32)
        vector = provider.embed("Ship MVP")
        norm = math.sqrt(sum(value * value for value in vector))
        assert norm == pytest.approx(1.0)

    def test_different_text_changes_vector(self) -> None:
        provider = HashEmbeddingProvider(dimension=32)
        assert provider.embed("Ship MVP") != provider.embed("Design homepage")

    def test_rejects_invalid_dimension(self) -> None:
        try:
            HashEmbeddingProvider(dimension=0)
        except ValueError as exc:
            assert "dimension" in str(exc)
        else:  # pragma: no cover - assertion guard
            raise AssertionError("Expected invalid dimension to raise")


class TestExtractionEmbedding:
    """Tests for embedding extracted entities."""

    def test_entity_text_includes_summary(self) -> None:
        entity = ExtractedEntity(
            name="Ship MVP",
            entity_type="goal",
            observed_at="2024-01-01T00:00:00Z",
            summary="Get product to market",
        )

        assert entity_embedding_text(entity) == "Ship MVP (goal) Get product to market"

    def test_embed_extraction_adds_entity_embeddings(self) -> None:
        provider = HashEmbeddingProvider(dimension=16)
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Ship MVP",
                    entity_type="goal",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="Get product to market",
                )
            ],
            edges=[],
            source_event_seq=1,
        )

        embedded = embed_extraction(result, provider)

        assert embedded.source_event_seq == result.source_event_seq
        assert embedded.edges == result.edges
        assert embedded.entities[0].embedding == provider.embed(
            "Ship MVP (goal) Get product to market"
        )

    def test_embed_extraction_preserves_existing_embeddings(self) -> None:
        provider = HashEmbeddingProvider(dimension=16)
        existing = [0.1, 0.2, 0.3]
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Ship MVP",
                    entity_type="goal",
                    observed_at="2024-01-01T00:00:00Z",
                    embedding=existing,
                )
            ],
            edges=[],
            source_event_seq=1,
        )

        embedded = embed_extraction(result, provider)

        assert embedded.entities[0].embedding == existing


class TestEmbeddingProviderFactory:
    """Tests for embedding provider selection."""

    def test_factory_returns_hash_provider_by_default(self) -> None:
        settings = Settings(_env_file=None)

        provider = build_embedding_provider(settings)

        assert isinstance(provider, HashEmbeddingProvider)

    def test_factory_returns_none_when_embeddings_disabled(self) -> None:
        settings = Settings(_env_file=None, embedding_enabled=False)

        assert build_embedding_provider(settings) is None

    def test_factory_returns_openai_provider_when_configured(self) -> None:
        settings = Settings(
            _env_file=None,
            embedding_provider="openai",
            openai_api_key="test-key",
        )

        provider = build_embedding_provider(settings)

        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_factory_returns_local_http_provider_when_configured(self) -> None:
        settings = Settings(
            _env_file=None,
            embedding_provider="local-http",
            embedding_http_url="http://localhost:8080/embed",
            embedding_http_model="bge-small",
        )

        provider = build_embedding_provider(settings)

        assert isinstance(provider, LocalHTTPEmbeddingProvider)

    def test_factory_returns_sentence_transformers_provider_when_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeModel:
            def encode(
                self,
                text: str,
                *,
                normalize_embeddings: bool,
            ) -> list[float]:
                del text, normalize_embeddings
                return [0.1, 0.2, 0.3]

        def fake_loader(model_name: str) -> FakeModel:
            assert model_name == "sentence-transformers/all-MiniLM-L6-v2"
            return FakeModel()

        monkeypatch.setattr("zaxy.embedding._load_sentence_transformer", fake_loader)
        settings = Settings(
            _env_file=None,
            embedding_provider="sentence-transformers",
            embedding_dimension=3,
        )

        provider = build_embedding_provider(settings)

        assert isinstance(provider, SentenceTransformersEmbeddingProvider)

    def test_factory_requires_local_http_url(self) -> None:
        settings = Settings(
            _env_file=None,
            embedding_provider="local-http",
            embedding_http_url=None,
        )

        with pytest.raises(ValueError, match="EMBEDDING_HTTP_URL"):
            build_embedding_provider(settings)

    def test_factory_requires_openai_key_in_openai_mode(self) -> None:
        settings = Settings(
            _env_file=None,
            embedding_provider="openai",
            openai_api_key=None,
        )

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            build_embedding_provider(settings)

    def test_factory_rejects_unknown_provider(self) -> None:
        settings = Settings(_env_file=None, embedding_provider="unknown")

        with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
            build_embedding_provider(settings)


class TestOpenAIEmbeddingProvider:
    """Tests for the hosted OpenAI embedding adapter."""

    def test_embed_posts_to_embeddings_endpoint(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return FakeResponse()

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            dimension=3,
            client=FakeClient(),
        )

        vector = provider.embed("Ship MVP")

        assert vector == [0.1, 0.2, 0.3]
        assert captured["url"] == "https://api.openai.com/v1/embeddings"
        assert captured["headers"] == {"Authorization": "Bearer test-key"}
        assert captured["json"] == {
            "model": "text-embedding-3-small",
            "input": "Ship MVP",
            "encoding_format": "float",
            "dimensions": 3,
        }

    def test_embed_rejects_unexpected_dimension(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"data": [{"embedding": [0.1, 0.2]}]}

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                return FakeResponse()

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            dimension=3,
            client=FakeClient(),
        )

        with pytest.raises(ValueError, match="dimension"):
            provider.embed("Ship MVP")

    def test_embed_retries_transient_server_errors(self) -> None:
        calls = 0

        class FakeResponse:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code
                self.request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        "server unavailable",
                        request=self.request,
                        response=httpx.Response(self.status_code, request=self.request),
                    )

            def json(self) -> dict[str, object]:
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                nonlocal calls
                del url, headers, json
                calls += 1
                return FakeResponse(503 if calls == 1 else 200)

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            dimension=3,
            client=FakeClient(),
            retry_backoff_seconds=0,
        )

        assert provider.embed("Ship MVP") == [0.1, 0.2, 0.3]
        assert calls == 2

    def test_embed_honors_retry_after_for_rate_limits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0
        sleeps: list[float] = []

        class FakeResponse:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code
                self.request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
                self.headers = {"retry-after": "2"}

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        "rate limited",
                        request=self.request,
                        response=httpx.Response(
                            self.status_code,
                            headers=self.headers,
                            request=self.request,
                        ),
                    )

            def json(self) -> dict[str, object]:
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                nonlocal calls
                del url, headers, json
                calls += 1
                return FakeResponse(429 if calls == 1 else 200)

        monkeypatch.setattr("zaxy.embedding.time.sleep", sleeps.append)
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            dimension=3,
            client=FakeClient(),
            retry_backoff_seconds=0.1,
        )

        assert provider.embed("Ship MVP") == [0.1, 0.2, 0.3]
        assert sleeps == [2.0]

    def test_embed_uses_longer_default_rate_limit_backoff(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0
        sleeps: list[float] = []

        class FakeResponse:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code
                self.request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        "rate limited",
                        request=self.request,
                        response=httpx.Response(self.status_code, request=self.request),
                    )

            def json(self) -> dict[str, object]:
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                nonlocal calls
                del url, headers, json
                calls += 1
                return FakeResponse(429 if calls == 1 else 200)

        monkeypatch.setattr("zaxy.embedding.time.sleep", sleeps.append)
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            dimension=3,
            client=FakeClient(),
            rate_limit_backoff_seconds=7,
        )

        assert provider.embed("Ship MVP") == [0.1, 0.2, 0.3]
        assert sleeps == [7]


class TestLocalHTTPEmbeddingProvider:
    """Tests for OpenAI-compatible or simple local embedding endpoints."""

    def test_embed_posts_to_local_endpoint(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"embedding": [0.1, 0.2, 0.3]}

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return FakeResponse()

        provider = LocalHTTPEmbeddingProvider(
            url="http://localhost:8080/embed",
            model="bge-small",
            dimension=3,
            api_key="local-key",
            client=FakeClient(),
        )

        vector = provider.embed("Ship MVP")

        assert vector == [0.1, 0.2, 0.3]
        assert captured["url"] == "http://localhost:8080/embed"
        assert captured["headers"] == {"Authorization": "Bearer local-key"}
        assert captured["json"] == {"input": "Ship MVP", "model": "bge-small"}

    def test_embed_accepts_openai_compatible_response(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                del url, headers, json
                return FakeResponse()

        provider = LocalHTTPEmbeddingProvider(
            url="http://localhost:8080/v1/embeddings",
            model=None,
            dimension=3,
            client=FakeClient(),
        )

        assert provider.embed("Ship MVP") == [0.1, 0.2, 0.3]


class TestSentenceTransformersEmbeddingProvider:
    """Tests for the optional in-process local semantic embedding adapter."""

    def test_loader_uses_optional_dependency_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, str] = {}

        class FakeModel:
            def encode(self, text: str, *, normalize_embeddings: bool) -> list[float]:
                del text, normalize_embeddings
                return [0.1]

        def fake_factory(model_name: str) -> FakeModel:
            captured["model_name"] = model_name
            return FakeModel()

        def fake_import_module(name: str) -> types.SimpleNamespace:
            assert name == "sentence_transformers"
            return types.SimpleNamespace(SentenceTransformer=fake_factory)

        monkeypatch.setattr("zaxy.embedding.importlib.import_module", fake_import_module)

        model = _load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

        assert model.encode("query", normalize_embeddings=True) == [0.1]
        assert captured == {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}

    def test_loader_raises_actionable_error_when_dependency_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_import_module(name: str) -> types.SimpleNamespace:
            assert name == "sentence_transformers"
            raise ImportError("missing")

        monkeypatch.setattr("zaxy.embedding.importlib.import_module", fake_import_module)

        with pytest.raises(ValueError, match=r"zaxy-memory\[local-embeddings\]"):
            _load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

    def test_embed_uses_sentence_transformers_model(self) -> None:
        captured: dict[str, object] = {}

        class FakeVector:
            def tolist(self) -> list[float]:
                return [0.1, 0.2, 0.3]

        class FakeModel:
            def encode(
                self,
                text: str,
                *,
                normalize_embeddings: bool,
            ) -> FakeVector:
                captured["text"] = text
                captured["normalize_embeddings"] = normalize_embeddings
                return FakeVector()

        provider = SentenceTransformersEmbeddingProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=3,
            model=FakeModel(),
        )

        assert provider.embed("doctor specialist") == [0.1, 0.2, 0.3]
        assert captured == {
            "text": "doctor specialist",
            "normalize_embeddings": True,
        }

    def test_embed_rejects_unexpected_dimension(self) -> None:
        class FakeModel:
            def encode(
                self,
                text: str,
                *,
                normalize_embeddings: bool,
            ) -> list[float]:
                del text, normalize_embeddings
                return [0.1, 0.2]

        provider = SentenceTransformersEmbeddingProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=3,
            model=FakeModel(),
        )

        with pytest.raises(ValueError, match="dimension"):
            provider.embed("doctor specialist")
