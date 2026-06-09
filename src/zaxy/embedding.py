"""Embedding utilities for graph ingestion and retrieval.

The default provider is deterministic and local. It gives Zaxy a production
baseline for vector search without introducing a network dependency in the
core write path.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import math
import time
from dataclasses import replace
from typing import Any, Protocol, cast

import httpx

from zaxy.extract import ExtractedEntity, ExtractionResult


class EmbeddingProvider(Protocol):
    """Protocol for swappable embedding providers."""

    dimension: int

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for text."""


class SentenceTransformerModel(Protocol):
    """Minimal protocol for sentence-transformers compatible models."""

    def encode(self, text: str, *, normalize_embeddings: bool) -> Any:
        """Return a vector-like embedding for text."""


class SentenceTransformerFactory(Protocol):
    """Callable constructor exposed by the optional sentence-transformers package."""

    def __call__(self, model_name: str) -> SentenceTransformerModel:
        """Build a local sentence-transformers model."""


class HashEmbeddingProvider:
    """Deterministic feature-hashing embedding provider.

    This is intentionally simple and offline. It enables vector index plumbing,
    testability, and deterministic local behavior; hosted embedding providers
    can implement ``EmbeddingProvider`` later without changing graph code.
    """

    def __init__(self, dimension: int = 1536) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Embed text into a sparse, unit-normalized vector."""
        vector = [0.0] * self.dimension
        tokens = _tokens(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    async def embed_async(self, text: str) -> list[float]:
        """Embed text without blocking the event loop."""
        return await asyncio.to_thread(self.embed, text)


class OpenAIEmbeddingProvider:
    """Hosted OpenAI embeddings provider."""

    _RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        client: Any | None = None,
        max_retries: int = 6,
        retry_backoff_seconds: float = 0.5,
        rate_limit_backoff_seconds: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if rate_limit_backoff_seconds < 0:
            raise ValueError("rate_limit_backoff_seconds must be non-negative")
        self.dimension = dimension
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=30.0)
        self._max_retries = max_retries
        self._retry_backoff_seconds: float = retry_backoff_seconds
        self._rate_limit_backoff_seconds: float = rate_limit_backoff_seconds

    def embed(self, text: str) -> list[float]:
        """Embed text with OpenAI's embeddings API."""
        response = self._post_with_retries(text)
        return self._vector_from_response(response)

    async def embed_async(self, text: str) -> list[float]:
        """Embed text with OpenAI's API without blocking the event loop."""
        response = await self._post_with_retries_async(text)
        return self._vector_from_response(response)

    def _vector_from_response(self, response: Any) -> list[float]:
        payload = response.json()
        vector = [float(value) for value in payload["data"][0]["embedding"]]
        _validate_dimension(vector, self.dimension)
        return vector

    def _post_with_retries(self, text: str) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self.model,
                        "input": text,
                        "encoding_format": "float",
                        "dimensions": self.dimension,
                    },
                )
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                if not self._should_retry(exc) or attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_delay(exc, attempt))
        assert last_error is not None
        raise last_error

    async def _post_with_retries_async(self, text: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await _client_post(
                    self._client,
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self.model,
                        "input": text,
                        "encoding_format": "float",
                        "dimensions": self.dimension,
                    },
                )
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                if not self._should_retry(exc) or attempt >= self._max_retries:
                    raise
                await asyncio.sleep(self._retry_delay(exc, attempt))
        assert last_error is not None
        raise last_error

    def _should_retry(self, exc: httpx.HTTPStatusError | httpx.TransportError) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        return exc.response.status_code in self._RETRYABLE_STATUS_CODES

    def _retry_delay(
        self,
        exc: httpx.HTTPStatusError | httpx.TransportError,
        attempt: int,
    ) -> float:
        if isinstance(exc, httpx.HTTPStatusError):
            retry_after = exc.response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
            if exc.response.status_code == 429:
                return self._rate_limit_backoff_seconds * (attempt + 1)
        return float(self._retry_backoff_seconds * (2 ** attempt))


class LocalHTTPEmbeddingProvider:
    """Model-agnostic local HTTP embedding provider.

    The endpoint may return either ``{"embedding": [...]}`` or an
    OpenAI-compatible ``{"data": [{"embedding": [...]}]}`` payload.
    """

    def __init__(
        self,
        url: str,
        *,
        dimension: int = 1536,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not url:
            raise ValueError("EMBEDDING_HTTP_URL is required for local-http embeddings")
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dimension = dimension
        self.model = model
        self._url = url
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=30.0)

    def embed(self, text: str) -> list[float]:
        """Embed text with a local HTTP endpoint."""
        request: dict[str, Any] = {"input": text}
        if self.model:
            request["model"] = self.model
        headers = (
            {"Authorization": f"Bearer {self._api_key}"}
            if self._api_key
            else {}
        )
        response = self._client.post(self._url, headers=headers, json=request)
        response.raise_for_status()
        vector = _embedding_from_payload(response.json())
        _validate_dimension(vector, self.dimension)
        return vector

    async def embed_async(self, text: str) -> list[float]:
        """Embed text with a local HTTP endpoint without blocking the event loop."""
        request: dict[str, Any] = {"input": text}
        if self.model:
            request["model"] = self.model
        headers = (
            {"Authorization": f"Bearer {self._api_key}"}
            if self._api_key
            else {}
        )
        response = await _client_post(self._client, self._url, headers=headers, json=request)
        response.raise_for_status()
        vector = _embedding_from_payload(response.json())
        _validate_dimension(vector, self.dimension)
        return vector


class SentenceTransformersEmbeddingProvider:
    """In-process local semantic embedding provider.

    This provider is optional and dependency-gated. It gives local deployments a
    real semantic vector signal without routing benchmark or production queries
    through a hosted API.
    """

    def __init__(
        self,
        model_name: str,
        *,
        dimension: int,
        model: SentenceTransformerModel | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("EMBEDDING_SENTENCE_TRANSFORMER_MODEL is required")
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dimension = dimension
        self.model_name = model_name
        self._model = model or _load_sentence_transformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed text with a local sentence-transformers model."""
        raw_vector = self._model.encode(text, normalize_embeddings=True)
        if hasattr(raw_vector, "tolist"):
            raw_vector = raw_vector.tolist()
        vector = [float(value) for value in raw_vector]
        if len(vector) != self.dimension:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )
        return vector

    async def embed_async(self, text: str) -> list[float]:
        """Embed text without blocking the event loop."""
        return await asyncio.to_thread(self.embed, text)


def _load_sentence_transformer(model_name: str) -> SentenceTransformerModel:
    """Load a sentence-transformers model with an actionable dependency error."""
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:  # pragma: no cover - exercised through factory behavior
        raise ValueError(
            "sentence-transformers is required when "
            "EMBEDDING_PROVIDER=sentence-transformers; install "
            "zaxy-memory[local-embeddings]"
        ) from exc
    factory = getattr(module, "SentenceTransformer", None)
    if factory is None:
        raise ValueError(
            "sentence-transformers did not expose SentenceTransformer; reinstall "
            "zaxy-memory[local-embeddings]"
        )
    return cast(SentenceTransformerFactory, factory)(model_name)


async def _client_post(
    client: Any,
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, object],
) -> Any:
    """Post with either an async client or a sync-compatible fake/client."""
    post = client.post
    if inspect.iscoroutinefunction(post):
        return await post(url, headers=headers, json=json)
    response = await asyncio.to_thread(post, url, headers=headers, json=json)
    if inspect.isawaitable(response):
        return await response
    return response


def _validate_dimension(vector: list[float], dimension: int) -> None:
    if len(vector) != dimension:
        raise ValueError(
            f"embedding dimension mismatch: expected {dimension}, got {len(vector)}"
        )


def build_embedding_provider(settings: Any) -> EmbeddingProvider | None:
    """Build the configured embedding provider."""
    if not settings.embedding_enabled:
        return None

    provider = str(settings.embedding_provider).casefold()
    if provider == "hash":
        return HashEmbeddingProvider(dimension=settings.embedding_dimension)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            base_url=settings.openai_base_url,
        )
    if provider in {"local-http", "local_http", "http"}:
        if not settings.embedding_http_url:
            raise ValueError("EMBEDDING_HTTP_URL is required when EMBEDDING_PROVIDER=local-http")
        return LocalHTTPEmbeddingProvider(
            url=settings.embedding_http_url,
            model=settings.embedding_http_model,
            api_key=settings.embedding_http_api_key,
            dimension=settings.embedding_dimension,
        )
    if provider in {
        "sentence-transformers",
        "sentence_transformers",
        "sentence-transformer",
        "sentence_transformer",
        "local-model",
        "local_model",
    }:
        return SentenceTransformersEmbeddingProvider(
            model_name=settings.embedding_sentence_transformer_model,
            dimension=settings.embedding_dimension,
        )
    raise ValueError(
        "EMBEDDING_PROVIDER must be 'hash', 'openai', 'local-http', "
        "or 'sentence-transformers'"
    )


def entity_embedding_text(entity: ExtractedEntity) -> str:
    """Return stable text used to embed an extracted entity."""
    base = f"{entity.name} ({entity.entity_type})"
    if entity.summary:
        return f"{base} {entity.summary}"
    return base


def embed_extraction(
    result: ExtractionResult,
    provider: EmbeddingProvider,
) -> ExtractionResult:
    """Return an extraction result with embeddings filled for entities."""
    entities = [
        entity
        if entity.embedding is not None
        else replace(entity, embedding=provider.embed(entity_embedding_text(entity)))
        for entity in result.entities
    ]
    return ExtractionResult(
        entities=entities,
        edges=result.edges,
        source_event_seq=result.source_event_seq,
        source_event_hash=result.source_event_hash,
        source_event_prev_hash=result.source_event_prev_hash,
        source_event_type=result.source_event_type,
        source_thread=result.source_thread,
    )


async def embed_extraction_async(
    result: ExtractionResult,
    provider: Any,
) -> ExtractionResult:
    """Return an extraction result with embeddings filled without blocking the event loop."""
    entities = []
    for entity in result.entities:
        if entity.embedding is not None:
            entities.append(entity)
            continue
        entities.append(
            replace(
                entity,
                embedding=await provider.embed_async(entity_embedding_text(entity)),
            )
        )
    return ExtractionResult(
        entities=entities,
        edges=result.edges,
        source_event_seq=result.source_event_seq,
        source_event_hash=result.source_event_hash,
        source_event_prev_hash=result.source_event_prev_hash,
        source_event_type=result.source_event_type,
        source_thread=result.source_thread,
    )


def _tokens(text: str) -> list[str]:
    """Tokenize for deterministic feature hashing."""
    tokens = text.casefold().split()
    return tokens or [""]


def _embedding_from_payload(payload: Any) -> list[float]:
    if isinstance(payload, dict):
        direct = payload.get("embedding")
        if isinstance(direct, list):
            return [float(value) for value in direct]
        data = payload.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return [float(value) for value in first["embedding"]]
    raise ValueError("embedding response missing embedding vector")
