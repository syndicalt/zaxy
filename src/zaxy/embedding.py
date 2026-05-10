"""Embedding utilities for graph ingestion and retrieval.

The default provider is deterministic and local. It gives Zaxy a production
baseline for vector search without introducing a network dependency in the
core write path.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import replace
from typing import Any, Protocol

import httpx

from zaxy.extract import ExtractedEntity, ExtractionResult


class EmbeddingProvider(Protocol):
    """Protocol for swappable embedding providers."""

    dimension: int

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for text."""


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
        payload = response.json()
        embedding = payload["data"][0]["embedding"]
        vector = [float(value) for value in embedding]
        if len(vector) != self.dimension:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )
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
    raise ValueError("EMBEDDING_PROVIDER must be 'hash' or 'openai'")


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


def _tokens(text: str) -> list[str]:
    """Tokenize for deterministic feature hashing."""
    tokens = text.casefold().split()
    return tokens or [""]
