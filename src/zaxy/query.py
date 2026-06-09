"""Query router: hybrid retrieval with temporal filtering.

The router fuses results from multiple search strategies (exact, traversal,
keyword) and applies temporal filters, deduplication, and ranking before
returning a context window suitable for injection into an agent prompt.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

from zaxy.graph import GraphEntity, SearchResult
from zaxy.metrics import get_metrics
from zaxy.projection import ProjectionStore
from zaxy.retrieval_plan import source_lane_queries
from zaxy.security import validate_limit, validate_query, validate_session_id, vector_has_signal

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_USER_ID_RE = re.compile(r"\buser-\d{4}\b")
_DURABLE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z0-9]+)+\b")
_EXACT_CANDIDATE_RE = (
    re.compile(r"\bGoal\s+\d{4}\b"),
    re.compile(r"\btask-\d{4}\b"),
    re.compile(r"\buser-\d{4}:[A-Za-z0-9_.-]+\b"),
    _USER_ID_RE,
)

QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "auth": ("authentication", "authorization"),
    "bachelor": ("undergraduate", "undergrad", "graduated", "graduation"),
    "bachelors": ("undergraduate", "undergrad", "graduated", "graduation"),
    "decision": ("decided", "choice", "rationale"),
    "decisions": ("decided", "choice", "rationale"),
    "degree": ("undergraduate", "undergrad", "graduated", "graduation"),
    "doc": ("document", "documentation"),
    "docs": ("document", "documentation"),
    "bug": ("defect", "failure", "regression"),
    "bugs": ("defect", "failure", "regression"),
    "task": ("todo", "work", "action"),
    "tasks": ("todo", "work", "action"),
}


@dataclass(frozen=True)
class ScoringProfile:
    """Named retrieval scoring policy."""

    name: str
    fusion_weights: dict[str, float]
    expansion_weight: float = 0.92
    temporal_weight: float = 0.12
    mmr_lambda: float = 0.7
    traversal_bonus: float = 0.1


@dataclass(frozen=True)
class RetentionPolicy:
    """Non-destructive retrieval-time retention policy."""

    mode: str = "none"
    decay_half_life_days: int = 30
    expired_weight: float = 0.0
    purpose_decay_half_life_days: dict[str, int] = field(default_factory=dict)
    purpose_expired_weights: dict[str, float] = field(default_factory=dict)


SCORING_PROFILES: dict[str, ScoringProfile] = {
    "balanced": ScoringProfile(
        name="balanced",
        fusion_weights={
            "exact": 1.0,
            "vector": 0.95,
            "traversal": 0.9,
            "keyword": 0.8,
        },
    ),
    "precision": ScoringProfile(
        name="precision",
        fusion_weights={
            "exact": 1.0,
            "vector": 0.85,
            "traversal": 0.75,
            "keyword": 0.65,
        },
        expansion_weight=0.75,
        temporal_weight=0.08,
        mmr_lambda=0.82,
        traversal_bonus=0.04,
    ),
    "recall": ScoringProfile(
        name="recall",
        fusion_weights={
            "exact": 1.0,
            "vector": 1.0,
            "traversal": 0.95,
            "keyword": 0.85,
        },
        expansion_weight=0.98,
        temporal_weight=0.1,
        mmr_lambda=0.58,
        traversal_bonus=0.14,
    ),
    "temporal": ScoringProfile(
        name="temporal",
        fusion_weights={
            "exact": 1.0,
            "vector": 0.9,
            "traversal": 0.9,
            "keyword": 0.78,
        },
        expansion_weight=0.9,
        temporal_weight=0.22,
        mmr_lambda=0.68,
        traversal_bonus=0.1,
    ),
}


class Reranker(Protocol):
    """Optional second-stage ranker for fused graph candidates."""

    async def rerank(self, query: str, results: list[SearchResult], *, limit: int) -> list[SearchResult]:
        """Return reranked results, truncated to limit."""


@dataclass(frozen=True)
class ContextChunk:
    """A ranked piece of context ready for the agent prompt."""

    content: str
    source: str
    score: float
    valid_from: str | None
    valid_to: str | None
    citation: str | None = None
    score_explanation: dict[str, Any] | None = None
    entity_name: str | None = None
    entity_type: str | None = None
    metadata: dict[str, Any] | None = None


async def _client_post(
    client: Any,
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, object],
) -> Any:
    """Post without blocking the event loop for sync-compatible clients."""
    post = client.post
    if inspect.iscoroutinefunction(post):
        return await post(url, headers=headers, json=json)
    response = await asyncio.to_thread(post, url, headers=headers, json=json)
    if inspect.isawaitable(response):
        return await response
    return response


class LexicalReranker:
    """Deterministic local reranker based on query-token overlap."""

    name = "lexical"
    strategy = "lexical_overlap"

    def __init__(self, weight: float = 0.8) -> None:
        self.weight = weight

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        limit: int,
        entity_token_cache: dict[tuple[str, str, str], set[str]] | None = None,
    ) -> list[SearchResult]:
        query_tokens = _tokens(query)
        entity_token_cache = _entity_token_cache(results, entity_token_cache=entity_token_cache)
        reranked: list[SearchResult] = []
        for result in results:
            score = _lexical_rerank_score_for_result(
                query_tokens,
                query,
                result,
                entity_token_cache=entity_token_cache,
            )
            weighted_score = result.score + (self.weight * score)
            reranked.append(
                replace(
                    result,
                    score=weighted_score,
                    ranking_score=weighted_score,
                    reranker=self.name,
                    rerank_score=score,
                    rerank_strategy=self.strategy,
                )
        )
        return sorted(reranked, key=lambda item: item.ranking_score or item.score, reverse=True)[:limit]


class HTTPReranker:
    """HTTP reranker for local or self-hosted model endpoints."""

    name = "http"
    strategy = "cross_encoder"

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        weight: float = 0.35,
        client: Any | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("RERANKER_URL is required for HTTP reranking")
        self.endpoint = endpoint
        self.api_key = api_key
        self.weight = weight
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def rerank(self, query: str, results: list[SearchResult], *, limit: int) -> list[SearchResult]:
        response = await _client_post(
            self._client,
            self.endpoint,
            headers=_auth_headers(self.api_key),
            json={
                "query": query,
                "candidates": [_candidate_payload(index, result) for index, result in enumerate(results)],
            },
        )
        response.raise_for_status()
        scores = _extract_rerank_scores(response.json(), expected=len(results))
        return _apply_rerank_scores(
            results,
            scores,
            limit=limit,
            weight=self.weight,
            reranker=self.name,
            rerank_strategy=self.strategy,
        )


class LateInteractionHTTPReranker:
    """HTTP reranker for token-level late-interaction endpoints.

    The endpoint receives compact candidate text plus deterministic token lists
    so ColBERT-style or other multi-vector rankers can score token alignment
    without changing Zaxy's default local retrieval path.
    """

    name = "late-interaction-http"
    strategy = "late_interaction"

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        weight: float = 0.35,
        client: Any | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("RERANKER_URL is required for late-interaction HTTP reranking")
        self.endpoint = endpoint
        self.api_key = api_key
        self.weight = weight
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def rerank(self, query: str, results: list[SearchResult], *, limit: int) -> list[SearchResult]:
        response = await _client_post(
            self._client,
            self.endpoint,
            headers=_auth_headers(self.api_key),
            json={
                "query": query,
                "query_tokens": _token_list(query),
                "rerank_strategy": self.strategy,
                "candidates": [
                    _late_interaction_candidate_payload(index, result)
                    for index, result in enumerate(results)
                ],
            },
        )
        response.raise_for_status()
        scores = _extract_rerank_scores(response.json(), expected=len(results))
        return _apply_rerank_scores(
            results,
            scores,
            limit=limit,
            weight=self.weight,
            reranker=self.name,
            rerank_strategy=self.strategy,
        )


class OpenAICompatibleReranker:
    """OpenAI-compatible chat-completions reranker.

    The provider expects the model to return JSON like:
    ``[{"index": 1, "score": 0.95}, {"index": 0, "score": 0.2}]``.
    """

    name = "openai-compatible"
    strategy = "hosted_model"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        base_url: str = "https://api.openai.com/v1",
        weight: float = 0.35,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-compatible reranking")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.weight = weight
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def rerank(self, query: str, results: list[SearchResult], *, limit: int) -> list[SearchResult]:
        response = await _client_post(
            self._client,
            f"{self.base_url}/chat/completions",
            headers=_auth_headers(self.api_key),
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Rank candidates for retrieval relevance. "
                            "Return only JSON: [{\"index\": int, \"score\": float}] with scores from 0 to 1."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "query": query,
                                "candidates": [
                                    _candidate_payload(index, result)
                                    for index, result in enumerate(results)
                                ],
                            },
                            separators=(",", ":"),
                        ),
                    },
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        scores = _extract_rerank_scores(json.loads(content), expected=len(results))
        return _apply_rerank_scores(
            results,
            scores,
            limit=limit,
            weight=self.weight,
            reranker=self.name,
            rerank_strategy=self.strategy,
        )


class QueryRouter:
    """Routes natural-language queries to the graph store and fuses results.

    Args:
        store: Connected projection store instance.
        default_limit: Max results per sub-query.
        fusion_weights: Dict of {source: weight} for score normalization.
    """

    def __init__(
        self,
        store: ProjectionStore,
        default_limit: int = 10,
        session_id: str = "default",
        fusion_weights: dict[str, float] | None = None,
        scoring_profile: str | ScoringProfile = "balanced",
        reranker: Reranker | None = None,
        retention_policy: str | RetentionPolicy = "none",
        retention_decay_half_life_days: int = 30,
        retention_expired_weight: float = 0.0,
    ) -> None:
        self.store = store
        self.default_limit = default_limit
        self.session_id = validate_session_id(session_id)
        self.scoring_profile = _resolve_scoring_profile(scoring_profile, fusion_weights)
        self.fusion_weights = self.scoring_profile.fusion_weights
        self.temporal_weight = self.scoring_profile.temporal_weight
        self.reranker = reranker
        self.retention_policy = _resolve_retention_policy(
            retention_policy,
            decay_half_life_days=retention_decay_half_life_days,
            expired_weight=retention_expired_weight,
        )
        self._traversal_available_by_session: dict[str, bool] = {}

    async def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int | None = None,
        embedding: list[float] | None = None,
        session_id: str | None = None,
        scoring_profile: str | ScoringProfile | None = None,
    ) -> list[ContextChunk]:
        """Run a hybrid query and return ranked context chunks.

        Strategy:
        1. Exact match (if the query looks like an entity name).
        2. Vector similarity search (if embedding provided).
        3. Keyword/BM25 search.
        4. Graph traversal from top keyword hits.
        5. Fuse, deduplicate, sort by score, truncate to limit.
        """
        validate_query(query)
        lim = validate_limit(limit, default=self.default_limit)
        candidate_limit = _candidate_limit(lim)
        scope = validate_session_id(session_id or self.session_id)
        active_profile = (
            self.scoring_profile
            if scoring_profile is None
            else _resolve_scoring_profile(scoring_profile, None)
        )
        fusion_weights = active_profile.fusion_weights
        temporal_weight = active_profile.temporal_weight
        results: list[SearchResult] = []

        # 1. Exact match attempt against the full query and structured entity
        # names embedded in natural-language questions.
        warnings: list[str] = []
        exact_hits: list[SearchResult] = []
        for candidate in _exact_candidates(query):
            try:
                for ent in await self.store.search_exact(
                    candidate,
                    temporal_point=temporal_point,
                    session_id=scope,
                ):
                    exact_hits.append(
                        _apply_salience_score(
                            SearchResult(
                                entity=ent,
                                score=1.0 * fusion_weights["exact"],
                                source="exact",
                                raw_score=1.0,
                                source_weight=fusion_weights["exact"],
                                matched_query=candidate,
                                scoring_profile=active_profile.name,
                            )
                        )
                    )
            except Exception:
                get_metrics().record_degraded_operation("query", "exact_search_unavailable")
                warnings.append("exact search unavailable")
        results.extend(exact_hits)

        # 2. Vector search (if embedding provided)
        vector_hits: list[SearchResult] = []
        if embedding and vector_has_signal(embedding):
            try:
                vector_hits = await self.store.search_vector(
                    embedding,
                    limit=candidate_limit,
                    temporal_point=temporal_point,
                    session_id=scope,
                )
            except Exception:
                get_metrics().record_degraded_operation("query", "vector_search_unavailable")
                warnings.append("vector search unavailable")
                vector_hits = []
            for hit in vector_hits:
                hit = SearchResult(
                    entity=hit.entity,
                    score=hit.score * fusion_weights["vector"],
                    source="vector",
                    raw_score=hit.score,
                    source_weight=fusion_weights["vector"],
                    matched_query=query,
                    scoring_profile=active_profile.name,
                )
                hit = _apply_temporal_score(hit, temporal_point, temporal_weight)
                results.append(_apply_salience_score(hit))

        # 3. Keyword search
        keyword_hits: list[SearchResult] = []
        identifier_terms = _identifier_terms(query)
        for keyword_query in _expanded_queries(query):
            query_weight = 1.0 if keyword_query == query else active_profile.expansion_weight
            try:
                query_hits = await self.store.search_keyword(
                    keyword_query,
                    limit=candidate_limit,
                    temporal_point=temporal_point,
                    session_id=scope,
                )
            except Exception:
                get_metrics().record_degraded_operation("query", "keyword_search_unavailable")
                warnings.append("keyword search unavailable")
                continue
            for hit in query_hits:
                raw_score = _identifier_boosted_score(
                    min(hit.score, 1.0),
                    hit.entity,
                    identifier_terms,
                )
                hit = SearchResult(
                    entity=hit.entity,
                    score=raw_score * fusion_weights["keyword"] * query_weight,
                    source="keyword",
                    raw_score=hit.score,
                    source_weight=fusion_weights["keyword"],
                    matched_query=keyword_query,
                    query_weight=query_weight,
                    scoring_profile=active_profile.name,
                )
                hit = _apply_temporal_score(hit, temporal_point, temporal_weight)
                hit = _apply_salience_score(hit)
                keyword_hits.append(hit)
                results.append(hit)

        # 4. Traversal from exact anchors, or from top keyword + vector hits
        # when no durable exact anchor is available.
        seen = {(r.entity.name, r.entity.entity_type) for r in results}
        traversal_seeds = _traversal_seeds(
            exact_hits=exact_hits,
            results=results,
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
            identifier_terms=identifier_terms,
        )
        if traversal_seeds and (temporal_point is not None or await self._has_traversal_edges(scope)):
            for hit in traversal_seeds:
                try:
                    neighbors = await self.store.search_traversal(
                        hit.entity.name,
                        depth=2,
                        temporal_point=temporal_point,
                        session_id=scope,
                    )
                except Exception:
                    get_metrics().record_degraded_operation("query", "traversal_search_unavailable")
                    warnings.append("traversal search unavailable")
                    continue
                for neighbor in neighbors:
                    neighbor_key = (neighbor.name, neighbor.entity_type)
                    if neighbor_key in seen and hit.source != "exact":
                        continue
                    raw_score = _traversal_raw_score(query, hit, neighbor)
                    results.append(
                        _apply_salience_score(
                            SearchResult(
                                entity=neighbor,
                                score=raw_score * fusion_weights["traversal"],
                                source="traversal",
                                raw_score=raw_score,
                                source_weight=fusion_weights["traversal"],
                                scoring_profile=active_profile.name,
                            )
                        )
                    )
                    seen.add(neighbor_key)

        # 5. Deduplicate by (name, type), keep highest score
        best: dict[tuple[str, str], SearchResult] = {}
        for r in results:
            retained = _apply_retention_policy(r, self.retention_policy, temporal_point)
            if retained is None:
                continue
            r = retained
            key = (r.entity.name, r.entity.entity_type)
            if key not in best or r.score > best[key].score:
                best[key] = r

        # 6. Suppress fuzzy near-neighbors when a durable identifier has a direct hit.
        filtered = _suppress_identifier_fuzzy_distractors(
            list(best.values()),
            identifier_terms,
        )

        # 7. Sort with either provider reranking or MMR diversity and truncate
        ranked = await self._rank(
            query,
            [_with_warnings(r, warnings) for r in filtered],
            lim,
            scoring_profile=active_profile,
        )

        return [_to_chunk(r) for r in ranked]

    async def _has_traversal_edges(self, session_id: str) -> bool:
        """Return cached traversal availability, defaulting open for older stores."""
        cached = self._traversal_available_by_session.get(session_id)
        if cached is not None:
            return cached
        checker = getattr(self.store, "has_traversal_edges", None)
        if checker is None:
            self._traversal_available_by_session[session_id] = True
            return True
        try:
            available = bool(await checker(session_id=session_id))
        except Exception:
            get_metrics().record_degraded_operation("query", "traversal_capability_unavailable")
            available = True
        self._traversal_available_by_session[session_id] = available
        return available

    async def _rank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
        *,
        scoring_profile: ScoringProfile | None = None,
    ) -> list[SearchResult]:
        active_profile = scoring_profile or self.scoring_profile
        mmr_pool_limit = _mmr_pool_limit(limit, has_reranker=self.reranker is not None)
        mmr_inputs = _bounded_mmr_candidates(
            results,
            limit=mmr_pool_limit,
            traversal_bonus=active_profile.traversal_bonus,
        )
        entity_token_cache = _entity_token_cache(mmr_inputs)
        candidates = _mmr_rank(
            mmr_inputs,
            limit=len(mmr_inputs) if self.reranker is not None else limit,
            lambda_score=active_profile.mmr_lambda,
            traversal_bonus=active_profile.traversal_bonus,
            entity_token_cache=entity_token_cache,
        )
        if self.reranker is None:
            return candidates[:limit]
        try:
            if isinstance(self.reranker, LexicalReranker):
                reranked = await self.reranker.rerank(
                    query,
                    candidates,
                    limit=limit,
                    entity_token_cache=entity_token_cache,
                )
            else:
                reranked = await self.reranker.rerank(query, candidates, limit=limit)
        except Exception:
            get_metrics().record_degraded_operation("query", "reranker_unavailable")
            return [_with_warnings(candidate, ["reranker unavailable"]) for candidate in candidates[:limit]]
        return reranked[:limit]


def _mmr_pool_limit(limit: int, *, has_reranker: bool) -> int:
    """Return the bounded candidate pool considered by MMR."""
    if has_reranker:
        return max(limit * 4, 20)
    return max(limit * 3, 16)


def _bounded_mmr_candidates(
    results: list[SearchResult],
    *,
    limit: int,
    traversal_bonus: float,
) -> list[SearchResult]:
    """Keep the highest-value candidates before the quadratic MMR stage."""
    if len(results) <= limit:
        return results
    source_bonus = {"traversal": traversal_bonus}
    return sorted(
        results,
        key=lambda result: result.score + source_bonus.get(result.source, 0.0),
        reverse=True,
    )[:limit]


def build_reranker(settings: Any) -> Reranker | None:
    """Build the configured reranker provider."""
    provider = str(getattr(settings, "reranker_provider", "none")).casefold()
    if provider in {"none", "disabled", "off", ""}:
        return None
    if provider == "lexical":
        return LexicalReranker()
    if provider == "http":
        endpoint = getattr(settings, "reranker_url", None)
        if not endpoint:
            raise ValueError("RERANKER_URL is required when RERANKER_PROVIDER=http")
        return HTTPReranker(
            endpoint=endpoint,
            api_key=getattr(settings, "reranker_api_key", None),
        )
    if provider in {"late-interaction-http", "late_interaction_http", "late-http", "late_http"}:
        endpoint = getattr(settings, "reranker_url", None)
        if not endpoint:
            raise ValueError("RERANKER_URL is required when RERANKER_PROVIDER=late-interaction-http")
        return LateInteractionHTTPReranker(
            endpoint=endpoint,
            api_key=getattr(settings, "reranker_api_key", None),
        )
    if provider == "openai":
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when RERANKER_PROVIDER=openai")
        return OpenAICompatibleReranker(
            api_key=api_key,
            model=getattr(settings, "openai_rerank_model", "gpt-5-mini"),
            base_url=getattr(settings, "openai_base_url", "https://api.openai.com/v1"),
        )
    raise ValueError("RERANKER_PROVIDER must be 'none', 'lexical', 'http', 'late-interaction-http', or 'openai'")


def build_retention_policy(settings: Any) -> RetentionPolicy:
    """Build the configured non-destructive retrieval retention policy."""
    return _resolve_retention_policy(
        str(getattr(settings, "retention_policy", "none")),
        decay_half_life_days=int(getattr(settings, "retention_decay_half_life_days", 30)),
        expired_weight=float(getattr(settings, "retention_expired_weight", 0.0)),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _candidate_limit(prompt_limit: int) -> int:
    """Return the internal retrieval budget used before final prompt truncation."""
    return min(50, max(prompt_limit, prompt_limit * 4))


def _to_chunk(result: SearchResult) -> ContextChunk:
    """Convert a SearchResult to a ContextChunk."""
    ent = result.entity
    # Build a concise natural-language summary of the entity
    content = f"{ent.name} ({ent.entity_type})"
    if ent.properties:
        safe_properties = _prompt_visible_properties(ent.properties)
        props = ", ".join(f"{k}={v}" for k, v in safe_properties)
        if props:
            content += f" — {props}"
    summary = ent.properties.get("summary")
    if isinstance(summary, str) and summary and summary not in content:
        content += f" — {summary}"

    return ContextChunk(
        content=content,
        source=result.source,
        score=round(result.score, 4),
        valid_from=ent.valid_from,
        valid_to=ent.valid_to,
        citation=_citation(ent),
        score_explanation=_score_explanation(result),
        entity_name=ent.name,
        entity_type=ent.entity_type,
        metadata=_diagnostic_metadata(ent),
    )


def _prompt_visible_properties(properties: dict[str, Any], *, limit: int = 3) -> list[tuple[str, Any]]:
    """Return bounded prompt properties while preserving source identity."""
    excluded = {"embedding", "created_at", "updated_at"}
    safe_items = [
        (key, value)
        for key, value in properties.items()
        if key not in excluded and not key.startswith("_")
    ]
    priority_prefixes = ("longmemeval_session_id",)
    priority_key_order = (
        "summary",
        "taskId",
        "task_id",
        "userId",
        "user_id",
        "goalTitle",
        "goal_title",
        "source_path",
        "source_start_line",
        "source_end_line",
        "source_event_seq",
        "source_event_hash",
        "source_thread",
        "transcript_source",
    )
    priority_keys = set(priority_key_order)
    priority: list[tuple[str, Any]] = []
    remaining: list[tuple[str, Any]] = []
    safe_by_key = dict(safe_items)
    for key in priority_key_order:
        if key in safe_by_key:
            priority.append((key, safe_by_key[key]))
    for key, value in safe_items:
        if key in priority_keys:
            continue
        if key.startswith(priority_prefixes):
            priority.append((key, value))
        else:
            remaining.append((key, value))
    selected: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key, value in [*priority, *remaining]:
        if key in seen:
            continue
        selected.append((key, value))
        seen.add(key)
        if len(selected) >= max(limit, len(priority)):
            break
    return selected


def _diagnostic_metadata(entity: GraphEntity) -> dict[str, Any] | None:
    """Return bounded structured properties needed by downstream diagnostics."""
    if entity.entity_type not in {"event", "skill_version", "skill_outcome"}:
        return None
    allowed_keys = {
        "applicability",
        "authority",
        "authority_scope",
        "contradiction_reason",
        "coordination_status",
        "evidence",
        "failure_modes",
        "feedback",
        "finding_status",
        "procedure",
        "promoted",
        "rollback",
        "skill_id",
        "stale",
        "status",
        "success_score",
        "summary",
        "superseded_by",
        "task",
        "version",
    }
    metadata = {
        key: value
        for key, value in entity.properties.items()
        if key in allowed_keys
    }
    return metadata or None


def _exact_candidates(query: str) -> list[str]:
    """Return exact entity candidates named inside a natural-language query."""
    candidates = [query]
    for pattern in _EXACT_CANDIDATE_RE:
        candidates.extend(match.group(0) for match in pattern.finditer(query))
    candidates.extend(_identifier_terms(query))
    candidates.extend(_structured_preference_candidates(query))

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _structured_preference_candidates(query: str) -> list[str]:
    """Infer deterministic preference entity names from common memory questions."""
    user_ids = [match.group(0) for match in _USER_ID_RE.finditer(query)]
    if not user_ids:
        return []
    lowered = query.casefold()
    keys = []
    for key in ("theme", "language", "timezone", "locale"):
        if key in lowered:
            keys.append(key)
    if "preference" not in lowered and not keys:
        return []
    if not keys:
        keys.append("preference")
    return [f"{user_id}:{key}" for user_id in user_ids for key in keys]


def _identifier_terms(query: str) -> tuple[str, ...]:
    """Extract durable identifiers that should dominate fuzzy similarity."""
    return tuple(
        dict.fromkeys(
            match.group(0)
            for match in _DURABLE_IDENTIFIER_RE.finditer(query)
            if _looks_like_durable_identifier(match.group(0))
        )
    )


def _looks_like_durable_identifier(value: str) -> bool:
    """Return true for code-like hyphenated IDs, not natural adjective compounds."""
    parts = value.split("-")
    if len(parts) < 2:
        return False
    for part in parts:
        for char in part:
            if char.isdigit():
                return True
    durable_markers = {"id", "uuid", "code", "ticket", "task", "goal"}
    for part in parts:  # noqa: SIM110 - avoid generator allocation on query hot paths.
        if part.casefold() in durable_markers:
            return True
    return False


def _identifier_boosted_score(
    base_score: float,
    entity: GraphEntity,
    identifiers: tuple[str, ...],
) -> float:
    if not identifiers:
        return base_score
    searchable = _entity_identifier_text(entity).casefold()
    if _identifier_in_text(searchable, identifiers):
        return max(base_score, 1.35)
    return base_score


def _suppress_identifier_fuzzy_distractors(
    results: list[SearchResult],
    identifiers: tuple[str, ...],
) -> list[SearchResult]:
    """Remove fuzzy candidates that miss durable query identifiers.

    Identifier-bearing questions are usually asking for exact provenance or
    source recovery. If at least one candidate contains the identifier, fuzzy
    keyword/vector neighbors that do not contain it are likely distractors.
    Exact and traversal results are kept because they come from explicit graph
    anchors rather than semantic similarity alone.
    """
    if not identifiers:
        return results
    has_identifier_match = False
    retained: list[SearchResult] = []
    for result in results:
        matches_identifier = _entity_matches_identifier(result.entity, identifiers)
        if matches_identifier:
            has_identifier_match = True
        if result.source == "exact" or result.source == "traversal" or matches_identifier:
            retained.append(result)
    if not has_identifier_match:
        return results
    return retained


def _entity_identifier_text(entity: GraphEntity) -> str:
    values = [entity.name, entity.entity_type]
    for key in ("summary", "source_path", "transcript_source"):
        value = entity.properties.get(key)
        if isinstance(value, str):
            values.append(value)
    return " ".join(values)


def _expanded_queries(query: str) -> list[str]:
    """Return the original query plus deterministic synonym expansions."""
    tokens = _TOKEN_RE.findall(query.lower())
    expanded_terms: list[str] = []
    for token in tokens:
        expanded_terms.extend(QUERY_EXPANSIONS.get(token, ()))
    expanded_terms.extend(_phrase_expansions(tokens))
    expanded_terms = _unique(expanded_terms)
    identifier_terms = list(_identifier_terms(query))
    planned_queries = [
        planned_query
        for planned_query in source_lane_queries(query, [])
        if planned_query != query
    ]
    if not expanded_terms:
        return _unique([query, *planned_queries, *identifier_terms])
    return _unique([query, f"{query} {' '.join(expanded_terms)}", *planned_queries, *identifier_terms])


def _phrase_expansions(tokens: list[str]) -> tuple[str, ...]:
    token_set = set(tokens)
    expansions: list[str] = []
    if {"computer", "science"} <= token_set:
        expansions.append("cs")
    return tuple(expansions)


def _candidate_payload(index: int, result: SearchResult) -> dict[str, object]:
    """Return a compact candidate payload for model rerankers."""
    return {
        "index": index,
        "content": _candidate_text(result.entity),
        "source": result.source,
        "score": round(result.score, 6),
    }


def _late_interaction_candidate_payload(index: int, result: SearchResult) -> dict[str, object]:
    """Return tokenized candidate payload for late-interaction rerankers."""
    payload = _candidate_payload(index, result)
    payload["tokens"] = _token_list(str(payload["content"]))
    return payload


def _candidate_text(entity: GraphEntity) -> str:
    parts = [f"{entity.name} ({entity.entity_type})"]
    summary = entity.properties.get("summary")
    if isinstance(summary, str) and summary:
        parts.append(summary)
    return " ".join(parts)


def _traversal_raw_score(query: str, seed: SearchResult, neighbor: GraphEntity) -> float:
    """Score graph neighbors by the relation path that supports them."""
    relations = {
        str(relation).casefold()
        for relation in neighbor.properties.get("_path_relation_types", [])
    }
    query_tokens = set(_tokens(query))
    score = 1.45 if seed.source == "exact" else 0.75

    asks_completion = bool(query_tokens & {"complete", "completed", "completion", "finish", "finished"})
    asks_task = "task" in query_tokens or "tasks" in query_tokens

    if asks_completion and "completed_task" in relations:
        score += 0.85
    if asks_completion and "proposed_task" in relations and "completed_task" not in relations:
        score -= 0.45
    if asks_task and "has_task" in relations:
        score += 0.25
    if neighbor.entity_type == "actor" and "completed_task" in relations:
        score += 0.35
    if neighbor.entity_type == "task" and "has_task" in relations:
        score += 0.15

    path_length = neighbor.properties.get("_path_length")
    if isinstance(path_length, int) and path_length > 1:
        score -= min(0.25, 0.05 * (path_length - 1))

    score *= _inferred_edge_trust_metadata(neighbor.properties)["multiplier"]

    return max(0.1, score)


def _inferred_edge_trust_metadata(properties: dict[str, Any]) -> dict[str, float | int]:
    """Return traversal trust metadata for paths that contain inferred edges."""
    inferred_edge_count = _positive_int(properties.get("_path_inferred_edge_count"))
    if inferred_edge_count == 0:
        return {
            "count": 0,
            "trust": 1.0,
            "multiplier": 1.0,
            "confidence": 1.0,
            "method_coverage": 1.0,
            "source_coverage": 1.0,
            "evidence_coverage": 1.0,
        }
    confidences = [
        _bounded_float(value, default=0.0)
        for value in _list_property(properties.get("_path_inferred_confidences"))
    ]
    confidence_total = 0.0
    confidence_count = 0
    for confidence in confidences:
        if confidence_count >= inferred_edge_count:
            break
        confidence_total += confidence
        confidence_count += 1
    average_confidence = confidence_total / inferred_edge_count if confidence_count else 0.0
    methods = [
        str(method).strip().casefold()
        for method in _list_property(properties.get("_path_inference_methods"))
    ]
    method_count = 0
    for method in methods:
        if method and method != "unknown":
            method_count += 1
    source_count = _positive_int(properties.get("_path_inferred_source_event_count"))
    evidenced_edge_count = _positive_int(properties.get("_path_inferred_evidenced_edge_count"))
    if evidenced_edge_count == 0 and _positive_int(properties.get("_path_inferred_evidence_count")) > 0:
        evidenced_edge_count = min(inferred_edge_count, _positive_int(properties.get("_path_inferred_evidence_count")))

    method_coverage = min(1.0, method_count / inferred_edge_count)
    source_coverage = min(1.0, source_count / inferred_edge_count)
    evidence_coverage = min(1.0, evidenced_edge_count / inferred_edge_count)
    provenance_coverage = (method_coverage + source_coverage + evidence_coverage) / 3.0
    trust = average_confidence * provenance_coverage
    multiplier = 0.65 + (0.5 * trust)
    return {
        "count": inferred_edge_count,
        "trust": trust,
        "multiplier": multiplier,
        "confidence": average_confidence,
        "method_coverage": method_coverage,
        "source_coverage": source_coverage,
        "evidence_coverage": evidence_coverage,
    }


def _traversal_seeds(
    *,
    exact_hits: list[SearchResult],
    results: list[SearchResult],
    vector_hits: list[SearchResult],
    keyword_hits: list[SearchResult],
    identifier_terms: tuple[str, ...],
) -> list[SearchResult]:
    """Choose bounded graph traversal anchors for the query."""
    if exact_hits:
        return exact_hits
    if identifier_terms:
        focused = [
            result
            for result in results
            if _entity_matches_identifier(result.entity, identifier_terms)
        ]
        if focused:
            return _unique_search_results(focused)[:3]
    return _unique_search_results([*results[:3], *(vector_hits + keyword_hits)[:3]])


def _entity_matches_identifier(entity: GraphEntity, identifiers: tuple[str, ...]) -> bool:
    searchable = _entity_identifier_text(entity).casefold()
    return _identifier_in_text(searchable, identifiers)


def _identifier_in_text(searchable: str, identifiers: tuple[str, ...]) -> bool:
    for identifier in identifiers:  # noqa: SIM110 - avoid generator allocation on query hot paths.
        if identifier.casefold() in searchable:
            return True
    return False


def _unique_search_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[tuple[str, str]] = set()
    unique: list[SearchResult] = []
    for result in results:
        key = (result.entity.name, result.entity.entity_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _extract_rerank_scores(payload: Any, *, expected: int) -> list[float]:
    """Extract per-candidate scores from common reranker response shapes."""
    scores: list[float | None] = [None] * expected
    if isinstance(payload, dict) and isinstance(payload.get("scores"), list):
        raw_scores = payload["scores"]
        if len(raw_scores) != expected:
            raise ValueError(f"reranker returned {len(raw_scores)} scores for {expected} candidates")
        return [float(score) for score in raw_scores]

    records = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("reranker response must be a score list or result records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("reranker result records must be objects")
        index = int(record["index"])
        if index < 0 or index >= expected:
            raise ValueError(f"reranker returned out-of-range candidate index: {index}")
        scores[index] = float(record["score"])
    return [0.0 if score is None else score for score in scores]


def _apply_rerank_scores(
    results: list[SearchResult],
    scores: list[float],
    *,
    limit: int,
    weight: float,
    reranker: str,
    rerank_strategy: str,
) -> list[SearchResult]:
    reranked = [
        replace(
            result,
            score=result.score + (weight * scores[index]),
            ranking_score=result.score + (weight * scores[index]),
            reranker=reranker,
            rerank_score=scores[index],
            rerank_strategy=rerank_strategy,
        )
        for index, result in enumerate(results)
    ]
    return sorted(reranked, key=lambda item: item.ranking_score or item.score, reverse=True)[:limit]


def _resolve_scoring_profile(
    scoring_profile: str | ScoringProfile,
    fusion_weights: dict[str, float] | None,
) -> ScoringProfile:
    """Resolve a named or explicit scoring profile."""
    if isinstance(scoring_profile, ScoringProfile):
        profile = scoring_profile
    else:
        try:
            profile = SCORING_PROFILES[scoring_profile]
        except KeyError as exc:
            options = ", ".join(sorted(SCORING_PROFILES))
            raise ValueError(f"Unknown scoring profile: {scoring_profile}. Expected one of: {options}") from exc

    if fusion_weights is None:
        return profile
    return replace(profile, fusion_weights={**profile.fusion_weights, **fusion_weights})


def _resolve_retention_policy(
    retention_policy: str | RetentionPolicy,
    *,
    decay_half_life_days: int,
    expired_weight: float,
) -> RetentionPolicy:
    if isinstance(retention_policy, RetentionPolicy):
        return retention_policy
    mode = retention_policy.casefold()
    if mode in {"", "off", "disabled"}:
        mode = "none"
    if mode not in {"none", "filter_expired", "decay"}:
        raise ValueError("RETENTION_POLICY must be 'none', 'filter_expired', or 'decay'")
    if decay_half_life_days < 1:
        raise ValueError("RETENTION_DECAY_HALF_LIFE_DAYS must be positive")
    if not 0.0 <= expired_weight <= 1.0:
        raise ValueError("RETENTION_EXPIRED_WEIGHT must be between 0 and 1")
    return RetentionPolicy(
        mode=mode,
        decay_half_life_days=decay_half_life_days,
        expired_weight=expired_weight,
        purpose_decay_half_life_days={
            "coordinate": max(decay_half_life_days, 180),
            "security": max(decay_half_life_days, 180),
            "review": max(decay_half_life_days, 90),
            "release": max(decay_half_life_days, 120),
            "support": max(decay_half_life_days, 90),
            "product": max(decay_half_life_days, 120),
            "sales": max(decay_half_life_days, 120),
            "legal": max(decay_half_life_days, 365),
            "executive": max(decay_half_life_days, 180),
            "coding": decay_half_life_days,
            "research": decay_half_life_days,
        },
        purpose_expired_weights={
            "coordinate": max(expired_weight, 0.15),
            "security": max(expired_weight, 0.1),
            "review": max(expired_weight, 0.05),
            "release": max(expired_weight, 0.1),
            "support": max(expired_weight, 0.05),
            "product": max(expired_weight, 0.05),
            "sales": max(expired_weight, 0.08),
            "legal": max(expired_weight, 0.2),
            "executive": max(expired_weight, 0.1),
        },
    )


def _mmr_rank(
    results: list[SearchResult],
    limit: int,
    lambda_score: float = 0.7,
    traversal_bonus: float = 0.1,
    entity_token_cache: dict[tuple[str, str, str], set[str]] | None = None,
) -> list[SearchResult]:
    """Rank by weighted relevance while penalizing near-duplicate context."""
    candidates = sorted(results, key=lambda x: x.score, reverse=True)
    selected: list[SearchResult] = []
    source_bonus = {"traversal": traversal_bonus}
    token_cache = _entity_token_cache(results, entity_token_cache=entity_token_cache)
    max_similarity_by_candidate = [0.0 for _ in candidates]
    latest_selected: SearchResult | None = None

    while candidates and len(selected) < limit:
        if not selected:
            first = candidates.pop(0)
            max_similarity_by_candidate.pop(0)
            selected.append(replace(first, ranking_score=first.score))
            latest_selected = first
            continue

        if latest_selected is not None:
            for index, candidate in enumerate(candidates):
                max_similarity_by_candidate[index] = max(
                    max_similarity_by_candidate[index],
                    _entity_similarity(candidate.entity, latest_selected.entity, token_cache),
                )

        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(candidates):
            similarity = max_similarity_by_candidate[index]
            ranking_score = (
                lambda_score * candidate.score
                - (1.0 - lambda_score) * similarity
                + source_bonus.get(candidate.source, 0.0)
            )
            if ranking_score > best_score:
                best_index = index
                best_score = ranking_score

        best = candidates.pop(best_index)
        max_similarity_by_candidate.pop(best_index)
        selected.append(replace(best, ranking_score=best_score))
        latest_selected = best

    return selected


def _entity_token_cache(
    results: list[SearchResult],
    *,
    entity_token_cache: dict[tuple[str, str, str], set[str]] | None = None,
) -> dict[tuple[str, str, str], set[str]]:
    """Return a token cache for the candidate entities, preserving caller state."""
    token_cache = entity_token_cache if entity_token_cache is not None else {}
    for result in results:
        key = _entity_token_cache_key(result.entity)
        if key not in token_cache:
            token_cache[key] = _entity_tokens(result.entity)
    return token_cache


def _entity_similarity(
    left: GraphEntity,
    right: GraphEntity,
    token_cache: dict[tuple[str, str, str], set[str]] | None = None,
) -> float:
    """Return token Jaccard similarity between two entities."""
    if token_cache is None:
        left_tokens = _entity_tokens(left)
        right_tokens = _entity_tokens(right)
    else:
        left_tokens = token_cache[_entity_token_cache_key(left)]
        right_tokens = token_cache[_entity_token_cache_key(right)]
    if not left_tokens or not right_tokens:
        return 0.0
    smaller, larger = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    intersection_count = 0
    for token in smaller:
        if token in larger:
            intersection_count += 1
    union_count = len(left_tokens) + len(right_tokens) - intersection_count
    return intersection_count / union_count if union_count else 0.0


def _entity_tokens(entity: GraphEntity) -> set[str]:
    text_parts = [entity.name, entity.entity_type]
    for key, value in entity.properties.items():
        if key in {"embedding", "created_at", "updated_at"}:
            continue
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, int | float | bool):
            text_parts.append(str(value))
    return {
        token
        for token in _TOKEN_RE.findall(" ".join(text_parts).lower())
        if len(token) > 1
    }


def _entity_token_cache_key(entity: GraphEntity) -> tuple[str, str, str]:
    return (entity.name, entity.entity_type, entity.valid_from)


def _lexical_rerank_score(
    query_tokens: set[str],
    entity: GraphEntity,
    *,
    entity_token_cache: dict[tuple[str, str, str], set[str]] | None = None,
) -> float:
    """Score an entity for second-stage lexical relevance."""
    if not query_tokens:
        return 0.0
    entity_tokens = (
        entity_token_cache[_entity_token_cache_key(entity)]
        if entity_token_cache is not None
        else _entity_tokens(entity)
    )
    overlap = len(query_tokens & entity_tokens) / len(query_tokens)
    phrase_bonus = 0.0
    lower_text = " ".join([entity.name, str(entity.properties.get("summary", ""))]).lower()
    for token in query_tokens:
        if token in lower_text:
            phrase_bonus += 0.03
    return min(overlap + phrase_bonus, 1.0)


def _lexical_rerank_score_for_result(
    base_tokens: set[str],
    query: str,
    result: SearchResult,
    *,
    entity_token_cache: dict[tuple[str, str, str], set[str]] | None = None,
) -> float:
    """Score a result against the user query and its matched expansion."""
    score = _lexical_rerank_score(
        base_tokens,
        result.entity,
        entity_token_cache=entity_token_cache,
    )
    matched_query = result.matched_query
    if not matched_query or matched_query == query:
        return score
    return max(
        score,
        _lexical_rerank_score(
            _tokens(matched_query),
            result.entity,
            entity_token_cache=entity_token_cache,
        ),
    )


def _tokens(value: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(value.lower()) if len(token) > 1}


def _token_list(value: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(value.lower()) if len(token) > 1]


def _apply_temporal_score(
    result: SearchResult,
    temporal_point: str | None,
    temporal_weight: float,
) -> SearchResult:
    """Apply a small, explainable as-of-time freshness adjustment."""
    temporal_score = _temporal_proximity(result.entity.valid_from, temporal_point)
    if temporal_score is None:
        return result
    multiplier = 1.0 + temporal_weight * (temporal_score - 0.5)
    return replace(
        result,
        score=result.score * multiplier,
        temporal_score=temporal_score,
        temporal_weight=temporal_weight,
    )


def _apply_salience_score(result: SearchResult) -> SearchResult:
    """Boost explicitly high-salience memory artifacts without changing provenance."""
    multiplier = _retrieval_salience(result.entity)
    if multiplier == 1.0:
        return result
    return replace(result, score=result.score * multiplier)


def _retrieval_salience(entity: GraphEntity) -> float:
    value = entity.properties.get("retrieval_salience")
    if value is None:
        return 1.0
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return 1.0
    if parsed <= 0.0:
        return 1.0
    return min(parsed, 10.0)


def _temporal_proximity(valid_from: str | None, temporal_point: str | None) -> float | None:
    """Score how close an assertion is to the requested as-of point."""
    if not valid_from or not temporal_point:
        return None
    try:
        start = _parse_iso_datetime(valid_from)
        point = _parse_iso_datetime(temporal_point)
    except ValueError:
        return None
    age_seconds = max((point - start).total_seconds(), 0.0)
    age_days = age_seconds / 86_400
    return 1.0 / (1.0 + (age_days / 365.0))


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _merge_warnings(existing: tuple[str, ...], warnings: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    merged = list(existing)
    for warning in warnings:
        if warning not in merged:
            merged.append(warning)
    return tuple(merged)


def _with_warnings(result: SearchResult, warnings: list[str] | tuple[str, ...]) -> SearchResult:
    if not warnings:
        return result
    return replace(result, warnings=_merge_warnings(result.warnings, warnings))


def _apply_retention_policy(
    result: SearchResult,
    policy: RetentionPolicy,
    temporal_point: str | None,
) -> SearchResult | None:
    if policy.mode == "none":
        return result
    now = _retention_now(temporal_point)
    expires_at = _parse_optional_datetime(result.entity.properties.get("expires_at"))
    if expires_at is not None and expires_at <= now:
        if policy.mode == "filter_expired":
            return None
        return _retention_replace(
            result,
            multiplier=_purpose_expired_weight(result, policy),
            policy=policy,
            expired=True,
        )
    if policy.mode != "decay":
        return _retention_replace(result, multiplier=1.0, policy=policy, expired=False)

    reference = (
        _parse_optional_datetime(result.entity.properties.get("last_reinforced_at"))
        or _parse_optional_datetime(result.entity.valid_from)
    )
    if reference is None:
        return _retention_replace(result, multiplier=1.0, policy=policy, expired=False)
    age_days = max(0.0, (now - reference).total_seconds() / 86400.0)
    half_life_days = _purpose_decay_half_life_days(result, policy)
    multiplier = math.pow(0.5, age_days / half_life_days)
    importance = _bounded_float(result.entity.properties.get("importance"), default=1.0)
    reinforcement_count = _bounded_float(
        result.entity.properties.get("reinforcement_count"),
        default=0.0,
    )
    reinforcement_boost = min(0.25, reinforcement_count * 0.03)
    multiplier = min(1.0, multiplier * importance + reinforcement_boost)
    return _retention_replace(
        result,
        multiplier=multiplier,
        policy=policy,
        expired=False,
        half_life_days=half_life_days,
    )


def _retention_replace(
    result: SearchResult,
    *,
    multiplier: float,
    policy: RetentionPolicy,
    expired: bool,
    half_life_days: int | None = None,
) -> SearchResult:
    metadata = dict(result.entity.properties)
    metadata["_retention_policy"] = policy.mode
    metadata["_retention_decay_multiplier"] = multiplier
    metadata["_retention_expired"] = expired
    if half_life_days is not None:
        metadata["_retention_half_life_days"] = half_life_days
    purpose_profile = _purpose_profile(result)
    if purpose_profile is not None:
        metadata["_retention_purpose_profile"] = purpose_profile
    return replace(
        result,
        entity=replace(result.entity, properties=metadata),
        score=result.score * multiplier,
        ranking_score=None if result.ranking_score is None else result.ranking_score * multiplier,
    )


def _purpose_profile(result: SearchResult) -> str | None:
    value = result.entity.properties.get("purpose_profile")
    if isinstance(value, str) and value.strip():
        return value.strip().casefold().replace(" ", "-")
    return None


def _purpose_decay_half_life_days(result: SearchResult, policy: RetentionPolicy) -> int:
    profile = _purpose_profile(result)
    if profile is None:
        return policy.decay_half_life_days
    return policy.purpose_decay_half_life_days.get(profile, policy.decay_half_life_days)


def _purpose_expired_weight(result: SearchResult, policy: RetentionPolicy) -> float:
    profile = _purpose_profile(result)
    if profile is None:
        return policy.expired_weight
    return policy.purpose_expired_weights.get(profile, policy.expired_weight)


def _retention_now(temporal_point: str | None) -> datetime:
    if temporal_point:
        return _parse_iso_datetime(temporal_point)
    return datetime.now(UTC)


def _parse_optional_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return _parse_iso_datetime(str(value))
    except ValueError:
        return None


def _bounded_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _list_property(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _score_explanation(result: SearchResult) -> dict[str, Any]:
    """Return compact score details for retrieval debugging."""
    inferred_trust = _inferred_edge_trust_metadata(result.entity.properties)
    inferred_relation_types = _inferred_relation_types(result.entity.properties)
    inference_methods = _inference_methods(result.entity.properties)
    path_relation_types = _unique_text(_list_property(result.entity.properties.get("_path_relation_types")))
    return {
        "source": result.source,
        "raw_score": round(result.raw_score if result.raw_score is not None else result.score, 4),
        "source_weight": round(result.source_weight if result.source_weight is not None else 1.0, 4),
        **({"scoring_profile": result.scoring_profile} if result.scoring_profile is not None else {}),
        **({"query_weight": round(result.query_weight, 4)} if result.query_weight is not None else {}),
        **({"matched_query": result.matched_query} if result.matched_query is not None else {}),
        **({"temporal_score": round(result.temporal_score, 4)} if result.temporal_score is not None else {}),
        **({"temporal_weight": round(result.temporal_weight, 4)} if result.temporal_weight is not None else {}),
        **(
            {"retrieval_salience": round(_retrieval_salience(result.entity), 4)}
            if _retrieval_salience(result.entity) != 1.0
            else {}
        ),
        **({"reranker": result.reranker} if result.reranker is not None else {}),
        **({"rerank_score": round(result.rerank_score, 4)} if result.rerank_score is not None else {}),
        **({"rerank_strategy": result.rerank_strategy} if result.rerank_strategy is not None else {}),
        **({"warnings": list(result.warnings)} if result.warnings else {}),
        **({"path_relation_types": path_relation_types} if result.source == "traversal" and path_relation_types else {}),
        **(
            {"retention_policy": result.entity.properties["_retention_policy"]}
            if "_retention_policy" in result.entity.properties
            else {}
        ),
        **(
            {
                "retention_decay_multiplier": round(
                    float(result.entity.properties["_retention_decay_multiplier"]),
                    4,
                )
            }
            if "_retention_decay_multiplier" in result.entity.properties
            else {}
        ),
        **(
            {"retention_expired": bool(result.entity.properties["_retention_expired"])}
            if "_retention_expired" in result.entity.properties
            else {}
        ),
        **(
            {"retention_half_life_days": int(result.entity.properties["_retention_half_life_days"])}
            if "_retention_half_life_days" in result.entity.properties
            else {}
        ),
        **(
            {"retention_purpose_profile": str(result.entity.properties["_retention_purpose_profile"])}
            if "_retention_purpose_profile" in result.entity.properties
            else {}
        ),
        **(
            {
                "inferred_edge_count": int(inferred_trust["count"]),
                "inferred_edge_trust": round(float(inferred_trust["trust"]), 4),
                "inferred_edge_trust_multiplier": round(float(inferred_trust["multiplier"]), 4),
                "inferred_edge_confidence": round(float(inferred_trust["confidence"]), 4),
                "inferred_edge_method_coverage": round(float(inferred_trust["method_coverage"]), 4),
                "inferred_edge_source_coverage": round(float(inferred_trust["source_coverage"]), 4),
                "inferred_edge_evidence_coverage": round(float(inferred_trust["evidence_coverage"]), 4),
                "inferred_relation_types": inferred_relation_types,
                "inference_methods": inference_methods,
            }
            if inferred_trust["count"]
            else {}
        ),
        "weighted_score": round(result.score, 4),
        "ranking_score": round(result.ranking_score if result.ranking_score is not None else result.score, 4),
    }


def _inferred_relation_types(properties: dict[str, Any]) -> list[str]:
    relation_types = [str(value) for value in _list_property(properties.get("_path_relation_types"))]
    inferred_flags = [
        bool(value) for value in _list_property(properties.get("_path_inferred_flags"))
    ]
    if inferred_flags and len(inferred_flags) == len(relation_types):
        return _unique_text(
            relation_type
            for relation_type, inferred in zip(relation_types, inferred_flags, strict=True)
            if inferred
        )
    return _unique_text(relation_types)


def _inference_methods(properties: dict[str, Any]) -> list[str]:
    return _unique_text(str(value) for value in _list_property(properties.get("_path_inference_methods")))


def _unique_text(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _citation(entity: GraphEntity) -> str | None:
    """Build a compact Eventloom citation for an entity result."""
    properties = entity.properties
    source_path = properties.get("source_path")
    source_start_line = properties.get("source_start_line")
    if source_path and source_start_line:
        return f"file://{source_path}:{source_start_line}"

    seq = properties.get("source_event_seq")
    event_hash = properties.get("source_event_hash")
    if seq is None or not event_hash:
        return None
    return f"eventloom://{entity.session_id}/events/{seq}#{str(event_hash)[:12]}"
