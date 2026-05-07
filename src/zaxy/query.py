"""Query router: hybrid retrieval with temporal filtering.

The router fuses results from multiple search strategies (exact, traversal,
keyword) and applies temporal filters, deduplication, and ranking before
returning a context window suitable for injection into an agent prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Protocol

from zaxy.graph import GraphEntity, GraphStore, SearchResult
from zaxy.security import validate_limit, validate_query, validate_session_id

QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "auth": ("authentication", "authorization"),
    "decision": ("decided", "choice", "rationale"),
    "decisions": ("decided", "choice", "rationale"),
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


class LexicalReranker:
    """Deterministic local reranker based on query-token overlap."""

    name = "lexical"

    def __init__(self, weight: float = 0.35) -> None:
        self.weight = weight

    async def rerank(self, query: str, results: list[SearchResult], *, limit: int) -> list[SearchResult]:
        query_tokens = _tokens(query)
        reranked: list[SearchResult] = []
        for result in results:
            score = _lexical_rerank_score(query_tokens, result.entity)
            weighted_score = result.score + (self.weight * score)
            reranked.append(
                replace(
                    result,
                    score=weighted_score,
                    ranking_score=weighted_score,
                    reranker=self.name,
                    rerank_score=score,
                )
            )
        return sorted(reranked, key=lambda item: item.ranking_score or item.score, reverse=True)[:limit]


class QueryRouter:
    """Routes natural-language queries to the graph store and fuses results.

    Args:
        store: Connected GraphStore instance.
        default_limit: Max results per sub-query.
        fusion_weights: Dict of {source: weight} for score normalization.
    """

    def __init__(
        self,
        store: GraphStore,
        default_limit: int = 10,
        session_id: str = "default",
        fusion_weights: dict[str, float] | None = None,
        scoring_profile: str | ScoringProfile = "balanced",
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.default_limit = default_limit
        self.session_id = validate_session_id(session_id)
        self.scoring_profile = _resolve_scoring_profile(scoring_profile, fusion_weights)
        self.fusion_weights = self.scoring_profile.fusion_weights
        self.temporal_weight = self.scoring_profile.temporal_weight
        self.reranker = reranker

    async def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int | None = None,
        embedding: list[float] | None = None,
        session_id: str | None = None,
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
        scope = validate_session_id(session_id or self.session_id)
        results: list[SearchResult] = []

        # 1. Exact match attempt against the full query and structured entity
        # names embedded in natural-language questions.
        exact_entities = []
        for candidate in _exact_candidates(query):
            exact_entities.extend(
                await self.store.search_exact(
                    candidate,
                    temporal_point=temporal_point,
                    session_id=scope,
                )
            )
        for ent in exact_entities:
            results.append(
                SearchResult(
                    entity=ent,
                    score=1.0 * self.fusion_weights["exact"],
                    source="exact",
                    raw_score=1.0,
                    source_weight=self.fusion_weights["exact"],
                    matched_query=candidate,
                    scoring_profile=self.scoring_profile.name,
                )
            )

        # 2. Vector search (if embedding provided)
        vector_hits: list[SearchResult] = []
        if embedding:
            vector_hits = await self.store.search_vector(
                embedding,
                limit=lim,
                temporal_point=temporal_point,
                session_id=scope,
            )
            for hit in vector_hits:
                hit = SearchResult(
                    entity=hit.entity,
                    score=hit.score * self.fusion_weights["vector"],
                    source="vector",
                    raw_score=hit.score,
                    source_weight=self.fusion_weights["vector"],
                    matched_query=query,
                    scoring_profile=self.scoring_profile.name,
                )
                results.append(_apply_temporal_score(hit, temporal_point, self.temporal_weight))

        # 3. Keyword search
        keyword_hits: list[SearchResult] = []
        for keyword_query in _expanded_queries(query):
            query_weight = 1.0 if keyword_query == query else self.scoring_profile.expansion_weight
            query_hits = await self.store.search_keyword(
                keyword_query,
                limit=lim,
                temporal_point=temporal_point,
                session_id=scope,
            )
            for hit in query_hits:
                hit = SearchResult(
                    entity=hit.entity,
                    score=min(hit.score, 1.0) * self.fusion_weights["keyword"] * query_weight,
                    source="keyword",
                    raw_score=hit.score,
                    source_weight=self.fusion_weights["keyword"],
                    matched_query=keyword_query,
                    query_weight=query_weight,
                    scoring_profile=self.scoring_profile.name,
                )
                hit = _apply_temporal_score(hit, temporal_point, self.temporal_weight)
                keyword_hits.append(hit)
                results.append(hit)

        # 4. Traversal from top keyword + vector hits (breadth expansion)
        seen = {r.entity.name for r in results}
        traversal_seeds = results[:3] + (vector_hits + keyword_hits)[:3]
        for hit in traversal_seeds:
            neighbors = await self.store.search_traversal(
                hit.entity.name,
                depth=2,
                temporal_point=temporal_point,
                session_id=scope,
            )
            for neighbor in neighbors:
                if neighbor.name in seen:
                    continue
                results.append(
                    SearchResult(
                        entity=neighbor,
                        score=0.95 * self.fusion_weights["traversal"],
                        source="traversal",
                        raw_score=0.95,
                        source_weight=self.fusion_weights["traversal"],
                        scoring_profile=self.scoring_profile.name,
                    )
                )
                seen.add(neighbor.name)

        # 5. Deduplicate by (name, type), keep highest score
        best: dict[tuple[str, str], SearchResult] = {}
        for r in results:
            key = (r.entity.name, r.entity.entity_type)
            if key not in best or r.score > best[key].score:
                best[key] = r

        # 6. Sort with either provider reranking or MMR diversity and truncate
        ranked = await self._rank(query, list(best.values()), lim)

        return [_to_chunk(r) for r in ranked]

    async def _rank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        candidates = _mmr_rank(
            results,
            limit=len(results),
            lambda_score=self.scoring_profile.mmr_lambda,
            traversal_bonus=self.scoring_profile.traversal_bonus,
        )
        if self.reranker is None:
            return candidates[:limit]
        reranked = await self.reranker.rerank(query, candidates, limit=limit)
        return reranked[:limit]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_chunk(result: SearchResult) -> ContextChunk:
    """Convert a SearchResult to a ContextChunk."""
    ent = result.entity
    # Build a concise natural-language summary of the entity
    content = f"{ent.name} ({ent.entity_type})"
    if ent.properties:
        safe_properties = {
            key: value
            for key, value in ent.properties.items()
            if key not in {"embedding", "created_at", "updated_at"}
        }
        props = ", ".join(f"{k}={v}" for k, v in list(safe_properties.items())[:3])
        if props:
            content += f" — {props}"

    return ContextChunk(
        content=content,
        source=result.source,
        score=round(result.score, 4),
        valid_from=ent.valid_from,
        valid_to=ent.valid_to,
        citation=_citation(ent),
        score_explanation=_score_explanation(result),
    )


def _exact_candidates(query: str) -> list[str]:
    """Return exact entity candidates named inside a natural-language query."""
    candidates = [query]
    patterns = [
        r"\bGoal\s+\d{4}\b",
        r"\btask-\d{4}\b",
        r"\buser-\d{4}:[A-Za-z0-9_.-]+\b",
        r"\buser-\d{4}\b",
    ]
    for pattern in patterns:
        candidates.extend(match.group(0) for match in re.finditer(pattern, query))

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _expanded_queries(query: str) -> list[str]:
    """Return the original query plus deterministic synonym expansions."""
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    expanded_terms: list[str] = []
    for token in tokens:
        expanded_terms.extend(QUERY_EXPANSIONS.get(token, ()))
    expanded_terms = _unique(expanded_terms)
    if not expanded_terms:
        return [query]
    return [query, f"{query} {' '.join(expanded_terms)}"]


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


def _mmr_rank(
    results: list[SearchResult],
    limit: int,
    lambda_score: float = 0.7,
    traversal_bonus: float = 0.1,
) -> list[SearchResult]:
    """Rank by weighted relevance while penalizing near-duplicate context."""
    candidates = sorted(results, key=lambda x: x.score, reverse=True)
    selected: list[SearchResult] = []
    source_bonus = {"traversal": traversal_bonus}

    while candidates and len(selected) < limit:
        if not selected:
            first = candidates.pop(0)
            selected.append(replace(first, ranking_score=first.score))
            continue

        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(candidates):
            similarity = max(_entity_similarity(candidate.entity, hit.entity) for hit in selected)
            ranking_score = (
                lambda_score * candidate.score
                - (1.0 - lambda_score) * similarity
                + source_bonus.get(candidate.source, 0.0)
            )
            if ranking_score > best_score:
                best_index = index
                best_score = ranking_score

        selected.append(replace(candidates.pop(best_index), ranking_score=best_score))

    return selected


def _entity_similarity(left: GraphEntity, right: GraphEntity) -> float:
    """Return token Jaccard similarity between two entities."""
    left_tokens = _entity_tokens(left)
    right_tokens = _entity_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


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
        for token in re.findall(r"[A-Za-z0-9]+", " ".join(text_parts).lower())
        if len(token) > 1
    }


def _lexical_rerank_score(query_tokens: set[str], entity: GraphEntity) -> float:
    """Score an entity for second-stage lexical relevance."""
    if not query_tokens:
        return 0.0
    entity_tokens = _entity_tokens(entity)
    overlap = len(query_tokens & entity_tokens) / len(query_tokens)
    phrase_bonus = 0.0
    lower_text = " ".join([entity.name, str(entity.properties.get("summary", ""))]).lower()
    for token in query_tokens:
        if token in lower_text:
            phrase_bonus += 0.03
    return min(overlap + phrase_bonus, 1.0)


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]+", value.lower()) if len(token) > 1}


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


def _score_explanation(result: SearchResult) -> dict[str, Any]:
    """Return compact score details for retrieval debugging."""
    return {
        "source": result.source,
        "raw_score": round(result.raw_score if result.raw_score is not None else result.score, 4),
        "source_weight": round(result.source_weight if result.source_weight is not None else 1.0, 4),
        **({"scoring_profile": result.scoring_profile} if result.scoring_profile is not None else {}),
        **({"query_weight": round(result.query_weight, 4)} if result.query_weight is not None else {}),
        **({"matched_query": result.matched_query} if result.matched_query is not None else {}),
        **({"temporal_score": round(result.temporal_score, 4)} if result.temporal_score is not None else {}),
        **({"temporal_weight": round(result.temporal_weight, 4)} if result.temporal_weight is not None else {}),
        **({"reranker": result.reranker} if result.reranker is not None else {}),
        **({"rerank_score": round(result.rerank_score, 4)} if result.rerank_score is not None else {}),
        "weighted_score": round(result.score, 4),
        "ranking_score": round(result.ranking_score if result.ranking_score is not None else result.score, 4),
    }


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
