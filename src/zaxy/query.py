"""Query router: hybrid retrieval with temporal filtering.

The router fuses results from multiple search strategies (exact, traversal,
keyword) and applies temporal filters, deduplication, and ranking before
returning a context window suitable for injection into an agent prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from zaxy.graph import GraphEntity, GraphStore, SearchResult
from zaxy.security import validate_limit, validate_query, validate_session_id


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
    ) -> None:
        self.store = store
        self.default_limit = default_limit
        self.session_id = validate_session_id(session_id)
        self.fusion_weights = fusion_weights or {
            "exact": 1.0,
            "vector": 0.95,
            "traversal": 0.9,
            "keyword": 0.8,
        }

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
                )
                results.append(hit)

        # 3. Keyword search
        keyword_hits = await self.store.search_keyword(
            query,
            limit=lim,
            temporal_point=temporal_point,
            session_id=scope,
        )
        for hit in keyword_hits:
            hit = SearchResult(
                entity=hit.entity,
                score=min(hit.score, 1.0) * self.fusion_weights["keyword"],
                source="keyword",
                raw_score=hit.score,
                source_weight=self.fusion_weights["keyword"],
            )
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
                    )
                )
                seen.add(neighbor.name)

        # 5. Deduplicate by (name, type), keep highest score
        best: dict[tuple[str, str], SearchResult] = {}
        for r in results:
            key = (r.entity.name, r.entity.entity_type)
            if key not in best or r.score > best[key].score:
                best[key] = r

        # 6. Sort with MMR diversity and truncate
        ranked = _mmr_rank(list(best.values()), limit=lim)

        return [_to_chunk(r) for r in ranked]


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


def _mmr_rank(results: list[SearchResult], limit: int, lambda_score: float = 0.7) -> list[SearchResult]:
    """Rank by weighted relevance while penalizing near-duplicate context."""
    candidates = sorted(results, key=lambda x: x.score, reverse=True)
    selected: list[SearchResult] = []
    source_bonus = {"traversal": 0.1}

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


def _score_explanation(result: SearchResult) -> dict[str, Any]:
    """Return compact score details for retrieval debugging."""
    return {
        "source": result.source,
        "raw_score": round(result.raw_score if result.raw_score is not None else result.score, 4),
        "source_weight": round(result.source_weight if result.source_weight is not None else 1.0, 4),
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
