"""Query router: hybrid retrieval with temporal filtering.

The router fuses results from multiple search strategies (exact, traversal,
keyword) and applies temporal filters, deduplication, and ranking before
returning a context window suitable for injection into an agent prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
                    )
                )
                seen.add(neighbor.name)

        # 5. Deduplicate by (name, type), keep highest score
        best: dict[tuple[str, str], SearchResult] = {}
        for r in results:
            key = (r.entity.name, r.entity.entity_type)
            if key not in best or r.score > best[key].score:
                best[key] = r

        # 6. Sort and truncate
        ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)[:lim]

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


def _citation(entity: GraphEntity) -> str | None:
    """Build a compact Eventloom citation for an entity result."""
    properties = entity.properties
    seq = properties.get("source_event_seq")
    event_hash = properties.get("source_event_hash")
    if seq is None or not event_hash:
        return None
    return f"eventloom://{entity.session_id}/events/{seq}#{str(event_hash)[:12]}"
