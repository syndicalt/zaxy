"""Query router: hybrid retrieval with temporal filtering.

The router fuses results from multiple search strategies (exact, traversal,
keyword) and applies temporal filters, deduplication, and ranking before
returning a context window suitable for injection into an agent prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from zaxy.graph import GraphStore, SearchResult


@dataclass(frozen=True)
class ContextChunk:
    """A ranked piece of context ready for the agent prompt."""

    content: str
    source: str
    score: float
    valid_from: str | None
    valid_to: str | None


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
        fusion_weights: dict[str, float] | None = None,
    ) -> None:
        self.store = store
        self.default_limit = default_limit
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
    ) -> list[ContextChunk]:
        """Run a hybrid query and return ranked context chunks.

        Strategy:
        1. Exact match (if the query looks like an entity name).
        2. Vector similarity search (if embedding provided).
        3. Keyword/BM25 search.
        4. Graph traversal from top keyword hits.
        5. Fuse, deduplicate, sort by score, truncate to limit.
        """
        lim = limit or self.default_limit
        results: list[SearchResult] = []

        # 1. Exact match attempt
        exact = await self.store.search_exact(query, temporal_point=temporal_point)
        for ent in exact:
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
                embedding, limit=lim, temporal_point=temporal_point
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
            query, limit=lim, temporal_point=temporal_point
        )
        for hit in keyword_hits:
            hit = SearchResult(
                entity=hit.entity,
                score=hit.score * self.fusion_weights["keyword"],
                source="keyword",
            )
            results.append(hit)

        # 4. Traversal from top keyword + vector hits (breadth expansion)
        seen = {r.entity.name for r in results}
        traversal_seeds = (vector_hits + keyword_hits)[:3]
        for hit in traversal_seeds:
            neighbors = await self.store.search_traversal(
                hit.entity.name,
                depth=2,
                temporal_point=temporal_point,
            )
            for neighbor in neighbors:
                if neighbor.name in seen:
                    continue
                results.append(
                    SearchResult(
                        entity=neighbor,
                        score=0.7 * self.fusion_weights["traversal"],
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
        props = ", ".join(f"{k}={v}" for k, v in list(ent.properties.items())[:3])
        content += f" — {props}"

    return ContextChunk(
        content=content,
        source=result.source,
        score=round(result.score, 4),
        valid_from=ent.valid_from,
        valid_to=ent.valid_to,
    )
