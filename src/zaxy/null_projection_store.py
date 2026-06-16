"""A no-op projection store for graph-degraded operation.

Used when the real embedded projection cannot be opened — most commonly because
a long-lived process (the MCP server) holds LadybugDB's exclusive write lock and
a second process (a CLI checkout) cannot open it concurrently. The previous
behavior stood up a throwaway per-process *empty* projection, which paid full
schema/index setup (~10s on first use) yet returned no graph results anyway.

This store skips all of that: it opens nothing, locks nothing, projects nothing,
and returns empty results for every lane. A checkout backed by it proceeds on the
verbatim + verified-replay lanes (which are independent of the graph), so the
result is the same a locked-out checkout produced before — minus the wasted
graph stand-up. It is a degraded mode, surfaced via checkout diagnostics, never a
silent substitute for the real projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from zaxy.extract import ExtractionResult
    from zaxy.graph import (
        GraphEntity,
        GraphEventProjectionStatus,
        GraphInferredEdgeStatus,
        SearchResult,
    )


class NullProjectionStore:
    """A ``ProjectionStore`` that holds no state and returns empty results."""

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def init_schema(self) -> None:
        return None

    async def upsert_extraction(
        self, result: ExtractionResult, session_id: str = "default"
    ) -> None:
        return None

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        return []

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        return []

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        return []

    async def search_causal_neighbors(
        self,
        entity_name: str,
        *,
        direction: Literal["successors", "predecessors"],
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        return []

    async def has_traversal_edges(self, session_id: str = "default") -> bool:
        return False

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        return []

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        return None

    async def retire_source_projections(
        self,
        *,
        source_path: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        return None

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        from zaxy.graph import GraphEventProjectionStatus

        return GraphEventProjectionStatus(
            session_id=session_id,
            event_count=0,
            latest_seq=None,
            latest_hash=None,
            eventloom_latest_seq=eventloom_latest_seq,
            eventloom_latest_hash=eventloom_latest_hash,
            projection_lag=None,
            latest_hash_matches=False,
            next_event_edges=0,
            previous_event_edges=0,
            missing_chain_links=0,
            integrity_ok=False,
        )

    async def inspect_inferred_edge_status(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> GraphInferredEdgeStatus:
        from zaxy.graph import GraphInferredEdgeStatus

        return GraphInferredEdgeStatus(
            session_id=session_id,
            total_edges=0,
            method_count=0,
            evidence_count=0,
            missing_evidence_count=0,
            missing_source_event_count=0,
            evidence_coverage=1.0,
            methods=(),
            samples=(),
        )
