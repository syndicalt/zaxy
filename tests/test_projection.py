"""Tests for projection-store backend contracts."""

from __future__ import annotations

from pathlib import Path
from typing import assert_type

import pytest

from zaxy.embedded_graph_store import EmbeddedGraphStore
from zaxy.extract import ExtractionResult
from zaxy.graph import (
    GraphEntity,
    GraphEventProjectionStatus,
    GraphInferredEdgeStatus,
    GraphStore,
    SearchResult,
)
from zaxy.latticedb_store import LatticeDBStore
from zaxy.pggraph_store import PgGraphStore
from zaxy.projection import ProjectionStore
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store


class FakeProjectionStore:
    """Small structural implementation used to pin the projection contract."""

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def init_schema(self) -> None:
        pass

    async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None:
        pass

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        pass

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

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        return []

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        raise NotImplementedError

    async def inspect_inferred_edge_status(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> GraphInferredEdgeStatus:
        raise NotImplementedError

def test_fake_projection_store_satisfies_contract() -> None:
    store: ProjectionStore = FakeProjectionStore()
    assert store is not None


async def _type_probe(store: ProjectionStore) -> None:
    exact = await store.search_exact("Alice")
    keyword = await store.search_keyword("ship memory")
    traversal = await store.search_traversal("Alice")
    vector = await store.search_vector([0.1, 0.2])

    assert_type(exact, list[GraphEntity])
    assert_type(keyword, list[SearchResult])
    assert_type(traversal, list[GraphEntity])
    assert_type(vector, list[SearchResult])


def test_build_projection_store_routes_neo4j_to_adapter() -> None:
    store = build_projection_store(
        ProjectionBackendConfig(
            backend="neo4j",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="testpassword",
            neo4j_ca_cert=None,
            neo4j_trust_all=False,
        )
    )

    assert isinstance(store, GraphStore)


def test_build_projection_store_routes_pggraph_to_adapter() -> None:
    store = build_projection_store(
        ProjectionBackendConfig(
            backend="pggraph",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="testpassword",
            neo4j_ca_cert=None,
            neo4j_trust_all=False,
            pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        )
    )

    assert isinstance(store, PgGraphStore)


def test_build_projection_store_routes_embedded_to_adapter(tmp_path: Path) -> None:
    graph_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"

    store = build_projection_store(
        ProjectionBackendConfig(
            backend="embedded",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="testpassword",
            neo4j_ca_cert=None,
            neo4j_trust_all=False,
            embedded_graph_path=graph_path,
        )
    )

    assert isinstance(store, EmbeddedGraphStore)
    assert store.path == graph_path


def test_build_projection_store_requires_embedded_graph_path() -> None:
    with pytest.raises(ValueError, match="embedded backend requires embedded_graph_path"):
        build_projection_store(
            ProjectionBackendConfig(
                backend="embedded",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="testpassword",
                neo4j_ca_cert=None,
                neo4j_trust_all=False,
            )
        )


@pytest.mark.asyncio
async def test_embedded_store_missing_kuzu_error_points_to_core_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedded Kuzu is a core dependency, so remediation should not point to a redundant extra."""
    monkeypatch.setattr("zaxy.embedded_graph_store.importlib.util.find_spec", lambda name: None)
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")

    with pytest.raises(RuntimeError, match='pip install "zaxy-memory"') as exc_info:
        await store.connect()

    assert "zaxy-memory[embedded]" not in str(exc_info.value)


def test_build_projection_store_routes_latticedb_to_adapter(tmp_path: Path) -> None:
    graph_path = tmp_path / ".eventloom" / "projections" / "memory.latticedb"

    store = build_projection_store(
        ProjectionBackendConfig(
            backend="latticedb",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="testpassword",
            neo4j_ca_cert=None,
            neo4j_trust_all=False,
            latticedb_path=graph_path,
            embedding_dimension=1536,
        )
    )

    assert isinstance(store, LatticeDBStore)
    assert store.path == graph_path
    assert store.vector_dimensions == 1536


def test_build_projection_store_requires_latticedb_path() -> None:
    with pytest.raises(ValueError, match="LatticeDB backend requires latticedb_path"):
        build_projection_store(
            ProjectionBackendConfig(
                backend="latticedb",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="testpassword",
                neo4j_ca_cert=None,
                neo4j_trust_all=False,
            )
        )


def test_build_projection_store_invalid_backend_error_lists_embedded_first() -> None:
    """Factory errors should present embedded as the default production path."""
    with pytest.raises(
        ValueError,
        match="projection backend must be one of: embedded, neo4j, pggraph, latticedb",
    ):
        build_projection_store(
            ProjectionBackendConfig(
                backend="unknown",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="testpassword",
                neo4j_ca_cert=None,
                neo4j_trust_all=False,
            )
        )
