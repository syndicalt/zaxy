"""Tests for projection-store backend contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import assert_type

import pytest

from zaxy.embedded_graph_internals import (
    _edge_provenance_from_row,
    _EdgeProvenance,
    _evidence_key_count,
    _path_inferred_metadata,
)
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
async def test_embedded_store_missing_engine_error_points_to_core_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedded LadybugDB is a core dependency, so remediation should not point to a redundant extra."""
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


# --------------------------------------------------------------------------
# Inferred-edge provenance on the default (embedded) backend.
# --------------------------------------------------------------------------


def test_evidence_key_count_counts_projectable_scalar_keys() -> None:
    """Evidence key counting should mirror Neo4j's namespaced-property filter."""
    assert (
        _evidence_key_count(
            json.dumps(
                {
                    "source_event_seq": 12,
                    "reason": "cited",
                    "tags": ["a", "b"],
                    "nested": {"not": "scalar"},
                    "1bad-key": "skipped",
                    "empty": None,
                }
            )
        )
        == 3
    )


@pytest.mark.parametrize("raw", ["", None, "{not-json", "[1, 2]", '"text"'])
def test_evidence_key_count_is_zero_for_missing_or_malformed_json(raw: object) -> None:
    """Malformed or non-object evidence blobs should count as zero, never raise."""
    assert _evidence_key_count(raw) == 0


def test_edge_provenance_from_row_ignores_non_inferred_edges() -> None:
    """A plain extracted edge carries no inferred provenance."""
    provenance = _edge_provenance_from_row(
        inferred=False,
        confidence=0.9,
        inference_method="ignored",
        source_event_seq=4,
        source_event_hash="a" * 64,
        evidence_json='{"reason": "ignored"}',
    )

    assert provenance.inferred is False
    assert provenance.confidence is None
    assert provenance.method is None
    assert provenance.has_source_event_ref is False
    assert provenance.evidence_key_count == 0


def test_edge_provenance_from_row_reads_stored_inferred_columns() -> None:
    """An inferred edge surfaces exactly the provenance stored on the row."""
    provenance = _edge_provenance_from_row(
        inferred=True,
        confidence=0.86,
        inference_method="task_completed_decision_citation_v1",
        source_event_seq=4,
        source_event_hash="a" * 64,
        evidence_json='{"reason": "cited", "source_event_seq": 4}',
    )

    assert provenance.inferred is True
    assert provenance.confidence == 0.86
    assert provenance.method == "task_completed_decision_citation_v1"
    assert provenance.has_source_event_ref is True
    assert provenance.evidence_key_count == 2


def test_edge_provenance_from_row_marks_missing_citation_refs() -> None:
    """An inferred edge without both seq and hash must not claim a source citation."""
    provenance = _edge_provenance_from_row(
        inferred=True,
        confidence=0.2,
        inference_method="  ",
        source_event_seq=None,
        source_event_hash="",
        evidence_json="{}",
    )

    assert provenance.has_source_event_ref is False
    assert provenance.method is None
    assert provenance.evidence_key_count == 0


def test_path_inferred_metadata_is_empty_for_fully_cited_paths() -> None:
    """A path with no inferred edges stays at the neutral trust multiplier."""
    plain = _EdgeProvenance(
        inferred=False,
        confidence=None,
        method=None,
        has_source_event_ref=False,
        evidence_key_count=0,
    )

    assert _path_inferred_metadata([plain, plain]) == {}


def test_path_inferred_metadata_aggregates_only_inferred_edges() -> None:
    """Path metadata counts confidences, methods and evidence from inferred edges only."""
    plain = _EdgeProvenance(
        inferred=False,
        confidence=None,
        method=None,
        has_source_event_ref=False,
        evidence_key_count=0,
    )
    inferred = _EdgeProvenance(
        inferred=True,
        confidence=0.86,
        method="task_completed_decision_citation_v1",
        has_source_event_ref=True,
        evidence_key_count=5,
    )

    metadata = _path_inferred_metadata([plain, inferred])

    assert metadata == {
        "_path_inferred_flags": [False, True],
        "_path_inferred_edge_count": 1,
        "_path_inferred_confidences": [0.86],
        "_path_inference_methods": ["task_completed_decision_citation_v1"],
        "_path_inferred_source_event_count": 1,
        "_path_inferred_evidence_count": 5,
        "_path_inferred_evidenced_edge_count": 1,
    }


@pytest.mark.skipif(importlib.util.find_spec("ladybug") is None, reason="ladybug is not installed")
async def test_embedded_traversal_downweights_uncited_inferred_paths(tmp_path: Path) -> None:
    """Real embedded events must produce inferred provenance that demotes uncited paths.

    End-to-end on the default backend with no hand-injected ``_path_inferred_*``
    properties: both inferred edges come from appended Eventloom events, so this
    fails if embedded traversal drops the provenance the trust scorer consumes.
    """
    from zaxy.core import MemoryFabric
    from zaxy.query import _inferred_edge_trust_metadata

    cited_decision = "Adopt embedded LadybugDB as the default backend"
    uncited_decision = "Rewrite the projection layer in Rust"
    session_id = "agent-inferred-trust"
    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        projection_backend="embedded",
        tracer_disabled=True,
    )
    await fabric.connect()
    try:
        decision_event = await fabric.append(
            "decision.made",
            actor="agent",
            payload={"decision": cited_decision, "rationale": "Zero-daemon and file-based."},
            session_id=session_id,
        )
        await fabric.append(
            "task.completed",
            actor="agent",
            payload={
                "taskId": "task-7",
                "summary": "Wired the embedded projection.",
                "decision": cited_decision,
                "decision_event_seq": decision_event.seq,
                "decision_event_hash": decision_event.hash,
            },
            session_id=session_id,
        )
        await fabric.append(
            "inference.edge.generated",
            actor="agent",
            payload={
                "source": {"name": "task-7", "entity_type": "task"},
                "target": {"name": uncited_decision, "entity_type": "decision"},
                "relation_type": "likely_implemented_decision",
                "confidence": 0.2,
                "inference_method": "unknown",
                "evidence": {},
            },
            session_id=session_id,
        )

        neighbors = await fabric.graph.search_traversal("task-7", depth=1, session_id=session_id)
        by_name = {entity.name: entity for entity in neighbors}
        assert cited_decision in by_name
        assert uncited_decision in by_name

        cited_properties = by_name[cited_decision].properties
        assert cited_properties["_path_inferred_edge_count"] == 1
        assert cited_properties["_path_inference_methods"] == [
            "task_completed_decision_citation_v1"
        ]
        assert cited_properties["_path_inferred_evidenced_edge_count"] == 1

        cited_trust = _inferred_edge_trust_metadata(cited_properties)
        uncited_trust = _inferred_edge_trust_metadata(by_name[uncited_decision].properties)
        assert uncited_trust["multiplier"] < 1.0 < cited_trust["multiplier"]

        results = await fabric.query_router.query(
            "Which decision did task-7 implement?",
            session_id=session_id,
            limit=6,
        )
        scores = {result.entity_name: result.score for result in results}
        assert scores[cited_decision] > scores[uncited_decision]

        checkout = await fabric.checkout_memory(
            "Which decision did task-7 implement?",
            session_id=session_id,
            limit=6,
        )
        inferred_context = checkout.diagnostics["inferred_context"]
        assert inferred_context["context_count"] >= 1
        assert inferred_context["edge_count"] >= 1
        assert inferred_context["relation_types"] == ["likely_implemented_decision"]
        assert inferred_context["inference_methods"] == [
            "task_completed_decision_citation_v1"
        ]
    finally:
        await fabric.close()
