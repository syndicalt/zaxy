"""Tests for the embedded graph projection backend."""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import zaxy.embedded_graph_store as embedded_graph_store
from zaxy.embedded_graph_store import (
    LEGACY_EMBEDDING_VERSION,
    EmbeddedGraphStore,
    _keyword_candidate_terms,
    _keyword_index_from_entities,
    _keyword_query_terms,
    _properties_reference_source,
    _terms,
)
from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult
from zaxy.graph import GraphEntity
from zaxy.graph_walk import AdjacencySnapshot

pytestmark = pytest.mark.skipif(importlib.util.find_spec("ladybug") is None, reason="ladybug is not installed")


@lru_cache(maxsize=1)
def _native_vector_index_available() -> bool:
    """Return whether this LadybugDB wheel exposes the native vector index."""
    if importlib.util.find_spec("ladybug") is None:
        return False
    store = EmbeddedGraphStore(Path(tempfile.mkdtemp()) / "vector-probe.kuzu")
    try:
        asyncio.run(store.connect())
        asyncio.run(store.init_schema())
        return store._vector_index_supported()
    except RuntimeError:
        return False
    finally:
        asyncio.run(store.close())


requires_native_vector_index = pytest.mark.skipif(
    not _native_vector_index_available(),
    reason="LadybugDB native vector index extension is not available in this environment",
)


class _FakeRows:
    def __init__(self, rows: list[list[object]]) -> None:
        self._rows = rows

    def get_all(self) -> list[list[object]]:
        return self._rows


class _CountingConnection:
    def __init__(self) -> None:
        self.active_state_loads = 0
        self.queries: list[tuple[str, dict[str, object] | None]] = []

    def execute(self, query: str, params: dict[str, object] | None = None) -> _FakeRows:
        self.queries.append((query, params))
        if "WHERE e.session_id = $session_id" in query and "RETURN e.node_key, e.name" in query:
            self.active_state_loads += 1
            return _FakeRows([])
        if "WHERE e.session_id = $session_id" in query and "RETURN e.node_key, e.summary" in query:
            self.active_state_loads += 1
            return _FakeRows([])
        return _FakeRows([])


def test_embedded_store_bulk_active_state_loads_session_once() -> None:
    """Bulk replay should not issue one active-entity lookup per new entity."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    connection = _CountingConnection()
    store._connection = connection
    store._bulk_projection_open = True

    assert store._active_entity_state(
        session_id="agent-1",
        entity_type="document",
        name="first",
    ) is None
    assert store._active_entity_state(
        session_id="agent-1",
        entity_type="document",
        name="second",
    ) is None

    assert connection.active_state_loads == 1


def test_embedded_store_clear_read_caches_removes_session_indexes_only() -> None:
    """Projection mutations should share one read-cache invalidation boundary."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    store._current_entity_index_cache = {"agent-1": [], "agent-2": []}
    store._current_entity_lookup_cache = {"agent-1": {}, "agent-2": {}}
    store._temporal_entity_index_cache = {("agent-1", "2026-05-20T01:00:00Z"): [], ("agent-2", "then"): []}
    store._temporal_entity_lookup_cache = {("agent-1", "2026-05-20T01:00:00Z"): {}, ("agent-2", "then"): {}}
    store._keyword_index_cache = {
        "agent-1": embedded_graph_store._KeywordIndex([], [], {}, {}, []),
        "agent-2": embedded_graph_store._KeywordIndex([], [], {}, {}, []),
    }
    store._temporal_keyword_index_cache = {
        ("agent-1", "2026-05-20T01:00:00Z"): embedded_graph_store._KeywordIndex([], [], {}, {}, []),
        ("agent-2", "then"): embedded_graph_store._KeywordIndex([], [], {}, {}, []),
    }
    store._vector_index_cache = {
        ("agent-1", None): embedded_graph_store._VectorIndex([], {}),
        ("agent-1", "2026-05-20T01:00:00Z"): embedded_graph_store._VectorIndex([], {}),
        ("agent-2", None): embedded_graph_store._VectorIndex([], {}),
    }
    store._traversal_index_cache = {
        "agent-1": embedded_graph_store._TraversalIndex({}, {}),
        "agent-2": embedded_graph_store._TraversalIndex({}, {}),
    }
    store._temporal_traversal_index_cache = {
        ("agent-1", "2026-05-20T01:00:00Z"): embedded_graph_store._TraversalIndex({}, {}),
        ("agent-2", "then"): embedded_graph_store._TraversalIndex({}, {}),
    }

    store._clear_read_caches("agent-1")

    assert "agent-1" not in store._current_entity_index_cache
    assert "agent-1" not in store._current_entity_lookup_cache
    assert all(key[0] != "agent-1" for key in store._temporal_entity_index_cache)
    assert all(key[0] != "agent-1" for key in store._temporal_entity_lookup_cache)
    assert "agent-1" not in store._keyword_index_cache
    assert all(key[0] != "agent-1" for key in store._temporal_keyword_index_cache)
    assert all(key[0] != "agent-1" for key in store._vector_index_cache)
    assert "agent-1" not in store._traversal_index_cache
    assert all(key[0] != "agent-1" for key in store._temporal_traversal_index_cache)
    assert "agent-2" in store._current_entity_index_cache
    assert "agent-2" in store._current_entity_lookup_cache
    assert ("agent-2", "then") in store._temporal_entity_index_cache
    assert ("agent-2", "then") in store._temporal_entity_lookup_cache
    assert "agent-2" in store._keyword_index_cache
    assert ("agent-2", "then") in store._temporal_keyword_index_cache
    assert ("agent-2", None) in store._vector_index_cache
    assert "agent-2" in store._traversal_index_cache
    assert ("agent-2", "then") in store._temporal_traversal_index_cache


def test_embedded_store_clear_session_keyed_cache_removes_session_only() -> None:
    """Tuple-keyed read caches should share one session-filtering helper."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    cache = {
        ("agent-1", None): "current",
        ("agent-1", "2026-05-20T01:00:00Z"): "temporal",
        ("agent-2", None): "other",
    }

    assert store._clear_session_keyed_cache(cache, "agent-1") == {("agent-2", None): "other"}


@pytest.mark.asyncio
async def test_embedded_store_close_clears_active_entity_cache(tmp_path: Path) -> None:
    """A closed embedded store should not retain active projection state."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    store._active_entity_cache[("agent-1", "goal", "Cached Goal")] = ("node-1", None, "{}")

    await store.close()

    assert store._active_entity_cache == {}
    with pytest.raises(RuntimeError, match="embedded graph store is not connected"):
        store._active_entity_state(
            session_id="agent-1",
            entity_type="goal",
            name="Cached Goal",
        )


def test_embedded_store_active_node_key_reuses_active_entity_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Active node lookup should share active entity state loading and caching."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    def active_entity_state(*, session_id: str, entity_type: str, name: str) -> tuple[str, str | None, str] | None:
        assert session_id == "agent-1"
        assert entity_type == "goal"
        assert name == "Shared Goal"
        return ("node-123", "summary", "{}")

    def fail_connection() -> object:
        raise AssertionError("active node key should reuse _active_entity_state")

    monkeypatch.setattr(store, "_active_entity_state", active_entity_state)
    monkeypatch.setattr(store, "_require_connection", fail_connection)

    assert store._active_node_key("agent-1", "goal", "Shared Goal") == "node-123"


def test_embedded_store_current_entity_lookup_orders_versions_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact lookup cache should store newest active versions first."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    older = GraphEntity(
        name="Policy",
        entity_type="decision",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )
    newer = GraphEntity(
        name="Policy",
        entity_type="decision",
        valid_from="2026-05-20T02:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )

    monkeypatch.setattr(store, "_current_entities", lambda session_id: [older, newer])

    lookup = store._current_entity_lookup("agent-1")

    assert lookup[("Policy", None)] == [newer, older]
    assert lookup[("Policy", "decision")] == [newer, older]


@pytest.mark.asyncio
async def test_embedded_store_cached_exact_search_does_not_resort(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact hot-path lookups should reuse lookup ordering without per-query sorting."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    entity = GraphEntity(
        name="Cached Goal",
        entity_type="goal",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )
    store._current_entity_lookup_cache = {"agent-1": {("Cached Goal", None): [entity]}}

    def fail_sorted(*args, **kwargs) -> object:
        raise AssertionError("cached exact lookup should not sort per query")

    monkeypatch.setattr(builtins, "sorted", fail_sorted)

    assert await store.search_exact("Cached Goal", session_id="agent-1") == [entity]


def test_embedded_store_temporal_entity_lookup_orders_versions_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temporal exact lookup cache should store newest matching versions first."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    older = GraphEntity(
        name="Policy",
        entity_type="decision",
        valid_from="2026-05-20T01:00:00Z",
        valid_to="2026-05-20T03:00:00Z",
        properties={},
        session_id="agent-1",
    )
    newer = GraphEntity(
        name="Policy",
        entity_type="decision",
        valid_from="2026-05-20T02:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )

    monkeypatch.setattr(store, "_temporal_entities", lambda session_id, temporal_point: [older, newer])

    lookup = store._temporal_entity_lookup("agent-1", "2026-05-20T02:30:00Z")

    assert lookup[("Policy", None)] == [newer, older]
    assert lookup[("Policy", "decision")] == [newer, older]


@pytest.mark.asyncio
async def test_embedded_store_temporal_exact_search_uses_temporal_lookup_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated temporal exact lookups should avoid per-query list scans and sorting."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    entity = GraphEntity(
        name="Historical Goal",
        entity_type="goal",
        valid_from="2026-05-20T01:00:00Z",
        valid_to="2026-05-20T03:00:00Z",
        properties={},
        session_id="agent-1",
    )

    def temporal_lookup(
        session_id: str,
        temporal_point: str,
    ) -> dict[tuple[str, str | None], list[GraphEntity]]:
        assert session_id == "agent-1"
        assert temporal_point == "2026-05-20T02:00:00Z"
        return {("Historical Goal", "goal"): [entity]}

    def fail_temporal_entities(session_id: str, temporal_point: str) -> list[GraphEntity]:
        raise AssertionError("temporal exact search should use temporal lookup cache")

    monkeypatch.setattr(store, "_temporal_entity_lookup", temporal_lookup)
    monkeypatch.setattr(store, "_temporal_entities", fail_temporal_entities)

    assert await store.search_exact(
        "Historical Goal",
        entity_type="goal",
        temporal_point="2026-05-20T02:00:00Z",
        session_id="agent-1",
    ) == [entity]


def test_embedded_store_temporal_keyword_index_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temporal keyword searches at the same point should reuse BM25 statistics."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    entity = GraphEntity(
        name="Historical Note",
        entity_type="document",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={"summary": "historical marker"},
        session_id="agent-1",
    )
    build_calls = 0
    original_builder = embedded_graph_store._keyword_index_from_entities

    def tracking_builder(entities: list[GraphEntity]) -> embedded_graph_store._KeywordIndex:
        nonlocal build_calls
        build_calls += 1
        return original_builder(entities)

    monkeypatch.setattr(store, "_temporal_entities", lambda session_id, temporal_point: [entity])
    monkeypatch.setattr(embedded_graph_store, "_keyword_index_from_entities", tracking_builder)

    first = store._keyword_index("agent-1", "2026-05-20T02:00:00Z")
    second = store._keyword_index("agent-1", "2026-05-20T02:00:00Z")

    assert first is second
    assert build_calls == 1


def test_embedded_store_temporal_traversal_index_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temporal traversal at the same point should reuse its adjacency index."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    build_calls = 0
    index = embedded_graph_store._TraversalIndex(adjacency={}, keys_by_name={})

    def build_traversal_index(session_id: str, temporal_point: str | None) -> embedded_graph_store._TraversalIndex:
        nonlocal build_calls
        assert session_id == "agent-1"
        assert temporal_point == "2026-05-20T02:00:00Z"
        build_calls += 1
        return index

    monkeypatch.setattr(store, "_build_traversal_index", build_traversal_index)

    first = store._traversal_index("agent-1", "2026-05-20T02:00:00Z")
    second = store._traversal_index("agent-1", "2026-05-20T02:00:00Z")

    assert first is index
    assert second is index
    assert build_calls == 1


@pytest.mark.asyncio
async def test_embedded_store_traversal_skips_duplicate_path_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate traversal paths should not rebuild path metadata for an already found target."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    target = GraphEntity(
        name="Shared Target",
        entity_type="task",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )
    index = embedded_graph_store._TraversalIndex(
        adjacency={
            "root-a": [("target", target, "relates_to")],
            "root-b": [("target", target, "relates_to")],
        },
        keys_by_name={"Root": {"root-a", "root-b"}},
    )
    metadata_calls = 0

    def traversal_index(session_id: str, temporal_point: str | None = None) -> embedded_graph_store._TraversalIndex:
        assert session_id == "agent-1"
        assert temporal_point is None
        return index

    def entity_with_path_metadata(entity: GraphEntity, *, relation_types: list[str]) -> GraphEntity:
        nonlocal metadata_calls
        metadata_calls += 1
        return GraphEntity(
            name=entity.name,
            entity_type=entity.entity_type,
            valid_from=entity.valid_from,
            valid_to=entity.valid_to,
            properties={"_path_relation_types": relation_types, "_path_length": len(relation_types)},
            session_id=entity.session_id,
        )

    monkeypatch.setattr(store, "_traversal_index", traversal_index)
    monkeypatch.setattr(embedded_graph_store, "_entity_with_path_metadata", entity_with_path_metadata)

    results = await store.search_traversal("Root", depth=1, session_id="agent-1")

    assert [entity.name for entity in results] == ["Shared Target"]
    assert metadata_calls == 1


def test_embedded_store_traversal_index_reuses_entities_by_node_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Traversal index construction should not rehydrate the same node once per edge row."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    rows = [
        [
            "source-key",
            "Hub",
            "goal",
            "2026-05-20T01:00:00Z",
            None,
            "hub",
            "{}",
            "agent-1",
            1,
            "hash-1",
            "supports",
            "target-1",
            "Leaf One",
            "task",
            "2026-05-20T01:00:00Z",
            None,
            "leaf one",
            "{}",
            "agent-1",
            1,
            "hash-1",
        ],
        [
            "source-key",
            "Hub",
            "goal",
            "2026-05-20T01:00:00Z",
            None,
            "hub",
            "{}",
            "agent-1",
            1,
            "hash-1",
            "blocks",
            "target-2",
            "Leaf Two",
            "task",
            "2026-05-20T01:00:00Z",
            None,
            "leaf two",
            "{}",
            "agent-1",
            1,
            "hash-1",
        ],
    ]

    class _TraversalConnection:
        def execute(self, query: str, params: dict[str, object] | None = None) -> _FakeRows:
            assert "MATCH (source:Entity)-[r:RELATES]->(target:Entity)" in query
            assert params == {"session_id": "agent-1"}
            return _FakeRows(rows)

    row_to_entity_calls = 0

    def row_to_entity(row: list[object]) -> GraphEntity:
        nonlocal row_to_entity_calls
        row_to_entity_calls += 1
        return GraphEntity(
            name=str(row[0]),
            entity_type=str(row[1]),
            valid_from=str(row[2]),
            valid_to=row[3] if isinstance(row[3], str) else None,
            properties={},
            session_id=str(row[6]),
        )

    store._connection = _TraversalConnection()
    monkeypatch.setattr(embedded_graph_store, "_row_to_entity", row_to_entity)

    index = store._build_traversal_index("agent-1", None)

    assert set(index.adjacency) == {"source-key", "target-1", "target-2"}
    assert row_to_entity_calls == 3


def test_embedded_store_current_traversal_index_uses_hot_path_query() -> None:
    """Current traversal warmup should not carry temporal predicates into the hot query."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    connection = _CountingConnection()
    store._connection = connection

    store._traversal_index("agent-1")

    query, params = connection.queries[-1]
    assert "$temporal_point" not in query
    assert params == {"session_id": "agent-1"}


def test_keyword_query_terms_drop_high_frequency_question_glue() -> None:
    """Keyword search should not expand candidates through low-signal glue terms."""
    assert _keyword_query_terms(
        "How many days did it take for me to find a house I loved after starting to work with Rachel?"
    ) == [
        "many",
        "days",
        "take",
        "find",
        "house",
        "loved",
        "after",
        "starting",
        "work",
        "rachel",
    ]
    assert _keyword_query_terms("Which seeds were started first, the tomatoes or the marigolds?") == [
        "seeds",
        "started",
        "tomatoes",
        "or",
        "marigolds",
    ]
    assert _keyword_query_terms(
        "Which event happened first, losing the phone charger or receiving the new phone case?"
    ) == [
        "event",
        "happened",
        "losing",
        "phone",
        "charger",
        "or",
        "receiving",
        "new",
        "phone",
        "case",
    ]
    assert _keyword_query_terms("How many charity events did I participate in before the Run for the Cure event?") == [
        "many",
        "charity",
        "events",
        "participate",
        "before",
        "run",
        "cure",
        "event",
    ]


def test_embedded_keyword_tokenizer_uses_compiled_regex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedded keyword indexing should not compile regex strings per query/entity."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("embedded keyword tokenizer should use compiled regex helpers")

    monkeypatch.setattr("zaxy.embedded_graph_store.re.findall", fail)

    assert _terms("Embedded Kuzu recall: graph-goal-0003!") == [
        "embedded",
        "kuzu",
        "recall",
        "graph",
        "goal",
        "0003",
    ]


def test_keyword_query_terms_uses_module_stopword_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedded keyword queries should not rebuild stopwords on every call."""
    monkeypatch.setattr(
        embedded_graph_store,
        "_KEYWORD_STOP_WORDS",
        frozenset({"many", "days"}),
        raising=False,
    )

    assert _keyword_query_terms("How many days until embedded recall?") == [
        "how",
        "until",
        "embedded",
        "recall",
    ]


def test_keyword_candidate_terms_prefer_rare_terms_for_broad_queries() -> None:
    """Broad keyword queries should seed candidates from rare terms before common ones."""
    entities = [
        *[
            GraphEntity(
                name=f"common-{index}",
                entity_type="document",
                valid_from="2026-05-20T01:00:00Z",
                valid_to=None,
                properties={"summary": "been up local open"},
            )
            for index in range(40)
        ],
        GraphEntity(
            name="target",
            entity_type="document",
            valid_from="2026-05-20T01:00:00Z",
            valid_to=None,
            properties={"summary": "comedy specials mic club"},
        ),
    ]
    index = _keyword_index_from_entities(entities)

    assert _keyword_candidate_terms(
        index,
        ["been", "up", "local", "open", "comedy", "specials", "mic", "club"],
        max_candidates=10,
        min_terms=4,
    ) == ["comedy", "specials", "mic", "club"]


def test_keyword_candidate_terms_avoid_postings_set_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broad keyword candidate narrowing should avoid per-term postings set copies."""
    index = embedded_graph_store._KeywordIndex(
        entities=[],
        term_counts=[],
        term_entity_ids={
            "common": tuple(range(20)),
            "rare": (21,),
            "focused": (22,),
            "target": (23,),
        },
        term_idf={"common": 0.1, "rare": 2.0, "focused": 1.9, "target": 1.8},
        document_length_norms=[],
    )
    monkeypatch.setattr(
        builtins,
        "set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate narrowing should not allocate postings sets")
        ),
    )

    assert _keyword_candidate_terms(
        index,
        ["common", "rare", "focused", "target"],
        max_candidates=3,
        min_terms=2,
    ) == ["rare", "focused", "target"]


def test_keyword_candidate_terms_stops_counting_after_candidate_cap() -> None:
    """Broad keyword candidate checks should not exhaust huge postings lists."""

    class _LargePostings:
        def __len__(self) -> int:
            return 100

        def __iter__(self):
            for index in range(100):
                if index > 4:
                    raise AssertionError("candidate cap check should stop after proving overflow")
                yield index

    index = embedded_graph_store._KeywordIndex(
        entities=[],
        term_counts=[],
        term_entity_ids={
            "common": _LargePostings(),  # type: ignore[dict-item]
            "rare": (101,),
            "focused": (102,),
        },
        term_idf={"common": 0.1, "rare": 2.0, "focused": 1.9},
        document_length_norms=[],
    )

    assert _keyword_candidate_terms(
        index,
        ["common", "rare", "focused"],
        max_candidates=3,
        min_terms=2,
    ) == ["rare", "focused"]


def test_keyword_index_build_consumes_entity_terms_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedded keyword index construction should avoid repeated token-stream passes."""

    class _SinglePassTerms:
        def __init__(self, terms: list[str]) -> None:
            self._terms = terms
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("keyword index construction should consume entity terms once")
            return iter(self._terms)

        def __len__(self) -> int:
            return len(self._terms)

    streams = [
        _SinglePassTerms(["alpha", "alpha", "beta"]),
        _SinglePassTerms(["beta", "gamma"]),
    ]

    def fake_terms(_text: str) -> _SinglePassTerms:
        return streams.pop(0)

    monkeypatch.setattr(embedded_graph_store, "_terms", fake_terms)
    entities = [
        GraphEntity(
            name="first",
            entity_type="document",
            valid_from="2026-05-20T01:00:00Z",
            valid_to=None,
            properties={},
        ),
        GraphEntity(
            name="second",
            entity_type="document",
            valid_from="2026-05-20T01:00:00Z",
            valid_to=None,
            properties={},
        ),
    ]

    index = _keyword_index_from_entities(entities)

    assert index.term_counts[0]["alpha"] == 2
    assert index.term_entity_ids["beta"] == (0, 1)
    assert index.term_idf["alpha"] > 0


@pytest.mark.asyncio
async def test_embedded_store_projects_entities_and_traversal_edges(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Goal A",
                entity_type="goal",
                observed_at="2026-05-20T01:00:00Z",
                summary="ship memory",
            ),
            ExtractedEntity(
                name="Task B",
                entity_type="task",
                observed_at="2026-05-20T01:00:00Z",
                summary="write tests",
            ),
        ],
        edges=[
            ExtractedEdge(
                source="Task B",
                target="Goal A",
                relation_type="supports",
                valid_from="2026-05-20T01:00:00Z",
            )
        ],
        source_event_seq=1,
        source_event_hash="abc",
        source_event_type="task.proposed",
        source_thread="agent-1",
    )

    await store.upsert_extraction(result, session_id="agent-1")

    exact = await store.search_exact("Goal A", session_id="agent-1")
    keyword = await store.search_keyword("ship", session_id="agent-1")
    neighbors = await store.search_traversal("Task B", depth=1, session_id="agent-1")

    assert [entity.name for entity in exact] == ["Goal A"]
    assert exact[0].properties["source_event_seq"] == 1
    assert exact[0].properties["source_event_hash"] == "abc"
    assert [result.entity.name for result in keyword] == ["Goal A"]
    assert keyword[0].entity.properties["source_event_seq"] == 1
    assert [entity.name for entity in neighbors] == ["Goal A"]
    assert neighbors[0].properties["source_event_hash"] == "abc"
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_warm_session_populates_read_indexes(tmp_path: Path) -> None:
    """Session warmup should populate read indexes before the first checkout."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Warm Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="warm indexed memory",
                    embedding=[1.0, 0.0],
                ),
                ExtractedEntity(
                    name="Warm Task",
                    entity_type="task",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="linked warm memory",
                    embedding=[0.0, 1.0],
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Warm Task",
                    target="Warm Goal",
                    relation_type="supports",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="warm-hash",
        ),
        session_id="agent-1",
    )

    await store.warm_session(session_id="agent-1")

    assert "agent-1" in store._current_entity_index_cache
    assert "agent-1" in store._keyword_index_cache
    assert ("agent-1", None) in store._vector_index_cache
    assert "agent-1" in store._traversal_index_cache
    assert store._traversal_index_cache["agent-1"].adjacency
    await store.close()


@pytest.mark.asyncio
async def test_upsert_extraction_preserves_read_caches_for_event_only_projection(tmp_path: Path) -> None:
    """Event-only lifecycle projection should preserve entity/search read caches."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
            source_event_prev_hash=None,
        ),
        session_id="agent-1",
    )
    store._current_entity_index_cache["agent-1"] = []
    store._current_entity_lookup_cache["agent-1"] = {}
    store._keyword_index_cache["agent-1"] = embedded_graph_store._KeywordIndex([], [], {}, {}, [])
    store._vector_index_cache[("agent-1", None)] = embedded_graph_store._VectorIndex([], {})
    store._traversal_index_cache["agent-1"] = embedded_graph_store._TraversalIndex({}, {})

    await store.upsert_extraction(
        ExtractionResult(
            entities=[],
            edges=[],
            source_event_seq=2,
            source_event_hash="hash-2",
            source_event_prev_hash="hash-1",
        ),
        session_id="agent-1",
    )

    assert "agent-1" in store._current_entity_index_cache
    assert "agent-1" in store._current_entity_lookup_cache
    assert "agent-1" in store._keyword_index_cache
    assert ("agent-1", None) in store._vector_index_cache
    assert "agent-1" in store._traversal_index_cache
    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=2,
        eventloom_latest_hash="hash-2",
    )
    assert status.event_count == 2
    assert status.next_event_edges == 1
    assert status.previous_event_edges == 1
    await store.close()


@pytest.mark.asyncio
async def test_upsert_extraction_preserves_read_caches_for_noop_entity_projection(tmp_path: Path) -> None:
    """Reasserting an unchanged entity should project the Event without cache churn."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    entity = ExtractedEntity(
        name="Stable Goal",
        entity_type="goal",
        observed_at="2026-05-20T01:00:00Z",
        summary="unchanged",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[entity],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
            source_event_prev_hash=None,
        ),
        session_id="agent-1",
    )
    await store.warm_session(session_id="agent-1")

    await store.upsert_extraction(
        ExtractionResult(
            entities=[entity],
            edges=[],
            source_event_seq=2,
            source_event_hash="hash-2",
            source_event_prev_hash="hash-1",
        ),
        session_id="agent-1",
    )

    assert "agent-1" in store._current_entity_index_cache
    assert "agent-1" in store._current_entity_lookup_cache
    assert "agent-1" in store._keyword_index_cache
    assert ("agent-1", None) in store._vector_index_cache
    assert "agent-1" in store._traversal_index_cache
    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=2,
        eventloom_latest_hash="hash-2",
    )
    assert status.event_count == 2
    assert status.next_event_edges == 1
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_has_traversal_edges_uses_warmed_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Traversal availability should not issue a DB count after warmup."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Warm Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="warm indexed memory",
                ),
                ExtractedEntity(
                    name="Warm Task",
                    entity_type="task",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="linked warm memory",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Warm Task",
                    target="Warm Goal",
                    relation_type="supports",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="warm-hash",
        ),
        session_id="agent-1",
    )
    await store.warm_session(session_id="agent-1")

    def fail_connection() -> object:
        raise AssertionError("warmed traversal availability should not query Kuzu")

    monkeypatch.setattr(store, "_require_connection", fail_connection)

    assert await store.has_traversal_edges(session_id="agent-1") is True
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_event_status_handles_missing_projection_schema(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=3,
        eventloom_latest_hash="hash-3",
    )

    assert status.session_id == "agent-1"
    assert status.event_count == 0
    assert status.latest_seq is None
    assert status.eventloom_latest_seq == 3
    assert status.eventloom_latest_hash == "hash-3"
    assert status.projection_lag == 3
    assert status.latest_hash_matches is False
    assert status.integrity_ok is False
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_event_status_requires_projected_chain_edges() -> None:
    """Embedded projection integrity should fail when Event chain edges are missing."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    event_rows = [
        [1, "hash-1", None],
        [2, "hash-2", "hash-1"],
    ]

    class _EventStatusConnection:
        def execute(self, query: str, params: dict[str, object] | None = None) -> _FakeRows:
            assert params == {"session_id": "agent-1"}
            if "MATCH (e:Event)" in query:
                return _FakeRows(event_rows)
            if "NEXT_EVENT" in query:
                return _FakeRows([[0]])
            if "PREVIOUS_EVENT" in query:
                return _FakeRows([[0]])
            raise AssertionError(query)

    store._connection = _EventStatusConnection()

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=2,
        eventloom_latest_hash="hash-2",
    )

    assert status.event_count == 2
    assert status.missing_chain_links == 0
    assert status.next_event_edges == 0
    assert status.previous_event_edges == 0
    assert status.integrity_ok is False


@pytest.mark.asyncio
async def test_embedded_store_inferred_status_handles_missing_projection_schema(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()

    status = await store.inspect_inferred_edge_status("agent-1")

    assert status.session_id == "agent-1"
    assert status.total_edges == 0
    assert status.method_count == 0
    assert status.evidence_count == 0
    assert status.missing_evidence_count == 0
    assert status.missing_source_event_count == 0
    assert status.evidence_coverage == 1.0
    assert status.methods == ()
    assert status.samples == ()
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_keyword_search_prioritizes_rare_phrase_matches(tmp_path: Path) -> None:
    """Keyword search should not let common question words bury answer-bearing phrases."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="common-words",
                    entity_type="document",
                    observed_at="2026-05-20T01:00:00Z",
                    summary=(
                        "Which group did I join first was the question I kept asking "
                        "while comparing several professional communities."
                    ),
                ),
                ExtractedEntity(
                    name="answer-bearing",
                    entity_type="document",
                    observed_at="2026-05-20T01:01:00Z",
                    summary="I joined the Page Turners book club before the other networking group.",
                ),
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    results = await store.search_keyword(
        "Which group did I join first, Page Turners or Marketing Professionals?",
        limit=2,
        session_id="agent-1",
    )

    assert [result.entity.name for result in results] == ["answer-bearing", "common-words"]
    index = store._keyword_index_cache["agent-1"]
    assert len(index.term_counts) == 2
    assert not hasattr(index, "terms")
    assert not hasattr(index, "document_lengths")
    assert not hasattr(index, "document_frequency")
    assert not hasattr(index, "average_length")
    assert len(index.document_length_norms) == 2
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_keyword_search_scores_only_matching_entities(tmp_path: Path, monkeypatch) -> None:
    """Keyword BM25 should skip current entities that cannot match the query."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                *[
                    ExtractedEntity(
                        name=f"background-{index}",
                        entity_type="document",
                        observed_at="2026-05-20T01:00:00Z",
                        summary=f"Background note {index} about unrelated planning.",
                    )
                    for index in range(20)
                ],
                ExtractedEntity(
                    name="target",
                    entity_type="document",
                    observed_at="2026-05-20T01:01:00Z",
                    summary="The embedded keyword marker is precise-needle-42.",
                ),
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    original_score = embedded_graph_store._bm25_score_from_precomputed
    score_calls = 0

    def tracking_score(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        return original_score(*args, **kwargs)

    monkeypatch.setattr(embedded_graph_store, "_bm25_score_from_precomputed", tracking_score)

    results = await store.search_keyword("precise needle 42", limit=1, session_id="agent-1")

    assert [result.entity.name for result in results] == ["target"]
    assert score_calls == 1
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_keyword_search_zero_limit_skips_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-result keyword queries should not build a read index."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    def fail_keyword_index(session_id: str, temporal_point: str | None = None) -> object:
        raise AssertionError(f"zero-limit keyword query should not build index for {session_id}:{temporal_point}")

    monkeypatch.setattr(store, "_keyword_index", fail_keyword_index)

    assert await store.search_keyword("embedded memory", limit=0, session_id="agent-1") == []


@pytest.mark.asyncio
async def test_embedded_store_keyword_search_converts_score_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keyword results should compute the float score once and reuse it as raw_score."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    entity = GraphEntity(
        name="Keyword Goal",
        entity_type="goal",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )
    index = embedded_graph_store._KeywordIndex(
        entities=[entity],
        term_counts=[Counter({"needle": 1})],
        term_entity_ids={"needle": (0,)},
        term_idf={"needle": 1.0},
        document_length_norms=[1.0],
    )

    class _CountingScore:
        float_calls = 0

        def __bool__(self) -> bool:
            return True

        def __float__(self) -> float:
            self.float_calls += 1
            return 1.0

    score = _CountingScore()

    monkeypatch.setattr(store, "_keyword_index", lambda session_id, temporal_point=None: index)
    monkeypatch.setattr(embedded_graph_store, "_bm25_score_from_precomputed", lambda *args, **kwargs: score)

    results = await store.search_keyword("needle", session_id="agent-1")

    assert [result.entity.name for result in results] == ["Keyword Goal"]
    assert results[0].score == 1.0
    assert results[0].raw_score == 1.0
    assert score.float_calls == 1


@pytest.mark.asyncio
async def test_embedded_store_keyword_search_scores_only_terms_present_in_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Candidate scoring should not scan query terms absent from that entity."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="alpha-node",
                    entity_type="document",
                    observed_at="2026-05-20T01:01:00Z",
                    summary="alpha topic only",
                ),
                ExtractedEntity(
                    name="beta-node",
                    entity_type="document",
                    observed_at="2026-05-20T01:02:00Z",
                    summary="beta topic only",
                ),
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    original_score = embedded_graph_store._bm25_score_from_precomputed
    scored_terms: list[tuple[str, ...]] = []

    def tracking_score(query_terms, *args, **kwargs):
        scored_terms.append(tuple(query_terms))
        return original_score(query_terms, *args, **kwargs)

    monkeypatch.setattr(embedded_graph_store, "_bm25_score_from_precomputed", tracking_score)

    results = await store.search_keyword("alpha beta missing", limit=2, session_id="agent-1")

    assert {result.entity.name for result in results} == {"alpha-node", "beta-node"}
    assert sorted(scored_terms) == [("alpha",), ("beta",)]
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_keyword_search_reuses_candidate_matched_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keyword search should not rescan all query terms for every candidate."""

    class _NoAbsentLookupCounter(Counter[str]):
        def __getitem__(self, key: str) -> int:
            if key not in self:
                raise AssertionError(f"candidate scoring should not probe absent term {key}")
            return super().__getitem__(key)

    store = EmbeddedGraphStore(Path("unused.kuzu"))
    alpha = GraphEntity(
        name="alpha-node",
        entity_type="document",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )
    beta = GraphEntity(
        name="beta-node",
        entity_type="document",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )
    index = embedded_graph_store._KeywordIndex(
        entities=[alpha, beta],
        term_counts=[
            _NoAbsentLookupCounter({"alpha": 1}),
            _NoAbsentLookupCounter({"beta": 1}),
        ],
        term_entity_ids={"alpha": (0,), "beta": (1,)},
        term_idf={"alpha": 1.0, "beta": 1.0},
        document_length_norms=[1.0, 1.0],
    )
    scored_terms: list[tuple[str, ...]] = []

    def tracking_score(query_terms, *args, **kwargs):
        scored_terms.append(tuple(query_terms))
        return 1.0

    monkeypatch.setattr(store, "_keyword_index", lambda session_id, temporal_point=None: index)
    monkeypatch.setattr(embedded_graph_store, "_keyword_query_terms", lambda query: ["alpha", "beta", "missing"])
    monkeypatch.setattr(embedded_graph_store, "_bm25_score_from_precomputed", tracking_score)

    results = await store.search_keyword("alpha beta missing", limit=2, session_id="agent-1")

    assert {result.entity.name for result in results} == {"alpha-node", "beta-node"}
    assert sorted(scored_terms) == [("alpha",), ("beta",)]


@pytest.mark.asyncio
async def test_embedded_store_keyword_search_uses_precomputed_bm25_statistics(
    tmp_path: Path,
) -> None:
    """Runtime keyword queries should not retain the old per-document BM25 scorer."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="target",
                    entity_type="document",
                    observed_at="2026-05-20T01:01:00Z",
                    summary="The embedded keyword marker is reusable-needle-7.",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    results = await store.search_keyword("reusable needle 7", limit=1, session_id="agent-1")

    assert [result.entity.name for result in results] == ["target"]
    assert not hasattr(embedded_graph_store, "_bm25_score_from_counts")
    assert not hasattr(embedded_graph_store, "_bm25_score")
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_exact_search_uses_current_entity_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact active lookups should avoid a Kuzu query per benchmark question."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Cached Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="cached exact lookup",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    exact = await store.search_exact("Cached Goal", session_id="agent-1")

    assert [entity.name for entity in exact] == ["Cached Goal"]
    assert "agent-1" in store._current_entity_index_cache
    assert ("Cached Goal", None) in store._current_entity_lookup_cache["agent-1"]

    def fail_current_entities(session_id: str) -> list[object]:
        raise AssertionError(f"exact lookup should use warmed lookup index for {session_id}")

    monkeypatch.setattr(store, "_current_entities", fail_current_entities)
    second = await store.search_exact("Cached Goal", session_id="agent-1")
    assert [entity.name for entity in second] == ["Cached Goal"]

    monkeypatch.undo()
    await store.invalidate_entity(
        "Cached Goal",
        "goal",
        "2026-05-20T02:00:00Z",
        session_id="agent-1",
    )

    assert "agent-1" not in store._current_entity_index_cache
    assert await store.search_exact("Cached Goal", session_id="agent-1") == []
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_temporal_exact_search_reuses_temporal_entities_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temporal exact lookups should share the temporal entity projection path."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()
    temporal_entities = [
        GraphEntity(
            name="Temporal Goal",
            entity_type="goal",
            valid_from="2026-05-20T01:00:00Z",
            valid_to="2026-05-20T02:00:00Z",
            properties={"summary": "historical exact match"},
            session_id="agent-1",
        ),
        GraphEntity(
            name="Other Temporal Goal",
            entity_type="goal",
            valid_from="2026-05-20T01:00:00Z",
            valid_to=None,
            properties={"summary": "not the exact match"},
            session_id="agent-1",
        ),
    ]

    def temporal_entities_helper(session_id: str, temporal_point: str) -> list[GraphEntity]:
        assert session_id == "agent-1"
        assert temporal_point == "2026-05-20T01:30:00Z"
        return temporal_entities

    def fail_connection() -> object:
        raise AssertionError("temporal exact lookup should reuse _temporal_entities")

    monkeypatch.setattr(store, "_temporal_entities", temporal_entities_helper)
    monkeypatch.setattr(store, "_require_connection", fail_connection)

    exact = await store.search_exact(
        "Temporal Goal",
        entity_type="goal",
        temporal_point="2026-05-20T01:30:00Z",
        session_id="agent-1",
    )

    assert exact == [temporal_entities[0]]
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_traversal_matches_neo4j_undirected_paths(tmp_path: Path) -> None:
    """Traversal should cross incoming and outgoing edges like the Neo4j control backend."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Graph Goal", entity_type="goal", observed_at="2026-05-20T01:00:00Z"),
                ExtractedEntity(name="Graph Task", entity_type="task", observed_at="2026-05-20T01:00:00Z"),
                ExtractedEntity(name="Finisher", entity_type="actor", observed_at="2026-05-20T01:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Graph Goal",
                    target="Graph Task",
                    relation_type="has_task",
                    valid_from="2026-05-20T01:00:00Z",
                ),
                ExtractedEdge(
                    source="Finisher",
                    target="Graph Task",
                    relation_type="completed_task",
                    valid_from="2026-05-20T01:00:00Z",
                ),
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    neighbors = await store.search_traversal("Graph Goal", depth=2, session_id="agent-1")

    assert {entity.name for entity in neighbors} == {"Graph Task", "Finisher"}
    finisher = next(entity for entity in neighbors if entity.name == "Finisher")
    assert finisher.properties["_path_relation_types"] == ["has_task", "completed_task"]
    assert finisher.properties["_path_length"] == 2
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_caches_traversal_index_and_invalidates_on_projection(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Root", entity_type="goal", observed_at="2026-05-20T01:00:00Z"),
                ExtractedEntity(name="First", entity_type="task", observed_at="2026-05-20T01:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Root",
                    target="First",
                    relation_type="has_task",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    first = await store.search_traversal("Root", depth=1, session_id="agent-1")
    assert [entity.name for entity in first] == ["First"]
    assert "agent-1" in store._traversal_index_cache

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Root", entity_type="goal", observed_at="2026-05-20T01:01:00Z"),
                ExtractedEntity(name="Second", entity_type="task", observed_at="2026-05-20T01:01:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Root",
                    target="Second",
                    relation_type="has_task",
                    valid_from="2026-05-20T01:01:00Z",
                )
            ],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )
    assert "agent-1" not in store._traversal_index_cache

    second = await store.search_traversal("Root", depth=1, session_id="agent-1")
    assert {entity.name for entity in second} == {"First", "Second"}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_preserves_active_relationships_across_entity_reassertion(tmp_path: Path) -> None:
    """Reasserting an entity should not sever active graph paths from prior events."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Graph Goal", entity_type="goal", observed_at="2026-05-20T01:00:00Z"),
                ExtractedEntity(name="Graph Task", entity_type="task", observed_at="2026-05-20T01:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Graph Goal",
                    target="Graph Task",
                    relation_type="has_task",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Graph Task", entity_type="task", observed_at="2026-05-20T02:00:00Z"),
                ExtractedEntity(name="Finisher", entity_type="actor", observed_at="2026-05-20T02:00:00Z"),
            ],
            edges=[
                ExtractedEdge(
                    source="Finisher",
                    target="Graph Task",
                    relation_type="completed_task",
                    valid_from="2026-05-20T02:00:00Z",
                )
            ],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )

    neighbors = await store.search_traversal("Graph Goal", depth=2, session_id="agent-1")

    assert {entity.name for entity in neighbors} == {"Graph Task", "Finisher"}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reuses_identical_active_entity_without_reversioning(tmp_path: Path) -> None:
    """Stable helper nodes should not be re-versioned for every projected document chunk."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    session = ExtractedEntity(
        name="longmemeval:session:s1",
        entity_type="longmemeval_session",
        observed_at="2024-01-01T00:00:00Z",
        summary="2024/01/01",
        properties={"longmemeval_session_id": "s1"},
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                session,
                ExtractedEntity(
                    name="chunk-1",
                    entity_type="document",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="first chunk",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source=session.name,
                    target="chunk-1",
                    relation_type="has_document_chunk",
                    valid_from="2024-01-01T00:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                session,
                ExtractedEntity(
                    name="chunk-2",
                    entity_type="document",
                    observed_at="2024-01-01T00:01:00Z",
                    summary="second chunk",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source=session.name,
                    target="chunk-2",
                    relation_type="has_document_chunk",
                    valid_from="2024-01-01T00:01:00Z",
                )
            ],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )

    rows = store._require_connection().execute(
        """
        MATCH (e:Entity)
        WHERE e.session_id = 'agent-1'
          AND e.name = 'longmemeval:session:s1'
          AND e.entity_type = 'longmemeval_session'
        RETURN count(e)
        """
    ).get_all()
    neighbors = await store.search_traversal("longmemeval:session:s1", depth=1, session_id="agent-1")

    assert rows[0][0] == 1
    assert {neighbor.name for neighbor in neighbors} == {"chunk-1", "chunk-2"}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_bulk_projection_transaction_commits_events(tmp_path: Path) -> None:
    """Bulk projection should group Kuzu writes without changing committed graph state."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.begin_bulk_projection()
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Bulk Goal",
                    entity_type="goal",
                    observed_at="2024-01-01T00:00:00Z",
                    summary="bulk projection",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    await store.commit_bulk_projection()

    assert ("Bulk Goal", None) in store._current_entity_lookup_cache["agent-1"]
    exact = await store.search_exact("Bulk Goal", session_id="agent-1")

    assert [entity.name for entity in exact] == ["Bulk Goal"]
    assert "agent-1" in store._keyword_index_cache
    assert ("agent-1", None) in store._vector_index_cache
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_versions_reasserted_entities_and_invalidates(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Policy",
                    entity_type="decision",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="legacyuniquetoken",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Policy",
                    entity_type="decision",
                    observed_at="2026-05-20T02:00:00Z",
                    summary="modernuniquetoken",
                )
            ],
            edges=[],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )

    current = await store.search_exact("Policy", entity_type="decision", session_id="agent-1")
    assert len(current) == 1
    assert current[0].valid_from == "2026-05-20T02:00:00Z"
    assert current[0].properties["summary"] == "modernuniquetoken"

    previous = await store.search_exact(
        "Policy",
        entity_type="decision",
        temporal_point="2026-05-20T01:30:00Z",
        session_id="agent-1",
    )
    assert len(previous) == 1
    assert previous[0].valid_from == "2026-05-20T01:00:00Z"
    assert previous[0].valid_to == "2026-05-20T02:00:00Z"
    assert previous[0].properties["summary"] == "legacyuniquetoken"

    historical_keyword = await store.search_keyword(
        "legacyuniquetoken",
        temporal_point="2026-05-20T01:30:00Z",
        session_id="agent-1",
    )
    assert [result.entity.properties["summary"] for result in historical_keyword] == ["legacyuniquetoken"]

    current_keyword = await store.search_keyword(
        "legacyuniquetoken",
        session_id="agent-1",
    )
    assert current_keyword == []

    await store.invalidate_entity(
        "Policy",
        "decision",
        invalid_at="2026-05-20T03:00:00Z",
        session_id="agent-1",
    )

    assert await store.search_exact("Policy", entity_type="decision", session_id="agent-1") == []
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reports_event_projection_status(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Goal A",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
            source_event_prev_hash=None,
        ),
        session_id="agent-1",
    )
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Goal B",
                    entity_type="goal",
                    observed_at="2026-05-20T02:00:00Z",
                )
            ],
            edges=[],
            source_event_seq=2,
            source_event_hash="hash-2",
            source_event_prev_hash="hash-1",
        ),
        session_id="agent-1",
    )

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=2,
        eventloom_latest_hash="hash-2",
    )

    assert status.event_count == 2
    assert status.latest_seq == 2
    assert status.latest_hash == "hash-2"
    assert status.projection_lag == 0
    assert status.latest_hash_matches is True
    assert status.integrity_ok is True
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reports_missing_schema_as_empty_projection(tmp_path: Path) -> None:
    """Read-only status should not crash before the embedded projection is initialized."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()

    status = await store.inspect_event_projection_status(
        "agent-1",
        eventloom_latest_seq=2,
        eventloom_latest_hash="hash-2",
    )

    assert status.event_count == 0
    assert status.latest_seq is None
    assert status.latest_hash is None
    assert status.projection_lag == 2
    assert status.latest_hash_matches is False
    assert status.integrity_ok is False
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_searches_entity_vectors(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version=LEGACY_EMBEDDING_VERSION,
    )
    await store.connect()
    await store.init_schema()

    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Vector Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="closest vector",
                    embedding=vector_a,
                ),
                ExtractedEntity(
                    name="Distant Task",
                    entity_type="task",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="far vector",
                    embedding=vector_b,
                ),
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    results = await store.search_vector(vector_a, limit=2, session_id="agent-1")

    assert [result.entity.name for result in results] == ["Vector Goal"]
    assert results[0].source == "vector"
    assert results[0].score > 0
    assert results[0].entity.properties["source_event_hash"] == "hash-1"
    assert results[0].exact is True
    index = store._vector_index_cache[("agent-1", None)]
    group = index.groups[(2, LEGACY_EMBEDDING_VERSION)]
    assert isinstance(group, embedded_graph_store._VectorGroup)
    assert group.entity_indexes == [0, 1]
    assert group.matrix.tolist() == [[1.0, 0.0], [0.0, 1.0]]

    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_zero_vector_search_skips_index_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-norm vector queries cannot match and should avoid projection index work."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    def fail_vector_index(session_id: str, temporal_point: str | None) -> object:
        raise AssertionError(f"zero vector query should not build index for {session_id}:{temporal_point}")

    monkeypatch.setattr(store, "_vector_index", fail_vector_index)

    assert await store.search_vector([0.0, 0.0], session_id="agent-1") == []


def test_embedded_vector_index_groups_store_unit_vectors() -> None:
    """The vector index must store unit-normalized rows grouped by dimension."""
    entity = GraphEntity(
        name="Vector Goal",
        entity_type="goal",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={"embedding": [3.0, 4.0]},
        session_id="agent-1",
    )
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    store._current_entity_index_cache["agent-1"] = [entity]

    index = store._vector_index("agent-1", None)

    group = index.groups[(2, LEGACY_EMBEDDING_VERSION)]
    assert isinstance(group, embedded_graph_store._VectorGroup)
    assert group.matrix.tolist() == [[0.6, 0.8]]
    assert group.entity_indexes == [0]
    assert entity.properties["embedding_dimension"] == 2


def _vector_entity(name: str, embedding: list[float]) -> GraphEntity:
    return GraphEntity(
        name=name,
        entity_type="goal",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={"embedding": embedding},
        session_id="agent-1",
    )


def test_embedded_vector_index_cache_evicts_lru_beyond_entry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vector index cache must cap how many session/temporal variants it holds."""
    monkeypatch.setattr(embedded_graph_store, "VECTOR_INDEX_CACHE_MAX_ENTRIES", 2)
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    for session in ("agent-1", "agent-2", "agent-3"):
        store._current_entity_index_cache[session] = [_vector_entity(session, [1.0, 0.0])]
        store._vector_index(session, None)

    assert list(store._vector_index_cache) == [("agent-2", None), ("agent-3", None)]


def test_embedded_vector_index_cache_evicts_lru_beyond_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding matrices beyond the byte budget must evict oldest indexes first."""
    # Each 2-dim float64 unit vector is 16 bytes; budget of 24 holds only one.
    monkeypatch.setattr(embedded_graph_store, "VECTOR_INDEX_CACHE_MAX_BYTES", 24)
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    for session in ("agent-1", "agent-2"):
        store._current_entity_index_cache[session] = [_vector_entity(session, [1.0, 0.0])]
        store._vector_index(session, None)

    assert list(store._vector_index_cache) == [("agent-2", None)]


def test_embedded_vector_index_cache_hit_refreshes_lru_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache hit must protect that index from the next eviction pass."""
    monkeypatch.setattr(embedded_graph_store, "VECTOR_INDEX_CACHE_MAX_ENTRIES", 2)
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    for session in ("agent-1", "agent-2"):
        store._current_entity_index_cache[session] = [_vector_entity(session, [1.0, 0.0])]
        store._vector_index(session, None)

    store._vector_index("agent-1", None)  # refresh agent-1 recency
    store._current_entity_index_cache["agent-3"] = [_vector_entity("agent-3", [1.0, 0.0])]
    store._vector_index("agent-3", None)

    assert list(store._vector_index_cache) == [("agent-1", None), ("agent-3", None)]


def test_properties_reference_source_uses_loop_hot_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source retirement checks should avoid generator allocation per projected row."""
    monkeypatch.setattr(
        builtins,
        "any",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source path checks should not allocate any() generators")
        ),
    )

    assert _properties_reference_source({"source_path": "docs/guide.md"}, "docs/guide.md") is True
    assert _properties_reference_source({"target_path": "docs/guide.md"}, "docs/guide.md") is True
    assert _properties_reference_source({"source_path": "docs/other.md"}, "docs/guide.md") is False


@pytest.mark.asyncio
async def test_embedded_store_vector_search_zero_limit_skips_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-result vector queries should not build a vector index."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    def fail_vector_index(session_id: str, temporal_point: str | None) -> object:
        raise AssertionError(f"zero-limit vector query should not build index for {session_id}:{temporal_point}")

    monkeypatch.setattr(store, "_vector_index", fail_vector_index)

    assert await store.search_vector([1.0, 0.0], limit=0, session_id="agent-1") == []


@pytest.mark.asyncio
async def test_embedded_store_vector_search_computes_candidate_score_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vector candidates should score via one matrix product and reuse it as raw_score."""
    store = EmbeddedGraphStore(
        Path("unused.kuzu"),
        active_embedding_version=LEGACY_EMBEDDING_VERSION,
    )

    entity = GraphEntity(
        name="Vector Goal",
        entity_type="goal",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties={},
        session_id="agent-1",
    )

    def vector_index(session_id: str, temporal_point: str | None) -> embedded_graph_store._VectorIndex:
        assert session_id == "agent-1"
        assert temporal_point is None
        return embedded_graph_store._VectorIndex(
            entities=[entity],
            groups={
                (1, LEGACY_EMBEDDING_VERSION): embedded_graph_store._VectorGroup(
                    matrix=np.array([[1.0]]),
                    entity_indexes=[0],
                )
            },
        )

    monkeypatch.setattr(store, "_vector_index", vector_index)

    results = await store.search_vector([2.0], session_id="agent-1")

    assert [result.entity.name for result in results] == ["Vector Goal"]
    assert results[0].score == 1.0
    assert results[0].raw_score == results[0].score


@pytest.mark.asyncio
async def test_embedded_store_caches_vector_index_and_invalidates_on_projection(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version=LEGACY_EMBEDDING_VERSION,
    )
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Original Vector",
                    entity_type="memory",
                    observed_at="2026-05-20T01:00:00Z",
                    embedding=[1.0, 0.0],
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    first = await store.search_vector([1.0, 0.0], limit=1, session_id="agent-1")
    assert [result.entity.name for result in first] == ["Original Vector"]
    assert ("agent-1", None) in store._vector_index_cache
    assert not hasattr(store._vector_index_cache[("agent-1", None)], "sparse_vectors")
    cached_group = store._vector_index_cache[("agent-1", None)].groups[(2, LEGACY_EMBEDDING_VERSION)]
    assert isinstance(cached_group, embedded_graph_store._VectorGroup)
    assert cached_group.matrix.tolist() == [[1.0, 0.0]]
    assert cached_group.entity_indexes == [0]

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="New Vector",
                    entity_type="memory",
                    observed_at="2026-05-20T01:01:00Z",
                    embedding=[1.0, 0.0],
                )
            ],
            edges=[],
            source_event_seq=2,
            source_event_hash="hash-2",
        ),
        session_id="agent-1",
    )
    assert ("agent-1", None) not in store._vector_index_cache

    second = await store.search_vector([1.0, 0.0], limit=2, session_id="agent-1")
    assert {result.entity.name for result in second} == {"Original Vector", "New Vector"}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_current_vector_index_reuses_warmed_entities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Current vector index construction should not duplicate warmed entity rows."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Reusable Vector",
                    entity_type="memory",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="reused current entity",
                    embedding=[1.0, 0.0],
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )
    warmed_entities = store._current_entities("agent-1")

    def fail_connection() -> object:
        raise AssertionError("current vector index should reuse warmed current entities")

    monkeypatch.setattr(store, "_require_connection", fail_connection)

    index = store._vector_index("agent-1", None)

    assert index.entities[0] is warmed_entities[0]
    assert not hasattr(index, "sparse_vectors")
    group = index.groups[(2, LEGACY_EMBEDDING_VERSION)]
    assert isinstance(group, embedded_graph_store._VectorGroup)
    assert group.matrix.tolist() == [[1.0, 0.0]]
    assert group.entity_indexes == [0]
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_temporal_vector_index_reuses_temporal_entities_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temporal vector indexing should share the temporal entity projection path."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()
    temporal_entities = [
        GraphEntity(
            name="Temporal Vector",
            entity_type="memory",
            valid_from="2026-05-20T01:00:00Z",
            valid_to=None,
            properties={"embedding": [0.0, 1.0]},
            session_id="agent-1",
        )
    ]

    def temporal_entities_helper(session_id: str, temporal_point: str) -> list[GraphEntity]:
        assert session_id == "agent-1"
        assert temporal_point == "2026-05-20T01:01:00Z"
        return temporal_entities

    def fail_connection() -> object:
        raise AssertionError("temporal vector index should reuse _temporal_entities")

    monkeypatch.setattr(store, "_temporal_entities", temporal_entities_helper)
    monkeypatch.setattr(store, "_require_connection", fail_connection)

    index = store._vector_index("agent-1", "2026-05-20T01:01:00Z")

    assert index.entities[0] is temporal_entities[0]
    group = index.groups[(2, LEGACY_EMBEDDING_VERSION)]
    assert isinstance(group, embedded_graph_store._VectorGroup)
    assert group.matrix.tolist() == [[0.0, 1.0]]
    assert group.entity_indexes == [0]
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reports_inferred_edge_status(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Decision A",
                    entity_type="decision",
                    observed_at="2026-05-20T01:00:00Z",
                ),
                ExtractedEntity(
                    name="Source B",
                    entity_type="source",
                    observed_at="2026-05-20T01:00:00Z",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Decision A",
                    target="Source B",
                    relation_type="supported_by",
                    valid_from="2026-05-20T01:00:00Z",
                    inferred=True,
                    confidence=0.8,
                    inference_method="cited_decision",
                    evidence={"source_event_seq": 7, "source_event_hash": "hash-7", "quote": "because"},
                )
            ],
            source_event_seq=7,
            source_event_hash="hash-7",
        ),
        session_id="agent-1",
    )

    status = await store.inspect_inferred_edge_status("agent-1")

    assert status.total_edges == 1
    assert status.method_count == 1
    assert status.evidence_count == 1
    assert status.evidence_coverage == 1.0
    assert status.methods[0].method == "cited_decision"
    assert status.methods[0].relation_types == ("supported_by",)
    assert status.samples[0].source == "Decision A"
    assert status.samples[0].target == "Source B"
    assert status.samples[0].source_event_seq == 7
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_inferred_edge_status_counts_all_rows_when_samples_are_limited() -> None:
    """Inferred-edge status totals should not be sampled by the display limit."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    store._connection = _CountingConnection()

    rows = [
        ["Manual Source", "Manual Target", "relates_to", None, "manual", None, None, "{}"],
        ["Cited Source", "Cited Target", "supported_by", 0.9, "cited_decision", 5, "hash-5", '{"quote":"because"}'],
    ]

    class _InferredEdgeConnection:
        def execute(self, query: str, params: dict[str, object] | None = None) -> _FakeRows:
            assert "r.inferred = true" in query
            assert params == {"session_id": "agent-1"}
            return _FakeRows(rows)

    store._connection = _InferredEdgeConnection()

    status = await store.inspect_inferred_edge_status("agent-1", limit=1)

    assert status.total_edges == 2
    assert status.method_count == 2
    assert status.evidence_count == 1
    assert status.missing_evidence_count == 1
    assert status.missing_source_event_count == 1
    assert status.evidence_coverage == 0.5
    assert [method.method for method in status.methods] == ["cited_decision", "manual"]
    assert len(status.samples) == 1
    assert status.samples[0].method == "manual"


@pytest.mark.asyncio
async def test_embedded_store_reports_missing_schema_as_empty_inferred_status(tmp_path: Path) -> None:
    """Read-only inferred-edge audit should not crash before projection exists."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()

    status = await store.inspect_inferred_edge_status("agent-1")

    assert status.session_id == "agent-1"
    assert status.total_edges == 0
    assert status.method_count == 0
    assert status.evidence_count == 0
    assert status.missing_evidence_count == 0
    assert status.missing_source_event_count == 0
    assert status.evidence_coverage == 1.0
    assert status.methods == ()
    assert status.samples == ()
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_reset_rebuilds_projection_artifact(tmp_path: Path) -> None:
    graph_path = tmp_path / "embedded.kuzu"
    store = EmbeddedGraphStore(graph_path)
    await store.connect()
    await store.init_schema()
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Temporary",
                    entity_type="fact",
                    observed_at="2026-05-20T01:00:00Z",
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    assert await store.search_exact("Temporary", session_id="agent-1")

    await store.reset_benchmark_projection()

    assert graph_path.exists()
    assert await store.search_exact("Temporary", session_id="agent-1") == []
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_benchmark_projection_marker_round_trips(tmp_path: Path) -> None:
    """Embedded benchmark projections should carry semantic reuse markers."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    assert await store.benchmark_projection_present("longmemeval-key") is False

    await store.mark_benchmark_projection(
        "longmemeval-key",
        [
            type("Event", (), {"seq": 1, "hash": "hash-1"})(),
            type("Event", (), {"seq": 2, "hash": "hash-2"})(),
        ],
    )

    assert await store.benchmark_projection_present("longmemeval-key") is True
    assert await store.benchmark_projection_present("other-key") is False
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_retires_source_projections_and_relationships(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="docs/guide.md",
                    entity_type="source",
                    observed_at="2026-05-20T01:00:00Z",
                    properties={"source_path": "docs/guide.md"},
                ),
                ExtractedEntity(
                    name="Guide Claim",
                    entity_type="fact",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="claim from retired source",
                    properties={"source_path": "docs/guide.md"},
                ),
                ExtractedEntity(
                    name="Stable Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="goal remains active",
                    properties={"source_path": "docs/other.md"},
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Guide Claim",
                    target="Stable Goal",
                    relation_type="supports",
                    valid_from="2026-05-20T01:00:00Z",
                )
            ],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    assert await store.search_exact("Guide Claim", entity_type="fact", session_id="agent-1")
    assert [entity.name for entity in await store.search_traversal("Guide Claim", depth=1, session_id="agent-1")] == [
        "Stable Goal"
    ]

    await store.retire_source_projections(
        source_path="docs/guide.md",
        invalid_at="2026-05-20T02:00:00Z",
        session_id="agent-1",
    )

    assert await store.search_exact("Guide Claim", entity_type="fact", session_id="agent-1") == []
    assert await store.search_exact("Stable Goal", entity_type="goal", session_id="agent-1")
    assert await store.search_traversal("Guide Claim", depth=1, session_id="agent-1") == []

    historical = await store.search_exact(
        "Guide Claim",
        entity_type="fact",
        temporal_point="2026-05-20T01:30:00Z",
        session_id="agent-1",
    )
    assert len(historical) == 1
    assert historical[0].valid_to == "2026-05-20T02:00:00Z"

    historical_neighbors = await store.search_traversal(
        "Guide Claim",
        depth=1,
        temporal_point="2026-05-20T01:30:00Z",
        session_id="agent-1",
    )
    assert [entity.name for entity in historical_neighbors] == ["Stable Goal"]
    await store.close()


def _versioned_vector_entity(
    name: str,
    embedding: list[float],
    version: str | None = None,
    session_id: str = "agent-1",
) -> GraphEntity:
    properties: dict[str, object] = {"embedding": embedding}
    if version is not None:
        properties["embedding_version"] = version
    return GraphEntity(
        name=name,
        entity_type="goal",
        valid_from="2026-05-20T01:00:00Z",
        valid_to=None,
        properties=properties,
        session_id=session_id,
    )


def _seeded_unit_embeddings(count: int, dimension: int, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((count, dimension))
    return [list(map(float, row)) for row in matrix]


@pytest.mark.asyncio
async def test_embedded_store_vector_search_isolates_version_groups() -> None:
    """Search must only score vectors carrying the active embedding version tag."""
    store = EmbeddedGraphStore(Path("unused.kuzu"), active_embedding_version="hash@aaaa-dim2")
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity("active tagged", [1.0, 0.0], "hash@aaaa-dim2"),
        _versioned_vector_entity("other tagged", [1.0, 0.0], "openai:text-embedding-3-small@1.0.0-dim2"),
        _versioned_vector_entity("legacy untagged", [1.0, 0.0]),
    ]

    active_results = await store.search_vector([1.0, 0.0], limit=10, session_id="agent-1")
    assert [result.entity.name for result in active_results] == ["active tagged"]
    assert active_results[0].exact is True

    index = store._vector_index_cache[("agent-1", None)]
    assert set(index.groups) == {
        (2, "hash@aaaa-dim2"),
        (2, "openai:text-embedding-3-small@1.0.0-dim2"),
        (2, LEGACY_EMBEDDING_VERSION),
    }


@pytest.mark.asyncio
async def test_embedded_store_vector_search_reaches_legacy_vectors_explicitly() -> None:
    """Untagged vectors group under the legacy tag and stay explicitly reachable."""
    store = EmbeddedGraphStore(Path("unused.kuzu"), active_embedding_version="hash@aaaa-dim2")
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity("active tagged", [1.0, 0.0], "hash@aaaa-dim2"),
        _versioned_vector_entity("legacy untagged", [1.0, 0.0]),
    ]

    legacy_results = await store.search_vector(
        [1.0, 0.0],
        limit=10,
        session_id="agent-1",
        embedding_version=LEGACY_EMBEDDING_VERSION,
    )

    assert [result.entity.name for result in legacy_results] == ["legacy untagged"]


@pytest.mark.asyncio
async def test_embedded_store_vector_search_resolves_active_version_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an explicit version, search uses the active provider's version tag."""
    import zaxy.embedding as embedding_module

    monkeypatch.setattr(
        embedding_module,
        "resolved_active_embedding_version_tag",
        lambda: "hash@settings-dim2",
    )
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity("settings tagged", [1.0, 0.0], "hash@settings-dim2"),
        _versioned_vector_entity("legacy untagged", [1.0, 0.0]),
    ]

    results = await store.search_vector([1.0, 0.0], limit=10, session_id="agent-1")

    assert [result.entity.name for result in results] == ["settings tagged"]


@pytest.mark.asyncio
async def test_embedded_store_re_embed_session_migrates_stale_vectors(tmp_path: Path) -> None:
    """Batch re-embedding upserts stale-version vectors onto the active tag."""
    from zaxy.embedding import HashEmbeddingProvider

    provider = HashEmbeddingProvider(dimension=8)
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Legacy Goal",
                    entity_type="goal",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="needs migration",
                    embedding=[1.0, 0.0],
                ),
                ExtractedEntity(
                    name="Plain Task",
                    entity_type="task",
                    observed_at="2026-05-20T01:00:00Z",
                    summary="no vector at all",
                ),
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="hash-1",
        ),
        session_id="agent-1",
    )

    report = await store.re_embed_session(session_id="agent-1", provider=provider)

    assert report == {"scanned": 1, "re_embedded": 1, "already_current": 0}
    migrated = await store.search_vector(
        provider.embed("Legacy Goal (goal) needs migration"),
        limit=5,
        session_id="agent-1",
        embedding_version=provider.version_tag,
    )
    assert [result.entity.name for result in migrated] == ["Legacy Goal"]
    assert await store.embedding_version_counts("agent-1") == {provider.version_tag: 1}

    second = await store.re_embed_session(session_id="agent-1", provider=provider)
    assert second == {"scanned": 1, "re_embedded": 0, "already_current": 1}
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_re_embed_session_requires_version_tag(tmp_path: Path) -> None:
    """Providers without a version tag cannot stamp migrated vectors."""

    class TaglessProvider:
        dimension = 4

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0]

    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    with pytest.raises(ValueError, match="version tag"):
        await store.re_embed_session(session_id="agent-1", provider=TaglessProvider())
    await store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_search_matches_exact_above_threshold(tmp_path: Path) -> None:
    """Above the ANN threshold, HNSW search recall@10 must stay >= 0.95 vs exact."""
    dimension = 32
    embeddings = _seeded_unit_embeddings(300, dimension, seed=7)
    entities = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]

    exact_store = EmbeddedGraphStore(
        tmp_path / "exact.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=10_000,
    )
    exact_store._current_entity_index_cache["agent-1"] = entities

    ann_store = EmbeddedGraphStore(
        tmp_path / "ann.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=50,
    )
    await ann_store.connect()
    await ann_store.init_schema()
    ann_store._current_entity_index_cache["agent-1"] = entities

    ann_group = ann_store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(ann_group, embedded_graph_store._AnnVectorGroup)
    assert ann_group.vector_count == 300
    assert ann_group.matrix_bytes == 0

    queries = _seeded_unit_embeddings(20, dimension, seed=11)
    recalls = []
    for query in queries:
        exact_names = [
            result.entity.name
            for result in await exact_store.search_vector(query, limit=10, session_id="agent-1")
        ]
        ann_results = await ann_store.search_vector(query, limit=10, session_id="agent-1")
        assert all(result.exact is False for result in ann_results)
        ann_names = [result.entity.name for result in ann_results]
        recalls.append(len(set(exact_names) & set(ann_names)) / max(1, len(exact_names)))
    assert sum(recalls) / len(recalls) >= 0.95
    await ann_store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_threshold_boundary_keeps_exact_path(tmp_path: Path) -> None:
    """Counts below the threshold stay exact; the threshold itself engages ANN.

    The 2.2 G4 count clause is inclusive (count >= threshold) so the lane
    evidence recorded at exactly 10^5 vectors covers corpora of exactly the
    default threshold size.
    """
    dimension = 8
    embeddings = _seeded_unit_embeddings(3, dimension, seed=3)
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=4,
    )
    await store.connect()
    await store.init_schema()
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]

    group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(group, embedded_graph_store._VectorGroup)
    results = await store.search_vector(embeddings[0], limit=3, session_id="agent-1")
    assert results and all(result.exact is True for result in results)

    store._clear_read_caches("agent-1")
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(_seeded_unit_embeddings(4, dimension, seed=3))
    ]
    crossed_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(crossed_group, embedded_graph_store._AnnVectorGroup)
    crossed = await store.search_vector(embeddings[0], limit=3, session_id="agent-1")
    assert crossed and all(result.exact is False for result in crossed)
    await store.close()


def test_embedded_store_ann_engagement_matrix_at_shipped_defaults() -> None:
    """Pin the shipped engagement rule at the 2.2 defaults.

    HNSW engages only when count >= VECTOR_ANN_THRESHOLD (default 100_000,
    lane-proven at exactly 10^5 / dim 64 on two consecutive ALL-criteria
    runs) AND dimension <= VECTOR_ANN_MAX_DIMENSION (default 64, the measured
    envelope: at dim 1536 / 50k gaussian the lane measured HNSW recall@10 of
    0.6 at efs 400 while exact answered in 22ms p50). Both boundaries are
    inclusive on the engaging side.
    """
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    assert store._ann_engagement_reason(count=99_999, dimension=64) is None
    assert store._ann_engagement_reason(count=100_000, dimension=64) == "count"
    assert store._ann_engagement_reason(count=100_000, dimension=65) is None
    assert store._ann_engagement_reason(count=100_000, dimension=1536) is None


def test_embedded_store_ann_byte_budget_clause_requires_dimension_ceiling() -> None:
    """The byte clause never overrides the dimension ceiling.

    Clause (b) — the exact float64 matrix (count * dimension * 8 bytes) would
    exceed VECTOR_INDEX_CACHE_MAX_BYTES — only applies at or below
    VECTOR_ANN_MAX_DIMENSION: the d1536-50k crossover measured exact search
    at 22ms p50 while 2.4x over budget (the newest matrix always stays
    resident; the budget bounds multi-scope cache totals), against HNSW
    recall@10 of 0.6. The clause arithmetic is pinned with the real constant
    on a store whose ceiling is explicitly raised to cover dim 1536.
    """
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    # Over budget at dim 1536 (22_000 * 1536 * 8 > 256 MiB) stays exact under
    # the shipped ceiling of 64.
    assert store._ann_engagement_reason(count=22_000, dimension=1536) is None

    # With the ceiling explicitly raised, the byte clause engages exactly one
    # row past the budget boundary: VECTOR_INDEX_CACHE_MAX_BYTES // (1536 * 8)
    # float64 rows fit, one more crosses.
    raised = EmbeddedGraphStore(Path("unused-raised.kuzu"), vector_ann_max_dimension=1536)
    boundary = embedded_graph_store.VECTOR_INDEX_CACHE_MAX_BYTES // (1536 * 8)
    assert boundary == 21_845
    assert raised._ann_engagement_reason(count=boundary, dimension=1536) is None
    assert raised._ann_engagement_reason(count=boundary + 1, dimension=1536) == "byte_budget"
    assert raised._ann_engagement_reason(count=20_000, dimension=1536) is None
    assert raised._ann_engagement_reason(count=22_000, dimension=1536) == "byte_budget"


def test_embedded_store_ann_explicit_threshold_is_absolute_for_count_clause() -> None:
    """An explicit threshold override owns clause (a); clause (b) still applies.

    Setting VECTOR_ANN_THRESHOLD very high silences the count clause at any
    corpus size, but within the dimension ceiling the byte clause engages
    independently — the documented full opt-out is the byte-budget engagement
    flag.
    """
    store = EmbeddedGraphStore(Path("unused.kuzu"), vector_ann_threshold=10**9)

    # Far above the shipped default, the raised override keeps the count
    # clause silent while the matrix stays under budget (dim 8 rows are 64
    # bytes; 4M rows are 256_000_000 bytes, just under the 256 MiB budget).
    assert store._ann_engagement_reason(count=4_000_000, dimension=8) is None
    # The same count at dim 64 is 2.048 GB of float64 — the byte clause
    # engages regardless of the explicit threshold.
    assert store._ann_engagement_reason(count=4_000_000, dimension=64) == "byte_budget"

    lowered = EmbeddedGraphStore(Path("unused-low.kuzu"), vector_ann_threshold=50)
    assert lowered._ann_engagement_reason(count=50, dimension=8) == "count"
    assert lowered._ann_engagement_reason(count=49, dimension=8) is None


def test_embedded_store_ann_byte_budget_engagement_escape_hatch() -> None:
    """VECTOR_ANN_BYTE_BUDGET_ENGAGEMENT=false disables clause (b) only.

    The ceiling is raised to dim 1536 in both stores so the flag — not the
    dimension guard — is what the assertions exercise.
    """
    store = EmbeddedGraphStore(
        Path("unused.kuzu"),
        vector_ann_threshold=10**9,
        vector_ann_max_dimension=1536,
        vector_ann_byte_budget_engagement=False,
    )
    # 22_000 * 1536 * 8 bytes exceeds the budget, but the flag forces exact.
    assert store._ann_engagement_reason(count=22_000, dimension=1536) is None

    counted = EmbeddedGraphStore(
        Path("unused-count.kuzu"),
        vector_ann_threshold=100,
        vector_ann_max_dimension=1536,
        vector_ann_byte_budget_engagement=False,
    )
    # The count clause is unaffected by the flag.
    assert counted._ann_engagement_reason(count=100, dimension=1536) == "count"


def test_embedded_store_ann_engagement_preserves_quantized_precedence() -> None:
    """int8 opt-in keeps its pre-G4 precedence against the new byte clause.

    Below the count threshold an explicit VECTOR_QUANTIZATION=int8 wins even
    over budget (int8 keeps ~1/8 of the float64 bytes resident); at or above
    the count threshold the ANN path is tried first, exactly as before 2.2 G4.
    Ceilings are raised to dim 1536 so the precedence — not the dimension
    guard — is what the assertions exercise.
    """
    quantized = EmbeddedGraphStore(
        Path("unused.kuzu"),
        vector_ann_threshold=10**9,
        vector_ann_max_dimension=1536,
        vector_quantization="int8",
    )
    assert quantized._ann_engagement_reason(count=22_000, dimension=1536) is None

    above_count = EmbeddedGraphStore(
        Path("unused-count.kuzu"),
        vector_ann_threshold=100,
        vector_ann_max_dimension=1536,
        vector_quantization="int8",
    )
    assert above_count._ann_engagement_reason(count=100, dimension=1536) == "count"


def test_embedded_store_quantized_opt_in_is_orthogonal_to_dimension_ceiling() -> None:
    """int8 stays the strategy above the ceiling — quantization is orthogonal.

    Above the count threshold AND above the dimension ceiling, the ANN path
    never engages, so an explicit int8 opt-in builds the quantized group it
    would have built anyway (the high-dimension posture: exact or opted-in
    int8, never HNSW, unless the ceiling is raised explicitly).
    """
    store = EmbeddedGraphStore(
        Path("unused.kuzu"),
        active_embedding_version="v1",
        vector_ann_threshold=4,
        vector_quantization="int8",
    )
    assert store._ann_engagement_reason(count=100_000, dimension=1536) is None

    dimension = 65
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(_seeded_unit_embeddings(6, dimension, seed=13))
    ]
    group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(group, embedded_graph_store._QuantizedVectorGroup)


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_dimension_ceiling_boundary_selects_group_type(tmp_path: Path) -> None:
    """At the same engaged count, dim 64 builds the ANN group and dim 65 stays exact.

    Both groups live in one store and one session so the only difference the
    strategy selection sees is the vector dimension against the shipped
    ceiling (boundary inclusive at 64, exclusive at 65).
    """
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=5,
    )
    await store.connect()
    await store.init_schema()
    at_ceiling = _seeded_unit_embeddings(5, 64, seed=17)
    above_ceiling = _seeded_unit_embeddings(5, 65, seed=19)
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"at-ceiling-{position}", vector, "v1")
        for position, vector in enumerate(at_ceiling)
    ] + [
        _versioned_vector_entity(f"above-ceiling-{position}", vector, "v1")
        for position, vector in enumerate(above_ceiling)
    ]

    groups = store._vector_index("agent-1", None).groups
    assert isinstance(groups[(64, "v1")], embedded_graph_store._AnnVectorGroup)
    assert isinstance(groups[(65, "v1")], embedded_graph_store._VectorGroup)
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_no_shadow_build_above_dimension_ceiling(tmp_path: Path) -> None:
    """Rebuild triggers never build a shadow generation above the ceiling.

    The dimension guard sits in the single strategy-selection choke point, so
    the lazy rebuild path (read-cache invalidation followed by a vector-index
    rebuild over a grown corpus) must not create any HNSW shadow table for an
    above-ceiling scope — an index that would never be queried.
    """
    dimension = 65
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=4,
    )
    await store.connect()
    await store.init_schema()
    embeddings = _seeded_unit_embeddings(12, dimension, seed=23)
    entities = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]

    store._current_entity_index_cache["agent-1"] = entities[:8]
    group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(group, embedded_graph_store._VectorGroup)

    # The rebuild trigger: a projection change clears read caches and the
    # next query rebuilds the vector index over the grown corpus.
    store._clear_read_caches("agent-1")
    store._current_entity_index_cache["agent-1"] = entities
    rebuilt = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(rebuilt, embedded_graph_store._VectorGroup)

    assert store._ann_generation_states == {}
    table_names = [
        str(row[0]) for row in store._execute("CALL SHOW_TABLES() RETURN name").get_all()
    ]
    assert not any(
        name.startswith(embedded_graph_store._ANN_SHADOW_TABLE_PREFIX) for name in table_names
    )
    results = await store.search_vector(embeddings[0], limit=3, session_id="agent-1")
    assert results and all(result.exact is True for result in results)
    await store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_byte_budget_clause_builds_ann_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-budget scopes build ANN groups below the count threshold; int8 stays quantized.

    The budget is monkeypatched to exactly 15 float64 rows at dim 8 so the
    16-row corpus crosses the ceiling with the real count * dimension * 8
    arithmetic.
    """
    dimension = 8
    row_bytes = dimension * 8
    monkeypatch.setattr(embedded_graph_store, "VECTOR_INDEX_CACHE_MAX_BYTES", 15 * row_bytes)
    embeddings = _seeded_unit_embeddings(16, dimension, seed=41)
    entities = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]

    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=10**9,
    )
    await store.connect()
    await store.init_schema()
    store._current_entity_index_cache["agent-1"] = entities
    over_budget_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(over_budget_group, embedded_graph_store._AnnVectorGroup)
    results = await store.search_vector(embeddings[0], limit=3, session_id="agent-1")
    assert results and all(result.exact is False for result in results)

    # Exactly at budget (15 rows * 64 bytes) the exact dense path is kept.
    store._clear_read_caches("agent-1")
    store._current_entity_index_cache["agent-1"] = entities[:15]
    at_budget_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(at_budget_group, embedded_graph_store._VectorGroup)
    await store.close()

    # An explicit int8 opt-in takes precedence over the byte clause.
    quantized_store = EmbeddedGraphStore(
        tmp_path / "quantized.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=10**9,
        vector_quantization="int8",
    )
    quantized_store._current_entity_index_cache["agent-1"] = entities
    quantized_group = quantized_store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(quantized_group, embedded_graph_store._QuantizedVectorGroup)


@pytest.mark.asyncio
async def test_embedded_store_ann_group_rebuilds_with_projection(tmp_path: Path) -> None:
    """ANN shadow rows follow the same rebuild trigger as the dense matrix."""
    dimension = 8
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await store.connect()
    await store.init_schema()
    first_corpus = [
        _versioned_vector_entity(f"first-{position}", vector, "v1")
        for position, vector in enumerate(_seeded_unit_embeddings(5, dimension, seed=5))
    ]
    store._current_entity_index_cache["agent-1"] = first_corpus
    query = first_corpus[0].properties["embedding"]
    assert isinstance(query, list)
    initial = await store.search_vector(query, limit=3, session_id="agent-1")
    assert all(result.entity.name.startswith("first-") for result in initial)

    store._clear_read_caches("agent-1")
    second_corpus = [
        _versioned_vector_entity(f"second-{position}", vector, "v1")
        for position, vector in enumerate(_seeded_unit_embeddings(5, dimension, seed=6))
    ]
    store._current_entity_index_cache["agent-1"] = second_corpus

    rebuilt = await store.search_vector(query, limit=3, session_id="agent-1")
    assert rebuilt and all(result.entity.name.startswith("second-") for result in rebuilt)
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_ann_capability_probe_falls_back_to_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime without CREATE_VECTOR_INDEX keeps the exact path above threshold."""
    dimension = 8
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await store.connect()
    await store.init_schema()
    real_connection = store._require_connection()

    class NoVectorFunctionConnection:
        def execute(self, query: str, params: dict[str, object] | None = None) -> object:
            if "CREATE NODE TABLE IF NOT EXISTS zaxy_vector_capability_probe" in query:
                return _FakeRows([])
            if "CALL CREATE_VECTOR_INDEX('zaxy_vector_capability_probe'" in query:
                raise RuntimeError("vector index extension is unavailable")
            if "DROP TABLE zaxy_vector_capability_probe" in query:
                return _FakeRows([])
            raise AssertionError(f"unexpected query during capability probe: {query}")

    monkeypatch.setattr(store, "_connection", NoVectorFunctionConnection())
    assert store._vector_index_supported() is False

    monkeypatch.setattr(store, "_connection", real_connection)
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(_seeded_unit_embeddings(5, dimension, seed=5))
    ]
    group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(group, embedded_graph_store._VectorGroup)
    results = await store.search_vector(
        group.matrix[0].tolist(),
        limit=3,
        session_id="agent-1",
    )
    assert results and all(result.exact is True for result in results)
    await store.close()


def test_embedded_store_ann_capability_probe_uses_real_vector_index_operation() -> None:
    """The vector capability probe should verify the operation Zaxy depends on."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    class VectorProbeConnection:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def execute(self, query: str, params: dict[str, object] | None = None) -> object:
            assert params is None
            self.queries.append(query)
            return _FakeRows([])

    connection = VectorProbeConnection()
    store._connection = connection

    assert store._vector_index_supported() is True
    assert store._vector_index_supported() is True
    assert connection.queries == [
        "CREATE NODE TABLE IF NOT EXISTS zaxy_vector_capability_probe("
        "id INT64, vec FLOAT[2], PRIMARY KEY(id))",
        "CALL CREATE_VECTOR_INDEX('zaxy_vector_capability_probe', "
        "'zaxy_vector_capability_probe_idx', 'vec', metric := 'cosine')",
        "CALL DROP_VECTOR_INDEX('zaxy_vector_capability_probe', "
        "'zaxy_vector_capability_probe_idx')",
        "DROP TABLE zaxy_vector_capability_probe",
    ]


def test_embedded_store_ann_capability_probe_suppresses_cleanup_failures() -> None:
    """Cleanup failures in transient probe artifacts must not disable the store."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    class CleanupFailureConnection:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def execute(self, query: str, params: dict[str, object] | None = None) -> object:
            assert params is None
            self.queries.append(query)
            if "DROP_VECTOR_INDEX" in query or query.startswith("DROP TABLE "):
                raise RuntimeError("probe artifact was already gone")
            return _FakeRows([])

    connection = CleanupFailureConnection()
    store._connection = connection

    assert store._vector_index_supported() is True
    assert connection.queries[-2:] == [
        "CALL DROP_VECTOR_INDEX('zaxy_vector_capability_probe', "
        "'zaxy_vector_capability_probe_idx')",
        "DROP TABLE zaxy_vector_capability_probe",
    ]


class _RecordingConnection:
    """Pass-through connection wrapper that records every executed query."""

    def __init__(self, real_connection: object) -> None:
        self._real_connection = real_connection
        self.queries: list[str] = []

    def execute(self, query: str, parameters: dict[str, object] | None = None) -> object:
        self.queries.append(query)
        if parameters is None:
            return self._real_connection.execute(query)  # type: ignore[attr-defined]
        return self._real_connection.execute(query, parameters)  # type: ignore[attr-defined]


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_query_is_direct_table_without_projected_graph(tmp_path: Path) -> None:
    """ANN search hits the per-scope shadow table directly with the configured efs.

    No PROJECT_GRAPH/DROP_PROJECTED_GRAPH may run on the query path: the
    per-(session, version) shadow table makes the common query unfiltered,
    and per-query projected graphs paid a prefilter mask scan per search.
    """
    dimension = 8
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await store.connect()
    await store.init_schema()
    embeddings = _seeded_unit_embeddings(30, dimension, seed=13)
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]
    group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(group, embedded_graph_store._AnnVectorGroup)

    recording = _RecordingConnection(store._require_connection())
    store._connection = recording
    results = await store.search_vector(embeddings[0], limit=10, session_id="agent-1")

    assert results and all(result.exact is False for result in results)
    assert all("PROJECT_GRAPH" not in query for query in recording.queries)
    vector_queries = [query for query in recording.queries if "QUERY_VECTOR_INDEX" in query]
    assert len(vector_queries) == 1
    assert f"'{group.table_name}'" in vector_queries[0]
    assert f"'{group.index_name}'" in vector_queries[0]
    # Pins the Settings.vector_ann_efs default (400 since the 2.2 efs sweep).
    assert "efs := 400" in vector_queries[0]
    await store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_efs_override_reaches_query(tmp_path: Path) -> None:
    """The efs override is inlined into the HNSW query, floored at the oversampled k."""
    dimension = 8
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
        vector_ann_efs=333,
    )
    await store.connect()
    await store.init_schema()
    embeddings = _seeded_unit_embeddings(60, dimension, seed=19)
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]
    store._vector_index("agent-1", None)

    recording = _RecordingConnection(store._require_connection())
    store._connection = recording
    assert await store.search_vector(embeddings[0], limit=10, session_id="agent-1")
    assert any("efs := 333" in query for query in recording.queries)

    # efs can never sit below the candidate count requested from the index.
    store._vector_ann_efs_override = 1
    recording.queries.clear()
    assert await store.search_vector(embeddings[0], limit=10, session_id="agent-1")
    oversampled_k = 10 * embedded_graph_store.VECTOR_SEARCH_OVERSAMPLE
    assert any(f"efs := {oversampled_k}" in query for query in recording.queries)
    await store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_rerank_matches_exact_when_oversample_covers_corpus(tmp_path: Path) -> None:
    """When k*oversample spans the corpus, the float64 rerank reproduces exact order."""
    dimension = 16
    embeddings = _seeded_unit_embeddings(30, dimension, seed=37)
    entities = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]
    exact_store = EmbeddedGraphStore(
        tmp_path / "exact.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=10_000,
    )
    exact_store._current_entity_index_cache["agent-1"] = entities
    ann_store = EmbeddedGraphStore(
        tmp_path / "ann.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await ann_store.connect()
    await ann_store.init_schema()
    ann_store._current_entity_index_cache["agent-1"] = entities

    for query in _seeded_unit_embeddings(5, dimension, seed=41):
        exact_results = await exact_store.search_vector(query, limit=10, session_id="agent-1")
        ann_results = await ann_store.search_vector(query, limit=10, session_id="agent-1")
        assert [result.entity.name for result in ann_results] == [
            result.entity.name for result in exact_results
        ]
        # Same float64 inputs, but matrix-product vs per-row dot rounding can
        # differ in the last bits; ordering equality above is the real claim.
        assert [result.score for result in ann_results] == pytest.approx(
            [result.score for result in exact_results],
            abs=1e-12,
        )
        assert all(result.exact is False for result in ann_results)
    await ann_store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_rebuild_writes_fresh_generation(tmp_path: Path) -> None:
    """Rebuilds write a fresh insert-only generation table and drop the old one.

    Generations remain the swap mechanism (queries only ever hit a fully
    built table, and single-statement delete-ALL under a live index still
    breaks searches on LadybugDB 0.17.1), but superseded generations are now
    dropped outright for full space reclaim: the fork fixed the kuzu#6040
    DROP_VECTOR_INDEX metadata corruption, verified through this very cycle.
    """
    dimension = 8
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await store.connect()
    await store.init_schema()
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"first-{position}", vector, "v1")
        for position, vector in enumerate(_seeded_unit_embeddings(5, dimension, seed=5))
    ]
    first_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(first_group, embedded_graph_store._AnnVectorGroup)
    assert first_group.table_name.endswith("_g0")

    store._clear_read_caches("agent-1")
    second_vectors = _seeded_unit_embeddings(5, dimension, seed=6)
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity(f"second-{position}", vector, "v1")
        for position, vector in enumerate(second_vectors)
    ]
    second_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(second_group, embedded_graph_store._AnnVectorGroup)
    assert second_group.table_name.endswith("_g1")

    rebuilt = await store.search_vector(second_vectors[0], limit=3, session_id="agent-1")
    assert rebuilt and all(result.entity.name.startswith("second-") for result in rebuilt)
    table_names = {str(row[0]) for row in store._execute("CALL SHOW_TABLES() RETURN name").get_all()}
    assert first_group.table_name not in table_names
    assert second_group.table_name in table_names

    # A second session above the threshold owns its own shadow scope.
    store._current_entity_index_cache["agent-2"] = [
        _versioned_vector_entity(f"other-{position}", vector, "v1", session_id="agent-2")
        for position, vector in enumerate(_seeded_unit_embeddings(5, dimension, seed=7))
    ]
    other_group = store._vector_index("agent-2", None).groups[(dimension, "v1")]
    assert isinstance(other_group, embedded_graph_store._AnnVectorGroup)
    assert other_group.table_name != second_group.table_name
    assert other_group.table_name.endswith("_g0")
    await store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_copy_and_unwind_loads_are_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-memory Arrow COPY load and the UNWIND fallback must serve identical results.

    Requires pyarrow (a transitive, not guaranteed, dependency): without it the
    COPY arm degrades to the UNWIND fallback and the comparison is vacuous.

    pyarrow is a transitive (not guaranteed) dependency, so the bulk loader
    degrades to batched UNWIND when it is absent; both loads store the same
    float32 rows and must answer the same queries identically.
    """
    pytest.importorskip("pyarrow")
    dimension = 16
    embeddings = _seeded_unit_embeddings(60, dimension, seed=23)
    entities = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]

    copy_store = EmbeddedGraphStore(
        tmp_path / "copy.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await copy_store.connect()
    await copy_store.init_schema()
    copy_store._current_entity_index_cache["agent-1"] = entities
    copy_recording = _RecordingConnection(copy_store._require_connection())
    copy_store._connection = copy_recording
    copy_group = copy_store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(copy_group, embedded_graph_store._AnnVectorGroup)
    assert any(query.startswith("COPY ") for query in copy_recording.queries)
    assert not any("UNWIND $rows" in query for query in copy_recording.queries)

    real_find_spec = importlib.util.find_spec

    def hide_pyarrow(name: str, package: str | None = None) -> object | None:
        if name == "pyarrow":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", hide_pyarrow)
    unwind_store = EmbeddedGraphStore(
        tmp_path / "unwind.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await unwind_store.connect()
    await unwind_store.init_schema()
    unwind_store._current_entity_index_cache["agent-1"] = entities
    unwind_recording = _RecordingConnection(unwind_store._require_connection())
    unwind_store._connection = unwind_recording
    unwind_group = unwind_store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(unwind_group, embedded_graph_store._AnnVectorGroup)
    assert not any(query.startswith("COPY ") for query in unwind_recording.queries)
    assert any("UNWIND $rows" in query for query in unwind_recording.queries)

    for query in _seeded_unit_embeddings(10, dimension, seed=29):
        copy_results = await copy_store.search_vector(query, limit=10, session_id="agent-1")
        unwind_results = await unwind_store.search_vector(query, limit=10, session_id="agent-1")
        assert [result.entity.name for result in copy_results] == [
            result.entity.name for result in unwind_results
        ]
        assert [result.score for result in copy_results] == [
            result.score for result in unwind_results
        ]
    await copy_store.close()
    await unwind_store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_generation_swap_stress(tmp_path: Path) -> None:
    """Consecutive rebuilds must swap generations cleanly, surviving reopen.

    Three full corpus replacements in one process, then a reopen (which
    drops the in-memory generation state) followed by another rebuild: every
    stage must answer queries from the fresh generation only, and every
    superseded generation table must be dropped (full space reclaim).
    """
    dimension = 8
    path = tmp_path / "embedded.kuzu"
    store = EmbeddedGraphStore(path, active_embedding_version="v1", vector_ann_threshold=2)
    await store.connect()
    await store.init_schema()

    seen_tables: list[str] = []
    for round_index in range(3):
        corpus = [
            _versioned_vector_entity(f"round{round_index}-{position}", vector, "v1")
            for position, vector in enumerate(
                _seeded_unit_embeddings(12, dimension, seed=100 + round_index)
            )
        ]
        store._clear_read_caches("agent-1")
        store._current_entity_index_cache["agent-1"] = corpus
        group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
        assert isinstance(group, embedded_graph_store._AnnVectorGroup)
        assert group.table_name.endswith(f"_g{round_index}")
        seen_tables.append(group.table_name)
        query = corpus[0].properties["embedding"]
        assert isinstance(query, list)
        results = await store.search_vector(query, limit=5, session_id="agent-1")
        assert results and all(
            result.entity.name.startswith(f"round{round_index}-") for result in results
        )
        table_names = {str(row[0]) for row in store._execute("CALL SHOW_TABLES() RETURN name").get_all()}
        for old_table in seen_tables[:-1]:
            assert old_table not in table_names

    await store.close()
    await store.connect()
    await store.init_schema()
    reopened_corpus = [
        _versioned_vector_entity(f"reopen-{position}", vector, "v1")
        for position, vector in enumerate(_seeded_unit_embeddings(12, dimension, seed=200))
    ]
    store._current_entity_index_cache["agent-1"] = reopened_corpus
    reopened_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(reopened_group, embedded_graph_store._AnnVectorGroup)
    assert reopened_group.table_name.endswith("_g3")
    query = reopened_corpus[0].properties["embedding"]
    assert isinstance(query, list)
    results = await store.search_vector(query, limit=5, session_id="agent-1")
    assert results and all(result.entity.name.startswith("reopen-") for result in results)
    table_names = {str(row[0]) for row in store._execute("CALL SHOW_TABLES() RETURN name").get_all()}
    for old_table in seen_tables:
        assert old_table not in table_names
    await store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_ann_delta_policy_boundary(tmp_path: Path) -> None:
    """The delta policy must pick reuse, incremental insert, or swap correctly.

    An unchanged corpus reuses the resident generation with zero writes; an
    extension at the fraction boundary inserts only the delta into the live
    index; one row past the boundary triggers a full COPY generation swap.

    Requires pyarrow: the swap arm asserts COPY-specific behavior, which
    degrades to the UNWIND fallback (covered separately) when it is absent.
    """
    pytest.importorskip("pyarrow")
    dimension = 8
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await store.connect()
    await store.init_schema()
    vectors = _seeded_unit_embeddings(67, dimension, seed=31)
    entities = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(vectors)
    ]

    store._current_entity_index_cache["agent-1"] = entities[:50]
    first_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(first_group, embedded_graph_store._AnnVectorGroup)
    assert first_group.table_name.endswith("_g0")

    # Unchanged corpus: the resident generation is reused with zero writes.
    store._clear_read_caches("agent-1")
    store._current_entity_index_cache["agent-1"] = entities[:50]
    recording = _RecordingConnection(store._require_connection())
    store._connection = recording
    reused_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(reused_group, embedded_graph_store._AnnVectorGroup)
    assert reused_group.table_name == first_group.table_name
    assert not any(
        "CREATE NODE TABLE" in query or query.startswith("COPY ") or "UNWIND $rows" in query
        for query in recording.queries
    )

    # Boundary extension (delta == 10% of 50): incremental insert, same table.
    store._clear_read_caches("agent-1")
    store._current_entity_index_cache["agent-1"] = entities[:55]
    recording.queries.clear()
    incremental_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(incremental_group, embedded_graph_store._AnnVectorGroup)
    assert incremental_group.table_name == first_group.table_name
    assert incremental_group.vector_count == 55
    assert any("UNWIND $rows" in query for query in recording.queries)
    assert not any(
        "CREATE NODE TABLE" in query or query.startswith("COPY ") for query in recording.queries
    )
    newest = vectors[54]
    results = await store.search_vector(newest, limit=3, session_id="agent-1")
    assert results and results[0].entity.name == "entity-54"

    # One past the boundary (delta 6 > 10% of 55): full COPY generation swap.
    store._clear_read_caches("agent-1")
    store._current_entity_index_cache["agent-1"] = entities[:61]
    recording.queries.clear()
    swapped_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(swapped_group, embedded_graph_store._AnnVectorGroup)
    assert swapped_group.table_name.endswith("_g1")
    assert any("CREATE NODE TABLE" in query for query in recording.queries)
    assert any(query.startswith("COPY ") for query in recording.queries)
    table_names = {str(row[0]) for row in store._execute("CALL SHOW_TABLES() RETURN name").get_all()}
    assert first_group.table_name not in table_names
    results = await store.search_vector(vectors[60], limit=3, session_id="agent-1")
    assert results and results[0].entity.name == "entity-60"

    # A small delta that mutates resident rows is not an extension: swap.
    mutated = list(entities[:61])
    mutated[0] = _versioned_vector_entity("entity-0", vectors[66], "v1")
    store._clear_read_caches("agent-1")
    store._current_entity_index_cache["agent-1"] = mutated
    mutated_group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(mutated_group, embedded_graph_store._AnnVectorGroup)
    assert mutated_group.table_name.endswith("_g2")
    results = await store.search_vector(vectors[66], limit=3, session_id="agent-1")
    assert results and results[0].entity.name == "entity-0"
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_rerank_skips_embedding_mutated_after_admission() -> None:
    """The shared rerank tolerates a candidate embedding mutated after indexing.

    Group admission validates embeddings, so the rerank bulk-converts them in
    one pass; if a property was mutated to garbage afterwards it must fall
    back to per-candidate validation and skip the row rather than fail the
    search.
    """
    store = EmbeddedGraphStore(
        Path("unused.kuzu"),
        active_embedding_version="v1",
        vector_quantization="int8",
    )
    entities = [
        _versioned_vector_entity("kept-a", [1.0, 0.0], "v1"),
        _versioned_vector_entity("mutated", [0.9, 0.1], "v1"),
        _versioned_vector_entity("kept-b", [0.5, 0.5], "v1"),
    ]
    store._current_entity_index_cache["agent-1"] = entities
    store._vector_index("agent-1", None)
    entities[1].properties["embedding"] = "garbage"

    results = await store.search_vector([1.0, 0.0], limit=3, session_id="agent-1")

    assert [result.entity.name for result in results] == ["kept-a", "kept-b"]
    assert all(result.exact is False for result in results)


def test_embedded_store_execute_rejects_unbound_parameters() -> None:
    """The query choke point must refuse placeholders without bindings.

    LadybugDB 0.17.1 silently evaluates an unbound $parameter to NULL — the
    query "succeeds" with wrong answers (Kuzu 0.11.3 at least crashed) — so
    the statement must never reach the runtime.
    """
    store = EmbeddedGraphStore(Path("unused.kuzu"))
    connection = _CountingConnection()
    store._connection = connection

    with pytest.raises(RuntimeError, match=r"unbound parameters.*session_id"):
        store._execute(
            "MATCH (e:Entity) WHERE e.session_id = $session_id RETURN e.name",
            {"wrong_name": "agent-1"},
        )
    assert connection.queries == []

    store._execute(
        "MATCH (e:Entity) WHERE e.session_id = $session_id RETURN e.name",
        {"session_id": "agent-1"},
    )
    assert len(connection.queries) == 1


@requires_native_vector_index
def test_embedded_store_ann_capability_probe_detects_vector_support(tmp_path: Path) -> None:
    """The pinned LadybugDB runtime must report native vector index support."""
    import asyncio

    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    asyncio.run(store.connect())
    try:
        assert store._vector_index_supported() is True
    finally:
        asyncio.run(store.close())


def test_embedded_vector_quantization_is_opt_in() -> None:
    """Without opting in, vector groups stay exact float64 matrices."""
    store = EmbeddedGraphStore(Path("unused.kuzu"), vector_quantization="none")
    store._current_entity_index_cache["agent-1"] = [
        _versioned_vector_entity("plain", [3.0, 4.0], "v1"),
    ]

    group = store._vector_index("agent-1", None).groups[(2, "v1")]

    assert isinstance(group, embedded_graph_store._VectorGroup)
    assert group.matrix.dtype == np.float64


@pytest.mark.asyncio
async def test_embedded_vector_quantization_recall_guard() -> None:
    """Int8 oversample + float rerank must keep top-10 recall >= 0.95 vs exact."""
    dimension = 32
    embeddings = _seeded_unit_embeddings(300, dimension, seed=17)
    entities = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(embeddings)
    ]

    exact_store = EmbeddedGraphStore(Path("unused-exact.kuzu"), active_embedding_version="v1")
    exact_store._current_entity_index_cache["agent-1"] = entities
    quantized_store = EmbeddedGraphStore(
        Path("unused-quantized.kuzu"),
        active_embedding_version="v1",
        vector_quantization="int8",
    )
    quantized_store._current_entity_index_cache["agent-1"] = entities

    group = quantized_store._vector_index("agent-1", None).groups[(dimension, "v1")]
    assert isinstance(group, embedded_graph_store._QuantizedVectorGroup)
    assert group.matrix.dtype == np.int8

    recalls = []
    for query in _seeded_unit_embeddings(20, dimension, seed=23):
        exact_names = [
            result.entity.name
            for result in await exact_store.search_vector(query, limit=10, session_id="agent-1")
        ]
        quantized_results = await quantized_store.search_vector(query, limit=10, session_id="agent-1")
        assert all(result.exact is False for result in quantized_results)
        quantized_names = [result.entity.name for result in quantized_results]
        recalls.append(len(set(exact_names) & set(quantized_names)) / max(1, len(exact_names)))
    assert sum(recalls) / len(recalls) >= 0.95


def test_embedded_vector_quantization_reports_quantized_bytes() -> None:
    """Byte accounting must reflect int8 storage plus per-vector scales."""
    dimension = 16
    count = 4
    entities = [
        _versioned_vector_entity(f"entity-{position}", vector, "v1")
        for position, vector in enumerate(_seeded_unit_embeddings(count, dimension, seed=29))
    ]

    dense_store = EmbeddedGraphStore(Path("unused-dense.kuzu"), active_embedding_version="v1")
    dense_store._current_entity_index_cache["agent-1"] = entities
    quantized_store = EmbeddedGraphStore(
        Path("unused-quantized.kuzu"),
        active_embedding_version="v1",
        vector_quantization="int8",
    )
    quantized_store._current_entity_index_cache["agent-1"] = entities

    dense_index = dense_store._vector_index("agent-1", None)
    quantized_index = quantized_store._vector_index("agent-1", None)

    assert dense_index.matrix_bytes == count * dimension * 8
    assert quantized_index.matrix_bytes == count * dimension + count * 8
    assert quantized_index.matrix_bytes < dense_index.matrix_bytes


def test_embedded_vector_index_eviction_respects_quantized_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eviction must count actual quantized bytes, not float64-equivalent bytes."""
    dimension = 16
    count = 4
    # One quantized index is 96 bytes (64 int8 + 32 scale); the float64
    # equivalent is 512 bytes. A 300-byte budget holds two quantized indexes
    # but only one dense index.
    monkeypatch.setattr(embedded_graph_store, "VECTOR_INDEX_CACHE_MAX_BYTES", 300)

    quantized_store = EmbeddedGraphStore(
        Path("unused-quantized.kuzu"),
        active_embedding_version="v1",
        vector_quantization="int8",
    )
    for session in ("agent-1", "agent-2"):
        quantized_store._current_entity_index_cache[session] = [
            _versioned_vector_entity(f"{session}-{position}", vector, "v1", session_id=session)
            for position, vector in enumerate(_seeded_unit_embeddings(count, dimension, seed=31))
        ]
        quantized_store._vector_index(session, None)
    assert list(quantized_store._vector_index_cache) == [("agent-1", None), ("agent-2", None)]

    dense_store = EmbeddedGraphStore(Path("unused-dense.kuzu"), active_embedding_version="v1")
    for session in ("agent-1", "agent-2"):
        dense_store._current_entity_index_cache[session] = [
            _versioned_vector_entity(f"{session}-{position}", vector, "v1", session_id=session)
            for position, vector in enumerate(_seeded_unit_embeddings(count, dimension, seed=31))
        ]
        dense_store._vector_index(session, None)
    assert list(dense_store._vector_index_cache) == [("agent-2", None)]


# --- Adjacency snapshots for graph-walk retrieval ----------------------------


def _node_key(session_id: str, entity_type: str, name: str, seq: int) -> str:
    return embedded_graph_store._node_key(session_id, entity_type, name, seq)


def _snapshot_edge_pairs(snapshot: AdjacencySnapshot) -> list[tuple[str, str]]:
    """Decode a CSR snapshot back into sorted (source, target) node-key pairs."""
    pairs: list[tuple[str, str]] = []
    for source_index in range(snapshot.node_count):
        start = int(snapshot.indptr[source_index])
        stop = int(snapshot.indptr[source_index + 1])
        for edge_index in range(start, stop):
            pairs.append(
                (
                    snapshot.node_ids[source_index],
                    snapshot.node_ids[int(snapshot.indices[edge_index])],
                )
            )
    return sorted(pairs)


async def _adjacency_store(tmp_path: Path) -> EmbeddedGraphStore:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()
    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Goal A",
                    entity_type="goal",
                    observed_at="2026-06-10T01:00:00Z",
                    summary="ship graph walk",
                ),
                ExtractedEntity(
                    name="Task B",
                    entity_type="task",
                    observed_at="2026-06-10T01:00:00Z",
                    summary="implement adjacency",
                ),
                ExtractedEntity(
                    name="Task C",
                    entity_type="task",
                    observed_at="2026-06-10T01:00:00Z",
                    summary="test adjacency",
                ),
                ExtractedEntity(
                    name="Island D",
                    entity_type="note",
                    observed_at="2026-06-10T01:00:00Z",
                    summary="isolated entity",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Task B",
                    target="Goal A",
                    relation_type="supports",
                    valid_from="2026-06-10T01:00:00Z",
                ),
                ExtractedEdge(
                    source="Task C",
                    target="Task B",
                    relation_type="depends_on",
                    valid_from="2026-06-10T01:00:00Z",
                ),
            ],
            source_event_seq=1,
            source_event_hash="adjacency-hash",
            source_event_type="task.proposed",
            source_thread="agent-1",
        ),
        session_id="agent-1",
    )
    return store


@pytest.mark.asyncio
async def test_fetch_adjacency_matches_hand_seeded_graph_exactly(tmp_path: Path) -> None:
    """Snapshot nodes and edges should match the seeded graph, reverse edges included."""
    store = await _adjacency_store(tmp_path)
    goal_a = _node_key("agent-1", "goal", "Goal A", 1)
    task_b = _node_key("agent-1", "task", "Task B", 1)
    task_c = _node_key("agent-1", "task", "Task C", 1)
    island_d = _node_key("agent-1", "note", "Island D", 1)

    snapshot = await store.fetch_adjacency("agent-1")

    assert snapshot.node_ids == tuple(sorted([goal_a, task_b, task_c, island_d]))
    assert _snapshot_edge_pairs(snapshot) == sorted(
        [
            (task_b, goal_a),
            (goal_a, task_b),  # reverse of supports
            (task_c, task_b),
            (task_b, task_c),  # reverse of depends_on
        ]
    )
    assert snapshot.signature.startswith("adjacency:sha256:")
    await store.close()


@pytest.mark.asyncio
async def test_fetch_adjacency_reverse_edges_match_traversal_index_semantics(
    tmp_path: Path,
) -> None:
    """Every adjacency pair the traversal index sees must appear in the snapshot."""
    store = await _adjacency_store(tmp_path)

    snapshot = await store.fetch_adjacency("agent-1")
    traversal = store._traversal_index("agent-1")

    traversal_pairs = sorted(
        (source_key, target_key)
        for source_key, neighbors in traversal.adjacency.items()
        for target_key, _entity, _relation in neighbors
    )
    assert _snapshot_edge_pairs(snapshot) == traversal_pairs
    await store.close()


@pytest.mark.asyncio
async def test_fetch_adjacency_caches_until_projection_changes(tmp_path: Path) -> None:
    """Unchanged stores reuse the cached snapshot; upserts invalidate it."""
    store = await _adjacency_store(tmp_path)

    first = await store.fetch_adjacency("agent-1")
    second = await store.fetch_adjacency("agent-1")
    assert second is first  # cache hit on unchanged store

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Task E",
                    entity_type="task",
                    observed_at="2026-06-10T02:00:00Z",
                    summary="late arrival",
                ),
            ],
            edges=[
                ExtractedEdge(
                    source="Task E",
                    target="Task E",
                    relation_type="self_check",
                    valid_from="2026-06-10T02:00:00Z",
                ),
            ],
            source_event_seq=2,
            source_event_hash="adjacency-hash-2",
            source_event_type="task.proposed",
            source_thread="agent-1",
        ),
        session_id="agent-1",
    )

    third = await store.fetch_adjacency("agent-1")
    assert third is not first
    assert _node_key("agent-1", "task", "Task E", 2) in third.node_ids
    assert third.signature != first.signature  # signature changes with the graph
    assert third.node_count == first.node_count + 1
    await store.close()


@pytest.mark.asyncio
async def test_fetch_adjacency_signature_is_stable_for_identical_graphs(tmp_path: Path) -> None:
    """Rebuilding the snapshot for an unchanged graph reproduces the signature."""
    store = await _adjacency_store(tmp_path)

    first = await store.fetch_adjacency("agent-1")
    store._adjacency_snapshot_cache.pop("agent-1")
    rebuilt = await store.fetch_adjacency("agent-1")

    assert rebuilt is not first
    assert rebuilt.signature == first.signature
    assert rebuilt.node_ids == first.node_ids
    await store.close()


@pytest.mark.asyncio
async def test_fetch_adjacency_empty_session_returns_empty_snapshot(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    snapshot = await store.fetch_adjacency("agent-empty")

    assert snapshot.node_ids == ()
    assert snapshot.edge_count == 0
    await store.close()


@pytest.mark.asyncio
async def test_fetch_adjacency_personalized_pagerank_end_to_end(tmp_path: Path) -> None:
    """PPR over a fetched snapshot ranks the seeded neighborhood deterministically."""
    from zaxy.graph_walk import personalized_pagerank

    store = await _adjacency_store(tmp_path)
    goal_a = _node_key("agent-1", "goal", "Goal A", 1)
    task_b = _node_key("agent-1", "task", "Task B", 1)
    task_c = _node_key("agent-1", "task", "Task C", 1)
    island_d = _node_key("agent-1", "note", "Island D", 1)

    snapshot = await store.fetch_adjacency("agent-1")
    ranked = personalized_pagerank(snapshot, [goal_a], iterations=100, tol=1e-12, top_n=10)
    again = personalized_pagerank(snapshot, [goal_a], iterations=100, tol=1e-12, top_n=10)

    assert again == ranked  # deterministic
    masses = dict(ranked)
    assert set(masses) == {goal_a, task_b, task_c}  # the island gets no walk mass
    assert island_d not in masses
    # On the undirected chain A - B - C seeded at A, the middle hub B carries
    # the most mass (it receives from both sides); the seed outranks the
    # 2-hop node, which earns the least.
    assert masses[task_b] > masses[goal_a] > masses[task_c] > 0.0
    assert sum(masses.values()) == pytest.approx(1.0, abs=1e-9)
    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_connect_migrates_pre_ladybug_projection(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pre-fork (Kuzu-format) projection file is moved aside, never deleted.

    LadybugDB refuses pre-fork storage with "The file is not a valid Lbug
    database file!" (verified against a real kuzu-0.11.3 artifact; foreign
    bytes raise the identical error). connect() must move the artifact and
    its WAL to <path>.pre-ladybug.bak, open a fresh store in place, and log
    the rebuild instructions — projections rebuild from the Eventloom log.
    """
    import logging

    path = tmp_path / "embedded.kuzu"
    foreign_bytes = b"pre-fork kuzu projection bytes " * 64
    path.write_bytes(foreign_bytes)
    wal_bytes = b"pre-fork wal bytes"
    path.with_name(path.name + ".wal").write_bytes(wal_bytes)

    store = EmbeddedGraphStore(path)
    with caplog.at_level(logging.WARNING, logger="zaxy.embedded_graph_store"):
        await store.connect()
    try:
        await store.init_schema()
        await store.upsert_extraction(
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Rebuilt Goal",
                        entity_type="goal",
                        observed_at="2026-06-11T01:00:00Z",
                        summary="rebuilt after engine migration",
                    )
                ],
                edges=[],
                source_event_seq=1,
                source_event_hash="rebuild-hash",
            ),
            session_id="agent-1",
        )
        results = await store.search_exact("Rebuilt Goal", session_id="agent-1")
        assert [entity.name for entity in results] == ["Rebuilt Goal"]
    finally:
        await store.close()

    backup = path.with_name(path.name + ".pre-ladybug.bak")
    assert backup.read_bytes() == foreign_bytes
    assert backup.with_name(backup.name + ".wal").read_bytes() == wal_bytes
    assert not path.with_name(path.name + ".wal").exists() or path.exists()
    assert backup in embedded_graph_store.pre_ladybug_backup_paths(path)
    assert any("pre-ladybug.bak" in record.message for record in caplog.records)
    assert any("zaxy reproject" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_embedded_store_connect_migration_never_overwrites_existing_backup(tmp_path: Path) -> None:
    """A second migration must not clobber an earlier .pre-ladybug.bak."""
    path = tmp_path / "embedded.kuzu"
    earlier_backup = path.with_name(path.name + ".pre-ladybug.bak")
    earlier_backup.write_bytes(b"earlier backup, do not touch")
    path.write_bytes(b"another foreign projection " * 64)

    store = EmbeddedGraphStore(path)
    await store.connect()
    await store.close()

    assert earlier_backup.read_bytes() == b"earlier backup, do not touch"
    numbered = path.with_name(path.name + ".pre-ladybug.bak.1")
    assert numbered.read_bytes() == b"another foreign projection " * 64
    backups = embedded_graph_store.pre_ladybug_backup_paths(path)
    assert earlier_backup in backups and numbered in backups


def test_embedded_store_incompatible_storage_error_matcher_is_specific() -> None:
    """Only the engine's incompatible-file refusal triggers the migration."""
    assert embedded_graph_store._is_incompatible_storage_error(
        RuntimeError(
            "Runtime exception: Unable to open database. The file is not a valid Lbug database file!"
        )
    )
    assert not embedded_graph_store._is_incompatible_storage_error(
        RuntimeError("IO exception: Could not set lock on file : embedded.kuzu")
    )
    assert not embedded_graph_store._is_incompatible_storage_error(RuntimeError("disk full"))


def test_embedded_store_armor_inlines_json_shaped_string_parameters() -> None:
    """JSON-shaped string bindings are rewritten to byte-faithful literals.

    LadybugDB 0.17.1's binding layer silently converts a bound string whose
    first character is { or [ into a STRUCT/LIST and stores its re-rendering
    (write-side corruption, verified via size()). The choke point must inline
    exactly those values as escaped literals and leave every other binding
    parameterized.
    """
    query, parameters = embedded_graph_store._armor_json_shaped_string_parameters(
        "MERGE (e:Entity {node_key: $node_key}) SET e.properties_json = $properties_json",
        {"node_key": "agent\x1fgoal\x1fG\x1f1", "properties_json": '{"a": "it\'s \\\\"}'},
    )
    assert parameters == {"node_key": "agent\x1fgoal\x1fG\x1f1"}
    assert "$properties_json" not in query
    assert "'{\"a\": \"it\\'s \\\\\\\\\"}'" in query

    untouched_query = "MATCH (e:Entity) WHERE e.session_id = $session_id RETURN e.name"
    untouched_params = {"session_id": "agent-1"}
    same_query, same_params = embedded_graph_store._armor_json_shaped_string_parameters(
        untouched_query, untouched_params
    )
    assert same_query == untouched_query
    assert same_params == untouched_params

    list_query, list_params = embedded_graph_store._armor_json_shaped_string_parameters(
        "RETURN $rows", {"rows": [{"entity_row": 1}]}
    )
    assert list_query == "RETURN $rows"
    assert list_params == {"rows": [{"entity_row": 1}]}


@pytest.mark.asyncio
async def test_embedded_store_json_properties_round_trip_byte_exact(tmp_path: Path) -> None:
    """properties_json must survive the engine round trip byte-for-byte.

    Pins the armor end-to-end: without it, LadybugDB re-renders the JSON as
    a struct ('{"a": 1}' comes back '{a: 1}') and every stored embedding and
    evidence payload silently corrupts.
    """
    import json

    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version=LEGACY_EMBEDDING_VERSION,
    )
    await store.connect()
    try:
        await store.init_schema()
        properties = {"path": "C:\\dir\\file", "note": "it's", "nested": {"k": [1, 2]}}
        await store.upsert_extraction(
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="JSON Entity",
                        entity_type="artifact",
                        observed_at="2026-06-11T01:00:00Z",
                        summary="json fidelity probe",
                        embedding=[0.6, 0.8],
                        properties=properties,
                    )
                ],
                edges=[],
                source_event_seq=1,
                source_event_hash="json-hash",
            ),
            session_id="agent-1",
        )
        expected = json.dumps({**properties, "embedding": [0.6, 0.8]}, sort_keys=True)
        rows = store._execute(
            "MATCH (e:Entity) WHERE e.session_id = $session_id RETURN e.properties_json",
            {"session_id": "agent-1"},
        ).get_all()
        assert rows == [[expected]]
        results = await store.search_vector([0.6, 0.8], limit=1, session_id="agent-1")
        assert results and results[0].entity.name == "JSON Entity"
    finally:
        await store.close()


@pytest.mark.asyncio
@requires_native_vector_index
async def test_embedded_store_never_issues_single_statement_delete_all(tmp_path: Path) -> None:
    """No store path may issue the delete-ALL statement shape.

    The one residual vector-index hole on LadybugDB 0.17.1: a single-statement
    MATCH ... DELETE with no predicate under a live HNSW index permanently
    breaks subsequent searches (re-verified). Zaxy's design avoids it —
    deltas are pure insert extensions, rebuilds swap generations, superseded
    generations are dropped, invalidation closes validity windows with SET —
    so a full write-path exercise must record zero DELETE statements.
    """
    import re as _re

    dimension = 8
    store = EmbeddedGraphStore(
        tmp_path / "embedded.kuzu",
        active_embedding_version="v1",
        vector_ann_threshold=2,
    )
    await store.connect()
    await store.init_schema()
    recording = _RecordingConnection(store._require_connection())
    store._connection = recording

    for round_index in range(2):
        store._clear_read_caches("agent-1")
        store._current_entity_index_cache["agent-1"] = [
            _versioned_vector_entity(f"round{round_index}-{position}", vector, "v1")
            for position, vector in enumerate(
                _seeded_unit_embeddings(8, dimension, seed=400 + round_index)
            )
        ]
        group = store._vector_index("agent-1", None).groups[(dimension, "v1")]
        assert isinstance(group, embedded_graph_store._AnnVectorGroup)

    await store.upsert_extraction(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Guard Goal",
                    entity_type="goal",
                    observed_at="2026-06-11T01:00:00Z",
                    summary="delete-all guard",
                    properties={"source_path": "src/guard.py"},
                )
            ],
            edges=[],
            source_event_seq=1,
            source_event_hash="guard-hash",
        ),
        session_id="agent-1",
    )
    await store.invalidate_entity("Guard Goal", "goal", "2026-06-11T02:00:00Z", session_id="agent-1")
    await store.retire_source_projections(
        source_path="src/guard.py",
        invalid_at="2026-06-11T03:00:00Z",
        session_id="agent-1",
    )

    delete_statement = _re.compile(r"\bDELETE\b", _re.IGNORECASE)
    offending = [query for query in recording.queries if delete_statement.search(query)]
    assert offending == []
    assert any("DROP TABLE" in query for query in recording.queries)

    source = Path(embedded_graph_store.__file__).read_text(encoding="utf-8")
    assert "DETACH DELETE" not in source
