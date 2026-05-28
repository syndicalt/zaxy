"""Tests for the embedded graph projection backend."""

from __future__ import annotations

import builtins
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

import zaxy.embedded_graph_store as embedded_graph_store
from zaxy.embedded_graph_store import (
    EmbeddedGraphStore,
    _keyword_candidate_terms,
    _keyword_index_from_entities,
    _keyword_query_terms,
    _properties_reference_source,
    _terms,
    _vector_norm,
)
from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult
from zaxy.graph import GraphEntity

pytestmark = pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")


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
        ("agent-1", None): embedded_graph_store._VectorIndex([], [], {}, []),
        ("agent-1", "2026-05-20T01:00:00Z"): embedded_graph_store._VectorIndex([], [], {}, []),
        ("agent-2", None): embedded_graph_store._VectorIndex([], [], {}, []),
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
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
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
    index = store._vector_index_cache[("agent-1", None)]
    assert index.postings[0] == [(0, 1.0)]
    assert index.postings[1] == [(1, 1.0)]

    await store.close()


@pytest.mark.asyncio
async def test_embedded_store_zero_vector_search_skips_index_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-norm vector queries cannot match and should avoid projection index work."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    def fail_vector_index(session_id: str, temporal_point: str | None) -> object:
        raise AssertionError(f"zero vector query should not build index for {session_id}:{temporal_point}")

    monkeypatch.setattr(store, "_vector_index", fail_vector_index)

    assert await store.search_vector([0.0, 0.0], session_id="agent-1") == []


def test_embedded_vector_norm_avoids_generator_sum_hot_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedded vector scoring should avoid generator allocation per candidate vector."""
    monkeypatch.setattr(
        builtins,
        "sum",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("embedded vector norm should not allocate a generator for sum")
        ),
    )

    assert _vector_norm([3.0, 4.0]) == 5.0


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
    """Vector candidates should compute cosine score once and reuse it as raw_score."""
    store = EmbeddedGraphStore(Path("unused.kuzu"))

    class _CountingNorm:
        score_calls = 0

        def __rmul__(self, _query_norm: float) -> _CountingNorm:
            return self

        def __rtruediv__(self, _dot: float) -> float:
            self.score_calls += 1
            return 1.0

    norm = _CountingNorm()
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
            norms=[norm],
            postings={0: [(0, 1.0)]},
            dimensions=[1],
        )

    monkeypatch.setattr(store, "_vector_index", vector_index)

    results = await store.search_vector([1.0], session_id="agent-1")

    assert [result.entity.name for result in results] == ["Vector Goal"]
    assert results[0].score == 1.0
    assert results[0].raw_score == 1.0
    assert norm.score_calls == 1


@pytest.mark.asyncio
async def test_embedded_store_caches_vector_index_and_invalidates_on_projection(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
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
    assert store._vector_index_cache[("agent-1", None)].postings == {0: [(0, 1.0)]}

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
    assert index.postings == {0: [(0, 1.0)]}
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
    assert index.postings == {1: [(0, 1.0)]}
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
