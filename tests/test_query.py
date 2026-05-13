"""Tests for zaxy.query — hybrid query router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.config import Settings
from zaxy.graph import GraphEntity, GraphStore, SearchResult
from zaxy.query import (
    HTTPReranker,
    LexicalReranker,
    OpenAICompatibleReranker,
    QueryRouter,
    _mmr_rank,
    _prompt_visible_properties,
    build_reranker,
    build_retention_policy,
)


@pytest.fixture
def mock_store() -> AsyncMock:
    """Return a mock GraphStore."""
    store = AsyncMock(spec=GraphStore)
    store.search_exact = AsyncMock(return_value=[])
    store.search_keyword = AsyncMock(return_value=[])
    store.search_traversal = AsyncMock(return_value=[])
    return store


@pytest.fixture
def router(mock_store: AsyncMock) -> QueryRouter:
    """Return a QueryRouter wired to the mock store."""
    return QueryRouter(store=mock_store, default_limit=5, session_id="agent-1")


# ------------------------------------------------------------------
# Query routing tests
# ------------------------------------------------------------------

class TestQueryRouting:
    """Tests for the hybrid query pipeline."""

    async def test_empty_store_returns_empty(self, router: QueryRouter) -> None:
        """When nothing matches, return an empty list."""
        results = await router.query("something obscure")
        assert results == []

    async def test_exact_match_boosted(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Exact matches should appear with the highest weight."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="Alice",
                entity_type="user",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={},
            )
        ]
        results = await router.query("Alice")
        assert len(results) == 1
        assert results[0].source == "exact"
        assert results[0].score == 1.0
        assert mock_store.search_exact.await_args.kwargs["session_id"] == "agent-1"

    async def test_keyword_results_included(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Keyword hits should be included with keyword weight."""
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Goal1", entity_type="goal",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=1.5,
                source="keyword",
            )
        ]
        results = await router.query("ship mvp")
        assert len(results) == 1
        assert results[0].source == "keyword"
        assert results[0].score == pytest.approx(0.8)
        assert mock_store.search_keyword.await_args.kwargs["session_id"] == "agent-1"

    async def test_exact_results_are_not_drowned_by_unbounded_keyword_scores(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Exact entity matches should outrank high BM25 scores."""
        exact = GraphEntity(
            name="Goal 0003", entity_type="goal",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )
        keyword = GraphEntity(
            name="Unrelated", entity_type="task",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )

        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [exact] if name == "Goal 0003" else []

        mock_store.search_exact.side_effect = _search_exact
        mock_store.search_keyword.return_value = [
            SearchResult(entity=keyword, score=99.0, source="keyword")
        ]

        results = await router.query("Which task is connected to Goal 0003?")

        assert results[0].content.startswith("Goal 0003")

    async def test_identifier_queries_suppress_fuzzy_distractors(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Identifier-bearing source queries should not include near-miss fuzzy matches."""
        target = GraphEntity(
            name="source-recall/target/service-0000.md:10-14",
            entity_type="document",
            valid_from="2024-04-01T00:00:00Z",
            valid_to=None,
            properties={
                "summary": (
                    "source-recall/target/service-0000.md records "
                    "source_recall_answer_code=source-answer-0000."
                ),
                "source_path": "source-recall/target/service-0000.md",
            },
        )
        distractor = GraphEntity(
            name="source-recall/distractor/service-0000.md:20-24",
            entity_type="document",
            valid_from="2024-04-01T00:00:00Z",
            valid_to=None,
            properties={
                "summary": (
                    "source-recall/distractor/service-0000.md discusses a nearby "
                    "source recall incident for service-0000."
                ),
                "source_path": "source-recall/distractor/service-0000.md",
            },
        )
        mock_store.search_keyword.return_value = [
            SearchResult(entity=target, score=1.0, source="keyword")
        ]
        mock_store.search_vector.return_value = [
            SearchResult(entity=distractor, score=0.99, source="vector")
        ]

        results = await router.query(
            "Which cited source records source-answer-0000?",
            embedding=[0.1, 0.2],
            limit=5,
        )

        assert [result.entity_name for result in results] == [
            "source-recall/target/service-0000.md:10-14"
        ]

    async def test_traversal_expansion(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Traversal from keyword hits should bring in neighbors."""
        keyword_ent = GraphEntity(
            name="Alice", entity_type="user",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )
        neighbor = GraphEntity(
            name="Bob", entity_type="user",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )
        mock_store.search_keyword.return_value = [
            SearchResult(entity=keyword_ent, score=1.0, source="keyword")
        ]
        mock_store.search_traversal.return_value = [neighbor]

        results = await router.query("Alice")
        sources = {r.source for r in results}
        assert "keyword" in sources
        assert "traversal" in sources
        assert mock_store.search_traversal.await_args.kwargs["session_id"] == "agent-1"

    async def test_direct_traversal_neighbors_from_exact_hits_survive_mmr(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Direct graph neighbors from exact anchors should not be crowded out by vector hits."""
        goal = GraphEntity(
            name="Goal 0000",
            entity_type="goal",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={},
        )
        direct_task = GraphEntity(
            name="task-0000",
            entity_type="task",
            valid_from="2024-01-02T00:00:00Z",
            valid_to=None,
            properties={"summary": "task-0000 implements Goal 0000 release path"},
        )
        wrong_tasks = [
            SearchResult(
                entity=GraphEntity(
                    name=f"task-000{i}",
                    entity_type="task",
                    valid_from="2024-01-02T00:00:00Z",
                    valid_to=None,
                    properties={"summary": f"task-000{i} implements Goal 000{i} release path"},
                ),
                score=0.99,
                source="vector",
            )
            for i in range(1, 8)
        ]

        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [goal] if name == "Goal 0000" else []

        async def _search_traversal(name: str, **_: object) -> list[GraphEntity]:
            return [direct_task] if name == "Goal 0000" else []

        mock_store.search_exact.side_effect = _search_exact
        mock_store.search_vector.return_value = wrong_tasks
        mock_store.search_traversal.side_effect = _search_traversal

        results = await router.query(
            "Which task is connected to Goal 0000?",
            embedding=[0.1, 0.2],
            limit=4,
        )

        result_names = [result.content.split(" ", 1)[0] for result in results]
        assert result_names.index("task-0000") <= 1

    async def test_identifier_keyword_hits_outrank_semantic_vector_neighbors(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Exact identifiers in content should beat semantically similar vector misses."""
        correct_doc = SearchResult(
            entity=GraphEntity(
                name="docs/runbooks/service-0015.md:106-111",
                entity_type="document",
                valid_from="2024-08-01T00:00:00Z",
                valid_to=None,
                properties={
                    "summary": "service-0015 runbook uses release marker doc-code-0015.",
                    "source_path": "docs/runbooks/service-0015.md",
                    "source_start_line": 106,
                    "source_end_line": 111,
                },
            ),
            score=0.5,
            source="keyword",
        )
        wrong_doc = SearchResult(
            entity=GraphEntity(
                name="docs/runbooks/service-0013.md:92-97",
                entity_type="document",
                valid_from="2024-08-01T00:00:00Z",
                valid_to=None,
                properties={"summary": "service-0013 runbook uses release marker doc-code-0013."},
            ),
            score=0.99,
            source="vector",
        )
        mock_store.search_vector.return_value = [wrong_doc]
        mock_store.search_keyword.return_value = [correct_doc]

        results = await router.query(
            "Which runbook mentions release marker doc-code-0015?",
            embedding=[0.1, 0.2],
            limit=2,
        )

        assert results[0].content.startswith("docs/runbooks/service-0015.md:106-111")
        keyword_queries = [call.args[0] for call in mock_store.search_keyword.await_args_list]
        assert "doc-code-0015" in keyword_queries

    async def test_session_identifier_keyword_hits_outrank_other_transcript_turns(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Session id matches should pin the intended transcript turn above generic hits."""
        correct_turn = SearchResult(
            entity=GraphEntity(
                name="session-0001:turn-2",
                entity_type="transcript_turn",
                valid_from="2024-09-01T00:01:00Z",
                valid_to=None,
                properties={
                    "summary": "assistant: We decided decision-code-0001 for workstream 0001.",
                },
            ),
            score=0.5,
            source="keyword",
        )
        wrong_turn = SearchResult(
            entity=GraphEntity(
                name="session-0010:turn-2",
                entity_type="transcript_turn",
                valid_from="2024-09-01T00:01:00Z",
                valid_to=None,
                properties={
                    "summary": "assistant: We decided decision-code-0010 for workstream 0010.",
                },
            ),
            score=0.99,
            source="vector",
        )
        mock_store.search_vector.return_value = [wrong_turn]
        mock_store.search_keyword.return_value = [correct_turn]

        results = await router.query(
            "What decision code was recorded in session-0001?",
            embedding=[0.1, 0.2],
            limit=2,
        )

        assert results[0].content.startswith("session-0001:turn-2")
        keyword_queries = [call.args[0] for call in mock_store.search_keyword.await_args_list]
        assert "session-0001" in keyword_queries

    async def test_identifier_keyword_hits_limit_traversal_seeds(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Identifier-pinned keyword hits should prevent noisy traversal fan-out."""
        correct_turn = SearchResult(
            entity=GraphEntity(
                name="session-0001:turn-2",
                entity_type="transcript_turn",
                valid_from="2024-09-01T00:01:00Z",
                valid_to=None,
                properties={
                    "summary": "assistant: We decided decision-code-0001 for workstream 0001.",
                },
            ),
            score=0.5,
            source="keyword",
        )
        generic_agent = SearchResult(
            entity=GraphEntity(
                name="agent",
                entity_type="actor",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"summary": "Generic actor attached to many memories."},
            ),
            score=0.99,
            source="vector",
        )
        mock_store.search_vector.return_value = [generic_agent]
        mock_store.search_keyword.return_value = [correct_turn]

        await router.query(
            "What decision code was recorded in session-0001?",
            embedding=[0.1, 0.2],
            limit=2,
        )

        traversal_names = [
            call.args[0]
            for call in mock_store.search_traversal.await_args_list
        ]
        assert traversal_names == ["session-0001:turn-2"]

    async def test_structured_entity_names_seed_traversal(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Queries naming structured entities should exact-match them before BM25."""
        goal = GraphEntity(
            name="Goal 0003", entity_type="goal",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )
        task = GraphEntity(
            name="task-0003", entity_type="task",
            valid_from="2024-01-02T00:00:00Z", valid_to=None, properties={}
        )

        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [goal] if name == "Goal 0003" else []

        mock_store.search_exact.side_effect = _search_exact
        mock_store.search_traversal.return_value = [task]
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name=f"Unrelated {i}", entity_type="task",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=99.0,
                source="keyword",
            )
            for i in range(5)
        ]

        results = await router.query("Which task is connected to Goal 0003?", limit=2)

        assert any(result.content.startswith("Goal 0003") for result in results)
        assert any(result.content.startswith("task-0003") for result in results)

    async def test_hyphenated_identifiers_seed_exact_traversal(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Durable ids like graph-goal-0003 should seed exact graph expansion."""
        goal = GraphEntity(
            name="graph-goal-0003",
            entity_type="goal",
            valid_from="2024-03-01T00:00:00Z",
            valid_to=None,
            properties={},
        )
        task = GraphEntity(
            name="graph-task-0003",
            entity_type="task",
            valid_from="2024-03-02T00:00:00Z",
            valid_to=None,
            properties={"summary": "Implementation task for graph-goal-0003."},
        )
        finisher = GraphEntity(
            name="graph-finisher-0003",
            entity_type="actor",
            valid_from="2024-03-03T00:00:00Z",
            valid_to=None,
            properties={},
        )

        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [goal] if name == "graph-goal-0003" else []

        mock_store.search_exact.side_effect = _search_exact
        mock_store.search_traversal.return_value = [task, finisher]

        results = await router.query(
            "Which actor completed the task connected to graph-goal-0003?",
            limit=3,
        )

        exact_names = [call.args[0] for call in mock_store.search_exact.await_args_list]
        assert "graph-goal-0003" in exact_names
        assert any(result.content.startswith("graph-task-0003") for result in results)
        assert any(result.content.startswith("graph-finisher-0003") for result in results)

    async def test_completed_task_paths_outrank_planner_and_distractors(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Completion path evidence should rank above weaker task-neighbor paths."""
        goal = GraphEntity(
            name="graph-goal-0003",
            entity_type="goal",
            valid_from="2024-03-01T00:00:00Z",
            valid_to=None,
            properties={},
        )
        task = GraphEntity(
            name="graph-task-0003",
            entity_type="task",
            valid_from="2024-03-02T00:00:00Z",
            valid_to=None,
            properties={
                "summary": "Implementation task for graph-goal-0003.",
                "_path_relation_types": ["has_task"],
                "_path_length": 1,
            },
        )
        finisher = GraphEntity(
            name="graph-finisher-0003",
            entity_type="actor",
            valid_from="2024-03-03T00:00:00Z",
            valid_to=None,
            properties={
                "_path_relation_types": ["has_task", "completed_task"],
                "_path_length": 2,
            },
        )
        planner = GraphEntity(
            name="planner",
            entity_type="actor",
            valid_from="2024-03-02T00:00:00Z",
            valid_to=None,
            properties={
                "_path_relation_types": ["has_task", "proposed_task"],
                "_path_length": 2,
            },
        )
        distractor = SearchResult(
            entity=GraphEntity(
                name="graph-finisher-distractor-0003",
                entity_type="actor",
                valid_from="2024-03-04T00:00:00Z",
                valid_to=None,
                properties={},
            ),
            score=1.0,
            source="keyword",
        )

        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [goal] if name == "graph-goal-0003" else []

        async def _search_traversal(name: str, **_: object) -> list[GraphEntity]:
            if name == "graph-goal-0003":
                return [planner, task, finisher]
            return []

        mock_store.search_exact.side_effect = _search_exact
        mock_store.search_traversal.side_effect = _search_traversal
        mock_store.search_keyword.return_value = [distractor]

        results = await router.query(
            "Which actor completed the task connected to graph-goal-0003?",
            limit=3,
        )

        traversal_names = [call.args[0] for call in mock_store.search_traversal.await_args_list]
        result_names = [result.entity_name for result in results]
        assert traversal_names == ["graph-goal-0003"]
        assert "graph-finisher-0003" in result_names
        assert "graph-task-0003" in result_names
        assert "graph-finisher-distractor-0003" not in result_names

    async def test_cited_inferred_paths_outrank_uncited_inferred_paths(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Inferred traversal should prefer cited, evidenced, high-confidence edges."""
        task = GraphEntity(
            name="task-7",
            entity_type="task",
            valid_from="2024-03-01T00:00:00Z",
            valid_to=None,
            properties={},
        )
        trusted_decision = GraphEntity(
            name="decision:Use Memory Checkout",
            entity_type="decision",
            valid_from="2024-03-02T00:00:00Z",
            valid_to=None,
            properties={
                "_path_relation_types": ["likely_implemented_decision"],
                "_path_inferred_flags": [True],
                "_path_length": 1,
                "_path_inferred_edge_count": 1,
                "_path_inferred_confidences": [0.86],
                "_path_inference_methods": ["task_completed_decision_citation_v1"],
                "_path_inferred_source_event_count": 1,
                "_path_inferred_evidence_count": 2,
            },
        )
        weak_decision = GraphEntity(
            name="decision:Loose guess",
            entity_type="decision",
            valid_from="2024-03-02T00:00:00Z",
            valid_to=None,
            properties={
                "_path_relation_types": ["likely_implemented_decision"],
                "_path_inferred_flags": [True],
                "_path_length": 1,
                "_path_inferred_edge_count": 1,
                "_path_inferred_confidences": [0.86],
                "_path_inference_methods": ["unknown"],
                "_path_inferred_source_event_count": 0,
                "_path_inferred_evidence_count": 0,
            },
        )

        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [task] if name == "task-7" else []

        mock_store.search_exact.side_effect = _search_exact
        mock_store.search_traversal.return_value = [weak_decision, trusted_decision]

        results = await router.query("Which decision did task-7 implement?", limit=3)

        result_names = [result.entity_name for result in results]
        assert result_names.index("decision:Use Memory Checkout") < result_names.index(
            "decision:Loose guess"
        )
        trusted = next(result for result in results if result.entity_name == "decision:Use Memory Checkout")
        weak = next(result for result in results if result.entity_name == "decision:Loose guess")
        assert trusted.score > weak.score
        assert trusted.score_explanation is not None
        assert trusted.score_explanation["inferred_edge_trust"] == pytest.approx(0.86)
        assert trusted.score_explanation["inferred_edge_trust_multiplier"] > 1.0
        assert trusted.score_explanation["inferred_edge_evidence_coverage"] == 1.0
        assert trusted.score_explanation["inferred_edge_source_coverage"] == 1.0
        assert trusted.score_explanation["inferred_relation_types"] == [
            "likely_implemented_decision"
        ]
        assert trusted.score_explanation["inference_methods"] == [
            "task_completed_decision_citation_v1"
        ]
        assert weak.score_explanation is not None
        assert weak.score_explanation["inferred_edge_trust_multiplier"] < 1.0
        assert weak.score_explanation["inferred_edge_evidence_coverage"] == 0.0

    async def test_preference_queries_exact_match_structured_preference_entity(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Preference questions should target user:key entities, not only users."""
        preference = GraphEntity(
            name="user-0003:theme",
            entity_type="preference",
            valid_from="2024-06-01T00:00:00Z",
            valid_to=None,
            properties={"summary": "theme=theme-new-3"},
        )

        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [preference] if name == "user-0003:theme" else []

        mock_store.search_exact.side_effect = _search_exact

        results = await router.query("What is the current theme preference for user-0003?")

        exact_names = [
            call.args[0]
            for call in mock_store.search_exact.await_args_list
        ]
        assert "user-0003:theme" in exact_names
        assert results[0].content.startswith("user-0003:theme")
        assert "theme=theme-new-3" in results[0].content

    async def test_temporal_preference_queries_use_same_structured_preference_anchor(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Historical preference questions should exact-match the temporal fact anchor."""
        preference = GraphEntity(
            name="user-0004:theme",
            entity_type="preference",
            valid_from="2024-02-01T00:00:00Z",
            valid_to="2024-06-01T00:00:00Z",
            properties={"summary": "theme=theme-old-4"},
        )

        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [preference] if name == "user-0004:theme" else []

        mock_store.search_exact.side_effect = _search_exact

        await router.query(
            "What was the theme preference for user-0004 in March 2024?",
            temporal_point="2024-03-01T00:00:00Z",
        )

        matching_call = next(
            call
            for call in mock_store.search_exact.await_args_list
            if call.args[0] == "user-0004:theme"
        )
        assert matching_call.kwargs["temporal_point"] == "2024-03-01T00:00:00Z"

    async def test_mixed_current_queries_suppress_superseded_preference_versions(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Mixed context should keep current facts without leaking stale preference values."""
        current_preference = GraphEntity(
            name="user-0003:theme",
            entity_type="preference",
            valid_from="2024-06-01T00:00:00Z",
            valid_to=None,
            properties={"summary": "theme=theme-new-3"},
        )
        stale_preference = GraphEntity(
            name="user-0003:theme",
            entity_type="preference",
            valid_from="2024-02-01T00:00:00Z",
            valid_to="2024-06-01T00:00:00Z",
            properties={"summary": "theme=theme-old-3"},
        )
        document = GraphEntity(
            name="docs/runbooks/service-0003.md:22-27",
            entity_type="document",
            valid_from="2024-08-01T00:00:00Z",
            valid_to=None,
            properties={
                "summary": "service-0003 production runbook uses release marker doc-code-0003.",
                "source_path": "docs/runbooks/service-0003.md",
            },
        )
        async def _search_exact(name: str, **_: object) -> list[GraphEntity]:
            return [current_preference] if name == "user-0003:theme" else []

        async def _search_keyword(query: str, **_: object) -> list[SearchResult]:
            results = [
                SearchResult(entity=stale_preference, score=0.99, source="keyword"),
                SearchResult(entity=document, score=0.95, source="keyword"),
            ]
            if "doc-code-0003" in query:
                return [results[1]]
            return results

        mock_store.search_exact.side_effect = _search_exact
        mock_store.search_keyword.side_effect = _search_keyword

        chunks = await router.query(
            "For user-0003 in workstream 0003, recover the current theme, "
            "runbook marker doc-code-0003, and session decision.",
            limit=10,
        )
        content = "\n".join(chunk.content for chunk in chunks)

        assert "theme=theme-new-3" in content
        assert "doc-code-0003" in content
        assert "theme=theme-old-3" not in content

    async def test_deduplication_keeps_highest_score(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """If the same entity appears from multiple sources, keep the highest score."""
        ent = GraphEntity(
            name="Alice", entity_type="user",
            valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
        )
        mock_store.search_exact.return_value = [ent]
        mock_store.search_keyword.return_value = [
            SearchResult(entity=ent, score=0.5, source="keyword")
        ]

        results = await router.query("Alice")
        assert len(results) == 1
        # exact weight is 1.0, keyword weight is 0.5*0.8=0.4 -> exact wins
        assert results[0].score == 1.0

    async def test_limit_truncates(self, mock_store: AsyncMock) -> None:
        """Result list should be truncated to the limit."""
        router = QueryRouter(store=mock_store, default_limit=2, session_id="agent-1")
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name=f"E{i}", entity_type="x",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=float(i),
                source="keyword",
            )
            for i in range(10)
        ]
        results = await router.query("many")
        assert len(results) == 2

    async def test_temporal_filter_passed_through(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Temporal point should be forwarded to all search methods."""
        await router.query("x", temporal_point="2024-03-01T00:00:00Z")
        mock_store.search_exact.assert_awaited_once()
        assert mock_store.search_exact.await_args.kwargs["temporal_point"] == "2024-03-01T00:00:00Z"
        assert mock_store.search_exact.await_args.kwargs["session_id"] == "agent-1"
        mock_store.search_keyword.assert_awaited_once()
        assert mock_store.search_keyword.await_args.kwargs["temporal_point"] == "2024-03-01T00:00:00Z"
        assert mock_store.search_keyword.await_args.kwargs["session_id"] == "agent-1"

    async def test_vector_search_is_session_scoped(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Vector search should be constrained to the router session."""
        await router.query("Alice", embedding=[0.1, 0.2])

        mock_store.search_vector.assert_awaited_once()
        assert mock_store.search_vector.await_args.kwargs["session_id"] == "agent-1"

    async def test_results_sorted_by_score(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Results should be ordered by descending score."""
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Low", entity_type="x",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=0.1,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="High", entity_type="x",
                    valid_from="2024-01-01T00:00:00Z", valid_to=None, properties={}
                ),
                score=2.0,
                source="keyword",
            ),
        ]
        results = await router.query("x")
        assert results[0].content.startswith("High")
        assert results[1].content.startswith("Low")

    async def test_retrieval_salience_boosts_compact_memory(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """High-salience memory artifacts should outrank generic matching documents."""
        generic = GraphEntity(
            name="docs/generic-yoga.md:1-20",
            entity_type="document",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={"summary": "yoga classes yoga classes yoga classes in my area"},
        )
        memory = GraphEntity(
            name="memory/serenity-yoga.md:1-6",
            entity_type="document",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={
                "summary": "I cannot make it to Serenity Yoga today.",
                "retrieval_salience": 4.0,
            },
        )
        mock_store.search_keyword.return_value = [
            SearchResult(entity=generic, score=1.0, source="keyword"),
            SearchResult(entity=memory, score=0.4, source="keyword"),
        ]

        results = await router.query("Where do I take yoga classes?", limit=2)

        assert results[0].entity_name == "memory/serenity-yoga.md:1-6"

    async def test_ranking_uses_mmr_to_diversify_near_duplicate_hits(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Near-duplicate hits should not crowd out useful adjacent context."""
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Alpha API design",
                    entity_type="doc",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": "Alpha API design reference"},
                ),
                score=1.0,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="Alpha API design notes",
                    entity_type="doc",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": "Alpha API design draft notes"},
                ),
                score=0.99,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="Billing rollout task",
                    entity_type="task",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": "Billing rollout depends on Alpha"},
                ),
                score=0.75,
                source="keyword",
            ),
        ]

        results = await router.query("Alpha API", limit=2)

        assert [r.content.split(" (", 1)[0] for r in results] == [
            "Alpha API design",
            "Billing rollout task",
        ]

    def test_mmr_ranking_tokenizes_each_candidate_once(self) -> None:
        """MMR should not repeatedly retokenize the same graph entities."""
        results = [
            SearchResult(
                entity=GraphEntity(
                    name=f"memory-{index:04d}",
                    entity_type="document",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": f"memory {index} shared context"},
                ),
                score=1.0 - (index * 0.01),
                source="keyword",
            )
            for index in range(20)
        ]
        tokenized_names: list[str] = []

        def _tokens(entity: GraphEntity) -> set[str]:
            tokenized_names.append(entity.name)
            return {entity.name, entity.entity_type}

        with patch("zaxy.query._entity_tokens", side_effect=_tokens):
            ranked = _mmr_rank(results, limit=5)

        assert len(ranked) == 5
        assert len(tokenized_names) == len(results)

    async def test_keyword_query_expansion_adds_domain_synonyms(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Keyword search should broaden terse agent queries with known synonyms."""
        await router.query("auth decision", limit=3)

        searched = [call.args[0] for call in mock_store.search_keyword.await_args_list]
        assert searched[0] == "auth decision"
        assert any("authentication" in query for query in searched[1:])
        assert any("authorization" in query for query in searched[1:])
        assert any("rationale" in query for query in searched[1:])

    async def test_temporal_point_boosts_facts_asserted_closer_to_that_time(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """As-of queries should prefer the active fact version closest to the point in time."""
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Legacy architecture decision",
                    entity_type="decision",
                    valid_from="2023-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=1.0,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="Current architecture decision",
                    entity_type="decision",
                    valid_from="2024-02-20T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=1.0,
                source="keyword",
            ),
        ]

        results = await router.query(
            "architecture decision",
            temporal_point="2024-03-01T00:00:00Z",
            limit=2,
        )

        assert results[0].content.startswith("Current architecture decision")
        assert results[0].score_explanation is not None
        assert results[0].score_explanation["temporal_score"] > results[1].score_explanation["temporal_score"]

    async def test_named_scoring_profile_changes_weights_and_explains_profile(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Scoring profiles should make retrieval policy explicit and auditable."""
        router = QueryRouter(store=mock_store, default_limit=5, session_id="agent-1", scoring_profile="precision")
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Auth decision",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=1.0,
                source="keyword",
            )
        ]

        results = await router.query("auth decision")

        assert results[0].score == pytest.approx(0.65)
        assert results[0].score_explanation is not None
        assert results[0].score_explanation["scoring_profile"] == "precision"
        assert results[0].score_explanation["source_weight"] == 0.65

    def test_custom_fusion_weights_override_individual_sources(self, mock_store: AsyncMock) -> None:
        """Partial fusion overrides should preserve unspecified source weights."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            fusion_weights={"keyword": 0.2},
        )

        assert router.fusion_weights["keyword"] == 0.2
        assert router.fusion_weights["exact"] == 1.0
        assert router.fusion_weights["traversal"] == 0.9

    async def test_filter_expired_retention_policy_hides_expired_results(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Expired memories should remain in storage but be filtered at retrieval time."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            retention_policy="filter_expired",
        )
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Expired decision",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"expires_at": "2024-02-01T00:00:00Z"},
                ),
                score=1.0,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="Current decision",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"expires_at": "2024-05-01T00:00:00Z"},
                ),
                score=0.9,
                source="keyword",
            ),
        ]

        results = await router.query("decision", temporal_point="2024-03-01T00:00:00Z", limit=5)

        assert [result.content.split(" (", 1)[0] for result in results] == ["Current decision"]
        assert results[0].score_explanation is not None
        assert results[0].score_explanation["retention_policy"] == "filter_expired"

    async def test_decay_retention_policy_downranks_stale_results(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Decay policy should reduce stale scores without deleting history."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            retention_policy="decay",
            retention_decay_half_life_days=30,
            retention_expired_weight=0.2,
        )
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Old decision",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=1.0,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="Recent decision",
                    entity_type="decision",
                    valid_from="2024-02-25T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=0.9,
                source="keyword",
            ),
        ]

        results = await router.query("decision", temporal_point="2024-03-01T00:00:00Z", limit=2)

        assert results[0].content.startswith("Recent decision")
        assert results[0].score_explanation is not None
        assert results[0].score_explanation["retention_policy"] == "decay"
        assert results[1].score_explanation is not None
        assert results[1].score_explanation["retention_decay_multiplier"] < 1.0

    def test_build_retention_policy_uses_configured_defaults(self) -> None:
        """Retention policy config should be explicit and auditable."""
        policy = build_retention_policy(
            Settings(
                _env_file=None,
                retention_policy="decay",
                retention_decay_half_life_days=14,
                retention_expired_weight=0.25,
            )
        )

        assert policy.mode == "decay"
        assert policy.decay_half_life_days == 14
        assert policy.expired_weight == 0.25

    async def test_lexical_reranker_can_promote_best_text_match(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """A reranker provider should get the fused candidates before final truncation."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            reranker=LexicalReranker(),
        )
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="General auth note",
                    entity_type="document",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": "Authentication background"},
                ),
                score=1.0,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="Auth decision",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": "Auth decision rationale"},
                ),
                score=0.9,
                source="keyword",
            ),
        ]

        results = await router.query("auth decision rationale", limit=1)

        assert results[0].content.startswith("Auth decision")
        assert results[0].score_explanation is not None
        assert results[0].score_explanation["reranker"] == "lexical"
        assert results[0].score_explanation["rerank_score"] > 0

    async def test_vector_failure_falls_back_to_keyword_results(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Vector outages should not prevent keyword/exact retrieval."""
        mock_store.search_vector.side_effect = RuntimeError("vector index unavailable")
        metrics = MagicMock()
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Auth decision",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=1.0,
                source="keyword",
            )
        ]

        with patch("zaxy.query.get_metrics", return_value=metrics):
            results = await router.query("auth decision", embedding=[0.1, 0.2])

        assert results[0].content.startswith("Auth decision")
        assert results[0].score_explanation is not None
        assert "vector search unavailable" in results[0].score_explanation["warnings"]
        metrics.record_degraded_operation.assert_called_with("query", "vector_search_unavailable")

    async def test_reranker_failure_falls_back_to_mmr_order(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Reranker outages should degrade to the built-in MMR ranking."""

        class BrokenReranker:
            async def rerank(
                self,
                query: str,
                results: list[SearchResult],
                *,
                limit: int,
            ) -> list[SearchResult]:
                raise RuntimeError("reranker down")

        router = QueryRouter(store=mock_store, default_limit=5, session_id="agent-1", reranker=BrokenReranker())
        metrics = MagicMock()
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Auth decision",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=1.0,
                source="keyword",
            )
        ]

        with patch("zaxy.query.get_metrics", return_value=metrics):
            results = await router.query("auth decision")

        assert results[0].content.startswith("Auth decision")
        assert results[0].score_explanation is not None
        assert "reranker unavailable" in results[0].score_explanation["warnings"]
        metrics.record_degraded_operation.assert_called_with("query", "reranker_unavailable")


class TestModelRerankers:
    """Tests for hosted and local model reranker providers."""

    async def test_openai_compatible_reranker_posts_chat_completion_request(self) -> None:
        """OpenAI-compatible reranking should use a fakeable chat-completions client."""
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "choices": [
                        {"message": {"content": '[{"index": 1, "score": 0.95}, {"index": 0, "score": 0.1}]'}}
                    ]
                }

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return FakeResponse()

        low = SearchResult(
            entity=GraphEntity(
                name="General note",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"summary": "Background"},
            ),
            score=0.8,
            source="keyword",
        )
        high = SearchResult(
            entity=GraphEntity(
                name="Auth decision",
                entity_type="decision",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"summary": "Auth decision rationale"},
            ),
            score=0.7,
            source="keyword",
        )
        reranker = OpenAICompatibleReranker(
            api_key="test-key",
            model="gpt-test",
            client=FakeClient(),
        )

        results = await reranker.rerank("auth decision rationale", [low, high], limit=1)

        assert results[0].entity.name == "Auth decision"
        assert results[0].reranker == "openai-compatible"
        assert results[0].rerank_score == pytest.approx(0.95)
        assert captured["url"] == "https://api.openai.com/v1/chat/completions"
        assert captured["headers"] == {"Authorization": "Bearer test-key"}
        body = captured["json"]
        assert isinstance(body, dict)
        assert body["model"] == "gpt-test"
        assert body["temperature"] == 0

    async def test_http_reranker_promotes_scores_from_local_endpoint(self) -> None:
        """Local HTTP rerankers should accept endpoint scores without network in tests."""
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"scores": [0.1, 0.9]}

        class FakeClient:
            def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: dict[str, object],
            ) -> FakeResponse:
                captured["url"] = url
                captured["headers"] = headers
                captured["json"] = json
                return FakeResponse()

        results = [
            SearchResult(
                entity=GraphEntity(
                    name="General note",
                    entity_type="document",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=0.8,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="Auth decision",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=0.7,
                source="keyword",
            ),
        ]
        reranker = HTTPReranker(
            endpoint="http://localhost:11434/rerank",
            api_key="local-key",
            client=FakeClient(),
        )

        reranked = await reranker.rerank("auth decision", results, limit=1)

        assert reranked[0].entity.name == "Auth decision"
        assert reranked[0].reranker == "http"
        assert captured["url"] == "http://localhost:11434/rerank"
        assert captured["headers"] == {"Authorization": "Bearer local-key"}

    def test_build_reranker_uses_configured_provider(self) -> None:
        """Reranker factory should build the configured provider."""
        lexical = build_reranker(Settings(_env_file=None, reranker_provider="lexical"))
        assert isinstance(lexical, LexicalReranker)

        http = build_reranker(
            Settings(
                _env_file=None,
                reranker_provider="http",
                reranker_url="http://localhost:11434/rerank",
            )
        )
        assert isinstance(http, HTTPReranker)

        openai = build_reranker(
            Settings(
                _env_file=None,
                reranker_provider="openai",
                openai_api_key="test-key",
            )
        )
        assert isinstance(openai, OpenAICompatibleReranker)

    def test_build_reranker_rejects_missing_required_config(self) -> None:
        """Hosted/local reranker providers should fail loudly when misconfigured."""
        with pytest.raises(ValueError, match="RERANKER_URL"):
            build_reranker(Settings(_env_file=None, reranker_provider="http"))

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            build_reranker(Settings(_env_file=None, reranker_provider="openai", openai_api_key=None))


# ------------------------------------------------------------------
# Context chunk tests
# ------------------------------------------------------------------

class TestContextChunk:
    """Tests for the output formatting."""

    async def test_chunk_content_format(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Chunks should contain entity name, type, and a few properties."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="Alice",
                entity_type="user",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"theme": "dark", "lang": "en", "region": "us", "extra": "ignored"},
            )
        ]
        results = await router.query("Alice")
        assert len(results) == 1
        chunk = results[0]
        assert "Alice" in chunk.content
        assert "user" in chunk.content
        assert "theme=dark" in chunk.content
        assert "lang=en" in chunk.content
        assert "region=us" in chunk.content
        # Should only include first few properties
        assert "extra=ignored" not in chunk.content

    async def test_chunk_content_preserves_source_identity_fields(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Source/session identity should not be truncated out of graph context."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="longmemeval/75499fd8/answer_723bf11f/chunk-0001.md:1-9",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={
                    "source_path": "longmemeval/75499fd8/answer_723bf11f/chunk-0001.md",
                    "source_start_line": 1,
                    "source_end_line": 9,
                    "longmemeval_session_id": "answer_723bf11f",
                    "summary": "Max is a Golden Retriever.",
                    "extra": "may be omitted",
                },
            )
        ]

        results = await router.query("What breed is Max?")

        assert "longmemeval_session_id=answer_723bf11f" in results[0].content
        assert "summary=Max is a Golden Retriever." in results[0].content
        assert "source_path=longmemeval/75499fd8/answer_723bf11f/chunk-0001.md" in results[0].content
        assert "source_start_line=1" in results[0].content

    async def test_chunk_exposes_entity_identity(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Feedback APIs should be able to reinforce the retrieved graph entity."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="Ship MVP",
                entity_type="goal",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={},
            )
        ]

        results = await router.query("Ship MVP")

        assert results[0].entity_name == "Ship MVP"
        assert results[0].entity_type == "goal"

    async def test_chunk_content_omits_embedding_vectors(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Chunks should not leak raw embedding vectors into prompt context."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="Alice",
                entity_type="user",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"embedding": [0.1, 0.2, 0.3], "summary": "Agent owner"},
            )
        ]

        results = await router.query("Alice")

        assert "embedding=" not in results[0].content
        assert "0.1" not in results[0].content
        assert "summary=Agent owner" in results[0].content

    async def test_chunk_validity_window(self, router: QueryRouter, mock_store: AsyncMock) -> None:
        """Chunks should preserve temporal validity."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="OldFact",
                entity_type="fact",
                valid_from="2024-01-01T00:00:00Z",
                valid_to="2024-06-01T00:00:00Z",
                properties={},
            )
        ]
        results = await router.query("OldFact")
        assert results[0].valid_from == "2024-01-01T00:00:00Z"
        assert results[0].valid_to == "2024-06-01T00:00:00Z"

    async def test_chunk_includes_event_citation(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Chunks should cite the originating Eventloom event when available."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="Ship MVP",
                entity_type="goal",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                session_id="agent-1",
                properties={
                    "source_event_seq": 42,
                    "source_event_hash": "abcdef1234567890" * 4,
                },
            )
        ]

        results = await router.query("Ship MVP")

        assert results[0].citation == "eventloom://agent-1/events/42#abcdef123456"

    async def test_chunk_prefers_file_line_citation_for_document_sources(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Document chunks should cite their source path and line."""
        mock_store.search_exact.return_value = [
            GraphEntity(
                name="docs/guide.md:4-8",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                session_id="agent-1",
                properties={
                    "source_path": "docs/guide.md",
                    "source_start_line": 4,
                    "source_end_line": 8,
                    "source_event_seq": 42,
                    "source_event_hash": "abcdef1234567890" * 4,
                },
            )
        ]

        results = await router.query("docs/guide.md:4-8")

        assert results[0].citation == "file://docs/guide.md:4"

    async def test_chunk_includes_explainable_score_metadata(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Chunks should expose enough score metadata to debug ranking."""
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Ship MVP",
                    entity_type="goal",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=1.5,
                source="keyword",
            )
        ]

        results = await router.query("Ship MVP")

        assert results[0].score_explanation == {
            "source": "keyword",
            "raw_score": 1.5,
            "source_weight": 0.8,
            "scoring_profile": "balanced",
            "query_weight": 1.0,
            "matched_query": "Ship MVP",
            "weighted_score": 0.8,
            "ranking_score": 0.8,
        }


def test_prompt_visible_properties_prioritize_domain_identity() -> None:
    """Domain identity should remain visible before source audit metadata."""
    visible = _prompt_visible_properties(
        {
            "summary": "Design landing page for Ship MVP.",
            "source_event_seq": 2,
            "source_event_hash": "abc",
            "source_thread": "agent-1",
            "taskId": "task-0001",
        }
    )

    assert ("taskId", "task-0001") in visible
    assert visible.index(("taskId", "task-0001")) < visible.index(("source_event_seq", 2))
