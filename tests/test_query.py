"""Tests for zaxy.query — hybrid query router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.config import Settings
from zaxy.graph import GraphEntity, GraphStore, SearchResult
from zaxy.query import (
    HTTPReranker,
    LateInteractionHTTPReranker,
    LexicalReranker,
    OpenAICompatibleReranker,
    QueryRouter,
    _entity_matches_identifier,
    _entity_similarity,
    _entity_tokens,
    _exact_candidates,
    _expanded_queries,
    _identifier_boosted_score,
    _inferred_edge_trust_metadata,
    _looks_like_durable_identifier,
    _mmr_rank,
    _prompt_visible_properties,
    _structured_preference_candidates,
    _suppress_identifier_fuzzy_distractors,
    _to_chunk,
    _tokens,
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

    def test_context_chunk_preserves_skill_memory_metadata(self) -> None:
        """Skill Memory analytics need structured graph properties after query routing."""
        chunk = _to_chunk(
            SearchResult(
                entity=GraphEntity(
                    name="skill:python-test-first:v2",
                    entity_type="skill_version",
                    valid_from="2026-05-17T00:00:00Z",
                    valid_to=None,
                    properties={
                        "skill_id": "python-test-first",
                        "version": "2",
                        "status": "validated",
                        "procedure": ["Write failing test", "Run pytest"],
                        "success_score": 0.96,
                    },
                ),
                score=0.9,
                source="keyword",
            )
        )

        assert chunk.metadata == {
            "skill_id": "python-test-first",
            "version": "2",
            "status": "validated",
            "procedure": ["Write failing test", "Run pytest"],
            "success_score": 0.96,
        }

    def test_context_chunk_does_not_copy_document_payload_metadata(self) -> None:
        """Large document payloads should stay in content, not hidden checkout metadata."""
        chunk = _to_chunk(
            SearchResult(
                entity=GraphEntity(
                    name="doc-1",
                    entity_type="document",
                    valid_from="2026-05-17T00:00:00Z",
                    valid_to=None,
                    properties={
                        "content": "large document body" * 500,
                        "source_path": "docs/large.md",
                        "sha256": "abc123",
                    },
                ),
                score=0.9,
                source="keyword",
            )
        )

        assert chunk.metadata is None

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

    async def test_evidence_plan_expansions_feed_raw_graph_keyword_retrieval(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Raw graph retrieval should share deterministic evidence-aware query expansion."""
        target = GraphEntity(
            name="bike-expense-ledger",
            entity_type="document",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={"summary": "bike helmet chain lights rack tune-up cost total $185"},
        )

        async def _search_keyword(query: str, **_: object) -> list[SearchResult]:
            if "bike bicycle helmet chain lights rack tune-up cost" in query:
                return [SearchResult(entity=target, score=1.0, source="keyword")]
            return []

        mock_store.search_keyword.side_effect = _search_keyword

        results = await router.query(
            "How much total money have I spent on bike-related expenses since the start of the year?"
        )

        assert [result.entity_name for result in results] == ["bike-expense-ledger"]
        keyword_queries = [call.args[0] for call in mock_store.search_keyword.await_args_list]
        assert any("bike bicycle helmet chain lights rack tune-up cost" in query for query in keyword_queries)

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

    async def test_natural_hyphenated_terms_do_not_suppress_expansion_hits(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Hyphenated adjectives like art-related are not durable identifiers."""
        distractor = SearchResult(
            entity=GraphEntity(
                name="generic-art-note",
                entity_type="document",
                valid_from="2024-04-01T00:00:00Z",
                valid_to=None,
                properties={"summary": "A generic note about art-related searches."},
            ),
            score=2.0,
            source="keyword",
        )
        target = SearchResult(
            entity=GraphEntity(
                name="event-ledger",
                entity_type="document",
                valid_from="2024-04-01T00:00:00Z",
                valid_to=None,
                properties={
                    "summary": (
                        "Children's Museum Art Afternoon guided tour "
                        "History Museum Art Gallery lecture street art."
                    ),
                },
            ),
            score=1.0,
            source="keyword",
        )

        async def _search_keyword(query: str, **_: object) -> list[SearchResult]:
            if "Children's Museum" in query:
                return [target]
            return [distractor]

        mock_store.search_keyword.side_effect = _search_keyword

        results = await router.query(
            "How many different art-related events did I attend in the past month?",
            limit=5,
        )

        assert "event-ledger" in [result.entity_name for result in results]

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

    async def test_traversal_skipped_when_backend_has_no_relationship_edges(self) -> None:
        """Routers should not issue per-seed traversal queries against edge-empty projections."""
        store = AsyncMock()
        store.search_exact = AsyncMock(return_value=[])
        store.search_vector = AsyncMock(return_value=[])
        store.search_keyword = AsyncMock(
            return_value=[
                SearchResult(
                    entity=GraphEntity(
                        name="session-0001:turn-2",
                        entity_type="transcript_turn",
                        valid_from="2024-09-01T00:01:00Z",
                        valid_to=None,
                        properties={"summary": "assistant: We decided decision-code-0001."},
                    ),
                    score=1.0,
                    source="keyword",
                )
            ]
        )
        store.search_traversal = AsyncMock(return_value=[])
        store.has_traversal_edges = AsyncMock(return_value=False)
        router = QueryRouter(store=store, default_limit=5, session_id="agent-1")

        results = await router.query("What decision code was recorded in session-0001?")

        assert [result.source for result in results] == ["keyword"]
        store.has_traversal_edges.assert_awaited_once_with(session_id="agent-1")
        store.search_traversal.assert_not_awaited()

    async def test_temporal_traversal_does_not_depend_on_current_edge_gate(self) -> None:
        """Historical queries should still attempt traversal after current edges retire."""
        anchor = GraphEntity(
            name="Guide Claim",
            entity_type="fact",
            valid_from="2026-05-20T01:00:00Z",
            valid_to="2026-05-20T02:00:00Z",
            properties={"summary": "claim from retired source"},
        )
        neighbor = GraphEntity(
            name="Stable Goal",
            entity_type="goal",
            valid_from="2026-05-20T01:00:00Z",
            valid_to=None,
            properties={"summary": "goal remains active"},
        )
        store = AsyncMock()
        store.search_exact = AsyncMock(return_value=[anchor])
        store.search_vector = AsyncMock(return_value=[])
        store.search_keyword = AsyncMock(return_value=[])
        store.search_traversal = AsyncMock(return_value=[neighbor])
        store.has_traversal_edges = AsyncMock(return_value=False)
        router = QueryRouter(store=store, default_limit=5, session_id="agent-1")

        results = await router.query(
            "Guide Claim",
            temporal_point="2026-05-20T01:30:00Z",
        )

        assert "Stable Goal" in [result.entity_name for result in results]
        store.has_traversal_edges.assert_not_awaited()
        store.search_traversal.assert_awaited_once_with(
            "Guide Claim",
            depth=2,
            temporal_point="2026-05-20T01:30:00Z",
            session_id="agent-1",
        )

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

    async def test_coordinate_proof_query_surfaces_artifact_candidate_and_ledger_row(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Mission proof traversal should expose the composed answer proof graph."""
        mission = GraphEntity(
            name="release-rc1",
            entity_type="mission",
            valid_from="2026-06-02T00:00:00Z",
            valid_to=None,
            properties={"summary": "Ship release candidate with accepted findings."},
        )
        proof = GraphEntity(
            name="sha256:proof",
            entity_type="coordination_proof_packet",
            valid_from="2026-06-02T00:01:00Z",
            valid_to=None,
            properties={
                "summary": "Compose accepted release findings.",
                "decision_scope": "handoff",
                "authority_scope": "parent_accepted_state",
                "_path_relation_types": ["mission_has_proof_packet"],
                "_path_length": 1,
            },
        )
        artifact = GraphEntity(
            name="sha256:proof",
            entity_type="synthesis_artifact",
            valid_from="2026-06-02T00:01:01Z",
            valid_to=None,
            properties={
                "answer_candidate_count": 1,
                "ledger_row_count": 1,
                "_path_relation_types": ["mission_has_proof_packet", "proof_links_synthesis_artifact"],
                "_path_length": 2,
            },
        )
        candidate = GraphEntity(
            name="sha256:proof:candidate:coordinate_handoff_answer",
            entity_type="synthesis_answer_candidate",
            valid_from="2026-06-02T00:01:02Z",
            valid_to=None,
            properties={
                "summary": "Accepted cause: expired JWKS cache.",
                "answer_key": "coordinate_handoff_answer",
                "_path_relation_types": [
                    "mission_has_proof_packet",
                    "proof_links_synthesis_artifact",
                    "artifact_has_answer_candidate",
                ],
                "_path_length": 3,
            },
        )
        row = GraphEntity(
            name="sha256:proof:ledger:auth-api:finding:1",
            entity_type="synthesis_ledger_row",
            valid_from="2026-06-02T00:01:03Z",
            valid_to=None,
            properties={
                "fact_id": "auth-api:finding:1",
                "source_group": "auth-api:finding:1",
                "include_reason": "accepted_parent_state",
                "_path_relation_types": [
                    "mission_has_proof_packet",
                    "proof_links_synthesis_artifact",
                    "artifact_has_ledger_row",
                ],
                "_path_length": 3,
            },
        )

        mock_store.search_keyword.return_value = [
            SearchResult(entity=mission, score=1.0, source="keyword")
        ]
        mock_store.search_traversal.return_value = [proof, artifact, candidate, row]
        mock_store.has_traversal_edges = AsyncMock(return_value=True)

        results = await router.query("release-rc1 handoff proof accepted findings", limit=8)

        by_type = {result.entity_type: result for result in results}
        assert "coordination_proof_packet" in by_type
        assert "synthesis_artifact" in by_type
        assert "synthesis_answer_candidate" in by_type
        assert "synthesis_ledger_row" in by_type
        assert by_type["synthesis_answer_candidate"].source == "traversal"
        assert "Accepted cause: expired JWKS cache." in by_type["synthesis_answer_candidate"].content
        assert by_type["synthesis_ledger_row"].score_explanation is not None
        assert by_type["synthesis_ledger_row"].score_explanation["path_relation_types"] == [
            "mission_has_proof_packet",
            "proof_links_synthesis_artifact",
            "artifact_has_ledger_row",
        ]
        mock_store.search_traversal.assert_awaited()

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

    async def test_zero_vector_skips_vector_search(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Zero-norm query embeddings should not call the projection backend."""
        await router.query("Alice", embedding=[0.0, 0.0])

        mock_store.search_vector.assert_not_awaited()

    async def test_router_overfetches_candidates_before_final_limit(
        self,
        router: QueryRouter,
        mock_store: AsyncMock,
    ) -> None:
        """Retrieval should collect enough candidates for reranking before truncating prompt context."""
        entities = [
            GraphEntity(
                name=f"candidate-{index}",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={},
            )
            for index in range(3)
        ]
        mock_store.search_keyword.return_value = [
            SearchResult(entity=entity, score=1.0 - (index * 0.1), source="keyword")
            for index, entity in enumerate(entities)
        ]

        results = await router.query("candidate ranking", limit=2, embedding=[0.1, 0.2])

        assert len(results) == 2
        assert mock_store.search_keyword.await_args.kwargs["limit"] > 2
        assert mock_store.search_vector.await_args.kwargs["limit"] > 2

    async def test_local_lexical_reranker_promotes_strong_overlap_over_vector_noise(self) -> None:
        """Local-first retrieval should let exact query evidence beat high-score semantic noise."""
        reranker = LexicalReranker()
        noisy = SearchResult(
            entity=GraphEntity(
                name="unrelated salient turn",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"summary": "A generic memory about a different topic."},
            ),
            score=1.7,
            source="vector",
        )
        relevant = SearchResult(
            entity=GraphEntity(
                name="house search with Rachel",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={
                    "summary": "I found a house I loved after starting to work with Rachel."
                },
            ),
            score=1.15,
            source="keyword",
        )

        results = await reranker.rerank(
            "How many days did it take to find a house I loved after starting to work with Rachel?",
            [noisy, relevant],
            limit=2,
        )

        assert results[0].entity.name == "house search with Rachel"

    async def test_local_lexical_reranker_uses_matched_expansion_terms(self) -> None:
        """Expansion hits should be reranked against the expansion that found them."""
        reranker = LexicalReranker()
        distractor = SearchResult(
            entity=GraphEntity(
                name="generic art inspiration",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"summary": "Local art inspiration and exhibitions near me."},
            ),
            score=1.6,
            source="keyword",
            matched_query="How many different art-related events did I attend in the past month?",
        )
        target = SearchResult(
            entity=GraphEntity(
                name="art event ledger",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={
                    "summary": "Children's Museum Art Afternoon guided tour History Museum Art Gallery lecture street art"
                },
            ),
            score=1.1,
            source="keyword",
            matched_query=(
                "art exhibition gallery museum festival studio attended event events past month "
                "Children's Museum History Museum Art Gallery Art Afternoon guided tour lecture street art"
            ),
        )

        results = await reranker.rerank(
            "How many different art-related events did I attend in the past month?",
            [distractor, target],
            limit=2,
        )

        assert results[0].entity.name == "art event ledger"

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

    @pytest.mark.asyncio
    async def test_lexical_reranker_tokenizes_each_entity_once_with_expansions(self) -> None:
        """Local lexical reranking should reuse entity tokens across matched-query scoring."""
        result = SearchResult(
            entity=GraphEntity(
                name="bike expense memory",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={"summary": "I bought bike lights for $40."},
            ),
            score=1.0,
            source="keyword",
            matched_query="bike bicycle cycling cost",
        )
        tokenized_names: list[str] = []

        def _tokens(entity: GraphEntity) -> set[str]:
            tokenized_names.append(entity.name)
            return {"bike", "expense", "memory", "lights"}

        with patch("zaxy.query._entity_tokens", side_effect=_tokens):
            reranked = await LexicalReranker().rerank(
                "How much total money have I spent on bike expenses?",
                [result],
                limit=1,
            )

        assert len(reranked) == 1
        assert tokenized_names == ["bike expense memory"]

    async def test_rank_reuses_entity_tokens_between_mmr_and_lexical_reranker(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Graph ranking should not retokenize candidates across ranker stages."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            reranker=LexicalReranker(),
        )
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
            for index in range(12)
        ]
        tokenized_names: list[str] = []

        def _tokens(entity: GraphEntity) -> set[str]:
            tokenized_names.append(entity.name)
            return {entity.name, entity.entity_type, "shared"}

        with patch("zaxy.query._entity_tokens", side_effect=_tokens):
            ranked = await router._rank("shared memory", results, limit=5)

        assert len(ranked) == 5
        assert len(tokenized_names) == len(results)

    async def test_rank_without_reranker_mmr_ranks_only_requested_limit(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """The built-in MMR path should not rank candidates that cannot be returned."""
        router = QueryRouter(store=mock_store, default_limit=5, session_id="agent-1")
        results = [
            SearchResult(
                entity=GraphEntity(
                    name=f"memory-{index:04d}",
                    entity_type="document",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": f"memory {index} shared context"},
                ),
                score=1.0 - (index * 0.001),
                source="keyword",
            )
            for index in range(60)
        ]

        with patch("zaxy.query._entity_similarity", return_value=0.0) as similarity:
            ranked = await router._rank("shared memory", results, limit=5)

        assert len(ranked) == 5
        assert similarity.call_count <= 60

    async def test_rank_with_lexical_reranker_uses_bounded_mmr_pool(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Lexical reranking should see a bounded MMR pool, not every fused candidate."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            reranker=LexicalReranker(),
        )
        results = [
            SearchResult(
                entity=GraphEntity(
                    name=f"memory-{index:04d}",
                    entity_type="document",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": f"memory {index} shared context"},
                ),
                score=1.0 - (index * 0.001),
                source="keyword",
            )
            for index in range(60)
        ]

        with patch("zaxy.query._entity_similarity", return_value=0.0) as similarity:
            ranked = await router._rank("shared memory", results, limit=5)

        assert len(ranked) == 5
        assert similarity.call_count <= 250

    async def test_rank_tokenizes_only_bounded_mmr_pool(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Ranking should not tokenize candidates already excluded from the MMR pool."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            reranker=LexicalReranker(),
        )
        results = [
            SearchResult(
                entity=GraphEntity(
                    name=f"memory-{index:04d}",
                    entity_type="document",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"summary": f"memory {index} shared context"},
                ),
                score=1.0 - (index * 0.001),
                source="keyword",
            )
            for index in range(60)
        ]
        tokenized_names: list[str] = []

        def _tokens(entity: GraphEntity) -> set[str]:
            tokenized_names.append(entity.name)
            return {entity.name, entity.entity_type, "shared"}

        with patch("zaxy.query._entity_tokens", side_effect=_tokens):
            ranked = await router._rank("shared memory", results, limit=5)

        assert len(ranked) == 5
        assert len(tokenized_names) == 20

    def test_entity_similarity_avoids_allocating_set_operations(self) -> None:
        """MMR similarity should not allocate intersection/union sets for every pair."""

        class NoAllocSet(set[str]):
            def __and__(self, other: object) -> set[str]:
                raise AssertionError("intersection allocation should not be used")

            def __or__(self, other: object) -> set[str]:
                raise AssertionError("union allocation should not be used")

        left = GraphEntity(
            name="left",
            entity_type="document",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={},
        )
        right = GraphEntity(
            name="right",
            entity_type="document",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={},
        )
        cache = {
            ("left", "document", "2024-01-01T00:00:00Z"): NoAllocSet({"alpha", "beta", "gamma"}),
            ("right", "document", "2024-01-01T00:00:00Z"): NoAllocSet({"beta", "gamma", "delta"}),
        }

        assert _entity_similarity(left, right, cache) == 0.5

    def test_entity_similarity_avoids_generator_sum_hot_path(self, monkeypatch) -> None:
        """MMR similarity should count overlap without generator allocations."""
        monkeypatch.setattr(
            "builtins.sum",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("entity similarity should not allocate a generator for sum")
            ),
        )
        left = GraphEntity(
            name="left",
            entity_type="document",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={},
        )
        right = GraphEntity(
            name="right",
            entity_type="document",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={},
        )
        cache = {
            ("left", "document", "2024-01-01T00:00:00Z"): {"alpha", "beta", "gamma"},
            ("right", "document", "2024-01-01T00:00:00Z"): {"beta", "gamma", "delta"},
        }

        assert _entity_similarity(left, right, cache) == 0.5

    def test_query_tokenizers_use_compiled_regex_helpers(self, monkeypatch) -> None:
        """Query ranking tokenizers should not compile regex strings on every call."""

        def fail(*args, **kwargs):  # noqa: ANN001
            raise AssertionError("query tokenizers should use compiled regex helpers")

        monkeypatch.setattr("zaxy.query.re.findall", fail)
        entity = GraphEntity(
            name="Bike expense memory",
            entity_type="document",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={"summary": "I bought bike lights for $40."},
        )

        assert _tokens("Bike expenses since 2024") == {"bike", "expenses", "since", "2024"}
        assert _entity_tokens(entity) >= {"bike", "expense", "memory", "document", "lights", "40"}

    def test_expanded_queries_use_compiled_tokenizer(self, monkeypatch) -> None:
        """Query expansion should reuse the compiled token regex on the retrieval hot path."""

        def fail(*args, **kwargs):  # noqa: ANN001
            raise AssertionError("query expansion should use compiled regex helpers")

        monkeypatch.setattr("zaxy.query.re.findall", fail)

        expanded = _expanded_queries("auth decision")

        assert expanded[0] == "auth decision"
        assert any("authentication" in query for query in expanded[1:])
        assert any("rationale" in query for query in expanded[1:])

    def test_structured_preference_candidates_use_compiled_user_id_regex(self, monkeypatch) -> None:
        """Preference expansion should avoid recompiling user-id regexes on query routing paths."""

        def fail(*args, **kwargs):  # noqa: ANN001
            raise AssertionError("preference expansion should use a compiled user-id regex")

        monkeypatch.setattr("zaxy.query.re.findall", fail)

        assert _structured_preference_candidates("What is user-0003's timezone preference?") == [
            "user-0003:timezone"
        ]

    def test_exact_candidates_use_compiled_regexes(self, monkeypatch) -> None:
        """Exact candidate extraction should reuse compiled regexes on query routing paths."""

        def fail(*args, **kwargs):  # noqa: ANN001
            raise AssertionError("exact candidate extraction should use compiled regex helpers")

        monkeypatch.setattr("zaxy.query.re.finditer", fail)

        candidates = _exact_candidates("Recall Goal 0003 and user-0003:timezone for graph-goal-0003")

        assert "Goal 0003" in candidates
        assert "user-0003:timezone" in candidates
        assert "graph-goal-0003" in candidates

    def test_durable_identifier_detection_uses_loop_hot_path(self, monkeypatch) -> None:
        """Identifier classification should avoid nested generator helpers."""
        monkeypatch.setattr(
            "builtins.any",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("identifier classification should not allocate any() generators")
            ),
        )

        assert _looks_like_durable_identifier("graph-goal-0003") is True
        assert _looks_like_durable_identifier("art-related") is False

    def test_entity_identifier_matching_uses_loop_hot_path(self, monkeypatch) -> None:
        """Identifier matching should avoid generator allocation in per-result paths."""
        monkeypatch.setattr(
            "builtins.any",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("identifier matching should not allocate any() generators")
            ),
        )
        entity = GraphEntity(
            name="graph-goal-0003",
            entity_type="goal",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={"source_path": "tasks/graph-goal-0003.md"},
        )

        assert _entity_matches_identifier(entity, ("graph-goal-0003",)) is True
        assert _entity_matches_identifier(entity, ("graph-goal-9999",)) is False

    def test_identifier_score_boost_uses_loop_hot_path(self, monkeypatch) -> None:
        """Identifier boosting should avoid generator allocation in fused ranking."""
        monkeypatch.setattr(
            "builtins.any",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("identifier boosting should not allocate any() generators")
            ),
        )
        entity = GraphEntity(
            name="graph-task-0003",
            entity_type="task",
            valid_from="2024-01-01T00:00:00Z",
            valid_to=None,
            properties={"summary": "Task graph-task-0003 completion evidence."},
        )

        assert _identifier_boosted_score(0.7, entity, ("graph-task-0003",)) == 1.35
        assert _identifier_boosted_score(0.7, entity, ("graph-task-9999",)) == 0.7

    def test_identifier_suppression_uses_single_loop_hot_path(self, monkeypatch) -> None:
        """Identifier suppression should avoid generator allocation over fused results."""
        monkeypatch.setattr(
            "builtins.any",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("identifier suppression should not allocate any() generators")
            ),
        )
        target = SearchResult(
            entity=GraphEntity(
                name="graph-goal-0003",
                entity_type="goal",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={},
            ),
            score=1.0,
            source="keyword",
        )
        distractor = SearchResult(
            entity=GraphEntity(
                name="nearby context",
                entity_type="document",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={},
            ),
            score=0.9,
            source="keyword",
        )
        traversal = SearchResult(
            entity=GraphEntity(
                name="neighbor",
                entity_type="task",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={},
            ),
            score=0.8,
            source="traversal",
        )

        assert _suppress_identifier_fuzzy_distractors(
            [target, distractor, traversal],
            ("graph-goal-0003",),
        ) == [target, traversal]

    def test_inferred_edge_trust_metadata_uses_loop_hot_path(self, monkeypatch) -> None:
        """Traversal trust scoring should avoid generator sums on ranked paths."""
        monkeypatch.setattr(
            "builtins.sum",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("inferred-edge trust metadata should not allocate generator sums")
            ),
        )

        metadata = _inferred_edge_trust_metadata(
            {
                "_path_inferred_edge_count": 2,
                "_path_inferred_confidences": [0.8, 0.6, 0.2],
                "_path_inference_methods": ["cited_decision", "unknown"],
                "_path_inferred_source_event_count": 1,
                "_path_inferred_evidenced_edge_count": 1,
            }
        )

        assert metadata["confidence"] == pytest.approx(0.7)
        assert metadata["method_coverage"] == 0.5
        assert metadata["source_coverage"] == 0.5
        assert metadata["evidence_coverage"] == 0.5

    def test_mmr_ranking_updates_similarity_penalties_incrementally(self) -> None:
        """MMR should not recompute old selected-candidate similarities every round."""
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

        with patch("zaxy.query._entity_similarity", return_value=0.0) as similarity:
            ranked = _mmr_rank(results, limit=5)

        assert len(ranked) == 5
        assert similarity.call_count <= 75

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

    async def test_query_scoring_profile_override_is_per_call(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Purpose checkout should be able to change ranking policy without mutating the router."""
        router = QueryRouter(store=mock_store, default_limit=5, session_id="agent-1", scoring_profile="balanced")
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Release gate",
                    entity_type="decision",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=1.0,
                source="keyword",
            )
        ]

        precision_results = await router.query("release gate", scoring_profile="precision")
        balanced_results = await router.query("release gate")

        assert precision_results[0].score == pytest.approx(0.65)
        assert precision_results[0].score_explanation is not None
        assert precision_results[0].score_explanation["scoring_profile"] == "precision"
        assert precision_results[0].score_explanation["source_weight"] == 0.65
        assert balanced_results[0].score == pytest.approx(0.8)
        assert balanced_results[0].score_explanation is not None
        assert balanced_results[0].score_explanation["scoring_profile"] == "balanced"
        assert router.scoring_profile.name == "balanced"

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

    async def test_decay_retention_policy_uses_purpose_specific_half_life(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Purpose metadata should let Coordinate authority resist generic decay."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            retention_policy="decay",
            retention_decay_half_life_days=30,
        )
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Coordinate accepted state",
                    entity_type="memory",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={"purpose_profile": "coordinate"},
                ),
                score=1.0,
                source="keyword",
            ),
            SearchResult(
                entity=GraphEntity(
                    name="Generic old note",
                    entity_type="memory",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={},
                ),
                score=0.9,
                source="keyword",
            ),
        ]

        results = await router.query("state note", temporal_point="2024-03-01T00:00:00Z", limit=2)

        assert results[0].content.startswith("Coordinate accepted state")
        assert results[0].score_explanation is not None
        assert results[0].score_explanation["retention_policy"] == "decay"
        assert results[0].score_explanation["retention_purpose_profile"] == "coordinate"
        assert results[0].score_explanation["retention_half_life_days"] == 180
        assert results[0].score_explanation["retention_decay_multiplier"] > results[1].score_explanation["retention_decay_multiplier"]

    async def test_decay_retention_policy_uses_purpose_specific_expired_weight(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Expired purpose-scoped memory should remain downweighted, not rewritten or deleted."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            retention_policy="decay",
            retention_expired_weight=0.0,
        )
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Expired coordinate handoff",
                    entity_type="memory",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={
                        "expires_at": "2024-02-01T00:00:00Z",
                        "purpose_profile": "coordinate",
                    },
                ),
                score=1.0,
                source="keyword",
            )
        ]

        results = await router.query("coordinate handoff", temporal_point="2024-03-01T00:00:00Z", limit=1)

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.1252)
        assert results[0].score_explanation is not None
        assert results[0].score_explanation["retention_expired"] is True
        assert results[0].score_explanation["retention_purpose_profile"] == "coordinate"
        assert results[0].score_explanation["retention_decay_multiplier"] == 0.15

    async def test_filter_expired_retention_policy_ignores_purpose_expired_weight(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Purpose overrides should not bypass explicit expired filtering."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            retention_policy="filter_expired",
        )
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Expired coordinate handoff",
                    entity_type="memory",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties={
                        "expires_at": "2024-02-01T00:00:00Z",
                        "purpose_profile": "coordinate",
                    },
                ),
                score=1.0,
                source="keyword",
            )
        ]

        assert await router.query("coordinate handoff", temporal_point="2024-03-01T00:00:00Z") == []

    async def test_retention_policy_diagnostics_do_not_mutate_source_entity(
        self,
        mock_store: AsyncMock,
    ) -> None:
        """Retention diagnostics should be copied onto result metadata only."""
        router = QueryRouter(
            store=mock_store,
            default_limit=5,
            session_id="agent-1",
            retention_policy="decay",
        )
        properties = {"purpose_profile": "Security Review"}
        mock_store.search_keyword.return_value = [
            SearchResult(
                entity=GraphEntity(
                    name="Security review note",
                    entity_type="memory",
                    valid_from="2024-01-01T00:00:00Z",
                    valid_to=None,
                    properties=properties,
                ),
                score=1.0,
                source="keyword",
            )
        ]

        results = await router.query("security review", temporal_point="2024-03-01T00:00:00Z")

        assert results[0].score_explanation is not None
        assert results[0].score_explanation["retention_purpose_profile"] == "security-review"
        assert results[0].score_explanation["retention_half_life_days"] == 30
        assert "_retention_policy" not in properties

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
        assert policy.purpose_decay_half_life_days["coordinate"] == 180
        assert policy.purpose_decay_half_life_days["security"] == 180
        assert policy.purpose_decay_half_life_days["review"] == 90
        assert policy.purpose_decay_half_life_days["support"] == 90
        assert policy.purpose_decay_half_life_days["product"] == 120
        assert policy.purpose_decay_half_life_days["sales"] == 120
        assert policy.purpose_decay_half_life_days["legal"] == 365
        assert policy.purpose_decay_half_life_days["executive"] == 180
        assert policy.purpose_expired_weights["coordinate"] == 0.25
        assert policy.purpose_expired_weights["legal"] == 0.25

        long_policy = build_retention_policy(
            Settings(
                _env_file=None,
                retention_policy="decay",
                retention_decay_half_life_days=365,
                retention_expired_weight=0.2,
            )
        )
        assert long_policy.purpose_decay_half_life_days["coordinate"] == 365
        assert long_policy.purpose_decay_half_life_days["security"] == 365
        assert long_policy.purpose_decay_half_life_days["review"] == 365
        assert long_policy.purpose_decay_half_life_days["legal"] == 365
        assert long_policy.purpose_decay_half_life_days["executive"] == 365

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
        assert results[0].score_explanation["rerank_strategy"] == "lexical_overlap"
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

    async def test_openai_compatible_reranker_awaits_async_client(self) -> None:
        """Hosted reranking should not require sync HTTP calls inside async paths."""
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"choices": [{"message": {"content": '[{"index": 0, "score": 0.9}]'}}]}

        class FakeAsyncClient:
            async def post(
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

        result = SearchResult(
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
        reranker = OpenAICompatibleReranker(api_key="test-key", client=FakeAsyncClient())

        reranked = await reranker.rerank("auth decision", [result], limit=1)

        assert reranked[0].entity.name == "Auth decision"
        assert captured["url"] == "https://api.openai.com/v1/chat/completions"

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
        assert reranked[0].rerank_strategy == "cross_encoder"
        assert captured["url"] == "http://localhost:11434/rerank"
        assert captured["headers"] == {"Authorization": "Bearer local-key"}

    async def test_http_reranker_awaits_async_client(self) -> None:
        """Local HTTP reranking should support non-blocking async clients."""
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"scores": [0.8]}

        class FakeAsyncClient:
            async def post(
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

        result = SearchResult(
            entity=GraphEntity(
                name="Auth decision",
                entity_type="decision",
                valid_from="2024-01-01T00:00:00Z",
                valid_to=None,
                properties={},
            ),
            score=0.7,
            source="keyword",
        )
        reranker = HTTPReranker(endpoint="http://localhost:11434/rerank", client=FakeAsyncClient())

        reranked = await reranker.rerank("auth decision", [result], limit=1)

        assert reranked[0].reranker == "http"
        assert captured["url"] == "http://localhost:11434/rerank"

    async def test_late_interaction_http_reranker_sends_tokenized_candidates(self) -> None:
        """Late-interaction rerankers should expose token-alignment inputs and metadata."""
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {"results": [{"index": 1, "score": 0.97}, {"index": 0, "score": 0.1}]}

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
                    properties={"summary": "Background"},
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
                    properties={"summary": "Auth decision rationale"},
                ),
                score=0.7,
                source="keyword",
            ),
        ]
        reranker = LateInteractionHTTPReranker(
            endpoint="http://localhost:8080/late-rerank",
            api_key="local-key",
            client=FakeClient(),
        )

        reranked = await reranker.rerank("auth decision rationale", results, limit=1)

        assert reranked[0].entity.name == "Auth decision"
        assert reranked[0].reranker == "late-interaction-http"
        assert reranked[0].rerank_strategy == "late_interaction"
        assert captured["url"] == "http://localhost:8080/late-rerank"
        assert captured["headers"] == {"Authorization": "Bearer local-key"}
        body = captured["json"]
        assert isinstance(body, dict)
        assert body["rerank_strategy"] == "late_interaction"
        assert body["query_tokens"] == ["auth", "decision", "rationale"]
        candidates = body["candidates"]
        assert isinstance(candidates, list)
        assert candidates[1]["tokens"] == ["auth", "decision", "decision", "auth", "decision", "rationale"]

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

        late_interaction = build_reranker(
            Settings(
                _env_file=None,
                reranker_provider="late-interaction-http",
                reranker_url="http://localhost:8080/late-rerank",
            )
        )
        assert isinstance(late_interaction, LateInteractionHTTPReranker)

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
