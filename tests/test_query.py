"""Tests for zaxy.query — hybrid query router."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from zaxy.config import Settings
from zaxy.graph import GraphEntity, GraphStore, SearchResult
from zaxy.query import (
    HTTPReranker,
    LexicalReranker,
    OpenAICompatibleReranker,
    QueryRouter,
    build_reranker,
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

        results = await router.query("auth decision", embedding=[0.1, 0.2])

        assert results[0].content.startswith("Auth decision")
        assert results[0].score_explanation is not None
        assert "vector search unavailable" in results[0].score_explanation["warnings"]

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

        assert results[0].content.startswith("Auth decision")
        assert results[0].score_explanation is not None
        assert "reranker unavailable" in results[0].score_explanation["warnings"]


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
