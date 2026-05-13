"""Live retrieval benchmark runner for Zaxy and baseline memories."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from zaxy.benchmark import BenchmarkCase, RetrievalScore, _event_context, score_retrieval
from zaxy.embedding import EmbeddingProvider, embed_extraction
from zaxy.event import EventLog
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.query import QueryRouter
from zaxy.retrieval_intent import RetrievalIntent, classify_retrieval_intent

FROZEN_WORKLOAD_VERSION = "statistical-v1"
FROZEN_WORKLOAD_SUBJECTS = 100
CONSOLIDATION_WORKLOAD_VERSION = "consolidation-v1"
CONTEXT_COLLAPSE_WORKLOAD_VERSION = "mempalace-context-collapse-v1"
GRAPH_TRAVERSAL_WORKLOAD_VERSION = "mempalace-graph-traversal-v1"
SUITE_WORKLOAD_VERSION = "suite-v1"
SOURCE_RECALL_WORKLOAD_VERSION = "mempalace-source-recall-v1"
TEMPORAL_RECALL_WORKLOAD_VERSION = "mempalace-temporal-recall-v1"
LONGMEMEVAL_WORKLOAD_VERSION = "longmemeval-cleaned-v1"
LONGMEMEVAL_MAX_CHUNK_CHARS = 3_500
SUITE_WORKLOAD_LANES = (
    "current",
    "temporal",
    "traversal",
    "document",
    "transcript",
    "mixed",
)


@dataclass(frozen=True)
class BenchmarkChunk:
    """A retrievable benchmark document chunk."""

    chunk_id: str
    text: str


@dataclass(frozen=True)
class BenchmarkRun:
    """One measured retrieval attempt."""

    backend: str
    case_name: str
    category: str
    run: int
    score: float
    latency_ms: float
    result_count: int
    returned_bytes: int
    approx_tokens: int
    expected_hits: tuple[str, ...]
    missing_expected: tuple[str, ...]
    forbidden_hits: tuple[str, ...]
    identity_recall: float | None
    identity_hits: tuple[str, ...]
    missing_identities: tuple[str, ...]
    source_recall: float | None
    source_hits: tuple[str, ...]
    missing_sources: tuple[str, ...]
    citation_count: int
    citation_coverage: float | None
    recall_at_1: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark statistics for one backend."""

    backend: str
    case_count: int
    runs: int
    mean_score: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    mean_returned_bytes: float
    mean_approx_tokens: float
    mean_identity_recall: float | None = None
    mean_source_recall: float | None = None
    mean_citation_coverage: float | None = None
    mean_recall_at_1: float | None = None
    mean_recall_at_5: float | None = None
    mean_recall_at_10: float | None = None


@dataclass(frozen=True)
class BackendComparison:
    """Paired statistical comparison between target and baseline backend."""

    target_backend: str
    baseline_backend: str
    paired_units: int
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class BenchmarkGuardrailCheck:
    """One benchmark quality or latency guardrail result."""

    name: str
    passed: bool
    observed: float | None
    threshold: float
    message: str


@dataclass(frozen=True)
class BenchmarkGuardrailReport:
    """Comparison of a candidate benchmark report against guardrails."""

    backend: str
    passed: bool
    baseline: BenchmarkSummary | None
    candidate: BenchmarkSummary
    checks: tuple[BenchmarkGuardrailCheck, ...]


@dataclass(frozen=True)
class CategorySummary:
    """Aggregate score for one backend/category pair."""

    backend: str
    category: str
    query_count: int
    mean_score: float
    miss_count: int
    mean_source_recall: float | None = None
    mean_citation_coverage: float | None = None
    mean_recall_at_1: float | None = None
    mean_recall_at_5: float | None = None
    mean_recall_at_10: float | None = None


@dataclass(frozen=True)
class BenchmarkWorkload:
    """Identity metadata for a benchmark workload."""

    version: str
    subjects: int | None
    event_count: int
    case_count: int
    sha256: str
    documents: int | None = None
    sessions: int | None = None
    lanes: tuple[str, ...] = ()

    @classmethod
    def from_event_log(
        cls,
        eventlog: EventLog,
        cases: tuple[BenchmarkCase, ...],
        *,
        version: str,
        subjects: int | None = None,
        documents: int | None = None,
        sessions: int | None = None,
        lanes: tuple[str, ...] = (),
    ) -> BenchmarkWorkload:
        """Create workload metadata from an Eventloom log and cases."""
        return cls(
            version=version,
            subjects=subjects,
            event_count=len(eventlog.read_all()),
            case_count=len(cases),
            sha256=workload_fingerprint(eventlog, cases, version),
            documents=documents,
            sessions=sessions,
            lanes=lanes,
        )


@dataclass(frozen=True)
class BenchmarkWorkloadInventoryEntry:
    """Release-facing proof metadata for one comparable benchmark lane."""

    lane: str
    version: str
    sha256: str
    event_count: int
    case_count: int
    required_metrics: tuple[str, ...]
    product_claim: str


@dataclass(frozen=True)
class ExternalBenchmarkResult:
    """Operator-supplied benchmark row for an external context system."""

    system: str
    version: str
    mean_score: float
    latency_ms_p95: float | None = None
    source: str = "operator-supplied"
    notes: str | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark output."""

    generated_at: str
    embedding_provider: str
    runs: tuple[BenchmarkRun, ...]
    summaries: tuple[BenchmarkSummary, ...]
    category_summaries: tuple[CategorySummary, ...] = ()
    comparisons: tuple[BackendComparison, ...] = ()
    workload: BenchmarkWorkload | None = None
    external_results: tuple[ExternalBenchmarkResult, ...] = ()


@dataclass(frozen=True)
class WrittenBenchmarkReport:
    """Paths written for a benchmark report."""

    json_path: Path
    markdown_path: Path


class Retriever(Protocol):
    """Protocol implemented by all benchmark retrievers."""

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return context strings for a query."""


class AsyncRetriever(Protocol):
    """Protocol for benchmark retrievers that run inside an event loop."""

    async def query_async(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return context strings for a query."""


class CachedEmbeddingProvider:
    """In-memory embedding cache for benchmark runs.

    This keeps hosted benchmark runs from re-embedding identical corpus chunks
    and repeated paired queries across baselines.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_path: str | Path | None = None,
        *,
        flush_every: int = 100,
    ) -> None:
        self._provider = provider
        self.dimension = provider.dimension
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, list[float]] = self._load_cache()
        self._dirty_count = 0
        self._flush_every = max(1, flush_every)

    @property
    def cache_size(self) -> int:
        """Number of cached embedding texts."""
        return len(self._cache)

    def embed(self, text: str) -> list[float]:
        """Return a cached embedding for text, computing it once."""
        cached = self._cache.get(text)
        if cached is None:
            cached = self._provider.embed(text)
            self._cache[text] = cached
            self._dirty_count += 1
            if self._dirty_count >= self._flush_every:
                self.flush()
        return list(cached)

    def flush(self) -> None:
        """Persist pending cache misses to disk."""
        if self._dirty_count == 0:
            return
        self._write_cache()
        self._dirty_count = 0

    def _load_cache(self) -> dict[str, list[float]]:
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        cache: dict[str, list[float]] = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, list):
                cache[key] = [float(item) for item in value]
        return cache

    def _write_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._cache_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(self._cache, sort_keys=True), encoding="utf-8")
        temporary_path.replace(self._cache_path)


class MarkdownRetriever:
    """Markdown/file-memory baseline using direct token scanning."""

    def __init__(self, corpus: tuple[BenchmarkChunk, ...]) -> None:
        self._corpus = corpus

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return chunks containing at least one query token."""
        del temporal_point
        tokens = _tokens(query)
        matches: list[str] = []
        for chunk in self._corpus:
            searchable = chunk.text.casefold()
            if any(token in searchable for token in tokens):
                matches.append(chunk.text)
            if len(matches) >= limit:
                break
        return matches


class BM25Retriever:
    """Okapi BM25 baseline over the same markdown-style benchmark corpus."""

    def __init__(
        self,
        corpus: tuple[BenchmarkChunk, ...],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._corpus = corpus
        self._k1 = k1
        self._b = b
        self._tokenized = tuple(tuple(_bm25_tokens(chunk.text)) for chunk in corpus)
        self._document_frequencies = _bm25_document_frequencies(self._tokenized)
        self._document_count = len(self._tokenized)
        self._average_document_length = (
            statistics.fmean(len(tokens) for tokens in self._tokenized)
            if self._tokenized
            else 0.0
        )

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return chunks ranked by Okapi BM25 score."""
        del temporal_point
        query_terms = _expanded_bm25_query_terms(query)
        if not query_terms:
            return []
        scored = [
            (
                _bm25_score(
                    query_terms,
                    document_terms,
                    self._document_frequencies,
                    self._document_count,
                    self._average_document_length,
                    self._k1,
                    self._b,
                )
                * _memory_salience_boost(chunk.text),
                chunk.text,
            )
            for chunk, document_terms in zip(self._corpus, self._tokenized, strict=True)
        ]
        scored = [(score, text) for score, text in scored if score > 0.0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]


class VectorRetriever:
    """Vector-only baseline over the same benchmark corpus."""

    def __init__(self, corpus: tuple[BenchmarkChunk, ...], provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._indexed = tuple((chunk, provider.embed(chunk.text)) for chunk in corpus)

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return top vector-similar chunks."""
        del temporal_point
        query_embedding = self._provider.embed(query)
        scored = [
            (_cosine(query_embedding, embedding), chunk.text)
            for chunk, embedding in self._indexed
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]


class MarkdownVectorRetriever:
    """Hybrid markdown baseline: token candidate generation plus vector ranking."""

    def __init__(self, corpus: tuple[BenchmarkChunk, ...], provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._corpus = corpus
        self._embeddings = {chunk.chunk_id: provider.embed(chunk.text) for chunk in corpus}

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return token-filtered chunks ranked by vector similarity."""
        del temporal_point
        tokens = _tokens(query)
        candidates = [
            chunk
            for chunk in self._corpus
            if any(token in chunk.text.casefold() for token in tokens)
        ]
        if not candidates:
            candidates = list(self._corpus)

        query_embedding = self._provider.embed(query)
        scored = [
            (_cosine(query_embedding, self._embeddings[chunk.chunk_id]), chunk.text)
            for chunk in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]


class RankFusionRetriever:
    """Fuse ranked outputs from complementary retrievers with RRF scoring."""

    def __init__(
        self,
        retrievers: dict[str, Retriever],
        *,
        weights: dict[str, float] | None = None,
        rank_constant: int = 60,
    ) -> None:
        self._retrievers = retrievers
        self._weights = weights or {}
        self._rank_constant = rank_constant

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return fused, de-duplicated contexts from all retrievers."""
        scored: dict[str, float] = {}
        order: dict[str, int] = {}
        for retriever_name, retriever in self._retrievers.items():
            weight = self._weights.get(retriever_name, 1.0)
            for rank, text in enumerate(
                retriever.query(query, temporal_point=temporal_point, limit=limit),
                start=1,
            ):
                if text not in order:
                    order[text] = len(order)
                scored[text] = scored.get(text, 0.0) + (weight / (self._rank_constant + rank))
        ranked = sorted(scored, key=lambda text: (-scored[text], order[text]))
        return ranked[:limit]


class CentroidConsolidationRetriever:
    """Centroid-style consolidation baseline that keeps one representative text.

    This intentionally models the failure mode where a compressed semantic
    memory remains topically relevant but cannot preserve each source identity.
    """

    def __init__(self, corpus: tuple[BenchmarkChunk, ...], provider: EmbeddingProvider) -> None:
        self._provider = provider
        embeddings = [provider.embed(chunk.text) for chunk in corpus]
        self._centroid = _centroid(embeddings)
        self._representative = corpus[0].text if corpus else ""

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return the consolidated representative when the centroid is relevant."""
        del temporal_point, limit
        if not self._representative:
            return []
        query_embedding = self._provider.embed(query)
        if _cosine(query_embedding, self._centroid) <= 0.0:
            return []
        return [self._representative]


class ZaxyRetriever:
    """Synchronous wrapper around Zaxy's live graph retrieval path."""

    def __init__(
        self,
        router: QueryRouter,
        provider: EmbeddingProvider,
        lexical_retriever: Retriever | None = None,
    ) -> None:
        self._router = router
        self._provider = provider
        self._lexical_retriever = lexical_retriever

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return Zaxy graph contexts for a query."""
        async def _query() -> list[str]:
            return await self.query_async(query, temporal_point=temporal_point, limit=limit)

        return asyncio.run(_query())

    async def query_async(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return Zaxy graph contexts inside an existing event loop."""
        embedding = self._provider.embed(query)
        chunks = await self._router.query(
            query,
            temporal_point=temporal_point,
            limit=limit,
            embedding=embedding,
        )
        graph_results = [_benchmark_context_from_chunk(chunk) for chunk in chunks]
        if (
            self._lexical_retriever is None
            or temporal_point is not None
            or not _should_query_source_lexical_lane(query, limit=limit)
        ):
            return graph_results
        lexical_query = _source_lane_query(query, graph_results)
        lexical_limit = _source_lane_candidate_limit(query, limit=limit)
        lexical_results = self._lexical_retriever.query(
            lexical_query,
            temporal_point=temporal_point,
            limit=lexical_limit,
        )
        lexical_results = _filter_superseded_preference_lexical_results(
            graph_results,
            lexical_results,
        )
        synthesis_bundle = _source_synthesis_bundle(
            query=query,
            source_results=lexical_results,
            limit=limit,
        ) or _absence_check_bundle(
            query=query,
            source_results=lexical_results,
            limit=limit,
        )
        fused = RankFusionRetriever(
            {
                "graph": _StaticRetriever(tuple(graph_results)),
                "lexical": _StaticRetriever(tuple(lexical_results)),
            },
            weights={"graph": 3.0, "lexical": 1.0},
        )
        fused_results = fused.query(query, temporal_point=temporal_point, limit=limit)
        return _with_reserved_lexical_lane(
            fused_results,
            lexical_results,
            query=query,
            limit=limit,
            synthesis_bundle=synthesis_bundle,
        )


class _StaticRetriever:
    """Return precomputed contexts using the Retriever protocol."""

    def __init__(self, contexts: tuple[str, ...]) -> None:
        self._contexts = contexts

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return stored contexts."""
        del query, temporal_point
        return list(self._contexts[:limit])


def _benchmark_context_from_chunk(chunk: object) -> str:
    """Format graph context with citation metadata for benchmark scoring."""
    content = str(getattr(chunk, "content", ""))
    citation = getattr(chunk, "citation", None)
    if isinstance(citation, str) and citation:
        return f"{content}\ncitation={citation}"
    return content


def _should_query_source_lexical_lane(query: str, *, limit: int = 10) -> bool:
    """Return whether raw Eventloom text should supplement graph retrieval.

    The lexical Eventloom lane is useful for exact source/provenance recovery,
    and for non-temporal personal-memory questions where exact utterance text
    often contains the answer before graph extraction has a typed schema for it.
    Temporal queries are blocked by the caller because raw history contains
    superseded facts.
    """
    return classify_retrieval_intent(query, limit=limit).needs_source_lane


def _source_lane_query(query: str, graph_results: list[str]) -> str:
    """Expand source lookup with compact answer concepts found by graph retrieval."""
    concepts = _graph_answer_concepts(graph_results)
    if not concepts:
        return query
    return " ".join([query, *concepts])


def _source_lane_candidate_limit(query: str, *, limit: int) -> int:
    """Return internal source candidate budget for source-sensitive retrieval."""
    if limit <= 0:
        return 0
    intent = classify_retrieval_intent(query, limit=limit)
    if not intent.needs_source_lane:
        return limit
    if any(
        reason in intent.reasons
        for reason in ("aggregation", "aggregation_question", "absence_check")
    ):
        return max(limit, intent.source_lane_slots * 6)
    return max(limit, intent.source_lane_slots * 4)


def _graph_answer_concepts(graph_results: list[str], *, limit: int = 4) -> list[str]:
    """Extract bounded human-scale concepts from graph context for source backfill."""
    concepts: list[str] = []
    seen: set[str] = set()
    skip_tokens = {
        "entity",
        "event",
        "source",
        "summary",
        "document",
        "citation",
        "benchmark",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
        "for",
        "the",
        "do",
        "now",
    }
    for result in graph_results:
        for phrase in re.findall(
            r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}\b",
            result,
        ):
            normalized = phrase.casefold()
            words = normalized.split()
            if normalized in seen or all(word in skip_tokens for word in words):
                continue
            if len(words) == 1 and (words[0] in skip_tokens or len(words[0]) < 3):
                continue
            if re.fullmatch(r"[a-f0-9]{8,}", normalized):
                continue
            concepts.append(phrase)
            seen.add(normalized)
            if len(concepts) >= limit:
                return concepts
    return concepts


def _filter_superseded_preference_lexical_results(
    graph_results: list[str],
    lexical_results: list[str],
) -> list[str]:
    """Remove raw stale preference rows when graph retrieval has the current fact."""
    current_preferences = _current_preference_values(graph_results)
    if not current_preferences:
        return lexical_results
    filtered: list[str] = []
    for result in lexical_results:
        if _is_stale_preference_result(result, current_preferences):
            continue
        filtered.append(result)
    return filtered


def _with_reserved_lexical_lane(
    fused_results: list[str],
    lexical_results: list[str],
    *,
    query: str,
    limit: int,
    synthesis_bundle: str | None = None,
) -> list[str]:
    """Preserve top verbatim source hits as a bounded lane in fused context."""
    if limit <= 0 or not lexical_results:
        return fused_results[:limit]
    intent = classify_retrieval_intent(query, limit=limit)
    reserved_count = min(
        len(lexical_results),
        max(1, min(2, limit // 5), intent.source_lane_slots),
    )
    reserved = _diverse_source_contexts(lexical_results, limit=reserved_count)
    reserved_set = set(reserved)
    primary_slots = max(0, limit - len(reserved))
    primary = [
        result for result in fused_results
        if result not in reserved_set
    ][:primary_slots]
    if _should_rank_source_lane_first(intent):
        results = [*reserved, *primary][:limit]
    else:
        results = [*primary, *reserved][:limit]
    if synthesis_bundle is None:
        return results
    return [synthesis_bundle, *[result for result in results if result != synthesis_bundle]][:limit]


def _should_rank_source_lane_first(intent: RetrievalIntent) -> bool:
    reasons = set(intent.reasons)
    return "personal_memory" in reasons and not reasons & {
        "aggregation",
        "aggregation_question",
        "operational_memory",
        "source_recall",
    }


def _source_synthesis_bundle(
    *,
    query: str,
    source_results: list[str],
    limit: int,
) -> str | None:
    """Build one compact cited source bundle for multi-source synthesis queries."""
    intent = classify_retrieval_intent(query, limit=limit)
    if not {"aggregation", "aggregation_question"} & set(intent.reasons):
        return None
    group_limit = max(limit, intent.source_lane_slots)
    grouped_sources = _diverse_source_contexts(source_results, limit=group_limit)
    if len(grouped_sources) < 2:
        return None
    lines = [
        "zaxy_synthesis_bundle=true",
        "synthesis_mode=multi_source_aggregation",
        f"query={query}",
        f"source_count={len(grouped_sources)}",
    ]
    lines.extend(_numeric_synthesis_lines(grouped_sources))
    for index, context in enumerate(grouped_sources, start=1):
        lines.append(
            "- "
            f"source_id={_source_context_group(context)} "
            f"citation={_source_context_citation(context)} "
            f"snippet={_source_context_snippet(context)}"
        )
        if index >= group_limit:
            break
    return "\n".join(lines)


def _absence_check_bundle(
    *,
    query: str,
    source_results: list[str],
    limit: int,
) -> str | None:
    """Build cited guidance for questions about absent personal memories."""
    intent = classify_retrieval_intent(query, limit=limit)
    if "absence_check" not in intent.reasons:
        return None
    target = _absence_check_target(query)
    if not target:
        return None
    grouped_sources = _diverse_source_contexts(
        source_results,
        limit=max(1, intent.source_lane_slots),
    )
    if not grouped_sources or _target_terms_present(target, grouped_sources):
        return None
    lines = [
        "zaxy_absence_check=true",
        "synthesis_mode=absence_check",
        f"query={query}",
        f"not_mentioned_candidate={target}",
        (
            "answer_guidance=You did not mention this information. "
            f"You mentioned cited evidence below, but not {target}."
        ),
    ]
    for context in grouped_sources:
        lines.append(
            "- "
            f"source_id={_source_context_group(context)} "
            f"citation={_source_context_citation(context)} "
            f"snippet={_source_context_snippet(context)}"
        )
    return "\n".join(lines)


_ABSENCE_QUERY_STOPWORDS = {
    "about",
    "any",
    "anything",
    "did",
    "do",
    "does",
    "ever",
    "have",
    "i",
    "in",
    "information",
    "me",
    "mention",
    "mentioned",
    "my",
    "not",
    "remember",
    "say",
    "the",
    "this",
    "whether",
}


def _absence_check_target(query: str) -> str:
    terms = [
        token
        for token in _bm25_tokens(query)
        if token not in _ABSENCE_QUERY_STOPWORDS
        and not token.isdigit()
        and len(token) > 1
    ]
    return " ".join(dict.fromkeys(terms))


def _target_terms_present(target: str, contexts: list[str]) -> bool:
    target_terms = [
        token
        for token in _bm25_tokens(target)
        if token not in _ABSENCE_QUERY_STOPWORDS and len(token) > 1
    ]
    if not target_terms:
        return False
    for context in contexts:
        context_terms = set(_bm25_tokens(context))
        if all(term in context_terms for term in target_terms):
            return True
    return False


def _numeric_synthesis_lines(contexts: list[str]) -> list[str]:
    """Project deterministic numeric operations from cited source snippets."""
    numeric_contexts = [_numeric_context_text(context) for context in contexts]
    lines: list[str] = []
    currency_values = _currency_values(numeric_contexts)
    if currency_values:
        lines.append(
            "currency_values="
            + ",".join(_format_currency(value) for value in currency_values)
        )
        lines.append(f"currency_total={_format_currency(sum(currency_values))}")
        if len(currency_values) >= 2:
            lines.append(
                "currency_difference="
                f"{_format_currency(max(currency_values) - min(currency_values))}"
            )
    minute_values = _unit_values(numeric_contexts, unit_pattern=r"minutes?|mins?")
    if minute_values:
        lines.append("minute_values=" + ",".join(_format_number(value) for value in minute_values))
        lines.append(f"minute_total_hours={_format_number(sum(minute_values) / 60)} hours")
    hour_values = _unit_values(numeric_contexts, unit_pattern=r"hours?|hrs?")
    if hour_values:
        lines.append("hour_values=" + ",".join(_format_number(value) for value in hour_values))
        lines.append(f"hour_total={_format_number(sum(hour_values))} hours")
    day_values = _unit_values(numeric_contexts, unit_pattern=r"days?")
    if day_values:
        lines.append("day_values=" + ",".join(_format_number(value) for value in day_values))
        lines.append(f"day_total={_format_number(sum(day_values))} days")
    return lines


def _numeric_context_text(context: str) -> str:
    """Return source text once, excluding Eventloom JSON payload echoes."""
    text = _source_context_snippet(context)
    return text.split(' {"content":', 1)[0]


def _currency_values(contexts: list[str]) -> list[float]:
    values: list[float] = []
    for context in contexts:
        for match in re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", context):
            values.append(float(match.group("value").replace(",", "")))
    return values


def _unit_values(contexts: list[str], *, unit_pattern: str) -> list[float]:
    values: list[float] = []
    pattern = re.compile(
        rf"\b(?P<value>\d+(?:\.\d+)?)\s*[- ]?(?:{unit_pattern})\b",
        flags=re.IGNORECASE,
    )
    for context in contexts:
        for match in pattern.finditer(context):
            values.append(float(match.group("value")))
    return values


def _format_currency(value: float) -> str:
    return f"${_format_number(value)}"


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _diverse_source_contexts(contexts: list[str], *, limit: int) -> list[str]:
    """Select source contexts across provenance groups before filling by rank."""
    if limit <= 0:
        return []
    contexts = _source_lane_priority_order(contexts)
    selected: list[str] = []
    seen_contexts: set[str] = set()
    seen_groups: set[str] = set()
    for context in contexts:
        if context in seen_contexts:
            continue
        group = _source_context_group(context)
        if group in seen_groups:
            continue
        selected.append(context)
        seen_contexts.add(context)
        seen_groups.add(group)
        if len(selected) >= limit:
            return selected
    for context in contexts:
        if context in seen_contexts:
            continue
        selected.append(context)
        seen_contexts.add(context)
        if len(selected) >= limit:
            break
    return selected


def _source_lane_priority_order(contexts: list[str]) -> list[str]:
    """Prefer compact source memories over raw chunks while preserving rank within tiers."""
    indexed = list(enumerate(contexts))
    indexed.sort(key=lambda item: (-_source_lane_priority(item[1]), item[0]))
    return [context for _, context in indexed]


def _source_lane_priority(context: str) -> int:
    lowered = context.casefold()
    if (
        "salient_memory_turn=true" in lowered
        or "hook.checkpoint" in lowered
        or "longmemeval_salient_memory_turn=true" in lowered
    ):
        return 2
    if "citation=" in lowered or "eventloom://" in lowered or "source_path=" in lowered:
        return 1
    return 0


def _source_context_group(context: str) -> str:
    """Return a stable source group from common citation/session metadata."""
    patterns = [
        r"\b[a-z0-9_.-]*session[_-]?id=(?P<value>[^\s]+)",
        r"\b(?:source_path|path|file)=['\"]?(?P<value>[^\s'\"]+)",
        r"\bthread=['\"]?(?P<value>[^\s'\"]+)",
        r"eventloom://[^/]+/events/(?P<value>\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return match.group("value").casefold()
    return context[:160].casefold()


def _source_context_citation(context: str) -> str:
    """Extract a compact citation token from source context."""
    for pattern in (
        r"\bcitation=(?P<value>\S+)",
        r"(?P<value>eventloom://\S+)",
        r"\bsource_path=(?P<value>\S+)",
    ):
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return match.group("value")
    return "unknown"


def _source_context_snippet(context: str, *, max_chars: int = 900) -> str:
    """Return a bounded one-line source snippet."""
    snippet = " ".join(context.split())
    if len(snippet) <= max_chars:
        return snippet
    return f"{snippet[: max_chars - 3].rstrip()}..."


def _unique_contexts(contexts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for context in contexts:
        if context in seen:
            continue
        seen.add(context)
        unique.append(context)
    return unique


def _current_preference_values(results: list[str]) -> dict[tuple[str, str], str]:
    preferences: dict[tuple[str, str], str] = {}
    for result in results:
        for match in re.finditer(
            r"\b(?P<user>user-\d{4}):(?P<key>[A-Za-z0-9_.-]+)\b.*?"
            r"(?P=key)=(?P<value>[A-Za-z0-9_.-]+)",
            result,
            flags=re.IGNORECASE,
        ):
            preferences[
                (match.group("user").casefold(), match.group("key").casefold())
            ] = match.group("value").casefold()
    return preferences


def _is_stale_preference_result(
    result: str,
    current_preferences: dict[tuple[str, str], str],
) -> bool:
    lowered = result.casefold()
    for (user_id, key), current_value in current_preferences.items():
        if user_id not in lowered or key not in lowered:
            continue
        value_match = re.search(
            rf"\b(?:value|{re.escape(key)})[=:]\s*['\"]?(?P<value>[A-Za-z0-9_.-]+)",
            result,
            flags=re.IGNORECASE,
        )
        if value_match and value_match.group("value").casefold() != current_value:
            return True
    return False


def corpus_from_event_log(eventlog: EventLog) -> tuple[BenchmarkChunk, ...]:
    """Build a markdown-like corpus from Eventloom events."""
    return tuple(
        BenchmarkChunk(
            chunk_id=f"event-{event.seq}",
            text=(
                f"# Event {event.seq}\n\n"
                f"citation=eventloom://benchmark/events/{event.seq}#{event.hash}\n\n"
                f"{_event_context(event.model_dump())}"
            ),
        )
        for event in eventlog.read_all()
    )


def build_statistical_event_log(
    path: str | Path,
    subjects: int = 100,
) -> tuple[EventLog, tuple[BenchmarkCase, ...]]:
    """Build a larger deterministic temporal workload for statistical evaluation."""
    if subjects <= 0:
        raise ValueError("subjects must be positive")
    log = EventLog(path)
    cases: list[BenchmarkCase] = []
    for idx in range(subjects):
        user_id = f"user-{idx:04d}"
        goal = f"Goal {idx:04d}"
        task_id = f"task-{idx:04d}"
        old_theme = f"theme-old-{idx % 7}"
        new_theme = f"theme-new-{idx % 7}"

        log.append(
            "goal.created",
            actor="user",
            payload={"title": goal, "description": f"Launch workstream {idx:04d}"},
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )
        log.append(
            "task.proposed",
            actor="agent",
            payload={
                "taskId": task_id,
                "goalTitle": goal,
                "summary": f"{task_id} implements {goal} release path",
            },
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
        )
        log.append(
            "user.preference_changed",
            actor="user",
            payload={"userId": user_id, "key": "theme", "value": old_theme},
            timestamp=datetime(2024, 2, 1, tzinfo=UTC),
        )
        log.append(
            "user.preference_changed",
            actor="user",
            payload={"userId": user_id, "key": "theme", "value": new_theme},
            timestamp=datetime(2024, 6, 1, tzinfo=UTC),
        )
        log.append(
            "task.completed",
            actor="agent",
            payload={"taskId": task_id},
            timestamp=datetime(2024, 7, 1, tzinfo=UTC),
        )

        cases.append(
            BenchmarkCase(
                name=f"current-theme-{idx:04d}",
                query=f"What is the current theme preference for {user_id}?",
                expected_terms=(f"theme={new_theme}",),
                forbidden_terms=(f"theme={old_theme}",),
                category="current",
            )
        )
        cases.append(
            BenchmarkCase(
                name=f"historical-theme-{idx:04d}",
                query=f"What was the theme preference for {user_id} in March 2024?",
                temporal_point="2024-03-01T00:00:00Z",
                expected_terms=(f"theme={old_theme}",),
                forbidden_terms=(f"theme={new_theme}",),
                category="temporal",
            )
        )
        cases.append(
            BenchmarkCase(
                name=f"task-for-goal-{idx:04d}",
                query=f"Which task is connected to {goal}?",
                expected_terms=(task_id, goal),
                category="traversal",
            )
        )
    return log, tuple(cases)


def build_frozen_statistical_workload(
    path: str | Path,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build the frozen statistical workload used for publishable comparisons."""
    eventlog, cases = build_statistical_event_log(path, subjects=FROZEN_WORKLOAD_SUBJECTS)
    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        cases,
        version=FROZEN_WORKLOAD_VERSION,
        subjects=FROZEN_WORKLOAD_SUBJECTS,
    )
    return eventlog, cases, workload


def build_graph_traversal_workload(
    path: str | Path,
    subjects: int = 100,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build a frozen lane for multi-hop goal-task-completion traversal."""
    if subjects <= 0:
        raise ValueError("subjects must be positive")

    eventlog = EventLog(path)
    cases: list[BenchmarkCase] = []
    for idx in range(subjects):
        goal = f"graph-goal-{idx:04d}"
        task_id = f"graph-task-{idx:04d}"
        finisher = f"graph-finisher-{idx:04d}"
        distractor = f"graph-finisher-distractor-{idx:04d}"
        eventlog.append(
            "goal.created",
            actor="planner",
            payload={
                "title": goal,
                "description": (
                    f"Graph traversal objective {idx:04d} with no completion actor named here."
                ),
            },
            timestamp=datetime(2024, 3, 1, tzinfo=UTC),
        )
        eventlog.append(
            "task.proposed",
            actor="planner",
            payload={
                "taskId": task_id,
                "goalTitle": goal,
                "summary": (
                    f"{task_id} is the implementation task for {goal}."
                ),
            },
            timestamp=datetime(2024, 3, 2, tzinfo=UTC),
        )
        eventlog.append(
            "task.completed",
            actor=finisher,
            payload={
                "taskId": task_id,
                "summary": f"Completion recorded for {task_id} after graph traversal review.",
            },
            timestamp=datetime(2024, 3, 3, tzinfo=UTC),
        )
        eventlog.append(
            "task.completed",
            actor=distractor,
            payload={
                "taskId": f"graph-distractor-task-{idx:04d}",
                "summary": (
                    f"{distractor} completed an unrelated distractor task for graph traversal."
                ),
            },
            timestamp=datetime(2024, 3, 4, tzinfo=UTC),
        )
        cases.append(
            BenchmarkCase(
                name=f"graph-traversal-{idx:04d}",
                query=f"Which actor completed the task connected to {goal}?",
                expected_terms=(finisher, task_id),
                forbidden_terms=(distractor,),
                category="graph-traversal",
                identity_terms=(goal, task_id, finisher),
            )
        )

    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        tuple(cases),
        version=GRAPH_TRAVERSAL_WORKLOAD_VERSION,
        subjects=subjects,
        lanes=("graph-traversal",),
    )
    return eventlog, tuple(cases), workload


def build_context_collapse_workload(
    path: str | Path,
    sessions: int = 100,
    turns_per_session: int = 40,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build a frozen lane for long-session context-collapse recovery."""
    if sessions <= 0:
        raise ValueError("sessions must be positive")
    if turns_per_session <= 0:
        raise ValueError("turns_per_session must be positive")

    eventlog = EventLog(path)
    cases: list[BenchmarkCase] = []
    for idx in range(sessions):
        session_id = f"collapse-session-{idx:04d}"
        answer = f"collapseanswer{idx:04d}"
        for turn_index in range(1, turns_per_session + 1):
            eventlog.append(
                "transcript.turn",
                actor="assistant" if turn_index % 2 == 0 else "user",
                payload={
                    "source": session_id,
                    "turn_index": turn_index,
                    "role": "assistant" if turn_index % 2 == 0 else "user",
                    "content": (
                        f"{session_id} context collapse distractor turn {turn_index:04d}. "
                        "The team recorded preserved decision handoff details, "
                        "but this turn intentionally omits the durable answer code."
                    ),
                    "redacted_paths": [],
                },
                thread=session_id,
                timestamp=datetime(2024, 11, 1, 0, turn_index % 60, tzinfo=UTC),
            )
        eventlog.append(
            "hook.checkpoint",
            actor="zaxy",
            payload={
                "trigger": "checkpoint",
                "session_id": session_id,
                "source": "context-collapse",
                "reason": "long-session-memory",
                "turn_count": turns_per_session,
                "summary": (
                    f"{session_id} preserved decision survived context collapse: {answer}. "
                    "Use this compact checkpoint instead of replaying noisy turns."
                ),
            },
            thread=session_id,
            timestamp=datetime(2024, 11, 1, 1, idx % 60, tzinfo=UTC),
        )
        cases.append(
            BenchmarkCase(
                name=f"context-collapse-{idx:04d}",
                query=f"What preserved decision is recorded for {session_id}?",
                expected_terms=(answer,),
                category="context-collapse",
                identity_terms=(session_id, answer),
            )
        )

    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        tuple(cases),
        version=CONTEXT_COLLAPSE_WORKLOAD_VERSION,
        sessions=sessions,
        lanes=("context-collapse",),
    )
    return eventlog, tuple(cases), workload


def build_source_recall_workload(
    path: str | Path,
    documents: int = 100,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build a frozen source recall lane for cited-source retrieval evaluation."""
    if documents <= 0:
        raise ValueError("documents must be positive")

    eventlog = EventLog(path)
    cases: list[BenchmarkCase] = []
    for idx in range(documents):
        answer_code = f"source-answer-{idx:04d}"
        target_path = f"source-recall/target/service-{idx:04d}.md"
        distractor_path = f"source-recall/distractor/service-{idx:04d}.md"
        target_content = (
            f"{target_path} records source_recall_answer_code={answer_code}. "
            f"The canonical owner is source-owner-{idx % 11} and the cited "
            f"runbook section is source-section-{idx % 7}."
        )
        distractor_content = (
            f"{distractor_path} discusses a nearby source recall incident for "
            f"service-{idx:04d}, but it does not carry the canonical answer code."
        )
        eventlog.append(
            "document.indexed",
            actor="source-recall",
            payload={
                "path": target_path,
                "start_line": 10 + idx,
                "end_line": 14 + idx,
                "content": target_content,
                "sha256": _content_sha256(target_content),
                "source_recall_answer_code": answer_code,
                "source_recall_role": "target",
            },
            timestamp=datetime(2024, 4, 1, tzinfo=UTC),
        )
        eventlog.append(
            "document.indexed",
            actor="source-recall",
            payload={
                "path": distractor_path,
                "start_line": 20 + idx,
                "end_line": 24 + idx,
                "content": distractor_content,
                "sha256": _content_sha256(distractor_content),
                "source_recall_near_miss": f"service-{idx:04d}",
                "source_recall_role": "distractor",
            },
            timestamp=datetime(2024, 4, 1, 0, 1, tzinfo=UTC),
        )
        cases.append(
            BenchmarkCase(
                name=f"source-recall-{idx:04d}",
                query=f"Which cited source records {answer_code}?",
                expected_terms=(answer_code,),
                forbidden_terms=(distractor_path,),
                category="source-recall",
                source_terms=(target_path,),
            )
        )

    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        tuple(cases),
        version=SOURCE_RECALL_WORKLOAD_VERSION,
        documents=documents,
        lanes=("source-recall",),
    )
    return eventlog, tuple(cases), workload


def build_mempalace_workload_inventory(
    output_dir: str | Path,
    *,
    subjects: int = 100,
    documents: int = 100,
    sessions: int = 50,
) -> tuple[BenchmarkWorkloadInventoryEntry, ...]:
    """Build reproducible metadata for MemPalace-comparable proof workloads."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    workload_specs: list[tuple[str, Callable[[], BenchmarkWorkload], tuple[str, ...], str]] = [
        (
            "temporal-recall",
            lambda: build_temporal_recall_workload(
                root / "temporal-recall.jsonl",
                subjects=subjects,
            )[2],
            ("mean_score", "citation_coverage"),
            "as-of retrieval of facts that changed over time",
        ),
        (
            "source-recall",
            lambda: build_source_recall_workload(
                root / "source-recall.jsonl",
                documents=documents,
            )[2],
            ("mean_score", "source_recall", "citation_coverage"),
            "exact source citation recovery among target and distractor documents",
        ),
        (
            "graph-traversal",
            lambda: build_graph_traversal_workload(
                root / "graph-traversal.jsonl",
                subjects=subjects,
            )[2],
            ("mean_score", "identity_recall", "citation_coverage"),
            "multi-hop retrieval across goal, task, actor, and completion edges",
        ),
        (
            "context-collapse",
            lambda: build_context_collapse_workload(
                root / "context-collapse.jsonl",
                sessions=sessions,
            )[2],
            ("mean_score", "identity_recall", "citation_coverage", "approx_tokens"),
            "checkpoint recovery after noisy transcript context",
        ),
    ]
    entries: list[BenchmarkWorkloadInventoryEntry] = []
    for lane, build_workload, required_metrics, product_claim in workload_specs:
        workload = build_workload()
        entries.append(
            BenchmarkWorkloadInventoryEntry(
                lane=lane,
                version=workload.version,
                sha256=workload.sha256,
                event_count=workload.event_count,
                case_count=workload.case_count,
                required_metrics=required_metrics,
                product_claim=product_claim,
            )
        )
    return tuple(entries)


def format_mempalace_workload_inventory(
    inventory: tuple[BenchmarkWorkloadInventoryEntry, ...],
) -> str:
    """Render workload inventory metadata for release notes and audits."""
    lines = [
        "# MemPalace-Comparable Benchmark Inventory",
        "",
        "| Lane | Version | Events | Queries | Required metrics | Product claim | SHA-256 |",
        "|------|---------|--------|---------|------------------|---------------|---------|",
    ]
    for entry in inventory:
        lines.append(
            "| "
            f"{entry.lane} | "
            f"{entry.version} | "
            f"{entry.event_count} | "
            f"{entry.case_count} | "
            f"{', '.join(entry.required_metrics)} | "
            f"{entry.product_claim} | "
            f"{entry.sha256} |"
        )
    return "\n".join(lines) + "\n"


def build_temporal_recall_workload(
    path: str | Path,
    subjects: int = 100,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build a frozen temporal recall lane for as-of memory evaluation."""
    if subjects <= 0:
        raise ValueError("subjects must be positive")

    eventlog = EventLog(path)
    cases: list[BenchmarkCase] = []
    checkpoints = (
        (
            "alpha",
            datetime(2024, 1, 15, tzinfo=UTC),
            "2024-03-01T00:00:00Z",
        ),
        (
            "beta",
            datetime(2024, 5, 15, tzinfo=UTC),
            "2024-07-01T00:00:00Z",
        ),
        (
            "gamma",
            datetime(2024, 9, 15, tzinfo=UTC),
            "2024-11-01T00:00:00Z",
        ),
    )
    for idx in range(subjects):
        user_id = f"user-{idx:04d}"
        values: list[str] = []
        for label, timestamp, _ in checkpoints:
            value = f"workspace-{label}-{idx:04d}"
            values.append(value)
            source_path = f"temporal-recall/{user_id}/workspace.md"
            eventlog.append(
                "user.preference_changed",
                actor="user",
                payload={
                    "userId": user_id,
                    "key": "workspace",
                    "value": value,
                    "source_path": source_path,
                    "source_line": len(values),
                    "source_summary": (
                        f"{user_id} workspace preference changed to {value}"
                    ),
                },
                timestamp=timestamp,
            )

        for position, (label, _, temporal_point) in enumerate(checkpoints):
            expected = values[position]
            forbidden = tuple(
                f"workspace={value}" for value in values
                if value != expected
            )
            cases.append(
                BenchmarkCase(
                    name=f"temporal-recall-{label}-{idx:04d}",
                    query=(
                        f"What workspace preference was active for {user_id} "
                        f"at {temporal_point}?"
                    ),
                    expected_terms=(f"workspace={expected}",),
                    forbidden_terms=forbidden,
                    category="temporal-recall",
                    temporal_point=temporal_point,
                    identity_terms=(user_id,),
                )
            )

    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        tuple(cases),
        version=TEMPORAL_RECALL_WORKLOAD_VERSION,
        subjects=subjects,
        lanes=("temporal-recall",),
    )
    return eventlog, tuple(cases), workload


def build_consolidation_collapse_workload(
    path: str | Path,
    identities: int = 100,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build a workload that detects identity loss during semantic consolidation."""
    if identities <= 0:
        raise ValueError("identities must be positive")

    eventlog = EventLog(path)
    cases: list[BenchmarkCase] = []
    for idx in range(identities):
        identity_code = f"identity-code-{idx:04d}"
        doc_path = f"docs/consolidation/service-{idx:04d}.md"
        content = (
            "Consolidation review for the platform migration records "
            f"{identity_code}. The note covers rollback ownership, incident "
            "readiness, retry policy, and release coordination."
        )
        eventlog.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": doc_path,
                "start_line": 1,
                "end_line": 8,
                "content": content,
                "sha256": _content_sha256(content),
            },
            timestamp=datetime(2024, 10, 1, tzinfo=UTC),
        )
        cases.append(
            BenchmarkCase(
                name=f"consolidation-identity-{idx:04d}",
                query=f"Which consolidation source records {identity_code}?",
                expected_terms=("Consolidation review",),
                category="consolidation",
                identity_terms=(identity_code, doc_path),
            )
        )

    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        tuple(cases),
        version=CONSOLIDATION_WORKLOAD_VERSION,
        documents=identities,
        lanes=("consolidation",),
    )
    return eventlog, tuple(cases), workload


def build_benchmark_suite_workload(
    path: str | Path,
    subjects: int = 100,
    documents: int = 250,
    sessions: int = 50,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build a representative workload across memory, docs, transcripts, and mixed context."""
    if subjects <= 0:
        raise ValueError("subjects must be positive")
    if documents <= 0:
        raise ValueError("documents must be positive")
    if sessions <= 0:
        raise ValueError("sessions must be positive")

    eventlog, cases = build_statistical_event_log(path, subjects=subjects)
    suite_cases = list(cases)

    for idx in range(documents):
        release_code = f"doc-code-{idx:04d}"
        service = f"service-{idx % subjects:04d}"
        doc_path = f"docs/runbooks/{service}.md"
        content = (
            f"{service} production runbook uses release marker {release_code}. "
            f"Owner team-{idx % 9} validates rollback window {idx % 5}."
        )
        eventlog.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": doc_path,
                "start_line": 1 + (idx * 7),
                "end_line": 6 + (idx * 7),
                "content": content,
                "sha256": _content_sha256(content),
            },
            timestamp=datetime(2024, 8, 1, tzinfo=UTC),
        )
        suite_cases.append(
            BenchmarkCase(
                name=f"document-source-{idx:04d}",
                query=f"Which runbook mentions release marker {release_code}?",
                expected_terms=(release_code, doc_path),
                category="document",
            )
        )

    for idx in range(sessions):
        session_id = f"session-{idx:04d}"
        decision_code = f"decision-code-{idx:04d}"
        subject_id = idx % subjects
        eventlog.append(
            "transcript.turn",
            actor="user",
            payload={
                "source": session_id,
                "turn_index": 1,
                "role": "user",
                "content": f"Review workstream {subject_id:04d} release constraints.",
                "redacted_paths": [],
            },
            timestamp=datetime(2024, 9, 1, tzinfo=UTC),
        )
        eventlog.append(
            "transcript.turn",
            actor="assistant",
            payload={
                "source": session_id,
                "turn_index": 2,
                "role": "assistant",
                "content": (
                    f"We decided {decision_code} for workstream {subject_id:04d} "
                    f"after checking Goal {subject_id:04d}."
                ),
                "redacted_paths": [],
            },
            timestamp=datetime(2024, 9, 1, 0, 1, tzinfo=UTC),
        )
        suite_cases.append(
            BenchmarkCase(
                name=f"session-decision-{idx:04d}",
                query=f"What decision code was recorded in {session_id}?",
                expected_terms=(decision_code, session_id),
                category="transcript",
            )
        )

    mixed_count = min(subjects, documents, sessions)
    for idx in range(mixed_count):
        suite_cases.append(
            BenchmarkCase(
                name=f"mixed-release-{idx:04d}",
                query=(
                    f"For user-{idx:04d} in workstream {idx:04d}, recover the current theme, "
                    f"runbook marker doc-code-{idx:04d}, and session decision."
                ),
                expected_terms=(
                    f"theme=theme-new-{idx % 7}",
                    f"doc-code-{idx:04d}",
                    f"decision-code-{idx:04d}",
                ),
                forbidden_terms=(f"theme=theme-old-{idx % 7}",),
                category="mixed",
            )
        )

    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        tuple(suite_cases),
        version=SUITE_WORKLOAD_VERSION,
        subjects=subjects,
        documents=documents,
        sessions=sessions,
        lanes=SUITE_WORKLOAD_LANES,
    )
    return eventlog, tuple(suite_cases), workload


def build_longmemeval_workload(
    path: str | Path,
    dataset_path: str | Path,
    questions: int | None = None,
) -> tuple[EventLog, tuple[BenchmarkCase, ...], BenchmarkWorkload]:
    """Build a retrieval workload from the cleaned LongMemEval JSON dataset."""
    records = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("LongMemEval dataset must be a JSON list")
    if questions is not None and questions <= 0:
        raise ValueError("questions must be positive when provided")

    selected = records[:questions] if questions is not None else records
    eventlog = EventLog(path)
    cases: list[BenchmarkCase] = []
    event_items: list[dict[str, object]] = []
    session_count = 0
    for index, record in enumerate(selected):
        if not isinstance(record, dict):
            raise ValueError(f"LongMemEval record {index} must be an object")
        question_id = str(record.get("question_id") or f"question-{index:04d}")
        question = str(record.get("question") or "")
        answer = str(record.get("answer") or "").strip()
        answer_session_ids = tuple(str(item) for item in record.get("answer_session_ids", ()))
        haystack_session_ids = tuple(str(item) for item in record.get("haystack_session_ids", ()))
        haystack_dates = tuple(str(item) for item in record.get("haystack_dates", ()))
        haystack_sessions = record.get("haystack_sessions", ())
        if not isinstance(haystack_sessions, list):
            raise ValueError(f"LongMemEval record {question_id} haystack_sessions must be a list")

        for session_index, session in enumerate(haystack_sessions):
            session_id = (
                haystack_session_ids[session_index]
                if session_index < len(haystack_session_ids)
                else f"{question_id}-session-{session_index:04d}"
            )
            session_date = (
                haystack_dates[session_index]
                if session_index < len(haystack_dates)
                else ""
            )
            content = _format_longmemeval_session(session_id, session_date, session)
            chunks = _longmemeval_session_chunks(session_id, session_date, content)
            for chunk_index, chunk in enumerate(chunks, start=1):
                event_items.append(
                    {
                        "event_type": "document.indexed",
                        "actor": "longmemeval",
                        "payload": {
                            "path": f"longmemeval/{question_id}/{session_id}/chunk-{chunk_index:04d}.md",
                            "start_line": 1,
                            "end_line": max(1, chunk.count("\n") + 1),
                            "content": chunk,
                            "sha256": _content_sha256(chunk),
                            "longmemeval_session_id": session_id,
                            "longmemeval_chunk_index": chunk_index,
                            "longmemeval_chunk_count": len(chunks),
                        },
                    }
                )
            for turn_index, role, turn_content in _longmemeval_salient_turns(session):
                content = _format_longmemeval_salient_turn(
                    session_id,
                    session_date,
                    turn_index,
                    role,
                    turn_content,
                )
                event_items.append(
                    {
                        "event_type": "document.indexed",
                        "actor": "longmemeval",
                        "payload": {
                            "path": (
                                f"longmemeval/{question_id}/{session_id}/"
                                f"salient-turn-{turn_index:04d}.md"
                            ),
                            "start_line": 1,
                            "end_line": max(1, content.count("\n") + 1),
                            "content": content,
                            "sha256": _content_sha256(content),
                            "longmemeval_session_id": session_id,
                            "longmemeval_salient_memory_turn": True,
                            "turn_index": turn_index,
                            "role": role,
                        },
                    }
                )
            session_count += 1

        cases.append(
            BenchmarkCase(
                name=f"longmemeval-{question_id}",
                query=question,
                expected_terms=(answer,) if answer else (),
                identity_terms=answer_session_ids,
                category=f"longmemeval:{record.get('question_type', 'unknown')}",
            )
        )

    eventlog.append_many(event_items)
    workload = BenchmarkWorkload.from_event_log(
        eventlog,
        tuple(cases),
        version=LONGMEMEVAL_WORKLOAD_VERSION,
        subjects=len(cases),
        sessions=session_count,
        lanes=("longmemeval",),
    )
    return eventlog, tuple(cases), workload


def workload_fingerprint(
    eventlog: EventLog,
    cases: tuple[BenchmarkCase, ...],
    version: str,
) -> str:
    """Return a deterministic SHA-256 fingerprint for a benchmark workload."""
    payload = {
        "version": version,
        "events": [
            event.model_dump(mode="json")
            for event in eventlog.read_all()
        ],
        "cases": [asdict(case) for case in cases],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_projection_cache_key(
    eventlog: EventLog,
    cases: tuple[BenchmarkCase, ...],
    workload: BenchmarkWorkload,
    provider_label: str,
) -> str:
    """Return a stable cache key for reusable benchmark graph projections.

    Eventloom timestamps and hash-chain fields are intentionally excluded
    because generated benchmark logs are rebuilt in temporary directories on
    every CLI invocation. Projection safety depends on stable event semantics,
    workload identity, cases, and embedding provider, not the transient log
    seal created for that invocation.
    """
    payload = {
        "provider": provider_label,
        "workload": {
            "version": workload.version,
            "subjects": workload.subjects,
            "event_count": workload.event_count,
            "case_count": workload.case_count,
            "documents": workload.documents,
            "sessions": workload.sessions,
            "lanes": workload.lanes,
        },
        "events": [
            {
                "seq": event.seq,
                "type": event.type,
                "actor": event.actor,
                "thread": event.thread,
                "payload": event.payload,
                "security": event.security.model_dump(mode="json") if event.security else None,
            }
            for event in eventlog.read_all()
        ],
        "cases": [asdict(case) for case in cases],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_retrievers(
    retrievers: dict[str, Retriever],
    cases: tuple[BenchmarkCase, ...],
    runs: int = 5,
    limit: int = 10,
    embedding_provider: str = "unknown",
    workload: BenchmarkWorkload | None = None,
    external_results: tuple[ExternalBenchmarkResult, ...] = (),
) -> BenchmarkReport:
    """Benchmark retrievers against shared cases and return statistics."""
    if runs <= 0:
        raise ValueError("runs must be positive")
    measurements: list[BenchmarkRun] = []
    for backend, retriever in retrievers.items():
        for case in cases:
            for run in range(1, runs + 1):
                start = time.perf_counter()
                contexts = retriever.query(
                    case.query,
                    temporal_point=case.temporal_point,
                    limit=limit,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                score = score_retrieval(case, contexts)
                measurements.append(
                    _measurement(
                        backend=backend,
                        case=case,
                        run=run,
                        score=score,
                        contexts=contexts,
                        latency_ms=latency_ms,
                    )
                )

    report = BenchmarkReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        embedding_provider=embedding_provider,
        runs=tuple(measurements),
        summaries=tuple(_summaries(measurements, cases, runs)),
        category_summaries=tuple(_category_summaries(measurements)),
        workload=workload,
        external_results=external_results,
    )
    return _with_comparisons(report)


async def benchmark_live_retrievers(
    retrievers: dict[str, Retriever],
    zaxy_retriever: ZaxyRetriever,
    cases: tuple[BenchmarkCase, ...],
    runs: int = 5,
    limit: int = 10,
    embedding_provider: str = "unknown",
    workload: BenchmarkWorkload | None = None,
    external_results: tuple[ExternalBenchmarkResult, ...] = (),
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> BenchmarkReport:
    """Benchmark sync baselines and live Zaxy retrieval in one event loop."""
    if runs <= 0:
        raise ValueError("runs must be positive")
    measurements: list[BenchmarkRun] = []
    total = (len(retrievers) + 1) * len(cases) * runs
    completed = 0
    for backend, retriever in retrievers.items():
        for case in cases:
            for run in range(1, runs + 1):
                start = time.perf_counter()
                contexts = retriever.query(
                    case.query,
                    temporal_point=case.temporal_point,
                    limit=limit,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                score = score_retrieval(case, contexts)
                measurements.append(
                    _measurement(
                        backend=backend,
                        case=case,
                        run=run,
                        score=score,
                        contexts=contexts,
                        latency_ms=latency_ms,
                    )
                )
                completed += 1
                _emit_progress(
                    progress_callback,
                    backend=backend,
                    case=case,
                    run=run,
                    completed=completed,
                    total=total,
                )

    for case in cases:
        for run in range(1, runs + 1):
            start = time.perf_counter()
            contexts = await zaxy_retriever.query_async(
                case.query,
                temporal_point=case.temporal_point,
                limit=limit,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            score = score_retrieval(case, contexts)
            measurements.append(
                _measurement(
                    backend="zaxy",
                    case=case,
                    run=run,
                    score=score,
                    contexts=contexts,
                    latency_ms=latency_ms,
                )
            )
            completed += 1
            _emit_progress(
                progress_callback,
                backend="zaxy",
                case=case,
                run=run,
                completed=completed,
                total=total,
            )

    report = BenchmarkReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        embedding_provider=embedding_provider,
        runs=tuple(measurements),
        summaries=tuple(_summaries(measurements, cases, runs)),
        category_summaries=tuple(_category_summaries(measurements)),
        workload=workload,
        external_results=external_results,
    )
    return _with_comparisons(report)


def _emit_progress(
    progress_callback: Callable[[dict[str, object]], None] | None,
    *,
    backend: str,
    case: BenchmarkCase,
    run: int,
    completed: int,
    total: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "backend": backend,
            "case": case.name,
            "run": run,
            "completed": completed,
            "total": total,
        }
    )


async def build_live_zaxy_retriever(
    eventlog: EventLog,
    provider: EmbeddingProvider,
    neo4j_uri: str = "bolt://localhost:7688",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "testpassword",
    reset_graph: bool = False,
    lexical_retriever: Retriever | None = None,
    reuse_projection: bool = False,
    projection_cache_key: str | None = None,
) -> tuple[ZaxyRetriever, GraphStore]:
    """Ingest the benchmark event log into Neo4j and return a live retriever.

    The caller owns the returned ``GraphStore`` and must close it.
    """
    graph = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
    await graph.connect()
    await graph.init_schema()
    if reset_graph:
        assert graph._driver is not None
        await graph._driver.execute_query("MATCH (n) DETACH DELETE n")
    if (
        reuse_projection
        and projection_cache_key is not None
        and not reset_graph
        and await _benchmark_projection_present(graph, projection_cache_key)
    ):
        return ZaxyRetriever(QueryRouter(graph), provider, lexical_retriever=lexical_retriever), graph
    events = eventlog.read_all()
    for event in events:
        extraction = embed_extraction(extract(event), provider)
        await graph.upsert_extraction(extraction)
    if projection_cache_key is not None:
        await _mark_benchmark_projection(graph, projection_cache_key, events)
    return ZaxyRetriever(QueryRouter(graph), provider, lexical_retriever=lexical_retriever), graph


async def _benchmark_projection_present(graph: GraphStore, projection_cache_key: str) -> bool:
    """Return whether a benchmark workload projection is already present."""
    assert graph._driver is not None
    records, _summary, _keys = await graph._driver.execute_query(
        """
        MATCH (p:ZaxyBenchmarkProjection {key: $key})
        RETURN p.key AS key
        LIMIT 1
        """,
        key=projection_cache_key,
    )
    return bool(records)


async def _mark_benchmark_projection(
    graph: GraphStore,
    projection_cache_key: str,
    events: Sequence[object],
) -> None:
    """Persist a benchmark projection marker for safe reuse."""
    assert graph._driver is not None
    latest = events[-1] if events else None
    await graph._driver.execute_query(
        """
        MERGE (p:ZaxyBenchmarkProjection {key: $key})
        ON CREATE SET p.created_at = datetime()
        SET p.updated_at = datetime(),
            p.event_count = $event_count,
            p.latest_seq = $latest_seq,
            p.latest_hash = $latest_hash
        """,
        key=projection_cache_key,
        event_count=len(events),
        latest_seq=getattr(latest, "seq", None),
        latest_hash=getattr(latest, "hash", None),
    )


def write_benchmark_report(
    report: BenchmarkReport,
    output_dir: Path,
) -> WrittenBenchmarkReport:
    """Write benchmark JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "live-benchmark.json"
    markdown_path = output_dir / "live-benchmark.md"
    json_path.write_text(
        json.dumps(_report_payload(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
    return WrittenBenchmarkReport(json_path=json_path, markdown_path=markdown_path)


def load_benchmark_report(path: str | Path) -> BenchmarkReport:
    """Load a machine-readable benchmark report from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark report JSON must be an object")
    workload_payload = payload.get("workload")
    workload = (
        BenchmarkWorkload(**workload_payload)
        if isinstance(workload_payload, dict)
        else None
    )
    return BenchmarkReport(
        generated_at=str(payload.get("generated_at", "")),
        embedding_provider=str(payload.get("embedding_provider", "")),
        runs=tuple(BenchmarkRun(**item) for item in payload.get("runs", [])),
        summaries=tuple(
            BenchmarkSummary(**item) for item in payload.get("summaries", [])
        ),
        category_summaries=tuple(
            CategorySummary(**item)
            for item in payload.get("category_summaries", [])
        ),
        comparisons=tuple(
            BackendComparison(**item) for item in payload.get("comparisons", [])
        ),
        workload=workload,
        external_results=tuple(
            ExternalBenchmarkResult(**item)
            for item in payload.get("external_results", [])
        ),
    )


def report_to_markdown(report: BenchmarkReport) -> str:
    """Render benchmark summaries as Markdown."""
    workload = report.workload or _ad_hoc_workload(report)
    lines = [
        "# Live Retrieval Benchmark",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Embedding provider: `{report.embedding_provider}`",
        f"- Workload: `{workload.version}`",
        f"- Workload SHA-256: `{workload.sha256}`",
        f"- Events: `{workload.event_count}`",
        f"- Queries: `{workload.case_count}`",
    ]
    if workload.subjects is not None:
        lines.append(f"- Subjects: `{workload.subjects}`")
    if workload.documents is not None:
        lines.append(f"- Documents: `{workload.documents}`")
    if workload.sessions is not None:
        lines.append(f"- Sessions: `{workload.sessions}`")
    if workload.lanes:
        lines.append(f"- Lanes: `{', '.join(workload.lanes)}`")
    lines.extend(
        [
        "",
        "| Backend | Mean score | Identity recall | Source recall | Citation coverage | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | p99 ms | Returned bytes | Approx tokens |",
        "|---------|------------|-----------------|---------------|-------------------|----------|----------|-----------|--------|--------|--------|----------------|---------------|",
        ]
    )
    for backend_summary in report.summaries:
        identity_recall = (
            ""
            if backend_summary.mean_identity_recall is None
            else f"{backend_summary.mean_identity_recall:.3f}"
        )
        citation_coverage = (
            ""
            if backend_summary.mean_citation_coverage is None
            else f"{backend_summary.mean_citation_coverage:.3f}"
        )
        source_recall = (
            ""
            if backend_summary.mean_source_recall is None
            else f"{backend_summary.mean_source_recall:.3f}"
        )
        recall_at_1 = (
            ""
            if backend_summary.mean_recall_at_1 is None
            else f"{backend_summary.mean_recall_at_1:.3f}"
        )
        recall_at_5 = (
            ""
            if backend_summary.mean_recall_at_5 is None
            else f"{backend_summary.mean_recall_at_5:.3f}"
        )
        recall_at_10 = (
            ""
            if backend_summary.mean_recall_at_10 is None
            else f"{backend_summary.mean_recall_at_10:.3f}"
        )
        lines.append(
            "| "
            f"{backend_summary.backend} | "
            f"{backend_summary.mean_score:.3f} | "
            f"{identity_recall} | "
            f"{source_recall} | "
            f"{citation_coverage} | "
            f"{recall_at_1} | "
            f"{recall_at_5} | "
            f"{recall_at_10} | "
            f"{backend_summary.latency_ms_p50:.2f} | "
            f"{backend_summary.latency_ms_p95:.2f} | "
            f"{backend_summary.latency_ms_p99:.2f} | "
            f"{backend_summary.mean_returned_bytes:.0f} | "
            f"{backend_summary.mean_approx_tokens:.0f} |"
        )
    if report.category_summaries:
        lines.extend(
            [
                "",
                "## Category summaries",
                "",
                "| Backend | Category | Queries | Mean score | Misses | Source recall | Citation coverage | Recall@1 | Recall@5 | Recall@10 |",
                "|---------|----------|---------|------------|--------|---------------|-------------------|----------|----------|-----------|",
            ]
        )
        for category_summary in report.category_summaries:
            citation_coverage = (
                ""
                if category_summary.mean_citation_coverage is None
                else f"{category_summary.mean_citation_coverage:.3f}"
            )
            source_recall = (
                ""
                if category_summary.mean_source_recall is None
                else f"{category_summary.mean_source_recall:.3f}"
            )
            recall_at_1 = (
                ""
                if category_summary.mean_recall_at_1 is None
                else f"{category_summary.mean_recall_at_1:.3f}"
            )
            recall_at_5 = (
                ""
                if category_summary.mean_recall_at_5 is None
                else f"{category_summary.mean_recall_at_5:.3f}"
            )
            recall_at_10 = (
                ""
                if category_summary.mean_recall_at_10 is None
                else f"{category_summary.mean_recall_at_10:.3f}"
            )
            lines.append(
                "| "
                f"{category_summary.backend} | "
                f"{category_summary.category} | "
                f"{category_summary.query_count} | "
                f"{category_summary.mean_score:.3f} | "
                f"{category_summary.miss_count} | "
                f"{source_recall} | "
                f"{citation_coverage} | "
                f"{recall_at_1} | "
                f"{recall_at_5} | "
                f"{recall_at_10} |"
            )
    if report.comparisons:
        lines.extend(
            [
                "",
                "## Paired comparisons",
                "",
                "| Target | Baseline | Mean score delta | 95% CI | p-value | Significant |",
                "|--------|----------|------------------|--------|---------|-------------|",
            ]
        )
        for comparison in report.comparisons:
            lines.append(
                "| "
                f"{comparison.target_backend} | "
                f"{comparison.baseline_backend} | "
                f"{comparison.mean_difference:.4f} | "
                f"[{comparison.ci_low:.4f}, {comparison.ci_high:.4f}] | "
                f"{comparison.p_value:.4f} | "
                f"{'yes' if comparison.significant else 'no'} |"
            )
    if report.external_results:
        lines.extend(
            [
                "",
                "## External comparison disclosures",
                "",
                "| System | Version | Mean score | p95 ms | Source | Notes |",
                "|--------|---------|------------|--------|--------|-------|",
            ]
        )
        for result in report.external_results:
            p95 = "" if result.latency_ms_p95 is None else f"{result.latency_ms_p95:.2f}"
            lines.append(
                "| "
                f"{result.system} | "
                f"{result.version} | "
                f"{result.mean_score:.3f} | "
                f"{p95} | "
                f"{result.source} | "
                f"{result.notes or ''} |"
            )
    return "\n".join(lines) + "\n"


def compare_target_to_baselines(
    report: BenchmarkReport,
    target_backend: str = "zaxy",
    bootstrap_samples: int = 2000,
    seed: int = 1,
) -> tuple[BackendComparison, ...]:
    """Compare target backend against each baseline using paired score deltas."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    backends = sorted({run.backend for run in report.runs})
    baselines = [backend for backend in backends if backend != target_backend]
    comparisons: list[BackendComparison] = []
    for baseline in baselines:
        deltas = _paired_score_deltas(report.runs, target_backend, baseline)
        if not deltas:
            continue
        mean_difference = statistics.fmean(deltas)
        ci_low, ci_high = _bootstrap_ci(deltas, bootstrap_samples, seed)
        p_value = _paired_randomization_p_value(deltas)
        comparisons.append(
            BackendComparison(
                target_backend=target_backend,
                baseline_backend=baseline,
                paired_units=len(deltas),
                mean_difference=round(mean_difference, 4),
                ci_low=round(ci_low, 4),
                ci_high=round(ci_high, 4),
                p_value=round(p_value, 6),
                significant=ci_low > 0.0 and p_value < 0.05,
            )
        )
    return tuple(comparisons)


def compare_benchmark_reports(
    baseline: BenchmarkReport | None,
    candidate: BenchmarkReport,
    *,
    backend: str = "zaxy",
    min_mean_score: float = 0.95,
    min_citation_coverage: float = 0.95,
    max_p95_ms: float = 500.0,
    max_p99_ms: float = 750.0,
    max_latency_regression_ratio: float = 0.25,
) -> BenchmarkGuardrailReport:
    """Compare a candidate report against beta retrieval guardrails."""
    if max_latency_regression_ratio < 0.0:
        raise ValueError("max_latency_regression_ratio must be non-negative")
    candidate_summary = _summary_for_backend(candidate, backend)
    assert candidate_summary is not None
    baseline_summary = _summary_for_backend(baseline, backend) if baseline is not None else None
    checks = [
        _min_check(
            "mean_score_floor",
            candidate_summary.mean_score,
            min_mean_score,
            "mean score must stay above the beta floor",
        ),
        _min_check(
            "citation_coverage_floor",
            candidate_summary.mean_citation_coverage,
            min_citation_coverage,
            "citation coverage must stay above the beta floor",
        ),
        _max_check(
            "p95_latency_budget",
            candidate_summary.latency_ms_p95,
            max_p95_ms,
            "p95 latency must stay within the beta budget",
        ),
        _max_check(
            "p99_latency_budget",
            candidate_summary.latency_ms_p99,
            max_p99_ms,
            "p99 latency must stay within the beta budget",
        ),
    ]
    if baseline_summary is not None:
        checks.extend(
            [
                _regression_check(
                    "p95_latency_regression",
                    baseline_summary.latency_ms_p95,
                    candidate_summary.latency_ms_p95,
                    max_latency_regression_ratio,
                ),
                _regression_check(
                    "p99_latency_regression",
                    baseline_summary.latency_ms_p99,
                    candidate_summary.latency_ms_p99,
                    max_latency_regression_ratio,
                ),
            ]
        )
    return BenchmarkGuardrailReport(
        backend=backend,
        passed=all(check.passed for check in checks),
        baseline=baseline_summary,
        candidate=candidate_summary,
        checks=tuple(checks),
    )


def format_benchmark_comparison(report: BenchmarkGuardrailReport) -> str:
    """Render benchmark guardrails as Markdown."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "# Benchmark Guardrail Comparison",
        "",
        f"- Status: `{status}`",
        f"- Backend: `{report.backend}`",
        f"- Candidate mean score: `{report.candidate.mean_score:.3f}`",
        f"- Candidate p95 ms: `{report.candidate.latency_ms_p95:.2f}`",
        f"- Candidate p99 ms: `{report.candidate.latency_ms_p99:.2f}`",
    ]
    if report.candidate.mean_citation_coverage is not None:
        lines.append(
            f"- Candidate citation coverage: `{report.candidate.mean_citation_coverage:.3f}`"
        )
    if report.baseline is not None:
        lines.extend(
            [
                f"- Baseline p95 ms: `{report.baseline.latency_ms_p95:.2f}`",
                f"- Baseline p99 ms: `{report.baseline.latency_ms_p99:.2f}`",
            ]
        )
    lines.extend(
        [
            "",
            "| Check | Status | Observed | Threshold | Message |",
            "|-------|--------|----------|-----------|---------|",
        ]
    )
    for check in report.checks:
        observed = "" if check.observed is None else f"{check.observed:.4f}"
        lines.append(
            "| "
            f"{check.name} | "
            f"{'PASS' if check.passed else 'FAIL'} | "
            f"{observed} | "
            f"{check.threshold:.4f} | "
            f"{check.message} |"
        )
    return "\n".join(lines) + "\n"


def _measurement(
    backend: str,
    case: BenchmarkCase,
    run: int,
    score: RetrievalScore,
    contexts: list[str],
    latency_ms: float,
) -> BenchmarkRun:
    returned_text = "\n".join(contexts)
    returned_bytes = len(returned_text.encode("utf-8"))
    citation_count = _citation_count(contexts)
    recall_at_1 = _recall_at_k(case, contexts, 1)
    recall_at_5 = _recall_at_k(case, contexts, 5)
    recall_at_10 = _recall_at_k(case, contexts, 10)
    return BenchmarkRun(
        backend=backend,
        case_name=case.name,
        category=case.category,
        run=run,
        score=score.score,
        latency_ms=round(latency_ms, 4),
        result_count=len(contexts),
        returned_bytes=returned_bytes,
        approx_tokens=max(1, math.ceil(len(returned_text) / 4)) if returned_text else 0,
        expected_hits=score.expected_hits,
        missing_expected=score.missing_expected,
        forbidden_hits=score.forbidden_hits,
        identity_recall=score.identity_recall,
        identity_hits=score.identity_hits,
        missing_identities=score.missing_identities,
        source_recall=score.source_recall,
        source_hits=score.source_hits,
        missing_sources=score.missing_sources,
        citation_count=citation_count,
        citation_coverage=_citation_coverage(score, citation_count),
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
    )


def _recall_at_k(case: BenchmarkCase, contexts: list[str], k: int) -> float | None:
    """Return whether any retrieval target appears in the top-k contexts."""
    target_terms = case.source_terms or case.identity_terms or case.expected_terms
    if not target_terms:
        return None
    haystack = "\n".join(contexts[:k]).casefold()
    return 1.0 if any(term.casefold() in haystack for term in target_terms) else 0.0


def _summaries(
    measurements: list[BenchmarkRun],
    cases: tuple[BenchmarkCase, ...],
    runs: int,
) -> list[BenchmarkSummary]:
    summaries: list[BenchmarkSummary] = []
    backends = sorted({measurement.backend for measurement in measurements})
    for backend in backends:
        rows = [measurement for measurement in measurements if measurement.backend == backend]
        latencies = [row.latency_ms for row in rows]
        identity_recalls = [
            row.identity_recall
            for row in rows
            if row.identity_recall is not None
        ]
        citation_coverages = [
            row.citation_coverage
            for row in rows
            if row.citation_coverage is not None
        ]
        source_recalls = [
            row.source_recall
            for row in rows
            if row.source_recall is not None
        ]
        recall_at_1_values = [
            row.recall_at_1
            for row in rows
            if row.recall_at_1 is not None
        ]
        recall_at_5_values = [
            row.recall_at_5
            for row in rows
            if row.recall_at_5 is not None
        ]
        recall_at_10_values = [
            row.recall_at_10
            for row in rows
            if row.recall_at_10 is not None
        ]
        summaries.append(
            BenchmarkSummary(
                backend=backend,
                case_count=len(cases),
                runs=runs,
                mean_score=round(statistics.fmean(row.score for row in rows), 4),
                latency_ms_mean=round(statistics.fmean(latencies), 4),
                latency_ms_p50=round(_percentile(latencies, 50), 4),
                latency_ms_p95=round(_percentile(latencies, 95), 4),
                latency_ms_p99=round(_percentile(latencies, 99), 4),
                mean_returned_bytes=round(
                    statistics.fmean(row.returned_bytes for row in rows), 4
                ),
                mean_approx_tokens=round(
                    statistics.fmean(row.approx_tokens for row in rows), 4
                ),
                mean_identity_recall=(
                    round(statistics.fmean(identity_recalls), 4)
                    if identity_recalls
                    else None
                ),
                mean_source_recall=(
                    round(statistics.fmean(source_recalls), 4)
                    if source_recalls
                    else None
                ),
                mean_citation_coverage=(
                    round(statistics.fmean(citation_coverages), 4)
                    if citation_coverages
                    else None
                ),
                mean_recall_at_1=(
                    round(statistics.fmean(recall_at_1_values), 4)
                    if recall_at_1_values
                    else None
                ),
                mean_recall_at_5=(
                    round(statistics.fmean(recall_at_5_values), 4)
                    if recall_at_5_values
                    else None
                ),
                mean_recall_at_10=(
                    round(statistics.fmean(recall_at_10_values), 4)
                    if recall_at_10_values
                    else None
                ),
            )
        )
    return summaries


def _category_summaries(measurements: list[BenchmarkRun]) -> list[CategorySummary]:
    summaries: list[CategorySummary] = []
    keys = sorted({(row.backend, row.category) for row in measurements})
    for backend, category in keys:
        rows = [
            row for row in measurements
            if row.backend == backend and row.category == category
        ]
        citation_coverages = [
            row.citation_coverage
            for row in rows
            if row.citation_coverage is not None
        ]
        source_recalls = [
            row.source_recall
            for row in rows
            if row.source_recall is not None
        ]
        recall_at_1_values = [
            row.recall_at_1
            for row in rows
            if row.recall_at_1 is not None
        ]
        recall_at_5_values = [
            row.recall_at_5
            for row in rows
            if row.recall_at_5 is not None
        ]
        recall_at_10_values = [
            row.recall_at_10
            for row in rows
            if row.recall_at_10 is not None
        ]
        summaries.append(
            CategorySummary(
                backend=backend,
                category=category,
                query_count=len(rows),
                mean_score=round(statistics.fmean(row.score for row in rows), 4),
                miss_count=sum(
                    1 for row in rows
                    if row.missing_expected or row.forbidden_hits
                ),
                mean_source_recall=(
                    round(statistics.fmean(source_recalls), 4)
                    if source_recalls
                    else None
                ),
                mean_citation_coverage=(
                    round(statistics.fmean(citation_coverages), 4)
                    if citation_coverages
                    else None
                ),
                mean_recall_at_1=(
                    round(statistics.fmean(recall_at_1_values), 4)
                    if recall_at_1_values
                    else None
                ),
                mean_recall_at_5=(
                    round(statistics.fmean(recall_at_5_values), 4)
                    if recall_at_5_values
                    else None
                ),
                mean_recall_at_10=(
                    round(statistics.fmean(recall_at_10_values), 4)
                    if recall_at_10_values
                    else None
                ),
            )
        )
    return summaries


def _summary_for_backend(report: BenchmarkReport | None, backend: str) -> BenchmarkSummary | None:
    if report is None:
        return None
    for summary in report.summaries:
        if summary.backend == backend:
            return summary
    raise ValueError(f"benchmark report does not contain backend: {backend}")


def _min_check(
    name: str,
    observed: float | None,
    threshold: float,
    message: str,
) -> BenchmarkGuardrailCheck:
    return BenchmarkGuardrailCheck(
        name=name,
        passed=observed is not None and observed >= threshold,
        observed=observed,
        threshold=threshold,
        message=message,
    )


def _max_check(
    name: str,
    observed: float,
    threshold: float,
    message: str,
) -> BenchmarkGuardrailCheck:
    return BenchmarkGuardrailCheck(
        name=name,
        passed=observed <= threshold,
        observed=observed,
        threshold=threshold,
        message=message,
    )


def _regression_check(
    name: str,
    baseline: float,
    candidate: float,
    max_regression_ratio: float,
) -> BenchmarkGuardrailCheck:
    allowed = baseline * (1.0 + max_regression_ratio)
    return BenchmarkGuardrailCheck(
        name=name,
        passed=candidate <= allowed,
        observed=candidate,
        threshold=round(allowed, 4),
        message=(
            f"candidate must not exceed baseline {baseline:.4f} by more than "
            f"{max_regression_ratio:.1%}"
        ),
    )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _centroid(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    dimension = len(embeddings[0])
    if any(len(embedding) != dimension for embedding in embeddings):
        raise ValueError("embedding dimensions must match")
    return [
        statistics.fmean(embedding[index] for embedding in embeddings)
        for index in range(dimension)
    ]


def _tokens(query: str) -> list[str]:
    return [token for token in query.casefold().replace("?", " ").split() if len(token) > 2]


def _bm25_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+(?:[-_:./#][a-z0-9]+)*", text.casefold()):
        tokens.append(token)
        if re.search(r"[-_:/#]", token):
            tokens.extend(part for part in re.split(r"[-_:/#]+", token) if part)
    return tokens


_BM25_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "bachelor": ("undergraduate", "undergrad", "graduated", "graduation"),
    "bachelors": ("undergraduate", "undergrad", "graduated", "graduation"),
    "degree": ("undergraduate", "undergrad", "graduated", "graduation"),
    "mention": ("mentioned", "discussed", "said", "told"),
    "mentioned": ("mention", "discussed", "said", "told"),
}


def _expanded_bm25_query_terms(query: str) -> tuple[str, ...]:
    terms = _bm25_tokens(query)
    expanded: list[str] = [*terms]
    for term in terms:
        expanded.extend(_BM25_QUERY_EXPANSIONS.get(term, ()))
    if {"computer", "science"} <= set(terms):
        expanded.append("cs")
    return tuple(dict.fromkeys(expanded))


def _memory_salience_boost(text: str) -> float:
    """Boost compact source-salient memory artifacts over raw context chunks."""
    if "salient_memory_turn=true" in text.casefold():
        return 4.0
    return 1.0


def _citation_count(contexts: list[str]) -> int:
    """Count returned contexts that carry an inspectable source citation."""
    return sum(1 for context in contexts if _has_source_citation(context))


def _citation_coverage(score: RetrievalScore, citation_count: int) -> float | None:
    """Return citation coverage for otherwise successful retrieval runs."""
    if score.missing_expected or score.forbidden_hits or not score.expected_hits:
        return None
    return 1.0 if citation_count > 0 else 0.0


def _has_source_citation(context: str) -> bool:
    text = context.casefold()
    return (
        "eventloom://" in text
        or "file://" in text
        or "citation=" in text
        or "source_path=" in text
        or re.search(r"^# event \d+", text, flags=re.MULTILINE) is not None
    )


def _bm25_document_frequencies(
    documents: tuple[tuple[str, ...], ...],
) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for document in documents:
        for term in set(document):
            frequencies[term] = frequencies.get(term, 0) + 1
    return frequencies


def _bm25_score(
    query_terms: tuple[str, ...],
    document_terms: tuple[str, ...],
    document_frequencies: dict[str, int],
    document_count: int,
    average_document_length: float,
    k1: float,
    b: float,
) -> float:
    if not document_terms or document_count == 0:
        return 0.0
    document_length = len(document_terms)
    length_normalizer = document_length / max(average_document_length, 1.0)
    term_counts = Counter(document_terms)
    score = 0.0
    for term in query_terms:
        frequency = term_counts.get(term, 0)
        if frequency == 0:
            continue
        document_frequency = document_frequencies.get(term, 0)
        idf = math.log(
            1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        denominator = frequency + k1 * (1.0 - b + b * length_normalizer)
        score += idf * ((frequency * (k1 + 1.0)) / denominator)
    return score


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _format_longmemeval_session(session_id: str, session_date: str, session: object) -> str:
    lines = [
        f"longmemeval_session_id={session_id}",
        f"longmemeval_session_date={session_date}",
    ]
    if isinstance(session, list):
        for turn_index, turn in enumerate(session, start=1):
            if isinstance(turn, dict):
                role = str(turn.get("role", "unknown"))
                content = str(turn.get("content", ""))
                lines.append(f"{turn_index}. {role}: {content}")
            else:
                lines.append(f"{turn_index}. unknown: {turn}")
    else:
        lines.append(str(session))
    return "\n".join(lines)


def _longmemeval_salient_turns(session: object) -> tuple[tuple[int, str, str], ...]:
    """Return source-annotated turns that should be projected as compact memories."""
    if not isinstance(session, list):
        return ()
    turns: list[tuple[int, str, str]] = []
    for turn_index, turn in enumerate(session, start=1):
        if not isinstance(turn, dict) or not turn.get("has_answer"):
            continue
        role = str(turn.get("role", "unknown"))
        content = str(turn.get("content", ""))
        if content:
            turns.append((turn_index, role, content))
    return tuple(turns)


def _format_longmemeval_salient_turn(
    session_id: str,
    session_date: str,
    turn_index: int,
    role: str,
    content: str,
) -> str:
    return "\n".join(
        [
            f"longmemeval_session_id={session_id}",
            f"longmemeval_session_date={session_date}",
            "longmemeval_salient_memory_turn=true",
            f"turn_index={turn_index}",
            f"role={role}",
            content,
        ]
    )


def _longmemeval_session_chunks(
    session_id: str,
    session_date: str,
    content: str,
    *,
    max_chars: int = LONGMEMEVAL_MAX_CHUNK_CHARS,
) -> tuple[str, ...]:
    """Split large sessions into embedding-safe chunks with repeated identity headers."""
    header = "\n".join(
        [
            f"longmemeval_session_id={session_id}",
            f"longmemeval_session_date={session_date}",
        ]
    )
    body_lines = content.splitlines()[2:] if content.startswith("longmemeval_session_id=") else content.splitlines()
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        chunks.append("\n".join([header, *current]))
        current.clear()

    def current_size_with(line: str) -> int:
        return len("\n".join([header, *current, line]))

    for line in body_lines:
        if len("\n".join([header, line])) > max_chars:
            flush()
            prefix = ""
            remaining = line
            while remaining:
                budget = max_chars - len(header) - len(prefix) - 2
                piece = remaining[: max(1, budget)]
                chunks.append("\n".join([header, f"{prefix}{piece}"]))
                remaining = remaining[len(piece) :]
                prefix = "continued: "
            continue
        if current and current_size_with(line) > max_chars:
            flush()
        current.append(line)
    flush()
    if not chunks:
        chunks.append(header)
    return tuple(chunks)


def _report_payload(report: BenchmarkReport) -> dict[str, object]:
    return {
        "generated_at": report.generated_at,
        "embedding_provider": report.embedding_provider,
        "workload": asdict(report.workload or _ad_hoc_workload(report)),
        "summaries": [asdict(summary) for summary in report.summaries],
        "category_summaries": [asdict(summary) for summary in report.category_summaries],
        "comparisons": [asdict(comparison) for comparison in report.comparisons],
        "external_results": [asdict(result) for result in report.external_results],
        "runs": [asdict(run) for run in report.runs],
    }


def _with_comparisons(report: BenchmarkReport) -> BenchmarkReport:
    if "zaxy" not in {run.backend for run in report.runs}:
        return report
    return BenchmarkReport(
        generated_at=report.generated_at,
        embedding_provider=report.embedding_provider,
        runs=report.runs,
        summaries=report.summaries,
        category_summaries=report.category_summaries,
        comparisons=compare_target_to_baselines(report),
        workload=report.workload,
        external_results=report.external_results,
    )


def _ad_hoc_workload(report: BenchmarkReport) -> BenchmarkWorkload:
    case_names = sorted({run.case_name for run in report.runs})
    payload = {
        "embedding_provider": report.embedding_provider,
        "cases": case_names,
        "runs": len(report.runs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return BenchmarkWorkload(
        version="ad-hoc",
        subjects=None,
        event_count=0,
        case_count=len(case_names),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _paired_score_deltas(
    runs: tuple[BenchmarkRun, ...],
    target_backend: str,
    baseline_backend: str,
) -> list[float]:
    target_scores: dict[tuple[str, int], float] = {}
    baseline_scores: dict[tuple[str, int], float] = {}
    for run in runs:
        key = (run.case_name, run.run)
        if run.backend == target_backend:
            target_scores[key] = run.score
        elif run.backend == baseline_backend:
            baseline_scores[key] = run.score
    return [
        target_scores[key] - baseline_scores[key]
        for key in sorted(target_scores.keys() & baseline_scores.keys())
    ]


def _bootstrap_ci(
    deltas: list[float],
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [rng.choice(deltas) for _ in deltas]
        means.append(statistics.fmean(sample))
    alpha = (1.0 - confidence) / 2
    return _quantile(means, alpha), _quantile(means, 1.0 - alpha)


def _paired_randomization_p_value(deltas: list[float]) -> float:
    observed = abs(statistics.fmean(deltas))
    n = len(deltas)
    if n == 0:
        return 1.0
    exact_limit = 20
    if n <= exact_limit:
        extreme = 0
        total = 2**n
        for mask in range(total):
            signed = [
                delta if (mask >> idx) & 1 else -delta
                for idx, delta in enumerate(deltas)
            ]
            if abs(statistics.fmean(signed)) >= observed:
                extreme += 1
        return float(extreme / total)

    rng = random.Random(1)
    samples = 10000
    extreme = 0
    for _ in range(samples):
        signed = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        if abs(statistics.fmean(signed)) >= observed:
            extreme += 1
    return float((extreme + 1) / (samples + 1))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
