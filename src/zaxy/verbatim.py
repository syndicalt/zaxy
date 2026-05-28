"""Verbatim retrieval over raw Eventloom-backed memory.

This module is the source-recall lane for "Git for agent memory": it indexes
raw document, transcript, and event payload text without replacing Eventloom as
the immutable source of truth.
"""

from __future__ import annotations

import heapq
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from zaxy.event import Event, EventLog
from zaxy.security import validate_limit, validate_query


@dataclass(frozen=True)
class VerbatimChunk:
    """One raw retrievable source chunk."""

    chunk_id: str
    content: str
    citation: str
    source_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerbatimHit:
    """A ranked verbatim retrieval hit."""

    content: str
    score: float
    citation: str
    source_kind: str
    metadata: dict[str, Any]


class VerbatimIndex:
    """In-memory BM25 index over raw Eventloom source chunks."""

    def __init__(self, chunks: tuple[VerbatimChunk, ...]) -> None:
        self._chunks = chunks
        self._tokenized = tuple(tuple(_tokens(chunk.content)) for chunk in chunks)
        self._term_counts = tuple(Counter(tokens) for tokens in self._tokenized)
        self._term_document_ids = _term_document_ids(self._term_counts)
        self._document_lengths = tuple(len(tokens) for tokens in self._tokenized)
        self._document_frequencies = _document_frequencies(self._tokenized)
        self._document_count = len(self._tokenized)
        self._average_document_length = (
            statistics.fmean(self._document_lengths)
            if self._tokenized
            else 0.0
        )
        self._term_idf = _term_idf(self._document_frequencies, self._document_count)
        self._document_length_norms = _document_length_norms(
            self._document_lengths,
            self._average_document_length,
        )

    @classmethod
    def from_event_logs(cls, eventlogs: list[EventLog] | tuple[EventLog, ...]) -> VerbatimIndex:
        """Build an index from one or more Eventloom logs."""
        chunks: list[VerbatimChunk] = []
        for eventlog in eventlogs:
            for event in eventlog.read_all():
                chunk = _chunk_from_event(event)
                if chunk is not None:
                    chunks.append(chunk)
        return cls(tuple(chunks))

    def query(self, query: str, *, limit: int = 10) -> list[VerbatimHit]:
        """Return exact source chunks ranked by lexical relevance."""
        validate_query(query)
        lim = validate_limit(limit)
        query_terms = tuple(dict.fromkeys(_tokens(query)))
        if not query_terms:
            return []
        scored = []
        candidate_terms: dict[int, list[str]] = {}
        for term in query_terms:
            for index in self._term_document_ids.get(term, ()):
                candidate_terms.setdefault(index, []).append(term)
        if not candidate_terms:
            return []
        for index, matched_terms in candidate_terms.items():
            chunk = self._chunks[index]
            term_counts = self._term_counts[index]
            score = _bm25_score_from_precomputed(
                tuple(matched_terms),
                term_counts,
                self._document_length_norms[index],
                self._term_idf,
            )
            if score <= 0.0:
                continue
            scored.append((score, chunk))
        return [
            VerbatimHit(
                content=chunk.content,
                score=round(score, 4),
                citation=chunk.citation,
                source_kind=chunk.source_kind,
                metadata=dict(chunk.metadata),
            )
            for score, chunk in heapq.nlargest(lim, scored, key=lambda item: item[0])
        ]


def _chunk_from_event(event: Event) -> VerbatimChunk | None:
    if event.type == "document.indexed":
        content = _text(event.payload.get("content"))
        if not content:
            return None
        source_path = _text(event.payload.get("path")) or "document"
        start_line = _int(event.payload.get("start_line"))
        end_line = _int(event.payload.get("end_line"))
        metadata = {
            "event_seq": event.seq,
            "event_type": event.type,
            "event_thread": event.thread,
            "event_timestamp": event.timestamp,
            "source_path": source_path,
            "source_start_line": start_line,
            "source_end_line": end_line,
            "source_sha256": _text(event.payload.get("sha256")),
        }
        return VerbatimChunk(
            chunk_id=f"{event.thread}:{event.seq}",
            content=content,
            citation=_event_citation(event),
            source_kind="document",
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    if event.type == "transcript.turn":
        content = _text(event.payload.get("content"))
        if not content:
            return None
        role = _text(event.payload.get("role")) or event.actor
        metadata = {
            "event_seq": event.seq,
            "event_type": event.type,
            "event_thread": event.thread,
            "event_timestamp": event.timestamp,
            "transcript_source": _text(event.payload.get("source")),
            "transcript_turn_index": _int(event.payload.get("turn_index")),
            "transcript_role": role,
        }
        return VerbatimChunk(
            chunk_id=f"{event.thread}:{event.seq}",
            content=f"{role}: {content}",
            citation=_event_citation(event),
            source_kind="transcript",
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    if event.type == "llm.packet.projected":
        content = _text(event.payload.get("summary"))
        if not content:
            return None
        metadata = {
            "event_seq": event.seq,
            "event_type": event.type,
            "event_thread": event.thread,
            "event_timestamp": event.timestamp,
            "session_id": _text(event.payload.get("session_id")),
            "source_event_seq": _int(event.payload.get("source_event_seq")),
            "source_event_hash": _text(event.payload.get("source_event_hash")),
            "provider_path": _text(event.payload.get("provider_path")),
            "status_code": _int(event.payload.get("status_code")),
            "model": _text(event.payload.get("model")),
        }
        return VerbatimChunk(
            chunk_id=f"{event.thread}:{event.seq}",
            content=content,
            citation=_event_citation(event),
            source_kind="packet_projection",
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    content = _payload_text(event.payload)
    if not content:
        return None
    return VerbatimChunk(
        chunk_id=f"{event.thread}:{event.seq}",
        content=content,
        citation=_event_citation(event),
        source_kind="event",
        metadata={
            "event_seq": event.seq,
            "event_type": event.type,
            "event_thread": event.thread,
            "event_timestamp": event.timestamp,
        },
    )


def _event_citation(event: Event) -> str:
    return f"eventloom://{event.thread}/events/{event.seq}#{event.hash}"


def _payload_text(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _tokens(text: str) -> tuple[str, ...]:
    tokens = []
    for token in re.findall(r"[a-zA-Z0-9_$./:-]+", text):
        normalized = token.strip(".,;!?()[]{}\"'`").casefold()
        if normalized:
            tokens.append(normalized)
    return tuple(tokens)


def _document_frequencies(documents: tuple[tuple[str, ...], ...]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(document))
    return frequencies


def _term_document_ids(term_counts: tuple[Counter[str], ...]) -> dict[str, tuple[int, ...]]:
    postings: defaultdict[str, list[int]] = defaultdict(list)
    for index, counts in enumerate(term_counts):
        for term in counts:
            postings[term].append(index)
    return {term: tuple(indices) for term, indices in postings.items()}


def _term_idf(document_frequencies: Counter[str], document_count: int) -> dict[str, float]:
    if document_count == 0:
        return {}
    return {
        term: math.log(1 + ((document_count - frequency + 0.5) / (frequency + 0.5)))
        for term, frequency in document_frequencies.items()
    }


def _document_length_norms(
    document_lengths: tuple[int, ...],
    average_document_length: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[float, ...]:
    safe_average = max(average_document_length, 1.0)
    return tuple(k1 * (1 - b + b * (document_length / safe_average)) for document_length in document_lengths)


def _bm25_score_from_precomputed(
    query_terms: tuple[str, ...],
    term_counts: Counter[str],
    document_length_norm: float,
    term_idf: dict[str, float],
    *,
    k1: float = 1.5,
) -> float:
    score = 0.0
    for term in query_terms:
        frequency = term_counts.get(term, 0)
        if frequency == 0:
            continue
        denominator = frequency + document_length_norm
        score += term_idf.get(term, 0.0) * ((frequency * (k1 + 1)) / denominator)
    return score


def _bm25_score(
    query_terms: tuple[str, ...],
    document_terms: tuple[str, ...],
    document_frequencies: Counter[str],
    document_count: int,
    average_document_length: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if document_count == 0 or not document_terms:
        return 0.0
    term_counts = Counter(document_terms)
    document_length = len(document_terms)
    return _bm25_score_from_counts(
        query_terms,
        term_counts,
        document_length,
        document_frequencies,
        document_count,
        average_document_length,
        k1=k1,
        b=b,
    )


def _bm25_score_from_counts(
    query_terms: tuple[str, ...],
    term_counts: Counter[str],
    document_length: int,
    document_frequencies: Counter[str],
    document_count: int,
    average_document_length: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if document_count == 0 or document_length <= 0:
        return 0.0
    score = 0.0
    for term in query_terms:
        frequency = term_counts.get(term, 0)
        if frequency == 0:
            continue
        document_frequency = document_frequencies.get(term, 0)
        inverse_document_frequency = math.log(
            1 + ((document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        )
        denominator = frequency + k1 * (
            1 - b + b * (document_length / max(average_document_length, 1.0))
        )
        score += inverse_document_frequency * ((frequency * (k1 + 1)) / denominator)
    return score
