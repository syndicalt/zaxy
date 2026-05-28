"""Deterministic retrieval intent classification for context assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalIntent:
    """Query-level retrieval needs that affect source-lane budgeting."""

    needs_source_lane: bool
    source_lane_slots: int
    reasons: tuple[str, ...] = ()


def classify_retrieval_intent(query: str, *, limit: int) -> RetrievalIntent:
    """Classify a memory query without model calls or benchmark-specific terms."""
    tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
    reasons: list[str] = []

    source_terms = {
        "citation",
        "cited",
        "document",
        "file",
        "provenance",
        "source",
        "sources",
    }
    personal_terms = {"i", "me", "my", "mine"}
    memory_terms = {
        "conversation",
        "discuss",
        "discussed",
        "hear",
        "heard",
        "mention",
        "mentioned",
        "memory",
        "name",
        "previous",
        "remember",
        "said",
        "say",
        "tell",
        "told",
    }
    question_terms = {
        "did",
        "do",
        "does",
        "how",
        "name",
        "what",
        "when",
        "where",
        "which",
        "who",
    }
    aggregation_terms = {
        "all",
        "amount",
        "attended",
        "bought",
        "count",
        "days",
        "different",
        "each",
        "hours",
        "list",
        "long",
        "many",
        "much",
        "paid",
        "spent",
        "sum",
        "total",
        "visited",
        "weeks",
    }
    absence_terms = {
        "did",
        "mention",
        "mentioned",
        "not",
        "never",
        "whether",
    }
    operational_memory_terms = {
        "checkpoint",
        "decision",
        "decisions",
        "handoff",
        "session",
        "turn",
    }
    operational_query_terms = {
        "current",
        "recover",
        "retrieve",
        "what",
        "which",
    }
    temporal_order_terms = {
        "before",
        "earlier",
        "event",
        "first",
        "happened",
        "which",
    }

    needs_source = False
    slots = 0
    if tokens & source_terms:
        needs_source = True
        slots = max(slots, 2)
        reasons.append("source_recall")
    if tokens & personal_terms and tokens & (question_terms | memory_terms):
        needs_source = True
        slots = max(slots, max(2, min(3, limit // 3)))
        reasons.append("personal_memory")
    aggregation_slots = max(4, min(8, (limit * 4) // 5))
    if tokens & personal_terms and tokens & aggregation_terms:
        needs_source = True
        slots = max(slots, aggregation_slots)
        reasons.append("aggregation")
    if {"how", "many"} <= tokens or {"how", "much"} <= tokens or {"how", "long"} <= tokens:
        needs_source = True
        slots = max(slots, aggregation_slots)
        reasons.append("aggregation_question")
    if tokens & personal_terms and len(tokens & absence_terms) >= 2:
        needs_source = True
        slots = max(slots, max(2, min(4, limit // 2)))
        reasons.append("absence_check")
    if tokens & operational_memory_terms and tokens & operational_query_terms:
        needs_source = True
        slots = max(slots, 1)
        reasons.append("operational_memory")
    if (
        {"first", "which"} <= tokens and tokens & temporal_order_terms
    ) or (
        tokens & {"meet", "met"} and tokens & {"first", "earlier", "before"}
    ):
        needs_source = True
        slots = max(slots, max(2, min(4, limit // 2)))
        reasons.append("temporal_order")

    if limit <= 0 or not needs_source:
        return RetrievalIntent(False, 0, ())
    return RetrievalIntent(True, min(limit, max(1, slots)), tuple(dict.fromkeys(reasons)))
