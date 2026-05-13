"""Tests for production checkout evidence building."""

from __future__ import annotations

from zaxy.evidence import build_evidence_set
from zaxy.retrieval_plan import build_evidence_plan


def test_evidence_set_groups_cited_sources_and_reports_sufficiency() -> None:
    """Evidence builder should group cited memories by source identity."""
    plan = build_evidence_plan("How many weddings did I attend?", limit=10)
    evidence = [
        {
            "content": "longmemeval_session_id=answer-1 I attended Rachel and Mike's wedding.",
            "source": "verbatim",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "source_lane": "verbatim",
        },
        {
            "content": "longmemeval_session_id=answer-1 Reception details mentioned dancing.",
            "source": "verbatim",
            "score": 0.83,
            "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
            "source_lane": "verbatim",
        },
        {
            "content": "longmemeval_session_id=answer-2 I attended Emily and Sarah's wedding.",
            "source": "graph",
            "score": 0.89,
            "citation": "eventloom://agent-1/events/3#cccccccccccc",
            "source_lane": "graph",
        },
    ]

    evidence_set = build_evidence_set(
        query="How many weddings did I attend?",
        evidence_plan=plan,
        current_facts=evidence,
        evidence=evidence,
    )

    assert evidence_set.to_diagnostics() == {
        "groups": [
            {
                "source_id": "answer-1",
                "evidence_count": 2,
                "citation_count": 2,
                "citations": [
                    "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                    "eventloom://agent-1/events/2#bbbbbbbbbbbb",
                ],
                "source_lanes": ["verbatim"],
                "top_score": 0.91,
                "snippet": "longmemeval_session_id=answer-1 I attended Rachel and Mike's wedding.",
            },
            {
                "source_id": "answer-2",
                "evidence_count": 1,
                "citation_count": 1,
                "citations": ["eventloom://agent-1/events/3#cccccccccccc"],
                "source_lanes": ["graph"],
                "top_score": 0.89,
                "snippet": "longmemeval_session_id=answer-2 I attended Emily and Sarah's wedding.",
            },
        ],
        "status": {
            "required_source_groups": 2,
            "observed_source_groups": 2,
            "satisfied": True,
        },
    }


def test_evidence_set_reports_missing_required_source_groups() -> None:
    """Evidence builder should expose a refresh query when evidence is incomplete."""
    plan = build_evidence_plan("How many weddings did I attend?", limit=10)

    evidence_set = build_evidence_set(
        query="How many weddings did I attend?",
        evidence_plan=plan,
        current_facts=[
            {
                "content": "longmemeval_session_id=answer-1 I attended Rachel and Mike's wedding.",
                "source": "verbatim",
                "score": 0.91,
                "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                "source_lane": "verbatim",
            }
        ],
        evidence=[],
    )

    assert evidence_set.status == {
        "required_source_groups": 2,
        "observed_source_groups": 1,
        "satisfied": False,
        "refresh_query": "broader cited evidence for: How many weddings did I attend?",
    }
