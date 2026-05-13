"""Tests for production checkout evidence building."""

from __future__ import annotations

from zaxy.evidence import build_evidence_set, select_checkout_evidence
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


def test_checkout_evidence_selection_promotes_required_cited_source_groups() -> None:
    """Evidence-sensitive queries should promote cited source facts before uncited summaries."""
    plan = build_evidence_plan("How many weddings did I attend?", limit=10)
    uncited_summary = {
        "content": "A high-score uncited summary says there were weddings.",
        "source": "keyword",
        "score": 0.99,
        "citation": None,
        "source_lane": "graph",
    }
    first_source = {
        "content": "session_id=answer-1 I attended Rachel and Mike's wedding.",
        "source": "verbatim",
        "score": 0.9,
        "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
        "source_lane": "verbatim",
    }
    second_source = {
        "content": "session_id=answer-2 I attended Emily and Sarah's wedding.",
        "source": "verbatim",
        "score": 0.89,
        "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
        "source_lane": "verbatim",
    }

    selection = select_checkout_evidence(
        query="How many weddings did I attend?",
        evidence_plan=plan,
        current_facts=[uncited_summary, first_source, second_source],
        evidence=[first_source, second_source],
    )

    assert selection.current_facts[:2] == [first_source, second_source]
    assert selection.current_facts[2] == uncited_summary
    assert selection.evidence == [first_source, second_source]


def test_checkout_evidence_selection_preserves_direct_fact_order() -> None:
    """Direct fact checkout should not reorder facts when no source promotion is required."""
    plan = build_evidence_plan("What is the current task?", limit=10)
    first = {
        "content": "Current task is release hardening.",
        "source": "keyword",
        "score": 0.92,
        "citation": None,
        "source_lane": "graph",
    }
    second = {
        "content": "Older cited context.",
        "source": "keyword",
        "score": 0.8,
        "citation": "eventloom://agent-1/events/4#dddddddddddd",
        "source_lane": "graph",
    }

    selection = select_checkout_evidence(
        query="What is the current task?",
        evidence_plan=plan,
        current_facts=[first, second],
        evidence=[second],
    )

    assert selection.current_facts == [first, second]
    assert selection.evidence == [second]
