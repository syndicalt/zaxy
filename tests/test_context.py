"""Tests for source-aware context assembly policy."""

from __future__ import annotations

from zaxy.context import Context, ContextAssemblyPolicy


def test_context_assembly_policy_reserves_verbatim_slot_at_limit_one() -> None:
    """A one-item assembly budget should prefer exact source recall when present."""
    policy = ContextAssemblyPolicy(verbatim_slots=1)

    contexts = policy.assemble(
        [Context(content="Graph summary", source="keyword", score=0.9)],
        [Context(content="Exact source", source="verbatim", score=0.6)],
        limit=1,
    )

    assert [context.source for context in contexts] == ["verbatim"]
    assert contexts[0].metadata == {"assembly_lane": "verbatim"}


def test_context_assembly_policy_can_disable_verbatim_lane() -> None:
    """Operators should be able to turn source recall out of default assembly."""
    policy = ContextAssemblyPolicy(verbatim_enabled=False, verbatim_slots=1)

    contexts = policy.assemble(
        [Context(content="Graph summary", source="keyword", score=0.9)],
        [Context(content="Exact source", source="verbatim", score=0.6)],
        limit=2,
    )

    assert [context.source for context in contexts] == ["keyword"]
    assert contexts[0].metadata == {"assembly_lane": "graph"}


def test_context_assembly_policy_dedupes_graph_and_verbatim_by_citation() -> None:
    """The same Eventloom citation should not consume two assembly slots."""
    citation = "eventloom://agent/events/3#abc"
    policy = ContextAssemblyPolicy(verbatim_slots=1)

    contexts = policy.assemble(
        [
            Context(
                content="Graph summary",
                source="keyword",
                score=0.9,
                metadata={"citation": citation},
            )
        ],
        [
            Context(
                content="Exact source",
                source="verbatim",
                score=0.6,
                metadata={"citation": citation},
            )
        ],
        limit=2,
    )

    assert [context.source for context in contexts] == ["verbatim"]
    assert contexts[0].metadata == {
        "assembly_lane": "verbatim",
        "citation": citation,
    }


def test_context_assembly_policy_reserves_packet_memory_lane() -> None:
    """Recent packet memory should have a bounded proactive assembly lane."""
    policy = ContextAssemblyPolicy(packet_memory_slots=1)

    contexts = policy.assemble(
        [Context(content="Graph summary", source="keyword", score=0.9)],
        [],
        [Context(content="Packet memory", source="packet_memory", score=0.7)],
        limit=2,
    )

    assert [context.source for context in contexts] == ["keyword", "packet_memory"]
    assert contexts[1].metadata == {"assembly_lane": "packet_memory"}


def test_context_assembly_policy_expands_verbatim_lane_for_aggregation_queries() -> None:
    """Aggregation queries need multiple source observations, not one top hit."""
    policy = ContextAssemblyPolicy(verbatim_slots=1)

    contexts = policy.assemble(
        [
            Context(content=f"Graph summary {index}", source="keyword", score=0.9)
            for index in range(6)
        ],
        [
            Context(content=f"Exact source {index}", source="verbatim", score=0.8)
            for index in range(4)
        ],
        limit=8,
        query="How many properties did I visit before making an offer?",
    )

    assert [context.source for context in contexts].count("verbatim") == 4


def test_context_assembly_policy_overfetches_source_candidates_for_aggregation() -> None:
    """Source-sensitive queries should retrieve more candidates than final prompt slots."""
    policy = ContextAssemblyPolicy(verbatim_slots=1)

    assert (
        policy.verbatim_candidate_limit(
            query="How many properties did I visit before making an offer?",
            limit=8,
        )
        == 36
    )


def test_context_assembly_policy_uses_limit_for_direct_fact_source_candidates() -> None:
    """Direct fact queries should not pay source-lane overfetch cost."""
    policy = ContextAssemblyPolicy(verbatim_slots=1)

    assert policy.verbatim_candidate_limit(query="What is the current task?", limit=8) == 8


def test_context_assembly_policy_expands_verbatim_lane_for_absence_queries() -> None:
    """Absence checks need nearby mentioned alternatives from source evidence."""
    policy = ContextAssemblyPolicy(verbatim_slots=1)

    contexts = policy.assemble(
        [
            Context(content=f"Graph summary {index}", source="keyword", score=0.9)
            for index in range(6)
        ],
        [
            Context(content=f"Exact source {index}", source="verbatim", score=0.8)
            for index in range(3)
        ],
        limit=6,
        query="Did I mention my hamster?",
    )

    assert [context.source for context in contexts].count("verbatim") == 3
