"""Tests for source-aware context assembly policy."""

from __future__ import annotations

from zaxy.context import (
    TIER_SESSION,
    TIER_VOLATILE,
    Context,
    ContextAssemblyPolicy,
    PromptSection,
    apply_assembly_prompt_budget,
    order_prompt_sections,
    render_prompt_sections,
    split_assembly_prompt,
    stable_prefix_chars,
)

_ASSEMBLY_PROMPT = "\n".join(
    [
        "# Active Memory Working Set",
        "- decision: Use MMR diversity (eventloom://agent-1/events/2#abc)",
        "",
        "# Recent Events",
        "[1] transcript.turn by user",
        "What retrieval policy did we pick?",
        "",
        "# Retrieved Context",
        "- Use MMR diversity (eventloom://agent-1/events/2#abc)",
        "",
        "# Context Warnings",
        "- Recent replay was compacted to fit the prompt budget.",
    ]
)


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
        == 72
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


def test_split_assembly_prompt_tags_sections_with_stability_tiers() -> None:
    """Assembly prompt sections should carry kinds and stability tiers."""
    sections = split_assembly_prompt(_ASSEMBLY_PROMPT)

    assert [(section.kind, section.tier) for section in sections] == [
        ("working_set", TIER_SESSION),
        ("recent_events", TIER_SESSION),
        ("retrieved_context", TIER_VOLATILE),
        ("context_warnings", TIER_VOLATILE),
    ]
    assert sections[3].mandatory is True


def test_split_assembly_prompt_round_trips_through_render() -> None:
    """Splitting then rendering must reproduce the canonical prompt byte for byte."""
    sections = split_assembly_prompt(_ASSEMBLY_PROMPT)

    assert render_prompt_sections(sections) == _ASSEMBLY_PROMPT


def test_order_prompt_sections_orders_tiers_and_is_stable_within_tier() -> None:
    """Sections render consolidated, then session, then volatile, stably."""
    sections = [
        PromptSection(section_id="b", kind="b", tier=TIER_VOLATILE, text="b"),
        PromptSection(section_id="c", kind="c", tier=TIER_SESSION, text="c"),
        PromptSection(section_id="a", kind="a", tier="consolidated", text="a"),
        PromptSection(section_id="d", kind="d", tier=TIER_SESSION, text="d"),
    ]

    assert [section.section_id for section in order_prompt_sections(sections)] == [
        "a",
        "c",
        "d",
        "b",
    ]


def test_assembly_prompt_has_no_consolidated_stable_prefix() -> None:
    """Assembled context has session-first content, so its stable prefix is empty."""
    assert stable_prefix_chars(split_assembly_prompt(_ASSEMBLY_PROMPT)) == 0


def test_apply_assembly_prompt_budget_large_budget_changes_nothing() -> None:
    """A generous budget keeps the prompt byte-identical and reports no elisions."""
    prompt, budget = apply_assembly_prompt_budget(_ASSEMBLY_PROMPT, max_tokens=100_000)

    assert prompt == _ASSEMBLY_PROMPT
    assert budget["budget_requested"] == 100_000
    assert budget["budget_used"] > 0
    assert budget["elided"] == {"count": 0, "kinds": [], "sections": []}


def test_apply_assembly_prompt_budget_zero_budget_keeps_mandatory_warnings() -> None:
    """A zero budget elides optional sections but never the warning contract."""
    prompt, budget = apply_assembly_prompt_budget(_ASSEMBLY_PROMPT, max_tokens=0)

    assert prompt == "# Context Warnings\n- Recent replay was compacted to fit the prompt budget."
    assert budget["elided"]["count"] == 3
    assert budget["elided"]["kinds"] == ["recent_events", "retrieved_context", "working_set"]
    for record in budget["elided"]["sections"]:
        assert record["estimated_tokens"] > 0


def test_apply_assembly_prompt_budget_is_monotone_in_budget() -> None:
    """A larger assembly budget never drops a previously included section."""
    previous_kinds: set[str] = set()
    all_kinds = {section.kind for section in split_assembly_prompt(_ASSEMBLY_PROMPT)}
    for budget_tokens in range(0, 200, 5):
        prompt, budget = apply_assembly_prompt_budget(_ASSEMBLY_PROMPT, max_tokens=budget_tokens)
        elided_kinds = set(budget["elided"]["kinds"])
        included_kinds = all_kinds - elided_kinds
        assert previous_kinds <= included_kinds
        previous_kinds = included_kinds
