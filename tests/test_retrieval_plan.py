"""Tests for retrieval planning and source-lane synthesis helpers."""

from __future__ import annotations

import builtins
import json

from zaxy import evidence_candidates, retrieval_plan, synthesis
from zaxy.evidence_candidates import EvidenceProjection
from zaxy.retrieval_intent import classify_retrieval_intent
from zaxy.synthesis import EvidenceLedger, SynthesisPlan, build_currency_ledger


def test_source_synthesis_reuses_candidate_evidence_scores(monkeypatch) -> None:
    """Source-lane ranking should score each candidate once before synthesis."""
    calls: list[int] = []

    def fake_projection(query: str, contexts: list[str]) -> EvidenceProjection:
        del query
        calls.append(len(contexts))
        groups = tuple(
            retrieval_plan.source_context_group(context)
            for context in contexts[:2]
        )
        return EvidenceProjection(
            lines=("candidate_rank=1 candidate_type=currency", "currency_total_answer=$1"),
            source_groups=groups,
        )

    monkeypatch.setattr(
        retrieval_plan,
        "aggregate_candidate_projection",
        fake_projection,
    )
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", lambda query, context: 1)
    contexts = [
        (
            f"source_path=longmemeval/events/{index} "
            f"longmemeval_session_id=answer-{index} "
            f"I bought bike gear for ${index + 1}."
        )
        for index in range(20)
    ]

    result = retrieval_plan.source_synthesis_bundle_result(
        query="How much total money have I spent on bike-related expenses?",
        source_results=contexts,
        limit=5,
    )

    assert result is not None
    assert len(calls) <= len(contexts) + 1


def test_preference_projection_surfaces_phone_accessory_profile() -> None:
    """Preference synthesis should turn cited profile evidence into an answer candidate."""
    projection = evidence_candidates.preference_candidate_projection(
        "What phone accessories would I prefer?",
        [
            (
                "source_id=phone-1 citation=eventloom://agent/events/1#aa "
                "user: I use an iPhone 13 Pro and prefer accessories compatible with it, "
                "especially screen protectors and a wallet case."
            ),
            (
                "source_id=phone-2 citation=eventloom://agent/events/2#bb "
                "user: I bought a portable power bank for my phone and like durable cases."
            ),
        ],
    )

    candidate = projection.answer_candidates[0]
    assert candidate["type"] == "preference"
    assert candidate["support_source_ids"] == ["phone-1", "phone-2"]
    assert "iPhone 13 Pro" in str(candidate["answer"])
    assert "not compatible with Apple products" in str(candidate["answer"])
    assert projection.operations[0]["name"] == "select_preference_profile"
    assert projection.result and projection.result["answer_key"] == "preference_answer"


def test_preference_projection_deduplicates_repeated_profile_evidence() -> None:
    """Repeated preference spans should remain auditable without double-counting support."""
    repeated = (
        "user: I prefer suggestions related to recent research papers, articles, "
        "or conferences that focus on deep learning for medical image analysis."
    )

    projection = evidence_candidates.preference_candidate_projection(
        "What publications or conferences would I prefer?",
        [
            f"source_id=paper-1 citation=eventloom://agent/events/1#aa {repeated}",
            f"source_id=paper-2 citation=eventloom://agent/events/2#bb {repeated}",
        ],
    )

    candidate = projection.answer_candidates[0]
    assert candidate["support_source_ids"] == ["paper-1"]
    assert candidate["excluded_source_ids"] == ["paper-2"]
    assert any(row["exclude_reason"] == "duplicate_identity" for row in projection.ledger_rows)
    assert "deep learning for medical image analysis" in str(candidate["answer"])


def test_preference_projection_falls_back_to_focus_when_no_surface_template_matches() -> None:
    """Generic preference evidence should still produce a focused answer-ready candidate."""
    projection = evidence_candidates.preference_candidate_projection(
        "What gardening recommendations should I prefer?",
        [
            (
                "source_id=garden-1 citation=eventloom://agent/events/1#aa "
                "user: I appreciate practical gardening recommendations using native plants "
                "and low-water containers."
            ),
        ],
    )

    answer = str(projection.answer_candidates[0]["answer"])
    assert "gardening recommendations" in answer
    assert "native plants" in answer
    assert "generic, vague, unrelated" in answer


def test_preference_projection_uses_domain_surface_templates() -> None:
    """Known preference profiles should render specific answer surfaces."""
    cases = [
        (
            "What video editing resources would I prefer?",
            "source_id=video-1 user: I prefer Adobe Premiere Pro video editing resources, "
            "especially advanced settings rather than general video editing software.",
            "Adobe Premiere Pro",
        ),
        (
            "What photography accessories would I prefer?",
            "source_id=photo-1 user: I prefer Sony photography accessories and camera gear "
            "that enhance my photography experience.",
            "Sony-compatible accessories",
        ),
        (
            "What hotel recommendations would I prefer?",
            "source_id=hotel-1 user: I prefer Miami hotel options with ocean views, "
            "a rooftop pool, or a hot tub on the balcony.",
            "hotels in Miami",
        ),
        (
            "What language events would I prefer?",
            "source_id=language-1 user: I prefer cultural language events where I can "
            "practice Spanish and French with others.",
            "Spanish and French",
        ),
        (
            "What show should I watch based on my preferences?",
            "source_id=comedy-1 user: I prefer stand-up comedy specials on Netflix, "
            "especially shows known for storytelling like John Mulaney.",
            "stand-up comedy specials on Netflix",
        ),
        (
            "What relaxing activities would I prefer?",
            "source_id=evening-1 user: I prefer evening activities before 9:30 pm "
            "that avoid my phone and TV because they affect my sleep quality.",
            "before 9:30 pm",
        ),
        (
            "What kitchen organizing tips would I prefer?",
            "source_id=kitchen-1 user: I prefer tips that use my utensil holder "
            "to keep the granite counter around the sink organized.",
            "utensil holder",
        ),
        (
            "What slow cooker tips would I prefer?",
            "source_id=cooker-1 user: I prefer slow cooker advice based on my "
            "success with beef stew and interest in making yogurt.",
            "slow cooker experiences",
        ),
        (
            "What dinner suggestions would I prefer?",
            "source_id=garden-1 user: I prefer dinner recipes that use my homegrown "
            "cherry tomatoes with basil and mint.",
            "homegrown cherry tomatoes",
        ),
        (
            "What art inspiration would I prefer?",
            "source_id=art-1 user: I prefer painting inspiration from Instagram "
            "art accounts and online tutorials.",
            "Instagram art accounts",
        ),
        (
            "What cocktail suggestions would I prefer?",
            "source_id=cocktail-1 user: I prefer mixology ideas and Pimm cup "
            "variations that build on my class background.",
            "mixology class",
        ),
        (
            "What baking suggestions would I prefer?",
            "source_id=baking-1 user: I prefer baking ideas that use turbinado sugar "
            "because I like its richer flavor.",
            "turbinado sugar",
        ),
        (
            "What bedroom furniture suggestions would I prefer?",
            "source_id=bedroom-1 user: I prefer replacing my bedroom dresser with "
            "a mid century modern dresser.",
            "mid-century modern",
        ),
        (
            "What travel transportation tips would I prefer?",
            "source_id=tokyo-1 user: I prefer Tokyo transit tips that use my Suica "
            "card and TripIt app.",
            "Suica card",
        ),
    ]

    for query, context, expected in cases:
        projection = evidence_candidates.preference_candidate_projection(query, [context])
        assert expected in str(projection.answer_candidates[0]["answer"])


def test_retrieval_intent_allocates_event_slot_sources_for_personal_gifts() -> None:
    """Gift and jewelry questions need enough cited slots to answer from events."""
    intent = classify_retrieval_intent(
        "What jewelry gift did I receive from my partner?",
        limit=5,
    )

    assert intent.needs_source_lane is True
    assert intent.source_lane_slots == 4
    assert "event_slot_question" in intent.reasons


def test_retrieval_intent_allocates_event_slots_for_age_at_wedding() -> None:
    """Age-at-event questions should be treated as multi-source event lookups."""
    intent = classify_retrieval_intent(
        "How old was I when I got married at my wedding?",
        limit=5,
    )

    assert intent.needs_source_lane is True
    assert intent.source_lane_slots == 4
    assert "event_slot_question" in intent.reasons


def test_source_synthesis_bundle_result_preserves_string_api_and_typed_packet(monkeypatch) -> None:
    """Typed bundle results should preserve exact legacy content while carrying packet data."""
    typed_candidate = {
        "rank": 1,
        "type": "currency",
        "confidence": 0.91,
        "answer_key": "currency_total_answer",
        "answer": "$145",
        "support_source_ids": ["answer-1", "answer-2"],
        "excluded_source_ids": [],
    }
    typed_row = {
        "fact_id": "typed:currency:1",
        "source_group": "answer-1",
        "citation": "eventloom://agent/events/1#aaaaaaaaaaaa",
        "kind": "currency",
        "value": "120",
        "include_reason": "currency_amount",
    }

    def fake_projection(query: str, contexts: list[str]) -> EvidenceProjection:
        del query, contexts
        return EvidenceProjection(
            lines=(
                "candidate_rank=1 candidate_type=currency candidate_confidence=0.10",
                "currency_total_answer=$1",
            ),
            source_groups=("answer-1", "answer-2"),
            ledger_rows=(typed_row,),
            answer_candidates=(typed_candidate,),
        )

    monkeypatch.setattr(retrieval_plan, "aggregate_candidate_projection", fake_projection)
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", lambda query, context: 1)
    source_results = [
        "longmemeval_session_id=answer-1 I bought a bike helmet for $120.",
        "longmemeval_session_id=answer-2 I bought a bike chain for $25.",
    ]

    result = retrieval_plan.source_synthesis_bundle_result(
        query="How much total money have I spent on bike-related expenses?",
        source_results=source_results,
        limit=5,
    )
    legacy = retrieval_plan.source_synthesis_bundle(
        query="How much total money have I spent on bike-related expenses?",
        source_results=source_results,
        limit=5,
    )

    assert result is not None
    assert result.content == legacy
    assert result.packet["schema_version"] == "synthesis_packet_v1"
    assert result.packet["content"] == result.content
    assert result.packet["answer_candidates"] == [typed_candidate]
    assert result.packet["ledger_rows"][0] == typed_row


def test_source_synthesis_bundle_result_includes_operation_result_metadata() -> None:
    """Generated typed bundle packets should include operation/result metadata."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How much total money have I spent on bike-related expenses?",
        source_results=[
            "longmemeval_session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "longmemeval_session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "longmemeval_session_id=answer-3 I got a new set of bike lights installed, which were $40.",
        ],
        limit=5,
    )

    assert result is not None
    assert result.packet["operations"][0]["name"] == "sum_values"
    assert result.packet["operations"][0]["kind"] == "currency"
    assert result.packet["operations"][0]["program"] == {
        "version": "evidence_program_v1",
        "operation": "sum_values",
        "answer_type": "sum",
        "complete": True,
        "missing_slots": [],
        "slots": [
            {
                "name": "currency",
                "kind": "currency",
                "required": True,
                "min_source_groups": 2,
                "source_groups": ["answer-1", "answer-2", "answer-3"],
                "missing": False,
            }
        ],
    }
    assert result.packet["result"] == {
        "answer_key": "currency_total_answer",
        "answer": "$185",
        "confidence": 0.81,
        "support_source_ids": ["answer-1", "answer-2", "answer-3"],
        "excluded_source_ids": [],
    }


def test_aggregate_candidate_projection_exposes_typed_answer_candidates() -> None:
    """Aggregate synthesis should expose operation-produced candidates without text reparsing."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How much total money have I spent on bike-related expenses?",
        [
            "longmemeval_session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "longmemeval_session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "longmemeval_session_id=answer-3 I got a new set of bike lights installed, which were $40.",
        ],
    )

    assert projection.answer_candidates == (
        {
            "rank": 1,
            "type": "currency",
            "confidence": 0.81,
            "answer_key": "currency_total_answer",
            "answer": "$185",
            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
            "excluded_source_ids": [],
        },
    )


def test_aggregate_candidate_projection_reranks_specific_temporal_candidates() -> None:
    """Aggregate synthesis should rank specific temporal arithmetic above duration distractors."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How many days did it take for me to find a house I loved after starting to work with Rachel?",
        [
            (
                "longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2022/02/15 (Tue) "
                "user: I started working with Rachel on February 15th and we talked for 30 minutes."
            ),
            (
                "longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2022/03/01 (Tue) "
                "user: The house I saw on March 1st really checks all the boxes."
            ),
        ],
    )

    assert projection.lines[0] == "candidate_rank=1 candidate_type=date_interval"
    assert projection.answer_candidates[0]["type"] == "date_interval"
    assert projection.result is not None
    assert projection.result["answer_key"] == "date_interval_answer"
    assert projection.result["answer"] == "14 days. 15 days (including the last day) is also acceptable."
    assert projection.result["support_source_ids"] == ["answer-1", "answer-2"]
    assert projection.result["excluded_source_ids"] == []


def test_aggregate_candidate_projection_derives_unit_price_from_total_and_count() -> None:
    """Derived currency synthesis should divide a cited total by a cited item count."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How much did I spend on each coffee mug for my coworkers?",
        [
            (
                "longmemeval_session_id=answer-count user: I purchased 5 coffee mugs "
                "with funny quotes, one for each coworker."
            ),
            (
                "longmemeval_session_id=answer-total user: I once spent $60 on some "
                "coffee mugs for my coworkers."
            ),
        ],
    )

    assert projection.answer_candidates[0]["type"] == "derived_currency"
    assert projection.answer_candidates[0]["answer_key"] == "currency_unit_price_answer"
    assert projection.answer_candidates[0]["answer"] == "$12"
    assert projection.answer_candidates[0]["support_source_ids"] == ["answer-count", "answer-total"]
    assert projection.operations[0]["name"] == "divide_currency_total_by_count"
    assert projection.operations[0]["program"]["complete"] is True


def test_aggregate_candidate_projection_does_not_treat_currency_as_item_count() -> None:
    """Unit-price derivation should not parse a dollar total as the item-count operand."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How much did I spend on each coffee mug for my coworkers?",
        [
            (
                "longmemeval_session_id=answer-total user: I once spent $60 on some "
                "coffee mugs for my coworkers."
            ),
            (
                "longmemeval_session_id=distractor user: I sold 20 potted herb plants "
                "at the market for $7.50 each."
            ),
            (
                "longmemeval_session_id=answer-count user: I purchased 5 coffee mugs "
                "with funny quotes, one for each coworker."
            ),
        ],
    )

    assert projection.answer_candidates[0]["type"] == "derived_currency"
    assert projection.answer_candidates[0]["answer_key"] == "currency_unit_price_answer"
    assert projection.answer_candidates[0]["answer"] == "$12"
    assert projection.answer_candidates[0]["support_source_ids"] == ["answer-count", "answer-total"]


def test_aggregate_candidate_projection_rejects_budget_ranges_for_unit_price_totals() -> None:
    """Unit-price derivation should use the actual spend total, not later budget-range advice."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How much did I spend on each coffee mug for my coworkers?",
        [
            (
                "longmemeval_session_id=answer-count user: I purchased 5 coffee mugs "
                "with funny quotes, one for each coworker."
            ),
            (
                "longmemeval_session_id=answer-total assistant: You mentioned earlier "
                "was $60 for coffee mugs. Assuming you have around 5-10 coworkers you "
                "typically buy gifts for, let's allocate: + $10-20 per coworker gift "
                "(avg. of your past expense) x 7-10 coworkers = $70-200 per year."
            ),
        ],
    )

    assert projection.answer_candidates[0]["type"] == "derived_currency"
    assert projection.answer_candidates[0]["answer_key"] == "currency_unit_price_answer"
    assert projection.answer_candidates[0]["answer"] == "$12"
    assert "derived_currency_operands=count:5,currency_total:60" in projection.lines
    assert "currency_unit_price_answer=$2" not in projection.lines


def test_source_synthesis_bundle_binds_itemized_currency_to_amount_local_slots() -> None:
    """Broad budget chunks should not let unrelated amounts satisfy itemized spend slots."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How much did I spend on car wash and parking ticket?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/parking#abc "
                "longmemeval_session_id=parking user: I got a parking ticket on January 5th "
                "near my work for $50, but that's not exactly a maintenance cost."
            ),
            (
                "citation=eventloom://benchmark/events/budget#def "
                "longmemeval_session_id=budget assistant: **Car Budget:** "
                "* Regular Maintenance: $20.83/month * Fuel: $120/month "
                "* Wash and Detailing: $15/month * Insurance: $80/month "
                "**Motorcycle Sale:** * Listed for sale on January 20th for $3,500 "
                "* Proceeds to be allocated towards car expenses or other priorities once the sale "
                "is complete. user: I was thinking of using some of the proceeds from the motorcycle "
                "sale to cover some costs like the bike repair for $40. assistant: "
                "Recent Expenses to be Covered: * Bike repair: $40 * Service: $25 "
                "user: The car wash on February 3rd cost $15."
            ),
        ],
        limit=5,
        preferred_source_groups=["parking", "budget"],
    )

    assert bundle is not None
    assert "currency_total_answer=$65" in bundle
    assert "currency_total_answer=$90" not in bundle
    assert "currency_total_answer=$3,565" not in bundle


def test_source_synthesis_bundle_prefers_derived_unit_price_over_unrelated_unit_price() -> None:
    """Source bundles should preserve derived unit-price candidates through ordering and packet parsing."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How much did I spend on each coffee mug for my coworkers?",
        source_results=[
            (
                "longmemeval_session_id=answer-total user: I once spent $60 on some "
                "coffee mugs for my coworkers."
            ),
            (
                "longmemeval_session_id=distractor user: I sold 20 potted herb plants "
                "at the market for $7.50 each."
            ),
            (
                "longmemeval_session_id=answer-count user: I purchased 5 coffee mugs "
                "with funny quotes, one for each coworker."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert result.packet["answer_candidates"][0]["type"] == "derived_currency"
    assert result.packet["answer_candidates"][0]["answer_key"] == "currency_unit_price_answer"
    assert result.packet["answer_candidates"][0]["answer"] == "$12"
    assert "currency_unit_price_answer=$12" in result.content


def test_source_synthesis_bundle_itemizes_quoted_work_duration_intervals() -> None:
    """Quoted work duration questions should bind start/finish dates per named target."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query=(
            "How many weeks in total do I spent on reading 'The Nightingale' and listening "
            "to 'Sapiens: A Brief History of Humankind' and 'The Power'?"
        ),
        source_results=[
            (
                "longmemeval_session_id=nightingale-start longmemeval_session_date=2022/01/01 (Sat) "
                "user: I started reading 'The Nightingale' by Kristin Hannah today."
            ),
            (
                "longmemeval_session_id=nightingale-finish longmemeval_session_date=2022/01/15 (Sat) "
                "user: I just finished reading \"The Nightingale\" by Kristin Hannah today."
            ),
            (
                "longmemeval_session_id=sapiens-start longmemeval_session_date=2022/02/01 (Tue) "
                "user: I just started listening to 'Sapiens: A Brief History of Humankind' today."
            ),
            (
                "longmemeval_session_id=sapiens-finish longmemeval_session_date=2022/03/01 (Tue) "
                "user: I just finished listening to 'Sapiens: A Brief History of Humankind' today."
            ),
            (
                "longmemeval_session_id=habit-distractor longmemeval_session_date=2022/02/01 (Tue) "
                "user: I started reading \"The Power of Habit\" today."
            ),
            (
                "longmemeval_session_id=power-start longmemeval_session_date=2022/03/06 (Sun) "
                "user: I started listening to \"The Power\" by Naomi Alderman today."
            ),
            (
                "longmemeval_session_id=power-finish longmemeval_session_date=2022/03/20 (Sun) "
                "user: I just finished listening to 'The Power' by Naomi Alderman today. "
                "It reminded me of the emotional impact of 'The Nightingale'."
            ),
            (
                "longmemeval_session_id=distractor longmemeval_session_date=2022/03/20 (Sun) "
                "user: I finished another audiobook in 10 minutes."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert result.packet["answer_candidates"][0]["type"] == "quoted_target_duration"
    assert result.packet["answer_candidates"][0]["answer_key"] == "quoted_target_duration_answer"
    assert result.packet["answer_candidates"][0]["answer"] == (
        "2 weeks for 'The Nightingale', 4 weeks for 'Sapiens: A Brief History of Humankind', "
        "and 2 weeks for 'The Power', so a total of 8 weeks."
    )
    assert "quoted_target_duration_answer=" in result.content


def test_multi_quoted_duration_source_lane_expands_each_target() -> None:
    """Duration aggregation source queries should fan out per quoted work target."""
    query = (
        "How many weeks in total do I spent on reading 'The Nightingale' and listening "
        "to 'Sapiens: A Brief History of Humankind' and 'The Power'?"
    )

    queries = retrieval_plan.source_lane_queries(query, [])

    assert any("The Nightingale" in source_query for source_query in queries[1:])
    assert any("Sapiens: A Brief History of Humankind" in source_query for source_query in queries[1:])
    assert any("The Power" in source_query for source_query in queries[1:])
    assert all(
        {"started", "finished", "reading", "listening"} <= set(retrieval_plan.source_tokens(source_query))
        for source_query in queries[1:]
        if any(target in source_query for target in ("The Nightingale", "Sapiens: A Brief History of Humankind", "The Power"))
    )


def test_paid_event_aggregation_source_lane_expands_payment_evidence() -> None:
    """Money-spent event aggregations should search for paid and free attendance evidence."""
    queries = retrieval_plan.source_lane_queries(
        "How much total money did I spend on attending workshops in the last four months?",
        [],
    )

    assert any(
        {"paid", "attended", "attend", "workshop", "workshops", "free"} <= set(retrieval_plan.source_tokens(query))
        for query in queries[1:]
    )


def test_aggregate_candidate_projection_derives_sales_total_from_quantity_and_unit_price() -> None:
    """Derived currency synthesis should multiply sold quantity by cited unit price."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How much have I made from selling eggs this month?",
        [
            "longmemeval_session_id=answer-quantity user: I've sold a total of 40 dozen eggs so far this month.",
            "longmemeval_session_id=answer-rate user: I've been selling the eggs to my neighbor for $3 a dozen.",
        ],
    )

    assert projection.answer_candidates[0]["type"] == "derived_currency"
    assert projection.answer_candidates[0]["answer_key"] == "currency_product_answer"
    assert projection.answer_candidates[0]["answer"] == "$120"
    assert projection.answer_candidates[0]["support_source_ids"] == ["answer-quantity", "answer-rate"]
    assert projection.operations[0]["name"] == "multiply_quantity_by_unit_price"
    assert projection.operations[0]["program"]["complete"] is True


def test_aggregate_candidate_projection_derives_discount_from_points_conversion() -> None:
    """Derived currency synthesis should convert cited loyalty points into a discount."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How much discount will I get on my next purchase at FreshMart?",
        [
            (
                "longmemeval_session_id=answer-points user: I just reached 500 points, "
                "the minimum points required for a discount at FreshMart."
            ),
            (
                "longmemeval_session_id=answer-rate user: Every 100 points translate "
                "to a $1 discount on my next purchase."
            ),
        ],
    )

    assert projection.answer_candidates[0]["type"] == "derived_currency"
    assert projection.answer_candidates[0]["answer_key"] == "currency_points_discount_answer"
    assert projection.answer_candidates[0]["answer"] == "$5"
    assert projection.answer_candidates[0]["support_source_ids"] == ["answer-points", "answer-rate"]
    assert projection.operations[0]["name"] == "convert_points_to_currency"
    assert projection.operations[0]["program"]["complete"] is True


def test_aggregate_candidate_projection_uses_exact_currency_arithmetic() -> None:
    """Derived currency synthesis should not leak binary-float artifacts for cents."""
    projection = evidence_candidates.aggregate_candidate_projection(
        "How much have I made from selling stickers this month?",
        [
            "longmemeval_session_id=answer-quantity user: I sold 3 items from my sticker batch this month.",
            "longmemeval_session_id=answer-rate user: I've been selling the stickers for $0.10 each.",
        ],
    )

    assert projection.answer_candidates[0]["type"] == "derived_currency"
    assert projection.answer_candidates[0]["answer_key"] == "currency_product_answer"
    assert projection.answer_candidates[0]["answer"] == "$0.30"
    assert "derived_currency_operands=quantity:3,unit_price:0.10" in projection.lines


def test_source_synthesis_bundle_emits_auditable_ledger_rows() -> None:
    """Generated synthesis bundles should carry ledger include/exclude decisions."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How much total money have I spent on bike-related expenses?",
        source_results=[
            "longmemeval_session_id=answer-1 I bought my Bell Zephyr bike helmet for $120.",
            "longmemeval_session_id=answer-2 I replaced the bike chain and it cost me $25.",
            "longmemeval_session_id=answer-3 I got a new set of bike lights installed, which were $40.",
            "longmemeval_session_id=answer-4 I recently got a new set of bike lights installed, which were $40.",
        ],
        limit=5,
    )

    assert result is not None
    bundle = result.content
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]

    assert [row["source_group"] for row in rows if not row.get("exclude_reason")] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
    assert [(row["source_group"], row["exclude_reason"]) for row in rows if row.get("exclude_reason")] == [
        ("answer-4", "duplicate_identity")
    ]


def test_age_average_bundle_uses_typed_aggregate_projection(monkeypatch) -> None:
    """Age-average output should come from typed synthesis operations, not ad hoc line rendering."""
    monkeypatch.setattr(retrieval_plan, "_age_average_synthesis_lines", lambda query, contexts: [])
    result = retrieval_plan.source_synthesis_bundle_result(
        query="What is the average age of me, my parents, and my grandparents?",
        source_results=[
            "longmemeval_session_id=answer_1 I just turned 32 on February 12th.",
            "longmemeval_session_id=answer_2 my parents are getting older too - my mom is 55 and my dad is 58.",
            "longmemeval_session_id=answer_3 My grandma is 75 and my grandpa is 78.",
        ],
        limit=5,
    )

    assert result is not None
    bundle = result.content
    assert "candidate_type=number" in bundle
    assert "age_values=32,55,58,75,78" in bundle
    assert "age_average=59.6" in bundle
    assert bundle.count("age_average=59.6") == 1
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    age_rows = [row for row in rows if row.get("include_reason") == "age_average_input"]
    assert [row["value"] for row in age_rows] == ["32", "55", "58", "75", "78"]
    assert {"fact_id", "citation", "kind", "value", "include_reason"} <= set(rows[0])


def test_elapsed_duration_at_event_bundle_emits_ledger_rows() -> None:
    """Elapsed-duration arithmetic should preserve both input rows in the ledger."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How long had I been taking guitar lessons when I bought the new guitar amp?",
        source_results=[
            (
                "longmemeval_session_id=answer_436d4309_1 "
                "I've been taking weekly guitar lessons with Alex for six weeks now."
            ),
            (
                "longmemeval_session_id=answer_436d4309_2 "
                "I just got a new amp two weeks ago and want to get the most out of it."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]

    assert "elapsed_at_event_answer=Four weeks" in bundle
    assert [(row["source_group"], row["kind"], row["value"], row["unit"]) for row in rows] == [
        ("answer_436d4309_1", "duration", "6", "weeks"),
        ("answer_436d4309_2", "duration", "2", "weeks_ago"),
    ]
    assert {row["include_reason"] for row in rows} == {
        "current_activity_duration",
        "event_age_duration",
    }


def test_social_media_break_bundle_emits_break_specific_ledger_rows() -> None:
    """Social-media break totals should be backed by break-specific ledger rows."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many days did I take social media breaks in total?",
        source_results=[
            (
                "longmemeval_session_id=answer_a4204937_1 "
                "Choose a daily time limit like 15 minutes or 1 hour for social media."
            ),
            (
                "longmemeval_session_id=answer_a4204937_1 "
                "I've been making an effort to cut down on social media lately - "
                "I even took a week-long break from it in mid-January."
            ),
            (
                "longmemeval_session_id=answer_a4204937_2 "
                "I've been making an effort to cut down on social media lately - "
                "I actually just got back from a 10-day break in mid-February."
            ),
            (
                "longmemeval_session_id=distractor "
                "Set a daily time limit of 15 minutes for Instagram Monday to Friday."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    break_rows = [row for row in rows if row.get("include_reason") == "social_media_break_duration"]

    assert "social_media_break_total=17 days" in bundle
    assert [(row["source_group"], row["value"], row["unit"]) for row in break_rows] == [
        ("answer_a4204937_1", "7", "days"),
        ("answer_a4204937_2", "10", "days"),
    ]
    assert any(
        row["source_group"] == "distractor" and row.get("exclude_reason") == "not_personal_memory"
        for row in rows
    )


def test_road_trip_drive_bundle_emits_drive_specific_ledger_rows() -> None:
    """Road-trip drive totals should be backed by destination-drive ledger rows."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many hours in total did I spend driving to my three road trip destinations combined?",
        source_results=[
            (
                "longmemeval_session_id=answer_526354c8_1 "
                "my recent trip to Outer Banks in North Carolina - "
                "it only took me four hours to drive there from my place."
            ),
            (
                "longmemeval_session_id=answer_526354c8_2 "
                "when I drove for six hours to Washington D.C. recently"
            ),
            (
                "longmemeval_session_id=answer_526354c8_3 "
                "my recent trip to the mountains in Tennessee - "
                "I drove for five hours to get there and it was totally worth it."
            ),
            (
                "longmemeval_session_id=distractor "
                "From the Outer Banks, it is about a 2-hour drive to Topsail Island, "
                "and then another 4-5 hours to Tybee Island from there."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    drive_rows = [row for row in rows if row.get("include_reason") == "road_trip_destination_drive_duration"]

    assert "road_trip_drive_total=15 hours" in bundle
    assert "road_trip_drive_total_round_trip=30 hours" in bundle
    assert [(row["source_group"], row["value"], row["unit"]) for row in drive_rows] == [
        ("answer_526354c8_1", "4", "hours"),
        ("answer_526354c8_3", "5", "hours"),
        ("answer_526354c8_2", "6", "hours"),
    ]
    assert any(
        row["source_group"] == "distractor" and row.get("exclude_reason") == "not_personal_memory"
        for row in rows
    )


def test_currency_synthesis_does_not_emit_unledgered_duration_fallback() -> None:
    """Currency-only synthesis should not leak unrelated duration totals."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How much more did I spend on accommodations per night in Hawaii compared to Tokyo?",
        source_results=[
            "longmemeval_session_id=answer-1 I spent $300 per night for the Hawaii hotel.",
            "longmemeval_session_id=answer-2 I spent $30 per night for the Tokyo capsule hotel.",
            "longmemeval_session_id=distractor I jogged for 30 minutes before checkout.",
        ],
        limit=5,
    )

    assert bundle is not None
    assert "currency_difference_answer=$270" in bundle
    assert "minute_total_hours=0.5 hours" not in bundle


def test_age_at_event_bundle_emits_operation_ledger_rows() -> None:
    """Age-at-event subtraction should preserve current age and elapsed-year inputs."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How old was I when I moved to the United States?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer_1 "
                "I'm 32-year-old male and have been updating my immigration paperwork."
            ),
            (
                "citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_id=answer_2 "
                "I've been living in the United States for the past five years on a work visa."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    age_rows = [row for row in rows if row.get("fact_id", "").startswith("age_at_event:")]

    assert "age_at_event_answer=27" in bundle
    assert [(row["source_group"], row["value"], row["unit"], row["include_reason"]) for row in age_rows] == [
        ("answer_1", "32", "years", "current_age"),
        ("answer_2", "5", "elapsed_years", "elapsed_since_event"),
    ]


def test_age_at_event_bundle_derives_birth_age_from_target_age() -> None:
    """Birth-age questions should subtract the named target's age from current age."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How old was I when Alex was born?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer_1 "
                "I'm glad to mentor Alex. He's just 21 and I'm excited to see him grow."
            ),
            (
                "citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_id=answer_2 "
                "I just turned 32 last month."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "age_current=32" in bundle
    assert "age_elapsed_years=21" in bundle
    assert "age_at_event_operation=32-21" in bundle
    assert "age_at_event_answer=11" in bundle


def test_future_age_at_event_bundle_adds_cited_future_offset() -> None:
    """Future first-person age questions should add current age and event offset."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many years will I be when my friend Rachel gets married?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer_1 "
                "My friend Rachel's getting married next year, and it's got me thinking "
                "about my own life goals."
            ),
            (
                "citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_id=answer_2 "
                "I'm 32, so I'm in my 30s. My main concerns are fine lines and wrinkles."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    future_rows = [
        row for row in rows if row.get("fact_id", "").startswith("future_age_at_event:")
    ]

    assert "future_age_at_event_operation=32+1" in bundle
    assert "future_age_at_event_answer=33" in bundle
    assert [
        (row["source_group"], row["value"], row["unit"], row["include_reason"])
        for row in future_rows
    ] == [
        ("answer_2", "32", "years", "current_age"),
        ("answer_1", "1", "future_years", "future_event_offset"),
    ]


def test_career_prior_duration_bundle_emits_operation_ledger_rows() -> None:
    """Career-prior subtraction should preserve total and current-role inputs."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How long have I been working before I started my current job at NovaTech?",
        source_results=[
            (
                "citation=eventloom://benchmark/events/1#abc "
                "longmemeval_session_id=answer_1 "
                "I've been working professionally for 9 years and I'm currently using a notebook."
            ),
            (
                "citation=eventloom://benchmark/events/2#def "
                "longmemeval_session_id=answer_2 "
                "I've been working at NovaTech for about 4 years and 3 months now."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    career_rows = [row for row in rows if row.get("fact_id", "").startswith("career_prior_duration:")]

    assert "career_prior_duration_answer=4 years and 9 months" in bundle
    assert [(row["source_group"], row["value"], row["unit"], row["include_reason"]) for row in career_rows] == [
        ("answer_1", "108", "months", "total_career_duration"),
        ("answer_2", "51", "months", "current_role_duration"),
    ]


def test_age_average_bundle_emits_age_ledger_rows() -> None:
    """Age-average fields should preserve each age input with source groups."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="What is the average age of me, my parents, and my grandparents?",
        source_results=[
            "longmemeval_session_id=answer_2504635e_1 I just turned 32 on February 12th.",
            "longmemeval_session_id=answer_2504635e_2 my parents are getting older too - my mom is 55 and my dad is 58.",
            "longmemeval_session_id=answer_2504635e_3 My grandma is 75 and my grandpa is 78.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    age_rows = [row for row in rows if row.get("include_reason") == "age_average_input"]

    assert "age_average=59.6" in bundle
    assert [(row["source_group"], row["value"], row["unit"]) for row in age_rows] == [
        ("answer_2504635e_1", "32", "years"),
        ("answer_2504635e_2", "55", "years"),
        ("answer_2504635e_2", "58", "years"),
        ("answer_2504635e_3", "75", "years"),
        ("answer_2504635e_3", "78", "years"),
    ]


def test_relative_week_interval_bundle_emits_anchor_ledger_rows() -> None:
    """Week-interval fields should preserve both relative-time anchors."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How long had I been a member of Book Lovers Unite when I attended the meetup?",
        source_results=[
            "longmemeval_session_id=joined I joined Book Lovers Unite three weeks ago.",
            "longmemeval_session_id=meetup I attended a meetup organized by Book Lovers Unite last week.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    interval_rows = [row for row in rows if row.get("include_reason") == "relative_week_anchor"]

    assert "week_interval_answer=Two weeks" in bundle
    assert [(row["source_group"], row["value"], row["unit"]) for row in interval_rows] == [
        ("joined", "3", "weeks_ago"),
        ("meetup", "1", "weeks_ago"),
    ]


def test_mixed_relative_interval_bundle_emits_month_and_week_ledger_rows() -> None:
    """Mixed month/week interval fields should preserve normalized anchors."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How long had I been using the new area rug when I rearranged my living room furniture?",
        source_results=[
            "longmemeval_session_id=answer_1 I recently got a new area rug for my living room a month ago.",
            "longmemeval_session_id=answer_2 I rearranged the furniture three weeks ago.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    interval_rows = [row for row in rows if row.get("include_reason") in {"relative_month_anchor", "relative_week_anchor"}]

    assert "relative_week_interval_answer=One week" in bundle
    assert [(row["source_group"], row["value"], row["unit"], row["include_reason"]) for row in interval_rows] == [
        ("answer_1", "1", "months_ago", "relative_month_anchor"),
        ("answer_2", "3", "weeks_ago", "relative_week_anchor"),
    ]


def test_query_temporal_anchor_bundle_answers_elapsed_months_from_session_date() -> None:
    """Relative-time questions should use the question date anchor, not source text durations."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many months ago did I attend the Seattle International Film Festival?",
        source_results=[
            (
                "query_temporal_anchor=true longmemeval_session_id=query-temporal-anchor "
                "longmemeval_session_date=2021/10/02 (query) role=query"
            ),
            (
                "longmemeval_session_id=festival longmemeval_session_date=2021/06/01 (Tue) "
                "user: I attended the Seattle International Film Festival and saw three documentaries."
            ),
            (
                "longmemeval_session_id=distractor longmemeval_session_date=2021/09/20 (Mon) "
                "user: I attended a work conference in Portland."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "candidate_type=relative_temporal_anchor" in bundle
    assert "relative_temporal_anchor_answer=4 months ago" in bundle
    assert "relative_temporal_anchor_source_id=festival" in bundle


def test_query_temporal_anchor_bundle_answers_elapsed_weeks_from_session_date() -> None:
    """Week elapsed answers should derive from session-date distance to the query anchor."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many weeks ago did I attend the 'Summer Nights' festival at Universal Studios Hollywood?",
        source_results=[
            (
                "query_temporal_anchor=true longmemeval_session_id=query-temporal-anchor "
                "longmemeval_session_date=2023/08/05 (query) role=query"
            ),
            (
                "longmemeval_session_id=summer-nights longmemeval_session_date=2023/07/15 (Sat) "
                "user: I attended the Summer Nights festival at Universal Studios Hollywood."
            ),
            (
                "longmemeval_session_id=month-duration longmemeval_session_date=2023/08/01 (Tue) "
                "user: I bought a pass that lasts 1 month."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "relative_temporal_anchor_answer=3 weeks ago" in bundle
    assert "relative_temporal_anchor_source_id=summer-nights" in bundle


def test_query_temporal_anchor_synthesis_ignores_since_duration_total_queries() -> None:
    """Relative anchors should not crowd arithmetic queries that merely mention an ago duration."""
    assert not retrieval_plan.query_temporal_anchor_synthesis_query(
        "How many total pieces of writing have I completed since I started writing again three weeks ago?"
    )


def test_query_temporal_anchor_bundle_selects_direct_answer_for_target_days() -> None:
    """Explicit N-days-ago questions should select the source by session-date distance."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="What kitchen appliance did I buy 10 days ago?",
        source_results=[
            (
                "query_temporal_anchor=true longmemeval_session_id=query-temporal-anchor "
                "longmemeval_session_date=2023/03/25 (query) role=query"
            ),
            (
                "longmemeval_session_id=smoker longmemeval_session_date=2023/03/15 (Wed) "
                "user: I bought a smoker so I can try slow-cooked ribs."
            ),
            (
                "longmemeval_session_id=blender longmemeval_session_date=2023/03/20 (Mon) "
                "user: I bought a blender for morning smoothies."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "relative_temporal_anchor_answer=a smoker" in bundle
    assert "relative_temporal_anchor_source_id=smoker" in bundle


def test_query_temporal_anchor_bundle_answers_days_with_inclusive_alternative() -> None:
    """Day elapsed answers should preserve the inclusive-count alternative used by the benchmark."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many days ago did I read the March 15th issue of The New Yorker?",
        source_results=[
            (
                "query_temporal_anchor=true longmemeval_session_id=query-temporal-anchor "
                "longmemeval_session_date=2023/04/01 (query) role=query"
            ),
            (
                "longmemeval_session_id=new-yorker longmemeval_session_date=2023/03/20 (Mon) "
                "user: I read the March 15th issue of The New Yorker."
            ),
        ],
        limit=5,
    )

    assert bundle is not None
    assert "relative_temporal_anchor_answer=12 days ago. 13 days (including the last day) is also acceptable." in bundle


def test_anniversary_interval_bundle_emits_month_day_ledger_rows() -> None:
    """Anniversary subtraction should preserve both dated inputs."""
    bundle = retrieval_plan.source_synthesis_bundle(
        query="How many months before my anniversary did Rachel get engaged?",
        source_results=[
            "longmemeval_session_id=answer_aaf71ce2_2 My close friend Rachel got engaged last month on May 15th.",
            "longmemeval_session_id=answer_aaf71ce2_3 Our anniversary is coming up on July 22nd.",
        ],
        limit=5,
    )

    assert bundle is not None
    rows = [
        json.loads(line.removeprefix("ledger_row="))
        for line in bundle.splitlines()
        if line.startswith("ledger_row=")
    ]
    anniversary_rows = [row for row in rows if row.get("fact_id", "").startswith("anniversary_engagement:")]

    assert "anniversary_engagement_interval_answer=2 months" in bundle
    assert [(row["source_group"], row["value"], row["unit"], row["include_reason"]) for row in anniversary_rows] == [
        ("answer_aaf71ce2_2", "5/15", "month_day", "engagement_date"),
        ("answer_aaf71ce2_3", "7/22", "month_day", "anniversary_date"),
    ]


def test_categorical_temporal_bundle_emits_choice_ledger_rows() -> None:
    """Parent, recency, and temporal-order answers should expose compared candidates."""
    cases = [
        (
            "Who became a parent first, Rachel or Alex?",
            [
                "longmemeval_session_id=answer_1 Rachel's twins Jackson and Julia were born on February 12th.",
                "longmemeval_session_id=answer_2 My cousin Alex just adopted a baby girl from China in January.",
            ],
            "parent_order_answer=Alex",
            "parent_order_candidate",
            [("answer_2", "Alex"), ("answer_1", "Rachel")],
        ),
        (
            "Which streaming service did I start using most recently?",
            [
                "longmemeval_session_id=answer_1 I started using Hulu a few months ago.",
                "longmemeval_session_id=answer_2 I started using Disney+ last week.",
            ],
            "recency_answer=Disney+",
            "recency_candidate",
            [("answer_2", "Disney+"), ("answer_1", "Hulu")],
        ),
        (
            "Who did I meet first, Mark and Sarah or Tom?",
            [
                "longmemeval_session_id=answer_1 I met Mark and Sarah on a beach trip about a month ago.",
                "longmemeval_session_id=answer_2 A few months ago, I volunteered and met a guy named Tom.",
            ],
            "temporal_order_answer=Tom",
            "temporal_order_candidate",
            [("answer_2", "Tom"), ("answer_1", "Mark and Sarah")],
        ),
    ]

    for query, source_results, answer_line, include_reason, expected in cases:
        bundle = retrieval_plan.source_synthesis_bundle(
            query=query,
            source_results=source_results,
            limit=5,
        )

        assert bundle is not None
        rows = [
            json.loads(line.removeprefix("ledger_row="))
            for line in bundle.splitlines()
            if line.startswith("ledger_row=")
        ]
        choice_rows = [row for row in rows if row.get("include_reason") == include_reason]

        assert answer_line in bundle
        assert [(row["source_group"], row["candidate"]) for row in choice_rows] == expected


def test_temporal_sequence_bundle_emits_ordered_candidate_packet() -> None:
    """Ordered-list temporal synthesis should expose answer candidates and evidence-program coverage."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="What is the order of the three trips I took in the past three months, from earliest to latest?",
        source_results=[
            (
                "content=longmemeval_session_id=answer-trip-3 "
                "user: I just got back from a solo camping trip to Yosemite National Park today."
            ),
            (
                "content=longmemeval_session_id=answer-trip-1 "
                "user: I went on a day hike to Muir Woods about two months ago."
            ),
            (
                "content=longmemeval_session_id=answer-trip-2 "
                "user: I got back from a road trip with friends to Big Sur and Monterey last month."
            ),
        ],
        limit=5,
    )

    assert result is not None
    bundle = result.content
    assert (
        "temporal_sequence_answer=First, day hike to Muir Woods. "
        "Then, road trip with friends to Big Sur and Monterey. "
        "Lastly, solo camping trip to Yosemite National Park."
    ) in bundle
    assert result.packet["answer_candidates"][0]["type"] == "temporal_sequence"
    assert result.packet["operations"][0]["name"] == "temporal_sequence"
    assert result.packet["operations"][0]["program"]["complete"] is True
    sequence_rows = [
        row
        for row in result.packet["ledger_rows"]
        if row.get("include_reason") == "relative_time_anchor"
    ]
    assert [(row["source_group"], row["label"]) for row in sequence_rows] == [
        ("answer-trip-1", "day hike to Muir Woods"),
        ("answer-trip-2", "road trip with friends to Big Sur and Monterey"),
        ("answer-trip-3", "solo camping trip to Yosemite National Park"),
    ]


def test_temporal_sequence_intent_overfetches_source_candidates() -> None:
    """Ordered-list temporal queries need enough source slots to cover multiple events."""
    intent = classify_retrieval_intent(
        "What is the order of the three trips I took in the past three months, from earliest to latest?",
        limit=10,
    )

    assert "temporal_sequence" in intent.reasons
    assert retrieval_plan.source_synthesis_candidate_limit(intent, limit=10) == 64


def test_temporal_sequence_museum_queries_expand_source_side_visit_actions() -> None:
    """Ordered venue-list queries should search source-side attendance and tour wording."""
    queries = retrieval_plan.source_lane_queries(
        "What is the order of the six museums I visited from earliest to latest?",
        [],
    )

    expansion = " ".join(queries[1:]).casefold()

    assert "museum" in expansion
    assert "attended" in expansion
    assert "lecture" in expansion
    assert "tour" in expansion
    assert "came back" in expansion


def test_numeric_state_bundle_prefers_latest_total_over_event_count() -> None:
    """Current-state count questions should expose stated totals ahead of generic event counts."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How many different species of birds have I seen in my local park?",
        source_results=[
            (
                "longmemeval_session_id=birds_1 "
                "user: I've managed to spot 27 different species so far in my local park."
            ),
            (
                "longmemeval_session_id=birds_2 "
                "user: I just saw a Northern Flicker, which brings my total species count to 32."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "numeric_state_answer=32" in result.content
    assert result.packet["answer_candidates"][0]["type"] == "numeric_state"
    assert result.packet["answer_candidates"][0]["answer"] == "32"
    assert result.packet["operations"][0]["program"]["complete"] is True


def test_numeric_state_bundle_carries_forward_increment_after_prior_total() -> None:
    """State-update questions should add later increments to the latest cited total."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How many pre-1920 American coins do I have in my collection?",
        source_results=[
            (
                "longmemeval_session_id=coins_1 "
                "user: I have a total of 37 pre-1920 American coins in that collection."
            ),
            (
                "longmemeval_session_id=coins_2 "
                "user: I just added a new coin to my collection of pre-1920 American coins - a 1915-S Barber quarter."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "numeric_state_operation=37+1" in result.content
    assert "numeric_state_answer=38" in result.content
    assert result.packet["answer_candidates"][0]["type"] == "numeric_state"
    assert result.packet["answer_candidates"][0]["answer"] == "38"


def test_numeric_state_bundle_answers_count_increase_before_duration() -> None:
    """Count-delta questions should prefer numeric-state difference over duration mentions."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="What was the approximate increase in Instagram followers I experienced in two weeks?",
        source_results=[
            (
                "longmemeval_session_id=followers_1 "
                "user: I started the year with 250 followers on Instagram, by the way."
            ),
            (
                "longmemeval_session_id=followers_2 "
                "user: After two weeks of posting regularly, I had around 350 followers on Instagram."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "numeric_state_operation=350-250" in result.content
    assert "numeric_state_difference_answer=100" in result.content
    assert "duration_total_answer=2 weeks" in result.content
    assert result.packet["answer_candidates"][0]["type"] == "numeric_state"
    assert result.packet["answer_candidates"][0]["answer"] == "100"


def test_percentage_comparison_bundle_answers_boolean_discount_comparison() -> None:
    """Percentage comparison questions should bind cited operands to named targets."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query=(
            "Did I receive a higher percentage discount on my first order from HelloFresh, "
            "compared to my first UberEats order?"
        ),
        source_results=[
            (
                "longmemeval_session_id=hellofresh "
                "user: I recently tried HelloFresh and got a 40% discount on my first order."
            ),
            (
                "longmemeval_session_id=ubereats "
                "user: Last week I got 20% off my UberEats order."
            ),
            (
                "longmemeval_session_id=distractor "
                "user: I usually compare takeout delivery fees before ordering dinner."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "candidate_type=boolean_comparison" in result.content
    assert "percentage_left_label=hellofresh" in result.content
    assert "percentage_left_value=40%" in result.content
    assert "percentage_right_label=ubereats" in result.content
    assert "percentage_right_value=20%" in result.content
    assert "boolean_comparison_answer=Yes" in result.content
    assert result.packet["answer_candidates"][0]["type"] == "boolean_comparison"
    assert result.packet["answer_candidates"][0]["answer_key"] == "boolean_comparison_answer"
    assert result.packet["answer_candidates"][0]["answer"] == "Yes"
    assert result.packet["answer_candidates"][0]["support_source_ids"] == ["hellofresh", "ubereats"]


def test_percentage_comparison_bundle_answers_negative_lower_question() -> None:
    """The same percentage-comparison path should answer No when the requested relation is false."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query=(
            "Did I receive a lower percentage discount on my first order from HelloFresh, "
            "compared to my first UberEats order?"
        ),
        source_results=[
            (
                "longmemeval_session_id=hellofresh "
                "user: I recently tried HelloFresh and got a 40% discount on my first order."
            ),
            (
                "longmemeval_session_id=ubereats "
                "user: Last week I got 20% off my UberEats order."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "boolean_comparison_operator=lower" in result.content
    assert "boolean_comparison_answer=No" in result.content
    assert result.packet["answer_candidates"][0]["type"] == "boolean_comparison"
    assert result.packet["answer_candidates"][0]["answer"] == "No"


def test_direct_boolean_evidence_bundle_answers_same_method_question() -> None:
    """Direct yes/no synthesis should require explicit cited equivalence evidence."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="Is my mom using the same grocery list method as me?",
        source_results=[
            (
                "longmemeval_session_id=paper "
                "user: I use a grocery list app now, but my mom is still stuck on paper lists."
            ),
            (
                "longmemeval_session_id=shared_app "
                "user: My mom is actually using the same grocery list app as me now, "
                "so we can share lists."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "candidate_type=boolean_evidence" in result.content
    assert "boolean_evidence_answer=Yes" in result.content
    assert result.packet["answer_candidates"][0]["type"] == "boolean_evidence"
    assert result.packet["answer_candidates"][0]["answer_key"] == "boolean_evidence_answer"
    assert result.packet["answer_candidates"][0]["answer"] == "Yes"
    assert result.packet["answer_candidates"][0]["support_source_ids"] == ["shared_app"]


def test_direct_boolean_evidence_bundle_answers_possession_question() -> None:
    """Current first-person possession evidence should answer direct have/has questions."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="Do I have a spare screwdriver for opening up my laptop?",
        source_results=[
            (
                "longmemeval_session_id=missing_tool "
                "user: I need to open up my laptop, but I misplaced the small screwdriver."
            ),
            (
                "longmemeval_session_id=found_tool "
                "user: I actually have a spare screwdriver that I picked up when I organized "
                "my computer desk, so I am all set for the laptop."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "candidate_type=boolean_evidence" in result.content
    assert "boolean_evidence_answer=Yes" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "Yes"
    assert result.packet["answer_candidates"][0]["support_source_ids"] == ["found_tool"]


def test_direct_boolean_evidence_does_not_answer_related_negated_mentions() -> None:
    """Related terms without explicit affirmative evidence should not synthesize Yes."""
    assert (
        retrieval_plan.source_synthesis_bundle_result(
            query="Do I have a spare screwdriver for opening up my laptop?",
            source_results=[
                (
                    "longmemeval_session_id=missing_tool "
                    "user: I need to open up my laptop, but I misplaced the small screwdriver."
                ),
                (
                    "longmemeval_session_id=store_advice "
                    "assistant: The electronics store near your place should have the "
                    "screwdriver you need."
                ),
            ],
            limit=5,
        )
        is None
    )


def test_direct_boolean_evidence_answers_temporal_frequency_increase() -> None:
    """More/less frequency questions should compare explicit earlier and later cadences."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="Do I go to the gym more frequently than I did previously?",
        source_results=[
            (
                "longmemeval_session_id=gym_1 "
                "user: I go to the gym on Tuesdays, Thursdays, and Saturdays."
            ),
            (
                "longmemeval_session_id=gym_2 "
                "user: I've been consistent with my gym routine - four times a week, actually."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "candidate_type=boolean_evidence" in result.content
    assert "frequency_previous_per_week=3" in result.content
    assert "frequency_current_per_week=4" in result.content
    assert "boolean_evidence_answer=Yes" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "Yes"
    assert result.packet["answer_candidates"][0]["support_source_ids"] == ["gym_1", "gym_2"]


def test_direct_boolean_evidence_refuses_equal_temporal_frequency() -> None:
    """Equal old/new cadences should not be converted into a yes/no comparison."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="Do I go to the gym more frequently than I did previously?",
        source_results=[
            "longmemeval_session_id=gym_1 user: I go to the gym three times a week.",
            "longmemeval_session_id=gym_2 user: My gym routine is still three times a week.",
        ],
        limit=5,
    )

    assert result is not None
    assert "boolean_evidence_answer=" not in result.content
    assert result.packet["answer_candidates"] == []


def test_query_bound_direct_answer_projects_current_record() -> None:
    """Current-state record questions should project the cited latest record surface."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="What is my current record in the recreational volleyball league?",
        source_results=[
            "longmemeval_session_id=league_1 user: In the volleyball league, we're 3-2 so far.",
            "longmemeval_session_id=league_2 user: Our volleyball team is doing well with a 5-2 record.",
        ],
        limit=5,
    )

    assert result is not None
    assert "candidate_type=query_bound_direct_answer" in result.content
    assert "query_bound_direct_answer=5-2" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "5-2"
    assert result.packet["answer_candidates"][0]["support_source_ids"] == ["league_2"]


def test_query_bound_direct_answer_projects_stated_weight_loss() -> None:
    """Direct stated quantity answers should bind to the requested activity terms."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How much weight have I lost since I started going to the gym consistently?",
        source_results=[
            "longmemeval_session_id=fitness_1 user: I set a goal to run a 10K by summer.",
            (
                "longmemeval_session_id=fitness_2 user: I've lost 10 pounds since I started "
                "going consistently to the gym 3 months ago."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "query_bound_direct_answer=10 pounds" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "10 pounds"


def test_query_bound_direct_answer_counts_weekly_class_days_across_sources() -> None:
    """Weekly class frequency should combine cited weekday schedules without summing distractor minutes."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How many days a week do I attend fitness classes?",
        source_results=[
            (
                "longmemeval_session_id=classes_1 user: I attend Zumba classes on Tuesdays "
                "and Thursdays, and a weightlifting class on Saturdays."
            ),
            (
                "longmemeval_session_id=classes_2 user: I recently started a yoga class on "
                "Wednesdays, which helps after weightlifting."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "query_bound_direct_answer=4 days" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "4 days"
    assert result.packet["answer_candidates"][0]["support_source_ids"] == ["classes_1", "classes_2"]


def test_query_bound_direct_answer_projects_limit_direction() -> None:
    """Limit-change questions should compare cited earlier and later state values."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="Did I mostly recently increase or decrease the limit on the number of cups of coffee in the morning?",
        source_results=[
            "longmemeval_session_id=coffee_1 user: I cut back to just one cup in the morning.",
            "longmemeval_session_id=coffee_2 user: I'm thinking of changing my morning coffee limit to two cups.",
        ],
        limit=5,
    )

    assert result is not None
    assert "query_bound_direct_answer=You increased the limit from one cup to two cups." in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "You increased the limit from one cup to two cups."


def test_query_bound_direct_answer_allows_single_source_duration() -> None:
    """Single-session personal memories should still emit direct duration answers."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How long was I in Japan for?",
        source_results=[
            (
                "longmemeval_session_id=japan_1 user: I was in Japan a few months ago. "
                "I spent two weeks traveling solo around the country."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "query_bound_direct_answer=two weeks" in result.content
    assert result.packet["answer_candidates"][0]["support_source_ids"] == ["japan_1"]


def test_query_bound_difference_answers_target_currency_comparison() -> None:
    """Currency differences should bind operands to query-named targets."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="For my daily commute, how much more expensive was the taxi ride compared to the train fare?",
        source_results=[
            (
                "longmemeval_session_id=train_1 user: My commute total was $240. "
                "The estimated train fare was $9.14, but my daily train fare is actually $6."
            ),
            (
                "longmemeval_session_id=taxi_1 user: I missed my train and had to take a taxi, "
                "which cost me $12."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "candidate_type=query_bound_difference" in result.content
    assert "difference_left_value=$12" in result.content
    assert "difference_right_value=$6" in result.content
    assert "query_bound_difference_answer=$6" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "$6"


def test_query_bound_difference_answers_marathon_target_overrun() -> None:
    """Duration differences should bind actual and target marathon times."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How many minutes did I exceed my target time by in the marathon?",
        source_results=[
            "longmemeval_session_id=actual user: I just completed my first full marathon in 4h 22min.",
            "longmemeval_session_id=target user: My target time for the marathon was 4 hours and 10 minutes.",
        ],
        limit=5,
    )

    assert result is not None
    assert "candidate_type=query_bound_difference" in result.content
    assert "difference_left_minutes=262" in result.content
    assert "difference_right_minutes=250" in result.content
    assert "query_bound_difference_answer=12" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "12"


def test_total_duration_query_keeps_aggregate_before_latest_state() -> None:
    """Total-duration questions should not be answered by one latest state value."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How many hours have I spent playing games in total?",
        source_results=[
            (
                "longmemeval_session_id=game_1 "
                "user: I spent around 70 hours playing Elden Ring last month."
            ),
            (
                "longmemeval_session_id=game_2 "
                "user: I spent around 30 hours playing The Last of Us Part II."
            ),
            (
                "longmemeval_session_id=game_3 "
                "user: I also logged 40 hours in Stardew Valley."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "duration_total_answer=140 hours" in result.content
    assert "latest_state_answer=30 hours" not in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "140 hours"


def test_query_bound_scalar_total_sums_people_reached_across_platforms() -> None:
    """Semantic reach totals should bind people/follower counts to named campaigns."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="What was the total number of people reached by my Facebook ad campaign and Instagram influencer collaboration?",
        source_results=[
            (
                "longmemeval_session_id=facebook "
                "user: My previous Facebook ad campaign reached around 2,000 people."
            ),
            (
                "longmemeval_session_id=instagram "
                "user: I recently collaborated with an influencer who promoted my product "
                "to her 10,000 followers."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "candidate_type=query_bound_scalar_total" in result.content
    assert "query_bound_scalar_total_values=2,000,10,000" in result.content
    assert "query_bound_scalar_total_answer=12,000" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "12,000"


def test_query_bound_scalar_total_sums_rare_collection_counts() -> None:
    """Rare item totals should sum query-bound collection counts across item types."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="How many rare items do I have in total?",
        source_results=[
            "longmemeval_session_id=books user: I have a small rare book collection of 5 books.",
            "longmemeval_session_id=figurines user: Yeah, I have 12 rare figurines in my collection.",
            "longmemeval_session_id=records user: I have 57 rare records cataloged.",
            "longmemeval_session_id=coins user: I actually have 25 rare coins that I need to store safely.",
        ],
        limit=5,
    )

    assert result is not None
    assert "query_bound_scalar_total_kind=rare_items" in result.content
    assert "query_bound_scalar_total_answer=99" in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "99"


def test_query_bound_scalar_total_sums_video_views() -> None:
    """Video view totals should bind YouTube and TikTok view counts to the query."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="What is the total number of views on my most popular videos on YouTube and TikTok?",
        source_results=[
            (
                "longmemeval_session_id=tiktok "
                "user: My TikTok video of Luna chasing a laser pointer has 1,456 views. "
                "An article I read recommended benchmarking videos with 1,400 views too."
            ),
            (
                "longmemeval_session_id=youtube "
                "user: My social media analytics tutorial on YouTube has 542 views."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "query_bound_scalar_total_kind=video_views" in result.content
    assert "query_bound_scalar_total_answer=1,998" in result.content
    assert "query_bound_scalar_total_answer=3,398" not in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "1,998"


def test_query_bound_scalar_total_sums_road_trip_miles() -> None:
    """Road-trip distance totals should sum cited covered-mile observations."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query="What is the total distance I covered in my four road trips?",
        source_results=[
            (
                "longmemeval_session_id=yellowstone "
                "user: I just got back from a road trip where we covered a total of 1,200 miles. "
                "The scenic detour was only 25 miles."
            ),
            (
                "longmemeval_session_id=recent "
                "user: Since I've covered a total of 1,800 miles on my recent three road trips, "
                "I'm comfortable with the drive."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "query_bound_scalar_total_kind=road_trip_miles" in result.content
    assert "query_bound_scalar_total_answer=3,000 miles" in result.content
    assert "query_bound_scalar_total_answer=3,025 miles" not in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "3,000 miles"


def test_missing_aggregation_target_uses_absence_bundle_before_generic_synthesis() -> None:
    """Aggregation questions with a precise missing target should be absence-first."""
    query = "What is the total cost of my recently purchased headphones and the iPad?"

    assert retrieval_plan.should_try_absence_bundle_first(query, limit=5) is True

    source_results = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=headphones_1 "
            "user: I recently purchased headphones for $200."
        ),
        (
            "citation=eventloom://benchmark/events/2#def "
            "longmemeval_session_id=headphones_2 "
            "user: I also bought a protective headphone case for $30."
        ),
    ]
    absence = retrieval_plan.absence_check_bundle(
        query=query,
        source_results=source_results,
        limit=5,
    )

    assert absence is not None
    assert "zaxy_absence_check=true" in absence
    assert "not_mentioned_candidate=ipad" in absence.casefold()
    assert retrieval_plan.source_synthesis_bundle_result(
        query=query,
        source_results=source_results,
        limit=5,
    ) is None


def test_itemized_total_cost_query_promotes_currency_bundle_without_personal_pronoun() -> None:
    """Total-cost wording should route into typed currency synthesis."""
    query = "What is the total cost of Lola's vet visit and flea medication?"

    intent = classify_retrieval_intent(query, limit=5)
    assert "aggregation_question" in intent.reasons

    result = retrieval_plan.source_synthesis_bundle_result(
        query=query,
        source_results=[
            (
                "source_path=longmemeval/events/vet "
                "longmemeval_session_id=vet "
                "content=I just took Lola to the vet last week and got a discounted "
                "consultation fee of $50 as a first-time customer."
            ),
            (
                "source_path=longmemeval/events/supplies "
                "longmemeval_session_id=supplies "
                "content=I also got her flea and tick prevention medication, it was $25 "
                "for a 3-month supply. I got Lola a bag of cat food from Petco, it was $35. "
                "Her carrier was $80 when I bought it last year."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert "currency_total_answer=$75" in result.content
    assert "currency_total_answer=$190" not in result.content
    assert result.packet["answer_candidates"][0]["answer"] == "$75"
    assert result.packet["operations"][0]["program"]["complete"] is True


def test_missing_temporal_order_target_uses_absence_bundle_before_generic_synthesis() -> None:
    """Order questions with one unsupported event should be absence-first."""
    query = "Which task did I complete first, fixing the fence or purchasing three cows from Peter?"

    assert retrieval_plan.should_try_absence_bundle_first(query, limit=5) is True

    source_results = [
        (
            "citation=eventloom://benchmark/events/1#abc "
            "longmemeval_session_id=fence_1 "
            "user: I finally finished fixing the fence this weekend."
        ),
        (
            "citation=eventloom://benchmark/events/2#def "
            "longmemeval_session_id=fence_2 "
            "user: I made a materials list for the pasture fencing project."
        ),
    ]
    absence = retrieval_plan.absence_check_bundle(
        query=query,
        source_results=source_results,
        limit=5,
    )

    assert absence is not None
    assert "zaxy_absence_check=true" in absence
    assert "not_mentioned_candidate=peter" in absence.casefold() or "not_mentioned_candidate=cows" in absence.casefold()
    assert retrieval_plan.source_synthesis_bundle_result(
        query=query,
        source_results=source_results,
        limit=5,
    ) is None


def test_order_of_events_query_promotes_temporal_sequence_bundle() -> None:
    """Order-of-events wording should use the typed temporal sequence synthesis path."""
    result = retrieval_plan.source_synthesis_bundle_result(
        query=(
            "What is the order of the three events: 'I signed up for the rewards "
            "program at ShopRite', 'I used a Buy One Get One Free coupon on Luvs "
            "diapers at Walmart', and 'I redeemed $12 cashback for a $10 Amazon "
            "gift card from Ibotta'?"
        ),
        source_results=[
            (
                "source_path=longmemeval/events/answer-3 "
                "longmemeval_session_id=answer-3 "
                "longmemeval_session_date=2023/03/20 (Mon) "
                "content=I signed up for the rewards program at ShopRite today."
            ),
            (
                "source_path=longmemeval/events/answer-1 "
                "longmemeval_session_id=answer-1 "
                "longmemeval_session_date=2023/03/05 (Sun) "
                "content=I used a Buy One Get One Free coupon on Luvs diapers at Walmart today."
            ),
            (
                "source_path=longmemeval/events/answer-2 "
                "longmemeval_session_id=answer-2 "
                "longmemeval_session_date=2023/03/12 (Sun) "
                "content=I redeemed $12 cashback for a $10 Amazon gift card from Ibotta today."
            ),
        ],
        limit=5,
    )

    assert result is not None
    assert (
        "temporal_sequence_answer=First, I used a Buy One Get One Free coupon on Luvs diapers at Walmart. "
        "Then, I redeemed $12 cashback for a $10 Amazon gift card from Ibotta. "
        "Lastly, I signed up for the rewards program at ShopRite."
    ) in result.content
    assert result.packet["operations"][0]["program"]["complete"] is True
    assert result.packet["result"]["answer_key"] == "temporal_sequence_answer"


def test_projection_result_tracks_operation_priority_rerank() -> None:
    """Merged projection result should point at the operation-prioritized candidate."""
    projection = evidence_candidates._merge_projections(
        evidence_candidates.EvidenceProjection(
            lines=("candidate_rank=1 candidate_type=count", "count_answer=2"),
            source_groups=("count-1", "count-2"),
            answer_candidates=(
                {
                    "rank": 1,
                    "type": "count",
                    "confidence": 0.99,
                    "answer_key": "count_answer",
                    "answer": "2",
                    "support_source_ids": ["count-1", "count-2"],
                    "excluded_source_ids": [],
                },
            ),
            result={
                "answer_key": "count_answer",
                "answer": "2",
                "confidence": 0.99,
                "support_source_ids": ["count-1", "count-2"],
                "excluded_source_ids": [],
            },
        ),
        evidence_candidates.EvidenceProjection(
            lines=("candidate_rank=2 candidate_type=numeric_state", "numeric_state_answer=32"),
            source_groups=("state-1", "state-2"),
            answer_candidates=(
                {
                    "rank": 2,
                    "type": "numeric_state",
                    "confidence": 0.80,
                    "answer_key": "numeric_state_answer",
                    "answer": "32",
                    "support_source_ids": ["state-1", "state-2"],
                    "excluded_source_ids": [],
                },
            ),
            result={
                "answer_key": "numeric_state_answer",
                "answer": "32",
                "confidence": 0.80,
                "support_source_ids": ["state-1", "state-2"],
                "excluded_source_ids": [],
            },
        ),
    )

    assert projection.answer_candidates[0]["type"] == "numeric_state"
    assert projection.result == {
        "answer_key": "numeric_state_answer",
        "answer": "32",
        "confidence": 0.80,
        "support_source_ids": ["state-1", "state-2"],
        "excluded_source_ids": [],
    }


def test_source_ordering_reuses_context_tokens_across_ranking_passes(monkeypatch) -> None:
    """Source synthesis ordering should not tokenize every context for every sort pass."""
    tokenized_contexts: list[str] = []
    original_source_tokens = retrieval_plan.source_tokens

    def tracking_source_tokens(text: str) -> list[str]:
        if text.startswith("source_path="):
            tokenized_contexts.append(text)
        return original_source_tokens(text)

    monkeypatch.setattr(retrieval_plan, "source_tokens", tracking_source_tokens)
    contexts = [
        f"source_path=doc-{index}.md longmemeval_session_id=answer-{index} I bought bike gear for ${index}."
        for index in range(8)
    ]
    token_cache = retrieval_plan._SourceTokenCache(tokens={})

    ordered = retrieval_plan.query_specific_source_order(
        "How much total money have I spent on bike-related expenses?",
        contexts,
        token_cache=token_cache,
    )
    retrieval_plan.evidence_source_order(
        "How much total money have I spent on bike-related expenses?",
        ordered,
        score_cache=retrieval_plan._SourceEvidenceScoreCache(
            query="irrelevant",
            scores=dict.fromkeys(contexts, 0),
        ),
        token_cache=token_cache,
    )

    assert len(tokenized_contexts) == len(contexts)


def test_retrieval_source_tokens_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Retrieval hot-path tokenization should not compile regex strings on every call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("source_tokens should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "findall", fail)
    monkeypatch.setattr(retrieval_plan.re, "search", fail)
    monkeypatch.setattr(retrieval_plan.re, "split", fail)

    assert retrieval_plan.source_tokens("source_path=longmemeval/foo-bar.md") == [
        "source_path",
        "source",
        "path",
        "longmemeval/foo-bar.md",
        "longmemeval",
        "foo",
        "bar.md",
    ]


def test_retrieval_source_tokens_uses_constant_time_separator_check(monkeypatch) -> None:
    """Retrieval token splitting should not run a regex search for every token."""
    monkeypatch.setattr(
        builtins,
        "any",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("separator checks should not allocate generator scans")
        ),
    )

    assert not hasattr(retrieval_plan, "_SOURCE_TOKEN_HAS_SEPARATOR_RE")

    assert retrieval_plan.source_tokens("source_path=longmemeval/foo-bar.md") == [
        "source_path",
        "source",
        "path",
        "longmemeval/foo-bar.md",
        "longmemeval",
        "foo",
        "bar.md",
    ]


def test_retrieval_source_tokens_caches_repeated_text_without_mutation_leak(monkeypatch) -> None:
    """Repeated source tokenization should reuse parsing while returning safe lists."""
    retrieval_plan._source_token_tuple.cache_clear()
    calls = 0
    original_token_re = retrieval_plan._SOURCE_TOKEN_RE

    class TrackingTokenRegex:
        def findall(self, text: str) -> list[str]:
            nonlocal calls
            calls += 1
            return original_token_re.findall(text)

    monkeypatch.setattr(retrieval_plan, "_SOURCE_TOKEN_RE", TrackingTokenRegex())
    text = "source_path=longmemeval/foo-bar.md I bought bike gear."

    first = retrieval_plan.source_tokens(text)
    first.append("mutated")
    second = retrieval_plan.source_tokens(text)

    assert "mutated" not in second
    assert second[:7] == [
        "source_path",
        "source",
        "path",
        "longmemeval/foo-bar.md",
        "longmemeval",
        "foo",
        "bar.md",
    ]
    assert calls == 1


def test_source_context_provenance_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Source-lane provenance parsing should not compile regex strings per candidate."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("source provenance parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    context = (
        "citation=file://longmemeval/session-1/chunk-0001.md "
        "source_path=longmemeval/session-1/chunk-0001.md "
        "longmemeval_session_id=answer-bike"
    )

    assert retrieval_plan.source_context_group(context) == "answer-bike"
    assert retrieval_plan.source_context_namespace(context) == "longmemeval/session-1"


def test_source_context_citation_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Citation extraction should not compile regex strings per source candidate."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("source citation parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert (
        retrieval_plan.source_context_citation(
            "role=user citation=eventloom://default/events/10#abc source_path=docs/guide.md"
        )
        == "eventloom://default/events/10#abc"
    )


def test_graph_answer_concepts_use_compiled_regex_helpers(monkeypatch) -> None:
    """Graph-to-source query expansion should not compile regex strings per result."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("graph concept extraction should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "findall", fail)
    monkeypatch.setattr(retrieval_plan.re, "fullmatch", fail)

    assert retrieval_plan.graph_answer_concepts(
        [
            "entity=Bike Goal summary=Rachel Project Alpha source=Event",
            "entity=deadbeefcafebabe summary=hash should not become a concept",
        ],
        limit=3,
    ) == ["Bike Goal", "Rachel Project Alpha"]


def test_valid_entity_alias_uses_compiled_regex_helper(monkeypatch) -> None:
    """Possessive alias validation should not compile regex strings per candidate."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("alias validation should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan.valid_entity_alias("Rachel", "parent") is True
    assert retrieval_plan.valid_entity_alias("parent", "parent") is False


def test_possessive_entity_targets_use_compiled_regex_helper(monkeypatch) -> None:
    """Possessive bridge target extraction should not compile regex strings per query."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("possessive target extraction should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan.possessive_entity_targets("What is my new bike timeline?") == ("bike",)
    assert retrieval_plan.possessive_entity_targets("What is my old project status?") == ()


def test_possessive_alias_extraction_uses_cached_compiled_regex_helpers(monkeypatch) -> None:
    """Possessive alias extraction should reuse compiled target-specific regexes."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("possessive alias extraction should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan.aliases_for_possessive_target(
        "I bought my new bike Trek last month.",
        "bike",
    ) == ("Trek",)
    assert retrieval_plan.aliases_for_possessive_target(
        "I ordered lights for Trek because it is my bike.",
        "bike",
    ) == ("Trek",)


def test_source_evidence_score_uses_bounded_scoring_context(monkeypatch) -> None:
    """Ranking should score a compact evidence view instead of full source chunks."""
    score_lengths: list[int] = []

    def fake_score(query: str, context: str) -> int:
        del query
        score_lengths.append(len(context))
        return 0

    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", fake_score)
    context = (
        "citation=eventloom://default/events/10#abc "
        "source_path=longmemeval/session/chunk-0001.md "
        "longmemeval_session_id=answer-1 "
        "I bought bike lights for $40. "
        + "noise " * 1_000
    )

    retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        context,
    )

    assert score_lengths
    assert max(score_lengths) <= 1_600


def test_source_evidence_score_uses_ledger_score_without_rendering_projection(monkeypatch) -> None:
    """Ranking should avoid per-source projection rendering in the hot path."""
    score_calls = 0

    def fake_projection(query: str, contexts: list[str]) -> EvidenceProjection:
        del query, contexts
        raise AssertionError("source ranking should not render aggregate projections")

    def fake_score(query: str, context: str) -> int:
        del query, context
        nonlocal score_calls
        score_calls += 1
        return 7

    monkeypatch.setattr(
        retrieval_plan,
        "aggregate_candidate_projection",
        fake_projection,
    )
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", fake_score)

    score = retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert score_calls == 1
    assert score >= 7


def test_currency_personal_evidence_hint_uses_compiled_regex_helper(monkeypatch) -> None:
    """Currency source filtering should not compile amount regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("currency personal evidence should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._currency_personal_evidence_hint("I bought bike lights for $40.") is True
    assert retrieval_plan._currency_personal_evidence_hint("Estimated travel budget is $400.") is False


def test_alternative_terms_uses_compiled_single_letter_regex(monkeypatch) -> None:
    """Alternative-term parsing should not compile identifier regexes per candidate."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("alternative terms should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._alternative_terms("Task A or task B") == ("a", "b")


def test_query_person_alternatives_uses_compiled_regex_helper(monkeypatch) -> None:
    """Person alternative parsing should not compile name regexes per query."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("person alternatives should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._query_person_alternatives(
        "Who became a parent first, Rachel or Alex?"
    ) == ("rachel", "alex")


def test_flight_count_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Flight count parsing should not compile fixed regexes per source context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("flight count parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)
    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._flight_count_in_context("I took two flights each way with United.") == 4
    assert retrieval_plan._flight_count_in_context("I booked three flights during April.") == 3


def test_road_trip_drive_hours_use_compiled_regex_helpers(monkeypatch) -> None:
    """Road-trip duration parsing should not compile fixed regexes per context batch."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("road-trip duration parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)
    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan._road_trip_drive_hour_values(
        [
            "longmemeval_session_id=drive-1 I went on a road trip. It took me 4 hours to drive there.",
            "longmemeval_session_id=drive-2 I drove for three hours to Tennessee mountains.",
            "longmemeval_session_id=route-noise I drove from home 2 hours then another 3 hours.",
        ]
    ) == [4, 3]


def test_current_activity_weeks_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Current-activity duration parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("current activity duration parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)
    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan._current_activity_weeks(
        "How long have I been taking guitar lessons?",
        ["I have been taking guitar lessons for six weeks now."],
    ) == 6


def test_event_weeks_ago_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Event-age duration parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("event weeks-ago parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._event_weeks_ago(
        "How long ago did I start pottery classes?",
        ["I started pottery classes four weeks ago."],
    ) == 4


def test_role_duration_months_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Current-role duration parsing should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("role duration parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert (
        retrieval_plan._role_duration_months(
            "I've been working at NovaTech for about 4 years and 3 months now."
        )
        == 51
    )


def test_career_total_months_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Career-total parsing should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("career total parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan._career_total_months(
        ["I've been working professionally for 9 years and currently use a notebook."]
    ) == 108


def test_current_role_months_uses_compiled_employer_regex(monkeypatch) -> None:
    """Current-role lookup should not compile employer-token regexes per query."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("current role parsing should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "findall", fail)

    assert retrieval_plan._current_role_months(
        "How long have I been working before I started my current job at NovaTech?",
        [
            "I've been working at Acme for about 2 years now.",
            "I've been working at NovaTech for about 4 years and 3 months now.",
        ],
    ) == 51


def test_current_role_months_casefolds_each_context_once() -> None:
    """Current-role lookup should not normalize the same context per employer token."""

    class TrackingContext(str):
        casefold_calls = 0

        def casefold(self) -> str:
            self.casefold_calls += 1
            return super().casefold()

    context = TrackingContext("I've been working at Acme for about 2 years now.")

    assert retrieval_plan._current_role_months(
        "How long at Google, Meta, or NovaTech?",
        [context],
    ) is None
    assert context.casefold_calls == 1


def test_personal_current_age_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Personal age extraction should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("personal age parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._personal_current_age_values(
        [
            "I am 32 years old.",
            "I just turned 33.",
            "my age is 34",
        ]
    ) == [32, 33, 34]


def test_elapsed_year_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Elapsed-year extraction should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("elapsed year parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._elapsed_year_values(
        [
            "I started piano lessons five years ago.",
            "I have spent the past 3 years training.",
        ]
    ) == [5, 3]


def test_age_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Average-age extraction should not compile fixed regexes per context."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("age parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "finditer", fail)

    assert retrieval_plan._age_values(
        [
            "I just turned 32.",
            "mom is 55 and dad is 58.",
        ]
    ) == [32, 55, 58]


def test_unit_values_caches_parameterized_regex(monkeypatch) -> None:
    """Parameterized unit extraction should compile once per unit pattern."""
    compile_calls = 0
    original_compile = retrieval_plan.re.compile

    def tracking_compile(*args, **kwargs):  # noqa: ANN001
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(retrieval_plan.re, "compile", tracking_compile)

    assert retrieval_plan._unit_values(["I spent 3 hours there."], unit_pattern=r"hours?") == [3.0]
    assert retrieval_plan._unit_values(["I spent 4 hours there."], unit_pattern=r"hours?") == [4.0]
    assert compile_calls == 1


def test_week_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Week duration parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("week duration parsing should use compiled regex helpers")

    retrieval_plan._unit_value_pattern.cache_clear()
    retrieval_plan._unit_value_pattern(r"weeks?")
    monkeypatch.setattr(retrieval_plan.re, "compile", fail)
    monkeypatch.setattr(retrieval_plan.re, "search", fail)

    assert retrieval_plan._week_values(["I trained for two weeks.", "I went last weekend."]) == [2.0, 1.0]


def test_month_values_use_compiled_regex_helpers(monkeypatch) -> None:
    """Month duration parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("month duration parsing should use compiled regex helpers")

    retrieval_plan._unit_value_pattern.cache_clear()
    retrieval_plan._unit_value_pattern(r"months?")
    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._month_values(["I trained for two months."]) == [2.0]


def test_clock_time_values_use_compiled_regex_helper(monkeypatch) -> None:
    """Clock time parsing should not compile a fixed regex per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("clock time parsing should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._clock_time_values(["I woke at 6:30 AM and slept at 10 PM."]) == [
        390,
        1320,
    ]


def test_relative_minute_offsets_use_compiled_regex_helper(monkeypatch) -> None:
    """Relative minute parsing should not compile a fixed regex per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("relative minute parsing should use a compiled regex helper")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._relative_minute_offsets(
        ["I woke 15 minutes earlier and went out 20 minutes later."]
    ) == [-15, 20]


def test_relative_days_ago_uses_compiled_regex_helpers(monkeypatch) -> None:
    """Relative-day parsing should not compile fixed regexes per call."""

    def fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("relative-day parsing should use compiled regex helpers")

    monkeypatch.setattr(retrieval_plan.re, "compile", fail)

    assert retrieval_plan._relative_days_ago("I met Tom about two weeks ago.") == 14


def test_source_evidence_score_reuses_query_synthesis_plan(monkeypatch) -> None:
    """Ranking should not rebuild the same query plan through each helper."""
    plan_calls = 0
    plan = SynthesisPlan(
        answer_type="sum",
        operation="sum_values",
        subject_terms=("bike",),
        required_kinds=("currency",),
        required_source_groups=2,
        reasons=("money",),
    )

    def fake_plan(query: str) -> SynthesisPlan:
        del query
        nonlocal plan_calls
        plan_calls += 1
        return plan

    monkeypatch.setattr(retrieval_plan, "build_synthesis_plan", fake_plan)
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", lambda query, context: 7)

    score = retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert score >= 7
    assert plan_calls == 1


def test_source_evidence_score_reuses_query_tokens_across_helpers(monkeypatch) -> None:
    """Ranking should not retokenize the same query in every helper branch."""
    query = "How much total money have I spent on bike-related expenses?"
    query_token_calls = 0
    original_source_tokens = retrieval_plan.source_tokens

    def tracking_source_tokens(text: str) -> list[str]:
        nonlocal query_token_calls
        if text == query:
            query_token_calls += 1
        return original_source_tokens(text)

    monkeypatch.setattr(retrieval_plan, "source_tokens", tracking_source_tokens)

    score = retrieval_plan.source_evidence_score(
        query,
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert score > 0
    assert query_token_calls <= 8


def test_source_evidence_score_cache_reuses_query_state_across_contexts(monkeypatch) -> None:
    """One source-ranking pass should not rebuild query state per context."""
    query = "How much total money have I spent on bike-related expenses?"
    query_token_calls = 0
    plan_calls = 0
    original_source_tokens = retrieval_plan.source_tokens
    plan = SynthesisPlan(
        answer_type="sum",
        operation="sum_values",
        subject_terms=("bike",),
        required_kinds=("currency",),
        required_source_groups=2,
        reasons=("money",),
    )

    def tracking_source_tokens(text: str) -> list[str]:
        nonlocal query_token_calls
        if text == query:
            query_token_calls += 1
        return original_source_tokens(text)

    def tracking_plan(text: str) -> SynthesisPlan:
        del text
        nonlocal plan_calls
        plan_calls += 1
        return plan

    monkeypatch.setattr(retrieval_plan, "source_tokens", tracking_source_tokens)
    monkeypatch.setattr(retrieval_plan, "build_synthesis_plan", tracking_plan)
    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", lambda query, context: 0)

    retrieval_plan.evidence_source_order(
        query,
        [
            f"source_path=doc-{index}.md I bought bike gear for ${index}."
            for index in range(10)
        ],
        score_cache=retrieval_plan._SourceEvidenceScoreCache(query=query, scores={}),
    )

    assert query_token_calls <= 2
    assert plan_calls == 1


def test_aggregate_evidence_score_builds_only_required_currency_ledger(monkeypatch) -> None:
    """Currency query ranking should not build unrelated count/duration/date ledgers."""
    calls: list[str] = []
    plan = SynthesisPlan(
        answer_type="sum",
        operation="sum_values",
        subject_terms=("bike",),
        required_kinds=("currency",),
        required_source_groups=2,
        reasons=("money",),
    )

    def empty_ledger(name: str):
        def build(query: str, contexts: list[str], **kwargs: object) -> EvidenceLedger:
            del query, contexts, kwargs
            calls.append(name)
            return EvidenceLedger(plan=plan, rows=())

        return build

    monkeypatch.setattr(evidence_candidates, "build_count_ledger", empty_ledger("count"))
    monkeypatch.setattr(evidence_candidates, "build_currency_ledger", empty_ledger("currency"))
    monkeypatch.setattr(evidence_candidates, "build_duration_ledger", empty_ledger("duration"))
    monkeypatch.setattr(evidence_candidates, "build_date_ledger", empty_ledger("date"))

    evidence_candidates.aggregate_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert calls == ["currency"]


def test_aggregate_evidence_score_reuses_plan_in_required_ledger(monkeypatch) -> None:
    """Single-source ranking should not rebuild the same synthesis plan inside ledgers."""
    plan_calls = 0
    plan = SynthesisPlan(
        answer_type="sum",
        operation="sum_values",
        subject_terms=("bike",),
        required_kinds=("currency",),
        required_source_groups=2,
        reasons=("money",),
    )

    def fake_plan(query: str) -> SynthesisPlan:
        del query
        nonlocal plan_calls
        plan_calls += 1
        return plan

    monkeypatch.setattr(evidence_candidates, "build_synthesis_plan", fake_plan)
    monkeypatch.setattr(synthesis, "build_synthesis_plan", fake_plan)

    score = evidence_candidates.aggregate_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        "longmemeval_session_id=answer-bike I bought bike lights for $40.",
    )

    assert score > 0
    assert plan_calls == 1


def test_source_evidence_score_skips_irrelevant_currency_documents(monkeypatch) -> None:
    """Ranking should avoid expensive currency parsing for unrelated money chunks."""
    score_calls = 0

    def fake_score(query: str, context: str) -> int:
        del query, context
        nonlocal score_calls
        score_calls += 1
        return 99

    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", fake_score)

    score = retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        (
            "longmemeval_session_id=answer-noise "
            "assistant: You might consider a budget of $1,000 for flights, "
            "$300 for lodging, and $200 for meals."
        ),
    )

    assert score_calls == 0
    assert score == 0


def test_source_evidence_score_skips_currency_ledger_when_context_has_no_amount(monkeypatch) -> None:
    """Currency ranking should not build a ledger for focused contexts with no money evidence."""
    score_calls = 0

    def fake_score(query: str, context: str) -> int:
        del query, context
        nonlocal score_calls
        score_calls += 1
        return 7

    monkeypatch.setattr(retrieval_plan, "aggregate_evidence_score", fake_score)

    score = retrieval_plan.source_evidence_score(
        "How much total money have I spent on bike-related expenses?",
        (
            "longmemeval_session_id=answer-bike role=user "
            "I cleaned my bike and checked the chain before the weekend ride."
        ),
    )

    assert score_calls == 0
    assert score >= 0


def test_luxury_currency_ledger_keeps_sentence_label_before_amount() -> None:
    """Currency filtering should keep labelled luxury purchases with amount-only clauses."""
    ledger = build_currency_ledger(
        "What is the total amount I spent on luxury items in the past few months?",
        [
            (
                "longmemeval_session_id=answer-gown role=user "
                "I recently bought a luxury evening gown for a wedding. "
                "It was a big purchase, $800, but I felt like I needed it."
            )
        ],
    )

    included = ledger.included(kind="currency")

    assert len(included) == 1
    assert included[0].value == "800.0"
    assert included[0].label == "luxury evening gown"


def test_luxury_currency_filter_uses_recovered_purchase_label() -> None:
    """Focused currency filtering should consider labels recovered from prior sentences."""
    ledger = build_currency_ledger(
        "What is the total amount I spent on luxury items in the past few months?",
        [
            (
                "longmemeval_session_id=answer-bag role=user "
                "I got a luxury designer handbag from Gucci for $1,200."
            ),
            (
                "longmemeval_session_id=answer-gown role=user "
                "I recently bought a luxury evening gown for a wedding. "
                "It was a big purchase, $800, but I felt like I needed it."
            ),
            (
                "longmemeval_session_id=answer-shirts role=user "
                "I recently bought a pack of graphic tees from H&M for $20."
            ),
        ],
    )

    included_values = {row.value for row in ledger.included(kind="currency")}
    excluded_values = {
        row.value: row.exclude_reason
        for row in ledger.excluded(kind="currency")
    }

    assert included_values == {"1200.0", "800.0"}
    assert excluded_values["20.0"] == "query_focus_mismatch"
