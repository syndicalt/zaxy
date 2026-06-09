"""Tests for typed synthesis packet extraction."""

from __future__ import annotations

from zaxy.synthesis_packet import synthesis_packet_from_diagnostics, synthesis_packet_from_items


def test_synthesis_packet_parses_candidates_and_ledger_rows_once() -> None:
    """Rendered synthesis bundles should become one typed packet contract."""
    items = [
        {
            "content": "\n".join(
                [
                    "zaxy_synthesis_bundle=true",
                    "candidate_rank=1 candidate_type=currency candidate_confidence=0.83",
                    "currency_total_answer=$185",
                    "candidate_support=answer-1,answer-2,answer-3",
                    "currency_excluded_source_ids=answer-4",
                    (
                        'ledger_row={"fact_id":"currency:0:0","source_group":"answer-1",'
                        '"citation":"eventloom://agent-1/events/1#aaaaaaaaaaaa",'
                        '"kind":"currency","value":"120","unit":"USD","label":"helmet",'
                        '"raw_span":"$120 helmet","normalized_identity":"currency:helmet:120",'
                        '"include_reason":"currency_amount","exclude_reason":"","confidence":0.83}'
                    ),
                    (
                        'ledger_row={"fact_id":"currency:0:0","source_group":"answer-1",'
                        '"citation":"duplicate","kind":"currency","value":"120"}'
                    ),
                ]
            )
        },
        {
            "content": "\n".join(
                [
                    "zaxy_synthesis_bundle=true",
                    "candidate_rank=1 candidate_type=currency candidate_confidence=0.83",
                    "currency_total_answer=$185",
                    "candidate_support=answer-1,answer-2,answer-3",
                ]
            )
        },
    ]

    packet = synthesis_packet_from_items(items)

    assert packet.answer_candidates == [
        {
            "rank": 1,
            "type": "currency",
            "confidence": 0.83,
            "answer_key": "currency_total_answer",
            "answer": "$185",
            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
            "excluded_source_ids": ["answer-4"],
        }
    ]
    assert packet.ledger_rows == [
        {
            "fact_id": "currency:0:0",
            "source_group": "answer-1",
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "kind": "currency",
            "value": "120",
            "unit": "USD",
            "label": "helmet",
            "raw_span": "$120 helmet",
            "normalized_identity": "currency:helmet:120",
            "include_reason": "currency_amount",
            "exclude_reason": "",
            "confidence": 0.83,
        }
    ]


def test_synthesis_packet_prefers_rendered_answer_text_surface() -> None:
    """Fallback packet parsing should promote answer-ready text over bare scalars."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "candidate_rank=1 candidate_type=count candidate_confidence=0.82",
                        "candidate_support=answer-1,answer-2,answer-3",
                        "count_answer=4",
                        "count_unit=events",
                        "count_answer_text=I attended four movie festivals.",
                    ]
                )
            }
        ]
    )

    assert packet.answer_candidates == [
        {
            "rank": 1,
            "type": "count",
            "confidence": 0.82,
            "answer_key": "count_answer_text",
            "answer": "I attended four movie festivals.",
            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
            "excluded_source_ids": [],
        }
    ]


def test_synthesis_packet_prefers_direct_interval_over_relative_elapsed_answer() -> None:
    """Before/since interval answers should outrank generic elapsed-time answers."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "candidate_rank=1 candidate_type=relative_week_interval candidate_confidence=0.0",
                        "week_interval_answer=Two weeks",
                        "relative_week_interval_answer=Three week",
                    ]
                )
            }
        ]
    )

    assert packet.answer_candidates[0]["answer_key"] == "week_interval_answer"
    assert packet.answer_candidates[0]["answer"] == "Two weeks"


def test_synthesis_packet_prefers_total_answer_over_auxiliary_difference() -> None:
    """Aggregate bundle parsing should expose requested totals before diagnostics."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "candidate_rank=1 candidate_type=currency candidate_confidence=0.73",
                        "candidate_support=answer-1,answer-2,answer-3",
                        "currency_values=$500,$200,$20",
                        "currency_total_answer=$720",
                        "currency_difference_answer=$480",
                    ]
                )
            }
        ]
    )

    assert packet.answer_candidates[0]["answer_key"] == "currency_total_answer"
    assert packet.answer_candidates[0]["answer"] == "$720"


def test_synthesis_packet_prefers_total_answer_for_combined_duration_query() -> None:
    """Combined-duration queries should expose total answers before interval diagnostics."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "query=How long did I take to finish two books combined?",
                        "candidate_rank=1 candidate_type=duration candidate_confidence=0.81",
                        "candidate_support=book-1,book-2",
                        "duration_values=2.5 weeks,3 weeks",
                        "duration_total_answer=5.5 weeks",
                        "week_interval_answer=Three weeks",
                    ]
                )
            }
        ]
    )

    assert packet.answer_candidates[0]["answer_key"] == "duration_total_answer"
    assert packet.answer_candidates[0]["answer"] == "5.5 weeks"


def test_synthesis_packet_prefers_specialized_total_answer_over_generic_duration() -> None:
    """Specialized deterministic totals should not be shadowed by generic duration fields."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "query=How many days did I take social media breaks in total?",
                        "candidate_rank=1 candidate_type=social_media_break candidate_confidence=0.84",
                        "candidate_support=break-1,break-2",
                        "social_media_break_total=17 days",
                        "social_media_break_total_answer=17 days",
                        "duration_total_answer=19 days",
                    ]
                )
            }
        ]
    )

    assert packet.answer_candidates[0]["answer_key"] == "social_media_break_total_answer"
    assert packet.answer_candidates[0]["answer"] == "17 days"


def test_synthesis_packet_ignores_malformed_ledger_rows() -> None:
    """Bad rendered rows should not poison valid candidates."""
    packet = synthesis_packet_from_items([
        {
            "content": "\n".join(
                [
                    "zaxy_synthesis_bundle=true",
                    "candidate_rank=2 candidate_type=date candidate_confidence=0.7",
                    "date_interval_answer=14 days",
                    "ledger_row={not-json",
                ]
            )
        }
    ])

    assert packet.answer_candidates[0]["answer"] == "14 days"
    assert packet.ledger_rows == []


def test_synthesis_packet_defaults_malformed_candidate_fields() -> None:
    """Rendered candidate parsing should keep useful answers despite bad metadata."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "candidate_rank=not-a-number candidate_type=preference candidate_confidence=bad",
                        "candidate_support=answer-1,,answer-2",
                        "preference_answer=The user would prefer native-plant gardening ideas.",
                    ]
                )
            }
        ]
    )

    assert packet.answer_candidates == [
        {
            "rank": 1,
            "type": "preference",
            "confidence": 0.0,
            "answer_key": "preference_answer",
            "answer": "The user would prefer native-plant gardening ideas.",
            "support_source_ids": ["answer-1", "answer-2"],
            "excluded_source_ids": [],
        }
    ]


def test_synthesis_packet_prefers_typed_candidates_and_merges_rendered_ledger_rows() -> None:
    """Typed candidates are authoritative while legacy rendered rows remain auditable."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "candidate_rank=1 candidate_type=currency candidate_confidence=0.10",
                        "currency_total_answer=$1",
                        'ledger_row={"fact_id":"rendered:1","source_group":"answer-2","citation":"eventloom://agent/events/2#bbbbbbbbbbbb","kind":"currency","value":"25","include_reason":"currency_amount"}',
                    ]
                ),
                "synthesis_packet": {
                    "schema_version": "synthesis_packet_v1",
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "currency",
                            "confidence": 0.91,
                            "answer_key": "currency_total_answer",
                            "answer": "$145",
                            "support_source_ids": ["answer-1", "answer-2"],
                            "excluded_source_ids": [],
                        }
                    ],
                    "ledger_rows": [
                        {
                            "fact_id": "typed:1",
                            "source_group": "answer-1",
                            "citation": "eventloom://agent/events/1#aaaaaaaaaaaa",
                            "kind": "currency",
                            "value": "120",
                            "include_reason": "currency_amount",
                        }
                    ],
                },
            }
        ]
    )

    assert packet.answer_candidates == [
        {
            "rank": 1,
            "type": "currency",
            "confidence": 0.91,
            "answer_key": "currency_total_answer",
            "answer": "$145",
            "support_source_ids": ["answer-1", "answer-2"],
            "excluded_source_ids": [],
        }
    ]
    assert [row["fact_id"] for row in packet.ledger_rows] == ["typed:1", "rendered:1"]


def test_synthesis_packet_keeps_additive_rendered_answer_candidate() -> None:
    """Rendered answer surfaces should survive when typed packets omit that type."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "query=How many tops have I bought from H&M so far?",
                        "candidate_rank=1 candidate_type=direct_numeric_value candidate_confidence=0.84",
                        "candidate_support=answer-1",
                        "direct_numeric_answer=five",
                    ]
                ),
                "synthesis_packet": {
                    "schema_version": "synthesis_packet_v1",
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "count",
                            "confidence": 0.72,
                            "answer_key": "count_answer",
                            "answer": "3",
                            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
                            "excluded_source_ids": [],
                        }
                    ],
                    "ledger_rows": [],
                },
            }
        ]
    )

    assert packet.answer_candidates == [
        {
            "rank": 1,
            "type": "count",
            "confidence": 0.72,
            "answer_key": "count_answer",
            "answer": "3",
            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
            "excluded_source_ids": [],
        },
        {
            "rank": 1,
            "type": "direct_numeric_value",
            "confidence": 0.84,
            "answer_key": "direct_numeric_answer",
            "answer": "five",
            "support_source_ids": ["answer-1"],
            "excluded_source_ids": [],
        },
    ]


def test_synthesis_packet_does_not_promote_same_block_diagnostics_over_typed_candidate() -> None:
    """Rendered diagnostic answers should not compete with a typed candidate block."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "query=How many months ago did I reserve the cabin?",
                        "candidate_rank=1 candidate_type=duration candidate_confidence=0.75",
                        "candidate_support=booking,retreat",
                        "duration_total_answer=Five months ago",
                        "month_interval_answer=One months",
                    ]
                ),
                "synthesis_packet": {
                    "schema_version": "synthesis_packet_v1",
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "duration",
                            "confidence": 0.75,
                            "answer_key": "duration_total_answer",
                            "answer": "Five months ago",
                            "support_source_ids": ["booking", "retreat"],
                            "excluded_source_ids": [],
                        }
                    ],
                    "ledger_rows": [],
                },
            }
        ]
    )

    assert packet.answer_candidates == [
        {
            "rank": 1,
            "type": "duration",
            "confidence": 0.75,
            "answer_key": "duration_total_answer",
            "answer": "Five months ago",
            "support_source_ids": ["booking", "retreat"],
            "excluded_source_ids": [],
        }
    ]


def test_synthesis_packet_prefers_elapsed_month_total_over_interval_diagnostic() -> None:
    """How-many-ago fallback parsing should use aggregate elapsed totals."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "query=How many months ago did I book the Airbnb in San Francisco?",
                        "month_values=3,2",
                        "month_total=5 months ago",
                        "month_total_words=Five months ago",
                        "month_interval=1 months",
                        "month_interval_answer=One months",
                    ]
                )
            }
        ]
    )

    assert packet.answer_candidates == [
        {
            "rank": 1,
            "type": "duration",
            "confidence": 0.0,
            "answer_key": "month_total_words",
            "answer": "Five months ago",
            "support_source_ids": [],
            "excluded_source_ids": [],
        }
    ]


def test_synthesis_packet_does_not_promote_broad_direct_numeric_fallback() -> None:
    """Broad aggregate calculations should not gain competing direct scalar fallbacks."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "query=What is the total number of comments on my Facebook Live session and YouTube video?",
                        "candidate_rank=1 candidate_type=direct_numeric_value candidate_confidence=0.84",
                        "candidate_support=answer-1",
                        "direct_numeric_answer=21",
                    ]
                ),
                "synthesis_packet": {
                    "schema_version": "synthesis_packet_v1",
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "count",
                            "confidence": 0.72,
                            "answer_key": "count_answer",
                            "answer": "33",
                            "support_source_ids": ["answer-1", "answer-2"],
                            "excluded_source_ids": [],
                        }
                    ],
                    "ledger_rows": [],
                },
            }
        ]
    )

    assert packet.answer_candidates == [
        {
            "rank": 1,
            "type": "count",
            "confidence": 0.72,
            "answer_key": "count_answer",
            "answer": "33",
            "support_source_ids": ["answer-1", "answer-2"],
            "excluded_source_ids": [],
        }
    ]


def test_synthesis_packet_preserves_operation_and_result_metadata() -> None:
    """Typed operation/result metadata should survive normalization as additive packet data."""
    packet = synthesis_packet_from_items(
        [
            {
                "synthesis_packet": {
                    "schema_version": "synthesis_packet_v1",
                    "operations": [
                        {
                            "name": "sum_values",
                            "kind": "currency",
                            "answer_key": "currency_total_answer",
                            "support_source_ids": ["answer-1", "answer-2"],
                        }
                    ],
                    "result": {
                        "answer_key": "currency_total_answer",
                        "answer": "$145",
                        "confidence": 0.91,
                    },
                    "answer_candidates": [
                        {
                            "rank": 1,
                            "type": "currency",
                            "confidence": 0.91,
                            "answer_key": "currency_total_answer",
                            "answer": "$145",
                            "support_source_ids": ["answer-1", "answer-2"],
                            "excluded_source_ids": [],
                        }
                    ],
                    "ledger_rows": [],
                }
            }
        ]
    )

    assert packet.operations == [
        {
            "name": "sum_values",
            "kind": "currency",
            "answer_key": "currency_total_answer",
            "support_source_ids": ["answer-1", "answer-2"],
        }
    ]
    assert packet.result == {
        "answer_key": "currency_total_answer",
        "answer": "$145",
        "confidence": 0.91,
    }


def test_synthesis_packet_from_diagnostics_preserves_operation_and_result_metadata() -> None:
    """Diagnostics normalization should preserve flat packet operation metadata."""
    packet = synthesis_packet_from_diagnostics(
        {
            "synthesis": {
                "operations": [
                    {
                        "name": "average_values",
                        "kind": "number",
                        "answer_key": "age_average",
                    }
                ],
                "result": {
                    "answer_key": "age_average",
                    "answer": "59.6",
                    "confidence": 0.89,
                },
                "answer_candidates": [
                    {
                        "rank": 1,
                        "type": "number",
                        "confidence": 0.89,
                        "answer_key": "age_average",
                        "answer": "59.6",
                        "support_source_ids": ["answer-1"],
                        "excluded_source_ids": [],
                    }
                ],
                "ledger_rows": [],
            }
        }
    )

    assert packet.operations == [
        {
            "name": "average_values",
            "kind": "number",
            "answer_key": "age_average",
        }
    ]
    assert packet.result["answer"] == "59.6"


def test_synthesis_packet_defaults_operations_and_result_for_legacy_packets() -> None:
    """Legacy rendered packets should get empty additive operation/result fields."""
    packet = synthesis_packet_from_items(
        [
            {
                "content": "\n".join(
                    [
                        "zaxy_synthesis_bundle=true",
                        "candidate_rank=1 candidate_type=count candidate_confidence=0.75",
                        "count_answer=2",
                    ]
                )
            }
        ]
    )

    assert packet.answer_candidates[0]["answer"] == "2"
    assert packet.operations == []
    assert packet.result == {}


def test_synthesis_packet_drops_non_json_operation_result_values() -> None:
    """Operation/result metadata should remain JSON-safe after normalization."""
    packet = synthesis_packet_from_items(
        [
            {
                "synthesis_packet": {
                    "operations": [
                        {
                            "name": "sum_values",
                            "bad": object(),
                            "nested": {"kept": "yes", "bad": object()},
                        }
                    ],
                    "result": {
                        "answer": "$145",
                        "bad": object(),
                        "nested": {"kept": True, "bad": object()},
                    },
                    "answer_candidates": [],
                    "ledger_rows": [],
                }
            }
        ]
    )

    assert packet.operations == [{"name": "sum_values", "nested": {"kept": "yes"}}]
    assert packet.result == {"answer": "$145", "nested": {"kept": True}}
