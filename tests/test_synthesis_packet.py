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
