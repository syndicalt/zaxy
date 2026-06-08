"""Tests for procedural planning context classification."""

from __future__ import annotations

from zaxy.context import Context
from zaxy.procedural_planning import classify_procedure_contexts


def _procedure_context(
    *,
    skill_id: str,
    status: str,
    citation: str | None = "eventloom://agent-1/events/7#aaaaaaaaaaaa",
    score: float = 0.9,
    valid_to: str | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> Context:
    metadata: dict[str, object] = {
        "entity_type": "skill_version",
        "skill_id": skill_id,
        "version": "2",
        "status": status,
        "procedure": ["Run the migration smoke test.", "Verify cited checkout output."],
        "applicability": ["release validation"],
        "summary": "Release validation procedure",
        "rollback": "Return to v1 if validation regresses.",
        "failure_modes": ["misses cross-session citation regression"],
        "contradiction_reason": "Superseded by validation harness change.",
    }
    if citation is not None:
        metadata["citation"] = citation
    if extra_metadata:
        metadata.update(extra_metadata)
    return Context(
        content="Procedure: validate release with cited memory checkout evidence.",
        source="skill_memory",
        score=score,
        valid_to=valid_to,
        metadata=metadata,
    )


def test_classify_procedure_contexts_returns_cited_open_validated_rows_as_applicable() -> None:
    contexts = [
        _procedure_context(skill_id="release-validation", status="validated", score=0.97),
        _procedure_context(
            skill_id="draft-validation",
            status="proposed",
            citation="eventloom://agent-1/events/8#bbbbbbbbbbbb",
        ),
    ]

    classified = classify_procedure_contexts(contexts, limit=5)

    assert classified["applicable"] == [
        {
            "content": contexts[0].content,
            "source": "skill_memory",
            "score": 0.97,
            "citation": "eventloom://agent-1/events/7#aaaaaaaaaaaa",
            "status": "validated",
            "metadata": contexts[0].metadata,
            "skill_id": "release-validation",
            "version": "2",
            "procedure": [
                "Run the migration smoke test.",
                "Verify cited checkout output.",
            ],
            "applicability": ["release validation"],
            "summary": "Release validation procedure",
            "rollback": "Return to v1 if validation regresses.",
            "failure_modes": ["misses cross-session citation regression"],
            "contradiction_reason": "Superseded by validation harness change.",
            "valid_from": None,
            "valid_to": None,
        }
    ]
    assert classified["diagnostic"][0]["skill_id"] == "draft-validation"
    assert classified["procedural_memory"]["applicable_count"] == 1
    assert classified["procedural_memory"]["diagnostic_count"] == 1
    assert classified["procedural_memory"]["excluded_count"] == 0


def test_classify_procedure_contexts_keeps_diagnostic_statuses_out_of_applicable() -> None:
    contexts = [
        _procedure_context(skill_id="draft", status="proposed"),
        _procedure_context(skill_id="waiting-review", status="pending"),
        _procedure_context(skill_id="postponed", status="deferred"),
    ]

    classified = classify_procedure_contexts(contexts, limit=5)

    assert classified["applicable"] == []
    assert [item["status"] for item in classified["diagnostic"]] == [
        "proposed",
        "pending",
        "deferred",
    ]
    assert all(item["operational_instruction"] is False for item in classified["diagnostic"])


def test_classify_procedure_contexts_excludes_unsafe_or_uncited_rows_with_reasons() -> None:
    contexts = [
        _procedure_context(skill_id="rejected", status="rejected"),
        _procedure_context(skill_id="conflicted", status="conflicted"),
        _procedure_context(skill_id="deprecated", status="deprecated"),
        _procedure_context(skill_id="contradicted", status="contradicted"),
        _procedure_context(skill_id="stale-status", status="stale"),
        _procedure_context(skill_id="stale-flag", status="validated", extra_metadata={"stale": True}),
        _procedure_context(skill_id="uncited", status="validated", citation=None),
        _procedure_context(
            skill_id="closed-validity",
            status="accepted",
            valid_to="2026-06-07T00:00:00Z",
        ),
        _procedure_context(
            skill_id="superseded",
            status="revised",
            extra_metadata={"superseded_by": "skill:superseded:v3"},
        ),
    ]

    classified = classify_procedure_contexts(contexts, limit=20)

    assert classified["applicable"] == []
    assert classified["diagnostic"] == []
    assert [item["skill_id"] for item in classified["excluded"]] == [
        "rejected",
        "conflicted",
        "deprecated",
        "contradicted",
        "stale-status",
        "stale-flag",
        "uncited",
        "closed-validity",
        "superseded",
    ]
    assert classified["excluded_reasons"] == {
        "rejected_status": 1,
        "conflicted_status": 1,
        "deprecated_status": 1,
        "contradicted_status": 1,
        "stale_status": 1,
        "stale_flag": 1,
        "missing_citation": 1,
        "valid_to_closed": 1,
        "superseded": 1,
    }


def test_classify_procedure_contexts_limits_applicable_without_hiding_diagnostics() -> None:
    contexts = [
        _procedure_context(
            skill_id="accepted",
            status="accepted",
            citation="eventloom://agent-1/events/1#aaaaaaaaaaaa",
        ),
        _procedure_context(
            skill_id="revised",
            status="revised",
            citation="eventloom://agent-1/events/2#bbbbbbbbbbbb",
        ),
        _procedure_context(
            skill_id="validated",
            status="validated",
            citation="eventloom://agent-1/events/3#cccccccccccc",
        ),
        _procedure_context(
            skill_id="diagnostic",
            status="pending",
            citation="eventloom://agent-1/events/4#dddddddddddd",
        ),
    ]

    classified = classify_procedure_contexts(contexts, limit=2)

    assert [item["skill_id"] for item in classified["applicable"]] == ["accepted", "revised"]
    assert [item["skill_id"] for item in classified["diagnostic"]] == ["diagnostic"]
    assert classified["procedural_memory"]["applicable_count"] == 2
    assert classified["procedural_memory"]["available_applicable_count"] == 3
