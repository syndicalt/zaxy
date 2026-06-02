from zaxy.purpose import purpose_profile, purpose_retrieval_policy


def test_release_security_and_research_purpose_presets_are_first_class() -> None:
    release = purpose_profile("release")
    security = purpose_profile("security")
    research = purpose_profile("research")

    assert release.expected_action == "ship_block_or_defer"
    assert "release_gate" in release.ontology_lens
    assert "external_blockers" in release.retain
    assert security.risk == "critical"
    assert "credential" in security.ontology_lens
    assert "risk_acceptance" in security.retain
    assert research.expected_action == "synthesize_or_identify_gap"
    assert "contradiction" in research.ontology_lens
    assert "open_questions" in research.retain


def test_purpose_retrieval_policy_applies_profile_specific_terms_and_recall_floor() -> None:
    policy = purpose_retrieval_policy(
        purpose_profile("security"),
        "review auth changes",
        prompt_limit=4,
        base_recall_limit=8,
    )

    assert policy.applied is True
    assert policy.min_recall_limit == 24
    assert policy.recall_multiplier == 4
    assert policy.retrieval_query.startswith("review auth changes purpose:security")
    assert "credential" in policy.emphasis_terms
    assert "risk_acceptance" in policy.emphasis_terms
    assert policy.to_diagnostics(base_recall_limit=8, resolved_recall_limit=24) == {
        "profile": "security",
        "applied": True,
        "emphasis_terms": list(policy.emphasis_terms),
        "scoring_profile": "precision",
        "recall_multiplier": 4,
        "min_recall_limit": 24,
        "base_recall_limit": 8,
        "resolved_recall_limit": 24,
    }


def test_general_purpose_retrieval_policy_is_noop() -> None:
    policy = purpose_retrieval_policy(
        purpose_profile(),
        "current task",
        prompt_limit=4,
        base_recall_limit=8,
    )

    assert policy.applied is False
    assert policy.retrieval_query == "current task"
    assert policy.emphasis_terms == ()
    assert policy.scoring_profile == "balanced"
    assert policy.min_recall_limit == 0
