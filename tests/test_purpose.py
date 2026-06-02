from zaxy.purpose import purpose_ontology_lens, purpose_profile, purpose_retrieval_policy


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


def test_broader_purpose_profiles_are_local_project_memory_policies() -> None:
    expected = {
        "support": ("triage_escalate_or_apply_workaround", "customer_impact", "workaround_history"),
        "product": ("prioritize_or_defer_with_evidence", "roadmap_signal", "experiment_outcomes"),
        "sales": ("prepare_commitment_or_followup", "buyer_commitment", "renewal_blockers"),
        "legal": ("quote_obligation_or_require_review", "legal_obligation", "deadlines"),
        "executive": ("summarize_risk_exception_or_pattern", "strategic_exception", "risk_summaries"),
    }

    for name, (expected_action, lens_term, retain_term) in expected.items():
        profile = purpose_profile(name)
        lens = purpose_ontology_lens(profile)
        policy = purpose_retrieval_policy(profile, "current work", prompt_limit=4, base_recall_limit=8)

        assert profile.permission_scope == "project-local"
        assert profile.expected_action == expected_action
        assert lens_term in profile.ontology_lens
        assert retain_term in profile.retain
        assert profile.required_evidence
        assert profile.suppress
        assert lens.applied is True
        assert lens.required_source_groups == profile.required_evidence
        assert policy.applied is True
        assert policy.min_recall_limit >= 20


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


def test_purpose_ontology_lens_maps_same_evidence_to_distinct_roles() -> None:
    text = (
        "Accepted auth release gate found credential exposure with failing test evidence "
        "and mitigation owner review."
    )

    coding = purpose_ontology_lens("coding")
    security = purpose_ontology_lens("security")
    release = purpose_ontology_lens("release")
    coordinate = purpose_ontology_lens("coordinate")

    assert "test" in coding.matched_entity_roles(text)
    assert "credential" in security.matched_entity_roles(text)
    assert "release_gate" in release.matched_entity_roles(text)
    assert "accepted_state" in coordinate.entity_roles
    assert security.path_multiplier(["exposes_secret"]) > release.path_multiplier(["exposes_secret"])
    assert coordinate.path_multiplier(["mission_has_proof_packet"]) > coding.path_multiplier(
        ["mission_has_proof_packet"]
    )
    assert security.to_diagnostics()["edge_trust_multipliers"]["exposes_secret"] == 1.6
