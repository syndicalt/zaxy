"""Memory Checkout builder and its ranking/purpose/skill/evidence helpers."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zaxy.checkout import (
    build_checkout_diagnostics,
    build_checkout_guidance,
    build_checkout_quality,
    build_compact_answer_contexts,
    format_memory_checkout_prompt,
)
from zaxy.context import Context
from zaxy.core.models import (
    ContextAssembly,
    MemoryCheckout,
)
from zaxy.event import (  # noqa: F401 - ReplayResult re-export for existing tests
    EventLog,
    IntegrityReport,
    ReplayResult,
    verify_event_chain,
)
from zaxy.evidence import select_checkout_evidence
from zaxy.purpose import PurposeProfile, purpose_profile
from zaxy.refs import MemoryRef
from zaxy.retrieval_plan import (
    build_evidence_plan,
    source_context_group,
    source_lane_candidate_limit,
    source_synthesis_bundle_result,
)
from zaxy.retrieval_profile import (
    RetrievalProfile,
)
from zaxy.salience import (
    CUE_MATCH_WEIGHT,
    REINFORCEMENT_EVENT_TYPE,
    SALIENCE_BASE,
    SALIENCE_HALF_LIFE_DAYS,
    SALIENCE_MAX,
    SALIENCE_MIN,
    EventRef,
    SalienceLedger,
    SalienceState,
    cue_overlap,
    cue_pairs,
    event_ref_index,
    resolve_citation_target,
    target_ref,
)
from zaxy.security import (
    MAX_QUERY_LIMIT,
    validate_limit,
)
from zaxy.synthesis_packet import synthesis_packet_from_items

_ENCODING_GATE_SKIPPED_EVENT_TYPES = frozenset(
    {
        REINFORCEMENT_EVENT_TYPE,
        "belief.update.proposed",
        "reasoning.primitive.called",
        "inference.edge.generated",
    }
)


_ENCODING_ENTITY_NAME_KEYS = ("entity_name", "name", "taskId", "task_id", "task")


_INTERFERENCE_EXCLUDED_PROPERTY_KEYS = frozenset(
    {
        "summary",
        "embedding",
        "embedding_version",
        "created_at",
        "updated_at",
        "observed_at",
        "expires_at",
        "last_reinforced_at",
        "importance",
        "reinforcement_count",
        "retrieval_salience",
        "source_event_seq",
        "source_event_hash",
        "source_event_prev_hash",
        "source_thread",
        "node_key",
        "session_id",
    }
)


_ENCODING_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _encoding_gate_eligible(event_type: str, payload: dict[str, Any]) -> bool:
    """Return whether an append should be classified by the encoding gate."""
    if event_type in _ENCODING_GATE_SKIPPED_EVENT_TYPES:
        return False
    return "encoding" not in payload


_ENCODING_CONTENT_EXCLUDED_KEYS = frozenset({"encoding", "cues"})


def _encoding_classification_content(payload: dict[str, Any]) -> str:
    """Return the canonical payload text the gate compares against memory.

    Mirrors the verbatim index's event-chunk text (sorted-key JSON), with
    gate/cue metadata stripped so tagging an event never dilutes later
    duplicate detection against it.
    """
    comparable = {
        key: value
        for key, value in payload.items()
        if key not in _ENCODING_CONTENT_EXCLUDED_KEYS
    }
    if not comparable:
        return ""
    try:
        return json.dumps(comparable, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return ""


def _encoding_tokens(text: str) -> set[str]:
    return set(_ENCODING_TOKEN_RE.findall(text.casefold()))


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / (len(left) + len(right) - intersection)


def _payload_entity_names(payload: dict[str, Any]) -> list[str]:
    """Return bounded candidate entity names declared by a payload."""
    names: list[str] = []
    for key in _ENCODING_ENTITY_NAME_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and len(value) <= 200:
            name = value.strip()
            if name not in names:
                names.append(name)
    return names


def _conflicting_property_value(
    old_properties: dict[str, Any],
    new_properties: dict[str, Any],
) -> tuple[str, Any, Any] | None:
    """Return the first shared scalar property whose values conflict."""
    for key in sorted(set(old_properties) & set(new_properties)):
        if key in _INTERFERENCE_EXCLUDED_PROPERTY_KEYS or key.startswith("_"):
            continue
        old_value = old_properties[key]
        new_value = new_properties[key]
        if not _is_comparable_scalar(old_value) or not _is_comparable_scalar(new_value):
            continue
        old_text = str(old_value).strip().casefold()
        new_text = str(new_value).strip().casefold()
        if old_text and new_text and old_text != new_text:
            return (key, old_value, new_value)
    return None


def _is_comparable_scalar(value: Any) -> bool:
    return isinstance(value, str | int | float | bool)


def _event_content(event: Any) -> str:
    payload = getattr(event, "payload", {})
    if not isinstance(payload, dict):
        return ""
    parts = [
        str(payload[key])
        for key in ("title", "summary", "content", "text", "decision", "task")
        if payload.get(key)
    ]
    if not parts:
        parts = [f"{getattr(event, 'type', 'event')} by {getattr(event, 'actor', 'unknown')}"]
    return " ".join(parts)


def _event_citation(event: Any) -> str | None:
    thread = getattr(event, "thread", None)
    seq = getattr(event, "seq", None)
    event_hash = getattr(event, "hash", None)
    if not isinstance(thread, str) or not isinstance(seq, int) or not isinstance(event_hash, str):
        return None
    return f"eventloom://{thread}/events/{seq}#{event_hash[:12]}"


def _checkout_source_id(ref: MemoryRef | None, events: list[Any], *, session_id: str) -> str:
    """Return the stable as-of identity a checkout packet was computed against.

    A ref checkout is identified by the resolved ref target; a HEAD checkout
    by the citation of the last replayed event — both identities checkout
    already produces, derived from data in hand (no log read).
    """
    if ref is not None:
        return f"eventloom://{ref.session_id}/events/{ref.target_seq}#{ref.target_hash[:12]}"
    citation = _event_citation(events[-1]) if events else None
    return citation if citation is not None else f"{session_id}:HEAD"


def _invalidation_source_id(*, entity_name: str, entity_type: str, invalid_at: str) -> str:
    """Return the natural key of one invalidation operation."""
    return f"invalidate:{entity_type}:{entity_name}@{invalid_at}"


def entity_reinforcement_targets(entities: Any) -> list[dict[str, Any]]:
    """Return builder-ready reinforcement targets from projected entities.

    Reads the ``source_event_seq`` / ``source_event_hash`` provenance stored
    on projected entity properties; entities without sealed full-hash
    provenance are skipped, and duplicates collapse to one target.
    """
    targets: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for entity in entities or []:
        properties = getattr(entity, "properties", None)
        if not isinstance(properties, dict):
            continue
        target = target_ref(properties.get("source_event_seq"), properties.get("source_event_hash"))
        if target is None:
            continue
        key = (int(target["seq"]), str(target["hash"]))
        if key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return targets


def _packet_memory_reinforcements(events: list[Any]) -> dict[str, dict[str, float | int]]:
    reinforcements: dict[str, dict[str, float | int]] = {}
    for event in events:
        if getattr(event, "type", "") != "memory.reinforced":
            continue
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict) or payload.get("entity_type") != "packet_memory":
            continue
        source_event_hash = payload.get("source_event_hash")
        if not isinstance(source_event_hash, str) or not source_event_hash:
            continue
        existing = reinforcements.setdefault(
            source_event_hash,
            {"count": 0, "importance": 0.0},
        )
        existing["count"] = int(existing["count"]) + 1
        importance = payload.get("importance")
        if isinstance(importance, int | float) and not isinstance(importance, bool):
            existing["importance"] = max(float(existing["importance"]), float(importance))
    return reinforcements


_POSITIVE_PURPOSE_OUTCOMES = {
    "avoided_failed_path",
    "blocked_unsafe_action",
    "changed_answer",
    "helpful",
    "prevented_redundant_investigation",
    "resolved_conflict",
    "supported_handoff",
    "used",
}


_NEGATIVE_PURPOSE_OUTCOMES = {
    "caused_regression",
    "corrected",
    "excluded",
    "failed",
    "irrelevant",
    "rejected",
}


def _purpose_outcome_aggregates(
    events: list[Any],
    profile: PurposeProfile,
) -> dict[str, dict[str, Any]]:
    """Return replay-derived purpose outcome counts keyed by stable memory identity."""
    if profile.profile == "general":
        return {}
    aggregates: dict[str, dict[str, Any]] = {}
    for event in events:
        event_type = getattr(event, "type", "")
        if event_type not in {
            "memory.reinforced",
            "memory.feedback",
            "memory.evidence.reinforced",
            "memory.evidence.excluded",
        }:
            continue
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict) or not _purpose_payload_matches(payload, profile):
            continue
        outcome = _purpose_feedback_outcome(event_type, payload)
        if outcome is None:
            continue
        polarity = _purpose_outcome_polarity(outcome)
        if polarity is None:
            continue
        keys = _purpose_outcome_payload_keys(payload)
        if not keys:
            continue
        for key in keys:
            aggregate = aggregates.setdefault(
                key,
                {
                    "positive_count": 0,
                    "negative_count": 0,
                    "outcomes": [],
                    "latest_event_seq": None,
                },
            )
            count_key = "positive_count" if polarity == "positive" else "negative_count"
            aggregate[count_key] = int(aggregate[count_key]) + 1
            if outcome not in aggregate["outcomes"]:
                aggregate["outcomes"].append(outcome)
            seq = getattr(event, "seq", None)
            if isinstance(seq, int):
                aggregate["latest_event_seq"] = seq
    return aggregates


def _apply_purpose_outcome_learning(
    contexts: list[Context],
    aggregates: dict[str, dict[str, Any]],
) -> list[Context]:
    """Return contexts scored with bounded replay-derived outcome learning."""
    if not aggregates:
        return contexts
    learned: list[Context] = []
    for context in contexts:
        aggregate = _purpose_outcome_for_context(context, aggregates)
        if aggregate is None:
            learned.append(context)
            continue
        positive_count = int(aggregate.get("positive_count", 0))
        negative_count = int(aggregate.get("negative_count", 0))
        boost = min(0.20, positive_count * 0.06)
        penalty = min(0.18, negative_count * 0.06)
        score_multiplier = max(0.1, 1.0 + boost - penalty)
        metadata = dict(context.metadata or {})
        outcome_payload = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "score_boost": round(boost, 4),
            "score_penalty": round(penalty, 4),
            "outcomes": list(aggregate.get("outcomes", [])),
            "suppression_candidate": negative_count >= 2 and negative_count >= positive_count,
        }
        if aggregate.get("latest_event_seq") is not None:
            outcome_payload["latest_event_seq"] = aggregate["latest_event_seq"]
        metadata.update(
            {
                "purpose_outcome_positive_count": positive_count,
                "purpose_outcome_negative_count": negative_count,
                "purpose_outcome_score_boost": round(boost, 4),
                "purpose_outcome_score_penalty": round(penalty, 4),
                "purpose_outcome_suppression_candidate": outcome_payload["suppression_candidate"],
            }
        )
        score_explanation = dict(metadata.get("score_explanation") or {})
        score_explanation["purpose_outcome"] = outcome_payload
        metadata["score_explanation"] = score_explanation
        learned.append(
            replace(
                context,
                score=context.score * score_multiplier,
                metadata=metadata,
            )
        )
    return sorted(learned, key=lambda item: item.score, reverse=True)


def _purpose_payload_matches(payload: dict[str, Any], profile: PurposeProfile) -> bool:
    purpose = payload.get("purpose")
    value = purpose.get("profile") if isinstance(purpose, dict) else purpose
    if not isinstance(value, str) or not value.strip():
        return False
    return value.strip().casefold().replace(" ", "-") == profile.profile


def _purpose_feedback_outcome(event_type: str, payload: dict[str, Any]) -> str | None:
    outcome = payload.get("outcome")
    if isinstance(outcome, str) and outcome.strip():
        return outcome.strip().casefold().replace(" ", "_")
    feedback = payload.get("feedback")
    if isinstance(feedback, str) and feedback.strip():
        return feedback.strip().casefold().replace(" ", "_")
    if event_type in {"memory.reinforced", "memory.evidence.reinforced"}:
        return "used"
    if event_type == "memory.evidence.excluded":
        return "excluded"
    return None


def _purpose_outcome_polarity(outcome: str) -> str | None:
    if outcome in _POSITIVE_PURPOSE_OUTCOMES:
        return "positive"
    if outcome in _NEGATIVE_PURPOSE_OUTCOMES:
        return "negative"
    return None


def _purpose_outcome_payload_keys(payload: dict[str, Any]) -> list[str]:
    citation = payload.get("citation")
    if isinstance(citation, str) and citation.strip():
        return [f"citation:{citation.strip()}"]
    source_event_hash = payload.get("source_event_hash")
    if isinstance(source_event_hash, str) and source_event_hash.strip():
        return [f"hash:{source_event_hash.strip()}"]
    source_group = payload.get("source_group")
    if isinstance(source_group, str) and source_group.strip():
        return [f"source_group:{source_group.strip()}"]
    entity_name = payload.get("entity_name")
    entity_type = payload.get("entity_type")
    if isinstance(entity_name, str) and entity_name.strip():
        kind = entity_type.strip() if isinstance(entity_type, str) and entity_type.strip() else "memory"
        return [f"entity:{kind}:{entity_name.strip()}"]
    return []


def _purpose_outcome_for_context(
    context: Context,
    aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    keys = _purpose_outcome_context_keys(context)
    merged: dict[str, Any] | None = None
    for key in keys:
        aggregate = aggregates.get(key)
        if aggregate is None:
            continue
        if merged is None:
            merged = {
                "positive_count": 0,
                "negative_count": 0,
                "outcomes": [],
                "latest_event_seq": None,
            }
        merged["positive_count"] = int(merged["positive_count"]) + int(aggregate.get("positive_count", 0))
        merged["negative_count"] = int(merged["negative_count"]) + int(aggregate.get("negative_count", 0))
        for outcome in aggregate.get("outcomes", []):
            if outcome not in merged["outcomes"]:
                merged["outcomes"].append(outcome)
        latest = aggregate.get("latest_event_seq")
        if isinstance(latest, int):
            current = merged.get("latest_event_seq")
            merged["latest_event_seq"] = latest if not isinstance(current, int) else max(current, latest)
    return merged


def _purpose_outcome_context_keys(context: Context) -> list[str]:
    metadata = context.metadata or {}
    citation = _context_citation(context)
    if citation:
        return [f"citation:{citation.strip()}"]
    source_event_hash = metadata.get("source_event_hash")
    if isinstance(source_event_hash, str) and source_event_hash.strip():
        return [f"hash:{source_event_hash.strip()}"]
    identity = _context_identity(context)
    return [f"entity:{identity['entity_type']}:{identity['entity_name']}"]


def _purpose_outcome_suppression_candidates(current_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for fact in current_facts:
        explanation = fact.get("score_explanation")
        if not isinstance(explanation, dict):
            continue
        outcome = explanation.get("purpose_outcome")
        if not isinstance(outcome, dict) or not outcome.get("suppression_candidate"):
            continue
        candidates.append(
            {
                "entity_name": fact.get("entity_name") or _context_content_identity(str(fact.get("content", ""))),
                "entity_type": fact.get("entity_type") or "memory",
                "citation": fact.get("citation"),
                "negative_count": int(outcome.get("negative_count", 0)),
                "positive_count": int(outcome.get("positive_count", 0)),
                "outcomes": [str(value) for value in outcome.get("outcomes", [])],
            }
        )
    return candidates


def _normalize_context_feedback(feedback: str) -> str:
    normalized = feedback.casefold().strip()
    if normalized not in {"used", "helpful", "irrelevant"}:
        raise ValueError("feedback must be one of: used, helpful, irrelevant")
    return normalized


def _feedback_purpose_payload(
    purpose: PurposeProfile | dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    if purpose is None:
        return None
    return purpose_profile(purpose).to_dict()


def _feedback_outcome(outcome: str | None) -> str | None:
    if outcome is None:
        return None
    value = str(outcome).strip()
    return value or None


def build_memory_checkout(
    *,
    query: str,
    assembly: ContextAssembly,
    ref: MemoryRef | None = None,
    purpose: PurposeProfile | dict[str, Any] | str | None = None,
    now: datetime | None = None,
    retrieval_profile: RetrievalProfile | None = None,
    cues: dict[str, str] | None = None,
    salience_floor: float = 0.15,
    salience_half_life_days: float = SALIENCE_HALF_LIFE_DAYS,
) -> MemoryCheckout:
    """Build the Memory Checkout contract from assembled context.

    ``now`` anchors the salience replay; callers on the serving paths pass
    an explicit ``datetime.now(UTC)`` and omitted values fall back to the
    same clock.

    Under every pre-cognitive profile (``retrieval_profile`` omitted or
    with its cognitive flags off) salience never changes ranking, ordering,
    or selection — it is exposed in diagnostics only, byte-identical to the
    pre-cognitive contract. Only the opt-in cognitive profile blends
    salience and cue overlap into ranking and applies the attenuation floor
    (see :func:`_rank_cognitive_contexts` for the blend).
    """
    checkout_now = now if now is not None else datetime.now(UTC)
    profile = purpose_profile(purpose)
    purpose_payload = profile.to_dict()
    checkout_contexts = _checkout_contexts_with_synthesis(query, assembly)
    cognitive = retrieval_profile is not None and (
        retrieval_profile.salience_ranking or retrieval_profile.cue_blending
    )
    attenuation: dict[str, Any] | None = None
    if cognitive and retrieval_profile is not None:
        ranked_contexts, attenuation = _rank_cognitive_contexts(
            checkout_contexts,
            query=query,
            replay_events=assembly.replay_events,
            retrieval_profile=retrieval_profile,
            cues=cues,
            salience_floor=salience_floor,
            salience_half_life_days=salience_half_life_days,
            now=checkout_now,
        )
    else:
        ranked_contexts = sorted(
            checkout_contexts,
            key=lambda context: _checkout_rank(context, query),
            reverse=True,
        )
    candidate_current_facts = [
        _checkout_fact(context) for context in ranked_contexts if context.valid_to is None
    ]
    candidate_evidence = [
        _checkout_evidence(context) for context in ranked_contexts if _context_citation(context)
    ]
    candidate_current_facts, candidate_evidence, purpose_policy = _apply_purpose_checkout_policy(
        profile,
        current_facts=candidate_current_facts,
        evidence=candidate_evidence,
    )
    selection = select_checkout_evidence(
        query=query,
        purpose=profile,
        evidence_plan=build_evidence_plan(query, limit=10),
        current_facts=candidate_current_facts,
        evidence=candidate_evidence,
    )
    current_facts = selection.current_facts
    evidence = selection.evidence
    provenance = [_checkout_provenance(context) for context in ranked_contexts if _context_citation(context)]
    warnings = list(assembly.warnings)
    if assembly.compacted and not any("compacted" in warning for warning in warnings):
        warnings.append("Recent replay was compacted to fit the checkout budget.")
    if current_facts and not evidence:
        warnings.append("Checkout contains current facts without Eventloom citations.")
    retention = {
        "policy": "current_only",
        "superseded_contexts_excluded": sum(1 for context in assembly.contexts if context.valid_to is not None),
    }
    if purpose_policy["suppressed_count"]:
        retention["purpose_policy"] = purpose_policy
    suppression_candidates = _purpose_outcome_suppression_candidates(current_facts)
    if suppression_candidates:
        existing_policy = retention.get("purpose_policy")
        policy_payload = dict(existing_policy) if isinstance(existing_policy, dict) else {}
        policy_payload["suppression_candidates"] = suppression_candidates
        retention["purpose_policy"] = policy_payload
        warnings.append("Purpose outcome history marks retrieved memory as a suppression candidate.")
    diagnostics = build_checkout_diagnostics(
        query=query,
        purpose=purpose_payload,
        source_lanes=_checkout_source_lanes(ranked_contexts),
        current_facts=current_facts,
        evidence=evidence,
        retention=retention,
        warnings=warnings,
    )
    if selection.accepted_state is not None:
        diagnostics = {**diagnostics, "accepted_state": _accepted_state_diagnostics(selection.accepted_state)}
    skills = _checkout_skills(ranked_contexts, query)
    if skills:
        diagnostics = {**diagnostics, "skills": {"count": len(skills), "items": skills}}
    skill_analytics = _checkout_skill_analytics(ranked_contexts)
    if skill_analytics["version_count"] or skill_analytics["outcome_count"]:
        diagnostics = {**diagnostics, "skill_analytics": skill_analytics}
    retrieval_profile_diagnostics = assembly.working_set.get("retrieval_profile")
    if isinstance(retrieval_profile_diagnostics, dict):
        diagnostics = {**diagnostics, "retrieval_profile": retrieval_profile_diagnostics}
    purpose_retrieval = assembly.working_set.get("purpose_retrieval_policy")
    if isinstance(purpose_retrieval, dict):
        diagnostics = {**diagnostics, "purpose_retrieval_policy": purpose_retrieval}
    recall_diagnostics = assembly.recall.to_diagnostics()
    if recall_diagnostics["candidate_count"] and recall_diagnostics["candidate_count"] != len(assembly.contexts):
        diagnostics = {**diagnostics, "recall": recall_diagnostics}
    salience = _checkout_salience_diagnostics(
        replay_events=assembly.replay_events,
        current_facts=current_facts,
        evidence=evidence,
        now=checkout_now,
        half_life_days=salience_half_life_days,
    )
    if salience is not None:
        diagnostics = {**diagnostics, "salience": salience}
    if attenuation is not None:
        diagnostics = {**diagnostics, "attenuation": attenuation}
    guidance = build_checkout_guidance(
        query=query,
        purpose=purpose_payload,
        current_facts=current_facts,
        retention=retention,
        evidence=evidence,
    )
    quality = build_checkout_quality(
        diagnostics=diagnostics,
        guidance=guidance,
    )
    compact_contexts = build_compact_answer_contexts(
        query=query,
        current_facts=current_facts,
        evidence=evidence,
        diagnostics=diagnostics,
        quality=quality,
    )
    if compact_contexts and "synthesis" in diagnostics:
        diagnostics = {**diagnostics, "compact_contexts": compact_contexts}
    prompt = format_memory_checkout_prompt(
        query=query,
        assembly_prompt=assembly.prompt,
        current_facts=current_facts,
        evidence=evidence,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )
    return MemoryCheckout(
        session_id=assembly.session_id,
        query=query,
        prompt=prompt,
        working_set=assembly.working_set,
        ref=ref.to_dict() if ref is not None else None,
        current_facts=current_facts,
        evidence=evidence,
        provenance=provenance,
        retention=retention,
        warnings=warnings,
        guidance=guidance,
        quality=quality,
        diagnostics=diagnostics,
        context_counts=assembly.context_counts,
        replay_event_count=assembly.replay_event_count,
        compacted=assembly.compacted,
        assembly_policy=assembly.assembly_policy,
        purpose=purpose_payload,
    )


def _rank_cognitive_contexts(
    contexts: list[Context],
    *,
    query: str,
    replay_events: list[Any],
    retrieval_profile: RetrievalProfile,
    cues: dict[str, str] | None,
    salience_floor: float,
    salience_half_life_days: float,
    now: datetime,
) -> tuple[list[Context], dict[str, Any] | None]:
    """Rank checkout contexts under the cognitive retrieval profile.

    Blend (documented contract):

    - **Salience multiplier** (``salience_ranking``): relevance is multiplied
      by normalized salience ``m = clamp(score / SALIENCE_BASE,
      [SALIENCE_MIN, SALIENCE_MAX])`` — never-reinforced memories keep the
      implicit base salience 1.0 and rank exactly as before. The multiplier
      applies to the primary token-overlap key and to the fused-score
      tiebreak; the citation/source/type priority keys are unchanged.
    - **Cue bonus** (``cue_blending``): ``CUE_MATCH_WEIGHT *
      jaccard(query_cues, stored_cues)`` is added to the primary key — a
      bounded bonus of at most ``CUE_MATCH_WEIGHT`` (0.25) for a perfect cue
      match. No cues on either side means zero bonus.
    - **Attenuation floor**: memories whose replayed salience is strictly
      below ``salience_floor`` are excluded from default checkout ranking
      and listed (labeled ``attenuated``) in the returned diagnostics.
      Authority-bearing state (accepted review status / accepted-authority
      scope) and payloads pinned with ``"pinned": true`` are exempt: they
      stay in ranking and are listed as exempt. Excluded memories remain
      fully reachable via explicit ``memory_query``/``memory_replay``, which
      never route through this function.
    """
    ref_index = event_ref_index(replay_events)
    payloads_by_seq = _payloads_by_seq(replay_events)
    states: dict[EventRef, SalienceState] = {}
    if retrieval_profile.salience_ranking:
        states = SalienceLedger(half_life_days=salience_half_life_days).replay(
            replay_events,
            now=now,
        )
    query_cues = cue_pairs(cues) if retrieval_profile.cue_blending else frozenset()
    ranked: list[tuple[tuple[float, int, int, int, float, str, float], Context]] = []
    excluded: list[dict[str, Any]] = []
    exempt: list[dict[str, Any]] = []
    for context in contexts:
        citation = _context_citation(context)
        ref = resolve_citation_target(citation, event_index=ref_index)
        payload = payloads_by_seq.get(ref.seq) if ref is not None else None
        multiplier = 1.0
        if retrieval_profile.salience_ranking:
            state = states.get(ref) if ref is not None else None
            salience_score = state.score if state is not None else SALIENCE_BASE
            multiplier = min(max(salience_score / SALIENCE_BASE, SALIENCE_MIN), SALIENCE_MAX)
            if state is not None and ref is not None and state.score < salience_floor:
                entry = {
                    "citation": citation,
                    "seq": ref.seq,
                    "hash": ref.hash,
                    "salience_score": round(state.score, 4),
                    "label": "attenuated",
                }
                exempt_reason = _attenuation_exempt_reason(context, payload)
                if exempt_reason is None:
                    excluded.append(entry)
                    continue
                exempt.append({**entry, "exempt_reason": exempt_reason})
        cue_bonus = 0.0
        if query_cues and payload is not None:
            cue_bonus = CUE_MATCH_WEIGHT * cue_overlap(query_cues, cue_pairs(payload.get("cues")))
        base = _checkout_rank(context, query)
        ranked.append(
            (
                (
                    base[0] * multiplier + cue_bonus,
                    base[1],
                    base[2],
                    base[3],
                    base[4],
                    base[5],
                    base[6] * multiplier,
                ),
                context,
            )
        )
    ordered = [context for _, context in sorted(ranked, key=lambda item: item[0], reverse=True)]
    attenuation: dict[str, Any] | None = None
    if retrieval_profile.salience_ranking:
        attenuation = {
            "authority_status": "non_authoritative",
            "floor": salience_floor,
            "label": "attenuated",
            "excluded_count": len(excluded),
            "excluded": excluded[:20],
            "exempt_count": len(exempt),
            "exempt": exempt[:20],
        }
    return ordered, attenuation


def _payloads_by_seq(replay_events: list[Any]) -> dict[int, dict[str, Any]]:
    """Map replayed event seq -> payload dict for cue/pin/authority lookups."""
    payloads: dict[int, dict[str, Any]] = {}
    for event in replay_events:
        seq = getattr(event, "seq", None)
        payload = getattr(event, "payload", None)
        if isinstance(seq, int) and isinstance(payload, dict):
            payloads[seq] = payload
    return payloads


_ATTENUATION_EXEMPT_AUTHORITIES = frozenset(
    {"accepted", "authoritative", "parent-accepted", "promoted"}
)


def _attenuation_exempt_reason(
    context: Context,
    payload: dict[str, Any] | None,
) -> str | None:
    """Return why a below-floor memory survives attenuation, or None.

    Exempt: payloads pinned with the additive ``"pinned": true`` metadata
    flag, and authority-bearing state — an accepted review status or an
    accepted/authoritative authority scope on either the projected context
    metadata or the source event payload.
    """
    if payload is not None and payload.get("pinned") is True:
        return "pinned"
    metadata = context.metadata or {}
    for source in (metadata, payload or {}):
        if _checkout_policy_text(source.get("review_status")) == "accepted":
            return "authority"
        authority = _checkout_policy_text(
            source.get("authority")
            or source.get("authority_scope")
            or source.get("authority_status")
        )
        if authority in _ATTENUATION_EXEMPT_AUTHORITIES:
            return "authority"
    return None


def _checkout_salience_diagnostics(
    *,
    replay_events: list[Any],
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    now: datetime,
    half_life_days: float = SALIENCE_HALF_LIFE_DAYS,
) -> dict[str, Any] | None:
    """Replay salience over the checkout's own replay for diagnostics only.

    Pure function of the replayed log slice and ``now``: reinforcement events
    fold into per-memory scores, then surfaced packet citations are resolved
    to their sealed refs and annotated with the score composition. Returns
    None when nothing surfaced carries replayed salience, keeping the
    diagnostics payload byte-identical to the pre-salience contract until
    reinforcement events exist.
    """
    if not replay_events:
        return None
    states = SalienceLedger(half_life_days=half_life_days).replay(replay_events, now=now)
    if not states:
        return None
    index = event_ref_index(replay_events)
    items: list[dict[str, Any]] = []
    seen: set[EventRef] = set()
    for item in [*current_facts, *evidence]:
        ref = resolve_citation_target(item.get("citation"), event_index=index)
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        state = states.get(ref)
        if state is None:
            continue
        items.append(
            {
                "citation": item.get("citation"),
                "seq": ref.seq,
                "hash": ref.hash,
                "composition": state.composition(),
            }
        )
    if not items:
        return None
    return {
        "authority_status": "non_authoritative",
        "half_life_days": half_life_days,
        "scored_count": len(items),
        "items": items,
    }


def _checkout_contexts_with_synthesis(query: str, assembly: ContextAssembly) -> list[Context]:
    """Return recall contexts plus a compact checkout-only synthesis proof when available."""
    checkout_contexts = list(assembly.recall.contexts() or assembly.contexts)
    if any(
        (context.metadata or {}).get("source_kind") == "source_synthesis"
        or "zaxy_synthesis_bundle=true" in context.content
        for context in checkout_contexts
    ):
        return checkout_contexts
    source_contexts = [
        context
        for context in checkout_contexts
        if _checkout_source_lane(context) in {"verbatim", "eventloom", "projection"}
    ]
    graph_contexts = [
        context
        for context in checkout_contexts
        if _checkout_source_lane(context) == "graph"
    ]
    synthesis_contexts = _prefer_verbatim_for_duplicate_source_groups(source_contexts, graph_contexts)
    if not synthesis_contexts:
        return checkout_contexts
    result = source_synthesis_bundle_result(
        query=query,
        source_results=synthesis_contexts,
        limit=10,
        preferred_source_groups=[
            source_context_group(_source_context_text(context))
            for context in graph_contexts
        ],
    )
    if result is None:
        return checkout_contexts
    bundle = result.content
    score = max((context.score for context in checkout_contexts), default=0.0) + 1.0
    return [
        Context(
            content=bundle,
            source="verbatim",
            score=score,
            metadata={
                "source_kind": "source_synthesis",
                "assembly_hint": "source_synthesis",
                "checkout_only": True,
                **_synthesis_packet_metadata(bundle, result.packet),
            },
        ),
        *checkout_contexts,
    ]


def _checkout_fact(context: Context) -> dict[str, Any]:
    metadata = context.metadata or {}
    fact: dict[str, Any] = {
        "content": context.content,
        "source": context.source,
        "score": context.score,
        "citation": _context_citation(context),
        "valid_from": context.valid_from,
        "valid_to": context.valid_to,
        "source_lane": _checkout_source_lane(context),
    }
    for key in ("entity_name", "entity_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            fact[key] = value
    for key in _CHECKOUT_METADATA_FIELDS:
        value = metadata.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            fact[key] = value
    score_explanation = metadata.get("score_explanation")
    if isinstance(score_explanation, dict):
        fact["score_explanation"] = score_explanation
    synthesis_packet = metadata.get("synthesis_packet")
    if isinstance(synthesis_packet, dict):
        fact["synthesis_packet"] = synthesis_packet
    return fact


def _checkout_evidence(context: Context) -> dict[str, Any]:
    citation = _context_citation(context)
    seq, event_hash = _citation_event_identity(citation)
    evidence: dict[str, Any] = {
        "citation": citation,
        "content": context.content,
        "source": context.source,
        "source_lane": _checkout_source_lane(context),
        "score": context.score,
        "event_seq": seq,
        "event_hash": event_hash,
    }
    metadata = context.metadata or {}
    score_explanation = metadata.get("score_explanation")
    if isinstance(score_explanation, dict):
        evidence["score_explanation"] = score_explanation
    for key in _CHECKOUT_METADATA_FIELDS:
        value = metadata.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            evidence[key] = value
    synthesis_packet = metadata.get("synthesis_packet")
    if isinstance(synthesis_packet, dict):
        evidence["synthesis_packet"] = synthesis_packet
    return evidence


_CHECKOUT_METADATA_FIELDS = (
    "entity_name",
    "entity_type",
    "event_type",
    "primitive",
    "phase",
    "review_status",
    "authority_status",
    "mission_id",
    "worker_id",
    "finding_id",
    "claim_key",
    "claim_value",
    "coordination_status",
    "finding_status",
    "promoted",
    "status",
    "authority",
    "authority_scope",
    "stale",
    "superseded_by",
    "purpose_outcome_positive_count",
    "purpose_outcome_negative_count",
    "purpose_outcome_score_boost",
    "purpose_outcome_score_penalty",
    "purpose_outcome_suppression_candidate",
)


def _apply_purpose_checkout_policy(
    profile: PurposeProfile,
    *,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply purpose suppress rules before facts become model-facing memory."""
    if profile.profile == "general" or not profile.suppress:
        return current_facts, evidence, _empty_purpose_policy(profile)
    kept_facts: list[dict[str, Any]] = []
    kept_evidence: list[dict[str, Any]] = []
    suppressed_ids: set[str] = set()
    reasons: dict[str, int] = {}
    examples: list[dict[str, str]] = []
    for item in current_facts:
        reason = _purpose_suppression_reason(profile, item)
        if reason is None:
            kept_facts.append(item)
            continue
        identity = _checkout_policy_item_id(item)
        suppressed_ids.add(identity)
        reasons[reason] = reasons.get(reason, 0) + 1
        if len(examples) < 5:
            examples.append({"id": identity, "reason": reason})
    for item in evidence:
        reason = _purpose_suppression_reason(profile, item)
        identity = _checkout_policy_item_id(item)
        if reason is None and identity not in suppressed_ids:
            kept_evidence.append(item)
            continue
        if identity in suppressed_ids:
            continue
        suppressed_ids.add(identity)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
    return kept_facts, kept_evidence, {
        "profile": profile.profile,
        "suppressed_count": len(suppressed_ids),
        "suppressed_reasons": reasons,
        "suppressed_examples": examples,
        "retain": list(profile.retain),
        "suppress": list(profile.suppress),
    }


def _empty_purpose_policy(profile: PurposeProfile) -> dict[str, Any]:
    return {
        "profile": profile.profile,
        "suppressed_count": 0,
        "suppressed_reasons": {},
        "suppressed_examples": [],
        "retain": list(profile.retain),
        "suppress": list(profile.suppress),
    }


def _accepted_state_diagnostics(selection: dict[str, Any]) -> dict[str, Any]:
    """Return bounded accepted-state selection diagnostics for checkout clients."""
    selected_citations = selection.get("selected_citations")
    return {
        "mode": str(selection.get("mode") or "coordinate_accepted_state"),
        "selected_count": int(selection.get("selected_count") or 0),
        "diagnostic_count": int(selection.get("diagnostic_count") or 0),
        "selected_citations": [
            citation
            for citation in selected_citations
            if isinstance(citation, str)
        ][:10]
        if isinstance(selected_citations, list)
        else [],
    }


def _purpose_suppression_reason(profile: PurposeProfile, item: dict[str, Any]) -> str | None:
    suppress = set(profile.suppress)
    status = _checkout_policy_status(item)
    authority = _checkout_policy_text(item.get("authority") or item.get("authority_scope"))
    if "worker_local_pending" in suppress and (
        status == "pending"
        or authority in {"worker", "worker-local", "worker_local", "pending"}
        or (authority.startswith("worker") and item.get("promoted") is False)
    ):
        return "worker_local_pending"
    if "pending_unreviewed_claim" in suppress and status == "pending":
        return "pending_unreviewed_claim"
    if "rejected_finding" in suppress and status in {"rejected", "unsupported"}:
        return "rejected_finding"
    if "stale_unpromoted_finding" in suppress and (
        bool(item.get("stale")) or status in {"stale", "superseded", "deprecated"}
    ) and authority not in {"accepted", "parent-accepted", "parent_accepted", "promoted"}:
        return "stale_unpromoted_finding"
    if "low_trust_inference" in suppress and _low_trust_inferred_item(item):
        return "low_trust_inference"
    if "superseded_context" in suppress and item.get("valid_to"):
        return "superseded_context"
    if "uncited_claim" in suppress and not item.get("citation"):
        return "uncited_claim"
    return None


def _checkout_policy_status(item: dict[str, Any]) -> str:
    return _checkout_policy_text(
        item.get("coordination_status")
        or item.get("finding_status")
        or item.get("status")
    )


def _checkout_policy_text(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "-")


def _low_trust_inferred_item(item: dict[str, Any]) -> bool:
    explanation = item.get("score_explanation")
    if not isinstance(explanation, dict):
        return False
    trust = explanation.get("inferred_edge_trust")
    return isinstance(trust, int | float) and not isinstance(trust, bool) and float(trust) < 0.7


def _checkout_policy_item_id(item: dict[str, Any]) -> str:
    for key in ("finding_id", "citation", "content"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _synthesis_packet_metadata(content: str, packet_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if packet_payload is not None:
        return {"synthesis_packet": packet_payload}
    packet = synthesis_packet_from_items([{"content": content}])
    if not packet.answer_candidates and not packet.ledger_rows:
        return {}
    return {
        "synthesis_packet": {
            "schema_version": "synthesis_packet_v1",
            "answer_candidates": packet.answer_candidates,
            "ledger_rows": packet.ledger_rows,
            "content": content,
        }
    }


def _checkout_provenance(context: Context) -> dict[str, Any]:
    citation = _context_citation(context)
    seq, event_hash = _citation_event_identity(citation)
    return {
        "citation": citation,
        "event_seq": seq,
        "event_hash": event_hash,
        "source": context.source,
        "source_lane": _checkout_source_lane(context),
        "valid_from": context.valid_from,
        "valid_to": context.valid_to,
    }


def _checkout_source_lanes(contexts: list[Context]) -> dict[str, int]:
    source_lanes: dict[str, int] = {}
    for context in contexts:
        lane = _checkout_source_lane(context)
        source_lanes[lane] = source_lanes.get(lane, 0) + 1
    return source_lanes


def _checkout_skills(contexts: list[Context], query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    skill_contexts = [
        context
        for context in contexts
        if (context.metadata or {}).get("entity_type") == "skill_version"
    ]
    if not skill_contexts:
        return []
    query_tokens = _checkout_tokens(query)
    skills: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for context in skill_contexts:
        metadata = context.metadata or {}
        skill_id = metadata.get("skill_id")
        entity_name = metadata.get("entity_name")
        if not isinstance(skill_id, str) or not skill_id:
            if isinstance(entity_name, str) and entity_name.startswith("skill:"):
                skill_id = entity_name.removeprefix("skill:").split(":v", 1)[0]
            else:
                continue
        version = str(metadata.get("version") or _skill_version_from_entity(entity_name) or "1")
        key = (skill_id, version)
        if key in seen:
            continue
        applicability = _metadata_text_list(metadata.get("applicability"))
        procedure = _metadata_text_list(metadata.get("procedure"))
        haystack = " ".join([context.content, *applicability, str(metadata.get("summary") or "")])
        if query_tokens and not (_checkout_tokens(haystack) & query_tokens):
            continue
        seen.add(key)
        skills.append(
            {
                "skill_id": skill_id,
                "version": version,
                "status": str(metadata.get("status") or "unknown"),
                "summary": str(metadata.get("summary") or context.content),
                "procedure": procedure,
                "applicability": applicability,
                "citation": _context_citation(context),
                "score": context.score,
            }
        )
        if len(skills) >= limit:
            break
    return skills


def _checkout_skill_analytics(contexts: list[Context]) -> dict[str, Any]:
    """Summarize skill outcome history without mutating Skill Memory."""
    versions: dict[tuple[str, str], dict[str, Any]] = {}
    outcomes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for context in contexts:
        metadata = context.metadata or {}
        entity_type = metadata.get("entity_type")
        if entity_type not in {"skill_version", "skill_outcome"}:
            continue
        key = _skill_context_key(metadata)
        if key is None:
            continue
        if entity_type == "skill_version":
            versions[key] = {
                "skill_id": key[0],
                "version": key[1],
                "status": str(metadata.get("status") or "unknown"),
                "citation": _context_citation(context),
                "failure_modes": _metadata_text_list(metadata.get("failure_modes")),
                "rollback": str(metadata.get("rollback") or "").strip(),
            }
        else:
            outcomes.setdefault(key, []).append(
                {
                    "success_score": _optional_float(metadata.get("success_score")),
                    "feedback": str(metadata.get("feedback") or "").casefold().strip(),
                    "citation": _context_citation(context),
                }
            )

    promotion_candidates: list[dict[str, Any]] = []
    rollback_candidates: list[dict[str, Any]] = []
    contradicted_keys = {
        key for key, version in versions.items() if version["status"] == "contradicted"
    }
    for key in sorted(set(versions) | set(outcomes)):
        version = versions.get(
            key,
            {
                "skill_id": key[0],
                "version": key[1],
                "status": "unknown",
                "citation": "",
                "failure_modes": [],
                "rollback": "",
            },
        )
        outcome_items = outcomes.get(key, [])
        scores = [
            score
            for item in outcome_items
            if (score := item.get("success_score")) is not None
        ]
        average_score = round(sum(scores) / len(scores), 4) if scores else None
        success_count = sum(
            1
            for item in outcome_items
            if _skill_outcome_is_success(item.get("feedback"), item.get("success_score"))
        )
        failure_count = sum(
            1
            for item in outcome_items
            if _skill_outcome_is_failure(item.get("feedback"), item.get("success_score"))
        )
        latest_citation = _latest_skill_citation(version, outcome_items)
        status = version["status"]
        if (
            status in {"validated", "revised", "outcome_recorded"}
            and key not in contradicted_keys
            and success_count > 0
            and failure_count == 0
            and (average_score is None or average_score >= 0.8)
        ):
            promotion_candidates.append(
                {
                    "skill_id": key[0],
                    "version": key[1],
                    "status": status,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "average_success_score": average_score,
                    "latest_citation": latest_citation,
                }
            )
        rollback_reason = _skill_rollback_reason(status, success_count, failure_count, average_score)
        if rollback_reason is not None:
            rollback_candidates.append(
                {
                    "skill_id": key[0],
                    "version": key[1],
                    "status": status,
                    "reason": rollback_reason,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "average_success_score": average_score,
                    "failure_modes": version["failure_modes"],
                    "rollback": version["rollback"],
                    "latest_citation": latest_citation,
                }
            )

    return {
        "version_count": len(versions),
        "outcome_count": sum(len(items) for items in outcomes.values()),
        "contradiction_count": len(contradicted_keys),
        "promotion_candidates": promotion_candidates[:5],
        "rollback_candidates": rollback_candidates[:5],
    }


def _skill_context_key(metadata: dict[str, Any]) -> tuple[str, str] | None:
    skill_id = metadata.get("skill_id")
    entity_name = metadata.get("entity_name")
    if not isinstance(skill_id, str) or not skill_id:
        if isinstance(entity_name, str) and entity_name.startswith("skill:"):
            skill_id = entity_name.removeprefix("skill:").split(":v", 1)[0]
        else:
            return None
    version = str(metadata.get("version") or _skill_version_from_entity(entity_name) or "1")
    return skill_id, version


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _skill_outcome_is_success(feedback: object, score: object) -> bool:
    normalized = str(feedback or "").casefold().strip()
    numeric_score = score if isinstance(score, float) else None
    return normalized in {"used", "helpful", "passed", "success"} or (
        numeric_score is not None and numeric_score >= 0.8
    )


def _skill_outcome_is_failure(feedback: object, score: object) -> bool:
    normalized = str(feedback or "").casefold().strip()
    numeric_score = score if isinstance(score, float) else None
    return normalized in {"failed", "failure", "irrelevant", "regressed"} or (
        numeric_score is not None and numeric_score < 0.5
    )


def _latest_skill_citation(version: dict[str, Any], outcomes: list[dict[str, Any]]) -> str:
    for outcome in reversed(outcomes):
        citation = outcome.get("citation")
        if isinstance(citation, str) and citation:
            return citation
    citation = version.get("citation")
    return citation if isinstance(citation, str) else ""


def _skill_rollback_reason(
    status: str,
    success_count: int,
    failure_count: int,
    average_score: float | None,
) -> str | None:
    if status in {"contradicted", "deprecated"}:
        return status
    if failure_count > success_count:
        return "failed_outcomes"
    if average_score is not None and average_score < 0.5:
        return "low_success_score"
    return None


def _metadata_text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            texts.append(text)
    return texts


def _skill_version_from_entity(value: object) -> str | None:
    if not isinstance(value, str) or ":v" not in value:
        return None
    version = value.rsplit(":v", 1)[1].strip()
    return version or None


def _checkout_recall_limit(query: str, limit: int) -> int:
    """Return the internal recall budget for checkout without inflating prompt context."""
    prompt_limit = validate_limit(limit)
    plan = build_evidence_plan(query, limit=max(prompt_limit, 10))
    if not plan.promote_cited_sources:
        return prompt_limit
    source_budget = source_lane_candidate_limit(query, limit=max(prompt_limit, 10))
    return min(
        MAX_QUERY_LIMIT,
        max(
            prompt_limit,
            source_budget,
            plan.required_source_groups * 8,
        ),
    )


def _checkout_source_lane(context: Context) -> str:
    metadata = context.metadata or {}
    lane = metadata.get("assembly_lane")
    if isinstance(lane, str) and lane:
        return lane
    if context.source in {"verbatim", "packet_memory", "projection", "eventloom"}:
        return context.source
    return "graph"


def _source_context_text(context: Context) -> str:
    """Return source text with compact metadata used by retrieval planning helpers."""
    metadata = context.metadata or {}
    prefixes: list[str] = []
    citation = metadata.get("citation")
    if citation:
        prefixes.append(f"citation={citation}")
    source_path = metadata.get("source_path")
    if source_path:
        prefixes.append(f"source_path={source_path}")
    event_thread = metadata.get("event_thread")
    if event_thread:
        prefixes.append(f"thread={event_thread}")
    source_kind = metadata.get("source_kind")
    if source_kind:
        prefixes.append(f"source_kind={source_kind}")
    if not prefixes:
        return context.content
    return " ".join([*prefixes, context.content])


def _unique_synthesis_context_texts(contexts: list[Context]) -> list[str]:
    """Return synthesis candidate text once while preserving rank order."""
    texts: list[str] = []
    seen: set[str] = set()
    for context in contexts:
        text = _source_context_text(context)
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return texts


def _prefer_verbatim_for_duplicate_source_groups(
    source_contexts: list[Context],
    graph_contexts: list[Context],
) -> list[str]:
    """Return synthesis candidates while avoiding graph summaries over full source text."""
    source_groups = {
        source_context_group(_source_context_text(context))
        for context in source_contexts
    }
    contexts = [
        *source_contexts,
        *[
            context
            for context in graph_contexts
            if source_context_group(_source_context_text(context)) not in source_groups
        ],
    ]
    return _unique_synthesis_context_texts(contexts)


def _append_context_once(
    target: list[Context],
    context: Context,
    seen: set[tuple[str, str]],
) -> None:
    metadata = context.metadata or {}
    citation = str(metadata.get("citation") or "")
    key = (citation, context.content)
    if key in seen:
        return
    seen.add(key)
    target.append(context)


def _context_citation(context: Context) -> str | None:
    metadata = context.metadata or {}
    citation = metadata.get("citation")
    return citation if isinstance(citation, str) and citation else None


_REASONING_EVENT_CITATION_RE = re.compile(
    r"^eventloom://[^/\s]+/events/[1-9][0-9]*#(?:[0-9a-f]{12}|[0-9a-f]{64})$"
)


_CLAIM_NEGATION_TERMS = {
    "not",
    "never",
    "no",
    "none",
    "false",
    "refute",
    "refuted",
    "conflict",
    "conflicted",
    "contradict",
    "contradicted",
}


def _strict_reasoning_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict: list[dict[str, Any]] = []
    for item in evidence:
        citation = item.get("citation")
        if isinstance(citation, str) and _REASONING_EVENT_CITATION_RE.fullmatch(citation):
            strict.append(dict(item))
    return strict


def _causal_result_reasoning_evidence(item: dict[str, Any]) -> dict[str, Any]:
    raw_source = item.get("source")
    raw_target = item.get("target")
    raw_evidence = item.get("evidence")
    source = raw_source if isinstance(raw_source, dict) else {}
    target = raw_target if isinstance(raw_target, dict) else {}
    evidence = raw_evidence if isinstance(raw_evidence, dict) else {}
    content = evidence.get("summary") or (
        f"{source.get('name', 'unknown source')} {item.get('relation_type', 'related')} "
        f"{target.get('name', 'unknown target')}"
    )
    return {
        "citation": item.get("citation", ""),
        "content": str(content),
        "source": "causal_predecessor",
        "confidence": item.get("confidence"),
        "review_status": item.get("review_status"),
        "authority_status": item.get("authority_status"),
    }


def _checkout_reasoning_evidence(item: dict[str, Any]) -> dict[str, Any] | None:
    citation = item.get("citation")
    if not isinstance(citation, str):
        return None
    content = item.get("content") or item.get("summary") or item.get("text")
    evidence: dict[str, Any] = {
        "citation": citation,
        "content": str(content or ""),
        "source": str(item.get("source") or "checkout"),
    }
    for key in (
        "entity_name",
        "entity_type",
        "event_type",
        "event_seq",
        "event_hash",
        "source_event_seq",
        "source_event_hash",
        "authority_status",
        "authority",
        "review_status",
        "status",
        "stale",
        "superseded_by",
        "primitive",
        "phase",
    ):
        value = item.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            evidence[key] = value
    return evidence


def _score_claim_evidence(
    claim: str,
    evidence_items: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    claim_tokens = _tokens(claim)
    evidence: list[dict[str, Any]] = []
    support_count = 0
    conflict_count = 0
    for item in evidence_items:
        if not _eligible_claim_confidence_evidence(item):
            continue
        evidence_item = _checkout_reasoning_evidence(item)
        if evidence_item is None:
            continue
        content = str(evidence_item.get("content") or "")
        content_tokens = _tokens(content)
        overlap = claim_tokens & content_tokens
        if not overlap:
            continue
        label = "support"
        if _is_conflicting_claim_evidence(content_tokens, content):
            label = "conflict"
            conflict_count += 1
        else:
            support_count += 1
        evidence_item["stance"] = label
        evidence_item["matched_terms"] = sorted(overlap)
        evidence.append(evidence_item)
        if len(evidence) >= limit:
            break
    denominator = support_count + conflict_count
    confidence = support_count / denominator if denominator else 0.0
    return {
        "confidence": round(confidence, 4),
        "support_count": support_count,
        "conflict_count": conflict_count,
        "evidence": evidence,
    }


def _bounded_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("min_confidence must be a number between 0.0 and 1.0")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    return parsed


def _claim_key(claim: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", claim.casefold()).strip("-")[:80] or "claim"


def _source_events_from_reasoning_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_events: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in evidence:
        seq = item.get("source_event_seq") or item.get("event_seq")
        event_hash = item.get("source_event_hash") or item.get("event_hash")
        if not isinstance(seq, int) or not isinstance(event_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", event_hash):
            citation = item.get("citation")
            parsed_seq, parsed_hash = _citation_event_identity(citation if isinstance(citation, str) else None)
            seq = parsed_seq
            event_hash = parsed_hash
        if not isinstance(seq, int) or not isinstance(event_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", event_hash):
            continue
        key = (seq, event_hash)
        if key in seen:
            continue
        seen.add(key)
        source_events.append({"seq": seq, "hash": event_hash})
    return source_events


def _source_events_reasoning_evidence(session_id: str, source_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for source_event in source_events:
        seq = source_event.get("seq")
        event_hash = source_event.get("hash")
        if isinstance(seq, int) and isinstance(event_hash, str):
            evidence.append(
                {
                    "citation": f"eventloom://{session_id}/events/{seq}#{event_hash[:12]}",
                    "source_event_seq": seq,
                    "source_event_hash": event_hash,
                }
            )
    return evidence


def _metacognition_payloads_reasoning_evidence(
    session_id: str,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for payload in payloads:
        source_groups = [
            payload.get("source_events"),
            payload.get("supporting_source_events"),
            payload.get("conflicting_source_events"),
        ]
        for group in source_groups:
            if not isinstance(group, list):
                continue
            for item in _source_events_reasoning_evidence(session_id, group):
                seq = item.get("source_event_seq")
                event_hash = item.get("source_event_hash")
                if not isinstance(seq, int) or not isinstance(event_hash, str):
                    continue
                key = (seq, event_hash)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(item)
        payload_evidence = payload.get("evidence")
        evidence_items = payload_evidence if isinstance(payload_evidence, list) else []
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            citation = item.get("citation")
            if isinstance(citation, str) and citation.strip():
                evidence.append({"citation": citation.strip()})
    return evidence


def _reverification_needs_from_events(
    events: list[dict[str, Any]],
    *,
    query: str | None,
    limit: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    terms = _tokens(query or "")
    needs: list[dict[str, Any]] = []
    for event in events:
        payload = event["payload"]
        haystack = " ".join(str(payload.get(key) or "") for key in ("claim", "claim_key", "query", "question", "reason"))
        if terms and not (_tokens(haystack) & terms):
            continue
        event_type = event["event_type"]
        if event_type == "metacognition.unknown.recorded" and payload.get("status") == "open":
            needs.append({"reason": "known_unknown_open", "event_type": event_type, **dict(payload)})
        elif event_type == "metacognition.reverify.requested" and payload.get("status") == "open":
            needs.append({"reason": "reverify_request_open", "event_type": event_type, **dict(payload)})
        elif event_type == "metacognition.conflict.clustered" and payload.get("resolution_status") == "unresolved":
            needs.append({"reason": "conflict_unresolved", "event_type": event_type, **dict(payload)})
        elif event_type == "metacognition.confidence.assessed":
            confidence = payload.get("confidence")
            conflict_count = payload.get("conflict_count")
            if (
                isinstance(confidence, int | float)
                and not isinstance(confidence, bool)
                and float(confidence) < min_confidence
            ) or (isinstance(conflict_count, int) and conflict_count > 0) or payload.get("requires_reverify") is True:
                needs.append({"reason": "confidence_requires_reverify", "event_type": event_type, **dict(payload)})
        if len(needs) >= limit:
            break
    return needs


def _eligible_claim_confidence_evidence(item: dict[str, Any]) -> bool:
    event_type = str(item.get("event_type") or "").strip()
    entity_type = str(item.get("entity_type") or "").strip()
    if event_type in {
        "belief.update.proposed",
        "reasoning.primitive.called",
        "metacognition.unknown.recorded",
        "metacognition.confidence.assessed",
        "metacognition.conflict.clustered",
        "metacognition.reverify.requested",
    }:
        return False
    if entity_type in {
        "belief_update_proposal",
        "reasoning_primitive_observation",
        "known_unknown",
        "confidence_assessment",
        "conflict_cluster",
        "reverify_request",
    }:
        return False
    review_status = str(item.get("review_status") or item.get("status") or "").casefold().strip()
    if review_status in {"pending", "rejected", "deferred", "unsupported", "stale", "conflicted"}:
        return False
    if item.get("stale") is True:
        return False
    superseded_by = item.get("superseded_by")
    return not (isinstance(superseded_by, str) and superseded_by.strip())


def _is_conflicting_claim_evidence(content_tokens: set[str], content: str) -> bool:
    lowered = content.casefold()
    if "did not" in lowered or "does not" in lowered or "not caused" in lowered:
        return True
    return bool(content_tokens & _CLAIM_NEGATION_TERMS)


def _procedure_contexts(contexts: list[Context], *, limit: int) -> list[dict[str, Any]]:
    procedures: list[dict[str, Any]] = []
    for context in contexts:
        metadata = context.metadata or {}
        if not _is_procedure_context(context):
            continue
        if _excluded_procedure_candidate(context):
            continue
        citation = _context_citation(context)
        procedures.append(
            {
                "content": context.content,
                "source": context.source,
                "score": context.score,
                "citation": citation,
                "metadata": dict(metadata),
            }
        )
        if len(procedures) >= limit:
            break
    return procedures


def _is_procedure_context(context: Context) -> bool:
    metadata = context.metadata or {}
    source = context.source.casefold()
    candidate_type = str(metadata.get("candidate_type") or metadata.get("kind") or "").casefold()
    event_type = str(metadata.get("event_type") or "").casefold()
    content = context.content.casefold()
    is_procedure = (
        candidate_type == "procedure"
        or "procedure" in event_type
        or content.startswith("procedure:")
        or "procedure" in content.split()[:5]
    )
    is_skill_or_consolidation = (
        "skill" in source
        or "consolidation" in source
        or event_type.startswith("skill.")
        or candidate_type == "procedure"
    )
    return is_procedure and is_skill_or_consolidation


def _excluded_procedure_candidate(context: Context) -> bool:
    metadata = context.metadata or {}
    review_status = str(metadata.get("review_status") or "").casefold()
    if review_status in {"rejected", "stale", "conflicted"}:
        return True
    if metadata.get("stale") is True:
        return True
    if context.valid_to is not None:
        return True
    return bool(metadata.get("superseded_by"))


def _procedure_reasoning_evidence(procedure: dict[str, Any]) -> dict[str, Any] | None:
    citation = procedure.get("citation")
    if not isinstance(citation, str):
        return None
    return {
        "citation": citation,
        "content": str(procedure.get("content") or ""),
        "source": str(procedure.get("source") or "procedure"),
    }


def _contexts_as_of_seq(contexts: list[Context], as_of_seq: int) -> list[Context]:
    filtered = []
    for context in contexts:
        citation = _context_citation(context)
        seq, _event_hash = _citation_event_identity(citation)
        if seq is None or seq <= as_of_seq:
            filtered.append(context)
    return filtered


def _checkout_rank(context: Context, query: str) -> tuple[float, int, int, int, float, str, float]:
    query_tokens = _checkout_tokens(query)
    content_tokens = _checkout_tokens(context.content)
    overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
    metadata = context.metadata or {}
    entity_type = metadata.get("entity_type")
    type_priority = 1 if entity_type in {"task", "decision", "goal", "memory"} else 0
    citation_priority = 1 if _context_citation(context) else 0
    source_lane = _checkout_source_lane(context)
    source_priority = 1 if source_lane in {"verbatim", "eventloom", "projection"} else 0
    purpose_outcome_rank = _purpose_outcome_rank(metadata)
    return (
        overlap,
        citation_priority,
        source_priority,
        type_priority,
        purpose_outcome_rank,
        context.valid_from or "",
        context.score,
    )


def _checkout_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 2}


def _purpose_outcome_rank(metadata: dict[str, Any]) -> float:
    positive = _numeric_metadata(metadata.get("purpose_outcome_positive_count"))
    negative = _numeric_metadata(metadata.get("purpose_outcome_negative_count"))
    return max(-0.18, min(0.20, positive * 0.06 - negative * 0.06))


def _numeric_metadata(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _citation_event_identity(citation: str | None) -> tuple[int | None, str | None]:
    if not citation:
        return None, None
    event_seq: int | None = None
    event_hash: str | None = None
    if "/events/" in citation:
        tail = citation.split("/events/", 1)[1]
        seq_text = tail.split("#", 1)[0].split("/", 1)[0]
        if seq_text.isdigit():
            event_seq = int(seq_text)
    if "#" in citation:
        fragment = citation.rsplit("#", 1)[1]
        event_hash = fragment or None
    return event_seq, event_hash


def _context_identity(context: Context) -> dict[str, str]:
    metadata = context.metadata or {}
    entity_name = metadata.get("entity_name")
    entity_type = metadata.get("entity_type")
    if isinstance(entity_name, str) and entity_name.strip():
        name = entity_name.strip()
    else:
        name = _context_content_identity(context.content)
    if isinstance(entity_type, str) and entity_type.strip():
        kind = entity_type.strip()
    elif context.source == "packet_memory":
        kind = "packet_memory"
    else:
        kind = "memory"
    return {"entity_name": name, "entity_type": kind}


def _context_feedback_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "authority",
        "authority_scope",
        "coordination_status",
        "finding_id",
        "mission_id",
        "source_kind",
        "source_event_seq",
        "source_event_hash",
        "stale",
        "status",
        "worker_id",
        "provider_path",
        "model",
    }
    return {
        key: value
        for key, value in metadata.items()
        if key in allowed and isinstance(value, str | int | float | bool)
    }


def _context_content_identity(content: str) -> str:
    text = content.strip()
    if not text:
        return "context"
    if " (" in text:
        return text.split(" (", 1)[0].strip() or "context"
    return text.split(" — ", 1)[0].strip() or "context"


def _context_warnings(contexts: list[Context], *, compacted: bool) -> list[str]:
    warnings: list[str] = []
    for context in contexts:
        if _is_compacted_context(context) and not _has_source_support(context):
            warnings.append(
                f"{context.source} context '{_warning_label(context.content)}' "
                "lacks source-level citation"
            )
    if compacted and not any(_has_source_support(context) for context in contexts):
        warnings.append(
            "recent replay was truncated and no retrieved source context was available"
        )
    return warnings


def _is_compacted_context(context: Context) -> bool:
    source = context.source.casefold()
    if source in {"projection", "compaction", "compacted"}:
        return True
    metadata = context.metadata or {}
    return bool(
        metadata.get("compacted")
        or metadata.get("projection_id")
        or metadata.get("compaction_projection")
    )


def _has_source_support(context: Context) -> bool:
    metadata = context.metadata or {}
    if metadata.get("citation"):
        return True
    citations = metadata.get("citations")
    return bool(citations)


def _warning_label(content: str) -> str:
    compact = " ".join(content.split())
    if len(compact) <= 80:
        return compact
    return f"{compact[:77]}..."


def _compaction_projection_paths(
    eventloom_path: Path,
    explicit_paths: list[str | Path] | tuple[str | Path, ...],
) -> tuple[Path, ...]:
    discovered = (
        sorted(eventloom_path.rglob("*.compaction.json"))
        if eventloom_path.exists() and eventloom_path.is_dir()
        else []
    )
    ordered = [*discovered, *(Path(path) for path in explicit_paths)]
    unique: dict[Path, Path] = {}
    for path in ordered:
        unique.setdefault(path.resolve(), path)
    return tuple(unique.values())


def _consolidation_candidate_ids(events: list[Any]) -> set[str]:
    candidate_ids: set[str] = set()
    for event in events:
        if getattr(event, "type", None) != "consolidation.candidate.created":
            continue
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            continue
        candidate_id = payload.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            candidate_ids.add(candidate_id)
    return candidate_ids


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _tokens(value: str) -> set[str]:
    import re

    return {token for token in re.findall(r"[A-Za-z0-9]+", value.lower()) if len(token) > 1}
