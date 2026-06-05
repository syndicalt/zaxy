"""Production evidence building for model-facing memory checkout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from zaxy.purpose import PurposeProfile, purpose_profile
from zaxy.retrieval_plan import EvidencePlan

_SOURCE_ID_PATTERNS = (
    re.compile(r"\blongmemeval_session_id[=:]\s*['\"]?(?P<value>[A-Za-z0-9_.-]+)"),
    re.compile(r"\bsession_id[=:]\s*['\"]?(?P<value>[A-Za-z0-9_.-]+)"),
    re.compile(r"\bsource_path[=:]\s*['\"]?(?P<value>[^\s,'\"]+)"),
    re.compile(r"\bpath[=:]\s*['\"]?(?P<value>[^\s,'\"]+)"),
)
_GROUP_LIMIT = 8
_CITATION_LIMIT = 3
_SNIPPET_LIMIT = 700
_GENERIC_SOURCE_NAMES = {
    "exact",
    "eventloom",
    "graph",
    "keyword",
    "packet_memory",
    "projection",
    "traversal",
    "vector",
    "verbatim",
}


@dataclass(frozen=True)
class EvidenceSet:
    """Grouped cited evidence and evidence-plan sufficiency status."""

    groups: list[dict[str, Any]]
    status: dict[str, Any] | None

    def to_diagnostics(self) -> dict[str, Any]:
        """Return stable checkout diagnostics for the evidence set."""
        diagnostics: dict[str, Any] = {"groups": self.groups}
        if self.status is not None:
            diagnostics["status"] = self.status
        return diagnostics


@dataclass(frozen=True)
class CheckoutEvidenceSelection:
    """Selected checkout facts and cited evidence after evidence-plan promotion."""

    current_facts: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    accepted_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvidencePolicyRequirement:
    """One purpose-scoped evidence requirement."""

    key: str
    description: str
    mode: str
    terms: tuple[str, ...]
    suggested_query: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable diagnostic payload."""
        return {
            "key": self.key,
            "description": self.description,
            "mode": self.mode,
            "terms": list(self.terms),
            "suggested_query": self.suggested_query,
        }


@dataclass(frozen=True)
class EvidencePolicyResult:
    """Purpose-scoped evidence policy evaluation result."""

    profile: str
    mode: str
    satisfied: bool
    requirements: tuple[EvidencePolicyRequirement, ...]
    satisfied_requirements: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    suggested_queries: tuple[str, ...]

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a stable checkout diagnostic payload."""
        return {
            "profile": self.profile,
            "mode": self.mode,
            "satisfied": self.satisfied,
            "requirements": [requirement.to_dict() for requirement in self.requirements],
            "satisfied_requirements": list(self.satisfied_requirements),
            "missing_requirements": list(self.missing_requirements),
            "failure_reasons": list(self.failure_reasons),
            "suggested_queries": list(self.suggested_queries),
        }


_EVIDENCE_POLICY_REQUIREMENTS: dict[str, tuple[EvidencePolicyRequirement, ...]] = {
    "security": (
        EvidencePolicyRequirement(
            key="source_citation",
            description="Security memory requires cited source evidence.",
            mode="block_checkout",
            terms=("eventloom://", "source_path", "source", "citation"),
            suggested_query="cited source evidence for security risk",
        ),
        EvidencePolicyRequirement(
            key="mitigation_or_risk_owner",
            description="Security memory requires mitigation or risk-owner evidence.",
            mode="require_refresh",
            terms=("mitigation", "mitigated", "risk_owner", "risk owner", "accepted risk"),
            suggested_query="mitigation or risk owner evidence for security risk",
        ),
    ),
    "release": (
        EvidencePolicyRequirement(
            key="release_gate",
            description="Release memory requires release-gate evidence.",
            mode="block_checkout",
            terms=("release gate", "release_gate", "doctor", "readiness", "gate"),
            suggested_query="release gate evidence with current readiness status",
        ),
        EvidencePolicyRequirement(
            key="verification_refs",
            description="Release memory requires test, changelog, or package verification evidence.",
            mode="require_refresh",
            terms=("test", "pytest", "changelog", "package", "twine", "build"),
            suggested_query="test changelog package evidence for release readiness",
        ),
    ),
    "coordinate": (
        EvidencePolicyRequirement(
            key="promotion_or_review_ref",
            description="Coordinate memory requires promotion or review evidence.",
            mode="block_checkout",
            terms=("promoted", "promotion", "review", "accepted", "parent state"),
            suggested_query="Coordinate promotion review accepted parent-state evidence",
        ),
        EvidencePolicyRequirement(
            key="source_event_ref",
            description="Coordinate memory requires source-event evidence.",
            mode="require_refresh",
            terms=("eventloom://", "source_event_seq", "source_event_hash", "citation"),
            suggested_query="Coordinate source event citation for accepted finding",
        ),
    ),
    "support": (
        EvidencePolicyRequirement(
            key="customer_report_ref",
            description="Support memory requires cited customer report evidence.",
            mode="require_refresh",
            terms=("customer", "case", "ticket", "report", "eventloom://", "citation"),
            suggested_query="cited customer report evidence for support case",
        ),
        EvidencePolicyRequirement(
            key="workaround_or_resolution_ref",
            description="Support memory requires workaround or resolution evidence.",
            mode="require_refresh",
            terms=("workaround", "resolution", "resolved", "mitigation"),
            suggested_query="workaround or resolution evidence for support case",
        ),
        EvidencePolicyRequirement(
            key="impact_ref",
            description="Support memory requires customer-impact evidence.",
            mode="warn",
            terms=("impact", "severity", "affected", "customer impact"),
            suggested_query="customer impact evidence for support case",
        ),
    ),
    "product": (
        EvidencePolicyRequirement(
            key="roadmap_signal_ref",
            description="Product memory requires cited roadmap signal evidence.",
            mode="require_refresh",
            terms=("roadmap", "signal", "customer", "request", "feedback", "citation"),
            suggested_query="roadmap signal evidence with cited source",
        ),
        EvidencePolicyRequirement(
            key="tradeoff_ref",
            description="Product memory requires tradeoff or constraint evidence.",
            mode="warn",
            terms=("tradeoff", "constraint", "cost", "risk", "defer"),
            suggested_query="product tradeoff or constraint evidence",
        ),
        EvidencePolicyRequirement(
            key="experiment_or_customer_ref",
            description="Product memory requires experiment outcome or customer evidence.",
            mode="warn",
            terms=("experiment", "outcome", "customer", "promise", "result"),
            suggested_query="experiment outcome or customer promise evidence",
        ),
    ),
    "sales": (
        EvidencePolicyRequirement(
            key="buyer_ref",
            description="Sales memory requires cited buyer or account evidence.",
            mode="require_refresh",
            terms=("buyer", "account", "stakeholder", "customer", "citation"),
            suggested_query="buyer account evidence with cited source",
        ),
        EvidencePolicyRequirement(
            key="commitment_ref",
            description="Sales memory requires explicit commitment evidence.",
            mode="require_refresh",
            terms=("commitment", "committed", "promised", "followup", "next step"),
            suggested_query="buyer commitment or follow-up evidence",
        ),
        EvidencePolicyRequirement(
            key="objection_or_renewal_ref",
            description="Sales memory requires objection or renewal-risk evidence.",
            mode="warn",
            terms=("objection", "renewal", "blocker", "risk", "budget"),
            suggested_query="buyer objection or renewal risk evidence",
        ),
    ),
    "legal": (
        EvidencePolicyRequirement(
            key="exact_quote_ref",
            description="Legal memory requires exact cited wording.",
            mode="block_checkout",
            terms=("exact quote", "quoted", "clause", "section"),
            suggested_query="exact quoted legal wording with citation",
        ),
        EvidencePolicyRequirement(
            key="authority_ref",
            description="Legal memory requires authority or approval evidence.",
            mode="require_refresh",
            terms=("authority", "approved", "approval", "counsel", "owner"),
            suggested_query="legal authority or approval evidence",
        ),
        EvidencePolicyRequirement(
            key="date_or_deadline_ref",
            description="Legal memory requires date or deadline evidence.",
            mode="require_refresh",
            terms=("date", "deadline", "expires", "effective", "due"),
            suggested_query="legal date deadline or effective-window evidence",
        ),
    ),
    "executive": (
        EvidencePolicyRequirement(
            key="decision_ref",
            description="Executive memory requires cited decision evidence.",
            mode="require_refresh",
            terms=("decision", "approved", "exception", "owner", "citation"),
            suggested_query="executive decision evidence with owner",
        ),
        EvidencePolicyRequirement(
            key="risk_or_metric_ref",
            description="Executive memory requires risk or metric evidence.",
            mode="warn",
            terms=("risk", "metric", "kpi", "market", "trend", "pattern"),
            suggested_query="executive risk metric or market-pattern evidence",
        ),
        EvidencePolicyRequirement(
            key="owner_or_source_ref",
            description="Executive memory requires owner or source evidence.",
            mode="require_refresh",
            terms=("owner", "source", "sponsor", "accountable", "eventloom://", "citation"),
            suggested_query="executive owner or source evidence",
        ),
    ),
}


def select_checkout_evidence(
    *,
    query: str | None,
    purpose: PurposeProfile | dict[str, Any] | str | None = None,
    evidence_plan: EvidencePlan | dict[str, object] | None,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> CheckoutEvidenceSelection:
    """Select and order checkout facts/evidence for the query's evidence plan."""
    deduped_current = _dedupe_items(current_facts)
    deduped_evidence = _dedupe_items([item for item in evidence if _citation(item)])
    accepted_state = (
        _select_accepted_state(query=query, current_facts=deduped_current, evidence=deduped_evidence)
        if purpose_profile(purpose).profile == "coordinate"
        else None
    )
    if accepted_state is not None:
        deduped_current = accepted_state["current_facts"]
        deduped_evidence = accepted_state["evidence"]
    if not _should_promote_cited_sources(evidence_plan):
        return CheckoutEvidenceSelection(
            current_facts=deduped_current,
            evidence=deduped_evidence,
            accepted_state=accepted_state,
        )
    promoted_evidence = _promote_evidence_groups(
        deduped_evidence,
        required_groups=_required_source_groups(evidence_plan),
    )
    citation_order = {
        citation: index
        for index, item in enumerate(promoted_evidence)
        if (citation := _citation(item)) is not None
    }
    promoted_current = sorted(
        deduped_current,
        key=lambda item: _current_fact_selection_key(item, citation_order),
    )
    return CheckoutEvidenceSelection(
        current_facts=promoted_current,
        evidence=promoted_evidence,
        accepted_state=accepted_state,
    )


def build_evidence_set(
    *,
    query: str | None,
    evidence_plan: EvidencePlan | dict[str, object] | None,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> EvidenceSet:
    """Build grouped evidence and sufficiency diagnostics for checkout."""
    groups = evidence_groups(evidence=evidence, current_facts=current_facts)
    status = evidence_plan_status(
        query=query,
        evidence_plan=evidence_plan,
        observed_source_groups=len(groups),
        current_citation_count=sum(1 for fact in current_facts if fact.get("citation")),
    )
    return EvidenceSet(groups=groups, status=status)


def evaluate_evidence_policy(
    *,
    profile: PurposeProfile | dict[str, Any] | str | None,
    query: str | None,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    evidence_set: EvidenceSet | None = None,
) -> EvidencePolicyResult | None:
    """Evaluate hard purpose evidence requirements for model-facing checkout."""
    normalized = purpose_profile(profile)
    requirements = _EVIDENCE_POLICY_REQUIREMENTS.get(normalized.profile)
    if not requirements:
        return None
    haystack = _policy_haystack(
        current_facts=current_facts,
        evidence=evidence,
        evidence_set=evidence_set,
    )
    satisfied: list[str] = []
    missing: list[EvidencePolicyRequirement] = []
    for requirement in requirements:
        if _requirement_satisfied(requirement, haystack):
            satisfied.append(requirement.key)
        else:
            missing.append(requirement)
    suggested_queries = tuple(
        _purpose_refresh_query(requirement, query=query, profile=normalized)
        for requirement in missing
    )
    return EvidencePolicyResult(
        profile=normalized.profile,
        mode=_strongest_mode(requirement.mode for requirement in missing),
        satisfied=not missing,
        requirements=requirements,
        satisfied_requirements=tuple(satisfied),
        missing_requirements=tuple(requirement.key for requirement in missing),
        failure_reasons=tuple(requirement.description for requirement in missing),
        suggested_queries=suggested_queries,
    )


def evidence_plan_status(
    *,
    query: str | None,
    evidence_plan: EvidencePlan | dict[str, object] | None,
    observed_source_groups: int,
    current_citation_count: int,
) -> dict[str, Any] | None:
    """Return whether evidence satisfies the query-level evidence plan."""
    required_groups = _required_source_groups(evidence_plan)
    if required_groups <= 0:
        return None
    observed_groups = observed_source_groups if observed_source_groups else min(1, current_citation_count)
    status: dict[str, Any] = {
        "required_source_groups": required_groups,
        "observed_source_groups": observed_groups,
        "satisfied": observed_groups >= required_groups,
    }
    if not status["satisfied"] and query:
        status["refresh_query"] = f"broader cited evidence for: {query}"
    return status


def evidence_groups(
    *,
    evidence: list[dict[str, Any]],
    current_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group cited evidence by durable source identity."""
    items = evidence if evidence else current_facts
    grouped: dict[str, dict[str, Any]] = {}
    seen_items: set[tuple[str, str, str]] = set()
    for item in items:
        citation = item.get("citation")
        if not isinstance(citation, str) or not citation:
            continue
        source_id = evidence_source_id(item)
        content = evidence_content(item)
        item_key = (source_id, citation, content)
        if item_key in seen_items:
            continue
        seen_items.add(item_key)
        group = grouped.setdefault(
            source_id,
            {
                "source_id": source_id,
                "evidence_count": 0,
                "citations": [],
                "source_lanes": set(),
                "top_score": 0.0,
                "snippet": "",
            },
        )
        group["evidence_count"] += 1
        if citation not in group["citations"]:
            group["citations"].append(citation)
        lane = item.get("source_lane")
        if isinstance(lane, str) and lane:
            group["source_lanes"].add(lane)
        score = _float_metric(item.get("score"))
        if score > group["top_score"]:
            group["top_score"] = score
        if not group["snippet"] and content:
            group["snippet"] = evidence_snippet(content)
    groups = [_finalize_group(group) for group in grouped.values()]
    groups.sort(key=lambda group: (-group["evidence_count"], -group["top_score"], group["source_id"]))
    return groups[:_GROUP_LIMIT]


def evidence_source_id(item: dict[str, Any]) -> str:
    """Return a durable source identity for a checkout evidence item."""
    content = evidence_content(item)
    for pattern in _SOURCE_ID_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group("value").strip()
    source = item.get("source")
    if isinstance(source, str) and source and source not in _GENERIC_SOURCE_NAMES:
        return source
    citation = item.get("citation")
    return citation if isinstance(citation, str) and citation else "unknown"


def evidence_content(item: dict[str, Any]) -> str:
    """Return normalized string content from an evidence item."""
    content = item.get("content")
    return content if isinstance(content, str) else ""


def evidence_snippet(content: str) -> str:
    """Return a bounded one-line evidence snippet."""
    snippet = " ".join(_semantic_evidence_text(content).split())
    if len(snippet) <= _SNIPPET_LIMIT:
        return snippet
    return f"{snippet[: _SNIPPET_LIMIT - 3].rstrip()}..."


def _semantic_evidence_text(content: str) -> str:
    """Prefer remembered text over projection metadata in evidence snippets."""
    for marker in (" role=user | ", " role=assistant | "):
        if marker in content:
            return content.split(marker, 1)[1]
    return content


def _promote_evidence_groups(
    evidence: list[dict[str, Any]],
    *,
    required_groups: int,
) -> list[dict[str, Any]]:
    if required_groups <= 0:
        return evidence
    groups = evidence_groups(evidence=evidence, current_facts=[])
    promoted_citations: list[str] = []
    for group in groups[:required_groups]:
        promoted_citations.extend(_text_list(group.get("citations")))
    citation_rank = {citation: index for index, citation in enumerate(promoted_citations)}
    return sorted(evidence, key=lambda item: _evidence_selection_key(item, citation_rank))


def _select_accepted_state(
    *,
    query: str | None,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select answerable Coordinate state while leaving diagnostics auditable elsewhere."""
    candidates = _dedupe_items([*current_facts, *evidence])
    authoritative = [
        item
        for item in candidates
        if _accepted_state_role(item) == "authoritative"
    ]
    if not authoritative:
        return None
    query_terms = _accepted_state_terms(query or "")
    accepted_terms = {
        term
        for item in authoritative
        for term in _accepted_state_terms(evidence_content(item))
    }
    selected_authority = sorted(
        authoritative,
        key=lambda item: _accepted_state_selection_key(item, query_terms),
    )
    selected_keys = {_item_key(item) for item in selected_authority}
    bridge_rows = [
        item
        for item in candidates
        if _accepted_state_role(item) == "bridge"
        and _item_key(item) not in selected_keys
        and _bridge_supports_accepted_state(item, query_terms=query_terms, accepted_terms=accepted_terms)
    ]
    selected = [*selected_authority, *sorted(bridge_rows, key=lambda item: -_float_metric(item.get("score")))]
    selected = _dedupe_items(selected)
    selected_citations = {
        citation
        for item in selected
        if (citation := _citation(item)) is not None
    }
    return {
        "mode": "coordinate_accepted_state",
        "selected_count": len(selected),
        "diagnostic_count": len(candidates) - len(selected),
        "selected_citations": sorted(selected_citations),
        "current_facts": selected,
        "evidence": [item for item in selected if _citation(item)],
    }


def _accepted_state_role(item: dict[str, Any]) -> str | None:
    if _is_non_answerable_coordinate_row(item):
        return None
    authority = _normalized_policy_text(item.get("authority") or item.get("authority_scope"))
    status = _normalized_policy_text(
        item.get("coordination_status") or item.get("finding_status") or item.get("status")
    )
    text = evidence_content(item).casefold()
    if authority in {"observation", "source", "evidence"}:
        return "bridge"
    if (
        item.get("promoted") is True
        or authority in {
            "accepted",
            "mission-parent",
            "parent",
            "parent-accepted",
            "parent_accepted",
            "parent_accepted_state",
            "promoted",
        }
        or status in {"accepted", "promoted"}
        or status == "current"
        and any(marker in text for marker in ("accepted", "current", "parent", "policy", "root cause", "diagnosis"))
    ):
        return "authoritative"
    if status == "current":
        return "bridge"
    return None


def _is_non_answerable_coordinate_row(item: dict[str, Any]) -> bool:
    authority = _normalized_policy_text(item.get("authority") or item.get("authority_scope"))
    status = _normalized_policy_text(
        item.get("coordination_status") or item.get("finding_status") or item.get("status")
    )
    return (
        item.get("stale") is True
        or status in {"deprecated", "rejected", "stale", "superseded", "unsupported"}
        or (authority.startswith("worker") and item.get("promoted") is not True)
    )


def _accepted_state_selection_key(
    item: dict[str, Any],
    query_terms: set[str],
) -> tuple[int, int, int, float]:
    authority = _normalized_policy_text(item.get("authority") or item.get("authority_scope"))
    status = _normalized_policy_text(
        item.get("coordination_status") or item.get("finding_status") or item.get("status")
    )
    text_terms = _accepted_state_terms(evidence_content(item))
    return (
        0 if item.get("promoted") is True else 1,
        0 if authority in {"parent", "parent-accepted", "parent_accepted", "parent_accepted_state", "mission-parent"} else 1,
        0 if status in {"accepted", "current", "promoted"} else 1,
        -float(len(query_terms & text_terms)) - _float_metric(item.get("score")),
    )


def _bridge_supports_accepted_state(
    item: dict[str, Any],
    *,
    query_terms: set[str],
    accepted_terms: set[str],
) -> bool:
    text_terms = _accepted_state_terms(evidence_content(item))
    if len(text_terms & accepted_terms) >= 2:
        return True
    return bool(query_terms and len(text_terms & query_terms) >= 2)


def _accepted_state_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2 and token not in {"accepted", "current", "event", "events", "source", "state"}
    }


def _normalized_policy_text(value: Any) -> str:
    return str(value or "").strip().casefold().replace(" ", "-")


def _evidence_selection_key(
    item: dict[str, Any],
    citation_rank: dict[str, int],
) -> tuple[int, int, float]:
    citation = _citation(item)
    if citation is not None and citation in citation_rank:
        return (0, citation_rank[citation], -_float_metric(item.get("score")))
    return (1, len(citation_rank), -_float_metric(item.get("score")))


def _current_fact_selection_key(
    item: dict[str, Any],
    citation_rank: dict[str, int],
) -> tuple[int, int, int, float]:
    citation = _citation(item)
    if citation is not None and citation in citation_rank:
        return (0, citation_rank[citation], 0, -_float_metric(item.get("score")))
    if citation is not None:
        return (1, len(citation_rank), 0, -_float_metric(item.get("score")))
    return (2, len(citation_rank), 1, -_float_metric(item.get("score")))


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = _item_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _item_key(item: dict[str, Any]) -> str:
    citation = _citation(item)
    if citation is not None:
        return f"citation:{citation}"
    return f"content:{' '.join(evidence_content(item).split()).casefold()}"


def _citation(item: dict[str, Any]) -> str | None:
    citation = item.get("citation")
    return citation if isinstance(citation, str) and citation else None


def _should_promote_cited_sources(evidence_plan: EvidencePlan | dict[str, object] | None) -> bool:
    if evidence_plan is None:
        return False
    if isinstance(evidence_plan, EvidencePlan):
        return evidence_plan.promote_cited_sources
    promote = evidence_plan.get("promote_cited_sources")
    return promote if isinstance(promote, bool) else False


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    citations = _text_list(group.get("citations"))
    source_lanes = group.get("source_lanes")
    lanes = sorted(source_lanes) if isinstance(source_lanes, set) else []
    return {
        "source_id": str(group["source_id"]),
        "evidence_count": _int_metric(group.get("evidence_count")),
        "citation_count": len(citations),
        "citations": citations[:_CITATION_LIMIT],
        "source_lanes": lanes,
        "top_score": round(_float_metric(group.get("top_score")), 4),
        "snippet": str(group.get("snippet", "")),
    }


def _required_source_groups(evidence_plan: EvidencePlan | dict[str, object] | None) -> int:
    if evidence_plan is None:
        return 0
    if isinstance(evidence_plan, EvidencePlan):
        return evidence_plan.required_source_groups
    return _int_metric(evidence_plan.get("required_source_groups"))


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _int_metric(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float_metric(value: Any) -> float:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _policy_haystack(
    *,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    evidence_set: EvidenceSet | None,
) -> str:
    values: list[str] = []
    for item in [*current_facts, *evidence]:
        for value in item.values():
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, int | float | bool):
                values.append(str(value))
    if evidence_set is not None:
        for group in evidence_set.groups:
            values.extend(
                str(value)
                for value in group.values()
                if isinstance(value, str | int | float | bool)
            )
    return "\n".join(values).casefold()


def _requirement_satisfied(requirement: EvidencePolicyRequirement, haystack: str) -> bool:
    return any(term.casefold() in haystack for term in requirement.terms)


def _strongest_mode(modes: Any) -> str:
    order = {
        "warn": 0,
        "suppress": 1,
        "require_refresh": 2,
        "block_checkout": 3,
    }
    strongest = "warn"
    for mode in modes:
        text = str(mode)
        if order.get(text, -1) > order[strongest]:
            strongest = text
    return strongest


def _purpose_refresh_query(
    requirement: EvidencePolicyRequirement,
    *,
    query: str | None,
    profile: PurposeProfile,
) -> str:
    base = requirement.suggested_query
    if query:
        return f"{base} for {profile.profile}: {query}"
    return f"{base} for {profile.profile}"
