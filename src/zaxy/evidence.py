"""Production evidence building for model-facing memory checkout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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


def select_checkout_evidence(
    *,
    query: str | None,
    evidence_plan: EvidencePlan | dict[str, object] | None,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> CheckoutEvidenceSelection:
    """Select and order checkout facts/evidence for the query's evidence plan."""
    del query
    deduped_current = _dedupe_items(current_facts)
    deduped_evidence = _dedupe_items([item for item in evidence if _citation(item)])
    if not _should_promote_cited_sources(evidence_plan):
        return CheckoutEvidenceSelection(
            current_facts=deduped_current,
            evidence=deduped_evidence,
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
