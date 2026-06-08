"""Procedural memory classification helpers for planning diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from zaxy.context import Context

APPLICABLE_STATUSES = frozenset({"validated", "revised", "accepted"})
DIAGNOSTIC_STATUSES = frozenset({"proposed", "pending", "deferred"})
EXCLUDED_STATUSES = frozenset(
    {"rejected", "conflicted", "deprecated", "contradicted", "stale"}
)


def classify_procedure_contexts(
    contexts: Iterable[Context],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Classify procedural context rows into operational and diagnostic buckets.

    Applicable rows are safe to present as candidate procedures, but they are
    still procedural memory and not authoritative facts. Diagnostic rows are
    surfaced only as lifecycle state. Excluded rows preserve rollback and
    contradiction details so callers can explain why a procedure was not used.
    """
    applicable: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    excluded_reasons: dict[str, int] = {}
    available_applicable_count = 0
    procedure_context_count = 0
    safe_limit = max(0, int(limit))

    for context in contexts:
        if not _is_procedure_context(context):
            continue
        procedure_context_count += 1
        item = _procedure_item(context)
        exclusion_reason = _exclusion_reason(context, item)
        if exclusion_reason is not None:
            excluded.append({**item, "excluded_reason": exclusion_reason})
            excluded_reasons[exclusion_reason] = excluded_reasons.get(exclusion_reason, 0) + 1
            continue

        status = str(item["status"])
        if status in DIAGNOSTIC_STATUSES:
            diagnostic.append({**item, "operational_instruction": False})
            continue

        if status in APPLICABLE_STATUSES:
            available_applicable_count += 1
            if len(applicable) < safe_limit:
                applicable.append(item)
            continue

        reason = "unknown_status"
        excluded.append({**item, "excluded_reason": reason})
        excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

    return {
        "applicable": applicable,
        "diagnostic": diagnostic,
        "excluded": excluded,
        "excluded_reasons": excluded_reasons,
        "procedural_memory": {
            "procedure_context_count": procedure_context_count,
            "applicable_count": len(applicable),
            "available_applicable_count": available_applicable_count,
            "diagnostic_count": len(diagnostic),
            "excluded_count": len(excluded),
            "excluded_reasons": dict(excluded_reasons),
            "limit": safe_limit,
            "authority": "non_authoritative_procedural_memory",
        },
    }


def _procedure_item(context: Context) -> dict[str, Any]:
    metadata = dict(context.metadata or {})
    citation = _context_citation(context)
    entity_name = metadata.get("entity_name")
    return {
        "content": context.content,
        "source": context.source,
        "score": context.score,
        "citation": citation,
        "status": _status(metadata),
        "metadata": metadata,
        "skill_id": _skill_id(metadata),
        "version": _version(metadata, entity_name),
        "procedure": _text_list(metadata.get("procedure")),
        "applicability": _text_list(metadata.get("applicability")),
        "summary": _text_or_default(metadata.get("summary"), context.content),
        "rollback": _text_or_default(metadata.get("rollback"), ""),
        "failure_modes": _text_list(metadata.get("failure_modes")),
        "contradiction_reason": _text_or_default(metadata.get("contradiction_reason"), ""),
        "valid_from": context.valid_from,
        "valid_to": context.valid_to,
    }


def _exclusion_reason(context: Context, item: dict[str, Any]) -> str | None:
    status = str(item["status"])
    if status in EXCLUDED_STATUSES:
        return f"{status}_status"
    if _truthy((context.metadata or {}).get("stale")):
        return "stale_flag"
    if not item.get("citation"):
        return "missing_citation"
    if context.valid_to is not None:
        return "valid_to_closed"
    if _has_text((context.metadata or {}).get("superseded_by")):
        return "superseded"
    return None


def _is_procedure_context(context: Context) -> bool:
    metadata = context.metadata or {}
    entity_type = str(metadata.get("entity_type") or "").casefold().strip()
    if entity_type in {"skill_version", "skill_outcome", "procedure", "procedure_candidate"}:
        return True
    if metadata.get("procedure"):
        return True

    source = context.source.casefold()
    candidate_type = str(metadata.get("candidate_type") or metadata.get("kind") or "").casefold()
    event_type = str(metadata.get("event_type") or "").casefold()
    content_prefix = context.content.casefold().split()[:5]
    return (
        "skill" in source
        or "procedure" in source
        or candidate_type == "procedure"
        or "procedure" in event_type
        or "procedure" in content_prefix
    )


def _context_citation(context: Context) -> str:
    metadata = context.metadata or {}
    citation = metadata.get("citation")
    if isinstance(citation, str) and citation.strip():
        return citation.strip()
    citations = metadata.get("citations")
    if isinstance(citations, list):
        for value in citations:
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _status(metadata: dict[str, Any]) -> str:
    for key in ("status", "review_status", "lifecycle_status"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold().replace(" ", "_").replace("-", "_")
    return "unknown"


def _skill_id(metadata: dict[str, Any]) -> str:
    skill_id = metadata.get("skill_id")
    if isinstance(skill_id, str) and skill_id.strip():
        return skill_id.strip()
    entity_name = metadata.get("entity_name")
    if isinstance(entity_name, str) and entity_name.startswith("skill:"):
        return entity_name.removeprefix("skill:").split(":v", 1)[0]
    return ""


def _version(metadata: dict[str, Any], entity_name: object) -> str:
    version = metadata.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    if isinstance(version, int) and not isinstance(version, bool):
        return str(version)
    if isinstance(entity_name, str) and ":v" in entity_name:
        inferred = entity_name.rsplit(":v", 1)[1].strip()
        if inferred:
            return inferred
    return "1"


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            texts.append(text)
    return texts


def _text_or_default(value: object, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "stale"}
    return False


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
