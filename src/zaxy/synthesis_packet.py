"""Typed synthesis packet normalization for checkout and artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_SYNTHESIS_KV_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)")

_CANDIDATE_FIELDS = (
    "rank",
    "type",
    "confidence",
    "answer_key",
    "answer",
    "support_source_ids",
    "excluded_source_ids",
)
_LEDGER_ROW_FIELDS = (
    "fact_id",
    "source_group",
    "citation",
    "kind",
    "entity",
    "value",
    "unit",
    "time",
    "label",
    "raw_span",
    "normalized_identity",
    "include_reason",
    "exclude_reason",
    "confidence",
)


@dataclass(frozen=True)
class SynthesisPacket:
    """Normalized answer candidates and ledger rows from synthesis context."""

    answer_candidates: list[dict[str, Any]]
    ledger_rows: list[dict[str, Any]]
    operations: list[dict[str, Any]]
    result: dict[str, Any]


def synthesis_packet_from_items(items: list[dict[str, Any]]) -> SynthesisPacket:
    """Build a typed packet from checkout facts/evidence with legacy fallback parsing."""
    packet = _packet_from_item_metadata(items)
    fallback = _packet_from_rendered_content(items)
    if packet is None:
        return fallback
    aggregate_total_query = _aggregate_total_query_text(_query_text_from_items(items))
    return SynthesisPacket(
        answer_candidates=_dedupe_candidates(
            [
                *packet.answer_candidates,
                *_additive_fallback_candidates(packet.answer_candidates, fallback.answer_candidates),
            ],
            aggregate_total_query=aggregate_total_query,
        ),
        ledger_rows=_dedupe_ledger_rows([*packet.ledger_rows, *fallback.ledger_rows]),
        operations=packet.operations or fallback.operations,
        result=packet.result or fallback.result,
    )


def synthesis_packet_from_diagnostics(diagnostics: dict[str, Any]) -> SynthesisPacket:
    """Normalize diagnostics.synthesis packet data into the stable public shape."""
    synthesis = diagnostics.get("synthesis")
    if not isinstance(synthesis, dict):
        return SynthesisPacket(answer_candidates=[], ledger_rows=[], operations=[], result={})
    packet = _typed_packet_value(synthesis)
    source = packet if packet is not None else synthesis
    return SynthesisPacket(
        answer_candidates=_clean_candidates(source.get("answer_candidates")),
        ledger_rows=_clean_ledger_rows(source.get("ledger_rows")),
        operations=_clean_operations(source.get("operations")),
        result=_clean_result(source.get("result")),
    )


def _packet_from_item_metadata(items: list[dict[str, Any]]) -> SynthesisPacket | None:
    candidates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    for item in items:
        packet = _typed_packet_value(item)
        if packet is None:
            continue
        candidates.extend(_clean_candidates(packet.get("answer_candidates")))
        rows.extend(_clean_ledger_rows(packet.get("ledger_rows")))
        operations.extend(_clean_operations(packet.get("operations")))
        if not result:
            result = _clean_result(packet.get("result"))
    if not candidates and not rows and not operations and not result:
        return None
    return SynthesisPacket(
        answer_candidates=_dedupe_candidates(candidates),
        ledger_rows=_dedupe_ledger_rows(rows),
        operations=_dedupe_operations(operations),
        result=result,
    )


def _typed_packet_value(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("synthesis_packet", "typed_packet", "packet"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload if any(key in payload for key in ("answer_candidates", "ledger_rows", "operations", "result")) else None


def _packet_from_rendered_content(items: list[dict[str, Any]]) -> SynthesisPacket:
    candidates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_candidates: set[tuple[int, str, str]] = set()
    seen_rows: set[str] = set()
    for item in items:
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if "zaxy_synthesis_bundle=true" in content:
            candidate = _candidate_from_content(content)
            if candidate is not None:
                identity = (
                    _parse_candidate_int(candidate.get("rank"), default=1),
                    str(candidate.get("type")),
                    str(candidate.get("answer")),
                )
                if identity not in seen_candidates:
                    seen_candidates.add(identity)
                    candidates.append(candidate)
        if "ledger_row=" not in content:
            continue
        for line in content.splitlines():
            if not line.startswith("ledger_row="):
                continue
            try:
                payload = json.loads(line.removeprefix("ledger_row="))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            row = _clean_ledger_row(payload)
            fact_id = row.get("fact_id")
            if not row or not isinstance(fact_id, str) or fact_id in seen_rows:
                continue
            seen_rows.add(fact_id)
            rows.append(row)
    candidates.sort(
        key=lambda candidate: (
            _parse_candidate_int(candidate.get("rank"), default=1),
            str(candidate.get("type")),
        )
    )
    return SynthesisPacket(answer_candidates=candidates, ledger_rows=rows, operations=[], result={})


def _candidate_from_content(content: str) -> dict[str, Any] | None:
    fields: dict[str, str] = {}
    for line in content.splitlines():
        for match in _SYNTHESIS_KV_RE.finditer(line):
            key = match.group("key")
            value = match.group("value")
            fields[key] = value
    answer_key = _preferred_answer_key(fields, content=content)
    answer = _line_value(content, answer_key) if answer_key else None
    if answer is None:
        answer = fields.get(answer_key or "")
    if answer is None:
        return None
    rank = _parse_candidate_int(fields.get("candidate_rank"), default=1)
    candidate_type = fields.get("candidate_type") or _answer_candidate_type(answer_key)
    if answer_key == "direct_numeric_answer" and not _direct_numeric_fallback_allowed(content):
        return None
    confidence = _parse_candidate_float(fields.get("candidate_confidence"))
    return {
        "rank": rank,
        "type": candidate_type,
        "confidence": round(confidence, 2),
        "answer_key": answer_key,
        "answer": answer,
        "support_source_ids": _split_csv(fields.get("candidate_support")),
        "excluded_source_ids": _split_csv(_first_field_suffix(fields, "_excluded_source_ids")),
    }


def _direct_numeric_fallback_allowed(content: str) -> bool:
    """Allow direct scalar fallback only for update-state, not broad aggregate, queries."""
    query = _line_value(content, "query") or content
    lowered = query.casefold()
    if re.search(r"\b(?:currently|so far|since|each day|daily|finished|collection|followers?)\b", lowered):
        return True
    return bool(re.search(r"\b(?:sessions?|issues?|tops?|stories?)\b", lowered) and not re.search(
        r"\btotal\b",
        lowered,
    ))


def _preferred_answer_key(fields: dict[str, str], *, content: str = "") -> str | None:
    if elapsed_total_key := _elapsed_total_answer_key(fields, query=_line_value(content, "query") or ""):
        return elapsed_total_key
    keys = [
        key
        for key in fields
        if key.endswith("_answer") or key.endswith("_answer_text")
    ]
    if not keys:
        return None
    if _aggregate_total_query_text(_line_value(content, "query") or ""):
        total_keys = [key for key in keys if key.endswith("_total_answer")]
        if total_keys:
            total_keys.sort(key=_answer_key_sort_key)
            return total_keys[0]
    keys.sort(key=_answer_key_sort_key)
    return keys[0]


def _elapsed_total_answer_key(fields: dict[str, str], *, query: str) -> str | None:
    """Return the aggregate elapsed-time surface for how-many-ago queries."""
    query_text = " ".join(query.casefold().split())
    match = re.search(r"\bhow\s+many\s+(?P<unit>days?|weeks?|months?)\s+ago\b", query_text)
    if not match:
        return None
    unit = match.group("unit").removesuffix("s")
    for key in (f"{unit}_total_words", f"{unit}_total"):
        if key in fields:
            return key
    return None


def _query_text_from_items(items: list[dict[str, Any]]) -> str:
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            query = _line_value(content, "query")
            if query:
                return query
    return ""


def _aggregate_total_query_text(query: str) -> bool:
    query_text = " ".join(query.casefold().split())
    return bool(re.search(r"\b(?:total|combined|altogether|sum)\b|\bin\s+total\b", query_text))


def _answer_key_sort_key(key: str) -> tuple[int, str]:
    if key in {"percentage_answer", "boolean_comparison_answer"}:
        return -1, key
    if key in {"day_total_words", "day_total", "week_total_words", "week_total", "month_total_words", "month_total"}:
        return 0, key
    if key in {
        "latest_state_answer",
        "query_bound_direct_answer",
        "query_bound_difference_answer",
        "query_bound_scalar_total_answer",
        "road_trip_drive_total_answer",
        "routine_time_total_answer",
        "social_media_break_total_answer",
        "relative_temporal_anchor_answer",
    }:
        return 0, key
    if key.endswith("_answer_text"):
        return 1, key
    if key in {
        "day_interval_answer",
        "week_interval_answer",
        "month_interval_answer",
        "year_interval_answer",
    }:
        return 2, key
    if key.endswith("_total_answer"):
        return 3, key
    if key in {
        "future_age_at_event_answer",
        "page_total_answer",
        "distance_total_answer",
        "pages_remaining_answer",
        "boolean_evidence_answer",
        "property_outcome_answer",
        "instrument_ownership_answer",
        "temporal_order_answer",
        "recency_answer",
        "direct_answer",
    }:
        return 4, key
    if key == "assistant_recall_answer":
        return 5, key
    return 6, key


def _clean_candidates(raw_candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        cleaned = {
            key: _json_value(candidate[key])
            for key in _CANDIDATE_FIELDS
            if key in candidate and _json_value(candidate[key]) is not None
        }
        if cleaned:
            candidates.append(cleaned)
    return _dedupe_candidates(candidates)


def _additive_fallback_candidates(
    typed_candidates: list[dict[str, Any]],
    fallback_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep rendered answer surfaces that add a distinct candidate operation."""
    typed_identities = {
        (
            str(candidate.get("type", "")),
            str(candidate.get("answer_key", "")),
        )
        for candidate in typed_candidates
    }
    typed_blocks = {
        (
            _parse_candidate_int(candidate.get("rank"), default=1),
            str(candidate.get("type", "")),
        )
        for candidate in typed_candidates
    }
    additive: list[dict[str, Any]] = []
    for candidate in fallback_candidates:
        identity = (
            str(candidate.get("type", "")),
            str(candidate.get("answer_key", "")),
        )
        if identity in typed_identities:
            continue
        block = (
            _parse_candidate_int(candidate.get("rank"), default=1),
            str(candidate.get("type", "")),
        )
        if block in typed_blocks:
            continue
        additive.append(candidate)
    return additive


def _clean_ledger_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        cleaned = _clean_ledger_row(row)
        if cleaned:
            rows.append(cleaned)
    return _dedupe_ledger_rows(rows)


def _clean_operations(raw_operations: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_operations, list):
        return []
    operations: list[dict[str, Any]] = []
    for operation in raw_operations:
        if not isinstance(operation, dict):
            continue
        cleaned = _json_value(operation)
        if isinstance(cleaned, dict) and cleaned:
            operations.append(cleaned)
    return _dedupe_operations(operations)


def _clean_result(raw_result: Any) -> dict[str, Any]:
    cleaned = _json_value(raw_result)
    return cleaned if isinstance(cleaned, dict) else {}


def _clean_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _json_value(row[key])
        for key in _LEDGER_ROW_FIELDS
        if key in row and _json_value(row[key]) is not None
    }


def _dedupe_candidates(
    candidates: list[dict[str, Any]],
    *,
    aggregate_total_query: bool = False,
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for candidate in candidates:
        identity = (
            _parse_candidate_int(candidate.get("rank"), default=1),
            str(candidate.get("type")),
            str(candidate.get("answer")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(candidate)
    deduped.sort(
        key=lambda candidate: (
            _parse_candidate_int(candidate.get("rank"), default=1),
            _aggregate_total_candidate_sort_key(candidate, aggregate_total_query=aggregate_total_query),
            _answer_key_sort_key(str(candidate.get("answer_key", ""))),
            str(candidate.get("type")),
        )
    )
    return deduped


def _aggregate_total_candidate_sort_key(candidate: dict[str, Any], *, aggregate_total_query: bool) -> int:
    if not aggregate_total_query:
        return 0
    answer_key = str(candidate.get("answer_key", ""))
    return 0 if answer_key.endswith("_total_answer") else 1


def _dedupe_ledger_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        fact_id = row.get("fact_id")
        if not isinstance(fact_id, str) or fact_id in seen:
            continue
        seen.add(fact_id)
        deduped.append(row)
    return deduped


def _dedupe_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for operation in operations:
        identity = json.dumps(operation, sort_keys=True, separators=(",", ":"))
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(operation)
    return deduped


def _json_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _json_value(item)) is not None]
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, item in value.items()
            if (cleaned := _json_value(item)) is not None
        }
    return None


def _answer_candidate_type(answer_key: str | None) -> str:
    if not answer_key:
        return "unknown"
    if answer_key in {"day_total_words", "day_total", "week_total_words", "week_total", "month_total_words", "month_total"}:
        return "duration"
    return answer_key.removesuffix("_answer")


def _first_field_suffix(fields: dict[str, str], suffix: str) -> str | None:
    for key, value in fields.items():
        if key.endswith(suffix):
            return value
    return None


def _line_value(content: str, key: str | None) -> str | None:
    if not key:
        return None
    prefix = f"{key}="
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return None


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in (item.strip() for item in value.split(",")) if part]


def _parse_candidate_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_candidate_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
