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
    return SynthesisPacket(
        answer_candidates=packet.answer_candidates or fallback.answer_candidates,
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
    answer_key: str | None = None
    for line in content.splitlines():
        for match in _SYNTHESIS_KV_RE.finditer(line):
            key = match.group("key")
            value = match.group("value")
            fields[key] = value
            if answer_key is None and key.endswith("_answer") and not key.endswith("_answer_text"):
                answer_key = key
    answer = _line_value(content, answer_key) if answer_key else None
    if answer is None:
        answer = fields.get(answer_key or "")
    if answer is None:
        return None
    rank = _parse_candidate_int(fields.get("candidate_rank"), default=1)
    candidate_type = fields.get("candidate_type") or _answer_candidate_type(answer_key)
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


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            str(candidate.get("type")),
        )
    )
    return deduped


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
