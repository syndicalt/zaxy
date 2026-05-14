"""Typed evidence candidates for retrieval-time answer synthesis."""

from __future__ import annotations

from dataclasses import dataclass

from zaxy.synthesis import (
    build_count_ledger,
    build_currency_ledger,
    build_date_ledger,
    build_duration_ledger,
    render_count_result,
    render_currency_result,
    render_date_interval_result,
    render_duration_result,
)


@dataclass(frozen=True)
class EvidenceProjection:
    """Rendered evidence candidates plus the source groups that support them."""

    lines: tuple[str, ...]
    source_groups: tuple[str, ...]


def aggregate_candidate_projection(query: str, contexts: list[str]) -> EvidenceProjection:
    """Build deterministic aggregate answer candidates from cited contexts."""
    lines: list[str] = []
    source_groups: list[str] = []
    rank = 1
    count_ledger = build_count_ledger(query, contexts)
    count_projection = render_count_result(count_ledger, query, rank=rank)
    if count_projection.lines:
        lines.extend(count_projection.lines)
        source_groups.extend(count_projection.support_source_groups)
        rank += 1
    currency_ledger = build_currency_ledger(query, contexts)
    currency_projection = render_currency_result(currency_ledger, rank=rank)
    if currency_projection.lines:
        lines.extend(currency_projection.lines)
        source_groups.extend(currency_projection.support_source_groups)
        rank += 1
    duration_ledger = build_duration_ledger(query, contexts)
    duration_projection = render_duration_result(duration_ledger, rank=rank)
    if duration_projection.lines:
        lines.extend(duration_projection.lines)
        source_groups.extend(duration_projection.support_source_groups)
        rank += 1
    date_ledger = build_date_ledger(query, contexts)
    date_projection = render_date_interval_result(date_ledger, rank=rank)
    if date_projection.lines:
        lines.extend(date_projection.lines)
        source_groups.extend(date_projection.support_source_groups)
    return EvidenceProjection(
        lines=tuple(lines),
        source_groups=tuple(dict.fromkeys(source_groups)),
    )


def aggregate_candidate_lines(query: str, contexts: list[str]) -> list[str]:
    """Render deterministic aggregate answer candidates from cited contexts."""
    return list(aggregate_candidate_projection(query, contexts).lines)
