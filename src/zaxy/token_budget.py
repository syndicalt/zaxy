"""Deterministic token budgeting for prompt-packet section packing.

This module owns the 2.1 budgeted-checkout primitives: a dependency-free token
estimator and a deterministic, monotone section packer. ``estimate_tokens``
deliberately ships as ``ceil(chars / 4)`` only in this release (decision
recorded in the 2.1-2.3 implementation plan): no network calls, no optional
tokenizer dependencies in the hot path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Return a deterministic token estimate: ``ceil(len(text) / 4)``."""
    return (len(text) + 3) // 4


@dataclass(frozen=True)
class BudgetSection:
    """One packable prompt section with budget metadata.

    ``weight`` is the section's salience proxy (section priority until the 2.2
    salience ledger lands). ``mandatory`` marks headers and trust-contract
    lines that must survive any budget.
    """

    section_id: str
    kind: str
    text: str
    weight: float = 0.5
    mandatory: bool = False


@dataclass(frozen=True)
class ElisionRecord:
    """One section excluded from a packed prompt by the token budget."""

    section_id: str
    kind: str
    estimated_tokens: int

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable elision record."""
        return {
            "section_id": self.section_id,
            "kind": self.kind,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(frozen=True)
class PackResult:
    """Deterministic packing outcome for one budget request.

    ``sections`` preserves the caller's original section order. ``budget_used``
    is the estimated token total of every included section; it can exceed
    ``budget_requested`` only when mandatory sections alone overflow the budget.
    """

    sections: list[BudgetSection]
    elided: list[ElisionRecord]
    budget_requested: int
    budget_used: int


def pack_sections(sections: Sequence[BudgetSection], max_tokens: int) -> PackResult:
    """Pack sections into ``max_tokens`` with a monotone greedy policy.

    Mandatory sections are always included. Optional sections are considered in
    descending weight-per-token order (original index breaks ties) and admitted
    as the longest prefix of that order whose cumulative estimated cost fits
    the budget remaining after mandatory sections. Stopping at the first
    section that does not fit makes packing monotone: raising the budget can
    only add sections, never remove a previously included one.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    costs = [estimate_tokens(section.text) for section in sections]
    included = {index for index, section in enumerate(sections) if section.mandatory}
    remaining = max_tokens - sum(costs[index] for index in included)
    optional_by_density = sorted(
        (index for index, section in enumerate(sections) if not section.mandatory),
        key=lambda index: (-_token_density(sections[index].weight, costs[index]), index),
    )
    for index in optional_by_density:
        if costs[index] > remaining:
            break
        included.add(index)
        remaining -= costs[index]
    packed = [section for index, section in enumerate(sections) if index in included]
    elided = [
        ElisionRecord(section_id=section.section_id, kind=section.kind, estimated_tokens=costs[index])
        for index, section in enumerate(sections)
        if index not in included
    ]
    return PackResult(
        sections=packed,
        elided=elided,
        budget_requested=max_tokens,
        budget_used=sum(costs[index] for index in included),
    )


def _token_density(weight: float, cost: int) -> float:
    if cost <= 0:
        return float("inf")
    return weight / cost
