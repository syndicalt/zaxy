"""Core memory dataclasses (assembly, checkout, page, handoff) + token math."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zaxy.context import Context
from zaxy.event import (  # noqa: F401 - ReplayResult re-export for existing tests
    EventLog,
    IntegrityReport,
    ReplayResult,
    verify_event_chain,
)
from zaxy.recall import RecallCandidateSet, empty_recall_candidate_set


@dataclass(frozen=True)
class ContextAssembly:
    """Prompt-ready assembled context from replay plus retrieval."""

    session_id: str
    prompt: str
    contexts: list[Context]
    replay_event_count: int
    compacted: bool = False
    warnings: list[str] = field(default_factory=list)
    assembly_policy: dict[str, bool | int] = field(default_factory=dict)
    context_counts: dict[str, int] = field(default_factory=dict)
    working_set: dict[str, object] = field(default_factory=dict)
    recall: RecallCandidateSet = field(default_factory=empty_recall_candidate_set)
    #: Full as-of-filtered replay the assembly was computed against. Carried so
    #: checkout can resolve citations to sealed event refs and replay salience
    #: without re-reading the log; never serialized into payloads.
    replay_events: list[Any] = field(default_factory=list)
    #: Enrollment-gated, cited, non-authoritative fleet-memory lane contexts. Off
    #: by default (``fleet_enabled``) and empty unless the agent is enrolled in a
    #: requested fleet; surfaced as a distinct ``fleet`` checkout lane.
    fleet_contexts: list[Context] = field(default_factory=list)
    #: Consolidated remote-tier contexts for the two-tier long-horizon checkout.
    #: Off by default (``long_horizon_enabled`` / the ``long_horizon`` param) and
    #: empty unless engaged with a session that exceeds the recent window; each is
    #: a cited, non-authoritative consolidation candidate for older history.
    long_horizon_contexts: list[Context] = field(default_factory=list)
    #: Episodic/consolidated split summary (``enabled``, ``recent_window``,
    #: ``episodic_count``, ``horizon_split_seq``) when two-tier assembly is
    #: engaged; ``None`` when off, keeping the checkout byte-identical to today.
    long_horizon: dict[str, Any] | None = None


@dataclass(frozen=True)
class MemoryCheckout:
    """Cited, prompt-ready current memory state for an agent turn."""

    session_id: str
    query: str
    prompt: str
    working_set: dict[str, object]
    ref: dict[str, object] | None
    current_facts: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    retention: dict[str, Any]
    warnings: list[str]
    guidance: dict[str, Any]
    quality: dict[str, Any]
    diagnostics: dict[str, Any]
    context_counts: dict[str, int]
    replay_event_count: int
    compacted: bool = False
    assembly_policy: dict[str, bool | int] = field(default_factory=dict)
    purpose: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable payload for tools and CLIs."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "prompt": self.prompt,
            "working_set": self.working_set,
            "ref": self.ref,
            "current_facts": self.current_facts,
            "evidence": self.evidence,
            "provenance": self.provenance,
            "retention": self.retention,
            "warnings": self.warnings,
            "guidance": self.guidance,
            "quality": self.quality,
            "diagnostics": self.diagnostics,
            "context_counts": self.context_counts,
            "replay_event_count": self.replay_event_count,
            "token_efficiency": checkout_token_efficiency(
                prompt=self.prompt,
                current_fact_count=len(self.current_facts),
                evidence_count=len(self.evidence),
            ),
            "compacted": self.compacted,
            "assembly_policy": self.assembly_policy,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class QueryPage:
    """A stable page of ranked memory query results."""

    contexts: list[Context]
    next_cursor: str | None
    cursor: str | None
    has_more: bool
    offset: int

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable pagination payload."""
        return {
            "contexts": [
                {
                    "content": context.content,
                    "source": context.source,
                    "score": context.score,
                    "valid_from": context.valid_from,
                    "valid_to": context.valid_to,
                    "metadata": context.metadata,
                }
                for context in self.contexts
            ],
            "next_cursor": self.next_cursor,
            "cursor": self.cursor,
            "has_more": self.has_more,
            "offset": self.offset,
        }


def checkout_token_efficiency(
    *,
    prompt: str,
    current_fact_count: int,
    evidence_count: int,
) -> dict[str, int | float]:
    """Estimate Memory Checkout token efficiency for activation diagnostics."""
    prompt_tokens = _approx_tokens(prompt)
    return {
        "prompt_tokens": prompt_tokens,
        "current_fact_count": current_fact_count,
        "evidence_count": evidence_count,
        "facts_per_1k_prompt_tokens": round((current_fact_count / prompt_tokens) * 1000, 3)
        if prompt_tokens
        else 0.0,
    }


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class HandoffBundle:
    """Portable handoff package for resuming a session or subagent."""

    session_id: str
    summary: dict[str, Any]
    prompt: str
    contexts: list[Context]
    replay_event_count: int
    integrity_ok: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable handoff payload."""
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "prompt": self.prompt,
            "contexts": [
                {
                    "content": context.content,
                    "source": context.source,
                    "score": context.score,
                    "valid_from": context.valid_from,
                    "valid_to": context.valid_to,
                    "metadata": context.metadata,
                }
                for context in self.contexts
            ],
            "replay_event_count": self.replay_event_count,
            "integrity_ok": self.integrity_ok,
        }


@dataclass(frozen=True)
class ContextRefreshReport:
    """Result of an incremental source refresh."""

    session_id: str
    kind: str
    event_count: int
    summary: dict[str, int | str]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable payload."""
        return {
            "session_id": self.session_id,
            "kind": self.kind,
            "event_count": self.event_count,
            "summary": self.summary,
        }
