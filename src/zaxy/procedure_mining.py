"""Deterministic procedure mining over captured tool-call traces.

This module reads ``tool.call.completed`` and ``command.completed`` lifecycle
events from an Eventloom log (sessions are Eventloom threads in one shared
log, so every citation stays resolvable by ``seq``/``hash``), mines recurring
successful tool-name n-grams across distinct sessions, and emits the results
as review-pending ``procedure`` consolidation candidates through the existing
consolidation pipeline. Mined procedures are never authoritative; acceptance
flows through the normal consolidation review path.

Contract details:

- A ``tool.call.completed`` step is successful when ``payload.status`` equals
  ``"succeeded"`` (the value the MCP capture producer emits; everything else,
  including ``"failed"`` or a missing status, breaks the sequence).
- A ``command.completed`` step is successful when ``payload.exit_code`` is
  ``0`` (falling back to ``payload.outcome == "passed"`` when no integer exit
  code is present).
- Confidence is derived from support: support 2 maps to 0.50, each additional
  supporting session adds 0.05, capped at 0.85. Mined procedures never reach
  1.0 because they are review-gated proposals, not verified knowledge.
- Citations cover every contributing occurrence up to
  ``MAX_CITED_OCCURRENCES``; the earliest occurrence in each supporting
  session is always cited first so the cross-session evidence survives the
  cap.
- Idempotency leans on the consolidation pipeline's deterministic
  ``candidate_id`` (a hash of candidate type, title, and cited source
  events): re-running the miner over an unchanged log rebuilds identical
  candidate ids, which are skipped when already present in the log.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zaxy.consolidation_pipeline import (
    ConsolidationSegment,
    ProposedConsolidation,
    build_segment_id,
    event_type_counts,
)
from zaxy.event import EventLog
from zaxy.security import validate_session_id

MIN_PROCEDURE_LENGTH = 2
MAX_PROCEDURE_LENGTH = 8
MINING_MIN_SUPPORT = 2
MAX_CITED_OCCURRENCES = 8
PROCEDURE_MINING_METHOD = "procedure-mining/ngram-v1"
DEFAULT_MINING_ACTOR = "zaxy-procedure-miner"

TRACE_EVENT_TYPES = frozenset({"tool.call.completed", "command.completed"})

_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_TITLE_MAX_CHARS = 200
_SUMMARY_MAX_CHARS = 240
_CONFIDENCE_BASE = 0.5
_CONFIDENCE_PER_EXTRA_SESSION = 0.05
_CONFIDENCE_CAP = 0.85


@dataclass(frozen=True)
class TraceStep:
    """One normalized tool-call or command completion step in a session."""

    session_id: str
    seq: int
    hash: str
    event_type: str
    step: str
    successful: bool


@dataclass(frozen=True)
class ProcedureOccurrence:
    """One contiguous successful occurrence of a mined step sequence."""

    session_id: str
    steps: tuple[TraceStep, ...]

    @property
    def first_seq(self) -> int:
        return self.steps[0].seq


@dataclass(frozen=True)
class MinedProcedure:
    """A recurring successful step sequence with its supporting evidence."""

    steps: tuple[str, ...]
    support_sessions: tuple[str, ...]
    occurrences: tuple[ProcedureOccurrence, ...]

    @property
    def support(self) -> int:
        return len(self.support_sessions)


@dataclass(frozen=True)
class AppendedProposal:
    """A procedure candidate event appended by ``mine_and_propose``."""

    candidate_id: str
    seq: int
    hash: str
    session_id: str


@dataclass(frozen=True)
class ProcedureMiningSummary:
    """Outcome of one ``mine_and_propose`` batch pass."""

    session_ids: tuple[str, ...]
    mined_count: int
    appended_count: int
    skipped_duplicate_count: int
    mined: tuple[MinedProcedure, ...]
    appended: tuple[AppendedProposal, ...]
    skipped_candidate_ids: tuple[str, ...]


def extract_session_traces(
    events: Sequence[Any],
    *,
    session_ids: Sequence[str] | None = None,
) -> dict[str, tuple[TraceStep, ...]]:
    """Extract per-session ordered tool-call/command traces from log events.

    Sessions are Eventloom threads. Events that are not tool-call or command
    completions are ignored; trace events with malformed envelopes (bad seq or
    hash) raise. Payload-level problems (missing tool name, missing status)
    are treated as unsuccessful steps so they break sequences instead of
    silently merging their neighbours.
    """
    allowed: set[str] | None = None
    if session_ids is not None:
        allowed = {validate_session_id(session_id) for session_id in session_ids}

    traces: dict[str, list[TraceStep]] = {}
    for event in events:
        event_type = getattr(event, "type", None)
        if event_type not in TRACE_EVENT_TYPES:
            continue
        session_id = _event_session_id(event)
        if allowed is not None and session_id not in allowed:
            continue
        seq = getattr(event, "seq", None)
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
            raise ValueError("trace event seq must be a positive integer")
        event_hash = getattr(event, "hash", None)
        if not isinstance(event_hash, str) or _EVENT_HASH_RE.fullmatch(event_hash) is None:
            raise ValueError("trace event hash must be 64 lowercase hex")
        name, successful = _normalize_step(event_type, getattr(event, "payload", None))
        traces.setdefault(session_id, []).append(
            TraceStep(
                session_id=session_id,
                seq=seq,
                hash=event_hash,
                event_type=event_type,
                step=name if name is not None else event_type,
                successful=successful and name is not None,
            )
        )

    return {
        session_id: tuple(sorted(traces[session_id], key=lambda step: step.seq))
        for session_id in sorted(traces)
    }


def mine_procedures(
    traces: Mapping[str, Sequence[TraceStep]],
    *,
    min_support: int = MINING_MIN_SUPPORT,
    max_length: int = MAX_PROCEDURE_LENGTH,
    min_length: int = MIN_PROCEDURE_LENGTH,
) -> list[MinedProcedure]:
    """Mine recurring successful step n-grams across distinct sessions.

    Failed (or unnameable) steps split each session trace into maximal
    successful runs; n-grams never span a failure. An n-gram is mined when it
    occurs in at least ``min_support`` distinct sessions. Shorter n-grams that
    are contiguous sub-sequences of a longer mined n-gram with an identical
    supporting-session set are subsumed and dropped. Output ordering is
    deterministic: support descending, then length descending, then
    lexicographic on the step names.
    """
    _validate_min_support(min_support)
    _validate_lengths(min_length=min_length, max_length=max_length)

    occurrences_by_ngram: dict[tuple[str, ...], list[ProcedureOccurrence]] = {}
    for session_id in sorted(traces):
        for run in _successful_runs(traces[session_id]):
            longest = min(max_length, len(run))
            for length in range(min_length, longest + 1):
                for start in range(len(run) - length + 1):
                    window = tuple(run[start : start + length])
                    key = tuple(step.step for step in window)
                    occurrences_by_ngram.setdefault(key, []).append(
                        ProcedureOccurrence(session_id=session_id, steps=window)
                    )

    mined: list[MinedProcedure] = []
    for steps, occurrences in occurrences_by_ngram.items():
        support_sessions = tuple(sorted({occurrence.session_id for occurrence in occurrences}))
        if len(support_sessions) < min_support:
            continue
        ordered = tuple(
            sorted(occurrences, key=lambda occurrence: (occurrence.first_seq, occurrence.session_id))
        )
        mined.append(
            MinedProcedure(steps=steps, support_sessions=support_sessions, occurrences=ordered)
        )

    kept = _drop_subsumed(mined)
    kept.sort(key=lambda procedure: (-procedure.support, -len(procedure.steps), procedure.steps))
    return kept


def build_procedure_proposal(
    procedure: MinedProcedure,
    *,
    purpose: str | None = None,
    max_cited_occurrences: int = MAX_CITED_OCCURRENCES,
) -> ProposedConsolidation:
    """Build a review-pending procedure proposal for one mined sequence.

    The proposal cites the trace events of every contributing occurrence up
    to ``max_cited_occurrences`` occurrences; the earliest occurrence in each
    supporting session is always cited first so the cross-session support
    evidence survives the cap. The candidate is emitted under the session of
    the earliest contributing occurrence.
    """
    if (
        not isinstance(max_cited_occurrences, int)
        or isinstance(max_cited_occurrences, bool)
        or max_cited_occurrences < 1
    ):
        raise ValueError("max_cited_occurrences must be a positive integer")

    cited = _cited_occurrences(procedure, max_cited_occurrences)
    rows: dict[int, dict[str, Any]] = {}
    for occurrence in cited:
        for step in occurrence.steps:
            rows.setdefault(
                step.seq,
                {
                    "seq": step.seq,
                    "hash": step.hash,
                    "event_type": step.event_type,
                    "summary": step.step,
                },
            )
    ordered_rows = [rows[seq] for seq in sorted(rows)]

    session_id = cited[0].session_id
    segment = ConsolidationSegment(
        session_id=session_id,
        segment_id=build_segment_id(session_id, [int(row["seq"]) for row in ordered_rows]),
        event_type_counts=event_type_counts(ordered_rows),
        source_events=ordered_rows,
    )
    return ProposedConsolidation(
        segment=segment,
        candidate_type="procedure",
        title=_procedure_title(procedure),
        summary=_procedure_summary(procedure),
        confidence=confidence_from_support(procedure.support),
        method=PROCEDURE_MINING_METHOD,
        purpose=purpose,
    )


def confidence_from_support(support: int) -> float:
    """Map cross-session support to proposal confidence.

    Support 2 maps to 0.50; each additional supporting session adds 0.05; the
    result is capped at 0.85 because mined procedures stay review-gated.
    """
    if not isinstance(support, int) or isinstance(support, bool) or support < MINING_MIN_SUPPORT:
        raise ValueError(f"support must be an integer >= {MINING_MIN_SUPPORT}")
    raw = _CONFIDENCE_BASE + _CONFIDENCE_PER_EXTRA_SESSION * (support - MINING_MIN_SUPPORT)
    return round(min(raw, _CONFIDENCE_CAP), 2)


def mine_and_propose(
    eventlog: EventLog,
    *,
    session_ids: Sequence[str] | None = None,
    min_support: int = MINING_MIN_SUPPORT,
    max_length: int = MAX_PROCEDURE_LENGTH,
    actor: str = DEFAULT_MINING_ACTOR,
    purpose: str | None = None,
    max_cited_occurrences: int = MAX_CITED_OCCURRENCES,
) -> ProcedureMiningSummary:
    """Mine recurring procedures from a log and append candidate proposals.

    This is read-only over the log except for the review-pending
    ``consolidation.candidate.created`` appends, which use the pipeline's
    normal candidate event shape. Re-running over the same log is idempotent:
    candidates whose deterministic ``candidate_id`` already exists in the log
    are reported as skipped duplicates instead of being appended again.
    """
    events = eventlog.read_all()
    traces = extract_session_traces(events, session_ids=session_ids)
    mined = mine_procedures(traces, min_support=min_support, max_length=max_length)
    existing_candidate_ids = _existing_candidate_ids(events)

    appended: list[AppendedProposal] = []
    skipped: list[str] = []
    for procedure in mined:
        proposal = build_procedure_proposal(
            procedure,
            purpose=purpose,
            max_cited_occurrences=max_cited_occurrences,
        )
        event_spec = proposal.to_candidate_event(actor=actor)
        payload: dict[str, Any] = event_spec["payload"]
        candidate_id = str(payload["candidate_id"])
        if candidate_id in existing_candidate_ids:
            skipped.append(candidate_id)
            continue
        event = eventlog.append(
            event_spec["event_type"],
            actor=event_spec["actor"],
            payload=payload,
            thread=event_spec["thread"],
        )
        existing_candidate_ids.add(candidate_id)
        appended.append(
            AppendedProposal(
                candidate_id=candidate_id,
                seq=event.seq,
                hash=event.hash,
                session_id=str(event_spec["thread"]),
            )
        )

    return ProcedureMiningSummary(
        session_ids=tuple(sorted(traces)),
        mined_count=len(mined),
        appended_count=len(appended),
        skipped_duplicate_count=len(skipped),
        mined=tuple(mined),
        appended=tuple(appended),
        skipped_candidate_ids=tuple(skipped),
    )


def _normalize_step(event_type: str, payload: object) -> tuple[str | None, bool]:
    """Return the normalized step name and honest success flag for one event."""
    if not isinstance(payload, Mapping):
        return None, False
    if event_type == "tool.call.completed":
        tool_name = _collapsed_text(payload.get("tool_name"))
        status = _collapsed_text(payload.get("status"))
        successful = status is not None and status.casefold() == "succeeded"
        return (f"tool:{tool_name}" if tool_name else None), successful

    command = _collapsed_text(payload.get("command"))
    command_name = command.split()[0] if command else None
    exit_code = payload.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        successful = exit_code == 0
    else:
        outcome = _collapsed_text(payload.get("outcome"))
        successful = outcome is not None and outcome.casefold() == "passed"
    return (f"command:{command_name}" if command_name else None), successful


def _successful_runs(steps: Sequence[TraceStep]) -> list[list[TraceStep]]:
    """Split a session trace into maximal runs of consecutive successful steps."""
    runs: list[list[TraceStep]] = []
    current: list[TraceStep] = []
    for step in steps:
        if step.successful:
            current.append(step)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _drop_subsumed(mined: list[MinedProcedure]) -> list[MinedProcedure]:
    """Drop n-grams contained in a longer n-gram with an identical support set."""
    return [
        candidate
        for candidate in mined
        if not any(
            len(other.steps) > len(candidate.steps)
            and other.support_sessions == candidate.support_sessions
            and _contains_contiguous(other.steps, candidate.steps)
            for other in mined
        )
    ]


def _contains_contiguous(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
    span = len(shorter)
    return any(longer[start : start + span] == shorter for start in range(len(longer) - span + 1))


def _cited_occurrences(procedure: MinedProcedure, cap: int) -> list[ProcedureOccurrence]:
    """Select occurrences to cite: earliest per supporting session first."""
    first_by_session: dict[str, ProcedureOccurrence] = {}
    for occurrence in procedure.occurrences:
        first_by_session.setdefault(occurrence.session_id, occurrence)

    cited = sorted(
        first_by_session.values(),
        key=lambda occurrence: (occurrence.first_seq, occurrence.session_id),
    )[:cap]
    if len(cited) < cap:
        for occurrence in procedure.occurrences:
            if len(cited) >= cap:
                break
            if occurrence not in cited:
                cited.append(occurrence)
    return sorted(cited, key=lambda occurrence: (occurrence.first_seq, occurrence.session_id))


def _existing_candidate_ids(events: Sequence[Any]) -> set[str]:
    candidate_ids: set[str] = set()
    for event in events:
        if getattr(event, "type", None) != "consolidation.candidate.created":
            continue
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            continue
        candidate_id = payload.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            candidate_ids.add(candidate_id)
    return candidate_ids


def _event_session_id(event: Any) -> str:
    raw = getattr(event, "thread", "default")
    if not isinstance(raw, str) or not raw:
        return "default"
    return raw


def _procedure_title(procedure: MinedProcedure) -> str:
    return _clip(f"Procedure: {' -> '.join(procedure.steps)}", _TITLE_MAX_CHARS)


def _procedure_summary(procedure: MinedProcedure) -> str:
    return _clip(
        (
            f"Recurring successful {len(procedure.steps)}-step tool sequence mined from "
            f"{len(procedure.occurrences)} occurrences across {procedure.support} sessions: "
            f"{' -> '.join(procedure.steps)}"
        ),
        _SUMMARY_MAX_CHARS,
    )


def _collapsed_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    collapsed = " ".join(value.split())
    return collapsed or None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _validate_min_support(min_support: int) -> None:
    if (
        not isinstance(min_support, int)
        or isinstance(min_support, bool)
        or min_support < MINING_MIN_SUPPORT
    ):
        raise ValueError(
            f"min_support must be an integer >= {MINING_MIN_SUPPORT}; "
            "procedures must recur across distinct sessions"
        )


def _validate_lengths(*, min_length: int, max_length: int) -> None:
    for name, value in (("min_length", min_length), ("max_length", max_length)):
        if not isinstance(value, int) or isinstance(value, bool) or value < MIN_PROCEDURE_LENGTH:
            raise ValueError(f"{name} must be an integer >= {MIN_PROCEDURE_LENGTH}")
    if max_length < min_length:
        raise ValueError("max_length must be >= min_length")
