"""Alpha.1 exact-scored causal and consolidation benchmark helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zaxy.causal import CAUSAL_RELATION_TYPES
from zaxy.consolidation import (
    CONSOLIDATION_CANDIDATE_TYPES,
    validate_consolidation_candidate_id,
)

CAUSAL_QUERY_TYPES = frozenset({"successor", "predecessor"})
NON_AUTHORITATIVE_STATUS = "non_authoritative"
_EVENTLOOM_CITATION_RE = re.compile(
    r"^eventloom://(?P<session>[^/\s]+)/events/(?P<seq>\d+)#(?P<hash>[a-f0-9]{6,})$"
)
_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_EVENT_REF_RE = re.compile(r"^(?P<seq>\d+):(?P<hash>[a-f0-9]{64})$")
_CONSOLIDATION_CANDIDATE_ID_RE = re.compile(
    r"^consolidation:(?P<candidate_type>[a-z]+):(?P<digest>[a-f0-9]{24})$"
)


@dataclass(frozen=True)
class CausalBenchmarkCase:
    """Gold labels for one alpha.1 causal edge retrieval case."""

    case_id: str
    query: str
    query_type: str
    source: Mapping[str, Any]
    target: Mapping[str, Any]
    relation_type: str
    citation: str

    def __post_init__(self) -> None:
        _require_text("case_id", self.case_id)
        _require_text("query", self.query)
        if self.query_type not in CAUSAL_QUERY_TYPES:
            raise ValueError("query_type must be 'successor' or 'predecessor'")
        _require_endpoint("source", self.source)
        _require_endpoint("target", self.target)
        _require_causal_relation_type(self.relation_type)
        _require_eventloom_citation("citation", self.citation)


@dataclass(frozen=True)
class ConsolidationBenchmarkCase:
    """Gold labels for one consolidation citation-fidelity case."""

    case_id: str
    candidate_id: str
    candidate_type: str
    source_events: Sequence[Mapping[str, Any]]
    citation: str
    authority_status: str = NON_AUTHORITATIVE_STATUS

    def __post_init__(self) -> None:
        _require_text("case_id", self.case_id)
        _require_consolidation_candidate_type(self.candidate_type)
        _require_consolidation_candidate_id(self.candidate_id, self.candidate_type)
        _require_source_events(self.source_events)
        _require_eventloom_citation("citation", self.citation)
        if self.authority_status != NON_AUTHORITATIVE_STATUS:
            raise ValueError(
                "authority_status must be non_authoritative for alpha.1 benchmark cases"
            )


def evaluate_causal_results(
    case: CausalBenchmarkCase,
    results: Sequence[Mapping[str, Any] | object],
) -> dict[str, Any]:
    """Score causal retrieval results against one gold case.

    The endpoint under evaluation follows query direction: successor queries
    match the gold target, while predecessor queries match the gold source.
    Among endpoint hits, the function reports the highest-quality match so
    stale, promoted, wrongly related, or uncited distractors do not hide a
    fully valid candidate later in the result set.
    """
    candidates = [_score_causal_result(case, result) for result in results]
    candidates.sort(key=lambda row: row["_ranking"], reverse=True)
    best = candidates[0] if candidates else _empty_causal_row(case)
    best.pop("_ranking", None)
    return best


def summarize_causal_benchmark(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summarize alpha.1 causal benchmark rows."""
    if not rows:
        return {
            "case_count": 0,
            "mean": 0.0,
            "hit_rate": 0.0,
            "citation_coverage": 0.0,
            "authority_boundary": 0.0,
        }
    return {
        "case_count": len(rows),
        "mean": _mean(float(row.get("score", 0.0)) for row in rows),
        "hit_rate": _mean(1.0 if row.get("hit") is True else 0.0 for row in rows),
        "citation_coverage": _mean(1.0 if row.get("citation") is True else 0.0 for row in rows),
        "authority_boundary": _mean(
            1.0 if row.get("authority_boundary") is True else 0.0 for row in rows
        ),
    }


def evaluate_consolidation_candidate(
    case: ConsolidationBenchmarkCase,
    candidate: Mapping[str, Any] | object,
) -> dict[str, Any]:
    """Score a consolidation projection candidate for citation fidelity."""
    candidate_match = (
        _text_value(candidate, "candidate_id") == case.candidate_id
        and _text_value(candidate, "candidate_type") == case.candidate_type
    )
    source_event_fidelity = _source_event_set(candidate) == _source_event_set(case)
    citation_coverage = _consolidation_citation_coverage(case, candidate)
    authority_boundary = (
        _text_value(candidate, "authority_status") == NON_AUTHORITATIVE_STATUS
        and case.authority_status == NON_AUTHORITATIVE_STATUS
    )
    score = _metric_score(
        candidate_match,
        source_event_fidelity,
        citation_coverage,
        authority_boundary,
    )
    return {
        "case_id": case.case_id,
        "candidate_match": candidate_match,
        "source_event_fidelity": source_event_fidelity,
        "citation_coverage": citation_coverage,
        "authority_boundary": authority_boundary,
        "score": score,
    }


def _score_causal_result(
    case: CausalBenchmarkCase, result: Mapping[str, Any] | object
) -> dict[str, Any]:
    hit = _endpoint_name(result, case.query_type) == _expected_endpoint_name(case)
    relation_match = hit and _text_value(result, "relation_type") == case.relation_type
    citation = hit and _result_citation_matches(case, result)
    authority_boundary = (
        hit
        and _text_value(result, "authority_status") == NON_AUTHORITATIVE_STATUS
        and not _is_stale(result)
    )
    score = _metric_score(hit, relation_match, citation, authority_boundary)
    return {
        "case_id": case.case_id,
        "query_type": case.query_type,
        "hit": hit,
        "relation_match": relation_match,
        "citation": citation,
        "authority_boundary": authority_boundary,
        "score": score,
        "matched_result": result if hit else None,
        "_ranking": (score, 1 if hit else 0),
    }


def _empty_causal_row(case: CausalBenchmarkCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "query_type": case.query_type,
        "hit": False,
        "relation_match": False,
        "citation": False,
        "authority_boundary": False,
        "score": 0.0,
        "matched_result": None,
        "_ranking": (0.0, 0),
    }


def _expected_endpoint_name(case: CausalBenchmarkCase) -> str:
    endpoint = case.target if case.query_type == "successor" else case.source
    return str(endpoint["name"])


def _endpoint_name(result: Mapping[str, Any] | object, query_type: str) -> str | None:
    endpoint_key = "target" if query_type == "successor" else "source"
    endpoint = _value(result, endpoint_key)
    if isinstance(endpoint, Mapping):
        name = endpoint.get("name")
        return name if isinstance(name, str) else None
    field_name = "target_name" if query_type == "successor" else "source_name"
    direct = _text_value(result, field_name)
    if direct is not None:
        return direct
    causal_field_name = f"causal_{field_name}"
    return _text_value(result, causal_field_name)


def _result_citation_matches(case: CausalBenchmarkCase, result: Mapping[str, Any] | object) -> bool:
    citations = _candidate_citations(result)
    return case.citation in citations and _is_eventloom_citation(case.citation)


def _consolidation_citation_coverage(
    case: ConsolidationBenchmarkCase,
    candidate: Mapping[str, Any] | object,
) -> bool:
    expected_source_events = _source_event_set(case)
    if not expected_source_events:
        return False

    expected_eventloom_refs = _eventloom_refs_from_source_events(
        session=_session_from_eventloom_citation(case.citation),
        source_events=expected_source_events,
    )

    # Production candidate payloads cite Eventloom events as {"seq", "hash"}
    # source_events. Graph-projected candidates may additionally expose
    # source_event_refs as "seq:hash". Citation coverage accepts either
    # production field without requiring benchmark-only source_events[].ref.
    source_event_refs = _candidate_source_event_refs(candidate)
    if source_event_refs:
        return expected_source_events.issubset(source_event_refs)

    citations = _candidate_citations(candidate)
    if citations:
        return expected_eventloom_refs.issubset(citations)

    candidate_source_events = _source_event_set(candidate)
    return expected_source_events == candidate_source_events


def _candidate_citations(candidate: Mapping[str, Any] | object) -> set[str]:
    citations: set[str] = set()
    citation = _text_value(candidate, "citation")
    if citation is not None and _is_eventloom_citation(citation):
        citations.add(citation)
    raw_citations = _value(candidate, "citations")
    if isinstance(raw_citations, Sequence) and not isinstance(raw_citations, str):
        for item in raw_citations:
            if isinstance(item, str) and _is_eventloom_citation(item):
                citations.add(item)
    return citations


def _source_event_set(
    value: ConsolidationBenchmarkCase | Mapping[str, Any] | object,
) -> set[tuple[int, str]]:
    raw_events = (
        value.source_events
        if isinstance(value, ConsolidationBenchmarkCase)
        else _value(value, "source_events")
    )
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, str):
        return set()
    events: set[tuple[int, str]] = set()
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            continue
        seq = raw_event.get("seq")
        event_hash = raw_event.get("hash")
        if (
            isinstance(seq, int)
            and not isinstance(seq, bool)
            and seq > 0
            and isinstance(event_hash, str)
            and _EVENT_HASH_RE.fullmatch(event_hash)
        ):
            events.add((seq, event_hash))
    return events


def _candidate_source_event_refs(candidate: Mapping[str, Any] | object) -> set[tuple[int, str]]:
    raw_refs = _value(candidate, "source_event_refs")
    if not isinstance(raw_refs, Sequence) or isinstance(raw_refs, str):
        return set()
    refs: set[tuple[int, str]] = set()
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, str):
            continue
        match = _SOURCE_EVENT_REF_RE.fullmatch(raw_ref)
        if match is not None:
            refs.add((int(match.group("seq")), match.group("hash")))
    return refs


def _eventloom_refs_from_source_events(
    *, session: str, source_events: set[tuple[int, str]]
) -> set[str]:
    return {
        f"eventloom://{session}/events/{seq}#{event_hash}"
        for seq, event_hash in source_events
    }


def _session_from_eventloom_citation(citation: str) -> str:
    match = _EVENTLOOM_CITATION_RE.match(citation)
    if match is None:
        raise ValueError("citation must be an Eventloom citation")
    return match.group("session")


def _is_stale(result: Mapping[str, Any] | object) -> bool:
    stale = _value(result, "stale")
    if stale is True:
        return True
    valid_to = _value(result, "valid_to")
    if isinstance(valid_to, str) and valid_to.strip():
        return True
    superseded_by = _value(result, "superseded_by")
    return isinstance(superseded_by, str) and bool(superseded_by.strip())


def _metric_score(*values: bool) -> float:
    return round(sum(1 for value in values if value) / len(values), 4)


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def _require_endpoint(field: str, value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    _require_text(f"{field}.name", value.get("name"))
    _require_text(f"{field}.entity_type", value.get("entity_type"))


def _require_source_events(value: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise ValueError("source_events must be a non-empty sequence")
    for index, event in enumerate(value):
        if not isinstance(event, Mapping):
            raise ValueError(f"source_events[{index}] must be a mapping")
        seq = event.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise ValueError(f"source_events[{index}].seq must be an integer")
        if seq <= 0:
            raise ValueError(f"source_events[{index}].seq must be a positive integer")
        event_hash = event.get("hash")
        if not isinstance(event_hash, str) or _EVENT_HASH_RE.fullmatch(event_hash) is None:
            raise ValueError(
                f"source_events[{index}].hash must be exactly 64 lowercase hex characters"
            )


def _require_causal_relation_type(value: Any) -> None:
    _require_text("relation_type", value)
    if value not in CAUSAL_RELATION_TYPES:
        valid = ", ".join(sorted(CAUSAL_RELATION_TYPES))
        raise ValueError(f"relation_type must be one of: {valid}")


def _require_consolidation_candidate_type(value: Any) -> None:
    _require_text("candidate_type", value)
    if value not in CONSOLIDATION_CANDIDATE_TYPES:
        valid = ", ".join(sorted(CONSOLIDATION_CANDIDATE_TYPES))
        raise ValueError(f"candidate_type must be one of: {valid}")


def _require_consolidation_candidate_id(candidate_id: Any, candidate_type: str) -> None:
    validated_candidate_id = validate_consolidation_candidate_id(candidate_id)
    match = _CONSOLIDATION_CANDIDATE_ID_RE.fullmatch(validated_candidate_id)
    if match is None:
        raise ValueError(
            "candidate_id must match consolidation:{candidate_type}:{24 lowercase hex characters}"
        )
    id_candidate_type = match.group("candidate_type")
    if id_candidate_type != candidate_type:
        raise ValueError("candidate_id candidate_type must match candidate_type")


def _require_text(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_eventloom_citation(field: str, value: Any) -> None:
    _require_text(field, value)
    if not _is_eventloom_citation(value):
        raise ValueError(f"{field} must be an Eventloom citation")


def _is_eventloom_citation(value: str) -> bool:
    return bool(_EVENTLOOM_CITATION_RE.match(value))


def _text_value(value: Mapping[str, Any] | object, key: str) -> str | None:
    raw = _value(value, key)
    return raw if isinstance(raw, str) else None


def _value(value: Mapping[str, Any] | object, key: str) -> Any:
    if isinstance(value, Mapping):
        raw = value.get(key)
        if raw is not None:
            return raw
        properties = value.get("properties")
        if isinstance(properties, Mapping):
            return properties.get(key)
        return None
    raw = getattr(value, key, None)
    if raw is not None:
        return raw
    properties = getattr(value, "properties", None)
    if isinstance(properties, Mapping):
        return properties.get(key)
    return None
