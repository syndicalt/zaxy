"""Causal memory payload contracts.

Causal edges are proposed, non-authoritative graph evidence. They must cite the
Eventloom event that justifies the edge so downstream retrieval can expose the
authority boundary instead of presenting inferred causality as fact.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

CAUSAL_RELATION_TYPES = frozenset(
    {
        "caused",
        "enabled",
        "blocked",
        "prevented",
        "regressed",
        "fixed",
        "explained",
    }
)

_DEFAULT_REVIEW_STATUS = "proposed"
_DEFAULT_AUTHORITY_STATUS = "non_authoritative"


@dataclass(frozen=True)
class CausalEdge:
    """A proposed causal edge with cited Eventloom evidence."""

    source: Mapping[str, Any]
    target: Mapping[str, Any]
    relation_type: str
    confidence: float
    method: str
    evidence: Mapping[str, Any]
    review_status: str = _DEFAULT_REVIEW_STATUS
    authority_status: str = _DEFAULT_AUTHORITY_STATUS
    graph_relation_type: str | None = None

    def __post_init__(self) -> None:
        _validate_entity_ref(self.source, field_name="source")
        _validate_entity_ref(self.target, field_name="target")
        _validate_causal_relation_type(self.relation_type)
        _validate_confidence(self.confidence)
        _validate_method(self.method)
        _validate_evidence(self.evidence)
        if self.graph_relation_type is not None and self.graph_relation_type != causal_relation_to_graph_relation(
            self.relation_type
        ):
            raise ValueError("graph_relation_type must match causal relation_type")
        _validate_status(self.review_status, field_name="review_status")
        _validate_status(self.authority_status, field_name="authority_status")

    def to_payload(self) -> dict[str, Any]:
        """Return the Eventloom payload representation for this causal edge."""
        return {
            "source": _entity_ref_to_dict(self.source),
            "target": _entity_ref_to_dict(self.target),
            "relation_type": self.relation_type,
            "graph_relation_type": self.graph_relation_type
            or causal_relation_to_graph_relation(self.relation_type),
            "confidence": self.confidence,
            "causal_method": self.method,
            "review_status": self.review_status,
            "authority_status": self.authority_status,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CausalQueryResult:
    """A causal graph query result that preserves review and authority metadata."""

    source: Mapping[str, Any]
    target: Mapping[str, Any]
    relation_type: str
    graph_relation_type: str
    confidence: float
    method: str
    citation: str
    review_status: str
    authority_status: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    path_length: int | None = None

    def __post_init__(self) -> None:
        _validate_entity_ref(self.source, field_name="source")
        _validate_entity_ref(self.target, field_name="target")
        _validate_causal_relation_type(self.relation_type)
        expected_graph_relation = causal_relation_to_graph_relation(self.relation_type)
        if self.graph_relation_type != expected_graph_relation:
            raise ValueError("graph_relation_type must match causal relation_type")
        _validate_confidence(self.confidence)
        _validate_method(self.method)
        _validate_status(self.review_status, field_name="review_status")
        _validate_status(self.authority_status, field_name="authority_status")
        if not isinstance(self.citation, str) or not self.citation:
            raise ValueError("citation must be a non-empty string")
        if self.path_length is not None and self.path_length < 1:
            raise ValueError("path_length must be at least 1 when set")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable causal retrieval result."""
        result: dict[str, Any] = {
            "source": _entity_ref_to_dict(self.source),
            "target": _entity_ref_to_dict(self.target),
            "relation_type": self.relation_type,
            "graph_relation_type": self.graph_relation_type,
            "confidence": self.confidence,
            "method": self.method,
            "citation": self.citation,
            "review_status": self.review_status,
            "authority_status": self.authority_status,
            "evidence": dict(self.evidence),
        }
        if self.path_length is not None:
            result["path_length"] = self.path_length
        return result


def causal_relation_to_graph_relation(relation_type: str) -> str:
    """Return the graph relation label used for a causal taxonomy relation."""
    _validate_causal_relation_type(relation_type)
    return f"causal_{relation_type}"


def build_causal_edge_event(
    *,
    actor: str,
    session_id: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    relation_type: str,
    confidence: float,
    method: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a causal.edge.generated Eventloom append payload.

    The returned shape is accepted by MemoryFabric/MCP append helpers. The event
    remains non-authoritative until a review workflow promotes or rejects it.
    """
    if not isinstance(actor, str) or not actor:
        raise ValueError("actor must be a non-empty string")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    edge = CausalEdge(
        source=source,
        target=target,
        relation_type=relation_type,
        confidence=confidence,
        method=method,
        evidence=evidence,
    )
    return {
        "event_type": "causal.edge.generated",
        "actor": actor,
        "payload": edge.to_payload(),
        "thread": session_id,
    }


def _validate_entity_ref(entity: Mapping[str, Any], *, field_name: str) -> None:
    if not isinstance(entity, Mapping):
        raise ValueError(f"{field_name} must be an entity reference mapping")
    for key in ("name", "entity_type"):
        value = entity.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}.{key} must be a non-empty string")


def _entity_ref_to_dict(entity: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": entity["name"], "entity_type": entity["entity_type"]}


def _validate_causal_relation_type(relation_type: str) -> None:
    if relation_type not in CAUSAL_RELATION_TYPES:
        valid = ", ".join(sorted(CAUSAL_RELATION_TYPES))
        raise ValueError(f"causal relation_type must be one of: {valid}")


def _validate_confidence(confidence: float) -> None:
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise ValueError("confidence must be a number between 0.0 and 1.0")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")


def _validate_method(method: str) -> None:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("causal method must be a non-empty string")


def _validate_evidence(evidence: Mapping[str, Any]) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be a mapping")
    source_event_seq = evidence.get("source_event_seq")
    if not isinstance(source_event_seq, int) or isinstance(source_event_seq, bool):
        raise ValueError("evidence.source_event_seq must be an integer")
    if source_event_seq <= 0:
        raise ValueError("evidence.source_event_seq must be a positive integer")
    source_event_hash = evidence.get("source_event_hash")
    if not isinstance(source_event_hash, str) or not _EVENT_HASH_RE.fullmatch(source_event_hash):
        raise ValueError("evidence.source_event_hash must be exactly 64 lowercase hex characters")


def _validate_status(status: str, *, field_name: str) -> None:
    if not isinstance(status, str) or not status:
        raise ValueError(f"{field_name} must be a non-empty string")
