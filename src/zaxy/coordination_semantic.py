"""Semantic conflict adapter factory for Zaxy Coordinate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from zaxy.coordination import (
    ConflictState,
    FindingState,
    LocalSemanticConflictDetector,
    SemanticConflictDetector,
)

_ADAPTER_CONTRACT = "zaxy-coordination-semantic-v1"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MIN_CONFIDENCE = 0.7
_SAFE_EVIDENCE_KEYS = {
    "source",
    "path",
    "file",
    "line",
    "line_start",
    "line_end",
    "url",
    "source_sha256",
    "source_event_seq",
    "source_event_hash",
}


class _HostedSemanticConflictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_ids: list[str] = Field(min_length=2)
    claim_key: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = None
    source_reference: str | None = None


class _HostedSemanticConflictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflicts: list[_HostedSemanticConflictItem] = Field(default_factory=list)


@dataclass(frozen=True)
class HTTPSemanticConflictDetector:
    """Hosted semantic conflict detector with a narrow, auditable schema."""

    endpoint: str
    api_key: str | None = None
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    client: httpx.Client | None = None

    def __post_init__(self) -> None:
        endpoint = self.endpoint.strip()
        if not endpoint:
            raise ValueError("endpoint is required")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "endpoint", endpoint)

    def __call__(self, findings: list[FindingState]) -> list[ConflictState]:
        if len(findings) < 2:
            return []
        finding_by_id = {finding.finding_id: finding for finding in findings}
        response = self._post_json(
            {
                "adapter_contract": _ADAPTER_CONTRACT,
                "findings": [_finding_payload(finding) for finding in findings],
            }
        )
        response.raise_for_status()
        try:
            response_payload = response.json()
            hosted_response = _HostedSemanticConflictResponse.model_validate(response_payload)
        except (ValueError, ValidationError) as exc:
            raise ValueError("hosted semantic conflict response is invalid") from exc

        conflicts: list[ConflictState] = []
        for raw_conflict in hosted_response.conflicts:
            if raw_conflict.confidence < self.min_confidence:
                continue
            conflict_findings = []
            seen_ids: set[str] = set()
            for finding_id in raw_conflict.finding_ids:
                if finding_id in seen_ids:
                    raise ValueError(f"duplicate hosted semantic conflict finding_id: {finding_id}")
                seen_ids.add(finding_id)
                finding = finding_by_id.get(finding_id)
                if finding is None:
                    raise ValueError(f"unknown hosted semantic conflict finding_id: {finding_id}")
                conflict_findings.append(finding)
            claim_key = raw_conflict.claim_key.strip()
            if not claim_key:
                raise ValueError("hosted semantic conflict claim_key is required")
            conflicts.append(
                ConflictState(
                    claim_key=claim_key,
                    findings=conflict_findings,
                    conflict_type="semantic",
                    reason=_optional_string(raw_conflict.reason) or "hosted_semantic_adapter",
                    source_reference=_optional_string(raw_conflict.source_reference) or "adapter:http",
                )
            )
        return conflicts

    def _post_json(self, payload: dict[str, Any]) -> httpx.Response:
        if self.client is not None:
            return self.client.post(
                self.endpoint,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(
                self.endpoint,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers


def build_semantic_conflict_detector(settings: Any) -> SemanticConflictDetector | None:
    """Build the configured coordination semantic conflict detector."""
    provider = str(getattr(settings, "coordination_semantic_conflict_provider", "none")).casefold().strip()
    if provider in {"", "none", "off", "disabled"}:
        return None
    if provider == "lexical":
        return LocalSemanticConflictDetector(
            min_shared_subject_tokens=int(
                getattr(settings, "coordination_semantic_min_shared_subject_tokens", 2)
            )
        )
    if provider in {"http", "https", "hosted"}:
        endpoint = str(getattr(settings, "coordination_semantic_conflict_url", "") or "").strip()
        if not endpoint:
            raise ValueError("coordination_semantic_conflict_url is required for http semantic conflicts")
        return HTTPSemanticConflictDetector(
            endpoint,
            api_key=getattr(settings, "coordination_semantic_conflict_api_key", None),
            min_confidence=float(
                getattr(
                    settings,
                    "coordination_semantic_conflict_min_confidence",
                    _DEFAULT_MIN_CONFIDENCE,
                )
            ),
            timeout_seconds=float(
                getattr(
                    settings,
                    "coordination_semantic_conflict_timeout_seconds",
                    _DEFAULT_TIMEOUT_SECONDS,
                )
            ),
        )
    raise ValueError(f"Unsupported coordination semantic conflict provider: {provider}")


def _finding_payload(finding: FindingState) -> dict[str, Any]:
    return {
        "finding_id": finding.finding_id,
        "mission_id": finding.mission_id,
        "worker_id": finding.worker_id,
        "summary": finding.summary,
        "claim_key": finding.claim_key,
        "claim_value": finding.claim_value,
        "confidence": finding.confidence,
        "source_event_seq": finding.source_event_seq,
        "source_event_hash": finding.source_event_hash,
        "evidence": [_safe_evidence_payload(item) for item in finding.evidence[:8]],
    }


def _safe_evidence_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in _SAFE_EVIDENCE_KEYS:
        if key not in evidence:
            continue
        value = evidence.get(key)
        if isinstance(value, str):
            safe[key] = value[:500]
        elif isinstance(value, int | float | bool) or value is None:
            safe[key] = value
    return safe


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("hosted semantic conflict optional string fields must be strings")
    value = value.strip()
    return value or None
