"""Tests for local coordination semantic conflict adapters."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from zaxy.config import Settings
from zaxy.coordination import FindingState, LocalSemanticConflictDetector
from zaxy.coordination_semantic import (
    HTTPSemanticConflictDetector,
    build_semantic_conflict_detector,
)


def test_semantic_conflict_helpers_are_public_exports() -> None:
    import zaxy

    assert zaxy.HTTPSemanticConflictDetector is HTTPSemanticConflictDetector
    assert zaxy.LocalSemanticConflictDetector is LocalSemanticConflictDetector
    assert zaxy.build_semantic_conflict_detector is build_semantic_conflict_detector


def test_build_semantic_conflict_detector_uses_configured_provider() -> None:
    assert build_semantic_conflict_detector(Settings(_env_file=None)) is None

    detector = build_semantic_conflict_detector(
        Settings(_env_file=None, coordination_semantic_conflict_provider="lexical")
    )

    assert isinstance(detector, LocalSemanticConflictDetector)


def test_build_semantic_conflict_detector_builds_http_provider() -> None:
    detector = build_semantic_conflict_detector(
        Settings(
            _env_file=None,
            coordination_semantic_conflict_provider="http",
            coordination_semantic_conflict_url="https://semantic.example/conflicts",
        )
    )

    assert isinstance(detector, HTTPSemanticConflictDetector)


def test_http_semantic_conflict_api_key_can_load_from_secret_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_file = tmp_path / "semantic-key"
    key_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("COORDINATION_SEMANTIC_CONFLICT_API_KEY_FILE", str(key_file))

    settings = Settings(
        _env_file=None,
        coordination_semantic_conflict_provider="http",
        coordination_semantic_conflict_url="https://semantic.example/conflicts",
    )
    detector = build_semantic_conflict_detector(settings)

    assert isinstance(detector, HTTPSemanticConflictDetector)
    assert detector.api_key == "file-secret"


def test_build_semantic_conflict_detector_rejects_http_provider_without_url() -> None:
    with pytest.raises(ValueError, match="coordination_semantic_conflict_url is required"):
        build_semantic_conflict_detector(
            Settings(_env_file=None, coordination_semantic_conflict_provider="http")
        )


def test_build_semantic_conflict_detector_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported coordination semantic conflict provider"):
        build_semantic_conflict_detector(
            Settings(_env_file=None, coordination_semantic_conflict_provider="oracle")
        )


def test_lexical_semantic_detector_ignores_same_polarity() -> None:
    detector = LocalSemanticConflictDetector()
    findings = [
        FindingState(
            finding_id="left",
            mission_id="auth-main",
            worker_id="auth-api",
            summary="JWKS cache refresh is enabled for auth tokens.",
            evidence=[],
        ),
        FindingState(
            finding_id="right",
            mission_id="auth-main",
            worker_id="auth-ui",
            summary="JWKS cache refresh is enabled in browser auth.",
            evidence=[],
        ),
    ]

    assert detector(findings) == []


def test_http_semantic_detector_posts_bounded_payload_and_converts_conflicts() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "conflicts": [
                    {
                        "finding_ids": ["left", "right"],
                        "claim_key": "semantic:jwks-cache-refresh",
                        "reason": "hosted_model_contradiction",
                        "confidence": 0.91,
                        "source_reference": "adapter:hosted:v1",
                    }
                ]
            },
        )

    detector = HTTPSemanticConflictDetector(
        "https://semantic.example/conflicts",
        api_key="secret-token",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    findings = [
        FindingState(
            finding_id="left",
            mission_id="auth-main",
            worker_id="auth-api",
            summary="JWKS cache refresh is enabled for auth tokens.",
            evidence=[
                {
                    "source": "src/auth.py",
                    "line": 12,
                    "large_payload": "x" * 1024,
                }
            ],
            claim_key="jwks.cache",
            claim_value="enabled",
            confidence=0.8,
            source_event_seq=3,
            source_event_hash="a" * 64,
        ),
        FindingState(
            finding_id="right",
            mission_id="auth-main",
            worker_id="auth-ui",
            summary="JWKS cache refresh is disabled in browser auth.",
            evidence=[],
            claim_key="jwks.cache",
            claim_value="disabled",
            confidence=0.7,
            source_event_seq=4,
            source_event_hash="b" * 64,
        ),
    ]

    conflicts = detector(findings)

    assert len(conflicts) == 1
    assert conflicts[0].claim_key == "semantic:jwks-cache-refresh"
    assert conflicts[0].conflict_type == "semantic"
    assert conflicts[0].reason == "hosted_model_contradiction"
    assert conflicts[0].source_reference == "adapter:hosted:v1"
    assert [finding.finding_id for finding in conflicts[0].findings] == ["left", "right"]
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer secret-token"
    payload = httpx.Response(200, content=requests[0].content).json()
    assert payload == {
        "findings": [
            {
                "finding_id": "left",
                "mission_id": "auth-main",
                "worker_id": "auth-api",
                "summary": "JWKS cache refresh is enabled for auth tokens.",
                "claim_key": "jwks.cache",
                "claim_value": "enabled",
                "confidence": 0.8,
                "source_event_seq": 3,
                "source_event_hash": "a" * 64,
                "evidence": [{"source": "src/auth.py", "line": 12}],
            },
            {
                "finding_id": "right",
                "mission_id": "auth-main",
                "worker_id": "auth-ui",
                "summary": "JWKS cache refresh is disabled in browser auth.",
                "claim_key": "jwks.cache",
                "claim_value": "disabled",
                "confidence": 0.7,
                "source_event_seq": 4,
                "source_event_hash": "b" * 64,
                "evidence": [],
            },
        ],
        "adapter_contract": "zaxy-coordination-semantic-v1",
    }


def test_http_semantic_detector_filters_low_confidence_conflicts() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "conflicts": [
                    {
                        "finding_ids": ["left", "right"],
                        "claim_key": "semantic:jwks",
                        "confidence": 0.69,
                    }
                ]
            },
        )

    detector = HTTPSemanticConflictDetector(
        "https://semantic.example/conflicts",
        min_confidence=0.7,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert detector(_two_findings()) == []


def test_http_semantic_detector_rejects_unknown_finding_ids() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "conflicts": [
                    {
                        "finding_ids": ["left", "missing"],
                        "claim_key": "semantic:jwks",
                        "confidence": 0.9,
                    }
                ]
            },
        )

    detector = HTTPSemanticConflictDetector(
        "https://semantic.example/conflicts",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ValueError, match="unknown hosted semantic conflict finding_id: missing"):
        detector(_two_findings())


def test_http_semantic_detector_rejects_duplicate_finding_ids() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "conflicts": [
                    {
                        "finding_ids": ["left", "left"],
                        "claim_key": "semantic:jwks",
                        "confidence": 0.9,
                    }
                ]
            },
        )

    detector = HTTPSemanticConflictDetector(
        "https://semantic.example/conflicts",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ValueError, match="duplicate hosted semantic conflict finding_id: left"):
        detector(_two_findings())


def test_http_semantic_detector_rejects_malformed_response_extra_fields() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "conflicts": [
                    {
                        "finding_ids": ["left", "right"],
                        "claim_key": "semantic:jwks",
                        "confidence": 0.9,
                        "unreviewed_payload": "must not pass through",
                    }
                ]
            },
        )

    detector = HTTPSemanticConflictDetector(
        "https://semantic.example/conflicts",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ValueError, match="hosted semantic conflict response is invalid"):
        detector(_two_findings())


def test_semantic_conflict_provider_defaults_to_off() -> None:
    settings = Settings(_env_file=None)

    assert settings.coordination_semantic_conflict_provider == "none"
    assert build_semantic_conflict_detector(settings) is None


def _two_findings() -> list[FindingState]:
    return [
        FindingState(
            finding_id="left",
            mission_id="auth-main",
            worker_id="auth-api",
            summary="JWKS cache refresh is enabled for auth tokens.",
            evidence=[],
        ),
        FindingState(
            finding_id="right",
            mission_id="auth-main",
            worker_id="auth-ui",
            summary="JWKS cache refresh is disabled in browser auth.",
            evidence=[],
        ),
    ]
