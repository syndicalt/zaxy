"""Memory Evolution Policy: the governed-autonomy contract for Zaxy 3 (I4).

Every active-memory *evolution* — consolidate, update, forget, rule-generate,
promote — routes through one explicit, configurable autonomy policy. The policy
decides whether an evolution may auto-apply, must be held for review, and (for
auto-applied evolutions) the rollback window. Every gate decision is recorded as
a non-authoritative, replayable Eventloom event (``evolution.gate.evaluated``),
so the decision itself is auditable and reversible by replay.

Zaxy 3 defaults to ``auto_with_rollback``: an evolution whose confidence reaches
its op's threshold auto-applies but stays reversible for the rollback window,
and anything below threshold is held for review. The stricter tiers
(``propose_only``, ``require_review``) are available globally or per-op for
deployments wanting tighter guardrails. Either way Zaxy's load-bearing invariant
holds — nothing auto-promotes to *authority*; gate decisions are themselves
non-authoritative, replayable events. See ``ZAXY-3.md`` (initiative I4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_AUTHORITY_STATUS = "non_authoritative"

GATE_EVENT_TYPE = "evolution.gate.evaluated"

#: Autonomy tiers, conservative-first.
AUTONOMY_TIERS: tuple[str, ...] = ("propose_only", "auto_with_rollback", "require_review")
DEFAULT_AUTONOMY_TIER = "auto_with_rollback"

#: Governed evolution operations.
EVOLUTION_OPS: tuple[str, ...] = ("consolidate", "update", "forget", "rule_generate", "promote")

DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_ROLLBACK_WINDOW_SECONDS = 86_400

DECISION_AUTO_APPLY = "auto_apply"
DECISION_REQUIRES_REVIEW = "requires_review"


@dataclass(frozen=True)
class MemoryEvolutionPolicy:
    """Resolved autonomy policy for governed memory evolution.

    ``default_tier`` applies to any op without an explicit ``op_tiers`` override;
    ``default_threshold`` likewise for ``op_thresholds``. Thresholds only matter
    for the ``auto_with_rollback`` tier (the confidence at/above which an op may
    auto-apply).
    """

    default_tier: str = DEFAULT_AUTONOMY_TIER
    op_tiers: Mapping[str, str] = field(default_factory=dict)
    op_thresholds: Mapping[str, float] = field(default_factory=dict)
    default_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    rollback_window_seconds: int = DEFAULT_ROLLBACK_WINDOW_SECONDS

    def __post_init__(self) -> None:
        _validate_tier(self.default_tier, field_name="default_tier")
        for op, tier in self.op_tiers.items():
            _validate_op(op)
            _validate_tier(tier, field_name=f"op_tiers[{op}]")
        for op, threshold in self.op_thresholds.items():
            _validate_op(op)
            _validate_threshold(threshold, field_name=f"op_thresholds[{op}]")
        _validate_threshold(self.default_threshold, field_name="default_threshold")
        if (
            isinstance(self.rollback_window_seconds, bool)
            or not isinstance(self.rollback_window_seconds, int)
            or self.rollback_window_seconds < 0
        ):
            raise ValueError("rollback_window_seconds must be a non-negative integer")

    def tier_for(self, op: str) -> str:
        """Return the autonomy tier governing ``op``."""
        _validate_op(op)
        return self.op_tiers.get(op, self.default_tier)

    def threshold_for(self, op: str) -> float:
        """Return the auto-apply confidence threshold for ``op``."""
        _validate_op(op)
        return self.op_thresholds.get(op, self.default_threshold)


@dataclass(frozen=True)
class EvolutionGateDecision:
    """The outcome of evaluating the evolution policy for one op."""

    op: str
    tier: str
    confidence: float
    threshold: float
    decision: str
    auto_apply: bool
    requires_review: bool
    rollback_window_seconds: int
    reason: str

    def to_payload(self, *, candidate_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return the auditable, non-authoritative event payload for this decision."""
        payload: dict[str, Any] = {
            "op": self.op,
            "tier": self.tier,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "decision": self.decision,
            "auto_apply": self.auto_apply,
            "requires_review": self.requires_review,
            "rollback_window_seconds": self.rollback_window_seconds,
            "reason": self.reason,
            "authority_status": _AUTHORITY_STATUS,
        }
        if candidate_ref is not None:
            payload["candidate_ref"] = _snapshot_candidate_ref(candidate_ref)
        return payload


def resolve_evolution_policy(settings: Any) -> MemoryEvolutionPolicy:
    """Build a policy from Settings.

    Reads ``evolution_autonomy_default`` (default ``auto_with_rollback`` — auto-apply
    above threshold, reversible within the rollback window), ``evolution_rollback_window_seconds``,
    ``evolution_confidence_threshold``, and the optional per-op ``evolution_op_autonomy``
    / ``evolution_op_thresholds`` override strings defensively (missing attributes fall
    back to the defaults), so a minimal or mocked settings object still yields a valid
    policy. The stricter tiers (``propose_only``, ``require_review``) are available per-op
    or globally for deployments that want tighter guardrails.
    """
    default_tier = getattr(settings, "evolution_autonomy_default", None) or DEFAULT_AUTONOMY_TIER
    rollback = getattr(settings, "evolution_rollback_window_seconds", None)
    rollback_seconds = (
        int(rollback)
        if isinstance(rollback, int) and not isinstance(rollback, bool)
        else DEFAULT_ROLLBACK_WINDOW_SECONDS
    )
    threshold = getattr(settings, "evolution_confidence_threshold", None)
    default_threshold = (
        float(threshold)
        if isinstance(threshold, int | float) and not isinstance(threshold, bool)
        else DEFAULT_CONFIDENCE_THRESHOLD
    )
    op_tiers = parse_op_autonomy(getattr(settings, "evolution_op_autonomy", None))
    op_thresholds = parse_op_thresholds(getattr(settings, "evolution_op_thresholds", None))
    return MemoryEvolutionPolicy(
        default_tier=str(default_tier),
        op_tiers=op_tiers,
        op_thresholds=op_thresholds,
        default_threshold=default_threshold,
        rollback_window_seconds=rollback_seconds,
    )


def parse_op_autonomy(spec: object) -> dict[str, str]:
    """Parse a per-op autonomy override string into a validated tier map.

    Format: ``"op=tier,op=tier"`` (e.g. ``"update=propose_only,forget=require_review"``).
    Empty/None yields an empty map. Raises ``ValueError`` on an unknown op/tier or
    malformed entry, so misconfiguration fails fast at policy resolution.
    """
    if spec is None:
        return {}
    if not isinstance(spec, str):
        raise ValueError("evolution_op_autonomy must be a string")
    overrides: dict[str, str] = {}
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"evolution_op_autonomy entry must be 'op=tier': {entry!r}")
        op, _, tier = entry.partition("=")
        op = op.strip()
        tier = tier.strip()
        _validate_op(op)
        _validate_tier(tier, field_name=f"evolution_op_autonomy[{op}]")
        overrides[op] = tier
    return overrides


def parse_op_thresholds(spec: object) -> dict[str, float]:
    """Parse a per-op confidence-threshold override string into a validated map.

    Format: ``"op=0.9,op=0.7"`` (e.g. ``"forget=0.95,update=0.6"``). Empty/None
    yields an empty map. Raises ``ValueError`` on an unknown op, an unparseable or
    out-of-range threshold, or a malformed entry, so misconfiguration fails fast at
    policy resolution rather than silently pinning an op to the default.
    """
    if spec is None:
        return {}
    if not isinstance(spec, str):
        raise ValueError("evolution_op_thresholds must be a string")
    overrides: dict[str, float] = {}
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"evolution_op_thresholds entry must be 'op=threshold': {entry!r}")
        op, _, raw_threshold = entry.partition("=")
        op = op.strip()
        raw_threshold = raw_threshold.strip()
        _validate_op(op)
        try:
            threshold = float(raw_threshold)
        except ValueError:
            raise ValueError(
                f"evolution_op_thresholds[{op}] must be a number: {raw_threshold!r}"
            ) from None
        if not math.isfinite(threshold):
            raise ValueError(f"evolution_op_thresholds[{op}] must be a finite number")
        _validate_threshold(threshold, field_name=f"evolution_op_thresholds[{op}]")
        overrides[op] = threshold
    return overrides


def evaluate_evolution_gate(
    op: str,
    confidence: float,
    *,
    policy: MemoryEvolutionPolicy,
) -> EvolutionGateDecision:
    """Decide whether an evolution op may auto-apply under ``policy``.

    ``propose_only`` and ``require_review`` always hold for review regardless of
    confidence. ``auto_with_rollback`` auto-applies only when ``confidence`` is at
    or above the op's threshold; otherwise it holds for review.
    """
    _validate_op(op)
    _validate_confidence(confidence)
    tier = policy.tier_for(op)
    threshold = policy.threshold_for(op)
    confidence_value = float(confidence)

    if tier == "auto_with_rollback" and confidence_value >= threshold:
        return EvolutionGateDecision(
            op=op,
            tier=tier,
            confidence=confidence_value,
            threshold=threshold,
            decision=DECISION_AUTO_APPLY,
            auto_apply=True,
            requires_review=False,
            rollback_window_seconds=policy.rollback_window_seconds,
            reason=(
                f"{op}: auto-applied under auto_with_rollback "
                f"(confidence {confidence_value:.3f} >= threshold {threshold:.3f}); "
                f"reversible within {policy.rollback_window_seconds}s"
            ),
        )

    if tier == "auto_with_rollback":
        reason = (
            f"{op}: held for review under auto_with_rollback "
            f"(confidence {confidence_value:.3f} < threshold {threshold:.3f})"
        )
    elif tier == "require_review":
        reason = f"{op}: held for review (tier require_review)"
    else:
        reason = f"{op}: proposed for review (tier propose_only; no autonomous promotion)"

    return EvolutionGateDecision(
        op=op,
        tier=tier,
        confidence=confidence_value,
        threshold=threshold,
        decision=DECISION_REQUIRES_REVIEW,
        auto_apply=False,
        requires_review=True,
        rollback_window_seconds=policy.rollback_window_seconds,
        reason=reason,
    )


def build_evolution_gate_event(
    *,
    actor: str,
    session_id: str,
    decision: EvolutionGateDecision,
    candidate_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an auditable, non-authoritative evolution-gate event spec."""
    _validate_non_empty_string(actor, field_name="actor")
    _validate_non_empty_string(session_id, field_name="session_id")
    if not isinstance(decision, EvolutionGateDecision):
        raise ValueError("decision must be an EvolutionGateDecision")
    return {
        "event_type": GATE_EVENT_TYPE,
        "actor": actor,
        "payload": decision.to_payload(candidate_ref=candidate_ref),
        "thread": session_id,
    }


def _validate_tier(tier: object, *, field_name: str = "tier") -> None:
    if tier not in AUTONOMY_TIERS:
        valid = ", ".join(AUTONOMY_TIERS)
        raise ValueError(f"{field_name} must be one of: {valid}")


def _validate_op(op: object) -> None:
    if op not in EVOLUTION_OPS:
        valid = ", ".join(EVOLUTION_OPS)
        raise ValueError(f"op must be one of: {valid}")


def _validate_confidence(confidence: object) -> None:
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("confidence must be between 0.0 and 1.0")


def _validate_threshold(threshold: object, *, field_name: str = "threshold") -> None:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


def _validate_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _snapshot_candidate_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(ref, Mapping):
        raise ValueError("candidate_ref must be a mapping")
    snapshot: dict[str, Any] = {}
    for key in ("candidate_id", "seq", "hash", "name", "entity_type"):
        if key in ref and ref[key] is not None:
            snapshot[key] = ref[key]
    if not snapshot:
        raise ValueError(
            "candidate_ref must include at least one of candidate_id, seq, hash, name"
        )
    return snapshot
