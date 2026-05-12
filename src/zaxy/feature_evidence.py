"""Evidence-based feature hardening and pruning helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureEvidence:
    """Observed product evidence for one feature surface."""

    name: str
    category: str
    usage_count: int
    enabled_by_default: bool
    retained_context_uses: int = 0
    operator_confirmations: int = 0

    @property
    def evidence_score(self) -> int:
        """Return a conservative evidence score from observed usage signals."""
        default_bonus = 3 if self.enabled_by_default else 0
        return (
            self.usage_count
            + self.retained_context_uses * 2
            + self.operator_confirmations * 3
            + default_bonus
        )


def feature_pruning_candidates(
    features: list[FeatureEvidence],
    *,
    minimum_usage: int = 3,
) -> list[FeatureEvidence]:
    """Return optional feature surfaces that lack enough observed use."""
    candidates = [
        feature
        for feature in features
        if not feature.enabled_by_default and feature.evidence_score < minimum_usage
    ]
    return sorted(candidates, key=lambda feature: (feature.evidence_score, feature.name))


def render_feature_evidence_report(
    features: list[FeatureEvidence],
    *,
    minimum_usage: int = 3,
) -> str:
    """Render a short product hardening report for roadmap reviews."""
    lines = ["Feature evidence report:"]
    for feature in sorted(features, key=lambda item: item.name):
        state = (
            "review"
            if not feature.enabled_by_default and feature.evidence_score < minimum_usage
            else "keep"
        )
        lines.append(
            f"- {feature.name}: {state} "
            f"(category={feature.category}, evidence_score={feature.evidence_score})"
        )
    return "\n".join(lines)
