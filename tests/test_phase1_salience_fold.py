"""Phase 1 (C1): a checkout folds salience once, not twice.

``build_memory_checkout`` needs the same salience replay for both the cognitive
ranking blend and the salience diagnostics. It now folds once and threads the
result into both via ``precomputed_states``. These tests pin the mechanism: when
states are supplied the helpers must not replay the ledger again, and when they
are not supplied the helpers still fold themselves (so a standalone call stays
correct).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import zaxy.core.checkout_build as checkout_build

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_diagnostics_reuse_precomputed_states_without_refolding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(self: object, *args: object, **kwargs: object) -> dict:
        raise AssertionError("SalienceLedger.replay must not run when states are precomputed")

    monkeypatch.setattr(checkout_build.SalienceLedger, "replay", _forbidden)

    result = checkout_build._checkout_salience_diagnostics(
        replay_events=[object()],  # truthy: gets past the empty-log guard
        current_facts=[],
        evidence=[],
        now=_NOW,
        precomputed_states={},
    )

    # Empty precomputed states short-circuit to None without a second fold.
    assert result is None


def test_diagnostics_fold_themselves_when_not_precomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds: list[object] = []

    def _spy(self: object, events: object, **kwargs: object) -> dict:
        folds.append(events)
        return {}

    monkeypatch.setattr(checkout_build.SalienceLedger, "replay", _spy)

    result = checkout_build._checkout_salience_diagnostics(
        replay_events=[object()],
        current_facts=[],
        evidence=[],
        now=_NOW,
    )

    # Standalone call (no precomputed states) still folds exactly once.
    assert result is None
    assert len(folds) == 1
