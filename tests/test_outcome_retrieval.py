"""End-to-end: a failed outcome demotes a memory in retrieval (3 / I1.3).

Closes the outcome-driven learning loop with the real retrieval path under the
default ``cognitive`` profile: recording a *failure* with a high reported prior
(a big prediction error) scales the invalidated reinforcement so strongly that
the recalled memory drops below the attenuation floor and is demoted out of the
ranked checkout packet, while the control memory keeps ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.config import get_settings
from zaxy.core import MemoryFabric
from zaxy.outcome_learning import OUTCOME_EVENT_TYPE

SID = "agent-1"
# A names every query term; B names only two -> A out-ranks B by overlap alone.
QUERY = "migration retry lock timeout legacy"
A_DECISION = "Retry the migration with the legacy lock timeout disabled."
B_DECISION = "Document the migration retry runbook for the on-call rotation."


def _wired_fabric(eventloom_dir: Path) -> MemoryFabric:
    """Real Eventloom + verbatim retrieval lane, mocked graph projection lane."""
    with patch("zaxy.core.fabric.build_projection_store") as mock_store:
        mock_store.return_value = AsyncMock()
        fabric = MemoryFabric(eventloom_path=str(eventloom_dir), tracer_disabled=True)
    fabric.query_router = MagicMock(query=AsyncMock(return_value=[]))
    fabric._connected = True
    return fabric


async def _seed_pair(fabric: MemoryFabric) -> tuple[Any, Any]:
    a = await fabric.append(
        "decision.made", actor="dev", payload={"decision": A_DECISION}, session_id=SID
    )
    b = await fabric.append(
        "decision.made", actor="dev", payload={"decision": B_DECISION}, session_id=SID
    )
    return a, b


def _fact_seq(fact: dict[str, Any]) -> int | None:
    citation = fact.get("citation")
    if not isinstance(citation, str) or "/events/" not in citation:
        return None
    try:
        return int(citation.split("/events/")[1].split("#")[0])
    except (IndexError, ValueError):
        return None


async def test_failed_outcome_demotes_memory_in_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the default cognitive retrieval profile (salience ranking on) so the
    # outcome-conditioned demotion runs through the real ranking path; the
    # autouse settings fixture clears the cache again on teardown.
    monkeypatch.setenv("RETRIEVAL_PROFILE", "cognitive")
    get_settings.cache_clear()

    # Control: with no recorded outcome the higher-overlap memory A ranks first.
    control = _wired_fabric(tmp_path / "control")
    assert control.retrieval_profile.salience_ranking
    a_ctrl, b_ctrl = await _seed_pair(control)
    control_checkout = await control.checkout_memory(
        QUERY, session_id=SID, limit=5, record_reinforcement=False
    )
    control_seqs = [
        seq for fact in control_checkout.current_facts if (seq := _fact_seq(fact)) is not None
    ]
    assert a_ctrl.seq in control_seqs and b_ctrl.seq in control_seqs
    assert control_seqs.index(a_ctrl.seq) < control_seqs.index(b_ctrl.seq)

    # Treatment: record a FAILURE with a high prior (big surprise) against A.
    treat = _wired_fabric(tmp_path / "treat")
    a, b = await _seed_pair(treat)
    result = await treat.record_outcome(
        outcome="failure",
        summary="the recalled fix did not hold",
        target_seq=a.seq,
        target_hash=a.hash,
        prior=0.9,
        session_id=SID,
    )
    assert result["reinforced"] == "invalidated"

    # The surprise is auditable on the cited outcome event.
    events = treat.session_manager.get(SID).eventlog.read_all()
    outcome_event = next(e for e in events if e.type == OUTCOME_EVENT_TYPE)
    assert outcome_event.payload["prior"] == 0.9
    assert outcome_event.payload["prediction_error"] == 0.9

    checkout = await treat.checkout_memory(QUERY, session_id=SID, limit=5)
    fact_seqs = [seq for fact in checkout.current_facts if (seq := _fact_seq(fact)) is not None]

    # B still surfaces and now out-ranks A. The high prediction error drove A's
    # salience below the attenuation floor (a demotion the default invalidated
    # multiplier 0.2 -- which sits above the floor -- would not achieve), so A is
    # excluded from the ranked packet and recorded with a lower salience score.
    assert b.seq in fact_seqs
    assert a.seq not in fact_seqs
    attenuation = checkout.diagnostics["attenuation"]
    excluded = {entry["seq"]: entry for entry in attenuation["excluded"]}
    assert a.seq in excluded
    assert b.seq not in excluded
    assert excluded[a.seq]["salience_score"] < attenuation["floor"]
