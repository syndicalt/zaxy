"""Fixture generator for rungs 3-4 (no model calls).

Builds a grounded item bank of declarative memory states, each rendered in three
lossless surface forms (prose / json / glyph), plus comprehension questions and
adherence tasks with programmatic ground truth, plus real transcript filler for
the buried-context condition.

The states are grounded in real zaxy facts from this project (session
consolidation, checkout staleness, project decisions) and varied deterministically
for sample size. Run: `python3 fixtures.py --log ../../.eventloom/zaxy-default.jsonl`
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from notation import MemoryFact, encode


# ---- three lossless renderings of one MemoryFact --------------------------

def render_glyph(fact: MemoryFact) -> str:
    return encode(fact)


def render_json(fact: MemoryFact) -> str:
    obj: dict[str, Any] = {
        "domain": fact.domain,
        "type": fact.etype,
        "actor": fact.actor,
    }
    if fact.entity is not None:
        obj["entity"] = fact.entity
    if fact.attrs:
        obj["attrs"] = fact.attrs
    if fact.relations:
        obj["recommends"] = fact.relations
    if fact.invalidations:
        obj["invalidated"] = fact.invalidations
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def render_prose(fact: MemoryFact) -> str:
    bits = [f"In session {fact.domain}, {fact.actor} recorded a {fact.etype}."]
    if fact.entity is not None:
        bits.append(f"It concerns {fact.entity}.")
    for k, v in fact.attrs.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        bits.append(f"The {k} is {v}.")
    for rel in fact.relations:
        bits.append(f"The recommended next action is {rel}.")
    for inv in fact.invalidations:
        bits.append(f"{inv} is invalidated and must not be used.")
    return " ".join(bits)


RENDERERS = {"prose": render_prose, "json": render_json, "glyph": render_glyph}


# ---- grounded state templates ---------------------------------------------

SESSIONS = ["zaxy-default", "personal-default", "zaxyhub-default", "eventloom-default"]
OLD_SESSIONS = ["zaxy", "default", "demo-default", "build_session"]
BACKENDS = ["Neo4j", "embedded-kuzu", "pggraph", "LadybugDB"]
OLD_BACKENDS = ["HelixDB", "Postgres", "SQLite", "in-memory"]


def _state_bank(rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Deterministic bank of declarative states + their semantic ground truth."""
    items: list[dict[str, Any]] = []
    for i in range(n):
        session = SESSIONS[i % len(SESSIONS)]
        old = OLD_SESSIONS[i % len(OLD_SESSIONS)]
        backend = BACKENDS[i % len(BACKENDS)]
        old_backend = OLD_BACKENDS[i % len(OLD_BACKENDS)]
        since = rng.randint(9, 14000)
        seq = rng.randint(100, 80000)
        kind = i % 4

        if kind == 0:  # checkout staleness + recommended action
            fact = MemoryFact(
                domain=session, etype="memory.checkout.state", actor="zaxy-memory",
                attrs={"stale": True, "since_ev": since, "last_checkout_seq": seq},
                relations=["memory_checkout"],
            )
            truth = {"domain": session, "scalar_field": "stale", "scalar_value": "True",
                     "action": "memory_checkout", "invalid": None}
        elif kind == 1:  # session consolidation: canonical + invalidated
            fact = MemoryFact(
                domain=session, etype="session.policy", actor="claude",
                attrs={"canonical": True, "writes_go_to": session},
                invalidations=[old],
            )
            truth = {"domain": session, "scalar_field": "writes_go_to",
                     "scalar_value": session, "action": None, "invalid": old}
        elif kind == 2:  # project decision: default backend (+ superseded one)
            fact = MemoryFact(
                domain=session, etype="decision.made", actor="codex",
                entity=f"graph-backend@{seq}",
                attrs={"default_backend": backend, "phase": "beta"},
                invalidations=[old_backend],
            )
            truth = {"domain": session, "scalar_field": "default_backend",
                     "scalar_value": backend, "action": None, "invalid": old_backend}
        else:  # goal with recommended next step
            fact = MemoryFact(
                domain=session, etype="goal.created", actor="claude",
                entity=f"goal@{seq}",
                attrs={"status": "open", "priority": (i % 3) + 1},
                relations=["run_rung3_comprehension"],
            )
            truth = {"domain": session, "scalar_field": "status", "scalar_value": "open",
                     "action": "run_rung3_comprehension", "invalid": old_backend if False else None}
        items.append({"id": f"s{i:03d}", "kind": kind, "fact": fact, "truth": truth})
    return items


# ---- rung 3 comprehension items -------------------------------------------

QUESTIONS = {
    "scalar": "Question: What is the value of the '{field}' field? Answer with only the value.",
    "domain": "Question: Which session/domain does this state belong to? Answer with only the id.",
    "action": "Question: What action does this state recommend taking next? Answer with only the action name, or 'none'.",
    "invalid": "Question: What does this state mark as invalidated / must-not-use? Answer with only the value, or 'none'.",
}


def build_rung3(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for st in states:
        fact, truth = st["fact"], st["truth"]
        qa = {
            "scalar": (QUESTIONS["scalar"].format(field=truth["scalar_field"]), truth["scalar_value"]),
            "domain": (QUESTIONS["domain"], truth["domain"]),
            "action": (QUESTIONS["action"], truth["action"] or "none"),
            "invalid": (QUESTIONS["invalid"], truth["invalid"] or "none"),
        }
        for qtype, (question, answer) in qa.items():
            for form, renderer in RENDERERS.items():
                out.append({
                    "id": f"{st['id']}-{qtype}-{form}",
                    "state_id": st["id"], "qtype": qtype, "form": form,
                    "rendered": renderer(fact),
                    "question": question,
                    "ground_truth": str(answer),
                })
    return out


# ---- rung 4 adherence tasks ------------------------------------------------

def build_rung4(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for st in states:
        fact, truth, kind = st["fact"], st["truth"], st["kind"]
        if kind == 0:
            task = ("Task: The user asks a question, but first decide whether you "
                    "should refresh memory. State the single action to take first, if any.")
            correct, trap, ttype = "memory_checkout", None, "recommended_action"
        elif kind == 1:
            task = ("Task: You must append a new memory event. Which session id do "
                    "you write to? Answer with only the session id.")
            correct, trap, ttype = truth["domain"], truth["invalid"], "use_session"
        elif kind == 2:
            task = ("Task: What is the current default graph projection backend for "
                    "beta? Answer with only the backend name.")
            correct, trap, ttype = truth["scalar_value"], truth["invalid"], "use_fact_avoid_invalidated"
        else:
            task = ("Task: For the highest-priority open goal, what is the recommended "
                    "next step? Answer with only the action name.")
            correct, trap, ttype = truth["action"], None, "use_fact"
        for form, renderer in RENDERERS.items():
            out.append({
                "id": f"{st['id']}-{ttype}-{form}",
                "state_id": st["id"], "ttype": ttype, "form": form,
                "rendered": renderer(fact),
                "task": task,
                "correct": str(correct),
                "trap": (str(trap) if trap is not None else None),
            })
        # control: no injected state at all
        out.append({
            "id": f"{st['id']}-{ttype}-none",
            "state_id": st["id"], "ttype": ttype, "form": "none",
            "rendered": "", "task": task,
            "correct": str(correct), "trap": (str(trap) if trap is not None else None),
        })
    return out


# ---- real transcript filler for the buried-context condition ---------------

def load_filler(log_path: str, max_chars: int = 60000) -> str:
    chunks: list[str] = []
    total = 0
    with open(log_path) as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "transcript.turn":
                continue
            payload = e.get("payload", {})
            text = payload.get("text") or payload.get("content") or json.dumps(payload, ensure_ascii=False)
            if not isinstance(text, str):
                continue
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                break
    return "\n".join(chunks)[:max_chars]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--n-states", type=int, default=24)
    ap.add_argument("--out", default="fixtures.json")
    args = ap.parse_args()

    rng = random.Random(20260613)
    states = _state_bank(rng, args.n_states)
    rung3 = build_rung3(states)
    rung4 = build_rung4(states)
    filler = load_filler(args.log)

    bank = {
        "n_states": len(states),
        "states": [
            {"id": s["id"], "kind": s["kind"], "truth": s["truth"],
             "renderings": {f: r(s["fact"]) for f, r in RENDERERS.items()}}
            for s in states
        ],
        "rung3": rung3,
        "rung4": rung4,
        "filler_chars": len(filler),
    }
    with open(args.out, "w") as fh:
        json.dump(bank, fh, ensure_ascii=False, indent=2)
    with open("filler.txt", "w") as fh:
        fh.write(filler)

    print(f"states={len(states)} rung3_items={len(rung3)} rung4_items={len(rung4)} "
          f"filler_chars={len(filler)}")
    ex = states[0]
    print("\nExample state, three forms:")
    for f, r in RENDERERS.items():
        print(f"  [{f}] {r(ex['fact'])}")


if __name__ == "__main__":
    main()
