"""Rungs 1 & 2: token density and round-trip fidelity over real Zaxy events.

Run from the experiment dir:
    python3 run_experiments.py --log ../../.eventloom/zaxy-default.jsonl

Rung 1 (density): for ~50 declarative states, count tiktoken tokens of the glyph
encoding vs three baselines -- faithful English prose, compact JSON, pretty JSON.
Token savings is treated as a WEAK signal per the goal.

Rung 2 (fidelity, GATE): event -> fact -> glyph -> fact -> content, compared to
the event's declarative content (type, actor, thread, payload). Run over a large
corpus sample. Reports representable coverage and round-trip pass rate. The
integrity envelope (seq/timestamp/hashes/security) is excluded by design and
reported as out-of-scope, not as failure.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

import tiktoken

from notation import (
    ENVELOPE_FIELDS,
    MemoryFact,
    decode,
    encode,
    event_content,
    event_to_fact,
    fact_to_content,
)

ENC = tiktoken.get_encoding("cl100k_base")


def ntok(s: str) -> int:
    return len(ENC.encode(s))


# Structured, human-meaningful event types -- the declarative subset.
DECLARATIVE_TYPES = [
    "goal.created", "decision.made", "task.completed", "task.started",
    "task.progress", "memory.checkout.completed", "memory.reinforced",
    "observation.recorded", "benchmark.completed", "workspace.instructions.updated",
    "issue.diagnosed", "verification.recorded", "handoff.created", "session.ended",
]


def load_events(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def render_prose(fact: MemoryFact) -> str:
    """Faithful, deterministic English serialization of a fact (lossless baseline)."""
    bits = [f"In session {fact.domain}, {fact.actor} recorded a {fact.etype} event."]
    if fact.entity is not None:
        bits.append(f"It concerns {fact.entity}.")
    for k, v in fact.attrs.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        bits.append(f"The {k} is {v}.")
    for rel in fact.relations:
        bits.append(f"It precedes {rel}.")
    for inv in fact.invalidations:
        bits.append(f"It invalidates {inv}.")
    return " ".join(bits)


def representable(event: dict[str, Any]) -> bool:
    payload = event.get("payload", {})
    return isinstance(payload, dict)


def roundtrip_ok(event: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        content_in = event_content(event)
        s = encode(event_to_fact(event))
        content_out = fact_to_content(decode(s))
        if content_in == content_out:
            return True, None
        return False, "content mismatch"
    except Exception as exc:  # noqa: BLE001 -- want the failure reason
        return False, f"{type(exc).__name__}: {exc}"


def pick_declarative_states(events: list[dict[str, Any]], target: int = 50) -> list[dict]:
    by_type: dict[str, list[dict]] = {t: [] for t in DECLARATIVE_TYPES}
    for e in events:
        t = e.get("type")
        if t in by_type and representable(e):
            by_type[t].append(e)
    # Round-robin across present types for a balanced sample.
    present = [t for t in DECLARATIVE_TYPES if by_type[t]]
    picked: list[dict] = []
    idx = 0
    while len(picked) < target and present:
        t = present[idx % len(present)]
        if by_type[t]:
            picked.append(by_type[t].pop(0))
        else:
            present.remove(t)
            continue
        idx += 1
    return picked


def rung1_density(states: list[dict]) -> dict[str, Any]:
    rows = []
    for e in states:
        fact = event_to_fact(e)
        glyph = encode(fact)
        prose = render_prose(fact)
        content = event_content(e)
        json_compact = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        json_pretty = json.dumps(content, ensure_ascii=False, indent=2)
        rows.append({
            "type": e.get("type"),
            "glyph": ntok(glyph),
            "prose": ntok(prose),
            "json_compact": ntok(json_compact),
            "json_pretty": ntok(json_pretty),
        })

    def ratio(base: str) -> list[float]:
        return [r[base] / r["glyph"] for r in rows if r["glyph"] > 0]

    summary = {"n": len(rows)}
    for base in ("prose", "json_compact", "json_pretty"):
        rs = ratio(base)
        summary[base] = {
            "median_ratio_vs_glyph": round(statistics.median(rs), 3),
            "mean_ratio_vs_glyph": round(statistics.fmean(rs), 3),
            "min": round(min(rs), 3),
            "max": round(max(rs), 3),
        }
    summary["total_tokens"] = {
        base: sum(r[base] for r in rows)
        for base in ("glyph", "prose", "json_compact", "json_pretty")
    }
    return {"summary": summary, "rows": rows}


def rung2_fidelity(events: list[dict], sample_cap: int = 8000) -> dict[str, Any]:
    sample = events[:sample_cap] if len(events) > sample_cap else events
    total = len(sample)
    repr_ok = 0
    rt_pass = 0
    failures: dict[str, int] = {}
    fail_examples: list[dict] = []
    by_type_pass: dict[str, list[int]] = {}

    for e in sample:
        t = e.get("type", "?")
        bucket = by_type_pass.setdefault(t, [0, 0])  # [pass, total]
        bucket[1] += 1
        if not representable(e):
            failures["unrepresentable_payload"] = failures.get("unrepresentable_payload", 0) + 1
            continue
        repr_ok += 1
        ok, reason = roundtrip_ok(e)
        if ok:
            rt_pass += 1
            bucket[0] += 1
        else:
            failures[reason or "?"] = failures.get(reason or "?", 0) + 1
            if len(fail_examples) < 5:
                fail_examples.append({"type": t, "reason": reason, "seq": e.get("seq")})

    # round-trip pass rate among representable events
    rt_rate_repr = (rt_pass / repr_ok) if repr_ok else 0.0
    rt_rate_all = (rt_pass / total) if total else 0.0
    type_rates = {
        t: {"pass": p, "total": n, "rate": round(p / n, 4)}
        for t, (p, n) in sorted(by_type_pass.items(), key=lambda kv: -kv[1][1])
    }
    return {
        "sampled": total,
        "representable": repr_ok,
        "representable_pct": round(100 * repr_ok / total, 2) if total else 0,
        "roundtrip_pass": rt_pass,
        "roundtrip_rate_among_representable": round(rt_rate_repr, 6),
        "roundtrip_rate_among_all": round(rt_rate_all, 6),
        "failures": failures,
        "fail_examples": fail_examples,
        "by_type": type_rates,
        "envelope_excluded_by_design": sorted(ENVELOPE_FIELDS),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    events = load_events(args.log)
    states = pick_declarative_states(events, target=50)
    r1 = rung1_density(states)
    r2 = rung2_fidelity(events)

    result = {
        "log": args.log,
        "total_events": len(events),
        "rung1_density": r1["summary"],
        "rung2_fidelity": r2,
    }
    with open(args.out, "w") as fh:
        json.dump({"meta": result, "rung1_rows": r1["rows"]}, fh, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
