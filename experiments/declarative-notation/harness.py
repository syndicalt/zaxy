"""Rung 3-4 harness: prompt assembly, model clients, scorers.

Default mode is --dry-run: builds every prompt, saves them for inspection, and
SELF-TESTS the scorers against ground truth (the correct answer must score pass;
a known-wrong answer must score fail) -- all with ZERO model calls. Run this
first to inspect exactly what would be sent and to trust the scoring logic.

To actually run (spends model budget), pass --provider {anthropic,openai,manual}.
  anthropic: needs ANTHROPIC_API_KEY; --model defaults to claude-opus-4-8.
  openai:    needs OPENAI_API_KEY (+ optional OPENAI_BASE_URL for Codex/compatible).
  manual:    writes prompts to prompts/, reads responses from responses/ (paste-in).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Callable

# ---------------------------------------------------------------- prompts

R3_SYSTEM = (
    "You are reading a single compact memory-state record. Answer using ONLY the "
    "information in that record. Reply with just the answer, no explanation."
)
R4_SYSTEM = (
    "You have an active memory record describing current session state. Treat it as "
    "authoritative and act on it when answering. Reply with just the answer."
)


def r3_prompt(item: dict) -> str:
    return f"MEMORY RECORD:\n{item['rendered']}\n\n{item['question']}"


def r4_prompt(item: dict, filler: str, context: str) -> str:
    if item["form"] == "none":
        return item["task"]
    block = f"[memory-state]\n{item['rendered']}\n[/memory-state]"
    if context == "buried":
        return f"{block}\n\n{filler}\n\n{item['task']}"
    return f"{block}\n\n{item['task']}"


# ---------------------------------------------------------------- scorers

def _norm(s: str) -> str:
    s = s.strip().lower()
    s = s.strip("\"'`.,!?():[]{}<>«»")
    return s.strip()


def score_r3(item: dict, response: str) -> bool:
    gt = _norm(item["ground_truth"])
    resp = _norm(response)
    if gt == "none":
        return resp in {"none", "n/a", "nothing", ""} or "none" in resp
    return gt == resp or gt in resp.split() or resp == gt


def _mentioned(token: str, text: str) -> bool:
    """True if token appears as a standalone id/word (hyphen-aware boundaries), so
    a trap like 'zaxy' is NOT matched inside the correct id 'zaxy-default'."""
    return re.search(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", text) is not None


def score_r4(item: dict, response: str) -> bool:
    resp = response.lower()
    ttype = item["ttype"]
    correct = item["correct"].lower()
    trap = item["trap"].lower() if item.get("trap") else None
    if ttype == "recommended_action":
        return any(k in resp for k in ("checkout", "memory_checkout", "refresh memory", "refresh"))
    if ttype in ("use_session", "use_fact_avoid_invalidated"):
        return _mentioned(correct, resp) and (trap is None or not _mentioned(trap, resp))
    if ttype == "use_fact":
        return _mentioned(correct, resp)
    raise ValueError(f"unknown ttype {ttype}")


# ---------------------------------------------------------------- model clients

def make_client(provider: str, model: str) -> Callable[[str, str], str]:
    if provider == "anthropic":
        import anthropic  # lazy
        client = anthropic.Anthropic()

        def call(system: str, user: str) -> str:
            msg = client.messages.create(
                model=model, max_tokens=128, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return call

    if provider == "openai":
        from openai import OpenAI  # lazy
        client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

        def call(system: str, user: str) -> str:
            r = client.chat.completions.create(
                model=model, max_tokens=128,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return r.choices[0].message.content or ""
        return call

    if provider == "manual":
        os.makedirs("prompts", exist_ok=True)
        os.makedirs("responses", exist_ok=True)

        def call(system: str, user: str) -> str:  # noqa: ARG001
            raise RuntimeError(
                "manual mode: prompts written to prompts/; add matching responses/ "
                "files then re-run with --provider manual --collect"
            )
        return call

    raise ValueError(f"unknown provider {provider}")


# ---------------------------------------------------------------- stats

def wilson(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return (round(center - half, 4), round(center + half, 4))


# ---------------------------------------------------------------- dry run

def selftest_scorers(fx: dict) -> dict:
    """Feed each scorer the ground-truth answer (expect pass) and a wrong answer
    (expect fail). Validates scoring logic with no model."""
    r3_pass = r3_fail = 0
    for it in fx["rung3"]:
        if score_r3(it, it["ground_truth"]):
            r3_pass += 1
        wrong = "zzdummy" if it["ground_truth"].lower() != "none" else "memory_checkout"
        if not score_r3(it, wrong):
            r3_fail += 1
    r4_pass = r4_fail = 0
    for it in fx["rung4"]:
        if score_r4(it, it["correct"]):
            r4_pass += 1
        if it["ttype"] == "recommended_action":
            wrong = "do nothing, just answer"
        elif it.get("trap"):
            wrong = it["trap"]
        else:
            wrong = "zzdummy"
        if not score_r4(it, wrong):
            r4_fail += 1
    return {
        "rung3_total": len(fx["rung3"]),
        "rung3_truth_scored_pass": r3_pass,
        "rung3_wrong_scored_fail": r3_fail,
        "rung4_total": len(fx["rung4"]),
        "rung4_truth_scored_pass": r4_pass,
        "rung4_wrong_scored_fail": r4_fail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="fixtures.json")
    ap.add_argument("--filler", default="filler.txt")
    ap.add_argument("--rung", choices=["3", "4"], default="3")
    ap.add_argument("--context", choices=["fresh", "buried", "both"], default="both")
    ap.add_argument("--provider", choices=["anthropic", "openai", "manual"])
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap items (smoke test)")
    args = ap.parse_args()

    fx = json.load(open(args.fixtures))
    filler = open(args.filler).read() if os.path.exists(args.filler) else ""

    if args.dry_run or not args.provider:
        st = selftest_scorers(fx)
        os.makedirs("dryrun_prompts", exist_ok=True)
        # dump a few sample prompts per rung/form for inspection
        samples = []
        for it in fx["rung3"][:6]:
            samples.append({"rung": 3, "form": it["form"], "qtype": it["qtype"],
                            "prompt": r3_prompt(it), "ground_truth": it["ground_truth"]})
        for it in fx["rung4"][:8]:
            for ctx in ("fresh", "buried"):
                p = r4_prompt(it, filler, ctx)
                samples.append({"rung": 4, "form": it["form"], "ttype": it["ttype"],
                                "context": ctx, "correct": it["correct"], "trap": it.get("trap"),
                                "prompt_preview": p[:400] + ("..." if len(p) > 400 else ""),
                                "prompt_chars": len(p)})
        json.dump(samples, open("dryrun_prompts/samples.json", "w"), ensure_ascii=False, indent=2)
        print("=== DRY RUN (no model calls) ===")
        print("scorer self-test:")
        print(json.dumps(st, indent=2))
        ok = (st["rung3_truth_scored_pass"] == st["rung3_total"]
              and st["rung3_wrong_scored_fail"] == st["rung3_total"]
              and st["rung4_truth_scored_pass"] == st["rung4_total"]
              and st["rung4_wrong_scored_fail"] == st["rung4_total"])
        print(f"\nSCORERS {'VALID' if ok else 'HAVE GAPS — review'}: "
              "ground-truth passes and wrong answers fail on all items"
              if ok else "SCORERS HAVE GAPS — inspect counts above")
        print("sample prompts written to dryrun_prompts/samples.json")
        return

    # live run
    call = make_client(args.provider, args.model)
    results = []
    if args.rung == "3":
        items = fx["rung3"][: args.limit or None]
        for it in items:
            resp = call(R3_SYSTEM, r3_prompt(it))
            results.append({**{k: it[k] for k in ("id", "form", "qtype")},
                            "ok": score_r3(it, resp), "response": resp})
    else:
        contexts = ["fresh", "buried"] if args.context == "both" else [args.context]
        items = fx["rung4"][: args.limit or None]
        for it in items:
            for ctx in contexts:
                if it["form"] == "none" and ctx == "buried":
                    continue  # control runs once
                resp = call(R4_SYSTEM, r4_prompt(it, filler, ctx))
                results.append({**{k: it[k] for k in ("id", "form", "ttype")},
                                "context": ctx, "ok": score_r4(it, resp), "response": resp})

    # aggregate
    cells: dict[tuple, list[int]] = {}
    for r in results:
        key = (r["form"], r.get("context", "-"))
        cells.setdefault(key, [0, 0])
        cells[key][1] += 1
        cells[key][0] += int(r["ok"])
    agg = {}
    for (form, ctx), (p, n) in sorted(cells.items()):
        rate = p / n if n else 0
        agg[f"{form}/{ctx}"] = {"pass": p, "n": n, "rate": round(rate, 4),
                                "wilson95": wilson(rate, n)}
    out = {"provider": args.provider, "model": args.model, "rung": args.rung,
           "aggregate": agg, "results": results}
    fn = f"rung{args.rung}_{args.provider}_{args.model.replace('/', '_')}.json"
    json.dump(out, open(fn, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
