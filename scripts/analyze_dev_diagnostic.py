"""Phase 0 analysis: judge the dev hypotheses, join with the retrieval diagnostic,
and print the retrieval-vs-reader 2x2 per category.

Run AFTER scripts/run_dev_diagnostic.py finishes generation.

    python scripts/analyze_dev_diagnostic.py

Quadrants (per question):
  retrieval_hit=T, correct=T  -> both good
  retrieval_hit=T, correct=F  -> READER failure   (gold session retrieved, answer wrong)
  retrieval_hit=F, correct=T  -> answered without gold session (other-session / synthesizable)
  retrieval_hit=F, correct=F  -> RETRIEVAL failure (gold session never retrieved)
"""
import json
import os
import pathlib
import subprocess
import sys
from collections import defaultdict

TAG = sys.argv[1] if len(sys.argv) > 1 else "dev"

REPO = pathlib.Path("/home/cheapseatsecon/Projects/Personal/zaxy")
os.environ["OPENAI_API_KEY"] = (REPO / "secrets/openai_api_key.txt").read_text().strip()

OUT = REPO / "reports/benchmarks/longmemeval-s-dev-diagnostic"
SHARD_DIR = OUT / "shards"
JUDGE = REPO / ".cache/zaxy/benchmarks/LongMemEval/src/evaluation/evaluate_qa.py"


def concat(pattern: str, dest: pathlib.Path, mode: str) -> int:
    parts = sorted(SHARD_DIR.glob(pattern))
    if mode == "jsonl":
        rows = []
        for p in parts:
            rows += [r for r in p.read_text().splitlines() if r.strip()]
        dest.write_text("\n".join(rows) + "\n")
        return len(rows)
    # json arrays -> merged array
    merged = []
    for p in parts:
        merged += json.loads(p.read_text())
    dest.write_text(json.dumps(merged))
    return len(merged)


def main() -> None:
    hyp = OUT / f"{TAG}-hyp.jsonl"
    ref = OUT / f"{TAG}-ref.json"
    diag = OUT / f"{TAG}-diagnostic.jsonl"
    n_hyp = concat(f"{TAG}-shard-*-hyp.jsonl", hyp, "jsonl")
    n_ref = concat(f"{TAG}-shard-*.json", ref, "json")
    concat(f"{TAG}-shard-*-diagnostic.jsonl", diag, "jsonl")
    print(f"hypotheses: {n_hyp}  reference: {n_ref}", flush=True)

    result_file = pathlib.Path(str(hyp) + ".eval-results-gpt-4o")
    if not result_file.exists():
        print("running gpt-4o judge...", flush=True)
        subprocess.run(
            [sys.executable, str(JUDGE), "gpt-4o", str(hyp), str(ref)],
            check=True,
            cwd=str(JUDGE.parent),
        )

    # correctness per question_id
    correct: dict[str, bool] = {}
    for r in result_file.read_text().splitlines():
        if not r.strip():
            continue
        d = json.loads(r)
        lbl = d.get("autoeval_label")
        correct[d["question_id"]] = bool(lbl.get("label") if isinstance(lbl, dict) else lbl)

    # retrieval_hit per question_id
    retrieval: dict[str, bool] = {}
    qtype: dict[str, str] = {}
    for r in diag.read_text().splitlines():
        if not r.strip():
            continue
        d = json.loads(r)
        retrieval[d["question_id"]] = bool(d["retrieval_hit"])
        qtype[d["question_id"]] = d["question_type"]

    # 2x2 per category
    cells = defaultdict(lambda: {"RT_CT": 0, "RT_CF": 0, "RF_CT": 0, "RF_CF": 0})
    for qid in correct:
        if qid not in retrieval:
            continue
        rt = retrieval[qid]
        ct = correct[qid]
        key = ("RT" if rt else "RF") + "_" + ("CT" if ct else "CF")
        cells[qtype[qid]][key] += 1
        cells["ALL"][key] += 1

    print("\n=== retrieval-vs-reader 2x2 (RT=gold session retrieved, CT=answer correct) ===")
    hdr = f"{'category':<26} {'n':>4} {'acc':>6} {'recall':>7} | {'both':>5} {'READER-fail':>11} {'no-gold-ok':>10} {'RETR-fail':>9}"
    print(hdr)
    print("-" * len(hdr))
    for cat in sorted(cells, key=lambda c: (c != "ALL", c)):
        c = cells[cat]
        n = c["RT_CT"] + c["RT_CF"] + c["RF_CT"] + c["RF_CF"]
        if not n:
            continue
        acc = (c["RT_CT"] + c["RF_CT"]) / n
        recall = (c["RT_CT"] + c["RT_CF"]) / n
        print(
            f"{cat:<26} {n:>4} {acc:>6.3f} {recall:>7.3f} | "
            f"{c['RT_CT']:>5} {c['RT_CF']:>11} {c['RF_CT']:>10} {c['RF_CF']:>9}"
        )
    print(
        "\nlegend: READER-fail = retrieved gold but answered wrong (reader/format problem);"
        "\n        RETR-fail   = never retrieved gold and wrong (retrieval-recall problem);"
        "\n        no-gold-ok  = correct without gold session (answerable from other context)."
    )


if __name__ == "__main__":
    main()
