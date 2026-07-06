"""Fast reader-only replay: sweep reader context budget / prompt over PERSISTED
retrieved contexts, with NO projection rebuild.

Requires a prior run that persisted contexts (run_dev_diagnostic.py writes
`<tag>-shard-*-contexts.jsonl`). Runs the pure gpt-4o reader over the saved
contexts at a chosen budget, recomputes retrieval_hit from the saved
answer_session_ids, runs the official gpt-4o judge, and prints the 2x2.

    python scripts/replay_reader.py --src pure40 --budget 25
"""
import argparse
import concurrent.futures as cf
import httpx
import json
import os
import pathlib
import subprocess
import sys
from collections import defaultdict

REPO = pathlib.Path("/home/cheapseatsecon/Projects/Personal/zaxy")
sys.path.insert(0, str(REPO))
os.environ["OPENAI_API_KEY"] = (REPO / "secrets/openai_api_key.txt").read_text().strip()

OUT = REPO / "reports/benchmarks/longmemeval-s-dev-diagnostic"
SHARD_DIR = OUT / "shards"
JUDGE = REPO / ".cache/zaxy/benchmarks/LongMemEval/src/evaluation/evaluate_qa.py"

from zaxy_benchmarks.longmembench import (  # noqa: E402
    _longmembench_answer_session_hits,
    _openai_compatible_answer,
)

IMPROVED_PROMPT = (
    "You answer questions from a user's long-term conversation memory. "
    "The evidence below is a set of dated snippets retrieved from many past "
    "sessions. Use only this evidence. Return only the final answer, concise, "
    "with no explanation.\n"
    "- The evidence spans MULTIPLE sessions and dates. Read all of it before "
    "deciding; the answer is often assembled from several snippets.\n"
    "- COUNTING ('how many ...'): find every distinct qualifying event in the "
    "evidence, then answer with the integer count.\n"
    "- ELAPSED TIME ('how many days/weeks/months ago', 'how long since'): "
    "compute the difference from the dated evidence and the question date; "
    "answer with the computed duration.\n"
    "- CHANGED FACTS ('current', 'most recent', 'latest'): the value may be "
    "stated several times over time; answer with the MOST RECENT value.\n"
    "- PREFERENCES ('suggest ...', 'what would I prefer'): answer with what the "
    "user would prefer, grounded in interests and constraints they stated.\n"
    "- Only answer 'I do not have enough information to answer.' if the evidence "
    "genuinely does not contain the answer. Prefer your best supported answer."
)

# v2 hardens the two residual failure modes under IMPROVED: counting (under/over
# counts across sessions) and date arithmetic. Adds explicit enumerate-then-count
# discipline and a compute-step for elapsed time, plus honest abstention when the
# evidence set looks incomplete for a count.
COUNTING_PROMPT = IMPROVED_PROMPT + (
    "\n\nEXTRA DISCIPLINE:\n"
    "- For any 'how many' question: first mentally list each distinct qualifying "
    "item with its date, deduplicate, then return the count as a bare integer. Do "
    "not estimate. If the qualifying items plausibly span dates not present in the "
    "evidence, answer 'I do not have enough information to answer.' rather than "
    "guessing a low count.\n"
    "- For elapsed-time questions: identify the two anchor dates from the evidence, "
    "then compute the difference in the unit asked (days/weeks/months). Do not "
    "confuse the question date with an event date."
)


def load_contexts(src: str) -> list[dict]:
    rows = []
    for p in sorted(SHARD_DIR.glob(f"{src}-shard-*-contexts.jsonl")):
        rows += [json.loads(r) for r in p.read_text().splitlines() if r.strip()]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="pure40", help="tag of the run that persisted contexts")
    ap.add_argument("--budget", type=int, required=True, help="reader context items to feed")
    ap.add_argument("--workers", type=int, default=4)  # gentle on the TPM rate limit
    ap.add_argument("--prompt", choices=["default", "improved", "counting"], default="default")
    ap.add_argument("--model", default="gpt-4o", help="reader model (e.g. gpt-4o, gpt-5)")
    ap.add_argument("--judge-model", default="gpt-4o", help="judge model (keep gpt-4o for comparability)")
    args = ap.parse_args()
    system_prompt = {"default": None, "improved": IMPROVED_PROMPT, "counting": COUNTING_PROMPT}[args.prompt]

    ctx_rows = load_contexts(args.src)
    if not ctx_rows:
        sys.exit(f"no persisted contexts for src={args.src!r} (run run_dev_diagnostic first)")
    print(f"loaded {len(ctx_rows)} persisted-context questions from {args.src}", flush=True)

    replay_tag = f"replay-{args.src}-b{args.budget}-{args.prompt}-{args.model}"

    def answer(row: dict) -> dict:
        try:
            hyp = _openai_compatible_answer(
                question=row["question"],
                contexts=row["contexts"],
                model=args.model,
                base_url="https://api.openai.com/v1",
                api_key=os.environ["OPENAI_API_KEY"],
                max_retries=5,
                pure_reader=True,
                reader_context_limit=args.budget,
                system_prompt=system_prompt,
                max_tokens=200,
            )
        except httpx.HTTPStatusError:
            # Rate-limit / quota / server errors must ABORT the run, never become
            # silent fallback answers that corrupt the score.
            raise
        except Exception as exc:  # a single malformed response must not kill 500
            print(f"  ! {row['question_id']} reader error: {exc}", flush=True)
            hyp = "I do not have enough information to answer."
        hits = _longmembench_answer_session_hits(row["contexts"], tuple(row["answer_session_ids"]))
        return {
            "question_id": row["question_id"],
            "hypothesis": hyp,
            "question_type": row["question_type"],
            "retrieval_hit": bool(hits),
        }

    # Resumable: a flushed per-answer sidecar survives TPM aborts. Reruns skip
    # question_ids already answered.
    progress = OUT / f"{replay_tag}-progress.jsonl"
    done: dict[str, dict] = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["question_id"]] = d
    todo = [r for r in ctx_rows if r["question_id"] not in done]
    print(f"{len(done)} already answered, {len(todo)} to go", flush=True)

    import threading
    lock = threading.Lock()
    handle = progress.open("a", encoding="utf-8")
    results: list[dict] = list(done.values())
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(answer, todo), 1):
                with lock:
                    handle.write(json.dumps(r) + "\n")
                    handle.flush()
                results.append(r)
                if i % 20 == 0:
                    print(f"  {i}/{len(todo)} answered", flush=True)
    finally:
        handle.close()

    hyp = OUT / f"{replay_tag}-hyp.jsonl"
    hyp.write_text("\n".join(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"]}) for r in results) + "\n")
    # reference must match the SRC split's questions (dev-ref for dev, test-ref for test)
    ref = OUT / f"{args.src}-ref.json"
    if not ref.exists():
        merged = []
        for p in sorted(SHARD_DIR.glob(f"{args.src}-shard-*.json")):
            merged += json.loads(p.read_text())
        ref.write_text(json.dumps(merged))

    result_file = pathlib.Path(str(hyp) + f".eval-results-{args.judge_model}")
    print(f"running {args.judge_model} judge...", flush=True)
    subprocess.run([sys.executable, str(JUDGE), args.judge_model, str(hyp), str(ref)], check=True, cwd=str(JUDGE.parent))

    correct = {}
    for r in result_file.read_text().splitlines():
        if not r.strip():
            continue
        d = json.loads(r)
        lbl = d.get("autoeval_label")
        correct[d["question_id"]] = bool(lbl.get("label") if isinstance(lbl, dict) else lbl)
    rt = {r["question_id"]: r["retrieval_hit"] for r in results}
    qt = {r["question_id"]: r["question_type"] for r in results}

    cells = defaultdict(lambda: {"RT_CT": 0, "RT_CF": 0, "RF_CT": 0, "RF_CF": 0})
    for qid in correct:
        key = ("RT" if rt[qid] else "RF") + "_" + ("CT" if correct[qid] else "CF")
        cells[qt[qid]][key] += 1
        cells["ALL"][key] += 1
    print(f"\n=== replay {replay_tag} (budget={args.budget}) ===")
    hdr = f"{'category':<26} {'n':>4} {'acc':>6} | {'both':>5} {'READER-fail':>11} {'RETR-fail':>9}"
    print(hdr); print("-" * len(hdr))
    for cat in sorted(cells, key=lambda c: (c != "ALL", c)):
        c = cells[cat]
        n = sum(c.values())
        if not n:
            continue
        acc = (c["RT_CT"] + c["RF_CT"]) / n
        print(f"{cat:<26} {n:>4} {acc:>6.3f} | {c['RT_CT']:>5} {c['RT_CF']:>11} {c['RF_CF']:>9}")


if __name__ == "__main__":
    main()
