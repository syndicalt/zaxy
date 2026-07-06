"""Honest full-haystack LongMemEval run (S or M).

Usage: python scripts/run_longmemeval.py {s|m}

Real full haystack, real OpenAI embeddings, gpt-5 reader over Zaxy-retrieved
context. Resumable: safe to re-launch; already-answered questions are skipped.
Generation only — the official evaluate_qa.py judge is a separate stage.
"""
import asyncio
import pathlib
import sys
import time

REPO = pathlib.Path("/home/cheapseatsecon/Projects/Personal/zaxy")
sys.path.insert(0, str(REPO))

VARIANT = (sys.argv[1] if len(sys.argv) > 1 else "s").lower()
if VARIANT not in ("s", "m"):
    raise SystemExit("variant must be 's' or 'm'")

DATA = REPO / f".cache/zaxy/benchmarks/LongMemEval/data/longmemeval_{VARIANT}_cleaned.json"
KEY = (REPO / "secrets/openai_api_key.txt").read_text().strip()
RUN_DIR = REPO / f"reports/benchmarks/longmemeval-{VARIANT}-full-gpt5-openai-emb"
RUN_DIR.mkdir(parents=True, exist_ok=True)

from zaxy_benchmarks.longmembench import generate_longmembench_hypotheses  # noqa: E402


async def main() -> None:
    t0 = time.time()
    report = await generate_longmembench_hypotheses(
        dataset_path=DATA,
        output_path=RUN_DIR / "zaxy-hypotheses.jsonl",
        report_path=RUN_DIR / "generation-report.json",
        questions=None,
        limit=10,
        answer_mode="openai-compatible",
        model="gpt-5",
        base_url="https://api.openai.com/v1",
        api_key=KEY,
        embedding_provider="openai",
        embedding_cache=REPO / f".cache/zaxy/longmemeval-{VARIANT}-openai-emb-cache.json",
        projection_backend="embedded",
        resume=True,
        fsync_rows=True,
        provider_retries=5,
    )
    print(f"DONE {VARIANT.upper()} generation in {(time.time()-t0)/3600:.2f}h", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
