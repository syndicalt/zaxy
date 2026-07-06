"""Validate the full real LongMemEval_S pipeline on a small slice via the codex
shim: Zaxy retrieval (hash embeddings) -> reader prompt -> codex -> hypothesis.

Usage: python scripts/codex_slice.py <n_questions>
Requires the codex shim running on 127.0.0.1:8899.
"""
import asyncio
import os
import pathlib
import sys
import time

# all-MiniLM-L6-v2 is 384-dim; align settings before zaxy imports (settings cache).
os.environ.setdefault("EMBEDDING_DIMENSION", "384")

REPO = pathlib.Path("/home/cheapseatsecon/Projects/Personal/zaxy")
sys.path.insert(0, str(REPO))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EMB = sys.argv[2] if len(sys.argv) > 2 else "local"
S = REPO / ".cache/zaxy/benchmarks/LongMemEval/data/longmemeval_s_cleaned.json"
OUT = REPO / f"reports/benchmarks/longmemeval-s-codex-slice-{EMB}"
OUT.mkdir(parents=True, exist_ok=True)

from zaxy_benchmarks.longmembench import generate_longmembench_hypotheses  # noqa: E402


async def main() -> None:
    t0 = time.time()
    await generate_longmembench_hypotheses(
        dataset_path=S,
        output_path=OUT / "zaxy-hypotheses.jsonl",
        report_path=OUT / "generation-report.json",
        questions=N,
        limit=10,
        answer_mode="openai-compatible",
        model="gpt-5",
        base_url="http://127.0.0.1:8899/v1",  # codex shim
        api_key="codex-shim",
        embedding_provider=EMB,
        projection_backend="embedded",
        resume=False,
    )
    dt = time.time() - t0
    print(f"\n{N} questions in {dt:.0f}s (~{dt/N:.0f}s/q); extrapolated S(500) ~{dt/N*500/3600:.1f}h")
    for line in (OUT / "zaxy-hypotheses.jsonl").read_text().splitlines():
        import json
        h = json.loads(line)
        print("  ", h.get("question_id"), "->", repr(str(h.get("hypothesis", ""))[:110]))


if __name__ == "__main__":
    asyncio.run(main())
