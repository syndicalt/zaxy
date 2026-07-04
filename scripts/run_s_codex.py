"""Full 500-question LongMemEval_S run via the Codex shim.

Hash retrieval + Codex reader (through the OpenAI-compatible shim on :8899).
Resumable; generation only (the official judge is a separate stage).
"""
import asyncio
import os
import pathlib
import sys
import time

os.environ.setdefault("EMBEDDING_DIMENSION", "384")

REPO = pathlib.Path("/home/cheapseatsecon/Projects/Personal/zaxy")
sys.path.insert(0, str(REPO))

S = REPO / ".cache/zaxy/benchmarks/LongMemEval/data/longmemeval_s_cleaned.json"
OUT = REPO / "reports/benchmarks/longmemeval-s-full-codex-hash"
OUT.mkdir(parents=True, exist_ok=True)

from zaxy_benchmarks.longmembench import generate_longmembench_hypotheses  # noqa: E402


async def main() -> None:
    t0 = time.time()
    await generate_longmembench_hypotheses(
        dataset_path=S,
        output_path=OUT / "zaxy-hypotheses.jsonl",
        report_path=OUT / "generation-report.json",
        questions=None,
        limit=10,
        answer_mode="openai-compatible",
        model="gpt-5",
        base_url="http://127.0.0.1:8899/v1",
        api_key="codex-shim",
        embedding_provider="hash",
        projection_backend="embedded",
        resume=True,
        fsync_rows=True,
        provider_retries=5,
    )
    print(f"DONE full S generation in {(time.time() - t0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
