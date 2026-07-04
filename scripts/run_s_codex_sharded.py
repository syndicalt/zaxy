"""Full LongMemEval_S via the Codex shim, run in memory-bounded shards.

The harness ingests every question's haystack into one in-memory projection, so
running all 500 S questions (~24k sessions) at once OOMs. This splits the dataset
into SHARD_SIZE-question shards, runs each in its own workload (built and torn
down per shard so peak memory stays ~one shard), and concatenates the
hypotheses. Resumable: a shard whose hypotheses already exist is skipped.

Hash retrieval + Codex reader (shim on :8899). Judge is a separate stage.
"""
import asyncio
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("EMBEDDING_DIMENSION", "384")

REPO = pathlib.Path("/home/cheapseatsecon/Projects/Personal/zaxy")
sys.path.insert(0, str(REPO))

SHARD_SIZE = 40
S = REPO / ".cache/zaxy/benchmarks/LongMemEval/data/longmemeval_s_cleaned.json"
OUT = REPO / "reports/benchmarks/longmemeval-s-full-codex-hash"
SHARD_DIR = OUT / "shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

from zaxy_benchmarks.longmembench import generate_longmembench_hypotheses  # noqa: E402


async def main() -> None:
    dataset = json.loads(S.read_text())
    shards = [dataset[i : i + SHARD_SIZE] for i in range(0, len(dataset), SHARD_SIZE)]
    print(f"{len(dataset)} questions -> {len(shards)} shards of {SHARD_SIZE}", flush=True)
    t0 = time.time()
    for idx, shard in enumerate(shards):
        shard_file = SHARD_DIR / f"s-shard-{idx:02d}.json"
        shard_hyp = SHARD_DIR / f"s-shard-{idx:02d}-hyp.jsonl"
        if shard_hyp.exists() and sum(1 for _ in shard_hyp.open()) == len(shard):
            print(f"shard {idx}: done, skipping", flush=True)
            continue
        shard_file.write_text(json.dumps(shard))
        st = time.time()
        await generate_longmembench_hypotheses(
            dataset_path=shard_file,
            output_path=shard_hyp,
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
        print(f"shard {idx}/{len(shards)-1} done in {time.time()-st:.0f}s "
              f"(elapsed {(time.time()-t0)/60:.0f}m)", flush=True)

    combined = OUT / "zaxy-hypotheses.jsonl"
    with combined.open("w") as out:
        for idx in range(len(shards)):
            sh = SHARD_DIR / f"s-shard-{idx:02d}-hyp.jsonl"
            if sh.exists():
                out.write(sh.read_text())
    total = sum(1 for _ in combined.open())
    print(f"DONE: {total}/{len(dataset)} hypotheses in {(time.time()-t0)/3600:.2f}h -> {combined}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
