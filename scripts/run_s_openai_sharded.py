"""Full 500-question LongMemEval_S via the real OpenAI API, memory-bounded shards.

gpt-4o reader + text-embedding-3-small (batched) semantic retrieval. ~40-question
shards built/torn down independently, concatenated. Resumable. Generation only.
"""
import asyncio
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path("/home/cheapseatsecon/Projects/Personal/zaxy")
sys.path.insert(0, str(REPO))
os.environ["OPENAI_API_KEY"] = (REPO / "secrets/openai_api_key.txt").read_text().strip()

SHARD_SIZE = 40
S = REPO / ".cache/zaxy/benchmarks/LongMemEval/data/longmemeval_s_cleaned.json"
OUT = REPO / "reports/benchmarks/longmemeval-s-full-openai-gpt4o"
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
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            embedding_provider="openai",
            embedding_cache=REPO / f".cache/zaxy/lme-s-openai-emb-shard{idx:02d}.json",
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
    print(f"DONE: {total}/{len(dataset)} hypotheses in {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
