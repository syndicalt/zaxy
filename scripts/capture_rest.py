"""Capture retrieved contexts for the LongMemEval-S questions not yet on disk,
so we can assemble a full-500 set (reusing the 259 dev+held-out contexts already
captured). pure_reader, persists contexts, sharded, resumable.

    python scripts/capture_rest.py
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
OUT = REPO / "reports/benchmarks/longmemeval-s-dev-diagnostic"
SHARD_DIR = OUT / "shards"

from zaxy_benchmarks.longmembench import generate_longmembench_hypotheses  # noqa: E402


def captured_ids() -> set[str]:
    ids: set[str] = set()
    for glob in ("pure40-shard-*-contexts.jsonl", "test-shard-*-contexts.jsonl", "rest-shard-*-contexts.jsonl"):
        for p in SHARD_DIR.glob(glob):
            for r in p.read_text().splitlines():
                if r.strip():
                    ids.add(json.loads(r)["question_id"])
    return ids


async def main() -> None:
    dataset = json.loads(S.read_text())
    have = captured_ids()
    missing = [q for q in dataset if q["question_id"] not in have]
    missing.sort(key=lambda q: q["question_id"])
    print(f"{len(dataset)} total, {len(have)} already captured, {len(missing)} to capture", flush=True)
    shards = [missing[i : i + SHARD_SIZE] for i in range(0, len(missing), SHARD_SIZE)]
    t0 = time.time()
    for idx, shard in enumerate(shards):
        shard_file = SHARD_DIR / f"rest-shard-{idx:02d}.json"
        shard_hyp = SHARD_DIR / f"rest-shard-{idx:02d}-hyp.jsonl"
        shard_ctx = SHARD_DIR / f"rest-shard-{idx:02d}-contexts.jsonl"
        if shard_hyp.exists() and sum(1 for _ in shard_hyp.open()) == len(shard):
            print(f"shard {idx}: done, skipping", flush=True)
            continue
        shard_file.write_text(json.dumps(shard))
        st = time.time()
        await generate_longmembench_hypotheses(
            dataset_path=shard_file,
            output_path=shard_hyp,
            contexts_path=shard_ctx,
            reader_context_limit=25,
            questions=None,
            limit=10,
            answer_mode="openai-compatible",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            embedding_provider="openai",
            embedding_cache=REPO / f".cache/zaxy/lme-s-rest-emb-shard{idx:02d}.json",
            projection_backend="embedded",
            resume=True,
            fsync_rows=True,
            provider_retries=8,
            pure_reader=True,
        )
        print(
            f"shard {idx}/{len(shards)-1} done in {time.time()-st:.0f}s "
            f"(elapsed {(time.time()-t0)/60:.0f}m)",
            flush=True,
        )
    print("capture-rest complete", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
