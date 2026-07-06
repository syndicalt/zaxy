"""Phase 0 retrieval-vs-reader diagnostic over a held-out LongMemEval_S dev split.

Deterministic, category-stratified 150-question dev subset (the remaining 350 are
the untouched test set for later scoring). For each question we persist, per row,
whether any gold answer_session_id landed in the retrieved context
(`retrieval_hit`) alongside the generated hypothesis. Joining that with the judge
label yields the retrieval-vs-reader 2x2 per category.

gpt-4o reader + text-embedding-3-small (batched) semantic retrieval. ~40-question
shards, built/torn down independently. Resumable (per-row flushed).

    python scripts/run_dev_diagnostic.py            # full 150-question dev run
    python scripts/run_dev_diagnostic.py --smoke 3  # 3-question instrumentation smoke
"""
import argparse
import asyncio
import hashlib
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


def _dev_bucket(question_id: str) -> int:
    """Stable 0..9 bucket for a question id (deterministic, no RNG)."""
    return int(hashlib.md5(question_id.encode()).hexdigest(), 16) % 10


def select_dev_split(dataset: list[dict]) -> list[dict]:
    """Category-stratified ~30% dev subset, deterministic by question_id hash."""
    dev = [q for q in dataset if _dev_bucket(str(q["question_id"])) < 3]
    dev.sort(key=lambda q: str(q["question_id"]))
    return dev


def select_test_sample(dataset: list[dict], target: int = 130) -> list[dict]:
    """Deterministic, category-stratified ~`target` sample from the HELD-OUT test
    split (bucket >= 3) — questions the tuned prompt never saw."""
    test = [q for q in dataset if _dev_bucket(str(q["question_id"])) >= 3]
    by_type: dict[str, list[dict]] = {}
    for q in test:
        by_type.setdefault(q.get("question_type", "unknown"), []).append(q)
    frac = target / len(test)
    sample: list[dict] = []
    for qs in by_type.values():
        qs.sort(key=lambda q: str(q["question_id"]))
        k = max(1, round(len(qs) * frac))
        sample.extend(qs[:k])
    sample.sort(key=lambda q: str(q["question_id"]))
    return sample


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0, help="run only N questions")
    parser.add_argument(
        "--pure-reader",
        action="store_true",
        help="bypass deterministic/preference answer-candidate short-circuits; gpt-4o reads context",
    )
    parser.add_argument(
        "--reader-context",
        type=int,
        default=12,
        help="how many retrieved context items to feed the reader (default 12)",
    )
    parser.add_argument(
        "--test-sample",
        action="store_true",
        help="use the held-out stratified test sample instead of the dev split",
    )
    args = parser.parse_args()

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(S.read_text())
    dev = select_test_sample(dataset) if args.test_sample else select_dev_split(dataset)
    if args.smoke:
        dev = dev[: args.smoke]
    split_name = "held-out test-sample" if args.test_sample else "dev"
    by_type: dict[str, int] = {}
    for q in dev:
        by_type[q.get("question_type", "unknown")] = by_type.get(q.get("question_type", "unknown"), 0) + 1
    print(f"{len(dataset)} total -> {len(dev)} {split_name} questions", flush=True)
    print(f"  by type: {json.dumps(by_type, sort_keys=True)}", flush=True)

    shards = [dev[i : i + SHARD_SIZE] for i in range(0, len(dev), SHARD_SIZE)]
    if args.smoke:
        tag = f"smoke{args.smoke}"
    elif args.test_sample:
        tag = "test"
    elif args.pure_reader:
        tag = f"pure{args.reader_context}" if args.reader_context != 12 else "pure"
    else:
        tag = "dev"
    t0 = time.time()
    for idx, shard in enumerate(shards):
        shard_file = SHARD_DIR / f"{tag}-shard-{idx:02d}.json"
        shard_hyp = SHARD_DIR / f"{tag}-shard-{idx:02d}-hyp.jsonl"
        shard_diag = SHARD_DIR / f"{tag}-shard-{idx:02d}-diagnostic.jsonl"
        shard_ctx = SHARD_DIR / f"{tag}-shard-{idx:02d}-contexts.jsonl"
        done = shard_hyp.exists() and sum(1 for _ in shard_hyp.open()) == len(shard)
        if done and not args.smoke:
            print(f"shard {idx}: done, skipping", flush=True)
            continue
        shard_file.write_text(json.dumps(shard))
        st = time.time()
        await generate_longmembench_hypotheses(
            dataset_path=shard_file,
            output_path=shard_hyp,
            diagnostic_path=shard_diag,
            contexts_path=shard_ctx,
            reader_context_limit=args.reader_context,
            questions=None,
            limit=10,
            answer_mode="openai-compatible",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            embedding_provider="openai",
            # cache keyed to the split (dev vs held-out test) so pure-reader reuses
            # the warm dev embeddings and the test sample gets its own cache.
            embedding_cache=REPO
            / f".cache/zaxy/lme-s-{'test' if args.test_sample else 'dev'}-emb-shard{idx:02d}.json",
            projection_backend="embedded",
            resume=not args.smoke,
            fsync_rows=True,
            provider_retries=8,
            pure_reader=args.pure_reader,
        )
        print(
            f"shard {idx}/{len(shards)-1} done in {time.time()-st:.0f}s "
            f"(elapsed {(time.time()-t0)/60:.0f}m)",
            flush=True,
        )

    # Concatenate diagnostics for a single join with judge labels.
    combined = OUT / f"{tag}-diagnostic.jsonl"
    with combined.open("w") as out:
        for idx in range(len(shards)):
            d = SHARD_DIR / f"{tag}-shard-{idx:02d}-diagnostic.jsonl"
            if d.exists():
                out.write(d.read_text())
    hits = sum(json.loads(r)["retrieval_hit"] for r in combined.open() if r.strip())
    total = sum(1 for r in combined.open() if r.strip())
    print(f"\ndiagnostic -> {combined}", flush=True)
    print(f"retrieval_hit (gold answer session in top-{10} context): {hits}/{total}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
