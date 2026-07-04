import asyncio, os, pathlib, sys, time, json
REPO = pathlib.Path("/home/cheapseatsecon/Projects/Personal/zaxy"); sys.path.insert(0, str(REPO))
os.environ["OPENAI_API_KEY"] = (REPO/"secrets/openai_api_key.txt").read_text().strip()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
S = REPO/".cache/zaxy/benchmarks/LongMemEval/data/longmemeval_s_cleaned.json"
OUT = REPO/"reports/benchmarks/longmemeval-s-openai-slice"; OUT.mkdir(parents=True, exist_ok=True)
from zaxy_benchmarks.longmembench import generate_longmembench_hypotheses  # noqa: E402
async def main():
    t0=time.time()
    await generate_longmembench_hypotheses(
        dataset_path=S, output_path=OUT/"hyp.jsonl", report_path=OUT/"report.json",
        questions=N, limit=10, answer_mode="openai-compatible", model="gpt-4o",
        base_url="https://api.openai.com/v1", api_key=os.environ["OPENAI_API_KEY"],
        embedding_provider="openai", embedding_cache=REPO/".cache/zaxy/lme-s-openai-emb.json",
        projection_backend="embedded", resume=False)
    dt=time.time()-t0
    print(f"\n{N} q in {dt:.0f}s (~{dt/N:.0f}s/q); S(500) sharded ~{dt/N*500/3600:.1f}h")
    for l in (OUT/"hyp.jsonl").read_text().splitlines():
        h=json.loads(l); print("  ", h["question_id"], "->", repr(str(h["hypothesis"])[:100]))
asyncio.run(main())
