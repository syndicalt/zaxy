# LongMemEval-Compatible 500 Publish Run Config

Generated report:

- `live-benchmark.json`
- `live-benchmark.md`

Run date: `2026-06-07`

Command:

```bash
EMBEDDED_GRAPH_PATH=reports/benchmarks/longmemeval-500-publish-20260607/embedded.kuzu \
python -m zaxy benchmark \
  --output-dir reports/benchmarks/longmemeval-500-publish-20260607 \
  --embedding-provider hash \
  --workload longmemeval \
  --dataset .cache/zaxy/benchmarks/longmemeval_oracle.json \
  --runs 1 \
  --limit 5 \
  --baseline-backends bm25 \
  --projection-backend embedded \
  --zaxy-backend checkout \
  --reset-graph \
  --progress
```

Frozen parameters:

- Full 500-question run by omitting `--questions`.
- Workload SHA-256:
  `90fb2307195d7e16b963a2b8a30f03b375bd42a45d41aeaa55423029dd84e3fc`.
- Embedding provider: `hash:1536`.
- Projection backend: `embedded`.
- Zaxy backend: `checkout`.
- Baseline backend: `bm25`.
- Graph path isolated to the report directory through `EMBEDDED_GRAPH_PATH`.
- Projection graph reset before the run with `--reset-graph`; no projection reuse.

Process hygiene:

- Stopped repo-local `zaxy serve`, repo-local `zaxy codex-capture`, and the
  interrupted benchmark process before rerunning.
- Left Zaxy processes from other repositories untouched.
- Removed transient `embedded.kuzu` and `embedded.kuzu.wal` files after report
  generation. The publish artifact is the JSON/Markdown report plus this
  frozen run config.
