# Zaxy 2.1.0: Cognitive Defaults, Measured First

Zaxy 2.1.0 ships the additive 2.1–2.3 agent-experience and cognitive-memory
feature set and promotes two defaults that earned the flip through their
measurement lanes.

## What Is New

The 2.1–2.3 surface is entirely additive on the unchanged immutable Eventloom
log: MCP tool listing profiles and umbrella tools, checkout token budgets with
a cache-stable consolidated prefix, the replayable salience reinforcement
ledger, encoding-specificity cues, the experimental `memory_feeling_of_knowing`
metamemory pre-check, procedure mining, vector re-embedding, and the internal
agent-experience and cognitive measurement lanes. No Eventloom envelope
changes, no projection migrations, no removed or renamed tools. The full
upgrade path, including both default flips and their opt-outs, is documented
in [docs/migration.md](../migration.md).

## Default Flips (and Their Evidence)

**`MCP_TOOL_PROFILE` now defaults to `core`.** The internal tool-adoption lane
measured the listed tool surface at 8,165 estimated tokens under `full` versus
1,344 under `core` — an 83.5% reduction — with the `memory_checkout` front
door listed first. Profiles change listing only: dispatch is never filtered,
so every tool stays callable by name, and `memory_capabilities` reports the
available-but-unlisted set. Opt out with `MCP_TOOL_PROFILE=full` or
`zaxy serve --profile full`.

**`RETRIEVAL_PROFILE` now defaults to `cognitive`.** The internal forgetting
lane measured exact cold-start parity (zero-reinforcement corpora rank
byte-identically to plain retrieval), no-recall-loss 1.0 (attenuated memories
always reachable by explicit query, and labeled), pin/authority exemptions
1.0, and ranking lift 1.0 versus 0.0 for plain. The cognitive profile composes
the same local retrieval stack as `local_fast` plus salience-blended ranking,
cue blending, and the personalized-PageRank graph walk. The graph-walk lane
separately measured multi-hop bridge lift on 6/6 entity-bridge queries (versus
0/6 under plain ranking) with zero single-hop regressions (4/4 direct queries
retained at rank 1 in both arms). Opt out with
`RETRIEVAL_PROFILE=local_fast`. Settings that leave the profile unset but
customize embedding/reranker/scoring knobs keep resolving to the `custom`
profile exactly as before.

Who is affected: clients that snapshot tool listings should refresh against
the core listing or pin `MCP_TOOL_PROFILE=full`; users with reinforcement
history (or graph structure the walk can traverse) may see ranking shifts,
while cold-start users see identical ranking.

## What Did Not Flip

- `ENCODING_GATE_ENABLED` stays `false`: the write-time gate's effect is
  unmeasured, and unmeasured defaults do not flip.
- The feeling-of-knowing surface stays experimental: its calibration lane
  Brier margin over the base-rate predictor is +0.001–0.006 — real but too
  thin to promote.
- Vector quantization stays `none` pending a production-dimension re-run of
  the vector-scale lane, although int8 is the strongest promotion candidate
  measured (recall@10 0.9938, 0.14x memory, 1.8x faster than exact at 10^5).
- `VECTOR_ANN_THRESHOLD` is **raised** from `50000` to `1000000`, keeping the
  Kuzu HNSW path effectively opt-in: the vector-scale lane measured it below
  exact search on both recall (0.90 versus the 0.95 bar at 10^5) and latency,
  with non-reproducible index builds. The path remains available by lowering
  the threshold explicitly; the default will come down only after per-query
  projected-graph reuse and recall tuning land with lane evidence.

## Fixes

A feeling-of-knowing verdict boundary was corrected: a 3-term query with one
known term scores exactly the `possible` threshold (`0.6 * (1/3) = 0.2`), but
binary floating point evaluated it just below and mislabeled it `unlikely`.
Threshold comparisons are now epsilon-tolerant; scores are unchanged.

## Upgrading

```bash
pip install --upgrade zaxy-memory
zaxy doctor
```

To keep 2.0 behavior exactly: `MCP_TOOL_PROFILE=full` and
`RETRIEVAL_PROFILE=local_fast`. Everything else upgrades without action. See
[docs/migration.md](../migration.md) and
[docs/configuration.md](../configuration.md).
