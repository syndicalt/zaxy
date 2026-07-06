# Reproducing the Zaxy LongMemEval-S result

Full-haystack LongMemEval-S, official GPT-4o judge. Every step below runs from
the Zaxy repo at the commit in `manifest.json`.

## 0. Prerequisites

- `pip install -e ".[dev]"` (Zaxy + `zaxy_benchmarks`).
- An OpenAI API key with access to the embedding model, the reader model, and
  the judge model. Export it: `export OPENAI_API_KEY=...` (never commit it).
- The **LongMemEval_S** dataset (`longmemeval_s_cleaned.json`) from the
  benchmark source in `manifest.json`. Verify it matches the recorded SHA-256:

  ```bash
  sha256sum longmemeval_s_cleaned.json   # must equal manifest.dataset.sha256
  ```

## 1. Retrieve + assemble context, and answer (per reader)

Zaxy ingests each question's haystack, retrieves via hybrid Memory Checkout,
assembles the reader context, and answers with the general reader prompt in
`reader-prompt.txt` (no answer hints). The harness persists the retrieved
contexts so the reader/judge can be re-run without re-retrieving.

```bash
# Capture retrieved contexts for all 500 questions (projection-bound, one-time).
python scripts/run_dev_diagnostic.py --pure-reader --reader-context 40   # dev split
python scripts/capture_rest.py                                            # remaining questions
# -> reports/.../shards/full500-shard-00-contexts.jsonl  (500 questions)

# Answer + judge over the persisted contexts, per reader:
python scripts/replay_reader.py --src full500 --budget 25 --prompt improved --model gpt-4o
python scripts/replay_reader.py --src full500 --budget 25 --prompt improved --model gpt-5
```

Each `replay_reader.py` run writes `replay-full500-b25-improved-<model>-hyp.jsonl`
(Zaxy's answers) and invokes the **official** judge.

## 2. Score with the official judge (standalone)

The replay calls it automatically; to re-score the committed answers directly:

```bash
python .cache/zaxy/benchmarks/LongMemEval/src/evaluation/evaluate_qa.py \
    gpt-4o \
    reports/benchmarks/longmemeval-s-official/artifacts/gpt-5-full500-hyp.jsonl \
    <path-to-longmemeval_s_cleaned.json>
```

This produces `...hyp.jsonl.eval-results-gpt-4o` — compare against the committed
`artifacts/gpt-5-full500-eval.jsonl` (expect agreement within the judge's
~±0.4% run-to-run LLM variance).

## 3. Aggregate

Answer accuracy = fraction of `autoeval_label.label == true` over the 500 rows,
overall and per `question_type`.

## Notes

- **Full-haystack, not oracle.** Retrieval runs against each question's real
  ~48-session haystack, not a small candidate pool.
- **No answer hints.** The reader prompt is general reasoning guidance
  (`reader-prompt.txt`); no memorized answers or per-question tuning.
- **Report the reader tier.** gpt-4o and gpt-5 are different tiers and not
  interchangeable; cross-tier comparisons are not apples-to-apples.
