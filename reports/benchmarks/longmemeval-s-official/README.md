# Zaxy on LongMemEval-S — full-haystack, official judge

A reproducible LongMemEval-S answer-accuracy result for [Zaxy](https://zaxy.io),
an event-sourced agent-memory system. Full-haystack retrieval (no oracle), the
**official** LongMemEval GPT-4o judge, a general reader prompt with **no answer
hints**. Every number here is reproducible from the committed artifacts +
`REPRODUCE.md`.

## Headline

| Reader | Answer accuracy | Scope |
|--------|----------------:|-------|
| **gpt-5** | **~0.90** (451/500) | full 500 |
| gpt-4o | 0.78 (101/130) | 130-question held-out sample |
| gpt-4o | *answers generated; not yet judged* | full 500 (needs a funded key to finish) |

Retrieval is not the bottleneck: gold-answer-session **Recall@5 ≈ 0.99** on the
full haystack. The gap to a perfect score is reader reasoning, not retrieval.

## Method

- **Benchmark:** LongMemEval-S, 500 questions, ~48 sessions / ~490 turns each.
  **Full-haystack** — retrieval runs against each question's real session pool,
  not a small oracle candidate set.
- **Retrieval:** Zaxy hybrid Memory Checkout (exact + BM25 + graph traversal +
  embeddings), OpenAI `text-embedding-3-small`. Top ~25 assembled context items
  fed to the reader.
- **Reader:** an LLM (gpt-4o or gpt-5) answers over the retrieved context using
  the general prompt in [`reader-prompt.txt`](reader-prompt.txt) — reasoning
  guidance only (count events, compute elapsed time, use the most-recent value,
  ground preferences). **No answer hints, no per-question tuning, no memorized
  answers.**
- **Judge:** the official `evaluation/evaluate_qa.py` with `gpt-4o-2024-08-06`,
  temperature 0. Answer accuracy (QA), **not** retrieval recall.

## Full-500 gpt-5 by category

| category | accuracy |
|---|--:|
| single-session-user | 1.000 |
| single-session-assistant | 0.982 |
| knowledge-update | 0.936 |
| temporal-reasoning | 0.895 |
| multi-session | 0.865 |
| single-session-preference | 0.633 |
| **overall (500)** | **0.902** |

## Honesty notes (read these)

- **Judge precision.** The official judge is itself an LLM. At temperature 0 it
  is near- but not perfectly deterministic: re-scoring the *identical* gpt-5
  hypotheses produced 449/500 and 451/500 on separate runs (~±0.4%). We report
  the committed artifact (451/500) and headline **~0.90**, not a spurious third
  decimal. Vendors quoting figures like 94.87% are over-precise for this metric.
- **Reader tier matters.** gpt-4o and gpt-5 are different tiers and are not
  interchangeable; only same-tier comparisons are apples-to-apples.
- **Answer accuracy, not recall.** Several published "90%+" figures are
  retrieval recall or use stronger/oracle setups. This is full-haystack QA.
- **No overfit.** A deterministic 130-question held-out subset the reader prompt
  was never tuned on scored gpt-5 0.892 / gpt-4o 0.777 — within a point of the
  full-set gpt-5 number, which is the evidence it generalizes.

## Where this sits

At the **same reader tier**, the gpt-4o result leads the one independently
published, apples-to-apples peer — **Zep / Graphiti, 0.712** (gpt-4o,
full-haystack, [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)). The gpt-5
result (~0.90) is competitive with the self-reported top tier while staying a
full-haystack, held-out, official-judge number.

## Publishing status

There is **no official LongMemEval leaderboard** to submit to — Papers with Code
was sunset (2025), and the [official repo](https://github.com/xiaowu0162/LongMemEval)
is an evaluation toolkit with no results registry. The de-facto path for a
citable result (used by Zep and Mem0) is an **arXiv tech report + this open
reproduction package**. This directory is that package; a short tech report is
the next step.

## Artifacts

- [`artifacts/gpt-5-full500-hyp.jsonl`](artifacts/gpt-5-full500-hyp.jsonl) — Zaxy's answers, gpt-5 reader.
- [`artifacts/gpt-5-full500-eval.jsonl`](artifacts/gpt-5-full500-eval.jsonl) — official judge labels.
- [`manifest.json`](manifest.json) — dataset SHA-256, models, commit, config.
- [`REPRODUCE.md`](REPRODUCE.md) — exact re-run commands.
- [`reader-prompt.txt`](reader-prompt.txt) — the reader prompt, verbatim.

The retrieved contexts (~1.3 GB) are reproducible from the public dataset via the
harness and are not committed. The dataset itself is the standard LongMemEval_S
(SHA in `manifest.json`); download it from the benchmark source.
