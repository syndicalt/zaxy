# Honest full-haystack LongMemEval-S with an event-sourced memory system

**A reproducible technical report.** Draft for a blog post / arXiv (cs.CL) submission.

## Abstract

We report LongMemEval-S answer-accuracy for **Zaxy**, an event-sourced
agent-memory system, under a deliberately conservative protocol: full-haystack
retrieval (no oracle candidate pool), the benchmark's **official** GPT-4o judge,
and a general reader prompt containing **no answer hints**. Zaxy scores **~0.90
(451/500) with a gpt-5 reader** and **0.78 on a held-out 130-question sample with
a gpt-4o reader** — the latter above the only independently published,
same-tier peer (Zep/Graphiti, 0.712). We find retrieval is not the bottleneck
(gold-session Recall@5 ≈ 0.99); accuracy is reader-bound. We also document two
evaluation hazards we removed or disclose — an oracle-mode inflation and
answer-hint contamination we retracted from our own earlier numbers, and the
±0.4% run-to-run variance of the LLM judge — and argue the field over-reports
precision on an LLM-judged metric. All numbers are reproducible from the
committed artifacts.

## 1. Introduction

LongMemEval (Wu et al., ICLR 2025) evaluates long-term conversational memory: a
question is answered against a haystack of ~48 prior sessions (~490 turns,
~115k tokens for the -S split). Reported figures in the agent-memory space vary
wildly (60% to 96%), largely because setups differ on three axes that are often
left implicit: (i) **oracle vs. full-haystack** retrieval, (ii) **reader model
tier**, and (iii) **answer accuracy vs. retrieval recall**. We fix all three to
the hardest, most comparable choices and report the result.

## 2. System

Zaxy is an event-sourced temporal-knowledge-graph memory. An append-only,
hash-chained event log (JSONL) is the source of truth; a bi-temporal graph
projection is derived from it. Retrieval is hybrid — exact lookup, BM25,
graph traversal, and dense embeddings (`text-embedding-3-small`) — fused into a
cited **Memory Checkout** that assembles the context an agent reads. For this
evaluation each question's full haystack is ingested, retrieved over, and the
top ~25 assembled context items are passed to an LLM reader.

## 3. Method

- **Benchmark:** LongMemEval-S, 500 questions, full-haystack (no oracle pool).
- **Retrieval:** Zaxy hybrid Memory Checkout; OpenAI `text-embedding-3-small`.
- **Reader:** gpt-4o or gpt-5, answering over the retrieved context with a
  single general prompt (`reader-prompt.txt`) that gives only reasoning guidance
  — count distinct events, compute elapsed time from dated evidence, use the
  most-recent value for changed facts, ground preferences. **No answer hints,
  no per-question tuning, no memorized answers.**
- **Judge:** the official `evaluate_qa.py`, `gpt-4o-2024-08-06`, temperature 0.
  Metric is answer accuracy (QA), not retrieval recall.
- **Held-out split:** a deterministic 130-question subset the prompt was never
  tuned on, used as a generalization check.

## 4. Results

**Answer accuracy (QA), official judge.**

| Reader | Accuracy | Scope |
|--------|---------:|-------|
| gpt-5  | **0.902** (451/500) | full 500 |
| gpt-5  | 0.892 | 130-q held-out |
| gpt-4o | 0.777 | 130-q held-out |

The held-out gpt-5 number (0.892) sits within a point of the full-set number
(0.902): the result generalizes rather than fitting the prompt to specific
questions.

**Full-500 gpt-5 by ability:** single-session-user 1.00, single-session-assistant
0.98, knowledge-update 0.94, temporal-reasoning 0.90, multi-session 0.87,
single-session-preference 0.63.

**Retrieval is not the bottleneck.** Gold-answer-session recall is ~0.99 at the
assembled-context level (Recall@5 ≈ 0.99, Recall@1 ≈ 0.92). Almost every
remaining error is the reader answering wrong with the correct evidence in
context — the residual is reasoning (cross-session counting, temporal
arithmetic, preference-rubric matching), not recall.

## 5. Discussion

**Two evaluation hazards.** (1) We previously reported a much higher LongMemEval
figure that we **retracted**: it ran in oracle mode (~1.9 candidate sessions per
question, so recall/citation were ~1.0 by construction) and drew preference
answers from a hardcoded gold-answer table. Removing that machinery and running
full-haystack with a clean reader is what produced the numbers here — a ~2×
correction. (2) The official judge is itself an LLM; at temperature 0 it is
near- but not perfectly deterministic. Re-scoring identical gpt-5 hypotheses
gave 449/500 and 451/500 across runs (~±0.4%). We therefore headline **~0.90**
rather than a spurious third decimal, and caution that field figures quoted to
0.01% (e.g. 94.87%) exceed this metric's precision.

**Reader tier is a first-order variable.** Moving gpt-4o→gpt-5 lifted accuracy
~11 points with identical retrieval, concentrated in the reasoning-heavy
abilities (temporal +23, multi-session +11). Cross-tier comparisons are not
apples-to-apples; results must state the reader.

**Comparison.** At the gpt-4o tier, Zaxy (0.78 held-out) exceeds the one
independently published, full-haystack, same-tier peer, Zep/Graphiti (0.712;
arXiv:2501.13956). Several higher figures in the field use stronger readers,
oracle setups, or report recall rather than QA.

## 6. Reproducibility

The committed package (`README.md`, `manifest.json`, `REPRODUCE.md`,
`reader-prompt.txt`, `artifacts/*.jsonl`) records the dataset SHA-256, model
IDs, config, and the answers + official-judge labels. Re-running the official
judge on the committed hypotheses reproduces the score within judge variance.

## 7. Limitations

- The gpt-4o full-500 pass is answered but not yet judged (the run's API
  key was revoked); the gpt-4o figure here is the 130-question held-out sample.
- A single LLM judge; a human-agreement study or a second judge model would
  tighten the confidence interval.
- One retrieval/reader configuration; no ablation of the retrieval components is
  reported here.

## References

- Wu et al. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
  Memory.* ICLR 2025. arXiv:2410.10813.
- *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.*
  arXiv:2501.13956.
- *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.*
  arXiv:2504.19413.
