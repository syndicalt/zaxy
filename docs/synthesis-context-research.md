# Synthesis Context Research

This note defines the next architecture target for Zaxy retrieval: a synthesis working-memory layer that sits between candidate retrieval and model-facing Memory Checkout.

The current benchmark pattern is clear: Zaxy often retrieves the relevant memory neighborhood, but still misses composed answers. In the latest 100-question LongMemEval hash run, Zaxy reached `Recall@5=0.870` and `Recall@10=0.900`, while the largest remaining miss class was still `synthesis_miss`. That means the next improvements should not add more broad retrieval surfaces first. They should make retrieved evidence easier to compose, verify, and project.

## Research Takeaways

Long context is not reliable context. Liu et al. show that language model performance can drop when relevant information is placed in the middle of a long prompt, even for long-context models. Zaxy should therefore avoid treating “more retrieved text” as the answer. It should produce a compact, ordered context object where the most decision-critical state is pinned at the top and cited evidence follows.

GraphRAG’s local search is the closest production pattern: combine graph-structured data with raw text units at query time, then rank and filter entity descriptions, text units, relationships, and reports into the prompt. Zaxy already has the raw ingredients, but it needs a stronger synthesis control plane over those ingredients.

SELF-RAG’s useful lesson is not its training recipe; it is the control loop. Retrieval should be adaptive, and generated answers should carry a reflection/critique signal over relevance and support. Zaxy can implement this deterministically as retrieval plan status, evidence sufficiency, contradiction checks, and answerability warnings.

ColBERT’s late interaction architecture is important because single-vector similarity loses token-level alignment. For Zaxy, this argues for a future optional late-interaction/rerank lane over compact source snippets, especially for source recall and “which item” questions. The local default can remain BM25/hash, but the interface should be able to accept multi-vector or token-interaction rerankers.

HyDE’s lesson is query-shape translation: generate or derive the answer-shaped query, then retrieve documents that would support that answer. Zaxy should do this without an LLM in the default path where possible: typed query plans should produce source-lane expansions such as candidate entities, units, expected answer type, operation, and constraints.

MemGPT’s strongest idea is virtual context management: main memory should stay small, and archival memory should be paged in deliberately. Zaxy’s “git for memory” version should make this auditable: the model gets a stable checkout with answer candidates and cited pages, not a raw dump.

A-MEM argues that memory organization should evolve dynamically through links and contextual metadata. For Zaxy, this maps to Eventloom-backed reinforcement, feedback, supersession, and learned source-selection metadata rather than untracked mutable summaries.

Recent Graph-RAG reasoning-bottleneck work is directly aligned with our benchmark failures: strong retrieval can still fail if the model cannot reason over the retrieved context. Structured prompting and graph-walk compression improved multi-hop QA while reducing context. Zaxy should treat graph traversal output as executable evidence paths, not just another text blob.

CompAct reinforces the same direction: multi-hop QA benefits from active compression that preserves key information. Zaxy’s compression must remain cited and replayable, but it should produce task-shaped evidence capsules instead of raw chunks.

## Product Principle

Zaxy should maintain context during synthesis as a typed, cited working set:

1. Intent: what kind of answer is needed.
2. Evidence requirements: how many independent sources, which entities, which time window, which operation.
3. Candidate ledger: normalized facts, values, units, entities, source ids, confidence, and exclusions.
4. Composition plan: count, sum, difference, average, argmax, absence, temporal interval, list, or graph path.
5. Result candidates: concise answer surfaces plus machine fields.
6. Support packet: minimal cited snippets and graph/event paths.
7. Verification: missing evidence, contradictions, stale facts, dedupe decisions, and confidence.

The model should see the answer candidate first, then the proof. The proof should be compact enough to prevent lost-in-the-middle behavior and complete enough to audit.

## Proposed Architecture

### 1. Query Plan

Create a first-class `SynthesisPlan` alongside the existing retrieval plan.

Fields:

- `answer_type`: `direct_fact`, `count`, `list`, `sum`, `difference`, `average`, `argmax`, `absence`, `temporal_interval`, `graph_path`.
- `subject_terms`: query-specific entities and nouns.
- `constraints`: time range, session range, entity filters, source filters.
- `required_evidence`: minimum source groups and required value kinds.
- `operation`: typed operator with expected units.
- `projection_budget`: target max answer bytes and evidence bytes.

This should be deterministic and testable. LLM planning can be added later as an optional adapter, not a dependency.

### 2. Evidence Ledger

Replace ad hoc candidate line construction with a structured ledger.

Rows:

- `fact_id`
- `source_group`
- `citation`
- `kind`
- `entity`
- `value`
- `unit`
- `time`
- `label`
- `raw_span`
- `normalized_identity`
- `include_reason`
- `exclude_reason`
- `confidence`

The ledger is the synthesis working memory. It should be serializable and optionally written back as an Eventloom projection artifact for audit and feedback.

### 3. Evidence Operations

Each operation should be a small pure function over ledger rows.

Initial operation set:

- `count_distinct`
- `list_items`
- `sum_values`
- `difference_between`
- `average_values`
- `max_by_value`
- `date_interval`
- `absence_contrast`
- `path_explain`

Each operation returns:

- answer fields
- natural answer text
- support ids
- excluded rows
- warnings

This makes numeric synthesis debuggable and stops one-off text patterns from spreading through retrieval code.

### 4. Context Packet

Memory Checkout should return a stable context packet:

```text
memory_checkout=true
answerability=answer_from_memory
synthesis_plan=sum_values
answer_candidate=$185
answer_text=I spent $185 on bike-related expenses.
confidence=0.86
support_source_ids=answer-1,answer-2,answer-3
excluded_source_ids=answer-4
exclusion_reason=duplicate_item:value=$40,label=bike lights

evidence:
- source_id=answer-1 value=$120 label=Bell Zephyr bike helmet citation=...
- source_id=answer-2 value=$25 label=bike chain citation=...
- source_id=answer-3 value=$40 label=bike lights citation=...
```

The model should not need to rediscover the arithmetic from raw snippets unless it wants to audit the answer.

### 5. Two-Phase Retrieval

World-class synthesis needs two retrieval phases:

1. Candidate recall: overfetch broadly from graph, BM25, vector, and verbatim lanes.
2. Evidence assembly: rerank and filter by the synthesis plan, not by generic query similarity.

This addresses current failures where the correct evidence exists somewhere in the top 10 but not in the synthesis bundle. The synthesis bundle should be built from a larger candidate pool and then projected into the top checkout packet.

### 6. Position-Aware Projection

Because long-context models are sensitive to where evidence appears, the checkout packet should use a stable order:

1. Answer candidate and answerability.
2. Required actions or warnings.
3. Structured evidence table.
4. Minimal cited snippets.
5. Raw context fallback.

Do not bury answer candidates in the middle of a large result list.

### 7. Feedback Loop

When the model uses or rejects an answer candidate, write feedback:

- `memory.synthesis.used`
- `memory.synthesis.rejected`
- `memory.synthesis.corrected`
- `memory.evidence.excluded`
- `memory.evidence.reinforced`

This gives Zaxy a learning surface without making the source of truth mutable.

## Implementation Sprints

### Sprint 1: Structured Synthesis Ledger

Build `SynthesisPlan`, `EvidenceLedger`, and `SynthesisResult` as pure models. Port existing count/currency/duration candidate code onto the ledger without changing external behavior.

Acceptance:

- Existing benchmark tests pass.
- Ledger rows expose include/exclude reasons.
- Memory Checkout can emit the ledger in diagnostics.

### Sprint 2: Evidence Assembly Overfetch

Build synthesis bundles from an internal overfetch pool instead of the final top-k display set.

Acceptance:

- For known arithmetic misses, the ledger includes all expected source groups when they appear in top 10 retrieval.
- The final packet remains bounded by budget.

### Sprint 3: Operation-Specific Projection

Implement operation classes for sum, difference, average, list, and temporal interval.

Acceptance:

- `$185`, `$270`, `140 hours`, `59.6`, and `$5,850` cases are explainable from the ledger.
- No benchmark-specific expected-answer strings are hardcoded.

### Sprint 4: Context Packet Contract

Move synthesis output into Memory Checkout as a first-class answer candidate section with support/exclusion diagnostics.

Acceptance:

- The model sees answer candidate first.
- Cited evidence remains available.
- Checkout token budget does not regress materially.

### Sprint 5: Adaptive Rerank Interfaces

Add optional late-interaction and hosted reranker interfaces behind the existing reranker abstraction. Keep local default deterministic.

Acceptance:

- Local path remains dependency-light.
- Hosted/late-interaction path can be benchmarked independently.
- Retrieval reports distinguish candidate recall from evidence assembly quality.

## References

- Liu et al., “Lost in the Middle: How Language Models Use Long Contexts”, 2023: https://arxiv.org/abs/2307.03172
- Microsoft Research, “From Local to Global: A Graph RAG Approach to Query-Focused Summarization”, 2024: https://www.microsoft.com/en-us/research/project/graphrag/publications/
- GraphRAG local search docs: https://microsoft.github.io/graphrag/query/local_search/
- Asai et al., “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection”, 2023: https://openreview.net/pdf?id=jbNjgmE0OP
- Khattab and Zaharia, “ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT”, 2020: https://arxiv.org/abs/2004.12832
- Gao et al., “Precise Zero-Shot Dense Retrieval without Relevance Labels”, 2022: https://arxiv.org/abs/2212.10496
- Packer et al., “MemGPT: Towards LLMs as Operating Systems”, 2023: https://arxiv.org/abs/2310.08560
- Xu et al., “A-MEM: Agentic Memory for LLM Agents”, 2025: https://arxiv.org/abs/2502.12110
- Yoon et al., “CompAct: Compressing Retrieved Documents Actively for Question Answering”, 2024: https://huggingface.co/papers/2407.09014
- Zarrinkia et al., “The Reasoning Bottleneck in Graph-RAG”, 2026: https://huggingface.co/papers/2603.14045
