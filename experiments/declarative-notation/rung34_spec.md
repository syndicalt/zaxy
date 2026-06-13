# Rungs 3–4 spec — directive force of lossless declarative notation

Prereq met: rung 2 proved the notation is lossless over declarative content
(8000/8000). Rung 1 killed the density rationale (~1.0× vs compact JSON/prose).
**The only hypothesis still worth testing is directive force**, so rungs 3–4
test exactly that — and test the *mechanism*, not the specific glyphs.

## Core design principle: 3 arms, same lossless state
Every item is one declarative memory state rendered three ways:
- **prose** — faithful terse English
- **json** — compact JSON of the same fields
- **glyph** — the `[domain]«etype»@actor{entity}(k:Tv) >>rel !Ø(inv)` notation

All three are lossless and carry identical information (verified: the same
`MemoryFact` produces all three). This isolates *surface form* as the only
variable, so we can tell apart:
- **structure/salience win** — json ≈ glyph, both > prose → ship the form with
  better comprehension + no exotic-token tax (likely json), NOT the glyphs.
- **symbol win** — glyph > both json and prose → the symbols themselves earn it.
- **no win** — glyph ≈ prose → reject; directive force was the last justification.

A 2-arm (prose vs glyph) design cannot make this distinction and would be a trap.

## Rung 3 — comprehension probe (cheap gate, BOTH models)
**Question:** can the model correctly *read* fields and operators from a single
encoded line? Gates rung 4 — no point testing adherence to a form the model
misreads.

- Bank: ~24 declarative states (grounded in real zaxy facts; see fixtures).
- Per state, 4 question types, each with a programmatically-derivable answer:
  - `scalar` — value of a named field (e.g. `stale`)
  - `domain` — which session/domain
  - `action` — the `>>` recommended-action target  ← **operator test**
  - `invalid` — what `!Ø(...)` marks invalidated     ← **operator test**
- Prompt = ONLY the encoded line (one form) + the question. No other context.
- Score: exact/normalized match to ground truth. Report accuracy per **form ×
  question-type × model**.
- Models: Claude (`claude-opus-4-8` or current) AND Codex/GPT. **Cross-model
  divergence on the `action`/`invalid` operators is the brittleness flag** — the
  operators are the fragile, RLHF-unfamiliar part.

**Gate to rung 4:** a form proceeds only if operator-comprehension ≥ 0.95 on
*both* models. Expectation: prose/json clear this easily; glyph operators are the
risk. If glyph fails comprehension, stop — adherence is moot.

## Rung 4 — adherence A/B (real cost) × context length
**Question:** does the model *act on* injected state more reliably in one form,
especially when the state is far from the query (long-session proxy)?

Task types (each requires using the injected fact to answer correctly):
- `use_session` — state asserts `session=X`; trap value `!Ø(Y)` present. Task asks
  which session to write to. Correct = X, fail = Y or other.
- `recommended_action` — state is `stale` with `>>memory_checkout`. Task: what to
  do before answering. Correct = states it will checkout/refresh memory first.
- `use_fact` — state carries a project fact (e.g. default backend). Task asks a
  question answerable only from it. Correct = uses the fact.
- `avoid_invalidated` — state carries a fact AND `!Ø(old_value)`. Correct answer
  avoids the invalidated value.

Factors (full cross):
- **form**: prose / json / glyph  (+ a `none` control: no injection → baseline
  forgetting rate / prior-driven guessing)
- **context**: `fresh` (state immediately before task) vs `buried` (state, then
  ~6–10k tokens of real `transcript.turn` filler from the log, then task). Buried
  = the long-session/post-compaction proxy and the condition we actually care about.
- **model**: Claude, Codex.

- Items per cell: **≥40** (configurable). Cells: 4 forms × 2 contexts × 2 models
  = 16, plus the `none` control. Score = deterministic adherence checker per task.
- Stats: per-cell adherence rate + Wilson 95% interval; report the **buried-context
  prose→glyph and json→glyph deltas** as the headline. Pre-register: a form "wins"
  only if its buried-context adherence beats prose by ≥10 pts with non-overlapping
  Wilson intervals on **both** models.

## Success criteria / decision
- **ADOPT glyph**: glyph passes rung 3 on both models AND beats *both* prose and
  json in buried-context adherence (per the pre-registered bar).
- **ADOPT json/structured (not glyph)**: json ≈ glyph and both beat prose → take
  the salience win without the glyph comprehension/tokenization risk.
- **REJECT**: glyph ≈ prose in adherence, or glyph fails rung-3 comprehension.

The deployment target either way is the `UserPromptSubmit`/`SessionStart`
`additionalContext` injection (per the earlier persistence-gap finding) — small
declarative state, where density is irrelevant (proven) and salience might matter.

## Budget estimate (model calls)
- Rung 3: 24 states × 4 Qs × 3 forms × 2 models ≈ **576 calls**, short outputs.
- Rung 4: 40 items × 4 forms × 2 contexts × 2 models ≈ **1,280 calls**; `buried`
  arm has long inputs (~6–10k tok) → the cost driver. `none` control adds ~80.
- Total ≈ **~1.9k calls**, majority cheap; rung-4 buried inputs dominate token spend.
- Mitigation: run rung 3 first; only fund rung 4 for forms that pass the gate.

## Harness
- `fixtures.py` — builds the grounded item bank + 3 renderings + filler. No model.
- `harness.py` — pluggable `ModelClient` (anthropic / openai-compatible / manual
  file-based), `--dry-run` builds + saves every prompt and **self-tests the scorers
  against ground truth** (passes ground-truth answer, fails a known-wrong answer)
  with zero model calls. Run dry-run first to inspect exactly what would be sent.
