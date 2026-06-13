# Declarative compact-notation experiment — results

Goal anchor: zaxy `goal.created` seq 77835 (session `zaxy-default`).
Branch: `exp/declarative-notation`. Corpus: `.eventloom/zaxy-default.jsonl`, 77,845 events.
Reproduce: `python3 run_experiments.py --log ../../.eventloom/zaxy-default.jsonl` (codec tests: `pytest test_notation.py`).

## Hypothesis
Declarative event STATE can be encoded losslessly into a dense symbolic notation
iff (a) closed vocabulary, (b) bijective token↔field, (c) read-not-generate — and
such a notation may be (i) more token-dense and/or (ii) more directive than prose.

## Scope decision (a finding in itself)
The notation encodes the **declarative content** of an event — `type`, `actor`,
`thread/domain`, `payload` — and deliberately **not** the integrity envelope
(`seq`, `timestamp`, `prev_hash`, `hash`, `id`, `parent_event_id`, `caused_by`,
`envelope_version`, `security`). Those are cryptographic metadata, not facts to
re-author. Full-event byte-identity is therefore out of scope by design;
round-trip fidelity is measured over declarative content.

Grammar: `[domain]«etype»@actor{entity}(k:Tv+k:Tv) >>rel !Ø(inv,inv)`
Value type codes: `s`tring `i`nt `f`loat `b`ool `z`null `j`son. All structural
metacharacters are backslash-escaped in leaf text.

## Codec correctness
`pytest test_notation.py` → **9 passed**, including a 2000-case fuzzer over random
nested/typed/special-char facts. The fuzzer caught one real bug (`['']` vs `[]`
collision in the invalidation section), now fixed. Codec is bijective over its
schema.

## Rung 2 — round-trip fidelity (THE GATE) ✅ PASS
`event → fact → glyph → fact → content`, compared to the event's declarative
content, over an 8,000-event sample:

| metric | value |
|---|---|
| sampled | 8,000 |
| representable (payload is an object) | 8,000 (**100%**) |
| round-trip pass | 8,000 (**100.000%**) |
| failures | none |
| per-type pass rate | **1.0 for every type present** (incl. high-volume `tool.call.completed`, `command.completed`, `transcript.turn`, `file.edit.applied`) |

**Verdict: the notation is lossless over the declarative content of the entire
corpus.** The closed-schema + bijection conditions hold empirically, not just in
theory.

## Rung 1 — token density (cl100k_base, 50 balanced declarative states) ⚠️ NULL RESULT
Ratio = baseline_tokens / glyph_tokens; **>1 means glyph is denser**.

| baseline | median | mean | min | max |
|---|---|---|---|---|
| faithful prose | **0.988×** | 1.026× | 0.845 | 1.179 |
| compact JSON | **1.00×** | 1.05× | 0.843 | 1.321 |
| pretty JSON | 1.296× | 1.421× | 1.019 | 2.179 |

Totals over the 50 states: glyph 8,060 · prose 7,907 · compact-JSON 7,943 · pretty-JSON 9,845 tokens.

**The token-savings thesis is rejected.** Against the only fair baselines — faithful
prose and compact JSON — the glyph notation is break-even to *slightly worse*. It
only "wins" against pretty-printed JSON, which is a strawman (nobody injects
indented JSON into context). Causes: (1) type codes + delimiter escaping + a JSON
fallback for nested values consume the density the glyphs save; (2) the rare glyph
chars (`« » Ø`) split into multiple BPE tokens. This is the expected information-
theoretic floor: a *lossless* re-encoding of the same closed information cannot
materially beat a compact lossless baseline. The pre-registered 1.3–1.5× guess was
optimistic.

## Go / No-Go at the gate
- **Fidelity gate (round-trip 100%): PASS.**
- **Token-density rationale: REJECTED** (~1.0× vs sane baselines).
- **Net:** the *only* surviving reason to adopt this notation is **directive force**
  (rungs 3–4) — it is **not** justified by token economy. The decision to fund
  rungs 3–4 should rest entirely on whether a distinctive symbolic surface is
  *obeyed/used more reliably* than prose, especially post-compaction.

**Recommendation: CONDITIONAL — do NOT adopt for compression.** Pause for human
review (per goal). Proceed to rung 3 (comprehension, both models) → rung 4
(adherence A/B × context length) only if directive force is the bet worth testing.
If pursued, the baseline to beat is **compact JSON / terse prose**, not pretty JSON,
and any adherence win must clear the cost of zero density benefit + glyph
tokenization overhead.

---

# Rungs 3–4 — directive force (LIVE, Claude `claude-opus-4-8` + OpenAI `gpt-4.1`)

3-arm design (prose / json / glyph) over identical lossless state. Models: Opus +
GPT-4.1. Reproduce: `python3 harness.py --rung {3,4} --provider {anthropic,openai}
--model <id> [--glyph-legend]`.

## Rung 3 — comprehension gate (answerable-only, none-cases removed)
The `invalid` question on records with nothing invalidated produced a uniform
"hallucinate an answer" artifact across *all* forms; reporting on answerable
items isolates real comprehension.

| form | Claude | GPT-4.1 | gate (≥0.95) |
|---|---|---|---|
| prose | 1.000 | 1.000 | **PASS** |
| json | 1.000 | 1.000 | **PASS** |
| glyph (no legend) | 0.736 | 0.569 | FAIL |
| glyph (+1-time legend) | 0.917 | 0.722 | **FAIL** |

- The `>>` recommended-action **operator reads fine** (both models ~1.0). The
  failure is the **type-coded scalar values** (`stale:bT`, `n:i42`) — not self-
  describing without a legend, and even *with* a legend neither model clears 0.95.
- **Cross-model divergence** (Claude 0.917 vs GPT 0.722 with legend) is the
  brittleness flag we pre-registered — disqualifying for a method meant to serve
  both clients.
- **Glyph is EXCLUDED at the gate.** prose and json proceed.

## Rung 4 — adherence A/B × context length
Tasks require *acting on* an injected fact (use the right session / take the
recommended action / use a fact / avoid an invalidated value). `none` = no
injection (baseline). `buried` = state placed before ~6k tokens of real
transcript filler (long-session proxy).

| form | Opus fresh | Opus buried | GPT-4.1 fresh | GPT-4.1 buried |
|---|---|---|---|---|
| none (control) | 0.250 | — | 0.250 | — |
| prose | 1.000 | **1.000** | 1.000 | **1.000** |
| json | 1.000 | **1.000** | 1.000 | **1.000** |
| glyph | 0.750 | 0.750 | 0.750 | 0.583 |

- **Injection works:** control 0.25 → injected 1.0. Models reliably act on injected
  declarative state.
- **Structure/salience hypothesis: REJECTED.** prose and json both hold at **1.000
  even buried**, on both models. The prose→json buried delta is **0.000** — there is
  no degradation for structure to rescue. Terse prose is sufficient.
- **Glyph** is lowest and the *only* form that degrades when buried (GPT 0.75→0.58),
  consistent with its comprehension failure.

## FINAL VERDICT
- **Reject the glyph notation outright.** No density benefit (rung 1), fails
  comprehension on both models even with a legend (rung 3), lowest + burial-fragile
  adherence (rung 4). The lossless-bijection property (rung 2) is real but buys
  nothing the model can use.
- **Reject "structured beats prose" for injection.** At realistic injection sizes,
  compact-JSON gives zero adherence advantage over terse prose.
- **Adopt the simple fix instead:** the original persistence gap is a *missing
  injection*, not a *format* problem. Inject terse-prose declarative memory state
  every turn (the `UserPromptSubmit`/`additionalContext` lever) — frontier models
  act on it reliably, even buried.

## Limitations (honest)
- **Ceiling effect:** prose & json both saturate at 1.0, so this task difficulty
  cannot detect a prose-vs-json difference if a small one exists. Conclusion holds
  *at realistic injection size*; a harder, multi-fact, adversarial task is untested.
- **Burial depth ~6k tokens** (reduced from 15k for OpenAI TPM limits). Full
  compaction-scale burial (50k+) is untested; prose might degrade there.
- Two model families only (Opus, GPT-4.1); one run each (no repeated-sampling
  variance). `gpt-4.1` stands in for the "Codex" arm (no codex-specific id exposed).
