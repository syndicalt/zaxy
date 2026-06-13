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
