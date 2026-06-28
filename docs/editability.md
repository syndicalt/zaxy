# Transparency & Controlled Editability

Zaxy's memory is an append-only, hash-chained Eventloom log. A written event is
sealed: its payload is folded into the event hash, and that hash links to the
previous one. Mutating or deleting a stored event would break the chain and fail
`EventLog.verify()`. So Zaxy never edits the past. Instead, corrections and
rollbacks are **new, cited, non-authoritative events** whose effect is honored
during replay/projection. This is the same additive discipline used by the
fleet plane's reversible `fleet.promotion.rolled_back` and the outcome-learning
loop's gated `memory.rule.generated` — see the [README](../README.md) and the
[consolidation guide](consolidation.md) for the surrounding lifecycle.

This is part I5a of the Zaxy 3 roadmap: a human-readable view → edit →
re-ingest round-trip, plus explicit rollback of a prior evolution. Both are
additive, reversible, and routed through the I4 evolution gate, so nothing is
ever destroyed and the hash chain stays provably intact.

## Edit → re-ingest round-trip

A memory can be exported to a small, human-readable editable block and parsed
back after a human edits it:

```text
--- zaxy:editable v1 ---
seq: 12
hash: <64-hex event hash>
session_id: default
---
The corrected, human-edited memory content goes here.
```

`render_editable(memory)` produces the block from a memory (its `seq`, `hash`,
and content); `parse_editable(text)` validates the edited block and returns the
correction fields (`target` `{seq, hash}` plus the new `content`). Re-ingesting
the edit appends a `memory.corrected` event that **cites** the original event,
carries the edited content, the reason, and a deterministic `correction_id`. The
original event is retained unchanged; retrieval surfaces the correction alongside
it. The correction is non-authoritative — it is evidence of a human edit, not a
silent overwrite.

* Python: `await fabric.edit_memory(target_seq=..., target_hash=..., new_content=..., reason=...)`
* MCP tool: `memory_edit`
* CLI: `zaxy memory edit --target-seq N --target-hash H --new-content "..." --reason "..."`

## Rollback of an evolution

An autonomous evolution — a consolidation acceptance, a generated preventive
rule, a gate decision, a fleet review — can be explicitly reversed. A
`memory.rolled_back` event cites the evolution event being reversed and carries
a `reverts` descriptor and a deterministic `rollback_id`. On replay/projection
the cited evolution is undone: a rolled-back consolidation acceptance reverts the
candidate to its prior (pre-acceptance) review status, so `consolidation_status`
no longer reports it as accepted. The rollback is additive and itself reversible;
the accepted review event is never mutated.

* Python: `await fabric.rollback_memory(target_seq=..., target_hash=..., reason=...)`
* MCP tool: `memory_rollback`
* CLI: `zaxy memory rollback --target-seq N --target-hash H --reason "..."`

## Why this is safe

Because corrections and rollbacks are new sealed events, every one of these
operations leaves `EventLog.verify().ok == True`: the hash chain is never
rewritten. The effect of an edit or a rollback is purely a function applied
during replay, exactly mirroring how reinforcement, consolidation review, and
fleet rollback already work. Both operations pass through the governed evolution
gate (op `update`), which records an auditable `evolution.gate.evaluated` event,
so the decision to apply a correction or reversal is itself part of the
transparent, replayable history. Verified forgetting / cryptographic erasure is a
separate concern (I5b) and is not what these events do — nothing here removes
content from the log.

## Verified forgetting (cryptographic erasure)

Reversible attenuation (salience decay) is the default way memory fades — a faded
memory leaves default ranking but stays one query away. When hard deletion is
genuinely required (e.g. a GDPR erasure request), Zaxy supports **verified
forgetting via cryptographic erasure** (I5b), which removes the plaintext
*without* breaking the hash chain.

Because a payload is sealed into the event hash, you cannot scrub a stored
plaintext payload without failing `verify()`. So forgettable memories are
**encrypted at append time**: with `FORGETTING_ENABLED=true`, an
`append(..., forgettable=True)` seals the payload as **ciphertext** (a
`__zaxy_cipher` cell). The data-encryption key (DEK) is wrapped under a
key-encryption key (KEK) and stored **only in an out-of-log erasure vault** —
never in the append-only log.

Forgetting then **destroys the wrapped DEK** and appends a cited,
non-authoritative `memory.forgotten` tombstone (audited, routed through the I4
`forget` gate). The on-disk ciphertext and its hash are untouched, so
`EventLog.verify()` stays green; the plaintext is permanently unrecoverable, and
readers see a `[FORGOTTEN]` sentinel. `verify()` never decrypts — it validates
the raw ciphertext chain.

* Python: `await fabric.verified_forget(target_seq=..., target_hash=..., reason=...)`
* MCP tool: `memory_forget`
* CLI: `zaxy memory forget --target-seq N --target-hash H --reason "..."`

The KEK defaults to `<eventloom_dir>/__erasure_kek__.key` (a dev key
auto-generated `0600` on first use); in production point `FORGETTING_KEK_PATH` at
a KMS- or secret-managed key. **Caveat:** the erasure crypto reuses Zaxy's
portable-bundle envelope, which is **experimental and unaudited** — do not rely
on it for high-value-secret or compliance guarantees without an independent
cryptographic review.
