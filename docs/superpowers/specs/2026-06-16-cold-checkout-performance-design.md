# Cold-Checkout Performance on Large Logs — Design

Status: **proposed (design-first; agree before code)**
Date: 2026-06-16
Owner: Zaxy core

## 1. Problem

A *cold* checkout — a fresh process with empty in-process caches — is slow on a
large session log. Measured on a real ~115 MB `zaxy-default.jsonl` (~118k
events): **~18–22 s**. This is paid by every CLI invocation (`zaxy memory
checkout`, hook-driven calls), on every server restart, and after a cache
invalidation. The warm path (a long-lived server with populated in-process
caches) is already ~1.6 s; the gap is entirely cold-start rebuild cost.

The in-process incremental caches (`SessionRetrievalCache`, shipped 2.4.2–2.4.4)
extend the verbatim index and verified replay with only the appended tail — but
they live in memory and die with the process, so a cold process rebuilds
everything from the whole log.

## 2. Evidence (cProfile, real 115 MB log)

| Cost | Time | Symbol |
|---|---|---|
| **Verbatim BM25 tokenization** | **~12 s** | `verbatim.py:_tokens` (5.6 s self / 79k events) + postings (`_term_document_ids`) inside `VerbatimIndex.from_events` |
| **Verified hash-chain replay** | **~6–9 s** | `event.py:verify` / `verify_event_chain` over every event, via `_cached_full_replay` |
| Salience ledger + other full-log replays | ~4–9 s | `eventlog.read_all()` consumers in `core/fabric.py` (e.g. 970, 1337, 1517, 1546, 2090, 2106, 2330) |
| (Redundant parse) | ~4–5 s | investigated and rejected as a fix — marginal and superseded (see §7) |

Key correction from an earlier assumption: the cost is **tokenization- and
verify-dominated, not read/parse-dominated.** The fix must avoid re-tokenizing
and re-verifying the whole log on cold start, not just avoid re-reading it.

## 3. Goals / Non-goals

**Goals**
- Cold checkout on a large log approaches the warm path (target trajectory in §6).
- Integrity is never weakened: the hash chain is still verified; the log remains
  the sole source of truth; a cold result is identical to a full rebuild.
- Survives across processes (CLI and server), not just within one process.

**Non-goals**
- No change to retrieval semantics or ranking (results must stay byte-identical).
- Not compaction/forgetting/consolidation (separate concerns; see the
  consolidation decision record).
- Not the warm-server path (already fast).
- No distributed/multi-host cache; single local machine, multiple local processes.

## 4. Invariants (hard constraints)

1. **Log is authority.** The checkpoint is a pure cache. Any mismatch, corruption,
   version skew, or shrink/rewrite → discard and full rebuild. The system must be
   correct with the checkpoint deleted.
2. **Integrity preserved.** The hash chain is verified up to the live tip on every
   cold load. We never trust cached derived state without anchoring it to a
   verified event hash from the live log (see trust model §5.2).
3. **Byte-identical results.** A checkpoint-loaded-and-tail-extended index/replay
   must produce results identical to a from-scratch rebuild over the same log.
   This is the core correctness test.

## 5. Design — persistent verified derived-state checkpoint

### 5.1 Where it lives

Per session, beside the projections it accompanies:
`.eventloom/projections/<session>.retrieval-cache/` containing a small header
plus the serialized derived structures. It is a derived artifact (cache),
git-ignored, and safe to delete at any time.

### 5.2 Trust model (the integrity anchor)

The user-selected approach: a **dedicated header recording the covered tip**.

The header records: `format_version`, `covered_seq`, `covered_hash` (the hash of
the event at `covered_seq`), and the source log path. On cold load:

1. Read the live event at `covered_seq`; if its hash ≠ `covered_hash`, the log was
   rewritten/compacted/shrunk → **discard, full rebuild**.
2. Otherwise the prefix `1..covered_seq` is anchored to a verified hash. Verify
   **only the tail** (`covered_seq+1 .. live tip`) with `verify_event_chain`,
   anchored at `covered_hash` — the exact mechanism `_extend_replay` already uses.
3. Extend the loaded derived structures with the tail (the existing
   `append_chunks` / `_extend_replay` paths). Persist the new tip.

The prefix is trustworthy because the checkpoint is only ever written *after* a
full verify, so `covered_hash` is a verified anchor; re-matching it against the
live log plus a verified tail re-establishes whole-chain integrity without
re-hashing the prefix.

### 5.3 What to persist

- **Verbatim index** (the ~12 s): persist enough derived state to skip
  re-tokenization. Candidate A: persist `_chunks` + `_tokenized` and recompute the
  cheaper postings/idf/norms on load. Candidate B: persist the full derived state
  (`_tokenized`, `_term_counts`, `_term_document_ids`, `_document_frequencies`,
  `_term_idf`, `_document_length_norms`, …) to skip all recompute. Choice depends
  on the §8 prototype: persisting tokens is smaller but recomputes postings
  (~2 s); persisting everything is larger but loads fastest. **The win only
  exists if load < rebuild — this must be measured, not assumed.**
- **Verified-replay tip** (the ~6–9 s): persist the verified tip `{seq, hash}` (and
  whatever replay-derived state the checkout consumers need). Cold start verifies
  only the tail past it instead of re-hashing the whole chain.

### 5.4 Serialization format (open — decided by prototype)

Constraints: load must be materially faster than rebuild; format must be safe to
load from a file on disk; reasonable size. Options to measure: a compact binary
(e.g. msgpack), a hand-rolled binary, or JSON. **Pickle is discouraged** (code
-execution risk on a tampered file); if chosen it must be strictly gated by the
header check and treated as same-trust-domain as the log — prefer a non-executable
format. Decided in §8.

### 5.5 Integration

All changes localized to `SessionRetrievalCache` (`src/zaxy/retrieval_cache.py`),
which already owns the cold/tail/invalidate logic and is the single shared
implementation behind both the fabric and the MCP front door:

- `verbatim_index` / `_cached_full_replay` cold paths: attempt checkpoint load +
  verify + tail-extend before falling back to the existing full rebuild.
- After a cold rebuild (or a tail extension that crosses a write threshold),
  persist the checkpoint **atomically** (temp file + `fsync` + atomic rename) so a
  torn write or a concurrent reader never sees a partial file.
- `invalidate` removes the on-disk checkpoint too.

### 5.6 Concurrency

Multiple local processes may read/write. Reads are always safe (load → verify →
rebuild-on-mismatch). Writes use atomic rename; last-writer-wins. A stale
checkpoint only means a longer tail to verify — still correct, never wrong. No
lock required; document this reasoning. (A lockfile is a possible optimization,
not a correctness requirement.)

## 6. Phasing (separate PRs; measurable targets)

- **Phase 0 — this design.** Agree before code.
- **Phase 1 — persist the verbatim index** (the ~12 s). Includes the §8
  prototype to pick format/what-to-persist. Target: ~20 s → ~8–10 s. Largest
  single lever; lands first with the trust header + full-rebuild fallback.
- **Phase 2 — persist the verified-replay tip** (~6–9 s). Target: → ~4–5 s.
- **Phase 3 — share one replay across remaining `read_all` consumers**
  (salience ledger and the other full-log readers on the checkout path; audit
  `core/fabric.py` call sites). Target: → near warm (~2–3 s).

Each phase: byte-identical results test, integrity preserved, full-rebuild
fallback exercised, ruff + mypy + full suite green, and a before/after cold
measurement on a large log recorded in the PR.

## 7. Rejected / superseded alternatives

- **Read-dedup quick win** (build verbatim from the cached replay events to avoid
  a second parse): prototyped and reverted. It worked mechanically (one full
  parse removed) but the gain (~4–5 s) was noise-masked, the dominant cost is
  tokenization not parsing, it coupled the verbatim cache to the replay cache, and
  it weakened a deliberate cache regression guard. Fully superseded by Phase 1
  (a loaded index neither reads nor tokenizes).
- **Keep-a-warm-daemon for the CLI**: the MCP server is already that for MCP
  clients; a second daemon is more moving parts than a persistent cache and
  doesn't help server restarts. Persistence is more universal.

## 8. Open questions (resolve during agreement / Phase-1 prototype)

1. **Serialization format + what-to-persist** — prototype load-vs-rebuild on the
   real 115 MB log for: (tokens only + recompute postings) vs (full derived
   state), in a compact binary vs JSON. Pick the fastest safe load. *This is the
   make-or-break measurement for Phase 1.*
2. **Salience ledger state** — persist it too (Phase 3), or just route consumers
   through the shared verified replay? (Leaning: share the replay first; persist
   only if still hot.)
3. **Write cadence** — persist on every cold build only, or also after the tail
   grows past a threshold? (Leaning: cold build + threshold, to keep CLI writes
   cheap.)
4. **Checkpoint location & gitignore** — confirm `.eventloom/projections/` and add
   the cache dir to ignore rules.

## 9. Done-when

Cold checkout on a large log is materially and measurably faster (trajectory in
§6), integrity and byte-identical-results invariants hold with tests proving
both the checkpoint path and the full-rebuild fallback, the checkpoint is safe to
delete/corrupt (always falls back), ruff + mypy + full suite green, and each
phase records a before/after cold measurement.

## References

- `src/zaxy/retrieval_cache.py` — `SessionRetrievalCache`, `verbatim_index`,
  `_cached_full_replay`, `_extend_replay`, `_eventlog_file_signature`
- `src/zaxy/verbatim.py` — `VerbatimIndex` derived-state fields, `from_events`,
  `append_chunks`, `_tokens`
- `src/zaxy/event.py` — `verify_event_chain`, `read_from_offset`, `read_all`
- `src/zaxy/core/fabric.py` — checkout assembly + the `read_all` consumers
- `goal-checkout-incremental-cache` (shipped v2.4.4) — the in-process incremental
  cache this persists across processes
- `2026-06-16-consolidation-continuous-vs-batch-decision.md` — names this work as
  the independent engineering carve-out
