# Zaxy 2.3 External-Validation Readiness Audit

Date: 2026-06-11. Auditor environment: Linux 6.17.0-29-generic, Python 3.13.13, bash.
Fresh clone: https://github.com/syndicalt/zaxy @ `3fbe6d6` (2.2.0, ahead of the local
working copy `034b95b`). All work in `/tmp/zaxy-23-research/`; production repo untouched.

Policy context: `docs/external-validation.md` is a v1.0-era first-run/Coordinate
policy with a machine-checkable report contract (`scripts/check-external-validation.py`).
It covers onboarding paths only — it has **no path for reproducing benchmark lanes or
the Harvey LAB rerun**. 2.3 needs a benchmark-reproduction analog of this policy.

---

## 1. Fresh-clone reproduction test (executed, not reviewed)

### Step timings and outcomes

| Step | Command | Time | Result |
|---|---|---|---|
| Clone | `git clone https://github.com/syndicalt/zaxy` | 5.1 s | OK |
| venv | `python3 -m venv .venv` | 2.3 s | OK |
| Install | `pip install -e ".[dev]"` (per CONTRIBUTING.md) | 20.1 s | OK — zaxy-memory 2.2.0 + 88 deps, no errors |
| Test smoke | `pytest tests/test_config.py tests/test_docs_site.py` | 8.1 s | **EXIT 1** (97/97 tests pass; coverage gate fails) |
| Test smoke (fixed) | same + `--no-cov` | 0.6–1.3 s | OK, 97 passed |
| AX lane | `zaxy agent-experience-lanes --lanes tool-adoption` | 0.24 s | **EXIT 2** (`zaxy_benchmarks` not importable) |
| AX lane (fixed) | `PYTHONPATH=. zaxy agent-experience-lanes --lanes tool-adoption` | 1.7 s | OK, report written |
| Vector lane 1k | `PYTHONPATH=. zaxy graph-scale-lanes --lanes vector-scale --scale-sizes 1000` | 2.4 s | OK |
| Vector lane 1k+10k (rerun) | same, `--scale-sizes 1000,10000` | ~14 s | OK; size-1000 deterministic block byte-identical to first run |

### Friction points (exact errors)

**F1 — documented test subset "fails" while passing.** `pyproject.toml` ships
`addopts = "-m 'not integration' --cov=zaxy --cov-report=term-missing --cov-fail-under=90"`,
so any narrow subset exits 1:

```
FAIL Required test coverage of 90% not reached. Total coverage: 0.79%
97 passed in 6.41s
```

`docs/testing.md` mentions `--no-cov` only for integration runs; CONTRIBUTING.md
("run the narrow tests that cover your change first") never warns about this. A
third-party validator's first signal from a fully passing suite is a red FAIL.

**F2 — every documented benchmark command fails as documented.** `docs/benchmarks.md`
says to run `zaxy agent-experience-lanes --lanes all` / `zaxy graph-scale-lanes --lanes all`.
From a fresh clone + editable install, the console script fails:

```
Invalid value: Benchmark and external-evaluation commands require the
optional source-checkout eval package `zaxy_benchmarks`. Run this command
from a Zaxy source checkout or install the eval tooling package.
```

The error's own advice is wrong: I *was* at the source-checkout root. Cause:
the hatch wheel only packages `src/zaxy` (`[tool.hatch.build.targets.wheel] packages = ["src/zaxy"]`),
`zaxy_benchmarks/` lives at the repo root, and a console script never puts cwd on
`sys.path`. Undocumented workarounds (both verified): `PYTHONPATH=. zaxy ...` or
`python -m zaxy ...` from the checkout root. Neither appears in any doc. This guard
also gates **all** `harvey-lab-*` subcommands, so it blocks the Harvey rerun too.

**F3 — no environment/provenance block in lane reports.** The emitted
`graph-scale-lanes.json` top-level keys are `lanes`, `validation`, `version` only:
no commit SHA, zaxy version, Python version, OS, CPU, or BLAS info. A third party
cannot prove *what* they reproduced or attribute latency differences.

Things that worked well: install is genuinely one command (20 s, no compile, no
Docker/Neo4j/credentials needed); `--output-dir` exists on both lane commands;
the lane CLIs have good `--help` with honest caveats (e.g. 100k opt-in warning).

### Artifact comparison vs `docs/research/artifacts/ann-2026-06/`

Fresh run: dim 64, hash distribution, `embedding_version_tag: hash@99329f18-dim64`,
sizes 1000 and 10000 — matched against `baseline-dim64.json`, `after-dim64-r1/r2/r3.json`,
`ann2-d64-10k-postcopy.json`.

Fields that SHOULD match across machines (per docs/benchmarks.md: "Corpus hashes,
exact/quantized recall (both metrics), bytes, and byte budgets are two-run reproducible")
and DID match:

- `corpus_sha256` @1000: `befb75bd…f3b3d` — **exact match** in all archived d64 artifacts.
- `corpus_sha256` @10000: `a4673fe6…f760` — **exact match**.
- `resident_index_bytes`: exact 512,000 / 5,120,000; quantized 72,000 / 720,000 — **match**.
- `bytes_vs_exact_ratio` 0.1406, `group_type`, `engaged`, `vector_count` — **match**.
- Exact-mode recall 1.0 — **match**.
- Same-machine two-run determinism of the whole deterministic block — **verified identical**.

Fields that did NOT match, and why:

- **Schema drift, all 20 archived artifacts.** None contain the shipped 2.2.0
  schema fields (`recall_at_k_strict` / `recall_at_k_tie_aware`, `byte_budget`,
  `distribution`); they carry the older single `recall_at_k`. Every archived JSON
  is a mid-development snapshot taken before the final 2.2 schema landed. A naive
  third-party diff fails on every artifact.
- **Quantized recall value drift.** Archived d64: 0.9938 @1k / 0.9906 @10k.
  Fresh 2.2.0 run: 1.0 / 1.0 (strict and tie-aware). The shipped quantized path
  is not the artifact-era algorithm, so the one supposedly deterministic *quality*
  number in the archive is unmatchable with released code.
- Latency/ANN-build numbers differ, as documented and expected (environment- and
  HNSW-nondeterminism-dependent); e.g. ANN first-query 9.1 s here vs 53 s in
  `BASELINE.md` at 10k/d64 — also reflecting 2.2 code improvements.

**Conclusion:** the determinism *machinery* works (hashes/bytes/recall reproduce
bit-exact cross-machine), but the *archive* is not reproducible as published:
no archived artifact was generated by the released code, and no artifact records
the commit it came from. `BASELINE.md` discloses only "Host: local dev machine."

---

## 2. Harvey LAB adapter-kit audit

What exists:

- Generator: `zaxy harvey-lab-adapter-kit --output-dir …` (in
  `zaxy_benchmarks/harvey_lab_benchmark.py`, `export_harvey_adapter_kit`). Emits
  exactly two files (verified): `README.md` (1.5 KB, 5-step runbook) and
  `raw_rg_memory.py` (1 KB shim exposing `scan_corpus`/`search`/`read`).
- Orchestration: `reports/benchmarks/harvey-lab-memory-ablation/run-harvey-lab-zaxy.sh`
  (22 KB, all 10 pinned tasks end-to-end) plus `harvey-lab-external-run.md/.json`
  manifest, and `harvey-lab-doctor`/`-preflight`/`-ready`/`-status`/`-validate`/
  `-gate`/`-publish` CLI gates. Referenced zaxy-side artifacts all exist on disk.
- Pins: Harvey commit `29748828133dff83ad2263af353fb035504f8f77` (in docs/benchmarks.md,
  NOT in the kit README), generator `gpt-5.5`, judge `gpt-5.4-mini` (env-overridable).

Readiness verdict: **NOT third-party-runnable as shipped.** Concrete blockers:

1. **Latent import breakage (high confidence).** The shim does
   `from zaxy_benchmarks.harvey_lab_benchmark import …`, but the run script sets
   `ZAXY_PYTHONPATH="$ZAXY_WORKTREE/src"` — and `zaxy_benchmarks/` is at the repo
   root, not under `src/`. Verified: `PYTHONPATH=<clone>/src python -c "import zaxy_benchmarks"`
   → `ModuleNotFoundError: No module named 'zaxy_benchmarks'`. The harness step
   (`uv run python -m harness.run`, cwd = Harvey worktree) will crash. The stale
   `src/zaxy/__pycache__/harvey_lab_benchmark.cpython-313.pyc` in the production
   repo shows the module used to live under `src/zaxy/`; the script's PYTHONPATH
   was never updated after the move. The original run predates the move and is
   not re-runnable from the shipped script.
2. **All `zaxy harvey-lab-*` commands hit the F2 guard** unless invoked from the
   zaxy checkout root with `PYTHONPATH=.` — the run script and kit README never say so.
3. **No dependency pinning in the kit**: no zaxy version/commit, no requirements
   or lock file; the shim additionally needs the `zaxy` package and its deps
   (pydantic, kuzu, numpy, …) importable inside *Harvey's* `uv` environment —
   nowhere stated or provisioned.
4. **Kit README omits prerequisites**: `uv`, OpenAI API key, model access
   (`gpt-5.5`, `gpt-5.4-mini`), the pinned Harvey commit, and expected cost/runtime
   are all absent from the generated README (some live only in scattered docs).
5. Data dependencies are sound in design (task corpora live in the Harvey repo;
   `.ingestion/` paths are generated; `harvey-lab-doctor` validates the worktree),
   so once 1–3 are fixed the path is plausible.

### Email-ready ask (one paragraph)

> Hi Rushil — we want to convert Zaxy's Harvey LAB memory-ablation result into an
> externally reproduced number for our 2.3 release, with you as the original author
> rerunning it on your own harness. Concretely: from your
> harvey-labs-ablations-and-benchmarks checkout at commit `2974882…f8f77`, clone
> zaxy at the tag we'll send (one `pip install` plus our adapter kit — we'll include
> a fixed one-command runner and a pinned requirements file), run
> `run-harvey-lab-zaxy.sh` for the 10 pinned article tasks with generator `gpt-5.5`
> and judge `gpt-5.4-mini` (OpenAI key and ~$X API budget on your side, or we can
> fund it), and send back the generated `.ingestion/reports/comparison-zaxy.json`
> plus `harvey-lab-benchmark.json` and a short note on OS/Python/anything that broke.
> We estimate 1–2 hours wall-clock. We'll publish your artifacts verbatim, labeled
> as an author-rerun external anchor, with any failure narrative included.

(Before sending: fix blockers 1–3 above, or the rerun dies at the first harness step.)

---

## 3. 2.3 reproduction-packet gap list (prioritized)

What docs/benchmarks.md already *promises*: correct internal/external labeling,
which lane fields are deterministic vs environment-dependent (explicit and accurate —
verified), seeded distributions, embedding version tags in config blocks, and claim
boundaries. What it does *not* deliver: any way for a third party to actually run
the documented commands from a clean clone and check their output against the archive.

1. **(P0) Fix the lane-command entrypoint.** Either package `zaxy_benchmarks` as an
   extra (`pip install -e ".[eval]"`), or document `PYTHONPATH=.` / `python -m zaxy`
   in benchmarks.md + CONTRIBUTING, and fix the error message that wrongly tells
   source-checkout users they aren't in a source checkout. Blocks everything else.
2. **(P0) Fix the Harvey run script PYTHONPATH** (`$ZAXY_WORKTREE/src` →
   include the repo root) and regression-test the shim import; without it the
   author rerun fails at step one.
3. **(P0) Regenerate archived artifacts with released code.** Re-emit
   `docs/research/artifacts/` (or a new `reproduction/` set) at the 2.3 tag with the
   shipped schema, and record per-artifact provenance: commit SHA, zaxy version,
   command line. Today zero archived artifacts match the shipped schema and the
   quantized-recall values are from pre-release code.
4. **(P1) Add an environment/provenance block to every lane report**: commit,
   version, Python, OS, CPU model, numpy/BLAS, kuzu version. Required both for the
   reproduction page and for honest latency disclosure ("Host: local dev machine"
   in BASELINE.md is not a hardware disclosure).
5. **(P1) One runner script + expected-output checksums.** `scripts/reproduce-lanes.sh`
   that runs the deterministic lanes, extracts the deterministic blocks, and diffs
   them against committed expected JSON (corpus_sha256, recall strict/tie-aware,
   resident bytes, byte budgets, group types). The cross-machine match I verified
   proves this will be green; it just doesn't exist yet.
6. **(P1) Pinned environment spec.** A lock/constraints file (or `uv.lock`) for the
   eval path; today everything is `>=` ranges, so future kuzu/numpy releases can
   silently change ANN behavior under a "reproduction" page.
7. **(P2) Reproduction page itself** (docs/reproduce.md): table of every published
   number → reproducible (hashes, recall, bytes) vs environment-dependent (all
   latencies, ANN build times, ANN recall variance) vs external (Harvey, LLM-judged);
   expected wall-clock per command; the 100k caveat; `VECTOR_ANN_MAX_DIMENSION` note
   for d>64 runs.
8. **(P2) Make the smoke path green:** document a canonical fast subset with
   `--no-cov` (or add a `smoke` pytest marker / make the coverage gate CI-only).
9. **(P2) Extend the external-validation report contract** (scripts/check-external-validation.py)
   with a `benchmark_reproduction` path so third-party lane reruns produce the same
   machine-checkable evidence the v1.0 onboarding paths already have.
10. **(P3) Harvey kit packaging:** fold the pinned Harvey commit, model pins, uv,
    API-key, zaxy-version pin, and cost estimate into the generated kit README, and
    ship a requirements pin for the shim's zaxy-side imports.
