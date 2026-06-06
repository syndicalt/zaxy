#!/usr/bin/env bash
set -euo pipefail

HARVEY_WORKTREE="${1:-.}"
OUTPUT_DIR="${2:-reports/benchmarks/harvey-lab-memory-ablation}"
TASK_FILTER="${3:-${HARVEY_TASK_FILTER:-}}"
GENERATOR_MODEL="${HARVEY_GENERATOR_MODEL:-gpt-5.5}"
JUDGE_MODEL="${HARVEY_JUDGE_MODEL:-gpt-5.4-mini}"
JUDGE_PARALLEL="${HARVEY_JUDGE_PARALLEL:-1}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZAXY_WORKTREE="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ZAXY_PYTHONPATH="$ZAXY_WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_MANIFEST_JSON="$SCRIPT_DIR/harvey-lab-external-run.json"
RUN_MANIFEST_JSON="$OUTPUT_DIR/harvey-lab-external-run.json"
READY_JSON="$OUTPUT_DIR/harvey-lab-ready.json"
STATUS_JSON="$OUTPUT_DIR/harvey-lab-status.json"
HARVEY_COMPARISON_JSON=".ingestion/reports/comparison-zaxy.json"
if [[ "$GENERATOR_MODEL" == "HARVEY_GENERATOR_MODEL" || "$JUDGE_MODEL" == "HARVEY_JUDGE_MODEL" ]]; then
  echo "Unresolved Harvey model placeholders; set HARVEY_GENERATOR_MODEL and HARVEY_JUDGE_MODEL or generate the plan with explicit models." >&2
  exit 2
fi
HARVEY_WORKTREE="$(cd "$HARVEY_WORKTREE" && pwd)"
if [[ "$SOURCE_MANIFEST_JSON" != "$RUN_MANIFEST_JSON" ]]; then
  cp "$SOURCE_MANIFEST_JSON" "$RUN_MANIFEST_JSON"
fi
ADAPTER_DIR="$(mktemp -d)"
HARVEY_ADAPTER_PATH="$HARVEY_WORKTREE/scripts/memory_ablation/raw_rg_memory.py"
HARVEY_ADAPTER_BACKUP="$ADAPTER_DIR/raw_rg_memory.py.original"
HARVEY_ADAPTER_HAD_ORIGINAL=0
restore_harvey_adapter() {
  if [[ "$HARVEY_ADAPTER_HAD_ORIGINAL" == "1" && -f "$HARVEY_ADAPTER_BACKUP" ]]; then
    cp "$HARVEY_ADAPTER_BACKUP" "$HARVEY_ADAPTER_PATH"
  else
    rm -f "$HARVEY_ADAPTER_PATH"
  fi
  rm -rf "$ADAPTER_DIR"
}
trap restore_harvey_adapter EXIT

zaxy harvey-lab-doctor "$HARVEY_WORKTREE"
zaxy harvey-lab-preflight "$HARVEY_WORKTREE" --task-filter "$TASK_FILTER"
zaxy harvey-lab-ready "$HARVEY_WORKTREE" --generator "$GENERATOR_MODEL" --judge "$JUDGE_MODEL" --task-filter "$TASK_FILTER" --json | tee "$READY_JSON"
zaxy harvey-lab-adapter-kit --output-dir "$ADAPTER_DIR"
mkdir -p "$HARVEY_WORKTREE/scripts/memory_ablation"
if [[ -f "$HARVEY_ADAPTER_PATH" ]]; then
  cp "$HARVEY_ADAPTER_PATH" "$HARVEY_ADAPTER_BACKUP"
  HARVEY_ADAPTER_HAD_ORIGINAL=1
fi
cp "$ADAPTER_DIR/raw_rg_memory.py" "$HARVEY_ADAPTER_PATH"
cd "$HARVEY_WORKTREE"

TASK_ID=corporate-governance/assess-impact-of-ftc-noncompete-ban-on-existing-employment-agreements
RUN_ID=zaxy-corporate-governance__assess-impact-of-ftc-noncompete-ban-on-existing-employment-agreements
SLUG=corporate-governance__assess-impact-of-ftc-noncompete-ban-on-existing-employment-agreements
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=corporate-ma/analyze-change-of-control-provisions-across-targets-material-contracts
RUN_ID=zaxy-corporate-ma__analyze-change-of-control-provisions-across-targets-material-contracts
SLUG=corporate-ma__analyze-change-of-control-provisions-across-targets-material-contracts
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=corporate-ma/draft-acquisition-due-diligence
RUN_ID=zaxy-corporate-ma__draft-acquisition-due-diligence
SLUG=corporate-ma__draft-acquisition-due-diligence
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=corporate-ma/review-data-room-red-flag-review
RUN_ID=zaxy-corporate-ma__review-data-room-red-flag-review
SLUG=corporate-ma__review-data-room-red-flag-review
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=data-privacy-cybersecurity/compare-privacy-program-documentation-against-applicable-data-protection-regulations
RUN_ID=zaxy-data-privacy-cybersecurity__compare-privacy-program-documentation-against-applicable-data-protection-regulations
SLUG=data-privacy-cybersecurity__compare-privacy-program-documentation-against-applicable-data-protection-regulations
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=litigation-dispute-resolution/build-litigation-case-timeline
RUN_ID=zaxy-litigation-dispute-resolution__build-litigation-case-timeline
SLUG=litigation-dispute-resolution__build-litigation-case-timeline
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=litigation-dispute-resolution/categorize-document-production-set-by-relevance-and-privilege
RUN_ID=zaxy-litigation-dispute-resolution__categorize-document-production-set-by-relevance-and-privilege
SLUG=litigation-dispute-resolution__categorize-document-production-set-by-relevance-and-privilege
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=litigation-dispute-resolution/review-document-production-set-for-attorney
RUN_ID=zaxy-litigation-dispute-resolution__review-document-production-set-for-attorney
SLUG=litigation-dispute-resolution__review-document-production-set-for-attorney
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=litigation-dispute-resolution/review-privilege-log-clawback-review
RUN_ID=zaxy-litigation-dispute-resolution__review-privilege-log-clawback-review
SLUG=litigation-dispute-resolution__review-privilege-log-clawback-review
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

TASK_ID=white-collar-defense-investigations/compare-document-production-set-against-subpoena-request-categories
RUN_ID=zaxy-white-collar-defense-investigations__compare-document-production-set-against-subpoena-request-categories
SLUG=white-collar-defense-investigations__compare-document-production-set-against-subpoena-request-categories
if [[ -n "$TASK_FILTER" && "$TASK_FILTER" != "$TASK_ID" && "$TASK_FILTER" != "$SLUG" && "$TASK_FILTER" != "$RUN_ID" ]]; then
  echo "Skipping $TASK_ID due to HARVEY_TASK_FILTER=$TASK_FILTER"
else
INDEX_DIR=".ingestion/indexes/$SLUG/zaxy"
NORMALIZATION_JSON="$(uv run python - "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path
import scripts.memory_ablation.normalize_corpus as normalize_corpus

task_id = sys.argv[1]
root = Path.cwd()
normalization = normalize_corpus.prepare_normalized_corpus(root / "tasks" / task_id / "documents", root / ".ingestion")
print(json.dumps(normalization))
PY
)"
NORMALIZED_ROOT="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["normalized_corpus_root"])')"
SOURCE_MAP="$(printf '%s' "$NORMALIZATION_JSON" | python -c 'import json, sys; print(json.load(sys.stdin)["source_map_path"])')"
zaxy harvey-lab-index --normalized-corpus-root "$NORMALIZED_ROOT" --source-map "$SOURCE_MAP" --output-dir "$INDEX_DIR"
HARVEY_MEMORY_MANIFEST="$INDEX_DIR/manifest.json" PYTHONPATH="$ZAXY_PYTHONPATH" uv run python -m harness.run --model "$GENERATOR_MODEL" --task "$TASK_ID" --run-id "$RUN_ID" --reasoning-effort low
uv run python -m evaluation.run_eval --judge-model "$JUDGE_MODEL" --parallel "$JUDGE_PARALLEL" --run-id "$RUN_ID" --task "$TASK_ID"
zaxy harvey-lab-normalize-run --harvey-worktree "$HARVEY_WORKTREE" --run-id "$RUN_ID" --task-id "$TASK_ID" --manifest "$INDEX_DIR/manifest.json"
uv run python scripts/memory_ablation/validate_result.py --run-dir ".ingestion/runs/$RUN_ID" --worktree-root "$HARVEY_WORKTREE"
fi

if [[ -n "$TASK_FILTER" ]]; then
  zaxy harvey-lab-status "$HARVEY_WORKTREE" --json | tee "$STATUS_JSON" || true
  uv run python scripts/memory_ablation/collect_results.py --worktree "$HARVEY_WORKTREE" --dedupe-latest --output "$HARVEY_COMPARISON_JSON" || true
else
  zaxy harvey-lab-status "$HARVEY_WORKTREE" --json | tee "$STATUS_JSON"
  uv run python scripts/memory_ablation/collect_results.py --worktree "$HARVEY_WORKTREE" --dedupe-latest --output "$HARVEY_COMPARISON_JSON"
fi
zaxy harvey-lab-import "$HARVEY_WORKTREE" --output-dir "$OUTPUT_DIR"
REPORT_JSON="$OUTPUT_DIR/harvey-lab-benchmark.json"
PUBLISH_MD="$OUTPUT_DIR/publishable-statistics.md"
if [[ -n "$TASK_FILTER" ]]; then
  zaxy harvey-lab-validate "$REPORT_JSON" || true
  echo "Filtered Harvey run imported; full publish gate is intentionally skipped until all tasks are complete."
else
  zaxy harvey-lab-validate "$REPORT_JSON" --require-complete
  zaxy harvey-lab-gate "$REPORT_JSON"
  zaxy harvey-lab-publish "$REPORT_JSON" --output "$PUBLISH_MD"
fi
