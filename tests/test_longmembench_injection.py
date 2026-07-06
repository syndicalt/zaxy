from __future__ import annotations

import shlex
from pathlib import Path

from zaxy_benchmarks.longmembench import (
    build_longmembench_external_run_manifest,
    write_longmembench_external_run_manifest,
)


def test_plan_script_quotes_adversarial_dataset_value(tmp_path: Path) -> None:
    """A dataset value with shell metacharacters must not be interpolated raw.

    The generated script runs each command via `eval "$command"`, so any
    unquoted, attacker-controlled value baked directly into a command string
    is a command-injection vector. The fix stores the dataset path in a single
    shlex-quoted DATASET_PATH variable and references it via ${DATASET_PATH}
    instead of inlining the raw string.
    """
    adversarial_dataset = 'evil.json"; touch /tmp/zaxy_pwned; echo "'

    manifest = build_longmembench_external_run_manifest(
        dataset=adversarial_dataset,
        output_dir=str(tmp_path / "run"),
    )
    written = write_longmembench_external_run_manifest(manifest, tmp_path / "run")
    script_text = written.script_path.read_text(encoding="utf-8")

    quoted_dataset = shlex.quote(adversarial_dataset)

    # The dataset must appear only inside a single, safely-quoted assignment.
    assert f"DATASET_PATH={quoted_dataset}" in script_text

    # The two command lines must reference the quoted variable, not the raw value.
    assert (
        "OFFICIAL_EVAL_COMMAND=\"python3 evaluate_qa.py ${EVALUATOR_MODEL} "
        "${RUN_OUTPUT_DIR}/zaxy-hypotheses.jsonl ${LONGMEMEVAL_WORKTREE}/${DATASET_PATH}\""
        in script_text
    )
    assert (
        "PRINT_METRICS_COMMAND=\"python3 ${LONGMEMEVAL_WORKTREE}/src/evaluation/print_qa_metrics.py "
        "${RUN_OUTPUT_DIR}/zaxy-hypotheses.jsonl.eval-results-${EVALUATOR_MODEL} "
        "${LONGMEMEVAL_WORKTREE}/${DATASET_PATH}\""
        in script_text
    )

    # The raw adversarial string must never appear unquoted/unescaped in the script,
    # i.e. the only place the raw payload shows up is inside the shlex-quoted assignment.
    raw_occurrences = script_text.count(adversarial_dataset)
    quoted_occurrences = script_text.count(quoted_dataset)
    assert raw_occurrences == quoted_occurrences

    # Sanity: the dangerous breakout substring must not appear outside of quoting.
    assert '"; touch /tmp/zaxy_pwned; echo "' not in script_text.replace(quoted_dataset, "")
