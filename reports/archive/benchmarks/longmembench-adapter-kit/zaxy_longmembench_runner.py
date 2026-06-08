"""Placeholder external LongMemBench runner for Zaxy.

This file documents the required hypothesis contract. The actual answer
generation path should live in the external validation checkout so model
settings, prompts, and provider credentials are visible to the validator.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_hypothesis(path: str | Path, question_id: str, hypothesis: str) -> None:
    """Append one official LongMemEval hypothesis row."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"question_id": question_id, "hypothesis": hypothesis}) + "\n")
