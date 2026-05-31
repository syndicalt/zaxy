#!/usr/bin/env python3
"""Validate a v1.0 external-validation report for release-gate use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from zaxy.external_validation import validate_external_validation_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="External-validation report JSON")
    args = parser.parse_args()

    errors = validate_external_validation_report(_load_json(args.report))
    if errors:
        print("External validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"External validation passed: {args.report}")
    return 0


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"__load_error__": f"cannot read {path}: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
