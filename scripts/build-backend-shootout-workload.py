#!/usr/bin/env python3
"""Materialize a LongMemEval dataset for backend shootout runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zaxy.live_benchmark import build_longmemeval_workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build backend-shootout Eventloom/query files from LongMemEval.")
    parser.add_argument("--dataset", required=True, type=Path, help="Cleaned LongMemEval JSON dataset")
    parser.add_argument("--questions", type=_positive_int, default=None, help="Maximum questions to materialize")
    parser.add_argument("--eventloom-output", required=True, type=Path, help="Output Eventloom JSONL file")
    parser.add_argument("--queries-output", required=True, type=Path, help="Output backend-shootout query JSON")
    args = parser.parse_args()

    args.eventloom_output.parent.mkdir(parents=True, exist_ok=True)
    args.queries_output.parent.mkdir(parents=True, exist_ok=True)
    if args.eventloom_output.exists():
        args.eventloom_output.unlink()

    try:
        _validate_strict_json_file(args.dataset)
        eventlog, cases, workload = build_longmemeval_workload(
            args.eventloom_output,
            args.dataset,
            questions=args.questions,
        )
    except json.JSONDecodeError as exc:
        raise SystemExit("dataset contains malformed JSON") from exc
    except NonStandardJsonConstantError as exc:
        raise SystemExit(f"dataset contains non-standard numeric constant {exc.constant}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    queries = [
        {
            "query": case.query,
            "expected_terms": list(case.expected_terms),
        }
        for case in cases
    ]
    if not queries:
        raise SystemExit("No backend-shootout queries were materialized")
    args.queries_output.write_text(_strict_json_dumps(queries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(queries)} backend-shootout queries and "
        f"{workload.event_count} Eventloom events to {eventlog.path}"
    )


def _strict_json_dumps(payload: object, **kwargs: object) -> str:
    return json.dumps(payload, allow_nan=False, **kwargs)


class NonStandardJsonConstantError(ValueError):
    def __init__(self, constant: str) -> None:
        self.constant = constant
        super().__init__(constant)


def _validate_strict_json_file(path: Path) -> None:
    def reject_constant(value: str) -> None:
        raise NonStandardJsonConstantError(value)

    json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("questions must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("questions must be a positive integer")
    return value


if __name__ == "__main__":
    main()
