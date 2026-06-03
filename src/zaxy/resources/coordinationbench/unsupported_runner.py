"""Explicit unsupported runner for pinned but unfinished competitor adapters.

The CoordinationBench runner protocol appends ``--workload`` and ``--output``
to the manifest command. This helper writes a small failure payload and exits
non-zero so unsupported Quarq/Hybi adapters leave durable logs without being
mistaken for same-harness scored results.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an unsupported CoordinationBench adapter result stub.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": args.adapter,
        "adapter_contract": "coordinationbench-v1",
        "status": "unsupported",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "workload_file": str(Path(args.workload)),
        "reason": (
            "This packaged manifest is pinned for disclosure, but no completed "
            "same-harness runner adapter is available yet."
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload["reason"], file=sys.stderr)
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
