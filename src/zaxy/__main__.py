"""CLI entrypoint for Zaxy.

Commands:
    serve       Start the MCP server over stdio or SSE.
    replay      Replay an Eventloom log and print integrity report.
    compact     Compact an Eventloom log (create snapshot).
    status      Check connectivity to Neo4j and Pathlight.

Example::

    python -m zaxy serve
    python -m zaxy replay .eventloom/work.jsonl
    python -m zaxy status

The command implementations live in the ``zaxy.cli`` package; this module
remains the ``python -m zaxy`` and console-script entrypoint.
"""

from __future__ import annotations

import sys

if len(sys.argv) > 1 and sys.argv[1] == "--version":
    from zaxy.release import package_version

    print(f"zaxy {package_version()}")
    raise SystemExit(0)

from zaxy.cli import app as app
from zaxy.cli import main as main

if __name__ == "__main__":
    main()
