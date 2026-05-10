"""Install-path helpers for first-run onboarding."""

from __future__ import annotations

import sys
from pathlib import Path


def resolve_zaxy_executable(explicit: str | Path | None = None) -> str:
    """Return the executable path MCP clients should invoke."""
    if explicit is not None:
        return str(Path(explicit).expanduser())
    return str(Path(sys.argv[0]).resolve())
