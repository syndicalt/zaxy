"""A plugin fixture that raises during import, before any object is defined."""

from __future__ import annotations

raise RuntimeError("fixture plugin exploded at import time")
