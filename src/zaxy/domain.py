"""Project domain helpers for safe default session separation."""

from __future__ import annotations

import re
from pathlib import Path


def slug_domain(value: str) -> str:
    """Return a safe lowercase domain slug for session IDs."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "project"


def derive_domain(root: str | Path | None = None) -> str:
    """Derive a project domain from a filesystem root or the current directory."""
    path = Path(root).resolve() if root is not None else Path.cwd().resolve()
    return slug_domain(path.name)


def domain_default_session(domain: str) -> str:
    """Return the default session ID for a project domain."""
    return f"{slug_domain(domain)}-default"
