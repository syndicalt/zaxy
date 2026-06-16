"""Hybrid extraction engine: rule-based + LLM fallback.

Public surface is unchanged: import names straight from ``zaxy.extract``.
The rule extractors live in ``zaxy.extract.rules`` and self-register on import.
"""

from zaxy.extract import rules as _rules  # noqa: F401  (registers extractors on import)
from zaxy.extract.core import (
    ExtractedEdge,
    ExtractedEntity,
    ExtractionResult,
    extract,
    register,
)

__all__ = ["ExtractedEdge", "ExtractedEntity", "ExtractionResult", "extract", "register"]
