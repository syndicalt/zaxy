"""A plugin fixture whose extractor genuinely segfaults the worker process."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult


def extract_segfault(event: Event) -> ExtractionResult:
    """Dereference a null pointer, killing this process with SIGSEGV."""
    ctypes.string_at(0)
    raise AssertionError("unreachable")


class SegfaultPlugin:
    """Fixture plugin registering an extractor that hard-crashes."""

    name = "remote-segfault"
    version = "1.0"

    def register(self, api: Any) -> None:
        """Install the segfaulting extractor."""
        api.register_extractor("remote.segfault", extract_segfault)


PLUGIN = SegfaultPlugin()
