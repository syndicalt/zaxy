"""A plugin fixture whose extractor never returns."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult


def extract_hang(event: Event) -> ExtractionResult:
    """Sleep forever, so the host must enforce its own deadline."""
    while True:
        time.sleep(3600)


class HangPlugin:
    """Fixture plugin registering an extractor that hangs indefinitely."""

    name = "remote-hang"
    version = "1.0"

    def register(self, api: Any) -> None:
        """Install the hanging extractor."""
        api.register_extractor("remote.hang", extract_hang)


PLUGIN = HangPlugin()
