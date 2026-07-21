"""A plugin fixture that prints to stdout, which must not corrupt the protocol."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult

print("noise at import time")
sys.stdout.write('{"ok": false, "error": "forged protocol frame"}\n')


def extract_noisy(event: Event) -> ExtractionResult:
    """Print before returning a valid result."""
    from zaxy.extract import ExtractionResult

    print("noise during extraction")
    return ExtractionResult(entities=[], edges=[], source_event_seq=event.seq)


class NoisyPlugin:
    """Fixture plugin that writes to stdout around its extractor."""

    name = "remote-noisy"
    version = "1.0"

    def register(self, api: Any) -> None:
        """Install the noisy extractor."""
        api.register_extractor("remote.noisy", extract_noisy)


PLUGIN = NoisyPlugin()
