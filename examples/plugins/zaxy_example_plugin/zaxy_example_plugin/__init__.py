"""A minimal, installable reference Zaxy plugin.

It demonstrates the full external plugin contract end to end:

* a ``PLUGIN`` object with ``name`` / ``version`` / ``register`` (the
  :class:`zaxy.plugins.ZaxyPlugin` protocol), and
* a rule extractor for the ``example.note`` event type, installed through the
  :class:`zaxy.plugins.PluginAPI` handed to ``register``.

Install it (``pip install .`` from this directory) to expose it via the
``zaxy.plugins`` entry point, or load it without installing by setting
``ZAXY_PLUGINS=zaxy_example_plugin:PLUGIN``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult
    from zaxy.plugins import PluginAPI

__version__ = "0.1.0"


def extract_example_note(event: Event) -> ExtractionResult:
    """Project an ``example.note`` event into a single example entity."""
    from zaxy.extract import ExtractedEntity, ExtractionResult

    payload = event.payload or {}
    text = payload.get("text") or payload.get("note")
    entity = ExtractedEntity(
        name=f"example-note:{event.seq}",
        entity_type="example_note",
        observed_at=event.timestamp,
        summary=str(text) if text else None,
    )
    return ExtractionResult(entities=[entity], edges=[], source_event_seq=event.seq)


class ExamplePlugin:
    """Reference plugin registering the ``example.note`` extractor."""

    name = "zaxy-example-plugin"
    version = __version__

    def register(self, api: PluginAPI) -> None:
        """Install this plugin's capabilities on the given :class:`PluginAPI`."""
        api.register_extractor("example.note", extract_example_note)


# The object referenced by the entry point / ZAXY_PLUGINS import string.
PLUGIN = ExamplePlugin()
