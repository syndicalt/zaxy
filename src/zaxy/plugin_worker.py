"""Child-process entry point for out-of-process Zaxy plugins.

Run as ``python -m zaxy.plugin_worker <module:attr>``. The worker imports the
plugin, collects the extractors it registers, and then serves ``describe`` /
``extract`` requests as newline-delimited JSON on stdin/stdout (see
:mod:`zaxy.plugin_ipc`).

Two details are load-bearing:

* The protocol owns a *duplicate* of the real stdout file descriptor and
  ``sys.stdout`` is re-pointed at stderr before any plugin code is imported.
  A plugin that calls ``print()`` therefore cannot corrupt the wire.
* Plugin code is imported only here, in the child. The host never imports it,
  which is what makes an import-time crash survivable.

Projection backends cannot cross the process boundary — a ``ProjectionStore`` is
a live object with connections, not data — so the capture API accepts the
registration and reports it as unsupported rather than silently dropping it.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult


class _CaptureAPI:
    """A :class:`~zaxy.plugins.PluginAPI` look-alike that records registrations.

    The worker hands this to the plugin instead of the real API: registrations
    are captured in-process here and exposed to the host as *names*, never as
    callables.
    """

    def __init__(self) -> None:
        self.extractors: dict[str, Callable[[Event], ExtractionResult]] = {}
        self.unsupported: list[str] = []

    def register_extractor(
        self,
        event_type: str,
        fn: Callable[[Event], ExtractionResult],
    ) -> None:
        """Record a rule extractor for ``event_type``."""
        self.extractors[event_type] = fn

    def register_projection_backend(self, name: str, factory: Callable[..., Any]) -> None:
        """Record that ``name`` was requested, and that it cannot be served remotely."""
        self.unsupported.append(name)


def _resolve(reference: str) -> Any:
    """Resolve a ``"module:attr"`` import string to its object."""
    import importlib

    module_name, separator, attr_path = reference.partition(":")
    if not separator or not module_name.strip() or not attr_path.strip():
        raise ValueError(f"plugin spec {reference!r} must be in 'module:attr' form")
    obj: Any = importlib.import_module(module_name.strip())
    for part in attr_path.strip().split("."):
        obj = getattr(obj, part)
    return obj


def _str_attr(plugin: object, attr: str, *, default: str) -> str:
    """Return a non-empty string attribute from ``plugin`` or ``default``."""
    value = getattr(plugin, attr, None)
    if isinstance(value, str) and value.strip():
        return value
    return default


def _serve(reference: str, stdin: Any, write: Callable[[str], None]) -> None:
    """Import the plugin and serve requests until stdin closes or shutdown."""
    from zaxy.plugin_ipc import (
        PROTOCOL_VERSION,
        decode_event,
        encode_extraction_result,
    )

    api = _CaptureAPI()
    load_error: str | None = None
    name = reference
    version = ""
    try:
        plugin = _resolve(reference)
        name = _str_attr(plugin, "name", default=reference)
        version = _str_attr(plugin, "version", default="")
        plugin.register(api)
    except BaseException as exc:  # noqa: BLE001 - the whole point is to report, not die
        load_error = f"{type(exc).__name__}: {exc}"

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            write(json.dumps({"ok": False, "error": f"malformed request: {exc}"}))
            continue

        op = request.get("op")
        if op == "shutdown":
            return
        if op == "describe":
            if load_error is not None:
                write(json.dumps({"ok": False, "error": load_error}))
                continue
            write(
                json.dumps(
                    {
                        "ok": True,
                        "protocol": PROTOCOL_VERSION,
                        "name": name,
                        "version": version,
                        "event_types": sorted(api.extractors),
                        "unsupported_backends": sorted(api.unsupported),
                    }
                )
            )
            continue
        if op == "extract":
            if load_error is not None:
                write(json.dumps({"ok": False, "error": load_error}))
                continue
            try:
                event_type = str(request["event_type"])
                extractor = api.extractors[event_type]
                result = extractor(decode_event(request["event"]))
                write(json.dumps({"ok": True, "result": encode_extraction_result(result)}))
            except Exception as exc:  # noqa: BLE001 - plugin faults are data, not crashes
                write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
            continue
        write(json.dumps({"ok": False, "error": f"unknown op {op!r}"}))


def main(argv: list[str] | None = None) -> int:
    """Serve one out-of-process plugin named by ``argv[0]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m zaxy.plugin_worker <module:attr>", file=sys.stderr)
        return 2

    # Take exclusive ownership of the protocol channel before importing plugin
    # code: dup the real stdout, then point sys.stdout at stderr so any plugin
    # print() lands in the log instead of mid-protocol.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    channel = os.fdopen(protocol_fd, "w", encoding="utf-8")
    sys.stdout = sys.stderr

    def write(line: str) -> None:
        channel.write(line + "\n")
        channel.flush()

    try:
        _serve(args[0], sys.stdin, write)
    finally:
        channel.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
