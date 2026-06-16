"""Outbound delivery for the memory export contract (the optional push layer).

Pull (a consumer calling the contract) stays primary. This module is the thin
*push* convenience: take whatever :func:`zaxy.export_view.build_memory_export`
produces and hand it to a generic sink. There is no second projection path — push
builds the bundle through the same shared helper the pull surfaces use.

Push is operator-side (CLI / library), never an MCP tool: the MCP surface stays
pull-only. Recurring delivery is left to an external scheduler (cron / the OS)
invoking the one-shot push; Zaxy does not run a delivery daemon.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from zaxy.export_view import build_memory_export

if TYPE_CHECKING:
    from zaxy.export_view import ExportSelector
    from zaxy.retrieval_cache import SessionRetrievalCache


class Sink(Protocol):
    """A destination an export bundle can be delivered to."""

    def deliver(self, bundle: dict[str, Any]) -> None:
        """Deliver one bundle. Raises on failure."""
        ...


class FileSink:
    """Write the bundle JSON to a local file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def deliver(self, bundle: dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class WebhookSink:
    """POST the bundle JSON to an HTTP(S) endpoint (dependency-free via urllib).

    Sends ``Authorization: Bearer <token>`` when a token is supplied. Only
    ``http``/``https`` URLs are accepted, so a misconfigured ``file://`` target
    cannot turn an outbound push into a local write.
    """

    def __init__(self, url: str, *, token: str | None = None, timeout: float = 30.0) -> None:
        scheme = urlsplit(url).scheme.lower()
        if scheme not in {"http", "https"}:
            raise ValueError("webhook url must be http or https")
        self.url = url
        self.token = token
        self.timeout = timeout

    def deliver(self, bundle: dict[str, Any]) -> None:
        data = json.dumps(bundle, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - scheme is validated http(s) in __init__
            response.read()


def push_memory_export(
    session_id: str,
    selector: ExportSelector | None = None,
    *,
    retrieval_cache: SessionRetrievalCache,
    signing_key: dict[str, Any] | None = None,
    sink: Sink,
) -> dict[str, Any]:
    """Build an export bundle and deliver it to ``sink``; return the bundle.

    Goes through :func:`build_memory_export`, so a pushed bundle is byte-identical
    to the same export pulled — one projection path, signed or unsigned.
    """
    bundle = build_memory_export(
        session_id, selector, retrieval_cache=retrieval_cache, signing_key=signing_key
    )
    sink.deliver(bundle)
    return bundle
