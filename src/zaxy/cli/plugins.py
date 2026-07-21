"""CLI surface for inspecting external Zaxy plugins.

``zaxy plugin list`` discovers plugins (installed ``zaxy.plugins`` entry points,
``ZAXY_PLUGINS`` config specs, and ``ZAXY_PLUGINS_OUT_OF_PROCESS`` subprocess
plugins), loads them with the same isolated, idempotent path the fabric uses, and
reports each plugin's name/version/source/status/error. Out-of-process plugins
report ``source="subprocess"``; one that fails to start is listed as ``failed``
with its error rather than being omitted.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import typer

plugin_app = typer.Typer(help="Inspect external Zaxy plugins (extractors and projection backends)")


@plugin_app.command("list")
def plugin_list(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List discovered Zaxy plugins and their load status."""
    from zaxy.config import get_settings
    from zaxy.plugins import load_plugins

    report = load_plugins(get_settings())
    if json_output:
        typer.echo(
            json.dumps(
                {"plugins": [asdict(result) for result in report.results]},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not report.results:
        typer.echo("No plugins discovered")
        return
    for result in report.results:
        line = f"{result.name} {result.version or '-'} [{result.source}] {result.status}"
        if result.error:
            line += f": {result.error}"
        typer.echo(line)
