#!/usr/bin/env python
"""Enforce Zaxy's total coverage ratchet from coverage.py XML output."""

from __future__ import annotations

import argparse
import sys
import tomllib
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root containing pyproject.toml")
    parser.add_argument(
        "--coverage-xml",
        default="coverage.xml",
        help="coverage.py XML report path, relative to the current directory unless absolute",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    coverage_xml = Path(args.coverage_xml)
    if not coverage_xml.is_absolute():
        coverage_xml = (Path.cwd() / coverage_xml).resolve()

    try:
        floor = _configured_floor(root / "pyproject.toml")
        observed = _observed_total_coverage(coverage_xml)
    except CoverageRatchetError as exc:
        print(f"Coverage ratchet failed: {exc}", file=sys.stderr)
        return 2

    if observed < floor:
        print(
            "Coverage ratchet failed: "
            f"observed {_format_percent(observed)} is below floor {_format_percent(floor)}",
            file=sys.stderr,
        )
        return 1

    print(
        "Coverage ratchet passed: "
        f"observed {_format_percent(observed)} meets floor {_format_percent(floor)}"
    )
    return 0


class CoverageRatchetError(ValueError):
    """Raised when the ratchet inputs cannot be read or parsed."""


def _configured_floor(pyproject_path: Path) -> Decimal:
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        raw_floor = pyproject["tool"]["zaxy"]["coverage"]["min_total_percent"]
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise CoverageRatchetError(f"missing coverage floor in {pyproject_path}") from exc

    try:
        return Decimal(str(raw_floor))
    except InvalidOperation as exc:
        raise CoverageRatchetError(f"invalid coverage floor {raw_floor!r}") from exc


def _observed_total_coverage(coverage_xml: Path) -> Decimal:
    try:
        root = ET.parse(coverage_xml).getroot()
    except (FileNotFoundError, ET.ParseError) as exc:
        raise CoverageRatchetError(f"cannot read coverage XML at {coverage_xml}") from exc

    line_rate = root.attrib.get("line-rate")
    if line_rate is None:
        raise CoverageRatchetError(f"coverage XML at {coverage_xml} has no line-rate")

    try:
        return Decimal(line_rate) * Decimal("100")
    except InvalidOperation as exc:
        raise CoverageRatchetError(f"invalid coverage line-rate {line_rate!r}") from exc


def _format_percent(value: Decimal) -> str:
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded}%"


if __name__ == "__main__":
    raise SystemExit(main())
