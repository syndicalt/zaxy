"""Regression tests for `zaxy export-keygen` private key permissions and overwrite safety.

Covers two security fixes:
- The private key file must never be briefly world/group-readable between
  write and chmod (TOCTOU window), regardless of the process umask.
- Re-running the command must not silently clobber an existing keypair
  unless --force is passed, matching the ide-config/hooks/init convention.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from typer.testing import CliRunner

from zaxy.__main__ import app


def test_export_keygen_writes_private_key_0600_under_permissive_umask(tmp_path: Path) -> None:
    out_private = tmp_path / "signing.pem"
    out_public = tmp_path / "signing.pub"

    previous_umask = os.umask(0)  # most permissive umask: nothing is masked off
    try:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "export-keygen",
                "--out-private",
                str(out_private),
                "--out-public",
                str(out_public),
            ],
        )
    finally:
        os.umask(previous_umask)

    assert result.exit_code == 0, result.output
    assert out_private.exists()
    mode = stat.S_IMODE(out_private.stat().st_mode)
    assert mode == 0o600, f"expected private key mode 0600, got {oct(mode)}"


def test_export_keygen_refuses_to_overwrite_existing_key_without_force(tmp_path: Path) -> None:
    out_private = tmp_path / "signing.pem"
    out_public = tmp_path / "signing.pub"

    runner = CliRunner()
    first = runner.invoke(
        app,
        ["export-keygen", "--out-private", str(out_private), "--out-public", str(out_public)],
    )
    assert first.exit_code == 0, first.output
    original_private_bytes = out_private.read_bytes()
    original_public_text = out_public.read_text()

    second = runner.invoke(
        app,
        ["export-keygen", "--out-private", str(out_private), "--out-public", str(out_public)],
    )

    assert second.exit_code != 0
    assert out_private.read_bytes() == original_private_bytes
    assert out_public.read_text() == original_public_text


def test_export_keygen_force_overwrites_existing_key(tmp_path: Path) -> None:
    out_private = tmp_path / "signing.pem"
    out_public = tmp_path / "signing.pub"

    runner = CliRunner()
    first = runner.invoke(
        app,
        ["export-keygen", "--out-private", str(out_private), "--out-public", str(out_public)],
    )
    assert first.exit_code == 0, first.output
    original_private_bytes = out_private.read_bytes()

    second = runner.invoke(
        app,
        [
            "export-keygen",
            "--out-private",
            str(out_private),
            "--out-public",
            str(out_public),
            "--force",
        ],
    )

    assert second.exit_code == 0, second.output
    assert out_private.read_bytes() != original_private_bytes
    mode = stat.S_IMODE(out_private.stat().st_mode)
    assert mode == 0o600
