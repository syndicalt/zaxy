"""Tests for operational backup/restore/rotation scripts."""

from __future__ import annotations

import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from zaxy.event import EventLog


def _write_deployment_fixture(root: Path) -> None:
    """Create a minimal production deployment fixture."""
    secrets = root / "secrets"
    secrets.mkdir(parents=True)
    for name in [
        "neo4j_auth.txt",
        "neo4j_password.txt",
        "mcp_admin_token.txt",
        "mcp_remote_auth_token.txt",
        "openai_api_key.txt",
        "pathlight_access_token.txt",
    ]:
        path = secrets / name
        path.write_text("secret\n", encoding="utf-8")
        path.chmod(0o600)

    (root / ".env").write_text(
        "\n".join(
            [
                "ZAXY_ENV=production",
                "NEO4J_URI=bolt://neo4j:7687",
                "NEO4J_CA_CERT=/ssl/bolt/trusted/public.crt",
                "MCP_ADMIN_TOKEN_FILE=secrets/mcp_admin_token.txt",
                "MCP_REMOTE_AUTH_TOKEN_FILE=secrets/mcp_remote_auth_token.txt",
                "MCP_REMOTE_SESSION_HEADER=x-zaxy-session-id",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_backup_creates_archive_manifest_and_excludes_secrets(tmp_path: Path) -> None:
    """Backup should archive event logs, write checksums, and exclude secrets."""
    root = tmp_path / "project"
    eventloom = root / ".eventloom"
    secrets = root / "secrets"
    backup_dir = root / "backups"
    eventloom.mkdir(parents=True)
    secrets.mkdir()
    (eventloom / "default.jsonl").write_text('{"seq":1}\n', encoding="utf-8")
    (root / ".env.example").write_text("EVENTLOOM_PATH=.eventloom\n", encoding="utf-8")
    (secrets / "neo4j_password.txt").write_text("super-secret\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "scripts/backup.sh",
            "--root",
            str(root),
            "--output-dir",
            str(backup_dir),
            "--name",
            "test-backup",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    archive = backup_dir / "test-backup.tar.gz"
    manifest = backup_dir / "test-backup.sha256"
    assert archive.exists()
    assert manifest.exists()
    assert "Created backup" in result.stdout
    assert archive.name in manifest.read_text(encoding="utf-8")

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())

    assert ".eventloom/default.jsonl" in names
    assert ".env.example" in names
    assert "secrets/neo4j_password.txt" not in names


def test_restore_validates_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    """Restore should validate checksums and refuse unsafe overwrites."""
    source = tmp_path / "source"
    source_eventloom = source / ".eventloom"
    backup_dir = source / "backups"
    source_eventloom.mkdir(parents=True)
    (source_eventloom / "default.jsonl").write_text('{"seq":1}\n', encoding="utf-8")

    subprocess.run(
        [
            "bash",
            "scripts/backup.sh",
            "--root",
            str(source),
            "--output-dir",
            str(backup_dir),
            "--name",
            "restore-source",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    archive = backup_dir / "restore-source.tar.gz"
    manifest = backup_dir / "restore-source.sha256"
    target = tmp_path / "target"

    subprocess.run(
        [
            "bash",
            "scripts/restore.sh",
            "--archive",
            str(archive),
            "--manifest",
            str(manifest),
            "--target",
            str(target),
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    assert (target / ".eventloom/default.jsonl").read_text(encoding="utf-8") == '{"seq":1}\n'

    blocked = subprocess.run(
        [
            "bash",
            "scripts/restore.sh",
            "--archive",
            str(archive),
            "--manifest",
            str(manifest),
            "--target",
            str(target),
        ],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert blocked.returncode != 0
    assert "already exists" in blocked.stderr


def test_restore_rejects_bad_manifest(tmp_path: Path) -> None:
    """Restore should fail closed when the manifest does not match the archive."""
    source = tmp_path / "source"
    source_eventloom = source / ".eventloom"
    backup_dir = source / "backups"
    source_eventloom.mkdir(parents=True)
    (source_eventloom / "default.jsonl").write_text('{"seq":1}\n', encoding="utf-8")

    subprocess.run(
        [
            "bash",
            "scripts/backup.sh",
            "--root",
            str(source),
            "--output-dir",
            str(backup_dir),
            "--name",
            "bad-manifest",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = backup_dir / "bad-manifest.sha256"
    manifest.write_text(
        "0" * 64 + "  bad-manifest.tar.gz\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/restore.sh",
            "--archive",
            str(backup_dir / "bad-manifest.tar.gz"),
            "--manifest",
            str(manifest),
            "--target",
            str(tmp_path / "target"),
        ],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "checksum" in result.stderr.lower()


def test_rotate_logs_archives_valid_log_and_truncates_active_file(tmp_path: Path) -> None:
    """Rotation should archive replayable JSONL and leave an empty active log."""
    root = tmp_path / "project"
    log_path = root / ".eventloom" / "default.jsonl"
    archive_dir = root / "archives"
    log = EventLog(log_path)
    log.append(
        "goal.created",
        actor="user",
        payload={"title": "Ship MVP"},
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    log.append(
        "task.proposed",
        actor="agent",
        payload={"taskId": "t1"},
        timestamp=datetime(2024, 1, 2, tzinfo=UTC),
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/rotate-logs.sh",
            "--log",
            str(log_path),
            "--archive-dir",
            str(archive_dir),
            "--name",
            "default-rotated",
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    rotated = archive_dir / "default-rotated.jsonl"
    manifest = archive_dir / "default-rotated.sha256"
    assert "Rotated log" in result.stdout
    assert rotated.exists()
    assert manifest.exists()
    assert log_path.read_text(encoding="utf-8") == ""
    assert EventLog(rotated).verify().ok is True
    assert len(EventLog(rotated).read_all()) == 2


def test_validate_deployment_accepts_secure_production_fixture(tmp_path: Path) -> None:
    """Deployment validation should pass for a secure production fixture."""
    root = tmp_path / "project"
    root.mkdir()
    _write_deployment_fixture(root)

    result = subprocess.run(
        ["bash", "scripts/validate-deployment.sh", "--root", str(root)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Deployment validation passed" in result.stdout


def test_validate_deployment_requires_remote_auth_token(tmp_path: Path) -> None:
    """Production SSE deployments should require remote bearer auth or OIDC."""
    root = tmp_path / "project"
    root.mkdir()
    _write_deployment_fixture(root)
    (root / ".env").write_text(
        "ZAXY_ENV=production\n"
        "NEO4J_URI=bolt://neo4j:7687\n"
        "NEO4J_CA_CERT=/ssl/bolt/trusted/public.crt\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", "scripts/validate-deployment.sh", "--root", str(root)],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "MCP_REMOTE_AUTH_TOKEN" in result.stderr


def test_validate_deployment_accepts_oidc_remote_auth(tmp_path: Path) -> None:
    """Production SSE deployments may use OIDC instead of static bearer auth."""
    root = tmp_path / "project"
    root.mkdir()
    _write_deployment_fixture(root)
    (root / ".env").write_text(
        "ZAXY_ENV=production\n"
        "NEO4J_URI=bolt://neo4j:7687\n"
        "NEO4J_CA_CERT=/ssl/bolt/trusted/public.crt\n"
        "MCP_ADMIN_TOKEN_FILE=secrets/mcp_admin_token.txt\n"
        "MCP_OIDC_ISSUER=https://idp.example\n"
        "MCP_OIDC_AUDIENCE=zaxy\n"
        "MCP_OIDC_JWKS_URL=https://idp.example/.well-known/jwks.json\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", "scripts/validate-deployment.sh", "--root", str(root)],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Deployment validation passed" in result.stdout


def test_validate_deployment_requires_admin_token(tmp_path: Path) -> None:
    """Production deployments should require an admin token for replay/invalidate."""
    root = tmp_path / "project"
    root.mkdir()
    _write_deployment_fixture(root)
    (root / ".env").write_text(
        "ZAXY_ENV=production\n"
        "NEO4J_URI=bolt://neo4j:7687\n"
        "NEO4J_CA_CERT=/ssl/bolt/trusted/public.crt\n"
        "MCP_REMOTE_AUTH_TOKEN_FILE=secrets/mcp_remote_auth_token.txt\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", "scripts/validate-deployment.sh", "--root", str(root)],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "MCP_ADMIN_TOKEN" in result.stderr


def test_validate_deployment_rejects_world_readable_secret(tmp_path: Path) -> None:
    """Deployment validation should fail closed on permissive secret files."""
    root = tmp_path / "project"
    root.mkdir()
    _write_deployment_fixture(root)
    (root / "secrets/mcp_remote_auth_token.txt").chmod(0o644)

    result = subprocess.run(
        ["bash", "scripts/validate-deployment.sh", "--root", str(root)],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "world-readable" in result.stderr


def test_validate_deployment_rejects_plaintext_neo4j_without_ca(tmp_path: Path) -> None:
    """Production Neo4j should not use plaintext bolt without custom CA config."""
    root = tmp_path / "project"
    root.mkdir()
    _write_deployment_fixture(root)
    (root / ".env").write_text(
        "ZAXY_ENV=production\n"
        "NEO4J_URI=bolt://neo4j:7687\n"
        "MCP_REMOTE_AUTH_TOKEN_FILE=secrets/mcp_remote_auth_token.txt\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", "scripts/validate-deployment.sh", "--root", str(root)],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "NEO4J_CA_CERT" in result.stderr


def test_release_check_runs_quality_gates_in_order(tmp_path: Path) -> None:
    """Release gate should run static checks, tests, packaging, and deployment validation."""
    root = tmp_path / "project"
    root.mkdir()
    _write_deployment_fixture(root)
    log_path = tmp_path / "commands.log"
    stub = tmp_path / "stub.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$0 $*\" >> \"$RELEASE_CHECK_LOG\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            "scripts/release-check.sh",
            "--root",
            str(root),
            "--ruff-cmd",
            str(stub),
            "--mypy-cmd",
            str(stub),
            "--pytest-cmd",
            str(stub),
            "--packet-smoke-cmd",
            str(stub),
            "--package-cmd",
            str(stub),
            "--docs-cmd",
            str(stub),
            "--validate-cmd",
            str(stub),
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
        env={"RELEASE_CHECK_LOG": str(log_path)},
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert "Release check passed" in result.stdout
    assert lines == [
        f"{stub} check src tests",
        f"{stub} src",
        f"{stub} --tb=short",
        f"{stub} ",
        f"{stub} --root {root}",
        f"{stub} --root {root}",
        f"{stub} --root {root}",
    ]


def test_release_check_fails_fast_on_quality_gate_failure(tmp_path: Path) -> None:
    """Release gate should stop when an earlier gate fails."""
    root = tmp_path / "project"
    root.mkdir()
    _write_deployment_fixture(root)
    log_path = tmp_path / "commands.log"
    ok = tmp_path / "ok.sh"
    fail = tmp_path / "fail.sh"
    ok.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$0 $*\" >> \"$RELEASE_CHECK_LOG\"\n",
        encoding="utf-8",
    )
    fail.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$0 $*\" >> \"$RELEASE_CHECK_LOG\"\n"
        "exit 7\n",
        encoding="utf-8",
    )
    ok.chmod(0o700)
    fail.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            "scripts/release-check.sh",
            "--root",
            str(root),
            "--ruff-cmd",
            str(ok),
            "--mypy-cmd",
            str(fail),
            "--pytest-cmd",
            str(ok),
            "--package-cmd",
            str(ok),
            "--docs-cmd",
            str(ok),
            "--validate-cmd",
            str(ok),
        ],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
        env={"RELEASE_CHECK_LOG": str(log_path)},
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert result.returncode == 7
    assert lines == [
        f"{ok} check src tests",
        f"{fail} src",
    ]
