"""Auditable Neo4j schema migrations for Zaxy projections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class SchemaMigration:
    """A named, idempotent Neo4j schema migration."""

    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        joined = "\n\n".join(statement.strip() for statement in self.statements)
        return sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SchemaMigrationStatus:
    """Recorded state for one schema migration."""

    name: str
    state: str
    expected_checksum: str
    recorded_checksum: str | None
    expected_statement_count: int
    recorded_statement_count: int | None
    applied_at: Any | None
    checksum_ok: bool
    statement_count_ok: bool


SCHEMA_MIGRATIONS: tuple[SchemaMigration, ...] = (
    SchemaMigration(
        "001_entity_version_identity",
        (
            "DROP CONSTRAINT entity_id IF EXISTS",
            "DROP CONSTRAINT entity_version_id IF EXISTS",
            "CREATE CONSTRAINT entity_version_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.session_id, e.name, e.entity_type, e.valid_from) IS UNIQUE",
            "CREATE INDEX entity_lookup IF NOT EXISTS "
            "FOR (e:Entity) ON (e.session_id, e.name, e.entity_type)",
        ),
    ),
    SchemaMigration(
        "002_entity_vector",
        (
            "CREATE VECTOR INDEX entity_vector IF NOT EXISTS "
            "FOR (e:Entity) ON (e.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}",
        ),
    ),
    SchemaMigration(
        "003_entity_fulltext",
        (
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.summary]",
        ),
    ),
    SchemaMigration(
        "004_provenance_backbone",
        (
            "CREATE CONSTRAINT session_id IF NOT EXISTS "
            "FOR (s:Session) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT event_identity IF NOT EXISTS "
            "FOR (ev:Event) REQUIRE (ev.session_id, ev.seq) IS UNIQUE",
            "CREATE INDEX event_hash IF NOT EXISTS "
            "FOR (ev:Event) ON (ev.session_id, ev.hash)",
        ),
    ),
    SchemaMigration(
        "005_event_hash_chain",
        (
            "CREATE INDEX event_prev_hash IF NOT EXISTS "
            "FOR (ev:Event) ON (ev.session_id, ev.prev_hash)",
        ),
    ),
    SchemaMigration(
        "006_source_citation_identity",
        (
            "CREATE CONSTRAINT source_identity IF NOT EXISTS "
            "FOR (src:Source) REQUIRE (src.session_id, src.path) IS UNIQUE",
        ),
    ),
)

CURRENT_SCHEMA_VERSION = SCHEMA_MIGRATIONS[-1].name


def pending_schema_migrations(
    migrations: Iterable[SchemaMigration] = SCHEMA_MIGRATIONS,
    applied_names: set[str] | None = None,
) -> list[SchemaMigration]:
    """Return migrations whose names are not already recorded as applied."""
    applied = applied_names or set()
    return [migration for migration in migrations if migration.name not in applied]


def render_schema_plan(
    migrations: Iterable[SchemaMigration] = SCHEMA_MIGRATIONS,
    applied_names: set[str] | None = None,
) -> str:
    """Render a human-readable migration plan for operators."""
    pending = pending_schema_migrations(migrations, applied_names=applied_names)
    lines = [f"Current schema version: {CURRENT_SCHEMA_VERSION}"]
    if not pending:
        lines.append("No pending migrations.")
        return "\n".join(lines)
    lines.append("Pending migrations:")
    for migration in pending:
        lines.append(f"- {migration.name} ({len(migration.statements)} statements, {migration.checksum[:12]})")
    return "\n".join(lines)


async def fetch_applied_schema_migrations(driver: Any) -> set[str]:
    """Fetch recorded schema migration names from Neo4j."""
    records = await fetch_schema_migration_records(driver)
    return set(records)


async def fetch_schema_migration_records(driver: Any) -> dict[str, dict[str, Any]]:
    """Fetch recorded schema migration audit metadata from Neo4j."""
    assert driver is not None, "Call connect() first"
    result = await driver.execute_query(
        """
        MATCH (m:ZaxySchemaMigration)
        RETURN m.name AS name,
               m.checksum AS checksum,
               m.statement_count AS statement_count,
               m.applied_at AS applied_at
        """
    )
    if not isinstance(result, tuple) or len(result) < 1:
        return {}
    records = result[0]
    applied: dict[str, dict[str, Any]] = {}
    for record in records:
        name = record.get("name")
        if isinstance(name, str):
            applied[name] = {
                "checksum": record.get("checksum"),
                "statement_count": record.get("statement_count"),
                "applied_at": record.get("applied_at"),
            }
    return applied


def schema_migration_status(
    *,
    migrations: Iterable[SchemaMigration] = SCHEMA_MIGRATIONS,
    records: dict[str, dict[str, Any]] | None = None,
) -> list[SchemaMigrationStatus]:
    """Classify migration records as applied, pending, partial, or drifted."""
    recorded = records or {}
    statuses: list[SchemaMigrationStatus] = []
    for migration in migrations:
        record = recorded.get(migration.name)
        if record is None:
            statuses.append(
                SchemaMigrationStatus(
                    name=migration.name,
                    state="pending",
                    expected_checksum=migration.checksum,
                    recorded_checksum=None,
                    expected_statement_count=len(migration.statements),
                    recorded_statement_count=None,
                    applied_at=None,
                    checksum_ok=False,
                    statement_count_ok=False,
                )
            )
            continue
        recorded_checksum = record.get("checksum")
        recorded_statement_count = record.get("statement_count")
        checksum_ok = recorded_checksum == migration.checksum
        statement_count_ok = recorded_statement_count == len(migration.statements)
        state = "applied" if checksum_ok and statement_count_ok else "partial"
        if not checksum_ok and statement_count_ok:
            state = "checksum_mismatch"
        statuses.append(
            SchemaMigrationStatus(
                name=migration.name,
                state=state,
                expected_checksum=migration.checksum,
                recorded_checksum=recorded_checksum if isinstance(recorded_checksum, str) else None,
                expected_statement_count=len(migration.statements),
                recorded_statement_count=(
                    recorded_statement_count if isinstance(recorded_statement_count, int) else None
                ),
                applied_at=record.get("applied_at"),
                checksum_ok=checksum_ok,
                statement_count_ok=statement_count_ok,
            )
        )
    return statuses


def render_schema_recovery_plan(statuses: Iterable[SchemaMigrationStatus]) -> str:
    """Render operator guidance for non-applied schema states."""
    actionable = [status for status in statuses if status.state != "applied"]
    if not actionable:
        return "Schema recovery plan:\nNo recovery actions required."
    lines = ["Schema recovery plan:"]
    for status in actionable:
        lines.append(f"- {status.name}: {status.state}")
        if status.state == "pending":
            lines.append("  Apply normally with `zaxy schema-plan` followed by service startup.")
        elif status.state == "checksum_mismatch":
            lines.append(
                "  Stop projection, inspect migration source drift, and reconcile the recorded checksum."
            )
        else:
            lines.append(
                "  Stop projection, verify Neo4j constraints/indexes manually, then Re-run `zaxy schema-plan`."
            )
    return "\n".join(lines)


async def apply_schema_migrations(
    driver: Any,
    migrations: Iterable[SchemaMigration] = SCHEMA_MIGRATIONS,
    *,
    applied_names: set[str] | None = None,
    dry_run: bool = False,
) -> list[SchemaMigration]:
    """Apply idempotent schema migrations and record audit metadata."""
    assert driver is not None, "Call connect() first"
    if applied_names is None:
        applied_names = await fetch_applied_schema_migrations(driver)
    pending = pending_schema_migrations(migrations, applied_names=applied_names)
    if dry_run:
        return pending

    applied: list[SchemaMigration] = []
    for migration in pending:
        for statement in migration.statements:
            await driver.execute_query(statement)
        await driver.execute_query(
            """
            MERGE (m:ZaxySchemaMigration {name: $name})
            SET m.applied_at = datetime(),
                m.checksum = $checksum,
                m.statement_count = $statement_count
            """,
            name=migration.name,
            checksum=migration.checksum,
            statement_count=len(migration.statements),
        )
        applied.append(migration)
    return applied
