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
    assert driver is not None, "Call connect() first"
    result = await driver.execute_query(
        """
        MATCH (m:ZaxySchemaMigration)
        RETURN m.name AS name
        """
    )
    if not isinstance(result, tuple) or len(result) < 1:
        return set()
    records = result[0]
    applied: set[str] = set()
    for record in records:
        name = record.get("name")
        if isinstance(name, str):
            applied.add(name)
    return applied


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
