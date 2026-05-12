"""Tests for Neo4j schema migration tooling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from zaxy.schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaMigration,
    apply_schema_migrations,
    fetch_applied_schema_migrations,
    fetch_schema_migration_records,
    pending_schema_migrations,
    render_schema_plan,
    render_schema_recovery_plan,
    schema_migration_status,
)


def test_pending_schema_migrations_filters_applied_names() -> None:
    migrations = [
        SchemaMigration("001_first", ("RETURN 1",)),
        SchemaMigration("002_second", ("RETURN 2",)),
    ]

    pending = pending_schema_migrations(migrations, applied_names={"001_first"})

    assert [migration.name for migration in pending] == ["002_second"]


def test_render_schema_plan_lists_current_migrations() -> None:
    plan = render_schema_plan()

    assert f"Current schema version: {CURRENT_SCHEMA_VERSION}" in plan
    assert "entity_version_identity" in plan
    assert "entity_vector" in plan
    assert "entity_fulltext" in plan
    assert "provenance_backbone" in plan
    assert "event_hash_chain" in plan


async def test_apply_schema_migrations_records_each_migration() -> None:
    driver = AsyncMock()
    driver.execute_query.side_effect = [
        ([{"name": "000_old"}], None, None),
        ([], None, None),
        ([], None, None),
        ([], None, None),
        ([], None, None),
    ]
    migrations = [
        SchemaMigration("001_first", ("RETURN 1",)),
        SchemaMigration("002_second", ("RETURN 2",)),
    ]

    applied = await apply_schema_migrations(driver, migrations=migrations)

    assert [migration.name for migration in applied] == ["001_first", "002_second"]
    cypher = [call.args[0] for call in driver.execute_query.await_args_list]
    assert "RETURN 1" in cypher
    assert "RETURN 2" in cypher
    assert sum("MERGE (m:ZaxySchemaMigration" in statement for statement in cypher) == 2


async def test_apply_schema_migrations_skips_recorded_migrations() -> None:
    driver = AsyncMock()
    driver.execute_query.return_value = ([{"name": "001_first"}], None, None)
    migrations = [
        SchemaMigration("001_first", ("RETURN 1",)),
        SchemaMigration("002_second", ("RETURN 2",)),
    ]

    applied = await apply_schema_migrations(driver, migrations=migrations)

    assert [migration.name for migration in applied] == ["002_second"]
    cypher = [call.args[0] for call in driver.execute_query.await_args_list]
    assert "RETURN 1" not in cypher
    assert "RETURN 2" in cypher


async def test_apply_schema_migrations_supports_dry_run() -> None:
    driver = AsyncMock()
    migrations = [SchemaMigration("001_first", ("RETURN 1",))]

    applied = await apply_schema_migrations(
        driver,
        migrations=migrations,
        applied_names=set(),
        dry_run=True,
    )

    assert [migration.name for migration in applied] == ["001_first"]
    driver.execute_query.assert_not_awaited()


async def test_apply_schema_migrations_requires_driver() -> None:
    with pytest.raises(AssertionError):
        await apply_schema_migrations(None)


async def test_fetch_applied_schema_migrations_handles_neo4j_records() -> None:
    driver = AsyncMock()
    driver.execute_query.return_value = ([{"name": "001_first"}, {"name": "002_second"}], None, None)

    applied = await fetch_applied_schema_migrations(driver)

    assert applied == {"001_first", "002_second"}


async def test_fetch_schema_migration_records_includes_audit_metadata() -> None:
    driver = AsyncMock()
    driver.execute_query.return_value = (
        [
            {
                "name": "001_first",
                "checksum": "abc",
                "statement_count": 2,
                "applied_at": "2026-05-11T00:00:00Z",
            }
        ],
        None,
        None,
    )

    records = await fetch_schema_migration_records(driver)

    assert records["001_first"] == {
        "checksum": "abc",
        "statement_count": 2,
        "applied_at": "2026-05-11T00:00:00Z",
    }


def test_schema_migration_status_detects_checksum_and_partial_states() -> None:
    first = SchemaMigration("001_first", ("RETURN 1", "RETURN 2"))
    second = SchemaMigration("002_second", ("RETURN 3",))

    statuses = schema_migration_status(
        migrations=[first, second],
        records={
            "001_first": {
                "checksum": "wrong",
                "statement_count": 1,
                "applied_at": "2026-05-11T00:00:00Z",
            }
        },
    )

    assert statuses[0].state == "partial"
    assert statuses[0].checksum_ok is False
    assert statuses[0].statement_count_ok is False
    assert statuses[1].state == "pending"


def test_render_schema_recovery_plan_gives_operator_actions() -> None:
    migration = SchemaMigration("001_first", ("RETURN 1", "RETURN 2"))
    statuses = schema_migration_status(
        migrations=[migration],
        records={
            "001_first": {
                "checksum": migration.checksum,
                "statement_count": 1,
                "applied_at": "2026-05-11T00:00:00Z",
            }
        },
    )

    plan = render_schema_recovery_plan(statuses)

    assert "Schema recovery plan:" in plan
    assert "001_first: partial" in plan
    assert "Re-run `zaxy schema-plan`" in plan
