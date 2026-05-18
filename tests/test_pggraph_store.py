"""Tests for the experimental pgGraph projection store."""

from __future__ import annotations

from typing import Any

import pytest

from zaxy.pggraph_store import PgGraphStore


class FakeCursor:
    """Small async cursor capturing SQL statements."""

    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.rows: list[dict[str, Any]] = []

    async def __aenter__(self) -> FakeCursor:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def execute(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] | None = None,
    ) -> None:
        self.connection.statements.append((sql, params))

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class FakeConnection:
    """Small async connection capturing statements and transaction calls."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, object | None]] = []
        self.cursor_obj = FakeCursor(self)
        self.commits = 0
        self.closed = False

    def cursor(self, *, row_factory: object | None = None) -> FakeCursor:
        return self.cursor_obj

    async def execute(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] | None = None,
    ) -> None:
        self.statements.append((sql, params))

    async def commit(self) -> None:
        self.commits += 1

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_pggraph_store_init_schema_creates_projection_tables_and_registers_pggraph() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.init_schema()

    sql = "\n".join(statement for statement, _params in connection.statements)
    assert "CREATE TABLE IF NOT EXISTS zaxy_pggraph_entities" in sql
    assert "CREATE TABLE IF NOT EXISTS zaxy_pggraph_edges" in sql
    assert "graph.add_table" in sql
    assert "graph.add_edge" in sql
    assert "graph.build" in sql
    assert connection.commits == 1


@pytest.mark.asyncio
async def test_pggraph_store_close_closes_existing_connection() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.close()

    assert connection.closed is True
