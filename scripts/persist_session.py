"""Persist this development session into Zaxy memory.

This demonstrates that Zaxy can store its own development context.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from zaxy.event import EventLog
from zaxy.extract import extract
from zaxy.graph import GraphStore


async def main() -> None:
    """Store the build session in Zaxy."""
    log = EventLog(".eventloom/build_session.jsonl")
    graph = GraphStore("bolt://localhost:7687", "neo4j", "testpassword")
    await graph.connect()
    await graph.init_schema()

    ts = datetime.now(timezone.utc)

    # Events from this development session
    session_events = [
        ("session.started", "developer", {"project": "zaxy", "goal": "Build event-sourced temporal KG fabric"}),
        ("bug.fixed", "assistant", {"component": "graph.py", "issue": "Cypher syntax error in search_exact WHERE clause", "commit": "0ea5bfd"}),
        ("test.added", "assistant", {"component": "test_mcp.py", "coverage_gained": "74% → 94%", "tests": ["main lifecycle", "unknown tool dispatch"]}),
        ("test.added", "assistant", {"component": "test_graph.py", "tests": ["keyword temporal filter", "traversal relation_type", "traversal temporal filter"]}),
        ("test.added", "assistant", {"component": "test_trace.py", "tests": ["trace_append error branch"]}),
        ("test.added", "assistant", {"component": "test_graph.py", "tests": ["full_pipeline_event_to_query"], "type": "integration"}),
        ("benchmark.added", "assistant", {"component": "test_benchmarks.py", "targets_verified": ["event_append <50ms", "extraction <10ms", "graph_search <5ms"]}),
        ("bug.fixed", "assistant", {"component": "graph.py", "issue": "search_traversal depth parameter not inlined for Neo4j"}),
        ("bug.fixed", "assistant", {"component": "graph.py", "issue": "search_traversal temporal Cypher using r instead of rel in ALL() predicate"}),
        ("bug.fixed", "assistant", {"component": "docker-compose.yml", "issue": "Neo4j plugin name gds → graph-data-science"}),
        ("bug.fixed", "assistant", {"component": "mcp_server.py", "issue": "CLI serve ignored configured server instance"}),
        ("config.updated", "assistant", {"component": "pyproject.toml", "change": "Added pytest-benchmark to dev deps"}),
        ("docs.updated", "assistant", {"component": "AGENTS.md", "changes": ["mark runbooks complete", "update metrics: 124 tests, 97.85% coverage"]}),
        ("demo.ran", "assistant", {"component": "scripts/test_drive.py", "result": "Full pipeline verified: append → extract → upsert → query → invalidate"}),
        ("demo.ran", "assistant", {"component": "scripts/mcp_smoke_test.py", "result": "MCP server smoke test passed: initialize, list_tools, memory_append, memory_query"}),
        ("session.metrics", "assistant", {"tests": 124, "coverage": "97.85%", "commits": 4, "docker_services": ["neo4j", "neo4j-test"]}),
    ]

    print("💾 Persisting build session to Zaxy...")
    for etype, actor, payload in session_events:
        event = log.append(etype, actor, payload, timestamp=ts)
        result = extract(event)
        await graph.upsert_extraction(result)
        print(f"   [{event.seq}] {etype}")

    # Query it back
    from zaxy.query import QueryRouter
    router = QueryRouter(store=graph, default_limit=10)

    print("\n🔍 Querying stored session context...")

    queries = [
        "Cypher syntax error",
        "full_pipeline_event_to_query",
        "benchmark",
        "MCP server smoke test",
    ]

    for q in queries:
        chunks = await router.query(q)
        print(f"\n   Query: '{q}' → {len(chunks)} results")
        for c in chunks:
            print(f"      [{c.source}] {c.content[:100]}...")

    # Show Eventloom integrity
    replay = log.replay()
    print(f"\n📋 Eventloom: {replay.integrity.total_events} events, hash chain OK={replay.integrity.ok}")

    await graph.close()
    print("\n✅ Session context persisted to Zaxy.")


if __name__ == "__main__":
    asyncio.run(main())
