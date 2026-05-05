"""Minimal end-to-end test of Zaxy without any agent framework.

This script demonstrates the core memory loop:
1. Append events to Eventloom
2. Extract entities/relations
3. Upsert to Neo4j
4. Query with temporal filters
5. Print context chunks

Run this to verify your Zaxy installation works:
    python scripts/test_drive.py
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone

from zaxy.event import EventLog
from zaxy.extract import extract
from zaxy.graph import GraphStore
from zaxy.query import QueryRouter


NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "testpassword")


async def main() -> None:
    """Run a simulated agent session through Zaxy."""
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        log_path = fh.name

    log = EventLog(log_path)
    graph = GraphStore(NEO4J_URI, *NEO4J_AUTH)
    await graph.connect()
    await graph.init_schema()

    # Clean slate (remove in production!)
    await graph._driver.execute_query("MATCH (n) DETACH DELETE n")

    # ------------------------------------------------------------------
    # Simulate an agent session
    # ------------------------------------------------------------------
    session_time = datetime(2024, 6, 15, tzinfo=timezone.utc)

    events = [
        ("goal.created", "user", {"title": "Ship MVP", "priority": "high"}),
        ("task.proposed", "agent", {"task_id": "t1", "summary": "Design landing page"}),
        ("task.claimed", "user", {"task_id": "t1"}),
        ("preference.changed", "user", {"theme": "dark", "lang": "en"}),
    ]

    print("📥 Appending events to Eventloom...")
    for i, (etype, actor, payload) in enumerate(events, start=1):
        event = log.append(etype, actor, payload, timestamp=session_time)
        result = extract(event)
        await graph.upsert_extraction(result)
        print(f"   [{event.seq}] {etype} → {len(result.entities)} entities, {len(result.edges)} edges")

    # ------------------------------------------------------------------
    # Query the memory
    # ------------------------------------------------------------------
    router = QueryRouter(store=graph, default_limit=10)

    print("\n🔍 Query: 'Ship MVP'")
    chunks = await router.query("Ship MVP")
    for c in chunks:
        print(f"   [{c.source}] {c.content} (score={c.score})")

    print("\n🔍 Query: 'landing page' (keyword)")
    chunks = await router.query("landing page")
    for c in chunks:
        print(f"   [{c.source}] {c.content} (score={c.score})")

    print("\n🔍 Query: 'Ship MVP' at 2024-07-01 (before invalidation)")
    chunks = await router.query("Ship MVP", temporal_point="2024-07-01T00:00:00Z")
    print(f"   Results: {len(chunks)}")

    # ------------------------------------------------------------------
    # Invalidate a fact and show temporal filtering
    # ------------------------------------------------------------------
    print("\n🗑️  Invalidating 'Ship MVP' goal as of 2024-08-01...")
    await graph.invalidate_entity("Ship MVP", "goal", "2024-08-01T00:00:00Z")

    print("\n🔍 Query: 'Ship MVP' at 2024-09-01 (after invalidation)")
    chunks = await router.query("Ship MVP", temporal_point="2024-09-01T00:00:00Z")
    print(f"   Results: {len(chunks)} (should be 0)")

    print("\n🔍 Query: 'Ship MVP' at 2024-07-01 (before invalidation)")
    chunks = await router.query("Ship MVP", temporal_point="2024-07-01T00:00:00Z")
    print(f"   Results: {len(chunks)} (should still find it)")

    # ------------------------------------------------------------------
    # Show Eventloom integrity
    # ------------------------------------------------------------------
    print("\n📋 Eventloom integrity report:")
    replay = log.replay()
    print(f"   Hash chain: {'OK' if replay.integrity.ok else 'FAILED'}")
    print(f"   Total events: {replay.integrity.total_events}")

    summary = log.handoff_summary()
    print(f"\n📊 Handoff summary:")
    print(f"   Goals: {summary.get('goals', [])}")
    print(f"   Open tasks: {len(summary.get('open_tasks', []))}")
    print(f"   Completed tasks: {len(summary.get('completed_tasks', []))}")
    print(f"   Last actor: {summary.get('last_actor')}")

    # Cleanup
    await graph._driver.execute_query("MATCH (n) DETACH DELETE n")
    await graph.close()
    print("\n✅ Test drive complete.")


if __name__ == "__main__":
    asyncio.run(main())
