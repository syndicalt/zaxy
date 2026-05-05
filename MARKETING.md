# Zaxy

**Event-sourced temporal knowledge graph fabric for AI agent memory.**

---

## The Problem

Today's AI agents forget. When you hand off from one session to the next, your agent starts from zero—its "memory" is a grab-bag of markdown files and vector chunks that have been stripped of relationships, timing, and causality. This causes four systemic failures:

| Failure | Why it hurts |
|---------|-------------|
| **No relational reasoning** | Vector similarity can't trace "Alice *reported to* Bob *until* June"—it just finds words that look alike. |
| **No temporal awareness** | Facts overwrite each other silently. The agent can't answer "What did we believe then vs. now?" |
| **Non-replayable** | When an agent hallucinates, you can't reconstruct the exact sequence of events that led it astray. |
| **Un-auditable** | No provenance chain for compliance, debugging, or post-mortems. |

The result? Agents that feel like goldfish with PhDs—brilliant in the moment, but unable to learn, reason across time, or explain themselves.

---

## Why Zaxy Wins

The dominant approaches today—markdown files, vector DBs, or the two combined—each fail in ways that look fine in a demo and collapse in production.

### Markdown Files Alone

Markdown is human-readable, which makes it tempting as an agent memory format. But it's a **write-only dump**: no query language, no relationships, no versioning. When your agent writes "Alice is the tech lead" to a markdown file, there's no way to later ask "Who reported to Alice in Q2?" or "When did Alice stop being tech lead?" The file just keeps growing, context windows overflow, and the agent drowns in irrelevant text.

**Zaxy replaces markdown dumps with structured events.** Every fact is typed (`goal.created`, `task.proposed`, `fact.invalidated`), extracted into entities and relationships, and stored in a queryable graph.

### Vector DBs Alone

Vector databases excel at semantic similarity: "find me things that *sound like* this query." But similarity is not truth. A vector DB can't tell you that "Alice *reported to* Bob" and "Bob *left the company*" together imply "Alice now reports to Carol." It can't do multi-hop reasoning, can't follow causal chains, and can't answer temporal questions like "what did we believe before the reorg?"

Worse, vector DBs are **destructive**: updating a fact means overwriting its embedding. The old version is gone. There's no history, no replay, no audit trail.

**Zaxy keeps vectors as one signal among many** (exact match, BM25, traversal), but the ground truth is always the immutable event log and the temporal graph derived from it.

### Markdown + Vector DB (The "Best Practice")

This is what most agent frameworks ship today: markdown files for "structured" memory, vector DB for "semantic" retrieval. It looks comprehensive. It isn't.

| Capability | Markdown + Vector | Zaxy |
|------------|-------------------|------|
| **Relational reasoning** | No—vectors don't encode edges | Yes—native graph traversal |
| **Temporal validity** | No—facts overwrite silently | Yes—`valid_from` / `valid_to` on every entity |
| **Replay / debug** | No—no sequence or provenance | Yes—hash-chained event log, deterministic replay |
| **Audit / compliance** | No—no integrity guarantees | Yes—SHA-256 chain, tamper-evident |
| **Multi-agent sharing** | Fragile—file locks, merge conflicts | Clean—session-sharded logs, shared graph |
| **Context token reduction** | Poor—raw text chunks bloat prompts | 70–90% reduction—structured paths vs. raw text |
| **Cost at scale** | High—LLM re-extraction on every read | Low—rule-based extraction, graph is pre-structured |

The markdown+vector approach is **lossy at every step**: unstructured text → chunked embeddings → similarity-ranked noise. Zaxy is **lossless at the foundation**: structured events → exact entities → precise retrieval.

---

## What We're Building

Zaxy is a **persistent memory layer** for AI agents that treats every fact as an immutable event in a hash-chained log, projected into a **bi-temporal knowledge graph** (Neo4j). Agents write events; Zaxy extracts entities and relationships; the graph answers structured queries across time.

```
Agent Action  →  Eventloom (immutable JSONL)  →  Hybrid Extraction  →  Neo4j (temporal KG)
                     ↑                                                          ↓
                     └────────────  Pathlight traces every step  ←  Hybrid Query Router
```

**Key capabilities:**

- **Immutable audit trail** — Every event is SHA-256 hashed and chained. Tamper-evident by design.
- **Bi-temporal graph** — Facts carry `valid_from` and `valid_to`. Query "what was true on March 3rd" natively.
- **Hybrid extraction** — Rule-based extractors for typed events (60–80% cheaper than LLM extraction), with LLM fallback for unstructured input.
- **Hybrid retrieval** — Exact match + vector similarity + BM25 keyword + graph traversal, fused and ranked.
- **Multi-agent session sharding** — Each agent/session gets its own append-only log while sharing a single graph for cross-agent reasoning.
- **MCP-native** — Drop-in memory for any Model Context Protocol client (LangGraph, CrewAI, Claude Desktop, etc.).
- **Observable** — Every operation traced to Pathlight for debugging, breakpoints, and diff.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11+ | Core runtime |
| **Immutable Log** | Eventloom (JSONL + SHA-256 hash chain) | Source of truth, append-only, cross-process safe |
| **Graph DB** | Neo4j Community 5.26+ | Bi-temporal entity/relationship storage |
| **Graph Driver** | `neo4j` (official) | Native Cypher with full schema control |
| **Extraction** | Rule-based registry + LLM fallback | Typed events → structured graph automatically |
| **Query Router** | Custom fusion engine | Exact + vector + BM25 + traversal with configurable weights |
| **Interface** | MCP (Model Context Protocol) | Framework-agnostic tools: `memory_append`, `memory_query`, `memory_replay`, `memory_invalidate` |
| **Transport** | stdio (default) or SSE (daemon mode) | Local agent integration or remote service |
| **Observability** | Pathlight | Execution tracing, breakpoints, diff |
| **Metrics** | Prometheus | SLO monitoring for append/query latency |
| **Testing** | pytest + pytest-asyncio + pytest-cov + pytest-benchmark | ≥90% coverage gate, benchmark regression tests |
| **Packaging** | Docker + docker-compose | One-command local stack |

---

## Performance

| Operation | Target | Achieved |
|-----------|--------|----------|
| Event append | <50ms | ~57μs (local lock + write) |
| Rule-based extraction | <10ms | ~1.9μs |
| Neo4j upsert | <100ms | ~107μs |
| Hybrid query | <200ms | ~83μs (exact), variable (BM25/traversal) |
| End-to-end context retrieval | <300ms | Sub-100ms for exact hits |

---

## Elevator Pitch

> **"Zaxy gives AI agents a memory they can actually reason with—not a junk drawer of vector chunks, but a structured, time-aware knowledge graph with a tamper-proof audit trail. Every fact has a birthday and a death certificate. Every decision is replayable. And it plugs into any agent framework in five minutes via MCP."**

---

## Who It's For

- **Agent framework builders** who need persistent, structured memory without vendor lock-in.
- **Multi-agent teams** where agents hand off work and need shared context with isolated write paths.
- **Compliance-sensitive applications** (finance, healthcare, legal) that require auditable decision trails.
- **Researchers** studying agent behavior who need to replay exact sessions and inspect what the agent "knew" at any point in time.

---

## Get Started

```bash
# 1. Clone and setup
git clone https://github.com/syndicalt/zaxy.git && cd zaxy
./scripts/setup.sh

# 2. Start the stack
docker compose up -d

# 3. Verify
python -m zaxy status
pytest

# 4. Connect your agent via MCP
# Add to your mcpServers config:
# {
#   "zaxy": {
#     "command": "python",
#     "args": ["-m", "zaxy", "serve"]
#   }
# }
```

---

## License

MIT

---

*Built with discipline. Tested obsessively. No magic—just immutable events, a graph, and clean contracts.*
