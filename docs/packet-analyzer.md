# LLM Packet Analyzer

Zaxy can run an observe-only OpenAI-compatible packet analyzer. This is optional
diagnostic/high-fidelity capture, not the default memory path. The analyzer
forwards each request to one configured upstream endpoint and records
request/response provenance to Eventloom, so it can consume provider quota and
requires the runtime and upstream to support the same wire API.

```bash
zaxy packet-analyzer \
  --upstream-base-url https://api.openai.com/v1 \
  --upstream-api-key "$OPENAI_API_KEY" \
  --eventloom-path .eventloom \
  --session-id zaxy-default \
  --host 127.0.0.1 \
  --port 8787
```

Point an OpenAI-compatible client at `http://127.0.0.1:8787/v1`. The analyzer
streams upstream response chunks back to the client, then appends an
`llm.packet.completed` event with:

- provider path, method, status code, model, and usage counts;
- request and response hashes;
- captured request/response JSON bodies;
- a small allowlist of non-secret headers.

The append is handled by a background Eventloom writer after the response body is
observed. Expensive work such as graph projection, embeddings, consolidation,
and memory summarization should run from the cold path after packet capture, not
in the request forwarding path.

## Cold-Path Projection

Default `zaxy init` uses deterministic capture through MCP lifecycle events,
observer hooks, transcript turns, command/file observations, and
`memory_checkout`. To include packet capture commands in first-run onboarding
output, opt in explicitly:

```bash
zaxy init . --capture-mode packet
# or: zaxy init . --capture-mode hybrid
```

Run the projection worker separately to turn completed packet captures into
compact, retrieval-ready `llm.packet.projected` events:

```bash
zaxy packet-project \
  --eventloom-path .eventloom \
  --session-id zaxy-default
```

For a local supervisor or terminal session, run the same worker in watch mode:

```bash
zaxy packet-project \
  --eventloom-path .eventloom \
  --session-id zaxy-default \
  --watch
```

Add `--graph` when Neo4j is available and the worker should upsert each newly
projected packet immediately:

```bash
zaxy packet-project \
  --eventloom-path .eventloom \
  --session-id zaxy-default \
  --graph
```

Projection is idempotent by source event hash. It extracts bounded request and
response summaries, preserves the source packet sequence and hash, and leaves the
raw packet capture available for audit replay.

Projected packets participate in source-aware retrieval as
`packet_projection` context. Context assembly can cite the projection event back
to Eventloom and include it in the active memory working set as `memory`.
By default, one recent projected packet memory can be selected proactively during
context assembly via `CONTEXT_PACKET_MEMORY_ENABLED=true` and
`CONTEXT_PACKET_MEMORY_SLOTS=1`. Packet memories reinforced through context
feedback receive a deterministic selection boost over merely recent packet
memories while keeping the original packet and projection events immutable.

`zaxy doctor` reports packet-memory status for the active session. It warns when
captured packets have not been projected yet, and `zaxy init` includes optional
next steps for starting both the analyzer and projector. The doctor report also
prints counters for `captured`, `projected`, `unprojected`, `reinforced`, and
currently `eligible` packet memories.

For focused inspection without the full doctor report:

```bash
zaxy packet-status \
  --eventloom-path .eventloom \
  --session-id zaxy-default
```

When no packets have been captured yet, `packet-status` prints concrete analyzer
and projector commands plus the local OpenAI-compatible base URL to configure in
the client.

## Modes

The first implementation is observe-only. It does not inject memory, choose a
model, retry provider calls, or transform responses. Future modes can add cached
memory injection while keeping live retrieval and graph writes out of the hot
path.
