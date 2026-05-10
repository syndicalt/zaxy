# LLM Packet Analyzer

Zaxy can run an observe-only OpenAI-compatible packet analyzer. The analyzer is
not a router: it forwards each request to one configured upstream endpoint and
records request/response provenance to Eventloom.

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

## Modes

The first implementation is observe-only. It does not inject memory, choose a
model, retry provider calls, or transform responses. Future modes can add cached
memory injection while keeping live retrieval and graph writes out of the hot
path.
