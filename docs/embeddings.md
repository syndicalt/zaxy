# Embeddings

Embeddings are optional but important for semantic retrieval. Zaxy supports a
deterministic local hash provider and an OpenAI-compatible hosted provider. Both
produce fixed-size vectors attached to extracted entities and query text, then
selected projection backend vector search participates in result fusion.

The hash provider is designed for tests, offline development, and deterministic
behavior. It does not provide high-quality semantic meaning, but it allows vector
code paths to run without network access or secrets. This is valuable for CI and
for local contributors who only need to verify mechanics.

For the local-first profile:

```bash
zaxy local-profile
zaxy local-profile --output .env.local
zaxy local-profile --projection-backend embedded --output .env.local
zaxy local-profile --check
```

The generated profile uses `PROJECTION_BACKEND=embedded`,
`EMBEDDING_PROVIDER=hash`, `RERANKER_PROVIDER=lexical`, and sidecar autostart
disabled. It intentionally omits hosted API key variables so the default local
path stays offline and deterministic. Use an explicit sidecar profile only when
you need a Neo4j or pgGraph comparison target.

The hosted provider is selected with:

```bash
EMBEDDING_ENABLED=true
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY_FILE=secrets/openai_api_key.txt
```

`OPENAI_BASE_URL` can point at any OpenAI-compatible embeddings endpoint. Keep
`EMBEDDING_DIMENSION` aligned with the model and the selected backend vector
index. If the dimension changes, rebuild the vector index and replay affected
events so entity vectors are regenerated consistently.

Secrets should be supplied through `OPENAI_API_KEY_FILE` in production. Direct
`OPENAI_API_KEY` is convenient for local testing but can leak through process
inspection or shell history. See [security.md](security.md) and
[configuration.md](configuration.md).

Embedding generation happens after extraction. The provider receives structured
entity text, not arbitrary raw payload dumps. This reduces the chance of
embedding secrets and improves result quality by keeping the vector input close
to the graph fact being stored.

Stored vectors in the embedded backend carry the producing provider's version
tag (for example `hash@<fingerprint>-dim1536`). Search never compares vectors
across version tags, so changing providers or models cannot silently return
garbage matches; `zaxy doctor` reports mixed-version corpora, and
`zaxy memory re-embed --session-id <session>` batch-migrates stale-version
vectors to the active provider without touching Eventloom. Scale is opt-in and
never silently lossy: below `VECTOR_ANN_THRESHOLD` (default `1000000` vectors
per session — deliberately high, keeping ANN opt-in until the HNSW path beats
exact search in the vector-scale lane) the embedded backend uses exact
dense-matrix search, above it a
Kuzu-native HNSW index whose results report `exact: false`, and
`VECTOR_QUANTIZATION=int8` opts in to quantized storage with exact float
reranking of oversampled candidates. See
[configuration.md](configuration.md) for the settings.

If hosted embedding calls fail, treat the event log as the recovery source. Fix
configuration, replay the Eventloom log, and rebuild graph projections. Do not
manually patch vectors in a projection backend unless you are doing a controlled
maintenance operation documented in [operations.md](operations.md).

Related pages: [retrieval.md](retrieval.md), [graph-schema.md](graph-schema.md),
[deployment.md](deployment.md), and [README.md](../README.md). The public site
summary is [site/index.html](../site/index.html).
