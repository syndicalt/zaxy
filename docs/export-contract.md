# Memory Export Contract

The export contract is the product-agnostic way to pull a session's Zaxy memory
out as a portable, cited, optionally-signed bundle that **any** consumer can
verify and reuse. Zaxy never needs to know who is asking: a consumer expresses
*what* it wants with a selector and receives entries that carry sealed Eventloom
provenance.

This page is the authoritative spec. It documents the entry schema, the selector,
the three bundle shapes, the two pull surfaces, and the versioning policy. The
load-bearing identifiers here are pinned against the code by a drift-guard test,
so this document cannot silently fall out of sync with what Zaxy emits.

> **Signing is EXPERIMENTAL and UNAUDITED.** The signed-bundle and partial-
> disclosure paths build on `zaxy.portable`, which is opt-in (install
> `zaxy-memory[export]`) and pending an independent cryptographic review before
> any general-availability use. Treat signed export as a preview. The *unsigned*
> canonical bundle has no such caveat and is usable today.

## Entry schema (`zaxy.export.v1`)

Every export entry is a self-describing, canonical JSON object. It carries its
own `schema_version` so a single entry stays meaningful when disclosed in
isolation. Fields:

| field | type | meaning |
| --- | --- | --- |
| `schema_version` | string | Always `zaxy.export.v1`. Pin this. |
| `grain` | string | `event` or `semantic`. |
| `kind` | string | For `event` grain: the event type (e.g. `decision.made`). For `semantic` grain: `entity:<entity_type>` or `edge:<relation_type>`. |
| `citation` | string | Sealed Eventloom provenance ref: `eventloom://<thread>/events/<seq>#<hash>` (full-hash form). |
| `seq` | integer | Source event sequence number. |
| `valid_from` | string \| null | ISO-8601 validity start (event timestamp / entity `observed_at` / edge `valid_from`). |
| `valid_to` | string \| null | ISO-8601 validity end (edge `valid_to`; otherwise `null`). |
| `source` | string | Lane label: `eventloom` for event grain, `extraction` for semantic grain. |
| `content` | object | The grain-specific payload (below). |

**Event grain** (`grain: "event"`, `source: "eventloom"`) — the sealed event:

```json
{
  "schema_version": "zaxy.export.v1",
  "grain": "event",
  "kind": "decision.made",
  "citation": "eventloom://agent-1/events/42#<64-hex-hash>",
  "seq": 42,
  "valid_from": "2026-06-15T00:00:00Z",
  "valid_to": null,
  "source": "eventloom",
  "content": {"type": "decision.made", "actor": "assistant", "thread": "agent-1", "payload": {"decision": "ship it"}}
}
```

**Semantic grain** (`grain: "semantic"`, `source: "extraction"`) — the
deterministic extraction that also feeds the graph. Entities:

```json
{
  "grain": "semantic",
  "kind": "entity:goal",
  "source": "extraction",
  "content": {"name": "...", "entity_type": "goal", "summary": "...", "properties": {}}
}
```

Edges use `kind: "edge:<relation_type>"` and a `content` of `{source, target,
relation_type, confidence, inferred, inference_method, evidence}`.

Entries are deterministically ordered (ascending `seq`; within an event the event
entry precedes its semantic entries) and byte-stable, so Merkle roots and
signatures over them are reproducible.

## Selector

The selector describes which memory to export. Every axis is optional; the
default selects both grains of the whole session.

| axis | applies to | meaning |
| --- | --- | --- |
| `grains` | both | Subset of `{event, semantic}` (default both). |
| `kinds` | projection | Restrict to these **event types** (gates both grains). |
| `since_seq` | both | Exclusive delta cursor: `seq` strictly greater. |
| `max_seq` | both | Inclusive upper bound on `seq`. |
| `since_time` / `until_time` | both | Inclusive ISO-8601 time window over `valid_from`. |
| `query` | projection-only | Lexical pre-filter via the verbatim index. |
| `query_limit` | projection-only | Top-N for the `query` pre-filter (default 50). |
| `exclude_sensitivities` | projection-only | Drop entries whose source event sensitivity tier is listed. |
| `limit` | projection | Cap to the most recent N matching events. |

### Projection-time vs entry-level

There are two contexts in which a selector is applied, and they support
**different** axes:

- **Projection** (building entries from the log) supports every axis. `query`
  and `exclude_sensitivities` are decided from the log/event and are therefore
  *projection-only*.
- **Partial disclosure** (selecting which already-built entries to reveal from a
  signed bundle) is decided from a static entry alone. It supports `grains`,
  `kinds` (matched against the entry's `kind`), and the `seq`/time bounds.
  `query`, `exclude_sensitivities`, and `limit` have no meaning over a static
  entry and are **ignored** during disclosure.

For event-grain entries `kinds` behaves identically in both contexts (the entry's
`kind` *is* the event type). For semantic entries, disclosure-time `kinds` matches
the `entity:`/`edge:` kind.

## Bundle shapes

### 1. Unsigned canonical bundle (`zaxy.export.unsigned.v1`)

Returned when no signing key is supplied. No crypto dependency required.

```json
{
  "version": "zaxy.export.unsigned.v1",
  "schema_version": "zaxy.export.v1",
  "session_id": "agent-1",
  "signed": false,
  "entries": [ /* canonical entries, in order */ ]
}
```

### 2. Signed bundle (`zaxy.portable.v1`)

Returned when a signing key is supplied. The signature binds the Merkle root and
all metadata. Verify with `verify_export(bundle, expect_public_key=...)`.

```json
{
  "version": "zaxy.portable.v1",
  "algorithm": "ml-dsa-65",
  "public_key": "<hex>",
  "session_id": "agent-1",
  "created_at": "2026-06-15T00:00:00+00:00",
  "nonce": "<hex>",
  "merkle_root": "<hex>",
  "anchor": null,
  "entries": [ {"id": "<sha256>", "content": { /* canonical entry */ }} ],
  "signature": "<hex>"
}
```

### 3. Disclosed subset

A verifiable partial disclosure of a signed bundle: only the chosen entries plus
Merkle inclusion proofs, so a recipient can prove each disclosed entry is an
authentic member of the signed set **without seeing the undisclosed ones**.
Verify with `verify_memory_export_subset(subset, expect_public_key=...)`.

```json
{
  "version": "zaxy.portable.v1",
  "algorithm": "ml-dsa-65",
  "public_key": "<hex>",
  "session_id": "agent-1",
  "created_at": "...", "nonce": "...", "merkle_root": "<hex>", "signature": "<hex>",
  "disclosed": [ {"index": 0, "content": { /* canonical entry */ }, "proof": [["<hex>", "left"]]} ]
}
```

Partial disclosure requires a **signed** bundle (it proves membership against the
signed Merkle root); an unsigned envelope cannot be disclosed.

## Pull surfaces

Both surfaces converge on a single projection + signing path, so they always
agree on entry contents.

### MCP tool: `memory_export`

Arguments mirror the selector: `session_id`, `grains`, `kinds`, `since_seq`,
`max_seq`, `since_time`, `until_time`, `query`, `query_limit`,
`exclude_sensitivities`, `limit`. Plus:

- `sign` (boolean): sign with the **server-configured** export key. A private key
  is never accepted as a tool argument.
- `disclose` (object): an entry-level sub-selector (`grains`, `kinds`,
  `since_seq`, `max_seq`, `since_time`, `until_time`). When present, the tool
  signs the full bundle over the main selector and returns the verifiable subset
  for entries matching `disclose` (requires the server key).
- `admin_token`: required when admin gating is configured.

The tool is admin-gated and session-scoped, and runs the projection off the
event loop. Server-side signing is configured via the
`mcp_export_signing_private_key_file`, `mcp_export_signing_public_key_file`, and
`mcp_export_signing_algorithm` settings.

### CLI

```sh
# Generate a self-sovereign signing keypair (optional; for signed bundles).
zaxy export-keygen --out-private k.pem --out-public k.pub --algorithm ed25519

# Export a bundle. Omit the keys for an unsigned canonical bundle.
zaxy export --out bundle.json --eventloom-path .eventloom --session-id agent-1 \
    --grains event,semantic --since 100 --types decision.made,goal.created \
    --private-key k.pem --public-key k.pub --algorithm ed25519

# Verify a signed bundle (pin the public key to establish trust).
zaxy verify-export bundle.json --expect-public-key <hex>

# Disclose a verifiable subset of a signed bundle, selected by predicate.
zaxy export-disclose bundle.json --grains event --kinds goal.created --out subset.json

# Verify a disclosed subset.
zaxy verify-export-subset subset.json --expect-public-key <hex>
```

The CLI `export` accepts the full selector: `--grains`, `--types` (event-type
`kinds`), `--since` (`since_seq`), `--max-seq`, `--since-time`, `--until-time`,
`--query`, `--exclude-sensitivities`, `--limit`.

## Versioning and compatibility

Three independent version strings let a consumer pin exactly what it depends on:

- **`schema_version`** (`zaxy.export.v1`) — the entry shape. Present on every
  entry and on the unsigned envelope.
- **The unsigned envelope `version`** (`zaxy.export.unsigned.v1`).
- **The signed bundle/subset `version`** (`zaxy.portable.v1`) — the portable
  signing + Merkle format.

Compatibility policy:

- **Additive change** (new optional entry field, new selector axis, new event
  type surfaced) does **not** bump a version. Consumers must ignore unknown
  fields.
- **Breaking change** (removing/renaming a field, changing a field's meaning, or
  changing canonicalization) bumps the relevant version string. A consumer should
  pin the version(s) it understands and reject others.

Verifiers establish trust in a signer out-of-band by pinning the public key
(`expect_public_key`); there is no certificate authority.

See also: the [MCP interface](mcp.md), the [Eventloom contract](eventloom.md),
and `docs/experimental/portable-export-security.md` for the signing security
model.
