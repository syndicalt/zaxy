---
name: store-doctor
description: Diagnose and repair Zaxy memory-store incidents — checkout segfaults/hangs, MCP disconnects, LadybugDB corruption (orphan-WAL and bloat variants), orphaned serve processes, and thin/telemetry-only memory. Use when zaxy checkout crashes or hangs, the MCP server keeps disconnecting, or checkout returns no real facts.
---

# Store doctor — Zaxy memory incident response

**Prime directive:** `.eventloom/*.jsonl` is the hash-chained source of truth —
never edit or delete it. Everything under `.eventloom/projections/` is a
**rebuildable projection**: the fix is almost always *move aside + rebuild*, and
"move aside" is reversible, so it never needs permission-level caution.

**Second directive:** never open a suspect store in your main process. A corrupt
store can hang ~2 minutes and then **segfault whatever opened it**. All probes
below run in `timeout`-wrapped subprocesses.

## 1. Triage — gather without touching

```bash
cd <workspace>
grep -E '^EVENTLOOM_PATH|^PROJECTION_BACKEND' .env 2>/dev/null
ls -la .eventloom/ && ls -la .eventloom/projections/
pgrep -af "zaxy serve"
```

For each serve PID, establish ownership before judging it:
```bash
tr '\0' '\n' < /proc/<PID>/environ | grep EVENTLOOM_PATH
```
Only PIDs bound to THIS store are suspects. **Never kill serve processes belonging
to other workspaces.**

## 2. Read the signature

| Observation | Diagnosis |
|---|---|
| Multiple serve PIDs, same `EVENTLOOM_PATH` | Orphaned multi-owner (writers corrupt the WAL) |
| Tiny `embedded.kuzu` (KBs) + large `.kuzu.wal` (100KB+) | Orphan-WAL corruption; open throws `RuntimeError ... wal_record.cpp ... UNREACHABLE_CODE` |
| Huge `embedded.kuzu` (100s of MB) vs small JSONL logs + tiny clean WAL | **Bloat variant** — open hangs ~2min then segfaults; bypasses the self-heal (native crash, no Python exception) |
| Checkout "hangs" / MCP disconnects repeatedly | Either of the above; do steps 3-4 before theorizing |
| Checkout works but returns only `hook.stop`/reminder noise | Not corruption — capture gap; go to step 7 |

Sanity anchor: a healthy projection for ~500KB of JSONL is ~tens of KB.

## 3. Rule out the library (30s, subprocess)

```bash
timeout 30 python - <<'PY'
import tempfile, os, faulthandler; faulthandler.enable()
import ladybug
d = tempfile.mkdtemp()
db = ladybug.Database(os.path.join(d, "fresh.kuzu"))
conn = ladybug.Connection(db)
conn.execute("CREATE NODE TABLE T(id INT64, PRIMARY KEY(id))")
print("FRESH DB OK — ladybug native is healthy; the problem is the store file")
PY
```
Fresh DB fails too ⇒ broken native install (reinstall `zaxy-memory` /
`ladybug`), stop here. Fresh DB fine ⇒ the store is the patient.

## 4. Confirm the suspect store (isolated, capped)

```bash
timeout 60 python -X faulthandler -c "
import ladybug; ladybug.Database('.eventloom/projections/embedded.kuzu'); print('OPENED OK')
" ; echo "RC=$?"
```
RC 124/139 (timeout/segv) or a WAL RuntimeError confirms corruption.

## 5. Repair

```bash
# a. Kill VERIFIED this-store orphans only (checked in step 1):
kill <PID>...            # SIGTERM, then SIGKILL if needed

# b. Move the projection aside — never rm:
BK=.eventloom/projections/_corrupt-backup-$(date +%F)
mkdir -p "$BK"
mv .eventloom/projections/embedded.kuzu "$BK/" 2>/dev/null
mv .eventloom/projections/embedded.kuzu.wal "$BK/" 2>/dev/null

# c. Rebuild by touching the front door (auto-replays from the log):
timeout 120 zaxy checkout "repair probe" --session-id <SESSION>
```
Also available: `zaxy doctor --repair` (reaps verified broken owners) and
`zaxy reproject <log.jsonl> --session-id <SESSION>` for explicit rebuilds.

## 6. Verify

```bash
ls -la .eventloom/projections/embedded.kuzu*    # sane size again (KBs-MBs, not 100s of MB)
timeout 90 zaxy checkout "<a real query>" --session-id <SESSION>   # RC=0, cited facts
```
Tell the user: what the signature was, that the JSONL was untouched, where the
backup is parked (safe to delete later), and file/extend the incident memory if
the signature was new.

## 7. Thin memory (no corruption): capture health

Checkout returning only telemetry means the log lacks substantive events:

```bash
python - <<'PY'
import json, collections
c = collections.Counter()
for line in open(".eventloom/<SESSION>.jsonl"):
    line = line.strip()
    if line:
        try: c[json.loads(line).get("type","?")] += 1
        except Exception: pass
for t, n in c.most_common(12): print(f"{n:5}  {t}")
PY
```
Healthy capture shows `transcript.turn`, `tool.call.completed`,
`command.completed`, `file.edit.applied` — not just `hook.stop` +
`memory.reminder.suggested`. If the substantive lanes are missing:

```bash
# Backfill from the client's own session logs (idempotent, ~1s incremental):
zaxy capture claude --session-id <SESSION> --workspace .
# Ensure the Stop hook runs capture (v3.1.1+ hook configs include it):
grep -A4 '"Stop"' .claude/settings.local.json
```
Session note: the CLI defaults to `--session-id default`, which is usually empty —
the populated session in this repo is `zaxy-default`.
