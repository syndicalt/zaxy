"""Static Eventloom/session viewer generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zaxy.event import Event, EventLog
from zaxy.purpose_control import build_purpose_status

BOOTSTRAP_EVENTS = {
    "session.genesis",
    "session.profile.corrected",
    "workspace.instructions.discovered",
    "workspace.instructions.updated",
}
LIFECYCLE_EVENTS = {
    "tool.call.completed",
    "command.completed",
    "file.edit.applied",
    "compaction.completed",
    "subagent.cleaned",
    "subagent.completed",
    "session.ended",
}


def build_viewer_model(path: str | Path) -> dict[str, Any]:
    """Build a JSON-serializable viewer model from a log file or directory."""
    source = Path(path)
    log_paths = _eventlog_paths(source)
    events: list[dict[str, Any]] = []
    sessions: dict[str, dict[str, Any]] = {}
    integrity: dict[str, dict[str, Any]] = {}

    for log_path in log_paths:
        log = EventLog(log_path)
        replay = log.replay()
        integrity[str(log_path)] = replay.integrity.model_dump()
        for event in replay.events:
            session_id = _event_session_id(event)
            category = _event_category(event.type)
            entry = {
                "seq": event.seq,
                "timestamp": event.timestamp,
                "type": event.type,
                "actor": event.actor,
                "thread": event.thread,
                "session_id": session_id,
                "category": category,
                "summary": _event_summary(event),
                "payload": event.payload,
                "hash": event.hash,
                "log_path": str(log_path),
            }
            events.append(entry)
            session = sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "event_count": 0,
                    "bootstrap_count": 0,
                    "lifecycle_count": 0,
                    "latest_timestamp": None,
                },
            )
            session["event_count"] += 1
            if category == "bootstrap":
                session["bootstrap_count"] += 1
            if category == "lifecycle":
                session["lifecycle_count"] += 1
            session["latest_timestamp"] = event.timestamp

    events.sort(key=lambda item: (str(item["log_path"]), int(item["seq"])))
    return {
        "source_path": str(source),
        "log_paths": [str(log_path) for log_path in log_paths],
        "total_events": len(events),
        "sessions": sorted(sessions.values(), key=lambda item: str(item["session_id"])),
        "events": events,
        "integrity": integrity,
        "purpose": build_purpose_status(source),
    }


def render_viewer_html(model: dict[str, Any]) -> str:
    """Render a standalone HTML viewer for an Eventloom model."""
    data = _safe_json(model)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Eventloom Session Viewer</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657282;
      --line: #d9dee7;
      --accent: #1663c7;
      --bootstrap: #1d7f5f;
      --lifecycle: #9a5b00;
      --event: #596579;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 18px 24px;
    }}
    h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); margin-top: 4px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(220px, 300px) 1fr;
      gap: 18px;
      padding: 18px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    aside, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    aside {{ padding: 14px; align-self: start; }}
    section {{ overflow: hidden; }}
    h2 {{ margin: 0 0 10px; font-size: 15px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 14px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .stat strong {{ display: block; font-size: 18px; }}
    .session {{ border-top: 1px solid var(--line); padding: 10px 0; }}
    .session:first-of-type {{ border-top: 0; }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: white;
      font: inherit;
    }}
    input {{ width: 100%; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{ font-size: 12px; color: var(--muted); background: #fbfcfe; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .pill {{
      display: inline-block;
      min-width: 76px;
      border-radius: 999px;
      padding: 2px 8px;
      color: white;
      font-size: 12px;
      text-align: center;
    }}
    .bootstrap {{ background: var(--bootstrap); }}
    .lifecycle {{ background: var(--lifecycle); }}
    .event {{ background: var(--event); }}
    details summary {{ cursor: pointer; color: var(--accent); }}
    pre {{
      max-width: 720px;
      overflow: auto;
      margin: 8px 0 0;
      padding: 10px;
      background: #f3f5f8;
      border-radius: 6px;
      white-space: pre-wrap;
    }}
    @media (max-width: 860px) {{
      main {{ grid-template-columns: 1fr; padding: 12px; }}
      .toolbar {{ grid-template-columns: 1fr; }}
      th:nth-child(1), td:nth-child(1), th:nth-child(3), td:nth-child(3) {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Eventloom Session Viewer</h1>
    <div class="subtle" id="source"></div>
  </header>
  <main>
    <aside>
      <h2>Sessions</h2>
      <div class="stats">
        <div class="stat"><strong id="total-events">0</strong><span>Events</span></div>
        <div class="stat"><strong id="total-sessions">0</strong><span>Sessions</span></div>
        <div class="stat"><strong id="total-logs">0</strong><span>Logs</span></div>
      </div>
      <h2>Purpose</h2>
      <div id="purpose"></div>
      <div id="sessions"></div>
    </aside>
    <section>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Search events, actors, summaries, payloads">
        <select id="category">
          <option value="all">All events</option>
          <option value="bootstrap">Bootstrap</option>
          <option value="lifecycle">Lifecycle</option>
          <option value="event">Other</option>
        </select>
      </div>
      <table>
        <thead>
          <tr>
            <th>Seq</th>
            <th>Category</th>
            <th>Time</th>
            <th>Type</th>
            <th>Session</th>
            <th>Summary</th>
          </tr>
        </thead>
        <tbody id="events"></tbody>
      </table>
    </section>
  </main>
  <script>
    window.__ZAXY_VIEWER_DATA__ = {data};
    const data = window.__ZAXY_VIEWER_DATA__;
    const source = document.querySelector("#source");
    const totalEvents = document.querySelector("#total-events");
    const totalSessions = document.querySelector("#total-sessions");
    const totalLogs = document.querySelector("#total-logs");
    const sessions = document.querySelector("#sessions");
    const tbody = document.querySelector("#events");
    const search = document.querySelector("#search");
    const category = document.querySelector("#category");

    source.textContent = data.source_path;
    totalEvents.textContent = data.total_events;
    totalSessions.textContent = data.sessions.length;
    totalLogs.textContent = data.log_paths.length;
    renderPurpose(data.purpose || {{}});
    sessions.innerHTML = data.sessions.map((session) => `
      <div class="session">
        <strong>${{escapeHtml(session.session_id)}}</strong>
        <div class="subtle">${{session.event_count}} events · ${{session.bootstrap_count}} bootstrap · ${{session.lifecycle_count}} lifecycle</div>
        <div class="subtle">${{escapeHtml(session.latest_timestamp || "")}}</div>
      </div>
    `).join("");

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\\"": "&quot;",
        "'": "&#39;"
      }}[char]));
    }}

    function renderPurpose(purpose) {{
      const el = document.querySelector("#purpose");
      const suppression = purpose.suppression || {{}};
      const consequence = purpose.consequence_history || {{}};
      const coordinate = purpose.coordinate || {{}};
      const missions = coordinate.missions || [];
      const accepted = missions.reduce((total, mission) => total + (mission.accepted_count || 0), 0);
      const pending = missions.reduce((total, mission) => total + (mission.pending_count || 0), 0);
      const stale = missions.reduce((total, mission) => total + (mission.stale_count || 0), 0);
      const proofPackets = missions.reduce((total, mission) => total + (mission.proof_packet_count || 0), 0);
      el.innerHTML = `
        <div class="session">
          <strong>${{escapeHtml(purpose.active_profile || "none")}}</strong>
          <div class="subtle">evidence: ${{escapeHtml((purpose.evidence_policy_status || {{}}).status || "missing")}}</div>
          <div class="subtle">suppressed: ${{suppression.count || 0}}</div>
          <div class="subtle">feedback: +${{consequence.positive_count || 0}} / -${{consequence.negative_count || 0}}</div>
          <div class="subtle">Coordinate: accepted=${{accepted}} pending=${{pending}} stale=${{stale}} proof_packets=${{proofPackets}}</div>
        </div>
      `;
    }}

    function renderEvents() {{
      const q = search.value.trim().toLowerCase();
      const selected = category.value;
      const rows = data.events.filter((event) => {{
        if (selected !== "all" && event.category !== selected) return false;
        if (!q) return true;
        return JSON.stringify(event).toLowerCase().includes(q);
      }});
      tbody.innerHTML = rows.map((event) => `
        <tr>
          <td><code>${{event.seq}}</code></td>
          <td><span class="pill ${{event.category}}">${{event.category}}</span></td>
          <td>${{escapeHtml(event.timestamp)}}</td>
          <td><code>${{escapeHtml(event.type)}}</code><div class="subtle">${{escapeHtml(event.actor)}}</div></td>
          <td>${{escapeHtml(event.session_id)}}</td>
          <td>
            ${{escapeHtml(event.summary)}}
            <details>
              <summary>Payload</summary>
              <pre>${{escapeHtml(JSON.stringify(event.payload, null, 2))}}</pre>
            </details>
          </td>
        </tr>
      `).join("");
    }}
    search.addEventListener("input", renderEvents);
    category.addEventListener("change", renderEvents);
    renderEvents();
  </script>
</body>
</html>
"""


def write_viewer_html(path: str | Path, output: str | Path) -> Path:
    """Write a standalone viewer HTML file and return the output path."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_viewer_html(build_viewer_model(path)), encoding="utf-8")
    return output_path


def _eventlog_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(candidate for candidate in path.glob("*.jsonl") if candidate.is_file())
    return [path]


def _event_category(event_type: str) -> str:
    if event_type in BOOTSTRAP_EVENTS:
        return "bootstrap"
    if event_type in LIFECYCLE_EVENTS:
        return "lifecycle"
    return "event"


def _event_session_id(event: Event) -> str:
    value = event.payload.get("session_id") or event.payload.get("subagent_session_id") or event.thread
    return str(value)


def _event_summary(event: Event) -> str:
    payload = event.payload
    for key in (
        "summary",
        "workspace_type",
        "tool_name",
        "command",
        "reason",
        "title",
        "decision",
    ):
        value = payload.get(key)
        if value:
            return str(value)
    return event.type


def _safe_json(value: dict[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
